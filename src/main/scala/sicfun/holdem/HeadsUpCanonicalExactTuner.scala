package sicfun.holdem

import scala.util.Random

/** CUDA tuning harness for exact canonical table generation.
  *
  * Runs a sequence of `(blockSize, maxChunkMatchups)` candidates in a single JVM process
  * and reports elapsed time plus native telemetry for each candidate.
  *
  * Usage:
  * {{{
  * HeadsUpCanonicalExactTuner [--key=value ...]
  * }}}
  *
  * Options:
  *   - `--maxMatchups=<long>` (default: 5000)
  *   - `--seed=<long>` (default: 1)
  *   - `--candidates=default,64x128,96x256,128x512` (default built-in set)
  *   - `--warmup=<true|false>` (default: true)
  */
object HeadsUpCanonicalExactTuner:
  private val ProviderProperty = "sicfun.gpu.provider"
  private val FallbackToCpuProperty = "sicfun.gpu.fallbackToCpu"
  private val NativeEngineProperty = "sicfun.gpu.native.engine"
  private val NativeCudaBlockSizeProperty = "sicfun.gpu.native.cuda.blockSize"
  private val NativeCudaMaxChunkMatchupsProperty = "sicfun.gpu.native.cuda.maxChunkMatchups"

  private final case class Candidate(
      name: String,
      blockSize: Option[Int],
      maxChunkMatchups: Option[Int]
  )

  private final case class Config(
      maxMatchups: Long = 5000L,
      seed: Long = 1L,
      candidates: Vector[Candidate] = defaultCandidates,
      warmup: Boolean = true
  )

  private val defaultCandidates: Vector[Candidate] = Vector(
    Candidate("default", None, None),
    Candidate("64x128", Some(64), Some(128)),
    Candidate("64x256", Some(64), Some(256)),
    Candidate("96x256", Some(96), Some(256)),
    Candidate("96x512", Some(96), Some(512)),
    Candidate("128x256", Some(128), Some(256)),
    Candidate("128x512", Some(128), Some(512)),
    Candidate("128x1024", Some(128), Some(1024)),
    Candidate("256x512", Some(256), Some(512)),
    Candidate("256x1024", Some(256), Some(1024))
  )

  def main(args: Array[String]): Unit =
    val config = parseArgs(args.toVector)
    require(config.maxMatchups > 0L, "maxMatchups must be positive")

    sys.props.update(ProviderProperty, "native")
    sys.props.update(FallbackToCpuProperty, "false")
    sys.props.update(NativeEngineProperty, "cuda")

    val availability = HeadsUpGpuRuntime.availability
    println("canonical exact cuda tuner")
    println(
      s"maxMatchups=${config.maxMatchups}, seed=${config.seed}, warmup=${config.warmup}, " +
        s"provider=${availability.provider}, available=${availability.available}"
    )
    println(s"providerDetail=${availability.detail}")

    if !availability.available || availability.provider != "native" then
      throw new IllegalStateException("native provider unavailable")

    if config.warmup then
      runCandidate(config.maxMatchups, config.seed, Candidate("warmup-default", None, None))

    val results = config.candidates.map(candidate => runCandidate(config.maxMatchups, config.seed, candidate))
    val successful = results.collect { case (name, Some(elapsed)) => name -> elapsed }
    if successful.isEmpty then
      throw new IllegalStateException("no successful candidates")
    val best = successful.minBy(_._2)
    println(f"best=${best._1} elapsed=${best._2}%.3fs")

  private def runCandidate(
      maxMatchups: Long,
      seed: Long,
      candidate: Candidate
  ): (String, Option[Double]) =
    candidate.blockSize match
      case Some(value) => sys.props.update(NativeCudaBlockSizeProperty, value.toString)
      case None => sys.props.remove(NativeCudaBlockSizeProperty)
    candidate.maxChunkMatchups match
      case Some(value) => sys.props.update(NativeCudaMaxChunkMatchupsProperty, value.toString)
      case None => sys.props.remove(NativeCudaMaxChunkMatchupsProperty)

    try
      val started = System.nanoTime()
      val table = HeadsUpEquityCanonicalTable.buildAll(
        mode = HeadsUpEquityTable.Mode.Exact,
        rng = new Random(seed),
        maxMatchups = maxMatchups,
        progress = None,
        backend = HeadsUpEquityTable.ComputeBackend.Gpu
      )
      val elapsed = (System.nanoTime() - started).toDouble / 1_000_000_000.0
      val telemetry = HeadsUpGpuRuntime.lastBatchTelemetry
        .map(t => s"provider=${t.provider}, success=${t.success}, detail=${t.detail}")
        .getOrElse("none")
      println(
        f"candidate=${candidate.name}%-12s size=${table.size}%6d elapsed=${elapsed}%.3fs telemetry=$telemetry"
      )
      (candidate.name, Some(elapsed))
    catch
      case ex: Throwable =>
        val telemetry = HeadsUpGpuRuntime.lastBatchTelemetry
          .map(t => s"provider=${t.provider}, success=${t.success}, detail=${t.detail}")
          .getOrElse("none")
        val message = Option(ex.getMessage).getOrElse(ex.getClass.getSimpleName)
        println(
          s"candidate=${candidate.name} failed error=$message telemetry=$telemetry"
        )
        (candidate.name, None)

  private def parseArgs(args: Vector[String]): Config =
    args.foldLeft(Config()) { (cfg, raw) =>
      val trimmed = raw.trim
      if !trimmed.startsWith("--") || !trimmed.contains("=") then
        throw new IllegalArgumentException(s"invalid argument '$raw' (expected --key=value)")
      val eq = trimmed.indexOf('=')
      val key = trimmed.substring(2, eq).trim
      val value = trimmed.substring(eq + 1).trim
      key match
        case "maxMatchups" => cfg.copy(maxMatchups = value.toLong)
        case "seed" => cfg.copy(seed = value.toLong)
        case "candidates" => cfg.copy(candidates = parseCandidates(value))
        case "warmup" => cfg.copy(warmup = parseBoolean(value, "warmup"))
        case other => throw new IllegalArgumentException(s"unknown option '$other'")
    }

  private def parseCandidates(raw: String): Vector[Candidate] =
    val tokens = raw.split(",").toVector.map(_.trim).filter(_.nonEmpty)
    require(tokens.nonEmpty, "candidates must be non-empty")
    tokens.map {
      case value if value.equalsIgnoreCase("default") =>
        Candidate("default", None, None)
      case value =>
        val parts = value.toLowerCase.split("x")
        if parts.length != 2 then
          throw new IllegalArgumentException(
            s"invalid candidate '$value' (expected default or <block>x<chunk>)"
          )
        val block = parts(0).toInt
        val chunk = parts(1).toInt
        require(block > 0 && chunk > 0, s"candidate values must be positive: $value")
        Candidate(value, Some(block), Some(chunk))
    }

  private def parseBoolean(raw: String, key: String): Boolean =
    raw.trim.toLowerCase match
      case "1" | "true" | "yes" | "on" => true
      case "0" | "false" | "no" | "off" => false
      case other =>
        throw new IllegalArgumentException(s"invalid boolean for $key: '$other' (expected true/false)")
