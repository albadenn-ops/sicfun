package sicfun.holdem.tablegen
import sicfun.holdem.cli.*
import sicfun.holdem.*
import sicfun.holdem.equity.*
import sicfun.holdem.gpu.*

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
  private val AllowedOptionKeys = Set("maxMatchups", "seed", "candidates", "warmup")

  /** A single (blockSize, maxChunkMatchups) candidate to benchmark.
    *
    * @param name               human-readable label (e.g. "128x512" or "default")
    * @param blockSize          CUDA threads per block (None = use runtime default)
    * @param maxChunkMatchups   max matchups per GPU dispatch chunk (None = use runtime default)
    */
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

  /** Entry point. Forces native CUDA provider, runs an optional warmup candidate,
    * then benchmarks all configured (blockSize x maxChunkMatchups) combinations.
    * The fastest candidate is reported at the end.
    */
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

  /** Applies a candidate's CUDA parameters (or clears them to use defaults) and runs
    * a canonical table build, returning elapsed seconds (None on failure).
    */
  private def runCandidate(
      maxMatchups: Long,
      seed: Long,
      candidate: Candidate
  ): (String, Option[Double]) =
    // Set or clear the system properties — the native runtime reads these on each batch call.
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
    val options = CliHelpers.requireOptions(args)
    CliHelpers.requireNoUnknownOptions(options, AllowedOptionKeys)
    Config(
      maxMatchups = CliHelpers.requireLongOption(options, "maxMatchups", 5000L),
      seed = CliHelpers.requireLongOption(options, "seed", 1L),
      candidates = options.get("candidates").map(parseCandidates).getOrElse(defaultCandidates),
      warmup = CliHelpers.requireBooleanOption(options, "warmup", true)
    )

  /** Parses a comma-separated candidate list. Each token is either "default" (use runtime
    * defaults) or "NxM" where N=blockSize and M=maxChunkMatchups (e.g. "128x512").
    */
  private def parseCandidates(raw: String): Vector[Candidate] =
    CliHelpers.requireCsvTokens(raw, "candidates").map {
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
