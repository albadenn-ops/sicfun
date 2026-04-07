package sicfun.holdem.tablegen
import sicfun.holdem.cli.*
import sicfun.holdem.*
import sicfun.holdem.equity.*
import sicfun.holdem.gpu.*

import scala.util.Random

/** Tuning harness for native CUDA board-major exact canonical generation.
  *
  * Runs a grid of `(chunkBoards, scoreThreads, matchThreads, prepareBoardsPerBlock)` candidates in a
  * single JVM process to avoid JNI native reload overhead between runs.
  *
  * Usage:
  * {{{
  * HeadsUpCanonicalExactBoardMajorTuner [--key=value ...]
  * }}}
  *
  * Options:
  *   - `--maxMatchups=<long>` (default: 5000)
  *   - `--seed=<long>` (default: 1)
  *   - `--warmup=<true|false>` (default: true)
  *   - `--chunkBoards=16384,32768,49152`
  *   - `--scoreThreads=64,96,128`
  *   - `--matchThreads=64,96,128,160`
  *   - `--prepareBoardsPerBlock=1,2,4`
  */
object HeadsUpCanonicalExactBoardMajorTuner:
  private val ProviderProperty = "sicfun.gpu.provider"
  private val FallbackToCpuProperty = "sicfun.gpu.fallbackToCpu"
  private val NativeEngineProperty = "sicfun.gpu.native.engine"
  private val PackedExactIoProperty = "sicfun.gpu.native.packedExactIo"
  private val BoardMajorProperty = "sicfun.gpu.native.exact.boardMajor"
  private val ChunkBoardsProperty = "sicfun.gpu.native.exact.boardMajor.chunkBoards"
  private val ScoreThreadsProperty = "sicfun.gpu.native.exact.boardMajor.scoreThreads"
  private val MatchThreadsProperty = "sicfun.gpu.native.exact.boardMajor.matchThreads"
  private val PrepareBoardsPerBlockProperty = "sicfun.gpu.native.exact.boardMajor.prepareBoardsPerBlock"
  private val AllowedOptionKeys =
    Set("maxMatchups", "seed", "warmup", "chunkBoards", "scoreThreads", "matchThreads", "prepareBoardsPerBlock")

  /** A single CUDA launch parameter combination to benchmark.
    *
    * @param chunkBoards           number of boards per GPU chunk dispatch
    * @param scoreThreads          CUDA threads per block for the scoring kernel
    * @param matchThreads          CUDA threads per block for the matchup evaluation kernel
    * @param prepareBoardsPerBlock boards prepared per block in the board preparation kernel
    */
  private final case class Candidate(
      chunkBoards: Int,
      scoreThreads: Int,
      matchThreads: Int,
      prepareBoardsPerBlock: Int
  ):
    val name: String =
      s"chunk=$chunkBoards,score=$scoreThreads,match=$matchThreads,prepareBoardsPerBlock=$prepareBoardsPerBlock"

  private final case class Config(
      maxMatchups: Long = 5000L,
      seed: Long = 1L,
      warmup: Boolean = true,
      chunkBoards: Vector[Int] = Vector(16384, 24576, 32768, 49152, 65536),
      scoreThreads: Vector[Int] = Vector(64, 96, 128),
      matchThreads: Vector[Int] = Vector(64, 96, 128, 160),
      prepareBoardsPerBlock: Vector[Int] = Vector(1)
  ):
    def candidates: Vector[Candidate] =
      for
        chunk <- chunkBoards
        score <- scoreThreads
        matching <- matchThreads
        prepare <- prepareBoardsPerBlock
      yield Candidate(chunk, score, matching, prepare)

  /** Entry point. Forces native CUDA provider with board-major exact mode enabled,
    * runs an optional warmup, then benchmarks every candidate from the parameter grid.
    * Reports the fastest candidate at the end.
    */
  def main(args: Array[String]): Unit =
    val config = parseArgs(args.toVector)
    require(config.maxMatchups > 0L, "maxMatchups must be positive")
    require(config.chunkBoards.nonEmpty, "chunkBoards must be non-empty")
    require(config.scoreThreads.nonEmpty, "scoreThreads must be non-empty")
    require(config.matchThreads.nonEmpty, "matchThreads must be non-empty")
    require(config.prepareBoardsPerBlock.nonEmpty, "prepareBoardsPerBlock must be non-empty")

    sys.props.update(ProviderProperty, "native")
    sys.props.update(FallbackToCpuProperty, "false")
    sys.props.update(NativeEngineProperty, "cuda")
    sys.props.update(PackedExactIoProperty, "true")
    sys.props.update(BoardMajorProperty, "true")

    val availability = HeadsUpGpuRuntime.availability
    println("canonical exact board-major tuner")
    println(
      s"maxMatchups=${config.maxMatchups}, seed=${config.seed}, warmup=${config.warmup}, " +
        s"provider=${availability.provider}, available=${availability.available}"
    )
    println(s"providerDetail=${availability.detail}")

    if !availability.available || availability.provider != "native" then
      throw new IllegalStateException("native provider unavailable")

    if config.warmup then
      runCandidate(config.maxMatchups, config.seed, Candidate(32768, 64, 96, 1), "warmup")

    val results = config.candidates.map { candidate =>
      runCandidate(config.maxMatchups, config.seed, candidate, "candidate")
    }
    val successful = results.collect { case (name, Some(elapsed)) => name -> elapsed }
    if successful.isEmpty then
      throw new IllegalStateException("no successful candidates")
    val best = successful.minBy(_._2)
    println(f"best=${best._1} elapsed=${best._2}%.3fs")

  /** Applies the candidate's CUDA parameters via system properties and runs a full
    * canonical table build, measuring elapsed wall time. Returns the candidate name
    * paired with elapsed seconds (None on failure).
    */
  private def runCandidate(
      maxMatchups: Long,
      seed: Long,
      candidate: Candidate,
      tag: String
  ): (String, Option[Double]) =
    // Set CUDA launch parameters via system properties — the native runtime reads these
    // to configure kernel launches for this specific run.
    sys.props.update(ChunkBoardsProperty, candidate.chunkBoards.toString)
    sys.props.update(ScoreThreadsProperty, candidate.scoreThreads.toString)
    sys.props.update(MatchThreadsProperty, candidate.matchThreads.toString)
    sys.props.update(PrepareBoardsPerBlockProperty, candidate.prepareBoardsPerBlock.toString)

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
        f"$tag%-9s ${candidate.name}%-36s size=${table.size}%6d elapsed=${elapsed}%.3fs telemetry=$telemetry"
      )
      (candidate.name, Some(elapsed))
    catch
      case ex: Throwable =>
        val telemetry = HeadsUpGpuRuntime.lastBatchTelemetry
          .map(t => s"provider=${t.provider}, success=${t.success}, detail=${t.detail}")
          .getOrElse("none")
        val message = Option(ex.getMessage).getOrElse(ex.getClass.getSimpleName)
        println(
          s"$tag ${candidate.name} failed error=$message telemetry=$telemetry"
        )
        (candidate.name, None)

  private def parseArgs(args: Vector[String]): Config =
    val options = CliHelpers.requireOptions(args)
    CliHelpers.requireNoUnknownOptions(options, AllowedOptionKeys)
    Config(
      maxMatchups = CliHelpers.requireLongOption(options, "maxMatchups", 5000L),
      seed = CliHelpers.requireLongOption(options, "seed", 1L),
      warmup = CliHelpers.requireBooleanOption(options, "warmup", true),
      chunkBoards = CliHelpers.optionalPositiveIntList(options, "chunkBoards").map(_.distinct).getOrElse(Vector(16384, 24576, 32768, 49152, 65536)),
      scoreThreads = CliHelpers.optionalPositiveIntList(options, "scoreThreads").map(_.distinct).getOrElse(Vector(64, 96, 128)),
      matchThreads = CliHelpers.optionalPositiveIntList(options, "matchThreads").map(_.distinct).getOrElse(Vector(64, 96, 128, 160)),
      prepareBoardsPerBlock = CliHelpers.optionalPositiveIntList(options, "prepareBoardsPerBlock").map(_.distinct).getOrElse(Vector(1))
    )
