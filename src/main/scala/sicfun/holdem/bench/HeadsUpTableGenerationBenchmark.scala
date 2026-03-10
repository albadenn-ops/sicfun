package sicfun.holdem.bench
import sicfun.holdem.*
import sicfun.holdem.equity.*
import sicfun.holdem.cli.*

import sicfun.core.HandEvaluator

import scala.util.Random

/** Micro-benchmark for heads-up equity table generation throughput.
  *
  * Measures matchups/second at different parallelism levels (worker thread counts)
  * to identify scaling behavior. Supports both full and canonical table types.
  * Reports throughput and speedup relative to the single-threaded baseline.
  *
  * '''Usage:'''
  * {{{
  * HeadsUpTableGenerationBenchmark [--key=value ...]
  * }}}
  *
  * '''Options:'''
  *   - `--table=full|canonical` (default: full)
  *   - `--mode=exact|mc` (default: mc)
  *   - `--trials=<int>` (default: 50)
  *   - `--maxMatchups=<long>` (default: 2000)
  *   - `--parallelism=<comma-sep ints>` (default: 1,2,4,8)
  *   - `--warmupRuns=<int>` / `--runs=<int>` / `--seed=<long>`
  *   - `--clearCachesPerRun=<boolean>` (default: true)
  */
object HeadsUpTableGenerationBenchmark:
  /** Benchmark configuration parsed from CLI arguments. */
  private final case class Config(
      table: String = "full",
      mode: String = "mc",
      trials: Int = 50,
      maxMatchups: Long = 2000L,
      parallelism: Vector[Int] = Vector(1, 2, 4, 8),
      warmupRuns: Int = 1,
      runs: Int = 2,
      seed: Long = 1L,
      clearCachesPerRun: Boolean = true
  )

  private val AllowedOptionKeys = Set("table", "mode", "trials", "maxMatchups", "parallelism", "warmupRuns", "runs", "seed", "clearCachesPerRun")

  private final case class BenchmarkResult(
      workers: Int,
      entries: Int,
      averageSeconds: Double,
      throughput: Double
  )

  def main(args: Array[String]): Unit =
    val config = parseArgs(args.toVector)
    val mode = parseMode(config.mode, config.trials)
    val backend = HeadsUpEquityTable.ComputeBackend.Cpu
    val cores = Runtime.getRuntime.availableProcessors()

    println("heads-up table generation benchmark")
    println(
      s"table=${config.table}, mode=${config.mode}, trials=${config.trials}, maxMatchups=${config.maxMatchups}, " +
        s"parallelism=${config.parallelism.mkString(",")}, warmupRuns=${config.warmupRuns}, runs=${config.runs}, " +
        s"clearCachesPerRun=${config.clearCachesPerRun}, cpuCores=$cores"
    )

    val results = config.parallelism.distinct.sorted.map { workers =>
      var warmup = 0
      while warmup < config.warmupRuns do
        runOnce(
          table = config.table,
          mode = mode,
          maxMatchups = config.maxMatchups,
          workers = workers,
          backend = backend,
          seed = config.seed + warmup,
          clearCaches = config.clearCachesPerRun
        )
        warmup += 1

      val measurements = Vector.tabulate(config.runs) { runIdx =>
        runOnce(
          table = config.table,
          mode = mode,
          maxMatchups = config.maxMatchups,
          workers = workers,
          backend = backend,
          seed = config.seed + 1000L + runIdx.toLong,
          clearCaches = config.clearCachesPerRun
        )
      }
      val entries = measurements.headOption.map(_._1).getOrElse(0)
      measurements.foreach { case (count, _) =>
        require(count == entries, s"inconsistent table size: expected $entries, got $count")
      }
      val durations = measurements.map(_._2)
      val averageSeconds = durations.sum / durations.length.toDouble
      val throughput = entries.toDouble / averageSeconds
      BenchmarkResult(
        workers = workers,
        entries = entries,
        averageSeconds = averageSeconds,
        throughput = throughput
      )
    }

    val baseline = results.find(_.workers == 1).getOrElse(results.head)
    println("results:")
    results.foreach { result =>
      val speedup = result.throughput / baseline.throughput
      println(
        f"workers=${result.workers}%2d entries=${result.entries}%6d avg=${result.averageSeconds}%.3fs " +
          f"throughput=${result.throughput}%.1f matchups/s speedup=${speedup}%.2fx"
      )
    }

  private def runOnce(
      table: String,
      mode: HeadsUpEquityTable.Mode,
      maxMatchups: Long,
      workers: Int,
      backend: HeadsUpEquityTable.ComputeBackend,
      seed: Long,
      clearCaches: Boolean
  ): (Int, Double) =
    if clearCaches then HandEvaluator.clearCaches()
    val started = System.nanoTime()
    val size =
      table match
        case "full" =>
          HeadsUpEquityTable.buildAll(
            mode = mode,
            rng = new Random(seed),
            maxMatchups = maxMatchups,
            progress = None,
            parallelism = workers,
            backend = backend
          ).size
        case "canonical" =>
          HeadsUpEquityCanonicalTable.buildAll(
            mode = mode,
            rng = new Random(seed),
            maxMatchups = maxMatchups,
            progress = None,
            parallelism = workers,
            backend = backend
          ).size
        case other =>
          throw new IllegalArgumentException(s"unknown table '$other' (expected full or canonical)")
    val elapsedSeconds = (System.nanoTime() - started).toDouble / 1_000_000_000.0
    (size, elapsedSeconds)

  private def parseMode(mode: String, trials: Int): HeadsUpEquityTable.Mode =
    mode.trim.toLowerCase match
      case "exact" => HeadsUpEquityTable.Mode.Exact
      case "mc" | "montecarlo" => HeadsUpEquityTable.Mode.MonteCarlo(trials)
      case other => throw new IllegalArgumentException(s"unknown mode '$other' (expected exact or mc)")

  private def parseArgs(args: Vector[String]): Config =
    val options = CliHelpers.requireOptions(args)
    CliHelpers.requireNoUnknownOptions(options, AllowedOptionKeys)
    Config(
      table = options.getOrElse("table", "full").trim.toLowerCase,
      mode = options.getOrElse("mode", "mc").trim.toLowerCase,
      trials = CliHelpers.requireIntOption(options, "trials", 50),
      maxMatchups = CliHelpers.requireLongOption(options, "maxMatchups", 2000L),
      parallelism = CliHelpers.optionalPositiveIntList(options, "parallelism").getOrElse(Vector(1, 2, 4, 8)),
      warmupRuns = CliHelpers.requireIntOption(options, "warmupRuns", 1),
      runs = CliHelpers.requireIntOption(options, "runs", 2),
      seed = CliHelpers.requireLongOption(options, "seed", 1L),
      clearCachesPerRun = CliHelpers.requireBooleanOption(options, "clearCachesPerRun", true)
    )
