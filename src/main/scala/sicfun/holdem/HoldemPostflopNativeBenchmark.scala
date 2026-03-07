package sicfun.holdem

import sicfun.core.{Card, DiscreteDistribution}

import scala.util.Random

/** Benchmark harness for postflop Monte Carlo runtime backends. */
object HoldemPostflopNativeBenchmark:
  private final case class Config(
      warmupRuns: Int = 2,
      measureRuns: Int = 8,
      trials: Int = 12_000,
      seed: Long = 23L,
      modes: Vector[String] = Vector("scala", "native-cpu", "native-cuda"),
      nativeCpuPath: Option[String] = None,
      nativeGpuPath: Option[String] = None
  )

  private final case class BenchmarkSpot(
      hero: HoleCards,
      board: Board,
      villains: Array[HoleCards],
      villainWeights: Array[Double],
      range: DiscreteDistribution[HoleCards]
  )

  private final case class Stats(
      count: Int,
      meanMs: Double,
      medianMs: Double,
      minMs: Double,
      maxMs: Double
  )

  private final case class ModeRun(
      mode: String,
      stats: Option[Stats],
      detail: String
  )

  def main(args: Array[String]): Unit =
    val config = parseArgs(args.toVector)
    require(config.warmupRuns >= 0, "warmupRuns must be non-negative")
    require(config.measureRuns > 0, "measureRuns must be positive")
    require(config.trials > 0, "trials must be positive")
    require(config.modes.nonEmpty, "at least one benchmark mode is required")

    val runSeeds = {
      val rng = new Random(config.seed)
      Vector.fill(config.warmupRuns + config.measureRuns)(rng.nextLong())
    }
    val spot = benchmarkSpot(config.seed ^ 0xD6E8FEB86659FD93L)

    println("=== Holdem Postflop Native Benchmark ===")
    println(
      s"config: warmupRuns=${config.warmupRuns}, measureRuns=${config.measureRuns}, " +
        s"trials=${config.trials}, seed=${config.seed}, modes=${config.modes.mkString(",")}"
    )

    val modeRuns = config.modes.map { mode =>
      runMode(mode, config, spot, runSeeds)
    }

    println()
    modeRuns.foreach { run =>
      run.stats match
        case Some(stats) =>
          println(
            f"${run.mode}%-11s count=${stats.count}%2d mean=${stats.meanMs}%.3fms " +
              f"median=${stats.medianMs}%.3fms min=${stats.minMs}%.3fms max=${stats.maxMs}%.3fms ${run.detail}"
          )
        case None =>
          println(f"${run.mode}%-11s skipped: ${run.detail}")
    }

    val scalaMeanOpt = modeRuns.find(_.mode == "scala").flatMap(_.stats).map(_.meanMs)
    scalaMeanOpt.foreach { scalaMean =>
      modeRuns.foreach { run =>
        run.stats.foreach { stats =>
          if run.mode != "scala" then
            println(f"speedup(${run.mode} vs scala): ${scalaMean / stats.meanMs}%.3fx")
        }
      }
    }
    println("=== Done ===")

  private def runMode(
      mode: String,
      config: Config,
      spot: BenchmarkSpot,
      runSeeds: Vector[Long]
  ): ModeRun =
    val modeConfig = mode.toLowerCase match
      case "scala" =>
        Some(
          (
          Seq(
            "sicfun.postflop.provider" -> Some("disabled")
          ),
          false
          )
        )
      case "native-cpu" =>
        Some(
          (
          Seq(
            "sicfun.postflop.provider" -> Some("native"),
            "sicfun.postflop.native.engine" -> Some("cpu"),
            "sicfun.postflop.native.path" -> config.nativeCpuPath
          ),
          true
          )
        )
      case "native-cuda" =>
        Some(
          (
          Seq(
            "sicfun.postflop.provider" -> Some("native"),
            "sicfun.postflop.native.engine" -> Some("cuda"),
            "sicfun.postflop.native.gpu.path" -> config.nativeGpuPath
          ),
          true
          )
        )
      case "native-auto" =>
        Some(
          (
          Seq(
            "sicfun.postflop.provider" -> Some("native"),
            "sicfun.postflop.native.engine" -> Some("auto"),
            "sicfun.postflop.native.path" -> config.nativeCpuPath,
            "sicfun.postflop.native.gpu.path" -> config.nativeGpuPath
          ),
          true
          )
        )
      case other =>
        None

    modeConfig match
      case None =>
        ModeRun(mode = mode, stats = None, detail = "unknown mode")
      case Some((propertyUpdates, availabilityCheck)) =>
        withSystemProperties(propertyUpdates) {
          HoldemPostflopNativeRuntime.resetLoadCacheForTests()
          if availabilityCheck then
            val availability = HoldemPostflopNativeRuntime.availability
            if !availability.available then
              ModeRun(mode, None, availability.detail)
            else if mode.startsWith("native") then
              probeNativeExecution(
                spot = spot,
                trials = math.min(1200, config.trials),
                seed = runSeeds.headOption.getOrElse(1L)
              ) match
                case Left(reason) =>
                  ModeRun(mode, None, s"native execution probe failed: $reason")
                case Right(_) =>
                  runMeasuredMode(mode, config, spot, runSeeds)
            else
              runMeasuredMode(mode, config, spot, runSeeds)
          else
            runMeasuredMode(mode, config, spot, runSeeds)
        }

  private def runMeasuredMode(
      mode: String,
      config: Config,
      spot: BenchmarkSpot,
      runSeeds: Vector[Long]
  ): ModeRun =
    var warmup = 0
    while warmup < config.warmupRuns do
      runOne(spot, config.trials, runSeeds(warmup))
      warmup += 1

    val samplesMs = new Array[Double](config.measureRuns)
    var i = 0
    while i < config.measureRuns do
      val started = System.nanoTime()
      runOne(spot, config.trials, runSeeds(config.warmupRuns + i))
      val elapsed = math.max(1L, System.nanoTime() - started)
      samplesMs(i) = elapsed.toDouble / 1_000_000.0
      i += 1

    val stats = computeStats(samplesMs.toVector)
    ModeRun(mode, Some(stats), detail = "")

  private def runOne(spot: BenchmarkSpot, trials: Int, seed: Long): EquityEstimate =
    HoldemEquity.equityMonteCarlo(
      hero = spot.hero,
      board = spot.board,
      villainRange = spot.range,
      trials = trials,
      rng = new Random(seed)
    )

  private def probeNativeExecution(spot: BenchmarkSpot, trials: Int, seed: Long): Either[String, Double] =
    HoldemPostflopNativeRuntime
      .computePostflopBatch(
        hero = spot.hero,
        board = spot.board,
        villains = spot.villains,
        trials = trials,
        seedBase = seed
      )
      .map { rows =>
        var weightedEquity = 0.0
        var weightSum = 0.0
        var i = 0
        while i < rows.length do
          val w = spot.villainWeights(i)
          weightedEquity += w * rows(i).equity
          weightSum += w
          i += 1
        if weightSum > 0.0 then weightedEquity / weightSum else 0.0
      }

  private def computeStats(valuesMs: Vector[Double]): Stats =
    val sorted = valuesMs.sorted
    val count = sorted.length
    require(count > 0, "cannot compute benchmark stats for empty sample set")
    val mean = sorted.sum / count.toDouble
    val median = quantile(sorted, 0.5)
    Stats(
      count = count,
      meanMs = mean,
      medianMs = median,
      minMs = sorted.head,
      maxMs = sorted.last
    )

  private def quantile(sorted: Vector[Double], q: Double): Double =
    if sorted.length == 1 then sorted.head
    else
      val p = q * (sorted.length - 1).toDouble
      val lo = math.floor(p).toInt
      val hi = math.ceil(p).toInt
      if lo == hi then sorted(lo)
      else
        val w = p - lo.toDouble
        sorted(lo) * (1.0 - w) + sorted(hi) * w

  private def parseArgs(args: Vector[String]): Config =
    val options = args.flatMap { token =>
      token.split("=", 2) match
        case Array(key, value) if key.startsWith("--") && value.nonEmpty =>
          Some(key.drop(2) -> value)
        case _ =>
          None
    }.toMap

    val rawModes = options.getOrElse("modes", "scala,native-cpu,native-cuda")
    val modes =
      rawModes.split(',').toVector.map(_.trim.toLowerCase).filter(_.nonEmpty).distinct

    Config(
      warmupRuns = options.get("warmupRuns").flatMap(_.toIntOption).getOrElse(2),
      measureRuns = options.get("measureRuns").flatMap(_.toIntOption).getOrElse(8),
      trials = options.get("trials").flatMap(_.toIntOption).getOrElse(12_000),
      seed = options.get("seed").flatMap(_.toLongOption).getOrElse(23L),
      modes = modes,
      nativeCpuPath = options.get("nativeCpuPath"),
      nativeGpuPath = options.get("nativeGpuPath")
    )

  private def benchmarkSpot(seed: Long): BenchmarkSpot =
    def card(token: String): Card =
      Card.parse(token).getOrElse(throw new IllegalArgumentException(s"invalid card token: $token"))

    def hole(a: String, b: String): HoleCards =
      HoleCards.from(Vector(card(a), card(b)))

    val hero = hole("Ac", "Kh")
    val board = Board.from(Seq(card("Ts"), card("9h"), card("8d")))
    val rng = new Random(seed)
    val villains = Vector(
      hole("Ah", "Qh") -> 0.12,
      hole("As", "Kd") -> 0.09,
      hole("Jc", "Td") -> 0.16,
      hole("9c", "9d") -> 0.13,
      hole("Qs", "Js") -> 0.12,
      hole("7h", "6h") -> 0.08,
      hole("Ad", "5d") -> 0.14,
      hole("Kc", "Qc") -> 0.16
    )
    val jittered = villains.map { case (hand, w) =>
      hand -> (w * (0.9 + 0.2 * rng.nextDouble()))
    }
    val sorted = jittered.sortBy(_._1.toToken)
    val weightsMap = sorted.toMap
    BenchmarkSpot(
      hero = hero,
      board = board,
      villains = sorted.map(_._1).toArray,
      villainWeights = sorted.map(_._2).toArray,
      range = DiscreteDistribution(weightsMap)
    )

  private def withSystemProperties[A](updates: Seq[(String, Option[String])])(thunk: => A): A =
    val previous = updates.map { case (key, _) => key -> sys.props.get(key) }.toMap
    updates.foreach {
      case (key, Some(value)) => sys.props.update(key, value)
      case (key, None) => sys.props.remove(key)
    }
    try thunk
    finally
      previous.foreach {
        case (key, Some(value)) => sys.props.update(key, value)
        case (key, None) => sys.props.remove(key)
      }
