package sicfun.holdem.bench
import sicfun.holdem.types.*
import sicfun.holdem.*
import sicfun.holdem.equity.*
import sicfun.holdem.gpu.*
import sicfun.holdem.cli.*

import sicfun.core.{Card, CardId, DiscreteDistribution}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/** Benchmark harness for postflop Monte Carlo runtime backends. */
object HoldemPostflopNativeBenchmark:
  private final case class Config(
      warmupRuns: Int = 2,
      measureRuns: Int = 8,
      trials: Int = 12_000,
      villains: Int = 8,
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

  private final case class ModeSetup(
      propertyUpdates: Seq[(String, Option[String])],
      requiresAvailability: Boolean
  )

  private final class BenchmarkRunner(config: Config):
    private val runSeeds = {
      val rng = new Random(config.seed)
      Vector.fill(config.warmupRuns + config.measureRuns)(rng.nextLong())
    }
    private val spot = benchmarkSpot(config.seed ^ 0xD6E8FEB86659FD93L, config.villains)

    def run(): Unit =
      printHeader(config)
      val modeRuns = config.modes.map { mode =>
        runMode(mode)
      }
      printModeRuns(modeRuns)
      printSpeedups(modeRuns)
      println("=== Done ===")

    private def runMode(mode: String): ModeRun =
      modeSetup(mode) match
        case None => ModeRun(mode = mode, stats = None, detail = "unknown mode")
        case Some(setup) =>
          withSystemProperties(setup.propertyUpdates) {
            executeMode(mode, setup)
          }

    private def executeMode(mode: String, setup: ModeSetup): ModeRun =
      HoldemPostflopNativeRuntime.resetLoadCacheForTests()
      skipReason(mode, setup) match
        case Some(reason) => ModeRun(mode, None, reason)
        case None => measureMode(mode)

    private def skipReason(mode: String, setup: ModeSetup): Option[String] =
      if !setup.requiresAvailability then None
      else
        val availability = HoldemPostflopNativeRuntime.availability
        if !availability.available then Some(availability.detail)
        else if mode.startsWith("native") then
          probeNativeExecution(
            spot = spot,
            trials = math.min(1200, config.trials),
            seed = runSeeds.headOption.getOrElse(1L)
          ).left.toOption.map(reason => s"native execution probe failed: $reason")
        else None

    private def measureMode(mode: String): ModeRun =
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

      ModeRun(mode, Some(computeStats(samplesMs.toVector)), detail = "")

    private def modeSetup(mode: String): Option[ModeSetup] =
      mode.toLowerCase match
        case "scala" =>
          Some(
            ModeSetup(
              propertyUpdates = Seq("sicfun.postflop.provider" -> Some("disabled")),
              requiresAvailability = false
            )
          )
        case "native-cpu" =>
          Some(
            ModeSetup(
              propertyUpdates = Seq(
                "sicfun.postflop.provider" -> Some("native"),
                "sicfun.postflop.native.engine" -> Some("cpu"),
                "sicfun.postflop.native.path" -> config.nativeCpuPath
              ),
              requiresAvailability = true
            )
          )
        case "native-cuda" =>
          Some(
            ModeSetup(
              propertyUpdates = Seq(
                "sicfun.postflop.provider" -> Some("native"),
                "sicfun.postflop.native.engine" -> Some("cuda"),
                "sicfun.postflop.native.gpu.path" -> config.nativeGpuPath
              ),
              requiresAvailability = true
            )
          )
        case "native-auto" =>
          Some(
            ModeSetup(
              propertyUpdates = Seq(
                "sicfun.postflop.provider" -> Some("native"),
                "sicfun.postflop.native.engine" -> Some("auto"),
                "sicfun.postflop.native.path" -> config.nativeCpuPath,
                "sicfun.postflop.native.gpu.path" -> config.nativeGpuPath
              ),
              requiresAvailability = true
            )
          )
        case _ =>
          None

  def main(args: Array[String]): Unit =
    val config = parseArgs(args.toVector)
    validateConfig(config)
    new BenchmarkRunner(config).run()

  private def validateConfig(config: Config): Unit =
    require(config.warmupRuns >= 0, "warmupRuns must be non-negative")
    require(config.measureRuns > 0, "measureRuns must be positive")
    require(config.trials > 0, "trials must be positive")
    require(config.villains > 0, "villains must be positive")
    require(config.modes.nonEmpty, "at least one benchmark mode is required")

  private def printHeader(config: Config): Unit =
    println("=== Holdem Postflop Native Benchmark ===")
    println(
      s"config: warmupRuns=${config.warmupRuns}, measureRuns=${config.measureRuns}, " +
        s"trials=${config.trials}, villains=${config.villains}, seed=${config.seed}, " +
        s"modes=${config.modes.mkString(",")}"
    )

  private def printModeRuns(modeRuns: Vector[ModeRun]): Unit =
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

  private def printSpeedups(modeRuns: Vector[ModeRun]): Unit =
    val scalaMeanOpt = modeRuns.find(_.mode == "scala").flatMap(_.stats).map(_.meanMs)
    scalaMeanOpt.foreach { scalaMean =>
      modeRuns.foreach { run =>
        run.stats.foreach { stats =>
          if run.mode != "scala" then
            println(f"speedup(${run.mode} vs scala): ${scalaMean / stats.meanMs}%.3fx")
        }
      }
    }

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
    val options = CliHelpers.requireOptions(args)
    CliHelpers.requireNoUnknownOptions(options, Set("warmupRuns", "measureRuns", "trials", "villains", "seed", "modes", "nativeCpuPath", "nativeGpuPath"))

    val rawModes = options.getOrElse("modes", "scala,native-cpu,native-cuda")
    val modes = CliHelpers.requireCsvTokens(rawModes, "modes").map(_.toLowerCase).distinct

    Config(
      warmupRuns = CliHelpers.requireIntOption(options, "warmupRuns", 2),
      measureRuns = CliHelpers.requireIntOption(options, "measureRuns", 8),
      trials = CliHelpers.requireIntOption(options, "trials", 12_000),
      villains = CliHelpers.requireIntOption(options, "villains", 8),
      seed = CliHelpers.requireLongOption(options, "seed", 23L),
      modes = modes,
      nativeCpuPath = options.get("nativeCpuPath"),
      nativeGpuPath = options.get("nativeGpuPath")
    )

  private def benchmarkSpot(seed: Long, villainCount: Int): BenchmarkSpot =
    def card(token: String): Card =
      Card.parse(token).getOrElse(throw new IllegalArgumentException(s"invalid card token: $token"))

    def hole(a: String, b: String): HoleCards =
      HoleCards.from(Vector(card(a), card(b)))

    val hero = hole("Ac", "Kh")
    val board = Board.from(Seq(card("Ts"), card("9h"), card("8d")))
    val rng = new Random(seed)
    val baseVillains = Vector(
      hole("Ah", "Qh") -> 0.12,
      hole("As", "Kd") -> 0.09,
      hole("Jc", "Td") -> 0.16,
      hole("9c", "9d") -> 0.13,
      hole("Qs", "Js") -> 0.12,
      hole("7h", "6h") -> 0.08,
      hole("Ad", "5d") -> 0.14,
      hole("Kc", "Qc") -> 0.16
    )

    val selected = ArrayBuffer.empty[(HoleCards, Double)]
    val seen = scala.collection.mutable.HashSet.empty[HoleCards]
    baseVillains.foreach { case (hand, w) =>
      if selected.length < villainCount then
        selected += ((hand, w))
        seen += hand
    }

    val deadIds = (hero.toVector ++ board.cards).map(CardId.toId).toSet
    var cursor = (math.abs((seed ^ (seed >>> 32)).toInt) % HoleCardsIndex.size + HoleCardsIndex.size) % HoleCardsIndex.size
    val step = 37 // coprime with 1326
    var guard = 0
    while selected.length < villainCount && guard < HoleCardsIndex.size * 2 do
      val hand = HoleCardsIndex.byId(cursor)
      val firstId = CardId.toId(hand.first)
      val secondId = CardId.toId(hand.second)
      if !deadIds.contains(firstId) && !deadIds.contains(secondId) && !seen.contains(hand) then
        val w = 0.05 + rng.nextDouble()
        selected += ((hand, w))
        seen += hand
      cursor = (cursor + step) % HoleCardsIndex.size
      guard += 1

    require(selected.length == villainCount, s"unable to build villain batch size=$villainCount")

    val jittered = selected.toVector.map { case (hand, w) =>
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
