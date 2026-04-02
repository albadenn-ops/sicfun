package sicfun.holdem.bench

import sicfun.core.DiscreteDistribution
import sicfun.holdem.bench.BenchSupport.{card, hole}
import sicfun.holdem.cfr.{HoldemCfrConfig, HoldemCfrNativeDecisionProfile, HoldemCfrNativeRuntime, HoldemCfrSolver}
import sicfun.holdem.cli.CliHelpers
import sicfun.holdem.types.*

import scala.collection.mutable

/** Stage breakdown benchmark for CFR decision solves.
  *
  * Reports steady-state end-to-end latency by provider and, for native modes,
  * splits the measured work into:
  *   - `prepareGame`
  *   - `buildNativeSpec`
  *   - `nativeSolve`
  *   - `unpackRootPolicy`
  *
  * Usage:
  *   sbt "runMain sicfun.holdem.bench.HoldemCfrStageBenchmark --scenario=turn"
  *   sbt "runMain sicfun.holdem.bench.HoldemCfrStageBenchmark --scenario=preflop --modes=scala,native-cpu"
  */
object HoldemCfrStageBenchmark:
  private final case class Config(
      warmupRuns: Int = 2,
      measureRuns: Int = 8,
      scenario: String = "turn",
      modes: Vector[String] = Vector("scala", "native-cpu", "native-gpu")
  )

  private final case class Spot(
      hero: HoleCards,
      state: GameState,
      posterior: DiscreteDistribution[HoleCards],
      candidateActions: Vector[PokerAction],
      config: HoldemCfrConfig,
      label: String
  )

  private final case class Sample(
      mode: String,
      stage: String,
      elapsedNanos: Long
  ):
    def elapsedMs: Double = elapsedNanos.toDouble / 1_000_000.0

  private final case class Stats(
      count: Int,
      meanMs: Double,
      medianMs: Double,
      minMs: Double,
      maxMs: Double
  )

  private final case class ModeMetadata(
      requestedMode: String,
      resolvedProviders: Vector[String],
      nativeProfile: Option[HoldemCfrNativeDecisionProfile]
  )

  def main(args: Array[String]): Unit =
    val config = parseArgs(args.toVector)
    validateConfig(config)
    val spot = benchmarkSpot(config.scenario)
    val metadataByMode = mutable.LinkedHashMap.empty[String, ModeMetadata]
    val samples = Vector.newBuilder[Sample]

    println("=== Holdem CFR Stage Benchmark ===")
    println(
      s"config: warmupRuns=${config.warmupRuns}, measureRuns=${config.measureRuns}, " +
        s"scenario=${spot.label}, modes=${config.modes.mkString(",")}"
    )
    println()

    config.modes.foreach { mode =>
      val modeResult = runMode(mode, spot, config)
      metadataByMode.update(mode, modeResult._1)
      samples ++= modeResult._2
    }

    printResults(samples.result(), metadataByMode.toMap)
    println("=== Done ===")

  private def runMode(
      mode: String,
      spot: Spot,
      config: Config
  ): (ModeMetadata, Vector[Sample]) =
    val resolvedProviders = mutable.LinkedHashSet.empty[String]
    var profileMetadata: Option[HoldemCfrNativeDecisionProfile] = None
    val samples = Vector.newBuilder[Sample]

    withModeContext(mode) {
      var warmup = 0
      while warmup < config.warmupRuns do
        val warmupPolicy = HoldemCfrSolver.solveDecisionPolicy(
          hero = spot.hero,
          state = spot.state,
          villainPosterior = spot.posterior,
          candidateActions = spot.candidateActions,
          config = spot.config
        )
        resolvedProviders += warmupPolicy.provider
        backendForMode(mode).foreach { backend =>
          HoldemCfrSolver.profileNativeDecisionPolicy(
            hero = spot.hero,
            state = spot.state,
            villainPosterior = spot.posterior,
            candidateActions = spot.candidateActions,
            config = spot.config,
            backend = backend
          ) match
            case Right(profile) =>
              profileMetadata = profileMetadata.orElse(Some(profile))
            case Left(_) => ()
        }
        warmup += 1

      var run = 0
      while run < config.measureRuns do
        val (policy, endToEndNanos) = timedResult {
          HoldemCfrSolver.solveDecisionPolicy(
            hero = spot.hero,
            state = spot.state,
            villainPosterior = spot.posterior,
            candidateActions = spot.candidateActions,
            config = spot.config
          )
        }
        resolvedProviders += policy.provider
        samples += Sample(mode, "endToEndDecision", endToEndNanos)

        backendForMode(mode).foreach { backend =>
          HoldemCfrSolver.profileNativeDecisionPolicy(
            hero = spot.hero,
            state = spot.state,
            villainPosterior = spot.posterior,
            candidateActions = spot.candidateActions,
            config = spot.config,
            backend = backend
          ) match
            case Right(profile) =>
              profileMetadata = profileMetadata.orElse(Some(profile))
              samples += Sample(mode, "prepareGame", profile.prepareNanos)
              samples += Sample(mode, "prepareSupport", profile.prepareSupportNanos)
              samples += Sample(mode, "prepareEquityLookup", profile.prepareEquityNanos)
              samples += Sample(mode, "prepareResponses", profile.prepareResponseNanos)
              samples += Sample(mode, "prepareGameBuild", profile.prepareGameBuildNanos)
              samples += Sample(mode, "buildNativeSpec", profile.specBuildNanos)
              samples += Sample(mode, "nativeSolve", profile.nativeSolveNanos)
              samples += Sample(mode, "unpackRootPolicy", profile.unpackNanos)
              samples += Sample(mode, "jvmMeasuredOverhead", profile.jvmMeasuredNanos)
              samples += Sample(mode, "profiledNativeTotal", profile.measuredTotalNanos)
            case Left(reason) =>
              throw new IllegalStateException(s"native profiling for mode '$mode' failed: $reason")
        }
        run += 1
    }

    (
      ModeMetadata(
        requestedMode = mode,
        resolvedProviders = resolvedProviders.toVector,
        nativeProfile = profileMetadata
      ),
      samples.result()
    )

  private def withModeContext[A](mode: String)(thunk: => A): A =
    val previousProvider = sys.props.get("sicfun.cfr.provider")
    val previousDirect = sys.props.get("sicfun.cfr.directShallowApproximation")
    HoldemCfrNativeRuntime.resetLoadCacheForTests()
    HoldemCfrSolver.resetAutoProviderForTests()
    System.setProperty("sicfun.cfr.provider", mode)
    System.setProperty("sicfun.cfr.directShallowApproximation", "false")
    try thunk
    finally
      previousProvider match
        case Some(value) => System.setProperty("sicfun.cfr.provider", value)
        case None => System.clearProperty("sicfun.cfr.provider")
      previousDirect match
        case Some(value) => System.setProperty("sicfun.cfr.directShallowApproximation", value)
        case None => System.clearProperty("sicfun.cfr.directShallowApproximation")
      HoldemCfrNativeRuntime.resetLoadCacheForTests()
      HoldemCfrSolver.resetAutoProviderForTests()

  private def timedResult[A](thunk: => A): (A, Long) =
    val started = System.nanoTime()
    val result = thunk
    (result, math.max(1L, System.nanoTime() - started))

  private def printResults(
      samples: Vector[Sample],
      metadataByMode: Map[String, ModeMetadata]
  ): Unit =
    val grouped = samples.groupBy(_.mode).toVector.sortBy(_._1)
    grouped.foreach { case (mode, modeSamples) =>
      val metadata = metadataByMode(mode)
      println(s"--- mode=$mode ---")
      println(s"resolvedProviders=${metadata.resolvedProviders.mkString(",")}")
      metadata.nativeProfile.foreach { profile =>
        println(
          s"nativeProfile: provider=${profile.provider}, villainSupport=${profile.villainSupport}, " +
            s"nodeCount=${profile.nodeCount}, infoSetCount=${profile.infoSetCount}, bestAction=${profile.bestAction}"
        )
      }
      val stageGroups = modeSamples.groupBy(_.stage).toVector.sortBy(_._1)
      stageGroups.foreach { case (stage, stageSamples) =>
        val stats = computeStats(stageSamples.map(_.elapsedMs))
        println(
          f"$stage%-20s count=${stats.count}%2d mean=${stats.meanMs}%.3fms " +
            f"median=${stats.medianMs}%.3fms min=${stats.minMs}%.3fms max=${stats.maxMs}%.3fms"
        )
      }
      metadata.nativeProfile.foreach { _ =>
        val endToEndMedian = medianMs(modeSamples, "endToEndDecision")
        val nativeSolveMedian = medianMs(modeSamples, "nativeSolve")
        val jvmMeasuredMedian = medianMs(modeSamples, "jvmMeasuredOverhead")
        val profiledTotalMedian = medianMs(modeSamples, "profiledNativeTotal")
        val nativeShare = if profiledTotalMedian > 0.0 then (nativeSolveMedian / profiledTotalMedian) * 100.0 else 0.0
        val jvmShare = if profiledTotalMedian > 0.0 then (jvmMeasuredMedian / profiledTotalMedian) * 100.0 else 0.0
        val profiledCoverage = if endToEndMedian > 0.0 then (profiledTotalMedian / endToEndMedian) * 100.0 else 0.0
        println(
          f"derived: nativeSolveShare=$nativeShare%.1f%% jvmMeasuredShare=$jvmShare%.1f%% " +
            f"profiledTotalVsEndToEnd=$profiledCoverage%.1f%%"
        )
      }
      println()
    }

  private def medianMs(samples: Vector[Sample], stage: String): Double =
    val values = samples.iterator.filter(_.stage == stage).map(_.elapsedMs).toVector.sorted
    require(values.nonEmpty, s"missing stage '$stage'")
    quantile(values, 0.5)

  private def computeStats(valuesMs: Vector[Double]): Stats =
    val sorted = valuesMs.sorted
    val count = sorted.length
    require(count > 0, "cannot compute benchmark stats for empty sample set")
    Stats(
      count = count,
      meanMs = sorted.sum / count.toDouble,
      medianMs = quantile(sorted, 0.5),
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

  private def benchmarkSpot(name: String): Spot =
    name.trim.toLowerCase(java.util.Locale.ROOT) match
      case "preflop" =>
        Spot(
          hero = hole("Ac", "Ad"),
          state = GameState(
            street = Street.Preflop,
            board = Board.empty,
            pot = 6.0,
            toCall = 2.0,
            position = Position.Button,
            stackSize = 100.0,
            betHistory = Vector.empty
          ),
          posterior = DiscreteDistribution(
            Map(
              hole("7c", "2d") -> 0.6,
              hole("Kc", "Qd") -> 0.3,
              hole("Ks", "Kh") -> 0.1
            )
          ),
          candidateActions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(8.0)),
          config = HoldemCfrConfig(
            iterations = 1_200,
            averagingDelay = 100,
            maxVillainHands = 32,
            equityTrials = 1_000,
            preferNativeBatch = true,
            rngSeed = 37L
          ),
          label = "preflop premium / 3-action root"
        )
      case "turn" =>
        Spot(
          hero = hole("Ac", "Kd"),
          state = GameState(
            street = Street.Turn,
            board = Board.from(Vector(card("2c"), card("7d"), card("Jh"), card("Qc"))),
            pot = 18.0,
            toCall = 4.0,
            position = Position.Button,
            stackSize = 82.0,
            betHistory = Vector(BetAction(1, PokerAction.Raise(4.0)))
          ),
          posterior = DiscreteDistribution(
            Map(
              hole("As", "Qd") -> 0.35,
              hole("Ts", "9s") -> 0.30,
              hole("Qh", "Js") -> 0.20,
              hole("7c", "7s") -> 0.15
            )
          ),
          candidateActions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(12.0)),
          config = HoldemCfrConfig(
            iterations = 700,
            averagingDelay = 100,
            maxVillainHands = 32,
            equityTrials = 700,
            preferNativeBatch = true,
            rngSeed = 41L
          ),
          label = "turn 4-card board / 3-action root"
        )
      case other =>
        throw new IllegalArgumentException(s"unsupported scenario: $other")

  private def parseArgs(args: Vector[String]): Config =
    val options = CliHelpers.requireOptions(args)
    CliHelpers.requireNoUnknownOptions(
      options,
      allowedKeys = Set("warmupRuns", "measureRuns", "scenario", "modes")
    )
    val modes =
      CliHelpers.requireCsvTokens(options.getOrElse("modes", "scala,native-cpu,native-gpu"), "modes")
        .map(_.trim.toLowerCase(java.util.Locale.ROOT))
        .distinct
    Config(
      warmupRuns = CliHelpers.requireIntOption(options, "warmupRuns", 2),
      measureRuns = CliHelpers.requireIntOption(options, "measureRuns", 8),
      scenario = options.getOrElse("scenario", "turn"),
      modes = modes
    )

  private def validateConfig(config: Config): Unit =
    require(config.warmupRuns >= 0, "warmupRuns must be non-negative")
    require(config.measureRuns > 0, "measureRuns must be positive")
    require(config.modes.nonEmpty, "modes must be non-empty")
    config.modes.foreach(backendForMode)

  private def backendForMode(mode: String): Option[HoldemCfrNativeRuntime.Backend] =
    mode match
      case "scala" => None
      case "native-cpu" => Some(HoldemCfrNativeRuntime.Backend.Cpu)
      case "native-gpu" => Some(HoldemCfrNativeRuntime.Backend.Gpu)
      case other => throw new IllegalArgumentException(s"unsupported mode: $other")
