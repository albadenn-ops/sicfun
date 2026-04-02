package sicfun.holdem.bench
import sicfun.holdem.types.*
import sicfun.holdem.*
import sicfun.holdem.equity.*
import sicfun.holdem.gpu.*
import sicfun.holdem.provider.*
import sicfun.holdem.model.*
import sicfun.holdem.cli.*

import sicfun.core.DiscreteDistribution
import sicfun.holdem.bench.BenchSupport.{card, hole}

import scala.collection.mutable

/** DDRE provider parity + benchmark gate.
  *
  * Runs a deterministic DDRE inference workload across selected providers, compares
  * posterior parity against a reference provider, and enforces configurable gate
  * thresholds (parity + optional speedup).
  *
  * Exit codes: 0 = PASS, 2 = FAIL.
  */
object HoldemDdreParityBenchmark:
  private val DdreOnnxArtifactDirProperty = "sicfun.ddre.onnx.artifactDir"
  private val DdreNativeCpuPathProperty = "sicfun.ddre.native.cpu.path"
  private val DdreNativeGpuPathProperty = "sicfun.ddre.native.gpu.path"
  private val DdreOnnxModelPathProperty = "sicfun.ddre.onnx.modelPath"
  private val DdreOnnxAllowExperimentalProperty = "sicfun.ddre.onnx.allowExperimental"
  private val AllowedOptionKeys = Set(
    "warmupRuns",
    "measureRuns",
    "hypothesisCount",
    "seed",
    "modes",
    "referenceMode",
    "maxL1Diff",
    "maxAbsDiff",
    "minSpeedupVsReference",
    "requireAllModes",
    "nativeCpuPath",
    "nativeGpuPath",
    "onnxModelPath",
    "onnxArtifactDir",
    "onnxAllowExperimental"
  )

  private final case class Config(
      warmupRuns: Int = 1,
      measureRuns: Int = 6,
      hypothesisCount: Int = 256,
      seed: Long = 29L,
      modes: Vector[String] = Vector("synthetic", "native-cpu", "native-gpu"),
      referenceMode: String = "synthetic",
      maxL1Diff: Double = 1e-6,
      maxAbsDiff: Double = 1e-7,
      minSpeedupVsReference: Double = 0.0,
      requireAllModes: Boolean = false,
      nativeCpuPath: Option[String] = None,
      nativeGpuPath: Option[String] = None,
      onnxModelPath: Option[String] = None,
      onnxArtifactDir: Option[String] = None,
      onnxAllowExperimental: Boolean = false
  )

  private final case class BenchmarkSpot(
      hero: HoleCards,
      board: Board,
      prior: DiscreteDistribution[HoleCards],
      observations: Vector[(PokerAction, GameState)],
      actionModel: PokerActionModel
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
      posterior: Option[DiscreteDistribution[HoleCards]],
      detail: String
  )

  private final case class ParitySummary(
      checked: Int,
      l1: Double,
      maxAbs: Double
  )

  private final case class ComparisonSummary(
      mode: String,
      parity: ParitySummary,
      speedupVsReference: Double,
      parityPass: Boolean,
      speedPass: Boolean
  )

  private final case class GateOutcome(
      pass: Boolean,
      failures: Vector[String]
  )

  private final case class SampleSet(
      stats: Stats,
      posterior: DiscreteDistribution[HoleCards]
  )

  private final class BenchmarkRunner(config: Config):
    private val spot = benchmarkSpot(config.hypothesisCount)
    private val propertyUpdates = modePropertyUpdates(config)

    def run(): Int =
      printHeader()
      val modeRuns = runModes()
      printModeRuns(modeRuns)
      val outcome = evaluateGate(modeRuns)
      printGateResult(outcome)
      if outcome.pass then 0 else 2

    private def printHeader(): Unit =
      println("=== Holdem DDRE Parity Benchmark ===")
      println(
        s"config: warmupRuns=${config.warmupRuns}, measureRuns=${config.measureRuns}, " +
          s"hypothesisCount=${spot.prior.weights.size}, seed=${config.seed}, " +
          s"modes=${config.modes.mkString(",")}, referenceMode=${config.referenceMode}, " +
          s"maxL1Diff=${config.maxL1Diff}, maxAbsDiff=${config.maxAbsDiff}, " +
          s"minSpeedupVsReference=${config.minSpeedupVsReference}, requireAllModes=${config.requireAllModes}"
      )
      println(
        s"nativeCpuPath=${config.nativeCpuPath.getOrElse("(runtime default)")}, " +
          s"nativeGpuPath=${config.nativeGpuPath.getOrElse("(runtime default)")}, " +
          s"onnxModelPath=${config.onnxModelPath.getOrElse("(unset)")}, " +
          s"onnxArtifactDir=${config.onnxArtifactDir.getOrElse("(unset)")}, " +
          s"onnxAllowExperimental=${config.onnxAllowExperimental}"
      )

    private def runModes(): Vector[ModeRun] =
      config.modes.map { mode =>
        runMode(mode)
      }

    private def runMode(mode: String): ModeRun =
      providerForMode(mode) match
        case None =>
          unavailableModeRun(mode, "unknown mode")
        case Some(provider) =>
          withSystemProperties(propertyUpdates) {
            executeMode(mode, provider)
          }

    private def executeMode(
        mode: String,
        provider: HoldemDdreProvider.Provider
    ): ModeRun =
      HoldemDdreNativeRuntime.resetLoadCacheForTests()
      availabilityReason(mode) match
        case Some(reason) =>
          unavailableModeRun(mode, reason)
        case None =>
          measureMode(mode, provider)

    private def availabilityReason(mode: String): Option[String] =
      checkAvailability(mode).left.toOption

    private def measureMode(
        mode: String,
        provider: HoldemDdreProvider.Provider
    ): ModeRun =
      warmupProvider(provider) match
        case Left(reason) =>
          unavailableModeRun(mode, s"warmup_failed:$reason")
        case Right(_) =>
          captureSamples(provider) match
            case Left(reason) =>
              unavailableModeRun(mode, s"measure_failed:$reason")
            case Right(sampleSet) =>
              ModeRun(
                mode = mode,
                stats = Some(sampleSet.stats),
                posterior = Some(sampleSet.posterior),
                detail = ""
              )

    private def warmupProvider(
        provider: HoldemDdreProvider.Provider
    ): Either[String, Unit] =
      var warmup = 0
      var failure = Option.empty[String]
      while warmup < config.warmupRuns && failure.isEmpty do
        runOne(provider, spot) match
          case Left(reason) =>
            failure = Some(reason)
          case Right(_) =>
            ()
        warmup += 1
      failure match
        case Some(reason) => Left(reason)
        case None => Right(())

    private def captureSamples(
        provider: HoldemDdreProvider.Provider
    ): Either[String, SampleSet] =
      val samplesMs = new Array[Double](config.measureRuns)
      var firstPosterior = Option.empty[DiscreteDistribution[HoleCards]]
      var failure = Option.empty[String]
      var i = 0
      while i < config.measureRuns && failure.isEmpty do
        val started = System.nanoTime()
        runOne(provider, spot) match
          case Left(reason) =>
            failure = Some(reason)
          case Right(posterior) =>
            if firstPosterior.isEmpty then firstPosterior = Some(posterior)
            val elapsed = math.max(1L, System.nanoTime() - started)
            samplesMs(i) = elapsed.toDouble / 1_000_000.0
        i += 1

      failure match
        case Some(reason) =>
          Left(reason)
        case None =>
          val posterior = firstPosterior.getOrElse(
            throw new IllegalStateException("posterior unexpectedly missing after successful measurements")
          )
          Right(SampleSet(computeStats(samplesMs.toVector), posterior))

    private def unavailableModeRun(mode: String, detail: String): ModeRun =
      ModeRun(mode = mode, stats = None, posterior = None, detail = detail)

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

    private def evaluateGate(modeRuns: Vector[ModeRun]): GateOutcome =
      val failures = mutable.ArrayBuffer.empty[String]
      if config.requireAllModes then
        modeRuns.foreach { run =>
          if run.stats.isEmpty || run.posterior.isEmpty then
            failures += s"mode_${run.mode}_unavailable"
        }

      referenceRun(modeRuns) match
        case None =>
          failures += s"reference_mode_${config.referenceMode}_unavailable"
        case Some(reference) =>
          val comparisons = comparisonSummaries(modeRuns, reference)
          if comparisons.isEmpty then
            failures += "no_candidate_modes_available_for_comparison"
          else
            printComparisons(comparisons)
            comparisons.foreach { summary =>
              if !summary.parityPass then failures += s"parity_${summary.mode}_failed"
              if !summary.speedPass then failures += s"speed_${summary.mode}_failed"
            }

      GateOutcome(failures.isEmpty, failures.toVector)

    private def referenceRun(modeRuns: Vector[ModeRun]): Option[ModeRun] =
      modeRuns.find(run =>
        run.mode == config.referenceMode &&
          run.stats.nonEmpty &&
          run.posterior.nonEmpty
      )

    private def comparisonSummaries(
        modeRuns: Vector[ModeRun],
        referenceRun: ModeRun
    ): Vector[ComparisonSummary] =
      val referenceStats = referenceRun.stats.getOrElse(
        throw new IllegalStateException("reference stats unexpectedly missing")
      )
      val referencePosterior = referenceRun.posterior.getOrElse(
        throw new IllegalStateException("reference posterior unexpectedly missing")
      )
      modeRuns.flatMap { candidate =>
        if candidate.mode == config.referenceMode then None
        else
          (candidate.stats, candidate.posterior) match
            case (Some(stats), Some(posterior)) =>
              val parity = computeParity(referencePosterior, posterior)
              val speedup = referenceStats.meanMs / stats.meanMs
              Some(
                ComparisonSummary(
                  mode = candidate.mode,
                  parity = parity,
                  speedupVsReference = speedup,
                  parityPass =
                    parity.l1 <= config.maxL1Diff &&
                      parity.maxAbs <= config.maxAbsDiff,
                  speedPass =
                    config.minSpeedupVsReference <= 0.0 ||
                      speedup + 1e-12 >= config.minSpeedupVsReference
                )
              )
            case _ => None
      }

    private def printComparisons(comparisons: Vector[ComparisonSummary]): Unit =
      println()
      comparisons.foreach { summary =>
        val parityStatus = if summary.parityPass then "PASS" else "FAIL"
        val speedStatus =
          if config.minSpeedupVsReference <= 0.0 then "n/a"
          else if summary.speedPass then "PASS"
          else "FAIL"
        println(
          f"parity(${summary.mode} vs ${config.referenceMode}): " +
            f"l1=${summary.parity.l1}%.8f maxAbs=${summary.parity.maxAbs}%.8f " +
            s"checked=${summary.parity.checked} parityGate=$parityStatus " +
            f"speedup=${summary.speedupVsReference}%.3fx speedGate=$speedStatus"
        )
      }

    private def printGateResult(outcome: GateOutcome): Unit =
      println()
      println(s"gate=${if outcome.pass then "PASS" else "FAIL"}")
      if !outcome.pass then
        println(s"gateFailures=${outcome.failures.mkString(",")}")

  def main(args: Array[String]): Unit =
    val config = parseArgs(args.toVector)
    validateConfig(config)
    sys.exit(new BenchmarkRunner(config).run())

  private def validateConfig(config: Config): Unit =
    require(config.warmupRuns >= 0, "warmupRuns must be non-negative")
    require(config.measureRuns > 0, "measureRuns must be positive")
    require(config.hypothesisCount > 0, "hypothesisCount must be positive")
    require(config.maxL1Diff >= 0.0 && config.maxL1Diff.isFinite, "maxL1Diff must be finite and non-negative")
    require(config.maxAbsDiff >= 0.0 && config.maxAbsDiff.isFinite, "maxAbsDiff must be finite and non-negative")
    require(
      config.minSpeedupVsReference >= 0.0 && config.minSpeedupVsReference.isFinite,
      "minSpeedupVsReference must be finite and non-negative"
    )
    require(config.modes.nonEmpty, "modes must be non-empty")
    require(config.modes.forall(isKnownMode), "modes contain unknown provider token")
    require(config.modes.contains(config.referenceMode), "referenceMode must be included in modes")

  private def runOne(
      provider: HoldemDdreProvider.Provider,
      spot: BenchmarkSpot
  ): Either[String, DiscreteDistribution[HoleCards]] =
    HoldemDdreProvider
      .inferPosterior(
        prior = spot.prior,
        observations = spot.observations,
        actionModel = spot.actionModel,
        hero = spot.hero,
        board = spot.board,
        config = HoldemDdreProvider.Config(
          mode = HoldemDdreProvider.Mode.BlendPrimary,
          provider = provider,
          alpha = 1.0,
          minEntropyBits = 0.0,
          timeoutMillis = 0
        )
      )
      .map(_.posterior)
      .left
      .map(failure => s"${failure.reasonCategory}:${failure.detail}")

  private def checkAvailability(mode: String): Either[String, Unit] =
    mode match
      case "synthetic" => Right(())
      case "native-cpu" =>
        val availability = HoldemDdreNativeRuntime.availability(HoldemDdreNativeRuntime.Backend.Cpu)
        if availability.available then Right(()) else Left(availability.detail)
      case "native-gpu" =>
        val availability = HoldemDdreNativeRuntime.availability(HoldemDdreNativeRuntime.Backend.Gpu)
        if availability.available then Right(()) else Left(availability.detail)
      case "onnx" =>
        HoldemDdreOnnxRuntime.configuredConfig().map(_ => ())
      case other =>
        Left(s"unknown mode '$other'")

  private def modePropertyUpdates(config: Config): Seq[(String, Option[String])] =
    Seq(
      DdreNativeCpuPathProperty -> config.nativeCpuPath,
      DdreNativeGpuPathProperty -> config.nativeGpuPath,
      DdreOnnxModelPathProperty -> config.onnxModelPath,
      DdreOnnxArtifactDirProperty -> config.onnxArtifactDir,
      DdreOnnxAllowExperimentalProperty -> Some(config.onnxAllowExperimental.toString)
    )

  private def providerForMode(mode: String): Option[HoldemDdreProvider.Provider] =
    mode match
      case "synthetic" => Some(HoldemDdreProvider.Provider.Synthetic)
      case "native-cpu" => Some(HoldemDdreProvider.Provider.NativeCpu)
      case "native-gpu" => Some(HoldemDdreProvider.Provider.NativeGpu)
      case "onnx" => Some(HoldemDdreProvider.Provider.Onnx)
      case _ => None

  private def isKnownMode(mode: String): Boolean =
    providerForMode(mode).nonEmpty

  private def benchmarkSpot(hypothesisCount: Int): BenchmarkSpot =
    val hero = hole("Ac", "Kd")
    val board = Board.from(Seq(card("7h"), card("9c"), card("2d")))
    val dead = hero.asSet ++ board.asSet

    val baseWeights =
      TableRanges
        .defaults(TableFormat.NineMax)
        .rangeFor(Position.BigBlind)
        .weights
        .toVector
        .collect {
          case (hand, weight)
              if weight > 0.0 &&
                !dead.contains(hand.first) &&
                !dead.contains(hand.second) =>
            hand -> weight
        }
        .sortBy { case (_, weight) => -weight }
        .take(hypothesisCount)

    require(baseWeights.nonEmpty, "benchmark spot prior must be non-empty")
    val prior = DiscreteDistribution(baseWeights.toMap).normalized

    val state = GameState(
      street = Street.Flop,
      board = board,
      pot = 18.0,
      toCall = 6.0,
      position = Position.BigBlind,
      stackSize = 82.0,
      betHistory = Vector.empty
    )
    val observations = Vector(
      PokerAction.Raise(18.0) -> state,
      PokerAction.Call -> state.copy(pot = 30.0, toCall = 8.0)
    )

    BenchmarkSpot(
      hero = hero,
      board = board,
      prior = prior,
      observations = observations,
      actionModel = PokerActionModel.uniform
    )

  private def computeParity(
      reference: DiscreteDistribution[HoleCards],
      candidate: DiscreteDistribution[HoleCards]
  ): ParitySummary =
    val allHands = reference.weights.keySet ++ candidate.weights.keySet
    var checked = 0
    var l1 = 0.0
    var maxAbs = 0.0
    allHands.foreach { hand =>
      val delta = math.abs(reference.probabilityOf(hand) - candidate.probabilityOf(hand))
      l1 += delta
      if delta > maxAbs then maxAbs = delta
      checked += 1
    }
    ParitySummary(checked = checked, l1 = l1, maxAbs = maxAbs)

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

  private def parseArgs(args: Vector[String]): Config =
    val options = CliHelpers.requireOptions(args)
    CliHelpers.requireNoUnknownOptions(options, AllowedOptionKeys)

    val modes = CliHelpers
      .requireCsvTokens(options.getOrElse("modes", "synthetic,native-cpu,native-gpu"), "modes")
      .map(_.toLowerCase)
      .distinct

    Config(
      warmupRuns = CliHelpers.requireIntOption(options, "warmupRuns", 1),
      measureRuns = CliHelpers.requireIntOption(options, "measureRuns", 6),
      hypothesisCount = CliHelpers.requireIntOption(options, "hypothesisCount", 256),
      seed = CliHelpers.requireLongOption(options, "seed", 29L),
      modes = modes,
      referenceMode = options.getOrElse("referenceMode", "synthetic").trim.toLowerCase,
      maxL1Diff = CliHelpers.requireDoubleOption(options, "maxL1Diff", 1e-6),
      maxAbsDiff = CliHelpers.requireDoubleOption(options, "maxAbsDiff", 1e-7),
      minSpeedupVsReference = CliHelpers.requireDoubleOption(options, "minSpeedupVsReference", 0.0),
      requireAllModes = CliHelpers.requireBooleanOption(options, "requireAllModes", false),
      nativeCpuPath = options.get("nativeCpuPath"),
      nativeGpuPath = options.get("nativeGpuPath"),
      onnxModelPath = options.get("onnxModelPath"),
      onnxArtifactDir = options.get("onnxArtifactDir"),
      onnxAllowExperimental = CliHelpers.requireBooleanOption(options, "onnxAllowExperimental", false)
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
      HoldemDdreNativeRuntime.resetLoadCacheForTests()
