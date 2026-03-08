package sicfun.holdem

import sicfun.core.{Card, DiscreteDistribution}

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

  def main(args: Array[String]): Unit =
    val config = parseArgs(args.toVector)
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

    val spot = benchmarkSpot(config.hypothesisCount)
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

    val modeRuns = config.modes.map { mode =>
      runMode(mode, config, spot)
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

    val gateFailures = mutable.ArrayBuffer.empty[String]
    if config.requireAllModes then
      modeRuns.foreach { run =>
        if run.stats.isEmpty || run.posterior.isEmpty then
          gateFailures += s"mode_${run.mode}_unavailable"
      }

    val referenceRunOpt =
      modeRuns.find(run =>
        run.mode == config.referenceMode &&
          run.stats.nonEmpty &&
          run.posterior.nonEmpty
      )

    referenceRunOpt match
      case None =>
        gateFailures += s"reference_mode_${config.referenceMode}_unavailable"
      case Some(referenceRun) =>
        val referenceStats = referenceRun.stats.getOrElse(
          throw new IllegalStateException("reference stats unexpectedly missing")
        )
        val referencePosterior = referenceRun.posterior.getOrElse(
          throw new IllegalStateException("reference posterior unexpectedly missing")
        )

        val comparisons = modeRuns.flatMap { candidate =>
          if candidate.mode == config.referenceMode then None
          else
            (candidate.stats, candidate.posterior) match
              case (Some(stats), Some(posterior)) =>
                val parity = computeParity(referencePosterior, posterior)
                val speedup = referenceStats.meanMs / stats.meanMs
                val parityPass =
                  parity.l1 <= config.maxL1Diff &&
                    parity.maxAbs <= config.maxAbsDiff
                val speedPass =
                  config.minSpeedupVsReference <= 0.0 ||
                    speedup + 1e-12 >= config.minSpeedupVsReference
                Some(
                  ComparisonSummary(
                    mode = candidate.mode,
                    parity = parity,
                    speedupVsReference = speedup,
                    parityPass = parityPass,
                    speedPass = speedPass
                  )
                )
              case _ => None
        }

        if comparisons.isEmpty then
          gateFailures += "no_candidate_modes_available_for_comparison"
        else
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

            if !summary.parityPass then
              gateFailures += s"parity_${summary.mode}_failed"
            if !summary.speedPass then
              gateFailures += s"speed_${summary.mode}_failed"
          }

    val gatePass = gateFailures.isEmpty
    println()
    println(s"gate=${if gatePass then "PASS" else "FAIL"}")
    if !gatePass then
      println(s"gateFailures=${gateFailures.mkString(",")}")

    if gatePass then sys.exit(0) else sys.exit(2)

  private def runMode(
      mode: String,
      config: Config,
      spot: BenchmarkSpot
  ): ModeRun =
    val provider = providerForMode(mode)
    provider match
      case None =>
        ModeRun(mode = mode, stats = None, posterior = None, detail = "unknown mode")
      case Some(value) =>
        withSystemProperties(modePropertyUpdates(config)) {
          HoldemDdreNativeRuntime.resetLoadCacheForTests()
          val availability = checkAvailability(mode)
          availability match
            case Left(reason) =>
              ModeRun(mode = mode, stats = None, posterior = None, detail = reason)
            case Right(_) =>
              runMeasuredMode(mode, value, config, spot)
        }

  private def runMeasuredMode(
      mode: String,
      provider: HoldemDdreProvider.Provider,
      config: Config,
      spot: BenchmarkSpot
  ): ModeRun =
    var warmup = 0
    while warmup < config.warmupRuns do
      runOne(provider, spot) match
        case Left(reason) =>
          return ModeRun(mode = mode, stats = None, posterior = None, detail = s"warmup_failed:$reason")
        case Right(_) =>
          ()
      warmup += 1

    val samplesMs = new Array[Double](config.measureRuns)
    var firstPosterior: Option[DiscreteDistribution[HoleCards]] = None
    var i = 0
    while i < config.measureRuns do
      val started = System.nanoTime()
      runOne(provider, spot) match
        case Left(reason) =>
          return ModeRun(mode = mode, stats = None, posterior = None, detail = s"measure_failed:$reason")
        case Right(posterior) =>
          if firstPosterior.isEmpty then firstPosterior = Some(posterior)
      val elapsed = math.max(1L, System.nanoTime() - started)
      samplesMs(i) = elapsed.toDouble / 1_000_000.0
      i += 1

    ModeRun(
      mode = mode,
      stats = Some(computeStats(samplesMs.toVector)),
      posterior = firstPosterior,
      detail = ""
    )

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
    val options = args.flatMap { token =>
      token.split("=", 2) match
        case Array(key, value) if key.startsWith("--") && value.nonEmpty =>
          Some(key.drop(2) -> value)
        case _ =>
          None
    }.toMap

    val modes = options
      .getOrElse("modes", "synthetic,native-cpu,native-gpu")
      .split(',')
      .toVector
      .map(_.trim.toLowerCase)
      .filter(_.nonEmpty)
      .distinct

    Config(
      warmupRuns = options.get("warmupRuns").flatMap(_.toIntOption).getOrElse(1),
      measureRuns = options.get("measureRuns").flatMap(_.toIntOption).getOrElse(6),
      hypothesisCount = options.get("hypothesisCount").flatMap(_.toIntOption).getOrElse(256),
      seed = options.get("seed").flatMap(_.toLongOption).getOrElse(29L),
      modes = modes,
      referenceMode = options.getOrElse("referenceMode", "synthetic").trim.toLowerCase,
      maxL1Diff = options.get("maxL1Diff").flatMap(_.toDoubleOption).getOrElse(1e-6),
      maxAbsDiff = options.get("maxAbsDiff").flatMap(_.toDoubleOption).getOrElse(1e-7),
      minSpeedupVsReference = options.get("minSpeedupVsReference").flatMap(_.toDoubleOption).getOrElse(0.0),
      requireAllModes = options.get("requireAllModes").exists(parseBoolean),
      nativeCpuPath = options.get("nativeCpuPath"),
      nativeGpuPath = options.get("nativeGpuPath"),
      onnxModelPath = options.get("onnxModelPath"),
      onnxArtifactDir = options.get("onnxArtifactDir"),
      onnxAllowExperimental = options.get("onnxAllowExperimental").exists(parseBoolean)
    )

  private def parseBoolean(raw: String): Boolean =
    raw.trim.toLowerCase match
      case "1" | "true" | "yes" | "on" => true
      case _ => false

  private def card(token: String): Card =
    Card.parse(token).getOrElse(throw new IllegalArgumentException(s"invalid card token: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

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
