package sicfun.holdem.provider
import sicfun.holdem.types.*
import sicfun.holdem.model.*
import sicfun.holdem.gpu.*

import sicfun.core.{DiscreteDistribution, Metrics, Probability}

import java.util.concurrent.atomic.AtomicBoolean
import scala.util.control.NonFatal

/** Phase-1 DDRE provider facade.
  *
  * Default behavior is fully disabled (`mode=off`, `provider=disabled`), so the
  * existing Bayesian path remains unchanged until DDRE is explicitly enabled.
  */
private[holdem] object HoldemDdreProvider:
  enum Mode:
    case Off
    case Shadow
    case BlendCanary
    case BlendPrimary

  enum Provider:
    case Disabled
    case Synthetic
    case NativeCpu
    case NativeGpu
    case Onnx

  final case class Config(
      mode: Mode,
      provider: Provider,
      alpha: Double,
      minEntropyBits: Double,
      timeoutMillis: Int
  ):
    require(alpha >= 0.0 && alpha <= 1.0 && alpha.isFinite, "alpha must be finite and in [0, 1]")
    require(minEntropyBits >= 0.0 && minEntropyBits.isFinite, "minEntropyBits must be finite and non-negative")
    require(timeoutMillis >= 0, "timeoutMillis must be non-negative")

  final case class InferenceResult(
      posterior: DiscreteDistribution[HoleCards],
      provider: Provider,
      latencyMillis: Long
  )

  final case class InferenceFailure(
      reasonCategory: String,
      detail: String,
      latencyMillis: Long
  )

  private val ModeProperty = "sicfun.ddre.mode"
  private val ModeEnv = "sicfun_DDRE_MODE"
  private val ProviderProperty = "sicfun.ddre.provider"
  private val ProviderEnv = "sicfun_DDRE_PROVIDER"
  private val AlphaProperty = "sicfun.ddre.alpha"
  private val AlphaEnv = "sicfun_DDRE_ALPHA"
  private val MinEntropyBitsProperty = "sicfun.ddre.minEntropyBits"
  private val MinEntropyBitsEnv = "sicfun_DDRE_MIN_ENTROPY_BITS"
  private val TimeoutMillisProperty = "sicfun.ddre.timeoutMillis"
  private val TimeoutMillisEnv = "sicfun_DDRE_TIMEOUT_MILLIS"

  private val DefaultAlpha = 0.20
  private val DefaultMinEntropyBits = 0.0
  private val SyntheticWarningEmitted = new AtomicBoolean(false)

  private inline val Eps = Probability.Eps
  private inline val MinLikelihood = 1e-6

  def modeLabel(mode: Mode): String =
    mode match
      case Mode.Off => "off"
      case Mode.Shadow => "shadow"
      case Mode.BlendCanary => "blend-canary"
      case Mode.BlendPrimary => "blend-primary"

  def providerLabel(provider: Provider): String =
    provider match
      case Provider.Disabled => "disabled"
      case Provider.Synthetic => "synthetic"
      case Provider.NativeCpu => "native-cpu"
      case Provider.NativeGpu => "native-gpu"
      case Provider.Onnx => "onnx"

  def configuredConfig(): Config =
    Config(
      mode = configuredMode,
      provider = configuredProvider,
      alpha = configuredAlpha,
      minEntropyBits = configuredMinEntropyBits,
      timeoutMillis = configuredTimeoutMillis
    )

  def inferPosterior(
      prior: DiscreteDistribution[HoleCards],
      observations: Seq[(PokerAction, GameState)],
      actionModel: PokerActionModel,
      hero: HoleCards,
      board: Board,
      config: Config = configuredConfig()
  ): Either[InferenceFailure, InferenceResult] =
    val startedAt = System.nanoTime()
    try
      val rawPosterior =
        config.provider match
          case Provider.Disabled =>
            return Left(
              InferenceFailure(
                reasonCategory = "provider_disabled",
                detail = "ddre provider disabled",
                latencyMillis = elapsedMillis(startedAt)
              )
            )
          case Provider.Synthetic =>
            warnSyntheticProviderOnce()
            syntheticPosterior(prior, observations, actionModel)
          case Provider.NativeCpu | Provider.NativeGpu =>
            nativePosterior(
              provider = config.provider,
              prior = prior,
              observations = observations,
              actionModel = actionModel
            ) match
              case Right(value) => value
              case Left(detail) =>
                return Left(
                  InferenceFailure(
                    reasonCategory = "native_error",
                    detail = detail,
                    latencyMillis = elapsedMillis(startedAt)
                  )
                )
          case Provider.Onnx =>
            HoldemDdreOnnxRuntime.configuredConfig() match
              case Left(detail) =>
                return Left(
                  InferenceFailure(
                    reasonCategory = "onnx_error",
                    detail = detail,
                    latencyMillis = elapsedMillis(startedAt)
                  )
                )
              case Right(onnxConfig) =>
                if isDecisionDrivingMode(config.mode) &&
                  !onnxConfig.decisionDrivingAllowed &&
                  !onnxConfig.allowExperimental
                then
                  val artifactLabel =
                    onnxConfig.artifactId
                      .orElse(onnxConfig.artifactDir.map(_.toString))
                      .getOrElse(onnxConfig.modelPath)
                  return Left(
                    InferenceFailure(
                      reasonCategory = "artifact_not_validated",
                      detail =
                        s"onnx artifact '$artifactLabel' is not validated for decision driving (status=${onnxConfig.validationStatus})",
                      latencyMillis = elapsedMillis(startedAt)
                    )
                  )
                onnxPosterior(
                  prior = prior,
                  observations = observations,
                  actionModel = actionModel,
                  onnxConfig = onnxConfig
                ) match
                  case Right(value) => value
                  case Left(detail) =>
                    return Left(
                      InferenceFailure(
                        reasonCategory = "onnx_error",
                        detail = detail,
                        latencyMillis = elapsedMillis(startedAt)
                      )
                    )

      val masked = applyLegalMask(rawPosterior, hero, board) match
        case Right(value) => value
        case Left(detail) =>
          return Left(
            InferenceFailure(
              reasonCategory = "invalid_output",
              detail = detail,
              latencyMillis = elapsedMillis(startedAt)
            )
          )

      val entropyBits = Metrics.entropy(masked.weights.values)
      if entropyBits + Eps < config.minEntropyBits then
        Left(
          InferenceFailure(
            reasonCategory = "entropy_guard",
            detail =
              f"entropy_below_min entropyBits=$entropyBits%.6f minEntropyBits=${config.minEntropyBits}%.6f",
            latencyMillis = elapsedMillis(startedAt)
          )
        )
      else
        val latency = elapsedMillis(startedAt)
        if config.timeoutMillis > 0 && latency > config.timeoutMillis.toLong then
          Left(
            InferenceFailure(
              reasonCategory = "timeout",
              detail = s"ddre inference exceeded timeoutMillis=${config.timeoutMillis}",
              latencyMillis = latency
            )
          )
        else
          Right(
            InferenceResult(
              posterior = masked,
              provider = config.provider,
              latencyMillis = latency
            )
          )
    catch
      case NonFatal(ex) =>
        Left(
          InferenceFailure(
            reasonCategory = "exception",
            detail = Option(ex.getMessage).map(_.trim).filter(_.nonEmpty).getOrElse(ex.getClass.getSimpleName),
            latencyMillis = elapsedMillis(startedAt)
          )
        )

  private def syntheticPosterior(
      prior: DiscreteDistribution[HoleCards],
      observations: Seq[(PokerAction, GameState)],
      actionModel: PokerActionModel
  ): DiscreteDistribution[HoleCards] =
    val hypotheses = prior.weights.keysIterator.toVector
    require(hypotheses.nonEmpty, "prior support must be non-empty")
    val obsCount = math.max(1, observations.length)
    val invObsCount = 1.0 / obsCount.toDouble
    val weights = Map.newBuilder[HoleCards, Double]
    var idx = 0
    while idx < hypotheses.length do
      val hand = hypotheses(idx)
      var score = math.sqrt(math.max(prior.probabilityOf(hand), MinLikelihood))
      var obsIdx = 0
      while obsIdx < observations.length do
        val (action, state) = observations(obsIdx)
        val likelihood = math.max(MinLikelihood, actionModel.likelihood(action, state, hand))
        score *= math.pow(likelihood, invObsCount)
        obsIdx += 1
      weights += hand -> math.max(score, MinLikelihood)
      idx += 1
    DiscreteDistribution(weights.result()).normalized

  private def warnSyntheticProviderOnce(): Unit =
    if SyntheticWarningEmitted.compareAndSet(false, true) then
      GpuRuntimeSupport.warn(
        "DDRE provider 'synthetic' is a heuristic scaffold, not a trained diffusion model; use it only for plumbing or fallback checks"
      )

  private def nativePosterior(
      provider: Provider,
      prior: DiscreteDistribution[HoleCards],
      observations: Seq[(PokerAction, GameState)],
      actionModel: PokerActionModel
  ): Either[String, DiscreteDistribution[HoleCards]] =
    val hypotheses = prior.weights.keysIterator.toVector
    if hypotheses.isEmpty then Left("prior_support_empty")
    else
      val priorArray = hypotheses.map(prior.probabilityOf).toArray
      val likelihoods = buildLikelihoodMatrix(observations, actionModel, hypotheses)
      val backendEither: Either[String, HoldemDdreNativeRuntime.Backend] =
        provider match
          case Provider.NativeCpu => Right(HoldemDdreNativeRuntime.Backend.Cpu)
          case Provider.NativeGpu => Right(HoldemDdreNativeRuntime.Backend.Gpu)
          case other => Left(s"unsupported_native_provider_${providerLabel(other)}")

      backendEither.flatMap { backend =>
        HoldemDdreNativeRuntime
          .inferPosterior(
            backend = backend,
            observationCount = observations.length,
            hypothesisCount = hypotheses.length,
            prior = priorArray,
            likelihoods = likelihoods
          )
          .flatMap { native =>
            distributionFromPosteriorArray(
              hypotheses = hypotheses,
              posterior = native.posterior,
              label = "native"
            )
          }
      }

  private def onnxPosterior(
      prior: DiscreteDistribution[HoleCards],
      observations: Seq[(PokerAction, GameState)],
      actionModel: PokerActionModel,
      onnxConfig: HoldemDdreOnnxRuntime.Config
  ): Either[String, DiscreteDistribution[HoleCards]] =
    val hypotheses = prior.weights.keysIterator.toVector
    if hypotheses.isEmpty then Left("prior_support_empty")
    else
      val priorArray = hypotheses.map(prior.probabilityOf).toArray
      val likelihoods = buildLikelihoodMatrix(observations, actionModel, hypotheses)
      HoldemDdreOnnxRuntime
        .inferPosterior(
          prior = priorArray,
          likelihoods = likelihoods,
          observationCount = observations.length,
          hypothesisCount = hypotheses.length,
          config = onnxConfig
        )
        .flatMap { posteriorRaw =>
          distributionFromPosteriorArray(
            hypotheses = hypotheses,
            posterior = posteriorRaw,
            label = "onnx"
          )
        }

  private[holdem] def buildLikelihoodMatrix(
      observations: Seq[(PokerAction, GameState)],
      actionModel: PokerActionModel,
      hypotheses: Vector[HoleCards]
  ): Array[Double] =
    if observations.isEmpty then Array.emptyDoubleArray
    else
      val observationCount = observations.length
      val hypothesisCount = hypotheses.length
      val matrix = new Array[Double](observationCount * hypothesisCount)
      var obsIdx = 0
      while obsIdx < observationCount do
        val (action, state) = observations(obsIdx)
        val rowOffset = obsIdx * hypothesisCount
        var hypIdx = 0
        while hypIdx < hypothesisCount do
          val likelihoodRaw = actionModel.likelihood(action, state, hypotheses(hypIdx))
          val likelihood =
            if likelihoodRaw.isFinite && likelihoodRaw > 0.0 then likelihoodRaw
            else MinLikelihood
          matrix(rowOffset + hypIdx) = likelihood
          hypIdx += 1
        obsIdx += 1
      matrix

  private[holdem] def distributionFromPosteriorArray(
      hypotheses: Vector[HoleCards],
      posterior: Array[Double],
      label: String
  ): Either[String, DiscreteDistribution[HoleCards]] =
    if posterior.length != hypotheses.length then
      Left(s"${label}_posterior_length_mismatch_expected_${hypotheses.length}_got_${posterior.length}")
    else
      val weights = Map.newBuilder[HoleCards, Double]
      var total = 0.0
      var idx = 0
      var invalidReason: String | Null = null
      while idx < hypotheses.length && invalidReason == null do
        val probability = posterior(idx)
        if !probability.isFinite || probability < 0.0 then
          invalidReason = s"${label}_invalid_probability_at_index_$idx"
        else if probability > 0.0 then
          weights += hypotheses(idx) -> probability
          total += probability
        idx += 1
      if invalidReason != null then Left(invalidReason.nn)
      else if total <= Eps then Left(s"${label}_zero_mass")
      else Right(DiscreteDistribution(weights.result()).normalized)

  private[holdem] def applyLegalMask(
      posterior: DiscreteDistribution[HoleCards],
      hero: HoleCards,
      board: Board
  ): Either[String, DiscreteDistribution[HoleCards]] =
    val dead = hero.asSet ++ board.asSet
    val entries = posterior.weights.toVector
    val weights = Map.newBuilder[HoleCards, Double]
    var idx = 0
    var total = 0.0
    while idx < entries.length do
      val (hand, probability) = entries(idx)
      if !probability.isFinite || probability < 0.0 then
        return Left(s"invalid_probability_for_${hand.toToken}")
      if !dead.contains(hand.first) &&
        !dead.contains(hand.second) &&
        probability > 0.0
      then
        weights += hand -> probability
        total += probability
      idx += 1
    if total <= Eps then Left("zero_legal_mass_after_mask")
    else Right(DiscreteDistribution(weights.result()).normalized)

  private def isDecisionDrivingMode(mode: Mode): Boolean =
    mode match
      case Mode.BlendCanary | Mode.BlendPrimary => true
      case _ => false

  private def configuredMode: Mode =
    GpuRuntimeSupport.resolveNonEmptyLower(ModeProperty, ModeEnv) match
      case Some("off" | "disabled" | "none") => Mode.Off
      case Some("shadow") => Mode.Shadow
      case Some("blend-canary" | "canary") => Mode.BlendCanary
      case Some("blend-primary" | "primary" | "blend") => Mode.BlendPrimary
      case Some(other) =>
        GpuRuntimeSupport.warn(s"unknown DDRE mode '$other'; using off")
        Mode.Off
      case None => Mode.Off

  private def configuredProvider: Provider =
    GpuRuntimeSupport.resolveNonEmptyLower(ProviderProperty, ProviderEnv) match
      case Some("disabled" | "off" | "none") => Provider.Disabled
      case Some("synthetic" | "mock" | "heuristic") => Provider.Synthetic
      case Some("native-cpu" | "native_cpu" | "cpu") => Provider.NativeCpu
      case Some("native-gpu" | "native_gpu" | "gpu" | "cuda") => Provider.NativeGpu
      case Some("onnx" | "onnx-runtime" | "onnxruntime") => Provider.Onnx
      case Some(other) =>
        GpuRuntimeSupport.warn(s"unknown DDRE provider '$other'; using disabled")
        Provider.Disabled
      case None => Provider.Disabled

  private def configuredAlpha: Double =
    GpuRuntimeSupport
      .resolveNonEmpty(AlphaProperty, AlphaEnv)
      .flatMap(_.toDoubleOption)
      .filter(value => value.isFinite && value >= 0.0 && value <= 1.0)
      .getOrElse(DefaultAlpha)

  private def configuredMinEntropyBits: Double =
    GpuRuntimeSupport
      .resolveNonEmpty(MinEntropyBitsProperty, MinEntropyBitsEnv)
      .flatMap(_.toDoubleOption)
      .filter(value => value.isFinite && value >= 0.0)
      .getOrElse(DefaultMinEntropyBits)

  private def configuredTimeoutMillis: Int =
    GpuRuntimeSupport
      .resolveNonEmpty(TimeoutMillisProperty, TimeoutMillisEnv)
      .flatMap(_.toIntOption)
      .filter(_ >= 0)
      .getOrElse(0)

  private def elapsedMillis(startedAt: Long): Long =
    math.max(0L, (System.nanoTime() - startedAt) / 1_000_000L)
