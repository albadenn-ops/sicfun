package sicfun.holdem

import sicfun.core.{DiscreteDistribution, Metrics, Probability}

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
            syntheticPosterior(prior, observations, actionModel)

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
    val hypotheses = prior.weights.keysIterator.toVector.sortBy(_.toToken)
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

  private def applyLegalMask(
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
      if !hand.asSet.exists(dead.contains) && probability > 0.0 then
        weights += hand -> probability
        total += probability
      idx += 1
    if total <= Eps then Left("zero_legal_mass_after_mask")
    else Right(DiscreteDistribution(weights.result()).normalized)

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
