package sicfun.holdem.engine
import sicfun.holdem.history.ShowdownRecord
import sicfun.holdem.types.*
import sicfun.holdem.gpu.*
import sicfun.holdem.model.*
import sicfun.holdem.equity.*
import sicfun.holdem.provider.*

import sicfun.core.{Card, CardId, CollapseMetrics, DiscreteDistribution, Probability}

import java.util.concurrent.ConcurrentHashMap

import scala.collection.mutable
import scala.util.Random
import scala.util.hashing.MurmurHash3

/** Observed villain action with the corresponding public game state. */
final case class VillainObservation(action: PokerAction, state: GameState)

/** Collapse summary for prior -> posterior range inference. */
final case class PosteriorCollapse(
    entropyReduction: Double,
    klDivergence: Double,
    effectiveSupportPrior: Double,
    effectiveSupportPosterior: Double,
    collapseRatio: Double
)

/** Result of range inference after bunching + Bayesian action updates. */
final class PosteriorInferenceResult private (
    val prior: DiscreteDistribution[HoleCards],
    val posterior: DiscreteDistribution[HoleCards],
    val compact: Option[HoldemEquity.CompactPosterior],
    val logEvidence: Double,
    collapseThunk: () => PosteriorCollapse
):
  // Most decision callers only need the posterior; defer collapse diagnostics until read.
  lazy val collapse: PosteriorCollapse = collapseThunk()

object PosteriorInferenceResult:
  def apply(
      prior: DiscreteDistribution[HoleCards],
      posterior: DiscreteDistribution[HoleCards],
      compact: Option[HoldemEquity.CompactPosterior],
      logEvidence: Double,
      collapse: => PosteriorCollapse
  ): PosteriorInferenceResult =
    new PosteriorInferenceResult(prior, posterior, compact, logEvidence, () => collapse)

/** EV evaluation of a candidate hero action against an inferred posterior. */
final case class ActionEvaluation(action: PokerAction, expectedValue: Double)

/** Best-response recommendation against an inferred posterior. */
final case class ActionRecommendation(
    heroEquity: EquityEstimate,
    actionEvaluations: Vector[ActionEvaluation],
    bestAction: PokerAction
)

/** End-to-end output: posterior inference and action recommendation. */
final case class InferenceDecisionResult(
    posteriorInference: PosteriorInferenceResult,
    recommendation: ActionRecommendation
)

/** Strategy for converting (action, state, equity) into a scalar EV. */
trait ActionValueModel:
  def expectedValue(action: PokerAction, state: GameState, heroEquity: Double): Double

object ActionValueModel:
  /** One-step chip-EV approximation.
    *
    * - Fold: EV = 0
    * - Check (when allowed): EV ~= equity * pot
    * - Call: uses [[HoldemEquity.evCall]]
    * - Raise(amount): optional fold-equity blend with single-street showdown EV
    */
  final case class ChipEv(raiseFoldProbability: Double = 0.0) extends ActionValueModel:
    require(
      raiseFoldProbability >= 0.0 && raiseFoldProbability <= 1.0,
      "raiseFoldProbability must be in [0, 1]"
    )

    def expectedValue(action: PokerAction, state: GameState, heroEquity: Double): Double =
      require(heroEquity >= 0.0 && heroEquity <= 1.0, "heroEquity must be in [0, 1]")
      action match
        case PokerAction.Fold => 0.0
        case PokerAction.Check =>
          if state.toCall <= 0.0 then heroEquity * state.pot else Double.NegativeInfinity
        case PokerAction.Call =>
          if state.toCall <= 0.0 then heroEquity * state.pot
          else HoldemEquity.evCall(state.pot, state.toCall, heroEquity)
        case PokerAction.Raise(amount) =>
          require(amount > 0.0, "raise amount must be positive")
          val showdownEv = (heroEquity * (state.pot + amount)) - amount
          (raiseFoldProbability * state.pot) + ((1.0 - raiseFoldProbability) * showdownEv)

/** Inference engine that combines:
  * 1) bunching-conditioned prior construction
  * 2) Bayesian posterior updates from observed villain actions
  * 3) action EV ranking against the inferred posterior
  *
  * ==Performance==
  * Posterior inference results are cached by observable context (hero, board,
  * folds, villain position, and observations) to avoid redundant bunching
  * Monte Carlo and Bayesian update computations when the same decision point
  * is evaluated multiple times (e.g., across candidate actions in
  * `inferAndRecommend`). The cache is bounded and automatically cleared
  * when it exceeds [[MaxPosteriorCacheSize]].
  */
object RangeInferenceEngine:
  private val MaxPosteriorCacheSize = 256
  private val MaxPriorCacheSize = 512
  private inline val ProfileEps = 1e-12
  private val EquityPosteriorMaxHandsProperty = "sicfun.range.equityPosterior.maxHands"
  private val EquityPosteriorMaxHandsEnv = "sicfun_RANGE_EQUITY_POSTERIOR_MAX_HANDS"
  private val EquityPosteriorMinMassProperty = "sicfun.range.equityPosterior.minMass"
  private val EquityPosteriorMinMassEnv = "sicfun_RANGE_EQUITY_POSTERIOR_MIN_MASS"
  private val DefaultEquityPosteriorMaxHands = 256
  private val DefaultEquityPosteriorMinMass = 0.995
  private val DefaultEquityPosteriorMinHands = 64

  private final case class AggregatedResponses(
      foldProbability: Double,
      continueProbability: Double,
      continueWeights: Map[HoleCards, Double],
      continuationMatchesPosterior: Boolean
  )

  /** Lightweight key for caching posterior inference results. */
  private case class PosteriorCacheKey(
      hero: HoleCards,
      board: Board,
      folds: Vector[PreflopFold],
      villainPos: Position,
      observations: Seq[VillainObservation],
      showdownHistoryHash: Int,
      ddreMode: HoldemDdreProvider.Mode,
      ddreProvider: HoldemDdreProvider.Provider,
      ddreAlpha: Double,
      ddreMinEntropyBits: Double,
      ddreTimeoutMillis: Int
  )

  private case class PriorCacheKey(
      hero: HoleCards,
      board: Board,
      folds: Vector[PreflopFold],
      villainPos: Position,
      tableRangesIdentity: Int,
      bunchingTrials: Int
  )

  private val posteriorCache =
    new ConcurrentHashMap[PosteriorCacheKey, PosteriorInferenceResult]()
  private val priorCache =
    new ConcurrentHashMap[PriorCacheKey, DiscreteDistribution[HoleCards]]()
  private val posteriorCacheMutationLock = new Object
  private val priorCacheMutationLock = new Object

  def clearPosteriorCache(): Unit =
    posteriorCache.clear()
    priorCache.clear()

  private[engine] def showdownHistoryHash(showdownHistory: Vector[ShowdownRecord]): Int =
    MurmurHash3.seqHash(showdownHistory.map(record => (record.handId, record.cards.toToken)))

  private def cachedOrCompute[K, V](
      cache: ConcurrentHashMap[K, V],
      key: K,
      maxSize: Int,
      mutationLock: Object
  )(compute: => V): V =
    val cached = cache.get(key)
    if cached != null then cached
    else
      val computed = compute
      mutationLock.synchronized {
        val existing = cache.get(key)
        if existing != null then existing
        else
          if cache.size() >= maxSize then cache.clear()
          cache.put(key, computed)
          computed
      }

  /** Computes a posterior villain range from folds + observed actions.
    *
    * Results are cached by `(hero, board, folds, villainPos, observations)` to
    * avoid redundant computation when called multiple times for the same context.
    * Pass `useCache = false` to bypass caching (e.g., when deterministic seeding matters).
    */
  def inferPosterior(
      hero: HoleCards,
      board: Board,
      folds: Vector[PreflopFold],
      tableRanges: TableRanges,
      villainPos: Position,
      observations: Seq[VillainObservation],
      actionModel: PokerActionModel,
      bunchingTrials: Int = 10_000,
      rng: Random = new Random(),
      useCache: Boolean = true,
      revealedCards: Option[HoleCards] = None,
      showdownHistory: Vector[ShowdownRecord] = Vector.empty
  ): PosteriorInferenceResult =
    revealedCards match
      case Some(cards) =>
        val delta = DiscreteDistribution(Map(cards -> 1.0))
        PosteriorInferenceResult(
          prior = delta,
          posterior = delta,
          compact = None,
          logEvidence = 0.0,
          collapse = PosteriorCollapse(
            entropyReduction = Double.PositiveInfinity,
            klDivergence = Double.PositiveInfinity,
            effectiveSupportPrior = 1326.0,
            effectiveSupportPosterior = 1.0,
            collapseRatio = 1.0 / 1326.0
          )
        )
      case None =>
        require(bunchingTrials > 0, "bunchingTrials must be positive")
        val ddreConfig = HoldemDdreProvider.configuredConfig()
        def computeResult: PosteriorInferenceResult =
          computePosterior(
            hero,
            board,
            folds,
            tableRanges,
            villainPos,
            observations,
            actionModel,
            bunchingTrials,
            rng,
            ddreConfig,
            showdownHistory
          )

        if useCache then
          val cacheKey = PosteriorCacheKey(
            hero = hero,
            board = board,
            folds = folds,
            villainPos = villainPos,
            observations = observations,
            showdownHistoryHash = showdownHistoryHash(showdownHistory),
            ddreMode = ddreConfig.mode,
            ddreProvider = ddreConfig.provider,
            ddreAlpha = ddreConfig.alpha,
            ddreMinEntropyBits = ddreConfig.minEntropyBits,
            ddreTimeoutMillis = ddreConfig.timeoutMillis
          )
          cachedOrCompute(
            cache = posteriorCache,
            key = cacheKey,
            maxSize = MaxPosteriorCacheSize,
            mutationLock = posteriorCacheMutationLock
          )(computeResult)
        else computeResult

  private def computePosterior(
      hero: HoleCards,
      board: Board,
      folds: Vector[PreflopFold],
      tableRanges: TableRanges,
      villainPos: Position,
      observations: Seq[VillainObservation],
      actionModel: PokerActionModel,
      bunchingTrials: Int,
      rng: Random,
      ddreConfig: HoldemDdreProvider.Config,
      showdownHistory: Vector[ShowdownRecord]
  ): PosteriorInferenceResult =
    // priorForContext always returns a normalized range.
    val normalizedPrior = priorForContext(
      hero = hero,
      board = board,
      folds = folds,
      tableRanges = tableRanges,
      villainPos = villainPos,
      bunchingTrials = bunchingTrials,
      rng = rng
    )
    val effectivePrior = ShowdownPriorBias.applyBias(
      prior = normalizedPrior,
      showdowns = showdownHistory,
      deadCards = hero.asSet ++ board.asSet
    )
    val observationsForBayes = observations.map(o => o.action -> o.state).toVector
    val canSkipBayesUpdate =
      observationsForBayes.nonEmpty &&
        actionModel.isEffectivelyUniform

    /** Applies DDRE policy for decision-time posterior selection.
      *
      * Modes:
      * - Off: pure Bayesian posterior.
      * - Shadow: compute DDRE for telemetry, still return Bayesian posterior.
      * - Blend*: validate and fuse Bayes/DDRE according to configured alpha.
      */
    def resolveDecisionPosteriorFor(
        bayesPosterior: DiscreteDistribution[HoleCards]
    ): DiscreteDistribution[HoleCards] =
      if ddreConfig.mode == HoldemDdreProvider.Mode.Off then bayesPosterior
      else
        resolveDecisionPosterior(
          hero = hero,
          board = board,
          prior = effectivePrior,
          bayesPosterior = bayesPosterior,
          observations = observationsForBayes,
          actionModel = actionModel,
          ddreConfig = ddreConfig
        )

    val (posterior, logEvidence, compact) =
      if observationsForBayes.isEmpty then
        (resolveDecisionPosteriorFor(effectivePrior), 0.0, None)
      else if canSkipBayesUpdate then
        (resolveDecisionPosteriorFor(effectivePrior), 0.0, None)
      else
        val bayesUpdate = HoldemBayesProvider.updatePosterior(
          prior = effectivePrior,
          observations = observationsForBayes,
          actionModel = actionModel
        )
        val usesCompact =
          ddreConfig.mode == HoldemDdreProvider.Mode.Off ||
            ddreConfig.mode == HoldemDdreProvider.Mode.Shadow
        val resolved = resolveDecisionPosteriorFor(bayesUpdate.posterior)
        val compactOption = if usesCompact then Some(bayesUpdate.compact) else None
        (resolved, bayesUpdate.logEvidence, compactOption)

    PosteriorInferenceResult(
      effectivePrior,
      posterior,
      compact,
      logEvidence,
      {
        val collapseSummary = CollapseMetrics.summary(effectivePrior, posterior)
        PosteriorCollapse(
          entropyReduction = collapseSummary.entropyReduction,
          klDivergence = collapseSummary.klDivergence,
          effectiveSupportPrior = collapseSummary.effectiveSupportPrior,
          effectiveSupportPosterior = collapseSummary.effectiveSupportPosterior,
          collapseRatio = collapseSummary.collapseRatio
        )
      }
    )

  private def priorForContext(
      hero: HoleCards,
      board: Board,
      folds: Vector[PreflopFold],
      tableRanges: TableRanges,
      villainPos: Position,
      bunchingTrials: Int,
      rng: Random
  ): DiscreteDistribution[HoleCards] =
    val key = PriorCacheKey(
      hero = hero,
      board = board,
      folds = folds,
      villainPos = villainPos,
      tableRangesIdentity = System.identityHashCode(tableRanges),
      bunchingTrials = bunchingTrials
    )
    cachedOrCompute(
      cache = priorCache,
      key = key,
      maxSize = MaxPriorCacheSize,
      mutationLock = priorCacheMutationLock
    ) {
      if bunchingTrials <= 1 || folds.isEmpty then
        naiveVillainRange(hero, board, tableRanges, villainPos)
      else
        BunchingEffect.adjustedRange(
          hero = hero,
          board = board,
          folds = folds,
          tableRanges = tableRanges,
          villainPos = villainPos,
          trials = bunchingTrials,
          rng = rng
        )
    }

  private def naiveVillainRange(
      hero: HoleCards,
      board: Board,
      tableRanges: TableRanges,
      villainPos: Position
  ): DiscreteDistribution[HoleCards] =
    val dead = hero.asSet ++ board.asSet
    val filtered = tableRanges.rangeFor(villainPos).weights.collect {
      case (hand, weight)
          if weight > 0.0 &&
            !dead.contains(hand.first) &&
            !dead.contains(hand.second) =>
        hand -> weight
    }
    require(filtered.nonEmpty, "villain range is empty after hero/board filtering")
    DiscreteDistribution(filtered).normalized

  /** Validates Bayes/DDRE candidates against legal support, then chooses/fuses based on mode.
    *
    * This method is fail-closed only when both sources are invalid in blend modes.
    * In all degradable paths it logs a `ddre_degraded` event and returns a safe fallback.
    */
  private def resolveDecisionPosterior(
      hero: HoleCards,
      board: Board,
      prior: DiscreteDistribution[HoleCards],
      bayesPosterior: DiscreteDistribution[HoleCards],
      observations: Seq[(PokerAction, GameState)],
      actionModel: PokerActionModel,
      ddreConfig: HoldemDdreProvider.Config
  ): DiscreteDistribution[HoleCards] =
    val support = prior.weights.keysIterator.map(hand => hand -> handMask(hand)).toVector
    val deadMask = handMask(hero) | cardsMask(board.cards)
    val bayesValidated = validatePosteriorForDecision(
      name = "bayes",
      distribution = bayesPosterior,
      support = support,
      deadMask = deadMask
    )
    def ddreValidated: Either[HoldemDdreProvider.InferenceFailure, DiscreteDistribution[HoleCards]] =
      HoldemDdreProvider
        .inferPosterior(
          prior = prior,
          observations = observations,
          actionModel = actionModel,
          hero = hero,
          board = board,
          config = ddreConfig
        )
        .flatMap { result =>
          validatePosteriorForDecision(
            name = "ddre",
            distribution = result.posterior,
            support = support,
            deadMask = deadMask
          ) match
            case Right(_) => Right(result.posterior)
            case Left(reason) =>
              Left(
                HoldemDdreProvider.InferenceFailure(
                  reasonCategory = "invalid_output",
                  detail = reason,
                  latencyMillis = result.latencyMillis
                )
              )
        }

    ddreConfig.mode match
      case HoldemDdreProvider.Mode.Off =>
        bayesValidated match
          case Right(_) => bayesPosterior
          case Left(reason) =>
            throw new IllegalStateException(s"Bayesian posterior invalid in DDRE off mode: $reason")

      case HoldemDdreProvider.Mode.Shadow =>
        ddreValidated.left.foreach { failure =>
          logDdreDegraded(
            mode = ddreConfig.mode,
            reasonCategory = failure.reasonCategory,
            detail = failure.detail,
            latencyMillis = failure.latencyMillis,
            bayesOk = bayesValidated.isRight,
            ddreOk = false,
            alphaApplied = 0.0
          )
        }
        bayesValidated match
          case Right(_) => bayesPosterior
          case Left(reason) =>
            throw new IllegalStateException(s"Bayesian posterior invalid in DDRE shadow mode: $reason")

      case HoldemDdreProvider.Mode.BlendCanary | HoldemDdreProvider.Mode.BlendPrimary =>
        val ddreCandidate = ddreValidated
        (bayesValidated, ddreCandidate) match
          case (Right(_), Right(ddre)) =>
            fusePosteriors(
              bayes = bayesPosterior,
              ddre = ddre,
              alpha = ddreConfig.alpha,
              support = support,
              deadMask = deadMask
            ) match
              case Right(fused) => fused
              case Left(reason) =>
                logDdreDegraded(
                  mode = ddreConfig.mode,
                  reasonCategory = "invalid_output",
                  detail = reason,
                  latencyMillis = 0L,
                  bayesOk = true,
                  ddreOk = true,
                  alphaApplied = 0.0
                )
                bayesPosterior

          case (Right(_), Left(failure)) =>
            logDdreDegraded(
              mode = ddreConfig.mode,
              reasonCategory = failure.reasonCategory,
              detail = failure.detail,
              latencyMillis = failure.latencyMillis,
              bayesOk = true,
              ddreOk = false,
              alphaApplied = 0.0
            )
            bayesPosterior

          case (Left(bayesReason), Right(ddre)) =>
            logDdreDegraded(
              mode = ddreConfig.mode,
              reasonCategory = "bayes_invalid",
              detail = bayesReason,
              latencyMillis = 0L,
              bayesOk = false,
              ddreOk = true,
              alphaApplied = 1.0
            )
            ddre

          case (Left(bayesReason), Left(ddreFailure)) =>
            logDdreDegraded(
              mode = ddreConfig.mode,
              reasonCategory = "fail_closed",
              detail = s"bayes=$bayesReason;ddre=${ddreFailure.reasonCategory}:${ddreFailure.detail}",
              latencyMillis = ddreFailure.latencyMillis,
              bayesOk = false,
              ddreOk = false,
              alphaApplied = 0.0
            )
            throw new IllegalStateException(
              s"Bayesian and DDRE posteriors are invalid (bayes=$bayesReason, ddre=${ddreFailure.reasonCategory})"
            )

  private def validatePosteriorForDecision(
      name: String,
      distribution: DiscreteDistribution[HoleCards],
      support: Vector[(HoleCards, Long)],
      deadMask: Long
  ): Either[String, Unit] =
    var idx = 0
    var legalMass = 0.0
    while idx < support.length do
      val (hand, mask) = support(idx)
      if (mask & deadMask) == 0L then
        val probability = distribution.probabilityOf(hand)
        if !probability.isFinite || probability < 0.0 then
          return Left(s"${name}_invalid_probability_for_${hand.toToken}")
        legalMass += probability
      idx += 1
    if legalMass <= Eps then Left(s"${name}_zero_legal_mass")
    else Right(())

  private def fusePosteriors(
      bayes: DiscreteDistribution[HoleCards],
      ddre: DiscreteDistribution[HoleCards],
      alpha: Double,
      support: Vector[(HoleCards, Long)],
      deadMask: Long
  ): Either[String, DiscreteDistribution[HoleCards]] =
    val clampedAlpha = math.max(0.0, math.min(1.0, alpha))
    if clampedAlpha <= Eps then return Right(bayes)
    if clampedAlpha >= 1.0 - Eps then return Right(ddre)
    val weights = Map.newBuilder[HoleCards, Double]
    var idx = 0
    var total = 0.0
    while idx < support.length do
      val (hand, mask) = support(idx)
      if (mask & deadMask) == 0L then
        val bayesP = bayes.probabilityOf(hand)
        val ddreP = ddre.probabilityOf(hand)
        if !bayesP.isFinite || !ddreP.isFinite || bayesP < 0.0 || ddreP < 0.0 then
          return Left(s"fused_invalid_probability_for_${hand.toToken}")
        val probability = ((1.0 - clampedAlpha) * bayesP) + (clampedAlpha * ddreP)
        if !probability.isFinite || probability < 0.0 then
          return Left(s"fused_invalid_probability_for_${hand.toToken}")
        if probability > 0.0 then
          weights += hand -> probability
          total += probability
      idx += 1
    if total <= Eps then Left("fused_zero_legal_mass")
    else Right(DiscreteDistribution(weights.result()).normalized)

  private def logDdreDegraded(
      mode: HoldemDdreProvider.Mode,
      reasonCategory: String,
      detail: String,
      latencyMillis: Long,
      bayesOk: Boolean,
      ddreOk: Boolean,
      alphaApplied: Double
  ): Unit =
    val alphaLabel = java.lang.String.format(java.util.Locale.ROOT, "%.6f", Double.box(alphaApplied))
    GpuRuntimeSupport.warn(
      s"ddre_degraded reason=${sanitizeLogToken(reasonCategory)} " +
        s"detail=${sanitizeLogToken(detail)} latencyMs=$latencyMillis " +
        s"mode=${HoldemDdreProvider.modeLabel(mode)} " +
        s"bayes_ok=$bayesOk ddre_ok=$ddreOk alpha_applied=$alphaLabel"
    )

  private def sanitizeLogToken(raw: String): String =
    raw.trim
      .replaceAll("\\s+", "_")
      .replaceAll("[^A-Za-z0-9_\\-.:=]", "_")

  private inline def handMask(hand: HoleCards): Long =
    cardMask(hand.first) | cardMask(hand.second)

  private def cardsMask(cards: Seq[Card]): Long =
    var mask = 0L
    var idx = 0
    while idx < cards.length do
      mask = mask | cardMask(cards(idx))
      idx += 1
    mask

  private inline def cardMask(card: Card): Long =
    1L << CardId.toId(card)

  /** Ranks candidate hero actions by EV vs a posterior villain range. */
  def recommendAction(
      hero: HoleCards,
      state: GameState,
      posterior: DiscreteDistribution[HoleCards],
      candidateActions: Vector[PokerAction],
      actionValueModel: ActionValueModel = ActionValueModel.ChipEv(),
      villainResponseModel: Option[VillainResponseModel] = None,
      equityTrials: Int = 50_000,
      rng: Random = new Random()
  ): ActionRecommendation =
    require(candidateActions.nonEmpty, "candidateActions must be non-empty")
    require(equityTrials > 0, "equityTrials must be positive")
    recommendActionAssumeNormalized(
      hero = hero,
      state = state,
      posterior = posterior.normalized,
      candidateActions = candidateActions,
      actionValueModel = actionValueModel,
      villainResponseModel = villainResponseModel,
      equityTrials = equityTrials,
      rng = rng
    )

  private[holdem] def recommendActionAssumeNormalized(
      hero: HoleCards,
      state: GameState,
      posterior: DiscreteDistribution[HoleCards],
      candidateActions: Vector[PokerAction],
      actionValueModel: ActionValueModel = ActionValueModel.ChipEv(),
      villainResponseModel: Option[VillainResponseModel] = None,
      equityTrials: Int = 50_000,
      rng: Random = new Random()
  ): ActionRecommendation =
    require(candidateActions.nonEmpty, "candidateActions must be non-empty")
    require(equityTrials > 0, "equityTrials must be positive")
    val equityPosterior = compactPosteriorForEquity(posterior)

    val heroEquity = HoldemEquity.equityMonteCarlo(
      hero = hero,
      board = state.board,
      villainRange = equityPosterior,
      trials = equityTrials,
      rng = rng
    )

    val evaluations = candidateActions.map { action =>
      val expectedValue =
        villainResponseModel match
          case Some(responseModel) =>
            action match
              case PokerAction.Raise(_) =>
                val raiseRng =
                  responseModel match
                    case _: UniformVillainResponseModel => rng
                    case _ => new Random(rng.nextLong())
                responseAwareRaiseEv(
                  hero = hero,
                  state = state,
                  posterior = equityPosterior,
                  heroEquityMean = heroEquity.mean,
                  action = action,
                  responseModel = responseModel,
                  equityTrials = equityTrials,
                  rng = raiseRng
                )
              case _ =>
                actionValueModel.expectedValue(action, state, heroEquity.mean)
          case None =>
            actionValueModel.expectedValue(action, state, heroEquity.mean)
      ActionEvaluation(
        action = action,
        expectedValue = expectedValue
      )
    }

    val best = evaluations.reduceLeft { (a, b) =>
      if a.expectedValue >= b.expectedValue then a else b
    }
    ActionRecommendation(heroEquity, evaluations, best.action)

  /** Variant that accepts an optional CompactPosterior for the equity fast path. */
  private[holdem] def recommendActionWithCompact(
      hero: HoleCards,
      state: GameState,
      posterior: DiscreteDistribution[HoleCards],
      compact: Option[HoldemEquity.CompactPosterior],
      candidateActions: Vector[PokerAction],
      actionValueModel: ActionValueModel = ActionValueModel.ChipEv(),
      villainResponseModel: Option[VillainResponseModel] = None,
      equityTrials: Int = 50_000,
      rng: Random = new Random()
  ): ActionRecommendation =
    require(candidateActions.nonEmpty, "candidateActions must be non-empty")
    require(equityTrials > 0, "equityTrials must be positive")

    val maxHands = configuredEquityPosteriorMaxHands
    val minMass = configuredEquityPosteriorMinMass
    val minHands = math.max(1, math.min(maxHands, DefaultEquityPosteriorMinHands))

    val heroEquity = compact match
      case Some(cp) =>
        val truncated = truncateCompact(cp, maxHands, minMass, minHands)
        HoldemEquity.equityMonteCarlo(hero, state.board, truncated, equityTrials, rng)
      case None =>
        val equityPosterior = compactPosteriorForEquity(posterior)
        HoldemEquity.equityMonteCarlo(hero, state.board, equityPosterior, equityTrials, rng)

    // For response-aware raise EV, fall back to DiscreteDistribution
    // (VillainResponseModel needs per-hand Map iteration)
    val equityPosteriorForResponse = compact match
      case Some(cp) => cp.distribution
      case None => compactPosteriorForEquity(posterior)

    val evaluations = candidateActions.map { action =>
      val expectedValue =
        villainResponseModel match
          case Some(responseModel) =>
            action match
              case PokerAction.Raise(_) =>
                val raiseRng =
                  responseModel match
                    case _: UniformVillainResponseModel => rng
                    case _ => new Random(rng.nextLong())
                responseAwareRaiseEv(
                  hero = hero,
                  state = state,
                  posterior = equityPosteriorForResponse,
                  heroEquityMean = heroEquity.mean,
                  action = action,
                  responseModel = responseModel,
                  equityTrials = equityTrials,
                  rng = raiseRng
                )
              case _ =>
                actionValueModel.expectedValue(action, state, heroEquity.mean)
          case None =>
            actionValueModel.expectedValue(action, state, heroEquity.mean)
      ActionEvaluation(action = action, expectedValue = expectedValue)
    }

    val best = evaluations.reduceLeft { (a, b) =>
      if a.expectedValue >= b.expectedValue then a else b
    }
    ActionRecommendation(heroEquity, evaluations, best.action)

  /** Truncates a CompactPosterior to top-k hands by weight.
    *
    * Retention guarantees:
    * - keep at most `maxHands`
    * - keep at least `minHands`
    * - continue keeping hands until cumulative mass reaches `minMass`
    */
  private def truncateCompact(
      cp: HoldemEquity.CompactPosterior,
      maxHands: Int,
      minMass: Double,
      minHands: Int
  ): HoldemEquity.CompactPosterior =
    if maxHands <= 0 || cp.size <= maxHands then cp
    else
      val indices = Array.tabulate(cp.size)(i => Integer.valueOf(i))
      java.util.Arrays.sort(indices, (a: Integer, b: Integer) =>
        java.lang.Integer.compare(cp.probWeights(b.intValue), cp.probWeights(a.intValue))
      )
      val hands = new Array[HoleCards](math.min(maxHands, cp.size))
      val weights = new Array[Int](hands.length)
      var keptCount = 0
      var mass = 0.0
      var idx = 0
      while idx < indices.length &&
        keptCount < maxHands &&
        (keptCount < minHands || mass < minMass)
      do
        val i = indices(idx).intValue
        val w = sicfun.core.Prob(cp.probWeights(i)).toDouble
        if w > 0.0 then
          hands(keptCount) = cp.hands(i)
          weights(keptCount) = cp.probWeights(i)
          mass += w
          keptCount += 1
        idx += 1
      if keptCount == 0 then cp
      else new HoldemEquity.CompactPosterior(hands, weights, keptCount)

  private def compactPosteriorForEquity(
      posterior: DiscreteDistribution[HoleCards]
  ): DiscreteDistribution[HoleCards] =
    val maxHands = configuredEquityPosteriorMaxHands
    if maxHands <= 0 || posterior.weights.size <= maxHands then posterior
    else
      val minMass = configuredEquityPosteriorMinMass
      val minHands = math.max(1, math.min(maxHands, DefaultEquityPosteriorMinHands))
      // Use Array sort directly instead of .toVector.sortBy to avoid
      // intermediate Vector allocation.
      val entries = posterior.weights.toArray
      java.util.Arrays.sort(entries, (a: (HoleCards, Double), b: (HoleCards, Double)) =>
        java.lang.Double.compare(b._2, a._2)
      )
      val kept = Map.newBuilder[HoleCards, Double]
      var keptCount = 0
      var mass = 0.0
      var idx = 0
      while idx < entries.length &&
        keptCount < maxHands &&
        (keptCount < minHands || mass < minMass)
      do
        val (hand, probability) = entries(idx)
        if probability > 0.0 then
          kept += hand -> probability
          mass += probability
          keptCount += 1
        idx += 1
      val compacted = kept.result()
      if compacted.isEmpty then posterior
      else DiscreteDistribution(compacted).normalized

  private def configuredEquityPosteriorMaxHands: Int =
    configuredEquityPosteriorMaxHandsCached

  private def configuredEquityPosteriorMinMass: Double =
    configuredEquityPosteriorMinMassCached

  private lazy val configuredEquityPosteriorMaxHandsCached: Int =
    GpuRuntimeSupport
      .resolveNonEmpty(EquityPosteriorMaxHandsProperty, EquityPosteriorMaxHandsEnv)
      .flatMap(_.toIntOption)
      .map(value => math.max(0, value))
      .getOrElse(DefaultEquityPosteriorMaxHands)

  private lazy val configuredEquityPosteriorMinMassCached: Double =
    GpuRuntimeSupport
      .resolveNonEmpty(EquityPosteriorMinMassProperty, EquityPosteriorMinMassEnv)
      .flatMap(_.toDoubleOption)
      .filter(value => value.isFinite && value > 0.0)
      .map(value => math.min(1.0, value))
      .getOrElse(DefaultEquityPosteriorMinMass)

  private inline val Eps = Probability.Eps

  private def responseAwareRaiseEv(
      hero: HoleCards,
      state: GameState,
      posterior: DiscreteDistribution[HoleCards],
      heroEquityMean: Double,
      action: PokerAction,
      responseModel: VillainResponseModel,
      equityTrials: Int,
      rng: Random
  ): Double =
    val raiseAmount = action match
      case PokerAction.Raise(amount) =>
        require(amount > 0.0, "raise amount must be positive")
        amount
      case _ => throw new IllegalArgumentException("responseAwareRaiseEv expects a raise action")

    responseModel match
      case uniform: UniformVillainResponseModel =>
        val profile = uniform.responseProfile(state, action)
        val foldProbability = math.max(0.0, math.min(1.0, profile.foldProbability))
        val continueProbability = math.max(0.0, math.min(1.0, profile.continueProbability))
        raiseEvFromContinuation(
          state = state,
          raiseAmount = raiseAmount,
          foldProbability = foldProbability,
          continueProbability = continueProbability,
          continuationEquityMean = heroEquityMean
        )
      case _ =>
        val aggregated = aggregateResponses(
          posterior = posterior,
          state = state,
          action = action,
          responseModel = responseModel
        )
        val foldProbability = aggregated.foldProbability
        val continueProbability = aggregated.continueProbability
        if continueProbability <= Eps then state.pot
        else
          val continuationEquityMean =
            if aggregated.continuationMatchesPosterior then heroEquityMean
            else
              val continuationRange = DiscreteDistribution(aggregated.continueWeights).normalized
              HoldemEquity.equityMonteCarlo(
                hero = hero,
                board = state.board,
                villainRange = continuationRange,
                trials = equityTrials,
                rng = rng
              ).mean
          raiseEvFromContinuation(
            state = state,
            raiseAmount = raiseAmount,
            foldProbability = foldProbability,
            continueProbability = continueProbability,
            continuationEquityMean = continuationEquityMean
          )

  /** Approximate EV for raise lines with response-aware continuation probabilities.
    *
    * Behavior model:
    * - folds win current pot immediately
    * - continuing lines assume a one-street realized-equity approximation
    */
  private def raiseEvFromContinuation(
      state: GameState,
      raiseAmount: Double,
      foldProbability: Double,
      continueProbability: Double,
      continuationEquityMean: Double
  ): Double =
    if continueProbability <= Eps then state.pot
    else
      // Approximate one-street raise EV: win current pot on folds,
      // otherwise equity realization in a called pot where both players
      // contribute the raise amount.
      val continuePot = state.pot + 2.0 * raiseAmount
      val continueEv = (continuationEquityMean * continuePot) - raiseAmount
      (foldProbability * state.pot) + (continueProbability * continueEv)

  private def aggregateResponses(
      posterior: DiscreteDistribution[HoleCards],
      state: GameState,
      action: PokerAction,
      responseModel: VillainResponseModel
  ): AggregatedResponses =
    val continueWeights = mutable.Map.empty[HoleCards, Double].withDefaultValue(0.0)
    var foldProbability = 0.0
    var continueProbability = 0.0
    var hasBaselineProfile = false
    var baselineFold = 0.0
    var baselineContinue = 0.0
    var baselineRaise = 0.0
    var continuationMatchesPosterior = true

    posterior.weights.foreach { case (hand, priorProb) =>
      if priorProb > 0.0 then
        val profile = responseModel.response(hand, state, action)
        if !hasBaselineProfile then
          hasBaselineProfile = true
          baselineFold = profile.foldProbability
          baselineContinue = profile.continueProbability
          baselineRaise = profile.raiseProbability
        else if continuationMatchesPosterior then
          val foldDiff = math.abs(profile.foldProbability - baselineFold)
          val continueDiff = math.abs(profile.continueProbability - baselineContinue)
          val raiseDiff = math.abs(profile.raiseProbability - baselineRaise)
          if foldDiff > ProfileEps || continueDiff > ProfileEps || raiseDiff > ProfileEps then
            continuationMatchesPosterior = false
        val weightedFold = priorProb * profile.foldProbability
        val weightedContinue = priorProb * profile.continueProbability
        foldProbability += weightedFold
        continueProbability += weightedContinue
        if weightedContinue > 0.0 then
          continueWeights.update(hand, continueWeights(hand) + weightedContinue)
    }
    AggregatedResponses(
      foldProbability = math.max(0.0, math.min(1.0, foldProbability)),
      continueProbability = math.max(0.0, math.min(1.0, continueProbability)),
      continueWeights = continueWeights.toMap,
      continuationMatchesPosterior = continuationMatchesPosterior
    )

  /** End-to-end helper: infer posterior then recommend best hero action. */
  def inferAndRecommend(
      hero: HoleCards,
      state: GameState,
      folds: Vector[PreflopFold],
      tableRanges: TableRanges,
      villainPos: Position,
      observations: Seq[VillainObservation],
      actionModel: PokerActionModel,
      candidateActions: Vector[PokerAction],
      actionValueModel: ActionValueModel = ActionValueModel.ChipEv(),
      villainResponseModel: Option[VillainResponseModel] = None,
      bunchingTrials: Int = 10_000,
      equityTrials: Int = 50_000,
      rng: Random = new Random()
  ): InferenceDecisionResult =
    val posteriorInference = inferPosterior(
      hero = hero,
      board = state.board,
      folds = folds,
      tableRanges = tableRanges,
      villainPos = villainPos,
      observations = observations,
      actionModel = actionModel,
      bunchingTrials = bunchingTrials,
      rng = new Random(rng.nextLong())
    )

    val recommendation = recommendActionWithCompact(
      hero = hero,
      state = state,
      posterior = posteriorInference.posterior,
      compact = posteriorInference.compact,
      candidateActions = candidateActions,
      actionValueModel = actionValueModel,
      villainResponseModel = villainResponseModel,
      equityTrials = equityTrials,
      rng = new Random(rng.nextLong())
    )

    InferenceDecisionResult(posteriorInference, recommendation)
