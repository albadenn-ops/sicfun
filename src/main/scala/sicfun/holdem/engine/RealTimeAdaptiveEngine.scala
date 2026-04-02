package sicfun.holdem.engine
import sicfun.holdem.history.ShowdownRecord
import sicfun.holdem.types.*
import sicfun.holdem.gpu.*
import sicfun.holdem.model.*
import sicfun.holdem.equity.*
import sicfun.holdem.provider.*
import sicfun.holdem.cfr.*

import sicfun.core.{CollapseMetrics, DiscreteDistribution}

import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.{AtomicLong, AtomicReference}
import scala.util.Random
import scala.util.control.NonFatal
import scala.util.hashing.MurmurHash3

/** High-level villain archetypes used by the real-time adaptive engine. */
enum PlayerArchetype:
  case Nit
  case Tag
  case Lag
  case CallingStation
  case Maniac

/** Normalized posterior over villain archetypes. */
final case class ArchetypePosterior(weights: Map[PlayerArchetype, Double]):
  require(weights.nonEmpty, "archetype posterior must be non-empty")
  require(
    weights.keySet == ArchetypePosterior.allArchetypeSet,
    "archetype posterior must define all archetypes"
  )
  require(weights.values.forall(w => w >= 0.0 && w.isFinite), "archetype weights must be finite and non-negative")
  private val total = weights.values.sum
  require(math.abs(total - 1.0) < 1e-9, s"archetype posterior must sum to 1.0, got $total")

  def probabilityOf(archetype: PlayerArchetype): Double =
    weights.getOrElse(archetype, 0.0)

  def mapEstimate: PlayerArchetype =
    weights.maxBy(_._2)._1

object ArchetypePosterior:
  // Cache the set of all archetypes to avoid repeated .toSet allocation
  // on every ArchetypePosterior construction (called per villain response observation).
  val allArchetypeSet: Set[PlayerArchetype] = PlayerArchetype.values.toSet

  val uniform: ArchetypePosterior =
    val archetypes = PlayerArchetype.values.toVector
    val p = 1.0 / archetypes.length.toDouble
    ArchetypePosterior(archetypes.map(a => a -> p).toMap)

/** Small telemetry snapshot for inference caching efficiency. */
final case class AdaptiveCacheStats(inferenceHits: Long, inferenceMisses: Long):
  require(inferenceHits >= 0L, "inferenceHits must be non-negative")
  require(inferenceMisses >= 0L, "inferenceMisses must be non-negative")

/** Optional equilibrium baseline configuration (CFR) for exploitability-aware blending. */
final case class EquilibriumBaselineConfig(
    iterations: Int = 1_200,
    blendWeight: Double = 0.35,
    maxLocalExploitabilityForTrust: Double = Double.PositiveInfinity,
    maxBaselineActionRegret: Double = Double.PositiveInfinity,
    cfrPlus: Boolean = true,
    averagingDelay: Int = 100,
    linearAveraging: Boolean = true,
    maxVillainHands: Int = 96,
    equityTrials: Int = 4_000,
    includeVillainReraises: Boolean = true,
    villainReraiseMultipliers: Vector[Double] = Vector(2.0),
    preferNativeBatch: Boolean = true
):
  require(iterations > 0, "iterations must be positive")
  require(blendWeight >= 0.0 && blendWeight <= 1.0, "blendWeight must be in [0, 1]")
  require(
    maxLocalExploitabilityForTrust >= 0.0 && !maxLocalExploitabilityForTrust.isNaN,
    "maxLocalExploitabilityForTrust must be >= 0 or positive infinity"
  )
  require(
    maxBaselineActionRegret >= 0.0 && !maxBaselineActionRegret.isNaN,
    "maxBaselineActionRegret must be >= 0 or positive infinity"
  )
  require(averagingDelay >= 0, "averagingDelay must be non-negative")
  require(maxVillainHands > 0, "maxVillainHands must be positive")
  require(equityTrials > 0, "equityTrials must be positive")
  require(
    villainReraiseMultipliers.forall(m => m > 1.0 && m.isFinite),
    "villainReraiseMultipliers must be finite and > 1.0"
  )

enum AdaptationDecisionSource:
  case AdaptiveOnly
  case BlendedWithBaseline
  case BaselineGuardrail

final case class AdaptationDecisionTrace(
    source: AdaptationDecisionSource,
    requestedBlendWeight: Double,
    effectiveBlendWeight: Double,
    baselineBestAction: Option[PokerAction],
    baselineChosenActionRegret: Double,
    baselineLocalExploitability: Option[Double],
    reason: Option[String]
):
  require(requestedBlendWeight >= 0.0 && requestedBlendWeight <= 1.0, "requestedBlendWeight must be in [0, 1]")
  require(effectiveBlendWeight >= 0.0 && effectiveBlendWeight <= 1.0, "effectiveBlendWeight must be in [0, 1]")
  require(
    baselineChosenActionRegret >= 0.0 && !baselineChosenActionRegret.isNaN,
    "baselineChosenActionRegret must be non-negative or positive infinity"
  )

object AdaptationDecisionTrace:
  val adaptiveOnly: AdaptationDecisionTrace =
    AdaptationDecisionTrace(
      source = AdaptationDecisionSource.AdaptiveOnly,
      requestedBlendWeight = 0.0,
      effectiveBlendWeight = 0.0,
      baselineBestAction = None,
      baselineChosenActionRegret = 0.0,
      baselineLocalExploitability = None,
      reason = None
    )

/** Adaptive recommendation over a provided posterior range. */
final case class AdaptiveRecommendationResult(
    recommendation: ActionRecommendation,
    archetypePosterior: ArchetypePosterior,
    archetypeMap: PlayerArchetype,
    equityTrialsUsed: Int,
    equilibriumBaseline: Option[HoldemCfrSolution] = None,
    adaptationTrace: AdaptationDecisionTrace = AdaptationDecisionTrace.adaptiveOnly
):
  require(equityTrialsUsed > 0, "equityTrialsUsed must be positive")

/** End-to-end adaptive decision (posterior inference + recommendation). */
final case class AdaptiveDecisionResult(
    decision: InferenceDecisionResult,
    archetypePosterior: ArchetypePosterior,
    archetypeMap: PlayerArchetype,
    equityTrialsUsed: Int,
    cacheStats: AdaptiveCacheStats,
    equilibriumBaseline: Option[HoldemCfrSolution] = None,
    adaptationTrace: AdaptationDecisionTrace = AdaptationDecisionTrace.adaptiveOnly
):
  require(equityTrialsUsed > 0, "equityTrialsUsed must be positive")

/** Real-time adaptive engine:
  *  - learns opponent archetype online from observed response-to-raise behavior
  *  - caches expensive posterior inference for repeated identical contexts
  *  - enforces a simple per-decision latency budget by scaling equity trials
  *
  * Latency policy is asymmetric by design:
  * - Inference can be short-circuited to a compact per-position posterior under a
  *   strict budget/uniform-model/no-DDRE/no-showdown-history gate.
  * - Recommendation quality is then scaled via equity trial count.
  */
final class RealTimeAdaptiveEngine(
    tableRanges: TableRanges,
    actionModel: PokerActionModel,
    bunchingTrials: Int = 10_000,
    defaultEquityTrials: Int = 50_000,
    minEquityTrials: Int = 2_000,
    equilibriumBaselineConfig: Option[EquilibriumBaselineConfig] = None
):
  require(bunchingTrials > 0, "bunchingTrials must be positive")
  require(defaultEquityTrials > 0, "defaultEquityTrials must be positive")
  require(minEquityTrials > 0, "minEquityTrials must be positive")
  require(minEquityTrials <= defaultEquityTrials, "minEquityTrials must be <= defaultEquityTrials")

  private final case class InferenceCacheKey(
      hero: HoleCards,
      board: Board,
      folds: Vector[PreflopFold],
      villainPos: Position,
      observationsHash: Int,
      showdownHistoryHash: Int,
      tableRangesIdentity: Int,
      bunchingTrials: Int
  )

  private final case class RecommendationOutcome(
      recommendation: ActionRecommendation,
      equilibriumBaseline: Option[HoldemCfrSolution],
      adaptationTrace: AdaptationDecisionTrace
  )

  private val MaxInferenceCacheSize = 64
  private val LowLatencyMaxPosteriorHands = 192
  private val tableRangesIdentity = System.identityHashCode(tableRanges)
  private val archetypePosteriorRef = new AtomicReference[ArchetypePosterior](ArchetypePosterior.uniform)
  private val inferenceCache = new ConcurrentHashMap[InferenceCacheKey, PosteriorInferenceResult]()
  private val inferenceHitsCounter = new AtomicLong(0L)
  private val inferenceMissesCounter = new AtomicLong(0L)
  private val lowLatencyPosteriorByPosition: Map[Position, PosteriorInferenceResult] =
    tableRanges.format.preflopOrder.map { position =>
      val compacted = compactRange(tableRanges.rangeFor(position).normalized, LowLatencyMaxPosteriorHands)
      val support = CollapseMetrics.effectiveSupport(compacted)
      position ->
        PosteriorInferenceResult(
          prior = compacted,
          posterior = compacted,
          compact = None,
          logEvidence = 0.0,
          collapse = PosteriorCollapse(
            entropyReduction = 0.0,
            klDivergence = 0.0,
            effectiveSupportPrior = support,
            effectiveSupportPosterior = support,
            collapseRatio = 0.0
          )
        )
    }.toMap

  private val adaptiveResponseModel = new UniformVillainResponseModel:
    override def responseProfile(
        state: GameState,
        heroAction: PokerAction
    ): VillainResponseProfile =
      heroAction match
        case PokerAction.Raise(_) => blendedRaiseResponse(archetypePosteriorRef.get())
        case _ => VillainResponseProfile(0.0, 1.0, 0.0)

  def archetypePosterior: ArchetypePosterior = archetypePosteriorRef.get()

  def cacheStats: AdaptiveCacheStats =
    AdaptiveCacheStats(inferenceHitsCounter.get(), inferenceMissesCounter.get())

  def clearInferenceCache(): Unit =
    inferenceCache.clear()

  def seedArchetypePosterior(posterior: ArchetypePosterior): Unit =
    archetypePosteriorRef.set(posterior)

  /** Online Bayesian update of villain archetype from observed response to a hero raise.
    *
    * Only fold/call/raise observations carry signal for this model.
    */
  def observeVillainResponseToRaise(villainAction: PokerAction): ArchetypePosterior =
    archetypePosteriorRef.updateAndGet(current =>
      ArchetypeLearning.updatePosterior(current, villainAction)
    )

  /** Low-latency adaptive recommendation against a provided posterior range. */
  def recommendAgainstPosterior(
      hero: HoleCards,
      state: GameState,
      posterior: sicfun.core.DiscreteDistribution[HoleCards],
      candidateActions: Vector[PokerAction],
      actionValueModel: ActionValueModel = ActionValueModel.ChipEv(),
      decisionBudgetMillis: Option[Long] = None,
      rng: Random = new Random()
  ): AdaptiveRecommendationResult =
    val trials = effectiveEquityTrials(decisionBudgetMillis, inferenceElapsedMs = 0L)
    val recommendationOutcome = buildRecommendationOutcome(
      hero = hero,
      state = state,
      posterior = posterior,
      compact = None,
      candidateActions = candidateActions,
      actionValueModel = actionValueModel,
      equityTrials = trials,
      rng = rng
    )
    val currentPosterior = archetypePosteriorRef.get()
    AdaptiveRecommendationResult(
      recommendation = recommendationOutcome.recommendation,
      archetypePosterior = currentPosterior,
      archetypeMap = currentPosterior.mapEstimate,
      equityTrialsUsed = trials,
      equilibriumBaseline = recommendationOutcome.equilibriumBaseline,
      adaptationTrace = recommendationOutcome.adaptationTrace
    )

  /** End-to-end adaptive decision with posterior inference cache. */
  def decide(
      hero: HoleCards,
      state: GameState,
      folds: Vector[PreflopFold],
      villainPos: Position,
      observations: Seq[VillainObservation],
      candidateActions: Vector[PokerAction],
      actionValueModel: ActionValueModel = ActionValueModel.ChipEv(),
      decisionBudgetMillis: Option[Long] = None,
      rng: Random = new Random(),
      revealedCards: Option[HoleCards] = None,
      showdownHistory: Vector[ShowdownRecord] = Vector.empty
  ): AdaptiveDecisionResult =
    val startedAt = System.nanoTime()
    val posteriorInference = revealedCards match
      case Some(_) =>
        RangeInferenceEngine.inferPosterior(
          hero = hero,
          board = state.board,
          folds = folds,
          tableRanges = tableRanges,
          villainPos = villainPos,
          observations = observations,
          actionModel = actionModel,
          bunchingTrials = 1,
          useCache = false,
          revealedCards = revealedCards,
          showdownHistory = showdownHistory
        )
      case None =>
        val effectiveBunching = effectiveBunchingTrials(decisionBudgetMillis)
        val useLowLatencyPosterior =
          effectiveBunching <= 1 &&
            actionModel.isEffectivelyUniform &&
            HoldemDdreProvider.configuredConfig().mode == HoldemDdreProvider.Mode.Off &&
            showdownHistory.isEmpty
        val observationsHash =
          if useLowLatencyPosterior then 0 else MurmurHash3.seqHash(observations)
        val showdownHistoryHash =
          if useLowLatencyPosterior then 0 else RangeInferenceEngine.showdownHistoryHash(showdownHistory)
        cachedPosteriorInference(
          key = InferenceCacheKey(
            hero = hero,
            board = state.board,
            folds = folds,
            villainPos = villainPos,
            observationsHash = observationsHash,
            showdownHistoryHash = showdownHistoryHash,
            tableRangesIdentity = tableRangesIdentity,
            bunchingTrials = effectiveBunching
          ),
          hero = hero,
          state = state,
          folds = folds,
          villainPos = villainPos,
          observations = observations,
          effectiveBunching = effectiveBunching,
          useLowLatencyPosterior = useLowLatencyPosterior,
          showdownHistory = showdownHistory,
          rng = rng
        )

    val inferenceElapsedMs = (System.nanoTime() - startedAt) / 1_000_000L
    val trials = effectiveEquityTrials(decisionBudgetMillis, inferenceElapsedMs)
    val recommendationOutcome = buildRecommendationOutcome(
      hero = hero,
      state = state,
      posterior = posteriorInference.posterior,
      compact = posteriorInference.compact,
      candidateActions = candidateActions,
      actionValueModel = actionValueModel,
      equityTrials = trials,
      rng = new Random(rng.nextLong())
    )

    val currentPosterior = archetypePosteriorRef.get()
    AdaptiveDecisionResult(
      decision = InferenceDecisionResult(posteriorInference, recommendationOutcome.recommendation),
      archetypePosterior = currentPosterior,
      archetypeMap = currentPosterior.mapEstimate,
      equityTrialsUsed = trials,
      cacheStats = cacheStats,
      equilibriumBaseline = recommendationOutcome.equilibriumBaseline,
      adaptationTrace = recommendationOutcome.adaptationTrace
    )

  private def blendedRaiseResponse(posterior: ArchetypePosterior): VillainResponseProfile =
    ArchetypeLearning.blendedRaiseResponse(posterior)

  /** Scales equity Monte Carlo trials to fit the remaining decision budget.
    *
    * `None` means "use full quality defaults". A budget `<= 1ms` forces the minimum.
    */
  private def effectiveEquityTrials(
      decisionBudgetMillis: Option[Long],
      inferenceElapsedMs: Long
  ): Int =
    decisionBudgetMillis match
      case None => defaultEquityTrials
      case Some(budget) if budget <= 1L => minEquityTrials
      case Some(budget) =>
        val remaining = budget - inferenceElapsedMs
        if remaining <= 0L then minEquityTrials
        else
          val scale = remaining.toDouble / budget.toDouble
          val scaled = math.round(defaultEquityTrials.toDouble * scale).toInt
          math.max(minEquityTrials, math.min(defaultEquityTrials, scaled))

  private def effectiveBunchingTrials(decisionBudgetMillis: Option[Long]): Int =
    decisionBudgetMillis match
      case Some(budget) if budget <= 1L => 1
      case _ => bunchingTrials

  /** Bounded cache lookup for inferred posteriors.
    *
    * Keys intentionally include pre-hashed observations/showdown history to keep
    * lookup overhead stable and avoid storing large vectors in the map.
    */
  private def cachedPosteriorInference(
      key: InferenceCacheKey,
      hero: HoleCards,
      state: GameState,
      folds: Vector[PreflopFold],
      villainPos: Position,
      observations: Seq[VillainObservation],
      effectiveBunching: Int,
      useLowLatencyPosterior: Boolean,
      showdownHistory: Vector[ShowdownRecord],
      rng: Random
  ): PosteriorInferenceResult =
    val cached = inferenceCache.get(key)
    if cached != null then
      inferenceHitsCounter.incrementAndGet()
      cached
    else
      inferenceMissesCounter.incrementAndGet()
      publishInference(
        key,
        computePosteriorInference(
          hero = hero,
          state = state,
          folds = folds,
          villainPos = villainPos,
          observations = observations,
          effectiveBunching = effectiveBunching,
          useLowLatencyPosterior = useLowLatencyPosterior,
          showdownHistory = showdownHistory,
          rng = rng
        )
      )

  private def computePosteriorInference(
      hero: HoleCards,
      state: GameState,
      folds: Vector[PreflopFold],
      villainPos: Position,
      observations: Seq[VillainObservation],
      effectiveBunching: Int,
      useLowLatencyPosterior: Boolean,
      showdownHistory: Vector[ShowdownRecord],
      rng: Random
  ): PosteriorInferenceResult =
    if useLowLatencyPosterior then
      lowLatencyPosteriorInference(villainPos)
    else
      RangeInferenceEngine.inferPosterior(
        hero = hero,
        board = state.board,
        folds = folds,
        tableRanges = tableRanges,
        villainPos = villainPos,
        observations = observations,
        actionModel = actionModel,
        bunchingTrials = effectiveBunching,
        rng = new Random(rng.nextLong()),
        showdownHistory = showdownHistory
      )

  private def publishInference(
      key: InferenceCacheKey,
      inference: PosteriorInferenceResult
  ): PosteriorInferenceResult =
    if inferenceCache.size() >= MaxInferenceCacheSize then inferenceCache.clear()
    val existing = inferenceCache.putIfAbsent(key, inference)
    if existing != null then existing else inference

  private def buildRecommendationOutcome(
      hero: HoleCards,
      state: GameState,
      posterior: DiscreteDistribution[HoleCards],
      compact: Option[HoldemEquity.CompactPosterior],
      candidateActions: Vector[PokerAction],
      actionValueModel: ActionValueModel,
      equityTrials: Int,
      rng: Random
  ): RecommendationOutcome =
    val baseRecommendation = RangeInferenceEngine.recommendActionWithCompact(
      hero = hero,
      state = state,
      posterior = posterior,
      compact = compact,
      candidateActions = candidateActions,
      actionValueModel = actionValueModel,
      villainResponseModel = Some(adaptiveResponseModel),
      equityTrials = equityTrials,
      rng = rng
    )
    if equilibriumBaselineConfig.isEmpty then
      RecommendationOutcome(baseRecommendation, None, AdaptationDecisionTrace.adaptiveOnly)
    else
      val (recommendation, equilibriumBaseline, adaptationTrace) =
        maybeBlendWithEquilibriumBaseline(
          hero = hero,
          state = state,
          posterior = posterior,
          candidateActions = candidateActions,
          baseRecommendation = baseRecommendation,
          rng = new Random(rng.nextLong())
        )
      RecommendationOutcome(recommendation, equilibriumBaseline, adaptationTrace)

  private def lowLatencyPosteriorInference(
      villainPos: Position
  ): PosteriorInferenceResult =
    lowLatencyPosteriorByPosition.getOrElse(
      villainPos,
      throw new IllegalArgumentException(s"villain position $villainPos is not part of table format")
    )

  private def compactRange(
      range: DiscreteDistribution[HoleCards],
      maxHands: Int
  ): DiscreteDistribution[HoleCards] =
    if maxHands <= 0 || range.weights.size <= maxHands then range
    else
      val sorted = range.weights.toVector.sortBy { case (_, probability) => -probability }
      val kept = Map.newBuilder[HoleCards, Double]
      var i = 0
      while i < sorted.length && i < maxHands do
        val (hand, probability) = sorted(i)
        if probability > 0.0 then kept += hand -> probability
        i += 1
      DiscreteDistribution(kept.result()).normalized

  /** Optional CFR baseline blend with two trust gates:
    * 1. local exploitability gate (whether baseline is trusted)
    * 2. action-regret guardrail (whether blended choice drifts too far from CFR best)
    */
  private def maybeBlendWithEquilibriumBaseline(
      hero: HoleCards,
      state: GameState,
      posterior: DiscreteDistribution[HoleCards],
      candidateActions: Vector[PokerAction],
      baseRecommendation: ActionRecommendation,
      rng: Random
  ): (ActionRecommendation, Option[HoldemCfrSolution], AdaptationDecisionTrace) =
    equilibriumBaselineConfig match
      case None =>
        (baseRecommendation, None, AdaptationDecisionTrace.adaptiveOnly)
      case Some(config) =>
        try
          val baseline = HoldemCfrSolver.solve(
            hero = hero,
            state = state,
            villainPosterior = posterior,
            candidateActions = candidateActions,
            config = HoldemCfrConfig(
              iterations = config.iterations,
              cfrPlus = config.cfrPlus,
              averagingDelay = config.averagingDelay,
              linearAveraging = config.linearAveraging,
              maxVillainHands = config.maxVillainHands,
              equityTrials = config.equityTrials,
              includeVillainReraises = config.includeVillainReraises,
              villainReraiseMultipliers = config.villainReraiseMultipliers,
              preferNativeBatch = config.preferNativeBatch,
              rngSeed = rng.nextLong()
            )
          )

          val baselineRecommendation = recommendationFromBaseline(baseRecommendation, baseline)
          val baselineTrusted = baseline.localExploitability <= config.maxLocalExploitabilityForTrust
          if !baselineTrusted then
            (
              baseRecommendation,
              Some(baseline),
              AdaptationDecisionTrace(
                source = AdaptationDecisionSource.AdaptiveOnly,
                requestedBlendWeight = config.blendWeight,
                effectiveBlendWeight = 0.0,
                baselineBestAction = Some(baseline.bestAction),
                baselineChosenActionRegret = 0.0,
                baselineLocalExploitability = Some(baseline.localExploitability),
                reason = Some("baseline_local_exploitability_exceeds_trust_threshold")
              )
            )
          else
            val proposedRecommendation =
              if config.blendWeight <= 0.0 then baseRecommendation
              else blendRecommendationWithBaseline(baseRecommendation, baseline, config.blendWeight)
            val baselineRegret = baselineActionRegret(baseline, proposedRecommendation.bestAction)
            if baselineRegret > config.maxBaselineActionRegret then
              (
                baselineRecommendation,
                Some(baseline),
                AdaptationDecisionTrace(
                  source = AdaptationDecisionSource.BaselineGuardrail,
                  requestedBlendWeight = config.blendWeight,
                  effectiveBlendWeight = 1.0,
                  baselineBestAction = Some(baseline.bestAction),
                  baselineChosenActionRegret = baselineRegret,
                  baselineLocalExploitability = Some(baseline.localExploitability),
                  reason = Some("baseline_action_regret_exceeds_threshold")
                )
              )
            else
              (
                proposedRecommendation,
                Some(baseline),
                AdaptationDecisionTrace(
                  source =
                    if config.blendWeight <= 0.0 then AdaptationDecisionSource.AdaptiveOnly
                    else AdaptationDecisionSource.BlendedWithBaseline,
                  requestedBlendWeight = config.blendWeight,
                  effectiveBlendWeight = config.blendWeight,
                  baselineBestAction = Some(baseline.bestAction),
                  baselineChosenActionRegret = baselineRegret,
                  baselineLocalExploitability = Some(baseline.localExploitability),
                  reason = None
                )
              )
        catch
          case NonFatal(error) =>
            GpuRuntimeSupport.warn(
              s"adaptive CFR baseline failed (${failureDetail(error)}); using range-inference recommendation"
            )
            (
              baseRecommendation,
              None,
              AdaptationDecisionTrace(
                source = AdaptationDecisionSource.AdaptiveOnly,
                requestedBlendWeight = config.blendWeight,
                effectiveBlendWeight = 0.0,
                baselineBestAction = None,
                baselineChosenActionRegret = 0.0,
                baselineLocalExploitability = None,
                reason = Some("baseline_unavailable")
              )
            )

  private def recommendationFromBaseline(
      baseRecommendation: ActionRecommendation,
      baseline: HoldemCfrSolution
  ): ActionRecommendation =
    ActionRecommendation(
      heroEquity = baseRecommendation.heroEquity,
      actionEvaluations = baseline.actionEvaluations,
      bestAction = baseline.bestAction
    )

  private def blendRecommendationWithBaseline(
      baseRecommendation: ActionRecommendation,
      baseline: HoldemCfrSolution,
      blendWeight: Double
  ): ActionRecommendation =
    val baselineEvByAction = baseline.actionEvaluations.map(eval => eval.action -> eval.expectedValue).toMap
    val blendedEvaluations = baseRecommendation.actionEvaluations.map { evaluation =>
      val baselineEv = baselineEvByAction.getOrElse(evaluation.action, evaluation.expectedValue)
      val blendedEv =
        ((1.0 - blendWeight) * evaluation.expectedValue) +
          (blendWeight * baselineEv)
      ActionEvaluation(evaluation.action, blendedEv)
    }
    val best = blendedEvaluations.reduceLeft { (a, b) =>
      if a.expectedValue >= b.expectedValue then a else b
    }
    baseRecommendation.copy(
      actionEvaluations = blendedEvaluations,
      bestAction = best.action
    )

  private def baselineActionRegret(
      baseline: HoldemCfrSolution,
      chosenAction: PokerAction
  ): Double =
    val baselineEvByAction = baseline.actionEvaluations.map(eval => eval.action -> eval.expectedValue).toMap
    val bestBaselineEv = baselineEvByAction.getOrElse(baseline.bestAction, 0.0)
    baselineEvByAction.get(chosenAction) match
      case Some(chosenBaselineEv) => math.max(0.0, bestBaselineEv - chosenBaselineEv)
      case None                   => Double.PositiveInfinity

  private def failureDetail(error: Throwable): String =
    Option(error.getMessage).map(_.trim).filter(_.nonEmpty).getOrElse(error.getClass.getSimpleName)
