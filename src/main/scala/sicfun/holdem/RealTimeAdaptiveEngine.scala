package sicfun.holdem

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
    weights.keySet == PlayerArchetype.values.toSet,
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
  require(averagingDelay >= 0, "averagingDelay must be non-negative")
  require(maxVillainHands > 0, "maxVillainHands must be positive")
  require(equityTrials > 0, "equityTrials must be positive")
  require(
    villainReraiseMultipliers.forall(m => m > 1.0 && m.isFinite),
    "villainReraiseMultipliers must be finite and > 1.0"
  )

/** Adaptive recommendation over a provided posterior range. */
final case class AdaptiveRecommendationResult(
    recommendation: ActionRecommendation,
    archetypePosterior: ArchetypePosterior,
    archetypeMap: PlayerArchetype,
    equityTrialsUsed: Int,
    equilibriumBaseline: Option[HoldemCfrSolution] = None
):
  require(equityTrialsUsed > 0, "equityTrialsUsed must be positive")

/** End-to-end adaptive decision (posterior inference + recommendation). */
final case class AdaptiveDecisionResult(
    decision: InferenceDecisionResult,
    archetypePosterior: ArchetypePosterior,
    archetypeMap: PlayerArchetype,
    equityTrialsUsed: Int,
    cacheStats: AdaptiveCacheStats,
    equilibriumBaseline: Option[HoldemCfrSolution] = None
):
  require(equityTrialsUsed > 0, "equityTrialsUsed must be positive")

/** Real-time adaptive engine:
  *  - learns opponent archetype online from observed response-to-raise behavior
  *  - caches expensive posterior inference for repeated identical contexts
  *  - enforces a simple per-decision latency budget by scaling equity trials
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

  // Heuristic response-to-raise profiles per archetype: VillainResponseProfile(foldFreq, callFreq, raiseFreq).
  // These approximate typical frequencies observed in online 6-max NLHE microstakes.
  private val archetypeRaiseResponse: Map[PlayerArchetype, VillainResponseProfile] = Map(
    PlayerArchetype.Nit            -> VillainResponseProfile(0.68, 0.28, 0.04),
    PlayerArchetype.Tag            -> VillainResponseProfile(0.48, 0.42, 0.10),
    PlayerArchetype.Lag            -> VillainResponseProfile(0.34, 0.50, 0.16),
    PlayerArchetype.CallingStation -> VillainResponseProfile(0.20, 0.73, 0.07),
    PlayerArchetype.Maniac         -> VillainResponseProfile(0.25, 0.30, 0.45)
  )

  private final case class InferenceCacheKey(
      hero: HoleCards,
      board: Board,
      folds: Vector[PreflopFold],
      villainPos: Position,
      observationsHash: Int,
      tableRangesIdentity: Int,
      bunchingTrials: Int
  )

  private val Eps = 1e-12
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

  /** Online Bayesian update of villain archetype from observed response to a hero raise.
    *
    * Only fold/call/raise observations carry signal for this model.
    */
  def observeVillainResponseToRaise(villainAction: PokerAction): ArchetypePosterior =
    responseOutcome(villainAction) match
      case None => archetypePosteriorRef.get()
      case Some(outcome) =>
        archetypePosteriorRef.updateAndGet { current =>
          val updated = Map.newBuilder[PlayerArchetype, Double]
          var total = 0.0
          PlayerArchetype.values.foreach { archetype =>
            val prior = current.probabilityOf(archetype)
            val likelihood = responseLikelihood(archetype, outcome)
            val posteriorUnnormalized = prior * likelihood
            updated += archetype -> posteriorUnnormalized
            total += posteriorUnnormalized
          }
          val normalized =
            if total <= Eps then ArchetypePosterior.uniform.weights
            else
              val inv = 1.0 / total
              updated.result().map { case (archetype, value) =>
                archetype -> (value * inv)
              }
          ArchetypePosterior(normalized)
        }

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
    val baseRecommendation = RangeInferenceEngine.recommendAction(
      hero = hero,
      state = state,
      posterior = posterior,
      candidateActions = candidateActions,
      actionValueModel = actionValueModel,
      villainResponseModel = Some(adaptiveResponseModel),
      equityTrials = trials,
      rng = rng
    )
    val (recommendation, equilibriumBaseline) =
      if equilibriumBaselineConfig.isEmpty then (baseRecommendation, None)
      else
        maybeBlendWithEquilibriumBaseline(
          hero = hero,
          state = state,
          posterior = posterior,
          candidateActions = candidateActions,
          baseRecommendation = baseRecommendation,
          rng = new Random(rng.nextLong())
        )
    val currentPosterior = archetypePosteriorRef.get()
    AdaptiveRecommendationResult(
      recommendation = recommendation,
      archetypePosterior = currentPosterior,
      archetypeMap = currentPosterior.mapEstimate,
      equityTrialsUsed = trials,
      equilibriumBaseline = equilibriumBaseline
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
      rng: Random = new Random()
  ): AdaptiveDecisionResult =
    val startedAt = System.nanoTime()
    val effectiveBunching = effectiveBunchingTrials(decisionBudgetMillis)
    val useLowLatencyPosterior =
      effectiveBunching <= 1 &&
        actionModel.isEffectivelyUniform &&
        HoldemDdreProvider.configuredConfig().mode == HoldemDdreProvider.Mode.Off
    val observationsHash =
      if useLowLatencyPosterior then 0 else MurmurHash3.seqHash(observations)
    val key = InferenceCacheKey(
      hero = hero,
      board = state.board,
      folds = folds,
      villainPos = villainPos,
      observationsHash = observationsHash,
      tableRangesIdentity = tableRangesIdentity,
      bunchingTrials = effectiveBunching
    )

    val cachedInference = inferenceCache.get(key)
    val posteriorInference =
      if cachedInference != null then
        inferenceHitsCounter.incrementAndGet()
        cachedInference
      else
        inferenceMissesCounter.incrementAndGet()
        val inferred =
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
              rng = new Random(rng.nextLong())
            )
        if inferenceCache.size() >= MaxInferenceCacheSize then inferenceCache.clear()
        inferenceCache.putIfAbsent(key, inferred)
        val published = inferenceCache.get(key)
        if published != null then published else inferred

    val inferenceElapsedMs = (System.nanoTime() - startedAt) / 1_000_000L
    val trials = effectiveEquityTrials(decisionBudgetMillis, inferenceElapsedMs)
    val baseRecommendation = RangeInferenceEngine.recommendAction(
      hero = hero,
      state = state,
      posterior = posteriorInference.posterior,
      candidateActions = candidateActions,
      actionValueModel = actionValueModel,
      villainResponseModel = Some(adaptiveResponseModel),
      equityTrials = trials,
      rng = new Random(rng.nextLong())
    )
    val (recommendation, equilibriumBaseline) =
      if equilibriumBaselineConfig.isEmpty then (baseRecommendation, None)
      else
        maybeBlendWithEquilibriumBaseline(
          hero = hero,
          state = state,
          posterior = posteriorInference.posterior,
          candidateActions = candidateActions,
          baseRecommendation = baseRecommendation,
          rng = new Random(rng.nextLong())
        )

    val currentPosterior = archetypePosteriorRef.get()
    AdaptiveDecisionResult(
      decision = InferenceDecisionResult(posteriorInference, recommendation),
      archetypePosterior = currentPosterior,
      archetypeMap = currentPosterior.mapEstimate,
      equityTrialsUsed = trials,
      cacheStats = cacheStats,
      equilibriumBaseline = equilibriumBaseline
    )

  private enum ResponseOutcome:
    case Fold
    case Call
    case Raise

  private def responseOutcome(action: PokerAction): Option[ResponseOutcome] =
    action match
      case PokerAction.Fold => Some(ResponseOutcome.Fold)
      case PokerAction.Call => Some(ResponseOutcome.Call)
      case PokerAction.Raise(_) => Some(ResponseOutcome.Raise)
      case _ => None

  private def responseLikelihood(
      archetype: PlayerArchetype,
      outcome: ResponseOutcome
  ): Double =
    val profile = archetypeRaiseResponse(archetype)
    outcome match
      case ResponseOutcome.Fold => profile.foldProbability
      case ResponseOutcome.Call => profile.callProbability
      case ResponseOutcome.Raise => profile.raiseProbability

  private def blendedRaiseResponse(posterior: ArchetypePosterior): VillainResponseProfile =
    var fold = 0.0
    var call = 0.0
    var raise = 0.0
    PlayerArchetype.values.foreach { archetype =>
      val weight = posterior.probabilityOf(archetype)
      val profile = archetypeRaiseResponse(archetype)
      fold += weight * profile.foldProbability
      call += weight * profile.callProbability
      raise += weight * profile.raiseProbability
    }
    val total = fold + call + raise
    if total <= Eps then VillainResponseProfile(0.0, 1.0, 0.0)
    else
      val inv = 1.0 / total
      VillainResponseProfile(fold * inv, call * inv, raise * inv)

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

  private def maybeBlendWithEquilibriumBaseline(
      hero: HoleCards,
      state: GameState,
      posterior: DiscreteDistribution[HoleCards],
      candidateActions: Vector[PokerAction],
      baseRecommendation: ActionRecommendation,
      rng: Random
  ): (ActionRecommendation, Option[HoldemCfrSolution]) =
    equilibriumBaselineConfig match
      case None =>
        (baseRecommendation, None)
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

          if config.blendWeight <= 0.0 then
            (baseRecommendation, Some(baseline))
          else
            val baselineEvByAction = baseline.actionEvaluations.map(eval => eval.action -> eval.expectedValue).toMap
            val blendedEvaluations = baseRecommendation.actionEvaluations.map { evaluation =>
              val baselineEv = baselineEvByAction.getOrElse(evaluation.action, evaluation.expectedValue)
              val blendedEv =
                ((1.0 - config.blendWeight) * evaluation.expectedValue) +
                  (config.blendWeight * baselineEv)
              ActionEvaluation(evaluation.action, blendedEv)
            }
            val best = blendedEvaluations.reduceLeft { (a, b) =>
              if a.expectedValue >= b.expectedValue then a else b
            }
            (
              baseRecommendation.copy(
                actionEvaluations = blendedEvaluations,
                bestAction = best.action
              ),
              Some(baseline)
            )
        catch
          case NonFatal(_) =>
            (baseRecommendation, None)
