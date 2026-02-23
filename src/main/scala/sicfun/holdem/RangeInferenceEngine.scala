package sicfun.holdem

import sicfun.core.{BayesianRange, CollapseMetrics, DiscreteDistribution}

import scala.collection.mutable
import scala.util.Random

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
final case class PosteriorInferenceResult(
    prior: DiscreteDistribution[HoleCards],
    posterior: DiscreteDistribution[HoleCards],
    logEvidence: Double,
    collapse: PosteriorCollapse
)

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
  */
object RangeInferenceEngine:
  /** Computes a posterior villain range from folds + observed actions. */
  def inferPosterior(
      hero: HoleCards,
      board: Board,
      folds: Vector[PreflopFold],
      tableRanges: TableRanges,
      villainPos: Position,
      observations: Seq[VillainObservation],
      actionModel: PokerActionModel,
      bunchingTrials: Int = 10_000,
      rng: Random = new Random()
  ): PosteriorInferenceResult =
    require(bunchingTrials > 0, "bunchingTrials must be positive")

    val prior = BunchingEffect.adjustedRange(
      hero = hero,
      board = board,
      folds = folds,
      tableRanges = tableRanges,
      villainPos = villainPos,
      trials = bunchingTrials,
      rng = rng
    )

    val bayes = BayesianRange(prior)
    val (updated, logEvidence) = bayes.updateAll(
      observations.map(o => o.action -> o.state),
      actionModel
    )
    val posterior = updated.distribution.normalized

    val collapse = PosteriorCollapse(
      entropyReduction = CollapseMetrics.entropyReduction(prior, posterior),
      klDivergence = CollapseMetrics.klDivergence(prior, posterior),
      effectiveSupportPrior = CollapseMetrics.effectiveSupport(prior),
      effectiveSupportPosterior = CollapseMetrics.effectiveSupport(posterior),
      collapseRatio = CollapseMetrics.collapseRatio(prior, posterior)
    )
    PosteriorInferenceResult(prior, posterior, logEvidence, collapse)

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

    val heroEquity = HoldemEquity.equityMonteCarlo(
      hero = hero,
      board = state.board,
      villainRange = posterior,
      trials = equityTrials,
      rng = rng
    )

    val evaluations = candidateActions.map { action =>
      val expectedValue =
        villainResponseModel match
          case Some(responseModel) =>
            action match
              case PokerAction.Raise(_) =>
                responseAwareRaiseEv(
                  hero = hero,
                  state = state,
                  posterior = posterior,
                  action = action,
                  responseModel = responseModel,
                  equityTrials = equityTrials,
                  rng = new Random(rng.nextLong())
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

  private val Eps = 1e-12

  private def responseAwareRaiseEv(
      hero: HoleCards,
      state: GameState,
      posterior: DiscreteDistribution[HoleCards],
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

    val (foldProbability, continueWeights) = aggregateResponses(
      posterior = posterior,
      state = state,
      action = action,
      responseModel = responseModel
    )
    val continueProbability = continueWeights.values.sum

    if continueProbability <= Eps then state.pot
    else
      val continuationRange = DiscreteDistribution(continueWeights).normalized
      val continuationEquity = HoldemEquity.equityMonteCarlo(
        hero = hero,
        board = state.board,
        villainRange = continuationRange,
        trials = equityTrials,
        rng = rng
      )
      // Approximate one-street raise EV: win current pot on folds,
      // otherwise equity realization in a called pot where both players
      // contribute the raise amount.
      val continuePot = state.pot + 2.0 * raiseAmount
      val continueEv = (continuationEquity.mean * continuePot) - raiseAmount
      (foldProbability * state.pot) + (continueProbability * continueEv)

  private def aggregateResponses(
      posterior: DiscreteDistribution[HoleCards],
      state: GameState,
      action: PokerAction,
      responseModel: VillainResponseModel
  ): (Double, Map[HoleCards, Double]) =
    val normPosterior = posterior.normalized
    val continueWeights = mutable.Map.empty[HoleCards, Double].withDefaultValue(0.0)
    var foldProbability = 0.0

    normPosterior.weights.foreach { case (hand, priorProb) =>
      if priorProb > 0.0 then
        val profile = responseModel.response(hand, state, action)
        foldProbability += priorProb * profile.foldProbability
        val continueProb = profile.continueProbability
        if continueProb > 0.0 then
          continueWeights.update(hand, continueWeights(hand) + priorProb * continueProb)
    }
    (math.max(0.0, math.min(1.0, foldProbability)), continueWeights.toMap)

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

    val recommendation = recommendAction(
      hero = hero,
      state = state,
      posterior = posteriorInference.posterior,
      candidateActions = candidateActions,
      actionValueModel = actionValueModel,
      villainResponseModel = villainResponseModel,
      equityTrials = equityTrials,
      rng = new Random(rng.nextLong())
    )

    InferenceDecisionResult(posteriorInference, recommendation)
