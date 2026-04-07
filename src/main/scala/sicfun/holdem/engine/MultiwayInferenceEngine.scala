package sicfun.holdem.engine

import sicfun.holdem.equity.*
import sicfun.holdem.model.*
import sicfun.holdem.types.*

import sicfun.core.{CollapseMetrics, DiscreteDistribution}

import scala.collection.mutable
import scala.util.Random

/** First-class multiway inferred-play helpers built from independent per-opponent posteriors.
  *
  * This object extends the heads-up inference engine to multiway pots (3+ players).
  * It does NOT implement a joint multi-opponent belief model (which would be exponentially
  * expensive). Instead, it uses the independence assumption:
  *
  *   1. '''Infer one posterior per live opponent''' via [[RangeInferenceEngine.inferPosterior]].
  *      Each opponent is modeled independently given the shared public state.
  *   2. '''Aggregate posteriors into multiway equity''' using either exact enumeration
  *      (for small ranges on turn/river) or Monte Carlo simulation.
  *   3. '''Re-evaluate raise branches''' by computing a raise-response estimate for each
  *      non-all-in opponent, then enumerating all 2^N fold/continue subsets to compute
  *      the expected EV across all possible response combinations.
  *
  * The subset enumeration (step 3) is the key design decision: for each raise candidate,
  * we consider every possible combination of opponents folding or continuing, weighted by
  * the product of their individual fold/continue probabilities. Each branch then gets its
  * own equity calculation against the continuing opponents' ranges.
  *
  * This is exponential in the number of opponents (2^N subsets) but practical for typical
  * poker table sizes (N <= 8 means 256 subsets maximum, which is manageable).
  */
object MultiwayInferenceEngine:
  /** Epsilon for money comparisons to avoid floating-point false zeros. */
  private inline val MoneyEpsilon = 1e-9
  /** Maximum exact evaluations before falling back to Monte Carlo for multiway equity. */
  val DefaultExactMaxEvaluations: Long = 250_000L

  /** Input descriptor for one opponent in a multiway pot.
    *
    * @param position the opponent's table position
    * @param folds preflop folds that occurred before this opponent acted (for bunching)
    * @param observations observed villain actions with their game states
    * @param stackSize the opponent's current stack in big blinds
    * @param contribution the opponent's chips already committed to the pot this street
    * @param isAllIn whether the opponent is all-in (won't respond to further raises)
    * @param posteriorOverride optional pre-computed posterior (bypasses inference)
    */
  final case class OpponentInput(
      position: Position,
      folds: Vector[PreflopFold],
      observations: Vector[VillainObservation],
      stackSize: Double,
      contribution: Double = 0.0,
      isAllIn: Boolean = false,
      posteriorOverride: Option[DiscreteDistribution[HoleCards]] = None
  ):
    require(stackSize >= 0.0 && stackSize.isFinite, "stackSize must be finite and non-negative")
    require(contribution >= 0.0 && contribution.isFinite, "contribution must be finite and non-negative")

  /** Combined result: per-opponent posteriors and the overall action recommendation. */
  final case class MultiwayInferenceResult(
      opponentPosteriors: Map[Position, PosteriorInferenceResult],
      recommendation: ActionRecommendation
  ):
    require(opponentPosteriors.nonEmpty, "opponentPosteriors must be non-empty")

  /** Estimated opponent response to a raise: fold/continue probabilities and the
    * narrowed continuation range (hands that would call or raise).
    */
  final case class RaiseResponseEstimate(
      foldProbability: Double,
      continueProbability: Double,
      continuationRange: DiscreteDistribution[HoleCards]
  ):
    require(foldProbability >= 0.0 && foldProbability <= 1.0, "foldProbability must be in [0, 1]")
    require(continueProbability >= 0.0 && continueProbability <= 1.0, "continueProbability must be in [0, 1]")

  /** Infers a posterior range for each opponent independently.
    *
    * For opponents with a `posteriorOverride`, uses the override directly.
    * For others, delegates to [[RangeInferenceEngine.inferPosterior]] with the
    * shared table ranges and action model.
    *
    * @return a map from position to posterior inference result
    */
  def inferOpponentPosteriors(
      hero: HoleCards,
      state: GameState,
      tableRanges: TableRanges,
      actionModel: PokerActionModel,
      opponents: Vector[OpponentInput],
      bunchingTrials: Int,
      rng: Random
  ): Map[Position, PosteriorInferenceResult] =
    require(opponents.nonEmpty, "opponents must be non-empty")
    require(bunchingTrials > 0, "bunchingTrials must be positive")
    requireDistinctPositions(opponents)
    opponents.map { opponent =>
      val result =
        opponent.posteriorOverride match
          case Some(overridePosterior) =>
            overridePosteriorResult(overridePosterior)
          case None =>
            RangeInferenceEngine.inferPosterior(
              hero = hero,
              board = state.board,
              folds = opponent.folds,
              tableRanges = tableRanges,
              villainPos = opponent.position,
              observations = opponent.observations,
              actionModel = actionModel,
              bunchingTrials = bunchingTrials,
              rng = new Random(rng.nextLong())
            )
      opponent.position -> result
    }.toMap

  /** End-to-end multiway decision: infer all opponent posteriors, compute base equity,
    * then re-evaluate raise branches with response-aware EV estimation.
    *
    * For non-raise actions, the base recommendation (equity * pot - cost) is used directly.
    * For raise actions, the EV is recomputed via [[raiseEvAgainstOpponents]] which
    * enumerates all 2^N fold/continue subsets of responsive opponents.
    *
    * @param equityTrialsForOpponentCount function mapping opponent count to equity trial budget
    * @return combined posteriors and action recommendation
    */
  def inferAndRecommend(
      hero: HoleCards,
      state: GameState,
      actorContribution: Double,
      actorBetHistoryIndex: Int,
      tableRanges: TableRanges,
      actionModel: PokerActionModel,
      opponents: Vector[OpponentInput],
      candidateActions: Vector[PokerAction],
      bunchingTrials: Int,
      equityTrialsForOpponentCount: Int => Int,
      rng: Random,
      actionValueModel: ActionValueModel = ActionValueModel.ChipEv(),
      exactMaxEvaluations: Long = DefaultExactMaxEvaluations
  ): MultiwayInferenceResult =
    require(candidateActions.nonEmpty, "candidateActions must be non-empty")
    require(actorContribution >= 0.0 && actorContribution.isFinite, "actorContribution must be finite and non-negative")
    require(actorBetHistoryIndex >= 0, "actorBetHistoryIndex must be non-negative")
    val opponentPosteriors = inferOpponentPosteriors(
      hero = hero,
      state = state,
      tableRanges = tableRanges,
      actionModel = actionModel,
      opponents = opponents,
      bunchingTrials = bunchingTrials,
      rng = new Random(rng.nextLong())
    )
    val opponentRangesByPosition = opponentPosteriors.view.mapValues(_.posterior).toMap
    val baseRecommendation = recommendActionAgainstOpponentRanges(
      hero = hero,
      state = state,
      opponentRanges = opponents.map(opponent => opponentRangesByPosition(opponent.position)),
      candidateActions = candidateActions,
      equityTrials = equityTrialsForOpponentCount(opponents.length),
      rng = new Random(rng.nextLong()),
      actionValueModel = actionValueModel,
      exactMaxEvaluations = exactMaxEvaluations
    )
    val evaluations = baseRecommendation.actionEvaluations.map { evaluation =>
      evaluation.action match
        case PokerAction.Raise(amount) =>
          evaluation.copy(
            expectedValue = raiseEvAgainstOpponents(
              hero = hero,
              state = state,
              actorContribution = actorContribution,
              actorBetHistoryIndex = actorBetHistoryIndex,
              raiseAmount = amount,
              opponents = opponents,
              opponentRangesByPosition = opponentRangesByPosition,
              actionModel = actionModel,
              equityTrialsForOpponentCount = equityTrialsForOpponentCount,
              rng = new Random(rng.nextLong()),
              exactMaxEvaluations = exactMaxEvaluations
            )
          )
        case _ => evaluation
    }
    val best = evaluations.reduceLeft { (a, b) =>
      if a.expectedValue >= b.expectedValue then a else b
    }
    MultiwayInferenceResult(
      opponentPosteriors = opponentPosteriors,
      recommendation = baseRecommendation.copy(
        actionEvaluations = evaluations,
        bestAction = best.action
      )
    )

  /** Estimates how an opponent's range responds to a raise.
    *
    * For each hand in the range, queries the action model for fold/call/raise likelihoods,
    * normalizes to legal actions, and aggregates into weighted fold/continue probabilities.
    * Hands that continue form the continuation range (a narrower distribution biased
    * toward hands that call or raise rather than fold).
    *
    * @param range the opponent's current posterior range
    * @param responseState the game state the opponent faces after the raise
    * @param actionModel the model predicting action likelihoods per hand
    * @param raiseAction the raise action being evaluated
    * @return fold/continue probabilities and the continuation range
    */
  def estimateRaiseResponseFromRange(
      range: DiscreteDistribution[HoleCards],
      responseState: GameState,
      actionModel: PokerActionModel,
      raiseAction: PokerAction
  ): RaiseResponseEstimate =
    require(responseState.toCall > MoneyEpsilon, "responseState must face a bet or raise")
    val boardDead = responseState.board.asSet
    val filteredWeights = range.weights.collect {
      case (hand, probability)
          if probability > 0.0 &&
            !boardDead.contains(hand.first) &&
            !boardDead.contains(hand.second) =>
        hand -> probability
    }
    val effectiveRange =
      if filteredWeights.nonEmpty then DiscreteDistribution(filteredWeights).normalized
      else range.normalized
    val continuationWeights = mutable.Map.empty[HoleCards, Double].withDefaultValue(0.0)
    var foldProbability = 0.0
    var continueProbability = 0.0
    effectiveRange.weights.foreach { case (hand, priorProbability) =>
      if priorProbability > 0.0 then
        val rawFold = actionModel.likelihood(PokerAction.Fold, responseState, hand)
        val rawCall = actionModel.likelihood(PokerAction.Call, responseState, hand)
        val rawRaise = actionModel.likelihood(raiseAction, responseState, hand)
        val legalMass = rawFold + rawCall + rawRaise
        val normalizedFold =
          if legalMass > MoneyEpsilon then rawFold / legalMass
          else 0.0
        val normalizedContinue =
          if legalMass > MoneyEpsilon then (rawCall + rawRaise) / legalMass
          else 1.0
        val weightedFold = priorProbability * normalizedFold
        val weightedContinue = priorProbability * normalizedContinue
        foldProbability += weightedFold
        continueProbability += weightedContinue
        if weightedContinue > 0.0 then
          continuationWeights.update(hand, continuationWeights(hand) + weightedContinue)
    }
    val totalMass = foldProbability + continueProbability
    val normalizedFoldProbability =
      if totalMass > MoneyEpsilon then clamp(foldProbability / totalMass)
      else 0.0
    val normalizedContinueProbability =
      if totalMass > MoneyEpsilon then clamp(continueProbability / totalMass)
      else 1.0
    val continuationRange =
      if continuationWeights.values.exists(_ > 0.0) then
        DiscreteDistribution(continuationWeights.toMap).normalized
      else effectiveRange
    RaiseResponseEstimate(
      foldProbability = normalizedFoldProbability,
      continueProbability = normalizedContinueProbability,
      continuationRange = continuationRange
    )

  /** Estimates hero equity against multiple opponent ranges.
    *
    * Strategy selection:
    *   - Single opponent: delegates to [[HoldemEquity.equityMonteCarlo]].
    *   - Multiple opponents with <= 2 missing board cards: tries exact multiway enumeration
    *     first, falls back to Monte Carlo if the combinatorial space exceeds maxEvaluations.
    *   - Multiple opponents with 3+ missing board cards: Monte Carlo only.
    *
    * @param opponentRanges one range distribution per live opponent
    * @param equityTrials Monte Carlo trial count (used when exact is infeasible)
    * @param exactMaxEvaluations threshold for switching from exact to Monte Carlo
    * @return equity estimate with mean, variance, and win/tie/loss rates
    */
  def estimateEquityAgainstOpponentRanges(
      hero: HoleCards,
      board: Board,
      opponentRanges: Seq[DiscreteDistribution[HoleCards]],
      equityTrials: Int,
      rng: Random,
      exactMaxEvaluations: Long = DefaultExactMaxEvaluations
  ): EquityEstimate =
    require(opponentRanges.nonEmpty, "opponentRanges must be non-empty")
    require(equityTrials > 0, "equityTrials must be positive")
    if opponentRanges.lengthCompare(1) == 0 then
      HoldemEquity.equityMonteCarlo(
        hero = hero,
        board = board,
        villainRange = opponentRanges.head,
        trials = equityTrials,
        rng = rng
      )
    else if board.missing <= 2 then
      try
        val exact = HoldemEquity.equityExactMulti(
          hero = hero,
          board = board,
          villainRanges = opponentRanges,
          maxEvaluations = exactMaxEvaluations
        )
        EquityEstimate(
          mean = exact.share,
          variance = 0.0,
          stderr = 0.0,
          trials = 1,
          winRate = exact.win,
          tieRate = exact.tie,
          lossRate = exact.loss
        )
      catch
        case _: IllegalArgumentException =>
          HoldemEquity.equityMonteCarloMulti(
            hero = hero,
            board = board,
            villainRanges = opponentRanges,
            trials = equityTrials,
            rng = rng
          )
    else
      HoldemEquity.equityMonteCarloMulti(
        hero = hero,
        board = board,
        villainRanges = opponentRanges,
        trials = equityTrials,
        rng = rng
      )

  /** Ranks candidate actions by EV against multiple opponent ranges (base path, no raise response).
    *
    * Computes hero equity once against the aggregate opponent ranges, then evaluates each
    * candidate action via the action value model (typically chip-EV).
    */
  def recommendActionAgainstOpponentRanges(
      hero: HoleCards,
      state: GameState,
      opponentRanges: Seq[DiscreteDistribution[HoleCards]],
      candidateActions: Vector[PokerAction],
      equityTrials: Int,
      rng: Random,
      actionValueModel: ActionValueModel = ActionValueModel.ChipEv(),
      exactMaxEvaluations: Long = DefaultExactMaxEvaluations
  ): ActionRecommendation =
    require(candidateActions.nonEmpty, "candidateActions must be non-empty")
    val heroEquity = estimateEquityAgainstOpponentRanges(
      hero = hero,
      board = state.board,
      opponentRanges = opponentRanges,
      equityTrials = equityTrials,
      rng = rng,
      exactMaxEvaluations = exactMaxEvaluations
    )
    val evaluations = candidateActions.map { action =>
      ActionEvaluation(
        action = action,
        expectedValue = actionValueModel.expectedValue(action, state, heroEquity.mean)
      )
    }
    val best = evaluations.reduceLeft { (a, b) =>
      if a.expectedValue >= b.expectedValue then a else b
    }
    ActionRecommendation(
      heroEquity = heroEquity,
      actionEvaluations = evaluations,
      bestAction = best.action
    )

  /** Computes raise EV by enumerating all 2^N fold/continue subsets of responsive opponents.
    *
    * For each subset (bitmask), computes the branch probability (product of individual
    * fold/continue probabilities), the pot size (base pot + actor contribution + continuing
    * opponents' contributions), and equity against the continuing opponents' continuation
    * ranges. All-in opponents are always included in the continuation set.
    *
    * Branch EV = equity * continuePot - actorAdded (for branches with live opponents)
    *           = pot (for branches where all responsive opponents fold)
    *
    * Total EV = sum over all subsets of (branchProbability * branchEV).
    */
  private def raiseEvAgainstOpponents(
      hero: HoleCards,
      state: GameState,
      actorContribution: Double,
      actorBetHistoryIndex: Int,
      raiseAmount: Double,
      opponents: Vector[OpponentInput],
      opponentRangesByPosition: Map[Position, DiscreteDistribution[HoleCards]],
      actionModel: PokerActionModel,
      equityTrialsForOpponentCount: Int => Int,
      rng: Random,
      exactMaxEvaluations: Long
  ): Double =
    val actorAdded = roundMoney(math.min(state.toCall + raiseAmount, state.stackSize))
    val lockedOpponents = opponents.filter(_.isAllIn)
    val lockedRanges = lockedOpponents.map(opponent => opponentRangesByPosition(opponent.position))
    val responsiveOpponents = opponents.filterNot(_.isAllIn)
    val raiseProxyAction = PokerAction.Raise(raiseAmount)
    val responses = responsiveOpponents.map { opponent =>
      val responseState = responseStateForOpponentFacingRaise(
        actorContribution = actorContribution,
        actorAdded = actorAdded,
        actorBetHistoryIndex = actorBetHistoryIndex,
        opponent = opponent,
        state = state,
        raiseAmount = raiseAmount
      )
      val estimate = estimateRaiseResponseFromRange(
        range = opponentRangesByPosition(opponent.position),
        responseState = responseState,
        actionModel = actionModel,
        raiseAction = raiseProxyAction
      )
      val continueContribution = roundMoney(math.min(responseState.toCall, opponent.stackSize))
      (opponent, estimate, continueContribution)
    }

    val subsetCount = 1 << responses.length
    var totalEv = 0.0
    var mask = 0
    while mask < subsetCount do
      var branchProbability = 1.0
      var branchContribution = 0.0
      val continuationRanges = mutable.ArrayBuffer.empty[DiscreteDistribution[HoleCards]]
      continuationRanges ++= lockedRanges
      var idx = 0
      while idx < responses.length do
        val (opponent, estimate, continueContribution) = responses(idx)
        val continues = ((mask >> idx) & 1) == 1
        if continues then
          branchProbability *= estimate.continueProbability
          branchContribution += continueContribution
          continuationRanges += estimate.continuationRange
        else
          branchProbability *= estimate.foldProbability
        idx += 1
      if branchProbability > MoneyEpsilon then
        val branchEv =
          if continuationRanges.isEmpty then state.pot
          else
            val continuePot = roundMoney(state.pot + actorAdded + branchContribution)
            val continuationEquity = estimateEquityAgainstOpponentRanges(
              hero = hero,
              board = state.board,
              opponentRanges = continuationRanges.toVector,
              equityTrials = equityTrialsForOpponentCount(continuationRanges.length),
              rng = new Random(rng.nextLong()),
              exactMaxEvaluations = exactMaxEvaluations
            ).mean
            (continuationEquity * continuePot) - actorAdded
        totalEv += branchProbability * branchEv
      mask += 1
    totalEv

  /** Constructs the GameState that an opponent would face after the hero raises.
    * Adjusts pot, toCall, and bet history to reflect the hero's raise action.
    */
  private def responseStateForOpponentFacingRaise(
      actorContribution: Double,
      actorAdded: Double,
      actorBetHistoryIndex: Int,
      opponent: OpponentInput,
      state: GameState,
      raiseAmount: Double
  ): GameState =
    val targetContribution = roundMoney(actorContribution + actorAdded)
    val opponentToCall = contributionGap(targetContribution, opponent.contribution)
    GameState(
      street = state.street,
      board = state.board,
      pot = roundMoney(state.pot + actorAdded),
      toCall = opponentToCall,
      position = opponent.position,
      stackSize = opponent.stackSize,
      betHistory = state.betHistory :+ BetAction(actorBetHistoryIndex, PokerAction.Raise(raiseAmount))
    )

  /** Wraps a user-provided override posterior into a PosteriorInferenceResult with
    * zero collapse diagnostics (since no inference was actually performed).
    */
  private def overridePosteriorResult(
      posterior: DiscreteDistribution[HoleCards]
  ): PosteriorInferenceResult =
    val normalized = posterior.normalized
    val support = CollapseMetrics.effectiveSupport(normalized)
    PosteriorInferenceResult(
      prior = normalized,
      posterior = normalized,
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

  private def requireDistinctPositions(opponents: Vector[OpponentInput]): Unit =
    val positions = opponents.map(_.position)
    require(positions.distinct.length == positions.length, "opponents must have distinct positions")

  private def contributionGap(targetContribution: Double, currentContribution: Double): Double =
    math.max(0.0, targetContribution - currentContribution)

  private def clamp(value: Double, lo: Double = 0.0, hi: Double = 1.0): Double =
    math.max(lo, math.min(hi, value))

  private def roundMoney(value: Double): Double =
    math.round(value * 100.0) / 100.0
