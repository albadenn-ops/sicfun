package sicfun.holdem.validation

import sicfun.core.DiscreteDistribution
import sicfun.holdem.cfr.{HoldemCfrConfig, HoldemCfrSolver}
import sicfun.holdem.equity.{HoldemEquity, RangeParser}
import sicfun.holdem.types.*

import scala.util.Random

/** Pluggable GTO baseline strategy for the simulated villain. */
trait VillainStrategy:
  def decide(
      hand: HoleCards,
      state: GameState,
      candidates: Vector[PokerAction],
      equityVsRandom: Double,
      rng: Random
  ): PokerAction

/** Equity-threshold heuristic — the original HeadsUpSimulator strategy. */
final class EquityBasedStrategy extends VillainStrategy:
  def decide(
      hand: HoleCards,
      state: GameState,
      candidates: Vector[PokerAction],
      equityVsRandom: Double,
      rng: Random
  ): PokerAction =
    if candidates.size <= 1 then return candidates.headOption.getOrElse(PokerAction.Check)

    val potOdds = state.potOdds
    val street = state.street

    if state.toCall > 0 then
      val raiseActions = candidates.collect { case r: PokerAction.Raise => r }
      if street == Street.Preflop && state.position == Position.Button then
        if equityVsRandom >= 0.35 then
          if raiseActions.nonEmpty then raiseActions.head else PokerAction.Call
        else if equityVsRandom >= 0.20 && rng.nextDouble() < 0.4 then
          if raiseActions.nonEmpty then raiseActions.head else PokerAction.Call
        else PokerAction.Fold
      else if equityVsRandom >= 0.75 then
        if raiseActions.nonEmpty && rng.nextDouble() < 0.6 then raiseActions.head
        else PokerAction.Call
      else if equityVsRandom >= potOdds + 0.05 then
        if street == Street.Preflop && raiseActions.nonEmpty && rng.nextDouble() < 0.25 then
          raiseActions.head
        else if equityVsRandom >= 0.55 && raiseActions.nonEmpty && rng.nextDouble() < 0.20 then
          raiseActions.head
        else PokerAction.Call
      else if street == Street.Preflop && rng.nextDouble() < 0.70 then
        if raiseActions.nonEmpty && rng.nextDouble() < 0.20 then raiseActions.head
        else PokerAction.Call
      else if rng.nextDouble() < 0.25 then
        PokerAction.Call
      else PokerAction.Fold
    else
      if equityVsRandom >= 0.60 then
        val raiseActions = candidates.collect { case r: PokerAction.Raise => r }
        if raiseActions.nonEmpty then raiseActions(rng.nextInt(raiseActions.size))
        else PokerAction.Check
      else if equityVsRandom <= 0.30 && rng.nextDouble() < 0.25 then
        val raiseActions = candidates.collect { case r: PokerAction.Raise => r }
        if raiseActions.nonEmpty then raiseActions.head else PokerAction.Check
      else PokerAction.Check

/** CFR equilibrium strategy — solves each decision point via HoldemCfrSolver.
  *
  * Computes Nash equilibrium mixed strategy, then samples an action.
  * Slower than EquityBasedStrategy but produces actual equilibrium play.
  *
  * Uses street-appropriate opponent ranges: uniform preflop (opponent could
  * have anything), narrowed postflop (opponent continued with a preflop-viable
  * hand). Without narrowing, uniform postflop ranges make aggression dominant
  * because ~60% of random hands miss the flop and fold to any raise.
  */
final class CfrVillainStrategy(
    config: HoldemCfrConfig = HoldemCfrConfig(
      iterations = 500,
      equityTrials = 1_000,
      maxVillainHands = 64,
      includeVillainReraises = true
    ),
    allowHeuristicFallback: Boolean = true
) extends VillainStrategy:

  private[validation] def allowsHeuristicFallback: Boolean = allowHeuristicFallback

  // HU Button opening range — the hands opponent would have opened preflop.
  // Parsed once and reused across all decisions.
  private lazy val buttonOpenHands: Set[HoleCards] =
    RangeParser.parseWithHands(CfrVillainStrategy.HuButtonOpenRange) match
      case Right(result) => result.hands
      case Left(_) => Set.empty // fallback: will use fullRange

  def decide(
      hand: HoleCards,
      state: GameState,
      candidates: Vector[PokerAction],
      equityVsRandom: Double,
      rng: Random
  ): PokerAction =
    if candidates.size <= 1 then return candidates.headOption.getOrElse(PokerAction.Check)
    try
      val opponentRange = opponentRangeForStreet(hand, state)
      val policy = HoldemCfrSolver.solveDecisionPolicy(
        hero = hand,
        state = state,
        villainPosterior = opponentRange,
        candidateActions = candidates,
        config = config
      )
      sampleAction(policy.actionProbabilities, candidates, rng)
    catch
      case err: Exception =>
        if allowHeuristicFallback then
          EquityBasedStrategy().decide(hand, state, candidates, equityVsRandom, rng)
        else throw err

  /** Preflop: uniform range (opponent could hold anything).
    * Postflop: narrowed to hands the opponent would have opened/continued with
    * preflop, excluding dead cards. Falls back to uniform if narrowed range is
    * too small (< 20 hands).
    */
  private def opponentRangeForStreet(
      hand: HoleCards,
      state: GameState
  ): DiscreteDistribution[HoleCards] =
    if state.street == Street.Preflop || buttonOpenHands.isEmpty then
      HoldemEquity.fullRange(hand, state.board)
    else
      val dead = hand.asSet ++ state.board.asSet
      val viable = buttonOpenHands.filter(h => !h.toVector.exists(dead.contains)).toSeq
      if viable.size >= 20 then DiscreteDistribution.uniform(viable)
      else HoldemEquity.fullRange(hand, state.board)

  private def sampleAction(
      probs: Map[PokerAction, Double],
      candidates: Vector[PokerAction],
      rng: Random
  ): PokerAction =
    val roll = rng.nextDouble()
    val cumulativeProbs = candidates.scanLeft(0.0)((acc, a) => acc + probs.getOrElse(a, 0.0)).tail
    val idx = cumulativeProbs.indexWhere(_ > roll)
    if idx >= 0 then candidates(idx) else probs.maxBy(_._2)._1

object CfrVillainStrategy:
  // Standard HU Button opening range (~85% of hands).
  // Source: TableFormat.defaultRangeStringsHeadsUp(Position.Button)
  val HuButtonOpenRange: String =
    "22+, A2s+, K2s+, Q4s+, J6s+, T6s+, 96s+, 86s+, 76s, 65s, 54s, A2o+, K7o+, Q8o+, J8o+, T8o+, 98o"
