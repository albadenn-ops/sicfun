package sicfun.holdem.validation

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
