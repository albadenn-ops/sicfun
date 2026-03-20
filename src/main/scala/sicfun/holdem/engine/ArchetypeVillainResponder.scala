package sicfun.holdem.engine

import sicfun.holdem.types.*

import scala.util.Random

/** Archetype-based villain decision model with fixed style profiles.
  *
  * Extracted from TexasHoldemPlayingHall. Uses HandStrengthEstimator for
  * hand evaluation and applies archetype-specific betting heuristics.
  *
  * Note: PlayerArchetype is defined in the same package (sicfun.holdem.engine)
  * in RealTimeAdaptiveEngine.scala, so it's accessible without explicit import.
  */
private[holdem] object ArchetypeVillainResponder:

  final case class VillainStyleProfile(looseness: Double, aggression: Double)

  def styleProfile(archetype: PlayerArchetype): VillainStyleProfile =
    archetype match
      case PlayerArchetype.Nit            => VillainStyleProfile(looseness = 0.20, aggression = 0.18)
      case PlayerArchetype.Tag            => VillainStyleProfile(looseness = 0.45, aggression = 0.40)
      case PlayerArchetype.Lag            => VillainStyleProfile(looseness = 0.68, aggression = 0.66)
      case PlayerArchetype.CallingStation => VillainStyleProfile(looseness = 0.86, aggression = 0.24)
      case PlayerArchetype.Maniac         => VillainStyleProfile(looseness = 0.80, aggression = 0.92)

  def villainResponds(
      hand: HoleCards,
      style: PlayerArchetype,
      state: GameState,
      allowRaise: Boolean,
      raiseSize: Double,
      rng: Random
  ): PokerAction =
    val profile = styleProfile(style)
    val strength = HandStrengthEstimator.streetStrength(hand, state.board, state.street, rng)

    if state.toCall <= 0.0 then
      if !allowRaise then PokerAction.Check
      else
        val betChance = HandStrengthEstimator.clamp((strength - 0.35) * 0.9 + (profile.aggression * 0.35), 0.02, 0.92)
        if rng.nextDouble() < betChance then PokerAction.Raise(raiseSize)
        else PokerAction.Check
    else
      val potOdds = state.potOdds
      val foldThreshold = HandStrengthEstimator.clamp(potOdds + (0.35 - profile.looseness * 0.2), 0.05, 0.95)
      val raiseThreshold = HandStrengthEstimator.clamp(0.68 - profile.aggression * 0.12 + potOdds * 0.2, 0.35, 0.9)
      if allowRaise && strength >= raiseThreshold && rng.nextDouble() < (0.15 + profile.aggression * 0.55) then
        PokerAction.Raise(raiseSize)
      else if strength < foldThreshold then PokerAction.Fold
      else PokerAction.Call
