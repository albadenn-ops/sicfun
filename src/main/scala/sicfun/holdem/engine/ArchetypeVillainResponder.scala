package sicfun.holdem.engine

import sicfun.holdem.types.*

import scala.util.Random

/** Archetype-based villain decision model with fixed style profiles.
  *
  * This object simulates villain decisions in the playing hall by combining a
  * noisy hand-strength estimate (from [[HandStrengthEstimator]]) with archetype-specific
  * looseness and aggression parameters. It is used in simulation/match-runner contexts
  * where the engine needs to generate realistic villain actions without a full solver.
  *
  * Extracted from TexasHoldemPlayingHall. The original code embedded these heuristics
  * inline; this extraction makes them testable and reusable across match runners.
  *
  * Design notes:
  *   - The style profiles (looseness, aggression) are hand-tuned constants, not learned.
  *   - Decision thresholds incorporate pot odds and are randomized via RNG to produce
  *     mixed strategies (not purely deterministic).
  *   - The model is intentionally simple: it does not consider draw equity, board texture,
  *     or multi-street planning. It just needs to produce "reasonable" villain behavior
  *     for simulation fidelity.
  *
  * Note: PlayerArchetype is defined in the same package (sicfun.holdem.engine)
  * in RealTimeAdaptiveEngine.scala, so it's accessible without explicit import.
  */
private[holdem] object ArchetypeVillainResponder:

  /** Two-dimensional style profile characterizing a villain archetype.
    *
    * @param looseness how wide the villain's continuing range is (0 = ultra-tight, 1 = plays everything)
    * @param aggression how often the villain bets/raises vs. checks/calls (0 = passive, 1 = hyper-aggressive)
    */
  final case class VillainStyleProfile(looseness: Double, aggression: Double)

  /** Maps each PlayerArchetype to its fixed (looseness, aggression) style profile.
    *
    * @param archetype the villain archetype
    * @return the corresponding style profile with looseness and aggression in [0, 1]
    */
  def styleProfile(archetype: PlayerArchetype): VillainStyleProfile =
    archetype match
      case PlayerArchetype.Nit            => VillainStyleProfile(looseness = 0.20, aggression = 0.18)
      case PlayerArchetype.Tag            => VillainStyleProfile(looseness = 0.45, aggression = 0.40)
      case PlayerArchetype.Lag            => VillainStyleProfile(looseness = 0.68, aggression = 0.66)
      case PlayerArchetype.CallingStation => VillainStyleProfile(looseness = 0.86, aggression = 0.24)
      case PlayerArchetype.Maniac         => VillainStyleProfile(looseness = 0.80, aggression = 0.92)

  /** Generates a villain action given their hand, archetype, and game state.
    *
    * The decision logic has two branches:
    *
    * '''Check-to-act (toCall <= 0):'''
    *   - If raises are not allowed, always checks.
    *   - Otherwise, computes a bet probability from hand strength and aggression,
    *     then randomly decides between raising and checking.
    *
    * '''Facing a bet (toCall > 0):'''
    *   - Computes a fold threshold from pot odds adjusted by looseness (looser players
    *     have a lower fold threshold, meaning they continue more often).
    *   - Computes a raise threshold from aggression and pot odds.
    *   - If strength exceeds the raise threshold AND a random roll passes, raises.
    *   - If strength is below the fold threshold, folds.
    *   - Otherwise, calls.
    *
    * @param hand the villain's hole cards
    * @param style the villain's archetype (determines looseness and aggression)
    * @param state the current game state (pot, toCall, board, street, etc.)
    * @param allowRaise whether the villain is allowed to raise (may be false due to bet cap)
    * @param raiseSize the raise amount in big blinds if the villain decides to raise
    * @param rng random number generator for mixed-strategy randomization
    * @return the chosen PokerAction (Check, Fold, Call, or Raise)
    */
  def villainResponds(
      hand: HoleCards,
      style: PlayerArchetype,
      state: GameState,
      allowRaise: Boolean,
      raiseSize: Double,
      rng: Random
  ): PokerAction =
    val profile = styleProfile(style)
    // Noisy hand strength estimate with RNG jitter for realistic variance
    val strength = HandStrengthEstimator.streetStrength(hand, state.board, state.street, rng)

    if state.toCall <= 0.0 then
      // Check-to-act: villain is first to act or facing a check
      if !allowRaise then PokerAction.Check
      else
        // Bet probability: higher for strong hands and aggressive archetypes
        // The formula shifts the curve left for aggressive players (more bets with weaker hands)
        val betChance = HandStrengthEstimator.clamp((strength - 0.35) * 0.9 + (profile.aggression * 0.35), 0.02, 0.92)
        if rng.nextDouble() < betChance then PokerAction.Raise(raiseSize)
        else PokerAction.Check
    else
      // Facing a bet: decide fold/call/raise based on strength vs. thresholds
      val potOdds = state.potOdds
      // Fold threshold: tighter for passive/tight players, looser for loose players
      // Looseness reduces the threshold, making the villain more likely to continue
      val foldThreshold = HandStrengthEstimator.clamp(potOdds + (0.35 - profile.looseness * 0.2), 0.05, 0.95)
      // Raise threshold: lower for aggressive players (they raise with weaker hands)
      val raiseThreshold = HandStrengthEstimator.clamp(0.68 - profile.aggression * 0.12 + potOdds * 0.2, 0.35, 0.9)
      // Raise branch: must pass both a strength gate and a random aggression roll
      if allowRaise && strength >= raiseThreshold && rng.nextDouble() < (0.15 + profile.aggression * 0.55) then
        PokerAction.Raise(raiseSize)
      else if strength < foldThreshold then PokerAction.Fold
      else PokerAction.Call
