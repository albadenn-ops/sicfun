package sicfun.holdem.types

/** The four betting rounds (streets) in Texas Hold'em, in temporal order.
  *
  * Ordinal values (0-3) are used as normalized features in ML pipelines.
  */
enum Street(val expectedBoardSize: Int):
  case Preflop extends Street(0)
  case Flop    extends Street(3)
  case Turn    extends Street(4)
  case River   extends Street(5)

/** Table position of a player, ordered from earliest to latest to act.
  *
  * Ordinal values (0-7) are normalized to [0, 1] when used as features.
  */
enum Position:
  case SmallBlind, BigBlind, UTG, UTG1, UTG2, Middle, Cutoff, Button

/** A single recorded action within a betting round, attributed to a player index.
  *
  * @param player zero-based seat index of the acting player
  * @param action the poker action taken
  */
final case class BetAction(player: Int, action: PokerAction)

/** Immutable snapshot of the observable game state at a player's decision point.
  *
  * Contains all information available to a player when they must act: the current
  * street, community cards, pot size, amount required to call, seat position,
  * remaining stack, and the sequence of prior bets in the current hand.
  *
  * Derived metrics [[potOdds]] and [[stackToPot]] are computed lazily for use
  * in feature extraction and decision logic.
  *
  * @param street      current betting round
  * @param board       community cards revealed so far
  * @param pot         total chips in the pot before the player acts
  * @param toCall      additional chips the player must contribute to call (0 if checking is allowed)
  * @param position    the player's table position
  * @param stackSize   the player's remaining stack (chips not yet committed)
  * @param betHistory  chronological sequence of all bets in this hand so far
  */
final case class GameState(
    street: Street,
    board: Board,
    pot: Double,
    toCall: Double,
    position: Position,
    stackSize: Double,
    betHistory: Vector[BetAction]
):
  require(pot >= 0.0, "pot must be non-negative")
  require(toCall >= 0.0, "toCall must be non-negative")
  require(stackSize >= 0.0, "stackSize must be non-negative")

  /** The fraction of the new pot represented by the call amount: `toCall / (pot + toCall)`.
    *
    * Returns 0.0 when there is nothing to call (i.e., a check situation).
    * This metric indicates the price being offered by the pot; lower values
    * mean a more favorable call.
    */
  inline def potOdds: Double =
    if toCall <= 0.0 then 0.0
    else toCall / (pot + toCall)

  /** Ratio of the player's remaining stack to the current pot: `stackSize / pot`.
    *
    * Returns [[Double.PositiveInfinity]] when the pot is zero or negative
    * (e.g., before blinds are posted). Higher values indicate deeper stacks
    * relative to the pot, allowing more post-flop manoeuvrability.
    */
  inline def stackToPot: Double =
    if pot <= 0.0 then Double.PositiveInfinity
    else stackSize / pot
