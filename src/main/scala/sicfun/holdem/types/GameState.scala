package sicfun.holdem.types

/**
  * Core game-state types for Texas Hold'em hand representation.
  *
  * This file defines the foundational domain model used throughout the sicfun poker analytics
  * system: the [[Street]] and [[Position]] enumerations that classify where in a hand and
  * at the table a decision occurs, the [[BetAction]] record that captures a single bet
  * attribution, and the central [[GameState]] snapshot that encapsulates all observable
  * information available to a player at a decision point.
  *
  * [[GameState]] is the primary input to feature extraction ([[sicfun.holdem.model.PokerFeatures]]),
  * action prediction models ([[sicfun.holdem.model.PokerActionModel]]), and EV analysis
  * pipelines. It is intentionally immutable and self-contained so that it can be serialized,
  * cached, and replayed without side effects.
  *
  * Design decisions:
  *   - All monetary values (pot, toCall, stackSize) are `Double` rather than `BigDecimal`
  *     because ML feature pipelines and equity calculations operate in floating-point space,
  *     and the sub-cent precision loss is irrelevant for poker chip amounts.
  *   - Derived metrics (`potOdds`, `stackToPot`) are `inline def` rather than stored fields
  *     to avoid stale-value bugs when constructing modified copies via `.copy(...)`.
  *   - `betHistory` is a `Vector[BetAction]` (not `List`) for efficient indexed access
  *     during feature extraction, where the number of prior bets matters.
  */

/** The four betting rounds (streets) in Texas Hold'em, in temporal order.
  *
  * Each street carries an `expectedBoardSize` that specifies how many community cards
  * should be visible during that round. This is used by [[PokerEvent]] to validate
  * that the board state is consistent with the street.
  *
  * Ordinal values (0-3) are used as normalized features in ML pipelines,
  * where they are divided by 3.0 to produce values in [0, 1].
  *
  * @param expectedBoardSize the number of community cards dealt by this street
  */
enum Street(val expectedBoardSize: Int):
  /** Pre-deal betting round; no community cards visible. */
  case Preflop extends Street(0)
  /** First three community cards are dealt. */
  case Flop    extends Street(3)
  /** Fourth community card is dealt. */
  case Turn    extends Street(4)
  /** Fifth and final community card is dealt. */
  case River   extends Street(5)

/** Table position of a player, ordered from earliest to latest to act.
  *
  * Positions are enumerated in standard 9-max table order. The ordinal values (0-8)
  * are normalized to [0, 1] when used as features in ML pipelines (ordinal / 8.0).
  *
  * Early positions (UTG, UTG1, UTG2) act first post-flop and require tighter ranges.
  * Late positions (Cutoff, Button) act last and can profitably play wider ranges.
  * Blinds (SmallBlind, BigBlind) post forced bets but act last preflop.
  */
enum Position:
  case SmallBlind, BigBlind, UTG, UTG1, UTG2, Middle, Hijack, Cutoff, Button

/** A single recorded action within a betting round, attributed to a player index.
  *
  * Used within [[GameState.betHistory]] to maintain a chronological log of all
  * actions taken in the current hand. The `player` index is zero-based and refers
  * to the seat position at the table (not to be confused with [[Position]],
  * which describes the strategic position like Button or BigBlind).
  *
  * @param player zero-based seat index of the acting player
  * @param action the poker action taken (fold, check, call, or raise)
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
