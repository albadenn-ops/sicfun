package sicfun.holdem.strategic.safety

import sicfun.core.Card
import sicfun.holdem.types.{Board, Street}
import sicfun.holdem.strategic.state.PublicState

/** Coarse board-texture bucket for baseline conditioning (P1).
  *
  * Postflop: pairing × suitedness × high-card. Preflop: a single `Preflop` token (v1 —
  * preflop is not per-class calibrated; it backs off to the (class,action) constants).
  * Deliberately coarse to keep artifact cells populated.
  */
enum BoardBucket:
  case Preflop
  case Postflop(
      pairing: BoardBucket.Pairing,
      suitedness: BoardBucket.Suitedness,
      highCard: BoardBucket.HighCard
  )

object BoardBucket:
  enum Pairing    { case Unpaired, Paired, Trips }
  enum Suitedness { case Rainbow, TwoTone, Monotone }
  enum HighCard   { case AceHigh, Broadway, Middle, Low }

  /** Stable token used as the artifact key segment and backoff key. */
  def token(b: BoardBucket): String = b match
    case Preflop                 => "PRE"
    case Postflop(p, s, h)       => s"$p-$s-$h"

  /** Bucket from a full public state (uses its street + board). */
  def of(publicState: PublicState): BoardBucket =
    ofBoard(publicState.street, publicState.board)

  def ofBoard(street: Street, board: Board): BoardBucket =
    if street == Street.Preflop || board.cards.isEmpty then Preflop
    else Postflop(pairingOf(board.cards), suitednessOf(board.cards), highCardOf(board.cards))

  private def pairingOf(cards: Vector[Card]): Pairing =
    val maxRankCount = cards.groupBy(_.rank).values.map(_.size).maxOption.getOrElse(0)
    if maxRankCount >= 3 then Pairing.Trips
    else if maxRankCount == 2 then Pairing.Paired
    else Pairing.Unpaired

  private def suitednessOf(cards: Vector[Card]): Suitedness =
    val maxSuitCount = cards.groupBy(_.suit).values.map(_.size).maxOption.getOrElse(0)
    if maxSuitCount >= 3 then Suitedness.Monotone
    else if maxSuitCount == 2 then Suitedness.TwoTone
    else Suitedness.Rainbow

  private def highCardOf(cards: Vector[Card]): HighCard =
    val top = cards.map(_.rank.value).maxOption.getOrElse(0)
    if top >= 14 then HighCard.AceHigh        // Ace
    else if top >= 11 then HighCard.Broadway  // J,Q,K (Ten counts as Middle)
    else if top >= 7 then HighCard.Middle     // 7..10
    else HighCard.Low                          // 2..6
