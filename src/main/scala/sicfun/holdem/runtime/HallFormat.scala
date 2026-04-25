package sicfun.holdem.runtime

import sicfun.core.Card
import sicfun.holdem.types.{Board, PokerAction, Position}

import java.util.Locale

/** Pure formatting helpers shared across the playing hall surface (TSV writers,
  * PokerStars-format hand history exporter, learning log).
  *
  * Carved out of [[TexasHoldemPlayingHall]] as the first F6 split slice — these
  * are the smallest most-leveraged stateless utilities the monolith depends on,
  * so giving them their own module unblocks any subsequent extraction (TSV
  * writers, hand-history exporter) that reads them.
  */
object HallFormat:
  /** Format a double with the given decimal-place count, locale-independent. */
  def fmt(value: Double, digits: Int): String =
    String.format(Locale.ROOT, s"%.${digits}f", java.lang.Double.valueOf(value))

  /** Round to two decimal places (cent precision) for currency display. */
  def roundMoney(value: Double): Double =
    math.round(value * 100.0) / 100.0

  /** Format a chip amount as a dollar string, e.g. `"$1.50"`. */
  def money(amount: Double): String = s"$$${fmt(roundMoney(amount), 2)}"

  /** Wrap a card sequence in PokerStars-style square brackets, e.g. `"[Ah Ks]"`. */
  def bracketedCards(cards: Seq[Card]): String =
    s"[${cards.map(_.toToken).mkString(" ")}]"

  /** TSV-friendly board representation: `"-"` if empty, otherwise space-delimited tokens. */
  def boardToken(board: Board): String =
    if board.cards.isEmpty then "-"
    else board.cards.map(_.toToken).mkString(" ")

  /** Lowercase TSV action token: `"fold"`, `"check"`, `"call"`, `"raise:<amount>"`. */
  def actionToken(action: PokerAction): String =
    action match
      case PokerAction.Fold => "fold"
      case PokerAction.Check => "check"
      case PokerAction.Call => "call"
      case PokerAction.Raise(amount) => s"raise:${fmt(amount, 3)}"

  /** Capitalized human-readable action token: `"Fold"`, `"Check"`, `"Call"`, `"Raise:<amount>"`.
    * Used in PokerStars-format review hand history.
    */
  def renderAction(action: PokerAction): String =
    action match
      case PokerAction.Fold => "Fold"
      case PokerAction.Check => "Check"
      case PokerAction.Call => "Call"
      case PokerAction.Raise(amount) => s"Raise:${fmt(amount, 2)}"

  /** Position that posts the small blind. Heads-up has the button post SB; multi-way uses SB. */
  def smallBlindPositionFor(playerCount: Int): Position =
    if playerCount <= 2 then Position.Button
    else Position.SmallBlind
