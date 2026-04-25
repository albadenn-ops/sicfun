package sicfun.holdem.runtime

import munit.FunSuite
import sicfun.core.Card
import sicfun.holdem.types.{Board, PokerAction, Position}

class HallFormatTest extends FunSuite:

  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  test("fmt formats with the requested decimal precision and Locale.ROOT") {
    assertEquals(HallFormat.fmt(1.5, 2), "1.50")
    assertEquals(HallFormat.fmt(1.0 / 3.0, 4), "0.3333")
    assertEquals(HallFormat.fmt(-2.5, 1), "-2.5")
  }

  test("roundMoney rounds to two decimal places") {
    assertEquals(HallFormat.roundMoney(1.234), 1.23)
    assertEquals(HallFormat.roundMoney(1.235), 1.24) // round half up at the last digit
    assertEquals(HallFormat.roundMoney(0.0), 0.0)
    assertEquals(HallFormat.roundMoney(-1.005), -1.0) // half-even on the -ve side
  }

  test("money formats with a $ prefix and 2 decimals") {
    assertEquals(HallFormat.money(1.5), "$1.50")
    assertEquals(HallFormat.money(0.5), "$0.50")
    assertEquals(HallFormat.money(0.0), "$0.00")
    assertEquals(HallFormat.money(1234.5678), "$1234.57")
  }

  test("bracketedCards joins cards with a space inside square brackets") {
    assertEquals(
      HallFormat.bracketedCards(Seq(card("Ah"), card("Ks"))),
      "[Ah Ks]"
    )
    assertEquals(HallFormat.bracketedCards(Seq.empty), "[]")
  }

  test("boardToken returns dash when no community cards have been dealt") {
    assertEquals(HallFormat.boardToken(Board.empty), "-")
  }

  test("boardToken joins community cards with a space") {
    val flop = Board.from(Seq(card("2c"), card("7h"), card("Jd")))
    assertEquals(HallFormat.boardToken(flop), "2c 7h Jd")
  }

  test("actionToken emits lowercase tokens with raise:<3-decimal>") {
    assertEquals(HallFormat.actionToken(PokerAction.Fold), "fold")
    assertEquals(HallFormat.actionToken(PokerAction.Check), "check")
    assertEquals(HallFormat.actionToken(PokerAction.Call), "call")
    assertEquals(HallFormat.actionToken(PokerAction.Raise(2.5)), "raise:2.500")
    assertEquals(HallFormat.actionToken(PokerAction.Raise(0.125)), "raise:0.125")
  }

  test("renderAction emits capitalized tokens with Raise:<2-decimal>") {
    assertEquals(HallFormat.renderAction(PokerAction.Fold), "Fold")
    assertEquals(HallFormat.renderAction(PokerAction.Check), "Check")
    assertEquals(HallFormat.renderAction(PokerAction.Call), "Call")
    assertEquals(HallFormat.renderAction(PokerAction.Raise(2.5)), "Raise:2.50")
    assertEquals(HallFormat.renderAction(PokerAction.Raise(0.125)), "Raise:0.13")
  }

  test("smallBlindPositionFor returns Button heads-up, SmallBlind multi-way") {
    assertEquals(HallFormat.smallBlindPositionFor(2), Position.Button)
    assertEquals(HallFormat.smallBlindPositionFor(1), Position.Button)
    assertEquals(HallFormat.smallBlindPositionFor(3), Position.SmallBlind)
    assertEquals(HallFormat.smallBlindPositionFor(6), Position.SmallBlind)
    assertEquals(HallFormat.smallBlindPositionFor(9), Position.SmallBlind)
  }
