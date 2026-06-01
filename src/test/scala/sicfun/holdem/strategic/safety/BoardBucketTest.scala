package sicfun.holdem.strategic.safety

import munit.FunSuite
import sicfun.core.Card
import sicfun.holdem.types.{Board, Street}
import sicfun.holdem.strategic.safety.BoardBucket.*

class BoardBucketTest extends FunSuite:
  private def board(tokens: String*): Board = Board(tokens.toVector.map(t => Card.parse(t).get))

  test("preflop board → PRE token regardless of street"):
    assertEquals(BoardBucket.ofBoard(Street.Preflop, Board.empty), BoardBucket.Preflop)
    assertEquals(BoardBucket.token(BoardBucket.Preflop), "PRE")

  test("unpaired rainbow ace-high flop"):
    val b = BoardBucket.ofBoard(Street.Flop, board("As", "Kd", "7c"))
    assertEquals(b, BoardBucket.Postflop(Pairing.Unpaired, Suitedness.Rainbow, HighCard.AceHigh))
    assertEquals(BoardBucket.token(b), "Unpaired-Rainbow-AceHigh")

  test("paired rainbow broadway flop"):
    val b = BoardBucket.ofBoard(Street.Flop, board("Ks", "Kh", "Qd"))
    assertEquals(b, BoardBucket.Postflop(Pairing.Paired, Suitedness.Rainbow, HighCard.Broadway))

  test("monotone middle flop"):
    val b = BoardBucket.ofBoard(Street.Flop, board("9h", "7h", "5h"))
    assertEquals(b, BoardBucket.Postflop(Pairing.Unpaired, Suitedness.Monotone, HighCard.Middle))

  test("trips low flop"):
    val b = BoardBucket.ofBoard(Street.Flop, board("4h", "4d", "4c"))
    assertEquals(b, BoardBucket.Postflop(Pairing.Trips, Suitedness.Rainbow, HighCard.Low))

  test("turn two-tone (max suit count 2 → TwoTone)"):
    val b = BoardBucket.ofBoard(Street.Turn, board("As", "Kd", "7c", "2s"))
    assertEquals(b.asInstanceOf[BoardBucket.Postflop].suitedness, Suitedness.TwoTone)
