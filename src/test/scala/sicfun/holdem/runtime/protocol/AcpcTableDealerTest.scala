package sicfun.holdem.runtime.protocol

import sicfun.holdem.runtime.protocol.BettingRoundEvent.*
import sicfun.holdem.runtime.protocol.BlindKind.*

class AcpcTableDealerTest extends munit.FunSuite:
  val cfg2 = TableConfig(2, 1L, 2L, 0L, 200L)
  val cfg6 = TableConfig(6, 1L, 2L, 0L, 200L)
  val cfg9 = TableConfig(9, 1L, 2L, 0L, 200L)

  test("blinds at N=2: button is SB"):
    val d = AcpcTableDealer(cfg2, SeatId(0), 42L)
    val events = d.postBlinds()
    assertEquals(events, Vector(
      PostBlind(SeatId(0), 1L, SmallBlind),
      PostBlind(SeatId(1), 2L, BigBlind)
    ))

  test("blinds at N=6: SB = BTN+1, BB = BTN+2"):
    val d = AcpcTableDealer(cfg6, SeatId(2), 42L)
    val events = d.postBlinds()
    assertEquals(events, Vector(
      PostBlind(SeatId(3), 1L, SmallBlind),
      PostBlind(SeatId(4), 2L, BigBlind)
    ))

  test("blinds wrap around seat indices"):
    val d = AcpcTableDealer(cfg9, SeatId(8), 42L)
    val events = d.postBlinds()
    assertEquals(events, Vector(
      PostBlind(SeatId(0), 1L, SmallBlind),
      PostBlind(SeatId(1), 2L, BigBlind)
    ))

  test("button rotates one seat per hand and cycles through all N=9 seats"):
    val d = AcpcTableDealer(cfg9, SeatId(0), 1L)
    val visited = (0 until 9).map { _ =>
      val b = d.buttonSeat
      d.advanceButton()
      b.index
    }.toSet
    assertEquals(visited, (0 until 9).toSet)

  test("button wraps modulo numSeats at N=2"):
    val d = AcpcTableDealer(cfg2, SeatId(1), 1L)
    d.advanceButton()
    assertEquals(d.buttonSeat, SeatId(0))

  test("dealHoleCards: 2 distinct cards per seat, 2N cards total, no duplicates"):
    val d = AcpcTableDealer(cfg6, SeatId(0), 42L)
    val hole = d.dealHoleCards()
    assertEquals(hole.size, 6)
    hole.values.foreach(cs => assertEquals(cs.size, 2))
    val allCards = hole.values.flatten.toVector
    assertEquals(allCards.distinct.size, 12)

  test("dealCommunity: flop=3, turn=1, river=1, all distinct from hole"):
    val d = AcpcTableDealer(cfg6, SeatId(0), 42L)
    val hole = d.dealHoleCards()
    val flop = d.dealCommunity(sicfun.holdem.types.Street.Flop)
    val turn = d.dealCommunity(sicfun.holdem.types.Street.Turn)
    val river = d.dealCommunity(sicfun.holdem.types.Street.River)
    assertEquals(flop.size, 3)
    assertEquals(turn.size, 4)        // board accumulates
    assertEquals(river.size, 5)
    val all = hole.values.flatten.toSet ++ river.toSet
    assertEquals(all.size, 12 + 5)    // all distinct
