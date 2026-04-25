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
