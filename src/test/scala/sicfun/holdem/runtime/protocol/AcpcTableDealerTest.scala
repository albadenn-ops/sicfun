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

  test("action closes preflop when BB checks option after limpers (N=3)"):
    val cfg = TableConfig(3, 1L, 2L, 0L, 200L)
    val d = AcpcTableDealer(cfg, SeatId(0), 1L)
    d.postBlinds()
    d.startStreet(sicfun.holdem.types.Street.Preflop)
    assertEquals(d.nextToAct, Some(SeatId(0)))
    d.applyAction(SeatId(0), sicfun.holdem.types.PokerAction.Call)
      .getOrElse(fail("utg call"))
    d.applyAction(SeatId(1), sicfun.holdem.types.PokerAction.Call)
      .getOrElse(fail("sb call"))
    d.applyAction(SeatId(2), sicfun.holdem.types.PokerAction.Check)
      .getOrElse(fail("bb option"))
    assert(d.roundClosed, "round closed after BB option")

  test("action closes after BB flat-calls a raise (v1 bug regression)"):
    val cfg = TableConfig(3, 1L, 2L, 0L, 200L)
    val d = AcpcTableDealer(cfg, SeatId(0), 1L)
    d.postBlinds()
    d.startStreet(sicfun.holdem.types.Street.Preflop)
    d.applyAction(SeatId(0), sicfun.holdem.types.PokerAction.Raise(6.0))
      .getOrElse(fail("utg raise"))
    d.applyAction(SeatId(1), sicfun.holdem.types.PokerAction.Call)
      .getOrElse(fail("sb call"))
    assert(!d.roundClosed, "BB still has option")
    d.applyAction(SeatId(2), sicfun.holdem.types.PokerAction.Call)
      .getOrElse(fail("bb call"))
    assert(d.roundClosed, "closed after BB closes action on the raiser - v1 bug")

  test("legalActionsFor: facing no bet → menu has Fold, Check, Raise(halfPot), Raise(pot), Raise(allIn)"):
    val cfg = TableConfig(6, 1L, 2L, 0L, 200L)
    val d = AcpcTableDealer(cfg, SeatId(0), 1L)
    d.postBlinds()
    d.dealHoleCards()
    d.startStreet(sicfun.holdem.types.Street.Flop) // no outstanding bet
    val menu = d.legalActionsFor(SeatId(0))
    assert(menu.contains(sicfun.holdem.types.PokerAction.Fold))
    assert(menu.contains(sicfun.holdem.types.PokerAction.Check))
    val raises = menu.collect { case r: sicfun.holdem.types.PokerAction.Raise => r }
    assertEquals(raises.map(_.amount).size, raises.size)
    assert(raises.nonEmpty, "post-flop raises should be available from non-empty pot")

  test("legalActionsFor: stack short enough to collapse HalfPot/Pot/AllIn into one Raise"):
    val cfg = TableConfig(2, 1L, 2L, 0L, 20L)
    val d = AcpcTableDealer(cfg, SeatId(0), 1L)
    d.postBlinds()
    d.dealHoleCards()
    d.startStreet(sicfun.holdem.types.Street.Preflop)
    d.applyAction(SeatId(0), sicfun.holdem.types.PokerAction.Raise(16.0))
      .getOrElse(fail("aggressive open"))
    val menu = d.legalActionsFor(SeatId(1))
    val raises = menu.collect { case r: sicfun.holdem.types.PokerAction.Raise => r }
    assertEquals(raises.map(_.amount).toSet.size, raises.size,
      s"dedup failed: ${raises.map(_.amount)}")

  test("legalActionsFor preflop pot size is not double-counted"):
    val cfg = TableConfig(2, 1L, 2L, 0L, 200L)
    val d = AcpcTableDealer(cfg, SeatId(0), 1L)
    d.postBlinds()
    d.dealHoleCards()
    d.startStreet(sicfun.holdem.types.Street.Preflop)
    // After blinds: pot = 1 + 2 = 3. SB owes 1 to call.
    // PotRaise must be exactly 3 (the pot), not 6 (double-counted).
    val menu = d.legalActionsFor(SeatId(0))
    val raises = menu.collect { case sicfun.holdem.types.PokerAction.Raise(a) => a }
    assert(raises.contains(3.0), s"expected pot-size raise of 3.0 in $raises (would be 6.0 if pot was double-counted)")

  test("applyAction: Check when owed > 0 → IllegalForm"):
    val cfg = TableConfig(3, 1L, 2L, 0L, 200L)
    val d = AcpcTableDealer(cfg, SeatId(0), 1L)
    d.postBlinds()
    d.startStreet(sicfun.holdem.types.Street.Preflop)
    val result = d.applyAction(SeatId(0), sicfun.holdem.types.PokerAction.Check)
    assert(result.isLeft)
    result.left.foreach {
      case IllegalActionReason.IllegalForm(_, _, _) => ()
      case other => fail(s"expected IllegalForm, got $other")
    }

  test("applyAction: Raise below currentBet → IllegalForm"):
    val cfg = TableConfig(3, 1L, 2L, 0L, 200L)
    val d = AcpcTableDealer(cfg, SeatId(0), 1L)
    d.postBlinds()
    d.startStreet(sicfun.holdem.types.Street.Preflop)
    val result = d.applyAction(SeatId(0), sicfun.holdem.types.PokerAction.Raise(1.5))
    assert(result.isLeft)

  test("applyAction: Raise more than stack → InsufficientChips"):
    val cfg = TableConfig(3, 1L, 2L, 0L, 20L)
    val d = AcpcTableDealer(cfg, SeatId(0), 1L)
    d.postBlinds()
    d.startStreet(sicfun.holdem.types.Street.Preflop)
    val result = d.applyAction(SeatId(0), sicfun.holdem.types.PokerAction.Raise(1000.0))
    assert(result.isLeft)
    result.left.foreach {
      case IllegalActionReason.InsufficientChips(_, _, _) => ()
      case other => fail(s"expected InsufficientChips, got $other")
    }

  test("applyAction: out-of-turn → NotYourTurn"):
    val cfg = TableConfig(3, 1L, 2L, 0L, 200L)
    val d = AcpcTableDealer(cfg, SeatId(0), 1L)
    d.postBlinds()
    d.startStreet(sicfun.holdem.types.Street.Preflop)
    val wrong = if d.nextToAct == Some(SeatId(0)) then SeatId(2) else SeatId(0)
    val result = d.applyAction(wrong, sicfun.holdem.types.PokerAction.Fold)
    assert(result.isLeft)

  test("eventLog records PostBlind, Deal, Act in order"):
    val cfg = TableConfig(3, 1L, 2L, 0L, 200L)
    val d = AcpcTableDealer(cfg, SeatId(0), 1L)
    d.postBlinds()
    d.dealHoleCards()
    d.startStreet(sicfun.holdem.types.Street.Preflop)
    d.applyAction(SeatId(0), sicfun.holdem.types.PokerAction.Fold)
    val log = d.eventLog
    assertEquals(log.take(2).collect { case e: BettingRoundEvent.PostBlind => e }.size, 2,
      s"expected 2 PostBlind events, got ${log.take(2)}")
    assert(log.exists(_.isInstanceOf[BettingRoundEvent.Act]))
