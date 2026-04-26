package sicfun.holdem.runtime.protocol

import munit.FunSuite
import sicfun.core.Card
import sicfun.holdem.types.*

/** Tests for [[SlumbotActionCodec]] — the Slumbot wire protocol parser and encoder.
  *
  * Verifies:
  *  - '''Empty action:''' An empty action string yields button-first preflop state with
  *    correct pot (1.5 BB), to-call (0.5 BB), and stack (199.5 BB).
  *  - '''Button open:''' A `b200` raise from the button is parsed as a villain raise (1.0 BB
  *    increment) when hero is big blind, with correct state updates.
  *  - '''Button limp:''' A `c` call from the button leaves big blind with checking option
  *    (to-call = 0).
  *  - '''Street advancement:''' A preflop open-call followed by a flop check (`b200c/k`)
  *    correctly advances to the Flop street with the dealt board visible.
  *  - '''Raise encoding:''' `incrementForAction` maps a Raise(2.0 BB) to `b300` (street
  *    bet-to of 300 chips = current 100 + increment of 200).
  */
class SlumbotActionCodecTest extends FunSuite:

  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  test("empty action gives button first preflop decision") {
    val parsed = SlumbotActionCodec.parse("", heroActual = 1, fullBoard = Board.empty).fold(fail(_), identity)
    val state = parsed.nextDecisionState.getOrElse(fail("expected decision state"))

    assertEquals(parsed.nextActorActual, 1)
    assertEquals(parsed.nextActorRelative, 0)
    assertEquals(parsed.steps.length, 0)
    assertEquals(state.street, Street.Preflop)
    assertEquals(state.position, Position.Button)
    assertEqualsDouble(state.pot, 1.5, 1e-9)
    assertEqualsDouble(state.toCall, 0.5, 1e-9)
    assertEqualsDouble(state.stackSize, 199.5, 1e-9)
  }

  test("button open is parsed as villain raise when hero is big blind") {
    val parsed = SlumbotActionCodec.parse("b200", heroActual = 0, fullBoard = Board.empty).fold(fail(_), identity)
    val state = parsed.nextDecisionState.getOrElse(fail("expected decision state"))

    assertEquals(parsed.steps.length, 1)
    val step = parsed.steps.head
    assertEquals(step.actualActor, 1)
    assertEquals(step.relativeActor, 1)
    assertEquals(step.action, PokerAction.Raise(1.0))
    assertEquals(step.stateBefore.position, Position.Button)
    assertEqualsDouble(step.stateBefore.toCall, 0.5, 1e-9)

    assertEquals(state.position, Position.BigBlind)
    assertEquals(state.street, Street.Preflop)
    assertEqualsDouble(state.pot, 3.0, 1e-9)
    assertEqualsDouble(state.toCall, 1.0, 1e-9)
    assertEqualsDouble(state.stackSize, 199.0, 1e-9)
  }

  test("button limp leaves big blind checking option") {
    val parsed = SlumbotActionCodec.parse("c", heroActual = 0, fullBoard = Board.empty).fold(fail(_), identity)
    val state = parsed.nextDecisionState.getOrElse(fail("expected decision state"))

    assertEquals(parsed.steps.length, 1)
    assertEquals(parsed.steps.head.action, PokerAction.Call)
    assertEquals(state.position, Position.BigBlind)
    assertEqualsDouble(state.pot, 2.0, 1e-9)
    assertEqualsDouble(state.toCall, 0.0, 1e-9)
  }

  test("preflop open call plus flop check advances street correctly") {
    val flop = Board.from(Vector(card("Ah"), card("7d"), card("2c")))
    val parsed = SlumbotActionCodec.parse("b200c/k", heroActual = 1, fullBoard = flop).fold(fail(_), identity)
    val state = parsed.nextDecisionState.getOrElse(fail("expected decision state"))

    assertEquals(parsed.steps.length, 3)
    assertEquals(parsed.steps.map(_.action), Vector(PokerAction.Raise(1.0), PokerAction.Call, PokerAction.Check))
    assertEquals(state.street, Street.Flop)
    assertEquals(state.position, Position.Button)
    assertEquals(state.board, flop)
    assertEqualsDouble(state.pot, 4.0, 1e-9)
    assertEqualsDouble(state.toCall, 0.0, 1e-9)
  }

  test("raise action is mapped back to Slumbot increment string") {
    val parsed = SlumbotActionCodec.parse("", heroActual = 1, fullBoard = Board.empty).fold(fail(_), identity)
    assertEquals(SlumbotActionCodec.incrementForAction(parsed, PokerAction.Raise(2.0)), "b300")
  }

  test("incrementForAction maps Fold to single-char 'f'") {
    val parsed = SlumbotActionCodec.parse("", heroActual = 1, fullBoard = Board.empty).fold(fail(_), identity)
    assertEquals(SlumbotActionCodec.incrementForAction(parsed, PokerAction.Fold), "f")
  }

  test("incrementForAction encodes Check as 'k' when nothing is owed, 'c' when toCall>0") {
    val afterLimp = SlumbotActionCodec.parse("c", heroActual = 0, fullBoard = Board.empty).fold(fail(_), identity)
    assertEquals(SlumbotActionCodec.incrementForAction(afterLimp, PokerAction.Check), "k",
      "with toCall=0, Check encodes as k")
    val initial = SlumbotActionCodec.parse("", heroActual = 1, fullBoard = Board.empty).fold(fail(_), identity)
    assertEquals(SlumbotActionCodec.incrementForAction(initial, PokerAction.Check), "c",
      "with toCall>0, Check round-trips as c")
  }

  test("incrementForAction encodes Call symmetrically (c when owed, k when free)") {
    val initial = SlumbotActionCodec.parse("", heroActual = 1, fullBoard = Board.empty).fold(fail(_), identity)
    assertEquals(SlumbotActionCodec.incrementForAction(initial, PokerAction.Call), "c")
    val afterLimp = SlumbotActionCodec.parse("c", heroActual = 0, fullBoard = Board.empty).fold(fail(_), identity)
    assertEquals(SlumbotActionCodec.incrementForAction(afterLimp, PokerAction.Call), "k")
  }

  test("positionForActual: actual 0 = BigBlind, actual 1 = Button (Slumbot reverse-blind seating)") {
    assertEquals(SlumbotActionCodec.positionForActual(0), Position.BigBlind)
    assertEquals(SlumbotActionCodec.positionForActual(1), Position.Button)
  }

  test("positionForActual rejects invalid actual indices") {
    intercept[IllegalArgumentException] { SlumbotActionCodec.positionForActual(-1) }
    intercept[IllegalArgumentException] { SlumbotActionCodec.positionForActual(2) }
  }

  test("chipsToBb / bbToChips round-trip at the BigBlind=100 scale") {
    assertEquals(SlumbotActionCodec.chipsToBb(0), 0.0)
    assertEquals(SlumbotActionCodec.chipsToBb(100), 1.0)
    assertEquals(SlumbotActionCodec.chipsToBb(250), 2.5)
    assertEquals(SlumbotActionCodec.bbToChips(0.0), 0)
    assertEquals(SlumbotActionCodec.bbToChips(1.0), 100)
    assertEquals(SlumbotActionCodec.bbToChips(2.5), 250)
  }

  test("bbToChips floors at zero (no negative chip counts)") {
    assertEquals(SlumbotActionCodec.bbToChips(-1.0), 0,
      "negative BB amounts must clamp to 0 to keep chip counts non-negative")
    assertEquals(SlumbotActionCodec.bbToChips(-0.001), 0)
  }

  test("bbToChips rounds via math.round (half-up at the .5 boundary)") {
    // math.round on Java/Scala uses HALF_UP for positive values: floor(x + 0.5)
    // 0.005 BB = 0.5 chips -> rounds to 1
    assertEquals(SlumbotActionCodec.bbToChips(0.005), 1)
    // 0.999 BB = 99.9 chips -> rounds to 100
    assertEquals(SlumbotActionCodec.bbToChips(0.999), 100)
    // 0.004 BB = 0.4 chips -> rounds to 0
    assertEquals(SlumbotActionCodec.bbToChips(0.004), 0)
    // 0.014 BB = 1.4 chips -> rounds to 1
    assertEquals(SlumbotActionCodec.bbToChips(0.014), 1)
  }
