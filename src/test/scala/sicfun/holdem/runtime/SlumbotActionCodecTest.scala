package sicfun.holdem.runtime

import munit.FunSuite
import sicfun.core.Card
import sicfun.holdem.types.*

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
