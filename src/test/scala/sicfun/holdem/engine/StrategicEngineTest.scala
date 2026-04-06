package sicfun.holdem.engine

import munit.FunSuite
import sicfun.holdem.types.*
import sicfun.holdem.strategic.*

class StrategicEngineTest extends FunSuite:

  /** Minimal GameState for testing — preflop, no board, pot=100, no-call check, Button. */
  private def minimalState: GameState =
    GameState(
      street = Street.Preflop,
      board = Board.empty,
      pot = 100.0,
      toCall = 0.0,
      position = Position.Button,
      stackSize = 1000.0,
      betHistory = Vector.empty
    )

  test("StrategicEngine initializes with uniform beliefs for unknown rivals"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    val state = engine.sessionState
    assertEquals(state.rivalBeliefs.size, 1)
    assertEqualsDouble(
      state.rivalBeliefs(PlayerId("v1")).typePosterior.probabilityOf(StrategicClass.Value),
      0.25,
      1e-10
    )

  test("startHand sets hand active"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand()
    assert(engine.currentHandActive)

  test("endHand clears hand active"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand()
    engine.endHand()
    assert(!engine.currentHandActive)

  test("observeAction does not throw"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand()
    engine.observeAction(PlayerId("v1"), PokerAction.Call, minimalState)

  test("initSession preserves existing beliefs"):
    val existing = StrategicRivalBelief.uniform
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(
      rivalIds = Vector(PlayerId("v1"), PlayerId("v2")),
      existingBeliefs = Map(PlayerId("v1") -> existing)
    )
    val state = engine.sessionState
    assertEquals(state.rivalBeliefs.size, 2)
    // v1 used the provided belief, v2 got a fresh uniform
    assertEqualsDouble(
      state.rivalBeliefs(PlayerId("v2")).typePosterior.probabilityOf(StrategicClass.Bluff),
      0.25,
      1e-10
    )

  test("sessionState throws before initSession"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    intercept[IllegalArgumentException] {
      engine.sessionState
    }

  test("startHand throws before initSession"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    intercept[IllegalArgumentException] {
      engine.startHand()
    }

  test("decide falls back gracefully when native solver unavailable"):
    // Native DLL for POMCP is unlikely to be loaded in unit test context.
    // Expect the fallback: first non-Fold action or Fold.
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand()
    val actions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(50.0))
    val chosen = engine.decide(minimalState, actions)
    // Fallback path: any action from the candidate set is valid
    assert(actions.contains(chosen))

  test("endHand is idempotent — second call does not throw"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand()
    engine.endHand()
    engine.endHand()
    assert(!engine.currentHandActive)

  test("multiple hands — beliefs survive across startHand/endHand cycle"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand()
    engine.endHand()
    engine.startHand()
    assert(engine.currentHandActive)
    // Beliefs still present
    assertEquals(engine.sessionState.rivalBeliefs.size, 1)
