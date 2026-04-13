package sicfun.holdem.engine

import munit.FunSuite
import sicfun.core.Card
import sicfun.holdem.types.*
import sicfun.holdem.strategic.types.*
import sicfun.holdem.engine.inference.ActionEvaluation

class StrategicEngineOverlayTest extends FunSuite:

  private def minimalState: GameState =
    GameState(
      street = Street.Flop,
      board = Board.empty,
      pot = 100.0,
      toCall = 50.0,
      position = Position.Button,
      stackSize = 1000.0,
      betHistory = Vector.empty
    )

  private def testHeroCards: HoleCards =
    val as = Card.parse("As").get
    val kh = Card.parse("Kh").get
    HoleCards.from(Vector(as, kh))

  test("overlay decide returns OverlayResult with correct upstream action"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand(testHeroCards)

    val evs = Vector(
      ActionEvaluation(PokerAction.Fold, -50.0),
      ActionEvaluation(PokerAction.Call, 5.0),
      ActionEvaluation(PokerAction.Raise(2.0), 10.0)
    )
    val result = engine.decide(minimalState, evs.map(_.action), evs)
    assertEquals(result.upstreamAction, PokerAction.Raise(2.0))
    assertEquals(result.selectedAction, PokerAction.Raise(2.0))
    assert(engine.lastOverlayResult.isDefined)

  test("overlay decide requires initialized session"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    val evs = Vector(ActionEvaluation(PokerAction.Call, 5.0))
    interceptMessage[IllegalArgumentException]("requirement failed: Session not initialized") {
      engine.decide(minimalState, Vector(PokerAction.Call), evs)
    }

  test("overlay decide requires active hand"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    val evs = Vector(ActionEvaluation(PokerAction.Call, 5.0))
    interceptMessage[IllegalArgumentException]("requirement failed: No hand in progress") {
      engine.decide(minimalState, Vector(PokerAction.Call), evs)
    }
