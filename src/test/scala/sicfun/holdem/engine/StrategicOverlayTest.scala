package sicfun.holdem.engine

import munit.FunSuite
import sicfun.core.DiscreteDistribution
import sicfun.holdem.types.*
import sicfun.holdem.strategic.types.*
import sicfun.holdem.strategic.state.*
import sicfun.holdem.strategic.exploitation.ExploitationState
import sicfun.holdem.engine.inference.ActionEvaluation

class StrategicOverlayTest extends FunSuite:

  private def minimalState(pot: Double, toCall: Double = 50.0): GameState =
    GameState(
      street = Street.Flop,
      board = Board.empty,
      pot = pot,
      toCall = toCall,
      position = Position.Button,
      stackSize = 1000.0,
      betHistory = Vector.empty
    )

  private def uniformBelief: StrategicRivalBelief =
    StrategicRivalBelief.uniform

  private def bluffHeavyBelief: StrategicRivalBelief =
    StrategicRivalBelief(
      DiscreteDistribution(Map(
        StrategicClass.Value -> 0.05,
        StrategicClass.Bluff -> 0.80,
        StrategicClass.Mixed -> 0.05,
        StrategicClass.StructuralBluff -> 0.10
      ))
    )

  private def defaultConfig: StrategicEngine.Config = StrategicEngine.Config()

  private def makeInput(
      evs: Vector[ActionEvaluation],
      beliefs: Map[PlayerId, StrategicRivalBelief] = Map(PlayerId("v1") -> uniformBelief),
      robustBounds: Option[Array[Double]] = None,
      pot: Double = 100.0
  ): OverlayInput =
    OverlayInput(
      gameState = minimalState(pot = pot),
      upstreamEvs = evs,
      rivalBeliefs = beliefs,
      exploitationStates = beliefs.map((id, _) => id -> ExploitationState.initial(defaultConfig.exploitConfig)),
      robustLowerBounds = robustBounds,
      config = defaultConfig
    )

  test("filter passes through upstream action when no penalties apply"):
    val evs = Vector(
      ActionEvaluation(PokerAction.Fold, -50.0),
      ActionEvaluation(PokerAction.Raise(2.0), 30.0)
    )
    val result = StrategicOverlay.filter(makeInput(evs))
    assertEquals(result.selectedAction, PokerAction.Raise(2.0))
    assertEquals(result.upstreamAction, PokerAction.Raise(2.0))
    assert(result.softVetoed.isEmpty)

  test("filter penalizes Call against bluff-heavy opponent"):
    val evs = Vector(
      ActionEvaluation(PokerAction.Fold, -50.0),
      ActionEvaluation(PokerAction.Call, 5.0),
      ActionEvaluation(PokerAction.Raise(2.0), 4.0)
    )
    val beliefs = Map(PlayerId("v1") -> bluffHeavyBelief)
    val result = StrategicOverlay.filter(makeInput(evs, beliefs = beliefs))
    assertEquals(result.upstreamAction, PokerAction.Call)
    assert(result.adjustments.exists(_.action == PokerAction.Call))

  test("filter with large pot fraction flips Call vs Raise"):
    val evsTight = Vector(
      ActionEvaluation(PokerAction.Call, 0.10),
      ActionEvaluation(PokerAction.Raise(2.0), 0.05)
    )
    val beliefs = Map(PlayerId("v1") -> bluffHeavyBelief)
    val result = StrategicOverlay.filter(makeInput(evsTight, beliefs = beliefs, pot = 800.0))
    assertEquals(result.selectedAction, PokerAction.Raise(2.0))
    assert(result.upstreamAction == PokerAction.Call)

  test("soft veto flags actions below threshold but does not remove them"):
    val evs = Vector(
      ActionEvaluation(PokerAction.Call, 5.0),
      ActionEvaluation(PokerAction.Raise(2.0), 10.0)
    )
    val bounds = Array(-2.0, 3.0)
    val result = StrategicOverlay.filter(makeInput(evs, robustBounds = Some(bounds)))
    assert(result.softVetoed.nonEmpty)
    assertEquals(result.softVetoed.head._1, PokerAction.Call)
    assertEquals(result.selectedAction, PokerAction.Raise(2.0))

  test("when all actions soft-vetoed, selects best robust lower bound"):
    val evs = Vector(
      ActionEvaluation(PokerAction.Call, 5.0),
      ActionEvaluation(PokerAction.Raise(2.0), 10.0)
    )
    val bounds = Array(-2.0, -1.0)
    val result = StrategicOverlay.filter(makeInput(evs, robustBounds = Some(bounds)))
    assertEquals(result.softVetoed.size, 2)
    assertEquals(result.selectedAction, PokerAction.Raise(2.0))

  test("empty upstream EVs returns Fold"):
    val result = StrategicOverlay.filter(makeInput(Vector.empty))
    assertEquals(result.selectedAction, PokerAction.Fold)

  test("aggregateBluffMass sums P(Bluff) across rivals"):
    val beliefs = Map(
      PlayerId("v1") -> bluffHeavyBelief,
      PlayerId("v2") -> uniformBelief
    )
    val mass = StrategicOverlay.aggregateBluffMass(beliefs)
    assertEqualsDouble(mass, 1.05, 1e-10)
