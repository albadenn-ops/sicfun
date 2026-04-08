package sicfun.holdem.engine

import sicfun.holdem.types.*

class PokerPftFormulationTest extends munit.FunSuite:

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

  test("buildTabularModel produces valid TabularGenerativeModel"):
    val heroActions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(2.0))
    val model = PokerPftFormulation.buildTabularModel(
      minimalState, Map.empty, heroActions, 5, Map.empty
    )

    assertEquals(model.numActions, heroActions.size)
    assert(model.numStates > 0)
    assert(model.numObs > 0)
    assertEquals(model.transitionTable.length, model.numStates * model.numActions)
    assertEquals(model.obsLikelihood.length, model.numStates * model.numActions * model.numObs)

  test("buildParticleBelief produces valid ParticleBelief"):
    val belief = PokerPftFormulation.buildParticleBelief(Map.empty, 50)
    assertEquals(belief.stateIndices.length, belief.weights.length)
    assert(belief.weights.sum > 0.99 && belief.weights.sum < 1.01)
