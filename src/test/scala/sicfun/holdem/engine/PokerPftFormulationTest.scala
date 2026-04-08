package sicfun.holdem.engine

import sicfun.holdem.types.*
import sicfun.holdem.strategic.StrategicClass

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

  private val testPriors: Map[(StrategicClass, PokerAction.Category), Double] =
    StrategicEngine.defaultActionPriors

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

  test("profile-conditioned models produce different rewards"):
    val heroActions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(50.0))
    val valueModel = PokerPftFormulation.buildTabularModel(
      minimalState, Map.empty, heroActions, 5, testPriors,
      profileClass = Some(StrategicClass.Value)
    )
    val bluffModel = PokerPftFormulation.buildTabularModel(
      minimalState, Map.empty, heroActions, 5, testPriors,
      profileClass = Some(StrategicClass.Bluff)
    )
    val mixedModel = PokerPftFormulation.buildTabularModel(
      minimalState, Map.empty, heroActions, 5, testPriors,
      profileClass = None
    )

    // Profile models must differ from each other
    assert(
      !valueModel.rewardTable.sameElements(bluffModel.rewardTable),
      "Value and Bluff profile models should have different reward tables"
    )
    // Profile models must differ from mixed
    assert(
      !valueModel.rewardTable.sameElements(mixedModel.rewardTable),
      "Value profile model should differ from mixed model"
    )

  test("profile-conditioned models produce different obs likelihoods"):
    val heroActions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(50.0))
    val valueModel = PokerPftFormulation.buildTabularModel(
      minimalState, Map.empty, heroActions, 5, testPriors,
      profileClass = Some(StrategicClass.Value)
    )
    val mixedModel = PokerPftFormulation.buildTabularModel(
      minimalState, Map.empty, heroActions, 5, testPriors,
      profileClass = None
    )

    // Profile obs likelihoods should concentrate on the profile type
    assert(
      !valueModel.obsLikelihood.sameElements(mixedModel.obsLikelihood),
      "Value profile obs likelihood should differ from uniform"
    )
    // Value profile should have higher obs likelihood for obs index 0 (Value ordinal)
    val valueObs0 = valueModel.obsLikelihood(0) // obs=Value for (s=0, a=0)
    val mixedObs0 = mixedModel.obsLikelihood(0)
    assert(valueObs0 > mixedObs0,
      s"Value profile obs[0] ($valueObs0) should exceed mixed ($mixedObs0)")

  test("profile-conditioned obs likelihoods sum to 1"):
    val heroActions = Vector(PokerAction.Fold, PokerAction.Call)
    for cls <- StrategicClass.values do
      val model = PokerPftFormulation.buildTabularModel(
        minimalState, Map.empty, heroActions, 5, testPriors,
        profileClass = Some(cls)
      )
      var i = 0
      while i < model.numStates * model.numActions do
        var sum = 0.0
        var o = 0
        while o < model.numObs do
          sum += model.obsLikelihood(i * model.numObs + o)
          o += 1
        assertEqualsDouble(sum, 1.0, 1e-12)
        i += 1
