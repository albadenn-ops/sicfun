package sicfun.holdem.engine

import sicfun.holdem.types.*
import sicfun.holdem.strategic.*

class FourWorldFormulationTest extends munit.FunSuite:

  private val gameState = GameState(
    street = Street.Flop,
    board = Board.empty,
    pot = 100.0,
    toCall = 20.0,
    position = Position.Button,
    stackSize = 500.0,
    betHistory = Vector.empty
  )
  private val heroActions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Check)
  private val heroBucket = 5
  private val actionPriors = Map(
    (StrategicClass.Value, PokerAction.Category.Fold)  -> 0.1,
    (StrategicClass.Value, PokerAction.Category.Call)   -> 0.4,
    (StrategicClass.Value, PokerAction.Category.Check)  -> 0.2,
    (StrategicClass.Value, PokerAction.Category.Raise)  -> 0.3,
    (StrategicClass.Bluff, PokerAction.Category.Fold)   -> 0.4,
    (StrategicClass.Bluff, PokerAction.Category.Call)    -> 0.2,
    (StrategicClass.Bluff, PokerAction.Category.Check)   -> 0.1,
    (StrategicClass.Bluff, PokerAction.Category.Raise)   -> 0.3
  )

  private val numObs = StrategicClass.values.length

  test("buildOpenLoopModel: obs likelihoods are uniform"):
    val model = PokerPftFormulation.buildOpenLoopModel(
      gameState, Map.empty, heroActions, heroBucket, actionPriors
    )
    val uniformP = 1.0 / numObs
    for i <- 0 until model.numStates * model.numActions do
      for o <- 0 until numObs do
        assertEqualsDouble(
          model.obsLikelihood(i * numObs + o), uniformP, 1e-12
        )

  test("buildOpenLoopModel: rewards match attrib model (profile-modulated)"):
    val openLoop = PokerPftFormulation.buildOpenLoopModel(
      gameState, Map.empty, heroActions, heroBucket, actionPriors,
      profileClass = Some(StrategicClass.Value)
    )
    val attrib = PokerPftFormulation.buildTabularModel(
      gameState, Map.empty, heroActions, heroBucket, actionPriors,
      profileClass = Some(StrategicClass.Value)
    )
    // Same rewards (attrib kernel drives reward model)
    openLoop.rewardTable.zip(attrib.rewardTable).foreach { (ol, at) =>
      assertEqualsDouble(ol, at, 1e-12)
    }

  test("buildBlindKernelModel: obs likelihoods are from rival beliefs (not uniform)"):
    import sicfun.core.DiscreteDistribution
    val belief = StrategicRivalBelief(
      DiscreteDistribution(Map(
        StrategicClass.Value -> 0.5,
        StrategicClass.Bluff -> 0.2,
        StrategicClass.Mixed -> 0.2,
        StrategicClass.StructuralBluff -> 0.1
      ))
    )
    val model = PokerPftFormulation.buildBlindKernelModel(
      gameState, Map(PlayerId("v1") -> belief), heroActions, heroBucket, actionPriors
    )
    // Obs should reflect the rival belief, not uniform
    val firstObs: IndexedSeq[Double] = (0 until numObs).map(o => model.obsLikelihood(o))
    assert(firstObs.max - firstObs.min > 0.05, "obs should be non-uniform")

  test("buildBlindKernelModel: rewards use baseline (no profile modulation)"):
    val blindModel = PokerPftFormulation.buildBlindKernelModel(
      gameState, Map.empty, heroActions, heroBucket, actionPriors
    )
    val mixedModel = PokerPftFormulation.buildTabularModel(
      gameState, Map.empty, heroActions, heroBucket, actionPriors,
      profileClass = None
    )
    // Both use profileClass=None -> same base rewards
    blindModel.rewardTable.zip(mixedModel.rewardTable).foreach { (bl, mx) =>
      assertEqualsDouble(bl, mx, 1e-12)
    }

  test("buildOpenLoopModel and buildBlindKernelModel have same dimensions as baseline"):
    val baseline = PokerPftFormulation.buildTabularModel(
      gameState, Map.empty, heroActions, heroBucket, actionPriors
    )
    val openLoop = PokerPftFormulation.buildOpenLoopModel(
      gameState, Map.empty, heroActions, heroBucket, actionPriors
    )
    val blind = PokerPftFormulation.buildBlindKernelModel(
      gameState, Map.empty, heroActions, heroBucket, actionPriors
    )
    assertEquals(openLoop.numStates, baseline.numStates)
    assertEquals(openLoop.numActions, baseline.numActions)
    assertEquals(openLoop.numObs, baseline.numObs)
    assertEquals(blind.numStates, baseline.numStates)
    assertEquals(blind.numActions, baseline.numActions)
    assertEquals(blind.numObs, baseline.numObs)

  test("buildBlindOpenLoopModel: obs likelihoods are uniform"):
    val model = PokerPftFormulation.buildBlindOpenLoopModel(
      gameState, Map.empty, heroActions, heroBucket, actionPriors
    )
    val uniformP = 1.0 / numObs
    for i <- 0 until model.numStates * model.numActions do
      for o <- 0 until numObs do
        assertEqualsDouble(
          model.obsLikelihood(i * numObs + o), uniformP, 1e-12
        )

  test("buildBlindOpenLoopModel: rewards match blind model (baseline, no profile modulation)"):
    val blindOpenLoop = PokerPftFormulation.buildBlindOpenLoopModel(
      gameState, Map.empty, heroActions, heroBucket, actionPriors
    )
    val blind = PokerPftFormulation.buildBlindKernelModel(
      gameState, Map.empty, heroActions, heroBucket, actionPriors
    )
    // Same rewards (both use profileClass=None)
    blindOpenLoop.rewardTable.zip(blind.rewardTable).foreach { (bol, bl) =>
      assertEqualsDouble(bol, bl, 1e-12)
    }

  test("buildBlindOpenLoopModel: same dimensions as other models"):
    val baseline = PokerPftFormulation.buildTabularModel(
      gameState, Map.empty, heroActions, heroBucket, actionPriors
    )
    val blindOl = PokerPftFormulation.buildBlindOpenLoopModel(
      gameState, Map.empty, heroActions, heroBucket, actionPriors
    )
    assertEquals(blindOl.numStates, baseline.numStates)
    assertEquals(blindOl.numActions, baseline.numActions)
    assertEquals(blindOl.numObs, baseline.numObs)

  test("four grid-world models are all distinct combinations"):
    // Verify the 2x2 grid: {baseline, blind} x {obs, uniform}
    val attribCl = PokerPftFormulation.buildTabularModel(
      gameState, Map.empty, heroActions, heroBucket, actionPriors,
      profileClass = Some(StrategicClass.Value)
    )
    val attribOl = PokerPftFormulation.buildOpenLoopModel(
      gameState, Map.empty, heroActions, heroBucket, actionPriors,
      profileClass = Some(StrategicClass.Value)
    )
    val blindCl = PokerPftFormulation.buildBlindKernelModel(
      gameState, Map.empty, heroActions, heroBucket, actionPriors
    )
    val blindOl = PokerPftFormulation.buildBlindOpenLoopModel(
      gameState, Map.empty, heroActions, heroBucket, actionPriors
    )
    // attribCl vs attribOl: same rewards, different obs
    attribCl.rewardTable.zip(attribOl.rewardTable).foreach { (a, b) =>
      assertEqualsDouble(a, b, 1e-12)
    }
    // blindCl vs blindOl: same rewards, different obs
    blindCl.rewardTable.zip(blindOl.rewardTable).foreach { (a, b) =>
      assertEqualsDouble(a, b, 1e-12)
    }
    // attribCl vs blindCl: different rewards (profile modulation), but may share obs structure
    val rewardsDiffer = attribCl.rewardTable.zip(blindCl.rewardTable).exists((a, b) => math.abs(a - b) > 1e-12)
    assert(rewardsDiffer, "attrib and blind should have different rewards")
