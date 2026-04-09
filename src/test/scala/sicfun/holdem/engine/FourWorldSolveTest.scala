package sicfun.holdem.engine

import sicfun.holdem.types.*
import sicfun.holdem.strategic.*

class FourWorldSolveTest extends munit.FunSuite:

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
  private val actionPriors = StrategicEngine.defaultActionPriors

  test("buildFourWorldModels builds three distinct models"):
    val models = StrategicEngine.buildFourWorldModels(
      gameState, Map.empty, heroActions, heroBucket, actionPriors
    )
    assertEquals(models.size, 3)
    // Open-loop model has uniform obs
    val numObs = StrategicClass.values.length
    val olObs = models.openLoop.obsLikelihood
    val uniformP = 1.0 / numObs
    for i <- 0 until models.openLoop.numStates * models.openLoop.numActions do
      for o <- 0 until numObs do
        assertEqualsDouble(olObs(i * numObs + o), uniformP, 1e-12)
    // Blind model has same rewards as baseline
    models.blind.rewardTable.zip(models.baseline.rewardTable).foreach { (bl, base) =>
      assertEqualsDouble(bl, base, 1e-12)
    }

  test("extractFourWorldValues: solver Q-values map to grid world values"):
    val baselineQ = Array(0.0, 10.0, 5.0) // V^{1,1} = max = 10.0
    val openLoopQ = Array(0.0, 7.0, 4.0)  // V^{1,0} = max = 7.0
    val blindQ = Array(0.0, 6.0, 3.0)     // V^{0,1} = max = 6.0
    val staticEquity = 4.0                 // V^{0,0}

    val fw = StrategicEngine.extractFourWorldValues(
      baselineQ, openLoopQ, blindQ, staticEquity
    )
    assertEqualsDouble(fw.v11.value, 10.0, 1e-12)
    assertEqualsDouble(fw.v10.value, 7.0, 1e-12)
    assertEqualsDouble(fw.v01.value, 6.0, 1e-12)
    assertEqualsDouble(fw.v00.value, 4.0, 1e-12)

  test("extractFourWorldValues: Theorem 4 identity holds"):
    val fw = StrategicEngine.extractFourWorldValues(
      baselineQ = Array(-1.0, 10.0, 8.0),
      openLoopQ = Array(-1.0, 7.0, 5.0),
      blindQ = Array(-1.0, 6.0, 4.0),
      staticEquity = 4.0
    )
    val reconstructed = fw.v00 + fw.deltaControl + fw.deltaSigStar + fw.deltaInteraction
    assertEqualsDouble(reconstructed.value, fw.v11.value, 1e-12)

  test("end-to-end: four-world values satisfy ordering constraints"):
    // V^{1,1} >= V^{1,0} (closed-loop >= open-loop under same kernel)
    // V^{1,1} >= V^{0,1} (attrib >= blind under same policy scope)
    // V^{0,0} is the minimum (blind + open-loop)
    val fw = StrategicEngine.extractFourWorldValues(
      baselineQ = Array(-1.0, 10.0, 8.0),
      openLoopQ = Array(-1.0, 7.0, 5.0),
      blindQ = Array(-1.0, 6.0, 4.0),
      staticEquity = 4.0
    )
    assert(fw.v11 >= fw.v10, s"V11=${fw.v11} should >= V10=${fw.v10}")
    assert(fw.v11 >= fw.v01, s"V11=${fw.v11} should >= V01=${fw.v01}")
    assert(fw.v10 >= fw.v00, s"V10=${fw.v10} should >= V00=${fw.v00}")
    assert(fw.v01 >= fw.v00, s"V01=${fw.v01} should >= V00=${fw.v00}")

  test("end-to-end: decomposition components have expected signs"):
    val fw = StrategicEngine.extractFourWorldValues(
      baselineQ = Array(-1.0, 10.0, 8.0),
      openLoopQ = Array(-1.0, 7.0, 5.0),
      blindQ = Array(-1.0, 6.0, 4.0),
      staticEquity = 4.0
    )
    // Delta_cont >= 0: control (closed-loop over open-loop) adds value
    assert(fw.deltaControl >= Ev.Zero, s"deltaControl=${fw.deltaControl} should be >= 0")
    // Delta_sig* >= 0: signaling (attrib over blind) adds value
    assert(fw.deltaSigStar >= Ev.Zero, s"deltaSigStar=${fw.deltaSigStar} should be >= 0")
