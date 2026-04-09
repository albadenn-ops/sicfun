package sicfun.holdem.strategic.bridge

import sicfun.holdem.strategic.*

class ValueBridgeTest extends munit.FunSuite:

  test("toFourWorldFromSolver: all four values are Exact when solver provides them"):
    val result = ValueBridge.toFourWorldFromSolver(
      v11 = 10.0, v10 = 7.0, v01 = 6.0, v00 = 4.0
    )
    result match
      case BridgeResult.Exact(fw) =>
        assertEqualsDouble(fw.v11.value, 10.0, 1e-12)
        assertEqualsDouble(fw.v10.value, 7.0, 1e-12)
        assertEqualsDouble(fw.v01.value, 6.0, 1e-12)
        assertEqualsDouble(fw.v00.value, 4.0, 1e-12)
      case other => fail(s"expected Exact, got $other")

  test("toFourWorldFromSolver: Theorem 4 decomposition is exact"):
    val result = ValueBridge.toFourWorldFromSolver(
      v11 = 10.0, v10 = 7.0, v01 = 6.0, v00 = 4.0
    )
    result match
      case BridgeResult.Exact(fw) =>
        // Theorem 4: V^{1,1} = V^{0,0} + Delta_cont + Delta_sig* + Delta_int
        val reconstructed = fw.v00 + fw.deltaControl + fw.deltaSigStar + fw.deltaInteraction
        assertEqualsDouble(reconstructed.value, fw.v11.value, 1e-12)
      case other => fail(s"expected Exact, got $other")

  test("toFourWorldFromSolver: decomposition components are meaningful"):
    val result = ValueBridge.toFourWorldFromSolver(
      v11 = 10.0, v10 = 7.0, v01 = 6.0, v00 = 4.0
    )
    result match
      case BridgeResult.Exact(fw) =>
        assertEqualsDouble(fw.deltaControl.value, 2.0, 1e-12)   // V01 - V00 = 6 - 4
        assertEqualsDouble(fw.deltaSigStar.value, 3.0, 1e-12)   // V10 - V00 = 7 - 4
        assertEqualsDouble(fw.deltaInteraction.value, 1.0, 1e-12) // V11 - V10 - V01 + V00
      case other => fail(s"expected Exact, got $other")

  test("toGridWorldValuesFromSolver: all four worlds are Exact"):
    val grid = ValueBridge.toGridWorldValuesFromSolver(
      v11 = 10.0, v10 = 7.0, v01 = 6.0, v00 = 4.0
    )
    assertEquals(grid.size, 4)
    grid.values.foreach {
      case BridgeResult.Exact(_) => ()
      case other => fail(s"expected Exact, got $other")
    }

  test("legacy toFourWorld still works (backward compat)"):
    val result = ValueBridge.toFourWorld(10.0, 4.0, controlFrac = 0.5)
    result match
      case BridgeResult.Approximate(fw, _) =>
        assertEqualsDouble(fw.v11.value, 10.0, 1e-12)
        assertEqualsDouble(fw.v00.value, 4.0, 1e-12)
      case other => fail(s"expected Approximate, got $other")
