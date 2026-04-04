package sicfun.holdem.strategic

class SignalDecompositionTest extends munit.FunSuite:

  private inline val Tol = 1e-12

  // -- Def 40: Total signal effect --

  test("deltaSig is qAttrib - qBlind"):
    val result = SignalDecomposition.deltaSig(qAttrib = Ev(10.0), qBlind = Ev(7.0))
    assertEqualsDouble(result.value, 3.0, Tol)

  // -- Def 41: Passive leakage --

  test("deltaPass is qRef - qBlind"):
    val result = SignalDecomposition.deltaPass(qRef = Ev(8.0), qBlind = Ev(7.0))
    assertEqualsDouble(result.value, 1.0, Tol)

  test("negative deltaPass indicates damaging leak"):
    val result = SignalDecomposition.deltaPass(qRef = Ev(5.0), qBlind = Ev(7.0))
    assert(result < Ev.Zero)

  // -- Def 42: Manipulation rent --

  test("deltaManip is qAttrib - qRef"):
    val result = SignalDecomposition.deltaManip(qAttrib = Ev(10.0), qRef = Ev(8.0))
    assertEqualsDouble(result.value, 2.0, Tol)

  // -- Theorem 3: delta_sig = delta_pass + delta_manip (exact telescoping) --

  test("Theorem 3: deltaSig == deltaPass + deltaManip for any Q values"):
    val qAttrib = Ev(15.3)
    val qRef = Ev(9.7)
    val qBlind = Ev(4.2)
    val sig = SignalDecomposition.deltaSig(qAttrib, qBlind)
    val pass = SignalDecomposition.deltaPass(qRef, qBlind)
    val manip = SignalDecomposition.deltaManip(qAttrib, qRef)
    assertEqualsDouble(sig.value, (pass + manip).value, Tol)

  // -- Theorem 5: attrib == ref => manip == 0 --

  test("Theorem 5: deltaManip is zero when attrib equals ref"):
    val q = Ev(12.0)
    val result = SignalDecomposition.deltaManip(qAttrib = q, qRef = q)
    assertEqualsDouble(result.value, 0.0, Tol)

  // -- PerRivalDelta construction --

  test("computePerRivalDelta builds correct PerRivalDelta"):
    val prd = SignalDecomposition.computePerRivalDelta(
      qAttrib = Ev(10.0), qRef = Ev(8.0), qBlind = Ev(6.0)
    )
    assertEqualsDouble(prd.deltaSig.value, 4.0, Tol)
    assertEqualsDouble(prd.deltaPass.value, 2.0, Tol)
    assertEqualsDouble(prd.deltaManip.value, 2.0, Tol)

  // -- Corollary 1: damaging passive leakage --

  test("Corollary 1: isDamagingLeak true when deltaPass < 0"):
    val prd = SignalDecomposition.computePerRivalDelta(
      qAttrib = Ev(5.0), qRef = Ev(3.0), qBlind = Ev(7.0)
    )
    assert(prd.isDamagingLeak)

  test("Corollary 1: isDamagingLeak false when deltaPass >= 0"):
    val prd = SignalDecomposition.computePerRivalDelta(
      qAttrib = Ev(10.0), qRef = Ev(8.0), qBlind = Ev(6.0)
    )
    assert(!prd.isDamagingLeak)

  // -- Def 43: Aggregate signal effect --

  test("deltaSigAggregate is qAttrib_all - qBlind_all"):
    val result = SignalDecomposition.deltaSigAggregate(
      qAttribAll = Ev(20.0), qBlindAll = Ev(14.0)
    )
    assertEqualsDouble(result.value, 6.0, Tol)

  // -- Non-additivity warning (Def 43) --

  test("deltaSigAggregate != sum of per-rival deltaSig in general"):
    // Per-rival: deltaSig_1 = 10-6=4, deltaSig_2 = 12-7=5, sum = 9
    val perRival1 = SignalDecomposition.computePerRivalDelta(Ev(10.0), Ev(8.0), Ev(6.0))
    val perRival2 = SignalDecomposition.computePerRivalDelta(Ev(12.0), Ev(9.0), Ev(7.0))
    val sumIndividual = perRival1.deltaSig + perRival2.deltaSig
    // Aggregate uses joint Q-functions which differ from per-rival sum
    val agg = SignalDecomposition.deltaSigAggregate(Ev(22.0), Ev(13.5))
    // agg = 22 - 13.5 = 8.5, but sumIndividual = 4 + 5 = 9 => non-additive
    assertEqualsDouble(agg.value, 8.5, Tol)
    assertEqualsDouble(sumIndividual.value, 9.0, Tol)
    assert(agg.value != sumIndividual.value, "aggregate should differ from sum of per-rival deltas")
