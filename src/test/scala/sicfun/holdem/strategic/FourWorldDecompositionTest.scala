package sicfun.holdem.strategic

class FourWorldDecompositionTest extends munit.FunSuite:

  private inline val Tol = 1e-12

  // -- Def 44: Four worlds from Q-function oracle --

  test("compute builds FourWorld from four value function evaluations"):
    val fw = FourWorldDecomposition.compute(
      vAttribClosedLoop = Ev(10.0),  // V^{1,1}
      vAttribOpenLoop = Ev(7.0),     // V^{1,0}
      vBlindClosedLoop = Ev(6.0),    // V^{0,1}
      vBlindOpenLoop = Ev(4.0)       // V^{0,0}
    )
    assertEqualsDouble(fw.v11.value, 10.0, Tol)
    assertEqualsDouble(fw.v10.value, 7.0, Tol)
    assertEqualsDouble(fw.v01.value, 6.0, Tol)
    assertEqualsDouble(fw.v00.value, 4.0, Tol)

  // -- Def 45: Control value --

  test("deltaControl = V^{0,1} - V^{0,0}"):
    val fw = FourWorldDecomposition.compute(Ev(10.0), Ev(7.0), Ev(6.0), Ev(4.0))
    assertEqualsDouble(fw.deltaControl.value, 2.0, Tol)

  // -- Def 46: Marginal signaling effect --

  test("deltaSigStar = V^{1,0} - V^{0,0}"):
    val fw = FourWorldDecomposition.compute(Ev(10.0), Ev(7.0), Ev(6.0), Ev(4.0))
    assertEqualsDouble(fw.deltaSigStar.value, 3.0, Tol)

  // -- Def 47: Interaction term --

  test("deltaInteraction = V^{1,1} - V^{1,0} - V^{0,1} + V^{0,0}"):
    val fw = FourWorldDecomposition.compute(Ev(10.0), Ev(7.0), Ev(6.0), Ev(4.0))
    // 10 - 7 - 6 + 4 = 1
    assertEqualsDouble(fw.deltaInteraction.value, 1.0, Tol)

  // -- Theorem 4: V^{1,1} = V^{0,0} + delta_cont + delta_sig* + delta_int --

  test("Theorem 4: exact aggregate decomposition identity"):
    val fw = FourWorldDecomposition.compute(Ev(10.0), Ev(7.0), Ev(6.0), Ev(4.0))
    val reconstructed = Ev(fw.v00.value + fw.deltaControl.value + fw.deltaSigStar.value + fw.deltaInteraction.value)
    assertEqualsDouble(reconstructed.value, fw.v11.value, Tol)

  test("Theorem 4 holds for negative values"):
    val fw = FourWorldDecomposition.compute(Ev(-2.0), Ev(-5.0), Ev(-3.0), Ev(-8.0))
    val reconstructed = Ev(fw.v00.value + fw.deltaControl.value + fw.deltaSigStar.value + fw.deltaInteraction.value)
    assertEqualsDouble(reconstructed.value, fw.v11.value, Tol)

  test("Theorem 4 holds for zero interaction (separable case)"):
    // V^{1,1} - V^{1,0} = V^{0,1} - V^{0,0} => delta_int = 0
    val fw = FourWorldDecomposition.compute(Ev(10.0), Ev(7.0), Ev(7.0), Ev(4.0))
    assertEqualsDouble(fw.deltaInteraction.value, 0.0, Tol)
    val reconstructed = Ev(fw.v00.value + fw.deltaControl.value + fw.deltaSigStar.value + fw.deltaInteraction.value)
    assertEqualsDouble(reconstructed.value, fw.v11.value, Tol)

  // -- Corollary 3: separability --

  test("Corollary 3: V^{1,1}-V^{1,0} == V^{0,1}-V^{0,0} implies delta_int == 0"):
    val fw = FourWorldDecomposition.compute(Ev(10.0), Ev(7.0), Ev(7.0), Ev(4.0))
    val lhs = fw.v11 - fw.v10
    val rhs = fw.v01 - fw.v00
    assertEqualsDouble(lhs.value, rhs.value, Tol)
    assertEqualsDouble(fw.deltaInteraction.value, 0.0, Tol)

  // -- Corollary 4: coarse interaction bound --

  test("Corollary 4: |delta_int| <= 4*Rmax/(1-gamma)"):
    val rMax = 100.0
    val gamma = 0.95
    val bound = 4.0 * rMax / (1.0 - gamma)
    val fw = FourWorldDecomposition.compute(
      Ev(bound / 4.0), Ev(-bound / 4.0), Ev(-bound / 4.0), Ev(bound / 4.0)
    )
    assert(fw.deltaInteraction.abs <= Ev(bound + Tol))

  // -- buildDeltaVocabulary --

  test("buildDeltaVocabulary assembles FourWorld + per-rival deltas"):
    val v1 = PlayerId("v1")
    val fw = FourWorldDecomposition.compute(Ev(10.0), Ev(7.0), Ev(6.0), Ev(4.0))
    val prd = SignalDecomposition.computePerRivalDelta(Ev(10.0), Ev(8.0), Ev(6.0))
    val vocab = FourWorldDecomposition.buildDeltaVocabulary(
      fourWorld = fw,
      perRivalDeltas = Map(v1 -> prd),
      deltaSigAggregate = Ev(4.0)
    )
    assertEquals(vocab.fourWorld, fw)
    assertEquals(vocab.perRivalDeltas.size, 1)
    assertEqualsDouble(vocab.deltaSigAggregate.value, 4.0, Tol)

  // -- Defs 48-49: Signaling sub-decomposition --

  test("deltaSigDesign = qDesign - qBlind"):
    val result = SignalingSubDecomposition.deltaSigDesign(qDesign = Ev(8.0), qBlind = Ev(6.0))
    assertEqualsDouble(result.value, 2.0, Tol)

  test("deltaSigReal = qAttrib - qDesign"):
    val result = SignalingSubDecomposition.deltaSigReal(qAttrib = Ev(10.0), qDesign = Ev(8.0))
    assertEqualsDouble(result.value, 2.0, Tol)

  // -- Theorem 3A: delta_sig = delta_sig,design + delta_sig,real --

  test("Theorem 3A: deltaSig == deltaSigDesign + deltaSigReal"):
    val qAttrib = Ev(15.0)
    val qDesign = Ev(11.0)
    val qBlind = Ev(7.0)
    val sig = SignalDecomposition.deltaSig(qAttrib, qBlind)
    val design = SignalingSubDecomposition.deltaSigDesign(qDesign, qBlind)
    val real = SignalingSubDecomposition.deltaSigReal(qAttrib, qDesign)
    assertEqualsDouble(sig.value, (design + real).value, Tol)

  test("computeSubDecomposition builds PerRivalSignalSubDecomposition"):
    val sub = SignalingSubDecomposition.compute(
      qAttrib = Ev(10.0), qDesign = Ev(8.0), qBlind = Ev(6.0)
    )
    assertEqualsDouble(sub.deltaSigDesign.value, 2.0, Tol)
    assertEqualsDouble(sub.deltaSigReal.value, 2.0, Tol)
    assertEqualsDouble(sub.total.value, 4.0, Tol)

  test("Theorem 3A: sub.total equals deltaSig for any values"):
    val qA = Ev(22.5)
    val qD = Ev(17.3)
    val qB = Ev(9.1)
    val sub = SignalingSubDecomposition.compute(qA, qD, qB)
    val sig = SignalDecomposition.deltaSig(qA, qB)
    assertEqualsDouble(sub.total.value, sig.value, Tol)
