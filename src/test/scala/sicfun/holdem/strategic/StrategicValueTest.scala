package sicfun.holdem.strategic

class StrategicValueTest extends munit.FunSuite:

  private inline val Tol = 1e-12

  test("FourWorld stores all four value functions"):
    val fw = FourWorld(v11 = Ev(10.0), v10 = Ev(7.0), v01 = Ev(6.0), v00 = Ev(4.0))
    assertEquals(fw.v11.value, 10.0)
    assertEquals(fw.v10.value, 7.0)
    assertEquals(fw.v01.value, 6.0)
    assertEquals(fw.v00.value, 4.0)

  test("deltaControl = V^{0,1} - V^{0,0}"):
    val fw = FourWorld(Ev(10.0), Ev(7.0), Ev(6.0), Ev(4.0))
    assertEqualsDouble(fw.deltaControl.value, 2.0, Tol)

  test("deltaSigStar = V^{1,0} - V^{0,0}"):
    val fw = FourWorld(Ev(10.0), Ev(7.0), Ev(6.0), Ev(4.0))
    assertEqualsDouble(fw.deltaSigStar.value, 3.0, Tol)

  test("deltaInteraction = V^{1,1} - V^{1,0} - V^{0,1} + V^{0,0}"):
    val fw = FourWorld(Ev(10.0), Ev(7.0), Ev(6.0), Ev(4.0))
    assertEqualsDouble(fw.deltaInteraction.value, 1.0, Tol)

  test("Theorem 4: exact aggregate decomposition"):
    val fw = FourWorld(Ev(10.0), Ev(7.0), Ev(6.0), Ev(4.0))
    val reconstructed = fw.v00 + fw.deltaControl + fw.deltaSigStar + fw.deltaInteraction
    assertEqualsDouble(reconstructed.value, fw.v11.value, Tol)

  test("Theorem 4 holds for negative values"):
    val fw = FourWorld(Ev(-2.0), Ev(-5.0), Ev(-3.0), Ev(-8.0))
    val reconstructed = fw.v00 + fw.deltaControl + fw.deltaSigStar + fw.deltaInteraction
    assertEqualsDouble(reconstructed.value, fw.v11.value, Tol)

  test("Theorem 4 holds when all worlds are equal"):
    val fw = FourWorld(Ev(5.0), Ev(5.0), Ev(5.0), Ev(5.0))
    assertEqualsDouble(fw.deltaControl.value, 0.0, Tol)
    assertEqualsDouble(fw.deltaSigStar.value, 0.0, Tol)
    assertEqualsDouble(fw.deltaInteraction.value, 0.0, Tol)
    val reconstructed = fw.v00 + fw.deltaControl + fw.deltaSigStar + fw.deltaInteraction
    assertEqualsDouble(reconstructed.value, fw.v11.value, Tol)

  test("Corollary 3: if V^{1,1}-V^{1,0} = V^{0,1}-V^{0,0} then delta_int = 0"):
    val fw = FourWorld(Ev(10.0), Ev(7.0), Ev(7.0), Ev(4.0))
    assertEqualsDouble(fw.deltaInteraction.value, 0.0, Tol)

  test("Corollary 4: |delta_int| <= 4*R_max/(1-gamma)"):
    val rMax = 100.0
    val gamma = 0.99
    val bound = 4.0 * rMax / (1.0 - gamma)
    val vBound = rMax / (1.0 - gamma)
    val fw = FourWorld(Ev(vBound), Ev(-vBound), Ev(-vBound), Ev(vBound))
    assert(fw.deltaInteraction.abs <= Ev(bound + Tol))

  test("PerRivalDelta: Theorem 3 -- sig = pass + manip"):
    val d = PerRivalDelta(deltaSig = Ev(5.0), deltaPass = Ev(2.0), deltaManip = Ev(3.0))
    assertEqualsDouble(d.deltaSig.value, (d.deltaPass + d.deltaManip).value, Tol)

  test("PerRivalDelta: negative deltaPass signals damaging leak (Corollary 1)"):
    val d = PerRivalDelta(deltaSig = Ev(-1.0), deltaPass = Ev(-3.0), deltaManip = Ev(2.0))
    assert(d.isDamagingLeak)

  test("PerRivalDelta: non-negative deltaPass is not a damaging leak"):
    val d = PerRivalDelta(deltaSig = Ev(4.0), deltaPass = Ev(1.0), deltaManip = Ev(3.0))
    assert(!d.isDamagingLeak)

  test("Theorem 5: correct beliefs -> hasCorrectBeliefs true"):
    val d = PerRivalDelta(deltaSig = Ev(2.0), deltaPass = Ev(2.0), deltaManip = Ev(0.0))
    assert(d.hasCorrectBeliefs)

  test("Theorem 5: incorrect beliefs -> hasCorrectBeliefs false"):
    val d = PerRivalDelta(deltaSig = Ev(5.0), deltaPass = Ev(2.0), deltaManip = Ev(3.0))
    assert(!d.hasCorrectBeliefs)

  test("PerRivalSignalSubDecomposition: Theorem 3A -- sig = design + real"):
    val sub = PerRivalSignalSubDecomposition(deltaSigDesign = Ev(1.5), deltaSigReal = Ev(3.5))
    assertEqualsDouble(sub.total.value, 5.0, Tol)

  test("DeltaVocabulary contains per-rival and aggregate primitives"):
    val perRival = Map(
      PlayerId("v1") -> PerRivalDelta(Ev(5.0), Ev(2.0), Ev(3.0))
    )
    val fourWorld = FourWorld(Ev(10.0), Ev(7.0), Ev(6.0), Ev(4.0))
    val vocab = DeltaVocabulary(
      fourWorld = fourWorld,
      perRivalDeltas = perRival,
      deltaSigAggregate = Ev(5.0)
    )
    assertEquals(vocab.perRivalDeltas.size, 1)
    assertEqualsDouble(vocab.fourWorld.deltaControl.value, 2.0, Tol)
