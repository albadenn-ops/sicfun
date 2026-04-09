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

  // ---- FourWorld keyed accessors (Wave 1) -----------------------------------

  test("FourWorld keyed accessor: (Attrib, ClosedLoop) -> v11"):
    val fw = FourWorld(Ev(10.0), Ev(7.0), Ev(6.0), Ev(4.0))
    assertEqualsDouble(fw(GridWorld(LearningChannel.Attrib, PolicyScope.ClosedLoop)).value, 10.0, Tol)

  test("FourWorld keyed accessor: (Attrib, OpenLoop) -> v10"):
    val fw = FourWorld(Ev(10.0), Ev(7.0), Ev(6.0), Ev(4.0))
    assertEqualsDouble(fw(GridWorld(LearningChannel.Attrib, PolicyScope.OpenLoop)).value, 7.0, Tol)

  test("FourWorld keyed accessor: (Blind, ClosedLoop) -> v01"):
    val fw = FourWorld(Ev(10.0), Ev(7.0), Ev(6.0), Ev(4.0))
    assertEqualsDouble(fw(GridWorld(LearningChannel.Blind, PolicyScope.ClosedLoop)).value, 6.0, Tol)

  test("FourWorld keyed accessor: (Blind, OpenLoop) -> v00"):
    val fw = FourWorld(Ev(10.0), Ev(7.0), Ev(6.0), Ev(4.0))
    assertEqualsDouble(fw(GridWorld(LearningChannel.Blind, PolicyScope.OpenLoop)).value, 4.0, Tol)

  test("FourWorld keyed accessors round-trip all GridWorld.all elements"):
    val fw = FourWorld(Ev(10.0), Ev(7.0), Ev(6.0), Ev(4.0))
    val expected = Seq(4.0, 6.0, 7.0, 10.0)
    val actual = GridWorld.all.map(gw => fw(gw).value).sorted
    assertEquals(actual, expected.toIndexedSeq)

  test("Theorem 4 decomposition holds under keyed accessors"):
    val fw = FourWorld(Ev(10.0), Ev(7.0), Ev(6.0), Ev(4.0))
    val v00 = fw(GridWorld(LearningChannel.Blind, PolicyScope.OpenLoop))
    val reconstructed = v00 + fw.deltaControl + fw.deltaSigStar + fw.deltaInteraction
    assertEqualsDouble(
      reconstructed.value,
      fw(GridWorld(LearningChannel.Attrib, PolicyScope.ClosedLoop)).value,
      Tol
    )

  // ---- ChainEdgeDelta / ChainRiskDelta (Wave 1) ----------------------------

  test("ChainEdgeDelta stores from/to/delta"):
    val from = ChainWorld(LearningChannel.Blind, ShowdownMode.Off)
    val to = ChainWorld(LearningChannel.Ref, ShowdownMode.Off)
    val edge = ChainEdgeDelta(from, to, Ev(3.5))
    assertEquals(edge.from, from)
    assertEquals(edge.to, to)
    assertEqualsDouble(edge.delta.value, 3.5, Tol)

  test("ChainRiskDelta stores from/to/riskDelta"):
    val from = ChainWorld(LearningChannel.Attrib, ShowdownMode.Off)
    val to = ChainWorld(LearningChannel.Attrib, ShowdownMode.On)
    val risk = ChainRiskDelta(from, to, Ev(-1.2))
    assertEquals(risk.from, from)
    assertEquals(risk.to, to)
    assertEqualsDouble(risk.riskDelta.value, -1.2, Tol)

  // ---- Proposition 8.1: TelescopicEdgeDecomposition -------------------------

  test("TelescopicEdgeDecomposition: single edge chain"):
    val chain = IndexedSeq(
      ChainWorld(LearningChannel.Blind, ShowdownMode.Off),
      ChainWorld(LearningChannel.Ref, ShowdownMode.Off)
    )
    val qValues = IndexedSeq(Ev(3.0), Ev(7.0))
    val edges = TelescopicEdgeDecomposition.computeEdgeDeltas(chain, qValues)
    assertEquals(edges.size, 1)
    assertEquals(edges(0).from, chain(0))
    assertEquals(edges(0).to, chain(1))
    assertEqualsDouble(edges(0).delta.value, 4.0, Tol)

  test("TelescopicEdgeDecomposition: multi-edge canonical chain"):
    val chain = ChainWorld.canonicalChain
    val qValues = IndexedSeq(Ev(1.0), Ev(3.0), Ev(5.5), Ev(8.0))
    val edges = TelescopicEdgeDecomposition.computeEdgeDeltas(chain, qValues)
    assertEquals(edges.size, 3)
    assertEqualsDouble(edges(0).delta.value, 2.0, Tol)  // 3.0 - 1.0
    assertEqualsDouble(edges(1).delta.value, 2.5, Tol)  // 5.5 - 3.0
    assertEqualsDouble(edges(2).delta.value, 2.5, Tol)  // 8.0 - 5.5

  test("TelescopicEdgeDecomposition: telescopic identity sum(edgeDeltas) == totalGap"):
    val chain = ChainWorld.canonicalChain
    val qValues = IndexedSeq(Ev(1.0), Ev(3.0), Ev(5.5), Ev(8.0))
    val (totalGap, edges) = TelescopicEdgeDecomposition.decompose(chain, qValues)
    val edgeSum = edges.foldLeft(Ev.Zero)((acc, e) => acc + e.delta)
    assertEqualsDouble(totalGap.value, 7.0, Tol)  // 8.0 - 1.0
    assertEqualsDouble(edgeSum.value, totalGap.value, Tol)

  test("TelescopicEdgeDecomposition: telescopic identity holds with negative deltas"):
    val chain = ChainWorld.canonicalChain
    val qValues = IndexedSeq(Ev(10.0), Ev(8.0), Ev(5.0), Ev(2.0))
    val (totalGap, edges) = TelescopicEdgeDecomposition.decompose(chain, qValues)
    val edgeSum = edges.foldLeft(Ev.Zero)((acc, e) => acc + e.delta)
    assertEqualsDouble(totalGap.value, -8.0, Tol)  // 2.0 - 10.0
    assertEqualsDouble(edgeSum.value, totalGap.value, Tol)

  test("TelescopicEdgeDecomposition: single-element chain returns zero gap and no edges"):
    val chain = IndexedSeq(ChainWorld(LearningChannel.Blind, ShowdownMode.Off))
    val qValues = IndexedSeq(Ev(5.0))
    val (totalGap, edges) = TelescopicEdgeDecomposition.decompose(chain, qValues)
    assertEquals(edges.size, 0)
    assertEqualsDouble(totalGap.value, 0.0, Tol)

  test("TelescopicEdgeDecomposition: empty chain returns zero gap and no edges"):
    val chain = IndexedSeq.empty[ChainWorld]
    val qValues = IndexedSeq.empty[Ev]
    val (totalGap, edges) = TelescopicEdgeDecomposition.decompose(chain, qValues)
    assertEquals(edges.size, 0)
    assertEqualsDouble(totalGap.value, 0.0, Tol)

  test("TelescopicEdgeDecomposition: requires chain and qValues same length"):
    val chain = IndexedSeq(
      ChainWorld(LearningChannel.Blind, ShowdownMode.Off),
      ChainWorld(LearningChannel.Ref, ShowdownMode.Off)
    )
    val qValues = IndexedSeq(Ev(1.0)) // mismatched length
    intercept[IllegalArgumentException]:
      TelescopicEdgeDecomposition.computeEdgeDeltas(chain, qValues)

  test("TelescopicEdgeDecomposition: equal Q-values yield zero deltas"):
    val chain = ChainWorld.canonicalChain
    val qValues = IndexedSeq(Ev(5.0), Ev(5.0), Ev(5.0), Ev(5.0))
    val (totalGap, edges) = TelescopicEdgeDecomposition.decompose(chain, qValues)
    assertEqualsDouble(totalGap.value, 0.0, Tol)
    edges.foreach(e => assertEqualsDouble(e.delta.value, 0.0, Tol))

  test("TelescopicEdgeDecomposition: edge from/to matches chain ordering"):
    val chain = ChainWorld.canonicalChain
    val qValues = IndexedSeq(Ev(1.0), Ev(2.0), Ev(4.0), Ev(7.0))
    val edges = TelescopicEdgeDecomposition.computeEdgeDeltas(chain, qValues)
    for k <- edges.indices do
      assertEquals(edges(k).from, chain(k))
      assertEquals(edges(k).to, chain(k + 1))

  // ---- Def 47A: ChainBaselineQ ------------------------------------------------

  test("ChainBaselineQ: keyed access by ChainWorld"):
    val qMap = Map(
      ChainWorld(LearningChannel.Blind, ShowdownMode.Off) -> Ev(1.0),
      ChainWorld(LearningChannel.Ref, ShowdownMode.Off) -> Ev(3.0),
      ChainWorld(LearningChannel.Attrib, ShowdownMode.Off) -> Ev(5.5),
      ChainWorld(LearningChannel.Attrib, ShowdownMode.On) -> Ev(8.0)
    )
    val cbq = ChainBaselineQ(qMap)
    assertEqualsDouble(cbq(ChainWorld(LearningChannel.Blind, ShowdownMode.Off)).value, 1.0, Tol)
    assertEqualsDouble(cbq(ChainWorld(LearningChannel.Attrib, ShowdownMode.On)).value, 8.0, Tol)

  test("ChainBaselineQ: alongChain extracts in order"):
    val qMap = Map(
      ChainWorld(LearningChannel.Blind, ShowdownMode.Off) -> Ev(1.0),
      ChainWorld(LearningChannel.Ref, ShowdownMode.Off) -> Ev(3.0),
      ChainWorld(LearningChannel.Attrib, ShowdownMode.Off) -> Ev(5.5),
      ChainWorld(LearningChannel.Attrib, ShowdownMode.On) -> Ev(8.0)
    )
    val cbq = ChainBaselineQ(qMap)
    val values = cbq.alongChain(ChainWorld.canonicalChain)
    assertEquals(values.size, 4)
    assertEqualsDouble(values(0).value, 1.0, Tol)
    assertEqualsDouble(values(1).value, 3.0, Tol)
    assertEqualsDouble(values(2).value, 5.5, Tol)
    assertEqualsDouble(values(3).value, 8.0, Tol)

  test("ChainBaselineQ: canonicalEdgeDeltas matches TelescopicEdgeDecomposition"):
    val qMap = Map(
      ChainWorld(LearningChannel.Blind, ShowdownMode.Off) -> Ev(1.0),
      ChainWorld(LearningChannel.Ref, ShowdownMode.Off) -> Ev(3.0),
      ChainWorld(LearningChannel.Attrib, ShowdownMode.Off) -> Ev(5.5),
      ChainWorld(LearningChannel.Attrib, ShowdownMode.On) -> Ev(8.0)
    )
    val cbq = ChainBaselineQ(qMap)
    val edges = cbq.canonicalEdgeDeltas
    assertEquals(edges.size, 3)
    assertEqualsDouble(edges(0).delta.value, 2.0, Tol)
    assertEqualsDouble(edges(1).delta.value, 2.5, Tol)
    assertEqualsDouble(edges(2).delta.value, 2.5, Tol)
    // Telescopic identity: sum of edges == total gap
    val edgeSum = edges.foldLeft(Ev.Zero)((acc, e) => acc + e.delta)
    assertEqualsDouble(edgeSum.value, 7.0, Tol)

  test("ChainBaselineQ: rejects empty map"):
    intercept[IllegalArgumentException]:
      ChainBaselineQ(Map.empty)

  test("ChainBaselineQ: missing world throws on access"):
    val cbq = ChainBaselineQ(Map(
      ChainWorld(LearningChannel.Blind, ShowdownMode.Off) -> Ev(1.0)
    ))
    intercept[IllegalArgumentException]:
      cbq(ChainWorld(LearningChannel.Ref, ShowdownMode.Off))
