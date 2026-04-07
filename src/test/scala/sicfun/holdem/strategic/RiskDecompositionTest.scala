package sicfun.holdem.strategic

class RiskDecompositionTest extends munit.FunSuite:

  private inline val Tol = 1e-12

  // ---- Def 56A: Chain-indexed robust one-step loss ----

  test("chainRobustLoss: max over belief points"):
    val baseline = IndexedSeq(Ev(10.0), Ev(8.0), Ev(12.0))
    val qValues  = IndexedSeq(Ev(9.0), Ev(3.0), Ev(11.0))
    // losses: 1.0, 5.0, 1.0 → max = 5.0
    val loss = RiskDecomposition.chainRobustLoss(baseline, qValues)
    assertEqualsDouble(loss.value, 5.0, Tol)

  test("chainRobustLoss: single belief point"):
    val loss = RiskDecomposition.chainRobustLoss(IndexedSeq(Ev(10.0)), IndexedSeq(Ev(7.0)))
    assertEqualsDouble(loss.value, 3.0, Tol)

  test("chainRobustLoss: zero when Q equals baseline"):
    val vals = IndexedSeq(Ev(5.0), Ev(3.0))
    val loss = RiskDecomposition.chainRobustLoss(vals, vals)
    assertEqualsDouble(loss.value, 0.0, Tol)

  test("chainRobustLoss: negative when Q exceeds baseline everywhere"):
    val baseline = IndexedSeq(Ev(3.0), Ev(2.0))
    val qValues  = IndexedSeq(Ev(5.0), Ev(4.0))
    val loss = RiskDecomposition.chainRobustLoss(baseline, qValues)
    assertEqualsDouble(loss.value, -2.0, Tol)

  // ---- Def 56B: Risk increment ----

  test("riskIncrement: positive when next world has more risk"):
    val inc = RiskDecomposition.riskIncrement(Ev(2.0), Ev(5.0))
    assertEqualsDouble(inc.value, 3.0, Tol)

  test("riskIncrement: negative when next world has less risk"):
    val inc = RiskDecomposition.riskIncrement(Ev(5.0), Ev(3.0))
    assertEqualsDouble(inc.value, -2.0, Tol)

  test("riskIncrement: zero when equal"):
    val inc = RiskDecomposition.riskIncrement(Ev(4.0), Ev(4.0))
    assertEqualsDouble(inc.value, 0.0, Tol)

  // ---- Proposition 9.7: Telescopic risk decomposition ----

  test("telescopic identity: total gap equals sum of increments"):
    val chain = ChainWorld.canonicalChain
    val losses = IndexedSeq(Ev(1.0), Ev(3.0), Ev(2.5), Ev(5.0))
    val profile = RiskDecomposition.ChainRiskProfile(chain, losses)
    val (totalGap, increments) = profile.telescopicDecomposition

    // Total gap: 5.0 - 1.0 = 4.0
    assertEqualsDouble(totalGap.value, 4.0, Tol)

    // Sum of increments must equal total gap
    val sumIncrements = increments.map(_.riskDelta).reduce(_ + _)
    assertEqualsDouble(sumIncrements.value, totalGap.value, Tol)

  test("telescopic identity: single chain edge"):
    val chain = IndexedSeq(
      ChainWorld(LearningChannel.Blind, ShowdownMode.Off),
      ChainWorld(LearningChannel.Ref, ShowdownMode.Off)
    )
    val losses = IndexedSeq(Ev(2.0), Ev(7.0))
    val profile = RiskDecomposition.ChainRiskProfile(chain, losses)
    val (totalGap, increments) = profile.telescopicDecomposition

    assertEqualsDouble(totalGap.value, 5.0, Tol)
    assertEquals(increments.size, 1)
    assertEqualsDouble(increments.head.riskDelta.value, 5.0, Tol)

  test("telescopic identity: all losses equal implies zero gap"):
    val chain = ChainWorld.canonicalChain
    val losses = IndexedSeq(Ev(3.0), Ev(3.0), Ev(3.0), Ev(3.0))
    val profile = RiskDecomposition.ChainRiskProfile(chain, losses)
    val (totalGap, increments) = profile.telescopicDecomposition

    assertEqualsDouble(totalGap.value, 0.0, Tol)
    increments.foreach(inc => assertEqualsDouble(inc.riskDelta.value, 0.0, Tol))

  test("telescopic identity: arbitrary 4-world chain"):
    for
      l0 <- Seq(Ev(-5.0), Ev(0.0), Ev(10.0))
      l1 <- Seq(Ev(-3.0), Ev(0.0), Ev(8.0))
      l2 <- Seq(Ev(-1.0), Ev(0.0), Ev(6.0))
      l3 <- Seq(Ev(-2.0), Ev(0.0), Ev(12.0))
    do
      val chain = ChainWorld.canonicalChain
      val losses = IndexedSeq(l0, l1, l2, l3)
      val profile = RiskDecomposition.ChainRiskProfile(chain, losses)
      val (totalGap, increments) = profile.telescopicDecomposition
      val sumInc = increments.map(_.riskDelta).reduce(_ + _)
      assertEqualsDouble(sumInc.value, totalGap.value, Tol)
      assertEqualsDouble(totalGap.value, (l3 - l0).value, Tol)

  // ---- Def 56C: Marginal efficiency ----

  test("marginalEfficiency: defined when risk increment nonzero"):
    val eff = RiskDecomposition.marginalEfficiency(Ev(6.0), Ev(3.0))
    assert(eff.isDefined)
    assertEqualsDouble(eff.get, 2.0, Tol)

  test("marginalEfficiency: undefined when risk increment is zero"):
    val eff = RiskDecomposition.marginalEfficiency(Ev(6.0), Ev(0.0))
    assert(eff.isEmpty)

  test("marginalEfficiency: negative when value and risk have opposite signs"):
    val eff = RiskDecomposition.marginalEfficiency(Ev(-2.0), Ev(4.0))
    assert(eff.isDefined)
    assertEqualsDouble(eff.get, -0.5, Tol)

  test("marginalEfficiency: handles negative risk increment"):
    val eff = RiskDecomposition.marginalEfficiency(Ev(3.0), Ev(-1.5))
    assert(eff.isDefined)
    assertEqualsDouble(eff.get, -2.0, Tol)

  // ---- computeProfile ----

  test("computeProfile: produces correct chain risk profile"):
    val chain = IndexedSeq(
      ChainWorld(LearningChannel.Blind, ShowdownMode.Off),
      ChainWorld(LearningChannel.Ref, ShowdownMode.Off)
    )
    val baseline = IndexedSeq(Ev(10.0), Ev(8.0))
    val qWorld0  = IndexedSeq(Ev(9.0), Ev(6.0))  // losses: 1.0, 2.0 → max = 2.0
    val qWorld1  = IndexedSeq(Ev(7.0), Ev(3.0))  // losses: 3.0, 5.0 → max = 5.0
    val profile = RiskDecomposition.computeProfile(chain, baseline, IndexedSeq(qWorld0, qWorld1))

    assertEqualsDouble(profile.robustLosses(0).value, 2.0, Tol)
    assertEqualsDouble(profile.robustLosses(1).value, 5.0, Tol)

  // ---- edgeEfficiencies ----

  test("edgeEfficiencies: pairs value and risk edges"):
    val w0 = ChainWorld(LearningChannel.Blind, ShowdownMode.Off)
    val w1 = ChainWorld(LearningChannel.Ref, ShowdownMode.Off)
    val valueEdges = IndexedSeq(ChainEdgeDelta(w0, w1, Ev(4.0)))
    val riskEdges  = IndexedSeq(ChainRiskDelta(w0, w1, Ev(2.0)))
    val effs = RiskDecomposition.edgeEfficiencies(valueEdges, riskEdges)

    assertEquals(effs.size, 1)
    assertEquals(effs.head._1, w0)
    assertEquals(effs.head._2, w1)
    assert(effs.head._3.isDefined)
    assertEqualsDouble(effs.head._3.get, 2.0, Tol)

  // ---- Chain ordering consistency ----

  test("risk profile uses same chain ordering as ChainWorld.canonicalChain"):
    val chain = ChainWorld.canonicalChain
    val losses = IndexedSeq(Ev(1.0), Ev(2.0), Ev(3.0), Ev(4.0))
    val profile = RiskDecomposition.ChainRiskProfile(chain, losses)
    val increments = profile.riskIncrements

    // Verify edges follow canonical chain ordering
    assertEquals(increments(0).from, ChainWorld(LearningChannel.Blind, ShowdownMode.Off))
    assertEquals(increments(0).to, ChainWorld(LearningChannel.Ref, ShowdownMode.Off))
    assertEquals(increments(1).from, ChainWorld(LearningChannel.Ref, ShowdownMode.Off))
    assertEquals(increments(1).to, ChainWorld(LearningChannel.Attrib, ShowdownMode.Off))
    assertEquals(increments(2).from, ChainWorld(LearningChannel.Attrib, ShowdownMode.Off))
    assertEquals(increments(2).to, ChainWorld(LearningChannel.Attrib, ShowdownMode.On))
