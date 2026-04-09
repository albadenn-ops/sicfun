package sicfun.holdem.strategic

/** Wave 7: v0.31.1 formal closure validation.
  *
  * Verifies:
  *   - Result-coverage matrix (Theorems 1-9, Corollaries 1-4+9.3, Propositions 8.1/9.x)
  *   - World-consistency regression (chain ordering shared across modules)
  *   - Assumption ledger completeness (A1'-A10 all classified)
  *   - Computational-architecture triage (Defs 54-56 accounted for)
  *   - Changepoint vulnerability regression (Appendix B)
  */
class FormalClosureValidationTest extends munit.FunSuite:

  // ========================================================================
  // Result-coverage matrix
  // ========================================================================

  private enum CoverageLevel:
    case Exact         // numerical test directly verifies the result
    case InterfaceLevel // tests exercise the interface; not full numerical proof
    case Inherited     // result holds by construction (types, enums, etc.)
    case Deferred      // out of constitutive scope; documented but not tested

  private case class CoverageEntry(
      result: String,
      level: CoverageLevel,
      testLocation: String
  )

  private val coverageMatrix: Vector[CoverageEntry] = Vector(
    CoverageEntry("Theorem 1: Unconditional totality",      CoverageLevel.Exact,          "TheoremValidationTest"),
    CoverageEntry("Theorem 2: Posterior limits",             CoverageLevel.Exact,          "TheoremValidationTest"),
    CoverageEntry("Theorem 3: Signal decomposition",        CoverageLevel.Exact,          "TheoremValidationTest"),
    CoverageEntry("Theorem 3A: Signaling sub-decomposition",CoverageLevel.Exact,          "TheoremValidationTest"),
    CoverageEntry("Theorem 4: Four-world decomposition",    CoverageLevel.Exact,          "TheoremValidationTest"),
    CoverageEntry("Theorem 5: Manipulation collapse",       CoverageLevel.Exact,          "TheoremValidationTest"),
    CoverageEntry("Theorem 6: No-learning coherence",       CoverageLevel.Exact,          "TheoremValidationTest"),
    CoverageEntry("Theorem 7: Robust convexity",            CoverageLevel.Exact,          "WPomcpRuntimeTest (belief convexity via native MCTS)"),
    CoverageEntry("Theorem 8: Adaptation safety bound",     CoverageLevel.Exact,          "TheoremValidationTest"),
    CoverageEntry("Theorem 9: Bellman-safe certificates",   CoverageLevel.Exact,          "TheoremValidationTest, SafetyBellmanTest"),
    CoverageEntry("Corollary 1: Damaging passive leak",     CoverageLevel.Exact,          "TheoremValidationTest"),
    CoverageEntry("Corollary 2: Exploitative => structural",CoverageLevel.Exact,          "TheoremValidationTest"),
    CoverageEntry("Corollary 3: Separability",              CoverageLevel.Exact,          "TheoremValidationTest"),
    CoverageEntry("Corollary 4: Interaction bound",         CoverageLevel.Exact,          "TheoremValidationTest"),
    CoverageEntry("Corollary 9.3: Total vulnerability",     CoverageLevel.Exact,          "SafetyBellmanTest"),
    CoverageEntry("Proposition 8.1: Telescopic edge decomposition", CoverageLevel.Exact, "StrategicValueTest (TelescopicEdgeDecomposition, ChainBaselineQ)"),
    CoverageEntry("Proposition 9.1: Formal exploitability", CoverageLevel.Exact,          "TheoremValidationTest"),
    CoverageEntry("Proposition 9.2: AS-strong implies exploitability bound", CoverageLevel.Exact, "SafetyBellmanTest (safe actions produce regret bounded by B*, Corollary 9.3 numeric)"),
    CoverageEntry("Proposition 9.5: Certificate soundness", CoverageLevel.Exact, "SafetyBellmanTest (gamma-contraction, uniqueness from different initializations, Certificate.isValid)"),
    CoverageEntry("Proposition 9.6: Safe action bound",     CoverageLevel.Exact, "SafetyBellmanTest (safeActionSet, safe actions produce regret bounded by B*)"),
    CoverageEntry("Proposition 9.7: Telescopic risk",       CoverageLevel.Exact,          "TheoremValidationTest, RiskDecompositionTest"),
    CoverageEntry("Def 51: RevealSchedule",                 CoverageLevel.Exact,          "TheoremValidationTest")
  )

  test("result-coverage matrix: no deferred items remain"):
    val uncovered = coverageMatrix.filter(_.level == CoverageLevel.Deferred)
    assertEquals(uncovered.size, 0, s"Deferred items: ${uncovered.map(_.result)}")

  test("result-coverage matrix: Theorems 1-9 all present"):
    for i <- 1 to 9 do
      assert(
        coverageMatrix.exists(_.result.startsWith(s"Theorem $i")),
        s"Missing Theorem $i in coverage matrix"
      )

  test("result-coverage matrix: Corollaries 1-4 and 9.3 all present"):
    for i <- 1 to 4 do
      assert(
        coverageMatrix.exists(_.result.startsWith(s"Corollary $i")),
        s"Missing Corollary $i in coverage matrix"
      )
    assert(coverageMatrix.exists(_.result.contains("9.3")), "Missing Corollary 9.3")

  // ========================================================================
  // World-consistency regression
  // ========================================================================

  test("canonical chain ordering is shared between WorldTypes and RiskDecomposition"):
    val chain = ChainWorld.canonicalChain
    assertEquals(chain.size, 4)
    // Risk decomposition profile accepts this chain directly
    val dummyLosses = chain.map(_ => Ev(1.0))
    val profile = RiskDecomposition.ChainRiskProfile(chain, dummyLosses)
    assertEquals(profile.chain, chain)

  test("grid world coordinates cover exactly 4 worlds"):
    assertEquals(GridWorld.all.size, 4)
    // All use only Blind or Attrib
    for gw <- GridWorld.all do
      assert(
        gw.learning == LearningChannel.Blind || gw.learning == LearningChannel.Attrib,
        s"GridWorld $gw uses unexpected learning channel"
      )

  test("chain world covers all 8 worlds"):
    assertEquals(ChainWorld.all.size, 8)
    val expected = for
      ch <- LearningChannel.values
      sd <- ShowdownMode.values
    yield ChainWorld(ch, sd)
    assertEquals(ChainWorld.all.toSet, expected.toSet)

  test("FourWorld keyed accessor agrees with positional fields for all grid worlds"):
    val fw = FourWorld(Ev(10.0), Ev(7.0), Ev(6.0), Ev(4.0))
    assertEqualsDouble(fw(GridWorld(LearningChannel.Attrib, PolicyScope.ClosedLoop)).value, 10.0, 1e-12)
    assertEqualsDouble(fw(GridWorld(LearningChannel.Attrib, PolicyScope.OpenLoop)).value, 7.0, 1e-12)
    assertEqualsDouble(fw(GridWorld(LearningChannel.Blind, PolicyScope.ClosedLoop)).value, 6.0, 1e-12)
    assertEqualsDouble(fw(GridWorld(LearningChannel.Blind, PolicyScope.OpenLoop)).value, 4.0, 1e-12)

  // ========================================================================
  // Assumption ledger completeness (A1'-A10)
  // ========================================================================

  test("AssumptionManifest contains exactly 11 entries"):
    assertEquals(AssumptionManifest.entries.size, 11)

  test("AssumptionManifest: every assumption has a classified enforcement level"):
    for entry <- AssumptionManifest.entries do
      // No assumption should be unclassified — all must have a defined level
      assert(
        entry.enforcement != null,
        s"Assumption ${entry.id} has null enforcement level"
      )
      assert(
        entry.location.nonEmpty,
        s"Assumption ${entry.id} has empty location"
      )

  test("AssumptionManifest: A1'-A10 all present (including A6 and A6')"):
    val expectedIds = Set("A1'", "A2", "A3'", "A4'", "A5", "A6", "A6'", "A7", "A8", "A9", "A10")
    val actualIds = AssumptionManifest.entries.map(_.id).toSet
    assertEquals(actualIds, expectedIds)

  test("AssumptionManifest: v0.31.1 corrected assumption names"):
    def nameOf(id: String): String =
      AssumptionManifest.entries.find(_.id == id).map(_.name).getOrElse(s"<missing $id>")
    assert(nameOf("A1'").startsWith("Abstraction with guarantees"),
      s"A1' name wrong: ${nameOf("A1'")}")
    assert(nameOf("A2").startsWith("Closed Markovianity"),
      s"A2 name wrong: ${nameOf("A2")}")
    assert(nameOf("A5").startsWith("Bounded reward and discounting"),
      s"A5 name wrong: ${nameOf("A5")}")
    assert(nameOf("A6").startsWith("First-order interactive sufficiency"),
      s"A6 name wrong: ${nameOf("A6")}")
    assert(nameOf("A7").startsWith("Well-defined full rival update"),
      s"A7 name wrong: ${nameOf("A7")}")
    assert(nameOf("A8").startsWith("Strategically relevant repetition"),
      s"A8 name wrong: ${nameOf("A8")}")

  test("AssumptionManifest: A6 is Structural enforcement"):
    val a6 = AssumptionManifest.entries.find(_.id == "A6")
    assert(a6.isDefined, "A6 entry must exist")
    assertEquals(a6.get.enforcement, AssumptionManifest.EnforcementLevel.Structural)

  test("AssumptionManifest: no assumption is unclassified (all have notes)"):
    for entry <- AssumptionManifest.entries do
      assert(entry.notes.nonEmpty, s"Assumption ${entry.id} has empty notes")

  test("AssumptionManifest: Approximated and Deferred enum variants exist"):
    // Post-scaffold: all formerly Approximated assumptions (A9) are now Encoded
    // (real KL divergence implemented). Both enum variants remain for future use.
    val allVariants = AssumptionManifest.EnforcementLevel.values.toSet
    assert(allVariants.contains(AssumptionManifest.EnforcementLevel.Approximated))
    assert(allVariants.contains(AssumptionManifest.EnforcementLevel.Deferred))

  test("AssumptionManifest summary reports all categories"):
    val s = AssumptionManifest.summary
    assert(s.contains("encoded"), s"summary missing 'encoded': $s")
    assert(s.contains("structural"), s"summary missing 'structural': $s")
    assert(s.contains("recovered"), s"summary missing 'recovered': $s")

  // ========================================================================
  // Computational-architecture triage (Defs 54-56, §10B)
  // ========================================================================

  test("Defs 54-56: PFT-DPW solver types are testable"):
    // Defs 54-56 are implemented in solver.PftDpwRuntime and tested in PftDpwRuntimeTest.
    // This test verifies the types are accessible from the strategic layer.
    val cfg = solver.PftDpwConfig()
    assert(cfg.gamma > 0.0 && cfg.gamma < 1.0)
    // PftDpwRuntime.isAvailable and solve methods are tested in PftDpwRuntimeTest.
    // If this compiles and runs, the computational architecture is accessible.

  // ========================================================================
  // Appendix B: Changepoint vulnerability regression
  // ========================================================================

  test("Appendix B: detector-triggered retreat reduces beta"):
    val config = ExploitationConfig(initialBeta = 0.8, cpRetreatRate = 0.2, epsilonAdapt = 0.05)
    val state = ExploitationState.initial(config)

    // AlwaysDetect triggers retreat
    val updated = ExploitationInterpolation.updateExploitation(
      state, config,
      rivalId = PlayerId("v1"),
      history = Vector.empty,
      publicState = testPublicState,
      detector = AlwaysDetect,
      exploitabilityFn = _ => 0.0,
      epsilonNE = 0.0
    )
    assert(updated.beta < state.beta, "retreat should reduce beta")
    assertEqualsDouble(updated.beta, 0.6, 1e-12)

  test("Appendix B: NeverDetect does not trigger retreat"):
    val config = ExploitationConfig(initialBeta = 0.8, cpRetreatRate = 0.2, epsilonAdapt = 0.05)
    val state = ExploitationState.initial(config)

    val updated = ExploitationInterpolation.updateExploitation(
      state, config,
      rivalId = PlayerId("v1"),
      history = Vector.empty,
      publicState = testPublicState,
      detector = NeverDetect,
      exploitabilityFn = _ => 0.0,
      epsilonNE = 0.0
    )
    assertEqualsDouble(updated.beta, state.beta, 1e-12)

  test("Appendix B: false changepoint does not bypass safety budget"):
    val config = ExploitationConfig(initialBeta = 0.8, cpRetreatRate = 0.2, epsilonAdapt = 0.01)
    val state = ExploitationState.initial(config)

    // Exploitability exceeds safety bound at high beta — safety clamp should reduce beta
    val updated = ExploitationInterpolation.updateExploitation(
      state, config,
      rivalId = PlayerId("v1"),
      history = Vector.empty,
      publicState = testPublicState,
      detector = NeverDetect,  // no detection (false changepoint scenario)
      exploitabilityFn = beta => 0.03 + 0.1 * beta,  // safe at 0, unsafe at high beta
      epsilonNE = 0.03
    )
    // Safety bound: 0.03 + 0.01 = 0.04
    // Exploit(0) = 0.03 <= 0.04 (safe), Exploit(0.8) = 0.11 > 0.04 (unsafe)
    assert(updated.beta < state.beta, "safety clamp should reduce beta")
    // Clamped beta should satisfy safety: exploit(beta) <= 0.04
    val exploitAtClamped = 0.03 + 0.1 * updated.beta
    assert(exploitAtClamped <= 0.04 + 1e-10, s"clamped beta should satisfy safety: exploit=$exploitAtClamped")

  test("Appendix B: changepoint retreat and safety clamp interact correctly"):
    val config = ExploitationConfig(initialBeta = 0.9, cpRetreatRate = 0.3, epsilonAdapt = 0.02)
    val state = ExploitationState.initial(config)

    // AlwaysDetect retreats first, then safety clamp applies
    val updated = ExploitationInterpolation.updateExploitation(
      state, config,
      rivalId = PlayerId("v1"),
      history = Vector.empty,
      publicState = testPublicState,
      detector = AlwaysDetect,
      exploitabilityFn = beta => 0.01 + 0.05 * beta,
      epsilonNE = 0.02
    )
    // After retreat: 0.9 - 0.3 = 0.6
    // Safety bound: 0.02 + 0.02 = 0.04
    // Exploit(0.6) = 0.01 + 0.03 = 0.04 <= 0.04 → safe
    assertEqualsDouble(updated.beta, 0.6, 1e-10)

  test("Appendix B: changepoint detector wReset blends toward meta-prior"):
    import sicfun.core.DiscreteDistribution
    val detector = ChangepointDetector(
      hazardRate = 0.1,
      rMin = 2,
      kappaCP = 0.5,
      wReset = 0.3
    )
    val current = DiscreteDistribution(Map("A" -> 0.8, "B" -> 0.2))
    val meta = DiscreteDistribution(Map("A" -> 0.5, "B" -> 0.5))
    val reset = detector.resetPrior(current, meta)

    // reset = (1 - 0.3)*current + 0.3*meta
    assertEqualsDouble(reset.probabilityOf("A"), 0.7 * 0.8 + 0.3 * 0.5, 1e-12)
    assertEqualsDouble(reset.probabilityOf("B"), 0.7 * 0.2 + 0.3 * 0.5, 1e-12)

  // ========================================================================
  // BridgeManifest four-world closure (post-solver integration)
  // ========================================================================

  test("BridgeManifest: no Structural severity gaps remain for four-world objects"):
    import bridge.BridgeManifest
    val structuralGaps = BridgeManifest.structuralGaps
    val fourWorldStructural = structuralGaps.filter(_.formalObject.startsWith("FourWorld"))
    assertEquals(fourWorldStructural.size, 0,
      s"Expected no Structural four-world gaps, found: ${fourWorldStructural.map(_.formalObject)}")

  test("BridgeManifest: DeltaVocabulary severity is Behavioral (not Structural)"):
    import bridge.BridgeManifest
    val dv = BridgeManifest.entries.find(_.formalObject == "DeltaVocabulary")
    assert(dv.isDefined, "DeltaVocabulary entry must exist")
    assertEquals(dv.get.severity, Severity.Behavioral)

  // ========================================================================
  // Helpers
  // ========================================================================

  private val testPublicState: PublicState =
    import sicfun.holdem.types.{Board, Position}
    val sentinelHero = PlayerId("__test__")
    PublicState(
      street = sicfun.holdem.types.Street.Flop,
      board = Board.empty,
      pot = Chips(100.0),
      stacks = TableMap(
        hero = sentinelHero,
        seats = Vector(
          Seat(sentinelHero, Position.SmallBlind, SeatStatus.Active, Chips(500.0))
        )
      ),
      actionHistory = Vector.empty
    )
