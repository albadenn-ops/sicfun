package sicfun.holdem.strategic

class SafetyBellmanTest extends munit.FunSuite:

  private inline val Tol = 1e-10
  private val selfLoop: (Int, Int, Int) => Int = (s, _, _) => s
  private val numProfilesOne = 1

  // ---- Def 58: Baseline loss ------------------------------------------------

  test("baselineLoss: value - Q"):
    assertEqualsDouble(SafetyBellman.baselineLoss(10.0, 8.0), 2.0, Tol)

  test("baselineLoss: zero when Q equals value"):
    assertEqualsDouble(SafetyBellman.baselineLoss(5.0, 5.0), 0.0, Tol)

  test("baselineLoss: negative when Q exceeds value"):
    assertEqualsDouble(SafetyBellman.baselineLoss(3.0, 7.0), -4.0, Tol)

  // ---- Def 59: Robust one-step loss -----------------------------------------

  test("robustOneStepLoss: max over belief points"):
    val vBase = Array(10.0, 8.0, 12.0)
    val qVals = Array(9.0, 3.0, 11.0)
    // losses: 1.0, 5.0, 1.0 → max = 5.0
    assertEqualsDouble(SafetyBellman.robustOneStepLoss(vBase, qVals), 5.0, Tol)

  test("robustOneStepLoss: single belief point"):
    assertEqualsDouble(SafetyBellman.robustOneStepLoss(Array(10.0), Array(7.0)), 3.0, Tol)

  // ---- Def 60: T_safe monotonicity ------------------------------------------

  test("T_safe is monotone: larger bound in → larger bound out"):
    val robustLosses = Array(Array(1.0, 2.0), Array(0.5, 1.5))
    val gamma = 0.9
    val small = SafetyBellman.tSafe(Array(0.0, 0.0), robustLosses, gamma, selfLoop, numProfilesOne)
    val large = SafetyBellman.tSafe(Array(5.0, 5.0), robustLosses, gamma, selfLoop, numProfilesOne)
    // Each element of large should be >= corresponding element of small
    for s <- small.indices do
      assert(large(s) >= small(s) - Tol, s"large($s)=${large(s)} < small($s)=${small(s)}")

  // ---- Def 60: Boundedness under bounded rewards ----------------------------

  test("T_safe bounded under bounded rewards"):
    val rMax = 10.0
    val gamma = 0.9
    val lossMax = rMax // worst-case one-step loss
    val robustLosses = Array(Array(lossMax, lossMax / 2))
    val bound = Array(0.0)
    val result = SafetyBellman.tSafe(bound, robustLosses, gamma, selfLoop, numProfilesOne)
    // T_safe(0) = min_a [L(s,a) + gamma*0] = rMax / 2
    assertEqualsDouble(result(0), rMax / 2, Tol)

  test("tSafe uses min_a with transition-aware futures"):
    // 2 states, 2 actions, 2 profiles
    val transitions: (Int, Int, Int) => Int = (s, a, p) => (s, a, p) match
      case (0, 0, 0) => 0
      case (0, 1, 0) => 1
      case (1, 0, 0) => 1
      case (1, 1, 0) => 0
      case (0, 0, 1) => 1
      case (0, 1, 1) => 0
      case (1, 0, 1) => 0
      case (1, 1, 1) => 1
      case _ => 0
    val numProfiles = 2
    val currentBound = Array(2.0, 5.0)
    val robustLosses = Array(Array(1.0, 3.0), Array(0.5, 2.0))
    val gamma = 0.9

    val result = SafetyBellman.tSafe(currentBound, robustLosses, gamma, transitions, numProfiles)

    // For state 0:
    //   action 0: L(0,0) + gamma * max_p B(T(0,0,p)) = 1.0 + 0.9 * max(B(0), B(1)) = 1.0 + 0.9*5 = 5.5
    //   action 1: L(0,1) + gamma * max_p B(T(0,1,p)) = 3.0 + 0.9 * max(B(1), B(0)) = 3.0 + 0.9*5 = 7.5
    //   min_a = min(5.5, 7.5) = 5.5
    // For state 1:
    //   action 0: L(1,0) + gamma * max_p B(T(1,0,p)) = 0.5 + 0.9 * max(B(1), B(0)) = 0.5 + 0.9*5 = 5.0
    //   action 1: L(1,1) + gamma * max_p B(T(1,1,p)) = 2.0 + 0.9 * max(B(0), B(1)) = 2.0 + 0.9*5 = 6.5
    //   min_a = min(5.0, 6.5) = 5.0
    assertEqualsDouble(result(0), 5.5, Tol)
    assertEqualsDouble(result(1), 5.0, Tol)

  // ---- Def 61: B* convergence -----------------------------------------------

  test("B* converges on 2-state toy MDP"):
    // 2 states, 2 actions each
    // Losses: state 0 -> [1.0, 2.0], state 1 -> [0.5, 1.0]
    val robustLosses = Array(Array(1.0, 2.0), Array(0.5, 1.0))
    val gamma = 0.5
    val bStar = SafetyBellman.computeBStar(robustLosses, gamma, selfLoop, numProfilesOne)

    // B* should be non-negative and finite
    for s <- bStar.indices do
      assert(bStar(s) >= -Tol, s"B*($s) = ${bStar(s)} should be non-negative")
      assert(bStar(s) < 1e10, s"B*($s) = ${bStar(s)} should be finite")

    // With min_a and self-loop:
    // State 0: min(1.0 + 0.5*B(0), 2.0 + 0.5*B(0)) = 1.0 + 0.5*B(0) → B(0) = 2.0
    // State 1: min(0.5 + 0.5*B(1), 1.0 + 0.5*B(1)) = 0.5 + 0.5*B(1) → B(1) = 1.0
    assertEqualsDouble(bStar(0), 2.0, 1e-8)
    assertEqualsDouble(bStar(1), 1.0, 1e-8)

    // B* should satisfy the fixed-point: T_safe(B*) ≈ B*
    val tResult = SafetyBellman.tSafe(bStar, robustLosses, gamma, selfLoop, numProfilesOne)
    for s <- bStar.indices do
      assertEqualsDouble(tResult(s), bStar(s), 1e-8)

  test("B* is zero when all losses are zero"):
    val robustLosses = Array(Array(0.0, 0.0), Array(0.0, 0.0))
    val bStar = SafetyBellman.computeBStar(robustLosses, 0.9, selfLoop, numProfilesOne)
    for s <- bStar.indices do
      assertEqualsDouble(bStar(s), 0.0, Tol)

  test("B* converges on single-state single-action"):
    // L = 3.0, gamma = 0.5
    // B* = L / (1 - gamma) = 3.0 / 0.5 = 6.0
    val robustLosses = Array(Array(3.0))
    val bStar = SafetyBellman.computeBStar(robustLosses, 0.5, selfLoop, numProfilesOne)
    assertEqualsDouble(bStar(0), 6.0, 1e-6)

  // ---- Def 62: Safe action sets ---------------------------------------------

  test("safe action set excludes unsafe actions"):
    val bound = Array(5.0, 5.0)
    // losses: state 0 -> [1.0, 10.0] (action 1 is unsafe)
    val robustLosses = Array(Array(1.0, 10.0), Array(0.5, 0.5))
    val gamma = 0.5
    val safe = SafetyBellman.safeActionSet(0, bound, robustLosses, gamma, selfLoop, numProfilesOne)
    assert(safe.contains(0), "action 0 should be safe")
    assert(!safe.contains(1), "action 1 should be unsafe")

  test("safe action set includes all actions when all are safe"):
    val bound = Array(100.0)
    val robustLosses = Array(Array(1.0, 2.0, 3.0))
    val gamma = 0.5
    val safe = SafetyBellman.safeActionSet(0, bound, robustLosses, gamma, selfLoop, numProfilesOne)
    assertEquals(safe.size, 3)

  // ---- Def 63: Safe-feasible policy selector --------------------------------

  test("safeFeasibleAction picks highest Q among safe actions"):
    val qValues = Array(3.0, 7.0, 5.0)
    val safeActions = IndexedSeq(0, 2) // action 1 is unsafe
    val best = SafetyBellman.safeFeasibleAction(qValues, safeActions)
    assertEquals(best, 2) // Q=5.0 is highest among safe

  test("safeFeasibleAction falls back to best overall when safe set empty"):
    val qValues = Array(3.0, 7.0, 5.0)
    val safeActions = IndexedSeq.empty[Int]
    val best = SafetyBellman.safeFeasibleAction(qValues, safeActions)
    assertEquals(best, 1) // Q=7.0 is highest overall

  // ---- Def 64: Required adaptation budget -----------------------------------

  test("requiredAdaptationBudget is max of B*"):
    val bStar = Array(1.0, 3.0, 2.0)
    assertEqualsDouble(SafetyBellman.requiredAdaptationBudget(bStar), 3.0, Tol)

  test("requiredAdaptationBudget is zero when B* is all zeros"):
    assertEqualsDouble(SafetyBellman.requiredAdaptationBudget(Array(0.0, 0.0)), 0.0, Tol)

  test("requiredAdaptationBudget is zero for empty array"):
    assertEqualsDouble(SafetyBellman.requiredAdaptationBudget(Array.empty), 0.0, Tol)

  // ---- Def 65: Certificate structural constraints ---------------------------

  test("Certificate satisfies all structural constraints"):
    val cert = SafetyBellman.Certificate(
      values = Array(3.0, 5.0, 0.0), // state 2 is terminal
      terminalStates = Set(2)
    )
    assert(cert.satisfiesTerminality)
    assert(cert.satisfiesNonNegativity)
    assert(cert.satisfiesGlobalBound(10.0))

  test("Certificate violates terminality when terminal state has non-zero value"):
    val cert = SafetyBellman.Certificate(
      values = Array(3.0, 5.0, 1.0), // state 2 is terminal but non-zero
      terminalStates = Set(2)
    )
    assert(!cert.satisfiesTerminality)

  test("Certificate violates non-negativity"):
    val cert = SafetyBellman.Certificate(
      values = Array(-1.0, 5.0, 0.0),
      terminalStates = Set(2)
    )
    assert(!cert.satisfiesNonNegativity)

  test("Certificate violates global bound"):
    val cert = SafetyBellman.Certificate(
      values = Array(3.0, 15.0, 0.0),
      terminalStates = Set(2)
    )
    assert(!cert.satisfiesGlobalBound(10.0))

  test("Certificate satisfies monotonicity when it is exact fixed point"):
    // B* is an exact fixed point: T_safe(B*) = B*, so B* dominates T_safe(B*)
    // Use tight tolerance to ensure residual gap < 1e-12 monotonicity check
    val robustLosses = Array(Array(1.0), Array(0.5))
    val gamma = 0.5
    val bStar = SafetyBellman.computeBStar(robustLosses, gamma, selfLoop, numProfilesOne, tolerance = 1e-14)
    val cert = SafetyBellman.Certificate(
      values = bStar.clone(),
      terminalStates = Set.empty
    )
    assert(cert.satisfiesMonotonicity(robustLosses, gamma, selfLoop, numProfilesOne))

  // ---- Def 65: Full certificate validation ----------------------------------

  test("Certificate.isValid passes for well-formed certificate"):
    // B* is a valid certificate: non-negative, bounded, and T_safe(B*) = B*
    val robustLosses = Array(Array(1.0), Array(0.5))
    val gamma = 0.5
    val bStar = SafetyBellman.computeBStar(robustLosses, gamma, selfLoop, numProfilesOne, tolerance = 1e-14)
    val cert = SafetyBellman.Certificate(
      values = bStar.clone(),
      terminalStates = Set.empty
    )
    assert(cert.isValid(robustLosses, gamma, maxBound = 100.0, selfLoop, numProfilesOne))

  // ---- Def 66: Certificate dominance ----------------------------------------

  test("certificate dominates B* when values >= B*"):
    val bStar = Array(2.0, 3.0, 1.0)
    val cert = Array(3.0, 4.0, 1.5)
    assert(SafetyBellman.certificateDominates(cert, bStar))

  test("certificate does not dominate B* when any value < B*"):
    val bStar = Array(2.0, 3.0, 1.0)
    val cert = Array(3.0, 2.5, 1.5)
    assert(!SafetyBellman.certificateDominates(cert, bStar))

  test("B* dominates itself"):
    val bStar = Array(2.0, 3.0, 1.0)
    assert(SafetyBellman.certificateDominates(bStar, bStar))

  // ---- Corollary 9.3: Total vulnerability -----------------------------------

  test("TotalVulnerability = epsilon_base + epsilon_adapt"):
    val baseline = DeploymentBaseline(Ev(0.03), 50, "test baseline")
    val epsilonAdapt = Ev(0.05)
    val (total, fidelity) = TotalVulnerability.compute(baseline, epsilonAdapt)
    assertEqualsDouble(total.value, 0.08, 1e-12)
    assertEquals(fidelity, Fidelity.Approximate)

  test("TotalVulnerability with zero adaptation budget"):
    val baseline = DeploymentBaseline(Ev(0.02), 10, "zero adapt")
    val (total, _) = TotalVulnerability.compute(baseline, Ev.Zero)
    assertEqualsDouble(total.value, 0.02, 1e-12)

  // ---- Def 60: terminalStates parameter ------------------------------------

  test("tSafe: terminal state gets value 0.0 regardless of losses"):
    // State 0 is terminal, state 1 is not
    val robustLosses = Array(Array(5.0, 10.0), Array(1.0, 2.0))
    val currentBound = Array(3.0, 3.0)
    val gamma = 0.9
    val result = SafetyBellman.tSafe(
      currentBound, robustLosses, gamma, selfLoop, numProfilesOne,
      terminalStates = Set(0)
    )
    assertEqualsDouble(result(0), 0.0, Tol) // terminal => forced to 0
    assert(result(1) > 0.0, "non-terminal state should have positive bound")

  test("tSafe: non-terminal states unaffected by terminal set"):
    // Same setup without terminal states
    val robustLosses = Array(Array(5.0, 10.0), Array(1.0, 2.0))
    val currentBound = Array(3.0, 3.0)
    val gamma = 0.9
    val withoutTerminal = SafetyBellman.tSafe(
      currentBound, robustLosses, gamma, selfLoop, numProfilesOne,
      terminalStates = Set.empty
    )
    val withTerminal = SafetyBellman.tSafe(
      currentBound, robustLosses, gamma, selfLoop, numProfilesOne,
      terminalStates = Set(0)
    )
    // State 1 should be the same whether or not state 0 is terminal
    // (since selfLoop means state 1 always transitions to itself)
    assertEqualsDouble(withTerminal(1), withoutTerminal(1), Tol)

  test("tSafe: all states terminal yields all zeros"):
    val robustLosses = Array(Array(5.0), Array(10.0), Array(3.0))
    val currentBound = Array(7.0, 8.0, 9.0)
    val gamma = 0.5
    val result = SafetyBellman.tSafe(
      currentBound, robustLosses, gamma, selfLoop, numProfilesOne,
      terminalStates = Set(0, 1, 2)
    )
    for s <- result.indices do
      assertEqualsDouble(result(s), 0.0, Tol)

  test("tSafe: terminal states remain zero across multiple iterations"):
    val robustLosses = Array(Array(2.0, 3.0), Array(1.0, 4.0), Array(0.5, 1.5))
    val gamma = 0.5
    // State 2 is terminal
    var bound = Array(0.0, 0.0, 0.0)
    for _ <- 0 until 10 do
      bound = SafetyBellman.tSafe(
        bound, robustLosses, gamma, selfLoop, numProfilesOne,
        terminalStates = Set(2)
      )
    assertEqualsDouble(bound(2), 0.0, Tol) // remains zero after all iterations

  test("computeBStar: terminal states converge to zero"):
    val robustLosses = Array(Array(3.0), Array(2.0), Array(1.0))
    val gamma = 0.5
    val bStar = SafetyBellman.computeBStar(
      robustLosses, gamma, selfLoop, numProfilesOne,
      terminalStates = Set(1)
    )
    assertEqualsDouble(bStar(1), 0.0, Tol) // terminal state
    // Non-terminal states should still converge normally
    // State 0: B(0) = 3.0 + 0.5*B(0) => B(0) = 6.0
    assertEqualsDouble(bStar(0), 6.0, 1e-6)
    // State 2: B(2) = 1.0 + 0.5*B(2) => B(2) = 2.0
    assertEqualsDouble(bStar(2), 2.0, 1e-6)

  test("computeBStar: terminal state with transition-dependent futures"):
    // 2 states, 1 action, transitions: action at state 0 goes to state 1
    val transitions: (Int, Int, Int) => Int = (s, _, _) => 1 - s // state 0->1, 1->0
    val robustLosses = Array(Array(2.0), Array(5.0))
    val gamma = 0.5
    // State 1 is terminal => B(1) = 0
    // State 0: B(0) = L(0,0) + gamma * B(T(0,0,0)) = 2.0 + 0.5 * B(1) = 2.0 + 0 = 2.0
    val bStar = SafetyBellman.computeBStar(
      robustLosses, gamma, transitions, numProfilesOne,
      terminalStates = Set(1)
    )
    assertEqualsDouble(bStar(1), 0.0, Tol)
    assertEqualsDouble(bStar(0), 2.0, 1e-6)

  // ---- Proposition 9.5: T_safe is a gamma-contraction -----------------------

  test("Proposition 9.5: T_safe is a gamma-contraction (||T(B1)-T(B2)||_inf <= gamma*||B1-B2||_inf)"):
    val robustLosses = Array(Array(1.0, 3.0), Array(2.0, 0.5), Array(1.5, 2.5))
    val gamma = 0.7
    val b1 = Array(4.0, 2.0, 6.0)
    val b2 = Array(1.0, 5.0, 3.0)
    val tb1 = SafetyBellman.tSafe(b1, robustLosses, gamma, selfLoop, numProfilesOne)
    val tb2 = SafetyBellman.tSafe(b2, robustLosses, gamma, selfLoop, numProfilesOne)

    val inputDiff = (0 until b1.length).map(s => math.abs(b1(s) - b2(s))).max
    val outputDiff = (0 until tb1.length).map(s => math.abs(tb1(s) - tb2(s))).max

    assert(outputDiff <= gamma * inputDiff + Tol,
      s"contraction violated: ||T(B1)-T(B2)||=$outputDiff > gamma*||B1-B2||=${gamma * inputDiff}")

  test("Proposition 9.5: contraction holds with non-trivial transitions and multiple profiles"):
    val transitions: (Int, Int, Int) => Int = (s, a, p) => (s, a, p) match
      case (0, 0, 0) => 1
      case (0, 1, 0) => 0
      case (1, 0, 0) => 0
      case (1, 1, 0) => 1
      case (0, 0, 1) => 0
      case (0, 1, 1) => 1
      case (1, 0, 1) => 1
      case (1, 1, 1) => 0
      case _ => 0
    val robustLosses = Array(Array(2.0, 1.0), Array(0.5, 3.0))
    val gamma = 0.8
    val b1 = Array(10.0, 2.0)
    val b2 = Array(3.0, 8.0)
    val tb1 = SafetyBellman.tSafe(b1, robustLosses, gamma, transitions, 2)
    val tb2 = SafetyBellman.tSafe(b2, robustLosses, gamma, transitions, 2)

    val inputDiff = (0 until b1.length).map(s => math.abs(b1(s) - b2(s))).max
    val outputDiff = (0 until tb1.length).map(s => math.abs(tb1(s) - tb2(s))).max

    assert(outputDiff <= gamma * inputDiff + Tol,
      s"contraction violated: output=$outputDiff > gamma*input=${gamma * inputDiff}")

  test("Proposition 9.5: B* uniqueness — different initializations converge to same fixed point"):
    val robustLosses = Array(Array(1.0, 2.0), Array(0.5, 1.5))
    val gamma = 0.5
    // Starting from zero
    val bStar1 = SafetyBellman.computeBStar(robustLosses, gamma, selfLoop, numProfilesOne)
    // Starting from a large bound via extra iterations from high initial
    var bound = Array(100.0, 100.0)
    for _ <- 0 until 500 do
      bound = SafetyBellman.tSafe(bound, robustLosses, gamma, selfLoop, numProfilesOne)
    // Both should converge to the same B*
    for s <- bStar1.indices do
      assertEqualsDouble(bound(s), bStar1(s), 1e-6)

  // ---- Proposition 9.2: AS-strong implies exploitability bound ---------------

  test("Proposition 9.2: safe actions produce regret bounded by B* at each state"):
    // 3 states, 2 actions, self-loop transitions
    val robustLosses = Array(Array(1.0, 3.0), Array(0.5, 2.0), Array(2.0, 0.8))
    val gamma = 0.5
    val bStar = SafetyBellman.computeBStar(robustLosses, gamma, selfLoop, numProfilesOne, tolerance = 1e-14)

    // For each state, every safe action should satisfy:
    //   L_robust(s, a) + gamma * max_sigma B*(T_sigma(s,a)) <= B*(s)
    for s <- bStar.indices do
      val safe = SafetyBellman.safeActionSet(s, bStar, robustLosses, gamma, selfLoop, numProfilesOne)
      assert(safe.nonEmpty, s"safe action set empty at state $s")
      for a <- safe do
        val cost = robustLosses(s)(a) + gamma * bStar(s) // self-loop: successor = s
        assert(cost <= bStar(s) + 1e-10,
          s"safe action ($s,$a) violates budget: cost=$cost > B*($s)=${bStar(s)}")

  test("Proposition 9.2: total vulnerability bound holds (Corollary 9.3 numeric)"):
    // If Exploit(pi_bar) <= epsilon_base and pi satisfies AS with tolerance epsilon_adapt,
    // then Exploit(pi) <= epsilon_base + epsilon_adapt.
    // Test via: epsilon*_adapt = max(B*) and check it gives a finite total vulnerability.
    val robustLosses = Array(Array(0.1, 0.3), Array(0.05, 0.2))
    val gamma = 0.9
    val bStar = SafetyBellman.computeBStar(robustLosses, gamma, selfLoop, numProfilesOne)
    val epsilonAdaptRequired = SafetyBellman.requiredAdaptationBudget(bStar)
    val epsilonBase = 0.05

    // Total vulnerability: epsilon_base + epsilon*_adapt
    val totalVulnerability = epsilonBase + epsilonAdaptRequired
    // Must be finite and non-negative
    assert(totalVulnerability >= 0.0, "total vulnerability must be non-negative")
    assert(totalVulnerability < Double.PositiveInfinity, "total vulnerability must be finite")
    // epsilon*_adapt = max(B*) which for self-loop is min_a L(s,a) / (1-gamma)
    // State 0: min(0.1, 0.3) / 0.1 = 1.0; State 1: min(0.05, 0.2) / 0.1 = 0.5
    // So max(B*) = 1.0
    assertEqualsDouble(epsilonAdaptRequired, 1.0, 1e-6)
