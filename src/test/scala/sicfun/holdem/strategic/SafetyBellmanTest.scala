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
