package sicfun.holdem.strategic

class AdaptationSafetyTest extends munit.FunSuite:

  private inline val Tol = 1e-12

  // -- Def 52: Adaptation safety --

  test("isSafe returns true when exploitability <= epsilon + delta"):
    assert(AdaptationSafety.isSafe(
      exploitability = 0.05,
      epsilonNE = 0.03,
      epsilonAdapt = 0.03
    ))

  test("isSafe returns true at exact boundary"):
    assert(AdaptationSafety.isSafe(
      exploitability = 0.06,
      epsilonNE = 0.03,
      epsilonAdapt = 0.03
    ))

  test("isSafe returns false when exploitability exceeds bound"):
    assert(!AdaptationSafety.isSafe(
      exploitability = 0.10,
      epsilonNE = 0.03,
      epsilonAdapt = 0.03
    ))

  // -- Theorem 8: betaBar computation --

  test("betaBar is well-defined and in [0,1]"):
    val beta = AdaptationSafety.betaBar(
      epsilonAdapt = 0.05,
      epsilonNE = 0.03,
      exploitabilityAtBeta = beta => 0.03 + 0.04 * beta
    )
    assert(beta >= 0.0)
    assert(beta <= 1.0)

  test("betaBar returns 0 when any exploitation violates safety"):
    val beta = AdaptationSafety.betaBar(
      epsilonAdapt = 0.0,
      epsilonNE = 0.0,
      exploitabilityAtBeta = _ => 0.01  // always exceeds eps + delta = 0
    )
    assertEqualsDouble(beta, 0.0, Tol)

  test("betaBar returns 1.0 when all betas are safe"):
    val beta = AdaptationSafety.betaBar(
      epsilonAdapt = 1.0,
      epsilonNE = 0.0,
      exploitabilityAtBeta = _ => 0.5
    )
    assertEqualsDouble(beta, 1.0, Tol)

  // -- Def 53: Affine equilibrium deterrence --

  test("deterrence holds when opponent exploit >= betaDet * gain"):
    assert(AdaptationSafety.affineDeterrenceHolds(
      opponentExploitability = 0.10,
      opponentGain = 0.50,
      betaDet = 0.15
    ))

  test("deterrence fails when opponent exploit < betaDet * gain"):
    assert(!AdaptationSafety.affineDeterrenceHolds(
      opponentExploitability = 0.05,
      opponentGain = 0.50,
      betaDet = 0.15
    ))

  test("deterrence holds trivially when opponent gain <= 0"):
    assert(AdaptationSafety.affineDeterrenceHolds(
      opponentExploitability = 0.0,
      opponentGain = -1.0,
      betaDet = 0.5
    ))

  // -- SafetyConfig --

  test("clampBeta reduces proposed beta to betaBar"):
    val clamped = AdaptationSafety.clampBeta(
      proposedBeta = 0.8,
      betaBar = 0.6
    )
    assertEqualsDouble(clamped, 0.6, Tol)

  test("clampBeta passes beta through when below betaBar"):
    val clamped = AdaptationSafety.clampBeta(
      proposedBeta = 0.3,
      betaBar = 0.6
    )
    assertEqualsDouble(clamped, 0.3, Tol)

  // -- Def 57: AS-strong predicate (exact per-profile form) --

  test("isStrongSafe: holds when J(pi,sigma) >= J(bar_pi,sigma) - eps for all (b, sigma)"):
    // 2 beliefs, 2 profiles each
    val jPolicy   = IndexedSeq(IndexedSeq(Ev(5.0), Ev(4.8)), IndexedSeq(Ev(3.0), Ev(3.1)))
    val jBaseline = IndexedSeq(IndexedSeq(Ev(5.5), Ev(5.0)), IndexedSeq(Ev(3.2), Ev(3.3)))
    assert(AdaptationSafety.isStrongSafe(jPolicy, jBaseline, Ev(0.5)))

  test("isStrongSafe: fails when any single (belief, profile) pair violates"):
    // belief 0 profile 0: J_pol=5.0 >= J_base=5.1 - 0.05 = 5.05? No (5.0 < 5.05)
    val jPolicy   = IndexedSeq(IndexedSeq(Ev(5.0), Ev(4.9)), IndexedSeq(Ev(3.0), Ev(3.0)))
    val jBaseline = IndexedSeq(IndexedSeq(Ev(5.1), Ev(4.9)), IndexedSeq(Ev(3.0), Ev(3.0)))
    assert(!AdaptationSafety.isStrongSafe(jPolicy, jBaseline, Ev(0.05)))

  test("isStrongSafe: security-value check passes but per-profile fails (demonstrates unsoundness of approx)"):
    // Profile 0: J(pi)=3, J(bar)=6 → gap=3
    // Profile 1: J(pi)=10, J(bar)=1 → gap=-9
    // Security values: V^sec(pi)=min(3,10)=3, V^sec(bar)=min(6,1)=1
    // Approx: robustRegretApprox = 1 - 3 = -2 ≤ 1.0 → safe (wrong!)
    // Exact: sup_sigma [J(bar,sigma) - J(pi,sigma)] = max(6-3, 1-10) = 3 > 1.0 → unsafe (correct!)
    val jPolicy   = IndexedSeq(IndexedSeq(Ev(3.0), Ev(10.0)))
    val jBaseline = IndexedSeq(IndexedSeq(Ev(6.0), Ev(1.0)))
    assert(!AdaptationSafety.isStrongSafe(jPolicy, jBaseline, Ev(1.0)))
    // But the approx version incorrectly says safe:
    val secPol  = IndexedSeq(Ev(3.0))  // min over profiles
    val secBase = IndexedSeq(Ev(1.0))  // min over profiles
    assert(AdaptationSafety.isStrongSafeApprox(secPol, secBase, Ev(1.0)))

  // -- Def 57: AS-strong approx (backward compat) --

  test("isStrongSafeApprox: holds when V^sec(pi) >= V^sec(bar) - eps"):
    val policy   = IndexedSeq(Ev(5.0), Ev(3.0), Ev(7.0))
    val baseline = IndexedSeq(Ev(5.5), Ev(3.2), Ev(7.1))
    assert(AdaptationSafety.isStrongSafeApprox(policy, baseline, Ev(0.5)))

  test("isStrongSafeApprox: policy strictly better is safe"):
    val policy   = IndexedSeq(Ev(10.0), Ev(8.0))
    val baseline = IndexedSeq(Ev(5.0), Ev(4.0))
    assert(AdaptationSafety.isStrongSafeApprox(policy, baseline, Ev(0.0)))

  // -- Def 57A: Robust regret (exact per-profile form) --

  test("robustRegret: sup over profiles of (J_base - J_pol)"):
    // profiles: gaps = 10-8=2, 5-9=-4, 7-6=1 → sup = 2
    val jBase = IndexedSeq(Ev(10.0), Ev(5.0), Ev(7.0))
    val jPol  = IndexedSeq(Ev(8.0), Ev(9.0), Ev(6.0))
    val rr = AdaptationSafety.robustRegret(jBase, jPol)
    assertEqualsDouble(rr.value, 2.0, Tol)

  test("robustRegret: all profiles identical yields zero"):
    val jBase = IndexedSeq(Ev(5.0), Ev(5.0))
    val jPol  = IndexedSeq(Ev(5.0), Ev(5.0))
    val rr = AdaptationSafety.robustRegret(jBase, jPol)
    assertEqualsDouble(rr.value, 0.0, Tol)

  test("robustRegret: negative only when ALL profiles favor policy"):
    val jBase = IndexedSeq(Ev(3.0), Ev(2.0))
    val jPol  = IndexedSeq(Ev(7.0), Ev(8.0))
    val rr = AdaptationSafety.robustRegret(jBase, jPol)
    assertEqualsDouble(rr.value, -4.0, Tol) // max(3-7, 2-8) = max(-4, -6) = -4

  // -- Def 57A: Robust regret approx (backward compat) --

  test("robustRegretApprox: positive when baseline exceeds policy"):
    val rr = AdaptationSafety.robustRegretApprox(Ev(10.0), Ev(8.0))
    assertEqualsDouble(rr.value, 2.0, Tol)

  test("robustRegretApprox: zero when equal"):
    val rr = AdaptationSafety.robustRegretApprox(Ev(5.0), Ev(5.0))
    assertEqualsDouble(rr.value, 0.0, Tol)

  // -- Def 57A: maxRobustRegret (exact per-profile form) --

  test("maxRobustRegret: picks worst (belief, profile) combination"):
    // belief 0: profiles gaps = 9-8=1, 7-9=-2 → sup = 1
    // belief 1: profiles gaps = 5-2=3, 4-6=-2 → sup = 3
    // belief 2: profiles gaps = 6.5-6=0.5, 7-7=0 → sup = 0.5
    // max over beliefs = 3
    val jPol  = IndexedSeq(IndexedSeq(Ev(8.0), Ev(9.0)), IndexedSeq(Ev(2.0), Ev(6.0)), IndexedSeq(Ev(6.0), Ev(7.0)))
    val jBase = IndexedSeq(IndexedSeq(Ev(9.0), Ev(7.0)), IndexedSeq(Ev(5.0), Ev(4.0)), IndexedSeq(Ev(6.5), Ev(7.0)))
    val mrr = AdaptationSafety.maxRobustRegret(jPol, jBase)
    assertEqualsDouble(mrr.value, 3.0, Tol)

  test("maxRobustRegretApprox: picks worst belief point"):
    val policy   = IndexedSeq(Ev(8.0), Ev(2.0), Ev(6.0))
    val baseline = IndexedSeq(Ev(9.0), Ev(5.0), Ev(6.5))
    val mrr = AdaptationSafety.maxRobustRegretApprox(policy, baseline)
    assertEqualsDouble(mrr.value, 3.0, Tol)

  // -- Def 57B: Adaptation-safe policy class (exact) --

  test("inAdaptationSafeClass: true when max per-profile robust regret <= epsilon"):
    // belief 0: profiles gaps = 9.5-9=0.5, 8-8.5=-0.5 → sup = 0.5
    // belief 1: profiles gaps = 5-4.5=0.5, 4.8-4.6=0.2 → sup = 0.5
    // max = 0.5 <= 0.5
    val jPol  = IndexedSeq(IndexedSeq(Ev(9.0), Ev(8.5)), IndexedSeq(Ev(4.5), Ev(4.6)))
    val jBase = IndexedSeq(IndexedSeq(Ev(9.5), Ev(8.0)), IndexedSeq(Ev(5.0), Ev(4.8)))
    assert(AdaptationSafety.inAdaptationSafeClass(jPol, jBase, Ev(0.5)))

  test("inAdaptationSafeClass: false when any profile at any belief exceeds epsilon"):
    // belief 1, profile 0: gap = 5-2 = 3 > 0.5
    val jPol  = IndexedSeq(IndexedSeq(Ev(9.0), Ev(9.0)), IndexedSeq(Ev(2.0), Ev(6.0)))
    val jBase = IndexedSeq(IndexedSeq(Ev(9.5), Ev(9.0)), IndexedSeq(Ev(5.0), Ev(4.0)))
    assert(!AdaptationSafety.inAdaptationSafeClass(jPol, jBase, Ev(0.5)))

  test("inAdaptationSafeClassApprox: true when securityValue regret <= epsilon"):
    val policy   = IndexedSeq(Ev(9.0), Ev(4.5))
    val baseline = IndexedSeq(Ev(9.5), Ev(5.0))
    assert(AdaptationSafety.inAdaptationSafeClassApprox(policy, baseline, Ev(0.5)))

  // -- SafetyConfig v0.31.1 alias --

  test("SafetyConfig.deltaAdapt aliases epsilonAdapt (backward compat)"):
    val cfg = SafetyConfig(epsilonNE = 0.02, epsilonAdapt = 0.05, betaDet = 0.1)
    assertEqualsDouble(cfg.deltaAdapt, 0.05, Tol) // alias returns same value
    assertEqualsDouble(cfg.epsilonAdapt, 0.05, Tol)
