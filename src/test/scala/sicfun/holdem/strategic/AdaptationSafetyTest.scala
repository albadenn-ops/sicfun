package sicfun.holdem.strategic

class AdaptationSafetyTest extends munit.FunSuite:

  private inline val Tol = 1e-12

  // -- Def 52: Adaptation safety --

  test("isSafe returns true when exploitability <= epsilon + delta"):
    assert(AdaptationSafety.isSafe(
      exploitability = 0.05,
      epsilonNE = 0.03,
      deltaAdapt = 0.03
    ))

  test("isSafe returns true at exact boundary"):
    assert(AdaptationSafety.isSafe(
      exploitability = 0.06,
      epsilonNE = 0.03,
      deltaAdapt = 0.03
    ))

  test("isSafe returns false when exploitability exceeds bound"):
    assert(!AdaptationSafety.isSafe(
      exploitability = 0.10,
      epsilonNE = 0.03,
      deltaAdapt = 0.03
    ))

  // -- Theorem 8: betaBar computation --

  test("betaBar is well-defined and in [0,1]"):
    val beta = AdaptationSafety.betaBar(
      deltaAdapt = 0.05,
      epsilonNE = 0.03,
      exploitabilityAtBeta = beta => 0.03 + 0.04 * beta
    )
    assert(beta >= 0.0)
    assert(beta <= 1.0)

  test("betaBar returns 0 when any exploitation violates safety"):
    val beta = AdaptationSafety.betaBar(
      deltaAdapt = 0.0,
      epsilonNE = 0.0,
      exploitabilityAtBeta = _ => 0.01  // always exceeds eps + delta = 0
    )
    assertEqualsDouble(beta, 0.0, Tol)

  test("betaBar returns 1.0 when all betas are safe"):
    val beta = AdaptationSafety.betaBar(
      deltaAdapt = 1.0,
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

  // -- Def 57: AS-strong predicate --

  test("isStrongSafe: holds when policy dominates baseline minus epsilon"):
    val policy   = IndexedSeq(Ev(5.0), Ev(3.0), Ev(7.0))
    val baseline = IndexedSeq(Ev(5.5), Ev(3.2), Ev(7.1))
    assert(AdaptationSafety.isStrongSafe(policy, baseline, Ev(0.5)))

  test("isStrongSafe: holds at exact boundary"):
    val policy   = IndexedSeq(Ev(5.0), Ev(3.0))
    val baseline = IndexedSeq(Ev(5.1), Ev(3.1))
    assert(AdaptationSafety.isStrongSafe(policy, baseline, Ev(0.1)))

  test("isStrongSafe: fails when gap exceeds epsilon at any belief"):
    val policy   = IndexedSeq(Ev(5.0), Ev(1.0))
    val baseline = IndexedSeq(Ev(5.1), Ev(3.0))
    // gap at belief 1: 3.0 - 1.0 = 2.0 > 0.5
    assert(!AdaptationSafety.isStrongSafe(policy, baseline, Ev(0.5)))

  test("isStrongSafe: policy strictly better than baseline is safe"):
    val policy   = IndexedSeq(Ev(10.0), Ev(8.0))
    val baseline = IndexedSeq(Ev(5.0), Ev(4.0))
    assert(AdaptationSafety.isStrongSafe(policy, baseline, Ev(0.0)))

  // -- Def 57A: Robust regret --

  test("robustRegret: positive when baseline exceeds policy"):
    val rr = AdaptationSafety.robustRegret(Ev(10.0), Ev(8.0))
    assertEqualsDouble(rr.value, 2.0, Tol)

  test("robustRegret: zero when equal"):
    val rr = AdaptationSafety.robustRegret(Ev(5.0), Ev(5.0))
    assertEqualsDouble(rr.value, 0.0, Tol)

  test("robustRegret: negative when policy exceeds baseline"):
    val rr = AdaptationSafety.robustRegret(Ev(3.0), Ev(7.0))
    assertEqualsDouble(rr.value, -4.0, Tol)

  test("maxRobustRegret: picks worst belief point"):
    val policy   = IndexedSeq(Ev(8.0), Ev(2.0), Ev(6.0))
    val baseline = IndexedSeq(Ev(9.0), Ev(5.0), Ev(6.5))
    // regrets: 1.0, 3.0, 0.5 → max = 3.0
    val mrr = AdaptationSafety.maxRobustRegret(policy, baseline)
    assertEqualsDouble(mrr.value, 3.0, Tol)

  // -- Def 57B: Adaptation-safe policy class --

  test("inAdaptationSafeClass: true when maxRobustRegret <= epsilon"):
    val policy   = IndexedSeq(Ev(9.0), Ev(4.5))
    val baseline = IndexedSeq(Ev(9.5), Ev(5.0))
    // regrets: 0.5, 0.5 → max = 0.5 <= 0.5
    assert(AdaptationSafety.inAdaptationSafeClass(policy, baseline, Ev(0.5)))

  test("inAdaptationSafeClass: false when maxRobustRegret > epsilon"):
    val policy   = IndexedSeq(Ev(9.0), Ev(2.0))
    val baseline = IndexedSeq(Ev(9.5), Ev(5.0))
    // regrets: 0.5, 3.0 → max = 3.0 > 0.5
    assert(!AdaptationSafety.inAdaptationSafeClass(policy, baseline, Ev(0.5)))

  // -- SafetyConfig v0.31.1 alias --

  test("SafetyConfig.epsilonAdapt aliases deltaAdapt"):
    val cfg = SafetyConfig(epsilonNE = 0.02, deltaAdapt = 0.05, betaDet = 0.1)
    assertEqualsDouble(cfg.epsilonAdapt, 0.05, Tol)
