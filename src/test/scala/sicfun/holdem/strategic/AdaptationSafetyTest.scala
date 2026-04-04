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
