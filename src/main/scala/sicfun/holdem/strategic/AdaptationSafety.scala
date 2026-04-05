package sicfun.holdem.strategic

/** Adaptation safety framework (Defs 52-53, v0.30.2); to be upgraded to
  * Defs 57/57A-C in Wave 4 of the v0.31.1 formal closure plan.
  *
  * SYMBOL-MAPPING NOTE (Wave 0 — spec hygiene):
  *   delta_adapt    -> epsilon_adapt  (v0.31.1 §9A' alias; same quantity)
  *   delta_retreat  -> delta_cp_retreat  (v0.31.1 §9B alias; same quantity)
  *   omega must always be qualified as chain-omega or grid-omega.
  *
  * Def 52 (v0.30.2): SICFUN's exploitation satisfies adaptation safety iff
  * for every opponent strategy sigma:
  *   Exploit(pi^S_{beta}) <= epsilon_NE + delta_adapt.
  *
  * Def 53 (v0.30.2): Affine equilibrium deterrence -- any opponent attempting
  * to exploit SICFUN must themselves become exploitable:
  *   Exploit_opp(sigma^exploit) >= beta_det * Gain_opp(sigma^exploit).
  *
  * Theorem 8: clamping beta <= betaBar(delta_adapt) enforces A10.
  *
  * COMPATIBILITY POLICY (Waves 1-6): legacy constructors and accessors are
  * preserved under their current names. Old names will be marked @deprecated
  * once Wave 4 replacements (Defs 57/57A-C) exist. Removal is Wave 7.
  */
object AdaptationSafety:

  /** Def 52: Check whether adaptation safety holds.
    * Safe iff Exploit(pi) <= epsilon_NE + delta_adapt.
    */
  def isSafe(exploitability: Double, epsilonNE: Double, deltaAdapt: Double): Boolean =
    exploitability <= epsilonNE + deltaAdapt

  /** Theorem 8: Compute betaBar -- the supremum of safe exploitation interpolation.
    *
    * betaBar(delta_adapt) = sup { beta' in [0,1] : Exploit(pi^S_{beta'}) <= eps + delta }
    *
    * Uses binary search over [0,1] since exploitability is upper-semicontinuous
    * in beta. The exploitabilityAtBeta function maps beta -> Exploit(pi^S_beta).
    *
    * @param deltaAdapt            adaptation safety budget
    * @param epsilonNE             baseline exploitability of the epsilon-NE strategy
    * @param exploitabilityAtBeta  oracle: beta -> exploitability at that beta
    * @param iterations            binary search iterations (default 50)
    * @return betaBar in [0,1]
    */
  def betaBar(
      deltaAdapt: Double,
      epsilonNE: Double,
      exploitabilityAtBeta: Double => Double,
      iterations: Int = 50
  ): Double =
    val bound = epsilonNE + deltaAdapt
    // Check beta=0 first (must be safe per Theorem 8 proof)
    if exploitabilityAtBeta(0.0) > bound then return 0.0
    // Check beta=1
    if exploitabilityAtBeta(1.0) <= bound then return 1.0
    // Binary search
    var lo = 0.0
    var hi = 1.0
    var i = 0
    while i < iterations do
      val mid = (lo + hi) / 2.0
      if exploitabilityAtBeta(mid) <= bound then lo = mid
      else hi = mid
      i += 1
    lo

  /** Clamp proposed beta to betaBar. Enforces A10. */
  def clampBeta(proposedBeta: Double, betaBar: Double): Double =
    math.min(proposedBeta, betaBar)

  /** Def 53: Affine equilibrium deterrence predicate.
    * Holds iff Exploit_opp(sigma^exploit) >= beta_det * Gain_opp(sigma^exploit).
    * Trivially holds when opponent gain <= 0.
    */
  def affineDeterrenceHolds(
      opponentExploitability: Double,
      opponentGain: Double,
      betaDet: Double
  ): Boolean =
    if opponentGain <= 0.0 then true
    else opponentExploitability >= betaDet * opponentGain

/** Configuration for adaptation safety (bundles Def 52 + 53 parameters). */
final case class SafetyConfig(
    epsilonNE: Double,
    deltaAdapt: Double,
    betaDet: Double
):
  require(epsilonNE >= 0.0, "epsilonNE must be non-negative")
  require(deltaAdapt >= 0.0, "deltaAdapt must be non-negative")
  require(betaDet > 0.0, "betaDet must be positive")
