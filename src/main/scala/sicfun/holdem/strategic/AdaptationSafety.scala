package sicfun.holdem.strategic

/** Adaptation safety framework (Defs 52-53 legacy + Defs 57/57A-C v0.31.1).
  *
  * Def 52: Exploit(pi^S_{beta}) <= epsilon_NE + epsilon_adapt.
  * Def 53: Affine equilibrium deterrence.
  * Theorem 8: clamping beta <= betaBar(epsilon_adapt) enforces A10.
  *
  * v0.31.1 additions:
  * Def 57: AS-strong predicate relative to deployment baseline.
  * Def 57A: Robust regret and max robust regret.
  * Def 57B: Adaptation-safe policy class.
  *
  * Symbol mapping:
  *   delta_adapt -> epsilon_adapt  (same quantity; SafetyConfig.epsilonAdapt alias)
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

  // ==== v0.31.1 Strong Safety (Defs 57/57A-C) ====

  /** Def 57: AS-strong predicate relative to deployment baseline.
    *
    * Adaptation is AS-strong iff for every rival profile sigma^{-S}:
    *   V^sec(b; pi) >= V^sec(b; bar_pi) - epsilon_adapt
    *
    * Over a finite belief set, this must hold at every belief point.
    *
    * @param securityValuePolicy security value of the adapted policy at each belief
    * @param securityValueBaseline security value of the baseline at each belief
    * @param epsilonAdapt adaptation safety tolerance
    */
  def isStrongSafe(
      securityValuePolicy: IndexedSeq[Ev],
      securityValueBaseline: IndexedSeq[Ev],
      epsilonAdapt: Ev
  ): Boolean =
    require(securityValuePolicy.size == securityValueBaseline.size,
      "must have same number of belief points")
    securityValuePolicy.zip(securityValueBaseline).forall { (vPol, vBase) =>
      vPol >= vBase - epsilonAdapt
    }

  /** Def 57A: Robust regret relative to baseline.
    *
    * RobustRegret(b; pi, bar_pi) = V^sec(b; bar_pi) - V^sec(b; pi)
    *
    * Non-negative when the baseline outperforms the adapted policy.
    */
  def robustRegret(securityValueBaseline: Ev, securityValuePolicy: Ev): Ev =
    securityValueBaseline - securityValuePolicy

  /** Def 57A: Maximum robust regret over a finite belief set. */
  def maxRobustRegret(
      securityValuePolicy: IndexedSeq[Ev],
      securityValueBaseline: IndexedSeq[Ev]
  ): Ev =
    require(securityValuePolicy.size == securityValueBaseline.size,
      "must have same number of belief points")
    securityValuePolicy.zip(securityValueBaseline)
      .map((vPol, vBase) => robustRegret(vBase, vPol))
      .reduce((a, b) => if a >= b then a else b)

  /** Def 57B: Adaptation-safe policy class.
    *
    * Pi_safe(epsilon_adapt) = { pi : max_{b in B_dep} RobustRegret(b; pi, bar_pi) <= epsilon_adapt }
    *
    * Returns true if the policy belongs to the safe class.
    */
  def inAdaptationSafeClass(
      securityValuePolicy: IndexedSeq[Ev],
      securityValueBaseline: IndexedSeq[Ev],
      epsilonAdapt: Ev
  ): Boolean =
    maxRobustRegret(securityValuePolicy, securityValueBaseline) <= epsilonAdapt

/** Configuration for adaptation safety (bundles Def 52 + 53 parameters).
  *
  * `deltaAdapt` is the same quantity as `epsilon_adapt` in v0.31.1.
  * Use `epsilonAdapt` alias for v0.31.1 code.
  */
final case class SafetyConfig(
    epsilonNE: Double,
    deltaAdapt: Double,
    betaDet: Double
):
  require(epsilonNE >= 0.0, "epsilonNE must be non-negative")
  require(deltaAdapt >= 0.0, "deltaAdapt must be non-negative")
  require(betaDet > 0.0, "betaDet must be positive")
  /** v0.31.1 alias for deltaAdapt. */
  inline def epsilonAdapt: Double = deltaAdapt
