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

  /** Def 52 (legacy): Check whether adaptation safety holds.
    * Safe iff Exploit(pi) <= epsilon_NE + epsilon_adapt.
    */
  def isSafe(exploitability: Double, epsilonNE: Double, epsilonAdapt: Double): Boolean =
    exploitability <= epsilonNE + epsilonAdapt

  /** Theorem 8: Compute betaBar -- the supremum of safe exploitation interpolation.
    *
    * betaBar(epsilon_adapt) = sup { beta' in [0,1] : Exploit(pi^S_{beta'}) <= eps + epsilon_adapt }
    *
    * Uses binary search over [0,1] since exploitability is upper-semicontinuous
    * in beta. The exploitabilityAtBeta function maps beta -> Exploit(pi^S_beta).
    *
    * @param epsilonAdapt          adaptation safety budget
    * @param epsilonNE             baseline exploitability of the epsilon-NE strategy
    * @param exploitabilityAtBeta  oracle: beta -> exploitability at that beta
    * @param iterations            binary search iterations (default 50)
    * @return betaBar in [0,1]
    */
  def betaBar(
      epsilonAdapt: Double,
      epsilonNE: Double,
      exploitabilityAtBeta: Double => Double,
      iterations: Int = 50
  ): Double =
    val bound = epsilonNE + epsilonAdapt
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
    * Adaptation is AS-strong iff for every belief b and every rival profile sigma^{-S}:
    *   J(b; pi, sigma^{-S}) >= J(b; bar_pi, sigma^{-S}) - epsilon_adapt
    *
    * Over a finite belief set and finite profile class, this checks every (b, sigma) pair.
    *
    * @param jPolicyPerBelief  J(b; pi, sigma) for each belief, then each profile: [belief][profile]
    * @param jBaselinePerBelief J(b; bar_pi, sigma) for each belief, then each profile: [belief][profile]
    * @param epsilonAdapt adaptation safety tolerance
    */
  def isStrongSafe(
      jPolicyPerBelief: IndexedSeq[IndexedSeq[Ev]],
      jBaselinePerBelief: IndexedSeq[IndexedSeq[Ev]],
      epsilonAdapt: Ev
  ): Boolean =
    require(jPolicyPerBelief.size == jBaselinePerBelief.size,
      "must have same number of belief points")
    jPolicyPerBelief.zip(jBaselinePerBelief).forall { (jPolProfiles, jBaseProfiles) =>
      require(jPolProfiles.size == jBaseProfiles.size, "must have same number of profiles per belief")
      jPolProfiles.zip(jBaseProfiles).forall { (jPol, jBase) =>
        jPol >= jBase - epsilonAdapt
      }
    }

  /** Def 57 (security-value approximation): conservative lower-bound check.
    *
    * Checks V^sec(b; pi) >= V^sec(b; bar_pi) - epsilon_adapt, which is a
    * NECESSARY but not sufficient condition for AS-strong (Def 57).
    * Use isStrongSafe for the exact per-profile check.
    */
  def isStrongSafeApprox(
      securityValuePolicy: IndexedSeq[Ev],
      securityValueBaseline: IndexedSeq[Ev],
      epsilonAdapt: Ev
  ): Boolean =
    require(securityValuePolicy.size == securityValueBaseline.size,
      "must have same number of belief points")
    securityValuePolicy.zip(securityValueBaseline).forall { (vPol, vBase) =>
      vPol >= vBase - epsilonAdapt
    }

  /** Def 57A: Robust regret relative to baseline (exact per-profile form).
    *
    * Reg^rob_b(pi || bar_pi) = sup_{sigma in Sigma^{-S}} [J(b; bar_pi, sigma) - J(b; pi, sigma)]
    *
    * @param jBaselinePerProfile J(b; bar_pi, sigma_i) for each profile at a fixed belief
    * @param jPolicyPerProfile   J(b; pi, sigma_i) for each profile at a fixed belief
    */
  def robustRegret(
      jBaselinePerProfile: IndexedSeq[Ev],
      jPolicyPerProfile: IndexedSeq[Ev]
  ): Ev =
    require(jBaselinePerProfile.size == jPolicyPerProfile.size && jBaselinePerProfile.nonEmpty,
      "must have same number of profiles and at least one")
    jBaselinePerProfile.zip(jPolicyPerProfile)
      .map((jBase, jPol) => jBase - jPol)
      .reduce((a, b) => if a >= b then a else b)

  /** Def 57A (security-value approximation): lower bound on robust regret.
    *
    * Computes inf_sigma J(bar_pi, sigma) - inf_sigma J(pi, sigma), which is
    * a lower bound on sup_sigma [J(bar_pi, sigma) - J(pi, sigma)].
    */
  def robustRegretApprox(securityValueBaseline: Ev, securityValuePolicy: Ev): Ev =
    securityValueBaseline - securityValuePolicy

  /** Def 57A: Maximum robust regret over a finite belief set (exact per-profile form).
    *
    * @param jPolicyPerBelief  [belief][profile] J values for the adapted policy
    * @param jBaselinePerBelief [belief][profile] J values for the baseline
    */
  def maxRobustRegret(
      jPolicyPerBelief: IndexedSeq[IndexedSeq[Ev]],
      jBaselinePerBelief: IndexedSeq[IndexedSeq[Ev]]
  ): Ev =
    require(jPolicyPerBelief.size == jBaselinePerBelief.size,
      "must have same number of belief points")
    jPolicyPerBelief.zip(jBaselinePerBelief)
      .map((jPolProfiles, jBaseProfiles) => robustRegret(jBaseProfiles, jPolProfiles))
      .reduce((a, b) => if a >= b then a else b)

  /** Def 57A: Maximum robust regret (security-value approximation, lower bound). */
  def maxRobustRegretApprox(
      securityValuePolicy: IndexedSeq[Ev],
      securityValueBaseline: IndexedSeq[Ev]
  ): Ev =
    require(securityValuePolicy.size == securityValueBaseline.size,
      "must have same number of belief points")
    securityValuePolicy.zip(securityValueBaseline)
      .map((vPol, vBase) => robustRegretApprox(vBase, vPol))
      .reduce((a, b) => if a >= b then a else b)

  /** Def 57B: Adaptation-safe policy class (exact per-profile form).
    *
    * Pi_safe(epsilon_adapt) = { pi : max_{b in B_dep} Reg^rob_b(pi || bar_pi) <= epsilon_adapt }
    *
    * Returns true if the policy belongs to the safe class.
    */
  def inAdaptationSafeClass(
      jPolicyPerBelief: IndexedSeq[IndexedSeq[Ev]],
      jBaselinePerBelief: IndexedSeq[IndexedSeq[Ev]],
      epsilonAdapt: Ev
  ): Boolean =
    maxRobustRegret(jPolicyPerBelief, jBaselinePerBelief) <= epsilonAdapt

  /** Def 57B (security-value approximation, lower bound). */
  def inAdaptationSafeClassApprox(
      securityValuePolicy: IndexedSeq[Ev],
      securityValueBaseline: IndexedSeq[Ev],
      epsilonAdapt: Ev
  ): Boolean =
    maxRobustRegretApprox(securityValuePolicy, securityValueBaseline) <= epsilonAdapt

  /** Def 57C: Safe optimal value under chain world.
    *
    * V^{safe,(omega^act,omega^sd)}(b) = sup_{pi in Pi^S_safe} V^{pi, Gamma^{(omega^act,omega^sd)}}(b)
    *
    * Evaluates the best attainable value within the adaptation-safe policy class
    * under a specific chain world's kernel profile. Over a finite action set with
    * a precomputed safe action set, this reduces to max over safe Q-values.
    *
    * @param safeQValues Q-values under the specified chain world, indexed by action,
    *                    restricted to the safe action set
    * @return the safe optimal value (max safe Q-value), or Ev.Zero if no safe actions
    */
  def safeOptimalValueForWorld(
      safeQValues: IndexedSeq[Ev]
  ): Ev =
    if safeQValues.isEmpty then Ev.Zero
    else safeQValues.reduce((a, b) => if a >= b then a else b)

/** Configuration for adaptation safety (bundles Def 53 + §9B parameters).
  *
  * v0.31.1 canonical: `epsilonAdapt` (was `deltaAdapt` in prior versions).
  */
final case class SafetyConfig(
    epsilonNE: Double,
    epsilonAdapt: Double,
    betaDet: Double
):
  require(epsilonNE >= 0.0, "epsilonNE must be non-negative")
  require(epsilonAdapt >= 0.0, "epsilonAdapt must be non-negative")
  require(betaDet > 0.0, "betaDet must be positive")
  /** Pre-v0.31.1 alias for epsilonAdapt. */
  inline def deltaAdapt: Double = epsilonAdapt
