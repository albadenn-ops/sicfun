package sicfun.holdem.strategic

/** Bellman-safe certificates and local safety operators (Wave 4 — v0.31.1 formal closure).
  *
  * Implements:
  *   - One-step baseline loss (Def 58)
  *   - Robust one-step loss (Def 59)
  *   - Safe Bellman operator T_safe (Def 60)
  *   - Fixed-point safety bound B* (Def 61)
  *   - Safe action sets (Def 62)
  *   - Safe-feasible policy selector (Def 63)
  *   - Required adaptation budget (Def 64)
  *   - Structural certificate B_beta (Def 65)
  *   - Certificate validation and dominance (Def 66)
  */
object SafetyBellman:

  /** One-step baseline loss (Def 58).
    *
    * L_base(s, a) = V_bar(s) - Q_bar(s, a)
    *
    * Measures how much value the agent loses at state s by taking action a
    * instead of following the baseline policy bar_pi.
    */
  def baselineLoss(baselineValue: Double, qBaseline: Double): Double =
    baselineValue - qBaseline

  /** Robust one-step loss (Def 59).
    *
    * L_robust(s, a) = sup_{b' in B(b, rho)} [V_bar(s, b') - Q(s, a, b')]
    *
    * Over a finite belief set, the supremum is the maximum.
    * Each entry in the arrays corresponds to one belief point.
    */
  def robustOneStepLoss(
      baselineValues: Array[Double],
      qValues: Array[Double]
  ): Double =
    require(baselineValues.length == qValues.length && baselineValues.length > 0,
      "arrays must be non-empty and same length")
    var maxLoss = Double.NegativeInfinity
    var i = 0
    while i < baselineValues.length do
      val loss = baselineValues(i) - qValues(i)
      if loss > maxLoss then maxLoss = loss
      i += 1
    maxLoss

  /** Safe Bellman operator T_safe (Def 60).
    *
    * (T_safe B)(s) = max_a [ L_robust(s, a) + gamma * max_{s'} B(s') ]
    *
    * For finite state/action problems. Returns the updated bound for each state.
    *
    * @param currentBound current safety bound per state (B(s))
    * @param robustLosses robust one-step losses, indexed [state][action]
    * @param gamma discount factor
    * @return updated bound per state
    */
  def tSafe(
      currentBound: Array[Double],
      robustLosses: Array[Array[Double]],
      gamma: Double
  ): Array[Double] =
    require(gamma >= 0.0 && gamma < 1.0, s"gamma must be in [0,1), got $gamma")
    val numStates = currentBound.length
    require(robustLosses.length == numStates, "robustLosses must have one row per state")
    val maxFuture = if numStates == 0 then 0.0 else currentBound.max
    val result = new Array[Double](numStates)
    var s = 0
    while s < numStates do
      val losses = robustLosses(s)
      var maxOverActions = Double.NegativeInfinity
      var a = 0
      while a < losses.length do
        val candidate = losses(a) + gamma * maxFuture
        if candidate > maxOverActions then maxOverActions = candidate
        a += 1
      result(s) = maxOverActions
      s += 1
    result

  /** Fixed-point iteration for B* (Def 61).
    *
    * Iterates T_safe until convergence or maxIterations.
    *
    * @param robustLosses robust one-step losses [state][action]
    * @param gamma discount factor
    * @param maxIterations maximum iterations
    * @param tolerance convergence tolerance
    * @return B* per state
    */
  def computeBStar(
      robustLosses: Array[Array[Double]],
      gamma: Double,
      maxIterations: Int = 200,
      tolerance: Double = 1e-10
  ): Array[Double] =
    val numStates = robustLosses.length
    var bound = Array.fill(numStates)(0.0)
    var iter = 0
    var converged = false
    while iter < maxIterations && !converged do
      val next = tSafe(bound, robustLosses, gamma)
      var maxDiff = 0.0
      var s = 0
      while s < numStates do
        val diff = math.abs(next(s) - bound(s))
        if diff > maxDiff then maxDiff = diff
        s += 1
      converged = maxDiff < tolerance
      bound = next
      iter += 1
    bound

  /** Safe action set (Def 62).
    *
    * A_safe(s, B) = { a : L_robust(s, a) + gamma * max_{s'} B(s') <= B(s) }
    *
    * Returns indices of safe actions at state s.
    */
  def safeActionSet(
      stateIndex: Int,
      bound: Array[Double],
      robustLosses: Array[Array[Double]],
      gamma: Double
  ): IndexedSeq[Int] =
    val maxFuture = bound.max
    val threshold = bound(stateIndex)
    val losses = robustLosses(stateIndex)
    (0 until losses.length).filter { a =>
      losses(a) + gamma * maxFuture <= threshold + 1e-12 // small tolerance for floating-point
    }

  /** Safe-feasible policy selector (Def 63).
    *
    * Given Q-values and a safe action set, selects the action with the
    * highest Q-value among safe actions. Falls back to the least-unsafe
    * action if the safe set is empty.
    */
  def safeFeasibleAction(
      qValues: Array[Double],
      safeActions: IndexedSeq[Int]
  ): Int =
    if safeActions.nonEmpty then
      safeActions.maxBy(a => qValues(a))
    else
      // Fallback: pick the action with the smallest robust loss
      // (approximated here as highest Q-value among all actions)
      qValues.indices.maxBy(a => qValues(a))

  /** Required adaptation budget (Def 64).
    *
    * epsilon_adapt = max_s B*(s)
    *
    * The minimum adaptation tolerance needed to guarantee safety under
    * the Bellman-safe law.
    */
  def requiredAdaptationBudget(bStar: Array[Double]): Double =
    if bStar.isEmpty then 0.0
    else bStar.max

  /** Structural certificate B_beta (Def 65).
    *
    * A certificate is a function B_beta: S -> R+ satisfying:
    *   (C1) Terminality: B_beta(s_terminal) = 0
    *   (C2) Non-negativity: B_beta(s) >= 0 for all s
    *   (C3) Global bound: max_s B_beta(s) < infinity
    *   (C4) Horizon monotonicity: B_beta dominates T_safe B_beta
    *
    * Represented as an array indexed by state.
    */
  final case class Certificate(
      values: Array[Double],
      terminalStates: Set[Int]
  ):
    /** (C1) Terminality: certificate is zero at all terminal states. */
    def satisfiesTerminality: Boolean =
      terminalStates.forall(s => s < values.length && math.abs(values(s)) < 1e-12)

    /** (C2) Non-negativity. */
    def satisfiesNonNegativity: Boolean =
      values.forall(_ >= -1e-12)

    /** (C3) Global bound (finite, bounded by provided limit). */
    def satisfiesGlobalBound(maxBound: Double): Boolean =
      values.forall(v => v <= maxBound + 1e-12)

    /** (C4) Horizon monotonicity: B_beta dominates T_safe(B_beta).
      * For all s: B_beta(s) >= (T_safe B_beta)(s).
      */
    def satisfiesMonotonicity(
        robustLosses: Array[Array[Double]],
        gamma: Double
    ): Boolean =
      val tSafeResult = SafetyBellman.tSafe(values, robustLosses, gamma)
      values.indices.forall(s => values(s) >= tSafeResult(s) - 1e-12)

    /** Validate all four structural constraints. */
    def isValid(
        robustLosses: Array[Array[Double]],
        gamma: Double,
        maxBound: Double
    ): Boolean =
      satisfiesTerminality &&
        satisfiesNonNegativity &&
        satisfiesGlobalBound(maxBound) &&
        satisfiesMonotonicity(robustLosses, gamma)

  /** Certificate dominance (Def 66).
    *
    * B_beta dominates B* iff B_beta(s) >= B*(s) for all s.
    */
  def certificateDominates(
      certificate: Array[Double],
      bStar: Array[Double]
  ): Boolean =
    require(certificate.length == bStar.length, "arrays must be same length")
    certificate.indices.forall(s => certificate(s) >= bStar(s) - 1e-12)

/** Total vulnerability budget (Corollary 9.3).
  *
  * epsilon_total = epsilon_base + epsilon_adapt
  *
  * Connects the deployment baseline (A10) to the adaptation safety budget.
  */
object TotalVulnerability:
  /** Compute total vulnerability budget.
    *
    * @param baseline the deployment baseline from A10
    * @param epsilonAdapt the adaptation safety tolerance
    * @param isExact whether both components are exact or conservative
    * @return (total vulnerability, fidelity description)
    */
  def compute(
      baseline: DeploymentBaseline,
      epsilonAdapt: Ev
  ): (Ev, Fidelity) =
    require(epsilonAdapt >= Ev.Zero, "epsilonAdapt must be non-negative")
    val total = baseline.baselineExploitability + epsilonAdapt
    // Both components are finite approximations in practice
    (total, Fidelity.Approximate)
