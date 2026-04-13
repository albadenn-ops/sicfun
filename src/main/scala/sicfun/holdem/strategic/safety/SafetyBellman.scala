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
    * L_robust(s, a) = sup_{sigma in Sigma^{-S}} [V_bar(s; sigma) - Q_bar(s, a; sigma)]_+
    *
    * Over a finite rival profile class, the supremum is the maximum.
    * Each entry in the arrays corresponds to one rival profile sigma^{-S}.
    * The [.]_+ (positive part) ensures the loss is non-negative per the spec.
    */
  def robustOneStepLoss(
      baselineValues: Array[Double],
      qValues: Array[Double]
  ): Double =
    require(baselineValues.length == qValues.length && baselineValues.length > 0,
      "arrays must be non-empty and same length")
    var maxLoss = 0.0 // [.]_+ clamp: loss is non-negative (Def 59)
    var i = 0
    while i < baselineValues.length do
      val loss = baselineValues(i) - qValues(i)
      if loss > maxLoss then maxLoss = loss
      i += 1
    maxLoss

  /** Safe Bellman operator T_safe (Def 60).
    *
    * (T_safe B)(s) = min_a [ L_robust(s, a) + gamma * max_σ B(T_σ(s, a)) ]
    *
    * The future term uses transition-aware successor bounds:
    * for each (s, a), the worst-case successor state reachable under any
    * profile σ determines the future cost. This is exact for deterministic
    * transitions; for stochastic transitions, use expectation inside max_σ.
    *
    * @param currentBound current safety bound per state B(s)
    * @param robustLosses robust one-step losses, indexed [state][action]
    * @param gamma discount factor
    * @param transitions (stateIdx, actionIdx, profileIdx) => successor stateIdx
    * @param numProfiles number of rival profiles σ
    * @return updated bound per state
    */
  def tSafe(
      currentBound: Array[Double],
      robustLosses: Array[Array[Double]],
      gamma: Double,
      transitions: (Int, Int, Int) => Int,
      numProfiles: Int,
      terminalStates: Set[Int] = Set.empty
  ): Array[Double] =
    require(gamma >= 0.0 && gamma < 1.0, s"gamma must be in [0,1), got $gamma")
    require(numProfiles > 0, "must have at least one profile")
    val numStates = currentBound.length
    require(robustLosses.length == numStates, "robustLosses must have one row per state")
    val result = new Array[Double](numStates)
    var s = 0
    while s < numStates do
      if terminalStates.contains(s) then
        result(s) = 0.0 // (T_safe B)(b) = 0 for b in B_term (Def 60)
      else
        val losses = robustLosses(s)
        var minOverActions = Double.PositiveInfinity
        var a = 0
        while a < losses.length do
          val maxFuture = worstCaseFuture(currentBound, s, a, transitions, numProfiles)
          val candidate = losses(a) + gamma * maxFuture
          if candidate < minOverActions then minOverActions = candidate
          a += 1
        result(s) = if losses.isEmpty then 0.0 else minOverActions
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
      transitions: (Int, Int, Int) => Int,
      numProfiles: Int,
      maxIterations: Int = 200,
      tolerance: Double = 1e-10,
      terminalStates: Set[Int] = Set.empty
  ): Array[Double] =
    val numStates = robustLosses.length
    var bound = Array.fill(numStates)(0.0)
    var iter = 0
    var converged = false
    while iter < maxIterations && !converged do
      val next = tSafe(bound, robustLosses, gamma, transitions, numProfiles, terminalStates)
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
    * A_safe(s, B) = { a : L_robust(s, a) + gamma * max_σ B(T_σ(s, a)) <= B(s) }
    *
    * Returns indices of safe actions at state s.
    */
  def safeActionSet(
      stateIndex: Int,
      bound: Array[Double],
      robustLosses: Array[Array[Double]],
      gamma: Double,
      transitions: (Int, Int, Int) => Int,
      numProfiles: Int
  ): IndexedSeq[Int] =
    val threshold = bound(stateIndex)
    val losses = robustLosses(stateIndex)
    (0 until losses.length).filter { a =>
      val maxFuture = worstCaseFuture(bound, stateIndex, a, transitions, numProfiles)
      losses(a) + gamma * maxFuture <= threshold + 1e-12
    }

  /** max_σ B(T_σ(s, a)): worst-case future bound across profiles. */
  private def worstCaseFuture(
      bound: Array[Double],
      s: Int, a: Int,
      transitions: (Int, Int, Int) => Int,
      numProfiles: Int
  ): Double =
    var maxFuture = Double.NegativeInfinity
    var p = 0
    while p < numProfiles do
      val successor = transitions(s, a, p)
      val futureVal = bound(successor)
      if futureVal > maxFuture then maxFuture = futureVal
      p += 1
    maxFuture

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
        gamma: Double,
        transitions: (Int, Int, Int) => Int,
        numProfiles: Int
    ): Boolean =
      val tSafeResult = SafetyBellman.tSafe(values, robustLosses, gamma, transitions, numProfiles, terminalStates)
      values.indices.forall(s => values(s) >= tSafeResult(s) - 1e-12)

    /** Validate all four structural constraints. */
    def isValid(
        robustLosses: Array[Array[Double]],
        gamma: Double,
        maxBound: Double,
        transitions: (Int, Int, Int) => Int,
        numProfiles: Int
    ): Boolean =
      satisfiesTerminality &&
        satisfiesNonNegativity &&
        satisfiesGlobalBound(maxBound) &&
        satisfiesMonotonicity(robustLosses, gamma, transitions, numProfiles)

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

  // ==== Chain-world indexed wrappers (§9C / §9D) ====

  /** Chain-indexed safety Bellman operator (Def 60, chain-indexed form).
    *
    * T_safe^{(omega^act, omega^sd)}: same algorithm as tSafe, but the caller
    * provides robust losses computed under the specified chain world's kernel.
    *
    * @param world the chain world determining which kernel profile computes the losses
    * @param currentBound current safety bound per state
    * @param robustLosses robust one-step losses under `world`'s kernel, indexed [state][action]
    * @param gamma discount factor
    * @return updated bound per state under this chain world
    */
  def tSafeForWorld(
      world: ChainWorld,
      currentBound: Array[Double],
      robustLosses: Array[Array[Double]],
      gamma: Double,
      transitions: (Int, Int, Int) => Int,
      numProfiles: Int,
      terminalStates: Set[Int] = Set.empty
  ): Array[Double] =
    tSafe(currentBound, robustLosses, gamma, transitions, numProfiles, terminalStates)

  /** Chain-indexed fixed-point B*^{(omega^act, omega^sd)} (Def 61, chain-indexed form).
    *
    * Iterates T_safe^{(omega^act, omega^sd)} to convergence.
    */
  def computeBStarForWorld(
      world: ChainWorld,
      robustLosses: Array[Array[Double]],
      gamma: Double,
      transitions: (Int, Int, Int) => Int,
      numProfiles: Int,
      maxIterations: Int = 200,
      tolerance: Double = 1e-10,
      terminalStates: Set[Int] = Set.empty
  ): Array[Double] =
    computeBStar(robustLosses, gamma, transitions, numProfiles, maxIterations, tolerance, terminalStates)

  /** Chain-indexed safe action set U*^{(omega^act, omega^sd)}_safe (Def 62, chain-indexed form). */
  def safeActionSetForWorld(
      world: ChainWorld,
      stateIndex: Int,
      bound: Array[Double],
      robustLosses: Array[Array[Double]],
      gamma: Double,
      transitions: (Int, Int, Int) => Int,
      numProfiles: Int
  ): IndexedSeq[Int] =
    safeActionSet(stateIndex, bound, robustLosses, gamma, transitions, numProfiles)

  /** Chain-indexed safe-feasible policy selector (Def 63, chain-indexed form).
    *
    * pi^{safe,(omega^act,omega^sd)}(b) in argmax_{u in U*^{(omega^act,omega^sd)}_safe(b)} Q^{(omega^act,omega^sd)}(b, u)
    */
  def safeFeasibleActionForWorld(
      world: ChainWorld,
      qValues: Array[Double],
      safeActions: IndexedSeq[Int]
  ): Int =
    safeFeasibleAction(qValues, safeActions)

  /** Belief-level safe action filtering (Def 62 analog).
    *
    * Lifts latent-state B* to belief level via particle expectation.
    * Conservative approximation — not exact belief-space Bellman.
    *
    * @param belief particle weights per state
    * @param bStar B* per latent state
    * @param robustLosses [state][action]
    * @param gamma discount factor
    * @param transitions (s, a, profileIdx) => successor state
    * @param numProfiles number of profiles
    * @return indices of safe actions at the belief level
    */
  def beliefLevelSafeActions(
      belief: Array[Double],
      bStar: Array[Double],
      robustLosses: Array[Array[Double]],
      gamma: Double,
      transitions: (Int, Int, Int) => Int,
      numProfiles: Int
  ): IndexedSeq[Int] =
    require(belief.length == bStar.length, "belief and bStar must match in size")
    val numStates = belief.length
    val numActions = if robustLosses.isEmpty then 0 else robustLosses(0).length
    val threshold = {
      var sum = 0.0
      var s = 0
      while s < numStates do
        sum += belief(s) * bStar(s)
        s += 1
      sum
    }
    (0 until numActions).filter { a =>
      var bBeliefA = 0.0
      var s = 0
      while s < numStates do
        val maxFuture = worstCaseFuture(bStar, s, a, transitions, numProfiles)
        bBeliefA += belief(s) * (robustLosses(s)(a) + gamma * maxFuture)
        s += 1
      bBeliefA <= threshold + 1e-12
    }

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
