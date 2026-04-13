package sicfun.holdem.strategic

import sicfun.holdem.strategic.solver.TabularGenerativeModel

/** Profile-conditioned per-state robust loss evaluation for the formal certification path. */
object PerStateLossEvaluator:

  /** Value iteration for a fixed policy on a tabular model.
    *
    * V^π(s) = R(s, π(s)) + γ * V^π(T(s, π(s)))
    */
  def valueIteration(
      model: TabularGenerativeModel,
      policy: Int => Int,
      gamma: Double,
      maxIterations: Int = 200,
      tolerance: Double = 1e-10
  ): Array[Double] =
    val n = model.numStates
    var values = Array.fill(n)(0.0)
    var iter = 0
    var converged = false
    while iter < maxIterations && !converged do
      val next = new Array[Double](n)
      var maxDiff = 0.0
      var s = 0
      while s < n do
        val a = policy(s)
        val reward = model.rewardTable(s * model.numActions + a)
        val successor = model.transitionTable(s * model.numActions + a)
        next(s) = reward + gamma * values(successor)
        val diff = math.abs(next(s) - values(s))
        if diff > maxDiff then maxDiff = diff
        s += 1
      converged = maxDiff < tolerance
      values = next
      iter += 1
    values

  /** Compute robust losses robustLosses[s][a] from profile-conditioned models.
    *
    * L_robust(s, a) = max_σ max(0, V^π_σ(s) - R_σ(s,a) - γ * V^π_σ(T_σ(s,a)))
    */
  def computeRobustLosses(
      profileModels: IndexedSeq[TabularGenerativeModel],
      refPolicy: Int => Int,
      gamma: Double,
      maxIterations: Int = 200,
      tolerance: Double = 1e-10
  ): Array[Array[Double]] =
    require(profileModels.nonEmpty, "need at least one profile model")
    val numStates = profileModels.head.numStates
    val numActions = profileModels.head.numActions

    // Evaluate V^π_σ for each profile
    val profileValues = profileModels.map(m => valueIteration(m, refPolicy, gamma, maxIterations, tolerance))

    // Compute per-state per-action robust losses
    val losses = Array.ofDim[Double](numStates, numActions)
    var s = 0
    while s < numStates do
      var a = 0
      while a < numActions do
        var maxLoss = 0.0
        var p = 0
        while p < profileModels.size do
          val model = profileModels(p)
          val values = profileValues(p)
          val reward = model.rewardTable(s * numActions + a)
          val successor = model.transitionTable(s * numActions + a)
          val loss = math.max(0.0, values(s) - reward - gamma * values(successor))
          if loss > maxLoss then maxLoss = loss
          p += 1
        losses(s)(a) = maxLoss
        a += 1
      s += 1
    losses
