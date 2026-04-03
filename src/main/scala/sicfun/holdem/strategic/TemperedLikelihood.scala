package sicfun.holdem.strategic

/** Two-layer tempered likelihood and posterior update (Defs 15, 15A, 15B).
  *
  * Implements three likelihood regularization modes from SICFUN-v0.30.2 Section 4.1:
  *
  * '''Two-layer tempered (default):'''
  * {{{L_{kappa,delta}(y | c) = Pr(y | c)^kappa_temp + delta_floor * eta(y)}}}
  *
  * '''Pure power-posterior:'''
  * Two-layer with delta_floor = 0. Totality conditional on at least one class
  * having positive base probability.
  *
  * '''Legacy epsilon-smoothing (backward compatibility):'''
  * {{{L_legacy(y | c) = (1 - epsilon) * Pr(y | c) + epsilon * eta(y)}}}
  * Recovers v0.29.1 behavior exactly. This is a SEPARATE formula, not a special
  * case of the two-layer form.
  *
  * Theorem 1: When delta_floor > 0 and eta has full support, the posterior
  * (Def 15B) is unconditionally well-defined for any prior with non-empty support.
  */
object TemperedLikelihood:

  /** Configuration for tempered inference (Def 15).
    *
    * Three semantic modes:
    *  - [[TemperedConfig.twoLayer]]: power-posterior + additive floor (default)
    *  - [[TemperedConfig.purePowerPosterior]]: delta_floor = 0
    *  - [[TemperedConfig.legacy]]: v0.29.1 epsilon-smoothing
    */
  enum TemperedConfig:
    /** Two-layer tempered semantics (Def 15, default mode).
      * @param kappaTemp  power-posterior exponent, in (0, 1]
      * @param deltaFloor additive safety floor, >= 0
      */
    case TwoLayer(kappaTemp: Double, deltaFloor: Double)

    /** Legacy epsilon-smoothing semantics (Def 15, backward compat).
      * L(y|c) = (1 - epsilon) * Pr(y|c) + epsilon * eta(y)
      * @param epsilon smoothing weight, in (0, 1)
      */
    case Legacy(epsilon: Double)

  object TemperedConfig:
    /** Creates a two-layer tempered config with validation.
      * @param kappaTemp  in (0, 1]
      * @param deltaFloor >= 0
      */
    def twoLayer(kappaTemp: Double, deltaFloor: Double): TemperedConfig =
      require(
        kappaTemp > 0.0 && kappaTemp <= 1.0 && java.lang.Double.isFinite(kappaTemp),
        s"kappaTemp must be in (0, 1], got $kappaTemp"
      )
      require(
        deltaFloor >= 0.0 && java.lang.Double.isFinite(deltaFloor),
        s"deltaFloor must be >= 0 and finite, got $deltaFloor"
      )
      TemperedConfig.TwoLayer(kappaTemp, deltaFloor)

    /** Creates a pure power-posterior config (delta_floor = 0). */
    def purePowerPosterior(kappaTemp: Double): TemperedConfig =
      twoLayer(kappaTemp = kappaTemp, deltaFloor = 0.0)

    /** Creates a legacy epsilon-smoothing config for v0.29.1 backward compatibility.
      * @param epsilon in (0, 1)
      */
    def legacy(epsilon: Double): TemperedConfig =
      require(
        epsilon > 0.0 && epsilon < 1.0 && java.lang.Double.isFinite(epsilon),
        s"epsilon must be in (0, 1), got $epsilon"
      )
      TemperedConfig.Legacy(epsilon)

  extension (cfg: TemperedConfig)
    /** Whether this is the legacy epsilon-smoothing mode. */
    def isLegacy: Boolean = cfg match
      case TemperedConfig.Legacy(_) => true
      case _ => false

    /** The kappaTemp value. Legacy mode returns 1.0 (no tempering). */
    def kappaTemp: Double = cfg match
      case TemperedConfig.TwoLayer(k, _) => k
      case TemperedConfig.Legacy(_) => 1.0

    /** The deltaFloor value. Legacy mode returns the epsilon. */
    def deltaFloor: Double = cfg match
      case TemperedConfig.TwoLayer(_, d) => d
      case TemperedConfig.Legacy(eps) => eps

    /** The epsilon value for legacy mode. Throws for non-legacy configs. */
    def epsilon: Double = cfg match
      case TemperedConfig.Legacy(eps) => eps
      case _ => throw new UnsupportedOperationException("epsilon only available in legacy mode")

  /** Default eta: uniform distribution over hypothesisCount classes. */
  def defaultEta(hypothesisCount: Int): Array[Double] =
    require(hypothesisCount > 0, s"hypothesisCount must be positive, got $hypothesisCount")
    val p = 1.0 / hypothesisCount.toDouble
    Array.fill(hypothesisCount)(p)

  /** Computes two-layer tempered likelihoods (Def 15A).
    *
    * L(y | c_i) = basePr(i)^kappaTemp + deltaFloor * eta(i)
    *
    * @param basePr      Pr(y | c) for each class, length = classCount
    * @param kappaTemp   power-posterior exponent in (0, 1]
    * @param deltaFloor  additive safety floor >= 0
    * @param eta         full-support distribution, length = classCount
    * @return tempered likelihoods, length = classCount
    */
  def computeLikelihoods(
      basePr: Array[Double],
      kappaTemp: Double,
      deltaFloor: Double,
      eta: Array[Double]
  ): Array[Double] =
    val n = basePr.length
    val result = new Array[Double](n)
    var i = 0
    while i < n do
      val base = math.max(0.0, basePr(i))
      val powered = if kappaTemp == 1.0 then base else math.pow(base, kappaTemp)
      result(i) = powered + deltaFloor * eta(i)
      i += 1
    result

  /** Computes legacy epsilon-smoothed likelihoods (Def 15, backward compat).
    *
    * L_legacy(y | c_i) = (1 - epsilon) * basePr(i) + epsilon * eta(i)
    *
    * @param basePr  Pr(y | c) for each class, length = classCount
    * @param epsilon smoothing weight in (0, 1)
    * @param eta     full-support distribution, length = classCount
    * @return smoothed likelihoods, length = classCount
    */
  def computeLikelihoodsLegacy(
      basePr: Array[Double],
      epsilon: Double,
      eta: Array[Double]
  ): Array[Double] =
    val n = basePr.length
    val oneMinusEps = 1.0 - epsilon
    val result = new Array[Double](n)
    var i = 0
    while i < n do
      result(i) = oneMinusEps * math.max(0.0, basePr(i)) + epsilon * eta(i)
      i += 1
    result

  /** Updates the posterior distribution using tempered likelihoods (Def 15B).
    *
    * mu_{t+1}(c) = L(y|c) * mu_t(c) / sum_c' L(y|c') * mu_t(c')
    *
    * When delta_floor = 0 and the denominator vanishes, the normalized prior is preserved.
    *
    * @param prior   prior distribution (unnormalized ok), length = classCount
    * @param basePr  Pr(y | c) for each class, length = classCount
    * @param eta     full-support distribution, length = classCount
    * @param config  tempered configuration
    * @return normalized posterior distribution, length = classCount
    */
  def updatePosterior(
      prior: Array[Double],
      basePr: Array[Double],
      eta: Array[Double],
      config: TemperedConfig
  ): Array[Double] =
    val n = prior.length
    // Step 1: normalize the prior
    var priorSum = 0.0
    var i = 0
    while i < n do
      priorSum += prior(i)
      i += 1
    require(priorSum > 0.0, "prior must have positive total mass")
    val invPriorSum = 1.0 / priorSum

    // Step 2: compute tempered likelihoods
    val likelihoods = config match
      case TemperedConfig.TwoLayer(kappa, delta) =>
        computeLikelihoods(basePr, kappa, delta, eta)
      case TemperedConfig.Legacy(eps) =>
        computeLikelihoodsLegacy(basePr, eps, eta)

    // Step 3: multiply prior by likelihoods and compute evidence
    val posterior = new Array[Double](n)
    var evidence = 0.0
    i = 0
    while i < n do
      val unnorm = (prior(i) * invPriorSum) * likelihoods(i)
      posterior(i) = unnorm
      evidence += unnorm
      i += 1

    // Step 4: normalize posterior (or preserve prior if evidence = 0, Def 15B)
    if evidence > 0.0 then
      val invEvidence = 1.0 / evidence
      i = 0
      while i < n do
        posterior(i) *= invEvidence
        i += 1
    else
      // Prior preservation (Def 15B: delta=0 and denominator vanishes)
      i = 0
      while i < n do
        posterior(i) = prior(i) * invPriorSum
        i += 1

    posterior
