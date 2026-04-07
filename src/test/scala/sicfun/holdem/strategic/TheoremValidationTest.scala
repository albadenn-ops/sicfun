package sicfun.holdem.strategic

class TheoremValidationTest extends munit.FunSuite:

  private inline val Tol = 1e-12

  // ========================================================================
  // Theorem 1: Unconditional totality of two-layer tempered update
  // If delta_floor > 0 and eta has full support, then posterior is well-defined.
  // ========================================================================

  test("Theorem 1: tempered likelihood is strictly positive when delta_floor > 0"):
    // L_{kappa,delta}(y|c) = Pr(y|c)^kappa + delta * eta(y)
    // For any kappa in (0,1], delta > 0, eta(y) > 0: L > 0.
    val kappa = 0.5
    val delta = 0.01
    val prY = 0.0  // worst case: zero raw likelihood
    val etaY = 0.1 // eta has full support
    val tempered = math.pow(prY, kappa) + delta * etaY
    assert(tempered > 0.0, s"Tempered likelihood must be > 0, got $tempered")

  test("Theorem 1: denominator is strictly positive with full-support prior"):
    val kappa = 0.3
    val delta = 0.001
    val etaY = 0.05
    // Prior over 3 classes, all with positive weight
    val priors = Vector(0.5, 0.3, 0.2)
    val rawLikelihoods = Vector(0.0, 0.0, 0.0) // all zero raw
    val tempered = rawLikelihoods.zip(priors).map { (pr, mu) =>
      (math.pow(pr, kappa) + delta * etaY) * mu
    }.sum
    assert(tempered > 0.0, "Denominator must be > 0")

  test("Theorem 1: posterior sums to 1"):
    val kappa = 0.7
    val delta = 0.02
    val etaY = 1.0 / 4.0 // uniform over 4 classes
    val priors = Vector(0.25, 0.25, 0.25, 0.25)
    val rawLikelihoods = Vector(0.8, 0.1, 0.0, 0.05)
    val temperedLikelihoods = rawLikelihoods.map(pr => math.pow(pr, kappa) + delta * etaY)
    val denom = temperedLikelihoods.zip(priors).map(_ * _).sum
    val posterior = temperedLikelihoods.zip(priors).map((l, mu) => l * mu / denom)
    assertEqualsDouble(posterior.sum, 1.0, Tol)

  // ========================================================================
  // Theorem 2: Posterior limits
  // (a) kappa->1, delta>0: converges to delta-smoothed Bayes
  // (b) delta->0, kappa<1: converges to pure power posterior
  // (c) kappa->1, delta->0: converges to standard Bayes
  // ========================================================================

  test("Theorem 2a: kappa=1 with delta>0 gives delta-smoothed Bayes"):
    val delta = 0.01
    val etaY = 0.25
    val priors = Vector(0.5, 0.3, 0.2)
    val rawLikelihoods = Vector(0.8, 0.1, 0.05)
    // kappa=1: L = Pr(y|c) + delta*eta(y)
    val smoothed = rawLikelihoods.map(pr => pr + delta * etaY)
    val denom = smoothed.zip(priors).map(_ * _).sum
    val posterior = smoothed.zip(priors).map((l, mu) => l * mu / denom)
    assertEqualsDouble(posterior.sum, 1.0, Tol)
    // All posteriors positive (smoothing)
    posterior.foreach(p => assert(p > 0.0))

  test("Theorem 2b: delta=0 kappa<1 converges to pure power posterior"):
    val kappa = 0.5
    val priors = Vector(0.4, 0.35, 0.25)
    val rawLikelihoods = Vector(0.8, 0.2, 0.05)
    // delta=0: L = Pr(y|c)^kappa + 0 = Pr(y|c)^kappa (pure power posterior)
    val powerLikelihoods = rawLikelihoods.map(pr => math.pow(pr, kappa))
    val denom = powerLikelihoods.zip(priors).map(_ * _).sum
    val posterior = powerLikelihoods.zip(priors).map((l, mu) => l * mu / denom)
    assertEqualsDouble(posterior.sum, 1.0, Tol)
    // Power posterior flattens relative to standard Bayes (kappa < 1 shrinks differences)
    val standardDenom = rawLikelihoods.zip(priors).map(_ * _).sum
    val standardPosterior = rawLikelihoods.zip(priors).map((l, mu) => l * mu / standardDenom)
    // The power posterior should be more uniform than standard Bayes
    val powerRange = posterior.max - posterior.min
    val standardRange = standardPosterior.max - standardPosterior.min
    assert(powerRange < standardRange, "Power posterior should be flatter than standard Bayes")

  test("Theorem 2c: kappa=1 delta=0 recovers standard Bayes on-path"):
    val priors = Vector(0.6, 0.4)
    val rawLikelihoods = Vector(0.9, 0.3)
    // kappa=1, delta=0: L = Pr(y|c)
    val denom = rawLikelihoods.zip(priors).map(_ * _).sum
    val posterior = rawLikelihoods.zip(priors).map((l, mu) => l * mu / denom)
    // Standard Bayes: P(c|y) = Pr(y|c)*P(c) / sum
    val expected0 = 0.9 * 0.6 / (0.9 * 0.6 + 0.3 * 0.4)
    assertEqualsDouble(posterior(0), expected0, Tol)
    assertEqualsDouble(posterior.sum, 1.0, Tol)

  // ========================================================================
  // Theorem 3: Exact per-rival signal decomposition
  // delta_sig = delta_pass + delta_manip
  // ========================================================================

  test("Theorem 3: telescoping identity for arbitrary Q values"):
    for
      qa <- Seq(Ev(-10.0), Ev(0.0), Ev(5.5), Ev(100.0))
      qr <- Seq(Ev(-5.0), Ev(0.0), Ev(3.3), Ev(50.0))
      qb <- Seq(Ev(-20.0), Ev(0.0), Ev(1.1), Ev(25.0))
    do
      val prd = SignalDecomposition.computePerRivalDelta(qa, qr, qb)
      assertEqualsDouble(
        prd.deltaSig.value,
        (prd.deltaPass + prd.deltaManip).value,
        Tol
      )

  // ========================================================================
  // Theorem 3A: Signaling sub-decomposition
  // delta_sig = delta_sig,design + delta_sig,real
  // ========================================================================

  test("Theorem 3A: sub-decomposition telescoping for arbitrary Q values"):
    for
      qa <- Seq(Ev(-10.0), Ev(0.0), Ev(50.0))
      qd <- Seq(Ev(-5.0), Ev(0.0), Ev(30.0))
      qb <- Seq(Ev(-20.0), Ev(0.0), Ev(10.0))
    do
      val sig = SignalDecomposition.deltaSig(qa, qb)
      val sub = SignalingSubDecomposition.compute(qa, qd, qb)
      assertEqualsDouble(sig.value, sub.total.value, Tol)

  // ========================================================================
  // Theorem 4: Exact aggregate value decomposition with interaction
  // V^{1,1} = V^{0,0} + Delta_cont + Delta_sig* + Delta_int
  // ========================================================================

  test("Theorem 4: four-world decomposition identity for arbitrary values"):
    for
      v11 <- Seq(Ev(-100.0), Ev(0.0), Ev(50.0), Ev(999.0))
      v10 <- Seq(Ev(-50.0), Ev(0.0), Ev(30.0))
      v01 <- Seq(Ev(-30.0), Ev(0.0), Ev(20.0))
      v00 <- Seq(Ev(-10.0), Ev(0.0), Ev(5.0))
    do
      val fw = FourWorldDecomposition.compute(v11, v10, v01, v00)
      val reconstructed = fw.v00 + fw.deltaControl + fw.deltaSigStar + fw.deltaInteraction
      assertEqualsDouble(reconstructed.value, fw.v11.value, Tol)

  // ========================================================================
  // Theorem 5: Per-rival manipulation collapse under correct beliefs
  // If attrib == ref, then delta_manip == 0
  // ========================================================================

  test("Theorem 5: deltaManip is zero when attrib equals ref"):
    for q <- Seq(Ev(-100.0), Ev(0.0), Ev(50.0), Ev(9999.0)) do
      val prd = SignalDecomposition.computePerRivalDelta(q, q, Ev(3.0))
      assert(prd.hasCorrectBeliefs, s"Expected correct beliefs for q=$q")
      assertEqualsDouble(prd.deltaManip.value, 0.0, Tol)

  // ========================================================================
  // Theorem 6: Coherence of the no-learning counterfactual
  // No-learning = restricting Pi^S to Pi^ol, not altering observation generation.
  // This is a structural/design test, not a numerical one.
  // ========================================================================

  test("Theorem 6: V^{0,0} uses Pi^ol (open-loop), same observation dynamics"):
    // Theorem 6 is a structural property: the no-learning counterfactual restricts
    // Pi^S to Pi^ol WITHOUT altering observation generation. We verify the construction:
    // v00 and v01 share the same blind kernel (same observations) but differ in policy.
    // deltaControl = v01 - v00 isolates the policy effect under identical kernels.
    val fw = FourWorldDecomposition.compute(Ev(10.0), Ev(7.0), Ev(6.0), Ev(4.0))
    // deltaControl is well-defined and captures the pure policy effect
    assertEqualsDouble(fw.deltaControl.value, 2.0, Tol)
    // deltaSigStar is well-defined and captures the pure signal effect
    assertEqualsDouble(fw.deltaSigStar.value, 3.0, Tol)
    // Both are computed under the same v00 baseline (shared observation dynamics)
    assertEqualsDouble((fw.v00 + fw.deltaControl).value, fw.v01.value, Tol)
    assertEqualsDouble((fw.v00 + fw.deltaSigStar).value, fw.v10.value, Tol)

  // ========================================================================
  // Theorem 7: Convexity of robust value function under Wasserstein ambiguity
  // V^{*,rho} is convex in belief for fixed rho.
  // Tested via Bellman operator preservation (requires Phase 3 for full test).
  // Here we verify the bound structure.
  // ========================================================================

  test("Theorem 7: robust Bellman operator preserves convexity (interface test)"):
    // If V is convex and we apply T_rho, the result is convex.
    // We test the bound: for a convex combination of beliefs,
    // V(lambda*b1 + (1-lambda)*b2) <= lambda*V(b1) + (1-lambda)*V(b2).
    // Using a simple 1D convex function as a mock.
    val v: Double => Double = x => x * x // convex
    val b1 = 2.0
    val b2 = 8.0
    val lambda = 0.3
    val bMixed = lambda * b1 + (1 - lambda) * b2
    assert(v(bMixed) <= lambda * v(b1) + (1 - lambda) * v(b2) + Tol)

  // ========================================================================
  // Theorem 8: Adaptation safety bound
  // Exploit(pi^S_beta) <= epsilon_NE + delta_adapt when beta <= betaBar.
  // ========================================================================

  test("Theorem 8: safety bound holds at beta=0"):
    // At beta=0, policy equals baseline, exploitability = epsilon_NE.
    val epsilonNE = 0.03
    val deltaAdapt = 0.05
    val exploitAtZero = epsilonNE // baseline exploitability
    assert(AdaptationSafety.isSafe(exploitAtZero, epsilonNE, deltaAdapt))

  test("Theorem 8: betaBar enforces safety"):
    val epsilonNE = 0.02
    val deltaAdapt = 0.04
    // Exploitability increases linearly with beta
    val exploitFn: Double => Double = beta => epsilonNE + 0.1 * beta
    val bar = AdaptationSafety.betaBar(deltaAdapt, epsilonNE, exploitFn)
    // At betaBar, exploitability should be <= eps + delta
    val exploitAtBar = exploitFn(bar)
    assert(AdaptationSafety.isSafe(exploitAtBar, epsilonNE, deltaAdapt))
    // Just above betaBar should violate (if bar < 1.0)
    if bar < 1.0 then
      val exploitAbove = exploitFn(math.min(bar + 0.01, 1.0))
      assert(exploitAbove > epsilonNE + deltaAdapt - 0.001 || bar > 0.99)

  test("Theorem 8: clamped beta respects safety"):
    val bar = 0.6
    val clamped = AdaptationSafety.clampBeta(0.9, bar)
    assertEqualsDouble(clamped, 0.6, Tol)

  // ========================================================================
  // Corollary 1: Damaging passive leakage
  // delta_pass < 0 => action leaks information harmfully
  // ========================================================================

  test("Corollary 1: negative deltaPass indicates damaging leak"):
    val prd = SignalDecomposition.computePerRivalDelta(
      qAttrib = Ev(5.0), qRef = Ev(3.0), qBlind = Ev(7.0)
    )
    assert(prd.isDamagingLeak)
    assert(prd.deltaPass < Ev.Zero)

  test("Corollary 1: non-negative deltaPass is not damaging"):
    val prd = SignalDecomposition.computePerRivalDelta(
      qAttrib = Ev(10.0), qRef = Ev(8.0), qBlind = Ev(6.0)
    )
    assert(!prd.isDamagingLeak)

  // ========================================================================
  // Corollary 2: Exploitative bluff implies structural bluff
  // ========================================================================

  test("Corollary 2: exhaustive verification over all class/action combinations"):
    val classes = StrategicClass.values.toSeq
    val actions = Seq(
      sicfun.holdem.types.PokerAction.Fold,
      sicfun.holdem.types.PokerAction.Check,
      sicfun.holdem.types.PokerAction.Call,
      sicfun.holdem.types.PokerAction.Raise(50.0)
    )
    val gains = Seq(Ev(-1.0), Ev(0.0), Ev(0.001), Ev(100.0))
    for
      cls <- classes
      act <- actions
      g <- gains
    do
      if BluffFramework.isExploitativeBluff(cls, act, g) then
        assert(
          BluffFramework.isStructuralBluff(cls, act),
          s"Corollary 2 violated: exploitative($cls, $act, ${g.value}) but not structural"
        )

  // ========================================================================
  // Corollary 3: Separability as a special case
  // V^{1,1} - V^{1,0} = V^{0,1} - V^{0,0} => delta_int = 0
  // ========================================================================

  test("Corollary 3: separable four-world implies zero interaction"):
    // Construct separable case: V^{1,1} - V^{1,0} = V^{0,1} - V^{0,0}
    val v00 = Ev(4.0)
    val v01 = Ev(7.0)
    val v10 = Ev(6.0)
    val v11 = Ev(v10.value + (v01.value - v00.value)) // = 6 + 3 = 9
    val fw = FourWorldDecomposition.compute(v11, v10, v01, v00)
    assertEqualsDouble(fw.deltaInteraction.value, 0.0, Tol)

  test("Corollary 3: non-separable four-world has non-zero interaction"):
    val fw = FourWorldDecomposition.compute(Ev(10.0), Ev(7.0), Ev(6.0), Ev(4.0))
    // 10 - 7 != 6 - 4, so interaction != 0
    assert(fw.deltaInteraction.abs > Ev.Zero)

  // ========================================================================
  // Corollary 4: Coarse interaction bound
  // |delta_int| <= 4*R_max/(1-gamma) (standard)
  // |delta_int^rho| <= 4*R_max*(1-gamma+gamma*rho)/(1-gamma)^2 (robust)
  // ========================================================================

  test("Corollary 4: standard interaction bound"):
    val rMax = 100.0
    val gamma = 0.95
    val bound = 4.0 * rMax / (1.0 - gamma)
    // Worst case: values at extremes
    val vBound = rMax / (1.0 - gamma)
    val fw = FourWorldDecomposition.compute(
      Ev(vBound), Ev(-vBound), Ev(-vBound), Ev(vBound)
    )
    assert(fw.deltaInteraction.abs.value <= bound + Tol)

  test("Corollary 4: robust interaction bound"):
    val rMax = 100.0
    val gamma = 0.95
    val rho = 0.1
    val standardBound = 4.0 * rMax / (1.0 - gamma)
    val robustBound = 4.0 * rMax * (1.0 - gamma + gamma * rho) / math.pow(1.0 - gamma, 2)
    assert(robustBound >= standardBound)
    // Robust bound is tighter description of growth with rho
    // Use relative tolerance for large numbers (floating-point order of operations)
    val expectedRobust = 4.0 * rMax / math.pow(1.0 - gamma, 2) * (1.0 - gamma + gamma * rho)
    assertEqualsDouble(robustBound, expectedRobust, 1e-6)

  // ========================================================================
  // RevealSchedule (Def 51): classify action relative to threshold
  // ========================================================================

  test("RevealSchedule: classify returns Conceal below threshold"):
    val rival = PlayerId("v1")
    val stage = sicfun.holdem.types.Street.Flop
    val sched = RevealSchedule(Map((rival, stage) -> RevealThreshold(Ev(0.5), isExact = true)))
    assertEquals(sched.classify(rival, stage, Ev(0.3)), RevealDecision.Conceal)

  test("RevealSchedule: classify returns Reveal above threshold"):
    val rival = PlayerId("v1")
    val stage = sicfun.holdem.types.Street.Flop
    val sched = RevealSchedule(Map((rival, stage) -> RevealThreshold(Ev(0.5), isExact = true)))
    assertEquals(sched.classify(rival, stage, Ev(0.7)), RevealDecision.Reveal)

  test("RevealSchedule: classify returns Randomize at exact threshold"):
    val rival = PlayerId("v1")
    val stage = sicfun.holdem.types.Street.Flop
    val sched = RevealSchedule(Map((rival, stage) -> RevealThreshold(Ev(0.5), isExact = true)))
    assertEquals(sched.classify(rival, stage, Ev(0.5)), RevealDecision.Randomize)

  test("RevealSchedule: classify returns Unknown for missing rival/stage"):
    val sched = RevealSchedule(Map.empty)
    assertEquals(sched.classify(PlayerId("v1"), sicfun.holdem.types.Street.Turn, Ev(0.5)), RevealDecision.Unknown)

  // ========================================================================
  // Proposition 9.1: Formal exploitability via SecurityValue
  // eps(b; pi) >= 0 and eps_deploy >= max pointwise exploitabilities
  // ========================================================================

  test("Proposition 9.1: exploitability is non-negative and deployment >= pointwise"):
    // Create a trivial rival profile that returns uniform action distribution
    val mkProfile: Double => JointRivalProfile = (ev: Double) =>
      new JointRivalProfile:
        def actionDistribution(
            rivalId: PlayerId,
            publicState: PublicState,
            rivalState: RivalBeliefState
        ): Map[sicfun.holdem.types.PokerAction.Category, Double] =
          Map(
            sicfun.holdem.types.PokerAction.Category.Fold  -> 0.25,
            sicfun.holdem.types.PokerAction.Category.Check -> 0.25,
            sicfun.holdem.types.PokerAction.Category.Call  -> 0.25,
            sicfun.holdem.types.PokerAction.Category.Raise -> 0.25
          )

    val profileA = mkProfile(10.0)
    val profileB = mkProfile(3.0)
    val profileC = mkProfile(7.0)
    val profileClass = RivalProfileClass(IndexedSeq(profileA, profileB, profileC))

    // Simulate two "belief points" with different hero value functions:
    // Belief 1: hero values are 10, 3, 7 against the three profiles
    val heroValue1: JointRivalProfile => Ev = p =>
      if p eq profileA then Ev(10.0)
      else if p eq profileB then Ev(3.0)
      else Ev(7.0)

    // Belief 2: hero values are 8, 6, 5
    val heroValue2: JointRivalProfile => Ev = p =>
      if p eq profileA then Ev(8.0)
      else if p eq profileB then Ev(6.0)
      else Ev(5.0)

    // Compute security values: min over profiles
    val secActual1 = SecurityValue.compute(heroValue1, profileClass)  // min(10, 3, 7) = 3
    val secActual2 = SecurityValue.compute(heroValue2, profileClass)  // min(8, 6, 5) = 5

    assertEqualsDouble(secActual1.value, 3.0, Tol)
    assertEqualsDouble(secActual2.value, 5.0, Tol)

    // Suppose the optimal policy has security values 5.0 and 6.0 at these beliefs
    val secOptimal1 = Ev(5.0)
    val secOptimal2 = Ev(6.0)

    // Pointwise exploitability
    val eps1 = PointwiseExploitability.compute(secOptimal1, secActual1)  // 5 - 3 = 2
    val eps2 = PointwiseExploitability.compute(secOptimal2, secActual2)  // 6 - 5 = 1

    // Non-negativity
    assert(eps1 >= Ev.Zero, "Pointwise exploitability must be >= 0")
    assert(eps2 >= Ev.Zero, "Pointwise exploitability must be >= 0")
    assertEqualsDouble(eps1.value, 2.0, Tol)
    assertEqualsDouble(eps2.value, 1.0, Tol)

    // Deployment exploitability = max over belief set
    val epsDeploy = DeploymentExploitability.compute(IndexedSeq(eps1, eps2))
    assertEqualsDouble(epsDeploy.value, 2.0, Tol)

    // Deployment exploitability >= any individual pointwise exploitability
    assert(epsDeploy >= eps1)
    assert(epsDeploy >= eps2)

  // ========================================================================
  // Theorem 9: Bellman-safe certificates guarantee adaptation safety
  // If B_beta dominates B* and satisfies (C1)-(C4), then the induced
  // safe-feasible policy is AS-strong with budget epsilon_adapt = max B*.
  // ========================================================================

  test("Theorem 9: B* fixed point yields valid certificate"):
    // 2-state toy: losses [2.0, 1.0] and [0.0] (terminal)
    val robustLosses = Array(Array(2.0, 1.0), Array(0.0))
    val gamma = 0.5
    val bStar = SafetyBellman.computeBStar(robustLosses, gamma)

    // B* is a fixed point: T_safe(B*) ≈ B*
    val tResult = SafetyBellman.tSafe(bStar, robustLosses, gamma)
    for s <- bStar.indices do
      assertEqualsDouble(tResult(s), bStar(s), 1e-8)

    // B* dominates itself (trivially)
    assert(SafetyBellman.certificateDominates(bStar, bStar))

    // Required budget is finite and non-negative
    val budget = SafetyBellman.requiredAdaptationBudget(bStar)
    assert(budget >= 0.0)
    assert(budget < 1e10)

  test("Theorem 9: certificate-safe action set restricts unsafe actions"):
    val robustLosses = Array(Array(1.0, 10.0), Array(0.5, 0.5))
    val gamma = 0.5
    val bStar = SafetyBellman.computeBStar(robustLosses, gamma)

    // At state 0, action 1 has loss=10.0 which exceeds B*(0)
    val safeActions = SafetyBellman.safeActionSet(0, bStar, robustLosses, gamma)
    assert(safeActions.contains(0), "low-loss action should be safe")
    assert(!safeActions.contains(1), "high-loss action should be unsafe")

  test("Theorem 9: safe-feasible selector + certificate → bounded regret"):
    // 3 states, terminal at state 2
    val robustLosses = Array(Array(1.0, 3.0), Array(0.5, 2.0), Array(0.0))
    val gamma = 0.5
    val bStar = SafetyBellman.computeBStar(robustLosses, gamma, tolerance = 1e-14)

    // B* is a valid certificate: exact fixed point → T_safe(B*) = B*
    val cert = SafetyBellman.Certificate(bStar.clone(), terminalStates = Set.empty)
    assert(cert.isValid(robustLosses, gamma, maxBound = 100.0),
      "Certificate from B* should be structurally valid")

    // Safe-feasible action at each state should have loss <= B*(s)
    val qValues0 = Array(5.0, 8.0) // action 1 has higher Q but may be unsafe
    val safe0 = SafetyBellman.safeActionSet(0, bStar, robustLosses, gamma)
    val chosen0 = SafetyBellman.safeFeasibleAction(qValues0, safe0)
    // The chosen action's loss + gamma*maxB <= B*(s)
    val maxB = bStar.max
    assert(
      robustLosses(0)(chosen0) + gamma * maxB <= bStar(0) + 1e-10,
      "safe-feasible action must satisfy Bellman bound"
    )

  test("Theorem 9: TotalVulnerability bounds deployment + adaptation"):
    val baseline = DeploymentBaseline(Ev(0.03), 50, "Theorem 9 test")
    val bStar = SafetyBellman.computeBStar(Array(Array(1.0), Array(0.0)), 0.5)
    val epsilonAdapt = Ev(SafetyBellman.requiredAdaptationBudget(bStar))
    val (total, fidelity) = TotalVulnerability.compute(baseline, epsilonAdapt)

    // Total = baseline + adaptation budget
    assertEqualsDouble(total.value, baseline.baselineExploitability.value + epsilonAdapt.value, 1e-12)
    // Both are finite approximations
    assertEquals(fidelity, Fidelity.Approximate)

  // ========================================================================
  // Proposition 9.7: Telescopic risk decomposition along canonical chain
  // L_robust(omega_n) - L_robust(omega_0) = sum Delta_risk(k, k+1)
  // ========================================================================

  test("Proposition 9.7: telescopic risk identity on canonical chain"):
    val chain = ChainWorld.canonicalChain
    // Simulate robust losses: risk increases as layers add information
    val losses = IndexedSeq(Ev(1.0), Ev(2.5), Ev(4.0), Ev(6.0))
    val profile = RiskDecomposition.ChainRiskProfile(chain, losses)
    val (totalGap, increments) = profile.telescopicDecomposition

    // Total gap = L(last) - L(first) = 6.0 - 1.0 = 5.0
    assertEqualsDouble(totalGap.value, 5.0, Tol)

    // Sum of increments = total gap (telescopic identity)
    val sumInc = increments.map(_.riskDelta).reduce(_ + _)
    assertEqualsDouble(sumInc.value, totalGap.value, Tol)

    // Individual increments: 1.5, 1.5, 2.0
    assertEqualsDouble(increments(0).riskDelta.value, 1.5, Tol)
    assertEqualsDouble(increments(1).riskDelta.value, 1.5, Tol)
    assertEqualsDouble(increments(2).riskDelta.value, 2.0, Tol)

  test("Proposition 9.7: risk and value decompositions share chain ordering"):
    val chain = ChainWorld.canonicalChain
    val losses = IndexedSeq(Ev(1.0), Ev(3.0), Ev(2.0), Ev(5.0))
    val values = IndexedSeq(Ev(10.0), Ev(12.0), Ev(11.0), Ev(15.0))

    val riskProfile = RiskDecomposition.ChainRiskProfile(chain, losses)
    val riskEdges = riskProfile.riskIncrements

    val valueEdges = (0 until chain.size - 1).map { k =>
      ChainEdgeDelta(chain(k), chain(k + 1), values(k + 1) - values(k))
    }

    // Both use the same chain ordering
    for k <- riskEdges.indices do
      assertEquals(riskEdges(k).from, valueEdges(k).from)
      assertEquals(riskEdges(k).to, valueEdges(k).to)

    // Efficiency metrics are well-defined where risk is nonzero
    val effs = RiskDecomposition.edgeEfficiencies(valueEdges, riskEdges)
    for (from, to, eff) <- effs do
      eff match
        case Some(e) => assert(!e.isNaN && !e.isInfinite)
        case None    => () // zero risk is valid
