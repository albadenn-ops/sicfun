package sicfun.holdem.strategic.solver

import munit.FunSuite
import sicfun.holdem.HoldemPomcpNativeBindings

/** Tests for PFT-DPW POMDP solver runtime.
  *
  * Tests are split into two groups:
  *   1. Pure Scala tests (no native library required): validation of data structures,
  *      configuration defaults, and the Definition 55 error bound formula.
  *   2. Native tests (tagged "native"): require sicfun_pomcp_native.dll to be on
  *      java.library.path. Skipped automatically when the DLL is not available.
  *
  * Run pure Scala tests only:
  *   sbt "testOnly *PftDpwRuntimeTest" -- --exclude-tags=native
  *
  * Run all tests (when native DLL is available):
  *   sbt "testOnly *PftDpwRuntimeTest"
  */
class PftDpwRuntimeTest extends FunSuite:

  /* Tag for tests that require the native POMCP library. */
  val NativeTag = new munit.Tag("native")

  /* Check whether the native library is available (non-throwing probe). */
  private def nativeAvailable: Boolean =
    try
      HoldemPomcpNativeBindings.lastEngineCode()
      true
    catch case _: UnsatisfiedLinkError => false

  // =========================================================================
  // Definition 54: Particle belief validation
  // =========================================================================

  test("ParticleBelief rejects empty arrays") {
    intercept[IllegalArgumentException] {
      ParticleBelief(Array.empty[Int], Array.empty[Double])
    }
  }

  test("ParticleBelief rejects mismatched lengths") {
    intercept[IllegalArgumentException] {
      ParticleBelief(Array(0, 1), Array(0.5))
    }
  }

  test("ParticleBelief accepts matching non-empty arrays") {
    val b = ParticleBelief(Array(0, 1, 2), Array(0.5, 0.3, 0.2))
    assertEquals(b.stateIndices.length, 3)
    assertEquals(b.weights.length, 3)
  }

  // =========================================================================
  // TabularGenerativeModel validation
  // =========================================================================

  test("TabularGenerativeModel rejects wrong transition table size") {
    intercept[IllegalArgumentException] {
      TabularGenerativeModel(
        transitionTable = Array(0),    // should be 2 * 2 = 4
        obsLikelihood = Array.fill(2 * 2 * 2)(0.5),
        rewardTable = Array.fill(2 * 2)(1.0),
        numStates = 2, numActions = 2, numObs = 2
      )
    }
  }

  test("TabularGenerativeModel rejects wrong obs likelihood size") {
    intercept[IllegalArgumentException] {
      TabularGenerativeModel(
        transitionTable = Array.fill(2 * 2)(0),
        obsLikelihood = Array(0.5),    // should be 2 * 2 * 2 = 8
        rewardTable = Array.fill(2 * 2)(1.0),
        numStates = 2, numActions = 2, numObs = 2
      )
    }
  }

  test("TabularGenerativeModel rejects wrong reward table size") {
    intercept[IllegalArgumentException] {
      TabularGenerativeModel(
        transitionTable = Array.fill(2 * 2)(0),
        obsLikelihood = Array.fill(2 * 2 * 2)(0.5),
        rewardTable = Array(1.0),      // should be 2 * 2 = 4
        numStates = 2, numActions = 2, numObs = 2
      )
    }
  }

  test("TabularGenerativeModel accepts correct dimensions") {
    val model = TabularGenerativeModel(
      transitionTable = Array.fill(2 * 2)(0),
      obsLikelihood = Array.fill(2 * 2 * 2)(0.5),
      rewardTable = Array.fill(2 * 2)(1.0),
      numStates = 2, numActions = 2, numObs = 2
    )
    assertEquals(model.numStates, 2)
    assertEquals(model.numActions, 2)
    assertEquals(model.numObs, 2)
  }

  // =========================================================================
  // PftDpwConfig defaults match C++ PftDpwConfig defaults
  // =========================================================================

  test("PftDpwConfig defaults are sane and match C++ defaults") {
    val cfg = PftDpwConfig()
    assert(cfg.numSimulations > 0, "numSimulations must be positive")
    assert(cfg.gamma > 0.0 && cfg.gamma < 1.0, "gamma must be in (0,1)")
    assert(cfg.rMax > 0.0, "rMax must be positive")
    assert(cfg.ucbC >= 0.0, "ucbC must be non-negative")
    assert(cfg.kAction > 0.0, "kAction must be positive")
    assert(cfg.alphaAction > 0.0 && cfg.alphaAction <= 1.0, "alphaAction must be in (0,1]")
    assert(cfg.kObs > 0.0, "kObs must be positive")
    assert(cfg.alphaObs > 0.0 && cfg.alphaObs <= 1.0, "alphaObs must be in (0,1]")
    assert(cfg.maxDepth > 0, "maxDepth must be positive")
    // Verify exact C++ defaults
    assertEquals(cfg.numSimulations, 1000)
    assertEqualsDouble(cfg.gamma, 0.99, 1e-15)
    assertEqualsDouble(cfg.rMax, 1.0, 1e-15)
    assertEqualsDouble(cfg.ucbC, 1.0, 1e-15)
    assertEqualsDouble(cfg.kAction, 2.0, 1e-15)
    assertEqualsDouble(cfg.alphaAction, 0.5, 1e-15)
    assertEqualsDouble(cfg.kObs, 2.0, 1e-15)
    assertEqualsDouble(cfg.alphaObs, 0.5, 1e-15)
    assertEquals(cfg.maxDepth, 50)
    assertEquals(cfg.seed, 42L)
  }

  // =========================================================================
  // Definition 55: Particle error bound
  // =========================================================================

  test("Def 55: uniform particle weights give zero error bound") {
    // Uniform weights: sum(w_j^2) * C = C * (1/C)^2 * C = 1 -> D2 = ln(1) = 0 -> bound = 0
    val n = 100
    val weights = Array.fill(n)(1.0 / n)
    val bound = PftDpwRuntime.particleErrorBound(weights, rMax = 1.0, gamma = 0.99)
    assertEqualsDouble(bound, 0.0, 1e-10)
  }

  test("Def 55: concentrated particle set gives positive error bound") {
    // One particle has all weight -> maximum concentration
    // D2 = ln(1.0^2 * 5) = ln(5) -> bound > 0
    val weights = Array(1.0, 1e-15, 1e-15, 1e-15, 1e-15)
    val bound = PftDpwRuntime.particleErrorBound(weights, rMax = 1.0, gamma = 0.99)
    assert(bound > 0.0, s"concentrated belief should have positive error bound, got $bound")
  }

  test("Def 55: error bound scales linearly with R_max") {
    val weights = Array(0.9, 0.05, 0.05)
    val bound1 = PftDpwRuntime.particleErrorBound(weights, rMax = 1.0, gamma = 0.9)
    val bound2 = PftDpwRuntime.particleErrorBound(weights, rMax = 2.0, gamma = 0.9)
    assertEqualsDouble(bound2 / bound1, 2.0, 1e-10)
  }

  test("Def 55: error bound scales with 1 / (1 - gamma)") {
    val weights = Array(0.9, 0.05, 0.05)
    val bound1 = PftDpwRuntime.particleErrorBound(weights, rMax = 1.0, gamma = 0.9)
    val bound3 = PftDpwRuntime.particleErrorBound(weights, rMax = 1.0, gamma = 0.5)
    // (1/(1-0.9)) / (1/(1-0.5)) = (1/0.1) / (1/0.5) = 10 / 2 = 5
    assertEqualsDouble(bound1 / bound3, 5.0, 1e-10)
  }

  test("PftDpwResult.isSuccess returns true for status 0") {
    val r = PftDpwResult(0, Array(1.0, 0.5), Array(10, 5), 0)
    assert(r.isSuccess)
  }

  test("PftDpwResult.isSuccess returns false for non-zero status") {
    val r = PftDpwResult(-1, Array(0.0), Array(0), 200)
    assert(!r.isSuccess)
  }

  // =========================================================================
  // Native solver tests: guarded by assume(nativeAvailable)
  // =========================================================================

  // Trivial 1-state 1-action MDP: value = r / (1 - gamma) for infinite horizon,
  // truncated at maxDepth steps.
  test("Def 30-31: trivial MDP Q-value converges to sum of discounted rewards".tag(NativeTag)) {
    assume(nativeAvailable, "native library not available")

    // 1 state, 1 action, 1 observation, deterministic self-loop, reward=1
    val model = TabularGenerativeModel(
      transitionTable = Array(0),   // state 0 -> state 0
      obsLikelihood = Array(1.0),   // always observe obs 0
      rewardTable = Array(1.0),     // reward = 1.0
      numStates = 1, numActions = 1, numObs = 1
    )
    val belief = ParticleBelief(Array(0), Array(1.0))
    val cfg = PftDpwConfig(
      numSimulations = 5000, gamma = 0.9, rMax = 1.0,
      maxDepth = 20, seed = 123L
    )
    val result = PftDpwRuntime.solve(model, belief, cfg)

    assert(result.isSuccess, s"solver failed with status ${result.status}")
    assertEquals(result.bestAction, 0)

    // Expected finite-horizon value: sum_{d=0}^{19} 0.9^d = (1 - 0.9^20) / (1 - 0.9) ~= 8.784
    val expected = (1.0 - math.pow(0.9, 20)) / (1.0 - 0.9)
    assertEqualsDouble(result.qValues(0), expected, 0.5)  // within 0.5 of true
  }

  // 2-action bandit: action 0 gives reward 1, action 1 gives reward 0.5
  test("Def 32: optimal Q selects higher-reward action".tag(NativeTag)) {
    assume(nativeAvailable, "native library not available")

    val model = TabularGenerativeModel(
      transitionTable = Array(0, 0),      // both actions stay in state 0
      obsLikelihood = Array(1.0, 1.0),    // always observe obs 0
      rewardTable = Array(1.0, 0.5),      // action 0: r=1.0, action 1: r=0.5
      numStates = 1, numActions = 2, numObs = 1
    )
    val belief = ParticleBelief(Array(0), Array(1.0))
    val cfg = PftDpwConfig(
      numSimulations = 2000, gamma = 0.9, rMax = 1.0,
      maxDepth = 15, seed = 456L
    )
    val result = PftDpwRuntime.solve(model, belief, cfg)

    assert(result.isSuccess, s"solver failed with status ${result.status}")
    assertEquals(result.bestAction, 0, "should prefer action 0 with higher reward")
    assert(
      result.qValues(0) > result.qValues(1),
      s"Q(a=0)=${result.qValues(0)} should exceed Q(a=1)=${result.qValues(1)}"
    )
  }

  // 2-state POMDP with informative observations: observations reveal which state we're in
  test("Def 54: POMDP with informative observations selects state-optimal action".tag(NativeTag)) {
    assume(nativeAvailable, "native library not available")

    // State 0: action 0 is good (r=1), action 1 is bad (r=0)
    // State 1: action 1 is good (r=1), action 0 is bad (r=0)
    // Observations perfectly reveal state (obs 0 -> state 0, obs 1 -> state 1)
    // States are absorbing (self-loops)
    val model = TabularGenerativeModel(
      transitionTable = Array(0, 0, 1, 1),   // (s=0,a=0)->0, (s=0,a=1)->0, (s=1,a=0)->1, (s=1,a=1)->1
      obsLikelihood = Array(
        1.0, 0.0,   // s=0, a=0: see obs 0 with P=1
        1.0, 0.0,   // s=0, a=1: see obs 0 with P=1
        0.0, 1.0,   // s=1, a=0: see obs 1 with P=1
        0.0, 1.0    // s=1, a=1: see obs 1 with P=1
      ),
      rewardTable = Array(1.0, 0.0, 0.0, 1.0), // (s=0,a=0)->1, (s=0,a=1)->0, (s=1,a=0)->0, (s=1,a=1)->1
      numStates = 2, numActions = 2, numObs = 2
    )

    // Belief concentrated in state 0 (4 identical particles)
    val belief0 = ParticleBelief(Array(0, 0, 0, 0), Array(0.25, 0.25, 0.25, 0.25))
    val cfg = PftDpwConfig(numSimulations = 1000, gamma = 0.9, maxDepth = 10, seed = 789L)
    val result0 = PftDpwRuntime.solve(model, belief0, cfg)

    assert(result0.isSuccess, s"solver failed with status ${result0.status}")
    assertEquals(result0.bestAction, 0, "in state 0, action 0 is optimal")

    // Belief concentrated in state 1 (4 identical particles)
    val belief1 = ParticleBelief(Array(1, 1, 1, 1), Array(0.25, 0.25, 0.25, 0.25))
    val result1 = PftDpwRuntime.solve(model, belief1, cfg)

    assert(result1.isSuccess, s"solver failed with status ${result1.status}")
    assertEquals(result1.bestAction, 1, "in state 1, action 1 is optimal")
  }

  // Config rejection: gamma = 1.0 is invalid (kStatusInvalidConfig = 200)
  test("config with gamma=1 is rejected by native solver".tag(NativeTag)) {
    assume(nativeAvailable, "native library not available")

    val model = TabularGenerativeModel(
      Array(0), Array(1.0), Array(1.0), 1, 1, 1
    )
    val belief = ParticleBelief(Array(0), Array(1.0))
    val cfg = PftDpwConfig(gamma = 1.0)   // invalid: must be in (0, 1)
    val result = PftDpwRuntime.solve(model, belief, cfg)

    assert(!result.isSuccess, s"gamma=1.0 should be rejected, got status ${result.status}")
  }

  // lastEngineCode() == 3 after a successful solve
  test("lastEngineCode returns 3 (PFT-DPW CPU) after successful solve".tag(NativeTag)) {
    assume(nativeAvailable, "native library not available")

    val model = TabularGenerativeModel(
      Array(0), Array(1.0), Array(1.0), 1, 1, 1
    )
    val belief = ParticleBelief(Array(0), Array(1.0))
    val cfg = PftDpwConfig(numSimulations = 10, maxDepth = 5)
    val result = PftDpwRuntime.solve(model, belief, cfg)

    assert(result.isSuccess)
    assertEquals(HoldemPomcpNativeBindings.lastEngineCode(), 3)
  }
