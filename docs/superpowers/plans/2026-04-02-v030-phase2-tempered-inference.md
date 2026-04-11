# Phase 2: Tempered Inference -- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace v0.29.1's flat epsilon-smoothing with the two-layer tempered likelihood from SICFUN-v0.30.2 Defs 15, 15A, 15B (power-posterior exponent kappa_temp + additive safety floor delta_floor), with a pure Scala reference implementation, Theorem 1 validation, backward-compatibility proof, and C++/CUDA native integration.

**Architecture:** A new `TemperedLikelihood` object in `sicfun.holdem.strategic` provides the pure Scala reference implementation (Defs 15, 15A, 15B) and the `TemperedConfig` ADT for configuration. The existing C++ update core (`BayesNativeUpdateCore.hpp`) gains a new `update_posterior_tempered_raw` function that applies `pow(likelihood, kappa_temp) + delta_floor * eta` before the posterior multiplication. JNI signatures gain three extra parameters (`kappaTemp`, `deltaFloor`, `eta` array). The Scala provider and native runtime pass these through. When `kappaTemp=1.0` and `deltaFloor` equals legacy epsilon, the legacy `(1-eps)*Pr + eps*eta` form is used for exact backward compatibility.

**Tech Stack:** Scala 3.8.1, C++17, CUDA 11.8, munit 1.2.2

**Depends on:** Phase 1 (TemperedConfig references strategic types from `sicfun.holdem.strategic`)

**Unlocks:** Phase 4 (exploitation interpolation uses tempered posterior output)

---

## Existing Code Summary

Before modifying anything, understand these files:

| File | Role | Key detail |
|---|---|---|
| `src/main/native/jni/BayesNativeUpdateCore.hpp` | Header-only C++ Bayesian engine | `update_posterior_raw()` uses 4-wide unrolled loops: `posterior[h] *= row[h]` then normalize. The `eps = 1e-12` constant is the zero-evidence guard, NOT the smoothing epsilon. |
| `src/main/native/jni/HoldemBayesNativeCpuBindings.cpp` | CPU JNI entry point | Acquires critical arrays, calls `update_posterior_raw()`, reports engine code 1. |
| `src/main/native/jni/HoldemBayesNativeGpuBindings.cu` | GPU JNI entry point | Structurally identical to CPU binding, reports engine code 2. Compiled by nvcc into `sicfun_gpu_kernel.dll`. |
| `src/main/java/sicfun/holdem/HoldemBayesNativeCpuBindings.java` | Java JNI class | `static native int updatePosterior(int, int, double[], double[], double[], double[])` |
| `src/main/java/sicfun/holdem/HoldemBayesNativeGpuBindings.java` | Java JNI class | Same signature as CPU bindings. |
| `src/main/scala/sicfun/holdem/gpu/HoldemBayesNativeRuntime.scala` | Scala native wrapper | `updatePosteriorInPlace()` dispatches to CPU or GPU JNI class. |
| `src/main/scala/sicfun/holdem/provider/HoldemBayesProvider.scala` | Scala provider/dispatcher | Builds likelihood matrix, dispatches to Scala/NativeCpu/NativeGpu, runs shadow validation. `MinLikelihood = 1e-6` clamps likelihoods. |
| `src/main/native/build-windows-llvm.ps1` | CPU DLL build script | Compiles `HoldemBayesNativeCpuBindings.cpp` to `sicfun_bayes_native.dll` with clang++ -O3 -std=c++17. |
| `src/main/native/build-windows-cuda11.ps1` | GPU DLL build script | Compiles `HoldemBayesNativeGpuBindings.cu` to `sicfun_bayes_cuda.dll` with nvcc -O3 -std=c++17. |

---

## Math Reference (from canonical spec)

### Def 15: Configuration semantics

Three modes, declared at configuration time:
- **Two-layer tempered (default):** kappa_temp in (0,1], delta_floor > 0
- **Pure power-posterior:** kappa_temp in (0,1], delta_floor = 0
- **Legacy epsilon-smoothing:** kappa_temp = 1, delta_floor = epsilon. Uses legacy form `(1-eps)*Pr + eps*eta`, NOT the two-layer formula with kappa=1.

### Def 15A: Two-layer tempered likelihood

```
L_{kappa,delta}(y | c, ...) = Pr(y | c, ...)^kappa_temp + delta_floor * eta(y)
```

Where eta is a full-support distribution over action signals (uniform by default).

### Def 15B: Posterior-on-class update

```
mu_{t+1}(c) = L(y|c) * mu_t(c) / sum_c' L(y|c') * mu_t(c')
```

When delta_floor > 0, denominator is always positive. When delta_floor = 0 and denominator = 0, preserve prior.

### Theorem 1: Unconditional totality

If delta_floor > 0 and eta has full support, then for every prior with non-empty support and every observation y, the posterior is well-defined and belongs to Delta(C^S).

### Backward compatibility (v0.30.2 Section 12)

The legacy form `L_legacy(y|c) = (1-epsilon)*Pr(y|c) + epsilon*eta(y)` is a SEPARATE formula. Setting kappa=1, delta=epsilon in the two-layer formula gives `Pr(y|c)^1 + epsilon*eta(y) = Pr(y|c) + epsilon*eta(y)`, which is NOT the same as `(1-epsilon)*Pr(y|c) + epsilon*eta(y)`. Therefore the implementation MUST detect the legacy configuration and dispatch to the legacy formula directly.

---

## Task Breakdown

### Task 1: TemperedLikelihoodTest -- failing tests for Defs 15, 15A, 15B and Theorem 1

**Goal:** Write the complete test file. All tests will fail because `TemperedLikelihood.scala` does not exist yet.

- [ ] **Step 1: Create the test file**

Create: `src/test/scala/sicfun/holdem/strategic/TemperedLikelihoodTest.scala`

```scala
package sicfun.holdem.strategic

import sicfun.holdem.strategic.TemperedLikelihood.*

class TemperedLikelihoodTest extends munit.FunSuite:

  // ==================== Def 15: TemperedConfig ====================

  test("Def 15: TemperedConfig.twoLayer validates kappa_temp in (0,1]"):
    // Valid
    TemperedConfig.twoLayer(kappaTemp = 0.9, deltaFloor = 1e-8)
    TemperedConfig.twoLayer(kappaTemp = 1.0, deltaFloor = 1e-8)
    TemperedConfig.twoLayer(kappaTemp = 0.01, deltaFloor = 1e-8)
    // Invalid
    intercept[IllegalArgumentException]:
      TemperedConfig.twoLayer(kappaTemp = 0.0, deltaFloor = 1e-8)
    intercept[IllegalArgumentException]:
      TemperedConfig.twoLayer(kappaTemp = -0.1, deltaFloor = 1e-8)
    intercept[IllegalArgumentException]:
      TemperedConfig.twoLayer(kappaTemp = 1.01, deltaFloor = 1e-8)
    intercept[IllegalArgumentException]:
      TemperedConfig.twoLayer(kappaTemp = Double.NaN, deltaFloor = 1e-8)

  test("Def 15: TemperedConfig.twoLayer validates delta_floor >= 0"):
    TemperedConfig.twoLayer(kappaTemp = 0.9, deltaFloor = 0.0)
    TemperedConfig.twoLayer(kappaTemp = 0.9, deltaFloor = 1e-8)
    intercept[IllegalArgumentException]:
      TemperedConfig.twoLayer(kappaTemp = 0.9, deltaFloor = -1e-8)
    intercept[IllegalArgumentException]:
      TemperedConfig.twoLayer(kappaTemp = 0.9, deltaFloor = Double.NaN)

  test("Def 15: TemperedConfig.purePowerPosterior sets deltaFloor = 0"):
    val cfg = TemperedConfig.purePowerPosterior(kappaTemp = 0.85)
    assertEquals(cfg.deltaFloor, 0.0)
    assertEquals(cfg.kappaTemp, 0.85)
    assert(!cfg.isLegacy)

  test("Def 15: TemperedConfig.legacy captures epsilon"):
    val cfg = TemperedConfig.legacy(epsilon = 1e-6)
    assert(cfg.isLegacy)
    assertEquals(cfg.epsilon, 1e-6)
    intercept[IllegalArgumentException]:
      TemperedConfig.legacy(epsilon = 0.0)
    intercept[IllegalArgumentException]:
      TemperedConfig.legacy(epsilon = -1e-6)
    intercept[IllegalArgumentException]:
      TemperedConfig.legacy(epsilon = 1.0)

  // ==================== Def 15A: Two-layer tempered likelihood ====================

  test("Def 15A: two-layer formula L = Pr^kappa + delta*eta"):
    val basePr = Array(0.6, 0.3, 0.1) // P(y | c) for 3 classes
    val eta = Array(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0) // uniform
    val kappa = 0.9
    val delta = 1e-4
    val result = TemperedLikelihood.computeLikelihoods(basePr, kappa, delta, eta)
    var i = 0
    while i < 3 do
      val expected = math.pow(basePr(i), kappa) + delta * eta(i)
      assertEquals(result(i), expected, 1e-15, s"class $i")
      i += 1

  test("Def 15A: kappa=1 gives Pr + delta*eta (NOT legacy form)"):
    val basePr = Array(0.5, 0.3, 0.2)
    val eta = Array(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    val result = TemperedLikelihood.computeLikelihoods(basePr, 1.0, 0.01, eta)
    // Two-layer with kappa=1: Pr(y|c)^1 + 0.01 * eta(y) = Pr(y|c) + 0.01/3
    var i = 0
    while i < 3 do
      val expected = basePr(i) + 0.01 / 3.0
      assertEquals(result(i), expected, 1e-15)
      i += 1

  test("Def 15A: property 1 -- unconditional totality when delta > 0"):
    val basePr = Array(0.0, 0.0, 0.0) // all-zero base probability
    val eta = Array(0.5, 0.3, 0.2)
    val delta = 1e-8
    val result = TemperedLikelihood.computeLikelihoods(basePr, 0.9, delta, eta)
    var i = 0
    while i < 3 do
      assert(result(i) > 0.0, s"class $i should be positive even with zero base Pr")
      assertEquals(result(i), delta * eta(i), 1e-20, s"class $i should equal delta*eta")
      i += 1

  test("Def 15A: property 2 -- likelihood ordering preservation"):
    // If Pr(y|c1) > Pr(y|c2), then L(y|c1) > L(y|c2) for any kappa in (0,1], delta >= 0
    val basePr = Array(0.7, 0.2)
    val eta = Array(0.5, 0.5)
    for kappa <- Seq(0.1, 0.5, 0.85, 0.95, 1.0) do
      for delta <- Seq(0.0, 1e-8, 0.01, 0.1) do
        val result = TemperedLikelihood.computeLikelihoods(basePr, kappa, delta, eta)
        assert(
          result(0) > result(1),
          s"ordering violated: L(c1)=${result(0)} <= L(c2)=${result(1)} at kappa=$kappa delta=$delta"
        )

  test("Def 15A: property 3 -- tempering attenuates extreme ratios"):
    val basePr = Array(0.99, 0.01)
    val eta = Array(0.5, 0.5)
    val rawRatio = basePr(0) / basePr(1) // 99.0
    val tempered = TemperedLikelihood.computeLikelihoods(basePr, 0.5, 0.0, eta)
    val temperedRatio = tempered(0) / tempered(1)
    assert(
      temperedRatio < rawRatio,
      s"tempered ratio $temperedRatio should be less than raw ratio $rawRatio"
    )

  // ==================== Def 15A: Legacy formula ====================

  test("Def 15A: legacy formula L = (1-eps)*Pr + eps*eta"):
    val basePr = Array(0.6, 0.3, 0.1)
    val eta = Array(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    val eps = 0.01
    val result = TemperedLikelihood.computeLikelihoodsLegacy(basePr, eps, eta)
    var i = 0
    while i < 3 do
      val expected = (1.0 - eps) * basePr(i) + eps * eta(i)
      assertEquals(result(i), expected, 1e-15)
      i += 1

  // ==================== Def 15B: Posterior update ====================

  test("Def 15B: posterior update with two-layer tempered likelihood"):
    val prior = Array(0.5, 0.3, 0.2)
    val basePr = Array(0.8, 0.15, 0.05) // Pr(y | c) for each class
    val eta = Array(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    val cfg = TemperedConfig.twoLayer(kappaTemp = 0.9, deltaFloor = 1e-8)
    val posterior = TemperedLikelihood.updatePosterior(prior, basePr, eta, cfg)
    // Verify it's a proper distribution
    val sum = posterior.sum
    assertEquals(sum, 1.0, 1e-12, "posterior must sum to 1")
    // All entries positive
    var i = 0
    while i < posterior.length do
      assert(posterior(i) > 0.0, s"posterior($i) must be positive")
      i += 1
    // Highest base Pr should get highest posterior (ordering preserved)
    assert(posterior(0) > posterior(1))
    assert(posterior(1) > posterior(2))

  test("Def 15B: posterior update with legacy config"):
    val prior = Array(0.5, 0.3, 0.2)
    val basePr = Array(0.8, 0.15, 0.05)
    val eta = Array(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    val cfg = TemperedConfig.legacy(epsilon = 1e-6)
    val posterior = TemperedLikelihood.updatePosterior(prior, basePr, eta, cfg)
    val sum = posterior.sum
    assertEquals(sum, 1.0, 1e-12)
    assert(posterior(0) > posterior(1))

  test("Def 15B: prior preservation when delta=0 and denominator=0"):
    val prior = Array(0.5, 0.3, 0.2)
    val basePr = Array(0.0, 0.0, 0.0) // all-zero likelihoods
    val eta = Array(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    val cfg = TemperedConfig.purePowerPosterior(kappaTemp = 0.9)
    val posterior = TemperedLikelihood.updatePosterior(prior, basePr, eta, cfg)
    // Prior should be preserved
    var i = 0
    while i < 3 do
      assertEquals(posterior(i), prior(i) / prior.sum, 1e-12, s"prior preservation at $i")
      i += 1

  // ==================== Theorem 1: Unconditional totality ====================

  test("Theorem 1: posterior is well-defined for all priors when delta > 0"):
    val eta = Array(0.25, 0.25, 0.25, 0.25)
    val cfg = TemperedConfig.twoLayer(kappaTemp = 0.85, deltaFloor = 1e-8)
    // Test with various priors including degenerate ones
    val priors = Seq(
      Array(1.0, 0.0, 0.0, 0.0),        // degenerate on class 0
      Array(0.0, 0.0, 0.0, 1.0),        // degenerate on class 3
      Array(0.25, 0.25, 0.25, 0.25),    // uniform
      Array(0.99, 0.005, 0.003, 0.002), // near-degenerate
      Array(1e-10, 1e-10, 1e-10, 1.0),  // near-zero entries
    )
    val basePrs = Seq(
      Array(0.0, 0.0, 0.0, 0.0),        // all zero
      Array(1.0, 0.0, 0.0, 0.0),        // degenerate
      Array(0.5, 0.3, 0.15, 0.05),      // typical
      Array(1e-20, 1e-20, 1e-20, 1e-20), // near-zero
    )
    for prior <- priors do
      for basePr <- basePrs do
        val posterior = TemperedLikelihood.updatePosterior(prior, basePr, eta, cfg)
        val sum = posterior.sum
        assertEquals(sum, 1.0, 1e-10, s"posterior must sum to 1 for prior=${prior.mkString(",")}")
        var i = 0
        while i < posterior.length do
          assert(
            posterior(i).isFinite && posterior(i) >= 0.0,
            s"posterior($i)=${posterior(i)} must be finite and non-negative"
          )
          i += 1

  test("Theorem 1: denominator is strictly positive when delta > 0 and eta full-support"):
    val eta = Array(0.2, 0.3, 0.5)
    val cfg = TemperedConfig.twoLayer(kappaTemp = 0.5, deltaFloor = 1e-10)
    // Even with all-zero base probabilities, the denominator must be > 0
    val prior = Array(0.4, 0.4, 0.2)
    val basePr = Array(0.0, 0.0, 0.0)
    val likelihoods = TemperedLikelihood.computeLikelihoods(basePr, cfg.kappaTemp, cfg.deltaFloor, eta)
    var denom = 0.0
    var i = 0
    while i < 3 do
      denom += likelihoods(i) * (prior(i) / prior.sum)
      i += 1
    assert(denom > 0.0, s"denominator=$denom must be strictly positive")

  // ==================== Backward compatibility ====================

  test("Backward compat: legacy config produces identical result to v0.29.1 epsilon-smoothing"):
    // The v0.29.1 formula: L(y|c) = (1-eps)*Pr(y|c) + eps*eta(y)
    val prior = Array(0.4, 0.35, 0.25)
    val basePr = Array(0.7, 0.2, 0.1)
    val eta = Array(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    val eps = 1e-6
    // Legacy path
    val legacyCfg = TemperedConfig.legacy(epsilon = eps)
    val legacyPosterior = TemperedLikelihood.updatePosterior(prior, basePr, eta, legacyCfg)
    // Manual v0.29.1 computation
    val normalized = prior.map(_ / prior.sum)
    val legacyLikelihoods = basePr.map(p => (1.0 - eps) * p + eps / 3.0)
    val unnorm = Array.tabulate(3)(i => normalized(i) * legacyLikelihoods(i))
    val evidence = unnorm.sum
    val expected = unnorm.map(_ / evidence)
    var i = 0
    while i < 3 do
      assertEquals(
        legacyPosterior(i), expected(i), 1e-15,
        s"legacy posterior($i) must match v0.29.1 exactly"
      )
      i += 1

  test("Backward compat: kappa=1 delta=eps is NOT the same as legacy"):
    // This test proves WHY we need a separate legacy code path
    val basePr = Array(0.7, 0.2, 0.1)
    val eta = Array(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    val eps = 0.01
    val twoLayerLikelihoods = TemperedLikelihood.computeLikelihoods(basePr, 1.0, eps, eta)
    val legacyLikelihoods = TemperedLikelihood.computeLikelihoodsLegacy(basePr, eps, eta)
    // Two-layer: Pr + eps*eta = 0.7 + 0.01/3 != (1-0.01)*0.7 + 0.01/3 = 0.693 + 0.00333
    assert(
      math.abs(twoLayerLikelihoods(0) - legacyLikelihoods(0)) > 1e-6,
      "two-layer with kappa=1 must differ from legacy form"
    )

  // ==================== Default eta ====================

  test("defaultEta produces uniform distribution"):
    val eta = TemperedLikelihood.defaultEta(5)
    assertEquals(eta.length, 5)
    var i = 0
    while i < 5 do
      assertEquals(eta(i), 0.2, 1e-15)
      i += 1

  // ==================== Edge cases ====================

  test("single-class posterior is always 1.0"):
    val prior = Array(1.0)
    val basePr = Array(0.5)
    val eta = Array(1.0)
    val cfg = TemperedConfig.twoLayer(kappaTemp = 0.9, deltaFloor = 1e-8)
    val posterior = TemperedLikelihood.updatePosterior(prior, basePr, eta, cfg)
    assertEquals(posterior(0), 1.0, 1e-15)

  test("two-class extreme tempering (kappa near 0) flattens likelihood ratio"):
    val basePr = Array(0.99, 0.01)
    val eta = Array(0.5, 0.5)
    val result = TemperedLikelihood.computeLikelihoods(basePr, 0.01, 0.0, eta)
    // 0.99^0.01 ~ 0.99990, 0.01^0.01 ~ 0.95499
    // Ratio should be very close to 1
    val ratio = result(0) / result(1)
    assert(ratio < 2.0, s"extreme tempering should flatten ratio to near 1, got $ratio")
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```
sbt "testOnly sicfun.holdem.strategic.TemperedLikelihoodTest"
```

Expected: compilation failure -- `TemperedLikelihood` does not exist.

---

### Task 2: TemperedLikelihood.scala -- pure Scala reference implementation

**Goal:** Implement Defs 15, 15A, 15B as a pure Scala object in `sicfun.holdem.strategic`.

- [ ] **Step 1: Verify directory exists, create if needed**

If `sicfun.holdem.strategic` package does not exist yet (Phase 1 not yet implemented), create the directories:

```
mkdir -p src/main/scala/sicfun/holdem/strategic
mkdir -p src/test/scala/sicfun/holdem/strategic
```

- [ ] **Step 2: Create the implementation file**

Create: `src/main/scala/sicfun/holdem/strategic/TemperedLikelihood.scala`

```scala
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
```

- [ ] **Step 3: Run tests to verify all pass**

```
sbt "testOnly sicfun.holdem.strategic.TemperedLikelihoodTest"
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```
git add src/main/scala/sicfun/holdem/strategic/TemperedLikelihood.scala \
        src/test/scala/sicfun/holdem/strategic/TemperedLikelihoodTest.scala
git commit -m "feat(strategic): add TemperedLikelihood reference impl (Defs 15, 15A, 15B) with Theorem 1 tests"
```

---

### Task 3: Modify BayesNativeUpdateCore.hpp -- add tempered update function

**Goal:** Add a new `update_posterior_tempered_raw` function to the C++ header that applies `pow(likelihood, kappa_temp) + delta_floor * eta[h]` to each likelihood value before the multiplication step. The existing `update_posterior_raw` is left unchanged for backward compatibility.

- [ ] **Step 1: Add the tempered update function to BayesNativeUpdateCore.hpp**

Insert the following AFTER the existing `update_posterior_raw` function (after line 227, before `update_posterior`) and BEFORE the `#undef BAYESNATIVE_RESTRICT` at the end. The existing `update_posterior_raw` and `update_posterior` functions are NOT modified -- they remain for callers that don't need tempering.

Add this block between `update_posterior_raw` (ends at line 227) and the existing `update_posterior` convenience wrapper (starts at line 234):

```cpp
/*
 * Tempered Bayesian posterior update (SICFUN v0.30.2 Def 15A/15B).
 *
 * Applies two-layer tempered likelihood before the standard Bayesian update:
 *   tempered_likelihood = pow(raw_likelihood, kappa_temp) + delta_floor * eta[h]
 *
 * When kappa_temp == 1.0 and delta_floor == 0.0, this reduces to the standard
 * update_posterior_raw (no pow, no additive floor).
 *
 * Legacy mode (use_legacy_form = true):
 *   tempered_likelihood = (1 - delta_floor) * raw_likelihood + delta_floor * eta[h]
 *   This recovers v0.29.1 epsilon-smoothing exactly.
 *
 * Parameters:
 *   observation_count  -- Number of sequential observations (rows in likelihood matrix).
 *   hypothesis_count   -- Number of hypotheses (columns).
 *   prior              -- Prior distribution array, length = hypothesis_count.
 *   likelihoods        -- Row-major likelihood matrix, length = observation_count * hypothesis_count.
 *   kappa_temp         -- Power-posterior exponent, in (0, 1].
 *   delta_floor        -- Additive safety floor, >= 0.
 *   eta                -- Full-support distribution over hypotheses, length = hypothesis_count.
 *                         If nullptr, uniform 1/hypothesis_count is used.
 *   use_legacy_form    -- If true, use legacy formula (1-eps)*L + eps*eta.
 *   posterior          -- Output buffer, length = hypothesis_count.
 *   out_log_evidence   -- Output scalar for cumulative log-evidence.
 *
 * Returns: kStatusOk on success, or an error status code.
 */
inline int update_posterior_tempered_raw(
    const int observation_count,
    const int hypothesis_count,
    const double* BAYESNATIVE_RESTRICT prior,
    const double* BAYESNATIVE_RESTRICT likelihoods,
    const double kappa_temp,
    const double delta_floor,
    const double* BAYESNATIVE_RESTRICT eta,
    const bool use_legacy_form,
    double* BAYESNATIVE_RESTRICT posterior,
    double* out_log_evidence) {
  if (!valid_length_product(observation_count, hypothesis_count)) {
    return kStatusInvalidConfig;
  }

  if (prior == nullptr || likelihoods == nullptr || posterior == nullptr || out_log_evidence == nullptr) {
    return kStatusNullArray;
  }

  /* Validate tempering parameters. */
  if (!(kappa_temp > 0.0 && kappa_temp <= 1.0)) {
    return kStatusInvalidConfig;
  }
  if (!(delta_floor >= 0.0)) {
    return kStatusInvalidConfig;
  }

  const double eps = 1e-12;

  /* Precompute default eta = 1/hypothesis_count if eta is null. */
  const double default_eta_val = 1.0 / static_cast<double>(hypothesis_count);

  /* Fast path: when kappa_temp == 1.0 and delta_floor == 0.0 and not legacy,
   * delegate to the original untampered function (no pow, no additive floor). */
  if (kappa_temp == 1.0 && delta_floor == 0.0 && !use_legacy_form) {
    return update_posterior_raw(
        observation_count, hypothesis_count, prior, likelihoods, posterior, out_log_evidence);
  }

  /* Step 1: Validate prior values and copy into posterior buffer. */
  double prior_sum = 0.0;
  int hypothesis = 0;
  for (; hypothesis + 3 < hypothesis_count; hypothesis += 4) {
    const double value0 = prior[hypothesis];
    const double value1 = prior[hypothesis + 1];
    const double value2 = prior[hypothesis + 2];
    const double value3 = prior[hypothesis + 3];
    if (!std::isfinite(value0) || value0 < 0.0 || !std::isfinite(value1) || value1 < 0.0 ||
        !std::isfinite(value2) || value2 < 0.0 || !std::isfinite(value3) || value3 < 0.0) {
      return kStatusInvalidPrior;
    }
    posterior[hypothesis] = value0;
    posterior[hypothesis + 1] = value1;
    posterior[hypothesis + 2] = value2;
    posterior[hypothesis + 3] = value3;
    prior_sum += (value0 + value1) + (value2 + value3);
  }
  for (; hypothesis < hypothesis_count; ++hypothesis) {
    const double value = prior[hypothesis];
    if (!std::isfinite(value) || value < 0.0) {
      return kStatusInvalidPrior;
    }
    posterior[hypothesis] = value;
    prior_sum += value;
  }

  if (!(prior_sum > 0.0)) {
    return kStatusInvalidPrior;
  }

  /* Step 2: Normalize prior. */
  const double inv_prior_sum = 1.0 / prior_sum;
  hypothesis = 0;
  for (; hypothesis + 3 < hypothesis_count; hypothesis += 4) {
    posterior[hypothesis] *= inv_prior_sum;
    posterior[hypothesis + 1] *= inv_prior_sum;
    posterior[hypothesis + 2] *= inv_prior_sum;
    posterior[hypothesis + 3] *= inv_prior_sum;
  }
  for (; hypothesis < hypothesis_count; ++hypothesis) {
    posterior[hypothesis] *= inv_prior_sum;
  }

  /* Step 3: Sequential Bayesian update with tempered likelihoods. */
  double log_evidence = 0.0;
  for (int obs = 0; obs < observation_count; ++obs) {
    const double* row =
        likelihoods + static_cast<long long>(obs) * static_cast<long long>(hypothesis_count);
    double evidence = 0.0;

    hypothesis = 0;
    for (; hypothesis + 3 < hypothesis_count; hypothesis += 4) {
      /* Apply tempering to each of 4 likelihoods. */
      double tempered0, tempered1, tempered2, tempered3;
      const double raw0 = row[hypothesis];
      const double raw1 = row[hypothesis + 1];
      const double raw2 = row[hypothesis + 2];
      const double raw3 = row[hypothesis + 3];
      const double eta0 = (eta != nullptr) ? eta[hypothesis]     : default_eta_val;
      const double eta1 = (eta != nullptr) ? eta[hypothesis + 1] : default_eta_val;
      const double eta2 = (eta != nullptr) ? eta[hypothesis + 2] : default_eta_val;
      const double eta3 = (eta != nullptr) ? eta[hypothesis + 3] : default_eta_val;

      if (use_legacy_form) {
        /* Legacy: (1 - delta_floor) * raw + delta_floor * eta */
        const double one_minus_delta = 1.0 - delta_floor;
        tempered0 = one_minus_delta * raw0 + delta_floor * eta0;
        tempered1 = one_minus_delta * raw1 + delta_floor * eta1;
        tempered2 = one_minus_delta * raw2 + delta_floor * eta2;
        tempered3 = one_minus_delta * raw3 + delta_floor * eta3;
      } else if (kappa_temp == 1.0) {
        /* Two-layer with kappa=1: raw + delta * eta (no pow). */
        tempered0 = raw0 + delta_floor * eta0;
        tempered1 = raw1 + delta_floor * eta1;
        tempered2 = raw2 + delta_floor * eta2;
        tempered3 = raw3 + delta_floor * eta3;
      } else {
        /* Full two-layer: pow(raw, kappa) + delta * eta. */
        tempered0 = std::pow(std::max(0.0, raw0), kappa_temp) + delta_floor * eta0;
        tempered1 = std::pow(std::max(0.0, raw1), kappa_temp) + delta_floor * eta1;
        tempered2 = std::pow(std::max(0.0, raw2), kappa_temp) + delta_floor * eta2;
        tempered3 = std::pow(std::max(0.0, raw3), kappa_temp) + delta_floor * eta3;
      }

      const double updated0 = posterior[hypothesis]     * tempered0;
      const double updated1 = posterior[hypothesis + 1] * tempered1;
      const double updated2 = posterior[hypothesis + 2] * tempered2;
      const double updated3 = posterior[hypothesis + 3] * tempered3;
      posterior[hypothesis]     = updated0;
      posterior[hypothesis + 1] = updated1;
      posterior[hypothesis + 2] = updated2;
      posterior[hypothesis + 3] = updated3;
      evidence += (updated0 + updated1) + (updated2 + updated3);
    }
    for (; hypothesis < hypothesis_count; ++hypothesis) {
      const double raw = row[hypothesis];
      const double eta_val = (eta != nullptr) ? eta[hypothesis] : default_eta_val;
      double tempered;
      if (use_legacy_form) {
        tempered = (1.0 - delta_floor) * raw + delta_floor * eta_val;
      } else if (kappa_temp == 1.0) {
        tempered = raw + delta_floor * eta_val;
      } else {
        tempered = std::pow(std::max(0.0, raw), kappa_temp) + delta_floor * eta_val;
      }
      const double updated = posterior[hypothesis] * tempered;
      posterior[hypothesis] = updated;
      evidence += updated;
    }

    if (!std::isfinite(evidence)) {
      return kStatusInvalidLikelihood;
    }

    /* Def 15B: when delta=0 and evidence=0, preserve prior. */
    if (!(evidence > eps)) {
      if (delta_floor == 0.0) {
        /* Preserve the current posterior (which at this point is the
         * normalized prior if obs==0, or the last valid posterior). */
        log_evidence += std::log(eps);
        continue;
      }
      return kStatusZeroEvidence;
    }

    const double inv_evidence = 1.0 / evidence;
    hypothesis = 0;
    for (; hypothesis + 3 < hypothesis_count; hypothesis += 4) {
      posterior[hypothesis]     *= inv_evidence;
      posterior[hypothesis + 1] *= inv_evidence;
      posterior[hypothesis + 2] *= inv_evidence;
      posterior[hypothesis + 3] *= inv_evidence;
    }
    for (; hypothesis < hypothesis_count; ++hypothesis) {
      posterior[hypothesis] *= inv_evidence;
    }
    log_evidence += std::log(evidence);
  }

  *out_log_evidence = log_evidence;
  return kStatusOk;
}

/*
 * Convenience wrapper for update_posterior_tempered_raw using std::vector containers.
 */
inline int update_posterior_tempered(const UpdateSpec& spec, const double kappa_temp,
    const double delta_floor, const std::vector<double>& eta,
    const bool use_legacy_form, UpdateOutput& output) {
  if (!valid_length_product(spec.observation_count, spec.hypothesis_count)) {
    return kStatusInvalidConfig;
  }
  if (static_cast<int>(spec.prior.size()) != spec.hypothesis_count) {
    return kStatusLengthMismatch;
  }
  const int expected_likelihood_size = spec.observation_count * spec.hypothesis_count;
  if (static_cast<int>(spec.likelihoods.size()) != expected_likelihood_size) {
    return kStatusLengthMismatch;
  }
  if (!eta.empty() && static_cast<int>(eta.size()) != spec.hypothesis_count) {
    return kStatusLengthMismatch;
  }
  output.posterior.resize(static_cast<size_t>(spec.hypothesis_count));
  return update_posterior_tempered_raw(
      spec.observation_count,
      spec.hypothesis_count,
      spec.prior.data(),
      spec.likelihoods.data(),
      kappa_temp,
      delta_floor,
      eta.empty() ? nullptr : eta.data(),
      use_legacy_form,
      output.posterior.data(),
      &output.log_evidence);
}
```

- [ ] **Step 2: Verify the file compiles**

```powershell
cd C:\Users\alexl\code\math\untitled\src\main\native
& "C:\Program Files\LLVM\bin\clang++.exe" -std=c++17 -O3 -DNDEBUG -fsyntax-only `
  "-I$env:JAVA_HOME/include" "-I$env:JAVA_HOME/include/win32" `
  jni/HoldemBayesNativeCpuBindings.cpp
```

Expected: clean compilation, no errors.

- [ ] **Step 3: Commit**

```
git add src/main/native/jni/BayesNativeUpdateCore.hpp
git commit -m "feat(native): add update_posterior_tempered_raw to BayesNativeUpdateCore (Defs 15A/15B)"
```

---

### Task 4: Add tempered JNI entry points to Java binding classes

**Goal:** Add `updatePosteriorTempered` native methods to both Java JNI classes. The new methods accept three extra parameters: `kappaTemp`, `deltaFloor`, and an `eta` array, plus a `useLegacyForm` boolean.

- [ ] **Step 1: Modify HoldemBayesNativeCpuBindings.java**

File: `src/main/java/sicfun/holdem/HoldemBayesNativeCpuBindings.java`

Add the following method after the existing `updatePosterior` (after line 27, before the `lastEngineCode` method):

```java
  /**
   * Applies sequential Bayesian updates with two-layer tempered likelihoods
   * (SICFUN v0.30.2 Def 15A/15B).
   *
   * <p>Tempered likelihood per hypothesis:
   * <ul>
   *   <li>Standard: {@code pow(likelihood, kappaTemp) + deltaFloor * eta[h]}</li>
   *   <li>Legacy:   {@code (1 - deltaFloor) * likelihood + deltaFloor * eta[h]}</li>
   * </ul>
   *
   * <p>Inputs:
   * <ul>
   *   <li>{@code prior.length == hypothesisCount}</li>
   *   <li>{@code likelihoods.length == observationCount * hypothesisCount} (row-major)</li>
   *   <li>{@code eta.length == hypothesisCount} (or null for uniform)</li>
   *   <li>{@code outPosterior.length == hypothesisCount}</li>
   *   <li>{@code outLogEvidence.length >= 1}</li>
   *   <li>{@code kappaTemp} in (0, 1]</li>
   *   <li>{@code deltaFloor} >= 0</li>
   * </ul>
   *
   * @return 0 on success, non-zero status code on failure.
   */
  public static native int updatePosteriorTempered(
      int observationCount,
      int hypothesisCount,
      double[] prior,
      double[] likelihoods,
      double kappaTemp,
      double deltaFloor,
      double[] eta,
      boolean useLegacyForm,
      double[] outPosterior,
      double[] outLogEvidence
  );
```

- [ ] **Step 2: Modify HoldemBayesNativeGpuBindings.java**

File: `src/main/java/sicfun/holdem/HoldemBayesNativeGpuBindings.java`

Add the same method after the existing `updatePosterior` (after line 15, before the `lastEngineCode` method):

```java
  /** Same ABI contract as {@link HoldemBayesNativeCpuBindings#updatePosteriorTempered}. */
  public static native int updatePosteriorTempered(
      int observationCount,
      int hypothesisCount,
      double[] prior,
      double[] likelihoods,
      double kappaTemp,
      double deltaFloor,
      double[] eta,
      boolean useLegacyForm,
      double[] outPosterior,
      double[] outLogEvidence
  );
```

- [ ] **Step 3: Verify compilation**

```
sbt compile
```

Expected: compiles. The `native` methods won't resolve at runtime until the DLLs are rebuilt, but the Java classes compile fine.

- [ ] **Step 4: Commit**

```
git add src/main/java/sicfun/holdem/HoldemBayesNativeCpuBindings.java \
        src/main/java/sicfun/holdem/HoldemBayesNativeGpuBindings.java
git commit -m "feat(jni): add updatePosteriorTempered native method signatures"
```

---

### Task 5: Add tempered JNI entry point to CPU C++ binding

**Goal:** Add the `Java_sicfun_holdem_HoldemBayesNativeCpuBindings_updatePosteriorTempered` function to `HoldemBayesNativeCpuBindings.cpp`.

- [ ] **Step 1: Modify HoldemBayesNativeCpuBindings.cpp**

File: `src/main/native/jni/HoldemBayesNativeCpuBindings.cpp`

Add the following function AFTER the existing `updatePosterior` function (after line 213, before `lastEngineCode` at line 222):

```cpp
/*
 * JNI entry point: sicfun.holdem.HoldemBayesNativeCpuBindings.updatePosteriorTempered()
 *
 * Two-layer tempered Bayesian update (SICFUN v0.30.2 Def 15A/15B).
 * Same critical-array protocol as updatePosterior, plus three additional
 * parameters: kappaTemp (double), deltaFloor (double), eta (double[]),
 * and useLegacyForm (boolean).
 *
 * The eta array may be null, in which case uniform 1/hypothesisCount is used.
 */
extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HoldemBayesNativeCpuBindings_updatePosteriorTempered(
    JNIEnv* env,
    jclass /*clazz*/,
    jint observationCount,
    jint hypothesisCount,
    jdoubleArray priorArray,
    jdoubleArray likelihoodArray,
    jdouble kappaTemp,
    jdouble deltaFloor,
    jdoubleArray etaArray,
    jboolean useLegacyForm,
    jdoubleArray outPosteriorArray,
    jdoubleArray outLogEvidenceArray) {
  if (priorArray == nullptr || likelihoodArray == nullptr || outPosteriorArray == nullptr ||
      outLogEvidenceArray == nullptr) {
    return bayesnative::kStatusNullArray;
  }

  if (!bayesnative::valid_length_product(static_cast<int>(observationCount),
                                         static_cast<int>(hypothesisCount))) {
    return bayesnative::kStatusInvalidConfig;
  }

  const jsize prior_length = env->GetArrayLength(priorArray);
  const jsize likelihood_length = env->GetArrayLength(likelihoodArray);
  const jsize out_posterior_length = env->GetArrayLength(outPosteriorArray);
  const jsize out_log_evidence_length = env->GetArrayLength(outLogEvidenceArray);

  if (clear_pending_jni_exception(env)) {
    return bayesnative::kStatusReadFailure;
  }

  const jsize expected_prior_length = static_cast<jsize>(hypothesisCount);
  const jsize expected_likelihood_length =
      static_cast<jsize>(static_cast<long long>(observationCount) *
                         static_cast<long long>(hypothesisCount));
  if (prior_length != expected_prior_length || likelihood_length != expected_likelihood_length ||
      out_posterior_length != expected_prior_length || out_log_evidence_length < 1) {
    return bayesnative::kStatusLengthMismatch;
  }

  /* Validate eta length if provided. */
  jsize eta_length = 0;
  if (etaArray != nullptr) {
    eta_length = env->GetArrayLength(etaArray);
    if (clear_pending_jni_exception(env)) {
      return bayesnative::kStatusReadFailure;
    }
    if (eta_length != expected_prior_length) {
      return bayesnative::kStatusLengthMismatch;
    }
  }

  int status = bayesnative::kStatusOk;
  jdouble* prior_values =
      acquire_critical_array(env, priorArray, bayesnative::kStatusReadFailure, status);
  if (prior_values == nullptr) {
    return status;
  }

  jdouble* likelihood_values =
      acquire_critical_array(env, likelihoodArray, bayesnative::kStatusReadFailure, status);
  if (likelihood_values == nullptr) {
    release_critical_array(env, priorArray, prior_values, JNI_ABORT,
                           bayesnative::kStatusReadFailure, status);
    return status;
  }

  /* Acquire eta critical array only if provided. */
  jdouble* eta_values = nullptr;
  if (etaArray != nullptr) {
    eta_values =
        acquire_critical_array(env, etaArray, bayesnative::kStatusReadFailure, status);
    if (eta_values == nullptr) {
      release_critical_array(env, likelihoodArray, likelihood_values, JNI_ABORT,
                             bayesnative::kStatusReadFailure, status);
      release_critical_array(env, priorArray, prior_values, JNI_ABORT,
                             bayesnative::kStatusReadFailure, status);
      return status;
    }
  }

  jdouble* out_posterior_values =
      acquire_critical_array(env, outPosteriorArray, bayesnative::kStatusWriteFailure, status);
  if (out_posterior_values == nullptr) {
    if (eta_values != nullptr) {
      release_critical_array(env, etaArray, eta_values, JNI_ABORT,
                             bayesnative::kStatusReadFailure, status);
    }
    release_critical_array(env, likelihoodArray, likelihood_values, JNI_ABORT,
                           bayesnative::kStatusReadFailure, status);
    release_critical_array(env, priorArray, prior_values, JNI_ABORT,
                           bayesnative::kStatusReadFailure, status);
    return status;
  }

  jdouble* out_log_evidence_values =
      acquire_critical_array(env, outLogEvidenceArray, bayesnative::kStatusWriteFailure, status);
  if (out_log_evidence_values == nullptr) {
    release_critical_array(env, outPosteriorArray, out_posterior_values, JNI_ABORT,
                           bayesnative::kStatusWriteFailure, status);
    if (eta_values != nullptr) {
      release_critical_array(env, etaArray, eta_values, JNI_ABORT,
                             bayesnative::kStatusReadFailure, status);
    }
    release_critical_array(env, likelihoodArray, likelihood_values, JNI_ABORT,
                           bayesnative::kStatusReadFailure, status);
    release_critical_array(env, priorArray, prior_values, JNI_ABORT,
                           bayesnative::kStatusReadFailure, status);
    return status;
  }

  status = bayesnative::update_posterior_tempered_raw(
      static_cast<int>(observationCount), static_cast<int>(hypothesisCount), prior_values,
      likelihood_values, static_cast<double>(kappaTemp), static_cast<double>(deltaFloor),
      eta_values, static_cast<bool>(useLegacyForm),
      out_posterior_values, out_log_evidence_values);

  /* Release input arrays (JNI_ABORT = read-only, no writes to commit). */
  release_critical_array(env, priorArray, prior_values, JNI_ABORT, bayesnative::kStatusReadFailure,
                         status);
  release_critical_array(env, likelihoodArray, likelihood_values, JNI_ABORT,
                         bayesnative::kStatusReadFailure, status);
  if (eta_values != nullptr) {
    release_critical_array(env, etaArray, eta_values, JNI_ABORT,
                           bayesnative::kStatusReadFailure, status);
  }

  /* Release output arrays (commit on success, discard on error). */
  const jint out_release_mode = (status == bayesnative::kStatusOk) ? 0 : JNI_ABORT;
  release_critical_array(env, outPosteriorArray, out_posterior_values, out_release_mode,
                         bayesnative::kStatusWriteFailure, status);
  release_critical_array(env, outLogEvidenceArray, out_log_evidence_values, out_release_mode,
                         bayesnative::kStatusWriteFailure, status);

  if (status != bayesnative::kStatusOk) {
    return status;
  }

  g_last_engine_code.store(kEngineCpu, std::memory_order_relaxed);
  return bayesnative::kStatusOk;
}
```

- [ ] **Step 2: Build the CPU DLL**

```powershell
cd C:\Users\alexl\code\math\untitled\src\main\native
.\build-windows-llvm.ps1
```

Expected: builds successfully, producing `sicfun_bayes_native.dll`.

- [ ] **Step 3: Commit**

```
git add src/main/native/jni/HoldemBayesNativeCpuBindings.cpp
git commit -m "feat(native): add tempered JNI entry point to CPU Bayesian binding"
```

---

### Task 6: Add tempered JNI entry point to GPU CUDA binding

**Goal:** Add the identical `updatePosteriorTempered` function to the GPU binding. The structure mirrors Task 5 exactly, but targets the GPU class name and reports engine code 2.

- [ ] **Step 1: Modify HoldemBayesNativeGpuBindings.cu**

File: `src/main/native/jni/HoldemBayesNativeGpuBindings.cu`

Add the following function AFTER the existing `updatePosterior` function (after line 181, before `lastEngineCode` at line 184):

```cpp
/*
 * JNI entry point: sicfun.holdem.HoldemBayesNativeGpuBindings.updatePosteriorTempered()
 *
 * GPU-context variant of the two-layer tempered Bayesian update.
 * Identical logic to the CPU variant. Reports engine code 2 (GPU).
 */
extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HoldemBayesNativeGpuBindings_updatePosteriorTempered(
    JNIEnv* env,
    jclass /*clazz*/,
    jint observationCount,
    jint hypothesisCount,
    jdoubleArray priorArray,
    jdoubleArray likelihoodArray,
    jdouble kappaTemp,
    jdouble deltaFloor,
    jdoubleArray etaArray,
    jboolean useLegacyForm,
    jdoubleArray outPosteriorArray,
    jdoubleArray outLogEvidenceArray) {
  if (priorArray == nullptr || likelihoodArray == nullptr || outPosteriorArray == nullptr ||
      outLogEvidenceArray == nullptr) {
    return bayesnative::kStatusNullArray;
  }

  if (!bayesnative::valid_length_product(static_cast<int>(observationCount),
                                         static_cast<int>(hypothesisCount))) {
    return bayesnative::kStatusInvalidConfig;
  }

  const jsize prior_length = env->GetArrayLength(priorArray);
  const jsize likelihood_length = env->GetArrayLength(likelihoodArray);
  const jsize out_posterior_length = env->GetArrayLength(outPosteriorArray);
  const jsize out_log_evidence_length = env->GetArrayLength(outLogEvidenceArray);

  if (clear_pending_jni_exception(env)) {
    return bayesnative::kStatusReadFailure;
  }

  const jsize expected_prior_length = static_cast<jsize>(hypothesisCount);
  const jsize expected_likelihood_length =
      static_cast<jsize>(static_cast<long long>(observationCount) *
                         static_cast<long long>(hypothesisCount));
  if (prior_length != expected_prior_length || likelihood_length != expected_likelihood_length ||
      out_posterior_length != expected_prior_length || out_log_evidence_length < 1) {
    return bayesnative::kStatusLengthMismatch;
  }

  jsize eta_length = 0;
  if (etaArray != nullptr) {
    eta_length = env->GetArrayLength(etaArray);
    if (clear_pending_jni_exception(env)) {
      return bayesnative::kStatusReadFailure;
    }
    if (eta_length != expected_prior_length) {
      return bayesnative::kStatusLengthMismatch;
    }
  }

  int status = bayesnative::kStatusOk;
  jdouble* prior_values =
      acquire_critical_array(env, priorArray, bayesnative::kStatusReadFailure, status);
  if (prior_values == nullptr) {
    return status;
  }

  jdouble* likelihood_values =
      acquire_critical_array(env, likelihoodArray, bayesnative::kStatusReadFailure, status);
  if (likelihood_values == nullptr) {
    release_critical_array(env, priorArray, prior_values, JNI_ABORT,
                           bayesnative::kStatusReadFailure, status);
    return status;
  }

  jdouble* eta_values = nullptr;
  if (etaArray != nullptr) {
    eta_values =
        acquire_critical_array(env, etaArray, bayesnative::kStatusReadFailure, status);
    if (eta_values == nullptr) {
      release_critical_array(env, likelihoodArray, likelihood_values, JNI_ABORT,
                             bayesnative::kStatusReadFailure, status);
      release_critical_array(env, priorArray, prior_values, JNI_ABORT,
                             bayesnative::kStatusReadFailure, status);
      return status;
    }
  }

  jdouble* out_posterior_values =
      acquire_critical_array(env, outPosteriorArray, bayesnative::kStatusWriteFailure, status);
  if (out_posterior_values == nullptr) {
    if (eta_values != nullptr) {
      release_critical_array(env, etaArray, eta_values, JNI_ABORT,
                             bayesnative::kStatusReadFailure, status);
    }
    release_critical_array(env, likelihoodArray, likelihood_values, JNI_ABORT,
                           bayesnative::kStatusReadFailure, status);
    release_critical_array(env, priorArray, prior_values, JNI_ABORT,
                           bayesnative::kStatusReadFailure, status);
    return status;
  }

  jdouble* out_log_evidence_values =
      acquire_critical_array(env, outLogEvidenceArray, bayesnative::kStatusWriteFailure, status);
  if (out_log_evidence_values == nullptr) {
    release_critical_array(env, outPosteriorArray, out_posterior_values, JNI_ABORT,
                           bayesnative::kStatusWriteFailure, status);
    if (eta_values != nullptr) {
      release_critical_array(env, etaArray, eta_values, JNI_ABORT,
                             bayesnative::kStatusReadFailure, status);
    }
    release_critical_array(env, likelihoodArray, likelihood_values, JNI_ABORT,
                           bayesnative::kStatusReadFailure, status);
    release_critical_array(env, priorArray, prior_values, JNI_ABORT,
                           bayesnative::kStatusReadFailure, status);
    return status;
  }

  status = bayesnative::update_posterior_tempered_raw(
      static_cast<int>(observationCount), static_cast<int>(hypothesisCount), prior_values,
      likelihood_values, static_cast<double>(kappaTemp), static_cast<double>(deltaFloor),
      eta_values, static_cast<bool>(useLegacyForm),
      out_posterior_values, out_log_evidence_values);

  release_critical_array(env, priorArray, prior_values, JNI_ABORT, bayesnative::kStatusReadFailure,
                         status);
  release_critical_array(env, likelihoodArray, likelihood_values, JNI_ABORT,
                         bayesnative::kStatusReadFailure, status);
  if (eta_values != nullptr) {
    release_critical_array(env, etaArray, eta_values, JNI_ABORT,
                           bayesnative::kStatusReadFailure, status);
  }

  const jint out_release_mode = (status == bayesnative::kStatusOk) ? 0 : JNI_ABORT;
  release_critical_array(env, outPosteriorArray, out_posterior_values, out_release_mode,
                         bayesnative::kStatusWriteFailure, status);
  release_critical_array(env, outLogEvidenceArray, out_log_evidence_values, out_release_mode,
                         bayesnative::kStatusWriteFailure, status);

  if (status != bayesnative::kStatusOk) {
    return status;
  }

  g_last_engine_code.store(kEngineGpu, std::memory_order_relaxed);
  return bayesnative::kStatusOk;
}
```

- [ ] **Step 2: Build the GPU DLL**

```powershell
cd C:\Users\alexl\code\math\untitled\src\main\native
.\build-windows-cuda11.ps1
```

Expected: builds successfully, producing `sicfun_bayes_cuda.dll`.

- [ ] **Step 3: Commit**

```
git add src/main/native/jni/HoldemBayesNativeGpuBindings.cu
git commit -m "feat(native): add tempered JNI entry point to GPU Bayesian binding"
```

---

### Task 7: Modify HoldemBayesNativeRuntime.scala -- add tempered dispatch

**Goal:** Add `updatePosteriorTemperedInPlace` to the Scala native runtime wrapper that dispatches to the new `updatePosteriorTempered` JNI methods.

- [ ] **Step 1: Add tempered method to HoldemBayesNativeRuntime**

File: `src/main/scala/sicfun/holdem/gpu/HoldemBayesNativeRuntime.scala`

Add the following method after `updatePosteriorInPlace` (after line 159), before the existing `updatePosteriorInPlaceUnchecked` private method:

```scala
  /** Lower-level in-place variant with two-layer tempering (SICFUN v0.30.2 Def 15A/15B).
    *
    * @param kappaTemp     power-posterior exponent in (0, 1]
    * @param deltaFloor    additive safety floor >= 0
    * @param eta           full-support distribution, length = hypothesisCount (null for uniform)
    * @param useLegacyForm if true, use legacy (1-eps)*L + eps*eta formula
    * @return `Right(engineCode)` on success, `Left(reason)` on failure
    */
  private[holdem] def updatePosteriorTemperedInPlace(
      backend: Backend,
      observationCount: Int,
      hypothesisCount: Int,
      prior: Array[Double],
      likelihoods: Array[Double],
      kappaTemp: Double,
      deltaFloor: Double,
      eta: Array[Double],
      useLegacyForm: Boolean,
      outPosterior: Array[Double],
      outLogEvidence: Array[Double]
  ): Either[String, Int] =
    if outPosterior.length != hypothesisCount then
      Left(
        s"native Bayesian tempered output posterior length mismatch: expected $hypothesisCount, found ${outPosterior.length}"
      )
    else if outLogEvidence.length < 1 then
      Left("native Bayesian tempered logEvidence output must have length >= 1")
    else if eta != null && eta.length != hypothesisCount then
      Left(
        s"native Bayesian tempered eta length mismatch: expected $hypothesisCount, found ${eta.length}"
      )
    else
      val loadResult =
        backend match
          case Backend.Cpu => cpuLoadResult()
          case Backend.Gpu => gpuLoadResult()
      loadResult match
        case Left(reason) =>
          Left(reason)
        case Right(_) =>
          try
            val status =
              backend match
                case Backend.Cpu =>
                  HoldemBayesNativeCpuBindings.updatePosteriorTempered(
                    observationCount,
                    hypothesisCount,
                    prior,
                    likelihoods,
                    kappaTemp,
                    deltaFloor,
                    eta,
                    useLegacyForm,
                    outPosterior,
                    outLogEvidence
                  )
                case Backend.Gpu =>
                  HoldemBayesNativeGpuBindings.updatePosteriorTempered(
                    observationCount,
                    hypothesisCount,
                    prior,
                    likelihoods,
                    kappaTemp,
                    deltaFloor,
                    eta,
                    useLegacyForm,
                    outPosterior,
                    outLogEvidence
                  )
            if status != 0 then Left(describeStatus(status))
            else
              val engineCode =
                backend match
                  case Backend.Cpu => CpuEngineCode
                  case Backend.Gpu => GpuEngineCode
              Right(engineCode)
          catch
            case ex: UnsatisfiedLinkError =>
              Left(s"${backendLabel(backend)} native tempered Bayesian symbols not found: ${ex.getMessage}")
            case ex: Throwable =>
              Left(
                Option(ex.getMessage)
                  .map(_.trim)
                  .filter(_.nonEmpty)
                  .getOrElse(ex.getClass.getSimpleName)
              )
```

- [ ] **Step 2: Verify compilation**

```
sbt compile
```

Expected: compiles.

- [ ] **Step 3: Commit**

```
git add src/main/scala/sicfun/holdem/gpu/HoldemBayesNativeRuntime.scala
git commit -m "feat(gpu): add tempered dispatch to HoldemBayesNativeRuntime"
```

---

### Task 8: Modify HoldemBayesProvider.scala -- integrate TemperedConfig

**Goal:** Add a tempered update path to the provider. The existing `updatePosterior` method signature does NOT change -- tempering is accessed via a new `updatePosteriorTempered` method and configuration via system properties. This ensures zero risk to existing behavior.

- [ ] **Step 1: Add configuration properties and tempered dispatch to HoldemBayesProvider**

File: `src/main/scala/sicfun/holdem/provider/HoldemBayesProvider.scala`

**1a.** Add import at the top of the file (after the existing import block ending at line 11):

```scala
import sicfun.holdem.strategic.TemperedLikelihood
import sicfun.holdem.strategic.TemperedLikelihood.TemperedConfig
```

**1b.** Add config properties after `DefaultShadowLogEvidenceMaxAbsDiff` (after line 92, before `MinLikelihood`):

```scala
  private val TemperedKappaProperty = "sicfun.bayes.tempered.kappaTemp"
  private val TemperedKappaEnv = "sicfun_BAYES_TEMPERED_KAPPA_TEMP"
  private val TemperedDeltaProperty = "sicfun.bayes.tempered.deltaFloor"
  private val TemperedDeltaEnv = "sicfun_BAYES_TEMPERED_DELTA_FLOOR"
  private val TemperedModeProperty = "sicfun.bayes.tempered.mode"
  private val TemperedModeEnv = "sicfun_BAYES_TEMPERED_MODE"
```

**1c.** Add `configuredTemperedConfig` and `nativeTemperedUpdate` methods at the end of the object (after `configuredNonNegativeFiniteDouble`, before the end of the object):

```scala
  /** Resolves the tempered inference configuration from system properties.
    *
    * Property `sicfun.bayes.tempered.mode`:
    *  - `"legacy"` -- v0.29.1 epsilon-smoothing (kappaTemp/deltaFloor ignored)
    *  - `"tempered"` (default) -- two-layer tempered with configured kappa/delta
    *  - `"off"` -- no tempering (kappaTemp=1, deltaFloor=0)
    */
  private[holdem] def configuredTemperedConfig(): Option[TemperedConfig] =
    GpuRuntimeSupport.resolveNonEmptyLower(TemperedModeProperty, TemperedModeEnv) match
      case Some("off") => None
      case Some("legacy") =>
        val eps = configuredNonNegativeFiniteDouble(
          TemperedDeltaProperty, TemperedDeltaEnv, default = 1e-6
        )
        Some(TemperedConfig.legacy(epsilon = math.max(eps, 1e-12)))
      case _ =>
        val kappa = GpuRuntimeSupport
          .resolveNonEmpty(TemperedKappaProperty, TemperedKappaEnv)
          .flatMap(_.toDoubleOption)
          .filter(v => v > 0.0 && v <= 1.0 && java.lang.Double.isFinite(v))
          .getOrElse(0.9)
        val delta = configuredNonNegativeFiniteDouble(
          TemperedDeltaProperty, TemperedDeltaEnv, default = 1e-8
        )
        Some(TemperedConfig.twoLayer(kappaTemp = kappa, deltaFloor = delta))

  /** Performs a tempered native Bayesian update using the v0.30.2 two-layer formula.
    *
    * Falls back to the Scala reference implementation in TemperedLikelihood if
    * native backends are unavailable.
    */
  private def nativeTemperedUpdate(
      backend: HoldemBayesNativeRuntime.Backend,
      selectedProvider: Provider,
      hypotheses: Vector[HoleCards],
      prior: Array[Double],
      likelihoods: Array[Double],
      observationCount: Int,
      hypothesisCount: Int,
      temperedConfig: TemperedConfig
  ): Option[UpdateResult] =
    val outPosterior = new Array[Double](hypothesisCount)
    val outLogEvidence = Array(0.0d)
    val eta: Array[Double] = null // null signals uniform to the C++ code
    val (kappaTemp, deltaFloor, useLegacy) = temperedConfig match
      case TemperedConfig.TwoLayer(k, d) => (k, d, false)
      case TemperedConfig.Legacy(eps) => (1.0, eps, true)
    HoldemBayesNativeRuntime.updatePosteriorTemperedInPlace(
      backend = backend,
      observationCount = observationCount,
      hypothesisCount = hypothesisCount,
      prior = prior,
      likelihoods = likelihoods,
      kappaTemp = kappaTemp,
      deltaFloor = deltaFloor,
      eta = eta,
      useLegacyForm = useLegacy,
      outPosterior = outPosterior,
      outLogEvidence = outLogEvidence
    ) match
      case Left(reason) =>
        GpuRuntimeSupport.log(
          s"native tempered Bayesian ${backend.toString.toLowerCase} update unavailable: $reason"
        )
        None
      case Right(_) =>
        val clamped = new Array[Double](hypotheses.length)
        var idx = 0
        while idx < hypotheses.length do
          clamped(idx) = math.max(0.0, outPosterior(idx))
          idx += 1
        val compact = HoldemEquity.buildCompactPosterior(hypotheses, clamped)
        Some(
          UpdateResult(
            posterior = compact.distribution,
            compact = compact,
            logEvidence = outLogEvidence(0),
            provider = selectedProvider
          )
        )
```

**Note:** The existing `updatePosterior` method is NOT modified. It continues to work unchanged. Tempering is used by callers that explicitly request it, or by a future engine integration (Phase 4).

- [ ] **Step 2: Verify compilation**

```
sbt compile
```

- [ ] **Step 3: Commit**

```
git add src/main/scala/sicfun/holdem/provider/HoldemBayesProvider.scala
git commit -m "feat(provider): add tempered config resolution and native tempered dispatch"
```

---

### Task 9: Native parity test -- Scala reference vs C++ tempered update

**Goal:** Add tests verifying the C++ `updatePosteriorTempered` produces results matching the Scala `TemperedLikelihood` reference implementation. These tests are gated by native DLL availability.

- [ ] **Step 1: Append parity tests to TemperedLikelihoodTest.scala**

Append the following tests to `src/test/scala/sicfun/holdem/strategic/TemperedLikelihoodTest.scala`, inside the class body:

```scala
  // ==================== Native parity tests (conditional on DLL availability) ====================

  private def nativeCpuAvailable: Boolean =
    try
      sicfun.holdem.gpu.HoldemBayesNativeRuntime
        .availability(sicfun.holdem.gpu.HoldemBayesNativeRuntime.Backend.Cpu)
        .available
    catch case _: Throwable => false

  test("Native parity: tempered C++ matches Scala reference (two-layer, kappa=0.9, delta=1e-8)"):
    assume(nativeCpuAvailable, "native CPU DLL not available")
    val hypothesisCount = 8
    val observationCount = 2
    val prior = Array(0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1)
    val likelihoods = Array(
      0.4, 0.3, 0.2, 0.05, 0.02, 0.01, 0.01, 0.01, // observation 0
      0.1, 0.1, 0.3, 0.2,  0.1,  0.1,  0.05, 0.05   // observation 1
    )
    val kappaTemp = 0.9
    val deltaFloor = 1e-8
    val eta = TemperedLikelihood.defaultEta(hypothesisCount)
    val cfg = TemperedConfig.twoLayer(kappaTemp = kappaTemp, deltaFloor = deltaFloor)

    // Scala reference: apply tempered update observation by observation
    var scalaPosterior = prior.map(_ / prior.sum)
    var obs = 0
    while obs < observationCount do
      val row = likelihoods.slice(obs * hypothesisCount, (obs + 1) * hypothesisCount)
      scalaPosterior = TemperedLikelihood.updatePosterior(scalaPosterior, row, eta, cfg)
      obs += 1

    // Native
    val outPosterior = new Array[Double](hypothesisCount)
    val outLogEvidence = Array(0.0)
    val result = sicfun.holdem.gpu.HoldemBayesNativeRuntime.updatePosteriorTemperedInPlace(
      backend = sicfun.holdem.gpu.HoldemBayesNativeRuntime.Backend.Cpu,
      observationCount = observationCount,
      hypothesisCount = hypothesisCount,
      prior = prior,
      likelihoods = likelihoods,
      kappaTemp = kappaTemp,
      deltaFloor = deltaFloor,
      eta = eta,
      useLegacyForm = false,
      outPosterior = outPosterior,
      outLogEvidence = outLogEvidence
    )
    assert(result.isRight, s"native call failed: $result")

    // Compare
    var i = 0
    while i < hypothesisCount do
      assertEquals(
        outPosterior(i), scalaPosterior(i), 1e-10,
        s"hypothesis $i: native=${outPosterior(i)} scala=${scalaPosterior(i)}"
      )
      i += 1

  test("Native parity: legacy mode matches Scala reference"):
    assume(nativeCpuAvailable, "native CPU DLL not available")
    val hypothesisCount = 4
    val observationCount = 1
    val prior = Array(0.4, 0.3, 0.2, 0.1)
    val likelihoods = Array(0.6, 0.2, 0.15, 0.05)
    val eps = 1e-6
    val eta = TemperedLikelihood.defaultEta(hypothesisCount)
    val cfg = TemperedConfig.legacy(epsilon = eps)

    // Scala reference
    val scalaPosterior = TemperedLikelihood.updatePosterior(prior, likelihoods, eta, cfg)

    // Native
    val outPosterior = new Array[Double](hypothesisCount)
    val outLogEvidence = Array(0.0)
    val result = sicfun.holdem.gpu.HoldemBayesNativeRuntime.updatePosteriorTemperedInPlace(
      backend = sicfun.holdem.gpu.HoldemBayesNativeRuntime.Backend.Cpu,
      observationCount = observationCount,
      hypothesisCount = hypothesisCount,
      prior = prior,
      likelihoods = likelihoods,
      kappaTemp = 1.0,
      deltaFloor = eps,
      eta = eta,
      useLegacyForm = true,
      outPosterior = outPosterior,
      outLogEvidence = outLogEvidence
    )
    assert(result.isRight, s"native call failed: $result")

    var i = 0
    while i < hypothesisCount do
      assertEquals(
        outPosterior(i), scalaPosterior(i), 1e-12,
        s"hypothesis $i: native=${outPosterior(i)} scala=${scalaPosterior(i)}"
      )
      i += 1

  test("Native parity: untampered fast-path matches original updatePosterior"):
    assume(nativeCpuAvailable, "native CPU DLL not available")
    val hypothesisCount = 4
    val observationCount = 1
    val prior = Array(0.4, 0.3, 0.2, 0.1)
    val likelihoods = Array(0.6, 0.2, 0.15, 0.05)

    // Original path
    val origPosterior = new Array[Double](hypothesisCount)
    val origLogEvidence = Array(0.0)
    val origResult = sicfun.holdem.gpu.HoldemBayesNativeRuntime.updatePosteriorInPlace(
      backend = sicfun.holdem.gpu.HoldemBayesNativeRuntime.Backend.Cpu,
      observationCount = observationCount,
      hypothesisCount = hypothesisCount,
      prior = prior,
      likelihoods = likelihoods,
      outPosterior = origPosterior,
      outLogEvidence = origLogEvidence
    )
    assert(origResult.isRight, s"original native call failed: $origResult")

    // Tempered with kappa=1, delta=0 (should delegate to original)
    val temperedPosterior = new Array[Double](hypothesisCount)
    val temperedLogEvidence = Array(0.0)
    val temperedResult = sicfun.holdem.gpu.HoldemBayesNativeRuntime.updatePosteriorTemperedInPlace(
      backend = sicfun.holdem.gpu.HoldemBayesNativeRuntime.Backend.Cpu,
      observationCount = observationCount,
      hypothesisCount = hypothesisCount,
      prior = prior,
      likelihoods = likelihoods,
      kappaTemp = 1.0,
      deltaFloor = 0.0,
      eta = null,
      useLegacyForm = false,
      outPosterior = temperedPosterior,
      outLogEvidence = temperedLogEvidence
    )
    assert(temperedResult.isRight, s"tempered native call failed: $temperedResult")

    // Must be bit-identical (fast path delegates to original function)
    var i = 0
    while i < hypothesisCount do
      assertEquals(
        temperedPosterior(i), origPosterior(i), 0.0,
        s"hypothesis $i must be bit-identical"
      )
      i += 1
    assertEquals(temperedLogEvidence(0), origLogEvidence(0), 0.0, "logEvidence must be bit-identical")
```

- [ ] **Step 2: Run all tests**

```
sbt "testOnly sicfun.holdem.strategic.TemperedLikelihoodTest"
```

Expected: Scala-only tests pass. Native parity tests pass if DLL available, skip otherwise.

- [ ] **Step 3: Commit**

```
git add src/test/scala/sicfun/holdem/strategic/TemperedLikelihoodTest.scala
git commit -m "test(strategic): add native parity tests for tempered Bayesian update"
```

---

### Task 10: Full backward-compatibility integration test

**Goal:** Prove that the complete pipeline (provider -> native runtime -> C++ engine) produces identical results with the identity tempering config (kappa=1, delta=0) vs the original untampered path.

- [ ] **Step 1: Append integration test to TemperedLikelihoodTest.scala**

Append to `src/test/scala/sicfun/holdem/strategic/TemperedLikelihoodTest.scala`:

```scala
  // ==================== Integration backward-compatibility ====================

  test("Integration backward compat: kappa=1 delta=0 through tempered path matches original"):
    assume(nativeCpuAvailable, "native CPU DLL not available")
    // This test verifies the critical invariant from v0.30.2 Section 12:
    //   Setting kappa=1, delta=0, not legacy, produces identical results to
    //   the original untampered path, bit-for-bit.
    val hypothesisCount = 6
    val observationCount = 2
    val prior = Array(0.2, 0.2, 0.15, 0.15, 0.15, 0.15)

    // Likelihoods with the MinLikelihood floor already applied (as the provider does)
    val likelihoods = Array(
      0.4, 0.25, 0.15, 0.1, 0.05, 0.05,
      0.1, 0.2,  0.3,  0.2, 0.1,  0.1
    )

    // Run original (untampered) native path
    val origPosterior = new Array[Double](hypothesisCount)
    val origLogEvidence = Array(0.0)
    val origResult = sicfun.holdem.gpu.HoldemBayesNativeRuntime.updatePosteriorInPlace(
      backend = sicfun.holdem.gpu.HoldemBayesNativeRuntime.Backend.Cpu,
      observationCount = observationCount,
      hypothesisCount = hypothesisCount,
      prior = prior,
      likelihoods = likelihoods,
      outPosterior = origPosterior,
      outLogEvidence = origLogEvidence
    )
    assert(origResult.isRight, s"original native call failed: $origResult")

    // Run tempered with kappa=1, delta=0, NOT legacy
    // This should produce identical results via the fast path
    val temperedPosterior = new Array[Double](hypothesisCount)
    val temperedLogEvidence = Array(0.0)
    val temperedResult = sicfun.holdem.gpu.HoldemBayesNativeRuntime.updatePosteriorTemperedInPlace(
      backend = sicfun.holdem.gpu.HoldemBayesNativeRuntime.Backend.Cpu,
      observationCount = observationCount,
      hypothesisCount = hypothesisCount,
      prior = prior,
      likelihoods = likelihoods,
      kappaTemp = 1.0,
      deltaFloor = 0.0,
      eta = null,
      useLegacyForm = false,
      outPosterior = temperedPosterior,
      outLogEvidence = temperedLogEvidence
    )
    assert(temperedResult.isRight, s"tempered native call failed: $temperedResult")

    var i = 0
    while i < hypothesisCount do
      assertEquals(
        temperedPosterior(i), origPosterior(i), 0.0,
        s"kappa=1 delta=0 must be bit-identical to original at hypothesis $i"
      )
      i += 1
    assertEquals(temperedLogEvidence(0), origLogEvidence(0), 0.0)
```

- [ ] **Step 2: Run all tempered tests**

```
sbt "testOnly sicfun.holdem.strategic.TemperedLikelihoodTest"
```

- [ ] **Step 3: Run existing test suite to verify no regressions**

```
sbt test
```

Expected: all existing tests pass unchanged. The formal layer is additive.

- [ ] **Step 4: Commit**

```
git add src/test/scala/sicfun/holdem/strategic/TemperedLikelihoodTest.scala
git commit -m "test(strategic): add integration backward-compat test for tempered inference"
```

---

## Self-Review Checklist

| Item | Status |
|---|---|
| Defs 15, 15A, 15B fully implemented in Scala | Covered in Task 2 |
| Theorem 1 (unconditional totality) tested | Covered in Task 1 (two tests) |
| Backward compatibility (v0.29.1 recovery) tested | Covered in Task 1 (two tests) + Task 10 |
| Legacy form is SEPARATE formula, not kappa=1 special case | Explicit in code and test (`kappa=1 delta=eps is NOT the same as legacy`) |
| C++ `update_posterior_tempered_raw` preserves 4-wide unrolling | Yes, Task 3 |
| C++ fast path when kappa=1, delta=0 delegates to original | Yes, Task 3 |
| C++ prior preservation when delta=0 and evidence=0 (Def 15B) | Yes, Task 3 |
| JNI signatures added to both Java classes | Task 4 |
| CPU JNI entry point added | Task 5 |
| GPU JNI entry point added | Task 6 |
| Scala native runtime wrapper updated | Task 7 |
| Scala provider gains tempered config and dispatch | Task 8 |
| Native parity: C++ vs Scala reference | Task 9 |
| Integration backward-compat: tempered path vs original path | Task 10 |
| Existing `updatePosterior` unchanged | Yes -- original functions untouched |
| Existing `update_posterior_raw` unchanged | Yes -- original function untouched |
| DLL rebuild instructions referenced | Tasks 5, 6 |
| No dependency on `sicfun.holdem.engine` from `strategic` | Yes -- TemperedLikelihood is pure |
| All tests use munit 1.2.2 FunSuite | Yes |
| `eta` nullable in C++ (defaults to uniform) | Yes |

## File Inventory

| File | Action | Task |
|---|---|---|
| `src/main/scala/sicfun/holdem/strategic/TemperedLikelihood.scala` | CREATE | 2 |
| `src/test/scala/sicfun/holdem/strategic/TemperedLikelihoodTest.scala` | CREATE | 1, 9, 10 |
| `src/main/native/jni/BayesNativeUpdateCore.hpp` | MODIFY | 3 |
| `src/main/java/sicfun/holdem/HoldemBayesNativeCpuBindings.java` | MODIFY | 4 |
| `src/main/java/sicfun/holdem/HoldemBayesNativeGpuBindings.java` | MODIFY | 4 |
| `src/main/native/jni/HoldemBayesNativeCpuBindings.cpp` | MODIFY | 5 |
| `src/main/native/jni/HoldemBayesNativeGpuBindings.cu` | MODIFY | 6 |
| `src/main/scala/sicfun/holdem/gpu/HoldemBayesNativeRuntime.scala` | MODIFY | 7 |
| `src/main/scala/sicfun/holdem/provider/HoldemBayesProvider.scala` | MODIFY | 8 |

## Risk Notes

1. **DLL rebuild required.** Tasks 5 and 6 require rebuilding the native DLLs. Without the rebuild, native parity tests (Task 9-10) will skip via `assume()` but all Scala-only tests pass. The plan is safe to execute partially.

2. **Phase 1 dependency.** Task 2 creates files in `sicfun.holdem.strategic`. If Phase 1 has not created this package yet, the directory must be created first (noted in Task 2 Step 1). TemperedLikelihood has no imports from other Phase 1 types -- it is self-contained.

3. **JNI function naming.** The C++ function names must exactly match the mangled names from the Java class. The names follow the pattern `Java_sicfun_holdem_HoldemBayesNativeCpuBindings_updatePosteriorTempered`. If the Java class is renamed or moved, the C++ names must be updated.

4. **The `eta = null` convention.** Passing `null` for the eta array from Scala to JNI is safe -- the Java signature uses `double[]` which is nullable, and the C++ code checks `eta != nullptr` before accessing it.

5. **Critical array release order.** The JNI binding in Tasks 5-6 acquires up to 5 critical arrays (prior, likelihoods, eta, outPosterior, outLogEvidence). On failure at any point, all previously acquired arrays must be released in reverse order with `JNI_ABORT`. The code follows the existing pattern from the original `updatePosterior` binding, extended for the optional eta array.
