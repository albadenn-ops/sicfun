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
    assertEqualsDouble(cfg.kappaTemp, 0.85, 1e-15)
    assert(!cfg.isLegacy)

  test("Def 15: TemperedConfig.legacy captures epsilon"):
    val cfg = TemperedConfig.legacy(epsilon = 1e-6)
    assert(cfg.isLegacy)
    assertEqualsDouble(cfg.epsilon, 1e-6, 1e-20)
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
      assertEqualsDouble(result(i), expected, 1e-15)
      i += 1

  test("Def 15A: kappa=1 gives Pr + delta*eta (NOT legacy form)"):
    val basePr = Array(0.5, 0.3, 0.2)
    val eta = Array(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    val result = TemperedLikelihood.computeLikelihoods(basePr, 1.0, 0.01, eta)
    // Two-layer with kappa=1: Pr(y|c)^1 + 0.01 * eta(y) = Pr(y|c) + 0.01/3
    var i = 0
    while i < 3 do
      val expected = basePr(i) + 0.01 / 3.0
      assertEqualsDouble(result(i), expected, 1e-15)
      i += 1

  test("Def 15A: property 1 -- unconditional totality when delta > 0"):
    val basePr = Array(0.0, 0.0, 0.0) // all-zero base probability
    val eta = Array(0.5, 0.3, 0.2)
    val delta = 1e-8
    val result = TemperedLikelihood.computeLikelihoods(basePr, 0.9, delta, eta)
    var i = 0
    while i < 3 do
      assert(result(i) > 0.0, s"class $i should be positive even with zero base Pr")
      assertEqualsDouble(result(i), delta * eta(i), 1e-25)
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
      assertEqualsDouble(result(i), expected, 1e-15)
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
    assertEqualsDouble(sum, 1.0, 1e-12)
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
    assertEqualsDouble(sum, 1.0, 1e-12)
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
      assertEqualsDouble(posterior(i), prior(i) / prior.sum, 1e-12)
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
        assertEqualsDouble(sum, 1.0, 1e-10)
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
      assertEqualsDouble(legacyPosterior(i), expected(i), 1e-15)
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
      assertEqualsDouble(eta(i), 0.2, 1e-15)
      i += 1

  // ==================== Edge cases ====================

  test("single-class posterior is always 1.0"):
    val prior = Array(1.0)
    val basePr = Array(0.5)
    val eta = Array(1.0)
    val cfg = TemperedConfig.twoLayer(kappaTemp = 0.9, deltaFloor = 1e-8)
    val posterior = TemperedLikelihood.updatePosterior(prior, basePr, eta, cfg)
    assertEqualsDouble(posterior(0), 1.0, 1e-15)

  test("two-class extreme tempering (kappa near 0) flattens likelihood ratio"):
    val basePr = Array(0.99, 0.01)
    val eta = Array(0.5, 0.5)
    val result = TemperedLikelihood.computeLikelihoods(basePr, 0.01, 0.0, eta)
    // 0.99^0.01 ~ 0.99990, 0.01^0.01 ~ 0.95499
    // Ratio should be very close to 1
    val ratio = result(0) / result(1)
    assert(ratio < 2.0, s"extreme tempering should flatten ratio to near 1, got $ratio")

  // ==================== Native parity tests (conditional on DLL availability) ====================

  private def nativeTemperedCpuAvailable: Boolean =
    // Probe whether the tempered JNI symbol exists in the loaded DLL.
    // availability() only checks the base library load; the tempered symbol
    // may not be present in older compiled DLLs, so we do a minimal probe call.
    val probe = sicfun.holdem.gpu.HoldemBayesNativeRuntime.updatePosteriorTemperedInPlace(
      backend = sicfun.holdem.gpu.HoldemBayesNativeRuntime.Backend.Cpu,
      observationCount = 1,
      hypothesisCount = 1,
      prior = Array(1.0),
      likelihoods = Array(1.0),
      kappaTemp = 1.0,
      deltaFloor = 0.0,
      eta = null,
      useLegacyForm = false,
      outPosterior = new Array[Double](1),
      outLogEvidence = Array(0.0)
    )
    probe match
      case Left(msg) if msg.contains("symbols not found") => false
      case Left(msg) if msg.contains("not available")     => false
      case _                                               => true

  test("Native parity: tempered C++ matches Scala reference (two-layer, kappa=0.9, delta=1e-8)"):
    assume(nativeTemperedCpuAvailable, "native CPU tempered symbol not available")
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
      assertEqualsDouble(outPosterior(i), scalaPosterior(i), 1e-10)
      i += 1

  test("Native parity: legacy mode matches Scala reference"):
    assume(nativeTemperedCpuAvailable, "native CPU tempered symbol not available")
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
      assertEqualsDouble(outPosterior(i), scalaPosterior(i), 1e-12)
      i += 1

  test("Native parity: untampered fast-path matches original updatePosterior"):
    assume(nativeTemperedCpuAvailable, "native CPU tempered symbol not available")
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
      assertEqualsDouble(temperedPosterior(i), origPosterior(i), 0.0)
      i += 1
    assertEqualsDouble(temperedLogEvidence(0), origLogEvidence(0), 0.0)

  // ==================== Integration backward-compatibility ====================

  test("Integration backward compat: kappa=1 delta=0 through tempered path matches original"):
    assume(nativeTemperedCpuAvailable, "native CPU tempered symbol not available")
    val hypothesisCount = 6
    val observationCount = 2
    val prior = Array(0.2, 0.2, 0.15, 0.15, 0.15, 0.15)

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
      assertEqualsDouble(temperedPosterior(i), origPosterior(i), 0.0)
      i += 1
    assertEqualsDouble(temperedLogEvidence(0), origLogEvidence(0), 0.0)
