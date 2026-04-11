package sicfun.core

import munit.FunSuite

/** Tests for [[CollapseMetrics]] distribution collapse and scoring metrics.
  *
  * Validates the mathematical properties of each metric:
  *  - '''Entropy reduction''': zero for identical distributions, positive when the posterior
  *    is more concentrated than the prior.
  *  - '''KL divergence''': zero for identical distributions, positive for different ones,
  *    and fails when the posterior has support outside the prior (infinite KL).
  *  - '''Effective support''': equals the support size for uniform distributions,
  *    equals 1 for point masses (the "perplexity" interpretation).
  *  - '''Collapse ratio''': 0 for no change, near 1 for extreme narrowing.
  *  - '''Brier score''': 0 for perfect predictions, 2 for maximally wrong binary predictions,
  *    with correct averaging in the mean variant.
  */
class CollapseMetricsTest extends FunSuite:
  test("entropy reduction is zero for identical distributions") {
    val dist = DiscreteDistribution.uniform(Seq(1, 2, 3, 4))
    assert(math.abs(CollapseMetrics.entropyReduction(dist, dist)) < 1e-9)
  }

  test("entropy reduction is positive when posterior is more concentrated") {
    val prior = DiscreteDistribution.uniform(Seq(1, 2, 3, 4))
    val posterior = DiscreteDistribution(Map(1 -> 0.7, 2 -> 0.1, 3 -> 0.1, 4 -> 0.1))
    assert(CollapseMetrics.entropyReduction(prior, posterior) > 0.0)
  }

  test("KL divergence is zero for identical distributions") {
    val dist = DiscreteDistribution.uniform(Seq(1, 2, 3, 4))
    assert(math.abs(CollapseMetrics.klDivergence(dist, dist)) < 1e-9)
  }

  test("KL divergence is positive for different distributions") {
    val prior = DiscreteDistribution.uniform(Seq(1, 2, 3, 4))
    val posterior = DiscreteDistribution(Map(1 -> 0.7, 2 -> 0.1, 3 -> 0.1, 4 -> 0.1))
    assert(CollapseMetrics.klDivergence(prior, posterior) > 0.0)
  }

  test("KL divergence fails when posterior has support outside prior") {
    val prior = DiscreteDistribution(Map(1 -> 0.5, 2 -> 0.5))
    val posterior = DiscreteDistribution(Map(1 -> 0.5, 3 -> 0.5))
    intercept[IllegalArgumentException] {
      CollapseMetrics.klDivergence(prior, posterior)
    }
  }

  test("effective support of uniform distribution equals support size") {
    val dist = DiscreteDistribution.uniform(Seq(1, 2, 3, 4))
    assert(math.abs(CollapseMetrics.effectiveSupport(dist) - 4.0) < 1e-9)
  }

  test("effective support of point mass is 1") {
    val dist = DiscreteDistribution(Map("a" -> 1.0))
    assert(math.abs(CollapseMetrics.effectiveSupport(dist) - 1.0) < 1e-9)
  }

  test("collapse ratio is 0 for identical distributions") {
    val dist = DiscreteDistribution.uniform(Seq(1, 2, 3, 4))
    assert(math.abs(CollapseMetrics.collapseRatio(dist, dist)) < 1e-9)
  }

  test("collapse ratio is close to 1 for extreme narrowing") {
    val prior = DiscreteDistribution.uniform(Seq(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
    val posterior = DiscreteDistribution(Map(1 -> 1.0))
    val ratio = CollapseMetrics.collapseRatio(prior, posterior)
    assert(ratio > 0.85 && ratio <= 1.0, s"expected high collapse ratio, got $ratio")
  }

  test("Brier score is 0 for perfect prediction") {
    val perfect = Vector(0.0, 1.0, 0.0)
    assertEquals(CollapseMetrics.brierScore(perfect, 1), 0.0)
  }

  test("Brier score is 2 for maximally wrong prediction") {
    val wrong = Vector(1.0, 0.0)
    assert(math.abs(CollapseMetrics.brierScore(wrong, 1) - 2.0) < 1e-9)
  }

  test("mean Brier score averages correctly") {
    val predictions = Seq(
      (Vector(1.0, 0.0), 0),
      (Vector(0.0, 1.0), 0)
    )
    assert(math.abs(CollapseMetrics.meanBrierScore(predictions) - 1.0) < 1e-9)
  }

  test("Brier score rejects out-of-range actual index") {
    intercept[IllegalArgumentException] {
      CollapseMetrics.brierScore(Vector(0.5, 0.5), -1)
    }
    intercept[IllegalArgumentException] {
      CollapseMetrics.brierScore(Vector(0.5, 0.5), 2)
    }
  }

  test("mean Brier score rejects empty predictions") {
    intercept[IllegalArgumentException] {
      CollapseMetrics.meanBrierScore(Seq.empty)
    }
  }
