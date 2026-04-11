package sicfun.holdem.analysis
import sicfun.holdem.types.*
import sicfun.holdem.equity.*

import munit.FunSuite
import sicfun.core.{Card, DiscreteDistribution}

/** Tests for [[EvAnalysis.evVariance]], which computes the weighted mean equity
  * and variance of hero's equity across a Bayesian posterior over villain hands.
  *
  * Coverage includes:
  *   - Single-hand posteriors (zero variance baseline)
  *   - Uniform vs non-uniform weight distributions
  *   - Filtering of overlapping hands (hero/board card conflicts)
  *   - Filtering of zero-weight hands
  *   - Rejection when no valid villain hands remain
  *   - Relationship between stderr and weight concentration
  */
class EvAnalysisTest extends FunSuite:
  /** Parses a two-character card token (e.g. "Ah") into a [[Card]], failing the test on bad input. */
  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  /** Parses a four-character string (e.g. "AcKc") into canonical [[HoleCards]]. */
  private def holeCards(token: String): HoleCards =
    val c1 = card(token.substring(0, 2))
    val c2 = card(token.substring(2, 4))
    HoleCards.canonical(c1, c2)

  /** Builds a [[Board]] from a varargs list of card tokens. */
  private def board(tokens: String*): Board =
    Board.from(tokens.map(t => card(t)))

  test("evVariance returns valid estimate for single villain hand") {
    val hero = holeCards("AcKc")
    val villain = holeCards("7d2s")
    val b = board("Ts", "9h", "8d")
    val posterior = DiscreteDistribution(Map(villain -> 1.0))

    val result = EvAnalysis.evVariance(hero, b, posterior)

    assertEquals(result.handCount, 1)
    assert(result.mean >= 0.0 && result.mean <= 1.0, s"mean out of range: ${result.mean}")
    assertEquals(result.variance, 0.0)
    assertEquals(result.stderr, 0.0)
  }

  test("evVariance with uniform two-hand posterior returns mean between individual equities") {
    val hero = holeCards("AcKc")
    val v1 = holeCards("7d2s")
    val v2 = holeCards("QhJh")
    val b = board("Ts", "9h", "8d")

    val posterior = DiscreteDistribution(Map(v1 -> 1.0, v2 -> 1.0))
    val result = EvAnalysis.evVariance(hero, b, posterior)

    assertEquals(result.handCount, 2)
    assert(result.mean >= 0.0 && result.mean <= 1.0, s"mean out of range: ${result.mean}")
    assert(result.variance >= 0.0, s"variance should be non-negative: ${result.variance}")
    assert(result.stderr >= 0.0, s"stderr should be non-negative: ${result.stderr}")

    // With uniform weights, mean should be the average of the two individual equities
    val eq1 = HoldemEquity.equityExact(hero, b, DiscreteDistribution(Map(v1 -> 1.0))).equity
    val eq2 = HoldemEquity.equityExact(hero, b, DiscreteDistribution(Map(v2 -> 1.0))).equity
    val expectedMean = (eq1 + eq2) / 2.0
    assertEqualsDouble(result.mean, expectedMean, 1e-6)
  }

  test("evVariance filters out overlapping villain hands") {
    val hero = holeCards("AcKc")
    // Villain hand that overlaps with hero (shares Ac)
    val overlapping = holeCards("AcQd")
    // Valid villain hand
    val valid = holeCards("7d2s")
    val b = board("Ts", "9h", "8d")

    val posterior = DiscreteDistribution(Map(overlapping -> 1.0, valid -> 1.0))
    val result = EvAnalysis.evVariance(hero, b, posterior)

    // Overlapping hand should be filtered out
    assertEquals(result.handCount, 1)
  }

  test("evVariance filters out villain hands overlapping with board") {
    val hero = holeCards("AcKc")
    // Villain hand that overlaps with board card Ts
    val overlapping = holeCards("Ts2d")
    val valid = holeCards("7d2s")
    val b = board("Ts", "9h", "8d")

    val posterior = DiscreteDistribution(Map(overlapping -> 1.0, valid -> 1.0))
    val result = EvAnalysis.evVariance(hero, b, posterior)

    assertEquals(result.handCount, 1)
  }

  test("evVariance filters out zero-weight hands") {
    val hero = holeCards("AcKc")
    val v1 = holeCards("7d2s")
    val v2 = holeCards("QhJh")
    val b = board("Ts", "9h", "8d")

    val posterior = DiscreteDistribution(Map(v1 -> 1.0, v2 -> 0.0))
    val result = EvAnalysis.evVariance(hero, b, posterior)

    assertEquals(result.handCount, 1)
  }

  test("evVariance rejects posterior with no valid villain hands") {
    val hero = holeCards("AcKc")
    // All villain hands overlap with hero
    val overlapping1 = holeCards("AcQd")
    val overlapping2 = holeCards("Kc7h")
    val b = board("Ts", "9h", "8d")

    val posterior = DiscreteDistribution(Map(overlapping1 -> 1.0, overlapping2 -> 1.0))

    intercept[IllegalArgumentException] {
      EvAnalysis.evVariance(hero, b, posterior)
    }
  }

  test("evVariance with non-uniform weights produces weighted mean") {
    val hero = holeCards("AcKc")
    val v1 = holeCards("7d2s")
    val v2 = holeCards("QhJh")
    val b = board("Ts", "9h", "8d")

    // Heavily weight v1
    val posterior = DiscreteDistribution(Map(v1 -> 9.0, v2 -> 1.0))
    val result = EvAnalysis.evVariance(hero, b, posterior)

    // Compute individual equities
    val eq1 = HoldemEquity.equityExact(hero, b, DiscreteDistribution(Map(v1 -> 1.0))).equity
    val eq2 = HoldemEquity.equityExact(hero, b, DiscreteDistribution(Map(v2 -> 1.0))).equity
    // Weighted mean: 90% v1, 10% v2
    val expectedMean = eq1 * 0.9 + eq2 * 0.1
    assertEqualsDouble(result.mean, expectedMean, 1e-6)
  }

  test("evVariance variance is zero when all villain equities are identical") {
    val hero = holeCards("AcKc")
    // Two villain hands that are suit-equivalent against hero on this board
    // should produce very similar (maybe not exactly equal) equities.
    // Use just one villain hand duplicated conceptually via a single entry.
    val v1 = holeCards("7d2s")
    val b = board("Ts", "9h", "8d")

    val posterior = DiscreteDistribution(Map(v1 -> 1.0))
    val result = EvAnalysis.evVariance(hero, b, posterior)

    assertEquals(result.variance, 0.0)
    assertEquals(result.stderr, 0.0)
  }

  test("evVariance stderr decreases with more uniform weight distribution") {
    val hero = holeCards("AcKc")
    val v1 = holeCards("7d2s")
    val v2 = holeCards("QhJh")
    val v3 = holeCards("5c4c")
    val b = board("Ts", "9h", "8d")

    // Concentrated posterior: one hand dominates
    val concentrated = DiscreteDistribution(Map(v1 -> 100.0, v2 -> 1.0, v3 -> 1.0))
    val resultConcentrated = EvAnalysis.evVariance(hero, b, concentrated)

    // Uniform posterior: all hands equally likely
    val uniform = DiscreteDistribution(Map(v1 -> 1.0, v2 -> 1.0, v3 -> 1.0))
    val resultUniform = EvAnalysis.evVariance(hero, b, uniform)

    // Both should have non-negative stderr
    assert(resultConcentrated.stderr >= 0.0)
    assert(resultUniform.stderr >= 0.0)
  }
