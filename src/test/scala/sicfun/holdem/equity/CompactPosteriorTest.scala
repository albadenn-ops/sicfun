package sicfun.holdem.equity

import munit.FunSuite
import sicfun.core.{Card, Prob}
import Prob.*
import sicfun.holdem.types.*

/**
  * Unit tests for CompactPosterior construction via buildCompactPosterior.
  *
  * Verifies:
  *   - Correct conversion from hypotheses + posterior arrays to Prob-weighted flat arrays
  *   - Zero-weight hypothesis filtering (only positive-weight hands are included)
  *   - Normalization of un-normalized input (weights summing to != 1.0)
  *   - Rejection of all-zero posteriors
  *   - Lazy distribution materialization produces correct DiscreteDistribution
  */
class CompactPosteriorTest extends FunSuite:
  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  test("buildCompactPosterior converts hypotheses + posterior to Prob weights") {
    val h1 = hole("As", "Ks")
    val h2 = hole("Qh", "Jd")
    val hypotheses = Vector(h1, h2)
    val posterior = Array(0.6, 0.4)

    val compact = HoldemEquity.buildCompactPosterior(hypotheses, posterior)

    assertEquals(compact.size, 2)
    assertEquals(compact.hands(0), h1)
    assertEquals(compact.hands(1), h2)
    assertEqualsDouble(Prob(compact.probWeights(0)).toDouble, 0.6, 1e-8)
    assertEqualsDouble(Prob(compact.probWeights(1)).toDouble, 0.4, 1e-8)
  }

  test("buildCompactPosterior skips zero-weight hypotheses") {
    val h1 = hole("As", "Ks")
    val h2 = hole("Qh", "Jd")
    val h3 = hole("Tc", "9c")
    val hypotheses = Vector(h1, h2, h3)
    val posterior = Array(0.7, 0.0, 0.3)

    val compact = HoldemEquity.buildCompactPosterior(hypotheses, posterior)

    assertEquals(compact.size, 2)
    assertEquals(compact.hands(0), h1)
    assertEquals(compact.hands(1), h3)
  }

  test("buildCompactPosterior normalizes un-normalized input") {
    val h1 = hole("As", "Ks")
    val h2 = hole("Qh", "Jd")
    val hypotheses = Vector(h1, h2)
    val posterior = Array(3.0, 7.0)

    val compact = HoldemEquity.buildCompactPosterior(hypotheses, posterior)

    assertEqualsDouble(Prob(compact.probWeights(0)).toDouble, 0.3, 1e-8)
    assertEqualsDouble(Prob(compact.probWeights(1)).toDouble, 0.7, 1e-8)
  }

  test("buildCompactPosterior fails on all-zero posterior") {
    val hypotheses = Vector(hole("As", "Ks"))
    val posterior = Array(0.0)

    interceptMessage[IllegalArgumentException]("requirement failed: all-zero posterior") {
      HoldemEquity.buildCompactPosterior(hypotheses, posterior)
    }
  }

  test("lazy distribution materializes correct DiscreteDistribution") {
    val h1 = hole("As", "Ks")
    val h2 = hole("Qh", "Jd")
    val hypotheses = Vector(h1, h2)
    val posterior = Array(0.6, 0.4)

    val compact = HoldemEquity.buildCompactPosterior(hypotheses, posterior)
    val dist = compact.distribution

    assertEqualsDouble(dist.probabilityOf(h1), 0.6, 1e-8)
    assertEqualsDouble(dist.probabilityOf(h2), 0.4, 1e-8)
  }
