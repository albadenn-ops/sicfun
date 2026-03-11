package sicfun.holdem.equity

import munit.FunSuite
import sicfun.core.{Card, DiscreteDistribution, Prob}
import Prob.*
import sicfun.holdem.types.*

class CompactPosteriorEquityTest extends FunSuite:
  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  private def board(tokens: String*): Board =
    Board.from(tokens.map(card))

  private val tolerance = 1e-3

  // Helper: build compact from a DiscreteDistribution (simulating Bayes output)
  private def compactFrom(dist: DiscreteDistribution[HoleCards]): HoldemEquity.CompactPosterior =
    val hypotheses = dist.weights.keysIterator.toVector
    val posterior = hypotheses.map(dist.probabilityOf).toArray
    HoldemEquity.buildCompactPosterior(hypotheses, posterior)

  test("equityExactProb(compact) matches equityExactProb(distribution) — river") {
    val hero = hole("As", "Ks")
    val b = board("2c", "3d", "4h", "5s", "9c")
    val range = DiscreteDistribution(Map(
      hole("Qh", "Jd") -> 0.6,
      hole("Tc", "9d") -> 0.4
    ))

    val fromMap = HoldemEquity.equityExactProb(hero, b, range)
    val fromCompact = HoldemEquity.equityExactProb(hero, b, compactFrom(range))

    assertEqualsDouble(fromCompact.win, fromMap.win, tolerance)
    assertEqualsDouble(fromCompact.tie, fromMap.tie, tolerance)
    assertEqualsDouble(fromCompact.loss, fromMap.loss, tolerance)
  }

  test("equityExactProb(compact) matches — turn, multi-hand range") {
    val hero = hole("Ah", "Kh")
    val b = board("2c", "7d", "Ts", "Qc")
    val range = DiscreteDistribution(Map(
      hole("Jc", "Jd") -> 0.4,
      hole("9s", "8s") -> 0.3,
      hole("5c", "4c") -> 0.3
    ))

    val fromMap = HoldemEquity.equityExactProb(hero, b, range)
    val fromCompact = HoldemEquity.equityExactProb(hero, b, compactFrom(range))

    assertEqualsDouble(fromCompact.win, fromMap.win, tolerance)
    assertEqualsDouble(fromCompact.tie, fromMap.tie, tolerance)
    assertEqualsDouble(fromCompact.loss, fromMap.loss, tolerance)
  }

  test("equityExactProb(compact) matches — flop, uniform range") {
    val hero = hole("Ah", "Kh")
    val b = board("2c", "7d", "Ts")
    val range = DiscreteDistribution.uniform(Seq(
      hole("Jc", "Jd"),
      hole("9s", "8s"),
      hole("5c", "4c")
    ))

    val fromMap = HoldemEquity.equityExactProb(hero, b, range)
    val fromCompact = HoldemEquity.equityExactProb(hero, b, compactFrom(range))

    assertEqualsDouble(fromCompact.equity, fromMap.equity, tolerance)
  }

  test("equityMonteCarlo(compact) matches equityMonteCarlo(distribution) — turn") {
    val hero = hole("Ah", "Kh")
    val b = board("2c", "7d", "Ts", "Qc")
    val range = DiscreteDistribution(Map(
      hole("Jc", "Jd") -> 0.5,
      hole("9s", "8s") -> 0.5
    ))

    val rng1 = new scala.util.Random(42L)
    val rng2 = new scala.util.Random(42L)
    val fromMap = HoldemEquity.equityMonteCarlo(hero, b, range, trials = 5000, rng = rng1)
    val fromCompact = HoldemEquity.equityMonteCarlo(hero, b, compactFrom(range), trials = 5000, rng = rng2)

    // MC with same seed should give very similar results (not identical
    // due to different villain iteration order, but close)
    assertEqualsDouble(fromCompact.mean, fromMap.mean, 0.02)
  }

  test("equityExactProb(compact) handles hero-card dead filtering") {
    val hero = hole("As", "Ks")
    val b = board("2c", "3d", "4h", "5s", "9c")
    // Include a hand that overlaps with hero cards — should be filtered out
    val range = DiscreteDistribution(Map(
      hole("As", "Qh") -> 0.5, // dead — shares As with hero
      hole("Tc", "9d") -> 0.5
    ))

    val fromMap = HoldemEquity.equityExactProb(hero, b, range)
    val fromCompact = HoldemEquity.equityExactProb(hero, b, compactFrom(range))

    assertEqualsDouble(fromCompact.win, fromMap.win, tolerance)
    assertEqualsDouble(fromCompact.tie, fromMap.tie, tolerance)
    assertEqualsDouble(fromCompact.loss, fromMap.loss, tolerance)
  }

  test("equityExactProb(compact) handles board-card dead filtering") {
    val hero = hole("As", "Ks")
    val b = board("2c", "3d", "4h", "5s", "9c")
    // Villain hand overlaps with board card 2c
    val range = DiscreteDistribution(Map(
      hole("2c", "Qh") -> 0.5, // dead — shares 2c with board
      hole("Tc", "9d") -> 0.5
    ))

    val fromMap = HoldemEquity.equityExactProb(hero, b, range)
    val fromCompact = HoldemEquity.equityExactProb(hero, b, compactFrom(range))

    assertEqualsDouble(fromCompact.win, fromMap.win, tolerance)
    assertEqualsDouble(fromCompact.tie, fromMap.tie, tolerance)
    assertEqualsDouble(fromCompact.loss, fromMap.loss, tolerance)
  }

  test("equityExactProb(compact) handles non-canonical hands via canonical dedup") {
    val hero = hole("Ah", "Kh")
    val b = board("2c", "7d", "Ts", "Qc")
    // Build compact with non-canonical hand order (Jd-Jc instead of Jc-Jd)
    val nonCanonical = hole("Jd", "Jc") // non-canonical order
    val canonical = hole("Jc", "Jd")    // canonical order
    val hands = Array(nonCanonical)
    val weights = Array(Prob.fromDouble(1.0).raw)
    val compact = new HoldemEquity.CompactPosterior(hands, weights, 1)

    // Compare against Map path with canonical hand
    val range = DiscreteDistribution(Map(canonical -> 1.0))
    val fromMap = HoldemEquity.equityExactProb(hero, b, range)
    val fromCompact = HoldemEquity.equityExactProb(hero, b, compact)

    assertEqualsDouble(fromCompact.equity, fromMap.equity, tolerance)
  }
