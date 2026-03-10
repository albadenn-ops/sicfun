package sicfun.holdem.equity

import munit.FunSuite
import sicfun.core.{Card, DiscreteDistribution}
import sicfun.holdem.types.*

class ProbEquityCorrectnessTest extends FunSuite:
  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  private def board(tokens: String*): Board =
    Board.from(tokens.map(card))

  // Integer division truncation can differ from Double by up to boardCount/2^30
  // per step. After normalization this is <0.1% relative error.
  private val tolerance = 1e-3

  test("river (0 missing) — single villain, hero wins") {
    val hero = hole("As", "Ks")
    val villain = hole("Qh", "Jd")
    val b = board("2c", "3d", "4h", "5s", "9c")
    val range = DiscreteDistribution(Map(villain -> 1.0))

    val dbl = HoldemEquity.equityExact(hero, b, range)
    val prob = HoldemEquity.equityExactProb(hero, b, range)

    assertEqualsDouble(prob.win, dbl.win, tolerance)
    assertEqualsDouble(prob.tie, dbl.tie, tolerance)
    assertEqualsDouble(prob.loss, dbl.loss, tolerance)
  }

  test("river — all ties") {
    val hero = hole("2c", "3d")
    val villain = hole("4h", "5c")
    val b = board("As", "Ks", "Qs", "Js", "Ts")
    val range = DiscreteDistribution(Map(villain -> 1.0))

    val dbl = HoldemEquity.equityExact(hero, b, range)
    val prob = HoldemEquity.equityExactProb(hero, b, range)

    assertEqualsDouble(prob.win, dbl.win, tolerance)
    assertEqualsDouble(prob.tie, dbl.tie, tolerance)
    assertEqualsDouble(prob.loss, dbl.loss, tolerance)
  }

  test("turn (1 missing) — multi-hand range with varied weights") {
    val hero = hole("Ah", "Kh")
    val b = board("2c", "7d", "Ts", "Qc")
    val range = DiscreteDistribution(Map(
      hole("Jc", "Jd") -> 0.4,
      hole("9s", "8s") -> 0.3,
      hole("5c", "4c") -> 0.3
    ))

    val dbl = HoldemEquity.equityExact(hero, b, range)
    val prob = HoldemEquity.equityExactProb(hero, b, range)

    assertEqualsDouble(prob.win, dbl.win, tolerance)
    assertEqualsDouble(prob.tie, dbl.tie, tolerance)
    assertEqualsDouble(prob.loss, dbl.loss, tolerance)
  }

  test("turn — single villain with fractional weight") {
    val hero = hole("Ah", "Kh")
    val b = board("2c", "7d", "Ts", "Qc")
    val range = DiscreteDistribution(Map(hole("Jc", "Jd") -> 0.3))

    val dbl = HoldemEquity.equityExact(hero, b, range)
    val prob = HoldemEquity.equityExactProb(hero, b, range)

    assertEqualsDouble(prob.equity, dbl.equity, tolerance)
  }

  test("flop (2 missing) — uniform range") {
    val hero = hole("Ah", "Kh")
    val b = board("2c", "7d", "Ts")
    val range = DiscreteDistribution.uniform(Seq(
      hole("Jc", "Jd"),
      hole("9s", "8s"),
      hole("5c", "4c")
    ))

    val dbl = HoldemEquity.equityExact(hero, b, range)
    val prob = HoldemEquity.equityExactProb(hero, b, range)

    assertEqualsDouble(prob.win, dbl.win, tolerance)
    assertEqualsDouble(prob.tie, dbl.tie, tolerance)
    assertEqualsDouble(prob.loss, dbl.loss, tolerance)
  }

  test("equity values match") {
    val hero = hole("Ah", "Kh")
    val b = board("2c", "7d", "Ts", "Qc")
    val range = DiscreteDistribution(Map(
      hole("Jc", "Jd") -> 0.5,
      hole("9s", "8s") -> 0.5
    ))

    val dbl = HoldemEquity.equityExact(hero, b, range)
    val prob = HoldemEquity.equityExactProb(hero, b, range)

    assertEqualsDouble(prob.equity, dbl.equity, tolerance)
  }
