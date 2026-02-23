package sicfun.holdem

import munit.FunSuite
import sicfun.core.Card

import scala.util.Random

class MonteCarloConvergenceTest extends FunSuite:
  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  private def board(tokens: String*): Board =
    Board.from(tokens.map(card))

  test("equityMonteCarlo converges to exact equity against fixed range") {
    val hero = hole("As", "Kd")
    val turnBoard = board("2c", "7h", "Jd", "9s")
    val range = "22+,A2s+,K9s+,QTs+,JTs,T9s,98s,A9o+,KTo+,QJo"

    val exact = HoldemEquity.equityExact(hero, turnBoard, range)
    val estimate = HoldemEquity.equityMonteCarlo(
      hero,
      turnBoard,
      range,
      trials = 20_000,
      rng = new Random(42L)
    )

    val diff = math.abs(estimate.mean - exact.equity)
    val bound = math.max(6.0 * estimate.stderr, 0.02)
    assert(diff <= bound, s"diff=$diff exceeded bound=$bound (stderr=${estimate.stderr})")
  }

  test("equityMonteCarloMulti converges to exact share against fixed ranges") {
    val hero = hole("Ah", "Kh")
    val turnBoard = board("2h", "7d", "Jc", "9h")
    val villainRanges = Seq("QQ+,AKs,AKo", "TT+,AQs+,AQo+")

    val exact = HoldemEquity.equityExactMulti(hero, turnBoard, villainRanges, maxEvaluations = 2_000_000L)
    val estimate = HoldemEquity.equityMonteCarloMulti(
      hero,
      turnBoard,
      villainRanges,
      trials = 25_000,
      rng = new Random(7L)
    )

    val diff = math.abs(estimate.mean - exact.share)
    val bound = math.max(6.0 * estimate.stderr, 0.03)
    assert(diff <= bound, s"diff=$diff exceeded bound=$bound (stderr=${estimate.stderr})")
  }
