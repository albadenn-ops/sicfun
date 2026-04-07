package sicfun.holdem.equity
import sicfun.holdem.types.*

import munit.FunSuite
import sicfun.core.{Card, Deck, DiscreteDistribution, HandEvaluator}

import scala.util.Random

/**
  * Comprehensive tests for the core HoldemEquity calculation engine.
  *
  * Covers:
  *   - equityExact: deterministic win on complete board, tie detection, range string input
  *   - equityMonteCarlo: matching exact results on complete boards, range string input
  *   - equityMonteCarloMulti: correct tie splitting in multi-way pots, outright wins
  *   - equityExactMulti: exact share computation validated against manual board enumeration
  *     on both turn (1 missing card) and flop (2 missing cards) boards
  *   - Input validation: non-positive trials, hero-board overlap, preflop rejection for
  *     exact multi, empty villain ranges, impossible non-overlapping villain sampling
  *   - evCall: rejection of invalid numeric inputs
  *   - Accelerated batch path: cpu-emulated GPU provider exercises the batch codepath
  *
  * Manual enumeration tests independently compute equity by iterating all remaining cards
  * and evaluating hands, then compare against the engine's output to within floating-point
  * tolerance (1e-12).
  */
class HoldemEquityTest extends FunSuite:
  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  private def board(tokens: String*): Board =
    Board.from(tokens.map(card))

  /** Local k-subset combination generator for manual enumeration in test assertions. */
  private def combinations[A](items: IndexedSeq[A], k: Int): Iterator[Vector[A]] =
    require(k >= 0 && k <= items.length)
    def loop(start: Int, kLeft: Int, acc: Vector[A]): Iterator[Vector[A]] =
      if kLeft == 0 then Iterator.single(acc)
      else
        val maxStart = items.length - kLeft
        (start to maxStart).iterator.flatMap { i =>
          loop(i + 1, kLeft - 1, acc :+ items(i))
        }
    loop(0, k, Vector.empty)

  /** Computes C(n,k) for assertion denominators in manual enumeration tests. */
  private def combinationsCount(n: Int, k: Int): Long =
    require(k >= 0 && k <= n)
    if k == 0 || k == n then 1L
    else
      val kk = math.min(k, n - k)
      var numer = 1L
      var denom = 1L
      var i = 1
      while i <= kk do
        numer *= (n - kk + i).toLong
        denom *= i.toLong
        i += 1
      numer / denom

  test("equityExact returns deterministic win on complete board") {
    val hero = hole("As", "Ks")
    val villain = hole("Qh", "Jd")
    val fullBoard = board("2c", "3d", "4h", "5s", "9c")
    val range = DiscreteDistribution(Map(villain -> 1.0))

    val result = HoldemEquity.equityExact(hero, fullBoard, range)
    assertEquals(result.win, 1.0)
    assertEquals(result.tie, 0.0)
    assertEquals(result.loss, 0.0)
    assertEquals(result.equity, 1.0)
  }

  test("equityExact returns tie when board is best possible hand") {
    val hero = hole("2c", "3d")
    val villain = hole("4h", "5c")
    val fullBoard = board("As", "Ks", "Qs", "Js", "Ts")
    val range = DiscreteDistribution(Map(villain -> 1.0))

    val result = HoldemEquity.equityExact(hero, fullBoard, range)
    assertEquals(result.win, 0.0)
    assertEquals(result.tie, 1.0)
    assertEquals(result.loss, 0.0)
    assertEquals(result.equity, 0.5)
  }

  test("equityExact accepts range string input") {
    val hero = hole("2c", "3d")
    val fullBoard = board("As", "Ks", "Qs", "Js", "Ts")
    val result = HoldemEquity.equityExact(hero, fullBoard, "AKs")
    assertEquals(result.win, 0.0)
    assertEquals(result.tie, 1.0)
    assertEquals(result.loss, 0.0)
  }

  test("equityMonteCarlo matches exact result with single opponent and full board") {
    val hero = hole("As", "Ks")
    val villain = hole("Qh", "Jd")
    val fullBoard = board("2c", "3d", "4h", "5s", "9c")
    val range = DiscreteDistribution(Map(villain -> 1.0))

    val estimate = HoldemEquity.equityMonteCarlo(hero, fullBoard, range, trials = 50, rng = new Random(1))
    assertEquals(estimate.mean, 1.0)
    assertEquals(estimate.winRate, 1.0)
    assertEquals(estimate.tieRate, 0.0)
    assertEquals(estimate.lossRate, 0.0)
  }

  test("equityMonteCarlo accepts range string input") {
    val hero = hole("2c", "3d")
    val fullBoard = board("As", "Ks", "Qs", "Js", "Ts")
    val estimate = HoldemEquity.equityMonteCarlo(hero, fullBoard, "AKs", trials = 20, rng = new Random(7))
    assertEquals(estimate.mean, 0.5)
    assertEquals(estimate.tieRate, 1.0)
  }

  test("equityMonteCarloMulti splits equity on full-board tie") {
    val hero = hole("2c", "3d")
    val v1 = hole("4h", "5c")
    val v2 = hole("6d", "7h")
    val fullBoard = board("As", "Ks", "Qs", "Js", "Ts")
    val estimate = HoldemEquity.equityMonteCarloMulti(
      hero,
      fullBoard,
      Seq(
        DiscreteDistribution(Map(v1 -> 1.0)),
        DiscreteDistribution(Map(v2 -> 1.0))
      ),
      trials = 30,
      rng = new Random(1)
    )
    assertEquals(estimate.mean, 1.0 / 3.0)
    assertEquals(estimate.tieRate, 1.0)
    assertEquals(estimate.lossRate, 0.0)
  }

  test("equityMonteCarloMulti returns win on fixed full-board hero winner") {
    val hero = hole("Ah", "Kh")
    val v1 = hole("Qs", "Jd")
    val v2 = hole("9c", "8d")
    val fullBoard = board("2h", "3h", "4h", "5h", "9s")
    val estimate = HoldemEquity.equityMonteCarloMulti(
      hero,
      fullBoard,
      Seq(
        DiscreteDistribution(Map(v1 -> 1.0)),
        DiscreteDistribution(Map(v2 -> 1.0))
      ),
      trials = 20,
      rng = new Random(2)
    )
    assertEquals(estimate.mean, 1.0)
    assertEquals(estimate.winRate, 1.0)
  }

  test("equityExactMulti returns split share on full-board tie") {
    val hero = hole("2c", "3d")
    val v1 = hole("4h", "5c")
    val v2 = hole("6d", "7h")
    val fullBoard = board("As", "Ks", "Qs", "Js", "Ts")
    val result = HoldemEquity.equityExactMulti(
      hero,
      fullBoard,
      Seq(
        DiscreteDistribution(Map(v1 -> 1.0)),
        DiscreteDistribution(Map(v2 -> 1.0))
      )
    )
    assertEquals(result.share, 1.0 / 3.0)
    assertEquals(result.tie, 1.0)
    assertEquals(result.loss, 0.0)
  }

  test("equityExactMulti matches manual river enumeration on turn board") {
    val hero = hole("As", "Ks")
    val v1 = hole("Qh", "Jd")
    val v2 = hole("9c", "8d")
    val turnBoard = board("2c", "3d", "4h", "5s")

    val result = HoldemEquity.equityExactMulti(
      hero,
      turnBoard,
      Seq(
        DiscreteDistribution(Map(v1 -> 1.0)),
        DiscreteDistribution(Map(v2 -> 1.0))
      )
    )

    val dead = hero.asSet ++ v1.asSet ++ v2.asSet ++ turnBoard.asSet
    val remaining = Deck.full.filterNot(dead.contains)
    var win = 0.0
    var tie = 0.0
    var loss = 0.0
    var share = 0.0

    remaining.foreach { river =>
      val boardCards = turnBoard.cards :+ river
      val heroRank = HandEvaluator.evaluate7(hero.toVector ++ boardCards)
      val r1 = HandEvaluator.evaluate7(v1.toVector ++ boardCards)
      val r2 = HandEvaluator.evaluate7(v2.toVector ++ boardCards)
      val best = List(heroRank, r1, r2).max
      if heroRank == best then
        val tied = List(r1, r2).count(_ == best)
        if tied == 0 then win += 1.0 else tie += 1.0
        share += 1.0 / (tied + 1).toDouble
      else loss += 1.0
    }
    val total = remaining.size.toDouble
    assert(math.abs(result.win - (win / total)) < 1e-12)
    assert(math.abs(result.tie - (tie / total)) < 1e-12)
    assert(math.abs(result.loss - (loss / total)) < 1e-12)
    assert(math.abs(result.share - (share / total)) < 1e-12)
  }

  test("equityExactMulti matches manual enumeration on flop board") {
    val hero = hole("Ah", "Kh")
    val v1 = hole("Qs", "Jd")
    val v2 = hole("9c", "8d")
    val flopBoard = board("2h", "3h", "4h")

    val result = HoldemEquity.equityExactMulti(
      hero,
      flopBoard,
      Seq(
        DiscreteDistribution(Map(v1 -> 1.0)),
        DiscreteDistribution(Map(v2 -> 1.0))
      ),
      maxEvaluations = 1_000_000L
    )

    val dead = hero.asSet ++ v1.asSet ++ v2.asSet ++ flopBoard.asSet
    val remaining = Deck.full.filterNot(dead.contains).toIndexedSeq
    var win = 0.0
    var tie = 0.0
    var loss = 0.0
    var share = 0.0
    val totalCombos = combinationsCount(remaining.length, 2).toDouble

    combinations(remaining, 2).foreach { extra =>
      val boardCards = flopBoard.cards ++ extra
      val heroRank = HandEvaluator.evaluate7(hero.toVector ++ boardCards)
      val r1 = HandEvaluator.evaluate7(v1.toVector ++ boardCards)
      val r2 = HandEvaluator.evaluate7(v2.toVector ++ boardCards)
      val best = List(heroRank, r1, r2).max
      if heroRank == best then
        val tied = List(r1, r2).count(_ == best)
        if tied == 0 then win += 1.0 else tie += 1.0
        share += 1.0 / (tied + 1).toDouble
      else loss += 1.0
    }

    assert(math.abs(result.win - (win / totalCombos)) < 1e-12)
    assert(math.abs(result.tie - (tie / totalCombos)) < 1e-12)
    assert(math.abs(result.loss - (loss / totalCombos)) < 1e-12)
    assert(math.abs(result.share - (share / totalCombos)) < 1e-12)
  }

  test("equityMonteCarlo rejects non-positive trials") {
    val hero = hole("As", "Ks")
    val villain = hole("Qh", "Jd")
    val fullBoard = board("2c", "3d", "4h", "5s", "9c")
    val range = DiscreteDistribution(Map(villain -> 1.0))
    intercept[IllegalArgumentException] {
      HoldemEquity.equityMonteCarlo(hero, fullBoard, range, trials = 0, rng = new Random(1))
    }
  }

  test("equityMonteCarlo preflop supports accelerated batch path with cpu-emulated provider") {
    TestSystemPropertyScope.withSystemProperties(
      Seq(
        "sicfun.holdem.preflopEquityBackend" -> Some("batch"),
        "sicfun.gpu.provider" -> Some("cpu-emulated")
      )
    ) {
      val hero = hole("As", "Ks")
      val range = DiscreteDistribution(
        Map(
          hole("Qh", "Jd") -> 0.4,
          hole("9c", "8d") -> 0.6
        )
      )
      val estimate = HoldemEquity.equityMonteCarlo(
        hero = hero,
        board = Board.empty,
        villainRange = range,
        trials = 25,
        rng = new Random(11)
      )
      assert(estimate.mean >= 0.0 && estimate.mean <= 1.0)
      assert(math.abs((estimate.winRate + estimate.tieRate + estimate.lossRate) - 1.0) < 1e-6)
      assert(estimate.stderr >= 0.0)
    }
  }

  test("equityExactMulti rejects preflop boards and empty villain list") {
    val hero = hole("As", "Ks")
    val emptyRanges = Seq.empty[DiscreteDistribution[HoleCards]]
    intercept[IllegalArgumentException] {
      HoldemEquity.equityExactMulti(hero, Board.empty, emptyRanges)
    }
    val villain = DiscreteDistribution(Map(hole("Qh", "Jd") -> 1.0))
    intercept[IllegalArgumentException] {
      HoldemEquity.equityExactMulti(hero, Board.empty, Seq(villain))
    }
  }

  test("equity methods reject hero-board overlap") {
    val hero = hole("As", "Ks")
    val overlappedBoard = board("As", "2d", "3h")
    val villain = DiscreteDistribution(Map(hole("Qh", "Jd") -> 1.0))
    intercept[IllegalArgumentException] {
      HoldemEquity.equityExact(hero, overlappedBoard, villain)
    }
    intercept[IllegalArgumentException] {
      HoldemEquity.equityMonteCarlo(hero, overlappedBoard, villain, trials = 5, rng = new Random(2))
    }
  }

  test("equityMonteCarloMulti fails on impossible non-overlapping villain sampling") {
    val hero = hole("As", "Ks")
    val sameVillain = DiscreteDistribution(Map(hole("Qh", "Jd") -> 1.0))
    intercept[IllegalArgumentException] {
      HoldemEquity.equityMonteCarloMulti(
        hero,
        Board.empty,
        Seq(sameVillain, sameVillain),
        trials = 5,
        rng = new Random(3)
      )
    }
  }

  test("evCall rejects invalid numeric inputs") {
    intercept[IllegalArgumentException] {
      HoldemEquity.evCall(potBeforeCall = -1.0, callSize = 1.0, equity = 0.5)
    }
    intercept[IllegalArgumentException] {
      HoldemEquity.evCall(potBeforeCall = 1.0, callSize = -1.0, equity = 0.5)
    }
    intercept[IllegalArgumentException] {
      HoldemEquity.evCall(potBeforeCall = 1.0, callSize = 1.0, equity = 1.1)
    }
  }
