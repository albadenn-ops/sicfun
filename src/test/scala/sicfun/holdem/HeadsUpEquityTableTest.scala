package sicfun.holdem

import munit.FunSuite

import scala.util.Random

class HeadsUpEquityTableTest extends FunSuite:
  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(
      sicfun.core.Card.parse(a).getOrElse(fail(s"invalid card: $a")),
      sicfun.core.Card.parse(b).getOrElse(fail(s"invalid card: $b"))
    ))

  test("HoleCardsIndex size is 1326") {
    assertEquals(HoleCardsIndex.size, 1326)
  }

  test("keyFor is symmetric and flips orientation") {
    val h1 = hole("As", "Ks")
    val h2 = hole("Qh", "Jd")
    val k1 = HeadsUpEquityTable.keyFor(h1, h2)
    val k2 = HeadsUpEquityTable.keyFor(h2, h1)
    assertEquals(k1.value, k2.value)
    assertEquals(k1.flipped, !k2.flipped)
  }

  test("keyFor rejects overlapping hands") {
    val h1 = hole("As", "Ks")
    val h2 = hole("As", "Qh")
    intercept[IllegalArgumentException] {
      HeadsUpEquityTable.keyFor(h1, h2)
    }
  }

  test("cache returns swapped equity when order is reversed") {
    val hero = hole("As", "Ks")
    val villain = hole("Qh", "Jd")
    val cache = HeadsUpEquityTable.cache(HeadsUpEquityTable.Mode.MonteCarlo(50), new Random(3))
    val first = cache.equity(hero, villain)
    val second = cache.equity(villain, hero)
    assertEquals(first.tie, second.tie)
    assertEquals(first.win, second.loss)
    assertEquals(first.loss, second.win)
  }

  test("cache remains orientation-correct when reverse lookup happens first") {
    val hero = hole("As", "Ks")
    val villain = hole("Qh", "Jd")
    val cache = HeadsUpEquityTable.cache(HeadsUpEquityTable.Mode.MonteCarlo(50), new Random(3))
    val first = cache.equity(villain, hero)
    val second = cache.equity(hero, villain)
    assertEquals(first.tie, second.tie)
    assertEquals(first.win, second.loss)
    assertEquals(first.loss, second.win)
  }

  test("buildAll MonteCarlo is deterministic across parallelism settings") {
    val mode = HeadsUpEquityTable.Mode.MonteCarlo(20)
    val maxMatchups = 1200L
    val seed = 19L
    val sequential = HeadsUpEquityTable.buildAll(
      mode = mode,
      rng = new Random(seed),
      maxMatchups = maxMatchups,
      parallelism = 1
    )
    val parallel = HeadsUpEquityTable.buildAll(
      mode = mode,
      rng = new Random(seed),
      maxMatchups = maxMatchups,
      parallelism = 4
    )
    assertEquals(sequential.values, parallel.values)
  }

  test("selectFullBatch rejects non-positive maxMatchups") {
    intercept[IllegalArgumentException] {
      HeadsUpEquityTable.selectFullBatch(0L)
    }
    intercept[IllegalArgumentException] {
      HeadsUpEquityTable.selectFullBatch(-5L)
    }
  }

  test("computeBatchCpu validates shape and parallelism contracts") {
    intercept[IllegalArgumentException] {
      HeadsUpEquityTable.computeBatchCpu(
        mode = HeadsUpEquityTable.Mode.MonteCarlo(5),
        packedKeys = Array(1L, 2L),
        keyMaterial = Array(1L),
        parallelism = 1,
        monteCarloSeedBase = 1L
      )
    }
    intercept[IllegalArgumentException] {
      HeadsUpEquityTable.computeBatchCpu(
        mode = HeadsUpEquityTable.Mode.MonteCarlo(5),
        packedKeys = Array.emptyLongArray,
        keyMaterial = Array.emptyLongArray,
        parallelism = 0,
        monteCarloSeedBase = 1L
      )
    }
  }

  test("compute backend parser rejects unknown backend") {
    intercept[IllegalArgumentException] {
      HeadsUpEquityTable.ComputeBackend.parse("tpu")
    }
  }
