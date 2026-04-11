package sicfun.holdem.equity
import sicfun.holdem.types.*

import munit.FunSuite
import scala.util.Random

/**
  * Tests for the suit-isomorphic canonical heads-up equity table.
  *
  * Verifies:
  *   - Canonical key symmetry: keyFor(A,B).value == keyFor(B,A).value with opposite flip flags
  *   - Suit invariance: suit-relabeled matchups produce identical canonical keys
  *   - buildAll respects the maxMatchups limit on canonical key count
  *   - Determinism: same seed produces identical results regardless of parallelism level
  *
  * Uses low Monte Carlo trial counts and small matchup limits for fast test execution.
  */
class HeadsUpEquityCanonicalTableTest extends FunSuite:
  private val PreflopBackendProperty = "sicfun.holdem.preflopEquityBackend"

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(
      sicfun.core.Card.parse(a).getOrElse(fail(s"invalid card: $a")),
      sicfun.core.Card.parse(b).getOrElse(fail(s"invalid card: $b"))
    ))

  test("keyFor is symmetric and flips orientation") {
    val h1 = hole("As", "Ks")
    val h2 = hole("Qh", "Jd")
    val k1 = HeadsUpEquityCanonicalTable.keyFor(h1, h2)
    val k2 = HeadsUpEquityCanonicalTable.keyFor(h2, h1)
    assertEquals(k1.value, k2.value)
    assertEquals(k1.flipped, !k2.flipped)
  }

  test("keyFor is invariant under suit permutation") {
    val h1 = hole("As", "Ks")
    val v1 = hole("Qh", "Jh")
    val h2 = hole("Ah", "Kh")
    val v2 = hole("Qs", "Js")
    val k1 = HeadsUpEquityCanonicalTable.keyFor(h1, v1)
    val k2 = HeadsUpEquityCanonicalTable.keyFor(h2, v2)
    assertEquals(k1.value, k2.value)
    assertEquals(k1.flipped, k2.flipped)
  }

  test("buildAll limit is applied to canonical key count") {
    TestSystemPropertyScope.withSystemProperties(
      Vector(PreflopBackendProperty -> Some("cpu"))
    ) {
      val table = HeadsUpEquityCanonicalTable.buildAll(
        mode = HeadsUpEquityTable.Mode.MonteCarlo(8),
        rng = new Random(11L),
        maxMatchups = 50L,
        parallelism = 1
      )
      assertEquals(table.size, 50)
    }
  }

  test("buildAll MonteCarlo is deterministic across parallelism settings") {
    TestSystemPropertyScope.withSystemProperties(
      Vector(PreflopBackendProperty -> Some("cpu"))
    ) {
      val mode = HeadsUpEquityTable.Mode.MonteCarlo(12)
      val maxMatchups = 400L
      val seed = 23L
      val sequential = HeadsUpEquityCanonicalTable.buildAll(
        mode = mode,
        rng = new Random(seed),
        maxMatchups = maxMatchups,
        parallelism = 1
      )
      val parallel = HeadsUpEquityCanonicalTable.buildAll(
        mode = mode,
        rng = new Random(seed),
        maxMatchups = maxMatchups,
        parallelism = 4
      )
      assertEquals(sequential.values, parallel.values)
    }
  }
