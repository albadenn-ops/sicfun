package sicfun.holdem.equity
import sicfun.holdem.types.*

import munit.FunSuite
import scala.concurrent.duration.*
import scala.util.Random

/**
  * Tests for the suit-isomorphic canonical heads-up equity table.
  *
  * Verifies:
  *   - Canonical key symmetry: keyFor(A,B).value == keyFor(B,A).value with opposite flip flags
  *   - Suit invariance: suit-relabeled matchups produce identical canonical keys
  *   - 4-suit cycle invariance: a non-trivial permutation of every suit preserves the key
  *   - Card-order invariance: swapping cards within a hand does not move the key
  *   - Disjoint precondition: overlapping hands raise IllegalArgumentException
  *   - flipIfNeeded contract: involutive when flipped=true, identity when flipped=false
  *   - buildAll respects the maxMatchups limit on canonical key count
  *   - Determinism: same seed produces identical results regardless of parallelism level
  *
  * Uses low Monte Carlo trial counts and small matchup limits for fast test execution.
  */
class HeadsUpEquityCanonicalTableTest extends FunSuite:
  override val munitTimeout: Duration = 90.seconds

  private val PreflopBackendProperty = "sicfun.holdem.preflopEquityBackend"

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(
      card(a),
      card(b)
    ))

  private def card(token: String): sicfun.core.Card =
    sicfun.core.Card.parse(token).getOrElse(fail(s"invalid card: $token"))

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

  test("keyFor rejects matchups whose hands share a card") {
    val hero = hole("As", "Ks")
    val sharedAs = HoleCards(card("As"), card("Qd"))
    val ex = intercept[IllegalArgumentException] {
      HeadsUpEquityCanonicalTable.keyFor(hero, sharedAs)
    }
    assert(
      ex.getMessage.contains("non-overlapping"),
      s"unexpected message: ${ex.getMessage}"
    )
  }

  test("keyFor is invariant under a non-trivial 4-suit cycle") {
    // Suit cycle s -> c, h -> s, d -> h, c -> d (every suit moves)
    val k1 = HeadsUpEquityCanonicalTable.keyFor(hole("As", "Kh"), hole("Qd", "Jc"))
    val k2 = HeadsUpEquityCanonicalTable.keyFor(hole("Ac", "Ks"), hole("Qh", "Jd"))
    assertEquals(k1.value, k2.value, "canonical key must ignore the global suit relabel")
    assertEquals(k1.flipped, k2.flipped, "flip flag must follow the same matchup orientation")
  }

  test("keyFor is invariant under card order within each hand") {
    // Build hands with the case-class ctor (no normalization) so we can swap order.
    val heroA = HoleCards(card("As"), card("Kh"))
    val heroB = HoleCards(card("Kh"), card("As"))
    val villainA = HoleCards(card("Qd"), card("Jc"))
    val villainB = HoleCards(card("Jc"), card("Qd"))
    val baseline = HeadsUpEquityCanonicalTable.keyFor(heroA, villainA)
    val heroSwap = HeadsUpEquityCanonicalTable.keyFor(heroB, villainA)
    val villainSwap = HeadsUpEquityCanonicalTable.keyFor(heroA, villainB)
    val bothSwap = HeadsUpEquityCanonicalTable.keyFor(heroB, villainB)
    assertEquals(heroSwap.value, baseline.value, "swapping hero cards must not change the key")
    assertEquals(villainSwap.value, baseline.value, "swapping villain cards must not change the key")
    assertEquals(bothSwap.value, baseline.value, "swapping both must not change the key")
    assertEquals(heroSwap.flipped, baseline.flipped, "flip flag must be insensitive to card order")
    assertEquals(villainSwap.flipped, baseline.flipped, "flip flag must be insensitive to card order")
    assertEquals(bothSwap.flipped, baseline.flipped, "flip flag must be insensitive to card order")
  }

  test("flipIfNeeded swaps win/loss and is involutive when flipped=true") {
    val r = EquityResultWithError(win = 0.42, tie = 0.04, loss = 0.54, stderr = 0.001)
    val once = HeadsUpEquityCanonicalTable.flipIfNeeded(r, flipped = true)
    assertEquals(once.win, r.loss, "single flip swaps win <-> loss")
    assertEquals(once.loss, r.win, "single flip swaps win <-> loss")
    assertEquals(once.tie, r.tie, "tie is symmetric, must not change")
    assertEquals(once.stderr, r.stderr, "stderr is symmetric, must not change")
    val twice = HeadsUpEquityCanonicalTable.flipIfNeeded(once, flipped = true)
    assertEquals(twice, r, "flipping twice must be identity")
  }

  test("flipIfNeeded with flipped=false is the identity") {
    val r = EquityResultWithError(win = 0.42, tie = 0.04, loss = 0.54, stderr = 0.001)
    assertEquals(HeadsUpEquityCanonicalTable.flipIfNeeded(r, flipped = false), r)
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
