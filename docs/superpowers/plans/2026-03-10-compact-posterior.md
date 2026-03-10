# Compact Posterior Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the `Map[HoleCards, Double]` round-trip between `HoldemBayesProvider` and `HoldemEquity` by threading a flat-array `CompactPosterior` through the decision pipeline.

**Architecture:** `CompactPosterior` stores the Bayesian posterior as parallel `Array[HoleCards]` + `Array[Int]` (Prob weights). Built directly from the flat arrays already computed in `scalaUpdate`/`nativeUpdate`. Equity methods consume it without Map allocation. Non-hot consumers access a lazy `DiscreteDistribution` materialized on demand.

**Tech Stack:** Scala 3.8.1, munit 1.2.2, existing `Prob` opaque type (Int32 @ 2^30)

**Spec:** `docs/superpowers/specs/2026-03-10-compact-posterior-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/main/scala/sicfun/holdem/equity/HoldemEquity.scala` | Modify | Add `CompactPosterior` type, `buildCompactPosterior`, `prepareRangeProbFromCompact`, `prepareRangeFromCompact`, equity overloads |
| `src/main/scala/sicfun/holdem/provider/HoldemBayesProvider.scala` | Modify | Add `compact` field to `UpdateResult`, build `CompactPosterior` in `scalaUpdate`/`nativeUpdate` |
| `src/main/scala/sicfun/holdem/engine/RangeInferenceEngine.scala` | Modify | Add `compact` to `PosteriorInferenceResult`, thread through `computePosterior`, use in `recommendActionWithCompact` |
| `src/main/scala/sicfun/holdem/engine/RealTimeAdaptiveEngine.scala` | Modify | Pass `compact = None` in `lowLatencyPosteriorByPosition`, thread compact through `buildRecommendationOutcome` |
| `src/test/scala/sicfun/holdem/engine/RangeInferenceEngineTest.scala` | Modify | Pass `compact = None` in test construction of `PosteriorInferenceResult` |
| `src/test/scala/sicfun/holdem/provider/HoldemBayesProviderTest.scala` | Modify | Update `UpdateResult` construction to include `compact` field |
| `src/test/scala/sicfun/holdem/equity/CompactPosteriorTest.scala` | Create | Unit tests for `CompactPosterior` type and `buildCompactPosterior` |
| `src/test/scala/sicfun/holdem/equity/CompactPosteriorEquityTest.scala` | Create | Correctness: compact path vs Map path equity results |
| `src/test/scala/sicfun/holdem/provider/CompactPosteriorBayesIntegrationTest.scala` | Create | Integration: Bayes → CompactPosterior → Equity round-trip |

---

## Chunk 1: CompactPosterior Type + Builder

### Task 1: Define CompactPosterior and buildCompactPosterior

**Files:**
- Modify: `src/main/scala/sicfun/holdem/equity/HoldemEquity.scala:30-38`
- Test: `src/test/scala/sicfun/holdem/equity/CompactPosteriorTest.scala`

- [ ] **Step 1: Write failing tests for CompactPosterior**

Create `src/test/scala/sicfun/holdem/equity/CompactPosteriorTest.scala`:

```scala
package sicfun.holdem.equity

import munit.FunSuite
import sicfun.core.{Card, DiscreteDistribution, Prob}
import Prob.*
import sicfun.holdem.types.*

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
    // Prob.fromDouble(0.6).raw ≈ 644245094, Prob.fromDouble(0.4).raw ≈ 429496730
    assert(math.abs(Prob(compact.probWeights(0)).toDouble - 0.6) < 1e-8)
    assert(math.abs(Prob(compact.probWeights(1)).toDouble - 0.4) < 1e-8)
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

    assert(math.abs(Prob(compact.probWeights(0)).toDouble - 0.3) < 1e-8)
    assert(math.abs(Prob(compact.probWeights(1)).toDouble - 0.7) < 1e-8)
  }

  test("buildCompactPosterior fails on all-zero posterior") {
    val hypotheses = Vector(hole("As", "Ks"))
    val posterior = Array(0.0)

    interceptMessage[IllegalArgumentException]("all-zero posterior") {
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

    assert(math.abs(dist.probabilityOf(h1) - 0.6) < 1e-8)
    assert(math.abs(dist.probabilityOf(h2) - 0.4) < 1e-8)
  }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly sicfun.holdem.equity.CompactPosteriorTest"`
Expected: Compilation error — `CompactPosterior` and `buildCompactPosterior` don't exist yet.

- [ ] **Step 3: Implement CompactPosterior type and builder**

In `src/main/scala/sicfun/holdem/equity/HoldemEquity.scala`, add after the `PreparedRangeProb` definition (after line 34):

```scala
  /** Flat-array posterior for the Bayes → Equity hot path.
    * Built from arrays already computed in HoldemBayesProvider.scalaUpdate/nativeUpdate.
    * Equity methods consume this directly without Map allocation.
    * Non-hot consumers access the lazy `distribution` field.
    */
  final class CompactPosterior(
      val hands: Array[HoleCards],
      val probWeights: Array[Int], // Prob raw values (Int32 @ 2^30), normalized
      val size: Int
  ):
    lazy val distribution: DiscreteDistribution[HoleCards] =
      val builder = Map.newBuilder[HoleCards, Double]
      builder.sizeHint(size)
      var i = 0
      while i < size do
        builder += hands(i) -> Prob(probWeights(i)).toDouble
        i += 1
      DiscreteDistribution(builder.result())

  /** Builds a CompactPosterior from flat arrays (as produced by Bayesian update).
    * Skips zero-weight hypotheses. Normalizes weights to sum to Prob.Scale.
    */
  def buildCompactPosterior(
      hypotheses: Vector[HoleCards],
      posterior: Array[Double]
  ): CompactPosterior =
    require(hypotheses.length == posterior.length,
      s"hypotheses.length (${hypotheses.length}) != posterior.length (${posterior.length})")
    var total = 0.0
    var positiveCount = 0
    var i = 0
    while i < hypotheses.length do
      val w = math.max(0.0, posterior(i))
      if w > 0.0 then
        total += w
        positiveCount += 1
      i += 1
    require(total > 0.0, "all-zero posterior")
    val invTotal = 1.0 / total
    val hands = new Array[HoleCards](positiveCount)
    val weights = new Array[Int](positiveCount)
    var j = 0
    i = 0
    while i < hypotheses.length do
      val w = math.max(0.0, posterior(i))
      if w > 0.0 then
        hands(j) = hypotheses(i)
        weights(j) = Prob.fromDouble(w * invTotal).raw
        j += 1
      i += 1
    new CompactPosterior(hands, weights, positiveCount)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `sbt "testOnly sicfun.holdem.equity.CompactPosteriorTest"`
Expected: All 5 tests PASS.

- [ ] **Step 5: Run existing tests to verify no regressions**

Run: `sbt "testOnly sicfun.holdem.equity.*"`
Expected: All existing equity tests PASS (HoldemEquityTest + ProbEquityCorrectnessTest + CompactPosteriorTest).

- [ ] **Step 6: Commit**

```bash
git add src/main/scala/sicfun/holdem/equity/HoldemEquity.scala src/test/scala/sicfun/holdem/equity/CompactPosteriorTest.scala
git commit -m "feat: add CompactPosterior type and buildCompactPosterior builder"
```

---

## Chunk 2: Equity Overloads (CompactPosterior → PreparedRangeProb/PreparedRange)

### Task 2: Add prepareRangeProbFromCompact and prepareRangeFromCompact

**Files:**
- Modify: `src/main/scala/sicfun/holdem/equity/HoldemEquity.scala:767-811` (near existing prepareRangeProb)
- Test: `src/test/scala/sicfun/holdem/equity/CompactPosteriorEquityTest.scala`

- [ ] **Step 1: Write failing correctness tests**

Create `src/test/scala/sicfun/holdem/equity/CompactPosteriorEquityTest.scala`:

```scala
package sicfun.holdem.equity

import munit.FunSuite
import sicfun.core.{Card, DiscreteDistribution}
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
    val weights = Array(sicfun.core.Prob.Prob.fromDouble(1.0).raw)
    val compact = new HoldemEquity.CompactPosterior(hands, weights, 1)

    // Compare against Map path with canonical hand
    val range = DiscreteDistribution(Map(canonical -> 1.0))
    val fromMap = HoldemEquity.equityExactProb(hero, b, range)
    val fromCompact = HoldemEquity.equityExactProb(hero, b, compact)

    assertEqualsDouble(fromCompact.equity, fromMap.equity, tolerance)
  }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `sbt "testOnly sicfun.holdem.equity.CompactPosteriorEquityTest"`
Expected: Compilation error — `equityExactProb(HoleCards, Board, CompactPosterior)` overload doesn't exist.

- [ ] **Step 3: Implement prepareRangeProbFromCompact**

In `HoldemEquity.scala`, add after `prepareRangeProb` (after line 811):

```scala
  /** Like prepareRangeProb but reads from CompactPosterior flat arrays.
    * Canonicalizes, deduplicates, filters dead cards, and re-normalizes.
    */
  private def prepareRangeProbFromCompact(
      compact: CompactPosterior,
      dead: Set[Card]
  ): PreparedRangeProb =
    val weightScratch = preparedRangeWeightScratch.get()
    val touchedIds = preparedRangeTouchedIdsScratch.get()
    var touchedCount = 0
    try
      var i = 0
      while i < compact.size do
        val hand = compact.hands(i)
        val weight = compact.probWeights(i)
        if weight > 0 &&
          !dead.contains(hand.first) &&
          !dead.contains(hand.second)
        then
          val canonical = HoleCards.canonical(hand.first, hand.second)
          val handId = HoleCardsIndex.fastIdOf(canonical)
          if weightScratch(handId) == 0.0 then
            touchedIds(touchedCount) = handId
            touchedCount += 1
          weightScratch(handId) += weight.toDouble // accumulate as Double for dedup merge
        i += 1
      require(touchedCount > 0, "villain range is empty after filtering")
      java.util.Arrays.sort(touchedIds, 0, touchedCount)

      val hands = new Array[HoleCards](touchedCount)
      val probWeights = new Array[Int](touchedCount)
      var total = 0.0
      i = 0
      while i < touchedCount do
        total += weightScratch(touchedIds(i))
        i += 1
      require(total > 0.0, "villain range is empty after filtering")
      val invTotal = 1.0 / total
      i = 0
      while i < touchedCount do
        val handId = touchedIds(i)
        hands(i) = HoleCardsIndex.byIdUnchecked(handId)
        probWeights(i) = Prob.fromDouble(weightScratch(handId) * invTotal).raw
        i += 1
      PreparedRangeProb(hands, probWeights, touchedCount)
    finally
      var i = 0
      while i < touchedCount do
        weightScratch(touchedIds(i)) = 0.0
        i += 1
```

- [ ] **Step 4: Implement prepareRangeFromCompact**

In `HoldemEquity.scala`, add after `prepareRangeProbFromCompact`:

```scala
  /** Like prepareRange but reads from CompactPosterior flat arrays.
    * For Monte Carlo path which needs Double weights.
    */
  private def prepareRangeFromCompact(
      compact: CompactPosterior,
      deadMaskValue: Long
  ): PreparedRange =
    val weightScratch = preparedRangeWeightScratch.get()
    val touchedIds = preparedRangeTouchedIdsScratch.get()
    var touchedCount = 0
    try
      var i = 0
      while i < compact.size do
        val hand = compact.hands(i)
        val weight = Prob(compact.probWeights(i)).toDouble
        if weight > 0.0 && ((handMask(hand) & deadMaskValue) == 0L) then
          val handId = HoleCardsIndex.fastIdOf(hand)
          if weightScratch(handId) == 0.0 then
            touchedIds(touchedCount) = handId
            touchedCount += 1
          weightScratch(handId) += weight
        i += 1
      require(touchedCount > 0, "villain range is empty after filtering")
      java.util.Arrays.sort(touchedIds, 0, touchedCount)

      val hands = new Array[HoleCards](touchedCount)
      val weights = new Array[Double](touchedCount)
      val handIds = new Array[Int](touchedCount)
      var total = 0.0
      i = 0
      while i < touchedCount do
        val handId = touchedIds(i)
        val weight = weightScratch(handId)
        hands(i) = HoleCardsIndex.byIdUnchecked(handId)
        weights(i) = weight
        handIds(i) = handId
        total += weight
        i += 1
      require(total > 0.0, "villain range is empty after filtering")
      val invTotal = 1.0 / total
      i = 0
      while i < weights.length do
        weights(i) *= invTotal
        i += 1
      PreparedRange(hands, weights, handIds)
    finally
      var i = 0
      while i < touchedCount do
        weightScratch(touchedIds(i)) = 0.0
        i += 1
```

- [ ] **Step 5: Add equityExactProb overload for CompactPosterior**

In `HoldemEquity.scala`, add after the existing `equityExactProb` method (after line 129):

```scala
  /** Fixed-point equity using CompactPosterior — bypasses Map entirely. */
  def equityExactProb(
      hero: HoleCards,
      board: Board,
      compact: CompactPosterior
  ): EquityResult =
    validateHeroBoard(hero, board)
    val dead = hero.asSet ++ board.asSet
    val prepared = prepareRangeProbFromCompact(compact, dead)
    val missing = board.missing

    var winL = 0L
    var tieL = 0L
    var lossL = 0L

    var i = 0
    while i < prepared.size do
      val villain = prepared.hands(i)
      val weightRaw = prepared.weights(i)
      if weightRaw > 0 then
        val remaining = Deck.full.filterNot(card => dead.contains(card) || villain.contains(card)).toIndexedSeq
        val boardCount = combinationsCount(remaining.length, missing)
        val perBoardWeight = weightRaw.toLong / boardCount

        val boards =
          if missing == 0 then Iterator.single(Vector.empty[Card])
          else HoldemCombinator.combinations(remaining, missing)

        boards.foreach { extra =>
          val fullBoard = board.cards ++ extra
          val heroRank = HandEvaluator.evaluate7Cached(hero.toVector ++ fullBoard)
          val villainRank = HandEvaluator.evaluate7Cached(villain.toVector ++ fullBoard)
          val cmp = heroRank.compare(villainRank)
          if cmp > 0 then winL += perBoardWeight
          else if cmp == 0 then tieL += perBoardWeight
          else lossL += perBoardWeight
        }
      i += 1

    val total = winL + tieL + lossL
    if total == 0L then EquityResult(0.0, 0.0, 0.0)
    else
      val invTotal = 1.0 / total.toDouble
      EquityResult(winL * invTotal, tieL * invTotal, lossL * invTotal)
```

- [ ] **Step 6: Add equityMonteCarlo overload for CompactPosterior**

In `HoldemEquity.scala`, add after the existing `equityMonteCarlo(DiscreteDistribution)` method (after line 338):

```scala
  /** Monte Carlo equity using CompactPosterior — bypasses Map entirely. */
  def equityMonteCarlo(
      hero: HoleCards,
      board: Board,
      compact: CompactPosterior,
      trials: Int,
      rng: Random = new Random()
  ): EquityEstimate =
    validateHeroBoard(hero, board)
    require(trials > 0, "trials must be positive")
    val deadMaskValue = deadMask(hero, board)
    val preparedRange = prepareRangeFromCompact(compact, deadMaskValue)
    maybeAcceleratedMonteCarlo(hero, board, preparedRange, trials, rng) match
      case Some(estimate) =>
        estimate
      case None =>
        // Delegate to existing MC implementation via distribution fallback
        equityMonteCarlo(hero, board, compact.distribution, trials, rng)
```

Note: The MC inner loop is identical to the existing one. Rather than duplicating ~60 lines, the non-accelerated path falls back to `compact.distribution`. The accelerated path (GPU) already receives a `PreparedRange` so it gets the flat-array benefit. If benchmarks show the fallback matters, we can duplicate the inner loop in a follow-up.

- [ ] **Step 7: Run tests to verify they pass**

Run: `sbt "testOnly sicfun.holdem.equity.CompactPosterior*"`
Expected: All CompactPosteriorTest + CompactPosteriorEquityTest PASS.

- [ ] **Step 8: Run full equity test suite**

Run: `sbt "testOnly sicfun.holdem.equity.*"`
Expected: All PASS (no regressions).

- [ ] **Step 9: Commit**

```bash
git add src/main/scala/sicfun/holdem/equity/HoldemEquity.scala src/test/scala/sicfun/holdem/equity/CompactPosteriorEquityTest.scala
git commit -m "feat: add equity overloads accepting CompactPosterior, bypassing Map"
```

---

## Chunk 3: Wire HoldemBayesProvider to Produce CompactPosterior

### Task 3: Update UpdateResult and scalaUpdate/nativeUpdate

**Files:**
- Modify: `src/main/scala/sicfun/holdem/provider/HoldemBayesProvider.scala:24-28,265-352`
- Test: `src/test/scala/sicfun/holdem/provider/CompactPosteriorBayesIntegrationTest.scala`

- [ ] **Step 1: Write failing integration test**

Create `src/test/scala/sicfun/holdem/provider/CompactPosteriorBayesIntegrationTest.scala`:

```scala
package sicfun.holdem.provider

import munit.FunSuite
import sicfun.core.{DiscreteDistribution, MultinomialLogistic, Prob}
import Prob.*
import sicfun.holdem.types.*
import sicfun.holdem.model.*
import sicfun.holdem.equity.HoldemEquity

class CompactPosteriorBayesIntegrationTest extends FunSuite:
  // Force Scala provider to avoid native/GPU dependencies
  override def beforeAll(): Unit =
    System.setProperty("sicfun.bayes.provider", "scala")
  override def afterAll(): Unit =
    System.clearProperty("sicfun.bayes.provider")
    HoldemBayesProvider.resetAutoProviderForTests()

  private def card(token: String) =
    sicfun.core.Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  private def trivialActionModel: PokerActionModel =
    val classCount = PokerAction.categories.length
    val featureCount = PokerFeatures.dimension
    val weights = Vector.tabulate(classCount)(c =>
      Vector.tabulate(featureCount)(f => 0.01 * (c + 1) * (f + 1))
    )
    val bias = Vector.tabulate(classCount)(c => 0.05 * c)
    PokerActionModel(
      logistic = MultinomialLogistic(weights, bias),
      categoryIndex = PokerActionModel.defaultCategoryIndex,
      featureDimension = PokerFeatures.dimension
    )

  test("UpdateResult.compact is present and matches posterior") {
    val h1 = hole("As", "Ks")
    val h2 = hole("Qh", "Jd")
    val h3 = hole("Tc", "9c")
    val prior = DiscreteDistribution(Map(h1 -> 0.4, h2 -> 0.3, h3 -> 0.3))
    val state = GameState(
      street = Street.Preflop,
      board = Board.empty,
      pot = 8.0, toCall = 2.0,
      position = Position.Button,
      stackSize = 100.0,
      betHistory = Vector.empty
    )
    val observations = Seq(PokerAction.Call -> state)

    val result = HoldemBayesProvider.updatePosterior(prior, observations, trivialActionModel)

    // compact must be present
    val compact = result.compact
    assert(compact.size > 0)
    assert(compact.size <= 3)

    // compact.distribution must match result.posterior
    val compactDist = compact.distribution
    prior.support.foreach { hand =>
      assertEqualsDouble(
        compactDist.probabilityOf(hand),
        result.posterior.probabilityOf(hand),
        1e-7
      )
    }
  }

  test("compact posterior produces same equity as Map posterior") {
    val hero = hole("Ah", "Kh")
    val h1 = hole("Jc", "Jd")
    val h2 = hole("9s", "8s")
    val prior = DiscreteDistribution(Map(h1 -> 0.5, h2 -> 0.5))
    val state = GameState(
      street = Street.Flop,
      board = Board.from(Seq(card("2c"), card("7d"), card("Ts"))),
      pot = 10.0, toCall = 3.0,
      position = Position.Button,
      stackSize = 100.0,
      betHistory = Vector.empty
    )
    val observations = Seq(PokerAction.Raise(6.0) -> state)

    val result = HoldemBayesProvider.updatePosterior(prior, observations, trivialActionModel)
    val b = state.board

    val fromMap = HoldemEquity.equityExactProb(hero, b, result.posterior)
    val fromCompact = HoldemEquity.equityExactProb(hero, b, result.compact)

    assertEqualsDouble(fromCompact.equity, fromMap.equity, 1e-3)
  }

  test("empty observations returns compact matching prior") {
    val h1 = hole("As", "Ks")
    val prior = DiscreteDistribution(Map(h1 -> 1.0))

    val result = HoldemBayesProvider.updatePosterior(prior, Seq.empty, trivialActionModel)

    assertEquals(result.compact.size, 1)
    assert(math.abs(Prob(result.compact.probWeights(0)).toDouble - 1.0) < 1e-8)
  }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `sbt "testOnly sicfun.holdem.provider.CompactPosteriorBayesIntegrationTest"`
Expected: Compilation error — `UpdateResult` doesn't have `compact` field.

- [ ] **Step 3: Update UpdateResult to include compact field**

In `HoldemBayesProvider.scala`, modify the `UpdateResult` case class (lines 24-28):

```scala
  final case class UpdateResult(
      posterior: DiscreteDistribution[HoleCards],
      compact: HoldemEquity.CompactPosterior,
      logEvidence: Double,
      provider: Provider
  )
```

Add import at top of file: `import sicfun.holdem.equity.HoldemEquity`

- [ ] **Step 4: Update scalaUpdate to build CompactPosterior**

In `HoldemBayesProvider.scala`, replace lines 342-352 (the Map building and return):

```scala
    val compact = HoldemEquity.buildCompactPosterior(hypotheses, posterior)
    UpdateResult(
      posterior = compact.distribution,
      compact = compact,
      logEvidence = logEvidence,
      provider = provider
    )
```

- [ ] **Step 5: Update nativeUpdate to build CompactPosterior**

In `HoldemBayesProvider.scala`, replace lines 290-302 (inside the `Right(_)` branch):

```scala
      case Right(_) =>
        val clamped = new Array[Double](hypotheses.length)
        var idx = 0
        while idx < hypotheses.length do
          clamped(idx) = math.max(0.0, outPosterior(idx))
          idx += 1
        val compact = HoldemEquity.buildCompactPosterior(hypotheses, clamped)
        Some(
          UpdateResult(
            posterior = compact.distribution,
            compact = compact,
            logEvidence = outLogEvidence(0),
            provider = selectedProvider
          )
        )
```

- [ ] **Step 6: Update empty-observations path**

In `HoldemBayesProvider.scala`, update the `observations.isEmpty` branch (lines 80-85) to also produce a `CompactPosterior`:

```scala
    if observations.isEmpty then
      val normalized = prior.normalized
      val hypotheses = normalized.weights.keysIterator.toVector
      val posteriorArray = hypotheses.map(normalized.probabilityOf).toArray
      val compact = HoldemEquity.buildCompactPosterior(hypotheses, posteriorArray)
      UpdateResult(
        posterior = normalized,
        compact = compact,
        logEvidence = 0.0,
        provider = Provider.Scala
      )
```

- [ ] **Step 7: Update HoldemBayesProviderTest.scala for UpdateResult breaking change**

In `src/test/scala/sicfun/holdem/provider/HoldemBayesProviderTest.scala`, find all direct `UpdateResult` constructions (lines ~120 and ~131 — the `computePosteriorDrift` test). Add `compact` field by building a `CompactPosterior` from the same data. For each `UpdateResult(posterior = dist, logEvidence = ..., provider = ...)`, change to:

```scala
val hypotheses = dist.weights.keysIterator.toVector
val posteriorArray = hypotheses.map(dist.probabilityOf).toArray
val compact = HoldemEquity.buildCompactPosterior(hypotheses, posteriorArray)
HoldemBayesProvider.UpdateResult(
  posterior = dist,
  compact = compact,
  logEvidence = ...,
  provider = ...
)
```

Add import: `import sicfun.holdem.equity.HoldemEquity`

- [ ] **Step 8: Run integration tests**

Run: `sbt "testOnly sicfun.holdem.provider.*"`
Expected: All PASS (both new and existing tests).

- [ ] **Step 9: Commit**

```bash
git add src/main/scala/sicfun/holdem/provider/HoldemBayesProvider.scala src/test/scala/sicfun/holdem/provider/CompactPosteriorBayesIntegrationTest.scala
git commit -m "feat: HoldemBayesProvider produces CompactPosterior, eliminates Map in scalaUpdate/nativeUpdate"
```

---

## Chunk 4: Thread CompactPosterior Through RangeInferenceEngine

### Task 4: Update PosteriorInferenceResult and computePosterior

**Files:**
- Modify: `src/main/scala/sicfun/holdem/engine/RangeInferenceEngine.scala:28-44,234-304,600-687`

- [ ] **Step 1: Update PosteriorInferenceResult to carry Optional CompactPosterior**

In `RangeInferenceEngine.scala`, modify `PosteriorInferenceResult` (lines 28-35):

```scala
final class PosteriorInferenceResult private (
    val prior: DiscreteDistribution[HoleCards],
    val posterior: DiscreteDistribution[HoleCards],
    val compact: Option[HoldemEquity.CompactPosterior],
    val logEvidence: Double,
    collapseThunk: () => PosteriorCollapse
):
  lazy val collapse: PosteriorCollapse = collapseThunk()
```

Update the companion `apply` (lines 37-44):

```scala
object PosteriorInferenceResult:
  def apply(
      prior: DiscreteDistribution[HoleCards],
      posterior: DiscreteDistribution[HoleCards],
      compact: Option[HoldemEquity.CompactPosterior],
      logEvidence: Double,
      collapse: => PosteriorCollapse
  ): PosteriorInferenceResult =
    new PosteriorInferenceResult(prior, posterior, compact, logEvidence, () => collapse)
```

Add import: `import sicfun.holdem.equity.HoldemEquity`

- [ ] **Step 2: Update computePosterior to thread compact**

In `RangeInferenceEngine.scala`, modify `computePosterior` (lines 276-304). Replace the `val (posterior, logEvidence)` block and `PosteriorInferenceResult` construction:

```scala
    val (posterior, logEvidence, compact) =
      if observationsForBayes.isEmpty then
        (resolveDecisionPosteriorFor(normalizedPrior), 0.0, None)
      else if canSkipBayesUpdate then
        (resolveDecisionPosteriorFor(normalizedPrior), 0.0, None)
      else
        val bayesUpdate = HoldemBayesProvider.updatePosterior(
          prior = normalizedPrior,
          observations = observationsForBayes,
          actionModel = actionModel
        )
        val usesCompact =
          ddreConfig.mode == HoldemDdreProvider.Mode.Off ||
            ddreConfig.mode == HoldemDdreProvider.Mode.Shadow
        val resolved = resolveDecisionPosteriorFor(bayesUpdate.posterior)
        val compactOption = if usesCompact then Some(bayesUpdate.compact) else None
        (resolved, bayesUpdate.logEvidence, compactOption)

    PosteriorInferenceResult(
      normalizedPrior,
      posterior,
      compact,
      logEvidence,
      {
        val collapseSummary = CollapseMetrics.summary(normalizedPrior, posterior)
        PosteriorCollapse(
          entropyReduction = collapseSummary.entropyReduction,
          klDivergence = collapseSummary.klDivergence,
          effectiveSupportPrior = collapseSummary.effectiveSupportPrior,
          effectiveSupportPosterior = collapseSummary.effectiveSupportPosterior,
          collapseRatio = collapseSummary.collapseRatio
        )
      }
    )
```

- [ ] **Step 3: Add truncateCompact helper for top-k truncation**

The spec requires applying the same top-k truncation as `compactPosteriorForEquity` (max 256 hands, min 99.5% mass). Add a helper in `RangeInferenceEngine.scala`:

```scala
  /** Truncates a CompactPosterior to top-k hands by weight, mirroring compactPosteriorForEquity. */
  private def truncateCompact(
      cp: HoldemEquity.CompactPosterior,
      maxHands: Int,
      minMass: Double,
      minHands: Int
  ): HoldemEquity.CompactPosterior =
    if maxHands <= 0 || cp.size <= maxHands then cp
    else
      // Sort indices by weight descending
      val indices = Array.tabulate(cp.size)(identity)
      java.util.Arrays.sort(indices, (a: Int, b: Int) =>
        java.lang.Integer.compare(cp.probWeights(b), cp.probWeights(a))
      )
      val hands = new Array[HoleCards](math.min(maxHands, cp.size))
      val weights = new Array[Int](hands.length)
      var keptCount = 0
      var mass = 0.0
      var idx = 0
      while idx < indices.length &&
        keptCount < maxHands &&
        (keptCount < minHands || mass < minMass)
      do
        val i = indices(idx)
        val w = Prob.Prob(cp.probWeights(i)).toDouble
        if w > 0.0 then
          hands(keptCount) = cp.hands(i)
          weights(keptCount) = cp.probWeights(i)
          mass += w
          keptCount += 1
        idx += 1
      if keptCount == 0 then cp
      else new HoldemEquity.CompactPosterior(hands, weights, keptCount)
```

- [ ] **Step 4: Add recommendActionWithCompact**

In `RangeInferenceEngine.scala`, add a new private helper after the existing `recommendActionAssumeNormalized`:

```scala
  /** Variant that accepts an optional CompactPosterior for the equity fast path. */
  private[holdem] def recommendActionWithCompact(
      hero: HoleCards,
      state: GameState,
      posterior: DiscreteDistribution[HoleCards],
      compact: Option[HoldemEquity.CompactPosterior],
      candidateActions: Vector[PokerAction],
      actionValueModel: ActionValueModel = ActionValueModel.ChipEv(),
      villainResponseModel: Option[VillainResponseModel] = None,
      equityTrials: Int = 50_000,
      rng: Random = new Random()
  ): ActionRecommendation =
    require(candidateActions.nonEmpty, "candidateActions must be non-empty")
    require(equityTrials > 0, "equityTrials must be positive")

    // Apply top-k truncation on compact path (same as compactPosteriorForEquity)
    val maxHands = configuredEquityPosteriorMaxHands
    val minMass = configuredEquityPosteriorMinMass
    val minHands = math.max(1, math.min(maxHands, DefaultEquityPosteriorMinHands))

    val heroEquity = compact match
      case Some(cp) =>
        val truncated = truncateCompact(cp, maxHands, minMass, minHands)
        HoldemEquity.equityMonteCarlo(hero, state.board, truncated, equityTrials, rng)
      case None =>
        val equityPosterior = compactPosteriorForEquity(posterior)
        HoldemEquity.equityMonteCarlo(hero, state.board, equityPosterior, equityTrials, rng)

    // For response-aware raise EV, fall back to DiscreteDistribution
    // (VillainResponseModel needs per-hand Map iteration)
    val equityPosteriorForResponse = compact match
      case Some(cp) => cp.distribution
      case None => compactPosteriorForEquity(posterior)

    val evaluations = candidateActions.map { action =>
      val expectedValue =
        villainResponseModel match
          case Some(responseModel) =>
            action match
              case PokerAction.Raise(_) =>
                val raiseRng =
                  responseModel match
                    case _: UniformVillainResponseModel => rng
                    case _ => new Random(rng.nextLong())
                responseAwareRaiseEv(
                  hero = hero,
                  state = state,
                  posterior = equityPosteriorForResponse,
                  heroEquityMean = heroEquity.mean,
                  action = action,
                  responseModel = responseModel,
                  equityTrials = equityTrials,
                  rng = raiseRng
                )
              case _ =>
                actionValueModel.expectedValue(action, state, heroEquity.mean)
          case None =>
            actionValueModel.expectedValue(action, state, heroEquity.mean)
      ActionEvaluation(action = action, expectedValue = expectedValue)
    }

    val best = evaluations.reduceLeft { (a, b) =>
      if a.expectedValue >= b.expectedValue then a else b
    }
    ActionRecommendation(heroEquity, evaluations, best.action)
```

- [ ] **Step 5: Update inferAndRecommend to use compact path**

In `RangeInferenceEngine.scala`, modify `inferAndRecommend` (lines 829-868). Change the `recommendation` line to use `recommendActionWithCompact`:

Replace:
```scala
    val recommendation = recommendActionAssumeNormalized(
```
With:
```scala
    val recommendation = recommendActionWithCompact(
      hero = hero,
      state = state,
      posterior = posteriorInference.posterior,
      compact = posteriorInference.compact,
      candidateActions = candidateActions,
      actionValueModel = actionValueModel,
      villainResponseModel = villainResponseModel,
      equityTrials = equityTrials,
      rng = new Random(rng.nextLong())
    )
```

- [ ] **Step 6: Fix PosteriorInferenceResult callers — RealTimeAdaptiveEngine**

In `src/main/scala/sicfun/holdem/engine/RealTimeAdaptiveEngine.scala`, find `lowLatencyPosteriorByPosition` (line ~160) which constructs `PosteriorInferenceResult` without `compact`. Add `compact = None`:

```scala
PosteriorInferenceResult(
    prior = compacted,
    posterior = compacted,
    compact = None,
    logEvidence = 0.0,
    collapse = PosteriorCollapse(...)
)
```

Also update `buildRecommendationOutcome` (line ~454) to thread compact through. Change:
```scala
val baseRecommendation = RangeInferenceEngine.recommendActionAssumeNormalized(
```
To:
```scala
val baseRecommendation = RangeInferenceEngine.recommendActionWithCompact(
    hero = hero,
    state = state,
    posterior = posterior,
    compact = posteriorInference.compact,
    candidateActions = ...,
    ...
)
```
(Adjust parameter names to match what's in scope at that call site.)

- [ ] **Step 7: Fix PosteriorInferenceResult callers — RangeInferenceEngineTest**

In `src/test/scala/sicfun/holdem/engine/RangeInferenceEngineTest.scala`, find the test that constructs `PosteriorInferenceResult` (line ~81). Add `compact = None`:

```scala
val result = PosteriorInferenceResult(
    prior = dist,
    posterior = dist,
    compact = None,
    logEvidence = 0.0,
    collapse = { ... }
)
```

- [ ] **Step 8: Run full test suite**

Run: `sbt test`
Expected: All PASS. This verifies no regressions across the entire codebase.

- [ ] **Step 9: Commit**

```bash
git add src/main/scala/sicfun/holdem/engine/RangeInferenceEngine.scala src/main/scala/sicfun/holdem/engine/RealTimeAdaptiveEngine.scala src/test/scala/sicfun/holdem/engine/RangeInferenceEngineTest.scala
git commit -m "feat: thread CompactPosterior through RangeInferenceEngine to equity"
```

---

## Chunk 5: Verification and Benchmark

### Task 5: Full regression test + A/B benchmark

**Files:**
- No new files — use existing benchmark infrastructure

- [ ] **Step 1: Run full test suite with verbose output**

Run: `sbt test`
Expected: All tests PASS. Count total: should be >= 34 (12 ProbTest + 6 ProbEquityCorrectnessTest + 16 HoldemEquityTest + new tests).

- [ ] **Step 2: Verify the compact path is actually exercised in integration**

Add a temporary debug log in `recommendActionWithCompact` to confirm the `Some(cp)` branch is taken during `inferAndRecommend`. Run any existing integration test that calls `inferAndRecommend` (e.g., `RealTimeAdaptiveEngineTest` or `TexasHoldemPlayingHallTest`). Verify the log appears. Remove the debug log after verification.

- [ ] **Step 3: Commit all changes**

```bash
git add -u
git commit -m "test: verify CompactPosterior path exercised in integration"
```

- [ ] **Step 4: Update memory file**

Update `C:\Users\MK1\.claude\projects\C--Users-alexl-code-math-untitled\memory\fixed-point-prob.md` with:
- Phase 3 (Compact Posterior) status
- Benchmark results
- Any design decisions that changed during implementation
