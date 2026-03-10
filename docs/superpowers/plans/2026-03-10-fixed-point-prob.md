# Fixed-Point Prob Type — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create an Int32@2^30 opaque type `Prob` for [0,1] probabilities, wire it into the equity exact enumeration hot path, and benchmark against the current Double implementation.

**Architecture:** New opaque type `Prob = Int` in `sicfun.core` with inline operations. A parallel method `equityExactProb` in `HoldemEquity` uses Prob/Long accumulation internally but returns the existing `EquityResult` (Double) to avoid cascading API changes while measuring performance. A/B benchmark compares both paths.

**Deviation from spec:** The design spec proposes migrating `EquityResult` to `Prob` fields. This plan deliberately keeps `EquityResult` as Double and adds a parallel method instead. This enables clean A/B comparison with zero downstream breakage. If the benchmark is positive, EquityResult migration happens in Phase 2.

**Spec correction:** The spec's "accumulate raw, divide once at end" approach assumes a uniform boardCount across all villain hands. In reality, boardCount varies per villain (different dead cards change the remaining deck size). The plan correctly divides weight by boardCount per villain hand, matching the original Double code's semantics. The integer division `weight.raw / boardCount` introduces truncation error of at most 1 LSB per board evaluation (~4e-8 relative error for typical boardCounts), which is negligible for poker equity precision needs (~4 decimal places).

**Tech Stack:** Scala 3.8.1, munit 1.2.2, existing benchmark infrastructure in `sicfun.holdem.bench`

---

## Chunk 1: Prob Type + Tests

### Task 1: Create Prob opaque type

**Files:**
- Create: `src/main/scala/sicfun/core/Prob.scala`

- [ ] **Step 1: Write the Prob source file**

Follows the existing `CardId` opaque type pattern (opaque type nested inside companion object):

```scala
package sicfun.core

/** Fixed-point probability in [0, 1], stored as Int32 with 2^30 scale.
  *
  * 1.0 = 1,073,741,824 (2^30). Uses 2^30 (not 2^31) to leave the sign bit free,
  * so subtraction stays safe. Multiply uses Long intermediate to avoid overflow.
  * No clamping or validation — caller is responsible for keeping values in range.
  */
object Prob:
  opaque type Prob = Int

  inline val Scale = 1 << 30 // 1,073,741,824

  inline def apply(raw: Int): Prob = raw
  def fromDouble(d: Double): Prob = (d * Scale + 0.5).toInt

  inline val One: Prob = Scale
  inline val Zero: Prob = 0
  inline val Half: Prob = Scale >> 1

  extension (p: Prob)
    inline def raw: Int = p
    inline def toDouble: Double = p.toDouble / Scale

    inline def +(q: Prob): Prob = p + q
    inline def -(q: Prob): Prob = p - q

    /** Multiply two probabilities. Uses Long intermediate to avoid overflow. */
    inline def *(q: Prob): Prob = ((p.toLong * q.toLong) >> 30).toInt

    /** Divide by an integer (e.g. boardCount). Truncates. */
    inline def /(n: Int): Prob = p / n

    inline def >(q: Prob): Boolean = p > q
    inline def <(q: Prob): Boolean = p < q
    inline def >=(q: Prob): Boolean = p >= q
    inline def <=(q: Prob): Boolean = p <= q

export Prob.Prob
```

- [ ] **Step 2: Verify it compiles**

Run: `sbt compile 2>&1 | tail -5`
Expected: `[success]`

- [ ] **Step 3: Commit**

```bash
git add src/main/scala/sicfun/core/Prob.scala
git commit -m "feat: add Prob opaque type (Int32 @ 2^30 fixed-point)"
```

### Task 2: Test Prob type

**Files:**
- Create: `src/test/scala/sicfun/core/ProbTest.scala`

- [ ] **Step 1: Write the test file**

```scala
package sicfun.core

import Prob.*
import munit.FunSuite

class ProbTest extends FunSuite:
  private val tolerance = 1e-9

  test("Zero and One constants") {
    assertEquals(Prob.Zero.raw, 0)
    assertEquals(Prob.One.raw, 1 << 30)
    assertEquals(Prob.Half.raw, 1 << 29)
  }

  test("fromDouble roundtrip for common values") {
    val cases = Seq(0.0, 0.25, 0.5, 0.75, 1.0, 1.0 / 3.0, 1.0 / 1326.0)
    cases.foreach { d =>
      val p = Prob.fromDouble(d)
      val back = p.toDouble
      assertEqualsDouble(back, d, tolerance, s"roundtrip failed for $d")
    }
  }

  test("addition is exact for small values") {
    val a = Prob.fromDouble(0.3)
    val b = Prob.fromDouble(0.7)
    val sum = a + b
    assert(math.abs(sum.raw - Prob.One.raw) <= 1, s"sum.raw=${sum.raw} vs One=${Prob.One.raw}")
  }

  test("subtraction") {
    val a = Prob.One
    val b = Prob.fromDouble(0.4)
    val diff = a - b
    assertEqualsDouble(diff.toDouble, 0.6, tolerance)
  }

  test("subtraction can produce negative raw value") {
    // Prob does not clamp to [0,1] — caller is responsible
    val result = Prob.Zero - Prob.One
    assert(result.raw < 0)
  }

  test("multiplication of two probabilities") {
    val a = Prob.fromDouble(0.5)
    val b = Prob.fromDouble(0.5)
    val product = a * b
    assertEqualsDouble(product.toDouble, 0.25, tolerance)
  }

  test("multiplication does not overflow for One * One") {
    val product = Prob.One * Prob.One
    assertEquals(product.raw, Prob.One.raw)
  }

  test("multiplication of very small probabilities rounds to zero") {
    // (1/1326)^2 ≈ 5.7e-7 → raw ≈ 0.6, truncates to 0
    val tiny = Prob.fromDouble(1.0 / 1326.0)
    val product = tiny * tiny
    assert(product.raw >= 0 && product.raw <= 1,
      s"expected ~0, got ${product.raw}")
  }

  test("division by integer") {
    val p = Prob.fromDouble(0.9)
    val divided = p / 3
    assertEqualsDouble(divided.toDouble, 0.3, 1e-7)
  }

  test("comparison operators") {
    val a = Prob.fromDouble(0.3)
    val b = Prob.fromDouble(0.7)
    assert(a < b)
    assert(b > a)
    assert(a <= a)
    assert(a >= a)
  }

  test("fromDouble precision for 1/1326") {
    val p = Prob.fromDouble(1.0 / 1326.0)
    assert(p.raw > 0, "1/1326 must be representable")
    val relError = math.abs(p.toDouble - (1.0 / 1326.0)) / (1.0 / 1326.0)
    assert(relError < 1e-6, s"relative error too large: $relError")
  }

  test("fromDouble for out-of-range values does not throw") {
    // Values >1.0 or <0.0 are caller's responsibility; fromDouble just converts
    val over = Prob.fromDouble(1.5)
    assert(over.raw > Prob.One.raw)
    val neg = Prob.fromDouble(-0.1)
    assert(neg.raw < 0)
  }
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `sbt "testOnly sicfun.core.ProbTest" 2>&1 | tail -20`
Expected: all tests pass

- [ ] **Step 3: Commit**

```bash
git add src/test/scala/sicfun/core/ProbTest.scala
git commit -m "test: add ProbTest covering roundtrip, arithmetic, edge cases"
```

---

## Chunk 2: Equity Hot Path Migration + Correctness Verification

### Task 3: Add equityExactProb method

Parallel implementation using Prob/Long accumulation alongside the existing Double method. Returns `EquityResult` (Double) — no downstream changes needed.

**Files:**
- Modify: `src/main/scala/sicfun/holdem/equity/HoldemEquity.scala`

- [ ] **Step 1: Add import and method**

Add `import sicfun.core.Prob.*` to the imports (after line 5).

Add the new method after `equityExact` (after line 70):

```scala
  /** Fixed-point variant of equityExact. Uses Int32 weights with Long accumulation
    * for deterministic arithmetic and better cache utilization.
    *
    * Integer division of weight.raw by boardCount introduces truncation error of at most
    * 1 LSB per board evaluation (~4e-8 relative error), negligible for poker equity needs.
    */
  def equityExactProb(
      hero: HoleCards,
      board: Board,
      villainRange: DiscreteDistribution[HoleCards]
  ): EquityResult =
    validateHeroBoard(hero, board)
    val dead = hero.asSet ++ board.asSet
    val range = sanitizeRange(villainRange, dead)
    val missing = board.missing

    // Accumulate as Long. Max: 1326 × 1081 × (2^30/1) ≈ 1.5e15, Long max 9.2e18.
    var win = 0L
    var tie = 0L
    var loss = 0L

    range.weights.foreach { case (villain, weightD) =>
      if weightD > 0.0 then
        val weight = Prob.fromDouble(weightD)
        val remaining = Deck.full.filterNot(card => dead.contains(card) || villain.contains(card)).toIndexedSeq
        val boardCount = combinationsCount(remaining.length, missing)
        val perBoardWeight = weight.raw.toLong / boardCount

        val boards =
          if missing == 0 then Iterator.single(Vector.empty[Card])
          else HoldemCombinator.combinations(remaining, missing)

        boards.foreach { extra =>
          val fullBoard = board.cards ++ extra
          val heroRank = HandEvaluator.evaluate7Cached(hero.toVector ++ fullBoard)
          val villainRank = HandEvaluator.evaluate7Cached(villain.toVector ++ fullBoard)
          val cmp = heroRank.compare(villainRank)
          if cmp > 0 then win += perBoardWeight
          else if cmp == 0 then tie += perBoardWeight
          else loss += perBoardWeight
        }
    }

    val total = win + tie + loss
    if total == 0L then EquityResult(0.0, 0.0, 0.0)
    else
      val invTotal = 1.0 / total.toDouble
      EquityResult(win * invTotal, tie * invTotal, loss * invTotal)
```

- [ ] **Step 2: Verify it compiles**

Run: `sbt compile 2>&1 | tail -5`
Expected: `[success]`

- [ ] **Step 3: Commit**

```bash
git add src/main/scala/sicfun/holdem/equity/HoldemEquity.scala
git commit -m "feat: add equityExactProb using fixed-point Prob accumulation"
```

### Task 4: Correctness test — Double vs Prob equivalence

**Files:**
- Create: `src/test/scala/sicfun/holdem/equity/ProbEquityCorrectnessTest.scala`

- [ ] **Step 1: Write the test file**

```scala
package sicfun.holdem.equity

import munit.FunSuite
import sicfun.core.{Card, Deck, DiscreteDistribution}
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
```

- [ ] **Step 2: Run tests**

Run: `sbt "testOnly sicfun.holdem.equity.ProbEquityCorrectnessTest" 2>&1 | tail -20`
Expected: all pass

- [ ] **Step 3: Run existing equity tests to verify no regressions**

Run: `sbt "testOnly sicfun.holdem.equity.HoldemEquityTest" 2>&1 | tail -20`
Expected: all pass (existing code unchanged)

- [ ] **Step 4: Commit**

```bash
git add src/test/scala/sicfun/holdem/equity/ProbEquityCorrectnessTest.scala
git commit -m "test: verify equityExactProb matches equityExact within tolerance"
```

---

## Chunk 3: A/B Benchmark

### Task 5: Create benchmark comparing Double vs Prob equity

**Files:**
- Create: `src/main/scala/sicfun/holdem/bench/ProbEquityBenchmark.scala`

- [ ] **Step 1: Write the benchmark**

Interleaves Double and Prob runs to reduce JVM warmup / GC bias:

```scala
package sicfun.holdem.bench

import sicfun.core.{Card, Deck, DiscreteDistribution}
import sicfun.holdem.types.*
import sicfun.holdem.equity.{HoldemEquity, HoldemCombinator}

/** A/B benchmark: equityExact (Double) vs equityExactProb (Int32 fixed-point).
  *
  * Usage: sbt "runMain sicfun.holdem.bench.ProbEquityBenchmark [warmup] [runs] [mode]"
  *   mode: "turn" (default) or "flop"
  */
object ProbEquityBenchmark:
  private def card(token: String): Card =
    Card.parse(token).getOrElse(throw new IllegalArgumentException(s"bad card: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  private def board(tokens: String*): Board =
    Board.from(tokens.map(card))

  def main(args: Array[String]): Unit =
    val warmup = if args.length > 0 then args(0).toInt else 5
    val runs = if args.length > 1 then args(1).toInt else 20
    val mode = if args.length > 2 then args(2) else "turn"

    val hero = hole("Ah", "Kh")
    val (b, label) = mode match
      case "flop" => (board("2c", "7d", "Ts"), "flop (2 missing)")
      case _      => (board("2c", "7d", "Ts", "Qc"), "turn (1 missing)")

    // Build a realistic range (~20 hands with varied weights)
    val remaining = Deck.full.filterNot(card => hero.contains(card) || b.asSet.contains(card))
    val allHands = HoldemCombinator.holeCardsFrom(remaining.toIndexedSeq).take(20)
    val weights = allHands.zipWithIndex.map { case (h, i) => h -> (1.0 + i * 0.1) }
    val range = DiscreteDistribution(weights.toMap).normalized

    println(s"=== Prob Equity A/B Benchmark ===")
    println(s"Board: $label, Range size: ${range.weights.size} hands")
    println(s"Warmup: $warmup, Runs: $runs")
    println()

    // Warmup both paths
    var w = 0
    while w < warmup do
      HoldemEquity.equityExact(hero, b, range)
      HoldemEquity.equityExactProb(hero, b, range)
      w += 1

    // Interleaved measurement to reduce systematic bias (JVM warmup, GC, thermals)
    val doubleTimes = new Array[Long](runs)
    val probTimes = new Array[Long](runs)
    var r = 0
    while r < runs do
      // Double first on even iterations, Prob first on odd
      if r % 2 == 0 then
        doubleTimes(r) = timeOnce { HoldemEquity.equityExact(hero, b, range) }
        probTimes(r) = timeOnce { HoldemEquity.equityExactProb(hero, b, range) }
      else
        probTimes(r) = timeOnce { HoldemEquity.equityExactProb(hero, b, range) }
        doubleTimes(r) = timeOnce { HoldemEquity.equityExact(hero, b, range) }
      r += 1

    // Correctness check
    val dblResult = HoldemEquity.equityExact(hero, b, range)
    val probResult = HoldemEquity.equityExactProb(hero, b, range)

    // Report
    val dblMedian = median(doubleTimes)
    val probMedian = median(probTimes)
    val speedup = dblMedian.toDouble / probMedian.toDouble

    println("--- Double (baseline) ---")
    printStats("Double", doubleTimes)
    println()
    println("--- Prob (fixed-point) ---")
    printStats("Prob  ", probTimes)
    println()
    println(f"Speedup (median): ${speedup}%.3fx")
    println()
    println("--- Correctness ---")
    println(f"Double: win=${dblResult.win}%.6f tie=${dblResult.tie}%.6f loss=${dblResult.loss}%.6f equity=${dblResult.equity}%.6f")
    println(f"Prob:   win=${probResult.win}%.6f tie=${probResult.tie}%.6f loss=${probResult.loss}%.6f equity=${probResult.equity}%.6f")
    val eqDiff = math.abs(dblResult.equity - probResult.equity)
    println(f"Equity diff: $eqDiff%.9f")

  private inline def timeOnce(inline body: => Any): Long =
    val t0 = System.nanoTime()
    body
    System.nanoTime() - t0

  private def median(arr: Array[Long]): Long =
    val sorted = arr.sorted
    sorted(sorted.length / 2)

  private def printStats(label: String, times: Array[Long]): Unit =
    val sorted = times.sorted
    val n = sorted.length
    val med = sorted(n / 2)
    val min = sorted.head
    val max = sorted.last
    val mean = times.map(_.toDouble).sum / n
    val p25 = sorted(n / 4)
    val p75 = sorted(3 * n / 4)
    println(f"$label median=${med / 1e6}%.2f ms  p25=${p25 / 1e6}%.2f  p75=${p75 / 1e6}%.2f  min=${min / 1e6}%.2f  max=${max / 1e6}%.2f  mean=${mean / 1e6}%.2f ms")
```

- [ ] **Step 2: Verify it compiles**

Run: `sbt compile 2>&1 | tail -5`
Expected: `[success]`

- [ ] **Step 3: Run the benchmark (turn)**

Run: `sbt "runMain sicfun.holdem.bench.ProbEquityBenchmark 5 20 turn" 2>&1 | tail -30`
Expected: output showing timing comparison and correctness check

- [ ] **Step 4: Run the benchmark (flop)**

Run: `sbt "runMain sicfun.holdem.bench.ProbEquityBenchmark 5 20 flop" 2>&1 | tail -30`
Expected: output showing timing comparison for bigger workload

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/bench/ProbEquityBenchmark.scala
git commit -m "bench: add interleaved A/B benchmark for Double vs Prob equity"
```

### Task 6: Evaluate results and document

- [ ] **Step 1: Analyze benchmark output**

Check:
- Is Prob faster, same, or slower than Double?
- Is the equity difference within acceptable tolerance (<0.001)?
- What's the speedup ratio?
- Is the p25-p75 range tight enough to trust the result?

- [ ] **Step 2: Document results**

Add results to the design spec at `docs/superpowers/specs/2026-03-10-fixed-point-prob-design.md` under a new `## 5. Benchmark Results` section with the actual numbers.

- [ ] **Step 3: Go/no-go decision**

Based on the criteria from the design spec:
- >= 5% faster → propagate incrementally (proceed to Phase 2 plan)
- +/- 5% neutral → proceed for determinism value
- > 5% slower → reevaluate

Note: the main performance win from Prob is expected in Phase 2 when `DiscreteDistribution` weights become `Array[Prob]` (4 bytes vs 8, 2x cache density). Phase 1 primarily validates correctness and measures the arithmetic overhead of the approach.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-03-10-fixed-point-prob-design.md
git commit -m "docs: add benchmark results to fixed-point Prob design spec"
```
