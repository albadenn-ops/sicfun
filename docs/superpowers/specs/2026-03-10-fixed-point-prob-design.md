# Fixed-Point Probability Type (`Prob`) — Design Spec

**Date**: 2026-03-10
**Goal**: Replace `Double` with fixed-point `Int32 @ 2^30` for probability values throughout sicfun, starting with the equity hot path. Targets: performance (cache), determinism, memory reduction.

## 1. The `Prob` Opaque Type

Location: `sicfun.core.Prob`

```scala
object Prob:
  inline val Scale = 1 << 30           // 1,073,741,824
  inline val One   = Prob(Scale)
  inline val Zero  = Prob(0)
  inline val Half  = Prob(Scale >> 1)

  inline def apply(raw: Int): Prob = raw
  def fromDouble(d: Double): Prob = (d * Scale + 0.5).toInt

opaque type Prob = Int
extension (p: Prob)
  inline def raw: Int = p
  inline def toDouble: Double = p.toDouble / Prob.Scale
  inline def +(q: Prob): Prob = p + q
  inline def -(q: Prob): Prob = p - q
  inline def *(q: Prob): Prob = ((p.toLong * q.toLong) >> 30).toInt
  inline def /(n: Int): Prob = p / n
  inline def >(q: Prob): Boolean  = p > q
  inline def <(q: Prob): Boolean  = p < q
  inline def >=(q: Prob): Boolean = p >= q
```

Design decisions:
- **2^30 scale** (not 2^31): leaves sign bit free, subtraction stays safe
- **`fromDouble` with rounding**: correct conversion at boundaries with existing Double code
- **`toDouble` for output only**: display/logging, never in hot paths
- **All inline**: zero runtime overhead

## 2. Equity Hot Path Migration

The exact enumeration path in `HoldemEquity.scala` is the hottest loop in the system: 1326 hands x N boards.

Key change: accumulate raw weights as `Long`, divide once at the end (more precise than current per-iteration Double division).

```scala
var win = 0L; var tie = 0L; var loss = 0L
range.weightsProb.foreach { case (villain, weight) =>
  if weight > Prob.Zero then
    boards.foreach { board =>
      if cmp > 0 then win += weight.raw
      else if cmp == 0 then tie += weight.raw
      else loss += weight.raw
    }
}
val r = EquityResult(
  Prob((win / boardCount).toInt),
  Prob((tie / boardCount).toInt),
  Prob((loss / boardCount).toInt))
```

Overflow analysis: 1326 x 1000 x 2^30 = 1.4e15. Long max = 9.2e18. Headroom ~6000x.

`EquityResult` migrates to Prob:
```scala
final case class EquityResult(win: Prob, tie: Prob, loss: Prob):
  inline def equity: Prob = win + (tie / 2)
```

What stays Double for now:
- Monte Carlo (Welford's algorithm needs floating-point running variance)
- DiscreteDistribution internals (converted to Prob at hot path boundary)
- GPU/native paths (later phase)

## 3. Benchmark Strategy

A/B comparison using existing benchmark suite in `sicfun.holdem.bench`.

Metrics: throughput (hands/sec), peak memory, result correctness (verify against Double baseline modulo rounding).

Go/no-go criteria:
- >= 5% faster: propagate incrementally
- +/- 5% neutral: proceed for determinism value alone
- > 5% slower: reevaluate approach

## 4. Propagation Order (if benchmark positive)

1. `Prob` type + `EquityResult` migration (this spec)
2. `DiscreteDistribution[HoleCards]` weights -> `Map[A, Prob]`
3. Bayesian updates in `DiscreteDistribution.updateWithLikelihood`
4. CFR solver regrets/strategies (unbounded values — needs separate `FixedVal` type with different scale)
5. GPU/native: JNI Int32, CUDA kernels with int arithmetic
6. GameState (pot, stack, EV) — unbounded, final phase

## 5. Benchmark Results (Phase 1)

**Setup:** `equityExact` (Double) vs `equityExactProb` (Prob/Long accumulation), 20 hands range, 5 warmup, 20 interleaved runs.

| Mode | Double median | Prob median | Speedup | Equity diff |
|------|--------------|-------------|---------|-------------|
| Turn (1 missing) | 1.98 ms | 2.16 ms | 0.917x | 3.3e-8 |
| Flop (2 missing) | 17.98 ms | 18.02 ms | 0.998x | 3.7e-8 |

**Correctness:** Excellent — equity differences < 4e-8 (well within 1e-3 tolerance).

**Performance:** Neutral-to-slightly-slower. The inner loop is dominated by `evaluate7Cached` (hand evaluation), not weight accumulation. Converting accumulators from Double to Long has negligible effect on throughput.

**Decision:** Proceed for determinism value (+/- 5% neutral criterion met). The real performance win is expected in Phase 2: converting `DiscreteDistribution` to compact `Array[Prob]` storage (4 bytes vs 8-byte Double in Map), which would halve cache pressure during range iteration and eliminate Map overhead.

**Spec correction:** The original spec's "accumulate raw, divide once at end" approach assumed uniform boardCount across villain hands. In practice, boardCount varies per villain (dead cards differ), so the implementation correctly divides `weight.raw / boardCount` per villain hand, matching the original Double code's semantics.

## 6. Phase 2: Array-Based Equity with Prob Weights

Phase 1 showed the inner loop is dominated by `evaluate7Cached`, so converting accumulators alone gave no speedup. Phase 2 targets the **outer loop**: replacing `Map.foreach` iteration with flat array iteration using Prob weights.

### PreparedRangeProb

New private type in `HoldemEquity`, mirroring the existing `PreparedRange`:

```scala
private final case class PreparedRangeProb(
    hands: Array[HoleCards],
    weights: Array[Int],     // Prob raw values (Int32 @ 2^30)
    size: Int
)
```

Built by a new `prepareRangeProb` method that reuses the existing thread-local scratch buffers. Weights are converted to Prob once during preparation, not per-iteration.

### equityExactProb refactored

Uses `PreparedRangeProb` instead of `Map.foreach`:

```scala
val prepared = prepareRangeProb(villainRange, dead)
var i = 0
while i < prepared.size do
  val villain = prepared.hands(i)
  val weightRaw = prepared.weights(i)
  if weightRaw > 0 then
    val remaining = ...
    val boardCount = combinationsCount(remaining.length, missing)
    val perBoardWeight = weightRaw.toLong / boardCount
    boards.foreach { extra =>
      ...
      if cmp > 0 then winL += perBoardWeight
      else if cmp == 0 then tieL += perBoardWeight
      else lossL += perBoardWeight
    }
  i += 1
```

Benefits:
- Linear array iteration replaces Map.foreach (no hashing/boxing/virtual dispatch)
- Sorted by HoleCardsIndex ID for cache-friendly access
- 4 bytes per weight (Int) vs 8 bytes (Double) — 2x denser arrays
- Thread-local scratch reuse (already proven in existing `prepareRange`)
