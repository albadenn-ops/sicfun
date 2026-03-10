# Compact Posterior — Eliminating the Map Round-Trip

**Date**: 2026-03-10
**Branch**: `feature/fixed-point-prob`
**Prerequisite**: Phases 1-2 of fixed-point Prob migration (complete)

## 1. Problem

The Bayesian posterior flows from `HoldemBayesProvider` through `RangeInferenceEngine` to `HoldemEquity`. Both ends operate on flat arrays, but the middle wraps/unwraps through `Map[HoleCards, Double]`:

```
scalaUpdate:  Array[Double] posterior (flat, already computed)
     ↓  Map.newBuilder → Map[HoleCards, Double]       ← allocation + boxing
     ↓  DiscreteDistribution(map).normalized           ← another Map allocation
     ↓  passes through RangeInferenceEngine
     ↓
prepareRange/prepareRangeProb:
     ↓  Map.foreach → scratch arrays                   ← hash iteration + unboxing
     ↓  normalize + Prob.fromDouble conversion
     ↓  PreparedRangeProb / PreparedRange
     ↓
equity inner loop:  linear array iteration (fast)
```

This costs:
- Two `Map` allocations (~500-1000 entries each, with HoleCards boxing)
- `Map.foreach` hash-based iteration (non-sequential memory access)
- `Double → Prob` conversion that could be done once at the source
- GC pressure from short-lived Map objects on every Bayesian update

## 2. Solution: `CompactPosterior`

A flat-array posterior representation that travels from Bayes to Equity without Map intermediary.

```scala
/** Flat-array posterior for the Bayes → Equity hot path.
  *
  * Built directly from the arrays already computed in scalaUpdate/nativeUpdate.
  * Equity methods consume this without ever creating a Map.
  * Non-hot consumers (DDRE, validation, collapse metrics) access the lazy
  * `distribution` field, which materializes the Map only when needed.
  */
final class CompactPosterior(
    val hands: Array[HoleCards],
    val probWeights: Array[Int],      // Prob raw values (Int32 @ 2^30), normalized
    val size: Int
):
  /** Lazy materialization for non-hot-path consumers. */
  lazy val distribution: DiscreteDistribution[HoleCards] =
    val builder = Map.newBuilder[HoleCards, Double]
    builder.sizeHint(size)
    var i = 0
    while i < size do
      builder += hands(i) -> Prob(probWeights(i)).toDouble
      i += 1
    DiscreteDistribution(builder.result())
```

Location: `sicfun.holdem.equity` (alongside `PreparedRangeProb`).

### Design decisions

- **Not generic**: This is `HoleCards`-specific, not `DiscreteDistribution[A]`. The generic type stays unchanged.
- **Prob weights, not Double**: Bayesian posterior doubles are converted to Prob once at construction, not per-equity-call.
- **Lazy distribution**: The `DiscreteDistribution` is only materialized when a non-hot consumer accesses it. On the pure Bayes→Equity path (DDRE off), the Map is never created. Scala's `lazy val` uses double-checked locking on the JVM, so concurrent access is safe (one thread blocks briefly on first materialization — acceptable for non-hot paths).
- **No dead-card filtering**: Dead card filtering happens in `HoldemEquity` (it depends on hero cards). `CompactPosterior` stores the full posterior.
- **Non-canonical hands allowed**: The `hands` array may contain non-canonical `HoleCards` (this depends on how the upstream prior was constructed). `prepareRangeProbFromCompact` must canonicalize and merge duplicates, same as the existing `prepareRangeProb`.
- **Validation**: `buildCompactPosterior` requires at least one positive weight. All-zero posteriors fail fast at construction, not deferred to lazy materialization.

## 3. Changes by Component

### 3.1 HoldemBayesProvider

`UpdateResult` gains a `CompactPosterior` field:

```scala
final case class UpdateResult(
    posterior: DiscreteDistribution[HoleCards],
    compact: CompactPosterior,
    logEvidence: Double,
    provider: Provider
)
```

**scalaUpdate** (lines 304-352): After computing the posterior `Array[Double]`, build `CompactPosterior` directly from `hypotheses: Vector[HoleCards]` and `posterior: Array[Double]`. Convert weights to Prob in a single pass. The existing Map construction stays (for `posterior` field) but can be deferred to `compact.distribution` in a follow-up if benchmarks show the Map is rarely accessed.

We eliminate the Map construction from scalaUpdate entirely:

```scala
// Build CompactPosterior directly from flat arrays
val compact = buildCompactPosterior(hypotheses, posterior)
UpdateResult(
    posterior = compact.distribution,  // lazy — only materialized if accessed
    compact = compact,
    logEvidence = logEvidence,
    provider = provider
)
```

Same for **nativeUpdate** (lines 265-302).

Note: `UpdateResult.posterior` becomes an alias for `compact.distribution` (same lazy val). Callers accessing `updateResult.posterior` will trigger lazy materialization. This is intentional — `posterior` is kept for backward compatibility with non-hot consumers. It may be removed in a future cleanup once all consumers are audited.

### 3.2 RangeInferenceEngine

**PosteriorInferenceResult** gains an optional `CompactPosterior`:

```scala
final class PosteriorInferenceResult private (
    val prior: DiscreteDistribution[HoleCards],
    val posterior: DiscreteDistribution[HoleCards],
    val compact: Option[CompactPosterior],
    val logEvidence: Double,
    collapseThunk: () => PosteriorCollapse
)
```

**computePosterior** (lines 234-304): When DDRE mode is `Off` and Bayesian update runs, thread `bayesUpdate.compact` through to `PosteriorInferenceResult`. When DDRE is active (fusion creates a new DiscreteDistribution), `compact = None` — falls back to existing Map-based path.

**recommendActionAssumeNormalized** (lines 600-655): If `compact` is available, apply the same top-k truncation that `compactPosteriorForEquity` performs (keep top `maxHands` entries covering `minMass` of probability) but operate on the flat arrays directly, then pass the truncated `CompactPosterior` to equity. This is critical — `compactPosteriorForEquity` doesn't just convert formats, it limits the villain count to ~256 hands for equity performance. Skipping this truncation would iterate all ~500-1000 hypotheses and negate the optimization. If `compact` is not available, fall back to existing Map-based path.

### 3.3 HoldemEquity

New overloads that accept `CompactPosterior`:

```scala
def equityExactProb(
    hero: HoleCards,
    board: Board,
    compact: CompactPosterior
): EquityResult

def equityMonteCarlo(
    hero: HoleCards,
    board: Board,
    compact: CompactPosterior,
    trials: Int,
    rng: Random = new Random()
): EquityEstimate
```

These methods filter dead cards from the flat arrays directly into `PreparedRangeProb` / `PreparedRange`, bypassing `prepareRangeProb(DiscreteDistribution)` entirely.

**Dead card filtering from flat arrays** (new private method):

```scala
private def prepareRangeProbFromCompact(
    compact: CompactPosterior,
    dead: Set[Card]
): PreparedRangeProb =
  // Single pass: filter dead cards, canonical dedup, write to scratch arrays
  // Same logic as prepareRangeProb but reads from compact.hands/compact.probWeights
  // instead of Map.foreach — sequential array access, no hashing
  // MUST canonicalize hands and merge weights for same canonical ID
  // MUST re-normalize after dead-card filtering (removed hands change total mass)

private def prepareRangeFromCompact(
    compact: CompactPosterior,
    deadMaskValue: Long
): PreparedRange =
  // Same as prepareRangeProbFromCompact but produces PreparedRange (Double weights)
  // for the Monte Carlo path which uses Welford's algorithm with floating-point
  // Converts Prob weights back to Double via .toDouble for MC accumulation
```

### 3.4 Non-hot-path consumers

All existing consumers that need `DiscreteDistribution` access it via:
- `compact.distribution` (lazy, materialized once)
- Or the `posterior` field on `PosteriorInferenceResult` / `UpdateResult`

These include:
- **validatePosteriorForDecision**: calls `probabilityOf` → uses `compact.distribution`
- **fusePosteriors**: calls `probabilityOf` → uses `compact.distribution`
- **CollapseMetrics.summary**: iterates weights → uses `compact.distribution`
- **Logging/display**: infrequent, cost is irrelevant

## 4. What Doesn't Change

- `DiscreteDistribution[A]` — untouched (generic core type)
- `Prob` type — untouched
- GPU/native kernels — untouched
- CFR solver — untouched
- `equityExact` (legacy Double path) — untouched, keeps existing signature
- Existing `equityExactProb(DiscreteDistribution)` — stays, used by non-Bayes callers
- Existing `equityMonteCarlo(DiscreteDistribution)` — stays, used by non-Bayes callers

## 5. DDRE Interaction

When DDRE mode is active, `resolveDecisionPosterior` fuses Bayesian + DDRE posteriors into a new `DiscreteDistribution`. This fusion uses `probabilityOf` (Map lookup) and builds a new Map. The `CompactPosterior` is NOT available after fusion.

Possible future optimization: `fusePosteriors` could produce a `CompactPosterior` from the fused weights. But DDRE fusion is not on the hot path (it's a feature toggle, not the common case), so this is deferred.

For now: when DDRE fusion is active (`BlendCanary`/`BlendPrimary`), `compact = None` and the existing Map-based equity path is used. In `Shadow` mode, DDRE runs but its result is discarded — the Bayes posterior is used for the decision, so `compact` is available and the fast path applies.

## 6. Benchmark Strategy

A/B comparison: existing path vs CompactPosterior path, measuring the full Bayes→Equity round-trip.

### Metrics
- **End-to-end decision latency**: `inferAndRecommend` wall time
- **Map allocation count**: GC pressure (can observe via `-verbose:gc` or allocation counters)
- **Equity throughput**: hands/sec for equityExactProb and equityMonteCarlo

### Benchmark design
- Use existing `ProbEquityBenchmark` infrastructure (interleaved A/B)
- Test cases: same range/board scenarios from Phase 2 benchmarks
- Measure both paths back-to-back to control for JIT/GC variance
- Separate measurement of Bayes→Equity overhead (construct Map vs construct CompactPosterior)

### Go/no-go
- **Any measurable speedup**: proceed (the determinism + memory wins justify even marginal speedup)
- **Measurably slower**: investigate (unlikely given we're removing work, but possible if lazy materialization adds overhead on paths that always need the Map)

## 7. Implementation Order

1. Define `CompactPosterior` type
2. Add `buildCompactPosterior` helper (Array[Double] + Vector[HoleCards] → CompactPosterior)
3. Update `HoldemBayesProvider.UpdateResult` and both scalaUpdate/nativeUpdate
4. Add `prepareRangeProbFromCompact` and `prepareRangeFromCompact` in HoldemEquity
5. Add equity overloads that accept CompactPosterior
6. Thread CompactPosterior through RangeInferenceEngine
7. Write correctness tests (Prob vs Double equity diff < 1e-7)
8. Write A/B benchmark
9. Benchmark and decide
