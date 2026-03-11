# Fixed-Point Probability Type (`Prob`) - Design Spec

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
4. CFR solver regrets/strategies (unbounded values - needs separate `FixedVal` type with different scale)
5. GPU/native: JNI Int32, CUDA kernels with int arithmetic
6. GameState (pot, stack, EV) - unbounded, final phase

## 5. Benchmark Results (Phase 1)

**Setup:** `equityExact` (Double) vs `equityExactProb` (Prob/Long accumulation), 20 hands range, 5 warmup, 20 interleaved runs.

| Mode | Double median | Prob median | Speedup | Equity diff |
|------|--------------|-------------|---------|-------------|
| Turn (1 missing) | 1.98 ms | 2.16 ms | 0.917x | 3.3e-8 |
| Flop (2 missing) | 17.98 ms | 18.02 ms | 0.998x | 3.7e-8 |

**Correctness:** Excellent - equity differences < 4e-8 (well within 1e-3 tolerance).

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
- Linear array iteration replaces `Map.foreach` (no hashing/boxing/virtual dispatch)
- Sorted by `HoleCardsIndex` ID for cache-friendly access
- 4 bytes per weight (`Int`) vs 8 bytes (`Double`) - 2x denser arrays
- Thread-local scratch reuse (already proven in existing `prepareRange`)
- Combines `sanitizeRange` + `prepareRange` into a single pass (no intermediate Map allocation)

### Phase 2 Benchmark Results

| Mode | Double median | Prob Phase 2 median | Speedup | Equity diff |
|------|--------------|---------------------|---------|-------------|
| Turn (1 missing, 20 runs) | 1.33 ms | 1.21 ms | 1.11x | 3.3e-8 |
| Turn (1 missing, 30 runs, 10 warmup) | 2.31 ms | 1.79 ms | 1.29x | 3.3e-8 |
| Flop (2 missing, 20 runs) | 20.25 ms | 20.06 ms | 1.01x | 3.7e-8 |

Phase 2 eliminated the `sanitizeRange` -> intermediate `DiscreteDistribution` -> `Map.foreach` overhead by combining filtering, canonical dedup, normalization, and Prob conversion into a single `prepareRangeProb` pass. The speedup is most visible on turn (shorter inner loop -> outer loop overhead is a larger fraction) and less visible on flop (inner loop dominates).

Correctness remains excellent - equity differences < 4e-8.

## 7. Phase 4: Experimental Fixed-Point CFR Core

Phase 4 was started with an incremental parallel path in the generic `sicfun.holdem.cfr.CfrSolver`, not by replacing the existing Double solver.

### Design

- New `sicfun.core.FixedVal`: signed fixed-point `Int32 @ 2^13` for CFR utilities and regrets.
- `solveFixed` / `solveRootPolicyFixed` added alongside the existing Double methods.
- Internal CFR loop now uses:
  - `Prob` for reach probabilities and current mixed strategies
  - `FixedVal` for terminal utilities, node values, regrets, and per-action utilities
  - `Array[Long]` for cumulative average-strategy mass, because that term grows roughly quadratically with iteration count under linear averaging
- Public output remains `TrainingResult` / `RootPolicyResult` with `Double`, so correctness and throughput can be compared A/B without downstream breakage.

### Benchmark Results

Benchmark: `sicfun.holdem.bench.CfrSolverFixedBenchmark` on Kuhn poker.

| Iterations | Double median | Fixed median | Speedup | EV diff | Largest reported action diff |
|-----------|---------------|--------------|---------|---------|------------------------------|
| 1500 (5 warmup, 30 runs) | 12.56 ms | 11.87 ms | 1.06x | 7.9e-5 | 9.9e-3 |
| 20000 (3 warmup, 10 runs) | 135.83 ms | 156.41 ms | 0.87x | 1.5e-5 | 2.1e-2 |

Correctness was good in both regimes: EV stayed within `8e-5` and the reported strategy deltas were around `1e-2` to `2e-2`.

Hold'em-specific gate: `sicfun.holdem.bench.HoldemCfrFixedBenchmark`.

| Operation | Scenario | Scala median | Scala-fixed median | Speedup | Policy/EV diff |
|-----------|----------|--------------|--------------------|---------|----------------|
| `solve` | preflop premium / 3-action root | 5.55 ms | 6.27 ms | 0.89x | identical one-hot policy |
| `solveDecisionPolicy` | turn / 3-action root | 5.32 ms | 11.81 ms | 0.45x | policy deltas <= 5e-6 |

That Hold'em benchmark is the more relevant signal: correctness is effectively unchanged, but the fixed-point path is clearly slower on the real decision workload.

### Decision

Do not propagate Phase 4 into `HoldemCfrSolver` yet.

Rationale:
- The fixed-point CFR core is promising on the smallest toy benchmark regime (`~1.06x` on Kuhn at 1500 iterations).
- It regresses on longer toy solves (`0.87x` at 20000 iterations).
- More importantly, it regresses on Hold'em-shaped workloads (`0.89x` for full solve preflop and `0.45x` for decision-only turn).
- Memory reduction is only partial so far because cumulative average-strategy mass still needs `Long`.

Current status:
- Keep `solveFixed` as an experimental, tested path.
- Keep `scala-fixed` as an explicit, opt-in benchmarking provider inside `HoldemCfrSolver`.
- Do not switch `HoldemCfrSolver` defaults or native CFR/JNI to it.
- Revisit only if a later JNI/native integer path can amortize the conversion cost, or if a different representation removes the current JVM slowdown.

## 8. Phase 5 Experiment: Native CPU Fixed-Point CFR

Implemented an integer JNI ABI for the CPU CFR solver:
- `Prob` raw (`Int32 @ 2^30`) for chance probabilities and output strategies
- `FixedVal` raw (`Int32 @ 2^13`) for terminal utilities and expected value
- native core entrypoint `solve_fixed` in `CfrNativeSolverCore.hpp`
- JNI bridge `HoldemCfrNativeCpuBindings.solveTreeFixed`
- experimental provider label `native-cpu-fixed`

Benchmark: `sicfun.holdem.bench.HoldemCfrNativeFixedBenchmark` against the existing `native-cpu` provider, using `src/main/native/build/sicfun_cfr_native.dll`.

| Operation | Scenario | Native CPU median | Native CPU fixed median | Speedup | Correctness |
|-----------|----------|-------------------|-------------------------|---------|-------------|
| `solve` | preflop premium / 3-action root | 5.93 ms | 5.36 ms | 1.11x | exact EV and one-hot policy parity |
| `solveDecisionPolicy` | preflop premium / 3-action root | 5.56 ms | 5.04 ms | 1.10x | exact root-policy parity |
| `solve` | turn / 3-action root | 4.50 ms | 4.90 ms | 0.92x | EV `2.975755 -> -2.943605`, policy collapsed to `Raise(12.0)` |
| `solveDecisionPolicy` | turn / 3-action root | 3.18 ms | 3.17 ms | 1.00x | best action flipped `Call -> Raise(12.0)`, policy diff ~`0.995` |

Inference:
- The representation itself is not the primary problem here, because the JVM `scala-fixed` path stayed close on the same turn spot.
- The regression is therefore much more likely a parity bug in the native fixed-point CFR implementation than an inherent limitation of `Prob` / `FixedVal`.

### Decision

Do not propagate Phase 5 fixed-point CFR into the recommended native provider path.

Current status:
- Keep the JNI/native fixed path as experimental instrumentation only.
- Keep `native-cpu-fixed` opt-in and warn when it is selected explicitly.
- Do not include it in auto-provider selection.
- Do not use it as the basis for GPU/native integer propagation until the native CPU solver matches `native-cpu` on the failing turn tree.

Next debugging gate:
- Compare per-iteration regrets / average-strategy accumulation for the failing turn tree between `CfrSolver.solveFixed` and `cfrnative::solve_fixed`.
- Only revisit GPU/native integer rollout after that parity gap is closed.

### Phase 5 GPU Follow-up

Added the same fixed-point JNI ABI to the CUDA-compiled CFR bridge:
- `HoldemCfrNativeGpuBindings.solveTreeFixed`
- `Provider.native-gpu-fixed`
- backend selector in `HoldemCfrNativeFixedBenchmark` so CPU and GPU-compiled bridges can be compared with the same workload

Important implementation note:
- `HoldemCfrNativeGpuBindings.cu` is still a CUDA-compiled host bridge around `cfrnative::solve` / `cfrnative::solve_fixed`.
- There is still no dedicated CFR CUDA kernel in this provider path, so this step validates compiler/bridge parity, not device-side integer arithmetic.

Benchmark: `java -cp target/scala-3.8.1/classes;... sicfun.holdem.bench.HoldemCfrNativeFixedBenchmark ... gpu`

| Operation | Scenario | Native GPU median | Native GPU fixed median | Speedup | Correctness |
|-----------|----------|-------------------|-------------------------|---------|-------------|
| `solve` | preflop premium / 3-action root | 7.64 ms | 8.66 ms | 0.88x | exact EV and one-hot policy parity |
| `solve` | turn / 3-action root | 5.65 ms | 6.52 ms | 0.87x | EV `2.975755 -> -2.943706`, policy collapsed to `Raise(12.0)` |
| `solveDecisionPolicy` | turn / 3-action root | 5.43 ms | 6.00 ms | 0.91x | best action flipped `Call -> Raise(12.0)`, policy diff ~`0.995` |

Decision update:
- Do not spend time porting the current fixed-point CFR core into a real CUDA kernel yet.
- The same parity failure reproduces through the CUDA-compiled bridge, so the problem is below the CPU-vs-GPU binding layer.
- The next worthwhile GPU step is blocked on fixing `cfrnative::solve_fixed` parity first; otherwise a device kernel would just preserve the wrong math faster.

### Fixed Parity Probe

Added `sicfun.holdem.bench.HoldemCfrFixedParityProbe` to compare `scala-fixed`, `native-cpu-fixed`, and `native-gpu-fixed` on the same Hold'em tree over explicit iteration counts.

Turn/full probe result on the failing 3-action turn spot:

| Iterations | Averaging delay | Scala-fixed | Native CPU fixed | Native GPU fixed | Observation |
|------------|-----------------|-------------|------------------|------------------|-------------|
| 1 | 1 | exact | exact | exact | no parity issue |
| 2 | 1 | exact | exact | exact | no parity issue |
| 4 | 1 | exact | exact | exact | no parity issue |
| 8 | 2 | baseline EV `2.561664` | EV `-2.802471`, max action diff `0.761905` | EV `-2.802471`, max action diff `0.761905` | divergence starts here |
| 16 | 4 | baseline EV `2.711108` | EV `-3.106684`, max action diff `0.820513` | EV `-3.889968`, max action diff `0.932771` | already catastrophic |
| 64 | 16 | baseline EV `2.866294` | EV `-2.921661`, max action diff `0.965986` | EV `-3.023985`, max action diff `0.986315` | fully collapsed |

Interpretation:
- The defect is not a long-horizon drift that only appears after hundreds of iterations.
- The defect is also not specific to the CPU JNI bridge, because the CUDA-compiled bridge reproduces it at the same small iteration counts.
- The next debugging target should be the native fixed regret/strategy update logic around iterations `5-8`, not JNI plumbing or device-kernel work.
