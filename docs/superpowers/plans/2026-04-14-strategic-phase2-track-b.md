# Strategic Phase 2 Track B — Benchmark Harness and Baseline Capture

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repeatable benchmark harness that captures overlay metrics (change rate, veto rate, action mix, latency) across hall self-play, ACPC, Slumbot, and a fixed decision corpus, then capture baseline measurements before any Track A formulation work starts.

**Architecture:** Overlay metrics are collected via a per-run `OverlayMetricsAccumulator` that records each `StrategicDecisionResult` (action + overlay result + timing). The accumulator is wired into all three runners (hall, ACPC, Slumbot). A fixed decision corpus of 6 spots provides deterministic before/after signal. Existing benchmark scripts are extended to accept `strategic` mode. A baseline capture script runs all surfaces and saves artifacts.

**Tech Stack:** Scala 3.8.1, SBT, munit 1.2.2, PowerShell scripts, `-Werror` compiler flag.

**Spec:** `docs/superpowers/specs/2026-04-14-strategic-phase2-design.md` (Track B, Gates G1–G5)

---

### File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/main/scala/sicfun/holdem/runtime/protocol/OverlayMetrics.scala` | Create | `OverlayStats`, `OverlayMetricsAccumulator` types |
| `src/test/scala/sicfun/holdem/runtime/protocol/OverlayMetricsTest.scala` | Create | Unit tests for accumulator |
| `src/main/scala/sicfun/holdem/engine/HeroDecisionPipeline.scala` | Modify | Add `StrategicDecisionResult`, change `decideHeroStrategic` return type, wrap with timing |
| `src/test/scala/sicfun/holdem/engine/StrategicEngineOverlayTest.scala` | Modify | Update to use new return type |
| `src/main/scala/sicfun/holdem/runtime/protocol/AcpcMatchRunner.scala` | Modify | Wire overlay metrics accumulator |
| `src/main/scala/sicfun/holdem/runtime/protocol/SlumbotMatchRunner.scala` | Modify | Wire overlay metrics accumulator |
| `src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala` | Modify | Wire overlay metrics accumulator |
| `src/main/scala/sicfun/holdem/runtime/protocol/MatchRunnerSupport.scala` | Modify | Extend `RunSummary` and `writeSummary` with overlay stats |
| `scripts/match/run-hall-matchups.ps1` | Modify | Add `strategic` to allowed hero styles |
| `scripts/match/run-slumbot-benchmark.ps1` | Modify | Add `strategic` to allowed hero modes |
| `src/main/scala/sicfun/holdem/bench/DecisionCorpusBenchmark.scala` | Create | Fixed corpus spots + deterministic replay harness |
| `src/test/scala/sicfun/holdem/bench/DecisionCorpusBenchmarkTest.scala` | Create | Determinism + regression tests |
| `scripts/bench/capture-baseline.ps1` | Create | Baseline capture across all surfaces |

---

### Task 1: OverlayMetrics — Accumulator and Stats Types

**Files:**
- Create: `src/main/scala/sicfun/holdem/runtime/protocol/OverlayMetrics.scala`
- Create: `src/test/scala/sicfun/holdem/runtime/protocol/OverlayMetricsTest.scala`

- [ ] **Step 1: Write the failing test for OverlayMetricsAccumulator**

```scala
package sicfun.holdem.runtime.protocol

import sicfun.holdem.engine.{OverlayResult, OverlayAdjustment, UpstreamSource}
import sicfun.holdem.engine.inference.ActionEvaluation
import sicfun.holdem.types.PokerAction

class OverlayMetricsTest extends munit.FunSuite:

  test("empty accumulator produces zero stats") {
    val acc = OverlayMetricsAccumulator()
    val stats = acc.snapshot()
    assertEquals(stats.decisions, 0)
    assertEquals(stats.overlayChanges, 0)
    assertEqualsDouble(stats.overlayChangeRate, 0.0, 1e-12)
    assertEqualsDouble(stats.vetoRate, 0.0, 1e-12)
    assertEqualsDouble(stats.meanLatencyMs, 0.0, 1e-12)
    assertEquals(stats.actionDistribution, Map.empty[String, Int])
  }

  test("accumulator counts overlay changes when selected differs from upstream") {
    val acc = OverlayMetricsAccumulator()
    // Decision 1: overlay keeps upstream (Raise)
    acc.record(makeResult(
      selected = PokerAction.Raise(2.5),
      upstream = PokerAction.Raise(2.5),
      softVetoed = Vector.empty,
      adjustments = Vector.empty,
      latencyNanos = 1_000_000L // 1ms
    ))
    // Decision 2: overlay changes upstream (Call -> Raise)
    acc.record(makeResult(
      selected = PokerAction.Raise(2.5),
      upstream = PokerAction.Call,
      softVetoed = Vector.empty,
      adjustments = Vector(OverlayAdjustment(PokerAction.Call, 0.5, 0.3, "belief-penalty")),
      latencyNanos = 2_000_000L // 2ms
    ))
    val stats = acc.snapshot()
    assertEquals(stats.decisions, 2)
    assertEquals(stats.overlayChanges, 1)
    assertEqualsDouble(stats.overlayChangeRate, 0.5, 1e-12)
    assertEquals(stats.adjustments, 1)
    assertEqualsDouble(stats.meanLatencyMs, 1.5, 1e-6)
  }

  test("accumulator tracks soft vetoes and action distribution") {
    val acc = OverlayMetricsAccumulator()
    acc.record(makeResult(
      selected = PokerAction.Fold,
      upstream = PokerAction.Fold,
      softVetoed = Vector((PokerAction.Call, "robust bound -0.15 < -0.10")),
      adjustments = Vector.empty,
      latencyNanos = 500_000L
    ))
    acc.record(makeResult(
      selected = PokerAction.Raise(2.5),
      upstream = PokerAction.Raise(2.5),
      softVetoed = Vector.empty,
      adjustments = Vector.empty,
      latencyNanos = 800_000L
    ))
    val stats = acc.snapshot()
    assertEquals(stats.decisionsWithVeto, 1)
    assertEquals(stats.totalVetoedActions, 1)
    assertEqualsDouble(stats.vetoRate, 0.5, 1e-12)
    assertEquals(stats.actionDistribution("Fold"), 1)
    assertEquals(stats.actionDistribution("Raise:2.50"), 1)
  }

  test("vetoRate counts decisions not total vetoed actions") {
    val acc = OverlayMetricsAccumulator()
    // One decision that vetoes 3 actions
    acc.record(makeResult(
      selected = PokerAction.Fold,
      upstream = PokerAction.Fold,
      softVetoed = Vector(
        (PokerAction.Call, "bound -0.15"),
        (PokerAction.Raise(2.5), "bound -0.20"),
        (PokerAction.Raise(5.0), "bound -0.30")
      ),
      adjustments = Vector.empty,
      latencyNanos = 1_000_000L
    ))
    val stats = acc.snapshot()
    assertEquals(stats.decisionsWithVeto, 1)
    assertEquals(stats.totalVetoedActions, 3)
    assertEqualsDouble(stats.vetoRate, 1.0, 1e-12) // 1 decision with veto / 1 decision
  }

  test("p95 and p99 latency computed from sorted samples (nearest-rank)") {
    val acc = OverlayMetricsAccumulator()
    // Record 100 decisions with latencies 1ms..100ms
    for i <- 1 to 100 do
      acc.record(makeResult(
        selected = PokerAction.Check,
        upstream = PokerAction.Check,
        softVetoed = Vector.empty,
        adjustments = Vector.empty,
        latencyNanos = i * 1_000_000L
      ))
    val stats = acc.snapshot()
    assertEquals(stats.decisions, 100)
    assertEqualsDouble(stats.meanLatencyMs, 50.5, 1e-6)
    // nearest-rank: ceil(0.95 * 100) - 1 = 94 (0-based) → 95ms
    assertEqualsDouble(stats.p95LatencyMs, 95.0, 1e-6)
    // nearest-rank: ceil(0.99 * 100) - 1 = 98 (0-based) → 99ms
    assertEqualsDouble(stats.p99LatencyMs, 99.0, 1e-6)
  }

  private def makeResult(
      selected: PokerAction,
      upstream: PokerAction,
      softVetoed: Vector[(PokerAction, String)],
      adjustments: Vector[OverlayAdjustment],
      latencyNanos: Long
  ): OverlayMetricsAccumulator.DecisionRecord =
    OverlayMetricsAccumulator.DecisionRecord(
      overlayResult = OverlayResult(
        selectedAction = selected,
        rankedActions = Vector(ActionEvaluation(selected, 1.0)),
        softVetoed = softVetoed,
        adjustments = adjustments,
        upstreamAction = upstream,
        upstreamSource = UpstreamSource.Adaptive
      ),
      totalLatencyNanos = latencyNanos
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly sicfun.holdem.runtime.protocol.OverlayMetricsTest"`
Expected: Compilation error — `OverlayMetricsAccumulator` not found.

- [ ] **Step 3: Write the OverlayMetrics implementation**

```scala
package sicfun.holdem.runtime.protocol

import sicfun.holdem.engine.OverlayResult
import sicfun.holdem.types.{PokerAction, PokerFormatting}

import scala.collection.mutable

/** Immutable snapshot of overlay statistics for a benchmark run. */
final case class OverlayStats(
    decisions: Int,
    overlayChanges: Int,
    overlayChangeRate: Double,
    decisionsWithVeto: Int,
    vetoRate: Double,
    totalVetoedActions: Int,
    adjustments: Int,
    meanLatencyMs: Double,
    p95LatencyMs: Double,
    p99LatencyMs: Double,
    actionDistribution: Map[String, Int]
)

/** Mutable accumulator that collects per-decision overlay data across a match run.
  *
  * Thread-unsafe — used within a single Runner instance, same as MatchStatistics.
  */
final class OverlayMetricsAccumulator:
  private var _decisions = 0
  private var _overlayChanges = 0
  private var _decisionsWithVeto = 0
  private var _totalVetoedActions = 0
  private var _adjustments = 0
  private val _latenciesNanos = mutable.ArrayBuffer.empty[Long]
  private val _actionCounts = mutable.Map.empty[String, Int].withDefaultValue(0)

  def record(rec: OverlayMetricsAccumulator.DecisionRecord): Unit =
    _decisions += 1
    if rec.overlayResult.selectedAction != rec.overlayResult.upstreamAction then
      _overlayChanges += 1
    if rec.overlayResult.softVetoed.nonEmpty then
      _decisionsWithVeto += 1
    _totalVetoedActions += rec.overlayResult.softVetoed.size
    _adjustments += rec.overlayResult.adjustments.size
    _latenciesNanos += rec.totalLatencyNanos
    val key = PokerFormatting.renderAction(rec.overlayResult.selectedAction)
    _actionCounts(key) = _actionCounts(key) + 1

  def snapshot(): OverlayStats =
    val sorted = _latenciesNanos.sorted.toArray
    OverlayStats(
      decisions = _decisions,
      overlayChanges = _overlayChanges,
      overlayChangeRate = if _decisions > 0 then _overlayChanges.toDouble / _decisions else 0.0,
      decisionsWithVeto = _decisionsWithVeto,
      vetoRate = if _decisions > 0 then _decisionsWithVeto.toDouble / _decisions else 0.0,
      totalVetoedActions = _totalVetoedActions,
      adjustments = _adjustments,
      meanLatencyMs = if sorted.nonEmpty then sorted.map(_ / 1e6).sum / sorted.length else 0.0,
      p95LatencyMs = percentileMs(sorted, 0.95),
      p99LatencyMs = percentileMs(sorted, 0.99),
      actionDistribution = _actionCounts.toMap
    )

  /** Nearest-rank percentile: ceil(p * n) - 1, clamped to [0, n-1]. */
  private def percentileMs(sorted: Array[Long], p: Double): Double =
    if sorted.isEmpty then 0.0
    else
      val idx = math.max(0, math.min(math.ceil(p * sorted.length).toInt - 1, sorted.length - 1))
      sorted(idx) / 1e6

object OverlayMetricsAccumulator:
  /** A single strategic decision record with overlay result and total elapsed time. */
  final case class DecisionRecord(
      overlayResult: OverlayResult,
      totalLatencyNanos: Long
  )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `sbt "testOnly sicfun.holdem.runtime.protocol.OverlayMetricsTest"`
Expected: All 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/protocol/OverlayMetrics.scala \
        src/test/scala/sicfun/holdem/runtime/protocol/OverlayMetricsTest.scala
git commit -m "feat(bench): add OverlayMetricsAccumulator for strategic benchmark stats"
```

---

### Task 2: HeroDecisionPipeline — Return StrategicDecisionResult with Timing

**Files:**
- Modify: `src/main/scala/sicfun/holdem/engine/HeroDecisionPipeline.scala`

- [ ] **Step 1: Add StrategicDecisionResult type and update decideHeroStrategic**

In `HeroDecisionPipeline.scala`, add the result type after the existing `StrategicDecisionContext` (around line 56):

```scala
  /** Result of a strategic overlay decision, including the overlay diagnostic and elapsed time. */
  final case class StrategicDecisionResult(
      action: PokerAction,
      overlayResult: OverlayResult,
      totalLatencyNanos: Long
  )
```

Then update `decideHeroStrategic` (around line 165) to return `StrategicDecisionResult` instead of `PokerAction`:

Replace the method signature and body:

```scala
  /** Strategic overlay decision dispatch.
    *
    * Runs the adaptive engine for upstream EVs (via heroCtx.engine, a
    * RealTimeAdaptiveEngine), then filters through the strategic overlay
    * (via strategicCtx.helper, a StrategicLifecycleHelper wrapping StrategicEngine).
    *
    * Returns a StrategicDecisionResult containing the selected action, the full
    * OverlayResult for diagnostics, and total elapsed time for benchmarking.
    */
  def decideHeroStrategic(
      strategicCtx: StrategicDecisionContext,
      heroCtx: HeroDecisionContext
  ): StrategicDecisionResult =
    val startNanos = System.nanoTime()
    // 1. Run adaptive engine for upstream EVs
    val adaptiveResult = heroCtx.engine.decide(
      hero = heroCtx.hero,
      state = heroCtx.state,
      folds = heroCtx.folds,
      villainPos = heroCtx.villainPos,
      observations = heroCtx.observations,
      candidateActions = heroCtx.candidates,
      decisionBudgetMillis = heroCtx.decisionBudgetMillis,
      rng = new Random(heroCtx.rng.nextLong())
    )
    // 2. Overlay: filter adaptive EVs through strategic beliefs
    val overlayResult = strategicCtx.helper.decideWithOverlay(
      heroCtx.state,
      heroCtx.candidates,
      adaptiveResult.decision.recommendation
    )
    val elapsedNanos = System.nanoTime() - startNanos
    StrategicDecisionResult(
      action = overlayResult.selectedAction,
      overlayResult = overlayResult,
      totalLatencyNanos = elapsedNanos
    )
```

- [ ] **Step 2: Update all callers to destructure the result**

There are exactly 3 callers of `decideHeroStrategic`. Each currently expects `PokerAction`
and must now extract `.action`.

**AcpcMatchRunner.scala** — find the `HeroMode.Strategic` branch in `decideHero` (around line 990).
The call looks like:

```scala
              HeroDecisionPipeline.decideHeroStrategic(
                HeroDecisionPipeline.StrategicDecisionContext(state, candidates, helper),
                heroCtx
              )
```

Replace with:

```scala
              HeroDecisionPipeline.decideHeroStrategic(
                HeroDecisionPipeline.StrategicDecisionContext(state, candidates, helper),
                heroCtx
              ).action
```

**SlumbotMatchRunner.scala** — find the same `HeroMode.Strategic` branch in `decideHero`.
Add `.action` to the `decideHeroStrategic(...)` call.

**TexasHoldemPlayingHall.scala** — find the strategic decision call site in the `resolveHand`
function. It calls `decideHeroStrategic` and uses the result as the chosen action.
Add `.action` to the call.

- [ ] **Step 3: Compile check**

Run: `sbt compile`
Expected: Compiles cleanly. The `.action` extraction preserves the existing PokerAction flow.

- [ ] **Step 4: Run existing overlay tests**

Run: `sbt "testOnly sicfun.holdem.engine.StrategicEngineOverlayTest"`
Expected: All tests pass. (These tests may not call `decideHeroStrategic` directly — verify
and update if any do.)

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/engine/HeroDecisionPipeline.scala \
        src/main/scala/sicfun/holdem/runtime/protocol/AcpcMatchRunner.scala \
        src/main/scala/sicfun/holdem/runtime/protocol/SlumbotMatchRunner.scala \
        src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala
git commit -m "feat(engine): return StrategicDecisionResult with timing from decideHeroStrategic"
```

---

### Task 3: Wire Overlay Metrics into ACPC and Slumbot Runners

**Files:**
- Modify: `src/main/scala/sicfun/holdem/runtime/protocol/AcpcMatchRunner.scala`
- Modify: `src/main/scala/sicfun/holdem/runtime/protocol/SlumbotMatchRunner.scala`

- [ ] **Step 1: Add overlay metrics accumulator to AcpcMatchRunner**

In `AcpcMatchRunner.scala`, in the `Runner` class fields (near `strategicHelperOpt`), add:

```scala
    private val overlayMetrics: Option[OverlayMetricsAccumulator] =
      if config.heroMode == HeroMode.Strategic then Some(OverlayMetricsAccumulator())
      else None
```

Add import at the top:

```scala
import sicfun.holdem.runtime.protocol.{OverlayMetricsAccumulator, OverlayStats}
```

- [ ] **Step 2: Record overlay metrics after each strategic decision in AcpcMatchRunner**

In the `HeroMode.Strategic` branch of `decideHero`, change from:

```scala
              HeroDecisionPipeline.decideHeroStrategic(
                HeroDecisionPipeline.StrategicDecisionContext(state, candidates, helper),
                heroCtx
              ).action
```

To:

```scala
              val strategicResult = HeroDecisionPipeline.decideHeroStrategic(
                HeroDecisionPipeline.StrategicDecisionContext(state, candidates, helper),
                heroCtx
              )
              overlayMetrics.foreach(_.record(
                OverlayMetricsAccumulator.DecisionRecord(strategicResult.overlayResult, strategicResult.totalLatencyNanos)
              ))
              strategicResult.action
```

- [ ] **Step 3: Repeat for SlumbotMatchRunner — add accumulator field**

In `SlumbotMatchRunner.scala`, in the `Runner` class fields (near `strategicHelperOpt`), add:

```scala
    private val overlayMetrics: Option[OverlayMetricsAccumulator] =
      if config.heroMode == HeroMode.Strategic then Some(OverlayMetricsAccumulator())
      else None
```

Add the same import as AcpcMatchRunner.

- [ ] **Step 4: Record overlay metrics after each strategic decision in SlumbotMatchRunner**

In the `HeroMode.Strategic` branch of `decideHero`, apply the same pattern as Step 2:

```scala
              val strategicResult = HeroDecisionPipeline.decideHeroStrategic(
                HeroDecisionPipeline.StrategicDecisionContext(state, candidates, helper),
                heroCtx
              )
              overlayMetrics.foreach(_.record(
                OverlayMetricsAccumulator.DecisionRecord(strategicResult.overlayResult, strategicResult.totalLatencyNanos)
              ))
              strategicResult.action
```

- [ ] **Step 5: Compile check**

Run: `sbt compile`
Expected: Compiles cleanly.

- [ ] **Step 6: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/protocol/AcpcMatchRunner.scala \
        src/main/scala/sicfun/holdem/runtime/protocol/SlumbotMatchRunner.scala
git commit -m "feat(protocol): wire overlay metrics accumulator into ACPC and Slumbot runners"
```

---

### Task 4: Wire Overlay Metrics into PlayingHall

**Files:**
- Modify: `src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala`

- [ ] **Step 1: Add overlay metrics accumulator field**

In `TexasHoldemPlayingHall.scala`, in the `HallRunner` class fields (near `strategicHelperOpt`),
add:

```scala
    private var overlayMetricsOpt: Option[OverlayMetricsAccumulator] = None
```

Add import:

```scala
import sicfun.holdem.runtime.protocol.{OverlayMetricsAccumulator, OverlayStats}
```

In `initializeArtifact()`, inside the `if config.heroMode == HeroMode.Strategic` block, after
creating the helper, add:

```scala
        overlayMetricsOpt = Some(OverlayMetricsAccumulator())
```

- [ ] **Step 2: Record overlay metrics at the strategic decision site**

Find where `decideHeroStrategic` is called in the hall (inside `resolveHand` or the
hero decision function). It currently ends with `.action`. Replace with the same pattern:

```scala
            val strategicResult = HeroDecisionPipeline.decideHeroStrategic(strategicCtx, heroCtx)
            overlayMetricsOpt.foreach(_.record(
              OverlayMetricsAccumulator.DecisionRecord(strategicResult.overlayResult, strategicResult.totalLatencyNanos)
            ))
            strategicResult.action
```

Note: The hall may thread the `overlayMetricsOpt` through `resolveHand` parameters, or you
may need to make it accessible from the resolution scope. Check how `strategicHelperOpt`
is threaded and follow the same pattern.

- [ ] **Step 3: Compile check**

Run: `sbt compile`
Expected: Compiles cleanly.

- [ ] **Step 4: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala
git commit -m "feat(hall): wire overlay metrics accumulator into PlayingHall"
```

---

### Task 5: Extend Summaries with Overlay Stats

**Files:**
- Modify: `src/main/scala/sicfun/holdem/runtime/protocol/MatchRunnerSupport.scala`
- Modify: `src/main/scala/sicfun/holdem/runtime/protocol/AcpcMatchRunner.scala`
- Modify: `src/main/scala/sicfun/holdem/runtime/protocol/SlumbotMatchRunner.scala`
- Modify: `src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala`

- [ ] **Step 1: Add overlay stats to RunSummary**

In `MatchRunnerSupport.scala`, add an optional `OverlayStats` field to `RunSummary`
(line 46, after `outDir`):

```scala
      overlayStats: Option[OverlayStats] = None
```

Add import:

```scala
import sicfun.holdem.runtime.protocol.OverlayStats
```

- [ ] **Step 2: Update MatchStatistics.buildSummary to accept overlay stats**

In `MatchRunnerSupport.scala`, update `buildSummary` signature (line 91):

```scala
    def buildSummary(heroMode: HeroMode, modelId: String, outDir: Path, bigBlindChips: Int = 100,
                     overlayStats: Option[OverlayStats] = None): RunSummary =
```

And add the field to the `RunSummary(...)` constructor call:

```scala
        overlayStats = overlayStats
```

- [ ] **Step 3: Update writeSummary to print overlay stats**

In `MatchRunnerSupport.writeSummary` (line 114), after the existing `modelId` line, add:

```scala
    ) ++ summary.overlayStats.toVector.flatMap { os =>
      Vector(
        s"overlayDecisions: ${os.decisions}",
        s"overlayChangeRate: ${PokerFormatting.fmtDouble(os.overlayChangeRate * 100.0, 1)}%",
        s"vetoRate: ${PokerFormatting.fmtDouble(os.vetoRate * 100.0, 1)}%",
        s"decisionsWithVeto: ${os.decisionsWithVeto}",
        s"totalVetoedActions: ${os.totalVetoedActions}",
        s"adjustments: ${os.adjustments}",
        s"meanLatencyMs: ${PokerFormatting.fmtDouble(os.meanLatencyMs, 3)}",
        s"p95LatencyMs: ${PokerFormatting.fmtDouble(os.p95LatencyMs, 3)}",
        s"p99LatencyMs: ${PokerFormatting.fmtDouble(os.p99LatencyMs, 3)}",
        s"actionDistribution: ${os.actionDistribution.toVector.sortBy(-_._2).map((k,v) => s"$k=$v").mkString(", ")}"
      )
    }
```

Note: The existing `lines` value ends with `modelId`. Change the code so the overlay lines
are appended. The simplest approach: change `val lines = Vector(...)` to
`val baseLines = Vector(...)`, then add `val lines = baseLines ++ overlayLines`.

- [ ] **Step 4: Wire overlay stats into AcpcMatchRunner.buildSummary**

In `AcpcMatchRunner.scala`, the `buildSummary()` method (line 1089) calls:

```scala
    private def buildSummary(): MatchRunnerSupport.RunSummary =
      stats.buildSummary(heroMode = config.heroMode, modelId = modelId, outDir = config.outDir, bigBlindChips = AcpcActionCodec.BigBlindChips)
```

Update to:

```scala
    private def buildSummary(): MatchRunnerSupport.RunSummary =
      stats.buildSummary(
        heroMode = config.heroMode, modelId = modelId, outDir = config.outDir,
        bigBlindChips = AcpcActionCodec.BigBlindChips,
        overlayStats = overlayMetrics.map(_.snapshot())
      )
```

- [ ] **Step 5: Wire overlay stats into SlumbotMatchRunner.buildSummary**

In `SlumbotMatchRunner.scala`, the `buildSummary()` method (line 877) calls:

```scala
    private def buildSummary(): MatchRunnerSupport.RunSummary =
      stats.buildSummary(heroMode = config.heroMode, modelId = modelId, outDir = config.outDir, bigBlindChips = SlumbotActionCodec.BigBlindChips)
```

Update to:

```scala
    private def buildSummary(): MatchRunnerSupport.RunSummary =
      stats.buildSummary(
        heroMode = config.heroMode, modelId = modelId, outDir = config.outDir,
        bigBlindChips = SlumbotActionCodec.BigBlindChips,
        overlayStats = overlayMetrics.map(_.snapshot())
      )
```

- [ ] **Step 6: Wire overlay stats into PlayingHall summary**

In `TexasHoldemPlayingHall.scala`, in `buildSummary()` (line 598), the `HallSummary`
constructor does not use `MatchRunnerSupport.RunSummary`. Instead it has its own type.

Add an optional `OverlayStats` field to `HallSummary` (line 136, after `perVillainNetChips`):

```scala
      overlayStats: Option[OverlayStats] = None
```

In `buildSummary()` (line 619), before the closing `)`, add:

```scala
        overlayStats = overlayMetricsOpt.map(_.snapshot())
```

In the `main` method where `HallSummary` fields are printed (search for
`println(s"actionCounts:` or similar), add after the last existing println:

```scala
        summary.overlayStats.foreach { os =>
          println(s"overlayDecisions: ${os.decisions}")
          println(f"overlayChangeRate: ${os.overlayChangeRate * 100.0}%.1f%%")
          println(f"vetoRate: ${os.vetoRate * 100.0}%.1f%%")
          println(s"decisionsWithVeto: ${os.decisionsWithVeto}")
          println(s"totalVetoedActions: ${os.totalVetoedActions}")
          println(f"meanLatencyMs: ${os.meanLatencyMs}%.3f")
          println(f"p95LatencyMs: ${os.p95LatencyMs}%.3f")
          println(f"p99LatencyMs: ${os.p99LatencyMs}%.3f")
          println(s"actionDistribution: ${os.actionDistribution.toVector.sortBy(-_._2).map((k,v) => s"$k=$v").mkString(", ")}")
        }
```

- [ ] **Step 7: Compile check**

Run: `sbt compile`
Expected: Compiles cleanly.

- [ ] **Step 8: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/protocol/MatchRunnerSupport.scala \
        src/main/scala/sicfun/holdem/runtime/protocol/AcpcMatchRunner.scala \
        src/main/scala/sicfun/holdem/runtime/protocol/SlumbotMatchRunner.scala \
        src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala
git commit -m "feat(bench): extend RunSummary and HallSummary with overlay stats"
```

---

### Task 6: Update Benchmark Scripts for Strategic Mode

**Files:**
- Modify: `scripts/match/run-hall-matchups.ps1`
- Modify: `scripts/match/run-slumbot-benchmark.ps1`

- [ ] **Step 1: Update run-hall-matchups.ps1**

In `scripts/match/run-hall-matchups.ps1`, find the `Resolve-HeroStyles` function
(around line 176). The allowed list is:

```powershell
  $allowed = @("adaptive", "gto")
```

Replace with:

```powershell
  $allowed = @("adaptive", "gto", "strategic")
```

Also find the `$HeroStyles` parameter default (around line 8):

```powershell
    [string]$HeroStyles = "adaptive,gto",
```

Leave the default as `"adaptive,gto"` — strategic is opt-in. The change only allows it
when explicitly requested.

- [ ] **Step 2: Update run-slumbot-benchmark.ps1**

In `scripts/match/run-slumbot-benchmark.ps1`, find the allowed modes validation
(around line 70):

```powershell
$allowedModes = @("adaptive", "gto")
```

Replace with:

```powershell
$allowedModes = @("adaptive", "gto", "strategic")
```

And update the error message (around line 73):

```powershell
    throw "Unsupported hero mode '$mode'. Allowed: adaptive, gto, strategic."
```

- [ ] **Step 3: Verify scripts parse without errors**

Syntax-check only (no execution):

Run: `pwsh -c "Get-Command ./scripts/match/run-hall-matchups.ps1 | Select-Object -ExpandProperty Parameters | ForEach-Object { $_.Keys }" 2>&1`
Expected: Lists parameter names including `HeroStyles` without parse errors.

Run: `pwsh -c "Get-Command ./scripts/match/run-slumbot-benchmark.ps1 | Select-Object -ExpandProperty Parameters | ForEach-Object { $_.Keys }" 2>&1`
Expected: Lists parameter names including `HeroModes` without parse errors.

- [ ] **Step 4: Commit**

```bash
git add scripts/match/run-hall-matchups.ps1 \
        scripts/match/run-slumbot-benchmark.ps1
git commit -m "feat(scripts): allow strategic hero mode in benchmark scripts"
```

---

### Task 7: Decision Corpus Format, Fixtures, and Replay Harness

**Files:**
- Create: `src/main/scala/sicfun/holdem/bench/DecisionCorpusBenchmark.scala`
- Create: `src/test/scala/sicfun/holdem/bench/DecisionCorpusBenchmarkTest.scala`

This task creates a fixed set of decision spots and a replay harness that runs each spot
through adaptive and strategic modes, recording per-spot metrics to a TSV file.

GTO mode is excluded from the corpus replay because it requires a CFR solve per spot
(seconds, not milliseconds) and would make the benchmark impractically slow for
before/after regression checks.

- [ ] **Step 1: Write the failing test**

```scala
package sicfun.holdem.bench

class DecisionCorpusBenchmarkTest extends munit.FunSuite:

  test("corpus has at least 6 spots covering required categories") {
    val corpus = DecisionCorpusBenchmark.corpus
    assert(corpus.size >= 6, s"corpus has only ${corpus.size} spots, need >= 6")
    // Required coverage: preflop, postflop flop, postflop turn, postflop river
    val streets = corpus.map(_.state.street).toSet
    assert(streets.contains(sicfun.holdem.types.Street.Preflop), "missing preflop spot")
    assert(streets.contains(sicfun.holdem.types.Street.Flop), "missing flop spot")
    assert(streets.contains(sicfun.holdem.types.Street.Turn), "missing turn spot")
    assert(streets.contains(sicfun.holdem.types.Street.River), "missing river spot")
  }

  test("corpus spots have non-empty candidates") {
    DecisionCorpusBenchmark.corpus.foreach { spot =>
      assert(spot.candidates.nonEmpty, s"spot ${spot.id} has empty candidates")
    }
  }

  test("replay produces deterministic results with fixed seed") {
    val results1 = DecisionCorpusBenchmark.replayAll(seed = 42L)
    val results2 = DecisionCorpusBenchmark.replayAll(seed = 42L)
    assertEquals(results1.size, results2.size)
    results1.zip(results2).foreach { (r1, r2) =>
      assertEquals(r1.spotId, r2.spotId)
      assertEquals(r1.mode, r2.mode)
      assertEquals(r1.selectedAction, r2.selectedAction,
        s"non-deterministic result for spot=${r1.spotId} mode=${r1.mode}")
    }
  }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly sicfun.holdem.bench.DecisionCorpusBenchmarkTest"`
Expected: Compilation error — `DecisionCorpusBenchmark` not found.

- [ ] **Step 3: Write the corpus and replay harness**

```scala
package sicfun.holdem.bench

import sicfun.core.{Card, Rank, Suit}
import sicfun.holdem.engine.*
import sicfun.holdem.engine.inference.{ActionEvaluation, VillainObservation}
import sicfun.holdem.runtime.StrategicLifecycleHelper
import sicfun.holdem.types.*
import sicfun.holdem.equity.{PreflopFold, TableFormat, TableRanges}
import sicfun.holdem.model.PokerActionModel
import sicfun.holdem.strategic.types.PlayerId

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import java.util.Random

/** Fixed decision corpus for deterministic before/after regression checks.
  *
  * Each spot defines a game state, hero hole, candidates, and villain context.
  * The replay harness runs every spot through adaptive and strategic modes.
  */
object DecisionCorpusBenchmark:

  /** A single decision spot in the corpus. */
  final case class Spot(
      id: String,
      description: String,
      state: GameState,
      heroHole: HoleCards,
      candidates: Vector[PokerAction],
      villainPos: Position,
      observations: Vector[VillainObservation]
  )

  /** Result of replaying one spot through one mode. */
  final case class SpotResult(
      spotId: String,
      mode: String,
      selectedAction: PokerAction,
      upstreamAction: Option[PokerAction],
      perActionEvs: Vector[(PokerAction, Double)],
      overlayChanged: Boolean,
      softVetoCount: Int,
      adjustmentCount: Int,
      latencyMs: Double
  )

  // --- Helper to construct VillainObservation(action, GameState) concisely ---
  private def obs(action: PokerAction, street: Street, board: Board,
                  pot: Double, toCall: Double, position: Position, stack: Double): VillainObservation =
    VillainObservation(action, GameState(
      street = street, board = board, pot = pot, toCall = toCall,
      position = position, stackSize = stack, betHistory = Vector.empty
    ))

  // --- Boards used across spots ---
  private val boardA72r = Board(Vector(
    Card(Rank.Ace, Suit.Hearts), Card(Rank.Seven, Suit.Diamonds), Card(Rank.Two, Suit.Clubs)))
  private val boardA72r4s = Board(Vector(
    Card(Rank.Ace, Suit.Hearts), Card(Rank.Seven, Suit.Diamonds), Card(Rank.Two, Suit.Clubs),
    Card(Rank.Four, Suit.Spades)))
  private val boardK95r32 = Board(Vector(
    Card(Rank.King, Suit.Hearts), Card(Rank.Nine, Suit.Diamonds), Card(Rank.Five, Suit.Clubs),
    Card(Rank.Three, Suit.Spades), Card(Rank.Two, Suit.Hearts)))
  private val boardQQ838 = Board(Vector(
    Card(Rank.Queen, Suit.Hearts), Card(Rank.Queen, Suit.Diamonds), Card(Rank.Eight, Suit.Clubs),
    Card(Rank.Three, Suit.Spades), Card(Rank.Eight, Suit.Hearts)))

  /** The fixed corpus. These spots must not change across runs. */
  val corpus: Vector[Spot] = Vector(
    // 1. Preflop open: hero on button with AKs
    Spot(
      id = "preflop-open-aks",
      description = "Hero on button, AKs, no prior action",
      state = GameState(
        street = Street.Preflop, board = Board.empty, pot = 1.5, toCall = 0.5,
        position = Position.Button, stackSize = 100.0, betHistory = Vector.empty),
      heroHole = HoleCards(Card(Rank.Ace, Suit.Spades), Card(Rank.King, Suit.Spades)),
      candidates = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(2.5)),
      villainPos = Position.BigBlind,
      observations = Vector.empty
    ),
    // 2. Preflop facing 3bet: hero on button with QQ
    Spot(
      id = "preflop-3bet-qq",
      description = "Hero on button, QQ, facing 3bet from BB",
      state = GameState(
        street = Street.Preflop, board = Board.empty, pot = 7.5, toCall = 5.0,
        position = Position.Button, stackSize = 97.5, betHistory = Vector(2.5, 7.5)),
      heroHole = HoleCards(Card(Rank.Queen, Suit.Hearts), Card(Rank.Queen, Suit.Diamonds)),
      candidates = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(15.0)),
      villainPos = Position.BigBlind,
      observations = Vector(
        obs(PokerAction.Raise(7.5), Street.Preflop, Board.empty,
            pot = 3.0, toCall = 2.5, position = Position.BigBlind, stack = 99.0)
      )
    ),
    // 3. Flop c-bet: hero on button, KK on A72r
    Spot(
      id = "flop-cbet-kk",
      description = "Hero on button, KK, flop A72 rainbow, first to act",
      state = GameState(
        street = Street.Flop, board = boardA72r, pot = 6.0, toCall = 0.0,
        position = Position.Button, stackSize = 97.0, betHistory = Vector.empty),
      heroHole = HoleCards(Card(Rank.King, Suit.Spades), Card(Rank.King, Suit.Clubs)),
      candidates = Vector(PokerAction.Check, PokerAction.Raise(3.0), PokerAction.Raise(4.5)),
      villainPos = Position.BigBlind,
      observations = Vector(
        obs(PokerAction.Call, Street.Preflop, Board.empty,
            pot = 3.0, toCall = 2.0, position = Position.BigBlind, stack = 99.0)
      )
    ),
    // 4. Turn barrel: hero on button, overpair on turn brick
    Spot(
      id = "turn-barrel-kk",
      description = "Hero on button, KK, turn 4s after flop cbet called",
      state = GameState(
        street = Street.Turn, board = boardA72r4s, pot = 12.0, toCall = 0.0,
        position = Position.Button, stackSize = 94.0, betHistory = Vector.empty),
      heroHole = HoleCards(Card(Rank.King, Suit.Spades), Card(Rank.King, Suit.Clubs)),
      candidates = Vector(PokerAction.Check, PokerAction.Raise(6.0), PokerAction.Raise(9.0)),
      villainPos = Position.BigBlind,
      observations = Vector(
        obs(PokerAction.Call, Street.Preflop, Board.empty,
            pot = 3.0, toCall = 2.0, position = Position.BigBlind, stack = 99.0),
        obs(PokerAction.Call, Street.Flop, boardA72r,
            pot = 9.0, toCall = 3.0, position = Position.BigBlind, stack = 96.0)
      )
    ),
    // 5. River bluff-catch: hero on BB, A-high facing large bet
    Spot(
      id = "river-bluffcatch-ahigh",
      description = "Hero on BB, Ace-high on river, facing pot-sized bet",
      state = GameState(
        street = Street.River, board = boardK95r32, pot = 24.0, toCall = 24.0,
        position = Position.BigBlind, stackSize = 76.0, betHistory = Vector(24.0)),
      heroHole = HoleCards(Card(Rank.Ace, Suit.Diamonds), Card(Rank.Jack, Suit.Clubs)),
      candidates = Vector(PokerAction.Fold, PokerAction.Call),
      villainPos = Position.Button,
      observations = Vector(
        obs(PokerAction.Raise(2.5), Street.Preflop, Board.empty,
            pot = 1.5, toCall = 0.5, position = Position.Button, stack = 100.0),
        obs(PokerAction.Raise(4.0), Street.Flop,
            Board(Vector(Card(Rank.King, Suit.Hearts), Card(Rank.Nine, Suit.Diamonds), Card(Rank.Five, Suit.Clubs))),
            pot = 5.0, toCall = 0.0, position = Position.Button, stack = 97.5),
        obs(PokerAction.Raise(8.0), Street.Turn,
            Board(Vector(Card(Rank.King, Suit.Hearts), Card(Rank.Nine, Suit.Diamonds),
              Card(Rank.Five, Suit.Clubs), Card(Rank.Three, Suit.Spades))),
            pot = 13.0, toCall = 0.0, position = Position.Button, stack = 93.5),
        obs(PokerAction.Raise(24.0), Street.River, boardK95r32,
            pot = 29.0, toCall = 0.0, position = Position.Button, stack = 85.5)
      )
    ),
    // 6. River value bet: hero on button, full house on paired board
    Spot(
      id = "river-valuebet-fullhouse",
      description = "Hero on button, full house on river, checking to hero",
      state = GameState(
        street = Street.River, board = boardQQ838, pot = 20.0, toCall = 0.0,
        position = Position.Button, stackSize = 90.0, betHistory = Vector.empty),
      heroHole = HoleCards(Card(Rank.Queen, Suit.Spades), Card(Rank.Eight, Suit.Spades)),
      candidates = Vector(PokerAction.Check, PokerAction.Raise(10.0), PokerAction.Raise(20.0)),
      villainPos = Position.BigBlind,
      observations = Vector(
        obs(PokerAction.Call, Street.Preflop, Board.empty,
            pot = 3.0, toCall = 2.0, position = Position.BigBlind, stack = 99.0),
        obs(PokerAction.Check, Street.Flop,
            Board(Vector(Card(Rank.Queen, Suit.Hearts), Card(Rank.Queen, Suit.Diamonds), Card(Rank.Eight, Suit.Clubs))),
            pot = 6.0, toCall = 0.0, position = Position.BigBlind, stack = 97.0),
        obs(PokerAction.Check, Street.Turn,
            Board(Vector(Card(Rank.Queen, Suit.Hearts), Card(Rank.Queen, Suit.Diamonds),
              Card(Rank.Eight, Suit.Clubs), Card(Rank.Three, Suit.Spades))),
            pot = 6.0, toCall = 0.0, position = Position.BigBlind, stack = 97.0),
        obs(PokerAction.Check, Street.River, boardQQ838,
            pot = 20.0, toCall = 0.0, position = Position.BigBlind, stack = 90.0)
      )
    )
  )

  /** Replay all corpus spots through adaptive and strategic modes.
    * Returns a flat vector of per-spot-per-mode results.
    */
  def replayAll(
      seed: Long = 42L,
      equityTrials: Int = 600,
      bunchingTrials: Int = 1
  ): Vector[SpotResult] =
    val tableRanges = TableRanges.defaults(TableFormat.HeadsUp)
    val folds = TableFormat.HeadsUp.foldsBeforeOpener(Position.Button).map(PreflopFold(_))
    val model = PokerActionModel.uniform

    // Create shared engines
    val preflopEngine = HeroDecisionPipeline.newAdaptiveEngine(
      tableRanges = tableRanges,
      model = model,
      equityTrials = equityTrials,
      bunchingTrials = bunchingTrials
    )
    val strategicHelper = StrategicLifecycleHelper.create()
    strategicHelper.initSession(
      rivalIds = Vector(PlayerId("villain")),
      positionMapping = Map.empty
    )

    val results = Vector.newBuilder[SpotResult]
    corpus.foreach { spot =>
      val spotRng = new Random(seed ^ spot.id.hashCode.toLong)

      // Replay through adaptive mode
      val adaptiveCtx = HeroDecisionPipeline.HeroDecisionContext(
        hero = spot.heroHole,
        state = spot.state,
        folds = folds,
        tableRanges = tableRanges,
        villainPos = spot.villainPos,
        observations = spot.observations,
        candidates = spot.candidates,
        engine = preflopEngine,
        actionModel = model,
        bunchingTrials = bunchingTrials,
        cfrIterations = 100,
        cfrVillainHands = 50,
        cfrEquityTrials = equityTrials,
        rng = new scala.util.Random(spotRng.nextLong())
      )
      val adaptiveStart = System.nanoTime()
      val adaptiveAction = HeroDecisionPipeline.decideHero(HeroMode.Adaptive, adaptiveCtx)
      val adaptiveElapsed = System.nanoTime() - adaptiveStart
      results += SpotResult(
        spotId = spot.id,
        mode = "adaptive",
        selectedAction = adaptiveAction,
        upstreamAction = None,
        perActionEvs = Vector.empty,
        overlayChanged = false,
        softVetoCount = 0,
        adjustmentCount = 0,
        latencyMs = adaptiveElapsed / 1e6
      )

      // Replay through strategic mode
      strategicHelper.updatePositionMapping(Map(spot.villainPos -> PlayerId("villain")))
      strategicHelper.startHand(spot.heroHole)
      spot.observations.foreach { obs =>
        // VillainObservation carries its own GameState — use it directly, NOT the
        // terminal decision state (spot.state), which would collapse all belief
        // updates onto the wrong game context.
        strategicHelper.observeVillainAction(spot.villainPos, obs.action, obs.state)
      }
      val strategicCtx = HeroDecisionPipeline.StrategicDecisionContext(
        state = spot.state,
        candidates = spot.candidates,
        helper = strategicHelper
      )
      val heroCtx = HeroDecisionPipeline.HeroDecisionContext(
        hero = spot.heroHole,
        state = spot.state,
        folds = folds,
        tableRanges = tableRanges,
        villainPos = spot.villainPos,
        observations = spot.observations,
        candidates = spot.candidates,
        engine = preflopEngine,
        actionModel = model,
        bunchingTrials = bunchingTrials,
        cfrIterations = 100,
        cfrVillainHands = 50,
        cfrEquityTrials = equityTrials,
        rng = new scala.util.Random(spotRng.nextLong())
      )
      val strategicResult = HeroDecisionPipeline.decideHeroStrategic(strategicCtx, heroCtx)
      strategicHelper.endHand()
      results += SpotResult(
        spotId = spot.id,
        mode = "strategic",
        selectedAction = strategicResult.action,
        upstreamAction = Some(strategicResult.overlayResult.upstreamAction),
        perActionEvs = strategicResult.overlayResult.rankedActions.map(ae => (ae.action, ae.expectedValue)),
        overlayChanged = strategicResult.action != strategicResult.overlayResult.upstreamAction,
        softVetoCount = strategicResult.overlayResult.softVetoed.size,
        adjustmentCount = strategicResult.overlayResult.adjustments.size,
        latencyMs = strategicResult.totalLatencyNanos / 1e6
      )
    }
    results.result()

  /** Write corpus results to a TSV file for comparison. */
  def writeResults(results: Vector[SpotResult], path: Path): Unit =
    Files.createDirectories(path.getParent)
    val header = "spotId\tmode\tselectedAction\tupstreamAction\tperActionEvs\toverlayChanged\tsoftVetoCount\tadjustmentCount\tlatencyMs"
    val rows = results.map { r =>
      val evsStr = if r.perActionEvs.isEmpty then "-"
        else r.perActionEvs.map((a, ev) => f"${PokerFormatting.renderAction(a)}=${ev}%.4f").mkString(",")
      Vector(
        r.spotId,
        r.mode,
        PokerFormatting.renderAction(r.selectedAction),
        r.upstreamAction.map(PokerFormatting.renderAction).getOrElse("-"),
        evsStr,
        r.overlayChanged.toString,
        r.softVetoCount.toString,
        r.adjustmentCount.toString,
        f"${r.latencyMs}%.3f"
      ).mkString("\t")
    }
    Files.write(path, (header +: rows).mkString(System.lineSeparator()).getBytes(StandardCharsets.UTF_8))

  /** Main entry point for standalone benchmark runs. */
  def main(args: Array[String]): Unit =
    val seed = args.headOption.flatMap(_.toLongOption).getOrElse(42L)
    val outDir = Path.of(if args.length > 1 then args(1) else "data/bench-decision-corpus")
    println(s"Running decision corpus benchmark (seed=$seed)")
    val results = replayAll(seed = seed)
    val outPath = outDir.resolve("corpus-results.tsv")
    writeResults(results, outPath)
    println(s"Wrote ${results.size} results to $outPath")
    // Print summary
    val strategic = results.filter(_.mode == "strategic")
    val changes = strategic.count(_.overlayChanged)
    val vetoes = strategic.map(_.softVetoCount).sum
    println(f"Strategic: ${strategic.size} spots, $changes overlay changes, $vetoes soft vetoes")
    println(f"Mean strategic latency: ${strategic.map(_.latencyMs).sum / strategic.size}%.3f ms")
```

Note: `HeadsUpMatchDefaults` is `private[runtime]` so the bench package cannot import it
directly. The corpus uses the same public APIs that `HeadsUpMatchDefaults` wraps:
`TableRanges.defaults(TableFormat.HeadsUp)` and
`TableFormat.HeadsUp.foldsBeforeOpener(Position.Button).map(PreflopFold(_))`.
The bootstrap model follows the same pattern as `AcpcMatchRunner.loadArtifact`'s
`None` branch (uniform model with dummy calibration).

- [ ] **Step 4: Run test to verify it passes**

Run: `sbt "testOnly sicfun.holdem.bench.DecisionCorpusBenchmarkTest"`
Expected: All 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/bench/DecisionCorpusBenchmark.scala \
        src/test/scala/sicfun/holdem/bench/DecisionCorpusBenchmarkTest.scala
git commit -m "feat(bench): add fixed decision corpus with deterministic replay harness"
```

---

### Task 8: Baseline Capture Script

**Files:**
- Create: `scripts/bench/capture-baseline.ps1`

- [ ] **Step 1: Create the baseline capture script**

```powershell
<#
.SYNOPSIS
    Captures Phase 2 baseline benchmarks across hall, decision corpus, and optionally Slumbot.

.DESCRIPTION
    Runs all benchmark surfaces with fixed seeds and saves artifacts to a timestamped
    output directory. Must be run BEFORE any Track A formulation changes merge.

    Surfaces:
    1. Decision corpus replay (deterministic, local)
    2. Hall self-play: adaptive vs strategic vs gto (1000 hands each, seed=42)
    3. Slumbot benchmark: adaptive vs strategic (50 hands each, seed=42) [optional]

.PARAMETER OutDir
    Root output directory. Default: data/bench-baseline-phase2

.PARAMETER Hands
    Hands per hall matchup. Default: 1000

.PARAMETER Seed
    RNG seed. Default: 42

.PARAMETER SkipSlumbot
    Skip the Slumbot benchmark (requires network access).

.PARAMETER SkipHall
    Skip the hall matchups.
#>
param(
    [string]$OutDir = "data/bench-baseline-phase2",
    [int]$Hands = 1000,
    [long]$Seed = 42,
    [switch]$SkipSlumbot,
    [switch]$SkipHall
)

$ErrorActionPreference = "Stop"
$timestamp = Get-Date -Format "yyyy-MM-dd-HHmm"
$baseDir = Join-Path $OutDir $timestamp

Write-Host "=== Phase 2 Baseline Capture ===" -ForegroundColor Cyan
Write-Host "Output: $baseDir"
Write-Host "Seed: $Seed"
Write-Host ""

# --- 1. Decision Corpus ---
Write-Host "--- Decision Corpus Benchmark ---" -ForegroundColor Yellow
$corpusDir = Join-Path $baseDir "decision-corpus"
sbt "runMain sicfun.holdem.bench.DecisionCorpusBenchmark $Seed $corpusDir"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Decision corpus benchmark failed"
    exit 1
}
Write-Host ""

# --- 2. Hall Matchups (via run-hall-matchups.ps1 for summary artifact output) ---
if (-not $SkipHall) {
    Write-Host "--- Hall Self-Play Benchmarks ---" -ForegroundColor Yellow
    $hallDir = Join-Path $baseDir "hall"
    & "$PSScriptRoot/../match/run-hall-matchups.ps1" `
        -HeroStyles "adaptive,gto,strategic" `
        -Hands $Hands `
        -Seed $Seed `
        -OutDir $hallDir
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Hall matchups failed (exit code $LASTEXITCODE)"
    }
    Write-Host ""
}

# --- 3. Slumbot ---
if (-not $SkipSlumbot) {
    Write-Host "--- Slumbot Benchmarks ---" -ForegroundColor Yellow
    $slumbotModes = @("adaptive", "strategic")
    foreach ($mode in $slumbotModes) {
        Write-Host "  Running slumbot: $mode (50 hands, seed=$Seed)" -ForegroundColor Gray
        $slumbotDir = Join-Path $baseDir "slumbot-$mode"
        sbt "runMain sicfun.holdem.runtime.protocol.SlumbotMatchRunner --hands=50 --heroMode=$mode --seed=$Seed --outDir=$slumbotDir"
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Slumbot $mode failed (exit code $LASTEXITCODE)"
        }
    }
    Write-Host ""
}

# --- Summary ---
Write-Host "=== Baseline Capture Complete ===" -ForegroundColor Cyan
Write-Host "Artifacts saved to: $baseDir"

# Write metadata
$metaPath = Join-Path $baseDir "baseline-meta.txt"
@"
Phase 2 Baseline Capture
Date: $timestamp
Seed: $Seed
HallHands: $Hands
SkipSlumbot: $SkipSlumbot
SkipHall: $SkipHall
GitCommit: $(git rev-parse HEAD)
GitBranch: $(git branch --show-current)
"@ | Set-Content -Path $metaPath -Encoding UTF8

Write-Host "Metadata written to: $metaPath"
```

- [ ] **Step 2: Verify script parses**

Run: `pwsh -c "Get-Help ./scripts/bench/capture-baseline.ps1"`
Expected: Shows the synopsis and parameter descriptions without errors.

- [ ] **Step 3: Commit**

```bash
git add scripts/bench/capture-baseline.ps1
git commit -m "feat(scripts): add Phase 2 baseline capture script"
```

---

### Task 9: Full Test Suite Verification

**Files:** None (verification only)

- [ ] **Step 1: Run full compile**

Run: `sbt compile`
Expected: Compiles cleanly with no warnings (project uses `-Werror`).

- [ ] **Step 2: Run full test suite**

Run: `sbt test`
Expected: All tests pass (1763+ — may increase from new OverlayMetricsTest and
DecisionCorpusBenchmarkTest).

- [ ] **Step 3: If any tests fail, investigate and fix**

Common failure modes:
- Import mismatches — check package paths
- Type mismatches from `StrategicDecisionResult` — callers need `.action` extraction
- Missing factory methods in `DecisionCorpusBenchmark` — adapt to actual API names
- `VillainObservation` constructor differences — check actual fields

- [ ] **Step 4: Commit any fixes**

```bash
git add -u
git commit -m "fix: resolve test failures from Track B benchmark harness"
```

---

### Task 10: Run Baseline Capture (G1 and G2 Gates)

**Files:** None (runtime verification)

This task captures the actual baseline and verifies Gates G1 (baseline exists) and
G2 (determinism).

- [ ] **Step 1: Run the baseline capture script (hall + corpus only)**

Run: `pwsh -c "./scripts/bench/capture-baseline.ps1 -SkipSlumbot -Hands 1000"`
Expected: Completes without errors. Creates `data/bench-baseline-phase2/<timestamp>/` with:
- `decision-corpus/corpus-results.tsv`
- `hall/results.tsv` — one row per hero-style x villain matchup
- `hall/summary.txt` — leaderboard across all matchups
- `hall/NN-<hero>-vs-<villain>/` — per-matchup: `hands.tsv`, `stdout.log`, `stderr.log`
- `baseline-meta.txt`

- [ ] **Step 2: Verify G2 determinism gate — re-run decision corpus with same seed**

Run: `sbt "runMain sicfun.holdem.bench.DecisionCorpusBenchmark 42 data/bench-baseline-phase2/verify-determinism"`

Compare corpus-results.tsv files (strip the latencyMs column which varies by run, then
compare the deterministic columns):

Run:
```bash
git diff --no-index --no-color -- \
  <(cut -f1-7 data/bench-baseline-phase2/<timestamp>/decision-corpus/corpus-results.tsv) \
  <(cut -f1-7 data/bench-baseline-phase2/verify-determinism/corpus-results.tsv)
```

Or in PowerShell:
```powershell
$a = Import-Csv -Delimiter "`t" "data/bench-baseline-phase2/<timestamp>/decision-corpus/corpus-results.tsv" | Select-Object * -ExcludeProperty latencyMs
$b = Import-Csv -Delimiter "`t" "data/bench-baseline-phase2/verify-determinism/corpus-results.tsv" | Select-Object * -ExcludeProperty latencyMs
Compare-Object $a $b -Property spotId,mode,selectedAction,upstreamAction,perActionEvs,overlayChanged,softVetoCount,adjustmentCount
```

Expected: No differences in the deterministic columns (spotId through adjustmentCount).

- [ ] **Step 3: Verify G1 baseline artifacts exist**

Check that all required files exist. The hall script writes per-matchup directories
(e.g., `01-adaptive-vs-nit/`) plus a `results.tsv` and `summary.txt` at the top level:

```bash
ls data/bench-baseline-phase2/<timestamp>/baseline-meta.txt
ls data/bench-baseline-phase2/<timestamp>/decision-corpus/corpus-results.tsv
ls data/bench-baseline-phase2/<timestamp>/hall/results.tsv
ls data/bench-baseline-phase2/<timestamp>/hall/summary.txt
```

Expected: All files exist and are non-empty. The hall `results.tsv` should contain rows
for adaptive, gto, and strategic hero styles.

- [ ] **Step 4: Print strategic overlay summary from hall run**

Run: `cat data/bench-baseline-phase2/<timestamp>/hall/summary.txt`

Expected: Shows per-matchup leaderboard with strategic hero style entries.

The overlay stats (overlayDecisions, overlayChangeRate, vetoRate, latency) are printed
to stdout by `TexasHoldemPlayingHall` and captured in each matchup's `stdout.log`.
Verify at least one strategic matchup directory has overlay output:

```bash
grep "overlayDecisions" data/bench-baseline-phase2/<timestamp>/hall/*strategic*/stdout.log
```

Expected: `overlayDecisions: <N>` (should be > 0 for strategic matchups).

These are the Phase 2 baseline numbers. Track A changes will be compared against them.

---

## Spec Coverage and Gate Thresholds

### Spec coverage

| Spec Requirement | Plan Coverage |
|------------------|---------------|
| B1: Fixed decision corpus, deterministic replay | Task 7: 6 spots, 4 streets, replay harness |
| B1: Heads-up preflop | Spots 1, 2 |
| B1: Heads-up postflop | Spots 3, 4, 5, 6 |
| B1: Multiway postflop | **Deferred** — Phase 1 overlay is heads-up only in all runners |
| B1: Non-trivial strategic beliefs | Spots 5-6 inherit accumulated beliefs from spots 1-4 (session persists) |
| B1: Certification/robust-bound spot | **Deferred** — requires deprecated 2-arg decide path; covered when Track A wires grounded certification |
| B1: Per-spot metrics (action, upstream, EVs, change rate, veto, latency) | SpotResult covers all 6 |
| B2: Hall self-play | Tasks 4-5, capture script |
| B2: ACPC local match | Metrics wired (Task 3); capture requires dealer setup — documented below |
| B2: Slumbot benchmark | Tasks 3, 5-6, capture script (--SkipSlumbot for offline) |
| B2: Scripts accept strategic | Task 6 |
| B2: Per-run metrics (bb/100, action dist, veto rate, latency) | OverlayStats in summary |
| B2: Exploitation beta trajectory | **Deferred** — requires per-hand extraction from StrategicEngine internals |
| B2: Protocol errors / invalid-action failures | Runners throw on protocol errors; no explicit counter added |
| G1: Baseline capture | Task 10 |
| G2: Determinism | Task 10, Step 2 |
| G3: Quality gate | Thresholds defined below |
| G4: Latency gate | Thresholds defined below |
| G5: Safety/diagnostic gate | Covered by overlay stats (veto rate, adjustment counts) |

### Gate thresholds (G3 and G4)

These thresholds apply when Track A changes are gated:

**G3 — Quality:**
- Hall strategic-vs-adaptive: bb/100 regression must not exceed 20 bb/100 over 1000 hands
  (accounts for high variance in short sample; tighten to 10 bb/100 if sample increases to 5000+)
- Hall strategic-vs-gto: same threshold
- ACPC/Slumbot: no negative bb/100 swing > 50 mbb/hand vs baseline
- Decision corpus: no spot may change selectedAction unless the corresponding Track A
  change explicitly justifies the difference

**G4 — Latency:**
- Mean strategic decision latency: must not exceed 2x the baseline mean
- p99 strategic decision latency: must not exceed 3x the baseline p99
- If the baseline mean is < 5ms, absolute deltas (mean < 10ms, p99 < 15ms) apply instead

### ACPC benchmark invocation (manual)

The ACPC benchmark requires starting the dealer process first:

```bash
# Terminal 1: Start dealer
sbt "runMain sicfun.holdem.runtime.protocol.AcpcHeadsUpDealer --hands=1000 --seed=42 --outDir=data/bench-baseline-phase2/<timestamp>/acpc-dealer"

# Terminal 2: Start runner
sbt "runMain sicfun.holdem.runtime.protocol.AcpcMatchRunner --hands=1000 --heroMode=strategic --seed=42 --outDir=data/bench-baseline-phase2/<timestamp>/acpc-strategic"
```

This is not automated in `capture-baseline.ps1` because the dealer and runner must run
concurrently as separate processes. A future enhancement could wrap this in a PowerShell
job pair.

### Items deferred to Track A or later

1. **Multiway corpus spots** — Add when multiway inference is exercised by the overlay path
2. **Certification/robust-bound corpus spots** — Add when Track A grounds the certification path (A5)
3. **Exploitation beta trajectory** — Add per-hand beta extraction when Track A touches exploitation state
4. **ACPC automation in capture script** — Wrap dealer+runner in concurrent PowerShell jobs
