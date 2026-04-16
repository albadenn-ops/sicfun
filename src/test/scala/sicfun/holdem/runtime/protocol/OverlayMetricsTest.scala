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
