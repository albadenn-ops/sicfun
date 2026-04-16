package sicfun.holdem.runtime.protocol

import sicfun.holdem.engine.OverlayResult
import sicfun.holdem.types.PokerFormatting

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
