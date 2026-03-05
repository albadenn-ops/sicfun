package sicfun.holdem

import sicfun.core.Metrics

/** A behavioral snapshot for a single time window in a longitudinal analysis.
  *
  * @param windowIndex      zero-based index of this window in the sliding sequence
  * @param startEpochMillis inclusive start timestamp of the window
  * @param endEpochMillis   exclusive end timestamp of the window
  * @param eventCount       number of qualifying events that fell within this window
  * @param signature        the [[PlayerSignature]] computed from the window's events
  */
final case class WindowSnapshot(
    windowIndex: Int,
    startEpochMillis: Long,
    endEpochMillis: Long,
    eventCount: Int,
    signature: PlayerSignature
):
  require(windowIndex >= 0, "windowIndex must be non-negative")
  require(endEpochMillis > startEpochMillis, "endEpochMillis must be > startEpochMillis")
  require(eventCount > 0, "eventCount must be positive")

/** Pairwise behavioral drift between two consecutive time windows.
  *
  * @param fromWindowIndex    index of the earlier window
  * @param toWindowIndex      index of the later window
  * @param distance           Euclidean distance between the two windows' [[PlayerSignature]] vectors
  * @param significantChange  `true` if `distance >= driftThreshold` from [[LongitudinalConfig]]
  */
final case class DriftMeasurement(
    fromWindowIndex: Int,
    toWindowIndex: Int,
    distance: Double,
    significantChange: Boolean
):
  require(fromWindowIndex >= 0, "fromWindowIndex must be non-negative")
  require(toWindowIndex >= 0, "toWindowIndex must be non-negative")
  require(distance >= 0.0, "distance must be non-negative")

/** Complete stability report for a single player, summarizing behavioral consistency over time.
  *
  * @param playerId identifier of the analyzed player
  * @param windows  all time windows that contained enough events (see [[LongitudinalConfig.minEventsPerWindow]])
  * @param drifts   pairwise drift measurements between consecutive windows
  * @param maxDrift largest single-step drift distance observed (0.0 if fewer than 2 windows)
  * @param meanDrift arithmetic mean of all drift distances (0.0 if fewer than 2 windows)
  */
final case class StabilityReport(
    playerId: String,
    windows: Vector[WindowSnapshot],
    drifts: Vector[DriftMeasurement],
    maxDrift: Double,
    meanDrift: Double
):
  require(playerId.trim.nonEmpty, "playerId must be non-empty")
  require(maxDrift >= 0.0, "maxDrift must be non-negative")
  require(meanDrift >= 0.0, "meanDrift must be non-negative")

/** Configuration for sliding-window longitudinal behavioral analysis.
  *
  * @param windowSizeMillis   duration of each observation window in milliseconds
  * @param slideStepMillis    stride between consecutive window start times in milliseconds
  * @param driftThreshold     Euclidean distance above which a window-to-window change is "significant" (default 0.2)
  * @param minEventsPerWindow minimum number of events required to compute a window signature (default 5)
  */
final case class LongitudinalConfig(
    windowSizeMillis: Long,
    slideStepMillis: Long,
    driftThreshold: Double = 0.2,
    minEventsPerWindow: Int = 5
):
  require(windowSizeMillis > 0L, "windowSizeMillis must be positive")
  require(slideStepMillis > 0L, "slideStepMillis must be positive")
  require(driftThreshold >= 0.0, "driftThreshold must be non-negative")
  require(minEventsPerWindow > 0, "minEventsPerWindow must be positive")

/** Sliding-window longitudinal analysis of player behavioral drift over time.
  *
  * Given a sequence of [[PokerEvent]]s for a specific player, divides the timeline into
  * overlapping windows of configurable size and stride. For each window with enough events,
  * computes a [[PlayerSignature]] and measures Euclidean drift between consecutive windows.
  * The resulting [[StabilityReport]] reveals whether a player's strategy is stable or shifting.
  */
object LongitudinalAnalysis:
  /** Runs a sliding-window analysis over a player's event history.
    *
    * Events are filtered to the target `playerId`, sorted chronologically, then partitioned
    * into overlapping windows. Each qualifying window produces a [[PlayerSignature]], and
    * consecutive windows are compared using [[PlayerSignature.distance]].
    *
    * @param events   all poker events (may contain multiple players; filtered internally)
    * @param playerId the player to analyze
    * @param config   window size, stride, drift threshold, and minimum event count
    * @return a [[StabilityReport]] summarizing the player's behavioral consistency
    * @throws IllegalArgumentException if `events` is empty or contains no events for `playerId`
    */
  def analyze(
      events: Seq[PokerEvent],
      playerId: String,
      config: LongitudinalConfig
  ): StabilityReport =
    require(playerId.trim.nonEmpty, "playerId must be non-empty")
    require(events.nonEmpty, "events must be non-empty")

    val filtered = events.filter(_.playerId == playerId)
    require(filtered.nonEmpty, s"no events found for playerId=$playerId")

    val ordered = filtered.sortBy(event =>
      (event.occurredAtEpochMillis, event.handId, event.sequenceInHand)
    )
    val orderedVector = ordered.toVector
    val minTs = ordered.head.occurredAtEpochMillis
    val maxTs = ordered.last.occurredAtEpochMillis

    val windows = scala.collection.mutable.ArrayBuffer.empty[WindowSnapshot]
    var start = minTs
    var windowIndex = 0
    var left = 0
    var right = 0
    val totalEvents = orderedVector.length

    while start <= maxTs do
      val end = start + config.windowSizeMillis

      while left < totalEvents && orderedVector(left).occurredAtEpochMillis < start do
        left += 1
      while right < totalEvents && orderedVector(right).occurredAtEpochMillis < end do
        right += 1

      val count = right - left
      if count >= config.minEventsPerWindow then
        val observations = Vector.newBuilder[(GameState, PokerAction)]
        var idx = left
        while idx < right do
          observations += eventToObservation(orderedVector(idx))
          idx += 1
        val signature = PlayerSignature.compute(observations.result())
        windows += WindowSnapshot(
          windowIndex = windowIndex,
          startEpochMillis = start,
          endEpochMillis = end,
          eventCount = count,
          signature = signature
        )
      start += config.slideStepMillis
      windowIndex += 1

    val snapshots = windows.toVector
    val drifts =
      snapshots.sliding(2).collect {
        case Seq(previous, current) =>
          val distance = PlayerSignature.distance(previous.signature, current.signature)
          DriftMeasurement(
            fromWindowIndex = previous.windowIndex,
            toWindowIndex = current.windowIndex,
            distance = distance,
            significantChange = distance >= config.driftThreshold
          )
      }.toVector

    val distances = drifts.map(_.distance)
    val maxDrift = if distances.isEmpty then 0.0 else distances.max
    val meanDrift = if distances.isEmpty then 0.0 else Metrics.mean(distances)

    StabilityReport(
      playerId = playerId,
      windows = snapshots,
      drifts = drifts,
      maxDrift = maxDrift,
      meanDrift = meanDrift
    )

  private def eventToObservation(event: PokerEvent): (GameState, PokerAction) =
    (
      GameState(
        street = event.street,
        board = event.board,
        pot = event.potBefore,
        toCall = event.toCall,
        position = event.position,
        stackSize = event.stackBefore,
        betHistory = event.betHistory
      ),
      event.action
    )
