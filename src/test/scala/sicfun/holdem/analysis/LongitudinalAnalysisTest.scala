package sicfun.holdem.analysis
import sicfun.holdem.types.*

import munit.FunSuite
import sicfun.core.Card

/** Tests for [[LongitudinalAnalysis.analyze]], which detects behavioral drift
  * across sliding time windows over a player's event stream.
  *
  * Coverage includes:
  *   - Single-window case (no drift possible, zero aggregates)
  *   - Significant drift between passive and aggressive windows
  *   - Player-id filtering from a mixed-player stream
  *   - Skipping windows below the minimum event threshold
  *   - Start-inclusive / end-exclusive window boundary semantics
  *   - Rejection when no events match the requested player
  *   - Consistency of maxDrift / meanDrift with computed drift distances
  */
class LongitudinalAnalysisTest extends FunSuite:
  /** Parses a card token, failing the test on invalid input. */
  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  /** Builds a synthetic [[PokerEvent]] with a derived toCall amount based on
    * the action type. Uses a fixed flop board (Ts 9h 8d) and Button position.
    */
  private def event(
      playerId: String,
      timestamp: Long,
      sequence: Long,
      action: PokerAction
  ): PokerEvent =
    // Derive toCall from action: calls/folds/raises face a bet; checks do not
    val toCall = action match
      case PokerAction.Call      => 10.0
      case PokerAction.Check     => 0.0
      case PokerAction.Fold      => 10.0
      case PokerAction.Raise(_)  => 10.0
    PokerEvent(
      handId = s"hand-${timestamp / 1000}",
      sequenceInHand = sequence,
      playerId = playerId,
      occurredAtEpochMillis = timestamp,
      street = Street.Flop,
      position = Position.Button,
      board = Board.from(Seq(card("Ts"), card("9h"), card("8d"))),
      potBefore = 20.0,
      toCall = toCall,
      stackBefore = 100.0,
      action = action
    )

  test("single window produces no drift and zero aggregates") {
    val events = Vector(
      event("alice", 0L, 0L, PokerAction.Call),
      event("alice", 100L, 1L, PokerAction.Fold),
      event("alice", 200L, 2L, PokerAction.Call)
    )
    val report = LongitudinalAnalysis.analyze(
      events,
      playerId = "alice",
      config = LongitudinalConfig(windowSizeMillis = 1000L, slideStepMillis = 1000L, minEventsPerWindow = 2)
    )

    assertEquals(report.windows.length, 1)
    assertEquals(report.drifts.length, 0)
    assertEquals(report.maxDrift, 0.0)
    assertEquals(report.meanDrift, 0.0)
  }

  test("detects significant behavior drift between adjacent windows") {
    val firstWindow = Vector(
      event("alice", 0L, 0L, PokerAction.Fold),
      event("alice", 100L, 1L, PokerAction.Fold),
      event("alice", 200L, 2L, PokerAction.Call)
    )
    val secondWindow = Vector(
      event("alice", 1000L, 0L, PokerAction.Raise(20.0)),
      event("alice", 1100L, 1L, PokerAction.Raise(18.0)),
      event("alice", 1200L, 2L, PokerAction.Raise(22.0))
    )

    val report = LongitudinalAnalysis.analyze(
      firstWindow ++ secondWindow,
      playerId = "alice",
      config = LongitudinalConfig(windowSizeMillis = 1000L, slideStepMillis = 1000L, driftThreshold = 0.2, minEventsPerWindow = 3)
    )

    assertEquals(report.windows.length, 2)
    assertEquals(report.drifts.length, 1)
    assert(report.drifts.head.distance > 0.2)
    assert(report.drifts.head.significantChange)
  }

  test("filters by playerId from mixed stream") {
    val aliceEvents = Vector(
      event("alice", 0L, 0L, PokerAction.Call),
      event("alice", 100L, 1L, PokerAction.Fold),
      event("alice", 200L, 2L, PokerAction.Call)
    )
    val bobEvents = Vector(
      event("bob", 10L, 0L, PokerAction.Raise(15.0)),
      event("bob", 110L, 1L, PokerAction.Raise(20.0)),
      event("bob", 210L, 2L, PokerAction.Raise(25.0))
    )

    val report = LongitudinalAnalysis.analyze(
      aliceEvents ++ bobEvents,
      playerId = "alice",
      config = LongitudinalConfig(windowSizeMillis = 1000L, slideStepMillis = 1000L, minEventsPerWindow = 2)
    )

    assertEquals(report.windows.length, 1)
    assertEquals(report.windows.head.eventCount, 3)
  }

  test("skips windows below minEventsPerWindow") {
    val events = Vector(
      event("alice", 0L, 0L, PokerAction.Call),
      event("alice", 100L, 1L, PokerAction.Call),
      event("alice", 200L, 2L, PokerAction.Call),
      event("alice", 1000L, 0L, PokerAction.Fold),
      event("alice", 1100L, 1L, PokerAction.Fold),
      event("alice", 2000L, 0L, PokerAction.Raise(20.0)),
      event("alice", 2100L, 1L, PokerAction.Raise(20.0)),
      event("alice", 2200L, 2L, PokerAction.Raise(20.0))
    )

    val report = LongitudinalAnalysis.analyze(
      events,
      playerId = "alice",
      config = LongitudinalConfig(windowSizeMillis = 1000L, slideStepMillis = 1000L, minEventsPerWindow = 3)
    )

    assertEquals(report.windows.map(_.windowIndex), Vector(0, 2))
    assertEquals(report.drifts.length, 1)
    assertEquals(report.drifts.head.fromWindowIndex, 0)
    assertEquals(report.drifts.head.toWindowIndex, 2)
  }

  test("window semantics are start-inclusive and end-exclusive") {
    val events = Vector(
      event("alice", 0L, 0L, PokerAction.Check),
      event("alice", 1000L, 0L, PokerAction.Check)
    )
    val report = LongitudinalAnalysis.analyze(
      events,
      playerId = "alice",
      config = LongitudinalConfig(windowSizeMillis = 1000L, slideStepMillis = 1000L, minEventsPerWindow = 1)
    )

    assertEquals(report.windows.length, 2)
    assertEquals(report.windows.map(_.eventCount), Vector(1, 1))
  }

  test("rejects when no events match requested player") {
    val events = Vector(
      event("bob", 0L, 0L, PokerAction.Call),
      event("bob", 100L, 1L, PokerAction.Fold)
    )
    intercept[IllegalArgumentException] {
      LongitudinalAnalysis.analyze(
        events,
        playerId = "alice",
        config = LongitudinalConfig(windowSizeMillis = 1000L, slideStepMillis = 1000L, minEventsPerWindow = 1)
      )
    }
  }

  test("maxDrift and meanDrift match computed drift distances") {
    val events = Vector(
      event("alice", 0L, 0L, PokerAction.Call),
      event("alice", 100L, 1L, PokerAction.Fold),
      event("alice", 200L, 2L, PokerAction.Call),
      event("alice", 1000L, 0L, PokerAction.Raise(20.0)),
      event("alice", 1100L, 1L, PokerAction.Raise(20.0)),
      event("alice", 1200L, 2L, PokerAction.Call),
      event("alice", 2000L, 0L, PokerAction.Check),
      event("alice", 2100L, 1L, PokerAction.Check),
      event("alice", 2200L, 2L, PokerAction.Fold)
    )
    val report = LongitudinalAnalysis.analyze(
      events,
      playerId = "alice",
      config = LongitudinalConfig(windowSizeMillis = 1000L, slideStepMillis = 1000L, minEventsPerWindow = 3)
    )
    val distances = report.drifts.map(_.distance)
    assert(distances.nonEmpty)
    assertEquals(report.maxDrift, distances.max)
    assertEquals(report.meanDrift, distances.sum / distances.length)
  }
