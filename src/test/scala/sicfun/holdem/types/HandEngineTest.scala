package sicfun.holdem.types

import sicfun.holdem.io.*

import munit.FunSuite
import sicfun.core.Card

import java.nio.file.Files
import scala.jdk.CollectionConverters.*
import scala.util.Try

class HandEngineTest extends FunSuite:

  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  private def makeEvent(
      handId: String = "h1",
      seq: Long = 0L,
      playerId: String = "player-1",
      ts: Long = 1000L,
      street: Street = Street.Preflop,
      position: Position = Position.Button,
      action: PokerAction = PokerAction.Call,
      pot: Double = 20.0,
      toCall: Double = 2.0,
      stack: Double = 200.0
  ): PokerEvent =
    PokerEvent(
      handId = handId,
      sequenceInHand = seq,
      playerId = playerId,
      occurredAtEpochMillis = ts,
      street = street,
      position = position,
      board = Board.empty,
      potBefore = pot,
      toCall = toCall,
      stackBefore = stack,
      action = action
    )

  // ---- HandEngine: state machine -----------------------------------------------

  test("newHand: creates empty state with correct handId") {
    val state = HandEngine.newHand("h1", 0L)
    assertEquals(state.handId, "h1")
    assert(state.isEmpty)
    assertEquals(state.eventCount, 0)
    assertEquals(state.currentStreet, Street.Preflop)
    assertEquals(state.currentBoard, Board.empty)
  }

  test("newHand: rejects empty or blank handId") {
    intercept[IllegalArgumentException] { HandEngine.newHand("", 0L) }
    intercept[IllegalArgumentException] { HandEngine.newHand("   ", 0L) }
  }

  test("applyEvent: stores single event and updates metadata") {
    val state0 = HandEngine.newHand("h1", 0L)
    val e = makeEvent(seq = 0L, ts = 1500L)
    val state1 = HandEngine.applyEvent(state0, e)
    assertEquals(state1.eventCount, 1)
    assertEquals(state1.lastUpdatedAt, 1500L)
    assertEquals(state1.appliedSequences, Set(0L))
  }

  test("applyEvent: idempotent — duplicate sequenceInHand is silently ignored") {
    val state0 = HandEngine.newHand("h1", 0L)
    val e = makeEvent(seq = 0L, ts = 1000L)
    val state1 = HandEngine.applyEvent(state0, e)
    val state2 = HandEngine.applyEvent(state1, e)  // duplicate
    assertEquals(state2.eventCount, 1)
    assertEquals(state2, state1)
  }

  test("applyEvent: duplicate sequence with different payload is rejected") {
    val state0 = HandEngine.newHand("h1", 0L)
    val e0 = makeEvent(seq = 0L, ts = 1000L, action = PokerAction.Call)
    val conflicting = e0.copy(action = PokerAction.Fold)
    val state1 = HandEngine.applyEvent(state0, e0)
    intercept[IllegalArgumentException] {
      HandEngine.applyEvent(state1, conflicting)
    }
  }

  test("applyEvent: out-of-order delivery produces sorted events") {
    val state0 = HandEngine.newHand("h1", 0L)
    val e2 = makeEvent(seq = 2L, ts = 1200L, action = PokerAction.Fold)
    val e0 = makeEvent(seq = 0L, ts = 1000L, action = PokerAction.Call)
    val e1 = makeEvent(seq = 1L, ts = 1100L, action = PokerAction.Check, toCall = 0.0)
    val state = HandEngine.applyEvents(state0, Seq(e2, e0, e1))
    assertEquals(state.events.map(_.sequenceInHand), Vector(0L, 1L, 2L))
    assertEquals(state.eventCount, 3)
    assertEquals(state.appliedSequences, Set(0L, 1L, 2L))
  }

  test("applyEvent: rejects event with mismatched handId") {
    val state = HandEngine.newHand("h1", 0L)
    val wrongEvent = makeEvent(handId = "h2")
    intercept[IllegalArgumentException] {
      HandEngine.applyEvent(state, wrongEvent)
    }
  }

  test("applyEvent: lastUpdatedAt is max of all seen timestamps") {
    val state0 = HandEngine.newHand("h1", 0L)
    val e0 = makeEvent(seq = 0L, ts = 1000L)
    // Intentionally send a later sequence with an earlier timestamp
    val e1 = makeEvent(seq = 1L, ts = 500L, action = PokerAction.Fold)
    val state = HandEngine.applyEvents(state0, Seq(e0, e1))
    assertEquals(state.lastUpdatedAt, 1000L)
  }

  test("applyEvent: first event timestamp defines lastUpdatedAt even if startedAt is later") {
    val state0 = HandEngine.newHand("h1", startedAt = 10_000L)
    val e0 = makeEvent(seq = 0L, ts = 1_000L)
    val state1 = HandEngine.applyEvent(state0, e0)
    assertEquals(state1.lastUpdatedAt, 1_000L)
  }

  test("applyEvents: batch with duplicates produces correct unique state") {
    val state0 = HandEngine.newHand("h1", 0L)
    val events = (0 until 5).map(i => makeEvent(seq = i.toLong, ts = 1000L + i))
    val state = HandEngine.applyEvents(state0, events ++ events)  // send twice
    assertEquals(state.eventCount, 5)
    assertEquals(state.appliedSequences, (0L until 5L).toSet)
  }

  test("currentStreet and currentBoard reflect the last applied event") {
    val flopBoard = Board.from(Seq(card("As"), card("Kd"), card("7c")))
    val state0 = HandEngine.newHand("h1", 0L)
    val flopEvent = PokerEvent(
      handId = "h1",
      sequenceInHand = 0L,
      playerId = "p1",
      occurredAtEpochMillis = 1000L,
      street = Street.Flop,
      position = Position.Button,
      board = flopBoard,
      potBefore = 20.0,
      toCall = 2.0,
      stackBefore = 200.0,
      action = PokerAction.Call
    )
    val state1 = HandEngine.applyEvent(state0, flopEvent)
    assertEquals(state1.currentStreet, Street.Flop)
    assertEquals(state1.currentBoard, flopBoard)
  }

  test("playerIds: returns set of all players who have acted") {
    val state0 = HandEngine.newHand("h1", 0L)
    val e0 = makeEvent(seq = 0L, playerId = "alice")
    val e1 = makeEvent(seq = 1L, playerId = "bob", action = PokerAction.Fold)
    val e2 = makeEvent(seq = 2L, playerId = "alice", action = PokerAction.Check, toCall = 0.0)
    val state = HandEngine.applyEvents(state0, Seq(e0, e1, e2))
    assertEquals(state.playerIds, Set("alice", "bob"))
  }

  // ---- toGameState --------------------------------------------------------------

  test("toGameState: returns None for player with no events") {
    val state = HandEngine.newHand("h1", 0L)
    assertEquals(HandEngine.toGameState(state, "ghost"), None)
  }

  test("toGameState: returns GameState from player's most recent event") {
    val state0 = HandEngine.newHand("h1", 0L)
    val e0 = makeEvent(seq = 0L, pot = 10.0, toCall = 2.0)
    val e1 = makeEvent(seq = 1L, pot = 14.0, toCall = 4.0)
    val state = HandEngine.applyEvents(state0, Seq(e0, e1))
    val gs = HandEngine.toGameState(state, "player-1").get
    assertEqualsDouble(gs.pot, 14.0, 1e-9)
    assertEqualsDouble(gs.toCall, 4.0, 1e-9)
  }

  test("toGameState: each player gets their own last event") {
    val state0 = HandEngine.newHand("h1", 0L)
    val alice0 = makeEvent(seq = 0L, playerId = "alice", pot = 10.0, toCall = 2.0)
    val bob0 = makeEvent(seq = 1L, playerId = "bob", pot = 12.0, toCall = 0.0, action = PokerAction.Check)
    val alice1 = makeEvent(seq = 2L, playerId = "alice", pot = 12.0, toCall = 5.0)
    val state = HandEngine.applyEvents(state0, Seq(alice0, bob0, alice1))
    assertEqualsDouble(HandEngine.toGameState(state, "alice").get.pot, 12.0, 1e-9)
    assertEqualsDouble(HandEngine.toGameState(state, "bob").get.pot, 12.0, 1e-9)
    assertEqualsDouble(HandEngine.toGameState(state, "alice").get.toCall, 5.0, 1e-9)
    assertEqualsDouble(HandEngine.toGameState(state, "bob").get.toCall, 0.0, 1e-9)
  }

  // ---- Snapshot roundtrip -------------------------------------------------------

  test("snapshot: roundtrip preserves all key fields including betHistory") {
    val tmpDir = Files.createTempDirectory("sicfun-snapshot-test")
    try
      val state0 = HandEngine.newHand("hand-roundtrip", 1000L)
      val events = Seq(
        makeEvent(handId = "hand-roundtrip", seq = 0L, ts = 1000L, action = PokerAction.Call),
        makeEvent(handId = "hand-roundtrip", seq = 1L, ts = 1100L, playerId = "player-2",
          action = PokerAction.Check, toCall = 0.0),
        makeEvent(handId = "hand-roundtrip", seq = 2L, ts = 1200L, action = PokerAction.Fold).copy(
          decisionTimeMillis = Some(450L),
          betHistory = Vector(BetAction(0, PokerAction.Fold), BetAction(1, PokerAction.Raise(20.0)))
        )
      )
      val state = HandEngine.applyEvents(state0, events)
      HandStateSnapshotIO.save(tmpDir, state)
      val loaded = HandStateSnapshotIO.load(tmpDir)

      assertEquals(loaded.handId, state.handId)
      assertEquals(loaded.eventCount, state.eventCount)
      assertEquals(loaded.appliedSequences, state.appliedSequences)
      assertEquals(loaded.lastUpdatedAt, state.lastUpdatedAt)

      loaded.events.zip(state.events).foreach { case (l, o) =>
        assertEquals(l.sequenceInHand, o.sequenceInHand)
        assertEquals(l.playerId, o.playerId)
        assertEquals(l.occurredAtEpochMillis, o.occurredAtEpochMillis)
        assertEquals(l.street, o.street)
        assertEquals(l.position, o.position)
        assertEquals(l.board, o.board)
        assertEqualsDouble(l.potBefore, o.potBefore, 1e-9)
        assertEqualsDouble(l.toCall, o.toCall, 1e-9)
        assertEqualsDouble(l.stackBefore, o.stackBefore, 1e-9)
        assertEquals(l.action, o.action)
        assertEquals(l.decisionTimeMillis, o.decisionTimeMillis)
        assertEquals(l.betHistory.map(ba => (ba.player, ba.action)), o.betHistory.map(ba => (ba.player, ba.action)))
      }
    finally
      Files.walk(tmpDir).sorted(java.util.Comparator.reverseOrder())
        .forEach(p => Try(Files.delete(p)))
  }

  test("snapshot: reload is idempotent with applyEvents result") {
    val tmpDir = Files.createTempDirectory("sicfun-snapshot-idempotent")
    try
      val state0 = HandEngine.newHand("idem", 0L)
      val events = (0 until 5).map(i => makeEvent(handId = "idem", seq = i.toLong, ts = 1000L + i))
      val state = HandEngine.applyEvents(state0, events)
      HandStateSnapshotIO.save(tmpDir, state)
      val loaded = HandStateSnapshotIO.load(tmpDir)
      // applying the same events to the loaded state must be idempotent
      val reapplied = HandEngine.applyEvents(loaded, events)
      assertEquals(reapplied.eventCount, state.eventCount)
    finally
      Files.walk(tmpDir).sorted(java.util.Comparator.reverseOrder())
        .forEach(p => Try(Files.delete(p)))
  }

  test("snapshot: load fails when metadata lastUpdatedAt disagrees with events") {
    val tmpDir = Files.createTempDirectory("sicfun-snapshot-invalid-lastUpdated")
    try
      val state0 = HandEngine.newHand("invalid-meta", 0L)
      val events = Seq(
        makeEvent(handId = "invalid-meta", seq = 0L, ts = 1000L),
        makeEvent(handId = "invalid-meta", seq = 1L, ts = 2000L)
      )
      val state = HandEngine.applyEvents(state0, events)
      HandStateSnapshotIO.save(tmpDir, state)

      val metaPath = tmpDir.resolve("state.properties")
      val original = Files.readAllLines(metaPath).asScala.toVector
      val patched = original.map { line =>
        if line.startsWith("lastUpdatedAt=") then "lastUpdatedAt=1234" else line
      }
      Files.write(metaPath, patched.asJava)

      intercept[IllegalArgumentException] {
        HandStateSnapshotIO.load(tmpDir)
      }
    finally
      Files.walk(tmpDir).sorted(java.util.Comparator.reverseOrder())
        .forEach(p => Try(Files.delete(p)))
  }

  // ---- Latency: p95 target -----------------------------------------------------

  test("p95 applyEvent latency is under 1ms for realistic hand size (20 events/hand)") {
    val nHands = 200
    val eventsPerHand = 20
    val latenciesMs = new Array[Double](nHands * eventsPerHand)
    var idx = 0
    for handIdx <- 0 until nHands do
      val handId = s"perf-hand-$handIdx"
      var state = HandEngine.newHand(handId, 0L)
      for seqIdx <- 0 until eventsPerHand do
        val e = makeEvent(handId = handId, seq = seqIdx.toLong, ts = 1000L + seqIdx)
        val t0 = System.nanoTime()
        state = HandEngine.applyEvent(state, e)
        latenciesMs(idx) = (System.nanoTime() - t0) / 1_000_000.0
        idx += 1
    val sorted = latenciesMs.clone()
    java.util.Arrays.sort(sorted)
    val p95 = sorted((latenciesMs.length * 0.95).toInt)
    assert(p95 < 1.0, f"p95 applyEvent latency ${p95}%.3fms exceeds 1ms target")
  }
