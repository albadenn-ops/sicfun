package sicfun.holdem.io
import sicfun.holdem.types.*

import munit.FunSuite
import sicfun.core.Card

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters.*

/**
 * Tests for [[HandStateSnapshotIO]] directory-based hand state persistence.
 *
 * Validates:
 *   - Save/load roundtrip preserves single and multiple events, all poker actions,
 *     board cards, bet history, decision times, and empty event lists
 *   - All Position and Street enum values survive the roundtrip
 *   - Directory creation for nested paths
 *   - String path overload convenience methods
 *   - Load-time validation: non-existent directory, event count mismatch,
 *     duplicate sequences, and header corruption
 */
class HandStateSnapshotIOTest extends FunSuite:
  /** Parses a card token string, failing the test if invalid. */
  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  /** Factory method for creating test events. Board is auto-generated based on street. */
  private def event(
      handId: String,
      sequence: Long,
      playerId: String = "player-1",
      ts: Long,
      street: Street = Street.Preflop,
      position: Position = Position.Button,
      action: PokerAction = PokerAction.Call,
      pot: Double = 10.0,
      toCall: Double = 2.0,
      stack: Double = 100.0,
      decisionMs: Option[Long] = Some(300L),
      betHistory: Vector[BetAction] = Vector.empty
  ): PokerEvent =
    val board = street match
      case Street.Preflop => Board.empty
      case Street.Flop    => Board.from(Seq(card("As"), card("Kd"), card("7c")))
      case Street.Turn    => Board.from(Seq(card("As"), card("Kd"), card("7c"), card("2h")))
      case Street.River   => Board.from(Seq(card("As"), card("Kd"), card("7c"), card("2h"), card("9d")))
    PokerEvent(
      handId = handId,
      sequenceInHand = sequence,
      playerId = playerId,
      occurredAtEpochMillis = ts,
      street = street,
      position = position,
      board = board,
      potBefore = pot,
      toCall = toCall,
      stackBefore = stack,
      action = action,
      decisionTimeMillis = decisionMs,
      betHistory = betHistory
    )

  /** Creates a temp directory, runs the test body, then recursively deletes the directory. */
  private def withTempDir(testName: String)(f: Path => Unit): Unit =
    val dir = Files.createTempDirectory(s"sicfun-snapshot-$testName-")
    try f(dir)
    finally
      Files.walk(dir).sorted(java.util.Comparator.reverseOrder()).forEach(Files.deleteIfExists(_))

  test("save/load roundtrip preserves a single-event hand") {
    withTempDir("single") { dir =>
      val e = event(handId = "h1", sequence = 0L, ts = 5000L)
      val state = HandState("h1", Vector(e), Set(0L), 5000L)

      HandStateSnapshotIO.save(dir, state)
      val loaded = HandStateSnapshotIO.load(dir)

      assertEquals(loaded.handId, "h1")
      assertEquals(loaded.events.length, 1)
      assertEquals(loaded.events.head, e)
      assertEquals(loaded.appliedSequences, Set(0L))
      assertEquals(loaded.lastUpdatedAt, 5000L)
    }
  }

  test("save/load roundtrip preserves multiple events in order") {
    withTempDir("multi") { dir =>
      val e0 = event(handId = "h2", sequence = 0L, ts = 1000L, action = PokerAction.Call, toCall = 2.0)
      val e1 = event(handId = "h2", sequence = 1L, ts = 1200L, action = PokerAction.Raise(8.0), toCall = 2.0)
      val e2 = event(handId = "h2", sequence = 2L, ts = 1400L, action = PokerAction.Fold, toCall = 8.0)
      val events = Vector(e0, e1, e2)
      val state = HandState("h2", events, Set(0L, 1L, 2L), 1400L)

      HandStateSnapshotIO.save(dir, state)
      val loaded = HandStateSnapshotIO.load(dir)

      assertEquals(loaded.events, events)
      assertEquals(loaded.lastUpdatedAt, 1400L)
      assertEquals(loaded.eventCount, 3)
    }
  }

  test("roundtrip preserves all poker actions including Raise amounts") {
    withTempDir("actions") { dir =>
      val e0 = event(handId = "h3", sequence = 0L, ts = 100L, action = PokerAction.Check, toCall = 0.0)
      val e1 = event(handId = "h3", sequence = 1L, ts = 200L, action = PokerAction.Raise(15.5), toCall = 2.0)
      val e2 = event(handId = "h3", sequence = 2L, ts = 300L, action = PokerAction.Call, toCall = 15.5)
      val e3 = event(handId = "h3", sequence = 3L, ts = 400L, action = PokerAction.Fold, toCall = 15.5)
      val events = Vector(e0, e1, e2, e3)
      val state = HandState("h3", events, Set(0L, 1L, 2L, 3L), 400L)

      HandStateSnapshotIO.save(dir, state)
      val loaded = HandStateSnapshotIO.load(dir)

      assertEquals(loaded.events.map(_.action), Vector(
        PokerAction.Check, PokerAction.Raise(15.5), PokerAction.Call, PokerAction.Fold
      ))
    }
  }

  test("roundtrip preserves board cards on flop/turn/river events") {
    withTempDir("boards") { dir =>
      val eFlop = event(handId = "h4", sequence = 0L, ts = 100L, street = Street.Flop, toCall = 0.0, action = PokerAction.Check)
      val eTurn = event(handId = "h4", sequence = 1L, ts = 200L, street = Street.Turn, toCall = 0.0, action = PokerAction.Check)
      val eRiver = event(handId = "h4", sequence = 2L, ts = 300L, street = Street.River, toCall = 0.0, action = PokerAction.Check)
      val events = Vector(eFlop, eTurn, eRiver)
      val state = HandState("h4", events, Set(0L, 1L, 2L), 300L)

      HandStateSnapshotIO.save(dir, state)
      val loaded = HandStateSnapshotIO.load(dir)

      assertEquals(loaded.events(0).board.size, 3)
      assertEquals(loaded.events(1).board.size, 4)
      assertEquals(loaded.events(2).board.size, 5)
      assertEquals(loaded.events(0).board, eFlop.board)
      assertEquals(loaded.events(1).board, eTurn.board)
      assertEquals(loaded.events(2).board, eRiver.board)
    }
  }

  test("roundtrip preserves betHistory with multiple bet actions") {
    withTempDir("bethistory") { dir =>
      val history = Vector(
        BetAction(0, PokerAction.Call),
        BetAction(1, PokerAction.Raise(12.0)),
        BetAction(0, PokerAction.Call)
      )
      val e = event(handId = "h5", sequence = 0L, ts = 1000L, betHistory = history)
      val state = HandState("h5", Vector(e), Set(0L), 1000L)

      HandStateSnapshotIO.save(dir, state)
      val loaded = HandStateSnapshotIO.load(dir)

      assertEquals(loaded.events.head.betHistory, history)
    }
  }

  test("roundtrip preserves decisionTimeMillis = None") {
    withTempDir("no-decision-time") { dir =>
      val e = event(handId = "h6", sequence = 0L, ts = 500L, decisionMs = None)
      val state = HandState("h6", Vector(e), Set(0L), 500L)

      HandStateSnapshotIO.save(dir, state)
      val loaded = HandStateSnapshotIO.load(dir)

      assertEquals(loaded.events.head.decisionTimeMillis, None)
    }
  }

  test("roundtrip preserves empty event list") {
    withTempDir("empty") { dir =>
      val state = HandState("h-empty", Vector.empty, Set.empty, 9999L)

      HandStateSnapshotIO.save(dir, state)
      val loaded = HandStateSnapshotIO.load(dir)

      assertEquals(loaded.handId, "h-empty")
      assertEquals(loaded.events, Vector.empty)
      assertEquals(loaded.lastUpdatedAt, 9999L)
    }
  }

  test("save creates directory if absent") {
    withTempDir("create-dir") { parentDir =>
      val subDir = parentDir.resolve("nested").resolve("deep")
      assert(!Files.exists(subDir))

      val e = event(handId = "h7", sequence = 0L, ts = 100L)
      val state = HandState("h7", Vector(e), Set(0L), 100L)

      HandStateSnapshotIO.save(subDir, state)
      assert(Files.isDirectory(subDir))

      val loaded = HandStateSnapshotIO.load(subDir)
      assertEquals(loaded.handId, "h7")
    }
  }

  test("save with string path overload works") {
    withTempDir("string-path") { dir =>
      val e = event(handId = "h8", sequence = 0L, ts = 200L)
      val state = HandState("h8", Vector(e), Set(0L), 200L)

      HandStateSnapshotIO.save(dir.toString, state)
      val loaded = HandStateSnapshotIO.load(dir.toString)
      assertEquals(loaded.handId, "h8")
    }
  }

  test("load rejects non-existent directory") {
    val bogus = Path.of("non-existent-snapshot-dir-xyz")
    intercept[IllegalArgumentException] {
      HandStateSnapshotIO.load(bogus)
    }
  }

  test("load rejects event count mismatch") {
    withTempDir("count-mismatch") { dir =>
      val e = event(handId = "h-bad", sequence = 0L, ts = 100L)
      val state = HandState("h-bad", Vector(e), Set(0L), 100L)
      HandStateSnapshotIO.save(dir, state)

      // Tamper with metadata to claim 5 events
      val metaPath = dir.resolve("state.properties")
      val content = Files.readString(metaPath, StandardCharsets.UTF_8)
      val tampered = content.replace("eventCount=1", "eventCount=5")
      Files.writeString(metaPath, tampered, StandardCharsets.UTF_8)

      intercept[IllegalArgumentException] {
        HandStateSnapshotIO.load(dir)
      }
    }
  }

  test("load rejects duplicate sequenceInHand values") {
    withTempDir("dup-seq") { dir =>
      val e0 = event(handId = "h-dup", sequence = 0L, ts = 100L, action = PokerAction.Call, toCall = 2.0)
      val e1 = event(handId = "h-dup", sequence = 1L, ts = 200L, action = PokerAction.Fold, toCall = 2.0)
      val state = HandState("h-dup", Vector(e0, e1), Set(0L, 1L), 200L)
      HandStateSnapshotIO.save(dir, state)

      // Tamper with the events file to duplicate sequence 0
      val eventsPath = dir.resolve("events.tsv")
      val lines = Files.readAllLines(eventsPath, StandardCharsets.UTF_8).asScala.toVector
      // Replace the second data row's sequence (1) with 0
      val tampered = lines.updated(2, lines(2).replaceFirst("^1\t", "0\t"))
      Files.write(eventsPath, tampered.asJava, StandardCharsets.UTF_8)

      // Also fix the metadata timestamp to match
      val metaPath = dir.resolve("state.properties")
      val metaContent = Files.readString(metaPath, StandardCharsets.UTF_8)
      Files.writeString(metaPath, metaContent, StandardCharsets.UTF_8)

      intercept[IllegalArgumentException] {
        HandStateSnapshotIO.load(dir)
      }
    }
  }

  test("load rejects wrong TSV header") {
    withTempDir("bad-header") { dir =>
      val e = event(handId = "h-hdr", sequence = 0L, ts = 100L)
      val state = HandState("h-hdr", Vector(e), Set(0L), 100L)
      HandStateSnapshotIO.save(dir, state)

      val eventsPath = dir.resolve("events.tsv")
      val lines = Files.readAllLines(eventsPath, StandardCharsets.UTF_8).asScala.toVector
      val badHeader = "wrong\theader\tcolumns"
      Files.write(eventsPath, (badHeader +: lines.drop(1)).asJava, StandardCharsets.UTF_8)

      intercept[IllegalArgumentException] {
        HandStateSnapshotIO.load(dir)
      }
    }
  }

  test("roundtrip preserves all Position values") {
    withTempDir("positions") { dir =>
      val positions = Position.values.toVector
      val events = positions.zipWithIndex.map { case (pos, i) =>
        event(handId = "h-pos", sequence = i.toLong, ts = 1000L + i.toLong, position = pos)
      }
      val state = HandState("h-pos", events, events.map(_.sequenceInHand).toSet, events.map(_.occurredAtEpochMillis).max)

      HandStateSnapshotIO.save(dir, state)
      val loaded = HandStateSnapshotIO.load(dir)

      assertEquals(loaded.events.map(_.position), positions)
    }
  }

  test("roundtrip preserves all Street values") {
    withTempDir("streets") { dir =>
      val streets = Street.values.toVector
      val events = streets.zipWithIndex.map { case (st, i) =>
        event(handId = "h-st", sequence = i.toLong, ts = 1000L + i.toLong, street = st, toCall = 0.0, action = PokerAction.Check)
      }
      val state = HandState("h-st", events, events.map(_.sequenceInHand).toSet, events.map(_.occurredAtEpochMillis).max)

      HandStateSnapshotIO.save(dir, state)
      val loaded = HandStateSnapshotIO.load(dir)

      assertEquals(loaded.events.map(_.street), streets)
    }
  }
