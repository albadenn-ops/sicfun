package sicfun.holdem.io
import sicfun.holdem.types.*

import munit.FunSuite

import java.nio.charset.StandardCharsets
import java.nio.file.Files
import scala.jdk.CollectionConverters.*

/**
 * Tests for [[DecisionLoopEventFeedIO]] append-only TSV event feed.
 *
 * Validates:
 *   - Append creates the file with a header and subsequent appends do not duplicate the header
 *   - Incremental read correctly tracks byte offsets and returns only new events
 *   - Header validation on incremental reads
 *   - Recovery from file truncation/rotation (file shrinks below the last-read offset)
 *   - Tolerance of empty files during rotation (returns empty, resets offset to 0)
 */
class DecisionLoopEventFeedIOTest extends FunSuite:
  /** Creates a test event for a two-player preflop scenario, alternating hero/villain by sequence parity. */
  private def event(sequence: Long, action: PokerAction): PokerEvent =
    PokerEvent(
      handId = "feed-hand-1",
      sequenceInHand = sequence,
      playerId = if sequence % 2 == 0 then "hero" else "villain",
      occurredAtEpochMillis = 1_800_000_000_000L + sequence,
      street = Street.Preflop,
      position = if sequence % 2 == 0 then Position.Button else Position.BigBlind,
      board = Board.empty,
      potBefore = 6.0,
      toCall = 2.0,
      stackBefore = 98.0,
      action = action,
      decisionTimeMillis = Some(100L + sequence),
      betHistory = Vector.empty
    )

  test("append writes header once and readIncremental consumes appended rows") {
    val path = Files.createTempFile("decision-loop-feed-", ".tsv")
    Files.delete(path)
    try
      DecisionLoopEventFeedIO.append(path, event(0L, PokerAction.Call))
      DecisionLoopEventFeedIO.append(path, event(1L, PokerAction.Raise(8.0)))

      val lines = Files.readAllLines(path, StandardCharsets.UTF_8).asScala.toVector
      assertEquals(lines.head, DecisionLoopEventFeedIO.Header)
      assertEquals(lines.count(_ == DecisionLoopEventFeedIO.Header), 1)

      val (events, offset) = DecisionLoopEventFeedIO.readIncremental(path, 0L)
      assertEquals(events.length, 2)
      assert(offset > 0L)

      val (none, offset2) = DecisionLoopEventFeedIO.readIncremental(path, offset)
      assertEquals(none, Vector.empty)
      assertEquals(offset2, offset)
    finally
      Files.deleteIfExists(path)
  }

  test("readIncremental validates header before reading rows") {
    val path = Files.createTempFile("decision-loop-feed-bad-", ".tsv")
    try
      Files.write(
        path,
        Vector(
          "bad\theader",
          "feed-hand-1\t0\thero\t1800000000000\tPreflop\tButton\t-\t6.0\t2.0\t98.0\tCall\t100\t-"
        ).asJava,
        StandardCharsets.UTF_8
      )

      intercept[IllegalArgumentException] {
        DecisionLoopEventFeedIO.readIncremental(path, 0L)
      }
    finally
      Files.deleteIfExists(path)
  }

  test("readIncremental recovers when the feed is recreated from a smaller file") {
    val path = Files.createTempFile("decision-loop-feed-rotated-", ".tsv")
    Files.delete(path)
    try
      DecisionLoopEventFeedIO.append(path, event(0L, PokerAction.Call))
      DecisionLoopEventFeedIO.append(path, event(1L, PokerAction.Raise(8.0)))

      val (_, offset) = DecisionLoopEventFeedIO.readIncremental(path, 0L)
      assert(offset > 0L)

      Files.delete(path)
      DecisionLoopEventFeedIO.append(path, event(9L, PokerAction.Fold))

      val (events, nextOffset) = DecisionLoopEventFeedIO.readIncremental(path, offset)
      assertEquals(events.map(_.event.sequenceInHand), Vector(9L))
      assert(nextOffset > 0L)
    finally
      Files.deleteIfExists(path)
  }

  test("readIncremental tolerates an empty recreated feed while waiting for a new header") {
    val path = Files.createTempFile("decision-loop-feed-empty-", ".tsv")
    Files.delete(path)
    try
      DecisionLoopEventFeedIO.append(path, event(0L, PokerAction.Call))
      DecisionLoopEventFeedIO.append(path, event(1L, PokerAction.Raise(8.0)))

      val (_, offset) = DecisionLoopEventFeedIO.readIncremental(path, 0L)
      assert(offset > 0L)

      Files.write(path, Array.emptyByteArray)

      val (events, nextOffset) = DecisionLoopEventFeedIO.readIncremental(path, offset)
      assertEquals(events, Vector.empty)
      assertEquals(nextOffset, 0L)
    finally
      Files.deleteIfExists(path)
  }
