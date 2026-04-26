package sicfun.holdem.runtime.protocol

import munit.FunSuite
import sicfun.core.Card
import sicfun.holdem.types.{Board, Street}

/** Pinning tests for [[AcpcActionCodec]] private[protocol] helpers.
  *
  * `AcpcMatchRunnerTest` already covers the wire-protocol parsing surface (parseBetting,
  * MATCHSTATE round-trips, hand-value at showdown). This file pins the small reusable
  * pieces that previously had no direct coverage:
  *
  *   - boardForStreet (slice full 5-card board down to street-appropriate prefix)
  *   - streetIndexForBoard (board size -> street index 0..3)
  *   - streetFromIndex (street index 0..3 -> Street enum)
  *   - relativeActorId (0 if hero, 1 otherwise)
  *   - showdownValue (multiway side-pot resolver, much more complex than the
  *     heads-up variant pinned in AcpcHeadsUpDealerTest)
  */
class AcpcMatchRunnerInternalsTest extends FunSuite:

  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  private val fullBoard = Board.from(Vector(
    card("2c"), card("7h"), card("Jd"), card("4s"), card("Ks")
  ))

  // ---- boardForStreet ----

  test("boardForStreet returns empty Board for Preflop") {
    assertEquals(ProtocolStreetMath.boardForStreet(fullBoard, Street.Preflop).cards, Vector.empty)
  }

  test("boardForStreet slices Flop to first 3 cards") {
    val flop = ProtocolStreetMath.boardForStreet(fullBoard, Street.Flop)
    assertEquals(flop.cards, fullBoard.cards.take(3))
  }

  test("boardForStreet slices Turn to first 4 cards") {
    val turn = ProtocolStreetMath.boardForStreet(fullBoard, Street.Turn)
    assertEquals(turn.cards, fullBoard.cards.take(4))
  }

  test("boardForStreet returns full 5-card board for River") {
    val river = ProtocolStreetMath.boardForStreet(fullBoard, Street.River)
    assertEquals(river.cards, fullBoard.cards)
  }

  test("boardForStreet rejects board with fewer cards than the street requires") {
    val flopOnly = Board.from(fullBoard.cards.take(3))
    intercept[IllegalArgumentException] {
      ProtocolStreetMath.boardForStreet(flopOnly, Street.Turn)
    }
  }

  // ---- streetIndexForBoard ----

  test("streetIndexForBoard maps board size to street index 0..3") {
    assertEquals(ProtocolStreetMath.streetIndexForBoard(Board.empty), 0)
    assertEquals(ProtocolStreetMath.streetIndexForBoard(Board.from(fullBoard.cards.take(3))), 1)
    assertEquals(ProtocolStreetMath.streetIndexForBoard(Board.from(fullBoard.cards.take(4))), 2)
    assertEquals(ProtocolStreetMath.streetIndexForBoard(Board.from(fullBoard.cards.take(5))), 3)
  }

  test("streetIndexForBoard rejects unsupported board sizes (1, 2)") {
    val twoCards = Board.from(fullBoard.cards.take(2))
    intercept[IllegalArgumentException] {
      ProtocolStreetMath.streetIndexForBoard(twoCards)
    }
  }

  // ---- streetFromIndex ----

  test("streetFromIndex round-trips with streetIndexForBoard") {
    val sizes = Vector(0, 3, 4, 5)
    sizes.zipWithIndex.foreach { case (size, expectedIdx) =>
      val board = Board.from(fullBoard.cards.take(size))
      val idx = ProtocolStreetMath.streetIndexForBoard(board)
      assertEquals(idx, expectedIdx, s"size=$size")
      val street = ProtocolStreetMath.streetFromIndex(idx)
      assertEquals(
        ProtocolStreetMath.boardForStreet(fullBoard, street).cards,
        board.cards,
        s"round-trip failed at size=$size"
      )
    }
  }

  test("streetFromIndex rejects out-of-range indices") {
    intercept[IllegalArgumentException] { ProtocolStreetMath.streetFromIndex(-1) }
    intercept[IllegalArgumentException] { ProtocolStreetMath.streetFromIndex(4) }
  }

  // ---- relativeActorId ----

  test("relativeActorId returns 0 when actor is hero, 1 otherwise") {
    assertEquals(ProtocolStreetMath.relativeActorId(actualActor = 0, heroActual = 0), 0)
    assertEquals(ProtocolStreetMath.relativeActorId(actualActor = 1, heroActual = 0), 1)
    assertEquals(ProtocolStreetMath.relativeActorId(actualActor = 0, heroActual = 1), 1)
    assertEquals(ProtocolStreetMath.relativeActorId(actualActor = 1, heroActual = 1), 0)
  }

  // ---- showdownValue (multiway side-pot resolver) ----

  test("showdownValue: heads-up winner gains opponent's contribution") {
    val spent = Vector(2000, 2000)
    val rank = Vector(7000, 5000)
    assertEquals(AcpcActionCodec.showdownValue(spent, rank, playerIdx = 0), 2000.0)
    assertEquals(AcpcActionCodec.showdownValue(spent, rank, playerIdx = 1), -2000.0)
  }

  test("showdownValue: heads-up tie splits the pot evenly (zero net)") {
    val spent = Vector(1000, 1000)
    val rank = Vector(6000, 6000)
    assertEquals(AcpcActionCodec.showdownValue(spent, rank, playerIdx = 0), 0.0)
    assertEquals(AcpcActionCodec.showdownValue(spent, rank, playerIdx = 1), 0.0)
  }

  test("showdownValue: 3-way main pot, single winner takes all losers' contribution") {
    // Three players each commit 1000. Player 0 has the best hand.
    // Player 0 wins 2000 (1000 from each loser); each loser nets -1000.
    val spent = Vector(1000, 1000, 1000)
    val rank = Vector(8000, 5000, 4000)
    assertEquals(AcpcActionCodec.showdownValue(spent, rank, playerIdx = 0), 2000.0)
    assertEquals(AcpcActionCodec.showdownValue(spent, rank, playerIdx = 1), -1000.0)
    assertEquals(AcpcActionCodec.showdownValue(spent, rank, playerIdx = 2), -1000.0)
  }

  test("showdownValue: side pot with unequal contributions") {
    // Player 0 (short stack) commits 500, Players 1+2 commit 1000 each.
    // Layer 1: 500 from each of the 3 players = 1500 main pot.
    // Layer 2: 500 each from players 1 and 2 = 1000 side pot (only 1 + 2 eligible).
    //
    // If Player 0 has the best hand: wins layer 1 only = 1000 profit (1500 pot - 500 own).
    // If Player 1 has the best hand: wins layer 1 (gains 1000 from losers) + layer 2 (gains 500 from player 2) = 1500 net.
    val spent = Vector(500, 1000, 1000)

    // Case A: short stack wins
    val rankA = Vector(9000, 5000, 4000)
    assertEquals(AcpcActionCodec.showdownValue(spent, rankA, playerIdx = 0), 1000.0,
      "short stack wins main pot only -> +500 from each of two losers")
    // Player 1 in case A: loses main pot (-500), wins side pot vs player 2 (+500) -> net 0
    assertEquals(AcpcActionCodec.showdownValue(spent, rankA, playerIdx = 1), 0.0,
      "middle rank loses main, wins side -> net 0")
    // Player 2 loses both layers: -500 main + -500 side = -1000
    assertEquals(AcpcActionCodec.showdownValue(spent, rankA, playerIdx = 2), -1000.0,
      "lowest rank forfeits both layers")

    // Case B: deep stack wins both layers
    val rankB = Vector(4000, 9000, 5000)
    assertEquals(AcpcActionCodec.showdownValue(spent, rankB, playerIdx = 1), 1500.0,
      "deep-stack winner takes both layers")
    assertEquals(AcpcActionCodec.showdownValue(spent, rankB, playerIdx = 0), -500.0,
      "short stack loses its full contribution (no side pot eligibility)")
    assertEquals(AcpcActionCodec.showdownValue(spent, rankB, playerIdx = 2), -1000.0,
      "deep loser forfeits everything")
  }

  test("showdownValue: net across all participants sums to zero (chip conservation)") {
    val cases = Vector(
      (Vector(1000, 1000), Vector(7000, 5000)),
      (Vector(500, 1000, 1000), Vector(9000, 5000, 4000)),
      (Vector(500, 1000, 1500), Vector(4000, 9000, 5000)),
      (Vector(2000, 2000, 2000), Vector(7000, 7000, 4000)) // tied winners
    )
    cases.foreach { case (spent, rank) =>
      val total = spent.indices.map { idx =>
        AcpcActionCodec.showdownValue(spent, rank, idx)
      }.sum
      assertEqualsDouble(total, 0.0, 1e-9, s"chip conservation for spent=$spent rank=$rank")
    }
  }

  test("showdownValue: rejects mismatched spent/rank vectors and out-of-range player indices") {
    intercept[IllegalArgumentException] {
      AcpcActionCodec.showdownValue(Vector(100, 200), Vector(500), playerIdx = 0)
    }
    intercept[IllegalArgumentException] {
      AcpcActionCodec.showdownValue(Vector(100, 200), Vector(500, 600), playerIdx = -1)
    }
    intercept[IllegalArgumentException] {
      AcpcActionCodec.showdownValue(Vector(100, 200), Vector(500, 600), playerIdx = 2)
    }
  }
