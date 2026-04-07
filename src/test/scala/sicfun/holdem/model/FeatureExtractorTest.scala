package sicfun.holdem.model

import munit.FunSuite
import sicfun.core.Card
import sicfun.holdem.types.*

/**
 * Tests for [[FeatureExtractor]] 8-dimensional observable feature extraction.
 *
 * Validates:
 *   - Output dimension is always 8, matching the declared constant
 *   - All features are normalized to [0, 1]
 *   - Individual feature correctness: potOdds, streetOrdinal, positionOrdinal,
 *     boardSize, decisionTime, toCallOverStack, historyLength
 *   - Edge cases: zero toCall, zero stack (all-in), clamped values (30s decision time, 20 bets)
 */
class FeatureExtractorTest extends FunSuite:
  /** Parses a card token string, failing the test if invalid. */
  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  /** Factory for creating test PokerEvent instances with configurable game state parameters. */
  private def mkEvent(
      street: Street = Street.Preflop,
      boardTokens: Seq[String] = Seq.empty,
      pot: Double = 10.0,
      toCall: Double = 2.0,
      position: Position = Position.Button,
      stack: Double = 100.0,
      action: PokerAction = PokerAction.Call,
      decisionMs: Option[Long] = Some(1000L),
      betHistory: Vector[BetAction] = Vector.empty
  ): PokerEvent =
    val board = if boardTokens.isEmpty then Board.empty else Board.from(boardTokens.map(card))
    PokerEvent(
      handId = "h1",
      sequenceInHand = 0L,
      playerId = "p1",
      occurredAtEpochMillis = 1000L,
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

  test("extract produces dimension-8 vector") {
    val f = FeatureExtractor.extract(mkEvent())
    assertEquals(f.dimension, 8)
    assertEquals(FeatureExtractor.dimension, 8)
  }

  test("featureNames has dimension entries") {
    assertEquals(FeatureExtractor.featureNames.length, FeatureExtractor.dimension)
  }

  test("all features are in [0, 1]") {
    val f = FeatureExtractor.extract(mkEvent())
    f.values.zipWithIndex.foreach { case (v, i) =>
      assert(v >= 0.0 && v <= 1.0, s"feature ${FeatureExtractor.featureNames(i)} = $v not in [0,1]")
    }
  }

  test("potOdds feature is 0 when toCall is 0") {
    val f = FeatureExtractor.extract(mkEvent(toCall = 0.0, action = PokerAction.Check))
    assertEqualsDouble(f.values(0), 0.0, 1e-12) // potOdds
  }

  test("streetOrdinal normalizes correctly") {
    val preflop = FeatureExtractor.extract(mkEvent(street = Street.Preflop))
    val river = FeatureExtractor.extract(mkEvent(street = Street.River, boardTokens = Seq("As", "Kd", "7c", "2h", "9d")))
    assertEqualsDouble(preflop.values(2), 0.0, 1e-12)   // Preflop = 0/3
    assertEqualsDouble(river.values(2), 1.0, 1e-12)     // River = 3/3
  }

  test("positionOrdinal normalizes SB to 0 and Button to 1") {
    val sb = FeatureExtractor.extract(mkEvent(position = Position.SmallBlind))
    val btn = FeatureExtractor.extract(mkEvent(position = Position.Button))
    assertEqualsDouble(sb.values(3), 0.0, 1e-12)
    assertEqualsDouble(btn.values(3), 1.0, 1e-12)
  }

  test("boardSize feature scales correctly") {
    val empty = FeatureExtractor.extract(mkEvent(boardTokens = Seq.empty))
    val flop = FeatureExtractor.extract(mkEvent(
      street = Street.Flop,
      boardTokens = Seq("As", "Kd", "7c")
    ))
    assertEqualsDouble(empty.values(4), 0.0, 1e-12)      // 0 cards
    assertEqualsDouble(flop.values(4), 3.0 / 5.0, 1e-12) // 3 cards
  }

  test("decisionTime feature clamps at 30 seconds") {
    val fast = FeatureExtractor.extract(mkEvent(decisionMs = Some(1000L)))
    val slow = FeatureExtractor.extract(mkEvent(decisionMs = Some(60_000L)))
    val none = FeatureExtractor.extract(mkEvent(decisionMs = None))
    assertEqualsDouble(fast.values(6), 1.0 / 30.0, 1e-12)
    assertEqualsDouble(slow.values(6), 1.0, 1e-12)  // clamped
    assertEqualsDouble(none.values(6), 0.0, 1e-12)  // None -> 0
  }

  test("toCallOverStack is 1.0 when stack is 0 (all-in)") {
    val f = FeatureExtractor.extract(mkEvent(stack = 0.0, toCall = 5.0))
    assertEqualsDouble(f.values(5), 1.0, 1e-12)
  }

  test("historyLength clamps at 20") {
    val history = (0 until 25).toVector.map(i => BetAction(i % 2, PokerAction.Call))
    val f = FeatureExtractor.extract(mkEvent(betHistory = history))
    assertEqualsDouble(f.values(7), 1.0, 1e-12) // 25 -> clamped to 20 -> 20/20 = 1.0
  }
