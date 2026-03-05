package sicfun.holdem

import munit.FunSuite
import sicfun.core.Card

class ActionDatasetContractTest extends FunSuite:
  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  private def boardFor(street: Street): Board =
    street match
      case Street.Preflop => Board.empty
      case Street.Flop =>
        Board.from(Seq(card("As"), card("Kd"), card("7c")))
      case Street.Turn =>
        Board.from(Seq(card("As"), card("Kd"), card("7c"), card("2h")))
      case Street.River =>
        Board.from(Seq(card("As"), card("Kd"), card("7c"), card("2h"), card("9d")))

  private def event(
      handId: String = "hand-1",
      sequence: Long = 0L,
      playerId: String = "player-1",
      ts: Long = 1_000L,
      street: Street = Street.Preflop,
      position: Position = Position.Button,
      action: PokerAction = PokerAction.Call,
      pot: Double = 10.0,
      toCall: Double = 2.0,
      stack: Double = 100.0,
      decisionMs: Option[Long] = Some(300L)
  ): PokerEvent =
    PokerEvent(
      handId = handId,
      sequenceInHand = sequence,
      playerId = playerId,
      occurredAtEpochMillis = ts,
      street = street,
      position = position,
      board = boardFor(street),
      potBefore = pot,
      toCall = toCall,
      stackBefore = stack,
      action = action,
      decisionTimeMillis = decisionMs,
      betHistory = Vector.empty
    )

  test("PokerEvent rejects empty required ids") {
    intercept[IllegalArgumentException] {
      event(handId = "   ")
    }
    intercept[IllegalArgumentException] {
      event(playerId = " ")
    }
  }

  test("PokerEvent validates street-board consistency and action semantics") {
    intercept[IllegalArgumentException] {
      PokerEvent(
        handId = "hand-1",
        sequenceInHand = 0L,
        playerId = "p1",
        occurredAtEpochMillis = 1000L,
        street = Street.Flop,
        position = Position.Button,
        board = Board.empty,
        potBefore = 10.0,
        toCall = 2.0,
        stackBefore = 100.0,
        action = PokerAction.Call
      )
    }

    intercept[IllegalArgumentException] {
      event(action = PokerAction.Check, toCall = 1.0)
    }
    intercept[IllegalArgumentException] {
      event(action = PokerAction.Call, toCall = 0.0)
    }
    intercept[IllegalArgumentException] {
      event(action = PokerAction.Raise(0.0))
    }
  }

  test("DatasetBuilder rejects duplicate sequence in same hand") {
    val events = Seq(
      event(handId = "h1", sequence = 0L, ts = 1000L),
      event(handId = "h1", sequence = 0L, ts = 1200L, action = PokerAction.Raise(6.0))
    )

    intercept[IllegalArgumentException] {
      DatasetBuilder.build(events, source = "unit-test", generatedAtEpochMillis = 42L)
    }
  }

  test("DatasetBuilder rejects timestamp regression in same hand") {
    val events = Seq(
      event(handId = "h1", sequence = 0L, ts = 1200L),
      event(handId = "h1", sequence = 1L, ts = 1100L, action = PokerAction.Raise(7.0))
    )

    intercept[IllegalArgumentException] {
      DatasetBuilder.build(events, source = "unit-test", generatedAtEpochMillis = 42L)
    }
  }

  test("DatasetBuilder output is reproducible regardless input order") {
    val a0 = event(handId = "h2", sequence = 0L, ts = 1000L, action = PokerAction.Call)
    val a1 = event(handId = "h2", sequence = 1L, ts = 1300L, action = PokerAction.Raise(8.0))
    val b0 = event(handId = "h1", sequence = 0L, ts = 900L, action = PokerAction.Check, toCall = 0.0)
    val input = Seq(a0, a1, b0)

    val ds1 = DatasetBuilder.build(input, source = "unit-test", generatedAtEpochMillis = 123L)
    val ds2 = DatasetBuilder.build(input.reverse, source = "unit-test", generatedAtEpochMillis = 123L)

    assertEquals(ds1, ds2)
  }

  test("DatasetBuilder fills provenance and feature dimensions") {
    val events = Seq(
      event(handId = "h1", sequence = 0L, ts = 1000L, action = PokerAction.Call),
      event(handId = "h1", sequence = 1L, ts = 1100L, action = PokerAction.Raise(8.0))
    )

    val dataset = DatasetBuilder.build(events, source = "unit-test", generatedAtEpochMillis = 777L)
    assertEquals(dataset.provenance.schemaVersion, PokerEvent.SchemaVersion)
    assertEquals(dataset.provenance.source, "unit-test")
    assertEquals(dataset.provenance.eventCount, 2)
    assertEquals(dataset.provenance.uniqueHandCount, 1)
    assertEquals(dataset.provenance.featureNames.length, FeatureExtractor.dimension)
    dataset.examples.foreach { example =>
      assertEquals(example.features.length, FeatureExtractor.dimension)
    }
  }

  test("statistics reports correct class distribution and feature ranges") {
    val events = Seq(
      event(handId = "h1", sequence = 0L, ts = 1000L, action = PokerAction.Call),
      event(handId = "h1", sequence = 1L, ts = 1100L, action = PokerAction.Raise(8.0)),
      event(handId = "h2", sequence = 0L, ts = 2000L, action = PokerAction.Fold),
      event(handId = "h2", sequence = 1L, ts = 2100L, action = PokerAction.Check, toCall = 0.0)
    )

    val dataset = DatasetBuilder.build(events, source = "unit-test", generatedAtEpochMillis = 42L)
    val stats = dataset.statistics

    assertEquals(stats.totalExamples, 4)
    assertEquals(stats.classDistribution.values.sum, 4)
    assertEquals(stats.classDistribution(PokerAction.Category.Call), 1)
    assertEquals(stats.classDistribution(PokerAction.Category.Raise), 1)
    assertEquals(stats.classDistribution(PokerAction.Category.Fold), 1)
    assertEquals(stats.classDistribution(PokerAction.Category.Check), 1)

    assertEquals(stats.featureStatistics.length, FeatureExtractor.dimension)
    stats.featureStatistics.foreach { fs =>
      assert(fs.min <= fs.max, s"feature ${fs.name}: min (${fs.min}) > max (${fs.max})")
      assert(fs.min >= 0.0, s"feature ${fs.name}: min (${fs.min}) < 0 (all features should be normalized)")
      assert(fs.max <= 1.0, s"feature ${fs.name}: max (${fs.max}) > 1 (all features should be normalized)")
    }
  }
