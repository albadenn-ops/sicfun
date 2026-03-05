package sicfun.holdem

import munit.FunSuite
import sicfun.core.Card

import java.nio.file.Files
import scala.util.Random

class HandHistoryAnalyzerTest extends FunSuite:

  private def card(t: String): Card =
    Card.parse(t).getOrElse(throw new IllegalArgumentException(s"bad card: $t"))

  test("analyzeWithHeroCards produces decisions for a single hand".tag(munit.Slow)) {
    // Build a minimal set of events for one hand
    val heroCards = CliHelpers.parseHoleCards("AcKh")
    val board = Board.from(Seq(card("Ts"), card("9h"), card("8d")))

    val events = Vector(
      PokerEvent(
        handId = "h1", sequenceInHand = 0L, playerId = "villain",
        occurredAtEpochMillis = 1000L, street = Street.Flop,
        position = Position.BigBlind, board = board,
        potBefore = 12.0, toCall = 0.0, stackBefore = 194.0,
        action = PokerAction.Raise(8.0), betHistory = Vector.empty
      ),
      PokerEvent(
        handId = "h1", sequenceInHand = 1L, playerId = "hero",
        occurredAtEpochMillis = 1500L, street = Street.Flop,
        position = Position.SmallBlind, board = board,
        potBefore = 20.0, toCall = 8.0, stackBefore = 194.0,
        action = PokerAction.Call,
        betHistory = Vector(BetAction(1, PokerAction.Raise(8.0)))
      )
    )

    // Bootstrap engine
    val baseState = GameState(
      street = Street.Flop,
      board = Board.from(Seq(card("Ts"), card("9h"), card("8d"))),
      pot = 20.0, toCall = 10.0, position = Position.BigBlind,
      stackSize = 180.0, betHistory = Vector.empty
    )
    val checkState = baseState.copy(toCall = 0.0)
    val strong = HoleCards.from(Vector(card("Ah"), card("Ad")))
    val medium = HoleCards.from(Vector(card("Qc"), card("Jc")))
    val weak = HoleCards.from(Vector(card("7c"), card("2d")))
    val data: Seq[(GameState, HoleCards, PokerAction)] =
      Vector.fill(24)((baseState, strong, PokerAction.Raise(25.0))) ++
        Vector.fill(24)((baseState, medium, PokerAction.Call)) ++
        Vector.fill(24)((baseState, weak, PokerAction.Fold)) ++
        Vector.fill(12)((checkState, medium, PokerAction.Check))
    val artifact = PokerActionModel.trainVersioned(
      trainingData = data, learningRate = 0.1, iterations = 200,
      l2Lambda = 0.001, validationFraction = 0.25, splitSeed = 42L,
      maxMeanBrierScore = 2.0, failOnGate = false,
      modelId = "test", schemaVersion = "v1", source = "test",
      trainedAtEpochMillis = 1_000_000_000_000L
    )
    val tableRanges = TableRanges.defaults(TableFormat.NineMax)
    val engine = new RealTimeAdaptiveEngine(
      tableRanges = tableRanges, actionModel = artifact.model,
      bunchingTrials = 50, defaultEquityTrials = 200, minEquityTrials = 100
    )

    val decisions = HandHistoryAnalyzer.analyzeWithHeroCards(
      events = events,
      heroPlayerId = "hero",
      heroCards = heroCards,
      engine = engine,
      tableRanges = tableRanges,
      budgetMs = 5000L,
      rng = new Random(42)
    )

    assertEquals(decisions.length, 1, s"expected 1 hero decision, got ${decisions.length}")
    val d = decisions.head
    assertEquals(d.handId, "h1")
    assertEquals(d.street, Street.Flop)
    assertEquals(d.actualAction, PokerAction.Call)
    assert(d.heroEquityMean >= 0.0 && d.heroEquityMean <= 1.0,
      s"equity out of range: ${d.heroEquityMean}")
    assert(d.heroCards.contains(heroCards))
  }

  test("run returns Left for missing feed file") {
    val result = HandHistoryAnalyzer.run(Array("/nonexistent/feed.tsv"))
    assert(result.isLeft)
  }

  test("run with a temp feed file produces summary".tag(munit.Slow)) {
    val tempDir = Files.createTempDirectory("analyzer-test-")
    val feedPath = tempDir.resolve("feed.tsv")

    try
      // Write header + 2 events
      val board = Board.from(Seq(card("Ts"), card("9h"), card("8d")))
      val villainEvent = PokerEvent(
        handId = "h1", sequenceInHand = 0L, playerId = "villain",
        occurredAtEpochMillis = 1000L, street = Street.Flop,
        position = Position.BigBlind, board = board,
        potBefore = 12.0, toCall = 0.0, stackBefore = 194.0,
        action = PokerAction.Raise(8.0), betHistory = Vector.empty
      )
      val heroEvent = PokerEvent(
        handId = "h1", sequenceInHand = 1L, playerId = "hero",
        occurredAtEpochMillis = 1500L, street = Street.Flop,
        position = Position.SmallBlind, board = board,
        potBefore = 20.0, toCall = 8.0, stackBefore = 194.0,
        action = PokerAction.Call,
        betHistory = Vector(BetAction(1, PokerAction.Raise(8.0)))
      )
      DecisionLoopEventFeedIO.append(feedPath, villainEvent)
      DecisionLoopEventFeedIO.append(feedPath, heroEvent)

      val result = HandHistoryAnalyzer.run(Array(
        feedPath.toString,
        "--hero=hero",
        "--seed=42",
        "--bunchingTrials=50",
        "--equityTrials=200"
      ))

      assert(result.isRight, s"analysis failed: $result")
      val summary = result.toOption.get
      assertEquals(summary.handsAnalyzed, 1)
      assertEquals(summary.decisionsAnalyzed, 1)
    finally
      Files.walk(tempDir).sorted(java.util.Comparator.reverseOrder()).forEach(Files.deleteIfExists)
  }
