package sicfun.holdem.runtime

import munit.FunSuite
import sicfun.holdem.types.*

class MatchRunnerSupportTest extends FunSuite:

  test("MatchStatistics starts at zero"):
    val stats = new MatchRunnerSupport.MatchStatistics()
    val summary = stats.buildSummary(heroMode = HeroMode.Adaptive, modelId = "test", outDir = java.nio.file.Path.of("out"))
    assertEquals(summary.handsPlayed, 0)
    assertEquals(summary.heroNetChips, 0.0)
    assertEquals(summary.heroWins, 0)

  test("MatchStatistics tracks wins and position breakdown"):
    val stats = new MatchRunnerSupport.MatchStatistics()
    stats.recordOutcome(Position.Button, heroNetChips = 150.0)
    stats.recordOutcome(Position.BigBlind, heroNetChips = -100.0)
    stats.recordOutcome(Position.Button, heroNetChips = 0.0)
    val summary = stats.buildSummary(heroMode = HeroMode.Adaptive, modelId = "m1", outDir = java.nio.file.Path.of("out"))
    assertEquals(summary.handsPlayed, 3)
    assertEquals(summary.heroNetChips, 50.0)
    assertEquals(summary.heroWins, 1)
    assertEquals(summary.heroLosses, 1)
    assertEquals(summary.heroTies, 1)
    assertEquals(summary.buttonHands, 2)
    assertEquals(summary.buttonNetChips, 150.0)
    assertEquals(summary.bigBlindHands, 1)
    assertEquals(summary.bigBlindNetChips, -100.0)

  test("MatchStatistics computes bb/100 correctly"):
    val stats = new MatchRunnerSupport.MatchStatistics()
    for _ <- 1 to 100 do stats.recordOutcome(Position.Button, heroNetChips = 2.0)
    val summary = stats.buildSummary(heroMode = HeroMode.Adaptive, modelId = "m", outDir = java.nio.file.Path.of("out"), bigBlindChips = 100)
    assert(math.abs(summary.heroBbPer100 - 2.0) < 0.001, s"expected ~2.0, got ${summary.heroBbPer100}")

  test("writeSummary writes expected format"):
    val summary = MatchRunnerSupport.RunSummary(
      handsPlayed = 100, heroNetChips = 250.0, heroBbPer100 = 2.5,
      heroWins = 55, heroTies = 5, heroLosses = 40,
      buttonHands = 50, buttonNetChips = 200.0,
      bigBlindHands = 50, bigBlindNetChips = 50.0,
      heroMode = HeroMode.Adaptive, modelId = "test-model", outDir = java.nio.file.Path.of("out")
    )
    val tmpFile = java.nio.file.Files.createTempFile("summary", ".txt")
    try
      MatchRunnerSupport.writeSummary(tmpFile, "Test Runner", summary)
      val content = java.nio.file.Files.readString(tmpFile)
      assert(content.contains("=== Test Runner ==="))
      assert(content.contains("handsPlayed: 100"))
      assert(content.contains("heroBbPer100: 2.500"))
      assert(content.contains("heroMode: adaptive"))
      assert(content.contains("modelId: test-model"))
    finally
      java.nio.file.Files.deleteIfExists(tmpFile)

  test("appendDecisionLog writes tab-separated fields"):
    val tmpFile = java.nio.file.Files.createTempFile("decisions", ".tsv")
    val writer = java.nio.file.Files.newBufferedWriter(tmpFile)
    try
      val state = GameState(
        street = Street.Flop,
        board = Board.empty,
        pot = 3.0,
        toCall = 1.0,
        position = Position.Button,
        stackSize = 50.0,
        betHistory = Vector.empty
      )
      MatchRunnerSupport.appendDecisionLog(
        writer = writer,
        handId = 42,
        decisionIndex = 1,
        state = state,
        candidates = Vector(PokerAction.Fold, PokerAction.Call),
        chosenAction = PokerAction.Call,
        wireAction = "c"
      )
      writer.flush()
      val content = java.nio.file.Files.readString(tmpFile)
      assert(content.contains("42\t1\tFlop\tButton\t3.000\t1.000\t50.000\tFold,Call\tCall\tc"))
    finally
      writer.close()
      java.nio.file.Files.deleteIfExists(tmpFile)
