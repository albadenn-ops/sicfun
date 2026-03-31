package sicfun.holdem.validation

import munit.FunSuite
import sicfun.holdem.cfr.HoldemCfrConfig
import sicfun.holdem.history.{HandHistoryImport, HandHistorySite}

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*

class AdaptiveProofHarnessTest extends FunSuite:
  override val munitTimeout: Duration = 180.seconds

  test("reduced adaptive proof harness run writes parseable mirrored exports".tag(munit.Slow)) {
    val root = Files.createTempDirectory("adaptive-proof-harness-test-")
    try
      val testOpponents = Vector(
        AdaptiveProofHarness.OpponentSpec(
          name = "Villain01_overcall",
          role = "Leaker",
          leak = Overcalls(0.35),
          strategy = EquityBasedStrategy(),
          strategyLabel = "equity-based",
          baselineNoise = 0.03
        ),
        AdaptiveProofHarness.OpponentSpec(
          name = "Villain06_gto",
          role = "Control",
          leak = NoLeak(),
          strategy = CfrVillainStrategy(
            config = HoldemCfrConfig(
              iterations = 60,
              equityTrials = 120,
              maxVillainHands = 32,
              includeVillainReraises = true
            ),
            allowHeuristicFallback = false
          ),
          strategyLabel = "cfr-no-fallback",
          baselineNoise = 0.0
        )
      )
      val result = AdaptiveProofHarness.run(
        AdaptiveProofHarness.Config(
          handsPerOpponent = 6,
          outputDir = root.resolve("adaptive-proof"),
          runLabel = Some("test-run"),
          seed = 19L,
          bunchingTrials = 8,
          equityTrials = 60,
          minEquityTrials = 20,
          budgetMs = 20L,
          opponents = testOpponents
        )
      )

      assert(result.isRight, s"adaptive proof harness failed: $result")
      val summary = result.toOption.getOrElse(fail("missing adaptive proof harness summary"))

      assertEquals(summary.opponents.size, testOpponents.size)
      assert(Files.exists(summary.runDir), s"missing run directory: ${summary.runDir}")
      assert(Files.exists(summary.manifestPath), s"missing manifest: ${summary.manifestPath}")
      assert(Files.exists(summary.groundTruthPath), s"missing ground truth: ${summary.groundTruthPath}")
      assert(Files.exists(summary.reportPath), s"missing report: ${summary.reportPath}")
      assert(Files.exists(summary.combinedHistoryPath), s"missing combined history: ${summary.combinedHistoryPath}")

      val groundTruth = ujson.read(Files.readString(summary.groundTruthPath, StandardCharsets.UTF_8))
      assertEquals(groundTruth("opponents").arr.size, testOpponents.size)

      val control = summary.opponents.find(_.name == "Villain06_gto").getOrElse(fail("missing control result"))
      assertEquals(control.leakFiredCount, 0)

      val historyFiles =
        summary.opponents.flatMap(opponent =>
          Vector(opponent.legButtonPath, opponent.legBigBlindPath, opponent.combinedPath)
        ) :+ summary.combinedHistoryPath

      historyFiles.foreach { path =>
        assert(Files.exists(path), s"missing history file: $path")
        val text = Files.readString(path, StandardCharsets.UTF_8)
        assert(text.nonEmpty, s"history file should be non-empty: $path")
        HandHistoryImport.parseText(text, Some(HandHistorySite.PokerStars), Some("Hero")) match
          case Right(hands) =>
            assert(hands.nonEmpty, s"expected parseable hands in $path")
          case Left(err) =>
            fail(s"parseText failed for $path: $err")
      }
    finally
      deleteRecursively(root)
  }

  private def deleteRecursively(path: Path): Unit =
    if Files.exists(path) then
      val stream = Files.walk(path)
      try
        val all = stream.iterator().asScala.toVector.sortBy(_.toString.length).reverse
        all.foreach(Files.deleteIfExists)
      finally
        stream.close()
