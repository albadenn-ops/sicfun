package sicfun.holdem.web

import sicfun.holdem.history.HandHistorySite
import sicfun.holdem.runtime.TexasHoldemPlayingHall

import munit.FunSuite

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*

/** End-to-end simulation proof test for the hand history review pipeline.
  *
  * Runs a reproducible 6-max [[TexasHoldemPlayingHall]] session with an
  * exact-GTO hero against a mixed villain pool (TAG, LAG, maniac), exports
  * the resulting hands in PokerStars review-upload format, then feeds them
  * back through [[HandHistoryReviewService]] to verify the full roundtrip:
  *
  *   1. Hall simulation produces a valid `review-upload-pokerstars.txt`
  *   2. The review service successfully parses all 12 exported hands
  *   3. All hands are analyzed (none skipped) with decisions extracted
  *   4. Multiple simulated opponents are profiled
  *   5. No warnings are generated for well-formed simulated data
  *   6. The diagnostic trace matches the top-level response fields
  *
  * This test uses a 180-second timeout due to the combined cost of the
  * hall simulation and the review analysis pass.
  */
class HandHistoryReviewSimulationProofTest extends FunSuite:
  override val munitTimeout: Duration = 180.seconds

  test("review service analyzes exported exact-gto hall corpus from a reproducible villain pool".tag(munit.Slow)) {
    val root = Files.createTempDirectory("hand-history-review-proof-")
    try
      val out = root.resolve("hall-proof-out")
      val hallResult = TexasHoldemPlayingHall.run(Array(
        "--hands=12",
        "--reportEvery=12",
        "--learnEveryHands=0",
        "--learningWindowSamples=50",
        "--seed=37",
        s"--outDir=$out",
        "--playerCount=6",
        "--heroPosition=Cutoff",
        "--heroStyle=gto",
        "--gtoMode=exact",
        "--villainPool=tag,lag,maniac",
        "--heroExplorationRate=0.0",
        "--raiseSize=2.5",
        "--bunchingTrials=8",
        "--equityTrials=80",
        "--saveTrainingTsv=false",
        "--saveDdreTrainingTsv=false",
        "--saveReviewHandHistory=true"
      ))

      assert(hallResult.isRight, s"hall proof run failed: $hallResult")

      val uploadPath = out.resolve("review-upload-pokerstars.txt")
      assert(Files.exists(uploadPath), s"missing review upload export: $uploadPath")

      val uploadText = Files.readString(uploadPath, StandardCharsets.UTF_8)
      val service = HandHistoryReviewService.create(
        HandHistoryReviewService.ServiceConfig(
          seed = 37L,
          bunchingTrials = 8,
          equityTrials = 240,
          budgetMs = 150L,
          maxDecisions = 16
        )
      ).fold(err => fail(err), identity)

      val response = service.analyze(
        HandHistoryReviewService.AnalysisRequest(
          handHistoryText = uploadText,
          site = Some(HandHistorySite.PokerStars),
          heroName = Some("Hero")
        )
      ).fold(err => fail(err), identity)

      assertEquals(response.site, "PokerStars")
      assertEquals(response.heroName, Some("Hero"))
      assertEquals(response.handsImported, 12)
      assert(response.handsAnalyzed > 0, s"expected analyzed hands, got ${response.handsAnalyzed}")
      assert(response.decisionsAnalyzed > 0, s"expected analyzed decisions, got ${response.decisionsAnalyzed}")
      assert(response.opponents.map(_.playerName).distinct.size >= 2, s"expected multiple simulated opponents, got ${response.opponents}")
      assertEquals(response.warnings, Vector.empty)
      assertEquals(response.trace.request.rawHeroName, Some("Hero"))
      assertEquals(response.trace.request.normalizedHeroName, Some("Hero"))
      assertEquals(response.trace.request.requestedSite, Some("PokerStars"))
      assert(response.trace.request.handHistoryBytes > 0)
      assertEquals(response.trace.importStage.handsImported, 12)
      assertEquals(response.trace.importStage.siteResolved, Some("PokerStars"))
      assertEquals(response.trace.importStage.heroNameResolved, Some("Hero"))
      assertEquals(response.trace.hands.length, 12)
      assert(response.trace.hands.forall(_.status == "analyzed"))
      assert(response.trace.hands.forall(_.heroCardsPresent))
      assertEquals(response.trace.summary.handsImported, response.handsImported)
      assertEquals(response.trace.summary.handsAnalyzed, response.handsAnalyzed)
      assertEquals(response.trace.summary.handsSkipped, response.handsSkipped)
      assertEquals(response.trace.summary.decisionsAnalyzed, response.decisionsAnalyzed)
      assertEquals(response.trace.summary.totalEvLost, response.totalEvLost)
      assertEquals(response.trace.summary.biggestMistakeEv, response.biggestMistakeEv)
      assertEquals(response.trace.summary.warningCount, response.warnings.length)
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
