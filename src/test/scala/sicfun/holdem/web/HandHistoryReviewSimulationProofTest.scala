package sicfun.holdem.web

import sicfun.holdem.history.HandHistorySite
import sicfun.holdem.runtime.TexasHoldemPlayingHall

import munit.FunSuite

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*

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
