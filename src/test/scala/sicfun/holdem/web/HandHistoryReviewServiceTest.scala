package sicfun.holdem.web

import sicfun.holdem.cli.CliHelpers
import sicfun.holdem.model.{PokerActionModel, PokerActionModelArtifactIO}
import sicfun.holdem.types.{Board, GameState, PokerAction, Position, Street}

import sicfun.core.Card

import munit.FunSuite

import java.nio.file.Files

class HandHistoryReviewServiceTest extends FunSuite:

  private val reviewableHand =
    """PokerStars Hand #1001:  Hold'em No Limit ($0.50/$1.00 USD) - 2026/03/10 12:00:00 ET
      |Table 'Alpha' 2-max Seat #1 is the button
      |Seat 1: Hero ($100.00 in chips)
      |Seat 2: Villain ($100.00 in chips)
      |Hero: posts small blind $0.50
      |Villain: posts big blind $1.00
      |*** HOLE CARDS ***
      |Dealt to Hero [Ac Kh]
      |Hero: raises $1.50 to $2.00
      |Villain: calls $1.00
      |*** FLOP *** [Ts 9h 8d]
      |Villain: checks
      |Hero: bets $3.00
      |Villain: raises $6.00 to $9.00
      |Hero: folds
      |*** SUMMARY ***
      |""".stripMargin

  private val hiddenHeroHand =
    """PokerStars Hand #1002:  Hold'em No Limit ($0.50/$1.00 USD) - 2026/03/10 12:01:00 ET
      |Table 'Bravo' 2-max Seat #1 is the button
      |Seat 1: Hero ($100.00 in chips)
      |Seat 2: Villain ($100.00 in chips)
      |Hero: posts small blind $0.50
      |Villain: posts big blind $1.00
      |*** HOLE CARDS ***
      |Hero: raises $1.50 to $2.00
      |Villain: folds
      |*** SUMMARY ***
      |""".stripMargin

  private val forumHeroAliasHand =
    """PokerStars Hand #166193791537: Hold'em No Limit ($0.02/$0.05 USD) - 2017/02/14 19:34:00 CUST [2017/02/14 7:34:00 ET]
      |Table 'Hekatostos' 6-max Seat #6 is the button
      |Seat 1: TUNGLIMING(HERO) ($6.21 in chips)
      |Seat 3: blvm ($5 in chips)
      |Seat 4: ISniffBluffs ($7.05 in chips)
      |Seat 5: JoeDavola27 ($5.05 in chips)
      |Seat 6: vitales08 ($5 in chips)
      |TUNGLIMING: posts small blind $0.02
      |RETR0-RUS98: is sitting out
      |blvm: posts big blind $0.05
      |*** HOLE CARDS ***
      |Dealt to TUNGLIMING [7d 7c]
      |ISniffBluffs: folds
      |JoeDavola27: folds
      |vitales08: folds
      |TUNGLIMING(HERO): raises $0.10 to $0.15
      |blvm: calls $0.10
      |*** FLOP *** [8d 3s 7s]
      |TUNGLIMING: checks
      |blvm: checks
      |*** TURN *** [8d 3s 7s] [6s]
      |TUNGLIMING: bets $0.15
      |blvm: raises $0.25 to $0.40
      |TUNGLIMING: calls $0.25
      |*** RIVER *** [8d 3s 7s 6s] [Td]
      |TUNGLIMING: checks
      |blvm: bets $0.55
      |TUNGLIMING: calls $0.55
      |*** SHOW DOWN ***
      |""".stripMargin

  private def card(token: String): Card =
    Card.parse(token).getOrElse(throw new IllegalArgumentException(s"bad card token: $token"))

  private def seedModelArtifact(modelDir: java.nio.file.Path): Unit =
    val flop = Board.from(Seq(card("Ts"), card("9h"), card("8d")))
    val facingBet = GameState(
      street = Street.Flop,
      board = flop,
      pot = 20.0,
      toCall = 10.0,
      position = Position.BigBlind,
      stackSize = 180.0,
      betHistory = Vector.empty
    )
    val checkedTo = facingBet.copy(toCall = 0.0)
    val training = Vector.fill(18)((facingBet, CliHelpers.parseHoleCards("AhAd"), PokerAction.Raise(25.0))) ++
      Vector.fill(18)((facingBet, CliHelpers.parseHoleCards("QcJc"), PokerAction.Call)) ++
      Vector.fill(18)((facingBet, CliHelpers.parseHoleCards("7c2d"), PokerAction.Fold)) ++
      Vector.fill(12)((checkedTo, CliHelpers.parseHoleCards("AsKs"), PokerAction.Check))
    val artifact = PokerActionModel.trainVersioned(
      trainingData = training,
      learningRate = 0.1,
      iterations = 180,
      l2Lambda = 0.001,
      validationFraction = 0.25,
      splitSeed = 11L,
      maxMeanBrierScore = 2.0,
      failOnGate = false,
      modelId = "web-review-test-model",
      source = "hand-history-review-service-test",
      trainedAtEpochMillis = 1_800_000_000_000L
    )
    PokerActionModelArtifactIO.save(modelDir, artifact)

  private def withService[A](run: HandHistoryReviewService => A): A =
    val tempDir = Files.createTempDirectory("hand-history-review-service-")
    val modelDir = tempDir.resolve("model")
    try
      seedModelArtifact(modelDir)
      val service = HandHistoryReviewService.create(
        HandHistoryReviewService.ServiceConfig(
          modelDir = Some(modelDir),
          seed = 7L,
          bunchingTrials = 50,
          equityTrials = 200,
          budgetMs = 1000L,
          maxDecisions = 8
        )
      ).fold(err => fail(err), identity)
      run(service)
    finally
      Files.walk(tempDir).sorted(java.util.Comparator.reverseOrder()).forEach(Files.deleteIfExists)

  test("analyzes an uploaded hand history and returns decision + opponent summaries".tag(munit.Slow)) {
    withService { service =>
      val responseEither = service.analyze(
        HandHistoryReviewService.AnalysisRequest(
          handHistoryText = reviewableHand,
          site = None,
          heroName = Some("Hero")
        )
      )

      assert(responseEither.isRight, s"analysis failed: $responseEither")
      val response = responseEither.toOption.get
      assertEquals(response.site, "PokerStars")
      assertEquals(response.heroName, Some("Hero"))
      assertEquals(response.handsImported, 1)
      assertEquals(response.handsAnalyzed, 1)
      assertEquals(response.handsSkipped, 0)
      assert(response.decisionsAnalyzed > 0, s"expected decisions, got ${response.decisionsAnalyzed}")
      assert(response.opponents.exists(_.playerName == "Villain"))
      assertEquals(response.warnings, Vector.empty)
    }
  }

  test("returns an honest skip warning when hero hole cards are missing") {
    withService { service =>
      val responseEither = service.analyze(
        HandHistoryReviewService.AnalysisRequest(
          handHistoryText = hiddenHeroHand,
          site = None,
          heroName = Some("Hero")
        )
      )

      assert(responseEither.isRight, s"analysis failed: $responseEither")
      val response = responseEither.toOption.get
      assertEquals(response.handsImported, 1)
      assertEquals(response.handsAnalyzed, 0)
      assertEquals(response.handsSkipped, 1)
      assertEquals(response.decisionsAnalyzed, 0)
      assert(response.warnings.exists(_.contains("hero hole cards")))
    }
  }

  test("normalizes forum hero aliases through the review service".tag(munit.Slow)) {
    withService { service =>
      val responseEither = service.analyze(
        HandHistoryReviewService.AnalysisRequest(
          handHistoryText = forumHeroAliasHand,
          site = Some(sicfun.holdem.history.HandHistorySite.PokerStars),
          heroName = Some("TUNGLIMING(HERO)")
        )
      )

      assert(responseEither.isRight, s"analysis failed: $responseEither")
      val response = responseEither.toOption.get
      assertEquals(response.site, "PokerStars")
      assertEquals(response.heroName, Some("TUNGLIMING"))
      assertEquals(response.handsImported, 1)
      assertEquals(response.handsAnalyzed, 1)
      assert(response.decisionsAnalyzed > 0, s"expected decisions, got ${response.decisionsAnalyzed}")
      assertEquals(response.warnings, Vector.empty)
    }
  }
