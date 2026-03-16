package sicfun.holdem.runtime

import munit.FunSuite
import sicfun.core.{Card, DiscreteDistribution}
import sicfun.holdem.cli.CliHelpers
import sicfun.holdem.model.PokerActionModel
import sicfun.holdem.types.*
import sicfun.core.MultinomialLogistic

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters.*
import scala.concurrent.duration.*
import scala.util.Random

class TexasHoldemPlayingHallTest extends FunSuite:
  override val munitTimeout: Duration = 180.seconds

  test("contribution gap preserves the full big blind call amount after a small blind raise") {
    assertEquals(TexasHoldemPlayingHall.contributionGap(targetContribution = 3.5, currentContribution = 1.0), 2.5)
    assertEquals(TexasHoldemPlayingHall.contributionGap(targetContribution = 3.5, currentContribution = 3.5), 0.0)
    assertEquals(TexasHoldemPlayingHall.contributionGap(targetContribution = 1.0, currentContribution = 3.5), 0.0)
  }

  test("normalizeAction caps unaffordable raises into all-in calls or smaller all-in raises") {
    assertEquals(
      TexasHoldemPlayingHall.normalizeAction(
        action = sicfun.holdem.types.PokerAction.Raise(2.5),
        toCall = 1.0,
        stackSize = 1.0,
        allowRaise = true,
        defaultRaise = 2.5
      ),
      sicfun.holdem.types.PokerAction.Call
    )
    assertEquals(
      TexasHoldemPlayingHall.normalizeAction(
        action = sicfun.holdem.types.PokerAction.Raise(2.5),
        toCall = 1.0,
        stackSize = 2.0,
        allowRaise = true,
        defaultRaise = 2.5
      ),
      sicfun.holdem.types.PokerAction.Raise(1.0)
    )
  }

  test("raiseReopensAction requires a full raise increment") {
    assert(!TexasHoldemPlayingHall.raiseReopensAction(raiseAmount = 1.0, minimumRaiseAmount = 2.5))
    assert(TexasHoldemPlayingHall.raiseReopensAction(raiseAmount = 2.5, minimumRaiseAmount = 2.5))
    assert(TexasHoldemPlayingHall.raiseReopensAction(raiseAmount = 3.0, minimumRaiseAmount = 2.5))
  }

  test("sidePotPayouts awards each contribution slice only to eligible winners") {
    val payouts = TexasHoldemPlayingHall.sidePotPayouts(
      contributions = Map(
        sicfun.holdem.types.Position.Button -> 100.0,
        sicfun.holdem.types.Position.BigBlind -> 60.0,
        sicfun.holdem.types.Position.Cutoff -> 20.0
      ),
      remainingPlayers = Vector(
        sicfun.holdem.types.Position.Button,
        sicfun.holdem.types.Position.BigBlind,
        sicfun.holdem.types.Position.Cutoff
      ),
      handStrengthByPosition = Map(
        sicfun.holdem.types.Position.Button -> 1,
        sicfun.holdem.types.Position.BigBlind -> 2,
        sicfun.holdem.types.Position.Cutoff -> 3
      )
    )

    assertEquals(payouts.getOrElse(sicfun.holdem.types.Position.Cutoff, 0.0), 60.0)
    assertEquals(payouts.getOrElse(sicfun.holdem.types.Position.BigBlind, 0.0), 80.0)
    assertEquals(payouts.getOrElse(sicfun.holdem.types.Position.Button, 0.0), 40.0)
  }

  test("mergedInferenceFolds keeps inactive seats and live preflop folds together") {
    val folds = TexasHoldemPlayingHall.mergedInferenceFolds(
      staticFoldedPositions = Vector(Position.UTG, Position.Middle),
      preflopFoldedPositions = Vector(Position.Cutoff, Position.Middle),
      perspectivePosition = Position.Button,
      targetPosition = Position.BigBlind
    )

    assertEquals(folds.map(_.position), Vector(Position.UTG, Position.Middle, Position.Cutoff))
  }

  test("estimateRaiseResponseFromRange renormalizes legal fold and continue mass when facing a raise") {
    val response = TexasHoldemPlayingHall.estimateRaiseResponseFromRange(
      range = DiscreteDistribution(Map(
        hole("AhKd") -> 0.4,
        hole("7c2d") -> 0.6
      )),
      responseState = GameState(
        street = Street.Flop,
        board = board("Qs", "Jh", "4c"),
        pot = 5.0,
        toCall = 2.5,
        position = Position.BigBlind,
        stackSize = 80.0,
        betHistory = Vector.empty
      ),
      actionModel = PokerActionModel.uniform,
      raiseAction = PokerAction.Raise(2.5)
    )

    assertEqualsDouble(response.foldProbability, 1.0 / 3.0, 1e-9)
    assertEqualsDouble(response.continueProbability, 2.0 / 3.0, 1e-9)
    assertEqualsDouble(response.foldProbability + response.continueProbability, 1.0, 1e-9)
    assertEqualsDouble(response.continuationRange.probabilityOf(hole("AhKd")), 0.4, 1e-9)
    assertEqualsDouble(response.continuationRange.probabilityOf(hole("7c2d")), 0.6, 1e-9)
  }

  test("estimateRaiseResponseFromRange shifts continuation range toward stronger hands under a non-uniform model") {
    val foldWeights = Vector(0.0, 0.0, 0.0, 0.0, -6.0)
    val checkWeights = Vector.fill(5)(0.0)
    val callWeights = Vector(0.0, 0.0, 0.0, 0.0, 5.5)
    val raiseWeights = Vector(0.0, 0.0, 0.0, 0.0, 3.5)
    val model = PokerActionModel(
      logistic = MultinomialLogistic(
        weights = Vector(foldWeights, checkWeights, callWeights, raiseWeights),
        bias = Vector(1.0, -8.0, -1.0, -2.0)
      ),
      categoryIndex = PokerActionModel.defaultCategoryIndex,
      featureDimension = 5
    )
    val strong = hole("QhQs")
    val weak = hole("3c4d")
    val response = TexasHoldemPlayingHall.estimateRaiseResponseFromRange(
      range = DiscreteDistribution(Map(
        strong -> 0.5,
        weak -> 0.5
      )),
      responseState = GameState(
        street = Street.River,
        board = board("Ah", "Ad", "Kc", "2s", "7h"),
        pot = 8.0,
        toCall = 3.0,
        position = Position.BigBlind,
        stackSize = 70.0,
        betHistory = Vector.empty
      ),
      actionModel = model,
      raiseAction = PokerAction.Raise(2.5)
    )

    assert(response.continueProbability > response.foldProbability)
    assert(response.continuationRange.probabilityOf(strong) > 0.5)
    assert(response.continuationRange.probabilityOf(weak) < 0.5)
  }

  test("estimateEquityAgainstOpponentRanges uses exact multiway share when river ranges are tiny") {
    val hero = hole("3c4d")
    val riverBoard = board("Ah", "Ad", "Kc", "Kd", "2s")
    val estimate = TexasHoldemPlayingHall.estimateEquityAgainstOpponentRanges(
      hero = hero,
      board = riverBoard,
      opponentRanges = Vector(
        singleRange("QhQs"),
        singleRange("JhJs")
      ),
      equityTrials = 32,
      rng = new Random(1L)
    )

    assertEquals(estimate.mean, 0.0)
    assertEquals(estimate.winRate, 0.0)
    assertEquals(estimate.tieRate, 0.0)
    assertEquals(estimate.lossRate, 1.0)
    assertEquals(estimate.trials, 1)
  }

  test("recommendActionAgainstOpponentRanges folds a dead bluff-catcher against two stronger ranges") {
    val riverBoard = board("Ah", "Ad", "Kc", "Kd", "2s")
    val state = GameState(
      street = Street.River,
      board = riverBoard,
      pot = 6.0,
      toCall = 1.0,
      position = Position.Button,
      stackSize = 50.0,
      betHistory = Vector.empty
    )

    val recommendation = TexasHoldemPlayingHall.recommendActionAgainstOpponentRanges(
      hero = hole("3c4d"),
      state = state,
      opponentRanges = Vector(
        singleRange("QhQs"),
        singleRange("JhJs")
      ),
      candidateActions = Vector(PokerAction.Fold, PokerAction.Call),
      equityTrials = 32,
      rng = new Random(2L)
    )

    assertEquals(recommendation.bestAction, PokerAction.Fold)
    assert(recommendation.actionEvaluations.exists(eval => eval.action == PokerAction.Call && eval.expectedValue < 0.0))
  }

  test("playing hall runs end-to-end and emits logs") {
    val root = Files.createTempDirectory("playing-hall-test-")
    try
      val out = root.resolve("hall-out")
      val result = TexasHoldemPlayingHall.run(Array(
        "--hands=120",
        "--tableCount=3",
        "--reportEvery=60",
        "--learnEveryHands=40",
        "--learningWindowSamples=200",
        "--seed=11",
        s"--outDir=$out",
        "--villainStyle=tag",
        "--raiseSize=2.5",
        "--bunchingTrials=20",
        "--equityTrials=120",
        "--saveTrainingTsv=true",
        "--saveDdreTrainingTsv=true"
      ))
      assert(result.isRight, s"hall run failed: $result")
      val summary = result.toOption.getOrElse(fail("missing hall summary"))
      assertEquals(summary.handsPlayed, 120)
      assertEquals(summary.tableCount, 3)
      assertEquals(summary.playerCount, 2)
      assert(summary.retrains >= 1, s"expected at least one retrain, got ${summary.retrains}")
      assert(Files.exists(out.resolve("hands.tsv")))
      assert(Files.exists(out.resolve("learning.tsv")))
      assert(Files.exists(out.resolve("training-selfplay.tsv")))
      assert(Files.exists(out.resolve("ddre-training-selfplay.tsv")))

      val handRows = Files.readAllLines(out.resolve("hands.tsv"), StandardCharsets.UTF_8).asScala.toVector
      assert(handRows.length > 1, "expected hand log rows")
      val handHeader = handRows.head.split("\t", -1).toVector
      val tableIdIdx = handHeader.indexOf("tableId")
      assert(tableIdIdx >= 0, "hands.tsv missing tableId column")
      val playerCountIdx = handHeader.indexOf("playerCount")
      assert(playerCountIdx >= 0, "hands.tsv missing playerCount column")
      val heroPositionIdx = handHeader.indexOf("heroPosition")
      assert(heroPositionIdx >= 0, "hands.tsv missing heroPosition column")
      val villainPositionIdx = handHeader.indexOf("villainPosition")
      assert(villainPositionIdx >= 0, "hands.tsv missing villainPosition column")
      val openerPositionIdx = handHeader.indexOf("openerPosition")
      assert(openerPositionIdx >= 0, "hands.tsv missing openerPosition column")
      val foldedPositionsIdx = handHeader.indexOf("foldedPositions")
      assert(foldedPositionsIdx >= 0, "hands.tsv missing foldedPositions column")
      val activePositionsIdx = handHeader.indexOf("activePositions")
      assert(activePositionsIdx >= 0, "hands.tsv missing activePositions column")
      val maxLivePlayersIdx = handHeader.indexOf("maxLivePlayers")
      assert(maxLivePlayersIdx >= 0, "hands.tsv missing maxLivePlayers column")
      val heroActionIdx = handHeader.indexOf("heroAction")
      assert(heroActionIdx >= 0, "hands.tsv missing heroAction column")
      val streetsPlayedIdx = handHeader.indexOf("streetsPlayed")
      assert(streetsPlayedIdx >= 0, "hands.tsv missing streetsPlayed column")
      val tableIds = handRows.drop(1).flatMap { row =>
        val fields = row.split("\t", -1).toVector
        fields.lift(tableIdIdx)
      }.toSet
      assertEquals(tableIds, Set("1", "2", "3"))
      val playerCounts = handRows.drop(1).flatMap { row =>
        val fields = row.split("\t", -1).toVector
        fields.lift(playerCountIdx)
      }.toSet
      assertEquals(playerCounts, Set("2"))
      val heroPositions = handRows.drop(1).flatMap { row =>
        val fields = row.split("\t", -1).toVector
        fields.lift(heroPositionIdx)
      }.toSet
      assertEquals(heroPositions, Set("Button"))
      val villainPositions = handRows.drop(1).flatMap { row =>
        val fields = row.split("\t", -1).toVector
        fields.lift(villainPositionIdx)
      }.toSet
      assertEquals(villainPositions, Set("BigBlind"))
      val openerPositions = handRows.drop(1).flatMap { row =>
        val fields = row.split("\t", -1).toVector
        fields.lift(openerPositionIdx)
      }.toSet
      assertEquals(openerPositions, Set("Button"))
      val foldedPositionValues = handRows.drop(1).flatMap { row =>
        val fields = row.split("\t", -1).toVector
        fields.lift(foldedPositionsIdx)
      }.toSet
      assertEquals(foldedPositionValues, Set("-"))
      val activePositionValues = handRows.drop(1).flatMap { row =>
        val fields = row.split("\t", -1).toVector
        fields.lift(activePositionsIdx)
      }.toSet
      assertEquals(activePositionValues, Set("Button|BigBlind"))
      val maxLivePlayers = handRows.drop(1).flatMap { row =>
        val fields = row.split("\t", -1).toVector
        fields.lift(maxLivePlayersIdx).flatMap(_.toIntOption)
      }.toSet
      assertEquals(maxLivePlayers, Set(2))
      val sawIllegalOpeningCheck = handRows.drop(1).exists { row =>
        val fields = row.split("\t", -1).toVector
        fields.lift(heroActionIdx).contains("Check")
      }
      assert(!sawIllegalOpeningCheck, "hero should not open the hand with Check while facing the big blind")
      val sawPostflop = handRows.drop(1).exists { row =>
        val fields = row.split("\t", -1).toVector
        fields.lift(streetsPlayedIdx).flatMap(_.toIntOption).exists(_ > 1)
      }
      assert(sawPostflop, "expected at least one hand to reach postflop")
      val learningRows = Files.readAllLines(out.resolve("learning.tsv"), StandardCharsets.UTF_8).asScala.toVector
      assert(learningRows.length > 1, "expected learning log rows")
      val trainingRows = Files.readAllLines(out.resolve("training-selfplay.tsv"), StandardCharsets.UTF_8).asScala.toVector
      assert(trainingRows.length > 1, "expected training rows")
      val sawPostflopTraining = trainingRows.drop(1).exists { row =>
        row.startsWith("Flop\t") || row.startsWith("Turn\t") || row.startsWith("River\t")
      }
      assert(sawPostflopTraining, "expected training data from postflop streets")

      val ddreRows = Files.readAllLines(out.resolve("ddre-training-selfplay.tsv"), StandardCharsets.UTF_8).asScala.toVector
      assert(ddreRows.length > 1, "expected DDRE training rows")
      val ddreHeader = ddreRows.head.split("\t", -1).toVector
      assert(ddreHeader.contains("bayesPosteriorSparse"), "DDRE TSV missing bayesPosteriorSparse column")
      val streetIdx = ddreHeader.indexOf("street")
      val sawPostflopDdre = ddreRows.drop(1).exists { row =>
        val fields = row.split("\t", -1).toVector
        fields.lift(streetIdx).exists(value => value == "Flop" || value == "Turn" || value == "River")
      }
      assert(sawPostflopDdre, "expected DDRE training data from postflop streets")
    finally
      deleteRecursively(root)
  }

  test("playing hall rejects invalid hands argument") {
    val result = TexasHoldemPlayingHall.run(Array("--hands=0"))
    assert(result.isLeft, "expected invalid hands value to fail")
  }

  test("playing hall rejects invalid tableCount argument") {
    val result = TexasHoldemPlayingHall.run(Array("--tableCount=0"))
    assert(result.isLeft, "expected invalid tableCount value to fail")
  }

  test("playing hall rejects hero positions that are not modeled for the table size") {
    val result = TexasHoldemPlayingHall.run(Array(
      "--playerCount=4",
      "--heroPosition=UTG"
    ))
    assert(result.isLeft, "expected invalid hero position for 4-handed table")
  }

  test("playing hall supports six-max cutoff contexts") {
    val root = Files.createTempDirectory("playing-hall-sixmax-test-")
    try
      val out = root.resolve("hall-sixmax-out")
      val result = TexasHoldemPlayingHall.run(Array(
        "--hands=12",
        "--reportEvery=6",
        "--learnEveryHands=0",
        "--learningWindowSamples=50",
        "--seed=41",
        s"--outDir=$out",
        "--playerCount=6",
        "--heroPosition=Cutoff",
        "--villainPool=tag,lag",
        "--heroExplorationRate=0.0",
        "--raiseSize=2.5",
        "--bunchingTrials=8",
        "--equityTrials=80",
        "--saveTrainingTsv=false",
        "--saveDdreTrainingTsv=false"
      ))
      assert(result.isRight, s"six-max hall run failed: $result")
      val summary = result.toOption.getOrElse(fail("missing hall summary"))
      assertEquals(summary.playerCount, 6)

      val handRows = Files.readAllLines(out.resolve("hands.tsv"), StandardCharsets.UTF_8).asScala.toVector
      assert(handRows.length > 1, "expected six-max hand log rows")
      val handHeader = handRows.head.split("\t", -1).toVector
      val playerCountIdx = handHeader.indexOf("playerCount")
      val heroPositionIdx = handHeader.indexOf("heroPosition")
      val villainPositionIdx = handHeader.indexOf("villainPosition")
      val openerPositionIdx = handHeader.indexOf("openerPosition")
      val foldedPositionsIdx = handHeader.indexOf("foldedPositions")
      val activePositionsIdx = handHeader.indexOf("activePositions")
      val maxLivePlayersIdx = handHeader.indexOf("maxLivePlayers")
      assert(playerCountIdx >= 0 && heroPositionIdx >= 0 && villainPositionIdx >= 0 && openerPositionIdx >= 0 && foldedPositionsIdx >= 0 && activePositionsIdx >= 0 && maxLivePlayersIdx >= 0)

      val playerCounts = handRows.drop(1).flatMap(_.split("\t", -1).toVector.lift(playerCountIdx)).toSet
      assertEquals(playerCounts, Set("6"))
      val heroPositions = handRows.drop(1).flatMap(_.split("\t", -1).toVector.lift(heroPositionIdx)).toSet
      assertEquals(heroPositions, Set("Cutoff"))
      val villainPositions = handRows.drop(1).flatMap(_.split("\t", -1).toVector.lift(villainPositionIdx)).toSet
      assert(villainPositions.subsetOf(Set("UTG", "Middle", "Button", "SmallBlind", "BigBlind")))
      val openerPositions = handRows.drop(1).flatMap(_.split("\t", -1).toVector.lift(openerPositionIdx)).toSet
      assert(openerPositions.subsetOf(Set("UTG", "Middle", "Cutoff", "Button", "SmallBlind", "BigBlind")))
      val foldedValues = handRows.drop(1).flatMap(_.split("\t", -1).toVector.lift(foldedPositionsIdx)).toSet
      assert(foldedValues.forall(_ != "-"))
      val activeValues = handRows.drop(1).flatMap(_.split("\t", -1).toVector.lift(activePositionsIdx)).toVector
      assert(activeValues.forall(value => value.split("\\|").contains("Cutoff")))
      assert(activeValues.forall(value => value.split("\\|").length >= 3), s"expected multiway active seats, got $activeValues")
      val maxLivePlayers = handRows.drop(1).flatMap(_.split("\t", -1).toVector.lift(maxLivePlayersIdx).flatMap(_.toIntOption)).toVector
      assert(maxLivePlayers.forall(_ >= 3), s"expected at least three live players, got $maxLivePlayers")
    finally
      deleteRecursively(root)
  }

  test("playing hall supports nine-max big blind defense contexts") {
    val root = Files.createTempDirectory("playing-hall-ninemax-test-")
    try
      val out = root.resolve("hall-ninemax-out")
      val result = TexasHoldemPlayingHall.run(Array(
        "--hands=18",
        "--reportEvery=9",
        "--learnEveryHands=0",
        "--learningWindowSamples=50",
        "--seed=43",
        s"--outDir=$out",
        "--playerCount=9",
        "--heroPosition=BigBlind",
        "--villainPool=tag,lag,maniac",
        "--heroExplorationRate=0.0",
        "--raiseSize=2.5",
        "--bunchingTrials=8",
        "--equityTrials=80",
        "--saveTrainingTsv=false",
        "--saveDdreTrainingTsv=false"
      ))
      assert(result.isRight, s"nine-max hall run failed: $result")
      val summary = result.toOption.getOrElse(fail("missing hall summary"))
      assertEquals(summary.playerCount, 9)

      val handRows = Files.readAllLines(out.resolve("hands.tsv"), StandardCharsets.UTF_8).asScala.toVector
      assert(handRows.length > 1, "expected nine-max hand log rows")
      val handHeader = handRows.head.split("\t", -1).toVector
      val playerCountIdx = handHeader.indexOf("playerCount")
      val heroPositionIdx = handHeader.indexOf("heroPosition")
      val villainPositionIdx = handHeader.indexOf("villainPosition")
      val openerPositionIdx = handHeader.indexOf("openerPosition")
      val foldedPositionsIdx = handHeader.indexOf("foldedPositions")
      val activePositionsIdx = handHeader.indexOf("activePositions")
      val maxLivePlayersIdx = handHeader.indexOf("maxLivePlayers")
      assert(playerCountIdx >= 0 && heroPositionIdx >= 0 && villainPositionIdx >= 0 && openerPositionIdx >= 0 && foldedPositionsIdx >= 0 && activePositionsIdx >= 0 && maxLivePlayersIdx >= 0)

      val playerCounts = handRows.drop(1).flatMap(_.split("\t", -1).toVector.lift(playerCountIdx)).toSet
      assertEquals(playerCounts, Set("9"))
      val heroPositions = handRows.drop(1).flatMap(_.split("\t", -1).toVector.lift(heroPositionIdx)).toSet
      assertEquals(heroPositions, Set("BigBlind"))
      val villainPositions = handRows.drop(1).flatMap(_.split("\t", -1).toVector.lift(villainPositionIdx)).toSet
      assert(villainPositions.subsetOf(Set("UTG", "UTG1", "UTG2", "Middle", "Hijack", "Cutoff", "Button", "SmallBlind")))
      assert(villainPositions.size >= 2, s"expected multiple opener positions, got $villainPositions")
      val openerPositions = handRows.drop(1).flatMap(_.split("\t", -1).toVector.lift(openerPositionIdx)).toSet
      assert(openerPositions.subsetOf(Set("UTG", "UTG1", "UTG2", "Middle", "Hijack", "Cutoff", "Button", "SmallBlind", "BigBlind")))
      val foldedValues = handRows.drop(1).flatMap(_.split("\t", -1).toVector.lift(foldedPositionsIdx)).toVector
      assert(foldedValues.forall(value => value.nonEmpty && value != "-"))
      val activeValues = handRows.drop(1).flatMap(_.split("\t", -1).toVector.lift(activePositionsIdx)).toVector
      assert(activeValues.forall(value => value.split("\\|").contains("BigBlind")))
      assert(activeValues.forall(value => value.split("\\|").length >= 3), s"expected multiway active seats, got $activeValues")
      val maxLivePlayers = handRows.drop(1).flatMap(_.split("\t", -1).toVector.lift(maxLivePlayersIdx).flatMap(_.toIntOption)).toVector
      assert(maxLivePlayers.forall(_ >= 3), s"expected at least three live players, got $maxLivePlayers")
      val seatAccounting = activeValues.zip(foldedValues).map { case (active, folded) =>
        active.split("\\|").count(_.nonEmpty) + folded.split("\\|").count(_.nonEmpty)
      }
      assert(seatAccounting.forall(_ == 9), s"expected true nine-max seat accounting, got $seatAccounting")
    finally
      deleteRecursively(root)
  }

  test("playing hall supports multiway gto hero decisions") {
    val root = Files.createTempDirectory("playing-hall-gto-multiway-test-")
    try
      val out = root.resolve("hall-gto-multiway-out")
      val result = TexasHoldemPlayingHall.run(Array(
        "--hands=10",
        "--reportEvery=5",
        "--learnEveryHands=0",
        "--learningWindowSamples=50",
        "--seed=47",
        s"--outDir=$out",
        "--playerCount=6",
        "--heroPosition=Cutoff",
        "--heroStyle=gto",
        "--gtoMode=fast",
        "--villainPool=tag,lag,gto",
        "--heroExplorationRate=0.0",
        "--raiseSize=2.5",
        "--bunchingTrials=8",
        "--equityTrials=90",
        "--saveTrainingTsv=false",
        "--saveDdreTrainingTsv=false"
      ))
      assert(result.isRight, s"multiway gto hall run failed: $result")

      val handRows = Files.readAllLines(out.resolve("hands.tsv"), StandardCharsets.UTF_8).asScala.toVector
      assert(handRows.length > 1, "expected multiway gto hand rows")
      val handHeader = handRows.head.split("\t", -1).toVector
      val activePositionsIdx = handHeader.indexOf("activePositions")
      val maxLivePlayersIdx = handHeader.indexOf("maxLivePlayers")
      assert(activePositionsIdx >= 0 && maxLivePlayersIdx >= 0)

      val activeValues = handRows.drop(1).flatMap(_.split("\t", -1).toVector.lift(activePositionsIdx)).toVector
      assert(activeValues.forall(value => value.split("\\|").contains("Cutoff")))
      assert(activeValues.forall(value => value.split("\\|").length >= 3), s"expected multiway seats, got $activeValues")
      val maxLivePlayers = handRows.drop(1).flatMap(_.split("\t", -1).toVector.lift(maxLivePlayersIdx).flatMap(_.toIntOption)).toVector
      assert(maxLivePlayers.forall(_ >= 3), s"expected multiway gto decisions, got $maxLivePlayers")
    finally
      deleteRecursively(root)
  }

  test("playing hall can export reproducible PokerStars review histories from a villain pool".tag(munit.Slow)) {
    val root = Files.createTempDirectory("playing-hall-review-export-test-")
    try
      val firstOut = root.resolve("hall-review-export-a")
      val secondOut = root.resolve("hall-review-export-b")
      val args = Array(
        "--hands=12",
        "--reportEvery=12",
        "--learnEveryHands=0",
        "--learningWindowSamples=50",
        "--seed=31",
        "--heroStyle=adaptive",
        "--heroExplorationRate=0.0",
        "--villainPool=tag,lag,maniac",
        "--raiseSize=2.5",
        "--bunchingTrials=8",
        "--equityTrials=80",
        "--saveTrainingTsv=false",
        "--saveDdreTrainingTsv=false",
        "--saveReviewHandHistory=true"
      )

      val first = TexasHoldemPlayingHall.run(args ++ Array(s"--outDir=$firstOut"))
      val second = TexasHoldemPlayingHall.run(args ++ Array(s"--outDir=$secondOut"))

      assert(first.isRight, s"first export run failed: $first")
      assert(second.isRight, s"second export run failed: $second")

      val firstUpload = firstOut.resolve("review-upload-pokerstars.txt")
      val secondUpload = secondOut.resolve("review-upload-pokerstars.txt")
      assert(Files.exists(firstUpload), "expected first review upload export")
      assert(Files.exists(secondUpload), "expected second review upload export")

      val firstText = Files.readString(firstUpload, StandardCharsets.UTF_8)
      val secondText = Files.readString(secondUpload, StandardCharsets.UTF_8)
      assertEquals(firstText, secondText)
      assert(firstText.contains("PokerStars Hand #"), "expected PokerStars hand headers in export")
      assert(firstText.contains("Dealt to Hero ["), "expected hero hole cards in export")

      val villainNames = firstText.linesIterator.collect {
        case line if line.startsWith("Seat 1: Villain") || line.startsWith("Seat 2: Villain") =>
          line.split(": ", 2).lift(1).map(_.takeWhile(_ != ' ')).getOrElse("")
      }.filter(_.nonEmpty).toSet
      assert(villainNames.size >= 2, s"expected multiple reproducible villain identities, got $villainNames")
    finally
      deleteRecursively(root)
  }

  test("playing hall supports gto villain mode") {
    val root = Files.createTempDirectory("playing-hall-gto-test-")
    try
      val out = root.resolve("hall-gto-out")
      val result = TexasHoldemPlayingHall.run(Array(
        "--hands=8",
        "--reportEvery=4",
        "--learnEveryHands=0",
        "--learningWindowSamples=50",
        "--seed=19",
        s"--outDir=$out",
        "--villainStyle=gto",
        "--heroExplorationRate=0.0",
        "--raiseSize=2.5",
        "--bunchingTrials=8",
        "--equityTrials=80",
        "--saveTrainingTsv=false",
        "--saveDdreTrainingTsv=false"
      ))
      assert(result.isRight, s"gto hall run failed: $result")
      val summary = result.toOption.getOrElse(fail("missing hall summary"))
      assertEquals(summary.handsPlayed, 8)
      val handRows = Files.readAllLines(out.resolve("hands.tsv"), StandardCharsets.UTF_8).asScala.toVector
      assert(handRows.length > 1, "expected hand log rows")
      val handHeader = handRows.head.split("\t", -1).toVector
      val archetypeIdx = handHeader.indexOf("archetype")
      assert(archetypeIdx >= 0, "hands.tsv missing archetype column")
      val sawGto =
        handRows.drop(1).exists { row =>
          val fields = row.split("\t", -1).toVector
          fields.lift(archetypeIdx).contains("gto")
        }
      assert(sawGto, "expected at least one hand row with archetype=gto")
    finally
      deleteRecursively(root)
  }

  test("playing hall supports hero big blind seat") {
    val root = Files.createTempDirectory("playing-hall-bigblind-test-")
    try
      val out = root.resolve("hall-bigblind-out")
      val result = TexasHoldemPlayingHall.run(Array(
        "--hands=12",
        "--reportEvery=6",
        "--learnEveryHands=0",
        "--learningWindowSamples=50",
        "--seed=29",
        s"--outDir=$out",
        "--heroSeat=bigblind",
        "--villainStyle=tag",
        "--heroExplorationRate=0.0",
        "--raiseSize=2.5",
        "--bunchingTrials=8",
        "--equityTrials=80",
        "--saveTrainingTsv=false",
        "--saveDdreTrainingTsv=false"
      ))
      assert(result.isRight, s"big blind hall run failed: $result")
      val handRows = Files.readAllLines(out.resolve("hands.tsv"), StandardCharsets.UTF_8).asScala.toVector
      assert(handRows.length > 1, "expected hand log rows")
      val handHeader = handRows.head.split("\t", -1).toVector
      val heroPositionIdx = handHeader.indexOf("heroPosition")
      val villainPositionIdx = handHeader.indexOf("villainPosition")
      assert(heroPositionIdx >= 0, "hands.tsv missing heroPosition column")
      assert(villainPositionIdx >= 0, "hands.tsv missing villainPosition column")
      val heroPositions = handRows.drop(1).flatMap { row =>
        val fields = row.split("\t", -1).toVector
        fields.lift(heroPositionIdx)
      }.toSet
      assertEquals(heroPositions, Set("BigBlind"))
      val villainPositions = handRows.drop(1).flatMap { row =>
        val fields = row.split("\t", -1).toVector
        fields.lift(villainPositionIdx)
      }.toSet
      assertEquals(villainPositions, Set("Button"))
    finally
      deleteRecursively(root)
  }

  test("playing hall supports gto vs gto mode") {
    val root = Files.createTempDirectory("playing-hall-gto-vs-gto-test-")
    try
      val out = root.resolve("hall-gto-vs-gto-out")
      val result = TexasHoldemPlayingHall.run(Array(
        "--hands=8",
        "--reportEvery=4",
        "--learnEveryHands=0",
        "--learningWindowSamples=50",
        "--seed=23",
        s"--outDir=$out",
        "--heroStyle=gto",
        "--villainStyle=gto",
        "--heroExplorationRate=0.0",
        "--raiseSize=2.5",
        "--bunchingTrials=8",
        "--equityTrials=80",
        "--saveTrainingTsv=false",
        "--saveDdreTrainingTsv=false"
      ))
      assert(result.isRight, s"gto vs gto hall run failed: $result")
      val summary = result.toOption.getOrElse(fail("missing hall summary"))
      assertEquals(summary.handsPlayed, 8)
      assert(summary.exactGtoCacheTotal > 0, "expected exact GTO cache lookups")
      assert(summary.exactGtoCacheMisses > 0, "expected at least one exact GTO cache miss")
      assert(summary.exactGtoSolvedByProvider.nonEmpty, "expected exact GTO provider telemetry")
      assert(summary.exactGtoServedByProvider.nonEmpty, "expected exact GTO served-provider telemetry")
      val handRows = Files.readAllLines(out.resolve("hands.tsv"), StandardCharsets.UTF_8).asScala.toVector
      assert(handRows.length > 1, "expected hand log rows")
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

  private def hole(token: String): HoleCards =
    CliHelpers.parseHoleCards(token)

  private def board(tokens: String*): Board =
    Board.from(tokens.map(parseCard))

  private def parseCard(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card token: $token"))

  private def singleRange(token: String): DiscreteDistribution[HoleCards] =
    DiscreteDistribution(Map(hole(token) -> 1.0))
