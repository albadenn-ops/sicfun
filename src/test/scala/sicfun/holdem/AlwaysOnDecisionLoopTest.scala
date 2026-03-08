package sicfun.holdem

import munit.FunSuite
import sicfun.core.Card

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters.*

class AlwaysOnDecisionLoopTest extends FunSuite:
  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card token: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  private def writeTrainingTsv(path: Path): Unit =
    val header = "street\tboard\tpotBefore\ttoCall\tposition\tstackBefore\taction\tholecards"
    val rows = Vector(
      "Flop\tTs 9h 8d\t20.0\t10.0\tBigBlind\t180.0\traise:25.0\tAh Ad",
      "Flop\tTs 9h 8d\t20.0\t10.0\tBigBlind\t180.0\tcall\tQc Jc",
      "Flop\tTs 9h 8d\t20.0\t10.0\tBigBlind\t180.0\tfold\t7c 2d",
      "Flop\tTs 9h 8d\t20.0\t0.0\tBigBlind\t180.0\tcheck\tAs Ks",
      "Flop\tTs 9h 8d\t20.0\t10.0\tBigBlind\t180.0\traise:22.0\tKh Kd",
      "Flop\tTs 9h 8d\t20.0\t10.0\tBigBlind\t180.0\tcall\tJh Jd",
      "Flop\tTs 9h 8d\t20.0\t10.0\tBigBlind\t180.0\tfold\t6c 2s",
      "Flop\tTs 9h 8d\t20.0\t0.0\tBigBlind\t180.0\tcheck\tQs Qd"
    )
    Files.write(path, (header +: rows).asJava, StandardCharsets.UTF_8)

  private def seedModelArtifact(modelDir: Path): Unit =
    val board = Board.from(Seq(card("Ts"), card("9h"), card("8d")))
    val state = GameState(
      street = Street.Flop,
      board = board,
      pot = 20.0,
      toCall = 10.0,
      position = Position.BigBlind,
      stackSize = 180.0,
      betHistory = Vector.empty
    )
    val checkState = state.copy(toCall = 0.0)
    val training = Vector.fill(12)((state, hole("Ah", "Ad"), PokerAction.Raise(25.0))) ++
      Vector.fill(12)((state, hole("Qc", "Jc"), PokerAction.Call)) ++
      Vector.fill(12)((state, hole("7c", "2d"), PokerAction.Fold)) ++
      Vector.fill(6)((checkState, hole("As", "Ks"), PokerAction.Check))

    val artifact = PokerActionModel.trainVersioned(
      trainingData = training,
      learningRate = 0.1,
      iterations = 300,
      l2Lambda = 0.001,
      validationFraction = 0.25,
      splitSeed = 7L,
      maxMeanBrierScore = 2.0,
      failOnGate = false,
      modelId = "always-on-loop-seed",
      source = "always-on-loop-test",
      trainedAtEpochMillis = 777777L
    )
    PokerActionModelArtifactIO.save(modelDir, artifact)

  private def seedFeed(feedPath: Path): Unit =
    val handId = "loop-hand-1"
    val board = Board.from(Seq(card("Ts"), card("9h"), card("8d")))

    val villain = PokerEvent(
      handId = handId,
      sequenceInHand = 0L,
      playerId = "villain",
      occurredAtEpochMillis = 1_800_000_000_000L,
      street = Street.Flop,
      position = Position.BigBlind,
      board = board,
      potBefore = 20.0,
      toCall = 0.0,
      stackBefore = 180.0,
      action = PokerAction.Raise(8.0),
      decisionTimeMillis = Some(250L),
      betHistory = Vector(BetAction(0, PokerAction.Call))
    )
    val hero = PokerEvent(
      handId = handId,
      sequenceInHand = 1L,
      playerId = "hero",
      occurredAtEpochMillis = 1_800_000_000_010L,
      street = Street.Flop,
      position = Position.Button,
      board = board,
      potBefore = 28.0,
      toCall = 8.0,
      stackBefore = 150.0,
      action = PokerAction.Call,
      decisionTimeMillis = Some(420L),
      betHistory = Vector(BetAction(0, PokerAction.Call), BetAction(1, PokerAction.Raise(8.0)))
    )

    DecisionLoopEventFeedIO.append(feedPath, villain)
    DecisionLoopEventFeedIO.append(feedPath, hero)

  test("always-on loop consumes feed and emits decisions/signals/snapshots") {
    val root = Files.createTempDirectory("always-on-loop-test-")
    try
      val feed = root.resolve("feed.tsv")
      val modelDir = root.resolve("model")
      val out = root.resolve("out")
      seedModelArtifact(modelDir)
      seedFeed(feed)

      val result = AlwaysOnDecisionLoop.run(Array(
        s"--feedPath=${feed}",
        s"--modelArtifactDir=${modelDir}",
        s"--outputDir=${out}",
        "--heroPlayerId=hero",
        "--heroCards=AcKh",
        "--villainPlayerId=villain",
        "--villainPosition=BigBlind",
        "--tableFormat=ninemax",
        "--openerPosition=Cutoff",
        "--candidateActions=fold,call,raise:20",
        "--bunchingTrials=80",
        "--equityTrials=700",
        "--maxPolls=1",
        "--pollMillis=0"
      ))

      assert(result.isRight, s"loop execution failed: $result")
      val summary = result.toOption.getOrElse(fail("missing loop summary"))
      assertEquals(summary.processedEvents, 2)
      assertEquals(summary.decisionsEmitted, 1)
      assertEquals(summary.retrainCount, 0)

      val decisionsPath = out.resolve("decisions.tsv")
      val signalsPath = out.resolve("signals.tsv")
      val snapshotEvents = out.resolve("snapshots").resolve("loop-hand-1").resolve("events.tsv")
      val contextPath = out.resolve("context-archive.md")
      assert(Files.exists(decisionsPath), s"missing decisions log: $decisionsPath")
      assert(Files.exists(signalsPath), s"missing signals log: $signalsPath")
      assert(Files.exists(snapshotEvents), s"missing snapshot events: $snapshotEvents")
      assert(Files.exists(contextPath), s"missing context archive: $contextPath")

      val decisionLines = Files.readAllLines(decisionsPath, StandardCharsets.UTF_8).asScala.toVector
      assertEquals(decisionLines.length, 2)
      val signals = SignalAuditLogIO.read(signalsPath)
      assertEquals(signals.length, 1)
    finally
      deleteRecursively(root)
  }

  test("always-on loop performs scheduled retraining and model rollover") {
    val root = Files.createTempDirectory("always-on-loop-retrain-test-")
    try
      val feed = root.resolve("feed.tsv")
      val modelDir = root.resolve("model")
      val trainingTsv = root.resolve("training.tsv")
      val out = root.resolve("out")
      seedModelArtifact(modelDir)
      seedFeed(feed)
      writeTrainingTsv(trainingTsv)

      val result = AlwaysOnDecisionLoop.run(Array(
        s"--feedPath=${feed}",
        s"--modelArtifactDir=${modelDir}",
        s"--outputDir=${out}",
        "--heroPlayerId=hero",
        "--heroCards=AcKh",
        "--villainPlayerId=villain",
        "--villainPosition=BigBlind",
        "--tableFormat=ninemax",
        "--openerPosition=Cutoff",
        "--candidateActions=fold,call,raise:20",
        "--bunchingTrials=80",
        "--equityTrials=700",
        "--maxPolls=1",
        "--pollMillis=0",
        "--retrainEnabled=true",
        "--retrainEveryDecisions=1",
        s"--trainingDataPath=${trainingTsv}",
        "--retrainIterations=120",
        "--retrainLearningRate=0.1",
        "--retrainL2Lambda=0.001",
        "--retrainValidationFraction=0.25",
        "--retrainSplitSeed=5",
        "--retrainMaxMeanBrierScore=2.0"
      ))

      assert(result.isRight, s"loop retrain execution failed: $result")
      val summary = result.toOption.getOrElse(fail("missing loop summary"))
      assertEquals(summary.retrainCount, 1)

      val updatesPath = out.resolve("model-updates.tsv")
      assert(Files.exists(updatesPath), s"missing model update log: $updatesPath")
      val updates = Files.readAllLines(updatesPath, StandardCharsets.UTF_8).asScala.toVector
      assert(updates.exists(_.contains("retrain-succeeded")), s"expected retrain-succeeded in updates: $updates")
    finally
      deleteRecursively(root)
  }

  test("always-on loop accepts CFR baseline flags and writes CFR decision columns") {
    val root = Files.createTempDirectory("always-on-loop-cfr-test-")
    try
      val feed = root.resolve("feed.tsv")
      val modelDir = root.resolve("model")
      val out = root.resolve("out")
      seedModelArtifact(modelDir)
      seedFeed(feed)

      val result = AlwaysOnDecisionLoop.run(Array(
        s"--feedPath=${feed}",
        s"--modelArtifactDir=${modelDir}",
        s"--outputDir=${out}",
        "--heroPlayerId=hero",
        "--heroCards=AcKh",
        "--villainPlayerId=villain",
        "--villainPosition=BigBlind",
        "--tableFormat=ninemax",
        "--openerPosition=Cutoff",
        "--candidateActions=fold,call,raise:20",
        "--bunchingTrials=80",
        "--equityTrials=700",
        "--maxPolls=1",
        "--pollMillis=0",
        "--cfrIterations=500",
        "--cfrBlend=0.4",
        "--cfrVillainHands=24",
        "--cfrEquityTrials=800",
        "--cfrVillainReraises=true"
      ))

      assert(result.isRight, s"loop execution with CFR failed: $result")

      val decisionsPath = out.resolve("decisions.tsv")
      val decisionLines = Files.readAllLines(decisionsPath, StandardCharsets.UTF_8).asScala.toVector
      assert(decisionLines.nonEmpty, "expected non-empty decisions output")
      assert(decisionLines.head.contains("cfrLocalExploitability"))
      assert(decisionLines.head.contains("cfrRootDeviationGap"))
      assert(decisionLines.head.contains("cfrVillainDeviationGap"))
    finally
      deleteRecursively(root)
  }

  test("pending hero-raise tracking is scoped per hand id") {
    val tracker = new AlwaysOnDecisionLoop.PendingHeroRaiseTracker()
    tracker.onHeroAction("hand-a", PokerAction.Raise(15.0))

    val crossHandResponse = tracker.onVillainAction("hand-b", PokerAction.Call)
    assertEquals(crossHandResponse, None)

    val sameHandResponse = tracker.onVillainAction("hand-a", PokerAction.Fold)
    assertEquals(sameHandResponse, Some(PokerAction.Fold))
  }

  private def deleteRecursively(path: Path): Unit =
    if Files.exists(path) then
      val stream = Files.walk(path)
      try
        val all = stream.iterator().asScala.toVector.sortBy(_.toString.length).reverse
        all.foreach(Files.deleteIfExists)
      finally
        stream.close()
