package sicfun.holdem.runtime
import sicfun.holdem.types.*
import sicfun.holdem.model.*
import sicfun.holdem.io.*
import sicfun.holdem.engine.*
import sicfun.holdem.history.*

import munit.FunSuite
import sicfun.core.Card

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters.*

/** Tests for [[AlwaysOnDecisionLoop]] — the polling-based live decision service.
  *
  * Verifies the full lifecycle of the always-on loop:
  *  - '''Feed consumption:''' Reading incremental events from a TSV feed file, updating
  *    per-hand state snapshots, and persisting byte offsets for resumption.
  *  - '''Decision emission:''' Producing hero recommendations for hero-originated events,
  *    writing decision logs, signal audit logs, context archives, and snapshot files.
  *  - '''Scheduled retraining:''' Triggering model retraining after N decisions, recording
  *    retrain success/failure in the model-updates log, and hot-swapping the active model.
  *  - '''Feed rotation:''' Handling empty feeds (e.g., log rotation) by resetting the
  *    byte offset to zero and resuming cleanly when the feed is repopulated.
  *  - '''CFR baseline:''' Verifying that CFR columns (attribution, blend weight, regret,
  *    exploitability, deviation gaps) appear in the decisions log when CFR is enabled.
  *  - '''Opponent memory:''' Preloading remembered archetype posteriors from persisted
  *    opponent profile stores, and persisting new villain observations back after processing.
  *  - '''Hero raise tracking:''' Scoping pending hero raises per hand ID to avoid cross-hand
  *    villain response attribution.
  *  - '''Archetype replay:''' Composing remembered priors with session raise-response counts
  *    to reconstruct the archetype posterior after a restart.
  *  - '''Observation deduplication:''' Merging remembered villain events with current-hand
  *    events, excluding duplicates by (handId, sequence, playerId) key.
  */
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

  private def seedRaiseResponseFeed(feedPath: Path): Unit =
    val handId = "loop-hand-2"
    val board = Board.from(Seq(card("Ts"), card("9h"), card("8d")))

    val hero = PokerEvent(
      handId = handId,
      sequenceInHand = 0L,
      playerId = "hero",
      occurredAtEpochMillis = 1_800_000_100_000L,
      street = Street.Flop,
      position = Position.BigBlind,
      board = board,
      potBefore = 20.0,
      toCall = 0.0,
      stackBefore = 180.0,
      action = PokerAction.Raise(12.0),
      decisionTimeMillis = Some(200L),
      betHistory = Vector.empty
    )
    val villain = PokerEvent(
      handId = handId,
      sequenceInHand = 1L,
      playerId = "villain",
      occurredAtEpochMillis = 1_800_000_100_010L,
      street = Street.Flop,
      position = Position.Button,
      board = board,
      potBefore = 32.0,
      toCall = 12.0,
      stackBefore = 180.0,
      action = PokerAction.Fold,
      decisionTimeMillis = Some(180L),
      betHistory = Vector(BetAction(0, PokerAction.Raise(12.0)))
    )

    DecisionLoopEventFeedIO.append(feedPath, hero)
    DecisionLoopEventFeedIO.append(feedPath, villain)

  private def seedOpponentStore(path: Path): Unit =
    val profile = OpponentProfile(
      site = "pokerstars",
      playerName = "villain",
      handsObserved = 12,
      firstSeenEpochMillis = 1_700_000_000_000L,
      lastSeenEpochMillis = 1_800_000_000_000L,
      actionSummary = OpponentActionSummary(folds = 2, raises = 8, calls = 2, checks = 1),
      raiseResponses = sicfun.holdem.engine.RaiseResponseCounts(folds = 0, calls = 1, raises = 12),
      recentEvents = Vector.empty,
      seenHandIds = Vector.tabulate(12)(idx => s"mem-$idx")
    )
    OpponentProfileStore.save(path, OpponentProfileStore(Vector(profile)))

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

  test("always-on loop resets persisted feed offset after an empty feed rotation") {
    val root = Files.createTempDirectory("always-on-loop-rotation-test-")
    try
      val feed = root.resolve("feed.tsv")
      val modelDir = root.resolve("model")
      val out = root.resolve("out")
      val offsetFile = out.resolve("feed-offset.txt")
      seedModelArtifact(modelDir)
      seedFeed(feed)

      val args = Array(
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
      )

      val initial = AlwaysOnDecisionLoop.run(args)
      assert(initial.isRight, s"initial loop execution failed: $initial")
      assert(Files.exists(offsetFile), s"missing offset file: $offsetFile")
      val firstOffset = Files.readString(offsetFile, StandardCharsets.UTF_8).trim.toLong
      assert(firstOffset > 0L, s"expected positive initial feed offset, got $firstOffset")

      Files.write(feed, Array.emptyByteArray)

      val emptyRotation = AlwaysOnDecisionLoop.run(args)
      assert(emptyRotation.isRight, s"loop execution after empty rotation failed: $emptyRotation")
      assertEquals(emptyRotation.toOption.getOrElse(fail("missing empty-rotation summary")).processedEvents, 0)
      assertEquals(Files.readString(offsetFile, StandardCharsets.UTF_8).trim, "0")

      seedFeed(feed)

      val replay = AlwaysOnDecisionLoop.run(args)
      assert(replay.isRight, s"loop execution after feed recreation failed: $replay")
      val replaySummary = replay.toOption.getOrElse(fail("missing replay summary"))
      assertEquals(replaySummary.processedEvents, 2)
      assertEquals(replaySummary.decisionsEmitted, 1)
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
      assert(decisionLines.head.contains("decisionAttribution"))
      assert(decisionLines.head.contains("decisionAttributionReason"))
      assert(decisionLines.head.contains("cfrRequestedBlendWeight"))
      assert(decisionLines.head.contains("cfrEffectiveBlendWeight"))
      assert(decisionLines.head.contains("cfrChosenActionRegret"))
      assert(decisionLines.head.contains("cfrLocalExploitability"))
      assert(decisionLines.head.contains("cfrRootDeviationGap"))
      assert(decisionLines.head.contains("cfrVillainDeviationGap"))
      assert(decisionLines.length >= 2)
      assert(decisionLines(1).contains("BlendedWithBaseline"))
    finally
      deleteRecursively(root)
  }

  test("always-on loop preloads remembered opponent archetype from store") {
    val root = Files.createTempDirectory("always-on-loop-memory-test-")
    try
      val feed = root.resolve("feed.tsv")
      val modelDir = root.resolve("model")
      val store = root.resolve("profiles.json")
      val out = root.resolve("out")
      seedModelArtifact(modelDir)
      seedFeed(feed)
      seedOpponentStore(store)

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
        s"--opponentStore=${store}",
        "--opponentSite=pokerstars",
        "--opponentName=villain"
      ))

      assert(result.isRight, s"loop execution failed: $result")
      val decisionsPath = out.resolve("decisions.tsv")
      val decisionLines = Files.readAllLines(decisionsPath, StandardCharsets.UTF_8).asScala.toVector
      assertEquals(decisionLines.length, 2)
      assert(decisionLines(1).contains("\tManiac\t"), s"expected remembered Maniac archetype in: ${decisionLines(1)}")
    finally
      deleteRecursively(root)
  }

  test("always-on loop persists new villain observations back to the opponent store") {
    val root = Files.createTempDirectory("always-on-loop-store-update-test-")
    try
      val feed = root.resolve("feed.tsv")
      val modelDir = root.resolve("model")
      val store = root.resolve("profiles.json")
      val out = root.resolve("out")
      seedModelArtifact(modelDir)
      seedRaiseResponseFeed(feed)
      seedOpponentStore(store)

      val result = AlwaysOnDecisionLoop.run(Array(
        s"--feedPath=${feed}",
        s"--modelArtifactDir=${modelDir}",
        s"--outputDir=${out}",
        "--heroPlayerId=hero",
        "--heroCards=AcKh",
        "--villainPlayerId=villain",
        "--villainPosition=Button",
        "--tableFormat=ninemax",
        "--openerPosition=Cutoff",
        "--candidateActions=fold,call,raise:20",
        "--bunchingTrials=80",
        "--equityTrials=700",
        "--maxPolls=1",
        "--pollMillis=0",
        s"--opponentStore=${store}",
        "--opponentSite=pokerstars",
        "--opponentName=villain"
      ))

      assert(result.isRight, s"loop execution failed: $result")
      val updated = OpponentProfileStore.load(store)
        .find("pokerstars", "villain")
        .getOrElse(fail("expected updated opponent profile"))

      assertEquals(updated.handsObserved, 13)
      assertEquals(updated.actionSummary.folds, 3)
      assertEquals(updated.raiseResponses.folds, 1)
      assert(updated.seenHandIds.contains("loop-hand-2"))
      assert(updated.recentEvents.exists(event =>
        event.handId == "loop-hand-2" && event.playerId == "villain" && event.action == PokerAction.Fold
      ))
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

  test("replayed archetype posterior composes remembered prior with session raise responses") {
    val rememberedPosterior = ArchetypeLearning.posteriorFromCounts(RaiseResponseCounts(folds = 1, calls = 2, raises = 3))
    val sessionRaiseResponses = RaiseResponseCounts(folds = 2, calls = 1, raises = 4)

    val actual = AlwaysOnDecisionLoop.replayedArchetypePosterior(
      rememberedPosterior = Some(rememberedPosterior),
      sessionRaiseResponses = sessionRaiseResponses
    ).getOrElse(fail("expected replayed posterior"))

    val expected = ArchetypeLearning.posteriorFromCounts(
      sessionRaiseResponses,
      prior = rememberedPosterior
    )

    PlayerArchetype.values.foreach { archetype =>
      assertEqualsDouble(
        actual.probabilityOf(archetype),
        expected.probabilityOf(archetype),
        1e-12
      )
    }
  }

  test("mergeVillainObservations excludes remembered duplicates from the active hand") {
    val board = Board.from(Seq(card("Ts"), card("9h"), card("8d")))
    val duplicateVillainEvent = PokerEvent(
      handId = "active-hand",
      sequenceInHand = 0L,
      playerId = "villain",
      occurredAtEpochMillis = 1_800_000_200_000L,
      street = Street.Flop,
      position = Position.BigBlind,
      board = board,
      potBefore = 20.0,
      toCall = 10.0,
      stackBefore = 180.0,
      action = PokerAction.Call,
      decisionTimeMillis = Some(150L),
      betHistory = Vector.empty
    )
    val rememberedPriorVillainEvent = duplicateVillainEvent.copy(
      handId = "remembered-hand",
      occurredAtEpochMillis = 1_800_000_100_000L,
      potBefore = 14.0,
      toCall = 4.0,
      action = PokerAction.Raise(9.0)
    )
    val heroCurrentEvent = duplicateVillainEvent.copy(
      sequenceInHand = 1L,
      playerId = "hero",
      position = Position.Button,
      potBefore = 30.0,
      toCall = 0.0,
      action = PokerAction.Raise(24.0)
    )

    val merged = AlwaysOnDecisionLoop.mergeVillainObservations(
      rememberedEvents = Vector(duplicateVillainEvent, rememberedPriorVillainEvent),
      currentHandEvents = Vector(duplicateVillainEvent, heroCurrentEvent),
      villainPlayerId = "villain"
    )

    assertEquals(merged.length, 2)
    assertEquals(merged.map(_.state.pot), Vector(14.0, 20.0))
    assertEquals(merged.map(_.action), Vector(PokerAction.Raise(9.0), PokerAction.Call))
  }

  private def deleteRecursively(path: Path): Unit =
    if Files.exists(path) then
      val stream = Files.walk(path)
      try
        val all = stream.iterator().asScala.toVector.sortBy(_.toString.length).reverse
        all.foreach(Files.deleteIfExists)
      finally
        stream.close()
