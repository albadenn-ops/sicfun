package sicfun.holdem

import munit.FunSuite
import sicfun.core.Card

import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters.*

class GenerateSignalsCliTest extends FunSuite:
  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  private def event(
      handId: String,
      sequence: Long,
      playerId: String,
      toCall: Double,
      stack: Double,
      pot: Double,
      action: PokerAction
  ): PokerEvent =
    PokerEvent(
      handId = handId,
      sequenceInHand = sequence,
      playerId = playerId,
      occurredAtEpochMillis = 10_000L + sequence,
      street = Street.Flop,
      position = Position.Button,
      board = Board.from(Seq(card("Ts"), card("9h"), card("8d"))),
      potBefore = pot,
      toCall = toCall,
      stackBefore = stack,
      action = action
    )

  private def artifact: TrainedPokerActionModel =
    TrainedPokerActionModel(
      version = ModelVersion(
        id = "model-signal-cli-v1",
        schemaVersion = "poker-action-model-v1",
        source = "unit-test",
        trainedAtEpochMillis = 1000L
      ),
      model = PokerActionModel.uniform,
      calibration = CalibrationSummary(meanBrierScore = 0.3, sampleCount = 10),
      gate = CalibrationGate(maxMeanBrierScore = 0.6),
      trainingSampleCount = 50,
      evaluationSampleCount = 10,
      evaluationStrategy = "holdout-split",
      validationFraction = Some(0.2),
      splitSeed = Some(1L)
    )

  test("run generates audit log from snapshot and artifact") {
    val tempRoot = Files.createTempDirectory("sicfun-generate-signals-")
    try
      val snapshotDir = tempRoot.resolve("snapshot")
      val modelDir = tempRoot.resolve("model")
      val outFile = tempRoot.resolve("signals.tsv")

      val state0 = HandEngine.newHand("hand-signal", startedAt = 0L)
      val events = Seq(
        event("hand-signal", 0L, "alice", toCall = 10.0, stack = 100.0, pot = 20.0, action = PokerAction.Call),
        event("hand-signal", 1L, "bob", toCall = 0.0, stack = 100.0, pot = 20.0, action = PokerAction.Check),
        event("hand-signal", 2L, "alice", toCall = 70.0, stack = 100.0, pot = 20.0, action = PokerAction.Call)
      )
      val state = HandEngine.applyEvents(state0, events)
      HandStateSnapshotIO.save(snapshotDir, state)
      PokerActionModelArtifactIO.save(modelDir, artifact)

      val result = GenerateSignals.run(Array(
        snapshotDir.toString,
        modelDir.toString,
        outFile.toString,
        "--generatedAtEpochMillis=12345",
        "--warningThreshold=0.4",
        "--criticalThreshold=0.7"
      ))

      assert(result.isRight, s"expected successful run, got $result")
      val runResult = result.toOption.get
      assertEquals(runResult.signalCount, 3)
      assert(Files.exists(outFile))

      val loaded = SignalAuditLogIO.read(outFile)
      assertEquals(loaded.length, 3)
      loaded.foreach { signal =>
        assertEquals(signal.payload.modelVersionId, "model-signal-cli-v1")
        assertEquals(signal.payload.generatedAtEpochMillis, 12345L)
        assertEquals(signal.reconstruction.snapshotDirectory, snapshotDir.toString)
        assertEquals(signal.reconstruction.modelArtifactDirectory, modelDir.toString)
      }
    finally
      deleteRecursively(tempRoot)
  }

  test("run supports filters and append mode") {
    val tempRoot = Files.createTempDirectory("sicfun-generate-signals-append-")
    try
      val snapshotDir = tempRoot.resolve("snapshot")
      val modelDir = tempRoot.resolve("model")
      val outFile = tempRoot.resolve("signals.tsv")

      val state0 = HandEngine.newHand("hand-append", startedAt = 0L)
      val events = Seq(
        event("hand-append", 0L, "alice", toCall = 5.0, stack = 100.0, pot = 20.0, action = PokerAction.Call),
        event("hand-append", 1L, "bob", toCall = 0.0, stack = 100.0, pot = 20.0, action = PokerAction.Check),
        event("hand-append", 2L, "alice", toCall = 15.0, stack = 100.0, pot = 20.0, action = PokerAction.Call)
      )
      val state = HandEngine.applyEvents(state0, events)
      HandStateSnapshotIO.save(snapshotDir, state)
      PokerActionModelArtifactIO.save(modelDir, artifact)

      val first = GenerateSignals.run(Array(
        snapshotDir.toString,
        modelDir.toString,
        outFile.toString,
        "--playerId=alice",
        "--maxSequence=0",
        "--generatedAtEpochMillis=20000"
      ))
      assert(first.isRight)
      assertEquals(first.toOption.get.signalCount, 1)

      val second = GenerateSignals.run(Array(
        snapshotDir.toString,
        modelDir.toString,
        outFile.toString,
        "--playerId=alice",
        "--minSequence=2",
        "--append=true",
        "--generatedAtEpochMillis=20001"
      ))
      assert(second.isRight)
      assertEquals(second.toOption.get.signalCount, 1)

      val loaded = SignalAuditLogIO.read(outFile)
      assertEquals(loaded.length, 2)
      assertEquals(loaded.map(_.payload.sequenceInHand), Vector(0L, 2L))
      assertEquals(loaded.map(_.payload.playerId).toSet, Set("alice"))
    finally
      deleteRecursively(tempRoot)
  }

  test("run returns Left on invalid options") {
    val result = GenerateSignals.run(Array("snap", "model", "out.tsv", "--criticalThreshold=abc"))
    assert(result.isLeft)
  }

  private def deleteRecursively(path: Path): Unit =
    if Files.exists(path) then
      val stream = Files.walk(path)
      try
        val all = stream.iterator().asScala.toVector.sortBy(_.toString.length).reverse
        all.foreach(Files.deleteIfExists)
      finally stream.close()
