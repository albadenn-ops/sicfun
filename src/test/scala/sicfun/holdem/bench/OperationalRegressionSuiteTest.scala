package sicfun.holdem.bench
import sicfun.holdem.types.*
import sicfun.holdem.analysis.*
import sicfun.holdem.io.*
import sicfun.holdem.model.*

import munit.FunSuite
import sicfun.holdem.bench.BenchSupport.card

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters.*

/** End-to-end operational regression test that exercises the full poker analytics
  * pipeline: training data on disk -> model artifact -> hand-state snapshot ->
  * signal audit log generation.
  *
  * This single test validates that:
  *   1. [[TrainPokerActionModel.run]] trains and persists a model artifact with metadata
  *   2. [[HandEngine]] / [[HandStateSnapshotIO]] correctly build and serialize hand state
  *   3. [[GenerateSignals.run]] reads both artifacts and produces a correct audit log
  *   4. Persisted signal fields (modelVersionId, generatedAtEpochMillis, paths) are accurate
  *
  * All artifacts are created in a temporary directory tree and cleaned up afterward.
  */
class OperationalRegressionSuiteTest extends FunSuite:
  /** Builds a synthetic [[PokerEvent]] on a fixed flop board (Ts 9h 8d). */
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
      occurredAtEpochMillis = 100_000L + sequence,
      street = Street.Flop,
      position = Position.Button,
      board = Board.from(Seq(card("Ts"), card("9h"), card("8d"))),
      potBefore = pot,
      toCall = toCall,
      stackBefore = stack,
      action = action
    )

  test("operational regression: train artifact, recover snapshot, and generate audit signals") {
    val tempRoot = Files.createTempDirectory("sicfun-ops-regression-")
    try
      val trainingPath = tempRoot.resolve("training.tsv")
      val modelDir = tempRoot.resolve("model-artifact")
      val snapshotDir = tempRoot.resolve("snapshot")
      val signalsPath = tempRoot.resolve("signals.tsv")

      val header = "street\tboard\tpotBefore\ttoCall\tposition\tstackBefore\taction\tholeCards"
      val rows = Vector(
        "Flop\tTs 9h 8d\t20.0\t10.0\tButton\t200.0\traise:20.0\tAh Kh",
        "Flop\tTs 9h 8d\t20.0\t10.0\tButton\t200.0\tfold\t7c 2d",
        "Flop\tTs 9h 8d\t20.0\t10.0\tButton\t200.0\tcall\tAd Kd",
        "Flop\tTs 9h 8d\t20.0\t0.0\tButton\t200.0\tcheck\t7h 2c",
        "Flop\tTs 9h 8d\t20.0\t10.0\tButton\t200.0\traise:15.0\tAs Ks",
        "Flop\tTs 9h 8d\t20.0\t10.0\tButton\t200.0\tfold\t6c 2s",
        "Flop\tTs 9h 8d\t20.0\t10.0\tButton\t200.0\tcall\tAc Kc",
        "Flop\tTs 9h 8d\t20.0\t0.0\tButton\t200.0\tcheck\t5c 2h",
        "Flop\tTs 9h 8d\t20.0\t10.0\tButton\t200.0\traise:18.0\tQh Jh",
        "Flop\tTs 9h 8d\t20.0\t10.0\tButton\t200.0\tfold\t4c 2d",
        "Flop\tTs 9h 8d\t20.0\t10.0\tButton\t200.0\tcall\tQc Jc",
        "Flop\tTs 9h 8d\t20.0\t0.0\tButton\t200.0\tcheck\t4h 3c"
      )
      Files.write(trainingPath, (header +: rows).asJava, StandardCharsets.UTF_8)

      val trainResult = TrainPokerActionModel.run(Array(
        trainingPath.toString,
        modelDir.toString,
        "--learningRate=0.1",
        "--iterations=300",
        "--l2Lambda=0.001",
        "--maxMeanBrierScore=2.0",
        "--validationFraction=0.25",
        "--splitSeed=7",
        "--failOnGate=true",
        "--modelId=ops-regression-v1",
        "--schemaVersion=schema-v1",
        "--source=ops-regression-test",
        "--trainedAtEpochMillis=777777"
      ))
      assert(trainResult.isRight, s"model training failed: $trainResult")
      assert(Files.exists(modelDir.resolve("metadata.properties")))

      val state0 = HandEngine.newHand("ops-hand-1", startedAt = 99_000L)
      val events = Seq(
        event("ops-hand-1", 0L, "alice", toCall = 10.0, stack = 100.0, pot = 20.0, action = PokerAction.Call),
        event("ops-hand-1", 1L, "bob", toCall = 0.0, stack = 100.0, pot = 20.0, action = PokerAction.Check),
        event("ops-hand-1", 2L, "alice", toCall = 40.0, stack = 90.0, pot = 30.0, action = PokerAction.Raise(40.0)),
        event("ops-hand-1", 3L, "bob", toCall = 40.0, stack = 100.0, pot = 70.0, action = PokerAction.Fold)
      )
      val state = HandEngine.applyEvents(state0, events)
      HandStateSnapshotIO.save(snapshotDir, state)

      val signalResult = GenerateSignals.run(Array(
        snapshotDir.toString,
        modelDir.toString,
        signalsPath.toString,
        "--generatedAtEpochMillis=888888",
        "--warningThreshold=0.4",
        "--criticalThreshold=0.7"
      ))
      assert(signalResult.isRight, s"signal generation failed: $signalResult")
      assertEquals(signalResult.toOption.get.signalCount, events.length)

      val loaded = SignalAuditLogIO.read(signalsPath)
      assertEquals(loaded.length, events.length)
      loaded.foreach { signal =>
        assertEquals(signal.payload.modelVersionId, "ops-regression-v1")
        assertEquals(signal.payload.generatedAtEpochMillis, 888888L)
        assertEquals(signal.reconstruction.snapshotDirectory, snapshotDir.toString)
        assertEquals(signal.reconstruction.modelArtifactDirectory, modelDir.toString)
      }
    finally
      deleteRecursively(tempRoot)
  }

  /** Recursively deletes a directory tree (deepest paths first) for temp cleanup. */
  private def deleteRecursively(path: Path): Unit =
    if Files.exists(path) then
      val stream = Files.walk(path)
      try
        val all = stream.iterator().asScala.toVector.sortBy(_.toString.length).reverse
        all.foreach(Files.deleteIfExists)
      finally stream.close()
