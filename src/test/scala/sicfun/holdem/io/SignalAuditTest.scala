package sicfun.holdem.io
import sicfun.holdem.types.*
import sicfun.holdem.model.*

import munit.FunSuite
import sicfun.core.Card

import java.nio.file.Files
import scala.jdk.CollectionConverters.*

/**
 * Tests for the signal audit system: [[SignalBuilder]] and [[SignalAuditLogIO]] integration.
 *
 * Validates:
 *   - [[SignalBuilder.actionRisk]] correctly populates model provenance, reconstruction
 *     paths, features, and metrics
 *   - Risk level classification: low-risk events get Info, high-risk get Critical
 *   - End-to-end write/read roundtrip preserves signal records through [[SignalAuditLogIO]]
 *   - Append operations maintain a single header row
 */
class SignalAuditTest extends FunSuite:
  /** Parses a card token string, failing the test if invalid. */
  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  /** Creates a test model artifact with a uniform (untrained) model and passing calibration gate. */
  private def artifact: TrainedPokerActionModel =
    TrainedPokerActionModel(
      version = ModelVersion(
        id = "model-signal-v1",
        schemaVersion = "poker-action-model-v1",
        source = "unit-test",
        trainedAtEpochMillis = 1000L
      ),
      model = PokerActionModel.uniform,
      calibration = CalibrationSummary(meanBrierScore = 0.2, sampleCount = 10),
      gate = CalibrationGate(maxMeanBrierScore = 0.5),
      trainingSampleCount = 100,
      evaluationSampleCount = 20,
      evaluationStrategy = "holdout-split",
      validationFraction = Some(0.2),
      splitSeed = Some(1L)
    )

  /** Creates a test event on the flop with a fixed board, parameterized by pot geometry and action. */
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

  test("SignalBuilder.actionRisk fills model and reconstruction fields") {
    val signal = SignalBuilder.actionRisk(
      event = event("h1", 3L, "p1", toCall = 40.0, stack = 100.0, pot = 20.0, action = PokerAction.Call),
      artifact = artifact,
      snapshotDirectory = "snapshots/h1",
      modelArtifactDirectory = "artifacts/model-signal-v1",
      generatedAtEpochMillis = 55_000L
    )

    assert(signal.signalId.nonEmpty)
    assertEquals(signal.payload.handId, "h1")
    assertEquals(signal.payload.playerId, "p1")
    assertEquals(signal.payload.modelVersionId, "model-signal-v1")
    assertEquals(signal.payload.generatedAtEpochMillis, 55_000L)
    assertEquals(signal.reconstruction.snapshotDirectory, "snapshots/h1")
    assertEquals(signal.reconstruction.modelArtifactDirectory, "artifacts/model-signal-v1")
    assertEquals(signal.reconstruction.eventSequenceInHand, 3L)
    assert(signal.payload.features.contains("potOdds"))
    assert(signal.payload.features.contains("toCallOverStack"))
    assert(signal.payload.metrics.contains("riskScore"))
  }

  test("SignalBuilder.actionRisk assigns expected levels") {
    val low = SignalBuilder.actionRisk(
      event = event("h-low", 1L, "p1", toCall = 0.0, stack = 100.0, pot = 20.0, action = PokerAction.Check),
      artifact = artifact,
      snapshotDirectory = "snapshots/h-low",
      modelArtifactDirectory = "artifacts/model-signal-v1",
      generatedAtEpochMillis = 77_000L
    )
    assertEquals(low.level, SignalLevel.Info)

    val high = SignalBuilder.actionRisk(
      event = event("h-high", 2L, "p1", toCall = 80.0, stack = 100.0, pot = 20.0, action = PokerAction.Call),
      artifact = artifact,
      snapshotDirectory = "snapshots/h-high",
      modelArtifactDirectory = "artifacts/model-signal-v1",
      generatedAtEpochMillis = 78_000L
    )
    assertEquals(high.level, SignalLevel.Critical)
  }

  test("SignalAuditLogIO write/read roundtrip preserves records") {
    val s1 = SignalBuilder.actionRisk(
      event = event("h1", 0L, "p1", toCall = 20.0, stack = 100.0, pot = 40.0, action = PokerAction.Call),
      artifact = artifact,
      snapshotDirectory = "snapshots/h1",
      modelArtifactDirectory = "artifacts/model-signal-v1",
      generatedAtEpochMillis = 100_000L
    )
    val s2 = SignalBuilder.actionRisk(
      event = event("h2", 1L, "p2", toCall = 10.0, stack = 100.0, pot = 30.0, action = PokerAction.Call),
      artifact = artifact,
      snapshotDirectory = "snapshots/h2",
      modelArtifactDirectory = "artifacts/model-signal-v1",
      generatedAtEpochMillis = 100_001L
    )
    val path = Files.createTempFile("sicfun-signal-audit-", ".tsv")
    try
      SignalAuditLogIO.write(path, Seq(s1, s2))
      val loaded = SignalAuditLogIO.read(path)
      assertEquals(loaded, Vector(s1, s2))
    finally
      Files.deleteIfExists(path)
  }

  test("SignalAuditLogIO append keeps single header and appends rows") {
    val s1 = SignalBuilder.actionRisk(
      event = event("h-append", 0L, "p1", toCall = 5.0, stack = 100.0, pot = 20.0, action = PokerAction.Call),
      artifact = artifact,
      snapshotDirectory = "snapshots/h-append",
      modelArtifactDirectory = "artifacts/model-signal-v1",
      generatedAtEpochMillis = 200_000L
    )
    val s2 = SignalBuilder.actionRisk(
      event = event("h-append", 1L, "p1", toCall = 15.0, stack = 100.0, pot = 20.0, action = PokerAction.Call),
      artifact = artifact,
      snapshotDirectory = "snapshots/h-append",
      modelArtifactDirectory = "artifacts/model-signal-v1",
      generatedAtEpochMillis = 200_001L
    )
    val path = Files.createTempFile("sicfun-signal-audit-append-", ".tsv")
    try
      Files.deleteIfExists(path)
      SignalAuditLogIO.append(path, s1)
      SignalAuditLogIO.append(path, s2)
      val loaded = SignalAuditLogIO.read(path)
      assertEquals(loaded.length, 2)

      val lines = Files.readAllLines(path).asScala.toVector
      val headerCount = lines.count(_.startsWith("signalId\tlevel\t"))
      assertEquals(headerCount, 1)
    finally
      Files.deleteIfExists(path)
  }
