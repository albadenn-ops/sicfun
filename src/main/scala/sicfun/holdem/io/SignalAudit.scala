package sicfun.holdem.io
import sicfun.holdem.types.*
import sicfun.holdem.model.*

import scala.util.hashing.MurmurHash3

/** Severity level assigned to a generated signal.
  *
  * Levels are ordered by increasing urgency: [[Info]] for routine observations,
  * [[Warning]] for elevated risk, and [[Critical]] for high-risk decision points.
  */
enum SignalLevel:
  case Info, Warning, Critical

/** The core data payload of a signal, carrying identification, timing, model provenance,
  * a human-readable message, and the numeric features/metrics that produced the signal.
  *
  * All string fields are validated non-empty; numeric maps are validated for finite values.
  *
  * @param signalType             signal category identifier (e.g. `"action-risk"`)
  * @param signalVersion          schema version of this signal type
  * @param handId                 unique identifier of the poker hand that triggered the signal
  * @param playerId               identifier of the player whose action was evaluated
  * @param sequenceInHand         ordinal position of the event within the hand's event sequence
  * @param occurredAtEpochMillis  wall-clock time when the original poker action occurred
  * @param generatedAtEpochMillis wall-clock time when this signal was generated
  * @param modelVersionId         version identifier of the model used for evaluation
  * @param modelSchemaVersion     schema version of the model artifact format
  * @param modelSource            provenance string describing how the model was trained
  * @param message                human-readable summary (e.g. risk score breakdown)
  * @param features               named input features fed to the risk scorer
  * @param metrics                named output metrics (e.g. risk score, Brier score)
  */
final case class SignalPayload(
    signalType: String,
    signalVersion: String,
    handId: String,
    playerId: String,
    sequenceInHand: Long,
    occurredAtEpochMillis: Long,
    generatedAtEpochMillis: Long,
    modelVersionId: String,
    modelSchemaVersion: String,
    modelSource: String,
    message: String,
    features: Map[String, Double],
    metrics: Map[String, Double]
):
  require(signalType.trim.nonEmpty, "signalType must be non-empty")
  require(signalVersion.trim.nonEmpty, "signalVersion must be non-empty")
  require(handId.trim.nonEmpty, "handId must be non-empty")
  require(playerId.trim.nonEmpty, "playerId must be non-empty")
  require(sequenceInHand >= 0L, "sequenceInHand must be non-negative")
  require(occurredAtEpochMillis >= 0L, "occurredAtEpochMillis must be non-negative")
  require(generatedAtEpochMillis >= 0L, "generatedAtEpochMillis must be non-negative")
  require(modelVersionId.trim.nonEmpty, "modelVersionId must be non-empty")
  require(modelSchemaVersion.trim.nonEmpty, "modelSchemaVersion must be non-empty")
  require(modelSource.trim.nonEmpty, "modelSource must be non-empty")
  require(message.trim.nonEmpty, "message must be non-empty")
  require(features.nonEmpty, "features must be non-empty")
  require(
    features.forall { case (k, v) => k.trim.nonEmpty && !v.isNaN && !v.isInfinite },
    "features must contain non-empty keys and finite values"
  )
  require(
    metrics.forall { case (k, v) => k.trim.nonEmpty && !v.isNaN && !v.isInfinite },
    "metrics must contain non-empty keys and finite values"
  )

/** Filesystem pointers enabling offline reconstruction of the exact state that produced a signal.
  *
  * By recording the snapshot directory (hand state), model artifact directory, and event
  * sequence number, any signal can be deterministically reproduced for debugging or audit.
  *
  * @param snapshotDirectory      path to the [[HandStateSnapshotIO]] directory used at generation time
  * @param modelArtifactDirectory path to the [[PokerActionModelArtifactIO]] directory
  * @param eventSequenceInHand    the `sequenceInHand` of the specific event that triggered the signal
  */
final case class ReconstructionPath(
    snapshotDirectory: String,
    modelArtifactDirectory: String,
    eventSequenceInHand: Long
):
  require(snapshotDirectory.trim.nonEmpty, "snapshotDirectory must be non-empty")
  require(modelArtifactDirectory.trim.nonEmpty, "modelArtifactDirectory must be non-empty")
  require(eventSequenceInHand >= 0L, "eventSequenceInHand must be non-negative")

/** Top-level signal container pairing a unique ID, severity level, data payload,
  * and reconstruction metadata.
  *
  * The `signalId` is a deterministic hash of the payload's identifying fields,
  * ensuring idempotent signal generation (the same input always yields the same ID).
  *
  * @param signalId       stable, deterministic identifier derived from payload fields via MurmurHash3
  * @param level          severity classification ([[SignalLevel.Info]], [[Warning]], or [[Critical]])
  * @param payload        the signal's data content
  * @param reconstruction filesystem pointers for offline reproduction of this signal
  */
final case class SignalEnvelope(
    signalId: String,
    level: SignalLevel,
    payload: SignalPayload,
    reconstruction: ReconstructionPath
):
  require(signalId.trim.nonEmpty, "signalId must be non-empty")

/** Factory for constructing [[SignalEnvelope]] instances from poker events and trained models.
  *
  * Currently supports a single signal type, `"action-risk"`, which scores decision-point
  * risk using a weighted combination of `toCallOverStack` (70%) and `potOdds` (30%),
  * clamped to [0, 1]. The resulting score is classified into [[SignalLevel]] tiers
  * based on configurable thresholds.
  */
object SignalBuilder:
  private inline val CommitmentWeight = 0.7
  private inline val PotOddsWeight = 0.3

  /** Builds an `"action-risk"` signal for a single poker event.
    *
    * The risk score formula is: `min(1.0, toCallOverStack * CommitmentWeight + potOdds * PotOddsWeight)`.
    * The score is classified as:
    *   - [[SignalLevel.Critical]] if `riskScore >= criticalThreshold`
    *   - [[SignalLevel.Warning]]  if `riskScore >= warningThreshold`
    *   - [[SignalLevel.Info]]     otherwise
    *
    * @param event                   the poker event to evaluate
    * @param artifact                the trained model artifact (used for provenance metadata)
    * @param snapshotDirectory       path recorded in [[ReconstructionPath]] for reproducibility
    * @param modelArtifactDirectory  path recorded in [[ReconstructionPath]]
    * @param generatedAtEpochMillis  generation timestamp (defaults to current wall-clock time)
    * @param warningThreshold        score threshold for [[SignalLevel.Warning]] (default 0.4)
    * @param criticalThreshold       score threshold for [[SignalLevel.Critical]] (default 0.7)
    * @return a fully populated [[SignalEnvelope]] with a deterministic signal ID
    */
  def actionRisk(
      event: PokerEvent,
      artifact: TrainedPokerActionModel,
      snapshotDirectory: String,
      modelArtifactDirectory: String,
      generatedAtEpochMillis: Long = System.currentTimeMillis(),
      warningThreshold: Double = 0.4,
      criticalThreshold: Double = 0.7
  ): SignalEnvelope =
    require(
      warningThreshold >= 0.0 && warningThreshold <= 1.0,
      "warningThreshold must be in [0, 1]"
    )
    require(
      criticalThreshold >= warningThreshold && criticalThreshold <= 1.0,
      "criticalThreshold must be in [warningThreshold, 1]"
    )

    val extracted = FeatureExtractor.extract(event)
    val features = FeatureExtractor.featureNames.zip(extracted.values).toMap
    val toCallOverStack = features.getOrElse("toCallOverStack", 0.0)
    val potOdds = features.getOrElse("potOdds", 0.0)
    val riskScore = math.min(1.0, toCallOverStack * CommitmentWeight + potOdds * PotOddsWeight)
    val level =
      if riskScore >= criticalThreshold then SignalLevel.Critical
      else if riskScore >= warningThreshold then SignalLevel.Warning
      else SignalLevel.Info

    val message =
      f"action-risk score=$riskScore%.4f (toCallOverStack=$toCallOverStack%.4f potOdds=$potOdds%.4f)"
    val payload = SignalPayload(
      signalType = "action-risk",
      signalVersion = "1",
      handId = event.handId,
      playerId = event.playerId,
      sequenceInHand = event.sequenceInHand,
      occurredAtEpochMillis = event.occurredAtEpochMillis,
      generatedAtEpochMillis = generatedAtEpochMillis,
      modelVersionId = artifact.version.id,
      modelSchemaVersion = artifact.version.schemaVersion,
      modelSource = artifact.version.source,
      message = message,
      features = features,
      metrics = Map(
        "riskScore" -> riskScore,
        "modelMeanBrierScore" -> artifact.calibration.meanBrierScore,
        "modelGatePassed" -> (if artifact.gatePassed then 1.0 else 0.0)
      )
    )
    val reconstruction = ReconstructionPath(
      snapshotDirectory = snapshotDirectory,
      modelArtifactDirectory = modelArtifactDirectory,
      eventSequenceInHand = event.sequenceInHand
    )
    val signalId = stableSignalId(
      payload.signalType,
      payload.handId,
      payload.playerId,
      payload.sequenceInHand.toString,
      payload.generatedAtEpochMillis.toString,
      payload.modelVersionId
    )
    SignalEnvelope(signalId, level, payload, reconstruction)

  /** Produces a deterministic hex signal ID from the concatenation of identifying fields.
    *
    * Uses [[scala.util.hashing.MurmurHash3.stringHash]] for fast, well-distributed hashing.
    * The result is prefixed with `"sig-"` for easy visual identification.
    */
  private def stableSignalId(parts: String*): String =
    val input = parts.mkString("|")
    val hash = MurmurHash3.stringHash(input)
    s"sig-${java.lang.Integer.toUnsignedString(hash, 16)}"
