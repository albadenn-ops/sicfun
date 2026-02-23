package sicfun.holdem

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths, StandardOpenOption}
import scala.jdk.CollectionConverters.*

/** Tab-separated-value (TSV) persistence for [[SignalEnvelope]] audit logs.
  *
  * Provides write, append, and read operations for signal audit log files.
  * Each file has a fixed header row followed by one TSV row per signal.
  *
  * '''Column layout:''' signalId, level, signalType, signalVersion, handId, playerId,
  * sequenceInHand, occurredAtEpochMillis, generatedAtEpochMillis, modelVersionId,
  * modelSchemaVersion, modelSource, message, features, metrics, snapshotDirectory,
  * modelArtifactDirectory, eventSequenceInHand.
  *
  * Map fields (features, metrics) are serialized as pipe-delimited `key=value` pairs
  * sorted by key, or `"-"` if empty. Tabs, newlines, and carriage returns in string
  * fields are replaced with spaces to preserve TSV structure.
  */
object SignalAuditLogIO:
  private val Header =
    "signalId\tlevel\tsignalType\tsignalVersion\thandId\tplayerId\tsequenceInHand" +
      "\toccurredAtEpochMillis\tgeneratedAtEpochMillis\tmodelVersionId\tmodelSchemaVersion" +
      "\tmodelSource\tmessage\tfeatures\tmetrics\tsnapshotDirectory\tmodelArtifactDirectory\teventSequenceInHand"
  private val ExpectedColumns = Header.split("\t", -1).toVector

  /** Writes a complete audit log file, replacing any existing content.
    *
    * @param path    file path (string form)
    * @param signals signals to write (may be empty; header is always written)
    */
  def write(path: String, signals: Seq[SignalEnvelope]): Unit =
    write(Paths.get(path), signals)

  /** Writes a complete audit log file, replacing any existing content. */
  def write(path: Path, signals: Seq[SignalEnvelope]): Unit =
    val rows = Vector(Header) ++ signals.toVector.map(serializeSignal)
    Files.write(path, rows.asJava, StandardCharsets.UTF_8)

  /** Appends a single signal to an existing audit log, creating the file with a header if absent.
    *
    * @param path   file path (string form)
    * @param signal the signal to append
    */
  def append(path: String, signal: SignalEnvelope): Unit =
    append(Paths.get(path), signal)

  /** Appends a single signal to an existing audit log, creating the file with a header if absent. */
  def append(path: Path, signal: SignalEnvelope): Unit =
    if !Files.exists(path) then
      write(path, Seq(signal))
    else
      val line = serializeSignal(signal) + System.lineSeparator()
      Files.write(
        path,
        line.getBytes(StandardCharsets.UTF_8),
        StandardOpenOption.APPEND
      )

  /** Reads and deserializes all signals from an audit log file.
    *
    * Validates that the file exists, is non-empty, and has a matching header row.
    *
    * @param path file path (string form)
    * @return all signal envelopes in file order
    * @throws IllegalArgumentException if the file is missing, empty, or has a header mismatch
    */
  def read(path: String): Vector[SignalEnvelope] =
    read(Paths.get(path))

  /** Reads and deserializes all signals from an audit log file. */
  def read(path: Path): Vector[SignalEnvelope] =
    require(Files.exists(path), s"audit log file not found: $path")
    val lines = Files.readAllLines(path, StandardCharsets.UTF_8).asScala.toVector
    require(lines.nonEmpty, s"audit log is empty: $path")
    val header = lines.head.split("\t", -1).toVector
    require(
      header == ExpectedColumns,
      s"audit log header mismatch. expected: ${ExpectedColumns.mkString(",")} got: ${header.mkString(",")}"
    )
    lines.drop(1).filter(_.trim.nonEmpty).zipWithIndex.map { case (line, index) =>
      deserializeSignal(path, index + 2, line)
    }

  private def serializeSignal(signal: SignalEnvelope): String =
    val payload = signal.payload
    val reconstruction = signal.reconstruction
    Vector(
      sanitizeField(signal.signalId),
      sanitizeField(signal.level.toString),
      sanitizeField(payload.signalType),
      sanitizeField(payload.signalVersion),
      sanitizeField(payload.handId),
      sanitizeField(payload.playerId),
      payload.sequenceInHand.toString,
      payload.occurredAtEpochMillis.toString,
      payload.generatedAtEpochMillis.toString,
      sanitizeField(payload.modelVersionId),
      sanitizeField(payload.modelSchemaVersion),
      sanitizeField(payload.modelSource),
      sanitizeField(payload.message),
      serializeMap(payload.features),
      serializeMap(payload.metrics),
      sanitizeField(reconstruction.snapshotDirectory),
      sanitizeField(reconstruction.modelArtifactDirectory),
      reconstruction.eventSequenceInHand.toString
    ).mkString("\t")

  private def deserializeSignal(path: Path, rowNum: Int, line: String): SignalEnvelope =
    val cols = line.split("\t", -1).toVector
    require(cols.length == ExpectedColumns.length, s"$path:$rowNum expected ${ExpectedColumns.length} columns, got ${cols.length}")
    def col(i: Int): String = cols(i).trim

    val payload = SignalPayload(
      signalType = col(2),
      signalVersion = col(3),
      handId = col(4),
      playerId = col(5),
      sequenceInHand = col(6).toLong,
      occurredAtEpochMillis = col(7).toLong,
      generatedAtEpochMillis = col(8).toLong,
      modelVersionId = col(9),
      modelSchemaVersion = col(10),
      modelSource = col(11),
      message = col(12),
      features = deserializeMap(col(13), path, rowNum, "features"),
      metrics = deserializeMap(col(14), path, rowNum, "metrics")
    )
    val reconstruction = ReconstructionPath(
      snapshotDirectory = col(15),
      modelArtifactDirectory = col(16),
      eventSequenceInHand = col(17).toLong
    )
    val level = SignalLevel.valueOf(col(1))
    SignalEnvelope(col(0), level, payload, reconstruction)

  private def serializeMap(values: Map[String, Double]): String =
    if values.isEmpty then "-"
    else
      values.toVector.sortBy(_._1).map { case (k, v) =>
        s"${sanitizeKey(k)}=${java.lang.Double.toString(v)}"
      }.mkString("|")

  private def deserializeMap(
      raw: String,
      path: Path,
      rowNum: Int,
      fieldName: String
  ): Map[String, Double] =
    if raw == "-" || raw.isEmpty then Map.empty
    else
      raw.split("\\|", -1).toVector.map { entry =>
        val parts = entry.split("=", 2)
        if parts.length != 2 then
          throw new IllegalArgumentException(s"$path:$rowNum invalid $fieldName entry: $entry")
        val key = parts(0).trim
        val value = parts(1).trim.toDoubleOption.getOrElse {
          throw new IllegalArgumentException(s"$path:$rowNum invalid numeric value in $fieldName entry: $entry")
        }
        key -> value
      }.toMap

  private def sanitizeField(value: String): String =
    value.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ').trim

  private def sanitizeKey(key: String): String =
    key.replace('|', '_').replace('=', '_').replace('\t', '_').replace('\n', '_').replace('\r', '_').trim
