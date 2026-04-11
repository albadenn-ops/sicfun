package sicfun.holdem.io
import sicfun.holdem.types.*

import sicfun.core.Card

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import java.util.Properties
import scala.jdk.CollectionConverters.*

/**
 * Directory-based snapshot persistence for poker hand state in the sicfun system.
 *
 * This object serializes and deserializes a [[HandState]] (the complete in-progress
 * state of a single poker hand) to a two-file directory layout. It is used by the
 * real-time advisor session to checkpoint hand state for crash recovery and by the
 * signal audit system for offline reconstruction of past decisions.
 *
 * Key design decisions:
 *   - '''Two-file layout''': Metadata (handId, timestamps, event count) is stored in
 *     a Java Properties file for quick inspection without parsing the full event log.
 *     Events are stored in a TSV file for human readability and easy grep-ability.
 *   - '''Structural validation on load''': The loader checks event count consistency,
 *     sequence uniqueness, sort order, and timestamp agreement between metadata and
 *     events. This catches silent corruption from manual edits or partial writes.
 *   - '''No handId in the events TSV''': Since all events in a snapshot belong to the
 *     same hand, the handId is stored once in metadata and injected during deserialization,
 *     reducing redundancy and file size.
 *
 * Persist and restore a HandState to/from a directory.
  *
  * Layout:
  *   state.properties  — handId, lastUpdatedAt, eventCount
  *   events.tsv        — one row per event, all PokerEvent fields
  *
  * BetHistory column format: "playerIndex:action" entries joined by "|", or "-" if empty.
  * Action tokens: Fold | Check | Call | Raise:<amount>
  * Board column: space-separated card tokens (e.g. "As Kd 7c"), or "-" if empty.
  * decisionTimeMillis column: Long value, or "-" if None.
  */
object HandStateSnapshotIO:
  private val MetadataFileName = "state.properties"
  private val EventsFileName = "events.tsv"
  private val TsvHeader =
    "sequenceInHand\tplayerId\toccurredAtEpochMillis\tstreet\tposition\tboard" +
      "\tpotBefore\ttoCall\tstackBefore\taction\tdecisionTimeMillis\tbetHistory"
  private val ExpectedColumns = TsvHeader.split("\t").map(_.trim.toLowerCase).toVector

  /** Persist a [[HandState]] to a directory, creating it if necessary.
    *
    * Writes two files:
    *  - `state.properties` containing `handId`, `lastUpdatedAt`, and `eventCount`
    *  - `events.tsv` with one header row followed by one row per [[PokerEvent]]
    *
    * @param directory target directory (will be created if absent)
    * @param state     the hand state to serialize
    */
  def save(directory: Path, state: HandState): Unit =
    Files.createDirectories(directory)
    writeMetadata(directory.resolve(MetadataFileName), state)
    writeEvents(directory.resolve(EventsFileName), state)

  /** Convenience overload accepting a string path. */
  def save(directory: String, state: HandState): Unit =
    save(Paths.get(directory), state)

  /** Restore a [[HandState]] from a snapshot directory.
    *
    * Validates structural invariants on load:
    *  - event count matches metadata
    *  - no duplicate `sequenceInHand` values
    *  - events are sorted ascending
    *  - `lastUpdatedAt` matches the maximum event timestamp
    *
    * @param directory the snapshot directory previously created by [[save]]
    * @throws IllegalArgumentException if any validation check fails
    */
  def load(directory: Path): HandState =
    require(Files.isDirectory(directory), s"snapshot directory does not exist: $directory")
    val meta = readMetadata(directory.resolve(MetadataFileName))
    val handId = readRequired(meta, "handId")
    val lastUpdatedAt = readRequired(meta, "lastUpdatedAt").toLong
    val expectedCount = readRequired(meta, "eventCount").toInt
    val events = readEvents(directory.resolve(EventsFileName), handId)
    require(
      events.length == expectedCount,
      s"expected $expectedCount events in snapshot, got ${events.length}"
    )
    val sequences = events.map(_.sequenceInHand)
    require(
      sequences.distinct.length == sequences.length,
      s"snapshot has duplicate sequenceInHand values for handId=$handId"
    )
    require(
      sequences == sequences.sorted,
      s"snapshot events are not sorted by sequenceInHand for handId=$handId"
    )
    val computedLastUpdated =
      if events.isEmpty then lastUpdatedAt
      else events.map(_.occurredAtEpochMillis).max
    require(
      computedLastUpdated == lastUpdatedAt,
      s"snapshot metadata lastUpdatedAt=$lastUpdatedAt does not match events max timestamp=$computedLastUpdated"
    )
    val appliedSequences = events.map(_.sequenceInHand).toSet
    HandState(handId, events, appliedSequences, lastUpdatedAt)

  def load(directory: String): HandState =
    load(Paths.get(directory))

  // ---- metadata ----------------------------------------------------------------

  /** Writes the hand state metadata (handId, lastUpdatedAt, eventCount) to a Java Properties file. */
  private def writeMetadata(path: Path, state: HandState): Unit =
    val props = Properties()
    props.setProperty("handId", state.handId)
    props.setProperty("lastUpdatedAt", state.lastUpdatedAt.toString)
    props.setProperty("eventCount", state.eventCount.toString)
    val writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)
    try props.store(writer, "HandState snapshot")
    finally writer.close()

  /** Reads and returns the Properties from the metadata file. */
  private def readMetadata(path: Path): Properties =
    require(Files.exists(path), s"missing state metadata: $path")
    val props = Properties()
    val reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)
    try props.load(reader)
    finally reader.close()
    props

  /** Extracts a required non-empty property value, throwing if missing or blank. */
  private def readRequired(props: Properties, key: String): String =
    Option(props.getProperty(key)).map(_.trim).filter(_.nonEmpty)
      .getOrElse(throw new IllegalArgumentException(s"missing required metadata key: $key"))

  // ---- events ------------------------------------------------------------------

  /** Writes all events to a TSV file with a header row. */
  private def writeEvents(path: Path, state: HandState): Unit =
    val writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)
    try
      writer.write(TsvHeader)
      writer.newLine()
      state.events.foreach { e =>
        writer.write(serializeEvent(e))
        writer.newLine()
      }
    finally writer.close()

  /** Reads events from the TSV file, injecting the handId from metadata into each event. */
  private def readEvents(path: Path, handId: String): Vector[PokerEvent] =
    require(Files.exists(path), s"missing events file: $path")
    val lines = Files.readAllLines(path, StandardCharsets.UTF_8).asScala.toVector
    require(lines.nonEmpty, s"events file is empty: $path")
    val header = lines.head.split("\t", -1).map(_.trim.toLowerCase).toVector
    require(
      header == ExpectedColumns,
      s"events TSV header mismatch. expected: ${ExpectedColumns.mkString(",")}; got: ${header.mkString(",")}"
    )
    lines.drop(1).filter(_.trim.nonEmpty).zipWithIndex.map { case (line, idx) =>
      deserializeEvent(handId, path, idx + 2, line)
    }

  // ---- event serialization -----------------------------------------------------

  /** Serializes a single event to a tab-delimited row (without the handId column). */
  private def serializeEvent(e: PokerEvent): String =
    Vector(
      e.sequenceInHand.toString,
      e.playerId,
      e.occurredAtEpochMillis.toString,
      e.street.toString,
      e.position.toString,
      serializeBoard(e.board),
      e.potBefore.toString,
      e.toCall.toString,
      e.stackBefore.toString,
      serializeAction(e.action),
      e.decisionTimeMillis.map(_.toString).getOrElse("-"),
      serializeBetHistory(e.betHistory)
    ).mkString("\t")

  /** Deserializes a TSV row into a PokerEvent, expecting 12 columns (handId is provided externally). */
  private def deserializeEvent(handId: String, path: Path, rowNum: Int, line: String): PokerEvent =
    val cols = line.split("\t", -1).toVector
    require(cols.length == 12, s"$path:$rowNum expected 12 columns, got ${cols.length}")
    def col(i: Int): String = cols(i).trim

    val sequenceInHand = col(0).toLong
    val playerId = col(1)
    val occurredAt = col(2).toLong
    val street = Street.values.find(_.toString == col(3))
      .getOrElse(throw new IllegalArgumentException(s"$path:$rowNum invalid street: ${col(3)}"))
    val position = Position.values.find(_.toString == col(4))
      .getOrElse(throw new IllegalArgumentException(s"$path:$rowNum invalid position: ${col(4)}"))
    val board = deserializeBoard(col(5), path, rowNum)
    val potBefore = col(6).toDouble
    val toCall = col(7).toDouble
    val stackBefore = col(8).toDouble
    val action = deserializeAction(col(9), path, rowNum)
    val decisionMs = col(10) match
      case "-" | "" => None
      case raw => Some(raw.toLong)
    val betHistory = deserializeBetHistory(col(11), path, rowNum)

    PokerEvent(
      handId = handId,
      sequenceInHand = sequenceInHand,
      playerId = playerId,
      occurredAtEpochMillis = occurredAt,
      street = street,
      position = position,
      board = board,
      potBefore = potBefore,
      toCall = toCall,
      stackBefore = stackBefore,
      action = action,
      decisionTimeMillis = decisionMs,
      betHistory = betHistory
    )

  // ---- field serializers -------------------------------------------------------

  private def serializeAction(action: PokerAction): String = action match
    case PokerAction.Fold      => "Fold"
    case PokerAction.Check     => "Check"
    case PokerAction.Call      => "Call"
    case PokerAction.Raise(a)  => s"Raise:$a"

  private def deserializeAction(raw: String, path: Path, rowNum: Int): PokerAction =
    raw.trim.toLowerCase match
      case "fold"                    => PokerAction.Fold
      case "check"                   => PokerAction.Check
      case "call"                    => PokerAction.Call
      case s if s.startsWith("raise:") =>
        val amountStr = raw.trim.drop(6)
        val amount = amountStr.toDoubleOption
          .getOrElse(throw new IllegalArgumentException(s"$path:$rowNum invalid raise amount: $raw"))
        if amount.isNaN || amount.isInfinite || amount <= 0.0 then
          throw new IllegalArgumentException(s"$path:$rowNum raise amount must be positive: $amount")
        PokerAction.Raise(amount)
      case _ =>
        throw new IllegalArgumentException(s"$path:$rowNum invalid action: $raw")

  private def serializeBoard(board: Board): String =
    if board.size == 0 then "-"
    else board.cards.map(_.toToken).mkString(" ")

  private def deserializeBoard(raw: String, path: Path, rowNum: Int): Board =
    if raw == "-" || raw.isEmpty then Board.empty
    else
      val cards = raw.split("\\s+").toVector.map { token =>
        Card.parse(token)
          .getOrElse(throw new IllegalArgumentException(s"$path:$rowNum invalid card token: $token"))
      }
      Board.from(cards)

  private def serializeBetHistory(history: Vector[BetAction]): String =
    if history.isEmpty then "-"
    else history.map(ba => s"${ba.player}:${serializeAction(ba.action)}").mkString("|")

  private def deserializeBetHistory(raw: String, path: Path, rowNum: Int): Vector[BetAction] =
    if raw == "-" || raw.isEmpty then Vector.empty
    else
      raw.split("\\|").toVector.map { part =>
        val colonIdx = part.indexOf(':')
        if colonIdx < 0 then
          throw new IllegalArgumentException(s"$path:$rowNum invalid betAction entry: $part")
        val player = part.take(colonIdx).trim.toInt
        val actionRaw = part.drop(colonIdx + 1)
        BetAction(player, deserializeAction(actionRaw, path, rowNum))
      }

