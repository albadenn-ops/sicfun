package sicfun.holdem

import sicfun.core.Card

import java.io.RandomAccessFile
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import scala.jdk.CollectionConverters.*

/** Reads an append-only TSV event feed for the always-on decision loop.
  *
  * Header:
  *   handId, sequenceInHand, playerId, occurredAtEpochMillis, street, position,
  *   board, potBefore, toCall, stackBefore, action, decisionTimeMillis, betHistory
  */
object DecisionLoopEventFeedIO:
  val Header: String =
    "handId\tsequenceInHand\tplayerId\toccurredAtEpochMillis\tstreet\tposition\tboard" +
      "\tpotBefore\ttoCall\tstackBefore\taction\tdecisionTimeMillis\tbetHistory"

  private val ExpectedColumns = Header.split("\t", -1).map(_.trim.toLowerCase).toVector
  private val appendLock = new AnyRef

  final case class FeedEvent(lineNumber: Int, event: PokerEvent)

  def read(path: String): Vector[FeedEvent] =
    read(Paths.get(path))

  def read(path: Path): Vector[FeedEvent] =
    require(Files.exists(path), s"event feed file not found: $path")
    val lines = Files.readAllLines(path, StandardCharsets.UTF_8).asScala.toVector
    require(lines.nonEmpty, s"event feed file is empty: $path")
    val header = lines.head.split("\t", -1).map(_.trim.toLowerCase).toVector
    require(
      header == ExpectedColumns,
      s"event feed header mismatch. expected: ${ExpectedColumns.mkString(",")} got: ${header.mkString(",")}"
    )
    lines.drop(1).zipWithIndex.flatMap { case (line, idx) =>
      val rowNum = idx + 2
      if line.trim.isEmpty then None
      else Some(FeedEvent(rowNum, deserializeEvent(path, rowNum, line)))
    }

  /** Reads only new lines appended after `byteOffset`.
    *
    * Returns the parsed events and the new byte offset (end of the data read).
    * A `byteOffset` of 0 skips the header line automatically.
    *
    * @param path       the feed file path
    * @param byteOffset the byte position to start reading from (0 = beginning of file)
    * @return (new events, updated byte offset)
    */
  def readIncremental(path: Path, byteOffset: Long): (Vector[FeedEvent], Long) =
    require(Files.exists(path), s"event feed file not found: $path")
    val raf = new RandomAccessFile(path.toFile, "r")
    try
      val fileLength = raf.length()
      if fileLength <= byteOffset then
        return (Vector.empty, byteOffset)
      val effectiveOffset =
        if byteOffset == 0L then
          raf.seek(0L)
          val headerRaw = raf.readLine()
          require(headerRaw != null, s"event feed file is empty: $path")
          val header = new String(headerRaw.getBytes(StandardCharsets.ISO_8859_1), StandardCharsets.UTF_8)
            .split("\t", -1)
            .map(_.trim.toLowerCase)
            .toVector
          require(
            header == ExpectedColumns,
            s"event feed header mismatch. expected: ${ExpectedColumns.mkString(",")} got: ${header.mkString(",")}"
          )
          raf.getFilePointer
        else
          byteOffset
      raf.seek(effectiveOffset)
      val builder = Vector.newBuilder[FeedEvent]
      var line = raf.readLine()
      var lineNum = 0
      while line != null do
        lineNum += 1
        val decoded = new String(line.getBytes(StandardCharsets.ISO_8859_1), StandardCharsets.UTF_8)
        if decoded.trim.nonEmpty then
          builder += FeedEvent(lineNum, deserializeEvent(path, lineNum, decoded))
        line = raf.readLine()
      (builder.result(), raf.getFilePointer)
    finally
      raf.close()

  private def deserializeEvent(path: Path, rowNum: Int, line: String): PokerEvent =
    val cols = line.split("\t", -1).toVector
    require(cols.length == 13, s"$path:$rowNum expected 13 columns, got ${cols.length}")
    def col(i: Int): String = cols(i).trim

    val handId = col(0)
    val sequenceInHand = col(1).toLong
    val playerId = col(2)
    val occurredAt = col(3).toLong
    val street = Street.values.find(_.toString == col(4))
      .getOrElse(throw new IllegalArgumentException(s"$path:$rowNum invalid street: ${col(4)}"))
    val position = Position.values.find(_.toString == col(5))
      .getOrElse(throw new IllegalArgumentException(s"$path:$rowNum invalid position: ${col(5)}"))
    val board = deserializeBoard(col(6), path, rowNum)
    val potBefore = col(7).toDouble
    val toCall = col(8).toDouble
    val stackBefore = col(9).toDouble
    val action = deserializeAction(col(10), path, rowNum)
    val decisionMs = col(11) match
      case "-" | "" => None
      case raw => Some(raw.toLong)
    val betHistory = deserializeBetHistory(col(12), path, rowNum)

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

  private def serializeAction(action: PokerAction): String = action match
    case PokerAction.Fold => "Fold"
    case PokerAction.Check => "Check"
    case PokerAction.Call => "Call"
    case PokerAction.Raise(a) => s"Raise:$a"

  private def deserializeAction(raw: String, path: Path, rowNum: Int): PokerAction =
    raw.trim.toLowerCase match
      case "fold" => PokerAction.Fold
      case "check" => PokerAction.Check
      case "call" => PokerAction.Call
      case s if s.startsWith("raise:") =>
        val amount = raw.trim.drop(6).toDoubleOption
          .getOrElse(throw new IllegalArgumentException(s"$path:$rowNum invalid raise amount: $raw"))
        if amount.isNaN || amount.isInfinite || amount <= 0.0 then
          throw new IllegalArgumentException(s"$path:$rowNum raise amount must be positive: $amount")
        PokerAction.Raise(amount)
      case _ =>
        throw new IllegalArgumentException(s"$path:$rowNum invalid action: $raw")

  private def deserializeBoard(raw: String, path: Path, rowNum: Int): Board =
    if raw == "-" || raw.isEmpty then Board.empty
    else
      val cards = raw.split("\\s+").toVector.map { token =>
        Card.parse(token)
          .getOrElse(throw new IllegalArgumentException(s"$path:$rowNum invalid card token: $token"))
      }
      Board.from(cards)

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

  def append(path: Path, event: PokerEvent): Unit =
    appendLock.synchronized {
      Option(path.getParent).foreach(parent => Files.createDirectories(parent))
      if !Files.exists(path) || Files.size(path) == 0L then
        Files.write(
          path,
          Vector(Header).asJava,
          StandardCharsets.UTF_8,
          java.nio.file.StandardOpenOption.CREATE,
          java.nio.file.StandardOpenOption.TRUNCATE_EXISTING
        )
      val row = Vector(
        event.handId,
        event.sequenceInHand.toString,
        event.playerId,
        event.occurredAtEpochMillis.toString,
        event.street.toString,
        event.position.toString,
        if event.board.cards.isEmpty then "-" else event.board.cards.map(_.toToken).mkString(" "),
        event.potBefore.toString,
        event.toCall.toString,
        event.stackBefore.toString,
        serializeAction(event.action),
        event.decisionTimeMillis.map(_.toString).getOrElse("-"),
        if event.betHistory.isEmpty then "-"
        else event.betHistory.map(ba => s"${ba.player}:${serializeAction(ba.action)}").mkString("|")
      ).mkString("\t")
      Files.write(
        path,
        Vector(row).asJava,
        StandardCharsets.UTF_8,
        java.nio.file.StandardOpenOption.CREATE,
        java.nio.file.StandardOpenOption.APPEND
      )
    }
