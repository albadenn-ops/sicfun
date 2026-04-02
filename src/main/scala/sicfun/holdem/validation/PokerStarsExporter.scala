package sicfun.holdem.validation

import sicfun.holdem.types.{Board, Street, PokerAction}

import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.Locale

/** Result of chunking hand records for export. */
final case class ExportChunk(
    chunkIndex: Int,
    handCount: Int,
    text: String
)

/** Exports HandRecords to PokerStars format parseable by HandHistoryImport. */
object PokerStarsExporter:
  private val BaseTimestamp = LocalDateTime.of(2026, 1, 1, 12, 0, 0)
  private val TimeFmt = DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss")
  private val StartingStack = 100.0

  def exportHands(records: Vector[HandRecord], heroName: String, villainName: String): String =
    val sb = new StringBuilder(records.length * 800)
    records.foreach(r => appendHand(sb, r, heroName, villainName))
    sb.toString()

  def exportChunked(
      records: Vector[HandRecord],
      heroName: String,
      villainName: String,
      chunkSize: Int = 1000
  ): Vector[ExportChunk] =
    records.grouped(chunkSize).zipWithIndex.map { case (chunk, idx) =>
      ExportChunk(idx, chunk.length, exportHands(chunk.toVector, heroName, villainName))
    }.toVector

  private def appendHand(sb: StringBuilder, record: HandRecord, heroName: String, villainName: String): Unit =
    val ts = BaseTimestamp.plusSeconds(record.handNumber.toLong)
    val recordVillainName = if record.villainName.trim.nonEmpty then record.villainName else villainName
    val seatAssignments = Vector(record.heroSeat -> heroName, record.villainSeat -> recordVillainName).sortBy(_._1)
    val bigBlindName = if record.heroIsButton then recordVillainName else heroName
    val smallBlindResolvedName = if record.heroIsButton then heroName else recordVillainName
    // Header
    sb.append(s"PokerStars Hand #${record.handId}: Hold'em No Limit (${money(0.50)}/${money(1.00)}) - ${TimeFmt.format(ts)}\n")
    sb.append(s"Table 'Validation' 2-max Seat #${record.buttonSeat} is the button\n")
    seatAssignments.foreach { case (seatNumber, name) =>
      sb.append(s"Seat $seatNumber: $name (${money(StartingStack)} in chips)\n")
    }
    sb.append(s"$smallBlindResolvedName: posts small blind ${money(0.50)}\n")
    sb.append(s"$bigBlindName: posts big blind ${money(1.00)}\n")

    // Hole cards
    sb.append("*** HOLE CARDS ***\n")
    sb.append(s"Dealt to $heroName [${record.heroCards.first.toToken} ${record.heroCards.second.toToken}]\n")

    // Actions grouped by street
    var currentStreet = Street.Preflop
    var committed = initialCommitted(record, heroName, recordVillainName)
    for action <- record.actions do
      if action.street != currentStreet then
        currentStreet = action.street
        appendStreetHeader(sb, currentStreet, record.board)
        committed = Map(heroName -> 0.0, recordVillainName -> 0.0)
      val actorCommitted = committed.getOrElse(action.player, 0.0)
      appendAction(sb, action, actorCommitted)
      committed = updateCommitted(committed, action)

    // Showdown or summary
    if !record.actions.lastOption.exists(_.action == PokerAction.Fold) then
      sb.append("*** SHOW DOWN ***\n")
      sb.append(s"$heroName: shows [${record.heroCards.first.toToken} ${record.heroCards.second.toToken}]\n")
      sb.append(s"$recordVillainName: shows [${record.villainCards.first.toToken} ${record.villainCards.second.toToken}]\n")

    sb.append("*** SUMMARY ***\n")
    sb.append("\n\n")

  private def appendStreetHeader(sb: StringBuilder, street: Street, board: Board): Unit =
    street match
      case Street.Flop =>
        val cards = board.cards.take(3).map(_.toToken).mkString(" ")
        sb.append(s"*** FLOP *** [$cards]\n")
      case Street.Turn =>
        val flop = board.cards.take(3).map(_.toToken).mkString(" ")
        val turn = board.cards.lift(3).map(_.toToken).getOrElse("??")
        sb.append(s"*** TURN *** [$flop] [$turn]\n")
      case Street.River =>
        val flopTurn = board.cards.take(4).map(_.toToken).mkString(" ")
        val river = board.cards.lift(4).map(_.toToken).getOrElse("??")
        sb.append(s"*** RIVER *** [$flopTurn] [$river]\n")
      case _ => ()

  private def appendAction(sb: StringBuilder, action: RecordedAction, actorCommitted: Double): Unit =
    val name = action.player
    action.action match
      case PokerAction.Fold =>
        sb.append(s"$name: folds\n")
      case PokerAction.Check =>
        sb.append(s"$name: checks\n")
      case PokerAction.Call =>
        sb.append(s"$name: calls ${money(action.toCall)}\n")
      case PokerAction.Raise(amount) =>
        if action.toCall > 0 then
          val totalTo = actorCommitted + amount
          sb.append(s"$name: raises ${money(amount - action.toCall)} to ${money(totalTo)}\n")
        else
          sb.append(s"$name: bets ${money(amount)}\n")

  private def initialCommitted(record: HandRecord, heroName: String, villainName: String): Map[String, Double] =
    if record.heroIsButton then Map(heroName -> 0.5, villainName -> 1.0)
    else Map(heroName -> 1.0, villainName -> 0.5)

  private def updateCommitted(
      committed: Map[String, Double],
      action: RecordedAction
  ): Map[String, Double] =
    val current = committed.getOrElse(action.player, 0.0)
    val updated =
      action.action match
        case PokerAction.Call => current + action.toCall
        case PokerAction.Raise(amount) => current + amount
        case _ => current
    committed.updated(action.player, updated)

  private def money(amount: Double): String =
    String.format(Locale.ROOT, "$%.2f", java.lang.Double.valueOf(amount))
