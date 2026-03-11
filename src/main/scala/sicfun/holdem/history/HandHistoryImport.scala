package sicfun.holdem.history
import sicfun.holdem.types.*

import sicfun.core.Card

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import java.time.format.DateTimeFormatter
import java.time.{Instant, LocalDateTime, ZoneId, ZoneOffset}
import scala.collection.mutable

enum HandHistorySite:
  case PokerStars

object HandHistorySite:
  def parse(raw: String): Either[String, HandHistorySite] =
    raw.trim.toLowerCase match
      case "auto" => Left("site auto-detection must use detect(...)")
      case "pokerstars" | "stars" => Right(HandHistorySite.PokerStars)
      case other => Left(s"unsupported hand-history site: $other")

  def detect(text: String): Either[String, HandHistorySite] =
    val firstNonEmpty = text.linesIterator.map(_.trim).find(_.nonEmpty)
    firstNonEmpty match
      case Some(line) if line.startsWith("PokerStars Hand #") => Right(HandHistorySite.PokerStars)
      case Some(line) => Left(s"could not auto-detect hand-history site from: $line")
      case None => Left("hand-history input is empty")

final case class ImportedPlayer(
    seatIndex: Int,
    seatNumber: Int,
    name: String,
    startingStack: Double,
    position: Position
):
  require(seatIndex >= 0, "seatIndex must be non-negative")
  require(seatNumber > 0, "seatNumber must be positive")
  require(name.trim.nonEmpty, "name must be non-empty")
  require(startingStack >= 0.0, "startingStack must be non-negative")

final case class ImportedHand(
    site: HandHistorySite,
    handId: String,
    tableName: String,
    startedAtEpochMillis: Long,
    buttonSeatNumber: Int,
    players: Vector[ImportedPlayer],
    heroName: Option[String],
    heroHoleCards: Option[HoleCards],
    events: Vector[PokerEvent]
):
  require(handId.trim.nonEmpty, "handId must be non-empty")
  require(tableName.trim.nonEmpty, "tableName must be non-empty")
  require(startedAtEpochMillis >= 0L, "startedAtEpochMillis must be non-negative")
  require(buttonSeatNumber > 0, "buttonSeatNumber must be positive")
  require(players.nonEmpty, "players must be non-empty")

object HandHistoryImport:
  private val HeaderPrefix = "PokerStars Hand #"
  private val TableRegex = """^Table '(.+)' .+ Seat #(\d+) is the button$""".r
  private val SeatRegex = """^Seat (\d+): (.+) \((.+) in chips\)$""".r
  private val BracketGroup = """\[([^\]]+)\]""".r
  private val TimestampFormatter = DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss")
  private val PositionFallbacksByRemaining: Map[Int, Vector[Position]] = Map(
    0 -> Vector.empty,
    1 -> Vector(Position.UTG),
    2 -> Vector(Position.UTG, Position.Cutoff),
    3 -> Vector(Position.UTG, Position.Middle, Position.Cutoff),
    4 -> Vector(Position.UTG, Position.UTG1, Position.Middle, Position.Cutoff),
    5 -> Vector(Position.UTG, Position.UTG1, Position.UTG2, Position.Middle, Position.Cutoff),
    6 -> Vector(Position.UTG, Position.UTG1, Position.UTG2, Position.Middle, Position.Middle, Position.Cutoff)
  )

  def parseFile(
      path: Path,
      site: Option[HandHistorySite] = None,
      heroName: Option[String] = None
  ): Either[String, Vector[ImportedHand]] =
    try
      val text = Files.readString(path, StandardCharsets.UTF_8)
      parseText(text, site, heroName)
    catch
      case e: Exception => Left(s"failed to read hand history file '$path': ${e.getMessage}")

  def parseText(
      text: String,
      site: Option[HandHistorySite],
      heroName: Option[String]
  ): Either[String, Vector[ImportedHand]] =
    val resolvedSiteEither = site match
      case Some(value) => Right(value)
      case None => HandHistorySite.detect(text)
    resolvedSiteEither.flatMap {
      case HandHistorySite.PokerStars => parsePokerStars(text, heroName)
    }

  private def parsePokerStars(
      text: String,
      heroName: Option[String]
  ): Either[String, Vector[ImportedHand]] =
    val blocks = splitPokerStarsHands(text)
    if blocks.isEmpty then Left("no PokerStars hands found in input")
    else
      val parsed = blocks.zipWithIndex.map { case (block, idx) =>
        parsePokerStarsHand(block, idx + 1, heroName)
      }
      parsed.collectFirst { case Left(err) => err } match
        case Some(err) => Left(err)
        case None => Right(parsed.collect { case Right(hand) => hand })

  private def splitPokerStarsHands(text: String): Vector[Vector[String]] =
    val lines = text.linesIterator.map(_.trim).toVector
    val blocks = Vector.newBuilder[Vector[String]]
    val current = mutable.ArrayBuffer.empty[String]
    lines.foreach { line =>
      if line.startsWith(HeaderPrefix) && current.nonEmpty then
        blocks += current.toVector
        current.clear()
      if line.nonEmpty then current += line
    }
    if current.nonEmpty then blocks += current.toVector
    blocks.result()

  private def parsePokerStarsHand(
      lines: Vector[String],
      handOrdinal: Int,
      heroNameHint: Option[String]
  ): Either[String, ImportedHand] =
    try
      val header = lines.headOption.getOrElse(
        throw new IllegalArgumentException("hand block is empty")
      )
      val handId = parseHandId(header)
      val startedAt = parseStartedAt(header, handOrdinal)
      val tableLine = lines.find(_.startsWith("Table ")).getOrElse(
        throw new IllegalArgumentException(s"hand $handId: missing table line")
      )
      val (tableName, buttonSeat) = parseTable(tableLine, handId)
      val seatRows = lines.collect { case SeatRegex(seat, name, stackRaw) =>
        SeatRow(
          seatNumber = seat.toInt,
          name = name.trim,
          startingStack = parseMoney(stackRaw),
          originalLine = s"Seat $seat: $name ($stackRaw in chips)"
        )
      }
      if seatRows.isEmpty then
        throw new IllegalArgumentException(s"hand $handId: no seat lines found")
      val (players, seatIndexByName) = buildPlayers(handId, buttonSeat, seatRows)
      val state = new PokerStarsState(
        handId = handId,
        startedAtEpochMillis = startedAt,
        players = players,
        seatIndexByName = seatIndexByName
      )
      val heroFromCards = parseHeroFromDealtLine(lines)
      val heroName = heroNameHint.orElse(heroFromCards.map(_._1))
      val heroCards = heroFromCards.map(_._2)

      var summaryReached = false
      lines.tail.foreach { line =>
        if !summaryReached then
          if line == "*** SUMMARY ***" then summaryReached = true
          else if line.startsWith("*** HOLE CARDS ***") then state.beginStreet(Street.Preflop, Board.empty)
          else if line.startsWith("*** FLOP ***") then state.beginStreet(Street.Flop, parseBoardFromStreetLine(line, handId))
          else if line.startsWith("*** TURN ***") then state.beginStreet(Street.Turn, parseBoardFromStreetLine(line, handId))
          else if line.startsWith("*** RIVER ***") then state.beginStreet(Street.River, parseBoardFromStreetLine(line, handId))
          else state.consumeLine(line)
      }

      Right(
        ImportedHand(
          site = HandHistorySite.PokerStars,
          handId = handId,
          tableName = tableName,
          startedAtEpochMillis = startedAt,
          buttonSeatNumber = buttonSeat,
          players = players,
          heroName = heroName,
          heroHoleCards = heroCards,
          events = state.events
        )
      )
    catch
      case e: Exception => Left(s"failed to parse PokerStars hand #$handOrdinal: ${e.getMessage}")

  private final case class SeatRow(
      seatNumber: Int,
      name: String,
      startingStack: Double,
      originalLine: String
  )

  private final class PokerStarsState(
      handId: String,
      startedAtEpochMillis: Long,
      players: Vector[ImportedPlayer],
      seatIndexByName: Map[String, Int]
  ):
    private val playerByName = players.map(p => p.name -> p).toMap
    private val currentStacks = mutable.Map.from(players.map(p => p.name -> p.startingStack))
    private val committedThisStreet = mutable.Map.from(players.map(p => p.name -> 0.0))
    private var street = Street.Preflop
    private var board = Board.empty
    private var pot = 0.0
    private var sequenceInHand = 0L
    private var betHistory = Vector.empty[BetAction]
    private val builtEvents = Vector.newBuilder[PokerEvent]

    def events: Vector[PokerEvent] = builtEvents.result()

    def beginStreet(nextStreet: Street, nextBoard: Board): Unit =
      street = nextStreet
      board = nextBoard
      if nextStreet != Street.Preflop then
        committedThisStreet.keys.foreach(name => committedThisStreet.update(name, 0.0))

    def consumeLine(line: String): Unit =
      parseBlindPost(line)
        .orElse(parseAntePost(line))
        .orElse(parseAction(line))
        .getOrElse(())

    private def parseBlindPost(line: String): Option[Unit] =
      parsePlayerSuffix(line).flatMap { case (playerName, suffix) =>
        if suffix.startsWith("posts small blind ") then
          val amount = parseLeadingAmount(suffix.drop("posts small blind ".length))
          postForcedBet(playerName, amount)
          Some(())
        else if suffix.startsWith("posts big blind ") then
          val amount = parseLeadingAmount(suffix.drop("posts big blind ".length))
          postForcedBet(playerName, amount)
          Some(())
        else None
      }

    private def parseAntePost(line: String): Option[Unit] =
      parsePlayerSuffix(line).flatMap { case (playerName, suffix) =>
        if suffix.startsWith("posts the ante ") then
          val amount = parseLeadingAmount(suffix.drop("posts the ante ".length))
          val stack = currentStacks.getOrElse(playerName, missingPlayer(playerName))
          currentStacks.update(playerName, math.max(0.0, stack - amount))
          pot += amount
          Some(())
        else None
      }

    private def parseAction(line: String): Option[Unit] =
      parsePlayerSuffix(line).flatMap { case (playerName, suffix) =>
        if !playerByName.contains(playerName) then None
        else if suffix.startsWith("folds") then
          appendEvent(playerName, PokerAction.Fold)
          Some(())
        else if suffix.startsWith("checks") then
          appendEvent(playerName, PokerAction.Check)
          applyCheck(playerName)
          Some(())
        else if suffix.startsWith("calls ") then
          val amount = parseLeadingAmount(suffix.drop("calls ".length))
          appendEvent(playerName, PokerAction.Call)
          applyCall(playerName, amount)
          Some(())
        else if suffix.startsWith("bets ") then
          val amount = parseLeadingAmount(suffix.drop("bets ".length))
          val totalAmount = roundChips(committedThisStreet(playerName) + amount)
          appendEvent(playerName, PokerAction.Raise(totalAmount))
          applyRaise(playerName, totalAmount)
          Some(())
        else if suffix.startsWith("raises ") then
          val toMarker = " to "
          val toIdx = suffix.indexOf(toMarker)
          if toIdx < 0 then
            throw new IllegalArgumentException(s"hand $handId: unsupported raise format '$line'")
          val toAmount = parseLeadingAmount(suffix.drop(toIdx + toMarker.length))
          val totalAmount = roundChips(toAmount)
          appendEvent(playerName, PokerAction.Raise(totalAmount))
          applyRaise(playerName, totalAmount)
          Some(())
        else None
      }

    private def postForcedBet(playerName: String, amount: Double): Unit =
      val stack = currentStacks.getOrElse(playerName, missingPlayer(playerName))
      currentStacks.update(playerName, math.max(0.0, stack - amount))
      committedThisStreet.update(playerName, committedThisStreet(playerName) + amount)
      pot += amount

    private def applyCheck(playerName: String): Unit =
      betHistory = betHistory :+ BetAction(seatIndexByName(playerName), PokerAction.Check)

    private def applyCall(playerName: String, callAmount: Double): Unit =
      val stack = currentStacks.getOrElse(playerName, missingPlayer(playerName))
      val actual = math.min(callAmount, stack)
      currentStacks.update(playerName, math.max(0.0, stack - actual))
      committedThisStreet.update(playerName, committedThisStreet(playerName) + actual)
      pot += actual
      betHistory = betHistory :+ BetAction(seatIndexByName(playerName), PokerAction.Call)

    private def applyRaise(playerName: String, totalAmount: Double): Unit =
      val stack = currentStacks.getOrElse(playerName, missingPlayer(playerName))
      val alreadyIn = committedThisStreet(playerName)
      val additional = math.max(0.0, totalAmount - alreadyIn)
      val actual = math.min(additional, stack)
      currentStacks.update(playerName, math.max(0.0, stack - actual))
      committedThisStreet.update(playerName, alreadyIn + actual)
      pot += actual
      betHistory = betHistory :+ BetAction(seatIndexByName(playerName), PokerAction.Raise(totalAmount))

    private def appendEvent(playerName: String, action: PokerAction): Unit =
      val player = playerByName.getOrElse(playerName, missingPlayer(playerName))
      val toCall = currentToCall(playerName)
      val stackBefore = currentStacks.getOrElse(playerName, missingPlayer(playerName))
      builtEvents += PokerEvent(
        handId = handId,
        sequenceInHand = sequenceInHand,
        playerId = playerName,
        occurredAtEpochMillis = startedAtEpochMillis + sequenceInHand,
        street = street,
        position = player.position,
        board = board,
        potBefore = roundChips(pot),
        toCall = roundChips(toCall),
        stackBefore = roundChips(stackBefore),
        action = action,
        betHistory = betHistory
      )
      sequenceInHand += 1L

    private def currentToCall(playerName: String): Double =
      val currentMax = committedThisStreet.values.maxOption.getOrElse(0.0)
      math.max(0.0, currentMax - committedThisStreet(playerName))

    private def missingPlayer(playerName: String) =
      throw new IllegalArgumentException(s"hand $handId: unknown player '$playerName'")

  private def parseHandId(header: String): String =
    val hashIdx = header.indexOf('#')
    val colonIdx = header.indexOf(':')
    if hashIdx < 0 || colonIdx < 0 || colonIdx <= hashIdx then
      throw new IllegalArgumentException(s"invalid PokerStars header: $header")
    header.substring(hashIdx + 1, colonIdx).trim

  private def parseStartedAt(header: String, handOrdinal: Int): Long =
    val marker = " - "
    val idx = header.lastIndexOf(marker)
    if idx < 0 then handOrdinal.toLong
    else
      val raw = header.substring(idx + marker.length).trim
      parseTimestamp(raw).getOrElse(handOrdinal.toLong)

  private def parseTimestamp(raw: String): Option[Long] =
    val parts = raw.split("\\s+").toVector
    if parts.length < 2 then None
    else
      val zoneToken = parts.drop(2).headOption.getOrElse("UTC")
      val localText = parts.take(2).mkString(" ")
      val zone = zoneFromToken(zoneToken)
      try
        Some(LocalDateTime.parse(localText, TimestampFormatter).atZone(zone).toInstant.toEpochMilli)
      catch
        case _: Exception =>
          try Some(Instant.parse(raw).toEpochMilli)
          catch case _: Exception => None

  private def zoneFromToken(token: String): ZoneId =
    token.trim.toUpperCase match
      case "ET" | "EST" | "EDT" => ZoneId.of("America/New_York")
      case "CT" | "CST" | "CDT" => ZoneId.of("America/Chicago")
      case "MT" | "MST" | "MDT" => ZoneId.of("America/Denver")
      case "PT" | "PST" | "PDT" => ZoneId.of("America/Los_Angeles")
      case "CET" => ZoneId.of("Europe/Paris")
      case "UTC" | "GMT" | "WET" => ZoneOffset.UTC
      case _ => ZoneOffset.UTC

  private def parseTable(
      line: String,
      handId: String
  ): (String, Int) =
    line match
      case TableRegex(name, buttonSeat) => (name.trim, buttonSeat.toInt)
      case _ => throw new IllegalArgumentException(s"hand $handId: invalid table line '$line'")

  private def buildPlayers(
      handId: String,
      buttonSeatNumber: Int,
      seatRows: Vector[SeatRow]
  ): (Vector[ImportedPlayer], Map[String, Int]) =
    val seatByNumber = seatRows.map(row => row.seatNumber -> row).toMap
    if !seatByNumber.contains(buttonSeatNumber) then
      throw new IllegalArgumentException(s"hand $handId: button seat $buttonSeatNumber has no matching seat row")
    val occupiedSeats = seatRows.map(_.seatNumber).sorted
    val buttonIdx = occupiedSeats.indexOf(buttonSeatNumber)
    if buttonIdx < 0 then
      throw new IllegalArgumentException(s"hand $handId: button seat $buttonSeatNumber is not occupied")
    val clockwiseSeats = Vector.tabulate(occupiedSeats.length) { idx =>
      occupiedSeats((buttonIdx + idx) % occupiedSeats.length)
    }
    val positionBySeat = positionsForSeats(clockwiseSeats)
    val players = clockwiseSeats.zipWithIndex.map { case (seatNumber, seatIndex) =>
      val row = seatByNumber(seatNumber)
      ImportedPlayer(
        seatIndex = seatIndex,
        seatNumber = seatNumber,
        name = row.name,
        startingStack = roundChips(row.startingStack),
        position = positionBySeat(seatNumber)
      )
    }
    (players, players.map(p => p.name -> p.seatIndex).toMap)

  private def positionsForSeats(
      clockwiseSeatsFromButton: Vector[Int]
  ): Map[Int, Position] =
    val count = clockwiseSeatsFromButton.length
    if count < 2 then
      throw new IllegalArgumentException("at least two seated players are required")
    else if count == 2 then
      Map(
        clockwiseSeatsFromButton(0) -> Position.SmallBlind,
        clockwiseSeatsFromButton(1) -> Position.BigBlind
      )
    else
      val buttonSeat = clockwiseSeatsFromButton(0)
      val smallBlindSeat = clockwiseSeatsFromButton(1)
      val bigBlindSeat = clockwiseSeatsFromButton(2)
      val remaining = clockwiseSeatsFromButton.drop(3)
      val remainingLabels = PositionFallbacksByRemaining.getOrElse(
        remaining.length,
        throw new IllegalArgumentException(s"unsupported player count: ${clockwiseSeatsFromButton.length}")
      )
      val base = mutable.Map(
        buttonSeat -> Position.Button,
        smallBlindSeat -> Position.SmallBlind,
        bigBlindSeat -> Position.BigBlind
      )
      remaining.zip(remainingLabels).foreach { case (seat, pos) => base.update(seat, pos) }
      base.toMap

  private def parseHeroFromDealtLine(
      lines: Vector[String]
  ): Option[(String, HoleCards)] =
    lines.collectFirst(Function.unlift(parseDealtLine))

  private def parseDealtLine(line: String): Option[(String, HoleCards)] =
    val prefix = "Dealt to "
    if !line.startsWith(prefix) then None
    else
      val bracketIdx = line.indexOf('[')
      if bracketIdx < 0 then None
      else
        val playerName = line.substring(prefix.length, bracketIdx).trim
        val groups = BracketGroup.findAllMatchIn(line).toVector
        groups.lastOption.map { group =>
          val cards = parseCardTokens(group.group(1))
          playerName -> HoleCards.from(cards)
        }

  private def parseBoardFromStreetLine(
      line: String,
      handId: String
  ): Board =
    val cards = BracketGroup.findAllMatchIn(line).toVector.flatMap(group => parseCardTokens(group.group(1)))
    try Board.from(cards)
    catch
      case e: Exception => throw new IllegalArgumentException(s"hand $handId: ${e.getMessage}")

  private def parseCardTokens(raw: String): Vector[Card] =
    raw.split("\\s+").toVector.filter(_.nonEmpty).map { token =>
      Card.parse(token).getOrElse(
        throw new IllegalArgumentException(s"invalid card token '$token'")
      )
    }

  private def parsePlayerSuffix(line: String): Option[(String, String)] =
    val idx = line.indexOf(": ")
    if idx <= 0 then None
    else Some(line.substring(0, idx).trim -> line.substring(idx + 2).trim)

  private def parseLeadingAmount(raw: String): Double =
    val token = raw.takeWhile(!_.isWhitespace)
    parseMoney(token)

  private def parseMoney(raw: String): Double =
    val cleaned = raw
      .replace(",", "")
      .replaceAll("[^0-9.\\-]", "")
      .trim
    if cleaned.isEmpty then
      throw new IllegalArgumentException(s"invalid money token '$raw'")
    val amount = cleaned.toDouble
    if !amount.isFinite || amount < 0.0 then
      throw new IllegalArgumentException(s"invalid money amount '$raw'")
    roundChips(amount)

  private def roundChips(amount: Double): Double =
    math.round(amount * 100.0) / 100.0
