package sicfun.holdem.history
import sicfun.holdem.types.*

import sicfun.core.Card

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import java.time.format.DateTimeFormatter
import java.time.{Instant, LocalDateTime, ZoneId, ZoneOffset}
import scala.collection.mutable

/** Supported external hand-history sources for import normalization.
  *
  * Each site has its own text format for hand histories. The [[HandHistoryImport]]
  * parser handles the site-specific variations and normalizes them into a common
  * [[ImportedHand]] representation. Auto-detection inspects the first non-empty line
  * to identify the site.
  */
enum HandHistorySite:
  case PokerStars
  case Winamax
  case GGPoker

object HandHistorySite:
  /** Parse a site name string into a HandHistorySite enum value.
    * Accepts common aliases (e.g., "stars" for PokerStars, "gg" for GGPoker).
    */
  def parse(raw: String): Either[String, HandHistorySite] =
    raw.trim.toLowerCase match
      case "auto" => Left("site auto-detection must use detect(...)")
      case "pokerstars" | "stars" => Right(HandHistorySite.PokerStars)
      case "winamax" | "wina" => Right(HandHistorySite.Winamax)
      case "ggpoker" | "gg" | "ggnetwork" => Right(HandHistorySite.GGPoker)
      case other => Left(s"unsupported hand-history site: $other")

  /** Auto-detect the hand history site from the first non-empty line of the text.
    *
    * Inspects known header prefixes: "PokerStars Hand #", "Winamax Poker -",
    * "Poker Hand #" / "GGPoker Hand #" / "#Game No :".
    */
  def detect(text: String): Either[String, HandHistorySite] =
    val firstNonEmpty = text.linesIterator.map(_.trim.stripPrefix("\uFEFF")).find(_.nonEmpty)
    firstNonEmpty match
      case Some(line) if line.startsWith("PokerStars Hand #") => Right(HandHistorySite.PokerStars)
      case Some(line) if line.startsWith("Winamax Poker -") => Right(HandHistorySite.Winamax)
      case Some(line)
          if line.startsWith("Poker Hand #") ||
            line.startsWith("Hand #") ||
            line.startsWith("GGPoker Hand #") ||
            line.startsWith("#Game No :") =>
        Right(HandHistorySite.GGPoker)
      case Some(line) => Left(s"could not auto-detect hand-history site from: $line")
      case None => Left("hand-history input is empty")

/** A player parsed from a hand history, with seat assignment and position.
  *
  * @param seatIndex     0-based index used internally for ordering
  * @param seatNumber    1-based seat number as shown in the hand history
  * @param name          player's screen name (normalized, forum suffixes stripped)
  * @param startingStack starting chip stack at the beginning of the hand
  * @param position      table position (Button, BigBlind, etc.) assigned during parsing
  */
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

/** A complete hand parsed and normalized from a site-specific hand history.
  *
  * Contains all information needed to replay the hand: players with positions,
  * hero identity and hole cards, the full event sequence, and showdown reveals.
  * Used by [[OpponentProfile.fromImportedHands]] to build opponent profiles.
  *
  * @param site                 the source site (PokerStars, Winamax, GGPoker)
  * @param handId               unique hand identifier from the site
  * @param tableName            table name from the hand history
  * @param startedAtEpochMillis timestamp when the hand started (epoch millis)
  * @param buttonSeatNumber     seat number of the button/dealer
  * @param players              all players at the table with positions
  * @param heroName             hero's screen name (if identified)
  * @param heroHoleCards        hero's hole cards (if dealt and visible)
  * @param events               normalized poker event sequence
  * @param showdownCards        map of player name -> hole cards revealed at showdown
  */
final case class ImportedHand(
    site: HandHistorySite,
    handId: String,
    tableName: String,
    startedAtEpochMillis: Long,
    buttonSeatNumber: Int,
    players: Vector[ImportedPlayer],
    heroName: Option[String],
    heroHoleCards: Option[HoleCards],
    events: Vector[PokerEvent],
    showdownCards: Map[String, HoleCards] = Map.empty
):
  require(handId.trim.nonEmpty, "handId must be non-empty")
  require(tableName.trim.nonEmpty, "tableName must be non-empty")
  require(startedAtEpochMillis >= 0L, "startedAtEpochMillis must be non-negative")
  require(buttonSeatNumber > 0, "buttonSeatNumber must be positive")
  require(players.nonEmpty, "players must be non-empty")

/** Parses and normalizes raw hand-history text into SICFUN canonical hand/event structures.
  *
  * The importer handles three poker sites (PokerStars, Winamax, GGPoker), each with
  * their own text format. The parsing strategy:
  *   1. Split the raw text into per-hand blocks based on site-specific header lines
  *   2. For each block, extract hand metadata (ID, timestamp, table, button seat)
  *   3. Parse seat lines to build the player list
  *   4. Assign table positions based on seat order relative to the button
  *   5. Parse action lines into normalized PokerEvent instances
  *   6. Extract hero hole cards and showdown reveals
  *
  * Design decisions:
  *   - Forum hero aliases like "PLAYERNAME(HERO)" are normalized by stripping the suffix
  *   - Money amounts are parsed tolerantly (handles $, EUR, commas, periods)
  *   - Position assignment uses a fallback table for non-standard player counts
  *   - Timestamp parsing supports multiple date formats and timezone indicators
  *   - BOM characters are stripped from the input
  *
  * @see [[ImportedHand]] for the output structure
  * @see [[OpponentProfile.fromImportedHands]] for the downstream consumer
  */
object HandHistoryImport:
  private val PokerStarsHeaderPrefix = "PokerStars Hand #"
  private val WinamaxHeaderPrefix = "Winamax Poker -"
  private val GGPokerHeaderPrefixes = Vector("Poker Hand #", "Hand #", "GGPoker Hand #", "#Game No :")
  private val ForumHeroSuffixRegex = """(?i)^(.*?)(?:\s*)\(hero\)$""".r
  private val TableRegexes = Vector(
    """^Table '(.+)' .+ Seat #(\d+) is the button$""".r,
    """^Table: '(.+)' .+ Seat #(\d+) is the button$""".r,
    """^Table '(.+)' .+ Dealer is seat #(\d+)$""".r,
    """^Table: '(.+)' .+ Dealer is seat #(\d+)$""".r
  )
  private val SeatRegexes = Vector(
    """^Seat (\d+): (.+) \((.+) in chips\)$""".r,
    """^Seat (\d+): (.+) \((.+)\)$""".r
  )
  private val BracketGroup = """\[([^\]]+)\]""".r
  private val TimestampFormatters = Vector(
    DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss"),
    DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")
  )
  private val TimestampSearchRegexes = Vector(
    """(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})(?: ([A-Za-z]{2,4}))?""".r,
    """(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?: ([A-Za-z]{2,4}))?""".r,
    """(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+\-]\d{2}:?\d{2})?)""".r
  )
  private val WinamaxHandIdRegex = """.*HandId:\s*#([^ ]+).*""".r
  private val GGPokerHandIdRegexes = Vector(
    """^Poker Hand #([^:]+):.*$""".r,
    """^GGPoker Hand #([^:]+):.*$""".r,
    """^#Game No :\s*([^\s]+).*$""".r,
    """^Hand #([^:]+):.*$""".r
  )
  private val HashTokenRegex = """#([A-Za-z0-9\-]+)""".r
  private val LeadingAmountRegex = """^\s*([$€£¥]?\s*-?\d[\d\s.,]*[$€£¥]?)""".r
  private val ShowdownPattern = """(?i)^(.+?)(?::)?\s+(?:shows?|showed)\s+\[([^\]]+)\].*""".r
  private val PlayerActionTokens = Vector(
    "posts small blind ",
    "posts big blind ",
    "posts the ante ",
    "posts ante ",
    "folds",
    "checks",
    "calls ",
    "bets ",
    "raises "
  )
  private val PositionFallbacksByRemaining: Map[Int, Vector[Position]] = Map(
    0 -> Vector.empty,
    1 -> Vector(Position.UTG),
    2 -> Vector(Position.UTG, Position.Cutoff),
    3 -> Vector(Position.UTG, Position.Middle, Position.Cutoff),
    4 -> Vector(Position.UTG, Position.UTG1, Position.Middle, Position.Cutoff),
    5 -> Vector(Position.UTG, Position.UTG1, Position.UTG2, Position.Middle, Position.Cutoff),
    6 -> Vector(Position.UTG, Position.UTG1, Position.UTG2, Position.Middle, Position.Hijack, Position.Cutoff)
  )

  /** Parse a hand history file from disk.
    *
    * @param path     path to the hand history text file (UTF-8)
    * @param site     optional site hint; if None, auto-detects from the file content
    * @param heroName optional hero screen name for hole card extraction
    * @return Right(hands) on success, Left(error) on failure
    */
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

  /** Parse hand history from a raw text string.
    *
    * Normalizes the hero name (strips forum suffixes), resolves or auto-detects
    * the site, then dispatches to the site-specific parser.
    *
    * @param text     raw hand history text (may contain multiple hands)
    * @param site     optional site hint; if None, auto-detects from the text
    * @param heroName optional hero screen name (forum aliases like "NAME(HERO)" accepted)
    * @return Right(hands) on success, Left(error) on failure
    */
  def parseText(
      text: String,
      site: Option[HandHistorySite],
      heroName: Option[String]
  ): Either[String, Vector[ImportedHand]] =
    val normalizedHeroName = heroName.map(normalizePlayerName).filter(_.nonEmpty)
    val resolvedSiteEither = site match
      case Some(value) => Right(value)
      case None => HandHistorySite.detect(text)
    resolvedSiteEither.flatMap {
      case HandHistorySite.PokerStars => parsePokerStars(text, normalizedHeroName)
      case HandHistorySite.Winamax => parseWinamax(text, normalizedHeroName)
      case HandHistorySite.GGPoker => parseGGPoker(text, normalizedHeroName)
    }

  /** Normalize a player name by stripping common forum hero suffixes like "(HERO)". */
  def normalizePlayerName(raw: String): String =
    raw.trim match
      case ForumHeroSuffixRegex(base) if base.trim.nonEmpty => base.trim
      case other => other

  private def parsePokerStars(
      text: String,
      heroName: Option[String]
  ): Either[String, Vector[ImportedHand]] =
    parseSite(text, heroName, "PokerStars", _.startsWith(PokerStarsHeaderPrefix))(
      parsePokerStarsHand
    )

  private def parseWinamax(
      text: String,
      heroName: Option[String]
  ): Either[String, Vector[ImportedHand]] =
    parseSite(text, heroName, "Winamax", _.startsWith(WinamaxHeaderPrefix))(
      parseWinamaxHand
    )

  private def parseGGPoker(
      text: String,
      heroName: Option[String]
  ): Either[String, Vector[ImportedHand]] =
    parseSite(text, heroName, "GGPoker", isGGPokerHeaderLine)(
      parseGGPokerHand
    )

  private def parseSite(
      text: String,
      heroName: Option[String],
      siteLabel: String,
      isHeaderLine: String => Boolean
  )(
      parseHand: (Vector[String], Int, Option[String]) => Either[String, ImportedHand]
  ): Either[String, Vector[ImportedHand]] =
    val blocks = splitHands(text, isHeaderLine)
    if blocks.isEmpty then Left(s"no $siteLabel hands found in input")
    else
      val parsed = blocks.zipWithIndex.map { case (block, idx) =>
        parseHand(block, idx + 1, heroName)
      }
      parsed.collectFirst { case Left(err) => err } match
        case Some(err) => Left(err)
        case None => Right(parsed.collect { case Right(hand) => hand })

  private def splitHands(
      text: String,
      isHeaderLine: String => Boolean
  ): Vector[Vector[String]] =
    val lines = text.linesIterator.map(_.trim.stripPrefix("\uFEFF")).toVector
    val blocks = Vector.newBuilder[Vector[String]]
    val current = mutable.ArrayBuffer.empty[String]
    lines.foreach { line =>
      if isHeaderLine(line) && current.nonEmpty then
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
    parseGenericHand(
      lines = lines,
      handOrdinal = handOrdinal,
      heroNameHint = heroNameHint,
      site = HandHistorySite.PokerStars,
      siteLabel = "PokerStars",
      parseHandId = parsePokerStarsHandId
    )

  private def parseWinamaxHand(
      lines: Vector[String],
      handOrdinal: Int,
      heroNameHint: Option[String]
  ): Either[String, ImportedHand] =
    parseGenericHand(
      lines = lines,
      handOrdinal = handOrdinal,
      heroNameHint = heroNameHint,
      site = HandHistorySite.Winamax,
      siteLabel = "Winamax",
      parseHandId = parseWinamaxHandId
    )

  private def parseGGPokerHand(
      lines: Vector[String],
      handOrdinal: Int,
      heroNameHint: Option[String]
  ): Either[String, ImportedHand] =
    parseGenericHand(
      lines = lines,
      handOrdinal = handOrdinal,
      heroNameHint = heroNameHint,
      site = HandHistorySite.GGPoker,
      siteLabel = "GGPoker",
      parseHandId = parseGGPokerHandId
    )

  private def parseGenericHand(
      lines: Vector[String],
      handOrdinal: Int,
      heroNameHint: Option[String],
      site: HandHistorySite,
      siteLabel: String,
      parseHandId: String => String
  ): Either[String, ImportedHand] =
    try
      val header = lines.headOption.getOrElse(
        throw new IllegalArgumentException("hand block is empty")
      )
      val handId = parseHandId(header)
      val startedAt = parseStartedAt(header, handOrdinal)
      val preamble = lines.takeWhile(line => !line.startsWith("*** "))
      val tableLine = preamble.find(isTableLine).getOrElse(
        throw new IllegalArgumentException(s"hand $handId: missing table line")
      )
      val (tableName, buttonSeat) = parseTable(tableLine, handId)
      val seatRows = preamble.flatMap(parseSeatRow)
      if seatRows.isEmpty then
        throw new IllegalArgumentException(s"hand $handId: no seat lines found")
      val (players, seatIndexByName) = buildPlayers(handId, buttonSeat, seatRows)
      val state = new ImportState(
        handId = handId,
        startedAtEpochMillis = startedAt,
        players = players,
        seatIndexByName = seatIndexByName
      )
      val heroFromCards = parseHeroFromDealtLine(lines)
      val heroName = heroNameHint.orElse(heroFromCards.map(_._1))
      val heroCards = heroFromCards.map(_._2)

      var summaryReached = false
      var showdownReached = false
      val showdownMap = mutable.Map.empty[String, HoleCards]
      lines.tail.foreach { line =>
        if !summaryReached then
          val upper = line.toUpperCase
          if upper == "*** SUMMARY ***" then summaryReached = true
          else if upper == "*** SHOW DOWN ***" then showdownReached = true
          else if showdownReached then
            line match
              case ShowdownPattern(name, cardsStr) =>
                val cards = cardsStr.trim.split("\\s+").flatMap(Card.parse)
                if cards.length == 2 then
                  showdownMap += (name.trim -> HoleCards.from(cards.toIndexedSeq))
              case _ => ()
          else if upper.startsWith("*** HOLE CARDS ***") || upper.startsWith("*** PRE-FLOP ***") || upper.startsWith("*** ANTE/BLINDS ***") then
            state.beginStreet(Street.Preflop, Board.empty)
          else if upper.startsWith("*** FLOP ***") then
            state.beginStreet(Street.Flop, parseBoardFromStreetLine(line, handId))
          else if upper.startsWith("*** TURN ***") then
            state.beginStreet(Street.Turn, parseBoardFromStreetLine(line, handId))
          else if upper.startsWith("*** RIVER ***") then
            state.beginStreet(Street.River, parseBoardFromStreetLine(line, handId))
          else state.consumeLine(line)
      }

      Right(
        ImportedHand(
          site = site,
          handId = handId,
          tableName = tableName,
          startedAtEpochMillis = startedAt,
          buttonSeatNumber = buttonSeat,
          players = players,
          heroName = heroName,
          heroHoleCards = heroCards,
          events = state.events,
          showdownCards = showdownMap.toMap
        )
      )
    catch
      case e: Exception => Left(s"failed to parse $siteLabel hand #$handOrdinal: ${e.getMessage}")

  private final case class SeatRow(
      seatNumber: Int,
      name: String,
      startingStack: Double,
      originalLine: String
  )

  private final class ImportState(
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
        val lower = suffix.toLowerCase
        if lower.startsWith("posts small blind ") then
          val amount = parseLeadingAmount(suffix.drop("posts small blind ".length))
          postForcedBet(playerName, amount)
          Some(())
        else if lower.startsWith("posts big blind ") then
          val amount = parseLeadingAmount(suffix.drop("posts big blind ".length))
          postForcedBet(playerName, amount)
          Some(())
        else None
      }

    private def parseAntePost(line: String): Option[Unit] =
      parsePlayerSuffix(line).flatMap { case (playerName, suffix) =>
        val lower = suffix.toLowerCase
        if lower.startsWith("posts the ante ") then
          val amount = parseLeadingAmount(suffix.drop("posts the ante ".length))
          val stack = currentStacks.getOrElse(playerName, missingPlayer(playerName))
          currentStacks.update(playerName, math.max(0.0, stack - amount))
          pot += amount
          Some(())
        else if lower.startsWith("posts ante ") then
          val amount = parseLeadingAmount(suffix.drop("posts ante ".length))
          val stack = currentStacks.getOrElse(playerName, missingPlayer(playerName))
          currentStacks.update(playerName, math.max(0.0, stack - amount))
          pot += amount
          Some(())
        else None
      }

    private def parseAction(line: String): Option[Unit] =
      parsePlayerSuffix(line).flatMap { case (playerName, suffix) =>
        val lower = suffix.toLowerCase
        if !playerByName.contains(playerName) then None
        else if lower.startsWith("folds") then
          appendEvent(playerName, PokerAction.Fold)
          Some(())
        else if lower.startsWith("checks") then
          appendEvent(playerName, PokerAction.Check)
          applyCheck(playerName)
          Some(())
        else if lower.startsWith("calls ") then
          val amount = parseLeadingAmount(suffix.drop("calls ".length))
          appendEvent(playerName, PokerAction.Call)
          applyCall(playerName, amount)
          Some(())
        else if lower.startsWith("bets ") then
          val amount = parseLeadingAmount(suffix.drop("bets ".length))
          val totalAmount = roundChips(committedThisStreet(playerName) + amount)
          appendEvent(playerName, PokerAction.Raise(totalAmount))
          applyRaise(playerName, totalAmount)
          Some(())
        else if lower.startsWith("raises to ") then
          val toAmount = parseLeadingAmount(suffix.drop("raises to ".length))
          val totalAmount = roundChips(toAmount)
          appendEvent(playerName, PokerAction.Raise(totalAmount))
          applyRaise(playerName, totalAmount)
          Some(())
        else if lower.startsWith("raises ") then
          val toMarker = " to "
          val toIdx = lower.indexOf(toMarker)
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

  private def parsePokerStarsHandId(header: String): String =
    parseHandId(header)

  private def parseWinamaxHandId(header: String): String =
    header match
      case WinamaxHandIdRegex(handId) => handId.trim
      case _ =>
        extractHashToken(header).getOrElse(
          throw new IllegalArgumentException(s"invalid Winamax header: $header")
        )

  private def parseGGPokerHandId(header: String): String =
    GGPokerHandIdRegexes.iterator.collectFirst(Function.unlift { regex =>
      header match
        case regex(handId) => Some(handId.trim)
        case _ => None
    }).orElse(extractHashToken(header)).getOrElse(
      throw new IllegalArgumentException(s"invalid GGPoker header: $header")
    )

  private def extractHashToken(text: String): Option[String] =
    HashTokenRegex.findFirstMatchIn(text).map(_.group(1).trim)

  private def parseStartedAt(header: String, handOrdinal: Int): Long =
    extractTimestamp(header)
      .flatMap(parseTimestamp)
      .getOrElse(handOrdinal.toLong)

  private def extractTimestamp(header: String): Option[String] =
    TimestampSearchRegexes.iterator.flatMap { regex =>
      regex.findFirstMatchIn(header).map { m =>
        val zoneToken =
          if m.groupCount >= 2 then Option(m.group(2)).map(_.trim).filter(_.nonEmpty)
          else None
        zoneToken match
          case Some(zone) => s"${m.group(1).trim} $zone"
          case None => m.group(1).trim
      }
    }.take(1).toVector.headOption

  private def parseTimestamp(raw: String): Option[Long] =
    val trimmed = raw.trim
    val parts = trimmed.split("\\s+").toVector
    if parts.length >= 2 && parts(0).matches("""\d{4}[/-]\d{2}[/-]\d{2}""") then
      val zoneToken = parts.drop(2).headOption.getOrElse("UTC")
      val localText = parts.take(2).mkString(" ")
      val zone = zoneFromToken(zoneToken)
      TimestampFormatters.iterator.flatMap { formatter =>
        try Some(LocalDateTime.parse(localText, formatter).atZone(zone).toInstant.toEpochMilli)
        catch case _: Exception => None
      }.take(1).toVector.headOption
    else
      try Some(Instant.parse(trimmed).toEpochMilli)
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
    TableRegexes.iterator.collectFirst(Function.unlift { regex =>
      line match
        case regex(name, buttonSeat) => Some(name.trim -> buttonSeat.toInt)
        case _ => None
    }).getOrElse(
      throw new IllegalArgumentException(s"hand $handId: invalid table line '$line'")
    )

  private def isTableLine(line: String): Boolean =
    line.startsWith("Table ") || line.startsWith("Table:")

  private def parseSeatRow(line: String): Option[SeatRow] =
    SeatRegexes.iterator.collectFirst(Function.unlift { regex =>
      line match
        case regex(seat, name, stackRaw) =>
          Some(
            SeatRow(
              seatNumber = seat.toInt,
              name = normalizePlayerName(name),
              startingStack = parseMoney(stackRaw),
              originalLine = line
            )
          )
        case _ => None
    })

  private def isGGPokerHeaderLine(line: String): Boolean =
    GGPokerHeaderPrefixes.exists(prefix => line.startsWith(prefix))

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
        clockwiseSeatsFromButton(0) -> Position.Button,
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
        val playerName = normalizePlayerName(line.substring(prefix.length, bracketIdx).trim)
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
    if idx > 0 then Some(normalizePlayerName(line.substring(0, idx).trim) -> line.substring(idx + 2).trim)
    else
      val lowered = line.toLowerCase
      PlayerActionTokens.iterator
        .flatMap { token =>
          val tokenIdx = lowered.indexOf(token)
          if tokenIdx > 0 then Some(tokenIdx) else None
        }
        .minOption
        .map { tokenIdx =>
          normalizePlayerName(line.substring(0, tokenIdx).trim) -> line.substring(tokenIdx).trim
        }

  private def parseLeadingAmount(raw: String): Double =
    raw match
      case LeadingAmountRegex(token) => parseMoney(token)
      case _ => parseMoney(raw)

  private def parseMoney(raw: String): Double =
    val usesEuroFormatting = raw.contains('€')
    val normalized = raw
      .trim
      .replace('\u00A0', ' ')
      .replace(" ", "")
      .replaceAll("[^0-9,\\.\\-]", "")
      .trim
    if normalized.isEmpty then
      throw new IllegalArgumentException(s"invalid money token '$raw'")
    val cleaned =
      if normalized.contains(',') && normalized.contains('.') then
        if normalized.lastIndexOf(',') > normalized.lastIndexOf('.') then
          normalized.replace(".", "").replace(',', '.')
        else normalized.replace(",", "")
      else if normalized.contains(',') then
        val integerPart = normalized.take(normalized.lastIndexOf(',')).replace("-", "")
        val decimals = normalized.length - normalized.lastIndexOf(',') - 1
        if usesEuroFormatting then normalized.replace(',', '.')
        else if decimals == 3 && integerPart != "0" then normalized.replace(",", "")
        else normalized.replace(',', '.')
      else normalized
    val signed =
      if cleaned.startsWith("-") then s"-${cleaned.drop(1).replace("-", "")}"
      else cleaned.replace("-", "")
    val amount = signed.toDouble
    if !amount.isFinite || amount < 0.0 then
      throw new IllegalArgumentException(s"invalid money amount '$raw'")
    roundChips(amount)

  private def roundChips(amount: Double): Double =
    math.round(amount * 100.0) / 100.0
