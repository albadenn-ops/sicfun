package sicfun.holdem.runtime

import sicfun.core.{Card, Deck, HandEvaluator}
import sicfun.holdem.cli.CliHelpers
import sicfun.holdem.types.{Board, HoleCards, Street}

import java.io.{BufferedReader, BufferedWriter, InputStreamReader, OutputStreamWriter}
import java.net.{ServerSocket, Socket}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import java.util.Locale
import scala.util.Random

object AcpcHeadsUpDealer:
  private val ProtocolVersion = "VERSION:2.0.0"
  private val SmallBlindChips = 50
  private val BigBlindChips = 100
  private val StackSizeChips = 20000

  private final case class Config(
      hands: Int,
      reportEvery: Int,
      outDir: Path,
      playerAName: String,
      playerBName: String,
      playerAScript: Path,
      playerBScript: Path,
      seed: Long,
      startTimeoutMillis: Int,
      actionTimeoutMillis: Int
  )

  final case class MatchSummary(
      handsPlayed: Int,
      elapsedSeconds: Double,
      handsPerSecond: Double,
      playerAName: String,
      playerBName: String,
      playerANetChips: Double,
      playerABbPer100: Double,
      playerBNetChips: Double,
      playerBBbPer100: Double,
      playerAButtonHands: Int,
      playerAButtonNetChips: Double,
      playerABigBlindHands: Int,
      playerABigBlindNetChips: Double,
      playerBButtonHands: Int,
      playerBButtonNetChips: Double,
      playerBBigBlindHands: Int,
      playerBBigBlindNetChips: Double,
      outDir: Path
  )

  private final case class HandOutcome(
      handNumber: Int,
      buttonPlayer: Int,
      holeByPos: Vector[HoleCards],
      board: Board,
      betting: String,
      playerNetByActual: Vector[Double]
  )

  private final class Client(
      val actualPlayer: Int,
      val name: String,
      val process: Process,
      val socket: Socket,
      val reader: BufferedReader,
      val writer: BufferedWriter
  ) extends AutoCloseable:
    def send(line: String): Unit =
      writer.write(line)
      writer.write("\r\n")
      writer.flush()

    def readRequiredLine(label: String): String =
      val line = reader.readLine()
      if line == null then throw new IllegalStateException(s"$name disconnected while waiting for $label")
      line

    override def close(): Unit =
      try writer.close()
      catch
        case _: Throwable => ()
      try reader.close()
      catch
        case _: Throwable => ()
      try socket.close()
      catch
        case _: Throwable => ()

  private final class HeadsUpHand(
      val handNumber: Int,
      val buttonPlayer: Int,
      val holeByPos: Vector[HoleCards],
      fullBoardCards: Vector[Card]
  ):
    private val actualByPos = Vector(buttonPlayer ^ 1, buttonPlayer)
    private val posByActual = actualByPos.zipWithIndex.toMap
    private val streetActions = Array.fill(Street.values.length)("")
    private var streetBaseContribution = Array(0, 0)
    private val totalContribution = Array(BigBlindChips, SmallBlindChips)
    private var streetContribution = Array(BigBlindChips, SmallBlindChips)
    private var currentStreetIdx = 0
    private var nextActorPos = 1
    private var streetLastBetTo = BigBlindChips
    private var lastBetSize = BigBlindChips - SmallBlindChips
    private var checkOrCallEndsStreet = false
    private var foldedPosOpt = Option.empty[Int]
    private var handOverFlag = false

    def bettingString: String =
      streetActions.take(currentStreetIdx + 1).mkString("/")

    def board: Board =
      currentStreet match
        case Street.Preflop => Board.empty
        case Street.Flop    => Board.from(fullBoardCards.take(3))
        case Street.Turn    => Board.from(fullBoardCards.take(4))
        case Street.River   => Board.from(fullBoardCards.take(5))

    def currentStreet: Street =
      currentStreetIdx match
        case 0 => Street.Preflop
        case 1 => Street.Flop
        case 2 => Street.Turn
        case 3 => Street.River
        case other => throw new IllegalStateException(s"invalid street index $other")

    def isHandOver: Boolean = handOverFlag
    def actorPos: Int = nextActorPos
    def actorActual: Int =
      if nextActorPos < 0 then -1 else actualByPos(nextActorPos)

    def renderForActual(actualPlayer: Int): String =
      val heroPos = posByActual.getOrElse(actualPlayer, throw new IllegalArgumentException(s"invalid player $actualPlayer"))
      val cards = renderCards(heroPos)
      s"MATCHSTATE:$heroPos:$handNumber:${bettingString}:$cards"

    def applyWireAction(wireAction: String): Either[String, Unit] =
      try
        require(!handOverFlag, "hand already finished")
        require(nextActorPos >= 0, "no actor available")
        val actor = nextActorPos
        val toCallChips = streetLastBetTo - streetContribution(actor)
        wireAction match
          case "f" =>
            require(toCallChips > 0, "fold is illegal when checking is free")
            streetActions(currentStreetIdx) += "f"
            foldedPosOpt = Some(actor)
            handOverFlag = true
            nextActorPos = -1
          case "c" =>
            streetActions(currentStreetIdx) += "c"
            if toCallChips > 0 then
              val remaining = StackSizeChips - totalContribution(actor)
              val paid = math.min(toCallChips, remaining)
              require(paid > 0, "call must contribute chips")
              streetContribution(actor) += paid
              totalContribution(actor) += paid
              lastBetSize = 0
              val allInResolved =
                totalContribution(actor) >= StackSizeChips || totalContribution(actor ^ 1) >= StackSizeChips
              if allInResolved then
                currentStreetIdx = Street.River.ordinal
                nextActorPos = -1
                handOverFlag = true
              else if checkOrCallEndsStreet then
                advanceStreetOrEnd()
              else
                nextActorPos = actor ^ 1
                checkOrCallEndsStreet = true
            else if checkOrCallEndsStreet then
              advanceStreetOrEnd()
            else
              nextActorPos = actor ^ 1
              checkOrCallEndsStreet = true
          case raw if raw.startsWith("r") =>
            val targetTotal = raw.drop(1).toIntOption.getOrElse(
              throw new IllegalArgumentException(s"invalid raise token '$wireAction'")
            )
            val target = targetTotal - streetBaseContribution(actor)
            val actorStreet = streetContribution(actor)
            val extraToCall = streetLastBetTo - actorStreet
            require(extraToCall >= 0, s"negative to-call while applying '$wireAction'")
            val paid = targetTotal - totalContribution(actor)
            val remaining = StackSizeChips - totalContribution(actor)
            require(paid > 0, "raise token must add chips")
            require(paid <= remaining, "raise exceeds stack")
            val isAllInCall =
              extraToCall > 0 && target == streetLastBetTo && paid == math.min(extraToCall, remaining) && paid == remaining
            if isAllInCall then
              streetActions(currentStreetIdx) += raw
              streetContribution(actor) += paid
              totalContribution(actor) += paid
              lastBetSize = 0
              val allInResolved =
                totalContribution(actor) >= StackSizeChips || totalContribution(actor ^ 1) >= StackSizeChips
              if allInResolved then
                currentStreetIdx = Street.River.ordinal
                nextActorPos = -1
                handOverFlag = true
              else if checkOrCallEndsStreet then
                advanceStreetOrEnd()
              else
                nextActorPos = actor ^ 1
                checkOrCallEndsStreet = true
            else
              require(target > streetLastBetTo, "raise target must exceed current bet")
              val extraIncrement = target - streetLastBetTo
              require(paid > extraToCall, "raise must add chips beyond call")
              val minIncrement =
                if lastBetSize > 0 then math.max(BigBlindChips, lastBetSize)
                else BigBlindChips
              val maxIncrement = remaining - extraToCall
              require(extraIncrement <= maxIncrement, "raise increment exceeds stack")
              require(
                extraIncrement == maxIncrement || extraIncrement >= minIncrement,
                "raise increment is below minimum legal size"
              )
              streetActions(currentStreetIdx) += raw
              streetContribution(actor) = target
              totalContribution(actor) = targetTotal
              streetLastBetTo = target
              lastBetSize = extraIncrement
              nextActorPos = actor ^ 1
              checkOrCallEndsStreet = true
          case other =>
            throw new IllegalArgumentException(s"unsupported action token '$other'")
        Right(())
      catch
        case error: IllegalArgumentException => Left(error.getMessage)

    def outcome(): HandOutcome =
      require(handOverFlag, "cannot compute outcome for unfinished hand")
      val rankByPos =
        if foldedPosOpt.isDefined then Vector.empty[Int]
        else
          holeByPos.map { hole =>
            HandEvaluator.evaluate7Cached(hole.toVector ++ board.cards).packed
          }
      val netByPos =
        foldedPosOpt match
          case Some(foldedPos) =>
            Vector.tabulate(2) { pos =>
              if pos == foldedPos then -totalContribution(pos).toDouble
              else totalContribution(foldedPos).toDouble
            }
          case None =>
            Vector.tabulate(2) { pos =>
              showdownValue(
                playerSpent = totalContribution.toVector,
                playerRank = rankByPos,
                playerIdx = pos
              )
            }
      val byActual = Vector.tabulate(2) { actual =>
        netByPos(posByActual(actual))
      }
      HandOutcome(
        handNumber = handNumber,
        buttonPlayer = buttonPlayer,
        holeByPos = holeByPos,
        board = board,
        betting = bettingString,
        playerNetByActual = byActual
      )

    private def advanceStreetOrEnd(): Unit =
      if currentStreetIdx >= Street.River.ordinal then
        handOverFlag = true
        nextActorPos = -1
      else
        currentStreetIdx += 1
        nextActorPos = 0
        streetBaseContribution = totalContribution.clone()
        streetContribution = Array(0, 0)
        streetLastBetTo = 0
        lastBetSize = 0
        checkOrCallEndsStreet = false

    private def renderCards(heroPos: Int): String =
      val revealVillain = handOverFlag && foldedPosOpt.isEmpty
      val holeSections = Vector.tabulate(2) { pos =>
        if pos == heroPos || revealVillain then holeByPos(pos).toToken else ""
      }
      val boardSections =
        currentStreetIdx match
          case 0 => Vector.empty[String]
          case 1 => Vector(fullBoardCards.take(3).map(_.toToken).mkString)
          case 2 => Vector(
              fullBoardCards.take(3).map(_.toToken).mkString,
              fullBoardCards.slice(3, 4).map(_.toToken).mkString
            )
          case 3 => Vector(
              fullBoardCards.take(3).map(_.toToken).mkString,
              fullBoardCards.slice(3, 4).map(_.toToken).mkString,
              fullBoardCards.slice(4, 5).map(_.toToken).mkString
            )
      holeSections.mkString("|") + boardSections.map("/" + _).mkString

  def main(args: Array[String]): Unit =
    val wantsHelp = args.contains("--help") || args.contains("-h")
    run(args) match
      case Right(summary) =>
        println("=== ACPC Heads-Up Dealer ===")
        println(s"handsPlayed: ${summary.handsPlayed}")
        println(s"elapsedSeconds: ${fmt(summary.elapsedSeconds, 3)}")
        println(s"handsPerSecond: ${fmt(summary.handsPerSecond, 3)}")
        println(s"${summary.playerAName}NetChips: ${fmt(summary.playerANetChips, 3)}")
        println(s"${summary.playerAName}BbPer100: ${fmt(summary.playerABbPer100, 3)}")
        println(s"${summary.playerBName}NetChips: ${fmt(summary.playerBNetChips, 3)}")
        println(s"${summary.playerBName}BbPer100: ${fmt(summary.playerBBbPer100, 3)}")
        println(s"outDir: ${summary.outDir.toAbsolutePath.normalize()}")
      case Left(error) =>
        if wantsHelp then println(error)
        else
          System.err.println(error)
          sys.exit(1)

  def run(args: Array[String]): Either[String, MatchSummary] =
    parseArgs(args).flatMap(config => new Runner(config).run())

  private final class Runner(config: Config):
    private val handsPath = config.outDir.resolve("hands.tsv")
    private val resultsPath = config.outDir.resolve("results.tsv")
    private val summaryPath = config.outDir.resolve("summary.txt")
    private val clientLogsDir = config.outDir.resolve("client-logs")
    private val rng = new Random(config.seed)
    private var startedAtNanos = 0L
    private var finishedAtNanos = 0L

    private var handsWriterOpt = Option.empty[BufferedWriter]
    private val playerNet = Array(0.0, 0.0)
    private val buttonHands = Array(0, 0)
    private val buttonNet = Array(0.0, 0.0)
    private val bigBlindHands = Array(0, 0)
    private val bigBlindNet = Array(0.0, 0.0)

    def run(): Either[String, MatchSummary] =
      var clients = Vector.empty[Client]
      try
        Files.createDirectories(config.outDir)
        Files.createDirectories(clientLogsDir)
        startedAtNanos = System.nanoTime()
        handsWriterOpt = Some(Files.newBufferedWriter(handsPath, StandardCharsets.UTF_8))
        writeLine(
          handsWriter,
          "hand\tbuttonPlayer\tbuttonHole\tbigBlindHole\tboard\tbetting\tplayerAName\tplayerANetChips\tplayerBName\tplayerBNetChips"
        )

        val sockets = Vector(openListener(), openListener())
        try
          clients = Vector(
            startClient(0, config.playerAName, config.playerAScript, sockets(0)),
            startClient(1, config.playerBName, config.playerBScript, sockets(1))
          )
        finally
          sockets.foreach(closeQuietly)

        var handNumber = 1
        while handNumber <= config.hands do
          val outcome = playHand(handNumber, clients)
          recordOutcome(outcome)
          appendHandLog(outcome)
          maybeReport(handNumber)
          handNumber += 1

        val summary = buildSummary()
        finishedAtNanos = System.nanoTime()
        val completedSummary = summary.copy(
          elapsedSeconds = elapsedSeconds,
          handsPerSecond = handsPerSecond
        )
        writeSummary(completedSummary)
        writeResults(completedSummary)
        Right(completedSummary)
      catch
        case error: Exception =>
          Left(s"acpc heads-up dealer failed: ${error.getMessage}")
      finally
        clients.foreach(stopClient)
        handsWriterOpt.foreach(closeQuietly)

    private def openListener(): ServerSocket =
      val server = new ServerSocket(0)
      server.setReuseAddress(true)
      server.setSoTimeout(config.startTimeoutMillis)
      server

    private def startClient(actualPlayer: Int, name: String, script: Path, server: ServerSocket): Client =
      val stdoutPath = clientLogsDir.resolve(s"$name.stdout.log")
      val stderrPath = clientLogsDir.resolve(s"$name.stderr.log")
      val command = buildCommand(script.toAbsolutePath.normalize(), "127.0.0.1", server.getLocalPort)
      val process = new ProcessBuilder(command*).directory(config.outDir.toFile).redirectOutput(stdoutPath.toFile).redirectError(stderrPath.toFile).start()
      val socket =
        try server.accept()
        catch
          case error: Exception =>
            process.destroyForcibly()
            throw new IllegalStateException(s"$name failed to connect: ${error.getMessage}")
      socket.setSoTimeout(config.actionTimeoutMillis)
      val reader = new BufferedReader(new InputStreamReader(socket.getInputStream, StandardCharsets.US_ASCII))
      val writer = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream, StandardCharsets.US_ASCII))
      val client = new Client(actualPlayer, name, process, socket, reader, writer)
      val version = client.readRequiredLine("protocol version")
      if version != ProtocolVersion then
        throw new IllegalStateException(s"$name reported unsupported protocol '$version'")
      client

    private def playHand(handNumber: Int, clients: Vector[Client]): HandOutcome =
      val buttonPlayer = (handNumber - 1) % 2
      val deck = rng.shuffle(Deck.full.toVector)
      val bigBlindHole = HoleCards.from(deck.take(2))
      val buttonHole = HoleCards.from(deck.slice(2, 4))
      val holeByPos = Vector(bigBlindHole, buttonHole)
      val fullBoard = deck.slice(4, 9)
      val hand = new HeadsUpHand(handNumber, buttonPlayer, holeByPos, fullBoard)

      sendStateToAll(hand, clients)
      while !hand.isHandOver do
        val actingPlayer = hand.actorActual
        val client = clients(actingPlayer)
        val expectedState = hand.renderForActual(actingPlayer)
        val response = client.readRequiredLine(s"action for hand $handNumber")
        val action = parseClientResponse(expectedState, response, client.name)
        hand.applyWireAction(action).fold(
          error => throw new IllegalStateException(s"${client.name} sent illegal action '$action': $error"),
          identity
        )
        sendStateToAll(hand, clients)

      hand.outcome()

    private def sendStateToAll(hand: HeadsUpHand, clients: Vector[Client]): Unit =
      clients.foreach { client =>
        client.send(hand.renderForActual(client.actualPlayer))
      }

    private def parseClientResponse(expectedState: String, response: String, playerName: String): String =
      val prefix = s"$expectedState:"
      if !response.startsWith(prefix) then
        throw new IllegalStateException(s"$playerName responded with malformed state echo: '$response'")
      val action = response.drop(prefix.length)
      if action == "f" || action == "c" || (action.startsWith("r") && action.drop(1).forall(_.isDigit) && action.length > 1) then action
      else throw new IllegalStateException(s"$playerName responded with unsupported action '$action'")

    private def recordOutcome(outcome: HandOutcome): Unit =
      playerNet(0) += outcome.playerNetByActual(0)
      playerNet(1) += outcome.playerNetByActual(1)

      val buttonActual = outcome.buttonPlayer
      val bigBlindActual = outcome.buttonPlayer ^ 1
      buttonHands(buttonActual) += 1
      buttonNet(buttonActual) += outcome.playerNetByActual(buttonActual)
      bigBlindHands(bigBlindActual) += 1
      bigBlindNet(bigBlindActual) += outcome.playerNetByActual(bigBlindActual)

    private def appendHandLog(outcome: HandOutcome): Unit =
      writeLine(
        handsWriter,
        Vector(
          outcome.handNumber.toString,
          playerName(outcome.buttonPlayer),
          outcome.holeByPos(1).toToken,
          outcome.holeByPos(0).toToken,
          outcome.board.cards.map(_.toToken).mkString,
          outcome.betting,
          config.playerAName,
          fmt(outcome.playerNetByActual(0), 3),
          config.playerBName,
          fmt(outcome.playerNetByActual(1), 3)
        ).mkString("\t")
      )

    private def maybeReport(handNumber: Int): Unit =
      if config.reportEvery > 0 && (handNumber % config.reportEvery == 0 || handNumber == config.hands) then
        println(
          s"[acpc-hu] hand=$handNumber ${config.playerAName}=${fmt(playerNet(0), 3)} ${config.playerBName}=${fmt(playerNet(1), 3)}"
        )

    private def buildSummary(): MatchSummary =
      MatchSummary(
        handsPlayed = config.hands,
        elapsedSeconds = 0.0,
        handsPerSecond = 0.0,
        playerAName = config.playerAName,
        playerBName = config.playerBName,
        playerANetChips = playerNet(0),
        playerABbPer100 = bbPer100(playerNet(0), config.hands),
        playerBNetChips = playerNet(1),
        playerBBbPer100 = bbPer100(playerNet(1), config.hands),
        playerAButtonHands = buttonHands(0),
        playerAButtonNetChips = buttonNet(0),
        playerABigBlindHands = bigBlindHands(0),
        playerABigBlindNetChips = bigBlindNet(0),
        playerBButtonHands = buttonHands(1),
        playerBButtonNetChips = buttonNet(1),
        playerBBigBlindHands = bigBlindHands(1),
        playerBBigBlindNetChips = bigBlindNet(1),
        outDir = config.outDir
      )

    private def elapsedSeconds: Double =
      if startedAtNanos <= 0L then 0.0
      else
        val end = if finishedAtNanos > 0L then finishedAtNanos else System.nanoTime()
        math.max(0.0, (end - startedAtNanos).toDouble / 1_000_000_000.0)

    private def handsPerSecond: Double =
      val elapsed = elapsedSeconds
      if elapsed > 0.0 then config.hands.toDouble / elapsed else 0.0

    private def writeSummary(summary: MatchSummary): Unit =
      val lines = Vector(
        "=== ACPC Heads-Up Dealer ===",
        s"handsPlayed: ${summary.handsPlayed}",
        s"elapsedSeconds: ${fmt(summary.elapsedSeconds, 3)}",
        s"handsPerSecond: ${fmt(summary.handsPerSecond, 3)}",
        s"${summary.playerAName}NetChips: ${fmt(summary.playerANetChips, 3)}",
        s"${summary.playerAName}BbPer100: ${fmt(summary.playerABbPer100, 3)}",
        s"${summary.playerAName}ButtonHands: ${summary.playerAButtonHands}",
        s"${summary.playerAName}ButtonNetChips: ${fmt(summary.playerAButtonNetChips, 3)}",
        s"${summary.playerAName}BigBlindHands: ${summary.playerABigBlindHands}",
        s"${summary.playerAName}BigBlindNetChips: ${fmt(summary.playerABigBlindNetChips, 3)}",
        s"${summary.playerBName}NetChips: ${fmt(summary.playerBNetChips, 3)}",
        s"${summary.playerBName}BbPer100: ${fmt(summary.playerBBbPer100, 3)}",
        s"${summary.playerBName}ButtonHands: ${summary.playerBButtonHands}",
        s"${summary.playerBName}ButtonNetChips: ${fmt(summary.playerBButtonNetChips, 3)}",
        s"${summary.playerBName}BigBlindHands: ${summary.playerBBigBlindHands}",
        s"${summary.playerBName}BigBlindNetChips: ${fmt(summary.playerBBigBlindNetChips, 3)}"
      )
      Files.write(summaryPath, lines.mkString(System.lineSeparator()).getBytes(StandardCharsets.UTF_8))

    private def writeResults(summary: MatchSummary): Unit =
      val rows = Vector(
        "player\thands\telapsedSeconds\thandsPerSecond\tnetChips\tbbPer100\tbuttonHands\tbuttonNetChips\tbigBlindHands\tbigBlindNetChips",
        Vector(
          summary.playerAName,
          summary.handsPlayed.toString,
          fmt(summary.elapsedSeconds, 3),
          fmt(summary.handsPerSecond, 3),
          fmt(summary.playerANetChips, 3),
          fmt(summary.playerABbPer100, 3),
          summary.playerAButtonHands.toString,
          fmt(summary.playerAButtonNetChips, 3),
          summary.playerABigBlindHands.toString,
          fmt(summary.playerABigBlindNetChips, 3)
        ).mkString("\t"),
        Vector(
          summary.playerBName,
          summary.handsPlayed.toString,
          fmt(summary.elapsedSeconds, 3),
          fmt(summary.handsPerSecond, 3),
          fmt(summary.playerBNetChips, 3),
          fmt(summary.playerBBbPer100, 3),
          summary.playerBButtonHands.toString,
          fmt(summary.playerBButtonNetChips, 3),
          summary.playerBBigBlindHands.toString,
          fmt(summary.playerBBigBlindNetChips, 3)
        ).mkString("\t")
      )
      Files.write(resultsPath, rows.mkString(System.lineSeparator()).getBytes(StandardCharsets.UTF_8))

    private def playerName(actualPlayer: Int): String =
      if actualPlayer == 0 then config.playerAName else config.playerBName

    private def handsWriter: BufferedWriter =
      handsWriterOpt.getOrElse(throw new IllegalStateException("hands writer not initialized"))

  private def stopClient(client: Client): Unit =
    closeQuietly(client)
    if client.process.isAlive then
      client.process.destroy()
      if client.process.isAlive then client.process.destroyForcibly()

  private def buildCommand(script: Path, host: String, port: Int): Vector[String] =
    val lower = script.getFileName.toString.toLowerCase(Locale.ROOT)
    if lower.endsWith(".cmd") || lower.endsWith(".bat") then
      Vector("cmd.exe", "/c", script.toString, host, port.toString)
    else if lower.endsWith(".ps1") then
      Vector("powershell", "-ExecutionPolicy", "Bypass", "-File", script.toString, host, port.toString)
    else
      Vector(script.toString, host, port.toString)

  private def showdownValue(playerSpent: Vector[Int], playerRank: Vector[Int], playerIdx: Int): Double =
    require(playerSpent.length == 2 && playerRank.length == 2, "heads-up showdown requires two players")
    val other = playerIdx ^ 1
    if playerRank(playerIdx) > playerRank(other) then playerSpent(other).toDouble
    else if playerRank(playerIdx) < playerRank(other) then -playerSpent(playerIdx).toDouble
    else (playerSpent(other).toDouble - playerSpent(playerIdx).toDouble) / 2.0

  private def bbPer100(netChips: Double, hands: Int): Double =
    if hands > 0 then (netChips / BigBlindChips.toDouble / hands.toDouble) * 100.0 else 0.0

  private def closeQuietly(resource: AutoCloseable): Unit =
    try resource.close()
    catch
      case _: Throwable => ()

  private def writeLine(writer: BufferedWriter, line: String): Unit =
    writer.write(line)
    writer.newLine()
    writer.flush()

  private def fmt(value: Double, digits: Int): String =
    String.format(Locale.ROOT, s"%.${digits}f", java.lang.Double.valueOf(value))

  private def parseArgs(args: Array[String]): Either[String, Config] =
    if args.contains("--help") || args.contains("-h") then Left(usage)
    else
      for
        options <- CliHelpers.parseOptions(args)
        hands <- CliHelpers.parseIntOptionEither(options, "hands", 100)
        _ <- if hands > 0 then Right(()) else Left("--hands must be > 0")
        reportEvery <- CliHelpers.parseIntOptionEither(options, "reportEvery", 20)
        _ <- if reportEvery >= 0 then Right(()) else Left("--reportEvery must be >= 0")
        outDir = Paths.get(options.getOrElse("outDir", "data/acpc-headsup-match"))
        playerAName = options.getOrElse("playerAName", "playerA")
        playerBName = options.getOrElse("playerBName", "playerB")
        playerAScript <- parseRequiredPath(options, "playerAScript")
        playerBScript <- parseRequiredPath(options, "playerBScript")
        seed <- CliHelpers.parseLongOptionEither(options, "seed", 42L)
        startTimeoutMillis <- CliHelpers.parseIntOptionEither(options, "startTimeoutMillis", 120000)
        _ <- if startTimeoutMillis > 0 then Right(()) else Left("--startTimeoutMillis must be > 0")
        actionTimeoutMillis <- CliHelpers.parseIntOptionEither(options, "actionTimeoutMillis", 15000)
        _ <- if actionTimeoutMillis > 0 then Right(()) else Left("--actionTimeoutMillis must be > 0")
      yield
        Config(
          hands = hands,
          reportEvery = reportEvery,
          outDir = outDir,
          playerAName = playerAName,
          playerBName = playerBName,
          playerAScript = playerAScript,
          playerBScript = playerBScript,
          seed = seed,
          startTimeoutMillis = startTimeoutMillis,
          actionTimeoutMillis = actionTimeoutMillis
        )

  private def parseRequiredPath(options: Map[String, String], key: String): Either[String, Path] =
    options.get(key) match
      case None => Left(s"--$key is required")
      case Some(raw) =>
        val path = Paths.get(raw)
        if Files.exists(path) then Right(path) else Left(s"--$key path '$raw' does not exist")

  private val usage =
    """Usage:
      |  runMain sicfun.holdem.runtime.AcpcHeadsUpDealer [--key=value ...]
      |
      |Options:
      |  --hands=100
      |  --reportEvery=20
      |  --outDir=data/acpc-headsup-match
      |  --playerAName=sicfun
      |  --playerBName=g5
      |  --playerAScript=scripts/start-sicfun-acpc.cmd
      |  --playerBScript=scripts/start-g5-acpc.cmd
      |  --seed=42
      |  --startTimeoutMillis=120000
      |  --actionTimeoutMillis=15000
      |""".stripMargin
