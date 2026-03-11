package sicfun.holdem.runtime

import sicfun.core.{Card, HandEvaluator}
import sicfun.holdem.cfr.{HoldemCfrConfig, HoldemCfrSolver}
import sicfun.holdem.cli.CliHelpers
import sicfun.holdem.engine.{RangeInferenceEngine, RealTimeAdaptiveEngine, VillainObservation}
import sicfun.holdem.equity.{PreflopFold, TableFormat, TableRanges}
import sicfun.holdem.model.{CalibrationGate, CalibrationSummary, ModelVersion, PokerActionModel, PokerActionModelArtifactIO, TrainedPokerActionModel}
import sicfun.holdem.types.*

import java.io.{BufferedReader, BufferedWriter, InputStreamReader, OutputStreamWriter}
import java.net.{InetSocketAddress, Socket}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import java.util.Locale
import scala.util.Random

private[holdem] object AcpcActionCodec:
  val SmallBlindChips = 50
  val BigBlindChips = 100
  val StackSizeChips = 20000

  final case class ActionStep(
      actualActor: Int,
      relativeActor: Int,
      action: PokerAction,
      stateBefore: GameState,
      streetContributionBeforeChips: Vector[Int],
      totalContributionBeforeChips: Vector[Int],
      streetLastBetToBeforeChips: Int,
      lastBetSizeBeforeChips: Int
  )

  final case class ParsedActionState(
      betting: String,
      heroActual: Int,
      nextActorActual: Int,
      currentStreet: Street,
      currentBoard: Board,
      potChips: Int,
      toCallChips: Int,
      stackRemainingChips: Int,
      streetContributionByActualChips: Vector[Int],
      totalContributionByActualChips: Vector[Int],
      streetLastBetToChips: Int,
      lastBetSizeChips: Int,
      betHistory: Vector[BetAction],
      steps: Vector[ActionStep],
      handOver: Boolean
  ):
    def nextActorRelative: Int =
      if nextActorActual < 0 then -1
      else if nextActorActual == heroActual then 0 else 1

    def nextDecisionState: Option[GameState] =
      if handOver || nextActorActual < 0 then None
      else
        Some(
          GameState(
            street = currentStreet,
            board = currentBoard,
            pot = chipsToBb(potChips),
            toCall = chipsToBb(toCallChips),
            position = positionForActual(nextActorActual),
            stackSize = chipsToBb(stackRemainingChips),
            betHistory = betHistory
          )
        )

  final case class MatchState(
      raw: String,
      position: Int,
      handNumber: Int,
      betting: String,
      heroActual: Int,
      heroHole: HoleCards,
      villainHole: Option[HoleCards],
      board: Board,
      parsed: ParsedActionState
  ):
    def heroPosition: Position = positionForActual(heroActual)
    def villainPosition: Position = positionForActual(heroActual ^ 1)

  def parseMatchState(raw: String): Either[String, MatchState] =
    try
      val parts = raw.split(":", 5).toVector
      require(parts.length == 5, s"invalid ACPC match state: '$raw'")
      require(parts.head == "MATCHSTATE", s"expected MATCHSTATE prefix, got '${parts.head}'")
      val position = parts(1).toIntOption.getOrElse(
        throw new IllegalArgumentException(s"invalid ACPC position '${parts(1)}'")
      )
      require(position == 0 || position == 1, s"unsupported ACPC heads-up position: $position")
      val handNumber = parts(2).toIntOption.getOrElse(
        throw new IllegalArgumentException(s"invalid ACPC hand number '${parts(2)}'")
      )
      val betting = parts(3)
      val cards = parts(4)
      val (heroHole, villainHole, board) = parseCards(cards, position)
      val parsed =
        parseBetting(betting, heroActual = position, fullBoard = board).fold(
          error => throw new IllegalArgumentException(error),
          identity
        )
      Right(
        MatchState(
          raw = raw,
          position = position,
          handNumber = handNumber,
          betting = betting,
          heroActual = position,
          heroHole = heroHole,
          villainHole = villainHole,
          board = board,
          parsed = parsed
        )
      )
    catch
      case error: IllegalArgumentException => Left(error.getMessage)

  def parseBetting(
      betting: String,
      heroActual: Int,
      fullBoard: Board
  ): Either[String, ParsedActionState] =
    try
      require(heroActual == 0 || heroActual == 1, s"heroActual must be 0 or 1, got $heroActual")

      var streetIdx = 0
      var nextActor = 1
      var streetLastBetTo = BigBlindChips
      var lastBetSize = BigBlindChips - SmallBlindChips
      var checkOrCallEndsStreet = false
      var handOver = false
      var streetBaseContribution = Array(0, 0)
      var streetContribution = Array(BigBlindChips, SmallBlindChips)
      var streetActionCount = 0
      val totalContribution = Array(BigBlindChips, SmallBlindChips)
      var betHistory = Vector.empty[BetAction]
      val steps = Vector.newBuilder[ActionStep]

      var i = 0
      while i < betting.length && !handOver do
        val actor = nextActor
        val stateBefore = buildState(
          actor = actor,
          streetIdx = streetIdx,
          fullBoard = fullBoard,
          totalContribution = totalContribution,
          streetContribution = streetContribution,
          streetLastBetTo = streetLastBetTo,
          betHistory = betHistory
        )
        val relativeActor = relativeActorId(actor, heroActual)
        val streetContributionBefore = streetContribution.toVector
        val totalContributionBefore = totalContribution.toVector

        betting.charAt(i) match
          case 'c' =>
            val stepAction =
              if stateBefore.toCall > 0.0 then PokerAction.Call else PokerAction.Check
            steps += ActionStep(
              actualActor = actor,
              relativeActor = relativeActor,
              action = stepAction,
              stateBefore = stateBefore,
              streetContributionBeforeChips = streetContributionBefore,
              totalContributionBeforeChips = totalContributionBefore,
              streetLastBetToBeforeChips = streetLastBetTo,
              lastBetSizeBeforeChips = lastBetSize
            )
            betHistory = betHistory :+ BetAction(relativeActor, stepAction)
            i += 1
            streetActionCount += 1

            if stateBefore.toCall > 0.0 then
              val toCallChips = bbToChips(stateBefore.toCall)
              val remaining = StackSizeChips - totalContribution(actor)
              val paid = math.min(toCallChips, remaining)
              require(paid > 0, "call must contribute chips")
              streetContribution(actor) += paid
              totalContribution(actor) += paid
              lastBetSize = 0
              val allInResolved =
                totalContribution(actor) >= StackSizeChips || totalContribution(actor ^ 1) >= StackSizeChips
              if allInResolved then
                while i < betting.length do
                  require(betting.charAt(i) == '/', "unexpected token after all-in call")
                  i += 1
                handOver = true
                nextActor = -1
                streetIdx = math.max(streetIdx, streetIndexForBoard(fullBoard))
              else if checkOrCallEndsStreet then
                if i < betting.length then
                  require(betting.charAt(i) == '/', "missing slash after street-ending call")
                  i += 1
                if streetIdx >= Street.values.length - 1 then
                  handOver = true
                  nextActor = -1
                else
                  streetIdx += 1
                  nextActor = 0
                  streetBaseContribution = totalContribution.clone()
                  streetContribution = Array(0, 0)
                  streetLastBetTo = 0
                  streetActionCount = 0
                  checkOrCallEndsStreet = false
              else
                nextActor = actor ^ 1
                checkOrCallEndsStreet = true
            else if checkOrCallEndsStreet then
              if i < betting.length then
                require(betting.charAt(i) == '/', "missing slash after street-ending check")
                i += 1
                if streetIdx >= Street.values.length - 1 then
                  handOver = true
                  nextActor = -1
                else
                  streetIdx += 1
                  nextActor = 0
                  streetBaseContribution = totalContribution.clone()
                  streetContribution = Array(0, 0)
                  streetLastBetTo = 0
                  lastBetSize = 0
                  streetActionCount = 0
                  checkOrCallEndsStreet = false
            else
              nextActor = actor ^ 1
              checkOrCallEndsStreet = true

          case 'f' =>
            require(stateBefore.toCall > 0.0, "illegal fold in ACPC betting string")
            val stepAction = PokerAction.Fold
            steps += ActionStep(
              actualActor = actor,
              relativeActor = relativeActor,
              action = stepAction,
              stateBefore = stateBefore,
              streetContributionBeforeChips = streetContributionBefore,
              totalContributionBeforeChips = totalContributionBefore,
              streetLastBetToBeforeChips = streetLastBetTo,
              lastBetSizeBeforeChips = lastBetSize
            )
            betHistory = betHistory :+ BetAction(relativeActor, stepAction)
            i += 1
            streetActionCount += 1
            require(i == betting.length, "unexpected token after fold")
            handOver = true
            nextActor = -1

          case 'r' =>
            i += 1
            val start = i
            while i < betting.length && betting.charAt(i).isDigit do i += 1
            require(i > start, "missing raise target in ACPC betting string")
            val newTotalBetTo = betting.substring(start, i).toInt
            val actorStreetContribution = streetContribution(actor)
            val newStreetBetTo = newTotalBetTo - streetBaseContribution(actor)
            val toCallChips = streetLastBetTo - actorStreetContribution
            val paid = newTotalBetTo - totalContribution(actor)
            val remaining = StackSizeChips - totalContribution(actor)
            require(toCallChips >= 0, s"negative toCall while parsing betting '$betting'")
            require(paid > 0, "raise token must add chips")
            require(paid <= remaining, "raise exceeds stack")
            val isAllInCall =
              toCallChips > 0 && newStreetBetTo == streetLastBetTo && paid == math.min(toCallChips, remaining) && paid == remaining
            if isAllInCall then
              val stepAction = PokerAction.Call
              steps += ActionStep(
                actualActor = actor,
                relativeActor = relativeActor,
                action = stepAction,
                stateBefore = stateBefore,
                streetContributionBeforeChips = streetContributionBefore,
                totalContributionBeforeChips = totalContributionBefore,
                streetLastBetToBeforeChips = streetLastBetTo,
                lastBetSizeBeforeChips = lastBetSize
              )
              betHistory = betHistory :+ BetAction(relativeActor, stepAction)
              streetActionCount += 1
              streetContribution(actor) += paid
              totalContribution(actor) += paid
              lastBetSize = 0
              val allInResolved =
                totalContribution(actor) >= StackSizeChips || totalContribution(actor ^ 1) >= StackSizeChips
              if allInResolved then
                while i < betting.length do
                  require(betting.charAt(i) == '/', "unexpected token after all-in call")
                  i += 1
                handOver = true
                nextActor = -1
                streetIdx = math.max(streetIdx, streetIndexForBoard(fullBoard))
              else if checkOrCallEndsStreet then
                if i < betting.length then
                  require(betting.charAt(i) == '/', "missing slash after street-ending call")
                  i += 1
                if streetIdx >= Street.values.length - 1 then
                  handOver = true
                  nextActor = -1
                else
                  streetIdx += 1
                  nextActor = 0
                  streetBaseContribution = totalContribution.clone()
                  streetContribution = Array(0, 0)
                  streetLastBetTo = 0
                  streetActionCount = 0
                  checkOrCallEndsStreet = false
              else
                nextActor = actor ^ 1
                checkOrCallEndsStreet = true
            else
              require(newStreetBetTo > streetLastBetTo, "raise target must exceed current bet level")
              val extraIncrement = newStreetBetTo - streetLastBetTo
              require(paid > toCallChips, "raise must add chips beyond call")
              val minIncrement =
                if lastBetSize > 0 then math.max(BigBlindChips, lastBetSize)
                else BigBlindChips
              val maxIncrement = remaining - toCallChips
              require(extraIncrement <= maxIncrement, "raise increment exceeds stack")
              require(
                extraIncrement == maxIncrement || extraIncrement >= minIncrement,
                "raise increment is below minimum legal size"
              )
              val stepAction = PokerAction.Raise(chipsToBb(extraIncrement))
              steps += ActionStep(
                actualActor = actor,
                relativeActor = relativeActor,
                action = stepAction,
                stateBefore = stateBefore,
                streetContributionBeforeChips = streetContributionBefore,
                totalContributionBeforeChips = totalContributionBefore,
                streetLastBetToBeforeChips = streetLastBetTo,
                lastBetSizeBeforeChips = lastBetSize
              )
              betHistory = betHistory :+ BetAction(relativeActor, stepAction)
              streetActionCount += 1
              streetContribution(actor) = newStreetBetTo
              totalContribution(actor) = newTotalBetTo
              streetLastBetTo = newStreetBetTo
              lastBetSize = extraIncrement
              nextActor = actor ^ 1
              checkOrCallEndsStreet = true

          case '/' =>
            throw new IllegalArgumentException("unexpected slash in ACPC betting string")

          case other =>
            throw new IllegalArgumentException(s"unexpected token '$other' in ACPC betting string")

      if !handOver && i >= betting.length && streetIdx >= Street.River.ordinal && checkOrCallEndsStreet && streetActionCount >= 2 && streetContribution(0) == streetContribution(1) then
        handOver = true
        nextActor = -1

      val currentStreet = streetFromIndex(streetIdx)
      val currentBoard = boardForStreet(fullBoard, currentStreet)
      val potChips = totalContribution.sum
      val toCallChips =
        if nextActor < 0 then 0
        else streetLastBetTo - streetContribution(nextActor)
      val stackRemaining =
        if nextActor < 0 then 0
        else StackSizeChips - totalContribution(nextActor)

      Right(
        ParsedActionState(
          betting = betting,
          heroActual = heroActual,
          nextActorActual = nextActor,
          currentStreet = currentStreet,
          currentBoard = currentBoard,
          potChips = potChips,
          toCallChips = math.max(0, toCallChips),
          stackRemainingChips = math.max(0, stackRemaining),
          streetContributionByActualChips = streetContribution.toVector,
          totalContributionByActualChips = totalContribution.toVector,
          streetLastBetToChips = streetLastBetTo,
          lastBetSizeChips = lastBetSize,
          betHistory = betHistory,
          steps = steps.result(),
          handOver = handOver
        )
      )
    catch
      case error: IllegalArgumentException => Left(error.getMessage)

  def wireActionFor(parsed: ParsedActionState, action: PokerAction): String =
    action match
      case PokerAction.Fold => "f"
      case PokerAction.Check => "c"
      case PokerAction.Call => "c"
      case PokerAction.Raise(amountBb) =>
        val actor = parsed.nextActorActual
        require(actor >= 0, "cannot emit raise for terminal ACPC state")
        val incrementChips = bbToChips(amountBb)
        val target = parsed.totalContributionByActualChips(actor) + parsed.toCallChips + incrementChips
        s"r$target"

  def handValue(state: MatchState): Either[String, Double] =
    try
      require(state.parsed.handOver, "cannot value non-terminal ACPC match state")
      val heroActual = state.heroActual
      val heroSpent = state.parsed.totalContributionByActualChips(heroActual)
      val villainSpent = state.parsed.totalContributionByActualChips(heroActual ^ 1)

      state.parsed.steps.lastOption match
        case Some(last) if last.action == PokerAction.Fold =>
          if last.actualActor == heroActual then Right(-heroSpent.toDouble)
          else Right(villainSpent.toDouble)
        case _ =>
          val villainHole = state.villainHole.getOrElse(
            throw new IllegalArgumentException("terminal showdown missing villain hole cards")
          )
          require(state.board.size == 5, s"terminal showdown requires 5 board cards, got ${state.board.size}")
          val heroRank = HandEvaluator.evaluate7Cached(state.heroHole.toVector ++ state.board.cards)
          val villainRank = HandEvaluator.evaluate7Cached(villainHole.toVector ++ state.board.cards)
          Right(
            showdownValue(
              playerSpent = Vector(heroSpent, villainSpent),
              playerRank = Vector(heroRank.packed, villainRank.packed),
              playerIdx = 0
            )
          )
    catch
      case error: IllegalArgumentException => Left(error.getMessage)

  def positionForActual(actualActor: Int): Position =
    actualActor match
      case 1 => Position.Button
      case 0 => Position.BigBlind
      case other => throw new IllegalArgumentException(s"invalid player index: $other")

  def chipsToBb(chips: Int): Double =
    chips.toDouble / BigBlindChips.toDouble

  def bbToChips(bb: Double): Int =
    math.max(0, math.round(bb * BigBlindChips.toDouble).toInt)

  private def parseCards(cards: String, heroActual: Int): (HoleCards, Option[HoleCards], Board) =
    val segments = cards.split("/", -1).toVector
    require(segments.nonEmpty, "ACPC cards section cannot be empty")
    val holeSegments = segments.head.split("\\|", -1).toVector
    require(holeSegments.length == 2, s"expected 2 hole-card sections, got '${segments.head}'")
    val holeByActual = holeSegments.map(parseHoleCardsOption)
    val heroHole = holeByActual(heroActual).getOrElse(
      throw new IllegalArgumentException("ACPC match state is missing hero hole cards")
    )
    val villainHole = holeByActual(heroActual ^ 1)
    val board = Board.from(segments.drop(1).flatMap(parseCardRun))
    require(
      Set(0, 3, 4, 5).contains(board.size),
      s"unsupported board length ${board.size} in ACPC match state"
    )
    val knownCards = heroHole.toVector ++ villainHole.toVector.flatMap(_.toVector) ++ board.cards
    require(knownCards.distinct.length == knownCards.length, "duplicate cards in ACPC match state")
    (heroHole, villainHole, board)

  private def parseHoleCardsOption(raw: String): Option[HoleCards] =
    if raw.isEmpty then None
    else Some(HoleCards.from(parseCardRun(raw)))

  private def parseCardRun(raw: String): Vector[Card] =
    require(raw.length % 2 == 0, s"invalid ACPC card run '$raw'")
    raw.grouped(2).map { token =>
      Card.parse(token).getOrElse(
        throw new IllegalArgumentException(s"invalid card token '$token' in ACPC match state")
      )
    }.toVector

  private def buildState(
      actor: Int,
      streetIdx: Int,
      fullBoard: Board,
      totalContribution: Array[Int],
      streetContribution: Array[Int],
      streetLastBetTo: Int,
      betHistory: Vector[BetAction]
  ): GameState =
    GameState(
      street = streetFromIndex(streetIdx),
      board = boardForStreet(fullBoard, streetFromIndex(streetIdx)),
      pot = chipsToBb(totalContribution.sum),
      toCall = chipsToBb(streetLastBetTo - streetContribution(actor)),
      position = positionForActual(actor),
      stackSize = chipsToBb(StackSizeChips - totalContribution(actor)),
      betHistory = betHistory
    )

  private def boardForStreet(fullBoard: Board, street: Street): Board =
    val expected = street.expectedBoardSize
    require(
      fullBoard.size >= expected,
      s"board has size ${fullBoard.size} but street $street requires at least $expected cards"
    )
    Board.from(fullBoard.cards.take(expected))

  private def streetIndexForBoard(board: Board): Int =
    board.size match
      case 0 => 0
      case 3 => 1
      case 4 => 2
      case 5 => 3
      case other => throw new IllegalArgumentException(s"unsupported board size: $other")

  private def streetFromIndex(streetIdx: Int): Street =
    streetIdx match
      case 0 => Street.Preflop
      case 1 => Street.Flop
      case 2 => Street.Turn
      case 3 => Street.River
      case other => throw new IllegalArgumentException(s"invalid street index: $other")

  private def relativeActorId(actualActor: Int, heroActual: Int): Int =
    if actualActor == heroActual then 0 else 1

  private def showdownValue(
      playerSpent: Vector[Int],
      playerRank: Vector[Int],
      playerIdx: Int
  ): Double =
    require(playerSpent.length == playerRank.length, "spent/rank vector size mismatch")
    require(playerIdx >= 0 && playerIdx < playerSpent.length, s"invalid player index $playerIdx")

    var spent = playerSpent
      .zip(playerRank)
      .filter(_._1 > 0)
      .toVector
    var idx = spent.indexWhere(_ == (playerSpent(playerIdx), playerRank(playerIdx)))
    require(idx >= 0, "hero is not participating in the terminal pot")
    var value = 0.0

    while true do
      val size = spent.map(_._1).min
      val winRank = spent.map(_._2).max
      val numWinners = spent.count(_._2 == winRank)

      if spent(idx)._2 == winRank then
        value += size.toDouble * (spent.length - numWinners).toDouble / numWinners.toDouble
      else
        value -= size.toDouble

      val oldHero = spent(idx)
      val reduced = spent.map { case (amount, rank) => (amount - size, rank) }.filter(_._1 > 0)
      if oldHero._1 - size <= 0 then return value
      spent = reduced
      idx = spent.indexWhere(_._2 == oldHero._2)
      require(idx >= 0, "hero disappeared from reduced side-pot state")

    value

object AcpcMatchRunner:
  private val ProtocolVersion = "VERSION:2.0.0"

  private enum HeroMode:
    case Adaptive
    case Gto

  private final case class Config(
      server: String,
      port: Int,
      reportEvery: Int,
      outDir: Path,
      modelArtifactDir: Option[Path],
      heroMode: HeroMode,
      bunchingTrials: Int,
      equityTrials: Int,
      cfrIterations: Int,
      cfrVillainHands: Int,
      cfrEquityTrials: Int,
      seed: Long,
      timeoutMillis: Int,
      connectTimeoutMillis: Int
  )

  final case class RunSummary(
      handsPlayed: Int,
      heroNetChips: Double,
      heroBbPer100: Double,
      heroWins: Int,
      heroTies: Int,
      heroLosses: Int,
      buttonHands: Int,
      buttonNetChips: Double,
      bigBlindHands: Int,
      bigBlindNetChips: Double,
      modelId: String,
      outDir: Path
  )

  private final case class HandOutcome(
      handNumber: Int,
      heroPosition: Position,
      heroHole: HoleCards,
      villainHole: Option[HoleCards],
      board: Board,
      betting: String,
      heroNetChips: Double,
      decisionCount: Int,
      villainObservationCount: Int
  )

  private final class LiveHand(
      val handNumber: Int,
      val heroPosition: Position,
      val villainPosition: Position,
      val heroHole: HoleCards
  ):
    var villainHole: Option[HoleCards] = None
    var processedSteps = 0
    var decisionCount = 0
    var pendingHeroRaise = false
    var villainObservations = Vector.empty[VillainObservation]
    var completed = false

  def main(args: Array[String]): Unit =
    val wantsHelp = args.contains("--help") || args.contains("-h")
    run(args) match
      case Right(summary) =>
        println("=== ACPC Match Runner ===")
        println(s"handsPlayed: ${summary.handsPlayed}")
        println(s"heroNetChips: ${fmt(summary.heroNetChips, 3)}")
        println(s"heroBbPer100: ${fmt(summary.heroBbPer100, 3)}")
        println(s"heroWins: ${summary.heroWins}")
        println(s"heroTies: ${summary.heroTies}")
        println(s"heroLosses: ${summary.heroLosses}")
        println(s"buttonHands: ${summary.buttonHands}")
        println(s"buttonNetChips: ${fmt(summary.buttonNetChips, 3)}")
        println(s"bigBlindHands: ${summary.bigBlindHands}")
        println(s"bigBlindNetChips: ${fmt(summary.bigBlindNetChips, 3)}")
        println(s"modelId: ${summary.modelId}")
        println(s"outDir: ${summary.outDir.toAbsolutePath.normalize()}")
      case Left(error) =>
        if wantsHelp then println(error)
        else
          System.err.println(error)
          sys.exit(1)

  def run(args: Array[String]): Either[String, RunSummary] =
    parseArgs(args).flatMap(config => new Runner(config).run())

  private final class Runner(config: Config):
    private val handsPath = config.outDir.resolve("hands.tsv")
    private val decisionsPath = config.outDir.resolve("decisions.tsv")
    private val summaryPath = config.outDir.resolve("summary.txt")

    private val tableRanges = TableRanges.defaults(TableFormat.NineMax)
    private val folds = TableFormat.NineMax.foldsBeforeOpener(Position.Button).map(PreflopFold(_))
    private val rng = new Random(config.seed)

    private var handsWriterOpt = Option.empty[BufferedWriter]
    private var decisionsWriterOpt = Option.empty[BufferedWriter]

    private var handsPlayed = 0
    private var heroNetChips = 0.0
    private var heroWins = 0
    private var heroTies = 0
    private var heroLosses = 0
    private var buttonHands = 0
    private var buttonNetChips = 0.0
    private var bigBlindHands = 0
    private var bigBlindNetChips = 0.0

    private val (artifact, modelId) = loadArtifact(config)
    private val engine = newAdaptiveEngine(artifact.model)

    def run(): Either[String, RunSummary] =
      var socket: Socket | Null = null
      var reader: BufferedReader | Null = null
      var writer: BufferedWriter | Null = null
      try
        Files.createDirectories(config.outDir)
        handsWriterOpt = Some(Files.newBufferedWriter(handsPath, StandardCharsets.UTF_8))
        decisionsWriterOpt = Some(Files.newBufferedWriter(decisionsPath, StandardCharsets.UTF_8))
        writeLine(
          handsWriter,
          "hand\theroPosition\theroHole\tvillainHole\tboard\theroNetChips\theroNetBb\tdecisions\tvillainObservations\tbetting"
        )
        writeLine(
          decisionsWriter,
          "hand\tdecisionIndex\tstreet\theroPosition\tpotBeforeBb\ttoCallBb\tstackBb\tcandidates\tchosenAction\tacpcAction"
        )

        socket = new Socket()
        socket.connect(new InetSocketAddress(config.server, config.port), config.connectTimeoutMillis)
        socket.setSoTimeout(config.timeoutMillis)
        reader = new BufferedReader(new InputStreamReader(socket.getInputStream, StandardCharsets.US_ASCII))
        writer = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream, StandardCharsets.US_ASCII))
        sendLine(writer, ProtocolVersion)
        loop(reader, writer)

        val summary = buildSummary()
        writeSummary(summary)
        Right(summary)
      catch
        case error: Exception =>
          Left(s"acpc match runner failed: ${error.getMessage}")
      finally
        decisionsWriterOpt.foreach(closeQuietly)
        handsWriterOpt.foreach(closeQuietly)
        if reader != null then closeQuietly(reader)
        if writer != null then closeQuietly(writer)
        if socket != null then closeQuietly(socket)

    private def loop(reader: BufferedReader, writer: BufferedWriter): Unit =
      var liveHandOpt = Option.empty[LiveHand]
      var line = reader.readLine()
      while line != null do
        if line.nonEmpty && line.charAt(0) != '#' && line.charAt(0) != ';' then
          val matchState = AcpcActionCodec.parseMatchState(line).fold(
            error => throw new IllegalStateException(s"failed to parse ACPC state '$line': $error"),
            identity
          )
          val liveHand =
            liveHandOpt match
              case Some(existing) if existing.handNumber == matchState.handNumber =>
                existing
              case _ =>
                val created = new LiveHand(
                  handNumber = matchState.handNumber,
                  heroPosition = matchState.heroPosition,
                  villainPosition = matchState.villainPosition,
                  heroHole = matchState.heroHole
                )
                liveHandOpt = Some(created)
                created
          liveHand.villainHole = matchState.villainHole.orElse(liveHand.villainHole)
          processSteps(matchState, liveHand)
          if matchState.parsed.handOver then
            if !liveHand.completed then
              val winnings = AcpcActionCodec.handValue(matchState).fold(
                error => throw new IllegalStateException(s"failed to value ACPC hand ${matchState.handNumber}: $error"),
                identity
              )
              val outcome = HandOutcome(
                handNumber = matchState.handNumber,
                heroPosition = liveHand.heroPosition,
                heroHole = liveHand.heroHole,
                villainHole = liveHand.villainHole,
                board = matchState.board,
                betting = matchState.betting,
                heroNetChips = winnings,
                decisionCount = liveHand.decisionCount,
                villainObservationCount = liveHand.villainObservations.length
              )
              liveHand.completed = true
              recordOutcome(outcome)
              appendHandLog(outcome)
              maybeReport()
          else if matchState.parsed.nextActorActual == matchState.heroActual then
            val heroState = matchState.parsed.nextDecisionState.getOrElse(
              throw new IllegalStateException("missing hero decision state")
            )
            val candidates = heroCandidates(matchState.parsed)
            val chosenAction = decideHero(
              hero = liveHand.heroHole,
              state = heroState,
              villainPosition = liveHand.villainPosition,
              villainObservations = liveHand.villainObservations,
              candidates = candidates
            )
            val acpcAction = AcpcActionCodec.wireActionFor(matchState.parsed, chosenAction)
            liveHand.decisionCount += 1
            appendDecisionLog(matchState.handNumber, liveHand.decisionCount, heroState, candidates, chosenAction, acpcAction)
            sendLine(writer, s"${matchState.raw}:$acpcAction")

        line = reader.readLine()

    private def processSteps(matchState: AcpcActionCodec.MatchState, liveHand: LiveHand): Unit =
      val newSteps = matchState.parsed.steps.drop(liveHand.processedSteps)
      liveHand.processedSteps = matchState.parsed.steps.length
      newSteps.foreach { step =>
        if step.relativeActor == 1 then
          liveHand.villainObservations = liveHand.villainObservations :+ VillainObservation(step.action, step.stateBefore)
          if liveHand.pendingHeroRaise then
            step.action match
              case PokerAction.Fold | PokerAction.Call | PokerAction.Raise(_) =>
                engine.observeVillainResponseToRaise(step.action)
                liveHand.pendingHeroRaise = false
              case _ => ()
        else
          step.action match
            case PokerAction.Raise(_) => liveHand.pendingHeroRaise = true
            case _ => ()
      }

    private def decideHero(
        hero: HoleCards,
        state: GameState,
        villainPosition: Position,
        villainObservations: Vector[VillainObservation],
        candidates: Vector[PokerAction]
    ): PokerAction =
      config.heroMode match
        case HeroMode.Adaptive =>
          engine
            .decide(
              hero = hero,
              state = state,
              folds = folds,
              villainPos = villainPosition,
              observations = villainObservations,
              candidateActions = candidates,
              decisionBudgetMillis = Some(1L),
              rng = new Random(rng.nextLong())
            )
            .decision
            .recommendation
            .bestAction
        case HeroMode.Gto =>
          val posterior = RangeInferenceEngine
            .inferPosterior(
              hero = hero,
              board = state.board,
              folds = folds,
              tableRanges = tableRanges,
              villainPos = villainPosition,
              observations = villainObservations,
              actionModel = artifact.model,
              bunchingTrials = config.bunchingTrials,
              rng = new Random(rng.nextLong())
            )
            .posterior
          HoldemCfrSolver
            .solveShallowDecisionPolicy(
              hero = hero,
              state = state,
              villainPosterior = posterior,
              candidateActions = candidates,
              config = HoldemCfrConfig(
                iterations = config.cfrIterations,
                maxVillainHands = config.cfrVillainHands,
                equityTrials = config.cfrEquityTrials,
                rngSeed = rng.nextLong()
              )
            )
            .bestAction

    private def heroCandidates(parsed: AcpcActionCodec.ParsedActionState): Vector[PokerAction] =
      val raises = legalRaiseCandidates(parsed)
      if parsed.toCallChips <= 0 then Vector(PokerAction.Check) ++ raises
      else Vector(PokerAction.Fold, PokerAction.Call) ++ raises

    private def legalRaiseCandidates(parsed: AcpcActionCodec.ParsedActionState): Vector[PokerAction] =
      val remaining = parsed.stackRemainingChips
      val toCall = parsed.toCallChips
      val maxIncrement = remaining - toCall
      if maxIncrement <= 0 then Vector.empty
      else
        val minIncrement =
          math.min(
            maxIncrement,
            if parsed.lastBetSizeChips > 0 then math.max(AcpcActionCodec.BigBlindChips, parsed.lastBetSizeChips)
            else AcpcActionCodec.BigBlindChips
          )
        val rawIncrements =
          if parsed.currentStreet == Street.Preflop && parsed.toCallChips > 0 && parsed.streetLastBetToChips == AcpcActionCodec.BigBlindChips then
            Vector(150, 200)
          else if parsed.currentStreet == Street.Preflop && parsed.toCallChips == 0 && parsed.streetLastBetToChips == AcpcActionCodec.BigBlindChips then
            Vector(200, 300)
          else if parsed.toCallChips <= 0 then
            Vector(
              roundedChips(parsed.potChips * 0.50),
              roundedChips(parsed.potChips * 0.75)
            )
          else
            Vector(
              minIncrement,
              roundedChips(parsed.potChips * 0.75)
            )
        rawIncrements
          .map(value => math.max(minIncrement, math.min(maxIncrement, value)))
          .distinct
          .sorted
          .map(value => PokerAction.Raise(AcpcActionCodec.chipsToBb(value)))
          .toVector

    private def recordOutcome(outcome: HandOutcome): Unit =
      handsPlayed += 1
      heroNetChips += outcome.heroNetChips
      if outcome.heroNetChips > 0.0 then heroWins += 1
      else if outcome.heroNetChips < 0.0 then heroLosses += 1
      else heroTies += 1

      outcome.heroPosition match
        case Position.Button =>
          buttonHands += 1
          buttonNetChips += outcome.heroNetChips
        case Position.BigBlind =>
          bigBlindHands += 1
          bigBlindNetChips += outcome.heroNetChips
        case other =>
          throw new IllegalStateException(s"unexpected hero position in ACPC runner: $other")

    private def appendHandLog(outcome: HandOutcome): Unit =
      writeLine(
        handsWriter,
        Vector(
          outcome.handNumber.toString,
          outcome.heroPosition.toString,
          outcome.heroHole.toToken,
          outcome.villainHole.map(_.toToken).getOrElse(""),
          outcome.board.cards.map(_.toToken).mkString,
          fmt(outcome.heroNetChips, 3),
          fmt(outcome.heroNetChips / AcpcActionCodec.BigBlindChips.toDouble, 3),
          outcome.decisionCount.toString,
          outcome.villainObservationCount.toString,
          outcome.betting
        ).mkString("\t")
      )

    private def appendDecisionLog(
        handNumber: Int,
        decisionIndex: Int,
        state: GameState,
        candidates: Vector[PokerAction],
        chosenAction: PokerAction,
        acpcAction: String
    ): Unit =
      writeLine(
        decisionsWriter,
        Vector(
          handNumber.toString,
          decisionIndex.toString,
          state.street.toString,
          state.position.toString,
          fmt(state.pot, 3),
          fmt(state.toCall, 3),
          fmt(state.stackSize, 3),
          candidates.map(renderAction).mkString(","),
          renderAction(chosenAction),
          acpcAction
        ).mkString("\t")
      )

    private def maybeReport(): Unit =
      if config.reportEvery > 0 && (handsPlayed % config.reportEvery == 0) then
        println(
          s"[acpc] hands=$handsPlayed netChips=${fmt(heroNetChips, 3)} bb100=${fmt(currentBbPer100, 3)} mode=${heroModeLabel(config.heroMode)} model=$modelId"
        )

    private def currentBbPer100: Double =
      if handsPlayed > 0 then
        (heroNetChips / AcpcActionCodec.BigBlindChips.toDouble / handsPlayed.toDouble) * 100.0
      else 0.0

    private def buildSummary(): RunSummary =
      RunSummary(
        handsPlayed = handsPlayed,
        heroNetChips = heroNetChips,
        heroBbPer100 = currentBbPer100,
        heroWins = heroWins,
        heroTies = heroTies,
        heroLosses = heroLosses,
        buttonHands = buttonHands,
        buttonNetChips = buttonNetChips,
        bigBlindHands = bigBlindHands,
        bigBlindNetChips = bigBlindNetChips,
        modelId = modelId,
        outDir = config.outDir
      )

    private def writeSummary(summary: RunSummary): Unit =
      val lines = Vector(
        "=== ACPC Match Runner ===",
        s"handsPlayed: ${summary.handsPlayed}",
        s"heroNetChips: ${fmt(summary.heroNetChips, 3)}",
        s"heroBbPer100: ${fmt(summary.heroBbPer100, 3)}",
        s"heroWins: ${summary.heroWins}",
        s"heroTies: ${summary.heroTies}",
        s"heroLosses: ${summary.heroLosses}",
        s"buttonHands: ${summary.buttonHands}",
        s"buttonNetChips: ${fmt(summary.buttonNetChips, 3)}",
        s"bigBlindHands: ${summary.bigBlindHands}",
        s"bigBlindNetChips: ${fmt(summary.bigBlindNetChips, 3)}",
        s"heroMode: ${heroModeLabel(config.heroMode)}",
        s"modelId: ${summary.modelId}"
      )
      Files.write(summaryPath, lines.mkString(System.lineSeparator()).getBytes(StandardCharsets.UTF_8))

    private def newAdaptiveEngine(model: PokerActionModel): RealTimeAdaptiveEngine =
      new RealTimeAdaptiveEngine(
        tableRanges = tableRanges,
        actionModel = model,
        bunchingTrials = config.bunchingTrials,
        defaultEquityTrials = config.equityTrials,
        minEquityTrials = math.max(8, math.min(config.equityTrials, config.equityTrials / 10))
      )

    private def handsWriter: BufferedWriter =
      handsWriterOpt.getOrElse(throw new IllegalStateException("hands writer not initialized"))

    private def decisionsWriter: BufferedWriter =
      decisionsWriterOpt.getOrElse(throw new IllegalStateException("decisions writer not initialized"))

  private def loadArtifact(config: Config): (TrainedPokerActionModel, String) =
    config.modelArtifactDir match
      case Some(dir) =>
        val artifact = PokerActionModelArtifactIO.load(dir)
        (artifact, artifact.version.id)
      case None =>
        val artifact = TrainedPokerActionModel(
          version = ModelVersion(
            id = "acpc-bootstrap-uniform",
            schemaVersion = "poker-action-model-v1",
            source = "acpc-match-runner-bootstrap",
            trainedAtEpochMillis = System.currentTimeMillis()
          ),
          model = PokerActionModel.uniform,
          calibration = CalibrationSummary(
            meanBrierScore = 0.75,
            sampleCount = 1,
            uniformBaselineBrier = 0.75,
            majorityBaselineBrier = 1.0
          ),
          gate = CalibrationGate(2.0),
          trainingSampleCount = 1,
          evaluationSampleCount = 1,
          evaluationStrategy = "bootstrap-uniform",
          validationFraction = None,
          splitSeed = None
        )
        (artifact, artifact.version.id)

  private def closeQuietly(resource: AutoCloseable): Unit =
    try resource.close()
    catch
      case _: Throwable => ()

  private def sendLine(writer: BufferedWriter, line: String): Unit =
    writer.write(line)
    writer.write("\r\n")
    writer.flush()

  private def writeLine(writer: BufferedWriter, line: String): Unit =
    writer.write(line)
    writer.newLine()
    writer.flush()

  private def roundedChips(value: Double): Int =
    math.max(50, math.round(value / 50.0).toInt * 50)

  private def renderAction(action: PokerAction): String =
    action match
      case PokerAction.Fold => "Fold"
      case PokerAction.Check => "Check"
      case PokerAction.Call => "Call"
      case PokerAction.Raise(amount) => s"Raise:${fmt(amount, 2)}"

  private def fmt(value: Double, digits: Int): String =
    String.format(Locale.ROOT, s"%.${digits}f", java.lang.Double.valueOf(value))

  private def heroModeLabel(mode: HeroMode): String =
    mode match
      case HeroMode.Adaptive => "adaptive"
      case HeroMode.Gto      => "gto"

  private def parseArgs(args: Array[String]): Either[String, Config] =
    if args.contains("--help") || args.contains("-h") then Left(usage)
    else
      for
        options <- CliHelpers.parseOptions(args)
        server <- options.get("server").toRight("--server is required")
        port <- CliHelpers.parseIntOptionEither(options, "port", -1)
        _ <- if port > 0 then Right(()) else Left("--port must be > 0")
        reportEvery <- CliHelpers.parseIntOptionEither(options, "reportEvery", 50)
        _ <- if reportEvery >= 0 then Right(()) else Left("--reportEvery must be >= 0")
        outDir = Paths.get(options.getOrElse("outDir", "data/acpc-match-runner"))
        modelDir <- parseOptionalPath(options, "model")
        heroMode <- heroModeOption(options, "heroMode", HeroMode.Adaptive)
        bunchingTrials <- CliHelpers.parseIntOptionEither(options, "bunchingTrials", 1)
        _ <- if bunchingTrials > 0 then Right(()) else Left("--bunchingTrials must be > 0")
        equityTrials <- CliHelpers.parseIntOptionEither(options, "equityTrials", 600)
        _ <- if equityTrials > 0 then Right(()) else Left("--equityTrials must be > 0")
        cfrIterations <- CliHelpers.parseIntOptionEither(options, "cfrIterations", 180)
        _ <- if cfrIterations > 0 then Right(()) else Left("--cfrIterations must be > 0")
        cfrVillainHands <- CliHelpers.parseIntOptionEither(options, "cfrVillainHands", 48)
        _ <- if cfrVillainHands > 0 then Right(()) else Left("--cfrVillainHands must be > 0")
        cfrEquityTrials <- CliHelpers.parseIntOptionEither(options, "cfrEquityTrials", 300)
        _ <- if cfrEquityTrials > 0 then Right(()) else Left("--cfrEquityTrials must be > 0")
        seed <- CliHelpers.parseLongOptionEither(options, "seed", 42L)
        timeoutMillis <- CliHelpers.parseIntOptionEither(options, "timeoutMillis", 15000)
        _ <- if timeoutMillis > 0 then Right(()) else Left("--timeoutMillis must be > 0")
        connectTimeoutMillis <- CliHelpers.parseIntOptionEither(options, "connectTimeoutMillis", 5000)
        _ <- if connectTimeoutMillis > 0 then Right(()) else Left("--connectTimeoutMillis must be > 0")
      yield
        Config(
          server = server,
          port = port,
          reportEvery = reportEvery,
          outDir = outDir,
          modelArtifactDir = modelDir,
          heroMode = heroMode,
          bunchingTrials = bunchingTrials,
          equityTrials = equityTrials,
          cfrIterations = cfrIterations,
          cfrVillainHands = cfrVillainHands,
          cfrEquityTrials = cfrEquityTrials,
          seed = seed,
          timeoutMillis = timeoutMillis,
          connectTimeoutMillis = connectTimeoutMillis
        )

  private def parseOptionalPath(options: Map[String, String], key: String): Either[String, Option[Path]] =
    options.get(key) match
      case None => Right(None)
      case Some(raw) =>
        val path = Paths.get(raw)
        if Files.isDirectory(path) then Right(Some(path))
        else Left(s"--$key: directory '$raw' does not exist")

  private def heroModeOption(
      options: Map[String, String],
      key: String,
      default: HeroMode
  ): Either[String, HeroMode] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) =>
        raw.trim.toLowerCase(Locale.ROOT) match
          case "adaptive" => Right(HeroMode.Adaptive)
          case "gto" => Right(HeroMode.Gto)
          case _ => Left("--heroMode must be one of: adaptive, gto")

  private val usage =
    """Usage:
      |  runMain sicfun.holdem.runtime.AcpcMatchRunner [--key=value ...]
      |
      |Notes:
      |  Supports ACPC heads-up no-limit reverse-blinds with 50/100 blinds and 200bb stacks.
      |
      |Options:
      |  --server=127.0.0.1          Dealer host or IP
      |  --port=12345                Dealer port for this seat
      |  --reportEvery=50            Progress report interval in completed hands (0 disables)
      |  --outDir=data/acpc-match-runner
      |  --model=<dir>               Optional saved action-model artifact directory
      |  --heroMode=adaptive         adaptive|gto
      |  --bunchingTrials=1          Posterior bunching trials
      |  --equityTrials=600          Equity trials for adaptive recommendations
      |  --cfrIterations=180         CFR iterations for heroMode=gto
      |  --cfrVillainHands=48        Max villain hands for CFR
      |  --cfrEquityTrials=300       Equity trials inside CFR
      |  --seed=42                   RNG seed
      |  --timeoutMillis=15000       Socket read timeout
      |  --connectTimeoutMillis=5000 Socket connect timeout
      |""".stripMargin
