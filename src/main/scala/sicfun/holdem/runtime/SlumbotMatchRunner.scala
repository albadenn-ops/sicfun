package sicfun.holdem.runtime

import sicfun.core.Card
import sicfun.holdem.cli.CliHelpers
import sicfun.holdem.engine.{EquilibriumBaselineConfig, HeroDecisionPipeline, VillainObservation}
import sicfun.holdem.equity.{PreflopFold, TableFormat, TableRanges}
import sicfun.holdem.model.{CalibrationGate, CalibrationSummary, ModelVersion, PokerActionModel, PokerActionModelArtifactIO, TrainedPokerActionModel}
import sicfun.holdem.types.*

import java.io.BufferedWriter
import java.net.URI
import java.net.http.{HttpClient, HttpRequest, HttpResponse}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import java.time.Duration
import java.util.Locale
import scala.util.Random

import ujson.{Num, Obj, Str, Value}

/** Slumbot action-string parser/encoder for heads-up no-limit Hold'em.
  *
  * Translates between the Slumbot HTTP API's compact action-string format and SICFUN's
  * internal [[PokerAction]] / [[GameState]] types.  The Slumbot format uses:
  *  - `k` for check, `c` for call, `f` for fold, `bNNN` for bet/raise to NNN chips.
  *  - `/` as the street delimiter (preflop/flop/turn/river).
  *  - Fixed 50/100 blinds with 20,000-chip stacks (200 BB).
  *
  * '''Key differences from ACPC protocol:'''
  *  - Check is `k` (not embedded in `c`).
  *  - Raise target is a street-relative chip amount after `b` (not a total hand contribution).
  *  - All values are in chips (not big blinds); conversion uses `chipsToBb` / `bbToChips`.
  *
  * '''Seat mapping:'''
  *  - Actual player 0 = big blind (out of position postflop).
  *  - Actual player 1 = button/small blind (in position postflop).
  *  - Preflop: button (player 1) acts first; postflop: big blind (player 0) acts first.
  *
  * @see [[SlumbotMatchRunner]] for the match runner that uses this codec.
  */
private[holdem] object SlumbotActionCodec:
  val SmallBlindChips = 50
  val BigBlindChips = 100
  val StackSizeChips = 20000

  /** A single parsed action step within a Slumbot action string.
    *
    * Captures the actor, the action taken, and the full game state before the action was applied
    * (for villain observation recording and archetype learning).
    */
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

  /** Fully reconstructed state after parsing a Slumbot action string.
    *
    * Contains the current street, board, pot, to-call, stack remaining, per-player chip
    * contributions, bet history, and all parsed action steps.  Also tracks whether the
    * hand is over (fold or all-in resolution) and who acts next.
    */
  final case class ParsedActionState(
      action: String,
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

  /** Parse a Slumbot action string into a fully reconstructed [[ParsedActionState]].
    *
    * Walks the action string character by character, maintaining a complete betting state
    * machine (street contributions, bet levels, min-raise tracking, street advancement).
    * Each action token (`k`/`c`/`f`/`bNNN`) produces an [[ActionStep]] with the pre-action
    * state snapshot.
    *
    * @param action     Raw Slumbot action string (e.g., "b200c/kb300").
    * @param heroActual Actual player index for the hero (0 = BB, 1 = BTN).
    * @param fullBoard  The full dealt board (up to 5 cards); sliced per street.
    * @return           `Right(ParsedActionState)` on success, `Left(error)` on illegal input.
    */
  def parse(
      action: String,
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
      var streetContribution = Array(BigBlindChips, SmallBlindChips)
      val totalContribution = Array(BigBlindChips, SmallBlindChips)
      var betHistory = Vector.empty[BetAction]
      val steps = Vector.newBuilder[ActionStep]

      var i = 0
      while i < action.length && !handOver do
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
        action.charAt(i) match
          case 'k' =>
            require(stateBefore.toCall == 0.0, "illegal check in Slumbot action string")
            val stepAction = PokerAction.Check
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
            if checkOrCallEndsStreet then
              if i < action.length then
                require(action.charAt(i) == '/', "missing slash after street-ending check")
                i += 1
              if streetIdx >= Street.values.length - 1 then
                handOver = true
                nextActor = -1
              else
                streetIdx += 1
                nextActor = 0
                streetContribution = Array(0, 0)
                streetLastBetTo = 0
                lastBetSize = 0
                checkOrCallEndsStreet = false
            else
              nextActor = actor ^ 1
              checkOrCallEndsStreet = true

          case 'c' =>
            require(stateBefore.toCall > 0.0, "illegal call in Slumbot action string")
            val toCallChips = bbToChips(stateBefore.toCall)
            val remaining = StackSizeChips - totalContribution(actor)
            val paid = math.min(toCallChips, remaining)
            require(paid > 0, "call must contribute chips")
            streetContribution(actor) += paid
            totalContribution(actor) += paid
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
            i += 1
            lastBetSize = 0
            val allInResolved =
              totalContribution(actor) >= StackSizeChips || totalContribution(actor ^ 1) >= StackSizeChips
            if allInResolved then
              while i < action.length do
                require(action.charAt(i) == '/', "unexpected token after all-in call")
                i += 1
              handOver = true
              nextActor = -1
              streetIdx = math.max(streetIdx, streetIndexForBoard(fullBoard))
            else if checkOrCallEndsStreet then
              if i < action.length then
                require(action.charAt(i) == '/', "missing slash after street-ending call")
                i += 1
              if streetIdx >= Street.values.length - 1 then
                handOver = true
                nextActor = -1
              else
                streetIdx += 1
                nextActor = 0
                streetContribution = Array(0, 0)
                streetLastBetTo = 0
                checkOrCallEndsStreet = false
            else
              nextActor = actor ^ 1
              checkOrCallEndsStreet = true

          case 'f' =>
            require(stateBefore.toCall > 0.0, "illegal fold in Slumbot action string")
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
            require(i == action.length, "unexpected token after fold")
            handOver = true
            nextActor = -1

          case 'b' =>
            i += 1
            val start = i
            while i < action.length && action.charAt(i).isDigit do i += 1
            require(i > start, "missing bet size in Slumbot action string")
            val newStreetBetTo = action.substring(start, i).toInt
            val actorStreetContribution = streetContribution(actor)
            val toCallChips = streetLastBetTo - actorStreetContribution
            require(toCallChips >= 0, s"negative toCall while parsing action '$action'")
            require(newStreetBetTo > streetLastBetTo, "raise target must exceed current bet level")
            val extraIncrement = newStreetBetTo - streetLastBetTo
            val paid = newStreetBetTo - actorStreetContribution
            val remaining = StackSizeChips - totalContribution(actor)
            require(paid > toCallChips, "raise must add chips beyond call")
            require(paid <= remaining, "raise exceeds stack")
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
            streetContribution(actor) = newStreetBetTo
            totalContribution(actor) += paid
            streetLastBetTo = newStreetBetTo
            lastBetSize = extraIncrement
            nextActor = actor ^ 1
            checkOrCallEndsStreet = true

          case other =>
            throw new IllegalArgumentException(s"unexpected token '$other' in Slumbot action string")

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
          action = action,
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

  /** Encode a [[PokerAction]] as a Slumbot action increment string for the API.
    *
    * Maps Fold -> "f", Check/Call -> "k"/"c" (based on to-call), and Raise -> "bNNN"
    * where NNN is the new street-level bet-to in chips.
    */
  def incrementForAction(parsed: ParsedActionState, action: PokerAction): String =
    action match
      case PokerAction.Fold => "f"
      case PokerAction.Check =>
        if parsed.toCallChips <= 0 then "k" else "c"
      case PokerAction.Call =>
        if parsed.toCallChips > 0 then "c" else "k"
      case PokerAction.Raise(amountBb) =>
        val incrementChips = bbToChips(amountBb)
        val target = parsed.streetLastBetToChips + incrementChips
        s"b$target"

  def positionForActual(actualActor: Int): Position =
    actualActor match
      case 1 => Position.Button
      case 0 => Position.BigBlind
      case other => throw new IllegalArgumentException(s"invalid player index: $other")

  def chipsToBb(chips: Int): Double =
    chips.toDouble / BigBlindChips.toDouble

  def bbToChips(bb: Double): Int =
    math.max(0, math.round(bb * BigBlindChips.toDouble).toInt)

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

/** Match runner that plays heads-up hands against the Slumbot HTTP API.
  *
  * Slumbot (slumbot.com) is a publicly available poker bot built on neural-network CFR.
  * This runner connects to its JSON API, plays a configurable number of hands, and logs
  * decisions and outcomes for offline analysis.
  *
  * '''Match lifecycle:'''
  *  1. Call `/api/new_hand` to start a session and receive hole cards + client position.
  *  2. Parse the Slumbot action string to reconstruct game state.
  *  3. When it's hero's turn, run the decision pipeline (adaptive or GTO) and send the
  *     chosen action via `/api/act`.
  *  4. Repeat until the hand is terminal (winnings field appears in the response).
  *  5. Record the outcome and start the next hand using the session token.
  *
  * '''Hero modes:'''
  *  - `Adaptive` — Uses [[RealTimeAdaptiveEngine]] with Bayesian range inference and
  *    archetype learning from villain raise responses.
  *  - `Gto` — Uses [[HeroDecisionPipeline.decideHero]] with CFR-based equilibrium solving.
  *
  * '''Output artifacts:'''
  *  - `hands.tsv` — Per-hand log with position, hole cards, board, winnings, and action string.
  *  - `decisions.tsv` — Per-decision log with state, candidates, chosen action, and wire action.
  *  - `summary.txt` — Aggregate statistics via [[MatchRunnerSupport.writeSummary]].
  *
  * '''Resilience:'''
  *  - HTTP requests retry up to 3 times with 300ms backoff on failure.
  *  - The API client validates response structure and reports Slumbot-side errors.
  *
  * @see [[SlumbotActionCodec]] for the action string parser/encoder.
  * @see [[AcpcMatchRunner]] for the similar runner that uses the ACPC TCP protocol.
  */
object SlumbotMatchRunner:
  private final case class Config(
      hands: Int,
      reportEvery: Int,
      outDir: Path,
      modelArtifactDir: Option[Path],
      heroMode: HeroMode,
      baseUrl: String,
      bunchingTrials: Int,
      equityTrials: Int,
      cfrIterations: Int,
      cfrVillainHands: Int,
      cfrEquityTrials: Int,
      seed: Long,
      timeoutMillis: Long,
      enableBaseline: Boolean,
      baselineBlendWeight: Double,
      decisionBudgetMillis: Long
  )

  /** Parsed JSON response from a Slumbot API call.
    *
    * @param oldAction   Action string before the latest server-side action (for delta tracking).
    * @param action      Current cumulative action string including all actions so far.
    * @param clientPos   Hero's actual player index (0 = BB, 1 = BTN).
    * @param holeCards   Hero's hole cards as 2-char tokens (e.g., ["As", "Ah"]).
    * @param board       Community cards as 2-char tokens (grows as streets are dealt).
    * @param token       Session token for continuing the match (passed to subsequent calls).
    * @param winnings    Present only when the hand is terminal; hero's signed chip result.
    */
  private final case class SlumbotApiResponse(
      oldAction: String,
      action: String,
      clientPos: Int,
      holeCards: Vector[String],
      board: Vector[String],
      token: Option[String],
      winnings: Option[Int],
      oppHoleCards: Vector[String]
  )

  /** Result of a single completed Slumbot hand, used for logging and statistics. */
  private final case class HandOutcome(
      token: String,
      heroPosition: Position,
      heroHole: HoleCards,
      villainHole: Option[HoleCards],
      board: Board,
      finalAction: String,
      winningsChips: Int,
      decisionCount: Int,
      villainObservationCount: Int,
      villainObservations: Vector[VillainObservation]
  )

  /** HTTP client wrapper for the Slumbot JSON API.
    *
    * Handles serialization/deserialization with ujson, retries (up to 3 attempts with
    * 300ms backoff), and error extraction from the `error_msg` response field.
    */
  private final class SlumbotApiClient(baseUrl: String, timeoutMillis: Long):
    private val client = HttpClient.newBuilder()
      .connectTimeout(Duration.ofMillis(math.max(1000L, timeoutMillis)))
      .build()

    def newHand(token: Option[String]): SlumbotApiResponse =
      post(
        path = "/api/new_hand",
        payload =
          token match
            case Some(value) => Obj("token" -> Str(value))
            case None => Obj()
      )

    def act(token: String, increment: String): SlumbotApiResponse =
      post(
        path = "/api/act",
        payload = Obj("token" -> Str(token), "incr" -> Str(increment))
      )

    private def post(path: String, payload: Value): SlumbotApiResponse =
      var attempt = 0
      var lastError: Option[Throwable] = None
      while attempt < 3 do
        try
          val request = HttpRequest.newBuilder()
            .uri(URI.create(s"${baseUrl.stripSuffix("/")}$path"))
            .timeout(Duration.ofMillis(math.max(1000L, timeoutMillis)))
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(ujson.write(payload), StandardCharsets.UTF_8))
            .build()
          val response = client.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8))
          if response.statusCode() != 200 then
            throw new IllegalStateException(s"HTTP ${response.statusCode()} for $path: ${response.body()}")
          return parseResponse(ujson.read(response.body()))
        catch
          case error: Throwable =>
            lastError = Some(error)
            attempt += 1
            if attempt < 3 then Thread.sleep(300L)
      throw new IllegalStateException(lastError.map(_.getMessage).getOrElse("unknown Slumbot API error"))

    private def parseResponse(json: Value): SlumbotApiResponse =
      val obj = json.obj
      obj.get("error_msg").foreach(value => throw new IllegalStateException(value.str))
      SlumbotApiResponse(
        oldAction = obj.get("old_action").map(_.str).getOrElse(""),
        action = obj.get("action").map(_.str).getOrElse(""),
        clientPos = obj.get("client_pos").map(asInt).getOrElse(
          throw new IllegalStateException("Slumbot response missing client_pos")
        ),
        holeCards = obj.get("hole_cards").map(asStringVector).getOrElse(Vector.empty),
        board = obj.get("board").map(asStringVector).getOrElse(Vector.empty),
        token = obj.get("token").map(_.str),
        winnings = obj.get("winnings").map(asInt),
        oppHoleCards = obj.get("bot_hole_cards").map(asStringVector).getOrElse(Vector.empty)
      )

    private def asInt(value: Value): Int =
      value match
        case Num(number) => math.round(number).toInt
        case Str(text) => text.toInt
        case other => throw new IllegalStateException(s"expected integer JSON value, got $other")

    private def asStringVector(value: Value): Vector[String] =
      value.arr.iterator.map(_.str).toVector

  def main(args: Array[String]): Unit =
    val wantsHelp = args.contains("--help") || args.contains("-h")
    run(args) match
      case Right(summary) =>
        println("=== Slumbot Match Runner ===")
        println(s"handsPlayed: ${summary.handsPlayed}")
        println(s"heroNetChips: ${summary.heroNetChips.toInt}")
        println(s"heroBbPer100: ${PokerFormatting.fmtDouble(summary.heroBbPer100, 3)}")
        println(s"heroWins: ${summary.heroWins}")
        println(s"heroTies: ${summary.heroTies}")
        println(s"heroLosses: ${summary.heroLosses}")
        println(s"buttonHands: ${summary.buttonHands}")
        println(s"buttonNetChips: ${summary.buttonNetChips.toInt}")
        println(s"bigBlindHands: ${summary.bigBlindHands}")
        println(s"bigBlindNetChips: ${summary.bigBlindNetChips.toInt}")
        println(s"modelId: ${summary.modelId}")
        println(s"outDir: ${summary.outDir.toAbsolutePath.normalize()}")
      case Left(error) =>
        if wantsHelp then println(error)
        else
          System.err.println(error)
          sys.exit(1)

  def run(args: Array[String]): Either[String, MatchRunnerSupport.RunSummary] =
    parseArgs(args).flatMap(config => new Runner(config).run())

  /** Orchestrates a Slumbot match: plays hands via the API, runs hero decisions,
    * logs outcomes and decision traces, and writes a final summary.
    */
  private final class Runner(config: Config):
    private val handsPath = config.outDir.resolve("hands.tsv")
    private val decisionsPath = config.outDir.resolve("decisions.tsv")
    private val showdownsPath = config.outDir.resolve("showdowns.tsv")
    private val summaryPath = config.outDir.resolve("summary.txt")

    private val tableRanges = TableRanges.defaults(TableFormat.NineMax)
    private val folds = TableFormat.NineMax.foldsBeforeOpener(Position.Button).map(PreflopFold(_))
    private val rng = new Random(config.seed)

    private var handsWriterOpt = Option.empty[BufferedWriter]
    private var decisionsWriterOpt = Option.empty[BufferedWriter]
    private var showdownsWriterOpt = Option.empty[BufferedWriter]

    private val stats = new MatchRunnerSupport.MatchStatistics()

    private val (artifact, modelId) = loadArtifact(config)
    private val baselineConfig =
      if config.enableBaseline then
        Some(EquilibriumBaselineConfig(
          iterations = config.cfrIterations,
          blendWeight = config.baselineBlendWeight,
          maxVillainHands = config.cfrVillainHands,
          equityTrials = config.cfrEquityTrials
        ))
      else None
    private val engine = HeroDecisionPipeline.newAdaptiveEngine(
      tableRanges = tableRanges,
      model = artifact.model,
      bunchingTrials = config.bunchingTrials,
      equityTrials = config.equityTrials,
      equilibriumBaselineConfig = baselineConfig
    )
    private val api = new SlumbotApiClient(config.baseUrl, config.timeoutMillis)

    def run(): Either[String, MatchRunnerSupport.RunSummary] =
      try
        Files.createDirectories(config.outDir)
        handsWriterOpt = Some(Files.newBufferedWriter(handsPath, StandardCharsets.UTF_8))
        decisionsWriterOpt = Some(Files.newBufferedWriter(decisionsPath, StandardCharsets.UTF_8))
        showdownsWriterOpt = Some(Files.newBufferedWriter(showdownsPath, StandardCharsets.UTF_8))
        writeLine(handsWriter, "hand\theroPosition\theroHole\tvillainHole\tboard\twinningsChips\twinningsBb\tdecisions\tvillainObservations\taction")
        writeLine(
          decisionsWriter,
          "hand\tdecisionIndex\tstreet\theroPosition\tpotBeforeBb\ttoCallBb\tstackBb\tcandidates\tchosenAction\tapiAction"
        )
        writeLine(showdownsWriter, "street\tboard\tpotBefore\ttoCall\tposition\tstackBefore\taction\tholeCards")

        var tokenOpt = Option.empty[String]
        var handNo = 1
        while handNo <= config.hands do
          val outcome = playHand(handNo, tokenOpt)
          tokenOpt = Some(outcome.token)
          recordOutcome(outcome)
          appendHandLog(handNo, outcome)
          appendShowdownTrainingData(outcome)
          maybeReport(handNo)
          handNo += 1

        val summary = buildSummary()
        writeSummary(summary)
        Right(summary)
      catch
        case error: Exception =>
          Left(s"slumbot match runner failed: ${error.getMessage}")
      finally
        handsWriterOpt.foreach(closeQuietly)
        decisionsWriterOpt.foreach(closeQuietly)
        showdownsWriterOpt.foreach(closeQuietly)

    /** Play a single hand against Slumbot.
      *
      * Calls `newHand` to get hole cards and position, then loops: parse the action string,
      * collect villain observations, and when it's hero's turn, run the decision pipeline
      * and send the action via `act`.  Continues until the response contains `winnings`.
      *
      * Villain raise-response tracking: when hero raises and the next villain action arrives,
      * it's fed to the engine for archetype learning.
      */
    private def playHand(handNo: Int, tokenOpt: Option[String]): HandOutcome =
      var response = api.newHand(tokenOpt)
      var token = response.token.orElse(tokenOpt).getOrElse(
        throw new IllegalStateException("Slumbot did not return a session token")
      )
      val heroHole = parseHoleCards(response.holeCards)
      val heroActual = response.clientPos
      val heroPosition = SlumbotActionCodec.positionForActual(heroActual)
      val villainPosition =
        if heroPosition == Position.Button then Position.BigBlind
        else Position.Button
      var villainObservations = Vector.empty[VillainObservation]
      var processedSteps = 0
      var decisionIndex = 0
      var pendingHeroRaise = false

      while true do
        val board = parseBoard(response.board)
        val parsed =
          SlumbotActionCodec
            .parse(response.action, heroActual = heroActual, fullBoard = board)
            .fold(error => throw new IllegalStateException(s"failed to parse Slumbot action '${response.action}': $error"), identity)
        val newSteps = parsed.steps.drop(processedSteps)
        processedSteps = parsed.steps.length

        newSteps.foreach { step =>
          if step.relativeActor == 1 then
            villainObservations = villainObservations :+ VillainObservation(step.action, step.stateBefore)
            if pendingHeroRaise then
              step.action match
                case PokerAction.Fold | PokerAction.Call | PokerAction.Raise(_) =>
                  engine.observeVillainResponseToRaise(step.action)
                  pendingHeroRaise = false
                case _ => ()
          else
            step.action match
              case PokerAction.Raise(_) => pendingHeroRaise = true
              case _ => ()
        }

        response.winnings match
          case Some(winnings) =>
            val villainHole =
              if response.oppHoleCards.length == 2 then
                Some(parseHoleCards(response.oppHoleCards))
              else None
            return HandOutcome(
              token = token,
              heroPosition = heroPosition,
              heroHole = heroHole,
              villainHole = villainHole,
              board = board,
              finalAction = response.action,
              winningsChips = winnings,
              decisionCount = decisionIndex,
              villainObservationCount = villainObservations.length,
              villainObservations = villainObservations
            )
          case None =>
            if parsed.nextActorActual != heroActual then
              throw new IllegalStateException(
                s"Slumbot response is not terminal and not hero-to-act: action='${response.action}' clientPos=$heroActual"
              )
            val heroState = parsed.nextDecisionState.getOrElse(
              throw new IllegalStateException("missing hero decision state")
            )
            val candidates = heroCandidates(parsed)
            val chosenAction = decideHero(
              hero = heroHole,
              state = heroState,
              villainPosition = villainPosition,
              villainObservations = villainObservations,
              candidates = candidates
            )
            val apiAction = SlumbotActionCodec.incrementForAction(parsed, chosenAction)
            decisionIndex += 1
            appendDecisionLog(handNo, decisionIndex, heroState, candidates, chosenAction, apiAction)
            response = api.act(token, apiAction)
            token = response.token.getOrElse(token)

      throw new IllegalStateException("unreachable")

    private def decideHero(
        hero: HoleCards,
        state: GameState,
        villainPosition: Position,
        villainObservations: Vector[VillainObservation],
        candidates: Vector[PokerAction]
    ): PokerAction =
      HeroDecisionPipeline.decideHero(
        config.heroMode,
        HeroDecisionPipeline.HeroDecisionContext(
          hero = hero,
          state = state,
          folds = folds,
          tableRanges = tableRanges,
          villainPos = villainPosition,
          observations = villainObservations,
          candidates = candidates,
          engine = engine,
          actionModel = artifact.model,
          bunchingTrials = config.bunchingTrials,
          cfrIterations = config.cfrIterations,
          cfrVillainHands = config.cfrVillainHands,
          cfrEquityTrials = config.cfrEquityTrials,
          rng = rng,
          decisionBudgetMillis = Some(config.decisionBudgetMillis)
        )
      )

    private def heroCandidates(parsed: SlumbotActionCodec.ParsedActionState): Vector[PokerAction] =
      val raises = legalRaiseCandidates(parsed)
      HeroDecisionPipeline.heroCandidates(toCallChips = parsed.toCallChips, raises = raises)

    private def legalRaiseCandidates(parsed: SlumbotActionCodec.ParsedActionState): Vector[PokerAction] =
      HeroDecisionPipeline.legalRaiseCandidates(
        HeroDecisionPipeline.RaiseSizingContext(
          stackRemainingChips = parsed.stackRemainingChips,
          toCallChips = parsed.toCallChips,
          lastBetSizeChips = parsed.lastBetSizeChips,
          potChips = parsed.potChips,
          currentStreet = parsed.currentStreet,
          streetLastBetToChips = parsed.streetLastBetToChips,
          bigBlindChips = SlumbotActionCodec.BigBlindChips
        )
      )

    private def recordOutcome(outcome: HandOutcome): Unit =
      stats.recordOutcome(outcome.heroPosition, outcome.winningsChips.toDouble)

    private def appendHandLog(handNo: Int, outcome: HandOutcome): Unit =
      writeLine(
        handsWriter,
        Vector(
          handNo.toString,
          outcome.heroPosition.toString,
          outcome.heroHole.toToken,
          outcome.villainHole.map(_.toToken).getOrElse("-"),
          outcome.board.cards.map(_.toToken).mkString,
          outcome.winningsChips.toString,
          PokerFormatting.fmtDouble(outcome.winningsChips.toDouble / SlumbotActionCodec.BigBlindChips.toDouble, 3),
          outcome.decisionCount.toString,
          outcome.villainObservationCount.toString,
          outcome.finalAction
        ).mkString("\t")
      )

    private def appendDecisionLog(
        handNo: Int,
        decisionIndex: Int,
        state: GameState,
        candidates: Vector[PokerAction],
        chosenAction: PokerAction,
        apiAction: String
    ): Unit =
      MatchRunnerSupport.appendDecisionLog(
        writer = decisionsWriter,
        handId = handNo,
        decisionIndex = decisionIndex,
        state = state,
        candidates = candidates,
        chosenAction = chosenAction,
        wireAction = apiAction
      )

    private def maybeReport(handNo: Int): Unit =
      if config.reportEvery > 0 && (handNo % config.reportEvery == 0 || handNo == config.hands) then
        println(
          s"[slumbot] hands=${stats.currentHandsPlayed} netChips=${stats.currentHeroNetChips.toInt} bb100=${PokerFormatting.fmtDouble(stats.currentBbPer100(SlumbotActionCodec.BigBlindChips.toDouble), 3)} mode=${PokerFormatting.heroModeLabel(config.heroMode)} model=$modelId"
        )

    private def buildSummary(): MatchRunnerSupport.RunSummary =
      stats.buildSummary(heroMode = config.heroMode, modelId = modelId, outDir = config.outDir, bigBlindChips = SlumbotActionCodec.BigBlindChips)

    private def writeSummary(summary: MatchRunnerSupport.RunSummary): Unit =
      MatchRunnerSupport.writeSummary(summaryPath, "Slumbot Match Runner", summary)

    private def handsWriter: BufferedWriter =
      handsWriterOpt.getOrElse(throw new IllegalStateException("hands writer not initialized"))

    private def decisionsWriter: BufferedWriter =
      decisionsWriterOpt.getOrElse(throw new IllegalStateException("decisions writer not initialized"))

    private def showdownsWriter: BufferedWriter =
      showdownsWriterOpt.getOrElse(throw new IllegalStateException("showdowns writer not initialized"))

    /** Write villain observation training data for hands that reached showdown.
      *
      * For each villain action in a showdown hand, writes a row to showdowns.tsv
      * pairing the action with the villain's revealed hole cards.  This produces
      * supervised training data for the action model: P(action | hand, state).
      */
    private def appendShowdownTrainingData(outcome: HandOutcome): Unit =
      outcome.villainHole.foreach { vHole =>
        outcome.villainObservations.foreach { obs =>
          val state = obs.state
          val boardStr =
            if state.board.size == 0 then "-"
            else state.board.cards.map(_.toToken).mkString(" ")
          val actionStr = PokerFormatting.renderAction(obs.action)
          writeLine(
            showdownsWriter,
            Vector(
              state.street.toString,
              boardStr,
              PokerFormatting.fmtDouble(state.pot, 3),
              PokerFormatting.fmtDouble(state.toCall, 3),
              state.position.toString,
              PokerFormatting.fmtDouble(state.stackSize, 3),
              actionStr,
              vHole.toToken
            ).mkString("\t")
          )
        }
      }

  private def loadArtifact(config: Config): (TrainedPokerActionModel, String) =
    config.modelArtifactDir match
      case Some(dir) =>
        val artifact = PokerActionModelArtifactIO.load(dir)
        (artifact, artifact.version.id)
      case None =>
        val artifact = TrainedPokerActionModel(
          version = ModelVersion(
            id = "slumbot-bootstrap-uniform",
            schemaVersion = "poker-action-model-v1",
            source = "slumbot-match-runner-bootstrap",
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

  private def parseHoleCards(tokens: Vector[String]): HoleCards =
    val cards = parseCards(tokens)
    require(cards.length == 2, s"expected 2 hole cards from Slumbot, got ${tokens.mkString(",")}")
    HoleCards.from(cards)

  private def parseBoard(tokens: Vector[String]): Board =
    Board.from(parseCards(tokens))

  private def parseCards(tokens: Vector[String]): Vector[Card] =
    tokens.map { token =>
      Card.parse(token).getOrElse(
        throw new IllegalArgumentException(s"invalid card token from Slumbot: $token")
      )
    }

  private def closeQuietly(writer: BufferedWriter): Unit =
    try writer.close()
    catch
      case _: Throwable => ()

  private def writeLine(writer: BufferedWriter, line: String): Unit =
    writer.write(line)
    writer.newLine()
    writer.flush()

  private def parseArgs(args: Array[String]): Either[String, Config] =
    if args.contains("--help") || args.contains("-h") then Left(usage)
    else
      for
        options <- CliHelpers.parseOptions(args)
        hands <- CliHelpers.parseIntOptionEither(options, "hands", 100)
        _ <- if hands > 0 then Right(()) else Left("--hands must be > 0")
        reportEvery <- CliHelpers.parseIntOptionEither(options, "reportEvery", 10)
        _ <- if reportEvery >= 0 then Right(()) else Left("--reportEvery must be >= 0")
        outDir <- parseOutDir(options)
        modelDir <- parseOptionalPath(options, "model")
        heroMode <- heroModeOption(options, "heroMode", HeroMode.Adaptive)
        baseUrl = options.getOrElse("baseUrl", "https://slumbot.com")
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
        timeoutMillis <- CliHelpers.parseLongOptionEither(options, "timeoutMillis", 15000L)
        _ <- if timeoutMillis > 0L then Right(()) else Left("--timeoutMillis must be > 0")
        enableBaseline = options.get("enableBaseline").exists(_.toLowerCase(Locale.ROOT) == "true")
        baselineBlendWeight <- CliHelpers.parseDoubleOptionEither(options, "baselineBlendWeight", 0.35)
        _ <- if baselineBlendWeight >= 0.0 && baselineBlendWeight <= 1.0 then Right(()) else Left("--baselineBlendWeight must be in [0, 1]")
        decisionBudgetMillis <- CliHelpers.parseLongOptionEither(options, "decisionBudgetMillis", 1L)
        _ <- if decisionBudgetMillis > 0L then Right(()) else Left("--decisionBudgetMillis must be > 0")
      yield
        Config(
          hands = hands,
          reportEvery = reportEvery,
          outDir = outDir,
          modelArtifactDir = modelDir,
          heroMode = heroMode,
          baseUrl = baseUrl,
          bunchingTrials = bunchingTrials,
          equityTrials = equityTrials,
          cfrIterations = cfrIterations,
          cfrVillainHands = cfrVillainHands,
          cfrEquityTrials = cfrEquityTrials,
          seed = seed,
          timeoutMillis = timeoutMillis,
          enableBaseline = enableBaseline,
          baselineBlendWeight = baselineBlendWeight,
          decisionBudgetMillis = decisionBudgetMillis
        )

  private def parseOutDir(options: Map[String, String]): Either[String, Path] =
    Right(Paths.get(options.getOrElse("outDir", "data/slumbot-match-runner")))

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
      |  runMain sicfun.holdem.runtime.SlumbotMatchRunner [--key=value ...]
      |
      |Options:
      |  --hands=100                 Number of Slumbot hands to play
      |  --reportEvery=10            Progress report interval (0 disables)
      |  --outDir=data/slumbot-match-runner
      |  --model=<dir>               Optional saved action-model artifact directory
      |  --heroMode=adaptive         adaptive|gto
      |  --baseUrl=https://slumbot.com
      |  --bunchingTrials=1          Posterior bunching trials
      |  --equityTrials=600          Equity trials for adaptive recommendations
      |  --cfrIterations=180         CFR iterations for heroMode=gto
      |  --cfrVillainHands=48        Max villain hands for CFR
      |  --cfrEquityTrials=300       Equity trials inside CFR
      |  --seed=42                   RNG seed
      |  --timeoutMillis=15000       HTTP timeout per request
      |""".stripMargin
