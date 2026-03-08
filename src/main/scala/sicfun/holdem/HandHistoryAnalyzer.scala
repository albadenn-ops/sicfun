package sicfun.holdem

import java.nio.file.{Files, Path, Paths}
import scala.util.Random

/** Batch hand history analyzer.
  *
  * Loads a DecisionLoopEventFeedIO TSV file, replays each hand through a
  * [[RealTimeAdaptiveEngine]], and annotates each hero decision with the
  * recommended action and EV comparison.
  *
  * Usage:
  *   runMain sicfun.holdem.HandHistoryAnalyzer <feedFile> [--key=value ...]
  *
  * Options:
  *   --hero=<playerId>       Hero player ID (default "hero")
  *   --heroCards=<token>     Hero hole cards (e.g. AcKh) for EV/recommendation analysis
  *   --model=<dir>           Model artifact directory (defaults to uniform model if omitted)
  *   --seed=42               RNG seed
  *   --bunchingTrials=400    Bunching Monte Carlo trials
  *   --equityTrials=4000     Equity Monte Carlo trials
  *   --budgetMs=2000         Decision budget in milliseconds
  *   --topN=10               Show top N worst decisions
  */
object HandHistoryAnalyzer:

  final case class AnalyzedDecision(
      handId: String,
      street: Street,
      heroCards: Option[HoleCards],
      actualAction: PokerAction,
      recommendedAction: PokerAction,
      actualEv: Double,
      recommendedEv: Double,
      evDifference: Double,
      heroEquityMean: Double
  )

  final case class AnalysisSummary(
      handsAnalyzed: Int,
      decisionsAnalyzed: Int,
      mistakes: Int,
      totalEvLost: Double,
      biggestMistakeEv: Double,
      decisions: Vector[AnalyzedDecision]
  )

  def main(args: Array[String]): Unit =
    run(args) match
      case Left(err) =>
        System.err.println(err)
        sys.exit(1)
      case Right(summary) =>
        printSummary(summary)

  def run(args: Array[String]): Either[String, AnalysisSummary] =
    for
      config <- parseArgs(args)
      events <- loadEvents(config.feedPath)
      summary <- analyze(events, config)
    yield summary

  private final case class CliConfig(
      feedPath: Path,
      heroPlayerId: String,
      heroCards: Option[HoleCards],
      modelDir: Option[Path],
      seed: Long,
      bunchingTrials: Int,
      equityTrials: Int,
      budgetMs: Long,
      topN: Int
  )

  private def loadEvents(path: Path): Either[String, Vector[DecisionLoopEventFeedIO.FeedEvent]] =
    try Right(DecisionLoopEventFeedIO.read(path))
    catch case e: Exception => Left(s"Failed to read event feed: ${e.getMessage}")

  private def analyze(
      events: Vector[DecisionLoopEventFeedIO.FeedEvent],
      config: CliConfig
  ): Either[String, AnalysisSummary] =
    try
      val maybeModel = config.modelDir.map(path => PokerActionModelArtifactIO.load(path).model)
      val tableRanges = TableRanges.defaults(TableFormat.NineMax)
      val engine = new RealTimeAdaptiveEngine(
        tableRanges = tableRanges,
        actionModel = maybeModel.getOrElse(PokerActionModel.uniform),
        bunchingTrials = config.bunchingTrials,
        defaultEquityTrials = config.equityTrials,
        minEquityTrials = math.max(200, config.equityTrials / 10)
      )
      val seedRng = new Random(config.seed)

      // Group events by hand
      val handGroups = events
        .map(_.event)
        .groupBy(_.handId)
        .toVector
        .sortBy { case (_, groupedEvents) =>
          groupedEvents.map(_.occurredAtEpochMillis).min
        }

      val analyzed = Vector.newBuilder[AnalyzedDecision]
      var handsAnalyzed = 0

      handGroups.foreach { case (_, handEvents) =>
        val sorted = handEvents.sortBy(_.sequenceInHand)
        val heroEvents = sorted.filter(_.playerId == config.heroPlayerId)

        if heroEvents.nonEmpty then
          handsAnalyzed += 1
          config.heroCards.foreach { cards =>
            analyzed ++= analyzeWithHeroCards(
              events = sorted,
              heroPlayerId = config.heroPlayerId,
              heroCards = cards,
              engine = engine,
              tableRanges = tableRanges,
              budgetMs = config.budgetMs,
              rng = new Random(seedRng.nextLong())
            )
          }
      }

      val allDecisions = analyzed.result()
      val mistakes = allDecisions.count(d => d.actualAction.category != d.recommendedAction.category)
      val evLost = allDecisions.map(d => math.min(0.0, d.evDifference)).sum
      val biggestMistake = if allDecisions.nonEmpty then allDecisions.map(d => math.abs(d.evDifference)).max else 0.0

      Right(AnalysisSummary(
        handsAnalyzed = handsAnalyzed,
        decisionsAnalyzed = allDecisions.length,
        mistakes = mistakes,
        totalEvLost = evLost,
        biggestMistakeEv = biggestMistake,
        decisions = allDecisions.sortBy(d => (d.handId, d.street.toString))
      ))
    catch
      case e: Exception => Left(s"Analysis failed: ${e.getMessage}")

  /** Analyze events with known hero hole cards (for programmatic use). */
  def analyzeWithHeroCards(
      events: Vector[PokerEvent],
      heroPlayerId: String,
      heroCards: HoleCards,
      engine: RealTimeAdaptiveEngine,
      tableRanges: TableRanges,
      budgetMs: Long = 2000L,
      rng: Random = new Random()
  ): Vector[AnalyzedDecision] =
    val sorted = events.sortBy(_.sequenceInHand)
    val heroEvents = sorted.filter(_.playerId == heroPlayerId)
    val villainEvents = sorted.filter(_.playerId != heroPlayerId)
    val analyzed = Vector.newBuilder[AnalyzedDecision]

    heroEvents.foreach { heroEvent =>
      val priorVillainObs = villainEvents
        .filter(_.sequenceInHand < heroEvent.sequenceInHand)
        .filter(e => e.action != PokerAction.Fold)
        .map { e =>
          val obsState = GameState(
            street = e.street, board = e.board, pot = e.potBefore,
            toCall = e.toCall, position = e.position,
            stackSize = e.stackBefore, betHistory = e.betHistory
          )
          VillainObservation(e.action, obsState)
        }

      val gameState = GameState(
        street = heroEvent.street, board = heroEvent.board,
        pot = heroEvent.potBefore, toCall = heroEvent.toCall,
        position = heroEvent.position, stackSize = heroEvent.stackBefore,
        betHistory = heroEvent.betHistory
      )

      val candidates = buildCandidates(gameState)
      val allCandidates =
        if candidates.exists(_.category == heroEvent.action.category) then candidates
        else candidates :+ heroEvent.action

      val villainPos = villainEvents.headOption.map(_.position)
        .getOrElse(if heroEvent.position == Position.SmallBlind then Position.BigBlind else Position.SmallBlind)

      val openerPos = if heroEvent.position == Position.SmallBlind then Position.Button else Position.BigBlind
      val folds = tableRanges.format.foldsBeforeOpener(openerPos).map(PreflopFold(_))

      try
        val result = engine.decide(
          hero = heroCards,
          state = gameState,
          folds = folds,
          villainPos = villainPos,
          observations = priorVillainObs,
          candidateActions = allCandidates,
          decisionBudgetMillis = Some(budgetMs),
          rng = new Random(rng.nextLong())
        )

        val rec = result.decision.recommendation
        val recEv = rec.actionEvaluations
          .find(_.action == rec.bestAction).map(_.expectedValue).getOrElse(0.0)
        val actEv = rec.actionEvaluations
          .find(e => e.action.category == heroEvent.action.category).map(_.expectedValue).getOrElse(0.0)

        analyzed += AnalyzedDecision(
          handId = heroEvent.handId,
          street = heroEvent.street,
          heroCards = Some(heroCards),
          actualAction = heroEvent.action,
          recommendedAction = rec.bestAction,
          actualEv = actEv,
          recommendedEv = recEv,
          evDifference = actEv - recEv,
          heroEquityMean = rec.heroEquity.mean
        )
      catch
        case _: Exception => () // skip decisions that fail (e.g., overlapping cards)
    }
    analyzed.result()

  // ---- Output ----

  private def printSummary(summary: AnalysisSummary): Unit =
    println("=== Hand History Analysis ===")
    println(f"  Hands analyzed: ${summary.handsAnalyzed}")
    println(f"  Decisions analyzed: ${summary.decisionsAnalyzed}")
    println(f"  Mistakes: ${summary.mistakes} (${if summary.decisionsAnalyzed > 0 then summary.mistakes * 100.0 / summary.decisionsAnalyzed else 0.0}%.1f%%)")
    println(f"  Total EV lost: ${summary.totalEvLost}%.2f")
    println(f"  Biggest mistake: ${summary.biggestMistakeEv}%.2f")

    if summary.decisions.nonEmpty then
      println()
      println("  Per-decision breakdown:")
      summary.decisions.foreach { d =>
        val mark = if d.actualAction.category == d.recommendedAction.category then "OK" else "MISS"
        println(f"    ${d.handId} ${d.street}: actual=${renderAction(d.actualAction)} rec=${renderAction(d.recommendedAction)} ev_diff=${d.evDifference}%+.2f $mark")
      }

  // ---- Helpers ----

  private def buildCandidates(state: GameState): Vector[PokerAction] =
    val pot = state.pot
    val toCall = state.toCall
    val stack = state.stackSize
    if toCall <= 0.0 then
      val raises = Vector(0.5, 0.75, 1.0, 1.5).map(f => PokerAction.Raise(roundChips(pot * f)))
        .filter { case PokerAction.Raise(a) => a > 0.0 && a <= stack; case _ => false }.distinct
      Vector(PokerAction.Check) ++ raises
    else
      val basis = pot + toCall
      val raises = Vector(0.5, 0.75, 1.0, 1.5).map(f => PokerAction.Raise(roundChips(basis * f)))
        .filter { case PokerAction.Raise(a) => a > toCall && a <= stack; case _ => false }.distinct
      Vector(PokerAction.Fold, PokerAction.Call) ++ raises

  private def roundChips(v: Double): Double = math.round(v * 2.0) / 2.0

  private def renderAction(action: PokerAction): String = action match
    case PokerAction.Fold       => "FOLD"
    case PokerAction.Check      => "CHECK"
    case PokerAction.Call       => "CALL"
    case PokerAction.Raise(amt) => f"RAISE($amt%.1f)"

  // ---- CLI arg parsing ----

  private def parseArgs(args: Array[String]): Either[String, CliConfig] =
    if args.isEmpty || args.contains("--help") || args.contains("-h") then Left(usage)
    else
      val feedPath = Paths.get(args.head)
      if !Files.exists(feedPath) then Left(s"Feed file not found: ${args.head}")
      else
        val optArgs = args.tail
        for
          options <- parseOptions(optArgs)
          heroId = options.getOrElse("hero", "hero")
          heroCards <- parseOptionalHoleCards(options, "heroCards")
          modelDir <- parseOptionalPath(options, "model")
          seed <- parseLongOption(options, "seed", 42L)
          bunchingTrials <- parseIntOption(options, "bunchingTrials", 400)
          equityTrials <- parseIntOption(options, "equityTrials", 4000)
          budgetMs <- parseLongOption(options, "budgetMs", 2000L)
          topN <- parseIntOption(options, "topN", 10)
        yield CliConfig(
          feedPath = feedPath,
          heroPlayerId = heroId,
          heroCards = heroCards,
          modelDir = modelDir,
          seed = seed,
          bunchingTrials = bunchingTrials,
          equityTrials = equityTrials,
          budgetMs = budgetMs,
          topN = topN
        )

  private def parseOptions(args: Array[String]): Either[String, Map[String, String]] =
    val pairs = args.toVector.map { token =>
      if !token.startsWith("--") then Left(s"invalid argument '$token'; expected --key=value")
      else
        val body = token.drop(2)
        val idx = body.indexOf('=')
        if idx <= 0 || idx == body.length - 1 then Left(s"invalid argument '$token'; expected --key=value")
        else Right(body.substring(0, idx).trim -> body.substring(idx + 1).trim)
    }
    pairs.foldLeft(Right(Map.empty): Either[String, Map[String, String]]) {
      case (Left(err), _)            => Left(err)
      case (Right(_), Left(err))     => Left(err)
      case (Right(acc), Right(pair)) => Right(acc + pair)
    }

  private def parseIntOption(options: Map[String, String], key: String, default: Int): Either[String, Int] =
    options.get(key) match
      case None      => Right(default)
      case Some(raw) => raw.toIntOption.toRight(s"--$key must be an integer")

  private def parseLongOption(options: Map[String, String], key: String, default: Long): Either[String, Long] =
    options.get(key) match
      case None      => Right(default)
      case Some(raw) => raw.toLongOption.toRight(s"--$key must be a long")

  private def parseOptionalPath(options: Map[String, String], key: String): Either[String, Option[Path]] =
    options.get(key) match
      case None => Right(None)
      case Some(raw) =>
        val path = Paths.get(raw)
        if Files.isDirectory(path) then Right(Some(path)) else Left(s"--$key: directory '$raw' not found")

  private def parseOptionalHoleCards(options: Map[String, String], key: String): Either[String, Option[HoleCards]] =
    options.get(key) match
      case None => Right(None)
      case Some(raw) =>
        try Right(Some(CliHelpers.parseHoleCards(raw)))
        catch
          case e: IllegalArgumentException => Left(s"--$key: ${e.getMessage}")

  private val usage =
    """Usage:
      |  runMain sicfun.holdem.HandHistoryAnalyzer <feedFile> [--key=value ...]
      |
      |Options:
      |  --hero=<playerId>       Hero player ID (default "hero")
      |  --heroCards=<token>     Hero hole cards (e.g. AcKh) for EV/recommendation analysis
      |  --model=<dir>           Model artifact directory
      |  --seed=42               RNG seed
      |  --bunchingTrials=400    Bunching Monte Carlo trials
      |  --equityTrials=4000     Equity Monte Carlo trials
      |  --budgetMs=2000         Decision budget in milliseconds
      |  --topN=10               Show top N worst decisions
      |""".stripMargin
