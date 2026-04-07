package sicfun.holdem.runtime
import sicfun.holdem.types.*
import sicfun.holdem.model.*
import sicfun.holdem.engine.*
import sicfun.holdem.equity.*
import sicfun.holdem.io.*
import sicfun.holdem.cli.*

import java.nio.file.{Files, Path, Paths}
import scala.util.Random

/** Batch hand history analyzer that replays recorded hands and evaluates hero decisions.
  *
  * '''Purpose:'''
  * Loads a [[DecisionLoopEventFeedIO]] TSV file (produced by [[AlwaysOnDecisionLoop]] or
  * manual conversion), replays each hand through a fresh [[RealTimeAdaptiveEngine]], and
  * annotates each hero decision point with the engine's recommended action and EV comparison.
  * This identifies "mistakes" — spots where the hero's actual action had lower EV than the
  * engine's recommendation — and quantifies total EV lost.
  *
  * '''Analysis pipeline:'''
  *  1. Group events by hand ID and sort by timestamp.
  *  2. For each hand containing hero events, create a fresh engine with the appropriate
  *     table format (based on the number of distinct positions observed).
  *  3. Replay villain observations up to each hero decision point for range inference.
  *  4. Run the engine's `decide` method to get the recommended action and EV for all candidates.
  *  5. Compare the hero's actual action EV to the recommended action EV.
  *  6. Aggregate into an [[AnalysisSummary]] with mistake count, total EV lost, and
  *     per-decision breakdown.
  *
  * '''Villain resolution:'''
  * In multiway hands, the analyzer picks the "active villain" — the player with the most
  * recent non-fold action — for range inference.  In heads-up, this is always the single opponent.
  *
  * '''EV matching:'''
  * When the hero's actual action doesn't exactly match a candidate (e.g., a raise to a
  * slightly different amount), [[expectedValueForObservedAction]] falls back to the
  * nearest-amount raise candidate, then to category matching.
  *
  * Usage:
  *   runMain sicfun.holdem.runtime.HandHistoryAnalyzer <feedFile> [--key=value ...]
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

  /** A single analyzed hero decision, comparing the actual action to the engine's recommendation.
    *
    * @param evDifference  `actualEv - recommendedEv`: negative means the hero's action was worse.
    * @param heroEquityMean  Hero's estimated equity (0..1) at this decision point.
    */
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

  /** Look up the EV for the hero's actual action from the engine's candidate evaluations.
    *
    * First tries exact match.  If not found (e.g., raise to a different amount), falls
    * back to the nearest-amount raise candidate, then to action-category matching
    * (e.g., any Raise if the exact amount isn't a candidate).  Returns 0.0 if no match.
    */
  def expectedValueForObservedAction(
      actionEvaluations: Vector[ActionEvaluation],
      action: PokerAction
  ): Double =
    actionEvaluations
      .find(_.action == action)
      .map(_.expectedValue)
      .orElse {
        action match
          case PokerAction.Raise(amount) =>
            actionEvaluations
              .collect { case ActionEvaluation(PokerAction.Raise(candidateAmount), expectedValue) =>
                math.abs(candidateAmount - amount) -> expectedValue
              }
              .sortBy(_._1)
              .headOption
              .map(_._2)
          case _ =>
            actionEvaluations
              .find(_.action.category == action.category)
              .map(_.expectedValue)
      }
      .getOrElse(0.0)

  /** A decision counts as a mistake if the EV difference is negative after rounding
    * to 2 decimal places (to avoid flagging trivially small differences as errors).
    */
  def countsAsMistake(decision: AnalyzedDecision): Boolean =
    roundDecisionEv(decision.evDifference) < 0.0

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
      Right(new AnalysisRunner(events, config).run())
    catch
      case e: Exception => Left(s"Analysis failed: ${e.getMessage}")

  private final class AnalysisRunner(
      events: Vector[DecisionLoopEventFeedIO.FeedEvent],
      config: CliConfig
  ):
    private val actionModel =
      config.modelDir.map(path => PokerActionModelArtifactIO.load(path).model).getOrElse(PokerActionModel.uniform)
    private val seedRng = new Random(config.seed)

    def run(): AnalysisSummary =
      val analyzed = Vector.newBuilder[AnalyzedDecision]
      var handsAnalyzed = 0
      groupedHandEvents().foreach { handEvents =>
        if handEvents.exists(_.playerId == config.heroPlayerId) then
          handsAnalyzed += 1
          analyzed ++= analyzeHand(handEvents)
      }
      buildSummary(handsAnalyzed, analyzed.result())

    private def groupedHandEvents(): Vector[Vector[PokerEvent]] =
      events
        .map(_.event)
        .groupBy(_.handId)
        .toVector
        .sortBy { case (_, groupedEvents) =>
          groupedEvents.map(_.occurredAtEpochMillis).min
        }
        .map { case (_, groupedEvents) =>
          groupedEvents.sortBy(_.sequenceInHand)
        }

    private def analyzeHand(handEvents: Vector[PokerEvent]): Vector[AnalyzedDecision] =
      val availablePositions = handEvents.iterator.map(_.position).toSet
      val tableRanges = TableRanges.defaults(TableFormat.forPlayerCount(math.max(2, availablePositions.size)))
      config.heroCards match
        case Some(cards) =>
          analyzeWithHeroCards(
            events = handEvents,
            heroPlayerId = config.heroPlayerId,
            heroCards = cards,
            engine = new RealTimeAdaptiveEngine(
              tableRanges = tableRanges,
              actionModel = actionModel,
              bunchingTrials = config.bunchingTrials,
              defaultEquityTrials = config.equityTrials,
              minEquityTrials = math.max(200, config.equityTrials / 10)
            ),
            tableRanges = tableRanges,
            availablePositions = availablePositions,
            budgetMs = config.budgetMs,
            rng = new Random(seedRng.nextLong())
          )
        case None => Vector.empty

    private def buildSummary(
        handsAnalyzed: Int,
        decisions: Vector[AnalyzedDecision]
    ): AnalysisSummary =
      val mistakes = decisions.count(countsAsMistake)
      val evLost = decisions.map(d => math.min(0.0, d.evDifference)).sum
      val biggestMistake =
        if decisions.nonEmpty then decisions.map(d => math.abs(d.evDifference)).max
        else 0.0
      AnalysisSummary(
        handsAnalyzed = handsAnalyzed,
        decisionsAnalyzed = decisions.length,
        mistakes = mistakes,
        totalEvLost = evLost,
        biggestMistakeEv = biggestMistake,
        decisions = decisions.sortBy(d => (d.handId, d.street.toString))
      )

  /** Analyze a single hand's events with known hero hole cards (for programmatic / test use).
    *
    * Creates a [[HeroDecisionAnalyzer]], replays all villain observations, and evaluates
    * each hero action against the engine's recommendation.  This is the main entry point
    * for tests and external callers who already have parsed events.
    *
    * @param events              Poker events for one hand, sorted by sequence.
    * @param heroPlayerId        Player ID string identifying the hero in the events.
    * @param heroCards           Hero's known hole cards.
    * @param engine              A fresh [[RealTimeAdaptiveEngine]] for this hand.
    * @param tableRanges         Table ranges appropriate for the table size.
    * @param availablePositions  Positions present in the hand (for preflop fold inference).
    * @param budgetMs            Decision time budget in milliseconds.
    * @param rng                 RNG for equity sampling.
    * @return                    Vector of analyzed decisions, one per hero action.
    */
  def analyzeWithHeroCards(
      events: Vector[PokerEvent],
      heroPlayerId: String,
      heroCards: HoleCards,
      engine: RealTimeAdaptiveEngine,
      tableRanges: TableRanges,
      availablePositions: Set[Position] = Set.empty,
      budgetMs: Long = 2000L,
      rng: Random = new Random()
  ): Vector[AnalyzedDecision] =
    new HeroDecisionAnalyzer(
      events = events,
      heroPlayerId = heroPlayerId,
      heroCards = heroCards,
      engine = engine,
      tableRanges = tableRanges,
      availablePositions = availablePositions,
      budgetMs = budgetMs,
      rng = rng
    ).analyze()

  /** Inner worker that analyzes all hero decisions within a single hand.
    *
    * Separates hero events from villain events, identifies the active villain player
    * (most significant opponent), and for each hero decision point:
    * 1. Collects all prior villain observations for range inference.
    * 2. Infers preflop folds from non-hero/non-villain players.
    * 3. Builds candidate actions (including the hero's actual action if not in the default set).
    * 4. Runs the engine's `decide` to get EV for each candidate.
    * 5. Builds an [[AnalyzedDecision]] comparing actual vs. recommended.
    */
  private final class HeroDecisionAnalyzer(
      events: Vector[PokerEvent],
      heroPlayerId: String,
      heroCards: HoleCards,
      engine: RealTimeAdaptiveEngine,
      tableRanges: TableRanges,
      availablePositions: Set[Position],
      budgetMs: Long,
      rng: Random
  ):
    private val sortedEvents = events.sortBy(_.sequenceInHand)
    private val heroEvents = sortedEvents.filter(_.playerId == heroPlayerId)
    private val villainEventsByPlayer =
      sortedEvents
        .filter(_.playerId != heroPlayerId)
        .groupBy(_.playerId)
        .view
        .mapValues(_.sortBy(_.sequenceInHand))
        .toMap
    private val activeVillainPlayerId = resolveActiveVillainPlayerId()
    private val activeVillainEvents =
      activeVillainPlayerId
        .flatMap(villainEventsByPlayer.get)
        .getOrElse(Vector.empty)
    private val eligiblePositions =
      if availablePositions.nonEmpty then availablePositions
      else tableRanges.format.preflopOrder.toSet

    def analyze(): Vector[AnalyzedDecision] =
      val analyzed = Vector.newBuilder[AnalyzedDecision]
      heroEvents.foreach { heroEvent =>
        analyzeDecision(heroEvent).foreach(analyzed += _)
      }
      analyzed.result()

    private def analyzeDecision(heroEvent: PokerEvent): Option[AnalyzedDecision] =
      try
        val recommendation = engine.decide(
          hero = heroCards,
          state = gameStateFor(heroEvent),
          folds = preflopFoldsFor(heroEvent),
          villainPos = villainPositionFor(heroEvent),
          observations = priorVillainObservations(heroEvent),
          candidateActions = candidateActionsFor(heroEvent),
          decisionBudgetMillis = Some(budgetMs),
          rng = new Random(rng.nextLong())
        ).decision.recommendation
        Some(buildDecision(heroEvent, recommendation))
      catch
        case _: Exception => None

    /** Pick the most "active" villain by sorting on: most recent non-fold action,
      * then most recent any-action, then most total actions.  In heads-up this is trivial;
      * in multiway it picks the primary opponent for range inference.
      */
    private def resolveActiveVillainPlayerId(): Option[String] =
      villainEventsByPlayer.toVector.sortBy { case (_, playerEvents) =>
        val latestNonFold =
          playerEvents.filter(_.action != PokerAction.Fold).lastOption.map(_.sequenceInHand).getOrElse(Long.MinValue)
        val latestAny = playerEvents.lastOption.map(_.sequenceInHand).getOrElse(Long.MinValue)
        val actionCount = playerEvents.length.toLong
        (-latestNonFold, -latestAny, -actionCount)
      }.headOption.map(_._1)

    private def priorVillainObservations(heroEvent: PokerEvent): Seq[VillainObservation] =
      activeVillainEvents
        .filter(_.sequenceInHand < heroEvent.sequenceInHand)
        .filter(_.action != PokerAction.Fold)
        .map(event => VillainObservation(event.action, gameStateFor(event)))

    private def gameStateFor(event: PokerEvent): GameState =
      GameState(
        street = event.street,
        board = event.board,
        pot = event.potBefore,
        toCall = event.toCall,
        position = event.position,
        stackSize = event.stackBefore,
        betHistory = event.betHistory
      )

    private def candidateActionsFor(heroEvent: PokerEvent): Vector[PokerAction] =
      val candidates = buildCandidates(gameStateFor(heroEvent))
      if candidates.exists(_.category == heroEvent.action.category) then candidates
      else candidates :+ heroEvent.action

    private def villainPositionFor(heroEvent: PokerEvent): Position =
      activeVillainEvents.headOption.map(_.position)
        .getOrElse(if heroEvent.position == Position.BigBlind then Position.Button else Position.BigBlind)

    private def preflopFoldsFor(heroEvent: PokerEvent): Vector[PreflopFold] =
      val actualPreflopFolds =
        sortedEvents
          .filter(_.street == Street.Preflop)
          .filter(event => event.playerId != heroPlayerId)
          .filter(event => !activeVillainPlayerId.contains(event.playerId))
          .filter(_.action == PokerAction.Fold)
          .map(_.position)
          .filter(eligiblePositions.contains)
          .distinct
      if actualPreflopFolds.nonEmpty then actualPreflopFolds.map(PreflopFold(_))
      else
        val openerPos =
          sortedEvents
            .filter(_.street == Street.Preflop)
            .filter(event => event.playerId == heroPlayerId || activeVillainPlayerId.contains(event.playerId))
            .headOption
            .map(_.position)
            .getOrElse(tableRanges.format.preflopOrder.head)
        tableRanges.format
          .foldsBeforeOpener(openerPos)
          .filter(eligiblePositions.contains)
          .map(PreflopFold(_))

    private def buildDecision(
        heroEvent: PokerEvent,
        recommendation: ActionRecommendation
    ): AnalyzedDecision =
      val recommendedEv = expectedValueFor(recommendation, recommendation.bestAction)
      val actualEv = expectedValueForObservedAction(recommendation.actionEvaluations, heroEvent.action)
      AnalyzedDecision(
        handId = heroEvent.handId,
        street = heroEvent.street,
        heroCards = Some(heroCards),
        actualAction = heroEvent.action,
        recommendedAction = recommendation.bestAction,
        actualEv = actualEv,
        recommendedEv = recommendedEv,
        evDifference = actualEv - recommendedEv,
        heroEquityMean = recommendation.heroEquity.mean
      )

    private def expectedValueFor(
        recommendation: ActionRecommendation,
        action: PokerAction
    ): Double =
      recommendation.actionEvaluations
        .find(_.action == action)
        .map(_.expectedValue)
        .getOrElse(0.0)

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
        val mark = if countsAsMistake(d) then "MISS" else "OK"
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

  private def roundDecisionEv(v: Double): Double =
    math.round(v * 100.0) / 100.0

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
          options <- CliHelpers.parseOptions(optArgs)
          heroId = options.getOrElse("hero", "hero")
          heroCards <- parseOptionalHoleCards(options, "heroCards")
          modelDir <- parseOptionalPath(options, "model")
          seed <- CliHelpers.parseLongOptionEither(options, "seed", 42L)
          bunchingTrials <- CliHelpers.parseIntOptionEither(options, "bunchingTrials", 400)
          equityTrials <- CliHelpers.parseIntOptionEither(options, "equityTrials", 4000)
          budgetMs <- CliHelpers.parseLongOptionEither(options, "budgetMs", 2000L)
          topN <- CliHelpers.parseIntOptionEither(options, "topN", 10)
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
      |  runMain sicfun.holdem.runtime.HandHistoryAnalyzer <feedFile> [--key=value ...]
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
