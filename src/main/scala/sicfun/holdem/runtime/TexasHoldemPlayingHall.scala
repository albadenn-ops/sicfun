package sicfun.holdem.runtime
import sicfun.holdem.types.*
import sicfun.holdem.model.*
import sicfun.holdem.engine.*
import sicfun.holdem.engine.GtoSolveEngine.{GtoMode, GtoSolveCacheKey, GtoCachedPolicy, GtoCacheStats}
import sicfun.holdem.provider.*
import sicfun.holdem.equity.*
import sicfun.holdem.cli.*

import sicfun.core.{Card, CardId, Deck, DiscreteDistribution, HandEvaluator}
import sicfun.holdem.validation.{
  ActionLine, EquityBasedStrategy, InjectedLeak, LeakInjectedVillain,
  NoLeak, Overcalls, OverbluffsTurnBarrel, OverfoldsToAggression, PassiveInBigPots,
  PreflopTooLoose, PreflopTooTight, SpotContext
}

import java.io.BufferedWriter
import java.net.URLEncoder
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.Locale
import scala.collection.mutable
import scala.util.Random

/** Large-volume self-play hall:
  *  - plays full hands (preflop -> river)
  *  - runs Bayesian range inference + action recommendation for hero decisions
  *  - logs hands + learning checkpoints
  *  - periodically retrains villain action model from generated data
  */
object TexasHoldemPlayingHall:
  private enum VillainMode:
    case Archetype(style: PlayerArchetype)
    case Gto
    case LeakInjected(leakId: String, severity: Double)

  private final case class VillainProfile(
      name: String,
      mode: VillainMode,
      label: String
  )

  private final case class Config(
      hands: Int,
      tableCount: Int,
      playerCount: Int,
      reportEvery: Int,
      learnEveryHands: Int,
      learningWindowSamples: Int,
      seed: Long,
      outDir: Path,
      modelArtifactDir: Option[Path],
      heroMode: HeroMode,
      heroPosition: Position,
      gtoMode: GtoMode,
      villainPool: Vector[VillainProfile],
      heroExplorationRate: Double,
      raiseSize: Double,
      bunchingTrials: Int,
      equityTrials: Int,
      saveTrainingTsv: Boolean,
      saveDdreTrainingTsv: Boolean,
      saveReviewHandHistory: Boolean,
      fullRing: Boolean
  )

  final case class HallSummary(
      handsPlayed: Int,
      tableCount: Int,
      playerCount: Int,
      heroNetChips: Double,
      heroBbPer100: Double,
      heroWins: Int,
      heroTies: Int,
      heroLosses: Int,
      actionCounts: Map[String, Int],
      retrains: Int,
      modelId: String,
      outDir: Path,
      exactGtoCacheHits: Long = 0L,
      exactGtoCacheMisses: Long = 0L,
      exactGtoSolvedByProvider: Map[String, Long] = Map.empty,
      exactGtoServedByProvider: Map[String, Long] = Map.empty
  ):
    def exactGtoCacheTotal: Long = exactGtoCacheHits + exactGtoCacheMisses
    def exactGtoCacheHitRate: Double =
      if exactGtoCacheTotal > 0 then exactGtoCacheHits.toDouble / exactGtoCacheTotal.toDouble
      else 0.0

  private final case class Deal(
      holeCardsByPosition: Map[Position, HoleCards],
      board: Board
  ):
    def holeCardsFor(position: Position): HoleCards = holeCardsByPosition(position)

  private final case class TableScenario(
      playerCount: Int,
      modeledPositions: Vector[Position],
      activePositions: Vector[Position],
      heroPosition: Position,
      primaryVillainPosition: Position,
      openerPosition: Position,
      foldedPositions: Vector[Position],
      seatNumberByPosition: Map[Position, Int],
      playerNameByPosition: Map[Position, String],
      villainProfileByPosition: Map[Position, VillainProfile]
  ):
    def heroSeatNumber: Int = seatNumberByPosition(heroPosition)
    def villainSeatNumber: Int = seatNumberByPosition(primaryVillainPosition)
    def buttonSeatNumber: Int = seatNumberByPosition(Position.Button)
    def nameFor(position: Position): String = playerNameByPosition(position)
    def activeVillainPositions: Vector[Position] =
      activePositions.filterNot(_ == heroPosition)
    def primaryVillainProfile: VillainProfile = villainProfileByPosition(primaryVillainPosition)
    def fallbackFoldsFor(
        targetPosition: Position,
        perspectivePosition: Position
    ): Vector[PreflopFold] =
      val realFolds =
        foldedPositions
          .distinct
          .filterNot(position => position == perspectivePosition || position == targetPosition)
      if realFolds.nonEmpty then realFolds.map(PreflopFold(_))
      else
        TableFormat
          .forPlayerCount(playerCount)
          .foldsBeforeOpener(openerPosition)
          .filter(modeledPositions.contains)
          .filterNot(position => position == perspectivePosition || position == targetPosition)
          .map(PreflopFold(_))

  private final case class HandResult(
      heroNet: Double,
      outcome: Int,
      tableScenario: TableScenario,
      villainDecision: Option[(GameState, PokerAction)],
      villainTrainingSamples: Vector[(GameState, HoleCards, PokerAction)],
      ddreTrainingSamples: Vector[DdreTrainingSample],
      raiseResponses: Vector[PokerAction],
      heroActions: Vector[PokerAction],
      villainActions: Vector[PokerAction],
      streetsPlayed: Int,
      reviewHistoryLines: Vector[String],
      maxLivePlayers: Int
  )

  private final case class ShowdownResolution(
      heroPayout: Double
  )

  private final case class DdreTrainingSample(
      decisionIndex: Int,
      state: GameState,
      villainPosition: Position,
      villainStackBefore: Double,
      observations: Vector[VillainObservation],
      heroHole: HoleCards,
      villainHole: HoleCards,
      prior: DiscreteDistribution[HoleCards],
      bayesPosterior: DiscreteDistribution[HoleCards],
      bayesLogEvidence: Double
  )

  private[holdem] final case class RaiseResponseEstimate(
      foldProbability: Double,
      continueProbability: Double,
      continuationRange: DiscreteDistribution[HoleCards]
  )

  private val SmallBlindAmount = 0.5
  private val BigBlindAmount = 1.0
  private val ReviewUploadFileName = "review-upload-pokerstars.txt"
  private val ReviewHeroName = "Hero"
  private val ReviewStartingStack = 100.0
  private val MoneyEpsilon = 1e-9
  private val MultiwayExactMaxEvaluations = 250_000L
  private val ReviewBaseTimestamp = LocalDateTime.of(2026, 3, 10, 12, 0, 0)
  private val ReviewTimestampFormatter = DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss")

  def main(args: Array[String]): Unit =
    val wantsHelp = args.contains("--help") || args.contains("-h")
    run(args) match
      case Right(summary) =>
        println("=== Texas Hold'em Playing Hall ===")
        println(s"handsPlayed: ${summary.handsPlayed}")
        println(s"tableCount: ${summary.tableCount}")
        println(s"playerCount: ${summary.playerCount}")
        println(f"heroNetChips: ${summary.heroNetChips}%.4f")
        println(f"heroBbPer100: ${summary.heroBbPer100}%.3f")
        println(s"heroWins: ${summary.heroWins}")
        println(s"heroTies: ${summary.heroTies}")
        println(s"heroLosses: ${summary.heroLosses}")
        println(s"actionCounts: ${summary.actionCounts}")
        println(s"retrains: ${summary.retrains}")
        if summary.exactGtoCacheTotal > 0 then
          println(s"exactGtoCacheHits: ${summary.exactGtoCacheHits}")
          println(s"exactGtoCacheMisses: ${summary.exactGtoCacheMisses}")
          println(f"exactGtoCacheHitRate: ${summary.exactGtoCacheHitRate * 100.0}%.1f%%")
          println(s"exactGtoSolvedByProvider: ${formatLongCountMap(summary.exactGtoSolvedByProvider)}")
          println(s"exactGtoServedByProvider: ${formatLongCountMap(summary.exactGtoServedByProvider)}")
        println(s"modelId: ${summary.modelId}")
        println(s"outDir: ${summary.outDir.toAbsolutePath.normalize()}")
      case Left(error) =>
        if wantsHelp then println(error)
        else
          System.err.println(error)
          sys.exit(1)

  def run(args: Array[String]): Either[String, HallSummary] =
    parseArgs(args).flatMap(runConfig)

  private def runConfig(config: Config): Either[String, HallSummary] =
    new HallRunner(config).run()

  private final class HallRunner(config: Config):
    private val modelsRoot = config.outDir.resolve("models")
    private val handsPath = config.outDir.resolve("hands.tsv")
    private val learningPath = config.outDir.resolve("learning.tsv")
    private val trainingPath = config.outDir.resolve("training-selfplay.tsv")
    private val ddrePath = config.outDir.resolve("ddre-training-selfplay.tsv")
    private val reviewUploadPath = config.outDir.resolve(ReviewUploadFileName)

    private var handsWriterOpt: Option[BufferedWriter] = None
    private var learningWriterOpt: Option[BufferedWriter] = None
    private var trainingWriterOpt: Option[BufferedWriter] = None
    private var ddreWriterOpt: Option[BufferedWriter] = None
    private var reviewWriterOpt: Option[BufferedWriter] = None

    private val tableRanges = TableRanges.defaults(TableFormat.forPlayerCount(config.playerCount))
    private val rng = new Random(config.seed)
    private val learningQueue = mutable.Queue.empty[(GameState, HoleCards, PokerAction)]
    private val raiseResponseHistory = mutable.ArrayBuffer.empty[PokerAction]
    private val exactGtoCache = mutable.HashMap.empty[GtoSolveCacheKey, GtoCachedPolicy]
    private val exactGtoCacheStats = GtoCacheStats()
    private val collectVillainTraining = config.learnEveryHands > 0 || config.saveTrainingTsv
    private val collectDdreTraining = config.saveDdreTrainingTsv

    private var activeArtifactOpt = Option.empty[TrainedPokerActionModel]
    private var preflopEngineOpt = Option.empty[RealTimeAdaptiveEngine]
    private var postflopEngineOpt = Option.empty[RealTimeAdaptiveEngine]
    private var heroNet = 0.0
    private var heroWins = 0
    private var heroTies = 0
    private var heroLosses = 0
    private val actionCounts = mutable.Map.empty[String, Int].withDefaultValue(0)
    private var retrains = 0

    def run(): Either[String, HallSummary] =
      try
        Files.createDirectories(config.outDir)
        Files.createDirectories(modelsRoot)
        handsWriterOpt = Some(openLogWriter(handsPath))
        learningWriterOpt = Some(openLogWriter(learningPath))
        trainingWriterOpt =
          if config.saveTrainingTsv then Some(openLogWriter(trainingPath))
          else None
        ddreWriterOpt =
          if config.saveDdreTrainingTsv then Some(openLogWriter(ddrePath))
          else None
        reviewWriterOpt =
          if config.saveReviewHandHistory then Some(openLogWriter(reviewUploadPath))
          else None
        initHandsLog(handsWriter)
        initLearningLog(learningWriter)
        trainingWriter.foreach(initTrainingTsv)
        ddreWriter.foreach(initDdreTrainingTsv)
        initializeArtifact()
        playHands()
        trimExactCacheIfNeeded()
        Right(buildSummary())
      catch
        case e: Exception =>
          Left(s"playing hall failed: ${e.getMessage}")
      finally
        handsWriterOpt.foreach(closeQuietly)
        learningWriterOpt.foreach(closeQuietly)
        trainingWriterOpt.foreach(closeQuietly)
        ddreWriterOpt.foreach(closeQuietly)
        reviewWriterOpt.foreach(closeQuietly)

    private def handsWriter: BufferedWriter =
      handsWriterOpt.getOrElse(throw new IllegalStateException("hands writer not initialized"))

    private def learningWriter: BufferedWriter =
      learningWriterOpt.getOrElse(throw new IllegalStateException("learning writer not initialized"))

    private def trainingWriter: Option[BufferedWriter] = trainingWriterOpt

    private def ddreWriter: Option[BufferedWriter] = ddreWriterOpt

    private def reviewWriter: Option[BufferedWriter] = reviewWriterOpt

    private def activeArtifact: TrainedPokerActionModel =
      activeArtifactOpt.getOrElse(throw new IllegalStateException("playing hall artifact not initialized"))

    private def preflopEngine: RealTimeAdaptiveEngine =
      preflopEngineOpt.getOrElse(throw new IllegalStateException("preflop engine not initialized"))

    private def postflopEngine: RealTimeAdaptiveEngine =
      postflopEngineOpt.getOrElse(throw new IllegalStateException("postflop engine not initialized"))

    private def initializeArtifact(): Unit =
      val (initialArtifact, _) = loadInitialArtifact(config, modelsRoot)
      activeArtifactOpt = Some(initialArtifact)
      rebuildEngines()

    private def playHands(): Unit =
      var handNo = 1
      while handNo <= config.hands do
        playHand(handNo)
        handNo += 1

    private def playHand(handNo: Int): Unit =
      val tableId = ((handNo - 1) % config.tableCount) + 1
      val tableScenario = buildTableScenario(
        playerCount = config.playerCount,
        heroPosition = config.heroPosition,
        villainPool = config.villainPool,
        headsUpOnly = config.saveReviewHandHistory && !config.fullRing,
        forceAllActive = config.fullRing,
        rng = rng
      )
      val deal = dealHand(tableScenario.modeledPositions, rng)
      val result = resolveHand(
        deal = deal,
        preflopEngine = preflopEngine,
        postflopEngine = postflopEngine,
        tableRanges = tableRanges,
        actionModel = activeArtifact.model,
        config = config,
        tableScenario = tableScenario,
        rng = new Random(rng.nextLong()),
        collectVillainTraining = collectVillainTraining,
        collectDdreTraining = collectDdreTraining,
        exactGtoCache = exactGtoCache,
        exactGtoCacheStats = exactGtoCacheStats
      )

      recordTrainingSamples(handNo, tableId, result)
      raiseResponseHistory ++= result.raiseResponses
      recordOutcome(result)
      recordHeroActions(result)
      appendHandLog(
        writer = handsWriter,
        hand = handNo,
        tableId = tableId,
        deal = deal,
        result = result,
        modelId = activeArtifact.version.id,
        archetype = tableScenario.primaryVillainProfile.label,
        playerCount = tableScenario.playerCount,
        heroPosition = tableScenario.heroPosition,
        villainPosition = result.tableScenario.primaryVillainPosition,
        openerPosition = tableScenario.openerPosition,
        foldedPositions = tableScenario.foldedPositions,
        activePositions = tableScenario.activePositions,
        maxLivePlayers = result.maxLivePlayers
      )
      reviewWriter.foreach { writer =>
        appendReviewHandHistory(
          writer = writer,
          hand = handNo,
          tableId = tableId,
          startedAt = ReviewBaseTimestamp.plusMinutes(handNo.toLong - 1L),
          deal = deal,
          result = result,
          tableScenario = tableScenario
        )
      }
      maybeReport(handNo)
      maybeRetrain(handNo)

    private def recordTrainingSamples(handNo: Int, tableId: Int, result: HandResult): Unit =
      result.villainTrainingSamples.foreach { sample =>
        learningQueue.enqueue(sample)
        if config.learningWindowSamples > 0 then
          while learningQueue.size > config.learningWindowSamples do learningQueue.dequeue()
        trainingWriter.foreach(w => appendTrainingSample(w, handNo, tableId, sample))
      }
      result.ddreTrainingSamples.foreach { sample =>
        ddreWriter.foreach(w => appendDdreTrainingSample(w, handNo, tableId, sample))
      }

    private def recordOutcome(result: HandResult): Unit =
      heroNet += result.heroNet
      if result.outcome > 0 then heroWins += 1
      else if result.outcome < 0 then heroLosses += 1
      else heroTies += 1

    private def recordHeroActions(result: HandResult): Unit =
      result.heroActions.foreach { action =>
        val key = renderAction(action)
        actionCounts.update(key, actionCounts(key) + 1)
      }

    private def maybeReport(handNo: Int): Unit =
      if config.reportEvery > 0 && (handNo % config.reportEvery == 0 || handNo == config.hands) then
        val bb100 = if handNo > 0 then (heroNet / handNo.toDouble) * 100.0 else 0.0
        println(
          f"[hall] hand=$handNo%,d net=${heroNet}%.2f bb100=$bb100%.2f retrains=$retrains model=${activeArtifact.version.id}"
        )

    private def maybeRetrain(handNo: Int): Unit =
      if config.learnEveryHands > 0 &&
        handNo % config.learnEveryHands == 0 &&
        handNo < config.hands &&
        learningQueue.nonEmpty
      then
        val trainingData = learningQueue.toVector
        val candidate = PokerActionModel.trainVersioned(
          trainingData = trainingData,
          learningRate = 0.08,
          iterations = 180,
          l2Lambda = 0.001,
          validationFraction = 0.2,
          splitSeed = config.seed + handNo.toLong,
          maxMeanBrierScore = 2.0,
          failOnGate = false,
          modelId = s"hall-$handNo",
          schemaVersion = "poker-action-model-v1",
          source = "texas-holdem-playing-hall",
          trainedAtEpochMillis = System.currentTimeMillis()
        )
        val candidateDir = modelsRoot.resolve(s"model-$handNo")
        PokerActionModelArtifactIO.save(candidateDir, candidate)
        appendLearningLog(
          writer = learningWriter,
          hand = handNo,
          modelId = candidate.version.id,
          meanBrierScore = candidate.calibration.meanBrierScore,
          gatePassed = candidate.gatePassed,
          sampleCount = trainingData.length
        )
        if candidate.gatePassed then
          activeArtifactOpt = Some(candidate)
          retrains += 1
          rebuildEngines()
          raiseResponseHistory.foreach { response =>
            preflopEngine.observeVillainResponseToRaise(response)
            postflopEngine.observeVillainResponseToRaise(response)
          }

    private def rebuildEngines(): Unit =
      preflopEngineOpt = Some(buildEngine(
        tableRanges = tableRanges,
        model = activeArtifact.model,
        bunchingTrials = config.bunchingTrials,
        equityTrials = config.equityTrials
      ))
      postflopEngineOpt = Some(buildEngine(
        tableRanges = tableRanges,
        model = activeArtifact.model,
        bunchingTrials = postflopBunchingTrials(config.bunchingTrials),
        equityTrials = postflopEquityTrials(config.equityTrials)
      ))

    private def trimExactCacheIfNeeded(): Unit =
      if exactGtoCache.size > GtoSolveEngine.MaxGtoCacheEntries then
        exactGtoCache.clear()

    private def buildSummary(): HallSummary =
      val bbPer100 =
        if config.hands > 0 then (heroNet / config.hands.toDouble) * 100.0
        else 0.0
      HallSummary(
        handsPlayed = config.hands,
        tableCount = config.tableCount,
        playerCount = config.playerCount,
        heroNetChips = heroNet,
        heroBbPer100 = bbPer100,
        heroWins = heroWins,
        heroTies = heroTies,
        heroLosses = heroLosses,
        actionCounts = actionCounts.toMap,
        retrains = retrains,
        modelId = activeArtifact.version.id,
        outDir = config.outDir,
        exactGtoCacheHits = exactGtoCacheStats.hits,
        exactGtoCacheMisses = exactGtoCacheStats.misses,
        exactGtoSolvedByProvider = exactGtoCacheStats.solvedByProviderSnapshot,
        exactGtoServedByProvider = exactGtoCacheStats.servedByProviderSnapshot
      )

  private def resolveHand(
      deal: Deal,
      preflopEngine: RealTimeAdaptiveEngine,
      postflopEngine: RealTimeAdaptiveEngine,
      tableRanges: TableRanges,
      actionModel: PokerActionModel,
      config: Config,
      tableScenario: TableScenario,
      rng: Random,
      collectVillainTraining: Boolean,
      collectDdreTraining: Boolean,
      exactGtoCache: mutable.HashMap[GtoSolveCacheKey, GtoCachedPolicy],
      exactGtoCacheStats: GtoCacheStats
  ): HandResult =
    new HandResolver(
      deal = deal,
      preflopEngine = preflopEngine,
      postflopEngine = postflopEngine,
      tableRanges = tableRanges,
      actionModel = actionModel,
      config = config,
      tableScenario = tableScenario,
      rng = rng,
      collectVillainTraining = collectVillainTraining,
      collectDdreTraining = collectDdreTraining,
      exactGtoCache = exactGtoCache,
      exactGtoCacheStats = exactGtoCacheStats
    ).play()

  private final class HandResolver(
      deal: Deal,
      preflopEngine: RealTimeAdaptiveEngine,
      postflopEngine: RealTimeAdaptiveEngine,
      tableRanges: TableRanges,
      actionModel: PokerActionModel,
      config: Config,
      tableScenario: TableScenario,
      rng: Random,
      collectVillainTraining: Boolean,
      collectDdreTraining: Boolean,
      exactGtoCache: mutable.HashMap[GtoSolveCacheKey, GtoCachedPolicy],
      exactGtoCacheStats: GtoCacheStats
  ):
    private val heroPosition = tableScenario.heroPosition
    private val preflopOrder = tableScenario.modeledPositions
    private val postflopOrder = tableScenario.modeledPositions.sortBy(postflopOrderIndex)
    private val participatingPositions = tableScenario.activePositions.toSet
    private val contributionByPosition =
      mutable.HashMap.from(tableScenario.modeledPositions.map { position =>
        position -> blindContributionFor(position, tableScenario.playerCount)
      })
    private val stackByPosition =
      mutable.HashMap.from(tableScenario.modeledPositions.map { position =>
        position -> (ReviewStartingStack - contributionByPosition(position))
      })
    private val actionLineByPosition = mutable.HashMap.empty[Position, Vector[PokerAction]]
      .withDefaultValue(Vector.empty)
    private val foldedPositions = mutable.HashSet.empty[Position]
    private val allInPositions = mutable.HashSet.empty[Position]
    private val preflopFoldedPositions = mutable.ArrayBuffer.empty[Position]
    private var pot = roundMoney(contributionByPosition.values.sum)
    private var betHistory = Vector.empty[BetAction]

    private val villainTrainingSamples = mutable.ArrayBuffer.empty[(GameState, HoleCards, PokerAction)]
    private val ddreTrainingSamples = mutable.ArrayBuffer.empty[DdreTrainingSample]
    private val raiseResponses = mutable.ArrayBuffer.empty[PokerAction]
    private val heroActions = mutable.ArrayBuffer.empty[PokerAction]
    private val villainActions = mutable.ArrayBuffer.empty[PokerAction]
    private val observationsByPosition = mutable.HashMap.empty[Position, mutable.ArrayBuffer[VillainObservation]]
    private val reviewHistoryLines = mutable.ArrayBuffer.empty[String]

    private var firstVillainDecision: Option[(GameState, PokerAction)] = None
    private var outcome = 0
    private var handOver = false
    private var streetsPlayed = 0

    // Pre-compute sorted boards once per hand to avoid repeated .take().sortBy() allocations.
    private val flopBoard = Board.from(deal.board.cards.take(3).sortBy(CardId.toId))
    private val turnBoard = Board.from(deal.board.cards.take(4).sortBy(CardId.toId))

    def play(): HandResult =
      if !handOver then playPreflop()
      if !handOver then playPostflopStreet(Street.Flop)
      if !handOver then playPostflopStreet(Street.Turn)
      if !handOver then playPostflopStreet(Street.River)
      val heroNet =
        if handOver && outcome > 0 then roundMoney(pot - contributionOf(heroPosition))
        else if handOver && outcome < 0 then -contributionOf(heroPosition)
        else
          val showdown = showdownResolution()
          roundMoney(showdown.heroPayout - contributionOf(heroPosition))
      outcome =
        if heroNet > MoneyEpsilon then 1
        else if heroNet < -MoneyEpsilon then -1
        else 0

      HandResult(
        heroNet = heroNet,
        outcome = outcome,
        tableScenario = tableScenario,
        villainDecision = firstVillainDecision,
        villainTrainingSamples = villainTrainingSamples.toVector,
        ddreTrainingSamples = ddreTrainingSamples.toVector,
        raiseResponses = raiseResponses.toVector,
        heroActions = heroActions.toVector,
        villainActions = villainActions.toVector,
        streetsPlayed = streetsPlayed,
        reviewHistoryLines = reviewHistoryLines.toVector,
        maxLivePlayers = tableScenario.activePositions.size
      )

    private def boardFor(street: Street): Board =
      street match
        case Street.Preflop => Board.empty
        case Street.Flop    => flopBoard
        case Street.Turn    => turnBoard
        case Street.River   => deal.board

    private def contributionOf(position: Position): Double =
      contributionByPosition.getOrElse(position, 0.0)

    private def stackOf(position: Position): Double =
      stackByPosition.getOrElse(position, 0.0)

    private def isAllIn(position: Position): Boolean =
      allInPositions.contains(position) || stackOf(position) <= MoneyEpsilon

    private def liveContestants: Vector[Position] =
      tableScenario.activePositions.filterNot(foldedPositions.contains)

    private def liveVillains: Vector[Position] =
      liveContestants.filterNot(_ == heroPosition)

    private def liveOpponentsFor(position: Position): Vector[Position] =
      liveContestants.filterNot(_ == position)

    private def multiwayRecommendationFor(
        actor: Position,
        state: GameState,
        candidateActions: Vector[PokerAction],
        posteriorOverrides: Map[Position, DiscreteDistribution[HoleCards]] = Map.empty
    ): Option[ActionRecommendation] =
      val opponents = liveOpponentsFor(actor)
      if opponents.lengthCompare(1) <= 0 then None
      else
        val result = MultiwayInferenceEngine.inferAndRecommend(
          hero = deal.holeCardsFor(actor),
          state = state,
          actorContribution = contributionOf(actor),
          actorBetHistoryIndex = betHistoryPlayerIndex(actor),
          tableRanges = tableRanges,
          actionModel = actionModel,
          opponents = opponents.map { opponent =>
            MultiwayInferenceEngine.OpponentInput(
              position = opponent,
              folds = foldsForInference(actor, opponent),
              observations = playerObservations(opponent),
              stackSize = stackOf(opponent),
              contribution = contributionOf(opponent),
              isAllIn = isAllIn(opponent),
              posteriorOverride = posteriorOverrides.get(opponent)
            )
          },
          candidateActions = candidateActions,
          bunchingTrials = inferenceBunchingTrials(state.street),
          equityTrialsForOpponentCount = opponentCount =>
            multiwayEquityTrials(state.street, config.equityTrials, opponentCount),
          rng = new Random(rng.nextLong())
        )
        Some(result.recommendation)

    private def payPosition(position: Position, amount: Double): Double =
      val paid = roundMoney(math.max(0.0, math.min(amount, stackOf(position))))
      stackByPosition.update(position, roundMoney(stackOf(position) - paid))
      contributionByPosition.update(position, roundMoney(contributionOf(position) + paid))
      pot = roundMoney(pot + paid)
      if stackOf(position) <= MoneyEpsilon then allInPositions += position
      paid

    private def recordObservation(
        position: Position,
        state: GameState,
        action: PokerAction
    ): Unit =
      val buffer = observationsByPosition.getOrElseUpdate(position, mutable.ArrayBuffer.empty)
      buffer += VillainObservation(action, state)

    private def recordVillainDecision(
        position: Position,
        state: GameState,
        action: PokerAction
    ): Unit =
      if collectVillainTraining then
        villainTrainingSamples += ((state, deal.holeCardsFor(position), action))
      villainActions += action
      recordObservation(position, state, action)
      if firstVillainDecision.isEmpty then firstVillainDecision = Some((state, action))

    private def appendReviewLine(playerName: String, suffix: String): Unit =
      reviewHistoryLines += s"$playerName: $suffix"

    private def appendReviewStreetHeader(street: Street): Unit =
      street match
        case Street.Preflop => ()
        case Street.Flop =>
          reviewHistoryLines += s"*** FLOP *** ${bracketedCards(flopBoard.cards)}"
        case Street.Turn =>
          reviewHistoryLines += s"*** TURN *** ${bracketedCards(flopBoard.cards)} ${bracketedCards(Vector(deal.board.cards(3)))}"
        case Street.River =>
          reviewHistoryLines += s"*** RIVER *** ${bracketedCards(turnBoard.cards)} ${bracketedCards(Vector(deal.board.cards(4)))}"

    private def appendFold(playerName: String): Unit =
      appendReviewLine(playerName, "folds")

    private def appendCheck(playerName: String): Unit =
      appendReviewLine(playerName, "checks")

    private def appendCall(playerName: String, amount: Double): Unit =
      appendReviewLine(playerName, s"calls ${money(amount)}")

    private def appendRaiseTo(playerName: String, totalAmount: Double): Unit =
      appendReviewLine(playerName, s"raises to ${money(totalAmount)}")

    private def totalContributionAfterRaise(
        currentContribution: Double,
        toCall: Double,
        raiseAmount: Double
    ): Double =
      roundMoney(currentContribution + toCall + raiseAmount)

    private def actorName(position: Position): String =
      tableScenario.nameFor(position)

    private def betHistoryPlayerIndex(position: Position): Int =
      tableScenario.seatNumberByPosition(position) - 1

    private def appendBetHistory(position: Position, action: PokerAction): Unit =
      betHistory = betHistory :+ BetAction(betHistoryPlayerIndex(position), action)

    private def playerObservations(position: Position): Vector[VillainObservation] =
      observationsByPosition.get(position).map(_.toVector).getOrElse(Vector.empty)

    private def nextLiveVillainAfter(
        anchor: Position,
        actionOrder: Vector[Position]
    ): Option[Position] =
      orderAfter(actionOrder, anchor).find(position =>
        position != heroPosition &&
          participatingPositions.contains(position) &&
          !foldedPositions.contains(position)
      )

    private def focusVillainForHero(
        lastAggressor: Option[Position],
        actionOrder: Vector[Position]
    ): Position =
      lastAggressor
        .filter(position =>
          position != heroPosition &&
            participatingPositions.contains(position) &&
            !foldedPositions.contains(position)
        )
        .orElse(nextLiveVillainAfter(heroPosition, actionOrder))
        .orElse(liveVillains.headOption)
        .getOrElse(tableScenario.primaryVillainPosition)

    private def inferenceBunchingTrials(street: Street): Int =
      if street == Street.Preflop then config.bunchingTrials
      else postflopBunchingTrials(config.bunchingTrials)

    private def foldsForInference(
        perspectivePosition: Position,
        targetPosition: Position
    ): Vector[PreflopFold] =
      val actual = mergedInferenceFolds(
        staticFoldedPositions = tableScenario.foldedPositions,
        preflopFoldedPositions = preflopFoldedPositions.toVector,
        perspectivePosition = perspectivePosition,
        targetPosition = targetPosition
      )
      if actual.nonEmpty then actual
      else tableScenario.fallbackFoldsFor(targetPosition, perspectivePosition)

    private def decideHero(
        street: Street,
        board: Board,
        toCall: Double,
        allowRaise: Boolean,
        lastAggressor: Option[Position],
        actionOrder: Vector[Position]
    ): PokerAction =
      val focusVillainPosition = focusVillainForHero(lastAggressor, actionOrder)
      val state = GameState(
        street = street,
        board = board,
        pot = pot,
        toCall = toCall,
        position = heroPosition,
        stackSize = stackOf(heroPosition),
        betHistory = betHistory
      )
      val candidates = heroCandidates(state, config.raiseSize, allowRaise)
      val sampled = config.heroMode match
        case HeroMode.Adaptive =>
          val engine = if street == Street.Preflop then preflopEngine else postflopEngine
          val adaptiveDecision = engine.decide(
            hero = deal.holeCardsFor(heroPosition),
            state = state,
            folds = foldsForInference(heroPosition, focusVillainPosition),
            villainPos = focusVillainPosition,
            observations = playerObservations(focusVillainPosition),
            candidateActions = candidates,
            decisionBudgetMillis = Some(1L),
            rng = new Random(rng.nextLong())
          )
          val greedy =
            multiwayRecommendationFor(
              actor = heroPosition,
              state = state,
              candidateActions = candidates,
              posteriorOverrides = Map(
                focusVillainPosition -> adaptiveDecision.decision.posteriorInference.posterior
              )
            ).map(_.bestAction)
              .getOrElse(adaptiveDecision.decision.recommendation.bestAction)
          if rng.nextDouble() < config.heroExplorationRate then candidates(rng.nextInt(candidates.length))
          else greedy
        case HeroMode.Gto =>
          multiwayRecommendationFor(
            actor = heroPosition,
            state = state,
            candidateActions = candidates
          ).map(_.bestAction)
            .getOrElse(
              GtoSolveEngine.gtoResponds(
                hand = deal.holeCardsFor(heroPosition),
                state = state,
                candidates = candidates,
                mode = config.gtoMode,
                opponentPosterior = tableRanges.rangeFor(focusVillainPosition),
                baseEquityTrials = config.equityTrials,
                rng = rng,
                perspective = 0,
                exactGtoCache = exactGtoCache,
                exactGtoCacheStats = exactGtoCacheStats
              )
            )
      val normalized = normalizeAction(
        action = sampled,
        toCall = toCall,
        stackSize = stackOf(heroPosition),
        allowRaise = allowRaise,
        defaultRaise = config.raiseSize
      )
      if collectDdreTraining then
        recordDdreTrainingSample(state, street, focusVillainPosition)
      recordObservation(heroPosition, state, normalized)
      heroActions += normalized
      normalized

    private def recordDdreTrainingSample(
        state: GameState,
        street: Street,
        focusVillainPosition: Position
    ): Unit =
      val observationsSnapshot = playerObservations(focusVillainPosition)
      val observationsForBayes = observationsSnapshot.map(obs => obs.action -> obs.state)
      val labelBunchingTrials = inferenceBunchingTrials(street)
      val prior = RangeInferenceEngine
        .inferPosterior(
          hero = deal.holeCardsFor(heroPosition),
          board = state.board,
          folds = foldsForInference(heroPosition, focusVillainPosition),
          tableRanges = tableRanges,
          villainPos = focusVillainPosition,
          observations = Vector.empty,
          actionModel = actionModel,
          bunchingTrials = labelBunchingTrials,
          rng = new Random(rng.nextLong())
        )
        .prior
        .normalized
      val (bayesPosterior, bayesLogEvidence) =
        if observationsForBayes.isEmpty then (prior, 0.0)
        else
          val update = HoldemBayesProvider.updatePosterior(
            prior = prior,
            observations = observationsForBayes,
            actionModel = actionModel
          )
          (update.posterior, update.logEvidence)
      ddreTrainingSamples += DdreTrainingSample(
        decisionIndex = ddreTrainingSamples.length + 1,
        state = state,
        villainPosition = focusVillainPosition,
        villainStackBefore = stackOf(focusVillainPosition),
        observations = observationsSnapshot,
        heroHole = deal.holeCardsFor(heroPosition),
        villainHole = deal.holeCardsFor(focusVillainPosition),
        prior = prior,
        bayesPosterior = bayesPosterior,
        bayesLogEvidence = bayesLogEvidence
      )

    private def decideVillain(
        position: Position,
        street: Street,
        board: Board,
        toCall: Double,
        allowRaise: Boolean,
        facingHeroRaise: Boolean
    ): PokerAction =
      val state = GameState(
        street = street,
        board = board,
        pot = pot,
        toCall = toCall,
        position = position,
        stackSize = stackOf(position),
        betHistory = betHistory
      )
      val villainProfile = tableScenario.villainProfileByPosition(position)
      val candidates = heroCandidates(state, config.raiseSize, allowRaise)
      val multiwayRecommendation =
        villainProfile.mode match
          case VillainMode.Gto =>
            multiwayRecommendationFor(
              actor = position,
              state = state,
              candidateActions = candidates
            )
          case _ => None
      val sampled = villainProfile.mode match
        case VillainMode.Archetype(style) =>
          ArchetypeVillainResponder.villainResponds(
            hand = deal.holeCardsFor(position),
            style = style,
            state = state,
            allowRaise = allowRaise,
            raiseSize = config.raiseSize,
            rng = rng
          )
        case VillainMode.Gto =>
          multiwayRecommendation
            .map(_.bestAction)
            .getOrElse(
              GtoSolveEngine.gtoResponds(
                hand = deal.holeCardsFor(position),
                state = state,
                candidates = candidates,
                mode = config.gtoMode,
                opponentPosterior = tableRanges.rangeFor(heroPosition),
                baseEquityTrials = config.equityTrials,
                rng = rng,
                perspective = 1,
                exactGtoCache = exactGtoCache,
                exactGtoCacheStats = exactGtoCacheStats
              )
            )
        case VillainMode.LeakInjected(leakId, severity) =>
          val villainHand = deal.holeCardsFor(position)
          val equityVsRandom = estimateEquityVsRandom(villainHand, boardFor(state.street), 50, rng)
          val equityStrategy = EquityBasedStrategy()
          val gtoAction = equityStrategy.decide(villainHand, state, candidates, equityVsRandom, rng)
          val leak = buildInjectedLeak(leakId, severity)
          val injectedVillain = LeakInjectedVillain(
            name = villainProfile.name,
            leaks = Vector(leak),
            baselineNoise = 0.03,
            seed = rng.nextLong()
          )
          val line = ActionLine(actionLineByPosition(position))
          val spot = SpotContext.build(
            gs = state,
            hero = villainHand,
            line = line,
            equityVsRandom = equityVsRandom,
            facingAction = None
          )
          val result = injectedVillain.decide(gtoAction, spot)
          result.action
      val action = normalizeAction(
        action = sampled,
        toCall = toCall,
        stackSize = stackOf(position),
        allowRaise = allowRaise,
        defaultRaise = config.raiseSize
      )
      recordVillainDecision(position, state, action)
      if facingHeroRaise then
        action match
          case PokerAction.Fold | PokerAction.Call | PokerAction.Raise(_) =>
            preflopEngine.observeVillainResponseToRaise(action)
            postflopEngine.observeVillainResponseToRaise(action)
            raiseResponses += action
          case _ => ()
      action

    private def appendAction(
        position: Position,
        action: PokerAction,
        toCallBefore: Double,
        stackBefore: Double
    ): Unit =
      action match
        case PokerAction.Fold =>
          appendFold(actorName(position))
        case PokerAction.Check =>
          appendCheck(actorName(position))
        case PokerAction.Call =>
          if toCallBefore > 0.0 then appendCall(actorName(position), math.min(toCallBefore, stackBefore))
          else appendCheck(actorName(position))
        case PokerAction.Raise(amount) =>
          appendRaiseTo(
            actorName(position),
            totalContributionAfterRaise(contributionOf(position), toCallBefore, amount)
          )

    private def markFolded(position: Position, street: Street): Unit =
      if !foldedPositions.contains(position) then
        foldedPositions += position
        if street == Street.Preflop && !preflopFoldedPositions.contains(position) then
          preflopFoldedPositions += position
      if position == heroPosition then
        handOver = true
        outcome = -1
      else if !foldedPositions.contains(heroPosition) && liveVillains.isEmpty then
        handOver = true
        outcome = 1

    private def applyAction(
        position: Position,
        action: PokerAction,
        toCallBefore: Double,
        street: Street
    ): Unit =
      actionLineByPosition.update(position, actionLineByPosition(position) :+ action)
      action match
        case PokerAction.Fold =>
          markFolded(position, street)
        case PokerAction.Check =>
          ()
        case PokerAction.Call =>
          payPosition(position, toCallBefore)
        case PokerAction.Raise(amount) =>
          val required = toCallBefore + amount
          payPosition(position, required)

    private def currentToCall(position: Position): Double =
      val currentMaxContribution = contributionByPosition.values.max
      TexasHoldemPlayingHall.contributionGap(currentMaxContribution, contributionOf(position))

    private def orderAfter(
        actionOrder: Vector[Position],
        position: Position
    ): Vector[Position] =
      val idx = actionOrder.indexOf(position)
      if idx < 0 then actionOrder
      else actionOrder.drop(idx + 1) ++ actionOrder.take(idx)

    private def initialQueueForStreet(
        street: Street,
        actionOrder: Vector[Position]
    ): Vector[Position] =
      actionOrder.filter(position => canQueue(position, street))

    private def canQueue(position: Position, street: Street): Boolean =
      if foldedPositions.contains(position) then false
      else if isAllIn(position) then false
      else if street == Street.Preflop then true
      else participatingPositions.contains(position)

    private def decidePosition(
        position: Position,
        street: Street,
        board: Board,
        toCall: Double,
        allowRaise: Boolean,
        lastAggressor: Option[Position],
        actionOrder: Vector[Position]
    ): PokerAction =
      if position == heroPosition then
        decideHero(
          street = street,
          board = board,
          toCall = toCall,
          allowRaise = allowRaise,
          lastAggressor = lastAggressor,
          actionOrder = actionOrder
        )
      else if participatingPositions.contains(position) then
        decideVillain(
          position = position,
          street = street,
          board = board,
          toCall = toCall,
          allowRaise = allowRaise,
          facingHeroRaise = lastAggressor.contains(heroPosition) && toCall > 0.0
        )
      else PokerAction.Fold

    private def playBettingRound(
        street: Street,
        board: Board,
        actionOrder: Vector[Position],
        maxRaises: Int
    ): Unit =
      var raiseCount = 0
      var lastAggressor = Option.empty[Position]
      var minimumRaiseAmount = BigBlindAmount
      val actedSinceFullRaise = mutable.HashSet.empty[Position]
      val raisingClosedFor = mutable.HashSet.empty[Position]
      var queue = initialQueueForStreet(street, actionOrder)
      while !handOver && queue.nonEmpty do
        val actor = queue.head
        queue = queue.tail
        if canQueue(actor, street) then
          val toCallBefore = currentToCall(actor)
          val stackBefore = stackOf(actor)
          val action = decidePosition(
            position = actor,
            street = street,
            board = board,
            toCall = toCallBefore,
            allowRaise = (raiseCount < maxRaises) && !raisingClosedFor.contains(actor),
            lastAggressor = lastAggressor,
            actionOrder = actionOrder
          )
          appendBetHistory(actor, action)
          appendAction(actor, action, toCallBefore, stackBefore)
          applyAction(actor, action, toCallBefore, street)
          actedSinceFullRaise += actor
          if !handOver then
            action match
              case PokerAction.Raise(amount) =>
                lastAggressor = Some(actor)
                queue = orderAfter(actionOrder, actor).filter(position => canQueue(position, street))
                if raiseReopensAction(amount, minimumRaiseAmount) then
                  raiseCount += 1
                  minimumRaiseAmount = roundMoney(amount)
                  actedSinceFullRaise.clear()
                  actedSinceFullRaise += actor
                  raisingClosedFor.clear()
                else
                  raisingClosedFor ++= actedSinceFullRaise.filter(_ != actor)
              case _ =>
                ()

    private def playPreflop(): Unit =
      streetsPlayed += 1
      playBettingRound(
        street = Street.Preflop,
        board = boardFor(Street.Preflop),
        actionOrder = preflopOrder,
        maxRaises = 2
      )

    private def playPostflopStreet(street: Street): Unit =
      if handOver then ()
      else
        streetsPlayed += 1
        val board = boardFor(street)
        appendReviewStreetHeader(street)
        playBettingRound(
          street = street,
          board = board,
          actionOrder = postflopOrder,
          maxRaises = 1
        )

    private def showdownResolution(): ShowdownResolution =
      val remainingPlayers = liveContestants
      if !remainingPlayers.contains(heroPosition) then ShowdownResolution(heroPayout = 0.0)
      else
        val boardCards = deal.board.cards
        val ranked = remainingPlayers.map { position =>
          val hand = deal.holeCardsFor(position)
          position -> HandEvaluator.evaluate7PackedDirect(
            hand.first,
            hand.second,
            boardCards(0),
            boardCards(1),
            boardCards(2),
            boardCards(3),
            boardCards(4)
          )
        }.toMap
        val payouts = sidePotPayouts(
          contributions = contributionByPosition.toMap,
          remainingPlayers = remainingPlayers,
          handStrengthByPosition = ranked
        )
        ShowdownResolution(heroPayout = payouts.getOrElse(heroPosition, 0.0))

  private[holdem] def contributionGap(targetContribution: Double, currentContribution: Double): Double =
    math.max(0.0, targetContribution - currentContribution)

  private[holdem] def mergedInferenceFolds(
      staticFoldedPositions: Seq[Position],
      preflopFoldedPositions: Seq[Position],
      perspectivePosition: Position,
      targetPosition: Position
  ): Vector[PreflopFold] =
    (staticFoldedPositions ++ preflopFoldedPositions)
      .distinct
      .filterNot(position => position == perspectivePosition || position == targetPosition)
      .map(PreflopFold(_))
      .toVector

  private[holdem] def estimateRaiseResponseFromRange(
      range: DiscreteDistribution[HoleCards],
      responseState: GameState,
      actionModel: PokerActionModel,
      raiseAction: PokerAction
  ): RaiseResponseEstimate =
    val estimate = MultiwayInferenceEngine.estimateRaiseResponseFromRange(
      range = range,
      responseState = responseState,
      actionModel = actionModel,
      raiseAction = raiseAction
    )
    RaiseResponseEstimate(
      foldProbability = estimate.foldProbability,
      continueProbability = estimate.continueProbability,
      continuationRange = estimate.continuationRange
    )

  private def minEquityTrials(configured: Int): Int =
    math.max(1, configured / 30)

  private def postflopBunchingTrials(configured: Int): Int =
    1

  private def postflopEquityTrials(configured: Int): Int =
    math.max(8, configured / 16)

  private def multiwayEquityTrials(
      street: Street,
      baseEquityTrials: Int,
      opponentCount: Int
  ): Int =
    require(opponentCount > 0, "opponentCount must be positive")
    val floor =
      street match
        case Street.Preflop => 80
        case Street.Flop    => 48
        case Street.Turn    => 32
        case Street.River   => 24
    val divisor =
      street match
        case Street.Preflop => opponentCount
        case _              => math.max(1, opponentCount - 1)
    math.max(floor, math.round(baseEquityTrials.toDouble / divisor.toDouble).toInt)

  private[holdem] def estimateEquityAgainstOpponentRanges(
      hero: HoleCards,
      board: Board,
      opponentRanges: Seq[DiscreteDistribution[HoleCards]],
      equityTrials: Int,
      rng: Random,
      exactMaxEvaluations: Long = MultiwayExactMaxEvaluations
  ): EquityEstimate =
    MultiwayInferenceEngine.estimateEquityAgainstOpponentRanges(
      hero = hero,
      board = board,
      opponentRanges = opponentRanges,
      equityTrials = equityTrials,
      rng = rng,
      exactMaxEvaluations = exactMaxEvaluations
    )

  private[holdem] def recommendActionAgainstOpponentRanges(
      hero: HoleCards,
      state: GameState,
      opponentRanges: Seq[DiscreteDistribution[HoleCards]],
      candidateActions: Vector[PokerAction],
      equityTrials: Int,
      rng: Random,
      actionValueModel: ActionValueModel = ActionValueModel.ChipEv()
  ): ActionRecommendation =
    MultiwayInferenceEngine.recommendActionAgainstOpponentRanges(
      hero = hero,
      state = state,
      opponentRanges = opponentRanges,
      candidateActions = candidateActions,
      equityTrials = equityTrials,
      rng = rng,
      actionValueModel = actionValueModel,
      exactMaxEvaluations = MultiwayExactMaxEvaluations
    )

  private def buildEngine(
      tableRanges: TableRanges,
      model: PokerActionModel,
      bunchingTrials: Int,
      equityTrials: Int
  ): RealTimeAdaptiveEngine =
    new RealTimeAdaptiveEngine(
      tableRanges = tableRanges,
      actionModel = model,
      bunchingTrials = bunchingTrials,
      defaultEquityTrials = equityTrials,
      minEquityTrials = minEquityTrials(equityTrials)
    )

  private def formatLongCountMap(counts: Map[String, Long]): String =
    if counts.isEmpty then "{}"
    else
      counts.toVector
        .sortBy { case (provider, count) => (-count, provider) }
        .map { case (provider, count) => s"$provider:$count" }
        .mkString("{", ", ", "}")

  private def villainModeLabel(mode: VillainMode): String =
    mode match
      case VillainMode.Archetype(style)             => style.toString
      case VillainMode.Gto                          => "gto"
      case VillainMode.LeakInjected(leakId, sev)    => s"Leak($leakId@$sev)"

  private def villainModeSlug(mode: VillainMode): String =
    mode match
      case VillainMode.Archetype(style)             => style.toString.toLowerCase(Locale.ROOT)
      case VillainMode.Gto                          => "gto"
      case VillainMode.LeakInjected(leakId, _)      => leakId.replace("-", "").take(12)

  private def buildVillainPool(
      fallbackMode: VillainMode,
      rawPool: Option[String]
  ): Either[String, Vector[VillainProfile]] =
    rawPool.map(_.trim).filter(_.nonEmpty) match
      case None =>
        Right(Vector(VillainProfile(name = "Villain", mode = fallbackMode, label = villainModeLabel(fallbackMode))))
      case Some(raw) =>
        val tokens = raw.split(",").toVector.map(_.trim).filter(_.nonEmpty)
        if tokens.isEmpty then Left("--villainPool must include at least one style")
        else
          tokens.zipWithIndex.foldLeft[Either[String, Vector[VillainProfile]]](Right(Vector.empty)) {
            case (Left(error), _) => Left(error)
            case (Right(acc), (token, idx)) =>
              parseVillainModeToken(token).left.map(error => s"--villainPool: $error").map { mode =>
                acc :+ VillainProfile(
                  name = f"Villain${idx + 1}%02d_${villainModeSlug(mode)}",
                  mode = mode,
                  label = villainModeLabel(mode)
                )
              }
          }

  private def parseVillainModeToken(raw: String): Either[String, VillainMode] =
    raw.trim.toLowerCase match
      case "nit"            => Right(VillainMode.Archetype(PlayerArchetype.Nit))
      case "tag"            => Right(VillainMode.Archetype(PlayerArchetype.Tag))
      case "lag"            => Right(VillainMode.Archetype(PlayerArchetype.Lag))
      case "callingstation" => Right(VillainMode.Archetype(PlayerArchetype.CallingStation))
      case "station"        => Right(VillainMode.Archetype(PlayerArchetype.CallingStation))
      case "maniac"         => Right(VillainMode.Archetype(PlayerArchetype.Maniac))
      case "gto"            => Right(VillainMode.Gto)
      case s if s.startsWith("leak:") =>
        parseLeakToken(s)
      case _ =>
        Left("style must be one of: nit, tag, lag, callingstation, station, maniac, gto, leak:<type>:<severity>")

  private def parseLeakToken(raw: String): Either[String, VillainMode] =
    val parts = raw.split(":")
    if parts.length != 3 then Left(s"leak token must be leak:<type>:<severity>, got: $raw")
    else
      val leakType = parts(1)
      val severityStr = parts(2)
      for
        severity <- try Right(severityStr.toDouble) catch case _: NumberFormatException =>
          Left(s"invalid severity: $severityStr")
        leakId <- leakType match
          case "overfold"      => Right("overfold-river-aggression")
          case "overcall"      => Right("overcall-big-bets")
          case "turnbluff"     => Right("overbluff-turn-barrel")
          case "passive"       => Right("passive-big-pots")
          case "prefloploose"  => Right("preflop-too-loose")
          case "prefloptight"  => Right("preflop-too-tight")
          case _               => Left(s"unknown leak type: $leakType (use: overfold, overcall, turnbluff, passive, prefloploose, prefloptight)")
      yield VillainMode.LeakInjected(leakId, severity)

  private def estimateEquityVsRandom(
      hand: HoleCards,
      board: Board,
      trials: Int,
      rng: Random
  ): Double =
    val handCards = hand.toVector
    val available = Deck.full.filterNot(c => handCards.contains(c) || board.cards.contains(c))
    if available.size < 2 then return 0.5
    val boardSize = board.cards.size
    val communityNeeded = 5 - boardSize
    val needed = 2 + communityNeeded
    var wins = 0
    var total = 0
    val arr = available.toArray
    val n = arr.length
    for _ <- 0 until trials do
      var i = 0
      while i < needed do
        val j = i + rng.nextInt(n - i)
        val tmp = arr(i); arr(i) = arr(j); arr(j) = tmp
        i += 1
      val fullBoard = if communityNeeded > 0 then
        board.cards ++ (2 until needed).map(arr(_))
      else board.cards
      val heroAll = (handCards ++ fullBoard).take(7)
      val oppAll = (Vector(arr(0), arr(1)) ++ fullBoard).take(7)
      if heroAll.size >= 7 && oppAll.size >= 7 then
        val heroRank = HandEvaluator.evaluate7(heroAll)
        val oppRank = HandEvaluator.evaluate7(oppAll)
        if heroRank > oppRank then wins += 1
        total += 1
    if total > 0 then wins.toDouble / total.toDouble else 0.5

  private def buildInjectedLeak(leakId: String, severity: Double): InjectedLeak =
    leakId match
      case "overfold-river-aggression" => OverfoldsToAggression(severity)
      case "overcall-big-bets"         => Overcalls(severity)
      case "overbluff-turn-barrel"     => OverbluffsTurnBarrel(severity)
      case "passive-big-pots"          => PassiveInBigPots(severity)
      case "preflop-too-loose"         => PreflopTooLoose(severity)
      case "preflop-too-tight"         => PreflopTooTight(severity)
      case _                           => NoLeak()

  private def bracketedCards(cards: Seq[Card]): String =
    s"[${cards.map(_.toToken).mkString(" ")}]"

  private def money(amount: Double): String =
    s"$$${fmt(roundMoney(amount), 2)}"

  private def heroCandidates(
      state: GameState,
      raiseSize: Double,
      allowRaise: Boolean
  ): Vector[PokerAction] =
    if state.toCall <= 0.0 then
      if allowRaise && state.stackSize > math.max(0.2, raiseSize * 0.25) then
        Vector(PokerAction.Check, PokerAction.Raise(raiseSize))
      else Vector(PokerAction.Check)
    else
      val base = Vector(PokerAction.Fold, PokerAction.Call)
      if allowRaise && state.stackSize > (state.toCall + math.max(0.2, raiseSize * 0.25)) then
        base :+ PokerAction.Raise(raiseSize)
      else base

  private[holdem] def normalizeAction(
      action: PokerAction,
      toCall: Double,
      stackSize: Double,
      allowRaise: Boolean,
      defaultRaise: Double
  ): PokerAction =
    val effectiveStack = roundMoney(math.max(0.0, stackSize))
    action match
      case PokerAction.Check =>
        if toCall <= MoneyEpsilon then PokerAction.Check else PokerAction.Call
      case PokerAction.Call =>
        if toCall > MoneyEpsilon then PokerAction.Call else PokerAction.Check
      case PokerAction.Fold =>
        if toCall > MoneyEpsilon then PokerAction.Fold else PokerAction.Check
      case PokerAction.Raise(amount) =>
        if !allowRaise then
          if toCall > MoneyEpsilon then PokerAction.Call else PokerAction.Check
        else
          val cleanAmount = if amount > 0.0 then amount else defaultRaise
          if cleanAmount <= MoneyEpsilon || effectiveStack <= MoneyEpsilon then
            if toCall > MoneyEpsilon then PokerAction.Call else PokerAction.Check
          else if toCall > MoneyEpsilon then
            if effectiveStack <= toCall + MoneyEpsilon then PokerAction.Call
            else
              val affordableRaise = roundMoney(math.min(cleanAmount, effectiveStack - toCall))
              if affordableRaise <= MoneyEpsilon then PokerAction.Call
              else PokerAction.Raise(affordableRaise)
          else PokerAction.Raise(roundMoney(math.min(cleanAmount, effectiveStack)))

  private[holdem] def raiseReopensAction(
      raiseAmount: Double,
      minimumRaiseAmount: Double
  ): Boolean =
    roundMoney(raiseAmount) + MoneyEpsilon >= roundMoney(minimumRaiseAmount)

  private[holdem] def sidePotPayouts(
      contributions: Map[Position, Double],
      remainingPlayers: Vector[Position],
      handStrengthByPosition: Map[Position, Int]
  ): Map[Position, Double] =
    if remainingPlayers.isEmpty then Map.empty
    else
      require(
        remainingPlayers.forall(handStrengthByPosition.contains),
        "remaining players must have showdown hand strengths"
      )
      val roundedContributions =
        contributions.iterator.map { case (position, amount) =>
          position -> roundMoney(math.max(0.0, amount))
        }.toMap
      val contributionLevels =
        roundedContributions.valuesIterator
          .filter(_ > MoneyEpsilon)
          .toVector
          .distinct
          .sorted
      val remainingSet = remainingPlayers.toSet
      val payouts = mutable.HashMap.empty[Position, Double].withDefaultValue(0.0)
      var previousLevel = 0.0
      contributionLevels.foreach { level =>
        val slice = roundMoney(level - previousLevel)
        if slice > MoneyEpsilon then
          val potParticipants =
            roundedContributions.collect { case (position, amount) if amount + MoneyEpsilon >= level => position }.toVector
          val eligibleWinners = potParticipants.filter(remainingSet.contains)
          if eligibleWinners.nonEmpty then
            val bestStrength = eligibleWinners.map(handStrengthByPosition).max
            val winners = eligibleWinners.filter(position => handStrengthByPosition(position) == bestStrength)
            val share = (slice * potParticipants.size.toDouble) / winners.size.toDouble
            winners.foreach { position =>
              payouts.update(position, payouts(position) + share)
            }
        previousLevel = level
      }
      payouts.iterator.map { case (position, amount) =>
        position -> roundMoney(amount)
      }.toMap

  private def dealHand(
      modeledPositions: Vector[Position],
      rng: Random
  ): Deal =
    val cards = Deck.full.toArray
    val holeCardsCount = modeledPositions.length * 2
    shufflePrefix(cards, holeCardsCount + 5, rng)
    val holeCardsByPosition =
      modeledPositions.zipWithIndex.map { case (position, idx) =>
        position -> HoleCards.from(Vector(cards(idx * 2), cards((idx * 2) + 1)))
      }.toMap
    val board = Board.from(cards.slice(holeCardsCount, holeCardsCount + 5).toVector.sortBy(CardId.toId))
    Deal(holeCardsByPosition, board)

  private def shufflePrefix[A](array: Array[A], count: Int, rng: Random): Unit =
    val limit = math.min(count, array.length)
    var i = 0
    while i < limit do
      val j = i + rng.nextInt(array.length - i)
      val tmp = array(i)
      array(i) = array(j)
      array(j) = tmp
      i += 1

  private def modeledPositionsForPlayerCount(playerCount: Int): Vector[Position] =
    playerCount match
      case 2 => Vector(Position.Button, Position.BigBlind)
      case 3 => Vector(Position.Button, Position.SmallBlind, Position.BigBlind)
      case 4 => Vector(Position.Cutoff, Position.Button, Position.SmallBlind, Position.BigBlind)
      case 5 => Vector(Position.Middle, Position.Cutoff, Position.Button, Position.SmallBlind, Position.BigBlind)
      case 6 => Vector(Position.UTG, Position.Middle, Position.Cutoff, Position.Button, Position.SmallBlind, Position.BigBlind)
      case 7 => Vector(Position.UTG, Position.UTG1, Position.Middle, Position.Cutoff, Position.Button, Position.SmallBlind, Position.BigBlind)
      case 8 => Vector(
        Position.UTG,
        Position.UTG1,
        Position.UTG2,
        Position.Middle,
        Position.Cutoff,
        Position.Button,
        Position.SmallBlind,
        Position.BigBlind
      )
      case 9 => Vector(
        Position.UTG,
        Position.UTG1,
        Position.UTG2,
        Position.Middle,
        Position.Hijack,
        Position.Cutoff,
        Position.Button,
        Position.SmallBlind,
        Position.BigBlind
      )
      case _ => Vector.empty

  private def smallBlindPositionFor(playerCount: Int): Position =
    if playerCount <= 2 then Position.Button
    else Position.SmallBlind

  private def blindContributionFor(position: Position, playerCount: Int): Double =
    if position == Position.BigBlind then BigBlindAmount
    else if position == smallBlindPositionFor(playerCount) then SmallBlindAmount
    else 0.0

  private def postflopOrderIndex(position: Position): Int =
    Vector(
      Position.SmallBlind,
      Position.BigBlind,
      Position.UTG,
      Position.UTG1,
      Position.UTG2,
      Position.Middle,
      Position.Hijack,
      Position.Cutoff,
      Position.Button
    ).indexOf(position)

  private def buildTableScenario(
      playerCount: Int,
      heroPosition: Position,
      villainPool: Vector[VillainProfile],
      headsUpOnly: Boolean,
      forceAllActive: Boolean,
      rng: Random
  ): TableScenario =
    val modeledPositions = modeledPositionsForPlayerCount(playerCount)
    require(modeledPositions.contains(heroPosition), s"hero position $heroPosition is not valid for playerCount=$playerCount")
    val availableVillains = modeledPositions.filterNot(_ == heroPosition)
    val activeVillainPositions =
      if forceAllActive then
        availableVillains.sortBy(modeledPositions.indexOf)
      else if headsUpOnly then
        if heroPosition == Position.BigBlind then
          Vector(availableVillains(rng.nextInt(availableVillains.length)))
        else
          Vector(Position.BigBlind)
      else
        val minVillainCount =
          if availableVillains.length <= 1 then 1
          else math.min(2, availableVillains.length)
        val maxVillainCount =
          math.min(availableVillains.length, if playerCount >= 6 then 3 else 2)
        val villainCount =
          if maxVillainCount <= minVillainCount then minVillainCount
          else minVillainCount + rng.nextInt(maxVillainCount - minVillainCount + 1)
        availableVillains
          .sortBy(_ => rng.nextDouble())
          .take(villainCount)
          .sortBy(modeledPositions.indexOf)
    val activePositions =
      modeledPositions.filter(position => position == heroPosition || activeVillainPositions.contains(position))
    val foldedPositions = modeledPositions.filterNot(activePositions.contains)
    val profileOffset = rng.nextInt(villainPool.length)
    val villainProfileByPosition =
      activeVillainPositions.zipWithIndex.map { case (position, idx) =>
        position -> villainPool((profileOffset + idx) % villainPool.length)
      }.toMap
    val seatNumberByPosition =
      modeledPositions.zipWithIndex.map { case (position, idx) => position -> (idx + 1) }.toMap
    val playerNameByPosition =
      modeledPositions.zipWithIndex.map { case (position, idx) =>
        val name =
          if position == heroPosition then ReviewHeroName
          else if villainProfileByPosition.contains(position) then villainProfileByPosition(position).name
          else f"Player${idx + 1}%02d_${position.toString}"
        position -> name
      }.toMap

    TableScenario(
      playerCount = playerCount,
      modeledPositions = modeledPositions,
      activePositions = activePositions,
      heroPosition = heroPosition,
      primaryVillainPosition = activeVillainPositions.head,
      openerPosition = activePositions.head,
      foldedPositions = foldedPositions,
      seatNumberByPosition = seatNumberByPosition,
      playerNameByPosition = playerNameByPosition,
      villainProfileByPosition = villainProfileByPosition
    )

  private def loadInitialArtifact(config: Config, modelsRoot: Path): (TrainedPokerActionModel, Path) =
    config.modelArtifactDir match
      case Some(path) =>
        (PokerActionModelArtifactIO.load(path), path)
      case None =>
        val bootstrapDir = modelsRoot.resolve("bootstrap-uniform")
        val metadataPath = bootstrapDir.resolve("metadata.properties")
        if Files.exists(metadataPath) then
          (PokerActionModelArtifactIO.load(bootstrapDir), bootstrapDir)
        else
          val bootstrap = TrainedPokerActionModel(
            version = ModelVersion(
              id = "hall-bootstrap-uniform",
              schemaVersion = "poker-action-model-v1",
              source = "texas-holdem-playing-hall-bootstrap",
              trainedAtEpochMillis = System.currentTimeMillis()
            ),
            model = PokerActionModel.uniform,
            calibration = CalibrationSummary(
              meanBrierScore = 0.75,
              sampleCount = 1,
              uniformBaselineBrier = 0.75,
              majorityBaselineBrier = 1.0
            ),
            gate = CalibrationGate(maxMeanBrierScore = 2.0),
            trainingSampleCount = 1,
            evaluationSampleCount = 1,
            evaluationStrategy = "bootstrap-uniform",
            validationFraction = None,
            splitSeed = None
          )
          // Avoid rewriting the same bootstrap artifact for short-lived benchmark runs.
          if config.learnEveryHands > 0 then
            PokerActionModelArtifactIO.save(bootstrapDir, bootstrap)
          (bootstrap, bootstrapDir)

  private def openLogWriter(path: Path): BufferedWriter =
    Files.newBufferedWriter(path, StandardCharsets.UTF_8)

  private def closeQuietly(writer: BufferedWriter): Unit =
    try writer.close()
    catch
      case _: Throwable => ()

  private def writeLine(writer: BufferedWriter, line: String): Unit =
    writer.write(line)
    writer.newLine()

  private def initHandsLog(writer: BufferedWriter): Unit =
    val header = Vector(
      "hand",
      "tableId",
      "playerCount",
      "heroPosition",
      "villainPosition",
      "openerPosition",
      "foldedPositions",
      "activePositions",
      "maxLivePlayers",
      "heroHole",
      "villainHole",
      "board",
      "heroAction",
      "villainAction",
      "heroNet",
      "modelId",
      "archetype",
      "streetsPlayed",
      "heroActions",
      "villainActions",
      "outcome"
    ).mkString("\t")
    writeLine(writer, header)

  private def appendHandLog(
      writer: BufferedWriter,
      hand: Int,
      tableId: Int,
      deal: Deal,
      result: HandResult,
      modelId: String,
      archetype: String,
      playerCount: Int,
      heroPosition: Position,
      villainPosition: Position,
      openerPosition: Position,
      foldedPositions: Vector[Position],
      activePositions: Vector[Position],
      maxLivePlayers: Int
  ): Unit =
    val boardToken = deal.board.cards.map(_.toToken).mkString(" ")
    val heroAction = result.heroActions.headOption.map(renderAction).getOrElse("-")
    val villainAction = result.villainActions.headOption.map(renderAction).getOrElse("-")
    val heroActionTrace = if result.heroActions.nonEmpty then result.heroActions.map(renderAction).mkString("|") else "-"
    val villainActionTrace = if result.villainActions.nonEmpty then result.villainActions.map(renderAction).mkString("|") else "-"
    val foldedToken =
      if foldedPositions.isEmpty then "-"
      else foldedPositions.map(_.toString).mkString("|")
    val activeToken =
      if activePositions.isEmpty then "-"
      else activePositions.map(_.toString).mkString("|")
    val outcomeLabel =
      if result.outcome > 0 then "win"
      else if result.outcome < 0 then "loss"
      else "tie"
    val row = Vector(
      hand.toString,
      tableId.toString,
      playerCount.toString,
      heroPosition.toString,
      villainPosition.toString,
      openerPosition.toString,
      foldedToken,
      activeToken,
      maxLivePlayers.toString,
      deal.holeCardsFor(heroPosition).toToken,
      deal.holeCardsFor(villainPosition).toToken,
      boardToken,
      heroAction,
      villainAction,
      fmt(result.heroNet, 4),
      modelId,
      archetype,
      result.streetsPlayed.toString,
      heroActionTrace,
      villainActionTrace,
      outcomeLabel
    ).mkString("\t")
    writeLine(writer, row)

  private def appendReviewHandHistory(
      writer: BufferedWriter,
      hand: Int,
      tableId: Int,
      startedAt: LocalDateTime,
      deal: Deal,
      result: HandResult,
      tableScenario: TableScenario
  ): Unit =
    val tableLabel =
      if tableScenario.playerCount <= 2 then "2-max"
      else s"${tableScenario.playerCount}-max"
    val header = s"PokerStars Hand #${1000 + hand}:  Hold'em No Limit (${money(0.5)}/${money(1.0)} USD) - ${startedAt.format(ReviewTimestampFormatter)} ET"
    val tableLine = s"Table 'SICFUN Proof $tableId' $tableLabel Seat #${tableScenario.buttonSeatNumber} is the button"
    val heroCardsLine = s"Dealt to $ReviewHeroName ${bracketedCards(deal.holeCardsFor(tableScenario.heroPosition).toVector)}"
    val seatLines =
      tableScenario.modeledPositions.map { position =>
        s"Seat ${tableScenario.seatNumberByPosition(position)}: ${tableScenario.nameFor(position)} (${money(ReviewStartingStack)} in chips)"
      }
    val smallBlindPosition = smallBlindPositionFor(tableScenario.playerCount)
    val blindLines = Vector(
      s"${tableScenario.nameFor(smallBlindPosition)}: posts small blind ${money(SmallBlindAmount)}",
      s"${tableScenario.nameFor(Position.BigBlind)}: posts big blind ${money(BigBlindAmount)}"
    )

    writeLine(writer, header)
    writeLine(writer, tableLine)
    seatLines.foreach(line => writeLine(writer, line))
    blindLines.foreach(line => writeLine(writer, line))
    writeLine(writer, "*** HOLE CARDS ***")
    writeLine(writer, heroCardsLine)
    result.reviewHistoryLines.foreach(line => writeLine(writer, line))
    writeLine(writer, "*** SUMMARY ***")
    writer.newLine()

  private def initLearningLog(writer: BufferedWriter): Unit =
    writeLine(writer, "hand\tmodelId\tmeanBrierScore\tgatePassed\tsampleCount")

  private def appendLearningLog(
      writer: BufferedWriter,
      hand: Int,
      modelId: String,
      meanBrierScore: Double,
      gatePassed: Boolean,
      sampleCount: Int
  ): Unit =
    val row = Vector(
      hand.toString,
      modelId,
      meanBrierScore.toString,
      gatePassed.toString,
      sampleCount.toString
    ).mkString("\t")
    writeLine(writer, row)

  private def initTrainingTsv(writer: BufferedWriter): Unit =
    writeLine(writer, "street\tboard\tpotBefore\ttoCall\tposition\tstackBefore\taction\tholecards\thand\ttableId")

  private def appendTrainingSample(
      writer: BufferedWriter,
      hand: Int,
      tableId: Int,
      sample: (GameState, HoleCards, PokerAction)
  ): Unit =
    val (state, hole, action) = sample
    val row = Vector(
      state.street.toString,
      boardToken(state.board),
      state.pot.toString,
      state.toCall.toString,
      state.position.toString,
      state.stackSize.toString,
      actionToken(action),
      s"${hole.first.toToken} ${hole.second.toToken}",
      hand.toString,
      tableId.toString
    ).mkString("\t")
    writeLine(writer, row)

  private def initDdreTrainingTsv(writer: BufferedWriter): Unit =
    writeLine(
      writer,
      "hand\ttableId\tdecisionIndex\tstreet\tboard\tpotBefore\ttoCall\theroPosition\tvillainPosition\theroStackBefore\tvillainStackBefore\tbetHistory\tvillainObservations\theroHole\tvillainHole\tbayesLogEvidence\tpriorSparse\tbayesPosteriorSparse"
    )

  private def appendDdreTrainingSample(
      writer: BufferedWriter,
      hand: Int,
      tableId: Int,
      sample: DdreTrainingSample
  ): Unit =
    val row = Vector(
      hand.toString,
      tableId.toString,
      sample.decisionIndex.toString,
      sample.state.street.toString,
      boardToken(sample.state.board),
      sample.state.pot.toString,
      sample.state.toCall.toString,
      sample.state.position.toString,
      sample.villainPosition.toString,
      sample.state.stackSize.toString,
      sample.villainStackBefore.toString,
      serializeBetHistory(sample.state.betHistory),
      serializeVillainObservations(sample.observations),
      sample.heroHole.toToken,
      sample.villainHole.toToken,
      sample.bayesLogEvidence.toString,
      serializePosteriorSparse(sample.prior),
      serializePosteriorSparse(sample.bayesPosterior)
    ).mkString("\t")
    writeLine(writer, row)

  private def boardToken(board: Board): String =
    if board.cards.isEmpty then "-"
    else board.cards.map(_.toToken).mkString(" ")

  private def serializeBetHistory(history: Vector[BetAction]): String =
    if history.isEmpty then "-"
    else
      history.map { item =>
        s"p=${item.player},a=${actionToken(item.action)}"
      }.mkString("|")

  private def serializeVillainObservations(observations: Vector[VillainObservation]): String =
    if observations.isEmpty then "-"
    else
      observations.map { item =>
        val observedState = item.state
        val encodedHistory = URLEncoder.encode(serializeBetHistory(observedState.betHistory), StandardCharsets.UTF_8)
        s"st=${observedState.street},a=${actionToken(item.action)},pot=${observedState.pot},call=${observedState.toCall},pos=${observedState.position},board=${boardToken(observedState.board)},stack=${observedState.stackSize},history=$encodedHistory"
      }.mkString("|")

  private def serializePosteriorSparse(posterior: DiscreteDistribution[HoleCards]): String =
    val entries = posterior.weights.toVector
      .collect { case (hand, probability) if probability > 0.0 =>
        HoleCardsIndex.idOf(hand) -> probability
      }
      .sortBy(_._1)
    if entries.isEmpty then "-"
    else entries.map { case (id, probability) => s"$id:$probability" }.mkString("|")

  private def actionToken(action: PokerAction): String =
    action match
      case PokerAction.Fold => "fold"
      case PokerAction.Check => "check"
      case PokerAction.Call => "call"
      case PokerAction.Raise(amount) => s"raise:${fmt(amount, 3)}"

  private def renderAction(action: PokerAction): String =
    action match
      case PokerAction.Fold => "Fold"
      case PokerAction.Check => "Check"
      case PokerAction.Call => "Call"
      case PokerAction.Raise(amount) => s"Raise:${fmt(amount, 2)}"

  private def fmt(value: Double, digits: Int): String =
    String.format(Locale.ROOT, s"%.${digits}f", java.lang.Double.valueOf(value))

  private def roundMoney(value: Double): Double =
    math.round(value * 100.0) / 100.0

  private def parseLegacyHeroSeat(raw: String): Either[String, Position] =
    raw.trim.toLowerCase match
      case "button" => Right(Position.Button)
      case "bigblind" | "bb" => Right(Position.BigBlind)
      case _ => Left("--heroSeat must be one of: button, bigblind")

  private def resolveHeroPosition(
      options: Map[String, String],
      playerCount: Int
  ): Either[String, Position] =
    val rawPosition =
      options.get("heroPosition") match
        case Some(raw) =>
          CliHelpers.parsePositionOptionEither(options, "heroPosition", Position.Button)
        case None =>
          options.get("heroSeat") match
            case Some(raw) => parseLegacyHeroSeat(raw)
            case None => Right(Position.Button)
    rawPosition.flatMap { position =>
      val modeledPositions = modeledPositionsForPlayerCount(playerCount)
      if modeledPositions.contains(position) then Right(position)
      else Left(s"--heroPosition $position is not valid for playerCount=$playerCount")
    }

  private def parseArgs(args: Array[String]): Either[String, Config] =
    if args.contains("--help") || args.contains("-h") then Left(usage)
    else
      for
        options <- CliHelpers.parseOptions(args)
        hands <- intOpt(options, "hands", 100000)
        _ <- if hands > 0 then Right(()) else Left("--hands must be > 0")
        tableCount <- intOpt(options, "tableCount", 1)
        _ <- if tableCount > 0 then Right(()) else Left("--tableCount must be > 0")
        playerCount <- intOpt(options, "playerCount", 2)
        _ <- if playerCount >= 2 && playerCount <= 9 then Right(()) else Left("--playerCount must be in [2,9]")
        reportEvery <- intOpt(options, "reportEvery", 10000)
        _ <- if reportEvery > 0 then Right(()) else Left("--reportEvery must be > 0")
        learnEveryHands <- intOpt(options, "learnEveryHands", 50000)
        _ <- if learnEveryHands >= 0 then Right(()) else Left("--learnEveryHands must be >= 0")
        learningWindowSamples <- intOpt(options, "learningWindowSamples", 200000)
        _ <- if learningWindowSamples >= 0 then Right(()) else Left("--learningWindowSamples must be >= 0")
        seed <- longOpt(options, "seed", 42L)
        outDir <- pathOpt(options, "outDir", Paths.get("data/playing-hall"))
        modelArtifactDir <- optionalPathOpt(options, "modelArtifactDir")
        heroMode <- heroModeOpt(options, "heroStyle", HeroMode.Adaptive)
        heroPosition <- resolveHeroPosition(options, playerCount)
        gtoMode <- gtoModeOpt(options, "gtoMode", GtoMode.Exact)
        villainMode <- villainModeOpt(options, "villainStyle", VillainMode.Archetype(PlayerArchetype.Tag))
        villainPool <- buildVillainPool(villainMode, options.get("villainPool"))
        heroExplorationRate <- doubleOpt(options, "heroExplorationRate", 0.05)
        _ <- if heroExplorationRate >= 0.0 && heroExplorationRate <= 1.0 then Right(())
        else Left("--heroExplorationRate must be in [0,1]")
        raiseSize <- doubleOpt(options, "raiseSize", 2.5)
        _ <- if raiseSize > 0.0 then Right(()) else Left("--raiseSize must be > 0")
        bunchingTrials <- intOpt(options, "bunchingTrials", 80)
        _ <- if bunchingTrials > 0 then Right(()) else Left("--bunchingTrials must be > 0")
        equityTrials <- intOpt(options, "equityTrials", 700)
        _ <- if equityTrials > 0 then Right(()) else Left("--equityTrials must be > 0")
        saveTrainingTsv <- boolOpt(options, "saveTrainingTsv", true)
        saveDdreTrainingTsv <- boolOpt(options, "saveDdreTrainingTsv", false)
        saveReviewHandHistory <- boolOpt(options, "saveReviewHandHistory", false)
        fullRing <- boolOpt(options, "fullRing", false)
      yield Config(
        hands = hands,
        tableCount = tableCount,
        playerCount = playerCount,
        reportEvery = reportEvery,
        learnEveryHands = learnEveryHands,
        learningWindowSamples = learningWindowSamples,
        seed = seed,
        outDir = outDir,
        modelArtifactDir = modelArtifactDir,
        heroMode = heroMode,
        heroPosition = heroPosition,
        gtoMode = gtoMode,
        villainPool = villainPool,
        heroExplorationRate = heroExplorationRate,
        raiseSize = raiseSize,
        bunchingTrials = bunchingTrials,
        equityTrials = equityTrials,
        saveTrainingTsv = saveTrainingTsv,
        saveDdreTrainingTsv = saveDdreTrainingTsv,
        saveReviewHandHistory = saveReviewHandHistory,
        fullRing = fullRing
      )

  private def intOpt(options: Map[String, String], key: String, default: Int): Either[String, Int] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) =>
        raw.toIntOption.toRight(s"--$key must be an integer")

  private def longOpt(options: Map[String, String], key: String, default: Long): Either[String, Long] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) =>
        raw.toLongOption.toRight(s"--$key must be a long")

  private def doubleOpt(options: Map[String, String], key: String, default: Double): Either[String, Double] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) =>
        raw.toDoubleOption.toRight(s"--$key must be a double")

  private def boolOpt(options: Map[String, String], key: String, default: Boolean): Either[String, Boolean] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) =>
        raw.trim.toLowerCase match
          case "true"  => Right(true)
          case "false" => Right(false)
          case _       => Left(s"--$key must be true or false")

  private def pathOpt(options: Map[String, String], key: String, default: Path): Either[String, Path] =
    Right(options.get(key).map(Paths.get(_)).getOrElse(default))

  private def optionalPathOpt(options: Map[String, String], key: String): Either[String, Option[Path]] =
    Right(options.get(key).map(Paths.get(_)))

  private def heroModeOpt(
      options: Map[String, String],
      key: String,
      default: HeroMode
  ): Either[String, HeroMode] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) =>
        raw.trim.toLowerCase match
          case "adaptive" => Right(HeroMode.Adaptive)
          case "gto"      => Right(HeroMode.Gto)
          case _          => Left("--heroStyle must be one of: adaptive, gto")

  private def gtoModeOpt(
      options: Map[String, String],
      key: String,
      default: GtoMode
  ): Either[String, GtoMode] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) =>
        raw.trim.toLowerCase match
          case "fast"  => Right(GtoMode.Fast)
          case "exact" => Right(GtoMode.Exact)
          case _       => Left("--gtoMode must be one of: fast, exact")

  private def villainModeOpt(
      options: Map[String, String],
      key: String,
      default: VillainMode
  ): Either[String, VillainMode] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) =>
        parseVillainModeToken(raw).left.map(_ => "--villainStyle must be one of: nit, tag, lag, callingstation, station, maniac, gto")

  private val usage =
    """Usage:
      |  runMain sicfun.holdem.runtime.TexasHoldemPlayingHall [--key=value ...]
      |
      |  --hands=<int>                 default 100000
      |  --tableCount=<int>            default 1
      |  --playerCount=<int>           table seats to model (2..9, default 2)
      |  --reportEvery=<int>           default 10000
      |  --learnEveryHands=<int>       default 50000 (0 disables learning)
      |  --learningWindowSamples=<int> default 200000 (0 = unbounded)
      |  --seed=<long>                 default 42
      |  --outDir=<path>               default data/playing-hall
      |  --modelArtifactDir=<path>     optional initial trained model
      |  --heroStyle=<style>           adaptive|gto
      |  --heroPosition=<Position>     explicit table position (defaults to Button)
      |  --heroSeat=<seat>             legacy heads-up alias: button|bigblind
      |  --gtoMode=<mode>              fast|exact (default exact)
      |  --villainStyle=<style>        nit|tag|lag|callingstation|station|maniac|gto
      |  --villainPool=<styles>        optional comma-separated villain pool overriding villainStyle
      |  --heroExplorationRate=<double> default 0.05 (epsilon-greedy, [0,1])
      |  --raiseSize=<double>          default 2.5
      |  --bunchingTrials=<int>        default 80
      |  --equityTrials=<int>          default 700
      |  --saveTrainingTsv=<bool>      default true
      |  --saveDdreTrainingTsv=<bool>  default false
      |  --saveReviewHandHistory=<bool> default false
      |  --fullRing=<bool>             default false (all villains always active)
      |""".stripMargin
