package sicfun.holdem

import sicfun.core.{Card, CardId, Deck, DiscreteDistribution, HandEvaluator}

import java.io.BufferedWriter
import java.net.URLEncoder
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
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
  private enum HeroMode:
    case Adaptive
    case Gto

  private enum GtoMode:
    case Fast
    case Exact

  private enum VillainMode:
    case Archetype(style: PlayerArchetype)
    case Gto

  private final case class Config(
      hands: Int,
      tableCount: Int,
      reportEvery: Int,
      learnEveryHands: Int,
      learningWindowSamples: Int,
      seed: Long,
      outDir: Path,
      modelArtifactDir: Option[Path],
      heroMode: HeroMode,
      gtoMode: GtoMode,
      villainMode: VillainMode,
      heroExplorationRate: Double,
      raiseSize: Double,
      bunchingTrials: Int,
      equityTrials: Int,
      saveTrainingTsv: Boolean,
      saveDdreTrainingTsv: Boolean
  )

  final case class HallSummary(
      handsPlayed: Int,
      tableCount: Int,
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

  private final case class Deal(hero: HoleCards, villain: HoleCards, board: Board)

  private final case class HandResult(
      heroNet: Double,
      outcome: Int,
      villainDecision: Option[(GameState, PokerAction)],
      villainTrainingSamples: Vector[(GameState, HoleCards, PokerAction)],
      ddreTrainingSamples: Vector[DdreTrainingSample],
      raiseResponses: Vector[PokerAction],
      heroActions: Vector[PokerAction],
      villainActions: Vector[PokerAction],
      streetsPlayed: Int
  )

  private final case class VillainStyleProfile(looseness: Double, aggression: Double)

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

  private final case class GtoSolveCacheKey(
      perspective: Int,
      canonicalHeroPacked: Long,
      streetOrdinal: Int,
      canonicalBoardPacked: Long,
      potBits: Long,
      toCallBits: Long,
      stackBits: Long,
      betHistoryHash: Int,
      candidateHash: Int,
      baseEquityTrials: Int
  )

  private final case class GtoCachedPolicy(
      orderedActionProbabilities: Vector[(PokerAction, Double)],
      bestAction: PokerAction,
      provider: String
  )

  private final case class GtoCacheStats(
      var hits: Long = 0L,
      var misses: Long = 0L,
      servedByProvider: mutable.Map[String, Long] = mutable.HashMap.empty[String, Long].withDefaultValue(0L),
      solvedByProvider: mutable.Map[String, Long] = mutable.HashMap.empty[String, Long].withDefaultValue(0L)
  ):
    def total: Long = hits + misses
    def hitRate: Double = if total > 0 then hits.toDouble / total.toDouble else 0.0
    def recordHit(provider: String): Unit =
      hits += 1L
      increment(servedByProvider, provider)
    def recordMiss(provider: String): Unit =
      misses += 1L
      increment(servedByProvider, provider)
      increment(solvedByProvider, provider)
    def servedByProviderSnapshot: Map[String, Long] = servedByProvider.toMap
    def solvedByProviderSnapshot: Map[String, Long] = solvedByProvider.toMap
    private def increment(counter: mutable.Map[String, Long], provider: String): Unit =
      counter.update(provider, counter(provider) + 1L)

  private val MaxExactGtoCacheEntries = 500000
  private val SuitPermutations: Array[Array[Int]] =
    Array(
      Array(0, 1, 2, 3), Array(0, 1, 3, 2), Array(0, 2, 1, 3), Array(0, 2, 3, 1),
      Array(0, 3, 1, 2), Array(0, 3, 2, 1), Array(1, 0, 2, 3), Array(1, 0, 3, 2),
      Array(1, 2, 0, 3), Array(1, 2, 3, 0), Array(1, 3, 0, 2), Array(1, 3, 2, 0),
      Array(2, 0, 1, 3), Array(2, 0, 3, 1), Array(2, 1, 0, 3), Array(2, 1, 3, 0),
      Array(2, 3, 0, 1), Array(2, 3, 1, 0), Array(3, 0, 1, 2), Array(3, 0, 2, 1),
      Array(3, 1, 0, 2), Array(3, 1, 2, 0), Array(3, 2, 0, 1), Array(3, 2, 1, 0)
    )

  def main(args: Array[String]): Unit =
    val wantsHelp = args.contains("--help") || args.contains("-h")
    run(args) match
      case Right(summary) =>
        println("=== Texas Hold'em Playing Hall ===")
        println(s"handsPlayed: ${summary.handsPlayed}")
        println(s"tableCount: ${summary.tableCount}")
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
    try
      Files.createDirectories(config.outDir)
      val modelsRoot = config.outDir.resolve("models")
      Files.createDirectories(modelsRoot)

      val handsWriter = openLogWriter(config.outDir.resolve("hands.tsv"))
      val learningWriter = openLogWriter(config.outDir.resolve("learning.tsv"))
      val trainingWriter = if config.saveTrainingTsv then Some(openLogWriter(config.outDir.resolve("training-selfplay.tsv"))) else None
      val ddreWriter = if config.saveDdreTrainingTsv then Some(openLogWriter(config.outDir.resolve("ddre-training-selfplay.tsv"))) else None

      try
        initHandsLog(handsWriter)
        initLearningLog(learningWriter)
        trainingWriter.foreach(initTrainingTsv)
        ddreWriter.foreach(initDdreTrainingTsv)

        val tableRanges = TableRanges.defaults(TableFormat.NineMax)
        val folds = TableFormat.NineMax.foldsBeforeOpener(Position.Button).map(PreflopFold(_))
        val rng = new Random(config.seed)

        val (initialArtifact, _) = loadInitialArtifact(config, modelsRoot)
        var activeArtifact = initialArtifact
        var preflopEngine = buildEngine(
          tableRanges = tableRanges,
          model = activeArtifact.model,
          bunchingTrials = config.bunchingTrials,
          equityTrials = config.equityTrials
        )
        var postflopEngine = buildEngine(
          tableRanges = tableRanges,
          model = activeArtifact.model,
          bunchingTrials = postflopBunchingTrials(config.bunchingTrials),
          equityTrials = postflopEquityTrials(config.equityTrials)
        )

        val learningQueue = mutable.Queue.empty[(GameState, HoleCards, PokerAction)]
        val raiseResponseHistory = mutable.ArrayBuffer.empty[PokerAction]
        val exactGtoCache = mutable.HashMap.empty[GtoSolveCacheKey, GtoCachedPolicy]
        val exactGtoCacheStats = GtoCacheStats()

        var heroNet = 0.0
        var heroWins = 0
        var heroTies = 0
        var heroLosses = 0
        val actionCounts = mutable.Map.empty[String, Int].withDefaultValue(0)
        var retrains = 0

        val collectVillainTraining = config.learnEveryHands > 0 || config.saveTrainingTsv
        val collectDdreTraining = config.saveDdreTrainingTsv
        val villainLabel = villainModeLabel(config.villainMode)

        var handNo = 1
        while handNo <= config.hands do
          val tableId = ((handNo - 1) % config.tableCount) + 1
          val deal = dealHand(rng)
          val result = resolveHand(
            deal = deal,
            preflopEngine = preflopEngine,
            postflopEngine = postflopEngine,
            tableRanges = tableRanges,
            actionModel = activeArtifact.model,
            config = config,
            folds = folds,
            rng = new Random(rng.nextLong()),
            collectVillainTraining = collectVillainTraining,
            collectDdreTraining = collectDdreTraining,
            exactGtoCache = exactGtoCache,
            exactGtoCacheStats = exactGtoCacheStats
          )

          result.villainTrainingSamples.foreach { sample =>
            learningQueue.enqueue(sample)
            if config.learningWindowSamples > 0 then
              while learningQueue.size > config.learningWindowSamples do learningQueue.dequeue()
            trainingWriter.foreach(w => appendTrainingSample(w, handNo, tableId, sample))
          }

          result.ddreTrainingSamples.foreach { sample =>
            ddreWriter.foreach(w => appendDdreTrainingSample(w, handNo, tableId, sample))
          }

          raiseResponseHistory ++= result.raiseResponses

          heroNet += result.heroNet
          if result.outcome > 0 then heroWins += 1
          else if result.outcome < 0 then heroLosses += 1
          else heroTies += 1

          result.heroActions.foreach { action =>
            val key = renderAction(action)
            actionCounts.update(key, actionCounts(key) + 1)
          }

          appendHandLog(
            writer = handsWriter,
            hand = handNo,
            tableId = tableId,
            deal = deal,
            result = result,
            modelId = activeArtifact.version.id,
            archetype = villainLabel
          )

          if config.reportEvery > 0 && (handNo % config.reportEvery == 0 || handNo == config.hands) then
            val bb100 = if handNo > 0 then (heroNet / handNo.toDouble) * 100.0 else 0.0
            println(
              f"[hall] hand=$handNo%,d net=${heroNet}%.2f bb100=$bb100%.2f retrains=$retrains model=${activeArtifact.version.id}"
            )

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
              activeArtifact = candidate
              retrains += 1
              preflopEngine = buildEngine(
                tableRanges = tableRanges,
                model = activeArtifact.model,
                bunchingTrials = config.bunchingTrials,
                equityTrials = config.equityTrials
              )
              postflopEngine = buildEngine(
                tableRanges = tableRanges,
                model = activeArtifact.model,
                bunchingTrials = postflopBunchingTrials(config.bunchingTrials),
                equityTrials = postflopEquityTrials(config.equityTrials)
              )
              raiseResponseHistory.foreach { response =>
                preflopEngine.observeVillainResponseToRaise(response)
                postflopEngine.observeVillainResponseToRaise(response)
              }

          handNo += 1

        if exactGtoCache.size > MaxExactGtoCacheEntries then
          exactGtoCache.clear()

        val bbPer100 =
          if config.hands > 0 then (heroNet / config.hands.toDouble) * 100.0
          else 0.0
        Right(
          HallSummary(
            handsPlayed = config.hands,
            tableCount = config.tableCount,
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
        )
      finally
        closeQuietly(handsWriter)
        closeQuietly(learningWriter)
        trainingWriter.foreach(closeQuietly)
        ddreWriter.foreach(closeQuietly)
    catch
      case e: Exception =>
        Left(s"playing hall failed: ${e.getMessage}")

  private def resolveHand(
      deal: Deal,
      preflopEngine: RealTimeAdaptiveEngine,
      postflopEngine: RealTimeAdaptiveEngine,
      tableRanges: TableRanges,
      actionModel: PokerActionModel,
      config: Config,
      folds: Vector[PreflopFold],
      rng: Random,
      collectVillainTraining: Boolean,
      collectDdreTraining: Boolean,
      exactGtoCache: mutable.HashMap[GtoSolveCacheKey, GtoCachedPolicy],
      exactGtoCacheStats: GtoCacheStats
  ): HandResult =
    var heroStack = 99.5
    var villainStack = 99.0
    var heroContribution = 0.5
    var villainContribution = 1.0
    var pot = heroContribution + villainContribution
    var betHistory = Vector.empty[BetAction]

    val villainTrainingSamples = mutable.ArrayBuffer.empty[(GameState, HoleCards, PokerAction)]
    val ddreTrainingSamples = mutable.ArrayBuffer.empty[DdreTrainingSample]
    val raiseResponses = mutable.ArrayBuffer.empty[PokerAction]
    val heroActions = mutable.ArrayBuffer.empty[PokerAction]
    val villainActions = mutable.ArrayBuffer.empty[PokerAction]
    val villainObservations = mutable.ArrayBuffer.empty[VillainObservation]

    var firstVillainDecision: Option[(GameState, PokerAction)] = None
    var outcome = 0
    var handOver = false
    var streetsPlayed = 0

    def boardFor(street: Street): Board =
      street match
        case Street.Preflop => Board.empty
        case Street.Flop    => Board.from(deal.board.cards.take(3).sortBy(CardId.toId))
        case Street.Turn    => Board.from(deal.board.cards.take(4).sortBy(CardId.toId))
        case Street.River   => deal.board

    def payHero(amount: Double): Double =
      val paid = math.max(0.0, math.min(amount, heroStack))
      heroStack -= paid
      heroContribution += paid
      pot += paid
      paid

    def payVillain(amount: Double): Double =
      val paid = math.max(0.0, math.min(amount, villainStack))
      villainStack -= paid
      villainContribution += paid
      pot += paid
      paid

    def recordVillainDecision(state: GameState, action: PokerAction): Unit =
      if collectVillainTraining then
        villainTrainingSamples += ((state, deal.villain, action))
      villainActions += action
      villainObservations += VillainObservation(action, state)
      if firstVillainDecision.isEmpty then firstVillainDecision = Some((state, action))

    def decideHero(
        street: Street,
        board: Board,
        toCall: Double,
        allowRaise: Boolean
    ): PokerAction =
      val state = GameState(
        street = street,
        board = board,
        pot = pot,
        toCall = toCall,
        position = Position.Button,
        stackSize = heroStack,
        betHistory = betHistory
      )
      val candidates = heroCandidates(state, config.raiseSize, allowRaise)
      val sampled = config.heroMode match
        case HeroMode.Adaptive =>
          val engine = if street == Street.Preflop then preflopEngine else postflopEngine
          val decisionBudgetMillis = Some(1L)
          val adaptiveDecision = engine.decide(
            hero = deal.hero,
            state = state,
            folds = folds,
            villainPos = Position.BigBlind,
            observations = villainObservations.toVector,
            candidateActions = candidates,
            decisionBudgetMillis = decisionBudgetMillis,
            rng = new Random(rng.nextLong())
          )
          val greedy = adaptiveDecision.decision.recommendation.bestAction
          if rng.nextDouble() < config.heroExplorationRate then candidates(rng.nextInt(candidates.length))
          else greedy
        case HeroMode.Gto =>
          gtoHeroResponds(
            hand = deal.hero,
            state = state,
            allowRaise = allowRaise,
            raiseSize = config.raiseSize,
            mode = config.gtoMode,
            tableRanges = tableRanges,
            baseEquityTrials = config.equityTrials,
            rng = rng,
            perspective = 0,
            exactGtoCache = exactGtoCache,
            exactGtoCacheStats = exactGtoCacheStats
          )
      val normalized = normalizeAction(
        action = sampled,
        toCall = toCall,
        allowRaise = allowRaise,
        defaultRaise = config.raiseSize
      )
      if collectDdreTraining then
        val observationsSnapshot = villainObservations.toVector
        val observationsForBayes = observationsSnapshot.map(obs => obs.action -> obs.state)
        val labelBunchingTrials =
          if street == Street.Preflop then config.bunchingTrials
          else postflopBunchingTrials(config.bunchingTrials)
        val prior = RangeInferenceEngine
          .inferPosterior(
            hero = deal.hero,
            board = state.board,
            folds = folds,
            tableRanges = tableRanges,
            villainPos = Position.BigBlind,
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
          villainPosition = Position.BigBlind,
          villainStackBefore = villainStack,
          observations = observationsSnapshot,
          heroHole = deal.hero,
          villainHole = deal.villain,
          prior = prior,
          bayesPosterior = bayesPosterior,
          bayesLogEvidence = bayesLogEvidence
        )
      heroActions += normalized
      normalized

    def decideVillain(
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
        position = Position.BigBlind,
        stackSize = villainStack,
        betHistory = betHistory
      )
      val sampled = config.villainMode match
        case VillainMode.Archetype(style) =>
          villainResponds(
            hand = deal.villain,
            style = style,
            state = state,
            allowRaise = allowRaise,
            raiseSize = config.raiseSize,
            rng = rng
          )
        case VillainMode.Gto =>
          gtoVillainResponds(
            hand = deal.villain,
            state = state,
            allowRaise = allowRaise,
            raiseSize = config.raiseSize,
            mode = config.gtoMode,
            tableRanges = tableRanges,
            baseEquityTrials = config.equityTrials,
            rng = rng,
            perspective = 1,
            exactGtoCache = exactGtoCache,
            exactGtoCacheStats = exactGtoCacheStats
          )
      val action = normalizeAction(
        action = sampled,
        toCall = toCall,
        allowRaise = allowRaise,
        defaultRaise = config.raiseSize
      )
      recordVillainDecision(state, action)
      if facingHeroRaise then
        action match
          case PokerAction.Fold | PokerAction.Call | PokerAction.Raise(_) =>
            preflopEngine.observeVillainResponseToRaise(action)
            postflopEngine.observeVillainResponseToRaise(action)
            raiseResponses += action
          case _ => ()
      action

    if !handOver then
      streetsPlayed += 1
      val board = boardFor(Street.Preflop)
      val heroAction = decideHero(Street.Preflop, board, toCall = 0.5, allowRaise = true)
      betHistory = betHistory :+ BetAction(0, heroAction)
      heroAction match
        case PokerAction.Fold =>
          handOver = true
          outcome = -1
        case PokerAction.Check =>
          val paid = payHero(0.5)
          if paid < 0.5 then
            handOver = true
            outcome = -1
        case PokerAction.Call =>
          val paid = payHero(0.5)
          if paid < 0.5 then
            handOver = true
            outcome = -1
        case PokerAction.Raise(amount) =>
          val paid = payHero(0.5 + amount)
          val villainToCall = math.max(0.0, paid - 0.5)
          if villainToCall <= 0.0 then
            handOver = true
            outcome = 1
          else
            val villainAction = decideVillain(
              street = Street.Preflop,
              board = board,
              toCall = villainToCall,
              allowRaise = true,
              facingHeroRaise = true
            )
            betHistory = betHistory :+ BetAction(1, villainAction)
            villainAction match
              case PokerAction.Fold =>
                handOver = true
                outcome = 1
              case PokerAction.Call =>
                payVillain(villainToCall)
              case PokerAction.Check =>
                payVillain(villainToCall)
              case PokerAction.Raise(reRaiseAmount) =>
                val paidVillain = payVillain(villainToCall + reRaiseAmount)
                val heroToCall = math.max(0.0, paidVillain - villainToCall)
                val heroResponse = decideHero(
                  street = Street.Preflop,
                  board = board,
                  toCall = heroToCall,
                  allowRaise = false
                )
                betHistory = betHistory :+ BetAction(0, heroResponse)
                heroResponse match
                  case PokerAction.Fold =>
                    handOver = true
                    outcome = -1
                  case PokerAction.Call | PokerAction.Check | PokerAction.Raise(_) =>
                    val paidHero = payHero(heroToCall)
                    if paidHero <= 0.0 then
                      handOver = true
                      outcome = -1

    def playPostflopStreet(street: Street): Unit =
      if handOver then ()
      else
        streetsPlayed += 1
        val board = boardFor(street)
        val villainLead = decideVillain(
          street = street,
          board = board,
          toCall = 0.0,
          allowRaise = true,
          facingHeroRaise = false
        )
        betHistory = betHistory :+ BetAction(1, villainLead)
        villainLead match
          case PokerAction.Check =>
            val heroAction = decideHero(street, board, toCall = 0.0, allowRaise = true)
            betHistory = betHistory :+ BetAction(0, heroAction)
            heroAction match
              case PokerAction.Check | PokerAction.Call =>
                ()
              case PokerAction.Fold =>
                handOver = true
                outcome = -1
              case PokerAction.Raise(amount) =>
                val paidHero = payHero(amount)
                val villainToCall = paidHero
                if villainToCall <= 0.0 then ()
                else
                  val villainResponse = decideVillain(
                    street = street,
                    board = board,
                    toCall = villainToCall,
                    allowRaise = false,
                    facingHeroRaise = true
                  )
                  betHistory = betHistory :+ BetAction(1, villainResponse)
                  villainResponse match
                    case PokerAction.Fold =>
                      handOver = true
                      outcome = 1
                    case PokerAction.Call | PokerAction.Check =>
                      payVillain(villainToCall)
                    case PokerAction.Raise(reRaiseAmount) =>
                      val paidVillain = payVillain(villainToCall + reRaiseAmount)
                      val heroToCall = math.max(0.0, paidVillain - villainToCall)
                      val heroResponse = decideHero(
                        street = street,
                        board = board,
                        toCall = heroToCall,
                        allowRaise = false
                      )
                      betHistory = betHistory :+ BetAction(0, heroResponse)
                      heroResponse match
                        case PokerAction.Fold =>
                          handOver = true
                          outcome = -1
                        case PokerAction.Call | PokerAction.Check | PokerAction.Raise(_) =>
                          payHero(heroToCall)
          case PokerAction.Raise(amount) =>
            val paidVillain = payVillain(amount)
            val heroToCall = paidVillain
            if heroToCall <= 0.0 then ()
            else
              val heroAction = decideHero(street, board, toCall = heroToCall, allowRaise = false)
              betHistory = betHistory :+ BetAction(0, heroAction)
              heroAction match
                case PokerAction.Fold =>
                  handOver = true
                  outcome = -1
                case PokerAction.Call | PokerAction.Check =>
                  payHero(heroToCall)
                case PokerAction.Raise(reRaiseAmount) =>
                  val paidHero = payHero(heroToCall + reRaiseAmount)
                  val villainToCall = math.max(0.0, paidHero - heroToCall)
                  if villainToCall <= 0.0 then ()
                  else
                    val villainResponse = decideVillain(
                      street = street,
                      board = board,
                      toCall = villainToCall,
                      allowRaise = false,
                      facingHeroRaise = true
                    )
                    betHistory = betHistory :+ BetAction(1, villainResponse)
                    villainResponse match
                      case PokerAction.Fold =>
                        handOver = true
                        outcome = 1
                      case PokerAction.Call | PokerAction.Check | PokerAction.Raise(_) =>
                        payVillain(villainToCall)
          case PokerAction.Call =>
            ()
          case PokerAction.Fold =>
            handOver = true
            outcome = 1

    if !handOver then playPostflopStreet(Street.Flop)
    if !handOver then playPostflopStreet(Street.Turn)
    if !handOver then playPostflopStreet(Street.River)

    if !handOver then
      outcome = showdownOutcome(deal)

    val heroNet =
      if handOver && outcome > 0 then pot - heroContribution
      else if handOver && outcome < 0 then -heroContribution
      else outcomeToNet(outcome, pot, heroContribution)

    HandResult(
      heroNet = heroNet,
      outcome = outcome,
      villainDecision = firstVillainDecision,
      villainTrainingSamples = villainTrainingSamples.toVector,
      ddreTrainingSamples = ddreTrainingSamples.toVector,
      raiseResponses = raiseResponses.toVector,
      heroActions = heroActions.toVector,
      villainActions = villainActions.toVector,
      streetsPlayed = streetsPlayed
    )

  private def outcomeToNet(outcome: Int, pot: Double, heroContribution: Double): Double =
    if outcome > 0 then pot - heroContribution
    else if outcome < 0 then -heroContribution
    else (pot * 0.5) - heroContribution

  private def minEquityTrials(configured: Int): Int =
    math.max(1, configured / 30)

  private def postflopBunchingTrials(configured: Int): Int =
    1

  private def postflopEquityTrials(configured: Int): Int =
    math.max(8, configured / 16)

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

  private def showdownOutcome(deal: Deal): Int =
    val heroRank = HandEvaluator.evaluate7Cached(deal.hero.toVector ++ deal.board.cards)
    val villainRank = HandEvaluator.evaluate7Cached(deal.villain.toVector ++ deal.board.cards)
    heroRank.compare(villainRank)

  private def villainResponds(
      hand: HoleCards,
      style: PlayerArchetype,
      state: GameState,
      allowRaise: Boolean,
      raiseSize: Double,
      rng: Random
  ): PokerAction =
    val profile = styleProfile(style)
    val strength = streetStrength(hand, state.board, state.street, rng)

    if state.toCall <= 0.0 then
      if !allowRaise then PokerAction.Check
      else
        val betChance = clamp((strength - 0.35) * 0.9 + (profile.aggression * 0.35), 0.02, 0.92)
        if rng.nextDouble() < betChance then PokerAction.Raise(raiseSize)
        else PokerAction.Check
    else
      val potOdds = state.potOdds
      val foldThreshold = clamp(potOdds + (0.35 - profile.looseness * 0.2), 0.05, 0.95)
      val raiseThreshold = clamp(0.68 - profile.aggression * 0.12 + potOdds * 0.2, 0.35, 0.9)
      if allowRaise && strength >= raiseThreshold && rng.nextDouble() < (0.15 + profile.aggression * 0.55) then
        PokerAction.Raise(raiseSize)
      else if strength < foldThreshold then PokerAction.Fold
      else PokerAction.Call

  private def gtoHeroResponds(
      hand: HoleCards,
      state: GameState,
      allowRaise: Boolean,
      raiseSize: Double,
      mode: GtoMode,
      tableRanges: TableRanges,
      baseEquityTrials: Int,
      rng: Random,
      perspective: Int,
      exactGtoCache: mutable.HashMap[GtoSolveCacheKey, GtoCachedPolicy],
      exactGtoCacheStats: GtoCacheStats
  ): PokerAction =
    val candidates = heroCandidates(state, raiseSize, allowRaise)
    if candidates.length <= 1 then candidates.head
    else
      mode match
        case GtoMode.Fast =>
          fastGtoResponds(
            hand = hand,
            state = state,
            candidates = candidates,
            allowRaise = allowRaise,
            rng = rng
          )
        case GtoMode.Exact =>
          val villainPosterior = villainPosteriorForHeroGto(hero = hand, board = state.board, tableRanges = tableRanges)
          solveGtoByCfr(
            hand = hand,
            state = state,
            candidates = candidates,
            villainPosterior = villainPosterior,
            baseEquityTrials = baseEquityTrials,
            rng = rng,
            perspective = perspective,
            exactGtoCache = exactGtoCache,
            exactGtoCacheStats = exactGtoCacheStats
          )

  private def gtoVillainResponds(
      hand: HoleCards,
      state: GameState,
      allowRaise: Boolean,
      raiseSize: Double,
      mode: GtoMode,
      tableRanges: TableRanges,
      baseEquityTrials: Int,
      rng: Random,
      perspective: Int,
      exactGtoCache: mutable.HashMap[GtoSolveCacheKey, GtoCachedPolicy],
      exactGtoCacheStats: GtoCacheStats
  ): PokerAction =
    val candidates = heroCandidates(state, raiseSize, allowRaise)
    if candidates.length <= 1 then candidates.head
    else
      mode match
        case GtoMode.Fast =>
          fastGtoResponds(
            hand = hand,
            state = state,
            candidates = candidates,
            allowRaise = allowRaise,
            rng = rng
          )
        case GtoMode.Exact =>
          val heroPosterior = heroPosteriorForGto(villain = hand, board = state.board, tableRanges = tableRanges)
          solveGtoByCfr(
            hand = hand,
            state = state,
            candidates = candidates,
            villainPosterior = heroPosterior,
            baseEquityTrials = baseEquityTrials,
            rng = rng,
            perspective = perspective,
            exactGtoCache = exactGtoCache,
            exactGtoCacheStats = exactGtoCacheStats
          )

  private def solveGtoByCfr(
      hand: HoleCards,
      state: GameState,
      candidates: Vector[PokerAction],
      villainPosterior: DiscreteDistribution[HoleCards],
      baseEquityTrials: Int,
      rng: Random,
      perspective: Int,
      exactGtoCache: mutable.HashMap[GtoSolveCacheKey, GtoCachedPolicy],
      exactGtoCacheStats: GtoCacheStats
  ): PokerAction =
    val canonicalSignature = canonicalHeroBoardSignature(hand = hand, board = state.board)
    val key = buildGtoSolveCacheKey(
      perspective = perspective,
      hand = hand,
      state = state,
      candidates = candidates,
      baseEquityTrials = baseEquityTrials,
      canonicalSignature = canonicalSignature
    )
    exactGtoCache.get(key) match
      case Some(cached) =>
        exactGtoCacheStats.recordHit(cached.provider)
        sampleActionByPolicy(
          ordered = cached.orderedActionProbabilities,
          fallback = cached.bestAction,
          rng = rng
        )
      case None =>
        val config = HoldemCfrConfig(
          iterations = gtoIterations(state.street, baseEquityTrials, candidates.length),
          maxVillainHands = gtoMaxVillainHands(state.street, candidates.length),
          equityTrials = gtoEquityTrials(state.street, baseEquityTrials, candidates.length),
          rngSeed = exactEquitySeed(
            perspective = perspective,
            baseEquityTrials = baseEquityTrials,
            boardSize = state.board.size,
            canonicalSignature = canonicalSignature
          )
        )
        try
          val solution = HoldemCfrSolver.solveShallowDecisionPolicy(
            hero = hand,
            state = state,
            villainPosterior = villainPosterior,
            candidateActions = candidates,
            config = config
          )
          val orderedActionProbabilities =
            orderedPositiveProbabilities(
              actions = candidates,
              probabilities = solution.actionProbabilities
            )
          exactGtoCacheStats.recordMiss(solution.provider)
          if exactGtoCache.size >= MaxExactGtoCacheEntries then exactGtoCache.clear()
          exactGtoCache.update(
            key,
            GtoCachedPolicy(
              orderedActionProbabilities = orderedActionProbabilities,
              bestAction = solution.bestAction,
              provider = solution.provider
            )
          )
          sampleActionByPolicy(
            ordered = orderedActionProbabilities,
            fallback = solution.bestAction,
            rng = rng
          )
        catch
          case _: Throwable =>
            // Preserve run continuity if a specific CFR solve fails.
            exactGtoCacheStats.recordMiss("random-fallback")
            candidates(rng.nextInt(candidates.length))

  private def fastGtoResponds(
      hand: HoleCards,
      state: GameState,
      candidates: Vector[PokerAction],
      allowRaise: Boolean,
      rng: Random
  ): PokerAction =
    val strength = fastGtoStrength(hand, state.board, state.street)
    val raiseCandidate =
      if allowRaise then candidates.collectFirst { case action @ PokerAction.Raise(_) => action }
      else None
    val callCandidate = candidates.find(_ == PokerAction.Call)
    val foldCandidate = candidates.find(_ == PokerAction.Fold)
    if state.toCall <= 0.0 then
      raiseCandidate match
        case None => PokerAction.Check
        case Some(raiseAction) =>
          val pureRaiseThreshold = fastGtoRaiseThreshold(state.street)
          val mixRaiseThreshold = pureRaiseThreshold - 0.18
          if strength >= pureRaiseThreshold then raiseAction
          else if strength >= mixRaiseThreshold then
            val mix = clamp(0.18 + ((strength - mixRaiseThreshold) * 1.7), 0.05, 0.80)
            if rng.nextDouble() < mix then raiseAction else PokerAction.Check
          else PokerAction.Check
    else
      val potOdds = state.potOdds
      val foldThreshold = clamp(potOdds + fastGtoFoldMargin(state.street), 0.06, 0.95)
      val raiseThreshold = clamp(foldThreshold + fastGtoRaiseGap(state.street), 0.24, 0.98)
      if raiseCandidate.nonEmpty && strength >= raiseThreshold then
        val raiseMix = clamp(0.20 + ((strength - raiseThreshold) * 1.3), 0.10, 0.92)
        if rng.nextDouble() < raiseMix then raiseCandidate.get
        else callCandidate.getOrElse(PokerAction.Call)
      else if strength >= foldThreshold then
        callCandidate.getOrElse(PokerAction.Call)
      else
        foldCandidate.getOrElse(PokerAction.Fold)

  private def fastGtoStrength(hand: HoleCards, board: Board, street: Street): Double =
    val pre = preflopStrength(hand)
    if street == Street.Preflop || board.cards.isEmpty then pre
    else
      val madeWeight =
        street match
          case Street.Flop  => 0.50
          case Street.Turn  => 0.56
          case Street.River => 0.62
          case _            => 0.50
      val categoryScore = bestCategoryStrength(hand, board)
      val drawBonus = drawPotential(hand, board)
      clamp(((1.0 - madeWeight) * pre) + (madeWeight * categoryScore) + drawBonus)

  private def fastGtoRaiseThreshold(street: Street): Double =
    street match
      case Street.Preflop => 0.78
      case Street.Flop    => 0.74
      case Street.Turn    => 0.71
      case Street.River   => 0.68

  private def fastGtoFoldMargin(street: Street): Double =
    street match
      case Street.Preflop => 0.05
      case Street.Flop    => 0.03
      case Street.Turn    => 0.01
      case Street.River   => -0.01

  private def fastGtoRaiseGap(street: Street): Double =
    street match
      case Street.Preflop => 0.27
      case Street.Flop    => 0.24
      case Street.Turn    => 0.22
      case Street.River   => 0.20

  private def villainPosteriorForHeroGto(
      hero: HoleCards,
      board: Board,
      tableRanges: TableRanges
  ): DiscreteDistribution[HoleCards] =
    tableRanges.rangeFor(Position.BigBlind)

  private def heroPosteriorForGto(
      villain: HoleCards,
      board: Board,
      tableRanges: TableRanges
  ): DiscreteDistribution[HoleCards] =
    tableRanges.rangeFor(Position.Button)

  private def gtoIterations(
      street: Street,
      baseEquityTrials: Int,
      candidateCount: Int
  ): Int =
    val base = math.max(72, math.min(224, math.round(baseEquityTrials / 3.0).toInt))
    val streetBase =
      street match
        case Street.Preflop => base + 32
        case Street.Flop    => base
        case Street.Turn    => math.max(72, math.round(base * 0.85).toInt)
        case Street.River   => math.max(56, math.round(base * 0.70).toInt)
    if candidateCount <= 2 then
      val floor =
        street match
          case Street.Preflop => 88
          case Street.Flop    => 64
          case Street.Turn    => 56
          case Street.River   => 48
      math.max(floor, math.round(streetBase * 0.60).toInt)
    else
      streetBase

  private def gtoMaxVillainHands(
      street: Street,
      candidateCount: Int
  ): Int =
    val base =
      street match
        case Street.Preflop => 72
        case Street.Flop    => 40
        case Street.Turn    => 32
        case Street.River   => 24
    if candidateCount <= 2 then math.max(24, base - 16) else base

  private def gtoEquityTrials(
      street: Street,
      baseEquityTrials: Int,
      candidateCount: Int
  ): Int =
    val base =
      street match
        case Street.Preflop => math.max(96, baseEquityTrials / 2)
        case Street.Flop    => math.max(56, baseEquityTrials / 5)
        case Street.Turn    => math.max(40, baseEquityTrials / 6)
        case Street.River   => 32
    if candidateCount <= 2 then
      val floor =
        street match
          case Street.Preflop => 80
          case Street.Flop    => 44
          case Street.Turn    => 32
          case Street.River   => 24
      math.max(floor, math.round(base * 0.70).toInt)
    else
      base

  private def buildGtoSolveCacheKey(
      perspective: Int,
      hand: HoleCards,
      state: GameState,
      candidates: Vector[PokerAction],
      baseEquityTrials: Int,
      canonicalSignature: (Long, Long)
  ): GtoSolveCacheKey =
    val (canonicalHeroPacked, canonicalBoardPacked) = canonicalSignature
    GtoSolveCacheKey(
      perspective = perspective,
      canonicalHeroPacked = canonicalHeroPacked,
      streetOrdinal = state.street.ordinal,
      canonicalBoardPacked = canonicalBoardPacked,
      potBits = java.lang.Double.doubleToLongBits(state.pot),
      toCallBits = java.lang.Double.doubleToLongBits(state.toCall),
      stackBits = java.lang.Double.doubleToLongBits(state.stackSize),
      betHistoryHash = hashBetHistory(state.betHistory),
      candidateHash = hashActions(candidates),
      baseEquityTrials = baseEquityTrials
    )

  private def exactEquitySeed(
      perspective: Int,
      baseEquityTrials: Int,
      boardSize: Int,
      canonicalSignature: (Long, Long)
  ): Long =
    val (canonicalHeroPacked, canonicalBoardPacked) = canonicalSignature
    mix64(
      canonicalHeroPacked ^
        java.lang.Long.rotateLeft(canonicalBoardPacked, 11) ^
        (perspective.toLong << 48) ^
        (baseEquityTrials.toLong << 16) ^
        boardSize.toLong
    )

  private def canonicalHeroBoardSignature(hand: HoleCards, board: Board): (Long, Long) =
    val boardSize = board.cards.length
    val remappedBoardIds = new Array[Int](boardSize)
    var bestHeroPacked = Long.MaxValue
    var bestBoardPacked = Long.MaxValue
    var permIdx = 0
    while permIdx < SuitPermutations.length do
      val suitMap = SuitPermutations(permIdx)
      val heroFirstId = remapCardId(hand.first, suitMap)
      val heroSecondId = remapCardId(hand.second, suitMap)
      val lowHero = math.min(heroFirstId, heroSecondId)
      val highHero = math.max(heroFirstId, heroSecondId)
      val heroPacked = ((lowHero.toLong << 6) | highHero.toLong) & 0xFFFL

      var idx = 0
      while idx < boardSize do
        remappedBoardIds(idx) = remapCardId(board.cards(idx), suitMap)
        idx += 1
      java.util.Arrays.sort(remappedBoardIds)
      var boardPacked = boardSize.toLong
      idx = 0
      while idx < boardSize do
        boardPacked = (boardPacked << 6) | remappedBoardIds(idx).toLong
        idx += 1

      if heroPacked < bestHeroPacked || (heroPacked == bestHeroPacked && boardPacked < bestBoardPacked) then
        bestHeroPacked = heroPacked
        bestBoardPacked = boardPacked
      permIdx += 1
    (bestHeroPacked, bestBoardPacked)

  private def remapCardId(card: Card, suitMap: Array[Int]): Int =
    val mappedSuit = suitMap(card.suit.ordinal)
    (mappedSuit * 13) + card.rank.ordinal

  private def hashBetHistory(history: Vector[BetAction]): Int =
    var hash = 1
    var idx = 0
    while idx < history.length do
      val item = history(idx)
      hash = 31 * hash + item.player
      hash = 31 * hash + hashAction(item.action)
      idx += 1
    hash

  private def hashActions(actions: Vector[PokerAction]): Int =
    var hash = 1
    var idx = 0
    while idx < actions.length do
      hash = 31 * hash + hashAction(actions(idx))
      idx += 1
    hash

  private def hashAction(action: PokerAction): Int =
    action match
      case PokerAction.Fold => 1
      case PokerAction.Check => 2
      case PokerAction.Call => 3
      case PokerAction.Raise(amount) =>
        31 * 4 + java.lang.Double.hashCode(amount)

  private def orderedPositiveProbabilities(
      actions: Vector[PokerAction],
      probabilities: Map[PokerAction, Double]
  ): Vector[(PokerAction, Double)] =
    actions.flatMap { action =>
      val probability = probabilities.getOrElse(action, 0.0)
      if probability.isFinite && probability > 0.0 then Some(action -> probability)
      else None
    }

  private def sampleActionByPolicy(
      ordered: Vector[(PokerAction, Double)],
      fallback: PokerAction,
      rng: Random
  ): PokerAction =
    val total = ordered.map(_._2).sum
    if total <= 0.0 then fallback
    else
      val target = rng.nextDouble() * total
      var cumulative = 0.0
      var idx = 0
      while idx < ordered.length do
        val (action, probability) = ordered(idx)
        cumulative += probability
        if target <= cumulative then return action
        idx += 1
      ordered.last._1

  private def formatLongCountMap(counts: Map[String, Long]): String =
    if counts.isEmpty then "{}"
    else
      counts.toVector
        .sortBy { case (provider, count) => (-count, provider) }
        .map { case (provider, count) => s"$provider:$count" }
        .mkString("{", ", ", "}")

  private def mix64(value: Long): Long =
    var z = value + 0x9E3779B97F4A7C15L
    z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L
    z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL
    z ^ (z >>> 31)

  private def preflopStrength(hand: HoleCards): Double =
    val r1 = hand.first.rank.value
    val r2 = hand.second.rank.value
    val high = math.max(r1, r2).toDouble / 14.0
    val low = math.min(r1, r2).toDouble / 14.0
    val pairBonus = if r1 == r2 then 0.30 + (high * 0.20) else 0.0
    val suitedBonus = if hand.first.suit == hand.second.suit then 0.06 else 0.0
    val gap = math.abs(r1 - r2)
    val connectorBonus =
      if gap == 0 then 0.0
      else if gap == 1 then 0.08
      else if gap == 2 then 0.04
      else 0.0
    clamp((0.45 * high) + (0.18 * low) + pairBonus + suitedBonus + connectorBonus)

  private def streetStrength(
      hand: HoleCards,
      board: Board,
      street: Street,
      rng: Random
  ): Double =
    val pre = preflopStrength(hand)
    if street == Street.Preflop || board.cards.isEmpty then
      clamp(pre + (rng.nextDouble() - 0.5) * 0.04)
    else
      val categoryScore = bestCategoryStrength(hand, board)
      val drawBonus = drawPotential(hand, board)
      val noise = (rng.nextDouble() - 0.5) * 0.05
      clamp(0.45 * pre + 0.45 * categoryScore + drawBonus + noise)

  private def bestCategoryStrength(hand: HoleCards, board: Board): Double =
    val cards = hand.toVector ++ board.cards
    cards.length match
      case 5 =>
        HandEvaluator.evaluate5Cached(cards).category.strength.toDouble / 8.0
      case 6 =>
        HoldemCombinator.combinations(cards.toIndexedSeq, 5).map { combo =>
          HandEvaluator.evaluate5Cached(combo).category.strength.toDouble / 8.0
        }.max
      case 7 =>
        HandEvaluator.evaluate7Cached(cards).category.strength.toDouble / 8.0
      case _ =>
        preflopStrength(hand)

  private def drawPotential(hand: HoleCards, board: Board): Double =
    if board.cards.isEmpty then 0.0
    else
      val all = hand.toVector ++ board.cards
      val bySuit = all.groupBy(_.suit).view.mapValues(_.size).toMap
      val maxSuit = bySuit.values.max
      val flushDrawBonus =
        if maxSuit >= 5 then 0.12
        else if maxSuit == 4 then 0.08
        else if maxSuit == 3 && board.size <= 3 then 0.03
        else 0.0

      val ranks = all.map(_.rank.value).distinct.sorted
      val straightDrawBonus =
        if ranks.length >= 4 && hasTightRun(ranks) then 0.05
        else 0.0

      val pairWithBoardBonus =
        if board.cards.exists(card => card.rank == hand.first.rank || card.rank == hand.second.rank) then 0.04
        else 0.0

      flushDrawBonus + straightDrawBonus + pairWithBoardBonus

  private def hasTightRun(sortedRanks: Seq[Int]): Boolean =
    if sortedRanks.length < 4 then false
    else
      val span4 = HoldemCombinator.combinations(sortedRanks.toIndexedSeq, 4).exists { combo =>
        combo.last - combo.head <= 4
      }
      val withWheelAce =
        if sortedRanks.contains(14) then
          val lowAce = sortedRanks.map(r => if r == 14 then 1 else r).sorted
          HoldemCombinator.combinations(lowAce.toIndexedSeq, 4).exists { combo =>
            combo.last - combo.head <= 4
          }
        else false
      span4 || withWheelAce

  private def styleProfile(archetype: PlayerArchetype): VillainStyleProfile =
    archetype match
      case PlayerArchetype.Nit            => VillainStyleProfile(looseness = 0.20, aggression = 0.18)
      case PlayerArchetype.Tag            => VillainStyleProfile(looseness = 0.45, aggression = 0.40)
      case PlayerArchetype.Lag            => VillainStyleProfile(looseness = 0.68, aggression = 0.66)
      case PlayerArchetype.CallingStation => VillainStyleProfile(looseness = 0.86, aggression = 0.24)
      case PlayerArchetype.Maniac         => VillainStyleProfile(looseness = 0.80, aggression = 0.92)

  private def villainModeLabel(mode: VillainMode): String =
    mode match
      case VillainMode.Archetype(style) => style.toString
      case VillainMode.Gto              => "gto"

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

  private def normalizeAction(
      action: PokerAction,
      toCall: Double,
      allowRaise: Boolean,
      defaultRaise: Double
  ): PokerAction =
    action match
      case PokerAction.Check =>
        if toCall <= 0.0 then PokerAction.Check else PokerAction.Call
      case PokerAction.Call =>
        if toCall > 0.0 then PokerAction.Call else PokerAction.Check
      case PokerAction.Fold =>
        if toCall > 0.0 then PokerAction.Fold else PokerAction.Check
      case PokerAction.Raise(amount) =>
        if !allowRaise then
          if toCall > 0.0 then PokerAction.Call else PokerAction.Check
        else
          val cleanAmount = if amount > 0.0 then amount else defaultRaise
          if cleanAmount <= 0.0 then
            if toCall > 0.0 then PokerAction.Call else PokerAction.Check
          else PokerAction.Raise(cleanAmount)

  private def clamp(value: Double, lo: Double = 0.0, hi: Double = 1.0): Double =
    math.max(lo, math.min(hi, value))

  private def dealHand(rng: Random): Deal =
    val cards = Deck.full.toArray
    shufflePrefix(cards, 9, rng)
    val hero = HoleCards.from(Vector(cards(0), cards(2)))
    val villain = HoleCards.from(Vector(cards(1), cards(3)))
    val board = Board.from(cards.slice(4, 9).toVector.sortBy(CardId.toId))
    Deal(hero, villain, board)

  private def shufflePrefix[A](array: Array[A], count: Int, rng: Random): Unit =
    val limit = math.min(count, array.length)
    var i = 0
    while i < limit do
      val j = i + rng.nextInt(array.length - i)
      val tmp = array(i)
      array(i) = array(j)
      array(j) = tmp
      i += 1

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
      archetype: String
  ): Unit =
    val boardToken = deal.board.cards.map(_.toToken).mkString(" ")
    val heroAction = result.heroActions.headOption.map(renderAction).getOrElse("-")
    val villainAction = result.villainActions.headOption.map(renderAction).getOrElse("-")
    val heroActionTrace = if result.heroActions.nonEmpty then result.heroActions.map(renderAction).mkString("|") else "-"
    val villainActionTrace = if result.villainActions.nonEmpty then result.villainActions.map(renderAction).mkString("|") else "-"
    val outcomeLabel =
      if result.outcome > 0 then "win"
      else if result.outcome < 0 then "loss"
      else "tie"
    val row = Vector(
      hand.toString,
      tableId.toString,
      deal.hero.toToken,
      deal.villain.toToken,
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

  private def parseArgs(args: Array[String]): Either[String, Config] =
    if args.contains("--help") || args.contains("-h") then Left(usage)
    else
      for
        options <- parseOptions(args)
        hands <- intOpt(options, "hands", 100000)
        _ <- if hands > 0 then Right(()) else Left("--hands must be > 0")
        tableCount <- intOpt(options, "tableCount", 1)
        _ <- if tableCount > 0 then Right(()) else Left("--tableCount must be > 0")
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
        gtoMode <- gtoModeOpt(options, "gtoMode", GtoMode.Exact)
        villainMode <- villainModeOpt(options, "villainStyle", VillainMode.Archetype(PlayerArchetype.Tag))
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
      yield Config(
        hands = hands,
        tableCount = tableCount,
        reportEvery = reportEvery,
        learnEveryHands = learnEveryHands,
        learningWindowSamples = learningWindowSamples,
        seed = seed,
        outDir = outDir,
        modelArtifactDir = modelArtifactDir,
        heroMode = heroMode,
        gtoMode = gtoMode,
        villainMode = villainMode,
        heroExplorationRate = heroExplorationRate,
        raiseSize = raiseSize,
        bunchingTrials = bunchingTrials,
        equityTrials = equityTrials,
        saveTrainingTsv = saveTrainingTsv,
        saveDdreTrainingTsv = saveDdreTrainingTsv
      )

  private def parseOptions(args: Array[String]): Either[String, Map[String, String]] =
    val pairs = args.toVector.map { token =>
      if !token.startsWith("--") then Left(s"invalid argument '$token'; expected --key=value")
      else
        val body = token.drop(2)
        val idx = body.indexOf('=')
        if idx <= 0 || idx == body.length - 1 then Left(s"invalid argument '$token'; expected --key=value")
        else
          val key = body.substring(0, idx).trim
          val value = body.substring(idx + 1).trim
          if key.isEmpty || value.isEmpty then Left(s"invalid argument '$token'; key/value must be non-empty")
          else Right(key -> value)
    }
    pairs.foldLeft(Right(Map.empty): Either[String, Map[String, String]]) {
      case (Left(err), _) => Left(err)
      case (Right(_), Left(err)) => Left(err)
      case (Right(acc), Right((k, v))) => Right(acc.updated(k, v))
    }

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
        raw.trim.toLowerCase match
          case "nit"            => Right(VillainMode.Archetype(PlayerArchetype.Nit))
          case "tag"            => Right(VillainMode.Archetype(PlayerArchetype.Tag))
          case "lag"            => Right(VillainMode.Archetype(PlayerArchetype.Lag))
          case "callingstation" => Right(VillainMode.Archetype(PlayerArchetype.CallingStation))
          case "station"        => Right(VillainMode.Archetype(PlayerArchetype.CallingStation))
          case "maniac"         => Right(VillainMode.Archetype(PlayerArchetype.Maniac))
          case "gto"            => Right(VillainMode.Gto)
          case _ =>
            Left("--villainStyle must be one of: nit, tag, lag, callingstation, station, maniac, gto")

  private val usage =
    """Usage:
      |  runMain sicfun.holdem.TexasHoldemPlayingHall [--key=value ...]
      |
      |  --hands=<int>                 default 100000
      |  --tableCount=<int>            default 1
      |  --reportEvery=<int>           default 10000
      |  --learnEveryHands=<int>       default 50000 (0 disables learning)
      |  --learningWindowSamples=<int> default 200000 (0 = unbounded)
      |  --seed=<long>                 default 42
      |  --outDir=<path>               default data/playing-hall
      |  --modelArtifactDir=<path>     optional initial trained model
      |  --heroStyle=<style>           adaptive|gto
      |  --gtoMode=<mode>              fast|exact (default exact)
      |  --villainStyle=<style>        nit|tag|lag|callingstation|station|maniac|gto
      |  --heroExplorationRate=<double> default 0.05 (epsilon-greedy, [0,1])
      |  --raiseSize=<double>          default 2.5
      |  --bunchingTrials=<int>        default 80
      |  --equityTrials=<int>          default 700
      |  --saveTrainingTsv=<bool>      default true
      |  --saveDdreTrainingTsv=<bool>  default false
      |""".stripMargin
