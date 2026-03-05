package sicfun.holdem

import sicfun.core.{Deck, HandEvaluator}

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths, StandardOpenOption}
import java.util.Locale
import scala.collection.mutable
import scala.jdk.CollectionConverters.*
import scala.util.Random

/** Large-volume self-play hall:
  *  - plays full hands (preflop -> river)
  *  - runs Bayesian range inference + action recommendation for hero decisions
  *  - logs hands + learning checkpoints
  *  - periodically retrains villain action model from generated data
  */
object TexasHoldemPlayingHall:
  private final case class Config(
      hands: Int,
      reportEvery: Int,
      learnEveryHands: Int,
      learningWindowSamples: Int,
      seed: Long,
      outDir: Path,
      modelArtifactDir: Option[Path],
      villainStyle: PlayerArchetype,
      heroExplorationRate: Double,
      raiseSize: Double,
      bunchingTrials: Int,
      equityTrials: Int,
      saveTrainingTsv: Boolean
  )

  final case class HallSummary(
      handsPlayed: Int,
      heroNetChips: Double,
      heroBbPer100: Double,
      heroWins: Int,
      heroTies: Int,
      heroLosses: Int,
      actionCounts: Map[String, Int],
      retrains: Int,
      modelId: String,
      outDir: Path
  )

  private final case class Deal(hero: HoleCards, villain: HoleCards, board: Board)

  private final case class HandResult(
      heroNet: Double,
      outcome: Int,
      villainDecision: Option[(GameState, PokerAction)],
      villainTrainingSamples: Vector[(GameState, HoleCards, PokerAction)],
      raiseResponses: Vector[PokerAction],
      heroActions: Vector[PokerAction],
      villainActions: Vector[PokerAction],
      streetsPlayed: Int
  )

  private final case class VillainStyleProfile(looseness: Double, aggression: Double)

  def main(args: Array[String]): Unit =
    val wantsHelp = args.contains("--help") || args.contains("-h")
    run(args) match
      case Right(summary) =>
        println("=== Texas Hold'em Playing Hall ===")
        println(s"handsPlayed: ${summary.handsPlayed}")
        println(f"heroNetChips: ${summary.heroNetChips}%.4f")
        println(f"heroBbPer100: ${summary.heroBbPer100}%.3f")
        println(s"heroWins: ${summary.heroWins}")
        println(s"heroTies: ${summary.heroTies}")
        println(s"heroLosses: ${summary.heroLosses}")
        println(s"actionCounts: ${summary.actionCounts}")
        println(s"retrains: ${summary.retrains}")
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
      val handsLog = config.outDir.resolve("hands.tsv")
      val learningLog = config.outDir.resolve("learning.tsv")
      val trainingTsv = config.outDir.resolve("training-selfplay.tsv")

      initHandsLog(handsLog)
      initLearningLog(learningLog)
      if config.saveTrainingTsv then initTrainingTsv(trainingTsv)

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

      var heroNet = 0.0
      var heroWins = 0
      var heroTies = 0
      var heroLosses = 0
      val actionCounts = mutable.Map.empty[String, Int].withDefaultValue(0)
      var retrains = 0

      var handNo = 1
      while handNo <= config.hands do
        val deal = dealHand(rng)
        val result = resolveHand(
          deal = deal,
          preflopEngine = preflopEngine,
          postflopEngine = postflopEngine,
          config = config,
          folds = folds,
          rng = new Random(rng.nextLong())
        )

        result.villainTrainingSamples.foreach { sample =>
          learningQueue.enqueue(sample)
          if config.learningWindowSamples > 0 then
            while learningQueue.size > config.learningWindowSamples do learningQueue.dequeue()
          if config.saveTrainingTsv then appendTrainingSample(trainingTsv, sample)
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
          path = handsLog,
          hand = handNo,
          deal = deal,
          result = result,
          modelId = activeArtifact.version.id,
          archetype = config.villainStyle
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
            path = learningLog,
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

      val bbPer100 =
        if config.hands > 0 then (heroNet / config.hands.toDouble) * 100.0
        else 0.0
      Right(
        HallSummary(
          handsPlayed = config.hands,
          heroNetChips = heroNet,
          heroBbPer100 = bbPer100,
          heroWins = heroWins,
          heroTies = heroTies,
          heroLosses = heroLosses,
          actionCounts = actionCounts.toMap,
          retrains = retrains,
          modelId = activeArtifact.version.id,
          outDir = config.outDir
        )
      )
    catch
      case e: Exception =>
        Left(s"playing hall failed: ${e.getMessage}")

  private def resolveHand(
      deal: Deal,
      preflopEngine: RealTimeAdaptiveEngine,
      postflopEngine: RealTimeAdaptiveEngine,
      config: Config,
      folds: Vector[PreflopFold],
      rng: Random
  ): HandResult =
    var heroStack = 99.5
    var villainStack = 99.0
    var heroContribution = 0.5
    var villainContribution = 1.0
    var pot = heroContribution + villainContribution
    var betHistory = Vector.empty[BetAction]

    val villainTrainingSamples = mutable.ArrayBuffer.empty[(GameState, HoleCards, PokerAction)]
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
        case Street.Flop    => Board.from(deal.board.cards.take(3))
        case Street.Turn    => Board.from(deal.board.cards.take(4))
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
      val engine = if street == Street.Preflop then preflopEngine else postflopEngine
      val decisionBudgetMillis = if street == Street.Preflop then None else Some(1L)
      val greedy = engine.decide(
        hero = deal.hero,
        state = state,
        folds = folds,
        villainPos = Position.BigBlind,
        observations = villainObservations.toVector,
        candidateActions = candidates,
        decisionBudgetMillis = decisionBudgetMillis,
        rng = new Random(rng.nextLong())
      ).decision.recommendation.bestAction
      val sampled =
        if rng.nextDouble() < config.heroExplorationRate then candidates(rng.nextInt(candidates.length))
        else greedy
      val normalized = normalizeAction(
        action = sampled,
        toCall = toCall,
        allowRaise = allowRaise,
        defaultRaise = config.raiseSize
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
      val sampled = villainResponds(
        hand = deal.villain,
        style = config.villainStyle,
        state = state,
        allowRaise = allowRaise,
        raiseSize = config.raiseSize,
        rng = rng
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
    math.max(1, configured / 10)

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
    val board = Board.from(cards.slice(4, 9).toVector)
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

  private def initHandsLog(path: Path): Unit =
    val header = Vector(
      "hand",
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
    Files.write(path, Vector(header).asJava, StandardCharsets.UTF_8)

  private def appendHandLog(
      path: Path,
      hand: Int,
      deal: Deal,
      result: HandResult,
      modelId: String,
      archetype: PlayerArchetype
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
      deal.hero.toToken,
      deal.villain.toToken,
      boardToken,
      heroAction,
      villainAction,
      fmt(result.heroNet, 4),
      modelId,
      archetype.toString,
      result.streetsPlayed.toString,
      heroActionTrace,
      villainActionTrace,
      outcomeLabel
    ).mkString("\t")
    Files.write(path, Vector(row).asJava, StandardCharsets.UTF_8, StandardOpenOption.APPEND)

  private def initLearningLog(path: Path): Unit =
    Files.write(
      path,
      Vector("hand\tmodelId\tmeanBrierScore\tgatePassed\tsampleCount").asJava,
      StandardCharsets.UTF_8
    )

  private def appendLearningLog(
      path: Path,
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
    Files.write(path, Vector(row).asJava, StandardCharsets.UTF_8, StandardOpenOption.APPEND)

  private def initTrainingTsv(path: Path): Unit =
    Files.write(
      path,
      Vector("street\tboard\tpotBefore\ttoCall\tposition\tstackBefore\taction\tholecards").asJava,
      StandardCharsets.UTF_8
    )

  private def appendTrainingSample(
      path: Path,
      sample: (GameState, HoleCards, PokerAction)
  ): Unit =
    val (state, hole, action) = sample
    val boardToken =
      if state.board.cards.isEmpty then "-"
      else state.board.cards.map(_.toToken).mkString(" ")
    val row = Vector(
      state.street.toString,
      boardToken,
      state.pot.toString,
      state.toCall.toString,
      state.position.toString,
      state.stackSize.toString,
      actionToken(action),
      s"${hole.first.toToken} ${hole.second.toToken}"
    ).mkString("\t")
    Files.write(path, Vector(row).asJava, StandardCharsets.UTF_8, StandardOpenOption.APPEND)

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
        reportEvery <- intOpt(options, "reportEvery", 10000)
        _ <- if reportEvery > 0 then Right(()) else Left("--reportEvery must be > 0")
        learnEveryHands <- intOpt(options, "learnEveryHands", 50000)
        _ <- if learnEveryHands >= 0 then Right(()) else Left("--learnEveryHands must be >= 0")
        learningWindowSamples <- intOpt(options, "learningWindowSamples", 200000)
        _ <- if learningWindowSamples >= 0 then Right(()) else Left("--learningWindowSamples must be >= 0")
        seed <- longOpt(options, "seed", 42L)
        outDir <- pathOpt(options, "outDir", Paths.get("data/playing-hall"))
        modelArtifactDir <- optionalPathOpt(options, "modelArtifactDir")
        villainStyle <- styleOpt(options, "villainStyle", PlayerArchetype.Tag)
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
      yield Config(
        hands = hands,
        reportEvery = reportEvery,
        learnEveryHands = learnEveryHands,
        learningWindowSamples = learningWindowSamples,
        seed = seed,
        outDir = outDir,
        modelArtifactDir = modelArtifactDir,
        villainStyle = villainStyle,
        heroExplorationRate = heroExplorationRate,
        raiseSize = raiseSize,
        bunchingTrials = bunchingTrials,
        equityTrials = equityTrials,
        saveTrainingTsv = saveTrainingTsv
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

  private def styleOpt(
      options: Map[String, String],
      key: String,
      default: PlayerArchetype
  ): Either[String, PlayerArchetype] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) =>
        raw.trim.toLowerCase match
          case "nit"            => Right(PlayerArchetype.Nit)
          case "tag"            => Right(PlayerArchetype.Tag)
          case "lag"            => Right(PlayerArchetype.Lag)
          case "callingstation" => Right(PlayerArchetype.CallingStation)
          case "station"        => Right(PlayerArchetype.CallingStation)
          case "maniac"         => Right(PlayerArchetype.Maniac)
          case _ =>
            Left("--villainStyle must be one of: nit, tag, lag, callingstation, station, maniac")

  private val usage =
    """Usage:
      |  runMain sicfun.holdem.TexasHoldemPlayingHall [--key=value ...]
      |
      |  --hands=<int>                 default 100000
      |  --reportEvery=<int>           default 10000
      |  --learnEveryHands=<int>       default 50000 (0 disables learning)
      |  --learningWindowSamples=<int> default 200000 (0 = unbounded)
      |  --seed=<long>                 default 42
      |  --outDir=<path>               default data/playing-hall
      |  --modelArtifactDir=<path>     optional initial trained model
      |  --villainStyle=<style>        nit|tag|lag|callingstation|station|maniac
      |  --heroExplorationRate=<double> default 0.05 (epsilon-greedy, [0,1])
      |  --raiseSize=<double>          default 2.5
      |  --bunchingTrials=<int>        default 80
      |  --equityTrials=<int>          default 700
      |  --saveTrainingTsv=<bool>      default true
      |""".stripMargin
