package sicfun.holdem.validation

import sicfun.core.{Card, Deck, HandEvaluator}
import sicfun.holdem.engine.{RealTimeAdaptiveEngine, VillainObservation}
import sicfun.holdem.types.*

import scala.collection.mutable
import scala.util.Random

/** A single recorded action within a simulated hand, tagged with leak metadata. */
final case class RecordedAction(
    street: Street,
    player: String,
    action: PokerAction,
    potBefore: Double,
    toCall: Double,
    stackBefore: Double,
    leakFired: Boolean,
    leakId: Option[String]
)

/** Complete record of one simulated hand. */
final case class HandRecord(
    handId: String,
    handNumber: Int,
    heroCards: HoleCards,
    villainCards: HoleCards,
    board: Board,
    actions: Vector[RecordedAction],
    heroNet: Double,
    streetsPlayed: Int
)

/** Focused heads-up hand simulator for validation.
  *
  * Hero uses RealTimeAdaptiveEngine in adaptive mode. Villain uses a separate
  * engine instance for "competent baseline" actions, then passes through
  * LeakInjectedVillain for potential deviation.
  *
  * This is intentionally simpler than TexasHoldemPlayingHall — heads-up only,
  * no side pots, no training data collection.
  */
final class HeadsUpSimulator(
    heroEngine: RealTimeAdaptiveEngine,
    villainEngine: RealTimeAdaptiveEngine,
    villain: LeakInjectedVillain,
    seed: Long,
    equityTrialsForCategory: Int = 500,
    startingStack: Double = 100.0,
    smallBlind: Double = 0.5,
    bigBlind: Double = 1.0,
    budgetMs: Long = 50L
):
  private val rng = new Random(seed)
  private val heroName = "Hero"
  private val streets = Vector(Street.Preflop, Street.Flop, Street.Turn, Street.River)

  def playHand(handNumber: Int): HandRecord =
    // Shuffle deck
    val deck = rng.shuffle(Deck.full)
    val heroCards = HoleCards(deck(0), deck(1))
    val villainCards = HoleCards(deck(2), deck(3))
    val communityCards = deck.slice(4, 9)

    val actions = mutable.ArrayBuffer.empty[RecordedAction]
    val villainObs = mutable.ArrayBuffer.empty[VillainObservation]
    val villainLine = mutable.ArrayBuffer.empty[PokerAction]
    var heroStack = startingStack
    var villainStack = startingStack
    var pot = 0.0
    var handOver = false
    var streetsPlayed = 0
    var lastFolderIsHero = false

    // Post blinds: hero = SB/Button, villain = BB
    heroStack -= smallBlind
    villainStack -= bigBlind
    pot = smallBlind + bigBlind

    def boardForStreet(street: Street): Board = street match
      case Street.Preflop => Board.empty
      case Street.Flop    => Board.from(communityCards.take(3))
      case Street.Turn    => Board.from(communityCards.take(4))
      case Street.River   => Board.from(communityCards)

    for street <- streets if !handOver do
      streetsPlayed += 1
      val board = boardForStreet(street)
      var toCall = if street == Street.Preflop then bigBlind - smallBlind else 0.0
      var streetDone = false
      var actionsThisStreet = 0
      // Preflop: SB(hero/Button) acts first. Postflop: villain(BB/OOP) acts first.
      var heroTurn = street == Street.Preflop

      while !streetDone && !handOver do
        if heroTurn then
          val gs = GameState(street, board, pot, toCall, Position.Button, heroStack, Vector.empty)
          val heroAction = decideHero(heroCards, gs, board, street, villainObs.toVector)
          actions += RecordedAction(street, heroName, heroAction, pot, toCall, heroStack,
            leakFired = false, leakId = None)
          heroAction match
            case PokerAction.Fold =>
              handOver = true
              lastFolderIsHero = true
            case PokerAction.Check =>
              actionsThisStreet += 1
              if actionsThisStreet >= 2 then streetDone = true
            case PokerAction.Call =>
              val callAmt = math.min(toCall, heroStack)
              heroStack -= callAmt
              pot += callAmt
              toCall = 0.0
              actionsThisStreet += 1
              if actionsThisStreet >= 2 then streetDone = true
            case PokerAction.Raise(amount) =>
              val totalCommit = math.min(amount, heroStack)
              heroStack -= totalCommit
              pot += totalCommit
              toCall = totalCommit // villain needs to match this
              actionsThisStreet += 1
        else
          val gs = GameState(street, board, pot, toCall, Position.BigBlind, villainStack, Vector.empty)
          val gtoAction = decideVillainGto(villainCards, gs, board, street)

          // Compute equity vs random for HandCategory classification
          val equityVsRandom = estimateEquityVsRandom(villainCards, board)
          val facingAction = actions.lastOption.map(_.action)
          val spot = SpotContext.build(
            gs, villainCards,
            ActionLine(villainLine.toVector),
            equityVsRandom,
            facingAction
          )
          val result = villain.decide(gtoAction, spot)

          actions += RecordedAction(street, villain.name, result.action, pot, toCall, villainStack,
            result.leakFired, result.leakId)

          // Track observation for hero's adaptive inference
          villainObs += VillainObservation(result.action, gs)
          villainLine += result.action

          result.action match
            case PokerAction.Fold =>
              handOver = true
              lastFolderIsHero = false
            case PokerAction.Check =>
              actionsThisStreet += 1
              if actionsThisStreet >= 2 then streetDone = true
            case PokerAction.Call =>
              val callAmt = math.min(toCall, villainStack)
              villainStack -= callAmt
              pot += callAmt
              toCall = 0.0
              actionsThisStreet += 1
              if actionsThisStreet >= 2 then streetDone = true
            case PokerAction.Raise(amount) =>
              val totalCommit = math.min(amount, villainStack)
              villainStack -= totalCommit
              pot += totalCommit
              toCall = totalCommit
              actionsThisStreet += 1

        heroTurn = !heroTurn

    // Determine outcome
    val finalBoard = boardForStreet(streets(math.min(streetsPlayed - 1, streets.length - 1)))
    val heroContribution = startingStack - heroStack
    val villainContribution = startingStack - villainStack

    val heroNet =
      if handOver then
        // Someone folded
        if lastFolderIsHero then -heroContribution
        else villainContribution // hero wins what villain put in
      else
        // Showdown
        val allHero = heroCards.toVector ++ finalBoard.cards
        val allVillain = villainCards.toVector ++ finalBoard.cards
        if allHero.size >= 7 && allVillain.size >= 7 then
          val heroRank = HandEvaluator.evaluate7(allHero.take(7))
          val villainRank = HandEvaluator.evaluate7(allVillain.take(7))
          if heroRank > villainRank then villainContribution
          else if heroRank < villainRank then -heroContribution
          else 0.0 // split pot
        else
          0.0 // shouldn't happen in a well-formed hand

    HandRecord(
      handId = f"SIM-${handNumber}%08d",
      handNumber = handNumber,
      heroCards = heroCards,
      villainCards = villainCards,
      board = finalBoard,
      actions = actions.toVector,
      heroNet = heroNet,
      streetsPlayed = streetsPlayed
    )

  private def decideHero(
      hero: HoleCards,
      gs: GameState,
      board: Board,
      street: Street,
      villainObservations: Vector[VillainObservation]
  ): PokerAction =
    val candidates = buildCandidates(gs)
    if candidates.size <= 1 then candidates.headOption.getOrElse(PokerAction.Check)
    else
      try
        val result = heroEngine.decide(
          hero = hero,
          state = gs,
          folds = Vector.empty, // heads-up, no preflop folds
          villainPos = Position.BigBlind,
          observations = villainObservations,
          candidateActions = candidates,
          decisionBudgetMillis = Some(budgetMs),
          rng = new Random(rng.nextLong())
        )
        result.decision.recommendation.bestAction
      catch
        case _: Exception => candidates.head

  private def decideVillainGto(
      villainHand: HoleCards,
      gs: GameState,
      board: Board,
      street: Street
  ): PokerAction =
    val candidates = buildCandidates(gs)
    if candidates.size <= 1 then candidates.headOption.getOrElse(PokerAction.Check)
    else
      try
        // Villain uses engine with uniform prior (no learned bias) as "competent baseline"
        val result = villainEngine.decide(
          hero = villainHand,
          state = gs,
          folds = Vector.empty,
          villainPos = Position.Button, // from villain's perspective, hero is in Button
          observations = Seq.empty, // no observations — fresh each decision
          candidateActions = candidates,
          decisionBudgetMillis = Some(budgetMs),
          rng = new Random(rng.nextLong())
        )
        result.decision.recommendation.bestAction
      catch
        case _: Exception => candidates.head

  private def buildCandidates(gs: GameState): Vector[PokerAction] =
    val candidates = Vector.newBuilder[PokerAction]
    if gs.toCall > 0 then
      candidates += PokerAction.Fold
      candidates += PokerAction.Call
      val raiseSize = math.min(gs.pot + gs.toCall * 2, gs.stackSize)
      if gs.stackSize > gs.toCall * 2 && raiseSize > gs.toCall then
        candidates += PokerAction.Raise(raiseSize)
    else
      candidates += PokerAction.Check
      val betSize = gs.pot * 0.66
      if gs.stackSize > betSize && betSize > 0 then
        candidates += PokerAction.Raise(betSize)
    candidates.result()

  /** Rough equity estimate vs a random hand on this board.
    * Uses hand evaluator rank as a proxy — not precise but fast.
    */
  private def estimateEquityVsRandom(hand: HoleCards, board: Board): Double =
    if board.cards.isEmpty then 0.5 // preflop: no evaluation, assume average
    else if board.cards.size < 3 then 0.5
    else
      // Monte Carlo mini-simulation: deal random opponent hands and compare
      val trials = math.min(equityTrialsForCategory, 200)
      val available = Deck.full.filterNot(c => hand.toVector.contains(c) || board.cards.contains(c))
      if available.size < 2 then return 0.5
      var wins = 0
      var total = 0
      val localRng = new Random(rng.nextLong())
      val heroAll = hand.toVector ++ board.cards
      val padded = if heroAll.size < 7 then
        // Pad to 7 with remaining community cards (shouldn't happen for flop+)
        heroAll ++ available.take(7 - heroAll.size)
      else heroAll.take(7)
      val heroRank = HandEvaluator.evaluate7(padded)

      for _ <- 0 until trials do
        val shuffled = localRng.shuffle(available)
        val oppCards = shuffled.take(2)
        val oppAll = oppCards ++ board.cards
        val oppPadded = if oppAll.size < 7 then oppAll ++ shuffled.drop(2).take(7 - oppAll.size) else oppAll.take(7)
        if oppPadded.size >= 7 then
          val oppRank = HandEvaluator.evaluate7(oppPadded)
          if heroRank > oppRank then wins += 1
          total += 1

      if total > 0 then wins.toDouble / total.toDouble else 0.5
