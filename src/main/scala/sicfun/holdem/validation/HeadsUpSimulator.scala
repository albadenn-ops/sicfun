package sicfun.holdem.validation

import sicfun.core.{Deck, HandEvaluator}
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
final case class HeroRaiseResponseEvent(
    street: Street,
    response: PokerAction
)

final case class HandRecord(
    handId: String,
    handNumber: Int,
    heroCards: HoleCards,
    villainCards: HoleCards,
    board: Board,
    actions: Vector[RecordedAction],
    heroNet: Double,
    streetsPlayed: Int,
    heroSeat: Int = 1,
    villainSeat: Int = 2,
    heroIsButton: Boolean = true,
    heroRaiseResponses: Vector[HeroRaiseResponseEvent] = Vector.empty,
    villainName: String = "Villain",
    leakApplicableSpots: Int = 0
):
  def buttonSeat: Int = if heroIsButton then heroSeat else villainSeat

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
    heroEngine: Option[RealTimeAdaptiveEngine] = None,
    villain: LeakInjectedVillain,
    seed: Long,
    equityTrialsForCategory: Int = 500,
    startingStack: Double = 100.0,
    smallBlind: Double = 0.5,
    bigBlind: Double = 1.0,
    budgetMs: Long = 50L,
    villainStrategy: VillainStrategy = EquityBasedStrategy(),
    heroIsButton: Boolean = true
):
  private val rng = new Random(seed)
  private val heroName = "Hero"
  private val streets = Vector(Street.Preflop, Street.Flop, Street.Turn, Street.River)
  private val heroSeat = if heroIsButton then 1 else 2
  private val villainSeat = if heroIsButton then 2 else 1
  private val heroPosition = if heroIsButton then Position.Button else Position.BigBlind
  private val villainPosition = if heroIsButton then Position.BigBlind else Position.Button

  def playHand(handNumber: Int): HandRecord =
    // Shuffle deck
    val deck = rng.shuffle(Deck.full)
    val heroCards = HoleCards(deck(0), deck(1))
    val villainCards = HoleCards(deck(2), deck(3))
    val communityCards = deck.slice(4, 9)

    val actions = mutable.ArrayBuffer.empty[RecordedAction]
    val villainObs = mutable.ArrayBuffer.empty[VillainObservation]
    val villainLine = mutable.ArrayBuffer.empty[PokerAction]
    val heroRaiseResponses = mutable.ArrayBuffer.empty[HeroRaiseResponseEvent]
    var applicableSpots = 0
    var heroStack = startingStack
    var villainStack = startingStack
    var pot = 0.0
    var handOver = false
    var streetsPlayed = 0
    var lastFolderIsHero = false
    var pendingHeroRaiseStreet: Option[Street] = None

    if heroIsButton then
      heroStack -= smallBlind
      villainStack -= bigBlind
    else
      villainStack -= smallBlind
      heroStack -= bigBlind
    pot = smallBlind + bigBlind

    def boardForStreet(street: Street): Board = street match
      case Street.Preflop => Board.empty
      case Street.Flop    => Board.from(communityCards.take(3))
      case Street.Turn    => Board.from(communityCards.take(4))
      case Street.River   => Board.from(communityCards)

    for street <- streets if !handOver do
      streetsPlayed += 1
      val board = boardForStreet(street)
      // Track per-street commitments to compute correct toCall after raises
      var heroCommitted =
        if street == Street.Preflop then
          if heroIsButton then smallBlind else bigBlind
        else 0.0
      var villainCommitted =
        if street == Street.Preflop then
          if heroIsButton then bigBlind else smallBlind
        else 0.0
      var toCall = if street == Street.Preflop then bigBlind - smallBlind else 0.0
      var streetDone = false
      var actionsThisStreet = 0
      // Preflop: SB/Button acts first. Postflop: BB/OOP acts first.
      var heroTurn =
        if street == Street.Preflop then heroIsButton
        else !heroIsButton

      while !streetDone && !handOver && actionsThisStreet < 8 do
        if heroTurn then
          val gs = GameState(street, board, pot, math.max(0.0, toCall), heroPosition, heroStack, Vector.empty)
          val heroAction = validateAction(decideHero(heroCards, gs, board, street, villainObs.toVector), toCall)
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
              heroCommitted += callAmt
              toCall = 0.0
              actionsThisStreet += 1
              if actionsThisStreet >= 2 then streetDone = true
            case PokerAction.Raise(amount) =>
              // amount = net new chips from stack (what buildCandidates computes)
              val netCost = math.min(amount, heroStack)
              heroStack -= netCost
              pot += netCost
              heroCommitted += netCost
              toCall = math.max(0.0, heroCommitted - villainCommitted)
              pendingHeroRaiseStreet = Some(street)
              actionsThisStreet += 1
        else
          val gs = GameState(street, board, pot, math.max(0.0, toCall), villainPosition, villainStack, Vector.empty)

          // Compute equity once — used for both GTO decision and SpotContext
          val equityVsRandom = estimateEquityVsRandom(villainCards, board)
          val gtoAction = decideVillainGto(villainCards, gs, board, street, equityVsRandom)
          val facingAction = actions.lastOption.map(_.action)
          val spot = SpotContext.build(
            gs, villainCards,
            ActionLine(villainLine.toVector),
            equityVsRandom,
            facingAction
          )
          val result = villain.decide(gtoAction, spot)
          if result.leakApplicable then applicableSpots += 1

          // Clamp action to be valid for current game state — leak deviations
          // and noise can produce e.g. Call when toCall=0 or Check when toCall>0
          val validatedAction = validateAction(result.action, toCall)

          actions += RecordedAction(street, villain.name, validatedAction, pot, toCall, villainStack,
            result.leakFired, result.leakId)

          // Track observation for hero's adaptive inference
          villainObs += VillainObservation(validatedAction, gs)
          villainLine += validatedAction
          pendingHeroRaiseStreet.foreach { responseStreet =>
            validatedAction match
              case PokerAction.Fold | PokerAction.Call | PokerAction.Raise(_) =>
                heroRaiseResponses += HeroRaiseResponseEvent(responseStreet, validatedAction)
              case _ => ()
          }
          pendingHeroRaiseStreet = None

          validatedAction match
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
              villainCommitted += callAmt
              toCall = 0.0
              actionsThisStreet += 1
              if actionsThisStreet >= 2 then streetDone = true
            case PokerAction.Raise(amount) =>
              val netCost = math.min(amount, villainStack)
              villainStack -= netCost
              pot += netCost
              villainCommitted += netCost
              toCall = math.max(0.0, villainCommitted - heroCommitted)
              actionsThisStreet += 1

        heroTurn = !heroTurn

      if pendingHeroRaiseStreet.contains(street) then
        pendingHeroRaiseStreet = None

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
      streetsPlayed = streetsPlayed,
      heroSeat = heroSeat,
      villainSeat = villainSeat,
      heroIsButton = heroIsButton,
      heroRaiseResponses = heroRaiseResponses.toVector,
      villainName = villain.name,
      leakApplicableSpots = applicableSpots
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
    else heroEngine match
      case Some(engine) =>
        try
          val result = engine.decide(
            hero = hero,
            state = gs,
            folds = Vector.empty, // heads-up, no preflop folds
            villainPos = villainPosition,
            observations = villainObservations,
            candidateActions = candidates,
            decisionBudgetMillis = Some(budgetMs),
            rng = new Random(rng.nextLong())
          )
          result.decision.recommendation.bestAction
        catch
          case _: Exception => candidates.head
      case None =>
        // Fast equity-based hero — no engine inference
        val equity = estimateEquityVsRandom(hero, board)
        equityBasedDecision(equity, gs, street, candidates)

  /** Lightweight equity-based villain decision — avoids full engine inference.
    *
    * Uses precomputed equity vs random to pick a "competent" action without the
    * cost of Bayesian posterior inference, bunching MC, or per-candidate EV estimation.
    */
  private def decideVillainGto(
      villainHand: HoleCards,
      gs: GameState,
      board: Board,
      street: Street,
      equity: Double
  ): PokerAction =
    val candidates = buildCandidates(gs)
    if candidates.size <= 1 then return candidates.headOption.getOrElse(PokerAction.Check)
    villainStrategy.decide(villainHand, gs, candidates, equity, new Random(rng.nextLong()))

  /** Equity-based action decision shared by fast hero and villain GTO paths. */
  private def equityBasedDecision(
      equity: Double,
      gs: GameState,
      street: Street,
      candidates: Vector[PokerAction]
  ): PokerAction =
    val potOdds = gs.potOdds
    val localRng = new Random(rng.nextLong())

    if gs.toCall > 0 then
      val raiseActions = candidates.collect { case r: PokerAction.Raise => r }
      // HU Button/SB preflop: open-raise strategy — raise most hands, rarely limp
      if street == Street.Preflop && gs.position == Position.Button then
        if equity >= 0.35 then
          if raiseActions.nonEmpty then raiseActions.head else PokerAction.Call
        else if equity >= 0.20 && localRng.nextDouble() < 0.4 then
          if raiseActions.nonEmpty then raiseActions.head else PokerAction.Call
        else PokerAction.Fold
      // Facing a bet: fold, call, or raise based on equity vs pot odds
      else if equity >= 0.75 then
        // Strong hand — raise or call
        if raiseActions.nonEmpty && localRng.nextDouble() < 0.6 then raiseActions.head
        else PokerAction.Call
      else if equity >= potOdds + 0.05 then
        // Enough equity to call; sometimes raise preflop (3-bet)
        if street == Street.Preflop && raiseActions.nonEmpty && localRng.nextDouble() < 0.25 then
          raiseActions.head
        else if equity >= 0.55 && raiseActions.nonEmpty && localRng.nextDouble() < 0.20 then
          raiseActions.head
        else PokerAction.Call
      else if street == Street.Preflop && localRng.nextDouble() < 0.70 then
        // HU preflop: defend wide (pot odds favorable, only ~30% fold)
        if raiseActions.nonEmpty && localRng.nextDouble() < 0.20 then raiseActions.head
        else PokerAction.Call
      else if localRng.nextDouble() < 0.25 then
        // Postflop: occasional float with weak hands
        PokerAction.Call
      else PokerAction.Fold
    else
      // No bet to call: check or bet
      if equity >= 0.60 then
        // Value bet
        val raiseActions = candidates.collect { case r: PokerAction.Raise => r }
        if raiseActions.nonEmpty then raiseActions(localRng.nextInt(raiseActions.size))
        else PokerAction.Check
      else if equity <= 0.30 && localRng.nextDouble() < 0.25 then
        // Bluff occasionally with air
        val raiseActions = candidates.collect { case r: PokerAction.Raise => r }
        if raiseActions.nonEmpty then raiseActions.head else PokerAction.Check
      else PokerAction.Check

  private def buildCandidates(gs: GameState): Vector[PokerAction] =
    val candidates = Vector.newBuilder[PokerAction]
    if gs.toCall > 0 then
      candidates += PokerAction.Fold
      candidates += PokerAction.Call
      val raiseSize = math.min(gs.pot + gs.toCall * 2, gs.stackSize)
      if gs.stackSize > gs.toCall * 2 && raiseSize > gs.toCall then
        candidates += PokerAction.Raise(raiseSize)
      // All-in at low SPR
      val spr = if gs.pot > 0 then gs.stackSize / gs.pot else Double.MaxValue
      if spr < 3.0 && gs.stackSize > raiseSize * 1.2 then
        candidates += PokerAction.Raise(gs.stackSize)
    else
      candidates += PokerAction.Check
      val smallBet = gs.pot * 0.66
      val bigBet = gs.pot * 1.0
      if gs.stackSize > smallBet && smallBet > 0 then
        candidates += PokerAction.Raise(smallBet)
      if gs.stackSize > bigBet && bigBet > smallBet * 1.2 then
        candidates += PokerAction.Raise(bigBet)
      // All-in at low SPR
      val spr = if gs.pot > 0 then gs.stackSize / gs.pot else Double.MaxValue
      if spr < 3.0 && gs.stackSize > bigBet * 1.2 then
        candidates += PokerAction.Raise(gs.stackSize)
    candidates.result()

  /** Clamp action to be valid for the current pot/call state.
    * Leak deviations and noise can produce e.g. Call when toCall=0.
    */
  private def validateAction(action: PokerAction, toCall: Double): PokerAction =
    action match
      case PokerAction.Call if toCall <= 0 => PokerAction.Check
      case PokerAction.Check if toCall > 0 => PokerAction.Call
      case PokerAction.Fold if toCall <= 0 => PokerAction.Check
      case PokerAction.Raise(amt) if amt <= 0.0 =>
        if toCall > 0 then PokerAction.Call else PokerAction.Check
      case other => other

  /** Equity estimate vs a random hand on this board via Monte Carlo.
    *
    * For preflop/incomplete boards, deals the remaining community cards
    * alongside random opponent cards. For postflop (3+ board cards),
    * deals only the remaining runout cards.
    */
  private def estimateEquityVsRandom(hand: HoleCards, board: Board): Double =
    val available = Deck.full.filterNot(c => hand.toVector.contains(c) || board.cards.contains(c))
    if available.size < 2 then return 0.5
    val boardSize = board.cards.size
    val communityNeeded = 5 - boardSize
    val needed = 2 + communityNeeded
    val trials = if boardSize < 3 then 50 else math.min(equityTrialsForCategory, 100)
    var wins = 0
    var total = 0
    val localRng = new Random(rng.nextLong())
    val arr = available.toArray
    val n = arr.length
    val heroBase = hand.toVector
    val boardCards = board.cards
    for _ <- 0 until trials do
      // Partial Fisher-Yates: only shuffle first `needed` positions (no allocation)
      var i = 0
      while i < needed do
        val j = i + localRng.nextInt(n - i)
        val tmp = arr(i); arr(i) = arr(j); arr(j) = tmp
        i += 1
      val fullBoard = if communityNeeded > 0 then
        boardCards ++ (2 until needed).map(arr(_))
      else boardCards
      val heroAll = (heroBase ++ fullBoard).take(7)
      val oppAll = (Vector(arr(0), arr(1)) ++ fullBoard).take(7)
      if heroAll.size >= 7 && oppAll.size >= 7 then
        val heroRank = HandEvaluator.evaluate7(heroAll)
        val oppRank = HandEvaluator.evaluate7(oppAll)
        if heroRank > oppRank then wins += 1
        total += 1
    if total > 0 then wins.toDouble / total.toDouble else 0.5
