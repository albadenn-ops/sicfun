package sicfun.holdem.runtime
import sicfun.holdem.types.*
import sicfun.holdem.engine.*
import sicfun.holdem.equity.*
import sicfun.holdem.cli.*
import sicfun.holdem.history.*

import sicfun.core.Card

import scala.util.Random

/** Configuration for a heads-up advisor session.
  *
  * @param startingStack        Chips each player starts with (default 200 = 100 BB).
  * @param smallBlind           Small blind size.
  * @param bigBlind             Big blind size.
  * @param heroStartsAsSB       If true, hero starts on the button (SB) in hand #1.
  * @param decisionBudgetMillis Max milliseconds the engine can spend per recommendation.
  */
final case class SessionConfig(
    startingStack: Double = 200.0,
    smallBlind: Double = 1.0,
    bigBlind: Double = 2.0,
    heroStartsAsSB: Boolean = true,
    decisionBudgetMillis: Long = 2000L
)

/** Cumulative session statistics. */
final case class AdvisorSessionStats(
    handsPlayed: Int = 0,
    heroNetChips: Double = 0.0,
    heroWins: Int = 0,
    heroLosses: Int = 0,
    heroSplits: Int = 0
)

/** Record of a hero decision point for post-hand review. */
final case class HeroDecisionRecord(
    street: Street,
    gameState: GameState,
    heroCards: HoleCards,
    actualAction: PokerAction,
    villainObservations: Vector[VillainObservation]
)

/** Events logged for undo support.
  *
  * Each user action appends a [[HandEvent]] to the hand's event log.  The event captures
  * a full snapshot of all mutable fields before the action was applied, so undo can
  * restore the exact previous state without recomputation.
  *
  * Three event types cover all state transitions:
  *  - [[HeroCardsSet]] — hero hole cards were assigned or changed.
  *  - [[ActionRecorded]] — a hero or villain action was applied (fold/check/call/raise).
  *  - [[BoardDealt]] — community cards were dealt (flop/turn/river).
  */
enum HandEvent:
  case HeroCardsSet(previous: Option[HoleCards])
  case ActionRecorded(
      isHero: Boolean,
      action: PokerAction,
      potBefore: Double,
      heroStackBefore: Double,
      villainStackBefore: Double,
      toCallBefore: Double,
      heroCommittedBefore: Double,
      villainCommittedBefore: Double,
      betHistoryBefore: Vector[BetAction],
      lastHeroRaiseBefore: Boolean,
      villainObsBefore: Vector[VillainObservation],
      heroDecisionsBefore: Vector[HeroDecisionRecord],
      finishedBefore: Boolean,
      heroNetBefore: Double,
      rememberedOpponentBefore: Option[OpponentProfile],
      opponentMemoryStoreBefore: Option[OpponentProfileStore],
      opponentMemoryDirtyBefore: Boolean
  )
  case BoardDealt(
      boardBefore: Board,
      streetBefore: Street,
      toCallBefore: Double,
      heroCommittedBefore: Double,
      villainCommittedBefore: Double
  )

/** In-progress hand state, tracking all the mutable aspects of a single hand.
  *
  * This is an immutable snapshot that gets replaced on every state change.
  * Key fields:
  *  - `heroCommittedThisStreet` / `villainCommittedThisStreet` — chips committed on the current street only (reset to 0 when a new street is dealt).
  *  - `villainObservations` — villain actions observed this hand, used for range inference.
  *  - `heroDecisions` — hero decision records for post-hand review.
  *  - `eventLog` — full undo history; each event captures a pre-action snapshot.
  *  - `lastHeroActionWasRaise` — tracks whether the villain's next action is a response to a hero raise (for archetype learning).
  */
final case class HandSnapshot(
    handNumber: Int,
    heroCards: Option[HoleCards] = None,
    board: Board = Board.empty,
    street: Street = Street.Preflop,
    pot: Double = 0.0,
    heroStack: Double = 0.0,
    villainStack: Double = 0.0,
    toCall: Double = 0.0,
    heroCommittedThisStreet: Double = 0.0,
    villainCommittedThisStreet: Double = 0.0,
    heroPosition: Position = Position.Button,
    villainPosition: Position = Position.BigBlind,
    betHistory: Vector[BetAction] = Vector.empty,
    villainObservations: Vector[VillainObservation] = Vector.empty,
    heroDecisions: Vector[HeroDecisionRecord] = Vector.empty,
    eventLog: Vector[HandEvent] = Vector.empty,
    lastHeroActionWasRaise: Boolean = false,
    finished: Boolean = false,
    heroNetResult: Double = 0.0,
    villainRevealedCards: Option[HoleCards] = None
)

/** Result of executing a single command. */
final case class CommandResult(
    session: AdvisorSession,
    output: Vector[String]
)

/** Pure-ish state machine for an interactive poker advisor session.
  *
  * Manages a sequence of heads-up poker hands, processing user commands (new hand,
  * set hero cards, record actions, deal board, request advice) and maintaining
  * cumulative session statistics and opponent memory.
  *
  * '''State management:'''
  * The `RealTimeAdaptiveEngine` is intentionally mutable (archetype posterior
  * updates are side-effectful) — this matches how it is used everywhere in
  * the codebase. The rest of the state is replaced functionally via the `updated()`
  * factory method, which creates a new session with the specified changes.
  *
  * '''Opponent memory:'''
  * If opponent memory is configured (via `opponentMemoryTarget`/`opponentMemorySite`/
  * `opponentMemoryName`), each villain action is persisted to the profile store.
  * The store is flushed to disk at hand boundaries (not after every action) to
  * reduce I/O.  Undo also restores the memory state to the pre-action snapshot.
  *
  * '''Undo support:'''
  * Every state-changing command appends a [[HandEvent]] to the hand's event log.
  * Each event captures a complete pre-action snapshot, so undo simply restores
  * those saved values without needing to replay the entire hand history.
  *
  * @param config                     Session-level configuration (stack, blinds, budget).
  * @param engine                     Shared mutable [[RealTimeAdaptiveEngine]] (archetype learning is side-effectful).
  * @param tableRanges                Preflop opening ranges for the table format.
  * @param hand                       Current hand in progress (None between hands).
  * @param stats                      Cumulative session statistics.
  * @param rng                        Seeded RNG for reproducible engine decisions.
  * @param rememberedOpponent         Loaded opponent profile (if any).
  * @param rememberedVillainObservations  Villain observations carried over from previous hands.
  * @param opponentMemoryTarget       Persistence target (file path or JDBC URL).
  * @param opponentMemorySite         Site key for opponent lookup (e.g., "pokerstars").
  * @param opponentMemoryName         Opponent screen name.
  * @param opponentMemoryStore        In-memory profile store (flushed to disk at boundaries).
  * @param opponentMemoryDirty        True if the store has unflushed changes.
  */
final class AdvisorSession(
    val config: SessionConfig,
    val engine: RealTimeAdaptiveEngine,
    val tableRanges: TableRanges,
    val hand: Option[HandSnapshot],
    val stats: AdvisorSessionStats,
    val rng: Random,
    val rememberedOpponent: Option[OpponentProfile] = None,
    val rememberedVillainObservations: Vector[VillainObservation] = Vector.empty,
    val opponentMemoryTarget: Option[OpponentMemoryTarget] = None,
    val opponentMemorySite: Option[String] = None,
    val opponentMemoryName: Option[String] = None,
    val opponentMemoryStore: Option[OpponentProfileStore] = None,
    val opponentMemoryDirty: Boolean = false
):
  private val HeroIdx = 0
  private val VillainIdx = 1

  /** Dispatch a user command and return the updated session + output lines.
    * Each command handler returns a [[CommandResult]] containing a new session
    * (with updated state) and a vector of lines to display to the user.
    */
  def execute(command: AdvisorCommand): CommandResult =
    command match
      case AdvisorCommand.NewHand          => doNewHand()
      case AdvisorCommand.HeroCards(cards) => doHeroCards(cards)
      case AdvisorCommand.HeroAction(a)    => doAction(isHero = true, a)
      case AdvisorCommand.VillainAction(a) => doAction(isHero = false, a)
      case AdvisorCommand.VillainShowdown(cards) => doVillainShowdown(cards)
      case AdvisorCommand.DealBoard(cards) => doDealBoard(cards)
      case AdvisorCommand.Advise           => doAdvise()
      case AdvisorCommand.Review           => doReview()
      case AdvisorCommand.SessionStats     => doSessionStats()
      case AdvisorCommand.Undo             => doUndo()
      case AdvisorCommand.Help             => doHelp()
      case AdvisorCommand.Quit             => doQuit()
      case AdvisorCommand.Unknown(input, reason) =>
        CommandResult(this, Vector(s"Unknown command: $reason. Type 'help' for usage."))

  // ---- NewHand ----

  /** Start a new hand: flush opponent memory if dirty, archive villain observations from
    * the previous hand, alternate hero's position (BTN/BB), post blinds, and initialize
    * a fresh [[HandSnapshot]].  Also clears the engine's inference cache.
    */
  private def doNewHand(): CommandResult =
    val prevHand = hand
    val newNumber = prevHand.map(_.handNumber + 1).getOrElse(1)
    val (persistedStore, persistedDirty) = flushOpponentMemoryIfDirty(opponentMemoryStore, opponentMemoryDirty)
    val archivedVillainObservations = prevHand match
      case Some(previous) =>
        (rememberedVillainObservations ++ previous.villainObservations).takeRight(OpponentProfile.MaxRecentEvents)
      case None =>
        rememberedVillainObservations

    // Alternate positions
    val heroPos =
      prevHand match
        case Some(h) => if h.heroPosition == Position.Button then Position.BigBlind else Position.Button
        case None    => if config.heroStartsAsSB then Position.Button else Position.BigBlind
    val villainPos = if heroPos == Position.Button then Position.BigBlind else Position.Button

    // Post blinds
    val heroBlind = if heroPos == Position.Button then config.smallBlind else config.bigBlind
    val villainBlind = if villainPos == Position.Button then config.smallBlind else config.bigBlind
    val pot = heroBlind + villainBlind
    val heroStack = config.startingStack - heroBlind
    val villainStack = config.startingStack - villainBlind

    // In HU preflop, the dealer posts the small blind and acts first.
    val toCall =
      if heroPos == Position.Button then config.bigBlind - config.smallBlind
      else 0.0 // BB has option to check (or raise)

    val snapshot = HandSnapshot(
      handNumber = newNumber,
      pot = pot,
      heroStack = heroStack,
      villainStack = villainStack,
      toCall = toCall,
      heroCommittedThisStreet = heroBlind,
      villainCommittedThisStreet = villainBlind,
      heroPosition = heroPos,
      villainPosition = villainPos
    )

    // Finalize previous hand stats if it wasn't explicitly finished
    val updatedStats = prevHand match
      case Some(h) if !h.finished => stats
      case _                      => stats

    engine.clearInferenceCache()

    val posStr = if heroPos == Position.Button then "BTN/SB" else "BB"
    val out = Vector(
      s"--- Hand #$newNumber --- Hero: $posStr",
      f"Blinds posted. Pot: ${pot}%.1f. Hero stack: ${heroStack}%.1f. Villain stack: ${villainStack}%.1f."
    ) ++ (if toCall > 0.0 then Vector(f"Hero to act. To call: ${toCall}%.1f.") else Vector("Hero has option."))

    CommandResult(
      updated(
        newHand = Some(snapshot),
        newStats = updatedStats,
        newRememberedVillainObservations = archivedVillainObservations,
        newOpponentMemoryStore = persistedStore,
        newOpponentMemoryDirty = persistedDirty
      ),
      out
    )

  // ---- HeroCards ----

  private def doHeroCards(cards: HoleCards): CommandResult =
    hand match
      case None => CommandResult(this, Vector("No hand in progress. Type 'new' to start."))
      case Some(h) if h.finished => CommandResult(this, Vector("Hand is finished. Type 'new' for a new hand."))
      case Some(h) =>
        // Validate no overlap with board
        val boardCards = h.board.asSet
        if cards.asSet.exists(boardCards.contains) then
          CommandResult(this, Vector("Hero cards overlap with board cards."))
        else
          val event = HandEvent.HeroCardsSet(h.heroCards)
          val updated = h.copy(
            heroCards = Some(cards),
            eventLog = h.eventLog :+ event
          )
          CommandResult(
            this.updated(newHand = Some(updated), newStats = stats),
            Vector(s"Hero: ${cards.first.toToken} ${cards.second.toToken}")
          )

  // ---- Action recording ----

  private def doAction(isHero: Boolean, action: PokerAction): CommandResult =
    hand match
      case None => CommandResult(this, Vector("No hand in progress. Type 'new' to start."))
      case Some(h) if h.finished => CommandResult(this, Vector("Hand is finished. Type 'new' for a new hand."))
      case Some(h) =>
        // Validate action
        val validationError: Option[String] = action match
          case PokerAction.Check if h.toCall > 0.0 => Some(f"Cannot check, there is ${h.toCall}%.1f to call.")
          case PokerAction.Call if h.toCall <= 0.0  => Some("Nothing to call. Use 'check' instead.")
          case PokerAction.Fold if h.toCall <= 0.0  => Some("Nothing to fold to. Use 'check' instead.")
          case _ => None

        if validationError.isDefined then CommandResult(this, Vector(validationError.get))
        else doValidatedAction(h, isHero, action)

  /** Apply a validated action to the hand state.
    *
    * This is the main action-processing method.  It:
    * 1. Captures a full pre-action snapshot as a [[HandEvent.ActionRecorded]] for undo.
    * 2. Updates chip contributions, pot, stacks, and to-call based on the action type.
    * 3. Feeds villain actions to the engine for archetype learning (if responding to a hero raise).
    * 4. Records villain observations for range inference (all non-fold villain actions).
    * 5. Persists villain actions to opponent memory if configured.
    * 6. Records hero decisions for post-hand review.
    * 7. Computes hero net result on fold (the only way a hand ends through actions here).
    * 8. Updates session statistics if the hand is finished.
    */
  private def doValidatedAction(h: HandSnapshot, isHero: Boolean, action: PokerAction): CommandResult =
    val event = HandEvent.ActionRecorded(
      isHero = isHero,
      action = action,
      potBefore = h.pot,
      heroStackBefore = h.heroStack,
      villainStackBefore = h.villainStack,
      toCallBefore = h.toCall,
      heroCommittedBefore = h.heroCommittedThisStreet,
      villainCommittedBefore = h.villainCommittedThisStreet,
      betHistoryBefore = h.betHistory,
      lastHeroRaiseBefore = h.lastHeroActionWasRaise,
      villainObsBefore = h.villainObservations,
      heroDecisionsBefore = h.heroDecisions,
      finishedBefore = h.finished,
      heroNetBefore = h.heroNetResult,
      rememberedOpponentBefore = rememberedOpponent,
      opponentMemoryStoreBefore = opponentMemoryStore,
      opponentMemoryDirtyBefore = opponentMemoryDirty
    )

    val playerIdx = if isHero then HeroIdx else VillainIdx
    val who = if isHero then "Hero" else "Villain"
    val out = Vector.newBuilder[String]

    var pot = h.pot
    var heroStack = h.heroStack
    var villainStack = h.villainStack
    var toCall = h.toCall
    var heroCommitted = h.heroCommittedThisStreet
    var villainCommitted = h.villainCommittedThisStreet
    var betHistory = h.betHistory
    var lastHeroRaise = h.lastHeroActionWasRaise
    var villainObs = h.villainObservations
    var heroDecisions = h.heroDecisions
    var finished = false
    var heroNet = h.heroNetResult
    var remembered = rememberedOpponent
    var memoryStore = opponentMemoryStore
    var memoryDirty = opponentMemoryDirty

    action match
      case PokerAction.Fold =>
        finished = true
        if isHero then
          out += s"$who folds. Villain wins pot of ${fmtChips(pot)}."
        else
          out += s"$who folds. Hero wins pot of ${fmtChips(pot)}."

      case PokerAction.Check =>
        betHistory = betHistory :+ BetAction(playerIdx, PokerAction.Check)
        out += s"$who checks."

      case PokerAction.Call =>
        val callAmount = math.min(toCall, if isHero then heroStack else villainStack)
        if isHero then
          heroStack -= callAmount
          heroCommitted += callAmount
        else
          villainStack -= callAmount
          villainCommitted += callAmount
        pot += callAmount
        toCall = 0.0
        betHistory = betHistory :+ BetAction(playerIdx, PokerAction.Call)
        out += s"$who calls ${fmtChips(callAmount)}. Pot: ${fmtChips(pot)}."

      case PokerAction.Raise(amount) =>
        val alreadyIn = if isHero then heroCommitted else villainCommitted
        val additional = math.min(amount - alreadyIn, if isHero then heroStack else villainStack)
        if isHero then
          heroStack -= additional
          heroCommitted += additional
        else
          villainStack -= additional
          villainCommitted += additional
        pot += additional
        toCall = (if isHero then heroCommitted else villainCommitted) -
          (if isHero then villainCommitted else heroCommitted)
        toCall = math.max(0.0, toCall)
        betHistory = betHistory :+ BetAction(playerIdx, PokerAction.Raise(amount))
        out += s"$who raises to ${fmtChips(amount)}. Pot: ${fmtChips(pot)}. To call: ${fmtChips(toCall)}."

    // Archetype learning: villain responding to a hero raise
    if !isHero && h.lastHeroActionWasRaise then
      action match
        case PokerAction.Fold | PokerAction.Call | PokerAction.Raise(_) =>
          engine.observeVillainResponseToRaise(action)
        case _ => ()

    // Record villain observation for range inference
    if !isHero && action != PokerAction.Fold then
      val obsState = GameState(
        street = h.street, board = h.board, pot = h.pot, toCall = h.toCall,
        position = h.villainPosition, stackSize = h.villainStack, betHistory = h.betHistory
      )
      villainObs = villainObs :+ VillainObservation(action, obsState)

    if !isHero then
      val (updatedStore, updatedOpponent, updatedDirty) = persistVillainObservation(h, action)
      memoryStore = updatedStore
      remembered = updatedOpponent
      memoryDirty = updatedDirty

    // Record hero decision for review
    if isHero && action != PokerAction.Fold then
      h.heroCards.foreach { cards =>
        val gs = GameState(
          street = h.street, board = h.board, pot = h.pot, toCall = h.toCall,
          position = h.heroPosition, stackSize = h.heroStack, betHistory = h.betHistory
        )
        heroDecisions = heroDecisions :+ HeroDecisionRecord(
          street = h.street, gameState = gs, heroCards = cards,
          actualAction = action, villainObservations = combinedVillainObservations(villainObs)
        )
      }

    if isHero then
      lastHeroRaise = action match
        case PokerAction.Raise(_) => true
        case _                    => false

    // Compute hero net on fold
    if finished then
      if isHero then
        heroNet = -(config.startingStack - heroStack)
      else
        heroNet = pot - (config.startingStack - heroStack)
      val flushedMemory = flushOpponentMemoryIfDirty(memoryStore, memoryDirty)
      memoryStore = flushedMemory._1
      memoryDirty = flushedMemory._2

    val updatedSnapshot = h.copy(
      pot = pot, heroStack = heroStack, villainStack = villainStack,
      toCall = toCall, heroCommittedThisStreet = heroCommitted,
      villainCommittedThisStreet = villainCommitted, betHistory = betHistory,
      lastHeroActionWasRaise = lastHeroRaise, villainObservations = villainObs,
      heroDecisions = heroDecisions, eventLog = h.eventLog :+ event,
      finished = finished, heroNetResult = heroNet
    )

    val updatedStats =
      if finished then
        stats.copy(
          handsPlayed = stats.handsPlayed + 1,
          heroNetChips = stats.heroNetChips + heroNet,
          heroWins = stats.heroWins + (if !isHero then 1 else 0),
          heroLosses = stats.heroLosses + (if isHero then 1 else 0)
        )
      else stats

    if finished then
      out += f"--- Hand #${h.handNumber} result: ${heroNet}%+.1f ---"

    CommandResult(
      updated(
        newHand = Some(updatedSnapshot),
        newStats = updatedStats,
        newRememberedOpponent = remembered,
        newOpponentMemoryStore = memoryStore,
        newOpponentMemoryDirty = memoryDirty
      ),
      out.result()
    )

  // ---- Villain showdown ----

  private def doVillainShowdown(cards: HoleCards): CommandResult =
    hand match
      case None => CommandResult(this, Vector("No hand in progress. Type 'new' to start."))
      case Some(h) =>
        val boardCards = h.board.asSet
        val heroCards = h.heroCards.map(_.asSet).getOrElse(Set.empty)
        val dead = boardCards ++ heroCards
        if cards.asSet.exists(dead.contains) then
          CommandResult(this, Vector("Showdown cards overlap with board or hero cards."))
        else
          val updatedSnapshot = h.copy(villainRevealedCards = Some(cards))
          CommandResult(
            updated(newHand = Some(updatedSnapshot), newStats = stats),
            Vector(
              s"Villain showed ${cards.first.toToken}${cards.second.toToken}.",
              "Range posterior collapsed to revealed hand."
            )
          )

  // ---- Deal board ----

  private def doDealBoard(cards: Vector[Card]): CommandResult =
    hand match
      case None => CommandResult(this, Vector("No hand in progress. Type 'new' to start."))
      case Some(h) if h.finished => CommandResult(this, Vector("Hand is finished. Type 'new' for a new hand."))
      case Some(h) =>
        // Validate: no overlap with hero cards
        val overlap = h.heroCards.toVector.flatMap(hc => cards.filter(c => hc.contains(c)))
        if overlap.nonEmpty then
          CommandResult(this, Vector(s"Board cards overlap with hero cards: ${overlap.map(_.toToken).mkString(", ")}"))
        else
          val currentBoardSize = h.board.size
          val newTotalSize = currentBoardSize + cards.length

          val streetResult: Option[(Street, String)] = newTotalSize match
            case 3 if currentBoardSize == 0 => Some((Street.Flop, "Flop"))
            case 4 if currentBoardSize == 3 => Some((Street.Turn, "Turn"))
            case 5 if currentBoardSize == 4 => Some((Street.River, "River"))
            case _ => None

          streetResult match
            case None =>
              CommandResult(this, Vector(
                s"Invalid board deal: current board has $currentBoardSize cards, adding ${cards.length} " +
                  s"would give $newTotalSize. Expected: 0+3=Flop, 3+1=Turn, 4+1=River."
              ))
            case Some((newStreet, streetName)) =>
              val newBoard = Board(h.board.cards ++ cards)
              val event = HandEvent.BoardDealt(
                boardBefore = h.board,
                streetBefore = h.street,
                toCallBefore = h.toCall,
                heroCommittedBefore = h.heroCommittedThisStreet,
                villainCommittedBefore = h.villainCommittedThisStreet
              )

              val updatedSnapshot = h.copy(
                board = newBoard,
                street = newStreet,
                toCall = 0.0,
                heroCommittedThisStreet = 0.0,
                villainCommittedThisStreet = 0.0,
                eventLog = h.eventLog :+ event
              )

              val cardStr = cards.map(_.toToken).mkString(" ")
              CommandResult(
                updated(newHand = Some(updatedSnapshot), newStats = stats),
                Vector(s"--- $streetName: $cardStr ---")
              )

  // ---- Advise ----

  /** Run the adaptive engine to produce an action recommendation for the current state.
    *
    * Builds a [[GameState]] from the current hand snapshot, constructs candidate actions
    * (check/fold/call + several raise sizes), combines remembered + live villain observations,
    * and invokes [[engine.decide]].  The result includes equity estimate, action EVs,
    * archetype posterior, CFR equilibrium baseline (if configured), and top posterior hands.
    */
  private def doAdvise(): CommandResult =
    hand match
      case None => CommandResult(this, Vector("No hand in progress. Type 'new' to start."))
      case Some(h) if h.finished => CommandResult(this, Vector("Hand is finished. Type 'new' for a new hand."))
      case Some(h) =>
        h.heroCards match
          case None => CommandResult(this, Vector("Set hero cards first (e.g., 'h AcKh')."))
          case Some(heroCards) =>
            val gameState = GameState(
              street = h.street,
              board = h.board,
              pot = h.pot,
              toCall = h.toCall,
              position = h.heroPosition,
              stackSize = h.heroStack,
              betHistory = h.betHistory
            )

            val candidates = buildCandidateActions(h)
            // Folds before the opener: use the table format to determine which
            // positions folded before the button opened.
            val openerPos = Position.Button
            val folds = tableRanges.format.foldsBeforeOpener(openerPos).map(PreflopFold(_))
            val showdownHistory = rememberedOpponent.map(_.showdownHands).getOrElse(Vector.empty)

            val result = engine.decide(
              hero = heroCards,
              state = gameState,
              folds = folds,
              villainPos = h.villainPosition,
              observations = combinedVillainObservations(h.villainObservations),
              candidateActions = candidates,
              decisionBudgetMillis = Some(config.decisionBudgetMillis),
              rng = new Random(rng.nextLong()),
              revealedCards = h.villainRevealedCards,
              showdownHistory = showdownHistory
            )

            val out = formatAdvice(result, h)
            CommandResult(this, out)

  // ---- Review ----

  /** Re-evaluate all hero decisions in the current hand, comparing each actual action
    * to the engine's recommendation.  Reports EV difference in big blinds and flags
    * decisions where the actual action had significantly lower EV as "MISTAKE".
    */
  private def doReview(): CommandResult =
    hand match
      case None => CommandResult(this, Vector("No hand to review."))
      case Some(h) if h.heroDecisions.isEmpty =>
        CommandResult(this, Vector("No hero decisions recorded in this hand."))
      case Some(h) =>
        val out = Vector.newBuilder[String]
        out += s"--- Hand #${h.handNumber} Review ---"

        h.heroDecisions.zipWithIndex.foreach { case (dec, i) =>
          val candidates = buildCandidateActionsForState(dec.gameState)
          val folds = Vector.empty[PreflopFold]
          val showdownHistory = rememberedOpponent.map(_.showdownHands).getOrElse(Vector.empty)

          val result = engine.decide(
            hero = dec.heroCards,
            state = dec.gameState,
            folds = folds,
            villainPos = h.villainPosition,
            observations = dec.villainObservations,
            candidateActions = candidates,
            decisionBudgetMillis = Some(config.decisionBudgetMillis),
            rng = new Random(rng.nextLong()),
            revealedCards = h.villainRevealedCards,
            showdownHistory = showdownHistory
          )

          val recommended = result.decision.recommendation.bestAction
          val recEv = result.decision.recommendation.actionEvaluations
            .find(_.action == recommended).map(_.expectedValue).getOrElse(0.0)
          val actEv = HandHistoryAnalyzer.expectedValueForObservedAction(
            result.decision.recommendation.actionEvaluations,
            dec.actualAction
          )
          val evDiffChips = actEv - recEv
          val evDiff = evDiffChips / config.bigBlind
          val mark = if math.round(evDiffChips * 100.0) < 0 then "MISTAKE" else "OK"

          out += f"  ${dec.street}: Actual=${renderAction(dec.actualAction)} Recommended=${renderAction(recommended)} ${evDiff}%+.1fbb $mark"
        }

        if h.finished then
          out += f"  Result: ${h.heroNetResult}%+.1f chips"

        CommandResult(this, out.result())

  // ---- Session stats ----

  private def doSessionStats(): CommandResult =
    val s = stats
    val bb100 =
      if s.handsPlayed > 0 then (s.heroNetChips / s.handsPlayed.toDouble) * (100.0 / config.bigBlind)
      else 0.0
    val out = Vector(
      "--- Session Stats ---",
      f"  Hands played: ${s.handsPlayed}",
      f"  Net chips: ${s.heroNetChips}%+.1f",
      f"  BB/100: ${bb100}%+.1f",
      f"  Wins: ${s.heroWins}  Losses: ${s.heroLosses}  Splits: ${s.heroSplits}",
      s"  Villain archetype: ${formatArchetypePosterior(engine.archetypePosterior)}"
    ) ++ rememberedOpponent.toVector.map { profile =>
      s"  Remembered opponent: ${profile.site}/${profile.playerName} (${profile.handsObserved} hands)"
    }
    CommandResult(this, out)

  private def doQuit(): CommandResult =
    val flushedMemory = flushOpponentMemoryIfDirty(opponentMemoryStore, opponentMemoryDirty)
    CommandResult(
      updated(
        newHand = hand,
        newStats = stats,
        newOpponentMemoryStore = flushedMemory._1,
        newOpponentMemoryDirty = flushedMemory._2
      ),
      Vector("Goodbye.")
    )

  // ---- Undo ----

  /** Undo the most recent event by restoring the pre-action snapshot captured in the event.
    *
    * For [[HandEvent.ActionRecorded]] that ended the hand (fold), also reverts session stats.
    * For opponent memory changes, restores the previous memory state and re-persists it.
    */
  private def doUndo(): CommandResult =
    hand match
      case None => CommandResult(this, Vector("Nothing to undo."))
      case Some(h) if h.eventLog.isEmpty => CommandResult(this, Vector("Nothing to undo."))
      case Some(h) =>
        val lastEvent = h.eventLog.last
        val remaining = h.eventLog.init

        val restored = lastEvent match
          case HandEvent.HeroCardsSet(prev) =>
            h.copy(heroCards = prev, eventLog = remaining)

          case e: HandEvent.ActionRecorded =>
        // If we're undoing a hand-ending fold, also undo the stats
            val restoredSnapshot = h.copy(
              pot = e.potBefore,
              heroStack = e.heroStackBefore,
              villainStack = e.villainStackBefore,
              toCall = e.toCallBefore,
              heroCommittedThisStreet = e.heroCommittedBefore,
              villainCommittedThisStreet = e.villainCommittedBefore,
              betHistory = e.betHistoryBefore,
              lastHeroActionWasRaise = e.lastHeroRaiseBefore,
              villainObservations = e.villainObsBefore,
              heroDecisions = e.heroDecisionsBefore,
              finished = e.finishedBefore,
              heroNetResult = e.heroNetBefore,
              eventLog = remaining
            )
            restoredSnapshot

          case e: HandEvent.BoardDealt =>
            h.copy(
              board = e.boardBefore,
              street = e.streetBefore,
              toCall = e.toCallBefore,
              heroCommittedThisStreet = e.heroCommittedBefore,
              villainCommittedThisStreet = e.villainCommittedBefore,
              eventLog = remaining
            )

        // If we undid a fold that ended the hand, revert stats
        val restoredStats = lastEvent match
          case e: HandEvent.ActionRecorded if e.action == PokerAction.Fold && !e.finishedBefore =>
            stats.copy(
              handsPlayed = stats.handsPlayed - 1,
              heroNetChips = stats.heroNetChips - h.heroNetResult,
              heroWins = if e.isHero then stats.heroWins else stats.heroWins - 1,
              heroLosses = if e.isHero then stats.heroLosses - 1 else stats.heroLosses
            )
          case _ => stats

        val restoredMemory = lastEvent match
          case e: HandEvent.ActionRecorded =>
            saveOpponentMemorySnapshot(e.opponentMemoryStoreBefore)
            (e.rememberedOpponentBefore, e.opponentMemoryStoreBefore, false)
          case _ =>
            (rememberedOpponent, opponentMemoryStore, opponentMemoryDirty)

        CommandResult(
          updated(
            newHand = Some(restored),
            newStats = restoredStats,
            newRememberedOpponent = restoredMemory._1,
            newOpponentMemoryStore = restoredMemory._2,
            newOpponentMemoryDirty = restoredMemory._3
          ),
          Vector("Undone.")
        )

  // ---- Help ----

  private def doHelp(): CommandResult =
    val out = Vector(
      "Commands:",
      "  new                        Start a new hand (alternates position)",
      "  h AcKh / hero AcKh         Set hero hole cards",
      "  h raise 6 / h call / h fold / h check   Record hero action",
      "  v raise 8 / v call / v fold / v check    Record villain action",
      "  v show QhQs                              Record villain showdown cards (collapses range)",
      "  board Ts9h8d / board Ts 9h 8d            Deal community cards",
      "  ? / advise                  Get recommendation",
      "  review                      Review hero decisions in current hand",
      "  session / stats             Show session statistics",
      "  undo                        Undo last action",
      "  help                        Show this help",
      "  quit / q                    Exit"
    )
    CommandResult(this, out)

  // ---- Internal helpers ----

  /** Create a new session with the specified state changes, preserving all other fields.
    * This is the functional-update pattern: the session itself is immutable (except for
    * the shared mutable engine), so each command returns a fresh session.
    */
  private def updated(
      newHand: Option[HandSnapshot],
      newStats: AdvisorSessionStats,
      newRememberedOpponent: Option[OpponentProfile] = rememberedOpponent,
      newRememberedVillainObservations: Vector[VillainObservation] = rememberedVillainObservations,
      newOpponentMemoryStore: Option[OpponentProfileStore] = opponentMemoryStore,
      newOpponentMemoryDirty: Boolean = opponentMemoryDirty
  ): AdvisorSession =
    new AdvisorSession(
      config = config,
      engine = engine,
      tableRanges = tableRanges,
      hand = newHand,
      stats = newStats,
      rng = rng,
      rememberedOpponent = newRememberedOpponent,
      rememberedVillainObservations = newRememberedVillainObservations,
      opponentMemoryTarget = opponentMemoryTarget,
      opponentMemorySite = opponentMemorySite,
      opponentMemoryName = opponentMemoryName,
      opponentMemoryStore = newOpponentMemoryStore,
      opponentMemoryDirty = newOpponentMemoryDirty
    )

  /** Record a villain action in the opponent profile store (if configured).
    *
    * Creates a synthetic [[PokerEvent]] from the current hand state and feeds it to
    * the store's `observeEvent`.  Also reports whether the villain is responding to
    * a hero raise (for archetype frequency tracking).  Returns the updated store,
    * opponent profile, and dirty flag.
    */
  private def persistVillainObservation(
      h: HandSnapshot,
      action: PokerAction
  ): (Option[OpponentProfileStore], Option[OpponentProfile], Boolean) =
    (opponentMemorySite, opponentMemoryName) match
      case (Some(site), Some(name)) =>
        val currentStore = opponentMemoryStore.getOrElse(OpponentProfileStore.empty)
        val updatedStore = currentStore.observeEvent(
          site = site,
          playerName = name,
          event = PokerEvent(
            handId = s"advisor-hand-${h.handNumber}",
            sequenceInHand = h.eventLog.count {
              case _: HandEvent.ActionRecorded => true
              case _ => false
            }.toLong,
            playerId = name,
            occurredAtEpochMillis = System.currentTimeMillis(),
            street = h.street,
            position = h.villainPosition,
            board = h.board,
            potBefore = h.pot,
            toCall = h.toCall,
            stackBefore = h.villainStack,
            action = action,
            decisionTimeMillis = None,
            betHistory = h.betHistory
          ),
          facedRaiseResponse = h.lastHeroActionWasRaise
        )
        (
          Some(updatedStore),
          updatedStore.find(site, name).orElse(rememberedOpponent),
          updatedStore != currentStore
        )
      case _ =>
        (opponentMemoryStore, rememberedOpponent, opponentMemoryDirty)

  private def saveOpponentMemorySnapshot(store: Option[OpponentProfileStore]): Unit =
    opponentMemoryTarget.foreach(target => OpponentProfileStorePersistence.save(target, store.getOrElse(OpponentProfileStore.empty)))

  /** Persist the opponent profile store to disk if it has unflushed changes.
    * Called at hand boundaries and on quit to avoid losing opponent observations.
    */
  private def flushOpponentMemoryIfDirty(
      store: Option[OpponentProfileStore],
      dirty: Boolean
  ): (Option[OpponentProfileStore], Boolean) =
    if dirty then
      saveOpponentMemorySnapshot(store)
      (store, false)
    else (store, false)

  private def buildCandidateActions(h: HandSnapshot): Vector[PokerAction] =
    buildCandidateActionsForState(
      GameState(h.street, h.board, h.pot, h.toCall, h.heroPosition, h.heroStack, h.betHistory)
    )

  private def buildCandidateActionsForState(state: GameState): Vector[PokerAction] =
    val pot = state.pot
    val toCall = state.toCall
    val stack = state.stackSize

    if toCall <= 0.0 then
      // Check situation
      val raises = Vector(0.5, 0.75, 1.0, 1.5).map { f =>
        val amount = roundChips(pot * f)
        PokerAction.Raise(amount)
      }.filter { case PokerAction.Raise(a) => a > 0.0 && a <= stack; case _ => false }.distinct
      Vector(PokerAction.Check) ++ raises
    else
      // Facing a bet
      val raiseBasis = pot + toCall
      val raises = Vector(0.5, 0.75, 1.0, 1.5).map { f =>
        val amount = roundChips(raiseBasis * f)
        PokerAction.Raise(amount)
      }.filter { case PokerAction.Raise(a) => a > toCall && a <= stack; case _ => false }.distinct
      Vector(PokerAction.Fold, PokerAction.Call) ++ raises

  private def roundChips(v: Double): Double =
    math.round(v * 2.0) / 2.0 // round to nearest 0.5

  private def formatHintMetrics(metrics: Vector[Double]): String =
    metrics.map(value => f"$value%.3f").mkString("[", ", ", "]")

  private def formatAdvice(result: AdaptiveDecisionResult, h: HandSnapshot): Vector[String] =
    val out = Vector.newBuilder[String]
    val bb = config.bigBlind

    rememberedOpponent.foreach { profile =>
      out += s"  Memory: ${profile.site}/${profile.playerName} (${profile.handsObserved} hands)"
      profile.exploitHintDetails.take(2).foreach { hint =>
        out += s"  Exploit: ${hint.text} ${formatHintMetrics(hint.metrics)}"
      }
    }

    // Archetype line
    out += s"  Villain: ${formatArchetypePosterior(result.archetypePosterior)}"

    // Equity line
    val eq = result.decision.recommendation.heroEquity
    val eqPct = eq.mean * 100.0
    val errPct = eq.stderr * 100.0
    out += f"  Equity: $eqPct%.1f%% +/- $errPct%.1f%%"

    // Action evaluations sorted by EV descending
    val evals = result.decision.recommendation.actionEvaluations
      .sortBy(-_.expectedValue)
    val best = result.decision.recommendation.bestAction

    val actionLine = evals.map { ae =>
      val evBb = ae.expectedValue / bb
      val mark = if ae.action.category == best.category then " *" else ""
      f"${renderAction(ae.action)} ${evBb}%+.1fbb$mark"
    }.mkString("  ")
    out += s"  $actionLine"

    result.equilibriumBaseline.foreach { baseline =>
      val mix = baseline.actionProbabilities.toVector
        .sortBy(-_._2)
        .map { case (action, probability) =>
          f"${renderAction(action)} ${probability * 100.0}%.0f%%"
        }
        .mkString(" | ")
      out += s"  Eq(CFR): $mix"
    }
    val adaptation = result.adaptationTrace
    val adaptationReason = adaptation.reason.getOrElse("ok")
    out +=
      f"  Adaptation: ${adaptation.source} requested=${adaptation.requestedBlendWeight}%.2f " +
        f"effective=${adaptation.effectiveBlendWeight}%.2f regret=${adaptation.baselineChosenActionRegret}%.3f " +
        s"reason=$adaptationReason"

    // Top posterior hands
    val posteriorWeights = result.decision.posteriorInference.posterior.weights
    val topHands = posteriorWeights.toVector.sortBy(-_._2).take(5)
    if topHands.nonEmpty then
      val handStr = topHands.map { case (hc, prob) =>
        f"${hc.toToken} ${prob * 100.0}%.1f%%"
      }.mkString(", ")
      out += s"  Top villain: $handStr"

    out.result()

  private def combinedVillainObservations(
      liveObservations: Vector[VillainObservation]
  ): Vector[VillainObservation] =
    rememberedVillainObservations ++ liveObservations

  private def formatArchetypePosterior(posterior: ArchetypePosterior): String =
    posterior.weights.toVector
      .sortBy(-_._2)
      .map { case (archetype, prob) =>
        f"${archetype} ${prob * 100.0}%.0f%%"
      }
      .mkString(" | ")

  private def renderAction(action: PokerAction): String =
    action match
      case PokerAction.Fold       => "FOLD"
      case PokerAction.Check      => "CHECK"
      case PokerAction.Call       => "CALL"
      case PokerAction.Raise(amt) => f"RAISE ${amt}%.1f"

  private def fmtChips(v: Double): String =
    if v == v.toLong.toDouble then f"${v}%.0f" else f"${v}%.1f"
