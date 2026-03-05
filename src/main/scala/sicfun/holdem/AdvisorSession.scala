package sicfun.holdem

import sicfun.core.Card

import scala.util.Random

/** Configuration for a heads-up advisor session. */
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

/** Events logged for undo support. */
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
      heroNetBefore: Double
  )
  case BoardDealt(
      boardBefore: Board,
      streetBefore: Street,
      toCallBefore: Double,
      heroCommittedBefore: Double,
      villainCommittedBefore: Double
  )

/** In-progress hand state. */
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
    heroPosition: Position = Position.SmallBlind,
    villainPosition: Position = Position.BigBlind,
    betHistory: Vector[BetAction] = Vector.empty,
    villainObservations: Vector[VillainObservation] = Vector.empty,
    heroDecisions: Vector[HeroDecisionRecord] = Vector.empty,
    eventLog: Vector[HandEvent] = Vector.empty,
    lastHeroActionWasRaise: Boolean = false,
    finished: Boolean = false,
    heroNetResult: Double = 0.0
)

/** Result of executing a single command. */
final case class CommandResult(
    session: AdvisorSession,
    output: Vector[String]
)

/** Pure-ish state machine for an interactive poker advisor session.
  *
  * The `RealTimeAdaptiveEngine` is intentionally mutable (archetype posterior
  * updates are side-effectful) — this matches how it is used everywhere in
  * the codebase. The rest of the state is replaced functionally on each command.
  */
final class AdvisorSession(
    val config: SessionConfig,
    val engine: RealTimeAdaptiveEngine,
    val tableRanges: TableRanges,
    val hand: Option[HandSnapshot],
    val stats: AdvisorSessionStats,
    val rng: Random
):
  private val HeroIdx = 0
  private val VillainIdx = 1

  def execute(command: AdvisorCommand): CommandResult =
    command match
      case AdvisorCommand.NewHand          => doNewHand()
      case AdvisorCommand.HeroCards(cards) => doHeroCards(cards)
      case AdvisorCommand.HeroAction(a)    => doAction(isHero = true, a)
      case AdvisorCommand.VillainAction(a) => doAction(isHero = false, a)
      case AdvisorCommand.DealBoard(cards) => doDealBoard(cards)
      case AdvisorCommand.Advise           => doAdvise()
      case AdvisorCommand.Review           => doReview()
      case AdvisorCommand.SessionStats     => doSessionStats()
      case AdvisorCommand.Undo             => doUndo()
      case AdvisorCommand.Help             => doHelp()
      case AdvisorCommand.Quit             => CommandResult(this, Vector("Goodbye."))
      case AdvisorCommand.Unknown(input, reason) =>
        CommandResult(this, Vector(s"Unknown command: $reason. Type 'help' for usage."))

  // ---- NewHand ----

  private def doNewHand(): CommandResult =
    val prevHand = hand
    val newNumber = prevHand.map(_.handNumber + 1).getOrElse(1)

    // Alternate positions
    val heroPos =
      prevHand match
        case Some(h) => if h.heroPosition == Position.SmallBlind then Position.BigBlind else Position.SmallBlind
        case None    => if config.heroStartsAsSB then Position.SmallBlind else Position.BigBlind
    val villainPos = if heroPos == Position.SmallBlind then Position.BigBlind else Position.SmallBlind

    // Post blinds
    val heroBlind = if heroPos == Position.SmallBlind then config.smallBlind else config.bigBlind
    val villainBlind = if villainPos == Position.SmallBlind then config.smallBlind else config.bigBlind
    val pot = heroBlind + villainBlind
    val heroStack = config.startingStack - heroBlind
    val villainStack = config.startingStack - villainBlind

    // In HU preflop, SB acts first. toCall = difference between blinds for the SB player.
    val toCall =
      if heroPos == Position.SmallBlind then config.bigBlind - config.smallBlind
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

    val posStr = if heroPos == Position.SmallBlind then "SB" else "BB"
    val out = Vector(
      s"--- Hand #$newNumber --- Hero: $posStr",
      f"Blinds posted. Pot: ${pot}%.1f. Hero stack: ${heroStack}%.1f. Villain stack: ${villainStack}%.1f."
    ) ++ (if toCall > 0.0 then Vector(f"Hero to act. To call: ${toCall}%.1f.") else Vector("Hero has option."))

    CommandResult(updated(newHand = Some(snapshot), newStats = updatedStats), out)

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
      heroNetBefore = h.heroNetResult
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

    // Record hero decision for review
    if isHero && action != PokerAction.Fold then
      h.heroCards.foreach { cards =>
        val gs = GameState(
          street = h.street, board = h.board, pot = h.pot, toCall = h.toCall,
          position = h.heroPosition, stackSize = h.heroStack, betHistory = h.betHistory
        )
        heroDecisions = heroDecisions :+ HeroDecisionRecord(
          street = h.street, gameState = gs, heroCards = cards,
          actualAction = action, villainObservations = villainObs
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

    CommandResult(updated(newHand = Some(updatedSnapshot), newStats = updatedStats), out.result())

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
            val openerPos = if h.heroPosition == Position.SmallBlind then Position.Button else Position.BigBlind
            val folds = tableRanges.format.foldsBeforeOpener(openerPos).map(PreflopFold(_))

            val result = engine.decide(
              hero = heroCards,
              state = gameState,
              folds = folds,
              villainPos = h.villainPosition,
              observations = h.villainObservations,
              candidateActions = candidates,
              decisionBudgetMillis = Some(config.decisionBudgetMillis),
              rng = new Random(rng.nextLong())
            )

            val out = formatAdvice(result, h)
            CommandResult(this, out)

  // ---- Review ----

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

          val result = engine.decide(
            hero = dec.heroCards,
            state = dec.gameState,
            folds = folds,
            villainPos = h.villainPosition,
            observations = dec.villainObservations,
            candidateActions = candidates,
            decisionBudgetMillis = Some(config.decisionBudgetMillis),
            rng = new Random(rng.nextLong())
          )

          val recommended = result.decision.recommendation.bestAction
          val recCategory = recommended.category
          val actCategory = dec.actualAction.category
          val mark = if recCategory == actCategory then "OK" else "MISTAKE"

          val recEv = result.decision.recommendation.actionEvaluations
            .find(_.action == recommended).map(_.expectedValue).getOrElse(0.0)
          val actEv = result.decision.recommendation.actionEvaluations
            .find(e => e.action.category == actCategory).map(_.expectedValue).getOrElse(0.0)
          val evDiff = (actEv - recEv) / config.bigBlind

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
    )
    CommandResult(this, out)

  // ---- Undo ----

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

        CommandResult(updated(newHand = Some(restored), newStats = restoredStats), Vector("Undone."))

  // ---- Help ----

  private def doHelp(): CommandResult =
    val out = Vector(
      "Commands:",
      "  new                        Start a new hand (alternates position)",
      "  h AcKh / hero AcKh         Set hero hole cards",
      "  h raise 6 / h call / h fold / h check   Record hero action",
      "  v raise 8 / v call / v fold / v check    Record villain action",
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

  private def updated(
      newHand: Option[HandSnapshot],
      newStats: AdvisorSessionStats
  ): AdvisorSession =
    new AdvisorSession(config, engine, tableRanges, newHand, newStats, rng)

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

  private def formatAdvice(result: AdaptiveDecisionResult, h: HandSnapshot): Vector[String] =
    val out = Vector.newBuilder[String]
    val bb = config.bigBlind

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

    // Top posterior hands
    val posteriorWeights = result.decision.posteriorInference.posterior.weights
    val topHands = posteriorWeights.toVector.sortBy(-_._2).take(5)
    if topHands.nonEmpty then
      val handStr = topHands.map { case (hc, prob) =>
        f"${hc.toToken} ${prob * 100.0}%.1f%%"
      }.mkString(", ")
      out += s"  Top villain: $handStr"

    out.result()

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
