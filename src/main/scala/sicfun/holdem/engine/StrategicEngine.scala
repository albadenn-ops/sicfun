package sicfun.holdem.engine

import sicfun.core.DiscreteDistribution
import sicfun.holdem.types.*
import sicfun.holdem.strategic.*
import sicfun.holdem.strategic.solver.WPomcpRuntime

/** Session/hand orchestrator for the Strategic decision mode.
  *
  * Manages per-rival beliefs across hands, builds the factored tabular model,
  * and delegates action selection to WPomcpRuntime.solveV2.
  *
  * Lifecycle:
  *   1. Call [[initSession]] once to register rival IDs and seed priors.
  *   2. Call [[startHand]] at the beginning of each hand.
  *   3. Call [[observeAction]] for each rival action observed mid-hand.
  *   4. Call [[decide]] when hero must act.
  *   5. Call [[endHand]] when the hand concludes.
  */
class StrategicEngine(val config: StrategicEngine.Config):

  private var _sessionState: StrategicEngine.SessionState | Null = null
  private var _handActive: Boolean = false
  private var _heroCards: Option[HoleCards] = None
  private var _actionHistory: Vector[PublicAction] = Vector.empty
  private var _lastBoard: Option[Board] = None
  private var _lastStreet: Option[Street] = None

  def sessionState: StrategicEngine.SessionState =
    require(_sessionState != null, "Session not initialized — call initSession first")
    _sessionState.nn

  def currentHandActive: Boolean = _handActive
  def isSessionInitialized: Boolean = _sessionState != null

  /** Initialize session with rival IDs. Uses uniform priors unless existing beliefs provided. */
  def initSession(
      rivalIds: Vector[PlayerId],
      rivalSeats: Map[PlayerId, StrategicEngine.RivalSeatInfo] = Map.empty,
      existingBeliefs: Map[PlayerId, StrategicRivalBelief] = Map.empty
  ): Unit =
    val beliefs = rivalIds.map { id =>
      id -> existingBeliefs.getOrElse(id, StrategicRivalBelief.uniform)
    }.toMap
    val exploitStates = rivalIds.map { id =>
      id -> ExploitationState.initial(config.exploitConfig)
    }.toMap
    _sessionState = StrategicEngine.SessionState(
      rivalBeliefs = beliefs,
      exploitationStates = exploitStates,
      rivalSeats = rivalSeats
    )

  /** Start a new hand. Resets hand-local state, preserves session beliefs.
    *
    * @param heroCards hero's hole cards for hand-strength bucket estimation
    */
  def startHand(heroCards: HoleCards): Unit =
    require(_sessionState != null, "Session not initialized — call initSession first")
    _handActive = true
    _heroCards = Some(heroCards)
    _actionHistory = Vector.empty
    _lastBoard = None
    _lastStreet = None

  /** Start a new hand without hero cards (fallback — uses neutral middle bucket). */
  def startHand(): Unit =
    require(_sessionState != null, "Session not initialized — call initSession first")
    _handActive = true
    _heroCards = None
    _actionHistory = Vector.empty
    _lastBoard = None
    _lastStreet = None

  /** Observe a rival's action.
    *
    * Performs a full Dynamics.fullStep belief update using the kernel pipeline:
    * converts the action to a TotalSignal, bridges the GameState to a strategic
    * PublicState, and applies the tempered likelihood update to rival beliefs.
    */
  def observeAction(actor: PlayerId, action: PokerAction, gameState: GameState): Unit =
    if _sessionState == null then return
    val session = _sessionState.nn
    if !session.rivalBeliefs.contains(actor) then return

    val actionSignal = bridgeActionSignal(action, gameState)
    _actionHistory = _actionHistory :+ PublicAction(actor, actionSignal)
    _lastBoard = Some(gameState.board)
    _lastStreet = Some(gameState.street)

    val signal = TotalSignal(
      actionSignal = actionSignal,
      showdown = None
    )
    val pubState = bridgePublicState(gameState)
    val kernelProfile = buildKernelProfile()
    val exploitConfigs = session.rivalBeliefs.keys.map(id => id -> config.exploitConfig).toMap

    val result = Dynamics.fullStep[StrategicRivalBelief](
      rivalStates = session.rivalBeliefs,
      exploitStates = session.exploitationStates,
      signal = signal,
      publicState = pubState,
      kernelProfile = kernelProfile,
      exploitConfigs = exploitConfigs,
      detector = NeverDetect,
      exploitabilityFn = beta => computeExploitabilityEstimate(beta),
      epsilonNE = 0.01
    )

    _sessionState = StrategicEngine.SessionState(
      rivalBeliefs = result.updatedRivals,
      exploitationStates = result.updatedExploitation,
      rivalSeats = session.rivalSeats
    )

  /** Choose an action using the WPomcp V2 solver.
    *
    * Falls back to the last candidate action if the native solver is unavailable.
    */
  def decide(gameState: GameState, candidateActions: Vector[PokerAction]): PokerAction =
    require(_sessionState != null, "Session not initialized")
    require(_handActive, "No hand in progress")
    require(candidateActions.nonEmpty, "No candidate actions")

    val heroBucket = estimateHeroBucket(gameState)

    val searchInput = PokerPomcpFormulation.buildSearchInputV2(
      gameState = gameState,
      rivalBeliefs = _sessionState.nn.rivalBeliefs,
      heroActions = candidateActions,
      heroBucket = heroBucket,
      particlesPerRival = config.particlesPerRival
    )

    WPomcpRuntime.solveV2(searchInput, WPomcpRuntime.Config(
      numSimulations = config.numSimulations,
      discount = config.discount,
      maxDepth = config.maxDepth,
      seed = config.seed
    )) match
      case Right(result) =>
        if result.bestAction >= 0 && result.bestAction < candidateActions.size then
          candidateActions(result.bestAction)
        else
          candidateActions.last
      case Left(_) =>
        candidateActions.find(_ != PokerAction.Fold).getOrElse(PokerAction.Fold)

  /** End the current hand. If showdown data is provided, applies ShowdownKernel
    * to update rival beliefs based on revealed hands.
    */
  def endHand(showdownResult: Option[Map[PlayerId, HoleCards]] = None): Unit =
    if _sessionState != null && showdownResult.exists(_.nonEmpty) then
      val session = _sessionState.nn
      val board = _lastBoard.getOrElse(Board.empty)
      val street = _lastStreet.getOrElse(Street.River)
      val updatedBeliefs = session.rivalBeliefs.map { case (rivalId, belief) =>
        showdownResult.flatMap(_.get(rivalId)) match
          case Some(revealedCards) =>
            val lastAct = _actionHistory.filter(_.actor == rivalId).lastOption.map(_.signal.action)
            val sdKernel = makeShowdownKernel(board, street, lastAct)
            val signal = ShowdownSignal(Vector(
              RevealedHand(rivalId, revealedCards.toVector)
            ))
            rivalId -> sdKernel.apply(belief, signal)
          case None =>
            rivalId -> belief
      }
      _sessionState = StrategicEngine.SessionState(
        rivalBeliefs = updatedBeliefs,
        exploitationStates = session.exploitationStates,
        rivalSeats = session.rivalSeats
      )
    _handActive = false
    _heroCards = None

  /** Compute exploitability estimate at a given beta level.
    * Uses posterior concentration as a proxy for exploitability.
    * Returns 0.0 when insufficient data is available.
    */
  private[holdem] def computeExploitabilityEstimate(beta: Double): Double =
    if _sessionState == null then return 0.0
    val session = _sessionState.nn
    val beliefs = session.rivalBeliefs.values.toIndexedSeq
    if beliefs.isEmpty then return 0.0
    val deviations = beliefs.map { belief =>
      val classes = StrategicClass.values
      val probs = classes.map(c => belief.typePosterior.probabilityOf(c))
      val maxDev = probs.max - 0.25  // deviation from uniform
      math.max(0.0, maxDev) * beta
    }
    if deviations.isEmpty then 0.0
    else deviations.max

  private def estimateHeroBucket(gameState: GameState): Int =
    _heroCards match
      case Some(cards) =>
        val strength = HandStrengthEstimator.fastGtoStrength(cards, gameState.board, gameState.street)
        math.min(9, math.max(0, (strength * 10.0).toInt))
      case None =>
        5 // Neutral middle bucket — no card info available

  /** Bridge GameState -> strategic PublicState for the kernel pipeline. */
  private def bridgePublicState(gameState: GameState): PublicState =
    val heroId = PlayerId("hero")
    val heroSeat = Seat(heroId, gameState.position, SeatStatus.Active, Chips(gameState.stackSize))
    val rivalSeats = _sessionState.nn.rivalSeats.map { case (id, info) =>
      Seat(id, info.position, SeatStatus.Active, Chips(info.stack))
    }.toVector
    val allSeats = if rivalSeats.nonEmpty then heroSeat +: rivalSeats
      else Vector(heroSeat)
    PublicState(
      street = gameState.street,
      board = gameState.board,
      pot = Chips(gameState.pot),
      stacks = TableMap(
        hero = heroId,
        seats = allSeats
      ),
      actionHistory = _actionHistory
    )

  /** Bridge PokerAction -> ActionSignal for the kernel pipeline. */
  private def bridgeActionSignal(action: PokerAction, gameState: GameState): ActionSignal =
    ActionSignal(
      action = action.category,
      sizing = action match
        case PokerAction.Raise(amount) =>
          Some(Sizing(
            Chips(amount),
            PotFraction(if gameState.pot > 0 then amount / gameState.pot else 1.0)
          ))
        case _ => None,
      timing = None,
      stage = gameState.street
    )

  private def actionPrior(cls: StrategicClass, cat: PokerAction.Category): Double =
    config.actionPriors.getOrElse((cls, cat), 0.25)

  /** Build the tempered likelihood function for kernel updates. */
  private def buildLikelihoodFn(): TemperedLikelihoodFn =
    (signal: ActionSignal, pubState: PublicState, rivalState: RivalBeliefState) =>
      val classes = StrategicClass.values
      val eta = TemperedLikelihood.defaultEta(classes.length)

      val basePr = classes.map { cls =>
        actionPrior(cls, signal.action)
      }

      val prior = rivalState match
        case srb: StrategicRivalBelief => classes.map(c => srb.typePosterior.probabilityOf(c))
        case _ => Array.fill(classes.length)(0.25)

      val posterior = TemperedLikelihood.updatePosterior(prior, basePr, eta, config.temperedConfig)
      DiscreteDistribution(classes.zip(posterior).toMap)

  /** Build the joint kernel profile for all rivals. */
  private def buildKernelProfile(): JointKernelProfile[StrategicRivalBelief] =
    val likelihood = buildLikelihoodFn()
    val actionKernel = KernelConstructor.buildActionKernelFull[StrategicRivalBelief](
      StrategicRivalBelief.updater,
      likelihood
    )
    val showdownKernel = makeShowdownKernel(
      _lastBoard.getOrElse(Board.empty),
      _lastStreet.getOrElse(Street.Preflop),
      _actionHistory.lastOption.map(_.signal.action)
    )
    val fullKernel = KernelConstructor.composeFullKernelFromFull(actionKernel, showdownKernel)
    JointKernelProfile(
      _sessionState.nn.rivalBeliefs.keys.map(id => id -> fullKernel).toMap
    )

  /** Real showdown kernel: classifies revealed hand and hard-shifts posterior. */
  private def makeShowdownKernel(
      board: Board, street: Street, lastAction: Option[PokerAction.Category]
  ): ShowdownKernel[StrategicRivalBelief] =
    new ShowdownKernel[StrategicRivalBelief]:
      def apply(state: StrategicRivalBelief, showdown: ShowdownSignal): StrategicRivalBelief =
        if showdown.revealedHands.isEmpty then return state
        val revealed = showdown.revealedHands.head
        val observedClass = classifyRevealedHand(revealed.cards, board, street, lastAction)
        val smoothing = 0.10
        val classes = StrategicClass.values
        val shifted = classes.map { cls =>
          val prior = state.typePosterior.probabilityOf(cls)
          val target = if cls == observedClass then 1.0 else 0.0
          cls -> ((1.0 - smoothing) * target + smoothing * prior)
        }.toMap
        StrategicRivalBelief(DiscreteDistribution(shifted))

  /** Classify a revealed hand into StrategicClass based on hand strength and action. */
  private def classifyRevealedHand(
      cards: Vector[sicfun.core.Card],
      board: Board,
      street: Street,
      lastAction: Option[PokerAction.Category]
  ): StrategicClass =
    if cards.size < 2 then return StrategicClass.Marginal
    val holeCards = HoleCards.from(cards.take(2))
    val strength = HandStrengthEstimator.fastGtoStrength(holeCards, board, street)
    val wasAggressive = lastAction.exists(_ == PokerAction.Category.Raise)
    if strength >= 0.65 then
      StrategicClass.Value
    else if strength < 0.35 && wasAggressive then
      StrategicClass.Bluff
    else if strength >= 0.35 && strength < 0.55 && wasAggressive then
      StrategicClass.SemiBluff
    else
      StrategicClass.Marginal

object StrategicEngine:

  /** Default action priors P(action_category | strategic_class).
    * These are initial estimates pending calibration from showdown data.
    * Exposed in Config so callers can override with calibrated values.
    */
  val defaultActionPriors: Map[(StrategicClass, sicfun.holdem.types.PokerAction.Category), Double] = {
    import sicfun.holdem.types.PokerAction.Category.*
    Map(
      (StrategicClass.Value, Fold) -> 0.05, (StrategicClass.Value, Check) -> 0.35,
      (StrategicClass.Value, Call) -> 0.40, (StrategicClass.Value, Raise) -> 0.20,
      (StrategicClass.Bluff, Fold) -> 0.10, (StrategicClass.Bluff, Check) -> 0.10,
      (StrategicClass.Bluff, Call) -> 0.15, (StrategicClass.Bluff, Raise) -> 0.65,
      (StrategicClass.SemiBluff, Fold) -> 0.05, (StrategicClass.SemiBluff, Check) -> 0.15,
      (StrategicClass.SemiBluff, Call) -> 0.30, (StrategicClass.SemiBluff, Raise) -> 0.50,
      (StrategicClass.Marginal, Fold) -> 0.15, (StrategicClass.Marginal, Check) -> 0.40,
      (StrategicClass.Marginal, Call) -> 0.35, (StrategicClass.Marginal, Raise) -> 0.10
    )
  }

  /** Configuration for a StrategicEngine session. */
  final case class Config(
      numSimulations: Int = 500,
      discount: Double = 0.95,
      maxDepth: Int = 20,
      seed: Long = 42L,
      particlesPerRival: Int = 100,
      exploitConfig: ExploitationConfig = ExploitationConfig(
        initialBeta = 1.0,
        retreatRate = 0.1,
        adaptationTolerance = 0.05
      ),
      temperedConfig: TemperedLikelihood.TemperedConfig = TemperedLikelihood.TemperedConfig.twoLayer(0.7, 0.01),
      actionPriors: Map[(StrategicClass, sicfun.holdem.types.PokerAction.Category), Double] = defaultActionPriors
  )

  /** Rival seat information provided at session init. */
  final case class RivalSeatInfo(position: Position, stack: Double)

  /** Per-session state: rival beliefs and exploitation states that survive across hands. */
  final case class SessionState(
      rivalBeliefs: Map[PlayerId, StrategicRivalBelief],
      exploitationStates: Map[PlayerId, ExploitationState],
      rivalSeats: Map[PlayerId, RivalSeatInfo] = Map.empty
  )
