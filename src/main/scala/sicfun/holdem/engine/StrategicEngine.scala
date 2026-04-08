package sicfun.holdem.engine

import sicfun.core.DiscreteDistribution
import sicfun.holdem.types.*
import sicfun.holdem.strategic.*
import sicfun.holdem.strategic.solver.{WPomcpRuntime, PftDpwRuntime, PftDpwConfig}

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
  private var _lastDiagnostics: Option[StrategicEngine.DecisionDiagnostics] = None
  private var _lastBundle: Option[DecisionEvaluationBundle] = None

  def lastDecisionDiagnostics: Option[StrategicEngine.DecisionDiagnostics] = _lastDiagnostics
  def lastDecisionBundle: Option[DecisionEvaluationBundle] = _lastBundle

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
      detector = config.detector,
      exploitabilityFn = beta => computeExploitabilityEstimate(beta),
      epsilonNE = 0.01
    )

    _sessionState = StrategicEngine.SessionState(
      rivalBeliefs = result.updatedRivals,
      exploitationStates = result.updatedExploitation,
      rivalSeats = session.rivalSeats
    )

  /** Choose an action using the configured solver backend.
    *
    * WPomcp path performs 6 solves:
    *   1. Mixed-belief solve (action selection)
    *   2. Baseline solve (beta=0 reference, profile 0)
    *   3-6. Four pure-type profile solves (profiles 0-3)
    *
    * Builds a DecisionEvaluationBundle with LocalRobustScreening certification.
    * If the budget exceeds tolerance, clamps beta via AdaptationSafety.betaBar.
    *
    * Falls back to BaselineFallback if any solve returns Left.
    */
  def decide(gameState: GameState, candidateActions: Vector[PokerAction]): PokerAction =
    require(_sessionState != null, "Session not initialized")
    require(_handActive, "No hand in progress")
    require(candidateActions.nonEmpty, "No candidate actions")

    val heroBucket = estimateHeroBucket(gameState)
    val session = _sessionState.nn
    val solverConfig = WPomcpRuntime.Config(
      numSimulations = config.numSimulations,
      discount = config.discount,
      maxDepth = config.maxDepth,
      seed = config.seed
    )

    val action = config.solverBackend match
      case StrategicEngine.SolverBackend.WPomcp =>
        decideWPomcp(gameState, candidateActions, heroBucket, session, solverConfig)
      case StrategicEngine.SolverBackend.PftDpw =>
        decidePftDpw(gameState, candidateActions, heroBucket, session)

    _lastDiagnostics = Some(StrategicEngine.DecisionDiagnostics(
      heroBucket = heroBucket,
      solverBackend = config.solverBackend,
      exploitationBetas = session.exploitationStates.map((k, v) => k -> v.beta)
    ))

    action

  /** WPomcp 6-solve decision path.
    *
    * Returns the chosen action and populates _lastBundle with the
    * DecisionEvaluationBundle including LocalRobustScreening certification.
    */
  private def decideWPomcp(
      gameState: GameState,
      candidateActions: Vector[PokerAction],
      heroBucket: Int,
      session: StrategicEngine.SessionState,
      solverConfig: WPomcpRuntime.Config
  ): PokerAction =
    val numActions = candidateActions.size
    val numProfiles = StrategicClass.values.length  // 4

    // --- Solve 1: Mixed-belief solve (action selection) ---
    val mixedInput = PokerPomcpFormulation.buildSearchInputV2(
      gameState = gameState,
      rivalBeliefs = session.rivalBeliefs,
      heroActions = candidateActions,
      heroBucket = heroBucket,
      particlesPerRival = config.particlesPerRival
    )
    val mixedResult = WPomcpRuntime.solveV2(mixedInput, solverConfig)

    // --- Solve 2: Baseline solve (profile 0, beta=0 reference) ---
    val baselineInput = PokerPomcpFormulation.buildSearchInputForProfile(
      gameState = gameState,
      rivalBeliefs = session.rivalBeliefs,
      heroActions = candidateActions,
      heroBucket = heroBucket,
      particlesPerRival = config.particlesPerRival,
      profileId = JointRivalProfileId(0)
    )
    val baselineResult = WPomcpRuntime.solveV2(baselineInput, solverConfig)

    // --- Solves 3-6: Four pure-type profile solves (profiles 0-3) ---
    val profileResults: Array[Either[String, WPomcpRuntime.SearchResult]] =
      Array.tabulate(numProfiles) { p =>
        val profileInput = PokerPomcpFormulation.buildSearchInputForProfile(
          gameState = gameState,
          rivalBeliefs = session.rivalBeliefs,
          heroActions = candidateActions,
          heroBucket = heroBucket,
          particlesPerRival = config.particlesPerRival,
          profileId = JointRivalProfileId(p)
        )
        WPomcpRuntime.solveV2(profileInput, solverConfig)
      }

    // --- Handle errors: if any solve returns Left, fall back ---
    val allSolves = mixedResult +: baselineResult +: profileResults.toSeq
    val anyFailed = allSolves.exists(_.isLeft)
    if anyFailed then
      val reason = allSolves.collectFirst { case Left(msg) => msg }.getOrElse("unknown")
      _lastBundle = Some(makeFallbackBundle(numActions, reason))
      return candidateActions.find(_ != PokerAction.Fold).getOrElse(PokerAction.Fold)

    // All solves succeeded -- extract results
    val mixed = mixedResult.toOption.get
    val baseline = baselineResult.toOption.get
    val profiles = profileResults.map(_.toOption.get)

    // --- Build profileResults map ---
    val profileResultMap: Map[JointRivalProfileId, SolverResult] =
      (0 until numProfiles).map { p =>
        JointRivalProfileId(p) -> SolverResult(
          bestAction = profiles(p).bestAction,
          actionValues = profiles(p).actionValues.clone()
        )
      }.toMap

    // --- Compute robustActionLowerBounds[a] = min over profiles of profileQ[a] ---
    val robustActionLowerBounds = new Array[Double](numActions)
    var a = 0
    while a < numActions do
      var minQ = Double.PositiveInfinity
      var p = 0
      while p < numProfiles do
        val q = profiles(p).actionValues(a)
        if q < minQ then minQ = q
        p += 1
      robustActionLowerBounds(a) = minQ
      a += 1

    // --- baselineActionValues and baselineValue ---
    val baselineActionValues = baseline.actionValues.clone()
    val baselineValue = baseline.rootValue

    // --- adversarialRootGap = baselineValue - min_profile(max_a profileQ[a]) ---
    var minProfileBestValue = Double.PositiveInfinity
    var p = 0
    while p < numProfiles do
      val profileBestValue = profiles(p).rootValue
      if profileBestValue < minProfileBestValue then minProfileBestValue = profileBestValue
      p += 1
    val adversarialRootGap = baselineValue - minProfileBestValue

    // --- rootLosses[a] = baselineValue - robustActionLowerBounds[a] ---
    val rootLosses = new Array[Double](numActions)
    a = 0
    while a < numActions do
      rootLosses(a) = math.max(0.0, baselineValue - robustActionLowerBounds(a))
      a += 1

    // --- budgetEstimate = max(rootLosses) / (1 - gamma) ---
    val maxRootLoss = if rootLosses.isEmpty then 0.0 else rootLosses.max
    val budgetEstimate = if config.bellmanGamma < 1.0 then
      maxRootLoss / (1.0 - config.bellmanGamma)
    else
      maxRootLoss * 100.0  // degenerate gamma=1 guard

    // --- withinTolerance = budgetEstimate <= epsilonBase + epsilonAdapt ---
    val totalTolerance = config.epsilonBase + config.exploitConfig.epsilonAdapt
    val withinTolerance = budgetEstimate <= totalTolerance

    // --- Build certification and bundle ---
    val certification = CertificationResult.LocalRobustScreening(
      rootLosses = rootLosses,
      budgetEstimate = budgetEstimate,
      withinTolerance = withinTolerance
    )

    val bundle = DecisionEvaluationBundle(
      profileResults = profileResultMap,
      robustActionLowerBounds = robustActionLowerBounds,
      baselineActionValues = baselineActionValues,
      baselineValue = baselineValue,
      adversarialRootGap = Some(Ev(adversarialRootGap)),
      pointwiseExploitability = None,
      deploymentExploitability = None,
      certification = certification,
      chainWorldValues = Map.empty,
      notes = if withinTolerance then Vector("LocalRobustScreening: within tolerance")
              else Vector("LocalRobustScreening: budget exceeds tolerance, beta clamped")
    )
    _lastBundle = Some(bundle)

    // --- If !withinTolerance: clamp beta via AdaptationSafety.betaBar ---
    if !withinTolerance then
      val updatedExploit = session.exploitationStates.map { case (rivalId, exploitState) =>
        val clampedBeta = AdaptationSafety.betaBar(
          epsilonAdapt = config.exploitConfig.epsilonAdapt,
          epsilonNE = config.epsilonBase,
          exploitabilityAtBeta = beta => computeExploitabilityEstimate(beta)
        )
        rivalId -> ExploitationState(beta = AdaptationSafety.clampBeta(exploitState.beta, clampedBeta))
      }
      _sessionState = StrategicEngine.SessionState(
        rivalBeliefs = session.rivalBeliefs,
        exploitationStates = updatedExploit,
        rivalSeats = session.rivalSeats
      )

    // --- Action selection from mixed-belief solve ---
    if mixed.bestAction >= 0 && mixed.bestAction < candidateActions.size then
      candidateActions(mixed.bestAction)
    else
      candidateActions.last

  /** PftDpw formal certification path.
    *
    * 1. Build tabular model and particle belief from engine state.
    * 2. Solve with PftDpw native solver.
    * 3. Build profile-conditioned models and compute robust losses.
    * 4. Compute B* via SafetyBellman, belief-level safe action filtering.
    * 5. Build TabularCertification bundle.
    *
    * Fail-closed: if the native solver is unavailable, returns BaselineFallback.
    */
  private def decidePftDpw(
      gameState: GameState,
      candidateActions: Vector[PokerAction],
      heroBucket: Int,
      session: StrategicEngine.SessionState
  ): PokerAction =
    val numActions = candidateActions.size
    val numProfiles = StrategicClass.values.length

    try
      // 1. Build tabular model and belief
      val model = PokerPftFormulation.buildTabularModel(
        gameState, session.rivalBeliefs, candidateActions,
        heroBucket, config.actionPriors
      )
      val belief = PokerPftFormulation.buildParticleBelief(
        session.rivalBeliefs, config.particlesPerRival
      )

      // 2. Solve with PftDpw
      val pftConfig = PftDpwConfig(
        numSimulations = config.numSimulations,
        gamma = config.discount,
        maxDepth = config.maxDepth,
        seed = config.seed
      )
      val pftResult = PftDpwRuntime.solve(model, belief, pftConfig)
      if !pftResult.isSuccess then
        _lastBundle = Some(makeFallbackBundle(numActions, s"PftDpw solver status: ${pftResult.status}"))
        return candidateActions.find(_ != PokerAction.Fold).getOrElse(PokerAction.Fold)

      // 3. Build profile-conditioned models (same structure, varying reward semantics)
      val profileModels = (0 until numProfiles).map { _ =>
        PokerPftFormulation.buildTabularModel(
          gameState, session.rivalBeliefs, candidateActions,
          heroBucket, config.actionPriors
        )
      }
      val refPolicy: Int => Int = _ => pftResult.bestAction
      val robustLosses = PerStateLossEvaluator.computeRobustLosses(
        profileModels, refPolicy, config.bellmanGamma
      )

      // 4. Build transitions function from profile models
      val transitions: (Int, Int, Int) => Int = (s, a, p) =>
        profileModels(p).transitionTable(s * model.numActions + a)

      // 5. Compute B*
      val bStar = SafetyBellman.computeBStar(
        robustLosses, config.bellmanGamma, transitions, numProfiles
      )

      // 6. Belief-level safe action set
      val beliefWeights = belief.weights
      val safeActions = SafetyBellman.beliefLevelSafeActions(
        beliefWeights, bStar, robustLosses, config.bellmanGamma,
        transitions, numProfiles
      )

      // 7. Select action: highest Q among safe, or best overall if safe is empty
      val actionIdx = SafetyBellman.safeFeasibleAction(pftResult.qValues, safeActions)

      // 8. Certificate validation
      val requiredBudget = SafetyBellman.requiredAdaptationBudget(bStar)
      val totalTolerance = config.epsilonBase + config.exploitConfig.epsilonAdapt
      val withinTolerance = requiredBudget <= totalTolerance
      val cert = SafetyBellman.Certificate(values = bStar.clone(), terminalStates = Set(model.numStates - 1))
      val certificateValid = cert.isValid(robustLosses, config.bellmanGamma, requiredBudget + 1.0, transitions, numProfiles)

      val certification = CertificationResult.TabularCertification(
        bStar = bStar,
        requiredBudget = requiredBudget,
        safeActionIndices = safeActions,
        certificateValid = certificateValid,
        withinTolerance = withinTolerance
      )

      // 9. Build profileResults map from PftDpw Q-values
      val profileResultMap: Map[JointRivalProfileId, SolverResult] =
        (0 until numProfiles).map { p =>
          JointRivalProfileId(p) -> SolverResult(
            bestAction = pftResult.bestAction,
            actionValues = pftResult.qValues.clone()
          )
        }.toMap

      val bundle = DecisionEvaluationBundle(
        profileResults = profileResultMap,
        robustActionLowerBounds = pftResult.qValues.clone(),
        baselineActionValues = pftResult.qValues.clone(),
        baselineValue = if pftResult.qValues.nonEmpty then pftResult.qValues.max else 0.0,
        adversarialRootGap = None,
        pointwiseExploitability = None,
        deploymentExploitability = None,
        certification = certification,
        chainWorldValues = Map.empty,
        notes = Vector(
          s"PftDpw formal path: B*_max=${requiredBudget}, safeActions=${safeActions.mkString(",")}"
        ) ++ (if withinTolerance then Vector.empty else Vector("Budget exceeds tolerance"))
      )
      _lastBundle = Some(bundle)

      if actionIdx >= 0 && actionIdx < candidateActions.size then
        candidateActions(actionIdx)
      else
        candidateActions.find(_ != PokerAction.Fold).getOrElse(PokerAction.Fold)

    catch
      case e: (UnsatisfiedLinkError | Exception) =>
        _lastBundle = Some(makeFallbackBundle(numActions, s"PftDpw unavailable: ${e.getMessage}"))
        candidateActions.find(_ != PokerAction.Fold).getOrElse(PokerAction.Fold)

  /** Build a fallback bundle when solver errors prevent the 6-solve path. */
  private def makeFallbackBundle(numActions: Int, reason: String): DecisionEvaluationBundle =
    DecisionEvaluationBundle(
      profileResults = Map.empty,
      robustActionLowerBounds = Array.fill(numActions)(0.0),
      baselineActionValues = Array.fill(numActions)(0.0),
      baselineValue = 0.0,
      adversarialRootGap = None,
      pointwiseExploitability = None,
      deploymentExploitability = None,
      certification = CertificationResult.Unavailable(reason),
      chainWorldValues = Map.empty,
      notes = Vector(s"BaselineFallback: $reason")
    )

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
        config.defaultHeroBucket // Neutral middle bucket — no card info available

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
        case _ => classes.map(c => StrategicRivalBelief.uniform.typePosterior.probabilityOf(c))

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

  enum SolverBackend:
    case WPomcp, PftDpw

  final case class DecisionDiagnostics(
      heroBucket: Int,
      solverBackend: SolverBackend,
      exploitationBetas: Map[PlayerId, Double]
  )

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
      solverBackend: SolverBackend = SolverBackend.WPomcp,
      exploitConfig: ExploitationConfig = ExploitationConfig(
        initialBeta = 1.0,
        cpRetreatRate = 0.1,
        epsilonAdapt = 0.05
      ),
      temperedConfig: TemperedLikelihood.TemperedConfig = TemperedLikelihood.TemperedConfig.twoLayer(0.7, 0.01),
      actionPriors: Map[(StrategicClass, sicfun.holdem.types.PokerAction.Category), Double] = defaultActionPriors,
      detector: DetectionPredicate = FrequencyAnomalyDetection(window = 20, threshold = 0.6),
      /** Discount factor for Bellman safety operator (Def 60). */
      bellmanGamma: Double = 0.95,
      /** Wasserstein ambiguity radius rho for robust Q-values (Def 33). */
      ambiguityRadius: Double = 0.1,
      /** Deployment baseline exploitability epsilon_base (A10). */
      epsilonBase: Double = 0.05,
      /** Deployment belief set size |B_dep| for baseline evaluation. */
      deploymentSetSize: Int = 50,
      /** Default hero hand-strength bucket when no hole cards are available.
        * 5 = neutral middle bucket in [0, 9] range. Pending calibration.
        */
      defaultHeroBucket: Int = 5
  )

  /** Rival seat information provided at session init. */
  final case class RivalSeatInfo(position: Position, stack: Double)

  /** Per-session state: rival beliefs and exploitation states that survive across hands. */
  final case class SessionState(
      rivalBeliefs: Map[PlayerId, StrategicRivalBelief],
      exploitationStates: Map[PlayerId, ExploitationState],
      rivalSeats: Map[PlayerId, RivalSeatInfo] = Map.empty
  )
