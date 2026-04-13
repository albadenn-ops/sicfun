package sicfun.holdem.engine

import sicfun.core.DiscreteDistribution
import sicfun.holdem.types.*
import sicfun.holdem.strategic.*
import sicfun.holdem.strategic.types.*
import sicfun.holdem.strategic.state.*
import sicfun.holdem.strategic.kernel.*
import sicfun.holdem.strategic.safety.*
import sicfun.holdem.strategic.exploitation.*
import sicfun.holdem.strategic.decomposition.*
import sicfun.holdem.strategic.{bridge => strategicBridge}
import sicfun.holdem.strategic.solver.{WPomcpRuntime, PftDpwRuntime, PftDpwConfig, PftDpwResult, TabularGenerativeModel, ParticleBelief, WassersteinDroRuntime}
import sicfun.holdem.engine.inference.ActionEvaluation

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

  /** Kernel-coupled attributed baseline (Def 10). Config-only, stateless. */
  private val _attributedBaseline: PosteriorAttributedBaseline =
    new PosteriorAttributedBaseline(config.actionPriors)

  private var _sessionState: StrategicEngine.SessionState | Null = null
  private var _handActive: Boolean = false
  private var _heroCards: Option[HoleCards] = None
  private var _actionHistory: Vector[PublicAction] = Vector.empty
  private var _lastBoard: Option[Board] = None
  private var _lastStreet: Option[Street] = None
  private var _lastDiagnostics: Option[StrategicEngine.DecisionDiagnostics] = None
  private var _lastBundle: Option[DecisionEvaluationBundle] = None
  private var _lastOverlayResult: Option[OverlayResult] = None

  def lastDecisionDiagnostics: Option[StrategicEngine.DecisionDiagnostics] = _lastDiagnostics
  def lastDecisionBundle: Option[DecisionEvaluationBundle] = _lastBundle
  def lastOverlayResult: Option[OverlayResult] = _lastOverlayResult

  /** Test-only: inject a bundle to simulate solver output for testing advisory clamp / deployment tracking. */
  private[engine] def injectTestBundle(bundle: DecisionEvaluationBundle): Unit =
    _lastBundle = Some(bundle)

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
    val cpdStates = config.cpdConfig match
      case Some(cpd) => rivalIds.map(id => id -> cpd.initial).toMap
      case None => Map.empty[PlayerId, ChangepointState]
    _sessionState = StrategicEngine.SessionState(
      rivalBeliefs = beliefs,
      exploitationStates = exploitStates,
      rivalSeats = rivalSeats,
      deploymentSet = EmpiricalDeploymentSet(Vector.empty, maxSize = config.deploymentSetSize),
      cpdStates = cpdStates
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

    config.cpdConfig match
      case Some(cpd) =>
        // Dynamics.fullStepWithCPD: rival update + CPD detection + prior reset (Defs 22-28)
        val cpdConfigs = session.rivalBeliefs.keys.map { id =>
          id -> RivalCPDConfig(cpd, StrategicRivalBelief.uniform.typePosterior)
        }.toMap
        val updaters = session.rivalBeliefs.keys.map { id =>
          id -> StrategicRivalBelief.updater
        }.toMap
        val result = Dynamics.fullStepWithCPD[StrategicRivalBelief](
          rivalStates = session.rivalBeliefs,
          exploitStates = session.exploitationStates,
          signal = signal,
          publicState = pubState,
          kernelProfile = kernelProfile,
          exploitConfigs = exploitConfigs,
          detector = config.detector,
          exploitabilityFn = beta => computeExploitabilityEstimate(beta),
          epsilonNE = config.epsilonBase,
          cpdConfigs = cpdConfigs,
          cpdStates = session.cpdStates,
          posteriorExtractor = (_, m) => m.typePosterior,
          updaters = updaters,
          predictiveProbFn = (_, _) => (r: Int) => 1.0 / math.max(1, r + 1)
        )
        _sessionState = StrategicEngine.SessionState(
          rivalBeliefs = result.updatedRivals,
          exploitationStates = result.updatedExploitation,
          rivalSeats = session.rivalSeats,
          deploymentSet = session.deploymentSet,
          cpdStates = result.updatedCpdStates
        )
      case None =>
        val result = Dynamics.fullStep[StrategicRivalBelief](
          rivalStates = session.rivalBeliefs,
          exploitStates = session.exploitationStates,
          signal = signal,
          publicState = pubState,
          kernelProfile = kernelProfile,
          exploitConfigs = exploitConfigs,
          detector = config.detector,
          exploitabilityFn = beta => computeExploitabilityEstimate(beta),
          epsilonNE = config.epsilonBase
        )
        _sessionState = StrategicEngine.SessionState(
          rivalBeliefs = result.updatedRivals,
          exploitationStates = result.updatedExploitation,
          rivalSeats = session.rivalSeats,
          deploymentSet = session.deploymentSet,
          cpdStates = session.cpdStates
        )

    // Advisory Bellman clamp from last evaluation bundle (design doc §6).
    // Uses cached budget estimate as an advisory bound — NOT formal B*.
    _lastBundle match
      case Some(bundle) =>
        val budget = bundle.certification match
          case lrs: CertificationResult.LocalRobustScreening => lrs.budgetEstimate
          case tc: CertificationResult.TabularCertification => tc.requiredBudget
          case _: CertificationResult.Unavailable => Double.MaxValue
        val totalTolerance = config.epsilonBase + config.exploitConfig.epsilonAdapt
        if budget < Double.MaxValue && budget > totalTolerance then
          val updatedSession = _sessionState.nn
          val clampedExploit = updatedSession.exploitationStates.map { case (rivalId, exploitState) =>
            val retreated = math.max(0.0, exploitState.beta - config.exploitConfig.cpRetreatRate)
            rivalId -> ExploitationState(beta = retreated)
          }
          _sessionState = StrategicEngine.SessionState(
            rivalBeliefs = updatedSession.rivalBeliefs,
            exploitationStates = clampedExploit,
            rivalSeats = updatedSession.rivalSeats,
            deploymentSet = updatedSession.deploymentSet
          )
      case None => () // No bundle yet — skip clamp

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
  @deprecated("Use overlay decide(gameState, candidates, upstreamEvs) instead", "v0.33")
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

    // Wire deployment exploitability from session's prior deployment set (Def 52D).
    // Retrospective: reflects max exploitability over beliefs accumulated in prior calls.
    val currentDeploy = _sessionState.nn.deploymentSet
    if currentDeploy.entries.nonEmpty then
      _lastBundle = _lastBundle.map(b => b.copy(
        deploymentExploitability = Some(currentDeploy.deploymentExploitability)
      ))

    val bundleOpt = _lastBundle
    _lastDiagnostics = Some(StrategicEngine.DecisionDiagnostics(
      heroBucket = heroBucket,
      solverBackend = config.solverBackend,
      exploitationBetas = session.exploitationStates.map((k, v) => k -> v.beta),
      adversarialRootGap = bundleOpt.flatMap(_.adversarialRootGap),
      safeActionCount = bundleOpt.flatMap(_.certification match
        case t: CertificationResult.TabularCertification => Some(t.safeActionIndices.size)
        case _ => None
      ),
      totalActionCount = candidateActions.size,
      certificationKind = bundleOpt.map(_.certification match
        case _: CertificationResult.LocalRobustScreening => "LocalRobustScreening"
        case _: CertificationResult.TabularCertification => "TabularCertification"
        case _: CertificationResult.Unavailable => "Unavailable"
      ).getOrElse("none")
    ))

    // Record deployment snapshot for Def 52D tracking
    _lastBundle.foreach { bundle =>
      bundle.pointwiseExploitability.foreach { pwExploit =>
        val deploySession = _sessionState.nn
        val beliefs = deploySession.rivalBeliefs.values
        val avgEntropy = if beliefs.isEmpty then 0.0
          else beliefs.map { b =>
            val probs = StrategicClass.values.map(c => b.typePosterior.probabilityOf(c))
            -probs.filter(_ > 0).map(p => p * math.log(p)).sum
          }.sum / beliefs.size
        val summary = DeploymentBeliefSummary(
          beliefEntropy = avgEntropy,
          exploitabilitySnapshot = pwExploit,
          timestamp = System.currentTimeMillis()
        )
        val updatedDeploy = deploySession.deploymentSet.add(summary)
        _sessionState = deploySession.copy(deploymentSet = updatedDeploy)
      }
    }

    action

  /** Overlay decision path: filters upstream EVs through rival-model beliefs.
    *
    * This is the Phase 1 entry point. The caller obtains EVs from the adaptive
    * or multiway engine and passes them here. The overlay applies belief-weighted
    * penalties and soft veto, then returns a full OverlayResult trace.
    *
    * The old two-arg decide() is deprecated but not removed — certification
    * tests may still exercise it.
    */
  def decide(
      gameState: GameState,
      candidateActions: Vector[PokerAction],
      upstreamEvs: Vector[ActionEvaluation]
  ): OverlayResult =
    require(_sessionState != null, "Session not initialized")
    require(_handActive, "No hand in progress")
    require(candidateActions.nonEmpty, "No candidate actions")

    val session = _sessionState.nn

    // Attach robust lower bounds from certification if previously run.
    // The bundle carries per-action lower bounds directly in robustActionLowerBounds
    // (computed as min_profile Q[a] in StrategicEngine.decideWPomcp).
    // Do NOT use rootLosses — those are non-negative losses (baselineValue - lowerBound).
    val robustBounds = _lastBundle.flatMap { bundle =>
      val bounds = bundle.robustActionLowerBounds
      if bounds != null && bounds.nonEmpty then Some(bounds) else None
    }

    val input = OverlayInput(
      gameState = gameState,
      upstreamEvs = upstreamEvs,
      rivalBeliefs = session.rivalBeliefs,
      exploitationStates = session.exploitationStates,
      robustLowerBounds = robustBounds,
      config = config
    )

    val result = StrategicOverlay.filter(input)
    _lastOverlayResult = Some(result)

    // Update deployment tracking from overlay result
    val deploySession = _sessionState.nn
    val beliefs = deploySession.rivalBeliefs.values
    if beliefs.nonEmpty then
      val avgEntropy = beliefs.map { b =>
        val probs = StrategicClass.values.map(c => b.typePosterior.probabilityOf(c))
        -probs.filter(_ > 0).map(p => p * math.log(p)).sum
      }.sum / beliefs.size
      val summary = DeploymentBeliefSummary(
        beliefEntropy = avgEntropy,
        exploitabilitySnapshot = Ev(0.0), // Phase 1: no pointwise exploit from overlay
        timestamp = System.currentTimeMillis()
      )
      val updatedDeploy = deploySession.deploymentSet.add(summary)
      _sessionState = deploySession.copy(deploymentSet = updatedDeploy)

    result

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
        rivalSeats = session.rivalSeats,
        deploymentSet = session.deploymentSet
      )

    // --- Action selection from mixed-belief solve ---
    if mixed.bestAction >= 0 && mixed.bestAction < candidateActions.size then
      candidateActions(mixed.bestAction)
    else
      candidateActions.last

  /** PftDpw formal certification path.
    *
    * 1. Build mixed-belief model and particle belief from engine state.
    * 2. Solve with PftDpw native solver.
    * 3. Certify via [[buildFormalCertification]].
    *
    * Fail-closed on native error: Unavailable certification + fold.
    */
  private def decidePftDpw(
      gameState: GameState,
      candidateActions: Vector[PokerAction],
      heroBucket: Int,
      session: StrategicEngine.SessionState
  ): PokerAction =
    val numActions = candidateActions.size

    try
      // 1. Build mixed-belief (baseline) model and belief
      val baselineModel = PokerPftFormulation.buildTabularModel(
        gameState, session.rivalBeliefs, candidateActions,
        heroBucket, config.actionPriors, profileClass = None
      )
      val belief = PokerPftFormulation.buildParticleBelief(
        session.rivalBeliefs, config.particlesPerRival, currentStreet = gameState.street
      )

      // 2. Solve with PftDpw on the mixed model
      val pftConfig = PftDpwConfig(
        numSimulations = config.numSimulations,
        gamma = config.discount,
        maxDepth = config.maxDepth,
        seed = config.seed
      )
      val pftResult = try PftDpwRuntime.solve(baselineModel, belief, pftConfig) catch
        case t: Throwable =>
          _lastBundle = Some(makeFallbackBundle(numActions, s"PftDpw unavailable: ${t.getClass.getName}: ${t.getMessage}"))
          return failClosedAction(candidateActions)
      if !pftResult.isSuccess then
        _lastBundle = Some(makeFallbackBundle(numActions, s"PftDpw solver status: ${pftResult.status}"))
        return failClosedAction(candidateActions)

      // 3. Four-world grid solve + signal decomposition (Theorem 4, Defs 40-43, 47, 50)
      val (fourWorldOpt, deltaVocabOpt, chainWorldQs, riskProfileOpt): (Option[FourWorld], Option[DeltaVocabulary], Map[ChainWorld, Ev], Option[RiskDecomposition.ChainRiskProfile]) = try
        val fwModels = StrategicEngine.buildFourWorldModels(
          gameState, session.rivalBeliefs, candidateActions, heroBucket, config.actionPriors
        )
        val olResult = PftDpwRuntime.solve(fwModels.openLoop, belief, pftConfig)
        val blindResult = PftDpwRuntime.solve(fwModels.blind, belief, pftConfig)
        val blindOlResult = PftDpwRuntime.solve(fwModels.blindOpenLoop, belief, pftConfig)
        if olResult.isSuccess && blindResult.isSuccess && blindOlResult.isSuccess then
          val fw = FourWorldDecomposition.compute(
            vAttribClosedLoop = Ev(pftResult.qValues.max),
            vAttribOpenLoop = Ev(olResult.qValues.max),
            vBlindClosedLoop = Ev(blindResult.qValues.max),
            vBlindOpenLoop = Ev(blindOlResult.qValues.max)
          )

          // Ref-kernel solve for per-rival signal decomposition (Defs 40-42)
          val refResultOpt: Option[PftDpwResult] = try
            val uniformBeliefs = session.rivalBeliefs.map((id, _) => id -> StrategicRivalBelief.uniform)
            val refModel = PokerPftFormulation.buildTabularModel(
              gameState, uniformBeliefs, candidateActions, heroBucket, config.actionPriors
            )
            val r = PftDpwRuntime.solve(refModel, belief, pftConfig)
            if r.isSuccess then Some(r) else None
          catch case _: Exception => None

          // Design-kernel solve for signaling sub-decomposition (Defs 48-49)
          val designResultOpt: Option[PftDpwResult] = try
            val designModel = PokerPftFormulation.buildDesignKernelModel(
              gameState, session.rivalBeliefs, candidateActions, heroBucket, config.actionPriors
            )
            val r = PftDpwRuntime.solve(designModel, belief, pftConfig)
            if r.isSuccess then Some(r) else None
          catch case _: Exception => None

          val perRivalDeltas: Map[PlayerId, PerRivalDelta] = refResultOpt match
            case Some(refResult) =>
              session.rivalBeliefs.keys.map { rivalId =>
                rivalId -> SignalDecomposition.computePerRivalDelta(
                  qAttrib = Ev(pftResult.qValues.max),
                  qRef = Ev(refResult.qValues.max),
                  qBlind = Ev(blindResult.qValues.max)
                )
              }.toMap
            case None => Map.empty

          // Per-rival signaling sub-decomposition (Defs 48-49, Theorem 3A)
          val perRivalSubDecomps: Map[PlayerId, PerRivalSignalSubDecomposition] =
            designResultOpt match
              case Some(designResult) =>
                session.rivalBeliefs.keys.map { rivalId =>
                  rivalId -> SignalingSubDecomposition.compute(
                    qAttrib = Ev(pftResult.qValues.max),
                    qDesign = Ev(designResult.qValues.max),
                    qBlind = Ev(blindResult.qValues.max)
                  )
                }.toMap
              case None => Map.empty

          val deltaSigAgg = SignalDecomposition.deltaSigAggregate(
            qAttribAll = Ev(pftResult.qValues.max),
            qBlindAll = Ev(blindResult.qValues.max)
          )

          // Chain world Q-values (Def 47A) — mapped from grid/ref solves
          import LearningChannel.*, ShowdownMode.*
          val cwQs: Map[ChainWorld, Ev] = {
            val base = Map(
              ChainWorld(Blind, Off)  -> fw.v00,
              ChainWorld(Attrib, Off) -> fw.v10,
              ChainWorld(Attrib, On)  -> fw.v11
            )
            val withRef = refResultOpt match
              case Some(refResult) => base + (ChainWorld(Ref, Off) -> Ev(refResult.qValues.max))
              case None => base
            designResultOpt match
              case Some(designResult) => withRef + (ChainWorld(Design, Off) -> Ev(designResult.qValues.max))
              case None => withRef
          }

          // Chain edge deltas (Def 47B, Proposition 8.1)
          val chainEdgeDeltas =
            if cwQs.size >= ChainWorld.canonicalChain.size then
              ChainBaselineQ(cwQs).canonicalEdgeDeltas
            else IndexedSeq.empty[ChainEdgeDelta]

          // Chain risk profile (Defs 67-69, Proposition 9.7)
          val chainForRisk = ChainWorld.canonicalChain.filter(cwQs.contains)
          val chainRiskProfile = if chainForRisk.size >= 2 then
            val baselineEvs = IndexedSeq(Ev(pftResult.qValues.max))
            val qsByWorld = chainForRisk.map(w => IndexedSeq(cwQs(w)))
            val profile = RiskDecomposition.computeProfile(chainForRisk, baselineEvs, qsByWorld)
            val riskDeltas = profile.riskIncrements
            val efficiencies = if chainEdgeDeltas.nonEmpty && riskDeltas.size == chainEdgeDeltas.size then
              RiskDecomposition.edgeEfficiencies(chainEdgeDeltas, riskDeltas)
            else IndexedSeq.empty
            Some(profile.copy() -> (riskDeltas, efficiencies))
          else None

          val vocab = FourWorldDecomposition.buildDeltaVocabulary(
            fourWorld = fw,
            perRivalDeltas = perRivalDeltas,
            deltaSigAggregate = deltaSigAgg,
            perRivalSubDecomps = perRivalSubDecomps
          ).copy(
            chainEdgeDeltas = chainEdgeDeltas,
            chainRiskDeltas = chainRiskProfile.map(_._2._1).getOrElse(IndexedSeq.empty),
            edgeEfficiencies = chainRiskProfile.map(_._2._2).getOrElse(IndexedSeq.empty)
          )

          (Some(fw), Some(vocab), cwQs, chainRiskProfile.map(_._1))
        else (Option.empty[FourWorld], Option.empty[DeltaVocabulary], Map.empty[ChainWorld, Ev], Option.empty[RiskDecomposition.ChainRiskProfile])
      catch
        case _: Exception => (Option.empty[FourWorld], Option.empty[DeltaVocabulary], Map.empty[ChainWorld, Ev], Option.empty[RiskDecomposition.ChainRiskProfile])

      // 4. Build profile models and certify
      val profileModels = (0 until StrategicClass.values.length).map { p =>
        PokerPftFormulation.buildTabularModel(
          gameState, session.rivalBeliefs, candidateActions,
          heroBucket, config.actionPriors, profileClass = Some(StrategicClass.fromOrdinal(p))
        )
      }

      val (action, bundle) = buildFormalCertification(
        baselineModel, profileModels, belief, pftResult,
        candidateActions, config.bellmanGamma,
        config.epsilonBase, config.exploitConfig.epsilonAdapt,
        rootState = gameState.street.ordinal,
        fourWorld = fourWorldOpt,
        deltaVocabulary = deltaVocabOpt,
        chainWorldValues = chainWorldQs,
        ambiguityRadius = config.ambiguityRadius
      )

      // 5. Bluff annotations (Defs 35-39)
      val heroClass = StrategicEngine.estimateHeroClass(heroBucket)
      val annotations = candidateActions.zipWithIndex.map { (act, idx) =>
        val isStructural = BluffFramework.isStructuralBluff(heroClass, act)
        val gain = if isStructural && idx < bundle.baselineActionValues.length then
          val nonBluffQs = candidateActions.zipWithIndex
            .filterNot((a, _) => BluffFramework.isStructuralBluff(heroClass, a))
            .collect { case (_, i) if i < bundle.baselineActionValues.length => bundle.baselineActionValues(i) }
          val bestNonBluff = if nonBluffQs.nonEmpty then Ev(nonBluffQs.max) else Ev(0.0)
          Some(BluffFramework.bluffGain(Ev(bundle.baselineActionValues(idx)), bestNonBluff))
        else None
        BluffAnnotation(isStructural, gain, BluffFramework.isExploitativeBluff(heroClass, act, gain.getOrElse(Ev.Zero)))
      }
      // 6. SpotPolarization (Def 25, A9) — per-action information disclosure
      val pubState = bridgePublicState(gameState)
      val polProfile: Map[Int, Double] = session.rivalBeliefs.values.headOption match
        case Some(rivalBelief) =>
          val polarization = new PosteriorDivergencePolarization(
            rivalBelief.typePosterior,
            Some(buildAttribLikelihoodFn())
          )
          candidateActions.zipWithIndex.collect {
            case (PokerAction.Raise(amount), idx) =>
              val sizing = Sizing(
                Chips(amount),
                PotFraction(if gameState.pot > 0 then amount / gameState.pot else 1.0)
              )
              idx -> polarization.polarization(
                PokerAction.Category.Raise, sizing, pubState, rivalBelief
              )
          }.toMap
        case None => Map.empty

      // 7. RevealSchedule (Def 51) — hero information disclosure decision
      val revealDec = config.revealSchedule.flatMap { schedule =>
        val equity = Ev(heroBucket / 9.0)
        session.rivalBeliefs.keys.headOption.map { rivalId =>
          schedule.classify(rivalId, gameState.street, equity)
        }
      }

      // 8. OperationalBaseline (A10) — compose from config + deployment set
      val opBaseline = Some(OperationalBaseline(
        epsilonBase = config.epsilonBase,
        deploymentSet = session.deploymentSet,
        description = s"PftDpw formal path, ${session.deploymentSet.entries.size} deployment entries"
      ))

      val annotatedBundle = bundle.copy(
        bluffAnnotations = annotations,
        chainRiskProfile = riskProfileOpt,
        polarizationProfile = polProfile,
        revealDecision = revealDec,
        operationalBaseline = opBaseline
      )

      _lastBundle = Some(annotatedBundle)
      action

    catch
      case e: UnsatisfiedLinkError =>
        _lastBundle = Some(makeFallbackBundle(numActions, s"PftDpw unavailable: ${e.getMessage}"))
        failClosedAction(candidateActions)
      case e: Exception =>
        _lastBundle = Some(makeFallbackBundle(numActions, s"PftDpw unavailable: ${e.getMessage}"))
        failClosedAction(candidateActions)

  /** Most conservative action: fold if available, otherwise first candidate. */
  private[engine] def failClosedAction(candidateActions: Vector[PokerAction]): PokerAction =
    candidateActions.find(_ == PokerAction.Fold).getOrElse(candidateActions.head)

  /** Post-solve certification logic, extracted for testability.
    *
    * Given a solver result and profile-conditioned models:
    * 1. Evaluate per-profile values under the reference policy.
    * 2. Compute baseline action values and robust lower bounds.
    * 3. Compute robust losses, B*, and belief-level safe actions.
    * 4. Validate certificate.
    * 5. Certified path: safe-feasible action from solver Q-values.
    *    Fail-closed path: reference policy action (pftResult.bestAction).
    *
    * @return (chosen action, bundle)
    */
  private[engine] def buildFormalCertification(
      baselineModel: TabularGenerativeModel,
      profileModels: IndexedSeq[TabularGenerativeModel],
      belief: ParticleBelief,
      pftResult: PftDpwResult,
      candidateActions: Vector[PokerAction],
      gamma: Double,
      epsilonBase: Double,
      epsilonAdapt: Double,
      rootState: Int = 0,
      fourWorld: Option[FourWorld] = None,
      deltaVocabulary: Option[DeltaVocabulary] = None,
      chainWorldValues: Map[ChainWorld, Ev] = Map.empty,
      ambiguityRadius: Double = 0.0
  ): (PokerAction, DecisionEvaluationBundle) =
    val numActions = candidateActions.size
    val numProfiles = profileModels.size

    // Evaluate per-profile values under the reference policy
    val refPolicy: Int => Int = _ => pftResult.bestAction
    val baselineValues = PerStateLossEvaluator.valueIteration(baselineModel, refPolicy, gamma)
    val profileValueArrays = profileModels.map(m =>
      PerStateLossEvaluator.valueIteration(m, refPolicy, gamma)
    )

    // Baseline action values: Q^π(rootState, a) under mixed model
    val baselineActionValues = new Array[Double](numActions)
    var a = 0
    while a < numActions do
      val idx = rootState * numActions + a
      val reward = baselineModel.rewardTable(idx)
      val successor = baselineModel.transitionTable(idx)
      baselineActionValues(a) = reward + gamma * baselineValues(successor)
      a += 1
    val baselineValue = if baselineValues.length > rootState then baselineValues(rootState) else 0.0

    // Robust action lower bounds: Wasserstein DRO over profile space (Def 34)
    val robustActionLowerBounds = new Array[Double](numActions)
    if ambiguityRadius > 0.0 && numProfiles > 1 then
      val profileBelief = Array.fill(numProfiles)(1.0 / numProfiles)
      val profileCostMatrix = Array.tabulate(numProfiles * numProfiles) { idx =>
        if idx / numProfiles == idx % numProfiles then 0.0 else 1.0
      }
      a = 0
      while a < numActions do
        val qPerProfile = new Array[Double](numProfiles)
        var p = 0
        while p < numProfiles do
          val model = profileModels(p)
          val idx = rootState * model.numActions + a
          val reward = model.rewardTable(idx)
          val successor = model.transitionTable(idx)
          qPerProfile(p) = reward + gamma * profileValueArrays(p)(successor)
          p += 1
        WassersteinDroRuntime.robustQValue(profileBelief, qPerProfile, ambiguityRadius, profileCostMatrix) match
          case Right(robustQ) => robustActionLowerBounds(a) = robustQ
          case Left(_) => robustActionLowerBounds(a) = qPerProfile.min
        a += 1
    else
      a = 0
      while a < numActions do
        var minQ = Double.PositiveInfinity
        var p = 0
        while p < numProfiles do
          val model = profileModels(p)
          val idx = rootState * model.numActions + a
          val reward = model.rewardTable(idx)
          val successor = model.transitionTable(idx)
          val qsa = reward + gamma * profileValueArrays(p)(successor)
          if qsa < minQ then minQ = qsa
          p += 1
        robustActionLowerBounds(a) = minQ
        a += 1

    // Per-profile results with actual per-profile Q-values at rootState
    val profileResultMap: Map[JointRivalProfileId, SolverResult] =
      (0 until numProfiles).map { p =>
        val model = profileModels(p)
        val profileQ = new Array[Double](numActions)
        var ai = 0
        while ai < numActions do
          val idx = rootState * numActions + ai
          val reward = model.rewardTable(idx)
          val successor = model.transitionTable(idx)
          profileQ(ai) = reward + gamma * profileValueArrays(p)(successor)
          ai += 1
        val bestA = profileQ.indices.maxBy(profileQ(_))
        JointRivalProfileId(p) -> SolverResult(bestAction = bestA, actionValues = profileQ)
      }.toMap

    // Compute robust losses from profile models
    val robustLosses = PerStateLossEvaluator.computeRobustLosses(profileModels, refPolicy, gamma)

    // Build transitions function from profile models (deterministic wrapper for Def 60)
    val transitions: (Int, Int, Int) => IndexedSeq[(Int, Double)] = SafetyBellman.deterministicTransition(
      (s, a, p) => profileModels(p).transitionTable(s * baselineModel.numActions + a)
    )

    // Compute B*
    val bStar = SafetyBellman.computeBStar(robustLosses, gamma, transitions, numProfiles)

    // Belief-level safe action set
    val safeActions = SafetyBellman.beliefLevelSafeActions(
      belief.weights, bStar, robustLosses, gamma, transitions, numProfiles
    )

    // Certificate validation
    val requiredBudget = SafetyBellman.requiredAdaptationBudget(bStar)
    val totalTolerance = epsilonBase + epsilonAdapt
    val withinTolerance = requiredBudget <= totalTolerance
    val cert = SafetyBellman.Certificate(
      values = bStar.clone(), terminalStates = Set(baselineModel.numStates - 1)
    )
    val certificateValid = cert.isValid(
      robustLosses, gamma, requiredBudget + 1.0, transitions, numProfiles
    )

    val certification = CertificationResult.TabularCertification(
      bStar = bStar,
      requiredBudget = requiredBudget,
      safeActionIndices = safeActions,
      certificateValid = certificateValid,
      withinTolerance = withinTolerance
    )

    // Adversarial root gap
    var minProfileBestValue = Double.PositiveInfinity
    var p = 0
    while p < numProfiles do
      val profBest = profileValueArrays(p)(rootState)
      if profBest < minProfileBestValue then minProfileBestValue = profBest
      p += 1
    val adversarialRootGap = baselineValue - minProfileBestValue

    // Pointwise exploitability (Def 52C):
    // eps(b; pi) = V^sec_optimal(b) - V^sec_actual(b)
    // V^sec_actual = min over profiles of root value under each profile
    // V^sec_optimal approximated by baselineValue (conservative: baseline ≈ near-optimal)
    val securityValueActual = Ev(minProfileBestValue)
    val securityValueOptimal = Ev(baselineValue)
    val pwExploit = PointwiseExploitability.compute(securityValueOptimal, securityValueActual)

    // Action selection: certified path vs fail-closed
    val chosenActionIdx = if certificateValid && withinTolerance && safeActions.nonEmpty then
      // Certified: highest Q among safe actions
      SafetyBellman.safeFeasibleAction(pftResult.qValues, safeActions)
    else
      // Fail-closed: reference policy action (no policy improvement).
      // Covers: invalid certificate, budget exceeds tolerance, OR empty safe set
      // at belief level (belief-lifted approximation can produce empty sets even
      // when the latent-state certificate validates).
      pftResult.bestAction

    val bundle = DecisionEvaluationBundle(
      profileResults = profileResultMap,
      robustActionLowerBounds = robustActionLowerBounds,
      baselineActionValues = baselineActionValues,
      baselineValue = baselineValue,
      adversarialRootGap = Some(Ev(adversarialRootGap)),
      pointwiseExploitability = Some(pwExploit),
      deploymentExploitability = None,
      certification = certification,
      chainWorldValues = chainWorldValues,
      fourWorld = fourWorld,
      deltaVocabulary = deltaVocabulary,
      notes = Vector(
        s"PftDpw formal path: B*_max=$requiredBudget, safeActions=${safeActions.mkString(",")}"
      ) ++ (if !certificateValid then Vector("Certificate invalid — fail-closed to reference policy")
            else if !withinTolerance then Vector("Budget exceeds tolerance — fail-closed to reference policy")
            else if safeActions.isEmpty then Vector("Empty belief-level safe set — fail-closed to reference policy")
            else Vector.empty)
    )

    val action = if chosenActionIdx >= 0 && chosenActionIdx < candidateActions.size then
      candidateActions(chosenActionIdx)
    else
      failClosedAction(candidateActions)

    (action, bundle)

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

  /** Build a StrategicSnapshot from the last decision bundle and current session state.
    *
    * Populates v0.31.1 optional fields (gridWorldValues, securityValue,
    * safetyCertificateSummary) from the bundle's certification data.
    *
    * @return None if no decision bundle is available
    */
  def buildSnapshot(gameState: GameState, heroAction: PokerAction): Option[strategicBridge.StrategicSnapshot] =
    _lastBundle.map { bundle =>
      val street = gameState.street
      val heroBucket = estimateHeroBucket(gameState)
      val heroClass = StrategicEngine.estimateHeroClass(heroBucket)
      val baseline = Ev(bundle.baselineValue)
      val fw = bundle.fourWorld.getOrElse(
        FourWorld(v11 = baseline, v10 = baseline, v01 = baseline, v00 = baseline)
      )

      val session = _sessionState.nn
      val opponentPosterior = session.rivalBeliefs.values.headOption.map(_.typePosterior)

      // v0.31.1 optional fields from certification data
      val gridWorldValues = bundle.fourWorld.map { fwv =>
        GridWorld.all.map(gw => gw -> BridgeResult.Exact(fwv(gw))).toMap
      }
      val securityValue = bundle.certification match
        case CertificationResult.TabularCertification(bStar, _, _, _, _) =>
          if bStar.nonEmpty then Some(Ev(bStar.min)) else None
        case _ => None
      val safetyCertSummary = bundle.certification match
        case CertificationResult.TabularCertification(_, budget, _, valid, _) =>
          Some((budget, valid))
        case _ => None

      // Reputation views for all rivals (ReputationalProjection)
      val reputationViews = session.rivalBeliefs.map { case (id, belief) =>
        id -> StrategicEngine.PosteriorReputationalProjection.project(belief)
      }

      strategicBridge.StrategicSnapshot(
        street = street,
        pot = Chips(gameState.pot),
        heroStack = Chips(gameState.stackSize),
        toCall = Chips(gameState.toCall),
        actionSignal = bridgeActionSignal(heroAction, gameState),
        strategicClass = heroClass,
        fourWorld = fw,
        baseline = baseline,
        opponentClassPosterior = opponentPosterior,
        gridWorldValues = gridWorldValues,
        securityValue = securityValue,
        safetyCertificateSummary = safetyCertSummary,
        reputationViews = reputationViews,
        bridgeFidelityNotes = Vector("snapshot from StrategicEngine decision bundle"),
        attributionEnabled = true
      )
    }

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
        rivalSeats = session.rivalSeats,
        deploymentSet = session.deploymentSet
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

  /** Build an attrib likelihood from an AttributedBaseline (Def 18 spec-literal).
    *
    * Transposes from action-space hat_pi(a | c, ...) to class-space posterior
    * via TemperedLikelihood.updatePosterior.
    */
  private def buildAttribLikelihoodFromBaseline(baseline: AttributedBaseline): TemperedLikelihoodFn =
    (signal: ActionSignal, pubState: PublicState, rivalState: RivalBeliefState) =>
      val classes = StrategicClass.values
      val eta = TemperedLikelihood.defaultEta(classes.length)

      val basePr = classes.map { cls =>
        baseline.probability(cls, signal.action, signal.sizing, pubState, rivalState)
      }

      val prior = rivalState match
        case srb: StrategicRivalBelief => classes.map(c => srb.typePosterior.probabilityOf(c))
        case _ => classes.map(c => StrategicRivalBelief.uniform.typePosterior.probabilityOf(c))

      val posterior = TemperedLikelihood.updatePosterior(prior, basePr, eta, config.temperedConfig)
      DiscreteDistribution(classes.zip(posterior).toMap)

  /** Build the attrib tempered likelihood function (Def 18: hat{pi}^{0,S,i}).
    * Thin wrapper: delegates to buildAttribLikelihoodFromBaseline using the engine's
    * PosteriorAttributedBaseline instance.
    */
  private def buildAttribLikelihoodFn(): TemperedLikelihoodFn =
    buildAttribLikelihoodFromBaseline(_attributedBaseline)

  /** Build the ref tempered likelihood function (Def 18: pi^{0,S}).
    * Does NOT condition on rival state — uses uniform prior for all rivals.
    */
  private def buildRefLikelihoodFn(): TemperedLikelihoodFn =
    (signal: ActionSignal, pubState: PublicState, rivalState: RivalBeliefState) =>
      val classes = StrategicClass.values
      val eta = TemperedLikelihood.defaultEta(classes.length)

      val basePr = classes.map { cls =>
        actionPrior(cls, signal.action)
      }

      // Ref kernel (Def 18): uniform prior, ignores rival-specific history
      val uniformPrior = classes.map(_ => 1.0 / classes.length)

      val posterior = TemperedLikelihood.updatePosterior(uniformPrior, basePr, eta, config.temperedConfig)
      DiscreteDistribution(classes.zip(posterior).toMap)

  /** Build the joint kernel profile for all rivals.
    *
    * Per Def 18, uses distinct Ref and Attrib likelihoods interpolated by
    * per-rival beta (Def 15C) via ExploitationInterpolation.buildInterpolatedKernelFull.
    */
  private def buildKernelProfile(): JointKernelProfile[StrategicRivalBelief] =
    val refLikelihood = buildRefLikelihoodFn()
    val attribLikelihood = buildAttribLikelihoodFn()
    val showdownKernel = makeShowdownKernel(
      _lastBoard.getOrElse(Board.empty),
      _lastStreet.getOrElse(Street.Preflop),
      _actionHistory.lastOption.map(_.signal.action)
    )
    val session = _sessionState.nn
    JointKernelProfile(
      session.rivalBeliefs.keys.map { id =>
        val beta = session.exploitationStates.get(id).map(_.beta).getOrElse(1.0)
        val interpolatedKernel = ExploitationInterpolation.buildInterpolatedKernelFull[StrategicRivalBelief](
          StrategicRivalBelief.updater,
          refLikelihood,
          attribLikelihood,
          beta
        )
        val fullKernel = KernelConstructor.composeFullKernelFromFull(interpolatedKernel, showdownKernel)
        id -> fullKernel
      }.toMap
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
    if cards.size < 2 then return StrategicClass.Mixed
    val holeCards = HoleCards.from(cards.take(2))
    val strength = HandStrengthEstimator.fastGtoStrength(holeCards, board, street)
    val wasAggressive = lastAction.exists(_ == PokerAction.Category.Raise)
    if strength >= 0.65 then
      StrategicClass.Value
    else if strength < 0.35 && wasAggressive then
      StrategicClass.Bluff
    else if strength >= 0.35 && strength < 0.55 && wasAggressive then
      StrategicClass.StructuralBluff
    else
      StrategicClass.Mixed

object StrategicEngine:

  enum SolverBackend:
    case WPomcp, PftDpw

  final case class DecisionDiagnostics(
      heroBucket: Int,
      solverBackend: SolverBackend,
      exploitationBetas: Map[PlayerId, Double],
      adversarialRootGap: Option[Ev] = None,
      safeActionCount: Option[Int] = None,
      totalActionCount: Int = 0,
      certificationKind: String = "none",
      assumptionSummary: String = AssumptionManifest.summary,
      changepointDetected: Set[PlayerId] = Set.empty
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
      (StrategicClass.StructuralBluff, Fold) -> 0.05, (StrategicClass.StructuralBluff, Check) -> 0.15,
      (StrategicClass.StructuralBluff, Call) -> 0.30, (StrategicClass.StructuralBluff, Raise) -> 0.50,
      (StrategicClass.Mixed, Fold) -> 0.15, (StrategicClass.Mixed, Check) -> 0.40,
      (StrategicClass.Mixed, Call) -> 0.35, (StrategicClass.Mixed, Raise) -> 0.10
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
      defaultHeroBucket: Int = 5,
      /** Optional changepoint detector for rival belief dynamics (Defs 26-28).
        * When set, observeAction uses Dynamics.fullStepWithCPD instead of fullStep.
        */
      cpdConfig: Option[ChangepointDetector] = None,
      /** Optional reveal schedule for hero information disclosure (Def 51). */
      revealSchedule: Option[RevealSchedule] = None
  )

  /** Rival seat information provided at session init. */
  final case class RivalSeatInfo(position: Position, stack: Double)

  /** Per-session state: rival beliefs and exploitation states that survive across hands. */
  final case class SessionState(
      rivalBeliefs: Map[PlayerId, StrategicRivalBelief],
      exploitationStates: Map[PlayerId, ExploitationState],
      rivalSeats: Map[PlayerId, RivalSeatInfo] = Map.empty,
      deploymentSet: EmpiricalDeploymentSet = EmpiricalDeploymentSet(Vector.empty, maxSize = 50),
      cpdStates: Map[PlayerId, ChangepointState] = Map.empty
  )

  /** Four tabular models for the four-world grid solve (Theorem 4). */
  final case class FourWorldModels(
      baseline: TabularGenerativeModel,      // V^{1,1}: attrib kernel, closed-loop
      openLoop: TabularGenerativeModel,      // V^{1,0}: attrib kernel, open-loop
      blind: TabularGenerativeModel,         // V^{0,1}: blind kernel, closed-loop
      blindOpenLoop: TabularGenerativeModel  // V^{0,0}: blind kernel, open-loop
  ):
    def size: Int = 4

  /** Build the four tabular models for the four-world grid (Theorem 4). */
  def buildFourWorldModels(
      gameState: GameState,
      rivalBeliefs: Map[PlayerId, StrategicRivalBelief],
      heroActions: Vector[PokerAction],
      heroBucket: Int,
      actionPriors: Map[(StrategicClass, PokerAction.Category), Double]
  ): FourWorldModels =
    FourWorldModels(
      baseline = PokerPftFormulation.buildTabularModel(
        gameState, rivalBeliefs, heroActions, heroBucket, actionPriors
      ),
      openLoop = PokerPftFormulation.buildOpenLoopModel(
        gameState, rivalBeliefs, heroActions, heroBucket, actionPriors
      ),
      blind = PokerPftFormulation.buildBlindKernelModel(
        gameState, rivalBeliefs, heroActions, heroBucket, actionPriors
      ),
      blindOpenLoop = PokerPftFormulation.buildBlindOpenLoopModel(
        gameState, rivalBeliefs, heroActions, heroBucket, actionPriors
      )
    )

  /** Extract FourWorld values from solver Q-value arrays (Theorem 4).
    *
    * Each Q-value array is per-action at the root state.
    * The grid world value is the max Q-value (best action) for each model.
    * All four values come from the same solver framework, so the
    * algebraic identity V^{1,1} = V^{0,0} + Delta_cont + Delta_sig* + Delta_int holds.
    */
  def extractFourWorldValues(
      baselineQ: Array[Double],
      openLoopQ: Array[Double],
      blindQ: Array[Double],
      blindOpenLoopQ: Array[Double]
  ): FourWorld =
    FourWorld(
      v11 = Ev(baselineQ.max),
      v10 = Ev(openLoopQ.max),
      v01 = Ev(blindQ.max),
      v00 = Ev(blindOpenLoopQ.max)
    )

  /** Concrete ReputationalProjection that derives reputation from type posterior.
    *
    * Maps strategic class probabilities to behavioral dimensions:
    * - perceivedTightness: P(Value) (value players are selective)
    * - perceivedAggression: P(Bluff) + P(StructuralBluff) (aggressive wager tendency)
    * - perceivedBluffFrequency: P(Bluff) (pure bluff frequency)
    */
  private[engine] object PosteriorReputationalProjection extends ReputationalProjection:
    def project(rivalState: RivalBeliefState): ReputationView =
      rivalState match
        case srb: StrategicRivalBelief =>
          val pValue = srb.typePosterior.probabilityOf(StrategicClass.Value)
          val pBluff = srb.typePosterior.probabilityOf(StrategicClass.Bluff)
          val pStructBluff = srb.typePosterior.probabilityOf(StrategicClass.StructuralBluff)
          val pMixed = srb.typePosterior.probabilityOf(StrategicClass.Mixed)
          ReputationView(
            perceivedTightness = PotFraction(pValue),
            perceivedAggression = PotFraction(pBluff + pStructBluff),
            perceivedBluffFrequency = PotFraction(pBluff),
            raw = Map(
              "pValue" -> pValue, "pBluff" -> pBluff,
              "pStructBluff" -> pStructBluff, "pMixed" -> pMixed
            )
          )
        case _ =>
          ReputationView(
            perceivedTightness = PotFraction(0.25),
            perceivedAggression = PotFraction(0.25),
            perceivedBluffFrequency = PotFraction(0.25)
          )

  /** Estimate hero's strategic class from hand-strength bucket (Def 35 support).
    * Maps [0,9] bucket to the most likely class for bluff annotation.
    */
  private[engine] def estimateHeroClass(heroBucket: Int): StrategicClass =
    if heroBucket >= 7 then StrategicClass.Value
    else if heroBucket <= 2 then StrategicClass.Bluff
    else if heroBucket <= 4 then StrategicClass.StructuralBluff
    else StrategicClass.Mixed
