package sicfun.holdem.strategic

import sicfun.core.DiscreteDistribution

/** Result of a full dynamics step.
  *
  * Contains updated rival states and exploitation states for all rivals.
  */
final case class DynamicsStepResult[M <: RivalBeliefState](
    updatedRivals: Map[PlayerId, M],
    updatedExploitation: Map[PlayerId, ExploitationState]
)

/** Extended result that also carries changepoint detection state (Defs 26-28). */
final case class DynamicsStepWithCPDResult[M <: RivalBeliefState](
    updatedRivals: Map[PlayerId, M],
    updatedExploitation: Map[PlayerId, ExploitationState],
    updatedCpdStates: Map[PlayerId, ChangepointState],
    changepointDetected: Set[PlayerId]
)

/** Per-rival changepoint detection configuration bundle. */
final case class RivalCPDConfig(
    detector: ChangepointDetector,
    metaPrior: DiscreteDistribution[StrategicClass]
)

/** Belief and rival-state dynamics (Defs 22-28). */
object Dynamics:

  /** Belief update (Def 22).
    *
    * b_{t+1} = tau_tilde(b_t, u_t, Z_{t+1})
    *
    * The belief update is parameterized by an updater function that
    * transforms the operative belief given the observed signal and
    * public state. This is abstract because the specific Bayesian
    * update depends on the state representation.
    */
  def beliefUpdate(
      belief: OperativeBelief,
      signal: TotalSignal,
      publicState: PublicState,
      updater: (OperativeBelief, TotalSignal, PublicState) => OperativeBelief
  ): OperativeBelief =
    updater(belief, signal, publicState)

  /** Full rival-state update (Def 23).
    *
    * m_{t+1}^{R,i} = Gamma^{full,bullet,i}(m_t^{R,i}, Y_t, x_t^pub)
    *
    * Applies the kernel profile to update all rival states.
    * The bullet (variant) is determined by which profile is passed in.
    * Rivals without a kernel in the profile are preserved unchanged.
    */
  def fullRivalUpdate[M <: RivalBeliefState](
      rivalStates: Map[PlayerId, M],
      signal: TotalSignal,
      publicState: PublicState,
      profile: JointKernelProfile[M]
  ): Map[PlayerId, M] =
    rivalStates.map { case (id, state) =>
      profile.kernels.get(id) match
        case Some(kernel) =>
          id -> kernel.apply(state, signal, publicState).asInstanceOf[M]
        case None =>
          id -> state
    }

  /** Full rival-state update under a specific chain world (Def 23, world-aware overload).
    *
    * Extracts the appropriate JointKernelProfile from a WorldIndexedKernelProfile
    * for the given ChainWorld and applies it.
    */
  def fullRivalUpdate[M <: RivalBeliefState](
      rivalStates: Map[PlayerId, M],
      signal: TotalSignal,
      publicState: PublicState,
      worldProfile: WorldIndexedKernelProfile[M],
      world: ChainWorld
  ): Map[PlayerId, M] =
    fullRivalUpdate(rivalStates, signal, publicState, worldProfile.forWorld(world))

  /** Counterfactual reference world (Def 24).
    *
    * The non-manipulative counterfactual world is the joint reference
    * profile Gamma^{ref}. The symbol pi_t^{cf,S} is retired.
    *
    * This is simply fullRivalUpdate with the reference profile.
    */
  def counterfactualReferenceWorld[M <: RivalBeliefState](
      rivalStates: Map[PlayerId, M],
      signal: TotalSignal,
      publicState: PublicState,
      refProfile: JointKernelProfile[M]
  ): Map[PlayerId, M] =
    fullRivalUpdate(rivalStates, signal, publicState, refProfile)

  /** Counterfactual reference world with explicit ChainWorld (Def 24, world-aware overload).
    *
    * Uses ChainWorld(Ref, Off) or ChainWorld(Ref, On) to select the correct
    * reference kernel from a WorldIndexedKernelProfile.
    */
  def counterfactualReferenceWorld[M <: RivalBeliefState](
      rivalStates: Map[PlayerId, M],
      signal: TotalSignal,
      publicState: PublicState,
      worldProfile: WorldIndexedKernelProfile[M],
      world: ChainWorld
  ): Map[PlayerId, M] =
    fullRivalUpdate(rivalStates, signal, publicState, worldProfile, world)

  /** Full dynamics step: rival update + exploitation update.
    *
    * Combines:
    * - Def 23: rival state update via kernel profile
    * - Def 15C: exploitation interpolation + detection retreat + safety clamp
    *
    * This is the main entry point for a single time step of the dynamics.
    */
  def fullStep[M <: RivalBeliefState](
      rivalStates: Map[PlayerId, M],
      exploitStates: Map[PlayerId, ExploitationState],
      signal: TotalSignal,
      publicState: PublicState,
      kernelProfile: JointKernelProfile[M],
      exploitConfigs: Map[PlayerId, ExploitationConfig],
      detector: DetectionPredicate,
      exploitabilityFn: Double => Double,
      epsilonNE: Double
  ): DynamicsStepResult[M] =
    // Step 1: Update rival states via kernel profile (Def 23)
    val updatedRivals = fullRivalUpdate(rivalStates, signal, publicState, kernelProfile)

    // Step 2: Update exploitation states (Def 15C + A6')
    val updatedExploit = exploitStates.map { case (rivalId, state) =>
      val config = exploitConfigs.getOrElse(rivalId,
        ExploitationConfig(initialBeta = 1.0, cpRetreatRate = 0.0, epsilonAdapt = Double.MaxValue)
      )
      val updated = ExploitationInterpolation.updateExploitation(
        state = state,
        config = config,
        rivalId = rivalId,
        history = publicState.actionHistory,
        publicState = publicState,
        detector = detector,
        exploitabilityFn = exploitabilityFn,
        epsilonNE = epsilonNE
      )
      rivalId -> updated
    }

    DynamicsStepResult(updatedRivals, updatedExploit)

  // ---------------------------------------------------------------------------
  // Changepoint detection integration (Defs 26-28)
  // ---------------------------------------------------------------------------

  /** Update a single rival's changepoint state (Def 27: run-length posterior update).
    *
    * @return (updated CPD state, whether changepoint was detected)
    */
  def changepointUpdate(
      detector: ChangepointDetector,
      state: ChangepointState,
      predictiveProb: Int => Double
  ): (ChangepointState, Boolean) =
    val updated = detector.update(state, predictiveProb)
    val detected = detector.isChangepointDetected(updated)
    (updated, detected)

  /** Apply changepoint-triggered prior reset for a single rival (Def 28).
    *
    * mu^{R,i} <- (1 - w_reset) * mu^{R,i} + w_reset * nu_meta^i
    *
    * Uses the StateEmbeddingUpdater (Def 16) to inject the reset posterior
    * back into the rival belief state.
    */
  def changepointReset[M <: RivalBeliefState](
      rivalState: M,
      detector: ChangepointDetector,
      currentPosterior: DiscreteDistribution[StrategicClass],
      metaPrior: DiscreteDistribution[StrategicClass],
      updater: StateEmbeddingUpdater[M]
  ): M =
    val resetPosterior = detector.resetPrior(currentPosterior, metaPrior)
    updater(rivalState, resetPosterior)

  /** Full dynamics step with changepoint detection (Defs 22-28 + Def 15C).
    *
    * Execution order per the spec:
    *   1. For each rival: update CPD state (Def 27)
    *   2. If changepoint detected: reset prior (Def 28) + retreat beta (Def 15C)
    *   3. Update rival states via kernel profile (Def 23)
    *   4. Detect modeling (A6') and retreat beta if triggered
    *   5. Clamp beta for safety (Def 15C)
    *
    * @param posteriorExtractor extracts the current class posterior from a rival state
    * @param updaters per-rival StateEmbeddingUpdater (Def 16) for injecting reset posteriors
    * @param predictiveProbFn per-rival predictive probability function for CPD run-length update
    */
  def fullStepWithCPD[M <: RivalBeliefState](
      rivalStates: Map[PlayerId, M],
      exploitStates: Map[PlayerId, ExploitationState],
      signal: TotalSignal,
      publicState: PublicState,
      kernelProfile: JointKernelProfile[M],
      exploitConfigs: Map[PlayerId, ExploitationConfig],
      detector: DetectionPredicate,
      exploitabilityFn: Double => Double,
      epsilonNE: Double,
      cpdConfigs: Map[PlayerId, RivalCPDConfig],
      cpdStates: Map[PlayerId, ChangepointState],
      posteriorExtractor: (PlayerId, M) => DiscreteDistribution[StrategicClass],
      updaters: Map[PlayerId, StateEmbeddingUpdater[M]],
      predictiveProbFn: (PlayerId, M) => (Int => Double)
  ): DynamicsStepWithCPDResult[M] =

    // Step 1-2: CPD update + conditional prior reset and beta retreat
    var currentRivals = rivalStates
    var currentExploit = exploitStates
    var newCpdStates = cpdStates
    var detected = Set.empty[PlayerId]

    for (rivalId, cpdConfig) <- cpdConfigs do
      val state = cpdStates.getOrElse(rivalId, cpdConfig.detector.initial)
      val rivalState = currentRivals.getOrElse(rivalId, null.asInstanceOf[M])
      if rivalState != null then
        val predProb = predictiveProbFn(rivalId, rivalState)
        val (updatedCpd, isDetected) = changepointUpdate(cpdConfig.detector, state, predProb)
        newCpdStates = newCpdStates.updated(rivalId, updatedCpd)

        if isDetected then
          detected = detected + rivalId

          // Def 28: reset prior
          updaters.get(rivalId).foreach { updater =>
            val currentPost = posteriorExtractor(rivalId, rivalState)
            val resetState = changepointReset(
              rivalState, cpdConfig.detector, currentPost, cpdConfig.metaPrior, updater
            )
            currentRivals = currentRivals.updated(rivalId, resetState)
          }

          // Def 15C: changepoint-triggered beta retreat
          currentExploit.get(rivalId).foreach { es =>
            val config = exploitConfigs.getOrElse(rivalId,
              ExploitationConfig(initialBeta = 1.0, cpRetreatRate = 0.0, epsilonAdapt = Double.MaxValue)
            )
            currentExploit = currentExploit.updated(rivalId,
              ExploitationInterpolation.retreat(es, config)
            )
          }

    // Step 3: Kernel-based rival state update (Def 23)
    val updatedRivals = fullRivalUpdate(currentRivals, signal, publicState, kernelProfile)

    // Step 4-5: Detection-based retreat + safety clamp (Def 15C + A6')
    val updatedExploit = currentExploit.map { case (rivalId, state) =>
      val config = exploitConfigs.getOrElse(rivalId,
        ExploitationConfig(initialBeta = 1.0, cpRetreatRate = 0.0, epsilonAdapt = Double.MaxValue)
      )
      val updated = ExploitationInterpolation.updateExploitation(
        state = state,
        config = config,
        rivalId = rivalId,
        history = publicState.actionHistory,
        publicState = publicState,
        detector = detector,
        exploitabilityFn = exploitabilityFn,
        epsilonNE = epsilonNE
      )
      rivalId -> updated
    }

    DynamicsStepWithCPDResult(updatedRivals, updatedExploit, newCpdStates, detected)
