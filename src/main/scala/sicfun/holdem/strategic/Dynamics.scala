package sicfun.holdem.strategic

/** Result of a full dynamics step.
  *
  * Contains updated rival states and exploitation states for all rivals.
  */
final case class DynamicsStepResult[M <: RivalBeliefState](
    updatedRivals: Map[PlayerId, M],
    updatedExploitation: Map[PlayerId, ExploitationState]
)

/** Belief and rival-state dynamics (Defs 22-25). */
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
        ExploitationConfig(initialBeta = 1.0, retreatRate = 0.0, adaptationTolerance = Double.MaxValue)
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
