package sicfun.holdem.strategic

import sicfun.core.DiscreteDistribution
import sicfun.holdem.types.{Board, Position}

/** Tempered likelihood function type.
  *
  * Given an action signal, public state, and rival belief state,
  * returns the updated posterior over strategic classes.
  *
  * This abstracts over the specific tempered likelihood computation
  * (Def 15A-15B), which lives in Phase 2. KernelConstructor depends
  * only on this function type, not on TemperedLikelihood internals.
  */
type TemperedLikelihoodFn = (
    ActionSignal,
    PublicState,
    RivalBeliefState
) => DiscreteDistribution[StrategicClass]

/** Kernel constructor (Defs 16-21 implementation).
  *
  * Builds concrete rival kernels from tempered likelihood functions
  * and state-embedding updaters. Each method corresponds to a
  * specific definition in §4.2.
  */
object KernelConstructor:

  /** Build an action kernel with explicit public state threading (Def 17, full form).
    *
    * This is the production form where public state flows through.
    */
  def buildActionKernelFull[M <: RivalBeliefState](
      updater: StateEmbeddingUpdater[M],
      likelihood: TemperedLikelihoodFn
  ): ActionKernelFull[M] =
    new ActionKernelFull[M]:
      def apply(state: M, signal: ActionSignal, publicState: PublicState): M =
        val posterior = likelihood(signal, publicState, state)
        updater(state, posterior)

  /** Build a design-signal kernel (Def 19A).
    *
    * Strips sizing and timing from the signal before computing likelihood.
    * Uses only the action category component a_t.
    *
    * Marginalization order: temper-then-marginalize (canonical).
    * Since we strip to just the action category before passing to the
    * likelihood function, the likelihood receives a signal with
    * sizing=None, timing=None. The likelihood function is responsible
    * for summing over (lambda, tau) internally if needed.
    *
    * Note: uses a minimal zero-state PublicState internally for backward
    * compatibility with chain-world composition. For production use where
    * real public state context matters, use `buildDesignKernelFull`.
    */
  def buildDesignKernel[M <: RivalBeliefState](
      updater: StateEmbeddingUpdater[M],
      likelihood: TemperedLikelihoodFn
  ): ActionKernel[M] =
    new ActionKernel[M]:
      def apply(state: M, signal: ActionSignal): M =
        // Strip sizing and timing — keep only action category and stage
        val designSignal = ActionSignal(
          action = signal.action,
          sizing = None,
          timing = None,
          stage = signal.stage
        )
        val placeholder = PlayerId("__design_placeholder__")
        val pub = PublicState(
          street = signal.stage,
          board = Board.empty,
          pot = Chips(0.0),
          stacks = TableMap(
            hero = placeholder,
            seats = Vector(
              Seat(placeholder, Position.SmallBlind, SeatStatus.Active, Chips(0.0))
            )
          ),
          actionHistory = Vector.empty
        )
        val posterior = likelihood(designSignal, pub, state)
        updater(state, posterior)

  /** Build a design-signal kernel with explicit public state (Def 19A, full form). */
  def buildDesignKernelFull[M <: RivalBeliefState](
      updater: StateEmbeddingUpdater[M],
      likelihood: TemperedLikelihoodFn
  ): ActionKernelFull[M] =
    new ActionKernelFull[M]:
      def apply(state: M, signal: ActionSignal, publicState: PublicState): M =
        val designSignal = ActionSignal(
          action = signal.action,
          sizing = None,
          timing = None,
          stage = signal.stage
        )
        val posterior = likelihood(designSignal, publicState, state)
        updater(state, posterior)

  /** Compose a full kernel from action + showdown channels (Def 20).
    *
    * Gamma^{full,bullet,i}(m, Y, x^pub) =
    *   if Y^sd != empty and bullet != blind:
    *     showdown(action(m, Y^act), Y^sd)
    *   if Y^sd == empty and bullet != blind:
    *     action(m, Y^act)
    *   if bullet == blind:
    *     m
    */
  def composeFullKernel[M <: RivalBeliefState](
      actionKernel: ActionKernel[M],
      showdownKernel: ShowdownKernel[M]
  ): FullKernel[M] =
    new FullKernel[M]:
      def apply(state: M, signal: TotalSignal, publicState: PublicState): M =
        val afterAction = actionKernel.apply(state, signal.actionSignal)
        signal.showdown match
          case Some(sd) => showdownKernel.apply(afterAction, sd)
          case None     => afterAction

  /** Compose a full kernel from a full-form action kernel + showdown (Def 20, production form).
    *
    * Like [[composeFullKernel]] but accepts [[ActionKernelFull]] so the public state
    * is threaded through to the action kernel rather than dropped.
    */
  def composeFullKernelFromFull[M <: RivalBeliefState](
      actionKernel: ActionKernelFull[M],
      showdownKernel: ShowdownKernel[M]
  ): FullKernel[M] =
    new FullKernel[M]:
      def apply(state: M, signal: TotalSignal, publicState: PublicState): M =
        val afterAction = actionKernel.apply(state, signal.actionSignal, publicState)
        signal.showdown match
          case Some(sd) => showdownKernel.apply(afterAction, sd)
          case None     => afterAction

  /** Compose a blind full kernel (Def 20, bullet=blind).
    *
    * Always returns the input state unchanged. Law L12: blind kernel is identity.
    */
  def composeBlindFullKernel[M <: RivalBeliefState](): FullKernel[M] =
    new FullKernel[M]:
      def apply(state: M, signal: TotalSignal, publicState: PublicState): M = state

  /** Compose a design full kernel: design action kernel + showdown kernel (Def 20, bullet=design).
    *
    * Parallel to [[composeFullKernel]] but uses a design action kernel that strips
    * sizing and timing from the signal before computing the likelihood.
    */
  def composeDesignFullKernel[M <: RivalBeliefState](
      designActionKernel: ActionKernel[M],
      showdownKernel: ShowdownKernel[M]
  ): FullKernel[M] =
    new FullKernel[M]:
      def apply(state: M, signal: TotalSignal, publicState: PublicState): M =
        val afterAction = designActionKernel.apply(state, signal.actionSignal)
        signal.showdown match
          case Some(sd) => showdownKernel.apply(afterAction, sd)
          case None     => afterAction

  /** Compose a full kernel for a specific [[ChainWorld]].
    *
    * Routes the (LearningChannel, ShowdownMode) pair to the correct kernel composition:
    *   - (Blind, Off) and (Blind, On) -> blind full kernel (identity)
    *   - (Ref, Off)                   -> action kernel only (showdown gated off)
    *   - (Ref, On)                    -> action + showdown
    *   - (Attrib, Off)                -> action kernel only (showdown gated off)
    *   - (Attrib, On)                 -> action + showdown
    *   - (Design, Off)                -> design kernel only (showdown gated off)
    *   - (Design, On)                 -> design kernel + showdown
    *
    * Per Def 18, Ref and Attrib channels use DISTINCT kernels:
    *   Ref uses BuildRivalKernel(pi^{0,S}); Attrib uses BuildRivalKernel(hat{pi}^{0,S,i}).
    *
    * @param world the chain world determining kernel variant and showdown gating
    * @param refActionKernel the action kernel for the Ref channel (built from pi^{0,S})
    * @param attribActionKernel the action kernel for the Attrib channel (built from hat{pi}^{0,S,i})
    * @param designActionKernel the design action kernel (strips sizing/timing)
    * @param showdownKernel the showdown kernel for On worlds
    */
  def composeFullKernelForWorld[M <: RivalBeliefState](
      world: ChainWorld,
      refActionKernel: ActionKernel[M],
      attribActionKernel: ActionKernel[M],
      designActionKernel: ActionKernel[M],
      showdownKernel: ShowdownKernel[M]
  ): FullKernel[M] =
    world.channel match
      case LearningChannel.Blind =>
        composeBlindFullKernel[M]()

      case LearningChannel.Design =>
        world.showdown match
          case ShowdownMode.Off =>
            new FullKernel[M]:
              def apply(state: M, signal: TotalSignal, publicState: PublicState): M =
                designActionKernel.apply(state, signal.actionSignal)
          case ShowdownMode.On =>
            composeDesignFullKernel(designActionKernel, showdownKernel)

      case LearningChannel.Ref =>
        world.showdown match
          case ShowdownMode.Off =>
            new FullKernel[M]:
              def apply(state: M, signal: TotalSignal, publicState: PublicState): M =
                refActionKernel.apply(state, signal.actionSignal)
          case ShowdownMode.On =>
            composeFullKernel(refActionKernel, showdownKernel)

      case LearningChannel.Attrib =>
        world.showdown match
          case ShowdownMode.Off =>
            new FullKernel[M]:
              def apply(state: M, signal: TotalSignal, publicState: PublicState): M =
                attribActionKernel.apply(state, signal.actionSignal)
          case ShowdownMode.On =>
            composeFullKernel(attribActionKernel, showdownKernel)

  /** Compose a full kernel for a specific [[ChainWorld]] using [[ActionKernelFull]] kernels.
    *
    * Production form of [[composeFullKernelForWorld]] that threads [[PublicState]] through
    * the action kernels. Per Def 18, Ref and Attrib use distinct kernels.
    *
    * @param refActionKernelFull the full-form action kernel for Ref channel (built from pi^{0,S})
    * @param attribActionKernelFull the full-form action kernel for Attrib channel (built from hat{pi}^{0,S,i})
    * @param designKernelFull the full-form design action kernel (strips sizing/timing)
    * @param showdownKernel the showdown kernel for On worlds
    * @param world the chain world determining kernel variant and showdown gating
    */
  def composeFullKernelForWorldFull[M <: RivalBeliefState](
      refActionKernelFull: ActionKernelFull[M],
      attribActionKernelFull: ActionKernelFull[M],
      designKernelFull: ActionKernelFull[M],
      showdownKernel: ShowdownKernel[M]
  )(world: ChainWorld): FullKernel[M] =
    world.channel match
      case LearningChannel.Blind =>
        composeBlindFullKernel[M]()

      case LearningChannel.Design =>
        world.showdown match
          case ShowdownMode.Off =>
            new FullKernel[M]:
              def apply(state: M, signal: TotalSignal, publicState: PublicState): M =
                designKernelFull.apply(state, signal.actionSignal, publicState)
          case ShowdownMode.On =>
            composeFullKernelFromFull(designKernelFull, showdownKernel)

      case LearningChannel.Ref =>
        world.showdown match
          case ShowdownMode.Off =>
            new FullKernel[M]:
              def apply(state: M, signal: TotalSignal, publicState: PublicState): M =
                refActionKernelFull.apply(state, signal.actionSignal, publicState)
          case ShowdownMode.On =>
            composeFullKernelFromFull(refActionKernelFull, showdownKernel)

      case LearningChannel.Attrib =>
        world.showdown match
          case ShowdownMode.Off =>
            new FullKernel[M]:
              def apply(state: M, signal: TotalSignal, publicState: PublicState): M =
                attribActionKernelFull.apply(state, signal.actionSignal, publicState)
          case ShowdownMode.On =>
            composeFullKernelFromFull(attribActionKernelFull, showdownKernel)

/** Action kernel with explicit public state (production form of Def 17). */
trait ActionKernelFull[M]:
  def apply(state: M, signal: ActionSignal, publicState: PublicState): M

/** Joint kernel profile (Def 21 implementation).
  *
  * Gamma^{attrib} := {Gamma^{full,attrib,i}}_{i in R}
  * Gamma^{ref}    := {Gamma^{full,ref,i}}_{i in R}
  * Gamma^{blind}  := {Gamma^{full,blind,i}}_{i in R}
  *
  * Extends the Phase 1 KernelProfile (which was action-only) to
  * full kernels (action + showdown composition).
  */
final case class JointKernelProfile[M <: RivalBeliefState](
    kernels: Map[PlayerId, FullKernel[M]]
):
  /** Apply the profile to update all rivals simultaneously (Def 23 helper). */
  def updateAll(
      states: Map[PlayerId, M],
      signal: TotalSignal,
      publicState: PublicState
  ): Map[PlayerId, M] =
    states.map { case (id, state) =>
      kernels.get(id) match
        case Some(kernel) => id -> kernel.apply(state, signal, publicState)
        case None         => id -> state // no kernel for this rival, preserve state
    }

/** World-indexed kernel profile: stores full kernels for each (PlayerId, ChainWorld) pair.
  *
  * This extends the Phase 1 JointKernelProfile to be chain-world aware.
  * Each rival may have a different kernel for each of the 8 chain worlds.
  */
final case class WorldIndexedKernelProfile[M <: RivalBeliefState](
    kernels: Map[(PlayerId, ChainWorld), FullKernel[M]]
):
  /** Extract a [[JointKernelProfile]] for a specific chain world. */
  def forWorld(world: ChainWorld): JointKernelProfile[M] =
    val worldKernels = kernels.collect {
      case ((pid, w), kernel) if w == world => pid -> kernel
    }
    JointKernelProfile(worldKernels)

  /** Apply the world-specific profile to update all rivals simultaneously. */
  def updateAll(
      states: Map[PlayerId, M],
      signal: TotalSignal,
      publicState: PublicState,
      world: ChainWorld
  ): Map[PlayerId, M] =
    forWorld(world).updateAll(states, signal, publicState)
