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

  /** Minimal sentinel hero player id used for dummy public states inside kernels. */
  private val sentinelHero: PlayerId = PlayerId("__sentinel__")

  /** Minimal dummy public state for the simplified `buildActionKernel` signature.
    *
    * Used when the caller does not thread a real PublicState through.
    * The TableMap must satisfy its invariants (hero present in seats).
    */
  private def dummyPublicState(street: sicfun.holdem.types.Street): PublicState =
    PublicState(
      street = street,
      board = Board.empty,
      pot = Chips(0.0),
      stacks = TableMap(
        hero = sentinelHero,
        seats = Vector(
          Seat(sentinelHero, Position.SmallBlind, SeatStatus.Active, Chips(0.0))
        )
      ),
      actionHistory = Vector.empty
    )

  /** Build an action kernel from an updater and likelihood (Def 17).
    *
    * BuildRivalKernel^i_{kappa,delta}(pi)(m, y, x^pub)
    *   := U^{R,i}_pi(m, mu^{R,i,pi,kappa,delta}_{t+1})
    *
    * The likelihood function encapsulates the policy reference (pi)
    * and tempering parameters (kappa, delta).
    *
    * Note: uses a sentinel public state internally. For production use
    * where real public state context matters, use `buildActionKernelFull`.
    */
  def buildActionKernel[M <: RivalBeliefState](
      updater: StateEmbeddingUpdater[M],
      likelihood: TemperedLikelihoodFn
  ): ActionKernel[M] =
    new ActionKernel[M]:
      def apply(state: M, signal: ActionSignal): M =
        val pub = dummyPublicState(signal.stage)
        val posterior = likelihood(signal, pub, state)
        updater(state, posterior)

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
        val pub = dummyPublicState(signal.stage)
        val posterior = likelihood(designSignal, pub, state)
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

  /** Compose a blind full kernel (Def 20, bullet=blind).
    *
    * Always returns the input state unchanged. Law L12: blind kernel is identity.
    */
  def composeBlindFullKernel[M <: RivalBeliefState](): FullKernel[M] =
    new FullKernel[M]:
      def apply(state: M, signal: TotalSignal, publicState: PublicState): M = state

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
