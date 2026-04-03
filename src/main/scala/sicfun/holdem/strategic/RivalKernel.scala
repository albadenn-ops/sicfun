package sicfun.holdem.strategic

import sicfun.core.DiscreteDistribution

/** Type alias for a function that updates a model state M given a posterior
  * distribution over strategic classes. Used to embed classification posteriors
  * back into belief states after a Bayesian update step (Def 16).
  */
type StateEmbeddingUpdater[M] = (M, DiscreteDistribution[StrategicClass]) => M

/** Kernel that updates a rival belief state M given a single observed ActionSignal.
  * Corresponds to Def 17: the per-action update step of the inference pipeline.
  */
trait ActionKernel[M]:
  def apply(state: M, signal: ActionSignal): M

/** Kernel that updates a rival belief state M given a showdown revelation.
  * Corresponds to Def 18: the hard-evidence update when hole cards are revealed.
  */
trait ShowdownKernel[M]:
  def apply(state: M, showdown: ShowdownSignal): M

/** Full kernel combining action and showdown signals with public state context.
  * Corresponds to Def 19: the combined update operator used in the main belief loop.
  */
trait FullKernel[M]:
  def apply(state: M, signal: TotalSignal, publicState: PublicState): M

/** Enumeration of kernel implementation variants available in the system.
  * Corresponds to Def 20.
  *
  *   - Ref: reference (analytically correct) kernel
  *   - Attrib: attributed kernel with interpretable reasoning chains
  *   - Blind: no-op kernel for baselines and testing
  *   - Design: experimental / under-development kernel
  */
enum KernelVariant:
  case Ref, Attrib, Blind, Design

/** A no-op [[ActionKernel]] that always returns the state unchanged.
  * Used as a baseline and in unit tests. Corresponds to Def 21.
  */
final class BlindActionKernel[M] extends ActionKernel[M]:
  def apply(state: M, signal: ActionSignal): M = state

/** Associates a named [[ActionKernel]] with each rival in the game.
  * Allows per-rival kernel specialisation within a single game session.
  */
final case class KernelProfile[M](
    kernels: Map[PlayerId, ActionKernel[M]]
)
