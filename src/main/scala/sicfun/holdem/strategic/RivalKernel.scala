package sicfun.holdem.strategic

import sicfun.core.DiscreteDistribution

// MIGRATION CHECKLIST (Wave 0 — spec hygiene fence)
// --------------------------------------------------
// Symbol aliases introduced in v0.31.1; old names remain canonical during Waves 1-6:
//   delta_adapt    -> epsilon_adapt  (§9A')
//   delta_retreat  -> delta_cp_retreat  (§9B)
//   omega must always be qualified as chain-omega or grid-omega.
//
// Compatibility policy:
//   - Keep all current kernel trait signatures and KernelVariant members unchanged
//     through Wave 6.
//   - Mark old names @deprecated once Wave 2 (kernel closure) replacements exist.
//   - Remove old names in Wave 7 only.
//
// Pending work tracked here:
//   PosteriorDivergencePolarization — proxy implementation; real KL due Wave 2
//   KernelVariant.Design            — placeholder; concrete design kernel due Wave 2

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
