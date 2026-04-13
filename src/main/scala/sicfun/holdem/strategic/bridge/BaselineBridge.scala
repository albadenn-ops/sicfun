package sicfun.holdem.strategic.bridge

import sicfun.holdem.strategic.*

/** Bridge: engine equity calculations -> RealBaseline, AttributedBaseline.
  *
  * Maps the engine's equity evaluations to the formal baseline types (Defs 9-10).
  *
  * v0.31.1 note: these baselines feed into DeploymentBaseline (A10) which
  * bounds epsilon_base in TotalVulnerability (Corollary 9.3). The approximation
  * quality here directly affects the safety budget computation.
  *
  * Fidelity:
  * - RealBaseline: Approximate (engine uses Monte Carlo equity, not exact)
  * - AttributedBaseline: Approximate (per-rival attribution requires kernel decomposition)
  */
object BaselineBridge:

  /** Convert engine equity to a RealBaseline value. */
  def toRealBaseline(equityEv: Double): BridgeResult[Ev] =
    BridgeResult.Approximate(Ev(equityEv), "Monte Carlo equity approximation")

  /** Bridge an AttributedBaseline into the bridge result layer.
    *
    * The bridge annotates fidelity; it no longer transforms the data.
    * The baseline is kernel-coupled via PosteriorAttributedBaseline.
    */
  def toAttributedBaseline(baseline: AttributedBaseline): BridgeResult[AttributedBaseline] =
    BridgeResult.Approximate(baseline, "kernel-coupled posterior-predictive attribution; per-rival via PosteriorAttributedBaseline")
