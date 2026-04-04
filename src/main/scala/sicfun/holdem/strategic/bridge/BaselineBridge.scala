package sicfun.holdem.strategic.bridge

import sicfun.holdem.strategic.*

/** Bridge: engine equity calculations -> RealBaseline, AttributedBaseline.
  *
  * Maps the engine's equity evaluations to the formal baseline types (Defs 9-10).
  *
  * Fidelity:
  * - RealBaseline: Approximate (engine uses Monte Carlo equity, not exact)
  * - AttributedBaseline: Approximate (per-rival attribution requires kernel decomposition)
  */
object BaselineBridge:

  /** Convert engine equity to a RealBaseline value. */
  def toRealBaseline(equityEv: Double): BridgeResult[Ev] =
    BridgeResult.Approximate(Ev(equityEv), "Monte Carlo equity approximation")

  /** Convert per-rival equity contributions to attributed baselines.
    * @param perRivalEquity map from rival id to their contribution to hero's baseline
    */
  def toAttributedBaselines(
      perRivalEquity: Map[PlayerId, Double]
  ): BridgeResult[Map[PlayerId, Ev]] =
    if perRivalEquity.isEmpty then
      BridgeResult.Absent("no per-rival equity data available")
    else
      BridgeResult.Approximate(
        perRivalEquity.map((pid, eq) => pid -> Ev(eq)),
        "attributed baseline from engine equity split; not formal kernel-based attribution"
      )
