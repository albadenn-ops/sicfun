package sicfun.holdem.strategic.bridge

import sicfun.holdem.strategic.*

// MIGRATION CHECKLIST (Wave 0 — spec hygiene fence)
// --------------------------------------------------
// Symbol aliases introduced in v0.31.1; old names remain canonical during Waves 1-6:
//   delta_adapt    -> epsilon_adapt  (§9A')
//   delta_retreat  -> delta_cp_retreat  (§9B)
//   omega must always be qualified as chain-omega or grid-omega.
//
// Compatibility policy:
//   - Keep toFourWorld / toDeltaVocabulary signatures unchanged through Wave 6.
//   - Mark old bridge methods @deprecated once Wave 6 bridge migration lands.
//   - Remove old bridge paths in Wave 7 only.
//
// Pending work tracked here:
//   V^{1,0}, V^{0,1}  — interpolated today; proper POMDP values due Phase 3 / Wave 3
//   controlFrac param  — heuristic split; to be replaced by formal decomposition in Wave 4

/** Bridge: engine EV calculations -> FourWorld, DeltaVocabulary.
  *
  * The four-world decomposition requires evaluating the same decision under
  * four kernel/policy combinations. The current engine evaluates only one
  * (the full attributed, closed-loop case: V^{1,1}).
  *
  * Until the POMDP solver (Phase 3) is integrated, the bridge provides:
  * - V^{1,1}: from engine EV (Exact)
  * - V^{0,0}: from engine static equity (Approximate)
  * - V^{1,0}, V^{0,1}: interpolated estimates (Approximate)
  *
  * Fidelity: Approximate (only V^{1,1} is directly available)
  */
object ValueBridge:

  /** Build a FourWorld from available engine data.
    *
    * @param engineEv      the engine's EV estimate (maps to V^{1,1})
    * @param staticEquity  the engine's equity without adaptation (maps to V^{0,0} approx)
    * @param controlFrac   estimated fraction of value from control (default 0.5)
    */
  def toFourWorld(
      engineEv: Double,
      staticEquity: Double,
      controlFrac: Double = 0.5
  ): BridgeResult[FourWorld] =
    require(controlFrac >= 0.0 && controlFrac <= 1.0, s"controlFrac must be in [0,1], got $controlFrac")
    val v11 = Ev(engineEv)
    val v00 = Ev(staticEquity)
    val gap = engineEv - staticEquity
    // Estimate: split the gap between control and signaling
    val v01 = Ev(staticEquity + gap * controlFrac)
    val v10 = Ev(staticEquity + gap * (1.0 - controlFrac))
    BridgeResult.Approximate(
      FourWorld(v11, v10, v01, v00),
      "V^{1,0} and V^{0,1} are interpolated estimates; only V^{1,1} and V^{0,0} approximate"
    )

  /** Build a DeltaVocabulary from a FourWorld and per-rival data. */
  def toDeltaVocabulary(
      fourWorld: FourWorld,
      perRivalDeltas: Map[PlayerId, PerRivalDelta],
      deltaSigAggregate: Ev,
      perRivalSubDecomps: Map[PlayerId, PerRivalSignalSubDecomposition] = Map.empty
  ): BridgeResult[DeltaVocabulary] =
    BridgeResult.Approximate(
      FourWorldDecomposition.buildDeltaVocabulary(fourWorld, perRivalDeltas, deltaSigAggregate, perRivalSubDecomps),
      "delta vocabulary built from approximate four-world values"
    )
