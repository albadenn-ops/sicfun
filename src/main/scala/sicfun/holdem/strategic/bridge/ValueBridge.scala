package sicfun.holdem.strategic.bridge

import sicfun.holdem.strategic.*

// STATUS (post-Wave 7 — v0.31.1 formal closure complete)
// -------------------------------------------------------
// toFourWorld       — legacy path; interpolates V^{1,0}/V^{0,1} via controlFrac heuristic
// toGridWorldValues — v0.31.1 path; reports V^{1,0}/V^{0,1} as Absent (honest)
// toDeltaVocabulary — wraps FourWorldDecomposition; always Approximate

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

  // ==== v0.31.1 keyed grid-world result (Wave 6) ====

  /** Build keyed grid-world values from available engine data.
    *
    * v0.31.1 requires keyed `GridWorld -> Ev` mapping rather than positional
    * V^{1,1} / V^{0,0} fields. Only V^{1,1} (from engine EV) and V^{0,0}
    * (from static equity) are directly available. V^{1,0} and V^{0,1} are
    * absent unless a POMDP solver provides them.
    *
    * Returns: map from each GridWorld to either an Approximate or Absent Ev.
    */
  def toGridWorldValues(
      engineEv: Double,
      staticEquity: Double
  ): Map[GridWorld, BridgeResult[Ev]] =
    Map(
      GridWorld(LearningChannel.Attrib, PolicyScope.ClosedLoop) ->
        BridgeResult.Approximate(Ev(engineEv), "engine EV (best available for V^{1,1})"),
      GridWorld(LearningChannel.Blind, PolicyScope.OpenLoop) ->
        BridgeResult.Approximate(Ev(staticEquity), "static equity approximation for V^{0,0}"),
      GridWorld(LearningChannel.Attrib, PolicyScope.OpenLoop) ->
        BridgeResult.Absent("V^{1,0} requires POMDP solver (Phase 3); not interpolated"),
      GridWorld(LearningChannel.Blind, PolicyScope.ClosedLoop) ->
        BridgeResult.Absent("V^{0,1} requires POMDP solver (Phase 3); not interpolated")
    )
