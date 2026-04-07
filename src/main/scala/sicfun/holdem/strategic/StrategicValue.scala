package sicfun.holdem.strategic

// STATUS (post-Wave 7 — v0.31.1 formal closure complete)
// -------------------------------------------------------
// v0.31.1 symbol aliases are canonical:
//   delta_adapt    -> epsilon_adapt  (§9A')
//   delta_retreat  -> delta_cp_retreat  (§9B)
//   omega must always be qualified as chain-omega or grid-omega.
//
// FourWorld coexists with GridWorld (keyed accessor via apply(gw: GridWorld)).
// DeltaVocabulary — Def 50 reference valid; delta field names unchanged.
// PerRivalDelta   — Defs 40-42; no rename planned.

final case class FourWorld(
    v11: Ev,
    v10: Ev,
    v01: Ev,
    v00: Ev
):
  inline def deltaControl: Ev = v01 - v00
  inline def deltaSigStar: Ev = v10 - v00
  inline def deltaInteraction: Ev = v11 - v10 - v01 + v00

  /** Keyed accessor: map GridWorld coordinates to the matching positional field.
    *
    * Grid mapping (Def 44):
    *   (Attrib, ClosedLoop) -> V^{1,1}
    *   (Attrib, OpenLoop)   -> V^{1,0}
    *   (Blind,  ClosedLoop) -> V^{0,1}
    *   (Blind,  OpenLoop)   -> V^{0,0}
    */
  def apply(gw: GridWorld): Ev = (gw.learning, gw.scope) match
    case (LearningChannel.Attrib, PolicyScope.ClosedLoop) => v11
    case (LearningChannel.Attrib, PolicyScope.OpenLoop)   => v10
    case (LearningChannel.Blind, PolicyScope.ClosedLoop)  => v01
    case (LearningChannel.Blind, PolicyScope.OpenLoop)    => v00
    case _ => throw IllegalArgumentException(s"Unexpected GridWorld: $gw")

final case class PerRivalDelta(
    deltaSig: Ev,
    deltaPass: Ev,
    deltaManip: Ev
):
  inline def isDamagingLeak: Boolean = deltaPass < Ev.Zero
  inline def hasCorrectBeliefs: Boolean = deltaManip.abs <= Ev(1e-12)

final case class PerRivalSignalSubDecomposition(
    deltaSigDesign: Ev,
    deltaSigReal: Ev
):
  inline def total: Ev = deltaSigDesign + deltaSigReal

final case class DeltaVocabulary(
    fourWorld: FourWorld,
    perRivalDeltas: Map[PlayerId, PerRivalDelta],
    deltaSigAggregate: Ev,
    perRivalSubDecompositions: Map[PlayerId, PerRivalSignalSubDecomposition] = Map.empty
)

/** Value delta along a chain edge (Def 29). */
final case class ChainEdgeDelta(from: ChainWorld, to: ChainWorld, delta: Ev)

/** Risk delta along a chain edge (Def 56). */
final case class ChainRiskDelta(from: ChainWorld, to: ChainWorld, riskDelta: Ev)
