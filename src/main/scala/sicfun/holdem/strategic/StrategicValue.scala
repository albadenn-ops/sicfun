package sicfun.holdem.strategic

// MIGRATION CHECKLIST (Wave 0 — spec hygiene fence)
// --------------------------------------------------
// Symbol aliases introduced in v0.31.1; old names remain canonical during Waves 1-6:
//   delta_adapt    -> epsilon_adapt  (§9A')
//   delta_retreat  -> delta_cp_retreat  (§9B)
//   omega must always be qualified as chain-omega or grid-omega.
//
// Compatibility policy:
//   - Keep all current constructors/accessors unchanged through Wave 6.
//   - Mark old names @deprecated once Wave 4 (Defs 57/57A-C) replacements exist.
//   - Remove old names in Wave 7 only.
//
// Pending renames tracked here (do NOT rename until Wave 4 lands):
//   FourWorld       — no rename; but Def 44 will be superseded by Wave 1 WorldAlgebra types
//   DeltaVocabulary — Def 50 reference stays valid; delta field names unchanged
//   PerRivalDelta   — Defs 40-42; no rename planned

final case class FourWorld(
    v11: Ev,
    v10: Ev,
    v01: Ev,
    v00: Ev
):
  inline def deltaControl: Ev = v01 - v00
  inline def deltaSigStar: Ev = v10 - v00
  inline def deltaInteraction: Ev = v11 - v10 - v01 + v00

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
