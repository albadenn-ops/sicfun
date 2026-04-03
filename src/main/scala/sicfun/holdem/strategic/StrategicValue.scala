package sicfun.holdem.strategic

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
