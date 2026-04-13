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

/** Canonical Delta-vocabulary (Def 50).
  *
  * Bundles all three categories of the spec's canonical vocabulary:
  *   1. Per-rival primitives: deltaSig, deltaPass, deltaManip, deltaSigDesign, deltaSigReal
  *   2. Aggregate grid primitives: deltaCont, deltaSig*, deltaInt (via FourWorld), deltaSigAggregate
  *   3. Chain primitives: ChainEdgeDelta, ChainRiskDelta, marginal efficiency rho
  */
final case class DeltaVocabulary(
    fourWorld: FourWorld,
    perRivalDeltas: Map[PlayerId, PerRivalDelta],
    deltaSigAggregate: Ev,
    perRivalSubDecompositions: Map[PlayerId, PerRivalSignalSubDecomposition] = Map.empty,
    chainEdgeDeltas: IndexedSeq[ChainEdgeDelta] = IndexedSeq.empty,
    chainRiskDeltas: IndexedSeq[ChainRiskDelta] = IndexedSeq.empty,
    edgeEfficiencies: IndexedSeq[(ChainWorld, ChainWorld, Option[Double])] = IndexedSeq.empty
)

/** Chain-indexed Q-function for baseline (Def 47A).
  *
  * Q^{bar_pi, (omega^act, omega^sd)}(b, u) := Q^{bar_pi, Gamma^{(omega^act, omega^sd)}}(b, u)
  *
  * Holds the baseline policy's Q-value evaluated under each chain world's kernel profile.
  * The values are keyed by ChainWorld and used as input to the telescopic edge decomposition
  * (Def 47B / Proposition 8.1) and the edge efficiency computation (Def 69).
  */
final case class ChainBaselineQ(values: Map[ChainWorld, Ev]):
  require(values.nonEmpty, "ChainBaselineQ must contain at least one world")

  def apply(world: ChainWorld): Ev =
    values.getOrElse(world, throw IllegalArgumentException(
      s"No baseline Q-value for chain world $world; available: ${values.keys.mkString(", ")}"
    ))

  /** Extract Q-values along an ordered chain for telescopic decomposition. */
  def alongChain(chain: IndexedSeq[ChainWorld]): IndexedSeq[Ev] =
    chain.map(apply)

  /** Compute edge deltas along the canonical chain (delegates to TelescopicEdgeDecomposition). */
  def canonicalEdgeDeltas: IndexedSeq[ChainEdgeDelta] =
    val chain = ChainWorld.canonicalChain
    TelescopicEdgeDecomposition.computeEdgeDeltas(chain, alongChain(chain))

/** Value delta along a chain edge (Def 47B). */
final case class ChainEdgeDelta(from: ChainWorld, to: ChainWorld, delta: Ev)

/** Risk delta along a chain edge (Def 68). */
final case class ChainRiskDelta(from: ChainWorld, to: ChainWorld, riskDelta: Ev)

/** Telescopic edge decomposition (Proposition 8.1).
  *
  * Q^{bar_pi, omega_m}(b, u) = Q^{bar_pi, omega_0}(b, u) + sum_{k=0}^{m-1} Delta^edge_{k->k+1}(b, u)
  *
  * Given Q-values at each chain world, computes the edge deltas and verifies
  * the telescopic identity.
  */
object TelescopicEdgeDecomposition:

  /** Compute edge deltas along an ordered chain of worlds.
    *
    * @param chain ordered chain of worlds
    * @param qValues Q-value at each chain world (same ordering as chain)
    * @return edge deltas for each adjacent pair
    */
  def computeEdgeDeltas(
      chain: IndexedSeq[ChainWorld],
      qValues: IndexedSeq[Ev]
  ): IndexedSeq[ChainEdgeDelta] =
    require(chain.size == qValues.size,
      s"chain length ${chain.size} must match qValues length ${qValues.size}")
    (0 until chain.size - 1).map { k =>
      ChainEdgeDelta(
        from = chain(k),
        to = chain(k + 1),
        delta = qValues(k + 1) - qValues(k)
      )
    }

  /** Verify and compute the telescopic decomposition (Proposition 8.1).
    *
    * Returns (totalGap, edgeDeltas) where totalGap = Q(omega_m) - Q(omega_0)
    * and sum(edgeDeltas) == totalGap by construction.
    */
  def decompose(
      chain: IndexedSeq[ChainWorld],
      qValues: IndexedSeq[Ev]
  ): (Ev, IndexedSeq[ChainEdgeDelta]) =
    val edges = computeEdgeDeltas(chain, qValues)
    val totalGap = if chain.size >= 2 then qValues.last - qValues.head else Ev.Zero
    (totalGap, edges)
