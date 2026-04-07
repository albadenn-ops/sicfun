package sicfun.holdem.strategic

/** World-aware risk decomposition (Wave 5 — v0.31.1 formal closure).
  *
  * Implements:
  *   - Chain-indexed robust one-step loss (Def 56A)
  *   - Risk increment between adjacent chain worlds (Def 56B)
  *   - Telescopic risk decomposition along canonical chain (Proposition 9.7)
  *   - Marginal layer efficiency metric rho_{k→k+1} (Def 56C)
  *
  * Uses the same chain-world ordering as `ChainWorld.canonicalChain` (Wave 1)
  * and `KernelConstructor.composeFullKernelForWorld` (Wave 2).
  */
object RiskDecomposition:

  /** Robust one-step loss at a specific chain world (Def 56A).
    *
    * L_robust(omega_k) = max_b [ V_bar(b) - Q(omega_k, b) ]
    *
    * Each entry in the arrays corresponds to one belief point.
    */
  def chainRobustLoss(
      baselineValues: IndexedSeq[Ev],
      qValuesAtWorld: IndexedSeq[Ev]
  ): Ev =
    require(baselineValues.size == qValuesAtWorld.size && baselineValues.nonEmpty,
      "arrays must be non-empty and same length")
    baselineValues.zip(qValuesAtWorld)
      .map((vBase, q) => vBase - q)
      .reduce((a, b) => if a >= b then a else b)

  /** Risk increment between adjacent chain worlds (Def 56B).
    *
    * Delta_risk(omega_k, omega_{k+1}) = L_robust(omega_{k+1}) - L_robust(omega_k)
    *
    * Positive value means the next world introduces MORE risk.
    * Negative value means the next world REDUCES risk.
    */
  def riskIncrement(lossAtK: Ev, lossAtKPlus1: Ev): Ev =
    lossAtKPlus1 - lossAtK

  /** Full chain risk profile: robust losses at each chain world. */
  final case class ChainRiskProfile(
      chain: IndexedSeq[ChainWorld],
      robustLosses: IndexedSeq[Ev]
  ):
    require(chain.size == robustLosses.size,
      s"chain length ${chain.size} must match robustLosses length ${robustLosses.size}")

    /** Risk increments along adjacent chain edges. */
    def riskIncrements: IndexedSeq[ChainRiskDelta] =
      (0 until chain.size - 1).map { k =>
        ChainRiskDelta(
          from = chain(k),
          to = chain(k + 1),
          riskDelta = riskIncrement(robustLosses(k), robustLosses(k + 1))
        )
      }

    /** Telescopic risk decomposition (Proposition 9.7).
      *
      * L_robust(omega_last) - L_robust(omega_0) = sum_{k=0}^{n-2} Delta_risk(k, k+1)
      *
      * Returns the total risk gap and the individual increments.
      */
    def telescopicDecomposition: (Ev, IndexedSeq[ChainRiskDelta]) =
      val increments = riskIncrements
      val totalGap = if chain.size >= 2 then
        robustLosses.last - robustLosses.head
      else Ev.Zero
      (totalGap, increments)

  /** Marginal layer efficiency (Def 56C).
    *
    * rho_{k->k+1} = Delta_value(k, k+1) / Delta_risk(k, k+1)
    *
    * Measures how much value the layer adds per unit of risk.
    * Undefined when the risk increment is zero (returns None).
    *
    * @param valueDelta value increment at this chain edge (from ChainEdgeDelta)
    * @param riskDelta risk increment at this chain edge (from ChainRiskDelta)
    */
  def marginalEfficiency(valueDelta: Ev, riskDelta: Ev): Option[Double] =
    if riskDelta.abs <= Ev(1e-15) then None
    else Some(valueDelta.value / riskDelta.value)

  /** Compute chain risk profile from baseline and per-world Q-values. */
  def computeProfile(
      chain: IndexedSeq[ChainWorld],
      baselineValues: IndexedSeq[Ev],
      qValuesByWorld: IndexedSeq[IndexedSeq[Ev]]
  ): ChainRiskProfile =
    require(chain.size == qValuesByWorld.size,
      s"chain length ${chain.size} must match qValuesByWorld length ${qValuesByWorld.size}")
    val losses = qValuesByWorld.map(qVals => chainRobustLoss(baselineValues, qVals))
    ChainRiskProfile(chain, losses)

  /** Compute efficiency metrics for each chain edge.
    *
    * Pairs value edges (from StrategicValue) with risk edges to compute
    * per-layer efficiency.
    */
  def edgeEfficiencies(
      valueEdges: IndexedSeq[ChainEdgeDelta],
      riskEdges: IndexedSeq[ChainRiskDelta]
  ): IndexedSeq[(ChainWorld, ChainWorld, Option[Double])] =
    require(valueEdges.size == riskEdges.size,
      "value and risk edge counts must match")
    valueEdges.zip(riskEdges).map { (ve, re) =>
      require(ve.from == re.from && ve.to == re.to,
        s"edge mismatch: value ${ve.from}->${ve.to} vs risk ${re.from}->${re.to}")
      (ve.from, ve.to, marginalEfficiency(ve.delta, re.riskDelta))
    }
