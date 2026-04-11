package sicfun.holdem.strategic

/** Signaling sub-decomposition (Defs 48-49).
  *
  * Splits the per-rival total signal effect into:
  * - Design-signal effect (Def 48): from action category alone (design kernel)
  * - Realization-signal effect (Def 49): from sizing/timing beyond category
  *
  * Theorem 3A: delta_sig = delta_sig,design + delta_sig,real (exact telescoping).
  *
  * Q-function naming:
  * - qAttrib: Q^{*, (Gamma^{full,attrib,i}, B^{-i})}
  * - qDesign: Q^{*, (Gamma^{full,design,i}, B^{-i})}
  * - qBlind:  Q^{*, (Gamma^{full,blind,i},  B^{-i})}
  */
object SignalingSubDecomposition:

  /** Def 48: Design-signal effect.
    * Delta_sig,design^{i|B^{-i}} = Q^{design,i} - Q^{blind,i}.
    */
  def deltaSigDesign(qDesign: Ev, qBlind: Ev): Ev = qDesign - qBlind

  /** Def 49: Realization-signal effect.
    * Delta_sig,real^{i|B^{-i}} = Q^{attrib,i} - Q^{design,i}.
    */
  def deltaSigReal(qAttrib: Ev, qDesign: Ev): Ev = qAttrib - qDesign

  /** Build a complete PerRivalSignalSubDecomposition.
    * Theorem 3A holds by construction: total = design + real.
    */
  def compute(qAttrib: Ev, qDesign: Ev, qBlind: Ev): PerRivalSignalSubDecomposition =
    PerRivalSignalSubDecomposition(
      deltaSigDesign = deltaSigDesign(qDesign, qBlind),
      deltaSigReal = deltaSigReal(qAttrib, qDesign)
    )
