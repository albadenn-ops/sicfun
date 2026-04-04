package sicfun.holdem.strategic

/** Per-rival signal decomposition operators (Defs 40-43).
  *
  * All operators take pre-computed Q-function values as inputs. The Q-function
  * evaluation itself is the responsibility of the solver layer (Phase 3/4a).
  *
  * Naming convention:
  * - qAttrib: Q^{*, (Gamma^{full,attrib,i}, B^{-i})}  -- attributed kernel
  * - qRef:    Q^{*, (Gamma^{full,ref,i},    B^{-i})}  -- reference kernel
  * - qBlind:  Q^{*, (Gamma^{full,blind,i},  B^{-i})}  -- blind kernel
  *
  * Theorem 3: deltaSig = deltaPass + deltaManip (exact telescoping, by construction).
  * Theorem 5: qAttrib == qRef => deltaManip == 0.
  */
object SignalDecomposition:

  /** Def 40: Total signal effect for rival i under fixed background B^{-i}.
    * Delta_sig^{i|B^{-i}}(b~, u) = Q^{attrib,i} - Q^{blind,i}.
    */
  def deltaSig(qAttrib: Ev, qBlind: Ev): Ev = qAttrib - qBlind

  /** Def 41: Passive leakage for rival i under fixed background B^{-i}.
    * Delta_pass^{i|B^{-i}}(b~, u) = Q^{ref,i} - Q^{blind,i}.
    */
  def deltaPass(qRef: Ev, qBlind: Ev): Ev = qRef - qBlind

  /** Def 42: Manipulation rent for rival i under fixed background B^{-i}.
    * Delta_manip^{i|B^{-i}}(b~, u) = Q^{attrib,i} - Q^{ref,i}.
    */
  def deltaManip(qAttrib: Ev, qRef: Ev): Ev = qAttrib - qRef

  /** Build a complete PerRivalDelta from the three Q-function values.
    * Theorem 3 holds by construction: deltaSig = deltaPass + deltaManip
    * because (qAttrib - qBlind) = (qRef - qBlind) + (qAttrib - qRef).
    */
  def computePerRivalDelta(qAttrib: Ev, qRef: Ev, qBlind: Ev): PerRivalDelta =
    PerRivalDelta(
      deltaSig = deltaSig(qAttrib, qBlind),
      deltaPass = deltaPass(qRef, qBlind),
      deltaManip = deltaManip(qAttrib, qRef)
    )

  /** Def 43: Aggregate signal effect.
    * Delta_sig^{agg}(b~, u) = Q^{*,attrib_all}(b~, u) - Q^{*,blind_all}(b~, u).
    *
    * Non-additivity warning: Delta_sig^{agg} != sum_i Delta_sig^i in general.
    * The aggregate uses joint kernel profiles, not individual sums.
    */
  def deltaSigAggregate(qAttribAll: Ev, qBlindAll: Ev): Ev = qAttribAll - qBlindAll
