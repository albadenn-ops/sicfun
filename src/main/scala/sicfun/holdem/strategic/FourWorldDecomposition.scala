package sicfun.holdem.strategic

/** Four-world aggregate decomposition (Defs 44-47 implementation).
  *
  * The FourWorld type itself is defined in Phase 1 (StrategicValue.scala).
  * This object provides the computation entry point and DeltaVocabulary assembly.
  *
  * Theorem 4: V^{1,1} = V^{0,0} + Delta_cont + Delta_sig* + Delta_int.
  * This identity holds by construction (algebraic telescoping of Defs 45-47).
  */
object FourWorldDecomposition:

  /** Compute FourWorld from the four value function evaluations.
    *
    * @param vAttribClosedLoop V^{1,1}: full attributed kernels, closed-loop policy
    * @param vAttribOpenLoop   V^{1,0}: full attributed kernels, open-loop policy
    * @param vBlindClosedLoop  V^{0,1}: blind kernels, closed-loop policy
    * @param vBlindOpenLoop    V^{0,0}: blind kernels, open-loop policy (baseline world)
    */
  def compute(
      vAttribClosedLoop: Ev,
      vAttribOpenLoop: Ev,
      vBlindClosedLoop: Ev,
      vBlindOpenLoop: Ev
  ): FourWorld =
    FourWorld(
      v11 = vAttribClosedLoop,
      v10 = vAttribOpenLoop,
      v01 = vBlindClosedLoop,
      v00 = vBlindOpenLoop
    )

  /** Assemble a complete DeltaVocabulary (Def 50).
    *
    * @param fourWorld            the four-world values
    * @param perRivalDeltas       per-rival decompositions (Defs 40-42)
    * @param deltaSigAggregate    aggregate signal effect (Def 43)
    * @param perRivalSubDecomps   optional per-rival sub-decompositions (Defs 48-49)
    */
  def buildDeltaVocabulary(
      fourWorld: FourWorld,
      perRivalDeltas: Map[PlayerId, PerRivalDelta],
      deltaSigAggregate: Ev,
      perRivalSubDecomps: Map[PlayerId, PerRivalSignalSubDecomposition] = Map.empty
  ): DeltaVocabulary =
    DeltaVocabulary(
      fourWorld = fourWorld,
      perRivalDeltas = perRivalDeltas,
      deltaSigAggregate = deltaSigAggregate,
      perRivalSubDecompositions = perRivalSubDecomps
    )
