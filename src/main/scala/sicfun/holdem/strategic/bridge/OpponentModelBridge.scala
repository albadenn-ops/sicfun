package sicfun.holdem.strategic.bridge

import sicfun.holdem.strategic.*
import sicfun.core.DiscreteDistribution

/** Bridge: engine opponent profiles -> formal RivalMap[OperativeBelief].
  *
  * The current engine does not maintain per-rival augmented beliefs in the
  * formal sense (Def 14). This bridge provides a best-effort mapping from
  * engine-level opponent model data (if available) to formal beliefs.
  *
  * v0.31.1 note: the heuristic mapping from VPIP/PFR/AF to class posteriors
  * is NOT a Bayesian update. It does not use tempered likelihoods or kernel
  * decomposition. The output is classified as Approximate, not Exact.
  *
  * Fidelity: Approximate (engine uses VPIP/PFR/AF stats, not strategic-class posteriors)
  */
object OpponentModelBridge:

  /** Convert engine-level player stats to a class posterior approximation.
    *
    * Maps aggregate stats to a distribution over StrategicClass.
    * This is a heuristic mapping, not a Bayesian update.
    */
  def statsToClassPosterior(
      vpip: Double,
      pfr: Double,
      af: Double
  ): BridgeResult[DiscreteDistribution[StrategicClass]] =
    if vpip < 0.0 || pfr < 0.0 || af < 0.0 then
      return BridgeResult.Absent(s"invalid stats: vpip=$vpip, pfr=$pfr, af=$af (all must be >= 0)")
    // Heuristic: high AF => more bluffs, high VPIP+low PFR => marginal
    val bluffWeight = math.min(af * 0.1, 0.4)
    val valueWeight = math.min(pfr * 0.02, 0.4)
    val semiBluffWeight = math.min((vpip - pfr).abs * 0.01, 0.2)
    val marginalWeight = 1.0 - bluffWeight - valueWeight - semiBluffWeight
    val weights = Map(
      StrategicClass.Value     -> math.max(valueWeight, 0.01),
      StrategicClass.Bluff     -> math.max(bluffWeight, 0.01),
      StrategicClass.Marginal  -> math.max(marginalWeight, 0.01),
      StrategicClass.SemiBluff -> math.max(semiBluffWeight, 0.01)
    )
    val total = weights.values.sum
    val normalized = weights.map((k, v) => k -> v / total)
    BridgeResult.Approximate(
      DiscreteDistribution(normalized),
      "heuristic mapping from VPIP/PFR/AF; not formal Bayesian update"
    )

  /** Bridge class posterior from kernel pipeline when Strategic mode is active.
    *
    * Reads typePosterior directly from StrategicRivalBelief objects in the beliefs map.
    * This is used when Strategic mode has wired the kernel pipeline and kernel-derived
    * beliefs are available.
    */
  def classPosteriorsFromBeliefs(
      beliefs: Map[PlayerId, StrategicRivalBelief]
  ): Map[PlayerId, DiscreteDistribution[StrategicClass]] =
    beliefs.map { case (id, belief) => id -> belief.typePosterior }
