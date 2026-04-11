package sicfun.holdem.strategic

import sicfun.core.DiscreteDistribution

/** Spot-conditioned polarization profile (Def 25, A9).
  *
  * Pol_t^i(lambda | x_t^pub, pi^{0,S}, m_t^{R,i})
  *
  * For each rival i, the polarization profile quantifies how much
  * information about rival i's private state is revealed by a particular
  * sizing lambda, conditioned on the public spot.
  *
  * This is a trait because the polarization computation depends on
  * the specific rival model and is computed differently for different
  * kernel variants.
  */
trait SpotPolarization:
  /** The fidelity level of this polarization implementation.
    *
    * Implementations self-report whether they compute exact posterior
    * divergence (Fidelity.Exact), use an approximation (Fidelity.Approximate),
    * or are absent/stubbed (Fidelity.Absent).
    */
  def fidelity: Fidelity

  /** Compute the polarization value for a given sizing in a spot.
    *
    * Higher polarization means the sizing reveals more about the
    * rival's range/type — i.e., the action is more informative.
    *
    * @param sizing the bet sizing to evaluate
    * @param publicState current public game state
    * @param rivalState rival's current belief state
    * @return polarization value in [0, 1]
    */
  def polarization(
      sizing: Sizing,
      publicState: PublicState,
      rivalState: RivalBeliefState
  ): Double

  /** Compute the full polarization profile over a set of candidate sizings.
    *
    * @param candidates candidate sizings to evaluate
    * @param publicState current public game state
    * @param rivalState rival's current belief state
    * @return map from sizing to polarization value
    */
  def profile(
      candidates: Vector[Sizing],
      publicState: PublicState,
      rivalState: RivalBeliefState
  ): Map[Sizing, Double] =
    candidates.map(s => s -> polarization(s, publicState, rivalState)).toMap

/** Uniform polarization baseline: all sizings are equally informative.
  * Returns 0.5 for every sizing. Used as the no-information baseline
  * when polarization analysis is disabled or as a reference for
  * measuring information gain of other implementations.
  *
  * This is analogous to BlindActionKernel — a real baseline object,
  * not a placeholder for a missing implementation.
  */
object UniformPolarization extends SpotPolarization:
  def fidelity: Fidelity = Fidelity.Exact

  def polarization(
      sizing: Sizing,
      publicState: PublicState,
      rivalState: RivalBeliefState
  ): Double = 0.5

/** Posterior-divergence polarization (Def 25).
  *
  * Pol(lambda) = 1 - exp(-D_KL(posterior_lambda || prior))
  *
  * Computes how much information a specific sizing reveals about the
  * rival's type by measuring the KL divergence between the posterior
  * (after observing the sizing) and the prior.
  *
  * Requires a `TemperedLikelihoodFn` to compute the posterior distribution
  * for a given sizing. When no likelihood is available, falls back to
  * a sizing-extremity proxy (Fidelity.Approximate).
  */
final class PosteriorDivergencePolarization(
    prior: DiscreteDistribution[StrategicClass],
    likelihood: Option[TemperedLikelihoodFn] = None
) extends SpotPolarization:
  def fidelity: Fidelity =
    if likelihood.isDefined then Fidelity.Exact else Fidelity.Approximate

  def polarization(
      sizing: Sizing,
      publicState: PublicState,
      rivalState: RivalBeliefState
  ): Double = likelihood match
    case Some(lf) => klPolarization(sizing, publicState, rivalState, lf)
    case None     => proxyPolarization(sizing)

  /** Real KL-divergence polarization (Def 25 exact). */
  private def klPolarization(
      sizing: Sizing,
      publicState: PublicState,
      rivalState: RivalBeliefState,
      lf: TemperedLikelihoodFn
  ): Double =
    // Build a signal representing this sizing as a raise action
    val signal = ActionSignal(
      action = sicfun.holdem.types.PokerAction.Category.Raise,
      sizing = Some(sizing),
      timing = None,
      stage = publicState.street
    )
    val posterior = lf(signal, publicState, rivalState)
    val kl = PosteriorDivergencePolarization.klDivergence(posterior, prior)
    math.max(0.0, math.min(1.0, 1.0 - math.exp(-kl)))

  /** Sizing-extremity proxy (fallback when no likelihood is available). */
  private def proxyPolarization(sizing: Sizing): Double =
    val f = sizing.fractionOfPot.value
    val extremity = math.abs(2.0 * f - 1.0)
    1.0 - math.exp(-2.0 * extremity)

object PosteriorDivergencePolarization:
  /** KL divergence D_KL(p || q) for discrete distributions over StrategicClass.
    *
    * D_KL(p || q) = sum_c p(c) * ln(p(c) / q(c))
    *
    * Follows the convention:
    *   - 0 * ln(0/q) = 0
    *   - p * ln(p/0) = +inf (in practice, tempered priors have full support)
    */
  def klDivergence(
      p: DiscreteDistribution[StrategicClass],
      q: DiscreteDistribution[StrategicClass]
  ): Double =
    import scala.util.boundary, boundary.break
    val allClasses = (p.support ++ q.support).toSet
    var kl = 0.0
    boundary:
      for cls <- allClasses do
        val pVal = p.probabilityOf(cls)
        val qVal = q.probabilityOf(cls)
        if pVal > 1e-15 then
          if qVal > 1e-15 then kl += pVal * math.log(pVal / qVal)
          else break(Double.PositiveInfinity)
      kl
