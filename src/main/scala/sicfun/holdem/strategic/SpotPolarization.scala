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

/** Uniform polarization (stub): all sizings are equally informative.
  * Returns 0.5 for every sizing. Used as a baseline or when
  * polarization analysis is disabled.
  */
object UniformPolarization extends SpotPolarization:
  def polarization(
      sizing: Sizing,
      publicState: PublicState,
      rivalState: RivalBeliefState
  ): Double = 0.5

/** Posterior-divergence polarization: an approximation of Def 25 polarization.
  *
  * IMPORTANT — this is NOT a real KL divergence computation.
  * The class name and the formula in the spec (Def 25) describe the ideal:
  *   Pol(lambda) = 1 - exp(-D_KL(posterior_lambda || prior))
  * but the current implementation does NOT compute that KL divergence.
  * It uses sizing fraction as a proxy instead (see method body).  The prior
  * parameter is accepted for API compatibility but is currently unused.
  *
  * The "canonical KL" framing in earlier doc versions was an overclaim.
  * This is a sizing-extremity proxy, not a true posterior-divergence measure.
  * A genuine KL-based implementation is deferred to Wave 2 (kernel closure).
  */
final class PosteriorDivergencePolarization(
    prior: DiscreteDistribution[StrategicClass]
) extends SpotPolarization:
  def polarization(
      sizing: Sizing,
      publicState: PublicState,
      rivalState: RivalBeliefState
  ): Double =
    // PROXY IMPLEMENTATION — does not compute actual KL divergence.
    // Uses sizing fraction as a stand-in: extreme sizings (very small or
    // very large relative to pot) are treated as more polarizing.
    // The `prior` field is unused until Wave 2 replaces this with real
    // posterior-divergence computation.
    val f = sizing.fractionOfPot.value
    val extremity = math.abs(2.0 * f - 1.0) // 0 at half-pot, 1 at 0 or full-pot
    // Sigmoid-like transform to [0, 1]
    1.0 - math.exp(-2.0 * extremity)
