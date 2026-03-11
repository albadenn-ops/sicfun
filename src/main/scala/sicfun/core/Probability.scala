package sicfun.core

/** A single outcome in a discrete probability distribution.
  *
  * @param probability the probability of this outcome occurring (should be in [0, 1])
  * @param value the numeric payoff or score associated with this outcome
  */
final case class Outcome(probability: Double, value: Double)

/** Utility methods for basic probability computations. */
object Probability:
  /** Tolerance threshold for numerical comparisons to avoid floating-point artifacts. */
  private[sicfun] inline val Eps = 1e-12

  private[core] inline def isFiniteNonNegative(value: Double): Boolean =
    java.lang.Double.isFinite(value) && value >= 0.0

/** Computes the expected value (weighted average) of a set of outcomes.
    *
    * Requires that the probabilities of all outcomes sum to 1 (within tolerance 1e-9).
    *
    * @param outcomes a sequence of (probability, value) pairs forming a complete distribution
    * @return the expected value: sum of `probability * value` over all outcomes
    * @throws IllegalArgumentException if probabilities do not sum to 1
    */
  def expectedValue(outcomes: Seq[Outcome]): Double =
    require(outcomes.nonEmpty, "outcomes must be non-empty")
    outcomes.foreach { outcome =>
      require(
        isFiniteNonNegative(outcome.probability),
        s"invalid probability: ${outcome.probability}"
      )
      require(java.lang.Double.isFinite(outcome.value), s"value must be finite, got ${outcome.value}")
    }
    val totalP = outcomes.map(_.probability).sum
    require(math.abs(totalP - 1.0) < 1e-9, s"probabilities must sum to 1, got $totalP")
    outcomes.map(o => o.probability * o.value).sum

  /** Normalizes a map of non-negative weights so that the values sum to 1.
    *
    * @tparam A the key type
    * @param weights a map from keys to non-negative weights
    * @return a new map with the same keys where values are divided by the total weight
    * @throws IllegalArgumentException if the total weight is effectively zero
    */
  def normalize[A](weights: Map[A, Double]): Map[A, Double] =
    var total = 0.0
    weights.foreach { case (key, weight) =>
      require(isFiniteNonNegative(weight), s"invalid weight for key '$key': $weight")
      total += weight
    }
    require(total > Eps, "cannot normalize empty or zero-sum weights")
    // Skip Map creation when weights are already normalized (common case
    // for distributions that were previously normalized).
    if math.abs(total - 1.0) <= 1e-10 then weights
    else
      val invTotal = 1.0 / total
      weights.view.mapValues(_ * invTotal).toMap

/** The result of a Monte Carlo estimation run.
  *
  * @param mean the estimated mean of the sampled quantity
  * @param variance the sample variance (using Bessel's correction, i.e., dividing by n-1)
  * @param stderr the standard error of the mean estimate (sqrt(variance / n))
  * @param trials the number of samples drawn
  */
final case class Estimate(mean: Double, variance: Double, stderr: Double, trials: Int)

object MonteCarlo:
  def estimateMean(trials: Int)(sample: => Double): Estimate =
    require(trials > 1, "trials must be > 1")
    var mean = 0.0
    var m2 = 0.0
    var i = 0
    while i < trials do
      val x = sample
      val delta = x - mean
      mean += delta / (i + 1)
      val delta2 = x - mean
      m2 += delta * delta2
      i += 1
    val variance = m2 / (trials - 1)
    val stderr = math.sqrt(variance / trials)
    Estimate(mean, variance, stderr, trials)

