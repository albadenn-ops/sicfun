package sicfun.core

/** Statistical and information-theoretic utility functions used across inference modules. */
object Metrics:
  private inline val Eps = Probability.Eps

  /** Computes the Shannon entropy of a probability distribution.
    *
    * H = -sum(p_i * log(p_i)) / log(base)
    *
    * Zero-probability entries are skipped (0 * log(0) is treated as 0, consistent
    * with the limit as p -> 0+). The caller is responsible for passing a valid
    * probability distribution (non-negative values summing to 1).
    *
    * @param probabilities the probability mass function values
    * @param base the logarithm base (default 2.0 for bits; use e for nats)
    * @return the Shannon entropy in the specified base
    */
  def entropy(probabilities: Iterable[Double], base: Double = 2.0): Double =
    require(base > 0.0 && base != 1.0, "entropy base must be positive and not 1")
    val logBase = math.log(base)
    probabilities.foldLeft(0.0) { (acc, p) =>
      if p <= 0.0 then acc else acc - (p * math.log(p) / logBase)
    }

  /** Computes the weighted conditional entropy H(Y | X) = sum_x P(x) * H(Y | X=x).
    *
    * Each element in `weightedActionDists` is a (weight, distribution) pair representing
    * the conditional distribution of Y given a specific value of X, along with the
    * probability weight of that X value.
    *
    * In the poker context, X is the hidden hand and Y is the observed action, so this
    * computes how unpredictable the action remains when the hand is known.
    *
    * @param weightedActionDists sequence of (weight, conditional probability distribution) pairs
    * @param base logarithm base (default 2.0 for bits)
    * @return the weighted average entropy
    */
  def conditionalEntropy(weightedActionDists: Seq[(Double, Iterable[Double])], base: Double = 2.0): Double =
    val totalWeight = weightedActionDists.map(_._1).sum
    require(totalWeight > Eps, "total weight must be positive")
    weightedActionDists.map { case (weight, probs) =>
      val w = weight / totalWeight
      w * entropy(probs, base)
    }.sum

  /** Computes the arithmetic mean of a non-empty collection of values.
    *
    * @param values the values to average
    * @return the arithmetic mean
    * @throws IllegalArgumentException if values is empty
    */
  def mean(values: Iterable[Double]): Double =
    val seq = values.toSeq
    require(seq.nonEmpty, "mean requires non-empty values")
    seq.sum / seq.length

  /** Computes the sample variance using Bessel's correction (dividing by n-1).
    *
    * @param values at least two values
    * @return the unbiased sample variance
    * @throws IllegalArgumentException if fewer than two values are provided
    */
  def variance(values: Iterable[Double]): Double =
    val seq = values.toSeq
    require(seq.length > 1, "variance requires at least two values")
    val m = mean(seq)
    seq.map(v => math.pow(v - m, 2)).sum / (seq.length - 1)

  /** Computes the weighted arithmetic mean: sum(w_i * v_i) / sum(w_i).
    *
    * @param values  the values to average
    * @param weights non-negative weights aligned with values
    * @return the weighted mean
    * @throws IllegalArgumentException if inputs are empty, misaligned, or weights sum to zero
    */
  def weightedMean(values: Iterable[Double], weights: Iterable[Double]): Double =
    val v = values.toSeq
    val w = weights.toSeq
    require(v.length == w.length && v.nonEmpty, "weightedMean requires aligned values and weights")
    val total = w.sum
    require(total > Eps, "weights must sum to a positive value")
    (v zip w).map { case (value, weight) => value * weight }.sum / total

  /** Computes the weighted variance: sum(w_i * (v_i - mean)^2) / sum(w_i).
    *
    * This is the population-style weighted variance (divides by total weight, not n-1).
    *
    * @param values  the values
    * @param weights non-negative weights aligned with values
    * @return the weighted variance
    * @throws IllegalArgumentException if inputs have fewer than 2 elements, are misaligned,
    *         or weights sum to zero
    */
  def weightedVariance(values: Iterable[Double], weights: Iterable[Double]): Double =
    val v = values.toSeq
    val w = weights.toSeq
    require(v.length == w.length && v.length > 1, "weightedVariance requires aligned values and weights")
    val meanValue = weightedMean(v, w)
    val total = w.sum
    require(total > Eps, "weights must sum to a positive value")
    (v zip w).map { case (value, weight) => weight * math.pow(value - meanValue, 2) }.sum / total
