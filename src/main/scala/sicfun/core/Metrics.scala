package sicfun.core

/** Statistical and information-theoretic utility functions used across inference modules. */
object Metrics:
  private inline val Eps = Probability.Eps

  def entropy(probabilities: Iterable[Double], base: Double = 2.0): Double =
    require(base > 0.0 && base != 1.0, "entropy base must be positive and not 1")
    val logBase = math.log(base)
    probabilities.foldLeft(0.0) { (acc, p) =>
      if p <= 0.0 then acc else acc - (p * math.log(p) / logBase)
    }

  def conditionalEntropy(weightedActionDists: Seq[(Double, Iterable[Double])], base: Double = 2.0): Double =
    val totalWeight = weightedActionDists.map(_._1).sum
    require(totalWeight > Eps, "total weight must be positive")
    weightedActionDists.map { case (weight, probs) =>
      val w = weight / totalWeight
      w * entropy(probs, base)
    }.sum

  def mean(values: Iterable[Double]): Double =
    val seq = values.toSeq
    require(seq.nonEmpty, "mean requires non-empty values")
    seq.sum / seq.length

  def variance(values: Iterable[Double]): Double =
    val seq = values.toSeq
    require(seq.length > 1, "variance requires at least two values")
    val m = mean(seq)
    seq.map(v => math.pow(v - m, 2)).sum / (seq.length - 1)

  def weightedMean(values: Iterable[Double], weights: Iterable[Double]): Double =
    val v = values.toSeq
    val w = weights.toSeq
    require(v.length == w.length && v.nonEmpty, "weightedMean requires aligned values and weights")
    val total = w.sum
    require(total > Eps, "weights must sum to a positive value")
    (v zip w).map { case (value, weight) => value * weight }.sum / total

  def weightedVariance(values: Iterable[Double], weights: Iterable[Double]): Double =
    val v = values.toSeq
    val w = weights.toSeq
    require(v.length == w.length && v.length > 1, "weightedVariance requires aligned values and weights")
    val meanValue = weightedMean(v, w)
    val total = w.sum
    require(total > Eps, "weights must sum to a positive value")
    (v zip w).map { case (value, weight) => weight * math.pow(value - meanValue, 2) }.sum / total
