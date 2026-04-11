package sicfun.core

/** Measures how much a probability distribution "collapses" after observing evidence.
  *
  * These metrics quantify the information gain from Bayesian updates. A large collapse
  * indicates the evidence was highly informative, narrowing the distribution significantly.
  * All entropy-based quantities are computed in bits (log base 2) unless noted otherwise.
  */
object CollapseMetrics:
  final case class CollapseSummary(
      entropyReduction: Double,
      klDivergence: Double,
      effectiveSupportPrior: Double,
      effectiveSupportPosterior: Double,
      collapseRatio: Double
  )

  private inline val Eps = Probability.Eps
  private inline val Ln2 = 0.6931471805599453 // math.log(2.0), precomputed to avoid runtime call

  /** Computes all collapse metrics in one pass over normalized distributions.
    *
    * This avoids repeatedly normalizing/scanning the same distributions when
    * callers need the full collapse summary.
    */
  def summary[A](
      prior: DiscreteDistribution[A],
      posterior: DiscreteDistribution[A]
  ): CollapseSummary =
    val normalizedPrior = prior.normalized
    val normalizedPosterior = posterior.normalized

    val priorEntropyNats = entropyNats(normalizedPrior.weights.values)
    var posteriorEntropyNats = 0.0
    var kl = 0.0

    // Single pass over the posterior: simultaneously compute posterior entropy and KL divergence.
    // This avoids normalizing/iterating over the distributions multiple times.
    normalizedPosterior.weights.foreach { case (a, qProb) =>
      if qProb > 0.0 then
        val logQ = math.log(qProb)
        // Accumulate posterior entropy in nats: H = -sum(q * log(q))
        posteriorEntropyNats -= qProb * logQ
        if qProb > Eps then
          val pProb = normalizedPrior.probabilityOf(a)
          require(pProb > Eps, s"posterior has support outside prior for element $a")
          // KL divergence in bits: KL(q||p) = sum(q * (log(q) - log(p)) / ln(2))
          kl += qProb * ((logQ - math.log(pProb)) / Ln2)
    }

    // Convert nats to bits by dividing by ln(2).
    val priorEntropyBits = priorEntropyNats / Ln2
    val posteriorEntropyBits = posteriorEntropyNats / Ln2
    // Effective support (perplexity) = exp(entropy in nats).
    // A uniform distribution over N outcomes has effective support = N.
    val priorSupport = math.exp(priorEntropyNats)
    val posteriorSupport = math.exp(posteriorEntropyNats)
    val ratio =
      if priorSupport <= Eps then 0.0
      else math.max(0.0, 1.0 - posteriorSupport / priorSupport)

    CollapseSummary(
      entropyReduction = priorEntropyBits - posteriorEntropyBits,
      klDivergence = math.max(kl, 0.0),
      effectiveSupportPrior = priorSupport,
      effectiveSupportPosterior = posteriorSupport,
      collapseRatio = ratio
    )

  private def entropyNats(probabilities: Iterable[Double]): Double =
    var entropy = 0.0
    probabilities.foreach { p =>
      if p > 0.0 then entropy -= p * math.log(p)
    }
    entropy

  /** Computes the entropy reduction (information gain) from prior to posterior.
    *
    * Defined as H(prior) - H(posterior). A positive value means the posterior
    * is more concentrated (less uncertain) than the prior.
    *
    * @param prior  the distribution before observing evidence
    * @param posterior the distribution after observing evidence
    * @return entropy reduction in bits; may be negative if posterior is more diffuse
    */
  def entropyReduction[A](
      prior: DiscreteDistribution[A],
      posterior: DiscreteDistribution[A]
  ): Double =
    val priorEntropyNats = entropyNats(prior.normalized.weights.values)
    val posteriorEntropyNats = entropyNats(posterior.normalized.weights.values)
    (priorEntropyNats - posteriorEntropyNats) / Ln2

  /** Computes the Kullback-Leibler divergence KL(posterior || prior) in bits.
    *
    * Measures the expected number of extra bits needed to code samples from
    * the posterior using the prior as the coding distribution. Always non-negative;
    * zero iff the distributions are identical.
    *
    * @throws IllegalArgumentException if the posterior has support outside the prior
    *         (KL divergence would be infinite)
    */
  def klDivergence[A](
      prior: DiscreteDistribution[A],
      posterior: DiscreteDistribution[A]
  ): Double =
    val p = prior.normalized
    val q = posterior.normalized
    var kl = 0.0
    q.weights.foreach { case (a, qProb) =>
      if qProb > Eps then
        val pProb = p.probabilityOf(a)
        require(pProb > Eps, s"posterior has support outside prior for element $a")
        kl += qProb * (math.log(qProb / pProb) / Ln2)
    }
    // Clamp to zero to guard against floating-point rounding yielding tiny negatives.
    math.max(kl, 0.0)

  /** Computes the effective support size (perplexity) of a distribution.
    *
    * Defined as exp(H) where H is the Shannon entropy in nats. This gives the
    * "effective number of equally likely outcomes" -- e.g., a uniform distribution
    * over N outcomes has effective support N, while a point mass has effective support 1.
    *
    * @return a value in [1, |support|] indicating how spread out the distribution is
    */
  def effectiveSupport[A](distribution: DiscreteDistribution[A]): Double =
    val d = distribution.normalized
    math.exp(entropyNats(d.weights.values))

  /** Computes the collapse ratio: the fraction of effective support eliminated by evidence.
    *
    * Defined as 1 - effectiveSupport(posterior) / effectiveSupport(prior).
    * A value of 0 means no collapse (evidence was uninformative); a value near 1
    * means the posterior is concentrated on very few outcomes.
    *
    * @return a value in [0, 1]; clamped to 0 if the prior is already degenerate
    */
  def collapseRatio[A](
      prior: DiscreteDistribution[A],
      posterior: DiscreteDistribution[A]
  ): Double =
    val priorSize = effectiveSupport(prior)
    val posteriorSize = effectiveSupport(posterior)
    if priorSize <= Eps then 0.0
    else math.max(0.0, 1.0 - posteriorSize / priorSize)

  /** Computes the Brier score for a single probabilistic prediction.
    *
    * The Brier score is a proper scoring rule defined as the sum of squared
    * differences between predicted probabilities and the one-hot indicator for
    * the actual outcome. Lower is better: 0 for a perfect prediction,
    * 2 for the worst possible prediction on a binary outcome.
    *
    * @param predicted  probability vector over outcomes (should sum to 1)
    * @param actualIndex zero-based index of the outcome that actually occurred
    * @return the Brier score (non-negative)
    */
  def brierScore(predicted: Vector[Double], actualIndex: Int): Double =
    require(actualIndex >= 0 && actualIndex < predicted.length,
      s"actualIndex $actualIndex out of range [0, ${predicted.length})")
    var score = 0.0
    var i = 0
    while i < predicted.length do
      val indicator = if i == actualIndex then 1.0 else 0.0
      val diff = predicted(i) - indicator
      score += diff * diff
      i += 1
    score

  /** Computes the mean Brier score across a sequence of predictions.
    *
    * @param predictions sequence of (predicted probability vector, actual outcome index) pairs
    * @return the arithmetic mean of individual Brier scores
    * @throws IllegalArgumentException if predictions is empty
    */
  def meanBrierScore(predictions: Seq[(Vector[Double], Int)]): Double =
    require(predictions.nonEmpty, "meanBrierScore requires at least one prediction")
    predictions.map { case (pred, actual) => brierScore(pred, actual) }.sum / predictions.length
