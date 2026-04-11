package sicfun.core

/** Immutable finite discrete distribution with non-negative finite weights.
  *
  * The structure does not force normalization on construction; callers can keep unnormalized
  * weights and call [[normalized]] when a proper probability mass function is needed.
  */
final case class DiscreteDistribution[A](weights: Map[A, Double]):
  import Probability.{Eps, isFiniteNonNegative}

  require(weights.values.forall(isFiniteNonNegative), "distribution weights must be finite and non-negative")

  /** Returns a normalized copy where weights sum to 1.0.
    *
    * If the distribution is already normalized (total weight within 1e-10 of 1.0),
    * returns `this` to avoid unnecessary allocation. This optimization matters because
    * normalization is called frequently during Bayesian update chains.
    *
    * @return a new distribution with weights summing to 1.0, or `this` if already normalized
    */
  def normalized: DiscreteDistribution[A] =
    val normalizedWeights = Probability.normalize(weights)
    if normalizedWeights eq weights then this
    else DiscreteDistribution(normalizedWeights)

  /** Returns the weight (probability, if normalized) of a specific element.
    *
    * @param a the element to query
    * @return the weight of `a`, or 0.0 if `a` is not in the support
    */
  inline def probabilityOf(a: A): Double = weights.getOrElse(a, 0.0)

  /** Returns the set of all elements with non-zero weight. */
  inline def support: Set[A] = weights.keySet

  /** Transforms the support elements by applying `f`, merging weights of elements
    * that map to the same output. Useful for coarsening a distribution (e.g.,
    * collapsing specific hands into hand categories).
    *
    * @param f the mapping function from the current support type to the new type
    * @return a new distribution over the image of `f`, with merged weights
    */
  def map[B](f: A => B): DiscreteDistribution[B] =
    val mapped = weights.toSeq.groupBy { case (a, _) => f(a) }
      .view
      .mapValues(_.map(_._2).sum)
      .toMap
    DiscreteDistribution(mapped)

  /** Performs a Bayesian update by multiplying each weight by a likelihood value.
    *
    * This is the core primitive for Bayesian inference: given a prior distribution and
    * a likelihood function P(evidence | hypothesis), produces the posterior distribution
    * P(hypothesis | evidence) via Bayes' rule. The result is automatically normalized.
    *
    * Returns both the updated (normalized) distribution and the marginal evidence
    * P(evidence) = sum_h P(evidence | h) * P(h), which is useful for model comparison
    * and log-evidence accumulation.
    *
    * @param likelihood a function returning P(evidence | hypothesis) for each hypothesis
    * @return a tuple of (normalized posterior distribution, marginal evidence)
    * @throws IllegalArgumentException if all likelihoods are zero (evidence is impossible)
    */
  def updateWithLikelihood(likelihood: A => Double): (DiscreteDistribution[A], Double) =
    // Single-pass: collect unnormalized weights into a fixed-size Array,
    // then build the normalized Map in one go (eliminates ArrayBuffer intermediate).
    val n = weights.size
    val keys = new Array[Any](n)
    val vals = new Array[Double](n)
    var evidence = 0.0
    var i = 0
    weights.foreach { case (a, w) =>
      val l = likelihood(a)
      require(isFiniteNonNegative(l), s"invalid likelihood for key '$a': $l")
      val updated = w * l
      keys(i) = a
      vals(i) = updated
      evidence += updated
      i += 1
    }
    require(evidence > Eps, "likelihoods produce zero evidence")
    val invEvidence = 1.0 / evidence
    val posterior = Map.newBuilder[A, Double]
    posterior.sizeHint(n)
    i = 0
    while i < n do
      posterior += keys(i).asInstanceOf[A] -> (vals(i) * invEvidence)
      i += 1
    (DiscreteDistribution(posterior.result()), evidence)

object DiscreteDistribution:
  /** Uniform distribution over distinct support values. */
  def uniform[A](values: Seq[A]): DiscreteDistribution[A] =
    val distinct = values.distinct
    require(distinct.nonEmpty, "uniform distribution requires non-empty support")
    val p = 1.0 / distinct.size
    DiscreteDistribution(distinct.map(_ -> p).toMap)
