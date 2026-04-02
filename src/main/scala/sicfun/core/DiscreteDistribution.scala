package sicfun.core

/** Immutable finite discrete distribution with non-negative finite weights.
  *
  * The structure does not force normalization on construction; callers can keep unnormalized
  * weights and call [[normalized]] when a proper probability mass function is needed.
  */
final case class DiscreteDistribution[A](weights: Map[A, Double]):
  import Probability.{Eps, isFiniteNonNegative}

  require(weights.values.forall(isFiniteNonNegative), "distribution weights must be finite and non-negative")

  def normalized: DiscreteDistribution[A] =
    val normalizedWeights = Probability.normalize(weights)
    if normalizedWeights eq weights then this
    else DiscreteDistribution(normalizedWeights)

  inline def probabilityOf(a: A): Double = weights.getOrElse(a, 0.0)

  inline def support: Set[A] = weights.keySet

  def map[B](f: A => B): DiscreteDistribution[B] =
    val mapped = weights.toSeq.groupBy { case (a, _) => f(a) }
      .view
      .mapValues(_.map(_._2).sum)
      .toMap
    DiscreteDistribution(mapped)

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
