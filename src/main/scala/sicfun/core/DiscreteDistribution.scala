package sicfun.core

final case class DiscreteDistribution[A](weights: Map[A, Double]):
  import Probability.{Eps, isFiniteNonNegative}

  require(weights.values.forall(isFiniteNonNegative), "distribution weights must be finite and non-negative")

  def normalized: DiscreteDistribution[A] =
    DiscreteDistribution(Probability.normalize(weights))

  inline def probabilityOf(a: A): Double = weights.getOrElse(a, 0.0)

  inline def support: Set[A] = weights.keySet

  def map[B](f: A => B): DiscreteDistribution[B] =
    val mapped = weights.toSeq.groupBy { case (a, _) => f(a) }
      .view
      .mapValues(_.map(_._2).sum)
      .toMap
    DiscreteDistribution(mapped)

  def updateWithLikelihood(likelihood: A => Double): (DiscreteDistribution[A], Double) =
    val raw = scala.collection.mutable.ArrayBuffer.empty[(A, Double)]
    var evidence = 0.0
    weights.foreach { case (a, w) =>
      val l = likelihood(a)
      require(isFiniteNonNegative(l), s"invalid likelihood for key '$a': $l")
      val updated = w * l
      raw += (a -> updated)
      evidence += updated
    }
    require(evidence > Eps, "likelihoods produce zero evidence")
    val invEvidence = 1.0 / evidence
    val posterior = Map.newBuilder[A, Double]
    raw.foreach { case (a, updated) =>
      posterior += a -> (updated * invEvidence)
    }
    (DiscreteDistribution(posterior.result()), evidence)

object DiscreteDistribution:
  def uniform[A](values: Seq[A]): DiscreteDistribution[A] =
    val distinct = values.distinct
    require(distinct.nonEmpty, "uniform distribution requires non-empty support")
    val p = 1.0 / distinct.size
    DiscreteDistribution(distinct.map(_ -> p).toMap)
