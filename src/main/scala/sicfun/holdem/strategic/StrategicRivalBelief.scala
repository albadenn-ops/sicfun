package sicfun.holdem.strategic

import sicfun.core.DiscreteDistribution

/** Concrete RivalBeliefState backed by a DiscreteDistribution over StrategicClass.
  *
  * This is the M type parameter for Dynamics[M], KernelProfile[M], etc.
  */
final case class StrategicRivalBelief(
    typePosterior: DiscreteDistribution[StrategicClass]
) extends RivalBeliefState:

  /** Identity update — real Bayesian update happens via kernel pipeline in Dynamics.fullStep(). */
  // REDUCTIONISM: identity pass-through — real update happens in kernel pipeline StateEmbeddingUpdater
  def update(signal: ActionSignal, publicState: PublicState): StrategicRivalBelief =
    this

  /** Convert belief to WPomcp particle arrays.
    *
    * Samples numParticles particles from the type posterior.
    * Each particle gets (type=StrategicClass.ordinal, privState=handBucket, weight=1/N).
    */
  def toParticles(numParticles: Int, handBucket: Int): (Array[Int], Array[Double]) =
    val types = new Array[Int](numParticles)
    val weights = new Array[Double](numParticles)
    val classes = StrategicClass.values
    val uniformWeight = 1.0 / numParticles.toDouble
    var idx = 0
    for cls <- classes do
      val prob = typePosterior.probabilityOf(cls)
      val count = math.round(prob * numParticles).toInt
      var j = 0
      while j < count && idx < numParticles do
        types(idx) = cls.ordinal
        weights(idx) = uniformWeight
        idx += 1
        j += 1
    // Fill remaining slots with MAP class
    val mapClass = classes.maxBy(typePosterior.probabilityOf)
    while idx < numParticles do
      types(idx) = mapClass.ordinal
      weights(idx) = uniformWeight
      idx += 1
    (types, weights)

object StrategicRivalBelief:
  /** Create with uniform prior over all four strategic classes. */
  def uniform: StrategicRivalBelief =
    StrategicRivalBelief(DiscreteDistribution(
      StrategicClass.values.map(c => c -> 0.25).toMap
    ))

  /** StateEmbeddingUpdater for StrategicRivalBelief. */
  val updater: StateEmbeddingUpdater[StrategicRivalBelief] =
    (state: StrategicRivalBelief, posterior: DiscreteDistribution[StrategicClass]) =>
      StrategicRivalBelief(posterior)
