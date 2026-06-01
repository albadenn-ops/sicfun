package sicfun.holdem.strategic.safety

import sicfun.holdem.types.{PokerAction, Street}
import sicfun.holdem.strategic.types.StrategicClass
import sicfun.holdem.strategic.state.{PublicState, Sizing}

/** Board-blind floor implementation of the Def 9 real baseline: returns the
  * configured `(class, action)` constant, or 0.25 when absent. This reproduces
  * EXACTLY the pre-P1 behavior (`actionPriors.getOrElse((cls,cat), 0.25)`), so wiring
  * it in with no calibration artifact is a guaranteed no-op.
  */
final class ConstantRealBaseline(
    actionPriors: Map[(StrategicClass, PokerAction.Category), Double]
) extends RealBaseline:
  def probability(
      cls: StrategicClass,
      action: PokerAction.Category,
      sizing: Option[Sizing],
      publicState: PublicState
  ): Double =
    actionPriors.getOrElse((cls, action), 0.25)

/** Calibrated, board/street-conditioned Def 9 baseline.
  *
  * Builds Laplace-smoothed action distributions at two granularities from raw observed
  * counts, then answers `probability` with a backoff ladder that guarantees a valid
  * distribution in every spot and is never worse than the `fallback` constants floor.
  */
final class RealBaselineImpl(
    artifact: BaselineArtifact,
    minCount: Int,
    alpha: Double,
    fallback: RealBaseline
) extends RealBaseline:

  private val actions = PokerAction.Category.values

  // (class, bucket, street) -> total observed
  private val cellGroupTotal: Map[(StrategicClass, String, Street), Long] =
    artifact.counts.groupMapReduce { case ((c, b, s, _), _) => (c, b, s) } { case (_, n) => n }(_ + _)

  // (class, street, action) -> total over all buckets
  private val streetActionCount: Map[(StrategicClass, Street, PokerAction.Category), Long] =
    artifact.counts.groupMapReduce { case ((c, _, s, a), _) => (c, s, a) } { case (_, n) => n }(_ + _)

  private val streetGroupTotal: Map[(StrategicClass, Street), Long] =
    streetActionCount.groupMapReduce { case ((c, s, _), _) => (c, s) } { case (_, n) => n }(_ + _)

  private def smoothed(count: Long, groupTotal: Long): Double =
    (count + alpha) / (groupTotal + alpha * actions.length)

  def probability(
      cls: StrategicClass,
      action: PokerAction.Category,
      sizing: Option[Sizing],
      publicState: PublicState
  ): Double =
    val bucketToken = BoardBucket.token(BoardBucket.of(publicState))
    val street = publicState.street
    val cellTotal = cellGroupTotal.getOrElse((cls, bucketToken, street), 0L)
    if cellTotal >= minCount then
      val n = artifact.counts.getOrElse((cls, bucketToken, street, action), 0L)
      smoothed(n, cellTotal)
    else
      val streetTotal = streetGroupTotal.getOrElse((cls, street), 0L)
      if streetTotal >= minCount then
        val n = streetActionCount.getOrElse((cls, street, action), 0L)
        smoothed(n, streetTotal)
      else
        fallback.probability(cls, action, sizing, publicState)
