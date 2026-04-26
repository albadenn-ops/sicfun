package sicfun.holdem.runtime.metrics

import scala.util.Random

final case class ConfidenceInterval(lower: Double, upper: Double, level: Double):
  override def toString: String = f"CI${(level * 100).toInt}%d=[$lower%.2f, $upper%.2f]"

object MbbMetrics:

  /** Winrate in milli-big-blinds per 100 hands.
    *
    * Contract: `winningsPerHandInBB` values are BB per hand (positive = hero won).
    * Formula: mean_BB_per_hand × 1000 (BB→mbb) × 100 (per-100) = mean × 100_000.
    *
    * v1 shipped `mean * 1000` which was mbb/hand, not mbb/100; tests reproduced
    * the bug so CI passed. Do not regress. */
  def mbbPer100(winningsPerHandInBB: Vector[Double]): Double =
    if winningsPerHandInBB.isEmpty then 0.0
    else
      val mean = winningsPerHandInBB.sum / winningsPerHandInBB.size
      mean * 100_000.0

  def bootstrapIC95(
      winningsPerHand: Vector[Double],
      iterations: Int,
      rngSeed: Long
  ): ConfidenceInterval =
    val rng = new Random(rngSeed)
    val n = winningsPerHand.size
    val resamples = (1 to iterations).map { _ =>
      val sample = (1 to n).map(_ => winningsPerHand(rng.nextInt(n))).toVector
      mbbPer100(sample)
    }.sorted
    val lower = resamples((iterations * 0.025).toInt)
    val upper = resamples((iterations * 0.975).toInt.min(iterations - 1))
    ConfidenceInterval(lower, upper, 0.95)
