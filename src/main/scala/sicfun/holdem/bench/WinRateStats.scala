package sicfun.holdem.bench

import sicfun.core.Metrics
import java.util.Random

/** Pure statistics for win-rate measurement. No poker dependencies. */
object WinRateStats:

  /** A bb/100 point estimate with a bootstrap percentile confidence interval. */
  final case class WinRateCI(
      pointEstimate: Double,
      lower: Double,
      upper: Double,
      ciLevel: Double,
      sampleSize: Int,
      resamples: Int
  )

  /** bb/100 = mean per-hand bb result * 100. Per-hand values are in bb (1 chip = 1 bb in the normalized hall). */
  def bbPer100(perHandBb: Vector[Double]): Double =
    if perHandBb.isEmpty then 0.0 else Metrics.mean(perHandBb) * 100.0

  /** Bootstrap percentile CI on bb/100 over independent per-hand results.
    * Deterministic given `seed`. Resampling reduces ESTIMATOR variance; it does not
    * remove the game's intrinsic variance (see the design doc's determinism section).
    */
  def bbPer100CI(
      perHandBb: Vector[Double],
      resamples: Int = 2000,
      ciLevel: Double = 0.95,
      seed: Long = 42L
  ): WinRateCI =
    require(resamples > 0, "resamples must be positive")
    require(ciLevel > 0.0 && ciLevel < 1.0, "ciLevel must be in (0,1)")
    val n = perHandBb.length
    val point = bbPer100(perHandBb)
    if n == 0 then WinRateCI(0.0, 0.0, 0.0, ciLevel, 0, resamples)
    else
      val rng = new Random(seed)
      val means = new Array[Double](resamples)
      var b = 0
      while b < resamples do
        var sum = 0.0
        var i = 0
        while i < n do
          sum += perHandBb(rng.nextInt(n))
          i += 1
        means(b) = (sum / n.toDouble) * 100.0
        b += 1
      val tail = (1.0 - ciLevel) / 2.0
      val resampleMeans = means.toVector
      WinRateCI(
        pointEstimate = point,
        lower = Metrics.percentile(resampleMeans, tail),
        upper = Metrics.percentile(resampleMeans, 1.0 - tail),
        ciLevel = ciLevel,
        sampleSize = n,
        resamples = resamples
      )
