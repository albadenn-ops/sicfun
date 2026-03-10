package sicfun.holdem.analysis
import sicfun.holdem.types.*
import sicfun.holdem.equity.*

import sicfun.core.{CollapseMetrics, DiscreteDistribution, Metrics}

/** Summary statistics for hero equity computed over a posterior distribution of villain hands.
  *
  * @param mean      weighted mean equity across all valid villain hands in the posterior
  * @param variance  weighted variance of per-hand equities (zero when only one villain hand is valid)
  * @param stderr    standard error of the mean, estimated as `sqrt(variance / effectiveSupport)`
  * @param handCount number of valid (non-overlapping, positive-weight) villain hands evaluated
  */
final case class EvEstimate(mean: Double, variance: Double, stderr: Double, handCount: Int)

/** Equity variance analysis across a Bayesian posterior over villain holdings.
  *
  * Given a hero hand, a board, and a posterior distribution over villain hole cards,
  * computes per-hand exact equity and aggregates weighted mean, variance, and standard
  * error. This is useful for measuring how "spread out" hero's equity is across the
  * opponent's range — a high variance indicates that the hand plays very differently
  * against different parts of the range.
  */
object EvAnalysis:

  /** Computes weighted equity statistics for `hero` against every hand in the `posterior`.
    *
    * For each villain hand ''h'' with positive weight that does not overlap hero or board
    * cards, computes exact heads-up equity. The results are aggregated into a weighted
    * mean, variance, and standard error using [[CollapseMetrics.effectiveSupport]] as
    * the effective sample size.
    *
    * @param hero      the hero's hole cards
    * @param board     current community cards (may be empty for preflop analysis)
    * @param posterior Bayesian posterior over villain hole cards (need not be normalized)
    * @return an [[EvEstimate]] summarizing equity distribution across the posterior
    * @throws IllegalArgumentException if no valid villain hands remain after dead-card filtering
    */
  def evVariance(
      hero: HoleCards,
      board: Board,
      posterior: DiscreteDistribution[HoleCards]
  ): EvEstimate =
    val norm = posterior.normalized
    val dead = hero.asSet ++ board.asSet
    // Remove villain hands that overlap with hero or board cards, or have zero weight.
    val validHands = norm.weights.toSeq.filter { case (h, w) =>
      w > 0.0 && !h.asSet.exists(dead.contains)
    }
    require(validHands.nonEmpty, "posterior has no valid villain hands after filtering dead cards")

    // Compute exact equity for hero vs each individual villain hand.
    val equities = validHands.map { case (villainHand, _) =>
      val singleDist = DiscreteDistribution(Map(villainHand -> 1.0))
      HoldemEquity.equityExact(hero, board, singleDist).equity
    }
    val weights = validHands.map(_._2)

    val mean = Metrics.weightedMean(equities, weights)
    val variance = if validHands.length > 1 then
      Metrics.weightedVariance(equities, weights)
    else 0.0

    // Use effective support (inverse of Herfindahl index) as effective sample size
    // for standard error estimation. This correctly downweights posteriors dominated
    // by a few high-weight hands.
    val effectiveN = CollapseMetrics.effectiveSupport(
      DiscreteDistribution(validHands.toMap)
    )
    val stderr = if effectiveN > 1.0 then math.sqrt(variance / effectiveN) else 0.0

    EvEstimate(mean, variance, stderr, validHands.length)
