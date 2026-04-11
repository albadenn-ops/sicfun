package sicfun.holdem.engine

import sicfun.core.{Card, DiscreteDistribution, Rank}
import sicfun.holdem.history.ShowdownRecord
import sicfun.holdem.types.*

/** Coarse hand classification used to group showdown observations into prior-bias buckets.
  *
  * This enum partitions all 1326 possible hole card combinations into five categories
  * based on simple rank/suit/connectivity criteria. The classification is used by
  * [[ShowdownPriorBias]] to compute similarity between a candidate hand and historically
  * observed showdown hands. Hands in the same class get a small similarity bonus (0.1),
  * which is much weaker than an exact match (1.0) or same-pair-rank match (0.3).
  *
  * Categories:
  *   - PremiumPair: pairs JJ+ (high pocket pairs)
  *   - Broadway: both cards Ten or higher (e.g., KQo, ATo)
  *   - SuitedConnector: suited with rank gap <= 2 (e.g., T9s, J9s)
  *   - Speculative: pairs below JJ, suited aces, suited low cards
  *   - Weak: everything else (off-suit junk, disconnected low cards)
  */
enum ShowdownHandClass:
  case PremiumPair
  case Broadway
  case SuitedConnector
  case Speculative
  case Weak

object ShowdownHandClass:
  /** Classifies a hand into one of five coarse buckets.
    * Priority order: PremiumPair > Broadway > SuitedConnector > Speculative > Weak.
    */
  def classify(hand: HoleCards): ShowdownHandClass =
    if isPremiumPair(hand) then ShowdownHandClass.PremiumPair
    else if isBroadway(hand) then ShowdownHandClass.Broadway
    else if isSuitedConnector(hand) then ShowdownHandClass.SuitedConnector
    else if isSpeculative(hand) then ShowdownHandClass.Speculative
    else ShowdownHandClass.Weak

  def isPair(hand: HoleCards): Boolean =
    hand.first.rank == hand.second.rank

  private def isPremiumPair(hand: HoleCards): Boolean =
    isPair(hand) && hand.first.rank.value >= Rank.Jack.value

  private def isBroadway(hand: HoleCards): Boolean =
    hand.first.rank.value >= Rank.Ten.value &&
      hand.second.rank.value >= Rank.Ten.value

  private def isSuitedConnector(hand: HoleCards): Boolean =
    isSuited(hand) && rankGap(hand) <= 2

  private def isSpeculative(hand: HoleCards): Boolean =
    isPair(hand) ||
      (isSuited(hand) && highestRankValue(hand) == Rank.Ace.value) ||
      (isSuited(hand) && highestRankValue(hand) <= Rank.Nine.value)

  private def isSuited(hand: HoleCards): Boolean =
    hand.first.suit == hand.second.suit

  private def rankGap(hand: HoleCards): Int =
    math.abs(hand.first.rank.value - hand.second.rank.value)

  private def highestRankValue(hand: HoleCards): Int =
    math.max(hand.first.rank.value, hand.second.rank.value)

/** Applies a conservative showdown-informed bias on top of a context prior range.
  *
  * When we have observed villain showdown hands from previous hands in the session,
  * this module adjusts the prior range to slightly upweight hands that are similar
  * to what the villain has previously shown down. This captures the tendency for
  * players to have consistent style preferences (e.g., a villain who has shown down
  * premium pairs is slightly more likely to hold premium pairs again).
  *
  * The bias is deliberately conservative:
  *   - Requires at least [[MinShowdowns]] (3) observations before applying any bias.
  *   - The blend weight caps at 15% even with 50+ showdowns (see [[blendWeight]]).
  *   - Individual hand multipliers are capped at [[MaxBiasMultiplier]] (1.5x).
  *   - The biased distribution is blended with the original prior, then renormalized.
  *
  * This avoids overfitting to small showdown samples while still extracting useful
  * information from repeated observations of the same villain.
  */
object ShowdownPriorBias:
  /** Minimum number of showdown records required before bias is applied. */
  val MinShowdowns = 3
  /** Maximum multiplier any single hand can receive from showdown similarity. */
  val MaxBiasMultiplier = 1.5

  /** Computes the blend weight for mixing the biased prior with the original prior.
    *
    * Linear ramp from 0% at 0 showdowns to 15% at 50+ showdowns, capped at 15%.
    * Formula: min(showdownCount / 50.0, 0.15).
    *
    * @param showdownCount number of observed showdown records
    * @return blend weight in [0, 0.15]
    */
  def blendWeight(showdownCount: Int): Double =
    math.min(math.max(showdownCount, 0).toDouble / 50.0, 0.15)

  /** Applies showdown-informed bias to a prior range.
    *
    * Flow:
    *   1. Filter out hands containing dead cards (hero cards + board cards).
    *   2. If fewer than [[MinShowdowns]] observations, return the filtered prior unchanged.
    *   3. Compute a biased distribution by multiplying each hand's weight by its
    *      [[biasMultiplierFor]] (based on similarity to showdown history).
    *   4. Blend the biased distribution with the original filtered prior using
    *      [[blendWeight]], then renormalize.
    *
    * @param prior the base villain range distribution
    * @param showdowns historical showdown records for this villain
    * @param deadCards cards that cannot appear in villain's hand (hero + board)
    * @return the bias-adjusted and normalized prior
    */
  def applyBias(
      prior: DiscreteDistribution[HoleCards],
      showdowns: Vector[ShowdownRecord],
      deadCards: Set[Card] = Set.empty
  ): DiscreteDistribution[HoleCards] =
    val filteredPrior = filterDeadCards(prior, deadCards)
    if showdowns.size < MinShowdowns then filteredPrior
    else
      val weight = blendWeight(showdowns.size)
      if weight <= 0.0 then filteredPrior
      else
        val biased = DiscreteDistribution(
          filteredPrior.weights.map { case (hand, baseWeight) =>
            hand -> (baseWeight * biasMultiplierFor(hand, showdowns))
          }
        ).normalized
        val blended = DiscreteDistribution(
          filteredPrior.weights.keysIterator.map { hand =>
            hand -> (
              ((1.0 - weight) * filteredPrior.probabilityOf(hand)) +
                (weight * biased.probabilityOf(hand))
            )
          }.toMap
        )
        blended.normalized

  /** Removes hands containing any dead card from the prior and renormalizes. */
  private def filterDeadCards(
      prior: DiscreteDistribution[HoleCards],
      deadCards: Set[Card]
  ): DiscreteDistribution[HoleCards] =
    if deadCards.isEmpty then prior
    else
      val filtered = prior.weights.collect {
        case (hand, weight)
            if weight > 0.0 &&
              !deadCards.contains(hand.first) &&
              !deadCards.contains(hand.second) =>
          hand -> weight
      }
      require(filtered.nonEmpty, "showdown bias removed all prior support")
      DiscreteDistribution(filtered).normalized

  /** Computes the bias multiplier for a candidate hand based on its average similarity
    * to all observed showdown hands. Capped at [[MaxBiasMultiplier]].
    *
    * The multiplier is 1.0 + meanSimilarity, so a hand with zero similarity to all
    * showdowns gets multiplier 1.0 (no change), while a hand identical to all showdowns
    * gets multiplier min(2.0, MaxBiasMultiplier) = 1.5.
    */
  private def biasMultiplierFor(
      hand: HoleCards,
      showdowns: Vector[ShowdownRecord]
  ): Double =
    val meanSimilarity =
      showdowns.iterator.map(record => similarity(hand, record.cards)).sum / showdowns.size.toDouble
    math.min(1.0 + meanSimilarity, MaxBiasMultiplier)

  /** Pairwise similarity between a candidate hand and a showdown-observed hand.
    *
    * Returns:
    *   - 1.0 for exact same hand (both cards identical)
    *   - 0.3 for same-rank pair (e.g., AhAs vs AcAd)
    *   - 0.1 for same ShowdownHandClass (e.g., both are SuitedConnectors)
    *   - 0.0 otherwise
    */
  private def similarity(
      hand: HoleCards,
      shown: HoleCards
  ): Double =
    if hand == shown then 1.0
    else if samePairRank(hand, shown) then 0.3
    else if ShowdownHandClass.classify(hand) == ShowdownHandClass.classify(shown) then 0.1
    else 0.0

  /** Checks if both hands are pairs of the same rank (e.g., KhKs and KcKd). */
  private def samePairRank(a: HoleCards, b: HoleCards): Boolean =
    ShowdownHandClass.isPair(a) &&
      ShowdownHandClass.isPair(b) &&
      a.first.rank == b.first.rank
