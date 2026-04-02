package sicfun.holdem.engine

import sicfun.core.{Card, DiscreteDistribution, Rank}
import sicfun.holdem.history.ShowdownRecord
import sicfun.holdem.types.*

/** Coarse hand classes used to map showdown observations into prior-bias buckets. */
enum ShowdownHandClass:
  case PremiumPair
  case Broadway
  case SuitedConnector
  case Speculative
  case Weak

object ShowdownHandClass:
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
  * The bias is bounded and blended with the original prior to avoid overfitting to small
  * showdown samples.
  */
object ShowdownPriorBias:
  val MinShowdowns = 3
  val MaxBiasMultiplier = 1.5

  def blendWeight(showdownCount: Int): Double =
    math.min(math.max(showdownCount, 0).toDouble / 50.0, 0.15)

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

  private def biasMultiplierFor(
      hand: HoleCards,
      showdowns: Vector[ShowdownRecord]
  ): Double =
    val meanSimilarity =
      showdowns.iterator.map(record => similarity(hand, record.cards)).sum / showdowns.size.toDouble
    math.min(1.0 + meanSimilarity, MaxBiasMultiplier)

  private def similarity(
      hand: HoleCards,
      shown: HoleCards
  ): Double =
    if hand == shown then 1.0
    else if samePairRank(hand, shown) then 0.3
    else if ShowdownHandClass.classify(hand) == ShowdownHandClass.classify(shown) then 0.1
    else 0.0

  private def samePairRank(a: HoleCards, b: HoleCards): Boolean =
    ShowdownHandClass.isPair(a) &&
      ShowdownHandClass.isPair(b) &&
      a.first.rank == b.first.rank
