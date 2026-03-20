package sicfun.holdem.engine

import sicfun.core.HandEvaluator
import sicfun.holdem.types.*
import sicfun.holdem.equity.HoldemCombinator

import scala.util.Random

/** Pure hand evaluation heuristics for poker decision engines.
  *
  * Extracted from TexasHoldemPlayingHall where these functions were shared
  * between the fast GTO path (fastGtoStrength) and archetype villain
  * decisions (streetStrength).
  */
private[holdem] object HandStrengthEstimator:

  def clamp(value: Double, lo: Double = 0.0, hi: Double = 1.0): Double =
    math.max(lo, math.min(hi, value))

  def preflopStrength(hand: HoleCards): Double =
    val r1 = hand.first.rank.value
    val r2 = hand.second.rank.value
    val high = math.max(r1, r2).toDouble / 14.0
    val low = math.min(r1, r2).toDouble / 14.0
    val pairBonus = if r1 == r2 then 0.30 + (high * 0.20) else 0.0
    val suitedBonus = if hand.first.suit == hand.second.suit then 0.06 else 0.0
    val gap = math.abs(r1 - r2)
    val connectorBonus =
      if gap == 0 then 0.0
      else if gap == 1 then 0.08
      else if gap == 2 then 0.04
      else 0.0
    clamp((0.45 * high) + (0.18 * low) + pairBonus + suitedBonus + connectorBonus)

  def bestCategoryStrength(hand: HoleCards, board: Board): Double =
    val cards = hand.toVector ++ board.cards
    cards.length match
      case 5 =>
        HandEvaluator.evaluate5Cached(cards).category.strength.toDouble / 8.0
      case 6 =>
        HoldemCombinator.combinations(cards.toIndexedSeq, 5).map { combo =>
          HandEvaluator.evaluate5Cached(combo).category.strength.toDouble / 8.0
        }.max
      case 7 =>
        HandEvaluator.evaluate7Cached(cards).category.strength.toDouble / 8.0
      case _ =>
        preflopStrength(hand)

  def drawPotential(hand: HoleCards, board: Board): Double =
    if board.cards.isEmpty then 0.0
    else
      val all = hand.toVector ++ board.cards
      val bySuit = all.groupBy(_.suit).view.mapValues(_.size).toMap
      val maxSuit = bySuit.values.max
      val flushDrawBonus =
        if maxSuit >= 5 then 0.12
        else if maxSuit == 4 then 0.08
        else if maxSuit == 3 && board.size <= 3 then 0.03
        else 0.0

      val ranks = all.map(_.rank.value).distinct.sorted
      val straightDrawBonus =
        if ranks.length >= 4 && hasTightRun(ranks) then 0.05
        else 0.0

      val pairWithBoardBonus =
        if board.cards.exists(card => card.rank == hand.first.rank || card.rank == hand.second.rank) then 0.04
        else 0.0

      flushDrawBonus + straightDrawBonus + pairWithBoardBonus

  def hasTightRun(sortedRanks: Seq[Int]): Boolean =
    if sortedRanks.length < 4 then false
    else
      val span4 = HoldemCombinator.combinations(sortedRanks.toIndexedSeq, 4).exists { combo =>
        combo.last - combo.head <= 4
      }
      val withWheelAce =
        if sortedRanks.contains(14) then
          val lowAce = sortedRanks.map(r => if r == 14 then 1 else r).sorted
          HoldemCombinator.combinations(lowAce.toIndexedSeq, 4).exists { combo =>
            combo.last - combo.head <= 4
          }
        else false
      span4 || withWheelAce

  /** Noisy hand strength for archetype villain decisions. Adds RNG jitter. */
  def streetStrength(
      hand: HoleCards,
      board: Board,
      street: Street,
      rng: Random
  ): Double =
    val pre = preflopStrength(hand)
    if street == Street.Preflop || board.cards.isEmpty then
      clamp(pre + (rng.nextDouble() - 0.5) * 0.04)
    else
      val categoryScore = bestCategoryStrength(hand, board)
      val drawBonus = drawPotential(hand, board)
      val noise = (rng.nextDouble() - 0.5) * 0.05
      clamp(0.45 * pre + 0.45 * categoryScore + drawBonus + noise)

  /** Deterministic hand strength for fast GTO heuristic path. No RNG noise. */
  def fastGtoStrength(hand: HoleCards, board: Board, street: Street): Double =
    val pre = preflopStrength(hand)
    if street == Street.Preflop || board.cards.isEmpty then pre
    else
      val madeWeight =
        street match
          case Street.Flop  => 0.50
          case Street.Turn  => 0.56
          case Street.River => 0.62
          case _            => 0.50
      val categoryScore = bestCategoryStrength(hand, board)
      val drawBonus = drawPotential(hand, board)
      clamp(((1.0 - madeWeight) * pre) + (madeWeight * categoryScore) + drawBonus)
