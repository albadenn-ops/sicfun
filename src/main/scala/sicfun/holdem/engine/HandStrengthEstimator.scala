package sicfun.holdem.engine

import sicfun.core.HandEvaluator
import sicfun.holdem.types.*
import sicfun.holdem.equity.HoldemCombinator

import scala.util.Random

/** Pure hand evaluation heuristics for poker decision engines.
  *
  * This object provides lightweight hand-strength estimates that combine preflop
  * heuristics (rank bonuses, suitedness, connectivity) with postflop made-hand
  * category scoring and draw potential. It serves two distinct consumers:
  *
  *   1. '''Fast GTO path''' ([[fastGtoStrength]]): deterministic, no RNG noise.
  *      Used by [[GtoSolveEngine]]'s fast mode to make quick threshold-based decisions.
  *   2. '''Archetype villain decisions''' ([[streetStrength]]): adds small RNG jitter
  *      to simulate realistic hand-reading uncertainty. Used by [[ArchetypeVillainResponder]].
  *
  * All strength values are normalized to [0, 1] via [[clamp]].
  *
  * Design decisions:
  *   - Preflop strength is a weighted linear combination of high card, low card, pair,
  *     suited, and connector bonuses. This is a coarse heuristic (not a preflop equity
  *     lookup) chosen for speed.
  *   - Postflop strength blends preflop strength with a made-hand category score (from
  *     the 5/7-card evaluator) plus draw potential bonuses. The blend weight shifts
  *     toward the made-hand score on later streets.
  *
  * Extracted from TexasHoldemPlayingHall where these functions were shared
  * between the fast GTO path and archetype villain decisions.
  */
private[holdem] object HandStrengthEstimator:

  /** Clamps a value to the range [lo, hi]. Defaults to [0, 1]. */
  def clamp(value: Double, lo: Double = 0.0, hi: Double = 1.0): Double =
    math.max(lo, math.min(hi, value))

  /** Heuristic preflop hand strength in [0, 1].
    *
    * Combines weighted high-card and low-card contributions with additive bonuses
    * for pairs (scaled by rank), suitedness, and connectedness (gap <= 2).
    * This is intentionally a fast approximation, not an exact preflop equity table.
    *
    * @param hand the two hole cards
    * @return a strength estimate in [0, 1] where ~0.95 is pocket aces and ~0.3 is 72o
    */
  def preflopStrength(hand: HoleCards): Double =
    val r1 = hand.first.rank.value
    val r2 = hand.second.rank.value
    // Normalize rank values to [0, 1] (Ace=14 -> 1.0, Two=2 -> ~0.14)
    val high = math.max(r1, r2).toDouble / 14.0
    val low = math.min(r1, r2).toDouble / 14.0
    // Pair bonus is large (0.30 base) and scales with rank (high pairs are stronger)
    val pairBonus = if r1 == r2 then 0.30 + (high * 0.20) else 0.0
    // Suited hands have ~3% more equity on average; 0.06 bonus reflects this
    val suitedBonus = if hand.first.suit == hand.second.suit then 0.06 else 0.0
    val gap = math.abs(r1 - r2)
    // Connector bonus: adjacent ranks can make straights more easily
    val connectorBonus =
      if gap == 0 then 0.0      // pairs already have their own bonus
      else if gap == 1 then 0.08 // one-gappers (e.g., T9, 87)
      else if gap == 2 then 0.04 // two-gappers (e.g., T8, 86)
      else 0.0
    clamp((0.45 * high) + (0.18 * low) + pairBonus + suitedBonus + connectorBonus)

  /** Best made-hand category strength from all possible 5-card combinations.
    *
    * Uses the cached hand evaluator to find the best hand category (high card through
    * straight flush) and normalizes to [0, 1] by dividing by 8.0 (the max category ordinal).
    *
    * For 6-card boards (turn with 2 hole + 4 board), enumerates all C(6,5)=6 combinations.
    * For 7-card boards (river), uses the optimized 7-card evaluator directly.
    *
    * Falls back to [[preflopStrength]] if no board cards are available.
    *
    * @param hand the hole cards
    * @param board the community cards
    * @return a category-based strength in [0, 1]
    */
  def bestCategoryStrength(hand: HoleCards, board: Board): Double =
    val cards = hand.toVector ++ board.cards
    cards.length match
      case 5 =>
        HandEvaluator.evaluate5Cached(cards).category.strength.toDouble / 8.0
      case 6 =>
        // Enumerate all C(6,5) = 6 possible 5-card hands and take the best category
        HoldemCombinator.combinations(cards.toIndexedSeq, 5).map { combo =>
          HandEvaluator.evaluate5Cached(combo).category.strength.toDouble / 8.0
        }.max
      case 7 =>
        HandEvaluator.evaluate7Cached(cards).category.strength.toDouble / 8.0
      case _ =>
        // No board or unexpected card count; fall back to preflop heuristic
        preflopStrength(hand)

  /** Estimates draw potential (flush draws, straight draws, pair-with-board).
    *
    * Returns an additive bonus in [0, ~0.17] that is added to the base strength
    * to reflect the equity value of incomplete draws. Only applies postflop.
    *
    * Bonuses:
    *   - Flush draw: 0.12 for made flush (5+ suited), 0.08 for 4-flush, 0.03 for 3-flush on flop
    *   - Straight draw: 0.05 if any 4 ranks span <= 4 (open-ended or gutshot)
    *   - Pair with board: 0.04 if either hole card pairs a board card
    *
    * @param hand the hole cards
    * @param board the community cards (returns 0 if empty)
    * @return the draw potential bonus
    */
  def drawPotential(hand: HoleCards, board: Board): Double =
    if board.cards.isEmpty then 0.0
    else
      val all = hand.toVector ++ board.cards
      // Flush draw detection: count cards by suit
      val bySuit = all.groupBy(_.suit).view.mapValues(_.size).toMap
      val maxSuit = bySuit.values.max
      val flushDrawBonus =
        if maxSuit >= 5 then 0.12       // Made flush
        else if maxSuit == 4 then 0.08  // 4-flush (one card away)
        else if maxSuit == 3 && board.size <= 3 then 0.03 // 3-flush on flop only (2 cards to come)
        else 0.0

      // Straight draw detection: check if any 4-rank subset spans <= 4 ranks
      val ranks = all.map(_.rank.value).distinct.sorted
      val straightDrawBonus =
        if ranks.length >= 4 && hasTightRun(ranks) then 0.05
        else 0.0

      // Pair-with-board bonus: hole card matching a board card
      val pairWithBoardBonus =
        if board.cards.exists(card => card.rank == hand.first.rank || card.rank == hand.second.rank) then 0.04
        else 0.0

      flushDrawBonus + straightDrawBonus + pairWithBoardBonus

  /** Checks whether any 4-element subset of sorted ranks spans at most 4 values.
    *
    * This detects open-ended straight draws (span=3), gutshot draws (span=4),
    * and made straights (span=4 with 5 consecutive). Also handles wheel-ace
    * straights by remapping Ace (14) to 1 and re-checking.
    *
    * @param sortedRanks distinct rank values in ascending order
    * @return true if a tight 4-card run exists
    */
  def hasTightRun(sortedRanks: Seq[Int]): Boolean =
    if sortedRanks.length < 4 then false
    else
      // Check all C(n,4) combinations for a span <= 4
      val span4 = HoldemCombinator.combinations(sortedRanks.toIndexedSeq, 4).exists { combo =>
        combo.last - combo.head <= 4
      }
      // Also check with Ace treated as low (value=1) for wheel straights (A-2-3-4-5)
      val withWheelAce =
        if sortedRanks.contains(14) then
          val lowAce = sortedRanks.map(r => if r == 14 then 1 else r).sorted
          HoldemCombinator.combinations(lowAce.toIndexedSeq, 4).exists { combo =>
            combo.last - combo.head <= 4
          }
        else false
      span4 || withWheelAce

  /** Noisy hand strength for archetype villain decisions. Adds RNG jitter.
    *
    * Preflop: returns preflopStrength with small uniform noise (+/- 0.02).
    * Postflop: blends preflop (45%), made-hand category (45%), and draw bonus,
    * with uniform noise (+/- 0.025). The noise simulates imperfect hand-reading.
    *
    * @param hand the hole cards
    * @param board the community cards
    * @param street the current betting street
    * @param rng source of randomness for jitter
    * @return a noisy strength estimate in [0, 1]
    */
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
      // Equal 45/45 blend of preflop and postflop category, plus draw bonus and noise
      clamp(0.45 * pre + 0.45 * categoryScore + drawBonus + noise)

  /** Street-dependent blend weights for made-hand category vs preflop strength.
    * Higher weight = more emphasis on the made-hand category score.
    * Initial estimates pending calibration from equity table regression.
    */
  val defaultBlendWeights: Map[Street, Double] = Map(
    Street.Flop  -> 0.50,
    Street.Turn  -> 0.56,
    Street.River -> 0.62
  )

  /** Deterministic hand strength for the fast GTO heuristic path. No RNG noise.
    *
    * Similar to [[streetStrength]] but fully deterministic for reproducible GTO decisions.
    * The blend weight between preflop and made-hand category shifts toward the category
    * score on later streets (50% flop -> 56% turn -> 62% river), reflecting the
    * increasing importance of board texture as more cards are dealt.
    *
    * @param hand the hole cards
    * @param board the community cards
    * @param street the current betting street
    * @param blendWeights per-street made-hand category weight; defaults to [[defaultBlendWeights]]
    * @return a deterministic strength estimate in [0, 1]
    */
  def fastGtoStrength(
      hand: HoleCards, board: Board, street: Street,
      blendWeights: Map[Street, Double] = defaultBlendWeights
  ): Double =
    val pre = preflopStrength(hand)
    if street == Street.Preflop || board.cards.isEmpty then pre
    else
      // Weight of made-hand category increases on later streets
      val madeWeight = blendWeights.getOrElse(street, 0.50)
      val categoryScore = bestCategoryStrength(hand, board)
      val drawBonus = drawPotential(hand, board)
      clamp(((1.0 - madeWeight) * pre) + (madeWeight * categoryScore) + drawBonus)
