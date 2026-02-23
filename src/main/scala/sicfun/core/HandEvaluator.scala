package sicfun.core

import java.util.concurrent.ConcurrentHashMap
import java.util.Arrays

/** Poker hand categories ordered by strength from weakest to strongest.
  *
  * Each variant carries an integer `strength` used for fast ordinal comparison.
  * The ordering matches standard poker rules: HighCard (0) < ... < StraightFlush (8).
  */
enum HandCategory(val strength: Int):
  case HighCard extends HandCategory(0)
  case OnePair extends HandCategory(1)
  case TwoPair extends HandCategory(2)
  case ThreeOfKind extends HandCategory(3)
  case Straight extends HandCategory(4)
  case Flush extends HandCategory(5)
  case FullHouse extends HandCategory(6)
  case FourOfKind extends HandCategory(7)
  case StraightFlush extends HandCategory(8)

/** A fully ordered poker hand ranking consisting of a category and tiebreak ranks.
  *
  * Comparison first checks [[HandCategory.strength]]; if equal, the `tiebreak`
  * vector is compared lexicographically. Tiebreak entries are rank values ordered
  * by descending significance (e.g., for TwoPair: high pair, low pair, kicker).
  *
  * @param category the hand category (e.g., Flush, FullHouse)
  * @param tiebreak rank values used to break ties within the same category
  */
final case class HandRank(category: HandCategory, tiebreak: Vector[Int]) extends Ordered[HandRank]:
  override def compare(that: HandRank): Int =
    val catCmp = this.category.strength.compare(that.category.strength)
    if catCmp != 0 then catCmp
    else lexicographicCompare(this.tiebreak, that.tiebreak)

  /** Element-wise comparison; shorter vectors are "less than" longer ones if all shared elements match. */
  private def lexicographicCompare(a: Vector[Int], b: Vector[Int]): Int =
    val len = math.min(a.length, b.length)
    var i = 0
    while i < len do
      val cmp = a(i).compare(b(i))
      if cmp != 0 then return cmp
      i += 1
    a.length.compare(b.length)

/** Companion providing a given [[Ordering]] instance so HandRank can be used
  * with standard library sort and collection operations.
  */
object HandRank:
  given Ordering[HandRank] with
    def compare(a: HandRank, b: HandRank): Int = a.compare(b)

/** Poker hand evaluation for 5-card and 7-card hands.
  *
  * Provides three evaluation strategies:
  *  - '''evaluate5 / evaluate7''': canonical evaluation returning a full [[HandRank]]
  *    (category + tiebreakers). Allocates intermediate collections.
  *  - '''evaluate5Cached / evaluate7Cached''': memoized variants backed by
  *    [[java.util.concurrent.ConcurrentHashMap]] for thread-safe, lock-free caching.
  *    Useful when the same hand may be evaluated repeatedly (e.g., equity simulations).
  *  - '''categorize5''': zero-allocation fast path that returns only the
  *    [[HandCategory]] without tiebreakers. Designed for bulk enumeration where
  *    only the category distribution is needed (see [[HandEvaluatorValidation]]).
  */
object HandEvaluator:
  // ConcurrentHashMap is chosen over a synchronized HashMap because hand evaluation
  // is a hot path in Monte Carlo simulations running across multiple threads.
  // ConcurrentHashMap provides lock-free reads and fine-grained locking on writes.
  private val cache5 = new ConcurrentHashMap[Long, HandRank]()
  private val cache7 = new ConcurrentHashMap[Long, HandRank]()
  // Pre-computed C(7,5) = 21 index combinations for selecting 5 cards from 7.
  private val combos7of5: Array[Array[Int]] = LookupTables.combos7of5

  /** Evaluates a 5-card poker hand, returning a full [[HandRank]] with tiebreakers.
    *
    * The evaluation groups cards by rank to determine the frequency pattern
    * (e.g., List(3,2) = FullHouse), then checks for flushes and straights.
    * Tiebreaker ranks are ordered so that the most significant group comes first.
    *
    * @param cards exactly 5 distinct cards
    * @return the hand's category and tiebreak vector for total ordering
    * @throws IllegalArgumentException if not exactly 5 distinct cards
    */
  def evaluate5(cards: Seq[Card]): HandRank =
    require(cards.length == 5, s"evaluate5 expects 5 cards, got ${cards.length}")
    require(cards.distinct.length == 5, "evaluate5 requires 5 distinct cards")

    val ranks = cards.map(_.rank.value)
    val suits = cards.map(_.suit)

    val isFlush = suits.distinct.length == 1
    val straightHighOpt = straightHigh(ranks)

    // Group ranks by frequency. Sort by (-count, -rank) so the dominant group
    // (and highest rank within equal counts) appears first.
    val groups = ranks.groupBy(identity).view.mapValues(_.size).toMap
    val countRank = groups.toList
      .map { case (rank, count) => (count, rank) }
      .sortBy { case (count, rank) => (-count, -rank) }

    // Extract just the counts to match against known frequency patterns.
    val pattern = countRank.map(_._1)

    (isFlush, straightHighOpt, pattern) match
      case (true, Some(high), _) =>
        HandRank(HandCategory.StraightFlush, Vector(high))
      case (_, _, List(4, 1)) =>
        val quad = countRank.head._2
        val kicker = countRank(1)._2
        HandRank(HandCategory.FourOfKind, Vector(quad, kicker))
      case (_, _, List(3, 2)) =>
        val trip = countRank.head._2
        val pair = countRank(1)._2
        HandRank(HandCategory.FullHouse, Vector(trip, pair))
      case (true, None, _) =>
        HandRank(HandCategory.Flush, ranks.sorted(using Ordering.Int.reverse).toVector)
      case (false, Some(high), _) =>
        HandRank(HandCategory.Straight, Vector(high))
      case (_, _, List(3, 1, 1)) =>
        val trip = countRank.head._2
        val kickers = countRank.tail.map(_._2).sorted(using Ordering.Int.reverse)
        HandRank(HandCategory.ThreeOfKind, (trip +: kickers).toVector)
      case (_, _, List(2, 2, 1)) =>
        val pairRanks = countRank.filter(_._1 == 2).map(_._2).sorted(using Ordering.Int.reverse)
        val kicker = countRank.find(_._1 == 1).get._2
        HandRank(HandCategory.TwoPair, (pairRanks :+ kicker).toVector)
      case (_, _, List(2, 1, 1, 1)) =>
        val pair = countRank.head._2
        val kickers = countRank.tail.map(_._2).sorted(using Ordering.Int.reverse)
        HandRank(HandCategory.OnePair, (pair +: kickers).toVector)
      case _ =>
        HandRank(HandCategory.HighCard, ranks.sorted(using Ordering.Int.reverse).toVector)

  /** Evaluates the best 5-card hand from 7 cards (Texas Hold'em style).
    *
    * Iterates over all C(7,5) = 21 possible 5-card subsets and returns
    * the highest-ranking one.
    *
    * @param cards exactly 7 distinct cards
    * @return the best possible [[HandRank]] among all 21 combinations
    * @throws IllegalArgumentException if not exactly 7 distinct cards
    */
  def evaluate7(cards: Seq[Card]): HandRank =
    require(cards.length == 7, s"evaluate7 expects 7 cards, got ${cards.length}")
    require(cards.distinct.length == 7, "evaluate7 requires 7 distinct cards")

    val indexed = cards.toIndexedSeq
    val first = combos7of5(0)
    var best = evaluate5(Vector(indexed(first(0)), indexed(first(1)), indexed(first(2)), indexed(first(3)), indexed(first(4))))
    var i = 1
    while i < combos7of5.length do
      val combo = combos7of5(i)
      val rank = evaluate5(Vector(indexed(combo(0)), indexed(combo(1)), indexed(combo(2)), indexed(combo(3)), indexed(combo(4))))
      if rank > best then best = rank
      i += 1
    best

  /** Cached variant of [[evaluate5]]. Returns a memoized result if available,
    * otherwise computes, caches, and returns the result.
    *
    * The cache key is a 30-bit Long encoding 5 sorted card IDs (6 bits each).
    * Thread safety is guaranteed by [[ConcurrentHashMap]].
    *
    * @see [[keyFor]] for the cache key encoding scheme
    */
  def evaluate5Cached(cards: Seq[Card]): HandRank =
    require(cards.length == 5, s"evaluate5Cached expects 5 cards, got ${cards.length}")
    require(cards.distinct.length == 5, "evaluate5Cached requires 5 distinct cards")
    val key = keyFor(cards)
    val cached = cache5.get(key)
    if cached != null then cached
    else
      val computed = evaluate5(cards)
      cache5.putIfAbsent(key, computed)
      computed

  /** Cached variant of [[evaluate7]]. Uses a 42-bit Long key (7 sorted card IDs x 6 bits). */
  def evaluate7Cached(cards: Seq[Card]): HandRank =
    require(cards.length == 7, s"evaluate7Cached expects 7 cards, got ${cards.length}")
    require(cards.distinct.length == 7, "evaluate7Cached requires 7 distinct cards")
    val key = keyFor(cards)
    val cached = cache7.get(key)
    if cached != null then cached
    else
      val computed = evaluate7(cards)
      cache7.putIfAbsent(key, computed)
      computed

  /** Category-only evaluation optimized for bulk enumeration.
    *
    * '''Zero object allocation''' on the hot path -- uses only primitive
    * comparisons and arithmetic. This is typically 10-50x faster than [[evaluate5]]
    * because it avoids Seq/Map creation, groupBy, and boxing.
    *
    * ==matchCount technique==
    * Instead of grouping ranks by frequency, we count the number of equal-rank
    * pairs among all C(5,2) = 10 pairwise comparisons. The count uniquely
    * identifies the rank frequency pattern:
    *
    * | matchCount | Pattern      | Category       |
    * |------------|-------------|----------------|
    * | 0          | 5 distinct  | HighCard/Straight/Flush/StraightFlush |
    * | 1          | one pair    | OnePair        |
    * | 2          | two pair    | TwoPair        |
    * | 3          | trips       | ThreeOfKind    |
    * | 4          | 3+2         | FullHouse      |
    * | 6          | quads       | FourOfKind     |
    *
    * Note: matchCount=5 is impossible with 5 cards (would require five-of-a-kind).
    *
    * ==Wheel detection==
    * When all 5 ranks are distinct (matchCount=0), a straight is detected by
    * checking `max - min == 4`. The A-2-3-4-5 wheel is a special case: its ranks
    * {14,2,3,4,5} have min=2, max=14, and sum=28, which is unique among all
    * 5-element subsets of {2..14} with 5 distinct values.
    *
    * @param r0 rank value of first card (2 to 14, where 14 = Ace)
    * @param r1 rank value of second card
    * @param r2 rank value of third card
    * @param r3 rank value of fourth card
    * @param r4 rank value of fifth card
    * @param isFlush true if all 5 cards share the same suit
    * @return the [[HandCategory]] without tiebreak information
    */
  def categorize5(r0: Int, r1: Int, r2: Int, r3: Int, r4: Int, isFlush: Boolean): HandCategory =
    // Count matching rank-pairs among the 5 cards (C(5,2) = 10 comparisons).
    // matchCount uniquely identifies the frequency pattern:
    //   0 → all distinct   1 → one pair   2 → two pair
    //   3 → three-of-a-kind   4 → full house   6 → four-of-a-kind
    var m = 0
    if r0 == r1 then m += 1
    if r0 == r2 then m += 1
    if r0 == r3 then m += 1
    if r0 == r4 then m += 1
    if r1 == r2 then m += 1
    if r1 == r3 then m += 1
    if r1 == r4 then m += 1
    if r2 == r3 then m += 1
    if r2 == r4 then m += 1
    if r3 == r4 then m += 1

    m match
      case 6 => HandCategory.FourOfKind
      case 4 => HandCategory.FullHouse
      case 3 => HandCategory.ThreeOfKind
      case 2 => HandCategory.TwoPair
      case 1 => HandCategory.OnePair
      case _ =>
        // All 5 ranks distinct — check for straight (consecutive or A-2-3-4-5 wheel)
        val mn = math.min(math.min(math.min(math.min(r0, r1), r2), r3), r4)
        val mx = math.max(math.max(math.max(math.max(r0, r1), r2), r3), r4)
        // Wheel: A(14)+2+3+4+5 = 28; with 5 distinct ranks, min=2, max=14, sum=28 is unique to the wheel.
        val isStraight = (mx - mn == 4) || (mx == 14 && mn == 2 && r0 + r1 + r2 + r3 + r4 == 28)
        if isFlush && isStraight then HandCategory.StraightFlush
        else if isFlush then HandCategory.Flush
        else if isStraight then HandCategory.Straight
        else HandCategory.HighCard

  /** Returns the current number of entries in the 5-card and 7-card caches. */
  def cacheSizes: (Int, Int) =
    (cache5.size(), cache7.size())

  /** Clears both evaluation caches. Useful for freeing memory between simulations. */
  def clearCaches(): Unit =
    cache5.clear()
    cache7.clear()

  /** Detects whether ranks form a straight and returns the high card if so.
    *
    * The wheel (A-2-3-4-5) is treated as a 5-high straight, not Ace-high.
    *
    * @return Some(highRank) if a straight is found, None otherwise
    */
  private def straightHigh(ranks: Seq[Int]): Option[Int] =
    val distinct = ranks.distinct.sorted
    if distinct.length != 5 then None
    else
      // Wheel: Ace plays low as the bottom of A-2-3-4-5; high card is 5.
      val wheel = distinct == List(2, 3, 4, 5, 14)
      if wheel then Some(5)
      else if isConsecutive(distinct) then Some(distinct.last)
      else None

  /** Checks whether a sorted ascending sequence has no gaps (each element = predecessor + 1). */
  private def isConsecutive(sortedAsc: Seq[Int]): Boolean =
    var i = 0
    while i < sortedAsc.length - 1 do
      if sortedAsc(i + 1) != sortedAsc(i) + 1 then return false
      i += 1
    true

  /** Encodes a set of cards into a single Long for use as a cache key.
    *
    * Each card is mapped to a unique integer ID (0..51) via [[CardId.toId]].
    * The IDs are sorted so that the same hand in any order produces the same key.
    * Each ID occupies 6 bits (sufficient for 0..63), and they are packed left-to-right
    * into a Long. For 5 cards this uses 30 bits; for 7 cards, 42 bits -- both fit
    * comfortably within a 64-bit Long, avoiding any object allocation for the key.
    */
  private def keyFor(cards: Seq[Card]): Long =
    val ids = new Array[Int](cards.length)
    var i = 0
    cards.foreach { card =>
      ids(i) = CardId.toId(card)
      i += 1
    }
    Arrays.sort(ids) // Sort so hand order doesn't affect the key
    var key = 0L
    var j = 0
    while j < ids.length do
      key = (key << 6) | (ids(j).toLong & 0x3fL) // Pack 6 bits per card ID
      j += 1
    key
