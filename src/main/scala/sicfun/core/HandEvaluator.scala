package sicfun.core

import sicfun.holdem.types.ScopedRuntimeProperties

import java.util.concurrent.ConcurrentHashMap

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

/** A fully ordered poker hand ranking packed into a single Int for zero-allocation comparison.
  *
  * Encoding: `(categoryStrength << 20) | (tb0 << 16) | (tb1 << 12) | (tb2 << 8) | (tb3 << 4) | tb4`
  *
  * 4 bits per tiebreak rank (values 0-14), 4 bits for category (values 0-8).
  * Natural Int ordering matches hand strength ordering (higher packed value = better hand).
  *
  * @param packed the packed integer encoding category and tiebreak ranks
  */
final case class HandRank(packed: Int) extends Ordered[HandRank]:
  override def compare(that: HandRank): Int =
    Integer.compare(this.packed, that.packed)

  def category: HandCategory =
    HandCategory.fromOrdinal((packed >>> 20) & 0x0f)

  /** Number of significant tiebreak entries for this hand's category. */
  def tiebreakLength: Int =
    ((packed >>> 20) & 0x0f) match
      case 0 | 5 => 5 // HighCard, Flush
      case 1     => 4 // OnePair
      case 2 | 3 => 3 // TwoPair, ThreeOfKind
      case 4 | 8 => 1 // Straight, StraightFlush
      case 6 | 7 => 2 // FullHouse, FourOfKind
      case _     => 0

  /** Returns the i-th tiebreak rank (0-indexed, most significant first). */
  def tiebreak(i: Int): Int =
    (packed >>> (16 - i * 4)) & 0x0f

  /** Returns the tiebreak ranks as a Vector (for backward compatibility / tests). */
  def tiebreakVector: Vector[Int] =
    val len = tiebreakLength
    val builder = Vector.newBuilder[Int]
    builder.sizeHint(len)
    var i = 0
    while i < len do
      builder += tiebreak(i)
      i += 1
    builder.result()

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
  *    (category + tiebreakers).
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
  private val Cache5MaxSizeProperty = "sicfun.handEvaluator.cache5.maxSize"
  private val Cache5MaxSizeEnv = "sicfun_HAND_EVALUATOR_CACHE5_MAX_SIZE"
  private val Cache7MaxSizeProperty = "sicfun.handEvaluator.cache7.maxSize"
  private val Cache7MaxSizeEnv = "sicfun_HAND_EVALUATOR_CACHE7_MAX_SIZE"
  private val DefaultMaxCache5Size = 250_000
  private val DefaultMaxCache7Size = 750_000
  private val cache5 = new ConcurrentHashMap[Long, HandRank]()
  private val cache7 = new ConcurrentHashMap[Long, HandRank]()
  private val rankCountScratch = new ThreadLocal[Array[Int]]:
    override def initialValue(): Array[Int] = new Array[Int](15)
  private val uniqueRankScratch = new ThreadLocal[Array[Int]]:
    override def initialValue(): Array[Int] = new Array[Int](5)

  /** Evaluates a 5-card poker hand, returning a full [[HandRank]] with tiebreakers.
    *
    * Tiebreaker ranks are ordered so that the most significant group comes first.
    *
    * @param cards exactly 5 distinct cards
    * @return the hand's category and tiebreak vector for total ordering
    * @throws IllegalArgumentException if not exactly 5 distinct cards
    */
  def evaluate5(cards: Seq[Card]): HandRank =
    val size = cards.length
    require(size == 5, s"evaluate5 expects 5 cards, got $size")
    val iterator = cards.iterator
    val c0 = iterator.next()
    val c1 = iterator.next()
    val c2 = iterator.next()
    val c3 = iterator.next()
    val c4 = iterator.next()
    require(!iterator.hasNext, s"evaluate5 expects 5 cards, got $size")
    require(allDistinct5(c0, c1, c2, c3, c4), "evaluate5 requires 5 distinct cards")
    unpackRank(evaluate5Packed(c0, c1, c2, c3, c4))

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
    val size = cards.length
    require(size == 7, s"evaluate7 expects 7 cards, got $size")
    val iterator = cards.iterator
    val c0 = iterator.next()
    val c1 = iterator.next()
    val c2 = iterator.next()
    val c3 = iterator.next()
    val c4 = iterator.next()
    val c5 = iterator.next()
    val c6 = iterator.next()
    require(!iterator.hasNext, s"evaluate7 expects 7 cards, got $size")
    require(allDistinct7(c0, c1, c2, c3, c4, c5, c6), "evaluate7 requires 7 distinct cards")

    unpackRank(evaluate7PackedDirect(c0, c1, c2, c3, c4, c5, c6))

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
      val maxEntries = configuredCacheLimit(Cache5MaxSizeProperty, Cache5MaxSizeEnv, DefaultMaxCache5Size)
      if maxEntries > 0 then
        if cache5.size() >= maxEntries then cache5.clear()
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
      val maxEntries = configuredCacheLimit(Cache7MaxSizeProperty, Cache7MaxSizeEnv, DefaultMaxCache7Size)
      if maxEntries > 0 then
        if cache7.size() >= maxEntries then cache7.clear()
        cache7.putIfAbsent(key, computed)
      computed

  /** Fast packed 7-card evaluation for callers that already guarantee distinct cards.
    *
    * This bypasses Seq validation, cache lookup, and HandRank allocation.
    */
  private[sicfun] def evaluate7PackedDirect(
      c0: Card,
      c1: Card,
      c2: Card,
      c3: Card,
      c4: Card,
      c5: Card,
      c6: Card
  ): Int =
    var bestPacked = evaluate5Packed(c0, c1, c2, c3, c4)

    inline def update(packed: Int): Unit =
      if packed > bestPacked then bestPacked = packed

    update(evaluate5Packed(c0, c1, c2, c3, c5))
    update(evaluate5Packed(c0, c1, c2, c3, c6))
    update(evaluate5Packed(c0, c1, c2, c4, c5))
    update(evaluate5Packed(c0, c1, c2, c4, c6))
    update(evaluate5Packed(c0, c1, c2, c5, c6))
    update(evaluate5Packed(c0, c1, c3, c4, c5))
    update(evaluate5Packed(c0, c1, c3, c4, c6))
    update(evaluate5Packed(c0, c1, c3, c5, c6))
    update(evaluate5Packed(c0, c1, c4, c5, c6))
    update(evaluate5Packed(c0, c2, c3, c4, c5))
    update(evaluate5Packed(c0, c2, c3, c4, c6))
    update(evaluate5Packed(c0, c2, c3, c5, c6))
    update(evaluate5Packed(c0, c2, c4, c5, c6))
    update(evaluate5Packed(c0, c3, c4, c5, c6))
    update(evaluate5Packed(c1, c2, c3, c4, c5))
    update(evaluate5Packed(c1, c2, c3, c4, c6))
    update(evaluate5Packed(c1, c2, c3, c5, c6))
    update(evaluate5Packed(c1, c2, c4, c5, c6))
    update(evaluate5Packed(c1, c3, c4, c5, c6))
    update(evaluate5Packed(c2, c3, c4, c5, c6))
    bestPacked

  private inline def allDistinct5(c0: Card, c1: Card, c2: Card, c3: Card, c4: Card): Boolean =
    c0 != c1 &&
      c0 != c2 &&
      c0 != c3 &&
      c0 != c4 &&
      c1 != c2 &&
      c1 != c3 &&
      c1 != c4 &&
      c2 != c3 &&
      c2 != c4 &&
      c3 != c4

  private inline def allDistinct7(
      c0: Card,
      c1: Card,
      c2: Card,
      c3: Card,
      c4: Card,
      c5: Card,
      c6: Card
  ): Boolean =
    allDistinct5(c0, c1, c2, c3, c4) &&
      c5 != c0 &&
      c5 != c1 &&
      c5 != c2 &&
      c5 != c3 &&
      c5 != c4 &&
      c6 != c0 &&
      c6 != c1 &&
      c6 != c2 &&
      c6 != c3 &&
      c6 != c4 &&
      c6 != c5

  private inline def packRank(categoryStrength: Int, t0: Int, t1: Int, t2: Int, t3: Int, t4: Int): Int =
    (((((categoryStrength << 4) | t0) << 4 | t1) << 4 | t2) << 4 | t3) << 4 | t4

  private inline def unpackRank(packed: Int): HandRank =
    HandRank(packed)

  private def evaluate5Packed(c0: Card, c1: Card, c2: Card, c3: Card, c4: Card): Int =
    val r0 = c0.rank.value
    val r1 = c1.rank.value
    val r2 = c2.rank.value
    val r3 = c3.rank.value
    val r4 = c4.rank.value

    val isFlush =
      c0.suit == c1.suit &&
        c1.suit == c2.suit &&
        c2.suit == c3.suit &&
        c3.suit == c4.suit

    val rankCounts = rankCountScratch.get()
    val uniqueRanks = uniqueRankScratch.get()
    var uniqueCount = 0

    if rankCounts(r0) == 0 then
      uniqueRanks(uniqueCount) = r0
      uniqueCount += 1
    rankCounts(r0) += 1

    if rankCounts(r1) == 0 then
      uniqueRanks(uniqueCount) = r1
      uniqueCount += 1
    rankCounts(r1) += 1

    if rankCounts(r2) == 0 then
      uniqueRanks(uniqueCount) = r2
      uniqueCount += 1
    rankCounts(r2) += 1

    if rankCounts(r3) == 0 then
      uniqueRanks(uniqueCount) = r3
      uniqueCount += 1
    rankCounts(r3) += 1

    if rankCounts(r4) == 0 then
      uniqueRanks(uniqueCount) = r4
      uniqueCount += 1
    rankCounts(r4) += 1

    val packed =
      if uniqueCount == 5 then
        val mn = math.min(math.min(math.min(math.min(r0, r1), r2), r3), r4)
        val mx = math.max(math.max(math.max(math.max(r0, r1), r2), r3), r4)
        val sum = r0 + r1 + r2 + r3 + r4
        val wheel = mx == 14 && mn == 2 && sum == 28
        val isStraight = (mx - mn == 4) || wheel
        if isStraight then
          val high = if wheel then 5 else mx
          if isFlush then packRank(HandCategory.StraightFlush.strength, high, 0, 0, 0, 0)
          else packRank(HandCategory.Straight.strength, high, 0, 0, 0, 0)
        else
          var i = 1
          while i < uniqueCount do
            val value = uniqueRanks(i)
            var j = i - 1
            while j >= 0 && uniqueRanks(j) < value do
              uniqueRanks(j + 1) = uniqueRanks(j)
              j -= 1
            uniqueRanks(j + 1) = value
            i += 1
          if isFlush then
            packRank(
              HandCategory.Flush.strength,
              uniqueRanks(0),
              uniqueRanks(1),
              uniqueRanks(2),
              uniqueRanks(3),
              uniqueRanks(4)
            )
          else
            packRank(
              HandCategory.HighCard.strength,
              uniqueRanks(0),
              uniqueRanks(1),
              uniqueRanks(2),
              uniqueRanks(3),
              uniqueRanks(4)
            )
      else if uniqueCount == 4 then
        var pair = 0
        var s0 = 0
        var s1 = 0
        var s2 = 0
        var singles = 0
        var i = 0
        while i < uniqueCount do
          val rank = uniqueRanks(i)
          if rankCounts(rank) == 2 then pair = rank
          else
            if singles == 0 then s0 = rank
            else if singles == 1 then s1 = rank
            else s2 = rank
            singles += 1
          i += 1

        if s0 < s1 then
          val tmp = s0
          s0 = s1
          s1 = tmp
        if s1 < s2 then
          val tmp = s1
          s1 = s2
          s2 = tmp
        if s0 < s1 then
          val tmp = s0
          s0 = s1
          s1 = tmp
        packRank(HandCategory.OnePair.strength, pair, s0, s1, s2, 0)
      else if uniqueCount == 3 then
        var trip = 0
        var k0 = 0
        var k1 = 0
        var kickers = 0
        var p0 = 0
        var p1 = 0
        var pairs = 0
        var kicker = 0
        var i = 0
        while i < uniqueCount do
          val rank = uniqueRanks(i)
          rankCounts(rank) match
            case 3 =>
              trip = rank
            case 2 =>
              if pairs == 0 then p0 = rank else p1 = rank
              pairs += 1
            case _ =>
              if kickers == 0 then k0 = rank else k1 = rank
              kickers += 1
              kicker = rank
          i += 1

        if trip != 0 then
          if k0 < k1 then
            val tmp = k0
            k0 = k1
            k1 = tmp
          packRank(HandCategory.ThreeOfKind.strength, trip, k0, k1, 0, 0)
        else
          if p0 < p1 then
            val tmp = p0
            p0 = p1
            p1 = tmp
          packRank(HandCategory.TwoPair.strength, p0, p1, kicker, 0, 0)
      else
        val ra = uniqueRanks(0)
        val rb = uniqueRanks(1)
        val ca = rankCounts(ra)
        if ca == 4 || ca == 1 then
          val quad = if ca == 4 then ra else rb
          val kicker = if ca == 1 then ra else rb
          packRank(HandCategory.FourOfKind.strength, quad, kicker, 0, 0, 0)
        else
          val trip = if ca == 3 then ra else rb
          val pair = if ca == 2 then ra else rb
          packRank(HandCategory.FullHouse.strength, trip, pair, 0, 0, 0)

    var clearIndex = 0
    while clearIndex < uniqueCount do
      rankCounts(uniqueRanks(clearIndex)) = 0
      clearIndex += 1
    packed

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
  inline def categorize5(r0: Int, r1: Int, r2: Int, r3: Int, r4: Int, isFlush: Boolean): HandCategory =
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

  /** Encodes a set of cards into a single Long for use as a cache key.
    *
    * Each card maps to bit position [[CardId.toId]] (0..51). The final key is
    * the OR of those bits, so card order never affects the key.
    */
  private inline def keyFor(cards: Seq[Card]): Long =
    var key = 0L
    cards.foreach { card =>
      key |= 1L << CardId.toId(card)
    }
    key

  private def configuredCacheLimit(property: String, env: String, default: Int): Int =
    ScopedRuntimeProperties
      .get(property)
      .flatten
      .orElse(sys.props.get(property))
      .orElse(sys.env.get(env))
      .flatMap(_.trim.toIntOption)
      .filter(_ >= 0)
      .getOrElse(default)
