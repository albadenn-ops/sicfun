package sicfun.core

/** Exhaustive combinatorial validation helpers for [[HandEvaluator]] category frequencies.
  *
  * This object provides the machinery to verify that [[HandEvaluator.categorize5]] produces
  * the mathematically correct number of each hand category when exhaustively enumerating
  * all C(52,5) = 2,598,960 possible five-card hands from a standard deck.
  *
  * The expected counts are well-known combinatorial constants from poker probability theory.
  * Any deviation indicates a bug in the evaluator's classification logic.
  *
  * This validation is the primary correctness proof for the hand evaluator and runs as
  * part of the test suite (see [[HandEvaluatorValidationTest]]).
  */
object HandEvaluatorValidation:
  /** C(52,5) = 52! / (5! * 47!) = 2,598,960 total distinct 5-card hands. */
  val TotalFiveCardHands: Long = 2_598_960L

  /** The mathematically exact number of 5-card hands for each poker category.
    * These are derived from combinatorial counting (not simulation) and serve as
    * the ground truth for evaluator validation.
    */
  val ExpectedFiveCardCategoryCounts: Map[HandCategory, Long] = Map(
    HandCategory.HighCard -> 1_302_540L,
    HandCategory.OnePair -> 1_098_240L,
    HandCategory.TwoPair -> 123_552L,
    HandCategory.ThreeOfKind -> 54_912L,
    HandCategory.Straight -> 10_200L,
    HandCategory.Flush -> 5_108L,
    HandCategory.FullHouse -> 3_744L,
    HandCategory.FourOfKind -> 624L,
    HandCategory.StraightFlush -> 40L
  )

  /** A count of hands observed per [[HandCategory]], produced by exhaustive enumeration. */
  final case class CategoryDistribution(counts: Map[HandCategory, Long]):
    def count(category: HandCategory): Long = counts.getOrElse(category, 0L)

    lazy val total: Long = HandCategory.values.iterator.map(count).sum

  /** The result of comparing an actual category distribution against the expected one.
    * Contains a list of mismatched categories (if any) and a convenience `isExactMatch` flag.
    */
  final case class DistributionCheckResult(
      actual: CategoryDistribution,
      expected: CategoryDistribution
  ):
    lazy val mismatches: Vector[(HandCategory, Long, Long)] =
      HandCategory.values.toVector.flatMap { category =>
        val actualCount = actual.count(category)
        val expectedCount = expected.count(category)
        Option.when(actualCount != expectedCount)((category, actualCount, expectedCount))
      }

    def isExactMatch: Boolean = mismatches.isEmpty

  val ExpectedFiveCardDistribution: CategoryDistribution =
    CategoryDistribution(ExpectedFiveCardCategoryCounts)

  /** Exhaustively enumerates all C(n,5) five-card hands from the given deck and counts
    * the number of hands in each [[HandCategory]].
    *
    * Uses [[HandEvaluator.categorize5]] (the zero-allocation fast path) for efficiency.
    * The five nested while-loops generate all 5-card combinations in lexicographic order
    * without allocating intermediate collections.
    *
    * Pre-extracts rank values and suit ordinals into primitive arrays to avoid
    * repeated Card field access inside the ~2.6M iteration hot loop.
    *
    * @param deck the cards to enumerate over (default: standard 52-card deck)
    * @return a [[CategoryDistribution]] with the observed counts per category
    */
  def evaluateFiveCardCategoryDistribution(deck: IndexedSeq[Card] = Deck.full): CategoryDistribution =
    require(deck.length >= 5, s"deck must contain at least 5 cards, got ${deck.length}")
    require(deck.distinct.length == deck.length, "deck must not contain duplicate cards")

    val counts = Array.fill[Long](HandCategory.values.length)(0L)
    val n = deck.length

    // Pre-extract rank values and suit ordinals into arrays to avoid
    // Card field access and object allocation inside the hot loop.
    val ranks = new Array[Int](n)
    val suits = new Array[Int](n)
    var i = 0
    while i < n do
      ranks(i) = deck(i).rank.value
      suits(i) = deck(i).suit.ordinal
      i += 1

    var a = 0
    while a <= n - 5 do
      val ra = ranks(a); val sa = suits(a)
      var b = a + 1
      while b <= n - 4 do
        val rb = ranks(b); val sb = suits(b)
        var c = b + 1
        while c <= n - 3 do
          val rc = ranks(c); val sc = suits(c)
          var d = c + 1
          while d <= n - 2 do
            val rd = ranks(d); val sd = suits(d)
            var e = d + 1
            while e <= n - 1 do
              val isFlush = sa == sb && sb == sc && sc == sd && sd == suits(e)
              val category = HandEvaluator.categorize5(ra, rb, rc, rd, ranks(e), isFlush)
              counts(category.ordinal) += 1L
              e += 1
            d += 1
          c += 1
        b += 1
      a += 1

    val byCategory = HandCategory.values.iterator.map { category =>
      category -> counts(category.ordinal)
    }.toMap
    CategoryDistribution(byCategory)

  /** Convenience method: enumerates the full standard deck and compares against expected counts.
    *
    * This is the primary validation entry point used by tests. An exact match proves
    * that [[HandEvaluator.categorize5]] correctly classifies every possible 5-card hand.
    */
  def checkStandardFiveCardCategoryDistribution(): DistributionCheckResult =
    val actual = evaluateFiveCardCategoryDistribution(Deck.full)
    DistributionCheckResult(actual, ExpectedFiveCardDistribution)
