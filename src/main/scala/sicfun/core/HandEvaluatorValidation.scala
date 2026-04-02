package sicfun.core

/** Exhaustive combinatorial validation helpers for [[HandEvaluator]] category frequencies. */
object HandEvaluatorValidation:
  val TotalFiveCardHands: Long = 2_598_960L

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

  final case class CategoryDistribution(counts: Map[HandCategory, Long]):
    def count(category: HandCategory): Long = counts.getOrElse(category, 0L)

    lazy val total: Long = HandCategory.values.iterator.map(count).sum

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

  def checkStandardFiveCardCategoryDistribution(): DistributionCheckResult =
    val actual = evaluateFiveCardCategoryDistribution(Deck.full)
    DistributionCheckResult(actual, ExpectedFiveCardDistribution)
