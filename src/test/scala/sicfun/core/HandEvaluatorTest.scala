package sicfun.core

class HandEvaluatorTest extends munit.FunSuite:
  private def cards(tokens: String*): Vector[Card] =
    Card.parseAll(tokens).getOrElse(fail(s"invalid cards: ${tokens.mkString(" ")}"))

  test("evaluate5 detects straight flush and ranks it correctly") {
    val hand = cards("As", "Ks", "Qs", "Js", "Ts")
    val rank = HandEvaluator.evaluate5(hand)
    assertEquals(rank.category, HandCategory.StraightFlush)
    assertEquals(rank.tiebreak, Vector(14))
  }

  test("evaluate5 handles wheel straight") {
    val hand = cards("Ah", "2d", "3s", "4c", "5h")
    val rank = HandEvaluator.evaluate5(hand)
    assertEquals(rank.category, HandCategory.Straight)
    assertEquals(rank.tiebreak, Vector(5))
  }

  test("evaluate5 detects four of a kind") {
    val hand = cards("Ah", "Ad", "Ac", "As", "Kd")
    val rank = HandEvaluator.evaluate5(hand)
    assertEquals(rank.category, HandCategory.FourOfKind)
    assertEquals(rank.tiebreak, Vector(14, 13))
  }

  test("evaluate5 detects full house") {
    val hand = cards("7s", "7d", "7c", "2h", "2d")
    val rank = HandEvaluator.evaluate5(hand)
    assertEquals(rank.category, HandCategory.FullHouse)
    assertEquals(rank.tiebreak, Vector(7, 2))
  }

  test("evaluate5 detects flush") {
    val hand = cards("As", "Js", "8s", "4s", "2s")
    val rank = HandEvaluator.evaluate5(hand)
    assertEquals(rank.category, HandCategory.Flush)
    assertEquals(rank.tiebreak, Vector(14, 11, 8, 4, 2))
  }

  test("evaluate5 detects two pair") {
    val hand = cards("Kc", "Kh", "4d", "4s", "9h")
    val rank = HandEvaluator.evaluate5(hand)
    assertEquals(rank.category, HandCategory.TwoPair)
    assertEquals(rank.tiebreak, Vector(13, 4, 9))
  }

  test("evaluate7 picks the best 5-card hand") {
    val seven = cards("As", "2d", "Ks", "Qs", "Js", "Ts", "3c")
    val rank = HandEvaluator.evaluate7(seven)
    assertEquals(rank.category, HandCategory.StraightFlush)
    assertEquals(rank.tiebreak, Vector(14))
  }

  test("hand ordering prefers stronger categories") {
    val straightFlush = HandEvaluator.evaluate5(cards("As", "Ks", "Qs", "Js", "Ts"))
    val fourKind = HandEvaluator.evaluate5(cards("Ah", "Ad", "Ac", "As", "Kd"))
    assert(straightFlush > fourKind)
  }

  test("cached evaluators match non-cached results") {
    val five = cards("As", "Ks", "Qs", "Js", "Ts")
    val seven = cards("As", "2d", "Ks", "Qs", "Js", "Ts", "3c")
    val r5 = HandEvaluator.evaluate5(five)
    val r5c = HandEvaluator.evaluate5Cached(five)
    val r7 = HandEvaluator.evaluate7(seven)
    val r7c = HandEvaluator.evaluate7Cached(seven)
    assertEquals(r5, r5c)
    assertEquals(r7, r7c)
  }

  test("categorize5 matches evaluate5 category on deterministic random sample") {
    val deck = Deck.full
    val rng = new scala.util.Random(123456789L)
    val samples = 25_000

    var i = 0
    while i < samples do
      val i0 = rng.nextInt(deck.length)
      var i1 = rng.nextInt(deck.length)
      while i1 == i0 do i1 = rng.nextInt(deck.length)
      var i2 = rng.nextInt(deck.length)
      while i2 == i0 || i2 == i1 do i2 = rng.nextInt(deck.length)
      var i3 = rng.nextInt(deck.length)
      while i3 == i0 || i3 == i1 || i3 == i2 do i3 = rng.nextInt(deck.length)
      var i4 = rng.nextInt(deck.length)
      while i4 == i0 || i4 == i1 || i4 == i2 || i4 == i3 do i4 = rng.nextInt(deck.length)

      val c0 = deck(i0)
      val c1 = deck(i1)
      val c2 = deck(i2)
      val c3 = deck(i3)
      val c4 = deck(i4)
      val hand = Vector(c0, c1, c2, c3, c4)

      val expected = HandEvaluator.evaluate5(hand).category
      val isFlush = c0.suit == c1.suit && c1.suit == c2.suit && c2.suit == c3.suit && c3.suit == c4.suit
      val actual = HandEvaluator.categorize5(
        c0.rank.value,
        c1.rank.value,
        c2.rank.value,
        c3.rank.value,
        c4.rank.value,
        isFlush
      )

      if actual != expected then
        fail(s"categorize5 mismatch for hand=$hand expected=$expected actual=$actual")
      i += 1
  }

  test("evaluate methods reject invalid card counts and duplicate cards") {
    val five = cards("As", "Ks", "Qs", "Js", "Ts")
    val seven = cards("As", "Ks", "Qs", "Js", "Ts", "9h", "8d")

    intercept[IllegalArgumentException] {
      HandEvaluator.evaluate5(five.take(4))
    }
    intercept[IllegalArgumentException] {
      HandEvaluator.evaluate5(Vector(five.head, five.head, five(2), five(3), five(4)))
    }
    intercept[IllegalArgumentException] {
      HandEvaluator.evaluate7(seven.take(6))
    }
    intercept[IllegalArgumentException] {
      HandEvaluator.evaluate7(Vector(seven.head, seven.head, seven(2), seven(3), seven(4), seven(5), seven(6)))
    }
    intercept[IllegalArgumentException] {
      HandEvaluator.evaluate5Cached(five.take(4))
    }
    intercept[IllegalArgumentException] {
      HandEvaluator.evaluate7Cached(seven.take(6))
    }
  }
