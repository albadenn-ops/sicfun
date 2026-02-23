package sicfun.holdem

import munit.FunSuite
import sicfun.core.Card

class RangeParserTest extends FunSuite:
  private def count(range: String): Int =
    RangeParser.parseWithHands(range) match
      case Left(err) => fail(err)
      case Right(result) => result.hands.size

  private def prob(range: String, a: String, b: String): Double =
    RangeParser.parseWithHands(range) match
      case Left(err) => fail(err)
      case Right(result) =>
        val cardA = Card.parse(a).getOrElse(fail(s"invalid card: $a"))
        val cardB = Card.parse(b).getOrElse(fail(s"invalid card: $b"))
        result.distribution.probabilityOf(HoleCards.from(Vector(cardA, cardB)))

  test("parse suited, offsuit, and any non-pair combos") {
    assertEquals(count("AKs"), 4)
    assertEquals(count("AKo"), 12)
    assertEquals(count("AK"), 16)
  }

  test("parse pair plus range") {
    assertEquals(count("QQ+"), 18)
  }

  test("parse non-pair plus range") {
    assertEquals(count("ATs+"), 16)
  }

  test("parse pair range with dash") {
    assertEquals(count("99-66"), 24)
  }

  test("parse suited wheel range with dash") {
    assertEquals(count("A5s-A2s"), 16)
  }

  test("deduplicate overlapping tokens") {
    assertEquals(count("AK, AKs"), 16)
  }

  test("weighted ranges adjust per-combo probability") {
    val suited = prob("AKs:0.5, AKo:1.0", "As", "Ks")
    val offsuit = prob("AKs:0.5, AKo:1.0", "As", "Kh")
    assert(math.abs((suited / offsuit) - 0.5) < 1e-9)
  }

  test("overlapping weighted tokens sum weights") {
    val suited = prob("AK:1.0, AKs:1.0", "As", "Ks")
    val offsuit = prob("AK:1.0, AKs:1.0", "As", "Kh")
    assert(math.abs((suited / offsuit) - 2.0) < 1e-9)
  }

  test("percentage weights are accepted") {
    val suited = prob("AKs:50%, AKo:100%", "As", "Ks")
    val offsuit = prob("AKs:50%, AKo:100%", "As", "Kh")
    assert(math.abs((suited / offsuit) - 0.5) < 1e-9)
  }

  test("at-sign weight separator is accepted") {
    val suited = prob("AKs@0.25, AKo@1.0", "As", "Ks")
    val offsuit = prob("AKs@0.25, AKo@1.0", "As", "Kh")
    assert(math.abs((suited / offsuit) - 0.25) < 1e-9)
  }

  test("parse rejects empty range") {
    val result = RangeParser.parse(" , , ")
    assert(result.isLeft)
  }

  test("parse rejects malformed or invalid weights") {
    assert(RangeParser.parse("AKs:").isLeft)
    assert(RangeParser.parse("AKs:foo").isLeft)
    assert(RangeParser.parse("AKs:-1").isLeft)
    assert(RangeParser.parse("AKs:NaN").isLeft)
    assert(RangeParser.parse("AKs:Infinity").isLeft)
  }

  test("parse rejects invalid suitedness and incompatible ranges") {
    assert(RangeParser.parse("AAs").isLeft)
    assert(RangeParser.parse("AKs-AQo").isLeft)
    assert(RangeParser.parse("A").isLeft)
  }
