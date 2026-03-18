package sicfun.holdem.bench

import munit.FunSuite
import sicfun.core.Card
import sicfun.holdem.types.HoleCards

class BenchSupportTest extends FunSuite:

  test("card parses valid token"):
    val c = BenchSupport.card("Ah")
    assertEquals(c, Card.parse("Ah").get)

  test("card throws on invalid token"):
    intercept[IllegalArgumentException]:
      BenchSupport.card("Xx")

  test("hole builds HoleCards from two tokens"):
    val h = BenchSupport.hole("Ah", "Kd")
    assertEquals(h, HoleCards.from(Vector(Card.parse("Ah").get, Card.parse("Kd").get)))
