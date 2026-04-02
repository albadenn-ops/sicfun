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

  test("BatchData.size returns packedKeys length"):
    val bd = BenchSupport.BatchData(Array(1L, 2L, 3L), Array(4L, 5L, 6L))
    assertEquals(bd.size, 3)

  test("loadBatch full returns non-empty batch"):
    val bd = BenchSupport.loadBatch("full", maxMatchups = 8L)
    assert(bd.size > 0)
    assert(bd.size <= 8)

  test("loadBatch canonical returns non-empty batch"):
    val bd = BenchSupport.loadBatch("canonical", maxMatchups = 8L)
    assert(bd.size > 0)
    assert(bd.size <= 8)

  test("loadBatch rejects unknown table"):
    intercept[IllegalArgumentException]:
      BenchSupport.loadBatch("bogus", maxMatchups = 8L)
