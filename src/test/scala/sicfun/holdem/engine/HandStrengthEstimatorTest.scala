package sicfun.holdem.engine

import munit.FunSuite
import sicfun.core.Card
import sicfun.holdem.types.*

class HandStrengthEstimatorTest extends FunSuite:

  private def card(token: String): Card =
    Card.parse(token).getOrElse(throw new IllegalArgumentException(s"bad card: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  private def board(tokens: String*): Board =
    Board.from(tokens.map(card).toVector)

  test("preflopStrength: pocket aces is high"):
    val strength = HandStrengthEstimator.preflopStrength(hole("As", "Ah"))
    assert(strength > 0.85, s"AA strength $strength should be > 0.85")

  test("preflopStrength: 72o is low"):
    val strength = HandStrengthEstimator.preflopStrength(hole("7d", "2c"))
    assert(strength < 0.40, s"72o strength $strength should be < 0.40")

  test("preflopStrength: suited connectors get bonuses"):
    val suited = HandStrengthEstimator.preflopStrength(hole("Ts", "9s"))
    val offsuit = HandStrengthEstimator.preflopStrength(hole("Td", "9c"))
    assert(suited > offsuit, s"suited $suited should be > offsuit $offsuit")

  test("bestCategoryStrength: flush on board is high"):
    val h = hole("As", "Ks")
    val b = board("Qs", "Js", "3s")
    val strength = HandStrengthEstimator.bestCategoryStrength(h, b)
    assert(strength > 0.5, s"flush strength $strength should be > 0.5")

  test("drawPotential: 4-flush has bonus"):
    val h = hole("As", "Ks")
    val b = board("Qs", "Jd", "3s")
    val potential = HandStrengthEstimator.drawPotential(h, b)
    assert(potential >= 0.08, s"4-flush potential $potential should be >= 0.08")

  test("drawPotential: no draws returns near zero"):
    val h = hole("2c", "7d")
    val b = board("Ah", "Ks", "9h")
    val potential = HandStrengthEstimator.drawPotential(h, b)
    assert(potential < 0.05, s"no-draw potential $potential should be < 0.05")

  test("hasTightRun: connected ranks"):
    assert(HandStrengthEstimator.hasTightRun(Seq(5, 6, 7, 8)))
    assert(HandStrengthEstimator.hasTightRun(Seq(3, 5, 6, 7, 8)))
    assert(!HandStrengthEstimator.hasTightRun(Seq(2, 5, 9, 13)))

  test("hasTightRun: wheel ace"):
    assert(HandStrengthEstimator.hasTightRun(Seq(2, 3, 4, 14)))

  test("clamp: within bounds"):
    assertEquals(HandStrengthEstimator.clamp(0.5), 0.5)
    assertEquals(HandStrengthEstimator.clamp(-0.1), 0.0)
    assertEquals(HandStrengthEstimator.clamp(1.5), 1.0)
    assertEquals(HandStrengthEstimator.clamp(5.0, 2.0, 8.0), 5.0)
    assertEquals(HandStrengthEstimator.clamp(1.0, 2.0, 8.0), 2.0)
    assertEquals(HandStrengthEstimator.clamp(9.0, 2.0, 8.0), 8.0)

  test("fastGtoStrength: preflop delegates to preflopStrength"):
    val h = hole("As", "Ah")
    val preflop = HandStrengthEstimator.preflopStrength(h)
    val fast = HandStrengthEstimator.fastGtoStrength(h, Board.empty, Street.Preflop)
    assertEquals(fast, preflop)

  test("streetStrength: postflop incorporates board"):
    val h = hole("As", "Ah")
    val b = board("Ac", "Kd", "2h")
    val rng = new scala.util.Random(42)
    val strength = HandStrengthEstimator.streetStrength(h, b, Street.Flop, rng)
    assert(strength > 0.5, s"trips on flop strength $strength should be > 0.5")
