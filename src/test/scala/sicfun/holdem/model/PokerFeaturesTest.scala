package sicfun.holdem.model
import sicfun.holdem.types.*

import munit.FunSuite
import sicfun.core.Card

/**
 * Tests for [[PokerFeatures]] 5-dimensional feature extraction (with hand strength).
 *
 * Validates:
 *   - PokerFeatures case class: dimension matches values length
 *   - Feature name constants: count and documented order
 *   - Individual feature correctness: potOdds, stackToPot, streetOrdinal, positionOrdinal
 *   - handStrengthProxy: preflop returns 0.5, postflop returns [0,1], strong > weak hand
 *   - All features are in [0, 1] on both preflop and river boards
 *   - Caching: repeated calls with same inputs return same results
 */
class PokerFeaturesTest extends FunSuite:
  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  // ---- PokerFeatures case class ----

  test("PokerFeatures: dimension matches values length") {
    val f = PokerFeatures(Vector(0.1, 0.2, 0.3, 0.4, 0.5))
    assertEquals(f.dimension, 5)
  }

  test("PokerFeatures: empty vector has dimension 0") {
    val f = PokerFeatures(Vector.empty)
    assertEquals(f.dimension, 0)
  }

  test("PokerFeatures: values are accessible and preserve order") {
    val vals = Vector(0.33, 0.5, 0.0, 1.0, 0.75)
    val f = PokerFeatures(vals)
    assertEquals(f.values, vals)
  }

  // ---- PokerFeatures companion: constants ----

  test("featureNames has exactly 5 elements matching dimension") {
    assertEquals(PokerFeatures.featureNames.length, PokerFeatures.dimension)
    assertEquals(PokerFeatures.dimension, 5)
  }

  test("featureNames are in the documented order") {
    assertEquals(PokerFeatures.featureNames(0), "potOdds")
    assertEquals(PokerFeatures.featureNames(1), "stackToPot")
    assertEquals(PokerFeatures.featureNames(2), "streetOrdinal")
    assertEquals(PokerFeatures.featureNames(3), "positionOrdinal")
    assertEquals(PokerFeatures.featureNames(4), "handStrengthProxy")
  }

  // ---- extract: preflop ----

  test("extract: preflop hand strength is always 0.5") {
    val state = GameState(Street.Preflop, Board.empty, pot = 3.0, toCall = 2.0,
      position = Position.Button, stackSize = 100.0, betHistory = Vector.empty)
    val features = PokerFeatures.extract(state, hole("As", "Ks"))
    assertEquals(features.values.last, 0.5)
  }

  test("extract: preflop produces correct dimension") {
    val state = GameState(Street.Preflop, Board.empty, pot = 10.0, toCall = 5.0,
      position = Position.SmallBlind, stackSize = 200.0, betHistory = Vector.empty)
    val features = PokerFeatures.extract(state, hole("7c", "2d"))
    assertEquals(features.dimension, PokerFeatures.dimension)
  }

  // ---- extract: potOdds feature ----

  test("extract: potOdds is 0 when toCall is 0") {
    val state = GameState(Street.Preflop, Board.empty, pot = 50.0, toCall = 0.0,
      position = Position.Button, stackSize = 100.0, betHistory = Vector.empty)
    val features = PokerFeatures.extract(state, hole("Ah", "Kh"))
    assertEqualsDouble(features.values(0), 0.0, 1e-9)
  }

  test("extract: potOdds computes toCall / (pot + toCall)") {
    val state = GameState(Street.Preflop, Board.empty, pot = 100.0, toCall = 50.0,
      position = Position.Button, stackSize = 500.0, betHistory = Vector.empty)
    val features = PokerFeatures.extract(state, hole("Ah", "Kh"))
    assertEqualsDouble(features.values(0), 50.0 / 150.0, 1e-9)
  }

  // ---- extract: stackToPot feature ----

  test("extract: stackToPot is clamped and normalized to [0, 1]") {
    // stackSize=500, pot=10 => SPR=50, clamped to 10, normalized to 1.0
    val state = GameState(Street.Preflop, Board.empty, pot = 10.0, toCall = 0.0,
      position = Position.Button, stackSize = 500.0, betHistory = Vector.empty)
    val features = PokerFeatures.extract(state, hole("Ah", "Kh"))
    assertEqualsDouble(features.values(1), 1.0, 1e-9)
  }

  test("extract: stackToPot for moderate ratio is scaled correctly") {
    // stackSize=50, pot=100 => SPR=0.5, normalized = 0.5/10 = 0.05
    val state = GameState(Street.Preflop, Board.empty, pot = 100.0, toCall = 0.0,
      position = Position.Button, stackSize = 50.0, betHistory = Vector.empty)
    val features = PokerFeatures.extract(state, hole("Ah", "Kh"))
    assertEqualsDouble(features.values(1), 0.05, 1e-9)
  }

  // ---- extract: streetOrdinal feature ----

  test("extract: streetOrdinal for Preflop is 0.0") {
    val state = GameState(Street.Preflop, Board.empty, pot = 10.0, toCall = 0.0,
      position = Position.Button, stackSize = 100.0, betHistory = Vector.empty)
    val features = PokerFeatures.extract(state, hole("Ah", "Kh"))
    assertEqualsDouble(features.values(2), 0.0, 1e-9)
  }

  test("extract: streetOrdinal for Flop is 1/3") {
    val board = Board.from(Seq(card("2c"), card("3d"), card("4h")))
    val state = GameState(Street.Flop, board, pot = 10.0, toCall = 0.0,
      position = Position.Button, stackSize = 100.0, betHistory = Vector.empty)
    val features = PokerFeatures.extract(state, hole("Ah", "Kh"))
    assertEqualsDouble(features.values(2), 1.0 / 3.0, 1e-9)
  }

  test("extract: streetOrdinal for Turn is 2/3") {
    val board = Board.from(Seq(card("2c"), card("3d"), card("4h"), card("5s")))
    val state = GameState(Street.Turn, board, pot = 10.0, toCall = 0.0,
      position = Position.Button, stackSize = 100.0, betHistory = Vector.empty)
    val features = PokerFeatures.extract(state, hole("Ah", "Kh"))
    assertEqualsDouble(features.values(2), 2.0 / 3.0, 1e-9)
  }

  test("extract: streetOrdinal for River is 1.0") {
    val board = Board.from(Seq(card("2c"), card("3d"), card("4h"), card("5s"), card("9c")))
    val state = GameState(Street.River, board, pot = 10.0, toCall = 0.0,
      position = Position.Button, stackSize = 100.0, betHistory = Vector.empty)
    val features = PokerFeatures.extract(state, hole("Ah", "Kh"))
    assertEqualsDouble(features.values(2), 1.0, 1e-9)
  }

  // ---- extract: positionOrdinal feature ----

  test("extract: positionOrdinal for SmallBlind is 0.0") {
    val state = GameState(Street.Preflop, Board.empty, pot = 10.0, toCall = 0.0,
      position = Position.SmallBlind, stackSize = 100.0, betHistory = Vector.empty)
    val features = PokerFeatures.extract(state, hole("Ah", "Kh"))
    assertEqualsDouble(features.values(3), 0.0, 1e-9)
  }

  test("extract: positionOrdinal for Button is 1.0") {
    val state = GameState(Street.Preflop, Board.empty, pot = 10.0, toCall = 0.0,
      position = Position.Button, stackSize = 100.0, betHistory = Vector.empty)
    val features = PokerFeatures.extract(state, hole("Ah", "Kh"))
    assertEqualsDouble(features.values(3), 1.0, 1e-9)
  }

  test("extract: positionOrdinal for middle positions is between 0 and 1") {
    val state = GameState(Street.Preflop, Board.empty, pot = 10.0, toCall = 0.0,
      position = Position.UTG, stackSize = 100.0, betHistory = Vector.empty)
    val features = PokerFeatures.extract(state, hole("Ah", "Kh"))
    val posVal = features.values(3)
    assert(posVal > 0.0 && posVal < 1.0, s"expected middle position ordinal in (0,1), got $posVal")
  }

  // ---- extract: all features in [0, 1] ----

  test("extract: all features are in [0, 1] on river") {
    val board = Board.from(Seq(card("2c"), card("3d"), card("4h"), card("5s"), card("9c")))
    val state = GameState(Street.River, board, pot = 100.0, toCall = 50.0,
      position = Position.Cutoff, stackSize = 500.0, betHistory = Vector.empty)
    val features = PokerFeatures.extract(state, hole("As", "Ks"))
    features.values.zipWithIndex.foreach { case (v, i) =>
      assert(v >= 0.0 && v <= 1.0, s"feature $i (${PokerFeatures.featureNames(i)}) = $v out of [0,1]")
    }
  }

  test("extract: all features are in [0, 1] preflop") {
    val state = GameState(Street.Preflop, Board.empty, pot = 3.0, toCall = 2.0,
      position = Position.BigBlind, stackSize = 100.0, betHistory = Vector.empty)
    val features = PokerFeatures.extract(state, hole("7h", "2s"))
    features.values.zipWithIndex.foreach { case (v, i) =>
      assert(v >= 0.0 && v <= 1.0, s"feature $i (${PokerFeatures.featureNames(i)}) = $v out of [0,1]")
    }
  }

  // ---- handStrengthProxy ----

  test("handStrengthProxy: preflop returns 0.5") {
    val strength = PokerFeatures.handStrengthProxy(Board.empty, hole("As", "Ad"))
    assertEquals(strength, 0.5)
  }

  test("handStrengthProxy: postflop returns value in [0, 1]") {
    val board = Board.from(Seq(card("2c"), card("3d"), card("4h")))
    val strength = PokerFeatures.handStrengthProxy(board, hole("As", "Ad"))
    assert(strength >= 0.0 && strength <= 1.0, s"hand strength $strength out of [0,1]")
  }

  test("handStrengthProxy: strong hand scores higher than weak hand on same board") {
    val board = Board.from(Seq(card("Ah"), card("Kd"), card("Qc"), card("Js"), card("2h")))
    val strongHand = hole("Ts", "9s") // Broadway straight: A-K-Q-J-T
    val weakHand = hole("3s", "4s")   // pair of nothing, low kickers
    val strongStr = PokerFeatures.handStrengthProxy(board, strongHand)
    val weakStr = PokerFeatures.handStrengthProxy(board, weakHand)
    assert(strongStr > weakStr,
      s"expected strong hand ($strongStr) > weak hand ($weakStr)")
  }

  // ---- extract: caching does not affect results ----

  test("extract: repeated calls with same inputs return same result") {
    val board = Board.from(Seq(card("Tc"), card("9d"), card("8h")))
    val state = GameState(Street.Flop, board, pot = 20.0, toCall = 10.0,
      position = Position.Button, stackSize = 200.0, betHistory = Vector.empty)
    val hand = hole("As", "Ks")
    val f1 = PokerFeatures.extract(state, hand)
    val f2 = PokerFeatures.extract(state, hand)
    assertEquals(f1.values, f2.values)
  }
