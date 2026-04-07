package sicfun.holdem.types

import sicfun.holdem.model.*

import munit.FunSuite
import sicfun.core.Card

/**
  * Tests for [[PokerAction]], [[PokerAction.Category]], [[GameState]] derived metrics,
  * and [[PokerFeatures]] feature extraction.
  *
  * This suite validates:
  *   - '''Category mapping''': Each [[PokerAction]] variant maps to the correct coarsened
  *     [[PokerAction.Category]] (used as ML label space).
  *   - '''Category count''': Exactly 4 categories exist (Fold, Check, Call, Raise).
  *   - '''potOdds computation''': Correct fraction and zero-toCall edge case.
  *   - '''Validation''': Negative pot is rejected.
  *   - '''Feature extraction''': [[PokerFeatures.extract]] produces the correct dimension,
  *     hand strength defaults to 0.5 preflop, and is in [0, 1] on the river.
  */
class PokerActionTest extends FunSuite:
  /** Parses a 2-character card token (e.g. "As") or fails the test. */
  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  /** Creates canonical hole cards from two card tokens. */
  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  test("category maps Raise to Raise category") {
    assertEquals(PokerAction.Raise(100.0).category, PokerAction.Category.Raise)
  }

  test("category maps each simple action correctly") {
    assertEquals(PokerAction.Fold.category, PokerAction.Category.Fold)
    assertEquals(PokerAction.Check.category, PokerAction.Category.Check)
    assertEquals(PokerAction.Call.category, PokerAction.Category.Call)
  }

  test("categories has exactly 4 elements") {
    assertEquals(PokerAction.categories.length, 4)
  }

  test("potOdds computes correctly") {
    val state = GameState(Street.Flop, Board.empty, pot = 100.0, toCall = 50.0,
      position = Position.Button, stackSize = 500.0, betHistory = Vector.empty)
    assert(math.abs(state.potOdds - 50.0 / 150.0) < 1e-9)
  }

  test("potOdds is zero when toCall is zero") {
    val state = GameState(Street.Flop, Board.empty, pot = 100.0, toCall = 0.0,
      position = Position.Button, stackSize = 500.0, betHistory = Vector.empty)
    assertEquals(state.potOdds, 0.0)
  }

  test("GameState rejects negative pot") {
    intercept[IllegalArgumentException] {
      GameState(Street.Flop, Board.empty, pot = -1.0, toCall = 0.0,
        position = Position.Button, stackSize = 500.0, betHistory = Vector.empty)
    }
  }

  test("extract produces correct dimension") {
    val b = Board.from(Seq(card("2c"), card("3d"), card("4h"), card("5s"), card("9c")))
    val state = GameState(Street.River, b, pot = 100.0, toCall = 50.0,
      position = Position.Button, stackSize = 500.0, betHistory = Vector.empty)
    val features = PokerFeatures.extract(state, hole("As", "Ks"))
    assertEquals(features.dimension, PokerFeatures.dimension)
  }

  test("hand strength is 0.5 preflop") {
    val state = GameState(Street.Preflop, Board.empty, pot = 3.0, toCall = 2.0,
      position = Position.Button, stackSize = 100.0, betHistory = Vector.empty)
    val features = PokerFeatures.extract(state, hole("As", "Ks"))
    assertEquals(features.values.last, 0.5)
  }

  test("hand strength on river is between 0 and 1") {
    val b = Board.from(Seq(card("2c"), card("3d"), card("4h"), card("5s"), card("9c")))
    val state = GameState(Street.River, b, pot = 100.0, toCall = 0.0,
      position = Position.Button, stackSize = 500.0, betHistory = Vector.empty)
    val features = PokerFeatures.extract(state, hole("As", "Ks"))
    val strength = features.values.last
    assert(strength >= 0.0 && strength <= 1.0, s"hand strength $strength out of [0,1]")
  }
