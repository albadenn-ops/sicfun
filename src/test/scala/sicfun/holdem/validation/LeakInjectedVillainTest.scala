package sicfun.holdem.validation

import munit.FunSuite
import sicfun.core.{Card, Rank, Suit}
import sicfun.holdem.types.{Board, PokerAction, Position, Street}

class LeakInjectedVillainTest extends FunSuite:

  test("villain returns GTO action when no leak applies"):
    val villain = LeakInjectedVillain(
      name = "test_noleak",
      leaks = Vector(OverfoldsToAggression(severity = 1.0)),
      baselineNoise = 0.0,
      seed = 42L
    )
    // Preflop spot -- OverfoldsToAggression only fires on river
    val gtoAction = PokerAction.Call
    val spot = SpotContext(
      street = Street.Preflop,
      board = Board.empty,
      boardTexture = BoardTexture.from(Board.empty),
      potGeometry = PotGeometry(spr = 50.0, potOdds = 0.33, betToPotRatio = 0.5, effectiveStack = 100.0),
      position = Position.BigBlind,
      facingAction = None,
      facingSizing = None,
      lineRepresented = ActionLine(Vector.empty),
      handStrengthVsBoard = HandCategory.Medium,
      rangeAdvantage = RangePosition.Capped
    )
    val result = villain.decide(gtoAction, spot)
    assertEquals(result.action, PokerAction.Call)
    assert(!result.leakFired)

  test("villain fires leak when applicable with severity=1.0"):
    val villain = LeakInjectedVillain(
      name = "test_overfold",
      leaks = Vector(OverfoldsToAggression(severity = 1.0)),
      baselineNoise = 0.0,
      seed = 42L
    )
    val board = Board.from(Vector(
      Card(Rank.Nine, Suit.Hearts),
      Card(Rank.Eight, Suit.Hearts),
      Card(Rank.Two, Suit.Clubs),
      Card(Rank.Five, Suit.Diamonds),
      Card(Rank.Jack, Suit.Hearts)
    ))
    val spot = SpotContext(
      street = Street.River,
      board = board,
      boardTexture = BoardTexture.from(board),
      potGeometry = PotGeometry(spr = 1.0, potOdds = 0.44, betToPotRatio = 0.8, effectiveStack = 50.0),
      position = Position.BigBlind,
      facingAction = Some(PokerAction.Raise(80.0)),
      facingSizing = Some(0.8),
      lineRepresented = ActionLine(Vector(PokerAction.Call)),
      handStrengthVsBoard = HandCategory.Weak,
      rangeAdvantage = RangePosition.Capped
    )
    val result = villain.decide(PokerAction.Call, spot)
    assertEquals(result.action, PokerAction.Fold)
    assert(result.leakFired)
    assertEquals(result.leakId, Some("overfold-river-aggression"))

  test("baseline noise perturbs action selection over many trials"):
    val spot = SpotContext(
      street = Street.Flop,
      board = Board.empty,
      boardTexture = BoardTexture.from(Board.empty),
      potGeometry = PotGeometry(spr = 10.0, potOdds = 0.2, betToPotRatio = 0.25, effectiveStack = 100.0),
      position = Position.Button,
      facingAction = None,
      facingSizing = None,
      lineRepresented = ActionLine(Vector.empty),
      handStrengthVsBoard = HandCategory.Medium,
      rangeAdvantage = RangePosition.Uncapped
    )
    // Use a single villain with 5% noise, call decide many times.
    // Each call consumes rng state, so successive calls are independent.
    val v = LeakInjectedVillain("n", Vector.empty, 0.05, seed = 42L)
    var deviations = 0
    for _ <- 0 until 1000 do
      val r = v.decide(PokerAction.Check, spot)
      if r.action != PokerAction.Check then deviations += 1
    assert(deviations > 20 && deviations < 100, s"Expected ~50 deviations, got $deviations")
