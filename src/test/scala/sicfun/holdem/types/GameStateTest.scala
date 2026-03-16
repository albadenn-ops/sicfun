package sicfun.holdem.types

import munit.FunSuite

class GameStateTest extends FunSuite:
  private def gs(
      pot: Double = 10.0,
      toCall: Double = 2.0,
      stackSize: Double = 100.0
  ): GameState =
    GameState(
      street = Street.Preflop,
      board = Board.empty,
      pot = pot,
      toCall = toCall,
      position = Position.Button,
      stackSize = stackSize,
      betHistory = Vector.empty
    )

  test("potOdds returns correct fraction") {
    val s = gs(pot = 10.0, toCall = 2.0)
    assertEqualsDouble(s.potOdds, 2.0 / 12.0, 1e-12)
  }

  test("potOdds returns 0 when toCall is 0") {
    val s = gs(pot = 10.0, toCall = 0.0)
    assertEqualsDouble(s.potOdds, 0.0, 1e-12)
  }

  test("stackToPot returns correct ratio") {
    val s = gs(pot = 10.0, stackSize = 50.0)
    assertEqualsDouble(s.stackToPot, 5.0, 1e-12)
  }

  test("stackToPot returns infinity when pot is zero") {
    val s = gs(pot = 0.0, stackSize = 100.0)
    assert(s.stackToPot.isPosInfinity)
  }

  test("rejects negative pot") {
    intercept[IllegalArgumentException] {
      gs(pot = -1.0)
    }
  }

  test("rejects negative toCall") {
    intercept[IllegalArgumentException] {
      gs(toCall = -1.0)
    }
  }

  test("rejects negative stackSize") {
    intercept[IllegalArgumentException] {
      gs(stackSize = -1.0)
    }
  }

  test("zero pot and zero toCall give potOdds = 0 and stackToPot = infinity") {
    val s = gs(pot = 0.0, toCall = 0.0, stackSize = 100.0)
    assertEqualsDouble(s.potOdds, 0.0, 1e-12)
    assert(s.stackToPot.isPosInfinity)
  }

  test("Position enum has 9 entries with Hijack at ordinal 6") {
    assertEquals(Position.values.length, 9)
    assertEquals(Position.Hijack.ordinal, 6)
    assertEquals(Position.values.toVector, Vector(
      Position.SmallBlind, Position.BigBlind,
      Position.UTG, Position.UTG1, Position.UTG2,
      Position.Middle, Position.Hijack, Position.Cutoff, Position.Button
    ))
  }

  test("GameState accepts all Position values including Hijack") {
    val s = GameState(
      street = Street.Preflop,
      board = Board.empty,
      pot = 10.0,
      toCall = 2.0,
      position = Position.Hijack,
      stackSize = 100.0,
      betHistory = Vector.empty
    )
    assertEquals(s.position, Position.Hijack)
    assertEqualsDouble(s.potOdds, 2.0 / 12.0, 1e-12)
  }

  test("betHistory is preserved") {
    val history = Vector(
      BetAction(0, PokerAction.Call),
      BetAction(1, PokerAction.Raise(8.0))
    )
    val s = GameState(
      street = Street.Flop,
      board = Board.empty,
      pot = 20.0,
      toCall = 8.0,
      position = Position.BigBlind,
      stackSize = 92.0,
      betHistory = history
    )
    assertEquals(s.betHistory, history)
  }
