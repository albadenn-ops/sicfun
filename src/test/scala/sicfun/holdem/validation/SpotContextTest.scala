package sicfun.holdem.validation

import munit.FunSuite
import sicfun.core.{Card, Rank, Suit}
import sicfun.holdem.types.{Board, HoleCards, GameState, Position, Street, PokerAction}

class SpotContextTest extends FunSuite:

  // ── BoardTexture ──

  test("dry rainbow board"):
    // Kh 7d 2c — no flush draws, no straight draws, unpaired
    val board = Board.from(Vector(
      Card(Rank.King, Suit.Hearts),
      Card(Rank.Seven, Suit.Diamonds),
      Card(Rank.Two, Suit.Clubs)
    ))
    val bt = BoardTexture.from(board)
    assert(!bt.flushDrawPossible)
    assert(!bt.straightDrawPossible)
    assert(!bt.paired)
    assert(!bt.monotone)
    assert(bt.isDry)

  test("wet connected suited board"):
    // 9h 8h 7d — flush draw possible (2 hearts), straight draw possible
    val board = Board.from(Vector(
      Card(Rank.Nine, Suit.Hearts),
      Card(Rank.Eight, Suit.Hearts),
      Card(Rank.Seven, Suit.Diamonds)
    ))
    val bt = BoardTexture.from(board)
    assert(bt.flushDrawPossible)
    assert(bt.straightDrawPossible)
    assert(!bt.paired)
    assert(bt.isWet)

  test("paired board"):
    val board = Board.from(Vector(
      Card(Rank.Queen, Suit.Hearts),
      Card(Rank.Queen, Suit.Diamonds),
      Card(Rank.Five, Suit.Clubs)
    ))
    val bt = BoardTexture.from(board)
    assert(bt.paired)

  test("monotone board"):
    val board = Board.from(Vector(
      Card(Rank.Ace, Suit.Spades),
      Card(Rank.Ten, Suit.Spades),
      Card(Rank.Four, Suit.Spades)
    ))
    val bt = BoardTexture.from(board)
    assert(bt.monotone)
    assert(bt.flushDrawPossible)

  test("empty board is dry"):
    val bt = BoardTexture.from(Board.empty)
    assert(bt.isDry)

  // ── PotGeometry ──

  test("PotGeometry from GameState"):
    // pot=175 includes opponent's bet of 75 into a pot of 100
    // betToPotRatio = 75 / (175 - 75) = 75/100 = 0.75
    val gs = GameState(
      street = Street.River,
      board = Board.empty,
      pot = 175.0,
      toCall = 75.0,
      position = Position.Button,
      stackSize = 150.0,
      betHistory = Vector.empty
    )
    val pg = PotGeometry.from(gs)
    assertEqualsDouble(pg.spr, 150.0 / 175.0, 0.01)
    assertEqualsDouble(pg.potOdds, 75.0 / 250.0, 0.01)
    assertEqualsDouble(pg.betToPotRatio, 0.75, 0.01)  // 75/(175-75)=75/100

  test("PotGeometry with zero pot"):
    val gs = GameState(Street.Preflop, Board.empty, 0.0, 1.0, Position.BigBlind, 100.0, Vector.empty)
    val pg = PotGeometry.from(gs)
    assert(pg.spr == Double.PositiveInfinity)

  // ── HandCategory ──

  test("HandCategory classification"):
    // Ordinal ordering: Nuts(0) < Strong(1) < Medium(2) < Weak(3) < Air(4)
    assert(HandCategory.Nuts.ordinal < HandCategory.Strong.ordinal)
    assert(HandCategory.Air.ordinal > HandCategory.Weak.ordinal)

  test("HandCategory.classify thresholds"):
    val dummyHero = HoleCards(Card(Rank.Ace, Suit.Hearts), Card(Rank.King, Suit.Spades))
    val board = Board.empty
    assertEquals(HandCategory.classify(dummyHero, board, 0.90), HandCategory.Nuts)
    assertEquals(HandCategory.classify(dummyHero, board, 0.85), HandCategory.Nuts)
    assertEquals(HandCategory.classify(dummyHero, board, 0.70), HandCategory.Strong)
    assertEquals(HandCategory.classify(dummyHero, board, 0.50), HandCategory.Medium)
    assertEquals(HandCategory.classify(dummyHero, board, 0.30), HandCategory.Weak)
    assertEquals(HandCategory.classify(dummyHero, board, 0.20), HandCategory.Air)

  // ── RangePosition ──

  test("RangePosition from preflop call line"):
    val line = ActionLine(Vector(PokerAction.Call))
    assertEquals(RangePosition.fromLine(line, Street.Flop), RangePosition.Capped)

  test("RangePosition from 3bet line"):
    val line = ActionLine(Vector(PokerAction.Raise(6.0), PokerAction.Raise(18.0)))
    assertEquals(RangePosition.fromLine(line, Street.Flop), RangePosition.Uncapped)

  test("RangePosition from check-raise line"):
    val line = ActionLine(Vector(PokerAction.Check, PokerAction.Raise(12.0)))
    assertEquals(RangePosition.fromLine(line, Street.Flop), RangePosition.Polarized)

  test("RangePosition from empty line"):
    val line = ActionLine(Vector.empty)
    assertEquals(RangePosition.fromLine(line, Street.Preflop), RangePosition.Capped)

  // ── SpotContext.build ──

  test("SpotContext.build assembles all components"):
    val board = Board.from(Vector(
      Card(Rank.Nine, Suit.Hearts),
      Card(Rank.Eight, Suit.Hearts),
      Card(Rank.Two, Suit.Clubs)
    ))
    val gs = GameState(Street.Flop, board, 20.0, 15.0, Position.Button, 85.0, Vector.empty)
    val hero = HoleCards(Card(Rank.Ace, Suit.Hearts), Card(Rank.King, Suit.Hearts))
    val line = ActionLine(Vector(PokerAction.Call))
    val equityVsRandom = 0.72

    val spot = SpotContext.build(gs, hero, line, equityVsRandom)
    assertEquals(spot.street, Street.Flop)
    assert(spot.boardTexture.flushDrawPossible) // two hearts
    assert(spot.boardTexture.isWet)
    assertEqualsDouble(spot.potGeometry.spr, 4.25, 0.01)
    assertEquals(spot.rangeAdvantage, RangePosition.Capped) // flat call line
    assertEquals(spot.handStrengthVsBoard, HandCategory.Strong) // 0.72 equity

  test("SpotContext.build with facingAction"):
    val gs = GameState(Street.River, Board.empty, 50.0, 40.0, Position.BigBlind, 60.0, Vector.empty)
    val hero = HoleCards(Card(Rank.Two, Suit.Clubs), Card(Rank.Three, Suit.Diamonds))
    val line = ActionLine(Vector(PokerAction.Call))
    val spot = SpotContext.build(gs, hero, line, 0.10, facingAction = Some(PokerAction.Raise(40.0)))
    assertEquals(spot.facingAction, Some(PokerAction.Raise(40.0)))
    // facingSizing = 40/50 = 0.8
    spot.facingSizing match
      case Some(s) => assertEqualsDouble(s, 0.8, 0.01)
      case None    => fail("expected facingSizing to be defined")
