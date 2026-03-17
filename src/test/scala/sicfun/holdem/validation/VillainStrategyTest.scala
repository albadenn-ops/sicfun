package sicfun.holdem.validation

import munit.FunSuite
import sicfun.core.{Card, Rank, Suit}
import sicfun.holdem.types.*

import scala.util.Random

class VillainStrategyTest extends FunSuite:

  private val AhAs = HoleCards(Card(Rank.Ace, Suit.Hearts), Card(Rank.Ace, Suit.Spades))
  private val TwoThree = HoleCards(Card(Rank.Two, Suit.Hearts), Card(Rank.Three, Suit.Diamonds))

  test("EquityBasedStrategy produces valid actions facing a bet"):
    val strategy = EquityBasedStrategy()
    val gs = GameState(
      street = Street.Flop,
      board = Board.empty,
      pot = 10.0,
      toCall = 5.0,
      position = Position.BigBlind,
      stackSize = 95.0,
      betHistory = Vector.empty
    )
    val candidates = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(20.0))
    val action = strategy.decide(AhAs, gs, candidates, 0.85, new Random(42))
    assert(action != PokerAction.Fold, s"Strong hand should not fold, got $action")

  test("EquityBasedStrategy folds weak hands facing a bet"):
    val strategy = EquityBasedStrategy()
    val gs = GameState(
      street = Street.River,
      board = Board.empty,
      pot = 10.0,
      toCall = 8.0,
      position = Position.BigBlind,
      stackSize = 92.0,
      betHistory = Vector.empty
    )
    val candidates = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(26.0))
    val foldCount = (0 until 100).count { i =>
      strategy.decide(TwoThree, gs, candidates, 0.10, new Random(i)) == PokerAction.Fold
    }
    assert(foldCount >= 50, s"Weak hand should fold often, only folded $foldCount/100 times")

  test("EquityBasedStrategy value bets when not facing a bet"):
    val strategy = EquityBasedStrategy()
    val gs = GameState(
      street = Street.River,
      board = Board.empty,
      pot = 20.0,
      toCall = 0.0,
      position = Position.BigBlind,
      stackSize = 90.0,
      betHistory = Vector.empty
    )
    val candidates = Vector(PokerAction.Check, PokerAction.Raise(13.2), PokerAction.Raise(20.0))
    val betCount = (0 until 100).count { i =>
      strategy.decide(AhAs, gs, candidates, 0.85, new Random(i)) match
        case PokerAction.Raise(_) => true
        case _ => false
    }
    assert(betCount >= 50, s"Strong hand should bet often when checked to, only bet $betCount/100 times")
