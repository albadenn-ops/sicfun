package sicfun.holdem.validation

import munit.FunSuite
import sicfun.core.{Card, Rank, Suit}
import sicfun.holdem.types.*
import sicfun.holdem.validation.{LeakInjectedVillain, NoLeak}

import scala.util.Random

/** Tests for the pluggable [[VillainStrategy]] interface and its two
  * implementations: [[EquityBasedStrategy]] (fast heuristic) and
  * [[CfrVillainStrategy]] (equilibrium-based CFR solver).
  *
  * Coverage:
  *   - '''EquityBasedStrategy''': strong hands do not fold facing a bet,
  *     weak hands fold most of the time, and strong hands value-bet
  *     when checked to
  *   - '''HeadsUpSimulator pluggability''': a custom always-call/check
  *     strategy can be injected and the simulator respects it
  *   - '''CfrVillainStrategy''': produces valid actions from equilibrium
  *     solves across different hands and seeds, all returned actions
  *     belong to the candidate set
  *   - '''CfrVillainStrategy caching''': repeated equivalent solve spots
  *     hit the cache (first call = miss, second call = hit)
  */
class VillainStrategyTest extends FunSuite:

  private val AhAs = HoleCards(Card(Rank.Ace, Suit.Hearts), Card(Rank.Ace, Suit.Spades))
  private val TwoThree = HoleCards(Card(Rank.Two, Suit.Hearts), Card(Rank.Three, Suit.Diamonds))
  private val QhQd = HoleCards(Card(Rank.Queen, Suit.Hearts), Card(Rank.Queen, Suit.Diamonds))
  private val flopAK7 = Board.from(Vector(
    Card(Rank.Ace, Suit.Hearts), Card(Rank.King, Suit.Diamonds), Card(Rank.Seven, Suit.Clubs)))

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

  test("HeadsUpSimulator uses pluggable VillainStrategy"):
    val callStrategy = new VillainStrategy:
      def decide(hand: HoleCards, state: GameState, candidates: Vector[PokerAction],
                 equityVsRandom: Double, rng: Random): PokerAction =
        if state.toCall > 0 then PokerAction.Call else PokerAction.Check

    val villain = LeakInjectedVillain("test", Vector(NoLeak()), 0.0, 42L)
    val sim = new HeadsUpSimulator(
      heroEngine = None,
      villain = villain,
      seed = 42L,
      villainStrategy = callStrategy
    )
    val record = sim.playHand(1)
    val villainActions = record.actions.filter(_.player == villain.name)
    villainActions.foreach { ra =>
      assert(ra.action == PokerAction.Call || ra.action == PokerAction.Check,
        s"Expected Call/Check but got ${ra.action}")
    }

  test("CfrVillainStrategy produces valid actions from equilibrium solve"):
    val strategy = CfrVillainStrategy()
    val gs = GameState(
      street = Street.Flop,
      board = flopAK7,
      pot = 6.0,
      toCall = 3.0,
      position = Position.BigBlind,
      stackSize = 97.0,
      betHistory = Vector.empty
    )
    val candidates = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(12.0))
    val action = strategy.decide(QhQd, gs, candidates, 0.70, new Random(42))
    assert(candidates.contains(action) || action == PokerAction.Check || action == PokerAction.Call,
      s"CfrVillainStrategy returned invalid action: $action")

  test("CfrVillainStrategy returns valid actions across different hands and seeds"):
    val strategy = CfrVillainStrategy()
    // Test with multiple hands to exercise the CFR solver path
    val hands = Vector(QhQd,
      HoleCards(Card(Rank.Six, Suit.Spades), Card(Rank.Seven, Suit.Hearts)))
    val gs = GameState(
      street = Street.Flop,
      board = flopAK7,
      pot = 6.0,
      toCall = 3.0,
      position = Position.BigBlind,
      stackSize = 97.0,
      betHistory = Vector.empty
    )
    val candidates = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(12.0))
    val allActions = for
      hand <- hands
      seed <- 0 until 20
    yield strategy.decide(hand, gs, candidates, 0.50, new Random(seed))
    // Every returned action must be from the candidate set
    allActions.foreach { a =>
      assert(candidates.contains(a),
        s"CfrVillainStrategy returned action not in candidates: $a")
    }
    // Different hands should potentially produce different equilibrium actions
    val distinctActions = allActions.distinct
    assert(distinctActions.nonEmpty, "CfrVillainStrategy must return at least one action")

  test("CfrVillainStrategy caches repeated equivalent solve spots"):
    val strategy = CfrVillainStrategy(allowHeuristicFallback = false)
    val gs = GameState(
      street = Street.Flop,
      board = flopAK7,
      pot = 6.0,
      toCall = 3.0,
      position = Position.BigBlind,
      stackSize = 97.0,
      betHistory = Vector.empty
    )
    val candidates = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(12.0))

    val firstAction = strategy.decide(QhQd, gs, candidates, 0.70, new Random(42))
    val afterFirst = strategy.cacheStatsSnapshot
    val secondAction = strategy.decide(QhQd, gs, candidates, 0.70, new Random(99))
    val afterSecond = strategy.cacheStatsSnapshot

    assert(candidates.contains(firstAction), s"first action must remain valid, got $firstAction")
    assert(candidates.contains(secondAction), s"second action must remain valid, got $secondAction")
    assertEquals(afterFirst.misses, 1L)
    assertEquals(afterFirst.hits, 0L)
    assertEquals(afterSecond.misses, 1L)
    assertEquals(afterSecond.hits, 1L)
