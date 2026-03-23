package sicfun.holdem.engine

import munit.FunSuite
import sicfun.core.{Card, DiscreteDistribution}
import sicfun.holdem.types.*

class GtoSolveEngineTest extends FunSuite:

  private def card(token: String): Card =
    Card.parse(token).getOrElse(throw new IllegalArgumentException(s"bad card: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  private def board(tokens: String*): Board =
    Board.from(tokens.map(card).toVector)

  // --- canonicalHeroBoardSignature tests ---

  test("canonicalHeroBoardSignature: suit-invariant for preflop"):
    val sig1 = GtoSolveEngine.canonicalHeroBoardSignature(hole("As", "Ks"), Board.empty)
    val sig2 = GtoSolveEngine.canonicalHeroBoardSignature(hole("Ah", "Kh"), Board.empty)
    assertEquals(sig1, sig2)

  test("canonicalHeroBoardSignature: suit-invariant for flop"):
    val sig1 = GtoSolveEngine.canonicalHeroBoardSignature(hole("As", "Ks"), board("Qs", "Jd", "3h"))
    val sig2 = GtoSolveEngine.canonicalHeroBoardSignature(hole("Ah", "Kh"), board("Qh", "Jc", "3s"))
    assertEquals(sig1, sig2)

  test("canonicalHeroBoardSignature: different hands produce different signatures"):
    val sig1 = GtoSolveEngine.canonicalHeroBoardSignature(hole("As", "Ks"), Board.empty)
    val sig2 = GtoSolveEngine.canonicalHeroBoardSignature(hole("As", "Qs"), Board.empty)
    assertNotEquals(sig1, sig2)

  // --- orderedPositiveProbabilities tests ---

  test("orderedPositiveProbabilities: filters zero and negative"):
    val actions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Check)
    val probs = Map[PokerAction, Double](
      PokerAction.Fold -> 0.3,
      PokerAction.Call -> 0.0,
      PokerAction.Check -> -0.1
    )
    val result = GtoSolveEngine.orderedPositiveProbabilities(actions, probs)
    assertEquals(result.length, 1)
    assertEquals(result.head._1, PokerAction.Fold)

  test("orderedPositiveProbabilities: preserves order"):
    val actions = Vector(PokerAction.Check, PokerAction.Raise(2.0))
    val probs = Map[PokerAction, Double](
      PokerAction.Check -> 0.6,
      PokerAction.Raise(2.0) -> 0.4
    )
    val result = GtoSolveEngine.orderedPositiveProbabilities(actions, probs)
    assertEquals(result.length, 2)
    assertEquals(result(0)._1, PokerAction.Check)
    assertEquals(result(1)._1, PokerAction.Raise(2.0))

  // --- sampleActionByPolicy tests ---

  test("sampleActionByPolicy: returns fallback when all zero"):
    val result = GtoSolveEngine.sampleActionByPolicy(
      ordered = Vector.empty,
      fallback = PokerAction.Fold,
      rng = new scala.util.Random(42)
    )
    assertEquals(result, PokerAction.Fold)

  test("sampleActionByPolicy: deterministic with single action"):
    val result = GtoSolveEngine.sampleActionByPolicy(
      ordered = Vector(PokerAction.Call -> 1.0),
      fallback = PokerAction.Fold,
      rng = new scala.util.Random(42)
    )
    assertEquals(result, PokerAction.Call)

  // --- GTO parametrization tests ---

  test("gtoIterations: preflop higher than river"):
    val preflop = GtoSolveEngine.gtoIterations(Street.Preflop, 600, 3)
    val river = GtoSolveEngine.gtoIterations(Street.River, 600, 3)
    assert(preflop > river, s"preflop $preflop should be > river $river")

  test("gtoIterations: 2 candidates gets floor reduction"):
    val full = GtoSolveEngine.gtoIterations(Street.Flop, 600, 3)
    val reduced = GtoSolveEngine.gtoIterations(Street.Flop, 600, 2)
    assert(reduced < full, s"2-candidate $reduced should be < 3-candidate $full")

  test("gtoMaxVillainHands: preflop highest"):
    val preflop = GtoSolveEngine.gtoMaxVillainHands(Street.Preflop, 3)
    val river = GtoSolveEngine.gtoMaxVillainHands(Street.River, 3)
    assert(preflop > river, s"preflop $preflop should be > river $river")

  // --- hashActions tests ---

  test("hashActions: deterministic"):
    val actions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(2.5))
    val hash1 = GtoSolveEngine.hashActions(actions)
    val hash2 = GtoSolveEngine.hashActions(actions)
    assertEquals(hash1, hash2)

  test("hashActions: different actions produce different hash"):
    val h1 = GtoSolveEngine.hashActions(Vector(PokerAction.Fold, PokerAction.Call))
    val h2 = GtoSolveEngine.hashActions(Vector(PokerAction.Check, PokerAction.Call))
    assertNotEquals(h1, h2)

  test("hashPosterior: deterministic across hand ordering"):
    val left = DiscreteDistribution(
      Map(
        hole("As", "Kd") -> 0.4,
        hole("Qh", "Qs") -> 0.6
      )
    )
    val right = DiscreteDistribution(
      Map(
        hole("Qh", "Qs") -> 0.6,
        hole("As", "Kd") -> 0.4
      )
    )
    assertEquals(GtoSolveEngine.hashPosterior(left), GtoSolveEngine.hashPosterior(right))

  test("hashPosterior: changes when opponent weights change"):
    val base = DiscreteDistribution(
      Map(
        hole("As", "Kd") -> 0.4,
        hole("Qh", "Qs") -> 0.6
      )
    )
    val shifted = DiscreteDistribution(
      Map(
        hole("As", "Kd") -> 0.7,
        hole("Qh", "Qs") -> 0.3
      )
    )
    assertNotEquals(GtoSolveEngine.hashPosterior(base), GtoSolveEngine.hashPosterior(shifted))

  test("buildGtoSolveCacheKey: includes opponent posterior hash"):
    val state = GameState(
      street = Street.Turn, pot = 18.0, toCall = 4.0, stackSize = 82.0,
      board = board("2c", "7d", "Jh", "Qc"), position = Position.Button,
      betHistory = Vector.empty
    )
    val candidates = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(12.0))
    val hero = hole("Ac", "Kd")
    val left = DiscreteDistribution(
      Map(
        hole("As", "Qd") -> 0.35,
        hole("Ts", "9s") -> 0.30,
        hole("Qh", "Js") -> 0.20,
        hole("7c", "7s") -> 0.15
      )
    )
    val right = DiscreteDistribution(
      Map(
        hole("As", "Qd") -> 0.15,
        hole("Ts", "9s") -> 0.20,
        hole("Qh", "Js") -> 0.30,
        hole("7c", "7s") -> 0.35
      )
    )
    val signature = GtoSolveEngine.canonicalHeroBoardSignature(hero, state.board)
    val leftKey = GtoSolveEngine.buildGtoSolveCacheKey(
      perspective = 0,
      hand = hero,
      state = state,
      candidates = candidates,
      opponentPosterior = left,
      baseEquityTrials = 800,
      canonicalSignature = signature
    )
    val rightKey = GtoSolveEngine.buildGtoSolveCacheKey(
      perspective = 0,
      hand = hero,
      state = state,
      candidates = candidates,
      opponentPosterior = right,
      baseEquityTrials = 800,
      canonicalSignature = signature
    )
    assertNotEquals(leftKey, rightKey)

  // --- gtoResponds Fast mode tests ---

  test("gtoResponds Fast: single candidate returns it immediately"):
    val state = GameState(
      street = Street.Flop, pot = 3.0, toCall = 1.0, stackSize = 50.0,
      board = board("Ah", "Kd", "2c"), position = Position.Button,
      betHistory = Vector.empty
    )
    val result = GtoSolveEngine.gtoResponds(
      hand = hole("7d", "2h"),
      state = state,
      candidates = Vector(PokerAction.Call),
      mode = GtoSolveEngine.GtoMode.Fast,
      opponentPosterior = null, // not used in Fast mode
      baseEquityTrials = 600,
      rng = new scala.util.Random(42),
      perspective = 0,
      exactGtoCache = scala.collection.mutable.HashMap.empty,
      exactGtoCacheStats = GtoSolveEngine.GtoCacheStats()
    )
    assertEquals(result, PokerAction.Call)

  test("gtoResponds Fast: strong hand facing bet does not fold"):
    val state = GameState(
      street = Street.Flop, pot = 5.0, toCall = 1.0, stackSize = 50.0,
      board = board("As", "Ad", "Kh"), position = Position.Button,
      betHistory = Vector.empty
    )
    // Pocket aces with an ace on board = very strong
    val results = (0 until 50).map { i =>
      GtoSolveEngine.gtoResponds(
        hand = hole("Ac", "Ah"),
        state = state,
        candidates = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(2.5)),
        mode = GtoSolveEngine.GtoMode.Fast,
        opponentPosterior = null,
        baseEquityTrials = 600,
        rng = new scala.util.Random(i),
        perspective = 0,
        exactGtoCache = scala.collection.mutable.HashMap.empty,
        exactGtoCacheStats = GtoSolveEngine.GtoCacheStats()
      )
    }
    val foldCount = results.count(_ == PokerAction.Fold)
    assertEquals(foldCount, 0, s"Quads should never fold, but folded $foldCount times")
