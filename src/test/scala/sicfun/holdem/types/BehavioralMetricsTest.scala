package sicfun.holdem.types

import sicfun.holdem.analysis.*

import munit.FunSuite
import sicfun.core.{Card, DiscreteDistribution}

class BehavioralMetricsTest extends FunSuite:

  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  // ---- EvAnalysis tests ----

  test("EvAnalysis: single-hand posterior yields zero variance") {
    val hero = hole("Ah", "Kh")
    val board = Board.from(Seq(card("Ts"), card("9h"), card("8d"), card("2c"), card("3c")))
    val villain = hole("7c", "6c")
    val posterior = DiscreteDistribution(Map(villain -> 1.0))

    val result = EvAnalysis.evVariance(hero, board, posterior)
    assertEqualsDouble(result.variance, 0.0, 1e-9)
    assertEquals(result.handCount, 1)
  }

  test("EvAnalysis: diverse posterior yields positive variance") {
    val hero = hole("Ah", "Kh")
    val board = Board.from(Seq(card("Ts"), card("9h"), card("8d"), card("2c"), card("3c")))
    // villain1: 4d5d → T-high, hero wins with A-high
    // villain2: QsJd → Q-J-T-9-8 straight, hero loses
    val villain1 = hole("4d", "5d")
    val villain2 = hole("Qs", "Jd")
    val posterior = DiscreteDistribution(Map(villain1 -> 0.5, villain2 -> 0.5))

    val result = EvAnalysis.evVariance(hero, board, posterior)
    assert(result.variance > 0.0, s"expected positive variance, got ${result.variance}")
    assertEquals(result.handCount, 2)
    assert(result.mean >= 0.0 && result.mean <= 1.0, s"mean equity out of range: ${result.mean}")
  }

  test("EvAnalysis: stderr is non-negative") {
    val hero = hole("Ah", "Kh")
    val board = Board.from(Seq(card("Ts"), card("9h"), card("8d"), card("2c"), card("3c")))
    val villain1 = hole("7c", "6c")
    val villain2 = hole("Qs", "Jd")
    val villain3 = hole("4s", "5s")
    val posterior = DiscreteDistribution(Map(villain1 -> 0.5, villain2 -> 0.3, villain3 -> 0.2))

    val result = EvAnalysis.evVariance(hero, board, posterior)
    assert(result.stderr >= 0.0, s"expected non-negative stderr, got ${result.stderr}")
  }

  // ---- PlayerSignature tests ----

  private val board = Board.from(Seq(card("Ts"), card("9h"), card("8d")))
  private val baseState = GameState(
    street = Street.Flop,
    board = board,
    pot = 20.0,
    toCall = 10.0,
    position = Position.Button,
    stackSize = 200.0,
    betHistory = Vector.empty
  )
  test("PlayerSignature: all folds yields foldRate = 1") {
    val observations = Seq.fill(10)((baseState, PokerAction.Fold))
    val sig = PlayerSignature.compute(observations)
    assertEqualsDouble(sig.values(0), 1.0, 1e-9) // foldRate
    assertEqualsDouble(sig.values(1), 0.0, 1e-9) // raiseRate
    assertEqualsDouble(sig.values(2), 0.0, 1e-9) // callRate
    assertEqualsDouble(sig.values(3), 0.0, 1e-9) // checkRate
  }

  test("PlayerSignature: distance to self is 0") {
    val observations = Seq(
      (baseState, PokerAction.Fold),
      (baseState, PokerAction.Call),
      (baseState, PokerAction.Raise(20.0))
    )
    val sig = PlayerSignature.compute(observations)
    assertEqualsDouble(PlayerSignature.distance(sig, sig), 0.0, 1e-9)
  }

  test("PlayerSignature: different profiles have positive distance") {
    val foldObs = Seq.fill(10)((baseState, PokerAction.Fold))
    val raiseObs = Seq.fill(10)((baseState, PokerAction.Raise(20.0)))
    val sigA = PlayerSignature.compute(foldObs)
    val sigB = PlayerSignature.compute(raiseObs)
    assert(PlayerSignature.distance(sigA, sigB) > 0.0,
      "expected positive distance between different profiles")
  }

  test("PlayerSignature: feature count matches featureNames") {
    val observations = Seq((baseState, PokerAction.Call))
    val sig = PlayerSignature.compute(observations)
    assertEquals(sig.values.length, PlayerSignature.featureNames.length)
  }

  test("PlayerSignature: avgPotOddsWhenCalling is correct") {
    val state1 = baseState.copy(pot = 30.0, toCall = 10.0)  // potOdds = 10/40 = 0.25
    val state2 = baseState.copy(pot = 10.0, toCall = 10.0)  // potOdds = 10/20 = 0.5
    val observations = Seq(
      (state1, PokerAction.Call),
      (state2, PokerAction.Call)
    )
    val sig = PlayerSignature.compute(observations)
    assertEqualsDouble(sig.values(5), 0.375, 1e-9) // (0.25 + 0.5) / 2
  }

  test("PlayerSignature: compute does not throw when board contains As or Ks") {
    // Regression: dummyHand was hardcoded As Ks; if either appears on the board
    // computeHandStrength produced duplicate cards and evaluate5Cached threw.
    val boardWithAce = Board.from(Seq(card("As"), card("9h"), card("8d")))
    val boardWithBoth = Board.from(Seq(card("As"), card("Ks"), card("2d")))
    val stateAce = baseState.copy(board = boardWithAce, street = Street.Flop)
    val stateBoth = baseState.copy(board = boardWithBoth, street = Street.Flop)

    val obsAce = Seq.fill(3)((stateAce, PokerAction.Fold))
    val obsBoth = Seq.fill(3)((stateBoth, PokerAction.Fold))

    val sigAce = PlayerSignature.compute(obsAce)
    val sigBoth = PlayerSignature.compute(obsBoth)

    assertEquals(sigAce.values.length, PlayerSignature.featureNames.length)
    assertEquals(sigBoth.values.length, PlayerSignature.featureNames.length)
  }

  test("PlayerSignature: entropy of uniform action distribution is log2(4) = 2.0 bits") {
    val checkState = baseState.copy(toCall = 0.0)
    val observations = Seq(
      (baseState, PokerAction.Fold),
      (checkState, PokerAction.Check),
      (baseState, PokerAction.Call),
      (baseState, PokerAction.Raise(20.0))
    )
    val sig = PlayerSignature.compute(observations)
    assertEqualsDouble(sig.values(4), 2.0, 1e-9)
  }

  test("PlayerSignature: entropy of single-action distribution is 0") {
    val observations = Seq.fill(10)((baseState, PokerAction.Fold))
    val sig = PlayerSignature.compute(observations)
    assertEqualsDouble(sig.values(4), 0.0, 1e-9)
  }
