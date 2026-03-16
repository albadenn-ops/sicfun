package sicfun.holdem.analysis

import munit.FunSuite
import sicfun.holdem.types.*

class PlayerSignatureTest extends FunSuite:
  private def gs(pot: Double = 10.0, toCall: Double = 2.0): GameState =
    GameState(
      street = Street.Preflop,
      board = Board.empty,
      pot = pot,
      toCall = toCall,
      position = Position.Button,
      stackSize = 100.0,
      betHistory = Vector.empty
    )

  test("compute produces correct dimension") {
    val obs = Seq((gs(), PokerAction.Call))
    val sig = PlayerSignature.compute(obs)
    assertEquals(sig.values.length, PlayerSignature.featureNames.length)
  }

  test("compute with all folds gives foldRate = 1") {
    val obs = (1 to 10).map(_ => (gs(), PokerAction.Fold))
    val sig = PlayerSignature.compute(obs)
    assertEqualsDouble(sig.values(0), 1.0, 1e-12) // foldRate
    assertEqualsDouble(sig.values(1), 0.0, 1e-12) // raiseRate
    assertEqualsDouble(sig.values(2), 0.0, 1e-12) // callRate
    assertEqualsDouble(sig.values(3), 0.0, 1e-12) // checkRate
  }

  test("compute with all raises gives raiseRate = 1") {
    val obs = (1 to 10).map(_ => (gs(), PokerAction.Raise(6.0)))
    val sig = PlayerSignature.compute(obs)
    assertEqualsDouble(sig.values(1), 1.0, 1e-12)
  }

  test("compute with mixed actions gives correct rates") {
    val obs = Seq(
      (gs(), PokerAction.Fold),
      (gs(), PokerAction.Call),
      (gs(), PokerAction.Raise(6.0)),
      (gs(), PokerAction.Check)
    )
    val sig = PlayerSignature.compute(obs)
    assertEqualsDouble(sig.values(0), 0.25, 1e-12) // foldRate
    assertEqualsDouble(sig.values(1), 0.25, 1e-12) // raiseRate
    assertEqualsDouble(sig.values(2), 0.25, 1e-12) // callRate
    assertEqualsDouble(sig.values(3), 0.25, 1e-12) // checkRate
    assert(sig.values(4) > 0.0, "entropy should be positive for uniform distribution")
  }

  test("compute rejects empty observations") {
    intercept[IllegalArgumentException] {
      PlayerSignature.compute(Seq.empty)
    }
  }

  test("avgPotOddsWhenCalling is 0 when no calls") {
    val obs = Seq((gs(), PokerAction.Fold))
    val sig = PlayerSignature.compute(obs)
    assertEqualsDouble(sig.values(5), 0.0, 1e-12)
  }

  test("avgPotOddsWhenCalling matches potOdds when all calls") {
    val state = gs(pot = 10.0, toCall = 5.0)
    val obs = (1 to 5).map(_ => (state, PokerAction.Call))
    val sig = PlayerSignature.compute(obs)
    assertEqualsDouble(sig.values(5), state.potOdds, 1e-12)
  }

  test("distance to self is 0") {
    val obs = Seq((gs(), PokerAction.Fold), (gs(), PokerAction.Call))
    val sig = PlayerSignature.compute(obs)
    assertEqualsDouble(PlayerSignature.distance(sig, sig), 0.0, 1e-12)
  }

  test("distance between different profiles is positive") {
    val allFold = PlayerSignature.compute((1 to 10).map(_ => (gs(), PokerAction.Fold)))
    val allRaise = PlayerSignature.compute((1 to 10).map(_ => (gs(), PokerAction.Raise(6.0))))
    assert(PlayerSignature.distance(allFold, allRaise) > 0.0)
  }

  test("distance rejects mismatched dimensions") {
    val a = PlayerSignature(Vector(1.0, 2.0))
    val b = PlayerSignature(Vector(1.0, 2.0, 3.0))
    intercept[IllegalArgumentException] {
      PlayerSignature.distance(a, b)
    }
  }
