package sicfun.holdem.strategic

import sicfun.holdem.types.Street

class SignalTest extends munit.FunSuite:

  test("Sizing stores absolute and pot-fraction"):
    val s = Sizing(Chips(100.0), PotFraction(0.5))
    assertEquals(s.absolute.value, 100.0)
    assertEquals(s.fractionOfPot.value, 0.5)

  test("ActionSignal with sizing marks isAggressiveWager true"):
    val sig = ActionSignal(
      action = sicfun.holdem.types.PokerAction.Category.Raise,
      sizing = Some(Sizing(Chips(100.0), PotFraction(0.5))),
      timing = None,
      stage = Street.Flop
    )
    assert(sig.isAggressiveWager)

  test("ActionSignal without sizing marks isAggressiveWager false"):
    val sig = ActionSignal(
      action = sicfun.holdem.types.PokerAction.Category.Call,
      sizing = None,
      timing = None,
      stage = Street.Preflop
    )
    assert(!sig.isAggressiveWager)

  test("isAggressiveWager is biconditional with sizing.isDefined"):
    val withSizing = ActionSignal(
      action = sicfun.holdem.types.PokerAction.Category.Raise,
      sizing = Some(Sizing(Chips(50.0), PotFraction(0.75))),
      timing = None,
      stage = Street.Turn
    )
    val withoutSizing = ActionSignal(
      action = sicfun.holdem.types.PokerAction.Category.Fold,
      sizing = None,
      timing = None,
      stage = Street.Preflop
    )
    assertEquals(withSizing.isAggressiveWager, withSizing.sizing.isDefined)
    assertEquals(withoutSizing.isAggressiveWager, withoutSizing.sizing.isDefined)

  test("TotalSignal with empty showdown"):
    val act = ActionSignal(
      action = sicfun.holdem.types.PokerAction.Category.Check,
      sizing = None,
      timing = None,
      stage = Street.River
    )
    val total = TotalSignal(act, showdown = None)
    assert(total.showdown.isEmpty)

  test("TotalSignal with showdown data"):
    val act = ActionSignal(
      action = sicfun.holdem.types.PokerAction.Category.Call,
      sizing = None,
      timing = None,
      stage = Street.River
    )
    val sd = ShowdownSignal(
      revealedHands = Vector(
        RevealedHand(PlayerId("v1"), Vector.empty)
      )
    )
    val total = TotalSignal(act, showdown = Some(sd))
    assert(total.showdown.isDefined)
    assertEquals(total.showdown.get.revealedHands.length, 1)

  test("actionChannel returns action signal"):
    val act = ActionSignal(
      action = sicfun.holdem.types.PokerAction.Category.Raise,
      sizing = Some(Sizing(Chips(200.0), PotFraction(1.0))),
      timing = None,
      stage = Street.Flop
    )
    val total = TotalSignal(act, showdown = None)
    assertEquals(total.actionChannel, act)

  test("revelationChannel returns showdown when present"):
    val act = ActionSignal(
      action = sicfun.holdem.types.PokerAction.Category.Call,
      sizing = None,
      timing = None,
      stage = Street.River
    )
    val sd = ShowdownSignal(revealedHands = Vector.empty)
    val total = TotalSignal(act, showdown = Some(sd))
    assertEquals(total.revelationChannel, Some(sd))

  test("revelationChannel returns None when no showdown"):
    val act = ActionSignal(
      action = sicfun.holdem.types.PokerAction.Category.Check,
      sizing = None,
      timing = None,
      stage = Street.Flop
    )
    val total = TotalSignal(act, showdown = None)
    assertEquals(total.revelationChannel, None)

  test("TimingBucket has exactly four variants"):
    assertEquals(TimingBucket.values.length, 4)
