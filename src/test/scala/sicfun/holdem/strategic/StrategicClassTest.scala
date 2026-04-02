package sicfun.holdem.strategic

import sicfun.holdem.types.PokerAction

class StrategicClassTest extends munit.FunSuite:

  test("StrategicClass has exactly four members"):
    assertEquals(StrategicClass.values.length, 4)

  test("StrategicClass values are Value, Bluff, Marginal, SemiBluff"):
    val names = StrategicClass.values.map(_.toString).toSet
    assertEquals(names, Set("Value", "Bluff", "Marginal", "SemiBluff"))

  test("isAggressiveWager true for Raise"):
    assert(StrategicClass.isAggressiveWager(PokerAction.Raise(50.0)))

  test("isAggressiveWager false for Fold, Check, Call"):
    assert(!StrategicClass.isAggressiveWager(PokerAction.Fold))
    assert(!StrategicClass.isAggressiveWager(PokerAction.Check))
    assert(!StrategicClass.isAggressiveWager(PokerAction.Call))

  test("structuralBluff requires Bluff class AND aggressive wager"):
    assert(StrategicClass.isStructuralBluff(StrategicClass.Bluff, PokerAction.Raise(100.0)))

  test("structuralBluff false when class is not Bluff"):
    assert(!StrategicClass.isStructuralBluff(StrategicClass.Value, PokerAction.Raise(100.0)))
    assert(!StrategicClass.isStructuralBluff(StrategicClass.Marginal, PokerAction.Raise(50.0)))
    assert(!StrategicClass.isStructuralBluff(StrategicClass.SemiBluff, PokerAction.Raise(50.0)))

  test("structuralBluff false when action is not aggressive"):
    assert(!StrategicClass.isStructuralBluff(StrategicClass.Bluff, PokerAction.Call))
    assert(!StrategicClass.isStructuralBluff(StrategicClass.Bluff, PokerAction.Check))
    assert(!StrategicClass.isStructuralBluff(StrategicClass.Bluff, PokerAction.Fold))

  test("structuralBluff exhaustive: false for all non-Bluff classes regardless of action"):
    val nonBluff = Seq(StrategicClass.Value, StrategicClass.Marginal, StrategicClass.SemiBluff)
    val actions = Seq(PokerAction.Fold, PokerAction.Check, PokerAction.Call, PokerAction.Raise(99.0))
    for
      cls <- nonBluff
      act <- actions
    do assert(!StrategicClass.isStructuralBluff(cls, act), s"Expected false for $cls + $act")

  test("isExploitativeBluff requires Bluff + Raise + positive deltaManip"):
    assert(StrategicClass.isExploitativeBluff(StrategicClass.Bluff, PokerAction.Raise(50.0), Ev(1.0)))
    assert(!StrategicClass.isExploitativeBluff(StrategicClass.Bluff, PokerAction.Raise(50.0), Ev(0.0)))
    assert(!StrategicClass.isExploitativeBluff(StrategicClass.Bluff, PokerAction.Raise(50.0), Ev(-0.5)))
    assert(!StrategicClass.isExploitativeBluff(StrategicClass.Bluff, PokerAction.Call, Ev(1.0)))
    assert(!StrategicClass.isExploitativeBluff(StrategicClass.Value, PokerAction.Raise(50.0), Ev(1.0)))

  test("Law L2: isExploitativeBluff implies isStructuralBluff for all inputs"):
    val classes = StrategicClass.values.toSeq
    val actions = Seq(PokerAction.Fold, PokerAction.Check, PokerAction.Call, PokerAction.Raise(50.0))
    val manipValues = Seq(Ev(-1.0), Ev(0.0), Ev(0.5), Ev(10.0))
    for
      cls <- classes
      act <- actions
      dManip <- manipValues
    do
      if StrategicClass.isExploitativeBluff(cls, act, dManip) then
        assert(
          StrategicClass.isStructuralBluff(cls, act),
          s"L2 violated: exploitative($cls, $act, ${dManip.value}) but not structural"
        )
