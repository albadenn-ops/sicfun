package sicfun.holdem.strategic

import sicfun.holdem.types.PokerAction

class BluffFrameworkTest extends munit.FunSuite:

  private inline val Tol = 1e-12

  // -- Def 35: Structural bluff (delegates to StrategicClass.isStructuralBluff) --

  test("structuralBluff true for Bluff class + Raise"):
    assert(BluffFramework.isStructuralBluff(StrategicClass.Bluff, PokerAction.Raise(100.0)))

  test("structuralBluff false for Value class + Raise"):
    assert(!BluffFramework.isStructuralBluff(StrategicClass.Value, PokerAction.Raise(100.0)))

  test("structuralBluff false for Bluff class + Call"):
    assert(!BluffFramework.isStructuralBluff(StrategicClass.Bluff, PokerAction.Call))

  // -- Def 36: Feasible action correspondence --

  test("feasibleActions returns non-empty set for any belief"):
    val actions = Vector(PokerAction.Fold, PokerAction.Check, PokerAction.Call, PokerAction.Raise(50.0))
    val feasible = BluffFramework.feasibleActions(actions)
    assert(feasible.nonEmpty)
    assertEquals(feasible, actions)

  // -- Def 37: Feasible non-bluff actions --

  test("feasibleNonBluffActions excludes structural bluffs"):
    val actions = Vector(PokerAction.Fold, PokerAction.Check, PokerAction.Call, PokerAction.Raise(50.0))
    val nonBluff = BluffFramework.feasibleNonBluffActions(StrategicClass.Bluff, actions)
    // For Bluff class, Raise is structural bluff => excluded
    assertEquals(nonBluff, Vector(PokerAction.Fold, PokerAction.Check, PokerAction.Call))

  test("feasibleNonBluffActions keeps all actions for Value class"):
    val actions = Vector(PokerAction.Fold, PokerAction.Check, PokerAction.Raise(50.0))
    val nonBluff = BluffFramework.feasibleNonBluffActions(StrategicClass.Value, actions)
    assertEquals(nonBluff, actions)

  // -- Def 38: Bluff gain --

  test("bluffGain is Q_attrib(u) - Q_ref(u_cf)"):
    val gain = BluffFramework.bluffGain(
      qAttribAction = Ev(12.0),
      qRefCounterfactual = Ev(8.0)
    )
    assertEqualsDouble(gain.value, 4.0, Tol)

  test("bluffGain can be negative"):
    val gain = BluffFramework.bluffGain(
      qAttribAction = Ev(5.0),
      qRefCounterfactual = Ev(9.0)
    )
    assertEqualsDouble(gain.value, -4.0, Tol)

  // -- Def 39: Exploitative bluff --

  test("isExploitativeBluff requires structural bluff AND positive gain"):
    assert(BluffFramework.isExploitativeBluff(
      cls = StrategicClass.Bluff,
      action = PokerAction.Raise(100.0),
      bestGainOverNonBluffs = Ev(2.0)
    ))

  test("isExploitativeBluff false when gain <= 0"):
    assert(!BluffFramework.isExploitativeBluff(
      cls = StrategicClass.Bluff,
      action = PokerAction.Raise(100.0),
      bestGainOverNonBluffs = Ev(0.0)
    ))
    assert(!BluffFramework.isExploitativeBluff(
      cls = StrategicClass.Bluff,
      action = PokerAction.Raise(100.0),
      bestGainOverNonBluffs = Ev(-1.0)
    ))

  test("isExploitativeBluff false when not structural bluff"):
    assert(!BluffFramework.isExploitativeBluff(
      cls = StrategicClass.Value,
      action = PokerAction.Raise(100.0),
      bestGainOverNonBluffs = Ev(5.0)
    ))

  // -- Corollary 2: exploitative bluff => structural bluff --

  test("Corollary 2: exploitative bluff always implies structural bluff"):
    val classes = StrategicClass.values.toSeq
    val actions = Seq(PokerAction.Fold, PokerAction.Check, PokerAction.Call, PokerAction.Raise(50.0))
    val gains = Seq(Ev(-1.0), Ev(0.0), Ev(0.5), Ev(10.0))
    for
      cls <- classes
      act <- actions
      g <- gains
    do
      if BluffFramework.isExploitativeBluff(cls, act, g) then
        assert(
          BluffFramework.isStructuralBluff(cls, act),
          s"Corollary 2 violated: exploitative($cls, $act, ${g.value}) but not structural"
        )
