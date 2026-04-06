package sicfun.holdem.engine

import munit.FunSuite
import sicfun.holdem.types.*

class PokerPomcpFormulationTest extends FunSuite:

  test("buildRivalPolicy produces correct dimensions"):
    val policy = PokerPomcpFormulation.buildRivalPolicy(
      numRivalTypes = 4,
      numPubStates = 192,
      numActions = 3
    )
    assertEquals(policy.length, 4 * 192 * 3)
    // Check it's a valid probability distribution per (type, pubState)
    val probPerSlice = policy.grouped(3).map(_.sum).toList
    assert(probPerSlice.forall(s => math.abs(s - 1.0) < 1e-10))

  test("buildActionEffects encodes fold correctly"):
    val actions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(3.0))
    val effects = PokerPomcpFormulation.buildActionEffects(actions, potChips = 100.0, stackChips = 500.0)
    assertEquals(effects.length, 9) // 3 actions * 3 fields
    assertEqualsDouble(effects(0), 0.0, 1e-10)  // fold pot_delta_frac
    assertEqualsDouble(effects(1), 1.0, 1e-10)  // fold is_fold
    assertEqualsDouble(effects(2), 0.0, 1e-10)  // fold is_allin

  test("buildShowdownEquity produces valid equity values"):
    val equity = PokerPomcpFormulation.buildShowdownEquity(
      numHeroBuckets = 10, numRivalBuckets = 10
    )
    assertEquals(equity.length, 100)
    assert(equity.forall(e => e >= 0.0 && e <= 1.0))

  test("buildTerminalFlags marks fold as HeroFold"):
    val flags = PokerPomcpFormulation.buildTerminalFlags(numPubStates = 192, numActions = 3)
    assertEquals(flags.length, 192 * 3)
    // Action 0 = fold -> HeroFold (1) at any pub state
    assert(flags(0 * 3 + 0) == 1, "action 0 should be HeroFold")
    assert(flags.forall(f => f >= 0 && f <= 3))

  test("buildTerminalFlags marks river non-fold as showdown"):
    val nPub = 192; val nAct = 3
    val flags = PokerPomcpFormulation.buildTerminalFlags(nPub, nAct)
    // River pub states start at street=3 -> index 3 * 8 * 6 = 144
    val riverStart = 144
    // Non-fold action at river should be showdown (3)
    assertEquals(flags(riverStart * nAct + 1), 3) // call at river = showdown
