package sicfun.holdem.validation

import munit.FunSuite

/** Tests for [[ValidationRunner]]'s strategy-selection logic.
  *
  * The validation pipeline assigns different [[VillainStrategy]] implementations
  * depending on the leak type under test:
  *   - The GTO-baseline control ([[NoLeak]]) must use [[CfrVillainStrategy]]
  *     with heuristic fallback disabled, so any CFR solve failure surfaces
  *     immediately rather than silently degrading to heuristic play.
  *   - All leak-injected players use [[EquityBasedStrategy]], the fast
  *     heuristic baseline, since CFR precision is unnecessary when the
  *     villain's deviations are intentionally injected.
  */
class ValidationRunnerTest extends FunSuite:

  test("gto-baseline control uses CFR strategy"):
    val strategy = ValidationRunner.villainStrategyFor(NoLeak())
    assert(strategy.isInstanceOf[CfrVillainStrategy],
      s"expected CfrVillainStrategy for gto-baseline control, got ${strategy.getClass.getSimpleName}")
    assert(!strategy.asInstanceOf[CfrVillainStrategy].allowsHeuristicFallback,
      "gto-baseline control must fail fast rather than silently falling back to heuristic play")

  test("leak-injected players keep the fast heuristic strategy"):
    val strategy = ValidationRunner.villainStrategyFor(Overcalls(0.9))
    assert(strategy.isInstanceOf[EquityBasedStrategy],
      s"expected EquityBasedStrategy for leak players, got ${strategy.getClass.getSimpleName}")

  test("strategic snapshot computation from HandRecord actions"):
    import sicfun.holdem.types.{Board, GameState, PokerAction, Position, Street}
    import sicfun.holdem.strategic.bridge.StrategicSnapshot

    // Simulate what runOnePlayer does: extract last hero action and build snapshot
    val ra = RecordedAction(
      street = Street.Flop,
      player = "Hero",
      action = PokerAction.Raise(50.0),
      potBefore = 200.0,
      toCall = 40.0,
      stackBefore = 800.0,
      leakFired = false,
      leakId = None
    )
    val gs = GameState(
      street = ra.street,
      board = Board.empty,
      pot = ra.potBefore,
      toCall = ra.toCall,
      position = Position.Button,
      stackSize = ra.stackBefore,
      betHistory = Vector.empty
    )
    val heroEquity = 0.55
    val snap = StrategicSnapshot.build(
      gameState = gs,
      heroAction = ra.action,
      heroEquity = heroEquity,
      engineEv = heroEquity,
      staticEquity = heroEquity * 0.95,
      hasDrawPotential = true
    )
    val summary = StrategicSummary(
      dominantClass = snap.strategicClass.toString,
      fidelityCoverage = snap.fidelitySummary,
      fourWorldV11 = snap.fourWorld.v11.value,
      fourWorldV00 = snap.fourWorld.v00.value
    )
    assert(summary.dominantClass.nonEmpty)
    assert(summary.fidelityCoverage.contains("exact"))
    assert(summary.fourWorldV11 > 0.0)
