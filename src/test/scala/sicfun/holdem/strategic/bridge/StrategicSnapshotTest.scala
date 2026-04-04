package sicfun.holdem.strategic.bridge

import sicfun.holdem.strategic.*
import sicfun.holdem.types.{Board, GameState, PokerAction, Position, Street}

class StrategicSnapshotTest extends munit.FunSuite:

  private def makeGS(
      street: Street = Street.Flop,
      pot: Double = 100.0,
      toCall: Double = 20.0,
      stack: Double = 500.0
  ): GameState =
    GameState(
      street = street,
      board = Board.empty,
      pot = pot,
      toCall = toCall,
      position = Position.Button,
      stackSize = stack,
      betHistory = Vector.empty
    )

  test("StrategicSnapshot.build produces snapshot with all fields populated"):
    val gs = makeGS(street = Street.Flop, pot = 200.0, toCall = 40.0, stack = 800.0)
    val snap = StrategicSnapshot.build(
      gameState = gs,
      heroAction = PokerAction.Raise(50.0),
      heroEquity = 0.72,
      engineEv = 0.65,
      staticEquity = 0.50,
      hasDrawPotential = false
    )
    // Public state fields
    assertEquals(snap.street, Street.Flop)
    assertEqualsDouble(snap.pot.value, 200.0, 1e-12)
    assertEqualsDouble(snap.heroStack.value, 800.0, 1e-12)
    assertEqualsDouble(snap.toCall.value, 40.0, 1e-12)
    // Classification
    assertEquals(snap.strategicClass, StrategicClass.Value) // 0.72 >= 0.65
    // Four-world
    assertEqualsDouble(snap.fourWorld.v11.value, 0.65, 1e-12)
    assertEqualsDouble(snap.fourWorld.v00.value, 0.50, 1e-12)
    // Baseline
    assertEqualsDouble(snap.baseline.value, 0.72, 1e-12)
    // Signal
    assertEquals(snap.actionSignal.action, PokerAction.Category.Raise)
    assert(snap.actionSignal.sizing.isDefined)

  test("StrategicSnapshot.build with Check action has no sizing"):
    val gs = makeGS()
    val snap = StrategicSnapshot.build(
      gameState = gs,
      heroAction = PokerAction.Check,
      heroEquity = 0.50,
      engineEv = 0.50,
      staticEquity = 0.50,
      hasDrawPotential = false
    )
    assertEquals(snap.actionSignal.action, PokerAction.Category.Check)
    assertEquals(snap.actionSignal.sizing, None)

  test("StrategicSnapshot.build low equity with draw -> SemiBluff"):
    val gs = makeGS()
    val snap = StrategicSnapshot.build(
      gameState = gs,
      heroAction = PokerAction.Call,
      heroEquity = 0.40,
      engineEv = 0.35,
      staticEquity = 0.30,
      hasDrawPotential = true
    )
    assertEquals(snap.strategicClass, StrategicClass.SemiBluff)

  test("StrategicSnapshot.build low equity no draw -> Bluff"):
    val gs = makeGS()
    val snap = StrategicSnapshot.build(
      gameState = gs,
      heroAction = PokerAction.Fold,
      heroEquity = 0.20,
      engineEv = 0.15,
      staticEquity = 0.15,
      hasDrawPotential = false
    )
    assertEquals(snap.strategicClass, StrategicClass.Bluff)

  test("StrategicSnapshot.fidelitySummary returns non-empty string"):
    val gs = makeGS()
    val snap = StrategicSnapshot.build(
      gameState = gs,
      heroAction = PokerAction.Call,
      heroEquity = 0.50,
      engineEv = 0.50,
      staticEquity = 0.50,
      hasDrawPotential = false
    )
    val summary = snap.fidelitySummary
    assert(summary.contains("exact"), s"expected 'exact' in: $summary")
    assert(summary.contains("approximate"), s"expected 'approximate' in: $summary")

  test("StrategicSnapshot.build with opponent stats produces classPosterior"):
    val gs = makeGS()
    val snap = StrategicSnapshot.build(
      gameState = gs,
      heroAction = PokerAction.Call,
      heroEquity = 0.50,
      engineEv = 0.50,
      staticEquity = 0.50,
      hasDrawPotential = false,
      opponentVpip = Some(0.30),
      opponentPfr = Some(0.20),
      opponentAf = Some(2.5)
    )
    assert(snap.opponentClassPosterior.isDefined)
    val dist = snap.opponentClassPosterior.get
    val total = dist.weights.values.sum
    assertEqualsDouble(total, 1.0, 1e-9)

  test("StrategicSnapshot.build without opponent stats has None classPosterior"):
    val gs = makeGS()
    val snap = StrategicSnapshot.build(
      gameState = gs,
      heroAction = PokerAction.Call,
      heroEquity = 0.50,
      engineEv = 0.50,
      staticEquity = 0.50,
      hasDrawPotential = false
    )
    assertEquals(snap.opponentClassPosterior, None)
