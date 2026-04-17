package sicfun.holdem.strategic.formulation

import munit.FunSuite
import sicfun.holdem.types.*
import sicfun.holdem.strategic.types.*
import sicfun.holdem.engine.PokerPomcpFormulation

class LegacyToyFormulationInputTest extends FunSuite:

  private def minimalState: GameState =
    GameState(
      street = Street.Preflop,
      board = Board.empty,
      pot = 100.0,
      toCall = 20.0,
      position = Position.Button,
      stackSize = 1000.0,
      betHistory = Vector.empty
    )

  private val defaultActions = Vector(
    PokerAction.Fold,
    PokerAction.Call,
    PokerAction.Raise(50.0)
  )

  private val pftPriors: Map[(StrategicClass, PokerAction.Category), Double] =
    Map(
      (StrategicClass.Value, PokerAction.Category.Fold) -> 0.05,
      (StrategicClass.Value, PokerAction.Category.Check) -> 0.35,
      (StrategicClass.Value, PokerAction.Category.Call) -> 0.40,
      (StrategicClass.Value, PokerAction.Category.Raise) -> 0.20,
      (StrategicClass.Bluff, PokerAction.Category.Fold) -> 0.10,
      (StrategicClass.Bluff, PokerAction.Category.Check) -> 0.10,
      (StrategicClass.Bluff, PokerAction.Category.Call) -> 0.15,
      (StrategicClass.Bluff, PokerAction.Category.Raise) -> 0.65,
      (StrategicClass.StructuralBluff, PokerAction.Category.Fold) -> 0.05,
      (StrategicClass.StructuralBluff, PokerAction.Category.Check) -> 0.15,
      (StrategicClass.StructuralBluff, PokerAction.Category.Call) -> 0.30,
      (StrategicClass.StructuralBluff, PokerAction.Category.Raise) -> 0.50,
      (StrategicClass.Mixed, PokerAction.Category.Fold) -> 0.15,
      (StrategicClass.Mixed, PokerAction.Category.Check) -> 0.40,
      (StrategicClass.Mixed, PokerAction.Category.Call) -> 0.35,
      (StrategicClass.Mixed, PokerAction.Category.Raise) -> 0.10
    )

  private val legacyRivalPriors = LegacyRivalPriors(
    pftActionPriors = pftPriors,
    pomcpClassPriors = PokerPomcpFormulation.defaultClassPriors
  )

  test("from produces FormulationInput with StrengthHint"):
    val input = LegacyToyFormulationInput.from(
      gameState = minimalState,
      candidateActions = defaultActions,
      rivalBeliefs = Map.empty,
      heroBucket = 5,
      rivalPriors = legacyRivalPriors
    )
    assertEquals(input.spot.gameState, minimalState)
    assertEquals(input.spot.candidateActions, defaultActions)
    input.spot.heroValueInput match
      case HeroValueInput.StrengthHint(bucket, source) =>
        assertEquals(bucket, 5)
        assert(source.nonEmpty)
      case other => fail(s"Expected StrengthHint, got $other")

  test("value source estimateSpotEquity matches heroBucket / 9.0"):
    val input = LegacyToyFormulationInput.from(
      minimalState, defaultActions, Map.empty, heroBucket = 7, legacyRivalPriors
    )
    input.valueSource.estimateSpotEquity(input.spot) match
      case BridgeResult.Approximate(value, _) =>
        assertEqualsDouble(value, 7.0 / 9.0, 1e-10)
      case other => fail(s"Expected Approximate, got $other")

  test("value source showdownEquityTable matches linear heuristic"):
    val input = LegacyToyFormulationInput.from(
      minimalState, defaultActions, Map.empty, heroBucket = 5, legacyRivalPriors
    )
    val expected = PokerPomcpFormulation.buildLinearShowdownEquity(10, 10)
    input.valueSource.showdownEquityTable(input.spot, 10, 10) match
      case BridgeResult.Approximate(table, _) =>
        assertEquals(table.length, expected.length)
        table.zip(expected).zipWithIndex.foreach { case ((actual, exp), i) =>
          assertEqualsDouble(actual, exp, 1e-10, s"Mismatch at index $i")
        }
      case other => fail(s"Expected Approximate, got $other")

  test("rival policy source returns normalized WPomcp class prior weights"):
    val input = LegacyToyFormulationInput.from(
      minimalState, defaultActions, Map.empty, heroBucket = 5, legacyRivalPriors
    )
    input.rivalPolicySource.actionPolicy(StrategicClass.Value, input.spot) match
      case BridgeResult.Approximate(weights, _) =>
        assertEquals(weights.length, 3)
        assertEqualsDouble(weights.sum, 1.0, 1e-10)
        assert(weights(0) < weights(1), s"fold ${weights(0)} should be < passive ${weights(1)}")
      case other => fail(s"Expected Approximate, got $other")

  test("LegacyPftPriorsAccessor exposes raw PFT priors"):
    val input = LegacyToyFormulationInput.from(
      minimalState, defaultActions, Map.empty, heroBucket = 5, legacyRivalPriors
    )
    input.rivalPolicySource match
      case accessor: LegacyPftPriorsAccessor =>
        assertEquals(accessor.pftActionPriors, pftPriors)
      case other => fail(s"Expected LegacyPftPriorsAccessor, got ${other.getClass}")

  test("action source fold produces HeroFold terminal"):
    val input = LegacyToyFormulationInput.from(
      minimalState, defaultActions, Map.empty, heroBucket = 5, legacyRivalPriors
    )
    input.actionSource.semanticsFor(input.spot, PokerAction.Fold) match
      case BridgeResult.Approximate(sem, _) =>
        assertEquals(sem.terminal, FormulationTerminalKind.HeroFold)
        assertEqualsDouble(sem.chipsCommitted, 0.0, 1e-10)
        assertEqualsDouble(sem.potDeltaChips, 0.0, 1e-10)
      case other => fail(s"Expected Approximate, got $other")

  test("action source call produces Continue terminal with correct chips"):
    val input = LegacyToyFormulationInput.from(
      minimalState, defaultActions, Map.empty, heroBucket = 5, legacyRivalPriors
    )
    input.actionSource.semanticsFor(input.spot, PokerAction.Call) match
      case BridgeResult.Approximate(sem, _) =>
        assertEquals(sem.terminal, FormulationTerminalKind.Continue)
        assertEqualsDouble(sem.chipsCommitted, 20.0, 1e-10)
        assertEqualsDouble(sem.potDeltaChips, 20.0, 1e-10)
      case other => fail(s"Expected Approximate, got $other")

  test("action source raise produces correct chips and allIn detection"):
    val input = LegacyToyFormulationInput.from(
      minimalState, defaultActions, Map.empty, heroBucket = 5, legacyRivalPriors
    )
    input.actionSource.semanticsFor(input.spot, PokerAction.Raise(50.0)) match
      case BridgeResult.Approximate(sem, _) =>
        assertEquals(sem.terminal, FormulationTerminalKind.Continue)
        assertEqualsDouble(sem.chipsCommitted, 50.0, 1e-10)
        assertEqualsDouble(sem.potDeltaChips, 50.0, 1e-10)
        assertEquals(sem.isAllIn, false)
      case other => fail(s"Expected Approximate, got $other")

    input.actionSource.semanticsFor(input.spot, PokerAction.Raise(1000.0)) match
      case BridgeResult.Approximate(sem, _) =>
        assertEquals(sem.isAllIn, true)
      case other => fail(s"Expected Approximate, got $other")

  test("action source river non-fold produces Showdown terminal"):
    val riverState = minimalState.copy(street = Street.River)
    val input = LegacyToyFormulationInput.from(
      riverState, defaultActions, Map.empty, heroBucket = 5, legacyRivalPriors
    )
    input.actionSource.semanticsFor(input.spot, PokerAction.Call) match
      case BridgeResult.Approximate(sem, _) =>
        assertEquals(sem.terminal, FormulationTerminalKind.Showdown)
      case other => fail(s"Expected Approximate, got $other")
