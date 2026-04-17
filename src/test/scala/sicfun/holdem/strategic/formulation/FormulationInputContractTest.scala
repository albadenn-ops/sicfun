package sicfun.holdem.strategic.formulation

import munit.FunSuite
import sicfun.core.{Card, Rank, Suit}
import sicfun.holdem.types.*
import sicfun.holdem.strategic.types.*
import sicfun.holdem.strategic.state.*

class FormulationInputContractTest extends FunSuite:

  private def minimalState: GameState =
    GameState(
      street = Street.Preflop,
      board = Board.empty,
      pot = 100.0,
      toCall = 0.0,
      position = Position.Button,
      stackSize = 1000.0,
      betHistory = Vector.empty
    )

  test("FormulationSpot holds game state and candidate actions"):
    val gs = minimalState
    val actions = Vector(PokerAction.Fold, PokerAction.Call)
    val spot = FormulationSpot(
      gameState = gs,
      candidateActions = actions,
      heroValueInput = HeroValueInput.StrengthHint(bucket = 5, source = "test"),
      rivalBeliefs = Map.empty
    )
    assertEquals(spot.gameState, gs)
    assertEquals(spot.candidateActions, actions)
    assertEquals(spot.rivalBeliefs, Map.empty[PlayerId, StrategicRivalBelief])

  test("HeroValueInput.ExactHoleCards holds hole cards"):
    val cards = HoleCards(Card(Rank.Ace, Suit.Spades), Card(Rank.King, Suit.Spades))
    val input = HeroValueInput.ExactHoleCards(cards)
    assertEquals(input.cards, cards)

  test("HeroValueInput.StrengthHint holds bucket and source"):
    val input = HeroValueInput.StrengthHint(bucket = 7, source = "estimateHeroBucket")
    assertEquals(input.bucket, 7)
    assertEquals(input.source, "estimateHeroBucket")

  test("FormulationTerminalKind has all four cases"):
    val kinds = FormulationTerminalKind.values
    assertEquals(kinds.length, 4)
    assertEquals(kinds.toSet, Set(
      FormulationTerminalKind.Continue,
      FormulationTerminalKind.HeroFold,
      FormulationTerminalKind.RivalFold,
      FormulationTerminalKind.Showdown
    ))

  test("FormulationActionSemantics captures action properties"):
    val sem = FormulationActionSemantics(
      chipsCommitted = 50.0,
      potDeltaChips = 50.0,
      isAllIn = false,
      terminal = FormulationTerminalKind.Continue,
      advancesStreet = false
    )
    assertEqualsDouble(sem.chipsCommitted, 50.0, 1e-10)
    assertEqualsDouble(sem.potDeltaChips, 50.0, 1e-10)
    assertEquals(sem.isAllIn, false)
    assertEquals(sem.terminal, FormulationTerminalKind.Continue)
    assertEquals(sem.advancesStreet, false)
