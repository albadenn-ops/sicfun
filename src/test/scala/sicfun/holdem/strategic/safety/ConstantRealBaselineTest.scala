package sicfun.holdem.strategic.safety

import munit.FunSuite
import sicfun.holdem.types.{Board, PokerAction, Position, Street}
import sicfun.holdem.strategic.types.*            // StrategicClass, Chips, PlayerId, TableMap, Seat, SeatStatus
import sicfun.holdem.strategic.state.PublicState
import sicfun.holdem.engine.StrategicEngine

class ConstantRealBaselineTest extends FunSuite:
  private def ps(street: Street, board: Board): PublicState =
    val hero = PlayerId("__test__")
    PublicState(
      street = street,
      board = board,
      pot = Chips(10.0),
      stacks = TableMap(hero, Vector(Seat(hero, Position.SmallBlind, SeatStatus.Active, Chips(500.0)))),
      actionHistory = Vector.empty
    )
  private val flop = ps(Street.Flop, Board.empty)
  private val base = ConstantRealBaseline(StrategicEngine.defaultActionPriors)

  test("returns the configured constant for a known (class,action)"):
    assertEqualsDouble(base.probability(StrategicClass.Value, PokerAction.Category.Raise, None, flop), 0.20, 1e-12)
    assertEqualsDouble(base.probability(StrategicClass.Bluff, PokerAction.Category.Raise, None, flop), 0.65, 1e-12)

  test("falls back to 0.25 for a key absent from the map"):
    val sparse = ConstantRealBaseline(Map.empty)
    assertEqualsDouble(sparse.probability(StrategicClass.Mixed, PokerAction.Category.Call, None, flop), 0.25, 1e-12)

  test("ignores board/street/sizing (board-blind floor)"):
    val river = ps(Street.River, Board.empty)
    assertEqualsDouble(
      base.probability(StrategicClass.Value, PokerAction.Category.Check, None, flop),
      base.probability(StrategicClass.Value, PokerAction.Category.Check, None, river),
      1e-12
    )
