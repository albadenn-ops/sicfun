package sicfun.holdem.runtime.protocol

import sicfun.holdem.types.Street

class TableDealerTypesTest extends munit.FunSuite:
  test("SidePot: eligible seats non-empty and amount positive"):
    val ok = SidePot(amount = 100L, eligibleSeats = Set(SeatId(0), SeatId(1)))
    assertEquals(ok.amount, 100L)
    intercept[IllegalArgumentException](SidePot(0L, Set(SeatId(0))))
    intercept[IllegalArgumentException](SidePot(100L, Set.empty))

  test("TableConfig: numSeats in [2, 9]"):
    TableConfig(numSeats = 2, smallBlind = 1L, bigBlind = 2L, ante = 0L, startingStack = 200L)
    TableConfig(9, 1L, 2L, 0L, 200L)
    intercept[IllegalArgumentException](TableConfig(1, 1L, 2L, 0L, 200L))
    intercept[IllegalArgumentException](TableConfig(10, 1L, 2L, 0L, 200L))
    intercept[IllegalArgumentException](TableConfig(2, 0L, 2L, 0L, 200L)) // sb must be > 0
    intercept[IllegalArgumentException](TableConfig(2, 3L, 2L, 0L, 200L)) // bb >= sb required
    intercept[IllegalArgumentException](TableConfig(2, 1L, 2L, 0L, 10L)) // startingStack < 10*bb

  test("Street reuses sicfun.holdem.types.Street, not a new type"):
    val s: Street = Street.Flop
    assertEquals(s.expectedBoardSize, 3)

  test("HandOutcome rejects empty netChange map"):
    intercept[IllegalArgumentException](
      HandOutcome(
        potsDistributed = Vector.empty,
        netChange = Map.empty[SeatId, Long],
        events = Vector.empty
      )
    )

  test("TableSnapshot rejects seat indices >= config.numSeats"):
    val cfg6 = TableConfig(6, 1L, 2L, 0L, 200L)
    intercept[IllegalArgumentException](
      TableSnapshot(
        config = cfg6,
        heroSeat = SeatId(7),
        holeCards = Vector.empty,
        board = Vector.empty,
        stacks = Map.empty,
        contributions = Map.empty,
        street = Street.Preflop,
        actionHistory = Vector.empty,
        buttonSeat = SeatId(0),
        activeSeats = Set.empty
      )
    )
    intercept[IllegalArgumentException](
      TableSnapshot(
        config = cfg6,
        heroSeat = SeatId(0),
        holeCards = Vector.empty,
        board = Vector.empty,
        stacks = Map(SeatId(7) -> 100L),
        contributions = Map.empty,
        street = Street.Preflop,
        actionHistory = Vector.empty,
        buttonSeat = SeatId(0),
        activeSeats = Set.empty
      )
    )
