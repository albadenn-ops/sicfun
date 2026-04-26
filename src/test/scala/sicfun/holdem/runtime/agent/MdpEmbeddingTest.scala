package sicfun.holdem.runtime.agent

import sicfun.holdem.runtime.protocol.*
import sicfun.holdem.types.{PokerAction, Street}

class MdpEmbeddingTest extends munit.FunSuite:
  private def snap(): TableSnapshot = TableSnapshot(
    config = TableConfig(6, 1L, 2L, 0L, 200L),
    heroSeat = SeatId(0),
    holeCards = Vector.empty, board = Vector.empty,
    stacks = (0 until 6).map(i => SeatId(i) -> 200L).toMap,
    contributions = (0 until 6).map(i => SeatId(i) -> 0L).toMap,
    street = Street.Preflop, actionHistory = Vector.empty,
    buttonSeat = SeatId(0), activeSeats = (0 until 6).map(SeatId(_)).toSet
  )

  test("PlaceholderMdpEmbedding: extends PlaceholderMarker with non-empty reason"):
    val emb = PlaceholderMdpEmbedding()
    val built = emb.build(snap(), Vector(PokerAction.Fold, PokerAction.Call), numProfiles = 3)
    assert(emb.placeholderReason.nonEmpty)
    assertEquals(built.robustLosses.length, built.numStates)
    built.robustLosses.foreach(row => assertEquals(row.length, 2))

  test("ZeroExploitabilityOracle: returns 0 and extends PlaceholderMarker"):
    val oracle = ZeroExploitabilityOracle()
    assert(oracle.placeholderReason.nonEmpty)
    assertEquals(oracle.exploitabilityFn(0.7)(0.5), 0.0)
