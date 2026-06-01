package sicfun.holdem.runtime.agent

import sicfun.holdem.runtime.protocol.*
import sicfun.holdem.types.{PokerAction, Street}
import java.nio.file.Files

class BlueprintOnlyAgentTest extends munit.FunSuite:
  private def emptyStore(): BlueprintStore =
    val tmp = Files.createTempFile("bp", ".blueprint").toFile
    BlueprintStore.write(tmp, BlueprintHeader("SICFBP01", 1, 0L, "h", 6, 0, 5), Map.empty)
    BlueprintStore.load(tmp, "h")

  private def snap(): TableSnapshot = TableSnapshot(
    config = TableConfig(6, 1L, 2L, 0L, 200L),
    heroSeat = SeatId(0),
    holeCards = Vector.empty,
    board = Vector.empty,
    stacks = (0 until 6).map(i => SeatId(i) -> 200L).toMap,
    contributions = (0 until 6).map(i => SeatId(i) -> 0L).toMap,
    street = Street.Preflop,
    actionHistory = Vector.empty,
    buttonSeat = SeatId(0),
    activeSeats = (0 until 6).map(SeatId(_)).toSet
  )

  test("always returns a legal action"):
    val agent = BlueprintOnlyAgent(SeatId(0), emptyStore(), PlaceholderInfostateHasher(), 1L)
    val legal = Set[PokerAction](PokerAction.Fold, PokerAction.Call, PokerAction.Raise(6.0))
    val a = agent.decide(snap(), legal)
    assert(legal.contains(a), s"$a not in $legal")

  test("determinism under same seed"):
    val store = emptyStore()
    val a1 = BlueprintOnlyAgent(SeatId(0), store, PlaceholderInfostateHasher(), 7L)
    val a2 = BlueprintOnlyAgent(SeatId(0), store, PlaceholderInfostateHasher(), 7L)
    val legal = Set[PokerAction](PokerAction.Fold, PokerAction.Call, PokerAction.Raise(6.0))
    assertEquals(a1.decide(snap(), legal), a2.decide(snap(), legal))

  test("Fold-when-cannot-fold -> Passive (Check if free, else Call)"):
    val agent = BlueprintOnlyAgent(SeatId(0), emptyStore(), PlaceholderInfostateHasher(), 1L)
    val legal = Set[PokerAction](PokerAction.Check, PokerAction.Raise(10.0))
    (1 to 50).foreach { _ =>
      val a = agent.decide(snap(), legal)
      assert(a != PokerAction.Fold, s"agent returned Fold when not legal: $a")
      assert(legal.contains(a), s"$a not legal")
    }
