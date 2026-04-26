package sicfun.holdem.runtime.agent

import sicfun.holdem.runtime.protocol.*
import sicfun.holdem.runtime.PlaceholderMarker
import sicfun.holdem.strategic.safety.{SafetyBellman, NeverDetect}
import sicfun.holdem.types.{PokerAction, Street}
import java.nio.file.Files

class StrategicAgentWiringTest extends munit.FunSuite:

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

  private def makeAgent(): StrategicAgent =
    val tmp = Files.createTempFile("bp", ".blueprint").toFile
    BlueprintStore.write(tmp, BlueprintHeader("SICFBP01", 1, 0L, "h", 6, 0, 5), Map.empty)
    val store = BlueprintStore.load(tmp, "h")
    StrategicAgent(
      seatId = SeatId(0),
      store = store,
      hasher = PlaceholderInfostateHasher(),
      embedding = PlaceholderMdpEmbedding(),
      oracle = ZeroExploitabilityOracle(),
      detector = NeverDetect,
      rngSeed = 1L,
      abstractActions = Vector(
        PokerAction.Fold,
        PokerAction.Call,
        PokerAction.Raise(4.0),
        PokerAction.Raise(8.0),
        PokerAction.Raise(200.0)
      )
    )

  test("safeActionSet: indices in range, can be empty under tight bound"):
    val losses = Array(Array(10.0, 10.0, 10.0), Array(0.0, 0.0, 0.0))
    val transitions: (Int, Int, Int) => IndexedSeq[(Int, Double)] =
      (s, _, _) => if s == 0 then Vector(1 -> 1.0) else Vector(1 -> 1.0)
    val _ = SafetyBellman.computeBStar(
      losses,
      0.95,
      transitions,
      numProfiles = 1,
      terminalStates = Set(1)
    )
    val tightBound = Array(0.1, 0.0)
    val safe = SafetyBellman.safeActionSet(0, tightBound, losses, 0.95, transitions, 1)
    assert(safe.forall(i => i >= 0 && i < 3), s"indices out of range: $safe")
    assertEquals(safe, IndexedSeq.empty[Int],
      s"expected empty safe set with tight bound, got $safe")

  test("safeFeasibleAction: empty safe set falls back to global argmax"):
    val qValues = Array(5.0, 10.0, 3.0)
    val chosen = SafetyBellman.safeFeasibleAction(qValues, IndexedSeq.empty[Int])
    assertEquals(chosen, 1, "expected argmax index 1 (qValues(1) = 10)")

  test("safeFeasibleAction: non-empty safe set picks argmax restricted to safe"):
    val qValues = Array(5.0, 10.0, 3.0)
    val chosen = SafetyBellman.safeFeasibleAction(qValues, IndexedSeq(0, 2))
    assertEquals(chosen, 0)

  test("decide returns an action in the legal set"):
    val agent = makeAgent()
    val legal = Set[PokerAction](
      PokerAction.Fold,
      PokerAction.Call,
      PokerAction.Raise(4.0)
    )
    val a = agent.decide(snap(), legal)
    assert(legal.contains(a), s"$a not in $legal")

  test("scanPlaceholders on StrategicAgent finds MdpEmbedding + Oracle + Hasher"):
    val agent = makeAgent()
    val found = PlaceholderMarker.scanPlaceholders(agent)
    val reasons = found.map(_.placeholderReason).mkString("|")
    assert(reasons.contains("MdpEmbedding"),
      s"expected MdpEmbedding placeholder reason, got: $reasons")
    assert(reasons.contains("ExploitabilityOracle"),
      s"expected ExploitabilityOracle placeholder reason, got: $reasons")
    assert(reasons.contains("InfostateHasher"),
      s"expected InfostateHasher placeholder reason, got: $reasons")
