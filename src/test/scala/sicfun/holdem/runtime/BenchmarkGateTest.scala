package sicfun.holdem.runtime

import sicfun.holdem.runtime.agent.*
import sicfun.holdem.strategic.safety.NeverDetect
import sicfun.holdem.types.PokerAction
import java.nio.file.Files

class BenchmarkGateTest extends munit.FunSuite:

  private def strategicAgent(): StrategicAgent =
    val tmp = Files.createTempFile("bp", ".blueprint").toFile
    BlueprintStore.write(tmp, BlueprintHeader("SICFBP01", 1, 0L, "h", 9, 0, 5), Map.empty)
    val store = BlueprintStore.load(tmp, "h")
    StrategicAgent(
      seatId = sicfun.holdem.runtime.protocol.SeatId(0),
      store = store,
      hasher = PlaceholderInfostateHasher(),
      embedding = PlaceholderMdpEmbedding(),
      oracle = ZeroExploitabilityOracle(),
      detector = NeverDetect,
      rngSeed = 1L,
      abstractActions = Vector(PokerAction.Fold, PokerAction.Call,
        PokerAction.Raise(4.0), PokerAction.Raise(8.0), PokerAction.Raise(200.0))
    )

  private def blueprintAgent(): BlueprintOnlyAgent =
    val tmp = Files.createTempFile("bp2", ".blueprint").toFile
    BlueprintStore.write(tmp, BlueprintHeader("SICFBP01", 1, 0L, "h", 9, 0, 5), Map.empty)
    val store = BlueprintStore.load(tmp, "h")
    BlueprintOnlyAgent(
      sicfun.holdem.runtime.protocol.SeatId(1), store, PlaceholderInfostateHasher(), 2L)

  test("benchmarkMode = true + StrategicAgent → Left(violation) listing all placeholders"):
    val agents = Vector[SeatAgent](strategicAgent())
    val result = BenchmarkGate.check(agents, benchmarkMode = true)
    assert(result.isLeft)
    result.left.foreach { v =>
      val msg = v.getMessage
      assert(msg.contains("MdpEmbedding"), msg)
      assert(msg.contains("ExploitabilityOracle"), msg)
      assert(msg.contains("InfostateHasher"), msg)
    }

  test("benchmarkMode = false → Right(()) even with placeholders"):
    val agents = Vector[SeatAgent](strategicAgent())
    val result = BenchmarkGate.check(agents, benchmarkMode = false)
    assert(result.isRight)

  test("benchmarkMode = true + BlueprintOnlyAgent only → still blocks on PlaceholderInfostateHasher"):
    val agents = Vector[SeatAgent](blueprintAgent())
    val result = BenchmarkGate.check(agents, benchmarkMode = true)
    assert(result.isLeft,
      "A.1 has no non-placeholder hasher; real InfostateHasher arrives in A.2")
