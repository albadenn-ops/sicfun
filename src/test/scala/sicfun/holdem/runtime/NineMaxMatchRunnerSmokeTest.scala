package sicfun.holdem.runtime

import sicfun.holdem.runtime.agent.*
import sicfun.holdem.runtime.protocol.*
import sicfun.holdem.strategic.safety.NeverDetect
import sicfun.holdem.types.PokerAction
import java.nio.file.Files

class NineMaxMatchRunnerSmokeTest extends munit.FunSuite:

  override val munitTimeout = scala.concurrent.duration.Duration(120, "seconds")

  private def store9(): BlueprintStore =
    val tmp = Files.createTempFile("bp", ".blueprint").toFile
    BlueprintStore.write(tmp, BlueprintHeader("SICFBP01", 1, 0L, "h", 9, 0, 5), Map.empty)
    BlueprintStore.load(tmp, "h")

  test("1000 hands with 9 BlueprintOnlyAgents: chip conservation per hand, per match"):
    val store = store9()
    val cfg = TableConfig(9, 1L, 2L, 0L, 200L)

    val agents: Vector[SeatAgent] = (0 until 9).map { i =>
      BlueprintOnlyAgent(SeatId(i), store, PlaceholderInfostateHasher(), rngSeed = i.toLong)
    }.toVector

    val result = NineMaxMatchRunner(
      tableConfig = cfg,
      agents = agents,
      numHands = 1000,
      rngSeed = 42L,
      matchId = "a1-smoke",
      strictNative = false,
      benchmarkMode = false
    ).run()

    assertEquals(result.handsPlayed, 1000)

    val netSum = result.netBySeat.values.sum
    assertEquals(netSum, 0L, s"per-match net sum != 0: $netSum")

    result.ci95BySeat.foreach { (s, ci) =>
      assert(
        ci.lower <= 0.0 && ci.upper >= 0.0,
        s"seat ${s.index} IC95=$ci does not cross zero"
      )
    }

    val lines = scala.io.Source.fromFile(result.matchLogPath).getLines().toVector
    assertEquals(lines.size, 1000)

  test("benchmarkMode = true with StrategicAgent -> BenchmarkGateViolation listing placeholders"):
    val store = store9()
    val cfg = TableConfig(9, 1L, 2L, 0L, 200L)
    val actions = Vector(
      PokerAction.Fold,
      PokerAction.Call,
      PokerAction.Raise(4.0),
      PokerAction.Raise(8.0),
      PokerAction.Raise(200.0)
    )

    val strategic = StrategicAgent(
      seatId = SeatId(0),
      store = store,
      hasher = PlaceholderInfostateHasher(),
      embedding = PlaceholderMdpEmbedding(),
      oracle = ZeroExploitabilityOracle(),
      detector = NeverDetect,
      rngSeed = 7L,
      abstractActions = actions
    )
    val rest: Vector[SeatAgent] = (1 until 9).map { i =>
      BlueprintOnlyAgent(SeatId(i), store, PlaceholderInfostateHasher(), i.toLong)
    }.toVector

    val thrown = intercept[BenchmarkGateViolation](
      NineMaxMatchRunner(
        cfg,
        strategic +: rest,
        100,
        1L,
        "a1-gate",
        strictNative = false,
        benchmarkMode = true
      ).run()
    )
    val msg = thrown.getMessage
    assert(msg.contains("MdpEmbedding"), msg)
    assert(msg.contains("ExploitabilityOracle"), msg)
    assert(msg.contains("InfostateHasher"), msg)
