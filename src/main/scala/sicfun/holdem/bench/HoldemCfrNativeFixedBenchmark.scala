package sicfun.holdem.bench

import sicfun.core.DiscreteDistribution
import sicfun.holdem.cfr.{HoldemCfrConfig, HoldemCfrDecisionPolicy, HoldemCfrNativeRuntime, HoldemCfrSolution, HoldemCfrSolver}
import sicfun.holdem.types.*
import sicfun.holdem.bench.BenchSupport.{card, hole}

/** A/B benchmark: Holdem native double vs native fixed-point for CPU or CUDA-compiled CFR providers.
  *
  * Requires the matching native CFR library, usually via either:
  * `-Dsicfun.cfr.native.cpu.path=<abs-path-to-sicfun_cfr_native.dll>`
  * or `sicfun_CFR_NATIVE_CPU_PATH=<abs-path-to-sicfun_cfr_native.dll>`
  * or for the CUDA-compiled bridge:
  * `-Dsicfun.cfr.native.gpu.path=<abs-path-to-sicfun_cfr_cuda.dll>`
  * or `sicfun_CFR_NATIVE_GPU_PATH=<abs-path-to-sicfun_cfr_cuda.dll>`
  *
  * Usage:
  *   sbt "runMain sicfun.holdem.bench.HoldemCfrNativeFixedBenchmark [warmup] [runs] [operation] [scenario] [backend]"
  */
object HoldemCfrNativeFixedBenchmark:
  private enum Operation:
    case Full
    case Decision

  private enum Backend:
    case Cpu
    case Gpu

  private final case class Spot(
      hero: HoleCards,
      state: GameState,
      posterior: DiscreteDistribution[HoleCards],
      candidateActions: Vector[PokerAction],
      config: HoldemCfrConfig,
      label: String
  )

  def main(args: Array[String]): Unit =
    val warmup = if args.length > 0 then args(0).toInt else 3
    val runs = if args.length > 1 then args(1).toInt else 10
    val operation =
      if args.length > 2 then parseOperation(args(2))
      else Operation.Full
    val scenario = if args.length > 3 then args(3).toLowerCase(java.util.Locale.ROOT) else "turn"
    val backend = if args.length > 4 then parseBackend(args(4)) else Backend.Cpu
    val spot = benchmarkSpot(scenario)
    val providerBaseline = providerName(backend, fixed = false)
    val providerFixed = providerName(backend, fixed = true)

    println("=== Holdem CFR Native Fixed A/B Benchmark ===")
    println(s"Backend: ${backend.toString.toLowerCase}, Operation: ${operation.toString.toLowerCase}, Scenario: ${spot.label}")
    println(s"Warmup: $warmup, Runs: $runs, Iterations: ${spot.config.iterations}")
    println(
      s"Native CPU path: property=${sys.props.getOrElse("sicfun.cfr.native.cpu.path", "<unset>")} " +
        s"env=${sys.env.getOrElse("sicfun_CFR_NATIVE_CPU_PATH", "<unset>")}"
    )
    println(
      s"Native GPU path: property=${sys.props.getOrElse("sicfun.cfr.native.gpu.path", "<unset>")} " +
        s"env=${sys.env.getOrElse("sicfun_CFR_NATIVE_GPU_PATH", "<unset>")}"
    )
    println()

    var w = 0
    while w < warmup do
      runWithProvider(providerBaseline, operation, spot)
      runWithProvider(providerFixed, operation, spot)
      w += 1

    val nativeTimes = new Array[Long](runs)
    val fixedTimes = new Array[Long](runs)
    var r = 0
    while r < runs do
      if r % 2 == 0 then
        nativeTimes(r) = timeOnce { runWithProvider(providerBaseline, operation, spot) }
        fixedTimes(r) = timeOnce { runWithProvider(providerFixed, operation, spot) }
      else
        fixedTimes(r) = timeOnce { runWithProvider(providerFixed, operation, spot) }
        nativeTimes(r) = timeOnce { runWithProvider(providerBaseline, operation, spot) }
      r += 1

    val nativeResult = runWithProvider(providerBaseline, operation, spot)
    val fixedResult = runWithProvider(providerFixed, operation, spot)

    val nativeMedian = median(nativeTimes)
    val fixedMedian = median(fixedTimes)
    val speedup = nativeMedian.toDouble / fixedMedian.toDouble

    println(s"--- ${providerBaseline} (double) ---")
    printStats("Native ", nativeTimes)
    println()
    println(s"--- ${providerFixed} ---")
    printStats("Fixed  ", fixedTimes)
    println()
    println(f"Speedup (median): $speedup%.3fx")
    println()
    println("--- Correctness ---")
    reportDiff(operation, spot.candidateActions, nativeResult, fixedResult)

  private def parseOperation(raw: String): Operation =
    raw.toLowerCase(java.util.Locale.ROOT) match
      case "full" => Operation.Full
      case "decision" | "root" => Operation.Decision
      case other => throw new IllegalArgumentException(s"unsupported operation: $other")

  private def parseBackend(raw: String): Backend =
    raw.toLowerCase(java.util.Locale.ROOT) match
      case "cpu" | "native-cpu" => Backend.Cpu
      case "gpu" | "cuda" | "native-gpu" => Backend.Gpu
      case other => throw new IllegalArgumentException(s"unsupported backend: $other")

  private def providerName(backend: Backend, fixed: Boolean): String =
    (backend, fixed) match
      case (Backend.Cpu, false) => "native-cpu"
      case (Backend.Cpu, true) => "native-cpu-fixed"
      case (Backend.Gpu, false) => "native-gpu"
      case (Backend.Gpu, true) => "native-gpu-fixed"

  private def benchmarkSpot(name: String): Spot =
    name match
      case "preflop" =>
        Spot(
          hero = hole("Ac", "Ad"),
          state = GameState(
            street = Street.Preflop,
            board = Board.empty,
            pot = 6.0,
            toCall = 2.0,
            position = Position.Button,
            stackSize = 100.0,
            betHistory = Vector.empty
          ),
          posterior = DiscreteDistribution(
            Map(
              hole("7c", "2d") -> 0.6,
              hole("Kc", "Qd") -> 0.3,
              hole("Ks", "Kh") -> 0.1
            )
          ),
          candidateActions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(8.0)),
          config = HoldemCfrConfig(
            iterations = 1_200,
            averagingDelay = 100,
            maxVillainHands = 32,
            equityTrials = 1_000,
            preferNativeBatch = true,
            rngSeed = 37L
          ),
          label = "preflop premium / 3-action root"
        )
      case "turn" =>
        Spot(
          hero = hole("Ac", "Kd"),
          state = GameState(
            street = Street.Turn,
            board = Board.from(Vector(card("2c"), card("7d"), card("Jh"), card("Qc"))),
            pot = 18.0,
            toCall = 4.0,
            position = Position.Button,
            stackSize = 82.0,
            betHistory = Vector(BetAction(1, PokerAction.Raise(4.0)))
          ),
          posterior = DiscreteDistribution(
            Map(
              hole("As", "Qd") -> 0.35,
              hole("Ts", "9s") -> 0.30,
              hole("Qh", "Js") -> 0.20,
              hole("7c", "7s") -> 0.15
            )
          ),
          candidateActions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(12.0)),
          config = HoldemCfrConfig(
            iterations = 700,
            averagingDelay = 100,
            maxVillainHands = 32,
            equityTrials = 700,
            preferNativeBatch = true,
            rngSeed = 41L
          ),
          label = "turn 4-card board / 3-action root"
        )
      case other =>
        throw new IllegalArgumentException(s"unsupported scenario: $other")

  private def runWithProvider(
      provider: String,
      operation: Operation,
      spot: Spot
  ): Either[HoldemCfrSolution, HoldemCfrDecisionPolicy] =
    withSystemProperty("sicfun.cfr.provider", provider) {
      operation match
        case Operation.Full =>
          Left(HoldemCfrSolver.solve(spot.hero, spot.state, spot.posterior, spot.candidateActions, spot.config))
        case Operation.Decision =>
          Right(HoldemCfrSolver.solveDecisionPolicy(spot.hero, spot.state, spot.posterior, spot.candidateActions, spot.config))
    }

  private def reportDiff(
      operation: Operation,
      actions: Vector[PokerAction],
      baseline: Either[HoldemCfrSolution, HoldemCfrDecisionPolicy],
      fixed: Either[HoldemCfrSolution, HoldemCfrDecisionPolicy]
  ): Unit =
    operation match
      case Operation.Full =>
        val left = baseline.left.getOrElse(throw new IllegalStateException("expected full result"))
        val right = fixed.left.getOrElse(throw new IllegalStateException("expected full result"))
        println(s"Providers baseline=${left.provider} fixed=${right.provider}")
        println(f"EV baseline=${left.expectedValuePlayer0}%.6f fixed=${right.expectedValuePlayer0}%.6f diff=${math.abs(left.expectedValuePlayer0 - right.expectedValuePlayer0)}%.6f")
        println(s"Best action baseline=${left.bestAction} fixed=${right.bestAction}")
        actions.foreach { action =>
          val pBaseline = left.actionProbabilities.getOrElse(action, 0.0)
          val pFixed = right.actionProbabilities.getOrElse(action, 0.0)
          println(f"$action baseline=$pBaseline%.6f fixed=$pFixed%.6f diff=${math.abs(pBaseline - pFixed)}%.6f")
        }
      case Operation.Decision =>
        val left = baseline.toOption.getOrElse(throw new IllegalStateException("expected decision result"))
        val right = fixed.toOption.getOrElse(throw new IllegalStateException("expected decision result"))
        println(s"Providers baseline=${left.provider} fixed=${right.provider}")
        println(s"Best action baseline=${left.bestAction} fixed=${right.bestAction}")
        actions.foreach { action =>
          val pBaseline = left.actionProbabilities.getOrElse(action, 0.0)
          val pFixed = right.actionProbabilities.getOrElse(action, 0.0)
          println(f"$action baseline=$pBaseline%.6f fixed=$pFixed%.6f diff=${math.abs(pBaseline - pFixed)}%.6f")
        }

  private def withSystemProperty[A](key: String, value: String)(thunk: => A): A =
    val previous = sys.props.get(key)
    System.setProperty(key, value)
    try thunk
    finally
      previous match
        case Some(existing) => System.setProperty(key, existing)
        case None => System.clearProperty(key)
      HoldemCfrNativeRuntime.resetLoadCacheForTests()
      HoldemCfrSolver.resetAutoProviderForTests()

  private def timeOnce(body: => Any): Long =
    val started = System.nanoTime()
    body
    System.nanoTime() - started

  private def median(values: Array[Long]): Long =
    val sorted = values.sorted
    sorted(sorted.length / 2)

  private def printStats(label: String, times: Array[Long]): Unit =
    val sorted = times.sorted
    val count = sorted.length
    val median = sorted(count / 2)
    val p25 = sorted(count / 4)
    val p75 = sorted((3 * count) / 4)
    val mean = times.map(_.toDouble).sum / count
    println(
      f"$label median=${median / 1e6}%.2f ms  p25=${p25 / 1e6}%.2f  p75=${p75 / 1e6}%.2f  min=${sorted.head / 1e6}%.2f  max=${sorted.last / 1e6}%.2f  mean=${mean / 1e6}%.2f ms"
    )
