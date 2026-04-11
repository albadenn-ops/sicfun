package sicfun.holdem.bench

import sicfun.core.DiscreteDistribution
import sicfun.holdem.cfr.{HoldemCfrConfig, HoldemCfrDecisionPolicy, HoldemCfrSolution, HoldemCfrSolver}
import sicfun.holdem.types.*
import sicfun.holdem.bench.BenchSupport.{card, hole}

/** A/B benchmark: HoldemCfrSolver with `scala` vs `scala-fixed`.
  *
  * Usage:
  *   sbt "runMain sicfun.holdem.bench.HoldemCfrFixedBenchmark [warmup] [runs] [operation] [scenario]"
  *
  * operation:
  *   - `full`     => HoldemCfrSolver.solve
  *   - `decision` => HoldemCfrSolver.solveDecisionPolicy
  *
  * scenario:
  *   - `preflop`
  *   - `turn`
  */
object HoldemCfrFixedBenchmark:
  private enum Operation:
    case Full
    case Decision

  private final case class Spot(
      hero: HoleCards,
      state: GameState,
      posterior: DiscreteDistribution[HoleCards],
      candidateActions: Vector[PokerAction],
      config: HoldemCfrConfig,
      label: String
  )

  /** Entry point. Parses CLI args, constructs the benchmark spot, then runs an interleaved
    * A/B comparison between `scala` (Double regret accumulators) and `scala-fixed` (Int32
    * fixed-point regret accumulators). Interleaving on even/odd runs mitigates JIT warmup
    * and GC ordering bias. After timing, runs both providers once more to capture solution
    * outputs for the correctness diff report.
    */
  def main(args: Array[String]): Unit =
    val warmup = if args.length > 0 then args(0).toInt else 3
    val runs = if args.length > 1 then args(1).toInt else 12
    val operation =
      if args.length > 2 then parseOperation(args(2))
      else Operation.Full
    val scenario = if args.length > 3 then args(3).toLowerCase(java.util.Locale.ROOT) else "preflop"
    val spot = benchmarkSpot(scenario)

    println("=== Holdem CFR Fixed A/B Benchmark ===")
    println(s"Operation: ${operation.toString.toLowerCase}, Scenario: ${spot.label}")
    println(s"Warmup: $warmup, Runs: $runs, Iterations: ${spot.config.iterations}")
    println()

    // Warmup both providers to stabilize JIT compilation before measurement.
    var w = 0
    while w < warmup do
      runWithProvider("scala", operation, spot)
      runWithProvider("scala-fixed", operation, spot)
      w += 1

    // Interleaved measurement: alternate which provider runs first per iteration
    // to cancel out any systematic ordering bias (cache state, GC pressure, etc.).
    val scalaTimes = new Array[Long](runs)
    val fixedTimes = new Array[Long](runs)
    var r = 0
    while r < runs do
      if r % 2 == 0 then
        scalaTimes(r) = timeOnce { runWithProvider("scala", operation, spot) }
        fixedTimes(r) = timeOnce { runWithProvider("scala-fixed", operation, spot) }
      else
        fixedTimes(r) = timeOnce { runWithProvider("scala-fixed", operation, spot) }
        scalaTimes(r) = timeOnce { runWithProvider("scala", operation, spot) }
      r += 1

    val scalaResult = runWithProvider("scala", operation, spot)
    val fixedResult = runWithProvider("scala-fixed", operation, spot)

    val scalaMedian = median(scalaTimes)
    val fixedMedian = median(fixedTimes)
    val speedup = scalaMedian.toDouble / fixedMedian.toDouble

    println("--- Scala (baseline) ---")
    printStats("Scala ", scalaTimes)
    println()
    println("--- Scala-Fixed ---")
    printStats("Fixed ", fixedTimes)
    println()
    println(f"Speedup (median): $speedup%.3fx")
    println()
    println("--- Correctness ---")
    reportDiff(operation, spot.candidateActions, scalaResult, fixedResult)

  private def parseOperation(raw: String): Operation =
    raw.toLowerCase(java.util.Locale.ROOT) match
      case "full" => Operation.Full
      case "decision" | "root" => Operation.Decision
      case other =>
        throw new IllegalArgumentException(s"unsupported operation: $other")

  /** Constructs a fixed benchmark scenario with hero hand, board, villain posterior,
    * candidate actions, and CFR config. The "preflop" scenario has no board cards;
    * the "turn" scenario has a 4-card board with deeper equity computation.
    */
  private def benchmarkSpot(name: String): Spot =
    name match
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
      case other =>
        throw new IllegalArgumentException(s"unsupported scenario: $other")

  /** Runs a single CFR solve under the given provider (set via system property).
    * Returns Left(solution) for full solves or Right(policy) for decision-only solves.
    */
  private def runWithProvider(
      provider: String,
      operation: Operation,
      spot: Spot
  ): Either[HoldemCfrSolution, HoldemCfrDecisionPolicy] =
    withSystemProperty("sicfun.cfr.provider", provider) {
      operation match
        case Operation.Full =>
          Left(
            HoldemCfrSolver.solve(
              hero = spot.hero,
              state = spot.state,
              villainPosterior = spot.posterior,
              candidateActions = spot.candidateActions,
              config = spot.config
            )
          )
        case Operation.Decision =>
          Right(
            HoldemCfrSolver.solveDecisionPolicy(
              hero = spot.hero,
              state = spot.state,
              villainPosterior = spot.posterior,
              candidateActions = spot.candidateActions,
              config = spot.config
            )
          )
    }

  /** Prints a correctness comparison between the Double baseline and fixed-point result.
    * For full solves, reports EV difference and per-action probability diffs.
    * For decision-only solves, reports best-action agreement and probability diffs.
    */
  private def reportDiff(
      operation: Operation,
      actions: Vector[PokerAction],
      baseline: Either[HoldemCfrSolution, HoldemCfrDecisionPolicy],
      fixed: Either[HoldemCfrSolution, HoldemCfrDecisionPolicy]
  ): Unit =
    operation match
      case Operation.Full =>
        val left = baseline.left.getOrElse(throw new IllegalStateException("expected full baseline"))
        val right = fixed.left.getOrElse(throw new IllegalStateException("expected full fixed result"))
        println(
          f"EV baseline=${left.expectedValuePlayer0}%.6f fixed=${right.expectedValuePlayer0}%.6f diff=${math.abs(left.expectedValuePlayer0 - right.expectedValuePlayer0)}%.6f"
        )
        println(s"Best action baseline=${left.bestAction} fixed=${right.bestAction}")
        actions.foreach { action =>
          val pBaseline = left.actionProbabilities.getOrElse(action, 0.0)
          val pFixed = right.actionProbabilities.getOrElse(action, 0.0)
          println(f"$action baseline=$pBaseline%.6f fixed=$pFixed%.6f diff=${math.abs(pBaseline - pFixed)}%.6f")
        }
      case Operation.Decision =>
        val left = baseline.toOption.getOrElse(throw new IllegalStateException("expected decision baseline"))
        val right = fixed.toOption.getOrElse(throw new IllegalStateException("expected decision fixed result"))
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
