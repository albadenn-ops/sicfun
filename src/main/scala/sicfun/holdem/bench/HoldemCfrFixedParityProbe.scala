package sicfun.holdem.bench

import sicfun.core.{Card, DiscreteDistribution}
import sicfun.holdem.cfr.{HoldemCfrConfig, HoldemCfrNativeRuntime, HoldemCfrSolver}
import sicfun.holdem.types.*

/** Probes when fixed-point native CFR starts diverging from the JVM fixed baseline. */
object HoldemCfrFixedParityProbe:
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

  private sealed trait ProbeResult:
    def provider: String
    def bestAction: PokerAction
    def actionProbabilities: Map[PokerAction, Double]

  private final case class FullResult(
      provider: String,
      bestAction: PokerAction,
      actionProbabilities: Map[PokerAction, Double],
      expectedValuePlayer0: Double
  ) extends ProbeResult

  private final case class DecisionResult(
      provider: String,
      bestAction: PokerAction,
      actionProbabilities: Map[PokerAction, Double]
  ) extends ProbeResult

  private def card(token: String): Card =
    Card.parse(token).getOrElse(throw new IllegalArgumentException(s"invalid card: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  def main(args: Array[String]): Unit =
    val scenario = if args.length > 0 then args(0).toLowerCase(java.util.Locale.ROOT) else "turn"
    val operation = if args.length > 1 then parseOperation(args(1)) else Operation.Full
    val iterations = if args.length > 2 then parseIterations(args(2)) else defaultIterations(scenario)
    val providers = if args.length > 3 then parseProviders(args(3)) else Vector("scala-fixed", "native-cpu-fixed", "native-gpu-fixed")
    val spot = benchmarkSpot(scenario)

    println("=== Holdem CFR Fixed Parity Probe ===")
    println(s"Scenario: ${spot.label}")
    println(s"Operation: ${operation.toString.toLowerCase}")
    println(s"Iterations: ${iterations.mkString(", ")}")
    println(s"Providers: ${providers.mkString(", ")}")
    println(
      s"Native CPU path: property=${sys.props.getOrElse("sicfun.cfr.native.cpu.path", "<unset>")} " +
        s"env=${sys.env.getOrElse("sicfun_CFR_NATIVE_CPU_PATH", "<unset>")}"
    )
    println(
      s"Native GPU path: property=${sys.props.getOrElse("sicfun.cfr.native.gpu.path", "<unset>")} " +
        s"env=${sys.env.getOrElse("sicfun_CFR_NATIVE_GPU_PATH", "<unset>")}"
    )
    println()

    iterations.foreach { iterationCount =>
      val config =
        spot.config.copy(
          iterations = iterationCount,
          averagingDelay = math.min(spot.config.averagingDelay, math.max(1, iterationCount / 4))
        )
      println(s"--- iterations=$iterationCount averagingDelay=${config.averagingDelay} ---")

      val results = providers.map { provider =>
        provider -> runWithProvider(provider, operation, spot, config)
      }.toMap

      val baseline =
        results.getOrElse("scala-fixed", throw new IllegalStateException("scala-fixed baseline is required"))

      providers.foreach { provider =>
        val result = results(provider)
        val maxActionDiff = maxProbabilityDiff(spot.candidateActions, baseline, result)
        val bestChanged = result.bestAction != baseline.bestAction
        val evSummary =
          (baseline, result) match
            case (left: FullResult, right: FullResult) =>
              f" ev=${right.expectedValuePlayer0}%.6f evDiff=${math.abs(right.expectedValuePlayer0 - left.expectedValuePlayer0)}%.6f"
            case _ =>
              ""

        println(
          s"$provider actual=${result.provider} best=${result.bestAction} bestChanged=$bestChanged " +
            f"maxActionDiff=$maxActionDiff%.6f$evSummary"
        )
        if provider != "scala-fixed" then
          spot.candidateActions.foreach { action =>
            val baselineProbability = baseline.actionProbabilities.getOrElse(action, 0.0)
            val providerProbability = result.actionProbabilities.getOrElse(action, 0.0)
            println(
              f"  $action baseline=$baselineProbability%.6f provider=$providerProbability%.6f diff=${math.abs(providerProbability - baselineProbability)}%.6f"
            )
          }
      }
      println()
    }

  private def parseOperation(raw: String): Operation =
    raw.toLowerCase(java.util.Locale.ROOT) match
      case "full" => Operation.Full
      case "decision" | "root" => Operation.Decision
      case other => throw new IllegalArgumentException(s"unsupported operation: $other")

  private def parseIterations(raw: String): Vector[Int] =
    raw.split(',').toVector.map(_.trim).filter(_.nonEmpty).map { token =>
      val parsed = token.toInt
      require(parsed > 0, s"iterations must be positive: $token")
      parsed
    }

  private def parseProviders(raw: String): Vector[String] =
    raw.split(',').toVector.map(_.trim.toLowerCase(java.util.Locale.ROOT)).filter(_.nonEmpty)

  private def defaultIterations(scenario: String): Vector[Int] =
    scenario match
      case "preflop" => Vector(100, 200, 400, 800, 1200)
      case "turn" => Vector(100, 200, 300, 400, 500, 700)
      case _ => Vector(100, 200, 400, 800)

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
      spot: Spot,
      config: HoldemCfrConfig
  ): ProbeResult =
    withSystemProperty("sicfun.cfr.provider", provider) {
      operation match
        case Operation.Full =>
          val solved = HoldemCfrSolver.solve(spot.hero, spot.state, spot.posterior, spot.candidateActions, config)
          FullResult(
            provider = solved.provider,
            bestAction = solved.bestAction,
            actionProbabilities = solved.actionProbabilities,
            expectedValuePlayer0 = solved.expectedValuePlayer0
          )
        case Operation.Decision =>
          val solved = HoldemCfrSolver.solveDecisionPolicy(spot.hero, spot.state, spot.posterior, spot.candidateActions, config)
          DecisionResult(
            provider = solved.provider,
            bestAction = solved.bestAction,
            actionProbabilities = solved.actionProbabilities
          )
    }

  private def maxProbabilityDiff(
      actions: Vector[PokerAction],
      baseline: ProbeResult,
      candidate: ProbeResult
  ): Double =
    actions.foldLeft(0.0) { (acc, action) =>
      val diff =
        math.abs(
          candidate.actionProbabilities.getOrElse(action, 0.0) -
            baseline.actionProbabilities.getOrElse(action, 0.0)
        )
      math.max(acc, diff)
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
