package sicfun.holdem.bench

import sicfun.holdem.cfr.CfrSolver

/** A/B benchmark: CfrSolver.solve (Double) vs solveFixed (FixedVal/Prob).
  *
  * Usage: sbt "runMain sicfun.holdem.bench.CfrSolverFixedBenchmark [warmup] [runs] [iterations]"
  */
object CfrSolverFixedBenchmark:
  private enum KuhnCard:
    case J
    case Q
    case K

    def rank: Int =
      this match
        case J => 0
        case Q => 1
        case K => 2

  private enum KuhnAction:
    case Check
    case Bet
    case Call
    case Fold

  private final case class KuhnState(
      history: String,
      p0: Option[KuhnCard],
      p1: Option[KuhnCard]
  )

  private object KuhnGame extends CfrSolver.ExtensiveFormGame[KuhnState, KuhnAction]:
    private val cards = Vector(KuhnCard.J, KuhnCard.Q, KuhnCard.K)
    private val dealOutcomes =
      for
        first <- cards
        second <- cards
        if first != second
      yield (first, second)

    override val root: KuhnState =
      KuhnState(history = "", p0 = None, p1 = None)

    override def actor(state: KuhnState): CfrSolver.Actor =
      if state.p0.isEmpty || state.p1.isEmpty then CfrSolver.Actor.Chance
      else if isTerminal(state.history) then CfrSolver.Actor.Terminal
      else
        state.history match
          case ""   => CfrSolver.Actor.Player0
          case "c"  => CfrSolver.Actor.Player1
          case "b"  => CfrSolver.Actor.Player1
          case "cb" => CfrSolver.Actor.Player0
          case _ =>
            throw new IllegalArgumentException(s"invalid non-terminal history '${state.history}'")

    override def legalActions(state: KuhnState): Vector[KuhnAction] =
      if state.p0.isEmpty || state.p1.isEmpty || isTerminal(state.history) then Vector.empty
      else
        state.history match
          case ""   => Vector(KuhnAction.Check, KuhnAction.Bet)
          case "c"  => Vector(KuhnAction.Check, KuhnAction.Bet)
          case "b"  => Vector(KuhnAction.Call, KuhnAction.Fold)
          case "cb" => Vector(KuhnAction.Call, KuhnAction.Fold)
          case _    => throw new IllegalArgumentException(s"unknown history '${state.history}'")

    override def informationSetKey(state: KuhnState, player: Int): String =
      val card =
        if player == 0 then state.p0.getOrElse(throw new IllegalStateException("missing p0 card"))
        else state.p1.getOrElse(throw new IllegalStateException("missing p1 card"))
      s"p$player:$card|${state.history}"

    override def transition(state: KuhnState, action: KuhnAction): KuhnState =
      state.history match
        case "" =>
          action match
            case KuhnAction.Check => state.copy(history = "c")
            case KuhnAction.Bet   => state.copy(history = "b")
            case _                => throw new IllegalArgumentException(s"invalid root action $action")
        case "c" =>
          action match
            case KuhnAction.Check => state.copy(history = "cc")
            case KuhnAction.Bet   => state.copy(history = "cb")
            case _                => throw new IllegalArgumentException(s"invalid action at history 'c': $action")
        case "b" =>
          action match
            case KuhnAction.Call => state.copy(history = "bc")
            case KuhnAction.Fold => state.copy(history = "bf")
            case _               => throw new IllegalArgumentException(s"invalid action at history 'b': $action")
        case "cb" =>
          action match
            case KuhnAction.Call => state.copy(history = "cbc")
            case KuhnAction.Fold => state.copy(history = "cbf")
            case _               => throw new IllegalArgumentException(s"invalid action at history 'cb': $action")
        case _ =>
          throw new IllegalArgumentException(s"cannot transition terminal/unknown history '${state.history}'")

    override def chanceOutcomes(state: KuhnState): Vector[(KuhnState, Double)] =
      if state.p0.nonEmpty && state.p1.nonEmpty then Vector.empty
      else
        val p = 1.0 / dealOutcomes.length.toDouble
        dealOutcomes.map { case (first, second) =>
          KuhnState(history = "", p0 = Some(first), p1 = Some(second)) -> p
        }.toVector

    override def terminalUtilityPlayer0(state: KuhnState): Double =
      val hero = state.p0.getOrElse(throw new IllegalStateException("missing p0 card at terminal"))
      val villain = state.p1.getOrElse(throw new IllegalStateException("missing p1 card at terminal"))
      state.history match
        case "cc" =>
          if hero.rank > villain.rank then 1.0 else -1.0
        case "bc" | "cbc" =>
          if hero.rank > villain.rank then 2.0 else -2.0
        case "bf" =>
          1.0
        case "cbf" =>
          -1.0
        case other =>
          throw new IllegalArgumentException(s"not a terminal history: '$other'")

    private def isTerminal(history: String): Boolean =
      history == "cc" || history == "bc" || history == "bf" || history == "cbc" || history == "cbf"

  def main(args: Array[String]): Unit =
    val warmup = if args.length > 0 then args(0).toInt else 3
    val runs = if args.length > 1 then args(1).toInt else 15
    val iterations = if args.length > 2 then args(2).toInt else 20_000
    val config = CfrSolver.Config(
      iterations = iterations,
      cfrPlus = true,
      averagingDelay = math.max(0, iterations / 8),
      linearAveraging = true
    )

    println("=== CFR Fixed A/B Benchmark ===")
    println(s"Game: Kuhn poker, iterations=$iterations, warmup=$warmup, runs=$runs")
    println()

    var w = 0
    while w < warmup do
      CfrSolver.solve(KuhnGame, config)
      CfrSolver.solveFixed(KuhnGame, config)
      w += 1

    val doubleTimes = new Array[Long](runs)
    val fixedTimes = new Array[Long](runs)
    var r = 0
    while r < runs do
      if r % 2 == 0 then
        doubleTimes(r) = timeOnce { CfrSolver.solve(KuhnGame, config) }
        fixedTimes(r) = timeOnce { CfrSolver.solveFixed(KuhnGame, config) }
      else
        fixedTimes(r) = timeOnce { CfrSolver.solveFixed(KuhnGame, config) }
        doubleTimes(r) = timeOnce { CfrSolver.solve(KuhnGame, config) }
      r += 1

    val baseline = CfrSolver.solve(KuhnGame, config)
    val fixed = CfrSolver.solveFixed(KuhnGame, config)

    val baselineMedian = median(doubleTimes)
    val fixedMedian = median(fixedTimes)
    val speedup = baselineMedian.toDouble / fixedMedian.toDouble

    println("--- Double (baseline) ---")
    printStats("Double", doubleTimes)
    println()
    println("--- Fixed (FixedVal/Prob) ---")
    printStats("Fixed ", fixedTimes)
    println()
    println(f"Speedup (median): $speedup%.3fx")
    println()
    println("--- Correctness ---")
    println(f"EV baseline=${baseline.expectedValuePlayer0}%.6f fixed=${fixed.expectedValuePlayer0}%.6f diff=${math.abs(baseline.expectedValuePlayer0 - fixed.expectedValuePlayer0)}%.6f")
    printActionDiff("p0:J|", baseline, fixed, KuhnAction.Bet)
    printActionDiff("p0:Q|", baseline, fixed, KuhnAction.Bet)
    printActionDiff("p0:K|", baseline, fixed, KuhnAction.Bet)

  private def printActionDiff(
      key: String,
      baseline: CfrSolver.TrainingResult[KuhnAction],
      fixed: CfrSolver.TrainingResult[KuhnAction],
      action: KuhnAction
  ): Unit =
    val baselineProbability = actionProbability(baseline, key, action)
    val fixedProbability = actionProbability(fixed, key, action)
    println(
      f"$key/$action baseline=$baselineProbability%.6f fixed=$fixedProbability%.6f diff=${math.abs(baselineProbability - fixedProbability)}%.6f"
    )

  private def actionProbability(
      result: CfrSolver.TrainingResult[KuhnAction],
      key: String,
      action: KuhnAction
  ): Double =
    val info = result.infosets(key)
    info.actions.zip(info.strategy).toMap.getOrElse(action, throw new IllegalStateException(s"missing action $action at $key"))

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
