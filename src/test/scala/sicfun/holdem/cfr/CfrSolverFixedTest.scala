package sicfun.holdem.cfr
import sicfun.holdem.*
import sicfun.core.FixedVal

import munit.FunSuite

class CfrSolverFixedTest extends FunSuite:
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

  private enum TinyAction:
    case Safe
    case Gamble

  private enum TinyState:
    case Root
    case GambleChance
    case SafeTerminal
    case WinTerminal
    case LoseTerminal

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

  private object TinySymmetricGame extends CfrSolver.ExtensiveFormGame[TinyState, TinyAction]:
    private val epsilon = 1.0 / FixedVal.Scale.toDouble

    override val root: TinyState = TinyState.Root

    override def actor(state: TinyState): CfrSolver.Actor =
      state match
        case TinyState.Root         => CfrSolver.Actor.Player0
        case TinyState.GambleChance => CfrSolver.Actor.Chance
        case TinyState.SafeTerminal | TinyState.WinTerminal | TinyState.LoseTerminal =>
          CfrSolver.Actor.Terminal

    override def legalActions(state: TinyState): Vector[TinyAction] =
      state match
        case TinyState.Root => Vector(TinyAction.Safe, TinyAction.Gamble)
        case _              => Vector.empty

    override def informationSetKey(state: TinyState, player: Int): String =
      if state != TinyState.Root || player != 0 then
        throw new IllegalArgumentException(s"unexpected infoset request: state=$state player=$player")
      "tiny-root"

    override def transition(state: TinyState, action: TinyAction): TinyState =
      (state, action) match
        case (TinyState.Root, TinyAction.Safe)   => TinyState.SafeTerminal
        case (TinyState.Root, TinyAction.Gamble) => TinyState.GambleChance
        case _ =>
          throw new IllegalArgumentException(s"invalid transition: state=$state action=$action")

    override def chanceOutcomes(state: TinyState): Vector[(TinyState, Double)] =
      state match
        case TinyState.GambleChance =>
          Vector(
            TinyState.WinTerminal -> 0.5,
            TinyState.LoseTerminal -> 0.5
          )
        case _ => Vector.empty

    override def terminalUtilityPlayer0(state: TinyState): Double =
      state match
        case TinyState.SafeTerminal => 0.0
        case TinyState.WinTerminal  => epsilon
        case TinyState.LoseTerminal => -epsilon
        case _ =>
          throw new IllegalArgumentException(s"not a terminal state: $state")

  private val config = CfrSolver.Config(
    iterations = 12_000,
    cfrPlus = true,
    averagingDelay = 1_500,
    linearAveraging = true
  )

  test("solveFixed stays close to double CFR on Kuhn") {
    val baseline = CfrSolver.solve(KuhnGame, config)
    val fixed = CfrSolver.solveFixed(KuhnGame, config)

    assertEqualsDouble(fixed.expectedValuePlayer0, baseline.expectedValuePlayer0, 0.03)
    assertProbabilityClose(baseline, fixed, "p0:J|", KuhnAction.Bet)
    assertProbabilityClose(baseline, fixed, "p0:Q|", KuhnAction.Bet)
    assertProbabilityClose(baseline, fixed, "p0:K|", KuhnAction.Bet)
    assertProbabilityClose(baseline, fixed, "p1:J|c", KuhnAction.Bet)
  }

  test("solveRootPolicyFixed matches double root-policy extraction") {
    val baseline = CfrSolver.solveRootPolicy(
      game = KuhnGame,
      rootInfoSetKey = "p0:K|",
      rootActions = Vector(KuhnAction.Check, KuhnAction.Bet),
      config = config
    )
    val fixed = CfrSolver.solveRootPolicyFixed(
      game = KuhnGame,
      rootInfoSetKey = "p0:K|",
      rootActions = Vector(KuhnAction.Check, KuhnAction.Bet),
      config = config
    )

    baseline.strategy.zip(fixed.strategy).foreach { case (expected, actual) =>
      assertEqualsDouble(actual, expected, 0.08)
    }
  }

  test("solveFixed preserves a symmetric tiny-EV gamble") {
    val tinyConfig = CfrSolver.Config(
      iterations = 64,
      cfrPlus = true,
      averagingDelay = 0,
      linearAveraging = false
    )

    val baseline = CfrSolver.solveRootPolicy(
      game = TinySymmetricGame,
      rootInfoSetKey = "tiny-root",
      rootActions = Vector(TinyAction.Safe, TinyAction.Gamble),
      config = tinyConfig
    )
    val fixed = CfrSolver.solveRootPolicyFixed(
      game = TinySymmetricGame,
      rootInfoSetKey = "tiny-root",
      rootActions = Vector(TinyAction.Safe, TinyAction.Gamble),
      config = tinyConfig
    )

    baseline.strategy.zip(fixed.strategy).foreach { case (expected, actual) =>
      assertEqualsDouble(actual, expected, 1e-9)
    }
    assertEqualsDouble(fixed.strategy(0), 0.5, 1e-9)
    assertEqualsDouble(fixed.strategy(1), 0.5, 1e-9)
  }

  test("applyFixedActionUpdates rescales regrets atomically before overflowing action") {
    val cumulativeRegret = Array(0, Int.MaxValue - 5, 0)
    val cumulativeStrategy = Array(0L, 0L, 0L)
    val regretDeltas = Array(10, 10, 10)
    val strategyDeltas = Array(0L, 0L, 0L)

    CfrSolver.applyFixedActionUpdates(
      cumulativeRegret = cumulativeRegret,
      cumulativeStrategy = cumulativeStrategy,
      regretDeltas = regretDeltas,
      strategyDeltas = strategyDeltas,
      cfrPlus = false
    )

    assertEquals(cumulativeRegret.toSeq, Seq(5, 536870915, 5))
  }

  test("applyFixedActionUpdates rescales strategy mass atomically before overflowing action") {
    val cumulativeRegret = Array(0, 0, 0)
    val cumulativeStrategy = Array(0L, Long.MaxValue - 5L, 0L)
    val regretDeltas = Array(0, 0, 0)
    val strategyDeltas = Array(10L, 10L, 10L)

    CfrSolver.applyFixedActionUpdates(
      cumulativeRegret = cumulativeRegret,
      cumulativeStrategy = cumulativeStrategy,
      regretDeltas = regretDeltas,
      strategyDeltas = strategyDeltas,
      cfrPlus = false
    )

    assertEquals(cumulativeStrategy.toSeq, Seq(5L, 2305843009213693955L, 5L))
  }

  private def assertProbabilityClose(
      baseline: CfrSolver.TrainingResult[KuhnAction],
      fixed: CfrSolver.TrainingResult[KuhnAction],
      key: String,
      action: KuhnAction
  ): Unit =
    val expected = actionProbability(baseline, key, action)
    val actual = actionProbability(fixed, key, action)
    assertEqualsDouble(actual, expected, 0.08, s"probability mismatch at $key/$action")

  private def actionProbability(
      result: CfrSolver.TrainingResult[KuhnAction],
      key: String,
      action: KuhnAction
  ): Double =
    val info = result.infosets.getOrElse(key, fail(s"missing infoset '$key'"))
    info.actions.zip(info.strategy).toMap.getOrElse(action, fail(s"missing action '$action' for infoset '$key'"))
