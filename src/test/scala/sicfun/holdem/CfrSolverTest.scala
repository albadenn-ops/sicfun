package sicfun.holdem

import munit.FunSuite

class CfrSolverTest extends FunSuite:
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

  test("CFR converges to Kuhn-like mixed strategy profile") {
    val result = CfrSolver.solve(
      game = KuhnGame,
      config = CfrSolver.Config(
        iterations = 40_000,
        cfrPlus = true,
        averagingDelay = 5_000,
        linearAveraging = true
      )
    )

    val p0JBet = actionProbability(result, key = "p0:J|", KuhnAction.Bet)
    val p0QBet = actionProbability(result, key = "p0:Q|", KuhnAction.Bet)
    val p0KBet = actionProbability(result, key = "p0:K|", KuhnAction.Bet)

    assert(p0JBet > 0.15 && p0JBet < 0.55, s"expected J bet mix near bluff frequency, got $p0JBet")
    assert(p0QBet < 0.20, s"expected Q mostly check, got $p0QBet")
    assert(p0KBet > 0.65, s"expected K to bet at high frequency, got $p0KBet")

    assert(
      result.expectedValuePlayer0 > -0.15 && result.expectedValuePlayer0 < 0.02,
      s"unexpected Kuhn game value for player0: ${result.expectedValuePlayer0}"
    )
  }

  private def actionProbability(
      result: CfrSolver.TrainingResult[KuhnAction],
      key: String,
      action: KuhnAction
  ): Double =
    val info = result.infosets.getOrElse(key, fail(s"missing infoset '$key'"))
    info.actions.zip(info.strategy).toMap.getOrElse(action, fail(s"missing action '$action' for infoset '$key'"))
