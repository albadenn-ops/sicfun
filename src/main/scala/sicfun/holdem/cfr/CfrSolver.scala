package sicfun.holdem.cfr
import sicfun.holdem.*

import scala.collection.mutable

/** Counterfactual regret minimization (CFR) for two-player zero-sum extensive-form games.
  *
  * The solver is game-agnostic and requires a game adapter that exposes:
  *  - terminal utility for player 0
  *  - chance outcomes with probabilities
  *  - legal actions and transitions at player nodes
  *  - information-set keys for player 0 and player 1
  */
object CfrSolver:
  enum Actor:
    case Player0
    case Player1
    case Chance
    case Terminal

  trait ExtensiveFormGame[S, A]:
    def root: S
    def actor(state: S): Actor
    def legalActions(state: S): Vector[A]
    def informationSetKey(state: S, player: Int): String
    def transition(state: S, action: A): S
    def chanceOutcomes(state: S): Vector[(S, Double)]
    def terminalUtilityPlayer0(state: S): Double

  final case class Config(
      iterations: Int = 2_000,
      cfrPlus: Boolean = true,
      averagingDelay: Int = 0,
      linearAveraging: Boolean = true
  ):
    require(iterations > 0, "iterations must be positive")
    require(averagingDelay >= 0, "averagingDelay must be non-negative")

  final case class InfoSetSnapshot[A](
      actions: Vector[A],
      strategy: Vector[Double],
      cumulativeRegret: Vector[Double],
      cumulativeStrategy: Vector[Double]
  ):
    require(actions.nonEmpty, "actions must be non-empty")
    require(strategy.length == actions.length, "strategy/action length mismatch")
    require(cumulativeRegret.length == actions.length, "cumulativeRegret/action length mismatch")
    require(cumulativeStrategy.length == actions.length, "cumulativeStrategy/action length mismatch")

  final case class TrainingResult[A](
      iterations: Int,
      expectedValuePlayer0: Double,
      infosets: Map[String, InfoSetSnapshot[A]]
  ):
    require(iterations > 0, "iterations must be positive")

  final case class RootPolicyResult[A](
      iterations: Int,
      actions: Vector[A],
      strategy: Vector[Double]
  ):
    require(iterations > 0, "iterations must be positive")
    require(actions.nonEmpty, "actions must be non-empty")
    require(strategy.length == actions.length, "strategy/actions length mismatch")

  private final case class InfoSetState[A](
      actions: Vector[A],
      cumulativeRegret: Array[Double],
      cumulativeStrategy: Array[Double]
  ):
    // Pre-allocated scratch arrays reused across CFR iterations to avoid
    // per-visit allocation in the hot loop (~1.65M calls per solve).
    val reusableStrategy: Array[Double] = new Array[Double](actions.length)
    val reusableUtilities: Array[Double] = new Array[Double](actions.length)

    def currentStrategy(cfrPlus: Boolean): Array[Double] =
      val strategy = reusableStrategy
      var positiveSum = 0.0
      var idx = 0
      while idx < actions.length do
        val regret = cumulativeRegret(idx)
        val positive =
          if cfrPlus then math.max(0.0, regret)
          else if regret > 0.0 then regret else 0.0
        strategy(idx) = positive
        positiveSum += positive
        idx += 1
      if positiveSum > 0.0 then
        val inv = 1.0 / positiveSum
        idx = 0
        while idx < strategy.length do
          strategy(idx) = strategy(idx) * inv
          idx += 1
      else
        val uniform = 1.0 / actions.length.toDouble
        idx = 0
        while idx < strategy.length do
          strategy(idx) = uniform
          idx += 1
      strategy

    def averageStrategy(cfrPlus: Boolean): Array[Double] =
      val strategy = new Array[Double](actions.length)
      var total = 0.0
      var idx = 0
      while idx < actions.length do
        val weight = cumulativeStrategy(idx)
        strategy(idx) = weight
        total += weight
        idx += 1
      if total > 0.0 then
        val inv = 1.0 / total
        idx = 0
        while idx < strategy.length do
          strategy(idx) = strategy(idx) * inv
          idx += 1
      else
        val fallback = currentStrategy(cfrPlus)
        idx = 0
        while idx < strategy.length do
          strategy(idx) = fallback(idx)
          idx += 1
      strategy

  def solve[S, A](
      game: ExtensiveFormGame[S, A],
      config: Config = Config()
  ): TrainingResult[A] =
    val infosets = runIterations(game = game, config = config)

    val snapshots = infosets.iterator.map { case (key, state) =>
      val avg = state.averageStrategy(config.cfrPlus).toVector
      val snapshot = InfoSetSnapshot(
        actions = state.actions,
        strategy = avg,
        cumulativeRegret = state.cumulativeRegret.toVector,
        cumulativeStrategy = state.cumulativeStrategy.toVector
      )
      key -> snapshot
    }.toMap

    val expectedValue = expectedUtility(game, game.root, snapshots)
    TrainingResult(
      iterations = config.iterations,
      expectedValuePlayer0 = expectedValue,
      infosets = snapshots
    )

  def solveRootPolicy[S, A](
      game: ExtensiveFormGame[S, A],
      rootInfoSetKey: String,
      rootActions: Vector[A],
      config: Config = Config()
  ): RootPolicyResult[A] =
    require(rootActions.nonEmpty, "rootActions must be non-empty")
    val infosets = runIterations(game = game, config = config)
    val strategy =
      infosets.get(rootInfoSetKey) match
        case Some(state) =>
          require(
            state.actions == rootActions,
            s"inconsistent legal action set for root infoset '$rootInfoSetKey'"
          )
          state.averageStrategy(config.cfrPlus).toVector
        case None =>
          val uniform = 1.0 / rootActions.length.toDouble
          Vector.fill(rootActions.length)(uniform)
    RootPolicyResult(
      iterations = config.iterations,
      actions = rootActions,
      strategy = strategy
    )

  private def runIterations[S, A](
      game: ExtensiveFormGame[S, A],
      config: Config
  ): mutable.HashMap[String, InfoSetState[A]] =
    val infosets = mutable.HashMap.empty[String, InfoSetState[A]]
    var iteration = 1
    while iteration <= config.iterations do
      cfr(
        game = game,
        state = game.root,
        reachPlayer0 = 1.0,
        reachPlayer1 = 1.0,
        iteration = iteration,
        infosets = infosets,
        config = config
      )
      iteration += 1
    infosets

  private def cfr[S, A](
      game: ExtensiveFormGame[S, A],
      state: S,
      reachPlayer0: Double,
      reachPlayer1: Double,
      iteration: Int,
      infosets: mutable.HashMap[String, InfoSetState[A]],
      config: Config
  ): Double =
    game.actor(state) match
      case Actor.Terminal =>
        game.terminalUtilityPlayer0(state)
      case Actor.Chance =>
        val outcomes = normalizeChanceOutcomes(game.chanceOutcomes(state))
        var value = 0.0
        var idx = 0
        while idx < outcomes.length do
          val (nextState, probability) = outcomes(idx)
          value += probability * cfr(
            game = game,
            state = nextState,
            reachPlayer0 = reachPlayer0 * probability,
            reachPlayer1 = reachPlayer1 * probability,
            iteration = iteration,
            infosets = infosets,
            config = config
          )
          idx += 1
        value
      case Actor.Player0 =>
        val key = game.informationSetKey(state, player = 0)
        val actions = game.legalActions(state)
        val info = lookupInfoSet(key, actions, infosets)
        val strategy = info.currentStrategy(config.cfrPlus)
        val utilities = info.reusableUtilities

        var nodeValue = 0.0
        var idx = 0
        while idx < actions.length do
          val action = actions(idx)
          val nextState = game.transition(state, action)
          val actionValue = cfr(
            game = game,
            state = nextState,
            reachPlayer0 = reachPlayer0 * strategy(idx),
            reachPlayer1 = reachPlayer1,
            iteration = iteration,
            infosets = infosets,
            config = config
          )
          utilities(idx) = actionValue
          nodeValue += strategy(idx) * actionValue
          idx += 1

        val averagingWeight = averagingWeightFor(iteration, config)
        idx = 0
        while idx < actions.length do
          val regretDelta = reachPlayer1 * (utilities(idx) - nodeValue)
          val updatedRegret = info.cumulativeRegret(idx) + regretDelta
          info.cumulativeRegret(idx) =
            if config.cfrPlus then math.max(0.0, updatedRegret) else updatedRegret
          info.cumulativeStrategy(idx) += averagingWeight * reachPlayer0 * strategy(idx)
          idx += 1
        nodeValue
      case Actor.Player1 =>
        val key = game.informationSetKey(state, player = 1)
        val actions = game.legalActions(state)
        val info = lookupInfoSet(key, actions, infosets)
        val strategy = info.currentStrategy(config.cfrPlus)
        val utilities = info.reusableUtilities

        var nodeValue = 0.0
        var idx = 0
        while idx < actions.length do
          val action = actions(idx)
          val nextState = game.transition(state, action)
          val actionValue = cfr(
            game = game,
            state = nextState,
            reachPlayer0 = reachPlayer0,
            reachPlayer1 = reachPlayer1 * strategy(idx),
            iteration = iteration,
            infosets = infosets,
            config = config
          )
          utilities(idx) = actionValue
          nodeValue += strategy(idx) * actionValue
          idx += 1

        val averagingWeight = averagingWeightFor(iteration, config)
        idx = 0
        while idx < actions.length do
          val regretDelta = reachPlayer0 * (nodeValue - utilities(idx))
          val updatedRegret = info.cumulativeRegret(idx) + regretDelta
          info.cumulativeRegret(idx) =
            if config.cfrPlus then math.max(0.0, updatedRegret) else updatedRegret
          info.cumulativeStrategy(idx) += averagingWeight * reachPlayer1 * strategy(idx)
          idx += 1
        nodeValue

  private def expectedUtility[S, A](
      game: ExtensiveFormGame[S, A],
      state: S,
      infosets: Map[String, InfoSetSnapshot[A]]
  ): Double =
    game.actor(state) match
      case Actor.Terminal =>
        game.terminalUtilityPlayer0(state)
      case Actor.Chance =>
        val outcomes = normalizeChanceOutcomes(game.chanceOutcomes(state))
        var value = 0.0
        var idx = 0
        while idx < outcomes.length do
          val (nextState, probability) = outcomes(idx)
          value += probability * expectedUtility(game, nextState, infosets)
          idx += 1
        value
      case Actor.Player0 =>
        expectedAtPlayerNode(game, state, infosets, player = 0)
      case Actor.Player1 =>
        expectedAtPlayerNode(game, state, infosets, player = 1)

  private def expectedAtPlayerNode[S, A](
      game: ExtensiveFormGame[S, A],
      state: S,
      infosets: Map[String, InfoSetSnapshot[A]],
      player: Int
  ): Double =
    val actions = game.legalActions(state)
    val key = game.informationSetKey(state, player)
    val strategy = strategyForExpected(actions, infosets.get(key))
    var value = 0.0
    var idx = 0
    while idx < actions.length do
      val nextState = game.transition(state, actions(idx))
      value += strategy(idx) * expectedUtility(game, nextState, infosets)
      idx += 1
    value

  private def strategyForExpected[A](
      legalActions: Vector[A],
      snapshot: Option[InfoSetSnapshot[A]]
  ): Array[Double] =
    snapshot match
      case Some(info) if info.actions == legalActions =>
        info.strategy.toArray
      case _ =>
        val uniform = 1.0 / legalActions.length.toDouble
        Array.fill(legalActions.length)(uniform)

  private def lookupInfoSet[A](
      key: String,
      legalActions: Vector[A],
      infosets: mutable.HashMap[String, InfoSetState[A]]
  ): InfoSetState[A] =
    require(legalActions.nonEmpty, s"legal actions must be non-empty for infoset '$key'")
    infosets.get(key) match
      case Some(state) =>
        require(
          state.actions == legalActions,
          s"inconsistent legal action set for infoset '$key'"
        )
        state
      case None =>
        val state = InfoSetState(
          actions = legalActions,
          cumulativeRegret = new Array[Double](legalActions.length),
          cumulativeStrategy = new Array[Double](legalActions.length)
        )
        infosets.put(key, state)
        state

  private def normalizeChanceOutcomes[S](raw: Vector[(S, Double)]): Vector[(S, Double)] =
    require(raw.nonEmpty, "chance node must provide at least one outcome")
    var total = 0.0
    var idx = 0
    while idx < raw.length do
      val probability = raw(idx)._2
      require(
        probability >= 0.0 && probability.isFinite,
        s"chance probability must be finite and non-negative, got $probability"
      )
      total += probability
      idx += 1
    require(total > 0.0, "chance probabilities must sum to > 0")
    // Skip allocation when outcomes are already normalized (common case for
    // pre-normalized distributions). Over 1500 CFR iterations this avoids
    // creating ~1500 Vector copies of the ~96-element villain distribution.
    if math.abs(total - 1.0) <= 1e-10 then raw
    else
      val inv = 1.0 / total
      raw.map { case (state, probability) => state -> (probability * inv) }

  private def averagingWeightFor(iteration: Int, config: Config): Double =
    if iteration <= config.averagingDelay then 0.0
    else if config.linearAveraging then (iteration - config.averagingDelay).toDouble
    else 1.0
