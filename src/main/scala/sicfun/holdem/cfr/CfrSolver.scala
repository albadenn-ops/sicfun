package sicfun.holdem.cfr
import sicfun.holdem.*
import sicfun.core.{FixedVal, Prob}
import sicfun.core.FixedVal.*
import sicfun.core.Prob.*

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
  private val RegretRescaleThresholdRaw = Int.MaxValue / 4
  private val StrategyRescaleThresholdRaw = Long.MaxValue / 4

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

  private final case class InfoSetStateFixed[A](
      actions: Vector[A],
      cumulativeRegret: Array[Int],
      cumulativeStrategy: Array[Long]
  ):
    // Fixed-point scratch reused across iterations. Strategy stays in Prob raw units,
    // utilities/regrets stay in FixedVal raw units.
    val reusableStrategy: Array[Int] = new Array[Int](actions.length)
    val reusableUtilities: Array[Int] = new Array[Int](actions.length)
    val reusableRegretDeltas: Array[Int] = new Array[Int](actions.length)
    val reusableStrategyDeltas: Array[Long] = new Array[Long](actions.length)

    def currentStrategy(cfrPlus: Boolean): Array[Int] =
      val strategy = reusableStrategy
      var positiveSum = 0L
      var lastPositive = -1
      var idx = 0
      while idx < actions.length do
        val regret = cumulativeRegret(idx)
        val positive =
          if cfrPlus then math.max(0, regret)
          else if regret > 0 then regret else 0
        strategy(idx) = positive
        positiveSum += positive.toLong
        if positive > 0 then lastPositive = idx
        idx += 1
      if positiveSum > 0L then
        var assigned = 0
        idx = 0
        while idx < strategy.length do
          val probRaw =
            if strategy(idx) == 0 then 0
            else if idx == lastPositive then Prob.Scale - assigned
            else ((strategy(idx).toLong * Prob.Scale.toLong) / positiveSum).toInt
          strategy(idx) = probRaw
          assigned += probRaw
          idx += 1
      else
        writeUniformProbabilities(strategy)
      strategy

    def averageStrategy(cfrPlus: Boolean): Array[Double] =
      val strategy = new Array[Double](actions.length)
      var total = 0L
      var idx = 0
      while idx < actions.length do
        val weight = cumulativeStrategy(idx)
        strategy(idx) = weight.toDouble / Prob.Scale.toDouble
        total += weight
        idx += 1
      if total > 0L then
        val inv = 1.0 / total.toDouble
        idx = 0
        while idx < strategy.length do
          strategy(idx) = cumulativeStrategy(idx).toDouble * inv
          idx += 1
      else
        val fallback = currentStrategy(cfrPlus)
        idx = 0
        while idx < strategy.length do
          strategy(idx) = fallback(idx).toDouble / Prob.Scale.toDouble
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

  /** Experimental fixed-point CFR core.
    *
    * Uses FixedVal/Prob in the training loop and converts back to Double only for
    * the public snapshot API, enabling A/B comparison against the existing solver.
    */
  def solveFixed[S, A](
      game: ExtensiveFormGame[S, A],
      config: Config = Config()
  ): TrainingResult[A] =
    val infosets = runIterationsFixed(game = game, config = config)

    val snapshots = infosets.iterator.map { case (key, state) =>
      val avg = state.averageStrategy(config.cfrPlus).toVector
      val cumulativeRegret = state.cumulativeRegret.iterator.map(raw => FixedVal(raw).toDouble).toVector
      val cumulativeStrategy =
        state.cumulativeStrategy.iterator.map(raw => raw.toDouble / Prob.Scale.toDouble).toVector
      val snapshot = InfoSetSnapshot(
        actions = state.actions,
        strategy = avg,
        cumulativeRegret = cumulativeRegret,
        cumulativeStrategy = cumulativeStrategy
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

  def solveRootPolicyFixed[S, A](
      game: ExtensiveFormGame[S, A],
      rootInfoSetKey: String,
      rootActions: Vector[A],
      config: Config = Config()
  ): RootPolicyResult[A] =
    require(rootActions.nonEmpty, "rootActions must be non-empty")
    val infosets = runIterationsFixed(game = game, config = config)
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

  private def runIterationsFixed[S, A](
      game: ExtensiveFormGame[S, A],
      config: Config
  ): mutable.HashMap[String, InfoSetStateFixed[A]] =
    val infosets = mutable.HashMap.empty[String, InfoSetStateFixed[A]]
    var iteration = 1
    while iteration <= config.iterations do
      cfrFixed(
        game = game,
        state = game.root,
        reachPlayer0 = Prob.One,
        reachPlayer1 = Prob.One,
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

  private def cfrFixed[S, A](
      game: ExtensiveFormGame[S, A],
      state: S,
      reachPlayer0: Prob,
      reachPlayer1: Prob,
      iteration: Int,
      infosets: mutable.HashMap[String, InfoSetStateFixed[A]],
      config: Config
  ): FixedVal =
    game.actor(state) match
      case Actor.Terminal =>
        FixedVal.fromDouble(game.terminalUtilityPlayer0(state))
      case Actor.Chance =>
        val outcomes = normalizeChanceOutcomesProb(game.chanceOutcomes(state))
        var valueRaw = 0
        var idx = 0
        while idx < outcomes.length do
          val (nextState, probability) = outcomes(idx)
          val childValue = cfrFixed(
            game = game,
            state = nextState,
            reachPlayer0 = Prob(multiplyProbRaw(reachPlayer0.raw, probability.raw)),
            reachPlayer1 = Prob(multiplyProbRaw(reachPlayer1.raw, probability.raw)),
            iteration = iteration,
            infosets = infosets,
            config = config
          )
          valueRaw = checkedFixedRaw(valueRaw.toLong + multiplyFixedByProbRaw(childValue.raw, probability.raw).toLong)
          idx += 1
        FixedVal(valueRaw)
      case Actor.Player0 =>
        val key = game.informationSetKey(state, player = 0)
        val actions = game.legalActions(state)
        val info = lookupInfoSetFixed(key, actions, infosets)
        val strategy = info.currentStrategy(config.cfrPlus)
        val utilities = info.reusableUtilities

        var nodeValueRaw = 0
        var idx = 0
        while idx < actions.length do
          val action = actions(idx)
          val nextState = game.transition(state, action)
          val actionValue = cfrFixed(
            game = game,
            state = nextState,
            reachPlayer0 = Prob(multiplyProbRaw(reachPlayer0.raw, strategy(idx))),
            reachPlayer1 = reachPlayer1,
            iteration = iteration,
            infosets = infosets,
            config = config
          )
          utilities(idx) = actionValue.raw
          nodeValueRaw = checkedFixedRaw(nodeValueRaw.toLong + multiplyFixedByProbRaw(actionValue.raw, strategy(idx)).toLong)
          idx += 1

        updateFixedInfoSet(
          info = info,
          strategy = strategy,
          utilities = utilities,
          nodeValueRaw = nodeValueRaw,
          regretReachRaw = reachPlayer1.raw,
          selfReachRaw = reachPlayer0.raw,
          averagingWeight = averagingWeightMultiplier(iteration, config),
          cfrPlus = config.cfrPlus,
          invertRegretSign = false
        )
        FixedVal(nodeValueRaw)
      case Actor.Player1 =>
        val key = game.informationSetKey(state, player = 1)
        val actions = game.legalActions(state)
        val info = lookupInfoSetFixed(key, actions, infosets)
        val strategy = info.currentStrategy(config.cfrPlus)
        val utilities = info.reusableUtilities

        var nodeValueRaw = 0
        var idx = 0
        while idx < actions.length do
          val action = actions(idx)
          val nextState = game.transition(state, action)
          val actionValue = cfrFixed(
            game = game,
            state = nextState,
            reachPlayer0 = reachPlayer0,
            reachPlayer1 = Prob(multiplyProbRaw(reachPlayer1.raw, strategy(idx))),
            iteration = iteration,
            infosets = infosets,
            config = config
          )
          utilities(idx) = actionValue.raw
          nodeValueRaw = checkedFixedRaw(nodeValueRaw.toLong + multiplyFixedByProbRaw(actionValue.raw, strategy(idx)).toLong)
          idx += 1

        updateFixedInfoSet(
          info = info,
          strategy = strategy,
          utilities = utilities,
          nodeValueRaw = nodeValueRaw,
          regretReachRaw = reachPlayer0.raw,
          selfReachRaw = reachPlayer1.raw,
          averagingWeight = averagingWeightMultiplier(iteration, config),
          cfrPlus = config.cfrPlus,
          invertRegretSign = true
        )
        FixedVal(nodeValueRaw)

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

  private def lookupInfoSetFixed[A](
      key: String,
      legalActions: Vector[A],
      infosets: mutable.HashMap[String, InfoSetStateFixed[A]]
  ): InfoSetStateFixed[A] =
    require(legalActions.nonEmpty, s"legal actions must be non-empty for infoset '$key'")
    infosets.get(key) match
      case Some(state) =>
        require(
          state.actions == legalActions,
          s"inconsistent legal action set for infoset '$key'"
        )
        state
      case None =>
        val state = InfoSetStateFixed(
          actions = legalActions,
          cumulativeRegret = new Array[Int](legalActions.length),
          cumulativeStrategy = new Array[Long](legalActions.length)
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

  private def normalizeChanceOutcomesProb[S](raw: Vector[(S, Double)]): Vector[(S, Prob)] =
    val normalized = normalizeChanceOutcomes(raw)
    val probabilities = new Array[Double](normalized.length)
    var idx = 0
    while idx < normalized.length do
      probabilities(idx) = normalized(idx)._2
      idx += 1
    val probabilityRaws = normalizeProbabilitiesToRaw(probabilities)
    normalized.indices.map(idx => normalized(idx)._1 -> Prob(probabilityRaws(idx))).toVector

  private def normalizeProbabilitiesToRaw(probabilities: Array[Double]): Array[Int] =
    val raws = new Array[Int](probabilities.length)
    var sum = 0
    var maxIdx = 0
    var maxRaw = Int.MinValue
    var idx = 0
    while idx < probabilities.length do
      val raw = Prob.fromDouble(probabilities(idx)).raw
      raws(idx) = raw
      sum += raw
      if raw > maxRaw then
        maxRaw = raw
        maxIdx = idx
      idx += 1
    if raws.nonEmpty then raws(maxIdx) += (Prob.Scale - sum)
    raws

  private def writeUniformProbabilities(target: Array[Int]): Unit =
    val base = Prob.Scale / target.length
    val remainder = Prob.Scale - (base * target.length)
    var idx = 0
    while idx < target.length do
      target(idx) = base + (if idx < remainder then 1 else 0)
      idx += 1

  private inline def multiplyProbRaw(left: Int, right: Int): Int =
    ((left.toLong * right.toLong) >> 30).toInt

  private def updateFixedInfoSet[A](
      info: InfoSetStateFixed[A],
      strategy: Array[Int],
      utilities: Array[Int],
      nodeValueRaw: Int,
      regretReachRaw: Int,
      selfReachRaw: Int,
      averagingWeight: Int,
      cfrPlus: Boolean,
      invertRegretSign: Boolean
  ): Unit =
    val regretDeltas = info.reusableRegretDeltas
    val strategyDeltas = info.reusableStrategyDeltas
    var idx = 0
    while idx < info.actions.length do
      val advantageRaw =
        if invertRegretSign then nodeValueRaw - utilities(idx)
        else utilities(idx) - nodeValueRaw
      regretDeltas(idx) = multiplyFixedByProbRaw(advantageRaw, regretReachRaw)
      val reachStrategyRaw = multiplyProbRaw(selfReachRaw, strategy(idx))
      strategyDeltas(idx) = averagingWeight.toLong * reachStrategyRaw.toLong
      idx += 1
    applyFixedActionUpdates(
      cumulativeRegret = info.cumulativeRegret,
      cumulativeStrategy = info.cumulativeStrategy,
      regretDeltas = regretDeltas,
      strategyDeltas = strategyDeltas,
      cfrPlus = cfrPlus
    )

  private[cfr] def applyFixedActionUpdates(
      cumulativeRegret: Array[Int],
      cumulativeStrategy: Array[Long],
      regretDeltas: Array[Int],
      strategyDeltas: Array[Long],
      cfrPlus: Boolean
  ): Unit =
    require(cumulativeRegret.length == cumulativeStrategy.length, "fixed update length mismatch")
    require(cumulativeRegret.length == regretDeltas.length, "fixed regret delta length mismatch")
    require(cumulativeStrategy.length == strategyDeltas.length, "fixed strategy delta length mismatch")

    while needsRegretEmergencyRescale(cumulativeRegret, regretDeltas) do
      halveIntArray(cumulativeRegret)
    while needsStrategyEmergencyRescale(cumulativeStrategy, strategyDeltas) do
      halveLongArray(cumulativeStrategy)

    var rescaleRegret = false
    var idx = 0
    while idx < cumulativeRegret.length do
      var updatedRegret = cumulativeRegret(idx).toLong + regretDeltas(idx).toLong
      if cfrPlus && updatedRegret < 0L then updatedRegret = 0L
      cumulativeRegret(idx) = checkedFixedRaw(updatedRegret)
      if math.abs(cumulativeRegret(idx).toLong) >= RegretRescaleThresholdRaw.toLong then rescaleRegret = true
      idx += 1
    if rescaleRegret then halveIntArray(cumulativeRegret)

    var rescaleStrategy = false
    idx = 0
    while idx < cumulativeStrategy.length do
      val updatedStrategyMass =
        checkedAddLong(cumulativeStrategy(idx), strategyDeltas(idx), "fixed-point CFR strategy mass")
      cumulativeStrategy(idx) = updatedStrategyMass
      if updatedStrategyMass >= StrategyRescaleThresholdRaw then rescaleStrategy = true
      idx += 1
    if rescaleStrategy then halveLongArray(cumulativeStrategy)

  private def needsRegretEmergencyRescale(
      cumulativeRegret: Array[Int],
      regretDeltas: Array[Int]
  ): Boolean =
    var idx = 0
    while idx < cumulativeRegret.length do
      val updated = cumulativeRegret(idx).toLong + regretDeltas(idx).toLong
      if updated < Int.MinValue.toLong || updated > Int.MaxValue.toLong then return true
      idx += 1
    false

  private def needsStrategyEmergencyRescale(
      cumulativeStrategy: Array[Long],
      strategyDeltas: Array[Long]
  ): Boolean =
    var idx = 0
    while idx < cumulativeStrategy.length do
      val delta = strategyDeltas(idx)
      require(delta >= 0L, s"fixed-point CFR strategy delta must be non-negative, got $delta")
      if cumulativeStrategy(idx) > Long.MaxValue - delta then return true
      idx += 1
    false

  private inline def multiplyFixedByProbRaw(valueRaw: Int, probabilityRaw: Int): Int =
    roundShift30Signed(valueRaw.toLong * probabilityRaw.toLong)

  // Fixed utilities/regrets can be negative, so we round on absolute magnitude
  // before restoring the sign. Arithmetic right-shift alone biases small
  // negative values downward.
  private inline def roundShift30Signed(product: Long): Int =
    val absProduct = if product >= 0L then product else -product
    val rounded = ((absProduct + (1L << 29)) >> 30).toInt
    if product >= 0L then rounded else -rounded

  private def checkedFixedRaw(raw: Long): Int =
    require(
      raw >= Int.MinValue.toLong && raw <= Int.MaxValue.toLong,
      s"fixed-point CFR value overflow: $raw"
    )
    raw.toInt

  private def checkedAddLong(left: Long, right: Long, label: String): Long =
    val sum = left + right
    val sameSign = (left ^ right) >= 0L
    val changedSign = (left ^ sum) < 0L
    if sameSign && changedSign then
      throw new IllegalStateException(s"$label overflow: $left + $right")
    sum

  private def halveIntArray(values: Array[Int]): Unit =
    var idx = 0
    while idx < values.length do
      values(idx) = values(idx) / 2
      idx += 1

  private def halveLongArray(values: Array[Long]): Unit =
    var idx = 0
    while idx < values.length do
      values(idx) = values(idx) / 2L
      idx += 1

  private def averagingWeightFor(iteration: Int, config: Config): Double =
    if iteration <= config.averagingDelay then 0.0
    else if config.linearAveraging then (iteration - config.averagingDelay).toDouble
    else 1.0

  private def averagingWeightMultiplier(iteration: Int, config: Config): Int =
    if iteration <= config.averagingDelay then 0
    else if config.linearAveraging then iteration - config.averagingDelay
    else 1
