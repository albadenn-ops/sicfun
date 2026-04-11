package sicfun.holdem.cfr
import sicfun.holdem.*
import sicfun.core.{FixedVal, Prob}
import sicfun.core.FixedVal.*
import sicfun.core.Prob.*

import scala.collection.mutable

/** Counterfactual regret minimization (CFR) for two-player zero-sum extensive-form games.
  *
  * This is the core game-theory solver used by the SICFUN poker analytics system to
  * compute Nash equilibrium (or epsilon-Nash) strategies. CFR works by repeatedly
  * traversing a game tree, accumulating "regret" for not having chosen each action,
  * and then using regret-matching to derive a mixed strategy that converges to
  * equilibrium over many iterations.
  *
  * The solver is game-agnostic: it operates on any two-player zero-sum extensive-form
  * game via the [[ExtensiveFormGame]] trait. The Hold'em-specific game abstraction
  * ([[HoldemDecisionGame]]) plugs in here to solve poker decision spots.
  *
  * Two arithmetic backends are provided:
  *  - '''Double-precision''' (`solve`, `solveRootPolicy`): standard IEEE 754 doubles.
  *  - '''Fixed-point''' (`solveFixed`, `solveRootPolicyFixed`): uses `FixedVal` (Q1.30)
  *    for utilities/regrets and `Prob` (unsigned Q0.30) for probabilities. This enables
  *    A/B comparison with native C/CUDA implementations that use identical fixed-point
  *    arithmetic to avoid floating-point non-determinism across platforms.
  *
  * Design decisions:
  *  - Hot loops use `while` with index variables instead of functional iterators for
  *    zero-allocation traversal (~1.65M node visits per solve).
  *  - Scratch arrays (`reusableStrategy`, `reusableUtilities`) are pre-allocated per
  *    information set and reused across iterations to avoid GC pressure.
  *  - Fixed-point accumulators use emergency rescaling (halving) to prevent overflow
  *    without losing relative proportions.
  *
  * Key CFR concepts:
  *  - '''Information set''': a set of game states that are indistinguishable to the
  *    acting player (same private cards, same public history).
  *  - '''Regret''': the difference between the value of an action and the node's
  *    expected value under the current strategy.
  *  - '''Regret matching''': choosing actions proportionally to their positive
  *    cumulative regret.
  *  - '''CFR+''': variant that floors negative cumulative regrets to zero each
  *    iteration, improving convergence speed.
  *  - '''Linear averaging''': weights later iterations more heavily in the average
  *    strategy, improving convergence quality.
  *  - '''Averaging delay''': skips early iterations when accumulating the average
  *    strategy, letting regrets stabilize before contributing to the output policy.
  *
  * @see [[HoldemCfrSolver]] for the Hold'em-specific solver that delegates to this core.
  */
object CfrSolver:
  // Threshold at which fixed-point cumulative regret arrays are halved to prevent
  // overflow. Set to Int.MaxValue/4 to leave headroom for one more delta addition.
  private val RegretRescaleThresholdRaw = Int.MaxValue / 4
  // Threshold at which fixed-point cumulative strategy arrays are halved to prevent
  // Long overflow. Set to Long.MaxValue/4 for the same headroom reason.
  private val StrategyRescaleThresholdRaw = Long.MaxValue / 4

  /** Identifies the acting entity at a given game state node. */
  enum Actor:
    /** Hero (player 0) acts — the player whose strategy we are optimizing. */
    case Player0
    /** Villain (player 1) acts — the opponent. */
    case Player1
    /** Nature/chance acts — deals cards or resolves random outcomes. */
    case Chance
    /** Leaf node — game has ended, utility is determined. */
    case Terminal

  /** Abstraction over any two-player zero-sum extensive-form game.
    *
    * Implementations must provide the game tree structure: who acts at each state,
    * what actions are available, how states transition, what chance outcomes exist,
    * and what the terminal payoffs are. The CFR algorithm is completely generic
    * over this trait.
    *
    * @tparam S the game state type (e.g., a poker hand history snapshot)
    * @tparam A the action type (e.g., fold/call/raise)
    */
  trait ExtensiveFormGame[S, A]:
    def root: S
    def actor(state: S): Actor
    def legalActions(state: S): Vector[A]
    def informationSetKey(state: S, player: Int): String
    def transition(state: S, action: A): S
    def chanceOutcomes(state: S): Vector[(S, Double)]
    def terminalUtilityPlayer0(state: S): Double

  /** Configuration parameters for a CFR training run.
    *
    * @param iterations       total number of full game-tree traversals
    * @param cfrPlus          if true, floor negative cumulative regrets to zero (CFR+ variant)
    * @param averagingDelay   skip the first N iterations when accumulating the average strategy,
    *                         letting regrets stabilize before contributing to the output policy
    * @param linearAveraging  if true, weight iteration t's contribution to the average strategy
    *                         by (t - averagingDelay), giving later iterations more influence
    */
  final case class Config(
      iterations: Int = 2_000,
      cfrPlus: Boolean = true,
      averagingDelay: Int = 0,
      linearAveraging: Boolean = true
  ):
    require(iterations > 0, "iterations must be positive")
    require(averagingDelay >= 0, "averagingDelay must be non-negative")

  /** Immutable snapshot of an information set's trained state, suitable for export.
    *
    * @param actions            the ordered legal actions at this information set
    * @param strategy           the average (output) strategy — probability of each action
    * @param cumulativeRegret   total accumulated regret for each action across all iterations
    * @param cumulativeStrategy total accumulated strategy weight for each action
    */
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

  /** Complete result of a CFR training run.
    *
    * @param iterations            number of game-tree traversals completed
    * @param expectedValuePlayer0  expected value of the game for player 0 under the computed strategies
    * @param infosets              map from information-set key to trained snapshot for every visited infoset
    */
  final case class TrainingResult[A](
      iterations: Int,
      expectedValuePlayer0: Double,
      infosets: Map[String, InfoSetSnapshot[A]]
  ):
    require(iterations > 0, "iterations must be positive")

  /** Lightweight result that only extracts the root information set's average strategy.
    *
    * Used by decision-time solvers that only need the hero's mixed policy at the
    * decision point, not the full infoset map or exploitability diagnostics.
    *
    * @param iterations number of game-tree traversals completed
    * @param actions    the ordered actions at the root information set
    * @param strategy   probability of each action under the average strategy
    */
  final case class RootPolicyResult[A](
      iterations: Int,
      actions: Vector[A],
      strategy: Vector[Double]
  ):
    require(iterations > 0, "iterations must be positive")
    require(actions.nonEmpty, "actions must be non-empty")
    require(strategy.length == actions.length, "strategy/actions length mismatch")

  /** Mutable per-information-set state used during double-precision CFR training.
    *
    * This is the workhorse data structure: one instance per information set, mutated
    * in place each iteration. The scratch arrays avoid per-visit allocation in the
    * hot loop (~1.65M calls per solve).
    */
  private final case class InfoSetState[A](
      actions: Vector[A],
      cumulativeRegret: Array[Double],
      cumulativeStrategy: Array[Double]
  ):
    // Pre-allocated scratch arrays reused across CFR iterations to avoid
    // per-visit allocation in the hot loop (~1.65M calls per solve).
    val reusableStrategy: Array[Double] = new Array[Double](actions.length)
    val reusableUtilities: Array[Double] = new Array[Double](actions.length)

    /** Computes the current iteration's strategy via regret matching.
      *
      * The strategy is proportional to each action's positive cumulative regret.
      * If all regrets are non-positive, falls back to a uniform distribution.
      * This implements the core regret-matching formula:
      *   sigma(a) = max(0, R(a)) / sum_b(max(0, R(b)))
      *
      * @param cfrPlus if true, negative regrets were already floored to zero
      * @return the current strategy array (reused scratch — do NOT retain across calls)
      */
    def currentStrategy(cfrPlus: Boolean): Array[Double] =
      val strategy = reusableStrategy
      var positiveSum = 0.0
      var idx = 0
      while idx < actions.length do
        val regret = cumulativeRegret(idx)
        // In CFR+, cumulative regrets are already non-negative. In vanilla CFR,
        // we still only use positive regrets for the strategy computation.
        val positive =
          if cfrPlus then math.max(0.0, regret)
          else if regret > 0.0 then regret else 0.0
        strategy(idx) = positive
        positiveSum += positive
        idx += 1
      if positiveSum > 0.0 then
        // Normalize to a probability distribution
        val inv = 1.0 / positiveSum
        idx = 0
        while idx < strategy.length do
          strategy(idx) = strategy(idx) * inv
          idx += 1
      else
        // No positive regret — play uniformly (no information to prefer any action)
        val uniform = 1.0 / actions.length.toDouble
        idx = 0
        while idx < strategy.length do
          strategy(idx) = uniform
          idx += 1
      strategy

    /** Computes the time-averaged strategy, which is the CFR output policy.
      *
      * The average strategy converges to a Nash equilibrium as iterations increase.
      * It is the weighted average of all per-iteration strategies, where the weight
      * depends on the averaging configuration (linear or uniform).
      *
      * @param cfrPlus if true, uses CFR+ regret flooring when computing fallback
      * @return a new array containing the average strategy probabilities
      */
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

  /** Mutable per-information-set state for the fixed-point CFR variant.
    *
    * Uses Int (Q1.30 FixedVal) for regrets and Long for strategy mass accumulation.
    * This mirrors the arithmetic used in native C/CUDA solvers for bit-exact parity.
    * Emergency rescaling (halving all values) prevents overflow while preserving
    * relative proportions.
    */
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

  /** Runs full CFR training using double-precision arithmetic and returns all infoset snapshots.
    *
    * This is the primary entry point for complete solves where exploitability diagnostics
    * and per-action EV reports are needed. For decision-time usage where only the root
    * policy matters, prefer [[solveRootPolicy]] which skips the expected-utility tree walk.
    *
    * @param game   the extensive-form game to solve
    * @param config CFR training parameters (iterations, CFR+ mode, averaging, etc.)
    * @return training result with all infoset strategies and the game's expected value
    */
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

  /** Runs CFR training and extracts only the root information set's average strategy.
    *
    * More efficient than [[solve]] for decision-time use: it skips the post-training
    * expected-utility tree walk and does not build the full infoset snapshot map.
    *
    * @param game            the extensive-form game to solve
    * @param rootInfoSetKey  the key identifying the root player's information set
    * @param rootActions     the legal actions at the root decision point
    * @param config          CFR training parameters
    * @return the root player's mixed strategy after training
    */
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

  /** Fixed-point variant of [[solveRootPolicy]].
    *
    * Uses FixedVal/Prob arithmetic internally and converts to Double for the output.
    * Provides bit-exact parity with native C/CUDA implementations.
    */
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

  /** Executes the main CFR training loop for the specified number of iterations.
    *
    * Each iteration performs one complete traversal of the game tree from the root,
    * updating cumulative regrets and cumulative strategy weights at every information set.
    *
    * @return the mutable map of all visited information sets with their trained state
    */
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

  /** Fixed-point variant of [[runIterations]]. Uses Prob/FixedVal arithmetic. */
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

  /** Recursive double-precision CFR tree traversal.
    *
    * This is the heart of the algorithm. At each node:
    *  - '''Terminal''': returns the utility for player 0.
    *  - '''Chance''': sums over all chance outcomes weighted by their probabilities.
    *  - '''Player 0/1''': computes the current strategy via regret matching, recurses
    *    into each action's subtree, then updates cumulative regrets and strategy weights.
    *
    * The reach probabilities track how likely each player is to have played to reach
    * this state under their current strategies. They are used to weight the regret
    * and strategy updates:
    *  - Regret for player i is weighted by the opponent's reach probability (counterfactual).
    *  - Strategy accumulation is weighted by the player's own reach probability.
    *
    * Note: Player 1's regret sign is inverted because the game is zero-sum — player 1's
    * utility is the negation of player 0's.
    *
    * @param reachPlayer0  probability that player 0 plays to reach this state
    * @param reachPlayer1  probability that player 1 plays to reach this state
    * @param iteration     current iteration number (1-based)
    * @return the expected utility for player 0 at this node
    */
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

        // Update regrets and strategy weights for Player 0.
        // Regret delta = opponent_reach * (action_value - node_value).
        // The opponent's reach is used because this is "counterfactual" regret:
        // it measures what player 0 would have gained by deviating, weighted by
        // how likely the opponent was to have reached this point.
        val averagingWeight = averagingWeightFor(iteration, config)
        idx = 0
        while idx < actions.length do
          val regretDelta = reachPlayer1 * (utilities(idx) - nodeValue)
          val updatedRegret = info.cumulativeRegret(idx) + regretDelta
          // In CFR+, floor negative cumulative regrets to zero for faster convergence
          info.cumulativeRegret(idx) =
            if config.cfrPlus then math.max(0.0, updatedRegret) else updatedRegret
          // Accumulate strategy weight: own_reach * strategy_probability * averaging_weight
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

        // Update regrets and strategy weights for Player 1.
        // Note the inverted regret sign: (nodeValue - utilities[idx]) instead of
        // (utilities[idx] - nodeValue), because the game is zero-sum and all utilities
        // are expressed from player 0's perspective. Player 1 gains when player 0 loses.
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

  /** Fixed-point variant of [[cfr]]. Uses Prob/FixedVal Q1.30 arithmetic throughout. */
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

  /** Computes the expected utility for player 0 under the trained average strategies.
    *
    * This is a post-training tree walk that uses the average (output) strategies
    * at every information set to compute the game's expected value. Called once
    * after training completes to populate [[TrainingResult.expectedValuePlayer0]].
    */
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

  /** Normalizes chance outcome probabilities to sum to 1.0.
    *
    * Skips allocation when outcomes are already normalized (common case for
    * pre-normalized distributions). Over 1500 CFR iterations this avoids
    * creating ~1500 Vector copies of the ~96-element villain distribution.
    */
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

  /** Normalizes chance outcomes and converts probabilities to fixed-point Prob values.
    * Ensures the Prob values sum exactly to Prob.Scale (1.0 in Q0.30).
    */
  private def normalizeChanceOutcomesProb[S](raw: Vector[(S, Double)]): Vector[(S, Prob)] =
    val normalized = normalizeChanceOutcomes(raw)
    val probabilities = new Array[Double](normalized.length)
    var idx = 0
    while idx < normalized.length do
      probabilities(idx) = normalized(idx)._2
      idx += 1
    val probabilityRaws = normalizeProbabilitiesToRaw(probabilities)
    normalized.indices.map(idx => normalized(idx)._1 -> Prob(probabilityRaws(idx))).toVector

  /** Converts double probabilities to fixed-point Prob raw values, ensuring they sum
    * exactly to Prob.Scale. The largest element absorbs any rounding remainder.
    */
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

  /** Writes a uniform probability distribution in fixed-point Prob format.
    * Distributes Prob.Scale evenly, assigning the integer remainder to the first elements.
    */
  private def writeUniformProbabilities(target: Array[Int]): Unit =
    val base = Prob.Scale / target.length
    val remainder = Prob.Scale - (base * target.length)
    var idx = 0
    while idx < target.length do
      target(idx) = base + (if idx < remainder then 1 else 0)
      idx += 1

  /** Multiplies two Q0.30 fixed-point probabilities. Result = (left * right) >> 30.
    * Both inputs represent values in [0, 1] scaled by 2^30.
    */
  private inline def multiplyProbRaw(left: Int, right: Int): Int =
    ((left.toLong * right.toLong) >> 30).toInt

  /** Computes and stages fixed-point regret and strategy deltas for one information set,
    * then delegates to [[applyFixedActionUpdates]] for the atomic update with overflow protection.
    *
    * @param regretReachRaw    opponent's reach probability in Q0.30 (weights counterfactual regret)
    * @param selfReachRaw      own reach probability in Q0.30 (weights strategy accumulation)
    * @param averagingWeight   integer multiplier for linear averaging (iteration - delay)
    * @param invertRegretSign  true for player 1 (zero-sum: regret sign is flipped)
    */
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

  /** Atomically applies regret and strategy deltas to cumulative accumulators with overflow protection.
    *
    * The method implements a two-phase overflow prevention strategy:
    *  1. '''Emergency pre-rescale''': if any element would overflow its Int/Long range when the
    *     delta is added, halve the entire array before applying deltas.
    *  2. '''Threshold post-rescale''': after applying deltas, if any element exceeds the
    *     rescale threshold (Max/4), halve the array to maintain headroom for future iterations.
    *
    * This is package-private for testing the overflow behavior directly.
    */
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

  /** Multiplies a Q1.30 fixed-point value by a Q0.30 probability, returning Q1.30.
    * Uses [[roundShift30Signed]] for unbiased rounding of signed products.
    */
  private inline def multiplyFixedByProbRaw(valueRaw: Int, probabilityRaw: Int): Int =
    roundShift30Signed(valueRaw.toLong * probabilityRaw.toLong)

  /** Right-shifts a 60-bit product by 30 bits with unbiased rounding for signed values.
    *
    * Fixed utilities/regrets can be negative, so we round on absolute magnitude
    * before restoring the sign. Plain arithmetic right-shift (>>) biases small
    * negative values downward (toward negative infinity), which would accumulate
    * systematic error over millions of CFR iterations.
    */
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

  /** Computes the double-precision averaging weight for the given iteration.
    *
    * Returns 0.0 during the averaging delay period (early iterations are discarded).
    * With linear averaging, later iterations receive proportionally higher weight,
    * which improves convergence quality by de-emphasizing early noisy strategies.
    */
  private def averagingWeightFor(iteration: Int, config: Config): Double =
    if iteration <= config.averagingDelay then 0.0
    else if config.linearAveraging then (iteration - config.averagingDelay).toDouble
    else 1.0

  /** Integer variant of [[averagingWeightFor]] for the fixed-point CFR path. */
  private def averagingWeightMultiplier(iteration: Int, config: Config): Int =
    if iteration <= config.averagingDelay then 0
    else if config.linearAveraging then iteration - config.averagingDelay
    else 1
