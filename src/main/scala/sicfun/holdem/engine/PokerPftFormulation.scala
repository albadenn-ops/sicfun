package sicfun.holdem.engine

import sicfun.holdem.types.*
import sicfun.holdem.strategic.*
import sicfun.holdem.strategic.solver.{TabularGenerativeModel, ParticleBelief}

/** Builds tabular POMDP models from poker game state for PftDpw solver. */
object PokerPftFormulation:

  /** Build a TabularGenerativeModel from engine state.
    *
    * State space: street (4) as primary axis.
    * Action space: |heroActions|.
    * Observation space: rival action categories (|StrategicClass|).
    *
    * Rewards are derived from pot-odds and hand equity:
    *   - Fold: forfeit pot equity proportional to hand strength and pot commitment
    *   - Call: EV proportional to (equity - breakeven) * call cost fraction
    *   - Check: zero immediate cost
    *   - Raise: EV proportional to (equity - structural-bluff threshold) * sizing fraction
    *
    * Profile-conditioned models adjust Call/Check by opponent aggression
    * and Raise by opponent fold tendency from `actionPriors`.
    *
    * Mixed model (profileClass=None): obs likelihoods from aggregate rival
    * posterior when available, otherwise uniform.
    */
  def buildTabularModel(
      gameState: GameState,
      rivalBeliefs: Map[PlayerId, StrategicRivalBelief],
      heroActions: Vector[PokerAction],
      heroBucket: Int,
      actionPriors: Map[(StrategicClass, PokerAction.Category), Double],
      profileClass: Option[StrategicClass] = None
  ): TabularGenerativeModel =
    val numStates = 4 // one per street
    val numActions = heroActions.size
    val numObs = StrategicClass.values.length

    // Game-state-derived factors
    val equity = heroBucket / 9.0 // hand strength proxy [0, 1]
    val potFraction = gameState.pot / math.max(gameState.stackSize, 1.0)
    val callFraction = gameState.toCall / math.max(gameState.stackSize, 1.0)

    // Deterministic transitions: action leads to next street or terminal
    val transitionTable = new Array[Int](numStates * numActions)
    var s = 0
    while s < numStates do
      var a = 0
      while a < numActions do
        transitionTable(s * numActions + a) = math.min(s + 1, numStates - 1)
        a += 1
      s += 1

    // Observation likelihood: profile-conditioned or rival-belief-informed
    val obsLikelihood = profileClass match
      case Some(cls) =>
        val profileIdx = cls.ordinal
        val concentration = 0.7
        val remainder = if numObs > 1 then (1.0 - concentration) / (numObs - 1) else 0.0
        val arr = new Array[Double](numStates * numActions * numObs)
        var i = 0
        while i < numStates * numActions do
          var o = 0
          while o < numObs do
            arr(i * numObs + o) = if o == profileIdx then concentration else remainder
            o += 1
          i += 1
        arr
      case None =>
        if rivalBeliefs.nonEmpty then
          val aggregate = aggregateRivalPosterior(rivalBeliefs)
          val arr = new Array[Double](numStates * numActions * numObs)
          var i = 0
          while i < numStates * numActions do
            var o = 0
            while o < numObs do
              arr(i * numObs + o) = aggregate(o)
              o += 1
            i += 1
          arr
        else
          Array.fill(numStates * numActions * numObs)(1.0 / numObs)

    // Reward table: pot-odds / equity-grounded base rewards + profile conditioning
    // Equity realization: preflop equity is speculative, river equity is fully realized.
    // Scale rewards by (s+1)/numStates so earlier streets discount and later streets amplify.
    val rewardTable = new Array[Double](numStates * numActions)
    s = 0
    while s < numStates do
      val eqRealization = (s + 1).toDouble / numStates
      var a = 0
      while a < numActions do
        val baseReward = heroActions(a) match
          case PokerAction.Fold =>
            -(equity * potFraction)
          case PokerAction.Call =>
            (equity - 0.5) * callFraction
          case PokerAction.Check =>
            0.0
          case r: PokerAction.Raise =>
            val raiseFraction = r.amount / math.max(gameState.stackSize, 1.0)
            (equity - 0.3) * raiseFraction
        val conditioned = profileClass match
          case Some(cls) =>
            val rivalFoldProb = actionPriors.getOrElse((cls, PokerAction.Category.Fold), 0.25)
            val rivalRaiseProb = actionPriors.getOrElse((cls, PokerAction.Category.Raise), 0.25)
            heroActions(a) match
              case PokerAction.Fold => baseReward
              case PokerAction.Call | PokerAction.Check =>
                baseReward - rivalRaiseProb * potFraction * 0.3
              case _: PokerAction.Raise =>
                baseReward * (0.5 + rivalFoldProb)
          case None => baseReward
        rewardTable(s * numActions + a) = conditioned * eqRealization
        a += 1
      s += 1

    TabularGenerativeModel(transitionTable, obsLikelihood, rewardTable,
      numStates, numActions, numObs)

  /** Aggregate rival type posteriors into a single observation distribution. */
  private def aggregateRivalPosterior(
      rivalBeliefs: Map[PlayerId, StrategicRivalBelief]
  ): Array[Double] =
    val classes = StrategicClass.values
    val numObs = classes.length
    val result = new Array[Double](numObs)
    for (_, belief) <- rivalBeliefs do
      var o = 0
      while o < numObs do
        result(o) += belief.typePosterior.probabilityOf(classes(o))
        o += 1
    val total = result.sum
    if total > 0 then
      var o = 0
      while o < numObs do
        result(o) /= total
        o += 1
    else
      java.util.Arrays.fill(result, 1.0 / numObs)
    result

  /** Build a model for V^{1,0} (attrib kernel, open-loop policy).
    *
    * Attrib rewards (rival adapts to hero's play) + uniform obs likelihoods
    * (hero cannot condition on observations). The POMDP solver degenerates
    * to MDP-like behavior because observations carry no information.
    *
    * Grid world: (Attrib, OpenLoop) in Omega^grid.
    */
  def buildOpenLoopModel(
      gameState: GameState,
      rivalBeliefs: Map[PlayerId, StrategicRivalBelief],
      heroActions: Vector[PokerAction],
      heroBucket: Int,
      actionPriors: Map[(StrategicClass, PokerAction.Category), Double],
      profileClass: Option[StrategicClass] = None
  ): TabularGenerativeModel =
    // Build attrib model (normal rewards), then override obs to uniform
    val attribModel = buildTabularModel(
      gameState, rivalBeliefs, heroActions, heroBucket, actionPriors, profileClass
    )
    val numObs = attribModel.numObs
    val uniformObs = Array.fill(
      attribModel.numStates * attribModel.numActions * numObs
    )(1.0 / numObs)
    attribModel.copy(obsLikelihood = uniformObs)

  /** Build a model for V^{0,1} (blind kernel, closed-loop policy).
    *
    * Baseline rewards (rival does NOT adapt — no profile modulation) +
    * normal obs likelihoods (hero observes rival normally).
    *
    * Grid world: (Blind, ClosedLoop) in Omega^grid.
    */
  def buildBlindKernelModel(
      gameState: GameState,
      rivalBeliefs: Map[PlayerId, StrategicRivalBelief],
      heroActions: Vector[PokerAction],
      heroBucket: Int,
      actionPriors: Map[(StrategicClass, PokerAction.Category), Double]
  ): TabularGenerativeModel =
    // Build baseline model (profileClass=None -> no profile modulation in rewards)
    // with normal obs likelihoods (hero can still observe)
    buildTabularModel(
      gameState, rivalBeliefs, heroActions, heroBucket, actionPriors,
      profileClass = None
    )

  /** Build a model for V^{0,0} (blind kernel, open-loop policy).
    *
    * Baseline rewards (rival does NOT adapt — no profile modulation) +
    * uniform obs likelihoods (hero cannot condition on observations).
    * The POMDP solver degenerates to a blind MDP: no learning, no signaling.
    *
    * Grid world: (Blind, OpenLoop) in Omega^grid.
    */
  def buildBlindOpenLoopModel(
      gameState: GameState,
      rivalBeliefs: Map[PlayerId, StrategicRivalBelief],
      heroActions: Vector[PokerAction],
      heroBucket: Int,
      actionPriors: Map[(StrategicClass, PokerAction.Category), Double]
  ): TabularGenerativeModel =
    // Build blind model (baseline rewards), then override obs to uniform
    val blindModel = buildBlindKernelModel(
      gameState, rivalBeliefs, heroActions, heroBucket, actionPriors
    )
    val numObs = blindModel.numObs
    val uniformObs = Array.fill(
      blindModel.numStates * blindModel.numActions * numObs
    )(1.0 / numObs)
    blindModel.copy(obsLikelihood = uniformObs)

  /** Build a ParticleBelief from rival beliefs.
    *
    * Concentrates belief mass on the current street state.
    */
  def buildParticleBelief(
      rivalBeliefs: Map[PlayerId, StrategicRivalBelief],
      particlesPerRival: Int,
      currentStreet: Street = Street.Preflop
  ): ParticleBelief =
    val numStates = 4
    val streetIdx = currentStreet.ordinal
    val numParticles = math.min(particlesPerRival, numStates)
    val indices = (0 until numParticles).toArray
    val targetIdx = math.min(streetIdx, numParticles - 1)
    val weights = Array.tabulate(numParticles) { i =>
      if i == targetIdx then 0.7
      else 0.3 / math.max(numParticles - 1, 1)
    }
    val sum = weights.sum
    var i = 0
    while i < weights.length do
      weights(i) /= sum
      i += 1
    ParticleBelief(indices, weights)
