package sicfun.holdem.engine

import sicfun.holdem.types.*
import sicfun.holdem.strategic.*
import sicfun.holdem.strategic.solver.{TabularGenerativeModel, ParticleBelief}

/** Builds tabular POMDP models from poker game state for PftDpw solver. */
object PokerPftFormulation:

  /** Build a TabularGenerativeModel from engine state.
    *
    * State space: street (4) as primary axis for v1.
    * Action space: |heroActions|.
    * Observation space: rival action categories (|StrategicClass|).
    *
    * When `profileClass` is None, observation likelihoods are uniform and
    * rewards reflect only hero action effects (mixed-belief model).
    * When `profileClass` is Some(cls), observation likelihoods concentrate
    * on the profile type and rewards are adjusted by the rival type's
    * response distribution from `actionPriors`.
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

    // Deterministic transitions: action leads to next street or terminal
    val transitionTable = new Array[Int](numStates * numActions)
    var s = 0
    while s < numStates do
      var a = 0
      while a < numActions do
        transitionTable(s * numActions + a) = math.min(s + 1, numStates - 1)
        a += 1
      s += 1

    // Observation likelihood: profile-conditioned or uniform
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
        Array.fill(numStates * numActions * numObs)(1.0 / numObs)

    // Reward table: base action effects + profile-conditioned rival response adjustment
    val rewardTable = new Array[Double](numStates * numActions)
    s = 0
    while s < numStates do
      var a = 0
      while a < numActions do
        val baseReward = heroActions(a) match
          case PokerAction.Fold => -1.0
          case PokerAction.Call => 0.0
          case PokerAction.Check => 0.0
          case r: PokerAction.Raise => r.amount * 0.01
        rewardTable(s * numActions + a) = profileClass match
          case Some(cls) =>
            // Rival's fold probability modulates raise profitability;
            // rival's raise probability modulates call/check risk.
            val rivalFoldProb = actionPriors.getOrElse((cls, PokerAction.Category.Fold), 0.25)
            val rivalRaiseProb = actionPriors.getOrElse((cls, PokerAction.Category.Raise), 0.25)
            heroActions(a) match
              case PokerAction.Fold => baseReward
              case PokerAction.Call | PokerAction.Check =>
                // More negative against aggressive rivals (higher rival raise prob)
                baseReward - rivalRaiseProb * 0.5
              case _: PokerAction.Raise =>
                // More profitable against foldy rivals, less against calling stations
                baseReward * (0.5 + rivalFoldProb)
          case None => baseReward
        a += 1
      s += 1

    TabularGenerativeModel(transitionTable, obsLikelihood, rewardTable,
      numStates, numActions, numObs)

  /** Build a ParticleBelief from rival beliefs.
    *
    * Simple uniform belief over states for v1.
    */
  def buildParticleBelief(
      rivalBeliefs: Map[PlayerId, StrategicRivalBelief],
      particlesPerRival: Int
  ): ParticleBelief =
    val numStates = 4
    val numParticles = math.min(particlesPerRival, numStates)
    val indices = (0 until numParticles).toArray
    val weights = Array.fill(numParticles)(1.0 / numParticles)
    ParticleBelief(indices, weights)
