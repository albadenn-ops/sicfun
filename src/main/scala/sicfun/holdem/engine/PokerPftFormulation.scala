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
    */
  def buildTabularModel(
      gameState: GameState,
      rivalBeliefs: Map[PlayerId, StrategicRivalBelief],
      heroActions: Vector[PokerAction],
      heroBucket: Int,
      actionPriors: Map[(StrategicClass, PokerAction.Category), Double]
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

    // Observation likelihood: uniform initially
    val obsLikelihood = Array.fill(numStates * numActions * numObs)(1.0 / numObs)

    // Reward table: based on action effects
    val rewardTable = new Array[Double](numStates * numActions)
    s = 0
    while s < numStates do
      var a = 0
      while a < numActions do
        val potFraction = if a < heroActions.size then
          heroActions(a) match
            case PokerAction.Fold => -1.0
            case PokerAction.Call => 0.0
            case PokerAction.Check => 0.0
            case r: PokerAction.Raise => r.amount * 0.01
        else 0.0
        rewardTable(s * numActions + a) = potFraction
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
