package sicfun.holdem

import sicfun.core.Metrics

/** A fixed-dimension behavioral profile vector summarizing a player's tendencies.
  *
  * Each element corresponds to a named feature (see [[PlayerSignature.featureNames]]).
  * Signatures are used as inputs to [[PlayerCluster]] for archetype classification
  * and [[LongitudinalAnalysis]] for behavioral drift detection.
  *
  * @param values ordered feature values; indices match [[PlayerSignature.featureNames]]
  */
final case class PlayerSignature(values: Vector[Double])

/** Factory and distance utilities for constructing [[PlayerSignature]] vectors
  * from observed poker actions.
  *
  * The signature captures six dimensions of player behavior:
  * fold rate, raise rate, call rate, check rate, average action entropy,
  * and average pot odds when calling.
  */
object PlayerSignature:

  /** Human-readable names for each feature dimension, in order. */
  val featureNames: Vector[String] = Vector(
    "foldRate",
    "raiseRate",
    "callRate",
    "checkRate",
    "avgActionEntropy",
    "avgPotOddsWhenCalling"
  )

  /** Build a fixed-dimension behavioral profile from observed actions.
    *
    * Entropy is computed directly from the empirical action frequency distribution
    * (Shannon entropy in bits). This measures how unpredictable the player's action
    * choices are without requiring a model or hole-card information.
    *
    * @param observations sequence of (state, action) pairs
    */
  def compute(
      observations: Seq[(GameState, PokerAction)]
  ): PlayerSignature =
    require(observations.nonEmpty, "observations must be non-empty")

    val n = observations.length.toDouble
    val categories = observations.map { case (_, action) => action.category }

    val foldRate = categories.count(_ == PokerAction.Category.Fold) / n
    val raiseRate = categories.count(_ == PokerAction.Category.Raise) / n
    val callRate = categories.count(_ == PokerAction.Category.Call) / n
    val checkRate = categories.count(_ == PokerAction.Category.Check) / n

    // Empirical Shannon entropy (in bits) of the observed action frequency distribution.
    val avgActionEntropy = Metrics.entropy(Vector(foldRate, raiseRate, callRate, checkRate))

    val callObs = observations.filter { case (_, action) =>
      action.category == PokerAction.Category.Call
    }
    val avgPotOddsWhenCalling =
      if callObs.isEmpty then 0.0
      else callObs.map(_._1.potOdds).sum / callObs.length

    PlayerSignature(Vector(foldRate, raiseRate, callRate, checkRate, avgActionEntropy, avgPotOddsWhenCalling))

  /** Euclidean distance between two player signatures. */
  def distance(a: PlayerSignature, b: PlayerSignature): Double =
    require(a.values.length == b.values.length, "signatures must have same dimension")
    math.sqrt(
      (a.values zip b.values).map { case (x, y) => math.pow(x - y, 2) }.sum
    )

