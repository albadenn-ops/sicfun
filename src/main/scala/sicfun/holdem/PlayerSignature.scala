package sicfun.holdem

import sicfun.core.{ActionMetrics, ActionModel, DiscreteDistribution}

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
    * @param observations sequence of (state, action) pairs
    * @param model the action model used for entropy computation
    */
  def compute(
      observations: Seq[(GameState, PokerAction)],
      model: ActionModel[GameState, PokerAction, HoleCards]
  ): PlayerSignature =
    require(observations.nonEmpty, "observations must be non-empty")

    val n = observations.length.toDouble
    val categories = observations.map { case (_, action) => PokerAction.categoryOf(action) }

    val foldRate = categories.count(_ == PokerAction.Category.Fold) / n
    val raiseRate = categories.count(_ == PokerAction.Category.Raise) / n
    val callRate = categories.count(_ == PokerAction.Category.Call) / n
    val checkRate = categories.count(_ == PokerAction.Category.Check) / n

    // Average action entropy across observations.
    // Use a uniform posterior over a dummy hand set since we measure
    // marginal action predictability per state.
    val actionSeq: Seq[PokerAction] = Seq(
      PokerAction.Fold, PokerAction.Check, PokerAction.Call, PokerAction.Raise(1.0)
    )
    val avgActionEntropy =
      val entropies = observations.map { case (state, _) =>
        // Pick a dummy hand that doesn't collide with the board so that
        // PokerFeatures.computeHandStrength never receives duplicate cards.
        val dummyPosterior = DiscreteDistribution.uniform(Seq(dummyHandFor(state.board)))
        ActionMetrics.actionEntropy(state, dummyPosterior, model, actionSeq)
      }
      entropies.sum / entropies.length

    val callObs = observations.filter { case (_, action) =>
      PokerAction.categoryOf(action) == PokerAction.Category.Call
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

  // Pick the first two deck cards not already on the board so that
  // PokerFeatures.computeHandStrength never sees duplicate cards.
  private def dummyHandFor(board: Board): HoleCards =
    import sicfun.core.Deck
    val boardSet = board.asSet
    val available = Deck.full.filterNot(boardSet.contains)
    HoleCards(available(0), available(1))
