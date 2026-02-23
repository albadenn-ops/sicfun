package sicfun.holdem

/** A fixed-dimension vector of observable (public) features extracted from a [[PokerEvent]].
  *
  * Unlike [[PokerFeatures]], this does not include hand-strength estimates derived
  * from private hole cards, making it suitable for opponent modeling from observed
  * actions alone.
  *
  * @param values the ordered feature values; indices correspond to [[FeatureExtractor.featureNames]]
  */
final case class ObservableFeatures(values: Vector[Double]):
  /** The number of features in this vector. */
  def dimension: Int = values.length

/** Extracts 8 normalized features from a [[PokerEvent]], using only publicly
  * observable information (no hole cards required).
  *
  * All features are scaled to [0, 1] with the following normalization ranges:
  *
  *   - '''potOdds''' -- `toCall / (pot + toCall)`, naturally in [0, 1)
  *   - '''stackToPot''' -- SPR clamped to [0, 10], then divided by 10
  *   - '''streetOrdinal''' -- street ordinal / 3 (Preflop=0.0 .. River=1.0)
  *   - '''positionOrdinal''' -- position ordinal / 7 (SB=0.0 .. BTN=1.0)
  *   - '''boardSize''' -- number of community cards / 5 (0.0 .. 1.0)
  *   - '''toCallOverStack''' -- `toCall / stack`, clamped to [0, 1] (1.0 if all-in)
  *   - '''decisionTime''' -- thinking time in seconds, clamped to [0, 30], then / 30
  *   - '''historyLength''' -- number of prior bets, clamped to [0, 20], then / 20
  */
object FeatureExtractor:
  /** Human-readable names for each feature dimension, in order. */
  val featureNames: Vector[String] = Vector(
    "potOdds",
    "stackToPot",
    "streetOrdinal",
    "positionOrdinal",
    "boardSize",
    "toCallOverStack",
    "decisionTime",
    "historyLength"
  )

  /** Number of features produced by [[extract]]. */
  val dimension: Int = featureNames.length

  /** Extracts an [[ObservableFeatures]] vector from a single poker event.
    *
    * @param event the poker event containing the decision-point context
    * @return normalized feature vector with [[dimension]] elements, all in [0, 1]
    */
  def extract(event: PokerEvent): ObservableFeatures =
    val potOdds =
      if event.toCall <= 0.0 then 0.0
      else event.toCall / (event.potBefore + event.toCall)

    val stackToPot =
      val ratio = if event.potBefore <= 0.0 then 10.0 else event.stackBefore / event.potBefore
      math.min(ratio, 10.0) / 10.0  // clamp SPR at 10, normalize to [0, 1]

    val streetOrdinal = event.street.ordinal.toDouble / 3.0       // Preflop=0.0, River=1.0
    val positionOrdinal = event.position.ordinal.toDouble / 7.0   // SB=0.0, BTN=1.0
    val boardSize = event.board.size.toDouble / 5.0               // 0..5 cards -> [0, 1]
    val toCallOverStack =
      if event.stackBefore <= 0.0 then 1.0  // effectively all-in
      else math.min(event.toCall / event.stackBefore, 1.0)
    val decisionTime =
      val seconds = event.decisionTimeMillis.getOrElse(0L).toDouble / 1000.0
      math.min(seconds, 30.0) / 30.0  // cap at 30 seconds
    val historyLength = math.min(event.betHistory.length.toDouble, 20.0) / 20.0  // cap at 20 actions

    ObservableFeatures(Vector(
      potOdds,
      stackToPot,
      streetOrdinal,
      positionOrdinal,
      boardSize,
      toCallOverStack,
      decisionTime,
      historyLength
    ))
