package sicfun.holdem.strategic

/** Detection predicate for rival modeling awareness (A6').
  *
  * DetectModeling^i : H_t -> {0, 1}
  *
  * Returns true when rival i is inferred to be actively modeling
  * SICFUN's strategy. Triggers exploitation retreat (Def 15C).
  *
  * This is a trait because the detection mechanism is pluggable:
  * concrete implementations may use frequency anomaly detection,
  * timing tells, strategy deviation signatures, etc.
  */
trait DetectionPredicate:
  /** Evaluate whether rival i appears to be modeling SICFUN.
    *
    * @param rivalId the rival under evaluation
    * @param history the observable action history (public actions only)
    * @param publicState current public game state
    * @return true if modeling is detected
    */
  def detectModeling(
      rivalId: PlayerId,
      history: Vector[PublicAction],
      publicState: PublicState
  ): Boolean

/** Always-false detection (default: never detect modeling).
  * Used when detection is disabled or as a test stub.
  * With this predicate, beta never retreats via detection.
  */
object NeverDetect extends DetectionPredicate:
  def detectModeling(
      rivalId: PlayerId,
      history: Vector[PublicAction],
      publicState: PublicState
  ): Boolean = false

/** Always-true detection (test stub: always detect modeling).
  * Forces immediate retreat on every update.
  */
object AlwaysDetect extends DetectionPredicate:
  def detectModeling(
      rivalId: PlayerId,
      history: Vector[PublicAction],
      publicState: PublicState
  ): Boolean = true

/** Frequency-anomaly detection: detects modeling when rival's action
  * distribution deviates from expected baseline by more than a threshold.
  *
  * This is a concrete implementation suitable for initial deployment.
  * Looks at the last `window` actions and checks if the fraction of
  * counter-exploitative adjustments exceeds `threshold`.
  *
  * A "counter-exploitative adjustment" is detected when the rival's
  * action frequency shifts toward actions that specifically counter
  * SICFUN's current strategy (e.g., increased 3-bet frequency when
  * SICFUN has been opening wide).
  */
final class FrequencyAnomalyDetection(
    window: Int,
    threshold: Double
) extends DetectionPredicate:
  require(window > 0, "window must be positive")
  require(threshold > 0.0 && threshold <= 1.0, "threshold must be in (0, 1]")

  def detectModeling(
      rivalId: PlayerId,
      history: Vector[PublicAction],
      publicState: PublicState
  ): Boolean =
    if history.size < window then false
    else
      val recentActions = history.takeRight(window)
      val rivalActions = recentActions.filter(_.actor == rivalId)
      if rivalActions.isEmpty then false
      else
        // Count aggressive actions (raises/reraises) as proxy for counter-exploitation
        val aggressiveCount = rivalActions.count(_.signal.isAggressiveWager)
        val aggressiveFraction = aggressiveCount.toDouble / rivalActions.size.toDouble
        aggressiveFraction > threshold
