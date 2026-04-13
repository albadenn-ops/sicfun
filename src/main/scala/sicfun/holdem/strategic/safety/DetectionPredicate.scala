package sicfun.holdem.strategic.safety
import sicfun.holdem.strategic.types.*
import sicfun.holdem.strategic.state.*

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
  * counter-exploitative adjustments exceeds the baseline frequency by
  * more than `threshold` (spec condition ii: detection fires only when
  * observed frequency exceeds baseline tolerance).
  *
  * @param window number of recent actions to consider
  * @param threshold detection margin above baseline (fires when observed - baseline > threshold)
  * @param baselineFrequency expected baseline aggressive frequency under non-modeling play
  */
final class FrequencyAnomalyDetection(
    window: Int,
    threshold: Double,
    baselineFrequency: Double = 0.0
) extends DetectionPredicate:
  require(window > 0, "window must be positive")
  require(threshold > 0.0 && threshold <= 1.0, "threshold must be in (0, 1]")
  require(baselineFrequency >= 0.0 && baselineFrequency <= 1.0, "baselineFrequency must be in [0, 1]")

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
        // Spec condition ii: compare against baseline, not absolute threshold.
        // Returns false (no detection) when observed frequency is within baseline tolerance.
        aggressiveFraction - baselineFrequency > threshold