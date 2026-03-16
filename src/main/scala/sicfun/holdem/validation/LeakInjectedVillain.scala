package sicfun.holdem.validation

import sicfun.holdem.types.PokerAction
import scala.util.Random

/** The result of a villain's decision, recording whether a leak fired and which one. */
final case class VillainDecisionResult(
    action: PokerAction,
    leakFired: Boolean,
    leakId: Option[String],
    leakApplicable: Boolean = false
)

/** A simulated opponent that plays a noised GTO baseline with optional injected leaks.
  *
  * Decision flow:
  * 1. Check leaks in order -- first applicable leak whose severity roll fires wins.
  * 2. If no leak fired, apply baseline noise jitter to the GTO action.
  *
  * @param name          human-readable villain identifier
  * @param leaks         ordered leak vector; first applicable leak that fires wins
  * @param baselineNoise probability [0, 1] of random action perturbation when no leak fires
  * @param seed          RNG seed for deterministic replay
  */
final case class LeakInjectedVillain(
    name: String,
    leaks: Vector[InjectedLeak],
    baselineNoise: Double,
    seed: Long
):
  private val rng = new Random(seed)

  /** Decide villain action given the GTO-optimal action and current spot context.
    *
    * Checks leaks in declaration order. The first applicable leak that actually
    * deviates (produces a different action than gtoAction) is returned. If no
    * leak fires, baseline noise is applied.
    */
  def decide(gtoAction: PokerAction, spot: SpotContext): VillainDecisionResult =
    val applicableLeaks = leaks.filter(_.applies(spot))
    val anyApplicable = applicableLeaks.nonEmpty

    val leakResult = applicableLeaks.iterator
      .map { leak =>
        val deviated = leak.deviate(gtoAction, spot, rng)
        (leak, deviated, deviated != gtoAction)
      }
      .find(_._3)

    leakResult match
      case Some((leak, deviatedAction, _)) =>
        VillainDecisionResult(deviatedAction, leakFired = true, leakId = Some(leak.id), leakApplicable = true)
      case None =>
        val noisedAction = applyNoise(gtoAction, spot)
        VillainDecisionResult(noisedAction, leakFired = false, leakId = None, leakApplicable = anyApplicable)

  /** Small random perturbation: with probability `baselineNoise`, swap to a random
    * alternative action from {Fold, Check, Call} excluding the current action type.
    */
  private def applyNoise(action: PokerAction, spot: SpotContext): PokerAction =
    if baselineNoise <= 0.0 || rng.nextDouble() >= baselineNoise then action
    else
      val alternatives = action match
        case PokerAction.Fold     => Vector(PokerAction.Check, PokerAction.Call)
        case PokerAction.Check    => Vector(PokerAction.Fold, PokerAction.Call)
        case PokerAction.Call     => Vector(PokerAction.Fold, PokerAction.Check)
        case PokerAction.Raise(_) => Vector(PokerAction.Call, PokerAction.Check)
      alternatives(rng.nextInt(alternatives.size))
