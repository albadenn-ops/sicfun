package sicfun.holdem.strategic.bridge

import sicfun.holdem.strategic.*
import sicfun.holdem.types.{PokerAction, Street}

/** Bridge: PokerAction -> ActionSignal, TotalSignal.
  *
  * Fidelity:
  * - action category: Exact
  * - sizing: Approximate (pot fraction requires pot context)
  * - timing: Absent (no timing data in current engine)
  * - showdown: Exact when present
  */
object SignalBridge:

  /** Convert a PokerAction + context into an ActionSignal. */
  def toActionSignal(
      action: PokerAction,
      stage: Street,
      potSize: Chips
  ): BridgeResult[ActionSignal] =
    val category = action.category
    val sizing = action match
      case PokerAction.Raise(amount) =>
        val frac = if potSize.value > 0.0 then PotFraction(amount / potSize.value)
                   else PotFraction(1.0)
        Some(Sizing(Chips(amount), frac))
      case _ => None

    val signal = ActionSignal(
      action = category,
      sizing = sizing,
      timing = None,  // Absent: no timing in engine
      stage = stage
    )
    if sizing.isDefined then
      BridgeResult.Approximate(signal, "pot-fraction is approximate; timing absent")
    else
      BridgeResult.Approximate(signal, "timing absent")

  /** Convert an action into a TotalSignal (no showdown). */
  def toTotalSignal(
      action: PokerAction,
      stage: Street,
      potSize: Chips
  ): BridgeResult[TotalSignal] =
    toActionSignal(action, stage, potSize).fold(
      onExact = act => BridgeResult.Exact(TotalSignal(actionSignal = act, showdown = None)),
      onApprox = (act, loss) => BridgeResult.Approximate(TotalSignal(actionSignal = act, showdown = None), loss),
      onAbsent = reason => BridgeResult.Absent(reason)
    )
