package sicfun.holdem.strategic.safety

import sicfun.holdem.types.PokerAction
import sicfun.holdem.strategic.types.StrategicClass
import sicfun.holdem.strategic.state.{PublicState, Sizing}

/** Board-blind floor implementation of the Def 9 real baseline: returns the
  * configured `(class, action)` constant, or 0.25 when absent. This reproduces
  * EXACTLY the pre-P1 behavior (`actionPriors.getOrElse((cls,cat), 0.25)`), so wiring
  * it in with no calibration artifact is a guaranteed no-op.
  */
final class ConstantRealBaseline(
    actionPriors: Map[(StrategicClass, PokerAction.Category), Double]
) extends RealBaseline:
  def probability(
      cls: StrategicClass,
      action: PokerAction.Category,
      sizing: Option[Sizing],
      publicState: PublicState
  ): Double =
    actionPriors.getOrElse((cls, action), 0.25)
