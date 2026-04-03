package sicfun.holdem.strategic

import sicfun.holdem.types.{PokerAction, Street}

/** Trait for real-world baseline strategies derived from empirical play data.
  *
  * A baseline represents the observed frequency distribution of actions taken
  * by players of a given strategic class in specific board contexts. Used as a
  * reference point for deviation analysis and exploit identification.
  */
trait RealBaseline:
  def probability(
      cls: StrategicClass,
      action: PokerAction.Category,
      sizing: Option[Sizing],
      street: Street
  ): Double

/** Trait for attributed baselines that support human-interpretable reasoning chains.
  *
  * In addition to predicting action probabilities, an attributed baseline can explain
  * why a strategic class would select that action in a given context.
  */
trait AttributedBaseline:
  def probability(
      cls: StrategicClass,
      action: PokerAction.Category,
      sizing: Option[Sizing],
      street: Street
  ): Double
