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

/** Attributed baseline: per-rival, state-conditioned policy (Def 10).
  *
  * hat{pi}^{0,S,i}(a, lambda | c, x^pub, m^{R,i})
  *
  * The attributed baseline conditions on the rival's belief state m^{R,i},
  * allowing different baselines per rival based on what we believe about them.
  */
trait AttributedBaseline:
  def probability(
      cls: StrategicClass,
      action: PokerAction.Category,
      sizing: Option[Sizing],
      street: Street,
      rivalState: RivalBeliefState
  ): Double
