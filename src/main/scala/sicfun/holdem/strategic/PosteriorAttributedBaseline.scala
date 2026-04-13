package sicfun.holdem.strategic

import sicfun.holdem.types.PokerAction
import sicfun.holdem.strategic.types.*
import sicfun.holdem.strategic.state.*
import sicfun.holdem.strategic.safety.AttributedBaseline

/** Posterior-predictive attributed baseline (Def 10).
  *
  * hat_pi(a | c, x, m) = pi0(a | c) * w(a, m) / Z(c, m)
  *
  * where:
  *   w(a, m)   = p_pred(a | m) / p_ref(a)
  *   p_pred    = sum_{c'} P(c' | m) * pi0(a | c')
  *   p_ref     = (1/|C|) * sum_{c'} pi0(a | c')
  *   Z(c, m)   = sum_{a'} pi0(a' | c) * w(a', m)
  *
  * Stateless over belief: captures only actionPriors (immutable config).
  * Per-rival differentiation from call-time rivalState.
  */
class PosteriorAttributedBaseline(
    actionPriors: Map[(StrategicClass, PokerAction.Category), Double]
) extends AttributedBaseline:

  private val Eps = 1e-10
  private val classes = StrategicClass.values
  private val actions = PokerAction.Category.values
  private val numClasses = classes.length

  private def pi0(cls: StrategicClass, cat: PokerAction.Category): Double =
    actionPriors.getOrElse((cls, cat), 0.25)

  def probability(
      cls: StrategicClass,
      action: PokerAction.Category,
      sizing: Option[Sizing],
      publicState: PublicState,
      rivalState: RivalBeliefState
  ): Double =
    rivalState match
      case srb: StrategicRivalBelief =>
        val posterior = srb.typePosterior
        val weights = actions.map { a =>
          val pPred = math.max(Eps, classes.map(c => posterior.probabilityOf(c) * pi0(c, a)).sum)
          val pRef = math.max(Eps, classes.map(c => pi0(c, a)).sum / numClasses)
          a -> (pPred / pRef)
        }.toMap
        val z = math.max(Eps, actions.map(a => pi0(cls, a) * weights(a)).sum)
        pi0(cls, action) * weights(action) / z
      case _ =>
        pi0(cls, action)
