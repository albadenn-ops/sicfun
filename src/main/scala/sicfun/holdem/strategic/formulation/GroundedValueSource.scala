package sicfun.holdem.strategic.formulation

import sicfun.holdem.engine.{HandStrengthEstimator, PokerPomcpFormulation}
import sicfun.holdem.strategic.types.*
import sicfun.holdem.types.*

/** Poker-grounded value source for the deprecated formulation path.
  *
  * Uses exact hole cards when available and falls back to the legacy bucket proxy
  * for A2 compatibility when only a StrengthHint is present.
  */
object GroundedValueSource extends FormulationValueSource:

  override def estimateSpotEquity(spot: FormulationSpot): BridgeResult[Double] =
    spot.heroValueInput match
      case HeroValueInput.ExactHoleCards(cards) =>
        val gs = spot.gameState
        BridgeResult.Exact(
          HandStrengthEstimator.fastGtoStrength(cards, gs.board, gs.street)
        )
      case HeroValueInput.StrengthHint(bucket, _) =>
        BridgeResult.Approximate(bucket / 9.0, "fallback: bucket / 9.0")

  override def showdownEquityTable(
      spot: FormulationSpot,
      numHeroBuckets: Int,
      numRivalBuckets: Int
  ): BridgeResult[Array[Double]] =
    BridgeResult.Approximate(
      PokerPomcpFormulation.buildLinearShowdownEquity(numHeroBuckets, numRivalBuckets),
      "A3: linear showdown equity heuristic retained"
    )

  override def estimateActionValue(
      spot: FormulationSpot,
      action: PokerAction
  ): BridgeResult[Ev] =
    val equity = estimateSpotEquity(spot) match
      case BridgeResult.Exact(eq) => eq
      case BridgeResult.Approximate(eq, _) => eq
      case BridgeResult.Absent(_) => 0.5

    val gs = spot.gameState
    val stack = math.max(gs.stackSize, 1.0)
    val potFraction = gs.pot / stack
    val callFraction = gs.toCall / stack
    val breakeven = gs.potOdds

    val value = action match
      case PokerAction.Fold =>
        -(equity * potFraction)
      case PokerAction.Call =>
        (equity - breakeven) * callFraction
      case PokerAction.Check =>
        0.0
      case raise: PokerAction.Raise =>
        val raiseFraction = raise.amount / stack
        (equity - breakeven) * raiseFraction

    spot.heroValueInput match
      case _: HeroValueInput.ExactHoleCards =>
        BridgeResult.Exact(Ev(value))
      case _: HeroValueInput.StrengthHint =>
        BridgeResult.Approximate(Ev(value), "derived from bucket equity")
