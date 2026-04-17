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
  private def bucketForStrength(strength: Double, numBuckets: Int): Int =
    math.min(numBuckets - 1, math.max(0, (strength * numBuckets).toInt))

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
    val calibrated = PokerPomcpFormulation.buildShowdownEquity(numHeroBuckets, numRivalBuckets)
    spot.heroValueInput match
      case HeroValueInput.ExactHoleCards(cards) =>
        val gs = spot.gameState
        val exactStrength = HandStrengthEstimator.fastGtoStrength(cards, gs.board, gs.street)
        val heroBucket = bucketForStrength(exactStrength, numHeroBuckets)
        val rowBase = heroBucket * numRivalBuckets
        var rivalBucket = 0
        while rivalBucket < numRivalBuckets do
          calibrated(rowBase + rivalBucket) = PokerPomcpFormulation.calibratedBucketEquity(
            heroStrength = exactStrength,
            rivalBucket = rivalBucket,
            numRivalBuckets = numRivalBuckets
          )
          rivalBucket += 1
        BridgeResult.Approximate(
          calibrated,
          "calibrated showdown equity with exact hero row"
        )
      case HeroValueInput.StrengthHint(_, _) =>
        BridgeResult.Approximate(
          calibrated,
          "calibrated percentile showdown equity"
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
