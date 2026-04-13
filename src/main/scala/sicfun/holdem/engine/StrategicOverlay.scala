package sicfun.holdem.engine

import sicfun.holdem.types.*
import sicfun.holdem.strategic.types.*
import sicfun.holdem.strategic.state.*
import sicfun.holdem.strategic.exploitation.ExploitationState
import sicfun.holdem.engine.inference.ActionEvaluation

/** Overlay result types — NOT a DecisionEvaluationBundle. */
final case class OverlayInput(
    gameState: GameState,
    upstreamEvs: Vector[ActionEvaluation],
    rivalBeliefs: Map[PlayerId, StrategicRivalBelief],
    exploitationStates: Map[PlayerId, ExploitationState],
    robustLowerBounds: Option[Array[Double]],
    config: StrategicEngine.Config
)

final case class OverlayResult(
    selectedAction: PokerAction,
    rankedActions: Vector[ActionEvaluation],
    softVetoed: Vector[(PokerAction, String)],
    adjustments: Vector[OverlayAdjustment],
    upstreamAction: PokerAction,
    upstreamSource: UpstreamSource
)

final case class OverlayAdjustment(
    action: PokerAction,
    originalEv: Double,
    adjustedEv: Double,
    reason: String
)

enum UpstreamSource:
  case Adaptive
  case Multiway(opponentCount: Int)

/** Pure policy filter: belief-weighted penalties + soft veto + re-rank.
  *
  * No side effects — caller is responsible for storing the result.
  */
object StrategicOverlay:

  /** Belief-weighted aggression penalty coefficients (calibrate from benchmark data). */
  private val CallPenaltyCoeff = -0.1
  private val CheckPenaltyCoeff = -0.05

  def filter(input: OverlayInput): OverlayResult =
    val upstreamEvs = input.upstreamEvs
    if upstreamEvs.isEmpty then
      return OverlayResult(
        selectedAction = PokerAction.Fold,
        rankedActions = Vector.empty,
        softVetoed = Vector.empty,
        adjustments = Vector.empty,
        upstreamAction = PokerAction.Fold,
        upstreamSource = UpstreamSource.Adaptive
      )

    val upstreamAction = upstreamEvs.maxBy(_.expectedValue).action
    val potFraction = input.gameState.pot / math.max(input.gameState.stackSize, 1.0)

    // Step 1: Compute aggregate bluff mass across all rivals
    val bluffMass = aggregateBluffMass(input.rivalBeliefs)

    // Step 2: Apply belief-weighted penalty per action
    val adjustments = Vector.newBuilder[OverlayAdjustment]
    val adjusted = upstreamEvs.map { ae =>
      val penalty = beliefPenalty(ae.action, bluffMass, potFraction)
      val adjustedEv = ae.expectedValue + penalty
      if math.abs(penalty) > 1e-12 then
        adjustments += OverlayAdjustment(
          action = ae.action,
          originalEv = ae.expectedValue,
          adjustedEv = adjustedEv,
          reason = f"belief-penalty: bluffMass=$bluffMass%.3f potFrac=$potFraction%.3f"
        )
      ActionEvaluation(ae.action, adjustedEv)
    }

    // Step 3: Soft veto from robust lower bounds
    val epsilonTotal = input.config.epsilonBase + input.config.exploitConfig.epsilonAdapt
    val vetoed = Vector.newBuilder[(PokerAction, String)]
    input.robustLowerBounds.foreach { bounds =>
      var i = 0
      while i < math.min(bounds.length, adjusted.size) do
        if bounds(i) < -epsilonTotal then
          vetoed += ((adjusted(i).action, f"robust lower bound ${bounds(i)}%.4f < ${-epsilonTotal}%.4f"))
        i += 1
    }

    // Step 4: Re-rank by adjusted EV
    val ranked = adjusted.sortBy(-_.expectedValue)
    val vetoedResult = vetoed.result()
    val vetoedActions = vetoedResult.map(_._1).toSet

    // Step 5: Select best action (prefer non-vetoed, fallback to best robust lower bound)
    val selectedAction =
      ranked.find(ae => !vetoedActions.contains(ae.action)).map(_.action)
        .getOrElse {
          // All soft-vetoed: pick action with highest robust lower bound
          input.robustLowerBounds match
            case Some(bounds) if bounds.nonEmpty =>
              val bestIdx = bounds.indices.maxBy(bounds(_))
              if bestIdx < upstreamEvs.size then upstreamEvs(bestIdx).action
              else ranked.head.action
            case _ => ranked.head.action
        }

    OverlayResult(
      selectedAction = selectedAction,
      rankedActions = ranked,
      softVetoed = vetoedResult,
      adjustments = adjustments.result(),
      upstreamAction = upstreamAction,
      upstreamSource = UpstreamSource.Adaptive // caller overrides for multiway
    )

  /** Sum of P(Bluff) across all rival beliefs. */
  private[engine] def aggregateBluffMass(
      beliefs: Map[PlayerId, StrategicRivalBelief]
  ): Double =
    if beliefs.isEmpty then 0.0
    else beliefs.values.map(_.typePosterior.probabilityOf(StrategicClass.Bluff)).sum

  /** Penalty for passive actions against likely-aggressive opponents. */
  private[engine] def beliefPenalty(
      action: PokerAction,
      bluffMass: Double,
      potFraction: Double
  ): Double =
    val coeff = action match
      case PokerAction.Call => CallPenaltyCoeff
      case PokerAction.Check => CheckPenaltyCoeff
      case _ => 0.0
    bluffMass * coeff * potFraction
