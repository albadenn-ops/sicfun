package sicfun.holdem.strategic

import sicfun.holdem.types.{Board, HoleCards, Street}
import sicfun.core.DiscreteDistribution

final case class PublicAction(
    actor: PlayerId,
    signal: ActionSignal
)

final case class PublicState(
    street: Street,
    board: Board,
    pot: Chips,
    stacks: TableMap[Chips],
    actionHistory: Vector[PublicAction]
):
  require(pot >= Chips(0.0), "pot must be non-negative")

trait RivalBeliefState:
  def update(signal: ActionSignal, publicState: PublicState): RivalBeliefState

final case class OpponentModelState(
    typeDistribution: DiscreteDistribution[String],
    beliefState: RivalBeliefState,
    attributedBaseline: Option[AttributedBaseline]
)

final case class OwnEvidence(
    globalSummary: Map[String, Double],
    perRivalSummary: Map[PlayerId, Map[String, Double]],
    relationalSummary: Map[(PlayerId, PlayerId), Map[String, Double]]
)

object OwnEvidence:
  val empty: OwnEvidence = OwnEvidence(
    globalSummary = Map.empty,
    perRivalSummary = Map.empty,
    relationalSummary = Map.empty
  )

final case class AugmentedState(
    publicState: PublicState,
    privateHand: HoleCards,
    opponents: RivalMap[OpponentModelState],
    ownEvidence: OwnEvidence
)

final case class OperativeBelief(
    distribution: DiscreteDistribution[AugmentedState]
)
