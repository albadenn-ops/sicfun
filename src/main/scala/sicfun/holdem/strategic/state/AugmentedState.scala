package sicfun.holdem.strategic.state
import sicfun.holdem.strategic.types.*
import sicfun.holdem.strategic.safety.AttributedBaseline

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

/** Per-rival opponent model state (Def 12).
  *
  * @param typePosterior Posterior μ^{R,i} over strategic classes. In the
  *   finite-type model, the posterior IS sufficient for the latent type
  *   θ^{R,i} (Def 12 note), so no separate θ field is needed.
  * @param beliefState Rival's current belief embedding state
  * @param attributedBaseline Optional attributed baseline for this rival
  */
final case class OpponentModelState(
    typePosterior: DiscreteDistribution[String],
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

/** Operative belief b̃ ∈ Δ(X̃) (Def 11).
  *
  * In the spec, the operative belief is a joint distribution over augmented states.
  * In SICFUN's factored representation, per-rival posteriors are tracked independently
  * in [[OpponentModelState.typePosterior]] (via [[StrategicRivalBelief]]), not as a
  * joint distribution. The spec's b̃ ∈ Δ(X̃) is realized through the factored
  * per-rival belief map inside [[AugmentedState.opponents]], not through this wrapper
  * directly. This wrapper exists for the unfactored particle-filter representation
  * used in diagnostic and certification code paths.
  */
final case class OperativeBelief(
    distribution: DiscreteDistribution[AugmentedState]
)