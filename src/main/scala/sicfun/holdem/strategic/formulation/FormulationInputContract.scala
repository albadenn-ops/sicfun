package sicfun.holdem.strategic.formulation

import sicfun.holdem.types.{GameState, HoleCards, PokerAction}
import sicfun.holdem.strategic.types.PlayerId
import sicfun.holdem.strategic.state.StrategicRivalBelief

/** Shared poker-facing data payload for both formulation paths. */
final case class FormulationSpot(
    gameState: GameState,
    candidateActions: Vector[PokerAction],
    heroValueInput: HeroValueInput,
    rivalBeliefs: Map[PlayerId, StrategicRivalBelief]
)

/** Hero private/value input. Replaces raw top-level heroBucket. */
sealed trait HeroValueInput

object HeroValueInput:
  /** Grounded target: exact hole cards for real equity computation. */
  final case class ExactHoleCards(cards: HoleCards) extends HeroValueInput

  /** A2 compatibility hook for the deprecated formulation path. */
  final case class StrengthHint(bucket: Int, source: String) extends HeroValueInput

/** Outcome kind at a formulation terminal node. */
enum FormulationTerminalKind:
  case Continue, HeroFold, RivalFold, Showdown

/** Action semantics for formulation-level reasoning. */
final case class FormulationActionSemantics(
    chipsCommitted: Double,
    potDeltaChips: Double,
    isAllIn: Boolean,
    terminal: FormulationTerminalKind,
    advancesStreet: Boolean
)
