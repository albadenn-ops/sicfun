package sicfun.holdem.strategic

import sicfun.holdem.types.{PokerAction, Street}

final case class Sizing(
    absolute: Chips,
    fractionOfPot: PotFraction
)

enum TimingBucket:
  case Fast, Normal, Slow, VeryLong

final case class RevealedHand(
    playerId: PlayerId,
    cards: Vector[sicfun.core.Card]
)

/** Size-aware public action signal (Def 5).
  *
  * Spec: Y_t^act = (a_t, lambda_t, tau_t).
  *
  * Encoding notes:
  *   - `sizing: Option[Sizing]` encodes lambda_t where None means non-sized action
  *     (fold, check, call have no sizing component).
  *   - `timing: Option[TimingBucket]` encodes tau_t where None means timing data
  *     is not available from the current observation source.
  *   - `stage: Street` is routing metadata from x^pub (not part of Y_t^act in the
  *     spec) carried here for kernel convenience. Authoritative source is PublicState.street.
  */
final case class ActionSignal(
    action: PokerAction.Category,
    sizing: Option[Sizing],
    timing: Option[TimingBucket],
    stage: Street
):
  inline def isAggressiveWager: Boolean = sizing.isDefined

final case class ShowdownSignal(
    revealedHands: Vector[RevealedHand]
)

final case class TotalSignal(
    actionSignal: ActionSignal,
    showdown: Option[ShowdownSignal]
):
  inline def actionChannel: ActionSignal = actionSignal
  inline def revelationChannel: Option[ShowdownSignal] = showdown
