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
