package sicfun.holdem.runtime.protocol

import sicfun.core.Card
import sicfun.holdem.types.{PokerAction, Street}

opaque type SeatId = Int
object SeatId:
  def apply(i: Int): SeatId =
    require(i >= 0 && i < 9, s"SeatId must be in [0, 9), got $i")
    i
  extension (s: SeatId) def index: Int = s
  given Ordering[SeatId] = Ordering.Int

final case class TableConfig(
    numSeats: Int,
    smallBlind: Long,
    bigBlind: Long,
    ante: Long,
    startingStack: Long
):
  require(numSeats >= 2 && numSeats <= 9, s"numSeats must be in [2, 9], got $numSeats")
  require(smallBlind > 0L, s"smallBlind must be > 0, got $smallBlind")
  require(bigBlind >= smallBlind, s"bigBlind ($bigBlind) must be >= smallBlind ($smallBlind)")
  require(ante >= 0L, "ante must be non-negative")
  require(
    startingStack >= 10L * bigBlind,
    s"startingStack must be at least 10 bb, got $startingStack for bb=$bigBlind"
  )

final case class SidePot(amount: Long, eligibleSeats: Set[SeatId]):
  require(amount > 0L, s"SidePot amount must be positive, got $amount")
  require(eligibleSeats.nonEmpty, "SidePot must have at least one eligible seat")

enum BlindKind:
  case SmallBlind, BigBlind

enum BettingRoundEvent:
  case PostBlind(seat: SeatId, amount: Long, kind: BlindKind)
  case PostAnte(seat: SeatId, amount: Long)
  case Act(seat: SeatId, action: PokerAction)
  case Deal(street: Street, cards: Vector[Card])
  case Showdown(revealed: Map[SeatId, Vector[Card]])
  case PotAwarded(pot: SidePot, distribution: Map[SeatId, Long])

final case class HandOutcome(
    potsDistributed: Vector[(SidePot, Map[SeatId, Long])],
    netChange: Map[SeatId, Long],
    events: Vector[BettingRoundEvent]
):
  /** A.1 invariant: chip conservation. Sum of net changes must be zero. */
  require(
    netChange.values.sum == 0L,
    s"chip conservation violated: sum(netChange) = ${netChange.values.sum}, expected 0"
  )

final case class TableSnapshot(
    config: TableConfig,
    heroSeat: SeatId,
    holeCards: Vector[Card],
    board: Vector[Card],
    stacks: Map[SeatId, Long],
    contributions: Map[SeatId, Long],
    street: Street,
    actionHistory: Vector[BettingRoundEvent],
    buttonSeat: SeatId,
    activeSeats: Set[SeatId]
)
