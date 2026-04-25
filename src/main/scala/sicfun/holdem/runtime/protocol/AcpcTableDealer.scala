package sicfun.holdem.runtime.protocol

import sicfun.holdem.runtime.protocol.BettingRoundEvent.*
import sicfun.holdem.runtime.protocol.BlindKind.*
import scala.annotation.nowarn
import scala.util.Random

final class AcpcTableDealer(
    val config: TableConfig,
    initialButtonSeat: SeatId,
    rngSeed: Long
):
  // rng is wired now and used by later tasks (deal/shuffle); keep deterministic seed plumbing in place.
  @nowarn("msg=unused private member")
  private val rng = new Random(rngSeed)
  private var _buttonSeat: SeatId = initialButtonSeat
  private val stacks = collection.mutable.Map[SeatId, Long]()
  private val contributions = collection.mutable.Map[SeatId, Long]()
    .withDefaultValue(0L)
  for i <- 0 until config.numSeats do stacks(SeatId(i)) = config.startingStack

  def buttonSeat: SeatId = _buttonSeat

  def advanceButton(): Unit =
    _buttonSeat = SeatId((_buttonSeat.index + 1) % config.numSeats)

  private def nextSeat(s: SeatId): SeatId =
    SeatId((s.index + 1) % config.numSeats)

  def smallBlindSeat: SeatId =
    if config.numSeats == 2 then _buttonSeat else nextSeat(_buttonSeat)

  def bigBlindSeat: SeatId = nextSeat(smallBlindSeat)

  def postBlinds(): Vector[BettingRoundEvent] =
    val sb = smallBlindSeat
    val bb = bigBlindSeat
    stacks(sb) -= config.smallBlind
    stacks(bb) -= config.bigBlind
    contributions(sb) = contributions(sb) + config.smallBlind
    contributions(bb) = contributions(bb) + config.bigBlind
    Vector(
      PostBlind(sb, config.smallBlind, SmallBlind),
      PostBlind(bb, config.bigBlind, BigBlind)
    )

  def currentStacks: Map[SeatId, Long] = stacks.toMap
  def currentContributions: Map[SeatId, Long] = contributions.toMap
