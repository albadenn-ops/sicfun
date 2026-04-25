package sicfun.holdem.runtime.protocol

import sicfun.core.{Card, Deck}
import sicfun.holdem.runtime.protocol.BettingRoundEvent.*
import sicfun.holdem.runtime.protocol.BlindKind.*
import sicfun.holdem.types.Street
import scala.util.Random

final class AcpcTableDealer(
    val config: TableConfig,
    initialButtonSeat: SeatId,
    rngSeed: Long
):
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

  private val deckBuf: collection.mutable.ArrayBuffer[Card] =
    collection.mutable.ArrayBuffer.from(rng.shuffle(Deck.full))
  private val hole: collection.mutable.Map[SeatId, Vector[Card]] =
    collection.mutable.Map.empty
  private val boardBuf: collection.mutable.ArrayBuffer[Card] =
    collection.mutable.ArrayBuffer.empty

  def dealHoleCards(): Map[SeatId, Vector[Card]] =
    (0 until config.numSeats).foreach { i =>
      val seat = SeatId(i)
      val c1 = deckBuf.remove(0)
      val c2 = deckBuf.remove(0)
      hole(seat) = Vector(c1, c2)
    }
    hole.toMap

  def dealCommunity(street: Street): Vector[Card] =
    val count = street match
      case Street.Preflop => 0
      case Street.Flop    => 3
      case Street.Turn    => 1
      case Street.River   => 1
    (1 to count).foreach { _ => boardBuf += deckBuf.remove(0) }
    boardBuf.toVector

  def currentBoard: Vector[Card] = boardBuf.toVector
  def allHoleCards: Map[SeatId, Vector[Card]] = hole.toMap
