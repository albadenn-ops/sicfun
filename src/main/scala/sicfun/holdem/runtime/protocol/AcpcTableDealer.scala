package sicfun.holdem.runtime.protocol

import sicfun.core.{Card, Deck}
import sicfun.holdem.runtime.protocol.BettingRoundEvent.*
import sicfun.holdem.runtime.protocol.BlindKind.*
import sicfun.holdem.types.{PokerAction, Street}
import scala.util.Random

enum IllegalActionReason:
  case NotYourTurn(seat: SeatId, expected: Option[SeatId])
  case IllegalForm(seat: SeatId, action: PokerAction, reason: String)
  case InsufficientChips(seat: SeatId, required: Long, available: Long)

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

  private var currentStreet: Street = Street.Preflop
  private var currentBet: Long = 0L
  private var lastAggressor: Option[SeatId] = None
  private var actOrder: Vector[SeatId] = Vector.empty
  private var actIdx: Int = 0
  private val folded: collection.mutable.Set[SeatId] =
    collection.mutable.Set.empty
  private val allIn: collection.mutable.Set[SeatId] =
    collection.mutable.Set.empty
  private val streetContribution: collection.mutable.Map[SeatId, Long] =
    collection.mutable.Map.empty.withDefaultValue(0L)
  private val eventBuffer: collection.mutable.ArrayBuffer[BettingRoundEvent] =
    collection.mutable.ArrayBuffer.empty

  /** Build action order starting at `firstToAct`, cycling through all N seats,
    * then filter out folded/allIn. If `excludeAggressor` is set, also drops
    * that seat — used for post-raise rebuild where the raiser does not re-act
    * unless there is a further raise. */
  private def buildActOrder(
      firstToAct: SeatId,
      excludeAggressor: Option[SeatId] = None
  ): Vector[SeatId] =
    val cycle = (0 until config.numSeats).map { off =>
      SeatId((firstToAct.index + off) % config.numSeats)
    }.toVector
    cycle.filter { s =>
      !folded.contains(s) &&
        !allIn.contains(s) &&
        !excludeAggressor.contains(s)
    }

  def startStreet(street: Street): Unit =
    currentStreet = street
    streetContribution.clear()
    if street == Street.Preflop then
      currentBet = config.bigBlind
      streetContribution(smallBlindSeat) = config.smallBlind
      streetContribution(bigBlindSeat) = config.bigBlind
      lastAggressor = Some(bigBlindSeat)
      // Preflop: BB is the initial aggressor but keeps option — include BB.
      actOrder = buildActOrder(firstToAct = nextSeat(bigBlindSeat))
    else
      currentBet = 0L
      lastAggressor = None
      actOrder = buildActOrder(firstToAct = firstActivePostflop)
    actIdx = 0

  private def firstActivePostflop: SeatId =
    var s = nextSeat(_buttonSeat)
    while folded.contains(s) || allIn.contains(s) do s = nextSeat(s)
    s

  def nextToAct: Option[SeatId] =
    if roundClosed then None else actOrder.lift(actIdx)

  def streetOf: Street = currentStreet
  def aggressor: Option[SeatId] = lastAggressor

  def roundClosed: Boolean =
    val eligible = (0 until config.numSeats).map(SeatId(_))
      .filter(s => !folded.contains(s) && !allIn.contains(s))
    if eligible.size <= 1 then true
    else
      val allMatched = eligible.forall(s => streetContribution(s) == currentBet)
      allMatched && actIdx >= actOrder.size

  def applyAction(seat: SeatId, action: PokerAction): Either[IllegalActionReason, Unit] =
    if nextToAct != Some(seat) then
      Left(IllegalActionReason.NotYourTurn(seat, nextToAct))
    else action match
      case PokerAction.Fold =>
        folded += seat
        actIdx += 1
        eventBuffer += BettingRoundEvent.Act(seat, action)
        Right(())
      case PokerAction.Check =>
        if streetContribution(seat) != currentBet then
          Left(IllegalActionReason.IllegalForm(seat, action,
            s"cannot check facing a bet (currentBet=$currentBet, ownContribution=${streetContribution(seat)})"))
        else
          actIdx += 1
          eventBuffer += BettingRoundEvent.Act(seat, action)
          Right(())
      case PokerAction.Call =>
        val owed = currentBet - streetContribution(seat)
        if owed <= 0L then
          Left(IllegalActionReason.IllegalForm(seat, action, "nothing to call"))
        else
          val pay = math.min(owed, stacks(seat))
          stacks(seat) -= pay
          streetContribution(seat) = streetContribution(seat) + pay
          contributions(seat) = contributions(seat) + pay
          if stacks(seat) == 0L then allIn += seat
          actIdx += 1
          eventBuffer += BettingRoundEvent.Act(seat, action)
          Right(())
      case PokerAction.Raise(amountDouble) =>
        val target = amountDouble.toLong
        if target <= currentBet then
          Left(IllegalActionReason.IllegalForm(seat, action,
            s"raise $target must exceed currentBet $currentBet"))
        else
          val pay = target - streetContribution(seat)
          if pay > stacks(seat) then
            Left(IllegalActionReason.InsufficientChips(seat, pay, stacks(seat)))
          else
            stacks(seat) -= pay
            streetContribution(seat) = streetContribution(seat) + pay
            contributions(seat) = contributions(seat) + pay
            currentBet = target
            lastAggressor = Some(seat)
            if stacks(seat) == 0L then allIn += seat
            // Post-raise rebuild: exclude the raiser. Everyone else gets one turn.
            actOrder = buildActOrder(
              firstToAct = nextSeat(seat),
              excludeAggressor = Some(seat)
            )
            actIdx = 0
            eventBuffer += BettingRoundEvent.Act(seat, action)
            Right(())

  def legalActionsFor(seat: SeatId): Set[PokerAction] =
    val owed = currentBet - streetContribution(seat)
    val stack = stacks(seat)
    val contribution = streetContribution(seat)
    val totalPot = contributions.values.sum + streetContribution.values.sum
    val builder = scala.collection.mutable.Set[PokerAction]()

    builder += PokerAction.Fold
    if owed <= 0L then builder += PokerAction.Check
    if owed > 0L && stack > 0L then builder += PokerAction.Call

    val maxTotalBet = stack + contribution
    if maxTotalBet > currentBet then
      val halfPot = (totalPot / 2).max(currentBet + 1L)
      val pot = totalPot.max(currentBet + 1L)
      val allIn = maxTotalBet
      Seq(halfPot, pot, allIn)
        .map(_.min(maxTotalBet))
        .filter(_ > currentBet)
        .map(amt => PokerAction.Raise(amt.toDouble))
        .foreach(builder += _)

    builder.toSet
