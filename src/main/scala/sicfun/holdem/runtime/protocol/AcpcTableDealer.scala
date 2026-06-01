package sicfun.holdem.runtime.protocol

import sicfun.core.{Card, Deck, HandEvaluator, HandRank}
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
    val events = Vector(
      PostBlind(sb, config.smallBlind, SmallBlind),
      PostBlind(bb, config.bigBlind, BigBlind)
    )
    events.foreach(eventBuffer += _)
    events

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
    val dealtThisStreet = (1 to count).map { _ => deckBuf.remove(0) }.toVector
    dealtThisStreet.foreach(boardBuf += _)
    if count > 0 then eventBuffer += BettingRoundEvent.Deal(street, dealtThisStreet)
    boardBuf.toVector

  def currentBoard: Vector[Card] = boardBuf.toVector
  def allHoleCards: Map[SeatId, Vector[Card]] = hole.toMap

  def eventLog: Vector[BettingRoundEvent] = eventBuffer.toVector

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
    val totalPot = contributions.values.sum
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

  def evaluateShowdown(): Map[SeatId, HandRank] =
    require(boardBuf.size == 5, s"showdown requires full board, got ${boardBuf.size}")
    val nonFolded = (0 until config.numSeats).map(SeatId(_)).filterNot(folded.contains)
    nonFolded.map { seat =>
      val seven = hole(seat) ++ boardBuf.toVector
      seat -> HandEvaluator.evaluate7(seven)
    }.toMap

  def computeSidePots(): Vector[SidePot] =
    val contribPairs = (0 until config.numSeats).map(SeatId(_))
      .map(s => s -> contributions(s))
      .filter(_._2 > 0L)
      .sortBy(_._2)
      .toVector

    var result = Vector.empty[SidePot]
    var prevLevel = 0L
    var remaining = contribPairs

    while remaining.nonEmpty do
      val level = remaining.head._2
      val delta = level - prevLevel
      val potAmount = delta * remaining.size
      val eligible = remaining.map(_._1).toSet -- folded.toSet
      if potAmount > 0L && eligible.nonEmpty then
        result = result :+ SidePot(potAmount, eligible)
      prevLevel = level
      remaining = remaining.filter(_._2 > level)

    result

  def distributePots(
      ranks: Map[SeatId, HandRank]
  ): Vector[(SidePot, Map[SeatId, Long])] =
    val pots = computeSidePots()
    pots.map { pot =>
      val contenders = pot.eligibleSeats.toVector.filter(ranks.contains)
      if contenders.isEmpty then pot -> Map.empty[SeatId, Long]
      else
        val bestRank = contenders.map(ranks).max(using summon[Ordering[HandRank]])
        val winners = contenders.filter(s => ranks(s).compare(bestRank) == 0)
        val share = pot.amount / winners.size
        val remainder = pot.amount - share * winners.size
        val winnerOrderFromButton = clockwiseFromButton(winners.toSet)
        val baseDist = winners.map(_ -> share).toMap
        val oddChipSeat = winnerOrderFromButton.headOption
        val finalDist = oddChipSeat match
          case Some(s) if remainder > 0 =>
            baseDist.updated(s, baseDist(s) + remainder)
          case _ => baseDist
        pot -> finalDist
    }

  private def clockwiseFromButton(seats: Set[SeatId]): Vector[SeatId] =
    (1 to config.numSeats).map { off =>
      SeatId((_buttonSeat.index + off) % config.numSeats)
    }.filter(seats.contains).toVector

  /** True when the hand can no longer make progress in betting:
    * either only one (or zero) non-folded seat remains, or the river round
    * has closed and we are ready to finalize. */
  def handEnded: Boolean =
    val activeNonFolded = (0 until config.numSeats).map(SeatId(_))
      .count(s => !folded.contains(s))
    activeNonFolded <= 1 ||
      (currentStreet == Street.River && roundClosed)

  /** Finalize the hand: compute side pots, evaluate showdown (or assign
    * unopposed-fold ranks), distribute pots to winners, and emit
    * [[BettingRoundEvent.PotAwarded]] events. Returns a [[HandOutcome]] whose
    * `netChange` is required (by construction) to sum to zero — the A.1
    * chip-conservation invariant. */
  def finalizeHand(): HandOutcome =
    val nonFolded = (0 until config.numSeats).map(SeatId(_))
      .filterNot(folded.contains)
      .toVector

    val ranks: Map[SeatId, HandRank] =
      if nonFolded.size <= 1 then
        // Unopposed fold (or fully-folded edge case): assign topRank to the
        // sole survivor (if any) and bottomRank to no one else — folded seats
        // are excluded from `eligibleSeats` so they need no rank.
        nonFolded.headOption match
          case Some(winner) => Map(winner -> topRankForFinalize)
          case None         => Map.empty[SeatId, HandRank]
      else
        // Showdown path: complete the board if the hand short-circuited
        // (e.g., everyone went all-in pre-river) and evaluate.
        if boardBuf.size < 3 then dealCommunity(Street.Flop)
        if boardBuf.size < 4 then dealCommunity(Street.Turn)
        if boardBuf.size < 5 then dealCommunity(Street.River)
        evaluateShowdown()

    val distributed = distributePots(ranks)
    val distributedBySeat: Map[SeatId, Long] =
      distributed.flatMap(_._2).groupMapReduce(_._1)(_._2)(_ + _)
    val netChange: Map[SeatId, Long] =
      (0 until config.numSeats).map { i =>
        val s = SeatId(i)
        s -> (distributedBySeat.getOrElse(s, 0L) - contributions(s))
      }.toMap

    distributed.foreach { case (pot, dist) =>
      eventBuffer += BettingRoundEvent.PotAwarded(pot, dist)
    }

    HandOutcome(distributed, netChange, eventBuffer.toVector)

  /** Synthetic high rank used to mark the unopposed-fold winner. The actual
    * value never matters for distribution because there is at most one
    * contender per pot in that branch — but it must be a real, evaluable
    * [[HandRank]] so [[distributePots]] does not crash. */
  private def topRankForFinalize: HandRank =
    HandEvaluator.evaluate7(Deck.full.takeRight(7).toVector)

  // Test-only shims
  private[protocol] def setStateForTest(
      stacks: Map[SeatId, Long],
      contributions: Map[SeatId, Long],
      folded: Set[SeatId]
  ): Unit =
    stacks.foreach { case (s, v) => this.stacks(s) = v }
    contributions.foreach { case (s, v) => this.contributions(s) = v }
    folded.foreach(this.folded += _)

  private[protocol] def setStackForTest(seat: SeatId, value: Long): Unit =
    stacks(seat) = value

  private[protocol] def streetContributionForTest(seat: SeatId): Long =
    streetContribution(seat)

  private[protocol] def currentBetForTest: Long = currentBet

  private[protocol] def foldedSetForTest: Set[SeatId] = folded.toSet

  private[protocol] def scriptedSevenForTest(seat: SeatId): Vector[Card] =
    Deck.full.take(7).toVector

  private[protocol] def scriptedRanksSeat2WinsAll: Map[SeatId, HandRank] =
    val loser = HandEvaluator.evaluate7(Deck.full.take(7).toVector)
    val winner = HandEvaluator.evaluate7(Deck.full.drop(7).take(7).toVector)
    val (w, l) = if winner.compare(loser) > 0 then (winner, loser) else (loser, winner)
    Map(SeatId(0) -> l, SeatId(1) -> l, SeatId(2) -> w)
