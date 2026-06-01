package sicfun.holdem.runtime

import sicfun.holdem.runtime.agent.*
import sicfun.holdem.runtime.protocol.*
import sicfun.holdem.runtime.metrics.*
import sicfun.holdem.types.Street
import java.io.{File, PrintWriter}

final case class ChipConservationError(handNumber: Int, sumNetChange: Long)
    extends RuntimeException(
      s"chip conservation violated on hand $handNumber: sum(netChange) = $sumNetChange, expected 0"
    )

final case class MatchResult(
    handsPlayed: Int,
    netBySeat: Map[SeatId, Long],
    mbbPer100BySeat: Map[SeatId, Double],
    ci95BySeat: Map[SeatId, ConfidenceInterval],
    matchLogPath: String
)

final class NineMaxMatchRunner(
    tableConfig: TableConfig,
    agents: Vector[SeatAgent],
    numHands: Int,
    rngSeed: Long,
    matchId: String,
    strictNative: Boolean = true,
    benchmarkMode: Boolean = false
):
  require(
    agents.size == tableConfig.numSeats,
    s"agent count ${agents.size} must equal numSeats ${tableConfig.numSeats}"
  )

  def run(): MatchResult =
    NativeStrictMode.verify(
      new File("src/main/native/build"),
      NativeStrictMode.CoreLibraries,
      strict = strictNative
    ) match
      case Left(v)  => throw v
      case Right(_) => ()

    BenchmarkGate.check(agents, benchmarkMode) match
      case Left(v)  => throw v
      case Right(_) => ()

    val winningsBySeat = (0 until tableConfig.numSeats)
      .map(i => SeatId(i) -> scala.collection.mutable.ArrayBuffer.empty[Double])
      .toMap
    var currentButton = SeatId(0)
    val bb = tableConfig.bigBlind.toDouble
    val logFile = new File(s"data/matches/$matchId.jsonl")
    logFile.getParentFile.mkdirs()
    val log = new PrintWriter(logFile)

    try
      agents.foreach(_.onMatchStart(tableConfig))

      (1 to numHands).foreach { handNum =>
        val dealer = new AcpcTableDealer(tableConfig, currentButton, rngSeed + handNum.toLong)
        val outcome = playHand(dealer, agents)

        val conservationSum = outcome.netChange.values.sum
        if conservationSum != 0L then
          throw ChipConservationError(handNum, conservationSum)

        outcome.netChange.foreach { (seat, delta) =>
          winningsBySeat(seat) += delta.toDouble / bb
        }

        log.println(
          s"""{"hand":$handNum,"net":{${outcome.netChange.toVector.sortBy(_._1.index)
              .map((s, d) => s""""${s.index}":$d""").mkString(",")}}}"""
        )
        currentButton = SeatId((currentButton.index + 1) % tableConfig.numSeats)
      }

      val summary = MatchSummary(
        totalHands = numHands,
        netBySeat = winningsBySeat.map((s, w) => s -> (w.sum * bb).toLong),
        handsBySeat = winningsBySeat.map((s, w) => s -> w.size)
      )
      agents.foreach(_.onMatchEnd(summary))

      MatchResult(
        handsPlayed = numHands,
        netBySeat = summary.netBySeat,
        mbbPer100BySeat = winningsBySeat.map((s, w) => s -> MbbMetrics.mbbPer100(w.toVector)),
        ci95BySeat = winningsBySeat.map((s, w) =>
          s -> MbbMetrics.bootstrapIC95(w.toVector, iterations = 500, rngSeed = rngSeed)
        ),
        matchLogPath = logFile.getPath
      )
    finally log.close()

  private def playHand(dealer: AcpcTableDealer, agents: Vector[SeatAgent]): HandOutcome =
    val holeCards = dealer.dealHoleCards()
    agents.foreach { a =>
      a.onHandStart(buildSnapshot(dealer, a.seatId, holeCards))
    }
    dealer.postBlinds()
    playStreet(Street.Preflop, dealer, agents, holeCards)
    if !dealer.handEnded then
      dealer.dealCommunity(Street.Flop)
      playStreet(Street.Flop, dealer, agents, holeCards)
    if !dealer.handEnded then
      dealer.dealCommunity(Street.Turn)
      playStreet(Street.Turn, dealer, agents, holeCards)
    if !dealer.handEnded then
      dealer.dealCommunity(Street.River)
      playStreet(Street.River, dealer, agents, holeCards)
    val outcome = dealer.finalizeHand()
    agents.foreach { a =>
      a.onHandEnd(buildSnapshot(dealer, a.seatId, holeCards), outcome)
    }
    outcome

  private def playStreet(
      street: Street,
      dealer: AcpcTableDealer,
      agents: Vector[SeatAgent],
      holeCards: Map[SeatId, Vector[sicfun.core.Card]]
  ): Unit =
    dealer.startStreet(street)
    while !dealer.roundClosed do
      dealer.nextToAct match
        case None => ()
        case Some(seat) =>
          val agent = agents(seat.index)
          val snap = buildSnapshot(dealer, seat, holeCards)
          val legal = dealer.legalActionsFor(seat)
          val action = agent.decide(snap, legal)
          dealer.applyAction(seat, action) match
            case Left(err) =>
              throw new RuntimeException(s"agent $seat returned illegal action: $err")
            case Right(_) => ()

  private def buildSnapshot(
      dealer: AcpcTableDealer,
      heroSeat: SeatId,
      holeCards: Map[SeatId, Vector[sicfun.core.Card]]
  ): TableSnapshot =
    TableSnapshot(
      config = tableConfig,
      heroSeat = heroSeat,
      holeCards = holeCards.getOrElse(heroSeat, Vector.empty),
      board = dealer.currentBoard,
      stacks = dealer.currentStacks,
      contributions = dealer.currentContributions,
      street = dealer.streetOf,
      actionHistory = dealer.eventLog,
      buttonSeat = dealer.buttonSeat,
      activeSeats = (0 until tableConfig.numSeats).map(SeatId(_))
        .filter(s => dealer.currentStacks.getOrElse(s, 0L) >= 0L).toSet
    )
