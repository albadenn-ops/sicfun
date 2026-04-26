package sicfun.holdem.runtime.agent

import sicfun.holdem.runtime.protocol.*
import sicfun.holdem.types.PokerAction

trait SeatAgent:
  def seatId: SeatId
  def onMatchStart(tableConfig: TableConfig): Unit = ()
  def onHandStart(snapshot: TableSnapshot): Unit = ()
  def decide(snapshot: TableSnapshot, legalActions: Set[PokerAction]): PokerAction
  def onHandEnd(snapshot: TableSnapshot, outcome: HandOutcome): Unit = ()
  def onMatchEnd(summary: MatchSummary): Unit = ()

final case class MatchSummary(
    totalHands: Int,
    netBySeat: Map[SeatId, Long],
    handsBySeat: Map[SeatId, Int]
)
