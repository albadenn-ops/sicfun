package sicfun.holdem.runtime.agent

import sicfun.holdem.runtime.PlaceholderMarker
import sicfun.holdem.runtime.protocol.*
import sicfun.holdem.types.PokerAction

final case class BuiltMdp(
    numStates: Int,
    robustLosses: Array[Array[Double]],
    qValues: Array[Double],
    transitions: (Int, Int, Int) => IndexedSeq[(Int, Double)],
    numProfiles: Int,
    terminalStates: Set[Int]
)

trait MdpEmbedding:
  def build(
      snapshot: TableSnapshot,
      actions: Vector[PokerAction],
      numProfiles: Int
  ): BuiltMdp

final class PlaceholderMdpEmbedding extends MdpEmbedding, PlaceholderMarker:
  val placeholderReason: String =
    "MdpEmbedding is a 2-state stub with hardcoded losses; wire ValueBridge + " +
      "OpponentModelBridge in A.2 before any benchmark claim."

  def build(
      snapshot: TableSnapshot,
      actions: Vector[PokerAction],
      numProfiles: Int
  ): BuiltMdp =
    val numStates = 2
    val terminalStates = Set(1)
    val robustLosses = Array.fill(numStates, actions.size)(0.0)
    (0 until actions.size).foreach { a =>
      robustLosses(0)(a) = estimateLossForAction(snapshot, actions(a))
    }
    val qValues = Array.tabulate(actions.size)(a => -robustLosses(0)(a))
    val transitions: (Int, Int, Int) => IndexedSeq[(Int, Double)] =
      (s, _, _) => if s == 0 then Vector(1 -> 1.0) else Vector(1 -> 1.0)
    BuiltMdp(numStates, robustLosses, qValues, transitions, numProfiles, terminalStates)

  private def estimateLossForAction(snapshot: TableSnapshot, action: PokerAction): Double =
    action match
      case PokerAction.Fold            => snapshot.contributions(snapshot.heroSeat).toDouble
      case PokerAction.Check           => 0.0
      case PokerAction.Call            => 0.5
      case PokerAction.Raise(amount)   => amount * 0.1
