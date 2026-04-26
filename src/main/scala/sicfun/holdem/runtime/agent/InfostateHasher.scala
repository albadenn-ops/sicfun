package sicfun.holdem.runtime.agent

import sicfun.holdem.runtime.protocol.{SeatId, TableSnapshot}

/** PlaceholderMarker is defined fully in Task 18. We forward-declare a minimal
  * stub here that Task 18 will move to its canonical location. Keep the trait
  * definition compatible: `def placeholderReason: String`. */
trait PlaceholderMarker:
  def placeholderReason: String

trait InfostateHasher:
  def hashFor(snapshot: TableSnapshot, seat: SeatId): Long

final class PlaceholderInfostateHasher extends InfostateHasher, PlaceholderMarker:
  val placeholderReason: String =
    "InfostateHasher is a naive field-hash; wire abstraction-aware hash in A.2"

  def hashFor(snapshot: TableSnapshot, seat: SeatId): Long =
    snapshot.street.ordinal * 31L +
      snapshot.holeCards.map(_.hashCode().toLong).sum * 17L +
      snapshot.board.map(_.hashCode().toLong).sum * 7L +
      snapshot.contributions.values.sum * 3L +
      seat.index
