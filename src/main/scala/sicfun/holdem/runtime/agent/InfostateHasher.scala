package sicfun.holdem.runtime.agent

import sicfun.holdem.runtime.PlaceholderMarker
import sicfun.holdem.runtime.protocol.{SeatId, TableSnapshot}

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
