package sicfun.holdem.strategic

import sicfun.holdem.types.Position

opaque type PlayerId = String
object PlayerId:
  inline def apply(id: String): PlayerId = id
  extension (pid: PlayerId)
    inline def value: String = pid

enum SeatStatus:
  case Active, Folded, AllIn, SittingOut

final case class Seat[+A](
    playerId: PlayerId,
    position: Position,
    status: SeatStatus,
    data: A
)

final case class TableMap[A](
    hero: PlayerId,
    seats: Vector[Seat[A]]
):
  require(
    seats.exists(_.playerId == hero),
    s"hero ${hero.value} must be present in seats"
  )
  require(
    seats.map(_.playerId).distinct.length == seats.length,
    "PlayerId must be unique across seats"
  )

  def toRivalMap: RivalMap[A] =
    RivalMap(seats.filter(_.playerId != hero))

  def activeRivalMap: RivalMap[A] =
    RivalMap(
      seats.filter(s =>
        s.playerId != hero &&
          (s.status == SeatStatus.Active || s.status == SeatStatus.AllIn)
      )
    )

final case class RivalMap[+A](rivals: Vector[Seat[A]]):
  require(rivals.nonEmpty, "RivalMap must contain at least one rival (L1)")
  require(
    rivals.map(_.playerId).distinct.length == rivals.length,
    "PlayerId must be unique across rivals"
  )

  def get(pid: PlayerId): Option[Seat[A]] =
    rivals.find(_.playerId == pid)

  def mapData[B](f: A => B): RivalMap[B] =
    RivalMap(rivals.map(s => s.copy(data = f(s.data))))

  inline def size: Int = rivals.length
