package sicfun.holdem.runtime.protocol

import sicfun.holdem.types.{Board, Street}

/** Pure helpers shared by both heads-up protocol implementations.
  *
  * `AcpcActionCodec` (in [[AcpcMatchRunner]]) and [[SlumbotMatchRunner]] previously
  * carried byte-identical copies of these four helpers. Extracted into one place
  * so the algorithm has one home (and one set of pinning tests).
  */
private[holdem] object ProtocolStreetMath:

  /** Slice the full 5-card board down to the prefix appropriate for the given street.
    * Preflop -> empty, Flop -> first 3 cards, Turn -> first 4, River -> all 5.
    *
    * @throws IllegalArgumentException if the full board is shorter than the street requires.
    */
  def boardForStreet(fullBoard: Board, street: Street): Board =
    val expected = street.expectedBoardSize
    require(
      fullBoard.size >= expected,
      s"board has size ${fullBoard.size} but street $street requires at least $expected cards"
    )
    Board.from(fullBoard.cards.take(expected))

  /** Map a community-card count to its street index: 0 = preflop, 1 = flop, 2 = turn, 3 = river. */
  def streetIndexForBoard(board: Board): Int =
    board.size match
      case 0 => 0
      case 3 => 1
      case 4 => 2
      case 5 => 3
      case other => throw new IllegalArgumentException(s"unsupported board size: $other")

  /** Reverse of [[streetIndexForBoard]]: street-index 0..3 -> Street enum value. */
  def streetFromIndex(streetIdx: Int): Street =
    streetIdx match
      case 0 => Street.Preflop
      case 1 => Street.Flop
      case 2 => Street.Turn
      case 3 => Street.River
      case other => throw new IllegalArgumentException(s"invalid street index: $other")

  /** Map an absolute ACPC actor id to a hero-relative one: 0 if hero, 1 otherwise. */
  def relativeActorId(actualActor: Int, heroActual: Int): Int =
    if actualActor == heroActual then 0 else 1
