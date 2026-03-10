package sicfun.holdem.types

/** Companion object for [[PokerEvent]] containing the schema version constant. */
object PokerEvent:
  /** Current event schema version used for serialization compatibility checks. */
  val SchemaVersion: String = "poker-event-v1"

/** An immutable record of a single player action with full decision-point context.
  *
  * Designed for event-sourced hand reconstruction: a complete hand history can be
  * rebuilt from an ordered sequence of `PokerEvent` instances sharing the same `handId`.
  *
  * Construction validates internal consistency:
  *   - Board size must match the street
  *   - Check requires `toCall == 0`; Call requires `toCall > 0`
  *   - Raise amount must be positive
  *   - All monetary and temporal fields must be non-negative
  *
  * @param handId                unique identifier for the hand this event belongs to
  * @param sequenceInHand        zero-based ordering index within the hand (monotonically increasing)
  * @param playerId              identifier of the acting player
  * @param occurredAtEpochMillis wall-clock timestamp when the action occurred (epoch millis)
  * @param street                the betting round during which the action occurred
  * @param position              the player's table position
  * @param board                 community cards visible at the time of the action
  * @param potBefore             pot size before this action
  * @param toCall                chips the player must contribute to call
  * @param stackBefore           player's stack before this action
  * @param action                the poker action taken
  * @param decisionTimeMillis    optional elapsed thinking time in milliseconds
  * @param betHistory            prior bet actions in the current hand, in chronological order
  */
final case class PokerEvent(
    handId: String,
    sequenceInHand: Long,
    playerId: String,
    occurredAtEpochMillis: Long,
    street: Street,
    position: Position,
    board: Board,
    potBefore: Double,
    toCall: Double,
    stackBefore: Double,
    action: PokerAction,
    decisionTimeMillis: Option[Long] = None,
    betHistory: Vector[BetAction] = Vector.empty
):
  require(handId.trim.nonEmpty, "handId must be non-empty")
  require(playerId.trim.nonEmpty, "playerId must be non-empty")
  require(sequenceInHand >= 0L, "sequenceInHand must be non-negative")
  require(occurredAtEpochMillis >= 0L, "occurredAtEpochMillis must be non-negative")
  require(potBefore >= 0.0, "potBefore must be non-negative")
  require(toCall >= 0.0, "toCall must be non-negative")
  require(stackBefore >= 0.0, "stackBefore must be non-negative")
  decisionTimeMillis.foreach { millis =>
    require(millis >= 0L, "decisionTimeMillis must be non-negative")
  }

  // Validate board size matches the expected card count for this street
  private val expectedBoard = street.expectedBoardSize
  require(
    board.size == expectedBoard,
    s"street $street expects board size $expectedBoard, got ${board.size}"
  )

  // Enforce action-specific invariants
  action match
    case PokerAction.Check =>
      require(toCall == 0.0, "check action requires toCall == 0")
    case PokerAction.Call =>
      require(toCall > 0.0, "call action requires toCall > 0")
    case PokerAction.Raise(amount) =>
      require(amount > 0.0, "raise amount must be positive")
    case PokerAction.Fold =>
      ()
