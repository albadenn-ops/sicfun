package sicfun.holdem.types

/** Immutable snapshot of an in-progress hand's event log.
  *
  * Events are maintained in sorted order by `sequenceInHand`, regardless of the
  * order in which they were applied. The `appliedSequences` set enables O(1)
  * idempotency checks: exact duplicate deliveries are absorbed; conflicting
  * duplicates are rejected as errors.
  *
  * @param handId           unique hand identifier
  * @param events           events applied so far, sorted ascending by sequenceInHand
  * @param appliedSequences set of sequenceInHand values already applied (O(1) idempotency check)
  * @param lastUpdatedAt    maximum occurredAtEpochMillis seen across all applied events
  */
final case class HandState(
    handId: String,
    events: Vector[PokerEvent],
    appliedSequences: Set[Long],
    lastUpdatedAt: Long
):
  /** True when no events have been applied yet. */
  def isEmpty: Boolean = events.isEmpty

  /** Number of events applied to this hand so far. */
  def eventCount: Int = events.length

  /** The street of the most recently sequenced event, defaulting to Preflop. */
  def currentStreet: Street = events.lastOption.map(_.street).getOrElse(Street.Preflop)

  /** The board as of the most recently sequenced event, defaulting to empty. */
  def currentBoard: Board = events.lastOption.map(_.board).getOrElse(Board.empty)

  /** Set of all distinct player IDs that appear in at least one event. */
  def playerIds: Set[String] = events.map(_.playerId).toSet

  /** Return the most recent event (by sequence order) for the given player. */
  def latestEventFor(playerId: String): Option[PokerEvent] =
    events.findLast(_.playerId == playerId)

/** In-progress hand state manager that applies [[PokerEvent]]s idempotently.
  *
  * Events may arrive in any order (out-of-sequence delivery, retries, etc.).
  * [[HandEngine]] guarantees:
  *  - '''Idempotency''': re-applying an identical event is a no-op; re-applying the
  *    same `sequenceInHand` with a different payload raises an error.
  *  - '''Sorted event sequence''': the internal event vector is always kept in
  *    ascending `sequenceInHand` order via order-preserving insertion.
  *  - '''Player-level views''': [[toGameState]] derives a [[GameState]] snapshot
  *    from the player's most recent event.
  */
object HandEngine:

  /** Create an empty HandState for a new hand.
    *
    * @param handId    unique identifier for this hand
    * @param startedAt epoch millis used as the initial `lastUpdatedAt`
    */
  def newHand(handId: String, startedAt: Long = System.currentTimeMillis()): HandState =
    require(handId.trim.nonEmpty, "handId must be non-empty")
    HandState(handId, Vector.empty, Set.empty, startedAt)

  /** Apply a single event to the state.
    *
    * Idempotent: if the event's sequenceInHand was already applied with identical payload
    * the state is returned unchanged.
    * If the same sequenceInHand arrives with different payload, an error is raised.
    * Events from any delivery order are inserted so that `events` stays sorted by sequenceInHand.
    */
  def applyEvent(state: HandState, event: PokerEvent): HandState =
    require(
      event.handId == state.handId,
      s"event handId '${event.handId}' does not match state handId '${state.handId}'"
    )
    if state.appliedSequences.contains(event.sequenceInHand) then
      // Idempotency: allow exact duplicates, reject conflicting payloads
      state.events.find(_.sequenceInHand == event.sequenceInHand) match
        case Some(existing) if existing == event => state
        case Some(existing) =>
          throw new IllegalArgumentException(
            s"conflicting duplicate event for sequenceInHand=${event.sequenceInHand}; existing=$existing incoming=$event"
          )
        case None =>
          throw new IllegalStateException(
            s"inconsistent HandState: sequence ${event.sequenceInHand} marked applied but event is missing"
          )
    else
      // Insert at the correct position to maintain ascending sequenceInHand order
      val idx = state.events.indexWhere(_.sequenceInHand > event.sequenceInHand)
      val inserted =
        if idx < 0 then state.events :+ event        // append: largest sequence so far
        else state.events.patch(idx, Vector(event), 0) // insert before the first larger sequence
      val updatedAt =
        if state.events.isEmpty then event.occurredAtEpochMillis
        else math.max(state.lastUpdatedAt, event.occurredAtEpochMillis)
      state.copy(
        events = inserted,
        appliedSequences = state.appliedSequences + event.sequenceInHand,
        lastUpdatedAt = updatedAt
      )

  /** Apply a sequence of events.
    *
    * Exact duplicates are skipped (idempotent no-op). If a duplicate sequence carries
    * a different payload, [[applyEvent]] throws and the fold stops with that error.
    */
  def applyEvents(state: HandState, events: Seq[PokerEvent]): HandState =
    events.foldLeft(state)(applyEvent)

  /** Derive a [[GameState]] for a specific player from their most recent event.
    *
    * The returned state reflects the game situation ''as the player last saw it'' --
    * street, board, pot, amount to call, position, stack, and bet history.
    *
    * @param state    current hand state
    * @param playerId the player whose view to derive
    * @return `None` if no event for that player has been applied yet
    */
  def toGameState(state: HandState, playerId: String): Option[GameState] =
    state.latestEventFor(playerId).map { event =>
      GameState(
        street = event.street,
        board = event.board,
        pot = event.potBefore,
        toCall = event.toCall,
        position = event.position,
        stackSize = event.stackBefore,
        betHistory = event.betHistory
      )
    }
