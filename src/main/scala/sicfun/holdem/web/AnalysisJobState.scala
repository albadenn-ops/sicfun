package sicfun.holdem.web

import ujson.Value

/** Job-state ADT shared by [[HandHistoryReviewServer]]'s analysis and playing-hall
  * job stores.
  *
  * Carved out of HandHistoryReviewServer (F3 / B3 split slice). Both
  * `AnalysisJobStore` and `PlayingHallJobStore` use these same state cases for
  * their async-job lifecycle; pulling them into a sibling module unblocks any
  * future per-store extraction without dragging the shared ADT along.
  *
  * State diagram:
  *   Queued -> Running -> Completed
  *                     -> Failed
  *                     -> Cancelled (also reachable directly from Queued via drain)
  *
  *   `isTerminal` is true for Completed, Failed, Cancelled. Terminal states
  *   carry a `completedAtEpochMs`; non-terminal states return `None`.
  */
private[web] sealed trait AnalysisJobState:
  def status: String
  def submittedAtEpochMs: Long
  def startedAtEpochMs: Option[Long]
  def completedAtEpochMs: Option[Long]
  def isTerminal: Boolean

private[web] object AnalysisJobState:
  final case class Queued(submittedAtEpochMs: Long) extends AnalysisJobState:
    override val status = "queued"
    override val startedAtEpochMs = None
    override val completedAtEpochMs = None
    override val isTerminal = false

  final case class Running(submittedAtEpochMs: Long, startedAt: Long) extends AnalysisJobState:
    override val status = "running"
    override val startedAtEpochMs = Some(startedAt)
    override val completedAtEpochMs = None
    override val isTerminal = false

  final case class Completed(
      submittedAtEpochMs: Long,
      startedAt: Long,
      completedAt: Long,
      result: Value
  ) extends AnalysisJobState:
    override val status = "completed"
    override val startedAtEpochMs = Some(startedAt)
    override val completedAtEpochMs = Some(completedAt)
    override val isTerminal = true

  final case class Failed(
      submittedAtEpochMs: Long,
      startedAt: Long,
      completedAt: Long,
      errorStatus: Int,
      error: String
  ) extends AnalysisJobState:
    override val status = "failed"
    override val startedAtEpochMs = Some(startedAt)
    override val completedAtEpochMs = Some(completedAt)
    override val isTerminal = true

  final case class Cancelled(
      submittedAtEpochMs: Long,
      startedAt: Option[Long],
      completedAt: Long,
      result: Option[Value]
  ) extends AnalysisJobState:
    override val status = "cancelled"
    override val startedAtEpochMs = startedAt
    override val completedAtEpochMs = Some(completedAt)
    override val isTerminal = true
