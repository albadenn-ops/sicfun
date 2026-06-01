package sicfun.holdem.web

import java.nio.charset.StandardCharsets
import java.util.UUID
import java.util.concurrent.{
  ArrayBlockingQueue,
  ConcurrentHashMap,
  Executors,
  RejectedExecutionException,
  ScheduledExecutorService,
  ScheduledFuture,
  ThreadFactory,
  ThreadPoolExecutor,
  TimeUnit
}
import java.util.concurrent.atomic.{AtomicBoolean, AtomicInteger}
import scala.util.control.NonFatal

import ujson.{Obj, Str, Value}

import sicfun.holdem.web.HandHistoryReviewServerApi.{JsonResponse, PlayingHallRequest, retryAfterSeconds}
import sicfun.holdem.web.HandHistoryReviewServerRuntime.{logInfo, logWarn}

/** Asynchronous job-processing machinery for [[HandHistoryReviewServer]].
  *
  * The two stores share the worker and timeout executors owned by
  * `HandHistoryReviewServer.startServer`; queue metrics intentionally reflect
  * that shared executor.
  */
private[web] object JobQueue:
  val AnalyzeJobPathPrefix = "/api/analyze-hand-history/jobs/"
  val PlayingHallPath = "/api/playing-hall"
  val PlayingHallJobPathPrefix = "/api/playing-hall/jobs/"
  private val DefaultPollAfterMs = 750
  private val CompletedJobRetentionMs = 15L * 60L * 1000L

  trait AnalysisBackend:
    def analyze(request: HandHistoryReviewService.AnalysisRequest): Either[String, Value]

  enum CancelOutcome:
    case Accepted, AlreadyTerminal, NotFound

  trait PlayingHallBackend:
    def run(request: PlayingHallRequest, cancelSignal: () => Boolean = () => false): Either[String, Value]


  /** Shared read-only metrics surface every job store exposes so readiness /
    * health aggregation can fold over a heterogeneous collection of stores
    * instead of hand-summing each store's counters at every call site. Both
    * [[AnalysisJobStore]] and [[PlayingHallJobStore]] implement it; adding a
    * third job store then only requires adding it to the aggregated `Seq` in
    * [[Readiness]], not editing every aggregation site. */
  trait JobStoreMetricsSource:
    /** Workers whose per-job timeout fired but whose thread has not yet exited. */
    def timedOutWorkersInFlightCount: Int
    /** Terminal jobs still retained in the store for status polling. */
    def retainedTerminalJobsCount: Int


  def newAnalysisExecutor(
      maxConcurrentJobs: Int,
      maxQueuedJobs: Int
  ): ThreadPoolExecutor =
    val executor = new ThreadPoolExecutor(
      maxConcurrentJobs,
      maxConcurrentJobs,
      0L,
      TimeUnit.MILLISECONDS,
      new ArrayBlockingQueue[Runnable](maxQueuedJobs),
      newThreadFactory("hand-review-analysis"),
      new ThreadPoolExecutor.AbortPolicy()
    )
    executor.prestartAllCoreThreads()
    executor

  def newTimeoutExecutor(): ScheduledExecutorService =
    Executors.newSingleThreadScheduledExecutor(newThreadFactory("hand-review-timeout"))

  def newThreadFactory(prefix: String): ThreadFactory =
    val nextId = new AtomicInteger(1)
    new ThreadFactory:
      override def newThread(runnable: Runnable): Thread =
        val thread = new Thread(runnable)
        thread.setName(s"$prefix-${nextId.getAndIncrement()}")
        thread.setDaemon(true)
        thread


  final case class AcceptedJob(
      jobId: String,
      submittedAtEpochMs: Long,
      statusUrl: String,
      pollAfterMs: Int
  )


  final case class AnalysisJobMetrics(
      maxConcurrentJobs: Int,
      maxQueuedJobs: Int,
      queuedJobs: Int,
      runningJobs: Int,
      timedOutWorkersInFlight: Int,
      retainedTerminalJobs: Int
  )


  sealed trait AnalysisJobState:
    def status: String
    def submittedAtEpochMs: Long
    def startedAtEpochMs: Option[Long]
    def completedAtEpochMs: Option[Long]
    def isTerminal: Boolean

  object AnalysisJobState:
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


  final class AnalysisJobStore(
      executor: ThreadPoolExecutor,
      timeoutExecutor: ScheduledExecutorService,
      backend: AnalysisBackend,
      analysisTimeoutMs: Long,
      nowMillis: () => Long = () => System.currentTimeMillis()
  ) extends JobStoreMetricsSource:
    import AnalysisJobState.*

    private val jobs = new ConcurrentHashMap[String, AnalysisJobState]()
    private val jobOwners = new ConcurrentHashMap[String, String]()
    private val timedOutWorkersInFlight = new AtomicInteger(0)

    def submit(
        request: HandHistoryReviewService.AnalysisRequest,
        ownerUserId: Option[String] = None,
        rejectIfUnavailable: () => Option[String] = () => None
    ): Either[String, AcceptedJob] =
      purgeExpiredJobs()
      rejectIfUnavailable() match
        case Some(error) =>
          // admissionRejectedMessage produces human-readable strings with
          // SPACES ("analysis service is draining; try another instance
          // or retry later"). Interpolated raw, those spaces split the
          // structured `key=value key=value` log shape so a log
          // aggregator tokenizing on whitespace orphans every subsequent
          // word. Same %20-escape pattern the startup banner and
          // heroName logging already use for the same reason.
          logWarn(
            s"job rejected unavailable queuedJobs=${executor.getQueue.size()} runningJobs=${executor.getActiveCount()} reason=${error.replace(" ", "%20")}"
          )
          Left(error)
        case None =>
          val jobId = UUID.randomUUID().toString
          val submittedAt = nowMillis()
          jobs.put(jobId, Queued(submittedAt))
          ownerUserId.foreach(owner => jobOwners.put(jobId, owner))
          try
            rejectIfUnavailable() match
              case Some(error) =>
                jobs.remove(jobId)
                jobOwners.remove(jobId)
                logWarn(
                  s"job rejected unavailable queuedJobs=${executor.getQueue.size()} runningJobs=${executor.getActiveCount()} reason=${error.replace(" ", "%20")}"
                )
                Left(error)
              case None =>
                executor.submit(new Runnable:
                  override def run(): Unit =
                    runJob(jobId, request, submittedAt)
                )
                logInfo(
                  s"job accepted jobId=$jobId queuedJobs=${executor.getQueue.size()} runningJobs=${executor.getActiveCount()} bytes=${request.handHistoryText.getBytes(StandardCharsets.UTF_8).length}"
                )
                Right(
                  AcceptedJob(
                    jobId = jobId,
                    submittedAtEpochMs = submittedAt,
                    statusUrl = s"$AnalyzeJobPathPrefix$jobId",
                    pollAfterMs = DefaultPollAfterMs
                  )
                )
          catch
            case _: RejectedExecutionException =>
              jobs.remove(jobId)
              jobOwners.remove(jobId)
              rejectIfUnavailable() match
                case Some(error) =>
                  // Same %20-escape as the two pre-submit reject sites above --
                  // admissionRejectedMessage strings contain spaces.
                  logWarn(
                    s"job rejected unavailable queuedJobs=${executor.getQueue.size()} runningJobs=${executor.getActiveCount()} reason=${error.replace(" ", "%20")}"
                  )
                  Left(error)
                case None =>
                  logWarn(
                    s"job rejected queue full queuedJobs=${executor.getQueue.size()} runningJobs=${executor.getActiveCount()} maxConcurrentJobs=${executor.getMaximumPoolSize} maxQueuedJobs=${queueCapacity(executor)}"
                  )
                  Left("analysis queue is full; try again later")

    def status(
        jobId: String,
        requesterUserId: Option[String] = None,
        enforceOwnership: Boolean = false
    ): Option[JsonResponse] =
      purgeExpiredJobs()
      val owner = Option(jobOwners.get(jobId))
      val accessible =
        if !enforceOwnership then true
        else owner.nonEmpty && requesterUserId.contains(owner.get)
      if !accessible then None
      else Option(jobs.get(jobId)).map(renderStatus(jobId, _))

    def timedOutWorkersInFlightCount: Int =
      timedOutWorkersInFlight.get()

    def retainedTerminalJobsCount: Int =
      purgeExpiredJobs()
      var count = 0
      val iterator = jobs.values().iterator()
      while iterator.hasNext do
        if iterator.next().isTerminal then count += 1
      count

    def metrics: AnalysisJobMetrics =
      // retainedTerminalJobsCount triggers the same purgeExpiredJobs() this
      // method used to run inline; the queued/running reads below are
      // unaffected by purge (it only evicts terminal jobs from the in-memory
      // map, never the executor queue), so the metrics value is identical.
      val retained = retainedTerminalJobsCount
      AnalysisJobMetrics(
        maxConcurrentJobs = executor.getMaximumPoolSize,
        maxQueuedJobs = queueCapacity(executor),
        queuedJobs = executor.getQueue.size(),
        runningJobs = executor.getActiveCount(),
        timedOutWorkersInFlight = timedOutWorkersInFlightCount,
        retainedTerminalJobs = retained
      )

    def acceptingNewJobs: Boolean =
      !isShuttingDown && executor.getQueue.remainingCapacity() > 0

    def isShuttingDown: Boolean =
      executor.isShutdown || executor.isTerminating || executor.isTerminated

    private def runJob(
        jobId: String,
        request: HandHistoryReviewService.AnalysisRequest,
        submittedAt: Long
    ): Unit =
      val startedAt = nowMillis()
      jobs.put(jobId, Running(submittedAt, startedAt))
      val timedOut = new AtomicBoolean(false)
      val timeoutTask = scheduleTimeout(jobId, submittedAt, startedAt, timedOut)
      // %20-escape spaces in heroName so a value like "Alice Smith" does
      // not split the structured key=value log fields. parseRequest's
      // 64-char cap + control-char rejection already bound the value;
      // this is the same shape AuthStack.formatSubmittedEmailForLog uses
      // for submitted emails.
      val loggedHeroName = request.heroName.getOrElse("-").replace(" ", "%20")
      logInfo(
        s"job started jobId=$jobId queuedJobs=${executor.getQueue.size()} runningJobs=${executor.getActiveCount()} heroName=$loggedHeroName site=${request.site.map(_.toString).getOrElse("auto")} timeoutMs=$analysisTimeoutMs"
      )
      val completedState =
        try
          val backendResult = backend.analyze(request)
          if timedOut.get() then timeoutFailure(submittedAt, startedAt)
          else
            backendResult match
              case Right(result) =>
                Completed(submittedAt, startedAt, nowMillis(), result)
              case Left(error) =>
                Failed(submittedAt, startedAt, nowMillis(), classifyAnalysisError(error), error)
        catch
          case _: InterruptedException if timedOut.get() =>
            timeoutFailure(submittedAt, startedAt)
          case NonFatal(e) =>
            if timedOut.get() then timeoutFailure(submittedAt, startedAt)
            else Failed(submittedAt, startedAt, nowMillis(), 500, s"analysis failed: ${e.getMessage}")
      timeoutTask.foreach(_.cancel(false))
      if timedOut.get() then
        Thread.interrupted()
      val finalState =
        if timedOut.get() then terminalFailureFor(jobId, submittedAt, startedAt)
        else
          jobs.put(jobId, completedState)
          completedState
      if timedOut.get() then
        timedOutWorkersInFlight.decrementAndGet()
      finalState match
        case Completed(_, _, completedAt, _) =>
          logInfo(
            s"job completed jobId=$jobId durationMs=${completedAt - startedAt} queuedJobs=${executor.getQueue.size()} runningJobs=${executor.getActiveCount()}"
          )
        case Failed(_, _, completedAt, errorStatus, error) =>
          // %20-escape spaces in the Failed.error value before it lands in
          // the structured log line. The value can be a backend-returned
          // message ('no hands found in upload'), a wrapped exception
          // ('analysis failed: <e.getMessage>'), or the timeoutFailure
          // string ('analysis timed out after 120000ms') -- all of which
          // contain spaces that would split the surrounding key=value
          // pairs when a log aggregator tokenizes on whitespace.
          logWarn(
            s"job failed jobId=$jobId durationMs=${completedAt - startedAt} errorStatus=$errorStatus queuedJobs=${executor.getQueue.size()} runningJobs=${executor.getActiveCount()} error=${error.replace(" ", "%20")}"
          )
        case _ => ()

    private def scheduleTimeout(
        jobId: String,
        submittedAt: Long,
        startedAt: Long,
        timedOut: AtomicBoolean
    ): Option[ScheduledFuture[?]] =
      if analysisTimeoutMs <= 0 then None
      else
        val workerThread = Thread.currentThread()
        Some(
          timeoutExecutor.schedule(
            new Runnable:
              override def run(): Unit =
                if tryMarkTimedOut(jobId, submittedAt, startedAt) then
                  timedOut.set(true)
                  timedOutWorkersInFlight.incrementAndGet()
                  logWarn(
                    s"job timed out jobId=$jobId timeoutMs=$analysisTimeoutMs queuedJobs=${executor.getQueue.size()} runningJobs=${executor.getActiveCount()}"
                  )
                  workerThread.interrupt()
            ,
            analysisTimeoutMs,
            TimeUnit.MILLISECONDS
          )
        )

    private def tryMarkTimedOut(
        jobId: String,
        submittedAt: Long,
        startedAt: Long
    ): Boolean =
      val timedOutState = timeoutFailure(submittedAt, startedAt)
      var marked = false
      var retry = true
      while retry do
        val current = jobs.get(jobId)
        if current == null || current.isTerminal then
          retry = false
        else if jobs.replace(jobId, current, timedOutState) then
          marked = true
          retry = false
      marked

    private def terminalFailureFor(
        jobId: String,
        submittedAt: Long,
        startedAt: Long
    ): Failed =
      tryMarkTimedOut(jobId, submittedAt, startedAt)
      jobs.get(jobId) match
        case failed: Failed => failed
        case _ => timeoutFailure(submittedAt, startedAt)

    private def timeoutFailure(
        submittedAt: Long,
        startedAt: Long
    ): Failed =
      Failed(
        submittedAtEpochMs = submittedAt,
        startedAt = startedAt,
        completedAt = nowMillis(),
        errorStatus = 504,
        error = s"analysis timed out after ${analysisTimeoutMs}ms"
      )

    private def renderStatus(jobId: String, state: AnalysisJobState): JsonResponse =
      state match
        case Queued(submittedAt) =>
          val json = baseStatus(jobId, state, submittedAt, None, None, Some(DefaultPollAfterMs))
          json("message") = Str("Queued for analysis")
          JsonResponse(
            status = 200,
            value = json,
            headers = Vector("Retry-After" -> retryAfterSeconds(DefaultPollAfterMs))
          )
        case Running(submittedAt, startedAt) =>
          val json = baseStatus(jobId, state, submittedAt, Some(startedAt), None, Some(DefaultPollAfterMs))
          json("message") = Str("Analysis in progress")
          JsonResponse(
            status = 200,
            value = json,
            headers = Vector("Retry-After" -> retryAfterSeconds(DefaultPollAfterMs))
          )
        case Completed(submittedAt, startedAt, completedAt, result) =>
          val json = baseStatus(jobId, state, submittedAt, Some(startedAt), Some(completedAt), None)
          json("durationMs") = ujson.Num((completedAt - startedAt).toDouble)
          json("result") = result
          JsonResponse(200, json)
        case Failed(submittedAt, startedAt, completedAt, errorStatus, error) =>
          val json = baseStatus(jobId, state, submittedAt, Some(startedAt), Some(completedAt), None)
          json("durationMs") = ujson.Num((completedAt - startedAt).toDouble)
          json("errorStatus") = ujson.Num(errorStatus)
          json("error") = Str(error)
          JsonResponse(200, json)
        case Cancelled(submittedAt, startedAt, completedAt, _) =>
          val json = baseStatus(jobId, state, submittedAt, startedAt, Some(completedAt), None)
          startedAt.foreach(s => json("durationMs") = ujson.Num((completedAt - s).toDouble))
          JsonResponse(200, json)

    private def baseStatus(
        jobId: String,
        state: AnalysisJobState,
        submittedAt: Long,
        startedAt: Option[Long],
        completedAt: Option[Long],
        pollAfterMs: Option[Int]
    ): Obj =
      val json = Obj(
        "jobId" -> Str(jobId),
        "status" -> Str(state.status),
        "statusUrl" -> Str(s"$AnalyzeJobPathPrefix$jobId"),
        "submittedAtEpochMs" -> ujson.Num(submittedAt.toDouble),
        "startedAtEpochMs" -> startedAt.map(value => ujson.Num(value.toDouble)).getOrElse(ujson.Null),
        "completedAtEpochMs" -> completedAt.map(value => ujson.Num(value.toDouble)).getOrElse(ujson.Null)
      )
      pollAfterMs.foreach(value => json("pollAfterMs") = ujson.Num(value))
      json

    private def purgeExpiredJobs(): Unit =
      val cutoff = nowMillis() - CompletedJobRetentionMs
      val iterator = jobs.entrySet().iterator()
      while iterator.hasNext do
        val entry = iterator.next()
        val state = entry.getValue
        if state.isTerminal && state.completedAtEpochMs.exists(_ < cutoff) then
          jobOwners.remove(entry.getKey)
          iterator.remove()


  final class PlayingHallJobStore(
      executor: ThreadPoolExecutor,
      timeoutExecutor: ScheduledExecutorService,
      backend: PlayingHallBackend,
      playingHallTimeoutMs: Long,
      nowMillis: () => Long = () => System.currentTimeMillis()
  ) extends JobStoreMetricsSource:
    import AnalysisJobState.*

    private val jobs = new ConcurrentHashMap[String, AnalysisJobState]()
    private val jobOwners = new ConcurrentHashMap[String, String]()
    private val cancelFlags = new ConcurrentHashMap[String, AtomicBoolean]()
    private val timedOutWorkersInFlight = new AtomicInteger(0)

    def submit(
        request: PlayingHallRequest,
        ownerUserId: Option[String] = None,
        rejectIfUnavailable: () => Option[String] = () => None
    ): Either[String, AcceptedJob] =
      purgeExpiredJobs()
      rejectIfUnavailable() match
        case Some(error) =>
          // Same %20-escape as the analysis path -- see AnalysisJobStore.submit.
          logWarn(
            s"playing hall job rejected unavailable queuedJobs=${executor.getQueue.size()} runningJobs=${executor.getActiveCount()} reason=${error.replace(" ", "%20")}"
          )
          Left(error)
        case None =>
          val jobId = UUID.randomUUID().toString
          val submittedAt = nowMillis()
          jobs.put(jobId, Queued(submittedAt))
          cancelFlags.put(jobId, new AtomicBoolean(false))
          ownerUserId.foreach(owner => jobOwners.put(jobId, owner))
          try
            rejectIfUnavailable() match
              case Some(error) =>
                jobs.remove(jobId)
                jobOwners.remove(jobId)
                cancelFlags.remove(jobId)
                logWarn(
                  s"playing hall job rejected unavailable queuedJobs=${executor.getQueue.size()} runningJobs=${executor.getActiveCount()} reason=${error.replace(" ", "%20")}"
                )
                Left(error)
              case None =>
                executor.submit(new Runnable:
                  override def run(): Unit =
                    runJob(jobId, request, submittedAt)
                )
                logInfo(
                  s"playing hall job accepted jobId=$jobId queuedJobs=${executor.getQueue.size()} runningJobs=${executor.getActiveCount()} ${request.logSummary}"
                )
                Right(
                  AcceptedJob(
                    jobId = jobId,
                    submittedAtEpochMs = submittedAt,
                    statusUrl = s"$PlayingHallJobPathPrefix$jobId",
                    pollAfterMs = DefaultPollAfterMs
                  )
                )
          catch
            case _: RejectedExecutionException =>
              jobs.remove(jobId)
              jobOwners.remove(jobId)
              cancelFlags.remove(jobId)
              rejectIfUnavailable() match
                case Some(error) =>
                  // Same %20-escape as the analysis path -- see AnalysisJobStore.submit.
                  logWarn(
                    s"playing hall job rejected unavailable queuedJobs=${executor.getQueue.size()} runningJobs=${executor.getActiveCount()} reason=${error.replace(" ", "%20")}"
                  )
                  Left(error)
                case None =>
                  logWarn(
                    s"playing hall job rejected queue full queuedJobs=${executor.getQueue.size()} runningJobs=${executor.getActiveCount()} maxConcurrentJobs=${executor.getMaximumPoolSize} maxQueuedJobs=${queueCapacity(executor)}"
                  )
                  Left("playing hall queue is full; try again later")

    def status(
        jobId: String,
        requesterUserId: Option[String] = None,
        enforceOwnership: Boolean = false
    ): Option[JsonResponse] =
      purgeExpiredJobs()
      val owner = Option(jobOwners.get(jobId))
      val accessible =
        if !enforceOwnership then true
        else owner.nonEmpty && requesterUserId.contains(owner.get)
      if !accessible then None
      else Option(jobs.get(jobId)).map(renderStatus(jobId, _))

    def cancel(
        jobId: String,
        requesterUserId: Option[String] = None,
        enforceOwnership: Boolean = false
    ): CancelOutcome =
      val owner = Option(jobOwners.get(jobId))
      val accessible =
        if !enforceOwnership then true
        else owner.nonEmpty && requesterUserId.contains(owner.get)
      Option(jobs.get(jobId)) match
        case None => CancelOutcome.NotFound
        case Some(_) if !accessible => CancelOutcome.NotFound
        case Some(state) if state.isTerminal => CancelOutcome.AlreadyTerminal
        case Some(_) =>
          Option(cancelFlags.get(jobId)).foreach(_.set(true))
          CancelOutcome.Accepted

    private def runJob(
        jobId: String,
        request: PlayingHallRequest,
        submittedAt: Long
    ): Unit =
      val startedAt = nowMillis()
      jobs.put(jobId, Running(submittedAt, startedAt))
      val timedOut = new AtomicBoolean(false)
      val cancelFlag = Option(cancelFlags.get(jobId)).getOrElse(new AtomicBoolean(false))
      val timeoutTask = scheduleTimeout(jobId, submittedAt, startedAt, timedOut)
      logInfo(
        s"playing hall job started jobId=$jobId queuedJobs=${executor.getQueue.size()} runningJobs=${executor.getActiveCount()} timeoutMs=$playingHallTimeoutMs ${request.logSummary}"
      )
      val completedState =
        try
          val backendResult = backend.run(request, () => cancelFlag.get())
          if timedOut.get() then timeoutFailure(submittedAt, startedAt)
          else
            backendResult match
              case Right(result) if cancelFlag.get() =>
                Cancelled(submittedAt, Some(startedAt), nowMillis(), Some(result))
              case Right(result) =>
                Completed(submittedAt, startedAt, nowMillis(), result)
              case Left(error) if cancelFlag.get() =>
                Cancelled(submittedAt, Some(startedAt), nowMillis(), None)
              case Left(error) =>
                Failed(submittedAt, startedAt, nowMillis(), classifyPlayingHallError(error), error)
        catch
          case _: InterruptedException if timedOut.get() =>
            timeoutFailure(submittedAt, startedAt)
          case NonFatal(e) =>
            if timedOut.get() then timeoutFailure(submittedAt, startedAt)
            else if cancelFlag.get() then Cancelled(submittedAt, Some(startedAt), nowMillis(), None)
            else Failed(submittedAt, startedAt, nowMillis(), 500, s"playing hall failed: ${e.getMessage}")
      timeoutTask.foreach(_.cancel(false))
      if timedOut.get() then
        Thread.interrupted()
      val finalState =
        if timedOut.get() then terminalFailureFor(jobId, submittedAt, startedAt)
        else
          jobs.put(jobId, completedState)
          completedState
      cancelFlags.remove(jobId)
      if timedOut.get() then
        timedOutWorkersInFlight.decrementAndGet()
      finalState match
        case Completed(_, _, completedAt, _) =>
          logInfo(
            s"playing hall job completed jobId=$jobId durationMs=${completedAt - startedAt} queuedJobs=${executor.getQueue.size()} runningJobs=${executor.getActiveCount()}"
          )
        case Failed(_, _, completedAt, errorStatus, error) =>
          // Same Failed.error %20-escape as the analyze branch above.
          logWarn(
            s"playing hall job failed jobId=$jobId durationMs=${completedAt - startedAt} errorStatus=$errorStatus queuedJobs=${executor.getQueue.size()} runningJobs=${executor.getActiveCount()} error=${error.replace(" ", "%20")}"
          )
        case Cancelled(_, _, completedAt, _) =>
          logInfo(
            s"playing hall job cancelled jobId=$jobId durationMs=${completedAt - startedAt} queuedJobs=${executor.getQueue.size()} runningJobs=${executor.getActiveCount()}"
          )
        case _ => ()

    private def scheduleTimeout(
        jobId: String,
        submittedAt: Long,
        startedAt: Long,
        timedOut: AtomicBoolean
    ): Option[ScheduledFuture[?]] =
      if playingHallTimeoutMs <= 0 then None
      else
        val workerThread = Thread.currentThread()
        Some(
          timeoutExecutor.schedule(
            new Runnable:
              override def run(): Unit =
                if tryMarkTimedOut(jobId, submittedAt, startedAt) then
                  timedOut.set(true)
                  cancelFlags.remove(jobId)
                  timedOutWorkersInFlight.incrementAndGet()
                  logWarn(
                    s"playing hall job timed out jobId=$jobId timeoutMs=$playingHallTimeoutMs queuedJobs=${executor.getQueue.size()} runningJobs=${executor.getActiveCount()}"
                  )
                  workerThread.interrupt()
            ,
            playingHallTimeoutMs,
            TimeUnit.MILLISECONDS
          )
        )

    private def tryMarkTimedOut(
        jobId: String,
        submittedAt: Long,
        startedAt: Long
    ): Boolean =
      val timedOutState = timeoutFailure(submittedAt, startedAt)
      var marked = false
      var retry = true
      while retry do
        val current = jobs.get(jobId)
        if current == null || current.isTerminal then
          retry = false
        else if jobs.replace(jobId, current, timedOutState) then
          marked = true
          retry = false
      marked

    private def timeoutFailure(submittedAt: Long, startedAt: Long): Failed =
      Failed(
        submittedAtEpochMs = submittedAt,
        startedAt = startedAt,
        completedAt = nowMillis(),
        errorStatus = 504,
        error = s"playing hall timed out after ${playingHallTimeoutMs}ms"
      )

    private def terminalFailureFor(
        jobId: String,
        submittedAt: Long,
        startedAt: Long
    ): Failed =
      tryMarkTimedOut(jobId, submittedAt, startedAt)
      jobs.get(jobId) match
        case failed: Failed => failed
        case _ => timeoutFailure(submittedAt, startedAt)

    def timedOutWorkersInFlightCount: Int =
      timedOutWorkersInFlight.get()

    def retainedTerminalJobsCount: Int =
      purgeExpiredJobs()
      var count = 0
      val iterator = jobs.values().iterator()
      while iterator.hasNext do
        if iterator.next().isTerminal then count += 1
      count

    private def renderStatus(jobId: String, state: AnalysisJobState): JsonResponse =
      state match
        case Queued(submittedAt) =>
          val json = baseStatus(jobId, state, submittedAt, None, None, Some(DefaultPollAfterMs))
          json("message") = Str("Queued for playing hall")
          JsonResponse(
            status = 200,
            value = json,
            headers = Vector("Retry-After" -> retryAfterSeconds(DefaultPollAfterMs))
          )
        case Running(submittedAt, startedAt) =>
          val json = baseStatus(jobId, state, submittedAt, Some(startedAt), None, Some(DefaultPollAfterMs))
          json("message") = Str("Playing hall run in progress")
          JsonResponse(
            status = 200,
            value = json,
            headers = Vector("Retry-After" -> retryAfterSeconds(DefaultPollAfterMs))
          )
        case Completed(submittedAt, startedAt, completedAt, result) =>
          val json = baseStatus(jobId, state, submittedAt, Some(startedAt), Some(completedAt), None)
          json("durationMs") = ujson.Num((completedAt - startedAt).toDouble)
          json("result") = result
          JsonResponse(200, json)
        case Failed(submittedAt, startedAt, completedAt, errorStatus, error) =>
          val json = baseStatus(jobId, state, submittedAt, Some(startedAt), Some(completedAt), None)
          json("durationMs") = ujson.Num((completedAt - startedAt).toDouble)
          json("errorStatus") = ujson.Num(errorStatus)
          json("error") = Str(error)
          JsonResponse(200, json)
        case Cancelled(submittedAt, startedAt, completedAt, result) =>
          val json = baseStatus(jobId, state, submittedAt, startedAt, Some(completedAt), None)
          startedAt.foreach(s => json("durationMs") = ujson.Num((completedAt - s).toDouble))
          result.foreach(r => json("result") = r)
          JsonResponse(200, json)

    private def baseStatus(
        jobId: String,
        state: AnalysisJobState,
        submittedAt: Long,
        startedAt: Option[Long],
        completedAt: Option[Long],
        pollAfterMs: Option[Int]
    ): Obj =
      val json = Obj(
        "jobId" -> Str(jobId),
        "status" -> Str(state.status),
        "statusUrl" -> Str(s"$PlayingHallJobPathPrefix$jobId"),
        "submittedAtEpochMs" -> ujson.Num(submittedAt.toDouble),
        "startedAtEpochMs" -> startedAt.map(value => ujson.Num(value.toDouble)).getOrElse(ujson.Null),
        "completedAtEpochMs" -> completedAt.map(value => ujson.Num(value.toDouble)).getOrElse(ujson.Null)
      )
      pollAfterMs.foreach(value => json("pollAfterMs") = ujson.Num(value))
      json

    private def purgeExpiredJobs(): Unit =
      val cutoff = nowMillis() - CompletedJobRetentionMs
      val iterator = jobs.entrySet().iterator()
      while iterator.hasNext do
        val entry = iterator.next()
        val state = entry.getValue
        if state.isTerminal && state.completedAtEpochMs.exists(_ < cutoff) then
          jobOwners.remove(entry.getKey)
          cancelFlags.remove(entry.getKey)
          iterator.remove()


  def classifyAnalysisError(error: String): Int =
    if error.startsWith("analysis timed out after") then 504
    else if error.startsWith("analysis failed:") then 500
    else 400

  def classifyPlayingHallError(error: String): Int =
    if error.startsWith("playing hall timed out after") then 504
    else if error.startsWith("playing hall failed:") then 500
    else 400


  private def queueCapacity(executor: ThreadPoolExecutor): Int =
    executor.getQueue.size() + executor.getQueue.remainingCapacity()
