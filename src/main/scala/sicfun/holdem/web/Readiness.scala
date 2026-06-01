package sicfun.holdem.web

import com.sun.net.httpserver.{HttpExchange, HttpHandler}

import java.nio.file.Files
import java.util.concurrent.atomic.{AtomicBoolean, AtomicInteger}

import ujson.{Bool, Num, Obj, Str, Value}

import sicfun.holdem.web.AuthStack.{authenticationEnabled, authenticationMode}
import sicfun.holdem.web.HandHistoryReviewServer.ServerConfig
import sicfun.holdem.web.HandHistoryReviewServerApi.JsonResponse
import sicfun.holdem.web.HandHistoryReviewServerRuntime.healthModelSource
import sicfun.holdem.web.JobQueue.{AnalysisJobMetrics, AnalysisJobStore, JobStoreMetricsSource, PlayingHallJobStore}
import sicfun.holdem.web.RateLimit.rateLimitClientIpSource

private[web] object Readiness:
  private val ReadyReasonAcceptingTraffic = "accepting-traffic"
  private val ReadyReasonDraining = "draining"
  private val ReadyReasonTimedOutWorker = "timed-out-worker"
  private val ReadyReasonQueueFull = "queue-full"

  final case class ReadinessStatus(
      ready: Boolean,
      reason: String,
      draining: Boolean,
      acceptingAnalysisJobs: Boolean,
      drainSignalPresent: Boolean,
      timedOutWorkersInFlight: Int
  )

  def readinessStatus(
      config: ServerConfig,
      jobStore: AnalysisJobStore,
      playingHallJobStore: PlayingHallJobStore,
      draining: AtomicBoolean
  ): ReadinessStatus =
    val jobStores = Seq[JobStoreMetricsSource](jobStore, playingHallJobStore)
    val drainSignalPresent = config.drainSignalFile.exists(path => Files.exists(path))
    val drainingNow = draining.get() || drainSignalPresent || jobStore.isShuttingDown
    val timedOutWorkers = sumTimedOutWorkersInFlight(jobStores)
    val acceptingAnalysisJobs = !drainingNow && timedOutWorkers == 0 && jobStore.acceptingNewJobs
    val reason =
      if drainingNow then ReadyReasonDraining
      else if timedOutWorkers > 0 then ReadyReasonTimedOutWorker
      else if acceptingAnalysisJobs then ReadyReasonAcceptingTraffic
      else ReadyReasonQueueFull
    ReadinessStatus(
      ready = acceptingAnalysisJobs,
      reason = reason,
      draining = drainingNow,
      acceptingAnalysisJobs = acceptingAnalysisJobs,
      drainSignalPresent = drainSignalPresent,
      timedOutWorkersInFlight = timedOutWorkers
    )

  def renderHealth(
      config: ServerConfig,
      boundPort: Int,
      startedAtEpochMs: Long,
      jobStore: AnalysisJobStore,
      playingHallJobStore: PlayingHallJobStore,
      activeHttpRequests: Int,
      draining: AtomicBoolean,
      platformAuthService: Option[PlatformUserAuth.Service]
  ): JsonResponse =
    val metrics = jobStore.metrics
    val readiness = readinessStatus(config, jobStore, playingHallJobStore, draining)
    val otherActiveHttpRequests = math.max(0, activeHttpRequests - 1)
    val jobStores = Seq[JobStoreMetricsSource](jobStore, playingHallJobStore)
    // /api/health's operator-side superset = the shared readiness fields plus
    // these health-only extras (liveness `ok`, the `readyReason` rename of
    // `reason`, the user-store gauges, process uptime, model + upload config,
    // and the cross-store retained-jobs total). The shared block is the SAME
    // builder /api/ready consumes, so every overlapping field agrees by
    // construction rather than by two parallel hand-built objects.
    val healthOnly: Seq[(String, Value)] = Seq(
      "ok" -> Bool(true),
      "readyReason" -> Str(readiness.reason),
      // Surface the user-store cap so dashboards can show "users / maxUsers"
      // and alert when capacity is being approached. Only present when
      // platform-user auth is enabled (basic auth and no-auth modes have
      // no user store).
      "userAuthMaxUsers" -> config.platformAuth.map(c => Num(c.maxUsers.toDouble)).getOrElse(ujson.Null),
      "userAuthStoredUsers" -> platformAuthService.map(s => Num(s.storedUserCount.toDouble)).getOrElse(ujson.Null),
      "userAuthActiveSessions" -> platformAuthService.map(s => Num(s.activeSessionCount.toDouble)).getOrElse(ujson.Null),
      "userAuthPendingOidcFlows" -> platformAuthService.map(s => Num(s.pendingOidcFlows.toDouble)).getOrElse(ujson.Null),
      "startedAtEpochMs" -> Num(startedAtEpochMs.toDouble),
      "uptimeMs" -> Num((System.currentTimeMillis() - startedAtEpochMs).toDouble),
      "modelConfigured" -> Bool(config.serviceConfig.modelDir.nonEmpty),
      "modelSource" -> Str(healthModelSource(config.serviceConfig)),
      "maxUploadBytes" -> Num(config.maxUploadBytes.toDouble),
      // Sum both stores so the operator sees the true number of completed
      // jobs being held for status polling, not just the analysis half.
      "retainedTerminalJobs" -> Num(sumRetainedTerminalJobs(jobStores).toDouble)
    )
    JsonResponse(
      status = 200,
      value = Obj.from(
        sharedReadinessFields(config, boundPort, readiness, metrics, otherActiveHttpRequests) ++ healthOnly
      )
    )

  def renderReadiness(
      config: ServerConfig,
      boundPort: Int,
      jobStore: AnalysisJobStore,
      playingHallJobStore: PlayingHallJobStore,
      activeHttpRequests: Int,
      draining: AtomicBoolean
  ): JsonResponse =
    val metrics = jobStore.metrics
    val readiness = readinessStatus(config, jobStore, playingHallJobStore, draining)
    val otherActiveHttpRequests = math.max(0, activeHttpRequests - 1)
    // /api/ready is the orchestrator-facing strict subset: the shared readiness
    // fields plus the single rename `reason` (vs /api/health's `readyReason`).
    val readyOnly: Seq[(String, Value)] = Seq(
      "reason" -> Str(readiness.reason)
    )
    JsonResponse(
      status = if readiness.ready then 200 else 503,
      value = Obj.from(
        sharedReadinessFields(config, boundPort, readiness, metrics, otherActiveHttpRequests) ++ readyOnly
      )
    )

  /** The fields present, with the SAME name and value, on BOTH /api/health and
    * /api/ready. Both render methods consume this single builder so the
    * endpoints cannot drift: every shared field is computed exactly once from
    * the same `readiness` snapshot, `config`, and analysis-store `metrics`.
    * Each endpoint then layers on only its endpoint-specific fields (health: the
    * operator extras; ready: the `reason` rename). */
  private def sharedReadinessFields(
      config: ServerConfig,
      boundPort: Int,
      readiness: ReadinessStatus,
      metrics: AnalysisJobMetrics,
      otherActiveHttpRequests: Int
  ): Seq[(String, Value)] =
    Seq(
      "service" -> Str("hand-history-review"),
      "host" -> Str(config.host),
      "port" -> Num(boundPort.toDouble),
      "ready" -> Bool(readiness.ready),
      "draining" -> Bool(readiness.draining),
      "acceptingAnalysisJobs" -> Bool(readiness.acceptingAnalysisJobs),
      "authenticationEnabled" -> Bool(authenticationEnabled(config.basicAuth, config.platformAuth)),
      "authenticationMode" -> Str(authenticationMode(config.basicAuth, config.platformAuth)),
      "drainSignalConfigured" -> Bool(config.drainSignalFile.nonEmpty),
      "drainSignalPresent" -> Bool(readiness.drainSignalPresent),
      "analysisTimeoutMs" -> Num(config.analysisTimeoutMs.toDouble),
      "playingHallTimeoutMs" -> Num(config.playingHallTimeoutMs.toDouble),
      "rateLimitSubmitsPerMinute" -> Num(config.rateLimitSubmitsPerMinute.toDouble),
      "rateLimitStatusPerMinute" -> Num(config.rateLimitStatusPerMinute.toDouble),
      "rateLimitAuthPerMinute" -> Num(config.rateLimitAuthPerMinute.toDouble),
      "rateLimitClientIpSource" -> Str(rateLimitClientIpSource(config.rateLimitClientIpHeader, config.rateLimitTrustedProxyIps)),
      "activeHttpRequests" -> Num(otherActiveHttpRequests.toDouble),
      "maxConcurrentJobs" -> Num(metrics.maxConcurrentJobs.toDouble),
      "maxQueuedJobs" -> Num(metrics.maxQueuedJobs.toDouble),
      "queuedJobs" -> Num(metrics.queuedJobs.toDouble),
      "runningJobs" -> Num(metrics.runningJobs.toDouble),
      "timedOutWorkersInFlight" -> Num(readiness.timedOutWorkersInFlight.toDouble)
    )

  /** Aggregate the timed-out-in-flight worker count across every job store via
    * the shared [[JobStoreMetricsSource]] contract. A new job store type only
    * needs to be added to the `Seq` passed in, not summed in by hand here. */
  private def sumTimedOutWorkersInFlight(stores: Seq[JobStoreMetricsSource]): Int =
    stores.map(_.timedOutWorkersInFlightCount).sum

  /** Aggregate the retained-terminal-jobs count across every job store via the
    * shared [[JobStoreMetricsSource]] contract. */
  private def sumRetainedTerminalJobs(stores: Seq[JobStoreMetricsSource]): Int =
    stores.map(_.retainedTerminalJobsCount).sum

  def admissionRejectedMessage(
      readiness: ReadinessStatus,
      serviceName: String = "analysis"
  ): String =
    readiness.reason match
      case ReadyReasonDraining => s"$serviceName service is draining; try another instance or retry later"
      case ReadyReasonTimedOutWorker => s"$serviceName worker timed out; instance is waiting for recovery"
      case _ => s"$serviceName queue is full; try again later"

  def trackActiveRequests(
      activeHttpRequests: AtomicInteger,
      trustedClientIpHeader: Option[String],
      trustedProxyIps: Set[String],
      delegate: HttpHandler
  ): HttpHandler =
    new HttpHandler:
      override def handle(exchange: HttpExchange): Unit =
        // Stash the audit-display client address on the exchange before the
        // handler runs so AuthStack.remoteAddress (and any other audit code
        // path that wants to log "who connected") sees the same identity the
        // rate limiter keys on. Behind a trusted proxy, that's the
        // X-Forwarded-For IP (or whichever header is configured), not the
        // proxy's loopback peer.
        exchange.setAttribute(
          AuthStack.AuditClientAddressAttribute,
          RateLimit.resolveAuditClientAddress(exchange, trustedClientIpHeader, trustedProxyIps)
        )
        activeHttpRequests.incrementAndGet()
        try delegate.handle(exchange)
        finally activeHttpRequests.decrementAndGet()
