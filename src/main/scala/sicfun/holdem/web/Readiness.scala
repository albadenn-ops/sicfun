package sicfun.holdem.web

import com.sun.net.httpserver.{HttpExchange, HttpHandler}

import java.nio.file.Files
import java.util.concurrent.atomic.{AtomicBoolean, AtomicInteger}

import ujson.{Bool, Num, Obj, Str}

import sicfun.holdem.web.AuthStack.{authenticationEnabled, authenticationMode}
import sicfun.holdem.web.HandHistoryReviewServer.ServerConfig
import sicfun.holdem.web.HandHistoryReviewServerApi.JsonResponse
import sicfun.holdem.web.HandHistoryReviewServerRuntime.healthModelSource
import sicfun.holdem.web.JobQueue.{AnalysisJobStore, PlayingHallJobStore}
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
    val metrics = jobStore.metrics
    val drainSignalPresent = config.drainSignalFile.exists(path => Files.exists(path))
    val drainingNow = draining.get() || drainSignalPresent || jobStore.isShuttingDown
    val timedOutWorkers = metrics.timedOutWorkersInFlight + playingHallJobStore.timedOutWorkersInFlightCount
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
    JsonResponse(
      status = 200,
      value = Obj(
        "ok" -> Bool(true),
        "ready" -> Bool(readiness.ready),
        "readyReason" -> Str(readiness.reason),
        "draining" -> Bool(readiness.draining),
        "acceptingAnalysisJobs" -> Bool(readiness.acceptingAnalysisJobs),
        "authenticationEnabled" -> Bool(authenticationEnabled(config.basicAuth, config.platformAuth)),
        "authenticationMode" -> Str(authenticationMode(config.basicAuth, config.platformAuth)),
        // Surface the user-store cap so dashboards can show "users / maxUsers"
        // and alert when capacity is being approached. Only present when
        // platform-user auth is enabled (basic auth and no-auth modes have
        // no user store).
        "userAuthMaxUsers" -> config.platformAuth.map(c => Num(c.maxUsers.toDouble)).getOrElse(ujson.Null),
        "userAuthStoredUsers" -> platformAuthService.map(s => Num(s.storedUserCount.toDouble)).getOrElse(ujson.Null),
        "userAuthActiveSessions" -> platformAuthService.map(s => Num(s.activeSessionCount.toDouble)).getOrElse(ujson.Null),
        "service" -> Str("hand-history-review"),
        "host" -> Str(config.host),
        "port" -> Num(boundPort.toDouble),
        "startedAtEpochMs" -> Num(startedAtEpochMs.toDouble),
        "uptimeMs" -> Num((System.currentTimeMillis() - startedAtEpochMs).toDouble),
        "modelConfigured" -> Bool(config.serviceConfig.modelDir.nonEmpty),
        "modelSource" -> Str(healthModelSource(config.serviceConfig)),
        "drainSignalConfigured" -> Bool(config.drainSignalFile.nonEmpty),
        "drainSignalPresent" -> Bool(readiness.drainSignalPresent),
        "maxUploadBytes" -> Num(config.maxUploadBytes.toDouble),
        "analysisTimeoutMs" -> Num(config.analysisTimeoutMs.toDouble),
        "playingHallTimeoutMs" -> Num(config.playingHallTimeoutMs.toDouble),
        "rateLimitSubmitsPerMinute" -> Num(config.rateLimitSubmitsPerMinute.toDouble),
        "rateLimitStatusPerMinute" -> Num(config.rateLimitStatusPerMinute.toDouble),
        "rateLimitAuthPerMinute" -> Num(config.rateLimitAuthPerMinute.toDouble),
        "rateLimitClientIpSource" -> Str(rateLimitClientIpSource(config.rateLimitClientIpHeader, config.rateLimitTrustedProxyIps)),
        "maxConcurrentJobs" -> Num(metrics.maxConcurrentJobs.toDouble),
        "maxQueuedJobs" -> Num(metrics.maxQueuedJobs.toDouble),
        "activeHttpRequests" -> Num(otherActiveHttpRequests.toDouble),
        "queuedJobs" -> Num(metrics.queuedJobs.toDouble),
        "runningJobs" -> Num(metrics.runningJobs.toDouble),
        "timedOutWorkersInFlight" -> Num(readiness.timedOutWorkersInFlight.toDouble),
        // Sum both stores so the operator sees the true number of completed
        // jobs being held for status polling, not just the analysis half.
        "retainedTerminalJobs" -> Num((metrics.retainedTerminalJobs + playingHallJobStore.retainedTerminalJobsCount).toDouble)
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
    JsonResponse(
      status = if readiness.ready then 200 else 503,
      value = Obj(
        "service" -> Str("hand-history-review"),
        "host" -> Str(config.host),
        "port" -> Num(boundPort.toDouble),
        "ready" -> Bool(readiness.ready),
        "reason" -> Str(readiness.reason),
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
    )

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
