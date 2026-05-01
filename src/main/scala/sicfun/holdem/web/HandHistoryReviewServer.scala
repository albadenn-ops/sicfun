package sicfun.holdem.web

import sicfun.holdem.cli.CliHelpers
import sicfun.holdem.history.HandHistorySite
import sicfun.holdem.runtime.TexasHoldemPlayingHall

import com.sun.net.httpserver.{HttpExchange, HttpHandler, HttpServer}
import ujson.{Arr, Obj, Str, Value}

import java.io.ByteArrayOutputStream
import java.time.Instant
import java.net.{BindException, InetAddress, InetSocketAddress, URI, URLDecoder}
import java.security.MessageDigest
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import java.util.{Base64, Locale, UUID}
import java.util.concurrent.{
  ArrayBlockingQueue,
  ConcurrentHashMap,
  ExecutorService,
  Executors,
  RejectedExecutionException,
  ScheduledExecutorService,
  ScheduledFuture,
  ThreadFactory,
  ThreadPoolExecutor,
  TimeUnit
}
import java.util.concurrent.atomic.{AtomicBoolean, AtomicInteger, AtomicLong}
import scala.jdk.CollectionConverters.*
import scala.util.control.NonFatal
import sicfun.holdem.web.WebResponses.*

/** Embedded HTTP server for hand-history review analysis, auth, and static UI hosting.
  *
  * Built on `com.sun.net.httpserver` for zero external dependencies. Provides:
  *
  * '''API endpoints:'''
  *   - POST /api/analyze-hand-history — submit a hand history for async analysis
  *   - GET /api/analyze-hand-history/jobs/{id} — poll for analysis job status/result
  *   - GET/POST /api/auth/&#42; — local password auth, session management, and OIDC callbacks
  *   - GET /health — liveness probe (always 200)
  *   - GET /ready — readiness probe (200 when accepting traffic, 503 when draining/overloaded)
  *
  * '''Architecture:'''
  *   - Analysis jobs are submitted to an [[AnalysisJobStore]] backed by an ArrayBlockingQueue
  *     and processed by a configurable number of worker threads
  *   - Rate limiting uses per-client-IP sliding window counters (configurable via X-Forwarded-For)
  *   - Authentication supports Basic auth (for API clients), session cookies (for web UI),
  *     and optional OIDC providers (Google) via [[PlatformUserAuth]]
  *   - Static file serving for the web UI with Content-Security-Policy headers
  *   - Graceful shutdown: stops accepting new jobs, waits for in-flight analysis, then shuts down
  *
  * '''Design decisions:'''
  *   - Async job model avoids holding HTTP connections during long-running analysis
  *   - Completed jobs are retained for 15 minutes, then purged by a scheduled cleanup task
  *   - CSRF protection on session-authenticated mutation endpoints
  *   - All responses include security headers (CSP, X-Content-Type-Options, X-Frame-Options)
  *
  * @see [[HandHistoryReviewService]] for the core analysis logic
  * @see [[PlatformUserAuth]] for the authentication module
  */
object HandHistoryReviewServer:
  private val AnalyzeJobPathPrefix = "/api/analyze-hand-history/jobs/"
  private val PlayingHallPath = "/api/playing-hall"
  private val PlayingHallJobPathPrefix = "/api/playing-hall/jobs/"
  private val AuthMePath = "/api/auth/me"
  private val AuthRegisterPath = "/api/auth/register"
  private val AuthLoginPath = "/api/auth/login"
  private val AuthLogoutPath = "/api/auth/logout"
  private val AuthProfilePath = "/api/auth/profile"
  private val DefaultPollAfterMs = 750
  private val CompletedJobRetentionMs = 15L * 60L * 1000L
  private val DefaultAnalysisTimeoutMs = 120000L
  private val DefaultPlayingHallTimeoutMs = 900000L
  private val DefaultShutdownGraceMs = 5000L
  private val DefaultRateLimitWindowMs = 60L * 1000L
  private val DefaultRateLimitSubmitsPerMinute = 6
  private val DefaultRateLimitStatusPerMinute = 240
  private val ReadyReasonAcceptingTraffic = "accepting-traffic"
  private val ReadyReasonDraining = "draining"
  private val ReadyReasonTimedOutWorker = "timed-out-worker"
  private val ReadyReasonQueueFull = "queue-full"
  private val BasicAuthRealm = "sicfun-hand-history-review"
  private val BasicAuthChallenge = s"""Basic realm="$BasicAuthRealm", charset="UTF-8""""
  private val AuthenticationRequiredMessage = "authentication required"
  private val SessionAuthenticationRequiredMessage = "sign in required"
  private val SessionCsrfRequiredMessage = "missing or invalid csrf token"
  private val RateLimitClientIpSourceRemoteAddress = "remote-address"
  private val AuthenticatedUserAttribute = "sicfun.hand-history.authenticated-user"
  private val DefaultPlayingHallRoot = Paths.get("data", "web-playing-hall")
  private val DefaultPlayingHallHands = 240
  private val DefaultPlayingHallTableCount = 2
  private val DefaultPlayingHallPlayerCount = 6
  private val DefaultPlayingHallSeed = 42L
  private val MaxPlayingHallHands = 5000
  private val MaxPlayingHallTableCount = 24
  private val MaxPlayingHallBunchingTrials = 600
  private val MaxPlayingHallEquityTrials = 6000
  private val MaxPlayingHallVillainPoolEntries = 8

  final case class BasicAuthConfig(
      username: String,
      password: String
  )

  /** Server configuration with sensible defaults for local development.
    *
    * @param host                      bind address (default: 127.0.0.1 for local-only access)
    * @param port                      HTTP port (default: 8080)
    * @param staticDir                 directory for serving static UI files
    * @param maxUploadBytes            maximum request body size for hand history uploads
    * @param analysisTimeoutMs         per-analysis timeout before a job is marked failed
    * @param playingHallTimeoutMs      per-playing-hall timeout before a job is marked failed
    * @param maxConcurrentJobs         number of worker threads for analysis processing
    * @param maxQueuedJobs             maximum pending jobs before rejecting submissions
    * @param shutdownGraceMs           grace period for in-flight analysis on shutdown
    * @param rateLimitSubmitsPerMinute  max analysis submissions per client per minute
    * @param rateLimitStatusPerMinute   max status polls per client per minute
    * @param rateLimitClientIpHeader    optional header for client IP (e.g. X-Forwarded-For behind proxy)
    */
  final case class ServerConfig(
      host: String,
      port: Int,
      staticDir: Path,
      maxUploadBytes: Int,
      analysisTimeoutMs: Long,
      playingHallTimeoutMs: Long,
      maxConcurrentJobs: Int,
      maxQueuedJobs: Int,
      shutdownGraceMs: Long,
      rateLimitSubmitsPerMinute: Int,
      rateLimitStatusPerMinute: Int,
      rateLimitClientIpHeader: Option[String],
      rateLimitTrustedProxyIps: Set[String],
      drainSignalFile: Option[Path],
      basicAuth: Option[BasicAuthConfig],
      serviceConfig: HandHistoryReviewService.ServiceConfig,
      platformAuth: Option[PlatformUserAuth.Config] = None
  )

  private[web] trait AnalysisBackend:
    def analyze(request: HandHistoryReviewService.AnalysisRequest): Either[String, Value]

  private[web] enum CancelOutcome:
    case Accepted, AlreadyTerminal, NotFound

  private[web] trait PlayingHallBackend:
    def run(request: PlayingHallRequest, cancelSignal: () => Boolean = () => false): Either[String, Value]

  private val livePlayingHallBackend = new PlayingHallBackend:
    override def run(request: PlayingHallRequest, cancelSignal: () => Boolean): Either[String, Value] =
      runPlayingHall(request, cancelSignal)

  def main(args: Array[String]): Unit =
    start(args) match
      case Left(err) =>
        System.err.println(err)
        sys.exit(1)
      case Right(server) =>
        val binding = server.binding
        logInfo(s"hand-history review web server listening on http://${binding.host}:${binding.port}/")
        logInfo(s"serving static site from ${binding.staticDir.toAbsolutePath.normalize()}")
        logInfo(s"model source: ${binding.modelSource}")
        blockUntilShutdown()

  final case class ServerBinding(
      host: String,
      port: Int,
      staticDir: Path,
      modelSource: String
  )

  final class RunningServer(
      val binding: ServerBinding,
      shutdownFn: () => Unit
  ) extends AutoCloseable:
    override def close(): Unit = shutdownFn()

  def start(args: Array[String]): Either[String, RunningServer] =
    for
      config <- parseArgs(args)
      service <- HandHistoryReviewService.create(config.serviceConfig)
      analysisBackend = new AnalysisBackend:
        override def analyze(
            request: HandHistoryReviewService.AnalysisRequest
        ): Either[String, Value] =
          service.analyze(request).map(service.writeJson)
      running <- startWithBackends(config, analysisBackend, livePlayingHallBackend)
    yield running

  private[web] def startWithBackend(
      config: ServerConfig,
      backend: AnalysisBackend
  ): Either[String, RunningServer] =
    startWithBackends(config, backend, livePlayingHallBackend)

  private[web] def startWithBackends(
      config: ServerConfig,
      backend: AnalysisBackend,
      playingHallBackend: PlayingHallBackend
  ): Either[String, RunningServer] =
    startServer(config, backend, playingHallBackend)

  def run(args: Array[String]): Either[String, ServerBinding] =
    start(args).map(_.binding)

  private def startServer(
      config: ServerConfig,
      backend: AnalysisBackend,
      playingHallBackend: PlayingHallBackend
  ): Either[String, RunningServer] =
    val platformAuthServiceEither =
      config.platformAuth match
        case Some(authConfig) => PlatformUserAuth.Service.create(authConfig).map(Some(_))
        case None => Right(None)
    val serverExecutor = newServerExecutor()
    val analysisExecutor = newAnalysisExecutor(config.maxConcurrentJobs, config.maxQueuedJobs)
    val analysisTimeoutExecutor = newTimeoutExecutor()
    val rateLimiter = new RequestRateLimiter(
      submitsPerMinute = config.rateLimitSubmitsPerMinute,
      statusPerMinute = config.rateLimitStatusPerMinute,
      trustedClientIpHeader = config.rateLimitClientIpHeader,
      trustedProxyIps = config.rateLimitTrustedProxyIps
    )
    val startedAtEpochMs = System.currentTimeMillis()
    val draining = new AtomicBoolean(false)
    try
      val platformAuthService = platformAuthServiceEither match
        case Left(error) => throw new IllegalArgumentException(error)
        case Right(value) => value
      if isNonLoopbackBindHost(config.host) then
        config.platformAuth.filterNot(_.cookieSecure).foreach { _ =>
          logWarn(
            "platform-user auth is bound to a non-loopback host with USER_AUTH_COOKIE_SECURE=false; set --userAuthCookieSecure=true / USER_AUTH_COOKIE_SECURE=true when serving users through HTTPS"
          )
        }
      val jobStore = new AnalysisJobStore(
        executor = analysisExecutor,
        timeoutExecutor = analysisTimeoutExecutor,
        backend = backend,
        analysisTimeoutMs = config.analysisTimeoutMs
      )
      val playingHallJobStore = new PlayingHallJobStore(
        executor = analysisExecutor,
        timeoutExecutor = analysisTimeoutExecutor,
        backend = playingHallBackend,
        analysisTimeoutMs = config.playingHallTimeoutMs
      )
      val activeHttpRequests = new AtomicInteger(0)
      val server = HttpServer.create(new InetSocketAddress(config.host, config.port), 0)
      server.createContext(
        "/api/health",
        trackActiveRequests(
          activeHttpRequests,
          new JsonHandler(_ =>
            Right(
              renderHealth(
                config,
                server.getAddress.getPort,
                startedAtEpochMs,
                jobStore,
                playingHallJobStore,
                activeHttpRequests.get(),
                draining
              )
            )
          )
        )
      )
      server.createContext(
        "/api/ready",
        trackActiveRequests(
          activeHttpRequests,
          new JsonHandler(_ =>
            Right(
              renderReadiness(
                config,
                server.getAddress.getPort,
                jobStore,
                playingHallJobStore,
                activeHttpRequests.get(),
                draining
              )
            )
          )
        )
      )
      server.createContext(
        AnalyzeJobPathPrefix,
        trackActiveRequests(
          activeHttpRequests,
          new JsonHandler(
            exchange => handleAnalyzeJobStatus(exchange, jobStore, platformAuthService),
            basicAuth = config.basicAuth,
            platformAuth = platformAuthService,
            authRequirement = AuthRequirement.Required,
            rateLimiter = Some(rateLimiter),
            rateLimitBucket = Some(RateLimitBucket.JobStatus)
          )
        )
      )
      server.createContext(
        "/api/analyze-hand-history",
        trackActiveRequests(
          activeHttpRequests,
          new JsonHandler(
            exchange =>
              handleAnalyzeSubmit(
                exchange,
                jobStore,
                config.maxUploadBytes,
                () => readinessStatus(config, jobStore, playingHallJobStore, draining),
                platformAuthService
              ),
            basicAuth = config.basicAuth,
            platformAuth = platformAuthService,
            authRequirement = AuthRequirement.Required,
            rateLimiter = Some(rateLimiter),
            rateLimitBucket = Some(RateLimitBucket.Submit)
          )
        )
      )
      server.createContext(
        PlayingHallPath,
        trackActiveRequests(
          activeHttpRequests,
          new JsonHandler(
            exchange =>
              handlePlayingHallSubmit(
                exchange,
                playingHallJobStore,
                config.maxUploadBytes,
                () => readinessStatus(config, jobStore, playingHallJobStore, draining),
                platformAuthService
              ),
            basicAuth = config.basicAuth,
            platformAuth = platformAuthService,
            authRequirement = AuthRequirement.Required,
            rateLimiter = Some(rateLimiter),
            rateLimitBucket = Some(RateLimitBucket.Submit)
          )
        )
      )
      server.createContext(
        PlayingHallJobPathPrefix,
        trackActiveRequests(
          activeHttpRequests,
          new JsonHandler(
            exchange => handlePlayingHallJobStatus(exchange, playingHallJobStore, platformAuthService),
            basicAuth = config.basicAuth,
            platformAuth = platformAuthService,
            authRequirement = AuthRequirement.Required,
            rateLimiter = Some(rateLimiter),
            rateLimitBucket = Some(RateLimitBucket.JobStatus)
          )
        )
      )
      server.createContext(
        "/",
        trackActiveRequests(
          activeHttpRequests,
          new StaticHandler(config.staticDir, basicAuth = config.basicAuth, platformAuth = platformAuthService)
        )
      )
      server.createContext(
        AuthMePath,
        trackActiveRequests(
          activeHttpRequests,
          new JsonHandler(
            exchange => handleAuthMe(exchange, config.basicAuth, platformAuthService),
            basicAuth = config.basicAuth,
            platformAuth = platformAuthService,
            authRequirement = AuthRequirement.Optional
          )
        )
      )
      server.createContext(
        AuthRegisterPath,
        trackActiveRequests(
          activeHttpRequests,
          new JsonHandler(
            exchange => handleAuthRegister(exchange, platformAuthService),
            basicAuth = config.basicAuth,
            platformAuth = platformAuthService,
            authRequirement = AuthRequirement.Optional
          )
        )
      )
      server.createContext(
        AuthLoginPath,
        trackActiveRequests(
          activeHttpRequests,
          new JsonHandler(
            exchange => handleAuthLogin(exchange, platformAuthService),
            basicAuth = config.basicAuth,
            platformAuth = platformAuthService,
            authRequirement = AuthRequirement.Optional
          )
        )
      )
      server.createContext(
        AuthLogoutPath,
        trackActiveRequests(
          activeHttpRequests,
          new JsonHandler(
            exchange => handleAuthLogout(exchange, platformAuthService),
            basicAuth = config.basicAuth,
            platformAuth = platformAuthService,
            authRequirement = AuthRequirement.Required
          )
        )
      )
      server.createContext(
        AuthProfilePath,
        trackActiveRequests(
          activeHttpRequests,
          new JsonHandler(
            exchange => handleAuthProfile(exchange, platformAuthService),
            basicAuth = config.basicAuth,
            platformAuth = platformAuthService,
            authRequirement = AuthRequirement.Required
          )
        )
      )
      platformAuthService.foreach { service =>
        service.providerSummaries.flatMap(_.startPath).foreach { startPath =>
          server.createContext(
            startPath,
            trackActiveRequests(
              activeHttpRequests,
              new RedirectHandler(exchange => handleOidcStart(exchange, service))
            )
          )
        }
        config.platformAuth.toVector.flatMap(_.oidcProviders).foreach { provider =>
          server.createContext(
            provider.callbackPath,
            trackActiveRequests(
              activeHttpRequests,
              new RedirectHandler(exchange => handleOidcCallback(exchange, service, provider.id))
            )
          )
        }
      }
      server.setExecutor(serverExecutor)
      server.start()
      val binding = ServerBinding(
        host = config.host,
        port = server.getAddress.getPort,
        staticDir = config.staticDir,
        modelSource = modelSource(config.serviceConfig)
      )
      val closed = new AtomicBoolean(false)
      def shutdown(): Unit =
        if closed.compareAndSet(false, true) then
          draining.set(true)
          val shutdownStartedAt = System.nanoTime()
          val activeRequestCount = activeHttpRequests.get()
          val httpDrainSeconds =
            if activeRequestCount > 0 then shutdownDelaySeconds(config.shutdownGraceMs)
            else 0
          logInfo(
            s"shutdown requested host=${binding.host} port=${binding.port} activeHttpRequests=$activeRequestCount queuedJobs=${jobStore.metrics.queuedJobs} runningJobs=${jobStore.metrics.runningJobs} httpDrainSeconds=$httpDrainSeconds"
          )
          analysisExecutor.shutdown()
          analysisTimeoutExecutor.shutdown()
          try server.stop(httpDrainSeconds)
          finally
            val elapsedMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - shutdownStartedAt)
            val remainingGraceMs = math.max(0L, config.shutdownGraceMs - elapsedMs)
            shutdownExecutor("http", serverExecutor, remainingGraceMs)
            awaitExecutorDrain("analysis-timeout", analysisTimeoutExecutor, remainingGraceMs)
            awaitExecutorDrain("analysis", analysisExecutor, remainingGraceMs)
            logInfo(s"shutdown complete host=${binding.host} port=${binding.port}")
      sys.addShutdownHook(shutdown())
      logInfo(
        s"startup complete host=${binding.host} port=${binding.port} modelSource=${binding.modelSource} analysisTimeoutMs=${config.analysisTimeoutMs} maxConcurrentJobs=${config.maxConcurrentJobs} maxQueuedJobs=${config.maxQueuedJobs} rateLimitSubmitsPerMinute=${config.rateLimitSubmitsPerMinute} rateLimitStatusPerMinute=${config.rateLimitStatusPerMinute} rateLimitClientIpSource=${rateLimitClientIpSource(config.rateLimitClientIpHeader, config.rateLimitTrustedProxyIps)} rateLimitTrustedProxyIps=${trustedProxyIpSummary(config.rateLimitTrustedProxyIps)} drainSignalFile=${config.drainSignalFile.map(_.toAbsolutePath.normalize().toString).getOrElse("-")} authenticationMode=${authenticationMode(config.basicAuth, config.platformAuth)}"
      )
      Right(new RunningServer(binding, () => shutdown()))
    catch
      case e: BindException =>
        serverExecutor.shutdownNow()
        analysisExecutor.shutdownNow()
        analysisTimeoutExecutor.shutdownNow()
        val message = s"failed to start web server: ${config.host}:${config.port} is unavailable (${e.getMessage})"
        logError(message)
        Left(message)
      case e: Exception =>
        serverExecutor.shutdownNow()
        analysisExecutor.shutdownNow()
        analysisTimeoutExecutor.shutdownNow()
        val message = s"failed to start web server: ${e.getMessage}"
        logError(message)
        Left(message)

  private def newServerExecutor(): ExecutorService =
    val workerCount = math.max(4, Runtime.getRuntime.availableProcessors())
    Executors.newFixedThreadPool(workerCount, newThreadFactory("hand-review-http"))

  private def newAnalysisExecutor(
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

  private def newTimeoutExecutor(): ScheduledExecutorService =
    Executors.newSingleThreadScheduledExecutor(newThreadFactory("hand-review-timeout"))

  private def newThreadFactory(prefix: String): ThreadFactory =
    val nextId = new AtomicInteger(1)
    new ThreadFactory:
      override def newThread(runnable: Runnable): Thread =
        val thread = new Thread(runnable)
        thread.setName(s"$prefix-${nextId.getAndIncrement()}")
        thread.setDaemon(true)
        thread

  private def handleAnalyzeSubmit(
      exchange: HttpExchange,
      jobStore: AnalysisJobStore,
      maxUploadBytes: Int,
      readiness: () => ReadinessStatus,
      platformAuth: Option[PlatformUserAuth.Service]
  ): Either[(Int, String), JsonResponse] =
    if !exchange.getRequestMethod.equalsIgnoreCase("POST") then Left(405 -> "POST required")
    else if !ensurePlatformCsrf(exchange, platformAuth) then Left(403 -> SessionCsrfRequiredMessage)
    else if !readiness().acceptingAnalysisJobs then
      Left(503 -> admissionRejectedMessage(readiness()))
    else
      readRequestBody(exchange, maxUploadBytes)
        .flatMap(parseRequest)
        .flatMap(request =>
          jobStore
            .submit(
              request,
              ownerUserId = authenticatedUser(exchange).map(_.userId),
              rejectIfUnavailable = () =>
                if readiness().acceptingAnalysisJobs then None
                else Some(admissionRejectedMessage(readiness()))
            )
            .left
            .map(error => 503 -> error)
        )
        .map { accepted =>
          JsonResponse(
            status = 202,
            value = Obj(
              "jobId" -> Str(accepted.jobId),
              "status" -> Str("queued"),
              "statusUrl" -> Str(accepted.statusUrl),
              "submittedAtEpochMs" -> ujson.Num(accepted.submittedAtEpochMs.toDouble),
              "pollAfterMs" -> ujson.Num(accepted.pollAfterMs)
            ),
            headers = Vector(
              "Location" -> accepted.statusUrl,
              "Retry-After" -> retryAfterSeconds(accepted.pollAfterMs)
            )
          )
        }

  private def handlePlayingHallSubmit(
      exchange: HttpExchange,
      jobStore: PlayingHallJobStore,
      maxUploadBytes: Int,
      readiness: () => ReadinessStatus,
      platformAuth: Option[PlatformUserAuth.Service]
  ): Either[(Int, String), JsonResponse] =
    if !exchange.getRequestMethod.equalsIgnoreCase("POST") then Left(405 -> "POST required")
    else if !ensurePlatformCsrf(exchange, platformAuth) then Left(403 -> SessionCsrfRequiredMessage)
    else if !readiness().acceptingAnalysisJobs then
      Left(503 -> admissionRejectedMessage(readiness(), "playing hall"))
    else
      readRequestBody(exchange, maxUploadBytes)
        .flatMap(parsePlayingHallRequest)
        .flatMap(request =>
          jobStore
            .submit(
              request,
              ownerUserId = authenticatedUser(exchange).map(_.userId),
              rejectIfUnavailable = () =>
                if readiness().acceptingAnalysisJobs then None
                else Some(admissionRejectedMessage(readiness(), "playing hall"))
            )
            .left
            .map(error => 503 -> error)
        )
        .map(renderAcceptedJobResponse)

  private def renderAcceptedJobResponse(accepted: AcceptedJob): JsonResponse =
    JsonResponse(
      status = 202,
      value = Obj(
        "jobId" -> Str(accepted.jobId),
        "status" -> Str("queued"),
        "statusUrl" -> Str(accepted.statusUrl),
        "submittedAtEpochMs" -> ujson.Num(accepted.submittedAtEpochMs.toDouble),
        "pollAfterMs" -> ujson.Num(accepted.pollAfterMs)
      ),
      headers = Vector(
        "Location" -> accepted.statusUrl,
        "Retry-After" -> retryAfterSeconds(accepted.pollAfterMs)
      )
    )

  private def admissionRejectedMessage(
      readiness: ReadinessStatus,
      serviceName: String = "analysis"
  ): String =
    readiness.reason match
      case ReadyReasonDraining => s"$serviceName service is draining; try another instance or retry later"
      case ReadyReasonTimedOutWorker => s"$serviceName worker timed out; instance is waiting for recovery"
      case _ => s"$serviceName queue is full; try again later"

  private def handleAnalyzeJobStatus(
      exchange: HttpExchange,
      jobStore: AnalysisJobStore,
      platformAuth: Option[PlatformUserAuth.Service]
  ): Either[(Int, String), JsonResponse] =
    if !exchange.getRequestMethod.equalsIgnoreCase("GET") then Left(405 -> "GET required")
    else
      extractJobId(exchange, AnalyzeJobPathPrefix, "analysis").flatMap { jobId =>
        jobStore
          .status(
            jobId = jobId,
            requesterUserId = authenticatedUser(exchange).map(_.userId),
            enforceOwnership = platformAuth.nonEmpty
          )
          .toRight(404 -> s"analysis job not found: $jobId")
      }

  private def handlePlayingHallJobStatus(
      exchange: HttpExchange,
      jobStore: PlayingHallJobStore,
      platformAuth: Option[PlatformUserAuth.Service]
  ): Either[(Int, String), JsonResponse] =
    val method = exchange.getRequestMethod
    if method.equalsIgnoreCase("GET") then
      extractJobId(exchange, PlayingHallJobPathPrefix, "playing hall").flatMap { jobId =>
        jobStore
          .status(
            jobId = jobId,
            requesterUserId = authenticatedUser(exchange).map(_.userId),
            enforceOwnership = platformAuth.nonEmpty
          )
          .toRight(404 -> s"playing hall job not found: $jobId")
      }
    else if method.equalsIgnoreCase("DELETE") then
      extractJobId(exchange, PlayingHallJobPathPrefix, "playing hall").flatMap { jobId =>
        jobStore.cancel(
          jobId = jobId,
          requesterUserId = authenticatedUser(exchange).map(_.userId),
          enforceOwnership = platformAuth.nonEmpty
        ) match
          case CancelOutcome.Accepted =>
            Right(JsonResponse(200, Obj("jobId" -> Str(jobId), "status" -> Str("cancelled"))))
          case CancelOutcome.AlreadyTerminal =>
            Left(409 -> "playing hall job already terminal")
          case CancelOutcome.NotFound =>
            Left(404 -> s"playing hall job not found: $jobId")
      }
    else Left(405 -> "GET or DELETE required")

  private def extractJobId(
      exchange: HttpExchange,
      pathPrefix: String,
      label: String
  ): Either[(Int, String), String] =
    val path = Option(exchange.getRequestURI.getPath).getOrElse("")
    if !path.startsWith(pathPrefix) then Left(404 -> "not found")
    else
      val jobId = path.substring(pathPrefix.length).trim
      Either.cond(jobId.nonEmpty && !jobId.contains("/"), jobId, 400 -> s"$label job id is required")

  private def parseRequest(body: String): Either[(Int, String), HandHistoryReviewService.AnalysisRequest] =
    try
      val json = ujson.read(body)
      val obj = json.obj
      val handHistoryText = requiredString(obj, "handHistoryText").map(_.trim)
      val heroName = optionalString(obj, "heroName").map(_.trim).filter(_.nonEmpty)
      val site = parseOptionalSite(optionalString(obj, "site"))
      for
        text <- handHistoryText
        parsedSite <- site
      yield HandHistoryReviewService.AnalysisRequest(
        handHistoryText = text,
        site = parsedSite,
        heroName = heroName
      )
    catch
      case NonFatal(e) => Left(400 -> s"invalid JSON request: ${e.getMessage}")

  private def parsePlayingHallRequest(body: String): Either[(Int, String), PlayingHallRequest] =
    try
      val obj = ujson.read(body).obj
      for
        hands <- requiredIntInRange(
          obj,
          key = "hands",
          default = DefaultPlayingHallHands,
          min = 1,
          max = MaxPlayingHallHands
        )
        tableCount <- requiredIntInRange(
          obj,
          key = "tableCount",
          default = DefaultPlayingHallTableCount,
          min = 1,
          max = MaxPlayingHallTableCount
        )
        playerCount <- requiredIntInRange(
          obj,
          key = "playerCount",
          default = DefaultPlayingHallPlayerCount,
          min = 2,
          max = 9
        )
        heroStyle <- requiredChoice(
          obj,
          key = "heroStyle",
          default = "adaptive",
          allowed = Set("adaptive", "gto", "strategic")
        )
        heroPosition <- requiredChoice(
          obj,
          key = "heroPosition",
          default = "Button",
          allowed = Set("SmallBlind", "BigBlind", "UTG", "UTG1", "UTG2", "Middle", "Hijack", "Cutoff", "Button")
        )
        gtoMode <- requiredChoice(
          obj,
          key = "gtoMode",
          default = "exact",
          allowed = Set("fast", "exact")
        )
        villainPool <- requiredVillainPool(obj)
        heroExplorationRate <- requiredDoubleInRange(
          obj,
          key = "heroExplorationRate",
          default = 0.0,
          min = 0.0,
          max = 1.0
        )
        raiseSize <- requiredDoubleInRange(
          obj,
          key = "raiseSize",
          default = 2.5,
          min = 0.25,
          max = 20.0
        )
        bunchingTrials <- requiredIntInRange(
          obj,
          key = "bunchingTrials",
          default = 40,
          min = 1,
          max = MaxPlayingHallBunchingTrials
        )
        equityTrials <- requiredIntInRange(
          obj,
          key = "equityTrials",
          default = 240,
          min = 1,
          max = MaxPlayingHallEquityTrials
        )
        learnEveryHands <- requiredIntInRange(
          obj,
          key = "learnEveryHands",
          default = 0,
          min = 0,
          max = MaxPlayingHallHands
        )
        learningWindowSamples <- requiredIntInRange(
          obj,
          key = "learningWindowSamples",
          default = 200,
          min = 0,
          max = 500000
        )
        seed <- optionalLong(obj, "seed").getOrElse(Right(DefaultPlayingHallSeed))
        saveReviewHandHistory <- optionalBoolean(obj, "saveReviewHandHistory").getOrElse(Right(false))
        fullRing <- optionalBoolean(obj, "fullRing").getOrElse(Right(false))
      yield PlayingHallRequest(
        hands = hands,
        tableCount = tableCount,
        playerCount = playerCount,
        heroStyle = heroStyle,
        heroPosition = heroPosition,
        gtoMode = gtoMode,
        villainPool = villainPool,
        heroExplorationRate = heroExplorationRate,
        raiseSize = raiseSize,
        bunchingTrials = bunchingTrials,
        equityTrials = equityTrials,
        learnEveryHands = learnEveryHands,
        learningWindowSamples = learningWindowSamples,
        saveReviewHandHistory = saveReviewHandHistory,
        fullRing = fullRing,
        seed = seed
      )
    catch
      case NonFatal(e) => Left(400 -> s"invalid JSON request: ${e.getMessage}")

  private def requiredIntInRange(
      obj: collection.Map[String, Value],
      key: String,
      default: Int,
      min: Int,
      max: Int
  ): Either[(Int, String), Int] =
    val value = optionalInt(obj, key).getOrElse(Right(default))
    value.flatMap(parsed =>
      Either.cond(parsed >= min && parsed <= max, parsed, 400 -> s"$key must be in [$min,$max]")
    )

  private def requiredDoubleInRange(
      obj: collection.Map[String, Value],
      key: String,
      default: Double,
      min: Double,
      max: Double
  ): Either[(Int, String), Double] =
    val value = optionalDouble(obj, key).getOrElse(Right(default))
    value.flatMap(parsed =>
      Either.cond(parsed >= min && parsed <= max, parsed, 400 -> s"$key must be in [$min,$max]")
    )

  private def requiredChoice(
      obj: collection.Map[String, Value],
      key: String,
      default: String,
      allowed: Set[String]
  ): Either[(Int, String), String] =
    val raw = optionalString(obj, key).map(_.trim).filter(_.nonEmpty).getOrElse(default)
    val normalized = raw.toLowerCase(Locale.ROOT)
    val canonicalByNormalized =
      allowed.map(value => value.toLowerCase(Locale.ROOT) -> value).toMap
    canonicalByNormalized
      .get(normalized)
      .toRight(400 -> s"$key must be one of: ${allowed.toVector.sorted.mkString(", ")}")

  private def requiredVillainPool(
      obj: collection.Map[String, Value]
  ): Either[(Int, String), Vector[String]] =
    val parsed =
      optionalStringArray(obj, "villainPool")
        .orElse(optionalString(obj, "villainPool").map(raw =>
          raw.split(',').toVector.map(_.trim).filter(_.nonEmpty)
        ))
        .getOrElse(Vector("tag", "gto"))
        .map(_.trim)
        .filter(_.nonEmpty)
    if parsed.isEmpty then Left(400 -> "villainPool must include at least one entry")
    else if parsed.length > MaxPlayingHallVillainPoolEntries then
      Left(400 -> s"villainPool must include at most $MaxPlayingHallVillainPoolEntries entries")
    else
      val normalized = parsed.map(_.toLowerCase(Locale.ROOT))
      val allowed = Set("nit", "tag", "lag", "callingstation", "station", "maniac", "gto")
      val invalid = normalized.filterNot(allowed.contains)
      if invalid.nonEmpty then
        Left(400 -> s"villainPool contains unsupported entries: ${invalid.distinct.sorted.mkString(", ")}")
      else Right(normalized)

  private def optionalInt(
      obj: collection.Map[String, Value],
      key: String
  ): Option[Either[(Int, String), Int]] =
    obj.get(key).map {
      case ujson.Num(value) if value.isWhole => Right(value.toInt)
      case Str(value) => value.trim.toIntOption.toRight(400 -> s"$key must be an integer")
      case other => Left(400 -> s"$key must be an integer, got ${ujson.write(other)}")
    }

  private def optionalLong(
      obj: collection.Map[String, Value],
      key: String
  ): Option[Either[(Int, String), Long]] =
    obj.get(key).map {
      case ujson.Num(value) if value.isWhole => Right(value.toLong)
      case Str(value) => value.trim.toLongOption.toRight(400 -> s"$key must be a long")
      case other => Left(400 -> s"$key must be a long, got ${ujson.write(other)}")
    }

  private def optionalDouble(
      obj: collection.Map[String, Value],
      key: String
  ): Option[Either[(Int, String), Double]] =
    obj.get(key).map {
      case ujson.Num(value) => Right(value)
      case Str(value) => value.trim.toDoubleOption.toRight(400 -> s"$key must be a number")
      case other => Left(400 -> s"$key must be a number, got ${ujson.write(other)}")
    }

  private def optionalBoolean(
      obj: collection.Map[String, Value],
      key: String
  ): Option[Either[(Int, String), Boolean]] =
    obj.get(key).map {
      case ujson.Bool(value) => Right(value)
      case Str(value) =>
        value.trim.toLowerCase(Locale.ROOT) match
          case "true" => Right(true)
          case "false" => Right(false)
          case _ => Left(400 -> s"$key must be true or false")
      case other => Left(400 -> s"$key must be true or false, got ${ujson.write(other)}")
    }

  private def optionalStringArray(
      obj: collection.Map[String, Value],
      key: String
  ): Option[Vector[String]] =
    obj.get(key).flatMap {
      case Arr(values) =>
        Some(
          values.collect {
            case Str(value) => value
            case other => other.str
          }.toVector
        )
      case _ => None
    }

  private def runPlayingHall(
      request: PlayingHallRequest,
      cancelSignal: () => Boolean
  ): Either[String, Value] =
    Files.createDirectories(DefaultPlayingHallRoot)
    val runRoot = DefaultPlayingHallRoot.resolve(request.runDirectoryName).toAbsolutePath.normalize()
    val reportEvery = math.max(1, math.min(request.hands, math.max(25, request.hands / 3)))
    val args = Array(
      s"--hands=${request.hands}",
      s"--tableCount=${request.tableCount}",
      s"--playerCount=${request.playerCount}",
      s"--reportEvery=$reportEvery",
      s"--learnEveryHands=${request.learnEveryHands}",
      s"--learningWindowSamples=${request.learningWindowSamples}",
      s"--seed=${request.seed}",
      s"--outDir=$runRoot",
      s"--heroStyle=${request.heroStyle}",
      s"--heroPosition=${request.heroPosition}",
      s"--gtoMode=${request.gtoMode}",
      s"--villainPool=${request.villainPool.mkString(",")}",
      s"--heroExplorationRate=${request.heroExplorationRate}",
      s"--raiseSize=${request.raiseSize}",
      s"--bunchingTrials=${request.bunchingTrials}",
      s"--equityTrials=${request.equityTrials}",
      "--saveTrainingTsv=false",
      "--saveDdreTrainingTsv=false",
      s"--saveReviewHandHistory=${request.saveReviewHandHistory}",
      s"--fullRing=${request.fullRing}"
    )
    TexasHoldemPlayingHall
      .runWithCancel(args, cancelSignal)
      .left
      .map(error => s"playing hall failed: $error")
      .map(summary => renderPlayingHallResult(request, summary))

  private def renderPlayingHallResult(
      request: PlayingHallRequest,
      summary: TexasHoldemPlayingHall.HallSummary
  ): Value =
    val outDir = summary.outDir.toAbsolutePath.normalize()
    Obj(
      "request" -> Obj(
        "hands" -> ujson.Num(request.hands.toDouble),
        "tableCount" -> ujson.Num(request.tableCount.toDouble),
        "playerCount" -> ujson.Num(request.playerCount.toDouble),
        "heroStyle" -> Str(request.heroStyle),
        "heroPosition" -> Str(request.heroPosition),
        "gtoMode" -> Str(request.gtoMode),
        "villainPool" -> Arr.from(request.villainPool.map(Str(_))),
        "heroExplorationRate" -> ujson.Num(request.heroExplorationRate),
        "raiseSize" -> ujson.Num(request.raiseSize),
        "bunchingTrials" -> ujson.Num(request.bunchingTrials.toDouble),
        "equityTrials" -> ujson.Num(request.equityTrials.toDouble),
        "learnEveryHands" -> ujson.Num(request.learnEveryHands.toDouble),
        "learningWindowSamples" -> ujson.Num(request.learningWindowSamples.toDouble),
        "saveReviewHandHistory" -> ujson.Bool(request.saveReviewHandHistory),
        "fullRing" -> ujson.Bool(request.fullRing),
        "seed" -> ujson.Num(request.seed.toDouble)
      ),
      "summary" -> Obj(
        "handsPlayed" -> ujson.Num(summary.handsPlayed.toDouble),
        "tableCount" -> ujson.Num(summary.tableCount.toDouble),
        "playerCount" -> ujson.Num(summary.playerCount.toDouble),
        "heroNetChips" -> ujson.Num(summary.heroNetChips),
        "heroBbPer100" -> ujson.Num(summary.heroBbPer100),
        "heroWins" -> ujson.Num(summary.heroWins.toDouble),
        "heroTies" -> ujson.Num(summary.heroTies.toDouble),
        "heroLosses" -> ujson.Num(summary.heroLosses.toDouble),
        "actionCounts" -> objFromCounts(summary.actionCounts),
        "retrains" -> ujson.Num(summary.retrains.toDouble),
        "modelId" -> Str(summary.modelId),
        "outDir" -> Str(outDir.toString),
        "exactGtoCacheHits" -> ujson.Num(summary.exactGtoCacheHits.toDouble),
        "exactGtoCacheMisses" -> ujson.Num(summary.exactGtoCacheMisses.toDouble),
        "exactGtoCacheHitRate" -> ujson.Num(summary.exactGtoCacheHitRate),
        "exactGtoSolvedByProvider" -> objFromLongCounts(summary.exactGtoSolvedByProvider),
        "exactGtoServedByProvider" -> objFromLongCounts(summary.exactGtoServedByProvider),
        "perVillainNetChips" -> objFromDoubleCounts(summary.perVillainNetChips),
        "perHandHeroNet" -> Arr.from(summary.perHandHeroNet.map(v => ujson.Num(v))),
        "heroDecisionEquities" -> Arr.from(summary.heroDecisionEquities.map(v => ujson.Num(v))),
        "overlayStats" -> summary.overlayStats.map(renderOverlayStats).getOrElse(ujson.Null),
        "outputFiles" -> Arr.from(existingOutputFiles(outDir).map(Str(_)))
      )
    )

  private def existingOutputFiles(outDir: Path): Vector[String] =
    Vector(
      "hands.tsv",
      "learning.tsv",
      "training-selfplay.tsv",
      "ddre-training-selfplay.tsv",
      "review-upload-pokerstars.txt"
    ).flatMap { name =>
      val candidate = outDir.resolve(name)
      Option.when(Files.exists(candidate))(candidate.toString)
    }

  private def renderOverlayStats(stats: sicfun.holdem.runtime.protocol.OverlayStats): Value =
    Obj(
      "decisions" -> ujson.Num(stats.decisions.toDouble),
      "overlayChangeRate" -> ujson.Num(stats.overlayChangeRate),
      "vetoRate" -> ujson.Num(stats.vetoRate),
      "decisionsWithVeto" -> ujson.Num(stats.decisionsWithVeto.toDouble),
      "totalVetoedActions" -> ujson.Num(stats.totalVetoedActions.toDouble),
      "meanLatencyMs" -> ujson.Num(stats.meanLatencyMs),
      "p95LatencyMs" -> ujson.Num(stats.p95LatencyMs),
      "p99LatencyMs" -> ujson.Num(stats.p99LatencyMs),
      "actionDistribution" -> objFromCounts(stats.actionDistribution)
    )

  private def objFromCounts(values: Map[String, Int]): Value =
    Obj.from(values.toVector.sortBy(_._1).map { case (key, value) =>
      key -> ujson.Num(value.toDouble)
    })

  private def objFromLongCounts(values: Map[String, Long]): Value =
    Obj.from(values.toVector.sortBy(_._1).map { case (key, value) =>
      key -> ujson.Num(value.toDouble)
    })

  private def objFromDoubleCounts(values: Map[String, Double]): Value =
    Obj.from(values.toVector.sortBy(_._1).map { case (key, value) =>
      key -> ujson.Num(value)
    })

  private def handleAuthMe(
      exchange: HttpExchange,
      basicAuth: Option[BasicAuthConfig],
      platformAuth: Option[PlatformUserAuth.Service]
  ): Either[(Int, String), JsonResponse] =
    if !exchange.getRequestMethod.equalsIgnoreCase("GET") then Left(405 -> "GET required")
    else
      val value =
        platformAuth match
          case Some(service) => service.authenticationState(authenticatedUser(exchange))
          case None =>
            Obj(
              "authenticationEnabled" -> ujson.Bool(authenticationEnabled(basicAuth, platformAuth)),
              "authenticationMode" -> Str(authenticationMode(basicAuth, platformAuth)),
              "authenticated" -> ujson.Bool(false),
              "allowLocalRegistration" -> ujson.Bool(false),
              "providers" -> Arr(),
              "user" -> ujson.Null,
              "csrfToken" -> ujson.Null
            )
      Right(JsonResponse(200, value))

  private def handleAuthRegister(
      exchange: HttpExchange,
      platformAuth: Option[PlatformUserAuth.Service]
  ): Either[(Int, String), JsonResponse] =
    if !exchange.getRequestMethod.equalsIgnoreCase("POST") then Left(405 -> "POST required")
    else if authenticatedUser(exchange).nonEmpty then Left(409 -> "already signed in")
    else
      platformAuth match
        case None => Left(404 -> "user auth is not enabled")
        case Some(service) =>
          readRequestBody(exchange, 16 * 1024)
            .flatMap(parseRegisterRequest)
            .flatMap { case (email, password, displayName) =>
              service.registerLocal(email, password, displayName).left.map(error => 400 -> error)
            }
            .map(result => loginJsonResponse(service, result, status = 201))

  private def handleAuthLogin(
      exchange: HttpExchange,
      platformAuth: Option[PlatformUserAuth.Service]
  ): Either[(Int, String), JsonResponse] =
    if !exchange.getRequestMethod.equalsIgnoreCase("POST") then Left(405 -> "POST required")
    else if authenticatedUser(exchange).nonEmpty then Left(409 -> "already signed in")
    else
      platformAuth match
        case None => Left(404 -> "user auth is not enabled")
        case Some(service) =>
          readRequestBody(exchange, 16 * 1024)
            .flatMap(parseLoginRequest)
            .flatMap { case (email, password) =>
              service.loginLocal(email, password).left.map(error => 401 -> error)
            }
            .map(result => loginJsonResponse(service, result, status = 200))

  private def handleAuthLogout(
      exchange: HttpExchange,
      platformAuth: Option[PlatformUserAuth.Service]
  ): Either[(Int, String), JsonResponse] =
    if !exchange.getRequestMethod.equalsIgnoreCase("POST") then Left(405 -> "POST required")
    else if !ensurePlatformCsrf(exchange, platformAuth) then Left(403 -> SessionCsrfRequiredMessage)
    else
      platformAuth match
        case None => Left(404 -> "user auth is not enabled")
        case Some(service) =>
          val clearedCookie = service.revokeSession(cookieHeader(exchange))
          Right(
            JsonResponse(
              200,
              service.authenticationState(None),
              headers = Vector("Set-Cookie" -> clearedCookie)
            )
          )

  private def handleAuthProfile(
      exchange: HttpExchange,
      platformAuth: Option[PlatformUserAuth.Service]
  ): Either[(Int, String), JsonResponse] =
    if !exchange.getRequestMethod.equalsIgnoreCase("POST") then Left(405 -> "POST required")
    else if !ensurePlatformCsrf(exchange, platformAuth) then Left(403 -> SessionCsrfRequiredMessage)
    else
      (platformAuth, authenticatedUser(exchange)) match
        case (Some(service), Some(user)) =>
          readRequestBody(exchange, 16 * 1024)
            .flatMap(parseProfileUpdateRequest)
            .flatMap { case (displayName, heroName, preferredSite, timeZone) =>
              service
                .updateProfile(user.userId, displayName, heroName, preferredSite, timeZone)
                .left
                .map(error => 400 -> error)
            }
            .map { updated =>
              val refreshedUser = user.copy(profile = updated)
              JsonResponse(200, service.authenticationState(Some(refreshedUser)))
            }
        case _ => Left(404 -> "user auth is not enabled")

  private def handleOidcStart(
      exchange: HttpExchange,
      platformAuth: PlatformUserAuth.Service
  ): Either[(Int, String), RedirectResponse] =
    if !exchange.getRequestMethod.equalsIgnoreCase("GET") then Left(405 -> "GET required")
    else
      extractOidcProviderId(exchange, "/start").flatMap { providerId =>
        platformAuth.startOidc(providerId)
          .left
          .map(error => 400 -> error)
          .map(location => RedirectResponse(location = location))
      }

  private def handleOidcCallback(
      exchange: HttpExchange,
      platformAuth: PlatformUserAuth.Service,
      providerId: String
  ): Either[(Int, String), RedirectResponse] =
    if !exchange.getRequestMethod.equalsIgnoreCase("GET") then Left(405 -> "GET required")
    else
      val query = parseQuery(exchange)
      query.get("error") match
        case Some(error) =>
          Right(RedirectResponse(location = PlatformUserAuth.oidcFailureRedirect(error)))
        case None =>
          (query.get("state"), query.get("code")) match
            case (Some(state), Some(code)) =>
              platformAuth.finishOidc(providerId, state, code) match
                case Left(error) =>
                  Right(RedirectResponse(location = PlatformUserAuth.oidcFailureRedirect(error)))
                case Right(result) =>
                  Right(
                    RedirectResponse(
                      location = PlatformUserAuth.oidcSuccessRedirect,
                      headers = Vector("Set-Cookie" -> result.cookieHeader)
                    )
                  )
            case _ =>
              Right(RedirectResponse(location = PlatformUserAuth.oidcFailureRedirect("missing_code_or_state")))

  private def loginJsonResponse(
      service: PlatformUserAuth.Service,
      result: PlatformUserAuth.LoginResult,
      status: Int
  ): JsonResponse =
    val authenticated = PlatformUserAuth.AuthenticatedUser(
      userId = result.user.userId,
      email = result.user.email,
      profile = result.user,
      csrfToken = result.csrfToken
    )
    JsonResponse(
      status = status,
      value = service.authenticationState(Some(authenticated)),
      headers = Vector("Set-Cookie" -> result.cookieHeader)
    )

  private def parseRegisterRequest(
      body: String
  ): Either[(Int, String), (String, String, Option[String])] =
    try
      val obj = ujson.read(body).obj
      for
        email <- requiredString(obj, "email")
        password <- requiredString(obj, "password")
      yield (email.trim, password, optionalString(obj, "displayName").map(_.trim).filter(_.nonEmpty))
    catch
      case NonFatal(e) => Left(400 -> s"invalid JSON request: ${e.getMessage}")

  private def parseLoginRequest(body: String): Either[(Int, String), (String, String)] =
    try
      val obj = ujson.read(body).obj
      for
        email <- requiredString(obj, "email")
        password <- requiredString(obj, "password")
      yield (email.trim, password)
    catch
      case NonFatal(e) => Left(400 -> s"invalid JSON request: ${e.getMessage}")

  private def parseProfileUpdateRequest(
      body: String
  ): Either[(Int, String), (Option[String], Option[String], Option[String], Option[String])] =
    try
      val obj = ujson.read(body).obj
      Right(
        (
          optionalString(obj, "displayName"),
          optionalString(obj, "heroName"),
          optionalString(obj, "preferredSite"),
          optionalString(obj, "timeZone")
        )
      )
    catch
      case NonFatal(e) => Left(400 -> s"invalid JSON request: ${e.getMessage}")

  private def extractOidcProviderId(
      exchange: HttpExchange,
      suffix: String
  ): Either[(Int, String), String] =
    val path = Option(exchange.getRequestURI.getPath).getOrElse("")
    val segments = path.stripPrefix("/").split('/').toVector
    segments match
      case Vector("api", "auth", "oidc", providerId, action) if s"/$action" == suffix =>
        Right(providerId)
      case _ => Left(404 -> "not found")

  private def parseQuery(exchange: HttpExchange): Map[String, String] =
    Option(exchange.getRequestURI.getRawQuery).toVector
      .flatMap(_.split('&').toVector)
      .flatMap { pair =>
        pair.split("=", 2) match
          case Array(name, value) => Some(urlDecode(name) -> urlDecode(value))
          case Array(name) if name.nonEmpty => Some(urlDecode(name) -> "")
          case _ => None
      }
      .toMap

  private def requiredString(
      obj: collection.Map[String, Value],
      key: String
  ): Either[(Int, String), String] =
    optionalString(obj, key).filter(_.nonEmpty).toRight(400 -> s"$key is required")

  private def optionalString(
      obj: collection.Map[String, Value],
      key: String
  ): Option[String] =
    obj.get(key).flatMap {
      case Str(value) => Some(value)
      case ujson.Null => None
      case other => Some(other.str)
    }

  private def parseOptionalSite(raw: Option[String]): Either[(Int, String), Option[HandHistorySite]] =
    raw.map(_.trim).filter(_.nonEmpty).filterNot(_.equalsIgnoreCase("auto")) match
      case None => Right(None)
      case Some(value) => HandHistorySite.parse(value).left.map(err => 400 -> err).map(Some(_))

  private def readRequestBody(
      exchange: HttpExchange,
      maxUploadBytes: Int
  ): Either[(Int, String), String] =
    Option(exchange.getRequestHeaders.getFirst("Content-Length"))
      .flatMap(_.toLongOption)
      .filter(_ > maxUploadBytes.toLong) match
        case Some(_) =>
          Left(413 -> s"request body exceeds max upload size of $maxUploadBytes bytes")
        case None =>
          val input = exchange.getRequestBody
          val buffer = Array.ofDim[Byte](8192)
          val output = new ByteArrayOutputStream(math.min(maxUploadBytes, 8192))
          var total = 0
          var bytesRead = input.read(buffer)
          while bytesRead != -1 && total <= maxUploadBytes do
            total += bytesRead
            if total <= maxUploadBytes then
              output.write(buffer, 0, bytesRead)
            bytesRead = input.read(buffer)
          if total > maxUploadBytes then
            Left(413 -> s"request body exceeds max upload size of $maxUploadBytes bytes")
          else Right(new String(output.toByteArray, StandardCharsets.UTF_8))

  private def parseArgs(args: Array[String]): Either[String, ServerConfig] =
    if args.contains("--help") || args.contains("-h") then Left(usage)
    else
      for
        options <- CliHelpers.parseOptionsAllowBlankValues(args)
        host = options.get("host").orElse(env("HOST")).getOrElse("127.0.0.1")
        port <- resolveIntOption(options, "port", env("PORT"), 8080)
        staticDir <- parseDirectory(
          options.get("staticDir").orElse(env("STATIC_DIR")).getOrElse("docs/site-preview-hybrid"),
          "staticDir"
        )
        maxUploadBytes <- resolveIntOption(options, "maxUploadBytes", env("MAX_UPLOAD_BYTES"), 2 * 1024 * 1024)
        _ <- Either.cond(maxUploadBytes > 0, (), "--maxUploadBytes must be positive")
        analysisTimeoutMs <- resolveLongOption(
          options,
          "analysisTimeoutMs",
          env("ANALYSIS_TIMEOUT_MS"),
          DefaultAnalysisTimeoutMs
        )
        _ <- Either.cond(analysisTimeoutMs >= 0, (), "--analysisTimeoutMs must be zero or positive")
        playingHallTimeoutMs <- resolveLongOption(
          options,
          "playingHallTimeoutMs",
          env("PLAYING_HALL_TIMEOUT_MS"),
          DefaultPlayingHallTimeoutMs
        )
        _ <- Either.cond(playingHallTimeoutMs >= 0, (), "--playingHallTimeoutMs must be zero or positive")
        maxConcurrentJobs <- resolveIntOption(
          options,
          "maxConcurrentJobs",
          env("MAX_CONCURRENT_JOBS"),
          defaultMaxConcurrentJobs()
        )
        _ <- Either.cond(maxConcurrentJobs > 0, (), "--maxConcurrentJobs must be positive")
        maxQueuedJobs <- resolveIntOption(
          options,
          "maxQueuedJobs",
          env("MAX_QUEUED_JOBS"),
          defaultMaxQueuedJobs(maxConcurrentJobs)
        )
        _ <- Either.cond(maxQueuedJobs > 0, (), "--maxQueuedJobs must be positive")
        shutdownGraceMs <- resolveLongOption(
          options,
          "shutdownGraceMs",
          env("SHUTDOWN_GRACE_MS"),
          DefaultShutdownGraceMs
        )
        _ <- Either.cond(shutdownGraceMs >= 0, (), "--shutdownGraceMs must be zero or positive")
        rateLimitSubmitsPerMinute <- resolveIntOption(
          options,
          "rateLimitSubmitsPerMinute",
          env("RATE_LIMIT_SUBMITS_PER_MINUTE"),
          DefaultRateLimitSubmitsPerMinute
        )
        _ <- Either.cond(
          rateLimitSubmitsPerMinute >= 0,
          (),
          "--rateLimitSubmitsPerMinute must be zero or positive"
        )
        rateLimitStatusPerMinute <- resolveIntOption(
          options,
          "rateLimitStatusPerMinute",
          env("RATE_LIMIT_STATUS_PER_MINUTE"),
          DefaultRateLimitStatusPerMinute
        )
        _ <- Either.cond(
          rateLimitStatusPerMinute >= 0,
          (),
          "--rateLimitStatusPerMinute must be zero or positive"
        )
        rateLimitClientIpHeader = options
          .get("rateLimitClientIpHeader")
          .orElse(env("RATE_LIMIT_CLIENT_IP_HEADER"))
          .map(_.trim)
          .filter(_.nonEmpty)
        rateLimitTrustedProxyIps <- parseTrustedProxyIps(
          options
            .get("rateLimitTrustedProxyIps")
            .orElse(env("RATE_LIMIT_TRUSTED_PROXY_IPS"))
            .map(_.trim)
            .filter(_.nonEmpty)
        )
        drainSignalFile <- parseOptionalPath(
          options.get("drainSignalFile").orElse(env("DRAIN_SIGNAL_FILE")),
          "drainSignalFile"
        )
        basicAuth <- resolveBasicAuthConfig(
          options.get("basicAuthUser").orElse(env("BASIC_AUTH_USER")).map(_.trim).filter(_.nonEmpty),
          options.get("basicAuthPassword").orElse(env("BASIC_AUTH_PASSWORD")).map(_.trim).filter(_.nonEmpty)
        )
        allowUnauthenticatedPublicBind <- resolveBooleanOption(
          options,
          "allowUnauthenticatedPublicBind",
          env("ALLOW_UNAUTHENTICATED_PUBLIC_BIND"),
          default = false
        )
        allowInsecureUserAuth <- resolveBooleanOption(
          options,
          "allowInsecureUserAuth",
          env("ALLOW_INSECURE_USER_AUTH"),
          default = false
        )
        userStorePath <- parseOptionalPath(
          options.get("userStorePath").orElse(env("USER_STORE_PATH")),
          "userStorePath"
        )
        userAuthAllowRegistration <- resolveBooleanOption(
          options,
          "userAuthAllowRegistration",
          env("USER_AUTH_ALLOW_REGISTRATION"),
          default = true
        )
        userAuthSessionTtlMs <- resolveLongOption(
          options,
          "userAuthSessionTtlMs",
          env("USER_AUTH_SESSION_TTL_MS"),
          12L * 60L * 60L * 1000L
        )
        _ <- Either.cond(userAuthSessionTtlMs > 0L, (), "--userAuthSessionTtlMs must be positive")
        userAuthCookieSecure <- resolveBooleanOption(
          options,
          "userAuthCookieSecure",
          env("USER_AUTH_COOKIE_SECURE"),
          default = false
        )
        googleOidcClientId = options.get("googleOidcClientId").orElse(env("GOOGLE_OIDC_CLIENT_ID")).map(_.trim).filter(_.nonEmpty)
        googleOidcClientSecret = options.get("googleOidcClientSecret").orElse(env("GOOGLE_OIDC_CLIENT_SECRET")).map(_.trim).filter(_.nonEmpty)
        googleOidcRedirectUri = options.get("googleOidcRedirectUri").orElse(env("GOOGLE_OIDC_REDIRECT_URI")).map(_.trim).filter(_.nonEmpty)
        googleOidc <- resolveGoogleOidcConfig(
          googleOidcClientId,
          googleOidcClientSecret,
          googleOidcRedirectUri
        )
        platformAuth <- resolvePlatformAuthConfig(
          userStorePath = userStorePath,
          allowLocalRegistration = userAuthAllowRegistration,
          sessionTtlMs = userAuthSessionTtlMs,
          cookieSecure = userAuthCookieSecure,
          oidcProviders = googleOidc.toVector
        )
        _ <- Either.cond(
          basicAuth.isEmpty || platformAuth.isEmpty,
          (),
          "basic auth and user auth cannot both be enabled"
        )
        _ <- Either.cond(
          allowUnauthenticatedPublicBind || !isNonLoopbackBindHost(host) || basicAuth.nonEmpty || platformAuth.nonEmpty,
          (),
          "refusing to bind to a non-loopback host without auth; configure BASIC_AUTH_*/USER_STORE_PATH or set --allowUnauthenticatedPublicBind=true / ALLOW_UNAUTHENTICATED_PUBLIC_BIND=true to override for a trusted private network"
        )
        _ <- validateNetworkUserAuthSafety(
          host = host,
          platformAuth = platformAuth,
          googleOidcRedirectUri = googleOidcRedirectUri,
          cookieSecure = userAuthCookieSecure,
          allowInsecureUserAuth = allowInsecureUserAuth
        )
        modelDir <- parseOptionalDirectory(options.get("model").orElse(env("MODEL_DIR")), "model")
        seed <- resolveLongOption(options, "seed", env("SEED"), 42L)
        bunchingTrials <- resolveIntOption(options, "bunchingTrials", env("BUNCHING_TRIALS"), 200)
        equityTrials <- resolveIntOption(options, "equityTrials", env("EQUITY_TRIALS"), 2000)
        budgetMs <- resolveLongOption(options, "budgetMs", env("BUDGET_MS"), 1500L)
        maxDecisions <- resolveIntOption(options, "maxDecisions", env("MAX_DECISIONS"), 12)
      yield ServerConfig(
        host = host,
        port = port,
        staticDir = staticDir,
        maxUploadBytes = maxUploadBytes,
        analysisTimeoutMs = analysisTimeoutMs,
        playingHallTimeoutMs = playingHallTimeoutMs,
        maxConcurrentJobs = maxConcurrentJobs,
        maxQueuedJobs = maxQueuedJobs,
        shutdownGraceMs = shutdownGraceMs,
        rateLimitSubmitsPerMinute = rateLimitSubmitsPerMinute,
        rateLimitStatusPerMinute = rateLimitStatusPerMinute,
        rateLimitClientIpHeader = rateLimitClientIpHeader,
        rateLimitTrustedProxyIps = rateLimitTrustedProxyIps,
        drainSignalFile = drainSignalFile,
        basicAuth = basicAuth,
        serviceConfig = HandHistoryReviewService.ServiceConfig(
          modelDir = modelDir,
          seed = seed,
          bunchingTrials = bunchingTrials,
          equityTrials = equityTrials,
          budgetMs = budgetMs,
          maxDecisions = maxDecisions
        ),
        platformAuth = platformAuth
      )

  private def parseDirectory(raw: String, label: String): Either[String, Path] =
    val path = Paths.get(raw).toAbsolutePath.normalize()
    if Files.isDirectory(path) then Right(path)
    else Left(s"--$label directory not found: $raw")

  private def parseOptionalDirectory(raw: Option[String], label: String): Either[String, Option[Path]] =
    raw.map(_.trim).filter(_.nonEmpty) match
      case None => Right(None)
      case Some(value) => parseDirectory(value, label).map(Some(_))

  private def parseOptionalPath(raw: Option[String], label: String): Either[String, Option[Path]] =
    raw.map(_.trim).filter(_.nonEmpty) match
      case None => Right(None)
      case Some(value) =>
        try Right(Some(Paths.get(value).toAbsolutePath.normalize()))
        catch
          case NonFatal(e) => Left(s"--$label is not a valid path: ${e.getMessage}")

  private def resolveBasicAuthConfig(
      maybeUsername: Option[String],
      maybePassword: Option[String]
  ): Either[String, Option[BasicAuthConfig]] =
    (maybeUsername, maybePassword) match
      case (None, None) => Right(None)
      case (Some(_), None) =>
        Left("basic auth requires both --basicAuthUser/BASIC_AUTH_USER and --basicAuthPassword/BASIC_AUTH_PASSWORD")
      case (None, Some(_)) =>
        Left("basic auth requires both --basicAuthUser/BASIC_AUTH_USER and --basicAuthPassword/BASIC_AUTH_PASSWORD")
      case (Some(username), Some(password)) =>
        if username.contains(":") then Left("basic auth username must not contain ':'")
        else if username.isEmpty then Left("basic auth username must be non-empty")
        else if password.isEmpty then Left("basic auth password must be non-empty")
        else Right(Some(BasicAuthConfig(username = username, password = password)))

  private def resolveIntOption(
      options: Map[String, String],
      key: String,
      envValue: Option[String],
      default: Int
  ): Either[String, Int] =
    options.get(key)
      .orElse(envValue)
      .map(_.trim)
      .filter(_.nonEmpty) match
        case None => Right(default)
        case Some(raw) => raw.toIntOption.toRight(s"--$key must be an integer")

  private def resolveLongOption(
      options: Map[String, String],
      key: String,
      envValue: Option[String],
      default: Long
  ): Either[String, Long] =
    options.get(key)
      .orElse(envValue)
      .map(_.trim)
      .filter(_.nonEmpty) match
        case None => Right(default)
        case Some(raw) => raw.toLongOption.toRight(s"--$key must be a long")

  private def resolveBooleanOption(
      options: Map[String, String],
      key: String,
      envValue: Option[String],
      default: Boolean
  ): Either[String, Boolean] =
    options.get(key)
      .orElse(envValue)
      .map(_.trim)
      .filter(_.nonEmpty) match
        case None => Right(default)
        case Some(raw) =>
          raw.toLowerCase match
            case "true" | "1" | "yes" | "on" => Right(true)
            case "false" | "0" | "no" | "off" => Right(false)
            case _ => Left(s"--$key must be a boolean")

  private def resolveGoogleOidcConfig(
      maybeClientId: Option[String],
      maybeClientSecret: Option[String],
      maybeRedirectUri: Option[String]
  ): Either[String, Option[PlatformUserAuth.OidcProvider]] =
    (maybeClientId, maybeClientSecret, maybeRedirectUri) match
      case (None, None, None) => Right(None)
      case (Some(clientId), Some(clientSecret), Some(redirectUri)) =>
        parseAbsoluteHttpUri(redirectUri, "--googleOidcRedirectUri/GOOGLE_OIDC_REDIRECT_URI").map { _ =>
          Some(
            new PlatformUserAuth.GoogleOidcProvider(
              PlatformUserAuth.GoogleOidcConfig(
                clientId = clientId,
                clientSecret = clientSecret,
                redirectUri = redirectUri
              )
            )
          )
        }
      case _ =>
        Left(
          "Google OIDC requires --googleOidcClientId/GOOGLE_OIDC_CLIENT_ID, --googleOidcClientSecret/GOOGLE_OIDC_CLIENT_SECRET, and --googleOidcRedirectUri/GOOGLE_OIDC_REDIRECT_URI"
        )

  private def validateNetworkUserAuthSafety(
      host: String,
      platformAuth: Option[PlatformUserAuth.Config],
      googleOidcRedirectUri: Option[String],
      cookieSecure: Boolean,
      allowInsecureUserAuth: Boolean
  ): Either[String, Unit] =
    if allowInsecureUserAuth || !isNonLoopbackBindHost(host) || platformAuth.isEmpty then Right(())
    else if !cookieSecure then
      Left(
        "platform-user auth on a non-loopback host requires --userAuthCookieSecure=true / USER_AUTH_COOKIE_SECURE=true; set --allowInsecureUserAuth=true / ALLOW_INSECURE_USER_AUTH=true only for trusted private-network testing"
      )
    else
      googleOidcRedirectUri match
        case Some(redirectUri) =>
          parseAbsoluteHttpUri(redirectUri, "--googleOidcRedirectUri/GOOGLE_OIDC_REDIRECT_URI").flatMap { uri =>
            if uri.getScheme.equalsIgnoreCase("https") then Right(())
            else
              Left(
                "Google OIDC on a non-loopback host requires an https:// redirect URI; set --allowInsecureUserAuth=true / ALLOW_INSECURE_USER_AUTH=true only for trusted private-network testing"
              )
          }
        case None => Right(())

  private def resolvePlatformAuthConfig(
      userStorePath: Option[Path],
      allowLocalRegistration: Boolean,
      sessionTtlMs: Long,
      cookieSecure: Boolean,
      oidcProviders: Vector[PlatformUserAuth.OidcProvider]
  ): Either[String, Option[PlatformUserAuth.Config]] =
    userStorePath match
      case None if oidcProviders.nonEmpty =>
        Left("user auth with OIDC requires --userStorePath/USER_STORE_PATH")
      case None => Right(None)
      case Some(path) =>
        Right(
          Some(
            PlatformUserAuth.Config(
              storePath = path,
              sessionTtlMs = sessionTtlMs,
              allowLocalRegistration = allowLocalRegistration,
              cookieSecure = cookieSecure,
              oidcProviders = oidcProviders
            )
          )
        )

  private def parseAbsoluteHttpUri(raw: String, label: String): Either[String, URI] =
    try
      val uri = URI.create(raw.trim)
      if !uri.isAbsolute || uri.getHost == null then Left(s"$label must be an absolute http:// or https:// URI")
      else if uri.getFragment != null then Left(s"$label must not contain a URI fragment")
      else
        uri.getScheme.toLowerCase(Locale.ROOT) match
          case "http" | "https" => Right(uri)
          case _ => Left(s"$label must use http:// or https://")
    catch
      case NonFatal(e) => Left(s"$label is not a valid URI: ${e.getMessage}")

  private def env(name: String): Option[String] =
    Option(System.getenv(name)).map(_.trim).filter(_.nonEmpty)

  private def defaultMaxConcurrentJobs(): Int =
    math.max(1, math.min(4, Runtime.getRuntime.availableProcessors() - 1))

  private def defaultMaxQueuedJobs(maxConcurrentJobs: Int): Int =
    math.max(8, maxConcurrentJobs * 8)

  private def isNonLoopbackBindHost(host: String): Boolean =
    val normalized = host.trim.stripPrefix("[").stripSuffix("]").toLowerCase(Locale.ROOT)
    normalized match
      case "" => false
      case "localhost" => false
      case "::1" => false
      case "0:0:0:0:0:0:0:1" => false
      case value if value.startsWith("127.") => false
      case _ => true

  private final case class JsonResponse(
      status: Int,
      value: Value,
      headers: Vector[(String, String)] = Vector.empty
  )

  private final case class RedirectResponse(
      location: String,
      headers: Vector[(String, String)] = Vector.empty,
      status: Int = 302
  )

  private final case class AcceptedJob(
      jobId: String,
      submittedAtEpochMs: Long,
      statusUrl: String,
      pollAfterMs: Int
  )

  private[web] final case class PlayingHallRequest(
      hands: Int,
      tableCount: Int,
      playerCount: Int,
      heroStyle: String,
      heroPosition: String,
      gtoMode: String,
      villainPool: Vector[String],
      heroExplorationRate: Double,
      raiseSize: Double,
      bunchingTrials: Int,
      equityTrials: Int,
      learnEveryHands: Int,
      learningWindowSamples: Int,
      saveReviewHandHistory: Boolean,
      fullRing: Boolean,
      seed: Long
  ):
    def runDirectoryName: String =
      s"${System.currentTimeMillis()}-${UUID.randomUUID().toString.take(8)}"

    def logSummary: String =
      s"hands=$hands tableCount=$tableCount playerCount=$playerCount heroStyle=$heroStyle heroPosition=$heroPosition gtoMode=$gtoMode villainPool=${villainPool.mkString(",")} seed=$seed"

  private final case class AnalysisJobMetrics(
      maxConcurrentJobs: Int,
      maxQueuedJobs: Int,
      queuedJobs: Int,
      runningJobs: Int,
      timedOutWorkersInFlight: Int,
      retainedTerminalJobs: Int
  )

  private final case class ReadinessStatus(
      ready: Boolean,
      reason: String,
      draining: Boolean,
      acceptingAnalysisJobs: Boolean,
      drainSignalPresent: Boolean,
      timedOutWorkersInFlight: Int
  )

  private enum RateLimitBucket:
    case Submit, JobStatus

    def id: String = this match
      case Submit => "submit"
      case JobStatus => "job-status"

    def description: String = this match
      case Submit => "submit"
      case JobStatus => "job status"

  private enum AuthRequirement:
    case None, Optional, Required

  private final case class RateLimitState(
      windowStartedAtMs: Long,
      requestCount: Int
  )

  private final case class RateLimitRejection(
      bucket: RateLimitBucket,
      limitPerMinute: Int,
      retryAfterMs: Long,
      clientKey: String
  )

  private sealed trait AnalysisJobState:
    def status: String
    def submittedAtEpochMs: Long
    def startedAtEpochMs: Option[Long]
    def completedAtEpochMs: Option[Long]
    def isTerminal: Boolean

  private object AnalysisJobState:
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

  private final class AnalysisJobStore(
      executor: ThreadPoolExecutor,
      timeoutExecutor: ScheduledExecutorService,
      backend: AnalysisBackend,
      analysisTimeoutMs: Long,
      nowMillis: () => Long = () => System.currentTimeMillis()
  ):
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
          logWarn(
            s"job rejected unavailable queuedJobs=${executor.getQueue.size()} runningJobs=${executor.getActiveCount()} reason=$error"
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
                  s"job rejected unavailable queuedJobs=${executor.getQueue.size()} runningJobs=${executor.getActiveCount()} reason=$error"
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
                  logWarn(
                    s"job rejected unavailable queuedJobs=${executor.getQueue.size()} runningJobs=${executor.getActiveCount()} reason=$error"
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

    def metrics: AnalysisJobMetrics =
      purgeExpiredJobs()
      var retainedTerminalJobs = 0
      val iterator = jobs.values().iterator()
      while iterator.hasNext do
        if iterator.next().isTerminal then
          retainedTerminalJobs += 1
      AnalysisJobMetrics(
        maxConcurrentJobs = executor.getMaximumPoolSize,
        maxQueuedJobs = queueCapacity(executor),
        queuedJobs = executor.getQueue.size(),
        runningJobs = executor.getActiveCount(),
        timedOutWorkersInFlight = timedOutWorkersInFlight.get(),
        retainedTerminalJobs = retainedTerminalJobs
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
      logInfo(
        s"job started jobId=$jobId queuedJobs=${executor.getQueue.size()} runningJobs=${executor.getActiveCount()} heroName=${request.heroName.getOrElse("-")} site=${request.site.map(_.toString).getOrElse("auto")} analysisTimeoutMs=$analysisTimeoutMs"
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
          logWarn(
            s"job failed jobId=$jobId durationMs=${completedAt - startedAt} errorStatus=$errorStatus queuedJobs=${executor.getQueue.size()} runningJobs=${executor.getActiveCount()} error=$error"
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

  private final class PlayingHallJobStore(
      executor: ThreadPoolExecutor,
      timeoutExecutor: ScheduledExecutorService,
      backend: PlayingHallBackend,
      analysisTimeoutMs: Long,
      nowMillis: () => Long = () => System.currentTimeMillis()
  ):
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
          logWarn(
            s"playing hall job rejected unavailable queuedJobs=${executor.getQueue.size()} runningJobs=${executor.getActiveCount()} reason=$error"
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
                  s"playing hall job rejected unavailable queuedJobs=${executor.getQueue.size()} runningJobs=${executor.getActiveCount()} reason=$error"
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
                  logWarn(
                    s"playing hall job rejected unavailable queuedJobs=${executor.getQueue.size()} runningJobs=${executor.getActiveCount()} reason=$error"
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
        s"playing hall job started jobId=$jobId queuedJobs=${executor.getQueue.size()} runningJobs=${executor.getActiveCount()} timeoutMs=$analysisTimeoutMs ${request.logSummary}"
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
      if timedOut.get() then
        timedOutWorkersInFlight.decrementAndGet()
      finalState match
        case Completed(_, _, completedAt, _) =>
          logInfo(
            s"playing hall job completed jobId=$jobId durationMs=${completedAt - startedAt} queuedJobs=${executor.getQueue.size()} runningJobs=${executor.getActiveCount()}"
          )
        case Failed(_, _, completedAt, errorStatus, error) =>
          logWarn(
            s"playing hall job failed jobId=$jobId durationMs=${completedAt - startedAt} errorStatus=$errorStatus queuedJobs=${executor.getQueue.size()} runningJobs=${executor.getActiveCount()} error=$error"
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
                    s"playing hall job timed out jobId=$jobId timeoutMs=$analysisTimeoutMs queuedJobs=${executor.getQueue.size()} runningJobs=${executor.getActiveCount()}"
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

    private def timeoutFailure(submittedAt: Long, startedAt: Long): Failed =
      Failed(
        submittedAtEpochMs = submittedAt,
        startedAt = startedAt,
        completedAt = nowMillis(),
        errorStatus = 504,
        error = s"playing hall timed out after ${analysisTimeoutMs}ms"
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
          iterator.remove()

  private final class RequestRateLimiter(
      submitsPerMinute: Int,
      statusPerMinute: Int,
      trustedClientIpHeader: Option[String],
      trustedProxyIps: Set[String],
      nowMillis: () => Long = () => System.currentTimeMillis()
  ):
    private val windows = new ConcurrentHashMap[String, RateLimitState]()
    private val lastCleanupAtMs = new AtomicLong(0L)

    def check(
        exchange: HttpExchange,
        bucket: RateLimitBucket,
        principalKey: Option[String] = None
    ): Option[RateLimitRejection] =
      val limitPerMinute = bucket match
        case RateLimitBucket.Submit => submitsPerMinute
        case RateLimitBucket.JobStatus => statusPerMinute
      if limitPerMinute <= 0 then None
      else
        val now = nowMillis()
        cleanupIfDue(now)
        val clientKey = principalKey.getOrElse(rateLimitClientKey(exchange, trustedClientIpHeader, trustedProxyIps))
        val key = s"${bucket.id}|$clientKey"
        var rejection = Option.empty[RateLimitRejection]
        windows.compute(
          key,
          (_, existing) =>
            if existing == null || now - existing.windowStartedAtMs >= DefaultRateLimitWindowMs then
              RateLimitState(windowStartedAtMs = now, requestCount = 1)
            else if existing.requestCount < limitPerMinute then
              existing.copy(requestCount = existing.requestCount + 1)
            else
              rejection = Some(
                RateLimitRejection(
                  bucket = bucket,
                  limitPerMinute = limitPerMinute,
                  retryAfterMs = math.max(1L, DefaultRateLimitWindowMs - (now - existing.windowStartedAtMs)),
                  clientKey = clientKey
                )
              )
              existing
        )
        rejection

    private def cleanupIfDue(now: Long): Unit =
      val lastCleanup = lastCleanupAtMs.get()
      if now - lastCleanup >= DefaultRateLimitWindowMs && lastCleanupAtMs.compareAndSet(lastCleanup, now) then
        val cutoff = now - (DefaultRateLimitWindowMs * 2L)
        val iterator = windows.entrySet().iterator()
        while iterator.hasNext do
          val entry = iterator.next()
          if entry.getValue.windowStartedAtMs < cutoff then
            iterator.remove()

  private def renderHealth(
      config: ServerConfig,
      boundPort: Int,
      startedAtEpochMs: Long,
      jobStore: AnalysisJobStore,
      playingHallJobStore: PlayingHallJobStore,
      activeHttpRequests: Int,
      draining: AtomicBoolean
  ): JsonResponse =
    val metrics = jobStore.metrics
    val readiness = readinessStatus(config, jobStore, playingHallJobStore, draining)
    val otherActiveHttpRequests = math.max(0, activeHttpRequests - 1)
    JsonResponse(
      status = 200,
      value = Obj(
        "ok" -> ujson.Bool(true),
        "ready" -> ujson.Bool(readiness.ready),
        "readyReason" -> Str(readiness.reason),
        "draining" -> ujson.Bool(readiness.draining),
        "acceptingAnalysisJobs" -> ujson.Bool(readiness.acceptingAnalysisJobs),
        "authenticationEnabled" -> ujson.Bool(authenticationEnabled(config.basicAuth, config.platformAuth)),
        "authenticationMode" -> Str(authenticationMode(config.basicAuth, config.platformAuth)),
        "service" -> Str("hand-history-review"),
        "host" -> Str(config.host),
        "port" -> ujson.Num(boundPort.toDouble),
        "startedAtEpochMs" -> ujson.Num(startedAtEpochMs.toDouble),
        "uptimeMs" -> ujson.Num((System.currentTimeMillis() - startedAtEpochMs).toDouble),
        "modelConfigured" -> ujson.Bool(config.serviceConfig.modelDir.nonEmpty),
        "modelSource" -> Str(healthModelSource(config.serviceConfig)),
        "drainSignalConfigured" -> ujson.Bool(config.drainSignalFile.nonEmpty),
        "drainSignalPresent" -> ujson.Bool(readiness.drainSignalPresent),
        "maxUploadBytes" -> ujson.Num(config.maxUploadBytes.toDouble),
        "analysisTimeoutMs" -> ujson.Num(config.analysisTimeoutMs.toDouble),
        "playingHallTimeoutMs" -> ujson.Num(config.playingHallTimeoutMs.toDouble),
        "rateLimitSubmitsPerMinute" -> ujson.Num(config.rateLimitSubmitsPerMinute.toDouble),
        "rateLimitStatusPerMinute" -> ujson.Num(config.rateLimitStatusPerMinute.toDouble),
        "rateLimitClientIpSource" -> Str(rateLimitClientIpSource(config.rateLimitClientIpHeader, config.rateLimitTrustedProxyIps)),
        "maxConcurrentJobs" -> ujson.Num(metrics.maxConcurrentJobs.toDouble),
        "maxQueuedJobs" -> ujson.Num(metrics.maxQueuedJobs.toDouble),
        "activeHttpRequests" -> ujson.Num(otherActiveHttpRequests.toDouble),
        "queuedJobs" -> ujson.Num(metrics.queuedJobs.toDouble),
        "runningJobs" -> ujson.Num(metrics.runningJobs.toDouble),
        "timedOutWorkersInFlight" -> ujson.Num(readiness.timedOutWorkersInFlight.toDouble),
        "retainedTerminalJobs" -> ujson.Num(metrics.retainedTerminalJobs.toDouble)
      )
    )

  private def renderReadiness(
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
        "port" -> ujson.Num(boundPort.toDouble),
        "ready" -> ujson.Bool(readiness.ready),
        "reason" -> Str(readiness.reason),
        "draining" -> ujson.Bool(readiness.draining),
        "acceptingAnalysisJobs" -> ujson.Bool(readiness.acceptingAnalysisJobs),
        "authenticationEnabled" -> ujson.Bool(authenticationEnabled(config.basicAuth, config.platformAuth)),
        "authenticationMode" -> Str(authenticationMode(config.basicAuth, config.platformAuth)),
        "drainSignalConfigured" -> ujson.Bool(config.drainSignalFile.nonEmpty),
        "drainSignalPresent" -> ujson.Bool(readiness.drainSignalPresent),
        "analysisTimeoutMs" -> ujson.Num(config.analysisTimeoutMs.toDouble),
        "playingHallTimeoutMs" -> ujson.Num(config.playingHallTimeoutMs.toDouble),
        "rateLimitSubmitsPerMinute" -> ujson.Num(config.rateLimitSubmitsPerMinute.toDouble),
        "rateLimitStatusPerMinute" -> ujson.Num(config.rateLimitStatusPerMinute.toDouble),
        "rateLimitClientIpSource" -> Str(rateLimitClientIpSource(config.rateLimitClientIpHeader, config.rateLimitTrustedProxyIps)),
        "activeHttpRequests" -> ujson.Num(otherActiveHttpRequests.toDouble),
        "maxConcurrentJobs" -> ujson.Num(metrics.maxConcurrentJobs.toDouble),
        "maxQueuedJobs" -> ujson.Num(metrics.maxQueuedJobs.toDouble),
        "queuedJobs" -> ujson.Num(metrics.queuedJobs.toDouble),
        "runningJobs" -> ujson.Num(metrics.runningJobs.toDouble),
        "timedOutWorkersInFlight" -> ujson.Num(readiness.timedOutWorkersInFlight.toDouble)
      )
    )

  private def readinessStatus(
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

  private def trackActiveRequests(
      activeHttpRequests: AtomicInteger,
      delegate: HttpHandler
  ): HttpHandler =
    new HttpHandler:
      override def handle(exchange: HttpExchange): Unit =
        activeHttpRequests.incrementAndGet()
        try delegate.handle(exchange)
        finally activeHttpRequests.decrementAndGet()

  private final class JsonHandler(
      handle: HttpExchange => Either[(Int, String), JsonResponse],
      basicAuth: Option[BasicAuthConfig] = None,
      platformAuth: Option[PlatformUserAuth.Service] = None,
      authRequirement: AuthRequirement = AuthRequirement.None,
      rateLimiter: Option[RequestRateLimiter] = None,
      rateLimitBucket: Option[RateLimitBucket] = None
  ) extends HttpHandler:
    override def handle(exchange: HttpExchange): Unit =
      try
        applySecurityHeaders(exchange)
        if authorizeJson(exchange, basicAuth, platformAuth, authRequirement) &&
            ensureWithinRateLimitJson(exchange, rateLimiter, rateLimitBucket) then
          val response = handle(exchange).fold(
            { case (status, error) => JsonResponse(status, Obj("error" -> Str(error))) },
            identity
          )
          response.headers.foreach { case (name, value) =>
            exchange.getResponseHeaders.add(name, value)
          }
          writeJson(exchange, response.status, response.value)
      catch
        case NonFatal(e) =>
          writeJson(exchange, 500, Obj("error" -> Str(s"internal server error: ${e.getMessage}")))
      finally
        exchange.close()

  private final class StaticHandler(
      staticDir: Path,
      basicAuth: Option[BasicAuthConfig] = None,
      platformAuth: Option[PlatformUserAuth.Service] = None
  ) extends HttpHandler:
    override def handle(exchange: HttpExchange): Unit =
      try
        applySecurityHeaders(exchange)
        if !ensureAuthenticatedStatic(exchange, basicAuth, platformAuth) then ()
        else if !exchange.getRequestMethod.equalsIgnoreCase("GET") then
          writePlain(exchange, 405, "GET required", "text/plain; charset=utf-8")
        else
          val requestPath = Option(exchange.getRequestURI.getPath).getOrElse("/")
          val relative = if requestPath == "/" then Paths.get("index.html") else Paths.get(requestPath.dropWhile(_ == '/'))
          val resolved = staticDir.resolve(relative).normalize()
          if !resolved.startsWith(staticDir) then
            writePlain(exchange, 403, "forbidden", "text/plain; charset=utf-8")
          else
            val target =
              if Files.isDirectory(resolved) then resolved.resolve("index.html")
              else resolved
            if Files.exists(target) && Files.isRegularFile(target) then
              exchange.getResponseHeaders.set("Content-Type", contentTypeFor(target))
              exchange.sendResponseHeaders(200, Files.size(target))
              val body = exchange.getResponseBody
              val input = Files.newInputStream(target)
              try
                input.transferTo(body)
                body.flush()
              finally input.close()
            else
              writePlain(exchange, 404, "not found", "text/plain; charset=utf-8")
      catch
        case NonFatal(e) =>
          writePlain(exchange, 500, s"internal server error: ${e.getMessage}", "text/plain; charset=utf-8")
      finally
        exchange.close()

  private final class RedirectHandler(
      delegate: HttpExchange => Either[(Int, String), RedirectResponse]
  ) extends HttpHandler:
    override def handle(exchange: HttpExchange): Unit =
      try
        applySecurityHeaders(exchange)
        delegate(exchange) match
          case Left((status, error)) =>
            writePlain(exchange, status, error, "text/plain; charset=utf-8")
          case Right(response) =>
            response.headers.foreach { case (name, value) =>
              exchange.getResponseHeaders.add(name, value)
            }
            writeRedirect(exchange, response.status, response.location)
      catch
        case NonFatal(e) =>
          writePlain(exchange, 500, s"internal server error: ${e.getMessage}", "text/plain; charset=utf-8")
      finally
        exchange.close()

  private def classifyAnalysisError(error: String): Int =
    if error.startsWith("analysis timed out after") then 504
    else if error.startsWith("analysis failed:") then 500
    else 400

  private def classifyPlayingHallError(error: String): Int =
    if error.startsWith("playing hall timed out after") then 504
    else if error.startsWith("playing hall failed:") then 500
    else 400

  private def authorizeJson(
      exchange: HttpExchange,
      basicAuth: Option[BasicAuthConfig],
      platformAuth: Option[PlatformUserAuth.Service],
      authRequirement: AuthRequirement
  ): Boolean =
    exchange.setAttribute(AuthenticatedUserAttribute, null)
    platformAuth.flatMap(_.resolveSession(cookieHeader(exchange))).foreach(user =>
      exchange.setAttribute(AuthenticatedUserAttribute, user)
    )
    authRequirement match
      case AuthRequirement.None | AuthRequirement.Optional =>
        true
      case AuthRequirement.Required =>
        basicAuth match
          case Some(_) =>
            validateBasicAuth(exchange, basicAuth) match
              case None => true
              case Some(_) =>
                exchange.getResponseHeaders.set("WWW-Authenticate", BasicAuthChallenge)
                writeJson(exchange, 401, Obj("error" -> Str(AuthenticationRequiredMessage)))
                false
          case None if platformAuth.nonEmpty =>
            if authenticatedUser(exchange).nonEmpty then true
            else
              writeJson(exchange, 401, Obj("error" -> Str(SessionAuthenticationRequiredMessage)))
              false
          case None => true

  private def ensureWithinRateLimitJson(
      exchange: HttpExchange,
      rateLimiter: Option[RequestRateLimiter],
      rateLimitBucket: Option[RateLimitBucket]
  ): Boolean =
    rateLimitBucket.flatMap(bucket =>
      rateLimiter.flatMap(
        _.check(
          exchange,
          bucket,
          principalKey = authenticatedUser(exchange).map(user => s"user:${user.userId}")
        )
      )
    ) match
      case None => true
      case Some(rejection) =>
        val retryAfter = retryAfterSeconds(rejection.retryAfterMs)
        logWarn(
          s"request rate limited path=${requestPath(exchange)} client=${rejection.clientKey} bucket=${rejection.bucket.id} limitPerMinute=${rejection.limitPerMinute} retryAfterMs=${rejection.retryAfterMs}"
        )
        exchange.getResponseHeaders.set("Retry-After", retryAfter)
        writeJson(
          exchange,
          429,
          Obj(
            "error" -> Str(s"${rejection.bucket.description} rate limit exceeded; retry later"),
            "rateLimitBucket" -> Str(rejection.bucket.id),
            "limitPerMinute" -> ujson.Num(rejection.limitPerMinute.toDouble),
            "retryAfterSeconds" -> ujson.Num(retryAfter.toLong.toDouble)
          )
        )
        false

  private def ensureAuthenticatedStatic(
      exchange: HttpExchange,
      basicAuth: Option[BasicAuthConfig],
      platformAuth: Option[PlatformUserAuth.Service]
  ): Boolean =
    basicAuth match
      case Some(_) =>
        validateBasicAuth(exchange, basicAuth) match
          case None => true
          case Some(_) =>
            exchange.getResponseHeaders.set("WWW-Authenticate", BasicAuthChallenge)
            writePlain(exchange, 401, AuthenticationRequiredMessage, "text/plain; charset=utf-8")
            false
      case None =>
        platformAuth.flatMap(_.resolveSession(cookieHeader(exchange))).foreach(user =>
          exchange.setAttribute(AuthenticatedUserAttribute, user)
        )
        true

  private def validateBasicAuth(
      exchange: HttpExchange,
      basicAuth: Option[BasicAuthConfig]
  ): Option[String] =
    basicAuth.flatMap { config =>
      val authHeader = Option(exchange.getRequestHeaders.getFirst("Authorization")).map(_.trim).filter(_.nonEmpty)
      val failure =
        authHeader match
          case None => Some("missing_authorization")
          case Some(header) if !header.regionMatches(true, 0, "Basic ", 0, 6) =>
            Some("unsupported_authorization_scheme")
          case Some(header) =>
            decodeBasicCredentials(header.drop(6).trim) match
              case None => Some("malformed_authorization")
              case Some((username, password)) if secureEquals(username, config.username) && secureEquals(password, config.password) =>
                None
              case Some(_) => Some("invalid_credentials")
      failure.foreach(reason => logWarn(s"request unauthorized path=${requestPath(exchange)} remote=${remoteAddress(exchange)} reason=$reason"))
      failure
    }

  private def decodeBasicCredentials(encoded: String): Option[(String, String)] =
    if encoded.isEmpty then None
    else
      try
        val decoded = new String(Base64.getDecoder.decode(encoded), StandardCharsets.UTF_8)
        val separator = decoded.indexOf(':')
        if separator < 0 then None
        else Some(decoded.substring(0, separator) -> decoded.substring(separator + 1))
      catch
        case _: IllegalArgumentException => None

  private def secureEquals(left: String, right: String): Boolean =
    MessageDigest.isEqual(left.getBytes(StandardCharsets.UTF_8), right.getBytes(StandardCharsets.UTF_8))

  private def cookieHeader(exchange: HttpExchange): Option[String] =
    Option(exchange.getRequestHeaders.getFirst("Cookie")).map(_.trim).filter(_.nonEmpty)

  private def urlDecode(value: String): String =
    URLDecoder.decode(value, StandardCharsets.UTF_8)

  private def authenticatedUser(exchange: HttpExchange): Option[PlatformUserAuth.AuthenticatedUser] =
    Option(exchange.getAttribute(AuthenticatedUserAttribute)).collect {
      case user: PlatformUserAuth.AuthenticatedUser => user
    }

  private def ensurePlatformCsrf(
      exchange: HttpExchange,
      platformAuth: Option[PlatformUserAuth.Service]
  ): Boolean =
    platformAuth.isEmpty || authenticatedUser(exchange).forall { user =>
      Option(exchange.getRequestHeaders.getFirst("X-CSRF-Token"))
        .map(_.trim)
        .contains(user.csrfToken)
    }

  private def retryAfterSeconds(pollAfterMs: Long): String =
    math.max(1L, (pollAfterMs + 999L) / 1000L).toString

  private def shutdownExecutor(name: String, executor: ExecutorService, graceMs: Long): Unit =
    executor.shutdown()
    try
      val terminated =
        if graceMs <= 0 then false
        else executor.awaitTermination(graceMs, TimeUnit.MILLISECONDS)
      if !terminated then
        logWarn(s"$name executor did not drain within ${graceMs}ms; forcing shutdown")
        executor.shutdownNow()
    catch
      case _: InterruptedException =>
        executor.shutdownNow()
        Thread.currentThread.interrupt()

  private def awaitExecutorDrain(name: String, executor: ExecutorService, graceMs: Long): Unit =
    try
      val terminated =
        if graceMs <= 0 then executor.isTerminated
        else executor.awaitTermination(graceMs, TimeUnit.MILLISECONDS)
      if !terminated then
        logWarn(s"$name executor did not drain within ${graceMs}ms; forcing shutdown")
        executor.shutdownNow()
    catch
      case _: InterruptedException =>
        executor.shutdownNow()
        Thread.currentThread.interrupt()

  private def blockUntilShutdown(): Unit =
    val latch = new java.util.concurrent.CountDownLatch(1)
    sys.addShutdownHook(latch.countDown())
    latch.await()

  private def queueCapacity(executor: ThreadPoolExecutor): Int =
    executor.getQueue.size() + executor.getQueue.remainingCapacity()

  private[web] def shutdownDelaySeconds(graceMs: Long): Int =
    if graceMs <= 0 then 0
    else
      val roundedUpSeconds = (graceMs + 999L) / 1000L
      math.min(Int.MaxValue.toLong, roundedUpSeconds).toInt

  private def modelSource(config: HandHistoryReviewService.ServiceConfig): String =
    config.modelDir.map(_.toAbsolutePath.normalize().toString).getOrElse("uniform fallback")

  private def healthModelSource(config: HandHistoryReviewService.ServiceConfig): String =
    if config.modelDir.nonEmpty then "configured artifact dir" else "uniform fallback"

  private def authenticationMode(
      basicAuth: Option[BasicAuthConfig],
      platformAuth: Option[?]
  ): String =
    if basicAuth.nonEmpty then "basic"
    else if platformAuth.nonEmpty then "users"
    else "none"

  private def authenticationEnabled(
      basicAuth: Option[BasicAuthConfig],
      platformAuth: Option[?]
  ): Boolean =
    basicAuth.nonEmpty || platformAuth.nonEmpty

  private def requestPath(exchange: HttpExchange): String =
    Option(exchange.getRequestURI).map(_.getPath).filter(_.nonEmpty).getOrElse("/")

  private def rateLimitClientIpSource(
      trustedClientIpHeader: Option[String],
      trustedProxyIps: Set[String]
  ): String =
    trustedClientIpHeader match
      case Some(header) if trustedProxyIps.nonEmpty =>
        s"header:$header via loopback-or-allowlisted-proxy"
      case Some(header) =>
        s"header:$header via loopback-only"
      case None =>
        RateLimitClientIpSourceRemoteAddress

  private def trustedProxyIpSummary(trustedProxyIps: Set[String]): String =
    trustedProxyIps.toVector.sorted match
      case Vector() => "-"
      case values => values.mkString(",")

  private def rateLimitClientKey(
      exchange: HttpExchange,
      trustedClientIpHeader: Option[String],
      trustedProxyIps: Set[String]
  ): String =
    trustedClientIpHeader
      .filter(_ => trustsRateLimitClientIpHeader(remoteInetAddress(exchange), trustedProxyIps))
      .flatMap(headerName => forwardedClientKey(exchange, headerName).map(value => s"header:$value"))
      .getOrElse(s"remote:${clientAddressKey(exchange)}")

  private def forwardedClientKey(exchange: HttpExchange, headerName: String): Option[String] =
    Option(exchange.getRequestHeaders.get(headerName))
      .map(_.asScala.toVector.map(_.trim).filter(_.nonEmpty))
      .collect { case Vector(singleValue) if !singleValue.contains(',') => singleValue }
      .flatMap(parseTrustedClientIpLiteral)

  private[web] def parseTrustedProxyIps(raw: Option[String]): Either[String, Set[String]] =
    raw match
      case None => Right(Set.empty)
      case Some(value) =>
        val entries = value.split(",").toVector.map(_.trim).filter(_.nonEmpty)
        entries.foldLeft[Either[String, Vector[String]]](Right(Vector.empty)) { (acc, entry) =>
          for
            parsed <- acc
            normalized <- parseTrustedClientIpLiteral(entry)
              .toRight(s"--rateLimitTrustedProxyIps must contain comma-separated IP literals; invalid entry: $entry")
          yield parsed :+ normalized
        }.map(_.toSet)

  private def parseTrustedClientIpLiteral(value: String): Option[String] =
    val looksLikeIpLiteral =
      value.nonEmpty &&
        value.exists(_.isDigit) &&
        (value.contains(".") || value.contains(":")) &&
        value.forall(ch =>
          ch.isDigit ||
            ch == '.' ||
            ch == ':' ||
            ch == '%' ||
            (ch >= 'a' && ch <= 'f') ||
            (ch >= 'A' && ch <= 'F')
        )
    if !looksLikeIpLiteral then None
    else
      try
        Some(InetAddress.getByName(value).getHostAddress)
      catch
        case _: Exception => None

  private[web] def trustsRateLimitClientIpHeader(
      remoteAddress: Option[InetAddress],
      trustedProxyIps: Set[String]
  ): Boolean =
    remoteAddress.exists(address => address.isLoopbackAddress || trustedProxyIps.contains(address.getHostAddress))

  private def remoteInetAddress(exchange: HttpExchange): Option[InetAddress] =
    Option(exchange.getRemoteAddress).flatMap(address => Option(address.getAddress))

  private def clientAddressKey(exchange: HttpExchange): String =
    Option(exchange.getRemoteAddress)
      .flatMap(address => Option(address.getAddress).map(_.getHostAddress).orElse(Option(address.getHostString)))
      .filter(_.nonEmpty)
      .getOrElse("unknown")

  private def remoteAddress(exchange: HttpExchange): String =
    Option(exchange.getRemoteAddress).map(address => s"${address.getHostString}:${address.getPort}").getOrElse("-")

  private def logInfo(message: String): Unit =
    log("INFO", message, System.out)

  private def logWarn(message: String): Unit =
    log("WARN", message, System.err)

  private def logError(message: String): Unit =
    log("ERROR", message, System.err)

  private def log(level: String, message: String, stream: java.io.PrintStream): Unit =
    stream.synchronized {
      stream.println(s"[${Instant.now()}] [$level] [hand-history-review] $message")
    }

  private val usage =
    """Usage:
      |  runMain sicfun.holdem.web.HandHistoryReviewServer [--key=value ...]
      |
      |Options:
      |  --host=127.0.0.1         Bind host (falls back to HOST env)
      |  --port=8080              Bind port (falls back to PORT env)
      |  --staticDir=docs/site-preview-hybrid
      |  --maxUploadBytes=2097152 Max raw upload size in bytes (falls back to MAX_UPLOAD_BYTES env)
      |  --analysisTimeoutMs=120000 Overall timeout per analysis job; 0 disables it (falls back to ANALYSIS_TIMEOUT_MS env)
      |  --playingHallTimeoutMs=900000 Overall timeout per playing-hall job; 0 disables it (falls back to PLAYING_HALL_TIMEOUT_MS env)
      |  --maxConcurrentJobs=<n>  Concurrent analysis worker count (falls back to MAX_CONCURRENT_JOBS env)
      |  --maxQueuedJobs=<n>      Max queued analyses before 503 overload rejection (falls back to MAX_QUEUED_JOBS env)
      |  --shutdownGraceMs=5000   Grace window for draining requests/jobs on shutdown (falls back to SHUTDOWN_GRACE_MS env)
      |  --rateLimitSubmitsPerMinute=6 Submit request cap per rate-limit key per minute; 0 disables it (falls back to RATE_LIMIT_SUBMITS_PER_MINUTE env)
      |  --rateLimitStatusPerMinute=240 Job-status poll cap per rate-limit key per minute; 0 disables it (falls back to RATE_LIMIT_STATUS_PER_MINUTE env)
      |  --rateLimitClientIpHeader=<header> Optional trusted single-value client-IP header for rate limiting behind a reverse proxy (falls back to RATE_LIMIT_CLIENT_IP_HEADER env)
      |  --rateLimitTrustedProxyIps=<csv> Optional comma-separated proxy peer IP allowlist for trusting --rateLimitClientIpHeader; loopback is always trusted (falls back to RATE_LIMIT_TRUSTED_PROXY_IPS env)
      |  --drainSignalFile=<path> Optional file that makes /api/ready fail and rejects new analysis submissions while present (falls back to DRAIN_SIGNAL_FILE env)
      |  --basicAuthUser=<user>   Optional HTTP Basic auth username (falls back to BASIC_AUTH_USER env)
      |  --basicAuthPassword=<pw> Optional HTTP Basic auth password (falls back to BASIC_AUTH_PASSWORD env)
      |  --allowUnauthenticatedPublicBind=<bool> Allow non-loopback binds without auth for trusted private networks only (falls back to ALLOW_UNAUTHENTICATED_PUBLIC_BIND env)
      |  --allowInsecureUserAuth=<bool> Allow non-loopback platform-user auth without secure cookies / HTTPS OIDC callback only for trusted private-network testing (falls back to ALLOW_INSECURE_USER_AUTH env)
      |  --model=<dir>            Optional model artifact directory (falls back to MODEL_DIR env)
      |  --seed=42                RNG seed (falls back to SEED env)
      |  --bunchingTrials=200     Monte Carlo bunching trials per analysis
      |  --equityTrials=2000      Monte Carlo equity trials per analysis
      |  --budgetMs=1500          Decision budget per analyzed action
      |  --maxDecisions=12        Max decisions returned to the page
      |""".stripMargin
