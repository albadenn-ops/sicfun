package sicfun.holdem.web

import com.sun.net.httpserver.{HttpExchange, HttpServer}

import java.net.{BindException, InetSocketAddress}
import java.time.Instant
import java.util.concurrent.{ExecutorService, Executors, TimeUnit}
import java.util.concurrent.atomic.{AtomicBoolean, AtomicInteger}

import sicfun.holdem.web.AuthStack.*
import sicfun.holdem.web.HandHistoryReviewServer.{RunningServer, ServerBinding, ServerConfig}
import sicfun.holdem.web.HandHistoryReviewServerApi.*
import sicfun.holdem.web.HandHistoryReviewServerConfig.isNonLoopbackBindHost
import sicfun.holdem.web.JobQueue.*
import sicfun.holdem.web.RateLimit.*
import sicfun.holdem.web.Readiness.*

/** HTTP server lifecycle and route wiring for [[HandHistoryReviewServer]]. */
private[web] object HandHistoryReviewServerRuntime:
  def startServer(
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
      authPerMinute = config.rateLimitAuthPerMinute,
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
          new JsonHandler(exchange =>
            // Accept GET and HEAD on the health endpoint. Monitoring tools
            // commonly probe with HEAD to skip the body; writeBytes is
            // HEAD-aware now and suppresses the body while still emitting
            // the headers, so HEAD returns 200 + Content-Length matching
            // what GET would send.
            val method = exchange.getRequestMethod
            if method.equalsIgnoreCase("OPTIONS") then Right(optionsResponse("GET, HEAD"))
            else if !method.equalsIgnoreCase("GET") && !method.equalsIgnoreCase("HEAD") then
              Right(methodNotAllowed("GET, HEAD"))
            else
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
          new JsonHandler(exchange =>
            val method = exchange.getRequestMethod
            if method.equalsIgnoreCase("OPTIONS") then Right(optionsResponse("GET, HEAD"))
            else if !method.equalsIgnoreCase("GET") && !method.equalsIgnoreCase("HEAD") then
              Right(methodNotAllowed("GET, HEAD"))
            else
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
          new StaticAssetsHandler(config.staticDir, basicAuth = config.basicAuth, platformAuth = platformAuthService)
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
            authRequirement = AuthRequirement.Optional,
            rateLimiter = Some(rateLimiter),
            rateLimitBucket = Some(RateLimitBucket.Auth)
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
            authRequirement = AuthRequirement.Optional,
            rateLimiter = Some(rateLimiter),
            rateLimitBucket = Some(RateLimitBucket.Auth)
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
        s"startup complete host=${binding.host} port=${binding.port} modelSource=${binding.modelSource} maxUploadBytes=${config.maxUploadBytes} analysisTimeoutMs=${config.analysisTimeoutMs} playingHallTimeoutMs=${config.playingHallTimeoutMs} maxConcurrentJobs=${config.maxConcurrentJobs} maxQueuedJobs=${config.maxQueuedJobs} rateLimitSubmitsPerMinute=${config.rateLimitSubmitsPerMinute} rateLimitStatusPerMinute=${config.rateLimitStatusPerMinute} rateLimitAuthPerMinute=${config.rateLimitAuthPerMinute} rateLimitClientIpSource=${rateLimitClientIpSource(config.rateLimitClientIpHeader, config.rateLimitTrustedProxyIps)} rateLimitTrustedProxyIps=${trustedProxyIpSummary(config.rateLimitTrustedProxyIps)} drainSignalFile=${config.drainSignalFile.map(_.toAbsolutePath.normalize().toString).getOrElse("-")} authenticationMode=${authenticationMode(config.basicAuth, config.platformAuth)}"
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

  def blockUntilShutdown(): Unit =
    val latch = new java.util.concurrent.CountDownLatch(1)
    sys.addShutdownHook(latch.countDown())
    latch.await()

  private[web] def shutdownDelaySeconds(graceMs: Long): Int =
    if graceMs <= 0 then 0
    else
      val roundedUpSeconds = (graceMs + 999L) / 1000L
      math.min(Int.MaxValue.toLong, roundedUpSeconds).toInt

  private def modelSource(config: HandHistoryReviewService.ServiceConfig): String =
    config.modelDir.map(_.toAbsolutePath.normalize().toString).getOrElse("uniform fallback")

  def healthModelSource(config: HandHistoryReviewService.ServiceConfig): String =
    if config.modelDir.nonEmpty then "configured artifact dir" else "uniform fallback"

  def logInfo(message: String): Unit =
    log("INFO", message, System.out)

  def logWarn(message: String): Unit =
    log("WARN", message, System.err)

  def logError(message: String): Unit =
    log("ERROR", message, System.err)

  /** Log an unhandled exception thrown from an HTTP handler. Logs request
    * method, path, exception class, and message at ERROR level (sanitized
    * via [[log]]) and then dumps the full stack trace to stderr so the same
    * NSSM log stream captures it. Used by handler-level catch blocks that
    * MUST avoid echoing the exception message back to the client (RFC-7807
    * "internal server error" is the only public detail). */
  private[web] def logHandlerException(
      exchange: HttpExchange,
      e: Throwable,
      label: String
  ): Unit =
    val method = Option(exchange.getRequestMethod).getOrElse("?")
    // getRawPath -- not getPath -- so percent-encoded sequences stay encoded
    // and a literal space in the URL does not split the `path=...` field at
    // the wrong column. Matches the requestPath helper in AuthStack.
    val path = Option(exchange.getRequestURI).map(_.getRawPath).getOrElse("?")
    val message = Option(e.getMessage).getOrElse("")
    logError(s"$label method=$method path=$path exception=${e.getClass.getName} message=$message")
    e.printStackTrace()

  /** Escapes control characters inside a log message so that user-controlled
    * values flowing into a log line cannot forge fake log entries or confuse
    * line-oriented tools. Log lines use spaces (0x20) as field separators and
    * printable ASCII for keys/values; any C0 control char in a value (request
    * path, remote address, job id, error string) is potentially line-eating
    * or display-breaking.
    *
    * Recognised escapes: `\` `\n` `\r` `\0` `\t` (the common cases). Other C0
    * control chars (0x01-0x08, 0x0B-0x0C, 0x0E-0x1F) and DEL (0x7F) are
    * rendered as `\xHH` (lowercase hex) so operators see something readable
    * rather than an invisible glyph. Backslash is handled first so the rest
    * of the escapes are unambiguous. */
  private[web] def sanitizeLogMessage(message: String): String =
    val intermediate = message
      .replace("\\", "\\\\")
      .replace("\n", "\\n")
      .replace("\r", "\\r")
      .replace("\u0000", "\\0")
      .replace("\t", "\\t")
    val sb = new StringBuilder(intermediate.length)
    intermediate.foreach { ch =>
      if (ch.toInt < 0x20 && ch != '\\') || ch.toInt == 0x7F then
        sb.append("\\x%02x".format(ch.toInt))
      else sb.append(ch)
    }
    sb.toString

  private def log(level: String, message: String, stream: java.io.PrintStream): Unit =
    stream.synchronized {
      stream.println(s"[${Instant.now()}] [$level] [hand-history-review] ${sanitizeLogMessage(message)}")
    }
