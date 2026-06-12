package sicfun.holdem.web

import com.sun.net.httpserver.{HttpExchange, HttpHandler, HttpServer}

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
        playingHallTimeoutMs = config.playingHallTimeoutMs
      )
      val activeHttpRequests = new AtomicInteger(0)
      // Single wrapper applied to every registered handler. Tracks active-request
      // counts AND stashes the audit-display client address so behind-a-proxy
      // audit logs show the resolved client IP rather than the proxy peer.
      def tracked(delegate: HttpHandler): HttpHandler =
        trackActiveRequests(
          activeHttpRequests,
          config.rateLimitClientIpHeader,
          config.rateLimitTrustedProxyIps,
          delegate
        )
      val server = HttpServer.create(new InetSocketAddress(config.host, config.port), 0)
      server.createContext(
        "/api/health",
        tracked(
          new JsonHandler(exchange =>
            // Accept GET and HEAD on the health endpoint. Monitoring tools
            // commonly probe with HEAD to skip the body; writeBytes is
            // HEAD-aware and suppresses the body while still emitting the
            // status code plus Content-Type, Cache-Control, Vary, and the
            // security headers a GET would carry, so a HEAD probe sees
            // the same readiness signal (200 while up, 503 on /api/ready
            // when draining / queue-full / timed-out-worker). Content-
            // Length is NOT sent on HEAD: the JDK HttpExchange contract
            // requires sendResponseHeaders(status, -1L) for HEAD, which
            // suppresses the length header. Content-Encoding is also
            // absent from the HEAD response, but for a different reason
            // -- writeBytes only sets it inside the GET-with-gzip branch
            // since the empty-body HEAD response wouldn't be gzipped
            // anyway. (StaticAssetsHandler takes the stricter approach
            // and DOES advertise Content-Encoding on HEAD via
            // wouldCompressIfGet for HEAD-then-GET cache parity, which
            // proves the JDK contract isn't the constraint here -- it
            // happily emits Content-Encoding alongside a -1L body.) RFC
            // 7231 sec 3.3 explicitly allows payload-header fields to
            // be omitted on HEAD, so this is compliant; a monitor that
            // needs the exact body size should fetch via GET.
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
                  draining,
                  platformAuthService
                )
              )
          )
        )
      )
      server.createContext(
        "/api/ready",
        tracked(
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
        tracked(
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
        tracked(
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
        tracked(
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
        tracked(
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
        tracked(
          new StaticAssetsHandler(config.staticDir, basicAuth = config.basicAuth, platformAuth = platformAuthService)
        )
      )
      server.createContext(
        AuthMePath,
        tracked(
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
        tracked(
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
        tracked(
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
        tracked(
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
        tracked(
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
            tracked(
              new RedirectHandler(exchange => handleOidcStart(exchange, service))
            )
          )
        }
        config.platformAuth.toVector.flatMap(_.oidcProviders).foreach { provider =>
          server.createContext(
            provider.callbackPath,
            tracked(
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
          // Compute the deadline ONCE up front and let each step claim the
          // wall time left until that deadline. Previously every step received
          // the same `remainingGraceMs` value, so a 5s budget could become a
          // 5+5+5+5 = 20s shutdown if every executor took the full window --
          // not what operators expect from `shutdownGraceMs`.
          val deadlineNanos = shutdownStartedAt + TimeUnit.MILLISECONDS.toNanos(config.shutdownGraceMs)
          def remainingMs(): Long =
            math.max(0L, TimeUnit.NANOSECONDS.toMillis(deadlineNanos - System.nanoTime()))
          try server.stop(httpDrainSeconds)
          finally
            shutdownExecutor("http", serverExecutor, remainingMs())
            awaitExecutorDrain("analysis-timeout", analysisTimeoutExecutor, remainingMs())
            awaitExecutorDrain("analysis", analysisExecutor, remainingMs())
            logInfo(s"shutdown complete host=${binding.host} port=${binding.port}")
      sys.addShutdownHook(shutdown())
      val userAuthMaxUsersField = config.platformAuth.map(_.maxUsers.toString).getOrElse("-")
      // At boot we have JUST loaded the user store; emit the current count so
      // operators can verify the persistent store survived restart and see
      // capacity headroom vs maxUsers without having to hit /api/health
      // first. `-` for basic-auth / no-auth deployments (no store to count).
      val userAuthStoredUsersField = platformAuthService.map(_.storedUserCount.toString).getOrElse("-")
      // %20-escape spaces in path-like fields so a Windows path such as
      // "C:\Program Files\model" or a drain-signal file in a user home
      // dir like "C:\Users\Alex Smith\drain.flag" does not split the
      // structured key=value pairs of the startup banner. Same shape as
      // formatSubmittedEmailForLog and the analyze heroName log field.
      val loggedModelSource = binding.modelSource.replace(" ", "%20")
      val loggedDrainSignalFile = config.drainSignalFile
        .map(_.toAbsolutePath.normalize().toString.replace(" ", "%20"))
        .getOrElse("-")
      // rateLimitClientIpSource can return values like "header:X-Real-IP via
      // loopback-only" with internal spaces -- same key=value split hazard.
      // The JSON form in /api/health doesn't need this escape (JSON
      // quoting handles spaces); only the log line does.
      val loggedRateLimitClientIpSource =
        rateLimitClientIpSource(config.rateLimitClientIpHeader, config.rateLimitTrustedProxyIps).replace(" ", "%20")
      logInfo(
        s"startup complete host=${binding.host} port=${binding.port} modelSource=$loggedModelSource maxUploadBytes=${config.maxUploadBytes} analysisTimeoutMs=${config.analysisTimeoutMs} playingHallTimeoutMs=${config.playingHallTimeoutMs} maxConcurrentJobs=${config.maxConcurrentJobs} maxQueuedJobs=${config.maxQueuedJobs} rateLimitSubmitsPerMinute=${config.rateLimitSubmitsPerMinute} rateLimitStatusPerMinute=${config.rateLimitStatusPerMinute} rateLimitAuthPerMinute=${config.rateLimitAuthPerMinute} rateLimitClientIpSource=$loggedRateLimitClientIpSource rateLimitTrustedProxyIps=${trustedProxyIpSummary(config.rateLimitTrustedProxyIps)} drainSignalFile=$loggedDrainSignalFile authenticationMode=${authenticationMode(config.basicAuth, config.platformAuth)} userAuthMaxUsers=$userAuthMaxUsersField userAuthStoredUsers=$userAuthStoredUsersField"
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
    // the wrong column. Matches the requestPath helper in AuthStack, and
    // applies the same 512-char cap so an attacker with a multi-KB URL
    // cannot blow up the exception-log path= field. sanitizeLogMessage's
    // 8 KB final cap would catch the whole line otherwise, but per-field
    // capping keeps the truncation marker in the right spot for forensics
    // and preserves the trailing `exception=` and `message=` fields.
    val rawPath = Option(exchange.getRequestURI).map(_.getRawPath).getOrElse("?")
    val path = if rawPath.length <= 512 then rawPath else rawPath.substring(0, 512) + "...(truncated)"
    val message = Option(e.getMessage).getOrElse("")
    logError(s"$label method=$method path=$path exception=${e.getClass.getName} message=$message")
    // Render the stack trace as a string and route each frame through the
    // same sanitized log path so a user-controlled value that ends up in
    // an exception message (e.g. a JSON parse error that surfaces the
    // offending byte at a position the parser quotes back) cannot inject
    // forged log lines. The raw `e.printStackTrace()` we used before wrote
    // straight to System.err bypassing sanitizeLogMessage and bypassing
    // the per-write synchronized block, so a malicious payload combined
    // with a concurrent legitimate log line could fabricate log entries
    // and confuse line-oriented log tools.
    val rendered = new java.io.StringWriter()
    e.printStackTrace(new java.io.PrintWriter(rendered))
    rendered.toString.split('\n').iterator.map(_.stripSuffix("\r")).filter(_.nonEmpty).foreach { line =>
      logError(s"$label stack=$line")
    }

  private val MaxSanitizedLogMessageLength = 8 * 1024

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
    * of the escapes are unambiguous.
    *
    * After escaping, the rendered message is clamped at 8 KB with a
    * "...(truncated)" marker. Per-field caps (submitted email in auth audit
    * lines, OIDC ?error=/?state=/?code= in callback handling) remain the
    * primary defense because their truncation markers land in the right
    * spot for forensics; this is the last-resort cap for any input source
    * that slipped past the per-field caps. */
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
    val rendered = sb.toString
    // Hard cap as last-resort defense against log-line inflation. Per-field
    // caps (submitted email, OIDC ?error=/?state=/?code=) are the primary
    // discipline -- their truncation markers land in the right spot for
    // forensics -- but this catches sources we did not anticipate. 8 KB is
    // generous: legitimate sanitized messages never approach it, and stack
    // traces are processed line-by-line in logHandlerException so individual
    // frames stay well under.
    if rendered.length <= MaxSanitizedLogMessageLength then rendered
    else rendered.substring(0, MaxSanitizedLogMessageLength) + "...(truncated)"

  private def log(level: String, message: String, stream: java.io.PrintStream): Unit =
    stream.synchronized {
      stream.println(s"[${Instant.now()}] [$level] [hand-history-review] ${sanitizeLogMessage(message)}")
    }
