package sicfun.holdem.web

import sicfun.holdem.cli.CliHelpers

import java.net.URI
import java.nio.file.{Files, Path, Paths}
import java.util.Locale
import scala.util.control.NonFatal

import sicfun.holdem.web.HandHistoryReviewServer.{BasicAuthConfig, ServerConfig}
import sicfun.holdem.web.RateLimit.parseTrustedProxyIps

/** Command-line and environment configuration parsing for [[HandHistoryReviewServer]]. */
private[web] object HandHistoryReviewServerConfig:
  private val DefaultAnalysisTimeoutMs = 120000L
  private val DefaultPlayingHallTimeoutMs = 900000L
  private val DefaultShutdownGraceMs = 5000L
  private val DefaultRateLimitSubmitsPerMinute = 6
  private val DefaultRateLimitStatusPerMinute = 240
  // 10 attempts/min/IP leaves room for a few legitimate fat-finger retries but
  // throttles a credential-stuffing attacker to ~600 attempts/hour — well below
  // what is useful for a dictionary attack against PBKDF2-hashed credentials.
  private val DefaultRateLimitAuthPerMinute = 10

  def parseArgs(args: Array[String]): Either[String, ServerConfig] =
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
        rateLimitAuthPerMinute <- resolveIntOption(
          options,
          "rateLimitAuthPerMinute",
          env("RATE_LIMIT_AUTH_PER_MINUTE"),
          DefaultRateLimitAuthPerMinute
        )
        _ <- Either.cond(
          rateLimitAuthPerMinute >= 0,
          (),
          "--rateLimitAuthPerMinute must be zero or positive"
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
        userAuthMaxUsers <- resolveIntOption(
          options,
          "userAuthMaxUsers",
          env("USER_AUTH_MAX_USERS"),
          100_000
        )
        _ <- Either.cond(userAuthMaxUsers > 0, (), "--userAuthMaxUsers must be positive")
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
          oidcProviders = googleOidc.toVector,
          maxUsers = userAuthMaxUsers
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
        rateLimitAuthPerMinute = rateLimitAuthPerMinute,
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
        parseAbsoluteHttpUri(redirectUri, "--googleOidcRedirectUri/GOOGLE_OIDC_REDIRECT_URI").flatMap { uri =>
          val provider = new PlatformUserAuth.GoogleOidcProvider(
            PlatformUserAuth.GoogleOidcConfig(
              clientId = clientId,
              clientSecret = clientSecret,
              redirectUri = redirectUri
            )
          )
          // The server's runtime registers exactly one context per provider at
          // `provider.callbackPath` (e.g. `/api/auth/oidc/google/callback`).
          // A redirect URI whose path does not match that means Google would
          // send the user to a path the server never registered -- the static
          // handler's catch-all returns a generic 404 with no breadcrumb that
          // OIDC was involved. Catch the mismatch at config time so the
          // operator sees a precise error in startup logs.
          val redirectPath = Option(uri.getPath).getOrElse("")
          if redirectPath != provider.callbackPath then
            Left(
              s"--googleOidcRedirectUri/GOOGLE_OIDC_REDIRECT_URI path must be '${provider.callbackPath}' so the server's registered callback handler matches the redirect URI Google sees; got '$redirectPath'"
            )
          else Right(Some(provider))
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
      oidcProviders: Vector[PlatformUserAuth.OidcProvider],
      maxUsers: Int
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
              oidcProviders = oidcProviders,
              maxUsers = maxUsers
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

  def isNonLoopbackBindHost(host: String): Boolean =
    val normalized = host.trim.stripPrefix("[").stripSuffix("]").toLowerCase(Locale.ROOT)
    normalized match
      case "" => false
      case "localhost" => false
      case "::1" => false
      case "0:0:0:0:0:0:0:1" => false
      case value if value.startsWith("127.") => false
      case _ => true

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
      |  --rateLimitAuthPerMinute=10 Auth (register/login) cap per rate-limit key per minute; 0 disables it (falls back to RATE_LIMIT_AUTH_PER_MINUTE env)
      |  --rateLimitClientIpHeader=<header> Optional trusted single-value client-IP header for rate limiting behind a reverse proxy (falls back to RATE_LIMIT_CLIENT_IP_HEADER env)
      |  --rateLimitTrustedProxyIps=<csv> Optional comma-separated proxy peer IP allowlist for trusting --rateLimitClientIpHeader; loopback is always trusted (falls back to RATE_LIMIT_TRUSTED_PROXY_IPS env)
      |  --drainSignalFile=<path> Optional file that makes /api/ready fail and rejects new analysis submissions while present (falls back to DRAIN_SIGNAL_FILE env)
      |  --basicAuthUser=<user>   Optional HTTP Basic auth username (falls back to BASIC_AUTH_USER env)
      |  --basicAuthPassword=<pw> Optional HTTP Basic auth password (falls back to BASIC_AUTH_PASSWORD env)
      |  --allowUnauthenticatedPublicBind=<bool> Allow non-loopback binds without auth for trusted private networks only (falls back to ALLOW_UNAUTHENTICATED_PUBLIC_BIND env)
      |  --allowInsecureUserAuth=<bool> Allow non-loopback platform-user auth without secure cookies / HTTPS OIDC callback only for trusted private-network testing (falls back to ALLOW_INSECURE_USER_AUTH env)
      |  --userAuthMaxUsers=100000 Hard cap on the total number of stored users; further registrations return 'temporarily unavailable' (falls back to USER_AUTH_MAX_USERS env)
      |  --model=<dir>            Optional model artifact directory (falls back to MODEL_DIR env)
      |  --seed=42                RNG seed (falls back to SEED env)
      |  --bunchingTrials=200     Monte Carlo bunching trials per analysis
      |  --equityTrials=2000      Monte Carlo equity trials per analysis
      |  --budgetMs=1500          Decision budget per analyzed action
      |  --maxDecisions=12        Max decisions returned to the page
      |""".stripMargin
