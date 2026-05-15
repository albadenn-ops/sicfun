package sicfun.holdem.web

import com.sun.net.httpserver.{HttpExchange, HttpHandler}

import java.net.URLDecoder
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.Base64
import scala.util.control.NonFatal

import ujson.{Arr, Obj, Str}

import sicfun.holdem.web.HandHistoryReviewServer.BasicAuthConfig
import sicfun.holdem.web.HandHistoryReviewServerApi.{
  JsonResponse,
  methodNotAllowed,
  optionsResponse,
  optionalString,
  readRequestBody,
  requiredString,
  retryAfterSeconds
}
import sicfun.holdem.web.HandHistoryReviewServerRuntime.{logHandlerException, logInfo, logWarn}
import sicfun.holdem.web.RateLimit.{RateLimitBucket, RequestRateLimiter}
import sicfun.holdem.web.WebResponses.*

private[web] object AuthStack:
  val AuthMePath = "/api/auth/me"
  val AuthRegisterPath = "/api/auth/register"
  val AuthLoginPath = "/api/auth/login"
  val AuthLogoutPath = "/api/auth/logout"
  val AuthProfilePath = "/api/auth/profile"

  private val BasicAuthRealm = "sicfun-hand-history-review"
  private val BasicAuthChallenge = s"""Basic realm="$BasicAuthRealm", charset="UTF-8""""
  val AuthenticationRequiredMessage = "authentication required"
  private val SessionAuthenticationRequiredMessage = "sign in required"
  val SessionCsrfRequiredMessage = "missing or invalid csrf token"
  private val AuthenticatedUserAttribute = "sicfun.hand-history.authenticated-user"
  // Stashed by the request wrapper (see `Readiness.trackActiveRequests`) before any
  // handler runs, so audit log fields like `remote=` see the SAME client identity
  // the rate limiter keys on -- a trusted X-Forwarded-For IP when the deployment is
  // behind a proxy in the trustedProxyIps allowlist, the direct TCP peer otherwise.
  // Without this, a reverse-proxied deployment would correctly rate-limit by client
  // IP but its audit log would show every request coming from the proxy.
  private[web] val AuditClientAddressAttribute = "sicfun.audit.client-address"

  def handleAuthMe(
      exchange: HttpExchange,
      basicAuth: Option[BasicAuthConfig],
      platformAuth: Option[PlatformUserAuth.Service]
  ): Either[(Int, String), JsonResponse] =
    if exchange.getRequestMethod.equalsIgnoreCase("OPTIONS") then Right(optionsResponse("GET"))
    else if !exchange.getRequestMethod.equalsIgnoreCase("GET") then Right(methodNotAllowed("GET"))
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

  def handleAuthRegister(
      exchange: HttpExchange,
      platformAuth: Option[PlatformUserAuth.Service]
  ): Either[(Int, String), JsonResponse] =
    if exchange.getRequestMethod.equalsIgnoreCase("OPTIONS") then Right(optionsResponse("POST"))
    else if !exchange.getRequestMethod.equalsIgnoreCase("POST") then Right(methodNotAllowed("POST"))
    else if authenticatedUser(exchange).nonEmpty then Left(409 -> "already signed in")
    else
      platformAuth match
        case None => Left(404 -> "user auth is not enabled")
        case Some(service) =>
          readRequestBody(exchange, 16 * 1024)
            .flatMap(parseRegisterRequest)
            .flatMap { case (email, password, displayName) =>
              service.registerLocal(email, password, displayName) match
                case Right(result) =>
                  // Log the canonical (normalized) email from the stored record so
                  // operators grepping for a user see a single consistent value
                  // across login/logout/profile events regardless of input casing.
                  logInfo(s"auth.register.success email=${result.user.email} remote=${remoteAddress(exchange)}")
                  Right(result)
                case Left(error) =>
                  // Failure path logs the SUBMITTED email -- there may be no
                  // canonical user, and operators want to see exactly what the
                  // attacker typed (which may differ from the stored email).
                  // Replace ASCII spaces with %20 so a submitted email like
                  // "alice bob@example.com" (which validateEmail rejects) does
                  // not split the structured key=value log fields.
                  logWarn(s"auth.register.failure email=${email.replace(" ", "%20")} remote=${remoteAddress(exchange)} reason=$error")
                  Left(400 -> error)
            }
            .map(result => loginJsonResponse(service, result, status = 201))

  def handleAuthLogin(
      exchange: HttpExchange,
      platformAuth: Option[PlatformUserAuth.Service]
  ): Either[(Int, String), JsonResponse] =
    if exchange.getRequestMethod.equalsIgnoreCase("OPTIONS") then Right(optionsResponse("POST"))
    else if !exchange.getRequestMethod.equalsIgnoreCase("POST") then Right(methodNotAllowed("POST"))
    else if authenticatedUser(exchange).nonEmpty then Left(409 -> "already signed in")
    else
      platformAuth match
        case None => Left(404 -> "user auth is not enabled")
        case Some(service) =>
          readRequestBody(exchange, 16 * 1024)
            .flatMap(parseLoginRequest)
            .flatMap { case (email, password) =>
              service.loginLocal(email, password) match
                case Right(result) =>
                  // Log the canonical (normalized) email so operators grepping
                  // for a user see a single value across all of their events,
                  // regardless of how they typed their email at the form.
                  logInfo(s"auth.login.success email=${result.user.email} remote=${remoteAddress(exchange)}")
                  Right(result)
                case Left(error) =>
                  // Email is what the attacker SUBMITTED, not a confirmed account;
                  // log it so operators can spot brute-force patterns (e.g. many
                  // failures from one IP across many emails, or many failures
                  // from many IPs against one email). %20-escape ASCII spaces
                  // so a probe with embedded whitespace doesn't split the
                  // structured key=value log fields.
                  logWarn(s"auth.login.failure email=${email.replace(" ", "%20")} remote=${remoteAddress(exchange)} reason=$error")
                  Left(401 -> error)
            }
            .map(result => loginJsonResponse(service, result, status = 200))

  def handleAuthLogout(
      exchange: HttpExchange,
      platformAuth: Option[PlatformUserAuth.Service]
  ): Either[(Int, String), JsonResponse] =
    if exchange.getRequestMethod.equalsIgnoreCase("OPTIONS") then Right(optionsResponse("POST"))
    else if !exchange.getRequestMethod.equalsIgnoreCase("POST") then Right(methodNotAllowed("POST"))
    else if !ensurePlatformCsrf(exchange, platformAuth) then Left(403 -> SessionCsrfRequiredMessage)
    else
      platformAuth match
        case None => Left(404 -> "user auth is not enabled")
        case Some(service) =>
          val email = authenticatedUser(exchange).map(_.email).getOrElse("-")
          val clearedCookie = service.revokeSession(cookieHeader(exchange))
          logInfo(s"auth.logout email=$email remote=${remoteAddress(exchange)}")
          Right(
            JsonResponse(
              200,
              service.authenticationState(None),
              headers = Vector("Set-Cookie" -> clearedCookie)
            )
          )

  def handleAuthProfile(
      exchange: HttpExchange,
      platformAuth: Option[PlatformUserAuth.Service]
  ): Either[(Int, String), JsonResponse] =
    if exchange.getRequestMethod.equalsIgnoreCase("OPTIONS") then Right(optionsResponse("POST"))
    else if !exchange.getRequestMethod.equalsIgnoreCase("POST") then Right(methodNotAllowed("POST"))
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

  def handleOidcStart(
      exchange: HttpExchange,
      platformAuth: PlatformUserAuth.Service
  ): Either[(Int, String), RedirectResponse] =
    if !exchange.getRequestMethod.equalsIgnoreCase("GET") then
      exchange.getResponseHeaders.set("Allow", "GET")
      Left(405 -> "GET required")
    else
      extractOidcProviderId(exchange, "/start").flatMap { providerId =>
        platformAuth.startOidc(providerId) match
          case Left(error) =>
            logWarn(s"auth.oidc.start.failure provider=$providerId remote=${remoteAddress(exchange)} reason=$error")
            Left(400 -> error)
          case Right(start) =>
            // INFO not WARN -- this is normal user behavior, but the log entry
            // lets operators correlate a later auth.oidc.success/failure with
            // the start so a missing callback (user abandoned the flow,
            // provider error, etc.) is visible.
            logInfo(s"auth.oidc.start provider=$providerId remote=${remoteAddress(exchange)}")
            // Bind the random state value to the user agent via a short-lived
            // cookie so a stolen state cannot be replayed by a different
            // browser (OAuth 2.0 BCP covert-redirect mitigation). The cookie
            // is required at the callback step.
            Right(RedirectResponse(
              location = start.location,
              headers = Vector("Set-Cookie" -> start.stateCookieHeader)
            ))
      }

  def handleOidcCallback(
      exchange: HttpExchange,
      platformAuth: PlatformUserAuth.Service,
      providerId: String
  ): Either[(Int, String), RedirectResponse] =
    if !exchange.getRequestMethod.equalsIgnoreCase("GET") then
      exchange.getResponseHeaders.set("Allow", "GET")
      Left(405 -> "GET required")
    else
      val query = parseQuery(exchange)
      query.get("error") match
        case Some(error) =>
          // The OIDC provider rejected the authorization (user denied consent,
          // expired code, etc.). Log so an unusual burst of provider-side
          // failures is visible alongside our own auth.oidc.failure entries.
          // No state cookie clear here: the cookie has a short Max-Age (~10
          // min) and is HttpOnly + same-origin, so leaving it lets a follow-up
          // /start reissue cleanly without needing the failure path to write
          // multiple Set-Cookie headers (which complicates failure-mode tests
          // that grep `Set-Cookie` for the session cookie's presence).
          logWarn(s"auth.oidc.failure provider=$providerId remote=${remoteAddress(exchange)} reason=provider-error:$error")
          Right(RedirectResponse(location = PlatformUserAuth.oidcFailureRedirect(error)))
        case None =>
          (query.get("state"), query.get("code")) match
            case (Some(state), Some(code)) =>
              // OAuth 2.0 BCP "covert-redirect" / login-CSRF mitigation: the
              // browser that arrives at /callback must carry the SAME state
              // value that we Set-Cookie'd at /start. Without this check, an
              // attacker who finished their own authorization could forward
              // their `?state=X&code=ATTACKER_CODE` to a victim, and our
              // OidcStateStore (which only knows that X is a state WE issued)
              // would happily exchange the code and bind the attacker's
              // identity to the victim's browser session.
              val expectedName = platformAuth.expectedOidcStateCookieName
              val cookieState = extractCookieFromExchange(exchange, expectedName)
              if cookieState.isEmpty || !cookieState.exists(value => secureEquals(value, state)) then
                val reason = if cookieState.isEmpty then "missing_state_cookie" else "state_cookie_mismatch"
                logWarn(s"auth.oidc.failure provider=$providerId remote=${remoteAddress(exchange)} reason=$reason")
                Right(RedirectResponse(location = PlatformUserAuth.oidcFailureRedirect(reason)))
              else
                platformAuth.finishOidc(providerId, state, code) match
                  case Left(error) =>
                    logWarn(s"auth.oidc.failure provider=$providerId remote=${remoteAddress(exchange)} reason=$error")
                    Right(RedirectResponse(location = PlatformUserAuth.oidcFailureRedirect(error)))
                  case Right(result) =>
                    logInfo(s"auth.oidc.success provider=$providerId email=${result.user.email} remote=${remoteAddress(exchange)}")
                    Right(
                      RedirectResponse(
                        location = PlatformUserAuth.oidcSuccessRedirect,
                        // Two Set-Cookie headers: install the session cookie
                        // AND clear the now-consumed state cookie. JsonHandler/
                        // RedirectHandler use `headers.add(...)` so both make
                        // it onto the wire.
                        headers = Vector(
                          "Set-Cookie" -> result.cookieHeader,
                          "Set-Cookie" -> platformAuth.oidcStateClearCookieHeader
                        )
                      )
                    )
            case _ =>
              logWarn(s"auth.oidc.failure provider=$providerId remote=${remoteAddress(exchange)} reason=missing_code_or_state")
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


  final case class RedirectResponse(
      location: String,
      headers: Vector[(String, String)] = Vector.empty,
      status: Int = 302
  )


  enum AuthRequirement:
    case None, Optional, Required

  final class JsonHandler(
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
          // RFC 7231 sec 6.6.4: 503 responses SHOULD include Retry-After so clients
          // back off intelligently rather than guessing. Handlers that have a more
          // specific value (e.g. job-store admission rejections compute one from the
          // poll-after hint) set it explicitly via the headers list; this fallback
          // covers the rest.
          if response.status == 503 && exchange.getResponseHeaders.getFirst("Retry-After") == null then
            exchange.getResponseHeaders.add("Retry-After", "5")
          writeJson(exchange, response.status, response.value)
      catch
        case NonFatal(e) =>
          // Log the exception with full stack trace server-side; respond with a
          // generic message so we don't leak internal class names, file paths,
          // or other implementation detail back to clients via e.getMessage.
          logHandlerException(exchange, e, "unhandled exception in JsonHandler")
          // If the original write already called sendResponseHeaders, the fallback
          // write will throw "headers already sent" -- swallow it so the catch
          // does not leak a second unhandled exception to the HTTP server frame.
          try writeJson(exchange, 500, Obj("error" -> Str("internal server error")))
          catch case NonFatal(_) => ()
      finally
        exchange.close()

  final class RedirectHandler(
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
          logHandlerException(exchange, e, "unhandled exception in RedirectHandler")
          try writePlain(exchange, 500, "internal server error", "text/plain; charset=utf-8")
          catch case NonFatal(_) => ()
      finally
        exchange.close()

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
            // validateBasicAuth logs the failure reason itself, so the JSON
            // path does not also emit a `request unauthorized` line.
            validateBasicAuth(exchange, basicAuth) match
              case None => true
              case Some(_) =>
                exchange.getResponseHeaders.set("WWW-Authenticate", BasicAuthChallenge)
                writeJson(exchange, 401, Obj("error" -> Str(AuthenticationRequiredMessage)))
                false
          case None if platformAuth.nonEmpty =>
            if authenticatedUser(exchange).nonEmpty then true
            else
              logWarn(s"request unauthorized path=${requestPath(exchange)} remote=${remoteAddress(exchange)} reason=session-missing-or-invalid")
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

  def ensureAuthenticatedStatic(
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
              case Some((username, password)) =>
                // Use the non-short-circuit `&` so both secureEquals calls
                // run regardless of the first result. With `&&` an attacker
                // could distinguish "wrong username" (only username compared)
                // from "username matches, password wrong" (both compared) by
                // timing -- enough to recover the username over enough
                // probes even when each individual compare is constant-time.
                val usernameOk = secureEquals(username, config.username)
                val passwordOk = secureEquals(password, config.password)
                if usernameOk & passwordOk then None
                else Some("invalid_credentials")
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

  // Parses the Cookie request header into Option[value] for the named cookie.
  // Mirrors the same shape used by PlatformUserAuth.extractCookie -- split on
  // `;`, trim, find the FIRST NON-EMPTY value among segments starting with
  // `<name>=`. The "non-empty among many" detail matters because RFC 6265
  // permits multiple cookies with the same name and leaves ordering
  // implementation-defined, so an attacker who can plant a cookie on a
  // sibling subdomain (which the `__Host-` prefix prevents in cookieSecure
  // mode but NOT in plain-HTTP mode) could otherwise pin `sicfun_session=`
  // with an empty value as the first segment and effectively log the victim
  // out by hiding the real cookie that comes later. Skipping empty matches
  // and continuing the scan defeats that specific DoS without relying on
  // browser cookie-ordering quirks.
  private def extractCookieFromExchange(exchange: HttpExchange, cookieName: String): Option[String] =
    cookieHeader(exchange).flatMap { header =>
      header.split(';').iterator
        .map(_.trim)
        .filter(_.startsWith(s"$cookieName="))
        .map(_.substring(cookieName.length + 1))
        .find(_.nonEmpty)
    }

  private def urlDecode(value: String): String =
    URLDecoder.decode(value, StandardCharsets.UTF_8)

  def authenticatedUser(exchange: HttpExchange): Option[PlatformUserAuth.AuthenticatedUser] =
    Option(exchange.getAttribute(AuthenticatedUserAttribute)).collect {
      case user: PlatformUserAuth.AuthenticatedUser => user
    }

  def ensurePlatformCsrf(
      exchange: HttpExchange,
      platformAuth: Option[PlatformUserAuth.Service]
  ): Boolean =
    if platformAuth.isEmpty then true
    else
      val passed = authenticatedUser(exchange).forall { user =>
        // CSRF tokens are session secrets; use constant-time comparison so a
        // timing oracle cannot recover the token character-by-character. The
        // basic-auth path already uses secureEquals for the same reason.
        Option(exchange.getRequestHeaders.getFirst("X-CSRF-Token"))
          .map(_.trim)
          .exists(submitted => secureEquals(submitted, user.csrfToken))
      }
      if !passed then
        // CSRF failure on a state-changing request is security-relevant: the
        // session resolved but the X-CSRF-Token header is missing or wrong.
        // Likely either an attacker attempting a cross-site request without
        // the JS frontend, or a session whose csrf cookie was cleared mid-
        // flow. Either way, operators tailing the logs want to see it.
        val email = authenticatedUser(exchange).map(_.email).getOrElse("-")
        logWarn(s"request forbidden path=${requestPath(exchange)} remote=${remoteAddress(exchange)} email=$email reason=csrf-missing-or-invalid")
      passed


  def authenticationMode(
      basicAuth: Option[BasicAuthConfig],
      platformAuth: Option[?]
  ): String =
    if basicAuth.nonEmpty then "basic"
    else if platformAuth.nonEmpty then "users"
    else "none"

  def authenticationEnabled(
      basicAuth: Option[BasicAuthConfig],
      platformAuth: Option[?]
  ): Boolean =
    basicAuth.nonEmpty || platformAuth.nonEmpty


  private def requestPath(exchange: HttpExchange): String =
    // getRawPath keeps percent-encoded sequences as-is; getPath would decode
    // them, and a `%20` in the URL would become a literal space in the
    // structured `path=...` log field -- which then splits at the wrong
    // column for any line-oriented parser. The raw form is uglier but
    // unambiguous and stable across log parsers.
    Option(exchange.getRequestURI).map(_.getRawPath).filter(_.nonEmpty).getOrElse("/")

  private def remoteAddress(exchange: HttpExchange): String =
    // Prefer the audit address stashed by the request wrapper -- it applies the
    // trusted-proxy policy so behind a reverse proxy the log shows the real
    // client IP from X-Forwarded-For (or whichever header is trusted) instead
    // of the proxy's loopback peer. Falls back to the raw TCP peer for code
    // paths that don't go through the wrapper (defense in depth).
    //
    // Bracket IPv6 hosts per RFC 3986 §3.2.2 so the host:port format stays
    // unambiguous in log lines. Without brackets, an audit entry like
    // `remote=::1:54321` cannot be split into host + port because every `:`
    // looks the same -- a log-line parser sees `::1` then `54321` as
    // separate IPv6 segments. With brackets, `remote=[::1]:54321` is clear.
    Option(exchange.getAttribute(AuditClientAddressAttribute))
      .collect { case s: String if s.nonEmpty => s }
      .getOrElse(RateLimit.formatAuditPeer(exchange))
