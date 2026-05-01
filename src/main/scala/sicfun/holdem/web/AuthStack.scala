package sicfun.holdem.web

import com.sun.net.httpserver.{HttpExchange, HttpHandler}

import java.net.URLDecoder
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.Base64
import scala.util.control.NonFatal

import ujson.{Arr, Obj, Str}

import sicfun.holdem.web.HandHistoryReviewServer.{
  BasicAuthConfig,
  JsonResponse,
  optionalString,
  readRequestBody,
  requiredString,
  retryAfterSeconds,
  logWarn
}
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

  def handleAuthMe(
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

  def handleAuthRegister(
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

  def handleAuthLogin(
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

  def handleAuthLogout(
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

  def handleAuthProfile(
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

  def handleOidcStart(
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

  def handleOidcCallback(
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
          writeJson(exchange, response.status, response.value)
      catch
        case NonFatal(e) =>
          writeJson(exchange, 500, Obj("error" -> Str(s"internal server error: ${e.getMessage}")))
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
          writePlain(exchange, 500, s"internal server error: ${e.getMessage}", "text/plain; charset=utf-8")
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

  def authenticatedUser(exchange: HttpExchange): Option[PlatformUserAuth.AuthenticatedUser] =
    Option(exchange.getAttribute(AuthenticatedUserAttribute)).collect {
      case user: PlatformUserAuth.AuthenticatedUser => user
    }

  def ensurePlatformCsrf(
      exchange: HttpExchange,
      platformAuth: Option[PlatformUserAuth.Service]
  ): Boolean =
    platformAuth.isEmpty || authenticatedUser(exchange).forall { user =>
      Option(exchange.getRequestHeaders.getFirst("X-CSRF-Token"))
        .map(_.trim)
        .contains(user.csrfToken)
    }


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
    Option(exchange.getRequestURI).map(_.getPath).filter(_.nonEmpty).getOrElse("/")

  private def remoteAddress(exchange: HttpExchange): String =
    Option(exchange.getRemoteAddress).map(address => s"${address.getHostString}:${address.getPort}").getOrElse("-")
