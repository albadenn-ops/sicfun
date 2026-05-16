package sicfun.holdem.web

import ujson.{Arr, Bool, Num, Obj, Str, Value}

import java.net.URLEncoder
import java.net.http.{HttpClient, HttpRequest, HttpResponse}
import java.net.URI
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths, StandardCopyOption}
import java.security.{MessageDigest, SecureRandom}
import java.time.Duration
import java.util.{Base64, UUID}
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong
import javax.crypto.SecretKeyFactory
import javax.crypto.spec.PBEKeySpec
import scala.util.control.NonFatal

/** Lightweight platform user-auth module (local credentials + optional OIDC providers).
  *
  * Provides a complete authentication system for the hand-history review web UI:
  *
  * '''Local auth:'''
  *   - Password hashing via PBKDF2WithHmacSHA256 (210,000 iterations, 256-bit key, 128-bit salt)
  *   - Registration (email + password) and login with rate limiting
  *   - User profiles with display name, preferred hero name, site, and timezone
  *
  * '''OIDC auth:'''
  *   - Google OIDC with PKCE (Proof Key for Code Exchange) for public clients
  *   - State parameter verification to prevent CSRF attacks on the OAuth flow
  *   - Automatic account linking when OIDC email matches an existing local account
  *
  * '''Session management:'''
  *   - 32-byte cryptographically random session tokens
  *   - Configurable TTL (default 12 hours)
  *   - CSRF tokens generated per session for mutation endpoint protection
  *   - HttpOnly, SameSite=Lax cookie attributes (optionally Secure for HTTPS)
  *
  * '''Storage:'''
  *   - JSON file-based user store (designed for single-instance deployments)
  *   - In-memory session store (sessions lost on restart)
  *   - In-memory OIDC state store with TTL-based cleanup
  *
  * @see [[HandHistoryReviewServer]] which integrates this module for auth middleware
  */
object PlatformUserAuth:
  private val LocalProviderId = "local"
  private val DefaultSessionCookieName = "sicfun_session"
  private val DefaultOidcStateCookieName = "sicfun_oidc_state"
  private val DefaultSessionTtlMs = 12L * 60L * 60L * 1000L
  private val DefaultOidcFlowTtlMs = 10L * 60L * 1000L
  private val PasswordAlgorithm = "PBKDF2WithHmacSHA256"
  private val PasswordSaltBytes = 16
  private val PasswordIterations = 210000
  private val PasswordKeyLengthBits = 256
  private val SessionTokenBytes = 32
  private val CsrfTokenBytes = 24
  private val OidcStateBytes = 24
  private val OidcCodeVerifierBytes = 32
  private val GoogleDiscoveryDocumentUri = URI.create("https://accounts.google.com/.well-known/openid-configuration")
  private val GoogleAuthEndpoint = "https://accounts.google.com/o/oauth2/v2/auth"
  private val GoogleTokenEndpoint = "https://oauth2.googleapis.com/token"
  private val GoogleUserInfoEndpoint = "https://openidconnect.googleapis.com/v1/userinfo"
  private val OidcSuccessRedirect = "/?auth=success"
  private val OidcFailureRedirectPrefix = "/?auth_error="

  final case class Config(
      storePath: Path,
      sessionTtlMs: Long = DefaultSessionTtlMs,
      allowLocalRegistration: Boolean = true,
      cookieSecure: Boolean = false,
      oidcProviders: Vector[OidcProvider] = Vector.empty,
      // Defensive cap on the total number of stored users. Without a cap, a
      // bot abusing public registration (10/min/IP via the auth rate limit)
      // can grow the user store ~43 MB/day/IP indefinitely -- a slow disk-
      // fill DoS over weeks. 100k is generous for any realistic private
      // deployment and far below the point where the in-memory linear-scan
      // by-email lookup becomes a perf concern. Operators expecting a much
      // larger user base or running heavy load-test scenarios can raise it;
      // tests use a much smaller value to exercise the rejection path.
      maxUsers: Int = 100_000
  ):
    require(sessionTtlMs > 0L, "sessionTtlMs must be positive")
    require(maxUsers > 0, "maxUsers must be positive")

  final case class UserProfile(
      displayName: String,
      heroName: Option[String] = None,
      preferredSite: Option[String] = None,
      timeZone: Option[String] = None,
      avatarUrl: Option[String] = None
  ):
    require(displayName.trim.nonEmpty, "displayName must be non-empty")

  final case class ProviderIdentity(
      provider: String,
      subject: String,
      emailAtLogin: Option[String],
      linkedAtEpochMs: Long
  ):
    require(provider.trim.nonEmpty, "provider must be non-empty")
    require(subject.trim.nonEmpty, "subject must be non-empty")

  final case class LocalPasswordCredential(
      saltBase64: String,
      hashBase64: String,
      iterations: Int,
      keyLengthBits: Int,
      updatedAtEpochMs: Long
  ):
    require(saltBase64.trim.nonEmpty, "saltBase64 must be non-empty")
    require(hashBase64.trim.nonEmpty, "hashBase64 must be non-empty")
    require(iterations > 0, "iterations must be positive")
    require(keyLengthBits > 0, "keyLengthBits must be positive")

  final case class StoredUser(
      userId: String,
      email: String,
      profile: UserProfile,
      identities: Vector[ProviderIdentity],
      localPassword: Option[LocalPasswordCredential],
      createdAtEpochMs: Long,
      updatedAtEpochMs: Long,
      lastLoginAtEpochMs: Option[Long]
  ):
    require(userId.trim.nonEmpty, "userId must be non-empty")
    require(email.trim.nonEmpty, "email must be non-empty")

  final case class UserView(
      userId: String,
      email: String,
      displayName: String,
      heroName: Option[String],
      preferredSite: Option[String],
      timeZone: Option[String],
      avatarUrl: Option[String],
      linkedProviders: Vector[String]
  )

  final case class AuthenticatedUser(
      userId: String,
      email: String,
      profile: UserView,
      csrfToken: String
  )

  final case class ProviderSummary(
      id: String,
      displayName: String,
      kind: String,
      startPath: Option[String]
  )

  final case class LoginResult(
      user: UserView,
      csrfToken: String,
      cookieHeader: String
  )

  /** Result of starting an OIDC authorization flow.
    *
    * `state` is the random value embedded in the redirect URL; the same value
    * is bound to a short-lived Set-Cookie (`stateCookieHeader`) the caller must
    * emit alongside the redirect. The callback handler will then require both
    * the URL state and the cookie state to match -- without the cookie, an
    * attacker who hijacks a valid state value cannot redirect a victim to our
    * callback URL and impersonate them (OAuth 2.0 BCP "mix-up" /
    * "covert-redirect" mitigation: bind the state to the user agent that
    * initiated the flow). */
  final case class OidcStartResult(
      location: String,
      state: String,
      stateCookieHeader: String
  )

  final case class OidcIdentity(
      subject: String,
      email: String,
      displayName: String,
      avatarUrl: Option[String]
  ):
    require(subject.trim.nonEmpty, "subject must be non-empty")
    require(email.trim.nonEmpty, "email must be non-empty")
    require(displayName.trim.nonEmpty, "displayName must be non-empty")

  trait OidcProvider:
    def id: String
    def displayName: String
    def startPath: String = s"/api/auth/oidc/$id/start"
    def callbackPath: String = s"/api/auth/oidc/$id/callback"
    def authorizationUri(state: String, codeChallenge: String): String
    def exchangeCode(code: String, codeVerifier: String): Either[String, OidcIdentity]

  final case class GoogleOidcConfig(
      clientId: String,
      clientSecret: String,
      redirectUri: String,
      scopes: Vector[String] = Vector("openid", "email", "profile")
  ):
    require(clientId.trim.nonEmpty, "clientId must be non-empty")
    require(clientSecret.trim.nonEmpty, "clientSecret must be non-empty")
    require(redirectUri.trim.nonEmpty, "redirectUri must be non-empty")
    require(scopes.nonEmpty, "scopes must be non-empty")

  // Default OIDC HTTP timeouts. Without these the JDK HttpClient blocks
  // indefinitely on a slow/hung token or userinfo endpoint, holding a server
  // executor thread per callback. 5s to connect and 10s end-to-end is well
  // above Google's typical latency (sub-second) and below anything that
  // would feel responsive to a user waiting on the OIDC redirect.
  private val DefaultOidcConnectTimeout = Duration.ofSeconds(5)
  private val DefaultOidcRequestTimeout = Duration.ofSeconds(10)

  private[web] def defaultOidcHttpClient(): HttpClient =
    HttpClient.newBuilder.connectTimeout(DefaultOidcConnectTimeout).build()

  final class GoogleOidcProvider(
      config: GoogleOidcConfig,
      httpClient: HttpClient = defaultOidcHttpClient(),
      requestTimeout: Duration = DefaultOidcRequestTimeout
  ) extends OidcProvider:
    override val id = "google"
    override val displayName = "Google"

    override def authorizationUri(state: String, codeChallenge: String): String =
      val query = formEncode(
        Vector(
          "client_id" -> config.clientId,
          "redirect_uri" -> config.redirectUri,
          "response_type" -> "code",
          "scope" -> config.scopes.mkString(" "),
          "state" -> state,
          "code_challenge" -> codeChallenge,
          "code_challenge_method" -> "S256",
          "access_type" -> "online",
          "include_granted_scopes" -> "true",
          "prompt" -> "select_account"
        )
      )
      s"$GoogleAuthEndpoint?$query"

    override def exchangeCode(code: String, codeVerifier: String): Either[String, OidcIdentity] =
      try
        val tokenRequest = HttpRequest.newBuilder(URI.create(GoogleTokenEndpoint))
          .header("Content-Type", "application/x-www-form-urlencoded")
          .timeout(requestTimeout)
          .POST(
            HttpRequest.BodyPublishers.ofString(
              formEncode(
                Vector(
                  "code" -> code,
                  "client_id" -> config.clientId,
                  "client_secret" -> config.clientSecret,
                  "redirect_uri" -> config.redirectUri,
                  "grant_type" -> "authorization_code",
                  "code_verifier" -> codeVerifier
                )
              ),
              StandardCharsets.UTF_8
            )
          )
          .build()
        val tokenResponse = httpClient.send(tokenRequest, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8))
        if tokenResponse.statusCode() / 100 != 2 then
          Left(s"Google token exchange failed with status ${tokenResponse.statusCode()}")
        else
          val tokenJson = ujson.read(tokenResponse.body())
          val accessToken = tokenJson.obj.get("access_token").map(_.str.trim).filter(_.nonEmpty)
          accessToken match
            case None => Left("Google token exchange did not return an access token")
            case Some(token) =>
              val userInfoRequest = HttpRequest.newBuilder(URI.create(GoogleUserInfoEndpoint))
                .header("Authorization", s"Bearer $token")
                .timeout(requestTimeout)
                .GET()
                .build()
              val userInfoResponse = httpClient.send(
                userInfoRequest,
                HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8)
              )
              if userInfoResponse.statusCode() / 100 != 2 then
                Left(s"Google userinfo request failed with status ${userInfoResponse.statusCode()}")
              else
                parseGoogleUserInfo(ujson.read(userInfoResponse.body()))
      catch
        case NonFatal(e) => Left(s"Google OIDC exchange failed: ${e.getMessage}")

    private def parseGoogleUserInfo(json: Value): Either[String, OidcIdentity] =
      val obj = json.obj
      val subject = obj.get("sub").map(_.str.trim).filter(_.nonEmpty)
      val email = obj.get("email").map(_.str.trim).filter(_.nonEmpty)
      val emailVerified = obj.get("email_verified").exists(_.bool)
      val displayName = obj.get("name").map(_.str.trim).filter(_.nonEmpty).orElse(email)
      val avatarUrl = obj.get("picture").map(_.str.trim).filter(_.nonEmpty)
      (subject, email, displayName) match
        case (Some(sub), Some(value), Some(name)) if emailVerified =>
          Right(OidcIdentity(subject = sub, email = value, displayName = name, avatarUrl = avatarUrl))
        case (Some(_), Some(_), Some(_)) =>
          Left("Google did not return a verified email address for this account")
        case _ =>
          Left("Google userinfo response was missing required identity fields")

  object GoogleOidcProvider:
    def discoveryDocumentUri: URI = GoogleDiscoveryDocumentUri

  final class Service private (
      config: Config,
      userStore: JsonUserStore,
      sessionManager: SessionManager,
      oidcStateStore: OidcStateStore
  ):
    private val providersById = config.oidcProviders.iterator.map(provider => provider.id -> provider).toMap

    def providerSummaries: Vector[ProviderSummary] =
      val local = Vector(
        ProviderSummary(
          id = LocalProviderId,
          displayName = "Email and password",
          kind = "password",
          startPath = None
        )
      )
      val oidc = config.oidcProviders.map(provider =>
        ProviderSummary(
          id = provider.id,
          displayName = provider.displayName,
          kind = "oidc",
          startPath = Some(provider.startPath)
        )
      )
      local ++ oidc

    def authenticationState(currentUser: Option[AuthenticatedUser]): Value =
      Obj(
        "authenticationEnabled" -> Bool(true),
        "authenticationMode" -> Str("users"),
        "authenticated" -> Bool(currentUser.nonEmpty),
        "allowLocalRegistration" -> Bool(config.allowLocalRegistration),
        "providers" -> Arr.from(providerSummaries.map(writeProviderSummary)),
        "user" -> currentUser.map(user => writeUserView(user.profile)).getOrElse(ujson.Null),
        "csrfToken" -> currentUser.map(user => Str(user.csrfToken)).getOrElse(ujson.Null)
      )

    def resolveSession(cookieHeader: Option[String]): Option[AuthenticatedUser] =
      sessionManager.resolve(cookieHeader, userStore)

    /** Number of users currently in the in-memory store. Exposed for the
      * `/api/health` dashboard so operators can see live usage against the
      * `maxUsers` cap. O(1) read of a volatile snapshot -- no lock taken --
      * because state is the @volatile var in JsonUserStore. */
    def storedUserCount: Int = userStore.storedUserCount

    /** Number of session records currently held in memory. Approximate -- may
      * include expired sessions that have not yet been purged by the next
      * resolve() or cleanup pass -- but accurate enough for the
      * `/api/health` dashboard. Useful as a capacity proxy: a steady-state
      * count higher than expected (e.g. an unusual spike during off-hours)
      * is a heads-up about credential stuffing succeeded or a leaked
      * automation script. Includes anonymous-OIDC-flow start cookies? No;
      * the state-cookie store is separate. */
    def activeSessionCount: Int = sessionManager.activeSessionCount

    def registerLocal(
        email: String,
        password: String,
        displayName: Option[String]
    ): Either[String, LoginResult] =
      if !config.allowLocalRegistration then Left("local registration is disabled")
      else userStore.registerLocal(email, password, displayName).map(createLoginResult)

    def loginLocal(email: String, password: String): Either[String, LoginResult] =
      userStore.authenticateLocal(email, password).map(createLoginResult)

    def updateProfile(
        userId: String,
        displayName: Option[String],
        heroName: Option[String],
        preferredSite: Option[String],
        timeZone: Option[String]
    ): Either[String, UserView] =
      userStore.updateProfile(userId, displayName, heroName, preferredSite, timeZone).map(toUserView)

    def revokeSession(cookieHeader: Option[String]): String =
      sessionManager.revoke(cookieHeader)

    def startOidc(providerId: String): Either[String, OidcStartResult] =
      providersById.get(providerId).toRight(s"unknown OIDC provider '$providerId'").map { provider =>
        val codeVerifier = randomBase64Url(OidcCodeVerifierBytes)
        val state = oidcStateStore.issue(provider.id, codeVerifier)
        OidcStartResult(
          location = provider.authorizationUri(state, codeChallenge(codeVerifier)),
          state = state,
          stateCookieHeader = oidcStateCookieHeader(state, DefaultOidcFlowTtlMs, config.cookieSecure)
        )
      }

    def expectedOidcStateCookieName: String = oidcStateCookieName(config.cookieSecure)

    def oidcStateClearCookieHeader: String = clearOidcStateCookieHeader(config.cookieSecure)

    def finishOidc(providerId: String, state: String, code: String): Either[String, LoginResult] =
      for
        provider <- providersById.get(providerId).toRight(s"unknown OIDC provider '$providerId'")
        issuedState <- oidcStateStore.consume(provider.id, state).toRight("OIDC login state expired or is invalid")
        identity <- provider.exchangeCode(code, issuedState.codeVerifier)
        user <- userStore.upsertOidcIdentity(provider.id, identity)
      yield createLoginResult(user)

    private def createLoginResult(user: StoredUser): LoginResult =
      val refreshedUser = userStore.touchLogin(user.userId)
      val session = sessionManager.create(refreshedUser)
      LoginResult(
        user = toUserView(refreshedUser),
        csrfToken = session.csrfToken,
        cookieHeader = session.cookieHeader
      )

  object Service:
    def create(config: Config): Either[String, Service] =
      if !config.allowLocalRegistration && config.oidcProviders.isEmpty then
        Left("user auth requires at least one sign-in method")
      else
        validateOidcProviderIds(config.oidcProviders).flatMap { _ =>
          try
            val store = new JsonUserStore(config.storePath, config.maxUsers)
            Right(
              new Service(
                config = config,
                userStore = store,
                sessionManager = new SessionManager(config.sessionTtlMs, config.cookieSecure),
                oidcStateStore = new OidcStateStore()
              )
            )
          catch
            case NonFatal(e) => Left(s"user auth failed to initialize: ${e.getMessage}")
        }

    /** Reject OIDC provider configurations that would either shadow the local
      * password provider or collide with another OIDC entry. Without this
      * check, two providers with the same id would silently collapse to one
      * in `providersById.toMap` AND then crash HTTP server registration with
      * an opaque "context already exists" IllegalArgumentException; an OIDC
      * provider with id="local" would shadow the local-password provider in
      * the /api/auth/me providers list. Both are deployment configuration
      * bugs the operator wants to see surfaced at startup, not at first use. */
    private def validateOidcProviderIds(providers: Vector[OidcProvider]): Either[String, Unit] =
      val reserved = providers.map(_.id).filter(_ == LocalProviderId)
      if reserved.nonEmpty then
        Left(s"OIDC provider id '$LocalProviderId' is reserved for local password sign-in; choose a different id")
      else
        val duplicates = providers
          .map(_.id)
          .groupBy(identity)
          .collect { case (id, occurrences) if occurrences.size > 1 => id }
          .toVector
          .sorted
        if duplicates.nonEmpty then
          Left(s"OIDC providers must have unique ids; duplicates: ${duplicates.mkString(", ")}")
        else Right(())

  private final case class StoreState(users: Vector[StoredUser])

  private final class JsonUserStore(path: Path, maxUsers: Int):
    @volatile private var state = load()

    def registerLocal(
        email: String,
        password: String,
        displayName: Option[String]
    ): Either[String, StoredUser] =
      synchronized:
        try
          val normalizedEmail = normalizeEmail(email)
          validateEmail(normalizedEmail)
          validatePassword(password)
          if findByEmailInternal(normalizedEmail).nonEmpty then
            Left("an account with that email already exists")
          else if state.users.length >= maxUsers then
            // Hard cap on the user store to keep public-registration
            // deployments from being slow-disk-filled by a bot that abuses
            // the auth bucket (10/min/IP -> ~14400 registrations/day/IP ->
            // ~43 MB/day/IP of stored user records). Generic error keeps
            // the response small AND avoids letting the attacker
            // fingerprint the limit by probing.
            Left("registration is temporarily unavailable")
          else
            val now = System.currentTimeMillis()
            val resolvedDisplayName = sanitizeDisplayName(displayName).getOrElse(defaultDisplayNameFor(normalizedEmail))
            val user = StoredUser(
              userId = UUID.randomUUID().toString,
              email = normalizedEmail,
              profile = UserProfile(displayName = resolvedDisplayName),
              identities = Vector(
                ProviderIdentity(
                  provider = LocalProviderId,
                  subject = normalizedEmail,
                  emailAtLogin = Some(normalizedEmail),
                  linkedAtEpochMs = now
                )
              ),
              localPassword = Some(hashPassword(password, now)),
              createdAtEpochMs = now,
              updatedAtEpochMs = now,
              lastLoginAtEpochMs = None
            )
            persist(sortedState(state.users :+ user))
            Right(user)
        catch
          case NonFatal(e) => Left(e.getMessage)

    def authenticateLocal(email: String, password: String): Either[String, StoredUser] =
      // Reject oversize password BEFORE any lookup or hashing so all email values
      // get the same fast response. Both the DoS guard (avoid PBKDF2 on multi-MB
      // input) and the timing-leak guard (no per-email branching) fall out of
      // this single short-circuit.
      if password.length > MaxPasswordLength then Left("invalid email or password")
      else
        synchronized:
          val normalizedEmail = normalizeEmail(email)
          findByEmailInternal(normalizedEmail) match
            case None =>
              // Do equivalent PBKDF2 work in the "no such user" path so the
              // response time matches the "known user, wrong password" path.
              // Otherwise an attacker can enumerate registered email addresses
              // by measuring login latency, even though the error string is
              // identical across both branches.
              dummyVerifyPassword(password)
              Left("invalid email or password")
            case Some(user) =>
              user.localPassword match
                case None =>
                  // OIDC-only user: keep timing AND message indistinguishable
                  // from the "no such user" and "wrong password" branches so
                  // neither email existence nor account type leaks.
                  dummyVerifyPassword(password)
                  Left("invalid email or password")
                case Some(credential) =>
                  if verifyPassword(password, credential) then Right(user)
                  else Left("invalid email or password")

    def upsertOidcIdentity(providerId: String, identity: OidcIdentity): Either[String, StoredUser] =
      synchronized:
        val normalizedEmail = normalizeEmail(identity.email)
        // Validate the email format and length the OIDC provider returned even
        // though Google checks emailVerified upstream. Defense in depth: a
        // future provider or a tampered userinfo response could deliver a
        // malformed or oversize value, and the local-register path applies
        // the same check -- keep both code paths consistent.
        try validateEmail(normalizedEmail)
        catch case e: IllegalArgumentException => return Left(e.getMessage)
        // Reject suspiciously long OIDC subject identifiers. Google's are
        // ~21 chars; 256 is two orders of magnitude over that, plenty for any
        // legitimate provider, but bounded so a hostile provider can't poison
        // the user store with megabyte-sized subjects (which are also used as
        // map keys for provider-identity lookup). Truncation here is unsafe --
        // two distinct subjects could collide on a truncated prefix.
        if identity.subject.length > 256 then return Left("OIDC subject is too long")
        // Truncate the provider-supplied display name to the same 96-char cap
        // the local-register path enforces. We truncate (rather than reject)
        // for the OIDC flow: a legitimate user with a long display name on
        // Google should still be able to sign in -- they can shorten it via
        // /api/auth/profile afterward. Without this, a malformed or huge
        // upstream `name` would bloat the user-store JSON unbounded.
        val truncatedDisplayName = identity.displayName.take(96)
        // Drop an avatar URL that exceeds a generous URL-length bound. Stored
        // verbatim but never rendered today; even so, an unbounded value would
        // bloat the user store and a future renderer would have to defend
        // against the bloat itself. 2048 is the de-facto URL length browsers
        // and proxies accept.
        val cappedAvatarUrl = identity.avatarUrl.filter(_.length <= 2048)
        val now = System.currentTimeMillis()
        findByProviderIdentityInternal(providerId, identity.subject) match
          case Some(existing) =>
            val updated = existing.copy(
              email = normalizedEmail,
              profile = existing.profile.copy(
                displayName = preferNonBlank(existing.profile.displayName, truncatedDisplayName),
                avatarUrl = cappedAvatarUrl.orElse(existing.profile.avatarUrl)
              ),
              identities = existing.identities.map { current =>
                if current.provider == providerId && current.subject == identity.subject then
                  current.copy(emailAtLogin = Some(normalizedEmail))
                else current
              },
              updatedAtEpochMs = now
            )
            persist(replaceUser(existing.userId, updated))
            Right(updated)
          case None =>
            findByEmailInternal(normalizedEmail) match
              case Some(_) =>
                Left("an account with that email already exists; sign in with its existing method")
              case None =>
                val created = StoredUser(
                  userId = UUID.randomUUID().toString,
                  email = normalizedEmail,
                  profile = UserProfile(
                    displayName = truncatedDisplayName,
                    avatarUrl = cappedAvatarUrl
                  ),
                  identities = Vector(
                    ProviderIdentity(
                      provider = providerId,
                      subject = identity.subject,
                      emailAtLogin = Some(normalizedEmail),
                      linkedAtEpochMs = now
                    )
                  ),
                  localPassword = None,
                  createdAtEpochMs = now,
                  updatedAtEpochMs = now,
                  lastLoginAtEpochMs = None
                )
                persist(sortedState(state.users :+ created))
                Right(created)

    def updateProfile(
        userId: String,
        displayName: Option[String],
        heroName: Option[String],
        preferredSite: Option[String],
        timeZone: Option[String]
    ): Either[String, StoredUser] =
      synchronized:
        findByUserIdInternal(userId).toRight("user not found").flatMap { current =>
          try
            val now = System.currentTimeMillis()
            val updated = current.copy(
              profile = current.profile.copy(
                displayName = sanitizeDisplayName(displayName).getOrElse(current.profile.displayName),
                heroName = sanitizeOptionalField(heroName, "heroName", 64),
                preferredSite = sanitizeOptionalField(preferredSite, "preferredSite", 32),
                timeZone = sanitizeOptionalField(timeZone, "timeZone", 64),
                avatarUrl = current.profile.avatarUrl
              ),
              updatedAtEpochMs = now
            )
            persist(replaceUser(current.userId, updated))
            Right(updated)
          catch
            case NonFatal(e) => Left(e.getMessage)
        }

    def touchLogin(userId: String): StoredUser =
      synchronized:
        val current = findByUserIdInternal(userId).getOrElse(
          throw new IllegalArgumentException(s"user not found: $userId")
        )
        val now = System.currentTimeMillis()
        val updated = current.copy(
          updatedAtEpochMs = now,
          lastLoginAtEpochMs = Some(now)
        )
        persist(replaceUser(current.userId, updated))
        updated

    def findByUserId(userId: String): Option[StoredUser] =
      synchronized:
        findByUserIdInternal(userId)

    // Lock-free read of the live user count for the /api/health JSON. state
    // is @volatile so the read sees a consistent snapshot without contending
    // for the per-store synchronized block; registerLocal / upsertOidcIdentity
    // / updateProfile all serialize on that lock so the value cannot tear.
    def storedUserCount: Int = state.users.length

    private def load(): StoreState =
      if !Files.exists(path) then StoreState(Vector.empty)
      else
        try
          val raw = Files.readString(path, StandardCharsets.UTF_8)
          val json = ujson.read(raw)
          val users = json.obj.get("users").map(_.arr.toVector.map(readStoredUser)).getOrElse(Vector.empty)
          StoreState(users = users)
        catch
          case NonFatal(e) =>
            // Wrap with file path + recovery hint so the operator can act. A bare ujson
            // parse error like "expected ']' got '}' at offset 1234" otherwise reaches
            // Service.create's catch with no indication of which file is corrupted.
            throw new RuntimeException(
              s"user store at ${path.toAbsolutePath} is unreadable: ${e.getMessage}. " +
                "Back up the file and restore from backup, or remove it to start fresh.",
              e
            )

    private def persist(next: StoreState): Unit =
      // Write to disk FIRST, publish to memory only on success. The previous
      // order (state = next; writeState(...)) left the in-memory store ahead
      // of disk if the write failed -- the caller saw an error and the user
      // store quietly held a ghost record that disappeared on the next
      // restart. With the journal-then-publish order, a failed write keeps
      // memory and disk consistent; the IOException propagates to the
      // synchronized caller (registerLocal / updateProfile / upsertOidcIdentity)
      // which returns Left and the user can retry cleanly.
      writeState(path, next)
      state = next

    private def replaceUser(userId: String, updated: StoredUser): StoreState =
      sortedState(state.users.map(current => if current.userId == userId then updated else current))

    private def sortedState(users: Vector[StoredUser]): StoreState =
      StoreState(users = users.sortBy(user => (user.email, user.userId)))

    private def findByEmailInternal(email: String): Option[StoredUser] =
      state.users.find(_.email == email)

    private def findByProviderIdentityInternal(provider: String, subject: String): Option[StoredUser] =
      state.users.find(_.identities.exists(identity => identity.provider == provider && identity.subject == subject))

    private def findByUserIdInternal(userId: String): Option[StoredUser] =
      state.users.find(_.userId == userId)

  private final case class SessionRecord(
      userId: String,
      csrfToken: String,
      createdAtEpochMs: Long,
      expiresAtEpochMs: Long,
      lastSeenAtEpochMs: Long
  )

  private final case class SessionMaterial(
      cookieHeader: String,
      csrfToken: String
  )

  private final class SessionManager(
      sessionTtlMs: Long,
      cookieSecure: Boolean,
      nowMillis: () => Long = () => System.currentTimeMillis()
  ):
    private val sessions = new ConcurrentHashMap[String, SessionRecord]()

    // Approximate count of in-memory session records. ConcurrentHashMap.size
    // is documented as "not a constant-time operation" but for our scale
    // (typically dozens to thousands of sessions) it's microseconds.
    // Includes records that have expired but not yet been purged by the
    // next resolve() / cleanup pass; close enough for dashboarding.
    def activeSessionCount: Int = sessions.size()

    def create(user: StoredUser): SessionMaterial =
      purgeExpired()
      val token = randomBase64Url(SessionTokenBytes)
      val hashedToken = sha256Hex(token)
      val now = nowMillis()
      val csrfToken = randomBase64Url(CsrfTokenBytes)
      sessions.put(
        hashedToken,
        SessionRecord(
          userId = user.userId,
          csrfToken = csrfToken,
          createdAtEpochMs = now,
          expiresAtEpochMs = now + sessionTtlMs,
          lastSeenAtEpochMs = now
        )
      )
      SessionMaterial(
        cookieHeader = sessionCookieHeader(token, sessionTtlMs, cookieSecure),
        csrfToken = csrfToken
      )

    def resolve(
        cookieHeader: Option[String],
        userStore: JsonUserStore
    ): Option[AuthenticatedUser] =
      purgeExpired()
      extractCookie(cookieHeader, sessionCookieName(cookieSecure))
        .flatMap { token =>
          val key = sha256Hex(token)
          // computeIfPresent atomically reads + conditionally updates + writes,
          // closing two concurrency races a plain get/filter/put loop has:
          //   - resurrection: a logout that calls `revoke` (sessions.remove)
          //     between our get and put would otherwise be undone by the put,
          //     bringing a revoked session back to life.
          //   - false expiry: a concurrent `purgeExpired` evicting an entry we
          //     just refreshed; with compute, both writes serialize on the bin
          //     so whichever runs second sees the other's update.
          // Returning null from the remapping function removes the entry, which
          // is what we want when the snapshot read fires for a session that
          // expired between the purge sweep and this lookup.
          val refreshed = sessions.computeIfPresent(
            key,
            (_, current) =>
              if current.expiresAtEpochMs > nowMillis() then
                val now = nowMillis()
                current.copy(expiresAtEpochMs = now + sessionTtlMs, lastSeenAtEpochMs = now)
              else
                null
          )
          Option(refreshed).flatMap { session =>
            userStore.findByUserId(session.userId).map { user =>
              AuthenticatedUser(
                userId = user.userId,
                email = user.email,
                profile = toUserView(user),
                csrfToken = session.csrfToken
              )
            }
          }
        }

    def revoke(cookieHeader: Option[String]): String =
      extractCookie(cookieHeader, sessionCookieName(cookieSecure)).foreach { token =>
        sessions.remove(sha256Hex(token))
      }
      clearSessionCookieHeader(cookieSecure)

    private def purgeExpired(): Unit =
      val now = nowMillis()
      val iterator = sessions.entrySet().iterator()
      while iterator.hasNext do
        val entry = iterator.next()
        if entry.getValue.expiresAtEpochMs <= now then
          // Compare-and-remove: iterator.remove() would unconditionally drop the
          // key, but a concurrent resolve() that refreshed the session between
          // the entry.getValue snapshot and this line would have its work
          // erased. computeIfPresent re-reads the current value under the bin
          // lock and only removes if the entry is STILL expired.
          sessions.computeIfPresent(
            entry.getKey,
            (_, current) =>
              if current.expiresAtEpochMs <= nowMillis() then null else current
          )

  private final case class OidcPendingState(
      providerId: String,
      codeVerifier: String,
      issuedAtEpochMs: Long
  )

  private final class OidcStateStore(
      flowTtlMs: Long = DefaultOidcFlowTtlMs,
      nowMillis: () => Long = () => System.currentTimeMillis()
  ):
    private val states = new ConcurrentHashMap[String, OidcPendingState]()
    private val lastCleanupAtMs = new AtomicLong(0L)

    def issue(providerId: String, codeVerifier: String): String =
      cleanupIfDue()
      val state = randomBase64Url(OidcStateBytes)
      states.put(
        state,
        OidcPendingState(
          providerId = providerId,
          codeVerifier = codeVerifier,
          issuedAtEpochMs = nowMillis()
        )
      )
      state

    def consume(providerId: String, state: String): Option[OidcPendingState] =
      cleanupIfDue()
      // Atomically remove only when providerId and TTL both match. The earlier
      // version called `states.remove(state)` unconditionally and then
      // filtered, which meant a callback with a valid state but the wrong
      // provider id would still evict the entry -- the legitimate user's
      // subsequent correct callback would then 404 because their state had
      // already been consumed (and discarded) by the wrong-provider request.
      // Non-exploitable in practice because state is 32-byte random, but the
      // atomic form removes the latent foot-gun.
      val holder = new java.util.concurrent.atomic.AtomicReference[Option[OidcPendingState]](None)
      states.computeIfPresent(state, (_, record) =>
        if record.providerId == providerId && nowMillis() - record.issuedAtEpochMs <= flowTtlMs then
          holder.set(Some(record))
          null  // signal removal
        else record
      )
      holder.get()

    private def cleanupIfDue(): Unit =
      val now = nowMillis()
      val lastCleanup = lastCleanupAtMs.get()
      if now - lastCleanup >= flowTtlMs && lastCleanupAtMs.compareAndSet(lastCleanup, now) then
        val iterator = states.entrySet().iterator()
        while iterator.hasNext do
          val entry = iterator.next()
          if now - entry.getValue.issuedAtEpochMs > flowTtlMs then
            iterator.remove()

  private def toUserView(user: StoredUser): UserView =
    UserView(
      userId = user.userId,
      email = user.email,
      displayName = user.profile.displayName,
      heroName = user.profile.heroName,
      preferredSite = user.profile.preferredSite,
      timeZone = user.profile.timeZone,
      avatarUrl = user.profile.avatarUrl,
      linkedProviders = user.identities.map(_.provider).distinct.sorted
    )

  private def writeState(path: Path, state: StoreState): Unit =
    val absolutePath = path.toAbsolutePath.normalize()
    Option(absolutePath.getParent).foreach(parent => Files.createDirectories(parent))
    val json = Obj(
      "version" -> Num(1),
      "users" -> Arr.from(state.users.map(writeStoredUser))
    )
    val encoded = ujson.write(json, indent = 2)
    val parent = Option(absolutePath.getParent).getOrElse(Paths.get(".").toAbsolutePath.normalize())
    val temp = Files.createTempFile(parent, "platform-users-", ".json.tmp")
    try
      Files.writeString(temp, encoded, StandardCharsets.UTF_8)
      try
        Files.move(temp, absolutePath, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE)
      catch
        case _: UnsupportedOperationException | _: java.nio.file.AtomicMoveNotSupportedException =>
          Files.move(temp, absolutePath, StandardCopyOption.REPLACE_EXISTING)
    finally
      // If the move succeeded, the temp path no longer exists and this is a no-op.
      // If writeString or a non-fallback move exception threw mid-way, this prevents
      // the orphaned tmp file from accumulating in the user-store directory across
      // repeated failures (disk full, permission flap, etc.).
      //
      // Swallow IOException here so a delete failure (e.g. the temp file was
      // locked by another process on Windows) does not mask the original
      // exception that triggered the cleanup. The original exception is the
      // useful diagnostic; an orphaned tmp file is a much smaller problem.
      try Files.deleteIfExists(temp)
      catch case _: java.io.IOException => ()

  private def writeStoredUser(user: StoredUser): Value =
    Obj(
      "userId" -> Str(user.userId),
      "email" -> Str(user.email),
      "profile" -> writeUserProfile(user.profile),
      "identities" -> Arr.from(user.identities.map(writeProviderIdentity)),
      "localPassword" -> user.localPassword.map(writeLocalPasswordCredential).getOrElse(ujson.Null),
      "createdAtEpochMs" -> Num(user.createdAtEpochMs.toDouble),
      "updatedAtEpochMs" -> Num(user.updatedAtEpochMs.toDouble),
      "lastLoginAtEpochMs" -> user.lastLoginAtEpochMs.map(value => Num(value.toDouble)).getOrElse(ujson.Null)
    )

  private def readStoredUser(json: Value): StoredUser =
    val obj = json.obj
    StoredUser(
      userId = obj("userId").str,
      email = normalizeEmail(obj("email").str),
      profile = readUserProfile(obj("profile")),
      identities = obj.get("identities").map(_.arr.toVector.map(readProviderIdentity)).getOrElse(Vector.empty),
      localPassword = obj.get("localPassword").filterNot(_ == ujson.Null).map(readLocalPasswordCredential),
      createdAtEpochMs = obj("createdAtEpochMs").num.toLong,
      updatedAtEpochMs = obj("updatedAtEpochMs").num.toLong,
      lastLoginAtEpochMs = obj.get("lastLoginAtEpochMs").filterNot(_ == ujson.Null).map(_.num.toLong)
    )

  private def writeUserProfile(profile: UserProfile): Value =
    Obj(
      "displayName" -> Str(profile.displayName),
      "heroName" -> profile.heroName.map(Str(_)).getOrElse(ujson.Null),
      "preferredSite" -> profile.preferredSite.map(Str(_)).getOrElse(ujson.Null),
      "timeZone" -> profile.timeZone.map(Str(_)).getOrElse(ujson.Null),
      "avatarUrl" -> profile.avatarUrl.map(Str(_)).getOrElse(ujson.Null)
    )

  private def readUserProfile(json: Value): UserProfile =
    val obj = json.obj
    UserProfile(
      displayName = obj("displayName").str,
      heroName = obj.get("heroName").filterNot(_ == ujson.Null).map(_.str),
      preferredSite = obj.get("preferredSite").filterNot(_ == ujson.Null).map(_.str),
      timeZone = obj.get("timeZone").filterNot(_ == ujson.Null).map(_.str),
      avatarUrl = obj.get("avatarUrl").filterNot(_ == ujson.Null).map(_.str)
    )

  private def writeProviderIdentity(identity: ProviderIdentity): Value =
    Obj(
      "provider" -> Str(identity.provider),
      "subject" -> Str(identity.subject),
      "emailAtLogin" -> identity.emailAtLogin.map(Str(_)).getOrElse(ujson.Null),
      "linkedAtEpochMs" -> Num(identity.linkedAtEpochMs.toDouble)
    )

  private def readProviderIdentity(json: Value): ProviderIdentity =
    val obj = json.obj
    ProviderIdentity(
      provider = obj("provider").str,
      subject = obj("subject").str,
      emailAtLogin = obj.get("emailAtLogin").filterNot(_ == ujson.Null).map(_.str),
      linkedAtEpochMs = obj("linkedAtEpochMs").num.toLong
    )

  private def writeLocalPasswordCredential(credential: LocalPasswordCredential): Value =
    Obj(
      "saltBase64" -> Str(credential.saltBase64),
      "hashBase64" -> Str(credential.hashBase64),
      "iterations" -> Num(credential.iterations.toDouble),
      "keyLengthBits" -> Num(credential.keyLengthBits.toDouble),
      "updatedAtEpochMs" -> Num(credential.updatedAtEpochMs.toDouble)
    )

  private def readLocalPasswordCredential(json: Value): LocalPasswordCredential =
    val obj = json.obj
    LocalPasswordCredential(
      saltBase64 = obj("saltBase64").str,
      hashBase64 = obj("hashBase64").str,
      iterations = obj("iterations").num.toInt,
      keyLengthBits = obj("keyLengthBits").num.toInt,
      updatedAtEpochMs = obj("updatedAtEpochMs").num.toLong
    )

  private def writeProviderSummary(summary: ProviderSummary): Value =
    Obj(
      "id" -> Str(summary.id),
      "displayName" -> Str(summary.displayName),
      "kind" -> Str(summary.kind),
      "startPath" -> summary.startPath.map(Str(_)).getOrElse(ujson.Null)
    )

  private def writeUserView(user: UserView): Value =
    Obj(
      "userId" -> Str(user.userId),
      "email" -> Str(user.email),
      "displayName" -> Str(user.displayName),
      "heroName" -> user.heroName.map(Str(_)).getOrElse(ujson.Null),
      "preferredSite" -> user.preferredSite.map(Str(_)).getOrElse(ujson.Null),
      "timeZone" -> user.timeZone.map(Str(_)).getOrElse(ujson.Null),
      "avatarUrl" -> user.avatarUrl.map(Str(_)).getOrElse(ujson.Null),
      "linkedProviders" -> Arr.from(user.linkedProviders.map(Str(_)))
    )

  // RFC 5321 sec 4.5.3.1.3: an email address (local-part @ domain) cannot exceed
  // 254 octets. Without this cap an attacker can register with a multi-megabyte
  // "email" that passes structural validation, persists in the user store JSON,
  // and slows future startups when load() reads it back. The cap is a fixed
  // defensive bound; legitimate addresses are well under it (most real-world
  // emails are under 50 chars).
  private val MaxEmailLength = 254

  private def validateEmail(email: String): Unit =
    if email.length > MaxEmailLength then
      throw new IllegalArgumentException(s"email must be at most $MaxEmailLength characters")
    // RFC 5321 §4.1.2 says local-part is built from `atext` which excludes
    // whitespace (unless quoted, which we do not support). Rejecting any
    // whitespace also keeps our `key=value` audit log lines parseable: a
    // value with an embedded space would split as two tokens for any line-
    // oriented parser. Same for any C0 control char.
    if email.exists(ch => ch.isWhitespace || ch.toInt < 0x20 || ch.toInt == 0x7F) then
      throw new IllegalArgumentException("email must not contain whitespace or control characters")
    val at = email.indexOf('@')
    val dot = email.lastIndexOf('.')
    if at <= 0 || dot <= at + 1 || dot == email.length - 1 then
      throw new IllegalArgumentException("email must be a valid address")

  // Cap the password length the server is willing to hash. PBKDF2's per-iteration
  // cost scales with input length, so without this cap an attacker submitting a 2 MB
  // password (just under MAX_UPLOAD_BYTES) and a high-iteration count could burn the
  // server's CPU on each login attempt. The auth-bucket rate limiter (default 10/min
  // /IP) caps requests, but the length cap is what bounds the work each individual
  // request is allowed to cost. 256 chars is generous for any realistic passphrase
  // and well above OWASP's 64-char minimum recommendation.
  private val MaxPasswordLength = 256

  private def validatePassword(password: String): Unit =
    // Validate the password as-is, NOT trimmed. NIST SP 800-63B says memorized
    // secrets must accept any printable ASCII including spaces, and explicitly
    // forbids truncation. The earlier version trimmed before validating length,
    // which created a foot-gun: a register call with "  hunter22  " (12 chars
    // raw, 8 trimmed) passed the length check on the trimmed form but stored
    // the hash of the RAW (whitespace-included) value, and login then required
    // the same whitespace -- silent UX failure when the user pasted with stray
    // whitespace at register but typed cleanly at login.
    if password.length < 10 then
      throw new IllegalArgumentException("password must be at least 10 characters")
    if password.length > MaxPasswordLength then
      throw new IllegalArgumentException(s"password must be at most $MaxPasswordLength characters")

  private def sanitizeDisplayName(displayName: Option[String]): Option[String] =
    sanitizeOptionalField(displayName, "displayName", 96)

  private def sanitizeOptionalField(raw: Option[String], label: String, maxLength: Int): Option[String] =
    raw.map(_.trim).filter(_.nonEmpty).map { value =>
      if value.length > maxLength then
        throw new IllegalArgumentException(s"$label must be at most $maxLength characters")
      // Reject embedded C0 controls and DEL. validateEmail already rejects ALL
      // whitespace + controls because email syntax forbids them; profile fields
      // (displayName, heroName, preferredSite, timeZone) allow internal spaces
      // for things like "John Smith", but a newline / NUL / ESC in a stored
      // profile field has no legitimate use and would either confuse JSON
      // consumers, corrupt audit log fields when the value eventually surfaces,
      // or trip line-oriented dashboards downstream.
      if value.exists(ch => ch.toInt < 0x20 || ch.toInt == 0x7F) then
        throw new IllegalArgumentException(s"$label must not contain control characters")
      value
    }

  private def defaultDisplayNameFor(email: String): String =
    email.takeWhile(_ != '@') match
      case "" => "SICFUN User"
      case value => value

  private def preferNonBlank(existing: String, fallback: String): String =
    if existing.trim.nonEmpty then existing else fallback

  private def normalizeEmail(email: String): String =
    email.trim.toLowerCase(java.util.Locale.ROOT)

  private def hashPassword(password: String, updatedAtEpochMs: Long): LocalPasswordCredential =
    val salt = randomBytes(PasswordSaltBytes)
    val hash = pbkdf2(password, salt, PasswordIterations, PasswordKeyLengthBits)
    LocalPasswordCredential(
      saltBase64 = Base64.getEncoder.encodeToString(salt),
      hashBase64 = Base64.getEncoder.encodeToString(hash),
      iterations = PasswordIterations,
      keyLengthBits = PasswordKeyLengthBits,
      updatedAtEpochMs = updatedAtEpochMs
    )

  private def verifyPassword(password: String, credential: LocalPasswordCredential): Boolean =
    val salt = Base64.getDecoder.decode(credential.saltBase64)
    val expected = Base64.getDecoder.decode(credential.hashBase64)
    val actual = pbkdf2(password, salt, credential.iterations, credential.keyLengthBits)
    MessageDigest.isEqual(expected, actual)

  // Fixed salt used only to make the "no such user" and "OIDC-only user" branches
  // of authenticateLocal spend equivalent CPU time to the "real verify" branch.
  // The output is discarded so no comparison occurs; the salt value cannot leak
  // useful information.
  private val PlaceholderSalt: Array[Byte] = Array.fill(PasswordSaltBytes)(0.toByte)

  private def dummyVerifyPassword(password: String): Unit =
    val _ = pbkdf2(password, PlaceholderSalt, PasswordIterations, PasswordKeyLengthBits)

  private def pbkdf2(
      password: String,
      salt: Array[Byte],
      iterations: Int,
      keyLengthBits: Int
  ): Array[Byte] =
    val factory = SecretKeyFactory.getInstance(PasswordAlgorithm)
    val spec = new PBEKeySpec(password.toCharArray, salt, iterations, keyLengthBits)
    try factory.generateSecret(spec).getEncoded
    finally spec.clearPassword()

  // Shared SecureRandom: instantiation is non-trivial (the JDK's strong-DRBG
  // setup samples OS entropy on first use), and SecureRandom.nextBytes is
  // documented thread-safe. Reusing one instance avoids repeating that work
  // for every session/CSRF/OIDC-state mint -- which on login fires four times
  // (session token, csrf token, oidc state, code verifier).
  private val secureRandom: SecureRandom = new SecureRandom()

  private def randomBytes(length: Int): Array[Byte] =
    val bytes = new Array[Byte](length)
    secureRandom.nextBytes(bytes)
    bytes

  private def randomBase64Url(length: Int): String =
    Base64.getUrlEncoder.withoutPadding().encodeToString(randomBytes(length))

  private def codeChallenge(codeVerifier: String): String =
    Base64.getUrlEncoder.withoutPadding()
      .encodeToString(MessageDigest.getInstance("SHA-256").digest(codeVerifier.getBytes(StandardCharsets.US_ASCII)))

  private def sha256Hex(raw: String): String =
    MessageDigest.getInstance("SHA-256")
      .digest(raw.getBytes(StandardCharsets.UTF_8))
      .map("%02x".format(_))
      .mkString

  private def extractCookie(cookieHeader: Option[String], cookieName: String): Option[String] =
    // Find the FIRST NON-EMPTY value among segments matching `<name>=`. RFC
    // 6265 permits multiple cookies with the same name and leaves ordering
    // up to the user agent, so picking strictly the first match (and then
    // dropping it if empty) would let an attacker with a foothold on a
    // sibling subdomain in plain-HTTP mode plant an empty
    // `sicfun_session=` cookie that masks the real session and logs the
    // victim out. Continuing the scan past empty matches defeats that
    // narrow DoS without relying on which cookie the browser sorts first.
    cookieHeader.flatMap { header =>
      header.split(';').iterator
        .map(_.trim)
        .filter(_.startsWith(s"$cookieName="))
        .map(_.substring(cookieName.length + 1))
        .find(_.nonEmpty)
    }

  // Resolve the wire-format cookie name for a given secure-mode setting. When
  // the deployment is HTTPS-backed (cookieSecure=true), use the `__Host-` prefix
  // so the browser enforces three extra invariants: the cookie was set over
  // HTTPS, Path=/, and no Domain attribute. That blocks a sibling subdomain
  // (compromised or rogue) from overwriting our session cookie or planting a
  // pre-set one. RFC 6265 sec 4.1.3.
  private[web] def sessionCookieName(secure: Boolean): String =
    if secure then s"__Host-$DefaultSessionCookieName" else DefaultSessionCookieName

  private def sessionCookieHeader(token: String, ttlMs: Long, secure: Boolean): String =
    val parts = Vector.newBuilder[String]
    parts += s"${sessionCookieName(secure)}=$token"
    parts += "Path=/"
    parts += s"Max-Age=${math.max(1L, ttlMs / 1000L)}"
    parts += "HttpOnly"
    parts += "SameSite=Lax"
    if secure then
      parts += "Secure"
    parts.result().mkString("; ")

  // OIDC state cookie binds the random `state` value embedded in the
  // authorization URL to the user-agent that initiated the flow. Without this
  // binding, an attacker who completes their own authorization up to the
  // redirect step can forward `?state=X&code=ATTACKER_CODE` to a victim; our
  // callback would happily exchange the code and the victim's browser would
  // end up holding a session for the attacker's account (OAuth 2.0 BCP
  // "covert-redirect" / "login CSRF"). With the cookie, only the user agent
  // that received the Set-Cookie at /start can supply the matching cookie at
  // /callback.
  private[web] def oidcStateCookieName(secure: Boolean): String =
    if secure then s"__Host-$DefaultOidcStateCookieName" else DefaultOidcStateCookieName

  private def oidcStateCookieHeader(state: String, ttlMs: Long, secure: Boolean): String =
    val parts = Vector.newBuilder[String]
    parts += s"${oidcStateCookieName(secure)}=$state"
    parts += "Path=/"
    parts += s"Max-Age=${math.max(1L, ttlMs / 1000L)}"
    parts += "HttpOnly"
    // SameSite=Lax: the OIDC callback is a top-level GET navigation back from
    // the provider, which Lax permits. SameSite=Strict would block the cookie
    // on the callback hop and break the flow.
    parts += "SameSite=Lax"
    if secure then
      parts += "Secure"
    parts.result().mkString("; ")

  private def clearOidcStateCookieHeader(secure: Boolean): String =
    val parts = Vector.newBuilder[String]
    parts += s"${oidcStateCookieName(secure)}="
    parts += "Path=/"
    parts += "Max-Age=0"
    parts += "HttpOnly"
    parts += "SameSite=Lax"
    if secure then
      parts += "Secure"
    parts.result().mkString("; ")

  private def clearSessionCookieHeader(secure: Boolean): String =
    val parts = Vector.newBuilder[String]
    parts += s"${sessionCookieName(secure)}="
    parts += "Path=/"
    parts += "Max-Age=0"
    parts += "HttpOnly"
    parts += "SameSite=Lax"
    if secure then
      parts += "Secure"
    parts.result().mkString("; ")

  private def formEncode(values: Vector[(String, String)]): String =
    values.map { case (name, value) =>
      s"${urlEncode(name)}=${urlEncode(value)}"
    }.mkString("&")

  private def urlEncode(value: String): String =
    URLEncoder.encode(value, StandardCharsets.UTF_8).replace("+", "%20")

  def oidcSuccessRedirect: String = OidcSuccessRedirect

  def oidcFailureRedirect(error: String): String =
    s"$OidcFailureRedirectPrefix${urlEncode(error)}"
