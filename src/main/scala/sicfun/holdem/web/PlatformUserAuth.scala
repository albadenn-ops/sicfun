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
      oidcProviders: Vector[OidcProvider] = Vector.empty
  ):
    require(sessionTtlMs > 0L, "sessionTtlMs must be positive")

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

    def startOidc(providerId: String): Either[String, String] =
      providersById.get(providerId).toRight(s"unknown OIDC provider '$providerId'").map { provider =>
        val codeVerifier = randomBase64Url(OidcCodeVerifierBytes)
        val state = oidcStateStore.issue(provider.id, codeVerifier)
        provider.authorizationUri(state, codeChallenge(codeVerifier))
      }

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
        try
          val store = new JsonUserStore(config.storePath)
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

  private final case class StoreState(users: Vector[StoredUser])

  private final class JsonUserStore(path: Path):
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
        val now = System.currentTimeMillis()
        findByProviderIdentityInternal(providerId, identity.subject) match
          case Some(existing) =>
            val updated = existing.copy(
              email = normalizedEmail,
              profile = existing.profile.copy(
                displayName = preferNonBlank(existing.profile.displayName, identity.displayName),
                avatarUrl = identity.avatarUrl.orElse(existing.profile.avatarUrl)
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
                    displayName = identity.displayName,
                    avatarUrl = identity.avatarUrl
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
      state = next
      writeState(path, next)

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
          Option(sessions.get(key))
            .filter(_.expiresAtEpochMs > nowMillis())
            .flatMap { session =>
              val now = nowMillis()
              sessions.put(
                key,
                session.copy(
                  expiresAtEpochMs = now + sessionTtlMs,
                  lastSeenAtEpochMs = now
                )
              )
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
          iterator.remove()

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
      Option(states.remove(state))
        .filter(record => record.providerId == providerId && nowMillis() - record.issuedAtEpochMs <= flowTtlMs)

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
      Files.deleteIfExists(temp)

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
    val trimmed = password.trim
    if trimmed.length < 10 then
      throw new IllegalArgumentException("password must be at least 10 characters")
    if trimmed.length > MaxPasswordLength then
      throw new IllegalArgumentException(s"password must be at most $MaxPasswordLength characters")

  private def sanitizeDisplayName(displayName: Option[String]): Option[String] =
    sanitizeOptionalField(displayName, "displayName", 96)

  private def sanitizeOptionalField(raw: Option[String], label: String, maxLength: Int): Option[String] =
    raw.map(_.trim).filter(_.nonEmpty).map { value =>
      if value.length > maxLength then
        throw new IllegalArgumentException(s"$label must be at most $maxLength characters")
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

  private def randomBytes(length: Int): Array[Byte] =
    val bytes = new Array[Byte](length)
    new SecureRandom().nextBytes(bytes)
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
    cookieHeader
      .flatMap(_.split(';').iterator.map(_.trim).find(_.startsWith(s"$cookieName=")))
      .map(_.substring(cookieName.length + 1))
      .filter(_.nonEmpty)

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
