package sicfun.holdem.web

import sicfun.holdem.runtime.TexasHoldemPlayingHall

import munit.FunSuite
import ujson.Value

import java.io.ByteArrayInputStream
import java.net.{InetAddress, URI}
import java.net.http.{HttpClient, HttpRequest, HttpResponse}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import java.util.Base64
import java.util.concurrent.{CountDownLatch, TimeUnit}
import java.util.zip.GZIPInputStream
import scala.jdk.CollectionConverters.*

/** Tests for [[HandHistoryReviewServer]], the embedded HTTP server that
  * exposes the hand history review service via a REST API.
  *
  * Uses real HTTP requests against a locally-bound server (port 0 for
  * ephemeral allocation) with configurable backends (immediate, blocking,
  * and busy) to test concurrency and timeout behavior.
  *
  * Coverage:
  *   - '''Shutdown grace period''': millisecond-to-second rounding for
  *     `HttpServer.stop` delay
  *   - '''Health / readiness probes''': JSON fields (ok, ready, draining,
  *     acceptingAnalysisJobs, model source, rate limit config, queue stats),
  *     security headers (X-Content-Type-Options, X-Frame-Options, CSP)
  *   - '''Static content''': serves index.html from the configured static dir
  *   - '''Upload size enforcement''': 413 for oversized payloads
  *   - '''Basic auth''': protects UI and analysis routes while leaving health
  *     and readiness probes open; returns 401 + WWW-Authenticate header
  *   - '''Platform user auth''': local registration, profile updates, per-user
  *     job ownership isolation, session management, and logout
  *   - '''OIDC flow''': advertises Google provider, completes authorization
  *     redirect + callback with a fake OIDC provider
  *   - '''Drain signal''': flips readiness to 503, rejects new analysis jobs,
  *     recovers when signal file is removed
  *   - '''Async job lifecycle''': submit returns 202 with job ID + status URL,
  *     server remains responsive during analysis, polling returns running/
  *     completed status with Retry-After headers
  *   - '''Rate limiting''': per-bucket (submit, job-status) limits with 429
  *     responses; zero limits disable throttling; trusted client IP header
  *     isolates rate limit buckets per proxied client; ambiguous/invalid
  *     header values fall back to remote address; trusted proxy allowlist
  *   - '''Bounded queue''': 503 when max concurrent + queued jobs exceeded;
  *     readiness degrades to "queue-full"
  *   - '''Error surfacing''': terminal failure status with 500 error code
  *   - '''Analysis timeout''': stuck jobs fail with 504; timed-out workers
  *     keep readiness failed until the worker thread exits
  *   - '''End-to-end proof''': async review of an exact-GTO hall export
  *     through the full server pipeline
  *   - '''Bind error''': duplicate port binding returns a descriptive error
  *   - '''Default host''': direct launches default to 127.0.0.1
  *   - '''Bind safety''': unauthenticated non-loopback binds require an
  *     explicit override
  *   - '''Networked user auth safety''': non-loopback session auth requires
  *     secure cookies, and non-loopback OIDC requires HTTPS callbacks unless
  *     explicitly overridden
  */
class HandHistoryReviewServerTest extends FunSuite:

  private val httpClient = HttpClient.newHttpClient()
  private val sampleAnalysisResult = ujson.read(
    """{
      |  "site": "PokerStars",
      |  "heroName": "Hero",
      |  "handsImported": 1,
      |  "handsAnalyzed": 1,
      |  "handsSkipped": 0,
      |  "decisionsAnalyzed": 1,
      |  "mistakes": 1,
      |  "totalEvLost": 1.25,
      |  "biggestMistakeEv": 1.25,
      |  "modelSource": "test-model",
      |  "warnings": [],
      |  "decisions": [
      |    {
      |      "handId": "PokerStars-1001",
      |      "street": "Flop",
      |      "heroCards": "AcKh",
      |      "actualAction": "Fold",
      |      "recommendedAction": "Call",
      |      "actualEv": -1.0,
      |      "recommendedEv": 0.25,
      |      "evDifference": -1.25,
      |      "heroEquityMean": 0.42
      |    }
      |  ],
      |  "opponents": [
      |    {
      |      "playerName": "Villain",
      |      "handsObserved": 1,
      |      "archetype": "balanced",
      |      "hints": [
      |        {
      |          "ruleId": "test-rule",
      |          "text": "test hint",
      |          "metrics": [0.1, 0.2, 0.3, 0.4]
      |        }
      |      ]
      |    }
      |  ],
      |  "trace": {
      |    "request": {
      |      "rawHeroName": "Hero",
      |      "normalizedHeroName": "Hero",
      |      "requestedSite": null,
      |      "handHistoryBytes": 19
      |    },
      |    "import": {
      |      "handsImported": 1,
      |      "siteResolved": "PokerStars",
      |      "heroNameResolved": "Hero",
      |      "distinctPlayersObserved": 2
      |    },
      |    "hands": [
      |      {
      |        "handId": "PokerStars-1001",
      |        "status": "analyzed",
      |        "playerCount": 2,
      |        "heroNameResolved": "Hero",
      |        "heroCardsPresent": true,
      |        "decisionsAnalyzed": 1,
      |        "skipReason": null,
      |        "warning": null
      |      }
      |    ],
      |    "summary": {
      |      "handsImported": 1,
      |      "handsAnalyzed": 1,
      |      "handsSkipped": 0,
      |      "decisionsAnalyzed": 1,
      |      "mistakes": 1,
      |      "totalEvLost": 1.25,
      |      "biggestMistakeEv": 1.25,
      |      "warningCount": 0,
      |      "opponentsProfiled": 1
      |    }
      |  }
      |}""".stripMargin
  )
  private val validUploadPayload = """{"handHistoryText":"PokerStars Hand #1","site":"auto","heroName":"Hero"}"""
  private val samplePlayingHallResult = ujson.read(
    """{
      |  "request": {
      |    "hands": 240,
      |    "tableCount": 2,
      |    "playerCount": 6,
      |    "heroStyle": "adaptive",
      |    "heroPosition": "Button",
      |    "gtoMode": "exact",
      |    "villainPool": ["tag", "gto"],
      |    "heroExplorationRate": 0,
      |    "raiseSize": 2.5,
      |    "bunchingTrials": 40,
      |    "equityTrials": 240,
      |    "learnEveryHands": 0,
      |    "learningWindowSamples": 200,
      |    "saveReviewHandHistory": false,
      |    "fullRing": false,
      |    "seed": 42
      |  },
      |  "summary": {
      |    "handsPlayed": 240,
      |    "tableCount": 2,
      |    "playerCount": 6,
      |    "heroNetChips": 13.5,
      |    "heroBbPer100": 5.6,
      |    "heroWins": 111,
      |    "heroTies": 8,
      |    "heroLosses": 121,
      |    "actionCounts": {
      |      "Fold": 41,
      |      "Call": 77,
      |      "Raise(2.5)": 122
      |    },
      |    "retrains": 0,
      |    "modelId": "uniform-baseline",
      |    "outDir": "data/web-playing-hall/test-run",
      |    "exactGtoCacheHits": 0,
      |    "exactGtoCacheMisses": 0,
      |    "exactGtoCacheHitRate": 0,
      |    "exactGtoSolvedByProvider": {},
      |    "exactGtoServedByProvider": {},
      |    "perVillainNetChips": {
      |      "Villain-1": -7.25,
      |      "Villain-2": -6.25
      |    },
      |    "overlayStats": null,
      |    "outputFiles": [
      |      "data/web-playing-hall/test-run/hands.tsv"
      |    ]
      |  }
      |}""".stripMargin
  )
  private val validPlayingHallPayload =
    """{"hands":120,"tableCount":2,"playerCount":6,"heroStyle":"adaptive","heroPosition":"Button","gtoMode":"exact","villainPool":["tag","gto"],"heroExplorationRate":0,"raiseSize":2.5,"bunchingTrials":40,"equityTrials":240}"""

  test("API JSON responses gzip when the client accepts gzip and the payload is large enough") {
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        val compressed = get(s"$baseUri/api/health", Map("Accept-Encoding" -> "gzip"))
        assertEquals(compressed.statusCode(), 200)
        assertEquals(headerValue(compressed, "Content-Encoding"), Some("gzip"))
        assertEquals(headerValue(compressed, "Vary"), Some("Accept-Encoding"))
        assertEquals(headerValue(compressed, "Cache-Control"), Some("no-store"))

        val plain = get(s"$baseUri/api/health")
        assertEquals(plain.statusCode(), 200)
        assertEquals(headerValue(plain, "Content-Encoding"), None)
        assertEquals(headerValue(plain, "Vary"), Some("Accept-Encoding"),
          clue = "Vary must still be set even when the response itself is not compressed")
      }
    }
  }

  test("responses below the gzip threshold skip compression even when accepted") {
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        // The 403 path-traversal response is "forbidden" (9 bytes), well under the 256-byte
        // MinGzipSize threshold. text/plain is compressible, so Vary is still set.
        val tinyResponse = get(s"$baseUri/../etc/passwd", Map("Accept-Encoding" -> "gzip"))
        assertEquals(tinyResponse.statusCode(), 403)
        assertEquals(tinyResponse.body(), "forbidden")
        assertEquals(headerValue(tinyResponse, "Content-Encoding"), None,
          clue = s"tiny response (${tinyResponse.body().length} bytes) should not be gzipped")
        assertEquals(headerValue(tinyResponse, "Vary"), Some("Accept-Encoding"),
          clue = "Vary still set so caches partition variants by Accept-Encoding")
      }
    }
  }

  test("registration rejects emails beyond RFC 5321's 254-character cap") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"
          // Local part 300 chars + "@example.com" = well over 254. Passes the
          // structural @-and-dot check but must be rejected by the length cap so
          // the user store doesn't get polluted with oversize records.
          val oversizeEmail = ("a" * 300) + "@example.com"
          val payload =
            s"""{"email":"$oversizeEmail","password":"correct-horse-battery","displayName":"Test"}"""
          val rejected = postJson(s"$baseUri/api/auth/register", payload)
          assertEquals(rejected.statusCode(), 400,
            clue = jsonBody(rejected).render(indent = 2))
          assert(jsonBody(rejected)("error").str.contains("at most"),
            clue = s"expected max-length rejection, got: ${jsonBody(rejected)("error").str}")
        }
      }
    }
  }

  test("session resolution skips empty cookie matches so a planted empty cookie does not log the user out") {
    // RFC 6265 permits multiple cookies with the same name and leaves
    // ordering implementation-defined. In plain-HTTP mode the cookie name is
    // `sicfun_session` (no __Host- prefix), so a sibling subdomain attacker
    // could set `sicfun_session=` on the victim. The browser would then send
    // BOTH the planted empty cookie AND the real session cookie in the
    // Cookie header. A parser that takes strictly the first `sicfun_session=`
    // segment and bails on the empty value would log the victim out;
    // continuing past empty matches finds the real session and resolves it.
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"
          val register = postJson(
            s"$baseUri/api/auth/register",
            """{"email":"victim@example.com","password":"correct-horse-battery","displayName":"Victim"}"""
          )
          assertEquals(register.statusCode(), 201)
          val realSessionCookie = sessionCookie(register)

          // Empty cookie planted FIRST, real cookie second.
          val pollutedFirst = Map("Cookie" -> s"sicfun_session=; $realSessionCookie")
          val meFirst = getJsonWithHeaders(s"$baseUri/api/auth/me", pollutedFirst)
          assertEquals(meFirst("authenticated").bool, true,
            clue = "real session must still resolve when an empty cookie is planted before it")

          // Empty cookie planted LAST -- belt-and-braces.
          val pollutedLast = Map("Cookie" -> s"$realSessionCookie; sicfun_session=")
          val meLast = getJsonWithHeaders(s"$baseUri/api/auth/me", pollutedLast)
          assertEquals(meLast("authenticated").bool, true,
            clue = "real session must still resolve when an empty cookie is planted after it")

          // No real cookie, only the empty plant -> not authenticated (regression test
          // for the simple negative case so we don't accidentally accept '=' as a session).
          val emptyOnly = Map("Cookie" -> "sicfun_session=")
          val meEmpty = getJsonWithHeaders(s"$baseUri/api/auth/me", emptyOnly)
          assertEquals(meEmpty("authenticated").bool, false,
            clue = "empty-only sicfun_session= must not authenticate anyone")
        }
      }
    }
  }

  test("body-reading endpoints reject Content-Encoding other than identity") {
    // The server reads the request body as raw UTF-8 and parses as JSON; it
    // does NOT decompress. Before this guard, a client that sent
    // Content-Encoding: gzip would have its gzipped bytes treated as JSON,
    // yielding a confusing 'invalid JSON request' 400. Worse, decompression
    // support would have been a footgun -- a hostile client could mail in a
    // small compressed payload that decompresses to many MB of JSON, evading
    // the maxUploadBytes cap before parsing even starts. Reject explicitly
    // with 415 so the client knows the contract.
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"
          val body = """{"email":"alice@example.com","password":"correct-horse-battery"}"""

          // gzip Content-Encoding -> 415.
          val gzipPost = HttpRequest.newBuilder(URI.create(s"$baseUri/api/auth/login"))
            .header("Content-Type", "application/json")
            .header("Content-Encoding", "gzip")
            .POST(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8))
            .build()
          val gzipResp = httpClient.send(gzipPost, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8))
          assertEquals(gzipResp.statusCode(), 415,
            clue = "Content-Encoding: gzip must be rejected with 415")
          assert(jsonBody(gzipResp)("error").str.contains("Content-Encoding"),
            clue = "415 error body should mention Content-Encoding so the client can see what failed")

          // identity Content-Encoding -> accepted (gets a normal 401 for bad creds).
          // RFC 7231 permits clients to explicitly send Content-Encoding: identity
          // to assert no encoding; reject only the actually-encoded forms.
          val identityPost = HttpRequest.newBuilder(URI.create(s"$baseUri/api/auth/login"))
            .header("Content-Type", "application/json")
            .header("Content-Encoding", "identity")
            .POST(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8))
            .build()
          val identityResp = httpClient.send(identityPost, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8))
          assertEquals(identityResp.statusCode(), 401,
            clue = "Content-Encoding: identity must be accepted (bad creds is the legitimate downstream outcome)")
        }
      }
    }
  }

  test("body-reading endpoints require Content-Type: application/json") {
    // Belt-and-braces login-CSRF / form-CSRF mitigation: a hostile cross-origin
    // site auto-submitting a <form action="/api/auth/login"> would deliver a
    // browser-default application/x-www-form-urlencoded body. JSON parsing
    // would already reject that with a 400, but returning a clean 415
    // Unsupported Media Type before any body parsing makes the contract
    // explicit and consistent with the documented same-origin model.
    // application/json with a charset suffix must still be accepted because
    // RFC 8259 allows it and standards-compliant clients emit it.
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"
          val body = """{"email":"alice@example.com","password":"correct-horse-battery"}"""

          // Form-encoded -> 415.
          val formPost = HttpRequest.newBuilder(URI.create(s"$baseUri/api/auth/login"))
            .header("Content-Type", "application/x-www-form-urlencoded")
            .POST(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8))
            .build()
          val formResp = httpClient.send(formPost, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8))
          assertEquals(formResp.statusCode(), 415,
            clue = "form-encoded Content-Type must be rejected with 415")

          // text/plain -> 415.
          val plainPost = HttpRequest.newBuilder(URI.create(s"$baseUri/api/auth/login"))
            .header("Content-Type", "text/plain")
            .POST(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8))
            .build()
          val plainResp = httpClient.send(plainPost, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8))
          assertEquals(plainResp.statusCode(), 415,
            clue = "text/plain Content-Type must be rejected with 415")

          // application/json with charset -> accepted (gets a normal 401 for bad creds).
          val jsonWithCharset = HttpRequest.newBuilder(URI.create(s"$baseUri/api/auth/login"))
            .header("Content-Type", "application/json; charset=utf-8")
            .POST(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8))
            .build()
          val jsonResp = httpClient.send(jsonWithCharset, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8))
          assertEquals(jsonResp.statusCode(), 401,
            clue = "application/json; charset=utf-8 must be accepted; 401 (bad creds) is the legitimate downstream outcome")
        }
      }
    }
  }

  test("/api/health surfaces userAuthMaxUsers and userAuthStoredUsers so dashboards can show usage vs cap") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath, maxUsers = 50))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          // No users yet: stored=0, max=50, sessions=0.
          val healthBefore = getJson(s"$baseUri/api/health")
          assertEquals(healthBefore("userAuthMaxUsers").num.toInt, 50)
          assertEquals(healthBefore("userAuthStoredUsers").num.toInt, 0)
          assertEquals(healthBefore("userAuthActiveSessions").num.toInt, 0,
            clue = "no users registered, no sessions issued, so active-session count starts at 0")
          assertEquals(healthBefore("userAuthPendingOidcFlows").num.toInt, 0,
            clue = "no OIDC /start calls yet, so pending-flow count starts at 0")

          // Register one user.
          val register = postJson(s"$baseUri/api/auth/register",
            """{"email":"alice@example.com","password":"correct-horse-battery","displayName":"Alice"}""")
          assertEquals(register.statusCode(), 201)

          val healthAfter = getJson(s"$baseUri/api/health")
          assertEquals(healthAfter("userAuthMaxUsers").num.toInt, 50)
          assertEquals(healthAfter("userAuthStoredUsers").num.toInt, 1,
            clue = "stored count should reflect the newly registered user")
          assertEquals(healthAfter("userAuthActiveSessions").num.toInt, 1,
            clue = "successful registration creates an initial session, so active-session count is 1")
        }
      }
    }
  }

  test("/api/health user-auth fields are null when platform-user auth is not configured") {
    // basic-auth / no-auth deployments have no user store, so the cap and
    // the live count are both null -- dashboards should treat null as
    // "not applicable" rather than show a misleading 0.
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val health = getJson(s"$baseUri/api/health")
        assert(health("userAuthMaxUsers").isNull,
          clue = s"userAuthMaxUsers must be null when platform-user auth is disabled; got: ${health("userAuthMaxUsers")}")
        assert(health("userAuthStoredUsers").isNull,
          clue = s"userAuthStoredUsers must be null when platform-user auth is disabled; got: ${health("userAuthStoredUsers")}")
        assert(health("userAuthActiveSessions").isNull,
          clue = s"userAuthActiveSessions must be null when platform-user auth is disabled; got: ${health("userAuthActiveSessions")}")
        assert(health("userAuthPendingOidcFlows").isNull,
          clue = s"userAuthPendingOidcFlows must be null when platform-user auth is disabled; got: ${health("userAuthPendingOidcFlows")}")
      }
    }
  }

  // Pin the four OIDC start-URL parameters the deploy doc + runbook +
  // OAuth 2.0 security model depend on. The existing OIDC tests in this
  // file all use FakeOidcProvider (whose authorizationUri returns a
  // hardcoded test URL without the production parameters), so the REAL
  // GoogleOidcProvider.authorizationUri is otherwise unexercised by the
  // suite -- a refactor that dropped any of the four documented
  // parameters would silently break the documented behavior without
  // any test failure. The four parameters and what depends on each:
  //   - `prompt=select_account`: deploy doc's Optional-OIDC bullet
  //     explicitly says "every /start forces Google's account picker
  //     even when the user has exactly one Google account already
  //     signed in -- useful for multi-account users (Work vs personal
  //     Google), and an explainer for the 'why does it ask me every
  //     time?' support question on single-account users." Dropping the
  //     parameter silently regresses the multi-account-user experience.
  //   - `code_challenge_method=S256`: the deploy doc's lead line says
  //     "OIDC + PKCE (SHA-256 challenge)". RFC 7636 also defines
  //     `plain` (no transform) which is strictly weaker -- an attacker
  //     who intercepts the authorization redirect URL gets the
  //     verifier directly. Dropping to plain silently weakens PKCE.
  //   - `response_type=code`: the authorization-code flow with PKCE
  //     is the only OAuth 2.0 flow the BCP still endorses. The
  //     deprecated `token` (implicit) flow puts the access token in
  //     the URL fragment where browser history / referrers / log
  //     analysers leak it. A refactor to `token` would silently
  //     regress to a deprecated insecure flow.
  //   - `scope=openid email profile` (URL-encoded as openid%20email
  //     %20profile per formEncode/urlEncode): the runbook's "Google
  //     sign-in fails" triage entry says "openid, email, and profile
  //     are all required. A missing email scope is a DIFFERENT
  //     failure mode -- the consent screen still renders fine, the
  //     callback DOES fire, and the auth.oidc.failure line appears
  //     with the verified-email reason." Dropping one of the three
  //     silently regresses to that failure path.
  // Pure unit test of GoogleOidcProvider.authorizationUri -- no server,
  // no HTTP. Same regression-pin pattern as the other "documented
  // contract has no test" pins recently added.
  test("GoogleOidcProvider.authorizationUri carries the four documented OIDC start-URL parameters (prompt=select_account, code_challenge_method=S256, response_type=code, scope=openid email profile)") {
    val provider = new PlatformUserAuth.GoogleOidcProvider(
      PlatformUserAuth.GoogleOidcConfig(
        clientId = "test-client-id",
        clientSecret = "test-client-secret",
        redirectUri = "https://example.test/api/auth/oidc/google/callback"
      )
    )
    val authUri = provider.authorizationUri("test-state-value", "test-code-challenge-value")
    assert(authUri.contains("prompt=select_account"),
      s"deploy doc's Optional-OIDC bullet documents prompt=select_account as the multi-account-user picker trigger; missing from authorizationUri output. Full URI: $authUri")
    assert(authUri.contains("code_challenge_method=S256"),
      s"deploy doc's lead line documents OIDC + PKCE (SHA-256 challenge); a refactor to `plain` would silently weaken PKCE protection. Full URI: $authUri")
    assert(authUri.contains("response_type=code"),
      s"OAuth 2.0 BCP recommends the authorization-code flow with PKCE; a refactor to `token` (implicit) would silently regress to a deprecated insecure flow that leaks the access token via the URL fragment. Full URI: $authUri")
    // formEncode uses urlEncode which replaces `+` with `%20`, so the
    // space-bearing scope string is URL-encoded as `%20`-separated.
    assert(authUri.contains("scope=openid%20email%20profile"),
      s"runbook 'Google sign-in fails' triage entry lists openid + email + profile as ALL required; dropping any one silently regresses to the documented separate failure mode (consent screen renders, callback fires, auth.oidc.failure with verified-email reason). Full URI: $authUri")
  }

  // Pins the userAuthPendingOidcFlows counter that operators rely on for
  // triage per the runbook's OIDC silent-failure entry (added in c7b18cd):
  // "userAuthPendingOidcFlows counting up steadily ... WITHOUT a
  // corresponding stream of auth.oidc.success / auth.oidc.failure log
  // lines is the signature of 'users start, never finish.'" The triage
  // depends on the counter incrementing on /start and decrementing when
  // the matching /callback consumes the state-store entry. If a future
  // refactor accidentally broke either side of that round-trip (always
  // reported 0, decremented on /start, etc.), the operator triage
  // guidance would silently mislead. Pin the round trip in a test so a
  // future regression is caught by CI rather than by a confused operator.
  test("userAuthPendingOidcFlows increments on /start and decrements on the matching /callback consume") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        val provider = new FakeOidcProvider
        withServer(
          staticDir,
          platformAuth = Some(
            PlatformUserAuth.Config(
              storePath = storePath,
              oidcProviders = Vector(provider)
            )
          )
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          // Baseline: no /start calls yet, counter at 0.
          val healthBefore = getJson(s"$baseUri/api/health")
          assertEquals(healthBefore("userAuthPendingOidcFlows").num.toInt, 0,
            clue = "no OIDC flows started, so pending count must be 0")

          // /start creates a state-store entry and returns a 302 redirect
          // to the provider. The counter increments by 1 -- this is the
          // signal operators use to spot the "users start, never finish"
          // pattern when /callback never lands.
          val start = get(s"$baseUri${provider.startPath}")
          assertEquals(start.statusCode(), 302)
          val redirect = headerValue(start, "Location").getOrElse(fail("missing OIDC redirect Location header"))
          val state = queryParam(redirect, "state").getOrElse(fail("missing OIDC state query parameter"))
          val stateCookie = headerValue(start, "Set-Cookie")
            .map(_.takeWhile(_ != ';'))
            .getOrElse(fail("missing OIDC state cookie"))

          val healthMidFlow = getJson(s"$baseUri/api/health")
          assertEquals(healthMidFlow("userAuthPendingOidcFlows").num.toInt, 1,
            clue = "after /start, exactly one OIDC flow is pending callback completion -- this is the field operators key on for the 'users start, never finish' triage pattern")

          // Matching /callback consumes the state-store entry. The counter
          // returns to 0 -- the pending count being 0 again is operationally
          // distinct from "no /start ever happened" because log lines
          // (auth.oidc.start + auth.oidc.success) still record the flow.
          val callback = get(
            s"$baseUri${provider.callbackPath}?state=$state&code=test-code",
            Map("Cookie" -> stateCookie)
          )
          assertEquals(callback.statusCode(), 302,
            clue = "callback should succeed with valid state + cookie")

          val healthAfter = getJson(s"$baseUri/api/health")
          assertEquals(healthAfter("userAuthPendingOidcFlows").num.toInt, 0,
            clue = "after the callback consumes the state entry, pending count returns to 0 -- the round trip closes cleanly")
        }
      }
    }
  }

  test("OIDC sign-up also honors the max-user cap so it cannot be a back door past the local-registration defense") {
    // The disk-fill defense applies to ALL new-user paths, not just local
    // registration. Before this fix, upsertOidcIdentity skipped the cap
    // check, so a deployment with OIDC enabled would let any new Google
    // account create a user record after the local path was already
    // saturated. Use a fake provider whose exchangeCode returns a never-
    // seen subject so the OIDC flow always tries to CREATE a new user.
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        // Cap at 1, then register one local user. The next OIDC sign-in
        // (which would otherwise create a 2nd user) must hit the same
        // 'temporarily unavailable' rejection.
        val provider = new FakeOidcProvider
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(
            storePath = storePath,
            maxUsers = 1,
            oidcProviders = Vector(provider)
          ))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          val register = postJson(s"$baseUri/api/auth/register",
            """{"email":"first@example.com","password":"correct-horse-battery","displayName":"First"}""")
          assertEquals(register.statusCode(), 201)

          // OIDC flow: start to get state cookie, then callback.
          val start = get(s"$baseUri${provider.startPath}")
          val state = queryParam(headerValue(start, "Location").getOrElse(""), "state")
            .getOrElse(fail("missing OIDC state"))
          val stateCookie = headerValue(start, "Set-Cookie")
            .map(_.takeWhile(_ != ';'))
            .getOrElse(fail("missing state cookie"))

          val callback = get(
            s"$baseUri${provider.callbackPath}?state=$state&code=test-code",
            Map("Cookie" -> stateCookie)
          )
          // The OIDC sign-in succeeded as far as cookie + state validation,
          // but the upsert step hit the maxUsers cap, so the failure
          // redirect surfaces the same generic 'temporarily unavailable'
          // message the local path emits.
          assertEquals(callback.statusCode(), 302)
          val location = headerValue(callback, "Location").getOrElse(fail("missing Location"))
          assert(location.contains("/?auth_error="),
            s"OIDC callback rejected at cap should redirect to the failure landing; got: $location")
          assert(location.contains("temporarily%20unavailable") || location.contains("temporarily+unavailable"),
            clue = s"failure redirect should carry the 'temporarily unavailable' reason (URL-encoded); got: $location")
        }
      }
    }
  }

  test("registration rejects once the configured max-user count is reached") {
    // Defense against slow disk-fill via public registration abuse: the
    // user store has a hard cap (default 100k, configurable). Beyond the
    // cap, registrations return a generic 'temporarily unavailable'
    // message that does not leak the cap to a probing attacker.
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath, maxUsers = 2))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          val first = postJson(s"$baseUri/api/auth/register",
            """{"email":"a@example.com","password":"correct-horse-battery","displayName":"A"}""")
          assertEquals(first.statusCode(), 201)

          val second = postJson(s"$baseUri/api/auth/register",
            """{"email":"b@example.com","password":"correct-horse-battery","displayName":"B"}""")
          assertEquals(second.statusCode(), 201)

          // Third registration -> cap hit, generic rejection.
          val third = postJson(s"$baseUri/api/auth/register",
            """{"email":"c@example.com","password":"correct-horse-battery","displayName":"C"}""")
          assertEquals(third.statusCode(), 400)
          val errorMessage = jsonBody(third)("error").str
          assert(errorMessage.contains("temporarily unavailable"),
            clue = s"hit-cap registration should surface 'temporarily unavailable'; got: $errorMessage")
          // The message must NOT name the cap so an attacker cannot
          // fingerprint the limit by probing.
          assert(!errorMessage.contains("2") && !errorMessage.contains("max"),
            clue = s"cap-hit error must not leak the configured limit; got: $errorMessage")
        }
      }
    }
  }

  test("registration rejects emails containing whitespace or control characters") {
    // RFC 5321 §4.1.2: unquoted local-part excludes whitespace. Beyond
    // compliance, a space in a stored email value also breaks the audit log
    // key=value parsing because the line would split at the wrong column.
    // The structural @-and-dot check passes for all these inputs, so the
    // rejection comes from the new whitespace/control-char guard.
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"
          val invalidEmails = Vector(
            "alice bob@example.com" -> "embedded space",
            "alice\\tbob@example.com" -> "embedded tab (JSON-escaped)",
            "alice\\u0001bob@example.com" -> "embedded SOH control char",
            "alice\\u007Fbob@example.com" -> "embedded DEL"
          )
          for (raw, description) <- invalidEmails do
            val payload =
              s"""{"email":"$raw","password":"correct-horse-battery","displayName":"Test"}"""
            val response = postJson(s"$baseUri/api/auth/register", payload)
            assertEquals(response.statusCode(), 400, clue = description)
            val errorMessage = jsonBody(response)("error").str
            assert(
              errorMessage.contains("whitespace") || errorMessage.contains("control"),
              s"$description: expected whitespace/control rejection, got: $errorMessage"
            )
        }
      }
    }
  }

  test("profile fields reject embedded C0 / DEL control characters") {
    // displayName, heroName, preferredSite, timeZone all flow through
    // sanitizeOptionalField. Email already rejects whitespace + controls
    // (handled by a separate test); profile fields permit internal spaces
    // for "John Smith" / "PokerStars NJ" / etc., but a NUL / newline / ESC /
    // DEL in a stored profile field has no legitimate use and would either
    // confuse downstream JSON consumers, corrupt audit log fields when the
    // value eventually surfaces, or trip line-oriented log dashboards.
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"
          val cases = Vector(
            // (json-escaped field value, label expected in the error)
            ("Alice\\nSmith", "displayName"),
            ("Alice\\u0000Smith", "displayName"),
            ("Alice\\u001bSmith", "displayName"),
            ("Alice\\u007fSmith", "displayName")
          )
          for (raw, label) <- cases do
            val payload =
              s"""{"email":"profile@example.com","password":"correct-horse-battery","displayName":"$raw"}"""
            val response = postJson(s"$baseUri/api/auth/register", payload)
            assertEquals(response.statusCode(), 400,
              clue = s"raw=$raw must be rejected with 400")
            val errorMessage = jsonBody(response)("error").str
            assert(errorMessage.contains("control"),
              s"raw=$raw expected `must not contain control characters`, got: $errorMessage")
        }
      }
    }
  }

  test("registration preserves password whitespace and login enforces the exact bytes") {
    // validatePassword no longer trims before checking length, so what the
    // user submits is what gets hashed. Pin both directions:
    //   1. A password with significant whitespace registers cleanly (the
    //      whole 13-char string passes >=10).
    //   2. Logging in with the same exact whitespace succeeds.
    //   3. Logging in with the same characters MINUS the surrounding
    //      whitespace fails -- because pre-trim validation used to silently
    //      strip whitespace from the validate step but not the hash step,
    //      regressing to that behaviour would falsely accept the trimmed
    //      form on login.
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"
          val padded = "  hunter22ab  "
          val trimmed = "hunter22ab"

          val register = postJson(s"$baseUri/api/auth/register",
            s"""{"email":"ws@example.com","password":"$padded","displayName":"WS"}""")
          assertEquals(register.statusCode(), 201)

          val loginExact = postJson(s"$baseUri/api/auth/login",
            s"""{"email":"ws@example.com","password":"$padded"}""")
          assertEquals(loginExact.statusCode(), 200,
            clue = "login with the exact registered password (whitespace included) must succeed")

          val loginTrimmed = postJson(s"$baseUri/api/auth/login",
            s"""{"email":"ws@example.com","password":"$trimmed"}""")
          assertEquals(loginTrimmed.statusCode(), 401,
            clue = "login with the trimmed password must fail -- raw bytes were hashed at register")
        }
      }
    }
  }

  // Footgun without the zero-width filter in normalizeEmail (see
  // PlatformUserAuth.scala): some paste sources (text editors saving as
  // UTF-8 with BOM, clipboard pipelines that inject zero-width chars)
  // prepend invisible bytes to copied text. Without normalization those
  // bytes survive `String.trim` (which only removes <= 0x20) AND
  // `validateEmail`'s `Character.isWhitespace` check (which doesn't
  // classify BOM or zero-width chars as whitespace), so the canonical
  // stored email ends up with the invisible byte embedded. The user can
  // never sign in afterwards because their subsequent typed input lacks
  // the invisible byte and the stored-vs-submitted comparison misses.
  // Pin the round trip: register with a BOM-prefixed email succeeds,
  // then login WITHOUT the BOM matches the same canonical record.
  test("normalizeEmail strips BOM and zero-width chars so register-then-login is round-trip stable") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"
          // U+FEFF (BOM) at the start of the email -- the canonical
          // failure case for "user pasted from a BOM-emitting source."
          val bomPrefixedEmail = "﻿alice@example.com"
          val bareEmail = "alice@example.com"

          val register = postJson(s"$baseUri/api/auth/register",
            s"""{"email":"$bomPrefixedEmail","password":"correcthorse","displayName":"Alice"}""")
          assertEquals(register.statusCode(), 201,
            clue = "register with BOM-prefixed email must succeed (the BOM is stripped during normalize)")

          // Login WITHOUT the BOM must succeed: the stored record's
          // canonical email matches the bare form after normalization.
          val loginBare = postJson(s"$baseUri/api/auth/login",
            s"""{"email":"$bareEmail","password":"correcthorse"}""")
          assertEquals(loginBare.statusCode(), 200,
            clue = "login with bare email must match the BOM-stripped canonical form")

          // Login WITH the same BOM-prefixed email also works -- both
          // forms normalize to the same canonical email so either
          // produces a successful lookup.
          val loginBom = postJson(s"$baseUri/api/auth/login",
            s"""{"email":"$bomPrefixedEmail","password":"correcthorse"}""")
          assertEquals(loginBom.statusCode(), 200,
            clue = "login with BOM-prefixed email must also match (both forms normalize identically)")
        }
      }
    }
  }

  test("registration and login reject passwords beyond the max-length cap") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          // 1 KB password is well above the 256-char cap. Registration must reject
          // with the at-most message so PBKDF2 is never called on the oversize input.
          val oversizePassword = "x" * 1024
          val payload =
            s"""{"email":"dos@example.com","password":"$oversizePassword","displayName":"DoS Tester"}"""
          val rejected = postJson(s"$baseUri/api/auth/register", payload)
          assertEquals(rejected.statusCode(), 400, clue = jsonBody(rejected).render(indent = 2))
          assert(jsonBody(rejected)("error").str.contains("at most"),
            clue = s"expected max-length rejection, got: ${jsonBody(rejected)("error").str}")

          // Now register a real user, then attempt login with a 1 KB password against
          // that known email -- the response must be "invalid email or password" (not
          // a "too long" message that leaks email existence) and must NOT hang on
          // PBKDF2 of the oversize input.
          val register = postJson(s"$baseUri/api/auth/register",
            """{"email":"victim@example.com","password":"correct-horse-battery","displayName":"Victim"}""")
          assertEquals(register.statusCode(), 201)

          val loginStart = System.currentTimeMillis()
          val dosLogin = postJson(s"$baseUri/api/auth/login",
            s"""{"email":"victim@example.com","password":"$oversizePassword"}""")
          val loginElapsed = System.currentTimeMillis() - loginStart
          assertEquals(dosLogin.statusCode(), 401)
          assert(jsonBody(dosLogin)("error").str == "invalid email or password",
            clue = "must not leak email existence by returning a 'too long' message")
          assert(loginElapsed < 1000,
            clue = s"login with oversize password must short-circuit before PBKDF2 ($loginElapsed ms)")
        }
      }
    }
  }

  test("JsonHandler 500 path returns a generic error and does not leak the exception message") {
    // The handler-level catch must never echo e.getMessage back to clients;
    // it should log server-side and return a fixed, generic body. This is the
    // last line of defense against an unhandled exception that includes file
    // paths, internal class names, or other implementation detail.
    val sentinelMessage = "SENTINEL_INTERNAL_LEAK_xyzzy_abc123"
    val throwingHandler = new AuthStack.JsonHandler(
      handle = _ => throw new RuntimeException(sentinelMessage)
    )
    val httpServer = com.sun.net.httpserver.HttpServer.create(
      new java.net.InetSocketAddress(InetAddress.getLoopbackAddress, 0),
      0
    )
    httpServer.createContext("/throw", throwingHandler)
    httpServer.start()
    try
      val baseUri = s"http://127.0.0.1:${httpServer.getAddress.getPort}"
      val response = httpClient.send(
        HttpRequest.newBuilder().uri(URI.create(s"$baseUri/throw")).GET().build(),
        HttpResponse.BodyHandlers.ofString()
      )
      assertEquals(response.statusCode(), 500)
      val body = response.body()
      assertEquals(ujson.read(body)("error").str, "internal server error")
      assert(!body.contains(sentinelMessage),
        s"500 response must not echo exception message back to client: $body")
      assert(!body.contains("RuntimeException"),
        s"500 response must not leak exception class name: $body")
    finally httpServer.stop(0)
  }

  test("login does PBKDF2 work even when the email is unknown to prevent timing-based email enumeration") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          // Register a known user so we have a known-email path to compare against.
          val register = postJson(s"$baseUri/api/auth/register",
            """{"email":"known@example.com","password":"correct-horse-battery","displayName":"Known"}""")
          assertEquals(register.statusCode(), 201)

          // Warm the JIT once so the first measured request is not penalized.
          postJson(s"$baseUri/api/auth/login",
            """{"email":"known@example.com","password":"wrong-but-normal-length"}""")

          def avgLoginMs(email: String, runs: Int): Long =
            val samples = (1 to runs).map { _ =>
              val start = System.nanoTime()
              val resp = postJson(s"$baseUri/api/auth/login",
                s"""{"email":"$email","password":"wrong-but-normal-length"}""")
              val elapsedMs = (System.nanoTime() - start) / 1000000L
              assertEquals(resp.statusCode(), 401)
              assertEquals(jsonBody(resp)("error").str, "invalid email or password")
              elapsedMs
            }
            samples.sum / samples.length

          val unknownEmailMs = avgLoginMs("nobody-was-ever-here@example.com", 3)
          val knownEmailMs = avgLoginMs("known@example.com", 3)

          // PBKDF2 at 210k iterations takes ~50-200ms on typical hardware. A no-hash
          // path completes in <5ms. The 25ms floor catches a regression that skips
          // hashing in either branch while staying robust to slow CI runners.
          assert(unknownEmailMs >= 25,
            s"unknown-email login must do PBKDF2 work, got $unknownEmailMs ms (known: $knownEmailMs ms)")
          assert(knownEmailMs >= 25,
            s"known-email login must do PBKDF2 work, got $knownEmailMs ms")
        }
      }
    }
  }

  // Locks in the handleAuthLogin 409 "already signed in" gate that the
  // recent doc chain (1f949a0, a9cffa5, 3c87767) documented as
  // security-relevant: this 409 BLOCKS a legitimate user from minting a
  // fresh session via the local-password auth form while a leaked-but-
  // still-valid token keeps the old session alive. Unlike handleOidcCallback
  // which revokes the pre-existing session on the same request, this path
  // is "explicit-reject" -- the test pins the reject behavior so a future
  // refactor that "fixes" the 409 by overwriting the session would be
  // caught by the assertion failure. See AuthStack.scala's handleAuthLogin
  // 409 comment block (added in 85f8008) for the full rationale.
  test("login while already authenticated returns 409 'already signed in' rather than minting a fresh session") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          // Register a user; the response Set-Cookies a fresh session.
          val register = postJson(
            s"$baseUri/api/auth/register",
            """{"email":"alice@example.com","password":"correct-horse-battery","displayName":"Alice"}"""
          )
          assertEquals(register.statusCode(), 201)
          val cookieHeader = sessionCookie(register)

          // POST /api/auth/login with the session cookie attached. The body
          // is a valid login (would otherwise succeed) -- the 409 gate fires
          // BEFORE the credential check, so we don't need correct credentials
          // here, but using valid credentials proves the gate isn't a side-
          // effect of a hashing failure.
          val secondLogin = postJson(
            s"$baseUri/api/auth/login",
            """{"email":"alice@example.com","password":"correct-horse-battery"}""",
            Map("Cookie" -> cookieHeader)
          )
          assertEquals(secondLogin.statusCode(), 409,
            clue = "login attempt with a valid session cookie must return 409, not 200 -- the gate prevents minting a fresh session that would orphan a leaked-but-still-valid token")
          assertEquals(jsonBody(secondLogin)("error").str, "already signed in",
            clue = "409 body must carry the exact `already signed in` text the frontend keys on for the state-divergence-refresh handler")
        }
      }
    }
  }

  // Parallel regression test for handleAuthRegister's 409 -- different
  // rationale than handleAuthLogin's 409 (user-error prevention only, no
  // leaked-token security implication; see the comment block on
  // handleAuthRegister added in 6d01931) but the same wire contract: 409
  // status + "already signed in" body. The frontend's submitAuth handler
  // (site.js) treats 409 from EITHER endpoint identically -- it calls
  // refreshAuthState to pull the live /api/auth/me state and flip the UI
  // from sign-in-form to signed-in-view. If a future refactor changed
  // register's 409 to a different code (e.g. 200 with a body field, or
  // 400), the security-relevant login 409 might keep working but the
  // frontend's state-divergence-refresh would silently break for the
  // register flow, leaving users stuck staring at a stale sign-in form
  // when their sibling tab actually signed them in. Pin both endpoints'
  // 409 contracts so a refactor of either is caught.
  test("register while already authenticated returns 409 'already signed in' for frontend state-divergence-refresh parity") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          // First register establishes a session.
          val firstRegister = postJson(
            s"$baseUri/api/auth/register",
            """{"email":"alice@example.com","password":"correct-horse-battery","displayName":"Alice"}"""
          )
          assertEquals(firstRegister.statusCode(), 201)
          val cookieHeader = sessionCookie(firstRegister)

          // Second register with the same session cookie + a DIFFERENT email
          // (so the duplicate-email-already-exists 400 doesn't fire first;
          // we're testing the auth-state 409 gate specifically, which fires
          // BEFORE the register-storage step).
          val secondRegister = postJson(
            s"$baseUri/api/auth/register",
            """{"email":"bob@example.com","password":"different-passphrase","displayName":"Bob"}""",
            Map("Cookie" -> cookieHeader)
          )
          assertEquals(secondRegister.statusCode(), 409,
            clue = "register attempt with a valid session cookie must return 409, not 201 -- the gate prevents the user from accidentally registering a second account while still signed in")
          assertEquals(jsonBody(secondRegister)("error").str, "already signed in",
            clue = "409 body must carry the exact `already signed in` text -- the frontend's submitAuth handler keys on this for the state-divergence-refresh and treats both register-409 and login-409 identically")
        }
      }
    }
  }

  test("PlatformUserAuth.Service.create returns a clear error for a corrupted user store file") {
    withUserStorePath { storePath =>
      // Write garbage that ujson will reject. Without the file-path-aware wrap,
      // Service.create's outer catch would surface only the raw ujson message.
      Files.writeString(storePath, "this is not valid JSON {[", StandardCharsets.UTF_8)
      val result = PlatformUserAuth.Service.create(PlatformUserAuth.Config(storePath = storePath))
      assert(result.isLeft, "expected create to fail on corrupted file")
      val error = result.left.getOrElse(fail("expected Left"))
      assert(error.contains("user store at"),
        clue = s"error should name the file path; got: $error")
      assert(error.contains(storePath.toAbsolutePath.toString),
        clue = s"error should include the absolute path; got: $error")
      assert(error.contains("Back up") || error.contains("restore") || error.contains("remove"),
        clue = s"error should hint at recovery action; got: $error")
    }
  }

  test("log message sanitization escapes line-structural and control characters") {
    import HandHistoryReviewServerRuntime.sanitizeLogMessage

    // Newline, carriage return, and backslash get escaped so user-controlled values
    // flowing into log interpolation can't forge a fake log line.
    assertEquals(sanitizeLogMessage("safe path"), "safe path")
    assertEquals(sanitizeLogMessage("path with\nnewline"), "path with\\nnewline")
    assertEquals(sanitizeLogMessage("CR\rLF\n"), "CR\\rLF\\n")
    assertEquals(sanitizeLogMessage("escape \\ first so \\n stays literal"),
      "escape \\\\ first so \\\\n stays literal",
      clue = "backslashes must be escaped before \\n so a literal \\n in input doesn't decode as newline")

    // NUL (0x00): would truncate log line display in less/grep/vim
    assertEquals(sanitizeLogMessage("before\u0000after"), "before\\0after",
      clue = "NUL must not pass through; would truncate the display in line-oriented tools")

    // Tab (0x09): would split key=value pairs in column-oriented parsers
    assertEquals(sanitizeLogMessage("before\tafter"), "before\\tafter",
      clue = "tab must not pass through; would split fields in column-oriented log parsers")

    // Other C0 control chars: rendered as \xHH lowercase hex so operators see
    // SOMETHING readable rather than an invisible glyph or terminal-bell character.
    assertEquals(sanitizeLogMessage("\u0001"), "\\x01")
    assertEquals(sanitizeLogMessage("\u0007"), "\\x07", clue = "BEL")
    assertEquals(sanitizeLogMessage("\u001b"), "\\x1b", clue = "ESC")
    assertEquals(sanitizeLogMessage("\u007f"), "\\x7f", clue = "DEL")

    // Printable ASCII (including space) and Unicode pass through unchanged.
    assertEquals(sanitizeLogMessage("plain spaces and slashes /-_."), "plain spaces and slashes /-_.")
    assertEquals(sanitizeLogMessage("unicode: éü"), "unicode: éü")

    // Hard length cap as a last-resort defense against log-line inflation.
    // Per-field caps (submitted email, OIDC ?error= / ?state= / ?code=) are
    // the primary discipline -- their truncation markers land in the right
    // forensic spot -- but anything that slipped past those still gets
    // clamped here so a single audit line can never exceed ~8 KB.
    val huge = "x" * 20000
    val sanitized = sanitizeLogMessage(huge)
    assert(sanitized.length < 9 * 1024,
      clue = s"sanitizeLogMessage must clamp huge inputs (got ${sanitized.length} bytes)")
    assert(sanitized.endsWith("...(truncated)"),
      clue = "clamped sanitization should signal the truncation with a visible marker")
    // A short input must NOT have the marker (the cap fires only when needed).
    assert(!sanitizeLogMessage("short").endsWith("...(truncated)"),
      clue = "short inputs must pass through without the truncation marker")
  }

  test("shutdown grace milliseconds round up to whole HttpServer stop seconds") {
    assertEquals(HandHistoryReviewServer.shutdownDelaySeconds(0L), 0)
    assertEquals(HandHistoryReviewServer.shutdownDelaySeconds(1L), 1)
    assertEquals(HandHistoryReviewServer.shutdownDelaySeconds(999L), 1)
    assertEquals(HandHistoryReviewServer.shutdownDelaySeconds(1000L), 1)
    assertEquals(HandHistoryReviewServer.shutdownDelaySeconds(1001L), 2)
  }

  test("start serves health/static content and rejects oversized uploads") {
    withStaticSite { staticDir =>
      withServer(staticDir, maxUploadBytes = 64) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        val health = get(s"$baseUri/api/health")
        assertEquals(health.statusCode(), 200)
        val healthJson = jsonBody(health)
        assertEquals(healthJson("ok").bool, true)
        assertEquals(healthJson("ready").bool, true)
        assertEquals(healthJson("readyReason").str, "accepting-traffic")
        assertEquals(healthJson("draining").bool, false)
        assertEquals(healthJson("acceptingAnalysisJobs").bool, true)
        assertEquals(healthJson("authenticationEnabled").bool, false)
        assertEquals(healthJson("authenticationMode").str, "none")
        assertEquals(healthJson("host").str, server.binding.host)
        assertEquals(healthJson("port").num.toInt, server.binding.port)
        assertEquals(healthJson("modelSource").str, "uniform fallback")
        assertEquals(healthJson("drainSignalPresent").bool, false)
        assertEquals(healthJson("maxUploadBytes").num.toInt, 64)
        assertEquals(healthJson("analysisTimeoutMs").num.toLong, 120000L)
        assertEquals(healthJson("rateLimitSubmitsPerMinute").num.toInt, 6)
        assertEquals(healthJson("rateLimitStatusPerMinute").num.toInt, 240)
        assertEquals(healthJson("rateLimitAuthPerMinute").num.toInt, 10)
        assertEquals(healthJson("rateLimitClientIpSource").str, "remote-address")
        assertEquals(healthJson("maxConcurrentJobs").num.toInt, 2)
        assertEquals(healthJson("maxQueuedJobs").num.toInt, 8)
        assertEquals(healthJson("activeHttpRequests").num.toInt, 0)
        assertEquals(healthJson("queuedJobs").num.toInt, 0)
        assertEquals(healthJson("runningJobs").num.toInt, 0)
        assertEquals(healthJson("timedOutWorkersInFlight").num.toInt, 0)
        assertEquals(healthJson("retainedTerminalJobs").num.toInt, 0)
        assertEquals(headerValue(health, "X-Content-Type-Options"), Some("nosniff"))
        assertEquals(headerValue(health, "X-Frame-Options"), Some("DENY"))
        assertEquals(headerValue(health, "Cross-Origin-Opener-Policy"), Some("same-origin"))
        assertEquals(headerValue(health, "Cross-Origin-Resource-Policy"), Some("same-origin"))
        assertEquals(headerValue(health, "X-Robots-Tag"), Some("noindex, nofollow"))

        val ready = get(s"$baseUri/api/ready")
        assertEquals(ready.statusCode(), 200)
        val readyJson = jsonBody(ready)
        assertEquals(readyJson("ready").bool, true)
        assertEquals(readyJson("reason").str, "accepting-traffic")
        assertEquals(readyJson("draining").bool, false)
        assertEquals(readyJson("acceptingAnalysisJobs").bool, true)
        assertEquals(readyJson("authenticationEnabled").bool, false)
        assertEquals(readyJson("authenticationMode").str, "none")
        assertEquals(readyJson("analysisTimeoutMs").num.toLong, 120000L)
        assertEquals(readyJson("rateLimitSubmitsPerMinute").num.toInt, 6)
        assertEquals(readyJson("rateLimitStatusPerMinute").num.toInt, 240)
        assertEquals(readyJson("rateLimitAuthPerMinute").num.toInt, 10)
        assertEquals(readyJson("rateLimitClientIpSource").str, "remote-address")
        assertEquals(readyJson("timedOutWorkersInFlight").num.toInt, 0)

        val index = get(s"$baseUri/")
        assertEquals(index.statusCode(), 200)
        assert(index.body().contains("Runtime smoke page"))
        assertEquals(headerValue(index, "X-Content-Type-Options"), Some("nosniff"))
        assertEquals(headerValue(index, "X-Frame-Options"), Some("DENY"))
        // Pin each defensive CSP directive so a regression that drops one (e.g.
        // removing object-src 'none' which kills legacy Flash injection, or
        // base-uri 'none' which prevents <base href> hijacking of relative
        // URLs) fails the test rather than passing on the still-present
        // default-src. The `img-src 'self' data:` directive is ALSO pinned
        // (with both source-list entries) because it's the one permissive
        // allow-rule in the policy and the `data:` half is load-bearing for
        // a specific frontend contract: index.html line 22 sets
        // `<link rel="icon" href="data:,">` -- an empty data-URI favicon
        // whose only purpose is to suppress the spurious GET /favicon.ico
        // that browsers default to when no <link rel="icon"> is declared.
        // A future CSP refactor dropping just the `data:` source (keeping
        // `img-src 'self'`) would silently break that suppression: the
        // browser would refuse the data: URI and fall back to fetching
        // /favicon.ico which the static handler 404s, producing an
        // operator-visible audit-log noise line every page load. Pinning
        // the full directive (both `'self'` AND `data:`) catches that
        // regression at test time rather than via support tickets about
        // 404 spam.
        val csp = headerValue(index, "Content-Security-Policy")
          .getOrElse(fail("expected Content-Security-Policy header"))
        for directive <- Vector(
          "default-src 'self'",
          "base-uri 'none'",
          "connect-src 'self'",
          "form-action 'self'",
          "frame-ancestors 'none'",
          "frame-src 'none'",
          "img-src 'self' data:",
          "manifest-src 'none'",
          "media-src 'none'",
          "object-src 'none'",
          "script-src 'self'",
          "style-src 'self'",
          "worker-src 'none'"
        ) do
          assert(csp.contains(directive), s"CSP missing `$directive`: $csp")
        val permissionsPolicy = headerValue(index, "Permissions-Policy").getOrElse(
          fail("expected Permissions-Policy header on index response"))
        assert(permissionsPolicy.contains("camera=()"), s"missing camera=() in: $permissionsPolicy")
        assert(permissionsPolicy.contains("microphone=()"), s"missing microphone=() in: $permissionsPolicy")
        assert(permissionsPolicy.contains("geolocation=()"), s"missing geolocation=() in: $permissionsPolicy")
        assert(permissionsPolicy.contains("interest-cohort=()"), s"missing interest-cohort=() in: $permissionsPolicy")
        assert(permissionsPolicy.contains("browsing-topics=()"),
          s"missing browsing-topics=() (modern Topics-API opt-out, replacement for FLoC) in: $permissionsPolicy")
        assert(permissionsPolicy.contains("usb=()"), s"missing usb=() in: $permissionsPolicy")
        assertEquals(headerValue(index, "Cross-Origin-Opener-Policy"), Some("same-origin"))
        assertEquals(headerValue(index, "Cross-Origin-Resource-Policy"), Some("same-origin"))
        assertEquals(headerValue(index, "X-Robots-Tag"), Some("noindex, nofollow"))

        val oversizedPayload = s"""{"handHistoryText":"${"A" * 256}"}"""
        val oversizedResponse = postJson(s"$baseUri/api/analyze-hand-history", oversizedPayload)
        assertEquals(oversizedResponse.statusCode(), 413)
        assert(oversizedResponse.body().contains("max upload size"))
      }
    }
  }

  test("health and readiness probes reject non-GET methods with 405 + Allow") {
    // Earlier the handlers accepted any method (POST, DELETE, PATCH …) and
    // happily returned the JSON metrics. Probes must be GET-only -- a misrouted
    // POST has no business getting cached or treated as a successful health
    // check.
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        for path <- Vector("/api/health", "/api/ready") do
          val rejected = postJson(s"$baseUri$path", "{}")
          assertEquals(rejected.statusCode(), 405, clue = path)
          assertEquals(headerValue(rejected, "Allow"), Some("GET, HEAD, OPTIONS"), clue = path)

          val options = httpClient.send(
            HttpRequest.newBuilder()
              .uri(URI.create(s"$baseUri$path"))
              .method("OPTIONS", HttpRequest.BodyPublishers.noBody())
              .build(),
            HttpResponse.BodyHandlers.ofString()
          )
          assertEquals(options.statusCode(), 200, clue = path)
          assertEquals(headerValue(options, "Allow"), Some("GET, HEAD, OPTIONS"), clue = path)

          val head = httpClient.send(
            HttpRequest.newBuilder()
              .uri(URI.create(s"$baseUri$path"))
              .method("HEAD", HttpRequest.BodyPublishers.noBody())
              .build(),
            HttpResponse.BodyHandlers.ofString()
          )
          assertEquals(head.statusCode(), 200, clue = s"$path HEAD must succeed for monitoring probes")
          assertEquals(head.body(), "", clue = s"$path HEAD must not include a body")

          // HEAD response carries the same security/cache headers as GET so a
          // probe that asserts on them still works in HEAD mode.
          val getResp = get(s"$baseUri$path")
          assertEquals(getResp.statusCode(), 200, clue = path)
          assertEquals(headerValue(head, "Content-Type"), headerValue(getResp, "Content-Type"),
            clue = s"$path HEAD Content-Type must match GET")
          assertEquals(headerValue(head, "Cache-Control"), headerValue(getResp, "Cache-Control"),
            clue = s"$path HEAD Cache-Control must match GET")
      }
    }
  }

  test("json auth endpoints answer OPTIONS with 200 and Allow header per RFC 7231 sec 4.3.7") {
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        def sendOptions(path: String): HttpResponse[String] =
          httpClient.send(
            HttpRequest.newBuilder()
              .uri(URI.create(s"$baseUri$path"))
              .method("OPTIONS", HttpRequest.BodyPublishers.noBody())
              .build(),
            HttpResponse.BodyHandlers.ofString()
          )

        val meOptions = sendOptions("/api/auth/me")
        assertEquals(meOptions.statusCode(), 200)
        // /api/auth/me accepts HEAD as well as GET so monitoring tools that
        // probe with HEAD see the same status + security headers as a GET
        // would emit, matching the convention used by /api/health and
        // /api/ready.
        assertEquals(headerValue(meOptions, "Allow"), Some("GET, HEAD, OPTIONS"))

        val loginOptions = sendOptions("/api/auth/login")
        assertEquals(loginOptions.statusCode(), 200)
        assertEquals(headerValue(loginOptions, "Allow"), Some("POST, OPTIONS"))

        val analyzeOptions = sendOptions("/api/analyze-hand-history")
        assertEquals(analyzeOptions.statusCode(), 200)
        assertEquals(headerValue(analyzeOptions, "Allow"), Some("POST, OPTIONS"))

        val playingHallJobOptions = sendOptions("/api/playing-hall/jobs/some-id")
        assertEquals(playingHallJobOptions.statusCode(), 200)
        // GET, HEAD, and DELETE are all accepted on job-status routes:
        // GET/HEAD for status reads (HEAD is body-less per writeBytes),
        // DELETE for cooperative cancellation.
        assertEquals(headerValue(playingHallJobOptions, "Allow"), Some("GET, HEAD, DELETE, OPTIONS"))
      }
    }
  }

  test("HEAD on /api/auth/me returns the same status and security headers as GET, with no body") {
    // /api/auth/me is a read-only auth-state probe. HTTP semantics expect
    // GET endpoints to also answer HEAD with the same status + headers and
    // no body (RFC 7231 sec 4.3.2), and monitoring tools commonly probe
    // with HEAD to skip the JSON payload.
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val request = HttpRequest.newBuilder(URI.create(s"$baseUri/api/auth/me"))
          .method("HEAD", HttpRequest.BodyPublishers.noBody())
          .build()
        val response = httpClient.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8))
        assertEquals(response.statusCode(), 200, clue = "HEAD must succeed like GET")
        assertEquals(response.body(), "", clue = "HEAD must not include a body")
        // Security headers still applied on HEAD via applySecurityHeaders.
        assertEquals(headerValue(response, "X-Content-Type-Options"), Some("nosniff"))
      }
    }
  }

  test("json auth endpoints include Allow header on 405 responses per RFC 7231 sec 6.5.5") {
    // /api/auth/me is GET-only; /api/auth/login is POST-only. RFC 7231 sec 6.5.5
    // REQUIRES the server emit Allow on a 405 so the client (and any cache or
    // OPTIONS-discovery tool) knows which methods the resource accepts.
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        val getOnlyWith405 = postJson(s"$baseUri/api/auth/me", "{}")
        assertEquals(getOnlyWith405.statusCode(), 405)
        assertEquals(headerValue(getOnlyWith405, "Allow"), Some("GET, HEAD, OPTIONS"))
        assertEquals(jsonBody(getOnlyWith405)("error").str, "GET, HEAD required")

        val postOnlyWith405 = get(s"$baseUri/api/auth/login")
        assertEquals(postOnlyWith405.statusCode(), 405)
        assertEquals(headerValue(postOnlyWith405, "Allow"), Some("POST, OPTIONS"))
        assertEquals(jsonBody(postOnlyWith405)("error").str, "POST required")
      }
    }
  }

  test("static handler returns 405 with Allow header for unsupported methods") {
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val response = postJson(s"$baseUri/index.html", "{}")
        assertEquals(response.statusCode(), 405)
        assertEquals(response.body(), "GET, HEAD, or OPTIONS required")
        assertEquals(headerValue(response, "Allow"), Some("GET, HEAD, OPTIONS"))
        assertEquals(headerValue(response, "Cache-Control"), Some("no-store"))
        assertEquals(headerValue(response, "ETag"), None)
      }
    }
  }

  test("static handler answers OPTIONS with 204 and Allow header") {
    // Static handler's OPTIONS path now returns 204 No Content (was 200)
    // to match RedirectHandler's OIDC OPTIONS shape; RFC 7231 sec 6.3.5
    // makes 204 the idiomatic status for body-less responses.
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val request = HttpRequest.newBuilder(URI.create(s"$baseUri/"))
          .method("OPTIONS", HttpRequest.BodyPublishers.noBody())
          .build()
        val response = httpClient.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8))
        assertEquals(response.statusCode(), 204)
        assertEquals(response.body(), "")
        assertEquals(headerValue(response, "Allow"), Some("GET, HEAD, OPTIONS"))
      }
    }
  }

  test("static handler returns 404 not 500 when the URL path contains characters that Paths.get rejects") {
    // On Windows, Paths.get throws InvalidPathException for path strings
    // containing NTFS-reserved characters (`<`, `>`, `:`, `*`, `?`, `|`, `"`).
    // An attacker sending `/file%3Cfoo` (decoded to `/file<foo`) would
    // otherwise fall through to the outer NonFatal catch and get a 500 plus
    // a logged exception per request -- both a 500-when-it-should-be-404 UX
    // bug AND a log-inflation lever (stack trace per request). The static
    // handler now catches InvalidPathException specifically and returns a
    // clean 404. On Linux the test still passes because Linux file systems
    // permit `<`/`>` in filenames, so Paths.get succeeds and the file simply
    // does not exist, hitting the regular 404 branch -- same observable
    // result on both platforms.
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        // %3C / %3E decode to `<` / `>`. URI.create accepts the encoded form.
        val response = get(s"$baseUri/file%3Cfoo%3Ebar")
        assertEquals(response.statusCode(), 404,
          clue = s"NTFS-reserved characters in URL path must produce 404, not 500; got body: ${response.body()}")
        // The 404 body must be the small generic "not found", NOT a stack
        // trace or other 500-style payload.
        assert(response.body().length < 256,
          clue = s"404 body should stay small; got ${response.body().length} bytes")
      }
    }
  }

  test("static handler answers HEAD with headers and no body") {
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val request = HttpRequest.newBuilder(URI.create(s"$baseUri/"))
          .method("HEAD", HttpRequest.BodyPublishers.noBody())
          .build()
        val response = httpClient.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8))
        assertEquals(response.statusCode(), 200)
        assertEquals(response.body(), "")
        assert(headerValue(response, "ETag").exists(_.startsWith("W/\"")), "ETag must be present on HEAD")
        assertEquals(headerValue(response, "Content-Type"), Some("text/html; charset=utf-8"))
        assertEquals(headerValue(response, "Cache-Control"), Some("public, max-age=0, must-revalidate"))
      }
    }
  }

  test("HEAD requests on static error paths return headers without a body") {
    // writeBytes now honors HEAD across the board, so the static handler's
    // 404/403/etc paths advertise Content-Length matching what GET would
    // send but emit no body. RFC 7231 sec 4.3.2.
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        def head(path: String): HttpResponse[String] =
          httpClient.send(
            HttpRequest.newBuilder(URI.create(s"$baseUri$path"))
              .method("HEAD", HttpRequest.BodyPublishers.noBody())
              .build(),
            HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8)
          )

        val notFound = head("/does-not-exist")
        assertEquals(notFound.statusCode(), 404)
        assertEquals(notFound.body(), "", "HEAD 404 must have an empty body")

        val forbidden = head("/../etc/passwd")
        assertEquals(forbidden.statusCode(), 403)
        assertEquals(forbidden.body(), "", "HEAD 403 must have an empty body")

        val dotfile = head("/.git/HEAD")
        assertEquals(dotfile.statusCode(), 404)
        assertEquals(dotfile.body(), "", "HEAD on dot-prefixed path must have an empty body")
      }
    }
  }

  test("server does not advertise its software identity in a Server response header") {
    // Many HTTP servers default to emitting `Server: JDK/17.0.5` or similar,
    // which gives an attacker a free version fingerprint for targeting
    // known JVM/HttpServer CVEs. Our server should keep that surface dark.
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val response = get(s"$baseUri/api/health")
        assertEquals(response.statusCode(), 200)
        val serverHeader = headerValue(response, "Server")
        assert(
          serverHeader.isEmpty || !serverHeader.exists(value => value.contains("JDK") || value.contains("Java") || value.contains("/")),
          s"Server header should not advertise JDK/Java version; got: $serverHeader"
        )
      }
    }
  }

  test("static handler returns the same ETag and Content-Encoding on HEAD as on GET for the same Accept-Encoding") {
    // RFC 7231: HEAD describes the GET response. A client that does HEAD to
    // validate a cached entry and then GET to refetch must see the same
    // variant headers, or the cache will discard the entry on the GET
    // response with different ETag/Content-Encoding.
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        def fetch(method: String, acceptEncoding: Option[String]): HttpResponse[String] =
          val builder = HttpRequest.newBuilder(URI.create(s"$baseUri/")).method(method, HttpRequest.BodyPublishers.noBody())
          acceptEncoding.foreach(builder.header("Accept-Encoding", _))
          httpClient.send(builder.build(), HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8))

        // Accept-Encoding: gzip -- the gzipped variant
        val headGz = fetch("HEAD", Some("gzip"))
        val getGz = fetch("GET", Some("gzip"))
        assertEquals(headerValue(headGz, "ETag"), headerValue(getGz, "ETag"),
          "HEAD and GET must report the same ETag for the gzipped variant")
        assertEquals(headerValue(headGz, "Content-Encoding"), headerValue(getGz, "Content-Encoding"),
          "HEAD and GET must report the same Content-Encoding for the gzipped variant")
        assert(headerValue(headGz, "ETag").exists(_.endsWith("-gz\"")),
          s"gzipped ETag must carry the -gz suffix, got: ${headerValue(headGz, "ETag")}")
        assertEquals(headGz.body(), "", "HEAD response must have no body")

        // No Accept-Encoding -- the plain variant
        val headPlain = fetch("HEAD", None)
        val getPlain = fetch("GET", None)
        assertEquals(headerValue(headPlain, "ETag"), headerValue(getPlain, "ETag"),
          "HEAD and GET must report the same ETag for the plain variant")
        assert(headerValue(headPlain, "Content-Encoding").isEmpty,
          s"plain variant must not carry Content-Encoding, got: ${headerValue(headPlain, "Content-Encoding")}")
        assert(!headerValue(headPlain, "ETag").exists(_.endsWith("-gz\"")),
          s"plain ETag must not carry the -gz suffix, got: ${headerValue(headPlain, "ETag")}")
      }
    }
  }

  // RFC 7231 sec 4.3.2: HEAD response MUST have the same headers a GET would
  // emit. The static handler at StaticAssetsHandler.scala already advertises
  // Content-Encoding: gzip on HEAD when the equivalent GET would compress
  // (test above). writeBytes (WebResponses.scala) is the parallel code path
  // for the JSON-API endpoints + the writePlain error paths -- HEAD on
  // /api/health should mirror what GET would have sent so a cache that
  // validates via HEAD then fetches via GET sees consistent Content-Encoding,
  // not "plain on HEAD, gzip on GET" which would invalidate the entry.
  test("JsonHandler HEAD advertises Content-Encoding: gzip when the GET variant would compress") {
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        def fetch(method: String, acceptEncoding: Option[String]): HttpResponse[String] =
          val builder = HttpRequest.newBuilder(URI.create(s"$baseUri/api/health")).method(method, HttpRequest.BodyPublishers.noBody())
          acceptEncoding.foreach(builder.header("Accept-Encoding", _))
          httpClient.send(builder.build(), HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8))

        // Accept-Encoding: gzip -- the gzipped variant. /api/health emits a
        // JSON body well over the 256-byte MinGzipSize floor (~30 fields), so
        // the GET response is always compressed for a gzip-accepting client.
        val headGz = fetch("HEAD", Some("gzip"))
        val getGz = fetch("GET", Some("gzip"))
        assertEquals(headGz.statusCode(), 200)
        assertEquals(headGz.body(), "", "HEAD response must have no body")
        assertEquals(
          headerValue(headGz, "Content-Encoding"),
          headerValue(getGz, "Content-Encoding"),
          "HEAD and GET on /api/health must agree on Content-Encoding for the gzipped variant"
        )
        assertEquals(headerValue(headGz, "Content-Encoding"), Some("gzip"),
          "HEAD on /api/health with Accept-Encoding: gzip must declare gzip")
        assertEquals(headerValue(headGz, "Vary"), Some("Accept-Encoding"),
          "Vary: Accept-Encoding must be present on the JSON HEAD response so caches key by encoding")

        // No Accept-Encoding -- the plain variant. Without the header the
        // server has no signal that gzip is acceptable, so neither HEAD nor
        // GET should emit Content-Encoding.
        val headPlain = fetch("HEAD", None)
        assertEquals(headPlain.statusCode(), 200)
        assert(
          headerValue(headPlain, "Content-Encoding").isEmpty,
          s"HEAD on /api/health without Accept-Encoding must not declare Content-Encoding, got: ${headerValue(headPlain, "Content-Encoding")}"
        )
      }
    }
  }

  test("static handler honors If-None-Match on HEAD by returning 304") {
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val firstHead = HttpRequest.newBuilder(URI.create(s"$baseUri/"))
          .method("HEAD", HttpRequest.BodyPublishers.noBody())
          .build()
        val first = httpClient.send(firstHead, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8))
        val etag = headerValue(first, "ETag").getOrElse(fail("expected ETag on first HEAD"))

        val revalidateHead = HttpRequest.newBuilder(URI.create(s"$baseUri/"))
          .header("If-None-Match", etag)
          .method("HEAD", HttpRequest.BodyPublishers.noBody())
          .build()
        val response = httpClient.send(revalidateHead, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8))
        assertEquals(response.statusCode(), 304)
        assertEquals(response.body(), "")
        assertEquals(headerValue(response, "ETag"), Some(etag))
      }
    }
  }

  test("static handler blocks path traversal attempts with 403") {
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val response = get(s"$baseUri/../etc/passwd")
        assertEquals(response.statusCode(), 403)
        assertEquals(response.body(), "forbidden")
        assertEquals(headerValue(response, "Cache-Control"), Some("no-store"))
        assertEquals(headerValue(response, "ETag"), None)
      }
    }
  }

  test("static handler returns 404 for non-existent files") {
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val response = get(s"$baseUri/missing-file.html")
        assertEquals(response.statusCode(), 404)
        assertEquals(response.body(), "not found")
        assertEquals(headerValue(response, "Cache-Control"), Some("no-store"))
        assertEquals(headerValue(response, "ETag"), None)
      }
    }
  }

  // Pins the documented STATIC_DIR-misconfig triage symptom (runbook
  // Troubleshooting entry: "Web upload UI returns `404 not found` but
  // `/api/health` is `200`"). The runbook tells operators that if STATIC_DIR
  // points to a non-existent or wrong directory (typo, partially-extracted
  // bundle, cwd mismatch), the server still starts, /api/health still
  // returns 200, but every GET / and GET /<asset> returns generic 404 +
  // text/plain "not found". A future refactor that fail-fast'd at startup
  // (validating STATIC_DIR existence) would break the documented
  // behavior the troubleshooting entry assumes -- without this test,
  // such a refactor would silently invalidate the triage guidance. Test
  // configures staticDir to a temp path that does NOT exist on disk;
  // checks the server starts, /api/health returns 200, and GET / returns
  // the documented 404 + text/plain shape an operator would curl-check.
  test("STATIC_DIR pointing to a non-existent directory still starts cleanly, /api/health stays 200, but GET / returns 404 -- pins the runbook STATIC_DIR-misconfig triage symptom") {
    val nonExistentStaticDir = Files.createTempDirectory("missing-static-").resolve("does-not-exist")
    // Sanity check: parent exists but the directory itself doesn't.
    assert(!Files.exists(nonExistentStaticDir),
      s"setup precondition: $nonExistentStaticDir must not exist for this test to exercise the misconfig path")
    try
      withServer(nonExistentStaticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        // /api/health stays 200 -- the runbook entry's premise.
        val health = get(s"$baseUri/api/health")
        assertEquals(health.statusCode(), 200,
          clue = "STATIC_DIR misconfig must NOT affect /api/health -- this is the diagnostic the triage entry relies on (200 health + 404 / together signal STATIC_DIR is wrong)")

        // GET / returns the documented 404 + text/plain "not found" shape.
        val root = get(s"$baseUri/")
        assertEquals(root.statusCode(), 404,
          clue = "STATIC_DIR pointing to a non-existent path must produce 404 on GET / -- this is the curl symptom the runbook tells operators to check")
        assertEquals(root.body(), "not found",
          clue = "404 body must be the generic 'not found' string the runbook documents as the STATIC_DIR-wrong signature")
        assertEquals(headerValue(root, "Content-Type"), Some("text/plain; charset=utf-8"),
          clue = "404 must carry text/plain Content-Type so the runbook's 'curl -i and check Content-Type' triage distinguishes it from the 200+text/html healthy case")
      }
    finally
      // Cleanup: the parent temp directory wasn't auto-cleaned by withServer
      // because staticDir.resolve("does-not-exist") was never written to.
      Files.deleteIfExists(nonExistentStaticDir.getParent)
  }

  // Pin the boot-error banner element + ARIA semantics + user-visible
  // message in the production bundle's index.html. The runbook's
  // "Users report 'the page doesn't respond'" triage entry (section 6,
  // just below the OIDC entry) keys on three operator-visible signals:
  // (1) `curl -s / | grep page-init-error` returns the element so the
  // bundle hasn't been silently regressed, (2) the banner carries
  // role=alert + aria-live=assertive so SR users get an announcement
  // on reveal, (3) the banner text matches the documented "This page
  // didn't finish loading." prefix the operator asks the user to
  // confirm. A future refactor that dropped the banner from
  // index.html (or changed its id, class, ARIA shape, or user-message
  // wording) would silently invalidate every one of those triage cues;
  // this test makes that regression loud. Same pin pattern as 8d5f49e
  // (STATIC_DIR-misconfig triage symptom): runbook-documented
  // diagnostics get pinned in tests so refactors can't silently break
  // the signals operators have been told to look for. Unlike the rest
  // of the static-handler tests in this file (which use the synthetic
  // withStaticSite temp directory), this test points withServer at
  // the REAL `docs/site-preview-hybrid` bundle path -- a synthetic
  // index would defeat the purpose of testing the SHIPPED bundle's
  // markup.
  test("bundled index.html ships the page-init-error banner with documented id + class + ARIA semantics + user message so the runbook's boot-error triage signal isn't silently regressed") {
    val bundleDir = Paths.get("docs", "site-preview-hybrid").toAbsolutePath.normalize()
    assert(Files.isDirectory(bundleDir),
      s"bundle directory $bundleDir must exist -- test must run from the project root (SBT default cwd). Other tests in this file use Paths.get(\"src\", \"main\", \"native\", ...) the same way (see HeadsUpRangeGpuRuntimeTest for the existing precedent).")
    withServer(bundleDir) { server =>
      val baseUri = s"http://${server.binding.host}:${server.binding.port}"
      val response = get(s"$baseUri/")
      assertEquals(response.statusCode(), 200,
        clue = "served bundle root must return 200 -- the runbook entry's pre-condition before the banner-element grep cue")
      val body = response.body()
      // The element id is the runbook's `curl | grep` anchor.
      assert(body.contains("""id="page-init-error""""),
        "missing #page-init-error banner element -- runbook's `curl -s / | grep page-init-error` triage cue depends on this exact id appearing in the served bytes")
      // Hidden by default via the .hidden class so the happy path
      // doesn't show the banner; site.js's showBootError removes
      // .hidden when boot() rejects.
      assert(body.contains("""class="noscript-notice hidden""""),
        "banner element missing 'noscript-notice hidden' class binding -- without 'hidden' the banner would be visible on every page load, falsely signaling boot failure on the happy path; without 'noscript-notice' it loses the warning-palette visual that mirrors the noscript-disabled element above it")
      // SR users get the announcement on reveal (no re-Tab needed).
      assert(body.contains("""role="alert""""),
        "banner element missing role=alert -- screen readers wouldn't announce the banner when site.js reveals it, defeating the a11y intent of role-based live-region semantics")
      assert(body.contains("""aria-live="assertive""""),
        "banner element missing aria-live=assertive -- SR announcement would be polite/deferred instead of immediate, leaving the user interacting with a half-broken page before being told it's broken")
      // The runbook tells operators to ask the user to confirm this
      // exact string is visible in the browser.
      assert(body.contains("This page didn't finish loading."),
        "banner element missing the documented user-visible message -- runbook's triage step (\"user reports seeing 'This page didn't finish loading'\") depends on this exact wording")
    }
  }

  // Pin the empty-data-URI favicon link in the production bundle's
  // index.html so a future maintainer removing the seemingly-unused
  // `<link rel="icon" href="data:,">` line (perhaps thinking it's a
  // no-op stub) fails this test rather than silently turning on per-
  // page-load audit-log noise from the spurious /favicon.ico GET that
  // browsers default to when no rel=icon is declared. This is the
  // HTML half of the favicon-suppression contract; the CSP half
  // (`img-src 'self' data:` in WebResponses.scala's
  // ContentSecurityPolicy constant) is pinned by the CSP-directive-
  // pin block in `start serves health/static content and rejects
  // oversized uploads`. Both halves are co-required -- dropping
  // either silently breaks the suppression. The in-source HTML
  // comment above the link in index.html names the contract; this
  // test catches the silent regression if a future maintainer
  // ignores the comment and removes the link anyway. Same
  // regression-pin pattern as the page-init-error banner test above
  // and the CSP `img-src 'self' data:` pin in the security-headers
  // block: operator-visible / audit-log-hygiene contracts get
  // pinned so refactors can't silently regress them.
  test("bundled index.html ships the empty-data-URI favicon link so /favicon.ico 404s don't pollute the audit log on every page load") {
    val bundleDir = Paths.get("docs", "site-preview-hybrid").toAbsolutePath.normalize()
    assert(Files.isDirectory(bundleDir),
      s"bundle directory $bundleDir must exist -- test must run from project root (SBT default cwd)")
    withServer(bundleDir) { server =>
      val baseUri = s"http://${server.binding.host}:${server.binding.port}"
      val response = get(s"$baseUri/")
      assertEquals(response.statusCode(), 200,
        clue = "served bundle root must return 200 before the favicon-link assertion can run")
      val body = response.body()
      // The exact-string assertion catches three regression vectors:
      //   - Removing the link entirely (most likely if a maintainer
      //     thinks it's an unused stub) -- substring vanishes.
      //   - Changing the href to `data:image/png;base64,...` for a
      //     real inline icon -- the `data:,` literal vanishes (and
      //     the CSP `data:` source-list entry is no longer needed,
      //     should be dropped too per the in-source comment's
      //     refactor guidance).
      //   - Reformatting the link with extra attributes (sizes=,
      //     type=, etc.) that split the exact substring -- the test
      //     fails until either the new shape is captured in the
      //     assertion OR the change is reverted.
      // Each of the three would silently break the suppression
      // contract; the assertion makes any of them loud.
      assert(body.contains("""<link rel="icon" href="data:,">"""),
        "missing the `<link rel=\"icon\" href=\"data:,\">` favicon-suppression link -- without it, browsers default to GET /favicon.ico, the static handler 404s, and every page load emits an audit-log noise line; the inline HTML comment above the link in index.html names the contract and the CSP-directive-pin block above (in 'start serves health/static content and rejects oversized uploads') enforces the CSP half (`img-src 'self' data:` in WebResponses.scala); both halves must stay in sync")
    }
  }

  // Pin the <noscript> fallback in the production bundle's index.html
  // so a future maintainer removing it (perhaps thinking "everyone has
  // JavaScript these days") fails this test rather than silently
  // degrading the page for users with JS disabled into a "page chrome
  // loaded but no controls work, no explanation of why" state. The
  // failure mode is operator-INVISIBLE -- a user with JS disabled
  // sees the static HTML, none of the submit handlers wire up, and
  // they have no way to know whether the deployment is broken or
  // their browser is the cause. The <noscript> element gives them
  // an immediate answer. Without a test, removal goes unnoticed
  // until the next time a JS-disabled user files a support ticket.
  // This is the THIRD operator-visible defensive HTML invariant
  // pinned via this regression-test pattern:
  //   - <div id="page-init-error"> banner for the "JS ran but
  //     boot() threw" case (page-init-error test above)
  //   - <link rel="icon" href="data:,"> for favicon suppression /
  //     audit-log hygiene (favicon test above)
  //   - <noscript> for the "JS is disabled entirely" case (this test)
  // All three are user-degraded-mode signals where silent removal
  // produces an operator-invisible regression that only surfaces
  // via support tickets from affected users.
  test("bundled index.html ships the <noscript> fallback message so users who disabled JavaScript see a friendly explanation instead of a silently broken page") {
    val bundleDir = Paths.get("docs", "site-preview-hybrid").toAbsolutePath.normalize()
    assert(Files.isDirectory(bundleDir),
      s"bundle directory $bundleDir must exist -- test must run from project root (SBT default cwd)")
    withServer(bundleDir) { server =>
      val baseUri = s"http://${server.binding.host}:${server.binding.port}"
      val response = get(s"$baseUri/")
      assertEquals(response.statusCode(), 200,
        clue = "served bundle root must return 200 before the <noscript> assertion can run")
      val body = response.body()
      // Element presence -- the bare <noscript> opens the fallback
      // block. Without it, JS-disabled users get the static page
      // with no functioning UI controls and no explanation.
      assert(body.contains("<noscript>"),
        "missing the <noscript> fallback element -- without it, users who disabled JavaScript see the page chrome but no functioning UI controls and no explanation of why; the operator only finds out about the regression via support tickets from those users, never from logs (the JS-disabled user never reaches any server endpoint that would log their visit)")
      // The inner element's role=alert + .noscript-notice class
      // binding gives SR users an immediate announcement AND
      // mirrors the boot-error banner's visual treatment so the
      // two "JS didn't fully run" failure modes (JS disabled / JS
      // threw) read identically as page-level warnings.
      assert(body.contains("""<div class="noscript-notice" role="alert">"""),
        "noscript inner div missing the documented class + role binding -- without role=alert the SR doesn't announce the warning when the user lands on the page, without .noscript-notice the visual treatment drops and the message looks like ordinary body copy rather than a page-level warning that the deployment is partially unusable")
      // The bold user-actionable lead. The exact phrasing matters
      // because it's the operator's ultimate fallback "go look at
      // the served HTML on the user's browser, do you see this
      // sentence?" diagnostic.
      assert(body.contains("<strong>JavaScript is disabled.</strong>"),
        "noscript message missing the documented bold lead 'JavaScript is disabled.' -- this is the user-actionable headline the entire fallback message is structured around and serves as the operator's last-resort grep target ('user says they see the page but no buttons work; ask them if they see the <strong>JavaScript is disabled.</strong> sentence above the header')")
    }
  }

  test("static handler dotfile blocking handles percent-encoded backslash on Windows-style paths") {
    // On Windows, both `/` and `\` are filesystem path separators. The dotfile-
    // block check only saw `/` initially, so an attacker sending a URL with a
    // percent-encoded backslash (`%5C`) immediately before `.git` could slip
    // past: getPath decoded `%5C` to a literal `\`, split('/') treated
    // `\.git` as a single segment that did NOT start with `.`, and the
    // dotfile check would pass. Path-traversal still kicks in if the file
    // is outside the static dir, but a file inside the static dir at that
    // backslashed path would be served. Split on both separators now.
    withStaticSite { staticDir =>
      Files.createDirectories(staticDir.resolve(".git"))
      Files.writeString(staticDir.resolve(".git/HEAD"), "ref: refs/heads/main\n", StandardCharsets.UTF_8)
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        // %5C decodes to '\' -- with the fix, this is treated as a segment
        // separator and `.git` is recognised as dot-prefixed.
        val response = get(s"$baseUri/%5C.git/HEAD")
        assert(
          response.statusCode() == 404 || response.statusCode() == 403,
          s"backslash-prefixed dotfile must be refused (404 dotfile-block or 403 path-traversal), got ${response.statusCode()}"
        )
        assert(!response.body().contains("ref: refs"),
          s"backslash-prefixed dotfile must not leak file content: ${response.body()}")
      }
    }
  }

  test("static handler does not follow symlinks that escape the static directory") {
    // A symlink under the static root that points outside it (or to anywhere
    // else on the filesystem) used to be followed by Files.isRegularFile and
    // Files.size, so a deploy that accidentally included such a symlink
    // could serve arbitrary host files via the static handler. NOFOLLOW_LINKS
    // now treats symlinks as non-regular files -- the request 404s rather
    // than leaking the host file.
    //
    // Creating a symlink requires either Linux/Mac or Windows Developer Mode.
    // The test gracefully skips when the JVM cannot create one (file system
    // doesn't support symlinks, or platform refuses the operation).
    withStaticSite { staticDir =>
      val outsideTarget = Files.createTempFile("escape-target-", ".txt")
      try
        Files.writeString(outsideTarget, "SHOULD-NEVER-LEAK", StandardCharsets.UTF_8)
        val symlinkPath = staticDir.resolve("escape.txt")
        val canSymlink =
          try
            Files.createSymbolicLink(symlinkPath, outsideTarget.toAbsolutePath)
            true
          catch
            case _: java.nio.file.FileSystemException => false
            case _: UnsupportedOperationException => false
            case _: SecurityException => false
        if !canSymlink then
          // Filesystem or platform does not allow symlinks here; the structural
          // NOFOLLOW_LINKS guard is still in place, just not exercisable.
          ()
        else
          withServer(staticDir) { server =>
            val baseUri = s"http://${server.binding.host}:${server.binding.port}"
            val response = get(s"$baseUri/escape.txt")
            assertEquals(response.statusCode(), 404)
            assert(!response.body().contains("SHOULD-NEVER-LEAK"),
              s"symlink target must not leak via static handler: ${response.body()}")
          }
      finally Files.deleteIfExists(outsideTarget)
    }
  }

  test("static handler returns 404 for any path with a dot-prefixed segment, even when the file exists") {
    // Defense in depth: a misconfigured deployment that ships a `.git/`,
    // `.env`, `.htaccess`, or other dot-prefixed file under the static root
    // would otherwise expose its contents to anyone who guesses the path.
    // The handler must refuse the request without confirming the file exists.
    withStaticSite { staticDir =>
      Files.createDirectories(staticDir.resolve(".git"))
      Files.writeString(staticDir.resolve(".git/HEAD"), "ref: refs/heads/main\n", StandardCharsets.UTF_8)
      Files.writeString(staticDir.resolve(".env"), "SECRET=should-not-leak\n", StandardCharsets.UTF_8)
      Files.writeString(staticDir.resolve(".htaccess"), "Deny from all\n", StandardCharsets.UTF_8)
      Files.createDirectories(staticDir.resolve("subdir"))
      Files.writeString(staticDir.resolve("subdir/.env"), "ALSO_SECRET=nope\n", StandardCharsets.UTF_8)
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        for path <- Vector("/.git/HEAD", "/.env", "/.htaccess", "/subdir/.env") do
          val response = get(s"$baseUri$path")
          assertEquals(response.statusCode(), 404, clue = s"$path must be refused")
          assert(!response.body().contains("SECRET"), s"$path body leaked file content: ${response.body()}")
          assert(!response.body().contains("ref: refs"), s"$path body leaked .git contents: ${response.body()}")
      }
    }
  }

  test("static handler resolves / to index.html with text/html content-type") {
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val response = get(s"$baseUri/")
        assertEquals(response.statusCode(), 200)
        assert(response.body().contains("Runtime smoke page"))
        assertEquals(headerValue(response, "Content-Type"), Some("text/html; charset=utf-8"))
      }
    }
  }

  test("static handler emits correct Content-Type for representative extensions") {
    withStaticSite { staticDir =>
      val cases = Seq(
        "test.css" -> "text/css; charset=utf-8",
        "test.js" -> "application/javascript; charset=utf-8",
        "test.svg" -> "image/svg+xml",
        "test.png" -> "image/png",
        "test.ico" -> "image/x-icon",
        "test.wasm" -> "application/wasm"
      )
      cases.foreach { case (filename, _) =>
        Files.writeString(staticDir.resolve(filename), "x", StandardCharsets.UTF_8)
      }
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        cases.foreach { case (filename, expectedContentType) =>
          val response = get(s"$baseUri/$filename")
          assertEquals(response.statusCode(), 200, clue = s"file=$filename")
          assertEquals(headerValue(response, "Content-Type"), Some(expectedContentType), clue = s"file=$filename")
        }
      }
    }
  }

  test("static handler emits ETag and revalidation Cache-Control for non-vendor assets") {
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val response = get(s"$baseUri/")
        assertEquals(response.statusCode(), 200)
        assertEquals(headerValue(response, "Cache-Control"), Some("public, max-age=0, must-revalidate"))
        assert(headerValue(response, "ETag").exists(_.startsWith("W/\"")), "ETag must be a weak validator")
      }
    }
  }

  test("static handler returns 304 when If-None-Match matches the current ETag") {
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val first = get(s"$baseUri/")
        assertEquals(first.statusCode(), 200)
        val etag = headerValue(first, "ETag").getOrElse(fail("expected ETag on first response"))

        val revalidated = get(s"$baseUri/", Map("If-None-Match" -> etag))
        assertEquals(revalidated.statusCode(), 304)
        assertEquals(revalidated.body(), "")
        assertEquals(headerValue(revalidated, "ETag"), Some(etag))
        assertEquals(headerValue(revalidated, "Cache-Control"), Some("public, max-age=0, must-revalidate"))
      }
    }
  }

  test("static handler honors If-None-Match: * by returning 304 for any existing resource") {
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val response = get(s"$baseUri/", Map("If-None-Match" -> "*"))
        assertEquals(response.statusCode(), 304)
        assertEquals(response.body(), "")
        assert(headerValue(response, "ETag").exists(_.startsWith("W/\"")), "ETag must still be emitted on 304")
      }
    }
  }

  test("static handler matches one of several comma-separated If-None-Match values") {
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val first = get(s"$baseUri/")
        val etag = headerValue(first, "ETag").getOrElse(fail("expected ETag on first response"))

        val multi = s"""W/"00-0", $etag, "stale-tag""""
        val response = get(s"$baseUri/", Map("If-None-Match" -> multi))
        assertEquals(response.statusCode(), 304)
        assertEquals(response.body(), "")
      }
    }
  }

  // RFC 7232 sec 2.3.2 mandates WEAK comparison for If-None-Match: two
  // entity-tags are equivalent if their opaque-tags match character-by-
  // character "regardless of either or both being tagged as 'weak'". The
  // server emits weak ETags like W/"123-456", and a well-behaved client
  // echoes that exact value -- but a middleware (CDN, reverse proxy) can
  // strip the `W/` prefix in transit, leaving the bare opaque-tag form
  // "123-456". The strong-comparison path (pre-fix) would miss this
  // because direct string equality requires both sides to be identically
  // prefixed, forcing a full-body re-fetch on every poll. Verify that the
  // weak-prefix-stripped form revalidates as the same resource.
  test("static handler treats If-None-Match: bare-opaque-tag as equivalent to the server's W/-prefixed ETag (weak comparison per RFC 7232 sec 2.3.2)") {
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val first = get(s"$baseUri/")
        val etag = headerValue(first, "ETag").getOrElse(fail("expected ETag on first response"))
        assert(etag.startsWith("W/\""), s"server should emit a weak ETag; got $etag")

        // Strip the `W/` prefix and resend; the server must still 304 because
        // weak comparison treats W/"x" and "x" as the same entity-tag.
        val stripped = etag.stripPrefix("W/")
        val response = get(s"$baseUri/", Map("If-None-Match" -> stripped))
        assertEquals(response.statusCode(), 304, clue = s"sent If-None-Match=$stripped against ETag=$etag")
        assertEquals(response.body(), "")
      }
    }
  }

  test("static handler emits Last-Modified header and honors If-Modified-Since") {
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val first = get(s"$baseUri/")
        assertEquals(first.statusCode(), 200)
        val lastModifiedHttp = headerValue(first, "Last-Modified")
          .getOrElse(fail("expected Last-Modified header on first response"))

        val sameTime = get(s"$baseUri/", Map("If-Modified-Since" -> lastModifiedHttp))
        assertEquals(sameTime.statusCode(), 304, clue = s"IMS=$lastModifiedHttp")
        assertEquals(sameTime.body(), "")
        assertEquals(headerValue(sameTime, "Last-Modified"), Some(lastModifiedHttp))

        val pastTime = get(s"$baseUri/", Map("If-Modified-Since" -> "Sun, 06 Nov 1994 08:49:37 GMT"))
        assertEquals(pastTime.statusCode(), 200)
        assert(pastTime.body().nonEmpty)
      }
    }
  }

  test("static handler ignores If-Modified-Since when If-None-Match is also present") {
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val first = get(s"$baseUri/")
        val etag = headerValue(first, "ETag").getOrElse(fail("expected ETag"))
        val lastModifiedHttp = headerValue(first, "Last-Modified").getOrElse(fail("expected Last-Modified"))

        // ETag mismatch + IMS that says "not modified": RFC 7232 sec 3.3 says ETag wins,
        // so the response must be 200 (full body), not 304.
        val response = get(s"$baseUri/", Map(
          "If-None-Match" -> "\"a-different-etag\"",
          "If-Modified-Since" -> lastModifiedHttp
        ))
        assertEquals(response.statusCode(), 200, clue = s"ETag $etag should win over IMS $lastModifiedHttp")
        assert(response.body().nonEmpty)
      }
    }
  }

  test("static handler gzips compressible content when client accepts gzip") {
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val compressed = get(s"$baseUri/", Map("Accept-Encoding" -> "gzip, deflate"))
        assertEquals(compressed.statusCode(), 200)
        assertEquals(headerValue(compressed, "Content-Encoding"), Some("gzip"))
        assertEquals(headerValue(compressed, "Vary"), Some("Accept-Encoding"))
        assert(headerValue(compressed, "ETag").exists(_.endsWith("-gz\"")),
          s"gzipped response ETag should carry -gz suffix, got ${headerValue(compressed, "ETag")}")

        val plain = get(s"$baseUri/")
        assertEquals(plain.statusCode(), 200)
        assertEquals(headerValue(plain, "Content-Encoding"), None)
        assertEquals(headerValue(plain, "Vary"), Some("Accept-Encoding"))
        assert(headerValue(plain, "ETag").exists(t => !t.endsWith("-gz\"")),
          s"plain response ETag should not carry -gz suffix, got ${headerValue(plain, "ETag")}")
      }
    }
  }

  test("static handler gzip output round-trips back to the original bytes") {
    withStaticSite { staticDir =>
      val plainBytes = Files.readAllBytes(staticDir.resolve("index.html"))
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        // Bypass the default String body handler so we receive the raw gzipped bytes
        // exactly as they came off the wire, then decompress manually.
        val request = HttpRequest.newBuilder(URI.create(s"$baseUri/"))
          .header("Accept-Encoding", "gzip")
          .GET().build()
        val response = httpClient.send(request, HttpResponse.BodyHandlers.ofByteArray())
        assertEquals(response.statusCode(), 200)
        assertEquals(
          Option(response.headers().firstValue("Content-Encoding").orElse(null)),
          Some("gzip")
        )

        val gzipped = response.body()
        assert(gzipped.length > 0, "gzipped body must not be empty")
        // Valid gzip stream starts with the two-byte magic 0x1f 0x8b.
        assertEquals(gzipped(0), 0x1f.toByte, clue = "gzip magic byte 1 must be 0x1f")
        assertEquals(gzipped(1), 0x8b.toByte, clue = "gzip magic byte 2 must be 0x8b")

        val gz = new GZIPInputStream(new ByteArrayInputStream(gzipped))
        val decompressed =
          try gz.readAllBytes()
          finally gz.close()
        assertEquals(decompressed.length, plainBytes.length,
          clue = s"decompressed length ${decompressed.length} should equal source ${plainBytes.length}")
        assert(java.util.Arrays.equals(decompressed, plainBytes),
          "decompressed bytes must exactly equal the source index.html bytes")
      }
    }
  }

  test("static handler honors Accept-Encoding: gzip;q=0 by sending plain") {
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        // RFC 7231 sec 5.3.4: q=0 explicitly rejects this encoding.
        val rejected = get(s"$baseUri/", Map("Accept-Encoding" -> "gzip;q=0"))
        assertEquals(rejected.statusCode(), 200)
        assertEquals(headerValue(rejected, "Content-Encoding"), None,
          clue = "q=0 must be honored; server must not compress")
        assertEquals(headerValue(rejected, "Vary"), Some("Accept-Encoding"))

        // Sanity: positive q is still accepted.
        val accepted = get(s"$baseUri/", Map("Accept-Encoding" -> "gzip;q=0.5"))
        assertEquals(accepted.statusCode(), 200)
        assertEquals(headerValue(accepted, "Content-Encoding"), Some("gzip"))
      }
    }
  }

  test("static handler does not gzip non-compressible binary content") {
    withStaticSite { staticDir =>
      Files.write(staticDir.resolve("logo.png"), Array[Byte](0x89.toByte, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A))
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val response = get(s"$baseUri/logo.png", Map("Accept-Encoding" -> "gzip"))
        assertEquals(response.statusCode(), 200)
        assertEquals(headerValue(response, "Content-Encoding"), None,
          clue = "image/png is not in the compressible whitelist")
        assertEquals(headerValue(response, "Vary"), None,
          clue = "non-compressible responses don't need Vary: Accept-Encoding")
      }
    }
  }

  test("static handler caches vendor assets aggressively") {
    withStaticSite { staticDir =>
      Files.createDirectories(staticDir.resolve("vendor"))
      Files.writeString(staticDir.resolve("vendor").resolve("lib.js"), "x", StandardCharsets.UTF_8)
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val response = get(s"$baseUri/vendor/lib.js")
        assertEquals(response.statusCode(), 200)
        assertEquals(headerValue(response, "Cache-Control"), Some("public, max-age=31536000"))
        assert(headerValue(response, "ETag").isDefined, "ETag must be present on vendor responses")
      }
    }
  }

  test("optional basic auth protects the UI and analysis routes while leaving health and readiness open") {
    withStaticSite { staticDir =>
      val authConfig = HandHistoryReviewServer.BasicAuthConfig(username = "operator", password = "s3cr3t-pass")
      val authHeaders = basicAuthHeaders(authConfig.username, authConfig.password)
      withServer(staticDir, basicAuth = Some(authConfig)) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        val health = getJson(s"$baseUri/api/health")
        assertEquals(health("authenticationEnabled").bool, true)
        assertEquals(health("authenticationMode").str, "basic")

        val ready = getJson(s"$baseUri/api/ready")
        assertEquals(ready("authenticationEnabled").bool, true)
        assertEquals(ready("authenticationMode").str, "basic")

        val unauthorizedIndex = get(s"$baseUri/")
        assertEquals(unauthorizedIndex.statusCode(), 401)
        assert(headerValue(unauthorizedIndex, "WWW-Authenticate").exists(_.startsWith("Basic ")))

        val unauthorizedSubmission = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload)
        assertEquals(unauthorizedSubmission.statusCode(), 401)
        assertEquals(jsonBody(unauthorizedSubmission)("error").str, "authentication required")
        assert(headerValue(unauthorizedSubmission, "WWW-Authenticate").exists(_.startsWith("Basic ")))

        val authorizedIndex = get(s"$baseUri/", authHeaders)
        assertEquals(authorizedIndex.statusCode(), 200)
        assert(authorizedIndex.body().contains("Runtime smoke page"))

        val submissionResponse = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload, authHeaders)
        assertEquals(submissionResponse.statusCode(), 202)
        val statusUri = s"$baseUri${jsonBody(submissionResponse)("statusUrl").str}"

        val unauthorizedStatus = get(statusUri)
        assertEquals(unauthorizedStatus.statusCode(), 401)

        val completed = awaitTerminalJob(statusUri, authHeaders)
        assertEquals(completed("status").str, "completed")
      }
    }
  }

  test("user auth supports local registration, profile updates, and per-user job ownership") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          val anonymousAuth = getJson(s"$baseUri/api/auth/me")
          assertEquals(anonymousAuth("authenticationEnabled").bool, true)
          assertEquals(anonymousAuth("authenticationMode").str, "users")
          assertEquals(anonymousAuth("authenticated").bool, false)
          assertEquals(anonymousAuth("providers").arr.toVector.map(_("id").str), Vector("local"))
          assertEquals(get(s"$baseUri/").statusCode(), 200)

          val unauthorized = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload)
          assertEquals(unauthorized.statusCode(), 401)
          assertEquals(jsonBody(unauthorized)("error").str, "sign in required")

          val register = postJson(
            s"$baseUri/api/auth/register",
            """{"email":"alice@example.com","password":"correct-horse-battery","displayName":"Alice"}"""
          )
          assertEquals(register.statusCode(), 201)
          val registerJson = jsonBody(register)
          assertEquals(registerJson("authenticated").bool, true)
          assertEquals(registerJson("user")("email").str, "alice@example.com")

          // Lock in the session cookie's wire-format attributes so a future refactor
          // of sessionCookieHeader can't silently drop the protections.
          val setCookie = headerValue(register, "Set-Cookie")
            .getOrElse(fail("expected Set-Cookie on registration response"))
          assert(setCookie.contains("HttpOnly"),
            s"session cookie must be HttpOnly to block JS read access; got: $setCookie")
          assert(setCookie.contains("SameSite=Lax"),
            s"session cookie must be SameSite=Lax for CSRF defense; got: $setCookie")
          assert(setCookie.contains("Path=/"),
            s"session cookie must scope to Path=/; got: $setCookie")
          assert(setCookie.contains("Max-Age="),
            s"session cookie must declare Max-Age so browsers expire it on schedule; got: $setCookie")
          assert(!setCookie.contains("Secure"),
            s"loopback test deployment with cookieSecure=false must NOT set Secure (would prevent cookie over HTTP); got: $setCookie")
          // Insecure mode: cookie name is `sicfun_session`. The `__Host-` prefix
          // is only added when cookieSecure=true so the browser doesn't reject
          // the cookie over plain HTTP.
          assert(setCookie.startsWith("sicfun_session="),
            s"insecure-mode cookie must use plain name, got: $setCookie")
          assert(!setCookie.startsWith("__Host-"),
            s"insecure-mode cookie must not use __Host- prefix (browsers reject __Host- over HTTP), got: $setCookie")

          val ownerHeaders = authSessionHeaders(register, registerJson("csrfToken").str)

          val profile = postJson(
            s"$baseUri/api/auth/profile",
            """{"heroName":"HeroPro","preferredSite":"pokerstars","timeZone":"Europe/Madrid"}""",
            ownerHeaders
          )
          assertEquals(profile.statusCode(), 200)
          val profileJson = jsonBody(profile)
          assertEquals(profileJson("user")("heroName").str, "HeroPro")
          assertEquals(profileJson("user")("preferredSite").str, "pokerstars")
          assertEquals(profileJson("user")("timeZone").str, "Europe/Madrid")

          val me = getJsonWithHeaders(s"$baseUri/api/auth/me", Map("Cookie" -> sessionCookie(register)))
          assertEquals(me("authenticated").bool, true)
          assertEquals(me("user")("heroName").str, "HeroPro")

          val submission = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload, ownerHeaders)
          assertEquals(submission.statusCode(), 202)
          val statusUri = s"$baseUri${jsonBody(submission)("statusUrl").str}"

          val secondUser = postJson(
            s"$baseUri/api/auth/register",
            """{"email":"bob@example.com","password":"correct-horse-battery","displayName":"Bob"}"""
          )
          assertEquals(secondUser.statusCode(), 201)
          val secondHeaders = Map("Cookie" -> sessionCookie(secondUser))
          assertEquals(get(statusUri, secondHeaders).statusCode(), 404)

          val completed = awaitTerminalJob(statusUri, Map("Cookie" -> sessionCookie(register)))
          assertEquals(completed("status").str, "completed")

          val logout = postJson(s"$baseUri/api/auth/logout", "{}", ownerHeaders)
          assertEquals(logout.statusCode(), 200)
          assertEquals(jsonBody(logout)("authenticated").bool, false)

          val afterLogout = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload)
          assertEquals(afterLogout.statusCode(), 401)
        }
      }
    }
  }

  test("session cookie uses __Host- prefix and Secure flag when cookieSecure=true") {
    // RFC 6265 sec 4.1.3: a cookie with the `__Host-` prefix is rejected by
    // browsers unless it was set over HTTPS, has Path=/, and has no Domain
    // attribute. That blocks a sibling subdomain (compromised or rogue) from
    // overwriting or planting a session cookie. We only emit the prefix when
    // cookieSecure=true so the cookie remains usable over plain HTTP for
    // localhost dev.
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath, cookieSecure = true))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          val register = postJson(
            s"$baseUri/api/auth/register",
            """{"email":"secure@example.com","password":"correct-horse-battery","displayName":"Secure"}"""
          )
          assertEquals(register.statusCode(), 201)

          val setCookie = headerValue(register, "Set-Cookie")
            .getOrElse(fail("expected Set-Cookie on registration response"))
          assert(setCookie.startsWith("__Host-sicfun_session="),
            s"secure-mode cookie must use __Host- prefix, got: $setCookie")
          assert(setCookie.contains("Secure"),
            s"secure-mode cookie must set Secure, got: $setCookie")
          assert(setCookie.contains("Path=/"),
            s"__Host- prefix requires Path=/, got: $setCookie")
          assert(!setCookie.contains("Domain="),
            s"__Host- prefix forbids Domain attribute, got: $setCookie")
        }
      }
    }
  }

  test("user auth advertises Google OIDC and completes the callback flow") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        val provider = new FakeOidcProvider
        withServer(
          staticDir,
          platformAuth = Some(
            PlatformUserAuth.Config(
              storePath = storePath,
              allowLocalRegistration = false,
              oidcProviders = Vector(provider)
            )
          )
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          val anonymousAuth = getJson(s"$baseUri/api/auth/me")
          assertEquals(anonymousAuth("authenticationMode").str, "users")
          assertEquals(anonymousAuth("allowLocalRegistration").bool, false)
          assertEquals(anonymousAuth("providers").arr.toVector.map(_("id").str), Vector("local", "google"))

          val start = get(s"$baseUri${provider.startPath}")
          assertEquals(start.statusCode(), 302)
          val redirect = headerValue(start, "Location").getOrElse(fail("missing OIDC redirect"))
          val state = queryParam(redirect, "state").getOrElse(fail("missing OIDC state"))
          // /start now Set-Cookies a `sicfun_oidc_state=<state>` value the
          // callback must echo back. Without it, /callback rejects with
          // `missing_state_cookie` to block OAuth covert-redirect attacks.
          val stateCookie = headerValue(start, "Set-Cookie")
            .map(_.takeWhile(_ != ';'))
            .getOrElse(fail("missing OIDC state Set-Cookie on /start"))

          val callback = get(
            s"$baseUri${provider.callbackPath}?state=$state&code=test-code",
            Map("Cookie" -> stateCookie)
          )
          assertEquals(callback.statusCode(), 302)
          assertEquals(headerValue(callback, "Location"), Some(PlatformUserAuth.oidcSuccessRedirect))

          val me = getJsonWithHeaders(s"$baseUri/api/auth/me", Map("Cookie" -> sessionCookie(callback)))
          assertEquals(me("authenticated").bool, true)
          assertEquals(me("user")("email").str, "oidc@example.com")
          assertEquals(me("user")("displayName").str, "OIDC User")
          assert(me("user")("linkedProviders").arr.toVector.map(_.str).contains("google"))
        }
      }
    }
  }

  test("successful OIDC callback revokes a pre-existing session in the same browser") {
    // If a user is already signed in (session A) and re-authenticates via
    // OIDC -- e.g. they revisited /api/auth/oidc/google/start because they
    // wanted to switch accounts, or just clicked the button while still
    // signed in -- the callback creates a new session B. Before this fix
    // the old session record stayed in the in-memory store until its TTL
    // expired, so an attacker who had previously stolen session A's token
    // (XSS, captured network, etc.) could keep using it for up to 12h
    // after the user thought they had re-authenticated. The browser
    // overwrites its cookie automatically; this fix gives the server-side
    // store parity by revoking A as soon as B is issued.
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        val provider = new FakeOidcProvider
        withServer(
          staticDir,
          platformAuth = Some(
            PlatformUserAuth.Config(
              storePath = storePath,
              allowLocalRegistration = false,
              oidcProviders = Vector(provider)
            )
          )
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          // First OIDC sign-in -> session A.
          val start1 = get(s"$baseUri${provider.startPath}")
          val state1 = queryParam(headerValue(start1, "Location").getOrElse(""), "state")
            .getOrElse(fail("missing OIDC state on first start"))
          val stateCookie1 = headerValue(start1, "Set-Cookie")
            .map(_.takeWhile(_ != ';'))
            .getOrElse(fail("missing state cookie on first start"))
          val callback1 = get(
            s"$baseUri${provider.callbackPath}?state=$state1&code=test-code",
            Map("Cookie" -> stateCookie1)
          )
          assertEquals(callback1.statusCode(), 302)
          val sessionCookieA = sessionCookie(callback1)

          // Confirm session A is alive.
          val meA = getJsonWithHeaders(s"$baseUri/api/auth/me", Map("Cookie" -> sessionCookieA))
          assertEquals(meA("authenticated").bool, true, clue = "session A should be alive after first OIDC sign-in")

          // Second OIDC sign-in WHILE session A is in the browser -> session B.
          val start2 = get(s"$baseUri${provider.startPath}", Map("Cookie" -> sessionCookieA))
          val state2 = queryParam(headerValue(start2, "Location").getOrElse(""), "state")
            .getOrElse(fail("missing OIDC state on second start"))
          val stateCookie2 = headerValue(start2, "Set-Cookie")
            .map(_.takeWhile(_ != ';'))
            .getOrElse(fail("missing state cookie on second start"))
          // Forward both session A AND the fresh state cookie to /callback.
          val callback2 = get(
            s"$baseUri${provider.callbackPath}?state=$state2&code=test-code",
            Map("Cookie" -> s"$sessionCookieA; $stateCookie2")
          )
          assertEquals(callback2.statusCode(), 302)
          val sessionCookieB = sessionCookie(callback2)
          assertNotEquals(sessionCookieB, sessionCookieA,
            clue = "second OIDC sign-in must issue a fresh session token")

          // Session B works.
          val meB = getJsonWithHeaders(s"$baseUri/api/auth/me", Map("Cookie" -> sessionCookieB))
          assertEquals(meB("authenticated").bool, true, clue = "session B should be alive after second OIDC sign-in")

          // Session A is REVOKED -- replaying its token alone (as a hypothetical
          // stolen-cookie attacker would) must no longer resolve.
          val meAReplay = getJsonWithHeaders(s"$baseUri/api/auth/me", Map("Cookie" -> sessionCookieA))
          assertEquals(meAReplay("authenticated").bool, false,
            clue = "session A must be revoked the moment session B is issued in the same browser; a previously stolen A token must stop working")
        }
      }
    }
  }

  test("OIDC callback rejects oversize state or code params with oversize_callback_param") {
    // Both state and code are bounded above by MaxOidcParamLength (256). Our
    // own state is 32 chars, legitimate provider codes are <200, so anything
    // larger is an attacker probing the callback to amplify CPU/memory cost
    // in the OidcStateStore lookup or the upstream POST body. Reject upfront
    // with a distinct reason before any cookie / state-store / network work.
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        val provider = new FakeOidcProvider
        withServer(
          staticDir,
          platformAuth = Some(
            PlatformUserAuth.Config(
              storePath = storePath,
              allowLocalRegistration = false,
              oidcProviders = Vector(provider)
            )
          )
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"
          val hugeState = "s" * 5000
          val hugeCode = "c" * 5000

          // Oversize state -> oversize_callback_param.
          val resp1 = get(s"$baseUri${provider.callbackPath}?state=$hugeState&code=test-code")
          assertEquals(resp1.statusCode(), 302)
          val loc1 = headerValue(resp1, "Location").getOrElse(fail("missing Location"))
          assert(loc1.contains("oversize_callback_param"),
            s"oversize state must redirect with oversize_callback_param reason; got: $loc1")

          // Oversize code -> oversize_callback_param.
          val resp2 = get(s"$baseUri${provider.callbackPath}?state=test-state&code=$hugeCode")
          assertEquals(resp2.statusCode(), 302)
          val loc2 = headerValue(resp2, "Location").getOrElse(fail("missing Location"))
          assert(loc2.contains("oversize_callback_param"),
            s"oversize code must redirect with oversize_callback_param reason; got: $loc2")

          // Redirect URLs must stay short -- the cap prevents the attacker
          // from blowing up the redirect Location via state or code.
          assert(loc1.length < 1024,
            s"oversize-state redirect should be small (got ${loc1.length}); the cap prevents echoing the attacker payload")
          assert(loc2.length < 1024,
            s"oversize-code redirect should be small (got ${loc2.length}); the cap prevents echoing the attacker payload")
        }
      }
    }
  }

  test("PlatformUserAuth.oidcFailureRedirect caps the error string so internal wrapped messages cannot blow up the redirect URL") {
    // Defense-in-depth complement to the AuthStack-level cap on the
    // provider's ?error= callback param. finishOidc's wrapped failure
    // messages (e.g. "Google OIDC exchange failed: <ujson InvalidData with
    // the full response body>") flow through oidcFailureRedirect unchecked
    // -- a 2 MB upstream parse failure would otherwise produce a 2 MB
    // Location header browsers refuse to follow. Bound the input to 256
    // chars + truncation marker, same shape as the other caps.
    val huge = "z" * 2000
    val location = PlatformUserAuth.oidcFailureRedirect(huge)
    assert(location.startsWith("/?auth_error="),
      clue = s"location should be the failure landing prefix; got: ${location.take(60)}")
    assert(location.length < 600,
      clue = s"location must stay small even for huge input; got ${location.length} bytes")
    assert(location.contains("%28truncated%29"),
      clue = s"location should include the URL-encoded truncation marker; got: ${location.take(120)}")

    // Short inputs pass through untruncated.
    val short = PlatformUserAuth.oidcFailureRedirect("invalid_request")
    assertEquals(short, "/?auth_error=invalid_request",
      clue = s"short inputs should pass through with simple URL encoding")
  }

  test("OIDC callback caps the provider-supplied ?error= string before logging and redirecting") {
    // The /callback route is not rate-limited (it's a normal user flow that
    // fires once per sign-in) so without a cap an attacker who hits it
    // directly with `?error=<huge string>` could bloat the audit log
    // arbitrarily AND produce a redirect URL longer than what browsers
    // accept (~2-8 KB), turning the polite auth_error landing into a
    // failed redirect. Cap at 256 chars.
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        val provider = new FakeOidcProvider
        withServer(
          staticDir,
          platformAuth = Some(
            PlatformUserAuth.Config(
              storePath = storePath,
              allowLocalRegistration = false,
              oidcProviders = Vector(provider)
            )
          )
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"
          // 10 KB attacker-controlled error string. Use only URL-safe chars so
          // URI.create accepts it.
          val huge = "x" * 10000
          val response = get(s"$baseUri${provider.callbackPath}?error=$huge")
          assertEquals(response.statusCode(), 302)
          val location = headerValue(response, "Location").getOrElse(fail("missing Location"))
          // The whole Location header (path + query) must stay well under
          // typical browser URL caps. 1 KB is generous slack above the 256-char
          // cap plus the `/?auth_error=` prefix and the "...(truncated)"
          // marker. If this assertion fires it means the cap regressed.
          assert(location.length < 1024,
            s"failure redirect Location is too long (${location.length} bytes): ${location.take(120)}...")
          // The truncation marker '...(truncated)' is URL-encoded in the
          // Location header as '...%28truncated%29' because urlEncode escapes
          // parentheses. The frontend's URLSearchParams will decode it back
          // before display, so the user sees the human-readable marker.
          assert(location.contains("%28truncated%29"),
            s"capped error should include the URL-encoded truncation marker so the user knows the value was clamped; got: $location")
        }
      }
    }
  }

  test("OIDC callback rejects requests that lack the state cookie issued at /start") {
    // OAuth 2.0 BCP "covert-redirect" / login-CSRF mitigation: even if an
    // attacker holds a state value that WE issued (e.g. they kicked off a
    // partial flow themselves), they cannot forward the resulting
    // ?state=X&code=Y URL to a victim and have the victim's browser end up
    // bound to the attacker's account -- because the victim's browser lacks
    // the `sicfun_oidc_state=X` cookie that /start set on the attacker's
    // browser. Without the cookie, /callback must abort before exchanging the
    // code with the provider.
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        val provider = new FakeOidcProvider
        withServer(
          staticDir,
          platformAuth = Some(
            PlatformUserAuth.Config(
              storePath = storePath,
              allowLocalRegistration = false,
              oidcProviders = Vector(provider)
            )
          )
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"
          val start = get(s"$baseUri${provider.startPath}")
          val state = queryParam(headerValue(start, "Location").getOrElse(""), "state")
            .getOrElse(fail("missing OIDC state"))

          // Callback without the state cookie -> must fail with
          // missing_state_cookie, NOT proceed to finishOidc.
          val noCookie = get(s"$baseUri${provider.callbackPath}?state=$state&code=test-code")
          assertEquals(noCookie.statusCode(), 302)
          val noCookieLocation = headerValue(noCookie, "Location").getOrElse(fail("missing Location"))
          assert(noCookieLocation.contains("missing_state_cookie"),
            s"callback without state cookie must surface missing_state_cookie; got: $noCookieLocation")
          assertEquals(headerValue(noCookie, "Set-Cookie"), None,
            "rejected callback must not install a session cookie")

          // Callback with a MISMATCHED state cookie -> state_cookie_mismatch.
          // Simulates an attacker forwarding their own state to the victim's
          // browser, which carries a different state cookie from its own flow.
          val mismatchedCookie = get(
            s"$baseUri${provider.callbackPath}?state=$state&code=test-code",
            Map("Cookie" -> "sicfun_oidc_state=different-state")
          )
          assertEquals(mismatchedCookie.statusCode(), 302)
          val mismatchedLocation = headerValue(mismatchedCookie, "Location").getOrElse(fail("missing Location"))
          assert(mismatchedLocation.contains("state_cookie_mismatch"),
            s"callback with wrong state cookie must surface state_cookie_mismatch; got: $mismatchedLocation")
          assertEquals(headerValue(mismatchedCookie, "Set-Cookie"), None,
            "rejected callback must not install a session cookie")
        }
      }
    }
  }

  test("OIDC routes answer OPTIONS with 204 + Allow: GET, OPTIONS and reject other verbs with 405 + the same Allow") {
    // The OIDC /start and /callback paths are GET-only. Before this fix they
    // returned 405 with Allow: GET on every non-GET method including OPTIONS,
    // which was inconsistent with the rest of the API (JsonHandler routes
    // accept OPTIONS preflight with 200 + Allow: <verbs>, OPTIONS). Now
    // RedirectHandler handles OPTIONS uniformly with 204 + Allow: GET, OPTIONS,
    // and the 405 response on POST/DELETE/etc also advertises the OPTIONS
    // method per RFC 7231 sec 7.4.1.
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        val provider = new FakeOidcProvider
        withServer(
          staticDir,
          platformAuth = Some(
            PlatformUserAuth.Config(
              storePath = storePath,
              allowLocalRegistration = false,
              oidcProviders = Vector(provider)
            )
          )
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          // OPTIONS on /start.
          val optionsStart = HttpRequest.newBuilder(URI.create(s"$baseUri${provider.startPath}"))
            .method("OPTIONS", HttpRequest.BodyPublishers.noBody())
            .build()
          val optionsStartResp = httpClient.send(optionsStart, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8))
          assertEquals(optionsStartResp.statusCode(), 204, clue = "/start OPTIONS must return 204")
          assertEquals(headerValue(optionsStartResp, "Allow"), Some("GET, OPTIONS"),
            clue = "/start OPTIONS must list both GET and OPTIONS in Allow")
          assertEquals(optionsStartResp.body(), "", clue = "/start OPTIONS must not include a body")

          // OPTIONS on /callback.
          val optionsCallback = HttpRequest.newBuilder(URI.create(s"$baseUri${provider.callbackPath}"))
            .method("OPTIONS", HttpRequest.BodyPublishers.noBody())
            .build()
          val optionsCallbackResp = httpClient.send(optionsCallback, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8))
          assertEquals(optionsCallbackResp.statusCode(), 204, clue = "/callback OPTIONS must return 204")
          assertEquals(headerValue(optionsCallbackResp, "Allow"), Some("GET, OPTIONS"),
            clue = "/callback OPTIONS must list both GET and OPTIONS in Allow")

          // POST on /start -> 405 with Allow: GET, OPTIONS (the 405 path now mirrors the OPTIONS advertisement).
          val postStart = HttpRequest.newBuilder(URI.create(s"$baseUri${provider.startPath}"))
            .POST(HttpRequest.BodyPublishers.noBody())
            .build()
          val postStartResp = httpClient.send(postStart, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8))
          assertEquals(postStartResp.statusCode(), 405, clue = "/start POST must return 405")
          assertEquals(headerValue(postStartResp, "Allow"), Some("GET, OPTIONS"),
            clue = "/start 405 Allow must include OPTIONS now that OPTIONS is supported")
        }
      }
    }
  }

  test("OIDC /start emits a HttpOnly SameSite=Lax state cookie bound to the redirect's state value") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        val provider = new FakeOidcProvider
        withServer(
          staticDir,
          platformAuth = Some(
            PlatformUserAuth.Config(
              storePath = storePath,
              allowLocalRegistration = false,
              oidcProviders = Vector(provider)
            )
          )
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"
          val start = get(s"$baseUri${provider.startPath}")
          val urlState = queryParam(headerValue(start, "Location").getOrElse(""), "state")
            .getOrElse(fail("missing OIDC state in redirect"))
          val setCookie = headerValue(start, "Set-Cookie").getOrElse(fail("missing Set-Cookie on /start"))
          assert(setCookie.startsWith("sicfun_oidc_state="),
            s"state cookie name should be sicfun_oidc_state when cookieSecure=false; got: $setCookie")
          val cookieValue = setCookie.takeWhile(_ != ';').drop("sicfun_oidc_state=".length)
          assertEquals(cookieValue, urlState,
            "state cookie value must match the URL state so the callback's compare succeeds")
          assert(setCookie.toLowerCase.contains("httponly"),
            s"state cookie must be HttpOnly so JS cannot read or set it; got: $setCookie")
          assert(setCookie.toLowerCase.contains("samesite=lax"),
            s"state cookie must be SameSite=Lax so the provider's top-level redirect can send it back; got: $setCookie")
          assert(setCookie.toLowerCase.contains("max-age="),
            s"state cookie must have a Max-Age (10 minutes); got: $setCookie")
        }
      }
    }
  }

  test("OIDC callback handles provider-side ?error= by redirecting to the failure landing page") {
    // When the user denies consent on Google's screen, or the authorization
    // code expires before redemption, the provider redirects back to our
    // callback URL with ?error=... (and no code). The handler must not crash
    // looking up state -- it should short-circuit to the failure redirect so
    // the frontend can show the user a polite error and an option to retry.
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        val provider = new FakeOidcProvider
        withServer(
          staticDir,
          platformAuth = Some(
            PlatformUserAuth.Config(
              storePath = storePath,
              allowLocalRegistration = false,
              oidcProviders = Vector(provider)
            )
          )
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          val callback = get(s"$baseUri${provider.callbackPath}?error=access_denied")
          assertEquals(callback.statusCode(), 302)
          val location = headerValue(callback, "Location").getOrElse(fail("missing failure redirect"))
          assert(location.contains("access_denied"),
            s"failure redirect should include the provider error code, got: $location")
          // No Set-Cookie on failure: the user must not end up with a half-baked session.
          assertEquals(headerValue(callback, "Set-Cookie"), None,
            "OIDC failure must not emit a session cookie")
        }
      }
    }
  }

  test("OIDC upsert rejects oversize subject and drops oversize avatar URL") {
    // upsertOidcIdentity now validates the email format/length, caps the
    // subject at 256 chars (reject), caps the avatar URL at 2048 chars (drop
    // the field but still upsert), and truncates displayName to 96 chars. The
    // configured Google provider would never emit these values, but a future
    // provider or a tampered response could. Use custom fake providers per
    // case to drive the validation paths end-to-end via the callback HTTP
    // flow.
    def runCase(provider: PlatformUserAuth.OidcProvider)(check: HttpResponse[String] => Unit): Unit =
      withStaticSite { staticDir =>
        withUserStorePath { storePath =>
          withServer(
            staticDir,
            platformAuth = Some(
              PlatformUserAuth.Config(
                storePath = storePath,
                allowLocalRegistration = false,
                oidcProviders = Vector(provider)
              )
            )
          ) { server =>
            val baseUri = s"http://${server.binding.host}:${server.binding.port}"
            val start = get(s"$baseUri${provider.startPath}")
            val state = queryParam(headerValue(start, "Location").getOrElse(""), "state")
              .getOrElse(fail("missing OIDC state"))
            val stateCookie = headerValue(start, "Set-Cookie")
              .map(_.takeWhile(_ != ';'))
              .getOrElse(fail("missing OIDC state Set-Cookie on /start"))
            val callback = get(
              s"$baseUri${provider.callbackPath}?state=$state&code=anything",
              Map("Cookie" -> stateCookie)
            )
            check(callback)
          }
        }
      }

    // 1. Oversize subject -> reject the upsert with a clear error in the redirect.
    val bigSubject = "x" * 300
    val oversizeSubjectProvider = new PlatformUserAuth.OidcProvider:
      override val id = "google"
      override val displayName = "Google"
      override def authorizationUri(state: String, codeChallenge: String): String =
        s"https://accounts.google.test/o/oauth2/v2/auth?state=$state&code_challenge=$codeChallenge"
      override def exchangeCode(code: String, codeVerifier: String): Either[String, PlatformUserAuth.OidcIdentity] =
        Right(PlatformUserAuth.OidcIdentity(
          subject = bigSubject,
          email = "oidc@example.com",
          displayName = "Big Subject User",
          avatarUrl = None
        ))
    runCase(oversizeSubjectProvider) { callback =>
      assertEquals(callback.statusCode(), 302)
      assertEquals(headerValue(callback, "Set-Cookie"), None,
        "rejected OIDC upsert must not emit a session cookie")
      val location = headerValue(callback, "Location").getOrElse(fail("missing redirect"))
      assert(location.toLowerCase.contains("subject"),
        s"failure redirect should mention 'subject', got: $location")
    }

    // 2. Oversize avatar URL -> drop the avatar but still upsert successfully.
    val bigAvatarProvider = new PlatformUserAuth.OidcProvider:
      override val id = "google"
      override val displayName = "Google"
      override def authorizationUri(state: String, codeChallenge: String): String =
        s"https://accounts.google.test/o/oauth2/v2/auth?state=$state&code_challenge=$codeChallenge"
      override def exchangeCode(code: String, codeVerifier: String): Either[String, PlatformUserAuth.OidcIdentity] =
        Right(PlatformUserAuth.OidcIdentity(
          subject = "fake-google-ok",
          email = "oidc@example.com",
          displayName = "Big Avatar User",
          avatarUrl = Some("https://example.com/" + ("a" * 3000) + ".png")
        ))
    runCase(bigAvatarProvider) { callback =>
      assertEquals(callback.statusCode(), 302)
      assertEquals(headerValue(callback, "Location"), Some(PlatformUserAuth.oidcSuccessRedirect),
        "oversize avatar should not block the upsert, just drop the field")
      assert(headerValue(callback, "Set-Cookie").isDefined,
        "successful OIDC upsert must emit a session cookie")
    }
  }

  test("OIDC callback refuses malformed queries (no params, partial params, unknown state)") {
    // Three callback-side failure modes besides the provider-error case:
    //   1. No query at all -- nothing to validate against.
    //   2. Only `state` or only `code` -- attacker probing for partial-form
    //      acceptance.
    //   3. Both present but `state` does not match any issued flow -- expired
    //      state, replay attempt, or attacker fishing for a working callback.
    // All four must redirect to the failure landing page without emitting a
    // session cookie. The first three share the same `missing_code_or_state`
    // reason; the fourth surfaces the finishOidc Left as the redirect query.
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        val provider = new FakeOidcProvider
        withServer(
          staticDir,
          platformAuth = Some(
            PlatformUserAuth.Config(
              storePath = storePath,
              allowLocalRegistration = false,
              oidcProviders = Vector(provider)
            )
          )
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          val cases = Vector(
            // (query, cookie-or-empty, expected-substring-in-failure-redirect)
            ("", Map.empty[String, String], "missing_code_or_state"),
            ("?state=abc", Map.empty[String, String], "missing_code_or_state"),
            ("?code=xyz", Map.empty[String, String], "missing_code_or_state"),
            // Unknown-state case now needs the state cookie to clear the
            // covert-redirect check FIRST so the underlying "state expired or
            // invalid" finishOidc error is what surfaces in the redirect.
            ("?state=this-state-was-never-issued&code=xyz",
              Map("Cookie" -> "sicfun_oidc_state=this-state-was-never-issued"),
              "OIDC")
          )
          for (query, headers, expectedSubstring) <- cases do
            val resp = get(s"$baseUri${provider.callbackPath}$query", headers)
            assertEquals(resp.statusCode(), 302, clue = s"query=$query")
            val location = headerValue(resp, "Location").getOrElse(fail(s"missing Location for query=$query"))
            assert(location.contains("/?auth_error="),
              s"query=$query must redirect to failure landing, got: $location")
            assert(location.toLowerCase.contains(expectedSubstring.toLowerCase),
              s"query=$query failure redirect should mention '$expectedSubstring', got: $location")
            assertEquals(headerValue(resp, "Set-Cookie"), None,
              s"query=$query must not emit a session cookie")
        }
      }
    }
  }

  test("drain signal flips readiness to 503 and rejects new analysis submissions") {
    withStaticSite { staticDir =>
      val root = Files.createTempDirectory("hand-history-review-drain-")
      try
        val drainSignalFile = root.resolve("deploy-drain.signal")
        withServer(staticDir, drainSignalFile = Some(drainSignalFile)) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          assertEquals(get(s"$baseUri/api/ready").statusCode(), 200)

          Files.writeString(drainSignalFile, "draining", StandardCharsets.UTF_8)

          val ready = get(s"$baseUri/api/ready")
          assertEquals(ready.statusCode(), 503)
          val readyJson = jsonBody(ready)
          assertEquals(readyJson("ready").bool, false)
          assertEquals(readyJson("reason").str, "draining")
          assertEquals(readyJson("draining").bool, true)
          assertEquals(readyJson("acceptingAnalysisJobs").bool, false)
          assertEquals(readyJson("authenticationEnabled").bool, false)
          assertEquals(readyJson("rateLimitSubmitsPerMinute").num.toInt, 6)
          assertEquals(readyJson("rateLimitStatusPerMinute").num.toInt, 240)
          assertEquals(readyJson("rateLimitClientIpSource").str, "remote-address")
          assertEquals(readyJson("drainSignalPresent").bool, true)

          val health = getJson(s"$baseUri/api/health")
          assertEquals(health("ok").bool, true)
          assertEquals(health("ready").bool, false)
          assertEquals(health("readyReason").str, "draining")
          assertEquals(health("draining").bool, true)
          assertEquals(health("acceptingAnalysisJobs").bool, false)
          assertEquals(health("authenticationEnabled").bool, false)
          assertEquals(health("rateLimitSubmitsPerMinute").num.toInt, 6)
          assertEquals(health("rateLimitStatusPerMinute").num.toInt, 240)
          assertEquals(health("rateLimitClientIpSource").str, "remote-address")
          assertEquals(health("drainSignalPresent").bool, true)
          assertEquals(health("timedOutWorkersInFlight").num.toInt, 0)

          val rejected = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload)
          assertEquals(rejected.statusCode(), 503)
          assert(rejected.body().contains("draining"))
          assertEquals(headerValue(rejected, "Retry-After"), Some("5"),
            clue = "RFC 7231 sec 6.6.4: 503 responses SHOULD include Retry-After so clients back off intelligently")

          Files.deleteIfExists(drainSignalFile)

          val recovered = get(s"$baseUri/api/ready")
          assertEquals(recovered.statusCode(), 200)
          val recoveredJson = jsonBody(recovered)
          assertEquals(recoveredJson("ready").bool, true)
          assertEquals(recoveredJson("reason").str, "accepting-traffic")
        }
      finally
        deleteRecursively(root)
    }
  }

  test("analysis submission returns a job id, keeps the server responsive, and completes via polling") {
    withStaticSite { staticDir =>
      val backend = new BlockingBackend(Right(sampleAnalysisResult))
      withServer(staticDir, backend = backend) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        val submissionResponse = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload)
        assertEquals(submissionResponse.statusCode(), 202)
        assertEquals(headerValue(submissionResponse, "Retry-After"), Some("1"))
        val submission = jsonBody(submissionResponse)
        assertEquals(submission("status").str, "queued")
        val statusUri = s"$baseUri${submission("statusUrl").str}"

        assert(backend.started.await(3, TimeUnit.SECONDS), "analysis backend never started")

        val runningResponse = get(statusUri)
        assertEquals(headerValue(runningResponse, "Retry-After"), Some("1"))
        val running = jsonBody(runningResponse)
        assertEquals(running("status").str, "running")
        val health = getJson(s"$baseUri/api/health")
        assertEquals(health("runningJobs").num.toInt, 1)
        assertEquals(health("queuedJobs").num.toInt, 0)
        assertEquals(get(s"$baseUri/").statusCode(), 200)

        backend.release.countDown()

        val completed = awaitTerminalJob(statusUri)
        assertEquals(completed("status").str, "completed")
        assertEquals(completed("result")("site").str, "PokerStars")
        assertEquals(completed("result")("handsImported").num.toInt, 1)
        assertEquals(completed("result")("handsAnalyzed").num.toInt, 1)
        assertEquals(completed("result")("trace")("request")("normalizedHeroName").str, "Hero")
        assertEquals(completed("result")("trace")("import")("handsImported").num.toInt, 1)
        assertEquals(completed("result")("trace")("hands")(0)("status").str, "analyzed")
        assertEquals(completed("result")("trace")("summary")("decisionsAnalyzed").num.toInt, 1)
      }
    }
  }

  test("playing hall submission returns a job id and completes via polling") {
    withStaticSite { staticDir =>
      val backend = new BlockingPlayingHallBackend(Right(samplePlayingHallResult))
      withServer(staticDir, playingHallBackend = backend) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        val submissionResponse = postJson(s"$baseUri/api/playing-hall", validPlayingHallPayload)
        assertEquals(submissionResponse.statusCode(), 202)
        assertEquals(headerValue(submissionResponse, "Retry-After"), Some("1"))
        val submission = jsonBody(submissionResponse)
        assertEquals(submission("status").str, "queued")
        val statusUri = s"$baseUri${submission("statusUrl").str}"

        assert(backend.started.await(3, TimeUnit.SECONDS), "playing hall backend never started")

        val runningResponse = get(statusUri)
        assertEquals(headerValue(runningResponse, "Retry-After"), Some("1"))
        val running = jsonBody(runningResponse)
        assertEquals(running("status").str, "running")

        backend.release.countDown()

        val completed = awaitTerminalJob(statusUri)
        assertEquals(completed("status").str, "completed")
        assertEquals(completed("result")("request")("heroStyle").str, "adaptive")
        assertEquals(completed("result")("summary")("handsPlayed").num.toInt, 240)
        assertEquals(completed("result")("summary")("actionCounts")("Raise(2.5)").num.toInt, 122)
      }
    }
  }

  test("playing hall running job can be cancelled with DELETE") {
    withStaticSite { staticDir =>
      val backend = new BlockingPlayingHallBackend(Right(samplePlayingHallResult))
      withServer(staticDir, playingHallBackend = backend) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val submissionResponse = postJson(s"$baseUri/api/playing-hall", validPlayingHallPayload)
        assertEquals(submissionResponse.statusCode(), 202)
        val statusUri = s"$baseUri${jsonBody(submissionResponse)("statusUrl").str}"

        assert(backend.started.await(3, TimeUnit.SECONDS), "playing hall backend never started")
        try
          val cancelResponse = delete(statusUri)
          assertEquals(cancelResponse.statusCode(), 200)
          val cancelBody = jsonBody(cancelResponse)
          assertEquals(cancelBody("status").str, "cancelled")

          backend.release.countDown()
          val cancelled = awaitTerminalJob(statusUri)
          assertEquals(cancelled("status").str, "cancelled")
          assertEquals(cancelled("result")("summary")("handsPlayed").num.toInt, 240)
        finally
          backend.release.countDown()
      }
    }
  }

  test("playing hall DELETE enforces CSRF under platform-user auth") {
    // DELETE is state-changing -- it MUST require the same X-CSRF-Token
    // header that POST /api/playing-hall and the other state-changing
    // routes require. The cross-origin browser path is already closed by
    // the missing CORS headers + the fact that DELETE triggers a
    // preflight, but defense-in-depth says to gate it the same way as
    // the rest. Without this check a scripted local proxy could fire
    // cooperative cancel using just the session cookie.
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        val backend = new BlockingPlayingHallBackend(Right(samplePlayingHallResult))
        withServer(
          staticDir,
          playingHallBackend = backend,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"
          val register = postJson(
            s"$baseUri/api/auth/register",
            """{"email":"hall-cancel-csrf@example.com","password":"correct-horse-battery","displayName":"Tester"}"""
          )
          assertEquals(register.statusCode(), 201)
          val registerJson = jsonBody(register)
          val ownerHeaders = authSessionHeaders(register, registerJson("csrfToken").str)

          val submission = postJson(s"$baseUri/api/playing-hall", validPlayingHallPayload, ownerHeaders)
          assertEquals(submission.statusCode(), 202)
          val statusUri = s"$baseUri${jsonBody(submission)("statusUrl").str}"
          assert(backend.started.await(3, TimeUnit.SECONDS), "playing hall backend never started")

          try
            // DELETE with the SESSION cookie but NO X-CSRF-Token: 403.
            val cookieOnly = Map("Cookie" -> sessionCookie(register))
            val withoutCsrf = delete(statusUri, cookieOnly)
            assertEquals(withoutCsrf.statusCode(), 403,
              clue = s"DELETE without X-CSRF-Token must return 403; got ${withoutCsrf.statusCode()}")
            assert(jsonBody(withoutCsrf)("error").str.toLowerCase.contains("csrf"),
              clue = "error message should mention csrf")

            // DELETE with the proper CSRF header succeeds.
            val withCsrf = delete(statusUri, ownerHeaders)
            assertEquals(withCsrf.statusCode(), 200,
              clue = "DELETE with X-CSRF-Token must succeed")
          finally
            backend.release.countDown()
        }
      }
    }
  }

  test("playing hall cancellation returns 404 for unknown jobs") {
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        val response = delete(s"$baseUri/api/playing-hall/jobs/not-a-real-job")
        assertEquals(response.statusCode(), 404)
        assert(jsonBody(response)("error").str.contains("not found"))
      }
    }
  }

  test("oversize jobId path segments return a generic 404 without echoing the value back through the response body") {
    // The job-id URL segment goes into Map.get keys (O(N) hash work) and
    // into the "job not found: <id>" 404 response body. A long jobId in the
    // URL was therefore both a CPU and bandwidth amplifier per request (an
    // authenticated attacker rate-limited at 240/min via JobStatus bucket
    // could echo ~15 MB/min of attacker-controlled payload back in 404
    // bodies). Cap at 128 chars (UUIDs are 36) and return the same 404 the
    // genuine-unknown-job path returns, with a generic "not found" message
    // that does NOT include the requested id.
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val huge = "z" * 4096
        val response = delete(s"$baseUri/api/playing-hall/jobs/$huge")
        assertEquals(response.statusCode(), 404)
        val errorBody = jsonBody(response)("error").str
        assert(errorBody.contains("not found"),
          clue = s"oversize jobId should still surface a 'not found' message; got: $errorBody")
        assert(!errorBody.contains(huge.take(200)),
          clue = "oversize jobId must NOT be echoed in the 404 response body (would burn upstream bandwidth and let the attacker amplify)")
        // Total response size must stay tiny -- the attacker mailed in ~4 KB
        // of jobId and they get back ~50 bytes of JSON, not 4 KB echoed.
        assert(response.body().length < 256,
          clue = s"404 response body should stay small; got ${response.body().length} bytes")
      }
    }
  }

  test("analyze-hand-history rejects whitespace-only handHistoryText with 400 before queueing a job") {
    // requiredString filters empty BEFORE trim, so "   " or "\n\n\n" used
    // to survive validation and queue a job whose payload was "" -- the
    // parser then produced a confusing 'no playable hands' failure deep
    // in the pipeline instead of a clean 400 at the API boundary. The
    // post-trim re-check now bounces it at parse time.
    withStaticSite { staticDir =>
      withServer(staticDir, maxUploadBytes = 4096) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        for whitespacePayload <- Vector(
            """{"handHistoryText":"   "}""",
            """{"handHistoryText":"\n\n\n"}""",
            """{"handHistoryText":"\t\r\n "}""",
            // BOM-only file: a leading U+FEFF survives String.trim
            // (which only drops chars <= U+0020) but stripPrefix("\uFEFF")
            // peels it off before the non-empty re-check. Without that
            // peel a "BOM + newlines" file would queue a job whose
            // payload is empty after the downstream BOM-stripping pass.
            "{\"handHistoryText\":\"\\uFEFF\\n\\n\\n\"}"
          )
        do
          val response = postJson(s"$baseUri/api/analyze-hand-history", whitespacePayload)
          assertEquals(response.statusCode(), 400,
            clue = s"whitespace-only handHistoryText must be rejected at the API boundary; payload=$whitespacePayload")
          val errorBody = jsonBody(response)("error").str
          assertEquals(errorBody, "handHistoryText is required",
            clue = s"error message should be the standard required-field message; payload=$whitespacePayload")
      }
    }
  }

  test("analyze-hand-history rejects heroName containing control characters") {
    // Defense in depth for log-injection: heroName with embedded \n / \r /
    // NUL / ESC has no legitimate use (the hand-history matcher compares
    // against player names parsed from the file, which never contain
    // controls) and would splice fake structured key=value pairs into
    // any future audit log line that includes heroName. Same control-char
    // rule PlatformUserAuth.sanitizeOptionalField uses for profile fields.
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        for control <- Vector("\\n", "\\r", "\\u0000", "\\u001b", "\\u007f") do
          val payload = s"""{"handHistoryText":"x","heroName":"Hero${control}Injected"}"""
          val response = postJson(s"$baseUri/api/analyze-hand-history", payload)
          assertEquals(response.statusCode(), 400,
            clue = s"heroName with control-char escape $control must be rejected at the API boundary; payload=$payload")
          val errorBody = jsonBody(response)("error").str
          assertEquals(errorBody, "heroName must not contain control characters",
            clue = s"error should be the generic control-char message; got: $errorBody")
      }
    }
  }

  test("analyze-hand-history rejects oversize heroName with 400 before queueing a job") {
    // Frontend caps heroName at 64 via <input maxlength="64">, matching the
    // PlatformUserAuth profile heroName cap. A scripted client that bypasses
    // the HTML would otherwise stuff a multi-kilobyte heroName into the
    // request body, hit the worker's comparison loop, and -- worse -- could
    // be quoted back through any future audit log line that includes
    // heroName. Cap upfront with a generic length message that does not
    // echo the value.
    withStaticSite { staticDir =>
      withServer(staticDir, maxUploadBytes = 4096) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val marker = "MARKER_HERO_PAYLOAD_AA"
        val hugeHero = marker + ("x" * 200)
        val payload = s"""{"handHistoryText":"x","heroName":"$hugeHero"}"""
        val response = postJson(s"$baseUri/api/analyze-hand-history", payload)
        assertEquals(response.statusCode(), 400)
        val errorBody = jsonBody(response)("error").str
        assert(errorBody.contains("at most"),
          clue = s"oversize heroName should surface a 'must be at most ...' message; got: $errorBody")
        assert(!errorBody.contains(marker),
          clue = s"oversize heroName value must NOT be echoed in the 400 response body; got: $errorBody")
      }
    }
  }

  test("analyze-hand-history rejects oversize 'site' field without echoing the value back") {
    // HandHistorySite.parse returns "unsupported hand-history site: <value>"
    // for unknown inputs. parseOptionalSite used to forward that verbatim,
    // so a request with a 2 KB attacker-controlled site value would land
    // 2 KB of attacker payload in the 400 response. Cap at 64 chars upfront
    // -- recognised site aliases are all under 12 chars -- so the echo path
    // is fenced off before it can amplify.
    withStaticSite { staticDir =>
      withServer(staticDir, maxUploadBytes = 4096) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val marker = "MARKER_PAYLOAD_SITE_AA"
        val hugeSite = marker + ("x" * 2000)
        val payload =
          s"""{"handHistoryText":"x","site":"$hugeSite","heroName":"Hero"}"""
        val response = postJson(s"$baseUri/api/analyze-hand-history", payload)
        assertEquals(response.statusCode(), 400)
        val errorBody = jsonBody(response)("error").str
        assert(errorBody.contains("at most"),
          clue = s"oversize site should surface a 'must be at most ...' message; got: $errorBody")
        assert(!errorBody.contains(marker),
          clue = s"oversize site value must NOT be echoed in the 400 response body; got: $errorBody")
      }
    }
  }

  test("JSON parse-error responses cap the underlying exception message so a non-object body cannot echo back at full size") {
    // ujson.read(body).obj throws ujson.Value.InvalidData when body is a JSON
    // value but not an object (e.g. a giant string or array). The exception's
    // getMessage is 'Expected Obj: <data.toString>' -- for a 4 KB raw string
    // body, ~4 KB of attacker payload would otherwise round-trip into the
    // 'invalid JSON request: ...' 400 response body. capParseErrorMessage
    // bounds that round-trip at 256 chars + truncation marker. The first
    // 256 chars of the message can still include attacker-controlled prefix
    // (we don't fully redact), but the total response body size is bounded.
    withStaticSite { staticDir =>
      withServer(staticDir, maxUploadBytes = 4096) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        // 3 KB raw string body (valid JSON, but a JSON string, not an object).
        val hugeString = "\"" + ("y" * 3000) + "\""
        val request = HttpRequest.newBuilder(URI.create(s"$baseUri/api/playing-hall"))
          .header("Content-Type", "application/json")
          .POST(HttpRequest.BodyPublishers.ofString(hugeString, StandardCharsets.UTF_8))
          .build()
        val response = httpClient.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8))
        // The response body must be bounded -- without the cap, the
        // 'invalid JSON request: Expected Obj: yyyy...3000y...' message
        // would be ~3 KB; with the cap it's ~256 chars + framing.
        assert(response.body().length < 600,
          clue = s"3 KB request body should produce <600-byte error response with cap; got ${response.body().length} bytes")
      }
    }
  }

  test("villainPool with non-string entries reports the type name, not the offending value") {
    // optionalStringArray used to fall through to `other.str` on non-string
    // entries; for Obj/Arr/Bool that throws ujson.Value.InvalidData whose
    // getMessage includes data.toString -- a multi-KB nested object inside
    // the array would round-trip back as a "invalid JSON request: <huge>"
    // 400 body. Now we map non-strings to their JSON type name instead, so
    // the downstream allowlist check emits a clean
    // "unsupported entries: object" with no attacker payload echoed.
    withStaticSite { staticDir =>
      withServer(staticDir, maxUploadBytes = 4096) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val payload =
          """{"hands":10,"tableCount":1,"playerCount":2,"heroStyle":"gto","heroPosition":"Button","gtoMode":"fast","villainPool":[{"MARKER_PAYLOAD_AA":"MARKER_PAYLOAD_BB"}],"heroExplorationRate":0,"raiseSize":2.5,"bunchingTrials":1,"equityTrials":1,"learnEveryHands":0,"learningWindowSamples":0,"seed":1}"""
        val response = postJson(s"$baseUri/api/playing-hall", payload)
        assertEquals(response.statusCode(), 400)
        val errorBody = jsonBody(response)("error").str
        assert(!errorBody.contains("MARKER_PAYLOAD"),
          clue = s"villainPool non-string entry must NOT be echoed in the response body; got: $errorBody")
      }
    }
  }

  test("JSON-type-mismatch errors report only the offending type, not the offending value") {
    // The optional* parsers used to format type-mismatch errors as
    // `"hands must be an integer, got ${ujson.write(other)}"`. For an
    // attacker who sent {"hands": <1.9 MB nested object>} via /api/playing-hall
    // (2 MB body cap), that produced 1.9 MB of attacker-controlled payload
    // echoed verbatim in the 400 response body. The type alone is all a
    // legitimate client needs to fix their request; the value adds nothing
    // they don't already know. Switch to reporting only the JSON type name
    // (string/number/boolean/null/array/object).
    withStaticSite { staticDir =>
      withServer(staticDir, maxUploadBytes = 4096) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        // `hands` must be int. Send an array of distinctive marker values
        // so we can verify NONE of them landed in the error body.
        val payload =
          """{"hands":["MARKER_PAYLOAD_AA","MARKER_PAYLOAD_BB","MARKER_PAYLOAD_CC"],"tableCount":1,"playerCount":2,"heroStyle":"gto","heroPosition":"Button","gtoMode":"fast","villainPool":["gto"],"heroExplorationRate":0,"raiseSize":2.5,"bunchingTrials":1,"equityTrials":1,"learnEveryHands":0,"learningWindowSamples":0,"seed":1}"""
        val response = postJson(s"$baseUri/api/playing-hall", payload)
        assertEquals(response.statusCode(), 400)
        val errorBody = jsonBody(response)("error").str
        assert(errorBody.contains("array"),
          clue = s"type-mismatch error should report the JSON type name; got: $errorBody")
        assert(!errorBody.contains("MARKER_PAYLOAD"),
          clue = s"type-mismatch error must NOT echo the offending value; got: $errorBody")
      }
    }
  }

  test("playing hall rejects oversize villainPool entries without echoing them in the error body") {
    // Supported archetype names are <= 16 chars; an entry over 32 chars is an
    // attacker probing the validation error path to amplify the response body
    // (the "unsupported entries: ..." branch would otherwise echo whatever
    // the client sent). Reject upfront with a generic length error instead,
    // keeping the response tiny.
    withStaticSite { staticDir =>
      // Bump maxUploadBytes for this test so the request body itself isn't
      // rejected before we exercise the villainPool length-cap branch (the
      // default-512-bytes withServer config 413s the wrapper JSON before we
      // can get there).
      withServer(staticDir, maxUploadBytes = 4096) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        // 64 chars: comfortably over the 32-char per-entry cap, comfortably
        // under the 1 KB level where this test stops being defensive.
        val tooLong = "x" * 64
        val payload =
          s"""{"hands":10,"tableCount":1,"playerCount":2,"heroStyle":"gto","heroPosition":"Button","gtoMode":"fast","villainPool":["$tooLong"],"heroExplorationRate":0,"raiseSize":2.5,"bunchingTrials":1,"equityTrials":1,"learnEveryHands":0,"learningWindowSamples":0,"seed":1}"""
        val response = postJson(s"$baseUri/api/playing-hall", payload)
        assertEquals(response.statusCode(), 400)
        val errorBody = jsonBody(response)("error").str
        assert(errorBody.contains("at most"),
          clue = s"oversize entry should surface a 'must be at most ...' message; got: $errorBody")
        assert(!errorBody.contains(tooLong),
          clue = "oversize villainPool entry must NOT be echoed in the response body")
      }
    }
  }

  test("playing hall cancellation returns 409 for terminal jobs") {
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val submissionResponse = postJson(s"$baseUri/api/playing-hall", validPlayingHallPayload)
        assertEquals(submissionResponse.statusCode(), 202)
        val statusUri = s"$baseUri${jsonBody(submissionResponse)("statusUrl").str}"

        val completed = awaitTerminalJob(statusUri)
        assertEquals(completed("status").str, "completed")

        val response = delete(statusUri)
        assertEquals(response.statusCode(), 409)
        assert(jsonBody(response)("error").str.contains("already terminal"))
      }
    }
  }

  test("playing hall submission validates the payload") {
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        val invalid = postJson(
          s"$baseUri/api/playing-hall",
          """{"hands":0,"villainPool":[],"heroStyle":"bad-mode"}"""
        )
        assertEquals(invalid.statusCode(), 400)
        assert(jsonBody(invalid)("error").str.contains("hands"))
      }
    }
  }

  test("timed-out playing hall workers keep readiness failed closed until the worker exits") {
    withStaticSite { staticDir =>
      val backend = new BusyPlayingHallBackend(runForMs = 2000L, result = Right(samplePlayingHallResult))
      withServer(staticDir, playingHallBackend = backend, maxConcurrentJobs = 1, maxQueuedJobs = 1, playingHallTimeoutMs = 100L) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        val submissionResponse = postJson(s"$baseUri/api/playing-hall", validPlayingHallPayload)
        assertEquals(submissionResponse.statusCode(), 202)
        val submission = jsonBody(submissionResponse)
        val statusUri = s"$baseUri${submission("statusUrl").str}"

        assert(backend.started.await(3, TimeUnit.SECONDS), "busy playing hall backend never started")

        val failed = awaitTerminalJob(statusUri)
        assertEquals(failed("status").str, "failed")
        assertEquals(failed("errorStatus").num.toInt, 504)

        val readyWhileWorkerRuns = get(s"$baseUri/api/ready")
        assertEquals(readyWhileWorkerRuns.statusCode(), 503)
        val readyWhileWorkerRunsJson = jsonBody(readyWhileWorkerRuns)
        assertEquals(readyWhileWorkerRunsJson("reason").str, "timed-out-worker")
        assertEquals(readyWhileWorkerRunsJson("timedOutWorkersInFlight").num.toInt, 1)

        assert(backend.finished.await(5, TimeUnit.SECONDS), "busy playing hall backend never finished")
        awaitReady(s"$baseUri/api/ready")
      }
    }
  }

  test("submit route rate limit returns 429 with retry-after without affecting probes or static content") {
    withStaticSite { staticDir =>
      val authConfig = HandHistoryReviewServer.BasicAuthConfig(username = "operator", password = "submit-limit")
      val authHeaders = basicAuthHeaders(authConfig.username, authConfig.password)
      withServer(
        staticDir,
        basicAuth = Some(authConfig),
        rateLimitSubmitsPerMinute = 1,
        rateLimitStatusPerMinute = 0
      ) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        val accepted = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload, authHeaders)
        assertEquals(accepted.statusCode(), 202)
        assertEquals(headerValue(accepted, "Retry-After"), Some("1"))

        val limited = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload, authHeaders)
        assertEquals(limited.statusCode(), 429)
        val limitedJson = jsonBody(limited)
        assert(limitedJson("error").str.contains("rate limit exceeded"))
        assertEquals(limitedJson("rateLimitBucket").str, "submit")
        assertEquals(limitedJson("limitPerMinute").num.toInt, 1)
        assertEquals(headerValue(limited, "Retry-After"), Some(limitedJson("retryAfterSeconds").num.toInt.toString))

        assertEquals(get(s"$baseUri/api/health").statusCode(), 200)
        assertEquals(get(s"$baseUri/api/ready").statusCode(), 200)
        assertEquals(get(s"$baseUri/", authHeaders).statusCode(), 200)
      }
    }
  }

  test("job-status route has its own rate limit bucket") {
    withStaticSite { staticDir =>
      val authConfig = HandHistoryReviewServer.BasicAuthConfig(username = "operator", password = "status-limit")
      val authHeaders = basicAuthHeaders(authConfig.username, authConfig.password)
      val backend = new BlockingBackend(Right(sampleAnalysisResult))
      withServer(
        staticDir,
        backend = backend,
        basicAuth = Some(authConfig),
        rateLimitSubmitsPerMinute = 0,
        rateLimitStatusPerMinute = 1
      ) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        val submission = jsonBody(postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload, authHeaders))
        val statusUri = s"$baseUri${submission("statusUrl").str}"

        val firstStatus = get(statusUri, authHeaders)
        assertEquals(firstStatus.statusCode(), 200)
        assertEquals(headerValue(firstStatus, "Retry-After"), Some("1"))

        val limitedStatus = get(statusUri, authHeaders)
        assertEquals(limitedStatus.statusCode(), 429)
        val limitedJson = jsonBody(limitedStatus)
        assertEquals(limitedJson("rateLimitBucket").str, "job-status")
        assertEquals(limitedJson("limitPerMinute").num.toInt, 1)
        assertEquals(headerValue(limitedStatus, "Retry-After"), Some(limitedJson("retryAfterSeconds").num.toInt.toString))

        backend.release.countDown()
      }
    }
  }

  test("auth route rate limit returns 429 with retry-after to throttle credential stuffing") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath)),
          rateLimitSubmitsPerMinute = 0,
          rateLimitStatusPerMinute = 0,
          rateLimitAuthPerMinute = 1
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          val firstLogin = postJson(s"$baseUri/api/auth/login",
            """{"email":"victim@example.com","password":"any-wrong-password"}""")
          assertEquals(firstLogin.statusCode(), 401)

          val limited = postJson(s"$baseUri/api/auth/login",
            """{"email":"victim@example.com","password":"another-wrong-password"}""")
          assertEquals(limited.statusCode(), 429)
          val limitedJson = jsonBody(limited)
          assert(limitedJson("error").str.contains("rate limit exceeded"),
            s"expected rate-limit-exceeded error, got: ${limitedJson("error").str}")
          assertEquals(limitedJson("rateLimitBucket").str, "auth")
          assertEquals(limitedJson("limitPerMinute").num.toInt, 1)
          assertEquals(
            headerValue(limited, "Retry-After"),
            Some(limitedJson("retryAfterSeconds").num.toInt.toString)
          )

          // Register is in the same auth bucket so attackers can't bypass the
          // login throttle by hammering registration with PBKDF2-cost requests.
          val limitedRegister = postJson(s"$baseUri/api/auth/register",
            """{"email":"new@example.com","password":"correct-horse-battery","displayName":"New"}""")
          assertEquals(limitedRegister.statusCode(), 429)

          // Probes and static content stay reachable.
          assertEquals(get(s"$baseUri/api/health").statusCode(), 200)
          assertEquals(get(s"$baseUri/api/ready").statusCode(), 200)
          assertEquals(get(s"$baseUri/").statusCode(), 200)
        }
      }
    }
  }

  test("zero rate limits disable submit throttling") {
    withStaticSite { staticDir =>
      val authConfig = HandHistoryReviewServer.BasicAuthConfig(username = "operator", password = "no-rate-limit")
      val authHeaders = basicAuthHeaders(authConfig.username, authConfig.password)
      withServer(
        staticDir,
        basicAuth = Some(authConfig),
        rateLimitSubmitsPerMinute = 0,
        rateLimitStatusPerMinute = 0
      ) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        val first = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload, authHeaders)
        val second = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload, authHeaders)
        val third = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload, authHeaders)

        assertEquals(first.statusCode(), 202)
        assertEquals(second.statusCode(), 202)
        assertEquals(third.statusCode(), 202)
      }
    }
  }

  test("configured trusted client IP header isolates rate limits for proxied clients") {
    withStaticSite { staticDir =>
      withServer(
        staticDir,
        rateLimitSubmitsPerMinute = 1,
        rateLimitStatusPerMinute = 0,
        rateLimitClientIpHeader = Some("X-Real-IP")
      ) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val firstClientHeaders = Map("X-Real-IP" -> "203.0.113.10")
        val secondClientHeaders = Map("X-Real-IP" -> "198.51.100.7")

        val acceptedFirstClient = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload, firstClientHeaders)
        assertEquals(acceptedFirstClient.statusCode(), 202)

        val limitedFirstClient = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload, firstClientHeaders)
        assertEquals(limitedFirstClient.statusCode(), 429)

        val acceptedSecondClient = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload, secondClientHeaders)
        assertEquals(acceptedSecondClient.statusCode(), 202)

        val health = getJson(s"$baseUri/api/health")
        assertEquals(health("rateLimitClientIpSource").str, "header:X-Real-IP via loopback-only")

        val ready = getJson(s"$baseUri/api/ready")
        assertEquals(ready("rateLimitClientIpSource").str, "header:X-Real-IP via loopback-only")
      }
    }
  }

  test("health and readiness report conditional header trust when an allowlist is configured") {
    withStaticSite { staticDir =>
      withServer(
        staticDir,
        rateLimitSubmitsPerMinute = 1,
        rateLimitStatusPerMinute = 0,
        rateLimitClientIpHeader = Some("X-Real-IP"),
        rateLimitTrustedProxyIps = Set("203.0.113.10")
      ) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        val health = getJson(s"$baseUri/api/health")
        assertEquals(health("rateLimitClientIpSource").str, "header:X-Real-IP via loopback-or-allowlisted-proxy")

        val ready = getJson(s"$baseUri/api/ready")
        assertEquals(ready("rateLimitClientIpSource").str, "header:X-Real-IP via loopback-or-allowlisted-proxy")
      }
    }
  }

  test("ambiguous trusted client IP header values fall back to the remote address bucket") {
    withStaticSite { staticDir =>
      withServer(
        staticDir,
        rateLimitSubmitsPerMinute = 1,
        rateLimitStatusPerMinute = 0,
        rateLimitClientIpHeader = Some("X-Forwarded-For")
      ) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val firstClientHeaders = Map("X-Forwarded-For" -> "203.0.113.10, 198.51.100.20")
        val secondClientHeaders = Map("X-Forwarded-For" -> "198.51.100.7, 198.51.100.21")

        val accepted = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload, firstClientHeaders)
        assertEquals(accepted.statusCode(), 202)

        val limited = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload, secondClientHeaders)
        assertEquals(limited.statusCode(), 429)
      }
    }
  }

  test("invalid trusted client IP header values fall back to the remote address bucket") {
    withStaticSite { staticDir =>
      withServer(
        staticDir,
        rateLimitSubmitsPerMinute = 1,
        rateLimitStatusPerMinute = 0,
        rateLimitClientIpHeader = Some("X-Real-IP")
      ) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val firstClientHeaders = Map("X-Real-IP" -> "not-an-ip")
        val secondClientHeaders = Map("X-Real-IP" -> "still-not-an-ip")

        val accepted = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload, firstClientHeaders)
        assertEquals(accepted.statusCode(), 202)

        val limited = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload, secondClientHeaders)
        assertEquals(limited.statusCode(), 429)
      }
    }
  }

  test("trusted client IP header is only trusted for loopback or allowlisted proxies") {
    assert(RateLimit.trustsRateLimitClientIpHeader(Some(InetAddress.getByName("127.0.0.1")), Set.empty))
    assert(RateLimit.trustsRateLimitClientIpHeader(Some(InetAddress.getByName("::1")), Set.empty))
    assert(!RateLimit.trustsRateLimitClientIpHeader(Some(InetAddress.getByName("203.0.113.10")), Set.empty))
    assert(RateLimit.trustsRateLimitClientIpHeader(
      Some(InetAddress.getByName("203.0.113.10")),
      Set("203.0.113.10")
    ))
    assert(!RateLimit.trustsRateLimitClientIpHeader(None, Set("203.0.113.10")))
  }

  test("auth.*.failure caps the submitted email before logging so an oversize submission cannot flood the audit log") {
    // The auth-endpoint body cap is 16 KB. Before this guard, a request like
    // {"email":"<15KB>","password":"x"} would land a 15 KB entry in the audit
    // log per failed register/login attempt. Attacker mints disk pressure
    // and noise per request. Cap the SUBMITTED email at 320 chars in the
    // log line (the SUCCESS path uses the canonical email which is already
    // bounded by validateEmail's 254-char cap, so this only affects the
    // failure path).
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"
          val errBuf = new java.io.ByteArrayOutputStream()
          val originalErr = System.err
          System.setErr(new java.io.PrintStream(errBuf, true, StandardCharsets.UTF_8))
          try
            // 10 KB email -> validateEmail rejects, but the submitted value
            // is what the failure log line records. The log line itself must
            // stay small.
            val hugeEmail = ("a" * 10000) + "@example.com"
            val payload = s"""{"email":"$hugeEmail","password":"correct-horse-battery","displayName":"Big"}"""
            val response = postJson(s"$baseUri/api/auth/register", payload)
            assertEquals(response.statusCode(), 400)
          finally
            System.setErr(originalErr)
          val captured = errBuf.toString(StandardCharsets.UTF_8)
          assert(captured.contains("auth.register.failure"),
            clue = s"expected auth.register.failure WARN in stderr; got: ${captured.take(200)}")
          // The full log line, INCLUDING timestamp / level / prefix / email
          // field / remote field / reason field, must stay well under 1 KB.
          // If the email was logged untruncated, this line would be >10 KB.
          val failureLine = captured.split('\n').iterator
            .find(_.contains("auth.register.failure"))
            .getOrElse(fail(s"no auth.register.failure line in stderr capture"))
          assert(failureLine.length < 1024,
            clue = s"auth.register.failure log line is ${failureLine.length} bytes; cap should keep it well under 1 KB")
          assert(failureLine.contains("...(truncated)"),
            clue = s"capped email should include the truncation marker; got: ${failureLine.take(200)}")
        }
      }
    }
  }

  test("audit log remote= shows the trusted-header client IP behind a reverse proxy") {
    // Behind a trusted reverse proxy, audit logs that previously read
    // `remote=<loopback>:<port>` (the proxy's TCP peer) would be useless for
    // forensics — every request looks the same. The same trusted-header policy
    // the rate limiter applies (peer is loopback or in trustedProxyIps; header
    // parses as a single IP) must also flow into the audit-log `remote=` field
    // so operators can see who is actually behind the proxy.
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath)),
          rateLimitClientIpHeader = Some("X-Real-IP")
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"
          val errBuf = new java.io.ByteArrayOutputStream()
          val originalErr = System.err
          System.setErr(new java.io.PrintStream(errBuf, true, StandardCharsets.UTF_8))
          try
            val response = postJson(
              s"$baseUri/api/auth/login",
              """{"email":"nobody@example.com","password":"wrong-password"}""",
              Map("X-Real-IP" -> "203.0.113.5")
            )
            assertEquals(response.statusCode(), 401)
          finally
            System.setErr(originalErr)
          val captured = errBuf.toString(StandardCharsets.UTF_8)
          assert(captured.contains("auth.login.failure"),
            clue = s"expected auth.login.failure WARN in stderr; got: $captured")
          assert(captured.contains("remote=203.0.113.5"),
            clue = s"expected audit log to surface the resolved client IP; got: $captured")
          assert(!captured.contains("remote=127.0.0.1") && !captured.contains("remote=[::1]"),
            clue = s"audit log must not show the loopback peer when behind a trusted proxy; got: $captured")
        }
      }
    }
  }

  test("audit log remote= shows the direct peer when no trusted header is configured") {
    // Default deployment (no proxy): `remote=` is the TCP peer in `host:port`
    // form with IPv6 brackets per RFC 3986. A spoofed X-Real-IP header from a
    // non-trusted client must be IGNORED, not blindly echoed into the audit log.
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
          // intentionally no rateLimitClientIpHeader -- audit must NOT trust client headers
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"
          val errBuf = new java.io.ByteArrayOutputStream()
          val originalErr = System.err
          System.setErr(new java.io.PrintStream(errBuf, true, StandardCharsets.UTF_8))
          try
            val response = postJson(
              s"$baseUri/api/auth/login",
              """{"email":"nobody@example.com","password":"wrong-password"}""",
              Map("X-Real-IP" -> "203.0.113.5")
            )
            assertEquals(response.statusCode(), 401)
          finally
            System.setErr(originalErr)
          val captured = errBuf.toString(StandardCharsets.UTF_8)
          assert(captured.contains("auth.login.failure"),
            clue = s"expected auth.login.failure WARN in stderr; got: $captured")
          assert(!captured.contains("remote=203.0.113.5"),
            clue = s"audit log must NOT trust a client-supplied X-Real-IP when no header is configured; got: $captured")
          assert(captured.contains("remote=127.0.0.1:") || captured.contains("remote=[::1]:"),
            clue = s"audit log must show the loopback TCP peer; got: $captured")
        }
      }
    }
  }

  test("invalid trusted proxy IP allowlist fails startup parsing") {
    withStaticSite { staticDir =>
      val startResult = HandHistoryReviewServer.start(Array(
        s"--staticDir=$staticDir",
        "--port=0",
        "--rateLimitTrustedProxyIps=not-an-ip"
      ))

      assert(startResult.isLeft)
      assert(startResult.left.toOption.get.contains("--rateLimitTrustedProxyIps"))
    }
  }

  test("analysis submission rejects overload once the bounded queue is full") {
    withStaticSite { staticDir =>
      val backend = new BlockingBackend(Right(sampleAnalysisResult))
      withServer(staticDir, backend = backend, maxConcurrentJobs = 1, maxQueuedJobs = 1) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        val firstSubmission = jsonBody(postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload))
        assertEquals(firstSubmission("status").str, "queued")
        assert(backend.started.await(3, TimeUnit.SECONDS), "first analysis backend never started")

        val secondSubmission = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload)
        assertEquals(secondSubmission.statusCode(), 202)
        val secondStatusUrl = s"$baseUri${jsonBody(secondSubmission)("statusUrl").str}"

        val rejected = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload)
        assertEquals(rejected.statusCode(), 503)
        assert(rejected.body().contains("queue is full"))

        val ready = get(s"$baseUri/api/ready")
        assertEquals(ready.statusCode(), 503)
        val readyJson = jsonBody(ready)
        assertEquals(readyJson("ready").bool, false)
        assertEquals(readyJson("reason").str, "queue-full")
        assertEquals(readyJson("draining").bool, false)
        assertEquals(readyJson("acceptingAnalysisJobs").bool, false)

        val health = getJson(s"$baseUri/api/health")
        assertEquals(health("ready").bool, false)
        assertEquals(health("readyReason").str, "queue-full")
        assertEquals(health("runningJobs").num.toInt, 1)
        assertEquals(health("queuedJobs").num.toInt, 1)

        backend.release.countDown()
        val completedSecond = awaitTerminalJob(secondStatusUrl)
        assertEquals(completedSecond("status").str, "completed")
      }
    }
  }

  test("analysis polling surfaces terminal failures") {
    withStaticSite { staticDir =>
      val backend = new BlockingBackend(Left("analysis failed: synthetic test failure"))
      withServer(staticDir, backend = backend) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        val submissionResponse = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload)
        assertEquals(submissionResponse.statusCode(), 202)
        val submission = jsonBody(submissionResponse)
        val statusUri = s"$baseUri${submission("statusUrl").str}"

        assert(backend.started.await(3, TimeUnit.SECONDS), "analysis backend never started")
        backend.release.countDown()

        val failed = awaitTerminalJob(statusUri)
        assertEquals(failed("status").str, "failed")
        assertEquals(failed("errorStatus").num.toInt, 500)
        assert(failed("error").str.contains("synthetic test failure"))
      }
    }
  }

  test("analysis timeout fails stuck jobs with 504 and frees worker capacity") {
    withStaticSite { staticDir =>
      val backend = new BlockingBackend(Right(sampleAnalysisResult))
      withServer(staticDir, backend = backend, maxConcurrentJobs = 1, maxQueuedJobs = 1, analysisTimeoutMs = 100L) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        val submissionResponse = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload)
        assertEquals(submissionResponse.statusCode(), 202)
        val submission = jsonBody(submissionResponse)
        val statusUri = s"$baseUri${submission("statusUrl").str}"

        assert(backend.started.await(3, TimeUnit.SECONDS), "analysis backend never started")

        val failed = awaitTerminalJob(statusUri)
        assertEquals(failed("status").str, "failed")
        assertEquals(failed("errorStatus").num.toInt, 504)
        assert(failed("error").str.contains("timed out"))

        awaitReady(s"$baseUri/api/ready")
      }
    }
  }

  test("timed-out workers keep readiness failed closed until the worker exits") {
    withStaticSite { staticDir =>
      val backend = new BusyBackend(runForMs = 2000L, result = Right(sampleAnalysisResult))
      withServer(staticDir, backend = backend, maxConcurrentJobs = 1, maxQueuedJobs = 1, analysisTimeoutMs = 100L) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        val submissionResponse = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload)
        assertEquals(submissionResponse.statusCode(), 202)
        val submission = jsonBody(submissionResponse)
        val statusUri = s"$baseUri${submission("statusUrl").str}"

        assert(backend.started.await(3, TimeUnit.SECONDS), "busy backend never started")

        val failed = awaitTerminalJob(statusUri)
        assertEquals(failed("status").str, "failed")
        assertEquals(failed("errorStatus").num.toInt, 504)

        val readyWhileWorkerRuns = get(s"$baseUri/api/ready")
        assertEquals(readyWhileWorkerRuns.statusCode(), 503)
        val readyWhileWorkerRunsJson = jsonBody(readyWhileWorkerRuns)
        assertEquals(readyWhileWorkerRunsJson("reason").str, "timed-out-worker")
        assertEquals(readyWhileWorkerRunsJson("timedOutWorkersInFlight").num.toInt, 1)

        val rejected = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload)
        assertEquals(rejected.statusCode(), 503)
        assert(rejected.body().contains("waiting for recovery"))

        assert(backend.finished.await(5, TimeUnit.SECONDS), "busy backend never finished")
        awaitReady(s"$baseUri/api/ready")

        val recoveredHealth = getJson(s"$baseUri/api/health")
        assertEquals(recoveredHealth("timedOutWorkersInFlight").num.toInt, 0)
      }
    }
  }

  test("async review flow accepts a reproducible exact-gto hall export".tag(munit.Slow)) {
    val root = Files.createTempDirectory("hand-history-review-server-proof-")
    try
      val uploadText = generateProofUpload(root.resolve("hall-proof-out"))
      withStaticSite { staticDir =>
        val server = HandHistoryReviewServer.start(Array(
          "--host=127.0.0.1",
          "--port=0",
          s"--staticDir=$staticDir",
          "--seed=37",
          "--bunchingTrials=8",
          "--equityTrials=240",
          "--budgetMs=150",
          "--maxDecisions=16"
        )).fold(err => fail(err), identity)
        try
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"
          val submissionResponse = postJson(
            s"$baseUri/api/analyze-hand-history",
            ujson.write(
              ujson.Obj(
                "handHistoryText" -> ujson.Str(uploadText),
                "site" -> ujson.Str("PokerStars"),
                "heroName" -> ujson.Str("Hero")
              )
            )
          )
          assertEquals(submissionResponse.statusCode(), 202)

          val submission = jsonBody(submissionResponse)
          val completed = awaitTerminalJob(s"$baseUri${submission("statusUrl").str}")

          assertEquals(completed("status").str, "completed")
          assertEquals(completed("result")("site").str, "PokerStars")
          assertEquals(completed("result")("heroName").str, "Hero")
          assertEquals(completed("result")("handsImported").num.toInt, 12)
          assert(completed("result")("handsAnalyzed").num.toInt > 0)
          assert(completed("result")("decisionsAnalyzed").num.toInt > 0)
          assertEquals(completed("result")("warnings").arr.toVector, Vector.empty)
          assert(completed("result")("opponents").arr.size >= 2)
          assertEquals(completed("result")("trace")("request")("normalizedHeroName").str, "Hero")
          assertEquals(completed("result")("trace")("import")("handsImported").num.toInt, 12)
          assertEquals(completed("result")("trace")("import")("siteResolved").str, "PokerStars")
          assertEquals(completed("result")("trace")("import")("heroNameResolved").str, "Hero")
          assertEquals(completed("result")("trace")("hands").arr.size, 12)
          assert(completed("result")("trace")("hands").arr.forall(_("status").str == "analyzed"))
          assertEquals(
            completed("result")("trace")("summary")("handsAnalyzed").num.toInt,
            completed("result")("handsAnalyzed").num.toInt
          )
          assertEquals(
            completed("result")("trace")("summary")("decisionsAnalyzed").num.toInt,
            completed("result")("decisionsAnalyzed").num.toInt
          )
        finally
          server.close()
      }
    finally
      deleteRecursively(root)
  }

  test("start returns a bind error that names the unavailable address") {
    withStaticSite { staticDir =>
      withServer(staticDir) { running =>
        val port = running.binding.port
        val secondStart = HandHistoryReviewServer.startWithBackend(
          HandHistoryReviewServer.ServerConfig(
            host = "127.0.0.1",
            port = port,
            staticDir = staticDir,
            maxUploadBytes = 512,
            analysisTimeoutMs = 120000L,
            playingHallTimeoutMs = 900000L,
            maxConcurrentJobs = 2,
            maxQueuedJobs = 8,
            shutdownGraceMs = 5000L,
            rateLimitSubmitsPerMinute = 6,
            rateLimitStatusPerMinute = 240,
            rateLimitAuthPerMinute = 10,
            rateLimitClientIpHeader = None,
            rateLimitTrustedProxyIps = Set.empty,
            drainSignalFile = None,
            basicAuth = None,
            serviceConfig = HandHistoryReviewService.ServiceConfig()
          ),
          immediateBackend(Right(sampleAnalysisResult))
        )

        assert(secondStart.isLeft)
        assert(secondStart.left.toOption.get.contains(s"127.0.0.1:$port is unavailable"))
      }
    }
  }

  test("start defaults direct launches to localhost") {
    withStaticSite { staticDir =>
      val server = HandHistoryReviewServer.start(Array(
        s"--staticDir=$staticDir",
        "--port=0"
      )).fold(err => fail(err), identity)
      try
        assertEquals(server.binding.host, "127.0.0.1")
        assertEquals(get(s"http://${server.binding.host}:${server.binding.port}/api/health").statusCode(), 200)
      finally
        server.close()
    }
  }

  test("start rejects unauthenticated non-loopback binds unless explicitly allowed") {
    withStaticSite { staticDir =>
      val result = HandHistoryReviewServer.start(Array(
        s"--staticDir=$staticDir",
        "--host=0.0.0.0",
        "--port=0"
      ))
      assert(result.isLeft)
      val error = result.left.toOption.getOrElse("")
      assert(error.contains("ALLOW_UNAUTHENTICATED_PUBLIC_BIND"))
      assert(error.contains("BASIC_AUTH_*/USER_STORE_PATH"))
    }
  }

  test("start allows unauthenticated non-loopback binds when explicitly overridden") {
    withStaticSite { staticDir =>
      val server = HandHistoryReviewServer.start(Array(
        s"--staticDir=$staticDir",
        "--host=0.0.0.0",
        "--port=0",
        "--allowUnauthenticatedPublicBind=true"
      )).fold(err => fail(err), identity)
      try
        assertEquals(server.binding.host, "0.0.0.0")
        assertEquals(get(s"http://127.0.0.1:${server.binding.port}/api/health").statusCode(), 200)
      finally
        server.close()
    }
  }

  test("non-loopback platform-user auth requires secure cookies unless explicitly allowed") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        val result = HandHistoryReviewServer.start(Array(
          s"--staticDir=$staticDir",
          "--host=0.0.0.0",
          "--port=0",
          s"--userStorePath=$storePath"
        ))
        assert(result.isLeft)
        val error = result.left.toOption.getOrElse("")
        assert(error.contains("USER_AUTH_COOKIE_SECURE"))
        assert(error.contains("ALLOW_INSECURE_USER_AUTH"))
      }
    }
  }

  test("non-loopback OIDC requires an HTTPS redirect URI unless explicitly allowed") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        val result = HandHistoryReviewServer.start(Array(
          s"--staticDir=$staticDir",
          "--host=0.0.0.0",
          "--port=0",
          s"--userStorePath=$storePath",
          "--userAuthCookieSecure=true",
          "--googleOidcClientId=test-client.apps.googleusercontent.com",
          "--googleOidcClientSecret=test-secret",
          "--googleOidcRedirectUri=http://review.example.com/api/auth/oidc/google/callback"
        ))
        assert(result.isLeft)
        val error = result.left.toOption.getOrElse("")
        assert(error.contains("https://"))
        assert(error.contains("ALLOW_INSECURE_USER_AUTH"))
      }
    }
  }

  test("Google OIDC redirect URI must match the registered callback path") {
    // The runtime registers exactly one context per OIDC provider at
    // /api/auth/oidc/<id>/callback. If the operator configures
    // GOOGLE_OIDC_REDIRECT_URI to a different path, Google would redirect the
    // user back to that path -- the server has no handler for it, so the
    // catch-all static handler returns a generic 404 with no breadcrumb that
    // OIDC was the failing flow. Catch the mismatch at config parse time
    // instead so the operator sees an actionable error.
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        val result = HandHistoryReviewServer.start(Array(
          s"--staticDir=$staticDir",
          "--host=127.0.0.1",
          "--port=0",
          s"--userStorePath=$storePath",
          "--googleOidcClientId=test-client.apps.googleusercontent.com",
          "--googleOidcClientSecret=test-secret",
          "--googleOidcRedirectUri=http://127.0.0.1:8080/oauth/wrong-callback-path"
        ))
        assert(result.isLeft, s"expected config error, got: $result")
        val error = result.left.toOption.getOrElse("")
        assert(error.contains("/api/auth/oidc/google/callback"),
          s"error should name the required callback path; got: $error")
        assert(error.contains("/oauth/wrong-callback-path"),
          s"error should echo the operator-supplied path so they can see what was wrong; got: $error")
      }
    }
  }

  test("localhost user auth still allows insecure cookies and HTTP OIDC callback for local development") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        val server = HandHistoryReviewServer.start(Array(
          s"--staticDir=$staticDir",
          "--host=127.0.0.1",
          "--port=0",
          s"--userStorePath=$storePath",
          "--googleOidcClientId=test-client.apps.googleusercontent.com",
          "--googleOidcClientSecret=test-secret",
          "--googleOidcRedirectUri=http://127.0.0.1:8080/api/auth/oidc/google/callback"
        )).fold(err => fail(err), identity)
        try
          assertEquals(get(s"http://${server.binding.host}:${server.binding.port}/api/health").statusCode(), 200)
        finally
          server.close()
      }
    }
  }

  test("PlatformUserAuth.Service.create rejects OIDC providers that share an id with the local password sign-in") {
    // The local-password sign-in path is registered under id "local" in the
    // /api/auth/me providers list. An OIDC provider with id="local" would
    // either shadow it in the UI or, worse, route /api/auth/oidc/local/start
    // to a real OIDC flow disguised as the local-password tab. Reject at
    // service-create time so the operator sees the misconfiguration during
    // startup rather than at first user click.
    withUserStorePath { storePath =>
      val collidingProvider = new PlatformUserAuth.OidcProvider:
        override val id = "local"
        override val displayName = "Bad collision"
        override def authorizationUri(state: String, codeChallenge: String): String = "https://nowhere.example/"
        override def exchangeCode(code: String, codeVerifier: String): Either[String, PlatformUserAuth.OidcIdentity] =
          Left("unreached")
      val result = PlatformUserAuth.Service.create(PlatformUserAuth.Config(
        storePath = storePath,
        oidcProviders = Vector(collidingProvider)
      ))
      assert(result.isLeft, s"expected service creation to fail, got: $result")
      val error = result.left.toOption.getOrElse(fail("missing error"))
      assert(error.contains("local") && error.contains("reserved"),
        s"error should mention the reserved 'local' id; got: $error")
    }
  }

  test("PlatformUserAuth.Service.create rejects OIDC provider ids with invalid shape") {
    // Provider id ends up in URL paths and as a JDK HttpServer context key.
    // Empty, slash-bearing, or non-URL-safe ids would either fail context
    // registration with an opaque JDK error or register routes that
    // extractOidcProviderId's 5-segment match couldn't reach. Reject at
    // service-create with a clear shape-rule message so the operator
    // fixes the config at startup, not at first user click.
    withUserStorePath { storePath =>
      def make(providerId: String) = new PlatformUserAuth.OidcProvider:
        override val id = providerId
        override val displayName = "Bad shape"
        override def authorizationUri(state: String, codeChallenge: String): String = "https://nowhere.example/"
        override def exchangeCode(code: String, codeVerifier: String): Either[String, PlatformUserAuth.OidcIdentity] =
          Left("unreached")
      for badId <- Vector("", "has/slash", "has\\backslash", "has space", "has?question", "has#hash", "ñoñascii", "a" * 65) do
        val result = PlatformUserAuth.Service.create(PlatformUserAuth.Config(
          storePath = storePath,
          oidcProviders = Vector(make(badId))
        ))
        assert(result.isLeft, s"expected service creation to fail for id='$badId', got: $result")
        val error = result.left.toOption.getOrElse(fail(s"missing error for id='$badId'"))
        assert(error.toLowerCase.contains("invalid") || error.toLowerCase.contains("contain only") || error.toLowerCase.contains("non-empty"),
          s"error should mention the shape rule for id='$badId'; got: $error")
    }
  }

  test("PlatformUserAuth.Service.create rejects duplicate OIDC provider ids") {
    // Two OIDC providers with the same id would silently collapse to one in
    // providersById.toMap AND then crash HTTP server startup with an opaque
    // "context already exists" exception when both tried to register the same
    // /api/auth/oidc/<id>/start path. Reject at service-create with a clear
    // message so the operator can fix the config without grepping stack traces.
    withUserStorePath { storePath =>
      def make(providerId: String, displayName: String) = new PlatformUserAuth.OidcProvider:
        override val id = providerId
        override val displayName = displayName
        override def authorizationUri(state: String, codeChallenge: String): String =
          s"https://nowhere.example/?state=$state"
        override def exchangeCode(code: String, codeVerifier: String): Either[String, PlatformUserAuth.OidcIdentity] =
          Left("unreached")
      val result = PlatformUserAuth.Service.create(PlatformUserAuth.Config(
        storePath = storePath,
        oidcProviders = Vector(make("google", "Google A"), make("google", "Google B"))
      ))
      assert(result.isLeft, s"expected service creation to fail, got: $result")
      val error = result.left.toOption.getOrElse(fail("missing error"))
      assert(error.contains("google") && error.toLowerCase.contains("duplicate"),
        s"error should mention the duplicate provider id 'google'; got: $error")
    }
  }

  test("non-loopback platform-user auth can be explicitly allowed for private-network testing") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        val server = HandHistoryReviewServer.start(Array(
          s"--staticDir=$staticDir",
          "--host=0.0.0.0",
          "--port=0",
          s"--userStorePath=$storePath",
          "--allowInsecureUserAuth=true"
        )).fold(err => fail(err), identity)
        try
          assertEquals(get(s"http://127.0.0.1:${server.binding.port}/api/health").statusCode(), 200)
        finally
          server.close()
      }
    }
  }

  private def withStaticSite[A](run: Path => A): A =
    val tempDir = Files.createTempDirectory("hand-history-review-server-")
    try
      Files.writeString(
        tempDir.resolve("index.html"),
        "<!doctype html><html><body><h1>Runtime smoke page</h1></body></html>",
        StandardCharsets.UTF_8
      )
      run(tempDir)
    finally
      deleteRecursively(tempDir)

  private def withServer[A](
      staticDir: Path,
      maxUploadBytes: Int = 512,
      backend: HandHistoryReviewServer.AnalysisBackend = immediateBackend(Right(sampleAnalysisResult)),
      playingHallBackend: HandHistoryReviewServer.PlayingHallBackend =
        immediatePlayingHallBackend(Right(samplePlayingHallResult)),
      maxConcurrentJobs: Int = 2,
      maxQueuedJobs: Int = 8,
      analysisTimeoutMs: Long = 120000L,
      playingHallTimeoutMs: Long = 900000L,
      shutdownGraceMs: Long = 5000L,
      rateLimitSubmitsPerMinute: Int = 6,
      rateLimitStatusPerMinute: Int = 240,
      rateLimitAuthPerMinute: Int = 10,
      rateLimitClientIpHeader: Option[String] = None,
      rateLimitTrustedProxyIps: Set[String] = Set.empty,
      drainSignalFile: Option[Path] = None,
      basicAuth: Option[HandHistoryReviewServer.BasicAuthConfig] = None,
      platformAuth: Option[PlatformUserAuth.Config] = None
  )(run: HandHistoryReviewServer.RunningServer => A): A =
    val server = HandHistoryReviewServer.startWithBackends(
      HandHistoryReviewServer.ServerConfig(
        host = "127.0.0.1",
        port = 0,
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
        serviceConfig = HandHistoryReviewService.ServiceConfig(),
        platformAuth = platformAuth
      ),
      backend,
      playingHallBackend
    ).fold(err => fail(err), identity)
    try run(server)
    finally server.close()

  private def jsonBody(response: HttpResponse[String]): Value =
    ujson.read(response.body())

  private def headerValue(response: HttpResponse[String], name: String): Option[String] =
    Option(response.headers().firstValue(name).orElse(null))

  private def getJson(uri: String): Value =
    jsonBody(get(uri))

  private def getJsonWithHeaders(uri: String, headers: Map[String, String]): Value =
    jsonBody(get(uri, headers))

  private def awaitReady(uri: String): Unit =
    val deadlineNanos = System.nanoTime() + TimeUnit.SECONDS.toNanos(5)
    var lastStatus = -1
    while System.nanoTime() < deadlineNanos do
      val response = get(uri)
      lastStatus = response.statusCode()
      if lastStatus == 200 then
        return
      Thread.sleep(50)
    fail(s"readiness did not recover, last status=$lastStatus")

  private def awaitTerminalJob(uri: String, headers: Map[String, String] = Map.empty): Value =
    val deadlineNanos = System.nanoTime() + TimeUnit.SECONDS.toNanos(5)
    var lastStatus = "<none>"
    while System.nanoTime() < deadlineNanos do
      val body = jsonBody(get(uri, headers))
      lastStatus = body("status").str
      if lastStatus == "completed" || lastStatus == "failed" || lastStatus == "cancelled" then
        return body
      Thread.sleep(50)
    fail(s"job did not reach a terminal state, last status=$lastStatus")

  private def immediateBackend(result: Either[String, Value]): HandHistoryReviewServer.AnalysisBackend =
    new HandHistoryReviewServer.AnalysisBackend:
      override def analyze(request: HandHistoryReviewService.AnalysisRequest): Either[String, Value] =
        result

  private def immediatePlayingHallBackend(
      result: Either[String, Value]
  ): HandHistoryReviewServer.PlayingHallBackend =
    new HandHistoryReviewServer.PlayingHallBackend:
      override def run(request: HandHistoryReviewServer.PlayingHallRequest, cancelSignal: () => Boolean): Either[String, Value] =
        result

  private final class BlockingBackend(result: Either[String, Value]) extends HandHistoryReviewServer.AnalysisBackend:
    val started = new CountDownLatch(1)
    val release = new CountDownLatch(1)

    override def analyze(request: HandHistoryReviewService.AnalysisRequest): Either[String, Value] =
      started.countDown()
      if !release.await(5, TimeUnit.SECONDS) then
        Left("analysis failed: blocking backend timed out")
      else result

  private final class BlockingPlayingHallBackend(result: Either[String, Value]) extends HandHistoryReviewServer.PlayingHallBackend:
    val started = new CountDownLatch(1)
    val release = new CountDownLatch(1)

    override def run(request: HandHistoryReviewServer.PlayingHallRequest, cancelSignal: () => Boolean): Either[String, Value] =
      started.countDown()
      if !release.await(5, TimeUnit.SECONDS) then
        Left("playing hall failed: blocking backend timed out")
      else result

  private final class BusyBackend(runForMs: Long, result: Either[String, Value]) extends HandHistoryReviewServer.AnalysisBackend:
    val started = new CountDownLatch(1)
    val finished = new CountDownLatch(1)

    override def analyze(request: HandHistoryReviewService.AnalysisRequest): Either[String, Value] =
      started.countDown()
      try
        val deadlineNanos = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(runForMs)
        while System.nanoTime() < deadlineNanos do
          Thread.interrupted()
          Thread.onSpinWait()
        result
      finally
        finished.countDown()

  private final class BusyPlayingHallBackend(runForMs: Long, result: Either[String, Value]) extends HandHistoryReviewServer.PlayingHallBackend:
    val started = new CountDownLatch(1)
    val finished = new CountDownLatch(1)

    override def run(request: HandHistoryReviewServer.PlayingHallRequest, cancelSignal: () => Boolean): Either[String, Value] =
      started.countDown()
      try
        val deadlineNanos = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(runForMs)
        while System.nanoTime() < deadlineNanos do
          Thread.interrupted()
          Thread.onSpinWait()
        result
      finally
        finished.countDown()

  private def get(uri: String, headers: Map[String, String] = Map.empty): HttpResponse[String] =
    val builder = HttpRequest.newBuilder(URI.create(uri))
    headers.foreach { case (name, value) => builder.header(name, value) }
    val request = builder.GET().build()
    httpClient.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8))

  private def postJson(
      uri: String,
      body: String,
      headers: Map[String, String] = Map.empty
  ): HttpResponse[String] =
    val builder = HttpRequest.newBuilder(URI.create(uri))
      .header("Content-Type", "application/json")
    headers.foreach { case (name, value) => builder.header(name, value) }
    val request = builder
      .POST(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8))
      .build()
    httpClient.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8))

  private def delete(uri: String, headers: Map[String, String] = Map.empty): HttpResponse[String] =
    val builder = HttpRequest.newBuilder(URI.create(uri))
    headers.foreach { case (name, value) => builder.header(name, value) }
    val request = builder.DELETE().build()
    httpClient.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8))

  private def basicAuthHeaders(username: String, password: String): Map[String, String] =
    val token = Base64.getEncoder.encodeToString(s"$username:$password".getBytes(StandardCharsets.UTF_8))
    Map("Authorization" -> s"Basic $token")

  private def sessionCookie(response: HttpResponse[String]): String =
    headerValue(response, "Set-Cookie")
      .map(_.takeWhile(_ != ';'))
      .getOrElse(fail("missing session cookie"))

  private def authSessionHeaders(response: HttpResponse[String], csrfToken: String): Map[String, String] =
    Map(
      "Cookie" -> sessionCookie(response),
      "X-CSRF-Token" -> csrfToken
    )

  private def queryParam(uri: String, key: String): Option[String] =
    Option(URI.create(uri).getRawQuery)
      .toVector
      .flatMap(_.split('&').toVector)
      .flatMap { pair =>
        pair.split("=", 2) match
          case Array(name, value) => Some(name -> java.net.URLDecoder.decode(value, StandardCharsets.UTF_8))
          case _ => None
      }
      .find(_._1 == key)
      .map(_._2)

  private def withUserStorePath[A](run: Path => A): A =
    val storeRoot = Files.createTempDirectory("platform-users-store-")
    try run(storeRoot.resolve("users.json"))
    finally deleteRecursively(storeRoot)

  private final class FakeOidcProvider extends PlatformUserAuth.OidcProvider:
    override val id = "google"
    override val displayName = "Google"

    override def authorizationUri(state: String, codeChallenge: String): String =
      s"https://accounts.google.test/o/oauth2/v2/auth?state=$state&code_challenge=$codeChallenge"

    override def exchangeCode(code: String, codeVerifier: String): Either[String, PlatformUserAuth.OidcIdentity] =
      Right(
        PlatformUserAuth.OidcIdentity(
          subject = s"fake-google-$code",
          email = "oidc@example.com",
          displayName = "OIDC User",
          avatarUrl = Some("https://example.com/avatar.png")
        )
      )

  private def generateProofUpload(outDir: Path): String =
    val hallResult = TexasHoldemPlayingHall.run(Array(
      "--hands=12",
      "--reportEvery=12",
      "--learnEveryHands=0",
      "--learningWindowSamples=50",
      "--seed=37",
      s"--outDir=$outDir",
      "--playerCount=6",
      "--heroPosition=Cutoff",
      "--heroStyle=gto",
      "--gtoMode=exact",
      "--villainPool=tag,lag,maniac",
      "--heroExplorationRate=0.0",
      "--raiseSize=2.5",
      "--bunchingTrials=8",
      "--equityTrials=80",
      "--saveTrainingTsv=false",
      "--saveDdreTrainingTsv=false",
      "--saveReviewHandHistory=true"
    ))
    assert(hallResult.isRight, s"hall proof run failed: $hallResult")

    val uploadPath = outDir.resolve("review-upload-pokerstars.txt")
    assert(Files.exists(uploadPath), s"missing review upload export: $uploadPath")
    Files.readString(uploadPath, StandardCharsets.UTF_8)

  private def deleteRecursively(path: Path): Unit =
    if Files.exists(path) then
      val stream = Files.walk(path)
      try
        val all = stream.iterator().asScala.toVector.sortBy(_.toString.length).reverse
        all.foreach(Files.deleteIfExists)
      finally stream.close()
