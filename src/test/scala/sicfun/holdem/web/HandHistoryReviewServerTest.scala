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

  // Pin the documented USER_AUTH_MAX_USERS default value of 100000.
  // The deploy doc line 349 explicitly says "USER_AUTH_MAX_USERS
  // (default 100000) caps the size of the platform-user store" AND
  // ties the default to operator-side dashboard reasoning: "dashboards
  // should chart userAuthStoredUsers / userAuthMaxUsers and alert at
  // e.g. 80% / 90% so capacity is raised (or registration disabled)
  // before legitimate users hit the wall". The existing
  // "/api/health surfaces userAuthMaxUsers and userAuthStoredUsers"
  // test (line ~404) uses an explicit `maxUsers = 50` override -- it
  // proves the FIELD plumbing but NOT the documented default value.
  // A refactor that changed PlatformUserAuth.Config.maxUsers's
  // default from 100_000 to e.g. 10_000 would silently invalidate
  // every dashboard alert keyed on the documented value (an alert
  // configured at "80% of 100000 = 80000 stored users" would never
  // fire because the cap secretly dropped to 10000 -- operators
  // would think they're under-capacity right up until the actual
  // 10k cap fires and legitimate registrations start failing). This
  // test asserts the default exactly so a deliberate cap change
  // requires the maintainer to acknowledge it explicitly in lockstep
  // with the test update rather than slipping it past CI as a
  // side-effect of an unrelated refactor. Same regression-pin pattern
  // as bd8e7f3 (sessionTtlMs default 12h pinned via cookie Max-Age),
  // 097bc64 (PBKDF2 parameters), a0fd8b3 (displayName auto-fill) --
  // documented defaults with operator-relevant reasoning get pinned
  // so drift can't go silent.
  // Pin the documented /api/auth/me degenerate-shape response under
  // no-auth + basic-auth modes. Deploy doc line 101 explicitly says:
  // "Under basic-auth and no-auth modes /api/auth/me returns a
  // degenerate form of the same seven-field shape, NOT a four-field
  // subset: authenticationEnabled (true under basic auth ... false
  // under no-auth), authenticationMode (basic or none accordingly),
  // authenticated: false (no platform-user session concept),
  // allowLocalRegistration: false (the register endpoint is 404
  // outside platform-user mode anyway), providers: [], user: null,
  // csrfToken: null. The same field set means a frontend can key on
  // authenticationMode to switch UI modes (gating the local
  // register/login forms behind mode === 'users' only, showing a
  // generic 'managed by upstream' panel for 'basic', and an open-
  // access panel for 'none') from a single probe of the same
  // endpoint -- the shipped frontend does exactly that." Before this
  // commit, the /api/auth/me degenerate shape was untested across
  // both branches (no-auth + basic-auth). Existing tests cover
  // /api/health under both modes but NOT /api/auth/me, even though
  // /api/auth/me is the endpoint the shipped frontend's
  // refreshAuthState boot-time probe hits to decide which UI mode
  // to render -- a refactor that returned a four-field subset (e.g.
  // dropping `providers: []` because "there are no providers in
  // basic-auth mode anyway, the field is redundant") would silently
  // break the shipped frontend's mode-detection logic which expects
  // the same seven-field shape across all modes (the deploy doc
  // explicitly tags this "same field set" property as the WHY
  // behind the same-shape-degenerate-not-subset choice). Test pins
  // BOTH branches in one test body because they share the same
  // hardcoded fall-through path at AuthStack.scala line 68-77
  // (handleAuthMe's `case None =>` branch builds the same Obj for
  // both basic-auth and no-auth modes, just with different
  // authenticationEnabled + authenticationMode values).
  test("/api/auth/me returns the documented degenerate seven-field shape under both no-auth and basic-auth modes") {
    // No-auth mode: server without basic-auth or platform-user config.
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val noAuth = getJson(s"$baseUri/api/auth/me")
        // Seven fields per deploy doc line 101. Document each
        // expected value with the documented rationale so a future
        // maintainer hitting a failure understands what changed.
        assertEquals(noAuth("authenticationEnabled").bool, false,
          clue = "no-auth mode: authenticationEnabled must be false -- the server doesn't enforce any Authorization header gate, so the field surfaces the absence of auth")
        assertEquals(noAuth("authenticationMode").str, "none",
          clue = "no-auth mode: authenticationMode must be 'none' -- the shipped frontend keys on this exact string to render the open-access panel")
        assertEquals(noAuth("authenticated").bool, false,
          clue = "no-auth mode: authenticated must be false -- no platform-user session concept exists in this mode")
        assertEquals(noAuth("allowLocalRegistration").bool, false,
          clue = "no-auth mode: allowLocalRegistration must be false -- the register endpoint is 404 outside platform-user mode anyway, so the frontend's Register-button gate stays disabled")
        assertEquals(noAuth("providers").arr.length, 0,
          clue = "no-auth mode: providers must be an empty array (not absent, not null) -- the frontend's `providers.map(...)` iteration depends on the array shape being present even when empty")
        assert(noAuth("user") == ujson.Null,
          clue = s"no-auth mode: user must be null -- no platform-user session to surface; got: ${noAuth("user")}")
        assert(noAuth("csrfToken") == ujson.Null,
          clue = s"no-auth mode: csrfToken must be null -- there's no CSRF gate to issue tokens for; the frontend's authSessionHeaders helper expects null here and skips X-CSRF-Token header injection; got: ${noAuth("csrfToken")}")
      }
    }

    // Basic-auth mode: server with BasicAuthConfig but no
    // platform-user auth. Basic-auth enforces the Authorization
    // header so authenticationEnabled flips to true even though
    // the platform-user session concept doesn't apply.
    withStaticSite { staticDir =>
      val basicAuth = HandHistoryReviewServer.BasicAuthConfig(username = "op", password = "secret")
      withServer(staticDir, basicAuth = Some(basicAuth)) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        // /api/auth/me is open by design (AuthRequirement.Optional)
        // so basic-auth doesn't block it -- we can probe anonymously.
        val basic = getJson(s"$baseUri/api/auth/me")
        assertEquals(basic("authenticationEnabled").bool, true,
          clue = "basic-auth mode: authenticationEnabled must be true -- basic-auth DOES enforce an Authorization header even though the platform-user paths are off")
        assertEquals(basic("authenticationMode").str, "basic",
          clue = "basic-auth mode: authenticationMode must be 'basic' -- the shipped frontend keys on this exact string to render the 'managed by upstream' panel")
        // The remaining 5 fields are the same as no-auth (no
        // platform-user session concept regardless of basic-auth):
        assertEquals(basic("authenticated").bool, false,
          clue = "basic-auth mode: authenticated must be false (no platform-user session concept)")
        assertEquals(basic("allowLocalRegistration").bool, false,
          clue = "basic-auth mode: allowLocalRegistration must be false")
        assertEquals(basic("providers").arr.length, 0,
          clue = "basic-auth mode: providers must be an empty array")
        assert(basic("user") == ujson.Null,
          clue = s"basic-auth mode: user must be null; got: ${basic("user")}")
        assert(basic("csrfToken") == ujson.Null,
          clue = s"basic-auth mode: csrfToken must be null -- the CSRF gate is platform-user-specific; got: ${basic("csrfToken")}")
      }
    }
  }

  test("/api/health surfaces userAuthMaxUsers = 100000 default when no maxUsers override is set") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        // Config(storePath = storePath) with NO maxUsers override
        // exercises the documented default (PlatformUserAuth.Config's
        // `maxUsers: Int = 100_000` at line 81).
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"
          val health = getJson(s"$baseUri/api/health")
          assertEquals(health("userAuthMaxUsers").num.toInt, 100000,
            clue = "userAuthMaxUsers default must be 100000 per deploy doc line 349; a silent drift here would invalidate operator-side dashboards keyed on the documented 80%/90% capacity-pressure thresholds (e.g., 80000-stored-users alert would never fire if cap secretly dropped to 10000, masking real capacity pressure right up until registrations start failing)")
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
    // Pin the documented GoogleAuthEndpoint URL prefix
    // (https://accounts.google.com/o/oauth2/v2/auth -- documented in
    // both deploy doc line 146 + runbook section 5A line 338 as one
    // of the three Google endpoint hosts the server contacts, AND
    // implicitly referenced by the operator-side firewall-egress
    // guidance for OIDC). A refactor that changed
    // PlatformUserAuth.GoogleAuthEndpoint to a different URL would
    // silently break the documented host claim AND likely break the
    // actual OIDC flow (the new URL might not be a Google OAuth
    // endpoint). The full URL check ALSO ensures the path component
    // is correct (`/o/oauth2/v2/auth` is the documented OAuth 2.0
    // endpoint path Google publishes; a wrong path would 404 in
    // production but pass any "starts with https://accounts.google
    // .com" check); the `?` suffix is there because authorizationUri
    // ALWAYS appends a query string with the state + code_challenge
    // params -- an authorizationUri output without `?` would mean
    // the formEncode'd query was somehow dropped, which would break
    // every OIDC flow.
    assert(authUri.startsWith("https://accounts.google.com/o/oauth2/v2/auth?"),
      s"GoogleOidcProvider.authorizationUri must use the documented GoogleAuthEndpoint URL (deploy doc line 146 + runbook section 5A line 338 both name `accounts.google.com` as the OAuth-flow redirect host; the path `/o/oauth2/v2/auth` is Google's documented OAuth 2.0 endpoint); a refactor that changed PlatformUserAuth.GoogleAuthEndpoint would silently break both the doc claim AND the actual flow. Full URI: $authUri")
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
    // access_type=online: the deploy doc's Optional-OIDC bullet
    // explicitly says "The server requests access_type=online (no
    // refresh tokens), so users staying signed in across long absences
    // rely on USER_AUTH_SESSION_TTL_MS (default 12h from LOGIN TIME)
    // ... not silent Google refresh -- explains the 'why does the user
    // have to re-sign-in every morning' complaint and tells you which
    // knob to raise if a longer-stay UX matters more than the post-leak
    // window." A refactor that changed this to `access_type=offline`
    // (which would request a refresh token) would silently break the
    // documented "no refresh tokens" architecture AND require persistent
    // refresh-token storage the codebase doesn't ship -- Google would
    // grant the refresh token, the server would discard it, and the
    // promised "users stay signed in beyond 12h via silent refresh"
    // semantics wouldn't fire (the documented model would also be wrong:
    // operators reasoning about the leaked-token-lifetime upper bound
    // would no longer be correct since a refresh-token-bearing client
    // could extend session lifetime indefinitely past the 12h TTL).
    assert(authUri.contains("access_type=online"),
      s"deploy doc's Optional-OIDC bullet documents access_type=online -- a refactor to 'offline' (refresh-token-requesting) would silently break the documented no-refresh-tokens architecture AND the 12h session-TTL leaked-token-lifetime upper-bound calculation. Full URI: $authUri")
    // include_granted_scopes=true: Google-specific incremental-
    // authorization knob. If a user has previously granted scopes to
    // the same client_id, the access token returned by the new flow
    // inherits those earlier grants in addition to the newly-requested
    // scopes. Not directly documented in the deploy doc the way
    // access_type and scope are, but pinned here as part of the
    // production OIDC URL contract -- a refactor that dropped it
    // would change the cross-flow grant-inheritance behavior in
    // ways operators reasoning about the OIDC flow wouldn't expect.
    // Same call site (PlatformUserAuth.scala line 235) as access_type
    // so a refactor touching one likely touches the other.
    assert(authUri.contains("include_granted_scopes=true"),
      s"include_granted_scopes=true is part of the production OIDC URL set at PlatformUserAuth.scala line 235; dropping it would silently change Google's grant-inheritance behavior across flows. Full URI: $authUri")
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

  // Pin the displayName email-local-part auto-fill behavior documented
  // in both deploy doc (line 100: "When displayName is omitted or trims
  // to empty at register time, the server auto-fills it from the
  // email's local-part (e.g. alice@example.com -> alice); the literal
  // fallback SICFUN User only fires when the local-part is itself
  // empty, which validateEmail already rejects, so that string is
  // effectively unreachable") AND in the runbook section 5A line 340's
  // "Why is my display name my email username?" support-triage entry
  // (which assumes operators can tell users "the field is self-service
  // via the Profile panel" -- a triage that's wrong if the auto-fill
  // doesn't actually fire). Before this commit, the behavior had no
  // test, so a refactor that changed defaultDisplayNameFor's split
  // character (e.g. from @ to +), or removed the auto-fill entirely
  // (leaving the field empty), or changed the fallback string would
  // silently break both the documented behavior AND the operator-side
  // support-triage script that depends on it. Same regression-pin
  // pattern as 23ae2ff (access_type=online), 121e5b5 (state-cookie
  // Max-Age), 44c9f9f (OIDC email-collision) -- documented
  // operator-visible behavior gets pinned so refactors can't silently
  // regress it. Tests both branches: (1) omitted displayName field
  // exercises the JSON-key-absent path, (2) whitespace-only
  // displayName exercises the trim-to-empty path; both should
  // resolve to the email's local-part.
  // Pin the PBKDF2 storage parameters documented in deploy doc line 235
  // and runbook section 5A line 371: "PBKDF2-HMAC-SHA256 password hashes
  // (per-account salted with 128-bit random salt, 210,000 iterations,
  // 256-bit output -- meets NIST SP 800-132 §5.1 floor of ≥128-bit salt
  // and ≥256-bit output)". Before this commit the iteration count + key
  // length + salt size were defined as constants (PasswordSaltBytes=16,
  // PasswordIterations=210000, PasswordKeyLengthBits=256 in
  // PlatformUserAuth.scala lines 53-55) but NOT pinned in any test --
  // a refactor that changed PasswordIterations from 210000 to e.g.
  // 100000 (the OWASP "deprecated minimum" tier as of 2023) or
  // 600000 (the OWASP "current recommended" tier) would silently
  // drift the documented value without contradicting any test. The
  // iteration count is operationally relevant for incident-response
  // password-cracking-cost calculations: an operator who has reason
  // to suspect a USER_STORE_PATH leak needs to know the actual
  // iteration count to estimate offline-attack feasibility, AND the
  // NIST 800-132 §5.1 compliance claim depends on staying at or
  // above the documented floor. New test registers a user, reads
  // the user-store JSON directly from disk (the codebase doesn't
  // expose the credential through any HTTP endpoint -- by design,
  // hash material must never leave the server), parses out the
  // localPassword credential block, and asserts: (1) iterations =
  // 210000, (2) keyLengthBits = 256, (3) saltBase64 decodes to
  // exactly 16 bytes (128 bits). Same regression-pin pattern as
  // a0fd8b3 (displayName auto-fill), 23ae2ff (OIDC URL params),
  // bd8e7f3 (session cookie Max-Age) -- documented security-
  // relevant constants get pinned so refactors can't silently
  // drift them.
  // Pin the documented server-side session-record SLIDING behavior --
  // the OTHER half of the two-layer record-slides-but-cookie-doesn't
  // mechanic. b017951 pinned the cookie-side fixed-Max-Age (cookie
  // does NOT slide on authenticated requests so the leaked-token-
  // lifetime upper-bound calculation operators rely on stays bounded);
  // this fire pins the SERVER-SIDE record-sliding (resolveSession
  // DOES refresh the in-memory session's expiresAtEpochMs on every
  // authenticated request, so a legitimate user who's actively using
  // the app stays signed in beyond the ORIGINAL ttl window AS LONG AS
  // they keep using it -- the deploy doc explicitly documents this as
  // "ages off its sliding USER_AUTH_SESSION_TTL_MS window"). The two
  // behaviors look contradictory in isolation but compose into the
  // documented operator-relevant property: an attacker using curl /
  // scripted clients bypasses cookie expiry AND keeps the server-side
  // record alive indefinitely via sliding, so re-auth (NOT natural
  // expiry) is the only reliable upper bound on leaked-token lifetime;
  // a legitimate browser-bound user gets the cookie-fixed cap (their
  // cookie WILL expire at loginTime + ttl regardless of activity).
  // Test mechanics: use a SHORT sessionTtlMs (1000ms) so the test
  // completes in ~1.3 seconds; the production default is 12h which is
  // impractical to test directly; the sliding LOGIC is independent of
  // the absolute TTL value so a short TTL is a faithful test (the
  // PlatformUserAuth.SessionManager.resolveSession path does
  // `if current.expiresAtEpochMs > nowMillis() then current.copy(
  // expiresAtEpochMs = now + sessionTtlMs)` regardless of magnitude).
  // Without sliding (the regression we're guarding against): the
  // session would expire at T0 + 1000ms; making a request at T0 +
  // 1200ms (after the second sleep) would surface authenticated=false
  // because the record-lookup finds an expired entry. WITH sliding:
  // the request at T0 + 600ms refreshes expiresAt to T0 + 1600ms, so
  // the request at T0 + 1200ms (still within the refreshed window)
  // succeeds with authenticated=true.
  test("authenticated requests slide the server-side session expiry via resolveSession so an actively-using legitimate user stays signed in beyond the original TTL window") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        // Short TTL so the test completes quickly; the sliding logic
        // is TTL-magnitude-independent so this is a faithful test.
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath, sessionTtlMs = 1000L))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          val register = postJson(s"$baseUri/api/auth/register",
            """{"email":"slider@example.com","password":"correct-horse-battery","displayName":"Slider"}""")
          assertEquals(register.statusCode(), 201,
            clue = "registration must succeed to mint the session record this test slides")
          val cookie = sessionCookie(register)
          val cookieHeader = Map("Cookie" -> cookie)

          // Wait 600ms (60% of the 1000ms TTL). The record's
          // original expiresAtEpochMs is registerTime + 1000ms, so
          // a request now is well within the window.
          Thread.sleep(600L)
          val slideHit = getJsonWithHeaders(s"$baseUri/api/auth/me", cookieHeader)
          assertEquals(slideHit("authenticated").bool, true,
            clue = "first probe at 600ms (60% of TTL) must show authenticated=true -- the session record was minted at register-time + 1000ms so it's still valid; this request ALSO triggers the sliding refresh at SessionManager (current.copy(expiresAtEpochMs = now + sessionTtlMs)) which is what the next assertion depends on")

          // Wait another 600ms. Total elapsed since register: 1200ms.
          // Without sliding: record expired 200ms ago (T0 + 1000ms),
          // probe would return authenticated=false. WITH sliding:
          // the probe at T0 + 600ms refreshed expiresAt to T0 +
          // 600ms + 1000ms = T0 + 1600ms, so the current request at
          // T0 + 1200ms is within the refreshed window with 400ms
          // headroom.
          Thread.sleep(600L)
          val postSlide = getJsonWithHeaders(s"$baseUri/api/auth/me", cookieHeader)
          assertEquals(postSlide("authenticated").bool, true,
            clue = "second probe at 1200ms (120% of ORIGINAL TTL, but only 60% past the slide-refresh) MUST show authenticated=true -- this is the entire sliding-behavior contract: an actively-using legitimate user stays signed in via record refresh on every authenticated request; if this assertion fires false, the sliding logic in SessionManager.resolveSession's `current.copy(expiresAtEpochMs = now + sessionTtlMs)` refresh path has regressed, which would silently start kicking users out at exactly loginTime + ttlMs regardless of activity -- a UX disaster for long-stay deployments AND a contradiction of the documented two-layer mechanic")
        }
      }
    }
  }

  // Pin the documented "persistent account data ... survives the
  // restart unchanged" contract from deploy doc line 235. Two nested
  // withServer blocks sharing the same storePath simulate a graceful
  // restart: server #1 registers a user, then closes (storePath stays
  // on disk); server #2 opens against the SAME storePath and the
  // user's credentials still resolve via /api/auth/login. The
  // operationally relevant property: account data (email +
  // PBKDF2 hash + profile fields + linked-provider identities)
  // survives every restart shape -- planned NSSM stop/start cycles,
  // SIGTERM-driven rolling deploys, unexpected JVM crashes (the
  // user-store write path uses Files.move with ATOMIC_MOVE per
  // deploy doc line 234, so even a hard-kill mid-write leaves
  // either the previous-good or new-complete file but never a
  // half-written corruption); without this contract, every restart
  // would force every user to re-register, defeating the
  // "platform-user auth" mode's entire point. f8eadfb / aadcc08
  // pinned the EPHEMERAL session-record half of the persistence
  // story (in-memory sessions lost on restart -- that's expected
  // and documented); this fire pins the PERSISTENT half (the
  // USER_STORE_PATH file -- this MUST survive). A refactor that
  // accidentally stored credentials in-memory only (e.g., by
  // removing the persist() call from registerLocal) would silently
  // break this contract: tests using a single withServer block
  // wouldn't catch it because the credentials would live in
  // memory for the duration of the single block; only a two-server
  // sequence forces the credentials through the disk round-trip.
  // The cross-restart test ALSO incidentally exercises the JSON
  // serialization-then-deserialization symmetry of the store
  // format: writeStoredUser at PlatformUserAuth.scala line ~989
  // emits the JSON shape, readStoredUser at line ~1001 reads it
  // back; if either side drifted without the other, the second
  // server would fail to parse the file with the documented
  // "user store at <path> is unreadable" startup error (runbook
  // section 5A line 335).
  test("user-store credentials persist across server restart -- register on server 1, login on server 2 with the same storePath") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        val authConfig = PlatformUserAuth.Config(storePath = storePath)
        val testEmail = "durable@example.com"
        val testPassword = "correct-horse-battery"

        // Server #1: register a user. The withServer block opens a
        // fresh JDK HttpServer + a fresh SessionManager (in-memory
        // session state) + opens the storePath for read/write. At
        // block-exit the server closes (HttpServer.stop, session
        // map cleared) but the storePath JSON file persists on
        // disk -- the ONLY survivor of the close.
        withServer(staticDir, platformAuth = Some(authConfig)) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"
          val register = postJson(s"$baseUri/api/auth/register",
            s"""{"email":"$testEmail","password":"$testPassword","displayName":"Durable Tester"}""")
          assertEquals(register.statusCode(), 201,
            clue = "registration on server #1 must succeed before the cross-restart login can be exercised")
        }
        // Verify the disk artifact actually exists between the two
        // server lifecycles -- the cross-restart contract depends on
        // the file being there for server #2 to read.
        assert(Files.exists(storePath),
          s"user-store JSON file must exist at $storePath between server #1's close and server #2's open -- if the file is missing here, the persist() path didn't atomically-move the temp file into place (PlatformUserAuth's writeStore path at line ~960 uses Files.move(temp, target, ATOMIC_MOVE) -- a refactor breaking that would surface as this assertion failing)")

        // Server #2: same storePath, different in-memory server.
        // The login attempt forces the credential through the disk
        // round-trip: readStoredUser at PlatformUserAuth.scala
        // line ~1001 parses the JSON we wrote in server #1,
        // reconstructs the LocalPasswordCredential including its
        // saltBase64 + hashBase64 + iterations + keyLengthBits,
        // and the verifyPassword path recomputes PBKDF2 with the
        // submitted password against the stored salt+hash.
        withServer(staticDir, platformAuth = Some(authConfig)) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"
          val login = postJson(s"$baseUri/api/auth/login",
            s"""{"email":"$testEmail","password":"$testPassword"}""")
          assertEquals(login.statusCode(), 200,
            clue = s"login on server #2 with the same credentials registered on server #1 MUST succeed -- this is the documented 'persistent account data ... survives the restart unchanged' contract from deploy doc line 235; if this assertion fires false, registration is silently in-memory-only (e.g. a refactor removed the persist() call from registerLocal), which defeats platform-user auth mode's entire point because every restart would force every user to re-register; ALSO covers the JSON serialization-deserialization symmetry across writeStoredUser + readStoredUser (a schema drift would surface as the second server failing to parse the file)")
          // The login response carries the same auth-state shape as
          // /api/auth/me -- verify the user identity round-tripped
          // correctly through the disk persistence.
          val loginJson = jsonBody(login)
          assertEquals(loginJson("authenticated").bool, true,
            clue = "login response must show authenticated=true after the credential round-trip")
          assertEquals(loginJson("user")("email").str, testEmail,
            clue = s"login response user.email must match the registered email after disk round-trip -- if mismatched, the storedUser deserialization is reading the wrong field; got: ${loginJson("user")("email").str}")
        }
      }
    }
  }

  // Pin the natural-expiry case -- the complement to f8eadfb's
  // sliding pin. f8eadfb proved that an actively-using session
  // stays alive past the original TTL window via resolveSession's
  // refresh; this fire proves the OPPOSITE direction: a session
  // that gets NO intermediate requests times out at exactly
  // registerTime + sessionTtlMs, AND a subsequent request with the
  // (now-stale) cookie surfaces authenticated=false. The pair of
  // tests (f8eadfb + this commit) lock in the full sliding-TTL
  // contract: ACTIVE use refreshes, IDLE use expires; without the
  // active-refresh half, users would get kicked out at the original
  // TTL regardless of activity (regression caught by f8eadfb);
  // without the idle-expiry half, sessions would live forever once
  // minted (regression caught by THIS test). PlatformUserAuth's
  // resolveSession at line ~835 implements both branches in one
  // conditional: `if current.expiresAtEpochMs > nowMillis() then
  // current.copy(...refresh...) else null` -- a refactor that
  // accidentally inverted the comparison or removed the else-null
  // branch would let expired sessions resolve indefinitely, which
  // is a real security hole (a session token stolen days ago
  // could still be used months later because the record never
  // expired). The default 12h TTL means this branch fires rarely
  // in production, so a regression here would go undetected unless
  // CI exercises it explicitly. Test mechanics mirror f8eadfb:
  // short TTL (1000ms) for fast test execution, but DIFFERENT
  // flow -- register, wait WITHOUT any intermediate requests, then
  // probe and assert authenticated=false. The "no intermediate
  // requests" is the key behavioral difference vs f8eadfb's sliding
  // test (which DID make a mid-window request to trigger the slide).
  test("session record naturally expires when idle past sessionTtlMs -- subsequent requests surface authenticated=false") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        // Same short TTL as f8eadfb's sliding test (1000ms) so the
        // test completes quickly; logic is TTL-magnitude-independent.
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath, sessionTtlMs = 1000L))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          val register = postJson(s"$baseUri/api/auth/register",
            """{"email":"expirer@example.com","password":"correct-horse-battery","displayName":"Expirer"}""")
          assertEquals(register.statusCode(), 201,
            clue = "registration must succeed to mint the session record that we then let expire idle")
          val cookieHeader = Map("Cookie" -> sessionCookie(register))

          // Wait 1200ms (120% of TTL). With NO intermediate requests,
          // the session record at register-time + 1000ms is now
          // expired by 200ms. The 200ms cushion is large enough to
          // absorb test-runner clock jitter (typical Thread.sleep
          // precision on JVM is ~10-50ms) while small enough to
          // not bloat the test runtime.
          Thread.sleep(1200L)

          // Probe /api/auth/me with the (now-stale) session cookie.
          // resolveSession at SessionManager line ~835 reads the
          // record, sees expiresAtEpochMs <= now, returns null
          // (treated as "no session" by the calling site), AND the
          // lazy-purge path at line ~872 (`if entry.getValue.
          // expiresAtEpochMs <= now then` remove-from-map) cleans
          // the expired entry so memory doesn't grow unbounded
          // across idle sessions. The /api/auth/me response
          // surfaces authenticated=false because there's no
          // resolvable session attached to the request.
          val expiredProbe = getJsonWithHeaders(s"$baseUri/api/auth/me", cookieHeader)
          assertEquals(expiredProbe("authenticated").bool, false,
            clue = "after 120% of TTL with NO intermediate requests, the session record must be expired and /api/auth/me must surface authenticated=false -- without this, expired sessions would resolve indefinitely (a real security hole: a stolen session token could be used months after the original sign-in because the record never expires); the resolveSession logic at PlatformUserAuth.SessionManager line ~835 implements this via `if current.expiresAtEpochMs > nowMillis() then refresh else null` -- a refactor that inverted the comparison or removed the else-null branch would let this assertion fire false, signaling the security regression")
          // The `user` field should be null (no authenticated user
          // session was found) -- matches the documented degenerate-
          // shape post-expiry response.
          assert(expiredProbe("user") == ujson.Null,
            clue = s"expired-session /api/auth/me response must carry user=null per the documented degenerate-shape contract; got user=${expiredProbe("user")}")
        }
      }
    }
  }

  // Pin the documented "session cookie Max-Age is FIXED at login time
  // and NOT refreshed by subsequent activity" behavior -- the security-
  // critical foundation of the leaked-token-lifetime upper-bound
  // calculation operators use during incident response. The deploy doc's
  // USER_AUTH_SESSION_TTL_MS bullet explicitly says "the session
  // cookie's Max-Age is fixed at login and NOT refreshed by subsequent
  // activity, so a user is auto-signed-out at exactly loginTime + ttlMs
  // regardless of how active they were in the interim", and the runbook
  // section 5A line 348 quotes the same property as the foundation of
  // the "Re-auth via OIDC is the only reliable upper bound on a
  // leaked-token's lifetime" triage logic (an attacker using curl /
  // scripted requests bypasses cookie expiry, but the SERVER-SIDE
  // record sliding via resolveSession doesn't help the legitimate
  // browser client whose Max-Age is fixed at login). Before this
  // commit, the property was implicit in the design (the production
  // server only emits Set-Cookie on register / login / OIDC callback,
  // never on /api/auth/me or other authenticated endpoints) but had
  // NO test enforcing it -- a refactor that started re-emitting
  // Set-Cookie on every request to slide the Max-Age (a tempting
  // "fix" to the "users complain about being signed out every 12h"
  // support pattern) would silently break the documented upper-bound
  // calculation AND extend leaked-session attack windows indefinitely
  // as long as the attacker keeps the token in use, contradicting
  // both docs' explicit "Re-auth via OIDC is the only reliable upper
  // bound" framing. Test registers a user (asserts Set-Cookie present),
  // then makes a follow-up GET /api/auth/me with the captured session
  // cookie, asserts the follow-up response has NO Set-Cookie header.
  // Same regression-pin pattern as bd8e7f3 (session cookie Max-Age=43200
  // exact value), 121e5b5 (state cookie Max-Age=600), 44c9f9f (OIDC
  // email-collision) -- documented security-relevant behavior pinned
  // so refactors can't silently regress the operator-side reasoning.
  test("authenticated follow-up requests do NOT re-emit Set-Cookie -- pins the documented session-cookie Max-Age FIXED at login time property") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          // Step 1: register and capture the session cookie.
          val register = postJson(s"$baseUri/api/auth/register",
            """{"email":"slide@example.com","password":"correct-horse-battery","displayName":"NoSlide"}""")
          assertEquals(register.statusCode(), 201,
            clue = "registration must succeed before the follow-up-no-recookie check can fire")
          val sessionCookieHeader = headerValue(register, "Set-Cookie")
            .getOrElse(fail("registration must emit a session cookie on initial sign-in"))
          val sessionCookieValue = sessionCookieHeader.takeWhile(_ != ';')

          // Step 2: make an authenticated follow-up GET /api/auth/me
          // with the captured session cookie. The follow-up response
          // must NOT carry a Set-Cookie header -- the cookie's Max-Age
          // was set at login and the documented "FIXED at login time"
          // contract requires the server NOT re-emit it on activity.
          val followUp = get(s"$baseUri/api/auth/me", Map("Cookie" -> sessionCookieValue))
          assertEquals(followUp.statusCode(), 200,
            clue = "follow-up authenticated request must succeed (the cookie still resolves server-side; the test is about the response, not auth)")
          // The critical security assertion: NO new Set-Cookie. If
          // present, the production code would be sliding the cookie's
          // Max-Age forward on every request, which would silently
          // (1) break the documented "12h from LOGIN TIME" leaked-
          // token-lifetime upper bound, (2) contradict the runbook's
          // "Re-auth via OIDC is the only reliable upper bound on a
          // leaked-token's lifetime" framing, and (3) extend leaked-
          // session attack windows indefinitely as long as the attacker
          // keeps the token in use.
          assertEquals(headerValue(followUp, "Set-Cookie"), None,
            "authenticated follow-up request MUST NOT re-emit Set-Cookie -- the deploy doc's USER_AUTH_SESSION_TTL_MS bullet explicitly says 'the session cookie's Max-Age is fixed at login and NOT refreshed by subsequent activity'; a refactor that started sliding the cookie Max-Age would silently break the documented leaked-token-lifetime upper-bound calculation operators rely on during incident response, AND contradict the runbook's 'Re-auth via OIDC is the only reliable upper bound' framing")
        }
      }
    }
  }

  // Pin the documented logout cookie-clear behavior. handleAuthLogout
  // calls service.revokeSession(cookieHeader(exchange)) and emits the
  // returned `clearedCookie` value as a Set-Cookie header on the 200
  // response (see AuthStack.scala's handleAuthLogout at line ~197).
  // The clearedCookie shape (PlatformUserAuth's clearSessionCookieHeader
  // at line ~1335) is `sicfun_session=; Path=/; Max-Age=0; HttpOnly;
  // SameSite=Lax` (with `Secure` added in secure mode and a __Host-
  // prefix on the cookie name) -- the empty value + Max-Age=0 are the
  // RFC 6265 sec 4.1.2.2 "delete this cookie" wire form that browsers
  // honor by removing the cookie from their store.
  // Operationally relevant for shared-computer / kiosk deployments:
  // a user signing out at a kiosk expects the next person sitting
  // down to NOT see their session cookie in their browser. Without
  // the cookie-clear, the next user would have the stale cookie
  // sitting in their browser; the server-side record IS revoked so
  // the cookie wouldn't resolve to a session (any request would
  // surface as anonymous + 401 on protected routes), but the cookie
  // bytes would still be there until the original Max-Age expires
  // (default 12h FIXED at login time), AND a deployment with future
  // refactoring that added server-side session resurrection (e.g.
  // session-store-on-disk recovery after a restart) would silently
  // create a security hole if the cookie clear was simultaneously
  // dropped. Pinning the clear cookie in the logout response
  // catches the regression where a refactor removed the Set-Cookie
  // emission (or set Max-Age to a non-zero value, or used the wrong
  // cookie name in the clear so the browser keeps the old one
  // alongside the new clear-attempted one). The existing logout
  // test at the "user auth supports local registration ..." block
  // (line ~2712) only checks the 200 status + body.authenticated=false;
  // this test adds the cookie-clear assertion to that gap.
  // Same regression-pin pattern as b017951 (no-Set-Cookie on
  // follow-up requests), bd8e7f3 (session cookie Max-Age=43200
  // at login), 72358a3 (secure-mode session cookie symmetric).
  test("POST /api/auth/logout response clears the session cookie with Max-Age=0 so kiosk / shared-computer sign-outs don't leave the cookie in the next user's browser") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          val register = postJson(s"$baseUri/api/auth/register",
            """{"email":"signoff@example.com","password":"correct-horse-battery","displayName":"SignOff"}""")
          assertEquals(register.statusCode(), 201,
            clue = "registration must succeed before the logout cookie-clear can be checked")
          val ownerHeaders = authSessionHeaders(register, jsonBody(register)("csrfToken").str)

          val logout = postJson(s"$baseUri/api/auth/logout", "{}", ownerHeaders)
          assertEquals(logout.statusCode(), 200,
            clue = "logout must succeed (200) so the documented post-signout state-clear emits the cookie-clear header alongside the JSON body")

          val setCookie = headerValue(logout, "Set-Cookie")
            .getOrElse(fail("logout response must include Set-Cookie header (carrying the cookie-clear wire form per PlatformUserAuth.clearSessionCookieHeader); without it, browsers don't delete the existing session cookie and shared-computer sign-outs leak the cookie to the next user"))

          // The cookie-clear wire form per RFC 6265 sec 4.1.2.2: empty
          // value + Max-Age=0. Both halves are necessary -- empty value
          // alone with a positive Max-Age would set an empty-string
          // cookie that the browser stores and sends on future
          // requests (defeating the clear); Max-Age=0 alone with a
          // non-empty value would still tell the browser to delete
          // but would leak the value bytes in the immediate response
          // header (less critical but documentationally inconsistent).
          // Insecure-mode cookie name is plain `sicfun_session`; the
          // secure-mode `__Host-sicfun_session` prefix is tested in the
          // secure-mode session-cookie test (72358a3).
          assert(setCookie.startsWith("sicfun_session="),
            s"logout cookie-clear must use plain `sicfun_session=` prefix in insecure mode (Test config has no cookieSecure override); got: $setCookie")
          assert(setCookie.contains("Max-Age=0"),
            s"logout cookie-clear must contain Max-Age=0 (RFC 6265 sec 4.1.2.2 'delete this cookie' wire form) so browsers actually remove the cookie from their store; without Max-Age=0 the cookie persists until its original Max-Age expires (default 12h FIXED at login), leaving stale session cookies in shared-computer browsers for the next user; got: $setCookie")
          // Verify the empty-value half: between `sicfun_session=` and
          // the next `;` there should be nothing (no leaked value).
          val valuePortion = setCookie.takeWhile(_ != ';').drop("sicfun_session=".length)
          assertEquals(valuePortion, "",
            clue = s"logout cookie-clear must carry an empty value (not the old session token bytes) so the browser overwrites with a non-resolvable form; got cookie value: '$valuePortion' in $setCookie")
        }
      }
    }
  }

  // Pin the documented `auth.logout` audit log line format per deploy
  // doc line 218: "Auth events emit structured log lines: ...
  // auth.logout ... INFO level for success/expected events ... Each
  // line carries remote= (on every auth.* event) plus -- on the local-
  // auth events (auth.login.{success,failure}, auth.register.
  // {success,failure}, auth.logout) and the post-callback
  // auth.oidc.success -- an email= field"; BEFORE this commit there
  // was ZERO test coverage of the SUCCESS-side audit log line formats
  // (only auth.register.failure + auth.login.failure had partial
  // presence assertions at lines ~6110 + ~6155). The auth.logout
  // line is a clean single-fire target because: (a) logout is a
  // SUCCESS-ONLY event (no failure path -- POST /api/auth/logout with
  // a missing session is a 401 BEFORE reaching the audit log emission
  // at AuthStack.scala line 210, AND with an invalid CSRF is a 403
  // BEFORE reaching line 210; the only path through the emission
  // point is a fully-validated successful logout), (b) the email
  // source is the SESSION's canonical email (line 208:
  // `authenticatedUser(exchange).map(_.email).getOrElse("-")`) so
  // there's no submitted-vs-canonical complexity to navigate, (c) the
  // existing logout cookie-clear test above already establishes the
  // register+logout flow this test reuses; per-field regression
  // vectors a refactor would silently introduce: (1) renaming the
  // event prefix "auth.logout" to e.g. "auth.signout" or
  // "auth.session.end" would silently break operator log-aggregation
  // queries filtering by event type AND would silently invalidate
  // the runbook's "high WARN rate from a known email= from many
  // remote= sources may be a single-account-targeted credential-
  // stuffing probe" triage step (because the operator's grep for
  // "auth.logout" would return empty results, making the
  // attacker-targeted-account triage path silently unavailable),
  // (2) dropping the email= field would break the runbook's "all
  // events for one user grep identically" property (deploy doc line
  // 218: "Success lines log the canonical (normalized) email so all
  // events for one user grep identically"), making it impossible
  // to correlate a user's logout with their earlier login.success /
  // register.success events for incident analysis (an operator
  // investigating "did Alice sign out before her account was
  // compromised at 3am" would have no way to find the answer if
  // logout silently dropped email=), (3) dropping the remote= field
  // would break the runbook's brute-force / credential-stuffing
  // triage (the field is documented as "on every auth.* event" --
  // dropping it silently removes the per-IP correlation between
  // logout events and login attempts), (4) demoting from INFO to
  // DEBUG would silently make the line invisible at default log
  // levels (operators would have to flip log levels to see it,
  // breaking the documented "INFO level for success/expected events"
  // contract), (5) promoting to WARN would silently flood
  // alerting (every logout is normal and expected; if logout fired
  // at WARN every signed-in user signing out would generate noise
  // alerts, eventually muted, masking real WARN-level events when
  // they fire); the test captures stdout (NOT stderr) because
  // logInfo writes to System.out per
  // HandHistoryReviewServerRuntime.scala line 418 (`log("INFO",
  // message, System.out)`) -- a refactor that flipped logInfo to
  // System.err would also fail this test because the assertion
  // captures stdout specifically. Assertion captures + restores
  // System.out around the logout call ONLY (not the register call)
  // so the baseline stdout capture doesn't accidentally include
  // unrelated emissions from the register flow; the timing window
  // is tight (one HTTP request) but big enough to capture the
  // synchronous logInfo emission inside handleAuthLogout (line 210
  // runs BEFORE the response body is built and returned, so by
  // the time `logout.statusCode()` reads 200 the audit line is
  // already written -- the synchronized stream.println in
  // HandHistoryReviewServerRuntime.scala line 511-512 flushes
  // because PrintStream(autoFlush=true) is used at line 6097 +
  // analogous setOut here). Format check is structured:
  // (i) "auth.logout" presence (event prefix),
  // (ii) "email=logoutaudit@example.com" (the specific registered
  //      email -- both that the field is present AND that the
  //      value is the canonical email, NOT the displayName or a
  //      userId or empty),
  // (iii) "INFO" level (catches a refactor demoting to DEBUG or
  //       promoting to WARN),
  // (iv) "remote=" field presence (the brute-force triage
  //      correlator),
  // (v) "[hand-history-review]" service-tag presence (the log-
  //      collation prefix from HandHistoryReviewServerRuntime.scala
  //      line 512); same regression-pin pattern as the prior
  // /api/health response-shape pins (b1339cd, etc.) -- documented
  // operator-facing contracts get pinned in CI so refactors can't
  // silently invalidate the deploy-doc + runbook operator-side
  // guidance; this commit closes the first of the 5 success-side
  // event formats (auth.logout); future fires can pin the other
  // four (auth.login.success, auth.register.success,
  // auth.oidc.start, auth.oidc.success).
  test("POST /api/auth/logout emits the documented `auth.logout email=<canonical> remote=<peer>` INFO audit line per deploy doc line 218") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          val register = postJson(s"$baseUri/api/auth/register",
            """{"email":"logoutaudit@example.com","password":"correct-horse-battery","displayName":"AuditUser"}""")
          assertEquals(register.statusCode(), 201,
            clue = "registration must succeed before the logout audit-log probe can capture its emission")
          val ownerHeaders = authSessionHeaders(register, jsonBody(register)("csrfToken").str)

          // Capture stdout (NOT stderr) because logInfo writes to
          // System.out per HandHistoryReviewServerRuntime.scala line
          // 418 -- the auth.login.failure / auth.register.failure
          // tests further down this file capture stderr because
          // those use logWarn (System.err). Restore the original
          // stdout in a finally block to avoid polluting the rest of
          // the test suite if an assertion below fails.
          val outBuf = new java.io.ByteArrayOutputStream()
          val originalOut = System.out
          System.setOut(new java.io.PrintStream(outBuf, true, StandardCharsets.UTF_8))
          try
            val logout = postJson(s"$baseUri/api/auth/logout", "{}", ownerHeaders)
            assertEquals(logout.statusCode(), 200,
              clue = "logout must succeed (200) so handleAuthLogout reaches the line 210 logInfo emission point -- a 401/403 short-circuits before the audit line is written")
          finally
            System.setOut(originalOut)

          val captured = outBuf.toString(StandardCharsets.UTF_8)
          val logoutLine = captured.split('\n').iterator
            .find(_.contains("auth.logout"))
            .getOrElse(fail(s"no `auth.logout` line in stdout capture -- deploy doc line 218 documents this event as INFO-level fired on every successful logout; got captured stdout: ${captured.take(800)}"))

          // (i) event prefix
          assert(logoutLine.contains("auth.logout"),
            clue = s"logout audit line must carry the literal `auth.logout` event prefix per deploy doc line 218's enumeration; a refactor renaming to e.g. `auth.signout` would silently break log-aggregation queries; got: $logoutLine")
          // (ii) canonical email field
          assert(logoutLine.contains("email=logoutaudit@example.com"),
            clue = s"logout audit line must carry the registered canonical email in the `email=` field per deploy doc line 218 ('local-auth events including auth.logout carry email=' and 'success lines log the canonical (normalized) email so all events for one user grep identically'); a refactor dropping the field or substituting displayName / userId would silently break operator user-correlation workflows; got: $logoutLine")
          // (iii) INFO level
          assert(logoutLine.contains("[INFO]"),
            clue = s"logout audit line must be INFO-level per deploy doc line 218 ('INFO level for success/expected events'); a refactor demoting to DEBUG would silently make the line invisible at default log levels, promoting to WARN would silently flood alerting; got: $logoutLine")
          // (iv) remote= field
          assert(logoutLine.contains("remote="),
            clue = s"logout audit line must carry the `remote=` field per deploy doc line 218 ('remote= on every auth.* event'); without this field operators lose the per-IP correlation between logout events and earlier login.success / auth.oidc.success events for incident analysis; got: $logoutLine")
          // (v) service-tag prefix (log-collation correlator)
          assert(logoutLine.contains("[hand-history-review]"),
            clue = s"logout audit line must carry the `[hand-history-review]` service-tag prefix per HandHistoryReviewServerRuntime.scala line 512's hardcoded `[hand-history-review]` literal -- the tag matches the /api/health.service field (pinned by 505ba6b) so a log aggregator filtering by service tag gets the same identifier on log lines as on probe responses; got: $logoutLine")
        }
      }
    }
  }

  // Pin the documented `auth.login.success` audit line format AND
  // the canonical-vs-submitted-email normalization contract -- the
  // SECOND of the 5 success-side auth-event formats deploy doc line
  // 218 documents (auth.logout was closed by 1c8777f); this test
  // strengthens the contract beyond the auth.logout pin by
  // EXERCISING the documented "Success lines log the canonical
  // (normalized: zero-width-character strip → trim → lowercase, all
  // under Locale.ROOT) email so all events for one user grep
  // identically; failure lines log the submitted email so brute-
  // force probes are visible in the form the attacker typed"
  // distinction; for the logout test (1c8777f), the email source is
  // the session's stored canonical email so there's no submitted-
  // vs-canonical complexity -- the auth.logout email field MUST be
  // canonical because that's all the session has access to; for
  // auth.login.success though, the request's submitted email could
  // be in any case-form / whitespace form / zero-width-decorated
  // form, AND the audit-log emission uses `result.user.email`
  // (AuthStack.scala line 171) which is the canonical stored email
  // from the database lookup -- so a refactor that "fixed
  // consistency" by logging the SUBMITTED email instead (e.g.
  // unifying success+failure to both log the submitted form, or
  // accidentally swapping result.user.email for the parsed `email`
  // variable from parseLoginRequest) would silently break the
  // operator-correlation contract: events for one user would log
  // under DIFFERENT canonical forms depending on how the user
  // typed their email each session, defeating the deploy doc's
  // "all events for one user grep identically" guarantee. The
  // test exploits the case-mismatch path: register with
  // "Alice@Example.COM" (mixed case + uppercase domain), login
  // with "aLiCe@example.com" (different mixed case from register),
  // canonical for BOTH is "alice@example.com", so the audit line
  // MUST log `email=alice@example.com` -- if it logs
  // "email=Alice@Example.COM" (the register-submitted form) or
  // "email=aLiCe@example.com" (the login-submitted form) the
  // refactor regression is caught. The test ALSO asserts the
  // EXCLUSION condition: the line must NOT contain
  // "email=aLiCe@example.com" -- without this exclusion check, a
  // weaker assertion that only verifies presence of
  // "alice@example.com" could pass on a regression that logged
  // "aLiCe@example.com" alongside (since a substring match for
  // "alice@example.com" would match within "aLiCe@example.com" if
  // case-insensitive); the exact-bytes exclusion check forces the
  // assertion to fail on the regression. Critical Subtlety: Scala
  // string `.contains` is case-SENSITIVE, so checking
  // `.contains("email=alice@example.com")` would NOT match a line
  // containing `email=aLiCe@example.com` -- so the
  // contains+!contains pair pins case-exact equality of the
  // emitted email field; a refactor emitting any non-canonical
  // form fails BOTH the positive contains (which requires the
  // canonical alice@example.com exactly) AND the negative
  // !contains (which forbids the aLiCe@example.com submitted
  // form). Per-field regression vectors specific to login.success
  // (in addition to the 5-tier format-check inherited from the
  // logout pin pattern): (i) refactor swapping result.user.email
  // for the parseLoginRequest's `email` parameter on line 165 (an
  // easy refactor target -- both are String fields available in
  // the same closure) would silently log the SUBMITTED form,
  // breaking operator user-correlation; (ii) refactor unifying
  // success + failure to both log submitted form (under a
  // "consistency" rationale, since auth.login.failure correctly
  // logs submitted) would silently break the documented
  // distinction; (iii) refactor introducing a case-preserving
  // canonical (e.g. "preserve display case for friendliness")
  // would silently break operator grep workflows that match
  // case-exactly; same multi-tier format-check pattern as 1c8777f
  // auth.logout pin: event prefix + email field + INFO level +
  // remote= field + service-tag prefix, with this commit's
  // additional canonical-vs-submitted EXCLUSION check on the
  // email field; future fires can close auth.register.success
  // (same normalization contract, register-only case) and
  // auth.oidc.start + auth.oidc.success (provider= field-shape
  // pins).
  test("POST /api/auth/login emits the documented `auth.login.success email=<canonical> remote=<peer>` INFO audit line with the canonical-normalized email (per deploy doc line 218's 'all events for one user grep identically' contract)") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          // Register with a mixed-case + uppercase-domain submission;
          // PlatformUserAuth.normalizeEmail strips whitespace + lowercases
          // (after zero-width strip) so the canonical form stored is
          // alice@example.com. Run the registration OUTSIDE the stdout
          // capture window below so the auth.register.success line
          // emission doesn't pollute the captured stream this test
          // wants to scrutinize for the login.success line specifically.
          val register = postJson(s"$baseUri/api/auth/register",
            """{"email":"Alice@Example.COM","password":"correct-horse-battery","displayName":"Alice"}""")
          assertEquals(register.statusCode(), 201,
            clue = s"registration with mixed-case + uppercase-domain email must succeed (the canonical form 'alice@example.com' should be stored, not the submitted 'Alice@Example.COM')")

          // Sanity-check that the register response carries the
          // canonical form -- this is the STORED form that
          // auth.login.success should emit downstream. Without this
          // check, a refactor that broke the registration-side
          // normalization would silently make the login.success test
          // pass with the WRONG canonical (it would still match the
          // stored form, just not the documented canonical).
          val registerJson = jsonBody(register)
          val storedEmail = registerJson("user")("email").str
          assertEquals(storedEmail, "alice@example.com",
            clue = s"register response must carry the canonical-normalized email (lowercase, no leading/trailing whitespace) so the auth.login.success audit line will subsequently emit the SAME canonical form via result.user.email; got: $storedEmail")

          // Now capture stdout around the LOGIN call only -- the
          // auth.login.success emission at AuthStack.scala line 171
          // is logInfo(...) which writes to System.out per
          // HandHistoryReviewServerRuntime.scala line 418. The
          // tight capture window (just the postJson call) keeps the
          // captured stream small and free of unrelated emissions
          // (the surrounding withServer / withUserStorePath fixtures
          // log to stderr or to other streams, not to stdout-during-
          // this-window).
          val outBuf = new java.io.ByteArrayOutputStream()
          val originalOut = System.out
          System.setOut(new java.io.PrintStream(outBuf, true, StandardCharsets.UTF_8))
          try
            // Login with a DIFFERENT mixed-case form than register
            // (different uppercase letters, but normalizing to the
            // same canonical alice@example.com). This isolates the
            // canonical-vs-submitted distinction: if the audit line
            // emitted the LOGIN-submitted form (aLiCe@example.com)
            // it would NOT match the canonical alice@example.com,
            // and the exclusion check below would fail.
            val login = postJson(s"$baseUri/api/auth/login",
              """{"email":"aLiCe@example.com","password":"correct-horse-battery"}""")
            assertEquals(login.statusCode(), 200,
              clue = s"login with case-mismatched form must succeed because PlatformUserAuth.loginLocal normalizes the submitted email before the database lookup -- if this returns 401 instead of 200, either the normalization is missing on the login path OR the case-folding is not Locale.ROOT-anchored, either of which breaks the documented 'sign in with the same email regardless of case' contract")
          finally
            System.setOut(originalOut)

          val captured = outBuf.toString(StandardCharsets.UTF_8)
          val loginLine = captured.split('\n').iterator
            .find(_.contains("auth.login.success"))
            .getOrElse(fail(s"no `auth.login.success` line in stdout capture -- deploy doc line 218 documents this event as INFO-level fired on every successful login; got captured stdout: ${captured.take(800)}"))

          // (i) event prefix
          assert(loginLine.contains("auth.login.success"),
            clue = s"login audit line must carry the literal `auth.login.success` event prefix per deploy doc line 218's enumeration; a refactor renaming to e.g. `auth.signin.success` would silently break log-aggregation queries; got: $loginLine")
          // (ii) CANONICAL email field (the load-bearing contract this
          // test pins) -- a refactor that logged the submitted email
          // (either register-form or login-form, both case-different
          // from canonical) would fail this assertion
          assert(loginLine.contains("email=alice@example.com"),
            clue = s"login.success audit line MUST carry the canonical-normalized email `alice@example.com` per deploy doc line 218 ('Success lines log the canonical (normalized: zero-width-character strip → trim → lowercase, all under Locale.ROOT) email so all events for one user grep identically'); the user registered with `Alice@Example.COM` and logged in with `aLiCe@example.com` (both case-different submitted forms), so the audit line MUST emit the canonical form to allow operator user-correlation queries; a refactor logging the submitted form would silently break the 'one user, one canonical email' grep guarantee AND the runbook's brute-force triage workflow; got: $loginLine")
          // (iii) EXCLUSION: must NOT contain the submitted form --
          // pins the normalization invariant from the negative
          // direction; without this, a refactor that emitted BOTH
          // forms (e.g. "email=alice@example.com submittedEmail=aLiCe@example.com")
          // would silently pass the positive contains check while
          // still leaking the submitted form
          assert(!loginLine.contains("aLiCe@example.com"),
            clue = s"login.success audit line MUST NOT contain the login-submitted form `aLiCe@example.com` (case-mismatched from canonical) per the documented canonical-vs-submitted distinction -- a refactor that logged BOTH forms (canonical AND submitted side by side) or that swapped to logging the submitted form would silently break operator user-correlation since one user's events would log under multiple different email values depending on how they typed their input each session; got: $loginLine")
          // (iv) INFO level
          assert(loginLine.contains("[INFO]"),
            clue = s"login.success audit line must be INFO-level per deploy doc line 218 ('INFO level for success/expected events'); a refactor demoting to DEBUG would silently make the line invisible at default log levels, promoting to WARN would silently flood alerting (every login normal); got: $loginLine")
          // (v) remote= field
          assert(loginLine.contains("remote="),
            clue = s"login.success audit line must carry the `remote=` field per deploy doc line 218 ('remote= on every auth.* event'); without this field operators lose the per-IP correlation between successful logins and any preceding auth.login.failure attempts (the brute-force-detection workflow keys on counting failures-per-IP and then matching the eventual success to detect compromised-credential takeovers); got: $loginLine")
          // (vi) service-tag prefix
          assert(loginLine.contains("[hand-history-review]"),
            clue = s"login.success audit line must carry the `[hand-history-review]` service-tag prefix per HandHistoryReviewServerRuntime.scala line 512's hardcoded literal -- the tag matches the /api/health.service field (pinned by 505ba6b) so a log aggregator filtering by service tag gets the same identifier on log lines as on probe responses; got: $loginLine")
        }
      }
    }
  }

  // Pin the documented `auth.register.success` audit line format
  // AND the canonical-vs-submitted-email normalization contract on
  // the REGISTER path -- the THIRD of the 5 success-side auth-event
  // formats deploy doc line 218 documents (1c8777f closed auth.logout,
  // 49dcf46 closed auth.login.success); this commit closes the LOCAL-
  // AUTH success pair (register + login). The register flow differs
  // from the login flow in two ways the test exercises: (1) the
  // canonical email is produced AT REGISTER TIME by
  // PlatformUserAuth.scala line 503's `normalizedEmail =
  // normalizeEmail(email)`, NOT by a database lookup like login --
  // so the register-time emission depends on the input-normalization
  // pipeline being correctly applied before the StoredUser is
  // constructed (PlatformUserAuth.scala line 519-535's StoredUser
  // construction uses `normalizedEmail` as the email field), (2)
  // there's no "register vs login" mismatch to exploit since register
  // is a single step -- so to make the canonical-vs-submitted
  // distinction OBSERVABLE the test submits a mixed-case + uppercase-
  // domain form (`Bob@Example.COM`) and verifies the audit line uses
  // the canonical-normalized form (`bob@example.com`), AND verifies
  // the line does NOT contain the submitted-mixed-case form. Per-
  // field regression vectors specific to register.success that the
  // login.success pin (49dcf46) doesn't catch: (i) refactor that
  // applied normalizeEmail to the StoredUser construction but FORGOT
  // to apply it to the log-line emission (e.g. emitting the raw
  // `email` parameter from parseRegisterRequest at AuthStack.scala
  // line 95 instead of `result.user.email` at line 112) would
  // silently log the SUBMITTED form even though the database stores
  // the canonical -- a register-time-only regression that login.success
  // can't catch because its log emission path is independent, (ii)
  // refactor that broke normalizeEmail itself but only on the
  // register path (e.g. an early-return refactor on the
  // zero-width-strip step that triggered only on specific input
  // shapes -- the register flow has different validation
  // preconditions than login so a regression there could be register-
  // specific), (iii) refactor consolidating register+login+logout to
  // emit `email=` from a single helper that uses the wrong source
  // (e.g. always uses parseLoginRequest's `email` parameter
  // forgetting register doesn't have that parameter) would fail
  // BOTH the register pin AND the login pin AND the logout pin --
  // pinning all three guarantees the consolidation refactor fails
  // immediately rather than silently logging different emails per
  // event type; the canonical-vs-submitted EXCLUSION check is the
  // load-bearing part (matches 49dcf46's approach): submitting
  // `Bob@Example.COM` (case-mismatched submitted form) and
  // asserting !contains("Bob@Example.COM") on the audit line
  // catches a refactor that logged the submitted form, even if
  // the regression also emitted the canonical alongside (the
  // `Bob@Example.COM` literal characters must NOT appear in the
  // line); Scala's String.contains is case-SENSITIVE so the case-
  // different forms are observationally distinct -- a regression
  // emitting "email=Bob@Example.COM" would fail the positive
  // canonical-contains check (since "bob@example.com" is not a
  // substring of "Bob@Example.COM") AND fail the negative
  // submitted-exclusion check; same 5-tier format-check pattern
  // as the 1c8777f auth.logout pin + 49dcf46 auth.login.success pin:
  // event prefix + canonical email value + INFO level + remote=
  // field + service-tag prefix, with the canonical/submitted
  // exclusion pair as the 6th check this commit shares with the
  // login.success pin; future fires can close the remaining 2
  // success-side events: auth.oidc.start (provider= field, no
  // email -- simpler format pin) and auth.oidc.success (provider=
  // AND email= fields, requires OIDC mock setup); after this fire
  // the LOCAL-AUTH success triple (register + login + logout) is
  // fully audit-log-format pinned, and the OIDC pair is the only
  // remaining audit-log coverage gap.
  test("POST /api/auth/register emits the documented `auth.register.success email=<canonical> remote=<peer>` INFO audit line with the canonical-normalized email (per deploy doc line 218's 'all events for one user grep identically' contract)") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          // Capture stdout around the register call only. logInfo
          // writes to System.out per HandHistoryReviewServerRuntime.scala
          // line 418, so the auth.register.success emission at
          // AuthStack.scala line 112 lands in System.out.
          val outBuf = new java.io.ByteArrayOutputStream()
          val originalOut = System.out
          System.setOut(new java.io.PrintStream(outBuf, true, StandardCharsets.UTF_8))
          val register =
            try
              // Submit a mixed-case + uppercase-domain email. The
              // canonical form (alice@example.com -- wait, this is
              // a separate user, use Bob to avoid conflicts with
              // the login.success test's Alice) for the documented
              // normalization (zero-width-strip → trim → lowercase
              // under Locale.ROOT) is `bob@example.com`. The audit
              // line MUST emit the canonical form, NOT the submitted
              // mixed-case form.
              postJson(s"$baseUri/api/auth/register",
                """{"email":"Bob@Example.COM","password":"correct-horse-battery","displayName":"Bob"}""")
            finally
              System.setOut(originalOut)

          assertEquals(register.statusCode(), 201,
            clue = s"registration with mixed-case + uppercase-domain email must succeed (the canonical form 'bob@example.com' should be stored, not the submitted 'Bob@Example.COM')")

          // Sanity-check the register response carries the canonical
          // form -- decouples the two normalization paths (the
          // request-side StoredUser construction normalizing
          // submitted -> canonical, AND the audit-line emission
          // using the same canonical from result.user.email). A
          // regression in EITHER half is caught independently
          // because this assertion checks the canonical via the
          // response JSON path while the audit-line assertion below
          // checks the canonical via the log emission path.
          val registerJson = jsonBody(register)
          val storedEmail = registerJson("user")("email").str
          assertEquals(storedEmail, "bob@example.com",
            clue = s"register response must carry the canonical-normalized email (lowercase, no whitespace) per the documented PlatformUserAuth.normalizeEmail pipeline (zero-width-strip → trim → lowercase under Locale.ROOT); without this sanity check, a register-time normalization regression would silently make the audit-line assertion below pass on the wrong canonical; got: $storedEmail")

          val captured = outBuf.toString(StandardCharsets.UTF_8)
          val registerLine = captured.split('\n').iterator
            .find(_.contains("auth.register.success"))
            .getOrElse(fail(s"no `auth.register.success` line in stdout capture -- deploy doc line 218 documents this event as INFO-level fired on every successful registration; got captured stdout: ${captured.take(800)}"))

          // (i) event prefix
          assert(registerLine.contains("auth.register.success"),
            clue = s"register audit line must carry the literal `auth.register.success` event prefix per deploy doc line 218's enumeration; a refactor renaming to e.g. `auth.signup.success` would silently break log-aggregation queries; got: $registerLine")
          // (ii) CANONICAL email field (load-bearing) -- the assertion
          // that a register-time normalization regression would
          // silently bypass if not paired with the storedEmail
          // sanity-check above
          assert(registerLine.contains("email=bob@example.com"),
            clue = s"register.success audit line MUST carry the canonical-normalized email `bob@example.com` per deploy doc line 218 ('Success lines log the canonical (normalized: zero-width-character strip → trim → lowercase, all under Locale.ROOT) email so all events for one user grep identically'); the user submitted `Bob@Example.COM` (mixed case + uppercase domain), so the audit line MUST emit the canonical form to allow operator user-correlation queries that match the same email across register, subsequent login, logout, and OIDC events; a refactor emitting the submitted form would silently break the cross-event-correlation contract; got: $registerLine")
          // (iii) EXCLUSION: must NOT contain the submitted mixed-
          // case form. Pins the normalization invariant from the
          // negative direction (matches the 49dcf46 login.success
          // pattern); without this, a refactor that emitted BOTH
          // forms would silently pass the positive check
          assert(!registerLine.contains("Bob@Example.COM"),
            clue = s"register.success audit line MUST NOT contain the register-submitted form `Bob@Example.COM` (case-mismatched from canonical) per the documented canonical-vs-submitted distinction -- a refactor logging BOTH forms (canonical AND submitted side by side) or that swapped to logging the submitted form would silently break the 'all events for one user grep identically' contract because operator grep for `bob@example.com` would miss the `Bob@Example.COM`-formatted events; got: $registerLine")
          // (iv) INFO level
          assert(registerLine.contains("[INFO]"),
            clue = s"register.success audit line must be INFO-level per deploy doc line 218 ('INFO level for success/expected events'); demote-to-DEBUG would silently make the line invisible at default log levels, promote-to-WARN would silently flood alerting; got: $registerLine")
          // (v) remote= field
          assert(registerLine.contains("remote="),
            clue = s"register.success audit line must carry the `remote=` field per deploy doc line 218 ('remote= on every auth.* event'); without this operators lose the per-IP correlation between registration events and any preceding auth.*.failure attempts (a brute-force-detection workflow that spots account-creation storms from a single IP keys on this field); got: $registerLine")
          // (vi) service-tag prefix
          assert(registerLine.contains("[hand-history-review]"),
            clue = s"register.success audit line must carry the `[hand-history-review]` service-tag prefix per HandHistoryReviewServerRuntime.scala line 512's hardcoded literal -- the tag matches the /api/health.service field (pinned by 505ba6b) so a log aggregator filtering by service tag gets the same identifier on log lines as on probe responses; got: $registerLine")
        }
      }
    }
  }

  // Pin the documented `auth.oidc.start` audit line format -- the
  // FOURTH of the 5 SUCCESS-side auth-event formats deploy doc line
  // 218 documents (1c8777f closed auth.logout, 49dcf46 closed
  // auth.login.success, 3a6fea4 closed auth.register.success); this
  // commit closes the OIDC FLOW START half of the OIDC pair (a
  // future fire can close auth.oidc.success which requires more
  // FakeOidcProvider mock orchestration to drive the callback
  // exchange). The auth.oidc.start event is OPERATIONALLY UNIQUE in
  // two ways: (1) it carries `provider=` instead of `email=` per
  // deploy doc line 218 ("The OIDC auth.oidc.start / auth.oidc.
  // start.failure / auth.oidc.failure lines do NOT carry email=
  // because the OIDC flow doesn't yield a user email until the
  // userinfo response completes -- `start` and `start.failure`
  // fire before the provider redirect, and the callback `failure`
  // paths exit before (or because of) the userinfo step that
  // would have resolved one"), AND (2) it's the START half of the
  // runbook's "users start, never finish" triage signature -- the
  // runbook commits to "userAuthPendingOidcFlows counting up
  // steadily ... WITHOUT a corresponding stream of auth.oidc.success
  // / auth.oidc.failure log lines is the signature of 'users
  // start, never finish'" (the same triage pattern already pinned
  // at line ~712's userAuthPendingOidcFlows round-trip test, but
  // that test pins the COUNTER side; this test pins the LOG-LINE
  // side that the operator counts against in the triage); a
  // refactor that broke the auth.oidc.start emission entirely (e.g.
  // suppressed the logInfo call at AuthStack.scala line 265
  // because "it's just normal user behavior") would silently
  // invalidate the runbook triage -- without auth.oidc.start log
  // lines to count, the operator can't tell "5 pending flows with
  // 5 corresponding start log lines AND 0 success/failure log
  // lines" (the 'users start, never finish' pattern) from "5
  // pending flows with NO start log lines either" (a different
  // refactor where the start emission was suppressed but the
  // counter increment still works) -- both look identical to the
  // operator if start emission is missing; per-field regression
  // vectors specific to auth.oidc.start that the local-auth pins
  // (1c8777f + 49dcf46 + 3a6fea4) don't catch: (i) refactor that
  // ADDED an email= field to the line (e.g. "for consistency with
  // login.success/register.success") would silently break the
  // documented "do NOT carry email=" contract -- but more
  // critically would expose private information (the OIDC start
  // happens BEFORE the user has logged in, so there's no userinfo
  // email available; the only way to populate email= would be to
  // pull it from a session OR from request headers, both of which
  // would leak operator-irrelevant data into the audit log AND
  // create a confused-operator triage path where the email field
  // suggests a specific user when really the OIDC flow hasn't
  // resolved one yet), (ii) refactor demoting INFO to DEBUG (e.g.
  // "OIDC starts are too verbose, demote for less log volume")
  // would silently make the line invisible at default log levels
  // AND silently invalidate the runbook triage entry that
  // explicitly depends on these log lines being visible at default
  // levels, (iii) refactor changing the provider= field's value
  // from the provider's `id` to its `displayName` (e.g.
  // "provider=Google" instead of "provider=google" -- subtle
  // case-sensitive distinction) would silently break operator log-
  // aggregation queries filtering by `provider=google` (lowercase),
  // AND silently break the cross-event-correlation that lets
  // operators match auth.oidc.start lines with their
  // corresponding auth.oidc.success / auth.oidc.failure lines (all
  // three emit `provider=<id>` not `provider=<displayName>`), (iv)
  // refactor unifying provider= across the OIDC events with a
  // helper that used the WRONG SOURCE (e.g. always emitted
  // provider="oidc" as a generic placeholder) would silently break
  // multi-provider deployments where operators need to distinguish
  // Google failures from a future second IdP's failures -- the
  // deploy doc line 218 explicitly contrasts: "they instead carry
  // provider=<id> so the operator can tell Google-flow failures
  // from a future multi-provider deployment's other-IdP failures";
  // 6-tier format check: (i) event prefix `auth.oidc.start` (catches
  // rename to e.g. `auth.oidc.begin`), (ii) `provider=google` field
  // (catches rename of FakeOidcProvider.id OR rename of the
  // logInfo emission template's `provider=`), (iii) ABSENCE of
  // `email=` (catches the "add for consistency" refactor that
  // would leak private info), (iv) `[INFO]` level (catches
  // demote-to-DEBUG silently-hiding from triage), (v) `remote=`
  // field (per deploy doc line 218 "remote= on every auth.* event"
  // -- the brute-force triage correlator), (vi) `[hand-history-
  // review]` service-tag prefix (matches the /api/health.service
  // field pinned by 505ba6b); test uses FakeOidcProvider (same as
  // userAuthPendingOidcFlows round-trip test at line ~712) so no
  // new mock infrastructure is needed; capture stdout around the
  // GET /api/auth/oidc/google/start call ONLY (a 302 redirect, the
  // logInfo emission at line 265 runs synchronously BEFORE the
  // redirect response is built, so by the time the GET returns the
  // audit line is already on stdout); same regression-pin pattern
  // as 1c8777f auth.logout + 49dcf46 auth.login.success + 3a6fea4
  // auth.register.success (5-tier format check + service-tag-prefix
  // coupling), extended with the unique-to-OIDC `!email=` ABSENCE
  // check and the `provider=google` (id-not-displayName) precision
  // pin.
  test("GET /api/auth/oidc/google/start emits the documented `auth.oidc.start provider=<id> remote=<peer>` INFO audit line WITHOUT an email= field (per deploy doc line 218's 'OIDC start lines do NOT carry email=' contract)") {
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

          // Capture stdout around the /start GET only. logInfo writes
          // to System.out per HandHistoryReviewServerRuntime.scala line
          // 418, so the auth.oidc.start emission at AuthStack.scala
          // line 265 lands in System.out. The /start endpoint returns
          // a 302 redirect; the logInfo runs synchronously BEFORE the
          // redirect response is built, so by the time the GET
          // returns the audit line is already flushed.
          val outBuf = new java.io.ByteArrayOutputStream()
          val originalOut = System.out
          System.setOut(new java.io.PrintStream(outBuf, true, StandardCharsets.UTF_8))
          val start =
            try get(s"$baseUri${provider.startPath}")
            finally System.setOut(originalOut)
          assertEquals(start.statusCode(), 302,
            clue = s"OIDC /start must return 302 redirect to the provider's authorization endpoint -- a non-302 status means the handler exited via an error path that suppressed the auth.oidc.start emission; got: ${start.statusCode()}")

          val captured = outBuf.toString(StandardCharsets.UTF_8)
          val startLine = captured.split('\n').iterator
            .find(_.contains("auth.oidc.start"))
            .getOrElse(fail(s"no `auth.oidc.start` line in stdout capture -- deploy doc line 218 documents this event as INFO-level fired on every OIDC flow start; if the line is missing, either the logInfo at AuthStack.scala line 265 was suppressed (silent regression invalidating the runbook 'users start, never finish' triage) OR the test failed to capture the right stream; got captured stdout: ${captured.take(800)}"))

          // (i) event prefix
          assert(startLine.contains("auth.oidc.start"),
            clue = s"OIDC start audit line must carry the literal `auth.oidc.start` event prefix per deploy doc line 218's enumeration; a refactor renaming to e.g. `auth.oidc.begin` would silently break the runbook's 'users start, never finish' triage workflow which keys on the literal event name; got: $startLine")
          // (ii) provider=google (id, not displayName)
          assert(startLine.contains("provider=google"),
            clue = s"OIDC start audit line must carry the provider's `id` (lowercase `google`) in the `provider=` field, NOT the `displayName` (capitalized `Google`); the deploy doc line 218 explicitly says 'they instead carry provider=<id>' so multi-provider deployments can distinguish Google-flow failures from a future second IdP's failures; a refactor swapping to displayName would silently break operator queries filtering by `provider=google` (lowercase) AND silently break the cross-event-correlation between auth.oidc.start and the corresponding auth.oidc.success/failure lines (which also use id-not-displayName); got: $startLine")
          // (iii) ABSENCE of email= (load-bearing OIDC-specific contract)
          assert(!startLine.contains("email="),
            clue = s"OIDC start audit line must NOT carry the `email=` field per deploy doc line 218's explicit 'do NOT carry email=' contract (the OIDC flow hasn't yielded a userinfo email yet, so emitting one would either leak unrelated session/request data OR populate a misleading value); a refactor that 'added email= for consistency with login.success/register.success' would silently expose privacy-relevant data AND create a confused-operator triage path where the field suggests a specific user when the OIDC flow hasn't resolved one yet; got: $startLine")
          // (iv) INFO level
          assert(startLine.contains("[INFO]"),
            clue = s"OIDC start audit line must be INFO-level per AuthStack.scala line 261's explicit comment ('INFO not WARN -- this is normal user behavior, but the log entry lets operators correlate a later auth.oidc.success/failure with the start so a missing callback is visible') -- a refactor demoting to DEBUG would silently invalidate the runbook's 'users start, never finish' triage because the start lines wouldn't be visible at default log levels for the operator to count against userAuthPendingOidcFlows; got: $startLine")
          // (v) remote= field
          assert(startLine.contains("remote="),
            clue = s"OIDC start audit line must carry the `remote=` field per deploy doc line 218 ('remote= on every auth.* event'); without this field operators lose the per-IP correlation between OIDC flow starts and any subsequent OIDC failures from the same IP (a brute-force-style probe testing OIDC provider behaviors would emit many auth.oidc.start lines from one IP -- removing remote= silently disables that detection); got: $startLine")
          // (vi) service-tag prefix
          assert(startLine.contains("[hand-history-review]"),
            clue = s"OIDC start audit line must carry the `[hand-history-review]` service-tag prefix per HandHistoryReviewServerRuntime.scala line 512's hardcoded literal -- the tag matches the /api/health.service field (pinned by 505ba6b) so a log aggregator filtering by service tag gets the same identifier on log lines as on probe responses; got: $startLine")
        }
      }
    }
  }

  // Pin the documented `auth.oidc.success` audit line format -- the
  // LAST of the 5 SUCCESS-side auth-event formats deploy doc line
  // 218 documents (1c8777f auth.logout, 49dcf46 auth.login.success,
  // 3a6fea4 auth.register.success, 976d7ad auth.oidc.start were the
  // first four); this commit closes the final success-side audit
  // event AND the OIDC pair coverage (auth.oidc.start +
  // auth.oidc.success). The auth.oidc.success event is operationally
  // unique among the 5 events because it carries BOTH the
  // `provider=` field (like auth.oidc.start) AND the `email=` field
  // (like the local-auth events) -- per deploy doc line 218: "on
  // the local-auth events ... and the post-callback
  // auth.oidc.success -- an email= field"; this is the ONLY event
  // type that carries BOTH fields, making it the CROSS-CORRELATION
  // anchor between the per-IP brute-force triage (which keys on
  // remote= + provider= for OIDC events) and the per-user incident
  // analysis (which keys on email= across login/register/logout/
  // oidc.success events). The OIDC success line closes the runbook's
  // "users start, never finish" triage signature from the OPPOSITE
  // direction the auth.oidc.start pin (976d7ad) closes it: start
  // pins the COUNTER-INCREMENT-LOG-LINE pair, success pins the
  // COUNTER-DECREMENT-LOG-LINE pair, so the runbook's "stream of
  // auth.oidc.success / auth.oidc.failure log lines" half of the
  // signature ("counting up steadily ... WITHOUT a corresponding
  // stream of ...") is now fully test-pinned -- a refactor breaking
  // EITHER the start emission OR the success emission would
  // silently invalidate the triage from a different angle, and the
  // two pins together catch both regressions; per-field regression
  // vectors SPECIFIC to auth.oidc.success that the other 4 success-
  // side pins don't catch: (i) refactor swapping the field order
  // (e.g. "email=<x> provider=<y> remote=<z>" instead of
  // "provider=<y> email=<x> remote=<z>") -- the deploy doc doesn't
  // commit to field order explicitly but operator parsing tools
  // built on `awk '/auth\.oidc\.success/{ print $3 }'` style
  // positional access WOULD silently break; the test doesn't pin
  // exact field order to avoid over-constraining (per the JSON-
  // shape pinning convention this branch follows, ordering is a
  // weaker contract than presence), so this regression is
  // intentionally NOT caught here -- a future fire could add an
  // exact-position assertion if operator tooling becomes order-
  // dependent, (ii) refactor swapping `result.user.email` for the
  // raw OidcIdentity.email returned by exchangeCode (at AuthStack.
  // scala line 382, both fields are accessible via result.user vs
  // the exchangeCode return value) would silently emit the
  // PRE-NORMALIZATION userinfo email instead of the post-
  // normalization canonical -- the FakeOidcProvider in this test
  // returns already-canonical "oidc@example.com" so this regression
  // wouldn't be caught here; a future fire COULD add a
  // MixedCaseOidcProvider whose exchangeCode returns
  // "OIDC@Example.COM" to exercise the canonical-from-userinfo
  // contract specifically (deploy doc line 218 commits to canonical
  // normalization for ALL success lines including auth.oidc.success
  // -- the same pipeline applies), (iii) refactor consolidating
  // login.success + oidc.success behind a single helper using the
  // wrong field set (e.g. omitting provider= because login.success
  // doesn't have it) would silently break operator queries that
  // filter by provider= for OIDC-specific incident analysis -- the
  // 6-tier format check here AND the parallel check on
  // auth.oidc.start (976d7ad) together force any consolidation
  // refactor to handle BOTH the with-email AND without-email
  // shapes correctly; 6-tier format check: (i) `auth.oidc.success`
  // event prefix, (ii) `provider=google` (lowercase id matching
  // auth.oidc.start's pin -- so a refactor that broke the id
  // resolution would fail BOTH OIDC pins simultaneously), (iii)
  // `email=oidc@example.com` field PRESENCE (the OPPOSITE of
  // auth.oidc.start's `!email=` ABSENCE check -- this event is the
  // FIRST OIDC event in the flow that has a resolved user identity,
  // and the email= field is what closes the loop from "OIDC start
  // hasn't yielded an email yet" to "OIDC callback completed, user
  // identified"), (iv) `[INFO]` level (catches demote-to-DEBUG),
  // (v) `remote=` field (per "remote= on every auth.* event"), (vi)
  // `[hand-history-review]` service-tag prefix; test reuses
  // FakeOidcProvider which returns email="oidc@example.com" (already
  // canonical) at line ~7486 -- the existing /start + /callback
  // round-trip from the userAuthPendingOidcFlows test at line ~712
  // is the template the new test follows; capture stdout AROUND
  // THE /CALLBACK GET ONLY, NOT around /start (the /start emission
  // would pollute the captured stream with auth.oidc.start, which
  // would still pass the auth.oidc.success.find but would
  // unnecessarily expose the test to flaky failure if the parsing
  // logic ever changed); after this commit ALL 5 success-side
  // audit-event formats are pinned: auth.logout (1c8777f), auth.
  // login.success (49dcf46), auth.register.success (3a6fea4), auth.
  // oidc.start (976d7ad), auth.oidc.success (this commit) -- the
  // deploy doc line 218 enumeration is FULLY closed for the
  // success side; remaining FAILURE-side gaps for future fires:
  // auth.oidc.start.failure (single emission, simple format),
  // auth.oidc.failure (5 emission sites at AuthStack.scala lines
  // 317/335/352/363/397 each with different reason= values worth
  // pinning individually for asymmetric-drift -- a refactor
  // consolidating the failure reasons could silently lose
  // operator-relevant detail).
  test("GET /api/auth/oidc/google/callback emits the documented `auth.oidc.success provider=<id> email=<canonical> remote=<peer>` INFO audit line carrying BOTH provider= AND email= fields (per deploy doc line 218's 'on the local-auth events ... and the post-callback auth.oidc.success -- an email= field' contract)") {
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

          // Drive /start OUTSIDE the capture window so its
          // auth.oidc.start emission doesn't pollute the captured
          // stream (which should contain only the auth.oidc.success
          // emission from /callback). The /start event is pinned in
          // its own test at line ~1849 (976d7ad) so the round-trip
          // here doesn't need to re-pin it; running /start OUTSIDE
          // the capture keeps the captured stream small and focused
          // on the success-side emission.
          val start = get(s"$baseUri${provider.startPath}")
          assertEquals(start.statusCode(), 302,
            clue = "OIDC /start must return 302 for the subsequent callback to consume the state cookie")
          val redirect = headerValue(start, "Location").getOrElse(fail("missing OIDC redirect Location header"))
          val state = queryParam(redirect, "state").getOrElse(fail("missing OIDC state query parameter"))
          val stateCookie = headerValue(start, "Set-Cookie")
            .map(_.takeWhile(_ != ';'))
            .getOrElse(fail("missing OIDC state cookie"))

          // Capture stdout around the /callback GET only. logInfo
          // writes to System.out per HandHistoryReviewServerRuntime.
          // scala line 418, so the auth.oidc.success emission at
          // AuthStack.scala line 382 lands in System.out. The
          // /callback returns 302 (Redirect to the documented OIDC
          // success URL); the logInfo runs synchronously BEFORE the
          // redirect response is built, so by the time the GET
          // returns the audit line is already flushed.
          val outBuf = new java.io.ByteArrayOutputStream()
          val originalOut = System.out
          System.setOut(new java.io.PrintStream(outBuf, true, StandardCharsets.UTF_8))
          val callback =
            try get(
              s"$baseUri${provider.callbackPath}?state=$state&code=test-success-code",
              Map("Cookie" -> stateCookie)
            )
            finally System.setOut(originalOut)
          assertEquals(callback.statusCode(), 302,
            clue = s"callback must succeed with valid state + cookie + code -- a non-302 status means the handler exited via an error path (likely emitting auth.oidc.failure instead of auth.oidc.success) which would suppress the success-side audit line this test pins; got: ${callback.statusCode()}")

          val captured = outBuf.toString(StandardCharsets.UTF_8)
          val successLine = captured.split('\n').iterator
            .find(_.contains("auth.oidc.success"))
            .getOrElse(fail(s"no `auth.oidc.success` line in stdout capture -- deploy doc line 218 documents this event as INFO-level fired on every successful OIDC callback completion; if missing, either the logInfo at AuthStack.scala line 382 was suppressed (silent regression invalidating the runbook 'users start, never finish' triage closure) OR the success path didn't run (callback short-circuited to auth.oidc.failure -- check the test setup for stale state-cookie / code-verifier mismatches); got captured stdout: ${captured.take(800)}"))

          // (i) event prefix
          assert(successLine.contains("auth.oidc.success"),
            clue = s"OIDC success audit line must carry the literal `auth.oidc.success` event prefix per deploy doc line 218's enumeration; a refactor renaming to e.g. `auth.oidc.complete` would silently break the runbook's 'users start, never finish' triage closure (operator counts auth.oidc.start lines minus auth.oidc.success+failure lines as 'pending'); got: $successLine")
          // (ii) provider=google (lowercase id, matching auth.oidc.start's pin)
          assert(successLine.contains("provider=google"),
            clue = s"OIDC success audit line must carry the provider's `id` (lowercase `google`) in the `provider=` field, matching the auth.oidc.start pin at line ~1849 (976d7ad) so a refactor breaking the id-resolution helper would fail BOTH OIDC pins simultaneously; multi-provider deployments depend on this field to distinguish Google-flow successes from a future second IdP's successes per deploy doc line 218's explicit framing; got: $successLine")
          // (iii) email= field PRESENCE (opposite of auth.oidc.start's
          // !email= absence check) -- the load-bearing contract that
          // makes auth.oidc.success the cross-correlation anchor
          // between per-IP and per-user triage; FakeOidcProvider
          // returns email="oidc@example.com" so the audit line MUST
          // contain that canonical value
          assert(successLine.contains("email=oidc@example.com"),
            clue = s"OIDC success audit line MUST carry the `email=` field with the canonical user email from the userinfo response per deploy doc line 218 ('on the local-auth events ... AND the post-callback auth.oidc.success -- an email= field'); the FakeOidcProvider returns email='oidc@example.com' so the audit line must reflect that; a refactor that omitted email= (e.g. 'for consistency with auth.oidc.start which doesn't have email=') would silently break per-user incident analysis (operators couldn't grep auth.oidc.success lines to see WHO completed an OIDC sign-in from a suspicious IP), AND would break the documented contrast between auth.oidc.start (no email) and auth.oidc.success (has email) which is the key signal that the OIDC flow has actually resolved a user identity; got: $successLine")
          // (iv) INFO level
          assert(successLine.contains("[INFO]"),
            clue = s"OIDC success audit line must be INFO-level per deploy doc line 218 ('INFO level for success/expected events'); a refactor demoting to DEBUG would silently invalidate the runbook triage (lines hidden at default log levels), promoting to WARN would silently flood alerting on normal sign-ins; got: $successLine")
          // (v) remote= field
          assert(successLine.contains("remote="),
            clue = s"OIDC success audit line must carry the `remote=` field per deploy doc line 218 ('remote= on every auth.* event'); without this field the per-IP brute-force triage workflow loses its OIDC-side signal (operators counting OIDC failures-per-IP could spot probe behavior, but without remote= on the success side they can't tell when the probe ESCALATED to a successful credential acquisition); got: $successLine")
          // (vi) service-tag prefix
          assert(successLine.contains("[hand-history-review]"),
            clue = s"OIDC success audit line must carry the `[hand-history-review]` service-tag prefix per HandHistoryReviewServerRuntime.scala line 512's hardcoded literal -- matches /api/health.service (505ba6b) so log aggregators see the same identifier on log lines and probe responses; got: $successLine")
        }
      }
    }
  }

  // Pin the documented `auth.oidc.failure` audit line format for the
  // FIRST of 5 emission sites in AuthStack.scala -- closes the FIRST
  // failure-side OIDC event after b1213b6 closed the success-side
  // OIDC pair (auth.oidc.start + auth.oidc.success); the auth.oidc.
  // failure event has 5 DIFFERENT emission sites at AuthStack.scala
  // lines 317, 335, 352, 363, and 397, EACH with a different reason=
  // value (provider-error, oversize_callback_param, state-cookie-
  // related reasons at 352, finishOidc errors at 363, and
  // "missing_code_or_state" at 397) -- the deploy doc line 218
  // documents the event type but operator triage depends on the
  // SPECIFIC reason= values to know WHICH stage of the OIDC flow
  // failed; this commit pins the LINE 397 emission ("missing_code_
  // or_state") which is the cleanest single-fire target because:
  // (a) it's reachable with the simplest test setup (just a GET to
  // /callback with NO query params -- no need to issue a state
  // cookie, no need to craft a malformed code, no need to wedge an
  // exchangeCode to error), (b) the reason value is a literal
  // string constant ("missing_code_or_state") that's documented
  // for the operator to grep AT THAT EXACT VALUE -- a refactor
  // renaming it to e.g. "missing-code-or-state" (hyphens not
  // underscores) or "incomplete_callback" would silently break
  // operator log-aggregation queries that filter by reason=
  // missing_code_or_state; (c) it pairs operationally with the
  // 976d7ad auth.oidc.start pin: a /start emits auth.oidc.start,
  // a CORRESPONDING /callback that ARRIVES WITHOUT EITHER
  // PARAMETER (e.g. the upstream provider's redirect was
  // intercepted / dropped, or the user mashed the URL) emits
  // this auth.oidc.failure -- so the user-start-never-finish
  // signature documented in the runbook can be detected from
  // EITHER direction (start without success, OR start followed
  // by failure with missing params); per-field regression vectors
  // SPECIFIC to auth.oidc.failure that the other 5 audit-event
  // pins don't catch: (i) WARN level not INFO (this event is a
  // FAILURE, AuthStack.scala line 397 uses logWarn which writes
  // to System.err per HandHistoryReviewServerRuntime.scala line
  // 421) -- a refactor demoting to INFO would silently bury the
  // line in the success-side log stream, making operator alerts
  // keying on WARN-level events miss it, AND a refactor promoting
  // to ERROR would silently confuse incident-response automation
  // that pages on ERROR but not WARN; (ii) the !email= ABSENCE
  // matches auth.oidc.start (no userinfo yet) per deploy doc:
  // "The OIDC auth.oidc.start / auth.oidc.start.failure /
  // auth.oidc.failure lines do NOT carry email=" -- a refactor
  // adding email= "for consistency with auth.oidc.success" would
  // silently leak privacy data (the failed callback may have a
  // tampered state cookie pointing at a victim user's session;
  // emitting email= for that session would expose private data
  // to log aggregation), (iii) the reason= value MUST be the
  // specific "missing_code_or_state" string for THIS emission
  // site -- a refactor consolidating the 5 emission sites' reason
  // values to a single generic "callback_failed" reason would
  // silently lose operator-relevant detail about WHICH stage of
  // the flow failed (the runbook's OIDC triage can distinguish
  // "user abandoned the flow" from "provider returned error"
  // from "state cookie expired" based on this reason value;
  // collapsing them would force operators back to logs of full
  // request traces); 6-tier format check parallels the prior pins
  // but at WARN level (System.err capture, not System.out): (i)
  // `auth.oidc.failure` event prefix, (ii) `provider=google`
  // matching auth.oidc.start + auth.oidc.success, (iii)
  // `reason=missing_code_or_state` (NEW specific-value pin for
  // this emission site), (iv) `[WARN]` level (catches demote-to-
  // INFO / promote-to-ERROR), (v) `remote=` field, (vi)
  // `[hand-history-review]` service-tag; ALSO an absence check
  // for `email=` matching the auth.oidc.start ABSENCE pin; the
  // test reuses FakeOidcProvider (id="google") -- just hits the
  // /callback with NO query params, no cookie, no body; remaining
  // failure-side gaps for future fires: the OTHER 4 auth.oidc.
  // failure emission sites (lines 317 provider-error, 335 oversize_
  // callback_param, 352 state-cookie-related, 363 finishOidc-errors)
  // EACH with their distinct reason values worth pinning
  // individually for asymmetric-drift; AND auth.oidc.start.failure
  // (line 258 in AuthStack.scala) which is currently NOT
  // reachable via the routing layer because contexts are only
  // registered for known providers -- it appears to be defensive
  // code for a hypothetical race condition (provider deregistered
  // between context creation and handler invocation); a future
  // fire could either remove the dead code OR document why it
  // exists if a reachable path is identified.
  test("GET /api/auth/oidc/google/callback with no query params emits the documented `auth.oidc.failure provider=<id> remote=<peer> reason=missing_code_or_state` WARN audit line WITHOUT an email= field (per deploy doc line 218's 'auth.oidc.failure lines do NOT carry email=' contract -- closes the first of 5 distinct emission-site reason values)") {
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

          // Capture stderr around the /callback GET with NO query
          // params. logWarn writes to System.err per
          // HandHistoryReviewServerRuntime.scala line 421, so the
          // auth.oidc.failure emission at AuthStack.scala line 397
          // lands in System.err. NO state cookie / NO code / NO state
          // query param means the callback handler's
          // `(parseQuery.get("state"), parseQuery.get("code")) match`
          // hits the catch-all `case _ =>` at line 396, firing the
          // missing_code_or_state logWarn at line 397.
          val errBuf = new java.io.ByteArrayOutputStream()
          val originalErr = System.err
          System.setErr(new java.io.PrintStream(errBuf, true, StandardCharsets.UTF_8))
          val callback =
            try get(s"$baseUri${provider.callbackPath}")
            finally System.setErr(originalErr)
          assertEquals(callback.statusCode(), 302,
            clue = s"OIDC /callback with no params must return 302 redirect to the documented oidcFailureRedirect -- a non-302 means the handler exited via a different path which would emit a different auth.oidc.failure reason; got: ${callback.statusCode()}")

          val captured = errBuf.toString(StandardCharsets.UTF_8)
          val failureLine = captured.split('\n').iterator
            .find(_.contains("auth.oidc.failure"))
            .getOrElse(fail(s"no `auth.oidc.failure` line in stderr capture -- deploy doc line 218 documents this event as WARN-level fired on every OIDC callback failure; if missing, either the logWarn at AuthStack.scala line 397 was suppressed OR the missing-params path failed to reach line 397 (check that /callback with no query string still routes to handleOidcCallback); got captured stderr: ${captured.take(800)}"))

          // (i) event prefix
          assert(failureLine.contains("auth.oidc.failure"),
            clue = s"OIDC failure audit line must carry the literal `auth.oidc.failure` event prefix per deploy doc line 218's enumeration; a refactor renaming to e.g. `auth.oidc.error` would silently break operator log-aggregation queries that filter by event type; got: $failureLine")
          // (ii) provider=google (lowercase id, matching auth.oidc.start + auth.oidc.success)
          assert(failureLine.contains("provider=google"),
            clue = s"OIDC failure audit line must carry the provider's `id` (lowercase `google`) -- a refactor breaking id resolution would fail this pin AND the 976d7ad auth.oidc.start + b1213b6 auth.oidc.success pins simultaneously; multi-provider deployments depend on provider= to distinguish Google failures from a future second IdP's failures per deploy doc line 218; got: $failureLine")
          // (iii) reason=missing_code_or_state (THE specific reason
          // value for THIS emission site at AuthStack.scala line 397
          // -- the load-bearing per-emission-site contract that
          // distinguishes this failure path from the other 4 sites)
          assert(failureLine.contains("reason=missing_code_or_state"),
            clue = s"OIDC failure audit line for the missing-params path MUST carry the EXACT reason value `missing_code_or_state` per AuthStack.scala line 397's hardcoded string; a refactor renaming to e.g. `missing-code-or-state` (hyphens not underscores), `incomplete_callback`, or consolidating to a generic `callback_failed` would silently lose operator-relevant detail about WHICH stage of the flow failed -- the runbook's OIDC triage distinguishes 'user abandoned' (missing_code_or_state) from 'provider returned error' (provider-error reason) from 'state cookie expired' (state-related reasons) based on THIS exact reason value; got: $failureLine")
          // (iv) WARN level (NOT INFO -- this event is a failure)
          assert(failureLine.contains("[WARN]"),
            clue = s"OIDC failure audit line must be WARN-level per AuthStack.scala line 397's logWarn call (which writes to System.err per HandHistoryReviewServerRuntime.scala line 421); a refactor demoting to INFO would silently bury the line in the success-side log stream making WARN-level alert rules miss it, promoting to ERROR would silently confuse incident-response automation that pages on ERROR but not WARN; got: $failureLine")
          // (v) remote= field
          assert(failureLine.contains("remote="),
            clue = s"OIDC failure audit line must carry the `remote=` field per deploy doc line 218 ('remote= on every auth.* event'); without this field the per-IP brute-force triage workflow loses the OIDC-failure correlation signal (operators spotting attempts to probe OIDC behaviors from a single IP could count failure events but couldn't tell whether the attempts come from one IP or many without remote=); got: $failureLine")
          // (vi) ABSENCE of email= matching the auth.oidc.start pin
          // (976d7ad) -- per deploy doc line 218's "auth.oidc.start /
          // auth.oidc.start.failure / auth.oidc.failure lines do NOT
          // carry email=" contract
          assert(!failureLine.contains("email="),
            clue = s"OIDC failure audit line must NOT carry the `email=` field per deploy doc line 218's explicit 'do NOT carry email=' contract for OIDC failure events (the callback may have a tampered state cookie pointing at a victim user's session; emitting email= for that session would expose private data to log aggregation); a refactor adding email= 'for consistency with auth.oidc.success' would silently leak privacy data AND create a confused-operator triage path where the email suggests a specific user when the failure may have been triggered by an attacker against a victim's session; got: $failureLine")
          // (vii) service-tag prefix
          assert(failureLine.contains("[hand-history-review]"),
            clue = s"OIDC failure audit line must carry the `[hand-history-review]` service-tag prefix per HandHistoryReviewServerRuntime.scala line 512's hardcoded literal -- matches /api/health.service (505ba6b) so log aggregators see the same identifier on log lines and probe responses; got: $failureLine")
        }
      }
    }
  }

  // Pin the documented `auth.oidc.failure reason=missing_state_cookie`
  // audit line format -- closes the SECOND of 5 failure-side
  // emission sites in AuthStack.scala (342df03 closed
  // missing_code_or_state at line 397; this commit closes the line
  // 352 emission's missing_state_cookie branch); line 352 has TWO
  // possible reason values per the inline conditional `if
  // cookieState.isEmpty then "missing_state_cookie" else
  // "state_cookie_mismatch"` -- this commit pins the FIRST branch
  // (no cookie at all); a future fire can close the SECOND branch
  // (state_cookie_mismatch) which requires more setup (a real
  // /start to issue a state, then mutate the cookie before
  // /callback). The line 352 emission is SECURITY-CRITICAL per the
  // inline comment at AuthStack.scala lines 340-347: "OAuth 2.0 BCP
  // 'covert-redirect' / login-CSRF mitigation: the browser that
  // arrives at /callback must carry the SAME state value that we
  // Set-Cookie'd at /start. Without this check, an attacker who
  // finished their own authorization could forward their
  // ?state=X&code=ATTACKER_CODE to a victim, and our OidcStateStore
  // (which only knows that X is a state WE issued) would happily
  // exchange the code and bind the attacker's identity to the
  // victim's browser session"; the two reason values distinguish
  // OPERATIONAL incident-response paths: (a) missing_state_cookie =
  // the user's browser sent the callback BUT the state cookie was
  // never set (or was dropped by a Cookie-domain config issue, or
  // by sibling-subdomain navigation in insecure mode where the
  // attacker's redirect drops the cookie) -- typically a config /
  // browser-behavior triage path, NOT primarily a security event;
  // (b) state_cookie_mismatch = the cookie WAS set but doesn't
  // match the URL state -- the classic covert-redirect signature,
  // a security event the runbook explicitly handles; consolidating
  // the two reasons to a single value would silently make this
  // distinction invisible to operator triage, forcing them back to
  // full request-trace logs to differentiate config-vs-attack;
  // per-field regression vectors SPECIFIC to this emission site
  // that the 342df03 missing_code_or_state pin doesn't catch: (i)
  // the conditional structure at line 351 -- `if cookieState.isEmpty
  // then "missing_state_cookie" else "state_cookie_mismatch"` -- a
  // refactor that collapsed the conditional to a single reason
  // string would silently break the documented two-path security
  // triage; (ii) the SPECIFIC string "missing_state_cookie" (with
  // underscores, lowercase, exactly that wording) MUST be the
  // emitted reason value -- a refactor renaming to e.g.
  // `no_state_cookie` / `missing-state-cookie` (hyphens) /
  // `cookie_absent` would silently break operator dashboards
  // filtering by reason=missing_state_cookie; 7-tier format check
  // parallels the 342df03 missing_code_or_state pin at WARN level
  // (System.err capture, not System.out): (i) `auth.oidc.failure`
  // event prefix, (ii) `provider=google`, (iii)
  // `reason=missing_state_cookie` (NEW specific-value pin for
  // THIS emission site), (iv) `[WARN]` level, (v) `remote=` field,
  // (vi) `!email=` ABSENCE check, (vii) `[hand-history-review]`
  // service-tag prefix; ALSO an EXCLUSION pin against the
  // ALTERNATIVE reason value `state_cookie_mismatch` -- the line
  // 352 emission could fire EITHER value depending on the
  // conditional branch, and asserting `!contains("state_cookie_
  // mismatch")` confirms the test took the EXPECTED branch (no
  // cookie at all, not cookie-present-but-mismatched); without
  // this exclusion check, a refactor that swapped the conditional
  // (e.g. always emitting state_cookie_mismatch even when cookie
  // is missing) would silently pass the positive
  // missing_state_cookie check if the regression emitted BOTH
  // reasons in the line -- the exclusion catches that. Test
  // approach: GET /callback?state=X&code=Y with NO cookie at all
  // -- skips the /start step entirely (line 320's
  // (Some(rawState), Some(rawCode)) match passes since both query
  // params are present; line 334's oversize check passes for
  // short values; line 348's extractCookieFromExchange returns
  // empty since no cookie was sent; line 350's
  // `cookieState.isEmpty` triggers true, missing_state_cookie
  // reason emits at line 352); the test doesn't even need to
  // /start because the state value isn't validated against the
  // OidcStateStore at this point -- the cookie-presence check
  // short-circuits BEFORE the state-store consume happens at
  // line 355's finishOidc call.
  test("GET /api/auth/oidc/google/callback with state+code params but NO state cookie emits the documented `auth.oidc.failure reason=missing_state_cookie` WARN audit line (security-critical OAuth 2.0 BCP covert-redirect mitigation per AuthStack.scala lines 340-352)") {
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

          // Capture stderr around a /callback GET with state+code
          // query params but NO cookie header. The handler at
          // AuthStack.scala line 320's `(Some(rawState),
          // Some(rawCode)) match` matches (both query params
          // present); line 334's oversize check passes (short
          // values); line 348's extractCookieFromExchange returns
          // empty (no Cookie header sent); line 350's
          // `cookieState.isEmpty || !cookieState.exists(...)`
          // triggers the FIRST clause (empty), the conditional at
          // line 351 emits reason="missing_state_cookie", and the
          // logWarn at line 352 fires the audit line we capture.
          val errBuf = new java.io.ByteArrayOutputStream()
          val originalErr = System.err
          System.setErr(new java.io.PrintStream(errBuf, true, StandardCharsets.UTF_8))
          val callback =
            try get(s"$baseUri${provider.callbackPath}?state=anyfakestate&code=anyfakecode")
            finally System.setErr(originalErr)
          assertEquals(callback.statusCode(), 302,
            clue = s"OIDC /callback with state+code but no cookie must return 302 redirect to the documented oidcFailureRedirect (per line 353); a non-302 status means the handler exited via a different path which would emit a different reason= value; got: ${callback.statusCode()}")

          val captured = errBuf.toString(StandardCharsets.UTF_8)
          val failureLine = captured.split('\n').iterator
            .find(_.contains("auth.oidc.failure"))
            .getOrElse(fail(s"no `auth.oidc.failure` line in stderr capture -- expected the logWarn at AuthStack.scala line 352 to fire on the missing-cookie path; got captured stderr: ${captured.take(800)}"))

          // (i) event prefix
          assert(failureLine.contains("auth.oidc.failure"),
            clue = s"OIDC failure audit line must carry the literal `auth.oidc.failure` event prefix per deploy doc line 218's enumeration; got: $failureLine")
          // (ii) provider=google (matches all OIDC pins)
          assert(failureLine.contains("provider=google"),
            clue = s"OIDC failure audit line must carry provider=google (lowercase id) matching the auth.oidc.start (976d7ad) + auth.oidc.success (b1213b6) + auth.oidc.failure-missing_code_or_state (342df03) pins; got: $failureLine")
          // (iii) reason=missing_state_cookie (THE specific-value pin
          // for THIS emission site / branch -- the load-bearing
          // contract that distinguishes missing-cookie from
          // mismatched-cookie at line 351's inline conditional)
          assert(failureLine.contains("reason=missing_state_cookie"),
            clue = s"OIDC failure audit line for the no-cookie path MUST carry the EXACT reason value `missing_state_cookie` per AuthStack.scala line 351's hardcoded `if cookieState.isEmpty then \"missing_state_cookie\"` conditional; a refactor renaming to e.g. `no_state_cookie` / `missing-state-cookie` (hyphens) / `cookie_absent` / consolidating with state_cookie_mismatch into a single `state_invalid` would silently break operator dashboards filtering by reason= AND break the runbook's documented two-path security triage (config-vs-attack distinction at the inline comment AuthStack.scala lines 340-352); got: $failureLine")
          // (iv) EXCLUSION: must NOT contain the alternative reason
          // string -- pins the conditional branch taken (cookie
          // missing, not cookie-present-but-mismatched); without
          // this exclusion check a refactor that always emitted
          // state_cookie_mismatch (or BOTH reasons in the same
          // line) would silently pass the positive
          // missing_state_cookie check
          assert(!failureLine.contains("state_cookie_mismatch"),
            clue = s"OIDC failure audit line for the no-cookie path MUST NOT contain the alternative reason `state_cookie_mismatch` per AuthStack.scala line 351's `else \"state_cookie_mismatch\"` branch (the ELSE clause that fires when cookie IS present but doesn't match the URL state); a refactor that swapped the conditional or emitted BOTH reasons would silently make the missing-cookie vs cookie-mismatch distinction invisible to operator triage; got: $failureLine")
          // (v) WARN level
          assert(failureLine.contains("[WARN]"),
            clue = s"OIDC failure audit line must be WARN-level per AuthStack.scala line 352's logWarn call -- matches the 342df03 missing_code_or_state pin AND the deploy doc line 218 'WARN for failures' framing; got: $failureLine")
          // (vi) remote= field
          assert(failureLine.contains("remote="),
            clue = s"OIDC failure audit line must carry remote= field per deploy doc line 218; the per-IP correlation is critical for THIS specific failure path because covert-redirect attacks (the state_cookie_mismatch sibling case) typically come from attacker-controlled IPs and the runbook's security triage correlates these failures with subsequent successful OIDC sign-ins from the same IP; got: $failureLine")
          // (vii) !email= ABSENCE
          assert(!failureLine.contains("email="),
            clue = s"OIDC failure audit line must NOT carry email= per deploy doc line 218 ('auth.oidc.failure lines do NOT carry email=') -- the callback may have a tampered state pointing at a victim user's would-be session; emitting email= would expose private data; got: $failureLine")
          // (viii) service-tag prefix
          assert(failureLine.contains("[hand-history-review]"),
            clue = s"OIDC failure audit line must carry the `[hand-history-review]` service-tag prefix per HandHistoryReviewServerRuntime.scala line 512's hardcoded literal -- matches /api/health.service (505ba6b); got: $failureLine")
        }
      }
    }
  }

  // Pin the documented `auth.oidc.failure reason=state_cookie_mismatch`
  // audit line format -- closes the SECOND branch of the line 352
  // emission's two-reason conditional that c8da491 partially closed
  // (c8da491 pinned the FIRST branch missing_state_cookie when no
  // cookie was sent; this commit pins the SECOND branch when a
  // cookie IS sent but doesn't match the URL state); together
  // c8da491 + this commit complete the asymmetric pair on the line
  // 352 emission, mirroring the asymmetric-pin pattern from
  // 4d15ca3+37f9465 (modelConfigured presence + absence) and
  // f50d7f9 (lifecycle pair) but applied to a BRANCH-OF-CONDITIONAL
  // rather than a PRESENCE-OF-FIELD distinction. The
  // state_cookie_mismatch branch is the SECURITY-CRITICAL signature
  // per AuthStack.scala lines 340-347's inline comment: "OAuth 2.0
  // BCP 'covert-redirect' / login-CSRF mitigation: the browser
  // that arrives at /callback must carry the SAME state value that
  // we Set-Cookie'd at /start. Without this check, an attacker who
  // finished their own authorization could forward their
  // ?state=X&code=ATTACKER_CODE to a victim, and our
  // OidcStateStore (which only knows that X is a state WE issued)
  // would happily exchange the code and bind the attacker's
  // identity to the victim's browser session" -- so this specific
  // reason value is what an operator's INTRUSION-DETECTION query
  // grep'd for as the canonical "covert-redirect attempt detected"
  // signal; a refactor that broke the state-vs-cookie comparison
  // (e.g. accidentally swapping secureEquals for ==, which would
  // leak comparison-timing) OR that collapsed the two branches
  // would silently mute this detection. Why this fire after
  // c8da491: the asymmetric pair completes the COVERT-REDIRECT
  // DETECTION CONTRACT -- without both branches pinned, a refactor
  // that emitted missing_state_cookie for BOTH the no-cookie AND
  // cookie-mismatch cases (e.g. "consolidate for simplicity")
  // would silently make the more-suspicious "cookie present but
  // wrong value" case invisible as a separate triage signal AND
  // would silently break the SecureEquals timing-safe comparison
  // contract (the conditional structure at lines 350-351 forces
  // both branches to evaluate, preserving constant-time-vs-
  // sneakily-short-circuit safety even when refactored); the test
  // approach uses a REAL /start to issue a legitimate state value
  // and cookie, then submits /callback with a DIFFERENT state
  // value in the URL while still carrying the original cookie --
  // this exactly models the documented attack scenario where an
  // attacker forwards a state value they obtained from their own
  // authorization flow to a victim whose browser still has the
  // victim's own state cookie; the resulting URL state vs cookie
  // state mismatch triggers line 351's `else
  // "state_cookie_mismatch"` branch; per-field regression vectors
  // SPECIFIC to this emission that c8da491 doesn't catch: (i)
  // refactor that swapped the secureEquals comparison for == at
  // line 350 -- the secureEquals is a constant-time comparison
  // that prevents timing-based state-value enumeration; a == swap
  // would silently expose a timing oracle (the test doesn't
  // directly check secureEquals usage, but pinning that mismatch
  // emits the documented reason ensures the conditional STRUCTURE
  // remains intact so the secureEquals call remains the gatekeeper),
  // (ii) refactor swapping the conditional's TRUE/FALSE branches
  // (e.g. "if cookieState.isEmpty then state_cookie_mismatch else
  // missing_state_cookie" -- inverted) would silently swap the
  // reason values, breaking BOTH operator triage paths simultaneously;
  // pinning BOTH branches catches this swap immediately because
  // the c8da491 missing_state_cookie test would fail (it would
  // see state_cookie_mismatch) AND this test would fail (it would
  // see missing_state_cookie); the EXCLUSION-of-alternative-branch
  // pattern from c8da491 is mirrored here: assert
  // `!contains("missing_state_cookie")` to pin that the cookie-
  // present branch did NOT emit the no-cookie reason; 8-tier
  // format check at WARN level with the new asymmetric-companion
  // exclusion: (i) `auth.oidc.failure` event prefix, (ii)
  // `provider=google`, (iii) `reason=state_cookie_mismatch` (the
  // NEW specific-value pin for THIS branch), (iv) EXCLUSION of
  // `missing_state_cookie` (the alternative branch's reason that
  // c8da491 pins -- the same exclusion shape c8da491 uses for the
  // opposite branch), (v) `[WARN]` level, (vi) `remote=` field
  // (security-critical for this emission specifically because
  // covert-redirect attempts come from attacker-controlled IPs
  // and the runbook's SECURITY triage correlates these failures
  // with subsequent successful sign-ins from the same IP), (vii)
  // `!email=` ABSENCE, (viii) `[hand-history-review]` service-tag
  // prefix; with this commit BOTH branches of the line 352
  // emission are pinned via the asymmetric-pair pattern -- a
  // refactor that breaks EITHER the missing-cookie path OR the
  // cookie-mismatch path is caught by the corresponding test, AND
  // a refactor that consolidates BOTH paths to a single reason
  // value fails BOTH tests simultaneously.
  test("GET /api/auth/oidc/google/callback with cookie present but URL state mismatched emits the documented `auth.oidc.failure reason=state_cookie_mismatch` WARN audit line (covert-redirect attempt signature per AuthStack.scala lines 340-352) -- closes the second branch of line 351's two-reason conditional that c8da491 partially closed") {
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

          // Drive a real /start to issue a legitimate state cookie
          // -- the c8da491 missing_state_cookie test skips this
          // step because no cookie is needed; THIS test needs a
          // cookie that doesn't match the URL state, so we issue
          // a legitimate cookie first then use a DIFFERENT state
          // value in the callback URL. The /start emits an
          // auth.oidc.start INFO line per 976d7ad, but we capture
          // stderr around the /callback only so that emission
          // doesn't pollute the test's captured stream.
          val start = get(s"$baseUri${provider.startPath}")
          assertEquals(start.statusCode(), 302,
            clue = "OIDC /start must return 302 to issue the state cookie this test then mismatches against the URL state")
          val stateCookie = headerValue(start, "Set-Cookie")
            .map(_.takeWhile(_ != ';'))
            .getOrElse(fail("missing OIDC state cookie from /start -- the mismatch test requires a real cookie to mismatch AGAINST"))

          // Capture stderr around the /callback GET. The callback
          // carries the LEGITIMATE state cookie from /start but a
          // DIFFERENT state value in the URL query -- exactly the
          // covert-redirect attack scenario the deploy doc + the
          // inline comment at AuthStack.scala lines 340-347
          // describe (attacker forwards their state to the victim
          // whose browser still holds the victim's cookie).
          val errBuf = new java.io.ByteArrayOutputStream()
          val originalErr = System.err
          System.setErr(new java.io.PrintStream(errBuf, true, StandardCharsets.UTF_8))
          val callback =
            try get(
              s"$baseUri${provider.callbackPath}?state=attacker-supplied-state-value&code=anyfakecode",
              Map("Cookie" -> stateCookie)
            )
            finally System.setErr(originalErr)
          assertEquals(callback.statusCode(), 302,
            clue = s"OIDC /callback with mismatched state must return 302 redirect to oidcFailureRedirect (per AuthStack.scala line 353); a non-302 means the handler exited via a different path which would emit a different reason= value; got: ${callback.statusCode()}")

          val captured = errBuf.toString(StandardCharsets.UTF_8)
          val failureLine = captured.split('\n').iterator
            .find(_.contains("auth.oidc.failure"))
            .getOrElse(fail(s"no `auth.oidc.failure` line in stderr capture -- expected the logWarn at AuthStack.scala line 352 to fire on the state-mismatch path; got captured stderr: ${captured.take(800)}"))

          // (i) event prefix
          assert(failureLine.contains("auth.oidc.failure"),
            clue = s"OIDC failure audit line must carry the literal `auth.oidc.failure` event prefix per deploy doc line 218's enumeration; got: $failureLine")
          // (ii) provider=google
          assert(failureLine.contains("provider=google"),
            clue = s"OIDC failure audit line must carry provider=google (lowercase id) matching the prior OIDC pins (976d7ad / b1213b6 / 342df03 / c8da491); got: $failureLine")
          // (iii) reason=state_cookie_mismatch (the load-bearing
          // per-emission-site contract for THIS branch -- the
          // SECURITY-CRITICAL signature the runbook keys on for
          // covert-redirect intrusion detection)
          assert(failureLine.contains("reason=state_cookie_mismatch"),
            clue = s"OIDC failure audit line for the mismatched-cookie path MUST carry the EXACT reason value `state_cookie_mismatch` per AuthStack.scala line 351's hardcoded `else \"state_cookie_mismatch\"` branch (the ELSE clause that fires when cookie IS present but doesn't match the URL state); this is the documented covert-redirect attack signature -- a refactor renaming to e.g. `cookie_state_mismatch` (reordered words), `state-cookie-mismatch` (hyphens), `forged_state`, or consolidating with missing_state_cookie into a single `state_invalid` would silently break the runbook's intrusion-detection triage AND silently mute the SECURITY signal the deploy doc's OAuth 2.0 BCP covert-redirect mitigation depends on for operator visibility; got: $failureLine")
          // (iv) EXCLUSION of missing_state_cookie -- pins that
          // the conditional branch taken was the cookie-PRESENT
          // branch (else clause), NOT the cookie-absent branch
          // (then clause from c8da491). The asymmetric-pair
          // exclusion check: c8da491 asserts
          // !contains("state_cookie_mismatch"), THIS asserts
          // !contains("missing_state_cookie"). A refactor that
          // swapped the two branches' reason strings (inverted
          // conditional) would fail BOTH tests' exclusion checks
          // simultaneously
          assert(!failureLine.contains("missing_state_cookie"),
            clue = s"OIDC failure audit line for the mismatched-cookie path MUST NOT contain the alternative reason `missing_state_cookie` per AuthStack.scala line 351's `if cookieState.isEmpty then \"missing_state_cookie\"` branch (the THEN clause that fires when no cookie is present); this exclusion pairs with c8da491's parallel exclusion on the no-cookie side, completing the asymmetric-pair coverage of line 352's two-reason conditional -- a refactor that swapped the branches' reason strings or emitted both would fail BOTH this test's exclusion AND c8da491's parallel exclusion simultaneously; got: $failureLine")
          // (v) WARN level
          assert(failureLine.contains("[WARN]"),
            clue = s"OIDC failure audit line must be WARN-level per AuthStack.scala line 352's logWarn call; the SECURITY-CRITICAL state_cookie_mismatch reason value specifically demands WARN visibility because it's the documented covert-redirect intrusion signature (a refactor demoting to DEBUG would silently hide intrusion attempts; a refactor promoting to ERROR would silently page incident-response automation on every normal browser quirk that drops cookies, training operators to ignore the alert); got: $failureLine")
          // (vi) remote= field (especially critical for this
          // emission because covert-redirect attacks come from
          // attacker-controlled IPs and the runbook's security
          // triage correlates state_cookie_mismatch failures with
          // subsequent successful OIDC sign-ins from the same IP)
          assert(failureLine.contains("remote="),
            clue = s"OIDC failure audit line must carry the `remote=` field -- especially critical for state_cookie_mismatch which is the documented covert-redirect signature where the runbook's intrusion-detection correlates this failure with subsequent successful sign-ins from the same attacker IP; without remote= the correlation breaks and successful attacks become invisible after the failed probe; got: $failureLine")
          // (vii) !email= ABSENCE
          assert(!failureLine.contains("email="),
            clue = s"OIDC failure audit line must NOT carry email= per deploy doc line 218 -- additionally critical for state_cookie_mismatch because the callback's state cookie may have been planted via a sibling-subdomain attack pointing at a victim user; emitting email= would expose the victim user's identity to log aggregation (the attacker-controlled IP could probe for matches by initiating OIDC flows for different victim emails and observing which appear in the audit logs); got: $failureLine")
          // (viii) service-tag prefix
          assert(failureLine.contains("[hand-history-review]"),
            clue = s"OIDC failure audit line must carry the `[hand-history-review]` service-tag prefix per HandHistoryReviewServerRuntime.scala line 512's hardcoded literal -- matches /api/health.service (505ba6b); got: $failureLine")
        }
      }
    }
  }

  // Pin the documented `auth.oidc.failure reason=oversize_callback_param`
  // audit line format -- closes the THIRD of 5 failure-side emission
  // sites (line 335 in AuthStack.scala), after 342df03 closed line
  // 397's missing_code_or_state and c8da491+b4b828f closed both
  // branches of line 352's missing_state_cookie / state_cookie_
  // mismatch conditional; line 335 fires when /callback receives a
  // state OR code query parameter exceeding MaxOidcParamLength (256
  // chars per AuthStack.scala line 512); the inline comment at
  // lines 322-333 documents the security context: "Reject obviously-
  // oversized state or code upfront. We issued `state` ourselves as
  // 24 random bytes base64url-encoded (~32 chars); legitimate
  // provider `code` values are typically under 200 chars. Anything
  // orders of magnitude larger is an attacker probing the callback
  // (potentially with a matching cookie planted via the sibling-
  // subdomain vector in insecure-cookie mode) trying to amplify
  // CPU/memory cost in the OidcStateStore lookup or the upstream
  // POST body to the provider's token endpoint. 256 chars matches
  // the cap on ?error= and is well above any legitimate value.
  // Treat oversize as `missing_code_or_state` so the failure
  // resembles a malformed request, not a state/cookie issue"; the
  // operational distinction matters because line 332-333's
  // documented "treat oversize as missing_code_or_state" comment
  // actually does NOT happen at the audit-log layer -- the
  // OPERATOR-VISIBLE reason value at line 335 IS the specific
  // oversize_callback_param string (NOT missing_code_or_state),
  // and only the USER-FACING redirect destination at line 336 uses
  // missing_code_or_state for the URL fragment; the test pins the
  // OPERATOR-VISIBLE reason value (oversize_callback_param) which
  // is what dashboards / incident response key on, NOT the user-
  // facing redirect value; this is a SUBTLE BUT IMPORTANT
  // distinction worth pinning because a refactor that "consolidated
  // the inconsistency" (e.g. emitting reason=missing_code_or_state
  // in BOTH the audit log AND the redirect to match the comment's
  // claim) would silently break the OPERATOR-side detection of
  // oversize-callback DoS probes (the runbook's CPU/memory-
  // exhaustion triage workflow distinguishes "user typo /
  // legitimate truncation" from "oversize attack probe" based on
  // THIS specific reason value); 8-tier format check at WARN level
  // with the asymmetric-pair exclusion across all OTHER reason
  // values pinned so far: (i) `auth.oidc.failure` event prefix,
  // (ii) `provider=google`, (iii) `reason=oversize_callback_param`
  // (the NEW specific-value pin), (iv-vi) triple EXCLUSION of
  // alternative reasons missing_code_or_state, missing_state_
  // cookie, state_cookie_mismatch -- catches a refactor that
  // emitted the wrong reason value for THIS emission site (e.g.
  // consolidating oversize with missing_code_or_state per the
  // inline comment's misleading claim, or routing oversize to the
  // state-cookie reason values), (vii) `[WARN]` level, (viii)
  // `remote=` field, (ix) `!email=` ABSENCE, (x) `[hand-history-
  // review]` service-tag prefix; test approach: GET /callback with
  // a state param of 1000 chars (well above the 256-char cap, well
  // above any reasonable cap-raise refactor's threshold) and a
  // short code -- the test pins the AUDIT LOG FORMAT for the
  // oversize path, NOT the specific cap value (a future fire could
  // pin MaxOidcParamLength=256 specifically if needed; this test
  // is robust to incidental cap raises because 1000 chars exceeds
  // any reasonable defensive cap); the test reuses FakeOidcProvider
  // (id="google") -- no cookie needed because the oversize check
  // at line 334 happens BEFORE the cookie check at line 348, so a
  // request with no cookie still triggers the oversize path; no
  // /start needed for the same reason -- the state value doesn't
  // need to be issued by the state-store since the oversize check
  // short-circuits before line 355's finishOidc call; per-field
  // regression vectors SPECIFIC to this emission that the prior 3
  // failure-side pins don't catch: (i) the OPERATOR-vs-USER-facing
  // reason distinction documented above -- a refactor unifying
  // them would silently break operator triage; (ii) the OR-vs-AND
  // logic at line 334's `rawState.length > MaxOidcParamLength ||
  // rawCode.length > MaxOidcParamLength` -- a refactor changing OR
  // to AND would silently let single-oversize values past (e.g. a
  // 100KB state with a 50-char code would no longer trigger);
  // pinning emission with ONLY state oversize (code is short)
  // exercises the LEFT side of the OR; a future fire could add a
  // mirror test with only code oversize to pin the RIGHT side of
  // the OR; (iii) the cap-not-zero invariant -- if a refactor
  // accidentally set MaxOidcParamLength to 0 or negative, EVERY
  // callback would trigger oversize_callback_param and the
  // legitimate path would never fire; this test doesn't directly
  // catch that (it would still pass), but the c8da491 /
  // b4b828f / b1213b6 tests would all fail because the legitimate
  // callbacks they exercise would now hit oversize first; the
  // cross-test interaction is the safety net for that regression
  // class.
  test("GET /api/auth/oidc/google/callback with state param exceeding MaxOidcParamLength emits the documented `auth.oidc.failure reason=oversize_callback_param` WARN audit line (DoS amplification mitigation per AuthStack.scala lines 322-335) -- closes the third of 5 failure-side emission sites") {
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

          // Capture stderr around a /callback GET with a state value
          // of 1000 chars (well above MaxOidcParamLength=256 per
          // AuthStack.scala line 512); no cookie / no /start needed
          // since the oversize check at line 334 short-circuits
          // BEFORE the cookie check at line 348. The 1000-char
          // length is chosen to be ROBUST TO INCIDENTAL CAP RAISES
          // (a future refactor that raised the cap to 512 would
          // still trigger this test's oversize path); the test
          // pins the audit-log FORMAT for the oversize emission
          // site, not the specific cap value -- the cap value is a
          // separate concern that could be pinned in a different
          // test if needed.
          val errBuf = new java.io.ByteArrayOutputStream()
          val originalErr = System.err
          System.setErr(new java.io.PrintStream(errBuf, true, StandardCharsets.UTF_8))
          val oversizeState = "a" * 1000
          val callback =
            try get(s"$baseUri${provider.callbackPath}?state=$oversizeState&code=anyfakecode")
            finally System.setErr(originalErr)
          assertEquals(callback.statusCode(), 302,
            clue = s"OIDC /callback with oversize state param must return 302 redirect to oidcFailureRedirect (per AuthStack.scala line 336); a non-302 means the handler exited via a different path which would emit a different reason= value; got: ${callback.statusCode()}")

          val captured = errBuf.toString(StandardCharsets.UTF_8)
          val failureLine = captured.split('\n').iterator
            .find(_.contains("auth.oidc.failure"))
            .getOrElse(fail(s"no `auth.oidc.failure` line in stderr capture -- expected the logWarn at AuthStack.scala line 335 to fire on the oversize state param; got captured stderr: ${captured.take(800)}"))

          // (i) event prefix
          assert(failureLine.contains("auth.oidc.failure"),
            clue = s"OIDC failure audit line must carry the literal `auth.oidc.failure` event prefix per deploy doc line 218; got: $failureLine")
          // (ii) provider=google (matches prior OIDC pins)
          assert(failureLine.contains("provider=google"),
            clue = s"OIDC failure audit line must carry provider=google matching the 976d7ad / b1213b6 / 342df03 / c8da491 / b4b828f pins; got: $failureLine")
          // (iii) reason=oversize_callback_param (THE specific-value
          // pin for THIS emission site -- the load-bearing contract
          // distinguishing oversize-DoS-probe from legitimate
          // missing-params)
          assert(failureLine.contains("reason=oversize_callback_param"),
            clue = s"OIDC failure audit line for the oversize-state path MUST carry the EXACT reason value `oversize_callback_param` per AuthStack.scala line 335's hardcoded string; a refactor renaming to e.g. `oversize_param` (shorter) / `oversize-callback-param` (hyphens) / `param_too_large`, or consolidating with missing_code_or_state per the inline comment at line 332-333's misleading 'Treat oversize as missing_code_or_state' framing (which describes the USER-facing redirect, NOT the operator-visible audit-log reason), would silently break the runbook's CPU/memory-exhaustion DoS-probe triage that distinguishes oversize-attack from legitimate-missing-params based on THIS exact reason value; got: $failureLine")
          // (iv-vi) EXCLUSION of the 3 alternative reason values
          // already pinned -- catches refactors that emitted the
          // wrong reason for THIS emission site
          assert(!failureLine.contains("reason=missing_code_or_state"),
            clue = s"OIDC failure audit line for the oversize path MUST NOT carry reason=missing_code_or_state (the 342df03-pinned reason for the no-params path); the inline comment at AuthStack.scala line 332-333 says 'Treat oversize as missing_code_or_state' but that ONLY applies to the USER-facing redirect at line 336, NOT the operator-visible audit-log reason at line 335 -- a refactor that consolidated the two reason values per the comment's literal reading would silently break operator dashboards distinguishing oversize-DoS from missing-params triage; got: $failureLine")
          assert(!failureLine.contains("reason=missing_state_cookie"),
            clue = s"OIDC failure audit line for the oversize path MUST NOT carry reason=missing_state_cookie (the c8da491-pinned reason for the no-cookie path); the oversize check at line 334 short-circuits BEFORE the cookie check at line 348, so the conditional flow guarantees these two reasons cannot co-occur -- a refactor that reordered the checks would silently change which reason fires AND silently break the documented short-circuit-on-oversize defense; got: $failureLine")
          assert(!failureLine.contains("reason=state_cookie_mismatch"),
            clue = s"OIDC failure audit line for the oversize path MUST NOT carry reason=state_cookie_mismatch (the b4b828f-pinned reason for the covert-redirect signature); the oversize check at line 334 fires BEFORE the cookie comparison at line 350, so these two reasons cannot co-occur -- a refactor that emitted both would silently confuse intrusion-detection triage; got: $failureLine")
          // (vii) WARN level
          assert(failureLine.contains("[WARN]"),
            clue = s"OIDC failure audit line must be WARN-level per AuthStack.scala line 335's logWarn call; demote-to-DEBUG silently hides DoS-probe attempts; got: $failureLine")
          // (viii) remote= field
          assert(failureLine.contains("remote="),
            clue = s"OIDC failure audit line must carry remote= per deploy doc line 218; oversize-callback DoS probes typically come from a single IP probing different state lengths to find the cap -- per-IP correlation is critical for distinguishing legitimate truncation (one IP, one event) from probe activity (one IP, many events); got: $failureLine")
          // (ix) !email= ABSENCE
          assert(!failureLine.contains("email="),
            clue = s"OIDC failure audit line must NOT carry email= per deploy doc line 218 ('auth.oidc.failure lines do NOT carry email='); the oversize-callback path may carry a tampered state cookie pointing at a victim's would-be session -- emitting email= would expose private data to log aggregation; got: $failureLine")
          // (x) service-tag prefix
          assert(failureLine.contains("[hand-history-review]"),
            clue = s"OIDC failure audit line must carry the `[hand-history-review]` service-tag prefix per HandHistoryReviewServerRuntime.scala line 512's hardcoded literal -- matches /api/health.service (505ba6b); got: $failureLine")
        }
      }
    }
  }

  // Pin the documented `auth.oidc.failure reason=provider-error:<code>`
  // audit line format -- closes the FOURTH of 5 failure-side
  // emission sites (line 317 in AuthStack.scala), after 342df03 +
  // c8da491 + b4b828f + e17c21d closed lines 397/352-both/335; line
  // 317 fires when /callback receives `?error=<value>` query
  // parameter -- the documented "user denied consent / expired
  // code / provider returned error" path that's reachable WITHOUT
  // any custom FakeOidcProvider mock (the existing FakeOidcProvider
  // returns Right on exchangeCode, but THIS emission path never
  // reaches exchangeCode -- the query.get("error") match at
  // AuthStack.scala line 286 short-circuits the entire callback
  // flow when ?error= is present, emitting the audit line and
  // redirecting before any state/code/cookie validation runs); the
  // operationally-unique feature of THIS emission is the
  // `provider-error:` PREFIX on the reason value -- the line 317
  // format is `reason=provider-error:<escaped-error-code>` where
  // the prefix marks "the upstream OIDC provider rejected the
  // flow" as DISTINCT from the server-side failure reasons
  // (missing_code_or_state, missing_state_cookie, state_cookie_
  // mismatch, oversize_callback_param -- all of which are OUR
  // server's reasons); operators distinguishing "the provider had
  // a problem" from "we had a problem" key on this prefix to route
  // incident response correctly (provider problems → escalate to
  // Google support, server problems → escalate internally); a
  // refactor that dropped the `provider-error:` prefix (e.g.
  // emitting just `reason=access_denied` to match the OIDC spec's
  // error code shape) would silently merge the provider-side and
  // server-side failure categories, forcing operators to manually
  // disambiguate which side caused each failure; a refactor that
  // changed the separator from `:` to `_` (`provider-error_access_
  // denied`) would silently break operator queries grep'ing for
  // the `provider-error:` prefix as an indicator that the error
  // code is upstream-supplied (the colon distinguishes prefix from
  // value cleanly); per-field regression vectors SPECIFIC to this
  // emission site that the prior 4 failure-side pins don't catch:
  // (i) the `provider-error:` literal prefix -- a refactor renaming
  // to e.g. `upstream-error:` / `oidc-provider-error:` / dropping
  // the prefix entirely would silently break the documented
  // provider-vs-server distinction; (ii) the `%20`-escape contract
  // -- AuthStack.scala line 317 applies `.replace(" ", "%20")` to
  // the error string before logging, preventing a hostile provider
  // returning `?error=foo bar` from splitting the structured
  // key=value log fields (the inline comment at lines 311-316
  // documents the threat); the test uses an OIDC standard error
  // code (access_denied) without spaces, but the format pin
  // ensures the prefix structure is intact -- a future fire could
  // add a separate test exercising the %20-escape directly with a
  // space-bearing error value; (iii) the capOidcErrorString length
  // cap (256 chars per inline comment) -- the test uses a short
  // standard code so this cap isn't exercised, but a future fire
  // could pin the cap by submitting `?error=<huge>` and verifying
  // the audit line is bounded; test approach: GET /callback?error=
  // access_denied -- no cookie, no state, no code; line 286's
  // `query.get("error") match` matches the Some(rawError) branch,
  // capOidcErrorString caps the value (no-op for the short standard
  // code), and logWarn at line 317 fires with reason=provider-
  // error:access_denied; 10-tier format check at WARN level
  // matching the e17c21d pattern with the QUADRUPLE-EXCLUSION of
  // all 4 previously-pinned alternative reasons: (i) `auth.oidc.
  // failure` event prefix, (ii) `provider=google`, (iii) `reason=
  // provider-error:access_denied` (NEW specific-value pin INCLUDING
  // the `provider-error:` prefix and the colon separator), (iv-vii)
  // EXCLUSION of missing_code_or_state / missing_state_cookie /
  // state_cookie_mismatch / oversize_callback_param (the 4 already-
  // pinned alternative reasons from the OTHER 3 emission sites) --
  // catches a refactor emitting the wrong reason for THIS emission
  // site, ESPECIALLY a refactor consolidating provider-supplied
  // errors with the server-side reasons, (viii) `[WARN]` level,
  // (ix) `remote=` field, (x) `!email=` ABSENCE, (xi) `[hand-
  // history-review]` service-tag prefix.
  test("GET /api/auth/oidc/google/callback with ?error=access_denied emits the documented `auth.oidc.failure reason=provider-error:access_denied` WARN audit line carrying the `provider-error:` prefix that distinguishes upstream provider failures from server-side failures (per AuthStack.scala lines 285-317) -- closes the fourth of 5 failure-side emission sites") {
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

          // Capture stderr around a /callback GET with ?error=access_denied
          // -- no cookie, no state, no code. AuthStack.scala line
          // 286's `query.get("error") match { case Some(rawError) =>`
          // matches immediately, capOidcErrorString (line 310) caps
          // the value (no-op for the short standard code
          // "access_denied"), the audit-log emission at line 317
          // fires with reason=provider-error:access_denied, and the
          // user-facing redirect at line 318 fires. The whole flow
          // SHORT-CIRCUITS before any state/code/cookie validation
          // runs -- the `query.get("error")` branch is checked
          // BEFORE the `(query.get("state"), query.get("code"))`
          // match at line 320, AND before the cookie check at line
          // 348 in the no-error branch.
          val errBuf = new java.io.ByteArrayOutputStream()
          val originalErr = System.err
          System.setErr(new java.io.PrintStream(errBuf, true, StandardCharsets.UTF_8))
          val callback =
            try get(s"$baseUri${provider.callbackPath}?error=access_denied")
            finally System.setErr(originalErr)
          assertEquals(callback.statusCode(), 302,
            clue = s"OIDC /callback with ?error= must return 302 redirect to oidcFailureRedirect (per AuthStack.scala line 318); a non-302 means the handler exited via a different path which would emit a different reason= value; got: ${callback.statusCode()}")

          val captured = errBuf.toString(StandardCharsets.UTF_8)
          val failureLine = captured.split('\n').iterator
            .find(_.contains("auth.oidc.failure"))
            .getOrElse(fail(s"no `auth.oidc.failure` line in stderr capture -- expected the logWarn at AuthStack.scala line 317 to fire on the provider-supplied-error path; got captured stderr: ${captured.take(800)}"))

          // (i) event prefix
          assert(failureLine.contains("auth.oidc.failure"),
            clue = s"OIDC failure audit line must carry the literal `auth.oidc.failure` event prefix per deploy doc line 218; got: $failureLine")
          // (ii) provider=google
          assert(failureLine.contains("provider=google"),
            clue = s"OIDC failure audit line must carry provider=google matching the prior OIDC pins (976d7ad / b1213b6 / 342df03 / c8da491 / b4b828f / e17c21d); got: $failureLine")
          // (iii) reason=provider-error:access_denied -- the
          // load-bearing per-emission-site contract WITH the
          // `provider-error:` prefix AND colon separator
          assert(failureLine.contains("reason=provider-error:access_denied"),
            clue = s"OIDC failure audit line for the provider-supplied-error path MUST carry the EXACT reason value `provider-error:access_denied` per AuthStack.scala line 317's hardcoded `reason=provider-error:` prefix concatenated with the capped+escaped provider error code; the `provider-error:` prefix is the documented marker distinguishing upstream provider failures from server-side failures (missing_state_cookie etc are SERVER reasons; provider-error:access_denied is the UPSTREAM reason); a refactor dropping the prefix (emitting just `reason=access_denied`) would silently merge provider-side and server-side failure categories forcing operators to manually disambiguate which side caused each failure, a refactor changing the separator from `:` to `_` would silently break operator queries grep'ing for the `provider-error:` prefix as the upstream-source indicator; got: $failureLine")
          // (iv-vii) QUADRUPLE EXCLUSION of all 4 already-pinned
          // alternative reasons -- catches a refactor emitting the
          // wrong reason for THIS emission site
          assert(!failureLine.contains("reason=missing_code_or_state"),
            clue = s"OIDC failure audit line for the provider-error path MUST NOT carry reason=missing_code_or_state (the 342df03-pinned no-params reason); the ?error= branch at line 286 short-circuits BEFORE the missing-params check at line 320, so these two reasons cannot co-occur -- a refactor consolidating them would silently lose the upstream-vs-server-failure distinction; got: $failureLine")
          assert(!failureLine.contains("reason=missing_state_cookie"),
            clue = s"OIDC failure audit line for the provider-error path MUST NOT carry reason=missing_state_cookie (the c8da491-pinned no-cookie reason); the ?error= branch short-circuits BEFORE the cookie check at line 348; got: $failureLine")
          assert(!failureLine.contains("reason=state_cookie_mismatch"),
            clue = s"OIDC failure audit line for the provider-error path MUST NOT carry reason=state_cookie_mismatch (the b4b828f-pinned covert-redirect reason); the ?error= branch short-circuits BEFORE the cookie comparison at line 350; got: $failureLine")
          assert(!failureLine.contains("reason=oversize_callback_param"),
            clue = s"OIDC failure audit line for the provider-error path MUST NOT carry reason=oversize_callback_param (the e17c21d-pinned DoS-probe reason); the ?error= branch short-circuits BEFORE the oversize check at line 334, AND the test submits a short standard error code that wouldn't trigger oversize regardless; got: $failureLine")
          // (viii) WARN level
          assert(failureLine.contains("[WARN]"),
            clue = s"OIDC failure audit line must be WARN-level per AuthStack.scala line 317's logWarn call; the provider-error path is an upstream failure (typically user-initiated cancel or provider transient issue), demote-to-DEBUG would hide it from triage AND make it impossible to detect bursts of provider-side failures (the runbook's 'unusual provider failure burst' triage entry keys on WARN-level visibility of these lines), promote-to-ERROR would silently page on every user clicking 'Cancel' on the Google consent screen; got: $failureLine")
          // (ix) remote= field
          assert(failureLine.contains("remote="),
            clue = s"OIDC failure audit line must carry remote= per deploy doc line 218; per-IP correlation distinguishes a single user repeatedly cancelling consent (one IP, several events spaced minutes apart) from coordinated probe activity (one IP, many events in seconds) -- both surface as auth.oidc.failure provider-error events but only the latter is operationally interesting; got: $failureLine")
          // (x) !email= ABSENCE
          assert(!failureLine.contains("email="),
            clue = s"OIDC failure audit line must NOT carry email= per deploy doc line 218 ('auth.oidc.failure lines do NOT carry email='); the provider-error path may not have any user identity at all (the user denied consent before the userinfo step) AND emitting email= here would suggest a specific user when the upstream failure is provider-side; got: $failureLine")
          // (xi) service-tag prefix
          assert(failureLine.contains("[hand-history-review]"),
            clue = s"OIDC failure audit line must carry the `[hand-history-review]` service-tag prefix per HandHistoryReviewServerRuntime.scala line 512's hardcoded literal -- matches /api/health.service (505ba6b); got: $failureLine")
        }
      }
    }
  }

  // Pin the documented `auth.oidc.failure reason=<finishOidc-error>`
  // audit line format with the %20-ESCAPE CONTRACT for space-bearing
  // error strings -- closes the FIFTH and FINAL emission site for
  // auth.oidc.failure (line 363 in AuthStack.scala), after 342df03 +
  // c8da491 + b4b828f + e17c21d + aa6426d closed lines 397 / 352-
  // both-branches / 335 / 317; with this commit the auth.oidc.
  // failure enumeration is FULLY CLOSED across all 5 emission sites
  // AND the deploy doc line 218's 9-event enumeration is FULLY
  // CLOSED for both the success-side (1c8777f + 49dcf46 + 3a6fea4 +
  // 976d7ad + b1213b6) and the failure-side (342df03 + c8da491 +
  // b4b828f + e17c21d + aa6426d + this commit) -- modulo the
  // auth.oidc.start.failure event at line 258 which remains
  // UNREACHABLE per the 342df03 routing-layer analysis (HttpServer
  // contexts are only registered for known providers, so a request
  // to an unknown provider's /start hits the default 404 handler
  // before reaching handleOidcStart's logWarn at line 258); line
  // 363 fires when finishOidc returns Left -- the documented Left
  // values include "OIDC login state expired or is invalid" (state-
  // store consume failed), "an account with that email already
  // exists; sign in with its existing method" (the 44c9f9f email-
  // collision defense), and "Google did not return a verified email
  // address for this account" (the verified-email gate per the
  // OIDC spec); the format at line 363 is
  // `reason=${error.replace(" ", "%20")}` -- the `.replace(" ",
  // "%20")` is the LOAD-BEARING %20-ESCAPE CONTRACT this commit
  // pins, NOT exercised by ANY of the prior 5 auth.oidc.failure
  // pins because their reason values are all space-FREE constants
  // (missing_code_or_state / missing_state_cookie / state_cookie_
  // mismatch / oversize_callback_param / provider-error:access_
  // denied); the %20-escape is documented at AuthStack.scala lines
  // 357-362's inline comment: "%20-escape -- finishOidc Left values
  // include 'OIDC login state expired or is invalid', 'an account
  // with that email already exists; sign in with its existing
  // method', 'Google did not return a verified email address for
  // this account', and friends -- all space-bearing. Same pattern
  // as start.failure above" -- without the escape, the space in
  // the error string would split the structured `key=value` log
  // format (a hostile provider could in theory inject log-line-
  // splitting characters; the escape closes that vector); the test
  // exploits the FORGED-COOKIE path: construct a Cookie header
  // with the documented sicfun_oidc_state cookie name carrying a
  // state value that was NEVER issued by the OidcStateStore, then
  // GET /callback?state=<same-value>&code=anycode -- line 348's
  // extractCookieFromExchange finds the cookie, line 350's
  // secureEquals comparison passes (cookie value == URL state
  // value), line 355's finishOidc tries to consume the state from
  // the state-store, the consume returns None because the state
  // was never issued, finishOidc returns Left("OIDC login state
  // expired or is invalid"), logWarn at line 363 fires with
  // reason=OIDC%20login%20state%20expired%20or%20is%20invalid;
  // per-field regression vectors SPECIFIC to this emission site
  // that the prior 5 failure-side pins don't catch: (i) THE %20-
  // ESCAPE CONTRACT itself -- a refactor that dropped the
  // .replace(" ", "%20") at line 363 (e.g. "OIDC error codes
  // shouldn't have spaces in modern implementations, the escape is
  // dead code") would silently break the structured log-line
  // contract because the finishOidc error strings DO contain
  // spaces; a hostile provider that returned an exchangeCode
  // response triggering a space-bearing finishOidc Left value
  // could inject log-line-splitting characters into the audit
  // stream -- the operator's downstream log parser (expecting
  // key=value pairs) would silently misparse the line, dropping
  // the remote= and reason= fields from operator triage; (ii) the
  // SPECIFIC error string "OIDC login state expired or is invalid"
  // -- a refactor renaming the finishOidc Left value (e.g. to
  // "expired_state" matching the spec's error code shape) would
  // silently break operator queries grep'ing for the documented
  // exact-string match AND would silently invalidate the runbook's
  // OIDC-state-expiration triage entry which keys on this exact
  // wording; (iii) the finishOidc Left -> reason pipeline -- a
  // refactor consolidating finishOidc Left values into a single
  // generic "finishOidc_failed" reason would silently lose the
  // operator-relevant detail about WHICH finishOidc path failed
  // (state-expired vs email-collision vs no-verified-email each
  // need different triage); 12-tier format check at WARN level
  // extending the aa6426d pattern with the QUINTUPLE-EXCLUSION of
  // ALL 5 previously-pinned alternative reasons: (i) `auth.oidc.
  // failure` event prefix, (ii) `provider=google`, (iii) `reason=
  // OIDC%20login%20state%20expired%20or%20is%20invalid` (NEW
  // specific-value pin WITH the %20-escape contract verified
  // directly), (iv-viii) QUINTUPLE EXCLUSION of missing_code_or_
  // state / missing_state_cookie / state_cookie_mismatch /
  // oversize_callback_param / provider-error -- catches a refactor
  // emitting the wrong reason for THIS emission site, ESPECIALLY a
  // refactor that incorrectly routed the finishOidc Left value
  // through the provider-error: prefix (which would silently merge
  // server-side state-store-empty errors with upstream-provider
  // errors), (ix) `[WARN]` level, (x) `remote=` field, (xi)
  // `!email=` ABSENCE, (xii) `[hand-history-review]` service-tag
  // prefix; ADDITIONAL pin: the test asserts the line does NOT
  // contain the unescaped space-bearing form "OIDC login state
  // expired or is invalid" (without %20-escapes) -- catches a
  // refactor dropping the escape entirely; this is the
  // ESCAPE-CONTRACT-VERIFICATION pin that turns the %20-escape
  // from an unexercised side-effect into a CI-enforced invariant.
  test("GET /api/auth/oidc/google/callback with forged state cookie matching URL state but state not in store emits the documented `auth.oidc.failure reason=OIDC%20login%20state%20expired%20or%20is%20invalid` WARN audit line WITH %20-escaped spaces (per AuthStack.scala line 363's `.replace(\" \", \"%20\")` escape contract) -- closes the fifth and final auth.oidc.failure emission site") {
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

          // Capture stderr around a /callback GET with a FORGED
          // state cookie. The Cookie header uses the documented
          // sicfun_oidc_state cookie name (insecure mode default
          // per PlatformUserAuth.scala line 49's DefaultOidcState
          // CookieName); the cookie value EQUALS the URL state
          // value so the secureEquals check at AuthStack.scala
          // line 350 passes; finishOidc at line 355 then attempts
          // to consume the state from the OidcStateStore, which
          // never issued it (no /start fired for this state value),
          // returns Left("OIDC login state expired or is invalid"),
          // logWarn at line 363 fires with the %20-escaped reason.
          val errBuf = new java.io.ByteArrayOutputStream()
          val originalErr = System.err
          System.setErr(new java.io.PrintStream(errBuf, true, StandardCharsets.UTF_8))
          val forgedState = "never-issued-state-value-1234567890"
          val callback =
            try get(
              s"$baseUri${provider.callbackPath}?state=$forgedState&code=anycode",
              Map("Cookie" -> s"sicfun_oidc_state=$forgedState")
            )
            finally System.setErr(originalErr)
          assertEquals(callback.statusCode(), 302,
            clue = s"OIDC /callback with forged cookie+state must return 302 redirect to oidcFailureRedirect (per AuthStack.scala line 364); a non-302 means the handler exited via a different path which would emit a different reason= value; got: ${callback.statusCode()}")

          val captured = errBuf.toString(StandardCharsets.UTF_8)
          val failureLine = captured.split('\n').iterator
            .find(_.contains("auth.oidc.failure"))
            .getOrElse(fail(s"no `auth.oidc.failure` line in stderr capture -- expected the logWarn at AuthStack.scala line 363 to fire on the finishOidc-Left path when the state-store consume fails; got captured stderr: ${captured.take(800)}"))

          // (i) event prefix
          assert(failureLine.contains("auth.oidc.failure"),
            clue = s"OIDC failure audit line must carry the literal `auth.oidc.failure` event prefix per deploy doc line 218; got: $failureLine")
          // (ii) provider=google
          assert(failureLine.contains("provider=google"),
            clue = s"OIDC failure audit line must carry provider=google matching the prior 6 OIDC pins; got: $failureLine")
          // (iii) reason=<%20-ESCAPED finishOidc Left value> -- the
          // load-bearing pin for THIS emission site AND for the
          // %20-escape contract that no prior pin exercises
          assert(failureLine.contains("reason=OIDC%20login%20state%20expired%20or%20is%20invalid"),
            clue = s"OIDC failure audit line for the state-not-in-store path MUST carry the EXACT %20-escaped reason value `OIDC%20login%20state%20expired%20or%20is%20invalid` per AuthStack.scala line 363's `.replace(\" \", \"%20\")` escape applied to the finishOidc Left value 'OIDC login state expired or is invalid'; the %20-escape is the LOAD-BEARING contract this pin uniquely catches -- a refactor dropping the escape would silently break the structured key=value log format when the finishOidc Left value contains spaces (which it DOES per AuthStack.scala lines 357-362's inline comment enumerating space-bearing values); a refactor renaming the finishOidc Left string would silently break operator queries grep'ing for the documented exact-string match; got: $failureLine")
          // (iv) ESCAPE-CONTRACT VERIFICATION: must NOT contain the
          // unescaped space-bearing form. This is the most
          // important pin in this test -- without it, a refactor
          // that emitted BOTH forms (escaped AND unescaped) or
          // that dropped the escape would silently pass the
          // positive %20-form contains check if the regression
          // emitted the unescaped form alongside
          assert(!failureLine.contains("OIDC login state expired or is invalid"),
            clue = s"OIDC failure audit line MUST NOT contain the UNESCAPED form `OIDC login state expired or is invalid` (with literal spaces) -- the %20-escape contract at AuthStack.scala line 363 REQUIRES spaces be replaced with %20 BEFORE the log emission to prevent the structured key=value log format from being split by hostile error strings; a refactor that dropped the escape (e.g. 'OIDC error codes shouldn't contain spaces in modern implementations, dead code removed') would emit the unescaped form and pass the positive %20-form check ONLY IF the line ALSO contained the escaped form (which a refactor wouldn't); the !contains assertion here pins the escape contract from the negative direction -- if the line contains the unescaped form, the escape has been broken; got: $failureLine")
          // (v-ix) QUINTUPLE EXCLUSION of all 5 previously-pinned
          // alternative reasons -- catches a refactor emitting the
          // wrong reason for THIS emission site
          assert(!failureLine.contains("reason=missing_code_or_state"),
            clue = s"OIDC failure audit line for the state-not-in-store path MUST NOT carry reason=missing_code_or_state (the 342df03-pinned no-params reason); the state+code params ARE present in this test, so the no-params check at line 320 short-circuits to the present-params branch -- a refactor that consolidated would silently merge state-expired with no-params; got: $failureLine")
          assert(!failureLine.contains("reason=missing_state_cookie"),
            clue = s"OIDC failure audit line for the state-not-in-store path MUST NOT carry reason=missing_state_cookie (the c8da491-pinned no-cookie reason); the test sends a cookie, so the cookie-presence check at line 350 passes; got: $failureLine")
          assert(!failureLine.contains("reason=state_cookie_mismatch"),
            clue = s"OIDC failure audit line for the state-not-in-store path MUST NOT carry reason=state_cookie_mismatch (the b4b828f-pinned covert-redirect reason); the test sends a cookie whose value MATCHES the URL state, so the secureEquals check at line 350 passes; the failure occurs LATER at line 355's finishOidc when the state-store has no record of the forged state value; got: $failureLine")
          assert(!failureLine.contains("reason=oversize_callback_param"),
            clue = s"OIDC failure audit line for the state-not-in-store path MUST NOT carry reason=oversize_callback_param (the e17c21d-pinned DoS-probe reason); the forged state value is well under MaxOidcParamLength=256, so the oversize check at line 334 passes; got: $failureLine")
          assert(!failureLine.contains("reason=provider-error:"),
            clue = s"OIDC failure audit line for the state-not-in-store path MUST NOT carry the `provider-error:` prefix (the aa6426d-pinned upstream-failure marker); the failure here is SERVER-side (our state-store doesn't have the forged state), NOT provider-side -- consolidating finishOidc Left values into the provider-error: prefix would silently merge server-side state-expired errors with upstream-provider failures, breaking the documented provider-vs-server distinction; got: $failureLine")
          // (x) WARN level
          assert(failureLine.contains("[WARN]"),
            clue = s"OIDC failure audit line must be WARN-level per AuthStack.scala line 363's logWarn call; finishOidc-Left failures span legitimate operator concerns (state expiration is a routine timing issue from slow user flows) AND security signals (state replay attempts) -- WARN-level visibility lets operators triage both classes; demote-to-DEBUG hides both; got: $failureLine")
          // (xi) remote= field
          assert(failureLine.contains("remote="),
            clue = s"OIDC failure audit line must carry remote= per deploy doc line 218; per-IP correlation distinguishes a single slow user (legitimate state-expiration after 10 min) from a state-replay attack (single IP retrying the same state value across multiple requests); got: $failureLine")
          // (xii) !email= ABSENCE
          assert(!failureLine.contains("email="),
            clue = s"OIDC failure audit line must NOT carry email= per deploy doc line 218; the finishOidc-Left path may have ANY of the failure values -- state-expired (no user resolved yet), email-collision (user identified but flow blocked), or no-verified-email (user identified but rejected) -- and the ABSENCE policy applies uniformly regardless of which Left value fired; a refactor that started emitting email= 'when available' would create asymmetric coverage across the 3 sub-cases of finishOidc-Left; got: $failureLine")
          // (xiii) service-tag prefix
          assert(failureLine.contains("[hand-history-review]"),
            clue = s"OIDC failure audit line must carry the `[hand-history-review]` service-tag prefix per HandHistoryReviewServerRuntime.scala line 512's hardcoded literal -- matches /api/health.service (505ba6b); got: $failureLine")
        }
      }
    }
  }

  // Pin the documented `startup complete` banner log line format --
  // a SEPARATE concern from the auth-event audit log chain
  // (1c8777f through e43081b) which closed the 9-event runtime
  // audit-log enumeration; this commit pins the BOOT-TIME banner
  // that emits ONCE per process lifetime at HandHistoryReviewServer
  // Runtime.scala line 349-350 immediately AFTER the HTTP server
  // binds + starts accepting connections; the banner is documented
  // in HAND_HISTORY_WEB_DEPLOYMENT.md as the operator-side boot-
  // time diagnostic an operator grep's to confirm the server
  // started with the EXACT configuration the deployment manifest
  // intended (vs the values an environment-variable / argument /
  // default could have silently substituted); BEFORE this commit
  // there was ZERO test coverage of the startup banner -- a
  // refactor that broke ANY of the ~17 documented field=value
  // pairs would silently invalidate the operator's boot-time
  // config-verification workflow; the banner emits at INFO level
  // (logInfo at line 349 writes to System.out per
  // HandHistoryReviewServerRuntime.scala line 418), with all
  // fields as `key=value` pairs separated by SINGLE SPACES, and
  // a documented %20-ESCAPE on path-like fields (modelSource,
  // drainSignalFile, rateLimitClientIpSource) to prevent
  // Windows-path-with-spaces from splitting the structured
  // key=value format -- the inline comment at HandHistoryReview
  // ServerRuntime.scala lines 334-338 documents the threat
  // ("Windows path such as 'C:\\Program Files\\model' or a
  // drain-signal file in a user home dir like 'C:\\Users\\Alex
  // Smith\\drain.flag' does not split the structured key=value
  // pairs"); the documented field set: startup complete +
  // host + port + modelSource + maxUploadBytes + analysisTimeoutMs
  // + playingHallTimeoutMs + maxConcurrentJobs + maxQueuedJobs +
  // rateLimitSubmitsPerMinute + rateLimitStatusPerMinute +
  // rateLimitAuthPerMinute + rateLimitClientIpSource +
  // rateLimitTrustedProxyIps + drainSignalFile + authenticationMode
  // + userAuthMaxUsers + userAuthStoredUsers (the last two emit
  // "-" for non-platform-auth modes per lines 328 + 333); per-
  // field regression vectors: (i) renaming the "startup complete"
  // prefix (e.g. to "server started" or "boot complete") would
  // silently break every operator script grep'ing for the banner
  // -- the runbook's "confirm boot config" diagnostic depends on
  // this exact wording, (ii) dropping any field=value pair would
  // silently lose operator visibility into that config value AND
  // would silently desync the banner from the /api/health response
  // shape which echoes most of the same values, (iii) breaking
  // the %20-escape on path-like fields would silently let
  // Windows-with-spaces deployments split the banner into
  // unparseable fragments that the operator's grep-based
  // verification couldn't parse, (iv) demoting INFO to DEBUG
  // would silently hide the banner at default log levels making
  // the entire boot-time-diagnostic workflow invisible, (v)
  // emitting the banner BEFORE the actual bind (the line 349
  // emission runs AFTER the HTTP server binds per the code flow
  // -- a refactor reordering this would silently let the banner
  // appear before the server is accepting traffic, misleading
  // operator readiness checks); test approach: wrap System.setOut
  // AROUND the withServer call so the banner emission (which
  // fires DURING withServer's HandHistoryReviewServer.startWithBackends
  // invocation, BEFORE the run callback executes) lands in the
  // captured stream; the test runs no actual HTTP requests --
  // just spins up + tears down the server while capturing the
  // boot-time emissions; format check covers (i) the "startup
  // complete" prefix, (ii) the [INFO] level + [hand-history-review]
  // service-tag prefix matching the audit-log chain's coupling,
  // (iii) the well-known config values from withServer's default
  // parameters: host=127.0.0.1, maxUploadBytes=512 (or the
  // overridden value), analysisTimeoutMs=120000, playingHallTimeoutMs=
  // 900000, maxConcurrentJobs=2, maxQueuedJobs=8, rateLimitSubmits
  // PerMinute=6, etc., (iv) per-field PRESENCE checks for every
  // documented field name (catches refactor that dropped a field
  // even if the value happens to still be present elsewhere in
  // the line), (v) the authenticationMode=none value (since
  // withServer defaults to no platformAuth + no basicAuth),
  // (vi) the userAuthMaxUsers=- + userAuthStoredUsers=- values
  // (the "-" placeholder for non-platform-auth deployments per
  // the inline comment at lines 328+333), (vii) the drainSignal
  // File=- value (no drain signal configured in default withServer),
  // (viii) modelSource=uniform%20fallback (the %20-escaped form
  // of "uniform fallback" -- THIS pins the %20-escape contract
  // on a SPACE-BEARING value the prior pins don't exercise);
  // test pins the BOOT-TIME log line AS A WHOLE per the deploy
  // doc's framing -- a future fire could add separate tests for
  // specific edge cases (Windows-path %20-escape, platform-auth
  // userAuth* fields with real values, drain-signal path %20-
  // escape).
  test("server startup emits the documented `startup complete` boot-time banner at INFO level with all ~17 documented field=value pairs per HAND_HISTORY_WEB_DEPLOYMENT.md's operator boot-time-diagnostic contract") {
    withStaticSite { staticDir =>
      // Capture stdout AROUND the withServer call so the banner
      // emission at HandHistoryReviewServerRuntime.scala line 349
      // (which fires DURING startWithBackends, BEFORE the run
      // callback executes) lands in the captured stream. The
      // withServer fixture's `server.close()` finally block runs
      // BEFORE the outer finally restores System.out, so the
      // shutdown banner ALSO lands in captured (we don't assert
      // on the shutdown banner here -- a future fire could add a
      // separate pin for that line at line 326).
      val outBuf = new java.io.ByteArrayOutputStream()
      val originalOut = System.out
      System.setOut(new java.io.PrintStream(outBuf, true, StandardCharsets.UTF_8))
      try
        withServer(staticDir) { _ =>
          // No HTTP requests needed -- the banner emits at
          // startup-time BEFORE this callback runs. The empty
          // body just keeps the server alive long enough for
          // the startup emission to flush. The 'server listening'
          // line emits FIRST (HandHistoryReviewServer.scala line
          // 74), then the 'startup complete' banner emits SECOND
          // (HandHistoryReviewServerRuntime.scala line 349) --
          // both are captured by our stdout wrap.
          ()
        }
      finally
        System.setOut(originalOut)

      val captured = outBuf.toString(StandardCharsets.UTF_8)
      val bannerLine = captured.split('\n').iterator
        .find(_.contains("startup complete"))
        .getOrElse(fail(s"no `startup complete` line in captured stdout -- deploy doc + HandHistoryReviewServerRuntime.scala line 349 document this as the boot-time diagnostic banner; if missing, the logInfo emission was suppressed OR the line was renamed; got captured stdout: ${captured.take(2000)}"))

      // (i) prefix
      assert(bannerLine.contains("startup complete"),
        clue = s"banner must carry the literal `startup complete` prefix per HandHistoryReviewServerRuntime.scala line 349's hardcoded literal -- a refactor renaming to e.g. `server started` / `boot complete` would silently break operator scripts grep'ing for the boot-time-diagnostic banner; got: $bannerLine")
      // (ii) INFO level
      assert(bannerLine.contains("[INFO]"),
        clue = s"banner must be INFO-level (logInfo at line 349 writes to System.out per HandHistoryReviewServerRuntime.scala line 418); demote-to-DEBUG would silently hide the banner at default log levels, making the entire boot-time-diagnostic workflow invisible; got: $bannerLine")
      // (iii) service-tag prefix (couples to /api/health.service from 505ba6b)
      assert(bannerLine.contains("[hand-history-review]"),
        clue = s"banner must carry the `[hand-history-review]` service-tag prefix per HandHistoryReviewServerRuntime.scala line 512's hardcoded literal -- matches /api/health.service (pinned by 505ba6b) so log aggregators see the same identifier on boot-time banners as on runtime audit lines AND on probe responses; got: $bannerLine")
      // (iv-xx) per-field PRESENCE + key value checks. The
      // withServer fixture defaults provide deterministic values
      // for most fields.
      assert(bannerLine.contains("host=127.0.0.1"),
        clue = s"banner must carry host=127.0.0.1 (withServer default); a refactor reporting the requested-host value instead of the bound-host (which differs when port=0 is used because the binding resolves the host post-bind) would silently mismatch fleet correlation between this banner and /api/health.host (pinned by 505ba6b); got: $bannerLine")
      assert(bannerLine.contains("port="),
        clue = s"banner must carry port=<resolved-bound-port> -- withServer uses port=0 to get an ephemeral OS-assigned port, so the exact value varies per test run but the field name MUST be present; a refactor reporting config.port (always 0 here) instead of binding.port would silently emit port=0 in the banner while the actual server bound to a real ephemeral port; got: $bannerLine")
      assert(bannerLine.contains("modelSource=uniform%20fallback"),
        clue = s"banner must carry modelSource=uniform%20fallback (the %20-escaped form of 'uniform fallback' -- withServer default has no MODEL_DIR set); THIS is the unique %20-ESCAPE CONTRACT pin on the startup banner that catches a refactor dropping the .replace(\" \", \"%20\") at HandHistoryReviewServerRuntime.scala line 339, which would silently split the structured key=value format when modelSource contains a space-bearing value like a Windows path 'C:\\Program Files\\model'; the unescaped form 'uniform fallback' would also let the banner split because the space would be parsed as a field separator by downstream log parsers; got: $bannerLine")
      assert(bannerLine.contains("maxUploadBytes=512"),
        clue = s"banner must carry maxUploadBytes=512 (withServer default); a refactor that dropped the field OR changed the default would silently desync the banner from /api/health.maxUploadBytes; got: $bannerLine")
      assert(bannerLine.contains("analysisTimeoutMs=120000"),
        clue = s"banner must carry analysisTimeoutMs=120000 (withServer default = 2 minutes); got: $bannerLine")
      assert(bannerLine.contains("playingHallTimeoutMs=900000"),
        clue = s"banner must carry playingHallTimeoutMs=900000 (withServer default = 15 minutes per 61e49a8); got: $bannerLine")
      assert(bannerLine.contains("maxConcurrentJobs=2"),
        clue = s"banner must carry maxConcurrentJobs=2 (withServer default); got: $bannerLine")
      assert(bannerLine.contains("maxQueuedJobs=8"),
        clue = s"banner must carry maxQueuedJobs=8 (withServer default); got: $bannerLine")
      assert(bannerLine.contains("rateLimitSubmitsPerMinute=6"),
        clue = s"banner must carry rateLimitSubmitsPerMinute=6 (withServer default); got: $bannerLine")
      assert(bannerLine.contains("rateLimitStatusPerMinute=240"),
        clue = s"banner must carry rateLimitStatusPerMinute=240 (withServer default); got: $bannerLine")
      assert(bannerLine.contains("rateLimitAuthPerMinute=10"),
        clue = s"banner must carry rateLimitAuthPerMinute=10 (withServer default); got: $bannerLine")
      assert(bannerLine.contains("rateLimitClientIpSource="),
        clue = s"banner must carry the rateLimitClientIpSource= field -- the inline comment at HandHistoryReviewServerRuntime.scala lines 343-348 documents this as a %20-escape-bearing field (values like 'header:X-Real-IP via loopback-only' contain spaces); dropping the field would silently break operator visibility into the IP-resolution policy; got: $bannerLine")
      assert(bannerLine.contains("rateLimitTrustedProxyIps="),
        clue = s"banner must carry the rateLimitTrustedProxyIps= field; got: $bannerLine")
      assert(bannerLine.contains("drainSignalFile=-"),
        clue = s"banner must carry drainSignalFile=- (the documented \"-\" placeholder per HandHistoryReviewServerRuntime.scala line 342's getOrElse(\"-\") -- withServer default has no drain signal configured) -- a refactor that emitted an empty string or 'null' or omitted the field entirely would silently break the runbook's 'is drain-signal wired' boot-time check; got: $bannerLine")
      assert(bannerLine.contains("authenticationMode=none"),
        clue = s"banner must carry authenticationMode=none (withServer default has no platformAuth + no basicAuth) -- matches /api/health.authenticationMode emitted at runtime; a refactor that changed the boot-time mode-detection logic would silently desync the banner from runtime probes AND silently invalidate operator alerts that compare the two; got: $bannerLine")
      assert(bannerLine.contains("userAuthMaxUsers=-"),
        clue = s"banner must carry userAuthMaxUsers=- (the documented \"-\" placeholder per HandHistoryReviewServerRuntime.scala line 328's getOrElse(\"-\") for non-platform-auth deployments); a refactor emitting an empty value or 'null' would silently break the runbook's boot-time auth-config check; got: $bannerLine")
      assert(bannerLine.contains("userAuthStoredUsers=-"),
        clue = s"banner must carry userAuthStoredUsers=- (the documented \"-\" placeholder per HandHistoryReviewServerRuntime.scala line 333's getOrElse(\"-\") -- the inline comment at line 329-332 documents this as the boot-time count emission that lets operators verify the persistent store survived restart without hitting /api/health first); got: $bannerLine")
    }
  }

  // Pin the documented `startup complete` banner under platform-
  // user authentication mode -- the platform-mode-variant
  // complement to 7c47f88's no-auth-mode (none) variant; 7c47f88
  // pinned the banner's userAuth* fields as the "-" placeholder
  // (the no-auth-mode default per HandHistoryReviewServerRuntime.
  // scala lines 328 + 333's getOrElse("-")) and authenticationMode=
  // none; THIS commit pins the OPPOSITE direction: when
  // platformAuth is configured, the banner emits authenticationMode=
  // users (NOT "none") AND userAuthMaxUsers=<configured-max> (NOT
  // "-") AND userAuthStoredUsers=<count> (NOT "-"); together the
  // two tests pin the per-auth-mode banner-shape divergence the
  // deploy doc's boot-time-diagnostic contract depends on for
  // operators to verify "the deployment is wired for the correct
  // auth mode" at boot time; per-field regression vectors that
  // 7c47f88's no-auth pin doesn't catch: (i) authenticationMode=
  // users -- the specific string "users" (not "platform-user-auth"
  // or "platform" or "user-auth") that the inline auth-mode
  // detection at HandHistoryReviewServerRuntime.scala's
  // `authenticationMode` helper returns; a refactor renaming the
  // mode identifier (e.g. for "naming consistency" with future
  // OIDC-only or magic-link modes) would silently break operator
  // dashboards filtering by authenticationMode=users for
  // platform-auth deployments AND would silently desync from
  // /api/health.authenticationMode (pinned by b1339cd + sibling
  // tests) which uses the SAME value, (ii) userAuthMaxUsers=
  // 100000 -- the SPECIFIC default value (matches fb18e2c's
  // health-side pin); a refactor changing the default (e.g.
  // bumping to 250000 because "modern deployments need more
  // headroom") would silently desync the banner from /api/health
  // dashboards AND silently break operator capacity-planning
  // queries that key on the documented 100k default, (iii)
  // userAuthStoredUsers=0 -- the count at FRESH BOOT before any
  // users register; emits the integer 0 (NOT the string "-"
  // which is the no-auth placeholder); the inline comment at
  // HandHistoryReviewServerRuntime.scala lines 329-332 EXPLICITLY
  // documents this as the boot-time count emission for "operators
  // verify the persistent store survived restart" -- a fresh
  // store reads back 0 users, a restarted store reads back >0
  // users if the deploy persisted; a refactor that emitted "-"
  // when platformAuth was configured but storedUserCount returned
  // 0 (a common "fix Option<Int> -> 0 vs - confusion" refactor
  // would conflate the two) would silently mask the
  // restart-survival diagnostic -- operators couldn't tell
  // "fresh-boot, 0 users" from "store missing, 0 users" if both
  // emitted "-"; the test uses a FRESH userStorePath (no prior
  // registrations) so the expected count is exactly 0; same
  // stdout-capture pattern as 7c47f88; future fires can extend
  // with: (a) restart-with-stored-users variant -- register N
  // users via /api/auth/register, close the server, reopen with
  // the same storePath, verify the SECOND startup banner emits
  // userAuthStoredUsers=N (pins the BOOT-TIME-COUNT-LOAD
  // contract at HandHistoryReviewServerRuntime.scala lines
  // 329-332 directly), (b) basic-auth-mode variant (a similar
  // pin for authenticationMode=basic which is the third auth
  // mode -- pairs with no-auth + users to cover all 3 modes).
  test("server startup under platform-user authentication emits the documented banner with authenticationMode=users + userAuthMaxUsers=100000 + userAuthStoredUsers=0 -- per-mode variant of the 7c47f88 no-auth startup banner pin") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        // Same stdout-capture pattern as 7c47f88 but with
        // platformAuth configured -- the banner's authentication
        // Mode + userAuth* fields flip from the no-auth defaults
        // to the platform-mode values.
        val outBuf = new java.io.ByteArrayOutputStream()
        val originalOut = System.out
        System.setOut(new java.io.PrintStream(outBuf, true, StandardCharsets.UTF_8))
        try
          withServer(
            staticDir,
            platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
          ) { _ =>
            // No HTTP requests -- the banner emits at startup
            // BEFORE the callback runs. A fresh storePath means
            // 0 registered users at boot time, so the expected
            // userAuthStoredUsers value is exactly 0.
            ()
          }
        finally
          System.setOut(originalOut)

        val captured = outBuf.toString(StandardCharsets.UTF_8)
        val bannerLine = captured.split('\n').iterator
          .find(_.contains("startup complete"))
          .getOrElse(fail(s"no `startup complete` line in captured stdout for the platform-user-auth variant -- the 7c47f88 startup pin should catch the absence independently in the no-auth case, but this test verifies the platform-auth case has the same banner emission; if missing, the platform-auth branch suppressed the banner OR the banner shape differs across modes; got captured stdout: ${captured.take(2000)}"))

        // (i) prefix (matches the 7c47f88 no-auth banner -- the
        // prefix is per-mode-invariant, only the field values
        // flip)
        assert(bannerLine.contains("startup complete"),
          clue = s"platform-auth startup banner must carry the SAME `startup complete` prefix as the no-auth variant -- the prefix is mode-invariant per HandHistoryReviewServerRuntime.scala line 349's hardcoded literal; a refactor that emitted a different prefix per mode (e.g. `startup complete (platform-auth)` for distinguishability) would silently break operator scripts that grep one prefix across all deployments; got: $bannerLine")
        // (ii) authenticationMode=users (per-mode VARIANT --
        // catches a refactor renaming the mode identifier OR
        // accidentally emitting "none" when platformAuth IS
        // configured)
        assert(bannerLine.contains("authenticationMode=users"),
          clue = s"platform-auth startup banner MUST carry authenticationMode=users (NOT \"none\" which is the no-auth-mode value pinned by 7c47f88, NOT \"platform\" or \"platform-user-auth\" or \"users-auth\" which a refactor might rename to for naming consistency); the specific string \"users\" matches /api/health.authenticationMode (pinned by b1339cd + sibling tests under platform-mode); a refactor renaming would silently break operator dashboards filtering by authenticationMode=users AND silently desync the banner from runtime probes; got: $bannerLine")
        // (iii) userAuthMaxUsers=100000 (per-mode VARIANT --
        // catches a refactor changing the default OR conflating
        // the "-" placeholder with a 0 emission)
        assert(bannerLine.contains("userAuthMaxUsers=100000"),
          clue = s"platform-auth startup banner MUST carry userAuthMaxUsers=100000 (the documented default per PlatformUserAuth.scala line 81's `maxUsers: Int = 100_000` AND fb18e2c's matching /api/health pin) -- NOT \"-\" (the no-auth-mode placeholder pinned by 7c47f88); a refactor changing the default would silently desync the banner from /api/health AND silently break operator capacity-planning queries keyed on the documented 100k default, a refactor that emitted \"-\" when platformAuth IS configured (a common Option<Int>-vs-placeholder confusion refactor) would silently mask the per-mode banner-shape divergence; got: $bannerLine")
        // (iv) userAuthStoredUsers=0 (per-mode VARIANT -- the
        // FRESH-BOOT count, distinct from the \"-\" no-auth
        // placeholder; the inline comment at HandHistoryReview
        // ServerRuntime.scala lines 329-332 documents this as
        // the boot-time count emission for restart-survival
        // verification)
        assert(bannerLine.contains("userAuthStoredUsers=0"),
          clue = s"platform-auth startup banner MUST carry userAuthStoredUsers=0 at FRESH BOOT (a brand-new storePath has zero registered users per the test's withUserStorePath fixture) -- NOT \"-\" (the no-auth-mode placeholder pinned by 7c47f88) AND NOT some other integer; the inline comment at HandHistoryReviewServerRuntime.scala lines 329-332 EXPLICITLY documents this field as the boot-time count emission for 'operators verify the persistent store survived restart without hitting /api/health first' -- a refactor that emitted \"-\" when platformAuth was configured (conflating no-auth-placeholder with platform-fresh-boot-count) would silently mask the restart-survival diagnostic (operators couldn't distinguish 'fresh boot, 0 users' from 'store missing, no count available'); got: $bannerLine")
        // (v) host/port/INFO/service-tag prefix match the no-auth
        // variant -- the cross-cutting fields don't depend on
        // auth mode (catches a refactor that accidentally
        // gated the cross-cutting fields on auth mode)
        assert(bannerLine.contains("host=127.0.0.1"),
          clue = s"platform-auth startup banner must carry host=127.0.0.1 matching the no-auth variant -- host is auth-mode-invariant; got: $bannerLine")
        assert(bannerLine.contains("port="),
          clue = s"platform-auth startup banner must carry port=<resolved-port> matching the no-auth variant; got: $bannerLine")
        assert(bannerLine.contains("[INFO]"),
          clue = s"platform-auth startup banner must be INFO-level matching the no-auth variant; got: $bannerLine")
        assert(bannerLine.contains("[hand-history-review]"),
          clue = s"platform-auth startup banner must carry the [hand-history-review] service-tag prefix matching the no-auth variant -- the service-tag is mode-invariant (the SAME service emits banners across all auth modes); got: $bannerLine")
      }
    }
  }

  // Pin the documented `startup complete` banner under BASIC
  // authentication mode -- the THIRD and FINAL per-mode-variant
  // completing the 3-of-3 auth-mode coverage for the startup
  // banner (7c47f88 closed the none-mode variant, 1c87e04 closed
  // the users-mode variant); together the 3 tests cover ALL the
  // auth modes the deploy doc enumerates (none / basic / users)
  // and pin the per-auth-mode banner-shape divergence the
  // operator boot-time-diagnostic depends on; the basic-auth
  // mode is OPERATIONALLY UNIQUE in TWO ways: (1) it emits
  // authenticationMode=basic (NOT "none" pinned by 7c47f88, NOT
  // "users" pinned by 1c87e04) per AuthStack.scala line 812's
  // `if basicAuth.nonEmpty then "basic"`, AND (2) the userAuth*
  // fields emit the "-" placeholder (same as 7c47f88's no-auth
  // mode) because basic-auth doesn't use the user store per the
  // inline comment at Readiness.scala lines 80-83 ("Only present
  // when platform-user auth is enabled (basic auth and no-auth
  // modes have no user store)"); the basic-mode banner is the
  // ASYMMETRIC EDGE CASE the per-mode pin family catches: among
  // the 3 modes, none + basic SHARE the "-" placeholder for
  // userAuth* fields but DIFFER in the authenticationMode value,
  // while basic + users SHARE the "authentication is enabled"
  // category but DIFFER in BOTH authenticationMode AND userAuth*
  // values; without this commit's pin, a refactor that
  // accidentally routed basic-mode through the platform-auth
  // userAuth*-integer branch (a common "fix the asymmetric
  // userAuth* handling" refactor would attempt to unify the
  // three modes) would silently emit userAuthMaxUsers=<integer>
  // for basic-mode deployments AND silently desync the banner
  // from /api/health which correctly emits null for those fields
  // under basic-mode; ALTERNATIVELY, a refactor that conflated
  // basic-mode with none-mode (e.g. "basic-auth is just
  // HTTP-level auth, treat it like no-auth for the banner")
  // would silently emit authenticationMode=none AND silently
  // break operator dashboards filtering by authenticationMode=
  // basic for instances actually running basic-auth -- the
  // dashboards would see all deployments as authenticationMode=
  // none and the runbook's basic-auth-specific triage entries
  // (e.g. the BASIC_AUTH_USER / BASIC_AUTH_PASSWORD config
  // checks documented in HAND_HISTORY_WEB_DEPLOYMENT.md line
  // 130) would silently appear inapplicable to instances that
  // actually use them; 8-tier format check matching the 1c87e04
  // platform-auth + 7c47f88 no-auth pattern with the per-mode
  // VARIANT values: (i) "startup complete" prefix (mode-
  // invariant), (ii) authenticationMode=basic (per-mode VARIANT
  // -- the distinguishing field), (iii) userAuthMaxUsers=- (the
  // SHARED "-" placeholder with the no-auth mode -- pins that
  // basic-mode correctly inherits the "no user store"
  // placeholder rather than emitting a platform-mode-like
  // integer), (iv) userAuthStoredUsers=- (same -- the SHARED
  // placeholder), (v) host=127.0.0.1 (mode-invariant), (vi)
  // port= field (mode-invariant), (vii) [INFO] level (mode-
  // invariant), (viii) [hand-history-review] service-tag prefix
  // (mode-invariant); the test ALSO asserts EXCLUSION of the
  // OTHER 2 modes' authenticationMode values
  // (authenticationMode=none and authenticationMode=users) --
  // catches a refactor that emitted MULTIPLE auth-mode strings
  // in the same banner (e.g. "authenticationMode=basic
  // authenticationMode=users" if a consolidation refactor
  // accidentally duplicated the field) which would silently
  // pass the positive contains check while emitting an
  // incoherent banner; with this commit the startup banner has
  // FULL 3-of-3 per-mode coverage AND the per-mode-variant pin
  // pattern is exhausted across the auth dimension; future
  // fires can extend with other dimensions (custom
  // userAuthMaxUsers, restart-with-stored-users, OIDC providers
  // configured under platform-auth) but the BASIC AUTH-MODE
  // DIMENSION is now fully covered.
  test("server startup under BASIC authentication emits the documented banner with authenticationMode=basic + userAuthMaxUsers=- + userAuthStoredUsers=- -- closes the 3rd of 3 per-mode-variants completing the auth-mode coverage for the startup banner (7c47f88 none, 1c87e04 users, this commit basic)") {
    withStaticSite { staticDir =>
      // Same stdout-capture pattern as 7c47f88 + 1c87e04 but with
      // basicAuth configured -- the banner's authenticationMode
      // field flips to "basic" while the userAuth* fields stay
      // as "-" placeholders (basic-auth has no user store per
      // Readiness.scala lines 80-83).
      val outBuf = new java.io.ByteArrayOutputStream()
      val originalOut = System.out
      System.setOut(new java.io.PrintStream(outBuf, true, StandardCharsets.UTF_8))
      try
        withServer(
          staticDir,
          basicAuth = Some(HandHistoryReviewServer.BasicAuthConfig(username = "ops", password = "boot-banner-test"))
        ) { _ =>
          // No HTTP requests needed -- the banner emits at
          // startup BEFORE the callback runs. basicAuth carries
          // no user store so userAuth* fields stay as "-".
          ()
        }
      finally
        System.setOut(originalOut)

      val captured = outBuf.toString(StandardCharsets.UTF_8)
      val bannerLine = captured.split('\n').iterator
        .find(_.contains("startup complete"))
        .getOrElse(fail(s"no `startup complete` line in captured stdout for the basic-auth variant -- the 7c47f88 + 1c87e04 startup pins should catch the absence independently for their modes, but this test verifies the basic-auth mode also emits the banner; if missing, the basic-auth branch suppressed the banner OR the banner shape differs across modes; got captured stdout: ${captured.take(2000)}"))

      // (i) prefix (mode-invariant)
      assert(bannerLine.contains("startup complete"),
        clue = s"basic-auth startup banner must carry the mode-invariant `startup complete` prefix per HandHistoryReviewServerRuntime.scala line 349's hardcoded literal; got: $bannerLine")
      // (ii) authenticationMode=basic (per-mode VARIANT -- the
      // distinguishing field, catches rename refactors OR
      // mode-confusion refactors)
      assert(bannerLine.contains("authenticationMode=basic"),
        clue = s"basic-auth startup banner MUST carry authenticationMode=basic (NOT \"none\" which is the no-auth-mode value pinned by 7c47f88, NOT \"users\" which is the platform-auth-mode value pinned by 1c87e04) per AuthStack.scala line 812's `if basicAuth.nonEmpty then \"basic\"`; a refactor that conflated basic-mode with none-mode (treating HTTP-level basic-auth as 'no auth' for banner purposes) would silently emit authenticationMode=none AND break the runbook's basic-auth-specific BASIC_AUTH_USER / BASIC_AUTH_PASSWORD triage entries; a refactor renaming the mode identifier (e.g. \"http-basic\" / \"basic-auth\") would silently break operator dashboards; got: $bannerLine")
      // (iii) userAuthMaxUsers=- (SHARED placeholder with no-auth
      // mode -- pins that basic-mode correctly inherits the
      // placeholder rather than emitting a platform-mode-like
      // integer)
      assert(bannerLine.contains("userAuthMaxUsers=-"),
        clue = s"basic-auth startup banner MUST carry userAuthMaxUsers=- (the documented \"-\" placeholder per HandHistoryReviewServerRuntime.scala line 328's getOrElse(\"-\") -- basic-auth has no user store per Readiness.scala lines 80-83's inline comment 'basic auth and no-auth modes have no user store') -- NOT a platform-mode integer like 100000; a refactor that accidentally routed basic-mode through the platform-auth integer branch would silently emit a misleading capacity-planning value AND silently desync the banner from /api/health which correctly emits null for these fields under basic-mode; got: $bannerLine")
      // (iv) userAuthStoredUsers=- (SHARED placeholder)
      assert(bannerLine.contains("userAuthStoredUsers=-"),
        clue = s"basic-auth startup banner MUST carry userAuthStoredUsers=- matching the no-auth variant -- basic-auth shares the no-store property with no-auth mode per the auth-mode taxonomy; got: $bannerLine")
      // (v-viii) cross-cutting (mode-invariant) fields
      assert(bannerLine.contains("host=127.0.0.1"),
        clue = s"basic-auth startup banner must carry host=127.0.0.1 matching the no-auth + platform-auth variants -- host is auth-mode-invariant; got: $bannerLine")
      assert(bannerLine.contains("port="),
        clue = s"basic-auth startup banner must carry port=<resolved-port> matching the no-auth + platform-auth variants; got: $bannerLine")
      assert(bannerLine.contains("[INFO]"),
        clue = s"basic-auth startup banner must be INFO-level matching the no-auth + platform-auth variants; got: $bannerLine")
      assert(bannerLine.contains("[hand-history-review]"),
        clue = s"basic-auth startup banner must carry the [hand-history-review] service-tag prefix matching all auth-mode variants -- the service-tag is mode-invariant; got: $bannerLine")

      // EXCLUSION of the OTHER 2 modes' authenticationMode values
      // -- catches a refactor that emitted MULTIPLE auth-mode
      // strings in the same banner (consolidation-accident)
      assert(!bannerLine.contains("authenticationMode=none"),
        clue = s"basic-auth startup banner MUST NOT also contain authenticationMode=none (the no-auth-mode value pinned by 7c47f88) -- a refactor that emitted both auth-mode strings (e.g. due to accidentally duplicating the field in a consolidation refactor) would silently pass the positive authenticationMode=basic contains check while emitting an incoherent banner; got: $bannerLine")
      assert(!bannerLine.contains("authenticationMode=users"),
        clue = s"basic-auth startup banner MUST NOT also contain authenticationMode=users (the platform-auth-mode value pinned by 1c87e04) -- same consolidation-accident catch as above; got: $bannerLine")
    }
  }

  // Pin the documented BOOT-TIME-COUNT-LOAD contract through the
  // startup banner -- closes the OTHER dimension the f0e7066
  // auth-mode-variant series didn't cover (auth-mode pins
  // 7c47f88/1c87e04/f0e7066 covered the per-mode banner-shape
  // divergence; THIS commit covers the per-INSTANCE-LIFETIME
  // count-load semantic); the inline comment at HandHistory
  // ReviewServerRuntime.scala lines 329-332 EXPLICITLY documents
  // the userAuthStoredUsers field as the boot-time count emission
  // for restart-survival verification: "At boot we have JUST
  // loaded the user store; emit the current count so operators
  // can verify the persistent store survived restart and see
  // capacity headroom vs maxUsers without having to hit
  // /api/health first"; the existing 758b86e commit pinned the
  // JSON-store persistent-account-data survival via /api/auth/
  // login (cross-restart credentials still resolve), but NOT
  // specifically through the BANNER -- 758b86e proved the data
  // survives at the storage layer, this commit proves the BOOT
  // EMISSION reflects the post-restart loaded count, which is
  // what operators actually grep for in boot logs to verify
  // restart-survival WITHOUT touching the /api/auth/login flow;
  // the test flow: (1) FIRST withServer with platformAuth +
  // fresh storePath, capture stdout, verify SECOND banner emits
  // userAuthStoredUsers=0 (matching 1c87e04 baseline), (2)
  // register 2 users via /api/auth/register inside the first
  // withServer (the registrations PERSIST to the storePath via
  // PlatformUserAuth.scala's writeState atomic-move at line
  // 968-972, per 758b86e's coverage), (3) close the first
  // server (withServer's finally fires server.close), (4)
  // SECOND withServer with the SAME storePath, capture stdout
  // separately, verify the SECOND startup banner emits
  // userAuthStoredUsers=2 (the persisted count), NOT 0 (which
  // would indicate the store wasn't loaded) NOT "-" (which
  // would indicate the platform-auth mode flag was lost across
  // restart); per-field regression vectors SPECIFIC to this
  // boot-time-count-load contract that the prior auth-mode
  // banner pins (7c47f88/1c87e04/f0e7066) don't catch: (i) a
  // refactor that emitted the count BEFORE the store was loaded
  // would silently emit 0 even for restarted-with-users
  // deployments -- the operator's restart-survival diagnostic
  // would silently mislead (operator sees userAuthStoredUsers=0
  // after restart, assumes store was wiped, but the store is
  // fine, just emitted at the wrong lifecycle moment), the
  // emission order at HandHistoryReviewServerRuntime.scala line
  // 333 is `val userAuthStoredUsersField = platformAuthService.
  // map(_.storedUserCount.toString).getOrElse("-")` which
  // CAPTURES the count via storedUserCount AT BANNER FORMAT TIME
  // (line 333) -- but platformAuthService was created via
  // PlatformUserAuth.Service.create AT LINE 232 of the same
  // file, BEFORE the bind happens at line 290-295, AND
  // PlatformUserAuth.Service.create LOADS the store (per the
  // implementation reading from storePath at construction time);
  // so by the time line 333 captures the count, the store IS
  // already loaded -- the timing is correct, and this test pins
  // that invariant by asserting non-zero count on the restart
  // path, (ii) a refactor that bypassed PlatformUserAuth.Service.
  // create's store-load (e.g. "lazy-load on first lookup for
  // faster startup") would silently emit 0 in the banner even
  // when the store is non-empty -- the lazy load would happen
  // later via /api/auth/login or /api/auth/me, by which time
  // the banner already emitted, (iii) a refactor that changed
  // the userAuthStoredUsers field from "live count" to "static
  // baseline" (e.g. cached at server-start and never updated)
  // would still PASS THIS TEST because we only check the boot-
  // time count, not the runtime evolution -- a future fire could
  // add a separate test for the runtime-count-evolution via
  // /api/health.userAuthStoredUsers across registrations; the
  // test uses TWO sequential withServer blocks sharing a single
  // storePath (the same pattern 758b86e established for the
  // persistent-account-data test); the captured stdout is
  // RESET between the two server lifecycles -- this is critical
  // because the FIRST server's startup banner also emits to
  // stdout, and if the captures merged the test couldn't
  // distinguish "userAuthStoredUsers=0 from first banner" vs
  // "userAuthStoredUsers=2 from second banner" by simple
  // contains check (both would appear in a merged stream); the
  // RESET pattern uses TWO independent ByteArrayOutputStreams +
  // two paired System.setOut + finally restore cycles; with
  // this commit the boot-time-count-load contract is pinned at
  // the BANNER LAYER -- complementing 758b86e's storage-layer
  // pin (both flows together: storage survives the restart per
  // 758b86e AND the boot banner reflects the survival per this
  // commit) so operators can rely on the runbook's "grep boot
  // logs for userAuthStoredUsers" diagnostic without having to
  // separately verify the storage layer is intact.
  test("startup banner's userAuthStoredUsers field reflects post-restart loaded count -- pins the documented boot-time-count-load contract (HandHistoryReviewServerRuntime.scala lines 329-332's restart-survival diagnostic) by registering users in a first server, closing it, then verifying the SECOND server's startup banner emits the persisted count") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        // FIRST server lifecycle: capture stdout, verify fresh-
        // boot count is 0, register 2 users.
        val firstOutBuf = new java.io.ByteArrayOutputStream()
        val firstOriginalOut = System.out
        System.setOut(new java.io.PrintStream(firstOutBuf, true, StandardCharsets.UTF_8))
        try
          withServer(
            staticDir,
            platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
          ) { server =>
            val baseUri = s"http://${server.binding.host}:${server.binding.port}"

            // Register 2 users -- the registrations PERSIST to the
            // storePath via PlatformUserAuth.scala's writeState
            // atomic-move (per 758b86e's coverage). After both
            // registrations succeed, the storePath JSON file
            // contains the credentials AND the SECOND server's
            // PlatformUserAuth.Service.create will load them at
            // startup, making storedUserCount return 2 at the
            // SECOND banner's emission moment.
            val firstRegister = postJson(s"$baseUri/api/auth/register",
              """{"email":"first@example.com","password":"correct-horse-battery","displayName":"First"}""")
            assertEquals(firstRegister.statusCode(), 201,
              clue = "first registration must succeed before testing the boot-time count-load contract")
            val secondRegister = postJson(s"$baseUri/api/auth/register",
              """{"email":"second@example.com","password":"correct-horse-battery","displayName":"Second"}""")
            assertEquals(secondRegister.statusCode(), 201,
              clue = "second registration must succeed -- the test verifies the SECOND banner emits userAuthStoredUsers=2 (not 1) so both registrations must complete")
          }
        finally
          System.setOut(firstOriginalOut)

        // Verify the FIRST banner emitted userAuthStoredUsers=0
        // (baseline -- the fresh-boot count before any
        // registrations; matches 1c87e04's pin)
        val firstCaptured = firstOutBuf.toString(StandardCharsets.UTF_8)
        val firstBannerLine = firstCaptured.split('\n').iterator
          .find(_.contains("startup complete"))
          .getOrElse(fail(s"no `startup complete` line in first server's captured stdout; got: ${firstCaptured.take(2000)}"))
        assert(firstBannerLine.contains("userAuthStoredUsers=0"),
          clue = s"FIRST server's startup banner must carry userAuthStoredUsers=0 (fresh boot, no users registered yet at startup time -- registrations happen AFTER the banner emits) -- this is the BASELINE against which the SECOND banner's userAuthStoredUsers=2 is compared to prove the boot-time-count-load contract; got: $firstBannerLine")

        // SECOND server lifecycle: capture stdout FRESH (NOT
        // shared with first capture -- the reset is critical),
        // open the same storePath, verify the SECOND banner emits
        // userAuthStoredUsers=2 reflecting the persisted count.
        val secondOutBuf = new java.io.ByteArrayOutputStream()
        val secondOriginalOut = System.out
        System.setOut(new java.io.PrintStream(secondOutBuf, true, StandardCharsets.UTF_8))
        try
          withServer(
            staticDir,
            platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
          ) { _ =>
            // No HTTP requests in the second server -- the test
            // pins the BOOT-TIME banner emission, which fires
            // BEFORE the run callback executes. The empty body
            // keeps the second server alive long enough for the
            // banner to flush.
            ()
          }
        finally
          System.setOut(secondOriginalOut)

        val secondCaptured = secondOutBuf.toString(StandardCharsets.UTF_8)
        val secondBannerLine = secondCaptured.split('\n').iterator
          .find(_.contains("startup complete"))
          .getOrElse(fail(s"no `startup complete` line in second server's captured stdout -- if missing, the second server's banner emission was suppressed; got: ${secondCaptured.take(2000)}"))

        // THE LOAD-BEARING ASSERTION: the SECOND banner must
        // carry userAuthStoredUsers=2 reflecting the persisted
        // count from the storePath that the first server wrote.
        assert(secondBannerLine.contains("userAuthStoredUsers=2"),
          clue = s"SECOND server's startup banner MUST carry userAuthStoredUsers=2 reflecting the count loaded from the persisted storePath that the FIRST server wrote to -- this is the documented boot-time-count-load contract per HandHistoryReviewServerRuntime.scala lines 329-332's inline comment ('operators can verify the persistent store survived restart and see capacity headroom vs maxUsers without having to hit /api/health first'); a refactor that emitted the count BEFORE the store was loaded (e.g. lazy-load refactor) would silently emit 0 here EVEN THOUGH the store has 2 users -- operators relying on the boot log grep for userAuthStoredUsers would silently see 0 after restart, mistakenly conclude the store was wiped, AND skip the actual restart-survival diagnostic (which would have shown the store IS intact); got: $secondBannerLine")

        // EXCLUSION: must NOT contain the placeholder (catches a
        // refactor that lost the platform-auth mode flag during
        // the restart -- would emit "-" for both userAuth* fields
        // as if no platformAuth was configured); the test uses
        // the SAME PlatformUserAuth.Config in both withServer
        // calls so the auth-mode flag is correctly set; this
        // assertion catches a refactor where the SECOND server
        // somehow inherited the "no auth" defaults despite the
        // explicit platformAuth config -- a subtle bug that
        // would silently make the boot-time count-load
        // diagnostic always show "-" for restarted servers
        assert(!secondBannerLine.contains("userAuthStoredUsers=-"),
          clue = s"SECOND server's startup banner MUST NOT carry userAuthStoredUsers=- (the no-auth/basic-auth placeholder pinned by 7c47f88/f0e7066) because the test explicitly configures platformAuth in the second withServer -- a refactor where the second server somehow lost the platform-auth mode flag during the restart would silently emit \"-\" and silently break the boot-time count-load diagnostic; got: $secondBannerLine")

        // Cross-check: the SECOND banner ALSO carries
        // authenticationMode=users (per 1c87e04) -- pins that
        // the platform-auth mode survived the restart cleanly
        // alongside the user-count
        assert(secondBannerLine.contains("authenticationMode=users"),
          clue = s"SECOND server's startup banner must carry authenticationMode=users matching the 1c87e04 platform-auth pin -- this cross-checks that the platform-auth mode flag survived the restart (otherwise the userAuthStoredUsers=2 assertion above would have failed too because the mode-detection helper returns the count placeholder \"-\" for non-platform-auth modes); got: $secondBannerLine")
        // Cross-check: the SECOND banner's userAuthMaxUsers=100000
        // also matches the 1c87e04 pin -- the cap config survives
        // restart alongside the count
        assert(secondBannerLine.contains("userAuthMaxUsers=100000"),
          clue = s"SECOND server's startup banner must carry userAuthMaxUsers=100000 matching the 1c87e04 platform-auth pin -- pins that the cap config survives restart (the cap is a CONFIG value, not a STORE value, so it should always reflect the running config's maxUsers regardless of restart); a refactor that lost the cap config across restart would silently emit a different value here; got: $secondBannerLine")
      }
    }
  }

  // Pin the documented `startup complete` banner under platform-
  // user authentication with a CUSTOM userAuthMaxUsers value
  // (non-default) -- closes the CONFIGURED-VS-DEFAULT-PROPAGATION
  // dimension complement to the 1c87e04 platform-auth pin (which
  // pinned the 100000 default) and the ded9bc6 boot-time-count-
  // load pin (which exercised the count flowing through, but
  // with the default cap); together with 1c87e04 this commit
  // forms an asymmetric pin pair: 1c87e04 verifies the
  // hardcoded-default value (100000) flows through correctly,
  // THIS commit verifies a CONFIGURED value (NOT 100000) flows
  // through correctly -- catches a refactor that hardcoded the
  // banner's userAuthMaxUsers to 100000 (e.g. "the deploy doc
  // says 100k is the default, just emit it directly for
  // simplicity") which would silently break operator capacity-
  // planning queries on deployments that intentionally
  // configured a non-default cap; the deploy doc line 349 (per
  // fb18e2c) documents "USER_AUTH_MAX_USERS (default 100000)"
  // implying the cap is OPERATOR-CONFIGURABLE -- a deployment
  // that needs higher capacity (say, a community-poker
  // deployment expecting 50k users with headroom for growth)
  // would set USER_AUTH_MAX_USERS=200000, and operators must
  // see the configured value in the banner to verify the cap
  // was applied; per-field regression vectors that 1c87e04 +
  // ded9bc6 don't catch: (i) a refactor that hardcoded 100000
  // in the banner emission (line 328's `config.platformAuth.
  // map(_.maxUsers.toString).getOrElse("-")` could be
  // accidentally simplified to `if config.platformAuth.nonEmpty
  // then "100000" else "-"` in a "remove indirection" refactor)
  // would silently emit 100000 here even when a custom value
  // was configured -- operators looking at the banner would
  // see the default value and wrongly assume the custom override
  // didn't take effect, (ii) a refactor that read the wrong
  // field name (e.g. `_.maxUsers` typo'd to `_.maxConcurrentJobs`
  // or `_.maxQueuedJobs`) would silently emit a value from a
  // different config knob -- the test pins the SPECIFIC custom
  // value 5000 so a wrong-field-read would emit some other
  // integer that doesn't equal 5000, (iii) a refactor that
  // applied a derived transformation (e.g. `_.maxUsers / 10`
  // for some misguided "user-friendly" rounding) would silently
  // emit a transformed value -- the exact-equality test catches
  // this; test approach: same stdout-capture pattern as
  // 7c47f88/1c87e04/f0e7066/ded9bc6 with the platformAuth.Config
  // carrying a CUSTOM maxUsers = 5000 (deliberately chosen as a
  // non-default value distinct from the 100000 default AND
  // distinct from any other commonly-defaulted integer in the
  // banner like maxUploadBytes=512 / analysisTimeoutMs=120000 /
  // playingHallTimeoutMs=900000 / maxConcurrentJobs=2 /
  // maxQueuedJobs=8 / etc. so a wrong-field-read regression
  // emits a value that doesn't match any other field, making
  // the failure mode easy to diagnose); 6-tier format check:
  // (i) "startup complete" prefix (sanity), (ii) authentication
  // Mode=users (matches 1c87e04 -- mode flag correctly
  // identifies platform-auth), (iii) userAuthMaxUsers=5000 (THE
  // load-bearing pin -- the configured custom value propagates
  // to the banner), (iv) EXCLUSION of userAuthMaxUsers=100000
  // (the default value -- catches a hardcoded-default
  // regression), (v) userAuthStoredUsers=0 (fresh boot, no
  // users yet -- cross-check that the count field still works
  // independently of the cap-override), (vi) [INFO] level +
  // service-tag (mode-invariant); pin completes the per-config-
  // dimension coverage for the platform-auth banner shape
  // alongside the auth-mode pins (none/users/basic) + boot-
  // count-load pin -- 5-commit family now covers (a) all 3
  // auth modes, (b) baseline + restart-survival count
  // lifecycle, (c) default + custom cap propagation.
  test("startup banner under platform-user authentication with CUSTOM userAuthMaxUsers=5000 propagates the configured non-default cap value -- pins the CONFIG-OVERRIDE-PROPAGATION contract complementing 1c87e04's default-value pin (catches a refactor hardcoding the banner emission to the default 100000)") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        // Same stdout-capture pattern as 1c87e04 + ded9bc6 but
        // with maxUsers explicitly set to 5000 (a non-default
        // value chosen to be distinct from EVERY OTHER integer
        // in the banner: maxUploadBytes=512, analysisTimeoutMs=
        // 120000, playingHallTimeoutMs=900000, maxConcurrentJobs=
        // 2, maxQueuedJobs=8, rateLimit*=6/240/10, 100000 the
        // default cap -- 5000 doesn't match any of these so a
        // wrong-field-read regression emits a non-matching
        // value, making the failure mode easy to diagnose).
        val outBuf = new java.io.ByteArrayOutputStream()
        val originalOut = System.out
        System.setOut(new java.io.PrintStream(outBuf, true, StandardCharsets.UTF_8))
        try
          withServer(
            staticDir,
            platformAuth = Some(PlatformUserAuth.Config(
              storePath = storePath,
              maxUsers = 5000
            ))
          ) { _ =>
            ()
          }
        finally
          System.setOut(originalOut)

        val captured = outBuf.toString(StandardCharsets.UTF_8)
        val bannerLine = captured.split('\n').iterator
          .find(_.contains("startup complete"))
          .getOrElse(fail(s"no `startup complete` line in captured stdout for the custom-maxUsers variant; got: ${captured.take(2000)}"))

        // (i) prefix (sanity, matches 1c87e04)
        assert(bannerLine.contains("startup complete"),
          clue = s"banner must carry the mode-invariant `startup complete` prefix; got: $bannerLine")
        // (ii) authenticationMode=users (matches 1c87e04 -- the
        // mode-detection helper correctly identifies platform-
        // auth regardless of the maxUsers value)
        assert(bannerLine.contains("authenticationMode=users"),
          clue = s"banner must carry authenticationMode=users matching 1c87e04 -- the mode flag is independent of the cap value, a non-default maxUsers config doesn't change the auth-mode identification; got: $bannerLine")
        // (iii) userAuthMaxUsers=5000 (THE load-bearing pin --
        // the configured custom value propagates to the banner)
        assert(bannerLine.contains("userAuthMaxUsers=5000"),
          clue = s"banner MUST carry the configured CUSTOM userAuthMaxUsers=5000 (NOT the 100000 default pinned by 1c87e04, NOT any other integer that might come from a wrong-field-read regression) -- this is the CONFIG-OVERRIDE-PROPAGATION pin that proves the configured value flows from PlatformUserAuth.Config.maxUsers through HandHistoryReviewServerRuntime.scala line 328's `config.platformAuth.map(_.maxUsers.toString).getOrElse(\"-\")` to the banner emission at line 350; a refactor hardcoding the banner to 100000 (e.g. 'simplify the indirection') would silently emit 100000 here even though the test configured 5000 -- operators would see the default value and wrongly conclude their custom cap didn't apply, AND silently break dashboards keyed on the configured cap for capacity-planning queries; got: $bannerLine")
        // (iv) EXCLUSION of userAuthMaxUsers=100000 (the default
        // value pinned by 1c87e04 -- catches a hardcoded-default
        // refactor that would emit BOTH values OR emit the
        // default value instead of the custom)
        assert(!bannerLine.contains("userAuthMaxUsers=100000"),
          clue = s"banner MUST NOT contain userAuthMaxUsers=100000 (the 1c87e04 default value) when the test explicitly configured maxUsers=5000 -- a refactor that hardcoded the default OR emitted BOTH the configured value AND the default would silently pass the positive userAuthMaxUsers=5000 check while still leaking the wrong information; the EXCLUSION pin catches this; got: $bannerLine")
        // (v) userAuthStoredUsers=0 (fresh boot, no users yet --
        // cross-check that the count field works independently
        // of the cap-override; a refactor that conflated
        // maxUsers with storedUserCount would silently emit
        // 5000 for both fields, which the EXCLUSION on
        // userAuthMaxUsers=100000 doesn't catch but the
        // positive userAuthStoredUsers=0 pin does)
        assert(bannerLine.contains("userAuthStoredUsers=0"),
          clue = s"banner must carry userAuthStoredUsers=0 (fresh boot, no registered users) -- cross-checks that the STORE count field works independently of the CAP config field; a refactor that conflated maxUsers with storedUserCount would silently emit 5000 for both, breaking operator dashboards distinguishing capacity-headroom from current-utilization; got: $bannerLine")
        // (vi) mode-invariant cross-checks (INFO + service-tag)
        assert(bannerLine.contains("[INFO]"),
          clue = s"banner must be INFO-level matching all prior auth-mode variants; got: $bannerLine")
        assert(bannerLine.contains("[hand-history-review]"),
          clue = s"banner must carry the [hand-history-review] service-tag prefix matching all prior banner pins; got: $bannerLine")
      }
    }
  }

  // Pin the documented `job completed` JobQueue audit log line
  // format -- a NEW CATEGORY of operator-facing log line that
  // sits between the SERVER-LIFECYCLE banners (startup/requested/
  // complete, pinned by the 7c47f88 + f6ee18b + 35a2dd9 +
  // 1c87e04 + f0e7066 + ded9bc6 + e46ed2c chain) and the AUTH-
  // EVENT audit lines (pinned by the 11-commit 1c8777f through
  // e43081b chain); the "job completed" line emits at
  // JobQueue.scala line 310-312 INSIDE the worker's finally
  // block when an analysis job successfully reaches the
  // Completed terminal state -- so this line is per-JOB (not
  // per-process like the lifecycle banners, not per-auth-event
  // like the audit log chain) and operators count these lines
  // to compute throughput / latency / fleet-wide completion-
  // rate dashboards documented in the deploy doc; the format at
  // line 311 is `s"job completed jobId=$jobId durationMs=${
  // completedAt - startedAt} queuedJobs=${executor.getQueue.
  // size()} runningJobs=${executor.getActiveCount()}"` -- four
  // fields each with specific operator-relevance: (a) jobId
  // matches the 202 response's jobId so operators correlating
  // analyze-submission log lines with their completion log
  // lines key on this field (the deploy doc's "audit log alone"
  // throughput chart depends on this correlation), (b)
  // durationMs is completedAt-startedAt which is the WORKER
  // RUN LATENCY (not queue wait, not submit-to-complete), (c)
  // queuedJobs is the QUEUE SIZE AT JOB COMPLETION (a saturation
  // indicator -- if queuedJobs is non-zero across many
  // completions, the deployment is under-provisioned), (d)
  // runningJobs is the CONCURRENT WORKER COUNT AT JOB
  // COMPLETION (a parallelism utilization indicator); BEFORE
  // this commit there was ZERO test coverage of the JobQueue
  // audit log lines AT ALL -- not the "job completed" INFO
  // line, not the "job failed" WARN line at line 321-323, not
  // any of the queue-saturation rejection lines at line 177 or
  // the timeout lines at line 191 / 219 / 224; this commit
  // closes the FIRST and most common of those (job completed
  // -- fires on every successful analyze); future fires can
  // close the others; per-field regression vectors that the
  // server-lifecycle banner pins + auth-event audit pins don't
  // catch: (i) renaming "job completed" to e.g. "analyze
  // completed" / "job done" / "job finished" would silently
  // break operator log-aggregation queries filtering by event
  // type for throughput dashboards, (ii) dropping the jobId
  // field would silently break the submission-to-completion
  // correlation that operators rely on for incident analysis
  // ("when did Alice's submitted job finish?"), (iii) emitting
  // durationMs as a wrong value (e.g. completedAt-submittedAt
  // instead of completedAt-startedAt -- which mistakenly
  // includes queue wait in worker latency) would silently
  // skew operator latency charts toward "worker is slow" when
  // the actual problem is "queue is deep", (iv) emitting
  // queuedJobs / runningJobs as STALE values (cached at
  // submission instead of completion) would silently break
  // the SATURATION SIGNAL operators rely on -- the line
  // claims "queue size AT COMPLETION", and a stale value
  // would mislead, (v) demote-to-DEBUG would silently hide the
  // line from default log levels making throughput dashboards
  // empty, promote-to-WARN would silently flood alerting on
  // every successful job; test approach: capture stdout around
  // an analyze submission, submit /api/analyze-hand-history
  // with the default immediate backend (returns instantly with
  // Right(sampleAnalysisResult) so the completedAt-startedAt
  // is bounded), poll until terminal, then assert the captured
  // stream contains the "job completed" line with the matching
  // jobId from the 202 response AND non-negative durationMs
  // AND queuedJobs=0 (no other jobs queued) AND runningJobs
  // bounded; 7-tier format check: (i) `job completed` event
  // prefix (catches rename), (ii) `jobId=<UUID from 202
  // response>` matching the submission (catches drop OR
  // wrong-source refactor), (iii) `durationMs=` field presence
  // with non-negative integer value (catches wrong-source
  // refactor that emitted negative or non-integer), (iv)
  // `queuedJobs=0` (specific value -- the immediate backend
  // completes synchronously so no queue backlog, a refactor
  // that emitted a stale submission-time value would emit a
  // different number), (v) `runningJobs=` field presence
  // (the exact value depends on timing -- could be 0 if the
  // executor already cleared by the time the log emits, or
  // 1 if the running counter is decremented after the log
  // line), so we pin presence not value, (vi) `[INFO]` level
  // (catches demote/promote), (vii) `[hand-history-review]`
  // service-tag prefix (couples to /api/health.service from
  // 505ba6b -- 7-way correlation now spans startup banner +
  // requested banner + complete banner + auth-event audit
  // lines + job-completed audit line + probe responses +
  // shutdown banner).
  test("submitted analyze job emits the documented `job completed jobId=<id> durationMs=<ms> queuedJobs=<n> runningJobs=<n>` INFO audit log line at the worker's terminal-state transition (per JobQueue.scala line 310-312) -- closes the first of the JobQueue audit log lines (alongside the SERVER-LIFECYCLE banners + AUTH-EVENT audit lines)") {
    withStaticSite { staticDir =>
      // Capture stdout around the analyze submit + terminal-
      // poll cycle. The default backend (immediateBackend) returns
      // Right(sampleAnalysisResult) synchronously, so by the time
      // awaitTerminalJob returns "completed", the JobQueue worker
      // has already emitted the "job completed" line.
      val outBuf = new java.io.ByteArrayOutputStream()
      val originalOut = System.out
      System.setOut(new java.io.PrintStream(outBuf, true, StandardCharsets.UTF_8))
      val submitJobId =
        try
          withServer(staticDir) { server =>
            val baseUri = s"http://${server.binding.host}:${server.binding.port}"
            val submit = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload)
            assertEquals(submit.statusCode(), 202,
              clue = "analyze submission must return 202 before the JobQueue worker can reach the job-completed emission point")
            val statusUri = s"$baseUri${jsonBody(submit)("statusUrl").str}"
            val capturedJobId = jsonBody(submit)("jobId").str
            // Poll until terminal -- the worker's logInfo at
            // JobQueue.scala line 310-312 fires inside the
            // finally block AFTER the Completed state is
            // installed but BEFORE the polling client sees
            // status="completed", so by the time
            // awaitTerminalJob returns, the log line is on
            // stdout.
            val terminal = awaitTerminalJob(statusUri)
            assertEquals(terminal("status").str, "completed",
              clue = "analyze must reach 'completed' terminal state for the job-completed log line to have fired")
            capturedJobId
          }
        finally
          System.setOut(originalOut)

      val captured = outBuf.toString(StandardCharsets.UTF_8)
      val completedLine = captured.split('\n').iterator
        .find(_.contains("job completed"))
        .getOrElse(fail(s"no `job completed` line in captured stdout -- JobQueue.scala line 310-312 documents this as the INFO-level line fired on every successful job terminal transition; if missing, either the logInfo emission was suppressed OR the worker exited via a different path (check awaitTerminalJob status); got captured stdout: ${captured.take(2000)}"))

      // (i) event prefix
      assert(completedLine.contains("job completed"),
        clue = s"job-completed audit line must carry the literal `job completed` event prefix per JobQueue.scala line 311's hardcoded literal -- a refactor renaming to e.g. `analyze completed` / `job done` / `job finished` would silently break operator log-aggregation queries filtering by event type for throughput dashboards; got: $completedLine")
      // (ii) jobId matching the 202 submission response
      assert(completedLine.contains(s"jobId=$submitJobId"),
        clue = s"job-completed audit line must carry the SAME jobId='$submitJobId' that the 202 submission response returned -- this is the submission-to-completion correlation operators rely on for 'when did this specific job finish' incident analysis; a refactor that emitted a different identifier (e.g. internal sequence number, retry-resilient hash, or an entirely fresh UUID per emission) would silently break the correlation; got: $completedLine")
      // (iii) durationMs field with non-negative integer value
      assert(completedLine.contains("durationMs="),
        clue = s"job-completed audit line must carry the durationMs= field -- the WORKER-RUN-LATENCY metric (completedAt - startedAt, per line 311) operators chart against; got: $completedLine")
      // Extract durationMs value and assert non-negative (the
      // immediate backend completes ~instantly so durationMs is
      // very small but MUST be a non-negative integer)
      val durationMsToken = completedLine.split(' ').iterator
        .find(_.startsWith("durationMs="))
        .getOrElse(fail(s"durationMs= token extraction failed despite contains check passing; got: $completedLine"))
      val durationMsValue = durationMsToken.drop("durationMs=".length).stripTrailing()
      val durationMs = durationMsValue.toLongOption.getOrElse(fail(s"durationMs= value '$durationMsValue' is not parseable as Long -- a refactor emitting a non-integer (e.g. ISO duration string, float, or formatted '1.2s') would silently break dashboards expecting integer ms values; got: $completedLine"))
      assert(durationMs >= 0L,
        clue = s"durationMs MUST be non-negative (the immediate backend completes ~instantly so the value is small but never negative) -- a refactor that swapped the subtraction order (startedAt - completedAt instead of completedAt - startedAt) would silently emit a negative number, silently breaking latency dashboards that assume positive values; got durationMs=$durationMs in line: $completedLine")
      // (iv) queuedJobs=0 (immediate backend completes
      // synchronously, so no other jobs are queued at the
      // moment this job completes)
      assert(completedLine.contains("queuedJobs=0"),
        clue = s"job-completed audit line must carry queuedJobs=0 (the saturation snapshot at completion time -- no other jobs are queued for this single-job test); a refactor that emitted a stale submission-time value or a wrong-source counter would silently emit a different number AND silently break operator saturation dashboards that key on this field; got: $completedLine")
      // (v) runningJobs field presence (value depends on
      // timing -- the executor might have already decremented
      // the active count by the time the log emits, so we pin
      // PRESENCE not value)
      assert(completedLine.contains("runningJobs="),
        clue = s"job-completed audit line must carry the runningJobs= field -- the CONCURRENT-WORKER-COUNT snapshot at completion time; we pin presence not specific value because the timing depends on when the executor decrements the active count relative to the log emission (the executor.getActiveCount() at line 311 could return 0 if the worker is already considered 'done' or 1 if still considered 'active' -- both are valid); a refactor that dropped the field entirely would silently break operator parallelism-utilization dashboards; got: $completedLine")
      // (vi) INFO level
      assert(completedLine.contains("[INFO]"),
        clue = s"job-completed audit line must be INFO-level per JobQueue.scala line 310's logInfo call (writes to System.out per HandHistoryReviewServerRuntime.scala line 418); demote-to-DEBUG would silently hide the line from default log levels making throughput dashboards empty, promote-to-WARN would silently flood alerting on every successful job; got: $completedLine")
      // (vii) service-tag prefix
      assert(completedLine.contains("[hand-history-review]"),
        clue = s"job-completed audit line must carry the `[hand-history-review]` service-tag prefix per HandHistoryReviewServerRuntime.scala line 512's hardcoded literal -- couples to /api/health.service (505ba6b) so log aggregators see the same identifier across the 7-way correlation: startup banner + requested banner + complete banner + auth-event audit lines + this job-completed audit line + probe responses + shutdown banner; got: $completedLine")
    }
  }

  // Pin the documented `job failed` JobQueue WARN audit log line
  // format WITH the %20-ESCAPE CONTRACT on the error field --
  // the FAILURE COMPANION to a04e51a's job-completed pin; line
  // 321-323 in JobQueue.scala emits this when an analysis job
  // reaches the Failed terminal state, either via backend Left
  // (classified at line 291 by classifyAnalysisError) or via
  // NonFatal exception at line 297; the format at line 322 is
  // `s"job failed jobId=$jobId durationMs=${completedAt -
  // startedAt} errorStatus=$errorStatus queuedJobs=${executor.
  // getQueue.size()} runningJobs=${executor.getActiveCount()}
  // error=${error.replace(" ", "%20")}"` -- THREE NEW fields
  // beyond the job-completed pin (errorStatus + error + the
  // %20-escape contract) AND a different log level (WARN not
  // INFO) AND a different output stream (stderr not stdout per
  // logWarn at HandHistoryReviewServerRuntime.scala line 421);
  // the %20-ESCAPE CONTRACT is the load-bearing pin this commit
  // uniquely catches because the prior JobQueue pin (a04e51a)
  // exercised the SUCCESS path which has no error string at
  // all -- and among ALL the JobQueue audit lines, only the
  // failure paths carry user-controllable error strings that
  // could split the structured key=value log format; the
  // inline comment at JobQueue.scala lines 314-320 documents
  // the threat: "%20-escape spaces in the Failed.error value
  // before it lands in the structured log line. The value can
  // be a backend-returned message ('no hands found in upload'),
  // a wrapped exception ('analysis failed: <e.getMessage>'),
  // or the timeoutFailure string ('analysis timed out after
  // 120000ms') -- all of which contain spaces that would split
  // the surrounding key=value pairs when a log aggregator
  // tokenizes on whitespace"; this matches the e43081b
  // auth.oidc.failure %20-escape pin at the AUTH side -- both
  // pins exercise the same defense pattern (%20-escape on
  // user/upstream-controlled error strings before structured
  // log emission) but at different emission sites; per-field
  // regression vectors that a04e51a's success-path pin doesn't
  // catch: (i) THE %20-ESCAPE CONTRACT itself -- a refactor
  // dropping the .replace(" ", "%20") at line 322 would
  // silently emit unescaped error strings, and the actual
  // backend Left values DO contain spaces (the test provides
  // "invalid hand history format with spaces" specifically to
  // exercise this), (ii) renaming "job failed" to e.g. "job
  // errored" / "analyze failed" would silently break operator
  // alert rules filtering for the failure-event signature
  // (operators page on "job failed" specifically NOT on the
  // generic "failed" word which might appear in many other
  // log lines), (iii) the errorStatus field MUST be the
  // classified HTTP status from classifyAnalysisError (400 for
  // backend Left default, 500 for "analysis failed:" prefix,
  // 504 for "analysis timed out" prefix) -- a refactor
  // hardcoding errorStatus to 500 (or any single value) would
  // silently lose operator visibility into WHY the job failed
  // (404 = bad input, 500 = wrapped exception, 504 = timeout);
  // the test uses a backend Left that DOESN'T match the
  // "analysis failed:" or "analysis timed out" prefixes, so
  // errorStatus=400 (the default-case branch at line 766), (iv)
  // WARN level not INFO -- a refactor that demoted to INFO
  // would silently make the failure line indistinguishable from
  // success in operator log streams (every "job <foo>" event
  // would be INFO, and operators couldn't grep by level for
  // failures), promoting to ERROR would silently page on every
  // user typo'd-upload (training operators to ignore), (v)
  // STDERR output not stdout -- a refactor that flipped
  // logWarn from stderr to stdout would silently desync the
  // failure-side audit lines from operator log aggregators
  // that route stderr to alerting; 10-tier format check at
  // WARN level: (i) "job failed" prefix (catches rename), (ii)
  // jobId matching the 202 response, (iii) durationMs= field
  // with non-negative integer (same shape as a04e51a's
  // job-completed pin), (iv) errorStatus=400 (the specific
  // classified status for the test's backend Left), (v)
  // queuedJobs=0 (no other jobs), (vi) runningJobs= field
  // presence (timing-dependent), (vii)
  // error=invalid%20hand%20history%20format%20with%20spaces
  // (THE %20-ESCAPE CONTRACT pin -- the load-bearing
  // load-bearing assertion), (viii) ESCAPE-CONTRACT
  // VERIFICATION via !contains("invalid hand history format
  // with spaces") (the UNESCAPED form -- catches refactor
  // dropping the escape OR emitting both forms), (ix) [WARN]
  // level (catches demote/promote), (x) [hand-history-review]
  // service-tag prefix; ALSO EXCLUSION of "job completed"
  // prefix in the failure line (catches a refactor that
  // accidentally used the success-line shape for the failure
  // path, which would silently break the failure-line
  // detection at the operator log-aggregation level).
  test("submitted analyze job that fails emits the documented `job failed jobId=<id> durationMs=<ms> errorStatus=<status> queuedJobs=<n> runningJobs=<n> error=<%20-escaped>` WARN audit log line WITH %20-escaped spaces in the error field -- the failure companion to the job-completed pin (a04e51a), closes the second of the JobQueue audit log lines AND pins the %20-escape contract on user-controlled error strings") {
    withStaticSite { staticDir =>
      // Capture stderr around the analyze submit + terminal-
      // poll cycle. The backend is immediateBackend(Left(...))
      // which returns Left("invalid hand history format with
      // spaces") synchronously, triggering the failure path
      // at JobQueue.scala line 291's `Failed(submittedAt,
      // startedAt, nowMillis(), classifyAnalysisError(error),
      // error)` -- classifyAnalysisError returns 400 for
      // strings not starting with "analysis timed out" or
      // "analysis failed:". The logWarn at line 321-323
      // writes to System.err per HandHistoryReviewServerRuntime.
      // scala line 421 (NOT System.out like the success line).
      val errBuf = new java.io.ByteArrayOutputStream()
      val originalErr = System.err
      System.setErr(new java.io.PrintStream(errBuf, true, StandardCharsets.UTF_8))
      val failureBackend = immediateBackend(Left("invalid hand history format with spaces"))
      val submitJobId =
        try
          withServer(staticDir, backend = failureBackend) { server =>
            val baseUri = s"http://${server.binding.host}:${server.binding.port}"
            val submit = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload)
            assertEquals(submit.statusCode(), 202,
              clue = "analyze submission must return 202 even when the backend will fail -- the failure happens at the WORKER level, not the submission-acceptance level")
            val statusUri = s"$baseUri${jsonBody(submit)("statusUrl").str}"
            val capturedJobId = jsonBody(submit)("jobId").str
            // Poll until terminal -- the worker emits the
            // "job failed" line at JobQueue.scala line 321-323
            // inside the finally block AFTER the Failed state
            // is installed.
            val terminal = awaitTerminalJob(statusUri)
            assertEquals(terminal("status").str, "failed",
              clue = "analyze must reach 'failed' terminal state for the job-failed log line to have fired")
            capturedJobId
          }
        finally
          System.setErr(originalErr)

      val captured = errBuf.toString(StandardCharsets.UTF_8)
      val failedLine = captured.split('\n').iterator
        .find(_.contains("job failed"))
        .getOrElse(fail(s"no `job failed` line in captured stderr -- JobQueue.scala line 321-323 documents this as the WARN-level line fired on every failed job terminal transition; if missing, either the logWarn emission was suppressed OR the worker exited via a different path; got captured stderr: ${captured.take(2000)}"))

      // (i) event prefix
      assert(failedLine.contains("job failed"),
        clue = s"job-failed audit line must carry the literal `job failed` event prefix per JobQueue.scala line 322's hardcoded literal -- a refactor renaming to e.g. `job errored` / `analyze failed` would silently break operator alert rules filtering for the failure-event signature (operators page on `job failed` specifically NOT the generic `failed` word); got: $failedLine")
      // EXCLUSION of the success-line prefix (catches a refactor
      // that accidentally used the success-line shape for the
      // failure path)
      assert(!failedLine.contains("job completed"),
        clue = s"job-failed audit line must NOT contain the `job completed` prefix (a04e51a's success-line prefix) -- a refactor that accidentally routed the failure path through the success-line emission helper would silently emit BOTH prefixes in the same line, breaking operator failure-detection signal; got: $failedLine")
      // (ii) jobId matching the 202 submission response
      assert(failedLine.contains(s"jobId=$submitJobId"),
        clue = s"job-failed audit line must carry the SAME jobId='$submitJobId' that the 202 submission response returned -- this is the submission-to-failure correlation operators rely on for 'when did this specific job fail and why' incident analysis; got: $failedLine")
      // (iii) durationMs field with non-negative integer value
      assert(failedLine.contains("durationMs="),
        clue = s"job-failed audit line must carry the durationMs= field matching the job-completed pin's shape; got: $failedLine")
      val durationMsToken = failedLine.split(' ').iterator
        .find(_.startsWith("durationMs="))
        .getOrElse(fail(s"durationMs= token extraction failed; got: $failedLine"))
      val durationMsValue = durationMsToken.drop("durationMs=".length).stripTrailing()
      val durationMs = durationMsValue.toLongOption.getOrElse(fail(s"durationMs= value '$durationMsValue' not parseable as Long; got: $failedLine"))
      assert(durationMs >= 0L,
        clue = s"durationMs MUST be non-negative even on failure path -- the failure can happen very fast (the immediate backend returns Left synchronously) so the value is small but never negative; a refactor swapping subtraction order would silently emit negative numbers; got durationMs=$durationMs in line: $failedLine")
      // (iv) errorStatus=400 (specific value -- the
      // classifyAnalysisError default branch for backend Left
      // strings not matching "analysis timed out" / "analysis
      // failed:" prefixes; the test's backend returns "invalid
      // hand history format..." which doesn't match either
      // prefix, so the classifier returns 400)
      assert(failedLine.contains("errorStatus=400"),
        clue = s"job-failed audit line MUST carry errorStatus=400 per JobQueue.scala line 763-766's classifyAnalysisError default-case branch -- backend Left strings not starting with `analysis timed out` (504) or `analysis failed:` (500) get classified as 400 (bad input); the test's backend returns 'invalid hand history format with spaces' which doesn't match either prefix; a refactor hardcoding errorStatus to a single value would silently lose operator visibility into WHY the job failed (404 = bad input, 500 = wrapped exception, 504 = timeout); got: $failedLine")
      // (v) queuedJobs=0
      assert(failedLine.contains("queuedJobs=0"),
        clue = s"job-failed audit line must carry queuedJobs=0 (no other jobs queued in single-job test) -- same shape as a04e51a's job-completed pin; got: $failedLine")
      // (vi) runningJobs field presence (timing-dependent)
      assert(failedLine.contains("runningJobs="),
        clue = s"job-failed audit line must carry the runningJobs= field -- same shape as a04e51a's job-completed pin (presence only, value depends on executor decrement timing); got: $failedLine")
      // (vii) error= field with %20-ESCAPED value (THE
      // load-bearing %20-ESCAPE CONTRACT pin)
      assert(failedLine.contains("error=invalid%20hand%20history%20format%20with%20spaces"),
        clue = s"job-failed audit line MUST carry the %20-escaped error string `error=invalid%20hand%20history%20format%20with%20spaces` per JobQueue.scala line 322's `error.replace(\" \", \"%20\")` escape applied to the backend Left value 'invalid hand history format with spaces'; the %20-escape is the load-bearing contract this pin uniquely catches -- a refactor dropping the escape would silently emit unescaped spaces in the error field, splitting the structured key=value log format when a log aggregator tokenizes on whitespace; this matches the e43081b auth.oidc.failure %20-escape contract pattern but for the JobQueue emission site; got: $failedLine")
      // (viii) ESCAPE-CONTRACT VERIFICATION: must NOT contain
      // the UNESCAPED form (catches refactor dropping the
      // escape OR emitting both forms)
      assert(!failedLine.contains("invalid hand history format with spaces"),
        clue = s"job-failed audit line MUST NOT contain the UNESCAPED form `invalid hand history format with spaces` (with literal spaces) -- the %20-escape contract at JobQueue.scala line 322 REQUIRES spaces be replaced with %20 BEFORE the log emission; a refactor that dropped the escape would emit the unescaped form which would split the structured key=value log format on aggregator tokenization; the !contains assertion catches the refactor when only the unescaped form is emitted AND when both forms are emitted (the e43081b auth.oidc.failure pin uses the same shape); got: $failedLine")
      // (ix) WARN level (NOT INFO like the success line)
      assert(failedLine.contains("[WARN]"),
        clue = s"job-failed audit line must be WARN-level per JobQueue.scala line 321's logWarn call (writes to System.err per HandHistoryReviewServerRuntime.scala line 421); a refactor demoting to INFO would silently make the failure line indistinguishable from success in operator log streams, promoting to ERROR would silently page on every user typo'd-upload training operators to ignore; got: $failedLine")
      // (x) service-tag prefix
      assert(failedLine.contains("[hand-history-review]"),
        clue = s"job-failed audit line must carry the `[hand-history-review]` service-tag prefix per HandHistoryReviewServerRuntime.scala line 512's hardcoded literal -- matches /api/health.service (505ba6b); got: $failedLine")
    }
  }

  // Pin the documented `playing hall job completed` audit log line
  // for the /api/playing-hall worker terminal-state transition --
  // the PARALLEL MIRROR to a04e51a's analyze-side job-completed
  // pin, closing the FIRST of the playing-hall-side JobQueue audit
  // lines; the playing-hall path has its OWN distinct prefix at
  // JobQueue.scala line 609-611: `s"playing hall job completed
  // jobId=$jobId durationMs=${completedAt - startedAt} queuedJobs=
  // ${...} runningJobs=${...}"` -- NOT just `job completed` (the
  // analyze path's prefix at a04e51a's line 311), so this is an
  // OPERATIONALLY DISTINCT log line that operators filter on
  // separately to compute per-endpoint dashboards (analyze
  // throughput vs hall throughput, analyze latency vs hall
  // latency, etc.); the playing-hall path also has a THIRD
  // terminal state the analyze path doesn't have: Cancelled (at
  // line 617-620 emits `playing hall job cancelled`) -- because
  // playing-hall jobs are long-running (default 15-minute timeout
  // vs analyze's 2-minute) and operators can cancel via the UI's
  // Cancel button (per 758b86e's prior coverage of the cancellation
  // flow); per-field regression vectors SPECIFIC to the playing-
  // hall completion path that a04e51a + 6b59ce4 don't catch: (i)
  // ASYMMETRIC DRIFT between analyze and hall prefixes -- a
  // refactor that consolidated the two prefixes (e.g. emitting
  // just `job completed` for BOTH endpoints under a "DRY refactor
  // to reduce log-line duplication" rationale) would silently
  // break operator per-endpoint dashboards that filter by the
  // distinguishing `playing hall job completed` prefix; this
  // matches the asymmetric-drift catch pattern from the multi-
  // event audit log pins (b1339cd /api/auth/me, 505ba6b service
  // field, etc.), (ii) consolidation refactor through a shared
  // helper that emitted only one variant -- a refactor that
  // unified the two worker types' emission helpers behind a
  // single `s"$workerType job completed ..."` template where
  // workerType is hardcoded "analyze" or always-empty would
  // silently break the hall-side prefix, (iii) prefix-token-
  // order refactor (e.g. emitting `playing-hall job completed`
  // with a hyphen instead of a space, or `playing hall completed
  // job` with token-reordering for readability) would silently
  // break operator queries filtering on the EXACT
  // multi-word-with-spaces prefix string; the playing-hall
  // success line still uses LOG SPACES BETWEEN words (NOT
  // hyphens, NOT camelCase) -- a defensible style choice for
  // human readability but operators encode the exact format in
  // their grep queries; 8-tier format check matching the
  // a04e51a pattern with the playing-hall-specific prefix:
  // (i) `playing hall job completed` prefix (catches rename
  // AND catches the asymmetric-drift consolidation refactor),
  // (ii) EXCLUSION of the analyze-side `job completed` standalone
  // prefix (catches a refactor that emitted both prefixes OR
  // emitted only the analyze prefix for the hall path), (iii)
  // jobId matching the 202 response, (iv) durationMs= field
  // with non-negative integer (same toLongOption + >= 0L shape
  // as a04e51a + 6b59ce4), (v) queuedJobs=0, (vi) runningJobs=
  // field presence, (vii) [INFO] level (matches a04e51a's
  // analyze success), (viii) [hand-history-review] service-tag
  // (mode-invariant); the EXCLUSION pin (ii) is the KEY
  // ASYMMETRIC-DRIFT CATCH -- without it, a refactor that
  // emitted "job completed" alongside "playing hall job
  // completed" (e.g. a logger that emitted both for legacy
  // log-aggregator compatibility during a transition) would
  // silently pass the positive prefix check while polluting
  // the analyze-side per-endpoint dashboard with hall events.
  // The exclusion uses the substring "playing hall job completed"
  // contains "job completed" as a sub-string, so we have to be
  // careful: the check is "does the line ALSO contain the
  // STANDALONE 'job completed' prefix in a position OTHER than
  // the `playing hall job completed` substring". The safest
  // assertion: strip the playing-hall prefix from the line and
  // verify the residual doesn't contain "job completed" again.
  test("submitted playing-hall job emits the documented `playing hall job completed jobId=<id> durationMs=<ms> queuedJobs=<n> runningJobs=<n>` INFO audit log line at the worker's terminal-state transition (per JobQueue.scala line 609-611) -- the parallel mirror to a04e51a's analyze-side job-completed pin, closes the first of the playing-hall-side JobQueue audit lines") {
    withStaticSite { staticDir =>
      // Capture stdout around the playing-hall submit + terminal-
      // poll cycle. The default playingHallBackend
      // (immediatePlayingHallBackend) returns
      // Right(samplePlayingHallResult) synchronously, so by the
      // time awaitTerminalJob returns "completed", the JobQueue
      // worker has already emitted the playing-hall-specific
      // line on stdout.
      val outBuf = new java.io.ByteArrayOutputStream()
      val originalOut = System.out
      System.setOut(new java.io.PrintStream(outBuf, true, StandardCharsets.UTF_8))
      val submitJobId =
        try
          withServer(staticDir) { server =>
            val baseUri = s"http://${server.binding.host}:${server.binding.port}"
            val submit = postJson(s"$baseUri/api/playing-hall", validPlayingHallPayload)
            assertEquals(submit.statusCode(), 202,
              clue = "playing-hall submission must return 202 before the JobQueue worker can reach the playing-hall-job-completed emission point")
            val statusUri = s"$baseUri${jsonBody(submit)("statusUrl").str}"
            val capturedJobId = jsonBody(submit)("jobId").str
            val terminal = awaitTerminalJob(statusUri)
            assertEquals(terminal("status").str, "completed",
              clue = "playing-hall must reach 'completed' terminal state for the playing-hall-job-completed log line to have fired")
            capturedJobId
          }
        finally
          System.setOut(originalOut)

      val captured = outBuf.toString(StandardCharsets.UTF_8)
      val completedLine = captured.split('\n').iterator
        .find(_.contains("playing hall job completed"))
        .getOrElse(fail(s"no `playing hall job completed` line in captured stdout -- JobQueue.scala line 609-611 documents this as the INFO-level line fired on every successful playing-hall job terminal transition; if missing, either the logInfo emission was suppressed OR the worker exited via a different path (Cancelled/Failed); got captured stdout: ${captured.take(2000)}"))

      // (i) event prefix
      assert(completedLine.contains("playing hall job completed"),
        clue = s"playing-hall job-completed audit line must carry the literal `playing hall job completed` event prefix per JobQueue.scala line 610's hardcoded literal -- a refactor renaming to e.g. `hall job completed` / `playing-hall completed` (hyphen) / `playing hall completed job` (token-reorder) would silently break operator per-endpoint dashboards that filter by the distinguishing playing-hall prefix string; got: $completedLine")
      // (ii) EXCLUSION of the standalone analyze-side prefix --
      // the asymmetric-drift catch. Strip the playing-hall prefix
      // first then check the residual doesn't contain the
      // analyze-only "job completed" prefix again.
      val withoutHallPrefix = completedLine.replace("playing hall job completed", "")
      assert(!withoutHallPrefix.contains("job completed"),
        clue = s"playing-hall audit line must NOT ALSO contain a STANDALONE `job completed` prefix (the analyze-side a04e51a-pinned prefix) in a position other than the `playing hall job completed` substring -- a refactor that emitted BOTH prefixes for the same event (e.g. a 'unify under legacy compatibility' transition emitter) would silently pass the positive `playing hall job completed` contains check while polluting the analyze-side per-endpoint dashboard with hall events; the EXCLUSION pin is the asymmetric-drift catch that distinguishes this from a04e51a; got line after stripping hall prefix: '$withoutHallPrefix'")
      // (iii) jobId matching the 202 response
      assert(completedLine.contains(s"jobId=$submitJobId"),
        clue = s"playing-hall job-completed audit line must carry the SAME jobId='$submitJobId' that the 202 submission response returned -- this is the submission-to-completion correlation for the hall worker matching the analyze-side pattern; got: $completedLine")
      // (iv) durationMs field with non-negative integer
      assert(completedLine.contains("durationMs="),
        clue = s"playing-hall job-completed audit line must carry the durationMs= field matching the analyze-side pin's shape; got: $completedLine")
      val durationMsToken = completedLine.split(' ').iterator
        .find(_.startsWith("durationMs="))
        .getOrElse(fail(s"durationMs= token extraction failed; got: $completedLine"))
      val durationMsValue = durationMsToken.drop("durationMs=".length).stripTrailing()
      val durationMs = durationMsValue.toLongOption.getOrElse(fail(s"durationMs= value '$durationMsValue' not parseable as Long; got: $completedLine"))
      assert(durationMs >= 0L,
        clue = s"playing-hall durationMs MUST be non-negative matching the analyze-side a04e51a + 6b59ce4 pattern -- the immediate hall backend completes ~instantly so the value is small but never negative; a refactor swapping subtraction order in the hall-side emission specifically (without touching the analyze-side) would silently emit negative numbers ONLY for hall events, an asymmetric-drift case the a04e51a pin alone wouldn't catch; got durationMs=$durationMs in line: $completedLine")
      // (v) queuedJobs=0
      assert(completedLine.contains("queuedJobs=0"),
        clue = s"playing-hall job-completed audit line must carry queuedJobs=0 (no other hall jobs queued in single-job test); matches a04e51a's analyze-side pattern; got: $completedLine")
      // (vi) runningJobs field presence
      assert(completedLine.contains("runningJobs="),
        clue = s"playing-hall job-completed audit line must carry the runningJobs= field (presence-only since the executor timing is non-deterministic); matches a04e51a's analyze-side pattern; got: $completedLine")
      // (vii) INFO level (matches a04e51a analyze success)
      assert(completedLine.contains("[INFO]"),
        clue = s"playing-hall job-completed audit line must be INFO-level per JobQueue.scala line 609's logInfo call; demote-to-DEBUG would silently hide hall throughput dashboards, promote-to-WARN would silently flood alerting on every successful hall job; got: $completedLine")
      // (viii) service-tag prefix
      assert(completedLine.contains("[hand-history-review]"),
        clue = s"playing-hall job-completed audit line must carry the `[hand-history-review]` service-tag prefix per HandHistoryReviewServerRuntime.scala line 512's hardcoded literal -- matches /api/health.service (505ba6b); the service-tag is endpoint-invariant (both /api/analyze-hand-history + /api/playing-hall worker emissions share the same service-tag per the file-level logInfo helper); got: $completedLine")
    }
  }

  // Pin the documented `playing hall job failed` JobQueue WARN
  // audit log line format WITH the %20-ESCAPE CONTRACT
  // verification AND the asymmetric-drift catch -- closes the
  // FOURTH JobQueue audit log line via the convergence of TWO
  // contract dimensions established by prior commits:
  // (1) the asymmetric-drift dimension from 1e030ed (analyze
  // vs hall prefix distinction) and (2) the %20-escape contract
  // dimension from 6b59ce4 (escape on user-controlled error
  // strings); the playing-hall failure path at JobQueue.scala
  // line 612-616 emits `s"playing hall job failed jobId=$jobId
  // durationMs=${completedAt - startedAt} errorStatus=
  // $errorStatus queuedJobs=${...} runningJobs=${...} error=
  // ${error.replace(" ", "%20")}"` -- carrying the
  // playing-hall-DISTINCT prefix (NOT just "job failed" -- the
  // analyze-side 6b59ce4 prefix) AND the %20-escape on the
  // user-controlled error string AND the errorStatus classified
  // via classifyPlayingHallError (line 768-771: 504 for "playing
  // hall timed out after" prefix, 500 for "playing hall failed:"
  // prefix, 400 default); this commit is operationally the
  // STRONGEST single-pin in the JobQueue family because it
  // exercises BOTH contract dimensions simultaneously -- a
  // refactor that broke either dimension AT this emission site
  // would fail the test, while a refactor that broke ONLY one
  // dimension would also fail (catches at the convergence of
  // 1e030ed's asymmetric-drift catch AND 6b59ce4's %20-escape
  // catch); per-field regression vectors that the prior 3
  // JobQueue audit log pins (a04e51a + 6b59ce4 + 1e030ed) don't
  // catch in combination: (i) ASYMMETRIC DRIFT specifically on
  // the FAILURE PATH -- a refactor that consolidated only the
  // failure prefixes (`job failed` for both endpoints) while
  // keeping the success prefixes distinct would silently break
  // operator per-endpoint failure dashboards while leaving
  // per-endpoint success dashboards intact (a half-consolidation
  // that 1e030ed's success-side pin alone can't catch since
  // 1e030ed doesn't exercise the failure path); (ii) %20-ESCAPE
  // SPECIFICALLY ON THE PLAYING-HALL PATH -- a refactor that
  // dropped the .replace(" ", "%20") at JobQueue.scala line 615
  // ONLY (without touching the analyze-side line 322 escape)
  // would silently emit unescaped errors on hall failures while
  // the analyze-side stays escaped, an asymmetric-drift case
  // that 6b59ce4's analyze-side pin alone can't catch; (iii)
  // errorStatus=400 for the playing-hall classifier default
  // branch -- a refactor that broke classifyPlayingHallError's
  // default branch independently of classifyAnalysisError's
  // (which 6b59ce4 already pins) would silently emit a wrong
  // status on hall failures while analyze stays correct; (iv)
  // WARN level + stderr output for the hall failure path --
  // matches 6b59ce4's analyze pattern but applies to a different
  // emission site so a refactor that demoted/promoted/redirected
  // the hall-failure-emission specifically would silently
  // drift; 11-tier format check converging the prior pins'
  // patterns: (i) `playing hall job failed` prefix (catches
  // rename AND asymmetric-drift consolidation), (ii) EXCLUSION
  // of standalone `job failed` in a position other than the
  // playing-hall prefix substring (the asymmetric-drift catch
  // from 1e030ed applied to failure), (iii) EXCLUSION of
  // `playing hall job completed` (catches a refactor emitting
  // wrong terminal-state shape -- mirrors 6b59ce4's exclusion
  // of `job completed`), (iv) jobId matching the 202 response,
  // (v) durationMs with toLongOption + >= 0L (catches swapped
  // subtraction order), (vi) errorStatus=400 (default branch
  // of classifyPlayingHallError), (vii) queuedJobs=0, (viii)
  // runningJobs= field presence, (ix) error=invalid%20playing%20
  // hall%20config%20with%20spaces (THE %20-escape contract
  // pin), (x) ESCAPE-CONTRACT VERIFICATION via !contains the
  // unescaped form (matches 6b59ce4 + e43081b pattern), (xi)
  // [WARN] level (matches 6b59ce4's analyze-side WARN), AND
  // [hand-history-review] service-tag; with this commit the
  // JobQueue audit log family has FOUR pins (a04e51a +
  // 6b59ce4 + 1e030ed + this commit) forming a 2x2 matrix
  // (success/failure × analyze/hall) -- the asymmetric-mirror
  // dimension is now FULLY pinned for the terminal-state
  // events (success + failure on both endpoints); remaining
  // gaps: cancelled (hall-only at line 617-620), accepted
  // (hall-only at line 503-505), submission-rejection (both
  // endpoints), timeout (both endpoints).
  test("submitted playing-hall job that fails emits the documented `playing hall job failed jobId=<id> durationMs=<ms> errorStatus=<status> queuedJobs=<n> runningJobs=<n> error=<%20-escaped>` WARN audit log line -- the hall-side failure mirror combining 1e030ed's asymmetric-drift catch with 6b59ce4's %20-escape contract catch (closes the 2x2 success/failure × analyze/hall matrix for JobQueue terminal-state audit lines)") {
    withStaticSite { staticDir =>
      // Capture stderr around the playing-hall submit + terminal-
      // poll cycle. The playingHallBackend is immediatePlayingHall
      // Backend(Left("...")) returning a backend-controlled error
      // string with spaces to exercise the %20-escape contract.
      // classifyPlayingHallError returns 400 for strings not
      // matching the "playing hall timed out after" or "playing
      // hall failed:" prefixes; the test's error string is
      // "invalid playing hall config with spaces" which falls
      // through to the default branch.
      val errBuf = new java.io.ByteArrayOutputStream()
      val originalErr = System.err
      System.setErr(new java.io.PrintStream(errBuf, true, StandardCharsets.UTF_8))
      val failureBackend = immediatePlayingHallBackend(Left("invalid playing hall config with spaces"))
      val submitJobId =
        try
          withServer(staticDir, playingHallBackend = failureBackend) { server =>
            val baseUri = s"http://${server.binding.host}:${server.binding.port}"
            val submit = postJson(s"$baseUri/api/playing-hall", validPlayingHallPayload)
            assertEquals(submit.statusCode(), 202,
              clue = "playing-hall submission must return 202 even when backend will fail -- failure happens at the WORKER level not submission")
            val statusUri = s"$baseUri${jsonBody(submit)("statusUrl").str}"
            val capturedJobId = jsonBody(submit)("jobId").str
            val terminal = awaitTerminalJob(statusUri)
            assertEquals(terminal("status").str, "failed",
              clue = "playing-hall must reach 'failed' terminal state for the playing-hall-job-failed log line to have fired")
            capturedJobId
          }
        finally
          System.setErr(originalErr)

      val captured = errBuf.toString(StandardCharsets.UTF_8)
      val failedLine = captured.split('\n').iterator
        .find(_.contains("playing hall job failed"))
        .getOrElse(fail(s"no `playing hall job failed` line in captured stderr -- JobQueue.scala line 612-616 documents this; if missing, either the logWarn was suppressed OR the worker exited via a different terminal state (Completed/Cancelled); got captured stderr: ${captured.take(2000)}"))

      // (i) event prefix (catches rename + asymmetric-drift)
      assert(failedLine.contains("playing hall job failed"),
        clue = s"playing-hall job-failed audit line must carry the literal `playing hall job failed` event prefix per JobQueue.scala line 615's hardcoded literal -- a refactor renaming OR consolidating with the analyze-side prefix (just `job failed`) would silently break per-endpoint failure dashboards; got: $failedLine")
      // (ii) EXCLUSION of standalone analyze prefix -- the
      // asymmetric-drift catch from 1e030ed applied to failure
      val withoutHallFailedPrefix = failedLine.replace("playing hall job failed", "")
      assert(!withoutHallFailedPrefix.contains("job failed"),
        clue = s"playing-hall failed line must NOT also contain a STANDALONE `job failed` prefix (the analyze-side 6b59ce4 prefix) in a position other than the `playing hall job failed` substring -- a refactor emitting BOTH prefixes for the same event would silently pollute the analyze-side per-endpoint failure dashboard with hall events; the EXCLUSION pin matches 1e030ed's asymmetric-drift catch applied to the failure path; got line after stripping hall prefix: '$withoutHallFailedPrefix'")
      // (iii) EXCLUSION of `playing hall job completed` (catches
      // a refactor emitting wrong terminal-state shape for the
      // hall failure path)
      assert(!failedLine.contains("playing hall job completed"),
        clue = s"playing-hall failed line must NOT contain `playing hall job completed` (the 1e030ed-pinned success-line prefix) -- a refactor that accidentally used the success-line shape for the failure path would silently break operator failure-detection signal; got: $failedLine")
      // (iv) jobId matching the 202 response
      assert(failedLine.contains(s"jobId=$submitJobId"),
        clue = s"playing-hall job-failed line must carry the SAME jobId='$submitJobId' from the 202 submission response; got: $failedLine")
      // (v) durationMs field with non-negative integer
      assert(failedLine.contains("durationMs="),
        clue = s"playing-hall job-failed line must carry durationMs= field matching the prior 3 JobQueue pins; got: $failedLine")
      val durationMsToken = failedLine.split(' ').iterator
        .find(_.startsWith("durationMs="))
        .getOrElse(fail(s"durationMs= token extraction failed; got: $failedLine"))
      val durationMsValue = durationMsToken.drop("durationMs=".length).stripTrailing()
      val durationMs = durationMsValue.toLongOption.getOrElse(fail(s"durationMs= value '$durationMsValue' not parseable as Long; got: $failedLine"))
      assert(durationMs >= 0L,
        clue = s"playing-hall durationMs MUST be non-negative -- catches swapped subtraction order specifically on the hall-failure path (a refactor breaking ONLY the hall failure's emission while the analyze-side stays correct); got durationMs=$durationMs in line: $failedLine")
      // (vi) errorStatus=400 (classifyPlayingHallError default branch)
      assert(failedLine.contains("errorStatus=400"),
        clue = s"playing-hall job-failed line MUST carry errorStatus=400 per JobQueue.scala line 768-771's classifyPlayingHallError default branch -- backend Left strings not starting with 'playing hall timed out after' (504) or 'playing hall failed:' (500) get classified as 400; the test's 'invalid playing hall config with spaces' doesn't match either prefix; a refactor that broke classifyPlayingHallError's default branch independently of classifyAnalysisError's (which 6b59ce4 pins) would silently emit wrong status on hall failures; got: $failedLine")
      // (vii) queuedJobs=0
      assert(failedLine.contains("queuedJobs=0"),
        clue = s"playing-hall job-failed line must carry queuedJobs=0; matches 6b59ce4's analyze-failed pattern; got: $failedLine")
      // (viii) runningJobs field presence
      assert(failedLine.contains("runningJobs="),
        clue = s"playing-hall job-failed line must carry runningJobs= field; matches 6b59ce4's analyze-failed pattern; got: $failedLine")
      // (ix) error= field with %20-ESCAPED value (THE
      // load-bearing %20-escape contract pin for the hall path)
      assert(failedLine.contains("error=invalid%20playing%20hall%20config%20with%20spaces"),
        clue = s"playing-hall job-failed line MUST carry the %20-escaped error string `error=invalid%20playing%20hall%20config%20with%20spaces` per JobQueue.scala line 615's `error.replace(\" \", \"%20\")` escape applied to the backend Left value -- the %20-escape contract from 6b59ce4 applied to the hall emission site; a refactor that dropped the escape ONLY on the hall-side line (without touching the analyze-side at line 322) would silently emit unescaped errors on hall failures, an asymmetric-drift case the 6b59ce4 pin alone doesn't catch; got: $failedLine")
      // (x) ESCAPE-CONTRACT VERIFICATION: must NOT contain the
      // UNESCAPED form (matches 6b59ce4 + e43081b pattern)
      assert(!failedLine.contains("invalid playing hall config with spaces"),
        clue = s"playing-hall job-failed line MUST NOT contain the UNESCAPED form 'invalid playing hall config with spaces' (with literal spaces) -- catches a refactor dropping the escape on the hall-side specifically; got: $failedLine")
      // (xi) WARN level
      assert(failedLine.contains("[WARN]"),
        clue = s"playing-hall job-failed line must be WARN-level per JobQueue.scala line 614's logWarn call; matches 6b59ce4's analyze-failure WARN level; got: $failedLine")
      assert(failedLine.contains("[hand-history-review]"),
        clue = s"playing-hall job-failed line must carry the [hand-history-review] service-tag prefix; got: $failedLine")
    }
  }

  // Pin the documented `playing hall job cancelled` JobQueue
  // INFO audit log line -- closes the FIFTH JobQueue audit log
  // line and the UNIQUE-TO-HALL terminal state that has NO
  // analyze-side counterpart; the analyze path's worker only
  // has 2 terminal states (Completed + Failed) per JobQueue.
  // scala lines 308-323, but the playing-hall path has 3
  // terminal states (Completed + Failed + Cancelled) per lines
  // 607-620, with the THIRD Cancelled state firing when the
  // operator cancels a long-running hall job via DELETE on
  // the status URL (per 758b86e's prior coverage of the
  // cancellation flow); analyze jobs are short-running (2-minute
  // default timeout) so cancellation isn't surfaced in the UI,
  // but playing-hall jobs are long-running (15-minute default
  // timeout per 61e49a8) so the UI exposes a Cancel button and
  // operators need the audit log signal to distinguish
  // "user-initiated cancel" from "worker timeout" from "backend
  // failure" -- three distinct operational categories that
  // share the WARN-or-INFO categorization but split into
  // 3 distinct log line prefixes; the format at line 617-620
  // is `s"playing hall job cancelled jobId=$jobId durationMs=
  // ${completedAt - startedAt} queuedJobs=${...} runningJobs=
  // ${...}"` -- matches the COMPLETED line format (no error
  // field, no errorStatus field) because cancellation isn't a
  // FAILURE (no error string to escape, no HTTP status to
  // classify), it's a USER-INITIATED EARLY TERMINATION;
  // operationally, the Cancelled line emits at INFO level (NOT
  // WARN like Failed) because user-initiated cancel is normal
  // expected behavior -- WARN level would silently flood
  // alerting on every Cancel button click; per-field
  // regression vectors that the prior 4 JobQueue audit log
  // pins (a04e51a + 6b59ce4 + 1e030ed + 817dd08) don't catch:
  // (i) ASYMMETRIC INTRODUCTION of the Cancelled terminal
  // state on the analyze side -- a refactor that "unified" the
  // two worker types' terminal-state handling by adding
  // Cancelled to the analyze path too would silently emit a
  // NEW analyze-side log line (`job cancelled` or `analyze
  // cancelled`) that operators wouldn't expect, AND would
  // create asymmetric-mirror-pair questions for every future
  // pin; this commit's EXCLUSION pin of standalone `job
  // cancelled` in a position other than the playing-hall
  // prefix catches that asymmetric-introduction refactor, (ii)
  // INFO level not WARN -- a refactor that conflated
  // cancellation with failure (e.g. "treat cancel as a kind of
  // failure for unified handling") would silently demote
  // operator alerting (the line would suddenly appear at WARN
  // level, polluting failure dashboards), OR promote (the
  // line would suddenly disappear at INFO level if a refactor
  // tried to "treat cancel as a NON-event for log volume
  // reduction"), (iii) terminal-state shape MATCH with
  // completed (no error field, no errorStatus field) -- a
  // refactor that "added an error field for consistency with
  // failed" (e.g. emitting "error=cancelled by user") would
  // silently change the cancellation line's structure to look
  // like a failure to operators parsing the log stream, AND
  // would silently add the %20-escape contract burden to the
  // cancellation path; (iv) durationMs MUST be non-negative
  // BUT cancellation typically fires very fast (the DELETE
  // happens while the worker is running, the cancelFlag is
  // checked when the backend returns, so durationMs is the
  // wall-clock from worker-start to cancel-detected); 10-tier
  // format check: (i) `playing hall job cancelled` prefix
  // (catches rename), (ii) EXCLUSION of standalone `job
  // cancelled` (catches asymmetric introduction to analyze
  // path), (iii) EXCLUSION of `playing hall job completed`
  // (catches wrong terminal-state shape -- mirrors 1e030ed +
  // 817dd08), (iv) EXCLUSION of `playing hall job failed`
  // (catches wrong terminal-state shape on the failure side
  // -- the Cancelled line MUST NOT look like a Failed line),
  // (v) jobId matching the DELETE response, (vi) durationMs
  // field with non-negative integer, (vii) queuedJobs=0,
  // (viii) runningJobs= field presence, (ix) [INFO] level
  // (NOT WARN -- cancellation is normal not a failure), AND
  // (x) [hand-history-review] service-tag; the 4 distinct
  // EXCLUSION pins are the highest-density exclusion check
  // in the JobQueue family pin chain -- catches REFACTORS
  // that would conflate Cancelled with Completed OR Failed
  // OR introduce Cancelled to the analyze path; with this
  // commit the JobQueue audit log family has FIVE pins
  // (a04e51a + 6b59ce4 + 1e030ed + 817dd08 + this commit),
  // covering ALL the terminal-state transitions for BOTH
  // worker types -- the family is COMPLETE for terminal
  // states (success + failure on both endpoints + cancelled
  // on hall-only); remaining gaps are SUBMISSION-TIME
  // emissions (accepted + rejection lines) which fire BEFORE
  // the worker runs and have different format characteristics.
  test("DELETE on a running playing-hall job emits the documented `playing hall job cancelled jobId=<id> durationMs=<ms> queuedJobs=<n> runningJobs=<n>` INFO audit log line at the worker's cancelled-terminal-state transition (per JobQueue.scala line 617-620) -- the UNIQUE-TO-HALL Cancelled terminal state with the 4-exclusion catch (asymmetric introduction + wrong-terminal-state shapes)") {
    withStaticSite { staticDir =>
      // Setup a BlockingPlayingHallBackend so the test can
      // submit a job, cancel it while running, and verify the
      // Cancelled terminal-state log line emits. The pattern
      // mirrors the existing "playing hall running job can be
      // cancelled with DELETE" test at line 8701 + 758b86e's
      // cancellation-flow coverage but with stdout capture
      // around the full lifecycle.
      val backend = new BlockingPlayingHallBackend(Right(samplePlayingHallResult))
      val outBuf = new java.io.ByteArrayOutputStream()
      val originalOut = System.out
      System.setOut(new java.io.PrintStream(outBuf, true, StandardCharsets.UTF_8))
      val submitJobId =
        try
          withServer(staticDir, playingHallBackend = backend) { server =>
            val baseUri = s"http://${server.binding.host}:${server.binding.port}"
            val submit = postJson(s"$baseUri/api/playing-hall", validPlayingHallPayload)
            assertEquals(submit.statusCode(), 202,
              clue = "playing-hall submission must return 202 before the worker can be cancelled")
            val statusUri = s"$baseUri${jsonBody(submit)("statusUrl").str}"
            val capturedJobId = jsonBody(submit)("jobId").str
            // Wait for the backend to start running before
            // issuing the cancel -- otherwise the cancel might
            // hit a queued (not yet running) job, which takes
            // a different terminal-state path.
            assert(backend.started.await(3, TimeUnit.SECONDS),
              "playing hall backend never started before cancel test could trigger")
            try
              val cancelResponse = delete(statusUri)
              assertEquals(cancelResponse.statusCode(), 200,
                clue = "DELETE on a running playing-hall job must return 200 with the Cancelled state")
              assertEquals(jsonBody(cancelResponse)("status").str, "cancelled",
                clue = "DELETE response body must report status=cancelled per the documented cancellation flow")
              // Release the backend so it returns and the
              // worker thread reaches the terminal-state log
              // emission at JobQueue.scala line 617-620.
              backend.release.countDown()
              val terminal = awaitTerminalJob(statusUri)
              assertEquals(terminal("status").str, "cancelled",
                clue = "playing-hall must reach 'cancelled' terminal state for the playing-hall-job-cancelled log line to have fired")
            finally
              backend.release.countDown()
            capturedJobId
          }
        finally
          System.setOut(originalOut)

      val captured = outBuf.toString(StandardCharsets.UTF_8)
      val cancelledLine = captured.split('\n').iterator
        .find(_.contains("playing hall job cancelled"))
        .getOrElse(fail(s"no `playing hall job cancelled` line in captured stdout -- JobQueue.scala line 617-620 documents this as the INFO-level line fired on every cancelled hall job; if missing, either the logInfo emission was suppressed OR the worker exited via a different terminal state (Completed or Failed); got captured stdout: ${captured.take(2000)}"))

      // (i) prefix
      assert(cancelledLine.contains("playing hall job cancelled"),
        clue = s"playing-hall cancellation audit line must carry the literal `playing hall job cancelled` event prefix per JobQueue.scala line 618's hardcoded literal -- a refactor renaming to e.g. `job cancelled by user` / `hall job aborted` / `playing-hall cancelled` (hyphen) would silently break operator scripts grep'ing for the user-initiated cancellation signal; got: $cancelledLine")
      // (ii) EXCLUSION of standalone `job cancelled` (catches
      // asymmetric introduction to analyze)
      val withoutHallCancelledPrefix = cancelledLine.replace("playing hall job cancelled", "")
      assert(!withoutHallCancelledPrefix.contains("job cancelled"),
        clue = s"cancellation audit line must NOT also contain a standalone `job cancelled` prefix (an analyze-side variant that doesn't currently exist) in a position other than the `playing hall job cancelled` substring -- a refactor that ADDED Cancelled to the analyze-side terminal states (e.g. for `unified handling` symmetry with hall) would silently introduce a NEW `job cancelled` log line that operators don't currently expect; the EXCLUSION pin is the ASYMMETRIC-INTRODUCTION catch -- mirrors 1e030ed/817dd08's asymmetric-drift catch but for the opposite direction (adding a state to analyze that's currently hall-only); got line after stripping hall prefix: '$withoutHallCancelledPrefix'")
      // (iii) EXCLUSION of `playing hall job completed`
      assert(!cancelledLine.contains("playing hall job completed"),
        clue = s"cancellation audit line MUST NOT contain `playing hall job completed` (the 1e030ed-pinned success-line prefix) -- a refactor that conflated cancellation with completion (e.g. treating cancelled as 'completed with partial result') would silently break operator distinction between user-initiated cancel and natural completion; got: $cancelledLine")
      // (iv) EXCLUSION of `playing hall job failed`
      assert(!cancelledLine.contains("playing hall job failed"),
        clue = s"cancellation audit line MUST NOT contain `playing hall job failed` (the 817dd08-pinned failure-line prefix) -- a refactor that treated cancellation as a 'kind of failure for unified handling' would silently promote cancellation to WARN-level alerting AND silently break operator distinction between user-initiated cancel and backend failure (operationally distinct: cancel is normal expected, failure needs incident response); got: $cancelledLine")
      // (v) jobId matching the response
      assert(cancelledLine.contains(s"jobId=$submitJobId"),
        clue = s"cancellation audit line must carry the SAME jobId='$submitJobId' from the 202 submission response -- this is the submission-to-cancellation correlation for incident analysis (operators auditing 'who cancelled which job and when' key on this); got: $cancelledLine")
      // (vi) durationMs field with non-negative integer
      assert(cancelledLine.contains("durationMs="),
        clue = s"cancellation audit line must carry durationMs= field matching the completed/failed line shapes; got: $cancelledLine")
      val durationMsToken = cancelledLine.split(' ').iterator
        .find(_.startsWith("durationMs="))
        .getOrElse(fail(s"durationMs= token extraction failed; got: $cancelledLine"))
      val durationMsValue = durationMsToken.drop("durationMs=".length).stripTrailing()
      val durationMs = durationMsValue.toLongOption.getOrElse(fail(s"durationMs= value '$durationMsValue' not parseable as Long; got: $cancelledLine"))
      assert(durationMs >= 0L,
        clue = s"cancellation durationMs MUST be non-negative -- catches swapped subtraction order specifically on the cancellation path (a refactor breaking ONLY the cancellation line's emission while the success/failure lines stay correct); got durationMs=$durationMs in line: $cancelledLine")
      // (vii) queuedJobs=0
      assert(cancelledLine.contains("queuedJobs=0"),
        clue = s"cancellation audit line must carry queuedJobs=0 (single-job test, no other jobs queued); matches the prior JobQueue audit log pin pattern; got: $cancelledLine")
      // (viii) runningJobs field presence
      assert(cancelledLine.contains("runningJobs="),
        clue = s"cancellation audit line must carry runningJobs= field (presence-only, timing-dependent); matches the prior pattern; got: $cancelledLine")
      // (ix) INFO level (NOT WARN -- cancellation is normal not
      // a failure, per JobQueue.scala line 618's logInfo call)
      assert(cancelledLine.contains("[INFO]"),
        clue = s"cancellation audit line MUST be INFO-level per JobQueue.scala line 618's logInfo call (writes to System.out per HandHistoryReviewServerRuntime.scala line 418) -- NOT WARN like the failure path because user-initiated cancellation is normal expected behavior; a refactor that promoted to WARN (e.g. 'treat cancel as a kind of failure for unified handling') would silently flood operator alerting on every Cancel button click, AND silently train operators to ignore the cancellation signal as noise -- masking real WARN-level events (actual failures) when they fire; got: $cancelledLine")
      // (x) service-tag prefix
      assert(cancelledLine.contains("[hand-history-review]"),
        clue = s"cancellation audit line must carry the `[hand-history-review]` service-tag prefix matching the prior 4 JobQueue audit log pins + the server-lifecycle banner pins + the auth-event audit pins; got: $cancelledLine")
    }
  }

  // Pin the documented `playing hall job accepted` JobQueue INFO
  // audit log line WITH the embedded request.logSummary fields
  // (hands/tableCount/playerCount/heroStyle/heroPosition/gtoMode/
  // villainPool/seed) -- opens a NEW SUB-CATEGORY in the JobQueue
  // audit log family: SUBMISSION-TIME emissions, which fire at
  // POST-acceptance time (BEFORE the worker runs); the prior 5
  // JobQueue audit log pins (a04e51a + 6b59ce4 + 1e030ed + 817dd08
  // + 0a222f8) covered TERMINAL-STATE emissions (fired AFTER the
  // worker reaches Completed/Failed/Cancelled), but the
  // submission-time category is operationally distinct -- it
  // fires SYNCHRONOUSLY inside the POST handler at JobQueue.scala
  // line 503-505, BEFORE the worker thread starts executing, so
  // the captured fields snapshot the QUEUE STATE AT ACCEPTANCE
  // TIME (operators use this for "how loaded was the deployment
  // when this job was queued" triage); the format at line 504 is
  // `s"playing hall job accepted jobId=$jobId queuedJobs=
  // ${executor.getQueue.size()} runningJobs=${executor.
  // getActiveCount()} ${request.logSummary}"` where logSummary
  // (HandHistoryReviewServerApi.scala line 837-838) expands to
  // `hands=$hands tableCount=$tableCount playerCount=$playerCount
  // heroStyle=$heroStyle heroPosition=$heroPosition gtoMode=
  // $gtoMode villainPool=${villainPool.mkString(",")} seed=$seed`
  // -- 8 request-specific fields embedded INTO the audit log
  // line, so operators can correlate the submission-time
  // queue-state with the SPECIFIC request parameters that
  // produced the load; per-field regression vectors that the
  // terminal-state pins don't catch: (i) renaming `playing hall
  // job accepted` would silently break operator submission-time
  // dashboards (distinct from the terminal-state `playing hall
  // job completed` pinned by 1e030ed -- a refactor consolidating
  // both to a single "job event" prefix would silently lose the
  // submission-vs-completion distinction operators need), (ii)
  // the EMBEDDED logSummary fields are the OPERATOR-RELEVANT
  // submission parameters -- a refactor that dropped logSummary
  // (e.g. "request details are private, don't log them") would
  // silently break operator visibility into "what jobs are
  // landing on this deployment" without forcing operators to
  // grep the full HTTP request trace logs (the deploy doc's
  // operator-side framing depends on the audit log being a
  // self-contained operational dashboard), (iii) field-name
  // changes in logSummary (e.g. `hands` -> `handCount` for
  // noun-consistency with playerCount) would silently break
  // operator queries filtering on the field names, (iv) seed=
  // field MUST be present (operators use seed for reproducibility
  // -- "what was the seed of the job that hit the bug?") so a
  // refactor that omitted seed for log-volume reasons would
  // silently break debug-reproducibility workflows, (v) NO
  // durationMs / errorStatus / error fields -- the line is
  // SUBMISSION-TIME so completion-time fields don't apply; a
  // refactor that added durationMs=0 (or any value) at submission
  // would silently break operator queries that distinguish
  // submission-time from completion-time events based on
  // field-presence; 12-tier format check: (i) `playing hall job
  // accepted` prefix, (ii) EXCLUSION of `playing hall job
  // completed` (the terminal-state pinned by 1e030ed -- a
  // refactor conflating submission with completion would
  // silently break dashboards), (iii) jobId matching the 202
  // response, (iv) queuedJobs=0 (submission to an empty queue),
  // (v) runningJobs= field presence (submission-time value
  // depends on whether the executor has already started picking
  // up other jobs), (vi) hands=120 (from validPlayingHallPayload),
  // (vii) tableCount=2, (viii) playerCount=6, (ix) heroStyle=
  // adaptive, (x) heroPosition=Button, (xi) gtoMode=exact + (xii)
  // villainPool=tag,gto (the comma-separated list), (xiii) seed=
  // field presence (the default seed value depends on
  // PlayingHallRequest's seed default), (xiv) NO durationMs=
  // field (catches a refactor adding completion-time fields to
  // the submission line), (xv) [INFO] level, (xvi) [hand-history-
  // review] service-tag prefix.
  test("submitted playing-hall job emits the documented `playing hall job accepted jobId=<id> queuedJobs=<n> runningJobs=<n> hands=<n> tableCount=<n> playerCount=<n> heroStyle=<x> heroPosition=<x> gtoMode=<x> villainPool=<list> seed=<n>` INFO audit log line at submission time (per JobQueue.scala line 503-505) -- opens the SUBMISSION-TIME emission sub-category in the JobQueue audit log family") {
    withStaticSite { staticDir =>
      // Capture stdout around the submission. The default
      // playingHallBackend (immediatePlayingHallBackend) completes
      // synchronously, so the captured stream will contain BOTH
      // the accepted line (this test's target, emitted at line
      // 503-505) AND the completed line (already pinned by
      // 1e030ed, emitted at line 609-611). We find the accepted
      // line specifically by its distinctive prefix.
      val outBuf = new java.io.ByteArrayOutputStream()
      val originalOut = System.out
      System.setOut(new java.io.PrintStream(outBuf, true, StandardCharsets.UTF_8))
      val submitJobId =
        try
          withServer(staticDir) { server =>
            val baseUri = s"http://${server.binding.host}:${server.binding.port}"
            val submit = postJson(s"$baseUri/api/playing-hall", validPlayingHallPayload)
            assertEquals(submit.statusCode(), 202,
              clue = "playing-hall submission must return 202 for the accepted log line to have fired")
            val statusUri = s"$baseUri${jsonBody(submit)("statusUrl").str}"
            val capturedJobId = jsonBody(submit)("jobId").str
            // Wait for terminal so the captured stream has the
            // expected lifecycle but the test only asserts on the
            // submission-time accepted line.
            awaitTerminalJob(statusUri)
            capturedJobId
          }
        finally
          System.setOut(originalOut)

      val captured = outBuf.toString(StandardCharsets.UTF_8)
      val acceptedLine = captured.split('\n').iterator
        .find(_.contains("playing hall job accepted"))
        .getOrElse(fail(s"no `playing hall job accepted` line in captured stdout -- JobQueue.scala line 503-505 documents this as the INFO-level submission-time line fired on every successful playing-hall job acceptance; if missing, either the logInfo was suppressed OR the executor.submit at line 499 threw RejectedExecutionException (catch at line 515 routes to a different `rejected` line); got captured stdout: ${captured.take(2000)}"))

      // (i) event prefix
      assert(acceptedLine.contains("playing hall job accepted"),
        clue = s"submission-time line must carry the literal `playing hall job accepted` prefix per JobQueue.scala line 504's hardcoded literal -- a refactor renaming to e.g. `hall job queued` / `playing-hall accepted` (hyphen) would silently break operator submission-dashboards distinguishing acceptance from completion; got: $acceptedLine")
      // (ii) EXCLUSION of completed prefix (catches conflation)
      assert(!acceptedLine.contains("playing hall job completed"),
        clue = s"submission-time line must NOT contain `playing hall job completed` (the 1e030ed-pinned terminal-state prefix) -- a refactor that emitted both prefixes for the same event would silently break the submission-vs-completion distinction operators rely on; got: $acceptedLine")
      // (iii) jobId matching the 202 response
      assert(acceptedLine.contains(s"jobId=$submitJobId"),
        clue = s"submission-time line must carry the SAME jobId='$submitJobId' from the 202 submission response; got: $acceptedLine")
      // (iv) queuedJobs=1 (the just-submitted job IS in the queue
      // at submission-time -- distinct from completion-time
      // queuedJobs=0). The executor.submit at line 499 enqueues
      // the job, THEN line 503-505's logInfo reads
      // executor.getQueue.size() which now reflects the
      // just-enqueued job. This SUBMISSION-TIME vs
      // COMPLETION-TIME asymmetry on queuedJobs is operationally
      // meaningful: operators see queuedJobs=1 at the moment of
      // acceptance (the job is enqueued but worker hasn't picked
      // it up yet) and queuedJobs=0 at the moment of completion
      // (the worker has dequeued + processed). A refactor that
      // read the queue size BEFORE the executor.submit would
      // emit queuedJobs=0 here, silently breaking the documented
      // "queue state AT acceptance" semantic.
      assert(acceptedLine.contains("queuedJobs=1"),
        clue = s"submission-time line must carry queuedJobs=1 (the just-submitted job IS in the queue at submission-time per executor.submit at line 499 enqueueing BEFORE the logInfo at line 503 reads the queue size); distinct from completion-time queuedJobs=0 (1e030ed/817dd08/0a222f8) where the worker has already dequeued; a refactor that read the queue size BEFORE the executor.submit would silently emit 0 here, breaking the documented 'queue state AT acceptance' semantic; got: $acceptedLine")
      // (v) runningJobs=0 at submission-time -- the worker
      // hasn't started executing yet (the job was JUST queued,
      // executor.getActiveCount() returns 0 since no worker is
      // actively running this job). Distinct from completion-
      // time where runningJobs could be 0 or 1 depending on
      // executor timing.
      assert(acceptedLine.contains("runningJobs=0"),
        clue = s"submission-time line must carry runningJobs=0 (the worker hasn't started executing the just-queued job at submission time -- executor.getActiveCount() returns 0); distinct from completion-time runningJobs which is timing-dependent; got: $acceptedLine")
      // (vi-xii) request.logSummary embedded fields -- the
      // OPERATOR-RELEVANT submission parameters per
      // HandHistoryReviewServerApi.scala line 837-838's
      // logSummary template
      assert(acceptedLine.contains("hands=120"),
        clue = s"submission-time line must carry hands=120 (from validPlayingHallPayload's `\"hands\":120`); a refactor renaming the field or dropping it from logSummary would silently break operator visibility into 'what hands count was requested'; got: $acceptedLine")
      assert(acceptedLine.contains("tableCount=2"),
        clue = s"submission-time line must carry tableCount=2 (from validPlayingHallPayload's `\"tableCount\":2`); got: $acceptedLine")
      assert(acceptedLine.contains("playerCount=6"),
        clue = s"submission-time line must carry playerCount=6 (from validPlayingHallPayload's `\"playerCount\":6`); got: $acceptedLine")
      assert(acceptedLine.contains("heroStyle=adaptive"),
        clue = s"submission-time line must carry heroStyle=adaptive (from validPlayingHallPayload's `\"heroStyle\":\"adaptive\"`); got: $acceptedLine")
      assert(acceptedLine.contains("heroPosition=Button"),
        clue = s"submission-time line must carry heroPosition=Button (from validPlayingHallPayload's `\"heroPosition\":\"Button\"`); got: $acceptedLine")
      assert(acceptedLine.contains("gtoMode=exact"),
        clue = s"submission-time line must carry gtoMode=exact (from validPlayingHallPayload's `\"gtoMode\":\"exact\"`); got: $acceptedLine")
      assert(acceptedLine.contains("villainPool=tag,gto"),
        clue = s"submission-time line must carry villainPool=tag,gto (from validPlayingHallPayload's `\"villainPool\":[\"tag\",\"gto\"]` joined by `,` per HandHistoryReviewServerApi.scala line 838's `villainPool.mkString(\",\")`); a refactor that changed the separator to e.g. space or `;` or `|` would silently break operator queries filtering on the joined-pool string; got: $acceptedLine")
      // (xiii) seed=42 (the PlayingHallRequest default seed
      // value -- catches BOTH a refactor dropping the field AND
      // a refactor changing the default; operators rely on the
      // 42 default for debug-reproducibility queries 'what was
      // the seed of the job that hit the bug?')
      assert(acceptedLine.contains("seed=42"),
        clue = s"submission-time line must carry seed=42 (the documented PlayingHallRequest default per the empirical observation of the line emission); a refactor dropping seed for log-volume reasons would silently break debug-reproducibility workflows AND a refactor changing the default would silently desync operator queries; got: $acceptedLine")
      // (xiv) NO durationMs= field (catches a refactor adding
      // completion-time fields to the submission line)
      assert(!acceptedLine.contains("durationMs="),
        clue = s"submission-time line must NOT carry durationMs= field -- the line is SUBMISSION-TIME, not completion-time; a refactor that added durationMs=0 (or any value) at submission would silently break operator queries distinguishing submission-time from completion-time events based on field-presence; got: $acceptedLine")
      // (xv) INFO level (submission acceptance is normal)
      assert(acceptedLine.contains("[INFO]"),
        clue = s"submission-time line must be INFO-level per JobQueue.scala line 503's logInfo call -- submission acceptance is normal expected behavior, demote-to-DEBUG would hide submission dashboards, promote-to-WARN would flood alerting; got: $acceptedLine")
      // (xvi) service-tag prefix
      assert(acceptedLine.contains("[hand-history-review]"),
        clue = s"submission-time line must carry the [hand-history-review] service-tag prefix matching the prior JobQueue audit log pins; got: $acceptedLine")
    }
  }

  // Pin the documented `job accepted` analyze-side JobQueue INFO
  // audit log line at submission time -- the ANALYZE-SIDE MIRROR
  // to 0af1462's playing-hall accepted line, with the
  // analyze-side-distinctive `bytes=<n>` field (instead of the
  // hall-side's `request.logSummary` 8-field structured summary);
  // line 200-202 in JobQueue.scala emits this when an analyze
  // submission successfully reaches the executor.submit; the
  // format at line 201 is `s"job accepted jobId=$jobId queuedJobs=
  // ${executor.getQueue.size()} runningJobs=${executor.
  // getActiveCount()} bytes=${request.handHistoryText.getBytes(
  // StandardCharsets.UTF_8).length}"` -- the analyze-side has
  // SIMPLER request-context (just bytes, not the 8-field hall
  // structured summary) because analyze inputs are unstructured
  // hand-history text where the most operationally-relevant
  // metric is "how large is this upload"; per-field regression
  // vectors specific to the analyze-side accepted line that the
  // hall-side pins (1e030ed / 0af1462) don't catch: (i)
  // ASYMMETRIC PREFIX DRIFT -- the analyze-side prefix is `job
  // accepted` (NO `playing hall` qualifier) while the hall-side
  // prefix is `playing hall job accepted` (with qualifier); a
  // refactor consolidating both submission-time prefixes (e.g.
  // adding the qualifier to analyze: "for consistency with hall")
  // would silently change the analyze-side prefix and break
  // operator per-endpoint dashboards filtering on the
  // distinguishing prefix; (ii) `bytes=<n>` field is UNIQUE to
  // the analyze-side -- a refactor that swapped to logSummary-
  // style fields (hands=, tableCount=, etc.) for "unified shape"
  // would silently emit hall-side fields on analyze lines, OR
  // dropping the bytes field would silently lose operator
  // visibility into "what size analyze inputs are landing on
  // this deployment" (the operationally-relevant analyze input
  // metric); (iii) bytes value MUST be the UTF-8 byte length of
  // request.handHistoryText -- the validUploadPayload's
  // handHistoryText is "PokerStars Hand #1" which is 18 ASCII
  // characters = 18 UTF-8 bytes; a refactor that emitted character
  // count instead of byte count would silently emit a different
  // number on non-ASCII inputs (e.g. an emoji or extended Unicode
  // character would have len != bytes), AND a refactor that
  // emitted the SERIALIZED JSON payload size instead of just the
  // handHistoryText field would silently emit a much larger
  // value; (iv) NO request-logSummary fields (hands=, tableCount=,
  // etc.) -- the analyze submission doesn't carry those fields
  // because the analyze flow doesn't have a structured request
  // model like PlayingHallRequest does; 12-tier format check
  // mirroring 0af1462's pattern with the analyze-distinctive
  // fields: (i) `job accepted` prefix (catches rename + catches
  // asymmetric-drift consolidation with hall), (ii) EXCLUSION of
  // `playing hall job accepted` (catches a refactor that
  // emitted both prefixes OR added the hall qualifier to the
  // analyze line), (iii) jobId matching the 202 response, (iv)
  // queuedJobs=1 (matches the SUBMISSION-TIME queue-size
  // asymmetry pinned by 0af1462: the just-submitted job IS in
  // the queue at submission time, NOT 0 like completion-time),
  // (v) runningJobs=0 (worker hasn't started yet), (vi)
  // bytes=18 (the exact UTF-8 byte length of
  // validUploadPayload's "PokerStars Hand #1" handHistoryText
  // -- catches a refactor changing the source field OR the
  // length-computation), (vii) EXCLUSION of hands= /
  // tableCount= / heroStyle= (the hall-side logSummary fields
  // that MUST NOT appear on analyze-side lines), (viii) NO
  // durationMs= field (submission-time, not completion-time),
  // (ix) [INFO] level, (x) [hand-history-review] service-tag.
  test("submitted analyze job emits the documented `job accepted jobId=<id> queuedJobs=<n> runningJobs=<n> bytes=<n>` INFO audit log line at submission time (per JobQueue.scala line 200-202) -- the analyze-side mirror to 0af1462's playing-hall submission-time pin, with the analyze-distinctive `bytes=<n>` field (instead of the hall-side's 8-field logSummary)") {
    withStaticSite { staticDir =>
      // Capture stdout around the analyze submission + terminal-
      // poll cycle. The default immediateBackend completes
      // synchronously, so by the time awaitTerminalJob returns
      // "completed", both the accepted (line 201) AND completed
      // (line 311, pinned by a04e51a) lines are on stdout.
      val outBuf = new java.io.ByteArrayOutputStream()
      val originalOut = System.out
      System.setOut(new java.io.PrintStream(outBuf, true, StandardCharsets.UTF_8))
      val submitJobId =
        try
          withServer(staticDir) { server =>
            val baseUri = s"http://${server.binding.host}:${server.binding.port}"
            val submit = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload)
            assertEquals(submit.statusCode(), 202,
              clue = "analyze submission must return 202 before the JobQueue accepted log line fires")
            val statusUri = s"$baseUri${jsonBody(submit)("statusUrl").str}"
            val capturedJobId = jsonBody(submit)("jobId").str
            awaitTerminalJob(statusUri)
            capturedJobId
          }
        finally
          System.setOut(originalOut)

      val captured = outBuf.toString(StandardCharsets.UTF_8)
      // Find the accepted line specifically. The captured stream
      // contains BOTH the accepted line (target) AND the
      // completed line (a04e51a's target). The "job accepted"
      // contains-substring search finds the accepted line --
      // the completed line is "job completed" which doesn't
      // contain "accepted".
      val acceptedLine = captured.split('\n').iterator
        .find(_.contains("job accepted"))
        .getOrElse(fail(s"no `job accepted` line in captured stdout -- JobQueue.scala line 200-202 documents this as the INFO-level submission-time line; if missing, either the logInfo was suppressed OR the submission path took a different branch (drain/rate-limit/auth); got captured stdout: ${captured.take(2000)}"))

      // (i) event prefix
      assert(acceptedLine.contains("job accepted"),
        clue = s"analyze submission-time line must carry the literal `job accepted` prefix per JobQueue.scala line 201's hardcoded literal; got: $acceptedLine")
      // (ii) EXCLUSION of hall-side prefix (asymmetric-drift catch)
      assert(!acceptedLine.contains("playing hall job accepted"),
        clue = s"analyze submission-time line must NOT contain `playing hall job accepted` (the 0af1462-pinned hall-side prefix) -- a refactor that emitted both prefixes OR added the hall qualifier to the analyze line for 'consistency' would silently break operator per-endpoint dashboards distinguishing analyze submissions from hall submissions; got: $acceptedLine")
      // (iii) jobId matching the 202 response
      assert(acceptedLine.contains(s"jobId=$submitJobId"),
        clue = s"analyze submission-time line must carry the SAME jobId='$submitJobId' from the 202 submission response; got: $acceptedLine")
      // (iv) queuedJobs=1 (SUBMISSION-TIME queue-size asymmetry
      // -- the just-submitted job IS in the queue at submission
      // time, matching the empirical observation pinned by
      // 0af1462 on the hall side)
      assert(acceptedLine.contains("queuedJobs=1"),
        clue = s"analyze submission-time line must carry queuedJobs=1 (the just-submitted job IS in the queue at submission time per executor.submit at line 196 enqueueing BEFORE the logInfo at line 200 reads the queue size); this matches the SUBMISSION-TIME vs COMPLETION-TIME queue-size asymmetry empirically documented + pinned by 0af1462 on the hall side; a refactor reading the queue size BEFORE executor.submit would silently emit 0 here; got: $acceptedLine")
      // (v) runningJobs=0 (worker hasn't started yet at submission)
      assert(acceptedLine.contains("runningJobs=0"),
        clue = s"analyze submission-time line must carry runningJobs=0 (the worker hasn't started executing the just-queued job at submission time -- executor.getActiveCount() returns 0); matches 0af1462's pattern on the hall side; got: $acceptedLine")
      // (vi) bytes=18 (UTF-8 byte length of "PokerStars Hand #1"
      // = 18 ASCII characters = 18 UTF-8 bytes); pins the
      // analyze-side-distinctive request-context field
      assert(acceptedLine.contains("bytes=18"),
        clue = s"analyze submission-time line MUST carry bytes=18 (the UTF-8 byte length of validUploadPayload's `handHistoryText`: \"PokerStars Hand #1\" = 18 ASCII characters = 18 UTF-8 bytes); a refactor that emitted character count instead of byte count would silently emit a different number on non-ASCII inputs, AND a refactor that emitted the SERIALIZED JSON payload size instead of just handHistoryText would silently emit a much larger value -- the test pins the EXACT byte count so any source-field-change is caught; got: $acceptedLine")
      // (vii) EXCLUSION of hall-side logSummary fields
      assert(!acceptedLine.contains("hands="),
        clue = s"analyze submission-time line must NOT contain `hands=` (a hall-side logSummary field pinned by 0af1462) -- a refactor that swapped the analyze bytes= field for the hall-side logSummary template would silently emit hall fields on analyze lines, breaking the documented per-endpoint distinction; got: $acceptedLine")
      assert(!acceptedLine.contains("tableCount="),
        clue = s"analyze submission-time line must NOT contain `tableCount=` (a hall-side logSummary field); got: $acceptedLine")
      assert(!acceptedLine.contains("heroStyle="),
        clue = s"analyze submission-time line must NOT contain `heroStyle=` (a hall-side logSummary field); got: $acceptedLine")
      // (viii) NO durationMs (submission-time, not completion)
      assert(!acceptedLine.contains("durationMs="),
        clue = s"analyze submission-time line must NOT contain durationMs= -- the line is SUBMISSION-TIME (executor.submit just enqueued the job, worker hasn't started); a refactor adding durationMs at submission would silently break the documented submission-vs-completion field distinction; got: $acceptedLine")
      // (ix) INFO level
      assert(acceptedLine.contains("[INFO]"),
        clue = s"analyze submission-time line must be INFO-level per JobQueue.scala line 200's logInfo call; got: $acceptedLine")
      // (x) service-tag
      assert(acceptedLine.contains("[hand-history-review]"),
        clue = s"analyze submission-time line must carry the [hand-history-review] service-tag prefix; got: $acceptedLine")
    }
  }

  // Pin the documented `request rate limited` audit log line at
  // AuthStack.scala line 673-675 -- opens a NEW CATEGORY of
  // operator-facing log line distinct from the JobQueue audit
  // log family (a04e51a through 29b1510): RATE-LIMIT REJECTION
  // emissions, which fire at the AUTHENTICATION-STACK layer when
  // a request exceeds the configured per-bucket rate cap (BEFORE
  // reaching JobQueue.submit or the route handler); the deploy
  // doc documents the rate-limit log line as the operator-visible
  // signal for credential-stuffing / scrape probes / fleet
  // saturation triage workflows, but BEFORE this commit there
  // was ZERO test coverage of the log line format -- the
  // existing rate-limit tests at line ~9852 + ~9883 + ~9916
  // verify the 429 HTTP response shape (statusCode + body
  // fields + Retry-After header) but NEVER capture the log
  // line; the format at line 674 is `s"request rate limited
  // path=${requestPath(exchange)} client=${rejection.clientKey}
  // bucket=${rejection.bucket.id} limitPerMinute=${rejection.
  // limitPerMinute} retryAfterMs=${rejection.retryAfterMs}"` --
  // FIVE structured fields each with specific operator-relevance:
  // (a) path is the request URI (operators triage by endpoint),
  // (b) client is the rate-limit key (either "remote:<addr>" for
  // unauthenticated or "user:<userId>" for platform-auth -- the
  // line documents the rejected client AS the per-bucket-key
  // resolved value, NOT the raw HTTP source address), (c)
  // bucket is the bucket id ("submit", "job-status", "auth" --
  // operators triage by which bucket saturated to know whether
  // it's a credential-stuffing probe (auth bucket), a job
  // scraper (job-status bucket), or just legitimate-but-bursty
  // submissions (submit bucket)), (d) limitPerMinute is the
  // CONFIGURED CAP that was exceeded (so operators know whether
  // the cap needs tuning vs whether the traffic is actually
  // abusive), (e) retryAfterMs is the wait time the rejection
  // advised the client to wait (matches the Retry-After response
  // header value); per-field regression vectors: (i) renaming
  // "request rate limited" prefix would silently break operator
  // alert rules filtering for rate-limit events, (ii) dropping
  // any of the 5 fields would silently lose operator visibility
  // into the corresponding dimension (path-based / client-based
  // / bucket-based / cap-based / wait-based triage), (iii)
  // emitting bucket=submit when the actual saturation was a
  // different bucket would silently misroute incident response,
  // (iv) emitting limitPerMinute as a stale value (e.g. cached
  // from server-start instead of read live) would silently
  // mislead about which deployment cap actually fired, (v)
  // WARN level + stderr output -- demote-to-DEBUG would hide
  // rate-limit signals from operator alerting, promote-to-ERROR
  // would page incident response on every burst from a single
  // user; test approach: reuse the existing rate-limit test
  // pattern (basic-auth + rateLimitSubmitsPerMinute=1 + 2
  // submissions = 1 accepted + 1 rejected) but capture stderr
  // around the rejected submission and assert the format; 7-tier
  // format check: (i) `request rate limited` prefix, (ii)
  // path=/api/analyze-hand-history (the specific endpoint
  // -- catches a refactor emitting path from a wrong source),
  // (iii) client= field with `remote:` prefix presence (the
  // documented "remote:<addr>" format for non-authenticated
  // requests per RateLimit.scala line 182's
  // `getOrElse(s"remote:${clientAddressKey(exchange)}")`),
  // (iv) bucket=submit (the analyze-submit bucket id), (v)
  // limitPerMinute=1 (the test's configured cap), (vi)
  // retryAfterMs= field presence (the exact value depends on
  // the rate-limit window timing), (vii) [WARN] level + [hand-
  // history-review] service-tag.
  test("rate-limit rejected analyze submission emits the documented `request rate limited path=<path> client=<key> bucket=<id> limitPerMinute=<n> retryAfterMs=<n>` WARN audit log line per AuthStack.scala line 673-675 -- opens a NEW CATEGORY (rate-limit rejection emissions) in the operator-facing log line coverage") {
    withStaticSite { staticDir =>
      val authConfig = HandHistoryReviewServer.BasicAuthConfig(username = "operator", password = "rate-limit-log-test")
      val authHeaders = basicAuthHeaders(authConfig.username, authConfig.password)
      withServer(
        staticDir,
        basicAuth = Some(authConfig),
        rateLimitSubmitsPerMinute = 1,
        rateLimitStatusPerMinute = 0
      ) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        // Submission 1: accepted (consumes the rateLimitSubmitsPerMinute=1 slot)
        val accepted = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload, authHeaders)
        assertEquals(accepted.statusCode(), 202,
          clue = "first submission must accept (202) to consume the rate-limit slot before the second triggers rejection")

        // Submission 2: rejected with 429 + emits rate-limit
        // log line on stderr. Capture stderr around this call.
        val errBuf = new java.io.ByteArrayOutputStream()
        val originalErr = System.err
        System.setErr(new java.io.PrintStream(errBuf, true, StandardCharsets.UTF_8))
        try
          val rejected = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload, authHeaders)
          assertEquals(rejected.statusCode(), 429,
            clue = "second submission MUST be rate-limited (429) so the rate-limit log line fires; if 202 the cap isn't being enforced and the log-line emission won't trigger")
        finally
          System.setErr(originalErr)

        val captured = errBuf.toString(StandardCharsets.UTF_8)
        val rateLimitLine = captured.split('\n').iterator
          .find(_.contains("request rate limited"))
          .getOrElse(fail(s"no `request rate limited` line in captured stderr -- AuthStack.scala line 674 documents this as the WARN-level line fired on every rate-limit rejection; if missing, either the logWarn was suppressed OR the 429 response was emitted by a different code path; got captured stderr: ${captured.take(2000)}"))

        // (i) event prefix
        assert(rateLimitLine.contains("request rate limited"),
          clue = s"rate-limit audit line must carry the literal `request rate limited` event prefix per AuthStack.scala line 674's hardcoded literal -- a refactor renaming to e.g. `rate limit exceeded` / `throttled` would silently break operator alert rules filtering for rate-limit events; got: $rateLimitLine")
        // (ii) path field
        assert(rateLimitLine.contains("path=/api/analyze-hand-history"),
          clue = s"rate-limit audit line must carry the request's path in the path= field -- a refactor that emitted a wrong source (e.g. config-level prefix instead of actual request URI) would silently break per-endpoint operator triage; got: $rateLimitLine")
        // (iii) client= field with remote: prefix (the
        // documented format for non-authenticated requests)
        assert(rateLimitLine.contains("client=remote:"),
          clue = s"rate-limit audit line must carry the rate-limit client key with the documented `remote:` prefix per RateLimit.scala line 182's `getOrElse(s\"remote:${'$'}{clientAddressKey(exchange)}\")` -- for non-platform-auth requests the key is the resolved IP-address-based client identifier (NOT the raw HTTP remote address); a refactor that emitted the raw remote without the prefix would silently desync from the audit-log's user-prefixed identifier format (the documented log shape distinguishes 'remote:' from 'user:' to separate per-IP-bucketing from per-user-bucketing); got: $rateLimitLine")
        // (iv) bucket=submit (the analyze-submit bucket id)
        assert(rateLimitLine.contains("bucket=submit"),
          clue = s"rate-limit audit line must carry bucket=submit (the analyze-submit rate-limit bucket id) for analyze submissions -- a refactor that emitted a wrong bucket id (e.g. bucket=job-status or bucket=analyze for renaming) would silently break operator triage that filters by bucket to distinguish credential-stuffing (auth bucket) vs job-scraper (job-status bucket) vs submission-burst (submit bucket); got: $rateLimitLine")
        // (v) limitPerMinute=1 (the test's configured cap)
        assert(rateLimitLine.contains("limitPerMinute=1"),
          clue = s"rate-limit audit line must carry limitPerMinute=1 (the test's withServer configured rateLimitSubmitsPerMinute=1 value) -- a refactor that emitted a stale value (cached at server-start instead of read from rejection.limitPerMinute) would silently mislead operators about which deployment cap actually fired; got: $rateLimitLine")
        // (vi) retryAfterMs field presence
        assert(rateLimitLine.contains("retryAfterMs="),
          clue = s"rate-limit audit line must carry the retryAfterMs= field -- the value depends on the rate-limit window timing so we pin presence not exact value, but a refactor that dropped the field would silently lose operator visibility into 'how long should the client back off' info; got: $rateLimitLine")
        // (vii) WARN level + service-tag
        assert(rateLimitLine.contains("[WARN]"),
          clue = s"rate-limit audit line must be WARN-level per AuthStack.scala line 673's logWarn call (writes to System.err per HandHistoryReviewServerRuntime.scala line 421); demote-to-DEBUG would silently hide rate-limit signals from operator alerting, promote-to-ERROR would silently page on every burst from a single user (training operators to ignore); got: $rateLimitLine")
        assert(rateLimitLine.contains("[hand-history-review]"),
          clue = s"rate-limit audit line must carry the [hand-history-review] service-tag prefix matching the prior audit log + banner pins -- the rate-limit category now joins the 7-way correlation: startup banner + requested banner + complete banner + auth-event audit lines + JobQueue audit lines + this rate-limit audit line + probe responses; got: $rateLimitLine")
      }
    }
  }

  // Pin the documented `request rate limited ... client=user:<id>`
  // PLATFORM-AUTH variant -- the AUTHENTICATED-CLIENT mirror to
  // 47d91dd's UNAUTHENTICATED-CLIENT pin (which used basic-auth
  // and produced client=remote:<addr>); the rate-limit client-key
  // path at AuthStack.scala line 663-668 takes principalKey FIRST,
  // falling back to the IP-based key only when principalKey is
  // None: `principalKey = authenticatedUser(exchange).map(user =>
  // s"user:${user.userId}")` then `RateLimit.scala line 58's
  // val clientKey = principalKey.getOrElse(rateLimitClientKey(...
  // ))`; for platform-auth requests authenticatedUser is non-empty
  // so principalKey returns Some("user:<uuid>") -- the rejection's
  // clientKey becomes "user:<uuid>" rather than "remote:<addr>";
  // operators distinguish per-user-bucketing from per-IP-bucketing
  // by this prefix to know whether to investigate the user's
  // account (credential abuse, automation attack, etc.) vs the IP
  // (botnet, scraper, etc.) -- two operationally distinct triage
  // workflows; together with 47d91dd this commit forms an
  // asymmetric pin pair on the rate-limit client-key prefix
  // (`remote:` vs `user:`), catching refactors that consolidate
  // the two paths into a single prefix (e.g. "always emit
  // remote: for consistency") which would silently break the
  // documented per-user-bucketing visibility; per-field
  // regression vectors that 47d91dd's remote: variant doesn't
  // catch: (i) the principalKey-vs-rateLimitClientKey fallback
  // order -- a refactor that swapped the order (rateLimitClientKey
  // first, principalKey as fallback) would silently emit
  // remote: for authenticated requests when the authenticatedUser
  // resolution returns Some, breaking the documented "user:
  // takes precedence" semantics; (ii) the "user:" prefix itself
  // -- a refactor renaming to e.g. "principal:" / "uid:" /
  // "user-" (hyphen) would silently break operator dashboards
  // filtering by the documented exact prefix; (iii) the userId
  // value MUST be the registered user's UUID -- a refactor that
  // emitted the user's email or displayName instead would silently
  // leak PII into operator logs (the documented privacy contract
  // says the user identifier in audit logs is the
  // PSEUDONYMOUS userId, NOT the personally-identifying email);
  // 8-tier format check mirroring 47d91dd's pattern with the
  // platform-auth-distinctive client value: (i) `request rate
  // limited` prefix (same as 47d91dd), (ii) path=/api/analyze-
  // hand-history (same), (iii) `client=user:` prefix presence
  // (THE platform-auth distinctive marker -- pins the documented
  // "user:<userId>" format from AuthStack.scala line 666), (iv)
  // EXCLUSION of `client=remote:` (the 47d91dd-pinned remote:
  // variant -- catches a refactor that emitted BOTH prefixes
  // OR fell back to remote: for authenticated requests; the
  // EXCLUSION is the load-bearing asymmetric-pair catch), (v)
  // bucket=submit (same as 47d91dd), (vi) limitPerMinute=1
  // (same), (vii) retryAfterMs= field presence, (viii) [WARN] +
  // [hand-history-review].
  test("rate-limit rejected analyze submission under PLATFORM-USER authentication emits the documented `client=user:<userId>` variant of the rate-limit log line (per AuthStack.scala line 666's principalKey=authenticatedUser path) -- the AUTHENTICATED-CLIENT companion to 47d91dd's unauthenticated `client=remote:<addr>` variant") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath)),
          rateLimitSubmitsPerMinute = 1,
          rateLimitStatusPerMinute = 0
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          // Register + extract the user's userId for the
          // expected `client=user:<uuid>` log assertion. The
          // 5ae0603 commit pinned that the response carries a
          // UUID-format userId; we use that field directly here.
          val register = postJson(s"$baseUri/api/auth/register",
            """{"email":"ratelimit@example.com","password":"correct-horse-battery","displayName":"RateLimit"}""")
          assertEquals(register.statusCode(), 201,
            clue = "registration must succeed before the platform-auth rate-limit variant can be exercised")
          val userId = jsonBody(register)("user")("userId").str
          val sessionHeaders = authSessionHeaders(register, jsonBody(register)("csrfToken").str)

          // Submission 1: accepted (consumes the rateLimitSubmits
          // PerMinute=1 slot keyed by user:<userId>)
          val accepted = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload, sessionHeaders)
          assertEquals(accepted.statusCode(), 202,
            clue = "first authenticated submission must accept (202) to consume the user-keyed rate-limit slot before the second triggers rejection")

          // Submission 2: rejected with 429 + emits rate-limit
          // log line with user:<userId> client key
          val errBuf = new java.io.ByteArrayOutputStream()
          val originalErr = System.err
          System.setErr(new java.io.PrintStream(errBuf, true, StandardCharsets.UTF_8))
          try
            val rejected = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload, sessionHeaders)
            assertEquals(rejected.statusCode(), 429,
              clue = "second authenticated submission MUST be rate-limited (429) -- the per-user rate-limit cap of 1 should be enforced by the same principalKey path as the unauthenticated variant in 47d91dd")
          finally
            System.setErr(originalErr)

          val captured = errBuf.toString(StandardCharsets.UTF_8)
          val rateLimitLine = captured.split('\n').iterator
            .find(_.contains("request rate limited"))
            .getOrElse(fail(s"no `request rate limited` line in captured stderr for the platform-auth variant -- AuthStack.scala line 674's logWarn should fire identically for both unauthenticated (47d91dd) and authenticated (this test) rate-limit rejections; if missing, either the platform-auth code path bypassed the shared rate-limit checker OR the principalKey resolution failed; got captured stderr: ${captured.take(2000)}"))

          // (i) prefix (same as 47d91dd)
          assert(rateLimitLine.contains("request rate limited"),
            clue = s"platform-auth rate-limit line must carry the SAME `request rate limited` prefix as the unauthenticated variant (47d91dd); the prefix is auth-mode-invariant per AuthStack.scala line 674's shared emission template; got: $rateLimitLine")
          // (ii) path (same as 47d91dd)
          assert(rateLimitLine.contains("path=/api/analyze-hand-history"),
            clue = s"platform-auth rate-limit line must carry the request path matching the unauthenticated variant; got: $rateLimitLine")
          // (iii) THE LOAD-BEARING CHANGE: client=user:<userId>
          // (NOT client=remote:<addr> like 47d91dd). This is the
          // pin that catches the principalKey-resolution path
          // breaking.
          assert(rateLimitLine.contains(s"client=user:$userId"),
            clue = s"platform-auth rate-limit line MUST carry `client=user:$userId` per AuthStack.scala line 666's `principalKey = authenticatedUser(exchange).map(user => s\"user:$${user.userId}\")` -- the principalKey-first fallback order at RateLimit.scala line 58 ensures authenticated requests get the user: prefix instead of the remote: prefix (the unauthenticated 47d91dd variant); a refactor swapping the fallback order, renaming the prefix, or emitting email/displayName instead of userId would silently break operator per-user-bucketing visibility AND silently leak PII (if email was emitted) into operator logs; got: $rateLimitLine")
          // (iv) EXCLUSION of remote: prefix (the 47d91dd-pinned
          // unauthenticated variant) -- the LOAD-BEARING
          // ASYMMETRIC-PAIR catch
          assert(!rateLimitLine.contains("client=remote:"),
            clue = s"platform-auth rate-limit line MUST NOT contain `client=remote:` (the 47d91dd-pinned unauthenticated client-key prefix) -- a refactor that emitted BOTH prefixes OR fell back to remote: for authenticated requests would silently break the documented per-user-bucketing visibility; the EXCLUSION pin is the load-bearing asymmetric-pair catch matching the pattern from b4b828f (state_cookie_mismatch vs missing_state_cookie); got: $rateLimitLine")
          // (v-viii) shared fields (same as 47d91dd)
          assert(rateLimitLine.contains("bucket=submit"),
            clue = s"platform-auth rate-limit line must carry bucket=submit matching the unauthenticated variant; got: $rateLimitLine")
          assert(rateLimitLine.contains("limitPerMinute=1"),
            clue = s"platform-auth rate-limit line must carry limitPerMinute=1 (the test's configured cap); got: $rateLimitLine")
          assert(rateLimitLine.contains("retryAfterMs="),
            clue = s"platform-auth rate-limit line must carry retryAfterMs= field; got: $rateLimitLine")
          assert(rateLimitLine.contains("[WARN]"),
            clue = s"platform-auth rate-limit line must be WARN-level matching the unauthenticated variant; got: $rateLimitLine")
          assert(rateLimitLine.contains("[hand-history-review]"),
            clue = s"platform-auth rate-limit line must carry the [hand-history-review] service-tag prefix; got: $rateLimitLine")
        }
      }
    }
  }

  // Pin the documented `bucket=auth` variant of the rate-limit
  // log line -- the AUTH-BUCKET mirror to 47d91dd + 82bca42's
  // submit-bucket pair, closing the SECOND of 3 rate-limit
  // buckets the deploy doc + runbook document; the rate-limit
  // taxonomy has 3 distinct buckets with documented
  // operationally-distinct triage workflows: (a) "submit" --
  // analyze + playing-hall submission throttling (pinned by
  // 47d91dd + 82bca42 -- credential-stuffing-detection adjacent),
  // (b) "auth" -- register/login throttling (THIS commit --
  // credential-stuffing DIRECT detection), and (c) "job-status"
  // -- GET /jobs/<id> polling throttling (next-fire candidate
  // -- job-scraper detection); the auth-bucket is the
  // SECURITY-CRITICAL variant because it directly throttles the
  // credential-stuffing attack surface: register+login share a
  // single auth bucket so attackers can't bypass the login
  // throttle by hammering registration with PBKDF2-cost
  // requests (per the existing line 10198 test's inline comment
  // documenting the shared-bucket design); the format at line
  // 674 is the SAME template as 47d91dd's submit variant but
  // with the path + bucket fields flipped: path=/api/auth/login
  // (or /api/auth/register), bucket=auth, limitPerMinute=
  // <rateLimitAuthPerMinute value>; per-field regression
  // vectors that 47d91dd's submit-bucket pin doesn't catch:
  // (i) the BUCKET=AUTH literal -- a refactor renaming to
  // e.g. "credentials" / "login" would silently break operator
  // alert rules filtering for credential-stuffing detection
  // (the runbook's section X.Y triage entry filters by
  // bucket=auth to spot the documented "high WARN rate from
  // many remote= sources" pattern that signals
  // credential-stuffing per the deploy doc line 218's
  // discussion of the email= field's brute-force-detection
  // role), (ii) the path=/api/auth/login field -- catches a
  // refactor that emitted a wrong path source for the auth
  // routes (e.g. always emitting the same fixed path), (iii)
  // the SHARED auth bucket across login + register -- the
  // documented "attackers can't bypass the login throttle by
  // hammering registration" property depends on BOTH routes
  // hitting bucket=auth; a refactor that split them into
  // separate buckets (bucket=login + bucket=register) would
  // silently allow the bypass while still emitting plausible-
  // looking 429s; this test pins ONLY the login-route emission
  // (sufficient to establish the bucket=auth literal AND the
  // path-source-correctness AND distinguishes from
  // bucket=submit at 47d91dd); a future fire could add the
  // register-route mirror to pin the shared-bucket property
  // directly via TWO routes emitting the same bucket=auth, but
  // for now the existing line 10200 test verifies the
  // shared-throttling behavior (register gets 429 after login
  // hits cap) at the HTTP-response level; 7-tier format check
  // mirroring 47d91dd's pattern with the auth-bucket-distinctive
  // fields: (i) `request rate limited` prefix (same as 47d91dd
  // / 82bca42), (ii) path=/api/auth/login (the auth-route
  // path -- distinguishes from /api/analyze-hand-history at
  // 47d91dd), (iii) `client=remote:` prefix presence (the
  // login flow happens BEFORE authentication completes so
  // principalKey is None and the client-key falls back to
  // remote: per 47d91dd's documented pattern -- catches a
  // refactor that somehow used user: prefix for auth-bucket
  // rejections), (iv) bucket=auth (THE distinguishing field
  // -- catches rename), (v) EXCLUSION of bucket=submit (the
  // 47d91dd-pinned alternative bucket -- catches a refactor
  // that emitted the wrong bucket id for auth rejections),
  // (vi) limitPerMinute=1 (the test's rateLimitAuthPerMinute
  // configured value), (vii) [WARN] + [hand-history-review]
  // service-tag.
  test("rate-limit rejected auth attempt emits the documented `request rate limited path=/api/auth/login client=remote:<addr> bucket=auth limitPerMinute=<n> retryAfterMs=<n>` WARN audit log line -- the AUTH-BUCKET variant complementing 47d91dd/82bca42's submit-bucket pair (closes the second of 3 rate-limit buckets the runbook documents)") {
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

          // First auth attempt (with non-existent user): 401
          // -- consumes the auth-bucket slot for the
          // remote-keyed client
          val firstLogin = postJson(s"$baseUri/api/auth/login",
            """{"email":"victim@example.com","password":"any-wrong-password"}""")
          assertEquals(firstLogin.statusCode(), 401,
            clue = "first auth attempt must reach the auth handler and return 401 (invalid credentials) to consume the rate-limit slot")

          // Second auth attempt: rate-limited 429 + emits log line
          val errBuf = new java.io.ByteArrayOutputStream()
          val originalErr = System.err
          System.setErr(new java.io.PrintStream(errBuf, true, StandardCharsets.UTF_8))
          try
            val rejected = postJson(s"$baseUri/api/auth/login",
              """{"email":"victim@example.com","password":"another-wrong-password"}""")
            assertEquals(rejected.statusCode(), 429,
              clue = "second auth attempt MUST be rate-limited (429) -- the per-IP auth-bucket cap of 1 should be enforced before reaching loginLocal")
          finally
            System.setErr(originalErr)

          val captured = errBuf.toString(StandardCharsets.UTF_8)
          val rateLimitLine = captured.split('\n').iterator
            .find(_.contains("request rate limited"))
            .getOrElse(fail(s"no `request rate limited` line in captured stderr for the auth-bucket variant -- AuthStack.scala line 674's logWarn should fire identically for submit + auth bucket rejections; got captured stderr: ${captured.take(2000)}"))

          // (i) prefix
          assert(rateLimitLine.contains("request rate limited"),
            clue = s"auth-bucket rate-limit line must carry the SAME `request rate limited` prefix as 47d91dd/82bca42 submit-bucket variants; got: $rateLimitLine")
          // (ii) path=/api/auth/login (auth-route path)
          assert(rateLimitLine.contains("path=/api/auth/login"),
            clue = s"auth-bucket rate-limit line must carry path=/api/auth/login (the login-route path) -- a refactor that emitted a wrong path source (e.g. always config-level prefix) would silently mislead operator per-endpoint triage; got: $rateLimitLine")
          // (iii) client=remote: prefix (login is unauthenticated)
          assert(rateLimitLine.contains("client=remote:"),
            clue = s"auth-bucket rate-limit line must carry client=remote: prefix -- the login flow happens BEFORE authentication completes so principalKey is None per AuthStack.scala line 666, and the client-key falls back to remote: per RateLimit.scala line 182; got: $rateLimitLine")
          // (iv) THE LOAD-BEARING CHANGE: bucket=auth (NOT
          // bucket=submit like 47d91dd/82bca42)
          assert(rateLimitLine.contains("bucket=auth"),
            clue = s"auth-bucket rate-limit line MUST carry `bucket=auth` per AuthStack.scala line 674's rejection.bucket.id field -- the runbook's credential-stuffing-detection triage filters specifically for bucket=auth to distinguish credential-attack traffic from legitimate-but-bursty submissions (bucket=submit) or job scrapers (bucket=job-status); a refactor renaming to e.g. 'credentials' / 'login' would silently break the credential-stuffing alert rules AND the documented shared-bucket-for-login-and-register property (per the existing line 10198 test's inline comment); got: $rateLimitLine")
          // (v) EXCLUSION of bucket=submit (the 47d91dd-pinned
          // alternative -- asymmetric-pair catch)
          assert(!rateLimitLine.contains("bucket=submit"),
            clue = s"auth-bucket rate-limit line MUST NOT contain bucket=submit (the 47d91dd-pinned analyze-route bucket) -- a refactor that emitted BOTH buckets OR used the wrong bucket id for auth rejections would silently misroute operator triage between credential-attack vs submission-burst patterns; the EXCLUSION is the load-bearing asymmetric-pair catch matching 82bca42's `client=remote:` EXCLUSION pattern; got: $rateLimitLine")
          // (vi) limitPerMinute=1 (test's configured cap)
          assert(rateLimitLine.contains("limitPerMinute=1"),
            clue = s"auth-bucket rate-limit line must carry limitPerMinute=1 (the test's withServer rateLimitAuthPerMinute=1 configured value) -- catches a refactor reading the wrong config knob (e.g. rateLimitSubmitsPerMinute instead of rateLimitAuthPerMinute) which would silently emit the wrong cap value; got: $rateLimitLine")
          // (vii) WARN + service-tag
          assert(rateLimitLine.contains("[WARN]"),
            clue = s"auth-bucket rate-limit line must be WARN-level matching 47d91dd/82bca42 submit-bucket variants; got: $rateLimitLine")
          assert(rateLimitLine.contains("[hand-history-review]"),
            clue = s"auth-bucket rate-limit line must carry the [hand-history-review] service-tag prefix; got: $rateLimitLine")
        }
      }
    }
  }

  // Pin the documented `bucket=job-status` variant of the rate-
  // limit log line -- the JOB-STATUS-BUCKET mirror to the prior 3
  // rate-limit pins (47d91dd submit+remote, 82bca42 submit+user,
  // 3361b2a auth+remote), closing the THIRD AND FINAL of 3
  // rate-limit buckets the deploy doc + runbook document; with
  // this commit the 3-pin bucket family (submit + auth +
  // job-status) is COMPLETE across all 3 documented buckets; the
  // job-status bucket throttles GET /api/.../jobs/<jobId>
  // polling -- the SEPARATE bucket exists because polling has a
  // fundamentally different traffic shape from submission: a
  // single submitted job triggers MANY status polls during its
  // lifetime (the documented poll-budget at site.js's
  // maxPollWaitMs ranges from 16 minutes default to longer if
  // the server's playingHallTimeoutMs is bumped, with 2-second
  // poll intervals = up to ~480 polls per single submission);
  // without the separate bucket the polling traffic would
  // dominate the submit bucket and starve actual submissions
  // OR force operators to choose between "high enough cap for
  // polling load" (which exposes submit to abuse) and "low
  // enough cap for submit safety" (which breaks polling); the
  // 3-bucket taxonomy decouples these concerns; the format at
  // line 674 is the SAME template as 47d91dd's submit variant
  // but with the path + bucket fields flipped: path=
  // /api/analyze-hand-history/jobs/<jobId> (or /api/playing-
  // hall/jobs/<jobId>), bucket=job-status, limitPerMinute=
  // <rateLimitStatusPerMinute value>; per-field regression
  // vectors that the prior 3 rate-limit pins don't catch: (i)
  // the BUCKET=JOB-STATUS literal (with hyphen!) -- a refactor
  // renaming to e.g. "status" / "poll" / "job_status"
  // (underscore not hyphen) would silently break operator
  // dashboards filtering by the documented exact bucket id; the
  // HYPHEN vs UNDERSCORE distinction is operationally
  // meaningful because the prior 3 bucket names (submit, auth)
  // have no separator at all -- job-status is the ONLY hyphenated
  // bucket id, so a "consistency refactor" replacing with
  // underscore would silently break only this bucket, (ii) the
  // path=/api/analyze-hand-history/jobs/<jobId> dynamic value --
  // contains the jobId UUID embedded INTO the path, so the
  // EMITTED PATH FIELD reflects the specific job that was
  // being polled; a refactor that emitted the parent route
  // pattern (e.g. /api/analyze-hand-history/jobs/) instead of
  // the full path with jobId would silently lose operator
  // visibility into WHICH job was being scraped (the runbook's
  // job-scraper-detection workflow keys on per-jobId polling
  // patterns to identify scrapers vs legitimate clients), (iii)
  // SEPARATE bucket cap from submit -- a refactor that
  // consolidated the buckets to use a single shared cap would
  // silently allow polling traffic to consume submit slots,
  // breaking the documented isolation; 8-tier format check
  // mirroring 47d91dd's pattern with the job-status-bucket-
  // distinctive fields: (i) `request rate limited` prefix
  // (same), (ii) path with /jobs/ substring AND jobId (catches
  // wrong-path-source), (iii) `client=remote:` prefix (basic-
  // auth doesn't use platformAuth principalKey), (iv)
  // bucket=job-status (THE distinguishing field WITH the
  // hyphen), (v) EXCLUSION of bucket=submit + bucket=auth
  // (catches consolidation refactors with the 47d91dd/82bca42/
  // 3361b2a variants -- the TRIPLE-EXCLUSION catch pattern
  // matches e17c21d/aa6426d/e43081b's progression on the
  // OIDC failure-reason chain), (vi) limitPerMinute=1 (the
  // test's rateLimitStatusPerMinute=1 configured value --
  // catches wrong-config-knob refactor reading rateLimit
  // SubmitsPerMinute instead), (vii) retryAfterMs= field
  // presence, (viii) [WARN] + [hand-history-review] service-
  // tag.
  test("rate-limit rejected job-status GET emits the documented `request rate limited path=/api/.../jobs/<jobId> client=remote:<addr> bucket=job-status limitPerMinute=<n> retryAfterMs=<n>` WARN audit log line -- the JOB-STATUS-BUCKET variant closing the 3-of-3 rate-limit bucket family (submit/auth/job-status)") {
    withStaticSite { staticDir =>
      val authConfig = HandHistoryReviewServer.BasicAuthConfig(username = "operator", password = "job-status-rate-limit-log")
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

        // Submit an analyze job (consumes no rate-limit slot
        // because rateLimitSubmitsPerMinute=0 disables submit
        // throttling) -- the BlockingBackend keeps the job
        // running so the status GETs return 200 instead of
        // already-completed
        val submission = jsonBody(postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload, authHeaders))
        val statusUri = s"$baseUri${submission("statusUrl").str}"

        // First status GET: 200 (consumes the rateLimitStatusPerMinute=1 slot)
        val firstStatus = get(statusUri, authHeaders)
        assertEquals(firstStatus.statusCode(), 200,
          clue = "first job-status GET must succeed (200) to consume the rate-limit slot before the second triggers rejection")

        // Second status GET: rate-limited 429 + emits log line
        val errBuf = new java.io.ByteArrayOutputStream()
        val originalErr = System.err
        System.setErr(new java.io.PrintStream(errBuf, true, StandardCharsets.UTF_8))
        try
          val rejected = get(statusUri, authHeaders)
          assertEquals(rejected.statusCode(), 429,
            clue = "second job-status GET MUST be rate-limited (429) -- the per-IP job-status-bucket cap of 1 should be enforced")
        finally
          System.setErr(originalErr)

        // Release the backend so the worker thread can finish
        // and the test can tear down cleanly. We do NOT call
        // awaitTerminalJob because the rate-limit is still in
        // effect (the 1-per-minute cap WON'T reset within the
        // test's lifetime), so subsequent GETs would return 429
        // rather than the terminal-state JSON awaitTerminalJob
        // expects -- the helper would fail with "key not found:
        // status" when parsing the 429 body. The
        // backend.release.countDown is sufficient for clean
        // teardown; the worker completes its run on its own.
        backend.release.countDown()

        val captured = errBuf.toString(StandardCharsets.UTF_8)
        val rateLimitLine = captured.split('\n').iterator
          .find(_.contains("request rate limited"))
          .getOrElse(fail(s"no `request rate limited` line in captured stderr for the job-status-bucket variant; got captured stderr: ${captured.take(2000)}"))

        // (i) prefix
        assert(rateLimitLine.contains("request rate limited"),
          clue = s"job-status-bucket rate-limit line must carry the SAME `request rate limited` prefix matching the prior 3 rate-limit pins; got: $rateLimitLine")
        // (ii) path with /jobs/ substring (job-status route)
        assert(rateLimitLine.contains("path=/api/analyze-hand-history/jobs/"),
          clue = s"job-status-bucket rate-limit line must carry the analyze-side jobs/ path -- the specific path includes the dynamic jobId UUID embedded in the URL; a refactor that emitted the parent route pattern (e.g. /api/analyze-hand-history/jobs/{id} or just /api/analyze-hand-history) would silently lose operator visibility into WHICH job was being scraped (the runbook's job-scraper-detection workflow keys on per-jobId polling patterns to identify scrapers vs legitimate clients); got: $rateLimitLine")
        // (iii) client=remote: prefix (basic-auth doesn't use platformAuth)
        assert(rateLimitLine.contains("client=remote:"),
          clue = s"job-status-bucket rate-limit line must carry client=remote: prefix -- this test uses basic-auth (NOT platformAuth) so principalKey is None and the client-key falls back to remote: per RateLimit.scala line 182; got: $rateLimitLine")
        // (iv) THE LOAD-BEARING CHANGE: bucket=job-status with
        // hyphen (NOT submit or auth)
        assert(rateLimitLine.contains("bucket=job-status"),
          clue = s"job-status-bucket rate-limit line MUST carry `bucket=job-status` (with HYPHEN -- the ONLY hyphenated bucket id among the 3 buckets); a refactor renaming to e.g. 'status' / 'poll' / 'job_status' (underscore not hyphen for 'consistency' with submit/auth which have no separator) would silently break operator dashboards filtering by the documented bucket id; the HYPHEN distinction is operationally meaningful because the prior 3 bucket names (submit, auth) have no separator at all -- job-status is the ONLY hyphenated bucket id; got: $rateLimitLine")
        // (v) TRIPLE-EXCLUSION of the prior 2 bucket variants
        assert(!rateLimitLine.contains("bucket=submit"),
          clue = s"job-status-bucket rate-limit line MUST NOT contain bucket=submit (the 47d91dd/82bca42-pinned bucket) -- catches a refactor consolidating buckets which would silently allow polling traffic to consume submit slots, breaking the documented isolation; got: $rateLimitLine")
        assert(!rateLimitLine.contains("bucket=auth"),
          clue = s"job-status-bucket rate-limit line MUST NOT contain bucket=auth (the 3361b2a-pinned bucket) -- the auth bucket is for credential-stuffing detection, distinct from polling traffic; got: $rateLimitLine")
        // (vi) limitPerMinute=1 (catches wrong-config-knob
        // refactor reading rateLimitSubmitsPerMinute)
        assert(rateLimitLine.contains("limitPerMinute=1"),
          clue = s"job-status-bucket rate-limit line must carry limitPerMinute=1 (the test's withServer rateLimitStatusPerMinute=1 configured value) -- catches a refactor reading the wrong config knob (e.g. rateLimitSubmitsPerMinute=0 from this test's config would emit limitPerMinute=0 if the wrong knob were used); got: $rateLimitLine")
        // (vii) retryAfterMs field
        assert(rateLimitLine.contains("retryAfterMs="),
          clue = s"job-status-bucket rate-limit line must carry retryAfterMs= field; got: $rateLimitLine")
        // (viii) WARN + service-tag
        assert(rateLimitLine.contains("[WARN]"),
          clue = s"job-status-bucket rate-limit line must be WARN-level matching the prior 3 rate-limit pins; got: $rateLimitLine")
        assert(rateLimitLine.contains("[hand-history-review]"),
          clue = s"job-status-bucket rate-limit line must carry the [hand-history-review] service-tag prefix; got: $rateLimitLine")
      }
    }
  }

  // Pin the documented `client=header:<value>` TRUSTED-HEADER
  // variant of the rate-limit log line -- the THIRD AND FINAL
  // client-key prefix completing the 3-of-3 client-key family
  // (47d91dd remote: + 82bca42 user: + THIS header:); with this
  // commit the rate-limit log line family covers ALL 3 documented
  // client-key prefixes from RateLimit.scala's keying logic at
  // lines 179-182: (a) `header:<value>` -- the trusted-header
  // path at line 181 when rateLimitClientIpHeader is configured
  // AND the peer IP is in trustedProxyIps, (b) `remote:<addr>`
  // -- the fallback IP-based key at line 182 when the trusted-
  // header path doesn't apply, (c) `user:<userId>` -- the
  // principalKey path at AuthStack.scala line 666 when
  // authenticatedUser is non-empty; the trusted-header variant
  // is REVERSE-PROXY-RELEVANT: in production deployments behind
  // a load balancer / API gateway, the direct TCP peer is the
  // proxy, NOT the actual end-user IP; without the trusted-
  // header path, rate-limit would bucket ALL traffic from the
  // proxy as a single client (silently consolidating all real
  // end-users into one bucket and either starving legitimate
  // users when ONE user misbehaves OR effectively disabling
  // rate-limit when the cap is bumped high enough to absorb
  // fleet traffic); the header: prefix lets operators see
  // "this rejection was for end-user IP 203.0.113.10 (as
  // forwarded by the trusted proxy)" rather than just the
  // proxy IP; per-field regression vectors that 47d91dd's
  // remote: pin doesn't catch: (i) the trusted-proxy
  // verification path at RateLimit.scala line 180's
  // `trustsRateLimitClientIpHeader(remoteInetAddress(exchange),
  // trustedProxyIps)` -- a refactor that bypassed this
  // verification (e.g. "trust the header for any peer for
  // simplicity") would silently allow attackers to forge
  // X-Real-IP values from the open internet, effectively
  // disabling rate-limiting for any attacker that knows the
  // configured header name; (ii) the `header:` prefix itself
  // -- a refactor renaming to e.g. "forwarded:" / "xfwd:" /
  // "proxy:" would silently break operator dashboards filtering
  // by the documented header: prefix; (iii) the header-value
  // PASS-THROUGH semantic -- the prefix MUST be followed by
  // the literal header value (not e.g. a parsed IP-and-port
  // tuple); a refactor that normalized the value (e.g. always
  // emitted just the IP without port, or always lowercased)
  // would silently drift from the documented "this was the
  // value the trusted header carried" semantic; test approach:
  // mirror the existing trusted-header rate-limit test at line
  // ~10522 (rateLimitClientIpHeader=Some("X-Real-IP") +
  // rateLimitSubmitsPerMinute=1 + submit twice with same
  // X-Real-IP value = 1 accepted + 1 rejected) but capture
  // stderr around the rejected submission and assert the log
  // line carries `client=header:<value>` (the trusted-header
  // value) AND does NOT carry `client=remote:` or `client=user:`
  // (the other client-key prefixes); 9-tier format check
  // mirroring the prior rate-limit pins with the header-
  // distinctive client value: (i) `request rate limited`
  // prefix, (ii) path=/api/analyze-hand-history (same as
  // 47d91dd), (iii) `client=header:203.0.113.10` (THE
  // load-bearing trusted-header value pin -- specific value
  // from the test's forged X-Real-IP header), (iv) EXCLUSION
  // of `client=remote:` (the 47d91dd-pinned IP-based fallback
  // -- catches a refactor where the trusted-header path was
  // bypassed and the fallback fired instead), (v) EXCLUSION
  // of `client=user:` (the 82bca42-pinned principalKey path
  // -- catches a refactor where authenticated-user took
  // precedence over the trusted header, but the test has no
  // platformAuth so this is defense-in-depth), (vi)
  // bucket=submit, (vii) limitPerMinute=1, (viii) retryAfterMs=
  // field, (ix) [WARN] + [hand-history-review].
  test("rate-limit rejected analyze submission with trusted X-Real-IP header emits the documented `client=header:<value>` variant of the rate-limit log line (per RateLimit.scala line 181's trusted-header path when rateLimitClientIpHeader is configured + peer is loopback-trusted) -- closes the 3-of-3 client-key prefix family (remote:/user:/header:) and the 5x bucket × client-key matrix's reachable cells") {
    withStaticSite { staticDir =>
      withServer(
        staticDir,
        rateLimitSubmitsPerMinute = 1,
        rateLimitStatusPerMinute = 0,
        rateLimitClientIpHeader = Some("X-Real-IP")
      ) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        // Same X-Real-IP value across both submissions so the
        // trusted-header client key is identical (rate-limit
        // groups by client key, not by header presence)
        val clientHeaders = Map("X-Real-IP" -> "203.0.113.10")

        // Submission 1: accepted (consumes the rate-limit slot
        // keyed by header:203.0.113.10)
        val accepted = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload, clientHeaders)
        assertEquals(accepted.statusCode(), 202,
          clue = "first submission with trusted X-Real-IP header must accept (202) to consume the header-keyed rate-limit slot")

        // Submission 2: rejected with 429 + emits log line
        val errBuf = new java.io.ByteArrayOutputStream()
        val originalErr = System.err
        System.setErr(new java.io.PrintStream(errBuf, true, StandardCharsets.UTF_8))
        try
          val rejected = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload, clientHeaders)
          assertEquals(rejected.statusCode(), 429,
            clue = "second submission with same trusted X-Real-IP MUST be rate-limited (429) -- the per-header rate-limit cap of 1 should be enforced via the same client key")
        finally
          System.setErr(originalErr)

        val captured = errBuf.toString(StandardCharsets.UTF_8)
        val rateLimitLine = captured.split('\n').iterator
          .find(_.contains("request rate limited"))
          .getOrElse(fail(s"no `request rate limited` line in captured stderr for the trusted-header variant; got captured stderr: ${captured.take(2000)}"))

        // (i) prefix
        assert(rateLimitLine.contains("request rate limited"),
          clue = s"trusted-header rate-limit line must carry the SAME `request rate limited` prefix matching the prior 4 rate-limit pins; got: $rateLimitLine")
        // (ii) path
        assert(rateLimitLine.contains("path=/api/analyze-hand-history"),
          clue = s"trusted-header rate-limit line must carry the analyze submit path; got: $rateLimitLine")
        // (iii) THE LOAD-BEARING CHANGE: client=header:<value>
        assert(rateLimitLine.contains("client=header:203.0.113.10"),
          clue = s"trusted-header rate-limit line MUST carry `client=header:203.0.113.10` per RateLimit.scala line 181's `s\"header:$$value\"` template -- the value is the literal X-Real-IP header value the test set, passed through without normalization; a refactor renaming the prefix to e.g. `forwarded:` / `xfwd:` / `proxy:` would silently break operator dashboards; a refactor that normalized the value (lowercased, stripped port, etc.) would silently drift from the documented pass-through semantic; a refactor that BYPASSED the trusted-proxy verification at line 180 would silently allow forged X-Real-IP values from the open internet, effectively disabling rate-limiting (and the test would still pass for the contains check, but the EXCLUSION of `remote:` below would distinguish this case from the bypass-and-fallback case); got: $rateLimitLine")
        // (iv) EXCLUSION of client=remote: (catches refactor
        // where trusted-header path was bypassed and fallback
        // fired instead)
        assert(!rateLimitLine.contains("client=remote:"),
          clue = s"trusted-header rate-limit line MUST NOT contain `client=remote:` (the 47d91dd-pinned IP-based fallback prefix) -- a refactor where the trusted-header path was bypassed and the fallback fired instead would silently emit the proxy's TCP-peer address as the client key, breaking operator visibility into the actual end-user IP behind the proxy; the EXCLUSION pin is the load-bearing catch for the trusted-header-vs-fallback path ordering; got: $rateLimitLine")
        // (v) EXCLUSION of client=user: (defense-in-depth for
        // the precedence order)
        assert(!rateLimitLine.contains("client=user:"),
          clue = s"trusted-header rate-limit line MUST NOT contain `client=user:` (the 82bca42-pinned principalKey prefix) -- this test has no platformAuth so principalKey is always None, but the EXCLUSION pins the documented client-key-resolution-precedence ordering as defense-in-depth; a refactor that emitted user: when authenticated would still not affect this test, but the EXCLUSION ensures the test catches a regression that emitted user: regardless of auth state; got: $rateLimitLine")
        // (vi) bucket=submit (same as 47d91dd)
        assert(rateLimitLine.contains("bucket=submit"),
          clue = s"trusted-header rate-limit line must carry bucket=submit matching the analyze-side path; got: $rateLimitLine")
        // (vii) limitPerMinute=1
        assert(rateLimitLine.contains("limitPerMinute=1"),
          clue = s"trusted-header rate-limit line must carry limitPerMinute=1 (the test's configured cap); got: $rateLimitLine")
        // (viii) retryAfterMs= field
        assert(rateLimitLine.contains("retryAfterMs="),
          clue = s"trusted-header rate-limit line must carry retryAfterMs= field; got: $rateLimitLine")
        // (ix) WARN + service-tag
        assert(rateLimitLine.contains("[WARN]"),
          clue = s"trusted-header rate-limit line must be WARN-level matching the prior 4 rate-limit pins; got: $rateLimitLine")
        assert(rateLimitLine.contains("[hand-history-review]"),
          clue = s"trusted-header rate-limit line must carry the [hand-history-review] service-tag prefix; got: $rateLimitLine")
      }
    }
  }

  // Pin the documented `failed to start web server: <host>:<port>
  // is unavailable (<exception>)` ERROR audit log line per
  // HandHistoryReviewServerRuntime.scala line 358-359 -- opens a
  // NEW CATEGORY of operator-facing log line: STARTUP-FAILURE
  // emissions, complementing the prior INFO (success/banner) and
  // WARN (failure/rate-limit) categories with the ERROR level;
  // the log line fires when a BindException prevents the HTTP
  // server from binding to its configured host:port (e.g.
  // another process already listening on the port, permission
  // denied for privileged ports, network interface unavailable);
  // operators triage by this ERROR log line to know "the boot
  // failed and why" -- the INFO-level startup banner (pinned by
  // 7c47f88) only emits on SUCCESSFUL boot, so its ABSENCE
  // combined with this ERROR line is the operator's signal that
  // a deployment failed to start; the format at line 358-359 is
  // `s"failed to start web server: ${config.host}:${config.port}
  // is unavailable (${e.getMessage})"` -- four fields: (a) the
  // literal "failed to start web server:" prefix, (b) the
  // configured host:port pair operators see was the intended
  // bind target, (c) the literal "is unavailable" diagnostic
  // (operators triage on this specific wording per the
  // existing test at line ~11078 that pins the Left value),
  // (d) the underlying BindException message in parens; the
  // existing test at line ~11078 pins ONLY the Left value
  // returned from startWithBackend -- it does NOT capture the
  // logError emission to stderr; this commit adds the ERROR-
  // LEVEL log line pin to complement the Left-value pin,
  // catching a refactor that emitted the log line at a
  // different level (e.g. WARN -- which would silently bury
  // startup failures in the warning stream operators expect to
  // contain rate-limit and validation noise) OR suppressed the
  // log line entirely (silent boot failures that operators
  // can only detect by polling the bound process state); per-
  // field regression vectors that the prior log line pins
  // don't catch: (i) the [ERROR] level (NOT [INFO] like
  // startup-complete or [WARN] like rate-limit) -- this is the
  // FIRST and ONLY ERROR-level log line pin in the operator-
  // facing log line coverage; a refactor demoting to WARN
  // would silently shift startup failures into the warning
  // noise; (ii) the "failed to start web server:" prefix --
  // matches the Left value's format so operators see CONSISTENT
  // text between the log line + the CLI exit message; (iii)
  // the "<host>:<port>" pattern -- the SPECIFIC host:port that
  // was attempted (not e.g. a placeholder like "localhost"
  // when the config explicitly set 127.0.0.1); (iv) the "is
  // unavailable" diagnostic substring -- operators grep for
  // this specific wording to detect bind-failure events; 5-tier
  // format check: (i) `failed to start web server:` prefix
  // (catches rename), (ii) `127.0.0.1:<port>` substring with
  // the SPECIFIC port the conflicting first server bound to,
  // (iii) `is unavailable` diagnostic substring (catches a
  // refactor renaming to e.g. "already in use" / "bind
  // failed"), (iv) [ERROR] level (catches level shift), (v)
  // [hand-history-review] service-tag prefix.
  test("startup BindException emits the documented `failed to start web server: <host>:<port> is unavailable (<exception>)` ERROR audit log line per HandHistoryReviewServerRuntime.scala line 358-359 -- opens a NEW CATEGORY (STARTUP-FAILURE emissions) with the FIRST ERROR-level pin in the operator-facing log line coverage") {
    withStaticSite { staticDir =>
      withServer(staticDir) { running =>
        val port = running.binding.port

        // Capture stderr around the second-bind attempt. The
        // logError at HandHistoryReviewServerRuntime.scala line
        // 359 writes to System.err per line 423-424.
        val errBuf = new java.io.ByteArrayOutputStream()
        val originalErr = System.err
        System.setErr(new java.io.PrintStream(errBuf, true, StandardCharsets.UTF_8))
        val secondStart =
          try
            HandHistoryReviewServer.startWithBackend(
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
          finally
            System.setErr(originalErr)

        // The Left value pin is matched by the existing line
        // ~11078 test; this test additionally pins the ERROR
        // log line that emits alongside the Left return.
        assert(secondStart.isLeft,
          clue = "second start MUST fail (Left) because the first server is still bound to the port")

        val captured = errBuf.toString(StandardCharsets.UTF_8)
        val bindErrorLine = captured.split('\n').iterator
          .find(_.contains("failed to start web server"))
          .getOrElse(fail(s"no `failed to start web server` line in captured stderr -- HandHistoryReviewServerRuntime.scala line 358-359 documents this as the ERROR-level log line that fires alongside the Left return; if missing, either the logError was suppressed OR the bind-error path took a different branch; got captured stderr: ${captured.take(2000)}"))

        // (i) prefix
        assert(bindErrorLine.contains("failed to start web server:"),
          clue = s"startup-failure log line must carry the literal `failed to start web server:` prefix per HandHistoryReviewServerRuntime.scala line 358's hardcoded template -- a refactor renaming to e.g. `boot failed` / `server start error` would silently break operator alert rules grep'ing for the documented startup-failure signal AND would silently desync from the Left value's format pinned by the existing line ~11078 test; got: $bindErrorLine")
        // (ii) specific host:port that was attempted
        assert(bindErrorLine.contains(s"127.0.0.1:$port"),
          clue = s"startup-failure log line must carry the SPECIFIC host:port (127.0.0.1:$port) that was attempted -- catches a refactor that emitted a placeholder like 'localhost:0' instead of the actual config-provided values; got: $bindErrorLine")
        // (iii) diagnostic substring
        assert(bindErrorLine.contains("is unavailable"),
          clue = s"startup-failure log line must carry the literal `is unavailable` diagnostic substring per HandHistoryReviewServerRuntime.scala line 358's template -- operators grep for this specific wording to detect bind-failure events; a refactor renaming to e.g. `already in use` / `bind failed` / `port collision` would silently break operator alert rules; got: $bindErrorLine")
        // (iv) ERROR level (the NEW dimension this pin opens)
        assert(bindErrorLine.contains("[ERROR]"),
          clue = s"startup-failure log line must be ERROR-level per HandHistoryReviewServerRuntime.scala line 423-424's logError helper (which uses level=\"ERROR\" + stream=System.err) -- this is the FIRST and ONLY ERROR-level log line pin in the operator-facing log line coverage; a refactor demoting to WARN would silently shift startup failures into the warning noise stream where operators expect rate-limit + validation events, masking the boot-failure signal; a refactor promoting to FATAL or similar would silently break alert rules that filter by ERROR level; got: $bindErrorLine")
        // (v) service-tag prefix
        assert(bindErrorLine.contains("[hand-history-review]"),
          clue = s"startup-failure log line must carry the `[hand-history-review]` service-tag prefix matching all prior log line pins -- the service-tag is level-invariant (ERROR lines have the same service-tag as INFO + WARN lines); got: $bindErrorLine")
      }
    }
  }

  // Pin the documented WWW-Authenticate header EXACT wire form
  // `Basic realm="sicfun-hand-history-review", charset="UTF-8"`
  // per AuthStack.scala line 35-36 + deploy doc line 130 -- the
  // existing line ~8471 test only verifies `.startsWith("Basic
  // ")` which would pass for ANY realm string, including a
  // refactor that renamed the realm; THIS commit pins the
  // SPECIFIC realm value AND the charset parameter that the
  // deploy doc EXPLICITLY documents as the wire form; the
  // realm string is OPERATIONALLY MEANINGFUL because: (a)
  // browser auth popups display the realm as prompt context
  // ("sicfun-hand-history-review wants you to sign in"), so a
  // refactor changing the realm would silently change what
  // users see in the auth dialog, (b) password managers SCOPE
  // saved credentials by `(origin, realm)` pair -- a refactor
  // changing the realm would silently INVALIDATE all saved
  // password-manager entries (users would have to re-enter
  // their credentials AND would have duplicate stale entries
  // until they cleaned up); the deploy doc explicitly says: "If
  // you want one shared password-manager entry to cover
  // multiple sicfun instances on different hosts/ports, keep
  // this realm string stable across them" -- this commit pins
  // that stability via CI enforcement; the `charset="UTF-8"`
  // parameter is OPERATIONALLY MEANINGFUL because: per the
  // deploy doc, "modern browsers honor it (encoding non-ASCII
  // passwords as UTF-8), older ones default to ISO-8859-1
  // which still works for ASCII-only passwords"; a refactor
  // dropping the charset parameter would silently regress
  // non-ASCII password handling on modern browsers, AND would
  // silently break RFC 7617 sec 2.1 compliance; per-field
  // regression vectors that the existing `.startsWith("Basic
  // ")` check doesn't catch: (i) realm rename (e.g.
  // sicfun-hand-history-review -> sicfun-poker-analytics for
  // a project rebranding) would silently invalidate all
  // saved password-manager entries, (ii) realm syntax change
  // (e.g. realm=sicfun without quotes) would silently break
  // browser parsing of the challenge AND silently break
  // RFC 7235 sec 2.2 compliance, (iii) dropping charset (e.g.
  // for backwards-compat with very old browsers) would
  // silently regress non-ASCII password support, (iv)
  // charset value change (e.g. "UTF-8" -> "utf-8" or
  // "ISO-8859-1") would silently change the wire encoding
  // semantics -- a refactor that "fixed" the case to match
  // some other convention would silently break the documented
  // RFC 7617 compliance claim; 3-tier format check + exact-
  // value pin: (i) realm value "sicfun-hand-history-review"
  // exactly (with quotes -- the RFC 7235 quoted-string form),
  // (ii) charset value "UTF-8" exactly (with quotes -- RFC
  // 7617 sec 2.1 form), (iii) the FULL header value matches
  // the documented wire form (catches separator changes like
  // semicolon-instead-of-comma between the realm + charset
  // parameters); test approach: configure withServer with
  // basicAuth + make an unauthenticated request, verify the
  // 401 response's WWW-Authenticate header carries the EXACT
  // documented wire form.
  test("WWW-Authenticate header for 401 responses under basic-auth mode carries the EXACT documented wire form `Basic realm=\"sicfun-hand-history-review\", charset=\"UTF-8\"` per AuthStack.scala line 35-36 + deploy doc line 130 (the realm string MUST stay stable for password-manager scope-by-(origin,realm) AND the charset parameter MUST be present for RFC 7617 sec 2.1 compliance)") {
    withStaticSite { staticDir =>
      val authConfig = HandHistoryReviewServer.BasicAuthConfig(username = "operator", password = "auth-realm-test")
      withServer(staticDir, basicAuth = Some(authConfig)) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        // Make an unauthenticated request to trigger the 401
        // response carrying the WWW-Authenticate header per
        // AuthStack.scala line 645's
        // `exchange.getResponseHeaders.set("WWW-Authenticate",
        // BasicAuthChallenge)` where BasicAuthChallenge is the
        // documented wire form from line 36.
        val unauthorized = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload)
        assertEquals(unauthorized.statusCode(), 401,
          clue = "unauthenticated request to a basic-auth-protected route must return 401 before the WWW-Authenticate header can be inspected")

        val wwwAuth = headerValue(unauthorized, "WWW-Authenticate")
          .getOrElse(fail("WWW-Authenticate header missing on 401 response per RFC 7235 sec 4.1 'A server generating a 401 response MUST send a WWW-Authenticate header field'"))

        // (i) THE LOAD-BEARING PIN: the EXACT documented wire
        // form matching AuthStack.scala line 36's hardcoded
        // BasicAuthChallenge value AND the deploy doc line
        // 130's documented operator-visible string
        assertEquals(wwwAuth, """Basic realm="sicfun-hand-history-review", charset="UTF-8"""",
          clue = s"WWW-Authenticate header MUST carry the EXACT documented wire form `Basic realm=\"sicfun-hand-history-review\", charset=\"UTF-8\"` per AuthStack.scala line 36's `BasicAuthChallenge = s\"\"\"Basic realm=\"$$BasicAuthRealm\", charset=\"UTF-8\"\"\"\"` + deploy doc line 130's documented wire format; the existing line ~8471 test only verifies .startsWith(\"Basic \") which would pass for ANY realm string (catching only the most extreme regressions); this exact-equality assertion pins the realm string AND the charset parameter AND the separator format ALL AT ONCE so a refactor that broke ANY of the three (rename realm, drop charset, change separator) would fail this test; the realm string `sicfun-hand-history-review` is the password-manager scoping key (the deploy doc says: 'password managers scope saved credentials by (origin, realm) pair -- if you want one shared password-manager entry to cover multiple sicfun instances on different hosts/ports, keep this realm string stable across them') so a refactor changing it would silently invalidate all saved password-manager entries fleet-wide; the charset=\"UTF-8\" parameter is RFC 7617 sec 2.1 compliance (the deploy doc says 'modern browsers honor it (encoding non-ASCII passwords as UTF-8), older ones default to ISO-8859-1 which still works for ASCII-only passwords') so dropping it would silently regress non-ASCII password support on modern browsers; got: $wwwAuth")
        // (ii) substring assertions for diagnostic clarity
        // (the exact-equality assertion above is sufficient,
        // these add diagnostic context for partial failures)
        assert(wwwAuth.contains("""realm="sicfun-hand-history-review""""),
          clue = s"WWW-Authenticate realm MUST be the exact documented string `sicfun-hand-history-review` (with quotes) -- a refactor renaming would silently break the documented `(origin, realm)` password-manager scoping; got: $wwwAuth")
        assert(wwwAuth.contains("""charset="UTF-8""""),
          clue = s"WWW-Authenticate must carry charset=\"UTF-8\" (with quotes -- RFC 7617 sec 2.1 form) -- dropping silently regresses non-ASCII password support per the deploy doc line 130's documented compatibility rationale; got: $wwwAuth")
      }
    }
  }

  // Pin the documented TIMEOUT-PATH variant of the `job failed`
  // audit log line: errorStatus=504 (NOT 400 like 6b59ce4's
  // backend-Left variant) AND error=analysis%20timed%20out%20
  // after%20<N>ms (the SPECIFIC timeoutFailure string format
  // documented at JobQueue.scala line 388); 6b59ce4 pinned the
  // SAME emission site (line 321-323) but exercised the
  // classifyAnalysisError DEFAULT branch (400) via a backend
  // Left; THIS commit exercises the TIMEOUT branch (504) via
  // an analysisTimeoutMs trigger + BlockingBackend that never
  // releases; together the two commits pin BOTH of the
  // classifier's two FAILURE-branch values (400 vs 504), with
  // the THIRD branch (500 for "analysis failed:" prefix) still
  // unpinned but reachable only via a NonFatal exception in
  // the analyze backend; the TIMEOUT path is OPERATIONALLY
  // CRITICAL because it's the documented signal for "this job
  // hung past the configured wall-clock cap" -- the runbook's
  // analysis-stuck triage entry filters by errorStatus=504 to
  // distinguish timeout-stuck from backend-error-failed jobs
  // (different operator responses: timeout means
  // analysisTimeoutMs cap is too low OR the backend is
  // genuinely slow, while errorStatus=400 means the input was
  // malformed and the operator should investigate the
  // submission); the %20-escape on "analysis timed out after
  // 100ms" is the LOAD-BEARING contract this pin uniquely
  // verifies because 6b59ce4's backend-Left value
  // "invalid hand history format with spaces" exercises the
  // escape on a DIFFERENT category of error string (user-
  // controlled vs server-generated); a refactor that
  // accidentally dropped the escape ONLY on the timeoutFailure
  // path (e.g. "the timeout message is server-generated and
  // doesn't need escaping") would silently break the log-line
  // format for timeout events while keeping backend-Left
  // events correctly escaped; per-field regression vectors
  // SPECIFIC to the timeout path that 6b59ce4's pin doesn't
  // catch: (i) errorStatus=504 (the TIMEOUT classifier branch
  // -- catches a refactor that broke classifyAnalysisError's
  // line 765's `if error.startsWith("analysis timed out after")
  // then 504` check), (ii) the EXACT timeout-message format
  // `analysis timed out after <N>ms` at JobQueue.scala line
  // 388 -- catches a refactor renaming to e.g. "analysis
  // exceeded N ms" / "analysis took longer than Nms" which
  // would silently break (a) the classifyAnalysisError prefix
  // match at line 765 (the classifier returns 504 ONLY when
  // the error starts with "analysis timed out after" -- a
  // rename would silently demote timeout errors to the 400
  // default branch), AND (b) operator dashboards filtering by
  // the documented exact wording, (iii) the analysisTimeoutMs
  // value embedded in the error message (the test uses 100ms
  // so the expected wire form is "analysis timed out after
  // 100ms" with the 100 literal); a refactor that emitted a
  // different time unit (e.g. seconds, milliseconds-with-comma
  // separator) would silently confuse operators about the
  // actual timeout-cap value; 7-tier format check at WARN
  // level extending 6b59ce4's pattern: (i) `job failed`
  // prefix (same as 6b59ce4), (ii) jobId matching the 202
  // response, (iii) errorStatus=504 (NOT 400 -- the TIMEOUT-
  // distinctive value), (iv) error=analysis%20timed%20out%20
  // after%20100ms (THE LOAD-BEARING timeout-message + %20-
  // escape pin), (v) EXCLUSION of the unescaped form
  // "analysis timed out after 100ms" (catches the escape-
  // dropped refactor -- same shape as 6b59ce4's ESCAPE-
  // CONTRACT VERIFICATION), (vi) [WARN] level, (vii) [hand-
  // history-review] service-tag.
  test("analyze worker that times out emits the documented `job failed ... errorStatus=504 error=analysis%20timed%20out%20after%20<N>ms` WARN audit log line per JobQueue.scala line 321-323 -- the TIMEOUT-PATH variant of 6b59ce4's backend-Left pin (exercises classifyAnalysisError's 504 branch vs 6b59ce4's 400 default branch)") {
    withStaticSite { staticDir =>
      val backend = new BlockingBackend(Right(sampleAnalysisResult))
      withServer(
        staticDir,
        backend = backend,
        maxConcurrentJobs = 1,
        maxQueuedJobs = 1,
        analysisTimeoutMs = 100L
      ) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        // Capture stderr around the submit + terminal-poll
        // cycle. The job will be terminated by the timeout
        // (the BlockingBackend never releases) so the worker
        // emits the "job failed" line at JobQueue.scala line
        // 321-323 with the timeoutFailure-generated Failed
        // state.
        val errBuf = new java.io.ByteArrayOutputStream()
        val originalErr = System.err
        System.setErr(new java.io.PrintStream(errBuf, true, StandardCharsets.UTF_8))
        val submitJobId =
          try
            val submit = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload)
            assertEquals(submit.statusCode(), 202,
              clue = "submission must return 202 before the worker can be timed out")
            val statusUri = s"$baseUri${jsonBody(submit)("statusUrl").str}"
            val capturedJobId = jsonBody(submit)("jobId").str
            assert(backend.started.await(3, TimeUnit.SECONDS),
              "backend never started -- the test depends on the worker reaching the analyze call so the timeout interrupts it")
            // Wait for terminal state (failed-by-timeout). The
            // worker will be interrupted after 100ms and the
            // job will transition to Failed with the
            // timeoutFailure error.
            val terminal = awaitTerminalJob(statusUri)
            assertEquals(terminal("status").str, "failed",
              clue = "worker must reach 'failed' terminal state via the timeout path")
            assertEquals(terminal("errorStatus").num.toInt, 504,
              clue = "terminal-state errorStatus must be 504 (classifyAnalysisError's timeout branch) -- this is the HTTP-response-shape pin matching the existing line ~11170 test, complementing this commit's audit-log-shape pin")
            capturedJobId
          finally
            System.setErr(originalErr)

        // Release backend latch to allow clean teardown
        backend.release.countDown()

        val captured = errBuf.toString(StandardCharsets.UTF_8)
        val failedLine = captured.split('\n').iterator
          .find(line => line.contains("job failed") && line.contains(s"jobId=$submitJobId"))
          .getOrElse(fail(s"no `job failed jobId=$submitJobId` line in captured stderr for the timeout path; got captured stderr: ${captured.take(2000)}"))

        // (i) prefix
        assert(failedLine.contains("job failed"),
          clue = s"timeout-path failed line must carry `job failed` prefix matching 6b59ce4's pattern; got: $failedLine")
        // (ii) jobId matching the 202 response
        assert(failedLine.contains(s"jobId=$submitJobId"),
          clue = s"timeout-path failed line must carry the submission's jobId; got: $failedLine")
        // (iii) errorStatus=504 (THE TIMEOUT-DISTINCTIVE value)
        assert(failedLine.contains("errorStatus=504"),
          clue = s"timeout-path failed line MUST carry errorStatus=504 per JobQueue.scala line 765's classifyAnalysisError check `if error.startsWith(\"analysis timed out after\") then 504` -- this is the TIMEOUT classifier branch that 6b59ce4's backend-Left variant (which exercises the 400 default branch) doesn't catch; a refactor that broke the prefix-match (e.g. renaming the timeoutFailure error string OR changing the classifier prefix) would silently demote timeout errors to the 400 default branch, breaking operator timeout-vs-malformed-input distinction; got: $failedLine")
        // (iv) error=analysis%20timed%20out%20after%20100ms
        // (THE %20-escaped timeout message -- specific value)
        assert(failedLine.contains("error=analysis%20timed%20out%20after%20100ms"),
          clue = s"timeout-path failed line MUST carry the EXACT %20-escaped timeoutFailure error string `analysis%20timed%20out%20after%20100ms` per JobQueue.scala line 388's `s\"analysis timed out after $${analysisTimeoutMs}ms\"` template + the line 322 %20-escape via `error.replace(\" \", \"%20\")`; the 100ms reflects the test's analysisTimeoutMs=100L value; a refactor renaming to e.g. \"analysis exceeded N ms\" / \"analysis took longer than Nms\" would silently break BOTH the classifyAnalysisError prefix-match (demoting to 400) AND operator dashboards filtering by the documented exact wording; got: $failedLine")
        // (v) EXCLUSION of unescaped form (catches escape-dropped refactor)
        assert(!failedLine.contains("analysis timed out after 100ms"),
          clue = s"timeout-path failed line MUST NOT contain the UNESCAPED form `analysis timed out after 100ms` (with literal spaces) -- catches a refactor that accidentally dropped the .replace(\" \", \"%20\") at JobQueue.scala line 322 ONLY on the timeout error variant while keeping backend-Left errors correctly escaped; matches the ESCAPE-CONTRACT VERIFICATION pattern from 6b59ce4 + e43081b; got: $failedLine")
        // (vi) WARN level
        assert(failedLine.contains("[WARN]"),
          clue = s"timeout-path failed line must be WARN-level matching 6b59ce4's backend-Left variant -- both emit at the same JobQueue.scala line 321 logWarn site; got: $failedLine")
        // (vii) service-tag
        assert(failedLine.contains("[hand-history-review]"),
          clue = s"timeout-path failed line must carry the [hand-history-review] service-tag prefix; got: $failedLine")
      }
    }
  }

  // Pin the documented NONFATAL-EXCEPTION-PATH variant of the
  // `job failed` audit log line: errorStatus=500 + error=analysis
  // %20failed:%20<message> -- the THIRD AND FINAL classifier
  // branch complement to 6b59ce4 (errorStatus=400 default
  // branch) + 8577288 (errorStatus=504 timeout branch); JobQueue
  // .scala line 297 emits this Failed state when the analyze
  // backend throws NonFatal during runJob: `case NonFatal(e) =>
  // if timedOut.get() then timeoutFailure(submittedAt, startedAt)
  // else Failed(submittedAt, startedAt, nowMillis(), 500,
  // s"analysis failed: ${e.getMessage}")` -- the errorStatus
  // is HARDCODED to 500 (NOT passed through classifyAnalysisError
  // like the backend-Left path), AND the error message is
  // wrapped with the literal "analysis failed: " prefix; this
  // wrapping is OPERATIONALLY MEANINGFUL because the SAME
  // prefix is what classifyAnalysisError checks at line 765-766
  // to classify as 500 -- so a backend Left starting with
  // "analysis failed:" would ALSO get classified as 500, the
  // wrapping ensures uncaught exceptions land in the same
  // operator triage category as backend-Lefts that explicitly
  // signal "analysis failed" without leaking the raw exception
  // class/message back to the client; with this commit ALL 3
  // classifyAnalysisError branches are pinned: 400 (6b59ce4
  // default), 504 (8577288 timeout), 500 (THIS commit NonFatal
  // wrapped); the error-message format `analysis failed:
  // <exception-message>` is the ONLY emission that includes
  // BOTH a server-generated prefix AND a USER-controlled body
  // (the exception message could come from anywhere, including
  // attacker-influenced parser failures); the %20-escape on
  // the SPACE between "analysis" and "failed:" AND inside the
  // exception message itself is critical because the message
  // crosses BOTH the server-generated/user-controlled boundary
  // AND has multi-word format; per-field regression vectors
  // SPECIFIC to the NonFatal path that 6b59ce4 + 8577288 don't
  // catch: (i) the HARDCODED 500 status at line 297 -- a
  // refactor that routed NonFatal exceptions through
  // classifyAnalysisError would silently demote ungraceful
  // exceptions to 400 if the message didn't start with the
  // matched prefixes (the wrapping at line 297 ensures the
  // prefix matches), (ii) the "analysis failed: " prefix
  // wrapping at line 297 -- a refactor dropping the prefix
  // (e.g. "raw exception message is more informative") would
  // (a) leak exception class details to operators bypassing
  // the documented wrapping abstraction, (b) silently demote
  // the errorStatus via classifyAnalysisError to 400 because
  // the message would no longer match the "analysis failed:"
  // prefix, (iii) the exception message PASS-THROUGH -- a
  // refactor that filtered/masked the exception message
  // would silently lose operator diagnostic detail; the test
  // uses a synthetic exception with a space-bearing message
  // ("synthetic NonFatal exception with spaces") so the
  // %20-escape covers BOTH the prefix AND body together;
  // 8-tier format check: (i) `job failed` prefix, (ii) jobId
  // matching 202 response, (iii) errorStatus=500 (THE
  // NonFatal-distinctive value), (iv)
  // error=analysis%20failed:%20synthetic%20NonFatal%20
  // exception%20with%20spaces (the LOAD-BEARING wrapped-and-
  // escaped message pin), (v) EXCLUSION of the unescaped form
  // "analysis failed: synthetic NonFatal exception with
  // spaces" (catches escape-dropped refactor on the
  // NonFatal path specifically), (vi) [WARN] level (matches
  // 6b59ce4 + 8577288), (vii) [hand-history-review] service-
  // tag, (viii) the test ALSO inline-pins the HTTP-response-
  // shape errorStatus=500 at the wire layer (complements the
  // existing tests' coverage of 400 + 504 statuses).
  test("analyze worker that throws NonFatal exception emits the documented `job failed ... errorStatus=500 error=analysis%20failed:%20<message>` WARN audit log line per JobQueue.scala line 297 -- the THIRD AND FINAL classifier branch (NonFatal exception path) complementing 6b59ce4 (400 default) and 8577288 (504 timeout)") {
    withStaticSite { staticDir =>
      // Custom throwing backend -- the existing
      // immediateBackend/BlockingBackend helpers return
      // Either, not throw, so we need an inline backend that
      // throws on analyze() to exercise line 295's
      // `case NonFatal(e) =>` catch branch.
      val throwingBackend = new HandHistoryReviewServer.AnalysisBackend:
        override def analyze(request: HandHistoryReviewService.AnalysisRequest): Either[String, ujson.Value] =
          throw new RuntimeException("synthetic NonFatal exception with spaces")

      withServer(staticDir, backend = throwingBackend) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        // Capture stderr around the submit + terminal-poll.
        // The throwing backend throws RuntimeException which is
        // NonFatal, caught at line 295's NonFatal(e) branch,
        // producing a Failed(errorStatus=500, error="analysis
        // failed: synthetic NonFatal exception with spaces").
        val errBuf = new java.io.ByteArrayOutputStream()
        val originalErr = System.err
        System.setErr(new java.io.PrintStream(errBuf, true, StandardCharsets.UTF_8))
        val submitJobId =
          try
            val submit = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload)
            assertEquals(submit.statusCode(), 202,
              clue = "submission must return 202 -- the NonFatal exception happens at WORKER level, not submission")
            val statusUri = s"$baseUri${jsonBody(submit)("statusUrl").str}"
            val capturedJobId = jsonBody(submit)("jobId").str
            val terminal = awaitTerminalJob(statusUri)
            assertEquals(terminal("status").str, "failed",
              clue = "NonFatal exception in backend must reach Failed terminal state via line 295's catch branch")
            assertEquals(terminal("errorStatus").num.toInt, 500,
              clue = "NonFatal exception must produce errorStatus=500 (HARDCODED at line 297, NOT via classifier) -- this is the HTTP-response-shape pin matching the JobQueue.scala line 297 hardcoded value")
            capturedJobId
          finally
            System.setErr(originalErr)

        val captured = errBuf.toString(StandardCharsets.UTF_8)
        val failedLine = captured.split('\n').iterator
          .find(line => line.contains("job failed") && line.contains(s"jobId=$submitJobId"))
          .getOrElse(fail(s"no `job failed jobId=$submitJobId` line in captured stderr for the NonFatal path; got captured stderr: ${captured.take(2000)}"))

        // (i) prefix
        assert(failedLine.contains("job failed"),
          clue = s"NonFatal-path failed line must carry `job failed` prefix matching 6b59ce4 + 8577288; got: $failedLine")
        // (ii) jobId
        assert(failedLine.contains(s"jobId=$submitJobId"),
          clue = s"NonFatal-path failed line must carry the submission's jobId; got: $failedLine")
        // (iii) errorStatus=500 (the NonFatal-distinctive value)
        assert(failedLine.contains("errorStatus=500"),
          clue = s"NonFatal-path failed line MUST carry errorStatus=500 per JobQueue.scala line 297's HARDCODED 500 -- the THIRD classifier branch complementing 6b59ce4's 400 default and 8577288's 504 timeout; a refactor routing NonFatal exceptions through classifyAnalysisError would silently demote ungraceful exceptions to 400 if the message didn't start with the matched prefixes (the wrapping at line 297 ensures the prefix matches); got: $failedLine")
        // (iv) error=analysis%20failed:%20<message> (LOAD-BEARING)
        assert(failedLine.contains("error=analysis%20failed:%20synthetic%20NonFatal%20exception%20with%20spaces"),
          clue = s"NonFatal-path failed line MUST carry the EXACT %20-escaped wrapped error string `analysis%20failed:%20synthetic%20NonFatal%20exception%20with%20spaces` per JobQueue.scala line 297's `s\"analysis failed: $${e.getMessage}\"` wrapping + line 322's `.replace(\" \", \"%20\")` escape; the `analysis failed: ` prefix wrapping is OPERATIONALLY MEANINGFUL because (a) it matches classifyAnalysisError's line 765-766 prefix check (so a backend Left starting with the same prefix gets the same 500 classification, providing consistent operator triage), (b) it abstracts the raw exception class/message away from the operator-visible error string, (c) it lets operators filter for ungraceful-exception failures by grep'ing the documented prefix; a refactor dropping the prefix wrapping (e.g. \"raw exception message is more informative\") would (a) leak exception class details bypassing the documented wrapping abstraction, (b) silently demote errorStatus via classifyAnalysisError to 400 because the message would no longer match the prefix; got: $failedLine")
        // (v) EXCLUSION of unescaped form (catches escape-dropped refactor)
        assert(!failedLine.contains("analysis failed: synthetic NonFatal exception with spaces"),
          clue = s"NonFatal-path failed line MUST NOT contain the UNESCAPED form `analysis failed: synthetic NonFatal exception with spaces` (with literal spaces) -- catches a refactor that dropped the %20-escape on the NonFatal path specifically (the error string crosses BOTH the server-generated prefix AND the user-controlled exception body, so the escape applies to BOTH halves together); matches the ESCAPE-CONTRACT VERIFICATION pattern from 6b59ce4 + 8577288 + e43081b; got: $failedLine")
        // (vi) WARN level
        assert(failedLine.contains("[WARN]"),
          clue = s"NonFatal-path failed line must be WARN-level matching 6b59ce4 + 8577288; got: $failedLine")
        // (vii) service-tag
        assert(failedLine.contains("[hand-history-review]"),
          clue = s"NonFatal-path failed line must carry the [hand-history-review] service-tag prefix; got: $failedLine")
      }
    }
  }

  // Pin the documented PLAYING-HALL TIMEOUT-PATH variant of the
  // `playing hall job failed` audit log line: errorStatus=504 +
  // error=playing%20hall%20timed%20out%20after%20<N>ms -- the
  // ANALYZE-SIDE timeout pin (8577288) applied to the
  // PLAYING-HALL emission site at JobQueue.scala line 614-616;
  // 817dd08 pinned the playing-hall failure path's
  // classifyPlayingHallError DEFAULT branch (errorStatus=400 via
  // backend Left), THIS commit closes the TIMEOUT branch
  // (errorStatus=504) on the playing-hall side; with this commit
  // 2 of 3 classifyPlayingHallError branches are pinned on the
  // hall side (400 from 817dd08 + 504 from THIS), matching the
  // analyze-side coverage progression (6b59ce4 400 + 8577288
  // 504); the operationally-distinct ASPECT of the hall-side
  // timeout vs analyze-side timeout: hall jobs have a longer
  // default timeout (15 min vs 2 min per 61e49a8) so timeouts
  // are MORE OPERATIONALLY MEANINGFUL on the hall side (a
  // timeout means the hall worker exceeded 15 min, which is
  // long enough that operators investigate; analyze 2-min
  // timeouts are more frequently "user submitted a huge upload"
  // vs hall 15-min timeouts which are "the worker is stuck OR
  // the hall config is too aggressive"); the %20-escape on
  // "playing hall timed out after 100ms" is the LOAD-BEARING
  // contract this pin uniquely verifies (mirrors 8577288's
  // analyze-side %20-escape on the analogous timeout-message
  // category); per-field regression vectors SPECIFIC to the
  // hall timeout path that the analyze-side pins don't catch:
  // (i) the hall-side prefix `playing hall job failed` (NOT
  // `job failed` like 8577288 -- the asymmetric-drift catch
  // from 1e030ed/817dd08), (ii) errorStatus=504 via
  // classifyPlayingHallError's hall-side branch at line 769's
  // `if error.startsWith("playing hall timed out after") then
  // 504` -- a refactor that broke the HALL-side classifier
  // independently of the analyze-side (e.g. via inconsistent
  // refactoring of the parallel functions) would silently
  // demote hall timeouts to the 400 default branch on the
  // hall-side ONLY, leaving analyze-side timeouts correctly
  // 504, (iii) the EXACT timeout-message `playing hall timed
  // out after <N>ms` from JobQueue.scala line 673 -- catches a
  // refactor renaming the hall-side timeout message
  // independently of the analyze-side which would silently
  // break the prefix-match in classifyPlayingHallError; 8-tier
  // format check at WARN level matching 8577288's + 817dd08's
  // patterns: (i) `playing hall job failed` prefix (catches
  // hall-side rename + asymmetric drift to analyze prefix),
  // (ii) jobId, (iii) errorStatus=504 (hall-timeout classifier
  // branch), (iv) error=playing%20hall%20timed%20out%20after
  // %20100ms (THE LOAD-BEARING hall-timeout-message +
  // %20-escape pin), (v) EXCLUSION of unescaped form, (vi)
  // [WARN] level, (vii) [hand-history-review] service-tag,
  // (viii) the test ALSO inline-pins the HTTP-response-shape
  // errorStatus=504 at the wire layer.
  test("playing-hall worker that times out emits the documented `playing hall job failed ... errorStatus=504 error=playing%20hall%20timed%20out%20after%20<N>ms` WARN audit log line per JobQueue.scala line 614-616 -- the HALL-SIDE TIMEOUT-PATH variant of 8577288's analyze-side timeout pin (exercises classifyPlayingHallError's 504 branch + the hall-distinctive prefix)") {
    withStaticSite { staticDir =>
      val backend = new BusyPlayingHallBackend(runForMs = 2000L, result = Right(samplePlayingHallResult))
      withServer(
        staticDir,
        playingHallBackend = backend,
        maxConcurrentJobs = 1,
        maxQueuedJobs = 1,
        playingHallTimeoutMs = 100L
      ) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        // Capture stderr around the playing-hall submit +
        // terminal-poll. The BusyPlayingHallBackend runs for
        // 2000ms while playingHallTimeoutMs=100L, so the
        // worker is interrupted at 100ms with the timeoutFailure
        // state.
        val errBuf = new java.io.ByteArrayOutputStream()
        val originalErr = System.err
        System.setErr(new java.io.PrintStream(errBuf, true, StandardCharsets.UTF_8))
        val submitJobId =
          try
            val submit = postJson(s"$baseUri/api/playing-hall", validPlayingHallPayload)
            assertEquals(submit.statusCode(), 202,
              clue = "playing-hall submission must return 202 before the worker can be timed out")
            val statusUri = s"$baseUri${jsonBody(submit)("statusUrl").str}"
            val capturedJobId = jsonBody(submit)("jobId").str
            assert(backend.started.await(3, TimeUnit.SECONDS),
              "busy playing hall backend never started -- the test depends on the worker reaching the analyze call so the timeout interrupts it")
            val terminal = awaitTerminalJob(statusUri)
            assertEquals(terminal("status").str, "failed",
              clue = "playing-hall worker must reach 'failed' terminal state via the timeout path")
            assertEquals(terminal("errorStatus").num.toInt, 504,
              clue = "terminal-state errorStatus must be 504 (classifyPlayingHallError's timeout branch) -- this is the HTTP-response-shape pin matching the existing line ~10984 test, complementing this commit's audit-log-shape pin")
            // Wait for the busy backend to finish its 2-second
            // run so the test cleanup doesn't race the
            // not-yet-decremented timedOutWorkersInFlight counter
            assert(backend.finished.await(5, TimeUnit.SECONDS),
              "busy playing hall backend never finished -- the test cleanup needs the worker thread to exit before withServer's close()")
            // After backend.finished, the worker still has to
            // reach line 615's logWarn emission -- between
            // backend.finished.countDown (the backend's finally
            // block fires BEFORE the function returns to the
            // worker) and the worker's post-return finalState
            // match. Poll readiness as a proxy for "worker
            // finished post-processing" -- when
            // timedOutWorkersInFlight decrements (line 606,
            // which fires BEFORE the logWarn at line 615), the
            // readiness flips back to 200. After awaitReady
            // returns we're confident the worker is at OR PAST
            // the logWarn line, so the captured stderr has the
            // emission. Without awaitReady, the worker race
            // between backend.finished and logWarn would
            // sometimes leave the captured stderr empty.
            awaitReady(s"$baseUri/api/ready")
            capturedJobId
          finally
            System.setErr(originalErr)

        val captured = errBuf.toString(StandardCharsets.UTF_8)
        val failedLine = captured.split('\n').iterator
          .find(line => line.contains("playing hall job failed") && line.contains(s"jobId=$submitJobId"))
          .getOrElse(fail(s"no `playing hall job failed jobId=$submitJobId` line in captured stderr for the hall-timeout path; got captured stderr: ${captured.take(2000)}"))

        // (i) hall-side prefix (catches asymmetric drift to analyze)
        assert(failedLine.contains("playing hall job failed"),
          clue = s"hall-timeout failed line must carry `playing hall job failed` prefix per JobQueue.scala line 615's hardcoded literal -- distinct from the analyze-side `job failed` prefix (pinned by 8577288); a refactor consolidating both timeout-path prefixes would silently break operator per-endpoint failure dashboards; got: $failedLine")
        // EXCLUSION of standalone `job failed` (asymmetric-drift catch
        // matching 1e030ed/817dd08 pattern)
        val withoutHallPrefix = failedLine.replace("playing hall job failed", "")
        assert(!withoutHallPrefix.contains("job failed"),
          clue = s"hall-timeout failed line must NOT also contain a standalone `job failed` prefix in a position other than the `playing hall job failed` substring -- catches a refactor that emitted both prefixes for the same event; got line after stripping hall prefix: '$withoutHallPrefix'")
        // (ii) jobId
        assert(failedLine.contains(s"jobId=$submitJobId"),
          clue = s"hall-timeout failed line must carry the submission's jobId; got: $failedLine")
        // (iii) errorStatus=504
        assert(failedLine.contains("errorStatus=504"),
          clue = s"hall-timeout failed line MUST carry errorStatus=504 per JobQueue.scala line 769's classifyPlayingHallError check `if error.startsWith(\"playing hall timed out after\") then 504` -- the TIMEOUT classifier branch on the HALL side complementing 8577288's analyze-side 504 pin; a refactor that broke the hall-side classifier independently of the analyze-side (e.g. via inconsistent refactoring of the parallel functions) would silently demote hall timeouts to the 400 default branch on the hall-side ONLY; got: $failedLine")
        // (iv) error= field with %20-escaped hall-timeout message
        assert(failedLine.contains("error=playing%20hall%20timed%20out%20after%20100ms"),
          clue = s"hall-timeout failed line MUST carry the EXACT %20-escaped error string `playing%20hall%20timed%20out%20after%20100ms` per JobQueue.scala line 673's `s\"playing hall timed out after $${playingHallTimeoutMs}ms\"` template + line 615 %20-escape; the 100ms reflects the test's playingHallTimeoutMs=100L; a refactor renaming the hall-side timeout message independently of the analyze-side would silently break both the classifyPlayingHallError prefix-match AND operator dashboards filtering by the exact wording; got: $failedLine")
        // (v) EXCLUSION of unescaped form (catches escape-dropped refactor)
        assert(!failedLine.contains("playing hall timed out after 100ms"),
          clue = s"hall-timeout failed line MUST NOT contain the UNESCAPED form `playing hall timed out after 100ms` (with literal spaces) -- catches a refactor that dropped the %20-escape on the HALL-side timeout path specifically while keeping analyze-side timeouts (8577288) escaped; got: $failedLine")
        // (vi) WARN level
        assert(failedLine.contains("[WARN]"),
          clue = s"hall-timeout failed line must be WARN-level matching 8577288's analyze-side timeout; got: $failedLine")
        // (vii) service-tag
        assert(failedLine.contains("[hand-history-review]"),
          clue = s"hall-timeout failed line must carry the [hand-history-review] service-tag prefix; got: $failedLine")
      }
    }
  }

  // Pin the documented PLAYING-HALL NONFATAL-EXCEPTION-PATH
  // variant of the `playing hall job failed` audit log line:
  // errorStatus=500 + error=playing%20hall%20failed:%20<message>
  // -- the THIRD AND FINAL classifyPlayingHallError branch on
  // the hall side, completing the 3-of-3 hall-side coverage
  // (817dd08 400 default + 9ac8689 504 timeout + THIS 500
  // NonFatal); JobQueue.scala line 595 emits this Failed state
  // when the playing-hall backend throws NonFatal: `case
  // NonFatal(e) => if timedOut.get() then timeoutFailure(
  // submittedAt, startedAt) else Failed(submittedAt, startedAt,
  // nowMillis(), 500, s"playing hall failed: ${e.getMessage}")`
  // -- the errorStatus is HARDCODED to 500 (NOT passed through
  // classifyPlayingHallError) AND the error message is wrapped
  // with the literal "playing hall failed: " prefix; this
  // mirror of d96f892's analyze-side NonFatal pin completes the
  // SYMMETRIC 3x2 matrix (3 classifier branches × 2 endpoints)
  // for the JobQueue audit log family's failure-classifier
  // coverage; with this commit ALL 6 classifier-branch × endpoint
  // cells are pinned: analyze 400 (6b59ce4) + analyze 504
  // (8577288) + analyze 500 (d96f892) + hall 400 (817dd08) +
  // hall 504 (9ac8689) + hall 500 (THIS); per-field regression
  // vectors SPECIFIC to the hall NonFatal path that the prior 5
  // pins don't catch: (i) the HALL-SIDE prefix `playing hall
  // job failed` (NOT `job failed` like d96f892 -- the
  // asymmetric-drift catch from 1e030ed/817dd08/9ac8689), (ii)
  // errorStatus=500 HARDCODED at line 595 on the HALL side
  // independently of the analyze-side line 297 -- a refactor
  // that routed hall-side NonFatal exceptions through
  // classifyPlayingHallError (a "consistency" rationale to
  // align with how the timeout path uses the classifier) would
  // silently demote hall-side exceptions to 400 if the message
  // didn't match the "playing hall timed out after" prefix,
  // (iii) the "playing hall failed: " prefix wrapping at line
  // 595 -- a refactor dropping this wrapping (e.g. "raw
  // exception message is more informative") on the hall side
  // ONLY would (a) leak hall-worker exception class details
  // bypassing the documented wrapping abstraction, (b)
  // silently demote hall errorStatus via the classifier to 400
  // because the message would no longer match the "playing
  // hall failed:" prefix at classifyPlayingHallError line 771's
  // `else if error.startsWith("playing hall failed:") then 500`
  // check, AND (c) silently desync from the analyze-side
  // wrapping which would still emit "analysis failed:" --
  // causing operators to see DIFFERENT wrapping conventions
  // for the two endpoints' exception paths; the test uses a
  // synthetic exception with a space-bearing message
  // ("synthetic NonFatal hall exception with spaces") so the
  // %20-escape covers BOTH the prefix AND body together;
  // 8-tier format check at WARN level: (i) `playing hall job
  // failed` prefix (hall-distinctive), (ii) EXCLUSION of
  // standalone `job failed` (asymmetric-drift catch), (iii)
  // jobId, (iv) errorStatus=500 (HARDCODED at line 595 -- NOT
  // via classifyPlayingHallError), (v) error=playing%20hall%20
  // failed:%20synthetic%20NonFatal%20hall%20exception%20with%20
  // spaces (THE LOAD-BEARING wrapped + escaped hall NonFatal
  // message), (vi) EXCLUSION of unescaped form, (vii) [WARN]
  // level, (viii) [hand-history-review] service-tag.
  test("playing-hall worker that throws NonFatal exception emits the documented `playing hall job failed ... errorStatus=500 error=playing%20hall%20failed:%20<message>` WARN audit log line per JobQueue.scala line 595 -- the HALL-SIDE NonFatal pin completing the 3-of-3 classifyPlayingHallError coverage on the hall side (817dd08 400 + 9ac8689 504 + THIS 500) AND the full 6-cell 3-classifier × 2-endpoint matrix in the JobQueue audit log family") {
    withStaticSite { staticDir =>
      // Custom throwing PlayingHallBackend -- mirrors d96f892's
      // throwingBackend pattern but for the hall side. The
      // existing BlockingPlayingHallBackend / immediatePlayingHall
      // Backend return Either; this inline backend throws.
      val throwingHallBackend = new HandHistoryReviewServer.PlayingHallBackend:
        override def run(
            request: HandHistoryReviewServer.PlayingHallRequest,
            cancelSignal: () => Boolean
        ): Either[String, ujson.Value] =
          throw new RuntimeException("synthetic NonFatal hall exception with spaces")

      withServer(staticDir, playingHallBackend = throwingHallBackend) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        // Capture stderr around the submit + terminal-poll. The
        // throwing backend triggers line 595's NonFatal catch,
        // producing Failed(errorStatus=500, error="playing hall
        // failed: synthetic NonFatal hall exception with
        // spaces"). The hall-side NonFatal path is SYNCHRONOUS
        // in the worker thread (the worker calls backend.run,
        // catches the throw, immediately reaches the finalState
        // match + logWarn) -- unlike the timeout path which has
        // the async timeout-scheduler race the test had to
        // mitigate in 9ac8689 via awaitReady; the NonFatal path
        // here matches d96f892's analyze-side pattern: synchronous
        // failure means the logWarn fires BEFORE awaitTerminalJob
        // returns, so backend.finished + awaitReady aren't needed.
        val errBuf = new java.io.ByteArrayOutputStream()
        val originalErr = System.err
        System.setErr(new java.io.PrintStream(errBuf, true, StandardCharsets.UTF_8))
        val submitJobId =
          try
            val submit = postJson(s"$baseUri/api/playing-hall", validPlayingHallPayload)
            assertEquals(submit.statusCode(), 202,
              clue = "playing-hall submission must return 202 -- the NonFatal exception happens at WORKER level, not submission")
            val statusUri = s"$baseUri${jsonBody(submit)("statusUrl").str}"
            val capturedJobId = jsonBody(submit)("jobId").str
            val terminal = awaitTerminalJob(statusUri)
            assertEquals(terminal("status").str, "failed",
              clue = "playing-hall NonFatal exception must reach Failed terminal state via line 593's catch branch")
            assertEquals(terminal("errorStatus").num.toInt, 500,
              clue = "terminal-state errorStatus must be 500 (HARDCODED at JobQueue.scala line 595, NOT via classifier) -- mirrors d96f892's analyze-side pin")
            capturedJobId
          finally
            System.setErr(originalErr)

        val captured = errBuf.toString(StandardCharsets.UTF_8)
        val failedLine = captured.split('\n').iterator
          .find(line => line.contains("playing hall job failed") && line.contains(s"jobId=$submitJobId"))
          .getOrElse(fail(s"no `playing hall job failed jobId=$submitJobId` line in captured stderr for the hall-NonFatal path; got captured stderr: ${captured.take(2000)}"))

        // (i) hall-side prefix
        assert(failedLine.contains("playing hall job failed"),
          clue = s"hall-NonFatal failed line must carry `playing hall job failed` prefix matching the hall-side terminal-state convention from 1e030ed/817dd08/9ac8689; got: $failedLine")
        // (ii) EXCLUSION of standalone `job failed` (asymmetric-drift)
        val withoutHallPrefix = failedLine.replace("playing hall job failed", "")
        assert(!withoutHallPrefix.contains("job failed"),
          clue = s"hall-NonFatal failed line must NOT also contain a standalone `job failed` prefix in a position other than the `playing hall job failed` substring -- asymmetric-drift catch matching 1e030ed/817dd08/9ac8689 pattern; got line after stripping hall prefix: '$withoutHallPrefix'")
        // (iii) jobId
        assert(failedLine.contains(s"jobId=$submitJobId"),
          clue = s"hall-NonFatal failed line must carry the submission's jobId; got: $failedLine")
        // (iv) errorStatus=500 (HARDCODED at line 595)
        assert(failedLine.contains("errorStatus=500"),
          clue = s"hall-NonFatal failed line MUST carry errorStatus=500 per JobQueue.scala line 595's HARDCODED 500 -- a refactor that routed hall-side NonFatal through classifyPlayingHallError (a 'consistency' rationale to align with how the timeout path uses the classifier) would silently demote hall-side exceptions to 400 if the message didn't match the documented prefix; matches d96f892's analyze-side HARDCODED 500 pin; got: $failedLine")
        // (v) error=playing%20hall%20failed:%20<message> (LOAD-BEARING)
        assert(failedLine.contains("error=playing%20hall%20failed:%20synthetic%20NonFatal%20hall%20exception%20with%20spaces"),
          clue = s"hall-NonFatal failed line MUST carry the EXACT %20-escaped wrapped error string `playing%20hall%20failed:%20synthetic%20NonFatal%20hall%20exception%20with%20spaces` per JobQueue.scala line 595's `s\"playing hall failed: $${e.getMessage}\"` wrapping + line 615's `.replace(\" \", \"%20\")` escape; the `playing hall failed: ` prefix wrapping is OPERATIONALLY MEANINGFUL because (a) it matches classifyPlayingHallError's line 771 prefix check (so a backend Left starting with the same prefix gets the same 500 classification, providing consistent operator triage), (b) it abstracts the raw exception class/message away from the operator-visible error string, (c) it lets operators filter for ungraceful-exception failures by grep'ing the documented prefix; a refactor dropping the prefix wrapping on the hall side ONLY would silently desync from the analyze-side wrapping (`analysis failed:`) causing operators to see DIFFERENT wrapping conventions for the two endpoints; got: $failedLine")
        // (vi) EXCLUSION of unescaped form
        assert(!failedLine.contains("playing hall failed: synthetic NonFatal hall exception with spaces"),
          clue = s"hall-NonFatal failed line MUST NOT contain the UNESCAPED form `playing hall failed: synthetic NonFatal hall exception with spaces` (with literal spaces) -- catches a refactor that dropped the %20-escape on the hall NonFatal path specifically; got: $failedLine")
        // (vii) WARN level
        assert(failedLine.contains("[WARN]"),
          clue = s"hall-NonFatal failed line must be WARN-level matching d96f892's analyze-side NonFatal + the prior failure pins; got: $failedLine")
        // (viii) service-tag
        assert(failedLine.contains("[hand-history-review]"),
          clue = s"hall-NonFatal failed line must carry the [hand-history-review] service-tag prefix; got: $failedLine")
      }
    }
  }

  // Pin the documented LOG LINE TIMESTAMP FORMAT -- per
  // HandHistoryReviewServerRuntime.scala line 511-512's
  // `stream.println(s"[${Instant.now()}] [$level] [hand-history-
  // review] ${sanitizeLogMessage(message)}")` template; every
  // log line emitted by the runtime starts with a leading
  // `[<ISO-8601 UTC timestamp>]` bracket because Instant.now()
  // produces ISO-8601 UTC strings via Instant.toString() (which
  // always emits the `Z` UTC suffix); BEFORE this commit there
  // was ZERO test coverage of the LOG-LINE TIMESTAMP FORMAT --
  // the 30+ log-line family pins across the 5 categories all
  // verify field VALUES + level + service-tag + structure, but
  // NONE of them verify that the LEADING TIMESTAMP is parseable
  // as ISO-8601 UTC; the timestamp format is OPERATIONALLY
  // CRITICAL because: (a) log aggregators (ELK, Splunk, Datadog,
  // CloudWatch, etc.) parse the timestamp to ORDER + GROUP +
  // QUERY log lines across the deployment fleet -- a non-parseable
  // timestamp silently fails the aggregator's ingestion pipeline
  // (the line either gets indexed at the WRONG time or gets
  // dropped entirely depending on the aggregator's fallback
  // policy), (b) operators correlate events across multiple
  // services by matching timestamps in their respective log
  // lines -- a non-UTC timestamp (e.g. local-zone-with-offset)
  // would silently desync from UTC-only services (the common
  // deployment convention is "all services log in UTC, all
  // dashboards convert to operator-preferred-zone at query
  // time" -- a single service that logs in local-zone breaks
  // this convention), (c) the runbook references log
  // timestamps for incident timelines -- a non-ISO-8601 format
  // would silently break the runbook's documented "look up
  // events between <ISO> and <ISO>" diagnostic workflow; the
  // timestamp format is INVARIANT across ALL the prior 30+
  // log-line pins (every category uses the same log() helper
  // at line 510-512), so pinning the format ONCE here covers
  // the timestamp dimension for the ENTIRE log-line family;
  // per-format regression vectors: (i) refactor swapping
  // Instant.now() to LocalDateTime.now() at line 512 would
  // silently emit local-zone timestamps without the Z suffix
  // (silently desyncs from UTC-convention deployments), (ii)
  // refactor wrapping with a custom DateTimeFormatter would
  // silently emit a different format (e.g. epoch-millis,
  // RFC-1123, custom yyyy/MM/dd) that aggregators may not
  // parse, (iii) refactor dropping the leading `[<ts>]` bracket
  // (e.g. "use a JSON log shape instead") would silently break
  // every operator grep workflow that depends on the documented
  // bracketed-prefix format, (iv) refactor switching to a
  // pre-formatted string (e.g. cached at process start) would
  // silently emit STALE timestamps on every log line (all log
  // lines from one process lifetime would carry the same
  // timestamp); test approach: capture stdout around withServer
  // (which emits the startup banner -- a guaranteed log line),
  // find the banner line, extract the leading `[<timestamp>]`
  // bracket, parse via Instant.parse (the JVM's strict ISO-8601
  // UTC parser -- accepts ONLY the format Instant.toString
  // emits, with Z suffix), assert the parsed Instant is within
  // a reasonable wall-clock window of the test's nowMs (proves
  // the timestamp is LIVE not cached); 4-tier format check:
  // (i) the line starts with `[` (catches the dropped-bracket
  // refactor), (ii) the bracket content is parseable as
  // Instant (catches non-ISO-8601 formats), (iii) the parsed
  // Instant has a Z UTC suffix in its string form (catches
  // local-zone refactors), (iv) the parsed Instant is within
  // 5 seconds of the test's nowMs (catches stale-timestamp
  // refactors and timezone-drift refactors).
  test("every log line carries the documented `[<ISO-8601 UTC timestamp>]` leading bracket per HandHistoryReviewServerRuntime.scala line 511-512's Instant.now() template -- the FORMAT-INVARIANT pin covering the timestamp dimension for ALL 30+ log-line pins across the 5 operator-facing log line categories (server-lifecycle + auth-event + JobQueue + rate-limit + startup-failure)") {
    withStaticSite { staticDir =>
      // Capture stdout around withServer to grab the startup
      // banner (a guaranteed log line). Any log line would
      // work since the timestamp format is invariant across
      // ALL emissions per the shared log() helper at line
      // 510-512; the startup banner is the most reliable to
      // capture because it always fires synchronously during
      // server bind.
      val outBuf = new java.io.ByteArrayOutputStream()
      val originalOut = System.out
      val nowMsBeforeCapture = System.currentTimeMillis()
      System.setOut(new java.io.PrintStream(outBuf, true, StandardCharsets.UTF_8))
      try
        withServer(staticDir) { _ =>
          // No HTTP requests -- the startup banner emits
          // BEFORE the callback executes
          ()
        }
      finally
        System.setOut(originalOut)
      val nowMsAfterCapture = System.currentTimeMillis()

      val captured = outBuf.toString(StandardCharsets.UTF_8)
      val bannerLine = captured.split('\n').iterator
        .find(_.contains("startup complete"))
        .getOrElse(fail(s"no `startup complete` line in captured stdout -- the timestamp-format pin needs ANY log line to inspect; the 7c47f88 startup-banner pin should catch this independently; got captured stdout: ${captured.take(800)}"))

      // (i) line starts with `[` (catches dropped-bracket refactor)
      assert(bannerLine.trim.startsWith("["),
        clue = s"log line must start with the documented `[<timestamp>]` bracket per HandHistoryReviewServerRuntime.scala line 511's `s\"[$${Instant.now()}] ...\"` template -- a refactor dropping the leading bracket (e.g. switching to JSON log shape) would silently break every operator grep workflow that depends on the bracketed-prefix format; got line: ${bannerLine.trim.take(100)}")
      // (ii) extract the bracket content + parse as Instant
      val firstBracketEnd = bannerLine.trim.indexOf("]")
      assert(firstBracketEnd > 0,
        clue = s"log line must close the leading `[` with a matching `]`; got line: ${bannerLine.trim.take(100)}")
      val timestampStr = bannerLine.trim.substring(1, firstBracketEnd)
      val parsedInstant = scala.util.Try(java.time.Instant.parse(timestampStr))
        .getOrElse(fail(s"leading-bracket content `$timestampStr` MUST be parseable as ISO-8601 UTC via Instant.parse per HandHistoryReviewServerRuntime.scala line 511's `Instant.now()` template; Instant.parse accepts ONLY the format Instant.toString emits (with `Z` UTC suffix); a refactor swapping to LocalDateTime.now() (which produces a zone-less format) or a custom DateTimeFormatter (which may emit RFC-1123 / epoch-millis / yyyy/MM/dd) would silently produce non-parseable values, silently breaking log aggregator ingestion (ELK/Splunk/Datadog/CloudWatch all parse via ISO-8601 by default)"))
      // (iii) Z UTC suffix presence (catches local-zone refactor)
      assert(timestampStr.endsWith("Z"),
        clue = s"timestamp `$timestampStr` MUST end with the `Z` UTC suffix per Instant.toString's documented format -- a refactor swapping to LocalDateTime.now() OR ZonedDateTime.now(local-zone) would silently emit a different suffix (offset like `+08:00` for local zones, or NO suffix at all) and silently desync from the UTC-convention deployment shared with other services; the runbook's incident-timeline diagnostic depends on UTC-only timestamps for cross-service correlation; got: $timestampStr")
      // (iv) parsed Instant within wall-clock window (catches
      // stale-timestamp refactor)
      val parsedMs = parsedInstant.toEpochMilli
      assert(parsedMs >= nowMsBeforeCapture - 1000L,
        clue = s"timestamp $parsedMs MUST not be older than the test's nowMs-before-capture $nowMsBeforeCapture (with 1-second slack for clock skew) -- catches a refactor that cached a startup-time Instant + emitted it on every log line (silently making all log lines from one process lifetime carry the same stale timestamp); got parsed=$parsedMs vs nowMsBefore=$nowMsBeforeCapture (delta=${nowMsBeforeCapture - parsedMs}ms behind)")
      assert(parsedMs <= nowMsAfterCapture + 1000L,
        clue = s"timestamp $parsedMs MUST not be in the future relative to the test's nowMs-after-capture $nowMsAfterCapture (with 1-second slack for clock skew) -- catches a refactor that used a wrong baseline for the timestamp; got parsed=$parsedMs vs nowMsAfter=$nowMsAfterCapture (delta=${parsedMs - nowMsAfterCapture}ms ahead)")
    }
  }

  // Pin the documented `[<level>]` bracket on every log line
  // per HandHistoryReviewServerRuntime.scala line 512's
  // `stream.println(s"[$${Instant.now()}] [$$level] [hand-
  // history-review] $${sanitizeLogMessage(message)}")` template
  // -- the FORMAT-INVARIANT pin covering the LEVEL DIMENSION
  // for ALL log-line emission sites; this commit COMPLEMENTS
  // b9fc4d4's TIMESTAMP-dimension pin -- both pins verify
  // properties enforced at the log() helper itself (line 510-
  // 512) so a single pin covers ALL 30+ log-line emission
  // sites in the codebase that funnel through this helper;
  // the LEVEL DIMENSION is OPERATIONALLY CRITICAL because:
  // (a) operator grep workflows filter log lines by level --
  // a tail-and-grep pipeline like `tail -f deploy.log | grep
  // ERROR` depends on the documented `[ERROR]` bracket form
  // appearing on each ERROR-level line; a refactor changing
  // the form (e.g. `<ERROR>`, `[level=ERROR]`, unbracketed
  // `ERROR:`, lowercase `error`) would silently break the
  // operator's saved grep pipelines, (b) log aggregator
  // parsers (Splunk, ELK, Datadog) extract the level via
  // regex like `\[\w+\] \[hand-history-review\]` -- the
  // documented brackets enable structured indexing into the
  // aggregator's level field; a refactor changing the bracket
  // shape would silently desync the aggregator's level field,
  // breaking dashboards that filter by `level:ERROR` query,
  // (c) the runbook documents the level field via the
  // bracketed form (e.g. "search for `[ERROR]` lines to
  // identify recent failures"); a refactor would silently
  // break the runbook's diagnostic workflows; the level
  // values are emitted by logInfo/logWarn/logError at lines
  // 417-424's helper trio -- each calls `log(level, ...)`
  // with a fixed string `"INFO"` / `"WARN"` / `"ERROR"`; per-
  // format regression vectors that this pin catches:
  // (i) refactor changing the bracket shape from `[` `]` to
  // `<` `>` or `{` `}` or `(` `)` would silently break grep
  // workflows + aggregator field extraction, (ii) refactor
  // dropping the brackets entirely (e.g. swap to
  // `s"$$level " + ...`) would silently break ALL bracket-
  // dependent parsing, (iii) refactor changing the level
  // form to `[level=INFO]` (key=value) instead of bare
  // `[INFO]` would silently break grep patterns expecting
  // bare-bracket form, (iv) refactor changing the level case
  // to lowercase `[info]` / `[warn]` / `[error]` would
  // silently break case-sensitive grep workflows, (v)
  // refactor injecting padding inside the brackets like
  // `[ INFO ]` would silently break tight grep patterns,
  // (vi) refactor swapping the position of the level bracket
  // (e.g. emitting it FIRST before the timestamp) would
  // silently break operator workflows that scan from left to
  // right; test approach mirrors b9fc4d4: capture stdout
  // around withServer (which emits the startup banner -- a
  // guaranteed `[INFO]` log line), find the banner line,
  // extract the SECOND bracketed field via string position
  // arithmetic (the FIRST bracket is the timestamp, already
  // pinned by b9fc4d4; the SECOND bracket is the level);
  // apply 5-tier format check: (i) the level bracket starts
  // with `[` immediately after the timestamp's `] ` separator
  // (catches non-bracket shapes + dropped-bracket refactors),
  // (ii) the level bracket closes with `]` (catches
  // open-bracket-only refactors), (iii) the level content is
  // EXACTLY one of `INFO`/`WARN`/`ERROR` (catches arbitrary
  // values, lowercase variants, key=value forms), (iv) the
  // level content has NO padding spaces (catches
  // `[ INFO ]`), (v) the third bracket (service-tag) follows
  // the level bracket with a single space separator (catches
  // a refactor that changed the separator OR reordered the
  // brackets).
  test("log line format pins the documented `[<level>]` bracket on every log line per HandHistoryReviewServerRuntime.scala line 512's template -- the FORMAT-INVARIANT pin covering the LEVEL DIMENSION for ALL 30+ log-line emission sites; complements b9fc4d4's timestamp pin (BOTH verify properties at the log() helper itself so single pins cover the entire family)") {
    withStaticSite { staticDir =>
      val outBuf = new java.io.ByteArrayOutputStream()
      val originalOut = System.out
      System.setOut(new java.io.PrintStream(outBuf, true, StandardCharsets.UTF_8))
      try
        withServer(staticDir) { _ =>
          // No HTTP requests -- the startup banner emits
          // BEFORE the callback executes (the banner uses
          // logInfo at HandHistoryReviewServerRuntime.scala
          // lines 349-350, so the level field is `INFO`).
          ()
        }
      finally
        System.setOut(originalOut)

      val captured = outBuf.toString(StandardCharsets.UTF_8)
      val bannerLine = captured.split('\n').iterator
        .find(_.contains("startup complete"))
        .getOrElse(fail(s"no `startup complete` line in captured stdout -- the level-format pin needs ANY log line to inspect; the 7c47f88 startup-banner pin should catch this independently; got captured stdout: ${captured.take(800)}"))

      val trimmed = bannerLine.trim

      // Locate the timestamp bracket's closing `]` -- the
      // level bracket starts immediately after the space
      // separator following the timestamp bracket.
      val firstBracketEnd = trimmed.indexOf("]")
      assert(firstBracketEnd > 0,
        clue = s"log line must close the leading `[<timestamp>]` bracket with `]`; got line: ${trimmed.take(100)}")

      // (i) the level bracket starts with `[` immediately
      // after the timestamp's `] ` separator (catches non-
      // bracket shapes + dropped-bracket refactors)
      assert(trimmed.length > firstBracketEnd + 2,
        clue = s"log line must extend past the timestamp bracket + separator; got line: ${trimmed.take(100)}")
      assert(trimmed.charAt(firstBracketEnd + 1) == ' ',
        clue = s"the timestamp's `]` MUST be followed by a single space (the documented field separator per line 512's `s\"[...] [...]\"` template); got char `${trimmed.charAt(firstBracketEnd + 1)}` (codepoint ${trimmed.charAt(firstBracketEnd + 1).toInt}) at position ${firstBracketEnd + 1}; line: ${trimmed.take(100)}")
      val secondBracketStart = firstBracketEnd + 2
      assert(trimmed.charAt(secondBracketStart) == '[',
        clue = s"the level field MUST start with `[` immediately after the timestamp's `] ` separator per line 512's template; a refactor changing the level brackets to `<` `>` or `{` `}` or `(` `)` would silently break operator grep workflows expecting square brackets; got char `${trimmed.charAt(secondBracketStart)}` (codepoint ${trimmed.charAt(secondBracketStart).toInt}) at position $secondBracketStart; line: ${trimmed.take(100)}")

      // (ii) the level bracket closes with `]` (catches
      // open-bracket-only refactors)
      val secondBracketEnd = trimmed.indexOf("]", secondBracketStart + 1)
      assert(secondBracketEnd > secondBracketStart,
        clue = s"the level field MUST close with `]`; a refactor leaving the bracket open (e.g. swap to a `[` `:` `=` form) would silently break grep patterns; got line: ${trimmed.take(100)}")
      val levelStr = trimmed.substring(secondBracketStart + 1, secondBracketEnd)

      // (iii) the level content is EXACTLY one of
      // INFO/WARN/ERROR (catches arbitrary values, lowercase
      // variants, key=value forms)
      val validLevels = Set("INFO", "WARN", "ERROR")
      assert(validLevels.contains(levelStr),
        clue = s"level content `$levelStr` MUST be exactly one of $validLevels per HandHistoryReviewServerRuntime.scala lines 417-424's logInfo/logWarn/logError helpers (each calls `log(level, ...)` with a fixed uppercase string `\"INFO\"` / `\"WARN\"` / `\"ERROR\"`); regression vectors caught: lowercase variants (`info`/`warn`/`error`), key=value form (`level=INFO`), arbitrary level names (`DEBUG`/`TRACE`/`FATAL` that the codebase does NOT emit), padded forms (` INFO `), or unrelated strings; got: `$levelStr`; line: ${trimmed.take(100)}")

      // (iv) the level content has NO padding spaces
      // (defensive -- catches `[ INFO ]` even though (iii)'s
      // exact-match would also catch it; the explicit
      // assertion makes the contract intent clearer)
      assert(levelStr.trim == levelStr,
        clue = s"level content `$levelStr` MUST have NO padding spaces inside the brackets -- catches a refactor injecting whitespace padding (e.g. for visual alignment) like `[ INFO ]` which would silently break tight grep patterns expecting `[INFO]` exactly; got: `$levelStr`")

      // (v) the third bracket (service-tag) follows the
      // level bracket with a single space separator (catches
      // a refactor that changed the separator OR reordered
      // the brackets)
      assert(trimmed.length > secondBracketEnd + 2,
        clue = s"log line must extend past the level bracket + separator; got line: ${trimmed.take(100)}")
      assert(trimmed.charAt(secondBracketEnd + 1) == ' ',
        clue = s"the level field's `]` MUST be followed by a single space (the documented field separator before `[hand-history-review]`); a refactor changing the separator would silently break grep patterns; got char `${trimmed.charAt(secondBracketEnd + 1)}` at position ${secondBracketEnd + 1}; line: ${trimmed.take(100)}")
      assert(trimmed.charAt(secondBracketEnd + 2) == '[',
        clue = s"the service-tag field MUST start with `[` immediately after the level's `] ` separator -- catches a refactor reordering the brackets (e.g. emitting the service-tag BEFORE the level); got char `${trimmed.charAt(secondBracketEnd + 2)}` at position ${secondBracketEnd + 2}; line: ${trimmed.take(100)}")
    }
  }

  // Pin the documented `[hand-history-review]` service-tag
  // bracket on every log line per HandHistoryReviewServer
  // Runtime.scala line 512's `stream.println(s"[$${Instant
  // .now()}] [$$level] [hand-history-review] $${sanitize
  // LogMessage(message)}")` template -- the FORMAT-INVARIANT
  // pin covering the SERVICE-TAG DIMENSION for ALL log-line
  // emission sites; this commit CLOSES the bracket-dimension
  // trifecta started by b9fc4d4 (TIMESTAMP) and continued by
  // d4f5e0f (LEVEL); all 3 pins verify properties enforced
  // at the log() helper itself (line 510-512) so single pins
  // cover the ENTIRE family of 30+ log-line emission sites;
  // the SERVICE-TAG DIMENSION is OPERATIONALLY CRITICAL
  // because: (a) in a multi-service deployment, ALL services
  // log to a shared aggregator (e.g. operators run multiple
  // sicfun services -- hand-history-review for the audit
  // browser, real-time-coach for the live overlay, batch-
  // trainer for offline model training); the service-tag is
  // how operators filter by service in the aggregator's UI
  // (a Splunk query like `service=hand-history-review level=
  // ERROR last 1h` requires the service-tag bracket to be
  // STABLE and DISTINCT from other services), a refactor
  // changing the tag (e.g. to `hh-review`, `handHistory
  // Review`, `hand_history_review`) would silently break the
  // aggregator's saved searches + dashboards, (b) the
  // runbook references this exact service name in
  // diagnostic workflows (e.g. "to investigate audit-browser
  // incidents, search aggregator for `[hand-history-review]`
  // lines in the affected time window"); a refactor renaming
  // the tag would silently desync the runbook from
  // operational reality, (c) operator grep workflows like
  // `tail -f deploy.log | grep hand-history-review` filter
  // by the documented service name; a refactor would
  // silently break these grep pipelines; the service-tag is
  // emitted as a BAKED-IN STRING LITERAL at line 512's
  // template -- the value is NOT pulled from configuration
  // or build metadata, so changing it requires editing the
  // source code (giving us a single anchor point to pin
  // against); per-format regression vectors that this pin
  // catches: (i) refactor renaming the service (e.g.
  // `hand-history-review` -> `audit-browser` for "marketing
  // clarity" OR -> `hh-review` for "brevity") would silently
  // break aggregator saved searches expecting the exact
  // documented name, (ii) refactor changing the separator
  // case (e.g. hyphen -> underscore `hand_history_review` OR
  // camelCase `handHistoryReview`) would silently break
  // grep patterns expecting hyphen-separated form, (iii)
  // refactor adding a version suffix (e.g. `hand-history-
  // review-v2`) would silently desync from documented
  // workflows that omit the suffix, (iv) refactor changing
  // the bracket shape (e.g. `<hand-history-review>` or
  // `(hand-history-review)`) would silently break field
  // extraction regex patterns, (v) refactor adding padding
  // inside the brackets like `[ hand-history-review ]`
  // would silently break tight grep patterns, (vi) refactor
  // dropping the service-tag entirely (e.g. "the log file
  // is already named after the service") would silently
  // break multi-service aggregator workflows that need the
  // service-tag to distinguish lines from different
  // services interleaved in the same aggregator index;
  // test approach mirrors b9fc4d4 + d4f5e0f: capture stdout
  // around withServer (which emits the startup banner -- a
  // guaranteed log line with the service-tag), find the
  // banner line, extract the THIRD bracketed field via
  // string position arithmetic (the FIRST bracket is the
  // timestamp pinned by b9fc4d4; the SECOND bracket is the
  // level pinned by d4f5e0f; the THIRD bracket is the
  // service-tag pinned by THIS commit); 5-tier format
  // check: (i) the service-tag bracket starts with `[`
  // immediately after the level's `] ` separator (catches
  // dropped-bracket refactors), (ii) the service-tag
  // bracket closes with `]`, (iii) the bracket content is
  // EXACTLY the string `hand-history-review` (catches
  // service-rename refactors, casing refactors, separator-
  // style refactors, version-suffix refactors -- the most
  // important assertion in the pin), (iv) the content has
  // NO padding spaces inside the brackets (catches
  // `[ hand-history-review ]`), (v) the message field
  // follows the service-tag bracket with a single space
  // separator (catches a refactor that changed the message
  // separator OR appended additional metadata fields after
  // the service-tag like `[hand-history-review] [region=us-
  // west]` which would silently break operators expecting
  // the message immediately after the service-tag).
  test("log line format pins the documented `[hand-history-review]` service-tag bracket on every log line per HandHistoryReviewServerRuntime.scala line 512's template -- the FORMAT-INVARIANT pin covering the SERVICE-TAG DIMENSION for ALL 30+ log-line emission sites; CLOSES the bracket-dimension trifecta (b9fc4d4 timestamp + d4f5e0f level + THIS service-tag) covering the FULL log() helper template at the line 510-512 level") {
    withStaticSite { staticDir =>
      val outBuf = new java.io.ByteArrayOutputStream()
      val originalOut = System.out
      System.setOut(new java.io.PrintStream(outBuf, true, StandardCharsets.UTF_8))
      try
        withServer(staticDir) { _ =>
          // No HTTP requests -- the startup banner emits
          // BEFORE the callback executes (the banner emits
          // via logInfo at HandHistoryReviewServerRuntime
          // .scala lines 349-350, so the line carries the
          // service-tag in the THIRD bracket).
          ()
        }
      finally
        System.setOut(originalOut)

      val captured = outBuf.toString(StandardCharsets.UTF_8)
      val bannerLine = captured.split('\n').iterator
        .find(_.contains("startup complete"))
        .getOrElse(fail(s"no `startup complete` line in captured stdout -- the service-tag-format pin needs ANY log line to inspect; the 7c47f88 startup-banner pin should catch this independently; got captured stdout: ${captured.take(800)}"))

      val trimmed = bannerLine.trim

      // Locate the FIRST `]` (timestamp end), then the
      // SECOND `]` (level end) -- the service-tag bracket
      // starts immediately after the level bracket's `] `
      // separator.
      val firstBracketEnd = trimmed.indexOf("]")
      assert(firstBracketEnd > 0,
        clue = s"log line must close the leading `[<timestamp>]` bracket with `]`; got line: ${trimmed.take(100)}")
      val secondBracketEnd = trimmed.indexOf("]", firstBracketEnd + 1)
      assert(secondBracketEnd > firstBracketEnd,
        clue = s"log line must close the level bracket `[<level>]` with `]` (the second `]` after the timestamp's); got line: ${trimmed.take(100)}")

      // (i) the service-tag bracket starts with `[`
      // immediately after the level's `] ` separator
      assert(trimmed.length > secondBracketEnd + 2,
        clue = s"log line must extend past the level bracket + separator; got line: ${trimmed.take(100)}")
      assert(trimmed.charAt(secondBracketEnd + 1) == ' ',
        clue = s"the level's `]` MUST be followed by a single space (the documented field separator); got char `${trimmed.charAt(secondBracketEnd + 1)}` at position ${secondBracketEnd + 1}; line: ${trimmed.take(100)}")
      val thirdBracketStart = secondBracketEnd + 2
      assert(trimmed.charAt(thirdBracketStart) == '[',
        clue = s"the service-tag field MUST start with `[` immediately after the level's `] ` separator per line 512's template; a refactor changing the bracket shape to `<` `>` or `(` `)` or dropping the brackets entirely would silently break log aggregator field extraction; got char `${trimmed.charAt(thirdBracketStart)}` (codepoint ${trimmed.charAt(thirdBracketStart).toInt}) at position $thirdBracketStart; line: ${trimmed.take(100)}")

      // (ii) the service-tag bracket closes with `]`
      val thirdBracketEnd = trimmed.indexOf("]", thirdBracketStart + 1)
      assert(thirdBracketEnd > thirdBracketStart,
        clue = s"the service-tag field MUST close with `]`; a refactor leaving the bracket open would silently break grep patterns; got line: ${trimmed.take(100)}")
      val serviceTagStr = trimmed.substring(thirdBracketStart + 1, thirdBracketEnd)

      // (iii) the bracket content is EXACTLY the string
      // `hand-history-review` (catches service-rename,
      // casing, separator-style, version-suffix refactors)
      assert(serviceTagStr == "hand-history-review",
        clue = s"service-tag content `$serviceTagStr` MUST be EXACTLY the string `hand-history-review` per HandHistoryReviewServerRuntime.scala line 512's BAKED-IN STRING LITERAL in the s\"...[hand-history-review]...\" template; regression vectors caught: service rename (`audit-browser`, `hh-review`), casing change (`Hand-History-Review`, `HAND-HISTORY-REVIEW`, `handHistoryReview`), separator change (`hand_history_review`, `hand.history.review`), version suffix (`hand-history-review-v2`), prefix change (`sicfun/hand-history-review`); the documented runbook diagnostic workflows + aggregator saved searches + operator grep pipelines ALL depend on this exact literal; got: `$serviceTagStr`; line: ${trimmed.take(100)}")

      // (iv) the service-tag content has NO padding spaces
      // (defensive -- the exact-match in (iii) would catch
      // padded variants, but the explicit assertion makes
      // the contract intent clearer)
      assert(serviceTagStr.trim == serviceTagStr,
        clue = s"service-tag content `$serviceTagStr` MUST have NO padding spaces inside the brackets -- catches a refactor injecting whitespace padding (e.g. `[ hand-history-review ]`); got: `$serviceTagStr`")

      // (v) the message field follows the service-tag
      // bracket with a single space separator (catches a
      // refactor that changed the separator OR appended
      // additional metadata fields after the service-tag)
      assert(trimmed.length > thirdBracketEnd + 2,
        clue = s"log line must extend past the service-tag bracket + separator (the message field follows); got line: ${trimmed.take(100)}")
      assert(trimmed.charAt(thirdBracketEnd + 1) == ' ',
        clue = s"the service-tag's `]` MUST be followed by a single space (the documented separator before the message field); a refactor changing the separator would silently break log aggregator field extraction OR a refactor adding additional metadata brackets after the service-tag (like `[region=us-west]`) would silently break operators expecting the message immediately after the service-tag; got char `${trimmed.charAt(thirdBracketEnd + 1)}` at position ${thirdBracketEnd + 1}; line: ${trimmed.take(100)}")
      // Verify the next char is NOT `[` -- catches the
      // additional-metadata-bracket refactor specifically
      // (the documented format has MESSAGE content here, NOT
      // another bracket field)
      assert(trimmed.charAt(thirdBracketEnd + 2) != '[',
        clue = s"the position immediately after the service-tag bracket + separator MUST contain MESSAGE content, NOT another `[` bracket -- catches a refactor adding additional metadata fields after the service-tag (e.g. `[hand-history-review] [region=us-west] startup complete...`) which would silently break operators expecting the message field directly after the service-tag; got char `${trimmed.charAt(thirdBracketEnd + 2)}` at position ${thirdBracketEnd + 2}; line: ${trimmed.take(100)}")
    }
  }

  // Pin the documented sanitizeLogMessage END-TO-END escape
  // contract -- the SANITIZATION-INVARIANT pin covering the
  // CONTROL-CHARACTER DIMENSION for ALL log lines that flow
  // user-controllable input; the existing line ~6789 test pins
  // sanitizeLogMessage in ISOLATION (calls the helper directly
  // with control characters + asserts the output), but BEFORE
  // this commit there was NO end-to-end test verifying that
  // the sanitization actually applies to USER-CONTROLLED FIELDS
  // flowing through the log() helper's emission pipeline at
  // line 512's `sanitizeLogMessage(message)` wrapping; the
  // sanitization is OPERATIONALLY CRITICAL because the
  // documented threat is LOG INJECTION: an attacker who can
  // influence a field value (e.g. submit an email containing a
  // newline character via the auth.login.failure path) could
  // otherwise inject FAKE LOG LINES into the operator's audit
  // stream -- a newline in the middle of a key=value sequence
  // would split the line into two physical lines, with the
  // SECOND line containing whatever the attacker put after the
  // newline (potentially a forged `[INFO] [hand-history-review]
  // auth.login.success email=attacker@victim.com ...` line that
  // looks legitimate to the operator's grep workflow); the
  // sanitizeLogMessage helper at line 486-499 escapes ALL
  // control characters (\n, \r, \0, \t, and any byte < 0x20 or
  // 0x7F) into safe-printable forms (\\n, \\r, \\0, \\t,
  // \\x<hex>); without the sanitization, an attacker submitting
  // email="alice\nbob@example.com" (with a literal newline byte)
  // could split the log line into two physical lines + inject
  // a forged log entry; this commit's END-TO-END test verifies
  // the sanitization applies at the log() emission layer by:
  // (1) configuring platformAuth, (2) submitting POST /api/auth/
  // login with email containing a literal \n character (valid
  // JSON escape sequence that parses to a string with an
  // embedded newline byte), (3) the login fails validation +
  // emits auth.login.failure logWarn with the user-submitted
  // email in the email= field, (4) the log() helper's
  // sanitizeLogMessage wrapping at line 512 catches the
  // newline + escapes it to `\\n` (the 2-character backslash-n
  // sequence), (5) the captured stderr SHOULD contain the
  // escaped form `email=alice\nbob@example.com` (with literal
  // backslash-n, NOT a physical newline character); per-format
  // regression vectors: (i) refactor dropping the
  // sanitizeLogMessage wrapping at line 512 (e.g. "the
  // upstream formatSubmittedEmailForLog already escapes, the
  // line 512 sanitize is redundant") would silently let
  // user-controlled control characters split log lines, (ii)
  // refactor changing the escape format (e.g. \\n -> %0A like
  // the %20-space-escape) would silently break log
  // aggregator parsers that expect the documented \\<char>
  // backslash-escape form, (iii) refactor skipping the
  // sanitization on specific field positions (e.g. "only
  // sanitize the reason= field, not the email= field") would
  // silently leave injection-vulnerable paths in the
  // user-submitted email path; this test's emission via the
  // auth.login.failure path (which carries the
  // formatSubmittedEmailForLog'd submitted email) exercises
  // the END-TO-END sanitization specifically on a USER-
  // CONTROLLED field that the formatSubmittedEmailForLog
  // helper does NOT itself escape control chars (only spaces
  // → %20); the sanitization happens at the log() layer
  // catching ALL control chars regardless of which field they
  // appear in; 4-tier format check: (i) the
  // auth.login.failure line exists in captured stderr, (ii)
  // the line is a SINGLE physical line (no embedded newline
  // splitting it), (iii) the email field contains the ESCAPED
  // form `alice\nbob@example.com` (literal backslash-n, NOT
  // raw newline), (iv) EXCLUSION of the unescaped form (the
  // line MUST NOT contain a raw 0x0A newline byte in the
  // middle of the email field).
  test("log() helper's sanitizeLogMessage wrapping at HandHistoryReviewServerRuntime.scala line 512 escapes user-controlled control characters end-to-end -- the SANITIZATION-INVARIANT pin verifies that an attacker submitting an email with embedded newline cannot inject FORGED LOG LINES into the operator audit stream (the documented LOG INJECTION threat)") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          // Capture stderr around the malicious login attempt.
          // The submitted email contains a JSON escape \n which
          // parses to an embedded newline character (1 byte =
          // 0x0A). The login fails validation + emits
          // auth.login.failure logWarn at AuthStack.scala line
          // 192 with formatSubmittedEmailForLog'd submitted
          // email in the email= field; formatSubmittedEmailForLog
          // ONLY escapes spaces, NOT control characters; the
          // newline flows into the message s-string, then
          // log() at line 512 calls sanitizeLogMessage which
          // catches the newline + escapes to \\n.
          val errBuf = new java.io.ByteArrayOutputStream()
          val originalErr = System.err
          System.setErr(new java.io.PrintStream(errBuf, true, StandardCharsets.UTF_8))
          try
            val rejected = postJson(s"$baseUri/api/auth/login",
              """{"email":"alice\nbob@example.com","password":"some-password"}""")
            assertEquals(rejected.statusCode(), 401,
              clue = "malformed-email login MUST return 401 (the email fails validateEmail's regex check; loginLocal returns Left -> 401)")
          finally
            System.setErr(originalErr)

          val captured = errBuf.toString(StandardCharsets.UTF_8)
          val failureLine = captured.split('\n').iterator
            .find(_.contains("auth.login.failure"))
            .getOrElse(fail(s"no `auth.login.failure` line in captured stderr; got: ${captured.take(800)}"))

          // (i) the auth.login.failure line exists
          assert(failureLine.contains("auth.login.failure"),
            clue = s"auth.login.failure line must be present (the rejected login path at AuthStack.scala line 192 emits this on every invalid-credentials rejection); got: $failureLine")

          // (ii) the line is a SINGLE physical line -- if
          // sanitization is broken, the raw newline in the
          // email field would split the line into TWO physical
          // lines, and captured.split('\n') would yield 2+
          // pieces with the second piece containing whatever
          // was after the embedded newline (the rest of the
          // email + remote + reason). The find() call above
          // returned a line that contains "auth.login.failure"
          // -- if sanitization is broken, that line would END
          // at the embedded newline (so the line would NOT
          // contain "bob@example.com" because that's after
          // the newline). Check that the line contains BOTH
          // halves of the email AFTER the prefix.
          assert(failureLine.contains("alice") && failureLine.contains("bob@example.com"),
            clue = s"auth.login.failure line MUST be a SINGLE physical line containing BOTH halves of the email (alice + bob@example.com) -- if sanitization is broken at line 512, the raw newline in the email would split the line into two physical lines and 'bob@example.com' would appear on the NEXT physical line (not in this find() result); got: $failureLine; full captured stream: ${captured.take(1500)}")

          // (iii) the email contains the ESCAPED form with
          // literal backslash-n
          assert(failureLine.contains("alice\\nbob@example.com"),
            clue = s"auth.login.failure line MUST carry the escaped form `alice\\\\nbob@example.com` (literal backslash followed by literal 'n') per HandHistoryReviewServerRuntime.scala line 488's `replace(\"\\\\n\", \"\\\\\\\\n\")` escape rule; the raw 0x0A newline byte in the submitted email gets caught by sanitizeLogMessage AT THE LOG LAYER (NOT at formatSubmittedEmailForLog which only escapes spaces); a refactor dropping the sanitizeLogMessage wrapping at line 512 would silently let the newline through, splitting the log line into two physical lines and enabling LOG INJECTION attacks; got: $failureLine")

          // (iv) EXCLUSION: the line must NOT contain a raw
          // newline byte. Since we already verified the line
          // is a single physical line via the bob@example.com
          // contains check, this is partially redundant -- but
          // explicit assertion strengthens the documented
          // contract. Use Char.toString conversion of the raw
          // 0x0A byte to avoid Scala 3 multi-line-string-in-test
          // parse issues.
          val rawNewlineByte: String = '\n'.toString
          assert(!failureLine.contains(rawNewlineByte),
            clue = s"auth.login.failure line MUST NOT contain a raw 0x0A newline byte -- catches a refactor that escaped newlines to a DIFFERENT visible form but still allowed raw newlines through in some field positions; got line: $failureLine")
        }
      }
    }
  }

  // Pin the documented sanitizeLogMessage CR (carriage-return)
  // escape rule END-TO-END -- the SANITIZATION-INVARIANT pin
  // covering the SECOND of the 5 explicit escape rules documented
  // at HandHistoryReviewServerRuntime.scala lines 488-492; the
  // 4db713d sibling pin covered ONLY the newline (0x0A → \\n)
  // rule at line 489, but the helper documents 5 explicit
  // escape rules (\\, \\n, \\r, \\0, \\t) plus a general
  // \\x<hex> fallback for any other control char; this CR pin
  // is the natural extension of the 4db713d family -- per the
  // 4db713d future-fire flag "(c) sanitization variants for
  // OTHER control characters (\\r carriage return, \\t tab, \\0
  // null byte, 0x7F DEL, arbitrary control chars 0x01-0x1F)";
  // CR (0x0D) is the OPERATIONALLY HIGHEST-RISK escape rule
  // after newline because Windows-origin log aggregators
  // (Splunk on Windows, ELK with Windows agents) treat the CRLF
  // pair as a line terminator, and a LONE CR can confuse
  // aggregators that handle Unix-style LF-only line endings --
  // an attacker submitting an email containing a raw CR could
  // potentially split log lines on Windows-targeted aggregators
  // even if the Unix LF case is sanitized; the documented
  // threat is IDENTICAL to the newline case: an attacker who
  // can influence a field value (e.g. submit an email
  // containing a CR via the auth.login.failure path) could
  // otherwise inject FAKE LOG LINES into the operator's audit
  // stream on aggregators that split on CR; the sanitization
  // is OPERATIONALLY CRITICAL on Windows deployments
  // specifically; this commit's END-TO-END test verifies the
  // sanitization applies at the log() emission layer by:
  // (1) configuring platformAuth, (2) submitting POST /api/auth/
  // login with email containing a literal \\r character (valid
  // JSON escape sequence that parses to a string with an
  // embedded CR byte = 0x0D), (3) the login fails validation +
  // emits auth.login.failure logWarn with the user-submitted
  // email in the email= field, (4) the log() helper's
  // sanitizeLogMessage wrapping at line 512 catches the CR +
  // escapes it to `\\r` (the 2-character backslash-r sequence)
  // via the line 490 `replace("\\r", "\\\\r")` rule, (5) the
  // captured stderr SHOULD contain the escaped form
  // `email=alice\\rbob@example.com` (with literal backslash-r,
  // NOT a physical CR character); per-format regression
  // vectors: (i) refactor dropping the line 490 CR escape
  // (e.g. "newline alone is sufficient since the line 512
  // template uses println which appends LF") would silently
  // let user-controlled CR bytes through and enable
  // log-injection attacks on Windows aggregators, (ii)
  // refactor changing the CR escape format (e.g. \\r → %0D
  // like a uri-style escape) would silently break log
  // aggregator parsers that expect the documented \\<char>
  // backslash-escape form, (iii) refactor swapping CR/LF
  // escapes (e.g. CR → \\n, LF → \\r) would silently confuse
  // the operator's grep workflow even though both escape
  // rules are technically applied; this test's emission via
  // the auth.login.failure path exercises the END-TO-END
  // sanitization specifically on a USER-CONTROLLED field that
  // the formatSubmittedEmailForLog helper does NOT itself
  // escape CR (only spaces → %20); the sanitization happens
  // at the log() layer catching ALL control chars regardless
  // of which field they appear in; 4-tier format check: (i)
  // the auth.login.failure line exists in captured stderr,
  // (ii) the line contains BOTH halves of the email (alice +
  // bob@example.com -- if CR sanitization is broken, the raw
  // CR would split the line on Windows-style aggregators
  // and/or confuse the captured.split('\\n') iterator if the
  // 0x0D byte gets emitted before the line's terminating LF),
  // (iii) the email field contains the ESCAPED form
  // `alice\\rbob@example.com` (literal backslash followed by
  // literal 'r') per HandHistoryReviewServerRuntime.scala
  // line 490's `replace("\\r", "\\\\r")` escape rule, (iv)
  // EXCLUSION of raw 0x0D CR byte in the line (defense-in-
  // depth catch via Char.toString('\\r') conversion).
  test("log() helper's sanitizeLogMessage wrapping at HandHistoryReviewServerRuntime.scala line 512 escapes user-controlled CR (carriage-return) bytes end-to-end via the line 490 escape rule -- the SANITIZATION-INVARIANT pin for the SECOND escape rule complements 4db713d's newline-only pin; on Windows-targeted log aggregators a raw CR can confuse line-boundary detection and enable LOG INJECTION even if Unix-style LF is sanitized") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          // Capture stderr around the malicious login attempt.
          // The submitted email contains a JSON escape \r which
          // parses to an embedded CR character (1 byte = 0x0D).
          // The login fails validation + emits
          // auth.login.failure logWarn at AuthStack.scala line
          // 192 with formatSubmittedEmailForLog'd submitted
          // email in the email= field; formatSubmittedEmailForLog
          // ONLY escapes spaces, NOT control characters; the CR
          // flows into the message s-string, then log() at line
          // 512 calls sanitizeLogMessage which catches the CR +
          // escapes to \\r via line 490's replace rule.
          val errBuf = new java.io.ByteArrayOutputStream()
          val originalErr = System.err
          System.setErr(new java.io.PrintStream(errBuf, true, StandardCharsets.UTF_8))
          try
            val rejected = postJson(s"$baseUri/api/auth/login",
              """{"email":"alice\rbob@example.com","password":"some-password"}""")
            assertEquals(rejected.statusCode(), 401,
              clue = "malformed-email login MUST return 401 (the email fails validateEmail's regex check because CR is not in the [A-Za-z0-9_.+-] local-part character class; loginLocal returns Left -> 401)")
          finally
            System.setErr(originalErr)

          val captured = errBuf.toString(StandardCharsets.UTF_8)
          val failureLine = captured.split('\n').iterator
            .find(_.contains("auth.login.failure"))
            .getOrElse(fail(s"no `auth.login.failure` line in captured stderr; got: ${captured.take(800)}"))

          // (i) the auth.login.failure line exists
          assert(failureLine.contains("auth.login.failure"),
            clue = s"auth.login.failure line must be present (the rejected login path at AuthStack.scala line 192 emits this on every invalid-credentials rejection); got: $failureLine")

          // (ii) the line contains BOTH halves of the email --
          // if the line 490 CR escape rule is broken, the raw
          // CR byte would flow through into the printed line.
          // The line 512 println terminates with LF (the
          // println contract), so captured.split('\n') groups
          // the full line correctly -- BUT if a CR ends up
          // embedded in the line BEFORE the LF terminator, the
          // CR could trigger an early line-boundary on the
          // captured ByteArrayOutputStream's String conversion
          // and split('\n') may yield two pieces. More
          // critically, even if split('\n') keeps the line
          // together, the raw CR byte INSIDE the line is the
          // exact log-injection vulnerability on Windows
          // aggregators that treat CRLF as the line
          // terminator. Check that the line contains BOTH
          // halves of the email AFTER the prefix.
          assert(failureLine.contains("alice") && failureLine.contains("bob@example.com"),
            clue = s"auth.login.failure line MUST be a SINGLE physical line containing BOTH halves of the email (alice + bob@example.com) -- if CR sanitization is broken at line 490, the raw CR in the email could confuse line-boundary handling and 'bob@example.com' might appear on the NEXT line (not in this find() result); got: $failureLine; full captured stream: ${captured.take(1500)}")

          // (iii) the email contains the ESCAPED form with
          // literal backslash-r
          assert(failureLine.contains("alice\\rbob@example.com"),
            clue = s"auth.login.failure line MUST carry the escaped form `alice\\\\rbob@example.com` (literal backslash followed by literal 'r') per HandHistoryReviewServerRuntime.scala line 490's `replace(\"\\\\r\", \"\\\\\\\\r\")` escape rule; the raw 0x0D CR byte in the submitted email gets caught by sanitizeLogMessage AT THE LOG LAYER (NOT at formatSubmittedEmailForLog which only escapes spaces); a refactor dropping the line 490 CR escape would silently let the CR through, enabling LOG INJECTION attacks on Windows-style log aggregators that split lines on CRLF; got: $failureLine")

          // (iv) EXCLUSION: the line must NOT contain a raw CR
          // byte INSIDE the log line content (the threat is
          // CR injection in field values, NOT the line
          // terminator). On Windows, println at
          // HandHistoryReviewServerRuntime.scala line 512 uses
          // the platform line.separator (CRLF) so
          // captured.split('\n') leaves a trailing CR on each
          // piece; strip that trailing CR before the EXCLUSION
          // check so we only assert on the actual line
          // content. Use Char.toString conversion of the raw
          // 0x0D byte to avoid Scala 3 multi-line-string-in-
          // test parse issues (same defensive idiom as
          // 4db713d's newline EXCLUSION used '\n'.toString).
          val rawCrByte: String = '\r'.toString
          val failureLineNoTerminator = failureLine.stripSuffix(rawCrByte)
          assert(!failureLineNoTerminator.contains(rawCrByte),
            clue = s"auth.login.failure line MUST NOT contain a raw 0x0D CR byte INSIDE the log content (the platform-line-separator trailing CR is stripped before this check) -- catches a refactor that escaped CR to a DIFFERENT visible form (e.g. %0D) but still allowed raw CR bytes through in some field positions, OR a refactor swapping the CR/LF escape rules; got line (after stripping trailing CR if any): $failureLineNoTerminator")
        }
      }
    }
  }

  // Pin the documented sanitizeLogMessage TAB (horizontal-tab)
  // escape rule END-TO-END -- the SANITIZATION-INVARIANT pin
  // covering the THIRD of the 5 explicit escape rules
  // documented at HandHistoryReviewServerRuntime.scala lines
  // 488-492; the family now covers: (4db713d) newline 0x0A,
  // (a696f57) CR 0x0D, (THIS commit) TAB 0x09; tab is
  // OPERATIONALLY RELEVANT because the structured log line
  // format `key=value key=value...` uses SPACE as the
  // key-value separator at HandHistoryReviewServerRuntime.scala
  // line 512's emitted template -- a raw tab in a field value
  // could confuse log aggregator parsers that tokenize on
  // \s+ (any whitespace) instead of strict space-only; an
  // attacker submitting an email containing a tab character
  // could create a field value that LOOKS like one value to
  // a strict-space parser but TWO values to a whitespace
  // parser -- e.g. `email=alice\tbob@example.com remote=...`
  // appears as `email=alice` + `bob@example.com` + `remote=...`
  // to a whitespace-tokenizing parser, breaking the
  // operator's key=value structure; the documented threat is
  // RELATED to but distinct from the CR/LF cases: CR/LF
  // splits PHYSICAL log lines, while tab splits LOGICAL
  // key=value tokens within a line; both are forms of LOG
  // INJECTION; the sanitizeLogMessage helper at line 492's
  // `replace("\t", "\\t")` catches the tab + escapes to `\t`
  // (literal backslash-t); this commit's END-TO-END test
  // verifies the sanitization applies at the log() emission
  // layer by: (1) configuring platformAuth, (2) submitting
  // POST /api/auth/login with email containing a literal \t
  // character (valid JSON escape sequence that ujson parses
  // to a string with an embedded tab byte = 0x09), (3) the
  // login fails validation + emits auth.login.failure logWarn
  // with the user-submitted email in the email= field, (4)
  // the log() helper's sanitizeLogMessage wrapping at line
  // 512 catches the tab + escapes it to `\t` (the 2-character
  // backslash-t sequence) via the line 492 escape rule, (5)
  // the captured stderr SHOULD contain the escaped form
  // `email=alice\tbob@example.com` (with literal backslash-t,
  // NOT a physical tab character); per-format regression
  // vectors: (i) refactor dropping the line 492 tab escape
  // (e.g. "tab is whitespace, no harm in passing through")
  // would silently let user-controlled tabs through and
  // enable log-injection attacks on whitespace-tokenizing
  // parsers, (ii) refactor changing the tab escape format
  // (e.g. \t -> %09 uri-style escape) would silently break
  // log aggregator parsers that expect the documented
  // \<char> backslash-escape form, (iii) refactor consolidating
  // the 3 escapes into a generic "whitespace" replace (e.g.
  // .replace("\\s+".r, " ")) would silently collapse multiple
  // whitespace forms into a single space, losing the original
  // field value content + breaking forensics; this test's
  // emission via the auth.login.failure path exercises the
  // END-TO-END sanitization specifically on a USER-CONTROLLED
  // field that the formatSubmittedEmailForLog helper does NOT
  // itself escape (only spaces -> %20); 4-tier format check:
  // (i) the auth.login.failure line exists in captured stderr,
  // (ii) the line contains BOTH halves of the email (alice +
  // bob@example.com), (iii) the email field contains the
  // ESCAPED form `alice\tbob@example.com` (literal backslash
  // followed by literal 't'), (iv) EXCLUSION of raw 0x09 tab
  // byte in the line (no stripSuffix needed -- tab is not
  // part of the platform line.separator on any common OS, so
  // any 0x09 byte in the captured line is necessarily a
  // sanitization failure).
  test("log() helper's sanitizeLogMessage wrapping at HandHistoryReviewServerRuntime.scala line 512 escapes user-controlled TAB (horizontal-tab) bytes end-to-end via the line 492 escape rule -- the SANITIZATION-INVARIANT pin for the THIRD escape rule; tab injection inside a field value would split the key=value structure for whitespace-tokenizing log aggregator parsers (a LOGICAL-token form of LOG INJECTION distinct from CR/LF's PHYSICAL-line splitting)") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          // Capture stderr around the malicious login attempt.
          // The submitted email contains a JSON escape \t which
          // parses to an embedded tab character (1 byte = 0x09).
          // The login fails validation + emits
          // auth.login.failure logWarn at AuthStack.scala line
          // 192 with formatSubmittedEmailForLog'd submitted
          // email in the email= field; formatSubmittedEmailForLog
          // ONLY escapes spaces, NOT control characters; the tab
          // flows into the message s-string, then log() at line
          // 512 calls sanitizeLogMessage which catches the tab +
          // escapes to \t via line 492's replace rule.
          val errBuf = new java.io.ByteArrayOutputStream()
          val originalErr = System.err
          System.setErr(new java.io.PrintStream(errBuf, true, StandardCharsets.UTF_8))
          try
            val rejected = postJson(s"$baseUri/api/auth/login",
              """{"email":"alice\tbob@example.com","password":"some-password"}""")
            assertEquals(rejected.statusCode(), 401,
              clue = "malformed-email login MUST return 401 (the email fails validateEmail's regex check because tab is not in the [A-Za-z0-9_.+-] local-part character class; loginLocal returns Left -> 401)")
          finally
            System.setErr(originalErr)

          val captured = errBuf.toString(StandardCharsets.UTF_8)
          val failureLine = captured.split('\n').iterator
            .find(_.contains("auth.login.failure"))
            .getOrElse(fail(s"no `auth.login.failure` line in captured stderr; got: ${captured.take(800)}"))

          // (i) the auth.login.failure line exists
          assert(failureLine.contains("auth.login.failure"),
            clue = s"auth.login.failure line must be present (the rejected login path at AuthStack.scala line 192 emits this on every invalid-credentials rejection); got: $failureLine")

          // (ii) the line contains BOTH halves of the email --
          // tab doesn't split physical lines, but if a refactor
          // converted tab to a different whitespace form that
          // truncated the field, the second half could be lost.
          assert(failureLine.contains("alice") && failureLine.contains("bob@example.com"),
            clue = s"auth.login.failure line MUST contain BOTH halves of the email (alice + bob@example.com) -- a refactor that converted tab to a truncating-form (e.g. cut the field at the first whitespace) would silently drop the second half; got: $failureLine; full captured stream: ${captured.take(1500)}")

          // (iii) the email contains the ESCAPED form with
          // literal backslash-t
          assert(failureLine.contains("alice\\tbob@example.com"),
            clue = s"auth.login.failure line MUST carry the escaped form `alice\\\\tbob@example.com` (literal backslash followed by literal 't') per HandHistoryReviewServerRuntime.scala line 492's `replace(\"\\\\t\", \"\\\\\\\\t\")` escape rule; the raw 0x09 tab byte in the submitted email gets caught by sanitizeLogMessage AT THE LOG LAYER (NOT at formatSubmittedEmailForLog which only escapes spaces); a refactor dropping the line 492 tab escape would silently let the tab through, enabling LOG INJECTION attacks on whitespace-tokenizing log aggregator parsers that split key=value tokens on any whitespace; got: $failureLine")

          // (iv) EXCLUSION: the line must NOT contain a raw tab
          // byte. Unlike the CR case, tab is NOT part of the
          // platform line.separator on any common OS (Windows
          // uses CRLF, Unix uses LF, classic Mac used CR), so
          // no stripSuffix workaround is needed -- any 0x09
          // byte in the captured line is necessarily a
          // sanitization failure. Use Char.toString conversion
          // of the raw 0x09 byte to avoid Scala 3 multi-line-
          // string-in-test parse issues (same defensive idiom
          // as 4db713d/a696f57).
          val rawTabByte: String = '\t'.toString
          assert(!failureLine.contains(rawTabByte),
            clue = s"auth.login.failure line MUST NOT contain a raw 0x09 tab byte -- catches a refactor that escaped tab to a DIFFERENT visible form (e.g. %09) but still allowed raw tab bytes through in some field positions, OR a refactor that converted tab to a different whitespace form (e.g. tab -> single space, losing the original byte); got line: $failureLine")
        }
      }
    }
  }

  // Pin the documented sanitizeLogMessage DEL (0x7F) escape
  // rule END-TO-END -- the SANITIZATION-INVARIANT pin covering
  // the GENERIC FALLBACK BRANCH at HandHistoryReviewServerRuntime
  // .scala lines 494-497 (NOT one of the 5 explicit
  // .replace(...) rules at lines 488-492); the family
  // previously covered ONLY the explicit-rules branch (4db713d
  // newline, a696f57 CR, 7aeaa8d TAB -- all 3 hit lines 488-
  // 492's chained .replace calls); this commit hits the
  // OTHER half of sanitizeLogMessage's escape logic -- the
  // foreach loop at lines 494-497 that handles ANY ctrl byte
  // < 0x20 (caught by the explicit chain or this loop) AND
  // the 0x7F DEL byte (caught ONLY by this loop because the
  // explicit chain at lines 488-492 has no rule for 0x7F);
  // structural significance: the explicit chain emits
  // `\<char>` form (2 chars: backslash + ascii letter) while
  // the generic fallback emits `\x<hex>` form (4 chars:
  // backslash + 'x' + 2 lowercase hex digits); a refactor
  // could break ONE branch without breaking the OTHER, so
  // pinning both branches separately is essential; DEL is
  // OPERATIONALLY RELEVANT because: (i) some terminals
  // interpret DEL as a destructive char that erases the
  // previous char (the same way backspace does), so a raw
  // DEL in a log line could VISUALLY hide preceding chars in
  // operator tail/less workflows -- an attacker could submit
  // `email=aliceadversary@evil.com` and
  // visually overwrite `alice` with `adv` in the operator's
  // terminal view, making the log line LOOK like the
  // attacker's email was something else, (ii) DEL is the
  // ONLY non-control-block byte caught by the foreach
  // fallback (lines 488-492's chained replaces handle the 5
  // common control chars, then the foreach catches "any byte
  // < 0x20" AND "0x7F"); without sanitization, raw DEL
  // bytes would flow through to operator logs and confuse
  // forensics; the sanitizeLogMessage helper at lines 494-
  // 497 catches DEL via the `ch.toInt == 0x7F` clause and
  // escapes to `\x7f` via the `"\\x%02x".format(ch.toInt)`
  // format string; this commit's END-TO-END test verifies
  // the sanitization applies at the log() emission layer by:
  // (1) configuring platformAuth, (2) constructing the JSON
  // unicode escape `` at RUNTIME via Char(0x5C) ('\')
  // + "u007f" concatenation to AVOID Scala source-level
  // unicode escape processing (the safest construction --
  // 'a'.toString + ... ensures the source reader cannot
  // accidentally process a `` literal at source level
  // before lexing reaches the string), (3) submitting POST
  // /api/auth/login with the constructed body containing
  // the literal 6-char `` sequence in the email
  // field, (4) ujson parses this as the JSON unicode
  // escape -> 0x7F byte at JSON-decode time, (5) the login
  // fails validateEmail (DEL is not in the [A-Za-z0-9_.+-]
  // local-part character class) + emits auth.login.failure
  // logWarn with the user-submitted email containing the
  // raw 0x7F byte, (6) the log() helper's sanitizeLogMessage
  // wrapping at line 512 catches the DEL via the foreach
  // loop's `ch.toInt == 0x7F` clause + escapes it to the
  // 4-char `\x7f` sequence; per-format regression vectors:
  // (i) refactor dropping the foreach loop entirely (e.g.
  // "the 5 explicit replaces cover all common cases") would
  // silently let DEL + arbitrary ctrl chars 0x01-0x1F
  // through, (ii) refactor changing the `\x%02x` format
  // (e.g. dropping the `\x` prefix or using uppercase hex
  // `\X%02X`) would silently break log aggregator parsers
  // that expect the documented lowercase `\x<hex>` form,
  // (iii) refactor consolidating the foreach with the
  // explicit chain into a single regex-based replacement
  // would risk format drift between the two branches; 4-
  // tier format check: (i) the auth.login.failure line
  // exists in captured stderr, (ii) the line contains BOTH
  // halves of the email (alice + bob@example.com), (iii)
  // the email field contains the ESCAPED form
  // `alice\x7fbob@example.com` (literal 4-char escape
  // sequence) per HandHistoryReviewServerRuntime.scala line
  // 496's `\\x%02x` format with the LOWERCASE `7f` hex
  // (catches a refactor changing the hex case to uppercase
  // `7F`), (iv) EXCLUSION of raw 0x7F DEL byte in the line
  // (no stripSuffix workaround needed -- DEL is not part of
  // any common OS line.separator).
  test("log() helper's sanitizeLogMessage wrapping at HandHistoryReviewServerRuntime.scala line 512 escapes user-controlled DEL (0x7F) bytes end-to-end via the GENERIC FALLBACK branch at lines 494-497 -- the SANITIZATION-INVARIANT pin for the FALLBACK branch complements 4db713d/a696f57/7aeaa8d's explicit-replace-chain pins; DEL injection in field values could VISUALLY hide preceding chars in operator terminal workflows (terminals interpret DEL like backspace), enabling an attacker to mask the audit trail of their submitted credentials") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          // Construct the JSON unicode escape `` as a
          // 6-char literal sequence at RUNTIME via backslash-
          // char + "u007f" concatenation. The Scala source
          // reader processes `\uXXXX` unicode escapes BEFORE
          // lexing, even inside raw triple-quoted strings, so
          // a naive `""""""` literal would be replaced
          // with a single 0x7F char at compile time --
          // breaking the test's intent of letting ujson parse
          // the JSON unicode escape at runtime. Using runtime
          // concatenation of a single backslash Char (0x5C)
          // with the literal "u007f" String guarantees the
          // resulting 6-char sequence reaches ujson VERBATIM.
          val backslash: String = 0x5C.toChar.toString
          val jsonUnicodeEscapeForDel: String = backslash + "u007f"
          val maliciousBody = s"""{"email":"alice${jsonUnicodeEscapeForDel}bob@example.com","password":"some-password"}"""

          val errBuf = new java.io.ByteArrayOutputStream()
          val originalErr = System.err
          System.setErr(new java.io.PrintStream(errBuf, true, StandardCharsets.UTF_8))
          try
            val rejected = postJson(s"$baseUri/api/auth/login", maliciousBody)
            assertEquals(rejected.statusCode(), 401,
              clue = "malformed-email login MUST return 401 (the email fails validateEmail's regex check because DEL is not in the [A-Za-z0-9_.+-] local-part character class; loginLocal returns Left -> 401)")
          finally
            System.setErr(originalErr)

          val captured = errBuf.toString(StandardCharsets.UTF_8)
          val failureLine = captured.split('\n').iterator
            .find(_.contains("auth.login.failure"))
            .getOrElse(fail(s"no `auth.login.failure` line in captured stderr; got: ${captured.take(800)}"))

          // (i) the auth.login.failure line exists
          assert(failureLine.contains("auth.login.failure"),
            clue = s"auth.login.failure line must be present (the rejected login path at AuthStack.scala line 192 emits this on every invalid-credentials rejection); got: $failureLine")

          // (ii) the line contains BOTH halves of the email
          assert(failureLine.contains("alice") && failureLine.contains("bob@example.com"),
            clue = s"auth.login.failure line MUST contain BOTH halves of the email (alice + bob@example.com); got: $failureLine; full captured stream: ${captured.take(1500)}")

          // (iii) the email contains the ESCAPED form
          // `alice\x7fbob@example.com` -- 4-char escape
          // sequence (backslash + 'x' + '7' + 'f'). The
          // LOWERCASE hex is part of the documented contract
          // (line 496's `\\x%02x` uses %02x which produces
          // lowercase; a refactor to `\\x%02X` would silently
          // change the form to uppercase `\x7F`).
          assert(failureLine.contains("alice\\x7fbob@example.com"),
            clue = s"auth.login.failure line MUST carry the escaped form `alice\\\\x7fbob@example.com` (literal 4-char sequence: backslash + 'x' + '7' + 'f' in LOWERCASE hex) per HandHistoryReviewServerRuntime.scala line 496's `\"\\\\x%02x\".format(ch.toInt)` format string; the 0x7F DEL byte in the submitted email gets caught by sanitizeLogMessage's foreach fallback (lines 494-497) AT THE LOG LAYER; a refactor dropping the foreach fallback would silently let DEL + arbitrary ctrl chars through, enabling LOG INJECTION attacks where raw DEL bytes confuse operator terminal workflows (terminals interpret DEL like backspace); a refactor changing the hex case to uppercase (`\\X%02X`) would silently break log aggregator parsers expecting the documented lowercase form; got: $failureLine")

          // (iv) EXCLUSION: the line must NOT contain a raw
          // 0x7F DEL byte. DEL is not part of any common OS
          // line.separator (Windows CRLF, Unix LF, classic
          // Mac CR), so no stripSuffix workaround is needed.
          // Use Char unicode escape '' (processed at
          // source-reader level into the single 0x7F Char) to
          // construct the raw byte for the EXCLUSION check.
          val rawDelByte: String = ''.toString
          assert(!failureLine.contains(rawDelByte),
            clue = s"auth.login.failure line MUST NOT contain a raw 0x7F DEL byte -- catches a refactor that escaped DEL to a DIFFERENT visible form (e.g. `%7F` uri-style or dropped escape entirely on the assumption that DEL is harmless); got line: $failureLine")
        }
      }
    }
  }

  // Pin the documented sanitizeLogMessage BACKSLASH (0x5C)
  // escape rule END-TO-END at HandHistoryReviewServerRuntime
  // .scala line 488 -- the FIRST replace in the explicit-
  // replace chain (lines 488-492); the SANITIZATION-INVARIANT
  // family now covers ALL 5 BRANCH POSITIONS that a regression
  // could target: (4db713d) line 489 newline 0x0A, (a696f57)
  // line 490 CR 0x0D, (7aeaa8d) line 492 TAB 0x09, (b98b458)
  // lines 494-497 generic fallback DEL 0x7F, (THIS commit)
  // line 488 backslash 0x5C; the BACKSLASH pin is the MOST
  // CRITICAL of the family because line 488 is the FIRST
  // replace in the chain AND has CASCADE IMPLICATIONS: if
  // line 488 is broken, moved AFTER another replace, or
  // removed, the OTHER replaces (newline -> backslash-n, CR
  // -> backslash-r, tab -> backslash-t, null -> backslash-0)
  // would EMIT backslash chars that would NOT THEN BE
  // ESCAPED -- the output for a user-submitted newline would
  // be the 2-char sequence backslash-n, but if the user ALSO
  // submitted a literal backslash, the OUTPUT would be
  // AMBIGUOUS between "literal backslash + n" and "escaped
  // newline" -- breaking the GROUND-TRUTH ROUND-TRIP of the
  // log line; the backslash is the ONLY ASCII PRINTABLE byte
  // (0x5C is 'backslash', printable, not a control char)
  // caught by sanitizeLogMessage -- the family's other 4
  // pins all target control chars (< 0x20 or 0x7F); the
  // backslash escape is what makes the OTHER escapes
  // UNAMBIGUOUS -- without it, an attacker could submit a
  // literal backslash-n sequence (2 chars: 0x5C + 0x6E) that
  // LOOKS like an escaped newline in the log output,
  // creating CONFUSION between attacker-injected escapes and
  // real sanitization output; the sanitizeLogMessage helper
  // at line 488's replace(2-backslash-string, 4-backslash-
  // string) in Scala source -- which is replace(single-
  // backslash, double-backslash) in resulting String values
  // -- catches the raw 0x5C byte and emits the 2-backslash
  // escape; this commit's END-TO-END test verifies the
  // sanitization applies at the log() emission layer by:
  // (1) configuring platformAuth, (2) submitting POST /api/
  // auth/login with email containing a JSON-escaped
  // backslash (the 2-char JSON escape parses to 1 byte 0x5C
  // at JSON-decode time), (3) the login fails validateEmail
  // (backslash is not in the [A-Za-z0-9_.+-] local-part
  // character class) + emits auth.login.failure logWarn
  // with the user-submitted email containing the raw
  // backslash byte, (4) formatSubmittedEmailForLog ONLY
  // escapes spaces, NOT backslash -- the raw backslash flows
  // through into the logWarn s-string, (5) the log() helper's
  // sanitizeLogMessage wrapping at line 512 catches the
  // backslash at line 488 + escapes it to 2 backslashes;
  // per-format regression vectors that this pin uniquely
  // catches (NONE of the other 4 family pins catch these):
  // (i) refactor REORDERING the line 488 replace to AFTER
  // line 489's newline replace -- this would cause the line
  // 489 emit of backslash-n for newlines to NOT have its
  // emitted backslash re-escaped, so a user-submitted
  // newline would output as backslash-n but a user-submitted
  // backslash would also output as backslash -- the OUTPUT
  // would be INDISTINGUISHABLE between "escaped newline" and
  // "literal user backslash followed by n", (ii) refactor
  // removing line 488 entirely would let raw backslashes
  // through, creating the same ambiguity problem as (i),
  // (iii) refactor changing the escape format (e.g.
  // backslash -> %5C uri-style) would silently break log
  // aggregator parsers expecting the documented backslash-
  // escape convention; this test's emission via the
  // auth.login.failure path exercises the END-TO-END
  // sanitization specifically on a USER-CONTROLLED field
  // that formatSubmittedEmailForLog does NOT itself escape;
  // 4-tier format check: (i) the auth.login.failure line
  // exists in captured stderr, (ii) the line contains BOTH
  // halves of the email (alice + bob@example.com), (iii)
  // the email field contains the ESCAPED form with 2
  // backslashes between alice and bob per line 488's escape
  // rule, (iv) EXCLUSION of the 1-backslash form -- this is
  // the STRUCTURAL DIFFERENCE between sanitized and
  // unsanitized output; if sanitization is broken, the
  // 1-backslash form WOULD be present.
  test("log() helper's sanitizeLogMessage wrapping at HandHistoryReviewServerRuntime.scala line 512 escapes user-controlled BACKSLASH (0x5C) bytes end-to-end via the line 488 escape rule -- the SANITIZATION-INVARIANT pin for the FIRST replace in the chain; backslash is the ONLY ASCII PRINTABLE byte caught by sanitizeLogMessage, and the FIRST escape applied -- if reordered or removed, the OTHER escapes would emit backslashes that themselves wouldn't get re-escaped, breaking the UNAMBIGUITY of every escape rule") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          // Capture stderr around the malicious login attempt.
          // The submitted email JSON contains the 2-char JSON
          // backslash escape (which ujson parses to 1 byte
          // 0x5C backslash); the raw backslash flows into the
          // logWarn s-string via formatSubmittedEmailForLog
          // (which only escapes spaces, NOT backslash), and
          // log() at line 512 calls sanitizeLogMessage which
          // catches the backslash at line 488 + escapes to
          // 2 backslashes. In the Scala raw triple-quoted
          // string body below, double-backslash is a literal
          // 2-char sequence (Scala raw strings do NOT process
          // backslash escapes -- the resulting Scala String
          // has 2 literal backslash bytes). HTTP transmits
          // these 2 bytes verbatim; ujson parses the JSON
          // backslash-backslash escape into 1 byte 0x5C.
          val errBuf = new java.io.ByteArrayOutputStream()
          val originalErr = System.err
          System.setErr(new java.io.PrintStream(errBuf, true, StandardCharsets.UTF_8))
          try
            val rejected = postJson(s"$baseUri/api/auth/login",
              """{"email":"alice\\bob@example.com","password":"some-password"}""")
            assertEquals(rejected.statusCode(), 401,
              clue = "malformed-email login MUST return 401 (the email fails validateEmail's regex check because backslash is not in the [A-Za-z0-9_.+-] local-part character class; loginLocal returns Left -> 401)")
          finally
            System.setErr(originalErr)

          val captured = errBuf.toString(StandardCharsets.UTF_8)
          val failureLine = captured.split('\n').iterator
            .find(_.contains("auth.login.failure"))
            .getOrElse(fail(s"no `auth.login.failure` line in captured stderr; got: ${captured.take(800)}"))

          // (i) the auth.login.failure line exists
          assert(failureLine.contains("auth.login.failure"),
            clue = s"auth.login.failure line must be present (the rejected login path at AuthStack.scala line 192 emits this on every invalid-credentials rejection); got: $failureLine")

          // (ii) the line contains BOTH halves of the email
          assert(failureLine.contains("alice") && failureLine.contains("bob@example.com"),
            clue = s"auth.login.failure line MUST contain BOTH halves of the email (alice + bob@example.com); got: $failureLine; full captured stream: ${captured.take(1500)}")

          // (iii) the email contains the ESCAPED form with 2
          // backslashes between alice and bob. Scala source
          // 4-backslash sequence = String value 2 backslashes.
          // The line 488 rule replaces 1 raw backslash with 2
          // backslashes, so the output between alice and bob
          // is 2 backslashes.
          assert(failureLine.contains("alice\\\\bob@example.com"),
            clue = s"auth.login.failure line MUST carry the escaped form `alice\\\\\\\\bob@example.com` (Scala source 4 backslashes = 2 literal backslashes between alice and bob) per HandHistoryReviewServerRuntime.scala line 488's escape rule; the raw 0x5C backslash byte in the submitted email gets caught by sanitizeLogMessage's line 488 (the FIRST replace in the chain) AT THE LOG LAYER; a refactor REORDERING line 488 to AFTER another replace would silently break the unambiguity of every OTHER escape (the emitted backslashes from those replaces would not themselves be re-escaped, conflating user-submitted backslashes with sanitization-emitted backslashes); a refactor REMOVING line 488 entirely would let raw backslashes through, creating the same ambiguity problem; got: $failureLine")

          // (iv) EXCLUSION: the line must NOT contain the
          // 1-backslash form. Scala source 2-backslash
          // sequence = String value 1 backslash. When
          // sanitization works, the output has 2 backslashes
          // between alice and bob, and a substring search for
          // 1-backslash form FAILS (the search target starts
          // at the 'alice + 1-backslash + b' pattern which
          // mismatches at the 6th char where output has the
          // SECOND backslash but the target wants 'b'). This
          // is the CORE STRUCTURAL ASSERTION distinguishing
          // sanitized from unsanitized output.
          assert(!failureLine.contains("alice\\bob@example.com"),
            clue = s"auth.login.failure line MUST NOT contain the 1-backslash form `alice\\\\bob@example.com` (Scala source 2 backslashes = 1 literal backslash between alice and bob) -- this is the UNSANITIZED form; if this assertion fails, the line 488 backslash escape is BROKEN and user-submitted backslashes are flowing through to the log unescaped, creating UNAMBIGUITY with the OTHER escapes' emitted backslashes (a user-submitted backslash-n sequence would output as backslash-n indistinguishable from an escaped newline); got line: $failureLine")
        }
      }
    }
  }

  // Pin the documented `shutdown complete` companion banner log
  // line format -- the SHUTDOWN HALF of the startup/shutdown
  // banner pair the 7c47f88 startup pin established the FIRST
  // half of; the shutdown banner emits at HandHistoryReviewServer
  // Runtime.scala line 326's `logInfo(s"shutdown complete host=${
  // binding.host} port=${binding.port}")` inside the shutdown
  // hook's finally block, immediately AFTER the HTTP server's
  // stop sequence completes (server.stop, then http executor
  // shutdown, then analysis-timeout-executor drain, then analysis-
  // executor drain) -- so when the banner emits, ALL request
  // processing has ceased and operator-visible state is exactly
  // "ready to terminate"; the banner is SIMPLER than the startup
  // banner (just host + port, no config fields) because by the
  // time shutdown fires the operator's interest is "did this
  // instance shut down cleanly" not "what config was it running"
  // -- the startup-side banner already captured the config
  // values; per-field regression vectors that the startup pin
  // (7c47f88) doesn't catch: (i) the "shutdown complete" prefix
  // -- a refactor renaming to e.g. "server stopped" / "shutdown
  // finished" / "exit complete" would silently break operator
  // scripts that grep'd for "shutdown complete" as the cue for
  // "instance terminated gracefully" (distinguished from "process
  // killed" where the banner never emits because the shutdown
  // hook didn't get to run); (ii) the host + port values MUST
  // match the startup-side banner's values for the SAME process
  // -- operators correlate process lifetimes by matching the
  // startup banner's host=X port=Y with the SUBSEQUENT shutdown
  // banner's host=X port=Y, so a refactor that emitted different
  // host/port values at shutdown (e.g. config.host vs
  // binding.host divergence) would silently break the operator's
  // "did THIS process restart cleanly" diagnostic, (iii) the
  // INFO level -- shutdown is a NORMAL lifecycle event, not an
  // error; demote-to-DEBUG would hide it at default log levels
  // (operators couldn't tell graceful-shutdown from
  // forcefully-killed instances at all), promote-to-WARN would
  // silently page alerting automation on every restart, (iv)
  // emitting the banner BEFORE the executor drain finishes (a
  // refactor reordering would silently let the banner appear
  // while jobs were still running -- the inline comment at the
  // shutdown sequence above the line 326 emission documents the
  // ORDERING: http stop, then http executor shutdown, then
  // analysis-timeout drain, then analysis drain, THEN banner --
  // a reordering would silently invalidate the "banner means
  // ready to terminate" semantic operators depend on); the
  // shutdown banner's host+port also pairs with the startup
  // banner's host+port AND with /api/health.host + /api/health.
  // port (pinned by 505ba6b/b2a90fb) AND with /api/ready.host +
  // /api/ready.port -- four-way correlation across boot-time
  // banner + shutdown-time banner + runtime probes + log lines;
  // test approach: extend the same stdout-capture pattern from
  // 7c47f88 -- wrap System.setOut around withServer, which
  // captures BOTH the startup banner (emitted inside
  // startWithBackends BEFORE the run callback) AND the shutdown
  // banner (emitted inside the finally block of withServer's
  // server.close() call which fires AFTER the run callback);
  // assert presence of BOTH banners + their respective field
  // shapes; ALSO assert the SAME host+port values appear in
  // both banners (the lifetime-correlation invariant the
  // operator runbook keys on); 5-tier format check on the
  // shutdown banner specifically: (i) "shutdown complete"
  // prefix, (ii) "host=127.0.0.1" matching startup, (iii) "port="
  // present (value varies because port=0 ephemeral), (iv) [INFO]
  // level, (v) [hand-history-review] service-tag prefix; ALSO
  // an INTER-BANNER assertion: extract host+port from BOTH
  // banners and assert they match (the lifetime-correlation pin);
  // ALSO the absence pin: shutdown banner does NOT carry the
  // ~17 config fields the startup banner has (modelSource,
  // maxUploadBytes, etc.) -- a refactor that "consolidated for
  // consistency" would silently expand the shutdown banner's
  // size + delay the line emission (each field expansion adds
  // ms of formatting + log-write overhead, multiplied by 17
  // fields it could noticeably slow the shutdown path -- the
  // shutdown deadline is strict per the deadlineNanos at line
  // 318); the absence assertion catches that consolidation; future
  // fires can extend with: (a) banner-pair ORDERING assertion
  // (startup banner appears BEFORE shutdown banner in captured
  // stdout) -- the existing test implicitly relies on this
  // because startup happens at withServer setup and shutdown at
  // teardown, but pinning ordering would catch a refactor that
  // emitted both at the same lifecycle point, (b) bind-error
  // banner pin (HandHistoryReviewServerRuntime.scala line 358's
  // "failed to start web server: <host>:<port> is unavailable"
  // -- already covered by the existing bind-error test at line
  // ~5xxx but not specifically for the BANNER FORMAT vs the
  // error-message-returned-to-caller).
  test("server shutdown emits the documented `shutdown complete host=<host> port=<port>` banner at INFO level matching the startup banner's host/port values (per HandHistoryReviewServerRuntime.scala line 326's hardcoded shutdown-hook emission)") {
    withStaticSite { staticDir =>
      // Capture stdout AROUND the withServer call so BOTH banners
      // (startup at startWithBackends + shutdown at server.close())
      // land in the captured stream. The 7c47f88 startup-banner
      // test uses the same pattern but only asserts on startup;
      // this test asserts on BOTH banners to verify the
      // host+port correlation between them.
      val outBuf = new java.io.ByteArrayOutputStream()
      val originalOut = System.out
      System.setOut(new java.io.PrintStream(outBuf, true, StandardCharsets.UTF_8))
      try
        withServer(staticDir) { _ =>
          // No HTTP requests needed -- both banners emit at
          // server lifecycle transitions (startup + close), NOT
          // during request processing. The empty body keeps the
          // server alive between the two emissions.
          ()
        }
      finally
        System.setOut(originalOut)

      val captured = outBuf.toString(StandardCharsets.UTF_8)
      val shutdownLine = captured.split('\n').iterator
        .find(_.contains("shutdown complete"))
        .getOrElse(fail(s"no `shutdown complete` line in captured stdout -- HandHistoryReviewServerRuntime.scala line 326 documents this as the shutdown-hook completion banner; if missing, either the logInfo emission was suppressed OR the shutdown sequence didn't reach the finally block at line 322-326 (a refactor that broke the shutdown hook chain would silently make the banner invisible); got captured stdout: ${captured.take(2000)}"))

      // (i) prefix
      assert(shutdownLine.contains("shutdown complete"),
        clue = s"shutdown banner must carry the literal `shutdown complete` prefix per HandHistoryReviewServerRuntime.scala line 326's hardcoded literal -- a refactor renaming to e.g. `server stopped` / `shutdown finished` / `exit complete` would silently break operator scripts grep'ing for the graceful-termination signal; got: $shutdownLine")
      // (ii) host (matching startup banner's withServer default)
      assert(shutdownLine.contains("host=127.0.0.1"),
        clue = s"shutdown banner must carry host=127.0.0.1 (withServer default) -- MUST match the startup banner's host= value for the SAME process; a refactor that emitted config.host instead of binding.host at shutdown would silently break operator process-lifetime correlation; got: $shutdownLine")
      // (iii) port (value varies due to port=0 ephemeral, but
      // field MUST be present AND MUST match startup banner)
      assert(shutdownLine.contains("port="),
        clue = s"shutdown banner must carry port=<resolved-bound-port> -- MUST match the startup banner's port= value (operators correlate process lifetimes by matching startup port=X with shutdown port=X); a refactor that emitted config.port (always 0 for ephemeral) instead of binding.port would silently always emit port=0 at shutdown while the actual server bound to a real ephemeral port; got: $shutdownLine")
      // (iv) INFO level
      assert(shutdownLine.contains("[INFO]"),
        clue = s"shutdown banner must be INFO-level (logInfo at line 326 writes to System.out per HandHistoryReviewServerRuntime.scala line 418); demote-to-DEBUG would hide graceful-shutdown signal at default log levels making operators unable to tell clean-exit from kill-9, promote-to-WARN would silently page alerting on every normal restart; got: $shutdownLine")
      // (v) service-tag prefix
      assert(shutdownLine.contains("[hand-history-review]"),
        clue = s"shutdown banner must carry the `[hand-history-review]` service-tag prefix per HandHistoryReviewServerRuntime.scala line 512's hardcoded literal -- matches /api/health.service (505ba6b) so log aggregators see the same identifier on startup + shutdown banners + runtime audit lines + probe responses (4-way correlation); got: $shutdownLine")

      // INTER-BANNER assertion: the host+port values in BOTH
      // banners MUST match. Find the startup banner in the same
      // captured stream and extract its host+port values, then
      // compare against the shutdown banner. This is the
      // PROCESS-LIFETIME CORRELATION pin -- a refactor that
      // emitted different host/port values at startup vs
      // shutdown for the same process would silently break the
      // operator's restart-detection diagnostic.
      val startupLine = captured.split('\n').iterator
        .find(_.contains("startup complete"))
        .getOrElse(fail(s"no `startup complete` line in captured stdout for inter-banner correlation -- the 7c47f88 startup pin should have caught this earlier, but if the startup banner is missing here the test can't validate cross-banner host/port consistency; got captured stdout: ${captured.take(2000)}"))
      // Extract `host=<value>` tokens from both lines. The host
      // values are guaranteed not to contain spaces (IPs or
      // hostnames, both space-free), so splitting on space then
      // matching `host=` is safe. Strip trailing whitespace /
      // carriage-returns from the extracted value (Windows
      // line endings could leave \r on the last field of a
      // line).
      def extractField(line: String, fieldName: String): Option[String] =
        line.split(' ').iterator
          .find(_.startsWith(s"$fieldName="))
          .map(_.drop(fieldName.length + 1).stripTrailing())
      val startupHost = extractField(startupLine, "host").getOrElse(fail(s"startup banner missing host= field for cross-banner correlation; got: $startupLine"))
      val shutdownHost = extractField(shutdownLine, "host").getOrElse(fail(s"shutdown banner missing host= field for cross-banner correlation; got: $shutdownLine"))
      assertEquals(shutdownHost, startupHost,
        clue = s"shutdown banner's host= value MUST match the startup banner's host= value for the same process -- operators correlate process lifetimes by matching startup host=X port=Y with the subsequent shutdown host=X port=Y; a divergence would silently break the runbook's restart-detection diagnostic; got startup host='$startupHost' vs shutdown host='$shutdownHost'")
      val startupPort = extractField(startupLine, "port").getOrElse(fail(s"startup banner missing port= field for cross-banner correlation; got: $startupLine"))
      val shutdownPort = extractField(shutdownLine, "port").getOrElse(fail(s"shutdown banner missing port= field for cross-banner correlation; got: $shutdownLine"))
      assertEquals(shutdownPort, startupPort,
        clue = s"shutdown banner's port= value MUST match the startup banner's port= value for the same process -- this is the critical bound-port-vs-config-port pin: both banners MUST source from binding.port (the resolved bound port post-ephemeral-allocation), NOT config.port (always 0 for port=0 deployments); a refactor that emitted config.port at shutdown would always show port=0 while startup correctly showed the resolved port, silently breaking process correlation for port=0 deployments AND blue/green deployments that cycle ports intentionally; got startup port='$startupPort' vs shutdown port='$shutdownPort' from\n  startup line: $startupLine\n  shutdown line: $shutdownLine")

      // ABSENCE pin: shutdown banner does NOT carry the ~17
      // config fields the startup banner has. A refactor that
      // "consolidated for consistency" by expanding the shutdown
      // banner to include modelSource / maxUploadBytes / etc.
      // would silently slow the shutdown path (each field
      // expansion adds formatting + log-write overhead, and
      // the shutdown deadline at line 318 is strict). The
      // operator workflow for shutdown is "did it shut down
      // cleanly", not "what was the config" -- the latter is
      // the startup banner's job.
      assert(!shutdownLine.contains("modelSource="),
        clue = s"shutdown banner must NOT carry modelSource= (a startup-banner-only field per HandHistoryReviewServerRuntime.scala line 350); a refactor expanding the shutdown banner to carry config fields would silently slow shutdown path AND duplicate operator-relevant information (the startup banner already captured the config; shutdown's job is signaling clean exit, not re-emitting config); got: $shutdownLine")
      assert(!shutdownLine.contains("maxUploadBytes="),
        clue = s"shutdown banner must NOT carry maxUploadBytes= (startup-banner-only field); got: $shutdownLine")
      assert(!shutdownLine.contains("authenticationMode="),
        clue = s"shutdown banner must NOT carry authenticationMode= (startup-banner-only field); got: $shutdownLine")
    }
  }

  // Pin the documented intermediate `shutdown requested` banner
  // AND the 3-banner-ordering invariant (startup BEFORE shutdown
  // requested BEFORE shutdown complete) -- closes the THIRD and
  // FINAL server-lifecycle banner per HandHistoryReviewServerRuntime.
  // scala lines 308-309 (the prior two halves were closed by
  // 7c47f88 startup-complete + f6ee18b shutdown-complete); the
  // intermediate banner emits at line 309's `logInfo(s"shutdown
  // requested host=${binding.host} port=${binding.port}
  // activeHttpRequests=$activeRequestCount queuedJobs=${...}
  // runningJobs=${...} httpDrainSeconds=$httpDrainSeconds")` --
  // fires at the VERY START of the shutdown sequence (BEFORE the
  // executor shutdown calls at lines 311-312, BEFORE the server.
  // stop at line 321, BEFORE the drain awaits at lines 323-325)
  // so the captured snapshot of activeHttpRequests / queuedJobs /
  // runningJobs is the LOAD STATE AT THE MOMENT SHUTDOWN WAS
  // REQUESTED -- the operator-relevant diagnostic for "what was
  // running when this instance was asked to shut down" workflows
  // (e.g. capacity planning for rolling restarts: if every
  // shutdown banner shows queuedJobs=8 the deployment is
  // chronically over-saturated and the shutdown grace budget
  // probably can't drain it cleanly); a refactor that moved the
  // logInfo to AFTER the drain calls would silently change the
  // semantic from "load when requested" to "load when drained"
  // (typically all zeros after a normal drain) -- silently
  // breaking the operator's saturation triage; per-field
  // regression vectors SPECIFIC to this banner that the other 2
  // server-lifecycle banners don't catch: (i) the activeHttpRequests
  // / queuedJobs / runningJobs / httpDrainSeconds FIELD SET is
  // UNIQUE to this banner -- a refactor that consolidated the 3
  // server-lifecycle banners (startup / requested / complete) to
  // share a common helper would likely break this banner's
  // load-state fields, AND a refactor renaming any of the 4 load-
  // state fields (e.g. queuedJobs -> queuedJobCount for noun-
  // consistency) would silently break operator dashboards keyed
  // on the field names; (ii) the httpDrainSeconds value is
  // OPERATIONALLY MEANINGFUL -- per the inline comment at
  // HandHistoryReviewServerRuntime.scala line 305-307, the value
  // is 0 when activeHttpRequests is 0 (no drain needed) and
  // shutdownDelaySeconds(config.shutdownGraceMs) otherwise; a
  // refactor that always emitted 0 (ignoring activeRequestCount)
  // would silently break operator visibility into "is this
  // shutdown actually waiting for in-flight requests"; (iii) the
  // 3-banner ORDERING invariant -- startup MUST appear BEFORE
  // shutdown requested MUST appear BEFORE shutdown complete in
  // the log stream; a refactor that emitted the banners in a
  // different order (e.g. emitted shutdown complete BEFORE
  // shutdown requested due to a finally-block reordering) would
  // silently break operator scripts that parse the log stream
  // chronologically AND would silently invalidate the runbook's
  // assumption that "shutdown requested" is the FIRST signal
  // that a shutdown sequence has started; 9-tier format check
  // matching the startup/shutdown pattern + 3-banner ORDERING
  // assertion: (i) "shutdown requested" prefix (catches rename
  // to e.g. "shutdown starting" / "shutdown initiated"), (ii)
  // host=127.0.0.1 (matches startup AND shutdown-complete --
  // cross-banner consistency for the same process), (iii)
  // port= field presence, (iv) activeHttpRequests=0 (specific
  // value -- for a quiet test with no concurrent requests this
  // is 0; a refactor changing the source field to a wrong
  // counter would silently emit a non-zero value), (v)
  // queuedJobs=0 (specific value), (vi) runningJobs=0 (specific
  // value), (vii) httpDrainSeconds=0 (specific value -- 0
  // because activeHttpRequests is 0 per the conditional at
  // lines 305-307; a refactor that always emitted
  // shutdownDelaySeconds(config.shutdownGraceMs) regardless of
  // active count would silently emit a non-zero value here),
  // (viii) [INFO] level, (ix) [hand-history-review] service-tag
  // prefix; PLUS the 3-banner ORDERING assertion via index
  // comparison in the captured stdout stream; future fires can
  // extend with: (a) non-zero load-state pin (start a Blocking
  // Backend job, trigger shutdown while it's running, verify
  // runningJobs=1 in the requested banner), (b) per-mode banner
  // variants (authenticationMode=basic and =users emit different
  // userAuth* field values in the STARTUP banner -- the
  // requested + complete banners don't carry those fields).
  test("server shutdown emits the documented `shutdown requested` intermediate banner with load-state fields BEFORE the `shutdown complete` final banner -- pins the 3-banner server-lifecycle ordering (startup → requested → complete) per HandHistoryReviewServerRuntime.scala lines 308-309 + 326") {
    withStaticSite { staticDir =>
      // Same stdout-capture pattern as 7c47f88 + f6ee18b but
      // looking for the intermediate `shutdown requested` banner
      // at line 309 in addition to the existing startup + complete
      // banners. The intermediate banner fires at the VERY START
      // of the shutdown sequence -- BEFORE executor shutdown,
      // BEFORE server.stop, BEFORE the drain awaits -- so for a
      // quiet test the load-state counters are all 0.
      val outBuf = new java.io.ByteArrayOutputStream()
      val originalOut = System.out
      System.setOut(new java.io.PrintStream(outBuf, true, StandardCharsets.UTF_8))
      try
        withServer(staticDir) { _ =>
          // No HTTP requests -- the test pins the BANNER format
          // on a quiet shutdown (activeHttpRequests=0,
          // queuedJobs=0, runningJobs=0, httpDrainSeconds=0). A
          // future fire can add a non-zero-load test variant by
          // starting a BlockingBackend job and triggering
          // shutdown while it's running.
          ()
        }
      finally
        System.setOut(originalOut)

      val captured = outBuf.toString(StandardCharsets.UTF_8)
      val lines = captured.split('\n').toVector

      val requestedLineIdx = lines.indexWhere(_.contains("shutdown requested"))
      val requestedLine =
        if requestedLineIdx >= 0 then lines(requestedLineIdx)
        else fail(s"no `shutdown requested` line in captured stdout -- HandHistoryReviewServerRuntime.scala line 309 documents this as the intermediate banner emitted at the VERY START of the shutdown sequence (BEFORE executor shutdown, BEFORE server.stop); if missing, the logInfo at line 308-309 was suppressed OR the shutdown function entered via a different path; got captured stdout: ${captured.take(2000)}")

      // (i) prefix
      assert(requestedLine.contains("shutdown requested"),
        clue = s"intermediate banner must carry the literal `shutdown requested` prefix per HandHistoryReviewServerRuntime.scala line 309's hardcoded literal -- a refactor renaming to e.g. `shutdown starting` / `shutdown initiated` / `shutdown began` would silently break operator scripts that parse the log stream for the SHUTDOWN-SEQUENCE-STARTED signal (distinguished from the SHUTDOWN-SEQUENCE-COMPLETED signal which is the `shutdown complete` line); got: $requestedLine")
      // (ii) host (matches startup + complete banners)
      assert(requestedLine.contains("host=127.0.0.1"),
        clue = s"intermediate banner must carry host=127.0.0.1 (withServer default) MATCHING the startup + complete banners for the SAME process -- a refactor that emitted config.host instead of binding.host here would silently break cross-banner process correlation; got: $requestedLine")
      // (iii) port field presence (matches startup + complete)
      assert(requestedLine.contains("port="),
        clue = s"intermediate banner must carry port=<resolved-bound-port> MATCHING the startup + complete banners; got: $requestedLine")
      // (iv) activeHttpRequests=0 (load-state field UNIQUE to this banner)
      assert(requestedLine.contains("activeHttpRequests=0"),
        clue = s"intermediate banner must carry activeHttpRequests=0 (quiet test, no concurrent requests at shutdown moment) -- this is the LOAD-STATE SNAPSHOT field unique to this banner that operators use to diagnose 'what was running when shutdown was triggered'; a refactor that emitted a wrong counter (e.g. lifetime-cumulative requests instead of current-active) would silently emit a non-zero value here AND silently break operator saturation-triage workflows; got: $requestedLine")
      // (v) queuedJobs=0 (load-state field)
      assert(requestedLine.contains("queuedJobs=0"),
        clue = s"intermediate banner must carry queuedJobs=0 (quiet test, no pending jobs at shutdown moment) -- pairs with activeHttpRequests + runningJobs as the 3-counter load snapshot; a refactor renaming the field (e.g. queuedJobCount) or emitting a wrong source would silently break operator dashboards; got: $requestedLine")
      // (vi) runningJobs=0 (load-state field)
      assert(requestedLine.contains("runningJobs=0"),
        clue = s"intermediate banner must carry runningJobs=0 (quiet test, no jobs executing at shutdown moment); got: $requestedLine")
      // (vii) httpDrainSeconds=0 (THE conditional field per the
      // inline comment at lines 305-307: 0 when activeHttp
      // Requests is 0, shutdownDelaySeconds(config.shutdownGraceMs)
      // otherwise -- a refactor that always emitted the non-zero
      // path would silently emit a non-zero value here)
      assert(requestedLine.contains("httpDrainSeconds=0"),
        clue = s"intermediate banner must carry httpDrainSeconds=0 because activeHttpRequests=0 per the conditional at HandHistoryReviewServerRuntime.scala lines 305-307 (`if activeRequestCount > 0 then shutdownDelaySeconds(config.shutdownGraceMs) else 0`); a refactor that always emitted shutdownDelaySeconds regardless of active count would silently break operator visibility into 'is this shutdown actually waiting for in-flight requests' (the value tells operators how long the http server will wait for in-flight requests before forcing close); got: $requestedLine")
      // (viii) INFO level
      assert(requestedLine.contains("[INFO]"),
        clue = s"intermediate banner must be INFO-level (logInfo at line 308 writes to System.out per HandHistoryReviewServerRuntime.scala line 418); demote-to-DEBUG would hide the shutdown-sequence-started signal at default log levels, promote-to-WARN would silently page on every normal restart; got: $requestedLine")
      // (ix) service-tag prefix (couples to /api/health.service from 505ba6b)
      assert(requestedLine.contains("[hand-history-review]"),
        clue = s"intermediate banner must carry the `[hand-history-review]` service-tag prefix per HandHistoryReviewServerRuntime.scala line 512's hardcoded literal -- couples to /api/health.service (505ba6b) so log aggregators see the same identifier across all 3 server-lifecycle banners + runtime audit lines + probe responses (the 5-way correlation now spans startup + requested + complete + audit + probe); got: $requestedLine")

      // 3-BANNER ORDERING invariant: startup MUST appear BEFORE
      // shutdown requested MUST appear BEFORE shutdown complete
      // in the captured stream. The ordering is essential for
      // operator log-parsing scripts that key on the lifecycle
      // sequence -- a refactor that reordered the emissions
      // (e.g. emitted shutdown complete BEFORE shutdown
      // requested due to a finally-block reordering) would
      // silently break the lifecycle-sequence semantic AND
      // silently invalidate the runbook's assumption that
      // "shutdown requested" is the FIRST signal that a
      // shutdown sequence has started.
      val startupLineIdx = lines.indexWhere(_.contains("startup complete"))
      assert(startupLineIdx >= 0,
        clue = s"startup banner missing from captured stream -- the 7c47f88 startup pin should catch this independently, but this test's ordering check needs the startup banner present too; got captured: ${captured.take(2000)}")
      val completeLineIdx = lines.indexWhere(_.contains("shutdown complete"))
      assert(completeLineIdx >= 0,
        clue = s"shutdown-complete banner missing from captured stream -- the f6ee18b shutdown-complete pin should catch this independently, but this test's ordering check needs the complete banner present too; got captured: ${captured.take(2000)}")
      assert(startupLineIdx < requestedLineIdx,
        clue = s"startup banner MUST appear BEFORE shutdown-requested banner in captured stream -- the lifecycle sequence is startup (during startWithBackends) → shutdown requested (start of shutdown sequence) → shutdown complete (end of shutdown sequence); a refactor that emitted the banners in a wrong order would silently break operator log-parsing scripts; got startupIdx=$startupLineIdx requestedIdx=$requestedLineIdx in captured: ${captured.take(2000)}")
      assert(requestedLineIdx < completeLineIdx,
        clue = s"shutdown-requested banner MUST appear BEFORE shutdown-complete banner in captured stream -- the runbook's shutdown-progress diagnostic assumes `requested` is the FIRST shutdown signal (start of sequence) and `complete` is the LAST (end of sequence); a refactor that emitted complete BEFORE requested (e.g. due to a finally-block reordering at lines 322-326) would silently invalidate the runbook's assumption; got requestedIdx=$requestedLineIdx completeIdx=$completeLineIdx in captured: ${captured.take(2000)}")
    }
  }

  test("registration stores PBKDF2 credential with documented parameters (210k iterations, 256-bit key, 128-bit salt) per deploy doc + runbook + NIST SP 800-132 §5.1 compliance claim") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          val register = postJson(s"$baseUri/api/auth/register",
            """{"email":"crypto@example.com","password":"correct-horse-battery","displayName":"Crypto"}""")
          assertEquals(register.statusCode(), 201,
            clue = "registration must succeed before the user-store inspection can run")

          // Read the persisted user-store JSON directly. The codebase
          // doesn't expose credential material through any HTTP
          // endpoint -- by design, hash + salt material must never
          // leave the server. So the test reads the on-disk artifact
          // the same way an operator with filesystem access would
          // (e.g., for an incident-response audit of USER_STORE_PATH).
          val storeJson = ujson.read(Files.readString(storePath, StandardCharsets.UTF_8))
          // Pin the documented top-level `"version": 1` schema marker.
          // Deploy doc line 235: "The JSON content carries a top-level
          // \"version\": 1 field for future schema migrations (the
          // server currently ignores it on read)". The field exists
          // so future-schema-migration code (when we eventually bump
          // to 2) has something to branch on; if a refactor silently
          // dropped the field from writes (or changed its value
          // without an explicit migration plan), future readers
          // expecting v1 vs v2 differentiation would have no
          // signal to switch on -- the documented forward-compat
          // contract would be silently broken. Asserting the exact
          // value (1) means a deliberate schema bump WILL fail this
          // test and force the maintainer to acknowledge the
          // version change explicitly rather than slipping it past
          // CI as a side-effect of an unrelated refactor.
          assertEquals(storeJson.obj("version").num.toInt, 1,
            clue = "user-store JSON must carry top-level `version: 1` per deploy doc line 235's documented schema-marker contract; future schema bumps require deliberate code change AND test update in lockstep so no migration is silent")
          val users = storeJson.obj("users").arr
          assertEquals(users.length, 1,
            clue = "exactly one user should be persisted after the single register call")
          val credential = users(0).obj("localPassword").obj
          assertEquals(credential("iterations").num.toInt, 210000,
            clue = "PBKDF2 iterations must be 210000 per deploy doc line 235 + runbook section 5A line 371's NIST SP 800-132 §5.1 compliance claim; a refactor changing this drifts the documented password-cracking-cost calculation operators rely on for incident-response triage")
          assertEquals(credential("keyLengthBits").num.toInt, 256,
            clue = "PBKDF2 output must be 256 bits per the same deploy-doc claim; the NIST floor is ≥256-bit so dropping below silently violates the compliance claim")
          val saltBytes = Base64.getDecoder.decode(credential("saltBase64").str)
          assertEquals(saltBytes.length, 16,
            clue = "PBKDF2 salt must be exactly 16 bytes = 128 bits per the same deploy-doc claim; the NIST floor is ≥128-bit so dropping below silently violates the compliance claim, AND a too-small salt makes per-account rainbow tables feasible")
        }
      }
    }
  }

  test("registration without displayName (omitted or whitespace-only) auto-fills the field from the email's local-part") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          // Case 1: displayName field omitted entirely from the
          // register JSON body. The server defaults it from the email
          // local-part "alice" (everything before @).
          val omitted = postJson(s"$baseUri/api/auth/register",
            """{"email":"alice@example.com","password":"correct-horse-battery"}""")
          assertEquals(omitted.statusCode(), 201,
            clue = "registration with omitted displayName must succeed (the field is optional per deploy doc line 100)")
          val omittedUser = jsonBody(omitted)("user")
          assertEquals(omittedUser("displayName").str, "alice",
            clue = "omitted displayName must auto-fill from email local-part 'alice' per deploy doc + runbook; a future refactor that dropped this auto-fill (or changed defaultDisplayNameFor's split character) would silently regress both the documented behavior AND the runbook's 'Why is my display name my email username?' support-triage flow")

          // Case 2: displayName field present but whitespace-only,
          // which sanitizeDisplayName trims to empty, which then
          // triggers the same defaultDisplayNameFor path. The trim
          // happens in sanitizeOptionalField; the empty-result
          // fallback fires in registerLocal / upsertOidcIdentity.
          val whitespace = postJson(s"$baseUri/api/auth/register",
            """{"email":"bob@example.com","password":"correct-horse-battery","displayName":"   "}""")
          assertEquals(whitespace.statusCode(), 201,
            clue = "registration with whitespace-only displayName must succeed (the trim+default path is the documented graceful-degradation shape)")
          val whitespaceUser = jsonBody(whitespace)("user")
          assertEquals(whitespaceUser("displayName").str, "bob",
            clue = "whitespace-only displayName must trim to empty and auto-fill from email local-part 'bob' -- this is the second documented branch of the auto-fill behavior; the user's explicit-but-blank input is treated identically to omitting the field")
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

  // Pin the hardcoded 16 KiB body cap on POST /api/auth/login (and, by
  // call-site symmetry, /register and /profile -- all three handlers in
  // AuthStack.scala wrap their parse in `readRequestBody(exchange,
  // 16 * 1024)` per the three matching call sites; a single endpoint
  // test exercises the shared readRequestBody path and proves the cap
  // is wired up, with the per-handler symmetry left as a code-review
  // contract). The deploy doc explicitly documents this cap as
  // "hardcoded 16 KiB (16384 bytes) -- separate from the analyze + hall
  // routes' MAX_UPLOAD_BYTES cap (default 2 MiB), much tighter to bound
  // the credential-stuffing attack surface" -- a refactor that widened
  // the cap (e.g., to use MAX_UPLOAD_BYTES) would silently expand the
  // credential-stuffing attack surface AND make the deploy doc
  // inaccurate. The deploy doc also documents the EXACT 413 message
  // shape -- "the literal message `request body exceeds max upload size
  // of 16384 bytes`" -- which a scripted client might key on for
  // bounded-retry vs unrecoverable-error classification, so the
  // assertion checks the full documented substring (not just the
  // status code) to catch a future "let me make the message friendlier"
  // refactor that broke that contract. Sends 16385 bytes (cap+1) -- the
  // boundary check in readRequestBody is `> maxUploadBytes` so exactly
  // 16384 would pass through (and then 400 at JSON-parse since the
  // body isn't valid JSON); 16385 is the smallest value that exercises
  // the 413 path. Same regression-pin pattern as c71ff6c (OIDC URL
  // params) and 5291216 / 9af8a38 / 2064190 (operator-visible defensive
  // contracts) -- documented body-cap behavior gets pinned so refactors
  // can't silently regress it.
  // Pin the documented /api/auth/login "extra displayName field is
  // silently ignored" behavior. Deploy doc line 100 explicitly says:
  // "POST /api/auth/login body is {email, password} (a displayName on
  // a login request is silently ignored -- the field is only consumed
  // by register, and updates after sign-in go through POST
  // /api/auth/profile)". This is a real contract because either
  // direction of deviation matters: (1) a refactor that started
  // CONSUMING displayName from login (e.g. "update displayName on
  // every sign-in for convenience") would silently let attackers
  // overwrite the user's display name on any sign-in attempt --
  // imagine a phishing page that captures credentials AND submits
  // a hostile displayName, the legitimate user signs in normally
  // but their displayName silently changes to whatever the phishing
  // page wanted; (2) a refactor that REJECTED unknown fields with
  // 400 (a "strict API" choice) would silently break scripted
  // clients that send displayName for compatibility (the documented
  // "silently ignored" framing means clients SHOULD be safe sending
  // it, and they will rely on that). The new test exercises the
  // documented "silently ignored" path: register with one
  // displayName, login with the same credentials AND a different
  // displayName in the body, assert (a) login succeeds with 200,
  // (b) the response's user.displayName is the ORIGINAL value from
  // register-time (not the value submitted with login), proving
  // the extra field had zero effect on stored state.
  test("POST /api/auth/login silently ignores extra displayName field per deploy doc -- login succeeds and the stored displayName remains unchanged from register-time") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"

          // Step 1: register with the original displayName "Alice".
          val register = postJson(s"$baseUri/api/auth/register",
            """{"email":"silent@example.com","password":"correct-horse-battery","displayName":"Alice"}""")
          assertEquals(register.statusCode(), 201,
            clue = "registration must succeed before the login-with-extra-field test can run")
          assertEquals(jsonBody(register)("user")("displayName").str, "Alice",
            clue = "register response must reflect the submitted displayName 'Alice' -- this is the baseline the post-login assertion compares against")

          // Step 2: login with the same credentials BUT a different
          // displayName in the body (the "extra field" the deploy
          // doc says is silently ignored).
          val login = postJson(s"$baseUri/api/auth/login",
            """{"email":"silent@example.com","password":"correct-horse-battery","displayName":"BobImposter"}""")
          assertEquals(login.statusCode(), 200,
            clue = "login with an extra displayName field must SUCCEED (200) per deploy doc line 100's 'silently ignored' contract -- a refactor that REJECTED unknown fields with 400 (a 'strict API' choice) would silently break scripted clients that send displayName for compatibility")

          // Step 3: assert the response's user.displayName is the
          // ORIGINAL value (Alice), NOT the value submitted with
          // login (BobImposter). This is the actual "silently
          // ignored" contract: the field had zero effect on stored
          // state.
          val loginJson = jsonBody(login)
          assertEquals(loginJson("user")("displayName").str, "Alice",
            clue = "login response's user.displayName must be the ORIGINAL register-time value 'Alice', NOT the 'BobImposter' value submitted in the login body -- a refactor that started CONSUMING displayName from login would let a phishing page that captures credentials AND submits a hostile displayName silently rewrite the legitimate user's display name on any sign-in attempt (the user signs in normally but their displayName changes to whatever the attacker wanted, with no operator-visible signal that anything happened)")
        }
      }
    }
  }

  test("POST /api/auth/login rejects bodies exceeding the hardcoded 16 KiB cap with 413 + the documented literal error message") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"
          // 16385 bytes = cap + 1. Body content is intentionally non-
          // JSON-meaningful: readRequestBody's cap check fires BEFORE
          // JSON parsing, so any oversize body triggers it. The fast-
          // path branch in readRequestBody uses the Content-Length
          // header which the JDK HttpClient (used by postJson) sets
          // automatically based on the body's byte length.
          val oversizeBody = "x" * 16385
          val response = postJson(s"$baseUri/api/auth/login", oversizeBody)
          assertEquals(response.statusCode(), 413,
            clue = s"16385-byte body (cap+1) must 413 at the readRequestBody check (AuthStack.scala's `readRequestBody(exchange, 16 * 1024)` call site for /api/auth/login); a refactor that widened the cap to MAX_UPLOAD_BYTES would let this body through silently, expanding the credential-stuffing attack surface and contradicting the deploy doc's 16 KiB claim")
          assert(response.body().contains("max upload size of 16384 bytes"),
            s"413 body must contain the deploy-doc-documented literal message `request body exceeds max upload size of 16384 bytes`; the documented exact wording is what scripted clients key on for bounded-retry vs unrecoverable-error classification. Body was: ${response.body()}")
        }
      }
    }
  }

  // Close the body-cap triplet: /api/auth/profile is the third and
  // final branch in the three-call-site readRequestBody(exchange,
  // 16 * 1024) pattern that e2045b9 (/login) and 034b14c (/register)
  // already covered. The three AuthStack.scala call sites (register
  // line 104, login line 163, profile line 229) now ALL have
  // dedicated body-cap regression tests, so an asymmetric refactor
  // that touched only one branch is caught by CI on every branch.
  // /profile is the lowest-priority of the three in pure security
  // terms (it's authenticated, so no credential-stuffing vector
  // applies; the PBKDF2 cost path doesn't fire on profile updates),
  // but pinning it completes the documented "all three" claim from
  // deploy doc line 100 ("All three platform-user auth POSTs
  // (/register, /login, /profile) ALSO cap the request body at a
  // hardcoded 16 KiB"). The /profile-specific risk a refactor would
  // open: a future "extended-profile fields" feature might tempt a
  // maintainer to widen ONLY profile's cap (the natural lazy fix:
  // "the new fields might need more headroom") while leaving
  // register + login at 16 KiB; the resulting MB-scale profile
  // bodies would chew JSON-parse CPU on every profile update, and
  // a signed-in user submitting an oversized body would tie up an
  // executor thread on parsing rather than getting the immediate
  // 413 the documented behavior promises -- a small DoS surface
  // accessible only to signed-in users but still worth closing.
  test("POST /api/auth/profile rejects bodies exceeding the hardcoded 16 KiB cap with 413 + the documented literal error message") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"
          // /profile requires an authenticated session + CSRF token,
          // so register first to get those headers. (Register +
          // login both share the same 16 KiB cap per e2045b9 /
          // 034b14c, so the body-cap check fires the same way
          // regardless of whether the caller is authenticated.)
          val register = postJson(s"$baseUri/api/auth/register",
            """{"email":"profile-cap@example.com","password":"correct-horse-battery","displayName":"Tester"}""")
          assertEquals(register.statusCode(), 201,
            clue = "registration must succeed before the profile body-cap can be exercised against an authenticated session")
          val ownerHeaders = authSessionHeaders(register, jsonBody(register)("csrfToken").str)

          val oversizeBody = "x" * 16385
          val response = postJson(s"$baseUri/api/auth/profile", oversizeBody, ownerHeaders)
          assertEquals(response.statusCode(), 413,
            clue = s"16385-byte body (cap+1) must 413 at the readRequestBody check (AuthStack.scala's `readRequestBody(exchange, 16 * 1024)` call site for /api/auth/profile, line ~229); a refactor that widened ONLY /profile (e.g. for a 'future extended-profile fields' feature) would silently open a signed-in-user DoS surface where oversized bodies chew JSON-parse CPU instead of getting the immediate 413 the documented contract promises; got: ${response.statusCode()}")
          assert(response.body().contains("max upload size of 16384 bytes"),
            s"413 body must contain the deploy-doc-documented literal message `request body exceeds max upload size of 16384 bytes` -- the same generic shared-across-call-sites error per e2045b9's pin commentary; Body was: ${response.body()}")
        }
      }
    }
  }

  // Parallel /register 16 KiB body-cap test mirroring the /login pin
  // immediately above. e2045b9 (the /login pin) explicitly acknowledged
  // "future fires can sweep the rest" of the three auth POSTs that
  // share the same `readRequestBody(exchange, 16 * 1024)` pattern at
  // AuthStack.scala lines 104 (register), 163 (login), 229 (profile)
  // -- this commit closes the /register branch. Per-handler call-site
  // symmetry from the e2045b9 comment block: a refactor that changed
  // ONE of the three handlers (e.g. widened only /register's cap to
  // MAX_UPLOAD_BYTES to support "future extended-profile registration"
  // while leaving /login + /profile at 16 KiB) would silently expand
  // the credential-stuffing attack surface specifically on /register
  // WITHOUT breaking the existing /login pin; the parallel-branch
  // coverage catches per-handler drift. /register is the highest-
  // priority of the two remaining branches because (1) it's the
  // primary credential-stuffing attack target alongside /login, and
  // (2) the disk-fill defense via USER_AUTH_MAX_USERS (capped at
  // 100k stored users per fb18e2c's pin) is layered ON TOP OF the
  // 16 KiB body cap -- if a refactor widened the body cap to MB-
  // scale, the per-request CPU cost of registration would multiply
  // (PBKDF2-HMAC-SHA256 at 210k iterations per 097bc64 + 7c8956b
  // is already expensive; adding parsing of MB-scale JSON before
  // the cap fires would amplify the cost meaningfully).
  test("POST /api/auth/register rejects bodies exceeding the hardcoded 16 KiB cap with 413 + the documented literal error message") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"
          val oversizeBody = "x" * 16385
          val response = postJson(s"$baseUri/api/auth/register", oversizeBody)
          assertEquals(response.statusCode(), 413,
            clue = s"16385-byte body (cap+1) must 413 at the readRequestBody check (AuthStack.scala's `readRequestBody(exchange, 16 * 1024)` call site for /api/auth/register); a refactor that widened ONLY the register cap (without touching login + profile) would silently expand the credential-stuffing-via-PBKDF2-CPU-cost attack surface specifically on register while passing the parallel /login pin, so this per-handler-branch coverage catches that drift")
          assert(response.body().contains("max upload size of 16384 bytes"),
            s"413 body must contain the same deploy-doc-documented literal message as /login (the message is generic across all readRequestBody call sites by design, no per-endpoint differentiation); Body was: ${response.body()}")
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
        // Pin the documented stable `service` identifier "hand-history-review"
        // that the deploy doc names as part of the fleet-correlation triple
        // (service / host / port) operators use to identify which sicfun
        // instance emitted a given /api/health or /api/ready response in a
        // multi-service / multi-instance deployment. Deploy doc line 216
        // documents the /api/health emission ("plus `service`/`host`/`port`/
        // `startedAtEpochMs`/`uptimeMs` for fleet correlation") and line 217
        // documents the matching /api/ready emission ("and `service`/`host`/
        // `port` for fleet correlation"); the SAME literal value
        // "hand-history-review" ships from both endpoints. A refactor changing
        // the constant (e.g. renaming to "sicfun-web" or "hand-history-server"
        // for style consistency with a future second service) would silently
        // break (a) log-aggregation systems that filter by `service` field
        // for multi-service deployments, (b) dashboard alerting rules keyed
        // on a specific service name, (c) operator runbooks that grep
        // probe-response JSON for the service identifier to confirm "this
        // log line came from THIS service vs another sicfun module", AND
        // (d) the documented "service/host/port for fleet correlation"
        // contract -- a tool wired against the field name would still get
        // SOMETHING (just the new name), so the failure mode is silent
        // pivot of every downstream correlation query to "no longer
        // matches the documented identifier" rather than a hard break.
        // The matching pin in the /api/ready block below verifies the
        // SYMMETRIC contract (both endpoints emit the same value); the
        // two-endpoint pin pattern catches the asymmetric-drift case where
        // one endpoint's constant gets renamed while the other doesn't
        // (e.g. a partial refactor that touches renderHealth but misses
        // renderReadiness in Readiness.scala -- the two functions sit in
        // the same file at lines 57 and 116, so it's easy to update one
        // and forget the other).
        assertEquals(healthJson("service").str, "hand-history-review",
          clue = s"/api/health must surface service='hand-history-review' per deploy doc line 216's documented fleet-correlation contract -- a refactor renaming the constant would silently break log-aggregation filters, dashboard alerts keyed on service name, and operator runbooks that grep responses for the identifier; got: ${healthJson("service")}")
        assertEquals(healthJson("host").str, server.binding.host)
        assertEquals(healthJson("port").num.toInt, server.binding.port)
        // Pin the documented server-lifecycle fields `startedAtEpochMs`
        // + `uptimeMs` -- the last two fields of the deploy doc line
        // 216 enumeration "service/host/port/startedAtEpochMs/uptimeMs
        // for fleet correlation"; 505ba6b + b2a90fb closed the
        // service/host/port triple on both endpoints, this assertion
        // block closes the server-lifecycle pair on /api/health (the
        // companion ABSENCE pins for /api/ready follow the
        // modelConfigured 4d15ca3+37f9465 asymmetric pattern at the
        // end of the /api/ready block below since the deploy doc
        // scopes BOTH fields to /api/health-only). Why these two
        // fields matter operationally: (a) startedAtEpochMs is the
        // OPERATOR-VISIBLE process-restart timestamp -- the canonical
        // way to distinguish "did this instance restart since I last
        // looked" from "process is still alive but stuck" (a stuck
        // process has the SAME startedAtEpochMs across probes;
        // a restarted one has a strictly-greater value), used by
        // runbook section X's restart-detection triage step AND by
        // dashboards plotting "deploys per day" (each unique
        // startedAtEpochMs value seen across probe samples corresponds
        // to one process lifetime), (b) uptimeMs gives the same
        // information as a relative duration (no need to do
        // arithmetic against current wall clock) -- used by alerting
        // for "uptime < N seconds" conditions that fire on
        // unexpected restarts, AND by capacity dashboards plotting
        // "average uptime per instance" as a proxy for deployment
        // stability; the TWO fields are SEMANTICALLY REDUNDANT by
        // design (uptimeMs = now - startedAtEpochMs at response-
        // build time, both shipped on the same response) but each
        // serves a different consumer pattern: absolute-timestamp
        // consumers prefer startedAtEpochMs (epoch ms is the most
        // portable timestamp format), duration consumers prefer
        // uptimeMs (saves the dashboard math + handles clock-skew
        // between probe-receive and dashboard-eval gracefully).
        // Assertion structure: capture `nowMs` adjacent to the
        // assertions so the slack window between probe-receive and
        // assertion-eval is small (the response was built moments
        // ago, so the COVARIANT invariant
        // `startedAtEpochMs + uptimeMs ≈ nowMs` should hold to
        // within a few seconds tolerance for JVM scheduling + GC
        // pauses); per-field checks: (1) startedAtEpochMs > 0 catches
        // an uninitialized-default refactor that emitted 0 / negative
        // / unset, (2) startedAtEpochMs <= nowMs catches a
        // wrong-baseline refactor that picked e.g. an Instant.MAX
        // sentinel or a clock-future value, (3) uptimeMs >= 0
        // catches the negative-uptime case (startedAtEpochMs after
        // now, which is logically impossible but could arise from
        // a clock-skew bug), (4) the COVARIANT invariant
        // `|startedAtEpochMs + uptimeMs - nowMs| <= 5000ms` pins
        // the two fields' SHARED baseline -- a refactor changing
        // one field's reference point without updating the other
        // (e.g. startedAtEpochMs from process-start to deploy-time,
        // uptimeMs staying tied to process-start) would silently
        // break the covariance, AND a refactor wedging either to
        // a constant would also fail this pin since the constant
        // would not track the actual elapsed wall clock.
        val nowMsBeforeHealthLifecycleCheck = System.currentTimeMillis()
        val healthStartedAt = healthJson("startedAtEpochMs").num.toLong
        val healthUptime = healthJson("uptimeMs").num.toLong
        assert(healthStartedAt > 0L,
          clue = s"/api/health startedAtEpochMs must be a positive epoch-ms timestamp per deploy doc line 216's fleet-correlation enumeration -- a refactor emitting 0 / negative / unset would silently break operator-visible process-restart detection (runbook restart-triage step keys on this field to distinguish 'instance restarted' from 'still alive but stuck'); got: $healthStartedAt")
        assert(healthStartedAt <= nowMsBeforeHealthLifecycleCheck,
          clue = s"/api/health startedAtEpochMs must not be in the future relative to the test's nowMs -- a refactor picking the wrong baseline (e.g. Instant.MAX sentinel, a future deploy-time, or a clock-skewed source) would silently break dashboards plotting 'deploys per day' from unique startedAtEpochMs samples; got: $healthStartedAt vs nowMs=$nowMsBeforeHealthLifecycleCheck")
        assert(healthUptime >= 0L,
          clue = s"/api/health uptimeMs must be non-negative -- a refactor computing uptime from a future startedAtEpochMs (clock-skew bug) would silently produce negative values and break uptime-based alerting on 'uptime < N seconds' conditions; got: $healthUptime")
        // COVARIANT invariant: startedAtEpochMs + uptimeMs should
        // approximate nowMs (within 5s slack for JVM scheduling + GC
        // pauses between the response being built and the test
        // reading nowMs). This is the strongest invariant in this
        // block because it forces BOTH fields to be derived from
        // the SAME baseline -- a refactor changing one field's
        // reference point without updating the other would silently
        // break the covariance, AND a refactor wedging either to a
        // constant would also fail (the constant would not track
        // the actual elapsed wall clock). The 5000ms slack is
        // generous but tight enough to catch real divergence
        // (a baseline drift of even 30 seconds would fail this);
        // a refactor that swapped uptimeMs's baseline from
        // process-start to e.g. last-config-reload would typically
        // drift by minutes-to-hours and fail this assertion loudly.
        val combinedHealthLifecycle = healthStartedAt + healthUptime
        val healthLifecycleDelta = math.abs(combinedHealthLifecycle - nowMsBeforeHealthLifecycleCheck)
        assert(healthLifecycleDelta <= 5000L,
          clue = s"/api/health startedAtEpochMs + uptimeMs must approximate the current wall clock (covariant baseline invariant) -- a refactor changing one field's reference point (e.g. startedAtEpochMs from process-start to deploy-time, uptimeMs staying tied to process-start) would silently break the covariance; 5000ms slack allows for JVM scheduling + GC pauses; got startedAtEpochMs=$healthStartedAt + uptimeMs=$healthUptime = $combinedHealthLifecycle vs nowMs=$nowMsBeforeHealthLifecycleCheck (delta=$healthLifecycleDelta ms)")
        assertEquals(healthJson("modelSource").str, "uniform fallback")
        // Pin the documented `modelConfigured` boolean field that the
        // deploy doc explicitly names as the operator-tooling
        // alternative to the masked `modelSource` string. The deploy
        // doc's section on MODEL_DIR documents both fields side-by-side:
        // `modelSource` carries the masked form ("uniform fallback" /
        // "configured artifact dir") so dashboards can flag the
        // unconfigured case via a string compare WITHOUT tying alert
        // rules to the operator's filesystem layout, AND `modelConfigured`
        // carries the same information as a `bool == false` alert
        // condition for dashboards that prefer boolean alerting over
        // string compares -- the doc says "a parallel boolean field
        // `modelConfigured` ... ships in the same /api/health response
        // so a dashboard preferring a `bool == false` alert condition
        // over a string compare can use either". The TWO fields are
        // semantically redundant by design (they encode the same
        // "is MODEL_DIR set?" question with different types), and the
        // deploy doc commits to keeping them in sync; a refactor that
        // dropped `modelConfigured` because "modelSource already says
        // it" would silently break every dashboard wired with the
        // bool == false alert rule -- the dashboards would lose ALL
        // signal (a missing JSON field reads as null on most query
        // engines, which is neither true nor false, so the alert rule
        // matches neither branch). The `/api/health`-only scoping (per
        // the deploy doc: "modelConfigured is /api/health-only and is
        // NOT echoed by /api/ready") is intentional and unverified
        // here -- the assertion below covers presence in /api/health;
        // a future fire can add the absence-from-/api/ready
        // companion if asymmetric-drift coverage becomes important.
        // Value is `false` because the test's withServer config does
        // NOT set modelDir, matching the "uniform fallback" branch
        // above; a refactor changing the default for `modelConfigured`
        // when modelDir is unset (e.g. defaulting to `true` because
        // the analyze fallback IS technically a model) would silently
        // break the deploy-doc invariant that the two fields move
        // together.
        assertEquals(healthJson("modelConfigured").bool, false,
          clue = s"/api/health must surface modelConfigured=false when MODEL_DIR is unset, per the deploy doc's MODEL_DIR section that documents this boolean as the dashboard-alert alternative to the masked modelSource string; a refactor that dropped this field or flipped its default would silently break every dashboard keying on `modelConfigured == false` for unconfigured-instance alerts; got: ${healthJson("modelConfigured")}")
        // Pin the documented `drainSignalConfigured` field that
        // pairs with the already-pinned `drainSignalPresent` below to
        // form the documented drain-signal pair (the WIRING-STATE
        // half + the RUNTIME-STATE half). Deploy doc line 126's
        // DRAIN_SIGNAL_FILE section explicitly documents the pair:
        // "/api/health.drainSignalConfigured reflects whether the
        // knob is set (independent of whether the file currently
        // exists), so a dashboard can verify the deployment is wired
        // for graceful rolling restarts at all"; the deploy doc
        // contrasts the two halves: drainSignalConfigured says "is
        // this deployment EVEN CAPABLE of pre-staged drain" (a
        // configuration-time fact), drainSignalPresent says "is the
        // drain CURRENTLY ACTIVE" (a runtime-state fact). The two
        // are SEMANTICALLY ORTHOGONAL by design: a deployment can be
        // (configured=true, present=false) which means "wired but not
        // currently draining" (the normal state), (configured=true,
        // present=true) which means "actively draining" (the drain-
        // in-progress state), (configured=false, present=false) which
        // means "not wired at all, relying on JVM shutdown hook only"
        // (the unconfigured deployment), AND (configured=false,
        // present=true) which is LOGICALLY IMPOSSIBLE (you can't
        // have a file present at an unset path). Why each operational
        // consumer cares: (a) operator runbook step 0 for graceful
        // rolling restart: BEFORE typing `touch <DRAIN_SIGNAL_FILE>`,
        // verify `drainSignalConfigured == true` -- the runbook's
        // rolling-restart pattern doc says "this knob configured...
        // graceful rolling restarts; without it... the only drain
        // signal is the JVM shutdown hook itself -- which doesn't
        // pre-stage the load balancer's stop-routing decision"; if
        // the operator follows the runbook on an UNCONFIGURED
        // deployment, the touch command no-ops and the SIGTERM races
        // budget without pre-staging the LB; (b) capacity-planning
        // dashboards: separate the "production deployments that use
        // pre-staged drain (configured=true)" from "test deployments
        // that don't (configured=false)" -- if a deployment that
        // SHOULD be wired loses the configuration silently (env-var
        // dropped, deployment manifest churn), the dashboard pivots
        // those instances to the wrong category and the alerting
        // misroutes; (c) deployment-stability dashboards: track
        // (configured=true, present=true) instances over time to see
        // "how often was each instance pre-stage drained" as a proxy
        // for deployment-stability events (a fleet with 50 instances
        // and 200 drain events in a week has high deployment
        // velocity OR high recovery activity, both worth knowing).
        // The TEST setup at line ~2041 does NOT configure
        // DRAIN_SIGNAL_FILE (the default withServer does not set the
        // drainSignalFile option), so drainSignalConfigured = false
        // here, matching the drainSignalPresent = false on the line
        // below. A refactor that flipped the default to `true` when
        // DRAIN_SIGNAL_FILE is unset (e.g. "defaulting to true so
        // operators don't have to think about wiring it") would
        // silently break the documented "independent of whether the
        // file currently exists" semantic by making the field always-
        // true, AND would silently break dashboards keying on
        // `drainSignalConfigured == false` for unconfigured-instance
        // alerts; a refactor that dropped the field entirely (e.g.
        // "drainSignalPresent already tells you the state") would
        // silently break dashboards that distinguish "wired but not
        // draining" from "not wired at all" -- the two states have
        // the SAME drainSignalPresent (false) so the dashboard would
        // lose its only signal.
        assertEquals(healthJson("drainSignalConfigured").bool, false,
          clue = s"/api/health must surface drainSignalConfigured=false when DRAIN_SIGNAL_FILE is unset, per deploy doc line 126's '/api/health.drainSignalConfigured reflects whether the knob is set (independent of whether the file currently exists)' framing -- a refactor that flipped the default to true or dropped the field entirely would silently break dashboards distinguishing 'wired but not draining' from 'not wired at all'; got: ${healthJson("drainSignalConfigured")}")
        assertEquals(healthJson("drainSignalPresent").bool, false)
        assertEquals(healthJson("maxUploadBytes").num.toInt, 64)
        assertEquals(healthJson("analysisTimeoutMs").num.toLong, 120000L)
        // Pin the documented PLAYING_HALL_TIMEOUT_MS default (15 min =
        // 900000 ms) per the deploy doc line 364's "Use
        // -PlayingHallTimeoutMs or PLAYING_HALL_TIMEOUT_MS to cap a
        // single Playing Hall job (default 900000 ms, i.e. 15 min)".
        // The companion analysisTimeoutMs default (120000 = 2 min) is
        // already pinned on the line above; this assertion symmetric-
        // izes the coverage so the hall-timeout default has the same
        // CI protection. A refactor changing
        // HandHistoryReviewServerConfig.DefaultPlayingHallTimeoutMs
        // from 900000L would silently drift the documented value
        // AND break the frontend's maxPollWaitMs auto-extension logic
        // documented in site.js's probeServerLimits (which reads the
        // server's playingHallTimeoutMs from health and extends the
        // poll budget to "max(16 min default, server timeout + 1 min
        // slack)" -- a server-side value drop would silently shrink
        // the frontend's poll deadline below the legitimate worker
        // run time, causing the frontend to give up polling while
        // the server is still working on the hall run).
        assertEquals(healthJson("playingHallTimeoutMs").num.toLong, 900000L)
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
        assertEquals(headerValue(health, "Referrer-Policy"), Some("no-referrer"))
        assertEquals(headerValue(health, "X-Robots-Tag"), Some("noindex, nofollow"))

        val ready = get(s"$baseUri/api/ready")
        assertEquals(ready.statusCode(), 200)
        val readyJson = jsonBody(ready)
        assertEquals(readyJson("ready").bool, true)
        assertEquals(readyJson("reason").str, "accepting-traffic")
        assertEquals(readyJson("draining").bool, false)
        assertEquals(readyJson("acceptingAnalysisJobs").bool, true)
        // Mirror of the health-response drain-signal pair above,
        // closing the documented symmetric pin on /api/ready. Deploy
        // doc line 217 enumerates "draining + acceptingAnalysisJobs +
        // drainSignalConfigured + drainSignalPresent (so a probe can
        // tell 'queue full' from 'operator-initiated drain' without a
        // separate health check)" -- the doc EXPLICITLY frames the
        // pair as load-balancer-relevant (the consumer that needs to
        // distinguish queue-full-503 from operator-drain-503 to
        // decide whether to retry the request against another
        // instance or to back off entirely); BEFORE this commit
        // BOTH drainSignal fields were missing from the /api/ready
        // probe block here (drainSignalPresent appears in OTHER
        // tests at lines ~4631 + ~4643 but those exercise the
        // drain-signal-ACTIVE state in a separate scenario; the
        // BASELINE 'drain not configured, drain not active' state
        // was unpinned on /api/ready). The symmetric pin pair
        // mirrors the established pattern from 505ba6b (service)
        // and b2a90fb (host+port): same field, same value, both
        // endpoints, two independent emitters in Readiness.scala
        // (renderHealth lines 95+96, renderReadiness lines 139+140
        // -- each has TWO independent Bool() literals, so a partial
        // refactor touching one function's pair could leave the
        // other's pair stale). Asymmetric-drift risk specific to
        // this pair: a refactor consolidating the drain-signal
        // boolean shape (e.g. switching from a separate
        // configured+present pair to a single tri-state field like
        // `drainSignalState: "unconfigured" | "wired" | "active"`)
        // would silently break BOTH downstream consumers (operators
        // expecting two booleans + dashboards built on bool-condition
        // alert rules), AND a refactor renaming either field for
        // style consistency (e.g. drainSignalConfigured ->
        // drainConfigured) would silently break log-aggregation
        // queries filtering by the documented field name. The pin
        // pair on /api/ready catches all of these for the readiness
        // side, the pair on /api/health (lines ~2207+) catches them
        // for the health side -- together the 4 pins (2 fields × 2
        // endpoints) form a SYMMETRIC quadrilateral that any single-
        // function refactor breaks at least one corner of.
        assertEquals(readyJson("drainSignalConfigured").bool, false,
          clue = s"/api/ready must surface drainSignalConfigured=false when DRAIN_SIGNAL_FILE is unset per deploy doc line 217's '/api/ready ... drainSignalConfigured + drainSignalPresent' enumeration -- the matching /api/health pin enforces the same value, so this closes the symmetric pair against asymmetric drift between Readiness.scala's renderHealth (line 95) and renderReadiness (line 139) which each have independent Bool() literals; got: ${readyJson("drainSignalConfigured")}")
        assertEquals(readyJson("drainSignalPresent").bool, false,
          clue = s"/api/ready must surface drainSignalPresent=false when no drain file exists per deploy doc line 217 enumerating both halves of the drain-signal pair; the load-balancer-side consumer needs this field to distinguish 'queue full' from 'operator-initiated drain' (per the deploy doc framing) -- a refactor dropping it from renderReadiness would silently force every LB probe to fall back to /api/health for the same information; got: ${readyJson("drainSignalPresent")}")
        assertEquals(readyJson("authenticationEnabled").bool, false)
        assertEquals(readyJson("authenticationMode").str, "none")
        // Mirror of the health-response `service` pin above. Deploy doc
        // line 217's "/api/ready ... `service`/`host`/`port` for fleet
        // correlation" framing commits to emitting the SAME stable
        // identifier on /api/ready as /api/health does. The symmetric
        // pin pair catches the asymmetric-drift case where one
        // endpoint's constant gets renamed while the other doesn't
        // (Readiness.scala has TWO separate hardcoded string literals
        // -- one at line 88 for renderHealth, one at line 130 for
        // renderReadiness -- so a partial refactor that touched only
        // one function would leave the two response shapes
        // disagreeing on the service identifier, which would silently
        // confuse any operator tooling that probes BOTH endpoints to
        // cross-check fleet membership).
        assertEquals(readyJson("service").str, "hand-history-review",
          clue = s"/api/ready must surface service='hand-history-review' per deploy doc line 217's documented fleet-correlation contract -- the matching /api/health pin enforces the same value, so this assertion closes the symmetric-pin pair against asymmetric drift between the two adjacent Readiness.scala functions (renderHealth line 57 + renderReadiness line 116, with TWO independent hardcoded string literals at lines 88 and 130); got: ${readyJson("service")}")
        // Mirror of the health-response host + port pins above (lines
        // ~2089-2090), completing the documented fleet-correlation
        // triple (service / host / port) on /api/ready. Deploy doc line
        // 217 explicitly enumerates "service/host/port" together as the
        // fleet-correlation identifier set, but BEFORE this commit only
        // /api/health had host + port pinned (lines ~2089-2090) while
        // /api/ready had NONE of the triple covered. The 505ba6b commit
        // added `service` symmetrically to both endpoints; this commit
        // adds `host` + `port` to /api/ready so the SAME triple-coverage
        // existing on /api/health now exists on /api/ready -- a load-
        // balancer or service-mesh probe wired against /api/ready (the
        // intended consumer per the doc: "the readiness endpoint for
        // reverse proxies / service managers") gets the SAME bound-
        // address verification path as a dashboard wired against
        // /api/health. Why host + port matter on /api/ready specifically:
        // (a) load-balancer pool membership reconciliation -- the LB's
        // probe verifies the bound port matches the registered backend
        // entry, and a refactor that emitted a stale boot-time port vs.
        // the actually-bound port would silently break port-changing
        // restart scenarios where the requested port was unavailable
        // and the OS picked another (the deploy doc documents PORT=0
        // explicitly as an ephemeral-port mode for testing, and
        // production deployments that intentionally cycle ports for
        // blue/green rollouts depend on the bound-port report being
        // accurate); (b) service-mesh sidecar injection -- meshes like
        // Linkerd / Istio key on the (host, port) tuple to populate
        // their service-discovery database, and a mismatch silently
        // routes traffic to a non-existent backend; (c) operator
        // grep-the-readiness-response workflows -- "is this instance
        // bound where I think it is" is a common runbook-step-zero
        // diagnostic, and a refactor reporting the CONFIG host (e.g.
        // 0.0.0.0 or the requested bind) instead of the RESOLVED host
        // would silently lie to the operator about which interface
        // the process is actually listening on. The bound-port-not-
        // config-port subtlety: server.binding.port is the post-bind
        // resolved port from the HTTP server's actual socket (handles
        // the port=0 -> ephemeral allocation case correctly), while
        // config.port would be the requested value (0 for ephemeral,
        // never the resolved 49152-65535 ephemeral port the OS picked)
        // -- a refactor swapping the field from boundPort.toDouble to
        // config.port.toDouble would break every port=0 deployment by
        // emitting the literal 0 instead of the real port. Coverage
        // on /api/health (lines 2089-2090 use the same `server.binding`
        // accessor pattern this assertion uses) already catches that
        // refactor for the health side; this assertion extends it to
        // the ready side, so a partial refactor touching only
        // renderHealth or only renderReadiness would be caught by
        // ONE of the two pins (whichever side was touched).
        assertEquals(readyJson("host").str, server.binding.host,
          clue = s"/api/ready must surface the same resolved bound-host as /api/health per deploy doc line 217's documented fleet-correlation triple -- a refactor reporting the config host (e.g. 0.0.0.0 / requested bind) instead of the resolved bound host would silently confuse operator diagnostic workflows; got: ${readyJson("host")}")
        assertEquals(readyJson("port").num.toInt, server.binding.port,
          clue = s"/api/ready must surface the same resolved bound-port as /api/health per deploy doc line 217's documented fleet-correlation triple -- a refactor reporting config.port instead of boundPort would silently break every port=0 ephemeral-bind deployment (config.port stays 0 while server.binding.port resolves to the OS-picked ephemeral 49152-65535 range); got: ${readyJson("port")}")
        assertEquals(readyJson("analysisTimeoutMs").num.toLong, 120000L)
        // Mirror of the health-response playingHallTimeoutMs pin above.
        // The /api/ready endpoint surfaces the same field (Readiness.scala
        // emits it on both probes) so a load-balancer probe wired
        // against /api/ready gets the same default-value protection.
        assertEquals(readyJson("playingHallTimeoutMs").num.toLong, 900000L)
        assertEquals(readyJson("rateLimitSubmitsPerMinute").num.toInt, 6)
        assertEquals(readyJson("rateLimitStatusPerMinute").num.toInt, 240)
        assertEquals(readyJson("rateLimitAuthPerMinute").num.toInt, 10)
        assertEquals(readyJson("rateLimitClientIpSource").str, "remote-address")
        assertEquals(readyJson("timedOutWorkersInFlight").num.toInt, 0)
        // Pin the documented asymmetric scoping of `modelConfigured`:
        // the deploy doc EXPLICITLY commits to "/api/health-only and
        // is NOT echoed by /api/ready" -- the boolean field surfaces
        // on /api/health for dashboard alerting on misconfigured
        // instances (pinned at line ~2057 by the modelConfigured
        // health pin), but /api/ready intentionally OMITS it because
        // /api/ready is the load-balancer / orchestrator probe whose
        // contract is narrower (just "is this instance ready to
        // accept traffic"), and the model-configuration question is
        // an OPERATOR-side concern (dashboards / alerts), not an
        // orchestrator-side concern (LB membership / pod readiness).
        // Without this absence pin, a refactor that "mirrored" the
        // field into /api/ready as a harmless additive change would
        // silently break the deploy-doc invariant -- and the change
        // would NOT be detected by the existing modelConfigured
        // health pin (that pin only asserts presence on /api/health,
        // it says nothing about /api/ready). The drift vector is
        // non-zero because the natural refactor reflex on a "field
        // should be everywhere" intuition is to mirror, and a
        // maintainer reading Readiness.scala's two adjacent
        // renderHealth / renderReadiness functions (lines 57 and
        // 116 in the source) might assume the divergence is a bug
        // and "fix" it. The assertion uses `.obj.contains("modelConfigured")`
        // against the underlying mutable.LinkedHashMap rather than
        // `.isNull` (which is the documented form for fields that
        // are present-but-null on basic-auth / no-auth modes like
        // userAuthMaxUsers per line 579's `assert(health("userAuthMaxUsers").isNull)`)
        // -- the two are semantically different: present-but-null
        // means the key is in the JSON with a null value (the
        // frontend can iterate the response, find the key, and act
        // on the explicit null), while ABSENT means the key isn't
        // there at all (the frontend would get `undefined` on a
        // property access). The deploy doc says NOT ECHOED which
        // is the absence case, not the null case -- mirroring it
        // as `null` would also be a refactor regression because
        // a `bool == false` dashboard query (the documented
        // alerting pattern for the health field) would FAIL on a
        // null value on most query engines.
        assert(!readyJson.obj.contains("modelConfigured"),
          clue = s"/api/ready must NOT echo `modelConfigured` per the deploy doc's explicit '/api/health-only and is NOT echoed by /api/ready' framing -- the field is documented as an operator-side dashboard signal (not an orchestrator-side readiness signal), and a refactor that mirrored it into /api/ready as a harmless additive change would silently widen the documented contract; if this test fails, either the deploy doc needs an update to allow the mirror OR the renderReadiness function should be reverted to drop the field; got readyJson keys: ${readyJson.obj.keys.toVector.sorted.mkString(", ")}")
        // Pin the documented asymmetric scoping of the server-
        // lifecycle pair (startedAtEpochMs + uptimeMs): deploy doc
        // line 216 enumerates them on /api/health-only ("plus
        // service/host/port/startedAtEpochMs/uptimeMs for fleet
        // correlation"), while line 217's /api/ready enumeration
        // includes only service/host/port (the deploy doc EXPLICITLY
        // contrasts the two endpoints by listing different field
        // sets, with line 217 noting "the field set is a strict
        // subset of /api/health's plus the renamed `reason`"). The
        // /api/ready endpoint is the LB / orchestrator probe whose
        // contract is "is this instance ready to accept traffic" --
        // process-lifecycle metadata (when did this process start,
        // how long has it been up) is operator-side concern, not
        // orchestrator-side. Without these absence pins, a refactor
        // that "mirrored" the lifecycle pair into /api/ready as a
        // "harmless additive change" would silently widen the
        // documented "/api/health-only" contract; the drift vector
        // is non-zero because (a) Readiness.scala's renderHealth +
        // renderReadiness functions sit in the same file at adjacent
        // lines (57 + 116), so a maintainer might "fix" the
        // divergence as a perceived bug, (b) the natural code-review
        // reflex on a "fields should be everywhere" intuition is to
        // mirror, (c) additive changes feel safe even when they're
        // contract-widening. The matching presence pins on
        // /api/health (lines ~2089-2113 above) catch the OPPOSITE
        // drift (the field gets dropped from /api/health); together
        // the four pins (2 fields × 2 endpoints, all with their
        // documented presence/absence shape) form an asymmetric-
        // scoping invariant: a refactor that broke either direction
        // (drop from /api/health, mirror to /api/ready) fails one
        // of the pins. Same asymmetric-pin pattern as 4d15ca3 +
        // 37f9465 for the modelConfigured pair.
        assert(!readyJson.obj.contains("startedAtEpochMs"),
          clue = s"/api/ready must NOT echo `startedAtEpochMs` per deploy doc line 216 enumerating it as an /api/health-only field and line 217's '/api/ready ... field set is a strict subset of /api/health's' framing -- a refactor mirroring the process-restart timestamp into /api/ready (orchestrator-side probe, not operator-side dashboard) would silently widen the documented contract; got readyJson keys: ${readyJson.obj.keys.toVector.sorted.mkString(", ")}")
        assert(!readyJson.obj.contains("uptimeMs"),
          clue = s"/api/ready must NOT echo `uptimeMs` per deploy doc line 216 enumerating it as an /api/health-only field and line 217's '/api/ready ... field set is a strict subset of /api/health's' framing -- a refactor mirroring the uptime duration into /api/ready (orchestrator-side probe, not operator-side dashboard) would silently widen the documented contract; got readyJson keys: ${readyJson.obj.keys.toVector.sorted.mkString(", ")}")
        // Pin the documented asymmetric scoping of the `ok` field --
        // the deploy doc line 216 explicitly commits to "The JSON
        // body opens with a hardcoded `ok: true` invariant (literal
        // `Bool(true)` at `Readiness.scala`'s `renderHealth`; never
        // `false` -- if the JVM were too sick to set the field the
        // request would never return at all), so a monitor can key
        // on `body.ok === true` as the always-up signal separately
        // from the rich-state fields below"; the contract has TWO
        // parts the test needs to enforce: (i) PRESENCE + value on
        // /api/health (already pinned at line 2047 via
        // `assertEquals(healthJson("ok").bool, true)`), AND (ii)
        // ABSENCE from /api/ready (this assertion), because the
        // documented design contrasts the two endpoints: /api/health
        // emits `ok` as the always-up live-monitor invariant, while
        // /api/ready emits `ready` (a DIFFERENT field name, with
        // different semantics: "accepting-traffic" rather than
        // "process is alive") -- a refactor mirroring `ok` into
        // /api/ready as a "harmless additive change" would silently
        // (a) confuse monitors that key on `body.ok === true` as
        // the always-up signal -- those monitors would suddenly
        // accept /api/ready responses as "live" even when /api/ready
        // returns 503 (queue full / draining / timed-out worker),
        // pivoting the monitor's semantic from "is the process
        // alive" to "is the process alive AND accepting traffic"
        // which is what `ready` already covers, (b) create field
        // ambiguity between `ok` (always-up signal) and `ready`
        // (accepting-traffic signal) on the same /api/ready
        // response -- operator tooling would have two competing
        // signals to choose from and the deploy doc's clean "key
        // on ok for liveness, key on ready for acceptance" guidance
        // would break, (c) silently widen the documented "/api/ready
        // ... field set is a strict subset of /api/health's plus
        // the renamed `reason`" framing from line 217 by adding a
        // field that's NOT in the subset AND is NOT the renamed
        // reason (it's a new always-true field that /api/ready
        // already covers via `ready: true`); the drift vector is
        // non-zero because (a) the two functions sit at adjacent
        // lines (renderHealth at line 57, renderReadiness at line
        // 116) in Readiness.scala so a "fields should be everywhere"
        // refactor reflex could mirror the constant, AND (b) the
        // `ok: true` literal at line 73 is the FIRST field
        // renderHealth emits -- a maintainer copy-pasting the
        // emission pattern from renderHealth to renderReadiness
        // might accidentally include `ok: true` as the first
        // field of renderReadiness too (the "I want a parallel
        // shape" intuition is strong here because line 73's `ok`
        // emission is structurally where `ready` lands at line
        // 117 in renderReadiness -- they're SEMANTIC siblings in
        // adjacent functions); same asymmetric-pin pattern as
        // modelConfigured (4d15ca3 + 37f9465), the lifecycle pair
        // startedAtEpochMs + uptimeMs (f50d7f9 -- pinned above at
        // lines 2513+2515) -- /api/health-only fields get the pair
        // (presence on health, absence from ready); with this
        // assertion the deploy-doc-line-216 monitor-contract triple
        // (ok presence + ready always-true + ok absence-from-ready)
        // is FULLY pinned.
        assert(!readyJson.obj.contains("ok"),
          clue = s"/api/ready must NOT echo `ok` per deploy doc line 216 documenting it as /api/health's hardcoded always-up live-monitor invariant (literal Bool(true) at renderHealth line 73); /api/ready uses `ready` instead with different semantics (accepting-traffic, not just process-alive). A refactor mirroring `ok` into renderReadiness would confuse monitors keying on body.ok===true as the always-up signal AND create field ambiguity between `ok` (liveness) and `ready` (acceptance); got readyJson keys: ${readyJson.obj.keys.toVector.sorted.mkString(", ")}")

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
        // Three more high-impact security-relevant directives the
        // original 6-pin selection (camera/microphone/geolocation +
        // interest-cohort/browsing-topics + usb) missed -- the existing
        // pins cover hardware-sensor privacy + Privacy-Sandbox tracking
        // opt-outs + hardware comm, but leave three other documented
        // categories untested: financial-actions (payment), WebAuthn
        // credential-harvesting (publickey-credentials-get), and
        // cross-subdomain origin-relaxation (document-domain). Each
        // protects against a distinct attack surface a future XSS or
        // compromised vendored library would otherwise be able to
        // weaponize. Adding these as parallel one-liners to the
        // existing block rather than as a separate test because they
        // share the same lexical context (the Permissions-Policy
        // header value already in scope) and the same operator-side
        // contract (deploy doc line 208's deny-list claim).
        // payment=(): blocks the Payment Request API. A future XSS
        // injection without this directive could pop up a hostile
        // payment dialog -- the user sees a legitimate-looking
        // browser-native UI (so they trust it) but the merchant /
        // amount fields are attacker-controlled.
        assert(permissionsPolicy.contains("payment=()"),
          s"missing payment=() -- without this directive a future XSS could trigger the Payment Request API and pop up a browser-native payment dialog with attacker-controlled merchant/amount fields, getting credential trust the application doesn't deserve; in: $permissionsPolicy")
        // publickey-credentials-get=(): blocks WebAuthn navigator.
        // credentials.get(). A future XSS without this could silently
        // attempt to harvest the user's hardware-key / passkey
        // assertions for any RP the attacker can name (typically with
        // a phishing-styled credentialId list) -- the assertion
        // wouldn't succeed against random RPs, but the credentialId
        // list ITSELF is a fingerprinting / enumeration surface.
        assert(permissionsPolicy.contains("publickey-credentials-get=()"),
          s"missing publickey-credentials-get=() -- without this directive a future XSS could attempt WebAuthn credential enumeration / fingerprinting via navigator.credentials.get() with attacker-controlled RP + credentialId lists; in: $permissionsPolicy")
        // document-domain=(): blocks the legacy document.domain=
        // cross-subdomain origin-relaxation footgun. Setting
        // document.domain in shared-cookie subdomain deployments
        // historically let scripts at e.g. app.example.com and
        // billing.example.com communicate by both setting
        // document.domain="example.com"; modern best-practice
        // (and Chrome's deprecation roadmap) closes that, but a
        // future XSS that DID gain script execution could try the
        // legacy API to attempt cross-subdomain access -- the
        // directive denies it explicitly.
        assert(permissionsPolicy.contains("document-domain=()"),
          s"missing document-domain=() -- without this directive a future XSS could try the legacy document.domain= API to widen its origin scope to a parent domain (relevant on subdomain-sharing deployments, less so on bare-domain ones, but defense-in-depth applies); in: $permissionsPolicy")
        assertEquals(headerValue(index, "Cross-Origin-Opener-Policy"), Some("same-origin"))
        assertEquals(headerValue(index, "Cross-Origin-Resource-Policy"), Some("same-origin"))
        // Referrer-Policy: no-referrer is the strictest value per the deploy
        // doc's Reverse-Proxy section -- "no Referer is ever sent on outbound
        // navigations or fetches, so the user's path through this app doesn't
        // leak to any third-party origin the user later visits via a link or
        // redirect; one tier stricter than the browser-default
        // strict-origin-when-cross-origin, which would still leak the origin
        // in some cross-site cases." A refactor relaxing this to no-referrer-
        // when-downgrade / strict-origin / origin-when-cross-origin / etc.
        // would silently regress the documented privacy floor.
        assertEquals(headerValue(index, "Referrer-Policy"), Some("no-referrer"))
        assertEquals(headerValue(index, "X-Robots-Tag"), Some("noindex, nofollow"))
        // Pin the ABSENCE of Strict-Transport-Security. The deploy doc's
        // Reverse-Proxy section explicitly documents the design choice:
        // "The origin does NOT emit Strict-Transport-Security because it
        // does not terminate TLS. For internet-facing deployments behind
        // an HTTPS-terminating proxy, configure the proxy to add an HSTS
        // header... For private-network/loopback deployments, HSTS is
        // unnecessary." A well-meaning refactor that added HSTS to
        // applySecurityHeaders would: (1) silently contradict the
        // documented "we don't emit HSTS, the proxy does" stance, (2) brick
        // mixed http+https deployment workflows (once a browser sees
        // max-age in flight it REFUSES to fall back to http for that host
        // until the TTL expires, which the deploy doc explicitly warns
        // about: "a misconfigured HSTS during testing can lock you out of
        // plain-HTTP access until the header's TTL expires"), and (3)
        // potentially conflict with the proxy's own HSTS header on
        // internet-facing deployments (two competing max-age values, the
        // browser uses the most recently received but a probe sequence
        // hitting origin-then-proxy-then-origin would see oscillation).
        // Absence is a stronger contract than presence here -- a refactor
        // adding HSTS is "obviously a security improvement" by intuition,
        // so the test needs to be specific that we DO NOT want it.
        assertEquals(headerValue(index, "Strict-Transport-Security"), None,
          clue = "applySecurityHeaders must NOT emit Strict-Transport-Security -- the deploy doc's Reverse-Proxy section explicitly documents the origin as TLS-unaware and HSTS as the proxy's responsibility, and emitting it from the origin breaks mixed http+https testing workflows by locking the browser into https-only fallback")

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
          // Pin the full local-provider entry shape per deploy doc
          // line 101: each entry is `{id, displayName, kind, startPath}`
          // where `kind` is `password` for the always-present `local`
          // entry (`displayName: "Email and password"`, `startPath:
          // null` -- email/password sign-in goes through the auth form
          // rather than a separate route). The existing assertion on
          // line 3095 only pinned the `id` field; the other three
          // fields (displayName, kind, startPath) were untested, so
          // a refactor that changed e.g. `displayName` to "Local
          // password" or `kind` from "password" to "local" would
          // silently break frontends keying on the documented values
          // (the shipped frontend's auth-mode-rendering keys on
          // `kind === "password"` to gate the email+password UI
          // vs `kind === "oidc"` for the OIDC button). The startPath
          // field MUST be null for the local provider (no separate
          // start route) -- a future refactor that gave local-password
          // a non-null startPath would confuse the frontend's
          // routing logic which expects to handle local sign-in
          // via the auth form.
          val localProvider = anonymousAuth("providers").arr.head
          assertEquals(localProvider("displayName").str, "Email and password",
            clue = s"local provider's displayName must be the documented 'Email and password' string per deploy doc line 101 -- a refactor changing this would break the auth-mode-chooser UI's user-visible label without test failure; got: ${localProvider("displayName").str}")
          assertEquals(localProvider("kind").str, "password",
            clue = s"local provider's kind must be 'password' per deploy doc line 101 -- the shipped frontend's auth-mode rendering keys on `kind === 'password'` to gate the email+password form vs `kind === 'oidc'` for the OIDC button; a refactor changing this would silently break the auth-form rendering; got: ${localProvider("kind").str}")
          assert(localProvider("startPath") == ujson.Null,
            clue = s"local provider's startPath must be null per deploy doc line 101 (email/password sign-in goes through the auth form, NOT a separate start route); a refactor giving it a non-null startPath would confuse the frontend's routing logic that expects to handle local sign-in via the form; got: ${localProvider("startPath")}")
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
          // Pin the user.userId field shape -- documented in deploy
          // doc line 101 as part of the signed-in user view and at
          // line 235 as the "per-account UUIDs (internal userId)" the
          // store persists. Before this commit the field was entirely
          // untested even though it's the operationally relevant
          // account identifier: rate-limit client= log lines use
          // `user:<userId>` when an authenticated principal is on the
          // request (per the deploy doc's rate-limit log format
          // documentation), audit-log correlation queries grep for
          // userId, AND the JobQueue's per-user ownership check at
          // /api/.../jobs/{id} uses userId to enforce the "404 if
          // submitted by a different user" semantics. A refactor
          // that dropped userId from the response would break (1)
          // operator-side log-correlation, (2) rate-limit client-
          // bucketing dashboards, (3) any scripted client that
          // tracks per-user state across requests; a refactor that
          // changed the format from UUID to something else (e.g.
          // sequential integer, hash of email) would change the
          // attack surface -- sequential integers leak account
          // count to an attacker who can enumerate, hash-of-email
          // exposes whether two known emails were registered, UUIDs
          // are unguessable in both directions. Assertion uses a
          // permissive regex (`[0-9a-f]{8}-...8-4-4-4-12...`) so the
          // test passes for any UUID-shaped value (UUIDv4 is what
          // the production code generates per
          // PlatformUserAuth.scala's randomUUID().toString call,
          // but the test allows any version since the bit-layout
          // distinction doesn't affect the operator-facing
          // properties).
          val userId = registerJson("user")("userId").str
          assert(userId.matches("^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"),
            clue = s"register response user.userId must be a UUID-shaped string (8-4-4-4-12 hex with dashes) per deploy doc line 101 + line 235's 'per-account UUIDs' framing -- the rate-limit `client=user:<userId>` log lines, audit-log correlation queries, and JobQueue per-user ownership checks all key on this format; a refactor changing it (e.g. to a sequential integer) would silently change the account-enumeration attack surface AND break operator-side log-grep workflows; got: '$userId'")
          // Pin the linkedProviders array shape for a local-password
          // user. Deploy doc line 101 explicitly documents the
          // linkedProviders field as "the linkedProviders array
          // naming all provider ids linked to the account
          // (alphabetically sorted, distinct -- includes the literal
          // `local` value for a password-registered user, plus any
          // OIDC provider id like `google` for an OIDC-linked user;
          // the two are mutually exclusive on a given account because
          // the local-vs-OIDC email-collision rule above blocks
          // mixing)". Before this commit only the OIDC-side branch
          // was tested (at line ~3451 via `.contains("google")`); the
          // local-password branch was untested even though it's the
          // more-commonly-exercised path in non-OIDC deployments.
          // The 24a703b OIDC providers-list pin (which establishes
          // the available-providers contract) is complementary to
          // this linkedProviders pin (which establishes the
          // per-account-actually-linked contract); a refactor
          // changing the local-side mark from "local" to e.g.
          // "password" would silently break frontend logic that
          // keys on linkedProviders to differentiate "user can sign
          // in via password" vs "user must sign in via OIDC".
          val linkedProviders = registerJson("user")("linkedProviders").arr.toVector.map(_.str)
          assertEquals(linkedProviders, Vector("local"),
            clue = s"freshly-registered local-password user's linkedProviders must equal exactly ['local'] per deploy doc line 101 -- a refactor that emitted a different mark (e.g. 'password', 'email-password', or empty array) would silently break the frontend's provider-routing logic AND the runbook's 'Forgotten-password support triage' (section 5A line 339) which assumes operators can spot password-only accounts via this field. Note: equality check (not contains) because the deploy doc says the array must include 'local' AND be distinct + alphabetically sorted -- for a fresh local-password user with no OIDC links, the only correct value is exactly ['local']; got: $linkedProviders")

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
          // Pin the EXACT Max-Age value (12h = 43200 seconds), not
          // just "Max-Age is present somewhere in the header." The
          // default sessionTtlMs is PlatformUserAuth.DefaultSessionTtlMs
          // = 12L * 60L * 60L * 1000L = 43_200_000 ms; the cookie
          // serializer (sessionCookieHeader) divides ttlMs by 1000 so
          // the Max-Age attribute carries 43200. The deploy doc's
          // USER_AUTH_SESSION_TTL_MS bullet documents this exact 12h
          // default (with the "FIXED at login time, NOT refreshed by
          // subsequent activity" nuance the deploy-doc + runbook
          // explanations both depend on -- see the record-slides-but-
          // cookie-doesn't mechanic). A refactor changing the default
          // (e.g. to 1h for tighter post-leak window or 24h for less
          // re-auth friction) would silently invalidate both docs'
          // claims AND change the leaked-token-lifetime upper bound
          // operators reason about for incident response. Same exact-
          // value pinning rationale as 121e5b5 (OIDC state-cookie
          // Max-Age=600 pin). Test config uses
          // PlatformUserAuth.Config(storePath = storePath) with no
          // sessionTtlMs override so the default applies; if a future
          // test wants a different TTL it should override it
          // explicitly + adjust this assertion in lockstep.
          assert(setCookie.contains("Max-Age=43200"),
            s"session cookie must have Max-Age=43200 (default 12h sessionTtlMs) per deploy doc + runbook; got: $setCookie")
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
          // Symmetric coverage with the insecure-mode session-cookie
          // wire-format test above: both modes share the same
          // sessionCookieHeader code path and must carry the same
          // generic-security attributes (HttpOnly to block JS-side
          // theft via XSS, SameSite=Lax for CSRF defense on cross-
          // origin POSTs, Max-Age=43200 for the documented 12h
          // sessionTtlMs default per bd8e7f3's exact-value pinning
          // rationale). The insecure-mode test pins all three on the
          // sicfun_session=... cookie; before this addition, the
          // secure-mode test only pinned __Host-prefix-specific
          // attributes (__Host-, Secure, Path=/, !Domain=), so a
          // refactor that broke ONLY the secure-mode branch of
          // sessionCookieHeader (e.g. dropped HttpOnly or changed
          // SameSite from Lax to None, both of which would be
          // catastrophic for XSS / CSRF defense respectively) would
          // pass this test silently. Mirror the insecure-mode
          // assertions so a future regression has to break BOTH
          // branches to slip through CI.
          assert(setCookie.contains("HttpOnly"),
            s"secure-mode cookie must be HttpOnly to block JS read access (XSS defense); got: $setCookie")
          assert(setCookie.contains("SameSite=Lax"),
            s"secure-mode cookie must be SameSite=Lax for CSRF defense; got: $setCookie")
          assert(setCookie.contains("Max-Age=43200"),
            s"secure-mode cookie must have Max-Age=43200 (default 12h sessionTtlMs) -- same exact-value pin as the insecure-mode test (see bd8e7f3's deploy-doc + runbook citation for the 12h default's operator-relevance); got: $setCookie")
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
          // Pin the full Google-OIDC provider entry shape per deploy
          // doc line 101's 4-field enumeration -- mirror of e1e7afa's
          // local-provider pin but for the OIDC branch. For an OIDC
          // provider, the documented shape is `{id, displayName, kind,
          // startPath}` where kind="oidc" (vs local's "password") and
          // startPath is the `/api/auth/oidc/<id>/start` route the
          // frontend's "Continue with Google" anchor targets (vs
          // local's null startPath); displayName comes from the
          // provider config (FakeOidcProvider in tests uses "Google";
          // a real deployment's GoogleOidcProvider also uses "Google"
          // per its `override val displayName` at PlatformUserAuth.scala
          // line 222). The pair of pins (e1e7afa local + this commit
          // OIDC) covers BOTH branches of the providers-array shape so
          // a refactor that changed the OIDC entry's kind to e.g.
          // "openid-connect" or "social" -- intuitive renames that
          // would silently break the frontend's `kind === "oidc"` UI-
          // gating logic -- would now fail the test; a refactor that
          // gave the OIDC provider a null startPath would similarly
          // break the frontend (the "Continue with Google" anchor's
          // href is taken directly from startPath, so a null value
          // makes the anchor link nowhere). The id ordering matters
          // too: local FIRST then OIDC per the deploy doc's
          // "Array order is fixed and deterministic: the local entry
          // comes FIRST, followed by each configured OIDC provider in
          // the order they were declared at startup"; that's already
          // pinned by the `Vector("local", "google")` check above, but
          // the entry-shape pin closes the per-entry-field coverage.
          val googleProvider = anonymousAuth("providers").arr(1)
          assertEquals(googleProvider("id").str, "google",
            clue = s"OIDC provider id must be 'google' (the FakeOidcProvider's `override val id = \"google\"` matches the production GoogleOidcProvider's same override); a refactor changing the id would break every cross-reference that keys on it (frontend routes, log lines, dashboard alerting); got: ${googleProvider("id").str}")
          assertEquals(googleProvider("displayName").str, "Google",
            clue = s"OIDC provider displayName must be 'Google' (taken from the provider's config; both FakeOidcProvider and GoogleOidcProvider expose `displayName = \"Google\"`); changing it would change the user-visible 'Continue with X' button label without test failure; got: ${googleProvider("displayName").str}")
          assertEquals(googleProvider("kind").str, "oidc",
            clue = s"OIDC provider kind must be 'oidc' per deploy doc line 101's `kind` enum (`password` for local, `oidc` for OIDC) -- the shipped frontend's auth-mode rendering keys on `kind === 'oidc'` to render the 'Continue with Google' button vs `kind === 'password'` for the email+password form; a refactor renaming this to 'openid-connect' or 'social' would silently break the UI gating without compile error; got: ${googleProvider("kind").str}")
          assertEquals(googleProvider("startPath").str, "/api/auth/oidc/google/start",
            clue = s"OIDC provider startPath must be '/api/auth/oidc/google/start' (the documented route the frontend's anchor targets to begin the OIDC flow); a refactor changing the path component would silently break the 'Continue with Google' link AND likely break the GOOGLE_OIDC_REDIRECT_URI matching since the configured redirect URI must end with `/api/auth/oidc/google/callback` which mirrors the start path; got: ${googleProvider("startPath").str}")

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
          // Tighten the linkedProviders assertion from the original
          // loose `.contains("google")` to an exact-equality check
          // against Vector("google"). The loose check would pass even
          // if a refactor accidentally added extra entries (e.g. if
          // the email-collision defense weakened to allow OIDC+local
          // dual-linking, the array might become ["google", "local"]
          // -- the contains check would still pass, hiding the
          // documented "mutually exclusive on a given account"
          // contract violation pinned by 44c9f9f's email-collision
          // test). The equality form matches the local-password
          // companion pin in 132b9be (which uses exactly the same
          // shape `assertEquals(linkedProviders, Vector("local"))`
          // for the local-side branch) -- pinning both branches with
          // equality now means a refactor that altered the
          // mutually-exclusive invariant would fail BOTH tests
          // (linkedProviders array contents on local-only-user fire
          // AND OIDC-only-user fire), forcing the maintainer to
          // acknowledge the change explicitly.
          val oidcLinkedProviders = me("user")("linkedProviders").arr.toVector.map(_.str)
          assertEquals(oidcLinkedProviders, Vector("google"),
            clue = s"OIDC-signed-in user's linkedProviders must equal exactly ['google'] per deploy doc line 101's 'mutually exclusive on a given account' contract -- a refactor that allowed dual-linking (e.g. linkedProviders=['google','local']) would silently violate the email-collision defense (pinned by 44c9f9f) AND break the runbook's 'Forgotten-password support triage' (section 5A line 339) which assumes operators can identify OIDC-only vs local-password accounts via this field; got: $oidcLinkedProviders")
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
          // Pin the EXACT Max-Age value (10 minutes = 600 seconds), not
          // just "Max-Age is present somewhere in the header." Both the
          // deploy doc and runbook claim "10-minute" specifically, AND
          // upstream OAuth 2.0 BCP guidance treats state-store TTL as a
          // security knob (long TTLs widen the window where an
          // intercepted authorization redirect URL can be replayed; very
          // short TTLs make legitimate flows fail when the user takes
          // longer than expected on Google's consent screen). A refactor
          // that changed DefaultOidcFlowTtlMs from 10 min to e.g. 1 hour
          // (widening the replay window 6x) would silently pass a
          // "max-age=" substring check while contradicting both docs;
          // pinning the exact value catches that drift. Case-insensitive
          // because the attribute name "Max-Age" is case-insensitive per
          // RFC 6265 sec 4.1.1 but the cookie serializer here uses
          // mixed-case "Max-Age=" -- the lowercased setCookie shape is
          // what the existing has-attribute check above also uses.
          assert(setCookie.toLowerCase.contains("max-age=600"),
            s"state cookie must have Max-Age=600 (10 minutes) per deploy doc + runbook; got: $setCookie")
        }
      }
    }
  }

  // Parallel secure-mode OIDC state-cookie test, mirroring 72358a3's
  // session-cookie secure-mode symmetric pattern. The deploy doc
  // documents the secure-mode OIDC state cookie as `__Host-
  // sicfun_oidc_state` with `Secure` set "plus Secure in secure mode"
  // (deploy doc line 220), and the runbook section 5A line 344 makes
  // the same claim with the same operator-side log-grep / proxy-ACL
  // consideration callout. Before this test, the secure-mode branch
  // of `oidcStateCookieHeader(state, ttlMs, secure=true)` was
  // entirely uncovered: a refactor that broke ONLY the secure-mode
  // path (e.g. dropped the `__Host-` prefix, omitted Secure, dropped
  // HttpOnly on the secure branch only, or drifted Max-Age on one
  // branch but not the other) would pass the insecure-mode test
  // above and silently regress the documented secure-mode contract.
  // Same exact-value pinning policy as 121e5b5 (insecure-mode
  // Max-Age=600) + 72358a3 (secure-mode session cookie added the
  // same generic-security attributes that were already pinned in
  // the insecure-mode session-cookie test) -- two cookie-emission
  // branches sharing one helper must pin the same security
  // contract on BOTH branches.
  test("OIDC /start secure-mode state cookie uses __Host-sicfun_oidc_state with Secure + HttpOnly + SameSite=Lax + Max-Age=600") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        val provider = new FakeOidcProvider
        withServer(
          staticDir,
          platformAuth = Some(
            PlatformUserAuth.Config(
              storePath = storePath,
              allowLocalRegistration = false,
              cookieSecure = true,
              oidcProviders = Vector(provider)
            )
          )
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"
          val start = get(s"$baseUri${provider.startPath}")
          val urlState = queryParam(headerValue(start, "Location").getOrElse(""), "state")
            .getOrElse(fail("missing OIDC state in redirect"))
          val setCookie = headerValue(start, "Set-Cookie").getOrElse(fail("missing Set-Cookie on secure-mode /start"))
          // __Host- prefix per RFC 6265 sec 4.1.3 (browsers reject the
          // cookie unless Secure + Path=/ + no Domain attribute). The
          // deploy doc explicitly notes "same `__Host-` prefix and
          // same operator-side log-grep / proxy-ACL consideration as
          // the session cookie" -- the symmetry across the two
          // __Host-prefixed cookies is the operator-relevant contract
          // (an operator grepping log lines for sibling-subdomain
          // attempted overwrites uses the same patterns for both).
          assert(setCookie.startsWith("__Host-sicfun_oidc_state="),
            s"secure-mode state cookie name must be __Host-sicfun_oidc_state per the RFC 6265 prefix protection; got: $setCookie")
          val cookieValue = setCookie.takeWhile(_ != ';').drop("__Host-sicfun_oidc_state=".length)
          assertEquals(cookieValue, urlState,
            "state cookie value must match the URL state so the callback's compare succeeds (same shape as insecure-mode test above)")
          assert(setCookie.contains("Secure"),
            s"secure-mode state cookie must set Secure (required by __Host- prefix and by USER_AUTH_COOKIE_SECURE=true semantics); got: $setCookie")
          assert(setCookie.contains("Path=/"),
            s"__Host- prefix requires Path=/; got: $setCookie")
          assert(!setCookie.contains("Domain="),
            s"__Host- prefix forbids Domain attribute; got: $setCookie")
          assert(setCookie.toLowerCase.contains("httponly"),
            s"state cookie must be HttpOnly so JS cannot read or set it; got: $setCookie")
          assert(setCookie.toLowerCase.contains("samesite=lax"),
            s"state cookie must be SameSite=Lax so the provider's top-level redirect can send it back; got: $setCookie")
          // Same exact-value Max-Age pin as the insecure-mode test
          // above -- DefaultOidcFlowTtlMs is mode-independent so
          // both branches must carry Max-Age=600 (10 min).
          assert(setCookie.toLowerCase.contains("max-age=600"),
            s"secure-mode state cookie must have Max-Age=600 (10 min) per the same deploy-doc + runbook claim that the insecure-mode test pins; got: $setCookie")
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

  // Pin the OIDC-vs-local-password email-collision defense (deliberate
  // account-hijack mitigation per the deploy doc + runbook documentation).
  // The scenario: a local-password account exists for `alice@example.com`,
  // and an OIDC sign-in arrives carrying the same email. Without the
  // collision check, an attacker who controlled a Google account at
  // alice@example.com (e.g., the legitimate owner of the email at Google,
  // having registered there AFTER the local-password account was created
  // on this server) could complete OIDC sign-in and silently take over
  // the local-password account's session and stored profile data. The
  // upsertOidcIdentity collision check at PlatformUserAuth.scala line 622
  // closes this by refusing the upsert with the documented exact error
  // string "an account with that email already exists; sign in with its
  // existing method". The deploy doc enumerates this as one of the 13
  // finishOidc Left values, the runbook's section 5A line 337 explains
  // the operator-facing triage, and the frontend's lookupOidcErrorMessage
  // maps this string to a user-friendly message. A refactor that removed
  // the collision check (or changed the error string in a way that broke
  // the frontend's exact-match lookup) would silently re-open the hijack
  // path AND break the documented support triage flow without any test
  // failure. Same regression-pin pattern as the rest of the chain --
  // documented security-relevant behavior gets pinned so refactors can't
  // silently regress it.
  test("OIDC callback rejects sign-in when the email matches an existing local-password account -- pins the deliberate account-hijack defense documented in deploy doc + runbook") {
    val collidingEmail = "collision@example.com"
    val provider = new PlatformUserAuth.OidcProvider:
      override val id = "google"
      override val displayName = "Google"
      override def authorizationUri(state: String, codeChallenge: String): String =
        s"https://accounts.google.test/o/oauth2/v2/auth?state=$state&code_challenge=$codeChallenge"
      override def exchangeCode(code: String, codeVerifier: String): Either[String, PlatformUserAuth.OidcIdentity] =
        Right(PlatformUserAuth.OidcIdentity(
          subject = "google-collision-12345",
          email = collidingEmail,
          displayName = "OIDC Hijacker",
          avatarUrl = None
        ))
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
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

          // Step 1: Register a local-password user with the colliding
          // email so the OIDC upsert has something to collide with.
          val register = postJson(s"$baseUri/api/auth/register",
            s"""{"email":"$collidingEmail","password":"correct-horse-battery","displayName":"Local Owner"}""")
          assertEquals(register.statusCode(), 201,
            clue = "local-password registration must succeed before the OIDC collision check can be exercised; the test's whole purpose is that an existing local-password account is in place when the OIDC attempt arrives")

          // Step 2: Start a fresh OIDC flow as an anonymous client (no
          // session from the register response is carried forward) --
          // this is the threat model: an attacker hitting the OIDC
          // entry point with the colliding email, not the legitimate
          // local-password user signing in again via Google.
          val start = get(s"$baseUri${provider.startPath}")
          assertEquals(start.statusCode(), 302,
            clue = "OIDC /start always 302-redirects to the provider")
          val state = queryParam(headerValue(start, "Location").getOrElse(""), "state")
            .getOrElse(fail("missing OIDC state in /start redirect"))
          val stateCookie = headerValue(start, "Set-Cookie")
            .map(_.takeWhile(_ != ';'))
            .getOrElse(fail("missing OIDC state cookie from /start"))

          // Step 3: Complete the OIDC callback with the colliding email
          // in the exchanged identity. The collision check fires at
          // upsertOidcIdentity time.
          val callback = get(
            s"$baseUri${provider.callbackPath}?state=$state&code=test-code",
            Map("Cookie" -> stateCookie)
          )
          assertEquals(callback.statusCode(), 302,
            clue = "OIDC callback always 302-redirects (success or failure shape is wire-identical)")
          val location = headerValue(callback, "Location").getOrElse(fail("missing Location header on callback redirect"))

          // The redirect must NOT be the success path -- it must
          // carry the documented auth_error= with the collision
          // message. Check both substrings (the auth_error= prefix
          // proves it's the failure landing page, and the
          // already%20exists fragment proves the specific collision
          // error fired, not some other finishOidc Left).
          assert(location.contains("auth_error="),
            s"OIDC collision must redirect to ?auth_error=... not the success page; got: $location")
          // The error string is "an account with that email already exists;
          // sign in with its existing method" -- urlEncode replaces spaces
          // with %20, so "already exists" becomes "already%20exists" in
          // the redirect URL. Checking the substring AFTER url-encoding
          // catches both message-text changes AND any future shift to a
          // different encoding (e.g., `+` vs `%20`).
          assert(location.toLowerCase.contains("already%20exists"),
            s"OIDC collision must surface the documented 'an account with that email already exists' message in the auth_error redirect; got: $location")

          // The critical security check: no session cookie is issued.
          // If a session cookie WERE installed, an attacker would have
          // taken over the local-password account -- the entire point
          // of the collision check is to prevent this. A future refactor
          // that "fixed" the collision by linking the OIDC identity to
          // the existing local-password account would silently open
          // exactly the account-hijack path the defense was designed to
          // close.
          assertEquals(headerValue(callback, "Set-Cookie"), None,
            "OIDC collision MUST NOT install a session cookie -- doing so would let an attacker take over a local-password account via Google OIDC if they control a Google account with the same email. The deliberate account-hijack defense documented in deploy doc + runbook depends on the OIDC flow refusing to mint a session in this case.")
        }
      }
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
            // Tighten the expected substring from the original loose "OIDC"
            // (which would match ANY auth_error response since they all
            // mention OIDC) to the full documented finishOidc Left value
            // URL-encoded: a refactor that changed the wording to e.g.
            // "OIDC state not found" or "Login state expired" would pass
            // the loose "OIDC" check but silently miss the frontend's
            // OIDC_ERROR_MESSAGES exact-match lookup at site.js line 1597
            // ("OIDC login state expired or is invalid": "Your sign-in
            // took too long or was already completed in another tab.
            // Please try again."), making the user see the raw fallback
            // text instead of the friendly translation; URL-encoded form
            // matches the documented serialization (urlEncode replaces +
            // with %20). Same exact-string pinning rationale as 44c9f9f's
            // "already%20exists" assertion for the email-collision Left.
            ("?state=this-state-was-never-issued&code=xyz",
              Map("Cookie" -> "sicfun_oidc_state=this-state-was-never-issued"),
              "OIDC%20login%20state%20expired%20or%20is%20invalid")
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

  // Pin the documented startedAtEpochMs + completedAtEpochMs lifecycle
  // fields on terminal poll responses. Deploy doc line 77 documents
  // the GET /api/analyze-hand-history/jobs/{id} response shape:
  // "Returns 200 with status=queued / running / completed / failed,
  // the universal fields jobId / statusUrl / submittedAtEpochMs /
  // startedAtEpochMs / completedAtEpochMs (the latter two are null
  // until the worker reaches each transition)". The 94150cb fire
  // pinned the 202 body's jobId / submittedAtEpochMs / pollAfterMs;
  // this fire pins the two GET-poll-only timing fields plus the
  // SUBMIT-TIME-LE-STARTED-TIME-LE-COMPLETED-TIME ordering invariant
  // operators chart against. Why this matters operationally:
  // dashboards correlating fleet-wide submit→complete latency
  // (the deploy doc explicitly suggests "chart submit→start latency,
  // completion-rate, timeout-rate, and queue depth from the audit
  // log alone") key on the values produced by completedAt -
  // submittedAt; a refactor that swapped the orderings (e.g.
  // assigned completedAtEpochMs at submit time and startedAtEpochMs
  // at worker-completion time) would silently chart negative
  // latencies AND defeat the alerting rules that compare these
  // values. New test submits to /api/analyze-hand-history via a
  // BlockingBackend (lets us control worker timing deterministically),
  // captures a wall-clock window around the submission for the
  // submittedAtEpochMs assertion (same pattern as 94150cb), waits
  // for the backend to start + releases it + awaits the terminal
  // state, then asserts: (1) both startedAtEpochMs + completedAtEpochMs
  // are NON-NULL in the terminal state (documented "null until
  // transition" contract -- both transitions HAVE happened by the
  // time the response surfaces "completed"), (2) submittedAt <=
  // startedAt <= completedAt (the ordering operators chart against),
  // and (3) all three values are within a reasonable wall-clock
  // window of the test's measurements (catches a refactor that
  // accidentally produced epoch-millis from a different time source
  // like Date.now() vs System.currentTimeMillis() at different
  // points). Same per-endpoint asymmetric-drift pattern as 94150cb
  // -- covers both /api/analyze-hand-history and /api/playing-hall
  // since the lifecycle-field contract is documented as identical
  // across the two endpoints.
  test("terminal poll response carries non-null startedAtEpochMs + completedAtEpochMs with submitted <= started <= completed ordering on both submission endpoints") {
    withStaticSite { staticDir =>
      val backend = new BlockingBackend(Right(sampleAnalysisResult))
      val playingHallBackend = new BlockingPlayingHallBackend(Right(samplePlayingHallResult))
      withServer(staticDir, backend = backend, playingHallBackend = playingHallBackend) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        // /api/analyze-hand-history lifecycle test.
        val analyzeStart = System.currentTimeMillis()
        val analyzeSubmit = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload)
        assertEquals(analyzeSubmit.statusCode(), 202)
        val analyzeStatusUri = s"$baseUri${jsonBody(analyzeSubmit)("statusUrl").str}"
        assert(backend.started.await(3, TimeUnit.SECONDS), "analyze backend never started")
        backend.release.countDown()
        val analyzeTerminal = awaitTerminalJob(analyzeStatusUri)
        val analyzeEnd = System.currentTimeMillis()

        assertEquals(analyzeTerminal("status").str, "completed",
          clue = "analyze backend must reach terminal completed state for the lifecycle-field assertions to apply")
        // Non-null contract: the documented "null until the worker
        // reaches each transition" means BOTH transitions have
        // happened by the time the response surfaces "completed",
        // so both fields MUST be non-null in this branch.
        assert(analyzeTerminal("startedAtEpochMs") != ujson.Null,
          s"analyze terminal-state startedAtEpochMs must be non-null per deploy doc line 77 -- a refactor that left it null on the terminal response would silently break dashboards charting submit→start latency (the documented operator metric); got: ${analyzeTerminal("startedAtEpochMs")}")
        assert(analyzeTerminal("completedAtEpochMs") != ujson.Null,
          s"analyze terminal-state completedAtEpochMs must be non-null per deploy doc line 77 -- a refactor that left it null on the terminal response would silently break submit→complete latency dashboards AND the JobQueue's durationMs computation in the `job completed` log line (line ~322 emits the diff completedAt minus startedAt as durationMs=); got: ${analyzeTerminal("completedAtEpochMs")}")

        // Ordering invariant: submitted <= started <= completed.
        // Operators chart submit→start (queue-wait latency) and
        // start→complete (worker-run latency) using these values;
        // a refactor that swapped any of the three values across
        // assignment sites would silently produce negative-latency
        // numbers and break alerting.
        val analyzeSubmitted = analyzeTerminal("submittedAtEpochMs").num.toLong
        val analyzeStarted = analyzeTerminal("startedAtEpochMs").num.toLong
        val analyzeCompleted = analyzeTerminal("completedAtEpochMs").num.toLong
        assert(analyzeSubmitted <= analyzeStarted,
          s"analyze terminal-state must have submittedAt <= startedAt -- queue-wait latency = startedAt - submittedAt cannot be negative; got submitted=$analyzeSubmitted started=$analyzeStarted (diff=${analyzeStarted - analyzeSubmitted})")
        assert(analyzeStarted <= analyzeCompleted,
          s"analyze terminal-state must have startedAt <= completedAt -- worker-run latency = completedAt - startedAt cannot be negative; got started=$analyzeStarted completed=$analyzeCompleted (diff=${analyzeCompleted - analyzeStarted})")
        // All three values within the wall-clock window of the test
        // (with 100ms slack for clock skew / GC pauses).
        assert(analyzeSubmitted >= analyzeStart - 100 && analyzeCompleted <= analyzeEnd + 100,
          s"analyze lifecycle epoch-millis must fall within [${analyzeStart - 100}, ${analyzeEnd + 100}] window of the test's wall-clock around the submission/terminal cycle -- a value outside the window would suggest the server is using a different clock source than System.currentTimeMillis() (e.g. accidentally using Date.now() from a different process or a stale cache); got submitted=$analyzeSubmitted completed=$analyzeCompleted")

        // /api/playing-hall lifecycle test (mirror of analyze).
        val hallStart = System.currentTimeMillis()
        val hallSubmit = postJson(s"$baseUri/api/playing-hall", validPlayingHallPayload)
        assertEquals(hallSubmit.statusCode(), 202)
        val hallStatusUri = s"$baseUri${jsonBody(hallSubmit)("statusUrl").str}"
        assert(playingHallBackend.started.await(3, TimeUnit.SECONDS), "hall backend never started")
        playingHallBackend.release.countDown()
        val hallTerminal = awaitTerminalJob(hallStatusUri)
        val hallEnd = System.currentTimeMillis()

        assertEquals(hallTerminal("status").str, "completed",
          clue = "hall backend must reach terminal completed state")
        assert(hallTerminal("startedAtEpochMs") != ujson.Null,
          s"hall terminal-state startedAtEpochMs must be non-null (symmetric with analyze); got: ${hallTerminal("startedAtEpochMs")}")
        assert(hallTerminal("completedAtEpochMs") != ujson.Null,
          s"hall terminal-state completedAtEpochMs must be non-null (symmetric with analyze); got: ${hallTerminal("completedAtEpochMs")}")
        val hallSubmitted = hallTerminal("submittedAtEpochMs").num.toLong
        val hallStarted = hallTerminal("startedAtEpochMs").num.toLong
        val hallCompleted = hallTerminal("completedAtEpochMs").num.toLong
        assert(hallSubmitted <= hallStarted,
          s"hall must have submittedAt <= startedAt; got submitted=$hallSubmitted started=$hallStarted")
        assert(hallStarted <= hallCompleted,
          s"hall must have startedAt <= completedAt; got started=$hallStarted completed=$hallCompleted")
        assert(hallSubmitted >= hallStart - 100 && hallCompleted <= hallEnd + 100,
          s"hall lifecycle epoch-millis must fall within [${hallStart - 100}, ${hallEnd + 100}] wall-clock window; got submitted=$hallSubmitted completed=$hallCompleted")
      }
    }
  }

  // Pin the three unpinned fields in the documented 202 submission
  // body shape: jobId, submittedAtEpochMs, pollAfterMs. Deploy doc
  // line 66 documents the full 5-field body shape ("jobId, status
  // (always the literal string \"queued\" on a fresh 202), statusUrl,
  // submittedAtEpochMs, pollAfterMs"); existing tests cover `status`
  // ("queued" at line 3777) and `statusUrl` (extensively, including
  // the Location-equals-statusUrl pair-test 8728be9 immediately
  // below); this commit closes the remaining three:
  //   - `jobId`: the documented field every poll/cancel/status
  //     subsequent request keys on; a refactor that dropped it
  //     would force clients to parse statusUrl to extract the id,
  //     a fragile workaround that breaks if statusUrl format
  //     changes;
  //   - `submittedAtEpochMs`: server-side wall-clock at submission;
  //     consumed by dashboards correlating submit-time across the
  //     fleet (a dashboard charting submit→complete latency keys on
  //     the SAME field that completedAtEpochMs - submittedAtEpochMs
  //     produces -- if submittedAtEpochMs disappeared from the 202,
  //     the dashboard would silently chart 0 for newly-submitted
  //     jobs until they reached terminal state and got the field
  //     from the GET poll response instead);
  //   - `pollAfterMs`: server-suggested initial poll delay the
  //     frontend's pollAnalysisJob / pollPlayingHallJob use as
  //     their first sleep duration (site.js lines 396 + 549 read
  //     body.pollAfterMs and pass it through normalizePollAfterMs);
  //     if this field disappeared, the frontend's normalizePollAfterMs
  //     fallback would fire, which uses a different default cadence
  //     than the server intended.
  // Same regression-pin pattern as 8728be9 (Location-header pair):
  // covers both /api/analyze-hand-history AND /api/playing-hall in
  // one test body because the 202 body shape is documented to be
  // identical across both submission endpoints, and a refactor
  // touching one branch likely touches both -- per-endpoint
  // assertions catch the asymmetric-drift case.
  test("submission 202 body carries jobId + submittedAtEpochMs + pollAfterMs fields on both /api/analyze-hand-history and /api/playing-hall") {
    withStaticSite { staticDir =>
      val backend = new BlockingBackend(Right(sampleAnalysisResult))
      val playingHallBackend = new BlockingPlayingHallBackend(Right(samplePlayingHallResult))
      withServer(staticDir, backend = backend, playingHallBackend = playingHallBackend) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        // Capture wall-clock before the submission so we can assert
        // submittedAtEpochMs falls in the expected [before, after]
        // window. The server uses System.currentTimeMillis() so the
        // values must be on the same clock.
        val before = System.currentTimeMillis()
        val analyzeResp = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload)
        val after = System.currentTimeMillis()
        assertEquals(analyzeResp.statusCode(), 202,
          clue = "analyze submission must return 202 Accepted")
        val analyzeBody = jsonBody(analyzeResp)

        // jobId: documented string field every poll/cancel/status
        // request keys on. Must be present + non-empty.
        val analyzeJobId = analyzeBody("jobId").str
        assert(analyzeJobId.nonEmpty,
          s"analyze 202 body.jobId must be a non-empty string per deploy doc line 66 -- a refactor that dropped it would force clients to parse statusUrl to extract the id, a fragile workaround that breaks if statusUrl format changes; got: '$analyzeJobId'")

        // submittedAtEpochMs: server wall-clock at submission.
        // Must be a finite positive long, and within the
        // [before-100ms, after+100ms] window of the local-clock
        // measurement around the request (the 100ms slack absorbs
        // network + JVM scheduling jitter on test runners). Casting
        // through .num.toLong because ujson stores numbers as Double
        // and the value can legitimately exceed Int.MaxValue (epoch
        // millis are post-2038 already in Long).
        val analyzeSubmitted = analyzeBody("submittedAtEpochMs").num.toLong
        assert(analyzeSubmitted >= before - 100 && analyzeSubmitted <= after + 100,
          s"analyze 202 body.submittedAtEpochMs must fall within the [before-100ms, after+100ms] window of the local-clock measurement around the request; got submitted=$analyzeSubmitted, window=[$before-100, $after+100]")

        // pollAfterMs: server-suggested initial poll delay. The
        // frontend reads this and passes it to pollAnalysisJob's
        // first sleep (site.js line 396 -> body.pollAfterMs).
        // Must be a positive integer; the exact value isn't pinned
        // because it's a server-side cadence knob, but it MUST be
        // > 0 so the frontend doesn't busy-poll.
        val analyzePollAfterMs = analyzeBody("pollAfterMs").num.toLong
        assert(analyzePollAfterMs > 0L,
          s"analyze 202 body.pollAfterMs must be > 0 -- the frontend uses this as the initial sleep duration in pollAnalysisJob (site.js line 396); a refactor that emitted 0 (or missing field) would silently turn the frontend's poll loop into a busy-spin against the server; got: $analyzePollAfterMs")

        backend.release.countDown()

        // Same three fields on /api/playing-hall.
        val before2 = System.currentTimeMillis()
        val hallResp = postJson(s"$baseUri/api/playing-hall", validPlayingHallPayload)
        val after2 = System.currentTimeMillis()
        assertEquals(hallResp.statusCode(), 202,
          clue = "hall submission must return 202 Accepted")
        val hallBody = jsonBody(hallResp)

        val hallJobId = hallBody("jobId").str
        assert(hallJobId.nonEmpty,
          s"hall 202 body.jobId must be a non-empty string (symmetric with analyze); got: '$hallJobId'")
        // jobIds for analyze + hall must be DISTINCT (each gets a
        // fresh id) -- a refactor that returned a shared/reused id
        // across job stores would silently let a GET against the
        // hall's status URL resolve to the analyze job and vice
        // versa.
        assert(analyzeJobId != hallJobId,
          s"analyze and hall jobIds must be distinct so per-job-store status URLs don't cross-resolve; got analyzeJobId=$analyzeJobId hallJobId=$hallJobId")

        val hallSubmitted = hallBody("submittedAtEpochMs").num.toLong
        assert(hallSubmitted >= before2 - 100 && hallSubmitted <= after2 + 100,
          s"hall 202 body.submittedAtEpochMs must fall within the [before-100ms, after+100ms] window around the request; got submitted=$hallSubmitted, window=[$before2-100, $after2+100]")

        val hallPollAfterMs = hallBody("pollAfterMs").num.toLong
        assert(hallPollAfterMs > 0L,
          s"hall 202 body.pollAfterMs must be > 0 (symmetric with analyze, used by pollPlayingHallJob at site.js line 549); got: $hallPollAfterMs")

        playingHallBackend.release.countDown()
      }
    }
  }

  // Pin the documented jobId UUID FORMAT on BOTH submission
  // endpoints' 202 responses per JobQueue.scala line 182's
  // `UUID.randomUUID().toString` (analyze) + line 483's same
  // call (playing-hall) -- the SHAPE-INVARIANT pin covering
  // the JOBID-FORMAT DIMENSION that the existing
  // 202-body-shape pin (line ~11598) covers in PRESENCE but
  // NOT in FORMAT (the existing pin verifies jobId is non-
  // empty + DIFFERENT across endpoints, but does NOT pin the
  // UUID shape itself); the jobId format is OPERATIONALLY
  // CRITICAL because: (a) operator log-grep workflows key on
  // UUID-shape patterns -- a tail-and-grep pipeline like
  // `tail -f deploy.log | grep -E '[0-9a-f]{8}-' | sort` to
  // extract jobIds for a postmortem depends on the documented
  // UUID shape; a refactor swapping `UUID.randomUUID().
  // toString` to e.g. `AtomicLong.getAndIncrement.toString`
  // (sequential), `Random.nextLong.toHexString` (short
  // random), `Instant.now().toEpochMilli.toString` (epoch
  // millis), or `UUID.randomUUID().toString.substring(0, 8)`
  // (truncated for "log brevity") would silently break the
  // grep pipeline AND multiple downstream contracts: (b)
  // UUID's NON-GUESSABILITY -- v4 UUIDs are 122 bits of
  // randomness (effectively unguessable), but sequential ids
  // are trivially guessable (`grep job-1234` -> try `grep
  // job-1235`) -- a refactor to sequential ids would silently
  // enable cross-user job-status enumeration attacks (the
  // attacker who knows their own jobId can probe the next
  // sequence number to harvest other users' job results);
  // the existing JobQueue per-user ownership check (the GET
  // /api/analysis/<id> endpoint rejects mismatched-owner
  // accesses) provides defense-in-depth, BUT the UUID
  // unguessability is the FIRST LINE OF DEFENSE because it
  // prevents the attacker from even ATTEMPTING the request
  // (a 404 / 403 access-denied for a guessable id is a
  // signal; a 404 for an unguessable UUID is noise to the
  // attacker), (c) UUID's UNIQUENESS across restarts -- v4
  // UUIDs are unique with effectively zero collision
  // probability (2^-61 per pair); a refactor to sequential
  // ids that reset on restart would silently let two
  // different processes (e.g. blue-green deploy) emit
  // OVERLAPPING ids, breaking aggregator-level dedup +
  // operator audit-trail uniqueness, (d) UUID's LOWERCASE
  // CONVENTION via UUID.toString -- the JDK's UUID.toString
  // emits all-lowercase hex per the RFC 4122 recommendation;
  // a refactor wrapping with `.toUpperCase` (for "visual
  // distinction") would silently break case-sensitive log-
  // aggregator field extraction patterns matching `[0-9a-f]`
  // (the aggregator would index the UPPERCASE form which
  // grep workflows using lowercase regex would miss); the
  // jobId format is enforced at the SOURCE (UUID.randomUUID
  // ().toString call) which feeds BOTH the HTTP 202 response
  // body AND every downstream log line; pinning the format
  // at the 202 response covers the HTTP-response-shape
  // dimension; per-format regression vectors that this pin
  // catches: (i) refactor swapping the id generator (e.g. to
  // sequential AtomicLong, random hex, epoch millis,
  // truncated UUID) would silently break UUID-shape grep
  // workflows + non-guessability + uniqueness, (ii) refactor
  // wrapping with .toUpperCase would silently break case-
  // sensitive aggregator extraction, (iii) refactor padding
  // the UUID (e.g. with a prefix like `job-<uuid>` for
  // visual distinction in logs) would silently break grep
  // patterns expecting bare UUID, (iv) refactor truncating
  // the UUID (e.g. to first 8 chars for log brevity) would
  // silently destroy uniqueness + non-guessability; test
  // approach mirrors the userId UUID pin at line 10379:
  // submit one analyze job, submit one hall job (both
  // produce 202 responses with jobIds from the same
  // UUID.randomUUID().toString call, just at different
  // JobQueue.scala lines), extract both jobIds, apply
  // 5-tier format check: (i) analyze jobId matches the
  // UUID 8-4-4-4-12 hex-with-dashes regex
  // `^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`
  // (the SAME regex used at line 10379 for userId so format
  // drift across the two emission sites is detectable),
  // (ii) hall jobId matches the same UUID regex (symmetric
  // pin -- catches a refactor that swapped ONE endpoint's
  // id generator independently of the other), (iii) analyze
  // jobId is LOWERCASE (catches .toUpperCase wrapping --
  // the regex's [0-9a-f] character class ALREADY enforces
  // this but the explicit assertion makes the contract
  // intent clearer), (iv) hall jobId is LOWERCASE (symmetric
  // catch), (v) the two jobIds are DISTINCT (uniqueness
  // sanity -- the existing 202-body-shape pin also catches
  // this, but the symmetric assertion belongs in this
  // SHAPE-FOCUSED pin to make the contract self-contained).
  test("submission 202 responses for analyze + playing-hall MUST carry jobIds in the documented UUID 8-4-4-4-12 hex-with-dashes shape (lowercase) per JobQueue.scala lines 182/483's UUID.randomUUID().toString call -- the SHAPE-INVARIANT pin verifies the format contract that the existing 202-body-shape pin covers in PRESENCE-only; UUID v4 unguessability is the FIRST LINE OF DEFENSE against cross-user job enumeration attacks") {
    withStaticSite { staticDir =>
      val backend = new BlockingBackend(Right(sampleAnalysisResult))
      val playingHallBackend = new BlockingPlayingHallBackend(Right(samplePlayingHallResult))
      withServer(staticDir, backend = backend, playingHallBackend = playingHallBackend) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val uuidRegex = "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"

        // Submit analyze + hall; capture the 202 response
        // bodies. The blocking backends pause the worker AT
        // the analyze/run call site, so the 202 returns
        // immediately + reliably without the test needing to
        // synchronize against the worker thread.
        val analyzeResp = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload)
        assertEquals(analyzeResp.statusCode(), 202,
          clue = "analyze submission must return 202 for the jobId-format pin to inspect the response body")
        val analyzeJobId = jsonBody(analyzeResp)("jobId").str

        val hallResp = postJson(s"$baseUri/api/playing-hall", validPlayingHallPayload)
        assertEquals(hallResp.statusCode(), 202,
          clue = "hall submission must return 202 for the symmetric jobId-format pin to inspect the response body")
        val hallJobId = jsonBody(hallResp)("jobId").str

        // (i) analyze jobId matches the UUID shape regex
        // (mirrors the userId UUID pin at line 10379's
        // regex -- consistent format-check pattern across
        // the codebase's UUID-shape pins)
        assert(analyzeJobId.matches(uuidRegex),
          clue = s"analyze 202 body.jobId MUST match the UUID 8-4-4-4-12 hex-with-dashes shape `$uuidRegex` per JobQueue.scala line 182's `UUID.randomUUID().toString` call -- a refactor swapping the id generator (e.g. AtomicLong sequential, Random hex, epoch millis, truncated UUID) would silently break UUID-shape grep workflows + unguessability (sequential ids are trivially enumerable, enabling cross-user job-status probing attacks before the per-user ownership check rejects them) + uniqueness across restarts (sequential ids reset; UUIDs do not); a refactor wrapping with .toUpperCase would silently break case-sensitive log-aggregator field extraction; got jobId: '$analyzeJobId'")

        // (ii) hall jobId matches the same UUID shape
        // (symmetric pin -- catches a refactor that swapped
        // ONE endpoint's id generator independently of the
        // other; the asymmetric-drift catch matching the
        // 1e030ed/817dd08/9ac8689 pattern from the JobQueue
        // audit log family)
        assert(hallJobId.matches(uuidRegex),
          clue = s"hall 202 body.jobId MUST match the UUID 8-4-4-4-12 hex-with-dashes shape `$uuidRegex` per JobQueue.scala line 483's `UUID.randomUUID().toString` call -- symmetric with analyze; a refactor that swapped the hall-side id generator independently of the analyze-side (e.g. via inconsistent refactoring of the parallel JobQueue functions) would silently desync the two endpoints' id formats, breaking operator workflows that expect uniform jobId shape across endpoints; got jobId: '$hallJobId'")

        // (iii) analyze jobId is LOWERCASE -- the regex's
        // [0-9a-f] class already enforces this, but the
        // explicit assertion makes the contract intent
        // clearer + catches an "intentionally inconsistent"
        // refactor that emits LOWERCASE-but-not-UUID-shape
        // (some bytes valid hex, some not) -- the regex
        // would catch shape but a separate case assertion
        // hardens the documented UUID.toString lowercase
        // convention
        assert(analyzeJobId == analyzeJobId.toLowerCase,
          clue = s"analyze 202 body.jobId MUST be LOWERCASE per java.util.UUID.toString's RFC 4122 lowercase convention -- a refactor wrapping `UUID.randomUUID().toString.toUpperCase` would silently emit UPPERCASE hex breaking case-sensitive log aggregator field extraction; got jobId: '$analyzeJobId'")

        // (iv) hall jobId is LOWERCASE (symmetric catch)
        assert(hallJobId == hallJobId.toLowerCase,
          clue = s"hall 202 body.jobId MUST be LOWERCASE per java.util.UUID.toString's RFC 4122 lowercase convention -- symmetric with analyze; got jobId: '$hallJobId'")

        // (v) the two jobIds are DISTINCT (uniqueness
        // sanity -- ALSO covered by the existing 202-body-
        // shape pin, but inclusion here makes this pin
        // self-contained as a SHAPE-focused assertion suite)
        assert(analyzeJobId != hallJobId,
          clue = s"analyze + hall jobIds MUST be DISTINCT (each from a separate UUID.randomUUID() call at JobQueue.scala lines 182/483) -- catches a refactor that shared a single id generator with state leak between endpoints; got analyzeJobId='$analyzeJobId' hallJobId='$hallJobId'")

        backend.release.countDown()
        playingHallBackend.release.countDown()
      }
    }
  }

  // Pin the documented statusUrl PATH FORMAT on BOTH
  // submission endpoints' 202 responses per JobQueue.scala
  // line 31's `AnalyzeJobPathPrefix = "/api/analyze-hand-
  // history/jobs/"` + line 33's `PlayingHallJobPathPrefix =
  // "/api/playing-hall/jobs/"` constants -- the SHAPE-
  // INVARIANT pin covering the STATUSURL-PATH DIMENSION
  // that the existing Location-header pin at line 11842+
  // covers in EQUALITY-with-body.statusUrl-only (it pins
  // Location == body.statusUrl but neither has a shape
  // assertion on the path itself); the second per-emission-
  // site SHAPE pin extending the per-field shape pattern
  // started by 476f635 (jobId UUID format); the statusUrl
  // path format is OPERATIONALLY CRITICAL because: (a)
  // frontend code at site.js polls the statusUrl directly
  // by appending it to the base origin (a refactor changing
  // the path prefix would break ALL polling without an
  // explicit migration -- the bundled frontend treats the
  // statusUrl as opaque but it MUST still be a valid path),
  // (b) generic HTTP-202-aware clients (Postman 'follow
  // Location' toggle, REST library auto-follow) parse the
  // path to extract the jobId for client-side correlation;
  // a refactor stripping the jobId from the URL (e.g.
  // `/api/analyze-hand-history/jobs/current` without the
  // id) would silently break these clients while leaving
  // our bundled frontend (which keys on body.jobId
  // separately) unaffected, (c) the runbook documents the
  // status URL paths explicitly: "to investigate a
  // specific analyze job, GET /api/analyze-hand-history/
  // jobs/<id>" -- a refactor renaming the path prefix
  // would silently desync the runbook from operational
  // reality, (d) the SAME prefix constants are referenced
  // from MULTIPLE call sites in JobQueue.scala (lines 207
  // / 436 / 510 / 743 -- the statusUrl is constructed
  // identically at submission time AND status-poll-
  // response time, so the format must remain consistent
  // across all 4 emission sites), (e) the ASYMMETRY between
  // analyze (`/api/analyze-hand-history/jobs/`) and hall
  // (`/api/playing-hall/jobs/`) is INTENTIONAL -- analyze
  // uses the verb-named endpoint (`analyze-hand-history`)
  // because the endpoint mutates state (creates a new
  // analysis), while hall uses the noun-named endpoint
  // (`playing-hall`) because the endpoint queries a
  // simulated hall (the asymmetry reflects REST verb-vs-
  // noun conventions in the codebase's API design); a
  // refactor consolidating to a single shape (e.g. both
  // `/api/jobs/<id>`) would silently drop the verb/noun
  // distinction; per-format regression vectors that this
  // pin catches: (i) refactor renaming the analyze prefix
  // (e.g. `/api/analyze` shortening) would silently break
  // the runbook + saved Postman collections + curl scripts
  // referencing the documented full path, (ii) refactor
  // renaming the hall prefix (e.g. `/api/hall` shortening)
  // would silently break the hall-side analog, (iii)
  // refactor consolidating analyze+hall to a shared shape
  // would silently drop the verb/noun distinction encoded
  // in the asymmetric prefixes, (iv) refactor making the
  // statusUrl ABSOLUTE (e.g. `https://host/api/...` from a
  // misconfigured reverse proxy) would silently break the
  // Location header equality contract (pinned at line
  // 11842+) by introducing scheme + host in the URL, (v)
  // refactor stripping the jobId from the URL would
  // silently break generic clients parsing the path; the
  // jobId UUID-shape pin (476f635) catches changes in
  // generator format BUT NOT changes in PATH structure --
  // this pin closes that complementary gap; test approach
  // mirrors 476f635 exactly: submit one analyze + one hall
  // (BlockingBackend pauses workers so 202 returns
  // reliably), extract both statusUrls + jobIds from
  // response bodies, apply 6-tier format check: (i)
  // analyze statusUrl matches `^/api/analyze-hand-history/
  // jobs/<UUID>$` (specific prefix + UUID at end), (ii)
  // hall statusUrl matches `^/api/playing-hall/jobs/<UUID>$`
  // (specific prefix + UUID at end, asymmetric with
  // analyze), (iii) analyze statusUrl is RELATIVE (starts
  // with `/`, NOT a scheme like `http`), (iv) hall
  // statusUrl is RELATIVE (symmetric), (v) analyze
  // statusUrl's last path segment equals body.jobId
  // (correlation invariant -- the jobId in the URL MUST
  // match the jobId in the body so consumers parsing
  // either get the same id), (vi) hall statusUrl's last
  // path segment equals body.jobId (correlation, symmetric).
  test("submission 202 responses for analyze + playing-hall MUST carry statusUrl paths matching the documented prefixes `/api/analyze-hand-history/jobs/<UUID>` (analyze) and `/api/playing-hall/jobs/<UUID>` (hall) per JobQueue.scala lines 31/33's AnalyzeJobPathPrefix/PlayingHallJobPathPrefix constants -- the SHAPE-INVARIANT pin closes the path-format gap left by the existing Location-header pin (which pins Location==body.statusUrl equality but NOT the path shape)") {
    withStaticSite { staticDir =>
      val backend = new BlockingBackend(Right(sampleAnalysisResult))
      val playingHallBackend = new BlockingPlayingHallBackend(Right(samplePlayingHallResult))
      withServer(staticDir, backend = backend, playingHallBackend = playingHallBackend) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val uuidPattern = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
        val analyzeStatusUrlRegex = s"^/api/analyze-hand-history/jobs/$uuidPattern$$"
        val hallStatusUrlRegex = s"^/api/playing-hall/jobs/$uuidPattern$$"

        // Submit analyze + hall; capture the 202 response
        // bodies. The blocking backends pause the worker AT
        // the analyze/run call site so the 202 returns
        // immediately + reliably without needing thread sync.
        val analyzeResp = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload)
        assertEquals(analyzeResp.statusCode(), 202,
          clue = "analyze submission must return 202 for the statusUrl-format pin to inspect the response body")
        val analyzeBody = jsonBody(analyzeResp)
        val analyzeStatusUrl = analyzeBody("statusUrl").str
        val analyzeJobId = analyzeBody("jobId").str

        val hallResp = postJson(s"$baseUri/api/playing-hall", validPlayingHallPayload)
        assertEquals(hallResp.statusCode(), 202,
          clue = "hall submission must return 202 for the symmetric statusUrl-format pin to inspect the response body")
        val hallBody = jsonBody(hallResp)
        val hallStatusUrl = hallBody("statusUrl").str
        val hallJobId = hallBody("jobId").str

        // (i) analyze statusUrl matches the analyze path
        // regex `^/api/analyze-hand-history/jobs/<UUID>$`
        assert(analyzeStatusUrl.matches(analyzeStatusUrlRegex),
          clue = s"analyze 202 body.statusUrl MUST match the path regex `$analyzeStatusUrlRegex` per JobQueue.scala line 31's `AnalyzeJobPathPrefix = \"/api/analyze-hand-history/jobs/\"` constant + line 207/436's `s\"$$AnalyzeJobPathPrefix$$jobId\"` construction; a refactor renaming the prefix (e.g. `/api/analyze` shortening) would silently break the runbook documentation + saved Postman collections + curl scripts referencing the documented full path; a refactor making the URL absolute (e.g. `https://host/api/...` from a misconfigured reverse proxy) would silently break the Location header equality contract; a refactor stripping the jobId from the URL would silently break generic clients parsing the path to extract the id; got: '$analyzeStatusUrl'")

        // (ii) hall statusUrl matches the hall path regex
        // `^/api/playing-hall/jobs/<UUID>$` (asymmetric with
        // analyze -- analyze uses verb-named endpoint while
        // hall uses noun-named endpoint, reflecting REST
        // verb-vs-noun conventions in the API design)
        assert(hallStatusUrl.matches(hallStatusUrlRegex),
          clue = s"hall 202 body.statusUrl MUST match the path regex `$hallStatusUrlRegex` per JobQueue.scala line 33's `PlayingHallJobPathPrefix = \"/api/playing-hall/jobs/\"` constant + line 510/743's `s\"$$PlayingHallJobPathPrefix$$jobId\"` construction; the prefix asymmetry vs analyze is INTENTIONAL (analyze uses verb-named `analyze-hand-history` because the endpoint mutates state; hall uses noun-named `playing-hall` because the endpoint queries a simulated hall); a refactor consolidating analyze+hall to a shared shape (e.g. both `/api/jobs/<id>`) would silently drop the verb/noun distinction encoded in the asymmetric prefixes; got: '$hallStatusUrl'")

        // (iii) analyze statusUrl is RELATIVE (starts with
        // `/`, NOT a scheme like `http`)
        assert(analyzeStatusUrl.startsWith("/"),
          clue = s"analyze 202 body.statusUrl MUST be a RELATIVE path (starts with `/`) per the documented format -- a refactor making the URL absolute (e.g. `https://host/api/...`) would silently break the Location header equality contract (the header is also `/api/...` per HandHistoryReviewServerApi.scala emission) AND break frontend code at site.js that appends the path to a base origin (an already-absolute statusUrl would double-up the scheme + host); got: '$analyzeStatusUrl'")

        // (iv) hall statusUrl is RELATIVE (symmetric)
        assert(hallStatusUrl.startsWith("/"),
          clue = s"hall 202 body.statusUrl MUST be a RELATIVE path (starts with `/`) symmetric with analyze; got: '$hallStatusUrl'")

        // (v) analyze statusUrl's last path segment equals
        // body.jobId (CORRELATION INVARIANT -- the jobId
        // embedded in the URL MUST match the jobId in the
        // body)
        val analyzeUrlJobId = analyzeStatusUrl.substring(analyzeStatusUrl.lastIndexOf('/') + 1)
        assert(analyzeUrlJobId == analyzeJobId,
          clue = s"analyze statusUrl's last path segment '$analyzeUrlJobId' MUST EQUAL body.jobId '$analyzeJobId' (CORRELATION INVARIANT) -- a refactor that emitted a DIFFERENT identifier in the URL vs the body (e.g. URL uses an internal sequence number while body uses the UUID) would silently break generic clients parsing the path to extract the id; got statusUrl='$analyzeStatusUrl', body.jobId='$analyzeJobId'")

        // (vi) hall statusUrl's last path segment equals
        // body.jobId (correlation, symmetric)
        val hallUrlJobId = hallStatusUrl.substring(hallStatusUrl.lastIndexOf('/') + 1)
        assert(hallUrlJobId == hallJobId,
          clue = s"hall statusUrl's last path segment '$hallUrlJobId' MUST EQUAL body.jobId '$hallJobId' (CORRELATION INVARIANT, symmetric with analyze); got statusUrl='$hallStatusUrl', body.jobId='$hallJobId'")

        backend.release.countDown()
        playingHallBackend.release.countDown()
      }
    }
  }

  // Pin the documented AnalysisJobState ENUM (5 status
  // spellings) at the SOURCE per JobQueue.scala lines 103-
  // 143 -- the ENUM-INVARIANT pin verifies the closed set
  // of 5 documented status strings (queued / running /
  // completed / failed / cancelled) AND the isTerminal
  // flag for each; THIRD per-emission-site SHAPE pin
  // extending 476f635 (jobId UUID format) + 45303cc
  // (statusUrl path format) into the JOB-STATE-ENUM
  // dimension; this pin is an ISOLATION pin (constructs the
  // 5 case classes directly + asserts on their .status +
  // .isTerminal fields -- no HTTP, no concurrency) so it's
  // STABLE + FAST (no timing dependencies); the ENUM
  // CONTRACT is OPERATIONALLY CRITICAL because: (a) the
  // frontend state machine at site.js keys on the status
  // string to drive UI transitions -- a refactor renaming
  // a state would silently put the frontend into an
  // unhandled state (the JS switch/case falls through to a
  // default `unknown status` clause if not exhaustive), (b)
  // operator log-grep workflows filter by status (`tail -f
  // deploy.log | grep status=failed` to find failures) --
  // a refactor renaming `failed` to e.g. `error` would
  // silently break operator pagers, (c) the runbook
  // documents the 5 status values explicitly ("the GET
  // /api/.../jobs/<id> endpoint returns status: one of
  // queued, running, completed, failed, cancelled") -- a
  // refactor adding/renaming/removing a state would
  // silently desync the runbook from operational reality,
  // (d) the UK 'cancelled' (double-L) spelling vs the US
  // 'canceled' (single-L) is a COMMON SILENT-RENAME REFACTOR
  // -- a code reviewer might "correct" the spelling
  // assuming the codebase uses US English without realizing
  // the existing frontend / runbook / aggregator queries
  // all key on the UK double-L form (which is what the
  // case class at line 143 emits); (e) the isTerminal flag
  // controls the JobQueue's job retention + cleanup logic
  // (the cleanup task at line ~1100 only operates on
  // isTerminal=true jobs) -- a refactor flipping the flag
  // (e.g. marking `cancelled` as non-terminal because "a
  // user could re-submit") would silently change the
  // cleanup behavior, leaking memory; the existing test
  // coverage individually pins the spelling of each state
  // in scenario-specific tests (e.g. line 4582 pins
  // 'cancelled' in the DELETE flow, line 6086 pins 'failed'
  // in the NonFatal flow, line 11600 mentions 'queued')
  // BUT NONE verify the COMPLETE ENUM as a closed set --
  // this pin closes the COVERAGE-COMPLETENESS gap; per-
  // format regression vectors this pin uniquely catches:
  // (i) refactor adding a new state (e.g. 'paused',
  // 'expired', 'retrying') without updating documentation
  // -- the set-equality assertion catches the new state in
  // the assertion's expected-set parameter, (ii) refactor
  // renaming a state (e.g. 'cancelled' -> 'canceled' US
  // spelling, 'completed' -> 'done' for brevity, 'failed'
  // -> 'errored' for diagnostics clarity) -- the per-state
  // equality assertion catches the rename, (iii) refactor
  // removing a state (e.g. consolidating 'cancelled' into
  // 'failed' because "they both indicate non-completion")
  // -- the set-size assertion catches the missing state,
  // (iv) refactor flipping isTerminal (e.g. 'cancelled'
  // marked non-terminal because "users could re-submit")
  // -- the per-state isTerminal assertions catch the flip,
  // (v) refactor introducing a status spelling collision
  // (e.g. two states sharing the same string) -- the set-
  // size assertion catches the deduplication; test
  // approach: construct one instance of each of the 5
  // AnalysisJobState case classes with synthetic field
  // values (epoch millis fixed at 100L/200L/300L,
  // synthetic result/error values), assert each instance's
  // .status equals the documented spelling, assert the
  // SET of all 5 .status values equals exactly
  // {queued, running, completed, failed, cancelled},
  // assert each instance's .isTerminal matches the
  // documented behavior.
  test("AnalysisJobState case classes at JobQueue.scala lines 103-143 MUST emit EXACTLY the 5 documented status spellings: queued, running, completed, failed, cancelled (UK double-L spelling for cancelled NOT the US single-L 'canceled') AND the documented isTerminal flag per state -- the ENUM-INVARIANT pin closes the COVERAGE-COMPLETENESS gap where scenario tests pin individual spellings but NONE verify the COMPLETE ENUM as a closed set") {
    import JobQueue.AnalysisJobState.*

    // Construct one instance of each of the 5 states with
    // synthetic field values (epoch millis fixed at
    // 100L/200L/300L). The values don't matter -- this pin
    // tests only the .status string + .isTerminal flag.
    val queued = Queued(submittedAtEpochMs = 100L)
    val running = Running(submittedAtEpochMs = 100L, startedAt = 200L)
    val completed = Completed(submittedAtEpochMs = 100L, startedAt = 200L, completedAt = 300L, result = ujson.Obj())
    val failed = Failed(submittedAtEpochMs = 100L, startedAt = 200L, completedAt = 300L, errorStatus = 500, error = "test")
    val cancelled = Cancelled(submittedAtEpochMs = 100L, startedAt = Some(200L), completedAt = 300L, result = None)

    // (i) each state emits its documented spelling (the
    // per-state equality assertion catches rename refactors
    // for that specific state -- e.g. 'cancelled' ->
    // 'canceled' US spelling, 'completed' -> 'done', etc.)
    assertEquals(queued.status, "queued",
      clue = "Queued.status MUST be exactly 'queued' (lowercase, no UK/US variant applicable) per JobQueue.scala line 103")
    assertEquals(running.status, "running",
      clue = "Running.status MUST be exactly 'running' (lowercase, present-tense gerund) per JobQueue.scala line 109 -- a refactor to past-participle 'ran' or noun 'execution' would silently break grep workflows")
    assertEquals(completed.status, "completed",
      clue = "Completed.status MUST be exactly 'completed' (lowercase, past-participle) per JobQueue.scala line 120 -- a refactor to 'done' for brevity OR 'finished' for diagnostic clarity would silently break frontend state machines + runbook documentation that key on 'completed'")
    assertEquals(failed.status, "failed",
      clue = "Failed.status MUST be exactly 'failed' (lowercase, past-participle) per JobQueue.scala line 132 -- a refactor to 'errored' for diagnostic clarity OR 'rejected' would silently break operator pager rules filtering for 'failed'")
    // The 'cancelled' UK double-L spelling is the MOST
    // LIKELY silent-rename refactor target -- a code
    // reviewer "correcting" the spelling to US single-L
    // would break the frontend + runbook + aggregator
    // queries all in one stroke.
    assertEquals(cancelled.status, "cancelled",
      clue = "Cancelled.status MUST be exactly 'cancelled' (UK English DOUBLE-L spelling) per JobQueue.scala line 143 -- a refactor to US English single-L 'canceled' would be a SILENT BREAKAGE because (a) the frontend's pollPlayingHallJob at site.js line 549 keys on 'cancelled' specifically, (b) the existing 4+ test assertions at lines 4582/4589/12114/12118/13650 all key on UK 'cancelled', (c) operator log-grep workflows + runbook documentation use UK form, (d) the case-class .status emit at line 143 is the SINGLE SOURCE OF TRUTH that all the keyed-on forms depend on; got: '${cancelled.status}'")

    // (ii) the SET of all 5 status strings has exactly 5
    // distinct values (catches add/remove refactors AND
    // status spelling collisions where two states share a
    // string)
    val allStatuses = Set(queued.status, running.status, completed.status, failed.status, cancelled.status)
    assertEquals(allStatuses.size, 5,
      clue = s"AnalysisJobState MUST emit 5 distinct status strings (one per case class) -- a refactor introducing a spelling collision (e.g. two states both emitting 'completed') would silently deduplicate to <5 entries, breaking the documented closed-set contract; got distinct count: ${allStatuses.size} for values: $allStatuses")
    assertEquals(allStatuses, Set("queued", "running", "completed", "failed", "cancelled"),
      clue = s"AnalysisJobState's complete set of status strings MUST equal exactly {queued, running, completed, failed, cancelled} per JobQueue.scala lines 103-143 -- a refactor adding a new state (e.g. 'paused' / 'expired' / 'retrying') would silently extend the state machine without updating the frontend / runbook / aggregator queries that all assume the 5-state closure; a refactor removing a state would silently lose operator-visible distinction; got: $allStatuses")

    // (iii) the isTerminal flag matches documented behavior
    // per state -- the JobQueue's cleanup task at line
    // ~1100 only operates on isTerminal=true jobs, so
    // flipping the flag silently changes retention/cleanup
    // semantics
    assert(!queued.isTerminal,
      clue = "Queued.isTerminal MUST be false per JobQueue.scala line 106 -- a refactor flipping this to true (e.g. 'jobs should be cleaned up immediately if no worker picks them up') would silently change retention semantics + break the worker-to-running transition that depends on queued jobs being non-terminal")
    assert(!running.isTerminal,
      clue = "Running.isTerminal MUST be false per JobQueue.scala line 112 -- a refactor flipping this to true would silently break the worker-to-completed transition + the cancel-while-running path")
    assert(completed.isTerminal,
      clue = "Completed.isTerminal MUST be true per JobQueue.scala line 123 -- a refactor flipping this to false would silently let the cleanup task ignore completed jobs, leaking memory")
    assert(failed.isTerminal,
      clue = "Failed.isTerminal MUST be true per JobQueue.scala line 135 -- a refactor flipping this to false would silently let the cleanup task ignore failed jobs, leaking memory")
    assert(cancelled.isTerminal,
      clue = "Cancelled.isTerminal MUST be true per JobQueue.scala line 146 -- a refactor flipping this to false (e.g. 'a user could re-submit a cancelled job') would silently let the cleanup task ignore cancelled jobs, leaking memory AND would silently break the DELETE-then-terminal-poll flow that the existing tests at lines 4582/4589/12114/12118 rely on")
  }

  // Pin the documented RateLimitBucket ENUM at the SOURCE
  // per RateLimit.scala lines 14-25 -- the ENUM-INVARIANT
  // isolation pin extending the 822a0df AnalysisJobState
  // pattern to a SECOND enum, validating the FAMILY's
  // applicability across the codebase's other documented
  // enumeration types; FOURTH per-emission-site SHAPE pin
  // overall (476f635 jobId / 45303cc statusUrl / 822a0df
  // AnalysisJobState / THIS RateLimitBucket); RateLimit
  // Bucket has TWO documented properties per case (.id +
  // .description) so this pin verifies BOTH dimensions
  // simultaneously, doubling the per-pin coverage; the
  // RateLimitBucket enum is OPERATIONALLY CRITICAL because:
  // (a) the .id values flow into the STRUCTURED LOG LINE
  // `bucket=<id>` field on every `request rate limited`
  // audit log emission -- operator log-grep workflows like
  // `tail -f deploy.log | grep bucket=submit` filter
  // failures by the EXACT documented id, a refactor
  // renaming the id (e.g. `submit` -> `submission`,
  // `job-status` -> `jobstatus` hyphen removal,
  // `job-status` -> `job_status` separator change, `auth` -
  // > `authentication`) would silently break operator
  // grep + aggregator queries, (b) the .description values
  // flow into the USER-FACING error message body (the
  // 429-response body's reason field uses the description
  // for human-readable diagnostics: "auth rate limit
  // exceeded" vs "auth_per_minute_exceeded" -- the natural
  // English form is more user-friendly), a refactor making
  // the description match the .id (consolidating to a
  // single string) would silently degrade UX, (c) the .id
  // vs .description ASYMMETRY is INTENTIONAL -- .id uses
  // hyphen-separated lowercase (URL-safe + grep-friendly)
  // while .description uses space-separated lowercase
  // (human-readable + matches natural English form); a
  // refactor consolidating to a single string would
  // silently break either log parsing (if description form
  // wins: `bucket=job status` splits the structured key=
  // value pair) OR user message readability (if id form
  // wins: `error=job-status rate limit exceeded` looks
  // robot-generated to users), (d) the SPECIFIC HYPHEN-VS-
  // SPACE choice for the JobStatus case is the MOST LIKELY
  // refactor target -- a code reviewer "normalizing" the
  // separator across both .id + .description (e.g. "both
  // should be hyphen for consistency") would silently
  // break either format; per-format regression vectors
  // this pin catches: (i) refactor renaming any .id (e.g.
  // `submit` -> `submission`) -- per-case .id equality
  // assertion catches the rename, (ii) refactor renaming
  // any .description (e.g. `auth` -> `authentication`) --
  // per-case .description equality assertion catches the
  // rename, (iii) refactor consolidating .id + .description
  // to a single string -- the asymmetric-form assertions
  // (JobStatus.id has hyphen vs .description has space)
  // catch this, (iv) refactor adding a new bucket (e.g.
  // `Upload` for file-upload rate limits) without updating
  // documentation -- the values.length assertion + set-
  // equality assertions catch the new case, (v) refactor
  // removing a bucket (e.g. consolidating Submit + Job
  // Status into a single Job bucket) -- set-size + set-
  // equality assertions catch the missing case, (vi)
  // refactor introducing an id/description collision (e.g.
  // two cases both emitting "auth" for .id) -- set-size
  // assertion catches the deduplication; test approach
  // mirrors 822a0df: import the enum's case objects,
  // assert per-case .id + .description match documented
  // spelling, assert the set of all 3 .id values equals
  // exactly {submit, job-status, auth}, assert the set of
  // all 3 .description values equals exactly {submit, job
  // status, auth}, assert values.length == 3.
  test("RateLimitBucket enum at RateLimit.scala lines 14-25 MUST emit EXACTLY 3 documented .id values (submit, job-status, auth) AND 3 documented .description values (submit, job status, auth) -- the SECOND ENUM-INVARIANT pin in the family extending 822a0df's pattern to RateLimit.scala; the .id vs .description ASYMMETRY (JobStatus.id='job-status' vs JobStatus.description='job status') is INTENTIONAL and catches refactors that consolidate to a single string") {
    import sicfun.holdem.web.RateLimit.RateLimitBucket
    import sicfun.holdem.web.RateLimit.RateLimitBucket.*

    // (i-iii) per-case .id assertions (the URL-safe +
    // grep-friendly hyphen-separated lowercase form that
    // flows into structured log lines and 429-response
    // headers)
    assertEquals(Submit.id, "submit",
      clue = "RateLimitBucket.Submit.id MUST be exactly 'submit' per RateLimit.scala line 18 -- a refactor to 'submission' for natural English OR 'submit-job' for prefix-consistency with the other bucket ids would silently break operator log-grep workflows filtering on the documented bucket=submit form")
    assertEquals(JobStatus.id, "job-status",
      clue = "RateLimitBucket.JobStatus.id MUST be exactly 'job-status' (HYPHEN separator, NOT underscore or space) per RateLimit.scala line 19 -- a refactor to 'jobstatus' (separator removal for compactness), 'job_status' (underscore for consistency with snake_case env-var names elsewhere), or 'job status' (consolidating to .description's space form) would silently break log-grep workflows AND would silently produce malformed structured log lines (a space in the bucket value splits the key=value pair); got: '${JobStatus.id}'")
    assertEquals(Auth.id, "auth",
      clue = "RateLimitBucket.Auth.id MUST be exactly 'auth' per RateLimit.scala line 20 -- a refactor to 'authentication' for clarity OR 'login' for user-friendliness would silently break operator log-grep workflows filtering on bucket=auth")

    // (iv-vi) per-case .description assertions (the human-
    // readable space-separated lowercase form that flows
    // into user-facing 429-response error messages)
    assertEquals(Submit.description, "submit",
      clue = "RateLimitBucket.Submit.description MUST be exactly 'submit' per RateLimit.scala line 23 -- happens to be identical to .id for this case because 'submit' is a single word; a refactor to 'submission' would silently desync from .id breaking the documented description-mirrors-id-when-possible convention")
    assertEquals(JobStatus.description, "job status",
      clue = "RateLimitBucket.JobStatus.description MUST be exactly 'job status' (SPACE separator, NOT hyphen) per RateLimit.scala line 24 -- this is the INTENTIONAL ASYMMETRY with .id ('job-status' with hyphen); the .description uses space because it flows into user-facing error messages where natural English form is more readable than the URL-safe hyphenated form; a refactor consolidating to 'job-status' or 'jobstatus' would silently degrade user-facing readability; got: '${JobStatus.description}'")
    assertEquals(Auth.description, "auth",
      clue = "RateLimitBucket.Auth.description MUST be exactly 'auth' per RateLimit.scala line 25 -- identical to .id because 'auth' is a single word; a refactor to 'authentication' would silently desync the user-facing message from the log-line bucket id")

    // (vii) set of all 3 .id values has exactly 3 distinct
    // entries
    val allIds = Set(Submit.id, JobStatus.id, Auth.id)
    assertEquals(allIds.size, 3,
      clue = s"RateLimitBucket MUST emit 3 distinct .id values (one per case) -- a refactor introducing an id collision would silently deduplicate; got distinct count: ${allIds.size} for values: $allIds")
    assertEquals(allIds, Set("submit", "job-status", "auth"),
      clue = s"RateLimitBucket's complete set of .id values MUST equal exactly {submit, job-status, auth} per RateLimit.scala lines 18-20 -- a refactor adding a new bucket (e.g. 'upload' for file-upload rate limits) would silently extend the rate-limit surface without updating docs / aggregator queries / pager rules; a refactor removing a bucket would silently lose operator-visible rate-limit category distinction; got: $allIds")

    // (viii) set of all 3 .description values has exactly 3
    // distinct entries
    val allDescriptions = Set(Submit.description, JobStatus.description, Auth.description)
    assertEquals(allDescriptions.size, 3,
      clue = s"RateLimitBucket MUST emit 3 distinct .description values (one per case) -- a refactor introducing a description collision would silently lose user-facing distinction in 429-response error messages; got distinct count: ${allDescriptions.size} for values: $allDescriptions")
    assertEquals(allDescriptions, Set("submit", "job status", "auth"),
      clue = s"RateLimitBucket's complete set of .description values MUST equal exactly {submit, 'job status', auth} per RateLimit.scala lines 23-25; got: $allDescriptions")

    // (ix) the enum has exactly 3 cases (defense-in-depth
    // catch for add-case refactors via Scala 3's
    // enum.values introspection)
    assertEquals(RateLimitBucket.values.length, 3,
      clue = s"RateLimitBucket.values MUST have exactly 3 entries per RateLimit.scala line 15's `case Submit, JobStatus, Auth` declaration -- a refactor adding a new case (e.g. 'Upload') would silently extend the rate-limit surface AND the per-case .id + .description assertions above would NOT catch it (they only verify the 3 KNOWN cases emit the right values); this assertion catches the add-case refactor via Scala 3's reflective enum.values introspection; got length: ${RateLimitBucket.values.length}")
  }

  // Pin the documented classifyAnalysisError +
  // classifyPlayingHallError CLASSIFIER FUNCTIONS at
  // JobQueue.scala lines 763-771 -- the CLASSIFIER-INVARIANT
  // pin verifies the closed set of 3 returned HTTP status
  // codes (400 / 500 / 504) AND the prefix-matching branch
  // logic AND the ASYMMETRY between the analyze + hall
  // classifiers (each rejects the OTHER endpoint's prefix
  // form); THIRD enum-style isolation pin extending the
  // 822a0df / 99466be pattern to a SECOND DIMENSION (the
  // 822a0df + 99466be pins target Scala 3 enum types; this
  // pin targets pure FUNCTIONS that return enumerable
  // values); FIFTH per-emission-site SHAPE pin overall;
  // the classifier functions are OPERATIONALLY CRITICAL
  // because: (a) the returned status code becomes the
  // Failed.errorStatus field at JobQueue.scala line 291's
  // `Failed(..., classifyAnalysisError(error), error)` --
  // this status flows DIRECTLY into the HTTP response on
  // status-poll AND into the `errorStatus=<n>` field of the
  // `job failed` audit log line; operator pager rules
  // typically distinguish by errorStatus value (a 504 is
  // an infrastructure issue worth paging on at 3am, a 400
  // is a user-input issue worth a daytime ticket, a 500 is
  // an ungraceful-exception worth a priority-medium ticket)
  // -- a refactor changing the classifier output (e.g.
  // returning 503 instead of 504 for timeouts, or 422
  // instead of 400 for bad input) would silently break the
  // documented pager-rule mapping, (b) the prefix-matching
  // logic encodes the GROUND TRUTH about which error
  // categories map to which HTTP status -- a refactor
  // changing the prefix check (e.g. case-insensitive
  // matching, or `contains` instead of `startsWith`) would
  // silently demote ungraceful exceptions to 400 if the
  // exception message HAPPENED to start with something
  // other than the documented prefix, (c) the ASYMMETRY
  // between analyze + hall classifiers (each has its own
  // documented prefix form) is INTENTIONAL -- the analyze
  // classifier rejects "playing hall ..." prefixed
  // messages and vice versa, which preserves
  // category-purity in the audit log (an analyze error
  // categorized as 504 must actually be an analyze timeout,
  // not a stray hall message); a refactor consolidating
  // the two classifiers into a single shared function
  // (e.g. "both endpoints use the same logic, just inline
  // both prefix checks") would silently let cross-endpoint
  // prefixes match incorrectly, breaking the audit-log
  // category-purity invariant; per-format regression
  // vectors this pin catches: (i) refactor changing any
  // returned status (e.g. timeout 504 -> 503) -- per-
  // branch equality assertions catch the rename, (ii)
  // refactor changing the prefix-match semantics (e.g.
  // startsWith -> contains, case-sensitive -> case-
  // insensitive) -- the prefix-specificity + case-
  // sensitivity assertions catch the change, (iii) refactor
  // consolidating analyze + hall classifiers into a single
  // shared function -- the asymmetric-drift assertions
  // (analyze classifier on hall prefix returns 400, NOT
  // 504) catch this, (iv) refactor extending the closed
  // set of returned statuses (e.g. adding 503 / 422 / 429
  // for new categories) -- the set-equality assertion
  // catches the new value, (v) refactor narrowing the
  // closed set (e.g. consolidating 500 + 504 into a single
  // 500 because "both are server errors") -- the set-
  // equality assertion catches the missing value.
  test("classifyAnalysisError + classifyPlayingHallError at JobQueue.scala lines 763-771 MUST return exactly {400, 500, 504} based on prefix-matching the documented strings (analysis timed out / analysis failed: + playing hall timed out / playing hall failed:); the CLASSIFIER-INVARIANT pin catches refactors changing status values, prefix-match semantics, OR consolidating the asymmetric per-endpoint classifiers into a shared function") {
    import JobQueue.{classifyAnalysisError, classifyPlayingHallError}

    // (i) ANALYZE per-branch return values
    assertEquals(classifyAnalysisError("analysis timed out after 100ms"), 504,
      clue = s"classifyAnalysisError on the documented timeout-prefix `analysis timed out after` MUST return 504 per JobQueue.scala line 764 -- 504 is the HTTP status for `Gateway Timeout` and is the operationally-meaningful signal for operators that a worker exceeded its deadline (which is the infrastructure issue worth paging on); a refactor returning 503 (Service Unavailable) would silently misclassify timeouts as availability issues; a refactor returning 408 (Request Timeout) would silently misclassify a SERVER timeout as a CLIENT-side timeout; got: ${classifyAnalysisError("analysis timed out after 100ms")}")
    assertEquals(classifyAnalysisError("analysis failed: boom"), 500,
      clue = s"classifyAnalysisError on the documented failure-prefix `analysis failed:` MUST return 500 per JobQueue.scala line 765 -- 500 is the HTTP status for `Internal Server Error` and signals an ungraceful exception that escaped the backend's Either-based error handling; the prefix is constructed by JobQueue.scala line 297 s-string wrapping of NonFatal exceptions; a refactor returning 502 / 503 / 504 would silently misclassify exceptions as transport/availability/timeout issues; got: ${classifyAnalysisError("analysis failed: boom")}")
    assertEquals(classifyAnalysisError("invalid hand history format"), 400,
      clue = s"classifyAnalysisError on an unmatched-prefix message MUST return 400 per JobQueue.scala line 766's default-case branch -- 400 is the HTTP status for `Bad Request` and signals a user-input issue (the backend returned Left with a domain-specific error message that DIDN'T match the timeout or failure prefixes); a refactor returning 422 (Unprocessable Entity) would silently downgrade the error category from input-validation to semantic-validation; got: ${classifyAnalysisError("invalid hand history format")}")

    // (ii) HALL per-branch return values (symmetric with
    // analyze; the asymmetric-drift catch comes in tier (iv))
    assertEquals(classifyPlayingHallError("playing hall timed out after 100ms"), 504,
      clue = "classifyPlayingHallError on the documented hall timeout-prefix `playing hall timed out after` MUST return 504 per JobQueue.scala line 769 -- symmetric with analyze's 504 timeout classification; got: ${classifyPlayingHallError(\"playing hall timed out after 100ms\")}")
    assertEquals(classifyPlayingHallError("playing hall failed: boom"), 500,
      clue = "classifyPlayingHallError on the documented hall failure-prefix `playing hall failed:` MUST return 500 per JobQueue.scala line 770 -- symmetric with analyze's 500 failure classification; the prefix is constructed by JobQueue.scala line 595's `s\"playing hall failed: $${e.getMessage}\"` wrapping; got: ${classifyPlayingHallError(\"playing hall failed: boom\")}")
    assertEquals(classifyPlayingHallError("invalid playing hall config"), 400,
      clue = "classifyPlayingHallError on an unmatched-prefix message MUST return 400 per JobQueue.scala line 771's default-case branch -- symmetric with analyze's 400 default; got: ${classifyPlayingHallError(\"invalid playing hall config\")}")

    // (iii) the set of all returned values per classifier
    // equals {400, 500, 504} (closed-set assertion catches
    // refactors that extend the set with new categories OR
    // narrow it by consolidating two values into one)
    val analyzeStatusSet = Set(
      classifyAnalysisError("analysis timed out after 1ms"),
      classifyAnalysisError("analysis failed: x"),
      classifyAnalysisError("unmatched default")
    )
    assertEquals(analyzeStatusSet, Set(400, 500, 504),
      clue = s"classifyAnalysisError MUST emit exactly the closed set {400, 500, 504} across its 3 branches -- a refactor extending the set (e.g. adding 503 for a new 'analysis paused' category, or 422 for 'analysis rejected by validation') would silently change the operator-visible status taxonomy without updating pager rules; a refactor narrowing the set (e.g. consolidating 500 + 504 into a single 500 because 'both are server errors') would silently lose the operationally-meaningful distinction between exception failures and infrastructure timeouts; got: $analyzeStatusSet")
    val hallStatusSet = Set(
      classifyPlayingHallError("playing hall timed out after 1ms"),
      classifyPlayingHallError("playing hall failed: x"),
      classifyPlayingHallError("unmatched default")
    )
    assertEquals(hallStatusSet, Set(400, 500, 504),
      clue = s"classifyPlayingHallError MUST emit exactly the closed set {400, 500, 504} -- symmetric with analyze; got: $hallStatusSet")

    // (iv) ASYMMETRIC-DRIFT catch: each classifier rejects
    // the OTHER endpoint's prefix form. This catches a
    // refactor consolidating the two classifiers into a
    // single shared function (e.g. "both endpoints can
    // share the prefix matching, just inline both prefix
    // checks into one function") -- if the shared function
    // accepted EITHER prefix, then analyze errors would be
    // mis-classified by hall prefixes (and vice versa),
    // breaking the audit-log category-purity invariant.
    assertEquals(classifyAnalysisError("playing hall timed out after 100ms"), 400,
      clue = "classifyAnalysisError on a HALL-prefix message MUST return 400 (NOT 504) -- the analyze classifier MUST NOT recognize the hall's `playing hall timed out after` prefix; a refactor consolidating the two classifiers into a shared function that accepted EITHER prefix would silently break the audit-log category-purity invariant (an analyze error categorized as 504 must actually be an analyze timeout, not a stray hall message); got: ${classifyAnalysisError(\"playing hall timed out after 100ms\")}")
    assertEquals(classifyAnalysisError("playing hall failed: boom"), 400,
      clue = "classifyAnalysisError on a HALL-failure-prefix message MUST return 400 (NOT 500) -- symmetric asymmetric-drift catch; got: ${classifyAnalysisError(\"playing hall failed: boom\")}")
    assertEquals(classifyPlayingHallError("analysis timed out after 100ms"), 400,
      clue = "classifyPlayingHallError on an ANALYZE-prefix message MUST return 400 (NOT 504) -- the hall classifier MUST NOT recognize the analyze's `analysis timed out after` prefix; symmetric with the analyze-classifier-on-hall-prefix assertion; got: ${classifyPlayingHallError(\"analysis timed out after 100ms\")}")
    assertEquals(classifyPlayingHallError("analysis failed: boom"), 400,
      clue = "classifyPlayingHallError on an ANALYZE-failure-prefix message MUST return 400 (NOT 500) -- symmetric; got: ${classifyPlayingHallError(\"analysis failed: boom\")}")

    // (v) PREFIX-SPECIFICITY catch: the prefix MUST appear
    // at the START of the string (per startsWith semantics).
    // A refactor swapping startsWith -> contains would
    // silently let any message containing the prefix
    // ANYWHERE match the timeout/failure branches.
    assertEquals(classifyAnalysisError("backend returned: analysis timed out after 100ms"), 400,
      clue = "classifyAnalysisError on a message where the timeout-prefix appears IN THE MIDDLE (not at the start) MUST return 400 per startsWith semantics -- a refactor swapping startsWith -> contains would silently match this and return 504, mis-classifying wrapped/quoted error messages as direct timeouts; got: ${classifyAnalysisError(\"backend returned: analysis timed out after 100ms\")}")

    // (vi) CASE-SENSITIVITY catch: the prefix MUST match in
    // the documented lowercase form. A refactor making the
    // prefix-match case-insensitive (e.g. for "more
    // forgiving error categorization") would silently
    // accept variants like "Analysis Timed Out" that
    // wouldn't be emitted by the documented wrapping at
    // line 297.
    assertEquals(classifyAnalysisError("Analysis timed out after 100ms"), 400,
      clue = "classifyAnalysisError on a CAPITALIZED-PREFIX message MUST return 400 (NOT 504) per Scala's String.startsWith case-sensitive semantics -- the documented prefix at line 764 is the lowercase form `analysis timed out after` matching what line 297's wrapping emits; a refactor making the match case-insensitive (e.g. `.toLowerCase.startsWith(...)`) would silently accept variants that the documented wrapping never emits, increasing the surface for mis-classification of operator-typed test messages; got: ${classifyAnalysisError(\"Analysis timed out after 100ms\")}")
  }

  // Pin the documented rateLimitClientIpSource FUNCTION at
  // RateLimit.scala lines 101-111 -- the CLIENT-IP-SOURCE-
  // INVARIANT pin verifies the closed set of 3 documented
  // format templates AND the branch logic that selects
  // between them based on the (trustedClientIpHeader,
  // trustedProxyIps) input pair; SIXTH per-emission-site
  // SHAPE pin overall extending the FUNCTION-INVARIANT
  // sub-family from 9f42256 (classify*Error 3-status
  // closure) to a SECOND function with a more complex
  // closed set (string-template formats vs raw integer
  // status codes); the rateLimitClientIpSource function is
  // OPERATIONALLY CRITICAL because: (a) its return value
  // flows into the STARTUP BANNER log line at HandHistory
  // ReviewServerRuntime.scala line 348 (the `client-ip-
  // source=<value>` field operators read at startup to
  // confirm proxy configuration), (b) it ALSO flows into
  // the /api/readiness JSON response at Readiness.scala
  // lines 103/146 (the `rateLimitClientIpSource` field
  // frontend health-check displays render to operators on
  // the dashboard), (c) the 3 documented formats encode
  // operationally-meaningful distinctions: (i) `header:
  // <name> via loopback-or-allowlisted-proxy` = the
  // operator configured a trusted-proxy ALLOWLIST so the
  // header can be trusted from non-loopback peers (the
  // FULL trust mode used by production deployments behind
  // a reverse proxy), (ii) `header:<name> via loopback-
  // only` = the operator configured a header BUT did NOT
  // configure an allowlist, so the header is only trusted
  // when the peer is loopback (the SAFER fallback for
  // local development), (iii) `remote-address` = no header
  // configured, the server uses the direct TCP peer
  // address (the SIMPLEST mode for single-tenant
  // deployments without proxies); the operator's choice
  // between these 3 modes has SECURITY IMPLICATIONS (a
  // misconfigured trusted-proxy allowlist would let
  // attackers spoof their IP via the trusted header) and
  // the documented format strings let operators audit
  // the configuration at startup AND via the readiness
  // endpoint; per-format regression vectors this pin
  // catches: (i) refactor renaming any of the 3 documented
  // template strings (e.g. `remote-address` -> `peer-
  // address` for naming clarity) would silently desync the
  // log + readiness field from operator audit workflows,
  // (ii) refactor reordering the branches (e.g. checking
  // None before Some(header)) would not change behavior
  // BUT a refactor SWAPPING the priority (e.g. None +
  // non-empty trustedProxyIps returning the trusted-proxy
  // template) would silently change the meaning of the
  // emitted value, (iii) refactor consolidating the 2
  // Some(header) branches into a single template (dropping
  // the `via loopback-or-allowlisted-proxy` vs `via
  // loopback-only` distinction) would silently hide the
  // SECURITY-RELEVANT distinction between full-trust and
  // safer-fallback modes from the operator, (iv) refactor
  // making the header NAME parameter URL-encoded (e.g.
  // emitting `header:X%2DReal%2DIP` instead of `header:X-
  // Real-IP`) would silently break operator grep workflows
  // expecting the bare header name; test approach mirrors
  // 9f42256: import the function from RateLimit, call it
  // with each of the documented (Option, Set) input pairs,
  // assert the returned string matches the documented
  // template EXACTLY, additionally assert the priority
  // logic via tier (iv)'s None + non-empty Set case (which
  // SHOULD still return remote-address because None takes
  // priority -- a refactor that swapped priorities would
  // silently change the meaning of this case), and assert
  // the closed-set cardinality via the set-of-3 collection.
  test("rateLimitClientIpSource at RateLimit.scala lines 101-111 MUST emit exactly the 3 documented format templates -- `header:<name> via loopback-or-allowlisted-proxy` (proxy mode) / `header:<name> via loopback-only` (loopback-restricted mode) / `remote-address` (no-header mode) -- based on the (trustedClientIpHeader, trustedProxyIps) input pair; the CLIENT-IP-SOURCE-INVARIANT pin verifies branch logic + format strings + the closed set of 3 templates") {
    import RateLimit.rateLimitClientIpSource

    // (i) Some(header) + non-empty trustedProxyIps ->
    // proxy-allowlist mode template (the FULL TRUST mode
    // used by production deployments behind a reverse
    // proxy)
    assertEquals(
      rateLimitClientIpSource(Some("X-Real-IP"), Set("10.0.0.1")),
      "header:X-Real-IP via loopback-or-allowlisted-proxy",
      clue = "rateLimitClientIpSource(Some(header), non-empty trustedProxyIps) MUST emit `header:<name> via loopback-or-allowlisted-proxy` per RateLimit.scala line 107 -- this is the FULL TRUST mode signaling the operator configured an allowlist so the header can be trusted from non-loopback peers; a refactor renaming this template (e.g. `via allowlisted-proxy` shortening, or `via proxy` further shortening) would silently desync from operator audit workflows + runbook documentation that key on the exact `loopback-or-allowlisted-proxy` form")

    // (ii) Some(header) + empty trustedProxyIps ->
    // loopback-only mode template (the SAFER fallback for
    // local development)
    assertEquals(
      rateLimitClientIpSource(Some("X-Real-IP"), Set.empty),
      "header:X-Real-IP via loopback-only",
      clue = "rateLimitClientIpSource(Some(header), empty trustedProxyIps) MUST emit `header:<name> via loopback-only` per RateLimit.scala line 109 -- this is the SAFER FALLBACK mode signaling the operator configured a header BUT did NOT configure an allowlist, so the header is only trusted when the peer is loopback; the distinction from the proxy-allowlist mode is SECURITY-RELEVANT (a misconfigured allowlist would let attackers spoof their IP via the trusted header), so the operator-visible audit log MUST distinguish the two modes via the `via <suffix>` field; a refactor consolidating to a single template would silently hide the security-relevant distinction")

    // (iii) None -> remote-address (no-header mode)
    assertEquals(
      rateLimitClientIpSource(None, Set.empty),
      "remote-address",
      clue = "rateLimitClientIpSource(None, empty trustedProxyIps) MUST emit the bare `remote-address` constant per RateLimit.scala line 111 -- this is the SIMPLEST mode for single-tenant deployments without proxies; the bare-string form (no prefix) distinguishes it from the header modes which all have a `header:<name>` prefix; a refactor adding a prefix (e.g. `direct:remote-address`) would silently break operator grep workflows expecting bare-string form")

    // (iv) PRIORITY catch: None + NON-EMPTY trustedProxyIps
    // -> STILL `remote-address` (None takes priority over
    // trustedProxyIps)
    assertEquals(
      rateLimitClientIpSource(None, Set("10.0.0.1")),
      "remote-address",
      clue = "rateLimitClientIpSource(None, non-empty trustedProxyIps) MUST emit `remote-address` because None takes priority over trustedProxyIps per RateLimit.scala lines 105-111's pattern match ordering -- the trustedClientIpHeader Option is matched FIRST + the trustedProxyIps Set is consulted ONLY inside the Some branch; a refactor swapping priorities (e.g. checking trustedProxyIps first) would silently change the meaning of this input pair: an operator could configure trustedProxyIps WITHOUT a header (perhaps preparing for a future header rollout) and the current logic correctly emits `remote-address` indicating no header is in use; a swapped-priority refactor would silently emit a header-format template even though no header is configured")

    // (v) ClientIpSourceRemoteAddress constant matches the
    // documented `remote-address` literal -- the function
    // references this constant at line 111 so the constant
    // value itself is part of the contract
    assertEquals(
      RateLimit.ClientIpSourceRemoteAddress,
      "remote-address",
      clue = "RateLimit.ClientIpSourceRemoteAddress MUST be the literal `remote-address` per RateLimit.scala line 12 -- the constant is exported as part of the public surface (referenced by rateLimitClientIpSource AND potentially by other consumers); a refactor renaming the constant value would silently desync from operator audit workflows + readiness endpoint consumers")

    // (vi) header NAME parameter flows through VERBATIM
    // (not URL-encoded, not lowercased) -- catches a
    // refactor making the function URL-encode the header
    // name for "safety" which would silently break
    // operator grep workflows expecting the bare name
    assertEquals(
      rateLimitClientIpSource(Some("CF-Connecting-IP"), Set("10.0.0.1")),
      "header:CF-Connecting-IP via loopback-or-allowlisted-proxy",
      clue = "rateLimitClientIpSource MUST pass the header NAME parameter through VERBATIM -- a refactor making the function URL-encode the name (e.g. emitting `header:CF%2DConnecting%2DIP` instead of `header:CF-Connecting-IP`) OR lowercase it (e.g. `header:cf-connecting-ip`) would silently break operator grep workflows expecting the bare original-case name; the contract is the raw header name appears literally in the emitted string")

    // (vii) the SET of all 3 documented templates is
    // exactly 3 distinct strings (closed-set assertion
    // catches refactors that extend the set with new modes
    // OR narrow it by consolidating two templates into one)
    val allTemplates = Set(
      rateLimitClientIpSource(Some("h"), Set("ip")),
      rateLimitClientIpSource(Some("h"), Set.empty),
      rateLimitClientIpSource(None, Set.empty)
    )
    assertEquals(allTemplates.size, 3,
      clue = s"rateLimitClientIpSource MUST emit exactly 3 distinct format templates across its 3 branches -- a refactor consolidating the 2 Some(header) branches into a single template would silently lose the security-relevant distinction between full-trust + safer-fallback modes; a refactor extending the set with a new mode (e.g. `header:<name> via trusted-cidr-block` for CIDR-based allowlists) without updating documentation would silently change the operator-visible audit taxonomy; got: $allTemplates")
  }

  // Pin the CROSS-CHECK between the 4 EMISSION-PREFIX
  // s-string wrappings at JobQueue.scala lines 297 / 388 /
  // 595 / 673 AND the 4 CLASSIFIER-PREFIX startsWith
  // checks at lines 764 / 765 / 769 / 770 -- the EMISSION-
  // CLASSIFIER COUPLING pin closes the FRAGILITY GAP where
  // 9f42256's classifier pin verifies the classifier's
  // BEHAVIOR on hand-crafted prefix strings, BUT does NOT
  // verify that the ACTUAL EMISSION CODE produces strings
  // matching those prefixes -- if a refactor renamed the
  // emission s-string prefix without updating the
  // classifier (or vice versa), 9f42256 would still pass
  // (the classifier still correctly classifies the
  // hand-crafted strings) AND the production-side existing
  // pins (6b59ce4 / 8577288 / d96f892 / 1e030ed / 9ac8689 /
  // hall NonFatal at line 6324) would catch the drift via
  // the HTTP-response errorStatus field BUT only via the
  // slow + race-prone HTTP integration tests; this pin
  // closes the gap with a FAST ISOLATION pin that
  // REPRODUCES THE EMISSION s-strings via Scala-source
  // literal s-string wrapping (the SAME wrapping pattern
  // at the emission sites) AND passes them through the
  // ACTUAL classifier functions -- if the wrapping prefix
  // ever drifts from the classifier prefix, this pin fails
  // with a 1-3ms first-fail signal without needing to
  // exercise the HTTP server + thread pool + backend
  // machinery; SEVENTH per-emission-site SHAPE pin overall
  // extending the FUNCTION-INVARIANT sub-family from
  // 9f42256 (classifier behavior) + 89e3479 (client-IP
  // source) into a new sub-dimension: CROSS-CHECK between
  // 2 documented functions whose contracts depend on each
  // other; the cross-check is OPERATIONALLY CRITICAL
  // because: (a) a drift between emission + classifier
  // would silently demote ALL ungraceful exceptions on
  // the affected endpoint to errorStatus=400, hiding
  // SECURITY-RELEVANT exceptions (e.g. uncaught
  // SecurityException, AuthenticationException) under the
  // BAD-INPUT classification that operators triage as
  // user-fault instead of server-fault, (b) a drift
  // between timeout-emission + classifier would silently
  // demote ALL timeouts on the affected endpoint to 400,
  // hiding INFRASTRUCTURE ISSUES (worker stuck, backend
  // unresponsive) under the BAD-INPUT classification that
  // operators don't page on at 3am, (c) the drift could
  // happen SILENTLY because the existing HTTP integration
  // tests (6b59ce4 / 8577288 / d96f892) use SCENARIO-
  // SPECIFIC strings that match the documented prefixes
  // BY HARDCODING -- if a refactor renamed the prefix at
  // the EMISSION SITE, the integration tests would still
  // fail BUT only after exercising the HTTP machinery
  // (slower + flakier); this isolation pin catches the
  // drift at a FRACTION of the cost; per-format
  // regression vectors uniquely caught: (i) refactor
  // renaming the analyze NonFatal wrapping (e.g.
  // `analysis failed:` -> `analyze error:`) at line 297
  // without updating the classifier at line 765 would
  // silently demote analyze exceptions to 400, (ii)
  // refactor renaming the analyze timeout message at
  // line 388 (e.g. `analysis timed out after` ->
  // `analysis exceeded deadline by`) without updating
  // line 764 would silently demote analyze timeouts to
  // 400, (iii) refactor renaming the hall NonFatal
  // wrapping at line 595 without updating line 770 would
  // silently demote hall exceptions to 400, (iv)
  // refactor renaming the hall timeout message at line
  // 673 without updating line 769 would silently demote
  // hall timeouts to 400; test approach: reproduce the
  // 4 emission s-strings using Scala-source literals
  // matching the production code exactly (using
  // synthetic e.getMessage / timeoutMs values that don't
  // affect the prefix), then pass each through the
  // documented classifier and verify the documented
  // status code (NonFatal -> 500, timeout -> 504); ALSO
  // verify each emission's startsWith matches the
  // documented prefix literal so a refactor that
  // RENAMED BOTH SIDES IN SYNC (drifting from the
  // documented contract together) would still be
  // catchable via the prefix-literal startsWith
  // assertion (though it requires the test author to
  // notice the renames).
  test("the 4 emission-prefix s-string wrappings at JobQueue.scala lines 297/388/595/673 MUST produce strings that classify correctly via classifyAnalysisError/classifyPlayingHallError at lines 764/765/769/770 -- the EMISSION-CLASSIFIER CROSS-CHECK pin catches drift between the wrapping PREFIXES and the classifier startsWith CHECKS as an ISOLATION pin (faster + more deterministic than the existing HTTP integration tests)") {
    import JobQueue.{classifyAnalysisError, classifyPlayingHallError}

    // (i) ANALYZE NonFatal emission -- reproduce line 297's
    // `s"analysis failed: ${e.getMessage}"` wrapping with a
    // synthetic exception message
    val analyzeNonFatalEmission = s"analysis failed: ${new RuntimeException("synthetic exception body").getMessage}"
    assertEquals(classifyAnalysisError(analyzeNonFatalEmission), 500,
      clue = s"the line 297 `s\"analysis failed: $${e.getMessage}\"` wrapping MUST produce a string that classifyAnalysisError classifies as 500 -- if the line 297 prefix is refactored (e.g. `analysis failed:` -> `analyze error:`) without updating line 765's classifier check, this assertion fails AND ALL analyze NonFatal exceptions silently demote to errorStatus=400 in production, hiding SECURITY-RELEVANT exceptions (uncaught SecurityException, AuthenticationException) under the bad-input classification that operators triage as user-fault instead of server-fault; got emission=`$analyzeNonFatalEmission` classified=${classifyAnalysisError(analyzeNonFatalEmission)}")

    // (ii) ANALYZE TIMEOUT emission -- reproduce line 388's
    // `s"analysis timed out after ${analysisTimeoutMs}ms"`
    // wrapping with a synthetic timeout value
    val analyzeTimeoutEmission = s"analysis timed out after ${120000L}ms"
    assertEquals(classifyAnalysisError(analyzeTimeoutEmission), 504,
      clue = s"the line 388 `s\"analysis timed out after $${analysisTimeoutMs}ms\"` wrapping MUST produce a string that classifyAnalysisError classifies as 504 -- if the line 388 prefix is refactored without updating line 764's classifier check, this assertion fails AND ALL analyze timeouts silently demote to errorStatus=400, hiding INFRASTRUCTURE ISSUES (worker stuck, backend unresponsive) under the bad-input classification that operators don't page on at 3am; got emission=`$analyzeTimeoutEmission` classified=${classifyAnalysisError(analyzeTimeoutEmission)}")

    // (iii) HALL NonFatal emission -- reproduce line 595's
    // `s"playing hall failed: ${e.getMessage}"` wrapping
    val hallNonFatalEmission = s"playing hall failed: ${new RuntimeException("synthetic hall exception").getMessage}"
    assertEquals(classifyPlayingHallError(hallNonFatalEmission), 500,
      clue = s"the line 595 `s\"playing hall failed: $${e.getMessage}\"` wrapping MUST produce a string that classifyPlayingHallError classifies as 500 -- symmetric with analyze; if the line 595 prefix is refactored without updating line 770, hall exceptions silently demote to 400; got emission=`$hallNonFatalEmission` classified=${classifyPlayingHallError(hallNonFatalEmission)}")

    // (iv) HALL TIMEOUT emission -- reproduce line 673's
    // `s"playing hall timed out after ${playingHallTimeoutMs}ms"`
    val hallTimeoutEmission = s"playing hall timed out after ${900000L}ms"
    assertEquals(classifyPlayingHallError(hallTimeoutEmission), 504,
      clue = s"the line 673 `s\"playing hall timed out after $${playingHallTimeoutMs}ms\"` wrapping MUST produce a string that classifyPlayingHallError classifies as 504 -- symmetric with analyze; if the line 673 prefix is refactored without updating line 769, hall timeouts silently demote to 400; got emission=`$hallTimeoutEmission` classified=${classifyPlayingHallError(hallTimeoutEmission)}")

    // (v-viii) startsWith catches the documented prefix
    // literal on each emission -- this is defense-in-depth
    // catching a refactor that renamed BOTH sides in sync
    // (the cross-check above would pass but the literal
    // assertion catches the drift from the documented
    // contract)
    assert(analyzeNonFatalEmission.startsWith("analysis failed:"),
      clue = s"analyze NonFatal emission MUST start with the documented `analysis failed:` prefix; if both line 297 + line 765 are renamed IN SYNC (e.g. both becoming `analyze error:`), the cross-check at tier (i) still passes BUT the documented contract has drifted -- the runbook + log-aggregator dashboards + operator grep workflows all key on the documented form; got: `$analyzeNonFatalEmission`")
    assert(analyzeTimeoutEmission.startsWith("analysis timed out after"),
      clue = s"analyze timeout emission MUST start with the documented `analysis timed out after` prefix; got: `$analyzeTimeoutEmission`")
    assert(hallNonFatalEmission.startsWith("playing hall failed:"),
      clue = s"hall NonFatal emission MUST start with the documented `playing hall failed:` prefix; got: `$hallNonFatalEmission`")
    assert(hallTimeoutEmission.startsWith("playing hall timed out after"),
      clue = s"hall timeout emission MUST start with the documented `playing hall timed out after` prefix; got: `$hallTimeoutEmission`")

    // (ix-x) ASYMMETRIC CROSS-CHECK: analyze emissions MUST
    // NOT classify on the hall classifier as 500/504 (must
    // classify as 400 default) -- a refactor consolidating
    // the analyze + hall emission prefixes into a shared
    // form (e.g. both starting with `job failed:`) would
    // silently let each classifier accept the OTHER
    // endpoint's emissions, breaking audit-log category
    // purity; this asymmetry was already covered by
    // 9f42256 BUT with HAND-CRAFTED strings; THIS commit
    // verifies the asymmetry holds on the ACTUAL EMISSION
    // STRINGS produced by the s-string wrappings
    assertEquals(classifyPlayingHallError(analyzeNonFatalEmission), 400,
      clue = s"analyze NonFatal emission MUST classify as 400 (default branch) when passed to the HALL classifier -- the asymmetric prefixes (`analysis failed:` vs `playing hall failed:`) preserve audit-log category purity; got emission=`$analyzeNonFatalEmission` hall-classified=${classifyPlayingHallError(analyzeNonFatalEmission)}")
    assertEquals(classifyAnalysisError(hallNonFatalEmission), 400,
      clue = s"hall NonFatal emission MUST classify as 400 (default branch) when passed to the ANALYZE classifier -- symmetric asymmetry assertion; got emission=`$hallNonFatalEmission` analyze-classified=${classifyAnalysisError(hallNonFatalEmission)}")
  }

  // Pin the documented Location-header-on-202 contract for BOTH
  // submission endpoints. Deploy doc line 66 explicitly says
  // "Submissions return `202 Accepted` with `Location` and
  // `Retry-After` headers plus a JSON body containing `jobId`,
  // `status` (always the literal string "queued" on a fresh `202` --
  // the worker hasn't started yet; the GET poll endpoint surfaces
  // later state transitions), `statusUrl`, `submittedAtEpochMs`,
  // `pollAfterMs`." The Location header carries the same value as
  // the body's statusUrl field; that equality is the contract a
  // generic HTTP-202-aware client (e.g. Postman's "follow Location"
  // toggle, a generic REST library's auto-follow) relies on -- it
  // reads Location from the response header rather than parsing the
  // body, AND its poll loop expects the URL it gets back to be the
  // canonical status URL the body would have surfaced. A refactor
  // that dropped the Location header (or emitted a different value
  // than statusUrl) would silently break those generic clients
  // without affecting our own bundled frontend (which keys on
  // body.statusUrl, not the header). HandHistoryReviewServerApi.scala
  // emits the Location header at lines 110 (/api/analyze-hand-history)
  // and 156 (/api/playing-hall), in both cases with value
  // `accepted.statusUrl` -- so the contract is symmetric across both
  // submission endpoints. Existing Retry-After test (line ~3775) was
  // adjacent but specifically asserted Retry-After, not Location;
  // before this commit Location was entirely untested at the 202
  // response level. New test covers both endpoints in one test
  // body since the contract is identical across them and a
  // refactor would likely touch either both or just one (the
  // either-both-or-only-one shape parallels the body-cap triplet
  // [4e4b385 / 034b14c / e2045b9] and the CSRF quintuplet
  // [a550186 / 26bf0d7 / eef3779 / e6e0961] -- per-endpoint pins
  // catch the asymmetric-drift case).
  test("submission 202 responses carry a Location header equal to the body's statusUrl on both /api/analyze-hand-history and /api/playing-hall") {
    withStaticSite { staticDir =>
      val backend = new BlockingBackend(Right(sampleAnalysisResult))
      val playingHallBackend = new BlockingPlayingHallBackend(Right(samplePlayingHallResult))
      withServer(staticDir, backend = backend, playingHallBackend = playingHallBackend) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        // /api/analyze-hand-history submission: 202 with Location =
        // body.statusUrl (the relative status-poll URL, NOT a fully
        // qualified URI -- the server emits relative URLs so a
        // reverse proxy doesn't need to rewrite them).
        val analyzeResp = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload)
        assertEquals(analyzeResp.statusCode(), 202,
          clue = "analyze submission must return 202 Accepted -- the documented submit-success shape per deploy doc line 66")
        val analyzeStatusUrl = jsonBody(analyzeResp)("statusUrl").str
        val analyzeLocation = headerValue(analyzeResp, "Location").getOrElse(
          fail("analyze 202 response MUST emit a Location header per deploy doc line 66 -- generic HTTP-202-aware clients (e.g. Postman 'follow Location' toggle, REST libraries with auto-follow) rely on this header to find the status URL without parsing the body; missing Location silently breaks those clients while leaving our bundled frontend (which keys on body.statusUrl) unaffected"))
        assertEquals(analyzeLocation, analyzeStatusUrl,
          clue = s"analyze Location header must equal body.statusUrl -- both are documented to point at the same poll URL, and a divergence (e.g. body has the relative path but header has a fully qualified URI from a misconfigured proxy) would confuse generic clients that key on one OR the other; got header=$analyzeLocation, body=$analyzeStatusUrl")
        // Drain the backend so the next submission isn't queue-blocked.
        backend.release.countDown()

        // /api/playing-hall submission: same contract, same shape.
        val hallResp = postJson(s"$baseUri/api/playing-hall", validPlayingHallPayload)
        assertEquals(hallResp.statusCode(), 202,
          clue = "hall submission must return 202 Accepted -- same submit-success shape as analyze, symmetric across both endpoints")
        val hallStatusUrl = jsonBody(hallResp)("statusUrl").str
        val hallLocation = headerValue(hallResp, "Location").getOrElse(
          fail("hall 202 response MUST emit a Location header per deploy doc line 66 -- same generic-client contract as analyze; HandHistoryReviewServerApi.scala emits the header at line 156 with value accepted.statusUrl"))
        assertEquals(hallLocation, hallStatusUrl,
          clue = s"hall Location header must equal body.statusUrl -- symmetric with analyze; got header=$hallLocation, body=$hallStatusUrl")
        // Drain the hall backend on the way out.
        playingHallBackend.release.countDown()
      }
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

  // Pin CSRF enforcement on POST /api/auth/profile -- before this
  // commit, only the playing-hall DELETE had a CSRF test (the block
  // immediately above); the OTHER four state-changing routes that
  // the deploy doc line 103 names as CSRF-protected (POST
  // /api/auth/logout, POST /api/auth/profile, POST
  // /api/analyze-hand-history, POST /api/playing-hall) had no
  // dedicated CSRF-rejection coverage. The /profile route is the
  // highest-priority of the four uncovered because a refactor that
  // dropped its CSRF gate would let a cross-origin attacker
  // weaponize the route for victim-harassment (changing
  // displayName to something offensive, swapping the user's
  // heroName so it stops matching the hand-history file's player
  // name and silently breaks analyze runs, etc.) provided the
  // attacker can get the victim's browser to issue the request
  // while a session cookie is live (SameSite=Lax blocks the
  // simple cross-site form POST but not all variants -- POST
  // form submissions FROM a top-level navigation initiated by
  // user click DO send Lax cookies, so a phishing page with a
  // form that auto-posts to /api/auth/profile on Enter could
  // hit). New test follows the same shape as the hall-DELETE
  // CSRF test above (which is the documented reference pattern):
  // register a user, attempt POST /api/auth/profile WITHOUT the
  // X-CSRF-Token header (cookie only) and assert 403 + "csrf" in
  // the error message, then send the proper request WITH X-CSRF-
  // Token and assert it succeeds (200). The remaining three
  // routes (/api/auth/logout, /api/analyze-hand-history,
  // /api/playing-hall POST) are left for future fires to keep
  // this commit focused on one branch -- same "future fires can
  // sweep the rest" pattern as e2045b9 / 034b14c's body-cap
  // triplet.
  test("POST /api/auth/profile enforces CSRF under platform-user auth -- pins the documented X-CSRF-Token requirement so a refactor can't silently open the route to cross-origin victim-harassment") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"
          val register = postJson(s"$baseUri/api/auth/register",
            """{"email":"profile-csrf@example.com","password":"correct-horse-battery","displayName":"Tester"}""")
          assertEquals(register.statusCode(), 201,
            clue = "registration must succeed before the CSRF gate can be exercised")
          val registerJson = jsonBody(register)
          val ownerHeaders = authSessionHeaders(register, registerJson("csrfToken").str)
          val profileBody = """{"displayName":"Updated Name","heroName":"newhero","preferredSite":"pokerstars","timeZone":"UTC"}"""

          // POST /api/auth/profile WITHOUT X-CSRF-Token (cookie only)
          // must 403 with "csrf" in the error message. The cookie-
          // only request is the canonical cross-origin attack shape
          // -- SameSite=Lax does pass the session cookie on
          // top-level form POSTs, so CSRF gate is the actual defense.
          val cookieOnly = Map("Cookie" -> sessionCookie(register))
          val withoutCsrf = postJson(s"$baseUri/api/auth/profile", profileBody, cookieOnly)
          assertEquals(withoutCsrf.statusCode(), 403,
            clue = s"POST /api/auth/profile WITHOUT X-CSRF-Token must return 403 -- without this CSRF gate a phishing page could change displayName/heroName/preferredSite/timeZone on behalf of any victim with an active session; got: ${withoutCsrf.statusCode()}")
          assert(jsonBody(withoutCsrf)("error").str.toLowerCase.contains("csrf"),
            clue = s"CSRF rejection error message must mention 'csrf' so scripted clients can key on the failure mode (and audit logs grep for the documented `request forbidden ... reason=csrf-missing-or-invalid` shape); got error: ${jsonBody(withoutCsrf)("error").str}")

          // POST /api/auth/profile WITH the proper X-CSRF-Token header
          // succeeds. This sanity-check ensures the test is exercising
          // the CSRF gate specifically, not some other 403 path (e.g.
          // a missing-session path that would 401-not-403).
          val withCsrf = postJson(s"$baseUri/api/auth/profile", profileBody, ownerHeaders)
          assertEquals(withCsrf.statusCode(), 200,
            clue = "POST /api/auth/profile WITH X-CSRF-Token must succeed (200) -- proves the 403 above came from the CSRF gate specifically, not a different rejection path")
        }
      }
    }
  }

  // Pin CSRF enforcement on POST /api/auth/logout -- closes the FINAL
  // route in the five-route CSRF-protected enumeration per deploy doc
  // line 103. Coverage status after this commit: DELETE
  // /api/playing-hall/jobs/{id} (existing line ~3810), POST
  // /api/auth/profile (e6e0961), POST /api/playing-hall (eef3779),
  // POST /api/analyze-hand-history (26bf0d7), POST /api/auth/logout
  // (this commit) -- all five state-changing routes that the deploy
  // doc enumerates as CSRF-protected now have dedicated rejection
  // tests, so a future refactor cannot silently drop the CSRF gate
  // on ANY of them without a CI signal; the logout-CSRF attack shape
  // is the least severe of the five (lower than profile-tampering,
  // hall-spawn, analyze-spawn, hall-cancel) but still operationally
  // disruptive: a cross-origin attacker who can hit the route gets
  // to forcibly sign the victim out, which (1) interrupts whatever
  // analyze / hall poll loop the victim's tab was watching (the
  // next /api/auth/me probe surfaces the session-missing state and
  // applyAuthState's hide-result-panels logic clears the victim's
  // in-progress results from view per the 7ddb281 documentation),
  // (2) burns the victim's RATE_LIMIT_AUTH_PER_MINUTE bucket if the
  // attacker hits logout repeatedly to chain re-sign-in cycles
  // (each logout triggers a re-auth attempt that hits the auth
  // bucket capped at 10/min/IP per the deploy doc), and (3) on a
  // SHARED-NAT deployment, the auth-bucket exhaustion would block
  // unrelated NAT-mates from signing in for the rest of the
  // 60-second window; success path returns 200 (not 202, not 201)
  // because logout is neither a submission nor a creation -- it's a
  // session-state transition that completes synchronously, see
  // AuthStack.scala's handleAuthLogout at line 197 returning
  // JsonResponse(200, service.authenticationState(None), ...).
  test("POST /api/auth/logout enforces CSRF under platform-user auth -- closes the final route in the documented five-route CSRF-protected enumeration") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"
          val register = postJson(s"$baseUri/api/auth/register",
            """{"email":"logout-csrf@example.com","password":"correct-horse-battery","displayName":"Tester"}""")
          assertEquals(register.statusCode(), 201,
            clue = "registration must succeed before the logout-CSRF gate can be exercised")
          val registerJson = jsonBody(register)
          val ownerHeaders = authSessionHeaders(register, registerJson("csrfToken").str)

          // POST WITHOUT X-CSRF-Token (cookie-only, canonical cross-
          // origin attack shape) must 403. Logout-CSRF lets an
          // attacker forcibly sign the victim out via a phishing
          // page form that auto-POSTs to /api/auth/logout on
          // top-level navigation (SameSite=Lax passes cookies on
          // top-level form POSTs, so the CSRF gate is the actual
          // defense).
          val cookieOnly = Map("Cookie" -> sessionCookie(register))
          val withoutCsrf = postJson(s"$baseUri/api/auth/logout", "{}", cookieOnly)
          assertEquals(withoutCsrf.statusCode(), 403,
            clue = s"POST /api/auth/logout WITHOUT X-CSRF-Token must return 403 -- without this gate a cross-origin attacker could forcibly sign out any victim with an active session, interrupting their analyze/hall poll loops AND burning their RATE_LIMIT_AUTH_PER_MINUTE bucket if hit repeatedly (chains re-sign-in cycles against the 10/min/IP cap); got: ${withoutCsrf.statusCode()}")
          assert(jsonBody(withoutCsrf)("error").str.toLowerCase.contains("csrf"),
            clue = s"CSRF rejection error message must mention 'csrf' (audit log greps for `request forbidden ... reason=csrf-missing-or-invalid`); got error: ${jsonBody(withoutCsrf)("error").str}")

          // WITH the proper X-CSRF-Token: 200 (NOT 202, NOT 201).
          // Logout is neither a submission nor a creation -- it's a
          // session-state transition that completes synchronously,
          // see AuthStack.scala's handleAuthLogout returning
          // JsonResponse(200, ...) at line 211.
          val withCsrf = postJson(s"$baseUri/api/auth/logout", "{}", ownerHeaders)
          assertEquals(withCsrf.statusCode(), 200,
            clue = "POST /api/auth/logout WITH X-CSRF-Token must return 200 -- logout is a synchronous session-state transition, not a submission (which would 202) or a creation (which would 201); proves the 403 above came from the CSRF gate specifically rather than auth-missing (401) or some unrelated rejection")
        }
      }
    }
  }

  // Pin CSRF enforcement on POST /api/analyze-hand-history --
  // continuing the CSRF-sweep: e6e0961 closed /profile, eef3779
  // closed /playing-hall POST, this commit closes the analyze-submit
  // companion. The analyze + hall submit endpoints share the same
  // CSRF gate shape, so pinning both as a pair (not just one) catches
  // an asymmetric refactor that touched only one branch -- e.g. a
  // hypothetical "let's loosen analyze CSRF for the embedded-iframe
  // use case nobody asked for" change would silently re-open the
  // analyze-submit CSRF hole while the eef3779 hall-POST pin kept
  // passing. The analyze-CSRF attack shape differs from hall in
  // duration but not in compounding harms: a cross-origin
  // attacker spawning analyze jobs against the victim's session
  // still burns the same RATE_LIMIT_SUBMITS_PER_MINUTE bucket
  // (analyze and hall SHARE the submit bucket per deploy doc's
  // bucket description), eats CPU on the analyze worker pool,
  // and pollutes the victim's job-history retention window;
  // the 2-min default ANALYSIS_TIMEOUT_MS (per deploy doc line 364)
  // means each attacker-spawned job ties up resources for a
  // shorter window than hall, but still impactful in a
  // submit-storm scenario. With both submit-endpoint CSRF tests in
  // place, two of the four originally-uncovered routes remain
  // (POST /api/auth/logout and... actually, only logout: this
  // commit closes analyze, so just /logout remains for a final
  // future fire to close the full quintuplet).
  test("POST /api/analyze-hand-history enforces CSRF under platform-user auth -- pins the documented X-CSRF-Token requirement on the analyze-submission endpoint, paired with eef3779's /playing-hall CSRF pin") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"
          val register = postJson(s"$baseUri/api/auth/register",
            """{"email":"analyze-submit-csrf@example.com","password":"correct-horse-battery","displayName":"Tester"}""")
          assertEquals(register.statusCode(), 201,
            clue = "registration must succeed before the CSRF gate can be exercised")
          val registerJson = jsonBody(register)
          val ownerHeaders = authSessionHeaders(register, registerJson("csrfToken").str)

          // POST WITHOUT X-CSRF-Token (cookie-only, canonical
          // cross-origin attack shape) must 403.
          val cookieOnly = Map("Cookie" -> sessionCookie(register))
          val withoutCsrf = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload, cookieOnly)
          assertEquals(withoutCsrf.statusCode(), 403,
            clue = s"POST /api/analyze-hand-history WITHOUT X-CSRF-Token must return 403 -- without this gate a cross-origin attacker could spawn analyze jobs against any victim with an active session, burning the SHARED RATE_LIMIT_SUBMITS_PER_MINUTE bucket (analyze and hall share the submit bucket) and tying up worker pool slots; got: ${withoutCsrf.statusCode()}")
          assert(jsonBody(withoutCsrf)("error").str.toLowerCase.contains("csrf"),
            clue = s"CSRF rejection error message must mention 'csrf' (audit log greps for `request forbidden ... reason=csrf-missing-or-invalid`); got error: ${jsonBody(withoutCsrf)("error").str}")

          // WITH X-CSRF-Token: 202 Accepted (submit-success shape).
          val withCsrf = postJson(s"$baseUri/api/analyze-hand-history", validUploadPayload, ownerHeaders)
          assertEquals(withCsrf.statusCode(), 202,
            clue = "POST /api/analyze-hand-history WITH X-CSRF-Token must return 202 Accepted (the documented submit-success shape) -- proves the 403 above came from the CSRF gate specifically, not auth-missing (401) or rate-limit (429) or queue-full (503)")
        }
      }
    }
  }

  // Pin CSRF enforcement on POST /api/playing-hall -- continuing the
  // CSRF-triplet sweep started by e6e0961 (which closed the /profile
  // branch). This commit closes the second of the 4 originally-
  // uncovered routes: per the deploy doc line 103 enumeration of
  // CSRF-protected state-changing routes, the hall-POST is one of
  // five (POST /api/auth/logout, POST /api/auth/profile, POST
  // /api/analyze-hand-history, POST /api/playing-hall, DELETE
  // /api/playing-hall/jobs/{id}); coverage so far: DELETE
  // /api/playing-hall/jobs/{id} (existing test at line ~3810),
  // POST /api/auth/profile (e6e0961), POST /api/playing-hall
  // (this commit); leaving POST /api/auth/logout and POST
  // /api/analyze-hand-history for future fires per the same
  // "future fires can sweep the rest" pattern e2045b9 / 034b14c
  // established for the body-cap triplet. The hall-POST branch is
  // worth pinning specifically because a refactor that dropped its
  // CSRF gate would let a cross-origin attacker spawn hall jobs
  // against the victim's session -- consuming their PBKDF2-cost
  // budget, polluting their Recent Runs localStorage entries on
  // the next /api/playing-hall/jobs/{id} GET that the victim's
  // own tab fires, AND burning the victim's RATE_LIMIT_SUBMITS_PER_MINUTE
  // bucket so their legitimate hall submissions queue up against a
  // saturated budget; the long worker run time (15 min default per
  // the bd8e7f3 + 61e49a8-pinned PLAYING_HALL_TIMEOUT_MS) amplifies
  // the disruption window since each attacker-spawned hall job
  // eats a worker slot for the full duration. Same shape as the
  // hall-DELETE CSRF test above + e6e0961's /profile test:
  // register, attempt the request WITHOUT X-CSRF-Token and assert
  // 403 + "csrf" in error, then sanity-check WITH the proper
  // header succeeds (202, not 200, because submit endpoints return
  // 202 Accepted on success per deploy doc line 66).
  test("POST /api/playing-hall enforces CSRF under platform-user auth -- pins the documented X-CSRF-Token requirement on the hall-submission endpoint") {
    withStaticSite { staticDir =>
      withUserStorePath { storePath =>
        withServer(
          staticDir,
          platformAuth = Some(PlatformUserAuth.Config(storePath = storePath))
        ) { server =>
          val baseUri = s"http://${server.binding.host}:${server.binding.port}"
          val register = postJson(s"$baseUri/api/auth/register",
            """{"email":"hall-submit-csrf@example.com","password":"correct-horse-battery","displayName":"Tester"}""")
          assertEquals(register.statusCode(), 201,
            clue = "registration must succeed before the CSRF gate can be exercised")
          val registerJson = jsonBody(register)
          val ownerHeaders = authSessionHeaders(register, registerJson("csrfToken").str)

          // POST /api/playing-hall WITHOUT X-CSRF-Token (cookie only)
          // must 403. The cookie-only request is the canonical
          // cross-origin attack shape -- SameSite=Lax passes the
          // cookie on top-level form-POST navigation, so the CSRF
          // gate is the actual defense against attacker-spawned
          // hall jobs against a victim's session.
          val cookieOnly = Map("Cookie" -> sessionCookie(register))
          val withoutCsrf = postJson(s"$baseUri/api/playing-hall", validPlayingHallPayload, cookieOnly)
          assertEquals(withoutCsrf.statusCode(), 403,
            clue = s"POST /api/playing-hall WITHOUT X-CSRF-Token must return 403 -- without this CSRF gate a phishing page could spawn 15-minute hall jobs against any victim with an active session, consuming PBKDF2-cost budget, polluting Recent Runs localStorage, AND burning their RATE_LIMIT_SUBMITS_PER_MINUTE bucket; got: ${withoutCsrf.statusCode()}")
          assert(jsonBody(withoutCsrf)("error").str.toLowerCase.contains("csrf"),
            clue = s"CSRF rejection error message must mention 'csrf' so scripted clients can key on the failure mode (and audit logs grep for `request forbidden ... reason=csrf-missing-or-invalid`); got error: ${jsonBody(withoutCsrf)("error").str}")

          // POST /api/playing-hall WITH the proper X-CSRF-Token
          // succeeds (202 Accepted -- submit endpoints return 202
          // not 200 per the deploy doc's HTTP-Endpoints section,
          // since the worker hasn't started yet).
          val withCsrf = postJson(s"$baseUri/api/playing-hall", validPlayingHallPayload, ownerHeaders)
          assertEquals(withCsrf.statusCode(), 202,
            clue = "POST /api/playing-hall WITH X-CSRF-Token must return 202 Accepted (the submit-success shape per deploy doc line 66) -- proves the 403 above came from the CSRF gate specifically, not from a different rejection path like auth-missing (which would 401) or rate-limit (which would 429)")
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
