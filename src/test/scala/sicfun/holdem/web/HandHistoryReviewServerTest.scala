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
