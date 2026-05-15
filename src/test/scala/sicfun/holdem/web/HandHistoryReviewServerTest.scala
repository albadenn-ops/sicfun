package sicfun.holdem.web

import sicfun.holdem.runtime.TexasHoldemPlayingHall

import munit.FunSuite
import ujson.Value

import java.io.ByteArrayInputStream
import java.net.{InetAddress, URI}
import java.net.http.{HttpClient, HttpRequest, HttpResponse}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
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

  test("log message sanitization escapes line-structural characters") {
    import HandHistoryReviewServerRuntime.sanitizeLogMessage

    // Newline, carriage return, and backslash get escaped so user-controlled values
    // flowing into log interpolation can't forge a fake log line.
    assertEquals(sanitizeLogMessage("safe path"), "safe path")
    assertEquals(sanitizeLogMessage("path with\nnewline"), "path with\\nnewline")
    assertEquals(sanitizeLogMessage("CR\rLF\n"), "CR\\rLF\\n")
    assertEquals(sanitizeLogMessage("escape \\ first so \\n stays literal"),
      "escape \\\\ first so \\\\n stays literal",
      clue = "backslashes must be escaped before \\n so a literal \\n in input doesn't decode as newline")
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
        assertEquals(readyJson("rateLimitClientIpSource").str, "remote-address")
        assertEquals(readyJson("timedOutWorkersInFlight").num.toInt, 0)

        val index = get(s"$baseUri/")
        assertEquals(index.statusCode(), 200)
        assert(index.body().contains("Runtime smoke page"))
        assertEquals(headerValue(index, "X-Content-Type-Options"), Some("nosniff"))
        assertEquals(headerValue(index, "X-Frame-Options"), Some("DENY"))
        assert(headerValue(index, "Content-Security-Policy").exists(_.contains("default-src 'self'")))
        val permissionsPolicy = headerValue(index, "Permissions-Policy").getOrElse(
          fail("expected Permissions-Policy header on index response"))
        assert(permissionsPolicy.contains("camera=()"), s"missing camera=() in: $permissionsPolicy")
        assert(permissionsPolicy.contains("microphone=()"), s"missing microphone=() in: $permissionsPolicy")
        assert(permissionsPolicy.contains("geolocation=()"), s"missing geolocation=() in: $permissionsPolicy")
        assert(permissionsPolicy.contains("interest-cohort=()"), s"missing interest-cohort=() in: $permissionsPolicy")
        assert(permissionsPolicy.contains("browsing-topics=()"),
          s"missing browsing-topics=() (modern Topics-API opt-out, replacement for FLoC) in: $permissionsPolicy")
        assert(permissionsPolicy.contains("usb=()"), s"missing usb=() in: $permissionsPolicy")

        val oversizedPayload = s"""{"handHistoryText":"${"A" * 256}"}"""
        val oversizedResponse = postJson(s"$baseUri/api/analyze-hand-history", oversizedPayload)
        assertEquals(oversizedResponse.statusCode(), 413)
        assert(oversizedResponse.body().contains("max upload size"))
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

  test("static handler answers OPTIONS with 200 and Allow header") {
    withStaticSite { staticDir =>
      withServer(staticDir) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"
        val request = HttpRequest.newBuilder(URI.create(s"$baseUri/"))
          .method("OPTIONS", HttpRequest.BodyPublishers.noBody())
          .build()
        val response = httpClient.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8))
        assertEquals(response.statusCode(), 200)
        assertEquals(response.body(), "")
        assertEquals(headerValue(response, "Allow"), Some("GET, HEAD, OPTIONS"))
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

          val callback = get(s"$baseUri${provider.callbackPath}?state=$state&code=test-code")
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
