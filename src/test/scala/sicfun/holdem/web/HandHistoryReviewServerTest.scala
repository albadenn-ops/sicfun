package sicfun.holdem.web

import sicfun.holdem.runtime.TexasHoldemPlayingHall

import munit.FunSuite
import ujson.Value

import java.net.{InetAddress, URI}
import java.net.http.{HttpClient, HttpRequest, HttpResponse}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import java.util.Base64
import java.util.concurrent.{CountDownLatch, TimeUnit}
import scala.jdk.CollectionConverters.*

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

        val oversizedPayload = s"""{"handHistoryText":"${"A" * 256}"}"""
        val oversizedResponse = postJson(s"$baseUri/api/analyze-hand-history", oversizedPayload)
        assertEquals(oversizedResponse.statusCode(), 413)
        assert(oversizedResponse.body().contains("max upload size"))
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
    assert(HandHistoryReviewServer.trustsRateLimitClientIpHeader(Some(InetAddress.getByName("127.0.0.1")), Set.empty))
    assert(HandHistoryReviewServer.trustsRateLimitClientIpHeader(Some(InetAddress.getByName("::1")), Set.empty))
    assert(!HandHistoryReviewServer.trustsRateLimitClientIpHeader(Some(InetAddress.getByName("203.0.113.10")), Set.empty))
    assert(HandHistoryReviewServer.trustsRateLimitClientIpHeader(
      Some(InetAddress.getByName("203.0.113.10")),
      Set("203.0.113.10")
    ))
    assert(!HandHistoryReviewServer.trustsRateLimitClientIpHeader(None, Set("203.0.113.10")))
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
      val backend = new BusyBackend(runForMs = 400L, result = Right(sampleAnalysisResult))
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

        assert(backend.finished.await(3, TimeUnit.SECONDS), "busy backend never finished")
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
      maxConcurrentJobs: Int = 2,
      maxQueuedJobs: Int = 8,
      analysisTimeoutMs: Long = 120000L,
      shutdownGraceMs: Long = 5000L,
      rateLimitSubmitsPerMinute: Int = 6,
      rateLimitStatusPerMinute: Int = 240,
      rateLimitClientIpHeader: Option[String] = None,
      rateLimitTrustedProxyIps: Set[String] = Set.empty,
      drainSignalFile: Option[Path] = None,
      basicAuth: Option[HandHistoryReviewServer.BasicAuthConfig] = None,
      platformAuth: Option[PlatformUserAuth.Config] = None
  )(run: HandHistoryReviewServer.RunningServer => A): A =
    val server = HandHistoryReviewServer.startWithBackend(
      HandHistoryReviewServer.ServerConfig(
        host = "127.0.0.1",
        port = 0,
        staticDir = staticDir,
        maxUploadBytes = maxUploadBytes,
        analysisTimeoutMs = analysisTimeoutMs,
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
      backend
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
      if lastStatus == "completed" || lastStatus == "failed" then
        return body
      Thread.sleep(50)
    fail(s"job did not reach a terminal state, last status=$lastStatus")

  private def immediateBackend(result: Either[String, Value]): HandHistoryReviewServer.AnalysisBackend =
    new HandHistoryReviewServer.AnalysisBackend:
      override def analyze(request: HandHistoryReviewService.AnalysisRequest): Either[String, Value] =
        result

  private final class BlockingBackend(result: Either[String, Value]) extends HandHistoryReviewServer.AnalysisBackend:
    val started = new CountDownLatch(1)
    val release = new CountDownLatch(1)

    override def analyze(request: HandHistoryReviewService.AnalysisRequest): Either[String, Value] =
      started.countDown()
      if !release.await(5, TimeUnit.SECONDS) then
        Left("analysis failed: blocking backend timed out")
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
