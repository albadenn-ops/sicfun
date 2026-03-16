package sicfun.holdem.web

import sicfun.holdem.runtime.TexasHoldemPlayingHall

import munit.FunSuite
import ujson.Value

import java.net.URI
import java.net.http.{HttpClient, HttpRequest, HttpResponse}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
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
      |      "hints": ["test hint"]
      |    }
      |  ]
      |}""".stripMargin
  )
  private val validUploadPayload = """{"handHistoryText":"PokerStars Hand #1","site":"auto","heroName":"Hero"}"""

  test("start serves health/static content and rejects oversized uploads") {
    withStaticSite { staticDir =>
      withServer(staticDir, maxUploadBytes = 64) { server =>
        val baseUri = s"http://${server.binding.host}:${server.binding.port}"

        val health = get(s"$baseUri/api/health")
        assertEquals(health.statusCode(), 200)
        assert(health.body().contains("\"ok\": true"))

        val index = get(s"$baseUri/")
        assertEquals(index.statusCode(), 200)
        assert(index.body().contains("Runtime smoke page"))

        val oversizedPayload = s"""{"handHistoryText":"${"A" * 256}"}"""
        val oversizedResponse = postJson(s"$baseUri/api/analyze-hand-history", oversizedPayload)
        assertEquals(oversizedResponse.statusCode(), 413)
        assert(oversizedResponse.body().contains("max upload size"))
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
        val submission = jsonBody(submissionResponse)
        assertEquals(submission("status").str, "queued")
        val statusUri = s"$baseUri${submission("statusUrl").str}"

        assert(backend.started.await(3, TimeUnit.SECONDS), "analysis backend never started")

        val running = getJson(statusUri)
        assertEquals(running("status").str, "running")
        assertEquals(get(s"$baseUri/api/health").statusCode(), 200)
        assertEquals(get(s"$baseUri/").statusCode(), 200)

        backend.release.countDown()

        val completed = awaitTerminalJob(statusUri)
        assertEquals(completed("status").str, "completed")
        assertEquals(completed("result")("site").str, "PokerStars")
        assertEquals(completed("result")("handsImported").num.toInt, 1)
        assertEquals(completed("result")("handsAnalyzed").num.toInt, 1)
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
      backend: HandHistoryReviewServer.AnalysisBackend = immediateBackend(Right(sampleAnalysisResult))
  )(run: HandHistoryReviewServer.RunningServer => A): A =
    val server = HandHistoryReviewServer.startWithBackend(
      HandHistoryReviewServer.ServerConfig(
        host = "127.0.0.1",
        port = 0,
        staticDir = staticDir,
        maxUploadBytes = maxUploadBytes,
        serviceConfig = HandHistoryReviewService.ServiceConfig()
      ),
      backend
    ).fold(err => fail(err), identity)
    try run(server)
    finally server.close()

  private def jsonBody(response: HttpResponse[String]): Value =
    ujson.read(response.body())

  private def getJson(uri: String): Value =
    jsonBody(get(uri))

  private def awaitTerminalJob(uri: String): Value =
    val deadlineNanos = System.nanoTime() + TimeUnit.SECONDS.toNanos(5)
    var lastStatus = "<none>"
    while System.nanoTime() < deadlineNanos do
      val body = getJson(uri)
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

  private def get(uri: String): HttpResponse[String] =
    val request = HttpRequest.newBuilder(URI.create(uri)).GET().build()
    httpClient.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8))

  private def postJson(uri: String, body: String): HttpResponse[String] =
    val request = HttpRequest.newBuilder(URI.create(uri))
      .header("Content-Type", "application/json")
      .POST(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8))
      .build()
    httpClient.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8))

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
