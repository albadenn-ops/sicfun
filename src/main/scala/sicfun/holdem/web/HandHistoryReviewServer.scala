package sicfun.holdem.web

import sicfun.holdem.cli.CliHelpers
import sicfun.holdem.history.HandHistorySite

import com.sun.net.httpserver.{HttpExchange, HttpHandler, HttpServer}
import ujson.{Obj, Str, Value}

import java.io.ByteArrayOutputStream
import java.net.{BindException, InetSocketAddress}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import java.util.UUID
import java.util.concurrent.{ConcurrentHashMap, ExecutorService, Executors, RejectedExecutionException, ThreadFactory}
import java.util.concurrent.atomic.{AtomicBoolean, AtomicInteger}
import scala.util.control.NonFatal

object HandHistoryReviewServer:
  private val AnalyzeJobPathPrefix = "/api/analyze-hand-history/jobs/"
  private val DefaultPollAfterMs = 750
  private val CompletedJobRetentionMs = 15L * 60L * 1000L

  final case class ServerConfig(
      host: String,
      port: Int,
      staticDir: Path,
      maxUploadBytes: Int,
      serviceConfig: HandHistoryReviewService.ServiceConfig
  )

  private[web] trait AnalysisBackend:
    def analyze(request: HandHistoryReviewService.AnalysisRequest): Either[String, Value]

  def main(args: Array[String]): Unit =
    start(args) match
      case Left(err) =>
        System.err.println(err)
        sys.exit(1)
      case Right(server) =>
        val binding = server.binding
        println(s"Hand-history review web server listening on http://${binding.host}:${binding.port}/")
        println(s"Serving static site from ${binding.staticDir.toAbsolutePath.normalize()}")
        println(s"Model source: ${binding.modelSource}")
        blockUntilShutdown()

  final case class ServerBinding(
      host: String,
      port: Int,
      staticDir: Path,
      modelSource: String
  )

  final class RunningServer(
      val binding: ServerBinding,
      shutdownFn: () => Unit
  ) extends AutoCloseable:
    override def close(): Unit = shutdownFn()

  def start(args: Array[String]): Either[String, RunningServer] =
    for
      config <- parseArgs(args)
      service <- HandHistoryReviewService.create(config.serviceConfig)
      running <- startWithBackend(
        config,
        new AnalysisBackend:
          override def analyze(
              request: HandHistoryReviewService.AnalysisRequest
          ): Either[String, Value] =
            service.analyze(request).map(service.writeJson)
      )
    yield running

  private[web] def startWithBackend(
      config: ServerConfig,
      backend: AnalysisBackend
  ): Either[String, RunningServer] =
    startServer(config, backend)

  def run(args: Array[String]): Either[String, ServerBinding] =
    start(args).map(_.binding)

  private def startServer(
      config: ServerConfig,
      backend: AnalysisBackend
  ): Either[String, RunningServer] =
    val serverExecutor = newServerExecutor()
    val analysisExecutor = newAnalysisExecutor()
    try
      val jobStore = new AnalysisJobStore(analysisExecutor, backend)
      val server = HttpServer.create(new InetSocketAddress(config.host, config.port), 0)
      server.createContext(
        "/api/health",
        new JsonHandler(_ => Right(JsonResponse(200, Obj("ok" -> ujson.Bool(true)))))
      )
      server.createContext(
        AnalyzeJobPathPrefix,
        new JsonHandler(exchange => handleAnalyzeJobStatus(exchange, jobStore))
      )
      server.createContext(
        "/api/analyze-hand-history",
        new JsonHandler(exchange => handleAnalyzeSubmit(exchange, jobStore, config.maxUploadBytes))
      )
      server.createContext("/", new StaticHandler(config.staticDir))
      server.setExecutor(serverExecutor)
      server.start()
      val binding = ServerBinding(
        host = config.host,
        port = server.getAddress.getPort,
        staticDir = config.staticDir,
        modelSource = config.serviceConfig.modelDir.map(_.toAbsolutePath.normalize().toString).getOrElse("uniform fallback")
      )
      val closed = new AtomicBoolean(false)
      def shutdown(): Unit =
        if closed.compareAndSet(false, true) then
          try server.stop(0)
          finally
            serverExecutor.shutdown()
            analysisExecutor.shutdownNow()
      sys.addShutdownHook(shutdown())
      Right(new RunningServer(binding, () => shutdown()))
    catch
      case e: BindException =>
        serverExecutor.shutdownNow()
        analysisExecutor.shutdownNow()
        Left(s"failed to start web server: ${config.host}:${config.port} is unavailable (${e.getMessage})")
      case e: Exception =>
        serverExecutor.shutdownNow()
        analysisExecutor.shutdownNow()
        Left(s"failed to start web server: ${e.getMessage}")

  private def newServerExecutor(): ExecutorService =
    val workerCount = math.max(4, Runtime.getRuntime.availableProcessors())
    Executors.newFixedThreadPool(workerCount, newThreadFactory("hand-review-http"))

  private def newAnalysisExecutor(): ExecutorService =
    val workerCount = math.max(1, math.min(4, Runtime.getRuntime.availableProcessors() - 1))
    Executors.newFixedThreadPool(workerCount, newThreadFactory("hand-review-analysis"))

  private def newThreadFactory(prefix: String): ThreadFactory =
    val nextId = new AtomicInteger(1)
    new ThreadFactory:
      override def newThread(runnable: Runnable): Thread =
        val thread = new Thread(runnable)
        thread.setName(s"$prefix-${nextId.getAndIncrement()}")
        thread.setDaemon(true)
        thread

  private def handleAnalyzeSubmit(
      exchange: HttpExchange,
      jobStore: AnalysisJobStore,
      maxUploadBytes: Int
  ): Either[(Int, String), JsonResponse] =
    if !exchange.getRequestMethod.equalsIgnoreCase("POST") then Left(405 -> "POST required")
    else
      readRequestBody(exchange, maxUploadBytes)
        .flatMap(parseRequest)
        .flatMap(request => jobStore.submit(request).left.map(error => 503 -> error))
        .map { accepted =>
          JsonResponse(
            status = 202,
            value = Obj(
              "jobId" -> Str(accepted.jobId),
              "status" -> Str("queued"),
              "statusUrl" -> Str(accepted.statusUrl),
              "submittedAtEpochMs" -> ujson.Num(accepted.submittedAtEpochMs.toDouble),
              "pollAfterMs" -> ujson.Num(accepted.pollAfterMs)
            ),
            headers = Vector(
              "Location" -> accepted.statusUrl,
              "Retry-After" -> retryAfterSeconds(accepted.pollAfterMs)
            )
          )
        }

  private def handleAnalyzeJobStatus(
      exchange: HttpExchange,
      jobStore: AnalysisJobStore
  ): Either[(Int, String), JsonResponse] =
    if !exchange.getRequestMethod.equalsIgnoreCase("GET") then Left(405 -> "GET required")
    else
      extractJobId(exchange).flatMap { jobId =>
        jobStore.status(jobId).toRight(404 -> s"analysis job not found: $jobId")
      }

  private def extractJobId(exchange: HttpExchange): Either[(Int, String), String] =
    val path = Option(exchange.getRequestURI.getPath).getOrElse("")
    if !path.startsWith(AnalyzeJobPathPrefix) then Left(404 -> "not found")
    else
      val jobId = path.substring(AnalyzeJobPathPrefix.length).trim
      Either.cond(jobId.nonEmpty && !jobId.contains("/"), jobId, 400 -> "analysis job id is required")

  private def parseRequest(body: String): Either[(Int, String), HandHistoryReviewService.AnalysisRequest] =
    try
      val json = ujson.read(body)
      val obj = json.obj
      val handHistoryText = requiredString(obj, "handHistoryText").map(_.trim)
      val heroName = optionalString(obj, "heroName").map(_.trim).filter(_.nonEmpty)
      val site = parseOptionalSite(optionalString(obj, "site"))
      for
        text <- handHistoryText
        parsedSite <- site
      yield HandHistoryReviewService.AnalysisRequest(
        handHistoryText = text,
        site = parsedSite,
        heroName = heroName
      )
    catch
      case NonFatal(e) => Left(400 -> s"invalid JSON request: ${e.getMessage}")

  private def requiredString(
      obj: collection.Map[String, Value],
      key: String
  ): Either[(Int, String), String] =
    optionalString(obj, key).filter(_.nonEmpty).toRight(400 -> s"$key is required")

  private def optionalString(
      obj: collection.Map[String, Value],
      key: String
  ): Option[String] =
    obj.get(key).flatMap {
      case Str(value) => Some(value)
      case ujson.Null => None
      case other => Some(other.str)
    }

  private def parseOptionalSite(raw: Option[String]): Either[(Int, String), Option[HandHistorySite]] =
    raw.map(_.trim).filter(_.nonEmpty).filterNot(_.equalsIgnoreCase("auto")) match
      case None => Right(None)
      case Some(value) => HandHistorySite.parse(value).left.map(err => 400 -> err).map(Some(_))

  private def readRequestBody(
      exchange: HttpExchange,
      maxUploadBytes: Int
  ): Either[(Int, String), String] =
    Option(exchange.getRequestHeaders.getFirst("Content-Length"))
      .flatMap(_.toLongOption)
      .filter(_ > maxUploadBytes.toLong) match
        case Some(_) =>
          Left(413 -> s"request body exceeds max upload size of $maxUploadBytes bytes")
        case None =>
          val input = exchange.getRequestBody
          val buffer = Array.ofDim[Byte](8192)
          val output = new ByteArrayOutputStream(math.min(maxUploadBytes, 8192))
          var total = 0
          var bytesRead = input.read(buffer)
          while bytesRead != -1 && total <= maxUploadBytes do
            total += bytesRead
            if total <= maxUploadBytes then
              output.write(buffer, 0, bytesRead)
            bytesRead = input.read(buffer)
          if total > maxUploadBytes then
            Left(413 -> s"request body exceeds max upload size of $maxUploadBytes bytes")
          else Right(new String(output.toByteArray, StandardCharsets.UTF_8))

  private def parseArgs(args: Array[String]): Either[String, ServerConfig] =
    if args.contains("--help") || args.contains("-h") then Left(usage)
    else
      for
        options <- CliHelpers.parseOptionsAllowBlankValues(args)
        host = options.get("host").orElse(env("HOST")).getOrElse("127.0.0.1")
        port <- resolveIntOption(options, "port", env("PORT"), 8080)
        staticDir <- parseDirectory(
          options.get("staticDir").orElse(env("STATIC_DIR")).getOrElse("docs/site-preview-hybrid"),
          "staticDir"
        )
        maxUploadBytes <- resolveIntOption(options, "maxUploadBytes", env("MAX_UPLOAD_BYTES"), 2 * 1024 * 1024)
        _ <- Either.cond(maxUploadBytes > 0, (), "--maxUploadBytes must be positive")
        modelDir <- parseOptionalDirectory(options.get("model").orElse(env("MODEL_DIR")), "model")
        seed <- resolveLongOption(options, "seed", env("SEED"), 42L)
        bunchingTrials <- resolveIntOption(options, "bunchingTrials", env("BUNCHING_TRIALS"), 200)
        equityTrials <- resolveIntOption(options, "equityTrials", env("EQUITY_TRIALS"), 2000)
        budgetMs <- resolveLongOption(options, "budgetMs", env("BUDGET_MS"), 1500L)
        maxDecisions <- resolveIntOption(options, "maxDecisions", env("MAX_DECISIONS"), 12)
      yield ServerConfig(
        host = host,
        port = port,
        staticDir = staticDir,
        maxUploadBytes = maxUploadBytes,
        serviceConfig = HandHistoryReviewService.ServiceConfig(
          modelDir = modelDir,
          seed = seed,
          bunchingTrials = bunchingTrials,
          equityTrials = equityTrials,
          budgetMs = budgetMs,
          maxDecisions = maxDecisions
        )
      )

  private def parseDirectory(raw: String, label: String): Either[String, Path] =
    val path = Paths.get(raw).toAbsolutePath.normalize()
    if Files.isDirectory(path) then Right(path)
    else Left(s"--$label directory not found: $raw")

  private def parseOptionalDirectory(raw: Option[String], label: String): Either[String, Option[Path]] =
    raw.map(_.trim).filter(_.nonEmpty) match
      case None => Right(None)
      case Some(value) => parseDirectory(value, label).map(Some(_))

  private def resolveIntOption(
      options: Map[String, String],
      key: String,
      envValue: Option[String],
      default: Int
  ): Either[String, Int] =
    options.get(key)
      .orElse(envValue)
      .map(_.trim)
      .filter(_.nonEmpty) match
        case None => Right(default)
        case Some(raw) => raw.toIntOption.toRight(s"--$key must be an integer")

  private def resolveLongOption(
      options: Map[String, String],
      key: String,
      envValue: Option[String],
      default: Long
  ): Either[String, Long] =
    options.get(key)
      .orElse(envValue)
      .map(_.trim)
      .filter(_.nonEmpty) match
        case None => Right(default)
        case Some(raw) => raw.toLongOption.toRight(s"--$key must be a long")

  private def env(name: String): Option[String] =
    Option(System.getenv(name)).map(_.trim).filter(_.nonEmpty)

  private final case class JsonResponse(
      status: Int,
      value: Value,
      headers: Vector[(String, String)] = Vector.empty
  )

  private final case class AcceptedJob(
      jobId: String,
      submittedAtEpochMs: Long,
      statusUrl: String,
      pollAfterMs: Int
  )

  private sealed trait AnalysisJobState:
    def status: String
    def submittedAtEpochMs: Long
    def startedAtEpochMs: Option[Long]
    def completedAtEpochMs: Option[Long]
    def isTerminal: Boolean

  private object AnalysisJobState:
    final case class Queued(submittedAtEpochMs: Long) extends AnalysisJobState:
      override val status = "queued"
      override val startedAtEpochMs = None
      override val completedAtEpochMs = None
      override val isTerminal = false

    final case class Running(submittedAtEpochMs: Long, startedAt: Long) extends AnalysisJobState:
      override val status = "running"
      override val startedAtEpochMs = Some(startedAt)
      override val completedAtEpochMs = None
      override val isTerminal = false

    final case class Completed(
        submittedAtEpochMs: Long,
        startedAt: Long,
        completedAt: Long,
        result: Value
    ) extends AnalysisJobState:
      override val status = "completed"
      override val startedAtEpochMs = Some(startedAt)
      override val completedAtEpochMs = Some(completedAt)
      override val isTerminal = true

    final case class Failed(
        submittedAtEpochMs: Long,
        startedAt: Long,
        completedAt: Long,
        errorStatus: Int,
        error: String
    ) extends AnalysisJobState:
      override val status = "failed"
      override val startedAtEpochMs = Some(startedAt)
      override val completedAtEpochMs = Some(completedAt)
      override val isTerminal = true

  private final class AnalysisJobStore(
      executor: ExecutorService,
      backend: AnalysisBackend,
      nowMillis: () => Long = () => System.currentTimeMillis()
  ):
    import AnalysisJobState.*

    private val jobs = new ConcurrentHashMap[String, AnalysisJobState]()

    def submit(
        request: HandHistoryReviewService.AnalysisRequest
    ): Either[String, AcceptedJob] =
      purgeExpiredJobs()
      val jobId = UUID.randomUUID().toString
      val submittedAt = nowMillis()
      jobs.put(jobId, Queued(submittedAt))
      try
        executor.submit(new Runnable:
          override def run(): Unit =
            runJob(jobId, request, submittedAt)
        )
        Right(
          AcceptedJob(
            jobId = jobId,
            submittedAtEpochMs = submittedAt,
            statusUrl = s"$AnalyzeJobPathPrefix$jobId",
            pollAfterMs = DefaultPollAfterMs
          )
        )
      catch
        case _: RejectedExecutionException =>
          jobs.remove(jobId)
          Left("analysis queue unavailable")

    def status(jobId: String): Option[JsonResponse] =
      purgeExpiredJobs()
      Option(jobs.get(jobId)).map(renderStatus(jobId, _))

    private def runJob(
        jobId: String,
        request: HandHistoryReviewService.AnalysisRequest,
        submittedAt: Long
    ): Unit =
      val startedAt = nowMillis()
      jobs.put(jobId, Running(submittedAt, startedAt))
      val completedState =
        try
          backend.analyze(request) match
            case Right(result) =>
              Completed(submittedAt, startedAt, nowMillis(), result)
            case Left(error) =>
              Failed(submittedAt, startedAt, nowMillis(), classifyAnalysisError(error), error)
        catch
          case NonFatal(e) =>
            Failed(submittedAt, startedAt, nowMillis(), 500, s"analysis failed: ${e.getMessage}")
      jobs.put(jobId, completedState)

    private def renderStatus(jobId: String, state: AnalysisJobState): JsonResponse =
      state match
        case Queued(submittedAt) =>
          val json = baseStatus(jobId, state, submittedAt, None, None, Some(DefaultPollAfterMs))
          json("message") = Str("Queued for analysis")
          JsonResponse(
            status = 200,
            value = json,
            headers = Vector("Retry-After" -> retryAfterSeconds(DefaultPollAfterMs))
          )
        case Running(submittedAt, startedAt) =>
          val json = baseStatus(jobId, state, submittedAt, Some(startedAt), None, Some(DefaultPollAfterMs))
          json("message") = Str("Analysis in progress")
          JsonResponse(
            status = 200,
            value = json,
            headers = Vector("Retry-After" -> retryAfterSeconds(DefaultPollAfterMs))
          )
        case Completed(submittedAt, startedAt, completedAt, result) =>
          val json = baseStatus(jobId, state, submittedAt, Some(startedAt), Some(completedAt), None)
          json("durationMs") = ujson.Num((completedAt - startedAt).toDouble)
          json("result") = result
          JsonResponse(200, json)
        case Failed(submittedAt, startedAt, completedAt, errorStatus, error) =>
          val json = baseStatus(jobId, state, submittedAt, Some(startedAt), Some(completedAt), None)
          json("durationMs") = ujson.Num((completedAt - startedAt).toDouble)
          json("errorStatus") = ujson.Num(errorStatus)
          json("error") = Str(error)
          JsonResponse(200, json)

    private def baseStatus(
        jobId: String,
        state: AnalysisJobState,
        submittedAt: Long,
        startedAt: Option[Long],
        completedAt: Option[Long],
        pollAfterMs: Option[Int]
    ): Obj =
      val json = Obj(
        "jobId" -> Str(jobId),
        "status" -> Str(state.status),
        "statusUrl" -> Str(s"$AnalyzeJobPathPrefix$jobId"),
        "submittedAtEpochMs" -> ujson.Num(submittedAt.toDouble),
        "startedAtEpochMs" -> startedAt.map(value => ujson.Num(value.toDouble)).getOrElse(ujson.Null),
        "completedAtEpochMs" -> completedAt.map(value => ujson.Num(value.toDouble)).getOrElse(ujson.Null)
      )
      pollAfterMs.foreach(value => json("pollAfterMs") = ujson.Num(value))
      json

    private def purgeExpiredJobs(): Unit =
      val cutoff = nowMillis() - CompletedJobRetentionMs
      val iterator = jobs.entrySet().iterator()
      while iterator.hasNext do
        val entry = iterator.next()
        val state = entry.getValue
        if state.isTerminal && state.completedAtEpochMs.exists(_ < cutoff) then
          iterator.remove()

  private final class JsonHandler(
      handle: HttpExchange => Either[(Int, String), JsonResponse]
  ) extends HttpHandler:
    override def handle(exchange: HttpExchange): Unit =
      try
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

  private final class StaticHandler(staticDir: Path) extends HttpHandler:
    override def handle(exchange: HttpExchange): Unit =
      try
        if !exchange.getRequestMethod.equalsIgnoreCase("GET") then
          writePlain(exchange, 405, "GET required", "text/plain; charset=utf-8")
        else
          val requestPath = Option(exchange.getRequestURI.getPath).getOrElse("/")
          val relative = if requestPath == "/" then Paths.get("index.html") else Paths.get(requestPath.dropWhile(_ == '/'))
          val resolved = staticDir.resolve(relative).normalize()
          if !resolved.startsWith(staticDir) then
            writePlain(exchange, 403, "forbidden", "text/plain; charset=utf-8")
          else
            val target =
              if Files.isDirectory(resolved) then resolved.resolve("index.html")
              else resolved
            if Files.exists(target) && Files.isRegularFile(target) then
              exchange.getResponseHeaders.add("Content-Type", contentTypeFor(target))
              exchange.sendResponseHeaders(200, Files.size(target))
              val body = exchange.getResponseBody
              val input = Files.newInputStream(target)
              try
                input.transferTo(body)
                body.flush()
              finally input.close()
            else
              writePlain(exchange, 404, "not found", "text/plain; charset=utf-8")
      catch
        case NonFatal(e) =>
          writePlain(exchange, 500, s"internal server error: ${e.getMessage}", "text/plain; charset=utf-8")
      finally
        exchange.close()

  private def writeJson(exchange: HttpExchange, status: Int, value: Value): Unit =
    val bytes = ujson.write(value, indent = 2).getBytes(StandardCharsets.UTF_8)
    writeBytes(exchange, status, bytes, "application/json; charset=utf-8")

  private def writePlain(exchange: HttpExchange, status: Int, body: String, contentType: String): Unit =
    writeBytes(exchange, status, body.getBytes(StandardCharsets.UTF_8), contentType)

  private def writeBytes(
      exchange: HttpExchange,
      status: Int,
      bytes: Array[Byte],
      contentType: String
  ): Unit =
    exchange.getResponseHeaders.add("Content-Type", contentType)
    exchange.getResponseHeaders.add("Cache-Control", "no-store")
    exchange.sendResponseHeaders(status, bytes.length.toLong)
    val body = exchange.getResponseBody
    body.write(bytes)
    body.flush()

  private def contentTypeFor(path: Path): String =
    path.getFileName.toString.toLowerCase match
      case name if name.endsWith(".html") => "text/html; charset=utf-8"
      case name if name.endsWith(".css") => "text/css; charset=utf-8"
      case name if name.endsWith(".js") => "application/javascript; charset=utf-8"
      case name if name.endsWith(".mjs") => "application/javascript; charset=utf-8"
      case name if name.endsWith(".map") => "application/json; charset=utf-8"
      case name if name.endsWith(".svg") => "image/svg+xml"
      case name if name.endsWith(".png") => "image/png"
      case name if name.endsWith(".jpg") || name.endsWith(".jpeg") => "image/jpeg"
      case name if name.endsWith(".webp") => "image/webp"
      case name if name.endsWith(".ico") => "image/x-icon"
      case name if name.endsWith(".wasm") => "application/wasm"
      case name if name.endsWith(".woff2") => "font/woff2"
      case name if name.endsWith(".json") => "application/json; charset=utf-8"
      case name if name.endsWith(".txt") => "text/plain; charset=utf-8"
      case _ => "application/octet-stream"

  private def classifyAnalysisError(error: String): Int =
    if error.startsWith("analysis failed:") then 500 else 400

  private def retryAfterSeconds(pollAfterMs: Int): String =
    math.max(1, (pollAfterMs + 999) / 1000).toString

  private def blockUntilShutdown(): Unit =
    val latch = new java.util.concurrent.CountDownLatch(1)
    sys.addShutdownHook(latch.countDown())
    latch.await()

  private val usage =
    """Usage:
      |  runMain sicfun.holdem.web.HandHistoryReviewServer [--key=value ...]
      |
      |Options:
      |  --host=127.0.0.1         Bind host (falls back to HOST env)
      |  --port=8080              Bind port (falls back to PORT env)
      |  --staticDir=docs/site-preview-hybrid
      |  --maxUploadBytes=2097152 Max raw upload size in bytes (falls back to MAX_UPLOAD_BYTES env)
      |  --model=<dir>            Optional model artifact directory (falls back to MODEL_DIR env)
      |  --seed=42                RNG seed (falls back to SEED env)
      |  --bunchingTrials=200     Monte Carlo bunching trials per analysis
      |  --equityTrials=2000      Monte Carlo equity trials per analysis
      |  --budgetMs=1500          Decision budget per analyzed action
      |  --maxDecisions=12        Max decisions returned to the page
      |""".stripMargin
