package sicfun.holdem.web

import ujson.Value

import java.nio.file.Path

/** Embedded HTTP server for hand-history review analysis, auth, and static UI hosting.
  *
  * The public entry points and compatibility types stay on this object. Runtime
  * wiring, API handling, and config parsing live in focused package-private modules.
  */
object HandHistoryReviewServer:
  final case class BasicAuthConfig(
      username: String,
      password: String
  )

  /** Server configuration with sensible defaults for local development.
    *
    * @param host                      bind address (default: 127.0.0.1 for local-only access)
    * @param port                      HTTP port (default: 8080)
    * @param staticDir                 directory for serving static UI files
    * @param maxUploadBytes            maximum request body size for hand history uploads
    * @param analysisTimeoutMs         per-analysis timeout before a job is marked failed
    * @param playingHallTimeoutMs      per-playing-hall timeout before a job is marked failed
    * @param maxConcurrentJobs         number of worker threads for analysis processing
    * @param maxQueuedJobs             maximum pending jobs before rejecting submissions
    * @param shutdownGraceMs           grace period for in-flight analysis on shutdown
    * @param rateLimitSubmitsPerMinute  max analysis submissions per client per minute
    * @param rateLimitStatusPerMinute   max status polls per client per minute
    * @param rateLimitAuthPerMinute     max auth (register/login) attempts per client per minute
    * @param rateLimitClientIpHeader    optional header for client IP (e.g. X-Forwarded-For behind proxy)
    */
  final case class ServerConfig(
      host: String,
      port: Int,
      staticDir: Path,
      maxUploadBytes: Int,
      analysisTimeoutMs: Long,
      playingHallTimeoutMs: Long,
      maxConcurrentJobs: Int,
      maxQueuedJobs: Int,
      shutdownGraceMs: Long,
      rateLimitSubmitsPerMinute: Int,
      rateLimitStatusPerMinute: Int,
      rateLimitAuthPerMinute: Int,
      rateLimitClientIpHeader: Option[String],
      rateLimitTrustedProxyIps: Set[String],
      drainSignalFile: Option[Path],
      basicAuth: Option[BasicAuthConfig],
      serviceConfig: HandHistoryReviewService.ServiceConfig,
      platformAuth: Option[PlatformUserAuth.Config] = None
  )


  private[web] type JsonResponse = HandHistoryReviewServerApi.JsonResponse
  private[web] val JsonResponse = HandHistoryReviewServerApi.JsonResponse
  private[web] type PlayingHallRequest = HandHistoryReviewServerApi.PlayingHallRequest
  private[web] val PlayingHallRequest = HandHistoryReviewServerApi.PlayingHallRequest
  private[web] type AnalysisBackend = JobQueue.AnalysisBackend
  private[web] type PlayingHallBackend = JobQueue.PlayingHallBackend

  private val livePlayingHallBackend = new PlayingHallBackend:
    override def run(request: PlayingHallRequest, cancelSignal: () => Boolean): Either[String, Value] =
      HandHistoryReviewServerApi.runPlayingHall(request, cancelSignal)

  def main(args: Array[String]): Unit =
    start(args) match
      case Left(err) =>
        System.err.println(err)
        sys.exit(1)
      case Right(server) =>
        val binding = server.binding
        HandHistoryReviewServerRuntime.logInfo(s"hand-history review web server listening on http://${binding.host}:${binding.port}/")
        HandHistoryReviewServerRuntime.logInfo(s"serving static site from ${binding.staticDir.toAbsolutePath.normalize()}")
        HandHistoryReviewServerRuntime.logInfo(s"model source: ${binding.modelSource}")
        HandHistoryReviewServerRuntime.blockUntilShutdown()

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
      config <- HandHistoryReviewServerConfig.parseArgs(args)
      service <- HandHistoryReviewService.create(config.serviceConfig)
      analysisBackend = new AnalysisBackend:
        override def analyze(
            request: HandHistoryReviewService.AnalysisRequest
        ): Either[String, Value] =
          service.analyze(request).map(service.writeJson)
      running <- startWithBackends(config, analysisBackend, livePlayingHallBackend)
    yield running

  private[web] def startWithBackend(
      config: ServerConfig,
      backend: AnalysisBackend
  ): Either[String, RunningServer] =
    startWithBackends(config, backend, livePlayingHallBackend)

  private[web] def startWithBackends(
      config: ServerConfig,
      backend: AnalysisBackend,
      playingHallBackend: PlayingHallBackend
  ): Either[String, RunningServer] =
    HandHistoryReviewServerRuntime.startServer(config, backend, playingHallBackend)

  def run(args: Array[String]): Either[String, ServerBinding] =
    start(args).map(_.binding)

  private[web] def shutdownDelaySeconds(graceMs: Long): Int =
    HandHistoryReviewServerRuntime.shutdownDelaySeconds(graceMs)
