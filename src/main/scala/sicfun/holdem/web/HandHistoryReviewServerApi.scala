package sicfun.holdem.web

import sicfun.holdem.history.HandHistorySite
import sicfun.holdem.runtime.TexasHoldemPlayingHall

import com.sun.net.httpserver.HttpExchange
import ujson.{Arr, Obj, Str, Value}

import java.io.ByteArrayOutputStream
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import java.util.{Locale, UUID}
import scala.util.control.NonFatal

import sicfun.holdem.web.AuthStack.{SessionCsrfRequiredMessage, authenticatedUser, ensurePlatformCsrf}
import sicfun.holdem.web.JobQueue.{
  AcceptedJob,
  AnalysisJobStore,
  AnalyzeJobPathPrefix,
  CancelOutcome,
  PlayingHallJobPathPrefix,
  PlayingHallJobStore
}
import sicfun.holdem.web.Readiness.{ReadinessStatus, admissionRejectedMessage}

/** API submit/status handlers and request parsing for [[HandHistoryReviewServer]]. */
private[web] object HandHistoryReviewServerApi:
  private val DefaultPlayingHallRoot = Paths.get("data", "web-playing-hall")
  private val DefaultPlayingHallHands = 240
  private val DefaultPlayingHallTableCount = 2
  private val DefaultPlayingHallPlayerCount = 6
  private val DefaultPlayingHallSeed = 42L
  private val MaxPlayingHallHands = 5000
  private val MaxPlayingHallTableCount = 24
  private val MaxPlayingHallBunchingTrials = 600
  private val MaxPlayingHallEquityTrials = 6000
  private val MaxPlayingHallVillainPoolEntries = 8

  def handleAnalyzeSubmit(
      exchange: HttpExchange,
      jobStore: AnalysisJobStore,
      maxUploadBytes: Int,
      readiness: () => ReadinessStatus,
      platformAuth: Option[PlatformUserAuth.Service]
  ): Either[(Int, String), JsonResponse] =
    if !exchange.getRequestMethod.equalsIgnoreCase("POST") then Left(405 -> "POST required")
    else if !ensurePlatformCsrf(exchange, platformAuth) then Left(403 -> SessionCsrfRequiredMessage)
    else if !readiness().acceptingAnalysisJobs then
      Left(503 -> admissionRejectedMessage(readiness()))
    else
      readRequestBody(exchange, maxUploadBytes)
        .flatMap(parseRequest)
        .flatMap(request =>
          jobStore
            .submit(
              request,
              ownerUserId = authenticatedUser(exchange).map(_.userId),
              rejectIfUnavailable = () =>
                if readiness().acceptingAnalysisJobs then None
                else Some(admissionRejectedMessage(readiness()))
            )
            .left
            .map(error => 503 -> error)
        )
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

  def handlePlayingHallSubmit(
      exchange: HttpExchange,
      jobStore: PlayingHallJobStore,
      maxUploadBytes: Int,
      readiness: () => ReadinessStatus,
      platformAuth: Option[PlatformUserAuth.Service]
  ): Either[(Int, String), JsonResponse] =
    if !exchange.getRequestMethod.equalsIgnoreCase("POST") then Left(405 -> "POST required")
    else if !ensurePlatformCsrf(exchange, platformAuth) then Left(403 -> SessionCsrfRequiredMessage)
    else if !readiness().acceptingAnalysisJobs then
      Left(503 -> admissionRejectedMessage(readiness(), "playing hall"))
    else
      readRequestBody(exchange, maxUploadBytes)
        .flatMap(parsePlayingHallRequest)
        .flatMap(request =>
          jobStore
            .submit(
              request,
              ownerUserId = authenticatedUser(exchange).map(_.userId),
              rejectIfUnavailable = () =>
                if readiness().acceptingAnalysisJobs then None
                else Some(admissionRejectedMessage(readiness(), "playing hall"))
            )
            .left
            .map(error => 503 -> error)
        )
        .map(renderAcceptedJobResponse)

  private def renderAcceptedJobResponse(accepted: AcceptedJob): JsonResponse =
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

  def handleAnalyzeJobStatus(
      exchange: HttpExchange,
      jobStore: AnalysisJobStore,
      platformAuth: Option[PlatformUserAuth.Service]
  ): Either[(Int, String), JsonResponse] =
    if !exchange.getRequestMethod.equalsIgnoreCase("GET") then Left(405 -> "GET required")
    else
      extractJobId(exchange, AnalyzeJobPathPrefix, "analysis").flatMap { jobId =>
        jobStore
          .status(
            jobId = jobId,
            requesterUserId = authenticatedUser(exchange).map(_.userId),
            enforceOwnership = platformAuth.nonEmpty
          )
          .toRight(404 -> s"analysis job not found: $jobId")
      }

  def handlePlayingHallJobStatus(
      exchange: HttpExchange,
      jobStore: PlayingHallJobStore,
      platformAuth: Option[PlatformUserAuth.Service]
  ): Either[(Int, String), JsonResponse] =
    val method = exchange.getRequestMethod
    if method.equalsIgnoreCase("GET") then
      extractJobId(exchange, PlayingHallJobPathPrefix, "playing hall").flatMap { jobId =>
        jobStore
          .status(
            jobId = jobId,
            requesterUserId = authenticatedUser(exchange).map(_.userId),
            enforceOwnership = platformAuth.nonEmpty
          )
          .toRight(404 -> s"playing hall job not found: $jobId")
      }
    else if method.equalsIgnoreCase("DELETE") then
      extractJobId(exchange, PlayingHallJobPathPrefix, "playing hall").flatMap { jobId =>
        jobStore.cancel(
          jobId = jobId,
          requesterUserId = authenticatedUser(exchange).map(_.userId),
          enforceOwnership = platformAuth.nonEmpty
        ) match
          case CancelOutcome.Accepted =>
            Right(JsonResponse(200, Obj("jobId" -> Str(jobId), "status" -> Str("cancelled"))))
          case CancelOutcome.AlreadyTerminal =>
            Left(409 -> "playing hall job already terminal")
          case CancelOutcome.NotFound =>
            Left(404 -> s"playing hall job not found: $jobId")
      }
    else Left(405 -> "GET or DELETE required")

  private def extractJobId(
      exchange: HttpExchange,
      pathPrefix: String,
      label: String
  ): Either[(Int, String), String] =
    val path = Option(exchange.getRequestURI.getPath).getOrElse("")
    if !path.startsWith(pathPrefix) then Left(404 -> "not found")
    else
      val jobId = path.substring(pathPrefix.length).trim
      Either.cond(jobId.nonEmpty && !jobId.contains("/"), jobId, 400 -> s"$label job id is required")

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

  private def parsePlayingHallRequest(body: String): Either[(Int, String), PlayingHallRequest] =
    try
      val obj = ujson.read(body).obj
      for
        hands <- requiredIntInRange(
          obj,
          key = "hands",
          default = DefaultPlayingHallHands,
          min = 1,
          max = MaxPlayingHallHands
        )
        tableCount <- requiredIntInRange(
          obj,
          key = "tableCount",
          default = DefaultPlayingHallTableCount,
          min = 1,
          max = MaxPlayingHallTableCount
        )
        playerCount <- requiredIntInRange(
          obj,
          key = "playerCount",
          default = DefaultPlayingHallPlayerCount,
          min = 2,
          max = 9
        )
        heroStyle <- requiredChoice(
          obj,
          key = "heroStyle",
          default = "adaptive",
          allowed = Set("adaptive", "gto", "strategic")
        )
        heroPosition <- requiredChoice(
          obj,
          key = "heroPosition",
          default = "Button",
          allowed = Set("SmallBlind", "BigBlind", "UTG", "UTG1", "UTG2", "Middle", "Hijack", "Cutoff", "Button")
        )
        gtoMode <- requiredChoice(
          obj,
          key = "gtoMode",
          default = "exact",
          allowed = Set("fast", "exact")
        )
        villainPool <- requiredVillainPool(obj)
        heroExplorationRate <- requiredDoubleInRange(
          obj,
          key = "heroExplorationRate",
          default = 0.0,
          min = 0.0,
          max = 1.0
        )
        raiseSize <- requiredDoubleInRange(
          obj,
          key = "raiseSize",
          default = 2.5,
          min = 0.25,
          max = 20.0
        )
        bunchingTrials <- requiredIntInRange(
          obj,
          key = "bunchingTrials",
          default = 40,
          min = 1,
          max = MaxPlayingHallBunchingTrials
        )
        equityTrials <- requiredIntInRange(
          obj,
          key = "equityTrials",
          default = 240,
          min = 1,
          max = MaxPlayingHallEquityTrials
        )
        learnEveryHands <- requiredIntInRange(
          obj,
          key = "learnEveryHands",
          default = 0,
          min = 0,
          max = MaxPlayingHallHands
        )
        learningWindowSamples <- requiredIntInRange(
          obj,
          key = "learningWindowSamples",
          default = 200,
          min = 0,
          max = 500000
        )
        seed <- optionalLong(obj, "seed").getOrElse(Right(DefaultPlayingHallSeed))
        saveReviewHandHistory <- optionalBoolean(obj, "saveReviewHandHistory").getOrElse(Right(false))
        fullRing <- optionalBoolean(obj, "fullRing").getOrElse(Right(false))
      yield PlayingHallRequest(
        hands = hands,
        tableCount = tableCount,
        playerCount = playerCount,
        heroStyle = heroStyle,
        heroPosition = heroPosition,
        gtoMode = gtoMode,
        villainPool = villainPool,
        heroExplorationRate = heroExplorationRate,
        raiseSize = raiseSize,
        bunchingTrials = bunchingTrials,
        equityTrials = equityTrials,
        learnEveryHands = learnEveryHands,
        learningWindowSamples = learningWindowSamples,
        saveReviewHandHistory = saveReviewHandHistory,
        fullRing = fullRing,
        seed = seed
      )
    catch
      case NonFatal(e) => Left(400 -> s"invalid JSON request: ${e.getMessage}")

  private def requiredIntInRange(
      obj: collection.Map[String, Value],
      key: String,
      default: Int,
      min: Int,
      max: Int
  ): Either[(Int, String), Int] =
    val value = optionalInt(obj, key).getOrElse(Right(default))
    value.flatMap(parsed =>
      Either.cond(parsed >= min && parsed <= max, parsed, 400 -> s"$key must be in [$min,$max]")
    )

  private def requiredDoubleInRange(
      obj: collection.Map[String, Value],
      key: String,
      default: Double,
      min: Double,
      max: Double
  ): Either[(Int, String), Double] =
    val value = optionalDouble(obj, key).getOrElse(Right(default))
    value.flatMap(parsed =>
      Either.cond(parsed >= min && parsed <= max, parsed, 400 -> s"$key must be in [$min,$max]")
    )

  private def requiredChoice(
      obj: collection.Map[String, Value],
      key: String,
      default: String,
      allowed: Set[String]
  ): Either[(Int, String), String] =
    val raw = optionalString(obj, key).map(_.trim).filter(_.nonEmpty).getOrElse(default)
    val normalized = raw.toLowerCase(Locale.ROOT)
    val canonicalByNormalized =
      allowed.map(value => value.toLowerCase(Locale.ROOT) -> value).toMap
    canonicalByNormalized
      .get(normalized)
      .toRight(400 -> s"$key must be one of: ${allowed.toVector.sorted.mkString(", ")}")

  private def requiredVillainPool(
      obj: collection.Map[String, Value]
  ): Either[(Int, String), Vector[String]] =
    val parsed =
      optionalStringArray(obj, "villainPool")
        .orElse(optionalString(obj, "villainPool").map(raw =>
          raw.split(',').toVector.map(_.trim).filter(_.nonEmpty)
        ))
        .getOrElse(Vector("tag", "gto"))
        .map(_.trim)
        .filter(_.nonEmpty)
    if parsed.isEmpty then Left(400 -> "villainPool must include at least one entry")
    else if parsed.length > MaxPlayingHallVillainPoolEntries then
      Left(400 -> s"villainPool must include at most $MaxPlayingHallVillainPoolEntries entries")
    else
      val normalized = parsed.map(_.toLowerCase(Locale.ROOT))
      val allowed = Set("nit", "tag", "lag", "callingstation", "station", "maniac", "gto")
      val invalid = normalized.filterNot(allowed.contains)
      if invalid.nonEmpty then
        Left(400 -> s"villainPool contains unsupported entries: ${invalid.distinct.sorted.mkString(", ")}")
      else Right(normalized)

  private def optionalInt(
      obj: collection.Map[String, Value],
      key: String
  ): Option[Either[(Int, String), Int]] =
    obj.get(key).map {
      case ujson.Num(value) if value.isWhole => Right(value.toInt)
      case Str(value) => value.trim.toIntOption.toRight(400 -> s"$key must be an integer")
      case other => Left(400 -> s"$key must be an integer, got ${ujson.write(other)}")
    }

  private def optionalLong(
      obj: collection.Map[String, Value],
      key: String
  ): Option[Either[(Int, String), Long]] =
    obj.get(key).map {
      case ujson.Num(value) if value.isWhole => Right(value.toLong)
      case Str(value) => value.trim.toLongOption.toRight(400 -> s"$key must be a long")
      case other => Left(400 -> s"$key must be a long, got ${ujson.write(other)}")
    }

  private def optionalDouble(
      obj: collection.Map[String, Value],
      key: String
  ): Option[Either[(Int, String), Double]] =
    obj.get(key).map {
      case ujson.Num(value) => Right(value)
      case Str(value) => value.trim.toDoubleOption.toRight(400 -> s"$key must be a number")
      case other => Left(400 -> s"$key must be a number, got ${ujson.write(other)}")
    }

  private def optionalBoolean(
      obj: collection.Map[String, Value],
      key: String
  ): Option[Either[(Int, String), Boolean]] =
    obj.get(key).map {
      case ujson.Bool(value) => Right(value)
      case Str(value) =>
        value.trim.toLowerCase(Locale.ROOT) match
          case "true" => Right(true)
          case "false" => Right(false)
          case _ => Left(400 -> s"$key must be true or false")
      case other => Left(400 -> s"$key must be true or false, got ${ujson.write(other)}")
    }

  private def optionalStringArray(
      obj: collection.Map[String, Value],
      key: String
  ): Option[Vector[String]] =
    obj.get(key).flatMap {
      case Arr(values) =>
        Some(
          values.collect {
            case Str(value) => value
            case other => other.str
          }.toVector
        )
      case _ => None
    }

  def runPlayingHall(
      request: PlayingHallRequest,
      cancelSignal: () => Boolean
  ): Either[String, Value] =
    Files.createDirectories(DefaultPlayingHallRoot)
    val runRoot = DefaultPlayingHallRoot.resolve(request.runDirectoryName).toAbsolutePath.normalize()
    val reportEvery = math.max(1, math.min(request.hands, math.max(25, request.hands / 3)))
    val args = Array(
      s"--hands=${request.hands}",
      s"--tableCount=${request.tableCount}",
      s"--playerCount=${request.playerCount}",
      s"--reportEvery=$reportEvery",
      s"--learnEveryHands=${request.learnEveryHands}",
      s"--learningWindowSamples=${request.learningWindowSamples}",
      s"--seed=${request.seed}",
      s"--outDir=$runRoot",
      s"--heroStyle=${request.heroStyle}",
      s"--heroPosition=${request.heroPosition}",
      s"--gtoMode=${request.gtoMode}",
      s"--villainPool=${request.villainPool.mkString(",")}",
      s"--heroExplorationRate=${request.heroExplorationRate}",
      s"--raiseSize=${request.raiseSize}",
      s"--bunchingTrials=${request.bunchingTrials}",
      s"--equityTrials=${request.equityTrials}",
      "--saveTrainingTsv=false",
      "--saveDdreTrainingTsv=false",
      s"--saveReviewHandHistory=${request.saveReviewHandHistory}",
      s"--fullRing=${request.fullRing}"
    )
    TexasHoldemPlayingHall
      .runWithCancel(args, cancelSignal)
      .left
      .map(error => s"playing hall failed: $error")
      .map(summary => renderPlayingHallResult(request, summary))

  private def renderPlayingHallResult(
      request: PlayingHallRequest,
      summary: TexasHoldemPlayingHall.HallSummary
  ): Value =
    val outDir = summary.outDir.toAbsolutePath.normalize()
    Obj(
      "request" -> Obj(
        "hands" -> ujson.Num(request.hands.toDouble),
        "tableCount" -> ujson.Num(request.tableCount.toDouble),
        "playerCount" -> ujson.Num(request.playerCount.toDouble),
        "heroStyle" -> Str(request.heroStyle),
        "heroPosition" -> Str(request.heroPosition),
        "gtoMode" -> Str(request.gtoMode),
        "villainPool" -> Arr.from(request.villainPool.map(Str(_))),
        "heroExplorationRate" -> ujson.Num(request.heroExplorationRate),
        "raiseSize" -> ujson.Num(request.raiseSize),
        "bunchingTrials" -> ujson.Num(request.bunchingTrials.toDouble),
        "equityTrials" -> ujson.Num(request.equityTrials.toDouble),
        "learnEveryHands" -> ujson.Num(request.learnEveryHands.toDouble),
        "learningWindowSamples" -> ujson.Num(request.learningWindowSamples.toDouble),
        "saveReviewHandHistory" -> ujson.Bool(request.saveReviewHandHistory),
        "fullRing" -> ujson.Bool(request.fullRing),
        "seed" -> ujson.Num(request.seed.toDouble)
      ),
      "summary" -> Obj(
        "handsPlayed" -> ujson.Num(summary.handsPlayed.toDouble),
        "tableCount" -> ujson.Num(summary.tableCount.toDouble),
        "playerCount" -> ujson.Num(summary.playerCount.toDouble),
        "heroNetChips" -> ujson.Num(summary.heroNetChips),
        "heroBbPer100" -> ujson.Num(summary.heroBbPer100),
        "heroWins" -> ujson.Num(summary.heroWins.toDouble),
        "heroTies" -> ujson.Num(summary.heroTies.toDouble),
        "heroLosses" -> ujson.Num(summary.heroLosses.toDouble),
        "actionCounts" -> objFromCounts(summary.actionCounts),
        "retrains" -> ujson.Num(summary.retrains.toDouble),
        "modelId" -> Str(summary.modelId),
        "outDir" -> Str(outDir.toString),
        "exactGtoCacheHits" -> ujson.Num(summary.exactGtoCacheHits.toDouble),
        "exactGtoCacheMisses" -> ujson.Num(summary.exactGtoCacheMisses.toDouble),
        "exactGtoCacheHitRate" -> ujson.Num(summary.exactGtoCacheHitRate),
        "exactGtoSolvedByProvider" -> objFromLongCounts(summary.exactGtoSolvedByProvider),
        "exactGtoServedByProvider" -> objFromLongCounts(summary.exactGtoServedByProvider),
        "perVillainNetChips" -> objFromDoubleCounts(summary.perVillainNetChips),
        "perHandHeroNet" -> Arr.from(summary.perHandHeroNet.map(v => ujson.Num(v))),
        "heroDecisionEquities" -> Arr.from(summary.heroDecisionEquities.map(v => ujson.Num(v))),
        "overlayStats" -> summary.overlayStats.map(renderOverlayStats).getOrElse(ujson.Null),
        "outputFiles" -> Arr.from(existingOutputFiles(outDir).map(Str(_)))
      )
    )

  private def existingOutputFiles(outDir: Path): Vector[String] =
    Vector(
      "hands.tsv",
      "learning.tsv",
      "training-selfplay.tsv",
      "ddre-training-selfplay.tsv",
      "review-upload-pokerstars.txt"
    ).flatMap { name =>
      val candidate = outDir.resolve(name)
      Option.when(Files.exists(candidate))(candidate.toString)
    }

  private def renderOverlayStats(stats: sicfun.holdem.runtime.protocol.OverlayStats): Value =
    Obj(
      "decisions" -> ujson.Num(stats.decisions.toDouble),
      "overlayChangeRate" -> ujson.Num(stats.overlayChangeRate),
      "vetoRate" -> ujson.Num(stats.vetoRate),
      "decisionsWithVeto" -> ujson.Num(stats.decisionsWithVeto.toDouble),
      "totalVetoedActions" -> ujson.Num(stats.totalVetoedActions.toDouble),
      "meanLatencyMs" -> ujson.Num(stats.meanLatencyMs),
      "p95LatencyMs" -> ujson.Num(stats.p95LatencyMs),
      "p99LatencyMs" -> ujson.Num(stats.p99LatencyMs),
      "actionDistribution" -> objFromCounts(stats.actionDistribution)
    )

  private def objFromCounts(values: Map[String, Int]): Value =
    Obj.from(values.toVector.sortBy(_._1).map { case (key, value) =>
      key -> ujson.Num(value.toDouble)
    })

  private def objFromLongCounts(values: Map[String, Long]): Value =
    Obj.from(values.toVector.sortBy(_._1).map { case (key, value) =>
      key -> ujson.Num(value.toDouble)
    })

  private def objFromDoubleCounts(values: Map[String, Double]): Value =
    Obj.from(values.toVector.sortBy(_._1).map { case (key, value) =>
      key -> ujson.Num(value)
    })

  def requiredString(
      obj: collection.Map[String, Value],
      key: String
  ): Either[(Int, String), String] =
    optionalString(obj, key).filter(_.nonEmpty).toRight(400 -> s"$key is required")

  def optionalString(
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

  def readRequestBody(
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

  final case class JsonResponse(
      status: Int,
      value: Value,
      headers: Vector[(String, String)] = Vector.empty
  )

  final case class PlayingHallRequest(
      hands: Int,
      tableCount: Int,
      playerCount: Int,
      heroStyle: String,
      heroPosition: String,
      gtoMode: String,
      villainPool: Vector[String],
      heroExplorationRate: Double,
      raiseSize: Double,
      bunchingTrials: Int,
      equityTrials: Int,
      learnEveryHands: Int,
      learningWindowSamples: Int,
      saveReviewHandHistory: Boolean,
      fullRing: Boolean,
      seed: Long
  ):
    def runDirectoryName: String =
      s"${System.currentTimeMillis()}-${UUID.randomUUID().toString.take(8)}"

    def logSummary: String =
      s"hands=$hands tableCount=$tableCount playerCount=$playerCount heroStyle=$heroStyle heroPosition=$heroPosition gtoMode=$gtoMode villainPool=${villainPool.mkString(",")} seed=$seed"

  def retryAfterSeconds(pollAfterMs: Long): String =
    math.max(1L, (pollAfterMs + 999L) / 1000L).toString
