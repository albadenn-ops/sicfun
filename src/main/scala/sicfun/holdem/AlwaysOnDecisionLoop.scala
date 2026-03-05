package sicfun.holdem

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths, StandardOpenOption}
import java.time.Instant
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import scala.collection.mutable
import scala.jdk.CollectionConverters.*
import scala.util.Random

/** Always-on decision service:
  *  - polls an append-only event feed
  *  - updates per-hand state snapshots
  *  - emits hero decisions in real time
  *  - appends signals and operational context records
  *  - optionally retrains/reloads the action model on a schedule
  */
object AlwaysOnDecisionLoop:
  private val IsoFormatter = DateTimeFormatter.ISO_INSTANT.withZone(ZoneOffset.UTC)

  private final case class CliConfig(
      feedPath: Path,
      modelArtifactDir: Path,
      outputDir: Path,
      heroPlayerId: String,
      heroCards: HoleCards,
      villainPlayerId: String,
      villainPosition: Position,
      tableFormat: TableFormat,
      openerPosition: Position,
      candidateActions: Vector[PokerAction],
      bunchingTrials: Int,
      equityTrials: Int,
      decisionBudgetMillis: Option[Long],
      pollMillis: Long,
      maxPolls: Int,
      seed: Long,
      retrainEnabled: Boolean,
      retrainEveryDecisions: Int,
      trainingDataPath: Option[Path],
      retrainLearningRate: Double,
      retrainIterations: Int,
      retrainL2Lambda: Double,
      retrainValidationFraction: Double,
      retrainSplitSeed: Long,
      retrainMaxMeanBrierScore: Double,
      retrainFailOnGate: Boolean,
      cfrIterations: Int,
      cfrBlend: Double,
      cfrVillainHands: Int,
      cfrEquityTrials: Int,
      cfrVillainReraises: Boolean
  )

  final case class RunSummary(
      processedEvents: Int,
      decisionsEmitted: Int,
      retrainCount: Int,
      latestModelDir: Path,
      outputDir: Path
  )

  def main(args: Array[String]): Unit =
    val wantsHelp = args.contains("--help") || args.contains("-h")
    run(args) match
      case Right(result) =>
        println("=== Always-On Decision Loop ===")
        println(s"processedEvents: ${result.processedEvents}")
        println(s"decisionsEmitted: ${result.decisionsEmitted}")
        println(s"retrainCount: ${result.retrainCount}")
        println(s"latestModelDir: ${result.latestModelDir.toAbsolutePath.normalize()}")
        println(s"outputDir: ${result.outputDir.toAbsolutePath.normalize()}")
      case Left(error) =>
        if wantsHelp then println(error)
        else
          System.err.println(error)
          sys.exit(1)

  def run(args: Array[String]): Either[String, RunSummary] =
    for
      config <- parseArgs(args)
      result <- runConfig(config)
    yield result

  private def runConfig(config: CliConfig): Either[String, RunSummary] =
    try
      Files.createDirectories(config.outputDir)
      val snapshotsRoot = config.outputDir.resolve("snapshots")
      val modelsRoot = config.outputDir.resolve("models")
      val decisionsLog = config.outputDir.resolve("decisions.tsv")
      val signalsLog = config.outputDir.resolve("signals.tsv")
      val retrainLog = config.outputDir.resolve("model-updates.tsv")
      val contextLog = config.outputDir.resolve("context-archive.md")
      val offsetFile = config.outputDir.resolve("feed-offset.txt")

      var activeModelDir = config.modelArtifactDir
      var activeArtifact = PokerActionModelArtifactIO.load(activeModelDir)
      var engine = newAdaptiveEngine(config, activeArtifact.model)

      val states = mutable.Map.empty[String, HandState]
      val observedVillainResponses = mutable.ArrayBuffer.empty[PokerAction]
      var processedOffset = readOffset(offsetFile)
      var processedEvents = 0
      var decisionsEmitted = 0
      var retrainCount = 0
      var poll = 0

      val folds = config.tableFormat.foldsBeforeOpener(config.openerPosition).map(PreflopFold(_))
      val seedRng = new Random(config.seed)

      while config.maxPolls < 0 || poll < config.maxPolls do
        val feed = DecisionLoopEventFeedIO.read(config.feedPath)
        if processedOffset < feed.length then
          val pending = feed.drop(processedOffset)
          pending.foreach { entry =>
            val event = entry.event
            val current = states.getOrElse(
              event.handId,
              HandEngine.newHand(event.handId, startedAt = event.occurredAtEpochMillis)
            )
            val updated = HandEngine.applyEvent(current, event)
            states.update(event.handId, updated)
            HandStateSnapshotIO.save(snapshotsRoot.resolve(event.handId), updated)
            processedEvents += 1

            if event.playerId == config.villainPlayerId then
              event.action match
                case PokerAction.Fold | PokerAction.Call | PokerAction.Raise(_) =>
                  observedVillainResponses += event.action
                  engine.observeVillainResponseToRaise(event.action)
                case _ => ()

            if event.playerId == config.heroPlayerId then
              HandEngine.toGameState(updated, config.heroPlayerId).foreach { heroState =>
                val observations = updated.events
                  .filter(_.playerId == config.villainPlayerId)
                  .map(eventToObservation)
                val decision = engine.decide(
                  hero = config.heroCards,
                  state = heroState,
                  folds = folds,
                  villainPos = config.villainPosition,
                  observations = observations,
                  candidateActions = config.candidateActions,
                  decisionBudgetMillis = config.decisionBudgetMillis,
                  rng = new Random(seedRng.nextLong())
                )
                appendDecision(
                  decisionsLog = decisionsLog,
                  event = event,
                  decision = decision,
                  modelId = activeArtifact.version.id
                )
                val signal = SignalBuilder.actionRisk(
                  event = event,
                  artifact = activeArtifact,
                  snapshotDirectory = snapshotsRoot.resolve(event.handId).toString,
                  modelArtifactDirectory = activeModelDir.toString
                )
                SignalAuditLogIO.append(signalsLog, signal)
                appendContextEntry(
                  contextLog,
                  title = "hero-decision",
                  summary = s"hand=${event.handId} seq=${event.sequenceInHand} bestAction=${renderAction(decision.decision.recommendation.bestAction)}"
                )

                decisionsEmitted += 1
                if config.retrainEnabled && decisionsEmitted % config.retrainEveryDecisions == 0 then
                  config.trainingDataPath match
                    case Some(trainingPath) =>
                      val retrainTarget = modelsRoot.resolve(s"model-retrain-$decisionsEmitted")
                      val retrainResult = TrainPokerActionModel.run(Array(
                        trainingPath.toString,
                        retrainTarget.toString,
                        s"--learningRate=${config.retrainLearningRate}",
                        s"--iterations=${config.retrainIterations}",
                        s"--l2Lambda=${config.retrainL2Lambda}",
                        s"--validationFraction=${config.retrainValidationFraction}",
                        s"--splitSeed=${config.retrainSplitSeed}",
                        s"--maxMeanBrierScore=${config.retrainMaxMeanBrierScore}",
                        s"--failOnGate=${config.retrainFailOnGate}",
                        s"--modelId=retrain-$decisionsEmitted",
                        "--source=always-on-decision-loop"
                      ))
                      retrainResult match
                        case Left(err) =>
                          appendModelUpdate(
                            path = retrainLog,
                            atEpochMillis = System.currentTimeMillis(),
                            oldModelId = activeArtifact.version.id,
                            newModelId = activeArtifact.version.id,
                            status = "retrain-failed",
                            message = err
                          )
                          appendContextEntry(
                            contextLog,
                            title = "retrain-failed",
                            summary = s"decision=$decisionsEmitted error=$err"
                          )
                        case Right(done) =>
                          val oldModelId = activeArtifact.version.id
                          activeModelDir = done.outputDir
                          activeArtifact = done.artifact
                          engine = newAdaptiveEngine(config, activeArtifact.model)
                          observedVillainResponses.foreach(engine.observeVillainResponseToRaise)
                          retrainCount += 1
                          appendModelUpdate(
                            path = retrainLog,
                            atEpochMillis = System.currentTimeMillis(),
                            oldModelId = oldModelId,
                            newModelId = activeArtifact.version.id,
                            status = "retrain-succeeded",
                            message = done.outputDir.toAbsolutePath.normalize().toString
                          )
                          appendContextEntry(
                            contextLog,
                            title = "retrain-succeeded",
                            summary = s"decision=$decisionsEmitted newModel=${activeArtifact.version.id}"
                          )
                    case None =>
                      ()
              }
          }
          processedOffset = feed.length
          writeOffset(offsetFile, processedOffset)

        poll += 1
        if config.maxPolls < 0 || poll < config.maxPolls then
          Thread.sleep(math.max(0L, config.pollMillis))

      Right(
        RunSummary(
          processedEvents = processedEvents,
          decisionsEmitted = decisionsEmitted,
          retrainCount = retrainCount,
          latestModelDir = activeModelDir,
          outputDir = config.outputDir
        )
      )
    catch
      case e: Exception =>
        Left(s"always-on decision loop failed: ${e.getMessage}")

  private def newAdaptiveEngine(config: CliConfig, model: PokerActionModel): RealTimeAdaptiveEngine =
    val equilibriumBaselineConfig =
      if config.cfrIterations <= 0 then None
      else
        Some(
          EquilibriumBaselineConfig(
            iterations = config.cfrIterations,
            blendWeight = config.cfrBlend,
            maxVillainHands = config.cfrVillainHands,
            equityTrials = config.cfrEquityTrials,
            includeVillainReraises = config.cfrVillainReraises
          )
        )
    new RealTimeAdaptiveEngine(
      tableRanges = TableRanges.defaults(config.tableFormat),
      actionModel = model,
      bunchingTrials = config.bunchingTrials,
      defaultEquityTrials = config.equityTrials,
      minEquityTrials = math.max(200, config.equityTrials / 10),
      equilibriumBaselineConfig = equilibriumBaselineConfig
    )

  private def eventToObservation(event: PokerEvent): VillainObservation =
    VillainObservation(
      action = event.action,
      state = GameState(
        street = event.street,
        board = event.board,
        pot = event.potBefore,
        toCall = event.toCall,
        position = event.position,
        stackSize = event.stackBefore,
        betHistory = event.betHistory
      )
    )

  private def appendDecision(
      decisionsLog: Path,
      event: PokerEvent,
      decision: AdaptiveDecisionResult,
      modelId: String
  ): Unit =
    if !Files.exists(decisionsLog) then
      Files.write(
        decisionsLog,
        Vector(
          "handId\tsequenceInHand\tplayerId\tbestAction\theroEquityMean\theroEquityStdErr\tarchetypeMap\tmodelId\toccurredAtEpochMillis\tcfrLocalExploitability\tcfrRootDeviationGap\tcfrVillainDeviationGap"
        ).asJava,
        StandardCharsets.UTF_8
      )
    val cfrLocalExploitability = decision.equilibriumBaseline.map(_.localExploitability.toString).getOrElse("")
    val cfrRootGap = decision.equilibriumBaseline.map(_.rootDeviationGap.toString).getOrElse("")
    val cfrVillainGap = decision.equilibriumBaseline.map(_.villainDeviationGap.toString).getOrElse("")
    val row = Vector(
      event.handId,
      event.sequenceInHand.toString,
      event.playerId,
      renderAction(decision.decision.recommendation.bestAction),
      decision.decision.recommendation.heroEquity.mean.toString,
      decision.decision.recommendation.heroEquity.stderr.toString,
      decision.archetypeMap.toString,
      modelId,
      event.occurredAtEpochMillis.toString,
      cfrLocalExploitability,
      cfrRootGap,
      cfrVillainGap
    ).mkString("\t")
    Files.write(
      decisionsLog,
      Vector(row).asJava,
      StandardCharsets.UTF_8,
      StandardOpenOption.APPEND
    )

  private def appendModelUpdate(
      path: Path,
      atEpochMillis: Long,
      oldModelId: String,
      newModelId: String,
      status: String,
      message: String
  ): Unit =
    if !Files.exists(path) then
      Files.write(
        path,
        Vector("atEpochMillis\toldModelId\tnewModelId\tstatus\tmessage").asJava,
        StandardCharsets.UTF_8
      )
    val row = Vector(
      atEpochMillis.toString,
      oldModelId,
      newModelId,
      status,
      message.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
    ).mkString("\t")
    Files.write(path, Vector(row).asJava, StandardCharsets.UTF_8, StandardOpenOption.APPEND)

  private def appendContextEntry(path: Path, title: String, summary: String): Unit =
    val timestamp = IsoFormatter.format(Instant.ofEpochMilli(System.currentTimeMillis()))
    if !Files.exists(path) then
      Files.write(
        path,
        Vector("# AI Context Archive", "", "Append-only operational context for future AI sessions.", "").asJava,
        StandardCharsets.UTF_8
      )
    val lines = Vector(
      s"## $timestamp - $title",
      summary,
      ""
    )
    Files.write(path, lines.asJava, StandardCharsets.UTF_8, StandardOpenOption.APPEND)

  private def readOffset(path: Path): Int =
    if !Files.exists(path) then 0
    else
      val raw = Files.readString(path, StandardCharsets.UTF_8).trim
      if raw.isEmpty then 0 else raw.toIntOption.getOrElse(0)

  private def writeOffset(path: Path, offset: Int): Unit =
    Files.writeString(path, offset.toString, StandardCharsets.UTF_8)

  private def parseArgs(args: Array[String]): Either[String, CliConfig] =
    if args.contains("--help") || args.contains("-h") then Left(usage)
    else
      for
        options <- parseOptions(args)
        feedPath <- parseRequiredPath(options, "feedPath")
        modelArtifactDir <- parseRequiredPath(options, "modelArtifactDir")
        outputDir <- parsePathOption(options, "outputDir", Paths.get("data/live-loop"))
        heroPlayerId <- parseStringOption(options, "heroPlayerId", "hero")
        heroCards <- parseHoleCardsOption(options, "heroCards", "AcKh")
        villainPlayerId <- parseStringOption(options, "villainPlayerId", "villain")
        villainPosition <- parsePositionOption(options, "villainPosition", Position.BigBlind)
        tableFormat <- parseTableFormatOption(options, "tableFormat", TableFormat.NineMax)
        openerPosition <- parsePositionOption(options, "openerPosition", Position.Cutoff)
        candidateActions <- parseCandidateActionsOption(options, "candidateActions", "fold,call,raise:20")
        bunchingTrials <- parseIntOption(options, "bunchingTrials", 300)
        _ <- if bunchingTrials > 0 then Right(()) else Left("--bunchingTrials must be > 0")
        equityTrials <- parseIntOption(options, "equityTrials", 3000)
        _ <- if equityTrials > 0 then Right(()) else Left("--equityTrials must be > 0")
        decisionBudgetMillis <- parseOptionalLongOption(options, "decisionBudgetMillis")
        _ <- decisionBudgetMillis match
          case Some(ms) if ms <= 0L => Left("--decisionBudgetMillis must be > 0")
          case _ => Right(())
        pollMillis <- parseLongOption(options, "pollMillis", 500L)
        _ <- if pollMillis >= 0L then Right(()) else Left("--pollMillis must be >= 0")
        maxPolls <- parseIntOption(options, "maxPolls", 1)
        _ <- if maxPolls == 0 || maxPolls >= -1 then Right(()) else Left("--maxPolls must be -1, 0, or > 0")
        seed <- parseLongOption(options, "seed", 42L)
        retrainEnabled <- parseBooleanOption(options, "retrainEnabled", false)
        retrainEveryDecisions <- parseIntOption(options, "retrainEveryDecisions", 50)
        _ <- if retrainEveryDecisions > 0 then Right(()) else Left("--retrainEveryDecisions must be > 0")
        trainingDataPath <- parseOptionalPathOption(options, "trainingDataPath")
        _ <- if retrainEnabled && trainingDataPath.isEmpty then
          Left("--trainingDataPath is required when --retrainEnabled=true")
        else Right(())
        retrainLearningRate <- parseDoubleOption(options, "retrainLearningRate", 0.1)
        retrainIterations <- parseIntOption(options, "retrainIterations", 300)
        retrainL2Lambda <- parseDoubleOption(options, "retrainL2Lambda", 0.001)
        retrainValidationFraction <- parseDoubleOption(options, "retrainValidationFraction", 0.2)
        retrainSplitSeed <- parseLongOption(options, "retrainSplitSeed", 1L)
        retrainMaxMeanBrierScore <- parseDoubleOption(options, "retrainMaxMeanBrierScore", 2.0)
        retrainFailOnGate <- parseBooleanOption(options, "retrainFailOnGate", false)
        cfrIterations <- parseIntOption(options, "cfrIterations", 0)
        _ <- if cfrIterations >= 0 then Right(()) else Left("--cfrIterations must be >= 0")
        cfrBlend <- parseDoubleOption(options, "cfrBlend", 0.35)
        _ <- if cfrBlend >= 0.0 && cfrBlend <= 1.0 then Right(()) else Left("--cfrBlend must be in [0,1]")
        cfrVillainHands <- parseIntOption(options, "cfrVillainHands", 96)
        _ <- if cfrVillainHands > 0 then Right(()) else Left("--cfrVillainHands must be > 0")
        cfrEquityTrials <- parseIntOption(options, "cfrEquityTrials", 4000)
        _ <- if cfrEquityTrials > 0 then Right(()) else Left("--cfrEquityTrials must be > 0")
        cfrVillainReraises <- parseBooleanOption(options, "cfrVillainReraises", true)
      yield CliConfig(
        feedPath = feedPath,
        modelArtifactDir = modelArtifactDir,
        outputDir = outputDir,
        heroPlayerId = heroPlayerId,
        heroCards = heroCards,
        villainPlayerId = villainPlayerId,
        villainPosition = villainPosition,
        tableFormat = tableFormat,
        openerPosition = openerPosition,
        candidateActions = candidateActions,
        bunchingTrials = bunchingTrials,
        equityTrials = equityTrials,
        decisionBudgetMillis = decisionBudgetMillis,
        pollMillis = pollMillis,
        maxPolls = maxPolls,
        seed = seed,
        retrainEnabled = retrainEnabled,
        retrainEveryDecisions = retrainEveryDecisions,
        trainingDataPath = trainingDataPath,
        retrainLearningRate = retrainLearningRate,
        retrainIterations = retrainIterations,
        retrainL2Lambda = retrainL2Lambda,
        retrainValidationFraction = retrainValidationFraction,
        retrainSplitSeed = retrainSplitSeed,
        retrainMaxMeanBrierScore = retrainMaxMeanBrierScore,
        retrainFailOnGate = retrainFailOnGate,
        cfrIterations = cfrIterations,
        cfrBlend = cfrBlend,
        cfrVillainHands = cfrVillainHands,
        cfrEquityTrials = cfrEquityTrials,
        cfrVillainReraises = cfrVillainReraises
      )

  private def parseOptions(args: Array[String]): Either[String, Map[String, String]] =
    val pairs = args.toVector.map { token =>
      if !token.startsWith("--") then Left(s"invalid argument '$token'; expected --key=value")
      else
        val body = token.drop(2)
        val idx = body.indexOf('=')
        if idx <= 0 || idx == body.length - 1 then Left(s"invalid argument '$token'; expected --key=value")
        else
          val key = body.substring(0, idx).trim
          val value = body.substring(idx + 1).trim
          if key.isEmpty || value.isEmpty then Left(s"invalid argument '$token'; key/value must be non-empty")
          else Right(key -> value)
    }
    pairs.foldLeft(Right(Map.empty): Either[String, Map[String, String]]) {
      case (Left(err), _) => Left(err)
      case (Right(_), Left(err)) => Left(err)
      case (Right(acc), Right((k, v))) => Right(acc.updated(k, v))
    }

  private def parseRequiredPath(options: Map[String, String], key: String): Either[String, Path] =
    options.get(key).map(v => Right(Paths.get(v))).getOrElse(Left(s"--$key is required"))

  private def parsePathOption(options: Map[String, String], key: String, default: Path): Either[String, Path] =
    options.get(key).map(v => Right(Paths.get(v))).getOrElse(Right(default))

  private def parseOptionalPathOption(options: Map[String, String], key: String): Either[String, Option[Path]] =
    Right(options.get(key).map(Paths.get(_)))

  private def parseStringOption(options: Map[String, String], key: String, default: String): Either[String, String] =
    Right(options.getOrElse(key, default))

  private def parseIntOption(options: Map[String, String], key: String, default: Int): Either[String, Int] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) => raw.toIntOption.toRight(s"--$key must be an integer")

  private def parseLongOption(options: Map[String, String], key: String, default: Long): Either[String, Long] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) => raw.toLongOption.toRight(s"--$key must be a long")

  private def parseOptionalLongOption(options: Map[String, String], key: String): Either[String, Option[Long]] =
    options.get(key) match
      case None => Right(None)
      case Some(raw) => raw.toLongOption.map(Some(_)).toRight(s"--$key must be a long")

  private def parseDoubleOption(options: Map[String, String], key: String, default: Double): Either[String, Double] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) => raw.toDoubleOption.toRight(s"--$key must be a double")

  private def parseBooleanOption(options: Map[String, String], key: String, default: Boolean): Either[String, Boolean] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) =>
        raw.trim.toLowerCase match
          case "true" => Right(true)
          case "false" => Right(false)
          case _ => Left(s"--$key must be true or false")

  private def parseHoleCardsOption(options: Map[String, String], key: String, default: String): Either[String, HoleCards] =
    try Right(CliHelpers.parseHoleCards(options.getOrElse(key, default)))
    catch case e: Exception => Left(s"--$key invalid: ${e.getMessage}")

  private def parsePositionOption(options: Map[String, String], key: String, default: Position): Either[String, Position] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) =>
        Position.values.find(_.toString.equalsIgnoreCase(raw.trim))
          .toRight(s"--$key invalid position: $raw")

  private def parseTableFormatOption(options: Map[String, String], key: String, default: TableFormat): Either[String, TableFormat] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) =>
        raw.trim.toLowerCase match
          case "headsup" | "heads-up" | "hu" => Right(TableFormat.HeadsUp)
          case "sixmax" | "6max" => Right(TableFormat.SixMax)
          case "ninemax" | "9max" => Right(TableFormat.NineMax)
          case _ => Left(s"--$key must be headsup|sixmax|ninemax")

  private def parseCandidateActionsOption(
      options: Map[String, String],
      key: String,
      default: String
  ): Either[String, Vector[PokerAction]] =
    val raw = options.getOrElse(key, default)
    val tokens = raw.split(",").toVector.map(_.trim).filter(_.nonEmpty)
    if tokens.isEmpty then Left(s"--$key must contain at least one action")
    else
      val parsed = tokens.map(parseActionToken)
      parsed.collectFirst { case Left(err) => err } match
        case Some(err) => Left(s"--$key invalid: $err")
        case None => Right(parsed.collect { case Right(action) => action })

  private def parseActionToken(token: String): Either[String, PokerAction] =
    token.toLowerCase match
      case "fold" => Right(PokerAction.Fold)
      case "check" => Right(PokerAction.Check)
      case "call" => Right(PokerAction.Call)
      case raw if raw.startsWith("raise:") =>
        raw.drop(6).toDoubleOption match
          case Some(amount) if amount > 0.0 => Right(PokerAction.Raise(amount))
          case _ => Left(s"invalid raise amount in '$token'")
      case _ => Left(s"unsupported action token '$token'")

  private def renderAction(action: PokerAction): String =
    action match
      case PokerAction.Fold => "Fold"
      case PokerAction.Check => "Check"
      case PokerAction.Call => "Call"
      case PokerAction.Raise(amount) => s"Raise:$amount"

  private val usage =
    """Usage:
      |  runMain sicfun.holdem.AlwaysOnDecisionLoop --feedPath=<events.tsv> --modelArtifactDir=<dir> [--key=value ...]
      |
      |Required:
      |  --feedPath=<path>                 Append-only TSV feed (DecisionLoopEventFeedIO header)
      |  --modelArtifactDir=<path>         Initial trained model artifact directory
      |
      |Core:
      |  --outputDir=<path>                Default data/live-loop
      |  --heroPlayerId=<id>               Default hero
      |  --heroCards=<AcKh>                Default AcKh
      |  --villainPlayerId=<id>            Default villain
      |  --villainPosition=<Position>      Default BigBlind
      |  --tableFormat=<ninemax|sixmax|headsup>  Default ninemax
      |  --openerPosition=<Position>       Default Cutoff
      |  --candidateActions=<csv>          Default fold,call,raise:20
      |  --bunchingTrials=<int>            Default 300
      |  --equityTrials=<int>              Default 3000
      |  --decisionBudgetMillis=<long>     Optional latency budget
      |  --pollMillis=<long>               Default 500
      |  --maxPolls=<int>                  Default 1 (-1 for continuous)
      |  --seed=<long>                     Default 42
      |
      |Scheduled retrain (optional):
      |  --retrainEnabled=<true|false>     Default false
      |  --retrainEveryDecisions=<int>     Default 50
      |  --trainingDataPath=<path>         Required if retrainEnabled=true
      |  --retrainLearningRate=<double>    Default 0.1
      |  --retrainIterations=<int>         Default 300
      |  --retrainL2Lambda=<double>        Default 0.001
      |  --retrainValidationFraction=<double> Default 0.2
      |  --retrainSplitSeed=<long>         Default 1
      |  --retrainMaxMeanBrierScore=<double> Default 2.0
      |  --retrainFailOnGate=<true|false>  Default false
      |
      |Equilibrium baseline (optional CFR):
      |  --cfrIterations=<int>             Default 0 (disabled)
      |  --cfrBlend=<double>               Default 0.35 (0..1)
      |  --cfrVillainHands=<int>           Default 96
      |  --cfrEquityTrials=<int>           Default 4000
      |  --cfrVillainReraises=<true|false> Default true
      |""".stripMargin
