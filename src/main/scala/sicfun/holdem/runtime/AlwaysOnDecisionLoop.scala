package sicfun.holdem.runtime
import sicfun.holdem.types.*
import sicfun.holdem.model.*
import sicfun.holdem.engine.*
import sicfun.holdem.equity.*
import sicfun.holdem.io.*
import sicfun.holdem.cli.*
import sicfun.holdem.history.*

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

  private type EventKey = (String, Long, String)

  private[holdem] final class PendingHeroRaiseTracker:
    private val pendingByHand = mutable.Map.empty[String, Boolean]

    def onHeroAction(handId: String, action: PokerAction): Unit =
      action match
        case PokerAction.Raise(_) => pendingByHand.update(handId, true)
        case _ => pendingByHand.remove(handId)

    def onVillainAction(handId: String, action: PokerAction): Option[PokerAction] =
      val response =
        action match
          case PokerAction.Fold | PokerAction.Call | PokerAction.Raise(_) =>
            if pendingByHand.getOrElse(handId, false) then Some(action) else None
          case _ => None
      pendingByHand.remove(handId)
      response

  private[holdem] def replayedArchetypePosterior(
      rememberedPosterior: Option[ArchetypePosterior],
      sessionRaiseResponses: RaiseResponseCounts
  ): Option[ArchetypePosterior] =
    rememberedPosterior match
      case Some(posterior) =>
        Some(ArchetypeLearning.posteriorFromCounts(sessionRaiseResponses, prior = posterior))
      case None if sessionRaiseResponses.total > 0 =>
        Some(ArchetypeLearning.posteriorFromCounts(sessionRaiseResponses))
      case None =>
        None

  private[holdem] def mergeVillainObservations(
      rememberedEvents: Seq[PokerEvent],
      currentHandEvents: Seq[PokerEvent],
      villainPlayerId: String
  ): Vector[VillainObservation] =
    val currentVillainEvents = currentHandEvents.filter(_.playerId == villainPlayerId).toVector
    val currentKeys = currentVillainEvents.iterator.map(eventKey).toSet
    val builder = Vector.newBuilder[VillainObservation]
    rememberedEvents.iterator
      .filter(event => event.playerId == villainPlayerId && event.action != PokerAction.Fold)
      .filterNot(event => currentKeys.contains(eventKey(event)))
      .foreach(event => builder += eventToObservation(event))
    currentVillainEvents.foreach(event => builder += eventToObservation(event))
    builder.result()

  private def eventKey(event: PokerEvent): EventKey =
    (event.handId, event.sequenceInHand, event.playerId)

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
      cfrMaxLocalExploitability: Double,
      cfrMaxBaselineActionRegret: Double,
      cfrVillainHands: Int,
      cfrEquityTrials: Int,
      cfrVillainReraises: Boolean,
      opponentStore: Option[OpponentMemoryTarget],
      opponentSite: Option[String],
      opponentName: Option[String]
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
    new LoopRunner(config).run()

  private final class LoopRunner(config: CliConfig):
    private val snapshotsRoot = config.outputDir.resolve("snapshots")
    private val modelsRoot = config.outputDir.resolve("models")
    private val decisionsLog = config.outputDir.resolve("decisions.tsv")
    private val signalsLog = config.outputDir.resolve("signals.tsv")
    private val retrainLog = config.outputDir.resolve("model-updates.tsv")
    private val contextLog = config.outputDir.resolve("context-archive.md")
    private val offsetFile = config.outputDir.resolve("feed-offset.txt")

    private var activeModelDir = config.modelArtifactDir
    private var activeArtifact = PokerActionModelArtifactIO.load(activeModelDir)
    private var engine = newAdaptiveEngine(config, activeArtifact.model)
    private var opponentProfileStore = config.opponentStore.map(OpponentProfileStorePersistence.load)
    private var opponentProfileStoreDirty = false
    private val rememberedOpponent = loadRememberedOpponent(opponentProfileStore, config)
    private val rememberedArchetypePosterior = rememberedOpponent.map(_.archetypePosterior)

    private val states = mutable.Map.empty[String, HandState]
    private var sessionRaiseResponses = RaiseResponseCounts()
    private var byteOffset = readByteOffset(offsetFile)
    private var processedEvents = 0
    private var decisionsEmitted = 0
    private var retrainCount = 0
    private var poll = 0
    private val pendingRaiseTracker = new PendingHeroRaiseTracker()
    private val folds = config.tableFormat.foldsBeforeOpener(config.openerPosition).map(PreflopFold(_))
    private val seedRng = new Random(config.seed)

    seedEngineArchetypePosterior(engine)

    def run(): Either[String, RunSummary] =
      try
        Files.createDirectories(config.outputDir)
        Files.createDirectories(snapshotsRoot)
        Files.createDirectories(modelsRoot)
        loop()
        flushOpponentProfileStore()
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
          flushOpponentProfileStore()
          Left(s"always-on decision loop failed: ${e.getMessage}")

    private def loop(): Unit =
      while config.maxPolls < 0 || poll < config.maxPolls do
        val (pending, newByteOffset) = DecisionLoopEventFeedIO.readIncremental(config.feedPath, byteOffset)
        if pending.nonEmpty then
          processPendingEvents(pending)
        persistByteOffset(newByteOffset)

        poll += 1
        if config.maxPolls < 0 || poll < config.maxPolls then
          Thread.sleep(math.max(0L, config.pollMillis))

    private def persistByteOffset(newByteOffset: Long): Unit =
      if newByteOffset != byteOffset then
        byteOffset = newByteOffset
        writeByteOffset(offsetFile, byteOffset)

    private def processPendingEvents(pending: Vector[DecisionLoopEventFeedIO.FeedEvent]): Unit =
      pending.foreach { entry =>
        val event = entry.event
        val updated = updateHandState(event)
        maybeObserveVillainResponse(event)
        maybeEmitHeroDecision(event, updated)
      }
      flushOpponentProfileStore()

    private def updateHandState(event: PokerEvent): HandState =
      val current = states.getOrElse(
        event.handId,
        HandEngine.newHand(event.handId, startedAt = event.occurredAtEpochMillis)
      )
      val updated = HandEngine.applyEvent(current, event)
      states.update(event.handId, updated)
      HandStateSnapshotIO.save(snapshotsRoot.resolve(event.handId), updated)
      processedEvents += 1
      updated

    private def maybeObserveVillainResponse(event: PokerEvent): Unit =
      if event.playerId == config.villainPlayerId then
        val responseToHeroRaise = pendingRaiseTracker.onVillainAction(event.handId, event.action)
        responseToHeroRaise.foreach { response =>
          sessionRaiseResponses = sessionRaiseResponses.observe(response)
          engine.observeVillainResponseToRaise(response)
        }
        persistVillainObservation(event, facedRaiseResponse = responseToHeroRaise.nonEmpty)

    private def persistVillainObservation(event: PokerEvent, facedRaiseResponse: Boolean): Unit =
      (config.opponentSite, config.opponentName) match
        case (Some(site), Some(name)) =>
          val currentStore = opponentProfileStore.getOrElse(OpponentProfileStore.empty)
          val updatedStore = currentStore.observeEvent(site, name, event, facedRaiseResponse)
          if updatedStore != currentStore then
            opponentProfileStore = Some(updatedStore)
            opponentProfileStoreDirty = true
        case _ => ()

    private def flushOpponentProfileStore(): Unit =
      if opponentProfileStoreDirty then
        (config.opponentStore, opponentProfileStore) match
          case (Some(target), Some(store)) =>
            OpponentProfileStorePersistence.save(target, store)
            opponentProfileStoreDirty = false
          case _ =>
            opponentProfileStoreDirty = false

    private def maybeEmitHeroDecision(event: PokerEvent, updated: HandState): Unit =
      if event.playerId == config.heroPlayerId then
        pendingRaiseTracker.onHeroAction(event.handId, event.action)
        HandEngine.toGameState(updated, config.heroPlayerId).foreach { heroState =>
          val decision = decideForHero(updated, heroState)
          logHeroDecision(event, decision)
          decisionsEmitted += 1
          maybeRetrain()
        }

    private def decideForHero(
        updated: HandState,
        heroState: GameState
    ): AdaptiveDecisionResult =
      val observations = mergeVillainObservations(
        rememberedEvents = currentRememberedVillainEvents(),
        currentHandEvents = updated.events,
        villainPlayerId = config.villainPlayerId
      )
      engine.decide(
        hero = config.heroCards,
        state = heroState,
        folds = folds,
        villainPos = config.villainPosition,
        observations = observations,
        candidateActions = config.candidateActions,
        decisionBudgetMillis = config.decisionBudgetMillis,
        rng = new Random(seedRng.nextLong())
      )

    private def logHeroDecision(event: PokerEvent, decision: AdaptiveDecisionResult): Unit =
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
        summary =
          s"hand=${event.handId} seq=${event.sequenceInHand} bestAction=${renderAction(decision.decision.recommendation.bestAction)} " +
            s"source=${decision.adaptationTrace.source} reason=${decision.adaptationTrace.reason.getOrElse("ok")}"
      )

    private def maybeRetrain(): Unit =
      if config.retrainEnabled && decisionsEmitted % config.retrainEveryDecisions == 0 then
        config.trainingDataPath.foreach(runRetrain)

    private def runRetrain(trainingPath: Path): Unit =
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
          applyRetrainResult(done)

    private def applyRetrainResult(done: TrainPokerActionModel.RunResult): Unit =
      val oldModelId = activeArtifact.version.id
      activeModelDir = done.outputDir
      activeArtifact = done.artifact
      engine = newAdaptiveEngine(config, activeArtifact.model)
      seedEngineArchetypePosterior(engine)
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

    private def seedEngineArchetypePosterior(targetEngine: RealTimeAdaptiveEngine): Unit =
      replayedArchetypePosterior(
        rememberedPosterior = rememberedArchetypePosterior,
        sessionRaiseResponses = sessionRaiseResponses
      ).foreach(targetEngine.seedArchetypePosterior)

    private def currentRememberedVillainEvents(): Vector[PokerEvent] =
      loadRememberedOpponent(opponentProfileStore, config).map(_.recentEvents).getOrElse(Vector.empty)

  private def newAdaptiveEngine(config: CliConfig, model: PokerActionModel): RealTimeAdaptiveEngine =
    val equilibriumBaselineConfig =
      if config.cfrIterations <= 0 then None
      else
        Some(
          EquilibriumBaselineConfig(
            iterations = config.cfrIterations,
            blendWeight = config.cfrBlend,
            maxLocalExploitabilityForTrust = disabledThreshold(config.cfrMaxLocalExploitability),
            maxBaselineActionRegret = disabledThreshold(config.cfrMaxBaselineActionRegret),
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

  private def loadRememberedOpponent(
      store: Option[OpponentProfileStore],
      config: CliConfig
  ): Option[OpponentProfile] =
    (store, config.opponentSite, config.opponentName) match
      case (Some(profileStore), Some(site), Some(name)) =>
        profileStore.find(site, name)
      case _ => None

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
          "handId\tsequenceInHand\tplayerId\tbestAction\theroEquityMean\theroEquityStdErr\tarchetypeMap\tmodelId\toccurredAtEpochMillis\tdecisionAttribution\tdecisionAttributionReason\tcfrRequestedBlendWeight\tcfrEffectiveBlendWeight\tcfrChosenActionRegret\tcfrLocalExploitability\tcfrRootDeviationGap\tcfrVillainDeviationGap"
        ).asJava,
        StandardCharsets.UTF_8
      )
    val decisionAttribution = decision.adaptationTrace.source.toString
    val decisionAttributionReason = decision.adaptationTrace.reason.getOrElse("")
    val cfrRequestedBlendWeight = decision.adaptationTrace.requestedBlendWeight.toString
    val cfrEffectiveBlendWeight = decision.adaptationTrace.effectiveBlendWeight.toString
    val cfrChosenActionRegret = decision.adaptationTrace.baselineChosenActionRegret.toString
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
      decisionAttribution,
      decisionAttributionReason,
      cfrRequestedBlendWeight,
      cfrEffectiveBlendWeight,
      cfrChosenActionRegret,
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

  private def readByteOffset(path: Path): Long =
    if !Files.exists(path) then 0L
    else
      val raw = Files.readString(path, StandardCharsets.UTF_8).trim
      if raw.isEmpty then 0L else raw.toLongOption.getOrElse(0L)

  private def writeByteOffset(path: Path, offset: Long): Unit =
    Files.writeString(path, offset.toString, StandardCharsets.UTF_8)

  private def parseArgs(args: Array[String]): Either[String, CliConfig] =
    if args.contains("--help") || args.contains("-h") then Left(usage)
    else
      for
        options <- CliHelpers.parseOptions(args)
        feedPath <- parseRequiredPath(options, "feedPath")
        modelArtifactDir <- parseRequiredPath(options, "modelArtifactDir")
        outputDir <- parsePathOption(options, "outputDir", Paths.get("data/live-loop"))
        heroPlayerId <- parseStringOption(options, "heroPlayerId", "hero")
        heroCards <- CliHelpers.parseHoleCardsOptionEither(options, "heroCards", "AcKh")
        villainPlayerId <- parseStringOption(options, "villainPlayerId", "villain")
        villainPosition <- CliHelpers.parsePositionOptionEither(options, "villainPosition", Position.BigBlind)
        tableFormat <- parseTableFormatOption(options, "tableFormat", TableFormat.NineMax)
        openerPosition <- CliHelpers.parsePositionOptionEither(options, "openerPosition", Position.Cutoff)
        candidateActions <- CliHelpers.parseCandidateActionsOptionEither(options, "candidateActions", "fold,call,raise:20", deduplicate = false)
        bunchingTrials <- CliHelpers.parseIntOptionEither(options, "bunchingTrials", 300)
        _ <- if bunchingTrials > 0 then Right(()) else Left("--bunchingTrials must be > 0")
        equityTrials <- CliHelpers.parseIntOptionEither(options, "equityTrials", 3000)
        _ <- if equityTrials > 0 then Right(()) else Left("--equityTrials must be > 0")
        decisionBudgetMillis <- CliHelpers.parseOptionalLongOptionEither(options, "decisionBudgetMillis")
        _ <- decisionBudgetMillis match
          case Some(ms) if ms <= 0L => Left("--decisionBudgetMillis must be > 0")
          case _ => Right(())
        pollMillis <- CliHelpers.parseLongOptionEither(options, "pollMillis", 500L)
        _ <- if pollMillis >= 0L then Right(()) else Left("--pollMillis must be >= 0")
        maxPolls <- CliHelpers.parseIntOptionEither(options, "maxPolls", 1)
        _ <- if maxPolls == 0 || maxPolls >= -1 then Right(()) else Left("--maxPolls must be -1, 0, or > 0")
        seed <- CliHelpers.parseLongOptionEither(options, "seed", 42L)
        retrainEnabled <- CliHelpers.parseStrictBooleanOptionEither(options, "retrainEnabled", false)
        retrainEveryDecisions <- CliHelpers.parseIntOptionEither(options, "retrainEveryDecisions", 50)
        _ <- if retrainEveryDecisions > 0 then Right(()) else Left("--retrainEveryDecisions must be > 0")
        trainingDataPath <- parseOptionalPathOption(options, "trainingDataPath")
        _ <- if retrainEnabled && trainingDataPath.isEmpty then
          Left("--trainingDataPath is required when --retrainEnabled=true")
        else Right(())
        retrainLearningRate <- CliHelpers.parseDoubleOptionEither(options, "retrainLearningRate", 0.1, "a double")
        retrainIterations <- CliHelpers.parseIntOptionEither(options, "retrainIterations", 300)
        retrainL2Lambda <- CliHelpers.parseDoubleOptionEither(options, "retrainL2Lambda", 0.001, "a double")
        retrainValidationFraction <- CliHelpers.parseDoubleOptionEither(options, "retrainValidationFraction", 0.2, "a double")
        retrainSplitSeed <- CliHelpers.parseLongOptionEither(options, "retrainSplitSeed", 1L)
        retrainMaxMeanBrierScore <- CliHelpers.parseDoubleOptionEither(options, "retrainMaxMeanBrierScore", 2.0, "a double")
        retrainFailOnGate <- CliHelpers.parseStrictBooleanOptionEither(options, "retrainFailOnGate", false)
        cfrIterations <- CliHelpers.parseIntOptionEither(options, "cfrIterations", 0)
        _ <- if cfrIterations >= 0 then Right(()) else Left("--cfrIterations must be >= 0")
        cfrBlend <- CliHelpers.parseDoubleOptionEither(options, "cfrBlend", 0.35, "a double")
        _ <- if cfrBlend >= 0.0 && cfrBlend <= 1.0 then Right(()) else Left("--cfrBlend must be in [0,1]")
        cfrMaxLocalExploitability <- CliHelpers.parseDoubleOptionEither(options, "cfrMaxLocalExploitability", -1.0, "a double")
        _ <- if cfrMaxLocalExploitability >= 0.0 || cfrMaxLocalExploitability == -1.0 then Right(())
          else Left("--cfrMaxLocalExploitability must be >= 0 or -1 to disable")
        cfrMaxBaselineActionRegret <- CliHelpers.parseDoubleOptionEither(options, "cfrMaxBaselineActionRegret", -1.0, "a double")
        _ <- if cfrMaxBaselineActionRegret >= 0.0 || cfrMaxBaselineActionRegret == -1.0 then Right(())
          else Left("--cfrMaxBaselineActionRegret must be >= 0 or -1 to disable")
        cfrVillainHands <- CliHelpers.parseIntOptionEither(options, "cfrVillainHands", 96)
        _ <- if cfrVillainHands > 0 then Right(()) else Left("--cfrVillainHands must be > 0")
        cfrEquityTrials <- CliHelpers.parseIntOptionEither(options, "cfrEquityTrials", 4000)
        _ <- if cfrEquityTrials > 0 then Right(()) else Left("--cfrEquityTrials must be > 0")
        cfrVillainReraises <- CliHelpers.parseBooleanOptionEither(options, "cfrVillainReraises", true)
        opponentStore <- parseOptionalOpponentStore(options)
        opponentSite = options.get("opponentSite").map(_.trim).filter(_.nonEmpty)
        opponentName = options.get("opponentName").map(_.trim).filter(_.nonEmpty)
        _ <- validateOpponentMemoryArgs(opponentStore, opponentSite, opponentName)
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
        cfrMaxLocalExploitability = cfrMaxLocalExploitability,
        cfrMaxBaselineActionRegret = cfrMaxBaselineActionRegret,
        cfrVillainHands = cfrVillainHands,
        cfrEquityTrials = cfrEquityTrials,
        cfrVillainReraises = cfrVillainReraises,
        opponentStore = opponentStore,
        opponentSite = opponentSite,
        opponentName = opponentName
      )

  private def parseRequiredPath(options: Map[String, String], key: String): Either[String, Path] =
    options.get(key).map(v => Right(Paths.get(v))).getOrElse(Left(s"--$key is required"))

  private def parsePathOption(options: Map[String, String], key: String, default: Path): Either[String, Path] =
    options.get(key).map(v => Right(Paths.get(v))).getOrElse(Right(default))

  private def parseOptionalPathOption(options: Map[String, String], key: String): Either[String, Option[Path]] =
    Right(options.get(key).map(Paths.get(_)))

  private def parseOptionalOpponentStore(options: Map[String, String]): Either[String, Option[OpponentMemoryTarget]] =
    options.get("opponentStore") match
      case None => Right(None)
      case Some(raw) =>
        OpponentMemoryTarget.parse(
          raw = raw,
          user = options.get("opponentStoreUser"),
          password = options.get("opponentStorePassword"),
          schema = options.getOrElse("opponentStoreSchema", "public")
        ).map(Some(_))

  private def validateOpponentMemoryArgs(
      opponentStore: Option[OpponentMemoryTarget],
      opponentSite: Option[String],
      opponentName: Option[String]
  ): Either[String, Unit] =
    if opponentStore.isDefined == opponentSite.isDefined &&
      opponentSite.isDefined == opponentName.isDefined
    then Right(())
    else Left("--opponentStore, --opponentSite, and --opponentName must be provided together")

  private def parseStringOption(options: Map[String, String], key: String, default: String): Either[String, String] =
    Right(options.getOrElse(key, default))

  private def disabledThreshold(raw: Double): Double =
    if raw < 0.0 then Double.PositiveInfinity else raw

  private def parseTableFormatOption(options: Map[String, String], key: String, default: TableFormat): Either[String, TableFormat] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) =>
        raw.trim.toLowerCase match
          case "headsup" | "heads-up" | "hu" => Right(TableFormat.HeadsUp)
          case "sixmax" | "6max" => Right(TableFormat.SixMax)
          case "ninemax" | "9max" => Right(TableFormat.NineMax)
          case _ => Left(s"--$key must be headsup|sixmax|ninemax")

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
      |  --cfrMaxLocalExploitability=<double> Default -1 (disabled; trust gate for CFR solve quality)
      |  --cfrMaxBaselineActionRegret=<double> Default -1 (disabled; clamps actions that drift too far from CFR-best EV)
      |  --cfrVillainHands=<int>           Default 96
      |  --cfrEquityTrials=<int>           Default 4000
      |  --cfrVillainReraises=<true|false> Default true
      |  --opponentStore=<path|jdbc:postgresql://...> Optional persisted opponent memory store
      |  --opponentStoreUser=<user>        Optional PostgreSQL user
      |  --opponentStorePassword=<password> Optional PostgreSQL password
      |  --opponentStoreSchema=<schema>    Optional PostgreSQL schema (default: public)
      |  --opponentSite=<id>               Opponent site key for lookup
      |  --opponentName=<name>             Opponent screen name for lookup
      |""".stripMargin
