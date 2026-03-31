package sicfun.holdem.validation

import sicfun.holdem.engine.RealTimeAdaptiveEngine
import sicfun.holdem.equity.{TableFormat, TableRanges}
import sicfun.holdem.model.{PokerActionModel, PokerActionModelArtifactIO}

import java.nio.file.{Files, Path, Paths}
import java.time.Instant
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter

/** EV-oriented heads-up proof harness for the adaptive hero. */
object AdaptiveProofHarness:
  private val HeroName = "Hero"
  private val RunLabelFormatter = DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss").withZone(ZoneOffset.UTC)

  final case class OpponentSpec(
      name: String,
      role: String,
      leak: InjectedLeak,
      strategy: VillainStrategy,
      strategyLabel: String,
      baselineNoise: Double
  )

  def defaultOpponentMatrix: Vector[OpponentSpec] =
    Vector(
      OpponentSpec(
        name = "Villain01_overfold",
        role = "Leaker",
        leak = OverfoldsToAggression(0.20),
        strategy = EquityBasedStrategy(),
        strategyLabel = "equity-based",
        baselineNoise = 0.03
      ),
      OpponentSpec(
        name = "Villain02_overcall",
        role = "Leaker",
        leak = Overcalls(0.25),
        strategy = EquityBasedStrategy(),
        strategyLabel = "equity-based",
        baselineNoise = 0.03
      ),
      OpponentSpec(
        name = "Villain03_turnbluff",
        role = "Leaker",
        leak = OverbluffsTurnBarrel(0.18),
        strategy = EquityBasedStrategy(),
        strategyLabel = "equity-based",
        baselineNoise = 0.03
      ),
      OpponentSpec(
        name = "Villain04_prefloploose",
        role = "Leaker",
        leak = PreflopTooLoose(0.22),
        strategy = EquityBasedStrategy(),
        strategyLabel = "equity-based",
        baselineNoise = 0.03
      ),
      OpponentSpec(
        name = "Villain05_prefloptight",
        role = "Leaker",
        leak = PreflopTooTight(0.15),
        strategy = EquityBasedStrategy(),
        strategyLabel = "equity-based",
        baselineNoise = 0.03
      ),
      OpponentSpec(
        name = "Villain06_gto",
        role = "Control",
        leak = NoLeak(),
        strategy = CfrVillainStrategy(allowHeuristicFallback = false),
        strategyLabel = "cfr-no-fallback",
        baselineNoise = 0.0
      )
    )

  final case class Config(
      handsPerOpponent: Int = 500,
      outputDir: Path = Paths.get("data", "adaptive-proof"),
      runLabel: Option[String] = None,
      modelDir: Option[Path] = None,
      seed: Long = 42L,
      bunchingTrials: Int = 100,
      equityTrials: Int = 500,
      minEquityTrials: Int = 100,
      budgetMs: Long = 50L,
      opponents: Vector[OpponentSpec] = defaultOpponentMatrix
  )

  final case class OpponentResult(
      name: String,
      role: String,
      leakId: String,
      severity: Double,
      strategy: String,
      outputDir: Path,
      legButtonPath: Path,
      legBigBlindPath: Path,
      combinedPath: Path,
      heroNetBbPer100: Double,
      heroNetBbPer100ByLeg: Map[String, Double],
      leakFiredCount: Int,
      leakApplicableSpots: Int,
      heroRaiseResponseCount: Int
  )

  final case class RunSummary(
      runDir: Path,
      seed: Long,
      handsPerOpponent: Int,
      opponents: Vector[OpponentResult],
      manifestPath: Path,
      groundTruthPath: Path,
      reportPath: Path,
      combinedHistoryPath: Path
  )

  private final case class LegResult(
      label: String,
      records: Vector[HandRecord],
      heroNetBbPer100: Double,
      leakFiredCount: Int,
      leakApplicableSpots: Int,
      heroRaiseResponseCount: Int
  )

  private final case class OpponentRun(
      result: OpponentResult,
      combinedRecords: Vector[HandRecord]
  )

  def main(args: Array[String]): Unit =
    val wantsHelp = args.contains("--help") || args.contains("-h")
    run(args) match
      case Right(summary) =>
        println("=== Adaptive Proof Harness ===")
        println(s"runDir: ${summary.runDir.toAbsolutePath.normalize()}")
        println(s"opponents: ${summary.opponents.size}")
        println(s"handsPerOpponent: ${summary.handsPerOpponent}")
        println(s"report: ${summary.reportPath.toAbsolutePath.normalize()}")
      case Left(error) =>
        if wantsHelp then println(error)
        else
          System.err.println(error)
          sys.exit(1)

  def run(args: Array[String]): Either[String, RunSummary] =
    parseConfig(args).flatMap(config => run(config))

  def run(config: Config): Either[String, RunSummary] =
    validateConfig(config).flatMap { validConfig =>
      try Right(runConfig(validConfig))
      catch case err: Exception => Left(s"adaptive proof harness failed: ${err.getMessage}")
    }

  private def runConfig(config: Config): RunSummary =
    val runDir = resolveRunDir(config)
    Files.createDirectories(runDir)

    val tableRanges = TableRanges.defaults(TableFormat.HeadsUp)
    val actionModel = loadActionModel(config)
    val handsButton = config.handsPerOpponent / 2
    val handsBigBlind = config.handsPerOpponent - handsButton

    val opponentRuns = config.opponents.zipWithIndex.map { case (opponent, idx) =>
      val opponentDir = runDir.resolve(opponent.name)
      Files.createDirectories(opponentDir)

      val heroEngine = new RealTimeAdaptiveEngine(
        tableRanges = tableRanges,
        actionModel = actionModel,
        bunchingTrials = config.bunchingTrials,
        defaultEquityTrials = config.equityTrials,
        minEquityTrials = config.minEquityTrials
      )

      val villain = LeakInjectedVillain(
        name = opponent.name,
        leaks = Vector(opponent.leak),
        baselineNoise = opponent.baselineNoise,
        seed = config.seed + idx.toLong * 1000L
      )

      val handBase = idx * config.handsPerOpponent
      val buttonLeg = runLeg(
        heroEngine = heroEngine,
        villain = villain,
        strategy = opponent.strategy,
        budgetMs = config.budgetMs,
        handCount = handsButton,
        handNumberStart = handBase + 1,
        seed = config.seed + idx.toLong * 1000L,
        heroIsButton = true
      )
      val bigBlindLeg = runLeg(
        heroEngine = heroEngine,
        villain = villain,
        strategy = opponent.strategy,
        budgetMs = config.budgetMs,
        handCount = handsBigBlind,
        handNumberStart = handBase + handsButton + 1,
        seed = config.seed + idx.toLong * 1000L + 1L,
        heroIsButton = false
      )

      val buttonHistory = PokerStarsExporter.exportHands(buttonLeg.records, HeroName, opponent.name)
      val bigBlindHistory = PokerStarsExporter.exportHands(bigBlindLeg.records, HeroName, opponent.name)
      val combinedRecords = buttonLeg.records ++ bigBlindLeg.records
      val combinedHistory = PokerStarsExporter.exportHands(combinedRecords, HeroName, opponent.name)

      val legButtonPath = opponentDir.resolve("leg-button.txt")
      val legBigBlindPath = opponentDir.resolve("leg-bigblind.txt")
      val combinedPath = opponentDir.resolve("combined.txt")
      Files.writeString(legButtonPath, buttonHistory)
      Files.writeString(legBigBlindPath, bigBlindHistory)
      Files.writeString(combinedPath, combinedHistory)

      val heroNetBbPer100 =
        if combinedRecords.nonEmpty then
          combinedRecords.map(_.heroNet).sum / combinedRecords.size.toDouble * 100.0
        else 0.0
      val leakFiredCount = buttonLeg.leakFiredCount + bigBlindLeg.leakFiredCount
      val leakApplicableSpots = buttonLeg.leakApplicableSpots + bigBlindLeg.leakApplicableSpots
      val heroRaiseResponseCount = buttonLeg.heroRaiseResponseCount + bigBlindLeg.heroRaiseResponseCount

      OpponentRun(
        result = OpponentResult(
          name = opponent.name,
          role = opponent.role,
          leakId = opponent.leak.id,
          severity = opponent.leak.severity,
          strategy = opponent.strategyLabel,
          outputDir = opponentDir,
          legButtonPath = legButtonPath,
          legBigBlindPath = legBigBlindPath,
          combinedPath = combinedPath,
          heroNetBbPer100 = heroNetBbPer100,
          heroNetBbPer100ByLeg = Map(
            buttonLeg.label -> buttonLeg.heroNetBbPer100,
            bigBlindLeg.label -> bigBlindLeg.heroNetBbPer100
          ),
          leakFiredCount = leakFiredCount,
          leakApplicableSpots = leakApplicableSpots,
          heroRaiseResponseCount = heroRaiseResponseCount
        ),
        combinedRecords = combinedRecords
      )
    }

    val allRecords = opponentRuns.flatMap(_.combinedRecords).toVector
    val combinedHistoryPath = runDir.resolve("combined-history.txt")
    Files.writeString(
      combinedHistoryPath,
      PokerStarsExporter.exportHands(allRecords, HeroName, "Villain")
    )

    val manifestPath = runDir.resolve("manifest.json")
    Files.writeString(manifestPath, renderManifest(config, runDir, opponentRuns.map(_.result).toVector))

    val groundTruthPath = runDir.resolve("ground-truth.json")
    Files.writeString(groundTruthPath, renderGroundTruth(config, opponentRuns.map(_.result).toVector))

    val reportPath = runDir.resolve("report.txt")
    Files.writeString(reportPath, renderReport(config, opponentRuns.map(_.result).toVector))

    RunSummary(
      runDir = runDir,
      seed = config.seed,
      handsPerOpponent = config.handsPerOpponent,
      opponents = opponentRuns.map(_.result).toVector,
      manifestPath = manifestPath,
      groundTruthPath = groundTruthPath,
      reportPath = reportPath,
      combinedHistoryPath = combinedHistoryPath
    )

  private def runLeg(
      heroEngine: RealTimeAdaptiveEngine,
      villain: LeakInjectedVillain,
      strategy: VillainStrategy,
      budgetMs: Long,
      handCount: Int,
      handNumberStart: Int,
      seed: Long,
      heroIsButton: Boolean
  ): LegResult =
    val simulator = new HeadsUpSimulator(
      heroEngine = Some(heroEngine),
      villain = villain,
      seed = seed,
      budgetMs = budgetMs,
      villainStrategy = strategy,
      heroIsButton = heroIsButton
    )
    val records = Vector.newBuilder[HandRecord]
    var leakFiredCount = 0
    var leakApplicableSpots = 0
    var heroRaiseResponseCount = 0
    var handIdx = 0

    while handIdx < handCount do
      val record = simulator.playHand(handNumberStart + handIdx)
      record.heroRaiseResponses.foreach(event => heroEngine.observeVillainResponseToRaise(event.response))
      records += record
      leakFiredCount += record.actions.count(_.leakFired)
      leakApplicableSpots += record.leakApplicableSpots
      heroRaiseResponseCount += record.heroRaiseResponses.length
      handIdx += 1

    val recordVector = records.result()
    val heroNetBbPer100 =
      if recordVector.nonEmpty then recordVector.map(_.heroNet).sum / recordVector.size.toDouble * 100.0
      else 0.0

    LegResult(
      label = if heroIsButton then "button" else "bigBlind",
      records = recordVector,
      heroNetBbPer100 = heroNetBbPer100,
      leakFiredCount = leakFiredCount,
      leakApplicableSpots = leakApplicableSpots,
      heroRaiseResponseCount = heroRaiseResponseCount
    )

  private def resolveRunDir(config: Config): Path =
    val label = config.runLabel.getOrElse(s"run-${RunLabelFormatter.format(Instant.now())}-${config.seed}")
    config.outputDir.resolve(label)

  private def loadActionModel(config: Config): PokerActionModel =
    config.modelDir
      .map(path => PokerActionModelArtifactIO.load(path).model)
      .getOrElse(PokerActionModel.uniform)

  private def renderManifest(
      config: Config,
      runDir: Path,
      opponents: Vector[OpponentResult]
  ): String =
    val json = ujson.Obj(
      "runDir" -> ujson.Str(runDir.toAbsolutePath.normalize().toString),
      "seed" -> ujson.Num(config.seed.toDouble),
      "handsPerOpponent" -> ujson.Num(config.handsPerOpponent.toDouble),
      "legsPerOpponent" -> ujson.Num(2.0),
      "opponentCount" -> ujson.Num(opponents.size.toDouble),
      "opponents" -> ujson.Arr.from(opponents.map { opponent =>
        ujson.Obj(
          "name" -> ujson.Str(opponent.name),
          "role" -> ujson.Str(opponent.role),
          "directory" -> ujson.Str(opponent.outputDir.getFileName.toString),
          "legButtonFile" -> ujson.Str(opponent.legButtonPath.getFileName.toString),
          "legBigBlindFile" -> ujson.Str(opponent.legBigBlindPath.getFileName.toString),
          "combinedFile" -> ujson.Str(opponent.combinedPath.getFileName.toString)
        )
      })
    )
    ujson.write(json, indent = 2)

  private def renderGroundTruth(config: Config, opponents: Vector[OpponentResult]): String =
    val json = ujson.Obj(
      "handsPerOpponent" -> ujson.Num(config.handsPerOpponent.toDouble),
      "legsPerOpponent" -> ujson.Num(2.0),
      "opponents" -> ujson.Arr.from(opponents.map { opponent =>
        ujson.Obj(
          "name" -> ujson.Str(opponent.name),
          "role" -> ujson.Str(opponent.role),
          "leakId" -> ujson.Str(opponent.leakId),
          "severity" -> ujson.Num(opponent.severity),
          "strategy" -> ujson.Str(opponent.strategy),
          "heroNetBbPer100" -> ujson.Num(opponent.heroNetBbPer100),
          "heroNetBbPer100ByLeg" -> ujson.Obj(
            "button" -> ujson.Num(opponent.heroNetBbPer100ByLeg.getOrElse("button", 0.0)),
            "bigBlind" -> ujson.Num(opponent.heroNetBbPer100ByLeg.getOrElse("bigBlind", 0.0))
          ),
          "leakFiredCount" -> ujson.Num(opponent.leakFiredCount.toDouble),
          "leakApplicableSpots" -> ujson.Num(opponent.leakApplicableSpots.toDouble),
          "heroRaiseResponseCount" -> ujson.Num(opponent.heroRaiseResponseCount.toDouble)
        )
      })
    )
    ujson.write(json, indent = 2)

  private def renderReport(config: Config, opponents: Vector[OpponentResult]): String =
    val leakers = opponents.filter(_.role == "Leaker")
    val leakerAverage =
      if leakers.nonEmpty then leakers.map(_.heroNetBbPer100).sum / leakers.size.toDouble
      else 0.0
    val gtoControl = opponents.find(_.role == "Control").map(_.heroNetBbPer100).getOrElse(0.0)
    val sb = new StringBuilder
    sb.append("=== Adaptive Proof Report ===\n")
    sb.append(s"Run seed: ${config.seed}\n")
    sb.append(s"Hands per opponent: ${config.handsPerOpponent}\n")
    sb.append("\n")
    sb.append("Per-opponent results:\n")
    opponents.foreach { opponent =>
      sb.append(
        f"  ${opponent.name}%-22s bb/100: ${formatSigned(opponent.heroNetBbPer100)}   " +
          f"button: ${formatSigned(opponent.heroNetBbPer100ByLeg.getOrElse("button", 0.0))}   " +
          f"bigBlind: ${formatSigned(opponent.heroNetBbPer100ByLeg.getOrElse("bigBlind", 0.0))}\n"
      )
    }
    sb.append("\n")
    sb.append(f"Leaker average bb/100: ${formatSigned(leakerAverage)}\n")
    sb.append(f"GTO control bb/100: ${formatSigned(gtoControl)}\n")
    sb.toString()

  private def formatSigned(value: Double): String =
    f"$value%+.1f"

  private def validateConfig(config: Config): Either[String, Config] =
    if config.handsPerOpponent < 2 then Left("handsPerOpponent must be at least 2")
    else if config.opponents.isEmpty then Left("at least one opponent is required")
    else if config.bunchingTrials <= 0 then Left("bunchingTrials must be positive")
    else if config.equityTrials <= 0 then Left("equityTrials must be positive")
    else if config.minEquityTrials <= 0 then Left("minEquityTrials must be positive")
    else if config.minEquityTrials > config.equityTrials then Left("minEquityTrials must be <= equityTrials")
    else Right(config)

  private def parseConfig(args: Array[String]): Either[String, Config] =
    if args.contains("--help") || args.contains("-h") then Left(usage)
    else
      try
        var config = Config()
        args.foreach { arg =>
          if arg.startsWith("--hands=") then
            config = config.copy(handsPerOpponent = arg.stripPrefix("--hands=").toInt)
          else if arg.startsWith("--output=") then
            config = config.copy(outputDir = Paths.get(arg.stripPrefix("--output=")))
          else if arg.startsWith("--model=") then
            config = config.copy(modelDir = Some(Paths.get(arg.stripPrefix("--model="))))
          else if arg.startsWith("--seed=") then
            config = config.copy(seed = arg.stripPrefix("--seed=").toLong)
          else if arg.startsWith("--bunchingTrials=") then
            config = config.copy(bunchingTrials = arg.stripPrefix("--bunchingTrials=").toInt)
          else if arg.startsWith("--equityTrials=") then
            config = config.copy(equityTrials = arg.stripPrefix("--equityTrials=").toInt)
          else if arg.startsWith("--minEquityTrials=") then
            config = config.copy(minEquityTrials = arg.stripPrefix("--minEquityTrials=").toInt)
          else if arg.startsWith("--budget=") then
            config = config.copy(budgetMs = arg.stripPrefix("--budget=").toLong)
          else if arg.startsWith("--runLabel=") then
            config = config.copy(runLabel = Some(arg.stripPrefix("--runLabel=")))
        }
        Right(config)
      catch
        case err: NumberFormatException => Left(s"invalid adaptive proof harness argument: ${err.getMessage}")

  private val usage =
    """Usage:
      |  runMain sicfun.holdem.validation.AdaptiveProofHarness [--key=value ...]
      |
      |Options:
      |  --hands=<n>              hands per opponent session (default: 500)
      |  --output=<path>          output root directory (default: data/adaptive-proof)
      |  --model=<path>           optional poker action model artifact directory
      |  --seed=<n>               RNG seed (default: 42)
      |  --bunchingTrials=<n>     adaptive engine bunching trials
      |  --equityTrials=<n>       adaptive engine default equity trials
      |  --minEquityTrials=<n>    adaptive engine minimum equity trials
      |  --budget=<millis>        per-decision budget passed to the hero engine
      |  --runLabel=<label>       optional fixed run directory label
      |""".stripMargin
