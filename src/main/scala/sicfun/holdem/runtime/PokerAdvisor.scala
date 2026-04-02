package sicfun.holdem.runtime
import sicfun.holdem.types.*
import sicfun.holdem.model.*
import sicfun.holdem.engine.*
import sicfun.holdem.equity.*
import sicfun.holdem.cli.*
import sicfun.holdem.history.*

import sicfun.core.Card

import java.nio.file.{Files, Path, Paths}
import scala.util.Random

/** Interactive heads-up poker advisor.
  *
  * Bootstraps an action model (synthetic or from disk), creates a
  * [[RealTimeAdaptiveEngine]], and runs a REPL where a human records
  * game events and requests real-time recommendations.
  *
  * Usage:
  *   sbt "runMain sicfun.holdem.runtime.PokerAdvisor [--key=value ...]"
  *
  * Options:
  *   --model=<dir>          Path to a saved model artifact directory (optional; bootstraps synthetic model if absent)
  *   --stack=200            Starting stack per player
  *   --sb=1                 Small blind
  *   --bb=2                 Big blind
  *   --seed=42              RNG seed
  *   --bunchingTrials=400   Bunching Monte Carlo trials
  *   --equityTrials=4000    Equity Monte Carlo trials
  *   --budgetMs=2000        Decision budget in milliseconds
  */
object PokerAdvisor:

  def main(args: Array[String]): Unit =
    parseArgs(args).flatMap(bootstrap) match
      case Left(err) =>
        System.err.println(err)
        sys.exit(1)
      case Right(session) =>
        new ReplRunner(session).run()

  private final case class CliConfig(
      modelDir: Option[Path],
      stack: Double,
      sb: Double,
      bb: Double,
      seed: Long,
      bunchingTrials: Int,
      equityTrials: Int,
      budgetMs: Long,
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

  private def bootstrap(config: CliConfig): Either[String, AdvisorSession] =
    try
      System.err.print("Bootstrapping action model... ")
      val tableRanges = TableRanges.defaults(TableFormat.HeadsUp)
      val artifact = loadArtifact(config)
      val engine = buildEngine(config, artifact, tableRanges)
      val opponentMemoryStore = loadRememberedOpponentStore(config)
      val rememberedOpponent = loadRememberedOpponent(opponentMemoryStore, config)
      rememberedOpponent.foreach(profile => engine.seedArchetypePosterior(profile.archetypePosterior))
      val session = buildSession(config, engine, tableRanges, rememberedOpponent, opponentMemoryStore)
      System.err.println("done.")
      printBootstrapBanner(config)
      rememberedOpponent.foreach { profile =>
        println(s"Loaded memory for ${profile.site}/${profile.playerName}: ${profile.handsObserved} hands")
      }
      Right(session)
    catch
      case e: Exception =>
        Left(s"Bootstrap failed: ${e.getMessage}")

  private def loadArtifact(config: CliConfig): TrainedPokerActionModel =
    config.modelDir match
      case Some(dir) =>
        PokerActionModelArtifactIO.load(dir)
      case None =>
        PokerActionModel.trainVersioned(
          trainingData = syntheticTrainingData(),
          learningRate = 0.1,
          iterations = 500,
          l2Lambda = 0.001,
          validationFraction = 0.25,
          splitSeed = config.seed,
          maxMeanBrierScore = 2.0,
          failOnGate = false,
          modelId = "advisor-bootstrap",
          schemaVersion = "poker-action-model-v1",
          source = "poker-advisor-bootstrap",
          trainedAtEpochMillis = System.currentTimeMillis()
        )

  private def buildEngine(
      config: CliConfig,
      artifact: TrainedPokerActionModel,
      tableRanges: TableRanges
  ): RealTimeAdaptiveEngine =
    new RealTimeAdaptiveEngine(
      tableRanges = tableRanges,
      actionModel = artifact.model,
      bunchingTrials = config.bunchingTrials,
      defaultEquityTrials = config.equityTrials,
      minEquityTrials = math.max(200, config.equityTrials / 10),
      equilibriumBaselineConfig = equilibriumBaselineConfig(config)
    )

  private def equilibriumBaselineConfig(
      config: CliConfig
  ): Option[EquilibriumBaselineConfig] =
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

  private def buildSession(
      config: CliConfig,
      engine: RealTimeAdaptiveEngine,
      tableRanges: TableRanges,
      rememberedOpponent: Option[OpponentProfile],
      opponentMemoryStore: Option[OpponentProfileStore]
  ): AdvisorSession =
    new AdvisorSession(
      config = SessionConfig(
        startingStack = config.stack,
        smallBlind = config.sb,
        bigBlind = config.bb,
        decisionBudgetMillis = config.budgetMs
      ),
      engine = engine,
      tableRanges = tableRanges,
      hand = None,
      stats = AdvisorSessionStats(),
      rng = new Random(config.seed),
      rememberedOpponent = rememberedOpponent,
      rememberedVillainObservations = rememberedOpponent.map(_.recentObservations).getOrElse(Vector.empty),
      opponentMemoryTarget = config.opponentStore,
      opponentMemorySite = config.opponentSite,
      opponentMemoryName = config.opponentName,
      opponentMemoryStore = opponentMemoryStore
    )

  private def loadRememberedOpponent(
      store: Option[OpponentProfileStore],
      config: CliConfig
  ): Option[OpponentProfile] =
    (store, config.opponentSite, config.opponentName) match
      case (Some(profileStore), Some(site), Some(name)) =>
        profileStore.find(site, name)
      case _ => None

  private def loadRememberedOpponentStore(
      config: CliConfig
  ): Option[OpponentProfileStore] =
    config.opponentStore.map(OpponentProfileStorePersistence.load)

  private def printBootstrapBanner(config: CliConfig): Unit =
    println(s"sicfun poker advisor")
    println(f"  Heads-up | Stack: ${config.stack}%.0f | Blinds: ${config.sb}%.1f/${config.bb}%.1f | Seed: ${config.seed}")

  private final class ReplRunner(initialSession: AdvisorSession):
    private var session = initialSession
    private var running = true

    def run(): Unit =
      printWelcome()
      while running do
        readCommand() match
          case None => running = false
          case Some(command) => handleCommand(command)

    private def printWelcome(): Unit =
      println("Type 'help' for commands, 'new' to start a hand, 'q' to quit.")
      println()

    private def readCommand(): Option[AdvisorCommand] =
      print("> ")
      Option(scala.io.StdIn.readLine()).map(AdvisorCommandParser.parse)

    private def handleCommand(command: AdvisorCommand): Unit =
      command match
        case AdvisorCommand.Quit =>
          printExitSummary()
          println("Goodbye.")
          running = false
        case _ =>
          val result = session.execute(command)
          session = result.session
          result.output.foreach(println)

    private def printExitSummary(): Unit =
      val finalStats = session.stats
      if finalStats.handsPlayed > 0 then
        println(s"Session: ${finalStats.handsPlayed} hands, ${fmtSigned(finalStats.heroNetChips)} chips.")

  // ---- Synthetic training data (same pattern as LiveHandSimulator) ----

  private def syntheticTrainingData(): Seq[(GameState, HoleCards, PokerAction)] =
    val baseState = GameState(
      street = Street.Flop,
      board = Board.from(Seq(card("Ts"), card("9h"), card("8d"))),
      pot = 20.0,
      toCall = 10.0,
      position = Position.BigBlind,
      stackSize = 180.0,
      betHistory = Vector.empty
    )
    val checkState = baseState.copy(toCall = 0.0)

    val strong = HoleCards.from(Vector(card("Ah"), card("Ad")))
    val medium = HoleCards.from(Vector(card("Qc"), card("Jc")))
    val weak = HoleCards.from(Vector(card("7c"), card("2d")))

    Vector.fill(24)((baseState, strong, PokerAction.Raise(25.0))) ++
      Vector.fill(24)((baseState, medium, PokerAction.Call)) ++
      Vector.fill(24)((baseState, weak, PokerAction.Fold)) ++
      Vector.fill(12)((checkState, medium, PokerAction.Check))

  private def card(token: String): Card =
    Card.parse(token).getOrElse(
      throw new IllegalArgumentException(s"invalid card token: $token")
    )

  // ---- CLI arg parsing (same pattern as LiveHandSimulator) ----

  private def parseArgs(args: Array[String]): Either[String, CliConfig] =
    if args.contains("--help") || args.contains("-h") then Left(usage)
    else
      for
        options <- CliHelpers.parseOptions(args)
        modelDir <- parseOptionalDirectory(options, "model")
        stack <- CliHelpers.parseDoubleOptionEither(options, "stack", 200.0)
        _ <- if stack > 0.0 then Right(()) else Left("--stack must be > 0")
        sb <- CliHelpers.parseDoubleOptionEither(options, "sb", 1.0)
        _ <- if sb > 0.0 then Right(()) else Left("--sb must be > 0")
        bb <- CliHelpers.parseDoubleOptionEither(options, "bb", 2.0)
        _ <- if bb > 0.0 then Right(()) else Left("--bb must be > 0")
        seed <- CliHelpers.parseLongOptionEither(options, "seed", 42L)
        bunchingTrials <- CliHelpers.parseIntOptionEither(options, "bunchingTrials", 400)
        _ <- if bunchingTrials > 0 then Right(()) else Left("--bunchingTrials must be > 0")
        equityTrials <- CliHelpers.parseIntOptionEither(options, "equityTrials", 4000)
        _ <- if equityTrials > 0 then Right(()) else Left("--equityTrials must be > 0")
        budgetMs <- CliHelpers.parseLongOptionEither(options, "budgetMs", 2000L)
        _ <- if budgetMs > 0L then Right(()) else Left("--budgetMs must be > 0")
        cfrIterations <- CliHelpers.parseIntOptionEither(options, "cfrIterations", 0)
        _ <- if cfrIterations >= 0 then Right(()) else Left("--cfrIterations must be >= 0")
        cfrBlend <- CliHelpers.parseDoubleOptionEither(options, "cfrBlend", 0.35)
        _ <- if cfrBlend >= 0.0 && cfrBlend <= 1.0 then Right(()) else Left("--cfrBlend must be in [0,1]")
        cfrMaxLocalExploitability <- CliHelpers.parseDoubleOptionEither(options, "cfrMaxLocalExploitability", -1.0)
        _ <- if cfrMaxLocalExploitability >= 0.0 || cfrMaxLocalExploitability == -1.0 then Right(())
          else Left("--cfrMaxLocalExploitability must be >= 0 or -1 to disable")
        cfrMaxBaselineActionRegret <- CliHelpers.parseDoubleOptionEither(options, "cfrMaxBaselineActionRegret", -1.0)
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
        modelDir = modelDir,
        stack = stack,
        sb = sb,
        bb = bb,
        seed = seed,
        bunchingTrials = bunchingTrials,
        equityTrials = equityTrials,
        budgetMs = budgetMs,
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

  private def parseOptionalDirectory(options: Map[String, String], key: String): Either[String, Option[Path]] =
    options.get(key) match
      case None => Right(None)
      case Some(raw) =>
        val path = Paths.get(raw)
        if Files.isDirectory(path) then Right(Some(path))
        else Left(s"--$key: directory '$raw' does not exist")

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

  private def fmtSigned(v: Double): String =
    if v >= 0 then f"+$v%.1f" else f"$v%.1f"

  private def disabledThreshold(raw: Double): Double =
    if raw < 0.0 then Double.PositiveInfinity else raw

  private val usage =
    """Usage:
      |  runMain sicfun.holdem.runtime.PokerAdvisor [--key=value ...]
      |
      |Options:
      |  --model=<dir>          Path to a saved model artifact directory
      |  --stack=200            Starting stack per player
      |  --sb=1                 Small blind
      |  --bb=2                 Big blind
      |  --seed=42              RNG seed
      |  --bunchingTrials=400   Bunching Monte Carlo trials
      |  --equityTrials=4000    Equity Monte Carlo trials
      |  --budgetMs=2000        Decision budget in milliseconds
      |  --cfrIterations=0      CFR baseline iterations (0 disables equilibrium baseline)
      |  --cfrBlend=0.35        Blend weight for CFR EV in final ranking (0..1)
      |  --cfrMaxLocalExploitability=-1  Max local exploitability allowed to trust CFR (-1 disables)
      |  --cfrMaxBaselineActionRegret=-1 Max EV loss vs CFR-best action before guardrail clamps (-1 disables)
      |  --cfrVillainHands=96   Max villain posterior hands kept for CFR
      |  --cfrEquityTrials=4000 Equity trials used inside CFR terminal evaluation
      |  --cfrVillainReraises=true  Allow villain 3-bet branch in CFR abstraction
      |  --opponentStore=<path|jdbc:postgresql://...> Persisted opponent memory store
      |  --opponentStoreUser=<user> Optional PostgreSQL user
      |  --opponentStorePassword=<password> Optional PostgreSQL password
      |  --opponentStoreSchema=<schema> Optional PostgreSQL schema (default: public)
      |  --opponentSite=<id>    Opponent site key (for example pokerstars)
      |  --opponentName=<name>  Opponent screen name to preload
      |""".stripMargin
