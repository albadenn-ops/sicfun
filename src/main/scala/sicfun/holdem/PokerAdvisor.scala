package sicfun.holdem

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
  *   sbt "runMain sicfun.holdem.PokerAdvisor [--key=value ...]"
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
    parseArgs(args) match
      case Left(err) =>
        System.err.println(err)
        sys.exit(1)
      case Right(config) =>
        val session = bootstrap(config) match
          case Left(err) =>
            System.err.println(err)
            sys.exit(1)
          case Right(s) => s

        println("Type 'help' for commands, 'new' to start a hand, 'q' to quit.")
        println()
        repl(session)

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
      cfrVillainHands: Int,
      cfrEquityTrials: Int,
      cfrVillainReraises: Boolean
  )

  private def bootstrap(config: CliConfig): Either[String, AdvisorSession] =
    try
      System.err.print("Bootstrapping action model... ")
      val rng = new Random(config.seed)

      val artifact = config.modelDir match
        case Some(dir) =>
          PokerActionModelArtifactIO.load(dir)
        case None =>
          val trainingData = syntheticTrainingData()
          PokerActionModel.trainVersioned(
            trainingData = trainingData,
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

      // Use NineMax format for table ranges: even in HU play, bunching effect
      // requires non-empty fold positions. NineMax Button/BB ranges are wide enough
      // for HU contexts, and the bunching dead-card adjustment is meaningful.
      val tableRanges = TableRanges.defaults(TableFormat.NineMax)
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
      val engine = new RealTimeAdaptiveEngine(
        tableRanges = tableRanges,
        actionModel = artifact.model,
        bunchingTrials = config.bunchingTrials,
        defaultEquityTrials = config.equityTrials,
        minEquityTrials = math.max(200, config.equityTrials / 10),
        equilibriumBaselineConfig = equilibriumBaselineConfig
      )

      System.err.println("done.")

      val sessionConfig = SessionConfig(
        startingStack = config.stack,
        smallBlind = config.sb,
        bigBlind = config.bb,
        decisionBudgetMillis = config.budgetMs
      )

      val session = new AdvisorSession(
        config = sessionConfig,
        engine = engine,
        tableRanges = tableRanges,
        hand = None,
        stats = AdvisorSessionStats(),
        rng = rng
      )

      println(s"sicfun poker advisor")
      println(f"  Heads-up | Stack: ${config.stack}%.0f | Blinds: ${config.sb}%.1f/${config.bb}%.1f | Seed: ${config.seed}")
      Right(session)

    catch
      case e: Exception =>
        Left(s"Bootstrap failed: ${e.getMessage}")

  private def repl(initial: AdvisorSession): Unit =
    var session = initial
    var running = true
    while running do
      print("> ")
      val line = scala.io.StdIn.readLine()
      if line == null then running = false
      else
        val command = AdvisorCommandParser.parse(line)
        command match
          case AdvisorCommand.Quit =>
            val finalStats = session.stats
            if finalStats.handsPlayed > 0 then
              println(s"Session: ${finalStats.handsPlayed} hands, ${fmtSigned(finalStats.heroNetChips)} chips.")
            println("Goodbye.")
            running = false
          case _ =>
            val result = session.execute(command)
            session = result.session
            result.output.foreach(println)

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
        options <- parseOptions(args)
        modelDir <- parseOptionalPath(options, "model")
        stack <- parseDoubleOption(options, "stack", 200.0)
        _ <- if stack > 0.0 then Right(()) else Left("--stack must be > 0")
        sb <- parseDoubleOption(options, "sb", 1.0)
        _ <- if sb > 0.0 then Right(()) else Left("--sb must be > 0")
        bb <- parseDoubleOption(options, "bb", 2.0)
        _ <- if bb > 0.0 then Right(()) else Left("--bb must be > 0")
        seed <- parseLongOption(options, "seed", 42L)
        bunchingTrials <- parseIntOption(options, "bunchingTrials", 400)
        _ <- if bunchingTrials > 0 then Right(()) else Left("--bunchingTrials must be > 0")
        equityTrials <- parseIntOption(options, "equityTrials", 4000)
        _ <- if equityTrials > 0 then Right(()) else Left("--equityTrials must be > 0")
        budgetMs <- parseLongOption(options, "budgetMs", 2000L)
        _ <- if budgetMs > 0L then Right(()) else Left("--budgetMs must be > 0")
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
      case (Left(err), _)             => Left(err)
      case (Right(_), Left(err))      => Left(err)
      case (Right(acc), Right(pair))  => Right(acc + pair)
    }

  private def parseIntOption(options: Map[String, String], key: String, default: Int): Either[String, Int] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) => raw.toIntOption.toRight(s"--$key must be an integer")

  private def parseLongOption(options: Map[String, String], key: String, default: Long): Either[String, Long] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) => raw.toLongOption.toRight(s"--$key must be a long")

  private def parseDoubleOption(options: Map[String, String], key: String, default: Double): Either[String, Double] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) => raw.toDoubleOption.toRight(s"--$key must be a number")

  private def parseOptionalPath(options: Map[String, String], key: String): Either[String, Option[Path]] =
    options.get(key) match
      case None => Right(None)
      case Some(raw) =>
        val path = Paths.get(raw)
        if Files.isDirectory(path) then Right(Some(path))
        else Left(s"--$key: directory '$raw' does not exist")

  private def parseBooleanOption(options: Map[String, String], key: String, default: Boolean): Either[String, Boolean] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) =>
        raw.trim.toLowerCase match
          case "true" | "1" | "yes"  => Right(true)
          case "false" | "0" | "no"  => Right(false)
          case _ => Left(s"--$key must be a boolean (true/false)")

  private def fmtSigned(v: Double): String =
    if v >= 0 then f"+$v%.1f" else f"$v%.1f"

  private val usage =
    """Usage:
      |  runMain sicfun.holdem.PokerAdvisor [--key=value ...]
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
      |  --cfrVillainHands=96   Max villain posterior hands kept for CFR
      |  --cfrEquityTrials=4000 Equity trials used inside CFR terminal evaluation
      |  --cfrVillainReraises=true  Allow villain 3-bet branch in CFR abstraction
      |""".stripMargin
