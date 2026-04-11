package sicfun.holdem.runtime
import sicfun.holdem.types.*
import sicfun.holdem.analysis.*
import sicfun.holdem.model.*
import sicfun.holdem.engine.*
import sicfun.holdem.equity.*
import sicfun.holdem.io.*
import sicfun.holdem.cli.*

import sicfun.core.Card

import java.nio.file.{Files, Path}
import java.util.Locale
import scala.jdk.CollectionConverters.*
import scala.util.Random

/** Runnable end-to-end simulator that exercises the full SICFUN pipeline:
  *  - ActionModel training (MultinomialLogistic with synthetic data)
  *  - Real-time adaptive range inference and decision via [[RealTimeAdaptiveEngine]]
  *  - HandEngine event ingestion and snapshot persistence
  *  - Signal generation from snapshot + model artifact via [[GenerateSignals]]
  *
  * '''Purpose:'''
  * This is primarily an integration smoke test / demo runner.  It constructs a fixed
  * flop scenario (Ts9h8d board), trains a lightweight action model on synthetic data,
  * seeds the engine with villain observations matching a configurable archetype
  * (Nit/Tag/Lag/CallingStation/Maniac), runs a hero decision, persists the hand
  * state snapshot and model artifact to temp files, generates signals, and reports
  * the result.  Useful for verifying that all components integrate correctly without
  * needing real hand histories or external services.
  *
  * '''Archetype simulation:'''
  * The `styleActionPool` method maps each [[PlayerArchetype]] to a weighted action
  * distribution.  For example, a Nit folds 80% and calls 20%, while a Maniac raises
  * 50%, calls 30%, and folds 20%.  These observations are fed to
  * [[engine.observeVillainResponseToRaise]] to bias the archetype posterior.
  *
  * Usage:
  *   runMain sicfun.holdem.runtime.LiveHandSimulator [--key=value ...]
  */
object LiveHandSimulator:
  private val HeroPlayerId = "hero-btn"
  private val VillainPlayerId = "villain-bb"

  private final case class CliConfig(
      hero: HoleCards,
      seed: Long,
      villainStyle: PlayerArchetype,
      adaptiveObservations: Int,
      bunchingTrials: Int,
      equityTrials: Int,
      decisionBudgetMillis: Option[Long],
      showTopPosterior: Int,
      keepArtifacts: Boolean
  )

  /** Result of a complete simulation run.
    *
    * @param bestAction       The action recommended by the adaptive engine.
    * @param heroEquityMean   Hero's estimated equity (0..1) against the inferred villain range.
    * @param heroEquityStdErr Standard error of the equity estimate (from Monte Carlo sampling).
    * @param archetypeMap     The most likely villain archetype after observation seeding.
    * @param topPosterior     Top N most likely villain hands with probabilities.
    * @param signalCount      Number of signals generated from the hand snapshot.
    * @param artifactRoot     Path to the temp directory with artifacts (None if cleaned up).
    */
  final case class RunResult(
      bestAction: PokerAction,
      heroEquityMean: Double,
      heroEquityStdErr: Double,
      archetypeMap: PlayerArchetype,
      topPosterior: Vector[(HoleCards, Double)],
      signalCount: Int,
      artifactRoot: Option[Path]
  )

  def main(args: Array[String]): Unit =
    run(args) match
      case Right(result) =>
        println("=== Live Hand Simulator ===")
        println(s"bestAction: ${renderAction(result.bestAction)}")
        println(s"heroEquityMean: ${fmt(result.heroEquityMean, 5)}")
        println(s"heroEquityStdErr: ${fmt(result.heroEquityStdErr, 6)}")
        println(s"archetypeMap: ${result.archetypeMap}")
        println(s"signalCount: ${result.signalCount}")
        result.artifactRoot match
          case Some(path) => println(s"artifactRoot: ${path.toAbsolutePath.normalize()}")
          case None => println("artifactRoot: <deleted>")
        println("topPosterior:")
        result.topPosterior.foreach { case (hand, prob) =>
          println(s"  ${hand.toToken} -> ${fmt(prob, 6)}")
        }
      case Left(error) =>
        System.err.println(error)
        sys.exit(1)

  def run(args: Array[String]): Either[String, RunResult] =
    for
      config <- parseArgs(args)
      result <- runConfig(config)
    yield result

  /** Execute the full simulation pipeline with the given configuration.
    *
    * Steps:
    * 1. Build a fixed flop board (Ts9h8d) and two game states (villain observation + hero decision).
    * 2. Train an action model on synthetic data (strong/medium/weak hands mapped to raise/call/fold).
    * 3. Create a [[RealTimeAdaptiveEngine]] with the trained model.
    * 4. Seed the engine with `adaptiveObservations` villain response actions drawn from the
    *    configured archetype's action pool to bias the Bayesian archetype posterior.
    * 5. Run the engine's `decide` method to get the hero recommendation.
    * 6. Build a simulated [[HandState]] snapshot and persist it + the model artifact to temp files.
    * 7. Run signal generation and read back the signal audit log.
    * 8. Return the [[RunResult]] with decision, equity, archetype, posterior, and signal count.
    */
  private def runConfig(config: CliConfig): Either[String, RunResult] =
    try
      val rng = new Random(config.seed)
      // Fixed flop board for the simulation scenario.
      val board = Board.from(Seq(card("Ts"), card("9h"), card("8d")))
      require(
        board.cards.forall(cardValue => !config.hero.contains(cardValue)),
        "hero cards must not overlap with fixed board Ts9h8d"
      )

      val villainObservationState = GameState(
        street = Street.Flop,
        board = board,
        pot = 20.0,
        toCall = 10.0,
        position = Position.BigBlind,
        stackSize = 180.0,
        betHistory = Vector.empty
      )
      val heroDecisionState = GameState(
        street = Street.Flop,
        board = board,
        pot = 24.0,
        toCall = 8.0,
        position = Position.Button,
        stackSize = 150.0,
        betHistory = Vector(
          BetAction(player = 0, action = PokerAction.Call),
          BetAction(player = 1, action = PokerAction.Raise(8.0))
        )
      )

      val trainingData = syntheticTrainingData(villainObservationState)
      val artifact = PokerActionModel.trainVersioned(
        trainingData = trainingData,
        learningRate = 0.1,
        iterations = 500,
        l2Lambda = 0.001,
        validationFraction = 0.25,
        splitSeed = config.seed,
        maxMeanBrierScore = 2.0,
        failOnGate = false,
        modelId = "live-simulator-model",
        schemaVersion = "poker-action-model-v1",
        source = "live-hand-simulator",
        trainedAtEpochMillis = 1_700_000_000_000L + config.seed
      )

      val tableRanges = TableRanges.defaults(TableFormat.NineMax)
      val engine = new RealTimeAdaptiveEngine(
        tableRanges = tableRanges,
        actionModel = artifact.model,
        bunchingTrials = config.bunchingTrials,
        defaultEquityTrials = config.equityTrials,
        minEquityTrials = math.max(200, config.equityTrials / 10)
      )

      val adaptationActions = styleActionPool(config.villainStyle)
      var i = 0
      while i < config.adaptiveObservations do
        val action = adaptationActions(rng.nextInt(adaptationActions.length))
        engine.observeVillainResponseToRaise(action)
        i += 1

      val folds = TableFormat.NineMax.foldsBeforeOpener(Position.Cutoff).map(PreflopFold(_))
      val observations = Seq(
        VillainObservation(PokerAction.Raise(25.0), villainObservationState)
      )
      val candidateActions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(20.0))
      val decision = engine.decide(
        hero = config.hero,
        state = heroDecisionState,
        folds = folds,
        villainPos = Position.BigBlind,
        observations = observations,
        candidateActions = candidateActions,
        decisionBudgetMillis = config.decisionBudgetMillis,
        rng = new Random(rng.nextLong())
      )

      val handState = simulatedHandState(
        board = board,
        heroAction = decision.decision.recommendation.bestAction
      )

      val tempRoot = Files.createTempDirectory("sicfun-live-hand-sim-")
      val modelDir = tempRoot.resolve("model-artifact")
      val snapshotDir = tempRoot.resolve("snapshot")
      val signalLog = tempRoot.resolve("signals.tsv")

      PokerActionModelArtifactIO.save(modelDir, artifact)
      HandStateSnapshotIO.save(snapshotDir, handState)
      val signalResult = GenerateSignals.run(
        Array(
          snapshotDir.toString,
          modelDir.toString,
          signalLog.toString,
          "--generatedAtEpochMillis=1800000000000",
          "--warningThreshold=0.4",
          "--criticalThreshold=0.7"
        )
      )
      val signals = SignalAuditLogIO.read(signalLog)

      val topPosterior = decision.decision.posteriorInference.posterior.weights.toVector
        .sortBy { case (_, probability) => -probability }
        .take(config.showTopPosterior)

      val outcome = signalResult match
        case Left(err) =>
          if !config.keepArtifacts then deleteRecursively(tempRoot)
          Left(s"signal generation failed: $err")
        case Right(_) =>
          val artifactPath =
            if config.keepArtifacts then Some(tempRoot)
            else
              deleteRecursively(tempRoot)
              None
          Right(
            RunResult(
              bestAction = decision.decision.recommendation.bestAction,
              heroEquityMean = decision.decision.recommendation.heroEquity.mean,
              heroEquityStdErr = decision.decision.recommendation.heroEquity.stderr,
              archetypeMap = decision.archetypeMap,
              topPosterior = topPosterior,
              signalCount = signals.length,
              artifactRoot = artifactPath
            )
          )
      outcome
    catch
      case e: Exception =>
        Left(s"live hand simulation failed: ${e.getMessage}")

  /** Generate synthetic training data for the action model.
    *
    * Maps hand strength to expected actions: strong hands (AA) -> Raise,
    * medium hands (QcJc) -> Call, weak hands (7c2d) -> Fold.
    * Also includes Check samples for zero-toCall states.
    * Each category is replicated 24x (12x for Check) to give the model enough samples.
    */
  private def syntheticTrainingData(
      state: GameState
  ): Seq[(GameState, HoleCards, PokerAction)] =
    val strong = HoleCards.from(Vector(card("Ah"), card("Ad")))
    val medium = HoleCards.from(Vector(card("Qc"), card("Jc")))
    val weak = HoleCards.from(Vector(card("7c"), card("2d")))
    val checkState = state.copy(toCall = 0.0)

    Vector.fill(24)((state, strong, PokerAction.Raise(25.0))) ++
      Vector.fill(24)((state, medium, PokerAction.Call)) ++
      Vector.fill(24)((state, weak, PokerAction.Fold)) ++
      Vector.fill(12)((checkState, medium, PokerAction.Check))

  /** Build a simulated [[HandState]] by constructing synthetic poker events and replaying
    * them through [[HandEngine]].  Creates a two-event sequence: villain raises on the flop,
    * then hero responds with the given action.  Used to test snapshot persistence and signal
    * generation without needing a real hand history.
    */
  private def simulatedHandState(board: Board, heroAction: PokerAction): HandState =
    val handId = "live-sim-hand-001"
    val startedAt = 1_800_000_000_000L
    val events = Seq(
      PokerEvent(
        handId = handId,
        sequenceInHand = 1L,
        playerId = HeroPlayerId,
        occurredAtEpochMillis = startedAt + 1010L,
        street = Street.Flop,
        position = Position.Button,
        board = board,
        potBefore = 28.0,
        toCall = 8.0,
        stackBefore = 150.0,
        action = heroAction,
        decisionTimeMillis = Some(420L),
        betHistory = Vector(BetAction(0, PokerAction.Call), BetAction(1, PokerAction.Raise(8.0)))
      ),
      PokerEvent(
        handId = handId,
        sequenceInHand = 0L,
        playerId = VillainPlayerId,
        occurredAtEpochMillis = startedAt + 1000L,
        street = Street.Flop,
        position = Position.BigBlind,
        board = board,
        potBefore = 20.0,
        toCall = 0.0,
        stackBefore = 180.0,
        action = PokerAction.Raise(8.0),
        decisionTimeMillis = Some(310L),
        betHistory = Vector(BetAction(0, PokerAction.Call))
      )
    )
    HandEngine.applyEvents(HandEngine.newHand(handId, startedAt), events)

  private def parseArgs(args: Array[String]): Either[String, CliConfig] =
    if args.contains("--help") || args.contains("-h") then Left(usage)
    else
      for
        options <- CliHelpers.parseOptions(args)
        hero <- parseHoleOption(options.getOrElse("hero", "AcKh"))
        seed <- CliHelpers.parseLongOptionEither(options, "seed", 42L)
        villainStyle <- parseArchetypeOption(options, "villainStyle", PlayerArchetype.Maniac)
        adaptiveObservations <- CliHelpers.parseIntOptionEither(options, "adaptiveObservations", 24)
        _ <- if adaptiveObservations > 0 then Right(()) else Left("--adaptiveObservations must be > 0")
        bunchingTrials <- CliHelpers.parseIntOptionEither(options, "bunchingTrials", 400)
        _ <- if bunchingTrials > 0 then Right(()) else Left("--bunchingTrials must be > 0")
        equityTrials <- CliHelpers.parseIntOptionEither(options, "equityTrials", 4000)
        _ <- if equityTrials > 0 then Right(()) else Left("--equityTrials must be > 0")
        decisionBudget <- CliHelpers.parseOptionalLongOptionEither(options, "decisionBudgetMillis")
        _ <- decisionBudget match
          case Some(value) if value <= 0L => Left("--decisionBudgetMillis must be > 0 when provided")
          case _ => Right(())
        showTopPosterior <- CliHelpers.parseIntOptionEither(options, "showTopPosterior", 5)
        _ <- if showTopPosterior > 0 then Right(()) else Left("--showTopPosterior must be > 0")
        keepArtifacts <- CliHelpers.parseStrictBooleanOptionEither(options, "keepArtifacts", false)
      yield CliConfig(
        hero = hero,
        seed = seed,
        villainStyle = villainStyle,
        adaptiveObservations = adaptiveObservations,
        bunchingTrials = bunchingTrials,
        equityTrials = equityTrials,
        decisionBudgetMillis = decisionBudget,
        showTopPosterior = showTopPosterior,
        keepArtifacts = keepArtifacts
      )

  private def parseHoleOption(value: String): Either[String, HoleCards] =
    try Right(CliHelpers.parseHoleCards(value))
    catch case e: Exception => Left(s"--hero invalid: ${e.getMessage}")

  private def parseArchetypeOption(
      options: Map[String, String],
      key: String,
      default: PlayerArchetype
  ): Either[String, PlayerArchetype] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) =>
        val normalized = raw.trim.toLowerCase
        normalized match
          case "nit" => Right(PlayerArchetype.Nit)
          case "tag" => Right(PlayerArchetype.Tag)
          case "lag" => Right(PlayerArchetype.Lag)
          case "callingstation" => Right(PlayerArchetype.CallingStation)
          case "station" => Right(PlayerArchetype.CallingStation)
          case "maniac" => Right(PlayerArchetype.Maniac)
          case _ =>
            Left("--villainStyle must be one of: nit, tag, lag, callingstation, station, maniac")

  /** Map a villain archetype to a weighted action pool for seeding observations.
    *
    * Each archetype has a characteristic action distribution (expressed as repeated
    * elements in a vector, so random selection produces the right frequencies):
    *  - Nit: 80% fold, 20% call (extremely tight, never raises).
    *  - Tag: 50% fold, 40% call, 10% raise (tight-aggressive, selective raises).
    *  - Lag: 30% fold, 50% call, 20% raise (loose-aggressive, frequent raises).
    *  - CallingStation: 20% fold, 70% call, 10% raise (calls too often, rarely raises).
    *  - Maniac: 20% fold, 30% call, 50% raise (hyper-aggressive, raise-heavy).
    */
  private def styleActionPool(archetype: PlayerArchetype): Vector[PokerAction] =
    archetype match
      case PlayerArchetype.Nit =>
        Vector.fill(8)(PokerAction.Fold) ++ Vector.fill(2)(PokerAction.Call)
      case PlayerArchetype.Tag =>
        Vector.fill(5)(PokerAction.Fold) ++ Vector.fill(4)(PokerAction.Call) ++ Vector.fill(1)(PokerAction.Raise(9.0))
      case PlayerArchetype.Lag =>
        Vector.fill(3)(PokerAction.Fold) ++ Vector.fill(5)(PokerAction.Call) ++ Vector.fill(2)(PokerAction.Raise(10.0))
      case PlayerArchetype.CallingStation =>
        Vector.fill(2)(PokerAction.Fold) ++ Vector.fill(7)(PokerAction.Call) ++ Vector.fill(1)(PokerAction.Raise(8.0))
      case PlayerArchetype.Maniac =>
        Vector.fill(2)(PokerAction.Fold) ++ Vector.fill(3)(PokerAction.Call) ++ Vector.fill(5)(PokerAction.Raise(12.0))

  private def deleteRecursively(path: Path): Unit =
    if Files.exists(path) then
      val stream = Files.walk(path)
      try
        val all = stream.iterator().asScala.toVector.sortBy(_.toString.length).reverse
        all.foreach(Files.deleteIfExists)
      finally
        stream.close()

  private def renderAction(action: PokerAction): String =
    action match
      case PokerAction.Fold => "Fold"
      case PokerAction.Check => "Check"
      case PokerAction.Call => "Call"
      case PokerAction.Raise(amount) => s"Raise(${fmt(amount, 2)})"

  private def fmt(value: Double, digits: Int): String =
    String.format(Locale.ROOT, s"%.${digits}f", java.lang.Double.valueOf(value))

  private def card(token: String): Card =
    Card.parse(token).getOrElse(
      throw new IllegalArgumentException(s"invalid card token: $token")
    )

  private val usage =
    """Usage:
      |  runMain sicfun.holdem.runtime.LiveHandSimulator [--key=value ...]
      |
      |Options:
      |  --hero=AcKh                       Hero hole cards (4-char canonical token)
      |  --villainStyle=maniac             nit|tag|lag|callingstation|station|maniac
      |  --seed=42                         RNG seed
      |  --adaptiveObservations=24         Number of villain response observations
      |  --bunchingTrials=400              Bunching Monte Carlo trials
      |  --equityTrials=4000               Equity Monte Carlo trials
      |  --decisionBudgetMillis=25         Optional latency budget
      |  --showTopPosterior=5              Number of posterior hands to print
      |  --keepArtifacts=false             Keep generated snapshot/model/signal files
      |""".stripMargin
