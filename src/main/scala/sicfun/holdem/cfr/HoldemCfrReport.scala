package sicfun.holdem.cfr
import sicfun.holdem.types.*
import sicfun.holdem.cli.*
import sicfun.holdem.*
import sicfun.holdem.equity.*

import sicfun.core.Card
import sicfun.core.DiscreteDistribution

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths, StandardOpenOption}
import java.time.Instant
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import scala.jdk.CollectionConverters.*

/** Offline CFR report and exploitability tracking CLI entrypoint.
  *
  * Solves a single heads-up Hold'em decision spot using [[HoldemCfrSolver]], computes
  * exploitability diagnostics (root deviation gap, villain deviation gap, local
  * exploitability), and optionally writes:
  *  - '''summary.txt''': human-readable solution summary with policy and action EVs
  *  - '''policy.tsv''': machine-readable policy with per-action probabilities and EVs
  *  - '''exploitability.tsv''': append-only tracking file for monitoring convergence
  *    across runs with different iteration counts or parameters
  *
  * This tool is designed for offline analysis and quality tracking, not real-time use.
  * It exposes all [[HoldemCfrConfig]] parameters via CLI flags.
  *
  * Usage: `runMain sicfun.holdem.cfr.HoldemCfrReport --hero=AcKh --board=Ts9h8d ...`
  */
object HoldemCfrReport:
  /** ISO-8601 UTC formatter for timestamping generated reports. */
  private val IsoFormatter = DateTimeFormatter.ISO_INSTANT.withZone(ZoneOffset.UTC)

  /** Parsed CLI arguments bundling the spot definition and CFR configuration. */
  private final case class CliConfig(
      hero: HoleCards,
      board: Board,
      pot: Double,
      toCall: Double,
      position: Position,
      stackSize: Double,
      villainRange: DiscreteDistribution[HoleCards],
      candidateActions: Vector[PokerAction],
      cfrConfig: HoldemCfrConfig,
      outDir: Option[Path],
      trackFile: Option[Path]
  )

  /** Result of a single-spot CFR report run, bundling the solved strategy
    * with optional output artifact paths.
    */
  final case class RunResult(
      solution: HoldemCfrSolution,
      outDir: Option[Path],
      trackFile: Option[Path]
  )

  /** CLI entry point. Parses arguments, runs the solver, and prints a
    * human-readable summary to stdout. Exits with code 1 on error.
    */
  def main(args: Array[String]): Unit =
    val wantsHelp = args.contains("--help") || args.contains("-h")
    run(args) match
      case Right(result) =>
        val solution = result.solution
        println("=== Holdem CFR Report ===")
        println(s"bestAction: ${renderAction(solution.bestAction)}")
        println(f"expectedValuePlayer0: ${solution.expectedValuePlayer0}%.6f")
        println(f"heroRootBestResponse: ${solution.heroRootBestResponseValue}%.6f")
        println(f"villainBestResponse: ${solution.villainBestResponseValue}%.6f")
        println(f"rootDeviationGap: ${solution.rootDeviationGap}%.6f")
        println(f"villainDeviationGap: ${solution.villainDeviationGap}%.6f")
        println(f"localExploitability: ${solution.localExploitability}%.6f")
        println(s"iterations: ${solution.iterations}")
        println(s"provider: ${solution.provider}")
        println(s"villainSupport: ${solution.villainSupport}")
        println("policy:")
        solution.actionProbabilities.toVector.sortBy(-_._2).foreach { case (action, probability) =>
          println(f"  ${renderAction(action)}%-12s ${probability * 100.0}%6.2f%%")
        }
        result.outDir.foreach(path => println(s"outDir: ${path.toAbsolutePath.normalize()}"))
        result.trackFile.foreach(path => println(s"trackFile: ${path.toAbsolutePath.normalize()}"))
      case Left(error) =>
        if wantsHelp then println(error)
        else
          System.err.println(error)
          sys.exit(1)

  /** Parses CLI arguments and runs the solver, returning the result or an error message.
    * This is the programmatic entry point (no `sys.exit`).
    */
  def run(args: Array[String]): Either[String, RunResult] =
    for
      config <- parseArgs(args)
      result <- runConfig(config)
    yield result

  /** Executes the CFR solve for a parsed CLI config.
    *
    * Steps:
    *  1. Validates hero/board card overlap
    *  2. Constructs a [[GameState]] from the CLI parameters
    *  3. Calls [[HoldemCfrSolver.solve]] to get a full solution with exploitability metrics
    *  4. Optionally writes summary.txt and policy.tsv to `outDir`
    *  5. Optionally appends an exploitability row to a tracking TSV file
    *
    * @return Right(RunResult) on success, Left(errorMessage) on failure
    */
  private def runConfig(config: CliConfig): Either[String, RunResult] =
    try
      require(
        config.hero.asSet.intersect(config.board.asSet).isEmpty,
        "hero cards must not overlap board cards"
      )

      val state = GameState(
        street =
          if config.board.size == 0 then Street.Preflop
          else if config.board.size == 3 then Street.Flop
          else if config.board.size == 4 then Street.Turn
          else Street.River,
        board = config.board,
        pot = config.pot,
        toCall = config.toCall,
        position = config.position,
        stackSize = config.stackSize,
        betHistory = Vector.empty
      )

      val solution = HoldemCfrSolver.solve(
        hero = config.hero,
        state = state,
        villainPosterior = config.villainRange,
        candidateActions = config.candidateActions,
        config = config.cfrConfig
      )

      val summaryLines = buildSummaryLines(config, solution)

      val resolvedOutDir = config.outDir.map { dir =>
        Files.createDirectories(dir)
        val summaryPath = dir.resolve("summary.txt")
        val policyPath = dir.resolve("policy.tsv")
        Files.write(summaryPath, summaryLines.asJava, StandardCharsets.UTF_8)
        writePolicy(policyPath, solution)
        dir
      }

      val resolvedTrackFile =
        config.trackFile.orElse(resolvedOutDir.map(_.resolve("exploitability.tsv"))).map { path =>
          val parent = path.getParent
          if parent != null then Files.createDirectories(parent)
          appendExploitabilityRow(path, config, solution)
          path
        }

      Right(RunResult(solution, resolvedOutDir, resolvedTrackFile))
    catch
      case e: Exception =>
        Left(s"holdem CFR report failed: ${e.getMessage}")

  /** Builds human-readable summary lines for the report output file.
    * Includes spot configuration, exploitability metrics, policy, and per-action EVs.
    */
  private def buildSummaryLines(config: CliConfig, solution: HoldemCfrSolution): Vector[String] =
    Vector(
      "Holdem CFR Report",
      s"generatedAt: ${IsoFormatter.format(Instant.ofEpochMilli(System.currentTimeMillis()))}",
      s"hero: ${config.hero.toToken}",
      s"board: ${renderBoard(config.board)}",
      f"pot: ${config.pot}%.3f",
      f"toCall: ${config.toCall}%.3f",
      s"position: ${config.position}",
      f"stackSize: ${config.stackSize}%.3f",
      s"candidateActions: ${config.candidateActions.map(renderAction).mkString(",")}",
      s"iterations: ${solution.iterations}",
      s"provider: ${solution.provider}",
      s"villainSupport: ${solution.villainSupport}",
      f"expectedValuePlayer0: ${solution.expectedValuePlayer0}%.8f",
      f"heroRootBestResponseValue: ${solution.heroRootBestResponseValue}%.8f",
      f"villainBestResponseValue: ${solution.villainBestResponseValue}%.8f",
      f"rootDeviationGap: ${solution.rootDeviationGap}%.8f",
      f"villainDeviationGap: ${solution.villainDeviationGap}%.8f",
      f"localExploitability: ${solution.localExploitability}%.8f",
      "",
      "policy:",
      solution.actionProbabilities.toVector
        .sortBy(-_._2)
        .map { case (action, probability) =>
          f"${renderAction(action)}%-12s ${probability * 100.0}%7.3f%%"
        }
        .mkString("\n"),
      "",
      "action_evaluations:",
      solution.actionEvaluations
        .sortBy(-_.expectedValue)
        .map(eval => f"${renderAction(eval.action)}%-12s ${eval.expectedValue}% .8f")
        .mkString("\n")
    )

  /** Writes a machine-readable TSV policy file with columns:
    * action, probability, expectedValue, best (1 if this is the argmax action, 0 otherwise).
    */
  private def writePolicy(path: Path, solution: HoldemCfrSolution): Unit =
    val header = "action\tprobability\texpectedValue\tbest"
    val rows = solution.actionEvaluations.map { evaluation =>
      val probability = solution.actionProbabilities.getOrElse(evaluation.action, 0.0)
      val best = if evaluation.action == solution.bestAction then "1" else "0"
      s"${renderAction(evaluation.action)}\t$probability\t${evaluation.expectedValue}\t$best"
    }
    Files.write(path, (header +: rows).asJava, StandardCharsets.UTF_8)

  /** Appends one row to an exploitability tracking TSV file. Creates the file
    * with a header if it does not already exist. This enables longitudinal
    * tracking of solver convergence across runs with different parameters.
    */
  private def appendExploitabilityRow(path: Path, config: CliConfig, solution: HoldemCfrSolution): Unit =
    val header =
      "generatedAtIso\thero\tboard\tpot\ttoCall\tposition\tstackSize\titerations\tvillainSupport\t" +
        "provider\t" +
        "expectedValuePlayer0\theroRootBestResponseValue\tvillainBestResponseValue\trootDeviationGap\t" +
        "villainDeviationGap\tlocalExploitability"
    if !Files.exists(path) then
      Files.write(path, Vector(header).asJava, StandardCharsets.UTF_8)

    val row = Vector(
      IsoFormatter.format(Instant.ofEpochMilli(System.currentTimeMillis())),
      config.hero.toToken,
      renderBoard(config.board),
      config.pot.toString,
      config.toCall.toString,
      config.position.toString,
      config.stackSize.toString,
      solution.iterations.toString,
      solution.villainSupport.toString,
      solution.provider,
      solution.expectedValuePlayer0.toString,
      solution.heroRootBestResponseValue.toString,
      solution.villainBestResponseValue.toString,
      solution.rootDeviationGap.toString,
      solution.villainDeviationGap.toString,
      solution.localExploitability.toString
    ).mkString("\t")
    Files.write(path, Vector(row).asJava, StandardCharsets.UTF_8, StandardOpenOption.APPEND)

  /** Parses CLI arguments into a [[CliConfig]], validating all constraints
    * (non-negative pot, positive stack, valid card tokens, etc.).
    * Returns Left(usage) for --help, Left(error) for invalid inputs.
    */
  private def parseArgs(args: Array[String]): Either[String, CliConfig] =
    if args.contains("--help") || args.contains("-h") then Left(usage)
    else
      for
        options <- CliHelpers.parseOptions(args)
        hero <- CliHelpers.parseHoleCardsOptionEither(options, "hero", "AcKh")
        board <- parseBoardOption(options, "board", "")
        pot <- parseDoubleOption(options, "pot", 20.0)
        _ <- if pot >= 0.0 then Right(()) else Left("--pot must be >= 0")
        toCall <- parseDoubleOption(options, "toCall", 10.0)
        _ <- if toCall >= 0.0 then Right(()) else Left("--toCall must be >= 0")
        position <- CliHelpers.parsePositionOptionEither(options, "position", Position.Button)
        stackSize <- parseDoubleOption(options, "stack", 100.0)
        _ <- if stackSize > 0.0 then Right(()) else Left("--stack must be > 0")
        villainRange <- parseRangeOption(
          options,
          "villainRange",
          "22+,A2s+,K5s+,Q7s+,J7s+,T7s+,97s+,87s,76s,65s,A7o+,K9o+,Q9o+,J9o+,T9o"
        )
        defaultActions = if toCall > 0.0 then "fold,call,raise:20" else "check,raise:20"
        candidateActions <- CliHelpers.parseCandidateActionsOptionEither(options, "candidateActions", defaultActions, deduplicate = true)
        iterations <- CliHelpers.parseIntOptionEither(options, "iterations", 1_500)
        _ <- if iterations > 0 then Right(()) else Left("--iterations must be > 0")
        averagingDelay <- CliHelpers.parseIntOptionEither(options, "averagingDelay", 200)
        _ <- if averagingDelay >= 0 then Right(()) else Left("--averagingDelay must be >= 0")
        maxVillainHands <- CliHelpers.parseIntOptionEither(options, "maxVillainHands", 96)
        _ <- if maxVillainHands > 0 then Right(()) else Left("--maxVillainHands must be > 0")
        equityTrials <- CliHelpers.parseIntOptionEither(options, "equityTrials", 4_000)
        _ <- if equityTrials > 0 then Right(()) else Left("--equityTrials must be > 0")
        cfrPlus <- CliHelpers.parseBooleanOptionEither(options, "cfrPlus", true)
        linearAveraging <- CliHelpers.parseBooleanOptionEither(options, "linearAveraging", true)
        includeVillainReraises <- CliHelpers.parseBooleanOptionEither(options, "includeVillainReraises", true)
        preferNativeBatch <- CliHelpers.parseBooleanOptionEither(options, "preferNativeBatch", true)
        rngSeed <- CliHelpers.parseLongOptionEither(options, "seed", 1L)
        villainReraiseMultipliers <- parseDoubleCsvOption(options, "villainReraiseMultipliers", Vector(2.0))
        outDir <- parseOptionalPathOption(options, "outDir")
        trackFile <- parseOptionalPathOption(options, "trackFile")
      yield
        val cfrConfig = HoldemCfrConfig(
          iterations = iterations,
          cfrPlus = cfrPlus,
          averagingDelay = averagingDelay,
          linearAveraging = linearAveraging,
          maxVillainHands = maxVillainHands,
          equityTrials = equityTrials,
          includeVillainReraises = includeVillainReraises,
          villainReraiseMultipliers = villainReraiseMultipliers,
          preferNativeBatch = preferNativeBatch,
          rngSeed = rngSeed
        )
        CliConfig(
          hero = hero,
          board = board,
          pot = pot,
          toCall = toCall,
          position = position,
          stackSize = stackSize,
          villainRange = villainRange,
          candidateActions = candidateActions,
          cfrConfig = cfrConfig,
          outDir = outDir,
          trackFile = trackFile
        )

  /** Parses a board from CLI options. Accepts space-separated tokens ("Ts 9h 8d"),
    * comma-separated ("Ts,9h,8d"), or concatenated ("Ts9h8d"). Returns Board.empty
    * for empty/none values.
    */
  private def parseBoardOption(
      options: Map[String, String],
      key: String,
      default: String
  ): Either[String, Board] =
    val raw = options.getOrElse(key, default).trim
    if raw.isEmpty || raw.equalsIgnoreCase("none") then Right(Board.empty)
    else
      val normalized = raw.replace(',', ' ').trim
      val tokens =
        if normalized.contains(" ") then
          normalized.split("\\s+").toVector.filter(_.nonEmpty)
        else
          if normalized.length % 2 != 0 then
            return Left(s"--$key invalid board token sequence '$raw'")
          normalized.grouped(2).toVector

      val parsed = tokens.map { token =>
        Card.parse(token).toRight(s"--$key invalid card token '$token'")
      }
      parsed.collectFirst { case Left(err) => err } match
        case Some(err) => Left(err)
        case None =>
          val cards = parsed.collect { case Right(card) => card }
          try Right(Board.from(cards))
          catch
            case e: Exception => Left(s"--$key invalid board: ${e.getMessage}")

  /** Parses a poker hand range expression (e.g., "22+,A2s+,K5s+") into a
    * discrete probability distribution over hole cards, using [[RangeParser]].
    */
  private def parseRangeOption(
      options: Map[String, String],
      key: String,
      default: String
  ): Either[String, DiscreteDistribution[HoleCards]] =
    RangeParser.parse(options.getOrElse(key, default)) match
      case Right(dist) => Right(dist)
      case Left(err) => Left(s"--$key invalid range: $err")

  private def parseDoubleOption(options: Map[String, String], key: String, default: Double): Either[String, Double] =
    CliHelpers.parseDoubleOptionEither(options, key, default)

  private def parseDoubleCsvOption(
      options: Map[String, String],
      key: String,
      default: Vector[Double]
  ): Either[String, Vector[Double]] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) =>
        val tokens = raw.split(",").toVector.map(_.trim).filter(_.nonEmpty)
        if tokens.isEmpty then Left(s"--$key must contain at least one value")
        else
          val values = tokens.map(_.toDoubleOption)
          if values.exists(_.isEmpty) then Left(s"--$key must be comma-separated doubles")
          else
            val parsed = values.collect { case Some(v) => v }
            if parsed.forall(v => v > 1.0 && v.isFinite) then Right(parsed)
            else Left(s"--$key values must be finite and > 1.0")

  private def parseOptionalPathOption(
      options: Map[String, String],
      key: String
  ): Either[String, Option[Path]] =
    Right(options.get(key).map(Paths.get(_)))

  /** Renders a PokerAction to its canonical string form: FOLD, CHECK, CALL, or RAISE:amount. */
  private def renderAction(action: PokerAction): String =
    action match
      case PokerAction.Fold => "FOLD"
      case PokerAction.Check => "CHECK"
      case PokerAction.Call => "CALL"
      case PokerAction.Raise(amount) => f"RAISE:${amount}%.3f"

  private def renderBoard(board: Board): String =
    if board.cards.isEmpty then "[]"
    else board.cards.map(_.toToken).mkString("[", " ", "]")

  private val usage =
    """Usage:
      |  runMain sicfun.holdem.cfr.HoldemCfrReport [--key=value ...]
      |
      |Core:
      |  --hero=<AcKh>                     Default AcKh
      |  --board=<Ts9h8d|Ts 9h 8d|none>    Default none (preflop)
      |  --pot=<double>                    Default 20.0
      |  --toCall=<double>                 Default 10.0
      |  --position=<Position>             Default Button
      |  --stack=<double>                  Default 100.0
      |  --villainRange=<range>            Default broad CO-style range
      |  --candidateActions=<csv>          Default fold,call,raise:20 (or check,raise:20 if toCall=0)
      |
      |CFR:
      |  --iterations=<int>                Default 1500
      |  --averagingDelay=<int>            Default 200
      |  --cfrPlus=<bool>                  Default true
      |  --linearAveraging=<bool>          Default true
      |  --maxVillainHands=<int>           Default 96
      |  --equityTrials=<int>              Default 4000
      |  --includeVillainReraises=<bool>   Default true
      |  --villainReraiseMultipliers=<csv> Default 2.0
      |  --preferNativeBatch=<bool>        Default true
      |  --seed=<long>                     Default 1
      |
      |Output/Tracking:
      |  --outDir=<path>                   Optional directory for summary.txt/policy.tsv
      |  --trackFile=<path>                Optional exploitability TSV append target
      |                                    If omitted and outDir is set, uses outDir/exploitability.tsv
      |""".stripMargin
