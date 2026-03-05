package sicfun.holdem

import sicfun.core.Card
import sicfun.core.DiscreteDistribution

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths, StandardOpenOption}
import java.time.Instant
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import scala.jdk.CollectionConverters.*

/** Offline CFR report/tracking entrypoint.
  *
  * Solves a one-street heads-up abstraction and writes:
  *  - summary.txt
  *  - policy.tsv
  *  - exploitability.tsv (append-only tracking)
  */
object HoldemCfrReport:
  private val IsoFormatter = DateTimeFormatter.ISO_INSTANT.withZone(ZoneOffset.UTC)

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

  final case class RunResult(
      solution: HoldemCfrSolution,
      outDir: Option[Path],
      trackFile: Option[Path]
  )

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

  def run(args: Array[String]): Either[String, RunResult] =
    for
      config <- parseArgs(args)
      result <- runConfig(config)
    yield result

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

  private def writePolicy(path: Path, solution: HoldemCfrSolution): Unit =
    val header = "action\tprobability\texpectedValue\tbest"
    val rows = solution.actionEvaluations.map { evaluation =>
      val probability = solution.actionProbabilities.getOrElse(evaluation.action, 0.0)
      val best = if evaluation.action == solution.bestAction then "1" else "0"
      s"${renderAction(evaluation.action)}\t$probability\t${evaluation.expectedValue}\t$best"
    }
    Files.write(path, (header +: rows).asJava, StandardCharsets.UTF_8)

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

  private def parseArgs(args: Array[String]): Either[String, CliConfig] =
    if args.contains("--help") || args.contains("-h") then Left(usage)
    else
      for
        options <- parseOptions(args)
        hero <- parseHoleCardsOption(options, "hero", "AcKh")
        board <- parseBoardOption(options, "board", "")
        pot <- parseDoubleOption(options, "pot", 20.0)
        _ <- if pot >= 0.0 then Right(()) else Left("--pot must be >= 0")
        toCall <- parseDoubleOption(options, "toCall", 10.0)
        _ <- if toCall >= 0.0 then Right(()) else Left("--toCall must be >= 0")
        position <- parsePositionOption(options, "position", Position.Button)
        stackSize <- parseDoubleOption(options, "stack", 100.0)
        _ <- if stackSize > 0.0 then Right(()) else Left("--stack must be > 0")
        villainRange <- parseRangeOption(
          options,
          "villainRange",
          "22+,A2s+,K5s+,Q7s+,J7s+,T7s+,97s+,87s,76s,65s,A7o+,K9o+,Q9o+,J9o+,T9o"
        )
        defaultActions = if toCall > 0.0 then "fold,call,raise:20" else "check,raise:20"
        candidateActions <- parseCandidateActionsOption(options, "candidateActions", defaultActions)
        iterations <- parseIntOption(options, "iterations", 1_500)
        _ <- if iterations > 0 then Right(()) else Left("--iterations must be > 0")
        averagingDelay <- parseIntOption(options, "averagingDelay", 200)
        _ <- if averagingDelay >= 0 then Right(()) else Left("--averagingDelay must be >= 0")
        maxVillainHands <- parseIntOption(options, "maxVillainHands", 96)
        _ <- if maxVillainHands > 0 then Right(()) else Left("--maxVillainHands must be > 0")
        equityTrials <- parseIntOption(options, "equityTrials", 4_000)
        _ <- if equityTrials > 0 then Right(()) else Left("--equityTrials must be > 0")
        cfrPlus <- parseBooleanOption(options, "cfrPlus", true)
        linearAveraging <- parseBooleanOption(options, "linearAveraging", true)
        includeVillainReraises <- parseBooleanOption(options, "includeVillainReraises", true)
        preferNativeBatch <- parseBooleanOption(options, "preferNativeBatch", true)
        rngSeed <- parseLongOption(options, "seed", 1L)
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

  private def parseHoleCardsOption(
      options: Map[String, String],
      key: String,
      default: String
  ): Either[String, HoleCards] =
    try
      Right(CliHelpers.parseHoleCards(options.getOrElse(key, default)))
    catch
      case e: Exception => Left(s"--$key invalid: ${e.getMessage}")

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

  private def parseRangeOption(
      options: Map[String, String],
      key: String,
      default: String
  ): Either[String, DiscreteDistribution[HoleCards]] =
    RangeParser.parse(options.getOrElse(key, default)) match
      case Right(dist) => Right(dist)
      case Left(err) => Left(s"--$key invalid range: $err")

  private def parsePositionOption(
      options: Map[String, String],
      key: String,
      default: Position
  ): Either[String, Position] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) =>
        Position.values.find(_.toString.equalsIgnoreCase(raw.trim))
          .toRight(s"--$key invalid position: $raw")

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
        case None =>
          val actions = parsed.collect { case Right(action) => action }.distinct
          if actions.nonEmpty then Right(actions)
          else Left(s"--$key must contain at least one valid action")

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

  private def parseBooleanOption(
      options: Map[String, String],
      key: String,
      default: Boolean
  ): Either[String, Boolean] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) =>
        raw.trim.toLowerCase match
          case "true" | "1" | "yes" => Right(true)
          case "false" | "0" | "no" => Right(false)
          case _ => Left(s"--$key must be a boolean (true/false)")

  private def parseOptionalPathOption(
      options: Map[String, String],
      key: String
  ): Either[String, Option[Path]] =
    Right(options.get(key).map(Paths.get(_)))

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
      |  runMain sicfun.holdem.HoldemCfrReport [--key=value ...]
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
