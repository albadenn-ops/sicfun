package sicfun.holdem.runtime

import sicfun.holdem.cli.CliHelpers
import sicfun.holdem.engine.GtoSolveEngine.GtoMode
import sicfun.holdem.engine.villain.PlayerArchetype
import sicfun.holdem.runtime.HallVillain.{VillainMode, VillainProfile}
import sicfun.holdem.types.{HeroMode, Position}

import java.nio.file.{Path, Paths}

/** CLI configuration types and parsing for the playing hall.
  *
  * Third F6 split slice (after [[HallFormat]] and [[HallVillain]]). Pulls every
  * `--key=value` parsing concern out of [[TexasHoldemPlayingHall]] into a
  * sibling object so the monolith stops carrying both the simulation runner
  * and its CLI scaffold:
  *
  *   - [[Config]]: fully-validated record produced by [[parseArgs]].
  *   - [[parseArgs]]: top-level CLI entry, returns `Either[String, Config]`.
  *   - typed-option helpers ([[intOpt]], [[longOpt]], [[doubleOpt]],
  *     [[boolOpt]], [[pathOpt]], [[optionalPathOpt]]).
  *   - enum option helpers ([[heroModeOpt]], [[gtoModeOpt]]).
  *   - position resolution ([[modeledPositionsForPlayerCount]],
  *     [[parseLegacyHeroSeat]], [[resolveHeroPosition]]).
  *   - the [[usage]] string.
  *
  * Visibility is `private[runtime]` so the runtime package retains the same
  * access it had pre-split; nothing outside `sicfun.holdem.runtime` ever
  * needed these types.
  */
private[runtime] object HallConfig:
  /** All CLI-parsed configuration for a playing hall run. Controls hand count,
    * table geometry, hero/villain modes, learning schedule, raise sizing,
    * inference budget, and output paths.
    */
  final case class Config(
      hands: Int,
      tableCount: Int,
      playerCount: Int,
      reportEvery: Int,
      learnEveryHands: Int,
      learningWindowSamples: Int,
      seed: Long,
      outDir: Path,
      modelArtifactDir: Option[Path],
      heroMode: HeroMode,
      heroPosition: Position,
      gtoMode: GtoMode,
      villainPool: Vector[VillainProfile],
      heroExplorationRate: Double,
      raiseSize: Double,
      bunchingTrials: Int,
      equityTrials: Int,
      saveTrainingTsv: Boolean,
      saveDdreTrainingTsv: Boolean,
      saveReviewHandHistory: Boolean,
      fullRing: Boolean
  )

  /** Returns the canonical sequence of modeled positions for an N-max table.
    * For 2-max: Button, BigBlind. For full ring (9-max): UTG through BigBlind.
    */
  def modeledPositionsForPlayerCount(playerCount: Int): Vector[Position] =
    playerCount match
      case 2 => Vector(Position.Button, Position.BigBlind)
      case 3 => Vector(Position.Button, Position.SmallBlind, Position.BigBlind)
      case 4 => Vector(Position.Cutoff, Position.Button, Position.SmallBlind, Position.BigBlind)
      case 5 => Vector(Position.Middle, Position.Cutoff, Position.Button, Position.SmallBlind, Position.BigBlind)
      case 6 => Vector(Position.UTG, Position.Middle, Position.Cutoff, Position.Button, Position.SmallBlind, Position.BigBlind)
      case 7 => Vector(Position.UTG, Position.UTG1, Position.Middle, Position.Cutoff, Position.Button, Position.SmallBlind, Position.BigBlind)
      case 8 => Vector(
        Position.UTG,
        Position.UTG1,
        Position.UTG2,
        Position.Middle,
        Position.Cutoff,
        Position.Button,
        Position.SmallBlind,
        Position.BigBlind
      )
      case 9 => Vector(
        Position.UTG,
        Position.UTG1,
        Position.UTG2,
        Position.Middle,
        Position.Hijack,
        Position.Cutoff,
        Position.Button,
        Position.SmallBlind,
        Position.BigBlind
      )
      case _ => Vector.empty

  /** Legacy heads-up `--heroSeat` alias: only `button` / `bigblind` (or `bb`). */
  def parseLegacyHeroSeat(raw: String): Either[String, Position] =
    raw.trim.toLowerCase match
      case "button" => Right(Position.Button)
      case "bigblind" | "bb" => Right(Position.BigBlind)
      case _ => Left("--heroSeat must be one of: button, bigblind")

  /** Resolves the hero's table position from CLI options, falling back
    * through `--heroPosition`, then `--heroSeat`, then Button. Validates that
    * the chosen position is modeled for the requested table size.
    */
  def resolveHeroPosition(
      options: Map[String, String],
      playerCount: Int
  ): Either[String, Position] =
    val rawPosition =
      options.get("heroPosition") match
        case Some(_) =>
          CliHelpers.parsePositionOptionEither(options, "heroPosition", Position.Button)
        case None =>
          options.get("heroSeat") match
            case Some(raw) => parseLegacyHeroSeat(raw)
            case None => Right(Position.Button)
    rawPosition.flatMap { position =>
      val modeledPositions = modeledPositionsForPlayerCount(playerCount)
      if modeledPositions.contains(position) then Right(position)
      else Left(s"--heroPosition $position is not valid for playerCount=$playerCount")
    }

  /** Parses CLI arguments into a validated [[Config]]. Uses a for-comprehension
    * over Either to chain validation: each parameter is parsed with a typed
    * helper, then range-checked. Returns `Left(errorMessage)` on the first
    * validation failure.
    */
  def parseArgs(args: Array[String]): Either[String, Config] =
    if args.contains("--help") || args.contains("-h") then Left(usage)
    else
      for
        options <- CliHelpers.parseOptions(args)
        hands <- intOpt(options, "hands", 100000)
        _ <- if hands > 0 then Right(()) else Left("--hands must be > 0")
        tableCount <- intOpt(options, "tableCount", 1)
        _ <- if tableCount > 0 then Right(()) else Left("--tableCount must be > 0")
        playerCount <- intOpt(options, "playerCount", 2)
        _ <- if playerCount >= 2 && playerCount <= 9 then Right(()) else Left("--playerCount must be in [2,9]")
        reportEvery <- intOpt(options, "reportEvery", 10000)
        _ <- if reportEvery > 0 then Right(()) else Left("--reportEvery must be > 0")
        learnEveryHands <- intOpt(options, "learnEveryHands", 50000)
        _ <- if learnEveryHands >= 0 then Right(()) else Left("--learnEveryHands must be >= 0")
        learningWindowSamples <- intOpt(options, "learningWindowSamples", 200000)
        _ <- if learningWindowSamples >= 0 then Right(()) else Left("--learningWindowSamples must be >= 0")
        seed <- longOpt(options, "seed", 42L)
        outDir <- pathOpt(options, "outDir", Paths.get("data/playing-hall"))
        modelArtifactDir <- optionalPathOpt(options, "modelArtifactDir")
        heroMode <- heroModeOpt(options, "heroStyle", HeroMode.Adaptive)
        heroPosition <- resolveHeroPosition(options, playerCount)
        gtoMode <- gtoModeOpt(options, "gtoMode", GtoMode.Exact)
        villainMode <- HallVillain.villainModeOpt(options, "villainStyle", VillainMode.Archetype(PlayerArchetype.Tag))
        villainPool <- HallVillain.buildVillainPool(villainMode, options.get("villainPool"))
        heroExplorationRate <- doubleOpt(options, "heroExplorationRate", 0.05)
        _ <- if heroExplorationRate >= 0.0 && heroExplorationRate <= 1.0 then Right(())
        else Left("--heroExplorationRate must be in [0,1]")
        raiseSize <- doubleOpt(options, "raiseSize", 2.5)
        _ <- if raiseSize > 0.0 then Right(()) else Left("--raiseSize must be > 0")
        bunchingTrials <- intOpt(options, "bunchingTrials", 80)
        _ <- if bunchingTrials > 0 then Right(()) else Left("--bunchingTrials must be > 0")
        equityTrials <- intOpt(options, "equityTrials", 700)
        _ <- if equityTrials > 0 then Right(()) else Left("--equityTrials must be > 0")
        saveTrainingTsv <- boolOpt(options, "saveTrainingTsv", true)
        saveDdreTrainingTsv <- boolOpt(options, "saveDdreTrainingTsv", false)
        saveReviewHandHistory <- boolOpt(options, "saveReviewHandHistory", false)
        fullRing <- boolOpt(options, "fullRing", false)
      yield Config(
        hands = hands,
        tableCount = tableCount,
        playerCount = playerCount,
        reportEvery = reportEvery,
        learnEveryHands = learnEveryHands,
        learningWindowSamples = learningWindowSamples,
        seed = seed,
        outDir = outDir,
        modelArtifactDir = modelArtifactDir,
        heroMode = heroMode,
        heroPosition = heroPosition,
        gtoMode = gtoMode,
        villainPool = villainPool,
        heroExplorationRate = heroExplorationRate,
        raiseSize = raiseSize,
        bunchingTrials = bunchingTrials,
        equityTrials = equityTrials,
        saveTrainingTsv = saveTrainingTsv,
        saveDdreTrainingTsv = saveDdreTrainingTsv,
        saveReviewHandHistory = saveReviewHandHistory,
        fullRing = fullRing
      )

  def intOpt(options: Map[String, String], key: String, default: Int): Either[String, Int] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) =>
        raw.toIntOption.toRight(s"--$key must be an integer")

  def longOpt(options: Map[String, String], key: String, default: Long): Either[String, Long] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) =>
        raw.toLongOption.toRight(s"--$key must be a long")

  def doubleOpt(options: Map[String, String], key: String, default: Double): Either[String, Double] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) =>
        raw.toDoubleOption.toRight(s"--$key must be a double")

  def boolOpt(options: Map[String, String], key: String, default: Boolean): Either[String, Boolean] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) =>
        raw.trim.toLowerCase match
          case "true"  => Right(true)
          case "false" => Right(false)
          case _       => Left(s"--$key must be true or false")

  def pathOpt(options: Map[String, String], key: String, default: Path): Either[String, Path] =
    Right(options.get(key).map(Paths.get(_)).getOrElse(default))

  def optionalPathOpt(options: Map[String, String], key: String): Either[String, Option[Path]] =
    Right(options.get(key).map(Paths.get(_)))

  def heroModeOpt(
      options: Map[String, String],
      key: String,
      default: HeroMode
  ): Either[String, HeroMode] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) =>
        raw.trim.toLowerCase match
          case "adaptive"  => Right(HeroMode.Adaptive)
          case "gto"       => Right(HeroMode.Gto)
          case "strategic" => Right(HeroMode.Strategic)
          case _           => Left("--heroStyle must be one of: adaptive, gto, strategic")

  def gtoModeOpt(
      options: Map[String, String],
      key: String,
      default: GtoMode
  ): Either[String, GtoMode] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) =>
        raw.trim.toLowerCase match
          case "fast"  => Right(GtoMode.Fast)
          case "exact" => Right(GtoMode.Exact)
          case _       => Left("--gtoMode must be one of: fast, exact")

  val usage: String =
    """Usage:
      |  runMain sicfun.holdem.runtime.TexasHoldemPlayingHall [--key=value ...]
      |
      |  --hands=<int>                 default 100000
      |  --tableCount=<int>            default 1
      |  --playerCount=<int>           table seats to model (2..9, default 2)
      |  --reportEvery=<int>           default 10000
      |  --learnEveryHands=<int>       default 50000 (0 disables learning)
      |  --learningWindowSamples=<int> default 200000 (0 = unbounded)
      |  --seed=<long>                 default 42
      |  --outDir=<path>               default data/playing-hall
      |  --modelArtifactDir=<path>     optional initial trained model
      |  --heroStyle=<style>           adaptive|gto
      |  --heroPosition=<Position>     explicit table position (defaults to Button)
      |  --heroSeat=<seat>             legacy heads-up alias: button|bigblind
      |  --gtoMode=<mode>              fast|exact (default exact)
      |  --villainStyle=<style>        nit|tag|lag|callingstation|station|maniac|gto
      |  --villainPool=<styles>        optional comma-separated villain pool overriding villainStyle
      |  --heroExplorationRate=<double> default 0.05 (epsilon-greedy, [0,1])
      |  --raiseSize=<double>          default 2.5
      |  --bunchingTrials=<int>        default 80
      |  --equityTrials=<int>          default 700
      |  --saveTrainingTsv=<bool>      default true
      |  --saveDdreTrainingTsv=<bool>  default false
      |  --saveReviewHandHistory=<bool> default false
      |  --fullRing=<bool>             default false (all villains always active)
      |""".stripMargin
