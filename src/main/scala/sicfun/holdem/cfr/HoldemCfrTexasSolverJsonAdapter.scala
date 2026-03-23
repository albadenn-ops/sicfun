package sicfun.holdem.cfr

import sicfun.holdem.cli.CliHelpers

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import ujson.{Arr, Num, Obj, Str, Value}

/** Adapts TexasSolver / TexasHoldemSolverJava root strategy dumps into the
  * generic provider JSON consumed by [[HoldemCfrExternalComparison]].
  *
  * This adapter intentionally stays file-based. It does not orchestrate the
  * external solver; it translates already-produced root strategy dumps into the
  * comparison contract used by this repo.
  */
object HoldemCfrTexasSolverJsonAdapter:
  final case class RunResult(
      providerName: String,
      spotCount: Int,
      outPath: Path
  )

  private final case class CliConfig(
      referencePath: Path,
      texasDir: Option[Path],
      texasFile: Option[Path],
      singleSpotId: Option[String],
      selectedSpotIds: Option[Set[String]],
      expectedPlayer: Option[Int],
      providerName: String,
      outPath: Path
  )

  private final case class ReferenceSpot(
      id: String,
      hero: String,
      state: Value,
      villainRange: Value,
      source: Value
  )

  private final case class TexasRootStrategy(
      actions: Vector[String],
      policyByHero: Map[String, Double],
      actingPlayer: Option[Int]
  )

  def main(args: Array[String]): Unit =
    val wantsHelp = args.contains("--help") || args.contains("-h")
    run(args) match
      case Right(result) =>
        println("=== Holdem CFR TexasSolver Adapter ===")
        println(s"provider: ${result.providerName}")
        println(s"spots: ${result.spotCount}")
        println(s"out: ${result.outPath.toAbsolutePath.normalize()}")
      case Left(error) =>
        if wantsHelp then println(error)
        else
          System.err.println(error)
          sys.exit(1)

  def run(args: Array[String]): Either[String, RunResult] =
    for
      config <- parseArgs(args)
      result <- adapt(
        referencePath = config.referencePath,
        texasDir = config.texasDir,
        texasFile = config.texasFile,
        singleSpotId = config.singleSpotId,
        selectedSpotIds = config.selectedSpotIds,
        expectedPlayer = config.expectedPlayer,
        providerName = config.providerName,
        outPath = config.outPath
      )
    yield result

  private[cfr] def adapt(
      referencePath: Path,
      texasDir: Option[Path],
      texasFile: Option[Path],
      singleSpotId: Option[String],
      selectedSpotIds: Option[Set[String]],
      expectedPlayer: Option[Int],
      providerName: String,
      outPath: Path
  ): Either[String, RunResult] =
    try
      val referenceSpots = loadReferenceSpots(referencePath, singleSpotId, selectedSpotIds)
      val providerSpots = referenceSpots.map { spot =>
        val texasPath = resolveTexasPath(spot.id, texasDir, texasFile, singleSpotId)
        val texasStrategy = loadTexasRootStrategy(texasPath, spot.hero)
        validateExpectedPlayer(texasStrategy.actingPlayer, expectedPlayer, texasPath)
        buildProviderSpot(spot, texasStrategy, texasPath)
      }
      Files.createDirectories(outPath.toAbsolutePath.normalize().getParent)
      Files.writeString(
        outPath,
        ujson.write(
          Obj(
            "providerName" -> Str(providerName),
            "spots" -> Arr.from(providerSpots)
          ),
          indent = 2
        ),
        StandardCharsets.UTF_8
      )
      Right(RunResult(providerName = providerName, spotCount = providerSpots.size, outPath = outPath))
    catch
      case e: Exception =>
        Left(s"holdem CFR TexasSolver adapter failed: ${e.getMessage}")

  private def loadReferenceSpots(
      referencePath: Path,
      singleSpotId: Option[String],
      selectedSpotIds: Option[Set[String]]
  ): Vector[ReferenceSpot] =
    require(Files.exists(referencePath), s"reference file does not exist: $referencePath")
    val json = ujson.read(Files.readString(referencePath, StandardCharsets.UTF_8))
    val spots = json("spots").arr.iterator.map { rawSpot =>
      val spot = rawSpot.obj
      val id = requiredString(spot, "id", "reference spot")
      ReferenceSpot(
        id = id,
        hero = requiredString(spot, "hero", s"reference spot '$id'"),
        state = spot.getOrElse(
          "state",
          throw new IllegalArgumentException(s"reference spot '$id' missing state")
        ),
        villainRange = spot.getOrElse(
          "villainRange",
          throw new IllegalArgumentException(s"reference spot '$id' missing villainRange")
        ),
        source = rawSpot
      )
    }.toVector
    val filtered =
      spots.filter { spot =>
        singleSpotId.forall(_ == spot.id) &&
        selectedSpotIds.forall(_.contains(spot.id))
      }
    require(filtered.nonEmpty, "no reference spots matched the selected spot ids")
    filtered

  private def resolveTexasPath(
      spotId: String,
      texasDir: Option[Path],
      texasFile: Option[Path],
      singleSpotId: Option[String]
  ): Path =
    texasFile match
      case Some(path) =>
        require(singleSpotId.contains(spotId), "--texasFile requires --spotId matching the selected reference spot")
        require(Files.exists(path), s"TexasSolver file does not exist: $path")
        path
      case None =>
        val dir = texasDir.getOrElse(
          throw new IllegalArgumentException("either --texasDir or --texasFile must be provided")
        )
        val path = dir.resolve(s"$spotId.json")
        require(Files.exists(path), s"TexasSolver file for spot '$spotId' does not exist: $path")
        path

  private def loadTexasRootStrategy(path: Path, heroToken: String): TexasRootStrategy =
    val json = ujson.read(Files.readString(path, StandardCharsets.UTF_8))
    val container = resolveStrategyContainer(json, path)
    val actions = parseStringArray(container, "actions", s"TexasSolver root strategy '$path'")
    require(actions.nonEmpty, s"TexasSolver root strategy '$path' must include at least one action")
    val strategyMap = container.get("strategy") match
      case Some(value) => value.obj
      case None =>
        throw new IllegalArgumentException(s"TexasSolver root strategy '$path' missing strategy map")
    val canonicalHero = CliHelpers.parseHoleCards(heroToken).toToken
    val policyVector = extractHeroPolicy(strategyMap, canonicalHero, actions.size, path)
    TexasRootStrategy(
      actions = actions,
      policyByHero = actions.zip(policyVector).toMap,
      actingPlayer = container.get("player").map {
        case Num(value) if value.isWhole => value.toInt
        case other =>
          throw new IllegalArgumentException(s"TexasSolver root strategy '$path' has non-integer player field: $other")
      }
    )

  private def resolveStrategyContainer(root: Value, path: Path): collection.Map[String, Value] =
    val obj = root.obj
    obj.get("strategy") match
      case Some(value) if value.obj.contains("actions") && value.obj.contains("strategy") => value.obj
      case _ if obj.contains("actions") && obj.contains("strategy") => obj
      case _ =>
        throw new IllegalArgumentException(
          s"TexasSolver file '$path' does not contain a root strategy object with actions and strategy"
        )

  private def extractHeroPolicy(
      strategyMap: collection.Map[String, Value],
      canonicalHero: String,
      expectedActionCount: Int,
      path: Path
  ): Vector[Double] =
    val matches = strategyMap.toVector.collect {
      case (rawHand, value) if CliHelpers.parseHoleCards(rawHand).toToken == canonicalHero =>
        value
    }
    require(matches.nonEmpty, s"TexasSolver file '$path' does not contain strategy for hero hand '$canonicalHero'")
    require(matches.size == 1, s"TexasSolver file '$path' contains duplicate strategy rows for '$canonicalHero'")
    val policy = matches.head.arr.iterator.map {
      case Num(value) if value.isFinite && value >= 0.0 => value
      case other =>
        throw new IllegalArgumentException(
          s"TexasSolver file '$path' contains non-finite or negative policy weight for '$canonicalHero': $other"
        )
    }.toVector
    require(
      policy.size == expectedActionCount,
      s"TexasSolver file '$path' action count mismatch for '$canonicalHero': expected $expectedActionCount, found ${policy.size}"
    )
    require(policy.exists(_ > 0.0), s"TexasSolver file '$path' contains all-zero policy for '$canonicalHero'")
    policy

  private def buildProviderSpot(
      spot: ReferenceSpot,
      texasStrategy: TexasRootStrategy,
      texasPath: Path
  ): Value =
    Obj(
      "id" -> Str(spot.id),
      "state" -> spot.state,
      "hero" -> Str(CliHelpers.parseHoleCards(spot.hero).toToken),
      "villainRange" -> spot.villainRange,
      "candidateActions" -> Arr.from(texasStrategy.actions.map(Str(_))),
      "policy" -> Obj.from(
        texasStrategy.policyByHero.toVector.map { case (action, probability) =>
          action -> Num(probability)
        }
      ),
      "sourceFile" -> Str(texasPath.getFileName.toString),
      "player" -> texasStrategy.actingPlayer.map(Num(_)).getOrElse(ujson.Null)
    )

  private def validateExpectedPlayer(
      actualPlayer: Option[Int],
      expectedPlayer: Option[Int],
      texasPath: Path
  ): Unit =
    expectedPlayer.foreach { expected =>
      val actual = actualPlayer.getOrElse(
        throw new IllegalArgumentException(
          s"TexasSolver file '$texasPath' is missing player but --expectedPlayer=$expected was requested"
        )
      )
      require(
        actual == expected,
        s"TexasSolver file '$texasPath' player mismatch: expected $expected, found $actual"
      )
    }

  private def requiredString(
      obj: collection.Map[String, Value],
      key: String,
      label: String
  ): String =
    obj.get(key) match
      case Some(Str(value)) if value.trim.nonEmpty => value.trim
      case Some(other) =>
        throw new IllegalArgumentException(s"$label.$key must be a non-empty string, found $other")
      case None =>
        throw new IllegalArgumentException(s"$label missing '$key'")

  private def parseStringArray(
      obj: collection.Map[String, Value],
      key: String,
      label: String
  ): Vector[String] =
    obj.get(key) match
      case Some(value) =>
        value.arr.iterator.map {
          case Str(text) if text.trim.nonEmpty => text.trim
          case other =>
            throw new IllegalArgumentException(s"$label.$key must contain non-empty strings, found $other")
        }.toVector
      case None =>
        throw new IllegalArgumentException(s"$label missing '$key'")

  private def parseArgs(args: Array[String]): Either[String, CliConfig] =
    if args.contains("--help") || args.contains("-h") then Left(usage)
    else
      for
        options <- CliHelpers.parseOptions(args)
        referencePath <- parseRequiredPath(options, "reference")
        outPath <- parseRequiredOutPath(options, "out")
        providerName = options.getOrElse("providerName", "TexasSolver")
        texasDir = options.get("texasDir").map(Paths.get(_))
        texasFile = options.get("texasFile").map(Paths.get(_))
        _ <- validateSourceChoice(texasDir, texasFile)
        singleSpotId = options.get("spotId")
        _ <- validateSingleFileOptions(texasFile, singleSpotId)
        selectedSpotIds <- parseOptionalSpotIds(options.get("spotIds"))
        expectedPlayer <- parseOptionalIntOption(options.get("expectedPlayer"), "expectedPlayer")
      yield
        CliConfig(
          referencePath = referencePath,
          texasDir = texasDir,
          texasFile = texasFile,
          singleSpotId = singleSpotId,
          selectedSpotIds = selectedSpotIds,
          expectedPlayer = expectedPlayer,
          providerName = providerName,
          outPath = outPath
        )

  private def parseRequiredPath(options: Map[String, String], key: String): Either[String, Path] =
    options.get(key) match
      case Some(raw) =>
        val path = Paths.get(raw)
        if Files.exists(path) then Right(path) else Left(s"--$key path '$raw' does not exist")
      case None => Left(s"--$key is required")

  private def parseRequiredOutPath(options: Map[String, String], key: String): Either[String, Path] =
    options.get(key) match
      case Some(raw) => Right(Paths.get(raw))
      case None      => Left(s"--$key is required")

  private def validateSourceChoice(
      texasDir: Option[Path],
      texasFile: Option[Path]
  ): Either[String, Unit] =
    (texasDir, texasFile) match
      case (Some(_), Some(_)) => Left("provide only one of --texasDir or --texasFile")
      case (None, None)       => Left("either --texasDir or --texasFile is required")
      case _                  => Right(())

  private def validateSingleFileOptions(
      texasFile: Option[Path],
      singleSpotId: Option[String]
  ): Either[String, Unit] =
    if texasFile.nonEmpty && singleSpotId.isEmpty then Left("--texasFile requires --spotId")
    else Right(())

  private def parseOptionalSpotIds(raw: Option[String]): Either[String, Option[Set[String]]] =
    raw match
      case None => Right(None)
      case Some(value) =>
        val ids = value.split(",").toVector.map(_.trim).filter(_.nonEmpty)
        if ids.isEmpty then Left("--spotIds must contain at least one non-empty id")
        else Right(Some(ids.toSet))

  private def parseOptionalIntOption(raw: Option[String], key: String): Either[String, Option[Int]] =
    raw match
      case None => Right(None)
      case Some(value) =>
        value.toIntOption match
          case Some(number) if number >= 0 => Right(Some(number))
          case _                           => Left(s"--$key must be a non-negative integer")

  private val usage =
    """Usage:
      |  runMain sicfun.holdem.cfr.HoldemCfrTexasSolverJsonAdapter [--key=value ...]
      |
      |Options:
      |  --reference=<path>      SICFUN external-comparison.json export
      |  --out=<path>            Output provider JSON for HoldemCfrExternalComparison
      |  --texasDir=<path>       Directory containing one TexasSolver JSON dump per spot id
      |                          named <spotId>.json
      |  --texasFile=<path>      Single TexasSolver JSON dump (requires --spotId)
      |  --spotId=<id>           Single reference spot id when using --texasFile
      |  --spotIds=<a,b,...>     Optional subset of reference spot ids
      |  --expectedPlayer=<int>  Optional expected acting-player index in Texas dumps
      |  --providerName=<name>   Default TexasSolver
      |""".stripMargin
