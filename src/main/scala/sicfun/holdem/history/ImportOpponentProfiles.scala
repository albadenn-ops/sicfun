package sicfun.holdem.history
import sicfun.holdem.cli.CliHelpers
import sicfun.holdem.io.DecisionLoopEventFeedIO

import java.nio.file.{Files, Path, Paths}

/** CLI tool for importing external hand histories and building opponent profiles.
  *
  * Pipeline:
  *   1. Parse a hand history file from disk (auto-detects site or uses --site flag)
  *   2. Build [[OpponentProfile]]s via `OpponentProfile.fromImportedHands`
  *   3. Optionally persist profiles to JSON file or PostgreSQL via [[OpponentProfileStorePersistence]]
  *   4. Optionally export normalized PokerEvent rows to a feed file for replay tooling
  *
  * The hero is excluded from opponent profiles. Multiple output targets can be combined
  * (e.g., both --store and --feedOut in the same run).
  *
  * Usage:
  * {{{
  * runMain sicfun.holdem.history.ImportOpponentProfiles \
  *   --input=hand_history.txt --site=pokerstars --heroName=MyName \
  *   --store=data/opponents.json --feedOut=data/events.tsv
  * }}}
  */
object ImportOpponentProfiles:
  final case class RunSummary(
      site: HandHistorySite,
      handsImported: Int,
      profilesWritten: Int,
      storeTarget: Option[OpponentMemoryTarget],
      feedPath: Option[Path],
      profiles: Vector[OpponentProfile]
  )

  private final case class CliConfig(
      inputPath: Path,
      site: Option[HandHistorySite],
      heroName: Option[String],
      storeTarget: Option[OpponentMemoryTarget],
      feedOut: Option[Path]
  )

  def main(args: Array[String]): Unit =
    run(args) match
      case Left(err) =>
        System.err.println(err)
        sys.exit(1)
      case Right(summary) =>
        println("=== Opponent Profile Import ===")
        println(s"site: ${summary.site}")
        println(s"handsImported: ${summary.handsImported}")
        println(s"profilesWritten: ${summary.profilesWritten}")
        summary.storeTarget.foreach {
          case OpponentMemoryTarget.Json(path) =>
            println(s"storePath: ${path.toAbsolutePath.normalize()}")
          case OpponentMemoryTarget.Postgres(jdbcUrl, _, _, schema) =>
            println(s"storeTarget: $jdbcUrl (schema=$schema)")
        }
        summary.feedPath.foreach(path => println(s"feedPath: ${path.toAbsolutePath.normalize()}"))
        summary.profiles.take(5).foreach { profile =>
          val hints = profile.exploitHintDetails
            .map(hint => s"${hint.text} ${formatMetrics(hint.metrics)}")
            .mkString(" | ")
          println(s"${profile.playerName}: hands=${profile.handsObserved} archetype=${profile.archetypePosterior.mapEstimate} hints=$hints")
        }

  /** Execute the import pipeline: parse file, build profiles, write outputs.
    *
    * @return Right(summary) on success, Left(error message) on failure
    */
  def run(args: Array[String]): Either[String, RunSummary] =
    for
      config <- parseArgs(args)
      imported <- HandHistoryImport.parseFile(config.inputPath, config.site, config.heroName)
      profiles = OpponentProfile.fromImportedHands(
        site = imported.head.site.toString.toLowerCase,
        hands = imported,
        excludePlayers = config.heroName.toSet
      )
      _ <- maybeWriteFeed(config.feedOut, imported)
      _ <- maybeWriteStore(config.storeTarget, profiles)
    yield RunSummary(
      site = imported.head.site,
      handsImported = imported.length,
      profilesWritten = profiles.length,
      storeTarget = config.storeTarget,
      feedPath = config.feedOut,
      profiles = profiles
    )

  /** Optionally export normalized PokerEvent rows to a feed file for replay tooling.
    *
    * Events are sorted by (timestamp, handId, sequenceInHand) for chronological ordering.
    * Fails if the output file already exists (to prevent accidental overwrites).
    */
  private def maybeWriteFeed(
      feedOut: Option[Path],
      hands: Vector[ImportedHand]
  ): Either[String, Unit] =
    feedOut match
      case None => Right(())
      case Some(path) =>
        if Files.exists(path) then Left(s"--feedOut already exists: $path")
        else
          val orderedEvents = hands
            .flatMap(_.events)
            .sortBy(event => (event.occurredAtEpochMillis, event.handId, event.sequenceInHand))
          orderedEvents.foreach(event => DecisionLoopEventFeedIO.append(path, event))
          Right(())

  /** Optionally persist profiles to the opponent profile store (JSON or PostgreSQL).
    *
    * Loads the current store, upserts the new profiles, and saves back.
    * Uses merge semantics: existing profiles for the same player are updated, not replaced.
    */
  private def maybeWriteStore(
      storeTarget: Option[OpponentMemoryTarget],
      profiles: Vector[OpponentProfile]
  ): Either[String, Unit] =
    storeTarget match
      case None => Right(())
      case Some(target) =>
        try
          val current = OpponentProfileStorePersistence.load(target)
          OpponentProfileStorePersistence.save(target, current.upsertAll(profiles))
          Right(())
        catch
          case e: Exception => Left(s"failed to write opponent profile store: ${e.getMessage}")

  private def parseArgs(args: Array[String]): Either[String, CliConfig] =
    if args.contains("--help") || args.contains("-h") then Left(usage)
    else
      for
        options <- CliHelpers.parseOptions(args)
        input <- parseRequiredPath(options, "input")
        site <- parseOptionalSite(options, "site")
        heroName = options.get("heroName").map(_.trim).filter(_.nonEmpty)
        storeTarget <- parseOptionalStore(options)
        feedOut = options.get("feedOut").map(Paths.get(_))
      yield CliConfig(
        inputPath = input,
        site = site,
        heroName = heroName,
        storeTarget = storeTarget,
        feedOut = feedOut
      )

  private def parseOptionalStore(options: Map[String, String]): Either[String, Option[OpponentMemoryTarget]] =
    options.get("store") match
      case None => Right(None)
      case Some(raw) =>
        OpponentMemoryTarget.parse(
          raw = raw,
          user = options.get("storeUser"),
          password = options.get("storePassword"),
          schema = options.getOrElse("storeSchema", "public")
        ).map(Some(_))

  private def parseRequiredPath(
      options: Map[String, String],
      key: String
  ): Either[String, Path] =
    options.get(key) match
      case None => Left(s"--$key is required")
      case Some(raw) =>
        val path = Paths.get(raw)
        if Files.exists(path) then Right(path)
        else Left(s"--$key path does not exist: $raw")

  private def parseOptionalSite(
      options: Map[String, String],
      key: String
  ): Either[String, Option[HandHistorySite]] =
    options.get(key) match
      case None => Right(None)
      case Some(raw) if raw.trim.equalsIgnoreCase("auto") => Right(None)
      case Some(raw) => HandHistorySite.parse(raw).map(Some(_))

  private val usage =
    """Usage:
      |  runMain sicfun.holdem.history.ImportOpponentProfiles --input=<path> [--key=value ...]
      |
      |Options:
      |  --site=<auto|pokerstars|winamax|ggpoker>   Default auto
      |  --heroName=<screenName>    Optional hero name to exclude from stored opponent profiles
      |  --store=<path|jdbc:postgresql://...> Optional persistent opponent profile store output
      |  --storeUser=<user>         Optional PostgreSQL user
      |  --storePassword=<password> Optional PostgreSQL password
      |  --storeSchema=<schema>     Optional PostgreSQL schema (default: public)
      |  --feedOut=<path>           Optional normalized PokerEvent TSV output
      |""".stripMargin

  private def formatMetrics(metrics: Vector[Double]): String =
    metrics.map(value => f"$value%.3f").mkString("[", ", ", "]")
