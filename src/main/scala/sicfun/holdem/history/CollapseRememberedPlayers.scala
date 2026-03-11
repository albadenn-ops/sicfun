package sicfun.holdem.history

import sicfun.holdem.cli.CliHelpers

/** Records a human assertion that two remembered players are the same entity. */
object CollapseRememberedPlayers:
  final case class RunSummary(
      canonicalPlayerUid: String,
      canonicalProfileUid: String,
      aliasPlayerUid: String,
      aliasProfileUid: String,
      collapseProfiles: Boolean
  )

  private final case class CliConfig(
      store: OpponentMemoryTarget,
      canonicalSite: String,
      canonicalName: String,
      aliasSite: String,
      aliasName: String,
      collapseProfiles: Boolean
  )

  def main(args: Array[String]): Unit =
    run(args) match
      case Left(err) =>
        System.err.println(err)
        sys.exit(1)
      case Right(summary) =>
        println("=== Remembered Players Collapsed ===")
        println(s"canonicalPlayerUid: ${summary.canonicalPlayerUid}")
        println(s"canonicalProfileUid: ${summary.canonicalProfileUid}")
        println(s"aliasPlayerUid: ${summary.aliasPlayerUid}")
        println(s"aliasProfileUid: ${summary.aliasProfileUid}")
        println(s"collapseProfiles: ${summary.collapseProfiles}")

  def run(args: Array[String]): Either[String, RunSummary] =
    for
      config <- parseArgs(args)
      store = OpponentProfileStorePersistence.load(config.store)
      canonicalBefore <- store.findPlayer(config.canonicalSite, config.canonicalName)
        .toRight(s"canonical player not found: ${config.canonicalSite}/${config.canonicalName}")
      aliasBefore <- store.findPlayer(config.aliasSite, config.aliasName)
        .toRight(s"alias player not found: ${config.aliasSite}/${config.aliasName}")
      collapsed = store.collapsePlayers(
        canonicalSite = config.canonicalSite,
        canonicalName = config.canonicalName,
        aliasSite = config.aliasSite,
        aliasName = config.aliasName,
        collapseProfiles = config.collapseProfiles
      )
      _ = OpponentProfileStorePersistence.save(config.store, collapsed)
    yield RunSummary(
      canonicalPlayerUid = canonicalBefore.playerUid,
      canonicalProfileUid = canonicalBefore.profileUid,
      aliasPlayerUid = aliasBefore.playerUid,
      aliasProfileUid = aliasBefore.profileUid,
      collapseProfiles = config.collapseProfiles
    )

  private def parseArgs(args: Array[String]): Either[String, CliConfig] =
    if args.contains("--help") || args.contains("-h") then Left(usage)
    else
      for
        options <- CliHelpers.parseOptions(args)
        storeRaw <- requiredOption(options, "store")
        store <- OpponentMemoryTarget.parse(
          raw = storeRaw,
          user = options.get("storeUser"),
          password = options.get("storePassword"),
          schema = options.getOrElse("storeSchema", "public")
        )
        canonicalSite <- requiredOption(options, "canonicalSite")
        canonicalName <- requiredOption(options, "canonicalName")
        aliasSite <- requiredOption(options, "aliasSite")
        aliasName <- requiredOption(options, "aliasName")
        collapseProfiles <- CliHelpers.parseBooleanOptionEither(options, "collapseProfiles", false)
      yield CliConfig(
        store = store,
        canonicalSite = canonicalSite,
        canonicalName = canonicalName,
        aliasSite = aliasSite,
        aliasName = aliasName,
        collapseProfiles = collapseProfiles
      )

  private def requiredOption(options: Map[String, String], key: String): Either[String, String] =
    options.get(key).map(_.trim).filter(_.nonEmpty).toRight(s"--$key is required")

  private val usage =
    """Usage:
      |  runMain sicfun.holdem.history.CollapseRememberedPlayers --store=<path|jdbc:postgresql://...> --canonicalSite=<site> --canonicalName=<name> --aliasSite=<site> --aliasName=<name> [--collapseProfiles=true]
      |
      |Options:
      |  --store=<path|jdbc:postgresql://...> Opponent memory store
      |  --storeUser=<user>         Optional PostgreSQL user
      |  --storePassword=<password> Optional PostgreSQL password
      |  --storeSchema=<schema>     Optional PostgreSQL schema (default: public)
      |  --canonicalSite=<site>     Canonical remembered player site
      |  --canonicalName=<name>     Canonical remembered player name
      |  --aliasSite=<site>         Player site to collapse into the canonical player
      |  --aliasName=<name>         Player name to collapse into the canonical player
      |  --collapseProfiles=<bool>  Also collapse the alias profile into the canonical profile
      |""".stripMargin
