package sicfun.holdem.history

import sicfun.holdem.cli.CliHelpers

/** CLI tool to print the canonical remembered-player population in TSV format.
  *
  * Loads the opponent profile store, optionally filters by site, and outputs
  * a tab-separated table with columns: playerUid, profileUid, site, name,
  * modelUid, behaviorUid. Sorted by (site, name, playerUid).
  *
  * Usage:
  * {{{
  * runMain sicfun.holdem.history.ListRememberedPlayers \
  *   --store=data/opponents.json [--site=pokerstars]
  * }}}
  */
object ListRememberedPlayers:
  private final case class CliConfig(
      store: OpponentMemoryTarget,
      siteFilter: Option[String]
  )

  def main(args: Array[String]): Unit =
    run(args) match
      case Left(err) =>
        System.err.println(err)
        sys.exit(1)
      case Right(lines) =>
        lines.foreach(println)

  /** Load the store, filter by site if requested, and return TSV rows (header + data). */
  def run(args: Array[String]): Either[String, Vector[String]] =
    for
      config <- parseArgs(args)
      store = OpponentProfileStorePersistence.load(config.store)
      filtered = store.population.filter(player =>
        config.siteFilter.forall(filter => OpponentIdentity.normalizeSite(player.canonicalSite) == OpponentIdentity.normalizeSite(filter))
      )
    yield
      val header = Vector("playerUid\tprofileUid\tsite\tname\tmodelUid\tbehaviorUid")
      val rows = filtered
        .sortBy(player => (player.canonicalSite, player.canonicalName, player.playerUid))
        .map { player =>
          Vector(
            player.playerUid,
            player.profileUid,
            player.canonicalSite,
            player.canonicalName,
            player.modelUid.getOrElse(""),
            player.behaviorUid.getOrElse("")
          ).mkString("\t")
        }
      header ++ rows

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
        siteFilter = options.get("site").map(_.trim).filter(_.nonEmpty)
      yield CliConfig(store = store, siteFilter = siteFilter)

  private def requiredOption(options: Map[String, String], key: String): Either[String, String] =
    options.get(key).map(_.trim).filter(_.nonEmpty).toRight(s"--$key is required")

  private val usage =
    """Usage:
      |  runMain sicfun.holdem.history.ListRememberedPlayers --store=<path|jdbc:postgresql://...> [--site=<site>]
      |
      |Options:
      |  --store=<path|jdbc:postgresql://...> Opponent memory store
      |  --storeUser=<user>         Optional PostgreSQL user
      |  --storePassword=<password> Optional PostgreSQL password
      |  --storeSchema=<schema>     Optional PostgreSQL schema (default: public)
      |  --site=<site>              Optional site filter, for example sicfun@localhost
      |""".stripMargin
