package sicfun.holdem.history

import sicfun.holdem.cli.CliHelpers

/** Promotes an existing remembered profile into a SICFUN-local canonical player identity.
  *
  * The new player is registered under `sicfun@localhost` and receives:
  *   - a behavior UID derived from the canonical profile representation
  *   - a stable player UID derived from `(modelUid, behaviorUid)`
  */
object RegisterSicfunPlayer:
  final case class RunSummary(
      playerUid: String,
      profileUid: String,
      site: String,
      playerName: String,
      handsObserved: Int
  )

  private final case class CliConfig(
      store: OpponentMemoryTarget,
      sourceSite: String,
      sourceName: String,
      playerName: String,
      modelUid: String
  )

  def main(args: Array[String]): Unit =
    run(args) match
      case Left(err) =>
        System.err.println(err)
        sys.exit(1)
      case Right(summary) =>
        println("=== SICFUN Player Registered ===")
        println(s"site: ${summary.site}")
        println(s"playerName: ${summary.playerName}")
        println(s"playerUid: ${summary.playerUid}")
        println(s"profileUid: ${summary.profileUid}")
        println(s"handsObserved: ${summary.handsObserved}")

  def run(args: Array[String]): Either[String, RunSummary] =
    for
      config <- parseArgs(args)
      store = OpponentProfileStorePersistence.load(config.store)
      sourceProfile <- store.find(config.sourceSite, config.sourceName)
        .toRight(s"source profile not found: ${config.sourceSite}/${config.sourceName}")
      updatedStore = store.registerSicfunPlayer(
        playerName = config.playerName,
        modelUid = config.modelUid,
        profile = sourceProfile
      )
      remembered <- updatedStore.find(OpponentIdentity.SicfunLocalSite, config.playerName)
        .toRight(s"failed to register SICFUN player '${config.playerName}'")
      _ = OpponentProfileStorePersistence.save(config.store, updatedStore)
    yield RunSummary(
      playerUid = remembered.playerUid.getOrElse(""),
      profileUid = remembered.profileUid.getOrElse(""),
      site = remembered.site,
      playerName = remembered.playerName,
      handsObserved = remembered.handsObserved
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
        sourceSite <- requiredOption(options, "sourceSite")
        sourceName <- requiredOption(options, "sourceName")
        playerName <- requiredOption(options, "playerName")
        modelUid <- requiredOption(options, "modelUid")
      yield CliConfig(
        store = store,
        sourceSite = sourceSite,
        sourceName = sourceName,
        playerName = playerName,
        modelUid = modelUid
      )

  private def requiredOption(options: Map[String, String], key: String): Either[String, String] =
    options.get(key).map(_.trim).filter(_.nonEmpty).toRight(s"--$key is required")

  private val usage =
    """Usage:
      |  runMain sicfun.holdem.history.RegisterSicfunPlayer --store=<path|jdbc:postgresql://...> --sourceSite=<site> --sourceName=<name> --playerName=<name> --modelUid=<uid>
      |
      |Options:
      |  --store=<path|jdbc:postgresql://...> Opponent memory store
      |  --storeUser=<user>         Optional PostgreSQL user
      |  --storePassword=<password> Optional PostgreSQL password
      |  --storeSchema=<schema>     Optional PostgreSQL schema (default: public)
      |  --sourceSite=<site>        Existing remembered profile site
      |  --sourceName=<name>        Existing remembered profile name
      |  --playerName=<name>        New SICFUN-local player name
      |  --modelUid=<uid>           Trained model UID for the SICFUN player
      |""".stripMargin
