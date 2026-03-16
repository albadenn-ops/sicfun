package sicfun.holdem.history

import java.sql.{Connection, DriverManager, PreparedStatement}
import java.util.Locale
import java.util.Properties
import scala.collection.mutable

object OpponentProfileStorePersistence:
  private[history] val PostgresProfilesTable = "opponent_profiles"
  private[history] val PostgresPlayersTable = "opponent_players"
  private[history] val PostgresAliasesTable = "opponent_player_aliases"
  private[history] val PostgresPlayerCollapseTable = "opponent_player_collapses"
  private[history] val PostgresProfileCollapseTable = "opponent_profile_collapses"

  private[history] val PostgresRequiredTables: Vector[String] = Vector(
    PostgresProfilesTable,
    PostgresPlayersTable,
    PostgresAliasesTable,
    PostgresPlayerCollapseTable,
    PostgresProfileCollapseTable
  )

  private[history] val PostgresRequiredColumns: Map[String, Vector[String]] = Map(
    PostgresProfilesTable -> Vector("profile_uid", "player_uid", "site", "player_name", "payload"),
    PostgresPlayersTable -> Vector("player_uid", "canonical_site", "canonical_name", "profile_uid", "payload"),
    PostgresAliasesTable -> Vector("site", "player_name", "player_uid", "payload"),
    PostgresPlayerCollapseTable -> Vector("alias_player_uid", "canonical_player_uid", "payload"),
    PostgresProfileCollapseTable -> Vector("alias_profile_uid", "canonical_profile_uid", "payload")
  )

  private[history] def postgresMissingTables(existingTables: Set[String]): Vector[String] =
    val normalized = existingTables.map(_.toLowerCase(Locale.ROOT))
    PostgresRequiredTables.filterNot(normalized.contains)

  private[history] def postgresMissingColumns(
      existingColumnsByTable: Map[String, Set[String]]
  ): Map[String, Vector[String]] =
    val normalizedColumnsByTable = existingColumnsByTable.map { case (table, columns) =>
      table.toLowerCase(Locale.ROOT) -> columns.map(_.toLowerCase(Locale.ROOT))
    }
    PostgresRequiredColumns.flatMap { case (table, requiredColumns) =>
      normalizedColumnsByTable.get(table).flatMap { existingColumns =>
        val missingColumns = requiredColumns.filterNot(existingColumns.contains)
        Option.when(missingColumns.nonEmpty)(table -> missingColumns)
      }
    }

  private[history] enum PostgresStoreState:
    case Empty
    case Ready
    case Partial(missingTables: Vector[String], missingColumns: Map[String, Vector[String]])

  private[history] def postgresStoreState(
      existingTables: Set[String],
      existingColumnsByTable: Map[String, Set[String]]
  ): PostgresStoreState =
    val missingTables = postgresMissingTables(existingTables)
    val missingColumns = postgresMissingColumns(existingColumnsByTable)
    if missingTables.length == PostgresRequiredTables.length then
      PostgresStoreState.Empty
    else if missingTables.isEmpty && missingColumns.isEmpty then
      PostgresStoreState.Ready
    else
      PostgresStoreState.Partial(missingTables, missingColumns)

  def load(target: OpponentMemoryTarget): OpponentProfileStore =
    target match
      case OpponentMemoryTarget.Json(path) => OpponentProfileStore.load(path)
      case config: OpponentMemoryTarget.Postgres => PostgresOpponentProfileStore.load(config)

  def save(target: OpponentMemoryTarget, store: OpponentProfileStore): Unit =
    target match
      case OpponentMemoryTarget.Json(path) => OpponentProfileStore.save(path, store)
      case config: OpponentMemoryTarget.Postgres => PostgresOpponentProfileStore.save(config, store)

private object PostgresOpponentProfileStore:
  private val InsertBatchSize = 512
  private val ProfilesTable = OpponentProfileStorePersistence.PostgresProfilesTable
  private val PlayersTable = OpponentProfileStorePersistence.PostgresPlayersTable
  private val AliasesTable = OpponentProfileStorePersistence.PostgresAliasesTable
  private val PlayerCollapseTable = OpponentProfileStorePersistence.PostgresPlayerCollapseTable
  private val ProfileCollapseTable = OpponentProfileStorePersistence.PostgresProfileCollapseTable
  private val RequiredTableSet = OpponentProfileStorePersistence.PostgresRequiredTables.toSet

  def load(config: OpponentMemoryTarget.Postgres): OpponentProfileStore =
    withConnection(config) { connection =>
      val schema = config.effectiveSchema
      val existingTables = existingStoreTables(connection, schema)
      val existingColumnsByTable = existingStoreColumns(connection, schema)
      OpponentProfileStorePersistence.postgresStoreState(existingTables, existingColumnsByTable) match
        case OpponentProfileStorePersistence.PostgresStoreState.Empty =>
          OpponentProfileStore.empty
        case OpponentProfileStorePersistence.PostgresStoreState.Partial(missingTables, missingColumns) =>
          throw new IllegalStateException(
            partialSchemaErrorMessage(schema, missingTables, missingColumns)
          )
        case OpponentProfileStorePersistence.PostgresStoreState.Ready =>
          OpponentProfileStore.normalize(
            OpponentProfileStore(
              profiles = readRows(connection, s"SELECT payload FROM $schema.$ProfilesTable ORDER BY site, player_name, profile_uid")(OpponentProfileStore.readProfile),
              players = readRows(connection, s"SELECT payload FROM $schema.$PlayersTable ORDER BY canonical_site, canonical_name, player_uid")(OpponentProfileStore.readPlayer),
              aliases = readRows(connection, s"SELECT payload FROM $schema.$AliasesTable ORDER BY site, player_name")(OpponentProfileStore.readAlias),
              playerCollapses = readRows(connection, s"SELECT payload FROM $schema.$PlayerCollapseTable ORDER BY alias_player_uid")(OpponentProfileStore.readPlayerCollapse),
              profileCollapses = readRows(connection, s"SELECT payload FROM $schema.$ProfileCollapseTable ORDER BY alias_profile_uid")(OpponentProfileStore.readProfileCollapse)
            )
          )
    }

  def save(config: OpponentMemoryTarget.Postgres, store: OpponentProfileStore): Unit =
    withConnection(config) { connection =>
      val schema = config.effectiveSchema
      val normalized = OpponentProfileStore.normalize(store)
      val previousAutoCommit = connection.getAutoCommit
      connection.setAutoCommit(false)
      try
        ensureSchema(connection, schema)
        clearTables(connection, schema)
        insertProfiles(connection, schema, normalized)
        insertPlayers(connection, schema, normalized)
        insertAliases(connection, schema, normalized)
        insertPlayerCollapses(connection, schema, normalized)
        insertProfileCollapses(connection, schema, normalized)
        connection.commit()
      catch
        case e: Exception =>
          connection.rollback()
          throw e
      finally
        connection.setAutoCommit(previousAutoCommit)
    }

  private def withConnection[A](config: OpponentMemoryTarget.Postgres)(fn: Connection => A): A =
    val props = new Properties()
    config.user.foreach(value => props.setProperty("user", value))
    config.password.foreach(value => props.setProperty("password", value))
    val connection =
      if props.isEmpty then DriverManager.getConnection(config.jdbcUrl)
      else DriverManager.getConnection(config.jdbcUrl, props)
    try fn(connection)
    finally connection.close()

  private def ensureSchema(connection: Connection, schema: String): Unit =
    execute(connection, s"CREATE SCHEMA IF NOT EXISTS $schema")
    execute(
      connection,
      s"""CREATE TABLE IF NOT EXISTS $schema.$ProfilesTable (
         |  profile_uid TEXT PRIMARY KEY,
         |  player_uid TEXT,
         |  site TEXT NOT NULL,
         |  player_name TEXT NOT NULL,
         |  payload JSONB NOT NULL
         |)""".stripMargin
    )
    execute(connection, s"CREATE INDEX IF NOT EXISTS ${ProfilesTable}_site_name_idx ON $schema.$ProfilesTable (site, player_name)")
    execute(connection, s"CREATE INDEX IF NOT EXISTS ${ProfilesTable}_player_uid_idx ON $schema.$ProfilesTable (player_uid)")

    execute(
      connection,
      s"""CREATE TABLE IF NOT EXISTS $schema.$PlayersTable (
         |  player_uid TEXT PRIMARY KEY,
         |  canonical_site TEXT NOT NULL,
         |  canonical_name TEXT NOT NULL,
         |  profile_uid TEXT NOT NULL,
         |  payload JSONB NOT NULL
         |)""".stripMargin
    )
    execute(connection, s"CREATE INDEX IF NOT EXISTS ${PlayersTable}_site_name_idx ON $schema.$PlayersTable (canonical_site, canonical_name)")
    execute(connection, s"CREATE INDEX IF NOT EXISTS ${PlayersTable}_profile_uid_idx ON $schema.$PlayersTable (profile_uid)")

    execute(
      connection,
      s"""CREATE TABLE IF NOT EXISTS $schema.$AliasesTable (
         |  site TEXT NOT NULL,
         |  player_name TEXT NOT NULL,
         |  player_uid TEXT NOT NULL,
         |  payload JSONB NOT NULL,
         |  PRIMARY KEY (site, player_name)
         |)""".stripMargin
    )
    execute(connection, s"CREATE INDEX IF NOT EXISTS ${AliasesTable}_player_uid_idx ON $schema.$AliasesTable (player_uid)")

    execute(
      connection,
      s"""CREATE TABLE IF NOT EXISTS $schema.$PlayerCollapseTable (
         |  alias_player_uid TEXT PRIMARY KEY,
         |  canonical_player_uid TEXT NOT NULL,
         |  payload JSONB NOT NULL
         |)""".stripMargin
    )
    execute(
      connection,
      s"""CREATE TABLE IF NOT EXISTS $schema.$ProfileCollapseTable (
         |  alias_profile_uid TEXT PRIMARY KEY,
         |  canonical_profile_uid TEXT NOT NULL,
         |  payload JSONB NOT NULL
         |)""".stripMargin
    )

  private def clearTables(connection: Connection, schema: String): Unit =
    execute(
      connection,
      s"TRUNCATE TABLE $schema.$AliasesTable, $schema.$PlayerCollapseTable, $schema.$ProfileCollapseTable, $schema.$PlayersTable, $schema.$ProfilesTable"
    )

  private def insertProfiles(connection: Connection, schema: String, store: OpponentProfileStore): Unit =
    withStatement(
      connection,
      s"""INSERT INTO $schema.$ProfilesTable (
         |  profile_uid, player_uid, site, player_name, payload
         |) VALUES (?, ?, ?, ?, CAST(? AS JSONB))""".stripMargin
    ) { statement =>
      var pending = 0
      store.profiles.foreach { profile =>
        statement.setString(
          1,
          profile.profileUid.getOrElse(
            throw new IllegalArgumentException("normalized PostgreSQL profile rows must define profileUid")
          )
        )
        statement.setString(2, profile.playerUid.orNull)
        statement.setString(3, profile.site)
        statement.setString(4, profile.playerName)
        statement.setString(5, ujson.write(OpponentProfileStore.writeProfile(profile)))
        statement.addBatch()
        pending += 1
        pending = flushBatchIfNeeded(statement, pending)
      }
      flushBatch(statement, pending)
    }

  private def insertPlayers(connection: Connection, schema: String, store: OpponentProfileStore): Unit =
    withStatement(
      connection,
      s"""INSERT INTO $schema.$PlayersTable (
         |  player_uid, canonical_site, canonical_name, profile_uid, payload
         |) VALUES (?, ?, ?, ?, CAST(? AS JSONB))""".stripMargin
    ) { statement =>
      var pending = 0
      store.players.foreach { player =>
        statement.setString(1, player.playerUid)
        statement.setString(2, player.canonicalSite)
        statement.setString(3, player.canonicalName)
        statement.setString(4, player.profileUid)
        statement.setString(5, ujson.write(OpponentProfileStore.writePlayer(player)))
        statement.addBatch()
        pending += 1
        pending = flushBatchIfNeeded(statement, pending)
      }
      flushBatch(statement, pending)
    }

  private def insertAliases(connection: Connection, schema: String, store: OpponentProfileStore): Unit =
    withStatement(
      connection,
      s"""INSERT INTO $schema.$AliasesTable (
         |  site, player_name, player_uid, payload
         |) VALUES (?, ?, ?, CAST(? AS JSONB))""".stripMargin
    ) { statement =>
      var pending = 0
      store.aliases.foreach { alias =>
        statement.setString(1, alias.site)
        statement.setString(2, alias.playerName)
        statement.setString(3, alias.playerUid)
        statement.setString(4, ujson.write(OpponentProfileStore.writeAlias(alias)))
        statement.addBatch()
        pending += 1
        pending = flushBatchIfNeeded(statement, pending)
      }
      flushBatch(statement, pending)
    }

  private def insertPlayerCollapses(connection: Connection, schema: String, store: OpponentProfileStore): Unit =
    withStatement(
      connection,
      s"""INSERT INTO $schema.$PlayerCollapseTable (
         |  alias_player_uid, canonical_player_uid, payload
         |) VALUES (?, ?, CAST(? AS JSONB))""".stripMargin
    ) { statement =>
      var pending = 0
      store.playerCollapses.foreach { collapse =>
        statement.setString(1, collapse.aliasPlayerUid)
        statement.setString(2, collapse.canonicalPlayerUid)
        statement.setString(3, ujson.write(OpponentProfileStore.writePlayerCollapse(collapse)))
        statement.addBatch()
        pending += 1
        pending = flushBatchIfNeeded(statement, pending)
      }
      flushBatch(statement, pending)
    }

  private def insertProfileCollapses(connection: Connection, schema: String, store: OpponentProfileStore): Unit =
    withStatement(
      connection,
      s"""INSERT INTO $schema.$ProfileCollapseTable (
         |  alias_profile_uid, canonical_profile_uid, payload
         |) VALUES (?, ?, CAST(? AS JSONB))""".stripMargin
    ) { statement =>
      var pending = 0
      store.profileCollapses.foreach { collapse =>
        statement.setString(1, collapse.aliasProfileUid)
        statement.setString(2, collapse.canonicalProfileUid)
        statement.setString(3, ujson.write(OpponentProfileStore.writeProfileCollapse(collapse)))
        statement.addBatch()
        pending += 1
        pending = flushBatchIfNeeded(statement, pending)
      }
      flushBatch(statement, pending)
    }

  private def readRows[A](connection: Connection, sql: String)(decode: ujson.Value => A): Vector[A] =
    withStatement(connection, sql) { statement =>
      val result = statement.executeQuery()
      val buffer = Vector.newBuilder[A]
      while result.next() do
        buffer += decode(ujson.read(result.getString(1)))
      result.close()
      buffer.result()
    }

  private def execute(connection: Connection, sql: String): Unit =
    withStatement(connection, sql)(_.execute())

  private def withStatement[A](connection: Connection, sql: String)(fn: PreparedStatement => A): A =
    val statement = connection.prepareStatement(sql)
    try fn(statement)
    finally statement.close()

  private def existingStoreTables(connection: Connection, schema: String): Set[String] =
    val tables = Set.newBuilder[String]
    val result = connection.getMetaData.getTables(null, schema.toLowerCase(Locale.ROOT), "%", Array("TABLE"))
    try
      while result.next() do
        Option(result.getString("TABLE_NAME"))
          .map(_.toLowerCase(Locale.ROOT))
          .filter(RequiredTableSet.contains)
          .foreach(tables += _)
    finally result.close()
    tables.result()

  private def existingStoreColumns(connection: Connection, schema: String): Map[String, Set[String]] =
    val columnsByTable = mutable.Map.empty[String, mutable.Set[String]]
    val result = connection.getMetaData.getColumns(null, schema.toLowerCase(Locale.ROOT), "%", "%")
    try
      while result.next() do
        val maybeTable = Option(result.getString("TABLE_NAME")).map(_.toLowerCase(Locale.ROOT))
        val maybeColumn = Option(result.getString("COLUMN_NAME")).map(_.toLowerCase(Locale.ROOT))
        for
          table <- maybeTable
          column <- maybeColumn
          if RequiredTableSet.contains(table)
        do
          columnsByTable.getOrElseUpdate(table, mutable.Set.empty) += column
    finally result.close()
    columnsByTable.view.mapValues(_.toSet).toMap

  private def partialSchemaErrorMessage(
      schema: String,
      missingTables: Vector[String],
      missingColumns: Map[String, Vector[String]]
  ): String =
    val details = Vector(
      Option.when(missingTables.nonEmpty)(s"missing tables: ${missingTables.mkString(", ")}"),
      Option.when(missingColumns.nonEmpty) {
        val formattedColumns = missingColumns.toVector
          .sortBy(_._1)
          .flatMap { case (table, columns) =>
            columns.map(column => s"$table.$column")
          }
        s"missing columns: ${formattedColumns.mkString(", ")}"
      }
    ).flatten
    s"PostgreSQL opponent profile store schema '$schema' is partially initialized; ${details.mkString("; ")}"

  private def flushBatchIfNeeded(statement: PreparedStatement, pending: Int): Int =
    if pending >= InsertBatchSize then
      statement.executeBatch()
      0
    else pending

  private def flushBatch(statement: PreparedStatement, pending: Int): Unit =
    if pending > 0 then statement.executeBatch()
