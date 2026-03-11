package sicfun.holdem.history

import java.sql.{Connection, DriverManager, PreparedStatement}
import java.util.Properties

object OpponentProfileStorePersistence:
  def load(target: OpponentMemoryTarget): OpponentProfileStore =
    target match
      case OpponentMemoryTarget.Json(path) => OpponentProfileStore.load(path)
      case config: OpponentMemoryTarget.Postgres => PostgresOpponentProfileStore.load(config)

  def save(target: OpponentMemoryTarget, store: OpponentProfileStore): Unit =
    target match
      case OpponentMemoryTarget.Json(path) => OpponentProfileStore.save(path, store)
      case config: OpponentMemoryTarget.Postgres => PostgresOpponentProfileStore.save(config, store)

private object PostgresOpponentProfileStore:
  private val ProfilesTable = "opponent_profiles"
  private val PlayersTable = "opponent_players"
  private val AliasesTable = "opponent_player_aliases"
  private val PlayerCollapseTable = "opponent_player_collapses"
  private val ProfileCollapseTable = "opponent_profile_collapses"

  def load(config: OpponentMemoryTarget.Postgres): OpponentProfileStore =
    withConnection(config) { connection =>
      ensureSchema(connection, config.schema)
      val schema = config.schema
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
      ensureSchema(connection, config.schema)
      val normalized = OpponentProfileStore.normalize(store)
      connection.setAutoCommit(false)
      try
        clearTables(connection, config.schema)
        insertProfiles(connection, config.schema, normalized)
        insertPlayers(connection, config.schema, normalized)
        insertAliases(connection, config.schema, normalized)
        insertPlayerCollapses(connection, config.schema, normalized)
        insertProfileCollapses(connection, config.schema, normalized)
        connection.commit()
      catch
        case e: Exception =>
          connection.rollback()
          throw e
      finally
        connection.setAutoCommit(true)
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
    execute(connection, s"DELETE FROM $schema.$AliasesTable")
    execute(connection, s"DELETE FROM $schema.$PlayerCollapseTable")
    execute(connection, s"DELETE FROM $schema.$ProfileCollapseTable")
    execute(connection, s"DELETE FROM $schema.$PlayersTable")
    execute(connection, s"DELETE FROM $schema.$ProfilesTable")

  private def insertProfiles(connection: Connection, schema: String, store: OpponentProfileStore): Unit =
    withStatement(
      connection,
      s"""INSERT INTO $schema.$ProfilesTable (
         |  profile_uid, player_uid, site, player_name, payload
         |) VALUES (?, ?, ?, ?, CAST(? AS JSONB))""".stripMargin
    ) { statement =>
      store.profiles.foreach { profile =>
        statement.setString(1, profile.profileUid.getOrElse(""))
        statement.setString(2, profile.playerUid.orNull)
        statement.setString(3, profile.site)
        statement.setString(4, profile.playerName)
        statement.setString(5, ujson.write(OpponentProfileStore.writeProfile(profile)))
        statement.addBatch()
      }
      statement.executeBatch()
    }

  private def insertPlayers(connection: Connection, schema: String, store: OpponentProfileStore): Unit =
    withStatement(
      connection,
      s"""INSERT INTO $schema.$PlayersTable (
         |  player_uid, canonical_site, canonical_name, profile_uid, payload
         |) VALUES (?, ?, ?, ?, CAST(? AS JSONB))""".stripMargin
    ) { statement =>
      store.players.foreach { player =>
        statement.setString(1, player.playerUid)
        statement.setString(2, player.canonicalSite)
        statement.setString(3, player.canonicalName)
        statement.setString(4, player.profileUid)
        statement.setString(5, ujson.write(OpponentProfileStore.writePlayer(player)))
        statement.addBatch()
      }
      statement.executeBatch()
    }

  private def insertAliases(connection: Connection, schema: String, store: OpponentProfileStore): Unit =
    withStatement(
      connection,
      s"""INSERT INTO $schema.$AliasesTable (
         |  site, player_name, player_uid, payload
         |) VALUES (?, ?, ?, CAST(? AS JSONB))""".stripMargin
    ) { statement =>
      store.aliases.foreach { alias =>
        statement.setString(1, alias.site)
        statement.setString(2, alias.playerName)
        statement.setString(3, alias.playerUid)
        statement.setString(4, ujson.write(OpponentProfileStore.writeAlias(alias)))
        statement.addBatch()
      }
      statement.executeBatch()
    }

  private def insertPlayerCollapses(connection: Connection, schema: String, store: OpponentProfileStore): Unit =
    withStatement(
      connection,
      s"""INSERT INTO $schema.$PlayerCollapseTable (
         |  alias_player_uid, canonical_player_uid, payload
         |) VALUES (?, ?, CAST(? AS JSONB))""".stripMargin
    ) { statement =>
      store.playerCollapses.foreach { collapse =>
        statement.setString(1, collapse.aliasPlayerUid)
        statement.setString(2, collapse.canonicalPlayerUid)
        statement.setString(3, ujson.write(OpponentProfileStore.writePlayerCollapse(collapse)))
        statement.addBatch()
      }
      statement.executeBatch()
    }

  private def insertProfileCollapses(connection: Connection, schema: String, store: OpponentProfileStore): Unit =
    withStatement(
      connection,
      s"""INSERT INTO $schema.$ProfileCollapseTable (
         |  alias_profile_uid, canonical_profile_uid, payload
         |) VALUES (?, ?, CAST(? AS JSONB))""".stripMargin
    ) { statement =>
      store.profileCollapses.foreach { collapse =>
        statement.setString(1, collapse.aliasProfileUid)
        statement.setString(2, collapse.canonicalProfileUid)
        statement.setString(3, ujson.write(OpponentProfileStore.writeProfileCollapse(collapse)))
        statement.addBatch()
      }
      statement.executeBatch()
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
