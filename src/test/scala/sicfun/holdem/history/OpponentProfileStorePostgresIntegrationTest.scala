package sicfun.holdem.history

import io.zonky.test.db.postgres.embedded.EmbeddedPostgres
import munit.FunSuite

import java.sql.Connection
import java.util.UUID
import scala.collection.mutable

/** Integration tests for the PostgreSQL backend of [[OpponentProfileStorePersistence]].
  *
  * Uses an embedded PostgreSQL server (via `io.zonky.test.db`) to exercise
  * the full JDBC persistence path without requiring an external database.
  * Each test gets a fresh randomly-named schema to avoid cross-contamination.
  *
  * Coverage:
  *   - '''Roundtrip''': save a normalized store (with profiles, players,
  *     aliases, player collapses, and profile collapses), reload it, and
  *     assert equality; verifies all 5 required tables are created
  *   - '''Empty schema bootstrap''': loading from a non-existent schema
  *     returns an empty store without creating the schema
  *   - '''Partial schema detection (missing tables)''': creating only one
  *     of the 5 required tables and attempting to load produces a clear
  *     `IllegalStateException` naming the missing tables
  *   - '''Partial schema detection (missing columns)''': creating all tables
  *     but dropping a required column (e.g. `payload`) produces a clear
  *     error naming the missing column
  */
class OpponentProfileStorePostgresIntegrationTest extends FunSuite:
  private var postgres: Option[EmbeddedPostgres] = None
  private val schemasToDrop = mutable.ArrayBuffer.empty[String]

  override def beforeAll(): Unit =
    super.beforeAll()
    postgres = Some(EmbeddedPostgres.builder().start())

  override def afterAll(): Unit =
    try
      cleanupSchemas()
    finally
      postgres.foreach(_.close())
      postgres = None
      super.afterAll()

  test("postgres persistence roundtrips the normalized store".tag(munit.Slow)) {
    val schema = freshSchema()
    val target = postgresTarget(schema)
    val expected = OpponentProfileStore.normalize(sampleStore())

    OpponentProfileStorePersistence.save(target, expected)

    val loaded = OpponentProfileStorePersistence.load(target)
    assertEquals(loaded, expected)
    assertEquals(existingTables(schema), OpponentProfileStorePersistence.PostgresRequiredTables.toSet)
  }

  test("postgres load from a missing schema returns empty without bootstrapping".tag(munit.Slow)) {
    val schema = freshSchema()
    val target = postgresTarget(schema)

    val loaded = OpponentProfileStorePersistence.load(target)

    assertEquals(loaded, OpponentProfileStore.empty)
    assert(!schemaExists(schema), s"load should not create missing schema '$schema'")
  }

  test("postgres load fails clearly on partially initialized schemas".tag(munit.Slow)) {
    val schema = freshSchema()
    withAdminConnection { connection =>
      execute(connection, s"CREATE SCHEMA $schema")
      execute(
        connection,
        s"""CREATE TABLE $schema.${OpponentProfileStorePersistence.PostgresProfilesTable} (
           |  profile_uid TEXT PRIMARY KEY,
           |  player_uid TEXT,
           |  site TEXT NOT NULL,
           |  player_name TEXT NOT NULL,
           |  payload JSONB NOT NULL
           |)""".stripMargin
      )
    }

    val error = intercept[IllegalStateException] {
      OpponentProfileStorePersistence.load(postgresTarget(schema))
    }

    assert(error.getMessage.contains("partially initialized"))
    assert(error.getMessage.contains(OpponentProfileStorePersistence.PostgresPlayersTable))
  }

  test("postgres load fails clearly when a required column is missing".tag(munit.Slow)) {
    val schema = freshSchema()
    withAdminConnection { connection =>
      execute(connection, s"CREATE SCHEMA $schema")
      createRequiredTableSet(connection, schema)
      execute(
        connection,
        s"ALTER TABLE $schema.${OpponentProfileStorePersistence.PostgresProfilesTable} DROP COLUMN payload"
      )
    }

    val error = intercept[IllegalStateException] {
      OpponentProfileStorePersistence.load(postgresTarget(schema))
    }

    assert(error.getMessage.contains("partially initialized"))
    assert(error.getMessage.contains(s"${OpponentProfileStorePersistence.PostgresProfilesTable}.payload"))
  }

  private def sampleStore(): OpponentProfileStore =
    val canonical = OpponentProfile(
      site = "pokerstars",
      playerName = "VillainA",
      handsObserved = 20,
      firstSeenEpochMillis = 1_700_000_000_000L,
      lastSeenEpochMillis = 1_700_000_000_500L,
      actionSummary = OpponentActionSummary(folds = 10, raises = 2, calls = 6, checks = 2),
      raiseResponses = sicfun.holdem.engine.RaiseResponseCounts(folds = 4, calls = 1, raises = 0),
      recentEvents = Vector.empty,
      seenHandIds = Vector("a-1", "a-2", "a-3")
    )
    val alias = OpponentProfile(
      site = "gg",
      playerName = "VillainB",
      handsObserved = 15,
      firstSeenEpochMillis = 1_700_000_001_000L,
      lastSeenEpochMillis = 1_700_000_001_500L,
      actionSummary = OpponentActionSummary(folds = 2, raises = 7, calls = 4, checks = 2),
      raiseResponses = sicfun.holdem.engine.RaiseResponseCounts(folds = 0, calls = 2, raises = 3),
      recentEvents = Vector.empty,
      seenHandIds = Vector("b-1", "b-2")
    )

    OpponentProfileStore.empty
      .upsert(canonical)
      .upsert(alias)
      .collapsePlayers(
        canonicalSite = canonical.site,
        canonicalName = canonical.playerName,
        aliasSite = alias.site,
        aliasName = alias.playerName,
        collapseProfiles = true,
        assertedAtEpochMillis = 123L
      )

  private def postgresTarget(schema: String): OpponentMemoryTarget.Postgres =
    OpponentMemoryTarget.Postgres(
      jdbcUrl = embeddedPostgres.getJdbcUrl("postgres", "postgres"),
      user = Some("postgres"),
      password = Some("postgres"),
      schema = schema
    )

  private def freshSchema(): String =
    val schema = s"itest_${UUID.randomUUID().toString.replace("-", "").toLowerCase}"
    schemasToDrop += schema
    schema

  private def withAdminConnection[A](fn: Connection => A): A =
    val connection = embeddedPostgres.getPostgresDatabase().getConnection()
    try fn(connection)
    finally connection.close()

  private def cleanupSchemas(): Unit =
    if schemasToDrop.nonEmpty && postgres.nonEmpty then
      withAdminConnection { connection =>
        schemasToDrop.reverseIterator.foreach { schema =>
          execute(connection, s"DROP SCHEMA IF EXISTS $schema CASCADE")
        }
      }
    schemasToDrop.clear()

  private def schemaExists(schema: String): Boolean =
    withAdminConnection { connection =>
      val statement = connection.prepareStatement(
        "SELECT 1 FROM information_schema.schemata WHERE schema_name = ?"
      )
      try
        statement.setString(1, schema)
        val result = statement.executeQuery()
        try result.next()
        finally result.close()
      finally statement.close()
    }

  private def existingTables(schema: String): Set[String] =
    withAdminConnection { connection =>
      val statement = connection.prepareStatement(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = ?"
      )
      try
        statement.setString(1, schema)
        val result = statement.executeQuery()
        val tables = Set.newBuilder[String]
        try
          while result.next() do
            tables += result.getString(1)
        finally result.close()
        tables.result()
      finally statement.close()
    }

  private def execute(connection: Connection, sql: String): Unit =
    val statement = connection.createStatement()
    try statement.execute(sql)
    finally statement.close()

  private def createRequiredTableSet(connection: Connection, schema: String): Unit =
    execute(
      connection,
      s"""CREATE TABLE $schema.${OpponentProfileStorePersistence.PostgresProfilesTable} (
         |  profile_uid TEXT PRIMARY KEY,
         |  player_uid TEXT,
         |  site TEXT NOT NULL,
         |  player_name TEXT NOT NULL,
         |  payload JSONB NOT NULL
         |)""".stripMargin
    )
    execute(
      connection,
      s"""CREATE TABLE $schema.${OpponentProfileStorePersistence.PostgresPlayersTable} (
         |  player_uid TEXT PRIMARY KEY,
         |  canonical_site TEXT NOT NULL,
         |  canonical_name TEXT NOT NULL,
         |  profile_uid TEXT NOT NULL,
         |  payload JSONB NOT NULL
         |)""".stripMargin
    )
    execute(
      connection,
      s"""CREATE TABLE $schema.${OpponentProfileStorePersistence.PostgresAliasesTable} (
         |  site TEXT NOT NULL,
         |  player_name TEXT NOT NULL,
         |  player_uid TEXT NOT NULL,
         |  payload JSONB NOT NULL,
         |  PRIMARY KEY (site, player_name)
         |)""".stripMargin
    )
    execute(
      connection,
      s"""CREATE TABLE $schema.${OpponentProfileStorePersistence.PostgresPlayerCollapseTable} (
         |  alias_player_uid TEXT PRIMARY KEY,
         |  canonical_player_uid TEXT NOT NULL,
         |  payload JSONB NOT NULL
         |)""".stripMargin
    )
    execute(
      connection,
      s"""CREATE TABLE $schema.${OpponentProfileStorePersistence.PostgresProfileCollapseTable} (
         |  alias_profile_uid TEXT PRIMARY KEY,
         |  canonical_profile_uid TEXT NOT NULL,
         |  payload JSONB NOT NULL
         |)""".stripMargin
    )

  private def embeddedPostgres: EmbeddedPostgres =
    postgres.getOrElse(fail("embedded PostgreSQL server was not started"))
end OpponentProfileStorePostgresIntegrationTest
