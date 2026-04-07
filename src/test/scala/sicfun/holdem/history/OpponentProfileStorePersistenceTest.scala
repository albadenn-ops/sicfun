package sicfun.holdem.history

import munit.FunSuite

/** Tests for the PostgreSQL schema state detection logic in
  * [[OpponentProfileStorePersistence]].
  *
  * These are pure-logic unit tests (no database connection required) that
  * verify the `postgresStoreState` function correctly classifies the
  * schema state as Empty, Ready, or Partial based on the sets of existing
  * tables and columns:
  *
  *   - '''Empty''': no required tables exist -> safe to bootstrap from scratch
  *   - '''Ready''': all 5 required tables with all required columns present
  *   - '''Partial (missing tables)''': some tables present, some missing;
  *     case-insensitive table name matching is verified
  *   - '''Partial (missing columns)''': all tables exist but a required column
  *     (e.g. `payload`) was dropped
  */
class OpponentProfileStorePersistenceTest extends FunSuite:
  test("postgres store state treats missing store tables as an empty bootstrap state") {
    assertEquals(
      OpponentProfileStorePersistence.postgresStoreState(Set.empty, Map.empty),
      OpponentProfileStorePersistence.PostgresStoreState.Empty
    )
  }

  test("postgres store state is ready when all required tables are present") {
    val requiredColumns = OpponentProfileStorePersistence.PostgresRequiredColumns.view.mapValues(_.toSet).toMap
    assertEquals(
      OpponentProfileStorePersistence.postgresStoreState(
        OpponentProfileStorePersistence.PostgresRequiredTables.toSet,
        requiredColumns
      ),
      OpponentProfileStorePersistence.PostgresStoreState.Ready
    )
  }

  test("postgres store state flags partial schemas and ignores table-name and column casing") {
    val partial = OpponentProfileStorePersistence.postgresStoreState(
      Set(
        "OPPONENT_PROFILES",
        "opponent_players",
        "opponent_player_aliases"
      ),
      Map(
        "OPPONENT_PROFILES" -> Set("PROFILE_UID", "PLAYER_UID", "SITE", "PLAYER_NAME", "PAYLOAD"),
        "opponent_players" -> Set("player_uid", "canonical_site", "canonical_name", "profile_uid", "payload"),
        "opponent_player_aliases" -> Set("site", "player_name", "player_uid", "payload")
      )
    )

    assertEquals(
      partial,
      OpponentProfileStorePersistence.PostgresStoreState.Partial(
        Vector("opponent_player_collapses", "opponent_profile_collapses"),
        Map.empty
      )
    )
  }

  test("postgres store state flags schemas with missing required columns") {
    val partial = OpponentProfileStorePersistence.postgresStoreState(
      OpponentProfileStorePersistence.PostgresRequiredTables.toSet,
      OpponentProfileStorePersistence.PostgresRequiredColumns.updated(
        OpponentProfileStorePersistence.PostgresProfilesTable,
        Vector("profile_uid", "player_uid", "site", "player_name")
      ).view.mapValues(_.toSet).toMap
    )

    assertEquals(
      partial,
      OpponentProfileStorePersistence.PostgresStoreState.Partial(
        Vector.empty,
        Map(OpponentProfileStorePersistence.PostgresProfilesTable -> Vector("payload"))
      )
    )
  }
end OpponentProfileStorePersistenceTest
