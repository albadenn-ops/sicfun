package sicfun.holdem.history

import munit.FunSuite

import java.nio.file.Paths

/** Tests for [[OpponentMemoryTarget.parse]], which routes a raw store path
  * string to either a [[OpponentMemoryTarget.Json]] file target or a
  * [[OpponentMemoryTarget.Postgres]] JDBC target.
  *
  * Coverage:
  *   - `postgres://` URLs are normalized to `jdbc:postgresql://` with
  *     user, password, and schema trimmed/lowercased
  *   - `postgresql://` URLs are kept intact with `jdbc:` prefix added
  *   - Mixed-case URL schemes (`PoStGrEs://`, `JDBC:PostgreSQL://`) are
  *     accepted and normalized
  *   - Non-PostgreSQL file paths fall back to [[OpponentMemoryTarget.Json]]
  *   - Unsupported URL-like targets (e.g. `mysql://`) are rejected with
  *     a descriptive error message
  *   - Invalid PostgreSQL schema identifiers (containing hyphens) are
  *     rejected
  */
class OpponentMemoryTargetTest extends FunSuite:
  test("parse normalizes postgres urls to PostgreSQL JDBC form") {
    val parsed = OpponentMemoryTarget.parse(
      "postgres://localhost:5432/sicfun",
      user = Some("  alice  "),
      password = Some("secret"),
      schema = " Analytics "
    )

    assertEquals(
      parsed,
      Right(
        OpponentMemoryTarget.Postgres(
          jdbcUrl = "jdbc:postgresql://localhost:5432/sicfun",
          user = Some("alice"),
          password = Some("secret"),
          schema = "analytics"
        )
      )
    )
  }

  test("parse keeps postgresql urls intact") {
    val parsed = OpponentMemoryTarget.parse("postgresql://localhost/sicfun")

    assertEquals(
      parsed,
      Right(
        OpponentMemoryTarget.Postgres(
          jdbcUrl = "jdbc:postgresql://localhost/sicfun"
        )
      )
    )
  }

  test("parse accepts mixed-case postgres url schemes") {
    val parsed = OpponentMemoryTarget.parse("PoStGrEs://localhost:5432/sicfun")

    assertEquals(
      parsed,
      Right(
        OpponentMemoryTarget.Postgres(
          jdbcUrl = "jdbc:postgresql://localhost:5432/sicfun"
        )
      )
    )
  }

  test("parse accepts mixed-case jdbc postgres url schemes") {
    val parsed = OpponentMemoryTarget.parse("JDBC:PostgreSQL://localhost/sicfun")

    assertEquals(
      parsed,
      Right(
        OpponentMemoryTarget.Postgres(
          jdbcUrl = "jdbc:postgresql://localhost/sicfun"
        )
      )
    )
  }

  test("parse falls back to a json path for non-postgres targets") {
    val parsed = OpponentMemoryTarget.parse("data/opponents.json")

    assertEquals(parsed, Right(OpponentMemoryTarget.Json(Paths.get("data/opponents.json"))))
  }

  test("parse rejects unsupported url-like targets") {
    val parsed = OpponentMemoryTarget.parse("mysql://localhost/sicfun")

    assertEquals(
      parsed,
      Left("unsupported opponent memory target 'mysql://localhost/sicfun'; expected a filesystem path or PostgreSQL URL")
    )
  }

  test("parse rejects invalid postgres schema identifiers") {
    val parsed = OpponentMemoryTarget.parse(
      "jdbc:postgresql://localhost/sicfun",
      schema = "bad-schema"
    )

    assertEquals(parsed, Left("--opponentStoreSchema must be a valid PostgreSQL identifier, got 'bad-schema'"))
  }
end OpponentMemoryTargetTest
