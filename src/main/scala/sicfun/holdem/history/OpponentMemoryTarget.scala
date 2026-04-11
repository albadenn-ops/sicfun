package sicfun.holdem.history

import java.nio.file.{Path, Paths}
import java.util.Locale

/** Persistence target for opponent-memory storage.
  *
  * Represents where opponent profiles are stored. Two backends are supported:
  *   - [[OpponentMemoryTarget.Json]]: local JSON file (simple, portable)
  *   - [[OpponentMemoryTarget.Postgres]]: PostgreSQL database (shared, durable)
  *
  * The target is parsed from CLI arguments by [[OpponentMemoryTarget.parse]], which
  * auto-detects PostgreSQL URLs (jdbc:postgresql://, postgresql://, postgres://) and
  * falls back to a filesystem path for everything else.
  */
sealed trait OpponentMemoryTarget

/** Parses and validates opponent-memory target definitions. */
object OpponentMemoryTarget:
  final case class Json(path: Path) extends OpponentMemoryTarget
  final case class Postgres(
      jdbcUrl: String,
      user: Option[String] = None,
      password: Option[String] = None,
      schema: String = "public"
  ) extends OpponentMemoryTarget:
    def effectiveSchema: String = schema.trim.toLowerCase(Locale.ROOT)
    require(jdbcUrl.trim.nonEmpty, "jdbcUrl must be non-empty")
    require(effectiveSchema.matches("[A-Za-z_][A-Za-z0-9_]*"), s"invalid PostgreSQL schema '$schema'")

  /** Parse a raw string into an OpponentMemoryTarget.
    *
    * Detects PostgreSQL URLs by prefix (jdbc:postgresql:, postgresql://, postgres://)
    * and normalizes them to the jdbc:postgresql: form. Everything else is treated as
    * a filesystem path for JSON storage. Validates the schema as a PostgreSQL identifier.
    *
    * @param raw      the raw target string (filesystem path or PostgreSQL URL)
    * @param user     optional PostgreSQL user
    * @param password optional PostgreSQL password
    * @param schema   PostgreSQL schema name (default: "public")
    */
  def parse(
      raw: String,
      user: Option[String] = None,
      password: Option[String] = None,
      schema: String = "public"
  ): Either[String, OpponentMemoryTarget] =
    val trimmed = raw.trim
    if trimmed.isEmpty then Left("opponent memory target must be non-empty")
    else
      val normalizedSchema = schema.trim.toLowerCase(Locale.ROOT)
      val lowerTrimmed = trimmed.toLowerCase(Locale.ROOT)
      val jdbcUrl =
        if lowerTrimmed.startsWith("jdbc:postgresql:") then
          Some(s"jdbc:postgresql:${trimmed.drop("jdbc:postgresql:".length)}")
        else if lowerTrimmed.startsWith("postgresql://") then
          Some(s"jdbc:postgresql://${trimmed.drop("postgresql://".length)}")
        else if lowerTrimmed.startsWith("postgres://") then
          Some(s"jdbc:postgresql://${trimmed.drop("postgres://".length)}")
        else None
      jdbcUrl match
        case Some(value) =>
          if normalizedSchema.matches("[A-Za-z_][A-Za-z0-9_]*") then
            Right(Postgres(jdbcUrl = value, user = user.map(_.trim).filter(_.nonEmpty), password = password, schema = normalizedSchema))
          else Left(s"--opponentStoreSchema must be a valid PostgreSQL identifier, got '$schema'")
        case None =>
          if trimmed.matches("[A-Za-z][A-Za-z0-9+.-]*://.+") then
            Left(s"unsupported opponent memory target '$trimmed'; expected a filesystem path or PostgreSQL URL")
          else Right(Json(Paths.get(trimmed)))
