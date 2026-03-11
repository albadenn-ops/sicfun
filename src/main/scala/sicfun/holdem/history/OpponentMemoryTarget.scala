package sicfun.holdem.history

import java.nio.file.{Path, Paths}

sealed trait OpponentMemoryTarget

object OpponentMemoryTarget:
  final case class Json(path: Path) extends OpponentMemoryTarget
  final case class Postgres(
      jdbcUrl: String,
      user: Option[String] = None,
      password: Option[String] = None,
      schema: String = "public"
  ) extends OpponentMemoryTarget:
    require(jdbcUrl.trim.nonEmpty, "jdbcUrl must be non-empty")
    require(schema.matches("[A-Za-z_][A-Za-z0-9_]*"), s"invalid PostgreSQL schema '$schema'")

  def parse(
      raw: String,
      user: Option[String] = None,
      password: Option[String] = None,
      schema: String = "public"
  ): Either[String, OpponentMemoryTarget] =
    val trimmed = raw.trim
    if trimmed.isEmpty then Left("opponent memory target must be non-empty")
    else
      val jdbcUrl =
        if trimmed.startsWith("jdbc:postgresql:") then Some(trimmed)
        else if trimmed.startsWith("postgresql://") || trimmed.startsWith("postgres://") then Some(s"jdbc:$trimmed")
        else None
      jdbcUrl match
        case Some(value) =>
          if schema.matches("[A-Za-z_][A-Za-z0-9_]*") then
            Right(Postgres(jdbcUrl = value, user = user.map(_.trim).filter(_.nonEmpty), password = password, schema = schema))
          else Left(s"--opponentStoreSchema must be a valid PostgreSQL identifier, got '$schema'")
        case None =>
          Right(Json(Paths.get(trimmed)))
