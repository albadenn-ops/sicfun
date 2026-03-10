package sicfun.holdem.cli
import sicfun.holdem.types.*
import sicfun.holdem.equity.*

import sicfun.core.{Card, DiscreteDistribution}
import java.util.Locale

/** Shared utilities for CLI comparison tools. */
private[holdem] object CliHelpers:
  def requireOptions(args: IterableOnce[String]): Map[String, String] =
    parseOptions(args.iterator.toArray) match
      case Right(options) => options
      case Left(error) => throw new IllegalArgumentException(error)

  def parseIntOptionEither(options: Map[String, String], key: String, default: Int): Either[String, Int] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) => raw.toIntOption.toRight(s"--$key must be an integer")

  def parseLongOptionEither(options: Map[String, String], key: String, default: Long): Either[String, Long] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) => raw.toLongOption.toRight(s"--$key must be a long")

  def parseOptionalLongOptionEither(options: Map[String, String], key: String): Either[String, Option[Long]] =
    options.get(key) match
      case None => Right(None)
      case Some(raw) => raw.toLongOption.map(Some(_)).toRight(s"--$key must be a long")

  def parseDoubleOptionEither(
      options: Map[String, String],
      key: String,
      default: Double,
      typeLabel: String = "a number"
  ): Either[String, Double] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) => raw.toDoubleOption.toRight(s"--$key must be $typeLabel")

  def parseStrictBooleanOptionEither(options: Map[String, String], key: String, default: Boolean): Either[String, Boolean] =
    parseBooleanOptionEither(
      options = options,
      key = key,
      default = default,
      trueTokens = Set("true"),
      falseTokens = Set("false"),
      errorMessage = s"--$key must be true or false"
    )

  def parseBooleanOptionEither(options: Map[String, String], key: String, default: Boolean): Either[String, Boolean] =
    parseBooleanOptionEither(
      options = options,
      key = key,
      default = default,
      trueTokens = Set("1", "true", "yes"),
      falseTokens = Set("0", "false", "no"),
      errorMessage = s"--$key must be a boolean (true/false)"
    )

  def parseExtendedBooleanOptionEither(options: Map[String, String], key: String, default: Boolean): Either[String, Boolean] =
    parseBooleanOptionEither(
      options = options,
      key = key,
      default = default,
      trueTokens = Set("1", "true", "yes", "on"),
      falseTokens = Set("0", "false", "no", "off"),
      errorMessage = s"--$key must be a boolean (true/false)"
    )

  def parseOptions(args: Array[String]): Either[String, Map[String, String]] =
    parseKeyValueOptions(
      args = args,
      invalidTokenMessage = token => s"invalid argument '$token'; expected --key=value",
      blankTokenMessage = token => s"invalid argument '$token'; key/value must be non-empty",
      allowBlankValue = false
    )

  def parseOptionsAllowBlankValues(args: Array[String]): Either[String, Map[String, String]] =
    parseKeyValueOptions(
      args = args,
      invalidTokenMessage = token => s"invalid option format '$token'; expected --key=value",
      blankTokenMessage = token => s"invalid option key in '$token'",
      allowBlankValue = true
    )

  def requireIntOption(options: Map[String, String], key: String, default: Int): Int =
    options.get(key) match
      case None => default
      case Some(raw) =>
        raw.toIntOption.getOrElse(
          throw new IllegalArgumentException(s"--$key must be an integer")
        )

  def requireLongOption(options: Map[String, String], key: String, default: Long): Long =
    options.get(key) match
      case None => default
      case Some(raw) =>
        raw.toLongOption.getOrElse(
          throw new IllegalArgumentException(s"--$key must be a long")
        )

  def requireDoubleOption(options: Map[String, String], key: String, default: Double): Double =
    options.get(key) match
      case None => default
      case Some(raw) =>
        raw.toDoubleOption.getOrElse(
          throw new IllegalArgumentException(s"--$key must be a number")
        )

  def requireBooleanOption(options: Map[String, String], key: String, default: Boolean): Boolean =
    options.get(key) match
      case None => default
      case Some(raw) =>
        raw.trim.toLowerCase(Locale.ROOT) match
          case "1" | "true" | "yes" | "on" => true
          case "0" | "false" | "no" | "off" => false
          case _ =>
            throw new IllegalArgumentException(s"--$key must be a boolean (true/false)")

  def optionalIntOption(options: Map[String, String], key: String): Option[Int] =
    options.get(key).map { raw =>
      raw.toIntOption.getOrElse(
        throw new IllegalArgumentException(s"--$key must be an integer")
      )
    }

  def optionalLongOption(options: Map[String, String], key: String): Option[Long] =
    options.get(key).map { raw =>
      raw.toLongOption.getOrElse(
        throw new IllegalArgumentException(s"--$key must be a long")
      )
    }

  def optionalDoubleOption(options: Map[String, String], key: String): Option[Double] =
    options.get(key).map { raw =>
      raw.toDoubleOption.getOrElse(
        throw new IllegalArgumentException(s"--$key must be a number")
      )
    }

  def requireCsvTokens(raw: String, key: String): Vector[String] =
    val tokens = raw
      .split(',')
      .toVector
      .map(_.trim)
      .filter(_.nonEmpty)
    require(tokens.nonEmpty, s"$key must be non-empty")
    tokens

  def requirePositiveIntList(raw: String, key: String): Vector[Int] =
    val values = requireCsvTokens(raw, key).map { token =>
      token.toIntOption.getOrElse(
        throw new IllegalArgumentException(s"--$key values must be integers")
      )
    }
    require(values.forall(_ > 0), s"$key values must be positive")
    values

  def optionalPositiveIntList(options: Map[String, String], key: String): Option[Vector[Int]] =
    options.get(key).map(requirePositiveIntList(_, key))

  def requireNoUnknownOptions(options: Map[String, String], allowedKeys: Set[String]): Unit =
    options.keysIterator.filterNot(allowedKeys.contains).foreach { other =>
      throw new IllegalArgumentException(s"unknown option '$other'")
    }

  def parseHoleCards(token: String): HoleCards =
    val t = token.trim
    require(t.length == 4, s"expected 4-char token like AcAs, got '$token'")
    val c1 = Card.parse(t.substring(0, 2)).getOrElse(
      throw new IllegalArgumentException(s"invalid card in '$token'")
    )
    val c2 = Card.parse(t.substring(2, 4)).getOrElse(
      throw new IllegalArgumentException(s"invalid card in '$token'")
    )
    HoleCards.canonical(c1, c2)

  def parseRangeDistribution(token: String): DiscreteDistribution[HoleCards] =
    RangeParser.parse(token.trim) match
      case Right(dist) => dist
      case Left(err) => throw new IllegalArgumentException(s"failed to parse range token '$token': $err")

  def parseHoleCardsOptionEither(
      options: Map[String, String],
      key: String,
      default: String
  ): Either[String, HoleCards] =
    try Right(parseHoleCards(options.getOrElse(key, default)))
    catch
      case e: Exception => Left(s"--$key invalid: ${e.getMessage}")

  def parsePositionOptionEither(
      options: Map[String, String],
      key: String,
      default: Position
  ): Either[String, Position] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) =>
        Position.values.find(_.toString.equalsIgnoreCase(raw.trim))
          .toRight(s"--$key invalid position: $raw")

  def parseCandidateActionsOptionEither(
      options: Map[String, String],
      key: String,
      default: String,
      deduplicate: Boolean
  ): Either[String, Vector[PokerAction]] =
    val tokens = requireCsvTokens(options.getOrElse(key, default), key)
    val parsed = tokens.map(parseActionTokenEither)
    parsed.collectFirst { case Left(err) => err } match
      case Some(err) =>
        Left(s"--$key invalid: $err")
      case None =>
        val actions = parsed.collect { case Right(action) => action }
        val normalized = if deduplicate then actions.distinct else actions
        if normalized.nonEmpty then Right(normalized)
        else Left(s"--$key must contain at least one valid action")

  def fmt2(value: Double): String =
    String.format(Locale.ROOT, "%6.2f", java.lang.Double.valueOf(value))

  def fmt5(value: Double): String =
    String.format(Locale.ROOT, "%8.5f", java.lang.Double.valueOf(value))

  private def parseKeyValueOptions(
      args: Array[String],
      invalidTokenMessage: String => String,
      blankTokenMessage: String => String,
      allowBlankValue: Boolean
  ): Either[String, Map[String, String]] =
    args.foldLeft[Either[String, Map[String, String]]](Right(Map.empty)) { (accEither, token) =>
      accEither.flatMap { acc =>
        if !token.startsWith("--") then Left(invalidTokenMessage(token))
        else
          val body = token.drop(2)
          val eq = body.indexOf('=')
          if eq <= 0 || eq == body.length - 1 then Left(invalidTokenMessage(token))
          else
            val key = body.take(eq).trim
            val value = body.drop(eq + 1).trim
            if key.isEmpty || (!allowBlankValue && value.isEmpty) then Left(blankTokenMessage(token))
            else Right(acc.updated(key, value))
      }
    }

  private def parseBooleanOptionEither(
      options: Map[String, String],
      key: String,
      default: Boolean,
      trueTokens: Set[String],
      falseTokens: Set[String],
      errorMessage: String
  ): Either[String, Boolean] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) =>
        raw.trim.toLowerCase(Locale.ROOT) match
          case value if trueTokens.contains(value) => Right(true)
          case value if falseTokens.contains(value) => Right(false)
          case _ => Left(errorMessage)

  private def parseActionTokenEither(token: String): Either[String, PokerAction] =
    token.toLowerCase(Locale.ROOT) match
      case "fold" => Right(PokerAction.Fold)
      case "check" => Right(PokerAction.Check)
      case "call" => Right(PokerAction.Call)
      case raw if raw.startsWith("raise:") =>
        raw.drop(6).toDoubleOption match
          case Some(amount) if amount > 0.0 => Right(PokerAction.Raise(amount))
          case _ => Left(s"invalid raise amount in '$token'")
      case _ => Left(s"unsupported action token '$token'")
