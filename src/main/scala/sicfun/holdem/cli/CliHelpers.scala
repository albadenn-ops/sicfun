package sicfun.holdem.cli
import sicfun.holdem.types.*
import sicfun.holdem.equity.*

import sicfun.core.{Card, DiscreteDistribution}
import java.util.Locale

/**
  * Shared command-line argument parsing and formatting utilities for the holdem CLI tools.
  *
  * This object provides a comprehensive library of helpers used by CLI entry points
  * (equity comparisons, batch training, advisor REPL, table generators, etc.) to parse
  * `--key=value` style arguments into typed Scala values.
  *
  * Two parsing paradigms are supported:
  *
  *   - '''Throwing (`require*`)''': Methods like [[requireIntOption]], [[requireBooleanOption]]
  *     throw `IllegalArgumentException` on invalid input. Used when the caller wants to
  *     fail-fast with a clear error message.
  *
  *   - '''Either-returning (`parse*Either`)''': Methods like [[parseIntOptionEither]],
  *     [[parseBooleanOptionEither]] return `Either[String, T]`, allowing the caller to
  *     accumulate errors or compose with other Either-based parsers via `flatMap`.
  *
  * The object also provides:
  *   - Poker-specific parsers: [[parseHoleCards]], [[parseRangeDistribution]],
  *     [[parsePositionOptionEither]], [[parseCandidateActionsOptionEither]]
  *   - Number formatters: [[fmt2]] (6-wide, 2 decimals) and [[fmt5]] (8-wide, 5 decimals)
  *   - CSV/list parsers: [[requireCsvTokens]], [[requirePositiveIntList]]
  *   - Unknown-option guard: [[requireNoUnknownOptions]]
  *
  * All string comparisons for boolean tokens and action tokens use `Locale.ROOT`
  * to avoid locale-dependent case conversion issues (e.g. Turkish 'I' problem).
  *
  * Package-private to `holdem` because these are internal CLI plumbing, not public API.
  */
private[holdem] object CliHelpers:
  /** Parses `--key=value` arguments, throwing on any malformed token.
    *
    * Delegates to [[parseOptions]] and unwraps the `Either`, converting `Left` to an exception.
    *
    * @param args the raw command-line arguments
    * @return a map of key -> value pairs (keys without the `--` prefix)
    * @throws IllegalArgumentException if any token is not in `--key=value` format
    */
  def requireOptions(args: IterableOnce[String]): Map[String, String] =
    parseOptions(args.iterator.toArray) match
      case Right(options) => options
      case Left(error) => throw new IllegalArgumentException(error)

  /** Extracts an integer option, returning `Left` with an error message if the value is not a valid int. */
  def parseIntOptionEither(options: Map[String, String], key: String, default: Int): Either[String, Int] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) => raw.toIntOption.toRight(s"--$key must be an integer")

  /** Extracts a long option, returning `Left` with an error message if the value is not a valid long. */
  def parseLongOptionEither(options: Map[String, String], key: String, default: Long): Either[String, Long] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) => raw.toLongOption.toRight(s"--$key must be a long")

  /** Extracts an optional long: returns `Right(None)` if key is absent, `Right(Some(v))` if valid, `Left` if invalid. */
  def parseOptionalLongOptionEither(options: Map[String, String], key: String): Either[String, Option[Long]] =
    options.get(key) match
      case None => Right(None)
      case Some(raw) => raw.toLongOption.map(Some(_)).toRight(s"--$key must be a long")

  /** Extracts a double option, returning `Left` with an error message if the value is not numeric.
    *
    * @param typeLabel optional label used in the error message (e.g. "a probability", "a number")
    */
  def parseDoubleOptionEither(
      options: Map[String, String],
      key: String,
      default: Double,
      typeLabel: String = "a number"
  ): Either[String, Double] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) => raw.toDoubleOption.toRight(s"--$key must be $typeLabel")

  /** Parses a boolean option accepting only "true" and "false" (strict mode). */
  def parseStrictBooleanOptionEither(options: Map[String, String], key: String, default: Boolean): Either[String, Boolean] =
    parseBooleanOptionEither(
      options = options,
      key = key,
      default = default,
      trueTokens = Set("true"),
      falseTokens = Set("false"),
      errorMessage = s"--$key must be true or false"
    )

  /** Parses a boolean option accepting "1"/"true"/"yes" and "0"/"false"/"no". */
  def parseBooleanOptionEither(options: Map[String, String], key: String, default: Boolean): Either[String, Boolean] =
    parseBooleanOptionEither(
      options = options,
      key = key,
      default = default,
      trueTokens = Set("1", "true", "yes"),
      falseTokens = Set("0", "false", "no"),
      errorMessage = s"--$key must be a boolean (true/false)"
    )

  /** Parses a boolean option accepting "1"/"true"/"yes"/"on" and "0"/"false"/"no"/"off". */
  def parseExtendedBooleanOptionEither(options: Map[String, String], key: String, default: Boolean): Either[String, Boolean] =
    parseBooleanOptionEither(
      options = options,
      key = key,
      default = default,
      trueTokens = Set("1", "true", "yes", "on"),
      falseTokens = Set("0", "false", "no", "off"),
      errorMessage = s"--$key must be a boolean (true/false)"
    )

  /** Parses `--key=value` arguments into a map. Rejects blank keys and blank values.
    *
    * @return `Right(map)` on success, `Left(errorMessage)` on the first malformed token
    */
  def parseOptions(args: Array[String]): Either[String, Map[String, String]] =
    parseKeyValueOptions(
      args = args,
      invalidTokenMessage = token => s"invalid argument '$token'; expected --key=value",
      blankTokenMessage = token => s"invalid argument '$token'; key/value must be non-empty",
      allowBlankValue = false
    )

  /** Like [[parseOptions]] but allows blank (whitespace-only) values after the `=` sign.
    *
    * Blank values are trimmed to empty strings in the result map. Useful for options
    * where an empty value means "clear this setting" (e.g. `--source=`).
    */
  def parseOptionsAllowBlankValues(args: Array[String]): Either[String, Map[String, String]] =
    parseKeyValueOptions(
      args = args,
      invalidTokenMessage = token => s"invalid option format '$token'; expected --key=value",
      blankTokenMessage = token => s"invalid option key in '$token'",
      allowBlankValue = true
    )

  /** Extracts an integer option, throwing if the value is not a valid integer. */
  def requireIntOption(options: Map[String, String], key: String, default: Int): Int =
    options.get(key) match
      case None => default
      case Some(raw) =>
        raw.toIntOption.getOrElse(
          throw new IllegalArgumentException(s"--$key must be an integer")
        )

  /** Extracts a long option, throwing if the value is not a valid long. */
  def requireLongOption(options: Map[String, String], key: String, default: Long): Long =
    options.get(key) match
      case None => default
      case Some(raw) =>
        raw.toLongOption.getOrElse(
          throw new IllegalArgumentException(s"--$key must be a long")
        )

  /** Extracts a double option, throwing if the value is not a valid number. */
  def requireDoubleOption(options: Map[String, String], key: String, default: Double): Double =
    options.get(key) match
      case None => default
      case Some(raw) =>
        raw.toDoubleOption.getOrElse(
          throw new IllegalArgumentException(s"--$key must be a number")
        )

  /** Extracts a boolean option accepting "1"/"true"/"yes"/"on" and "0"/"false"/"no"/"off".
    * Throws on unrecognized tokens.
    */
  def requireBooleanOption(options: Map[String, String], key: String, default: Boolean): Boolean =
    options.get(key) match
      case None => default
      case Some(raw) =>
        raw.trim.toLowerCase(Locale.ROOT) match
          case "1" | "true" | "yes" | "on" => true
          case "0" | "false" | "no" | "off" => false
          case _ =>
            throw new IllegalArgumentException(s"--$key must be a boolean (true/false)")

  /** Extracts an optional integer: returns `None` if key is absent, throws if value is invalid. */
  def optionalIntOption(options: Map[String, String], key: String): Option[Int] =
    options.get(key).map { raw =>
      raw.toIntOption.getOrElse(
        throw new IllegalArgumentException(s"--$key must be an integer")
      )
    }

  /** Extracts an optional long: returns `None` if key is absent, throws if value is invalid. */
  def optionalLongOption(options: Map[String, String], key: String): Option[Long] =
    options.get(key).map { raw =>
      raw.toLongOption.getOrElse(
        throw new IllegalArgumentException(s"--$key must be a long")
      )
    }

  /** Extracts an optional double: returns `None` if key is absent, throws if value is invalid. */
  def optionalDoubleOption(options: Map[String, String], key: String): Option[Double] =
    options.get(key).map { raw =>
      raw.toDoubleOption.getOrElse(
        throw new IllegalArgumentException(s"--$key must be a number")
      )
    }

  /** Splits a comma-separated string into trimmed, non-empty tokens.
    *
    * @param raw the raw comma-separated string
    * @param key the option name, used in the error message if the result is empty
    * @return a non-empty vector of trimmed tokens
    * @throws IllegalArgumentException if no non-empty tokens remain after splitting and trimming
    */
  def requireCsvTokens(raw: String, key: String): Vector[String] =
    val tokens = raw
      .split(',')
      .toVector
      .map(_.trim)
      .filter(_.nonEmpty)
    require(tokens.nonEmpty, s"$key must be non-empty")
    tokens

  /** Parses a comma-separated string into a vector of positive integers.
    *
    * @throws IllegalArgumentException if any token is not an integer or is not positive
    */
  def requirePositiveIntList(raw: String, key: String): Vector[Int] =
    val values = requireCsvTokens(raw, key).map { token =>
      token.toIntOption.getOrElse(
        throw new IllegalArgumentException(s"--$key values must be integers")
      )
    }
    require(values.forall(_ > 0), s"$key values must be positive")
    values

  /** Optionally extracts a positive integer list from a comma-separated option value. */
  def optionalPositiveIntList(options: Map[String, String], key: String): Option[Vector[Int]] =
    options.get(key).map(requirePositiveIntList(_, key))

  /** Validates that all keys in the options map are in the allowed set.
    *
    * Throws on the first unknown key encountered. Used as a guard at the end of
    * option parsing to catch typos in option names.
    *
    * @param options     the parsed options map
    * @param allowedKeys the set of recognized option keys
    * @throws IllegalArgumentException if any key is not in `allowedKeys`
    */
  def requireNoUnknownOptions(options: Map[String, String], allowedKeys: Set[String]): Unit =
    options.keysIterator.filterNot(allowedKeys.contains).foreach { other =>
      throw new IllegalArgumentException(s"unknown option '$other'")
    }

  /** Parses a 4-character token (e.g. "AcKh") into canonical [[HoleCards]].
    *
    * The token must be exactly 4 characters: two 2-character card tokens concatenated.
    * Cards are placed in canonical (deck-index) order via [[HoleCards.canonical]].
    *
    * @param token a 4-character string like "AcKh" or "7s2d"
    * @return canonically ordered hole cards
    * @throws IllegalArgumentException if the token is not exactly 4 characters or contains invalid cards
    */
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

  /** Parses a range notation string (e.g. "AA,KQs,AJo") into a discrete distribution over hole cards.
    *
    * Delegates to [[RangeParser]] for the actual grammar; wraps `Left` errors as exceptions.
    *
    * @param token the range notation string
    * @return a probability distribution over all matching hole card combos
    * @throws IllegalArgumentException if the range string cannot be parsed
    */
  def parseRangeDistribution(token: String): DiscreteDistribution[HoleCards] =
    RangeParser.parse(token.trim) match
      case Right(dist) => dist
      case Left(err) => throw new IllegalArgumentException(s"failed to parse range token '$token': $err")

  /** Extracts hole cards from an option map, using a default if absent.
    * Returns `Left` with an error message on parse failure.
    */
  def parseHoleCardsOptionEither(
      options: Map[String, String],
      key: String,
      default: String
  ): Either[String, HoleCards] =
    try Right(parseHoleCards(options.getOrElse(key, default)))
    catch
      case e: Exception => Left(s"--$key invalid: ${e.getMessage}")

  /** Extracts a [[Position]] from an option map by case-insensitive name match.
    * Returns `Left` with an error message if the position name is not recognized.
    */
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

  /** Parses a comma-separated list of action tokens (e.g. "fold,call,raise:20") into typed actions.
    *
    * Each token is parsed via [[parseActionTokenEither]]. If `deduplicate` is true,
    * duplicate actions are removed (keeping first occurrence).
    *
    * @param deduplicate if true, remove duplicate actions from the result
    * @return `Right(actions)` on success, `Left(error)` on the first invalid token
    */
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

  /** Formats a double as a 6-wide, 2-decimal-place string (e.g. "  3.14"). */
  def fmt2(value: Double): String =
    String.format(Locale.ROOT, "%6.2f", java.lang.Double.valueOf(value))

  /** Formats a double as an 8-wide, 5-decimal-place string (e.g. " 0.31416"). */
  def fmt5(value: Double): String =
    String.format(Locale.ROOT, "%8.5f", java.lang.Double.valueOf(value))

  /** Core `--key=value` parser used by both [[parseOptions]] and [[parseOptionsAllowBlankValues]].
    *
    * Iterates through the args array, folding into a `Right(Map)` that accumulates parsed
    * key-value pairs. The fold short-circuits to `Left` on the first malformed token.
    *
    * @param args                the raw argument tokens
    * @param invalidTokenMessage function to produce an error message for a malformed token
    * @param blankTokenMessage   function to produce an error message for blank key/value
    * @param allowBlankValue     if true, values that trim to empty are allowed
    * @return `Right(map)` on success, `Left(error)` on the first bad token
    */
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

  /** Generic boolean parser parameterized by the accepted true/false token sets.
    *
    * Converts the raw value to lowercase using `Locale.ROOT` before matching against
    * the token sets, avoiding locale-dependent case conversion issues.
    */
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

  /** Parses a single action token like "fold", "check", "call", or "raise:20.0".
    *
    * Raise tokens use a colon separator (e.g. "raise:20.0") to distinguish from
    * the raise keyword. The amount after the colon must be a positive number.
    *
    * @param token the lowercased action token
    * @return `Right(action)` on success, `Left(error)` on invalid format
    */
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
