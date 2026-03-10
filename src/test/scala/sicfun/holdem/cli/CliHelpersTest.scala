package sicfun.holdem.cli
import sicfun.holdem.types.*

import munit.FunSuite

class CliHelpersTest extends FunSuite:
  test("strict option parsing rejects blank trimmed values") {
    val parsed = CliHelpers.parseOptions(Array("--hero=   "))
    assertEquals(parsed, Left("invalid argument '--hero=   '; key/value must be non-empty"))
  }

  test("relaxed option parsing preserves blank trimmed values") {
    val parsed = CliHelpers.parseOptionsAllowBlankValues(Array("--source=   "))
    assertEquals(parsed, Right(Map("source" -> "")))
  }

  test("relaxed option parsing still requires a token after equals") {
    val parsed = CliHelpers.parseOptionsAllowBlankValues(Array("--source="))
    assertEquals(parsed, Left("invalid option format '--source='; expected --key=value"))
  }

  test("requireOptions throws on malformed tokens") {
    val error = intercept[IllegalArgumentException] {
      CliHelpers.requireOptions(Vector("oops"))
    }
    assertEquals(error.getMessage, "invalid argument 'oops'; expected --key=value")
  }

  test("typed option helpers reject invalid numbers and booleans") {
    val numberError = intercept[IllegalArgumentException] {
      CliHelpers.requireIntOption(Map("runs" -> "nope"), "runs", 1)
    }
    assertEquals(numberError.getMessage, "--runs must be an integer")

    val booleanError = intercept[IllegalArgumentException] {
      CliHelpers.requireBooleanOption(Map("enabled" -> "maybe"), "enabled", default = false)
    }
    assertEquals(booleanError.getMessage, "--enabled must be a boolean (true/false)")
  }

  test("either option helpers decode numeric and boolean values") {
    assertEquals(CliHelpers.parseIntOptionEither(Map("runs" -> "4"), "runs", 1), Right(4))
    assertEquals(CliHelpers.parseLongOptionEither(Map("seed" -> "7"), "seed", 1L), Right(7L))
    assertEquals(CliHelpers.parseOptionalLongOptionEither(Map("budget" -> "9"), "budget"), Right(Some(9L)))
    assertEquals(CliHelpers.parseDoubleOptionEither(Map("alpha" -> "0.35"), "alpha", 0.1), Right(0.35))
    assertEquals(CliHelpers.parseStrictBooleanOptionEither(Map("enabled" -> "true"), "enabled", false), Right(true))
    assertEquals(CliHelpers.parseBooleanOptionEither(Map("enabled" -> "yes"), "enabled", false), Right(true))
    assertEquals(CliHelpers.parseExtendedBooleanOptionEither(Map("enabled" -> "off"), "enabled", true), Right(false))
  }

  test("shared holdem option helpers decode cards positions and actions") {
    assertEquals(
      CliHelpers.parseHoleCardsOptionEither(Map.empty, "hero", "AcKh").map(_.toToken),
      Right("AcKh")
    )
    assertEquals(
      CliHelpers.parsePositionOptionEither(Map("position" -> "button"), "position", Position.BigBlind),
      Right(Position.Button)
    )
    assertEquals(
      CliHelpers.parseCandidateActionsOptionEither(
        Map("candidateActions" -> "fold,call,raise:20,call"),
        "candidateActions",
        "fold",
        deduplicate = true
      ),
      Right(Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(20.0)))
    )
  }

  test("csv and positive integer list helpers reject malformed values") {
    val csvError = intercept[IllegalArgumentException] {
      CliHelpers.requireCsvTokens(" , ", "modes")
    }
    assertEquals(csvError.getMessage, "requirement failed: modes must be non-empty")

    val integerError = intercept[IllegalArgumentException] {
      CliHelpers.requirePositiveIntList("4, nope", "parallelism")
    }
    assertEquals(integerError.getMessage, "--parallelism values must be integers")

    val positiveError = intercept[IllegalArgumentException] {
      CliHelpers.requirePositiveIntList("1, 0", "parallelism")
    }
    assertEquals(positiveError.getMessage, "requirement failed: parallelism values must be positive")
  }

  test("either boolean helpers reject unsupported tokens") {
    assertEquals(
      CliHelpers.parseStrictBooleanOptionEither(Map("enabled" -> "yes"), "enabled", false),
      Left("--enabled must be true or false")
    )
    assertEquals(
      CliHelpers.parseBooleanOptionEither(Map("enabled" -> "maybe"), "enabled", false),
      Left("--enabled must be a boolean (true/false)")
    )
  }

  test("requireNoUnknownOptions rejects unexpected keys") {
    val error = intercept[IllegalArgumentException] {
      CliHelpers.requireNoUnknownOptions(Map("bad" -> "1"), Set("good"))
    }
    assertEquals(error.getMessage, "unknown option 'bad'")
  }
