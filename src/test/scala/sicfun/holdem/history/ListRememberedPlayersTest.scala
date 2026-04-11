package sicfun.holdem.history

import munit.FunSuite

/** Tests for the [[ListRememberedPlayers]] CLI tool's argument parsing.
  *
  * Validates that:
  *   - `--help` and `-h` return usage text as a Left result
  *   - Missing `--store` option produces an error mentioning the flag
  *   - Empty `--store=` value is rejected
  *
  * These tests exercise only CLI parsing, not the actual listing logic
  * (which requires a populated store file or database).
  */
class ListRememberedPlayersTest extends FunSuite:
  test("run with --help returns usage") {
    val result = ListRememberedPlayers.run(Array("--help"))
    assert(result.isLeft)
    assert(result.left.toOption.get.contains("Usage"))
  }

  test("run with -h returns usage") {
    val result = ListRememberedPlayers.run(Array("-h"))
    assert(result.isLeft)
    assert(result.left.toOption.get.contains("Usage"))
  }

  test("run without --store returns error") {
    val result = ListRememberedPlayers.run(Array.empty)
    assert(result.isLeft)
    assert(result.left.toOption.get.contains("--store"))
  }

  test("run with empty --store returns error") {
    val result = ListRememberedPlayers.run(Array("--store="))
    assert(result.isLeft)
    assert(result.left.toOption.get.contains("--store"))
  }
