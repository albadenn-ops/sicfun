package sicfun.holdem.history

import munit.FunSuite

class RegisterSicfunPlayerTest extends FunSuite:
  test("run with --help returns usage") {
    val result = RegisterSicfunPlayer.run(Array("--help"))
    assert(result.isLeft)
    assert(result.left.toOption.get.contains("Usage"))
  }

  test("run with -h returns usage") {
    val result = RegisterSicfunPlayer.run(Array("-h"))
    assert(result.isLeft)
    assert(result.left.toOption.get.contains("Usage"))
  }

  test("run without required args returns error") {
    val result = RegisterSicfunPlayer.run(Array.empty)
    assert(result.isLeft)
    assert(result.left.toOption.get.contains("--store"))
  }

  test("run with missing sourceSite returns error") {
    val result = RegisterSicfunPlayer.run(Array("--store=/tmp/test"))
    assert(result.isLeft)
    assert(result.left.toOption.get.contains("--sourceSite"))
  }

  test("run with missing playerName returns error") {
    val result = RegisterSicfunPlayer.run(Array(
      "--store=/tmp/test",
      "--sourceSite=pokerstars",
      "--sourceName=Villain"
    ))
    assert(result.isLeft)
    assert(result.left.toOption.get.contains("--playerName"))
  }

  test("RunSummary fields are accessible") {
    val summary = RegisterSicfunPlayer.RunSummary(
      playerUid = "uid-1",
      profileUid = "prof-1",
      site = "sicfun@localhost",
      playerName = "TestBot",
      handsObserved = 42
    )
    assertEquals(summary.playerUid, "uid-1")
    assertEquals(summary.handsObserved, 42)
  }
