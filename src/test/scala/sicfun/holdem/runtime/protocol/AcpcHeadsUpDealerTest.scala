package sicfun.holdem.runtime.protocol

import munit.FunSuite

import java.nio.file.Paths

/** Pure-function tests for [[AcpcHeadsUpDealer]] internals.
  *
  * The dealer itself spawns external player processes over TCP; integration testing it
  * end-to-end is heavyweight. These tests pin the small reusable pieces:
  *
  *   - [[AcpcHeadsUpDealer.buildCommand]]: shell-launch dispatch by script extension.
  *   - [[AcpcHeadsUpDealer.showdownValue]]: heads-up signed chip resolution at showdown.
  *   - [[AcpcHeadsUpDealer.bbPer100]]: standard bb/100 win-rate conversion.
  *
  * These functions are private[protocol] so the tests live in the same package.
  */
class AcpcHeadsUpDealerTest extends FunSuite:

  // --- buildCommand: extension-driven shell selection ---

  test("buildCommand dispatches .cmd via cmd.exe /c") {
    val cmd = AcpcHeadsUpDealer.buildCommand(Paths.get("scripts/player.cmd"), "127.0.0.1", 18001)
    assertEquals(cmd.head, "cmd.exe")
    assertEquals(cmd(1), "/c")
    assert(cmd(2).endsWith("player.cmd"), s"unexpected script in command: ${cmd(2)}")
    assertEquals(cmd(3), "127.0.0.1")
    assertEquals(cmd(4), "18001")
  }

  test("buildCommand dispatches .bat via cmd.exe /c") {
    val cmd = AcpcHeadsUpDealer.buildCommand(Paths.get("legacy/player.bat"), "host.local", 9000)
    assertEquals(cmd.head, "cmd.exe")
    assertEquals(cmd(1), "/c")
    assertEquals(cmd(3), "host.local")
    assertEquals(cmd(4), "9000")
  }

  test("buildCommand dispatches .ps1 via powershell -ExecutionPolicy Bypass") {
    val cmd = AcpcHeadsUpDealer.buildCommand(Paths.get("scripts/run-bot.ps1"), "0.0.0.0", 17000)
    assertEquals(cmd.head, "powershell")
    assertEquals(cmd(1), "-ExecutionPolicy")
    assertEquals(cmd(2), "Bypass")
    assertEquals(cmd(3), "-File")
    assert(cmd(4).endsWith("run-bot.ps1"))
    assertEquals(cmd(5), "0.0.0.0")
    assertEquals(cmd(6), "17000")
  }

  test("buildCommand runs bare executables directly") {
    val cmd = AcpcHeadsUpDealer.buildCommand(Paths.get("/usr/local/bin/player"), "10.0.0.1", 18020)
    assertEquals(cmd.length, 3)
    assert(cmd.head.endsWith("player"))
    assertEquals(cmd(1), "10.0.0.1")
    assertEquals(cmd(2), "18020")
  }

  test("buildCommand dispatches by lowercased extension (case insensitive)") {
    val cmdUpper = AcpcHeadsUpDealer.buildCommand(Paths.get("BOT.CMD"), "h", 1)
    assertEquals(cmdUpper.head, "cmd.exe")
    val ps1Upper = AcpcHeadsUpDealer.buildCommand(Paths.get("Bot.PS1"), "h", 1)
    assertEquals(ps1Upper.head, "powershell")
  }

  // --- showdownValue: heads-up signed chip math ---

  test("showdownValue returns +opponent contribution when player wins") {
    val spent = Vector(1500, 2000)
    val rank = Vector(7000, 5000) // player 0 has higher rank
    assertEquals(
      AcpcHeadsUpDealer.showdownValue(spent, rank, playerIdx = 0),
      2000.0,
      "winner gets opponent contribution"
    )
  }

  test("showdownValue returns -own contribution when player loses") {
    val spent = Vector(1500, 2000)
    val rank = Vector(7000, 5000)
    assertEquals(
      AcpcHeadsUpDealer.showdownValue(spent, rank, playerIdx = 1),
      -2000.0,
      "loser is out their own contribution"
    )
  }

  test("showdownValue returns half the contribution-difference when ranks tie") {
    val spent = Vector(1000, 1500)
    val rank = Vector(6000, 6000)
    assertEquals(
      AcpcHeadsUpDealer.showdownValue(spent, rank, playerIdx = 0),
      250.0,
      "tied with smaller spend nets +half-difference"
    )
    assertEquals(
      AcpcHeadsUpDealer.showdownValue(spent, rank, playerIdx = 1),
      -250.0,
      "tied with larger spend nets -half-difference"
    )
  }

  test("showdownValue with equal spend and tied rank is zero") {
    val spent = Vector(1000, 1000)
    val rank = Vector(6000, 6000)
    assertEquals(AcpcHeadsUpDealer.showdownValue(spent, rank, playerIdx = 0), 0.0)
    assertEquals(AcpcHeadsUpDealer.showdownValue(spent, rank, playerIdx = 1), 0.0)
  }

  test("showdownValue requires exactly two-player vectors") {
    intercept[IllegalArgumentException] {
      AcpcHeadsUpDealer.showdownValue(Vector(100), Vector(500), playerIdx = 0)
    }
    intercept[IllegalArgumentException] {
      AcpcHeadsUpDealer.showdownValue(Vector(100, 200, 300), Vector(500, 600, 700), playerIdx = 0)
    }
  }

  // --- bbPer100: win-rate conversion ---

  test("bbPer100 is zero when no hands have been played") {
    assertEquals(AcpcHeadsUpDealer.bbPer100(netChips = 5000, hands = 0), 0.0)
    assertEquals(AcpcHeadsUpDealer.bbPer100(netChips = -5000, hands = 0), 0.0)
  }

  test("bbPer100 normalizes against BigBlindChips=100 across hand count") {
    // BigBlindChips is 100 in the dealer; the metric is "big blinds won per 100 hands".
    // netChips=10000 / 100 chips/bb / 100 hands * 100 = 100 bb/100.
    assertEquals(AcpcHeadsUpDealer.bbPer100(netChips = 10000.0, hands = 100), 100.0)
    // Twice the chips over twice the hands -> the same per-100-hand rate.
    assertEquals(AcpcHeadsUpDealer.bbPer100(netChips = 20000.0, hands = 200), 100.0)
    // Twice the chips over the same hands -> twice the rate.
    assertEquals(AcpcHeadsUpDealer.bbPer100(netChips = 20000.0, hands = 100), 200.0)
    // Negative net chips produce negative bb/100.
    assertEquals(AcpcHeadsUpDealer.bbPer100(netChips = -5000.0, hands = 100), -50.0)
  }
