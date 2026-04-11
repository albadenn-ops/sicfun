package sicfun.holdem.types

import munit.FunSuite

/**
  * Tests for [[PokerFormatting]] shared display utilities.
  *
  * Validates:
  *   - '''renderAction''': Correct string output for each [[PokerAction]] variant,
  *     including raise amounts formatted to 2 decimal places.
  *   - '''fmtDouble''': Correct precision formatting and locale-independent decimal separator
  *     (period, not comma, regardless of JVM locale).
  *   - '''roundedChips''': Rounds to nearest 50 with a minimum of 50,
  *     testing boundary values at 0, 25, 74, 75, 100, 125.
  *   - '''heroModeLabel''': Returns lowercase "adaptive" or "gto" for each [[HeroMode]].
  */
class PokerFormattingTest extends FunSuite:

  test("renderAction formats Fold"):
    assertEquals(PokerFormatting.renderAction(PokerAction.Fold), "Fold")

  test("renderAction formats Check"):
    assertEquals(PokerFormatting.renderAction(PokerAction.Check), "Check")

  test("renderAction formats Call"):
    assertEquals(PokerFormatting.renderAction(PokerAction.Call), "Call")

  test("renderAction formats Raise with amount"):
    assertEquals(PokerFormatting.renderAction(PokerAction.Raise(2.50)), "Raise:2.50")

  test("renderAction formats Raise with integer-like amount"):
    assertEquals(PokerFormatting.renderAction(PokerAction.Raise(3.00)), "Raise:3.00")

  test("fmtDouble formats with specified precision"):
    assertEquals(PokerFormatting.fmtDouble(3.14159, 2), "3.14")
    assertEquals(PokerFormatting.fmtDouble(3.14159, 4), "3.1416")
    assertEquals(PokerFormatting.fmtDouble(0.0, 3), "0.000")

  test("fmtDouble uses Locale.ROOT to avoid comma decimals"):
    assertEquals(PokerFormatting.fmtDouble(1234.5, 1), "1234.5")

  test("roundedChips rounds to nearest 50 with minimum 50"):
    assertEquals(PokerFormatting.roundedChips(0.0), 50)
    assertEquals(PokerFormatting.roundedChips(25.0), 50)
    assertEquals(PokerFormatting.roundedChips(74.0), 50)
    assertEquals(PokerFormatting.roundedChips(75.0), 100)
    assertEquals(PokerFormatting.roundedChips(100.0), 100)
    assertEquals(PokerFormatting.roundedChips(125.0), 150)

  test("heroModeLabel returns lowercase string"):
    assertEquals(PokerFormatting.heroModeLabel(HeroMode.Adaptive), "adaptive")
    assertEquals(PokerFormatting.heroModeLabel(HeroMode.Gto), "gto")
