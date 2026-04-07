package sicfun.holdem.analysis
import sicfun.holdem.equity.*
import sicfun.holdem.cli.*

import java.util.Locale

/** Compares one specific matchup using:
  *   1) table lookup from a canonical exact table
  *   2) direct JVM exact enumeration
  *
  * Usage:
  * {{{
  * CompareSingleMatchupExact <canonicalExactTablePath> <hero4> <villain4>
  * Example: CompareSingleMatchupExact data/heads-up-equity-canonical-exact-cuda-full.bin TcTs AhKh
  * }}}
  */
object CompareSingleMatchupExact:
  /** Entry point. Compares a pre-computed canonical table value against a fresh JVM exact
    * enumeration for one specific hero-villain matchup.
    *
    * This is a diagnostic tool for verifying individual table entries. Any non-zero delta
    * indicates a table generation bug, since both paths enumerate all 48-choose-5 = 1,712,304
    * possible boards exactly.
    *
    * @param args expects 3 positional arguments: table path, hero 4-char token, villain 4-char token
    */
  def main(args: Array[String]): Unit =
    if args.length < 3 then
      System.err.println("Usage: CompareSingleMatchupExact <canonicalExactTablePath> <hero4> <villain4>")
      sys.exit(1)

    val (table, meta) = HeadsUpEquityCanonicalTableIO.readWithMeta(args(0))
    if !meta.canonical then throw new IllegalArgumentException("expected canonical table input")
    if meta.mode != "exact" then throw new IllegalArgumentException(s"expected exact table input, got mode=${meta.mode}")

    val hero = CliHelpers.parseHoleCards(args(1))
    val villain = CliHelpers.parseHoleCards(args(2))
    if !HoleCardsIndex.areDisjoint(hero, villain) then
      throw new IllegalArgumentException("hero and villain overlap")

    // Path 1: table lookup (pre-computed, possibly on GPU).
    val tableResult = table.equity(hero, villain)
    // Path 2: fresh JVM CPU exact enumeration (always correct if the evaluator is correct).
    val cpuResult =
      HeadsUpEquityTable.computeEquityDeterministic(
        hero = hero,
        villain = villain,
        mode = HeadsUpEquityTable.Mode.Exact,
        monteCarloSeedBase = 1L, // Not used in exact mode, but required by API.
        keyMaterial = 1L
      )

    println(s"hero=${hero.toToken} villain=${villain.toToken}")
    printRow("table", tableResult.win, tableResult.tie, tableResult.loss)
    printRow("cpu", cpuResult.win, cpuResult.tie, cpuResult.loss)
    // Delta in percentage points — should be 0.0 if table and CPU agree.
    printRow(
      "delta(pp)",
      (tableResult.win - cpuResult.win) * 100.0,
      (tableResult.tie - cpuResult.tie) * 100.0,
      (tableResult.loss - cpuResult.loss) * 100.0
    )

  /** Prints a labeled row of win/tie/loss/equity values. */
  private def printRow(label: String, win: Double, tie: Double, loss: Double): Unit =
    val equity = win + (tie / 2.0)
    println(
      f"$label%-10s win=${fmtPct(win)} tie=${fmtPct(tie)} loss=${fmtPct(loss)} equity=${fmtPct(equity)}"
    )

  /** Formats a fraction in [0,1] as a percentage string with 8 decimal places. */
  private def fmtPct(value01: Double): String =
    String.format(Locale.ROOT, "%.8f%%", java.lang.Double.valueOf(value01 * 100.0))
