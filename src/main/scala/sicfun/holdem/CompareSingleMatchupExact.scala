package sicfun.holdem

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

    val tableResult = table.equity(hero, villain)
    val cpuResult =
      HeadsUpEquityTable.computeEquityDeterministic(
        hero = hero,
        villain = villain,
        mode = HeadsUpEquityTable.Mode.Exact,
        monteCarloSeedBase = 1L,
        keyMaterial = 1L
      )

    println(s"hero=${hero.toToken} villain=${villain.toToken}")
    printRow("table", tableResult.win, tableResult.tie, tableResult.loss)
    printRow("cpu", cpuResult.win, cpuResult.tie, cpuResult.loss)
    printRow(
      "delta(pp)",
      (tableResult.win - cpuResult.win) * 100.0,
      (tableResult.tie - cpuResult.tie) * 100.0,
      (tableResult.loss - cpuResult.loss) * 100.0
    )

  private def printRow(label: String, win: Double, tie: Double, loss: Double): Unit =
    val equity = win + (tie / 2.0)
    println(
      f"$label%-10s win=${fmtPct(win)} tie=${fmtPct(tie)} loss=${fmtPct(loss)} equity=${fmtPct(equity)}"
    )

  private def fmtPct(value01: Double): String =
    String.format(Locale.ROOT, "%.8f%%", java.lang.Double.valueOf(value01 * 100.0))
