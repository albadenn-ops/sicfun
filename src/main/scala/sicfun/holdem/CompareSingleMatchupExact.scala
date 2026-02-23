package sicfun.holdem

import sicfun.core.Card

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

    val hero = parseHoleCards(args(1))
    val villain = parseHoleCards(args(2))
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

    println(s"hero=${handToken(hero)} villain=${handToken(villain)}")
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

  private def parseHoleCards(token: String): HoleCards =
    val t = token.trim
    require(t.length == 4, s"expected 4-char token like AcAs, got '$token'")
    val c1 = Card.parse(t.substring(0, 2)).getOrElse(throw new IllegalArgumentException(s"invalid card in '$token'"))
    val c2 = Card.parse(t.substring(2, 4)).getOrElse(throw new IllegalArgumentException(s"invalid card in '$token'"))
    HoleCards.canonical(c1, c2)

  private def handToken(hand: HoleCards): String =
    s"${cardToken(hand.first)}${cardToken(hand.second)}"

  private def cardToken(card: Card): String =
    val rank =
      card.rank match
        case sicfun.core.Rank.Two => "2"
        case sicfun.core.Rank.Three => "3"
        case sicfun.core.Rank.Four => "4"
        case sicfun.core.Rank.Five => "5"
        case sicfun.core.Rank.Six => "6"
        case sicfun.core.Rank.Seven => "7"
        case sicfun.core.Rank.Eight => "8"
        case sicfun.core.Rank.Nine => "9"
        case sicfun.core.Rank.Ten => "T"
        case sicfun.core.Rank.Jack => "J"
        case sicfun.core.Rank.Queen => "Q"
        case sicfun.core.Rank.King => "K"
        case sicfun.core.Rank.Ace => "A"
    val suit =
      card.suit match
        case sicfun.core.Suit.Clubs => "c"
        case sicfun.core.Suit.Diamonds => "d"
        case sicfun.core.Suit.Hearts => "h"
        case sicfun.core.Suit.Spades => "s"
    rank + suit

  private def fmtPct(value01: Double): String =
    String.format(Locale.ROOT, "%.8f%%", java.lang.Double.valueOf(value01 * 100.0))
