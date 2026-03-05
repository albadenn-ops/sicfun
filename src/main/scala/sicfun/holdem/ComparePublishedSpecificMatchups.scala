package sicfun.holdem

import CliHelpers.{fmt2, fmt5}

/** Compares exact table values against published specific preflop matchup odds.
  *
  * Usage:
  * {{{
  * ComparePublishedSpecificMatchups <canonicalExactTablePath>
  * }}}
  */
object ComparePublishedSpecificMatchups:
  private[holdem] final case class PublishedMatchup(
      label: String,
      hero: String,    // Either exact 4-char token (e.g. AcAs) or range class token (e.g. AA, AQs)
      villain: String, // Either exact 4-char token or range class token
      sourceWinPct: Double,
      sourceTiePct: Double,
      sourceUrl: String
  ):
    def sourceEqPct: Double = sourceWinPct + (0.5 * sourceTiePct)

  private[holdem] val Matchups: Vector[PublishedMatchup] = Vector(
    PublishedMatchup(
      label = "AA vs KK",
      hero = "AA",
      villain = "KK",
      sourceWinPct = 81.71,
      sourceTiePct = 0.46,
      sourceUrl = "https://www.cardfight.com/AA_KK.html"
    ),
    PublishedMatchup(
      label = "AA vs AQs",
      hero = "AA",
      villain = "AQs",
      sourceWinPct = 86.84,
      sourceTiePct = 1.24,
      sourceUrl = "https://www.cardfight.com/AA_AQs.html"
    ),
    PublishedMatchup(
      label = "AA vs AJs",
      hero = "AA",
      villain = "AJs",
      sourceWinPct = 86.45,
      sourceTiePct = 1.23,
      sourceUrl = "https://www.cardfight.com/AA_AJs.html"
    ),
    PublishedMatchup(
      label = "AA vs ATs",
      hero = "AA",
      villain = "ATs",
      sourceWinPct = 86.05,
      sourceTiePct = 1.22,
      sourceUrl = "https://www.cardfight.com/AA_ATs.html"
    ),
    PublishedMatchup(
      label = "TT vs AKs",
      hero = "TT",
      villain = "AKs",
      sourceWinPct = 53.86,
      sourceTiePct = 0.41,
      sourceUrl = "https://www.cardfight.com/TT_AKs.html"
    ),
    PublishedMatchup(
      label = "TT vs AKo",
      hero = "TT",
      villain = "AKo",
      sourceWinPct = 56.68,
      sourceTiePct = 0.39,
      sourceUrl = "https://www.cardfight.com/TT_AKo.html"
    ),
    PublishedMatchup(
      label = "TT vs AQo",
      hero = "TT",
      villain = "AQo",
      sourceWinPct = 56.76,
      sourceTiePct = 0.38,
      sourceUrl = "https://www.cardfight.com/TT_AQo.html"
    )
  )

  def main(args: Array[String]): Unit =
    if args.length < 1 then
      System.err.println("Usage: ComparePublishedSpecificMatchups <canonicalExactTablePath>")
      sys.exit(1)

    val (table, meta) = HeadsUpEquityCanonicalTableIO.readWithMeta(args(0))
    if !meta.canonical then
      throw new IllegalArgumentException("expected canonical table input")
    if meta.mode != "exact" then
      throw new IllegalArgumentException(s"expected exact table input, got mode=${meta.mode}")

    println("Specific matchup validation vs published odds")
    println("label sourceWin sourceTie sourceEq ourWin ourTie ourEq dWin(pp) dTie(pp) dEq(pp) source")

    Matchups.foreach { row =>
      val (ourWin, ourTie) = computeWinTiePct(table, row.hero, row.villain)
      val ourEq = ourWin + (0.5 * ourTie)
      val dWin = ourWin - row.sourceWinPct
      val dTie = ourTie - row.sourceTiePct
      val dEq = ourEq - row.sourceEqPct

      println(
        s"${row.label} " +
          f"${fmt2(row.sourceWinPct)} ${fmt2(row.sourceTiePct)} ${fmt2(row.sourceEqPct)} " +
          f"${fmt5(ourWin)} ${fmt5(ourTie)} ${fmt5(ourEq)} ${fmt5(dWin)} ${fmt5(dTie)} ${fmt5(dEq)} " +
          s"${row.sourceUrl}"
      )
    }

  private[holdem] def computeWinTiePct(table: HeadsUpEquityCanonicalTable, heroToken: String, villainToken: String): (Double, Double) =
    val heroExact = parseExactHoleCardsOrNone(heroToken)
    val villainExact = parseExactHoleCardsOrNone(villainToken)

    (heroExact, villainExact) match
      case (Some(hero), Some(villain)) =>
        val result = table.equity(hero, villain)
        (result.win * 100.0, result.tie * 100.0)
      case _ =>
        val heroDist = CliHelpers.parseRangeDistribution(heroToken)
        val villainDist = CliHelpers.parseRangeDistribution(villainToken)
        var win = 0.0
        var tie = 0.0
        var loss = 0.0
        var total = 0.0

        heroDist.weights.foreach { case (hero, heroWeight) =>
          villainDist.weights.foreach { case (villain, villainWeight) =>
            if HoleCardsIndex.areDisjoint(hero, villain) then
              val pairWeight = heroWeight * villainWeight
              val result = table.equity(hero, villain)
              win += pairWeight * result.win
              tie += pairWeight * result.tie
              loss += pairWeight * result.loss
              total += pairWeight
          }
        }

        require(total > 0.0, s"no non-overlapping combo pairs for $heroToken vs $villainToken")
        val ourWin = (win / total) * 100.0
        val ourTie = (tie / total) * 100.0
        val norm = (win + tie + loss) / total
        require(math.abs(norm - 1.0) <= 1e-10, s"normalization drift for $heroToken vs $villainToken: $norm")
        (ourWin, ourTie)

  private def parseExactHoleCardsOrNone(token: String): Option[HoleCards] =
    if token.trim.length == 4 then Some(CliHelpers.parseHoleCards(token.trim))
    else None
