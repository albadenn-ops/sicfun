package sicfun.holdem

import sicfun.core.Card

import java.util.Locale

/** Compares exact table values against published specific preflop matchup odds.
  *
  * Usage:
  * {{{
  * ComparePublishedSpecificMatchups <canonicalExactTablePath>
  * }}}
  */
object ComparePublishedSpecificMatchups:
  private final case class PublishedMatchup(
      label: String,
      hero: String,    // Either exact 4-char token (e.g. AcAs) or range class token (e.g. AA, AQs)
      villain: String, // Either exact 4-char token or range class token
      sourceWinPct: Double,
      sourceTiePct: Double,
      sourceUrl: String
  ):
    def sourceEqPct: Double = sourceWinPct + (0.5 * sourceTiePct)

  private val Matchups: Vector[PublishedMatchup] = Vector(
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
      hero = "TcTs",
      villain = "AhKh",
      sourceWinPct = 53.86,
      sourceTiePct = 0.41,
      sourceUrl = "https://www.cardfight.com/TT_AKs.html"
    ),
    PublishedMatchup(
      label = "TT vs AKo",
      hero = "TcTs",
      villain = "AhKd",
      sourceWinPct = 56.68,
      sourceTiePct = 0.39,
      sourceUrl = "https://www.cardfight.com/TT_AKo.html"
    ),
    PublishedMatchup(
      label = "TT vs AQo",
      hero = "TcTs",
      villain = "AhQd",
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

  private def computeWinTiePct(table: HeadsUpEquityCanonicalTable, heroToken: String, villainToken: String): (Double, Double) =
    val heroExact = parseExactHoleCardsOrNone(heroToken)
    val villainExact = parseExactHoleCardsOrNone(villainToken)

    (heroExact, villainExact) match
      case (Some(hero), Some(villain)) =>
        val result = table.equity(hero, villain)
        (result.win * 100.0, result.tie * 100.0)
      case _ =>
        val heroDist = parseRangeDistribution(heroToken)
        val villainDist = parseRangeDistribution(villainToken)
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

  private def parseRangeDistribution(token: String) =
    RangeParser.parse(token.trim) match
      case Right(dist) => dist
      case Left(err) => throw new IllegalArgumentException(s"invalid range token '$token': $err")

  private def parseExactHoleCardsOrNone(token: String): Option[HoleCards] =
    if token.trim.length == 4 then Some(parseHoleCards(token.trim))
    else None

  private def parseHoleCards(token: String): HoleCards =
    require(token.length == 4, s"expected 4-char hole cards token like AcAs, got '$token'")
    val c1 = Card.parse(token.substring(0, 2)).getOrElse(
      throw new IllegalArgumentException(s"invalid card token in '$token'")
    )
    val c2 = Card.parse(token.substring(2, 4)).getOrElse(
      throw new IllegalArgumentException(s"invalid card token in '$token'")
    )
    HoleCards.canonical(c1, c2)

  private def fmt2(value: Double): String =
    String.format(Locale.ROOT, "%6.2f", java.lang.Double.valueOf(value))

  private def fmt5(value: Double): String =
    String.format(Locale.ROOT, "%8.5f", java.lang.Double.valueOf(value))
