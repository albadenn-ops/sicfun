package sicfun.holdem.analysis
import sicfun.holdem.types.*
import sicfun.holdem.equity.*
import sicfun.holdem.cli.*

import sicfun.holdem.cli.CliHelpers.{fmt2, fmt5}

/** Validates our canonical exact equity table against published specific preflop matchup odds
  * from cardfight.com (e.g., AA vs KK, TT vs AKs).
  *
  * Unlike [[ComparePublishedPreflopVsRandom]] which compares against "vs random" aggregate odds,
  * this tool validates specific hero-vs-villain matchups where both hands are known. This catches
  * different categories of table bugs (e.g., miskeyed individual matchups vs systematic aggregation errors).
  *
  * Supports both exact 4-card tokens (e.g., "AcAs") and range class tokens (e.g., "AA", "AQs").
  * When range class tokens are used, equity is averaged over all non-overlapping combo pairs
  * in the two classes.
  *
  * Usage:
  * {{{
  * ComparePublishedSpecificMatchups <canonicalExactTablePath>
  * }}}
  */
object ComparePublishedSpecificMatchups:
  /** A published matchup from an external source with known hero vs villain equity.
    *
    * @param label        human-readable label (e.g., "AA vs KK")
    * @param hero         hero hand token: exact 4-char (e.g., "AcAs") or range class (e.g., "AA")
    * @param villain      villain hand token: exact 4-char or range class
    * @param sourceWinPct published hero win percentage
    * @param sourceTiePct published tie percentage
    * @param sourceUrl    URL of the published reference for traceability
    */
  private[holdem] final case class PublishedMatchup(
      label: String,
      hero: String,
      villain: String,
      sourceWinPct: Double,
      sourceTiePct: Double,
      sourceUrl: String
  ):
    /** Standard poker equity: wins plus half of ties. */
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

  /** Entry point. Loads a canonical exact table and prints a comparison for each published matchup.
    *
    * @param args expects a single positional argument: path to a canonical exact binary table
    */
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

  /** Computes our win% and tie% for a hero-vs-villain matchup from the canonical table.
    *
    * Two code paths depending on token format:
    *   1. '''Exact tokens''' (4-char, e.g. "AcAs"): single direct table lookup.
    *   2. '''Range class tokens''' (e.g. "AA", "AKs"): enumerate all combo pairs, skip
    *      overlapping pairs (shared cards), weight by distribution, and normalize.
    *
    * Normalization: since some combo pairs are skipped (card overlap), the raw weights
    * do not sum to 1.0. We divide by `total` (sum of non-overlapping pair weights) to
    * get conditional probabilities given disjoint hands.
    *
    * @param table        canonical exact equity table
    * @param heroToken    hero hand token (exact 4-char or range class)
    * @param villainToken villain hand token (exact 4-char or range class)
    * @return (winPct, tiePct) as percentages in [0, 100]
    */
  private[holdem] def computeWinTiePct(table: HeadsUpEquityCanonicalTable, heroToken: String, villainToken: String): (Double, Double) =
    val heroExact = parseExactHoleCardsOrNone(heroToken)
    val villainExact = parseExactHoleCardsOrNone(villainToken)

    (heroExact, villainExact) match
      // Fast path: both tokens specify exact combos — single table lookup.
      case (Some(hero), Some(villain)) =>
        val result = table.equity(hero, villain)
        (result.win * 100.0, result.tie * 100.0)
      // Slow path: at least one token is a range class — enumerate all combo pairs.
      case _ =>
        val heroDist = CliHelpers.parseRangeDistribution(heroToken)
        val villainDist = CliHelpers.parseRangeDistribution(villainToken)
        var win = 0.0
        var tie = 0.0
        var loss = 0.0
        var total = 0.0

        heroDist.weights.foreach { case (hero, heroWeight) =>
          villainDist.weights.foreach { case (villain, villainWeight) =>
            // Only include pairs where hero and villain do not share any cards.
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
        // Normalize by total weight of non-overlapping pairs to get conditional probabilities.
        val ourWin = (win / total) * 100.0
        val ourTie = (tie / total) * 100.0
        val norm = (win + tie + loss) / total
        require(math.abs(norm - 1.0) <= 1e-10, s"normalization drift for $heroToken vs $villainToken: $norm")
        (ourWin, ourTie)

  /** Attempts to parse a token as exact hole cards (4 characters = rank+suit for each card).
    * Returns None for range class tokens (2-3 characters like "AA" or "AKs").
    */
  private def parseExactHoleCardsOrNone(token: String): Option[HoleCards] =
    if token.trim.length == 4 then Some(CliHelpers.parseHoleCards(token.trim))
    else None
