package sicfun.holdem.analysis
import sicfun.holdem.types.*
import sicfun.holdem.equity.*
import sicfun.holdem.cli.*

import java.util.Locale

/** Compares this project's exact canonical table against published heads-up preflop
  * hand-vs-random benchmarks.
  *
  * Source dataset currently embedded:
  *   - caniwin.com Texas Holdem Heads-Up Preflop Odds (truncated percentages)
  *
  * Usage:
  * {{{
  * ComparePublishedPreflopVsRandom <canonicalExactTablePath>
  * }}}
  */
object ComparePublishedPreflopVsRandom:
  /** A single row from the published preflop odds reference table.
    *
    * @param rank    ordinal rank from the source (1 = best starting hand)
    * @param hand    hand class token as published, e.g. "AAo", "AKs"
    * @param winPct  published win percentage (truncated, not rounded)
    * @param tiePct  published tie percentage (truncated, not rounded)
    */
  private[holdem] final case class PublishedRow(
      rank: Int,
      hand: String,
      winPct: Double,
      tiePct: Double
  ):
    /** Equity = win + half of ties (standard poker equity formula). */
    def sourceEquityPct: Double = winPct + (0.5 * tiePct)

  /** Computed equity for one hand class under both aggregation models, paired with its published reference.
    *
    * "Combo-uniform" (Model A) weights each remaining deck combo equally.
    * "Class-uniform" (Model B) weights each of 169 hand classes equally, then combos within each class equally.
    * Deltas are in percentage points (pp) for direct comparison.
    *
    * @param published    the reference row from caniwin.com
    * @param comboWinPct  our computed win% under combo-uniform model
    * @param comboTiePct  our computed tie% under combo-uniform model
    * @param classWinPct  our computed win% under class-uniform model
    * @param classTiePct  our computed tie% under class-uniform model
    */
  private[holdem] final case class ComputedRow(
      published: PublishedRow,
      comboWinPct: Double,
      comboTiePct: Double,
      classWinPct: Double,
      classTiePct: Double
  ):
    def comboEquityPct: Double = comboWinPct + (0.5 * comboTiePct)
    def classEquityPct: Double = classWinPct + (0.5 * classTiePct)
    /** Delta in percentage points between our combo-uniform win% and the published win%. */
    def deltaComboWinPctPoints: Double = comboWinPct - published.winPct
    def deltaComboTiePctPoints: Double = comboTiePct - published.tiePct
    def deltaComboEquityPctPoints: Double = comboEquityPct - published.sourceEquityPct
    def deltaClassWinPctPoints: Double = classWinPct - published.winPct
    def deltaClassTiePctPoints: Double = classTiePct - published.tiePct
    def deltaClassEquityPctPoints: Double = classEquityPct - published.sourceEquityPct

  // Published values from:
  // https://caniwin.com/texasholdem/preflop/heads-up.php
  // The page states values are truncated (not rounded).
  // Snapshot captured circa 2026-02 from the live page.
  // As of 2026-03-23, the live page shows slight drift (e.g. AKo: 64.46/1.70 vs 64.55/1.39 here).
  // Treat as supporting equity evidence, not exact current reference.
  private[holdem] val PublishedTopRows: Vector[PublishedRow] = Vector(
    PublishedRow(1, "AAo", 84.93, 0.54),
    PublishedRow(2, "KKo", 82.11, 0.55),
    PublishedRow(3, "QQo", 79.63, 0.58),
    PublishedRow(4, "JJo", 77.15, 0.63),
    PublishedRow(5, "TTo", 74.66, 0.70),
    PublishedRow(6, "99o", 71.66, 0.78),
    PublishedRow(7, "88o", 68.71, 0.89),
    PublishedRow(8, "AKs", 66.21, 1.65),
    PublishedRow(9, "77o", 65.72, 1.02),
    PublishedRow(10, "AQs", 65.31, 1.79),
    PublishedRow(11, "AKo", 64.55, 1.39),
    PublishedRow(12, "AJs", 64.40, 1.92),
    PublishedRow(13, "66o", 62.72, 1.17),
    PublishedRow(14, "ATs", 62.39, 2.06),
    PublishedRow(15, "AQo", 62.21, 1.50),
    PublishedRow(16, "AJo", 61.80, 1.64),
    PublishedRow(17, "KQs", 61.79, 2.06)
  )

  /** Entry point. Loads a canonical exact table, computes equity under both models for each
    * published row, prints a comparison table, and reports aggregate error statistics.
    *
    * @param args expects a single positional argument: the path to a canonical exact binary table
    */
  def main(args: Array[String]): Unit =
    if args.length < 1 then
      System.err.println("Usage: ComparePublishedPreflopVsRandom <canonicalExactTablePath>")
      sys.exit(1)

    val (table, meta) = HeadsUpEquityCanonicalTableIO.readWithMeta(args(0))
    if !meta.canonical then
      throw new IllegalArgumentException("expected canonical table input")
    if meta.mode != "exact" then
      throw new IllegalArgumentException(s"expected exact table input, got mode=${meta.mode}")

    // Pre-compute the full 169-class villain distributions once (used by Model B for every row).
    val villainClassDistributions = allHandClassTokens.map(CliHelpers.parseRangeDistribution)
    val computed = PublishedTopRows.map(row => computeRow(table, row, villainClassDistributions))

    println("Published source: https://caniwin.com/texasholdem/preflop/heads-up.php")
    println(
      "Note: source values are truncated (not rounded), so tiny differences are expected even when exact."
    )
    println(
      "Model A = random dealt combo from remaining deck.")
    println(
      "Model B = random hand class (169-class uniform), then random disjoint combo in that class.")
    println(
      "rank hand pubWin pubTie pubEq Awin Atie Aeq dAeq(pp) Bwin Btie Beq dBeq(pp)"
    )
    computed.foreach { row =>
      println(
        f"${row.published.rank}%4d ${row.published.hand}%-4s " +
          f"${CliHelpers.fmt2(row.published.winPct)} ${CliHelpers.fmt2(row.published.tiePct)} ${CliHelpers.fmt2(row.published.sourceEquityPct)} " +
          f"${CliHelpers.fmt5(row.comboWinPct)} ${CliHelpers.fmt5(row.comboTiePct)} ${CliHelpers.fmt5(row.comboEquityPct)} ${CliHelpers.fmt5(row.deltaComboEquityPctPoints)} " +
          f"${CliHelpers.fmt5(row.classWinPct)} ${CliHelpers.fmt5(row.classTiePct)} ${CliHelpers.fmt5(row.classEquityPct)} ${CliHelpers.fmt5(row.deltaClassEquityPctPoints)}"
      )
    }

    // Aggregate error metrics: worst-case and average absolute equity delta for each model.
    // Small deltas (< 0.01 pp) are expected because the published source truncates rather than rounds.
    val maxAbsDeltaComboEq = computed.map(r => math.abs(r.deltaComboEquityPctPoints)).max
    val meanAbsDeltaComboEq = computed.map(r => math.abs(r.deltaComboEquityPctPoints)).sum / computed.length.toDouble
    val maxAbsDeltaClassEq = computed.map(r => math.abs(r.deltaClassEquityPctPoints)).max
    val meanAbsDeltaClassEq = computed.map(r => math.abs(r.deltaClassEquityPctPoints)).sum / computed.length.toDouble
    println()
    println(f"ModelA maxAbsDeltaEq(pp)=$maxAbsDeltaComboEq%.6f")
    println(f"ModelA meanAbsDeltaEq(pp)=$meanAbsDeltaComboEq%.6f")
    println(f"ModelB maxAbsDeltaEq(pp)=$maxAbsDeltaClassEq%.6f")
    println(f"ModelB meanAbsDeltaEq(pp)=$meanAbsDeltaClassEq%.6f")

  /** Computes our equity for one published hand class under both aggregation models.
    *
    * Normalizes the published token (e.g. "AAo" -> "AA"), parses it into a hero
    * distribution (which may contain multiple combos for a class like "AKs"),
    * then runs both Model A (combo-uniform) and Model B (class-uniform) aggregation.
    *
    * @param table                    the canonical exact equity lookup table
    * @param published                one row from the published reference data
    * @param villainClassDistributions pre-computed distributions for all 169 hand classes
    * @return a [[ComputedRow]] with equity under both models and deltas vs published
    */
  private[holdem] def computeRow(
      table: HeadsUpEquityCanonicalTable,
      published: PublishedRow,
      villainClassDistributions: Vector[sicfun.core.DiscreteDistribution[HoleCards]]
  ): ComputedRow =
    val token = normalizeHandToken(published.hand)
    val heroDist = CliHelpers.parseRangeDistribution(token)

    val (comboWin, comboTie) = computeAgainstComboUniform(table, heroDist)
    val (classWin, classTie) = computeAgainstClassUniform(table, heroDist, villainClassDistributions)

    ComputedRow(
      published = published,
      comboWinPct = comboWin * 100.0,
      comboTiePct = comboTie * 100.0,
      classWinPct = classWin * 100.0,
      classTiePct = classTie * 100.0
    )

  /** Model A: combo-uniform aggregation.
    *
    * For each hero combo in the distribution, enumerates ALL 1326 possible villain combos
    * that do not share cards with the hero. Each disjoint villain combo is weighted equally
    * (1 / count of disjoint villains). The hero combos themselves are weighted by their
    * distribution weights (uniform within a hand class, e.g. 6 combos for a pocket pair).
    *
    * This models the opponent as holding a uniformly random dealt hand from the remaining deck.
    *
    * @param table    canonical exact equity table for lookups
    * @param heroDist hero hole-cards distribution (all combos in the hand class)
    * @return (winFraction, tieFraction) as fractions in [0,1]
    */
  private[holdem] def computeAgainstComboUniform(
      table: HeadsUpEquityCanonicalTable,
      heroDist: sicfun.core.DiscreteDistribution[HoleCards]
  ): (Double, Double) =
    var win = 0.0
    var tie = 0.0
    var loss = 0.0

    heroDist.weights.foreach { case (hero, heroWeight) =>
      // Find all villain hands that do not share any cards with this hero combo.
      val villains = HoleCardsIndex.all.filter(v => HoleCardsIndex.areDisjoint(hero, v))
      val villainWeight = 1.0 / villains.size.toDouble
      villains.foreach { villain =>
        val result = table.equity(hero, villain)
        val w = heroWeight * villainWeight
        win += w * result.win
        tie += w * result.tie
        loss += w * result.loss
      }
    }

    // Sanity check: win + tie + loss should sum to 1.0 for a properly normalized distribution.
    val total = win + tie + loss
    if math.abs(total - 1.0) > 1e-10 then
      throw new IllegalStateException(s"combo-uniform probability total drift: $total")

    (win, tie)

  /** Model B: class-uniform aggregation.
    *
    * Each of 169 hand classes gets equal weight (1/169). Within a class, villain combos are
    * weighted uniformly after removing those that share cards with the hero. This models the
    * opponent as first choosing a hand class uniformly at random, then a specific combo
    * within that class. This can yield different results from Model A because hand classes
    * have varying numbers of combos (e.g. pocket pairs have 6 combos, suited hands have 4,
    * offsuit non-pairs have 12).
    *
    * @param table                    canonical exact equity table for lookups
    * @param heroDist                 hero hole-cards distribution
    * @param villainClassDistributions pre-computed distributions for all 169 hand classes
    * @return (winFraction, tieFraction) as fractions in [0,1]
    */
  private def computeAgainstClassUniform(
      table: HeadsUpEquityCanonicalTable,
      heroDist: sicfun.core.DiscreteDistribution[HoleCards],
      villainClassDistributions: Vector[sicfun.core.DiscreteDistribution[HoleCards]]
  ): (Double, Double) =
    var win = 0.0
    var tie = 0.0
    var loss = 0.0

    // Each hand class gets equal weight (1/169).
    val classWeight = 1.0 / villainClassDistributions.length.toDouble

    heroDist.weights.foreach { case (hero, heroWeight) =>
      villainClassDistributions.foreach { villainClass =>
        // Filter to only villain combos that do not share cards with hero,
        // then re-normalize within that class (conditional probability).
        val disjoint = villainClass.weights.iterator.filter { case (villain, _) =>
          HoleCardsIndex.areDisjoint(hero, villain)
        }.toVector
        val disjointTotal = disjoint.map(_._2).sum
        if disjointTotal > 0.0 then
          disjoint.foreach { case (villain, villainWeight) =>
            val conditionalVillainWeight = villainWeight / disjointTotal
            val w = heroWeight * classWeight * conditionalVillainWeight
            val result = table.equity(hero, villain)
            win += w * result.win
            tie += w * result.tie
            loss += w * result.loss
          }
      }
    }

    val total = win + tie + loss
    if math.abs(total - 1.0) > 1e-10 then
      throw new IllegalStateException(s"class-uniform probability total drift: $total")

    (win, tie)

  /** Generates all 169 hand class tokens: 13 pairs (e.g. "AA", "KK") plus
    * 78 suited + 78 offsuit non-pair combinations (e.g. "AKs", "AKo").
    *
    * Order: pairs first (A-high to 2-high), then non-pairs in rank-descending order
    * with suited before offsuit. Total = 13 + 78*2 = 169.
    */
  private[holdem] def allHandClassTokens: Vector[String] =
    val ranks = Vector("A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2")
    val pairs = ranks.map(r => s"$r$r")
    val nonPairs =
      for
        i <- ranks.indices
        j <- (i + 1) until ranks.length
        hi = ranks(i)
        lo = ranks(j)
        suitedness <- Vector("s", "o")
      yield s"$hi$lo$suitedness"
    pairs ++ nonPairs

  /** Normalizes hand tokens from the published source to our parser format.
    *
    * The caniwin.com source uses "AAo"/"KKo"/etc. for pairs, but our range parser
    * expects "AA"/"KK" (no suitedness suffix for pairs, since pairs are inherently offsuit).
    *
    * @param raw the published hand token, possibly with trailing "o" on pairs
    * @return the normalized token suitable for [[CliHelpers.parseRangeDistribution]]
    */
  private[holdem] def normalizeHandToken(raw: String): String =
    val token = raw.trim.toUpperCase(Locale.ROOT)
    // Source uses AAo/KKo/... for pairs; our parser expects AA/KK/...
    if token.length == 3 && token(0) == token(1) && token(2) == 'O' then token.substring(0, 2)
    else token

