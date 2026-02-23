package sicfun.holdem

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
  private final case class PublishedRow(
      rank: Int,
      hand: String,
      winPct: Double,
      tiePct: Double
  ):
    def sourceEquityPct: Double = winPct + (0.5 * tiePct)

  private final case class ComputedRow(
      published: PublishedRow,
      comboWinPct: Double,
      comboTiePct: Double,
      classWinPct: Double,
      classTiePct: Double
  ):
    def comboEquityPct: Double = comboWinPct + (0.5 * comboTiePct)
    def classEquityPct: Double = classWinPct + (0.5 * classTiePct)
    def deltaComboWinPctPoints: Double = comboWinPct - published.winPct
    def deltaComboTiePctPoints: Double = comboTiePct - published.tiePct
    def deltaComboEquityPctPoints: Double = comboEquityPct - published.sourceEquityPct
    def deltaClassWinPctPoints: Double = classWinPct - published.winPct
    def deltaClassTiePctPoints: Double = classTiePct - published.tiePct
    def deltaClassEquityPctPoints: Double = classEquityPct - published.sourceEquityPct

  // Published values from:
  // https://caniwin.com/texasholdem/preflop/heads-up.php
  // The page states values are truncated (not rounded).
  private val PublishedTopRows: Vector[PublishedRow] = Vector(
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

  def main(args: Array[String]): Unit =
    if args.length < 1 then
      System.err.println("Usage: ComparePublishedPreflopVsRandom <canonicalExactTablePath>")
      sys.exit(1)

    val (table, meta) = HeadsUpEquityCanonicalTableIO.readWithMeta(args(0))
    if !meta.canonical then
      throw new IllegalArgumentException("expected canonical table input")
    if meta.mode != "exact" then
      throw new IllegalArgumentException(s"expected exact table input, got mode=${meta.mode}")

    val villainClassDistributions = allHandClassTokens.map(parseRangeDistribution)
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
          f"${fmt2(row.published.winPct)} ${fmt2(row.published.tiePct)} ${fmt2(row.published.sourceEquityPct)} " +
          f"${fmt5(row.comboWinPct)} ${fmt5(row.comboTiePct)} ${fmt5(row.comboEquityPct)} ${fmt5(row.deltaComboEquityPctPoints)} " +
          f"${fmt5(row.classWinPct)} ${fmt5(row.classTiePct)} ${fmt5(row.classEquityPct)} ${fmt5(row.deltaClassEquityPctPoints)}"
      )
    }

    val maxAbsDeltaComboEq = computed.map(r => math.abs(r.deltaComboEquityPctPoints)).max
    val meanAbsDeltaComboEq = computed.map(r => math.abs(r.deltaComboEquityPctPoints)).sum / computed.length.toDouble
    val maxAbsDeltaClassEq = computed.map(r => math.abs(r.deltaClassEquityPctPoints)).max
    val meanAbsDeltaClassEq = computed.map(r => math.abs(r.deltaClassEquityPctPoints)).sum / computed.length.toDouble
    println()
    println(f"ModelA maxAbsDeltaEq(pp)=$maxAbsDeltaComboEq%.6f")
    println(f"ModelA meanAbsDeltaEq(pp)=$meanAbsDeltaComboEq%.6f")
    println(f"ModelB maxAbsDeltaEq(pp)=$maxAbsDeltaClassEq%.6f")
    println(f"ModelB meanAbsDeltaEq(pp)=$meanAbsDeltaClassEq%.6f")

  private def computeRow(
      table: HeadsUpEquityCanonicalTable,
      published: PublishedRow,
      villainClassDistributions: Vector[sicfun.core.DiscreteDistribution[HoleCards]]
  ): ComputedRow =
    val token = normalizeHandToken(published.hand)
    val heroDist = parseRangeDistribution(token)

    val (comboWin, comboTie) = computeAgainstComboUniform(table, heroDist)
    val (classWin, classTie) = computeAgainstClassUniform(table, heroDist, villainClassDistributions)

    ComputedRow(
      published = published,
      comboWinPct = comboWin * 100.0,
      comboTiePct = comboTie * 100.0,
      classWinPct = classWin * 100.0,
      classTiePct = classTie * 100.0
    )

  private def computeAgainstComboUniform(
      table: HeadsUpEquityCanonicalTable,
      heroDist: sicfun.core.DiscreteDistribution[HoleCards]
  ): (Double, Double) =
    var win = 0.0
    var tie = 0.0
    var loss = 0.0

    heroDist.weights.foreach { case (hero, heroWeight) =>
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

    val total = win + tie + loss
    if math.abs(total - 1.0) > 1e-10 then
      throw new IllegalStateException(s"combo-uniform probability total drift: $total")

    (win, tie)

  private def computeAgainstClassUniform(
      table: HeadsUpEquityCanonicalTable,
      heroDist: sicfun.core.DiscreteDistribution[HoleCards],
      villainClassDistributions: Vector[sicfun.core.DiscreteDistribution[HoleCards]]
  ): (Double, Double) =
    var win = 0.0
    var tie = 0.0
    var loss = 0.0

    val classWeight = 1.0 / villainClassDistributions.length.toDouble

    heroDist.weights.foreach { case (hero, heroWeight) =>
      villainClassDistributions.foreach { villainClass =>
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

  private def allHandClassTokens: Vector[String] =
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

  private def parseRangeDistribution(token: String) =
    RangeParser.parse(token) match
      case Right(dist) => dist
      case Left(err) => throw new IllegalArgumentException(s"failed to parse hand token '$token': $err")

  private def normalizeHandToken(raw: String): String =
    val token = raw.trim.toUpperCase(Locale.ROOT)
    // Source uses AAo/KKo/... for pairs; our parser expects AA/KK/...
    if token.length == 3 && token(0) == token(1) && token(2) == 'O' then token.substring(0, 2)
    else token

  private def fmt2(value: Double): String =
    String.format(Locale.ROOT, "%6.2f", java.lang.Double.valueOf(value))

  private def fmt5(value: Double): String =
    String.format(Locale.ROOT, "%8.5f", java.lang.Double.valueOf(value))
