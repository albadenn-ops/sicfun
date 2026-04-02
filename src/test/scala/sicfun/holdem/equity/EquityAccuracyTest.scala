package sicfun.holdem.equity
import sicfun.holdem.analysis.*
import sicfun.holdem.cli.*

import munit.FunSuite

/** Verifies equity computation accuracy against published reference data.
  *
  * These tests require a canonical exact equity table. Table resolution is handled
  * by [[CanonicalExactTableTestFixture]] and will fail fast if a full table cannot
  * be loaded or generated.
  */
class EquityAccuracyTest extends FunSuite:
  private lazy val canonicalExactTable: HeadsUpEquityCanonicalTable =
    CanonicalExactTableTestFixture.load()

  // ---- ComparePublishedSpecificMatchups: 7 matchups from cardfight.com ----

  private val SpecificMatchupTolerancePp = 1.2 // percentage points

  ComparePublishedSpecificMatchups.Matchups.foreach { row =>
    test(s"specific matchup: ${row.label} (vs cardfight.com)") {
      val table = canonicalExactTable
      val (ourWin, ourTie) = ComparePublishedSpecificMatchups.computeWinTiePct(table, row.hero, row.villain)
      val ourEq = ourWin + (0.5 * ourTie)
      val dEq = math.abs(ourEq - row.sourceEqPct)

      assert(dEq <= SpecificMatchupTolerancePp,
        s"equity delta ${dEq} pp exceeds tolerance $SpecificMatchupTolerancePp pp " +
          s"(ours=${ourEq}, published=${row.sourceEqPct}, source=${row.sourceUrl})")
    }
  }

  // ---- ComparePublishedPreflopVsRandom: 17 rows from caniwin.com ----
  // Snapshot circa 2026-02; live page shows slight drift as of 2026-03-23.
  // These are supporting equity evidence, not exact current reference.

  // The source table is published with truncated percentages and a model that does not
  // perfectly align with our exact combo-conditioned formulation for every class.
  // Keep these as sanity guards (coarse agreement), not exact parity checks.
  private val PreflopComboPerRowTolerancePp = 1.75
  private val PreflopComboMeanTolerancePp = 0.80
  private val PreflopClassPerRowTolerancePp = 1.95
  private val PreflopClassMeanTolerancePp = 1.10

  test("preflop vs random: combo-uniform model matches published within tolerance") {
    val table = canonicalExactTable
    val villainClassDists = ComparePublishedPreflopVsRandom.allHandClassTokens.map(CliHelpers.parseRangeDistribution)

    val computed = ComparePublishedPreflopVsRandom.PublishedTopRows.map { row =>
      ComparePublishedPreflopVsRandom.computeRow(table, row, villainClassDists)
    }

    computed.foreach { row =>
      val dEq = math.abs(row.deltaComboEquityPctPoints)
      assert(dEq <= PreflopComboPerRowTolerancePp,
        s"${row.published.hand}: combo equity delta ${dEq} pp exceeds ${PreflopComboPerRowTolerancePp} pp")
    }

    val meanAbsDelta = computed.map(r => math.abs(r.deltaComboEquityPctPoints)).sum / computed.length
    assert(meanAbsDelta <= PreflopComboMeanTolerancePp,
      s"mean absolute combo equity delta ${meanAbsDelta} pp exceeds ${PreflopComboMeanTolerancePp} pp")
  }

  test("preflop vs random: class-uniform model matches published within tolerance") {
    val table = canonicalExactTable
    val villainClassDists = ComparePublishedPreflopVsRandom.allHandClassTokens.map(CliHelpers.parseRangeDistribution)

    val computed = ComparePublishedPreflopVsRandom.PublishedTopRows.map { row =>
      ComparePublishedPreflopVsRandom.computeRow(table, row, villainClassDists)
    }

    computed.foreach { row =>
      val dEq = math.abs(row.deltaClassEquityPctPoints)
      assert(dEq <= PreflopClassPerRowTolerancePp,
        s"${row.published.hand}: class equity delta ${dEq} pp exceeds ${PreflopClassPerRowTolerancePp} pp")
    }

    val meanAbsDelta = computed.map(r => math.abs(r.deltaClassEquityPctPoints)).sum / computed.length
    assert(meanAbsDelta <= PreflopClassMeanTolerancePp,
      s"mean absolute class equity delta ${meanAbsDelta} pp exceeds ${PreflopClassMeanTolerancePp} pp")
  }
