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
  /** Lazily loaded canonical exact table; shared across all tests in this suite.
    * Loading may trigger table generation on first access (potentially slow).
    */
  private lazy val canonicalExactTable: HeadsUpEquityCanonicalTable =
    CanonicalExactTableTestFixture.load()

  // ---- ComparePublishedSpecificMatchups: 7 specific hero-vs-villain matchups from cardfight.com ----
  // These test our exact equity against known published values for specific hand matchups.

  /** Maximum allowed difference in percentage points between our equity and the published value. */
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

  // ---- ComparePublishedPreflopVsRandom: 17 preflop hand classes from caniwin.com ----
  // Snapshot circa 2026-02; live page shows slight drift as of 2026-03-23.
  // These test preflop equity of top hand classes vs a random hand using two models:
  //   - Combo-uniform: each concrete combo weighted equally
  //   - Class-uniform: each hand class weighted equally (then combos within uniformly)
  // These are supporting equity evidence (sanity guards), not exact parity checks.

  /** Per-row tolerance for combo-uniform preflop equity vs published values. */
  private val PreflopComboPerRowTolerancePp = 1.75
  /** Mean absolute error tolerance across all rows for combo-uniform model. */
  private val PreflopComboMeanTolerancePp = 0.80
  /** Per-row tolerance for class-uniform preflop equity vs published values. */
  private val PreflopClassPerRowTolerancePp = 1.95
  /** Mean absolute error tolerance across all rows for class-uniform model. */
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
