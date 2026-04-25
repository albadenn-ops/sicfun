package sicfun.holdem.equity
import sicfun.holdem.types.*

import munit.FunSuite
import scala.concurrent.duration.*

/**
  * Heavyweight Exact-vs-MonteCarlo parity tests for the canonical heads-up equity table.
  *
  * Tagged Slow because Mode.Exact enumerates C(48,5) = 1,712,304 boards per matchup; a
  * single matchup takes ~50s on a single CPU core. Run isolated:
  *
  * {{{
  * sbt 'testOnly sicfun.holdem.equity.HeadsUpEquityCanonicalParityTest'
  * }}}
  *
  * Filter out from a default run with:
  *
  * {{{
  * sbt 'testOnly * -- --exclude-tags=slow'
  * }}}
  *
  * The MonteCarloConvergenceTest covers the equityExact-vs-equityMonteCarlo (single hero
  * vs range) parity for [[sicfun.holdem.HoldemEquity]]; this suite covers the
  * [[HeadsUpEquityTable]]/[[HeadsUpEquityCanonicalTable]] surface specifically.
  */
class HeadsUpEquityCanonicalParityTest extends FunSuite:
  override val munitTimeout: Duration = 5.minutes

  private val SlowTest = new munit.Tag("slow")

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(
      sicfun.core.Card.parse(a).getOrElse(fail(s"invalid card: $a")),
      sicfun.core.Card.parse(b).getOrElse(fail(s"invalid card: $b"))
    ))

  test("Mode.Exact and Mode.MonteCarlo agree within stderr on a fixed matchup".tag(SlowTest)) {
    // AsKs vs QcQd is a textbook overpair-vs-broadway-suited spot. Exact is the
    // ground truth; MonteCarlo at 50k trials must land within 6 stderr (a
    // less-than-1-in-500M false-positive bound under the central-limit
    // approximation that holds for outcome means).
    val hero = hole("As", "Ks")
    val villain = hole("Qc", "Qd")

    val exact = HeadsUpEquityTable.computeEquityDeterministic(
      hero,
      villain,
      HeadsUpEquityTable.Mode.Exact,
      monteCarloSeedBase = 0L,
      keyMaterial = 0L
    )
    val mc = HeadsUpEquityTable.computeEquityDeterministic(
      hero,
      villain,
      HeadsUpEquityTable.Mode.MonteCarlo(50_000),
      monteCarloSeedBase = 17L,
      keyMaterial = 1L
    )

    assertEquals(exact.stderr, 0.0, "exact mode must report zero stderr")
    assert(
      exact.total > 0.99 && exact.total < 1.01,
      s"exact win+tie+loss must sum near 1: got ${exact.total}"
    )
    assert(
      mc.total > 0.99 && mc.total < 1.01,
      s"MC win+tie+loss must sum near 1: got ${mc.total}"
    )

    val diff = math.abs(mc.equity - exact.equity)
    val bound = math.max(6.0 * mc.stderr, 0.005)
    assert(
      diff <= bound,
      s"MC equity ${mc.equity} drifted from exact ${exact.equity} by $diff, exceeding 6-sigma bound $bound (stderr=${mc.stderr})"
    )
  }
