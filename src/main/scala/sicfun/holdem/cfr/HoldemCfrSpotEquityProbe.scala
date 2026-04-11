package sicfun.holdem.cfr

import sicfun.holdem.analysis.EvAnalysis
import sicfun.holdem.equity.HoldemEquity

/** Prints exact equity and immediate call EV for selected approximation spots.
  *
  * This is a narrow diagnostic utility for auditing whether an externally-failing
  * spot is at least directionally plausible under exact showdown equity. When a
  * spot fails an external comparison gate (e.g., against TexasSolver), this probe
  * lets you check whether the disagreement is in the equity estimates themselves
  * or in the CFR strategy layer above them.
  *
  * For each spot in [[HoldemCfrApproximationReport.DefaultSuite]], this prints:
  *  - '''Exact equity''': win/tie/loss computed via full enumeration (river) or
  *    exact combinatorial calculation, not Monte Carlo sampling.
  *  - '''Immediate call EV''': the expected value of calling (or checking) without
  *    any further strategic play — a lower bound on the value of non-fold actions.
  *  - '''Variance statistics''': mean, variance, stderr, and hand count from
  *    [[EvAnalysis.evVariance]], useful for assessing how noisy the equity estimate
  *    would be under Monte Carlo sampling.
  *
  * Usage: `runMain sicfun.holdem.cfr.HoldemCfrSpotEquityProbe [spotId1 spotId2 ...]`
  * If no spot ids are provided, all default suite spots are probed.
  */
object HoldemCfrSpotEquityProbe:

  /** Entry point. Accepts optional spot id arguments to filter the default suite.
    * If no arguments are given, probes all spots in [[HoldemCfrApproximationReport.DefaultSuite]].
    */
  def main(args: Array[String]): Unit =
    // If no spot ids given on the command line, probe every spot in the default suite
    val requestedIds =
      if args.isEmpty then HoldemCfrApproximationReport.DefaultSuite.map(_.id).toSet
      else args.toSet

    HoldemCfrApproximationReport.DefaultSuite
      .filter(spot => requestedIds.contains(spot.id))
      .foreach { spot =>
        // Compute exact showdown equity (full enumeration, no sampling noise)
        val exact = HoldemEquity.equityExact(spot.hero, spot.state.board, spot.villainRange)
        // Compute EV variance statistics for assessing Monte Carlo sampling noise
        val variance = EvAnalysis.evVariance(spot.hero, spot.state.board, spot.villainRange)
        // Immediate call EV: equity * total pot if calling, minus the call cost.
        // This is the value of simply calling (or checking) without strategic play.
        val immediateCallEv =
          if spot.state.toCall > 0.0 then
            (exact.equity * (spot.state.pot + spot.state.toCall)) - spot.state.toCall
          else exact.equity * spot.state.pot
        println(s"spot=${spot.id}")
        println(f"  equity=${exact.equity}%.6f win=${exact.win}%.6f tie=${exact.tie}%.6f loss=${exact.loss}%.6f")
        println(f"  immediateCallEv=$immediateCallEv%.6f pot=${spot.state.pot}%.3f toCall=${spot.state.toCall}%.3f")
        println(
          f"  varianceMean=${variance.mean}%.6f variance=${variance.variance}%.6f stderr=${variance.stderr}%.6f hands=${variance.handCount}%d"
        )
      }
