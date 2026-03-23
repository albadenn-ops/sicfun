package sicfun.holdem.cfr

import sicfun.holdem.analysis.EvAnalysis
import sicfun.holdem.equity.HoldemEquity

/** Prints exact equity and immediate call EV for selected approximation spots.
  *
  * This is a narrow diagnostic utility for auditing whether an externally-failing
  * spot is at least directionally plausible under exact showdown equity.
  */
object HoldemCfrSpotEquityProbe:
  def main(args: Array[String]): Unit =
    val requestedIds =
      if args.isEmpty then HoldemCfrApproximationReport.DefaultSuite.map(_.id).toSet
      else args.toSet

    HoldemCfrApproximationReport.DefaultSuite
      .filter(spot => requestedIds.contains(spot.id))
      .foreach { spot =>
        val exact = HoldemEquity.equityExact(spot.hero, spot.state.board, spot.villainRange)
        val variance = EvAnalysis.evVariance(spot.hero, spot.state.board, spot.villainRange)
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
