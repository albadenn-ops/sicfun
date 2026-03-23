package sicfun.holdem.validation

import java.util.Locale

/** Per-player validation result. */
final case class PlayerValidationResult(
    villainName: String,
    leakId: String,
    severity: Double,
    totalHands: Int,
    leakApplicableSpots: Int,
    leakFiredCount: Int,
    heroNetBbPer100: Double,
    convergence: ConvergenceSummary,
    assignedArchetype: String,
    archetypeConvergenceChunk: Option[Int],
    clusterId: Option[Int]
)

/** Formats the final validation report — per-player and aggregate. */
object ValidationScorecard:
  def format(results: Vector[PlayerValidationResult]): String =
    val sb = new StringBuilder
    sb.append("=== PROFILING VALIDATION SCORECARD ===\n\n")

    results.foreach { r =>
      val gtoCanary = isGtoCanary(r)
      sb.append(s"Player: ${r.villainName} (severity=${r.severity})\n")
      sb.append(s"  Hands played:     ${fmt(r.totalHands)}\n")
      sb.append(s"  Leak fired:       ${fmt(r.leakFiredCount)} / ${fmt(r.leakApplicableSpots)} applicable spots")
      if r.leakApplicableSpots > 0 then
        val pct = r.leakFiredCount.toDouble / r.leakApplicableSpots * 100.0
        sb.append(s" (${fmtD(pct, "%.1f")}%)")
      sb.append("\n")
      sb.append(s"  Hero net EV:      ${fmtD(r.heroNetBbPer100, "%+.1f")} bb/100\n")
      sb.append("\n")

      if gtoCanary then
        sb.append("  PRIMARY: False Positive Canary\n")
        val status = if r.convergence.detected then "FAIL" else "PASS"
        sb.append(s"    Status:         $status\n")
        r.convergence.handsToDetect.foreach(h => sb.append(s"    Hands to false-positive: ${fmt(h)}\n"))
        sb.append(s"    Confidence:     ${fmtD(r.convergence.finalConfidence, "%.2f")}\n")
        sb.append(s"    False positives: ${r.convergence.totalFalsePositives}\n")
      else
        sb.append("  PRIMARY: Leak Detection\n")
        val detected = if r.convergence.detected then "DETECTED" else "NOT DETECTED"
        sb.append(s"    Status:         $detected\n")
        r.convergence.handsToDetect.foreach(h => sb.append(s"    Hands to detect: ${fmt(h)}\n"))
        sb.append(s"    Confidence:     ${fmtD(r.convergence.finalConfidence, "%.2f")}\n")
        sb.append(s"    False positives: ${r.convergence.totalFalsePositives}\n")
      sb.append("\n")

      sb.append("  SECONDARY: Archetype Classification\n")
      sb.append(s"    Assigned:       ${r.assignedArchetype}\n")
      r.archetypeConvergenceChunk.foreach(c => sb.append(s"    Convergence:    chunk $c\n"))
      sb.append("\n")

      r.clusterId.foreach(c => sb.append(s"  SECONDARY: Cluster ID $c\n\n"))
      sb.append("---\n\n")
    }

    // Aggregate
    val total = results.length
    if total > 0 then
      val leakResults = results.filterNot(isGtoCanary)
      val canaryResults = results.filter(isGtoCanary)
      val detected = leakResults.count(_.convergence.detected)
      val medianHands = leakResults.flatMap(_.convergence.handsToDetect).sorted
      val median = if medianHands.nonEmpty then medianHands(medianHands.length / 2) else 0
      val avgFP = results.map(_.convergence.totalFalsePositives).sum.toDouble / total
      val winratePopulation = if leakResults.nonEmpty then leakResults else results
      val avgWinrate = winratePopulation.map(_.heroNetBbPer100).sum / winratePopulation.size.toDouble

      sb.append("=== AGGREGATE ===\n")
      sb.append(s"  Players:          $total\n")
      if leakResults.nonEmpty then
        sb.append(s"  Leak players:     ${leakResults.size}\n")
        sb.append(s"  Leaks detected:   $detected/${leakResults.size} (${fmtD(detected.toDouble / leakResults.size * 100, "%.1f")}%)\n")
        sb.append(s"  Median hands-to-detect: ${fmt(median)}\n")
      if canaryResults.nonEmpty then
        val passing = canaryResults.count(r => !r.convergence.detected)
        sb.append(
          s"  GTO canaries passing: $passing/${canaryResults.size} " +
            s"(${fmtD(passing.toDouble / canaryResults.size * 100, "%.1f")}%)\n"
        )
      sb.append(s"  Avg false positives: ${fmtD(avgFP, "%.1f")} per player\n")
      val winrateLabel = if leakResults.nonEmpty then "Hero winrate vs leak players" else "Hero winrate"
      sb.append(s"  $winrateLabel: ${fmtD(avgWinrate, "%+.1f")} bb/100 avg")
      if leakResults.nonEmpty then
        if avgWinrate > 0 then sb.append(" (adaptive beats exploitable: CONFIRMED)")
        else sb.append(" (adaptive does NOT beat exploitable: INVESTIGATE)")
      sb.append("\n")

    sb.toString()

  private def isGtoCanary(result: PlayerValidationResult): Boolean =
    result.leakId == NoLeak.Id

  private def fmt(n: Int): String =
    String.format(Locale.ROOT, "%,d", Integer.valueOf(n))

  private def fmtD(v: Double, pattern: String): String =
    String.format(Locale.ROOT, pattern, java.lang.Double.valueOf(v))
