package sicfun.holdem.validation

import munit.FunSuite

/** Tests for [[ValidationScorecard]], which formats the final human-readable
  * validation report from a vector of [[PlayerValidationResult]]s.
  *
  * Coverage:
  *   - Per-player sections include player name, detection status, archetype,
  *     and hands-to-detect count
  *   - Undetected leaks show "NOT DETECTED" label
  *   - GTO canary entries use PASS/FAIL semantics (PASS = no false positive,
  *     FAIL = false positive detected) with appropriate detail labels
  *   - Aggregate section shows overall detection ratio
  *   - Empty results still produce a valid scorecard with header
  *   - Leak fire rate percentage (fired / applicable spots) is displayed
  *   - Positive hero winrate shows "CONFIRMED" (adaptive beats exploitable)
  *   - Negative hero winrate shows "INVESTIGATE" warning
  *   - Leak detection and GTO canary counts are reported separately in the
  *     aggregate section
  */
class ValidationScorecardTest extends FunSuite:

  private def result(
      name: String = "test_player",
      leakId: String = "test-leak",
      severity: Double = 0.5,
      detected: Boolean = true,
      handsToDetect: Option[Int] = Some(5000),
      confidence: Double = 0.8,
      falsePositives: Int = 0,
      heroNet: Double = 5.0,
      archetype: String = "Tag",
      archetypeChunk: Option[Int] = Some(3)
  ): PlayerValidationResult =
    PlayerValidationResult(
      villainName = name,
      leakId = leakId,
      severity = severity,
      totalHands = 10000,
      leakApplicableSpots = 500,
      leakFiredCount = 250,
      heroNetBbPer100 = heroNet,
      convergence = ConvergenceSummary(
        leakId = leakId,
        detected = detected,
        firstDetectedChunk = handsToDetect.map(_ / 1000),
        handsToDetect = handsToDetect,
        finalConfidence = confidence,
        totalFalsePositives = falsePositives
      ),
      assignedArchetype = archetype,
      archetypeConvergenceChunk = archetypeChunk,
      clusterId = None
    )

  test("format produces per-player section"):
    val r = result()
    val text = ValidationScorecard.format(Vector(r))
    assert(text.contains("test_player"), "must include player name")
    assert(text.contains("DETECTED"), "must show detection status")
    assert(text.contains("Tag"), "must show archetype")
    assert(text.contains("5,000"), "must show hands to detect")

  test("format shows NOT DETECTED for undetected leaks"):
    val r = result(detected = false, handsToDetect = None)
    val text = ValidationScorecard.format(Vector(r))
    assert(text.contains("NOT DETECTED"))

  test("format shows PASS/FAIL semantics for gto canary"):
    val passing = result(leakId = "gto-baseline", detected = false, handsToDetect = None)
    val failing = result(leakId = "gto-baseline", detected = true, handsToDetect = Some(1000))
    val passText = ValidationScorecard.format(Vector(passing))
    val failText = ValidationScorecard.format(Vector(failing))
    assert(passText.contains("False Positive Canary"))
    assert(passText.contains("PASS"))
    assert(failText.contains("FAIL"))
    assert(failText.contains("Hands to false-positive"))

  test("format includes aggregate section"):
    val results = Vector(
      result(name = "p1", detected = true, heroNet = 10.0),
      result(name = "p2", detected = false, handsToDetect = None, heroNet = -5.0)
    )
    val text = ValidationScorecard.format(results)
    assert(text.contains("AGGREGATE"), "must have aggregate section")
    assert(text.contains("1/2"), "must show detection ratio")

  test("format handles empty results"):
    val text = ValidationScorecard.format(Vector.empty)
    assert(text.contains("SCORECARD"), "must still have header")

  test("format shows leak fire rate percentage"):
    val r = result()
    val text = ValidationScorecard.format(Vector(r))
    assert(text.contains("250"), "must show fired count")
    assert(text.contains("500"), "must show applicable spots")
    assert(text.contains("50.0%"), "must show fire rate percentage")

  test("format shows positive hero winrate confirmation"):
    val results = Vector(result(heroNet = 15.0))
    val text = ValidationScorecard.format(results)
    assert(text.contains("CONFIRMED"), "positive winrate should confirm adaptive beats exploitable")

  test("format splits leak detection and canary aggregate counts"):
    val results = Vector(
      result(name = "leak-player", leakId = "overcall-big-bets", detected = true, heroNet = 20.0),
      result(name = "gto-control", leakId = "gto-baseline", detected = false, handsToDetect = None, heroNet = -2.0)
    )
    val text = ValidationScorecard.format(results)
    assert(text.contains("Leak players:     1"))
    assert(text.contains("Leaks detected:   1/1"))
    assert(text.contains("GTO canaries passing: 1/1"))
    assert(text.contains("Hero winrate vs leak players"))

  test("format shows negative hero winrate warning"):
    val results = Vector(result(heroNet = -15.0))
    val text = ValidationScorecard.format(results)
    assert(text.contains("INVESTIGATE"), "negative winrate should flag investigation")

  test("PlayerValidationResult accepts strategicSummary field"):
    val withSummary = PlayerValidationResult(
      villainName = "test",
      leakId = "overcall-big-bets",
      severity = 0.6,
      totalHands = 1000,
      leakApplicableSpots = 100,
      leakFiredCount = 50,
      heroNetBbPer100 = 5.0,
      convergence = ConvergenceSummary("overcall-big-bets", true, Some(3), Some(300), 0.85, 0),
      assignedArchetype = "CallingStation",
      archetypeConvergenceChunk = Some(5),
      clusterId = None,
      strategicSummary = Some(StrategicSummary(
        dominantClass = "Value",
        fidelityCoverage = "8 exact, 14 approximate, 2 absent (24 total)",
        fourWorldV11 = 0.65,
        fourWorldV00 = 0.50
      ))
    )
    assert(withSummary.strategicSummary.isDefined)
    assertEquals(withSummary.strategicSummary.get.dominantClass, "Value")

  test("scorecard formats strategic section when strategicSummary present"):
    val results = Vector(
      PlayerValidationResult(
        villainName = "overcall_severe",
        leakId = "overcall-big-bets",
        severity = 0.9,
        totalHands = 1000,
        leakApplicableSpots = 100,
        leakFiredCount = 60,
        heroNetBbPer100 = 8.0,
        convergence = ConvergenceSummary("overcall-big-bets", true, Some(2), Some(200), 0.90, 0),
        assignedArchetype = "CallingStation",
        archetypeConvergenceChunk = Some(3),
        clusterId = None,
        strategicSummary = Some(StrategicSummary(
          dominantClass = "Value",
          fidelityCoverage = "8 exact, 14 approximate, 2 absent (24 total)",
          fourWorldV11 = 0.65,
          fourWorldV00 = 0.50
        ))
      )
    )
    val report = ValidationScorecard.format(results)
    assert(report.contains("STRATEGIC"), s"expected STRATEGIC section in:\n$report")
    assert(report.contains("Value"), s"expected dominant class in:\n$report")
    assert(report.contains("V^{1,1}"), s"expected four-world label in:\n$report")

  test("scorecard omits strategic section when strategicSummary is None"):
    val results = Vector(result(detected = false, handsToDetect = None))
    val report = ValidationScorecard.format(results)
    assert(!report.contains("STRATEGIC"), s"should NOT contain STRATEGIC when summary is None:\n$report")
