package sicfun.holdem.validation

import munit.FunSuite

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

  test("format shows negative hero winrate warning"):
    val results = Vector(result(heroNet = -15.0))
    val text = ValidationScorecard.format(results)
    assert(text.contains("INVESTIGATE"), "negative winrate should flag investigation")
