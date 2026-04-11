package sicfun.holdem.cfr

import munit.FunSuite
import sicfun.holdem.types.TestSystemPropertyScope

import java.nio.charset.StandardCharsets
import java.nio.file.Files
import scala.jdk.CollectionConverters.*

/** Tests for [[HoldemCfrApproximationReport]], the offline diagnostics suite that
  * solves representative poker spots and measures CFR approximation quality.
  *
  * Two main scenarios are tested:
  *  1. '''Output stability''': A small subset of the default suite is solved with
  *     reduced iterations, verifying that summary.txt, spots.tsv, and
  *     external-comparison.json are written correctly with valid structure and content.
  *  2. '''Exploitability budget''': The full default suite is solved at production
  *     iteration counts, verifying that mean and per-spot exploitability stay within
  *     regression budgets. This catches solver quality regressions.
  *
  * All tests force the "scala" provider via system properties to ensure deterministic
  * behavior regardless of native library availability.
  */
class HoldemCfrApproximationReportTest extends FunSuite:
  /** Runs a test block with specified system properties, resetting native runtime
    * and auto-provider caches before and after to ensure test isolation.
    */
  private def withSystemProperties(properties: Map[String, String])(thunk: => Unit): Unit =
    TestSystemPropertyScope.withSystemProperties(properties.toSeq.map { case (key, value) => key -> Some(value) }) {
      HoldemCfrNativeRuntime.resetLoadCacheForTests()
      HoldemCfrSolver.resetAutoProviderForTests()
      try thunk
      finally
        HoldemCfrNativeRuntime.resetLoadCacheForTests()
        HoldemCfrSolver.resetAutoProviderForTests()
    }

  // Verifies that runSuite writes all three output files (summary.txt, spots.tsv,
  // external-comparison.json) with correct structure, column counts, and valid
  // exploitability/policy values. Also checks the JSON export includes spot
  // signatures, villain ranges, bet history, and normalized policies.
  test("approximation report writes stable TSV and external comparison exports") {
    withSystemProperties(Map("sicfun.cfr.provider" -> "scala")) {
      val root = Files.createTempDirectory("holdem-cfr-approx-report-test-")
      try
        val suite = HoldemCfrApproximationReport.DefaultSuite.take(2)
        val outDir = root.resolve("out")
        val result = HoldemCfrApproximationReport.runSuite(
          suiteName = "test-suite",
          spots = suite,
          cfrConfig = HoldemCfrConfig(
            iterations = 250,
            averagingDelay = 50,
            maxVillainHands = 16,
            equityTrials = 400,
            preferNativeBatch = false,
            rngSeed = 17L
          ),
          outDir = Some(outDir)
        )

        assert(result.isRight, s"expected run success, got $result")
        val runResult = result.toOption.getOrElse(fail("missing run result"))
        assertEquals(runResult.aggregate.spotCount, suite.size)
        assertEquals(runResult.spotResults.map(_.spot.id), suite.map(_.id))
        assertEquals(runResult.externalComparison.spots.map(_.id), suite.map(_.id))

        val summaryPath = outDir.resolve("summary.txt")
        val spotsPath = outDir.resolve("spots.tsv")
        val comparisonPath = outDir.resolve(HoldemCfrApproximationReport.ExternalComparisonFileName)
        assert(Files.exists(summaryPath), s"missing summary output: $summaryPath")
        assert(Files.exists(spotsPath), s"missing spots output: $spotsPath")
        assert(Files.exists(comparisonPath), s"missing comparison output: $comparisonPath")

        val summaryLines = Files.readAllLines(summaryPath, StandardCharsets.UTF_8).asScala.toVector
        assert(summaryLines.exists(_.contains("meanLocalExploitability")))
        assert(summaryLines.exists(_.contains("maxLocalExploitability")))
        assert(summaryLines.exists(_.contains("test-suite")))

        val rows = Files.readAllLines(spotsPath, StandardCharsets.UTF_8).asScala.toVector
        assertEquals(rows.head, HoldemCfrApproximationReport.SpotsTsvHeader)
        assertEquals(rows.tail.size, suite.size)

        val parsed = rows.tail.map(_.split("\t", -1).toVector)
        assert(parsed.forall(_.length == 18), s"unexpected column count: ${parsed.map(_.length)}")
        assertEquals(parsed.map(_(0)), suite.map(_.id))

        parsed.foreach { cols =>
          assert(cols(13).toDouble.isFinite, s"expected finite EV in row: ${cols.mkString("\t")}")
          assert(cols(14).toDouble >= 0.0, s"expected non-negative root gap in row: ${cols.mkString("\t")}")
          assert(cols(15).toDouble >= 0.0, s"expected non-negative villain gap in row: ${cols.mkString("\t")}")
          assert(cols(16).toDouble >= 0.0, s"expected non-negative exploitability in row: ${cols.mkString("\t")}")
          assert(cols(17).nonEmpty, s"expected non-empty policy summary in row: ${cols.mkString("\t")}")
        }

        val json = ujson.read(Files.readString(comparisonPath, StandardCharsets.UTF_8))
        assertEquals(json("suiteName").str, "test-suite")
        assertEquals(json("cfrConfig")("iterations").num.toInt, 250)
        assertEquals(json("cfrConfig")("maxVillainHands").num.toInt, 16)
        assertEquals(json("cfrConfig")("preferNativeBatch").bool, false)
        assertEquals(json("aggregate")("spotCount").num.toInt, suite.size)
        assertEquals(json("spots").arr.map(_("id").str).toVector, suite.map(_.id))

        json("spots").arr.foreach { spot =>
          assert(spot("spotSignature").str.nonEmpty)
          assert(spot("hero").str.nonEmpty)
          assert(spot("state")("pot").num.isFinite)
          assert(spot("state")("toCall").num >= 0.0)
          assert(spot("expectedValuePlayer0").num.isFinite)
          assert(spot("rootDeviationGap").num >= 0.0)
          assert(spot("villainDeviationGap").num >= 0.0)
          assert(spot("localExploitability").num >= 0.0)

          val candidateActions = spot("candidateActions").arr.map(_.str).toVector
          val policyKeys = spot("policy").obj.keys.toVector.sorted
          val actionEvKeys = spot("actionEvs").obj.keys.toVector.sorted
          assertEquals(policyKeys, candidateActions.sorted)
          assertEquals(actionEvKeys, candidateActions.sorted)
          assert(math.abs(spot("policy").obj.values.map(_.num).sum - 1.0) < 1e-6)

          val villainRange = spot("villainRange").arr
          assert(villainRange.nonEmpty, s"expected non-empty villain range in spot ${spot("id").str}")
          villainRange.foreach { entry =>
            assert(entry("hand").str.nonEmpty)
            assert(entry("probability").num > 0.0)
          }
        }

        val bigBlindDefense = json("spots").arr.find(_("id").str == "hu_flop_bigblind_defense").getOrElse(
          fail("missing big blind defense spot in external comparison export")
        )
        assertEquals(bigBlindDefense("state")("betHistory").arr.length, 1)
        assertEquals(bigBlindDefense("state")("betHistory")(0)("player").num.toInt, 0)
        assertEquals(bigBlindDefense("state")("betHistory")(0)("action").str, "RAISE:2.500")
      finally
        deleteRecursively(root)
    }
  }

  // Regression gate: runs the full 6-spot default suite at 1200 iterations and
  // verifies that mean/max exploitability and per-spot exploitability stay within
  // historically calibrated budgets. Failures indicate a solver quality regression.
  test("default approximation suite stays within exploitability budget") {
    withSystemProperties(Map("sicfun.cfr.provider" -> "scala")) {
      val result = HoldemCfrApproximationReport.runSuite(
        suiteName = "default",
        spots = HoldemCfrApproximationReport.DefaultSuite,
        cfrConfig = HoldemCfrConfig(
          iterations = 1_200,
          averagingDelay = 200,
          maxVillainHands = 48,
          equityTrials = 1_200,
          preferNativeBatch = false,
          rngSeed = 11L
        ),
        outDir = None
      )

      assert(result.isRight, s"expected run success, got $result")
      val runResult = result.toOption.getOrElse(fail("missing run result"))
      val aggregate = runResult.aggregate

      assertEquals(runResult.spotResults.map(_.spot.id), HoldemCfrApproximationReport.DefaultSuite.map(_.id))
      assertEquals(aggregate.providerCounts, Map("scala" -> HoldemCfrApproximationReport.DefaultSuite.size))
      assert(
        aggregate.meanLocalExploitability <= 0.016,
        s"mean exploitability regression: ${aggregate.meanLocalExploitability}"
      )
      assert(
        aggregate.maxLocalExploitability <= 0.044,
        s"max exploitability regression: ${aggregate.maxLocalExploitability}"
      )

      val exploitabilityBySpot = runResult.spotResults.map(result =>
        result.spot.id -> result.solution.localExploitability
      ).toMap

      assert(
        exploitabilityBySpot("hu_turn_button_vs_probe") <= 0.045,
        s"turn probe exploitability regression: ${exploitabilityBySpot("hu_turn_button_vs_probe")}"
      )
      assert(
        exploitabilityBySpot("hu_turn_button_bet_or_check") <= 0.013,
        s"turn bet/check exploitability regression: ${exploitabilityBySpot("hu_turn_button_bet_or_check")}"
      )
      assert(
        exploitabilityBySpot("hu_river_bluffcatch") <= 0.032,
        s"river bluffcatch exploitability regression: ${exploitabilityBySpot("hu_river_bluffcatch")}"
      )
      assert(
        exploitabilityBySpot("hu_river_bigblind_bet_or_check") <= 0.001,
        s"river bet/check exploitability regression: ${exploitabilityBySpot("hu_river_bigblind_bet_or_check")}"
      )
    }
  }

  private def deleteRecursively(path: java.nio.file.Path): Unit =
    if Files.exists(path) then
      val stream = Files.walk(path)
      try
        val all = stream.iterator().asScala.toVector.sortBy(_.toString.length).reverse
        all.foreach(Files.deleteIfExists)
      finally
        stream.close()
