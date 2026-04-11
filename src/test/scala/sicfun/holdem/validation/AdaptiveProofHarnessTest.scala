package sicfun.holdem.validation

import sicfun.holdem.history.{HandHistoryImport, HandHistorySite}

import munit.FunSuite

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*

/** Tests for the 9-max adaptive proof harness orchestrator.
  *
  * Validates that [[AdaptiveProofHarness]] can run a complete proof cycle
  * (simulating a multi-player playing hall, exporting review hand histories,
  * writing ground-truth JSON and a human-readable report) and that the
  * exported PokerStars-format text roundtrips through [[HandHistoryImport]].
  *
  * The test uses a minimal configuration (1 block of 12 hands) to keep
  * runtime reasonable while still exercising the full orchestration path:
  * hall simulation, review upload generation, ground-truth serialization,
  * and report formatting.
  */
class AdaptiveProofHarnessTest extends FunSuite:
  override val munitTimeout: Duration = 300.seconds

  test("9-max adaptive proof harness produces valid hall output and report".tag(munit.Slow)) {
    val root = Files.createTempDirectory("adaptive-proof-9max-test-")
    try
      val config = AdaptiveProofHarness.Config(
        handsPerBlock = 12,
        blocks = 1,
        seed = 37L,
        budgetMs = 50L,
        bunchingTrials = 8,
        equityTrials = 80,
        outputDir = root
      )

      val result = AdaptiveProofHarness.run(config)

      // 1. One block completed
      assertEquals(result.blocks.size, 1)
      val block = result.blocks.head
      assert(block.hallSummary.handsPlayed > 0, "expected hands played")

      // 2. Hall output directory exists with review upload
      val hallOutDir = root.resolve("block-0")
      assert(Files.isDirectory(hallOutDir), s"missing hall output dir")
      val reviewPath = hallOutDir.resolve("review-upload-pokerstars.txt")
      assert(Files.exists(reviewPath), "missing review hand history")

      // 3. Review history is non-empty and parseable
      val reviewText = Files.readString(reviewPath, StandardCharsets.UTF_8)
      assert(reviewText.nonEmpty, "empty review history")
      val parsed = HandHistoryImport.parseText(reviewText, Some(HandHistorySite.PokerStars), Some("Hero"))
      assert(parsed.isRight, s"parse failed: ${parsed.left.getOrElse("")}")

      // 4. Write outputs and verify files
      val runDir = AdaptiveProofHarness.writeOutputs(result, config.outputDir)
      assert(Files.exists(runDir.resolve("ground-truth.json")))
      assert(Files.exists(runDir.resolve("report.txt")))

      // 5. Ground truth has block data
      val gt = ujson.read(Files.readString(runDir.resolve("ground-truth.json"), StandardCharsets.UTF_8))
      assertEquals(gt("blocks").arr.length, 1)
      assert(gt("totalHands").num > 0)

      // 6. Report contains expected header
      val report = Files.readString(runDir.resolve("report.txt"), StandardCharsets.UTF_8)
      assert(report.contains("Adaptive Proof Report"))

    finally
      deleteRecursively(root)
  }

  test("formatReport includes bridge fidelity summary"):
    val config = AdaptiveProofHarness.Config(handsPerBlock = 10, blocks = 0, seed = 1L)
    val result = AdaptiveProofHarness.RunResult(config, Vector.empty)
    val report = AdaptiveProofHarness.formatReport(result)
    assert(report.contains("Bridge Fidelity"), s"expected Bridge Fidelity in:\n$report")
    assert(report.contains("exact"), s"expected 'exact' in:\n$report")

  private def deleteRecursively(path: Path): Unit =
    if Files.exists(path) then
      val stream = Files.walk(path)
      try
        val all = stream.iterator().asScala.toVector.sortBy(_.toString.length).reverse
        all.foreach(Files.deleteIfExists)
      finally
        stream.close()
