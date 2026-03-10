package sicfun.holdem.cfr

import munit.FunSuite

import java.nio.charset.StandardCharsets
import java.nio.file.Files
import scala.jdk.CollectionConverters.*

class HoldemCfrReportTest extends FunSuite:
  test("report run writes summary policy and exploitability tracking rows") {
    val root = Files.createTempDirectory("holdem-cfr-report-test-")
    try
      val outDir = root.resolve("out")
      val track = root.resolve("metrics").resolve("exploitability.tsv")
      val result = HoldemCfrReport.run(Array(
        "--hero=AcKh",
        "--board=none",
        "--pot=6",
        "--toCall=2",
        "--position=Button",
        "--stack=100",
        "--villainRange=22+,A2s+,KTo+",
        "--candidateActions=fold,call,raise:8",
        "--iterations=800",
        "--averagingDelay=100",
        "--maxVillainHands=24",
        "--equityTrials=900",
        "--includeVillainReraises=true",
        "--villainReraiseMultipliers=2.0",
        "--preferNativeBatch=true",
        "--seed=9",
        s"--outDir=${outDir}",
        s"--trackFile=${track}"
      ))

      assert(result.isRight, s"expected run success, got $result")
      val summary = result.toOption.getOrElse(fail("missing run summary"))
      assert(summary.solution.actionProbabilities.nonEmpty)
      assert(summary.solution.localExploitability >= 0.0)

      val summaryPath = outDir.resolve("summary.txt")
      val policyPath = outDir.resolve("policy.tsv")
      assert(Files.exists(summaryPath), s"missing summary output: $summaryPath")
      assert(Files.exists(policyPath), s"missing policy output: $policyPath")
      assert(Files.exists(track), s"missing exploitability tracking file: $track")

      val trackLines = Files.readAllLines(track, StandardCharsets.UTF_8).asScala.toVector
      assert(trackLines.length >= 2, s"expected header + at least one row, got ${trackLines.length}")
      assert(trackLines.head.contains("localExploitability"))
      assert(trackLines.head.contains("provider"))
    finally
      deleteRecursively(root)
  }

  test("report run fails on invalid range") {
    val result = HoldemCfrReport.run(Array("--villainRange=NOT_A_RANGE"))
    assert(result.isLeft, s"expected parse failure, got $result")
  }

  private def deleteRecursively(path: java.nio.file.Path): Unit =
    if Files.exists(path) then
      val stream = Files.walk(path)
      try
        val all = stream.iterator().asScala.toVector.sortBy(_.toString.length).reverse
        all.foreach(Files.deleteIfExists)
      finally
        stream.close()
