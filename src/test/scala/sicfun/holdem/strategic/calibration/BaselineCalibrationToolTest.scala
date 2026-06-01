package sicfun.holdem.strategic.calibration

import munit.FunSuite
import java.nio.file.{Files, Paths}
import sicfun.holdem.strategic.safety.BaselineArtifactIO

class BaselineCalibrationToolTest extends FunSuite:
  test("calibrate over the showdown sample produces a non-empty postflop count artifact"):
    val corpus = Paths.get(getClass.getResource("/handhistory/p1-showdown-sample.txt").toURI)
    val outDir = Files.createTempDirectory("baseline-calib-test")
    try
      val artifact = BaselineCalibrationTool.calibrate(corpus, corpusId = "p1-sample")
      assert(artifact.counts.nonEmpty, "expected ≥1 postflop showdown decision tallied")
      assert(artifact.counts.keys.forall { case (_, bucket, street, _) =>
        bucket != "PRE" && street != sicfun.holdem.types.Street.Preflop
      }, "v1 calibrates postflop only")
      assert(artifact.metadata.showdownDecisionCount >= 1, "expected at least 1 showdown decision")
      BaselineCalibrationTool.write(outDir, artifact)
      val reloaded = BaselineArtifactIO.load(outDir)
      assertEquals(reloaded.counts, artifact.counts)
    finally
      Files.walk(outDir).sorted(java.util.Comparator.reverseOrder()).forEach(p => Files.deleteIfExists(p))

  test("main writes an artifact directory"):
    val corpus = Paths.get(getClass.getResource("/handhistory/p1-showdown-sample.txt").toURI)
    val outDir = Files.createTempDirectory("baseline-calib-main")
    try
      BaselineCalibrationTool.main(Array(corpus.toString, outDir.toString))
      assert(Files.isRegularFile(outDir.resolve("baseline-counts.tsv")))
      assert(Files.isRegularFile(outDir.resolve("metadata.properties")))
    finally
      Files.walk(outDir).sorted(java.util.Comparator.reverseOrder()).forEach(p => Files.deleteIfExists(p))
