package sicfun.holdem.model

import munit.FunSuite

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters.*

class TrainPokerActionModelCliTest extends FunSuite:
  test("run trains and saves artifact directory") {
    val header = "street\tboard\tpotBefore\ttoCall\tposition\tstackBefore\taction\tholeCards"
    val rows = Vector(
      "Flop\tTs 9h 8d\t20.0\t10.0\tButton\t200.0\traise:20.0\tAh Kh",
      "Flop\tTs 9h 8d\t20.0\t10.0\tButton\t200.0\tfold\t7c 2d",
      "Flop\tTs 9h 8d\t20.0\t10.0\tButton\t200.0\traise:20.0\tAd Kd",
      "Flop\tTs 9h 8d\t20.0\t10.0\tButton\t200.0\tfold\t7h 2c",
      "Flop\tTs 9h 8d\t20.0\t10.0\tButton\t200.0\traise:20.0\tAs Ks",
      "Flop\tTs 9h 8d\t20.0\t10.0\tButton\t200.0\tfold\t6c 2s",
      "Flop\tTs 9h 8d\t20.0\t10.0\tButton\t200.0\traise:20.0\tAc Kc",
      "Flop\tTs 9h 8d\t20.0\t10.0\tButton\t200.0\tfold\t5c 2h"
    )
    val trainingPath = writeTempTsv(header +: rows)
    val outDir = Files.createTempDirectory("sicfun-train-cli-")

    try
      val args = Array(
        trainingPath.toString,
        outDir.toString,
        "--learningRate=0.1",
        "--iterations=300",
        "--l2Lambda=0.001",
        "--maxMeanBrierScore=1.0",
        "--validationFraction=0.25",
        "--splitSeed=7",
        "--failOnGate=true",
        "--modelId=cli-v1",
        "--schemaVersion=schema-v1",
        "--source=cli-test",
        "--trainedAtEpochMillis=123456"
      )

      val result = TrainPokerActionModel.run(args)
      assert(result.isRight, s"expected successful run, got $result")
      val runResult = result.toOption.get
      assert(Files.exists(runResult.outputDir.resolve("metadata.properties")))
      assert(Files.exists(runResult.outputDir.resolve("weights.tsv")))
      assert(Files.exists(runResult.outputDir.resolve("bias.tsv")))
      assert(Files.exists(runResult.outputDir.resolve("category-index.tsv")))

      val loaded = PokerActionModelArtifactIO.load(outDir)
      assertEquals(loaded.version.id, "cli-v1")
      assertEquals(loaded.version.schemaVersion, "schema-v1")
      assertEquals(loaded.version.source, "cli-test")
      assertEquals(loaded.version.trainedAtEpochMillis, 123456L)
      assertEquals(loaded.evaluationStrategy, "holdout-split")
      assertEquals(loaded.validationFraction, Some(0.25))
      assertEquals(loaded.splitSeed, Some(7L))
      assert(loaded.gatePassed)
    finally
      Files.deleteIfExists(trainingPath)
      deleteRecursively(outDir)
  }

  test("run returns Left on invalid options") {
    val result = TrainPokerActionModel.run(Array("train.tsv", "out", "--iterations=oops"))
    assert(result.isLeft)
  }

  private def writeTempTsv(lines: Seq[String]): Path =
    val path = Files.createTempFile("sicfun-cli-training-", ".tsv")
    Files.write(path, lines.asJava, StandardCharsets.UTF_8)
    path

  private def deleteRecursively(path: Path): Unit =
    if Files.exists(path) then
      val stream = Files.walk(path)
      try
        val all = stream.iterator().asScala.toVector.sortBy(_.toString.length).reverse
        all.foreach(Files.deleteIfExists)
      finally stream.close()
