package sicfun.holdem.cli

import munit.FunSuite

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters.*

/**
  * Tests for [[BatchTrainer]] multi-shard training pipeline.
  *
  * Uses temporary TSV shard files written to the filesystem to simulate realistic
  * batch training scenarios. Each test creates a temporary directory, writes shards,
  * runs the pipeline, and cleans up in a `finally` block.
  *
  * '''Concatenated mode tests''':
  *   - Single shard produces an aggregate model with correct sample counts and finite Brier score.
  *   - Shard processing order is deterministic (sorted by path), regardless of input order.
  *
  * '''Per-shard mode tests''':
  *   - Two shards produce independent reports with per-shard training/evaluation counts.
  *   - Aggregate Brier score matches the evaluation-count-weighted mean of per-shard scores.
  *   - No aggregate model is produced.
  *
  * '''Empty shard handling''':
  *   - An empty shard produces a report with NaN Brier score, zero counts, and gate=false.
  *   - Non-empty shards in the same batch are unaffected.
  *
  * '''Reproducibility''': Same inputs + config produce identical results across runs.
  *
  * '''Validation''':
  *   - Empty shard list is rejected.
  *   - All-empty shard batch is rejected.
  *
  * Test data uses 6 rows per shard (Flop, same board, mixed actions) with a 0.25
  * validation fraction and relaxed Brier gate (2.0) to ensure training always succeeds.
  */
class BatchTrainerTest extends FunSuite:
  /** TSV header matching the format expected by [[PokerActionTrainingDataIO.readTsv]]. */
  private val Header = "street\tboard\tpotBefore\ttoCall\tposition\tstackBefore\taction\tholeCards"

  /** First shard data: 6 flop decisions with a mix of raise, fold, call, and check actions. */
  private val rowsA = Vector(
    "Flop\tTs 9h 8d\t20.0\t10.0\tButton\t200.0\traise:20.0\tAh Kh",
    "Flop\tTs 9h 8d\t20.0\t10.0\tButton\t200.0\tfold\t7c 2d",
    "Flop\tTs 9h 8d\t20.0\t10.0\tButton\t200.0\traise:18.0\tAd Kd",
    "Flop\tTs 9h 8d\t20.0\t10.0\tButton\t200.0\tfold\t7h 2c",
    "Flop\tTs 9h 8d\t20.0\t10.0\tButton\t200.0\tcall\tAs Ks",
    "Flop\tTs 9h 8d\t20.0\t0.0\tButton\t200.0\tcheck\t6c 2s"
  )

  /** Second shard data: 6 flop decisions with different hole cards but same board/pot setup. */
  private val rowsB = Vector(
    "Flop\tTs 9h 8d\t20.0\t10.0\tButton\t200.0\tcall\tAc Kc",
    "Flop\tTs 9h 8d\t20.0\t0.0\tButton\t200.0\tcheck\t5c 2h",
    "Flop\tTs 9h 8d\t20.0\t10.0\tButton\t200.0\traise:15.0\tQh Jh",
    "Flop\tTs 9h 8d\t20.0\t10.0\tButton\t200.0\tfold\t4c 2d",
    "Flop\tTs 9h 8d\t20.0\t10.0\tButton\t200.0\tcall\tQc Jc",
    "Flop\tTs 9h 8d\t20.0\t0.0\tButton\t200.0\tcheck\t4h 3c"
  )

  /** Test config with fast iterations (250), relaxed Brier gate (2.0), and deterministic seed. */
  private val defaultConfig = BatchTrainerConfig(
    mode = BatchMode.Concatenated,
    learningRate = 0.1,
    iterations = 250,
    l2Lambda = 0.001,
    validationFraction = 0.25,
    splitSeed = 7L,
    maxMeanBrierScore = 2.0,
    failOnGate = true,
    modelIdPrefix = "batch-test",
    schemaVersion = "schema-v1",
    source = "batch-trainer-test",
    trainedAtEpochMillis = 123456L
  )

  test("concatenated mode trains aggregate model on single shard") {
    val root = Files.createTempDirectory("sicfun-batch-single-")
    try
      val shard = writeShard(root.resolve("single.tsv"), rowsA)
      val report = BatchTrainer.run(Vector(shard.toString), defaultConfig.copy(mode = BatchMode.Concatenated))

      assertEquals(report.shardReports.length, 1)
      assert(report.aggregateModel.isDefined)
      assertEquals(report.shardReports.head.sampleCount, rowsA.length)
      assert(report.aggregateMeanBrierScore.isFinite)
      assert(report.totalTrainingSamples > 0)
      assert(report.totalEvaluationSamples > 0)
    finally
      deleteRecursively(root)
  }

  test("shard processing order is deterministic and sorted by path") {
    val root = Files.createTempDirectory("sicfun-batch-order-")
    try
      val shardB = writeShard(root.resolve("b.tsv"), rowsB)
      val shardA = writeShard(root.resolve("a.tsv"), rowsA)
      val report = BatchTrainer.run(
        Vector(shardB.toString, shardA.toString),
        defaultConfig.copy(mode = BatchMode.Concatenated)
      )

      val paths = report.shardReports.map(_.shardPath)
      assertEquals(paths, paths.sorted)
    finally
      deleteRecursively(root)
  }

  test("per-shard mode trains each shard independently and uses weighted aggregate") {
    val root = Files.createTempDirectory("sicfun-batch-per-shard-")
    try
      val shard1 = writeShard(root.resolve("1.tsv"), rowsA)
      val shard2 = writeShard(root.resolve("2.tsv"), rowsB)

      val report = BatchTrainer.run(
        Vector(shard2.toString, shard1.toString),
        defaultConfig.copy(mode = BatchMode.PerShard)
      )

      assertEquals(report.shardReports.length, 2)
      assert(report.aggregateModel.isEmpty)
      assert(report.shardReports.forall(_.trainingSampleCount > 0))
      assert(report.shardReports.forall(_.evaluationSampleCount > 0))

      val expectedWeighted = {
        val weights = report.shardReports.map(_.evaluationSampleCount.toDouble)
        val scores = report.shardReports.map(_.meanBrierScore)
        scores.zip(weights).map { case (score, weight) => score * weight }.sum / weights.sum
      }
      assert(math.abs(report.aggregateMeanBrierScore - expectedWeighted) < 1e-12)
    finally
      deleteRecursively(root)
  }

  test("empty shard is skipped with NaN score report") {
    val root = Files.createTempDirectory("sicfun-batch-empty-shard-")
    try
      val emptyShard = writeShard(root.resolve("empty.tsv"), Vector.empty)
      val nonEmptyShard = writeShard(root.resolve("data.tsv"), rowsA)

      val report = BatchTrainer.run(
        Vector(nonEmptyShard.toString, emptyShard.toString),
        defaultConfig.copy(mode = BatchMode.PerShard)
      )

      val emptyReport = report.shardReports.find(_.shardPath == emptyShard.toString).getOrElse(fail("missing empty shard report"))
      assertEquals(emptyReport.sampleCount, 0)
      assertEquals(emptyReport.trainingSampleCount, 0)
      assertEquals(emptyReport.evaluationSampleCount, 0)
      assert(emptyReport.meanBrierScore.isNaN)
      assert(!emptyReport.gatePassed)
    finally
      deleteRecursively(root)
  }

  test("run is reproducible for same inputs and config") {
    val root = Files.createTempDirectory("sicfun-batch-repro-")
    try
      val shard1 = writeShard(root.resolve("1.tsv"), rowsA)
      val shard2 = writeShard(root.resolve("2.tsv"), rowsB)
      val paths = Vector(shard2.toString, shard1.toString)

      val first = BatchTrainer.run(paths, defaultConfig.copy(mode = BatchMode.Concatenated))
      val second = BatchTrainer.run(paths, defaultConfig.copy(mode = BatchMode.Concatenated))

      assertEquals(first.aggregateMeanBrierScore, second.aggregateMeanBrierScore)
      assertEquals(first.shardReports.map(_.meanBrierScore), second.shardReports.map(_.meanBrierScore))
      assertEquals(first.shardReports.map(_.gatePassed), second.shardReports.map(_.gatePassed))
      assertEquals(first.totalTrainingSamples, second.totalTrainingSamples)
      assertEquals(first.totalEvaluationSamples, second.totalEvaluationSamples)
    finally
      deleteRecursively(root)
  }

  test("run rejects empty shard list") {
    intercept[IllegalArgumentException] {
      BatchTrainer.run(Vector.empty, defaultConfig)
    }
  }

  test("run rejects when all shards are empty") {
    val root = Files.createTempDirectory("sicfun-batch-all-empty-")
    try
      val empty1 = writeShard(root.resolve("e1.tsv"), Vector.empty)
      val empty2 = writeShard(root.resolve("e2.tsv"), Vector.empty)

      intercept[IllegalArgumentException] {
        BatchTrainer.run(Vector(empty1.toString, empty2.toString), defaultConfig)
      }
    finally
      deleteRecursively(root)
  }

  /** Writes a TSV shard file with the standard header and the given data rows. */
  private def writeShard(path: Path, rows: Vector[String]): Path =
    Files.write(path, (Header +: rows).asJava, StandardCharsets.UTF_8)
    path

  /** Recursively deletes a directory and all its contents (for test cleanup). */
  private def deleteRecursively(path: Path): Unit =
    if Files.exists(path) then
      val stream = Files.walk(path)
      try
        val all = stream.iterator().asScala.toVector.sortBy(_.toString.length).reverse
        all.foreach(Files.deleteIfExists)
      finally stream.close()
