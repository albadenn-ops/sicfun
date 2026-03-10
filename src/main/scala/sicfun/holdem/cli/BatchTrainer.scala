package sicfun.holdem.cli
import sicfun.holdem.types.*
import sicfun.holdem.model.*

import sicfun.core.Metrics

/** Strategy for combining multiple training data shards.
  *
  *   - [[PerShard]]: trains an independent model on each shard and reports per-shard metrics
  *     (no aggregate model is produced).
  *   - [[Concatenated]]: concatenates all shards into a single dataset and trains one model,
  *     then evaluates that model on each shard individually.
  */
enum BatchMode:
  case PerShard
  case Concatenated

/** Hyperparameters and metadata for a batch training run.
  *
  * @param mode                  shard combination strategy ([[BatchMode.PerShard]] or [[BatchMode.Concatenated]])
  * @param learningRate          gradient descent step size
  * @param iterations            number of training iterations per model
  * @param l2Lambda              L2 regularization strength
  * @param validationFraction    fraction of data reserved for evaluation (must be in (0, 1))
  * @param splitSeed             random seed for train/validation split
  * @param maxMeanBrierScore     calibration gate threshold; models exceeding this fail the gate
  * @param failOnGate            if `true`, throw on gate failure; if `false`, record but continue
  * @param modelIdPrefix         prefix for generated model version IDs
  * @param schemaVersion         schema version tag written into the model artifact
  * @param source                provenance string (e.g. `"batch-trainer"`)
  * @param trainedAtEpochMillis  timestamp recorded in the model version metadata
  */
final case class BatchTrainerConfig(
    mode: BatchMode = BatchMode.Concatenated,
    learningRate: Double = 0.01,
    iterations: Int = 1000,
    l2Lambda: Double = 0.001,
    validationFraction: Double = 0.2,
    splitSeed: Long = 1L,
    maxMeanBrierScore: Double = 1.0,
    failOnGate: Boolean = false,
    modelIdPrefix: String = "batch-model",
    schemaVersion: String = "poker-action-model-v1",
    source: String = "batch-trainer",
    trainedAtEpochMillis: Long = System.currentTimeMillis()
):
  require(iterations > 0, "iterations must be positive")
  require(learningRate.isFinite, "learningRate must be finite")
  require(l2Lambda.isFinite, "l2Lambda must be finite")
  require(maxMeanBrierScore >= 0.0 && maxMeanBrierScore.isFinite, "maxMeanBrierScore must be finite and >= 0")
  require(validationFraction > 0.0 && validationFraction < 1.0, "validationFraction must be in (0, 1)")
  require(modelIdPrefix.trim.nonEmpty, "modelIdPrefix must be non-empty")
  require(schemaVersion.trim.nonEmpty, "schemaVersion must be non-empty")
  require(source.trim.nonEmpty, "source must be non-empty")
  require(trainedAtEpochMillis >= 0L, "trainedAtEpochMillis must be non-negative")

/** Per-shard training/evaluation metrics.
  *
  * @param shardPath             filesystem path of the source shard TSV
  * @param sampleCount           total rows loaded from the shard
  * @param trainingSampleCount   rows used for training
  * @param evaluationSampleCount rows used for evaluation (validation split or full-shard eval in concatenated mode)
  * @param meanBrierScore        mean Brier score on the evaluation set (`NaN` for empty shards)
  * @param gatePassed            whether the model passed the calibration gate for this shard
  */
final case class ShardReport(
    shardPath: String,
    sampleCount: Int,
    trainingSampleCount: Int,
    evaluationSampleCount: Int,
    meanBrierScore: Double,
    gatePassed: Boolean
):
  require(shardPath.trim.nonEmpty, "shardPath must be non-empty")
  require(sampleCount >= 0, "sampleCount must be >= 0")
  require(trainingSampleCount >= 0, "trainingSampleCount must be >= 0")
  require(evaluationSampleCount >= 0, "evaluationSampleCount must be >= 0")

/** Aggregate result of a batch training run across all shards.
  *
  * @param shardReports            per-shard metrics (one per input shard path)
  * @param aggregateMeanBrierScore evaluation-count-weighted mean Brier score across all shards
  * @param totalTrainingSamples    total training rows across all shards
  * @param totalEvaluationSamples  total evaluation rows across all shards
  * @param aggregateModel          the trained model (present only in [[BatchMode.Concatenated]] mode)
  */
final case class BatchTrainingReport(
    shardReports: Vector[ShardReport],
    aggregateMeanBrierScore: Double,
    totalTrainingSamples: Int,
    totalEvaluationSamples: Int,
    aggregateModel: Option[TrainedPokerActionModel]
):
  require(shardReports.nonEmpty, "shardReports must be non-empty")
  require(totalTrainingSamples >= 0, "totalTrainingSamples must be >= 0")
  require(totalEvaluationSamples >= 0, "totalEvaluationSamples must be >= 0")

/** Multi-shard model training pipeline.
  *
  * Loads training data from multiple TSV shard files via [[PokerActionTrainingDataIO]],
  * trains one or more [[PokerActionModel]] instances depending on the [[BatchMode]],
  * and produces a [[BatchTrainingReport]] with per-shard and aggregate metrics.
  */
object BatchTrainer:
  /** A shard loaded into memory with its source path. */
  private final case class LoadedShard(path: String, samples: Vector[(GameState, HoleCards, PokerAction)]):
    def isEmpty: Boolean = samples.isEmpty

  /** Runs the batch training pipeline.
    *
    * @param shardPaths filesystem paths to TSV training data files (at least one required)
    * @param config     training hyperparameters and metadata
    * @return a [[BatchTrainingReport]] summarizing per-shard and aggregate results
    * @throws IllegalArgumentException if all shards are empty
    */
  def run(shardPaths: Vector[String], config: BatchTrainerConfig = BatchTrainerConfig()): BatchTrainingReport =
    require(shardPaths.nonEmpty, "shardPaths must be non-empty")
    val normalizedPaths = shardPaths.map(_.trim).filter(_.nonEmpty).sorted
    require(normalizedPaths.nonEmpty, "shardPaths must contain at least one non-blank path")

    val loaded = normalizedPaths.map(path => LoadedShard(path, PokerActionTrainingDataIO.readTsv(path)))
    require(loaded.exists(shard => !shard.isEmpty), "at least one non-empty shard is required")

    config.mode match
      case BatchMode.PerShard     => runPerShard(loaded, config)
      case BatchMode.Concatenated => runConcatenated(loaded, config)

  private def runPerShard(
      loaded: Vector[LoadedShard],
      config: BatchTrainerConfig
  ): BatchTrainingReport =
    val reports = loaded.zipWithIndex.map { case (shard, index) =>
      if shard.isEmpty then emptyShardReport(shard.path)
      else
        val artifact = PokerActionModel.trainVersioned(
          trainingData = shard.samples,
          learningRate = config.learningRate,
          iterations = config.iterations,
          l2Lambda = config.l2Lambda,
          validationFraction = config.validationFraction,
          splitSeed = config.splitSeed,
          maxMeanBrierScore = config.maxMeanBrierScore,
          failOnGate = config.failOnGate,
          modelId = s"${config.modelIdPrefix}-shard-${index + 1}",
          schemaVersion = config.schemaVersion,
          source = s"${config.source}:${shard.path}",
          trainedAtEpochMillis = config.trainedAtEpochMillis
        )
        ShardReport(
          shardPath = shard.path,
          sampleCount = shard.samples.length,
          trainingSampleCount = artifact.trainingSampleCount,
          evaluationSampleCount = artifact.evaluationSampleCount,
          meanBrierScore = artifact.calibration.meanBrierScore,
          gatePassed = artifact.gatePassed
        )
    }

    val aggregateMean = aggregateWeightedMeanBrier(reports)
    val totalTraining = reports.map(_.trainingSampleCount).sum
    val totalEvaluation = reports.map(_.evaluationSampleCount).sum

    BatchTrainingReport(
      shardReports = reports,
      aggregateMeanBrierScore = aggregateMean,
      totalTrainingSamples = totalTraining,
      totalEvaluationSamples = totalEvaluation,
      aggregateModel = None
    )

  private def runConcatenated(
      loaded: Vector[LoadedShard],
      config: BatchTrainerConfig
  ): BatchTrainingReport =
    val concatenated = loaded.filterNot(_.isEmpty).flatMap(_.samples)
    val artifact = PokerActionModel.trainVersioned(
      trainingData = concatenated,
      learningRate = config.learningRate,
      iterations = config.iterations,
      l2Lambda = config.l2Lambda,
      validationFraction = config.validationFraction,
      splitSeed = config.splitSeed,
      maxMeanBrierScore = config.maxMeanBrierScore,
      failOnGate = config.failOnGate,
      modelId = s"${config.modelIdPrefix}-concatenated",
      schemaVersion = config.schemaVersion,
      source = config.source,
      trainedAtEpochMillis = config.trainedAtEpochMillis
    )

    val reports = loaded.map { shard =>
      if shard.isEmpty then emptyShardReport(shard.path)
      else
        val summary = PokerActionModel.calibrationSummary(artifact.model, shard.samples)
        ShardReport(
          shardPath = shard.path,
          sampleCount = shard.samples.length,
          trainingSampleCount = artifact.trainingSampleCount,
          evaluationSampleCount = summary.sampleCount,
          meanBrierScore = summary.meanBrierScore,
          gatePassed = artifact.gate.passed(summary)
        )
    }

    val aggregateMean = aggregateWeightedMeanBrier(reports)
    val totalEvaluation = reports.map(_.evaluationSampleCount).sum

    BatchTrainingReport(
      shardReports = reports,
      aggregateMeanBrierScore = aggregateMean,
      totalTrainingSamples = artifact.trainingSampleCount,
      totalEvaluationSamples = totalEvaluation,
      aggregateModel = Some(artifact)
    )

  private def emptyShardReport(path: String): ShardReport =
    ShardReport(
      shardPath = path,
      sampleCount = 0,
      trainingSampleCount = 0,
      evaluationSampleCount = 0,
      meanBrierScore = Double.NaN,
      gatePassed = false
    )

  private def aggregateWeightedMeanBrier(reports: Vector[ShardReport]): Double =
    val valid = reports.filter(report =>
      report.evaluationSampleCount > 0 && !report.meanBrierScore.isNaN
    )
    require(valid.nonEmpty, "at least one shard with evaluation samples is required for aggregate mean Brier score")
    Metrics.weightedMean(
      values = valid.map(_.meanBrierScore),
      weights = valid.map(_.evaluationSampleCount.toDouble)
    )
