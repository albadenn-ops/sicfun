package sicfun.holdem.model
import sicfun.holdem.cli.*

import java.nio.file.{Path, Paths}

/**
 * Command-line interface for training and exporting versioned poker action models in sicfun.
 *
 * This is the production entry point for offline model training. It reads a TSV training
 * data file (via [[PokerActionTrainingDataIO]]), trains a multinomial logistic regression
 * model (via [[PokerActionModel.trainVersioned]]), and persists the complete artifact
 * (via [[PokerActionModelArtifactIO]]).
 *
 * Usage: `TrainPokerActionModel <trainingTsvPath> <artifactOutputDir> [--key=value ...]`
 *
 * The CLI supports all training hyperparameters (learning rate, iterations, L2 lambda),
 * evaluation configuration (holdout split vs. external evaluation set), quality gating
 * (max Brier score threshold), and provenance metadata (model ID, schema version, source).
 *
 * Key design decisions:
 *   - '''Either-based error handling''': The `run` method returns `Either[String, RunResult]`
 *     for clean composition in test harnesses. The `main` method wraps this with exit codes.
 *   - '''All options have sensible defaults''': Only the training file and output directory
 *     are required. All hyperparameters and metadata fields have production-ready defaults.
 *   - '''Separation of parsing and execution''': `parseArgs` returns a `CliConfig` case class,
 *     which `runConfig` consumes. This makes the CLI logic testable without filesystem side effects.
 *
 * CLI entrypoint for training and exporting versioned poker action models. */
object TrainPokerActionModel:
  /** Parsed CLI configuration holding all training parameters and metadata. */
  private final case class CliConfig(
      trainingPath: String,
      outputDir: Path,
      evaluationPath: Option[String],
      learningRate: Double,
      iterations: Int,
      l2Lambda: Double,
      maxMeanBrierScore: Double,
      validationFraction: Double,
      splitSeed: Long,
      failOnGate: Boolean,
      modelId: String,
      schemaVersion: String,
      source: String,
      trainedAtEpochMillis: Long
  )

  /** Result of a successful training run: the output directory and the trained artifact. */
  final case class RunResult(outputDir: Path, artifact: TrainedPokerActionModel)

  /** Main entry point: trains the model and prints results to stdout, or errors to stderr with exit(1). */
  def main(args: Array[String]): Unit =
    run(args) match
      case Right(result) =>
        val artifact = result.artifact
        println(s"artifactDir: ${result.outputDir.toAbsolutePath.normalize()}")
        println(s"modelId: ${artifact.version.id}")
        println(s"schemaVersion: ${artifact.version.schemaVersion}")
        println(s"source: ${artifact.version.source}")
        println(s"trainedAtEpochMillis: ${artifact.version.trainedAtEpochMillis}")
        println(s"trainingSampleCount: ${artifact.trainingSampleCount}")
        println(s"evaluationSampleCount: ${artifact.evaluationSampleCount}")
        println(s"evaluationStrategy: ${artifact.evaluationStrategy}")
        println(s"maxMeanBrierScore: ${artifact.gate.maxMeanBrierScore}")
        println(f"meanBrierScore: ${artifact.calibration.meanBrierScore}%.8f")
        println(s"gatePassed: ${artifact.gatePassed}")
      case Left(error) =>
        System.err.println(error)
        sys.exit(1)

  /** Parses arguments and runs the training pipeline. Returns Left(error) or Right(result). */
  def run(args: Array[String]): Either[String, RunResult] =
    for
      config <- parseArgs(args)
      result <- runConfig(config)
    yield result

  /** Executes the training pipeline: reads data, trains model, saves artifact.
   * Wraps all exceptions as Left(error message).
   */
  private def runConfig(config: CliConfig): Either[String, RunResult] =
    try
      val trainingData = PokerActionTrainingDataIO.readTsv(config.trainingPath)
      val evaluationData = config.evaluationPath.map(PokerActionTrainingDataIO.readTsv).getOrElse(Vector.empty)
      val source = if config.source.trim.nonEmpty then config.source.trim else config.trainingPath

      val artifact = PokerActionModel.trainVersioned(
        trainingData = trainingData,
        learningRate = config.learningRate,
        iterations = config.iterations,
        l2Lambda = config.l2Lambda,
        evaluationData = evaluationData,
        validationFraction = config.validationFraction,
        splitSeed = config.splitSeed,
        maxMeanBrierScore = config.maxMeanBrierScore,
        failOnGate = config.failOnGate,
        modelId = config.modelId,
        schemaVersion = config.schemaVersion,
        source = source,
        trainedAtEpochMillis = config.trainedAtEpochMillis
      )
      PokerActionModelArtifactIO.save(config.outputDir, artifact)
      Right(RunResult(config.outputDir, artifact))
    catch
      case e: Exception => Left(s"training failed: ${e.getMessage}")

  /** Parses command-line arguments into a [[CliConfig]].
   *
   * The first two positional arguments are the training TSV path and output directory.
   * Remaining arguments are parsed as `--key=value` options via [[CliHelpers.parseOptionsAllowBlankValues]].
   */
  private def parseArgs(args: Array[String]): Either[String, CliConfig] =
    if args.contains("--help") || args.contains("-h") then Left(usage)
    else if args.length < 2 then Left(usage)
    else
      val trainingPath = args(0)
      val outputDir = Paths.get(args(1))
      val optionsRaw = args.drop(2)
      for
        options <- CliHelpers.parseOptionsAllowBlankValues(optionsRaw)
        evaluationPath = options.get("evaluationPath").map(_.trim).filter(_.nonEmpty)
        learningRate <- parseDoubleOption(options, "learningRate", 0.01)
        iterations <- parseIntOption(options, "iterations", 1000)
        _ <- if iterations > 0 then Right(()) else Left("--iterations must be positive")
        l2Lambda <- parseDoubleOption(options, "l2Lambda", 0.001)
        maxBrier <- parseDoubleOption(options, "maxMeanBrierScore", 1.0)
        validationFraction <- parseDoubleOption(options, "validationFraction", 0.2)
        splitSeed <- parseLongOption(options, "splitSeed", 1L)
        failOnGate <- parseBooleanOption(options, "failOnGate", false)
        trainedAt <- parseLongOption(options, "trainedAtEpochMillis", System.currentTimeMillis())
        modelId = options.getOrElse("modelId", s"model-${System.currentTimeMillis()}").trim
        schemaVersion = options.getOrElse("schemaVersion", "poker-action-model-v1").trim
        source = options.getOrElse("source", trainingPath).trim
        _ <- if modelId.nonEmpty then Right(()) else Left("--modelId must be non-empty")
        _ <- if schemaVersion.nonEmpty then Right(()) else Left("--schemaVersion must be non-empty")
        _ <- if source.nonEmpty then Right(()) else Left("--source must be non-empty")
      yield CliConfig(
        trainingPath = trainingPath,
        outputDir = outputDir,
        evaluationPath = evaluationPath,
        learningRate = learningRate,
        iterations = iterations,
        l2Lambda = l2Lambda,
        maxMeanBrierScore = maxBrier,
        validationFraction = validationFraction,
        splitSeed = splitSeed,
        failOnGate = failOnGate,
        modelId = modelId,
        schemaVersion = schemaVersion,
        source = source,
        trainedAtEpochMillis = trainedAt
      )

  private def parseDoubleOption(options: Map[String, String], key: String, default: Double): Either[String, Double] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) =>
        try
          val value = raw.toDouble
          if value.isNaN || value.isInfinite then Left(s"--$key must be finite")
          else Right(value)
        catch
          case _: NumberFormatException => Left(s"--$key must be a valid number, got '$raw'")

  private def parseIntOption(options: Map[String, String], key: String, default: Int): Either[String, Int] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) =>
        try Right(raw.toInt)
        catch
          case _: NumberFormatException => Left(s"--$key must be a valid integer, got '$raw'")

  private def parseLongOption(options: Map[String, String], key: String, default: Long): Either[String, Long] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) =>
        try Right(raw.toLong)
        catch
          case _: NumberFormatException => Left(s"--$key must be a valid long, got '$raw'")

  private def parseBooleanOption(options: Map[String, String], key: String, default: Boolean): Either[String, Boolean] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) =>
        raw.trim.toLowerCase match
          case "true"  => Right(true)
          case "false" => Right(false)
          case other   => Left(s"--$key must be true or false, got '$other'")

  private val usage =
    """Usage: TrainPokerActionModel <trainingTsvPath> <artifactOutputDir> [--key=value ...]
      |
      |Required TSV columns:
      |  street  board  potBefore  toCall  position  stackBefore  action  holeCards
      |
      |Options:
      |  --evaluationPath=<path>           Optional external evaluation TSV
      |  --learningRate=<double>           Default 0.01
      |  --iterations=<int>                Default 1000
      |  --l2Lambda=<double>               Default 0.001
      |  --maxMeanBrierScore=<double>      Default 1.0
      |  --validationFraction=<double>     Default 0.2 (used when no evaluationPath)
      |  --splitSeed=<long>                Default 1
      |  --failOnGate=<true|false>         Default false
      |  --modelId=<id>                    Default model-<currentEpochMillis>
      |  --schemaVersion=<string>          Default poker-action-model-v1
      |  --source=<string>                 Default training TSV path
      |  --trainedAtEpochMillis=<long>     Default current epoch millis
      |""".stripMargin
