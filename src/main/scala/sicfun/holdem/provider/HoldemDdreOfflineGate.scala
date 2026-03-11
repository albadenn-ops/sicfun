package sicfun.holdem.provider
import sicfun.holdem.types.*
import sicfun.holdem.cli.*
import sicfun.holdem.io.*
import sicfun.holdem.model.*
import sicfun.holdem.equity.*

import sicfun.core.CollapseMetrics

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import scala.jdk.CollectionConverters.*

/** Offline validation gate for DDRE ONNX artifacts against exported self-play data. */
object HoldemDdreOfflineGate:
  final case class GateThresholds(
      minSamples: Int = 100,
      maxMeanNll: Double = 8.0,
      maxMeanKlVsBayes: Double = 8.0,
      maxBlockerViolationRate: Double = 0.0,
      maxFailureRate: Double = 0.0,
      maxP95LatencyMillis: Double = 50.0
  ):
    require(minSamples > 0, "minSamples must be positive")
    require(maxMeanNll >= 0.0 && maxMeanNll.isFinite, "maxMeanNll must be finite and non-negative")
    require(maxMeanKlVsBayes >= 0.0 && maxMeanKlVsBayes.isFinite, "maxMeanKlVsBayes must be finite and non-negative")
    require(
      maxBlockerViolationRate >= 0.0 && maxBlockerViolationRate <= 1.0 && maxBlockerViolationRate.isFinite,
      "maxBlockerViolationRate must be in [0,1]"
    )
    require(maxFailureRate >= 0.0 && maxFailureRate <= 1.0 && maxFailureRate.isFinite, "maxFailureRate must be in [0,1]")
    require(maxP95LatencyMillis >= 0.0 && maxP95LatencyMillis.isFinite, "maxP95LatencyMillis must be finite and non-negative")

  final case class GateSummary(
      totalSamples: Int,
      successfulSamples: Int,
      meanNll: Double,
      meanKlVsBayes: Double,
      blockerViolationRate: Double,
      failureRate: Double,
      p50LatencyMillis: Double,
      p95LatencyMillis: Double,
      gatePass: Boolean,
      artifactId: String,
      artifactDir: Path
  )

  private final case class CliConfig(
      datasetPath: Path,
      artifactDir: Path,
      actionModelDir: Option[Path],
      sampleLimit: Int,
      summaryPath: Option[Path],
      writeMetadata: Boolean,
      thresholds: GateThresholds
  )

  private final case class SampleMetrics(
      nll: Double,
      klVsBayes: Double,
      blockerViolation: Boolean,
      latencyMillis: Double
  )

  private final case class EvaluatedSamples(
      successes: Vector[SampleMetrics],
      failureCount: Int
  ):
    def totalSamples: Int = successes.length + failureCount

  private val Eps = 1e-12

  def main(args: Array[String]): Unit =
    val wantsHelp = args.contains("--help") || args.contains("-h")
    run(args) match
      case Right(summary) =>
        printSummary(summary)
      case Left(err) =>
        if wantsHelp then println(err)
        else
          System.err.println(err)
          sys.exit(1)

  def run(args: Array[String]): Either[String, GateSummary] =
    for
      config <- parseArgs(args)
      summary <- evaluate(config)
    yield summary

  private def evaluate(config: CliConfig): Either[String, GateSummary] =
    for
      artifact <- loadArtifact(config)
      samples <- loadSamples(config)
    yield
      val evaluation = evaluateSamples(config, samples, artifact)
      val summary = summarizeEvaluation(config, artifact, evaluation)
      persistMetadata(config, artifact, summary)
      config.summaryPath.foreach(path => writeSummary(path, summary))
      summary

  private def loadArtifact(
      config: CliConfig
  ): Either[String, HoldemDdreArtifactIO.OnnxArtifact] =
    HoldemDdreArtifactIO.load(config.artifactDir).flatMap { artifact =>
      if artifact.modelFile.nonEmpty then Right(artifact)
      else Left("artifact model.file must be non-empty")
    }

  private def evaluateSamples(
      config: CliConfig,
      samples: Vector[HoldemDdreDatasetIO.Sample],
      artifact: HoldemDdreArtifactIO.OnnxArtifact
  ): EvaluatedSamples =
    val onnxConfig = buildRuntimeConfig(config, artifact)
    val resolvedActionModel = resolveActionModel(config)
    val results = samples.map(sample => evaluateSample(sample, onnxConfig, resolvedActionModel))
    EvaluatedSamples(
      successes = results.collect { case Right(value) => value },
      failureCount = results.count(_.isLeft)
    )

  private def resolveActionModel(config: CliConfig): PokerActionModel =
    config.actionModelDir
      .map(path => PokerActionModelArtifactIO.load(path).model)
      .getOrElse(PokerActionModel.uniform)

  private def buildRuntimeConfig(
      config: CliConfig,
      artifact: HoldemDdreArtifactIO.OnnxArtifact
  ): HoldemDdreOnnxRuntime.Config =
    HoldemDdreOnnxRuntime.configFromArtifact(
      directory = config.artifactDir,
      artifact = artifact,
      allowExperimental = true
    )

  private def summarizeEvaluation(
      config: CliConfig,
      artifact: HoldemDdreArtifactIO.OnnxArtifact,
      evaluation: EvaluatedSamples
  ): GateSummary =
    val failureRate =
      if evaluation.totalSamples == 0 then 1.0
      else evaluation.failureCount.toDouble / evaluation.totalSamples.toDouble
    val latencyValues = evaluation.successes.map(_.latencyMillis).sorted
    val meanNll = meanOrInfinity(evaluation.successes.map(_.nll))
    val meanKl = meanOrInfinity(evaluation.successes.map(_.klVsBayes))
    val blockerViolationRate =
      if evaluation.successes.nonEmpty then
        evaluation.successes.count(_.blockerViolation).toDouble / evaluation.successes.length.toDouble
      else 1.0
    val p50Latency = quantileOrInfinity(latencyValues, 0.5)
    val p95Latency = quantileOrInfinity(latencyValues, 0.95)
    val gatePass = gatePasses(config.thresholds, evaluation, failureRate, blockerViolationRate, meanNll, meanKl, p95Latency)
    GateSummary(
      totalSamples = evaluation.totalSamples,
      successfulSamples = evaluation.successes.length,
      meanNll = meanNll,
      meanKlVsBayes = meanKl,
      blockerViolationRate = blockerViolationRate,
      failureRate = failureRate,
      p50LatencyMillis = p50Latency,
      p95LatencyMillis = p95Latency,
      gatePass = gatePass,
      artifactId = artifact.artifactId,
      artifactDir = config.artifactDir
    )

  private def meanOrInfinity(values: Vector[Double]): Double =
    if values.nonEmpty then values.sum / values.length.toDouble
    else Double.PositiveInfinity

  private def quantileOrInfinity(sorted: Vector[Double], q: Double): Double =
    if sorted.nonEmpty then quantile(sorted, q)
    else Double.PositiveInfinity

  private def gatePasses(
      thresholds: GateThresholds,
      evaluation: EvaluatedSamples,
      failureRate: Double,
      blockerViolationRate: Double,
      meanNll: Double,
      meanKl: Double,
      p95Latency: Double
  ): Boolean =
    evaluation.totalSamples >= thresholds.minSamples &&
      failureRate <= thresholds.maxFailureRate + Eps &&
      blockerViolationRate <= thresholds.maxBlockerViolationRate + Eps &&
      meanNll <= thresholds.maxMeanNll + Eps &&
      meanKl <= thresholds.maxMeanKlVsBayes + Eps &&
      p95Latency <= thresholds.maxP95LatencyMillis + Eps

  private def persistMetadata(
      config: CliConfig,
      artifact: HoldemDdreArtifactIO.OnnxArtifact,
      summary: GateSummary
  ): Unit =
    if config.writeMetadata then
      HoldemDdreArtifactIO.save(
        config.artifactDir,
        artifact.copy(
          validationStatus = if summary.gatePass then "validated" else "failed",
          decisionDrivingAllowed = summary.gatePass,
          validationSampleCount = Some(summary.totalSamples),
          meanNll = Some(summary.meanNll),
          meanKlVsBayes = Some(summary.meanKlVsBayes),
          blockerViolationRate = Some(summary.blockerViolationRate),
          failureRate = Some(summary.failureRate),
          p50LatencyMillis = Some(summary.p50LatencyMillis),
          p95LatencyMillis = Some(summary.p95LatencyMillis),
          gateMinSamples = Some(config.thresholds.minSamples),
          gateMaxMeanNll = Some(config.thresholds.maxMeanNll),
          gateMaxMeanKlVsBayes = Some(config.thresholds.maxMeanKlVsBayes),
          gateMaxBlockerViolationRate = Some(config.thresholds.maxBlockerViolationRate),
          gateMaxFailureRate = Some(config.thresholds.maxFailureRate),
          gateMaxP95LatencyMillis = Some(config.thresholds.maxP95LatencyMillis)
        )
      )

  private def evaluateSample(
      sample: HoldemDdreDatasetIO.Sample,
      runtimeConfig: HoldemDdreOnnxRuntime.Config,
      actionModel: PokerActionModel
  ): Either[String, SampleMetrics] =
    val hypotheses = sample.prior.weights.keysIterator.toVector.sortBy(HoleCardsIndex.idOf)
    val priorArray = hypotheses.map(sample.prior.probabilityOf).toArray
    val observations = sample.observations.map(obs => obs.action -> obs.state)
    val likelihoods = HoldemDdreProvider.buildLikelihoodMatrix(observations, actionModel, hypotheses)
    val startedAt = System.nanoTime()
    HoldemDdreOnnxRuntime
      .inferPosterior(
        prior = priorArray,
        likelihoods = likelihoods,
        observationCount = observations.length,
        hypothesisCount = hypotheses.length,
        config = runtimeConfig
      )
      .flatMap { posteriorRaw =>
        val latencyMillis = math.max(0.0, (System.nanoTime() - startedAt).toDouble / 1_000_000.0)
        val rawDistribution =
          HoldemDdreProvider.distributionFromPosteriorArray(hypotheses, posteriorRaw, "onnx-gate")
        rawDistribution.flatMap { rawPosterior =>
          val blockerViolation = rawPosterior.weights.keysIterator.exists { hand =>
            !hand.isDisjointFrom(sample.heroHole) || sample.state.board.asSet.exists(hand.contains)
          }
          HoldemDdreProvider
            .applyLegalMask(rawPosterior, sample.heroHole, sample.state.board)
            .map { maskedPosterior =>
              val realized = math.max(maskedPosterior.probabilityOf(sample.villainHole), Eps)
              SampleMetrics(
                nll = -math.log(realized),
                klVsBayes = CollapseMetrics.klDivergence(sample.bayesPosterior, maskedPosterior),
                blockerViolation = blockerViolation,
                latencyMillis = latencyMillis
              )
            }
        }
      }

  private def loadSamples(config: CliConfig): Either[String, Vector[HoldemDdreDatasetIO.Sample]] =
    try
      val loaded = HoldemDdreDatasetIO.read(config.datasetPath)
      val limited =
        if config.sampleLimit >= loaded.length then loaded
        else loaded.take(config.sampleLimit)
      if limited.nonEmpty then Right(limited)
      else Left(s"DDRE dataset produced no samples: ${config.datasetPath}")
    catch
      case ex: Exception =>
        Left(Option(ex.getMessage).getOrElse(ex.getClass.getSimpleName))

  private def parseArgs(args: Array[String]): Either[String, CliConfig] =
    if args.contains("--help") || args.contains("-h") then Left(usage)
    else
      for
        options <- CliHelpers.parseOptions(args)
        datasetPath <- requiredPath(options, "dataset")
        artifactDir <- requiredPath(options, "artifactDir")
        actionModelDir <- optionalPathOption(options, "actionModel")
        sampleLimit <- CliHelpers.parseIntOptionEither(options, "sampleLimit", Int.MaxValue)
        _ <- if sampleLimit > 0 then Right(()) else Left("--sampleLimit must be > 0")
        summaryPath <- optionalPathOption(options, "summaryPath")
        writeMetadata <- CliHelpers.parseExtendedBooleanOptionEither(options, "writeMetadata", true)
        minSamples <- CliHelpers.parseIntOptionEither(options, "minSamples", 100)
        maxMeanNll <- CliHelpers.parseDoubleOptionEither(options, "maxMeanNll", 8.0)
        maxMeanKlVsBayes <- CliHelpers.parseDoubleOptionEither(options, "maxMeanKlVsBayes", 8.0)
        maxBlockerViolationRate <- CliHelpers.parseDoubleOptionEither(options, "maxBlockerViolationRate", 0.0)
        maxFailureRate <- CliHelpers.parseDoubleOptionEither(options, "maxFailureRate", 0.0)
        maxP95LatencyMillis <- CliHelpers.parseDoubleOptionEither(options, "maxP95LatencyMillis", 50.0)
      yield
        CliConfig(
          datasetPath = datasetPath,
          artifactDir = artifactDir,
          actionModelDir = actionModelDir,
          sampleLimit = sampleLimit,
          summaryPath = summaryPath,
          writeMetadata = writeMetadata,
          thresholds = GateThresholds(
            minSamples = minSamples,
            maxMeanNll = maxMeanNll,
            maxMeanKlVsBayes = maxMeanKlVsBayes,
            maxBlockerViolationRate = maxBlockerViolationRate,
            maxFailureRate = maxFailureRate,
            maxP95LatencyMillis = maxP95LatencyMillis
          )
        )

  private def requiredPath(options: Map[String, String], key: String): Either[String, Path] =
    options.get(key).map(value => Right(Paths.get(value))).getOrElse(Left(s"--$key is required"))

  private def optionalPathOption(options: Map[String, String], key: String): Either[String, Option[Path]] =
    Right(options.get(key).map(Paths.get(_)))

  private def quantile(sorted: Vector[Double], q: Double): Double =
    if sorted.length == 1 then sorted.head
    else
      val p = q * (sorted.length - 1).toDouble
      val lo = math.floor(p).toInt
      val hi = math.ceil(p).toInt
      if lo == hi then sorted(lo)
      else
        val w = p - lo.toDouble
        sorted(lo) * (1.0 - w) + sorted(hi) * w

  private def printSummary(summary: GateSummary): Unit =
    println("=== Holdem DDRE Offline Gate ===")
    println(s"artifactId: ${summary.artifactId}")
    println(s"artifactDir: ${summary.artifactDir.toAbsolutePath.normalize()}")
    println(s"totalSamples: ${summary.totalSamples}")
    println(s"successfulSamples: ${summary.successfulSamples}")
    println(f"meanNll: ${summary.meanNll}%.6f")
    println(f"meanKlVsBayes: ${summary.meanKlVsBayes}%.6f")
    println(f"blockerViolationRate: ${summary.blockerViolationRate}%.6f")
    println(f"failureRate: ${summary.failureRate}%.6f")
    println(f"p50LatencyMillis: ${summary.p50LatencyMillis}%.6f")
    println(f"p95LatencyMillis: ${summary.p95LatencyMillis}%.6f")
    println(s"gate: ${if summary.gatePass then "PASS" else "FAIL"}")

  private def writeSummary(path: Path, summary: GateSummary): Unit =
    Option(path.getParent).foreach(parent => Files.createDirectories(parent))
    val lines = Vector(
      "artifactId\ttotalSamples\tsuccessfulSamples\tmeanNll\tmeanKlVsBayes\tblockerViolationRate\tfailureRate\tp50LatencyMillis\tp95LatencyMillis\tgatePass",
      Vector(
        summary.artifactId,
        summary.totalSamples.toString,
        summary.successfulSamples.toString,
        summary.meanNll.toString,
        summary.meanKlVsBayes.toString,
        summary.blockerViolationRate.toString,
        summary.failureRate.toString,
        summary.p50LatencyMillis.toString,
        summary.p95LatencyMillis.toString,
        summary.gatePass.toString
      ).mkString("\t")
    )
    Files.write(path, lines.asJava, StandardCharsets.UTF_8)

  private val usage =
    """Usage:
      |  runMain sicfun.holdem.provider.HoldemDdreOfflineGate --dataset=<ddre-training.tsv> --artifactDir=<dir> [--key=value ...]
      |
      |Options:
      |  --actionModel=<dir>                 Optional poker action-model artifact used for likelihood reconstruction
      |  --sampleLimit=<int>                 Maximum rows to evaluate (default all)
      |  --writeMetadata=true|false          Persist gate metrics back into artifact metadata (default true)
      |  --summaryPath=<path>                Optional TSV summary output
      |  --minSamples=<int>                  Minimum samples required to pass (default 100)
      |  --maxMeanNll=<double>               Maximum allowed mean NLL (default 8.0)
      |  --maxMeanKlVsBayes=<double>         Maximum allowed mean KL(ddre || bayes) in bits (default 8.0)
      |  --maxBlockerViolationRate=<double>  Maximum raw illegal-output rate (default 0.0)
      |  --maxFailureRate=<double>           Maximum inference failure rate (default 0.0)
      |  --maxP95LatencyMillis=<double>      Maximum p95 inference latency in ms (default 50.0)
      |""".stripMargin
