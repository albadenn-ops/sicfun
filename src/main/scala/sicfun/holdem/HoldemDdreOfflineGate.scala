package sicfun.holdem

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
      artifact <- HoldemDdreArtifactIO.load(config.artifactDir)
      _ <- if artifact.modelFile.nonEmpty then Right(()) else Left("artifact model.file must be non-empty")
      samples <- loadSamples(config)
    yield
      val actionModel =
        config.actionModelDir
          .map(path => PokerActionModelArtifactIO.load(path).model)
          .getOrElse(PokerActionModel.uniform)

      val runtimeConfig = HoldemDdreOnnxRuntime.configFromArtifact(
        directory = config.artifactDir,
        artifact = artifact,
        allowExperimental = true
      )

      val results = samples.map(sample => evaluateSample(sample, runtimeConfig, actionModel))
      val totalSamples = results.length
      val successes = results.collect { case Right(value) => value }
      val failureCount = totalSamples - successes.length
      val failureRate =
        if totalSamples == 0 then 1.0 else failureCount.toDouble / totalSamples.toDouble

      val latencyValues = successes.map(_.latencyMillis).sorted
      val meanNll =
        if successes.nonEmpty then successes.map(_.nll).sum / successes.length.toDouble
        else Double.PositiveInfinity
      val meanKl =
        if successes.nonEmpty then successes.map(_.klVsBayes).sum / successes.length.toDouble
        else Double.PositiveInfinity
      val blockerViolationRate =
        if successes.nonEmpty then successes.count(_.blockerViolation).toDouble / successes.length.toDouble
        else 1.0
      val p50Latency = if latencyValues.nonEmpty then quantile(latencyValues, 0.5) else Double.PositiveInfinity
      val p95Latency = if latencyValues.nonEmpty then quantile(latencyValues, 0.95) else Double.PositiveInfinity

      val gatePass =
        totalSamples >= config.thresholds.minSamples &&
          failureRate <= config.thresholds.maxFailureRate + Eps &&
          blockerViolationRate <= config.thresholds.maxBlockerViolationRate + Eps &&
          meanNll <= config.thresholds.maxMeanNll + Eps &&
          meanKl <= config.thresholds.maxMeanKlVsBayes + Eps &&
          p95Latency <= config.thresholds.maxP95LatencyMillis + Eps

      val summary = GateSummary(
        totalSamples = totalSamples,
        successfulSamples = successes.length,
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

      if config.writeMetadata then
        HoldemDdreArtifactIO.save(
          config.artifactDir,
          artifact.copy(
            validationStatus = if gatePass then "validated" else "failed",
            decisionDrivingAllowed = gatePass,
            validationSampleCount = Some(totalSamples),
            meanNll = Some(meanNll),
            meanKlVsBayes = Some(meanKl),
            blockerViolationRate = Some(blockerViolationRate),
            failureRate = Some(failureRate),
            p50LatencyMillis = Some(p50Latency),
            p95LatencyMillis = Some(p95Latency),
            gateMinSamples = Some(config.thresholds.minSamples),
            gateMaxMeanNll = Some(config.thresholds.maxMeanNll),
            gateMaxMeanKlVsBayes = Some(config.thresholds.maxMeanKlVsBayes),
            gateMaxBlockerViolationRate = Some(config.thresholds.maxBlockerViolationRate),
            gateMaxFailureRate = Some(config.thresholds.maxFailureRate),
            gateMaxP95LatencyMillis = Some(config.thresholds.maxP95LatencyMillis)
          )
        )

      config.summaryPath.foreach(path => writeSummary(path, summary))
      summary

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
      val options = args.flatMap { token =>
        token.split("=", 2) match
          case Array(key, value) if key.startsWith("--") && value.nonEmpty =>
            Some(key.drop(2) -> value)
          case _ => None
      }.toMap

      for
        datasetPath <- requiredPath(options, "dataset")
        artifactDir <- requiredPath(options, "artifactDir")
        actionModelDir = options.get("actionModel").map(Paths.get(_))
        sampleLimit <- parseIntOption(options, "sampleLimit", Int.MaxValue)
        _ <- if sampleLimit > 0 then Right(()) else Left("--sampleLimit must be > 0")
        summaryPath = options.get("summaryPath").map(Paths.get(_))
        writeMetadata = options.get("writeMetadata").forall(parseBoolean)
        minSamples <- parseIntOption(options, "minSamples", 100)
        maxMeanNll <- parseDoubleOption(options, "maxMeanNll", 8.0)
        maxMeanKlVsBayes <- parseDoubleOption(options, "maxMeanKlVsBayes", 8.0)
        maxBlockerViolationRate <- parseDoubleOption(options, "maxBlockerViolationRate", 0.0)
        maxFailureRate <- parseDoubleOption(options, "maxFailureRate", 0.0)
        maxP95LatencyMillis <- parseDoubleOption(options, "maxP95LatencyMillis", 50.0)
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

  private def parseIntOption(options: Map[String, String], key: String, default: Int): Either[String, Int] =
    options.get(key) match
      case None => Right(default)
      case Some(value) =>
        value.toIntOption.toRight(s"--$key must be an integer")

  private def parseDoubleOption(options: Map[String, String], key: String, default: Double): Either[String, Double] =
    options.get(key) match
      case None => Right(default)
      case Some(value) =>
        value.toDoubleOption.toRight(s"--$key must be a number")

  private def parseBoolean(raw: String): Boolean =
    GpuRuntimeSupport.parseTruthy(raw)

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
      |  runMain sicfun.holdem.HoldemDdreOfflineGate --dataset=<ddre-training.tsv> --artifactDir=<dir> [--key=value ...]
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
