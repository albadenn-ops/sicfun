package sicfun.holdem.analysis
import sicfun.holdem.cli.*
import sicfun.holdem.io.*
import sicfun.holdem.model.*

import java.nio.file.{Path, Paths}

/** CLI entry point for batch signal generation from a hand-state snapshot and trained model.
  *
  * Reads a [[HandState]] from a snapshot directory and a [[TrainedPokerActionModel]] from
  * a model artifact directory, then generates [[SignalEnvelope]] instances for each matching
  * event. Results are written (or appended) to a TSV audit log via [[SignalAuditLogIO]].
  *
  * '''Usage:'''
  * {{{
  * GenerateSignals <snapshotDir> <modelArtifactDir> <outputAuditLog> [--key=value ...]
  * }}}
  *
  * '''Options:'''
  *   - `--warningThreshold=<double>` — risk score threshold for Warning level (default 0.4)
  *   - `--criticalThreshold=<double>` — risk score threshold for Critical level (default 0.7)
  *   - `--generatedAtEpochMillis=<long>` — fixed generation timestamp for all signals
  *   - `--append=<true|false>` — append to existing log instead of overwriting (default false)
  *   - `--playerId=<id>` — filter events to a specific player
  *   - `--minSequence=<long>` / `--maxSequence=<long>` — filter events by sequence range
  */
object GenerateSignals:
  /** Internal configuration parsed from CLI arguments. */
  private final case class CliConfig(
      snapshotDirectory: Path,
      modelArtifactDirectory: Path,
      outputAuditLog: Path,
      warningThreshold: Double,
      criticalThreshold: Double,
      generatedAtEpochMillis: Option[Long],
      append: Boolean,
      playerId: Option[String],
      minSequence: Option[Long],
      maxSequence: Option[Long]
  )

  /** Summary of a completed signal generation run.
    *
    * @param outputAuditLog path to the written/appended audit log file
    * @param signalCount    total number of signals generated
    * @param levelCounts    breakdown of signal counts by [[SignalLevel]]
    */
  final case class RunResult(
      outputAuditLog: Path,
      signalCount: Int,
      levelCounts: Map[SignalLevel, Int]
  )

  def main(args: Array[String]): Unit =
    run(args) match
      case Right(result) =>
        println(s"outputAuditLog: ${result.outputAuditLog.toAbsolutePath.normalize()}")
        println(s"signalCount: ${result.signalCount}")
        SignalLevel.values.foreach { level =>
          val count = result.levelCounts.getOrElse(level, 0)
          println(s"${level.toString.toLowerCase}Count: $count")
        }
      case Left(error) =>
        System.err.println(error)
        sys.exit(1)

  def run(args: Array[String]): Either[String, RunResult] =
    for
      config <- parseArgs(args)
      result <- runConfig(config)
    yield result

  private def runConfig(config: CliConfig): Either[String, RunResult] =
    try
      val state = HandStateSnapshotIO.load(config.snapshotDirectory)
      val artifact = PokerActionModelArtifactIO.load(config.modelArtifactDirectory)
      val generatedAt = config.generatedAtEpochMillis.getOrElse(System.currentTimeMillis())

      val filtered = state.events.filter { event =>
        val byPlayer = config.playerId.forall(_ == event.playerId)
        val byMinSeq = config.minSequence.forall(event.sequenceInHand >= _)
        val byMaxSeq = config.maxSequence.forall(event.sequenceInHand <= _)
        byPlayer && byMinSeq && byMaxSeq
      }

      val signals = filtered.map { event =>
        SignalBuilder.actionRisk(
          event = event,
          artifact = artifact,
          snapshotDirectory = config.snapshotDirectory.toString,
          modelArtifactDirectory = config.modelArtifactDirectory.toString,
          generatedAtEpochMillis = generatedAt,
          warningThreshold = config.warningThreshold,
          criticalThreshold = config.criticalThreshold
        )
      }

      if config.append then
        signals.foreach(signal => SignalAuditLogIO.append(config.outputAuditLog, signal))
      else
        SignalAuditLogIO.write(config.outputAuditLog, signals)

      val counts = signals.groupBy(_.level).view.mapValues(_.length).toMap
      Right(RunResult(config.outputAuditLog, signals.length, counts))
    catch
      case e: Exception =>
        Left(s"signal generation failed: ${e.getMessage}")

  private def parseArgs(args: Array[String]): Either[String, CliConfig] =
    if args.contains("--help") || args.contains("-h") then Left(usage)
    else if args.length < 3 then Left(usage)
    else
      val snapshotDirectory = Paths.get(args(0))
      val modelArtifactDirectory = Paths.get(args(1))
      val outputAuditLog = Paths.get(args(2))
      val optionsRaw = args.drop(3)

      for
        options <- CliHelpers.parseOptionsAllowBlankValues(optionsRaw)
        warningThreshold <- parseDoubleOption(options, "warningThreshold", 0.4)
        criticalThreshold <- parseDoubleOption(options, "criticalThreshold", 0.7)
        _ <- if warningThreshold >= 0.0 && warningThreshold <= 1.0 then Right(()) else Left("--warningThreshold must be in [0, 1]")
        _ <- if criticalThreshold >= warningThreshold && criticalThreshold <= 1.0 then Right(()) else Left("--criticalThreshold must be in [warningThreshold, 1]")
        generatedAt <- parseOptionalLongOption(options, "generatedAtEpochMillis")
        append <- parseBooleanOption(options, "append", false)
        playerId = options.get("playerId").map(_.trim).filter(_.nonEmpty)
        minSequence <- parseOptionalLongOption(options, "minSequence")
        maxSequence <- parseOptionalLongOption(options, "maxSequence")
        _ <- (minSequence, maxSequence) match
          case (Some(min), Some(max)) if min > max =>
            Left("--minSequence must be <= --maxSequence")
          case _ => Right(())
      yield CliConfig(
        snapshotDirectory = snapshotDirectory,
        modelArtifactDirectory = modelArtifactDirectory,
        outputAuditLog = outputAuditLog,
        warningThreshold = warningThreshold,
        criticalThreshold = criticalThreshold,
        generatedAtEpochMillis = generatedAt,
        append = append,
        playerId = playerId,
        minSequence = minSequence,
        maxSequence = maxSequence
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

  private def parseOptionalLongOption(options: Map[String, String], key: String): Either[String, Option[Long]] =
    options.get(key) match
      case None => Right(None)
      case Some(raw) =>
        try Right(Some(raw.toLong))
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
    """Usage: GenerateSignals <snapshotDirectory> <modelArtifactDirectory> <outputAuditLogPath> [--key=value ...]
      |
      |Options:
      |  --warningThreshold=<double>       Default 0.4
      |  --criticalThreshold=<double>      Default 0.7
      |  --generatedAtEpochMillis=<long>   Optional fixed generation timestamp for all signals
      |  --append=<true|false>             Default false
      |  --playerId=<id>                   Optional filter by player
      |  --minSequence=<long>              Optional lower bound on sequenceInHand
      |  --maxSequence=<long>              Optional upper bound on sequenceInHand
      |""".stripMargin
