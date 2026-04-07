package sicfun.holdem.model
import sicfun.holdem.types.*

import sicfun.core.MultinomialLogistic

import java.io.{BufferedReader, BufferedWriter}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import java.util.Properties
import scala.jdk.CollectionConverters.*

/**
 * Directory-based artifact persistence for trained poker action models in sicfun.
 *
 * This object handles the full save/load lifecycle for [[TrainedPokerActionModel]] artifacts.
 * Each artifact is stored as a directory containing four files that separate metadata from
 * numeric parameters, enabling easy inspection and comparison between model versions.
 *
 * Key design decisions:
 *   - '''Four-file layout''': Metadata (Properties), weights (TSV matrix), bias (TSV vector),
 *     and category index (TSV mapping) are stored separately. This allows diffing individual
 *     components and inspecting metadata without loading the full weight matrix.
 *   - '''Lossless numeric round-tripping''': All doubles are serialized with
 *     [[java.lang.Double.toString]], which preserves all 64 bits of IEEE 754 precision.
 *   - '''Format versioning''': The `format.version` key in metadata allows future schema
 *     evolution without breaking existing artifacts.
 *   - '''Lifecycle state persistence''': Retirement timestamp and reason are stored in
 *     metadata, so a model's full lifecycle is recoverable from the artifact directory alone.
 *
 * Serializes and deserializes [[TrainedPokerActionModel]] artifacts to/from a directory.
  *
  * '''Directory layout:'''
  *   - `metadata.properties` — model version, calibration summary, gate config, sample counts
  *   - `weights.tsv` — logistic regression weight matrix (one row per class, tab-delimited)
  *   - `bias.tsv` — logistic regression bias vector (single tab-delimited row)
  *   - `category-index.tsv` — mapping from [[PokerAction.Category]] names to integer indices
  *
  * The format is versioned (`format.version = "1"`) and loading validates that the stored
  * version matches. All numeric values are serialized with [[java.lang.Double.toString]] for
  * lossless round-tripping.
  */
object PokerActionModelArtifactIO:
  private val FormatVersion = "1"
  private val MetadataFileName = "metadata.properties"
  private val WeightsFileName = "weights.tsv"
  private val BiasFileName = "bias.tsv"
  private val CategoryIndexFileName = "category-index.tsv"

  /** Persists a trained model artifact to a directory, creating it if necessary.
    *
    * @param directory target directory (string form)
    * @param artifact  the trained model to serialize
    */
  def save(directory: String, artifact: TrainedPokerActionModel): Unit =
    save(Paths.get(directory), artifact)

  /** Persists a trained model artifact to a directory, creating it if necessary. */
  def save(directory: Path, artifact: TrainedPokerActionModel): Unit =
    Files.createDirectories(directory)
    writeMetadata(directory.resolve(MetadataFileName), artifact)
    writeWeights(directory.resolve(WeightsFileName), artifact.model.logistic.weights)
    writeBias(directory.resolve(BiasFileName), artifact.model.logistic.bias)
    writeCategoryIndex(directory.resolve(CategoryIndexFileName), artifact.model.categoryIndex)

  /** Loads a trained model artifact from a directory.
    *
    * @param directory artifact directory (string form)
    * @return the deserialized [[TrainedPokerActionModel]]
    * @throws IllegalArgumentException if the directory is missing or the format version is unsupported
    */
  def load(directory: String): TrainedPokerActionModel =
    load(Paths.get(directory))

  /** Loads a trained model artifact from a directory. */
  def load(directory: Path): TrainedPokerActionModel =
    require(Files.isDirectory(directory), s"artifact directory does not exist: $directory")
    val metadata = readMetadata(directory.resolve(MetadataFileName))
    val format = readRequired(metadata, "format.version")
    require(format == FormatVersion, s"unsupported artifact format version: $format")

    val classCount = readRequired(metadata, "logistic.classCount").toInt
    val featureCount = readRequired(metadata, "logistic.featureCount").toInt
    val weights = readWeights(directory.resolve(WeightsFileName), classCount, featureCount)
    val bias = readBias(directory.resolve(BiasFileName), classCount)
    val logistic = MultinomialLogistic(weights, bias)
    val categoryIndex = readCategoryIndex(directory.resolve(CategoryIndexFileName))
    val model = PokerActionModel(logistic, categoryIndex, featureCount)

    val version = ModelVersion(
      id = readRequired(metadata, "model.id"),
      schemaVersion = readRequired(metadata, "model.schemaVersion"),
      source = readRequired(metadata, "model.source"),
      trainedAtEpochMillis = readRequired(metadata, "model.trainedAtEpochMillis").toLong
    )
    val calibration = CalibrationSummary(
      meanBrierScore = readRequired(metadata, "calibration.meanBrierScore").toDouble,
      sampleCount = readRequired(metadata, "calibration.sampleCount").toInt,
      uniformBaselineBrier = readOptional(metadata, "calibration.uniformBaselineBrier").map(_.toDouble).getOrElse(-1.0),
      majorityBaselineBrier = readOptional(metadata, "calibration.majorityBaselineBrier").map(_.toDouble).getOrElse(-1.0)
    )
    val gate = CalibrationGate(
      maxMeanBrierScore = readRequired(metadata, "gate.maxMeanBrierScore").toDouble
    )

    TrainedPokerActionModel(
      version = version,
      model = model,
      calibration = calibration,
      gate = gate,
      trainingSampleCount = readRequired(metadata, "training.sampleCount").toInt,
      evaluationSampleCount = readRequired(metadata, "evaluation.sampleCount").toInt,
      evaluationStrategy = readRequired(metadata, "evaluation.strategy"),
      validationFraction = readOptional(metadata, "evaluation.validationFraction").map(_.toDouble),
      splitSeed = readOptional(metadata, "evaluation.splitSeed").map(_.toLong),
      retiredAtEpochMillis = readOptional(metadata, "lifecycle.retiredAtEpochMillis").map(_.toLong),
      retirementReason = readOptional(metadata, "lifecycle.retirementReason")
    )

  /** Writes all model metadata to a Java Properties file, including version info,
   * calibration metrics, gate thresholds, sample counts, evaluation strategy, and lifecycle state.
   */
  private def writeMetadata(path: Path, artifact: TrainedPokerActionModel): Unit =
    val props = Properties()
    props.setProperty("format.version", FormatVersion)
    props.setProperty("model.id", artifact.version.id)
    props.setProperty("model.schemaVersion", artifact.version.schemaVersion)
    props.setProperty("model.source", artifact.version.source)
    props.setProperty("model.trainedAtEpochMillis", artifact.version.trainedAtEpochMillis.toString)
    props.setProperty("calibration.meanBrierScore", java.lang.Double.toString(artifact.calibration.meanBrierScore))
    props.setProperty("calibration.sampleCount", artifact.calibration.sampleCount.toString)
    props.setProperty("calibration.uniformBaselineBrier", java.lang.Double.toString(artifact.calibration.uniformBaselineBrier))
    props.setProperty("calibration.majorityBaselineBrier", java.lang.Double.toString(artifact.calibration.majorityBaselineBrier))
    props.setProperty("gate.maxMeanBrierScore", java.lang.Double.toString(artifact.gate.maxMeanBrierScore))
    props.setProperty("training.sampleCount", artifact.trainingSampleCount.toString)
    props.setProperty("evaluation.sampleCount", artifact.evaluationSampleCount.toString)
    props.setProperty("evaluation.strategy", artifact.evaluationStrategy)
    props.setProperty("evaluation.validationFraction", artifact.validationFraction.map(java.lang.Double.toString).getOrElse(""))
    props.setProperty("evaluation.splitSeed", artifact.splitSeed.map(_.toString).getOrElse(""))
    props.setProperty("lifecycle.retiredAtEpochMillis", artifact.retiredAtEpochMillis.map(_.toString).getOrElse(""))
    props.setProperty("lifecycle.retirementReason", artifact.retirementReason.getOrElse(""))
    props.setProperty("logistic.classCount", artifact.model.logistic.weights.length.toString)
    props.setProperty("logistic.featureCount", artifact.model.logistic.weights.headOption.map(_.length).getOrElse(0).toString)

    withWriter(path) { writer =>
      props.store(writer, "PokerActionModel artifact")
    }

  private def readMetadata(path: Path): Properties =
    require(Files.exists(path), s"missing metadata file: $path")
    val props = Properties()
    withReader(path) { reader =>
      props.load(reader)
    }
    props

  /** Writes the weight matrix as a TSV file, one row per class (tab-delimited feature weights). */
  private def writeWeights(path: Path, weights: Vector[Vector[Double]]): Unit =
    withWriter(path) { writer =>
      weights.foreach { row =>
        writer.write(row.map(java.lang.Double.toString).mkString("\t"))
        writer.newLine()
      }
    }

  /** Reads the weight matrix TSV, validating row and column counts match the metadata. */
  private def readWeights(path: Path, expectedRows: Int, expectedCols: Int): Vector[Vector[Double]] =
    require(Files.exists(path), s"missing weights file: $path")
    val lines = Files.readAllLines(path, StandardCharsets.UTF_8).asScala.toVector.filter(_.nonEmpty)
    require(lines.length == expectedRows, s"expected $expectedRows weight rows, got ${lines.length}")
    lines.map { line =>
      val cols = line.split("\t", -1).toVector
      require(cols.length == expectedCols, s"expected $expectedCols weight cols, got ${cols.length}")
      cols.map(_.toDouble)
    }

  /** Writes the bias vector as a single tab-delimited TSV row. */
  private def writeBias(path: Path, bias: Vector[Double]): Unit =
    withWriter(path) { writer =>
      writer.write(bias.map(java.lang.Double.toString).mkString("\t"))
      writer.newLine()
    }

  /** Reads the bias vector from a single-row TSV file, validating the expected element count. */
  private def readBias(path: Path, expectedCount: Int): Vector[Double] =
    require(Files.exists(path), s"missing bias file: $path")
    val lines = Files.readAllLines(path, StandardCharsets.UTF_8).asScala.toVector.filter(_.nonEmpty)
    require(lines.length == 1, s"expected 1 bias row, got ${lines.length}")
    val values = lines.head.split("\t", -1).toVector.map(_.toDouble)
    require(values.length == expectedCount, s"expected $expectedCount bias values, got ${values.length}")
    values

  /** Writes the category-to-index mapping as a two-column TSV (category name, index), sorted by index. */
  private def writeCategoryIndex(path: Path, categoryIndex: Map[PokerAction.Category, Int]): Unit =
    val rows = categoryIndex.toVector.sortBy(_._2)
    withWriter(path) { writer =>
      rows.foreach { case (category, index) =>
        writer.write(s"${category.toString}\t$index")
        writer.newLine()
      }
    }

  /** Reads the category-to-index mapping from a two-column TSV file, validating no duplicates. */
  private def readCategoryIndex(path: Path): Map[PokerAction.Category, Int] =
    require(Files.exists(path), s"missing category index file: $path")
    val rows = Files.readAllLines(path, StandardCharsets.UTF_8).asScala.toVector.filter(_.nonEmpty)
    val pairs = rows.map { row =>
      val parts = row.split("\t", -1)
      require(parts.length == 2, s"invalid category-index row: $row")
      val category = PokerAction.Category.valueOf(parts(0))
      val index = parts(1).toInt
      category -> index
    }
    val map = pairs.toMap
    require(map.size == pairs.length, "duplicate categories in category-index file")
    map

  private def readRequired(props: Properties, key: String): String =
    Option(props.getProperty(key))
      .map(_.trim)
      .filter(_.nonEmpty)
      .getOrElse(throw new IllegalArgumentException(s"missing required metadata key: $key"))

  private def readOptional(props: Properties, key: String): Option[String] =
    Option(props.getProperty(key)).map(_.trim).filter(_.nonEmpty)

  private def withWriter(path: Path)(f: BufferedWriter => Unit): Unit =
    val writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)
    try f(writer)
    finally writer.close()

  private def withReader(path: Path)(f: BufferedReader => Unit): Unit =
    val reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)
    try f(reader)
    finally reader.close()
