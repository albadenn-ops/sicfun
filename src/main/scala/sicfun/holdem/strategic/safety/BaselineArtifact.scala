package sicfun.holdem.strategic.safety

import java.nio.file.{Files, Path}
import java.util.Properties
import scala.jdk.CollectionConverters.*
import sicfun.holdem.types.{PokerAction, Street}
import sicfun.holdem.strategic.types.StrategicClass

/** Metadata describing how a baseline count artifact was produced. */
final case class BaselineMetadata(
    formatVersion: String,
    bucketSchemeVersion: String,
    corpusId: String,
    handCount: Long,
    showdownDecisionCount: Long,
    recommendedMinCount: Int,
    recommendedSmoothingAlpha: Double,
    calibrationEpochMillis: Long
)

/** Raw observed action counts keyed by (class, board-bucket token, street, action category),
  * plus provenance metadata. Smoothing and backoff are applied at load by RealBaselineImpl —
  * the artifact stores only counts so policy is tunable without recalibration.
  */
final case class BaselineArtifact(
    counts: Map[(StrategicClass, String, Street, PokerAction.Category), Long],
    metadata: BaselineMetadata
)

/** Persists a [[BaselineArtifact]] as a directory of `metadata.properties` + `baseline-counts.tsv`.
  * Mirrors the flat-file, diffable convention of `PokerActionModelArtifactIO`.
  */
object BaselineArtifactIO:
  val FormatVersion: String = "1"
  private val MetadataFile = "metadata.properties"
  private val CountsFile = "baseline-counts.tsv"
  private val Header = "class\tboardBucket\tstreet\taction\tcount"

  def save(directory: Path, artifact: BaselineArtifact): Unit =
    Files.createDirectories(directory)
    writeMetadata(directory.resolve(MetadataFile), artifact.metadata)
    writeCounts(directory.resolve(CountsFile), artifact.counts)

  def load(directory: Path): BaselineArtifact =
    require(Files.isDirectory(directory), s"baseline artifact directory does not exist: $directory")
    val meta = readMetadata(directory.resolve(MetadataFile))
    require(meta.formatVersion == FormatVersion,
      s"unsupported baseline artifact format version: ${meta.formatVersion} (expected $FormatVersion)")
    val counts = readCounts(directory.resolve(CountsFile))
    BaselineArtifact(counts, meta)

  private def writeMetadata(path: Path, m: BaselineMetadata): Unit =
    val props = new Properties()
    props.setProperty("format.version", m.formatVersion)
    props.setProperty("bucketScheme.version", m.bucketSchemeVersion)
    props.setProperty("calibration.corpusId", m.corpusId)
    props.setProperty("calibration.handCount", m.handCount.toString)
    props.setProperty("calibration.showdownDecisionCount", m.showdownDecisionCount.toString)
    props.setProperty("calibration.recommendedMinCount", m.recommendedMinCount.toString)
    props.setProperty("calibration.recommendedSmoothingAlpha", java.lang.Double.toString(m.recommendedSmoothingAlpha))
    props.setProperty("calibration.epochMillis", m.calibrationEpochMillis.toString)
    val writer = Files.newBufferedWriter(path)
    try props.store(writer, "BaselineArtifact metadata") finally writer.close()

  private def readMetadata(path: Path): BaselineMetadata =
    require(Files.isRegularFile(path), s"missing $MetadataFile in artifact")
    val props = new Properties()
    val reader = Files.newBufferedReader(path)
    try props.load(reader) finally reader.close()
    def req(k: String): String =
      val v = props.getProperty(k)
      require(v != null, s"missing metadata key: $k"); v
    BaselineMetadata(
      formatVersion = req("format.version"),
      bucketSchemeVersion = props.getProperty("bucketScheme.version", "v1"),
      corpusId = props.getProperty("calibration.corpusId", ""),
      handCount = props.getProperty("calibration.handCount", "0").toLong,
      showdownDecisionCount = props.getProperty("calibration.showdownDecisionCount", "0").toLong,
      recommendedMinCount = props.getProperty("calibration.recommendedMinCount", "30").toInt,
      recommendedSmoothingAlpha = props.getProperty("calibration.recommendedSmoothingAlpha", "1.0").toDouble,
      calibrationEpochMillis = props.getProperty("calibration.epochMillis", "0").toLong
    )

  private def writeCounts(path: Path, counts: Map[(StrategicClass, String, Street, PokerAction.Category), Long]): Unit =
    val sb = new StringBuilder().append(Header).append('\n')
    counts.toVector.sortBy { case ((c, b, s, a), _) => (c.toString, b, s.toString, a.toString) }
      .foreach { case ((c, b, s, a), n) =>
        sb.append(c.toString).append('\t').append(b).append('\t')
          .append(s.toString).append('\t').append(a.toString).append('\t').append(n.toString).append('\n')
      }
    Files.writeString(path, sb.toString)

  private def readCounts(path: Path): Map[(StrategicClass, String, Street, PokerAction.Category), Long] =
    require(Files.isRegularFile(path), s"missing $CountsFile in artifact")
    val lines = Files.readAllLines(path).asScala.toVector
    lines.drop(1).filter(_.trim.nonEmpty).map { line =>
      val cols = line.split("\t", -1)
      require(cols.length == 5, s"malformed counts row: $line")
      val key = (
        StrategicClass.valueOf(cols(0)),
        cols(1),
        Street.valueOf(cols(2)),
        PokerAction.Category.valueOf(cols(3))
      )
      key -> cols(4).toLong
    }.toMap
