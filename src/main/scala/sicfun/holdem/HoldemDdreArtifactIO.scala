package sicfun.holdem

import java.io.{BufferedReader, BufferedWriter}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import java.util.Properties

/** Reads and writes DDRE ONNX artifact manifests.
  *
  * The manifest separates "an ONNX file exists" from "this model is cleared to
  * drive decisions". Runtime code can then reject smoke or unvalidated artifacts
  * by default while tools still evaluate them explicitly.
  */
object HoldemDdreArtifactIO:
  final case class OnnxArtifact(
      artifactId: String,
      source: String,
      createdAtEpochMillis: Long,
      modelFile: String,
      priorInputName: String,
      likelihoodInputName: String,
      outputName: String,
      executionProvider: String,
      cudaDevice: Int,
      intraOpThreads: Option[Int],
      interOpThreads: Option[Int],
      validationStatus: String,
      decisionDrivingAllowed: Boolean,
      validationSampleCount: Option[Int],
      meanNll: Option[Double],
      meanKlVsBayes: Option[Double],
      blockerViolationRate: Option[Double],
      failureRate: Option[Double],
      p50LatencyMillis: Option[Double],
      p95LatencyMillis: Option[Double],
      gateMinSamples: Option[Int],
      gateMaxMeanNll: Option[Double],
      gateMaxMeanKlVsBayes: Option[Double],
      gateMaxBlockerViolationRate: Option[Double],
      gateMaxFailureRate: Option[Double],
      gateMaxP95LatencyMillis: Option[Double],
      notes: Option[String]
  ):
    require(artifactId.trim.nonEmpty, "artifactId must be non-empty")
    require(source.trim.nonEmpty, "source must be non-empty")
    require(createdAtEpochMillis >= 0L, "createdAtEpochMillis must be non-negative")
    require(modelFile.trim.nonEmpty, "modelFile must be non-empty")
    require(priorInputName.trim.nonEmpty, "priorInputName must be non-empty")
    require(likelihoodInputName.trim.nonEmpty, "likelihoodInputName must be non-empty")
    require(outputName.trim.nonEmpty, "outputName must be non-empty")
    require(Set("cpu", "cuda").contains(executionProvider), "executionProvider must be cpu or cuda")
    require(cudaDevice >= 0, "cudaDevice must be non-negative")
    require(intraOpThreads.forall(_ > 0), "intraOpThreads must be positive when provided")
    require(interOpThreads.forall(_ > 0), "interOpThreads must be positive when provided")
    require(validationStatus.trim.nonEmpty, "validationStatus must be non-empty")

  private val FormatVersion = "1"
  private val MetadataFileName = "metadata.properties"

  def load(directory: Path): Either[String, OnnxArtifact] =
    try
      require(Files.isDirectory(directory), s"DDRE artifact directory does not exist: $directory")
      val props = readMetadata(directory.resolve(MetadataFileName))
      val format = readRequired(props, "format.version")
      require(format == FormatVersion, s"unsupported DDRE artifact format version: $format")
      val kind = readRequired(props, "artifact.kind")
      require(kind == "onnx", s"unsupported DDRE artifact kind: $kind")

      Right(
        OnnxArtifact(
          artifactId = readRequired(props, "artifact.id"),
          source = readRequired(props, "artifact.source"),
          createdAtEpochMillis = readRequired(props, "artifact.createdAtEpochMillis").toLong,
          modelFile = readRequired(props, "model.file"),
          priorInputName = readRequired(props, "model.input.prior"),
          likelihoodInputName = readRequired(props, "model.input.likelihoods"),
          outputName = readRequired(props, "model.output.posterior"),
          executionProvider = readRequired(props, "runtime.executionProvider"),
          cudaDevice = readOptional(props, "runtime.cudaDevice").flatMap(_.toIntOption).getOrElse(0),
          intraOpThreads = readOptional(props, "runtime.intraOpThreads").flatMap(_.toIntOption),
          interOpThreads = readOptional(props, "runtime.interOpThreads").flatMap(_.toIntOption),
          validationStatus = readOptional(props, "validation.status").getOrElse("experimental"),
          decisionDrivingAllowed =
            readOptional(props, "validation.decisionDrivingAllowed").exists(GpuRuntimeSupport.parseTruthy),
          validationSampleCount = readOptional(props, "validation.sampleCount").flatMap(_.toIntOption),
          meanNll = readOptional(props, "validation.meanNll").flatMap(_.toDoubleOption),
          meanKlVsBayes = readOptional(props, "validation.meanKlVsBayes").flatMap(_.toDoubleOption),
          blockerViolationRate =
            readOptional(props, "validation.blockerViolationRate").flatMap(_.toDoubleOption),
          failureRate = readOptional(props, "validation.failureRate").flatMap(_.toDoubleOption),
          p50LatencyMillis = readOptional(props, "validation.p50LatencyMillis").flatMap(_.toDoubleOption),
          p95LatencyMillis = readOptional(props, "validation.p95LatencyMillis").flatMap(_.toDoubleOption),
          gateMinSamples = readOptional(props, "gate.minSamples").flatMap(_.toIntOption),
          gateMaxMeanNll = readOptional(props, "gate.maxMeanNll").flatMap(_.toDoubleOption),
          gateMaxMeanKlVsBayes = readOptional(props, "gate.maxMeanKlVsBayes").flatMap(_.toDoubleOption),
          gateMaxBlockerViolationRate =
            readOptional(props, "gate.maxBlockerViolationRate").flatMap(_.toDoubleOption),
          gateMaxFailureRate = readOptional(props, "gate.maxFailureRate").flatMap(_.toDoubleOption),
          gateMaxP95LatencyMillis = readOptional(props, "gate.maxP95LatencyMillis").flatMap(_.toDoubleOption),
          notes = readOptional(props, "artifact.notes")
        )
      )
    catch
      case ex: Exception =>
        Left(Option(ex.getMessage).getOrElse(ex.getClass.getSimpleName))

  def save(directory: Path, artifact: OnnxArtifact): Unit =
    Files.createDirectories(directory)
    val props = Properties()
    props.setProperty("format.version", FormatVersion)
    props.setProperty("artifact.kind", "onnx")
    props.setProperty("artifact.id", artifact.artifactId)
    props.setProperty("artifact.source", artifact.source)
    props.setProperty("artifact.createdAtEpochMillis", artifact.createdAtEpochMillis.toString)
    props.setProperty("artifact.notes", artifact.notes.getOrElse(""))
    props.setProperty("model.file", artifact.modelFile)
    props.setProperty("model.input.prior", artifact.priorInputName)
    props.setProperty("model.input.likelihoods", artifact.likelihoodInputName)
    props.setProperty("model.output.posterior", artifact.outputName)
    props.setProperty("runtime.executionProvider", artifact.executionProvider)
    props.setProperty("runtime.cudaDevice", artifact.cudaDevice.toString)
    props.setProperty("runtime.intraOpThreads", artifact.intraOpThreads.map(_.toString).getOrElse(""))
    props.setProperty("runtime.interOpThreads", artifact.interOpThreads.map(_.toString).getOrElse(""))
    props.setProperty("validation.status", artifact.validationStatus)
    props.setProperty("validation.decisionDrivingAllowed", artifact.decisionDrivingAllowed.toString)
    props.setProperty("validation.sampleCount", artifact.validationSampleCount.map(_.toString).getOrElse(""))
    props.setProperty("validation.meanNll", artifact.meanNll.map(java.lang.Double.toString).getOrElse(""))
    props.setProperty(
      "validation.meanKlVsBayes",
      artifact.meanKlVsBayes.map(java.lang.Double.toString).getOrElse("")
    )
    props.setProperty(
      "validation.blockerViolationRate",
      artifact.blockerViolationRate.map(java.lang.Double.toString).getOrElse("")
    )
    props.setProperty("validation.failureRate", artifact.failureRate.map(java.lang.Double.toString).getOrElse(""))
    props.setProperty(
      "validation.p50LatencyMillis",
      artifact.p50LatencyMillis.map(java.lang.Double.toString).getOrElse("")
    )
    props.setProperty(
      "validation.p95LatencyMillis",
      artifact.p95LatencyMillis.map(java.lang.Double.toString).getOrElse("")
    )
    props.setProperty("gate.minSamples", artifact.gateMinSamples.map(_.toString).getOrElse(""))
    props.setProperty("gate.maxMeanNll", artifact.gateMaxMeanNll.map(java.lang.Double.toString).getOrElse(""))
    props.setProperty(
      "gate.maxMeanKlVsBayes",
      artifact.gateMaxMeanKlVsBayes.map(java.lang.Double.toString).getOrElse("")
    )
    props.setProperty(
      "gate.maxBlockerViolationRate",
      artifact.gateMaxBlockerViolationRate.map(java.lang.Double.toString).getOrElse("")
    )
    props.setProperty(
      "gate.maxFailureRate",
      artifact.gateMaxFailureRate.map(java.lang.Double.toString).getOrElse("")
    )
    props.setProperty(
      "gate.maxP95LatencyMillis",
      artifact.gateMaxP95LatencyMillis.map(java.lang.Double.toString).getOrElse("")
    )

    withWriter(directory.resolve(MetadataFileName)) { writer =>
      props.store(writer, "Holdem DDRE ONNX artifact")
    }

  private def readMetadata(path: Path): Properties =
    require(Files.isRegularFile(path), s"missing DDRE artifact metadata file: $path")
    val props = Properties()
    withReader(path) { reader =>
      props.load(reader)
    }
    props

  private def readRequired(props: Properties, key: String): String =
    readOptional(props, key).getOrElse(
      throw new IllegalArgumentException(s"missing required DDRE artifact metadata key: $key")
    )

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
