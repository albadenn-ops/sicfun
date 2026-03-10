package sicfun.holdem.provider
import sicfun.holdem.io.*
import sicfun.holdem.gpu.*

import java.nio.file.{Files, Path, Paths}

/** Optional ONNX Runtime adapter for DDRE inference.
  *
  * Uses reflection to avoid a hard compile-time dependency on `ai.onnxruntime`.
  * If runtime classes/model are unavailable, callers receive a descriptive error
  * and can degrade safely to Bayesian fallback.
  */
private[holdem] object HoldemDdreOnnxRuntime:
  final case class Config(
      modelPath: String,
      priorInputName: String,
      likelihoodInputName: String,
      outputName: String,
      executionProvider: String,
      cudaDevice: Int,
      intraOpThreads: Option[Int],
      interOpThreads: Option[Int],
      artifactDir: Option[Path],
      artifactId: Option[String],
      validationStatus: String,
      decisionDrivingAllowed: Boolean,
      allowExperimental: Boolean,
      rawModel: Boolean
  ):
    require(modelPath.trim.nonEmpty, "modelPath must be non-empty")
    require(priorInputName.trim.nonEmpty, "priorInputName must be non-empty")
    require(likelihoodInputName.trim.nonEmpty, "likelihoodInputName must be non-empty")
    require(outputName.trim.nonEmpty, "outputName must be non-empty")
    require(Set("cpu", "cuda").contains(executionProvider), "executionProvider must be cpu or cuda")
    require(cudaDevice >= 0, "cudaDevice must be non-negative")
    require(intraOpThreads.forall(_ > 0), "intraOpThreads must be positive when provided")
    require(interOpThreads.forall(_ > 0), "interOpThreads must be positive when provided")
    require(validationStatus.trim.nonEmpty, "validationStatus must be non-empty")

  private val ArtifactDirProperty = "sicfun.ddre.onnx.artifactDir"
  private val ArtifactDirEnv = "sicfun_DDRE_ONNX_ARTIFACT_DIR"
  private val ModelPathProperty = "sicfun.ddre.onnx.modelPath"
  private val ModelPathEnv = "sicfun_DDRE_ONNX_MODEL_PATH"
  private val PriorInputNameProperty = "sicfun.ddre.onnx.input.prior"
  private val PriorInputNameEnv = "sicfun_DDRE_ONNX_INPUT_PRIOR"
  private val LikelihoodInputNameProperty = "sicfun.ddre.onnx.input.likelihoods"
  private val LikelihoodInputNameEnv = "sicfun_DDRE_ONNX_INPUT_LIKELIHOODS"
  private val OutputNameProperty = "sicfun.ddre.onnx.output.posterior"
  private val OutputNameEnv = "sicfun_DDRE_ONNX_OUTPUT_POSTERIOR"
  private val ExecutionProviderProperty = "sicfun.ddre.onnx.executionProvider"
  private val ExecutionProviderEnv = "sicfun_DDRE_ONNX_EXECUTION_PROVIDER"
  private val CudaDeviceProperty = "sicfun.ddre.onnx.cuda.device"
  private val CudaDeviceEnv = "sicfun_DDRE_ONNX_CUDA_DEVICE"
  private val IntraOpThreadsProperty = "sicfun.ddre.onnx.intraOpThreads"
  private val IntraOpThreadsEnv = "sicfun_DDRE_ONNX_INTRA_OP_THREADS"
  private val InterOpThreadsProperty = "sicfun.ddre.onnx.interOpThreads"
  private val InterOpThreadsEnv = "sicfun_DDRE_ONNX_INTER_OP_THREADS"
  private val AllowExperimentalProperty = "sicfun.ddre.onnx.allowExperimental"
  private val AllowExperimentalEnv = "sicfun_DDRE_ONNX_ALLOW_EXPERIMENTAL"

  private val DefaultPriorInputName = "prior"
  private val DefaultLikelihoodInputName = "likelihoods"
  private val DefaultOutputName = "posterior"

  def configuredConfig(): Either[String, Config] =
    val allowExperimental = GpuRuntimeSupport
      .resolveNonEmpty(AllowExperimentalProperty, AllowExperimentalEnv)
      .exists(GpuRuntimeSupport.parseTruthy)

    GpuRuntimeSupport.resolveNonEmpty(ArtifactDirProperty, ArtifactDirEnv) match
      case Some(rawDirectory) =>
        val directory = Paths.get(rawDirectory).toAbsolutePath.normalize()
        HoldemDdreArtifactIO
          .load(directory)
          .map(artifact => configFromArtifact(directory, artifact, allowExperimental))
      case None =>
        val modelPathOpt = GpuRuntimeSupport.resolveNonEmpty(ModelPathProperty, ModelPathEnv)
        modelPathOpt match
          case None =>
            Left(
              s"ddre onnx artifact/model not configured ($ArtifactDirProperty or $ArtifactDirEnv or $ModelPathProperty or $ModelPathEnv)"
            )
          case Some(modelPath) =>
            val priorInputName = GpuRuntimeSupport
              .resolveNonEmpty(PriorInputNameProperty, PriorInputNameEnv)
              .getOrElse(DefaultPriorInputName)
              .trim
            val likelihoodInputName = GpuRuntimeSupport
              .resolveNonEmpty(LikelihoodInputNameProperty, LikelihoodInputNameEnv)
              .getOrElse(DefaultLikelihoodInputName)
              .trim
            val outputName = GpuRuntimeSupport
              .resolveNonEmpty(OutputNameProperty, OutputNameEnv)
              .getOrElse(DefaultOutputName)
              .trim
            val executionProvider = GpuRuntimeSupport
              .resolveNonEmptyLower(ExecutionProviderProperty, ExecutionProviderEnv)
              .getOrElse("cpu")
              .trim
            val cudaDevice = GpuRuntimeSupport
              .resolveNonEmpty(CudaDeviceProperty, CudaDeviceEnv)
              .flatMap(_.toIntOption)
              .getOrElse(0)
            val intraOpThreads = GpuRuntimeSupport
              .resolveNonEmpty(IntraOpThreadsProperty, IntraOpThreadsEnv)
              .flatMap(_.toIntOption)
            val interOpThreads = GpuRuntimeSupport
              .resolveNonEmpty(InterOpThreadsProperty, InterOpThreadsEnv)
              .flatMap(_.toIntOption)

            if !Set("cpu", "cuda").contains(executionProvider) then
              Left(s"invalid ddre onnx executionProvider '$executionProvider'; expected cpu|cuda")
            else
              Right(
                Config(
                  modelPath = modelPath,
                  priorInputName = priorInputName,
                  likelihoodInputName = likelihoodInputName,
                  outputName = outputName,
                  executionProvider = executionProvider,
                  cudaDevice = math.max(0, cudaDevice),
                  intraOpThreads = intraOpThreads.filter(_ > 0),
                  interOpThreads = interOpThreads.filter(_ > 0),
                  artifactDir = None,
                  artifactId = None,
                  validationStatus = "raw",
                  decisionDrivingAllowed = false,
                  allowExperimental = allowExperimental,
                  rawModel = true
                )
              )

  private[holdem] def configFromArtifact(
      directory: Path,
      artifact: HoldemDdreArtifactIO.OnnxArtifact,
      allowExperimental: Boolean
  ): Config =
    val modelPath = Paths.get(artifact.modelFile)
    val resolvedModel =
      if modelPath.isAbsolute then modelPath
      else directory.resolve(modelPath).normalize()

    Config(
      modelPath = resolvedModel.toString,
      priorInputName = artifact.priorInputName,
      likelihoodInputName = artifact.likelihoodInputName,
      outputName = artifact.outputName,
      executionProvider = artifact.executionProvider,
      cudaDevice = artifact.cudaDevice,
      intraOpThreads = artifact.intraOpThreads,
      interOpThreads = artifact.interOpThreads,
      artifactDir = Some(directory),
      artifactId = Some(artifact.artifactId),
      validationStatus = artifact.validationStatus,
      decisionDrivingAllowed = artifact.decisionDrivingAllowed,
      allowExperimental = allowExperimental,
      rawModel = false
    )

  def inferPosterior(
      prior: Array[Double],
      likelihoods: Array[Double],
      observationCount: Int,
      hypothesisCount: Int,
      config: Config
  ): Either[String, Array[Double]] =
    if observationCount < 0 then Left(s"ddre onnx observationCount must be >= 0, found $observationCount")
    else if hypothesisCount <= 0 then Left(s"ddre onnx hypothesisCount must be > 0, found $hypothesisCount")
    else if prior.length != hypothesisCount then
      Left(s"ddre onnx prior length mismatch: expected $hypothesisCount, found ${prior.length}")
    else if likelihoods.length != observationCount * hypothesisCount then
      Left(
        s"ddre onnx likelihood matrix length mismatch: expected ${observationCount * hypothesisCount}, found ${likelihoods.length}"
      )
    else
      val model = Paths.get(config.modelPath)
      if !Files.isRegularFile(model) then
        Left(s"ddre onnx model file not found: ${model.toAbsolutePath.normalize()}")
      else
        runOnnx(prior, likelihoods, observationCount, hypothesisCount, config)

  private def runOnnx(
      prior: Array[Double],
      likelihoods: Array[Double],
      observationCount: Int,
      hypothesisCount: Int,
      config: Config
  ): Either[String, Array[Double]] =
    try
      val ortEnvironmentClass = Class.forName("ai.onnxruntime.OrtEnvironment")
      val ortSessionClass = Class.forName("ai.onnxruntime.OrtSession")
      val sessionOptionsClass = Class.forName("ai.onnxruntime.OrtSession$SessionOptions")
      val onnxTensorClass = Class.forName("ai.onnxruntime.OnnxTensor")

      val environment = ortEnvironmentClass.getMethod("getEnvironment").invoke(null).asInstanceOf[AnyRef]
      val sessionOptions = sessionOptionsClass.getConstructor().newInstance().asInstanceOf[AnyRef]
      configureSessionOptions(sessionOptionsClass, sessionOptions, config)

      val session = ortEnvironmentClass
        .getMethod("createSession", classOf[String], sessionOptionsClass)
        .invoke(environment, config.modelPath, sessionOptions)
        .asInstanceOf[AnyRef]

      val priorInput = Array(prior.map(_.toFloat))
      val likelihoodInput =
        if observationCount > 0 then
          Array.tabulate(observationCount) { row =>
            val start = row * hypothesisCount
            val endExclusive = start + hypothesisCount
            likelihoods.slice(start, endExclusive).map(_.toFloat)
          }
        else
          Array(Array.fill(hypothesisCount)(1.0f))

      val createTensor = onnxTensorClass.getMethod("createTensor", ortEnvironmentClass, classOf[Object])
      val priorTensor = createTensor
        .invoke(null, environment, priorInput.asInstanceOf[Object])
        .asInstanceOf[AnyRef]
      val likelihoodTensor = createTensor
        .invoke(null, environment, likelihoodInput.asInstanceOf[Object])
        .asInstanceOf[AnyRef]

      val inputs = new java.util.HashMap[String, AnyRef]()
      inputs.put(config.priorInputName, priorTensor)
      inputs.put(config.likelihoodInputName, likelihoodTensor)

      val result = ortSessionClass
        .getMethod("run", classOf[java.util.Map[?, ?]])
        .invoke(session, inputs)
        .asInstanceOf[AnyRef]

      val outputEither = extractPosteriorFromResult(result, config.outputName, hypothesisCount)
      closeQuietly(priorTensor)
      closeQuietly(likelihoodTensor)
      closeQuietly(result)
      closeQuietly(session)
      closeQuietly(sessionOptions)
      outputEither
    catch
      case _: ClassNotFoundException =>
        Left("ai.onnxruntime classes not found on classpath; add ONNX Runtime dependency")
      case ex: Throwable =>
        Left(
          Option(ex.getMessage)
            .map(_.trim)
            .filter(_.nonEmpty)
            .getOrElse(ex.getClass.getSimpleName)
        )

  private def configureSessionOptions(
      sessionOptionsClass: Class[?],
      sessionOptions: AnyRef,
      config: Config
  ): Unit =
    config.intraOpThreads.foreach { threads =>
      try
        sessionOptionsClass
          .getMethod("setIntraOpNumThreads", classOf[Int])
          .invoke(sessionOptions, Int.box(threads))
      catch
        case _: Throwable => ()
    }
    config.interOpThreads.foreach { threads =>
      try
        sessionOptionsClass
          .getMethod("setInterOpNumThreads", classOf[Int])
          .invoke(sessionOptions, Int.box(threads))
      catch
        case _: Throwable => ()
    }
    if config.executionProvider == "cuda" then
      try
        sessionOptionsClass
          .getMethod("addCUDA", classOf[Int])
          .invoke(sessionOptions, Int.box(config.cudaDevice))
      catch
        case _: Throwable =>
          try
            sessionOptionsClass.getMethod("addCUDA").invoke(sessionOptions)
          catch
            case _: Throwable => ()

  private def extractPosteriorFromResult(
      result: AnyRef,
      outputName: String,
      expectedSize: Int
  ): Either[String, Array[Double]] =
    val resultClass = result.getClass

    val outputValueOpt =
      resultClass.getMethods.find(m =>
        m.getName == "get" &&
          m.getParameterCount == 1 &&
          m.getParameterTypes.head == classOf[String]
      ) match
        case Some(getByName) =>
          val named = getByName.invoke(result, outputName)
          named match
            case optional: java.util.Optional[?] =>
              if optional.isPresent then Some(optional.get().asInstanceOf[AnyRef]) else None
            case any if any != null => Some(any.asInstanceOf[AnyRef])
            case _ => None
        case None =>
          resultClass.getMethods.find(m =>
            m.getName == "get" &&
              m.getParameterCount == 1 &&
              m.getParameterTypes.head == classOf[Int]
          ) match
            case Some(getByIndex) =>
              Option(getByIndex.invoke(result, Int.box(0))).map(_.asInstanceOf[AnyRef])
            case None => None

    outputValueOpt match
      case None =>
        Left(s"ddre onnx output '$outputName' not found")
      case Some(outputValue) =>
        val value = outputValue.getClass.getMethod("getValue").invoke(outputValue)
        flattenNumericOutput(value, expectedSize)

  private def flattenNumericOutput(value: Any, expectedSize: Int): Either[String, Array[Double]] =
    val flattened =
      value match
        case array: Array[Float] =>
          array.map(_.toDouble)
        case array: Array[Double] =>
          array
        case matrix: Array[Array[Float]] =>
          matrix.flatten.map(_.toDouble)
        case matrix: Array[Array[Double]] =>
          matrix.flatten
        case list: java.util.List[?] =>
          list.toArray.toVector.flatMap {
            case f: java.lang.Float => Vector(f.doubleValue())
            case d: java.lang.Double => Vector(d.doubleValue())
            case arr: Array[Float] => arr.toVector.map(_.toDouble)
            case arr: Array[Double] => arr.toVector
            case _ => Vector.empty
          }.toArray
        case other =>
          return Left(s"ddre onnx output has unsupported value type: ${other.getClass.getName}")

    if flattened.length != expectedSize then
      Left(
        s"ddre onnx posterior length mismatch: expected $expectedSize, found ${flattened.length}"
      )
    else
      Right(flattened)

  private def closeQuietly(resource: AnyRef): Unit =
    if resource != null then
      try
        resource.getClass.getMethod("close").invoke(resource)
      catch
        case _: Throwable => ()
