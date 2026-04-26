package sicfun.holdem.provider
import sicfun.holdem.io.*
import sicfun.holdem.gpu.*

import ai.onnxruntime.{OnnxTensor, OnnxValue, OrtEnvironment, OrtException, OrtSession}
import ai.onnxruntime.OrtSession.{Result, SessionOptions}

import java.nio.file.{Files, Path, Paths}
import scala.util.control.NonFatal

/** Optional ONNX Runtime adapter for DDRE posterior inference.
  *
  * This object provides an inference path for DDRE models exported as ONNX files,
  * enabling the poker engine to use trained neural-network posteriors as an
  * alternative or complement to pure Bayesian inference.
  *
  * ==Reflection-Based Design==
  * The ONNX Runtime Java API (`ai.onnxruntime`) is accessed entirely via reflection
  * to avoid a hard compile-time dependency. This means:
  *  - The project compiles and runs without ONNX Runtime on the classpath.
  *  - When the runtime is present, it's loaded on first use.
  *  - If missing, callers get a descriptive `Left` and can fall back to Bayesian.
  *
  * ==Artifact Resolution==
  * Two configuration modes:
  *  1. '''Artifact directory''' (`sicfun.ddre.onnx.artifactDir`) -- loads metadata
  *     from [[HoldemDdreArtifactIO]], including validation status, I/O tensor names,
  *     and execution provider settings.
  *  2. '''Direct model path''' (`sicfun.ddre.onnx.modelPath`) -- raw ONNX file path
  *     with individual tensor name overrides. Treated as unvalidated.
  *
  * ==CUDA Execution Provider==
  * When `executionProvider=cuda`, the adapter calls `SessionOptions.addCUDA(deviceIndex)`
  * to run inference on GPU. Falls back gracefully if CUDA is not available.
  *
  * ==I/O Contract==
  * - Prior input: `float32[1, hypothesisCount]` -- normalised prior probabilities
  * - Likelihood input: `float32[observationCount, hypothesisCount]` -- per-observation likelihoods
  * - Posterior output: `float32[1, hypothesisCount]` -- unnormalised posterior (caller normalises)
  *
  * All internal computation uses float32 to match ONNX model precision; the JVM
  * interface accepts and returns double arrays, with conversion at the boundary.
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

  /** Builds an ONNX runtime configuration from system properties / environment variables.
    * Tries artifact-dir-based resolution first, then falls back to direct model path.
    *
    * @return `Right(config)` if a valid configuration was resolved, `Left(reason)` otherwise
    */
  /** Reflection-bridge self-test report: status of each Class.forName / method lookup
    * the ONNX path depends on. Use [[selfTest]] to populate.
    */
  final case class SelfTestReport(checks: Vector[(String, Either[String, Unit])]):
    /** True iff every check resolved successfully. */
    def allOk: Boolean = checks.forall(_._2.isRight)
    /** Human-readable summary, one line per check. */
    def summary: String =
      checks.map { case (name, result) =>
        result match
          case Right(_)     => s"  OK  $name"
          case Left(reason) => s"  FAIL $name: $reason"
      }.mkString("\n")

  /** Validate that every ai.onnxruntime class and method the [[runOnnx]] reflection
    * chain depends on is reachable on the current classpath.
    *
    * Intended as an opt-in pre-flight check (not auto-run at startup) so the synthetic
    * DDRE path can keep running on machines without ONNX. Catches API drift after an
    * onnxruntime upgrade -- the compiler cannot, because the bridge is reflection-only.
    *
    * Returns Right(report) when every lookup resolved; Left(reason) only on a critical
    * failure (e.g. SecurityManager blocking reflective access). The report itself is
    * always populated, so callers can inspect which specific lookups failed.
    */
  def selfTest(): SelfTestReport =
    def lookup(name: String, body: => Unit): (String, Either[String, Unit]) =
      try
        body
        (name, Right(()))
      catch
        case ex: ClassNotFoundException =>
          (name, Left(s"class not found: ${ex.getMessage}"))
        case ex: NoSuchMethodException =>
          (name, Left(s"method not found: ${ex.getMessage}"))
        case ex: Throwable =>
          (name, Left(Option(ex.getMessage).getOrElse(ex.getClass.getSimpleName)))

    val checks = Vector(
      lookup("Class ai.onnxruntime.OrtEnvironment", {
        Class.forName("ai.onnxruntime.OrtEnvironment")
        ()
      }),
      lookup("Class ai.onnxruntime.OrtSession", {
        Class.forName("ai.onnxruntime.OrtSession")
        ()
      }),
      lookup("Class ai.onnxruntime.OrtSession$SessionOptions", {
        Class.forName("ai.onnxruntime.OrtSession$SessionOptions")
        ()
      }),
      lookup("Class ai.onnxruntime.OnnxTensor", {
        Class.forName("ai.onnxruntime.OnnxTensor")
        ()
      }),
      lookup("OrtEnvironment.getEnvironment()", {
        Class.forName("ai.onnxruntime.OrtEnvironment").getMethod("getEnvironment")
        ()
      }),
      lookup("SessionOptions()", {
        Class.forName("ai.onnxruntime.OrtSession$SessionOptions").getConstructor()
        ()
      }),
      lookup("OrtEnvironment.createSession(String, SessionOptions)", {
        val env = Class.forName("ai.onnxruntime.OrtEnvironment")
        val opts = Class.forName("ai.onnxruntime.OrtSession$SessionOptions")
        env.getMethod("createSession", classOf[String], opts)
        ()
      }),
      lookup("OnnxTensor.createTensor(OrtEnvironment, Object)", {
        val env = Class.forName("ai.onnxruntime.OrtEnvironment")
        val tensor = Class.forName("ai.onnxruntime.OnnxTensor")
        tensor.getMethod("createTensor", env, classOf[Object])
        ()
      }),
      lookup("OrtSession.run(Map)", {
        Class.forName("ai.onnxruntime.OrtSession")
          .getMethod("run", classOf[java.util.Map[?, ?]])
        ()
      })
    )
    SelfTestReport(checks)

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

  /** Builds an ONNX Config from a loaded artifact descriptor.
    * Resolves relative model paths against the artifact directory.
    */
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

  /** Runs ONNX inference to produce a posterior distribution over hypotheses.
    *
    * @param prior            prior probabilities (length = hypothesisCount)
    * @param likelihoods      row-major likelihood matrix (length = observationCount * hypothesisCount)
    * @param observationCount number of observations (rows)
    * @param hypothesisCount  number of hypotheses (columns)
    * @param config           ONNX runtime configuration (model path, tensor names, etc.)
    * @return `Right(posterior)` as a double array, or `Left(reason)` on failure
    */
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

  /** Core ONNX inference implementation using the typed `ai.onnxruntime` API.
    *
    * Steps:
    *  1. Acquire the singleton OrtEnvironment.
    *  2. Build SessionOptions (intra/inter-op threads, CUDA EP if requested).
    *  3. Open an OrtSession from the configured model path.
    *  4. Convert prior (double[]) -> float[1][hypothesisCount] and
    *     likelihoods (double[obs*hyp]) -> float[obs][hyp].
    *  5. Build named OnnxTensor inputs.
    *  6. Run the session, extract the posterior output, flatten to double[].
    *  7. Close every ONNX resource (tensors, result, session, options).
    *
    * When observationCount=0, a dummy row of all-ones likelihoods is used
    * (the model is expected to handle this as a no-op observation).
    *
    * Errors are surfaced as `Left` strings: typed `OrtException` carries the
    * native error message when available; other non-fatal failures fall back
    * to the exception's class name. Fatal errors (OOM, StackOverflow, etc.)
    * propagate via `NonFatal` rather than being swallowed.
    */
  private def runOnnx(
      prior: Array[Double],
      likelihoods: Array[Double],
      observationCount: Int,
      hypothesisCount: Int,
      config: Config
  ): Either[String, Array[Double]] =
    var session: OrtSession = null
    var sessionOptions: SessionOptions = null
    var priorTensor: OnnxTensor = null
    var likelihoodTensor: OnnxTensor = null
    var result: Result = null
    try
      val environment = OrtEnvironment.getEnvironment()
      sessionOptions = new SessionOptions()
      configureSessionOptions(sessionOptions, config)

      session = environment.createSession(config.modelPath, sessionOptions)

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

      priorTensor = OnnxTensor.createTensor(environment, priorInput)
      likelihoodTensor = OnnxTensor.createTensor(environment, likelihoodInput)

      val inputs = new java.util.HashMap[String, OnnxTensor]()
      inputs.put(config.priorInputName, priorTensor)
      inputs.put(config.likelihoodInputName, likelihoodTensor)

      result = session.run(inputs)
      extractPosteriorFromResult(result, config.outputName, hypothesisCount)
    catch
      case ex: OrtException =>
        Left(
          Option(ex.getMessage)
            .map(_.trim)
            .filter(_.nonEmpty)
            .map(m => s"ONNX runtime error: $m")
            .getOrElse(s"ONNX runtime error (${ex.getClass.getSimpleName})")
        )
      case NonFatal(ex) =>
        Left(
          Option(ex.getMessage)
            .map(_.trim)
            .filter(_.nonEmpty)
            .getOrElse(ex.getClass.getSimpleName)
        )
    finally
      closeQuietly(priorTensor)
      closeQuietly(likelihoodTensor)
      closeQuietly(result)
      closeQuietly(session)
      closeQuietly(sessionOptions)

  /** Configures ONNX session options: thread counts and CUDA execution provider.
    * Each setter is wrapped in a typed try/catch so an OrtException from one
    * optional configuration call doesn't abort the others. CUDA registration
    * tries the `addCUDA(int)` variant first, falls back to `addCUDA()` for
    * older runtimes.
    */
  private def configureSessionOptions(
      sessionOptions: SessionOptions,
      config: Config
  ): Unit =
    config.intraOpThreads.foreach { threads =>
      try sessionOptions.setIntraOpNumThreads(threads)
      catch case _: OrtException => ()
    }
    config.interOpThreads.foreach { threads =>
      try sessionOptions.setInterOpNumThreads(threads)
      catch case _: OrtException => ()
    }
    if config.executionProvider == "cuda" then
      try sessionOptions.addCUDA(config.cudaDevice)
      catch
        case _: OrtException =>
          try sessionOptions.addCUDA()
          catch case _: OrtException => ()

  /** Extracts the posterior array from the ONNX Result object.
    * Looks the named output up via `Result.get(String): Optional[OnnxValue]`.
    * Supports float[], double[], float[][], and double[][] output shapes via
    * [[flattenNumericOutput]].
    */
  private def extractPosteriorFromResult(
      result: Result,
      outputName: String,
      expectedSize: Int
  ): Either[String, Array[Double]] =
    val opt = result.get(outputName)
    if !opt.isPresent then
      Left(s"ddre onnx output '$outputName' not found")
    else
      val onnxValue: OnnxValue = opt.get()
      flattenNumericOutput(onnxValue.getValue, expectedSize)

  /** Flattens various ONNX output types (1D array, 2D matrix, Java List) into
    * a flat Double array. Validates that the result has the expected size.
    */
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

  /** Calls `close()` on an ONNX AutoCloseable resource, swallowing only
    * non-fatal exceptions. Used to clean up tensors, sessions, options, and
    * results in `finally` blocks without leaking native memory or hiding
    * fatal errors like OOM.
    */
  private def closeQuietly(resource: AutoCloseable): Unit =
    if resource != null then
      try resource.close()
      catch case NonFatal(_) => ()
