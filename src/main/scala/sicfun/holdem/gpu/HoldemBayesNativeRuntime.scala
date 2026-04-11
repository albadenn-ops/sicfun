package sicfun.holdem.gpu
import sicfun.holdem.*

import java.util.concurrent.atomic.AtomicReference

/** Runtime wrapper for native Bayesian posterior update computation via JNI.
  *
  * This object manages the lifecycle of two separate native libraries:
  *  - '''CPU backend''' (`sicfun_bayes_native`) -- pure C implementation of sequential
  *    Bayesian posterior updates, providing ~35-40x speedup over the JVM Scala path.
  *  - '''GPU backend''' (`sicfun_bayes_cuda`) -- CUDA-accelerated parallel posterior
  *    update, beneficial for large hypothesis spaces.
  *
  * The Bayesian update computes: `posterior(h) = prior(h) * product(likelihoods(o,h))`,
  * normalised across all hypotheses, for each observation `o` sequentially.
  *
  * ==Thread Safety==
  * Library loading is guarded by `AtomicReference` CAS, ensuring each library is
  * loaded at most once. The actual JNI calls are stateless (no mutable native state
  * between calls), so concurrent calls from different threads are safe.
  *
  * ==Error Handling==
  * Native status codes 100-163 are translated via [[describeStatus]].
  * `UnsatisfiedLinkError` is caught and reported as a descriptive `Left`.
  */
private[holdem] object HoldemBayesNativeRuntime:
  /** Which native library to use for computation. */
  enum Backend:
    case Cpu
    case Gpu

  /** Describes whether a specific backend library is loaded and ready.
    * @param available `true` if the library was loaded successfully
    * @param backend   which backend this status describes
    * @param detail    human-readable load status message
    */
  final case class Availability(available: Boolean, backend: Backend, detail: String)

  /** Result of a successful native Bayesian posterior update.
    * @param posterior      normalised posterior distribution over hypotheses
    * @param logEvidence    sum of log-evidence across all observations
    * @param lastEngineCode native engine code (1=cpu, 2=gpu) for telemetry
    */
  final case class NativeUpdateResult(
      posterior: Array[Double],
      logEvidence: Double,
      lastEngineCode: Int
  )

  private val CpuEngineCode = 1
  private val GpuEngineCode = 2

  private val CpuPathProperty = "sicfun.bayes.native.cpu.path"
  private val CpuPathEnv = "sicfun_BAYES_NATIVE_CPU_PATH"
  private val CpuLibProperty = "sicfun.bayes.native.cpu.lib"
  private val CpuLibEnv = "sicfun_BAYES_NATIVE_CPU_LIB"
  private val DefaultCpuLib = "sicfun_bayes_native"

  private val GpuPathProperty = "sicfun.bayes.native.gpu.path"
  private val GpuPathEnv = "sicfun_BAYES_NATIVE_GPU_PATH"
  private val GpuLibProperty = "sicfun.bayes.native.gpu.lib"
  private val GpuLibEnv = "sicfun_BAYES_NATIVE_GPU_LIB"
  private val DefaultGpuLib = "sicfun_bayes_cuda"

  private val cpuLoadResultRef = new AtomicReference[Either[String, String]](null)
  private val gpuLoadResultRef = new AtomicReference[Either[String, String]](null)

  private[holdem] def resetLoadCacheForTests(): Unit =
    cpuLoadResultRef.set(null)
    gpuLoadResultRef.set(null)

  /** Checks whether the specified backend library is loaded and ready for calls. */
  def availability(backend: Backend): Availability =
    val loadResult =
      backend match
        case Backend.Cpu => cpuLoadResult()
        case Backend.Gpu => gpuLoadResult()
    loadResult match
      case Right(source) =>
        Availability(
          available = true,
          backend = backend,
          detail = s"${backendLabel(backend)} native Bayesian provider loaded ($source)"
        )
      case Left(reason) =>
        Availability(
          available = false,
          backend = backend,
          detail = reason
        )

  /** Performs a Bayesian posterior update using the specified native backend.
    *
    * Allocates fresh output arrays, delegates to [[updatePosteriorInPlace]],
    * and wraps the result in a [[NativeUpdateResult]].
    *
    * @param backend          CPU or GPU native library
    * @param observationCount number of sequential observations (rows in the likelihood matrix)
    * @param hypothesisCount  number of hypotheses (columns in the likelihood matrix)
    * @param prior            prior probability for each hypothesis (length = hypothesisCount)
    * @param likelihoods      row-major likelihood matrix (length = observationCount * hypothesisCount)
    * @return `Right(result)` on success, `Left(reason)` on failure
    */
  def updatePosterior(
      backend: Backend,
      observationCount: Int,
      hypothesisCount: Int,
      prior: Array[Double],
      likelihoods: Array[Double]
  ): Either[String, NativeUpdateResult] =
    val outPosterior = new Array[Double](hypothesisCount)
    val outLogEvidence = Array(0.0d)
    updatePosteriorInPlace(
      backend = backend,
      observationCount = observationCount,
      hypothesisCount = hypothesisCount,
      prior = prior,
      likelihoods = likelihoods,
      outPosterior = outPosterior,
      outLogEvidence = outLogEvidence
    ).map { engineCode =>
      NativeUpdateResult(
        posterior = outPosterior,
        logEvidence = outLogEvidence(0),
        lastEngineCode = engineCode
      )
    }

  /** Lower-level in-place variant: writes the posterior into caller-provided arrays.
    * Used by [[HoldemBayesProvider]] to avoid an extra allocation when running
    * shadow validation against the Scala reference implementation.
    *
    * @return `Right(engineCode)` on success, `Left(reason)` on failure
    */
  private[holdem] def updatePosteriorInPlace(
      backend: Backend,
      observationCount: Int,
      hypothesisCount: Int,
      prior: Array[Double],
      likelihoods: Array[Double],
      outPosterior: Array[Double],
      outLogEvidence: Array[Double]
  ): Either[String, Int] =
    if outPosterior.length != hypothesisCount then
      Left(
        s"native Bayesian output posterior length mismatch: expected $hypothesisCount, found ${outPosterior.length}"
      )
    else if outLogEvidence.length < 1 then
      Left("native Bayesian logEvidence output must have length >= 1")
    else
      updatePosteriorInPlaceUnchecked(
        backend = backend,
        observationCount = observationCount,
        hypothesisCount = hypothesisCount,
        prior = prior,
        likelihoods = likelihoods,
        outPosterior = outPosterior,
        outLogEvidence = outLogEvidence
      )

  /** Lower-level in-place variant with two-layer tempering (SICFUN v0.30.2 Def 15A/15B).
    *
    * @param kappaTemp     power-posterior exponent in (0, 1]
    * @param deltaFloor    additive safety floor >= 0
    * @param eta           full-support distribution, length = hypothesisCount (null for uniform)
    * @param useLegacyForm if true, use legacy (1-eps)*L + eps*eta formula
    * @return `Right(engineCode)` on success, `Left(reason)` on failure
    */
  private[holdem] def updatePosteriorTemperedInPlace(
      backend: Backend,
      observationCount: Int,
      hypothesisCount: Int,
      prior: Array[Double],
      likelihoods: Array[Double],
      kappaTemp: Double,
      deltaFloor: Double,
      eta: Array[Double],
      useLegacyForm: Boolean,
      outPosterior: Array[Double],
      outLogEvidence: Array[Double]
  ): Either[String, Int] =
    if outPosterior.length != hypothesisCount then
      Left(
        s"native Bayesian tempered output posterior length mismatch: expected $hypothesisCount, found ${outPosterior.length}"
      )
    else if outLogEvidence.length < 1 then
      Left("native Bayesian tempered logEvidence output must have length >= 1")
    else if eta != null && eta.length != hypothesisCount then
      Left(
        s"native Bayesian tempered eta length mismatch: expected $hypothesisCount, found ${eta.length}"
      )
    else
      val loadResult =
        backend match
          case Backend.Cpu => cpuLoadResult()
          case Backend.Gpu => gpuLoadResult()
      loadResult match
        case Left(reason) =>
          Left(reason)
        case Right(_) =>
          try
            val status =
              backend match
                case Backend.Cpu =>
                  HoldemBayesNativeCpuBindings.updatePosteriorTempered(
                    observationCount,
                    hypothesisCount,
                    prior,
                    likelihoods,
                    kappaTemp,
                    deltaFloor,
                    eta,
                    useLegacyForm,
                    outPosterior,
                    outLogEvidence
                  )
                case Backend.Gpu =>
                  HoldemBayesNativeGpuBindings.updatePosteriorTempered(
                    observationCount,
                    hypothesisCount,
                    prior,
                    likelihoods,
                    kappaTemp,
                    deltaFloor,
                    eta,
                    useLegacyForm,
                    outPosterior,
                    outLogEvidence
                  )
            if status != 0 then Left(describeStatus(status))
            else
              val engineCode =
                backend match
                  case Backend.Cpu => CpuEngineCode
                  case Backend.Gpu => GpuEngineCode
              Right(engineCode)
          catch
            case ex: UnsatisfiedLinkError =>
              Left(s"${backendLabel(backend)} native tempered Bayesian symbols not found: ${ex.getMessage}")
            case ex: Throwable =>
              Left(
                Option(ex.getMessage)
                  .map(_.trim)
                  .filter(_.nonEmpty)
                  .getOrElse(ex.getClass.getSimpleName)
              )

  /** Dispatches the actual JNI call to the appropriate backend bindings class.
    * Assumes all pre-conditions have been validated by the caller.
    */
  private def updatePosteriorInPlaceUnchecked(
      backend: Backend,
      observationCount: Int,
      hypothesisCount: Int,
      prior: Array[Double],
      likelihoods: Array[Double],
      outPosterior: Array[Double],
      outLogEvidence: Array[Double]
  ): Either[String, Int] =
    val loadResult =
      backend match
        case Backend.Cpu => cpuLoadResult()
        case Backend.Gpu => gpuLoadResult()
    loadResult match
      case Left(reason) =>
        Left(reason)
      case Right(_) =>
        try
          val status =
            backend match
              case Backend.Cpu =>
                HoldemBayesNativeCpuBindings.updatePosterior(
                  observationCount,
                  hypothesisCount,
                  prior,
                  likelihoods,
                  outPosterior,
                  outLogEvidence
                )
              case Backend.Gpu =>
                HoldemBayesNativeGpuBindings.updatePosterior(
                  observationCount,
                  hypothesisCount,
                  prior,
                  likelihoods,
                  outPosterior,
                  outLogEvidence
                )

          if status != 0 then Left(describeStatus(status))
          else
            val engineCode =
              backend match
                case Backend.Cpu => CpuEngineCode
                case Backend.Gpu => GpuEngineCode
            Right(engineCode)
        catch
          case ex: UnsatisfiedLinkError =>
            Left(s"${backendLabel(backend)} native Bayesian symbols not found: ${ex.getMessage}")
          case ex: Throwable =>
            Left(
              Option(ex.getMessage)
                .map(_.trim)
                .filter(_.nonEmpty)
                .getOrElse(ex.getClass.getSimpleName)
            )

  /** Lazily loads the CPU Bayesian native library using CAS for thread safety.
    * Returns a cached result on subsequent calls.
    */
  private def cpuLoadResult(): Either[String, String] =
    val cached = cpuLoadResultRef.get()
    if cached != null then cached
    else
      val loaded = GpuRuntimeSupport.loadNativeLibrary(
        pathProperty = CpuPathProperty,
        pathEnv = CpuPathEnv,
        libProperty = CpuLibProperty,
        libEnv = CpuLibEnv,
        defaultLib = DefaultCpuLib,
        label = "native Bayesian CPU library"
      )
      cpuLoadResultRef.compareAndSet(null, loaded)
      cpuLoadResultRef.get()

  /** Lazily loads the GPU (CUDA) Bayesian native library using CAS for thread safety. */
  private def gpuLoadResult(): Either[String, String] =
    val cached = gpuLoadResultRef.get()
    if cached != null then cached
    else
      val loaded = GpuRuntimeSupport.loadNativeLibrary(
        pathProperty = GpuPathProperty,
        pathEnv = GpuPathEnv,
        libProperty = GpuLibProperty,
        libEnv = GpuLibEnv,
        defaultLib = DefaultGpuLib,
        label = "native Bayesian GPU library"
      )
      gpuLoadResultRef.compareAndSet(null, loaded)
      gpuLoadResultRef.get()

  private def backendLabel(backend: Backend): String =
    backend match
      case Backend.Cpu => "CPU"
      case Backend.Gpu => "GPU"

  /** Translates native Bayesian JNI status codes to human-readable descriptions.
    * Codes 100-102/124 are shared JNI validation errors; 160-163 are Bayesian-specific.
    */
  def describeStatus(status: Int): String =
    val detail =
      status match
        case 100 => "null JNI array argument"
        case 101 => "native Bayesian arrays have mismatched lengths"
        case 102 => "failed reading JNI input arrays"
        case 124 => "failed writing JNI output arrays"
        case 160 => "invalid observation/hypothesis configuration"
        case 161 => "invalid prior distribution values"
        case 162 => "invalid likelihood values"
        case 163 => "likelihoods produced zero evidence"
        case _ => "unknown native Bayesian status"
    s"native Bayesian update returned status=$status ($detail)"
