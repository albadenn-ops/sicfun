package sicfun.holdem

import java.util.concurrent.atomic.AtomicReference

/** Runtime wrapper for native Bayesian posterior update providers (CPU and CUDA-compiled provider). */
private[holdem] object HoldemBayesNativeRuntime:
  enum Backend:
    case Cpu
    case Gpu

  final case class Availability(available: Boolean, backend: Backend, detail: String)

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
