package sicfun.holdem

import java.util.concurrent.atomic.AtomicReference

/** Runtime wrapper for native DDRE posterior inference providers (CPU and CUDA-compiled provider). */
private[holdem] object HoldemDdreNativeRuntime:
  enum Backend:
    case Cpu
    case Gpu

  final case class Availability(available: Boolean, backend: Backend, detail: String)

  final case class NativeInferenceResult(
      posterior: Array[Double],
      lastEngineCode: Int
  )

  private val CpuEngineCode = 1
  private val GpuEngineCode = 2

  private val CpuPathProperty = "sicfun.ddre.native.cpu.path"
  private val CpuPathEnv = "sicfun_DDRE_NATIVE_CPU_PATH"
  private val CpuLibProperty = "sicfun.ddre.native.cpu.lib"
  private val CpuLibEnv = "sicfun_DDRE_NATIVE_CPU_LIB"
  private val DefaultCpuLib = "sicfun_ddre_native"

  private val GpuPathProperty = "sicfun.ddre.native.gpu.path"
  private val GpuPathEnv = "sicfun_DDRE_NATIVE_GPU_PATH"
  private val GpuLibProperty = "sicfun.ddre.native.gpu.lib"
  private val GpuLibEnv = "sicfun_DDRE_NATIVE_GPU_LIB"
  private val DefaultGpuLib = "sicfun_ddre_cuda"

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
          detail = s"${backendLabel(backend)} native DDRE provider loaded ($source)"
        )
      case Left(reason) =>
        Availability(
          available = false,
          backend = backend,
          detail = reason
        )

  def inferPosterior(
      backend: Backend,
      observationCount: Int,
      hypothesisCount: Int,
      prior: Array[Double],
      likelihoods: Array[Double]
  ): Either[String, NativeInferenceResult] =
    val outPosterior = new Array[Double](hypothesisCount)
    inferPosteriorInPlace(
      backend = backend,
      observationCount = observationCount,
      hypothesisCount = hypothesisCount,
      prior = prior,
      likelihoods = likelihoods,
      outPosterior = outPosterior
    ).map { engineCode =>
      NativeInferenceResult(
        posterior = outPosterior,
        lastEngineCode = engineCode
      )
    }

  private[holdem] def inferPosteriorInPlace(
      backend: Backend,
      observationCount: Int,
      hypothesisCount: Int,
      prior: Array[Double],
      likelihoods: Array[Double],
      outPosterior: Array[Double]
  ): Either[String, Int] =
    if observationCount < 0 then
      Left(s"native DDRE observationCount must be >= 0, found $observationCount")
    else if hypothesisCount <= 0 then
      Left(s"native DDRE hypothesisCount must be > 0, found $hypothesisCount")
    else if outPosterior.length != hypothesisCount then
      Left(
        s"native DDRE output posterior length mismatch: expected $hypothesisCount, found ${outPosterior.length}"
      )
    else
      inferPosteriorInPlaceUnchecked(
        backend = backend,
        observationCount = observationCount,
        hypothesisCount = hypothesisCount,
        prior = prior,
        likelihoods = likelihoods,
        outPosterior = outPosterior
      )

  private def inferPosteriorInPlaceUnchecked(
      backend: Backend,
      observationCount: Int,
      hypothesisCount: Int,
      prior: Array[Double],
      likelihoods: Array[Double],
      outPosterior: Array[Double]
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
                HoldemDdreNativeCpuBindings.inferPosterior(
                  observationCount,
                  hypothesisCount,
                  prior,
                  likelihoods,
                  outPosterior
                )
              case Backend.Gpu =>
                HoldemDdreNativeGpuBindings.inferPosterior(
                  observationCount,
                  hypothesisCount,
                  prior,
                  likelihoods,
                  outPosterior
                )

          if status != 0 then Left(describeStatus(status))
          else
            val engineCode =
              backend match
                case Backend.Cpu => safeLastEngineCodeCpu().getOrElse(CpuEngineCode)
                case Backend.Gpu => safeLastEngineCodeGpu().getOrElse(GpuEngineCode)
            Right(engineCode)
        catch
          case ex: UnsatisfiedLinkError =>
            Left(s"${backendLabel(backend)} native DDRE symbols not found: ${ex.getMessage}")
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
        label = "native DDRE CPU library"
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
        label = "native DDRE GPU library"
      )
      gpuLoadResultRef.compareAndSet(null, loaded)
      gpuLoadResultRef.get()

  private def safeLastEngineCodeCpu(): Option[Int] =
    try Some(HoldemDdreNativeCpuBindings.lastEngineCode())
    catch case _: Throwable => None

  private def safeLastEngineCodeGpu(): Option[Int] =
    try Some(HoldemDdreNativeGpuBindings.lastEngineCode())
    catch case _: Throwable => None

  private def backendLabel(backend: Backend): String =
    backend match
      case Backend.Cpu => "CPU"
      case Backend.Gpu => "GPU"

  def describeStatus(status: Int): String =
    val detail =
      status match
        case 100 => "null JNI array argument"
        case 101 => "native DDRE arrays have mismatched lengths"
        case 102 => "failed reading JNI input arrays"
        case 124 => "failed writing JNI output arrays"
        case 160 => "invalid observation/hypothesis configuration"
        case 161 => "invalid prior distribution values"
        case 162 => "invalid likelihood values"
        case 163 => "likelihood/prior blend produced zero mass"
        case _ => "unknown native DDRE status"
    s"native DDRE inference returned status=$status ($detail)"
