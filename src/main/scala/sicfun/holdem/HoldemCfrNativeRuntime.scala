package sicfun.holdem

import java.util.concurrent.atomic.AtomicReference

/** Runtime wrapper for native CFR providers (CPU and CUDA-compiled provider). */
private[holdem] object HoldemCfrNativeRuntime:
  enum Backend:
    case Cpu
    case Gpu

  final case class Availability(available: Boolean, backend: Backend, detail: String)

  final case class NativeTreeSpec(
      rootNodeId: Int,
      rootInfoSetIndex: Int,
      nodeTypes: Array[Int],
      nodeStarts: Array[Int],
      nodeCounts: Array[Int],
      nodeInfosets: Array[Int],
      edgeChildIds: Array[Int],
      edgeProbabilities: Array[Double],
      terminalUtilities: Array[Double],
      infosetKeys: Vector[String],
      infosetPlayers: Array[Int],
      infosetActions: Vector[Vector[PokerAction]],
      infosetActionCounts: Array[Int]
  ):
    require(rootNodeId >= 0, "rootNodeId must be non-negative")
    require(rootInfoSetIndex >= 0, "rootInfoSetIndex must be non-negative")
    require(nodeTypes.length == nodeStarts.length, "nodeTypes/nodeStarts length mismatch")
    require(nodeTypes.length == nodeCounts.length, "nodeTypes/nodeCounts length mismatch")
    require(nodeTypes.length == nodeInfosets.length, "nodeTypes/nodeInfosets length mismatch")
    require(nodeTypes.length == terminalUtilities.length, "nodeTypes/terminalUtilities length mismatch")
    require(edgeChildIds.length == edgeProbabilities.length, "edgeChildIds/edgeProbabilities length mismatch")
    require(infosetKeys.length == infosetPlayers.length, "infosetKeys/infosetPlayers length mismatch")
    require(infosetKeys.length == infosetActions.length, "infosetKeys/infosetActions length mismatch")
    require(infosetKeys.length == infosetActionCounts.length, "infosetKeys/infosetActionCounts length mismatch")

  final case class NativeSolveResult(
      averageStrategiesFlattened: Array[Double],
      expectedValuePlayer0: Double,
      lastEngineCode: Int
  )

  private val CpuPathProperty = "sicfun.cfr.native.cpu.path"
  private val CpuPathEnv = "sicfun_CFR_NATIVE_CPU_PATH"
  private val CpuLibProperty = "sicfun.cfr.native.cpu.lib"
  private val CpuLibEnv = "sicfun_CFR_NATIVE_CPU_LIB"
  private val DefaultCpuLib = "sicfun_cfr_native"

  private val GpuPathProperty = "sicfun.cfr.native.gpu.path"
  private val GpuPathEnv = "sicfun_CFR_NATIVE_GPU_PATH"
  private val GpuLibProperty = "sicfun.cfr.native.gpu.lib"
  private val GpuLibEnv = "sicfun_CFR_NATIVE_GPU_LIB"
  private val DefaultGpuLib = "sicfun_cfr_cuda"

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
          detail = s"${backendLabel(backend)} native CFR provider loaded ($source)"
        )
      case Left(reason) =>
        Availability(
          available = false,
          backend = backend,
          detail = reason
        )

  def solveTree(
      backend: Backend,
      spec: NativeTreeSpec,
      config: CfrSolver.Config
  ): Either[String, NativeSolveResult] =
    val loadResult =
      backend match
        case Backend.Cpu => cpuLoadResult()
        case Backend.Gpu => gpuLoadResult()
    loadResult match
      case Left(reason) =>
        Left(reason)
      case Right(_) =>
        try
          val outAverageStrategies = new Array[Double](spec.infosetActionCounts.sum)
          val outExpectedValue = Array(0.0d)
          val status =
            backend match
              case Backend.Cpu =>
                HoldemCfrNativeCpuBindings.solveTree(
                  config.iterations,
                  config.averagingDelay,
                  config.cfrPlus,
                  config.linearAveraging,
                  spec.rootNodeId,
                  spec.nodeTypes,
                  spec.nodeStarts,
                  spec.nodeCounts,
                  spec.nodeInfosets,
                  spec.edgeChildIds,
                  spec.edgeProbabilities,
                  spec.terminalUtilities,
                  spec.infosetPlayers,
                  spec.infosetActionCounts,
                  outAverageStrategies,
                  outExpectedValue
                )
              case Backend.Gpu =>
                HoldemCfrNativeGpuBindings.solveTree(
                  config.iterations,
                  config.averagingDelay,
                  config.cfrPlus,
                  config.linearAveraging,
                  spec.rootNodeId,
                  spec.nodeTypes,
                  spec.nodeStarts,
                  spec.nodeCounts,
                  spec.nodeInfosets,
                  spec.edgeChildIds,
                  spec.edgeProbabilities,
                  spec.terminalUtilities,
                  spec.infosetPlayers,
                  spec.infosetActionCounts,
                  outAverageStrategies,
                  outExpectedValue
                )

          if status != 0 then Left(describeStatus(status))
          else
            val engineCode =
              backend match
                case Backend.Cpu => safeLastEngineCodeCpu()
                case Backend.Gpu => safeLastEngineCodeGpu()
            Right(
              NativeSolveResult(
                averageStrategiesFlattened = outAverageStrategies,
                expectedValuePlayer0 = outExpectedValue(0),
                lastEngineCode = engineCode
              )
            )
        catch
          case ex: UnsatisfiedLinkError =>
            Left(s"${backendLabel(backend)} native CFR symbols not found: ${ex.getMessage}")
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
        label = "native CFR CPU library"
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
        label = "native CFR GPU library"
      )
      gpuLoadResultRef.compareAndSet(null, loaded)
      gpuLoadResultRef.get()

  private def safeLastEngineCodeCpu(): Int =
    try HoldemCfrNativeCpuBindings.lastEngineCode()
    catch case _: Throwable => 0

  private def safeLastEngineCodeGpu(): Int =
    try HoldemCfrNativeGpuBindings.lastEngineCode()
    catch case _: Throwable => 0

  private def backendLabel(backend: Backend): String =
    backend match
      case Backend.Cpu => "CPU"
      case Backend.Gpu => "GPU"

  def describeStatus(status: Int): String =
    val detail =
      status match
        case 100 => "null JNI array argument"
        case 101 => "native CFR arrays have mismatched lengths"
        case 102 => "failed reading JNI input arrays"
        case 111 => "invalid native CFR mode/config"
        case 124 => "failed writing JNI output arrays"
        case 146 => "invalid root node id"
        case 147 => "invalid iteration configuration"
        case 148 => "invalid node type"
        case 149 => "invalid node edge layout"
        case 150 => "invalid edge child node index"
        case 151 => "invalid infoset index/player"
        case 152 => "infoset action-count mismatch"
        case 153 => "invalid chance probabilities"
        case _ => "unknown native CFR status"
    s"native CFR returned status=$status ($detail)"
