package sicfun.holdem.cfr
import sicfun.holdem.types.*
import sicfun.holdem.*
import sicfun.holdem.gpu.*
import sicfun.core.{FixedVal, Prob}

import java.util.concurrent.atomic.AtomicReference

/** Runtime wrapper for native CFR providers (CPU and CUDA-compiled C/C++ libraries).
  *
  * This module manages the lifecycle of native CFR solver libraries loaded via JNI.
  * It provides a unified Scala API that delegates to either CPU (sicfun_cfr_native)
  * or GPU/CUDA (sicfun_cfr_cuda) compiled solvers, abstracting away the JNI binding
  * details and native library discovery.
  *
  * The native solvers implement the same CFR algorithm as [[CfrSolver]] but in
  * optimized C/CUDA code for 35-50x speedups. Three solve modes are supported:
  *  - '''Full tree solve''' (`solveTree`): returns average strategies for all infosets.
  *  - '''Root-only solve''' (`solveTreeRoot`): returns only the root infoset's strategy.
  *  - '''Fixed-point solve''' (`solveTreeFixed`): uses integer arithmetic matching
  *    the Scala fixed-point path for bit-exact cross-platform parity.
  *  - '''Batch solve''' (`solveTreeBatch`): GPU-only, solves multiple trees with
  *    identical topology but different terminal utilities/chance weights in one kernel launch.
  *
  * Library loading is lazy and cached via AtomicReference for thread safety. The
  * library path can be configured via system properties or environment variables:
  *  - CPU: `sicfun.cfr.native.cpu.path` / `sicfun_CFR_NATIVE_CPU_PATH`
  *  - GPU: `sicfun.cfr.native.gpu.path` / `sicfun_CFR_NATIVE_GPU_PATH`
  *
  * The `NativeTreeSpec` and `BatchTreeSpec` structures flatten the game tree into
  * parallel arrays suitable for efficient JNI transfer and cache-friendly native traversal.
  */
private[holdem] object HoldemCfrNativeRuntime:
  /** Which native backend to use for CFR solving. */
  enum Backend:
    case Cpu
    case Gpu

  /** Result of checking whether a native backend is loadable and ready. */
  final case class Availability(available: Boolean, backend: Backend, detail: String)

  /** Flattened game tree representation for double-precision native solvers.
    *
    * The game tree is encoded as parallel arrays indexed by node ID. Each node has a
    * type (terminal=0, chance=1, player=2/3), an edge range (start+count into the
    * edge arrays), and an infoset index (for player nodes). This flat layout enables
    * efficient JNI array transfer and cache-friendly traversal in native code.
    *
    * @param rootNodeId          index of the root node in the node arrays
    * @param rootInfoSetIndex    infoset index of the root decision point
    * @param nodeTypes           node type per node (0=terminal, 1=chance, 2=player0, 3=player1)
    * @param nodeStarts          start index into edge arrays for each node's children
    * @param nodeCounts          number of child edges for each node
    * @param nodeInfosets        infoset index for each node (-1 for non-player nodes)
    * @param edgeChildIds        child node ID for each edge
    * @param edgeProbabilities   chance probability for each edge (0.0 for player edges)
    * @param terminalUtilities   player 0 utility at each terminal node (0.0 for non-terminal)
    * @param infosetKeys         string key for each information set
    * @param infosetPlayers      acting player (0 or 1) for each information set
    * @param infosetActions      ordered legal actions for each information set
    * @param infosetActionCounts number of actions per information set
    */
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
    require(rootInfoSetIndex < infosetActionCounts.length, "rootInfoSetIndex must reference an infoset")

  /** Fixed-point variant of [[NativeTreeSpec]].
    * Edge probabilities use Q0.30 Prob raw ints, terminal utilities use Q1.30 FixedVal raw ints.
    */
  final case class NativeTreeSpecFixed(
      rootNodeId: Int,
      rootInfoSetIndex: Int,
      nodeTypes: Array[Int],
      nodeStarts: Array[Int],
      nodeCounts: Array[Int],
      nodeInfosets: Array[Int],
      edgeChildIds: Array[Int],
      edgeProbabilitiesRaw: Array[Int],
      terminalUtilitiesRaw: Array[Int],
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
    require(nodeTypes.length == terminalUtilitiesRaw.length, "nodeTypes/terminalUtilitiesRaw length mismatch")
    require(edgeChildIds.length == edgeProbabilitiesRaw.length, "edgeChildIds/edgeProbabilitiesRaw length mismatch")
    require(infosetKeys.length == infosetPlayers.length, "infosetKeys/infosetPlayers length mismatch")
    require(infosetKeys.length == infosetActions.length, "infosetKeys/infosetActions length mismatch")
    require(infosetKeys.length == infosetActionCounts.length, "infosetKeys/infosetActionCounts length mismatch")
    require(rootInfoSetIndex < infosetActionCounts.length, "rootInfoSetIndex must reference an infoset")

  /** Result from a native full-tree or fixed-point solve.
    * @param averageStrategiesFlattened flattened average strategies for all infosets, concatenated in infoset order
    * @param expectedValuePlayer0      game value for player 0 under the computed strategies
    * @param lastEngineCode            native engine diagnostic code (0 = normal)
    */
  final case class NativeSolveResult(
      averageStrategiesFlattened: Array[Double],
      expectedValuePlayer0: Double,
      lastEngineCode: Int
  )

  /** Result from a native root-only solve. Only the root infoset's strategy is returned. */
  final case class NativeRootSolveResult(
      rootStrategy: Array[Double],
      lastEngineCode: Int
  )

  /** Batched game tree specification for GPU kernel launch.
    *
    * All trees in the batch must share identical topology (same node types, edges,
    * infoset structure) but may differ in terminal utilities and chance weights.
    * The per-tree data is packed into flat arrays of size batchSize * nodeCount
    * (or batchSize * edgeCount) for efficient GPU memory transfer.
    *
    * @param batchSize number of trees in the batch
    */
  final case class BatchTreeSpec(
      rootNodeId: Int,
      nodeTypes: Array[Int],
      nodeStarts: Array[Int],
      nodeCounts: Array[Int],
      nodeInfosets: Array[Int],
      edgeChildIds: Array[Int],
      infosetPlayers: Array[Int],
      infosetActionCounts: Array[Int],
      infosetOffsets: Array[Int],
      terminalUtilities: Array[Float],
      chanceWeights: Array[Float],
      batchSize: Int
  ):
    require(batchSize > 0, "batchSize must be positive")
    val nodeCount: Int = nodeTypes.length
    val edgeCount: Int = edgeChildIds.length
    val infosetCount: Int = infosetActionCounts.length
    val strategySize: Int = infosetOffsets.last
    require(terminalUtilities.length == batchSize * nodeCount, "terminal utilities size mismatch")
    require(chanceWeights.length == batchSize * edgeCount, "chance weights size mismatch")

  /** NOTE: expectedValues are NOT computed by the current kernel (always 0.0f).
    * EV computation requires an additional tree walk after the CFR loop, which
    * is deferred to a follow-up. For decision policies, only strategies are needed.
    */
  final case class BatchSolveResult(
      averageStrategiesFlattened: Array[Float],
      expectedValues: Array[Float],
      batchSize: Int,
      strategySize: Int
  ):
    def strategiesForTree(treeIdx: Int): Array[Float] =
      val offset = treeIdx * strategySize
      java.util.Arrays.copyOfRange(averageStrategiesFlattened, offset, offset + strategySize)

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

  /** Checks whether the specified native backend is available (library loaded successfully). */
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

  /** Solves a full game tree via the native backend, returning average strategies for all infosets.
    *
    * Allocates output arrays, calls the JNI binding, and wraps the result.
    * Returns Left with a diagnostic message if the native library is unavailable or
    * the solver returns a non-zero status code.
    */
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

  /** Solves a game tree and returns only the root infoset's average strategy.
    * More efficient than [[solveTree]] when only the decision-point policy is needed.
    */
  def solveTreeRoot(
      backend: Backend,
      spec: NativeTreeSpec,
      config: CfrSolver.Config
  ): Either[String, NativeRootSolveResult] =
    val loadResult =
      backend match
        case Backend.Cpu => cpuLoadResult()
        case Backend.Gpu => gpuLoadResult()
    loadResult match
      case Left(reason) =>
        Left(reason)
      case Right(_) =>
        try
          val outRootStrategy = new Array[Double](spec.infosetActionCounts(spec.rootInfoSetIndex))
          val status =
            backend match
              case Backend.Cpu =>
                HoldemCfrNativeCpuBindings.solveTreeRoot(
                  config.iterations,
                  config.averagingDelay,
                  config.cfrPlus,
                  config.linearAveraging,
                  spec.rootNodeId,
                  spec.rootInfoSetIndex,
                  spec.nodeTypes,
                  spec.nodeStarts,
                  spec.nodeCounts,
                  spec.nodeInfosets,
                  spec.edgeChildIds,
                  spec.edgeProbabilities,
                  spec.terminalUtilities,
                  spec.infosetPlayers,
                  spec.infosetActionCounts,
                  outRootStrategy
                )
              case Backend.Gpu =>
                HoldemCfrNativeGpuBindings.solveTreeRoot(
                  config.iterations,
                  config.averagingDelay,
                  config.cfrPlus,
                  config.linearAveraging,
                  spec.rootNodeId,
                  spec.rootInfoSetIndex,
                  spec.nodeTypes,
                  spec.nodeStarts,
                  spec.nodeCounts,
                  spec.nodeInfosets,
                  spec.edgeChildIds,
                  spec.edgeProbabilities,
                  spec.terminalUtilities,
                  spec.infosetPlayers,
                  spec.infosetActionCounts,
                  outRootStrategy
                )
          if status != 0 then Left(describeStatus(status))
          else
            val engineCode =
              backend match
                case Backend.Cpu => safeLastEngineCodeCpu()
                case Backend.Gpu => safeLastEngineCodeGpu()
            Right(
              NativeRootSolveResult(
                rootStrategy = outRootStrategy,
                lastEngineCode = engineCode
              )
            )
        catch
          case ex: UnsatisfiedLinkError =>
            Left(s"${backendLabel(backend)} native CFR root-only symbols not found: ${ex.getMessage}")
          case ex: Throwable =>
            Left(
              Option(ex.getMessage)
                .map(_.trim)
                .filter(_.nonEmpty)
                .getOrElse(ex.getClass.getSimpleName)
            )

  /** Fixed-point variant of [[solveTree]]. Uses integer arrays for probabilities and utilities.
    * Converts the fixed-point output back to doubles for the public API.
    */
  def solveTreeFixed(
      backend: Backend,
      spec: NativeTreeSpecFixed,
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
          val outAverageStrategiesRaw = new Array[Int](spec.infosetActionCounts.sum)
          val outExpectedValueRaw = Array(0)
          val status =
            backend match
              case Backend.Cpu =>
                HoldemCfrNativeCpuBindings.solveTreeFixed(
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
                  spec.edgeProbabilitiesRaw,
                  spec.terminalUtilitiesRaw,
                  spec.infosetPlayers,
                  spec.infosetActionCounts,
                  outAverageStrategiesRaw,
                  outExpectedValueRaw
                )
              case Backend.Gpu =>
                HoldemCfrNativeGpuBindings.solveTreeFixed(
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
                  spec.edgeProbabilitiesRaw,
                  spec.terminalUtilitiesRaw,
                  spec.infosetPlayers,
                  spec.infosetActionCounts,
                  outAverageStrategiesRaw,
                  outExpectedValueRaw
                )
          if status != 0 then Left(describeStatus(status))
          else
            val engineCode =
              backend match
                case Backend.Cpu => safeLastEngineCodeCpu()
                case Backend.Gpu => safeLastEngineCodeGpu()
            Right(
              NativeSolveResult(
                averageStrategiesFlattened = outAverageStrategiesRaw.map(raw => raw.toDouble / Prob.Scale.toDouble),
                expectedValuePlayer0 = outExpectedValueRaw(0).toDouble / FixedVal.Scale.toDouble,
                lastEngineCode = engineCode
              )
            )
        catch
          case ex: UnsatisfiedLinkError =>
            Left(s"${backendLabel(backend)} native CFR fixed symbols not found: ${ex.getMessage}")
          case ex: Throwable =>
            Left(
              Option(ex.getMessage)
                .map(_.trim)
                .filter(_.nonEmpty)
                .getOrElse(ex.getClass.getSimpleName)
            )

  /** GPU-only batched solve: solves multiple trees with identical topology in one kernel launch.
    * Each tree may have different terminal utilities and chance weights (e.g., different hero hands).
    */
  def solveTreeBatch(
      spec: BatchTreeSpec,
      config: CfrSolver.Config
  ): Either[String, BatchSolveResult] =
    gpuLoadResult() match
      case Left(reason) => Left(reason)
      case Right(_) =>
        try
          val outStrategies = new Array[Float](spec.batchSize * spec.strategySize)
          val outEv = new Array[Float](spec.batchSize)
          val status = HoldemCfrNativeGpuBindings.solveTreeBatch(
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
            spec.infosetPlayers,
            spec.infosetActionCounts,
            spec.infosetOffsets,
            spec.terminalUtilities,
            spec.chanceWeights,
            outStrategies,
            outEv,
            spec.batchSize
          )
          if status != 0 then Left(describeStatus(status))
          else Right(BatchSolveResult(
            averageStrategiesFlattened = outStrategies,
            expectedValues = outEv,
            batchSize = spec.batchSize,
            strategySize = spec.strategySize
          ))
        catch
          case ex: UnsatisfiedLinkError =>
            Left(s"GPU batch CFR symbols not found: ${ex.getMessage}")
          case ex: Throwable =>
            Left(Option(ex.getMessage).map(_.trim).filter(_.nonEmpty)
              .getOrElse(ex.getClass.getSimpleName))

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

  /** Translates a native CFR status code into a human-readable diagnostic message.
    * Status codes 100-153 are defined by the C/CUDA solver implementation.
    */
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
