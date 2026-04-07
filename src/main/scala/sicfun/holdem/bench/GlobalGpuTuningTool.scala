package sicfun.holdem.bench

import sicfun.holdem.*
import sicfun.holdem.cli.*
import sicfun.holdem.equity.*
import sicfun.holdem.gpu.*
import sicfun.holdem.tablegen.{HeadsUpCanonicalExactBoardMajorTuner, HeadsUpCanonicalExactTuner}

import java.io.{BufferedReader, ByteArrayOutputStream, File, FileInputStream, FileOutputStream, InputStreamReader, OutputStream, PrintStream}
import java.util.Properties
import java.util.concurrent.atomic.AtomicReference

/** Operator-facing entrypoint that consolidates the persistent GPU/native tuning surfaces.
  *
  * Targets:
  *   - `runtime` (default): backend MC/exact, range, postflop
  *   - `hybrid`: backend hybrid MC
  *   - `research`: canonical exact research harnesses
  *   - `all`: runtime + hybrid + research
  *
  * Concrete target names are also supported for narrower runs.
  */
object GlobalGpuTuningTool:
  private final case class Config(
      targets: Vector[String] = Vector("runtime"),
      force: Boolean = false
  )

  private final case class TargetResult(
      name: String,
      status: String,
      detail: String
  )

  private final case class NativeArtifacts(
      headsupGpuDll: File,
      postflopGpuDll: File
  )

  private final case class RequiredNativeArtifact(
      file: File,
      label: String
  )

  private final case class RangeCacheEntry(
      index: Int,
      fingerprint: String,
      blockSize: Int,
      maxChunkHeroes: Int,
      memoryPath: String
  )

  private final case class PostflopCacheEntry(
      index: Int,
      fingerprint: String,
      blockSize: Int,
      maxChunkMatchups: Int
  )

  private final case class ResearchCacheEntry(
      signature: String,
      target: String,
      summary: String
  )

  private val AllowedOptionKeys = Set("targets", "force")

  private val RuntimeTargets = Vector(
    "backend-full-montecarlo",
    "backend-canonical-montecarlo",
    "backend-full-exact",
    "backend-canonical-exact",
    "range",
    "postflop"
  )
  private val HybridTargets = Vector(
    "hybrid-full-montecarlo",
    "hybrid-canonical-montecarlo"
  )
  private val ResearchTargets = Vector(
    "canonical-exact",
    "canonical-board-major"
  )

  private val RangeCachePath = "data/headsup-range-autotune.properties"
  private val PostflopCachePath = "data/postflop-autotune.properties"
  private val ResearchCachePath = "data/global-gpu-tuning-autotune.properties"
  private val ResearchCacheVersion = "1"
  private val CanonicalExactTunerSource = "src/main/scala/sicfun/holdem/tablegen/HeadsUpCanonicalExactTuner.scala"
  private val CanonicalBoardMajorTunerSource = "src/main/scala/sicfun/holdem/tablegen/HeadsUpCanonicalExactBoardMajorTuner.scala"

  private val BackendMonteCarloTrials = 700
  private val BackendMonteCarloMatchups = 12000L
  private val BackendExactMatchups = 768L
  private val ExpectedRangeHeroes = 256
  private val ExpectedRangeEntriesPerHero = 128
  private val ExpectedRangeTrials = 256
  private val ExpectedRangeWarmupRuns = 1
  private val ExpectedRangeRuns = 3
  private val ExpectedRangeSeedBase = 0x38A7B35C1DF6241EL
  private val ExpectedRangeBlockCandidates = "32,64,128,256"
  private val ExpectedRangeChunkHeroesCandidates = "128,256,512,1024,2048,4096"
  private val ExpectedRangeMemoryPathCandidates = "global,readonly"
  private val ExpectedPostflopVillains = 1024
  private val ExpectedPostflopTrials = 2000
  private val ExpectedPostflopWarmupRuns = 1
  private val ExpectedPostflopRuns = 3
  private val ExpectedPostflopSeedBase = 0x6F31E52D9A4BC117L
  private val ExpectedPostflopBlockCandidates = "64,96,128,160,192,256"
  private val ExpectedPostflopChunkCandidates = "256,512,1024,2048,4096"

  private val GpuProviderProperty = "sicfun.gpu.provider"
  private val GpuAutoTuneProperty = "sicfun.gpu.autotune"
  private val GpuNativeEngineProperty = "sicfun.gpu.native.engine"
  private val GpuNativeCudaBlockSizeProperty = "sicfun.gpu.native.cuda.blockSize"
  private val GpuNativeCudaMaxChunkProperty = "sicfun.gpu.native.cuda.maxChunkMatchups"
  private val GpuNativePathProperty = "sicfun.gpu.native.path"
  private val GpuNativePathEnv = "sicfun_GPU_NATIVE_PATH"
  private val GpuNativeLibProperty = "sicfun.gpu.native.lib"
  private val GpuNativeLibEnv = "sicfun_GPU_NATIVE_LIB"
  private val DefaultGpuNativeLibrary = "sicfun_gpu_kernel"

  private val PostflopGpuPathProperty = "sicfun.postflop.native.gpu.path"
  private val PostflopGpuPathEnv = "sicfun_POSTFLOP_NATIVE_GPU_PATH"
  private val PostflopGpuLibProperty = "sicfun.postflop.native.gpu.lib"
  private val PostflopGpuLibEnv = "sicfun_POSTFLOP_NATIVE_GPU_LIB"
  private val DefaultPostflopGpuLibrary = "sicfun_postflop_cuda"
  private val GpuBuildJavaHomeProperty = "sicfun.gpu.build.javaHome"
  private val GpuBuildJavaHomeEnv = "SICFUN_GPU_BUILD_JAVA_HOME"
  private val GpuBuildCudaRootProperty = "sicfun.gpu.build.cudaRoot"
  private val GpuBuildCudaRootEnv = "SICFUN_GPU_BUILD_CUDA_ROOT"
  private val GpuBuildVcVarsProperty = "sicfun.gpu.build.vcvars"
  private val GpuBuildVcVarsEnv = "SICFUN_GPU_BUILD_VCVARS"
  private val GpuBuildArchProperty = "sicfun.gpu.build.arch"
  private val GpuBuildArchEnv = "SICFUN_GPU_BUILD_ARCH"
  private val GpuBuildArchitecturesProperty = "sicfun.gpu.build.architectures"
  private val GpuBuildArchitecturesEnv = "SICFUN_GPU_BUILD_ARCHITECTURES"
  private val GpuBuildInstallPrereqsProperty = "sicfun.gpu.build.installPrereqs"
  private val GpuBuildInstallPrereqsEnv = "SICFUN_GPU_BUILD_INSTALL_PREREQS"
  private val RepoRootProperty = "sicfun.repo.root"
  private val RepoRootEnv = "SICFUN_REPO_ROOT"
  private val NativeBuildScriptRelativePath = "src/main/native/build-windows-cuda11.ps1"
  private val HeadsupGpuDllRelativePath = "src/main/native/build/sicfun_gpu_kernel.dll"
  private val PostflopGpuDllRelativePath = "src/main/native/build/sicfun_postflop_cuda.dll"
  private val NativeBuildCommandText = s"powershell -ExecutionPolicy Bypass -File $NativeBuildScriptRelativePath"
  private val cudaBuildResultRef = new AtomicReference[Either[String, Unit]](null)
  private val SupportedWindowsCudaBuildArchitectures = Set("amd64", "x86_64", "x64")

  /** Entry point. Parses CLI args, expands target groups (e.g. "runtime" -> 6 individual
    * targets, "all" -> runtime + hybrid + research), then runs each target sequentially.
    * Each target auto-discovers or auto-builds the required native DLL artifacts, then
    * delegates to the appropriate auto-tuner (HeadsUpBackendAutoTuner, HeadsUpRangeGpuAutoTuner,
    * HoldemPostflopGpuAutoTuner, or HeadsUpCanonicalExactTuner). Results are cached in
    * properties files so subsequent runs skip already-tuned configurations unless --force=true.
    * Throws if any target fails.
    */
  def main(args: Array[String]): Unit =
    val config = parseArgs(args.toVector)
    val targets = expandTargets(config.targets)
    println("global gpu tuning tool")
    println(s"targets=${targets.mkString(", ")}, force=${config.force}")

    val results = targets.map(runTarget(_, config.force))
    println("global gpu tuning summary")
    results.foreach { result =>
      println(s"  ${result.name}: ${result.status} (${result.detail})")
    }

    val failures = results.filter(_.status == "failed")
    if failures.nonEmpty then
      throw new IllegalStateException("global gpu tuning failed for: " + failures.map(_.name).mkString(", "))

  /** Dispatches a single named target to the appropriate tuning workflow.
    * Backend targets (backend-*) run HeadsUpBackendAutoTuner for GPU kernel parameter selection.
    * Hybrid targets (hybrid-*) run HeadsUpBackendAutoTuner in hybrid multi-device mode.
    * "range" runs HeadsUpRangeGpuAutoTuner for CSR range evaluation parameters.
    * "postflop" runs HoldemPostflopGpuAutoTuner for postflop MC parameters.
    * Research targets run HeadsUpCanonicalExactTuner/HeadsUpCanonicalExactBoardMajorTuner.
    */
  private def runTarget(target: String, force: Boolean): TargetResult =
    try
      target match
        case "backend-full-montecarlo" =>
          withHeadsupGpuNativeArtifacts {
            runBackendTarget(
              name = target,
              provider = "native",
              tableKind = "full",
              mode = HeadsUpEquityTable.Mode.MonteCarlo(BackendMonteCarloTrials),
              maxMatchups = BackendMonteCarloMatchups,
              force = force
            )
          }
        case "backend-canonical-montecarlo" =>
          withHeadsupGpuNativeArtifacts {
            runBackendTarget(
              name = target,
              provider = "native",
              tableKind = "canonical",
              mode = HeadsUpEquityTable.Mode.MonteCarlo(BackendMonteCarloTrials),
              maxMatchups = BackendMonteCarloMatchups,
              force = force
            )
          }
        case "backend-full-exact" =>
          withHeadsupGpuNativeArtifacts {
            runBackendTarget(
              name = target,
              provider = "native",
              tableKind = "full",
              mode = HeadsUpEquityTable.Mode.Exact,
              maxMatchups = BackendExactMatchups,
              force = force
            )
          }
        case "backend-canonical-exact" =>
          withHeadsupGpuNativeArtifacts {
            runBackendTarget(
              name = target,
              provider = "native",
              tableKind = "canonical",
              mode = HeadsUpEquityTable.Mode.Exact,
              maxMatchups = BackendExactMatchups,
              force = force
            )
          }
        case "hybrid-full-montecarlo" =>
          withHeadsupGpuNativeArtifacts {
            runBackendTarget(
              name = target,
              provider = "hybrid",
              tableKind = "full",
              mode = HeadsUpEquityTable.Mode.MonteCarlo(BackendMonteCarloTrials),
              maxMatchups = BackendMonteCarloMatchups,
              force = force
            )
          }
        case "hybrid-canonical-montecarlo" =>
          withHeadsupGpuNativeArtifacts {
            runBackendTarget(
              name = target,
              provider = "hybrid",
              tableKind = "canonical",
              mode = HeadsUpEquityTable.Mode.MonteCarlo(BackendMonteCarloTrials),
              maxMatchups = BackendMonteCarloMatchups,
              force = force
            )
          }
        case "range" =>
          withHeadsupGpuNativeArtifacts {
            runRangeTarget(force)
          }
        case "postflop" =>
          withPostflopGpuNativeArtifacts {
            runPostflopTarget(force)
          }
        case "canonical-exact" =>
          withHeadsupGpuNativeArtifacts {
            runResearchTarget(
              name = target,
              args = Array(
                "--maxMatchups=5000",
                "--seed=1",
                "--warmup=true"
              )
            ) {
              HeadsUpCanonicalExactTuner.main(_)
            }(force)
          }
        case "canonical-board-major" =>
          withHeadsupGpuNativeArtifacts {
            runResearchTarget(
              name = target,
              args = Array(
                "--maxMatchups=5000",
                "--seed=1",
                "--warmup=true"
              )
            ) {
              HeadsUpCanonicalExactBoardMajorTuner.main(_)
            }(force)
          }
        case other =>
          TargetResult(other, "failed", s"unknown target '$other'")
    catch
      case ex: Throwable =>
        TargetResult(
          target,
          "failed",
          Option(ex.getMessage).map(_.trim).filter(_.nonEmpty).getOrElse(ex.getClass.getSimpleName)
        )

  private def runBackendTarget(
      name: String,
      provider: String,
      tableKind: String,
      mode: HeadsUpEquityTable.Mode,
      maxMatchups: Long,
      force: Boolean
  ): TargetResult =
    withSystemProperties(
      Seq(
        GpuAutoTuneProperty -> Some("true"),
        GpuProviderProperty -> Some(provider),
        GpuNativeEngineProperty -> None,
        GpuNativeCudaBlockSizeProperty -> None,
        GpuNativeCudaMaxChunkProperty -> None
      )
    ) {
      HeadsUpGpuRuntime.resetLoadCacheForTests()
      val availability = HeadsUpGpuRuntime.availability
      if !availability.available then
        TargetResult(name, "failed", availability.detail)
      else
        provider match
          case "hybrid" =>
            HeadsUpBackendAutoTuner.configureForHybrid(
              tableKind = tableKind,
              mode = mode,
              maxMatchups = maxMatchups,
              forceRetune = force
            )
          case _ =>
            HeadsUpBackendAutoTuner.configureForGeneration(
              tableKind = tableKind,
              mode = mode,
              maxMatchups = maxMatchups,
              backend = HeadsUpEquityTable.ComputeBackend.Gpu,
              forceRetune = force
            )
        val modeLabel =
          mode match
            case HeadsUpEquityTable.Mode.Exact => "exact"
            case HeadsUpEquityTable.Mode.MonteCarlo(trials) => s"montecarlo:$trials"
        TargetResult(name, if force then "retuned" else "checked", s"provider=$provider table=$tableKind mode=$modeLabel")
    }

  private def runRangeTarget(force: Boolean): TargetResult =
    val cacheFile = new File(RangeCachePath)
    if !force then
      loadRangeCacheSummary(cacheFile) match
        case Some(summary) =>
          println(s"range: cached $summary")
          return TargetResult("range", "cached", summary)
        case None => ()
    val availability = withSystemProperties(
      Seq(
        GpuProviderProperty -> Some("native"),
        GpuNativeEngineProperty -> Some("cuda")
      )
    ) {
      HeadsUpGpuRuntime.resetLoadCacheForTests()
      HeadsUpGpuRuntime.availability
    }
    if !availability.available then
      return TargetResult("range", "failed", availability.detail)
    HeadsUpRangeGpuAutoTuner.main(
      Array(
        "--heroes=256",
        "--entriesPerHero=128",
        "--trials=256",
        "--warmupRuns=1",
        "--runs=3",
        s"--cachePath=$RangeCachePath"
      )
    )
    TargetResult("range", if force then "retuned" else "tuned", "cache refreshed")

  private def runPostflopTarget(force: Boolean): TargetResult =
    val cacheFile = new File(PostflopCachePath)
    if !force then
      loadPostflopCacheSummary(cacheFile) match
        case Some(summary) =>
          println(s"postflop: cached $summary")
          return TargetResult("postflop", "cached", summary)
        case None => ()
    val availability = withSystemProperties(
      Seq(
        "sicfun.postflop.provider" -> Some("native"),
        "sicfun.postflop.native.engine" -> Some("cuda"),
        "sicfun.postflop.autotune" -> Some("false")
      )
    ) {
      HoldemPostflopNativeRuntime.resetLoadCacheForTests()
      HoldemPostflopNativeRuntime.availability
    }
    if !availability.available then
      return TargetResult("postflop", "failed", availability.detail)
    HoldemPostflopGpuAutoTuner.main(
      Array(
        "--villains=1024",
        "--trials=2000",
        "--warmupRuns=1",
        "--runs=3",
        s"--cachePath=$PostflopCachePath"
      )
    )
    TargetResult("postflop", if force then "retuned" else "tuned", "cache refreshed")

  private def runResearchTarget(
      name: String,
      args: Array[String]
  )(
      runner: Array[String] => Unit
  )(force: Boolean): TargetResult =
    val signature = researchSignature(name, args)
    if !force then
      loadResearchCacheEntry(signature) match
        case Some(entry) =>
          println(s"$name: cached ${entry.summary}")
          return TargetResult(name, "cached", entry.summary)
        case None => ()
    val (_, output) = captureStdout {
      runner(args)
    }
    val summary = output.linesIterator.filter(_.trim.nonEmpty).toVector.reverseIterator.find(_.trim.startsWith("best=")).getOrElse("completed")
    saveResearchCacheEntry(signature, name, summary)
    TargetResult(name, if force then "retuned" else "tuned", summary)

  private def parseArgs(args: Vector[String]): Config =
    val options = CliHelpers.requireOptions(args)
    CliHelpers.requireNoUnknownOptions(options, AllowedOptionKeys)
    Config(
      targets = options.get("targets").map(raw => CliHelpers.requireCsvTokens(raw, "targets")).getOrElse(Vector("runtime")),
      force = CliHelpers.requireBooleanOption(options, "force", false)
    )

  private def expandTargets(rawTargets: Vector[String]): Vector[String] =
    rawTargets.foldLeft(Vector.empty[String]) { (acc, rawTarget) =>
      val expanded =
        rawTarget.trim.toLowerCase match
          case "runtime" => RuntimeTargets
          case "hybrid" => HybridTargets
          case "research" => ResearchTargets
          case "all" => RuntimeTargets ++ HybridTargets ++ ResearchTargets
          case concrete => Vector(concrete)
      expanded.foldLeft(acc) { (deduped, target) =>
        if deduped.contains(target) then deduped else deduped :+ target
      }
    }

  /** Loads and validates a cached range auto-tune result. Returns None if the cache file
    * is missing, version-mismatched, has a different native library identity, uses different
    * tuning parameters, or was produced for a different CUDA device topology.
    */
  private def loadRangeCacheSummary(file: File): Option[String] =
    if !file.isFile then None
    else
      val props = new Properties()
      val in = new FileInputStream(file)
      try props.load(in)
      finally in.close()
      val version = Option(props.getProperty("version")).map(_.trim).getOrElse("")
      val cachedNativeLibraryIdentity = Option(props.getProperty("nativeLibraryIdentity")).map(_.trim).getOrElse("")
      if version != "1" || cachedNativeLibraryIdentity.isEmpty || cachedNativeLibraryIdentity != configuredHeadsupGpuLibraryIdentity then None
      else if
        Option(props.getProperty("tune.heroes")).map(_.trim).getOrElse("") != ExpectedRangeHeroes.toString ||
          Option(props.getProperty("tune.entriesPerHero")).map(_.trim).getOrElse("") != ExpectedRangeEntriesPerHero.toString ||
          Option(props.getProperty("tune.trials")).map(_.trim).getOrElse("") != ExpectedRangeTrials.toString ||
          Option(props.getProperty("tune.warmupRuns")).map(_.trim).getOrElse("") != ExpectedRangeWarmupRuns.toString ||
          Option(props.getProperty("tune.runs")).map(_.trim).getOrElse("") != ExpectedRangeRuns.toString ||
          Option(props.getProperty("tune.seedBase")).map(_.trim).getOrElse("") != ExpectedRangeSeedBase.toString ||
          Option(props.getProperty("tune.blockCandidates")).map(_.trim).getOrElse("") != ExpectedRangeBlockCandidates ||
          Option(props.getProperty("tune.chunkHeroesCandidates")).map(_.trim).getOrElse("") != ExpectedRangeChunkHeroesCandidates ||
          Option(props.getProperty("tune.memoryPathCandidates")).map(_.trim).getOrElse("") != ExpectedRangeMemoryPathCandidates
      then None
      else
        val currentDevices = currentHeadsupCudaDevices()
        val entries = loadRangeCacheEntries(props)
        val entryMap = entries.map(entry => (entry.index, entry.fingerprint) -> entry).toMap
        if currentDevices.isEmpty || currentDevices.size != entries.size then None
        else
          val matched = currentDevices.flatMap(entryMap.get)
          if matched.size != currentDevices.size then None
          else
            Some(
              matched
                .sortBy(_.index)
                .map(entry => s"device=${entry.index} block=${entry.blockSize} chunkHeroes=${entry.maxChunkHeroes} memoryPath=${entry.memoryPath}")
                .mkString("; ")
            )

  private def loadPostflopCacheSummary(file: File): Option[String] =
    if !file.isFile then None
    else
      val props = new Properties()
      val in = new FileInputStream(file)
      try props.load(in)
      finally in.close()
      val version = Option(props.getProperty("version")).map(_.trim).getOrElse("")
      val cachedGpuLibraryIdentity = Option(props.getProperty("gpuLibraryIdentity")).map(_.trim).getOrElse("")
      if version != "1" || cachedGpuLibraryIdentity.isEmpty || cachedGpuLibraryIdentity != configuredPostflopGpuLibraryIdentity then None
      else if
        Option(props.getProperty("tune.villains")).map(_.trim).getOrElse("") != ExpectedPostflopVillains.toString ||
          Option(props.getProperty("tune.trials")).map(_.trim).getOrElse("") != ExpectedPostflopTrials.toString ||
          Option(props.getProperty("tune.warmupRuns")).map(_.trim).getOrElse("") != ExpectedPostflopWarmupRuns.toString ||
          Option(props.getProperty("tune.runs")).map(_.trim).getOrElse("") != ExpectedPostflopRuns.toString ||
          Option(props.getProperty("tune.seedBase")).map(_.trim).getOrElse("") != ExpectedPostflopSeedBase.toString ||
          Option(props.getProperty("tune.blockCandidates")).map(_.trim).getOrElse("") != ExpectedPostflopBlockCandidates ||
          Option(props.getProperty("tune.chunkCandidates")).map(_.trim).getOrElse("") != ExpectedPostflopChunkCandidates
      then None
      else
        val currentDevices = currentPostflopCudaDevices()
        val entries = loadPostflopCacheEntries(props)
        val entryMap = entries.map(entry => (entry.index, entry.fingerprint) -> entry).toMap
        if currentDevices.isEmpty || currentDevices.size != entries.size then None
        else
          val matched = currentDevices.flatMap(entryMap.get)
          if matched.size != currentDevices.size then None
          else
            Some(
              matched
                .sortBy(_.index)
                .map(entry => s"device=${entry.index} block=${entry.blockSize} chunkMatchups=${entry.maxChunkMatchups}")
                .mkString("; ")
            )

  private def loadRangeCacheEntries(props: Properties): Vector[RangeCacheEntry] =
    val count = GpuRuntimeSupport.parsePositiveIntOpt(props.getProperty("device.count")).getOrElse(0)
    (0 until count).flatMap { idx =>
      val prefix = s"device.$idx."
      for
        index <- GpuRuntimeSupport.parseNonNegativeIntOpt(props.getProperty(s"${prefix}index"))
        fingerprint <- Option(props.getProperty(s"${prefix}fingerprint")).map(_.trim).filter(_.nonEmpty)
        blockSize <- GpuRuntimeSupport.parsePositiveIntOpt(props.getProperty(s"${prefix}blockSize"))
        maxChunkHeroes <- GpuRuntimeSupport.parsePositiveIntOpt(props.getProperty(s"${prefix}maxChunkHeroes"))
        memoryPath <- Option(props.getProperty(s"${prefix}memoryPath")).map(_.trim).filter(_.nonEmpty)
      yield RangeCacheEntry(index, fingerprint, blockSize, maxChunkHeroes, memoryPath)
    }.toVector

  private def loadPostflopCacheEntries(props: Properties): Vector[PostflopCacheEntry] =
    val count = GpuRuntimeSupport.parsePositiveIntOpt(props.getProperty("device.count")).getOrElse(0)
    (0 until count).flatMap { idx =>
      val prefix = s"device.$idx."
      for
        index <- GpuRuntimeSupport.parseNonNegativeIntOpt(props.getProperty(s"${prefix}index"))
        fingerprint <- Option(props.getProperty(s"${prefix}fingerprint")).map(_.trim).filter(_.nonEmpty)
        blockSize <- GpuRuntimeSupport.parsePositiveIntOpt(props.getProperty(s"${prefix}blockSize"))
        maxChunkMatchups <-
          GpuRuntimeSupport
            .parsePositiveIntOpt(props.getProperty(s"${prefix}maxChunkMatchups"))
            .orElse(GpuRuntimeSupport.parsePositiveIntOpt(props.getProperty(s"${prefix}maxChunk")))
      yield PostflopCacheEntry(index, fingerprint, blockSize, maxChunkMatchups)
    }.toVector

  private def currentHeadsupCudaDevices(): Vector[(Int, String)] =
    try
      (0 until math.max(0, HeadsUpGpuNativeBindings.cudaDeviceCount())).flatMap { idx =>
        Option(HeadsUpGpuNativeBindings.cudaDeviceInfo(idx)).map(_.trim).filter(_.nonEmpty).map(info => idx -> info)
      }.toVector
    catch
      case _: Throwable => Vector.empty

  private def currentPostflopCudaDevices(): Vector[(Int, String)] =
    try
      (0 until math.max(0, HoldemPostflopNativeGpuBindings.cudaDeviceCount())).flatMap { idx =>
        Option(HoldemPostflopNativeGpuBindings.cudaDeviceInfo(idx)).map(_.trim).filter(_.nonEmpty).map(info => idx -> info)
      }.toVector
    catch
      case _: Throwable => Vector.empty

  private def configuredHeadsupGpuLibraryIdentity: String =
    GpuRuntimeSupport.resolveNonEmpty(GpuNativePathProperty, GpuNativePathEnv) match
      case Some(path) =>
        val file = new File(path)
        val mtime = if file.exists() then file.lastModified() else 0L
        s"path=${file.getAbsolutePath}|mtime=$mtime"
      case None =>
        val lib = GpuRuntimeSupport.resolveNonEmpty(GpuNativeLibProperty, GpuNativeLibEnv).getOrElse(DefaultGpuNativeLibrary)
        s"lib=$lib"

  private def configuredPostflopGpuLibraryIdentity: String =
    GpuRuntimeSupport.resolveNonEmpty(PostflopGpuPathProperty, PostflopGpuPathEnv) match
      case Some(path) =>
        val file = new File(path)
        val mtime = if file.exists() then file.lastModified() else 0L
        s"path=${file.getAbsolutePath}|mtime=$mtime"
      case None =>
        val lib = GpuRuntimeSupport.resolveNonEmpty(PostflopGpuLibProperty, PostflopGpuLibEnv).getOrElse(DefaultPostflopGpuLibrary)
        s"lib=$lib"

  /** Builds a cache signature for research targets that includes OS, arch, Java version,
    * CUDA device topology, native library identity, and the source file's last-modified
    * timestamp. Any change to the tuner source invalidates the cached result.
    */
  private def researchSignature(name: String, args: Array[String]): String =
    val os = System.getProperty("os.name", "unknown").trim.toLowerCase
    val arch = System.getProperty("os.arch", "unknown").trim.toLowerCase
    val javaVersion = System.getProperty("java.version", "unknown").trim.toLowerCase
    val devices = currentHeadsupCudaDevices().map { case (idx, info) => s"$idx:$info" }.mkString(";")
    s"v=$ResearchCacheVersion|target=$name|args=${args.mkString(" ")}|os=$os|arch=$arch|java=$javaVersion|lib=$configuredHeadsupGpuLibraryIdentity|devices=$devices|source=${researchSourceIdentity(name)}"

  private def researchSourceIdentity(name: String): String =
    name match
      case "canonical-exact" => sourceFileIdentity(CanonicalExactTunerSource)
      case "canonical-board-major" => sourceFileIdentity(CanonicalBoardMajorTunerSource)
      case other => s"unknown:$other"

  private def sourceFileIdentity(path: String): String =
    val file = new File(path)
    val mtime = if file.isFile then file.lastModified() else 0L
    s"path=$path|mtime=$mtime"

  private def loadResearchCacheEntry(signature: String): Option[ResearchCacheEntry] =
    val file = new File(ResearchCachePath)
    if !file.isFile then None
    else
      val props = new Properties()
      val in = new FileInputStream(file)
      try props.load(in)
      finally in.close()
      val version = Option(props.getProperty("version")).map(_.trim).getOrElse("")
      if version != ResearchCacheVersion then None
      else
        val count = GpuRuntimeSupport.parsePositiveIntOpt(props.getProperty("entry.count")).getOrElse(0)
        (0 until count).iterator.flatMap { idx =>
          val prefix = s"entry.$idx."
          val cachedSignature = Option(props.getProperty(s"${prefix}signature")).map(_.trim).filter(_.nonEmpty)
          val target = Option(props.getProperty(s"${prefix}target")).map(_.trim).filter(_.nonEmpty)
          val summary = Option(props.getProperty(s"${prefix}summary")).map(_.trim).filter(_.nonEmpty)
          for
            entrySignature <- cachedSignature
            entryTarget <- target
            entrySummary <- summary
          yield ResearchCacheEntry(entrySignature, entryTarget, entrySummary)
        }.find(_.signature == signature)

  private def saveResearchCacheEntry(signature: String, target: String, summary: String): Unit =
    val file = new File(ResearchCachePath)
    val parent = file.getParentFile
    if parent != null then parent.mkdirs()
    val props = new Properties()
    if file.isFile then
      val in = new FileInputStream(file)
      try props.load(in)
      finally in.close()
    val merged =
      (loadResearchCacheEntries(props).filterNot(_.signature == signature) :+ ResearchCacheEntry(signature, target, summary)).zipWithIndex
    props.clear()
    props.setProperty("version", ResearchCacheVersion)
    props.setProperty("entry.count", merged.size.toString)
    merged.foreach { case (entry, idx) =>
      val prefix = s"entry.$idx."
      props.setProperty(s"${prefix}signature", entry.signature)
      props.setProperty(s"${prefix}target", entry.target)
      props.setProperty(s"${prefix}summary", entry.summary)
    }
    props.setProperty("updatedAtMillis", System.currentTimeMillis().toString)
    val out = new FileOutputStream(file)
    try props.store(out, "global gpu tuning cache")
    finally out.close()

  private def loadResearchCacheEntries(props: Properties): Vector[ResearchCacheEntry] =
    val version = Option(props.getProperty("version")).map(_.trim).getOrElse("")
    if version != ResearchCacheVersion then Vector.empty
    else
      val count = GpuRuntimeSupport.parsePositiveIntOpt(props.getProperty("entry.count")).getOrElse(0)
      (0 until count).flatMap { idx =>
        val prefix = s"entry.$idx."
        val signature = Option(props.getProperty(s"${prefix}signature")).map(_.trim).filter(_.nonEmpty)
        val target = Option(props.getProperty(s"${prefix}target")).map(_.trim).filter(_.nonEmpty)
        val summary = Option(props.getProperty(s"${prefix}summary")).map(_.trim).filter(_.nonEmpty)
        for
          entrySignature <- signature
          entryTarget <- target
          entrySummary <- summary
        yield ResearchCacheEntry(entrySignature, entryTarget, entrySummary)
      }.toVector

  private def withHeadsupGpuNativeArtifacts[A](thunk: => A): A =
    val artifacts = ensureRequiredNativeArtifacts(Vector(requiredHeadsupGpuArtifact))
    withSystemProperties(
      Seq(
        GpuNativePathProperty -> Some(artifacts.headsupGpuDll.getAbsolutePath),
        GpuNativeLibProperty -> None
      )
    )(thunk)

  private def withPostflopGpuNativeArtifacts[A](thunk: => A): A =
    val artifacts = ensureRequiredNativeArtifacts(Vector(requiredPostflopGpuArtifact))
    withSystemProperties(
      Seq(
        PostflopGpuPathProperty -> Some(artifacts.postflopGpuDll.getAbsolutePath),
        PostflopGpuLibProperty -> None
      )
    )(thunk)

  /** Checks that all required native DLL files exist. If any are missing, attempts an
    * automatic CUDA build via the PowerShell build script. The build is attempted at most
    * once per JVM session (result cached in an AtomicReference). Throws if artifacts
    * remain missing after the build attempt.
    */
  private def ensureRequiredNativeArtifacts(required: Vector[RequiredNativeArtifact]): NativeArtifacts =
    val artifacts = resolvedNativeArtifacts()
    val missing = required.filterNot(artifact => artifact.file.isFile)
    if missing.isEmpty then artifacts
    else
      val buildResult = ensureCudaBuildAttempted(missing)
      val stillMissing = required.filterNot(artifact => artifact.file.isFile)
      if stillMissing.nonEmpty then
        val reason =
          buildResult match
            case Left(detail) => detail
            case Right(()) =>
              s"auto-build finished, but the required artifact(s) were still missing under ${resolvedNativeBuildDir.getAbsolutePath}"
        throw new IllegalStateException(formatMissingArtifactsMessage(stillMissing, reason))
      buildResult match
        case Left(reason) =>
          println(
            s"global gpu tuning: required artifact(s) are now present; continuing after build warning: $reason"
          )
        case Right(()) => ()
      artifacts

  private def ensureCudaBuildAttempted(missing: Vector[RequiredNativeArtifact]): Either[String, Unit] =
    val cached = cudaBuildResultRef.get()
    if cached != null then cached
    else
      val attempted = preflightCudaBuild(missing).flatMap(_ => runCudaBuild())
      cudaBuildResultRef.compareAndSet(null, attempted)
      cudaBuildResultRef.get()

  private def preflightCudaBuild(missing: Vector[RequiredNativeArtifact]): Either[String, Unit] =
    val osName = System.getProperty("os.name", "unknown").trim
    if !osName.toLowerCase.contains("win") then
      Left(s"auto-build is only supported on Windows, but os.name='$osName'")
    else if !isSupportedWindowsCudaBuildHost then
      Left(
        s"auto-build currently supports Windows x64 only because the native build script resolves vcvars64.bat and produces x64 DLLs; " +
          s"os.arch='${System.getProperty("os.arch", "unknown")}'"
      )
    else if !resolvedNativeBuildScript.isFile then
      Left(
        s"native build script not found: ${resolvedNativeBuildScript.getAbsolutePath}. " +
          s"Launch from the repo root, or set -D$RepoRootProperty=<repo-root>"
      )
    else
      resolveWindowsPowerShellExecutable().map(_ => ())

  private def runCudaBuild(): Either[String, Unit] =
    resolveWindowsPowerShellExecutable().flatMap { powershellExe =>
      println(
        s"global gpu tuning: required native artifact(s) missing, running $NativeBuildCommandText"
      )
      val command = new java.util.ArrayList[String]()
      command.add(powershellExe.getAbsolutePath)
      command.add("-ExecutionPolicy")
      command.add("Bypass")
      command.add("-File")
      command.add(resolvedNativeBuildScript.getAbsolutePath)
      configuredCudaBuildJavaHome.foreach { value =>
        command.add("-JavaHome")
        command.add(value)
      }
      configuredCudaBuildCudaRoot.foreach { value =>
        command.add("-CudaRoot")
        command.add(value)
      }
      configuredCudaBuildVcVars.foreach { value =>
        command.add("-VcVars")
        command.add(value)
      }
      configuredCudaBuildArch.foreach { value =>
        command.add("-Arch")
        command.add(value)
      }
      configuredCudaBuildArchitectures.foreach { value =>
        command.add("-Architectures")
        command.add(value)
      }
      if configuredCudaBuildInstallPrereqs then
        command.add("-InstallMissingPrerequisites")
      val processBuilder = new ProcessBuilder(command)
      processBuilder.directory(repoRoot)
      processBuilder.redirectErrorStream(true)
      try
        val process = processBuilder.start()
        val lines = readProcessOutput(process)
        val exitCode = process.waitFor()
        if exitCode == 0 then Right(())
        else
          val detail = lines.reverseIterator.map(_.trim).find(_.nonEmpty).getOrElse(s"exitCode=$exitCode")
          Left(s"$NativeBuildCommandText exited with code $exitCode ($detail)")
      catch
        case ex: Throwable =>
          Left(
            Option(ex.getMessage)
              .map(_.trim)
              .filter(_.nonEmpty)
              .map(message => s"failed to start $NativeBuildCommandText: $message")
              .getOrElse(s"failed to start $NativeBuildCommandText")
          )
    }

  private def readProcessOutput(process: Process): Vector[String] =
    val reader = new BufferedReader(new InputStreamReader(process.getInputStream))
    val lines = Vector.newBuilder[String]
    try
      var line = reader.readLine()
      while line != null do
        println(s"[native-build] $line")
        lines += line
        line = reader.readLine()
    finally reader.close()
    lines.result()

  private def resolveWindowsPowerShellExecutable(): Either[String, File] =
    val systemRootCandidate =
      Option(sys.env.getOrElse("SystemRoot", "")).map(_.trim).filter(_.nonEmpty)
        .map(root => new File(root, "System32\\WindowsPowerShell\\v1.0\\powershell.exe"))
    val pathCandidates = Vector("powershell.exe", "pwsh.exe").flatMap(resolveExecutableOnPath)
    val candidates = (systemRootCandidate.toVector ++ pathCandidates).distinct.map(_.getAbsoluteFile)
    candidates.find(_.isFile) match
      case Some(file) => Right(file)
      case None =>
        val attempted =
          if candidates.nonEmpty then candidates.map(_.getAbsolutePath).mkString(", ")
          else "SystemRoot and PATH did not provide powershell.exe or pwsh.exe"
        Left(s"unable to resolve a PowerShell executable. Tried: $attempted")

  private def formatMissingArtifactsMessage(
      missing: Vector[RequiredNativeArtifact],
      reason: String
  ): String =
    val labels = missing.map(_.label).mkString(", ")
    val paths = missing.map(artifact => s"${artifact.label} -> ${artifact.file.getAbsolutePath}").mkString("; ")
    s"required native artifact(s) missing for global GPU tuning: $labels. " +
      s"Expected path(s): $paths. " +
      s"Auto-build could not proceed or did not produce the expected file(s): $reason. " +
      s"Required build command: $NativeBuildCommandText. " +
      s"Repo root override: -D$RepoRootProperty=<repo-root>. " +
      s"Prerequisite auto-install opt-in: -D$GpuBuildInstallPrereqsProperty=true. " +
      s"Optional overrides: -D$GpuBuildJavaHomeProperty=<jdk>, -D$GpuBuildCudaRootProperty=<cuda-root>, " +
      s"-D$GpuBuildVcVarsProperty=<vcvars64.bat>, -D$GpuBuildArchProperty=<sm_xy>, " +
      s"-D$GpuBuildArchitecturesProperty=<csv>"

  private def resolvedNativeArtifacts(): NativeArtifacts =
    NativeArtifacts(
      headsupGpuDll = resolveRepoRelativeFile(HeadsupGpuDllRelativePath),
      postflopGpuDll = resolveRepoRelativeFile(PostflopGpuDllRelativePath)
    )

  private def requiredHeadsupGpuArtifact: RequiredNativeArtifact =
    RequiredNativeArtifact(
      file = resolvedNativeArtifacts().headsupGpuDll,
      label = "sicfun_gpu_kernel.dll"
    )

  private def requiredPostflopGpuArtifact: RequiredNativeArtifact =
    RequiredNativeArtifact(
      file = resolvedNativeArtifacts().postflopGpuDll,
      label = "sicfun_postflop_cuda.dll"
    )

  private def repoRoot: File =
    locateRepoRoot().getOrElse(new File(System.getProperty("user.dir", ".")).getAbsoluteFile)

  private def resolvedNativeBuildDir: File =
    resolveRepoRelativeFile("src/main/native/build")

  private def resolvedNativeBuildScript: File =
    resolveRepoRelativeFile(NativeBuildScriptRelativePath)

  private def resolveRepoRelativeFile(path: String): File =
    new File(repoRoot, path).getAbsoluteFile

  private def locateRepoRoot(): Option[File] =
    configuredRepoRoot.flatMap(searchRepoRootFrom)
      .orElse(searchRepoRootFrom(new File(System.getProperty("user.dir", "."))))
      .orElse(codeSourceDirectory.flatMap(searchRepoRootFrom))

  private def searchRepoRootFrom(start: File): Option[File] =
    ancestorFiles(start.getAbsoluteFile).find(isRepoRootDirectory)

  private def ancestorFiles(start: File): Iterator[File] =
    Iterator.iterate(Option(start))(_.flatMap(file => Option(file.getParentFile)))
      .takeWhile(_.nonEmpty)
      .flatten

  private def isRepoRootDirectory(candidate: File): Boolean =
    new File(candidate, "build.sbt").isFile &&
      new File(candidate, NativeBuildScriptRelativePath).isFile

  private def codeSourceDirectory: Option[File] =
    Option(getClass.getProtectionDomain)
      .flatMap(domain => Option(domain.getCodeSource))
      .flatMap(source => Option(source.getLocation))
      .flatMap(location => scala.util.Try(new File(location.toURI)).toOption)
      .map(file => if file.isFile then file.getParentFile else file)
      .flatMap(file => Option(file))

  private def configuredRepoRoot: Option[File] =
    GpuRuntimeSupport.resolveNonEmpty(RepoRootProperty, RepoRootEnv).map(path => new File(path).getAbsoluteFile)

  private def resolveExecutableOnPath(executable: String): Option[File] =
    Option(sys.env.getOrElse("PATH", ""))
      .toVector
      .flatMap(_.split(File.pathSeparator).toVector)
      .map(_.trim)
      .filter(_.nonEmpty)
      .map(segment => new File(segment, executable))
      .find(_.isFile)

  private def isSupportedWindowsCudaBuildHost: Boolean =
    SupportedWindowsCudaBuildArchitectures.contains(System.getProperty("os.arch", "").trim.toLowerCase)

  private def configuredCudaBuildJavaHome: Option[String] =
    GpuRuntimeSupport.resolveNonEmpty(GpuBuildJavaHomeProperty, GpuBuildJavaHomeEnv)

  private def configuredCudaBuildCudaRoot: Option[String] =
    GpuRuntimeSupport.resolveNonEmpty(GpuBuildCudaRootProperty, GpuBuildCudaRootEnv)

  private def configuredCudaBuildVcVars: Option[String] =
    GpuRuntimeSupport.resolveNonEmpty(GpuBuildVcVarsProperty, GpuBuildVcVarsEnv)

  private def configuredCudaBuildArch: Option[String] =
    GpuRuntimeSupport.resolveNonEmpty(GpuBuildArchProperty, GpuBuildArchEnv)

  private def configuredCudaBuildArchitectures: Option[String] =
    GpuRuntimeSupport.resolveNonEmpty(GpuBuildArchitecturesProperty, GpuBuildArchitecturesEnv)

  private def configuredCudaBuildInstallPrereqs: Boolean =
    GpuRuntimeSupport.resolveNonEmpty(GpuBuildInstallPrereqsProperty, GpuBuildInstallPrereqsEnv)
      .exists(GpuRuntimeSupport.parseTruthy)

  private def withSystemProperties[A](updates: Seq[(String, Option[String])])(thunk: => A): A =
    val previous = updates.map { case (key, _) => key -> sys.props.get(key) }
    updates.foreach {
      case (key, Some(value)) => sys.props.update(key, value)
      case (key, None) => sys.props.remove(key)
    }
    try thunk
    finally
      previous.foreach {
        case (key, Some(value)) => sys.props.update(key, value)
        case (key, None) => sys.props.remove(key)
      }

  private def captureStdout[A](thunk: => A): (A, String) =
    val buffer = new ByteArrayOutputStream()
    val tee = new PrintStream(new TeeOutputStream(Console.out, buffer), true, "UTF-8")
    val result =
      try Console.withOut(tee)(thunk)
      finally tee.flush()
    (result, buffer.toString("UTF-8"))

  private final class TeeOutputStream(primary: OutputStream, secondary: OutputStream) extends OutputStream:
    override def write(value: Int): Unit =
      primary.write(value)
      secondary.write(value)

    override def write(bytes: Array[Byte], off: Int, len: Int): Unit =
      primary.write(bytes, off, len)
      secondary.write(bytes, off, len)

    override def flush(): Unit =
      primary.flush()
      secondary.flush()
