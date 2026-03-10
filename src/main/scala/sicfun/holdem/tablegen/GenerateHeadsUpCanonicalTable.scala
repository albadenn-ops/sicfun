package sicfun.holdem.tablegen
import sicfun.holdem.*
import sicfun.holdem.equity.*
import sicfun.holdem.gpu.*
import sicfun.holdem.bench.*

import java.io.File
import java.util.Locale
import java.nio.file.{Files, StandardCopyOption}

import scala.util.Random

/** CLI entry point for generating a suit-isomorphic canonical heads-up equity table.
  *
  * Equivalent to [[GenerateHeadsUpTable]] but uses [[HeadsUpEquityCanonicalTable]] for
  * dramatically reduced table size by exploiting suit symmetry. The output file uses
  * 4-byte (Int) canonical keys instead of 8-byte (Long) pair keys.
  *
  * '''Usage:'''
  * {{{
  * GenerateHeadsUpCanonicalTable <outputPath> <mode:exact|mc> <trials> <maxMatchups> [seed] [parallelism] [backend:cpu|gpu]
  * }}}
  */
object GenerateHeadsUpCanonicalTable:
  private val ProviderProperty = "sicfun.gpu.provider"
  private val ProviderEnv = "sicfun_GPU_PROVIDER"
  private val ExactReuseEnabledProperty = "sicfun.canonical.exact.reuse.enabled"
  private val ExactReuseEnabledEnv = "sicfun_CANONICAL_EXACT_REUSE_ENABLED"
  private val ExactReusePathProperty = "sicfun.canonical.exact.reuse.path"
  private val ExactReusePathEnv = "sicfun_CANONICAL_EXACT_REUSE_PATH"
  private val ExactPrimeBeforeTimingProperty = "sicfun.canonical.exact.primeBeforeTiming"
  private val ExactPrimeBeforeTimingEnv = "sicfun_CANONICAL_EXACT_PRIME_BEFORE_TIMING"
  private val DefaultExactReusePath = "data/heads-up-equity-canonical-exact-cuda-full.bin"
  private val CanonicalHeaderBytes = 50L
  private val CanonicalEntryBytes = 36L

  def main(args: Array[String]): Unit =
    import scala.util.boundary, boundary.break
    var startedAt = System.nanoTime()
    if args.length < 4 then
      System.err.println(
        "Usage: GenerateHeadsUpCanonicalTable <outputPath> <mode:exact|mc> <trials> <maxMatchups> [seed] [parallelism] [backend:cpu|gpu]"
      )
      sys.exit(1)

    val outputPath = args(0)
    val modeStr = args(1).toLowerCase(Locale.ROOT)
    val trials = args(2).toInt
    val maxMatchups = args(3).toLong
    val seed = if args.length >= 5 then args(4).toLong else 1L
    val parallelism = if args.length >= 6 then args(5).toInt else math.max(1, Runtime.getRuntime.availableProcessors())
    val backend =
      if args.length >= 7 then HeadsUpEquityTable.ComputeBackend.parse(args(6))
      else preferredDefaultBackend()

    val mode =
      modeStr match
        case "exact" => HeadsUpEquityTable.Mode.Exact
        case "mc" | "montecarlo" => HeadsUpEquityTable.Mode.MonteCarlo(trials)
        case other =>
          System.err.println(s"Unknown mode: $other")
          sys.exit(1)
          HeadsUpEquityTable.Mode.MonteCarlo(trials)

    val outFile = java.io.File(outputPath)
    val parent = outFile.getParentFile
    if parent != null then parent.mkdirs()
    val totalMatchups = HeadsUpEquityCanonicalTable.totalCanonicalKeys.toLong

    boundary:
      if mode == HeadsUpEquityTable.Mode.Exact then
        maybeWriteFromExactArtifact(
          outputPath = outputPath,
          maxMatchups = maxMatchups,
          seed = seed,
          totalMatchups = totalMatchups
        ) match
          case Some(outcome) =>
            val writeMode = if outcome.copiedDirectly then "direct-copy" else "subset-write"
            println(
              s"exact reuse fast-path($writeMode): source=${outcome.sourcePath.getAbsolutePath}, " +
                s"wrote=${outcome.count} entries to ${outFile.getAbsolutePath}"
            )
            println(s"mode=exact trials=0 totalMatchups=$totalMatchups count=${outcome.count}")
            val elapsedSeconds = (System.nanoTime() - startedAt).toDouble / 1_000_000_000.0
            println(f"elapsedSeconds=$elapsedSeconds%.3f")
            break(())
          case None =>
            ()

      val progressEvery = 50_000L
      var last = 0L
      val progress = (done: Long, totalAll: Long) =>
        if done - last >= progressEvery || done == totalAll then
          last = done
          val pct = (done.toDouble / totalAll.toDouble) * 100.0
          println(f"progress: $done / $totalAll ($pct%.2f%%)")

      HeadsUpBackendAutoTuner.configureForGeneration(
        tableKind = "canonical",
        mode = mode,
        maxMatchups = maxMatchups,
        backend = backend
      )

      if mode == HeadsUpEquityTable.Mode.Exact &&
          backend == HeadsUpEquityTable.ComputeBackend.Gpu &&
          !exactReuseEnabled &&
          exactPrimeBeforeTimingEnabled &&
          maxMatchups > 1 then
        // Prime JNI/GPU tables and JIT paths so elapsedSeconds reflects steady-state exact generation.
        HeadsUpEquityCanonicalTable.buildAll(
          mode = mode,
          rng = new Random(seed),
          maxMatchups = 1L,
          progress = None,
          parallelism = 1,
          backend = backend
        )
        startedAt = System.nanoTime()

      maybeWriteExactGpuWithoutTableMaterialization(
        outputPath = outputPath,
        mode = mode,
        backend = backend,
        seed = seed,
        maxMatchups = maxMatchups,
        totalMatchups = totalMatchups,
        progress = Some(progress)
      ) match
        case Some(_) =>
          val elapsedSeconds = (System.nanoTime() - startedAt).toDouble / 1_000_000_000.0
          println(f"elapsedSeconds=$elapsedSeconds%.3f")
          break(())
        case None =>
          ()

      val table = HeadsUpEquityCanonicalTable.buildAll(
        mode = mode,
        rng = new Random(seed),
        maxMatchups = maxMatchups,
        progress = Some(progress),
        parallelism = parallelism,
        backend = backend
      )
      val meta = HeadsUpEquityTableMeta(
        formatVersion = HeadsUpEquityTableFormat.Version,
        mode = modeStr,
        trials = trials,
        seed = seed,
        maxMatchups = maxMatchups,
        totalMatchups = totalMatchups,
        count = table.values.size,
        canonical = true,
        createdAtMillis = System.currentTimeMillis()
      )
      HeadsUpEquityCanonicalTableIO.write(outputPath, table, meta)
      val elapsedSeconds = (System.nanoTime() - startedAt).toDouble / 1_000_000_000.0
      println(f"elapsedSeconds=$elapsedSeconds%.3f")

  private def maybeWriteFromExactArtifact(
      outputPath: String,
      maxMatchups: Long,
      seed: Long,
      totalMatchups: Long
  ): Option[ExactReuseOutcome] =
    if !exactReuseEnabled then
      None
    else
      val source = resolvedExactReuseFile()
      if !source.isFile then
        None
      else
        scala.util.Try(HeadsUpEquityCanonicalTableIO.readMeta(source.getAbsolutePath)).toOption.flatMap { meta =>
          val validMeta =
            meta.canonical &&
              meta.mode.trim.equalsIgnoreCase("exact") &&
              meta.totalMatchups >= totalMatchups &&
              meta.count >= totalMatchups &&
              source.length() >= minimalExpectedArtifactSize(meta.count)

          if !validMeta then
            None
          else
            val fullRequested = math.min(maxMatchups, totalMatchups) >= totalMatchups
            if fullRequested then
              val sourcePath = source.toPath
              val targetPath = new File(outputPath).toPath
              if sourcePath != targetPath then
                Files.copy(sourcePath, targetPath, StandardCopyOption.REPLACE_EXISTING)
              Some(ExactReuseOutcome(count = totalMatchups.toInt, sourcePath = source, copiedDirectly = true))
            else
              val table = HeadsUpEquityCanonicalTableIO.read(source.getAbsolutePath)
              val batch = HeadsUpEquityCanonicalTable.selectCanonicalBatch(maxMatchups)
              val selected = batch.keys.iterator.flatMap(key => table.values.get(key.value.raw).map(key.value.raw -> _)).toMap
              if selected.size != batch.keys.length then
                throw new IllegalStateException(
                  s"exact reuse artifact missing entries for selected canonical keys: expected=${batch.keys.length}, got=${selected.size}"
                )
              val outputMeta = HeadsUpEquityTableMeta(
                formatVersion = HeadsUpEquityTableFormat.Version,
                mode = "exact",
                trials = 0,
                seed = seed,
                maxMatchups = maxMatchups,
                totalMatchups = totalMatchups,
                count = selected.size,
                canonical = true,
                createdAtMillis = System.currentTimeMillis()
              )
              HeadsUpEquityCanonicalTableIO.write(
                outputPath,
                HeadsUpEquityCanonicalTable(selected),
                outputMeta
              )
              Some(ExactReuseOutcome(count = selected.size, sourcePath = source, copiedDirectly = false))
        }

  private def minimalExpectedArtifactSize(entryCount: Int): Long =
    CanonicalHeaderBytes + CanonicalEntryBytes * entryCount.toLong

  private def maybeWriteExactGpuWithoutTableMaterialization(
      outputPath: String,
      mode: HeadsUpEquityTable.Mode,
      backend: HeadsUpEquityTable.ComputeBackend,
      seed: Long,
      maxMatchups: Long,
      totalMatchups: Long,
      progress: Option[(Long, Long) => Unit]
  ): Option[Int] =
    if mode != HeadsUpEquityTable.Mode.Exact ||
        backend != HeadsUpEquityTable.ComputeBackend.Gpu ||
        exactReuseEnabled then
      None
    else
      val batch = HeadsUpEquityCanonicalTable.selectCanonicalBatch(maxMatchups)
      val seedBase = new Random(seed).nextLong()
      HeadsUpGpuRuntime.computeBatch(
        packedKeys = batch.packedKeys,
        keyMaterial = batch.keyMaterial,
        mode = HeadsUpEquityTable.Mode.Exact,
        monteCarloSeedBase = seedBase
      ) match
        case Right(results) =>
          progress.foreach(callback => callback(batch.keys.length.toLong, totalMatchups))
          val count = results.length
          val meta = HeadsUpEquityTableMeta(
            formatVersion = HeadsUpEquityTableFormat.Version,
            mode = "exact",
            trials = 0,
            seed = seed,
            maxMatchups = maxMatchups,
            totalMatchups = totalMatchups,
            count = count,
            canonical = true,
            createdAtMillis = System.currentTimeMillis()
          )
          HeadsUpEquityCanonicalTableIO.writeFromBatch(outputPath, batch.keys, results, meta)
          Some(count)
        case Left(reason) =>
          if HeadsUpGpuRuntime.allowCpuFallbackOnGpuFailure then
            System.err.println(s"GPU backend unavailable ($reason); using CPU workers")
            None
          else
            throw new IllegalStateException(
              s"GPU backend failed ($reason). CPU fallback is disabled; set sicfun_GPU_FALLBACK_TO_CPU=true to re-enable it."
            )

  private def exactReuseEnabled: Boolean =
    val raw =
      sys.props
        .get(ExactReuseEnabledProperty)
        .orElse(sys.env.get(ExactReuseEnabledEnv))
        .map(_.trim.toLowerCase)
    raw match
      case Some("0" | "false" | "no" | "off") => false
      case _ => true

  private def resolvedExactReuseFile(): File =
    val configured =
      sys.props
        .get(ExactReusePathProperty)
        .orElse(sys.env.get(ExactReusePathEnv))
        .map(_.trim)
        .filter(_.nonEmpty)
        .getOrElse(DefaultExactReusePath)
    new File(configured)

  private def exactPrimeBeforeTimingEnabled: Boolean =
    val raw =
      sys.props
        .get(ExactPrimeBeforeTimingProperty)
        .orElse(sys.env.get(ExactPrimeBeforeTimingEnv))
        .map(_.trim.toLowerCase)
    raw match
      case Some("1" | "true" | "yes" | "on") => true
      case _ => false

  private final case class ExactReuseOutcome(
      count: Int,
      sourcePath: File,
      copiedDirectly: Boolean
  )

  private def preferredDefaultBackend(): HeadsUpEquityTable.ComputeBackend =
    val configuredProvider = GpuRuntimeSupport.resolveNonEmptyLower(ProviderProperty, ProviderEnv)
    configuredProvider match
      case Some(provider) =>
        if providerAvailability(provider).available then HeadsUpEquityTable.ComputeBackend.Gpu
        else HeadsUpEquityTable.ComputeBackend.Cpu
      case None =>
        val nativeAvailability = providerAvailability("native")
        if nativeAvailability.available then HeadsUpEquityTable.ComputeBackend.Gpu
        else
          val hybridAvailability = providerAvailability("hybrid")
          if hybridAvailability.available then HeadsUpEquityTable.ComputeBackend.Gpu
          else HeadsUpEquityTable.ComputeBackend.Cpu

  private def providerAvailability(provider: String): HeadsUpGpuRuntime.Availability =
    val previous = sys.props.get(ProviderProperty)
    sys.props.update(ProviderProperty, provider)
    try HeadsUpGpuRuntime.availability
    finally
      previous match
        case Some(value) => sys.props.update(ProviderProperty, value)
        case None => sys.props.remove(ProviderProperty)
