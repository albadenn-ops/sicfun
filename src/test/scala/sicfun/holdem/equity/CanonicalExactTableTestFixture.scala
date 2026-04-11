package sicfun.holdem.equity
import sicfun.holdem.types.*
import sicfun.holdem.tablegen.*
import sicfun.holdem.gpu.*

import java.io.File
import java.util.Locale
import scala.util.control.NonFatal

/**
  * Test fixture that ensures equity-accuracy tests run against a real canonical exact table.
  *
  * This is a shared, thread-safe, lazily-initialized fixture that provides a fully computed
  * [[HeadsUpEquityCanonicalTable]] for use across multiple test suites. It handles the
  * complexity of locating or generating the table artifact, which is expensive to compute
  * (~170K canonical matchups via exact enumeration).
  *
  * '''Resolution order:'''
  *   1. System property `-Dsicfun.test.canonicalExactTablePath` (must point to a valid full table)
  *   2. Legacy local artifact at `data/heads-up-equity-canonical-exact-cuda-full.bin`
  *   3. Generated fallback at `target/test-generated/heads-up-equity-canonical-exact.bin`
  *
  * '''Generation strategy (when no artifact exists):'''
  *   - Probes GPU providers (native, then hybrid) for availability
  *   - Falls back to CPU-only generation if no GPU is available
  *   - Enforces non-fallback GPU execution unless user explicitly opts in
  *
  * The fixture validates that the loaded table is both canonical and exact-mode before caching.
  *
  * @see [[EquityAccuracyTest]] which is the primary consumer of this fixture
  */
object CanonicalExactTableTestFixture:
  private val TablePathProperty = "sicfun.test.canonicalExactTablePath"
  private val DefaultArtifactPath = "data/heads-up-equity-canonical-exact-cuda-full.bin"
  private val GeneratedArtifactPath = "target/test-generated/heads-up-equity-canonical-exact.bin"
  private val ProviderProperty = "sicfun.gpu.provider"
  private val ProviderEnv = "sicfun_GPU_PROVIDER"
  private val FallbackToCpuProperty = "sicfun.gpu.fallbackToCpu"
  private val FallbackToCpuEnv = "sicfun_GPU_FALLBACK_TO_CPU"

  private val lock = new AnyRef
  @volatile private var cached: Option[(String, HeadsUpEquityCanonicalTable)] = None

  /** Describes how to generate the canonical exact table: which backend to use,
    * which system properties to set during generation, and a human-readable reason.
    */
  private final case class GenerationPlan(
      backend: String,
      propertyUpdates: Vector[(String, Option[String])],
      reason: String
  )

  /** Returns the cached canonical exact table, loading/generating it on first call. */
  def load(): HeadsUpEquityCanonicalTable =
    loadWithPath()._2

  /** Returns the file path of the canonical exact table artifact. */
  def path(): String =
    loadWithPath()._1

  /** Double-checked locking to ensure the table is loaded exactly once across all test threads. */
  private def loadWithPath(): (String, HeadsUpEquityCanonicalTable) =
    cached match
      case Some(value) => value
      case None =>
        lock.synchronized {
          cached match
            case Some(value) => value
            case None =>
              val tablePath = ensurePath()
              val (table, meta) = HeadsUpEquityCanonicalTableIO.readWithMeta(tablePath)
              require(
                meta.canonical && meta.mode.trim.equalsIgnoreCase("exact"),
                s"expected canonical exact table at $tablePath (canonical=${meta.canonical}, mode=${meta.mode})"
              )
              val loaded = (tablePath, table)
              cached = Some(loaded)
              loaded
        }

  /** Resolves the table path following the priority order, generating if necessary. */
  private def ensurePath(): String =
    sys.props.get(TablePathProperty).map(_.trim).filter(_.nonEmpty) match
      case Some(configuredPath) =>
        validateOrThrow(configuredPath, source = s"-D$TablePathProperty")
        configuredPath
      case None =>
        val defaultArtifact = new File(DefaultArtifactPath)
        if isFullCanonicalExact(defaultArtifact) then
          val path = defaultArtifact.getAbsolutePath
          sys.props.update(TablePathProperty, path)
          path
        else
          val generatedArtifact = new File(GeneratedArtifactPath)
          if !isFullCanonicalExact(generatedArtifact) then
            generate(generatedArtifact)
          val path = generatedArtifact.getAbsolutePath
          validateOrThrow(path, source = "generated canonical exact table")
          sys.props.update(TablePathProperty, path)
          path

  /** Generates a full canonical exact table using the best available backend. */
  private def generate(output: File): Unit =
    val parent = output.getParentFile
    if parent != null then parent.mkdirs()
    val fullCount = HeadsUpEquityCanonicalTable.totalCanonicalKeys.toLong
    val plan = generationPlan()
    System.err.println(
      s"canonical-exact-fixture: generating with backend=${plan.backend} (${plan.reason})"
    )
    TestSystemPropertyScope.withSystemProperties(plan.propertyUpdates) {
      GenerateHeadsUpCanonicalTable.main(
        Array(
          output.getAbsolutePath,
          "exact",
          "0",
          fullCount.toString,
          "1",
          math.max(1, Runtime.getRuntime.availableProcessors()).toString,
          plan.backend
        )
      )
    }

  /** Determines the optimal generation strategy based on available GPU providers.
    * Checks user-configured provider first, then probes native and hybrid in order.
    */
  private def generationPlan(): GenerationPlan =
    val userProvider = configuredProvider
    userProvider match
      case Some(provider) =>
        val availability = providerAvailability(provider)
        if availability.available then
          GenerationPlan(
            backend = "gpu",
            propertyUpdates = fallbackPolicyUpdates,
            reason = s"user-configured provider '$provider' available (${availability.detail})"
          )
        else
          GenerationPlan(
            backend = "cpu",
            propertyUpdates = Vector.empty,
            reason = s"user-configured provider '$provider' unavailable (${availability.detail}); JVM is the only option"
          )
      case None =>
        val nativeAvailability = providerAvailability("native")
        if nativeAvailability.available then
          GenerationPlan(
            backend = "gpu",
            propertyUpdates = withProvider("native") ++ fallbackPolicyUpdates,
            reason = s"auto-selected native provider (${nativeAvailability.detail})"
          )
        else
          val hybridAvailability = providerAvailability("hybrid")
          if hybridAvailability.available then
            GenerationPlan(
              backend = "gpu",
              propertyUpdates = withProvider("hybrid") ++ fallbackPolicyUpdates,
              reason = s"auto-selected hybrid provider (${hybridAvailability.detail})"
            )
          else
            GenerationPlan(
              backend = "cpu",
              propertyUpdates = Vector.empty,
              reason =
                s"native/hybrid unavailable (native=${nativeAvailability.detail}; hybrid=${hybridAvailability.detail}); JVM is the only option"
            )

  private def configuredProvider: Option[String] =
    sys.props
      .get(ProviderProperty)
      .orElse(sys.env.get(ProviderEnv))
      .map(_.trim.toLowerCase(Locale.ROOT))
      .filter(_.nonEmpty)

  private def providerAvailability(provider: String): HeadsUpGpuRuntime.Availability =
    TestSystemPropertyScope.withSystemProperties(
      Vector(ProviderProperty -> Some(provider))
    ) {
      HeadsUpGpuRuntime.availability
    }

  private def withProvider(provider: String): Vector[(String, Option[String])] =
    Vector(ProviderProperty -> Some(provider))

  private def fallbackPolicyUpdates: Vector[(String, Option[String])] =
    if sys.props.contains(FallbackToCpuProperty) || sys.env.contains(FallbackToCpuEnv) then
      Vector.empty
    else
      // Enforce non-JVM execution for GPU backend unless user explicitly allows fallback.
      Vector(FallbackToCpuProperty -> Some("false"))

  private def validateOrThrow(path: String, source: String): Unit =
    val file = new File(path)
    if !file.isFile then
      throw new IllegalStateException(s"$source points to a missing file: $path")
    if !isFullCanonicalExact(file) then
      throw new IllegalStateException(s"$source is not a full canonical exact table: $path")

  /** Checks if a file is a valid, complete canonical exact table by reading its header metadata. */
  private def isFullCanonicalExact(file: File): Boolean =
    if !file.isFile then false
    else
      try
        val meta = HeadsUpEquityCanonicalTableIO.readMeta(file.getAbsolutePath)
        val fullCount = HeadsUpEquityCanonicalTable.totalCanonicalKeys
        meta.canonical &&
        meta.mode.trim.equalsIgnoreCase("exact") &&
        meta.totalMatchups >= fullCount &&
        meta.count >= fullCount
      catch
        case NonFatal(_) => false
