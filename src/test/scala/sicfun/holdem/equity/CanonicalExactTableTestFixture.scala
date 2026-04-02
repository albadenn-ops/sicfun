package sicfun.holdem.equity
import sicfun.holdem.types.*
import sicfun.holdem.tablegen.*
import sicfun.holdem.gpu.*

import java.io.File
import java.util.Locale
import scala.util.control.NonFatal

/** Ensures equity-accuracy tests run against a real canonical exact table.
  *
  * Resolution order:
  *   1) `-Dsicfun.test.canonicalExactTablePath` (must be valid/full)
  *   2) optional local artifact at the legacy `data/` path
  *   3) generated fallback in `target/test-generated/`
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

  private final case class GenerationPlan(
      backend: String,
      propertyUpdates: Vector[(String, Option[String])],
      reason: String
  )

  def load(): HeadsUpEquityCanonicalTable =
    loadWithPath()._2

  def path(): String =
    loadWithPath()._1

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
