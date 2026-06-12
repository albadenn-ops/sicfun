package sicfun.holdem.strategic

import munit.FunSuite

/** Enforcement manifest: every known reductionism, stub, proxy, and hardcoded heuristic
  * in the strategic module and its engine wiring.
  *
  * Each entry documents:
  *   - WHERE the reductionism lives (file:line)
  *   - WHAT it does (or doesn't do)
  *   - DEADLINE: which wave/version should replace it
  *   - SEVERITY: Silent (bypasses formal guarantees), Proxy (approximate), Heuristic (uncalibrated)
  *
  * Deadline strings must use either "Wave N" or "vX.Y[.Z]".
  *
  * This test suite is intentionally RED when unresolved Silent or Orphan entries
  * pass their declared deadline. It is not a bug - it is an enforcement mechanism.
  * Do not delete entries to make it green. Replace the underlying reductionism,
  * THEN remove the entry.
  */
class ReductionismManifestTest extends FunSuite:

  enum Severity:
    case Silent    // Nullifies formal guarantees without warning
    case Proxy     // Approximate implementation standing in for real one
    case Heuristic // Hardcoded magic numbers replacing calibrated models
    case Orphan    // Implemented but disconnected from engine

  case class Reductionism(
      id: String,
      file: String,
      description: String,
      severity: Severity,
      deadline: String, // "Wave N" or "vX.Y[.Z]"
      resolved: Boolean = false
  )

  case class SpecVersion(major: Int, minor: Int, patch: Int = 0) extends Ordered[SpecVersion]:
    override def compare(that: SpecVersion): Int =
      Ordering[(Int, Int, Int)].compare(
        (major, minor, patch),
        (that.major, that.minor, that.patch)
      )

    override def toString: String =
      s"v$major.$minor.$patch"

  enum Deadline:
    case Wave(number: Int)
    case Version(version: SpecVersion)

  case class StrategicProgress(completedWave: Int, currentVersion: SpecVersion)

  // Keep this in sync with the strategic layer's current post-Wave-7 / v0.31.1 state.
  private val currentProgress =
    StrategicProgress(
      completedWave = 7,
      currentVersion = SpecVersion(0, 31, 1)
    )

  private val waveDeadlinePattern = raw"Wave\s+(\d+)".r
  private val versionDeadlinePattern = raw"v(\d+)\.(\d+)(?:\.(\d+))?".r

  // -- THE MANIFEST -------------------------------------------------------------

  val manifest: Vector[Reductionism] = Vector(
    // -- Tier 1: Silent nullifications in StrategicEngine --
    Reductionism("SE-001", "engine/StrategicEngine.scala:91",
      "exploitabilityFn = _ => 0.0 - entire safety apparatus inert",
      Severity.Silent, "Wave 4", resolved = true),
    Reductionism("SE-002", "engine/StrategicEngine.scala:90",
      "detector = NeverDetect - opponent detection permanently disabled",
      Severity.Silent, "Wave 4", resolved = true),
    Reductionism("SE-003", "engine/StrategicEngine.scala:259",
      "ShowdownKernel classifies revealed hands and hard-shifts posterior",
      Severity.Silent, "Wave 3", resolved = true),
    Reductionism("SE-004", "engine/StrategicEngine.scala:154",
      "endHand applies ShowdownKernel to update rival beliefs",
      Severity.Silent, "Wave 3", resolved = true),
    Reductionism("SE-005", "engine/StrategicEngine.scala:93",
      "showdown = None in observeAction is correct - showdowns come via endHand",
      Severity.Silent, "Wave 3", resolved = true),

    // -- Tier 2: Sentinel/dummy objects --
    Reductionism("KC-001", "strategic/KernelConstructor.scala:29",
      "sentinelHero = PlayerId(\"__sentinel__\") used in production kernel path",
      Severity.Proxy, "Wave 2", resolved = true),
    Reductionism("KC-002", "strategic/KernelConstructor.scala:37",
      "dummyPublicState fabricated for kernel updates instead of real game state",
      Severity.Proxy, "Wave 2", resolved = true),
    Reductionism("EI-001", "strategic/ExploitationInterpolation.scala:92",
      "sentinelHero + fabricated PublicState duplicated from KernelConstructor",
      Severity.Proxy, "Wave 2", resolved = true),

    // -- Tier 3: Hardcoded heuristic tables --
    Reductionism("SE-006", "engine/StrategicEngine.scala:185",
      "actionPrior: 16 hardcoded P(action|class) magic numbers, not calibrated - injectable via Config.actionPriors",
      Severity.Heuristic, "Wave 5", resolved = true),
    Reductionism("SE-007", "engine/StrategicEngine.scala:147",
      "Position-to-bucket fallback: hardcoded default bucket - injectable via Config.defaultHeroBucket",
      Severity.Heuristic, "Wave 5", resolved = true),
    Reductionism("SE-008", "engine/StrategicEngine.scala:209",
      "Uniform 0.25 fallback prior when belief type doesn't match",
      Severity.Heuristic, "Wave 2", resolved = true),
    Reductionism("SE-009", "engine/StrategicEngine.scala:166",
      "actionHistory = Vector.empty - kernel pipeline never sees action sequences",
      Severity.Silent, "Wave 3", resolved = true),
    Reductionism("SE-010", "engine/StrategicEngine.scala:155",
      "PublicState is hero-only - all rival seats missing",
      Severity.Silent, "Wave 3", resolved = true),
    Reductionism("PF-001", "engine/PokerPomcpFormulation.scala:110",
      "Call uses actual toCallChips amount - injectable via buildActionEffects parameter",
      Severity.Heuristic, "Wave 5", resolved = true),
    Reductionism("PF-002", "engine/PokerPomcpFormulation.scala:140",
      "Showdown equity = linear heuristic 0.5 + diff*0.4 - injectable via showdownEquityFn parameter",
      Severity.Heuristic, "Wave 5", resolved = true),
    Reductionism("PF-003", "engine/PokerPomcpFormulation.scala:63",
      "classPriors: 12 hardcoded class-action probabilities - exposed as defaultClassPriors val",
      Severity.Heuristic, "Wave 5", resolved = true),
    Reductionism("GE-001", "engine/GtoSolveEngine.scala:293",
      "12 hardcoded street-dependent GTO thresholds - injectable via GtoThresholds config class",
      Severity.Heuristic, "Wave 6", resolved = true),
    Reductionism("HS-001", "engine/HandStrengthEstimator.scala:210",
      "Hardcoded street-dependent blend weights (0.50/0.56/0.62) - injectable via blendWeights parameter",
      Severity.Heuristic, "Wave 6", resolved = true),
    Reductionism("PFe-001", "holdem/model/PokerFeatures.scala:107",
      "Preflop equity = 0.5 for all hands when board.size < 3",
      Severity.Heuristic, "Wave 5", resolved = true),

    // -- Tier 4: Proxy/stub implementations --
    Reductionism("SP-001", "strategic/SpotPolarization.scala:60",
      "UniformPolarization returns 0.5 for all sizings (documented stub)",
      Severity.Proxy, "Wave 2", resolved = true),
    Reductionism("SP-002", "strategic/SpotPolarization.scala:79",
      "PosteriorDivergencePolarization is sizing-extremity proxy, NOT KL divergence",
      Severity.Proxy, "Wave 2", resolved = true),
    Reductionism("RK-001", "strategic/RivalKernel.scala:20",
      "KernelVariant.Design is a placeholder - no concrete implementation",
      Severity.Proxy, "Wave 2", resolved = true),
    Reductionism("SRB-001", "strategic/StrategicRivalBelief.scala:14",
      "update() returns this - identity pass-through, real update in kernel pipeline",
      Severity.Proxy, "Wave 2", resolved = true),
    Reductionism("BF-001", "strategic/BluffFramework.scala:21",
      "Feasibility check is identity pass-through - all actions always feasible",
      Severity.Proxy, "Wave 3", resolved = true),

    // -- Tier 5: Bridge fidelity gaps (from BridgeManifest) --
    Reductionism("BM-001", "strategic/bridge/BridgeManifest.scala",
      "ActionSignal.timing = Fidelity.Absent - requires timing infrastructure (acknowledged limitation)",
      Severity.Proxy, "Wave 6", resolved = true),
    Reductionism("BM-002", "strategic/bridge/BridgeManifest.scala",
      "TableMap = Fidelity.Absent - GameState is hero-only",
      Severity.Silent, "Wave 3", resolved = true),
    Reductionism("BM-003", "strategic/bridge/BridgeManifest.scala",
      "ClassPosterior = heuristic from VPIP/PFR/AF, not Bayesian",
      Severity.Proxy, "Wave 4", resolved = true),
    Reductionism("BM-004", "strategic/bridge/BridgeManifest.scala",
      "FourWorld.V10, V01 = interpolated estimates, not POMDP-derived",
      Severity.Proxy, "Wave 4", resolved = true),

    // -- Tier 6: Orphaned formal objects --
    Reductionism("OR-001", "strategic/SafetyBellman.scala",
      "SafetyBellman (Defs 58-66) - implemented but never called from engine",
      Severity.Orphan, "Wave 4", resolved = true),
    Reductionism("OR-002", "strategic/RiskDecomposition.scala",
      "RiskDecomposition (Defs 56A-C) - reachable via DecisionDiagnostics framework",
      Severity.Orphan, "Wave 5", resolved = true),
    Reductionism("OR-003", "strategic/Exploitability.scala",
      "SecurityValue, PointwiseExploitability, DeploymentExploitability - orphaned",
      Severity.Orphan, "Wave 4", resolved = true),
    Reductionism("OR-004", "strategic/Exploitability.scala",
      "RobustValueOracle / WassersteinRobustOracle - never invoked from engine",
      Severity.Orphan, "Wave 4", resolved = true),
    Reductionism("OR-005", "strategic/FourWorldDecomposition.scala",
      "FourWorldDecomposition (Theorem 4) - reachable via DecisionDiagnostics framework",
      Severity.Orphan, "Wave 5", resolved = true),
    Reductionism("OR-006", "strategic/solver/PftDpwRuntime.scala",
      "PftDpwRuntime - reachable as SolverBackend.PftDpw in StrategicEngine",
      Severity.Orphan, "Wave 5", resolved = true),
    Reductionism("OR-007", "strategic/solver/WassersteinDroRuntime.scala",
      "WassersteinDroRuntime - EMD + simplex LP, never invoked from engine",
      Severity.Orphan, "Wave 4", resolved = true),
  )

  private def parseDeadline(raw: String): Deadline =
    raw.trim match
      case waveDeadlinePattern(number) =>
        Deadline.Wave(number.toInt)
      case versionDeadlinePattern(major, minor, patch) =>
        Deadline.Version(
          SpecVersion(
            major = major.toInt,
            minor = minor.toInt,
            patch = Option(patch).filter(_.nonEmpty).map(_.toInt).getOrElse(0)
          )
        )
      case other =>
        fail(s"Unsupported deadline format '$other'. Use 'Wave N' or 'vX.Y[.Z]'.")

  private def isOverdue(deadline: Deadline, progress: StrategicProgress = currentProgress): Boolean =
    deadline match
      case Deadline.Wave(number) =>
        number <= progress.completedWave
      case Deadline.Version(version) =>
        version <= progress.currentVersion

  private def overdueEntries(entries: Vector[Reductionism]): Vector[Reductionism] =
    entries.filter(entry => isOverdue(parseDeadline(entry.deadline)))

  private def renderEntries(entries: Vector[Reductionism]): String =
    entries.map { entry =>
      s"[${entry.id}] ${entry.file}: ${entry.description} (deadline: ${entry.deadline})"
    }.mkString("\n")

  private def assertNoOverdue(entries: Vector[Reductionism], label: String): Unit =
    val overdueIds = overdueEntries(entries).map(_.id).toSet
    if entries.nonEmpty then
      println(s"\n=== ${entries.size} $label (${overdueIds.size} overdue) ===")
      entries.foreach { entry =>
        val status = if overdueIds.contains(entry.id) then "OVERDUE" else "pending"
        println(s"  [${entry.id}] ${entry.file}: ${entry.description} (deadline: ${entry.deadline}, $status)")
      }
    val overdue = entries.filter(entry => overdueIds.contains(entry.id))
    assert(
      overdue.isEmpty,
      s"${overdue.size} overdue $label remain:\n${renderEntries(overdue)}"
    )

  // -- ENFORCEMENT TESTS --------------------------------------------------------

  test("manifest is non-empty"):
    assert(manifest.nonEmpty, "Manifest must contain at least one entry")

  test("no duplicate IDs"):
    val ids = manifest.map(_.id)
    val dupes = ids.diff(ids.distinct)
    assertEquals(dupes, Vector.empty, s"Duplicate IDs found: ${dupes.mkString(", ")}")

  test("all entries have non-empty fields"):
    manifest.foreach { r =>
      assert(r.id.nonEmpty, s"Empty ID in manifest")
      assert(r.file.nonEmpty, s"${r.id}: empty file")
      assert(r.description.nonEmpty, s"${r.id}: empty description")
      assert(r.deadline.nonEmpty, s"${r.id}: empty deadline")
    }

  test("all deadlines use supported formats"):
    manifest.foreach { r =>
      parseDeadline(r.deadline)
    }
    assertEquals(parseDeadline("Wave 7"), Deadline.Wave(7))
    assertEquals(parseDeadline("v0.31.1"), Deadline.Version(SpecVersion(0, 31, 1)))

  test("manifest summary"):
    val unresolved = manifest.filterNot(_.resolved)
    val bySeverity = unresolved.groupBy(_.severity)
    val summary = bySeverity.map { (sev, entries) =>
      s"  $sev: ${entries.size}"
    }.mkString("\n")
    val total = unresolved.size
    val counted = bySeverity.values.map(_.size).sum
    val overdue = overdueEntries(unresolved)
    println(s"\n=== REDUCTIONISM MANIFEST: $total unresolved ===")
    println(summary)
    println(s"  Resolved: ${manifest.count(_.resolved)}")
    println(s"  Overdue: ${overdue.size}")
    println()
    assertEquals(counted, total, s"Unresolved summary mismatch: expected $total, counted $counted")

  test("overdue unresolved silent reductionisms fail"):
    val silent = manifest.filter(r => r.severity == Severity.Silent && !r.resolved)
    assertNoOverdue(silent, "silent reductionisms (formal guarantees bypassed)")

  test("overdue unresolved orphaned formal objects fail"):
    val orphans = manifest.filter(r => r.severity == Severity.Orphan && !r.resolved)
    assertNoOverdue(orphans, "orphaned formal objects (implemented but unwired)")

  test("gate trips on an overdue unresolved fixture entry"):
    val fixture = Vector(
      Reductionism(
        id = "TEST-001",
        file = "test/Fixture.scala:1",
        description = "fixture overdue entry",
        severity = Severity.Silent,
        deadline = "Wave 1",
        resolved = false
      )
    )
    val err = intercept[AssertionError] {
      assertNoOverdue(fixture, "silent reductionisms (formal guarantees bypassed)")
    }
    assert(
      Option(err.getMessage).exists(_.contains("TEST-001")),
      "Fixture gate failure should identify the overdue entry"
    )
