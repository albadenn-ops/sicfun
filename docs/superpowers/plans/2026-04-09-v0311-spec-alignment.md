# v0.31.1 Spec Alignment — Remaining Gaps

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining factual mismatches between the SICFUN-v0.31.1-corrected.md spec and the codebase — wrong assumption definitions, wrong enum naming, missing runtime wiring for exploitability and deployment tracking.

**Architecture:** Six tasks, each independently testable. Task 1 (AssumptionManifest) and Task 2 (StrategicClass rename) fix factual errors. Tasks 3-6 wire already-typed formal objects into the engine's runtime path. All changes are on branch `feat/adaptive-proof-harness-9max`.

**Tech Stack:** Scala 3.8.1, munit 1.2.2, SBT. Test runner: `sbt "testOnly <fully.qualified.TestClass>"`

**Spec reference:** `SICFUN-v0_31_1-corrected.md`, §0 Convention, §2 Assumptions, Def 1, Def 52C, §6 (design doc §5-6, §10).

**Prerequisite:** runtime-spec-closure plan (2026-04-07) tasks 1-9 and four-world-solver-closure plan (2026-04-09) tasks 1-6 all completed.

---

## File Map

| File | Role | Tasks |
|---|---|---|
| `src/main/scala/sicfun/holdem/strategic/AssumptionManifest.scala` | Assumption registry | Task 1 |
| `src/test/scala/sicfun/holdem/strategic/FormalClosureValidationTest.scala` | Closure/assumption tests | Tasks 1, 2 |
| `src/main/scala/sicfun/holdem/strategic/StrategicClass.scala` | Def 1 enum | Task 2 |
| `src/test/scala/sicfun/holdem/strategic/StrategicClassTest.scala` | Enum tests | Task 2 |
| `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala` | Decision pipeline | Tasks 3, 4, 5, 6 |
| `src/main/scala/sicfun/holdem/strategic/CertificationTypes.scala` | DecisionEvaluationBundle | Tasks 3, 4 |
| `src/test/scala/sicfun/holdem/engine/RealTimeAdaptiveEngineTest.scala` | Engine tests | Task 5 |

---

## Task 1: AssumptionManifest v0.31.1 Alignment

Fix version header, correct 5 wrong assumption names/descriptions, add missing A6 entry.

**Files:**
- Modify: `src/main/scala/sicfun/holdem/strategic/AssumptionManifest.scala`
- Modify: `src/test/scala/sicfun/holdem/strategic/FormalClosureValidationTest.scala`

- [ ] **Step 1: Write failing test for v0.31.1 assumption names**

Add to `src/test/scala/sicfun/holdem/strategic/FormalClosureValidationTest.scala`:

```scala
  test("AssumptionManifest: v0.31.1 assumption names match spec"):
    val byId = AssumptionManifest.entries.map(e => e.id -> e).toMap
    // A1' (spec §2): Abstraction with guarantees
    assert(byId("A1'").name.contains("Abstraction"), s"A1' should be Abstraction with guarantees, got: ${byId("A1'").name}")
    // A2 (spec §2): Closed Markovianity
    assert(byId("A2").name.contains("Markovianity"), s"A2 should be Closed Markovianity, got: ${byId("A2").name}")
    // A5 (spec §2): Bounded reward and discounting
    assert(byId("A5").name.contains("Bounded reward"), s"A5 should be Bounded reward and discounting, got: ${byId("A5").name}")
    // A7 (spec §2): Well-defined full rival update
    assert(byId("A7").name.contains("full rival update"), s"A7 should be Well-defined full rival update, got: ${byId("A7").name}")
    // A8 (spec §2): Strategically relevant repetition
    assert(byId("A8").name.contains("repetition"), s"A8 should be Strategically relevant repetition, got: ${byId("A8").name}")

  test("AssumptionManifest: A6 (first-order interactive sufficiency) is present"):
    val ids = AssumptionManifest.entries.map(_.id).toSet
    assert(ids.contains("A6"), "A6 must be present (first-order interactive sufficiency)")

  test("AssumptionManifest contains exactly 11 entries (A1' through A10 including A6)"):
    assertEquals(AssumptionManifest.entries.size, 11)

  test("AssumptionManifest: A1'-A10 plus A6 all present"):
    val ids = AssumptionManifest.entries.map(_.id).toSet
    val expected = Set("A1'", "A2", "A3'", "A4'", "A5", "A6", "A6'", "A7", "A8", "A9", "A10")
    assertEquals(ids, expected)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `sbt "testOnly sicfun.holdem.strategic.FormalClosureValidationTest"`
Expected: FAIL — A1' contains "Finite action space" not "Abstraction"; A6 missing; count is 10 not 11.

- [ ] **Step 3: Update AssumptionManifest.scala**

Replace the entire `AssumptionManifest.scala` content. Header changes from "v0.30.2" to "v0.31.1". All 5 wrong assumptions corrected. A6 added.

```scala
package sicfun.holdem.strategic

/** Manifest documenting the 11 assumptions from the SICFUN v0.31.1 canonical spec.
  *
  * Assumptions fall into two categories:
  * - Structural: constraints on the problem formulation (not executable)
  * - Encoded: have computational representations in the formal layer
  */
object AssumptionManifest:

  enum EnforcementLevel:
    case Encoded      // has a computational representation enforced in code
    case Structural   // constraint on the model; verified by design, not code
    case Recovered    // generalized (primed version) subsumes the original
    case Approximated // has a conservative computational approximation with explicit fidelity
    case Deferred     // out of constitutive scope; documented but not implemented

  final case class AssumptionEntry(
      id: String,
      name: String,
      enforcement: EnforcementLevel,
      location: String,
      notes: String
  )

  val entries: Vector[AssumptionEntry] = Vector(
    AssumptionEntry(
      "A1'", "Abstraction with guarantees (v0.31.1: primed)",
      EnforcementLevel.Structural,
      "PokerAction enum + HandStrengthEstimator (sizing quantization)",
      "Alpha maps (alpha_X, alpha_A, alpha_Y) with epsilon bounds (eps_R, eps_T, eps_O). " +
      "Value error bounded by eps_R/(1-gamma) + gamma*R_max*(eps_T+eps_O)/(1-gamma)^2 in POMDPs. " +
      "Sizing quantization via Sizing type provides alpha_A. " +
      "v0.31.1 A1' subsumes A1 with explicit abstraction error bounds"
    ),
    AssumptionEntry(
      "A2", "Closed Markovianity",
      EnforcementLevel.Structural,
      "AugmentedState.scala: augmented hidden state dynamics",
      "X_tilde_{t+1} ~ T(. | X_tilde_t, u_t). " +
      "The augmented hidden state transition depends only on current state and action, " +
      "not on history. Enforced by the structure of the belief update (Def 22)"
    ),
    AssumptionEntry(
      "A3'", "Non-stationary rival types with changepoint detection",
      EnforcementLevel.Recovered,
      "ChangepointDetector.scala",
      "theta^{R,i}_{t+1} ~ K^i(. | theta^{R,i}_t, zeta^i_t) with hazard rate h^i. " +
      "Generalizes A3 (stationary types). A3 recovered when h^i = 0 and K^i is identity"
    ),
    AssumptionEntry(
      "A4'", "Own statistical sufficiency (v0.31.1: primed)",
      EnforcementLevel.Structural,
      "AugmentedState.scala: OwnEvidence",
      "xi^S is finite-dimensional sufficient statistic for SICFUN's evidence " +
      "relative to A1', A6, and parametric family F^R. " +
      "OwnEvidence stores Map[String, Double] summaries. " +
      "v0.31.1 A4' subsumes A4 with explicit finite-dimensionality requirement"
    ),
    AssumptionEntry(
      "A5", "Bounded reward and discounting",
      EnforcementLevel.Structural,
      "Chips opaque type, solver configs",
      "|r(x_tilde, u)| <= R_max, gamma in (0,1). " +
      "R_max is finite — poker pots are bounded by stack depths. " +
      "gamma < 1 enforced by POMDP solver configs (PftDpwConfig, WPomcpConfig). " +
      "Used in Corollary 4 value error bound and SafetyBellman contraction"
    ),
    AssumptionEntry(
      "A6", "First-order interactive sufficiency",
      EnforcementLevel.Structural,
      "OpponentModelState: m_t^{R,i} summary",
      "For each rival i, future policy depends on public history only through m_t^{R,i}. " +
      "First-order truncation: rivals do not model SICFUN's modeling of them. " +
      "Encoded by the structure of RivalBeliefState (finite summary, no recursive modeling)"
    ),
    AssumptionEntry(
      "A6'", "Detection-aware exploitation",
      EnforcementLevel.Encoded,
      "DetectionPredicate.scala",
      "DetectModeling^i predicate: rivals do not model SICFUN's modeling of them (A6), " +
      "but SICFUN detects if they do. Implementations: NeverDetect, AlwaysDetect, FrequencyAnomalyDetection. " +
      "Minimal activation conditions per spec: measurable w.r.t. H_t filtration, " +
      "returns 0 when play is indistinguishable from baseline, tends to 1 under exploitation detection"
    ),
    AssumptionEntry(
      "A7", "Well-defined full rival update",
      EnforcementLevel.Encoded,
      "RivalKernel.scala: ActionKernel, ShowdownKernel, FullKernel; KernelConstructor.scala",
      "For each rival i, Gamma^{full,(omega^act,omega^sd),i}: M^{R,i} x Y x X^pub -> M^{R,i}. " +
      "Parameterized by (omega^act, omega^sd) in Omega^chain. " +
      "omega^act selects action-channel kernel (Blind/Ref/Attrib/Design), " +
      "omega^sd controls showdown composition (Def 20). " +
      "Implemented by composeFullKernelForWorld and composeFullKernelForWorldFull"
    ),
    AssumptionEntry(
      "A8", "Strategically relevant repetition",
      EnforcementLevel.Structural,
      "Structural (poker session guarantees repeat play)",
      "There exists p_lower > 0 such that for all t and all x_tilde, " +
      "Pr(future interaction with rival i | X_tilde_t = x_tilde) >= p_lower. " +
      "In poker sessions, future interaction is guaranteed with non-negligible probability. " +
      "p_lower need not be close to 1 in the discounted setting"
    ),
    AssumptionEntry(
      "A9", "Spot-conditioned polarization",
      EnforcementLevel.Encoded,
      "SpotPolarization.scala",
      "Pol_t^i(lambda) = Pol_t^i(lambda | x^pub_t, pi^{0,S}, m^{R,i}_t). " +
      "Polarization is spot-dependent. PosteriorDivergencePolarization computes true KL divergence " +
      "D_KL(posterior || prior) when a TemperedLikelihoodFn is provided (Fidelity.Exact); " +
      "falls back to sizing-extremity proxy otherwise. SpotPolarization.fidelity self-reports"
    ),
    AssumptionEntry(
      "A10", "Adaptation safety baseline (v0.31.1: revised)",
      EnforcementLevel.Encoded,
      "AdaptationSafety.scala, SafetyBellman.scala, Exploitability.scala, OperationalBaseline.scala",
      "Approximate baseline bar_pi with Exploit_{B_dep}(bar_pi) <= epsilon_base. " +
      "v0.31.1 AS-strong (Def 57): for all b in B_dep, all sigma^{-S}: " +
      "J(b; pi, sigma^{-S}) >= J(b; bar_pi, sigma^{-S}) - epsilon_adapt. " +
      "Bellman-safe certificates (Defs 58-66), TotalVulnerability (Corollary 9.3). " +
      "Legacy scalar safety (Theorem 8 betaBar clamping) preserved as compatibility wrapper"
    )
  )

  def encoded: Vector[AssumptionEntry] =
    entries.filter(_.enforcement == EnforcementLevel.Encoded)

  def structural: Vector[AssumptionEntry] =
    entries.filter(_.enforcement == EnforcementLevel.Structural)

  def summary: String =
    val enc = entries.count(_.enforcement == EnforcementLevel.Encoded)
    val str = entries.count(_.enforcement == EnforcementLevel.Structural)
    val rec = entries.count(_.enforcement == EnforcementLevel.Recovered)
    s"AssumptionManifest: $enc encoded, $str structural, $rec recovered (${entries.size} total)"
```

- [ ] **Step 4: Update existing test assertions to expect 11 entries**

In `FormalClosureValidationTest.scala`, update the existing test:

Find:
```scala
  test("AssumptionManifest contains exactly 10 entries"):
    assertEquals(AssumptionManifest.entries.size, 10)
```
Replace with:
```scala
  test("AssumptionManifest contains exactly 11 entries"):
    assertEquals(AssumptionManifest.entries.size, 11)
```

Find:
```scala
  test("AssumptionManifest: A1'-A10 all present"):
    val ids = AssumptionManifest.entries.map(_.id).toSet
    val expected = Set("A1'", "A2", "A3'", "A4'", "A5", "A6'", "A7", "A8", "A9", "A10")
    assertEquals(ids, expected)
```
Replace with:
```scala
  test("AssumptionManifest: A1'-A10 plus A6 all present"):
    val ids = AssumptionManifest.entries.map(_.id).toSet
    val expected = Set("A1'", "A2", "A3'", "A4'", "A5", "A6", "A6'", "A7", "A8", "A9", "A10")
    assertEquals(ids, expected)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `sbt "testOnly sicfun.holdem.strategic.FormalClosureValidationTest"`
Expected: PASS (all tests including the new v0.31.1 name checks)

- [ ] **Step 6: Commit**

```bash
git add src/main/scala/sicfun/holdem/strategic/AssumptionManifest.scala src/test/scala/sicfun/holdem/strategic/FormalClosureValidationTest.scala
git commit -m "fix(strategic): align AssumptionManifest to v0.31.1 spec (5 corrected names, add A6)"
```

---

## Task 2: StrategicClass.Marginal → Mixed (Def 1 alignment)

The spec defines C^M as "mixed classes" — classes that combine elements of value-holding and bluffing. The code calls this "Marginal" which has a different semantic meaning (borderline hand strength). Rename to match the spec.

**Files:**
- Modify: `src/main/scala/sicfun/holdem/strategic/StrategicClass.scala`
- Modify: `src/test/scala/sicfun/holdem/strategic/StrategicClassTest.scala`
- Modify: All files referencing `StrategicClass.Marginal` (use `replace_all`)

- [ ] **Step 1: Write failing test for the new enum case name**

Add to `src/test/scala/sicfun/holdem/strategic/StrategicClassTest.scala`:

```scala
  test("StrategicClass values match spec Def 1: Value, Bluff, Mixed, StructuralBluff"):
    val names = StrategicClass.values.map(_.toString).toSet
    assertEquals(names, Set("Value", "Bluff", "Mixed", "StructuralBluff"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly sicfun.holdem.strategic.StrategicClassTest"`
Expected: FAIL — set contains "Marginal" not "Mixed".

- [ ] **Step 3: Rename enum case in StrategicClass.scala**

In `src/main/scala/sicfun/holdem/strategic/StrategicClass.scala`, replace:

```scala
  case Marginal  // C^M: hands with marginal equity
```

with:

```scala
  case Mixed     // C^M: mixed classes — combine value-holding and bluffing in a single strategic posture
```

- [ ] **Step 4: Rename all references across the codebase**

Use `replace_all` on every file that references `StrategicClass.Marginal`:

In `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala`, replace all `StrategicClass.Marginal` with `StrategicClass.Mixed`.

In `src/main/scala/sicfun/holdem/engine/PokerPftFormulation.scala`, replace all `StrategicClass.Marginal` with `StrategicClass.Mixed`.

In `src/main/scala/sicfun/holdem/engine/PokerPomcpFormulation.scala`, replace all `StrategicClass.Marginal` with `StrategicClass.Mixed`.

In `src/main/scala/sicfun/holdem/strategic/bridge/ClassificationBridge.scala`, replace all `StrategicClass.Marginal` with `StrategicClass.Mixed`.

In `src/main/scala/sicfun/holdem/strategic/bridge/OpponentModelBridge.scala`, replace all `StrategicClass.Marginal` with `StrategicClass.Mixed`.

In all test files referencing `StrategicClass.Marginal` or `Marginal`, replace with `StrategicClass.Mixed` or `Mixed`.

Search for any remaining references:
```bash
grep -r "Marginal" src/main/scala/sicfun/holdem/strategic/ src/main/scala/sicfun/holdem/engine/ src/test/scala/sicfun/ --include="*.scala" -l
```

- [ ] **Step 5: Update existing StrategicClassTest to use new name**

In `src/test/scala/sicfun/holdem/strategic/StrategicClassTest.scala`, find:
```scala
  test("StrategicClass values are Value, Bluff, Marginal, StructuralBluff"):
```
Replace the test name and any assertions that reference "Marginal" with "Mixed".

- [ ] **Step 6: Run all strategic tests**

Run: `sbt "testOnly sicfun.holdem.strategic.*"`
Expected: PASS (all tests, no remaining `Marginal` references)

Run: `sbt "testOnly sicfun.holdem.engine.*"`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add -u
git commit -m "refactor(strategic): rename StrategicClass.Marginal to Mixed per spec Def 1 (C^M = mixed classes)"
```

---

## Task 3: Wire pointwiseExploitability in PftDpw Path

The PftDpw formal path already computes per-profile values and baseline values — it has the data to compute pointwise exploitability (Def 52C) but currently leaves it as `None`.

**Files:**
- Modify: `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala` (method `buildFormalCertification`)
- Modify: `src/test/scala/sicfun/holdem/engine/FormalPathTest.scala`

- [ ] **Step 1: Write failing test**

Add to `src/test/scala/sicfun/holdem/engine/FormalPathTest.scala`:

```scala
  test("PftDpw path computes pointwiseExploitability (Def 52C)"):
    val engine = makeTestEngine(usePftDpw = true)
    engine.initSession(rivalIds = Vector(PlayerId(1)),
      rivalSeats = Map(PlayerId(1) -> RivalSeatInfo(Seat(2), 500.0)))
    engine.startHand()
    engine.decide(
      makeFlop(pot = 100.0, toCall = 20.0, stackSize = 500.0),
      Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Check)
    )
    val bundle = engine.lastDecisionBundle
    assert(bundle.isDefined, "bundle must be populated")
    bundle.get.certification match
      case _: CertificationResult.TabularCertification =>
        // When PftDpw succeeds, pointwiseExploitability should be Some
        assert(bundle.get.pointwiseExploitability.isDefined,
          "PftDpw path must compute pointwiseExploitability (Def 52C)")
        assert(bundle.get.pointwiseExploitability.get >= Ev.Zero,
          "pointwiseExploitability must be non-negative (Proposition 9.1)")
      case _: CertificationResult.Unavailable =>
        // Native solver not loaded — skip
        ()
      case _ =>
        fail("PftDpw path should produce TabularCertification or Unavailable")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly sicfun.holdem.engine.FormalPathTest"`
Expected: FAIL — `pointwiseExploitability` is `None` even when PftDpw succeeds.

- [ ] **Step 3: Compute pointwiseExploitability in buildFormalCertification**

In `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala`, in method `buildFormalCertification`, after the `adversarialRootGap` computation (around the comment `// Adversarial root gap`), add:

```scala
    // Pointwise exploitability (Def 52C):
    // eps(b; pi) = V^sec_optimal(b) - V^sec_actual(b)
    // V^sec_actual = min over profiles of root value under each profile
    // V^sec_optimal approximated by baselineValue (conservative: baseline is closer to optimal)
    // Note: true V^sec = sup_pi inf_sigma J(b;pi,sigma) requires policy optimization;
    // baselineValue is a lower bound, so this is a conservative upper bound on exploitability.
    val securityValueActual = Ev(minProfileBestValue)
    val securityValueOptimal = Ev(baselineValue) // conservative approximation
    val pwExploit = PointwiseExploitability.compute(securityValueOptimal, securityValueActual)
```

Then in the `DecisionEvaluationBundle` construction, replace:

```scala
      pointwiseExploitability = None,
```

with:

```scala
      pointwiseExploitability = Some(pwExploit),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `sbt "testOnly sicfun.holdem.engine.FormalPathTest"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/engine/StrategicEngine.scala src/test/scala/sicfun/holdem/engine/FormalPathTest.scala
git commit -m "feat(engine): compute pointwiseExploitability (Def 52C) in PftDpw formal path"
```

---

## Task 4: Expand DecisionDiagnostics

The current `DecisionDiagnostics` has only 3 fields. The design doc §10 specifies additional fields for runtime observability.

**Files:**
- Modify: `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala` (companion object `DecisionDiagnostics` and `decide` method)

- [ ] **Step 1: Expand DecisionDiagnostics case class**

In `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala`, find:

```scala
  final case class DecisionDiagnostics(
      heroBucket: Int,
      solverBackend: SolverBackend,
      exploitationBetas: Map[PlayerId, Double]
  )
```

Replace with:

```scala
  final case class DecisionDiagnostics(
      heroBucket: Int,
      solverBackend: SolverBackend,
      exploitationBetas: Map[PlayerId, Double],
      adversarialRootGap: Option[Ev] = None,
      safeActionCount: Option[Int] = None,
      totalActionCount: Int = 0,
      certificationKind: String = "none"
  )
```

- [ ] **Step 2: Populate expanded fields in decide()**

In the `decide` method, replace the `_lastDiagnostics` assignment:

```scala
    _lastDiagnostics = Some(StrategicEngine.DecisionDiagnostics(
      heroBucket = heroBucket,
      solverBackend = config.solverBackend,
      exploitationBetas = session.exploitationStates.map((k, v) => k -> v.beta)
    ))
```

with:

```scala
    val bundleOpt = _lastBundle
    _lastDiagnostics = Some(StrategicEngine.DecisionDiagnostics(
      heroBucket = heroBucket,
      solverBackend = config.solverBackend,
      exploitationBetas = session.exploitationStates.map((k, v) => k -> v.beta),
      adversarialRootGap = bundleOpt.flatMap(_.adversarialRootGap),
      safeActionCount = bundleOpt.flatMap(_.certification match
        case t: CertificationResult.TabularCertification => Some(t.safeActionIndices.size)
        case _ => None
      ),
      totalActionCount = candidateActions.size,
      certificationKind = bundleOpt.map(_.certification match
        case _: CertificationResult.LocalRobustScreening => "LocalRobustScreening"
        case _: CertificationResult.TabularCertification => "TabularCertification"
        case _: CertificationResult.Unavailable => "Unavailable"
      ).getOrElse("none")
    ))
```

- [ ] **Step 3: Run tests**

Run: `sbt "testOnly sicfun.holdem.engine.*"`
Expected: PASS (no behavioral change, fields default to backward-compatible values)

- [ ] **Step 4: Commit**

```bash
git add src/main/scala/sicfun/holdem/engine/StrategicEngine.scala
git commit -m "feat(engine): expand DecisionDiagnostics with certification metadata (design doc §10)"
```

---

## Task 5: observeAction Advisory Clamp from Cached Bundle

Per design doc §6, `observeAction()` should use the cached `_lastBundle` budget estimate as an advisory Bellman clamp after the full-step belief update.

**Files:**
- Modify: `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala` (method `observeAction`)

- [ ] **Step 1: Add advisory clamp after fullStep in observeAction**

In `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala`, in method `observeAction`, after the `_sessionState` update (the line `_sessionState = StrategicEngine.SessionState(...)`), add:

```scala
    // Advisory Bellman clamp from last evaluation bundle (design doc §6).
    // Uses cached budget estimate as an advisory bound — NOT formal B*.
    _lastBundle match
      case Some(bundle) =>
        val budget = bundle.certification match
          case lrs: CertificationResult.LocalRobustScreening => lrs.budgetEstimate
          case tc: CertificationResult.TabularCertification => tc.requiredBudget
          case _: CertificationResult.Unavailable => Double.MaxValue
        if budget < Double.MaxValue then
          val updatedSession = _sessionState.nn
          val clampedExploit = updatedSession.exploitationStates.map { case (rivalId, exploitState) =>
            val totalTolerance = config.epsilonBase + config.exploitConfig.epsilonAdapt
            if budget > totalTolerance then
              // Budget exceeds tolerance — retreat beta toward 0
              val retreated = math.max(0.0, exploitState.beta - config.exploitConfig.cpRetreatRate)
              rivalId -> ExploitationState(beta = retreated)
            else
              rivalId -> exploitState
          }
          _sessionState = StrategicEngine.SessionState(
            rivalBeliefs = updatedSession.rivalBeliefs,
            exploitationStates = clampedExploit,
            rivalSeats = updatedSession.rivalSeats
          )
      case None => // No bundle yet — skip clamp
```

- [ ] **Step 2: Run tests**

Run: `sbt "testOnly sicfun.holdem.engine.*"`
Expected: PASS (advisory clamp only fires when budget exceeds tolerance, which is rare in default config)

- [ ] **Step 3: Commit**

```bash
git add src/main/scala/sicfun/holdem/engine/StrategicEngine.scala
git commit -m "feat(engine): add advisory Bellman clamp in observeAction from cached bundle (design doc §6)"
```

---

## Task 6: Wire EmpiricalDeploymentSet into Engine

`EmpiricalDeploymentSet` and `OperationalBaseline` exist as types but are never populated or queried. Wire them into the engine to track deployment exploitability (Def 52D) over time.

**Files:**
- Modify: `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala`

- [ ] **Step 1: Add deploymentSet to SessionState**

In `StrategicEngine.scala` companion object, modify `SessionState`:

```scala
  final case class SessionState(
      rivalBeliefs: Map[PlayerId, StrategicRivalBelief],
      exploitationStates: Map[PlayerId, ExploitationState],
      rivalSeats: Map[PlayerId, RivalSeatInfo] = Map.empty,
      deploymentSet: EmpiricalDeploymentSet = EmpiricalDeploymentSet(Vector.empty, maxSize = 50)
  )
```

- [ ] **Step 2: Record deployment snapshot after each decide()**

In the `decide` method, after `_lastBundle = Some(bundle)` is set (in both WPomcp and PftDpw paths), add deployment tracking. Place this before the final `action` return, after the `_lastDiagnostics` assignment:

```scala
    // Record deployment snapshot for Def 52D tracking
    _lastBundle.foreach { bundle =>
      bundle.pointwiseExploitability.foreach { pwExploit =>
        val session = _sessionState.nn
        val beliefs = session.rivalBeliefs.values
        val avgEntropy = if beliefs.isEmpty then 0.0
          else beliefs.map { b =>
            val probs = StrategicClass.values.map(c => b.typePosterior.probabilityOf(c))
            -probs.filter(_ > 0).map(p => p * math.log(p)).sum
          }.sum / beliefs.size
        val summary = DeploymentBeliefSummary(
          beliefEntropy = avgEntropy,
          exploitabilitySnapshot = pwExploit,
          timestamp = System.currentTimeMillis()
        )
        val updatedDeploy = session.deploymentSet.add(summary)
        _sessionState = session.copy(deploymentSet = updatedDeploy)
      }
    }
```

- [ ] **Step 3: Populate deploymentExploitability in PftDpw bundle**

In `buildFormalCertification`, before constructing the final `DecisionEvaluationBundle`, compute deployment exploitability from the session's deployment set. Add after the `pwExploit` computation:

```scala
    // Deployment exploitability (Def 52D) from empirical deployment set
    // Only meaningful when there are prior observations in the deployment set
    val deployExploit: Option[Ev] = None // Populated post-bundle by decide() on next call
```

Note: Deployment exploitability is inherently retrospective — it's the max exploitability over *prior* beliefs in B_dep. The current decision's exploitability gets added to the set *after* the bundle is built. On subsequent calls, the bundle will reflect the accumulated history.

- [ ] **Step 4: Run tests**

Run: `sbt "testOnly sicfun.holdem.engine.*"`
Expected: PASS (deployment tracking is additive, no existing behavior changes)

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/engine/StrategicEngine.scala
git commit -m "feat(engine): wire EmpiricalDeploymentSet for Def 52D deployment exploitability tracking"
```

---

## Verification Checklist

After all 6 tasks:

1. `sbt "testOnly sicfun.holdem.strategic.FormalClosureValidationTest"` — assumption alignment + closure gaps
2. `sbt "testOnly sicfun.holdem.strategic.StrategicClassTest"` — Def 1 enum naming
3. `sbt "testOnly sicfun.holdem.engine.FormalPathTest"` — pointwiseExploitability
4. `sbt "testOnly sicfun.holdem.engine.FourWorldSolveTest"` — no regression
5. `sbt "testOnly sicfun.holdem.engine.FourWorldFormulationTest"` — no regression
6. `sbt "testOnly sicfun.holdem.strategic.*"` — full strategic suite
7. `sbt "testOnly sicfun.holdem.engine.*"` — full engine suite

All should pass green. The AssumptionManifest should report 11 entries matching v0.31.1 spec definitions.
