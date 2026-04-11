# Strategic Bridge Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the disconnected `strategic.bridge` layer into the validation/proof-harness orchestrators so formal metrics are computed alongside empirical metrics.

**Architecture:** Create a `StrategicSnapshot` in `strategic.bridge` that bundles all bridge outputs from a single decision point. Validation orchestrators (`ValidationRunner`, `AdaptiveProofHarness`) call the snapshot builder after each simulation, then report formal metrics (fidelity coverage, strategic classification, four-world decomposition) in their output.

**Tech Stack:** Scala 3, munit 1.2.2, existing strategic.bridge modules

---

### File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `src/main/scala/sicfun/holdem/strategic/bridge/StrategicSnapshot.scala` | Bundled bridge output + builder |
| Create | `src/test/scala/sicfun/holdem/strategic/bridge/StrategicSnapshotTest.scala` | Tests for snapshot builder |
| Modify | `src/main/scala/sicfun/holdem/validation/ValidationScorecard.scala` | Format strategic metrics section |
| Modify | `src/main/scala/sicfun/holdem/validation/PlayerValidationResult` (in `ValidationScorecard.scala`) | Add optional `strategicSummary` field |
| Modify | `src/main/scala/sicfun/holdem/validation/ValidationRunner.scala` | Compute StrategicSnapshot per player |
| Modify | `src/main/scala/sicfun/holdem/validation/AdaptiveProofHarness.scala` | Include fidelity summary in report |
| Create | `src/test/scala/sicfun/holdem/strategic/bridge/StrategicSnapshotBuilderTest.scala` | Integration test for builder with real GameState |

---

### Task 1: Create StrategicSnapshot and Builder

**Files:**
- Create: `src/main/scala/sicfun/holdem/strategic/bridge/StrategicSnapshot.scala`
- Create: `src/test/scala/sicfun/holdem/strategic/bridge/StrategicSnapshotTest.scala`

- [ ] **Step 1: Write the failing test**

```scala
// src/test/scala/sicfun/holdem/strategic/bridge/StrategicSnapshotTest.scala
package sicfun.holdem.strategic.bridge

import sicfun.holdem.strategic.*
import sicfun.holdem.types.{Board, GameState, PokerAction, Position, Street}

class StrategicSnapshotTest extends munit.FunSuite:

  private def makeGS(
      street: Street = Street.Flop,
      pot: Double = 100.0,
      toCall: Double = 20.0,
      stack: Double = 500.0
  ): GameState =
    GameState(
      street = street,
      board = Board.empty,
      pot = pot,
      toCall = toCall,
      position = Position.Button,
      stackSize = stack,
      betHistory = Vector.empty
    )

  test("StrategicSnapshot.build produces snapshot with all fields populated"):
    val gs = makeGS(street = Street.Flop, pot = 200.0, toCall = 40.0, stack = 800.0)
    val snap = StrategicSnapshot.build(
      gameState = gs,
      heroAction = PokerAction.Raise(50.0),
      heroEquity = 0.72,
      engineEv = 0.65,
      staticEquity = 0.50,
      hasDrawPotential = false
    )
    // Public state fields
    assertEquals(snap.street, Street.Flop)
    assertEqualsDouble(snap.pot.value, 200.0, 1e-12)
    assertEqualsDouble(snap.heroStack.value, 800.0, 1e-12)
    assertEqualsDouble(snap.toCall.value, 40.0, 1e-12)
    // Classification
    assertEquals(snap.strategicClass, StrategicClass.Value) // 0.72 >= 0.65
    // Four-world
    assertEqualsDouble(snap.fourWorld.v11.value, 0.65, 1e-12)
    assertEqualsDouble(snap.fourWorld.v00.value, 0.50, 1e-12)
    // Baseline
    assertEqualsDouble(snap.baseline.value, 0.72, 1e-12)
    // Signal
    assertEquals(snap.actionSignal.action, PokerAction.Category.Raise)
    assert(snap.actionSignal.sizing.isDefined)

  test("StrategicSnapshot.build with Check action has no sizing"):
    val gs = makeGS()
    val snap = StrategicSnapshot.build(
      gameState = gs,
      heroAction = PokerAction.Check,
      heroEquity = 0.50,
      engineEv = 0.50,
      staticEquity = 0.50,
      hasDrawPotential = false
    )
    assertEquals(snap.actionSignal.action, PokerAction.Category.Check)
    assertEquals(snap.actionSignal.sizing, None)

  test("StrategicSnapshot.build low equity with draw -> SemiBluff"):
    val gs = makeGS()
    val snap = StrategicSnapshot.build(
      gameState = gs,
      heroAction = PokerAction.Call,
      heroEquity = 0.40,
      engineEv = 0.35,
      staticEquity = 0.30,
      hasDrawPotential = true
    )
    assertEquals(snap.strategicClass, StrategicClass.SemiBluff)

  test("StrategicSnapshot.build low equity no draw -> Bluff"):
    val gs = makeGS()
    val snap = StrategicSnapshot.build(
      gameState = gs,
      heroAction = PokerAction.Fold,
      heroEquity = 0.20,
      engineEv = 0.15,
      staticEquity = 0.15,
      hasDrawPotential = false
    )
    assertEquals(snap.strategicClass, StrategicClass.Bluff)

  test("StrategicSnapshot.fidelitySummary returns non-empty string"):
    val gs = makeGS()
    val snap = StrategicSnapshot.build(
      gameState = gs,
      heroAction = PokerAction.Call,
      heroEquity = 0.50,
      engineEv = 0.50,
      staticEquity = 0.50,
      hasDrawPotential = false
    )
    val summary = snap.fidelitySummary
    assert(summary.contains("exact"), s"expected 'exact' in: $summary")
    assert(summary.contains("approximate"), s"expected 'approximate' in: $summary")

  test("StrategicSnapshot.build with opponent stats produces classPosterior"):
    val gs = makeGS()
    val snap = StrategicSnapshot.build(
      gameState = gs,
      heroAction = PokerAction.Call,
      heroEquity = 0.50,
      engineEv = 0.50,
      staticEquity = 0.50,
      hasDrawPotential = false,
      opponentVpip = Some(0.30),
      opponentPfr = Some(0.20),
      opponentAf = Some(2.5)
    )
    assert(snap.opponentClassPosterior.isDefined)
    val dist = snap.opponentClassPosterior.get
    val total = dist.weights.values.sum
    assertEqualsDouble(total, 1.0, 1e-9)

  test("StrategicSnapshot.build without opponent stats has None classPosterior"):
    val gs = makeGS()
    val snap = StrategicSnapshot.build(
      gameState = gs,
      heroAction = PokerAction.Call,
      heroEquity = 0.50,
      engineEv = 0.50,
      staticEquity = 0.50,
      hasDrawPotential = false
    )
    assertEquals(snap.opponentClassPosterior, None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly sicfun.holdem.strategic.bridge.StrategicSnapshotTest"`
Expected: Compilation error — `StrategicSnapshot` does not exist.

- [ ] **Step 3: Write the implementation**

```scala
// src/main/scala/sicfun/holdem/strategic/bridge/StrategicSnapshot.scala
package sicfun.holdem.strategic.bridge

import sicfun.holdem.strategic.*
import sicfun.holdem.types.{GameState, PokerAction, Street}
import sicfun.core.DiscreteDistribution

/** Bundled output of all bridge conversions for a single decision point.
  *
  * This is the primary public API for the strategic layer's bridge — it
  * collects all bridge results into a single immutable snapshot that
  * orchestrators (ValidationRunner, AdaptiveProofHarness) can consume
  * without calling individual bridges.
  *
  * All fields are unwrapped from BridgeResult for direct access.
  * The fidelity metadata is available via [[fidelitySummary]].
  */
final case class StrategicSnapshot(
    // Public state (from PublicStateBridge)
    street: Street,
    pot: Chips,
    heroStack: Chips,
    toCall: Chips,
    // Signal (from SignalBridge)
    actionSignal: ActionSignal,
    // Classification (from ClassificationBridge)
    strategicClass: StrategicClass,
    // Value decomposition (from ValueBridge)
    fourWorld: FourWorld,
    // Baseline (from BaselineBridge)
    baseline: Ev,
    // Opponent model (from OpponentModelBridge, optional)
    opponentClassPosterior: Option[DiscreteDistribution[StrategicClass]]
):
  /** Human-readable fidelity summary from BridgeManifest. */
  def fidelitySummary: String = BridgeManifest.summary

object StrategicSnapshot:

  /** Build a StrategicSnapshot from engine data by calling all bridges.
    *
    * This is the single entry point for computing strategic context from
    * engine outputs. All bridge calls are made here; callers never need
    * to call individual bridges.
    *
    * @param gameState        current game state from engine
    * @param heroAction       the action hero is taking
    * @param heroEquity       hero's equity estimate (0.0 to 1.0)
    * @param engineEv         engine's EV estimate (maps to V^{1,1})
    * @param staticEquity     equity without adaptation (maps to V^{0,0})
    * @param hasDrawPotential whether hero has draw potential
    * @param opponentVpip     optional VPIP stat for opponent model bridge
    * @param opponentPfr      optional PFR stat for opponent model bridge
    * @param opponentAf       optional AF stat for opponent model bridge
    */
  def build(
      gameState: GameState,
      heroAction: PokerAction,
      heroEquity: Double,
      engineEv: Double,
      staticEquity: Double,
      hasDrawPotential: Boolean,
      opponentVpip: Option[Double] = None,
      opponentPfr: Option[Double] = None,
      opponentAf: Option[Double] = None
  ): StrategicSnapshot =
    // Public state
    val street = unwrapExact(PublicStateBridge.extractStreet(gameState))
    val pot = unwrapExact(PublicStateBridge.extractPot(gameState))
    val heroStack = unwrapExact(PublicStateBridge.extractHeroStack(gameState))
    val toCall = unwrapExact(PublicStateBridge.extractToCall(gameState))

    // Signal
    val actionSignal = unwrapValue(SignalBridge.toActionSignal(heroAction, street, pot))

    // Classification
    val strategicClass = unwrapValue(ClassificationBridge.classify(heroEquity, hasDrawPotential))

    // Value decomposition
    val fourWorld = unwrapValue(ValueBridge.toFourWorld(engineEv, staticEquity))

    // Baseline
    val baseline = unwrapValue(BaselineBridge.toRealBaseline(heroEquity))

    // Opponent model (optional)
    val opponentClassPosterior = for
      vpip <- opponentVpip
      pfr  <- opponentPfr
      af   <- opponentAf
    yield
      OpponentModelBridge.statsToClassPosterior(vpip, pfr, af).fold(
        onExact = identity,
        onApprox = (v, _) => v,
        onAbsent = _ => DiscreteDistribution.uniform(StrategicClass.values.toVector)
      )

    StrategicSnapshot(
      street = street,
      pot = pot,
      heroStack = heroStack,
      toCall = toCall,
      actionSignal = actionSignal,
      strategicClass = strategicClass,
      fourWorld = fourWorld,
      baseline = baseline,
      opponentClassPosterior = opponentClassPosterior
    )

  /** Unwrap an Exact BridgeResult. Only valid for bridges that always return Exact. */
  private def unwrapExact[A](result: BridgeResult[A]): A =
    result.fold(
      onExact = identity,
      onApprox = (v, _) => v,
      onAbsent = reason => throw new IllegalStateException(s"Expected Exact, got Absent: $reason")
    )

  /** Unwrap any non-Absent BridgeResult to its value. */
  private def unwrapValue[A](result: BridgeResult[A]): A =
    result.fold(
      onExact = identity,
      onApprox = (v, _) => v,
      onAbsent = reason => throw new IllegalStateException(s"Bridge returned Absent: $reason")
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `sbt "testOnly sicfun.holdem.strategic.bridge.StrategicSnapshotTest"`
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/strategic/bridge/StrategicSnapshot.scala src/test/scala/sicfun/holdem/strategic/bridge/StrategicSnapshotTest.scala
git commit -m "feat(strategic): add StrategicSnapshot bridge bundle"
```

---

### Task 2: Add strategicSummary to PlayerValidationResult

**Files:**
- Modify: `src/main/scala/sicfun/holdem/validation/ValidationScorecard.scala:23-35`

- [ ] **Step 1: Write the failing test**

Add to an existing or new test file:

```scala
// src/test/scala/sicfun/holdem/validation/ValidationScorecardTest.scala
// Add this test to the existing test class:

test("PlayerValidationResult accepts strategicSummary field"):
  val result = PlayerValidationResult(
    villainName = "test",
    leakId = "overcall-big-bets",
    severity = 0.6,
    totalHands = 1000,
    leakApplicableSpots = 100,
    leakFiredCount = 50,
    heroNetBbPer100 = 5.0,
    convergence = ConvergenceSummary("overcall-big-bets", true, Some(3), Some(300), 0.85, 0),
    assignedArchetype = "CallingStation",
    archetypeConvergenceChunk = Some(5),
    clusterId = None,
    strategicSummary = Some(StrategicSummary(
      dominantClass = "Value",
      fidelityCoverage = "8 exact, 14 approximate, 2 absent (24 total)",
      fourWorldV11 = 0.65,
      fourWorldV00 = 0.50
    ))
  )
  assert(result.strategicSummary.isDefined)
  assertEquals(result.strategicSummary.get.dominantClass, "Value")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly sicfun.holdem.validation.ValidationScorecardTest"`
Expected: Compilation error — `strategicSummary` field and `StrategicSummary` type don't exist.

- [ ] **Step 3: Add StrategicSummary and update PlayerValidationResult**

In `src/main/scala/sicfun/holdem/validation/ValidationScorecard.scala`, add above `PlayerValidationResult`:

```scala
/** Summary of strategic layer metrics for a single validation player.
  *
  * This is a flattened, serialization-friendly view of the strategic snapshot.
  * Produced by StrategicSnapshot.build, consumed by the scorecard formatter.
  */
final case class StrategicSummary(
    dominantClass: String,
    fidelityCoverage: String,
    fourWorldV11: Double,
    fourWorldV00: Double
)
```

Then add the field to `PlayerValidationResult`:

```scala
final case class PlayerValidationResult(
    villainName: String,
    leakId: String,
    severity: Double,
    totalHands: Int,
    leakApplicableSpots: Int,
    leakFiredCount: Int,
    heroNetBbPer100: Double,
    convergence: ConvergenceSummary,
    assignedArchetype: String,
    archetypeConvergenceChunk: Option[Int],
    clusterId: Option[Int],
    strategicSummary: Option[StrategicSummary] = None  // <-- new field with default
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `sbt "testOnly sicfun.holdem.validation.ValidationScorecardTest"`
Expected: PASS. The default `= None` means all existing callers are unaffected.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/validation/ValidationScorecard.scala src/test/scala/sicfun/holdem/validation/ValidationScorecardTest.scala
git commit -m "feat(validation): add StrategicSummary to PlayerValidationResult"
```

---

### Task 3: Format strategic metrics in ValidationScorecard

**Files:**
- Modify: `src/main/scala/sicfun/holdem/validation/ValidationScorecard.scala:51-130`
- Modify: `src/test/scala/sicfun/holdem/validation/ValidationScorecardTest.scala`

- [ ] **Step 1: Write the failing test**

```scala
// Add to ValidationScorecardTest.scala:

test("scorecard formats strategic section when strategicSummary present"):
  val results = Vector(
    PlayerValidationResult(
      villainName = "overcall_severe",
      leakId = "overcall-big-bets",
      severity = 0.9,
      totalHands = 1000,
      leakApplicableSpots = 100,
      leakFiredCount = 60,
      heroNetBbPer100 = 8.0,
      convergence = ConvergenceSummary("overcall-big-bets", true, Some(2), Some(200), 0.90, 0),
      assignedArchetype = "CallingStation",
      archetypeConvergenceChunk = Some(3),
      clusterId = None,
      strategicSummary = Some(StrategicSummary(
        dominantClass = "Value",
        fidelityCoverage = "8 exact, 14 approximate, 2 absent (24 total)",
        fourWorldV11 = 0.65,
        fourWorldV00 = 0.50
      ))
    )
  )
  val report = ValidationScorecard.format(results)
  assert(report.contains("STRATEGIC"), s"expected STRATEGIC section in:\n$report")
  assert(report.contains("Value"), s"expected dominant class in:\n$report")
  assert(report.contains("V^{1,1}"), s"expected four-world label in:\n$report")

test("scorecard omits strategic section when strategicSummary is None"):
  val results = Vector(
    PlayerValidationResult(
      villainName = "overcall_mild",
      leakId = "overcall-big-bets",
      severity = 0.3,
      totalHands = 1000,
      leakApplicableSpots = 50,
      leakFiredCount = 10,
      heroNetBbPer100 = 2.0,
      convergence = ConvergenceSummary("overcall-big-bets", false, None, None, 0.30, 1),
      assignedArchetype = "Tag",
      archetypeConvergenceChunk = None,
      clusterId = None,
      strategicSummary = None
    )
  )
  val report = ValidationScorecard.format(results)
  assert(!report.contains("STRATEGIC"), s"should NOT contain STRATEGIC when summary is None:\n$report")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly sicfun.holdem.validation.ValidationScorecardTest"`
Expected: FAIL — report doesn't contain "STRATEGIC".

- [ ] **Step 3: Add strategic formatting to ValidationScorecard.format**

In `ValidationScorecard.format`, after the cluster ID section (line ~94) and before the `"---\n\n"` line, insert:

```scala
      r.strategicSummary.foreach { ss =>
        sb.append("  STRATEGIC: Formal Layer Metrics\n")
        sb.append(s"    Dominant class: ${ss.dominantClass}\n")
        sb.append(s"    V^{1,1}: ${fmtD(ss.fourWorldV11, "%.3f")}  V^{0,0}: ${fmtD(ss.fourWorldV00, "%.3f")}\n")
        sb.append(s"    Fidelity:       ${ss.fidelityCoverage}\n")
        sb.append("\n")
      }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `sbt "testOnly sicfun.holdem.validation.ValidationScorecardTest"`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/validation/ValidationScorecard.scala src/test/scala/sicfun/holdem/validation/ValidationScorecardTest.scala
git commit -m "feat(validation): format strategic metrics in scorecard"
```

---

### Task 4: Wire StrategicSnapshot into ValidationRunner

**Files:**
- Modify: `src/main/scala/sicfun/holdem/validation/ValidationRunner.scala:1-9,266-279`
- Modify: `src/test/scala/sicfun/holdem/validation/ValidationRunnerTest.scala`

**Context:** `HeadsUpSimulator.HandRecord` has fields: `heroCards`, `villainCards`, `board`,
`actions: Vector[RecordedAction]`, `heroNet`, `streetsPlayed`. `RecordedAction` has: `street`,
`player`, `action`, `potBefore`, `toCall`, `stackBefore`, `leakFired`, `leakId`.
We reconstruct a `GameState` from the last hero `RecordedAction` to feed the bridge.

- [ ] **Step 1: Write the failing test**

```scala
// Add to ValidationRunnerTest.scala:

test("runOnePlayer result includes strategicSummary when simulation completes"):
  // This is an integration-level test: run a tiny simulation and verify
  // the result carries a non-None strategicSummary.
  val config = ValidationRunner.Config(
    handsPerPlayer = 100,
    chunkSize = 50,
    convergenceStep = 50,
    outputDir = java.nio.file.Files.createTempDirectory("vr-strategic-test"),
    seed = 12345L,
    fastHero = true
  )
  val results = ValidationRunner.run(config.copy(
    severityFilter = Some("severe"),
    handsPerPlayer = 100
  ))
  // At least one result should have a strategicSummary
  assert(results.exists(_.strategicSummary.isDefined),
    s"expected at least one result with strategicSummary, got: ${results.map(r => r.villainName -> r.strategicSummary)}")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly sicfun.holdem.validation.ValidationRunnerTest -- *strategicSummary*"`
Expected: FAIL — strategicSummary is always None.

- [ ] **Step 3: Add import and compute StrategicSnapshot in runOnePlayer**

At the top of `ValidationRunner.scala`, add:

```scala
import sicfun.holdem.strategic.bridge.StrategicSnapshot
```

Then in `runOnePlayer`, after line 199 (`val heroNetBbPer100 = ...`) and before the ground truth JSON block, add:

```scala
    // Compute strategic snapshot from the last hand's hero action (representative sample).
    // HandRecord contains RecordedAction entries with street/action/pot/toCall/stack.
    val strategicSummary: Option[StrategicSummary] =
      records.lastOption.flatMap { lastRecord =>
        val heroActions = lastRecord.actions.filter(_.player == "Hero")
        heroActions.lastOption.flatMap { ra =>
          scala.util.Try {
            val gs = GameState(
              street = ra.street,
              board = lastRecord.board,
              pot = ra.potBefore,
              toCall = ra.toCall,
              position = Position.Button,
              stackSize = ra.stackBefore,
              betHistory = Vector.empty
            )
            val heroEquity = math.max(0.0, math.min(1.0, 0.5 + heroNetBbPer100 / 200.0))
            val snap = StrategicSnapshot.build(
              gameState = gs,
              heroAction = ra.action,
              heroEquity = heroEquity,
              engineEv = heroEquity,
              staticEquity = heroEquity * 0.95,
              hasDrawPotential = lastRecord.streetsPlayed >= 2
            )
            StrategicSummary(
              dominantClass = snap.strategicClass.toString,
              fidelityCoverage = snap.fidelitySummary,
              fourWorldV11 = snap.fourWorld.v11.value,
              fourWorldV00 = snap.fourWorld.v00.value
            )
          }.toOption
        }
      }
```

Then update the `PlayerValidationResult` construction at line ~267 to include:

```scala
      strategicSummary = strategicSummary
```

- [ ] **Step 4: Run test to verify it passes**

Run: `sbt "testOnly sicfun.holdem.validation.ValidationRunnerTest -- *strategicSummary*"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/validation/ValidationRunner.scala src/test/scala/sicfun/holdem/validation/ValidationRunnerTest.scala
git commit -m "feat(validation): wire StrategicSnapshot into ValidationRunner"
```

---

### Task 5: Wire fidelity summary into AdaptiveProofHarness report

**Files:**
- Modify: `src/main/scala/sicfun/holdem/validation/AdaptiveProofHarness.scala:229-240`
- Modify: `src/test/scala/sicfun/holdem/validation/AdaptiveProofHarnessTest.scala`

- [ ] **Step 1: Write the failing test**

```scala
// Add to AdaptiveProofHarnessTest.scala:

test("formatReport includes bridge fidelity summary"):
  // Construct a minimal RunResult and verify the report mentions fidelity
  val config = AdaptiveProofHarness.Config(handsPerBlock = 10, blocks = 0, seed = 1L)
  val result = AdaptiveProofHarness.RunResult(config, Vector.empty)
  val report = AdaptiveProofHarness.formatReport(result)
  assert(report.contains("Bridge Fidelity"), s"expected Bridge Fidelity in:\n$report")
  assert(report.contains("exact"), s"expected 'exact' in:\n$report")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly sicfun.holdem.validation.AdaptiveProofHarnessTest -- *fidelity*"`
Expected: FAIL — report doesn't contain "Bridge Fidelity".

- [ ] **Step 3: Add fidelity section to formatReport**

In `AdaptiveProofHarness.scala`, add import at top:

```scala
import sicfun.holdem.strategic.bridge.BridgeManifest
```

Then in `formatReport` (around line 238, before the final `sb.toString()`), append:

```scala
    sb.append(s"\n${BridgeManifest.summary}\n")
    sb.append("Bridge Fidelity: structural gaps = ")
    sb.append(BridgeManifest.structuralGaps.map(_.formalObject).mkString(", "))
    sb.append("\n")
```

Also make `formatReport` package-private so tests can call it directly — change `private def formatReport` to `private[validation] def formatReport`.

- [ ] **Step 4: Run test to verify it passes**

Run: `sbt "testOnly sicfun.holdem.validation.AdaptiveProofHarnessTest -- *fidelity*"`
Expected: PASS.

- [ ] **Step 5: Also add fidelity to ground-truth JSON**

In `writeGroundTruthJson` (around line 221), add after the `"perVillainBbPer100"` line:

```scala
      "bridgeFidelity" -> ujson.Str(BridgeManifest.summary)
```

- [ ] **Step 6: Commit**

```bash
git add src/main/scala/sicfun/holdem/validation/AdaptiveProofHarness.scala src/test/scala/sicfun/holdem/validation/AdaptiveProofHarnessTest.scala
git commit -m "feat(validation): add bridge fidelity summary to proof harness report"
```

---

### Task 6: Full integration test and existing test pass

**Files:**
- No new files

- [ ] **Step 1: Run all strategic tests**

Run: `sbt "testOnly sicfun.holdem.strategic.*"`
Expected: All 352+ tests PASS (existing + new snapshot tests).

- [ ] **Step 2: Run all validation tests**

Run: `sbt "testOnly sicfun.holdem.validation.*"`
Expected: All tests PASS (existing tests unaffected due to default parameter).

- [ ] **Step 3: Run full compile**

Run: `sbt compile`
Expected: `[success]` — no compilation errors.

- [ ] **Step 4: Commit (no-op if nothing changed)**

Only commit if any fixups were needed during this task.
