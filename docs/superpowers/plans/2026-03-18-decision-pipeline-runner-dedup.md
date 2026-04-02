# Shared Decision Infrastructure + Runner Deduplication

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate ~200 lines of character-for-character identical code between AcpcMatchRunner.Runner and SlumbotMatchRunner.Runner by extracting shared decision logic, runner infrastructure, and utility functions.

**Architecture:** Extract horizontal shared layers (types → engine → runtime) from two vertical slices (ACPC runner, Slumbot runner) that were built by copy-paste. New shared code lives in three files: `HeroMode` enum and `PokerFormatting` utilities in `types/`, `HeroDecisionPipeline` in `engine/`, and `MatchRunnerSupport` in `runtime/`. Both Runner classes then delegate to shared code, keeping only protocol-specific logic (socket vs HTTP, codec differences).

**Tech Stack:** Scala 3.8.1, munit 1.2.2, SBT

---

## Scope

### This Plan (Plan 1 of 3)

Targets the P0 findings from the cohesion audit:
- **3x private `HeroMode` enum** (ACPC, Slumbot, PlayingHall) → shared enum in `types/`
- **9x `renderAction`**, **5x `fmt`**, **2x `roundedChips`**, **2x `heroModeLabel`** → `PokerFormatting` in `types/`
- **2x identical `decideHero`** (ACPC, Slumbot Runners) → `HeroDecisionPipeline` in `engine/`
- **2x identical `legalRaiseCandidates`**, **2x `heroCandidates`** → `HeroDecisionPipeline` in `engine/`
- **2x identical `newAdaptiveEngine`** → `HeroDecisionPipeline` in `engine/`
- **2x near-identical `recordOutcome`**, `buildSummary`, `writeSummary`, `appendDecisionLog` → `MatchRunnerSupport` in `runtime/`

### Future Plans (not in scope)
- **Plan 2: bench/ consolidation** — 8x card/hole, 4x BatchData, 3x loadBatch, GPU property constants
- **Plan 3: PlayingHall decomposition** — extract trapped GTO methods, unify with HeroDecisionPipeline

## File Map

### New Files

| File | Responsibility |
|---|---|
| `src/main/scala/sicfun/holdem/types/HeroMode.scala` | Shared `HeroMode` enum (Adaptive, Gto) — replaces 3 private copies |
| `src/main/scala/sicfun/holdem/types/PokerFormatting.scala` | `renderAction`, `fmtDouble`, `roundedChips`, `heroModeLabel` — replaces 9+5+2+2 copies |
| `src/main/scala/sicfun/holdem/engine/HeroDecisionPipeline.scala` | `decideHero`, `legalRaiseCandidates`, `heroCandidates`, `newAdaptiveEngine` + `RaiseSizingContext` case class |
| `src/main/scala/sicfun/holdem/runtime/MatchRunnerSupport.scala` | `MatchStatistics` (mutable counters + `recordOutcome` + `buildSummary`), `RunSummary`, `writeSummary`, `appendDecisionLog` |
| `src/test/scala/sicfun/holdem/types/PokerFormattingTest.scala` | Unit tests for all formatting functions |
| `src/test/scala/sicfun/holdem/engine/HeroDecisionPipelineTest.scala` | Unit tests for `legalRaiseCandidates`, `heroCandidates` (pure functions) |
| `src/test/scala/sicfun/holdem/runtime/MatchRunnerSupportTest.scala` | Unit tests for `MatchStatistics`, `writeSummary`, `appendDecisionLog` |

### Modified Files

| File | Changes |
|---|---|
| `src/main/scala/sicfun/holdem/runtime/AcpcMatchRunner.scala` | Remove private `HeroMode`, `renderAction`, `fmt`, `roundedChips`, `heroModeLabel`, `decideHero`, `legalRaiseCandidates`, `heroCandidates`, `recordOutcome`, `buildSummary`, `writeSummary`, `appendDecisionLog`, `newAdaptiveEngine`, `RunSummary`. Runner delegates to shared code. |
| `src/main/scala/sicfun/holdem/runtime/SlumbotMatchRunner.scala` | Same removals as ACPC. Runner delegates to shared code. |
| `src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala` | Remove private `HeroMode` enum, import shared `HeroMode` from `types`. No other changes (PlayingHall's GTO methods are Plan 3). |

### Dependency Order

```
Task 1 (HeroMode) ──┐
                     ├──→ Task 3 (HeroDecisionPipeline) ──┐
Task 2 (Formatting) ─┤                                     ├──→ Task 5 (Refactor ACPC) ──┐
                     └──→ Task 4 (MatchRunnerSupport) ─────┘                               ├→ Task 7 (Verify)
                                                            └──→ Task 6 (Refactor Slumbot) ┘
```

---

## Task 1: Extract HeroMode Enum

**Files:**
- Create: `src/main/scala/sicfun/holdem/types/HeroMode.scala`
- Modify: `src/main/scala/sicfun/holdem/runtime/AcpcMatchRunner.scala`
- Modify: `src/main/scala/sicfun/holdem/runtime/SlumbotMatchRunner.scala`
- Modify: `src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala`

- [ ] **Step 1: Create shared HeroMode enum**

```scala
package sicfun.holdem.types

/** Decision mode for hero play: adaptive (Bayesian engine) or GTO (CFR solver). */
enum HeroMode:
  case Adaptive
  case Gto
```

- [ ] **Step 2: Update AcpcMatchRunner to use shared HeroMode**

In `AcpcMatchRunner.scala`, remove the private enum definition (lines 552-554):
```scala
// DELETE:
  private enum HeroMode:
    case Adaptive
    case Gto
```

Add import at top: `import sicfun.holdem.types.HeroMode`

The `Config` case class field `heroMode: HeroMode` and all `config.heroMode match` patterns remain unchanged — they now reference the shared type.

- [ ] **Step 3: Update SlumbotMatchRunner to use shared HeroMode**

Same changes as Step 2 in `SlumbotMatchRunner.scala` (lines 358-360).

- [ ] **Step 4: Update TexasHoldemPlayingHall to use shared HeroMode**

In `TexasHoldemPlayingHall.scala`, remove the private enum (lines 29-31):
```scala
// DELETE:
  private enum HeroMode:
    case Adaptive
    case Gto
```

Add import at top: `import sicfun.holdem.types.HeroMode`

Note: `GtoMode` (Fast, Exact) remains private to PlayingHall — it's PlayingHall-specific.

- [ ] **Step 5: Full test check**

Run: `sbt test`
Expected: ALL PASS — all three files now reference the same `HeroMode` type. Running full tests (not just compile) to catch any enum visibility regressions.

- [ ] **Step 6: Commit**

```bash
git add src/main/scala/sicfun/holdem/types/HeroMode.scala \
  src/main/scala/sicfun/holdem/runtime/AcpcMatchRunner.scala \
  src/main/scala/sicfun/holdem/runtime/SlumbotMatchRunner.scala \
  src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala
git commit -m "refactor: extract shared HeroMode enum from 3 private copies"
```

---

## Task 2: Extract PokerFormatting Utilities

**Files:**
- Create: `src/main/scala/sicfun/holdem/types/PokerFormatting.scala`
- Create: `src/test/scala/sicfun/holdem/types/PokerFormattingTest.scala`

- [ ] **Step 1: Write failing tests for renderAction**

```scala
package sicfun.holdem.types

import munit.FunSuite

class PokerFormattingTest extends FunSuite:

  test("renderAction formats Fold"):
    assertEquals(PokerFormatting.renderAction(PokerAction.Fold), "Fold")

  test("renderAction formats Check"):
    assertEquals(PokerFormatting.renderAction(PokerAction.Check), "Check")

  test("renderAction formats Call"):
    assertEquals(PokerFormatting.renderAction(PokerAction.Call), "Call")

  test("renderAction formats Raise with amount"):
    assertEquals(PokerFormatting.renderAction(PokerAction.Raise(2.50)), "Raise:2.50")

  test("renderAction formats Raise with integer-like amount"):
    assertEquals(PokerFormatting.renderAction(PokerAction.Raise(3.00)), "Raise:3.00")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `sbt "testOnly sicfun.holdem.types.PokerFormattingTest"`
Expected: FAIL — `PokerFormatting` does not exist.

- [ ] **Step 3: Write failing tests for fmtDouble and roundedChips**

Append to the test file:

```scala
  test("fmtDouble formats with specified precision"):
    assertEquals(PokerFormatting.fmtDouble(3.14159, 2), "3.14")
    assertEquals(PokerFormatting.fmtDouble(3.14159, 4), "3.1416")
    assertEquals(PokerFormatting.fmtDouble(0.0, 3), "0.000")

  test("fmtDouble uses Locale.ROOT to avoid comma decimals"):
    assertEquals(PokerFormatting.fmtDouble(1234.5, 1), "1234.5")

  test("roundedChips rounds to nearest 50 with minimum 50"):
    assertEquals(PokerFormatting.roundedChips(0.0), 50)
    assertEquals(PokerFormatting.roundedChips(25.0), 50)
    assertEquals(PokerFormatting.roundedChips(74.0), 50)
    assertEquals(PokerFormatting.roundedChips(75.0), 100)
    assertEquals(PokerFormatting.roundedChips(100.0), 100)
    assertEquals(PokerFormatting.roundedChips(125.0), 150)

  test("heroModeLabel returns lowercase string"):
    assertEquals(PokerFormatting.heroModeLabel(HeroMode.Adaptive), "adaptive")
    assertEquals(PokerFormatting.heroModeLabel(HeroMode.Gto), "gto")
```

- [ ] **Step 4: Implement PokerFormatting**

```scala
package sicfun.holdem.types

import java.util.Locale

/** Shared formatting utilities for poker actions, chip amounts, and mode labels.
  *
  * Consolidates duplicated private helpers from AcpcMatchRunner (9 copies of renderAction),
  * SlumbotMatchRunner, PlayingHall, AdvisorSession, AlwaysOnDecisionLoop, etc.
  */
private[holdem] object PokerFormatting:

  def renderAction(action: PokerAction): String =
    action match
      case PokerAction.Fold          => "Fold"
      case PokerAction.Check         => "Check"
      case PokerAction.Call          => "Call"
      case PokerAction.Raise(amount) => s"Raise:${fmtDouble(amount, 2)}"

  def fmtDouble(value: Double, digits: Int): String =
    String.format(Locale.ROOT, s"%.${digits}f", java.lang.Double.valueOf(value))

  def roundedChips(value: Double): Int =
    math.max(50, math.round(value / 50.0).toInt * 50)

  def heroModeLabel(mode: HeroMode): String =
    mode match
      case HeroMode.Adaptive => "adaptive"
      case HeroMode.Gto      => "gto"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `sbt "testOnly sicfun.holdem.types.PokerFormattingTest"`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/main/scala/sicfun/holdem/types/PokerFormatting.scala \
  src/test/scala/sicfun/holdem/types/PokerFormattingTest.scala
git commit -m "refactor: extract PokerFormatting utilities (renderAction, fmtDouble, roundedChips, heroModeLabel)"
```

---

## Task 3: Extract HeroDecisionPipeline

**Files:**
- Create: `src/main/scala/sicfun/holdem/engine/HeroDecisionPipeline.scala`
- Create: `src/test/scala/sicfun/holdem/engine/HeroDecisionPipelineTest.scala`

**Depends on:** Task 1 (HeroMode), Task 2 (PokerFormatting.roundedChips)

- [ ] **Step 1: Write failing tests for RaiseSizingContext + legalRaiseCandidates**

```scala
package sicfun.holdem.engine

import munit.FunSuite
import sicfun.holdem.types.*

class HeroDecisionPipelineTest extends FunSuite:

  // Helper to build a RaiseSizingContext for testing
  private def ctx(
      stackRemaining: Int,
      toCall: Int,
      lastBetSize: Int,
      pot: Int,
      street: Street,
      streetLastBetTo: Int,
      bigBlindChips: Int = 100
  ): HeroDecisionPipeline.RaiseSizingContext =
    HeroDecisionPipeline.RaiseSizingContext(
      stackRemainingChips = stackRemaining,
      toCallChips = toCall,
      lastBetSizeChips = lastBetSize,
      potChips = pot,
      currentStreet = street,
      streetLastBetToChips = streetLastBetTo,
      bigBlindChips = bigBlindChips
    )

  test("legalRaiseCandidates returns empty when no room to raise"):
    val result = HeroDecisionPipeline.legalRaiseCandidates(
      ctx(stackRemaining = 100, toCall = 100, lastBetSize = 100, pot = 300, street = Street.Flop, streetLastBetTo = 200)
    )
    assertEquals(result, Vector.empty[PokerAction])

  test("legalRaiseCandidates preflop facing open returns 1.5bb and 2bb sizes"):
    val result = HeroDecisionPipeline.legalRaiseCandidates(
      ctx(stackRemaining = 10000, toCall = 100, lastBetSize = 100, pot = 300, street = Street.Preflop, streetLastBetTo = 100)
    )
    assertEquals(result.length, 2)
    // 150 chips = 1.5bb and 200 chips = 2.0bb
    assertEquals(result(0), PokerAction.Raise(1.5))
    assertEquals(result(1), PokerAction.Raise(2.0))

  test("legalRaiseCandidates preflop unraised returns 2bb and 3bb sizes"):
    val result = HeroDecisionPipeline.legalRaiseCandidates(
      ctx(stackRemaining = 10000, toCall = 0, lastBetSize = 0, pot = 150, street = Street.Preflop, streetLastBetTo = 100)
    )
    assertEquals(result.length, 2)
    assertEquals(result(0), PokerAction.Raise(2.0))
    assertEquals(result(1), PokerAction.Raise(3.0))

  test("legalRaiseCandidates postflop check-to-act returns 50% and 75% pot sizes"):
    val result = HeroDecisionPipeline.legalRaiseCandidates(
      ctx(stackRemaining = 10000, toCall = 0, lastBetSize = 0, pot = 400, street = Street.Flop, streetLastBetTo = 0)
    )
    assertEquals(result.length, 2)
    // 50% of 400 = 200 chips = 2.0bb, 75% of 400 = 300 chips = 3.0bb
    assertEquals(result(0), PokerAction.Raise(2.0))
    assertEquals(result(1), PokerAction.Raise(3.0))

  test("legalRaiseCandidates postflop facing bet returns min legal and 75% pot"):
    val result = HeroDecisionPipeline.legalRaiseCandidates(
      ctx(stackRemaining = 10000, toCall = 200, lastBetSize = 200, pot = 600, street = Street.Flop, streetLastBetTo = 400)
    )
    assertEquals(result.length, 2)
    // minIncrement = max(100, 200) = 200 chips = 2.0bb, 75% of 600 = 450 rounded to 450 = 4.5bb
    assertEquals(result(0), PokerAction.Raise(2.0))
    assertEquals(result(1), PokerAction.Raise(4.5))

  test("heroCandidates when no call needed returns Check + raises"):
    val raises = Vector(PokerAction.Raise(2.0), PokerAction.Raise(3.0))
    val result = HeroDecisionPipeline.heroCandidates(toCallChips = 0, raises = raises)
    assertEquals(result, Vector(PokerAction.Check, PokerAction.Raise(2.0), PokerAction.Raise(3.0)))

  test("heroCandidates when facing bet returns Fold + Call + raises"):
    val raises = Vector(PokerAction.Raise(4.0))
    val result = HeroDecisionPipeline.heroCandidates(toCallChips = 200, raises = raises)
    assertEquals(result, Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(4.0)))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `sbt "testOnly sicfun.holdem.engine.HeroDecisionPipelineTest"`
Expected: FAIL — `HeroDecisionPipeline` does not exist.

- [ ] **Step 3: Implement HeroDecisionPipeline**

```scala
package sicfun.holdem.engine

import sicfun.holdem.types.*
import sicfun.holdem.cfr.{HoldemCfrConfig, HoldemCfrSolver}
import sicfun.holdem.equity.TableRanges

import java.util.Random

/** Shared hero decision pipeline used by match runners (ACPC, Slumbot).
  *
  * Extracted from AcpcMatchRunner.Runner and SlumbotMatchRunner.Runner where
  * decideHero, legalRaiseCandidates, heroCandidates, and newAdaptiveEngine
  * were character-for-character identical.
  */
private[holdem] object HeroDecisionPipeline:

  /** Protocol-neutral context for raise sizing. Both AcpcActionCodec.ParsedActionState and
    * SlumbotActionCodec.ParsedActionState provide these fields.
    */
  final case class RaiseSizingContext(
      stackRemainingChips: Int,
      toCallChips: Int,
      lastBetSizeChips: Int,
      potChips: Int,
      currentStreet: Street,
      streetLastBetToChips: Int,
      bigBlindChips: Int
  )

  /** Context for a hero decision. Bundles all parameters needed by both Adaptive and GTO modes. */
  final case class HeroDecisionContext(
      hero: HoleCards,
      state: GameState,
      folds: Vector[PreflopFold],
      tableRanges: TableRanges,
      villainPos: Position,
      observations: Vector[VillainObservation],
      candidates: Vector[PokerAction],
      engine: RealTimeAdaptiveEngine,
      actionModel: PokerActionModel,
      bunchingTrials: Int,
      cfrConfig: HoldemCfrConfig,
      rng: Random
  )

  /** Decide hero action using the specified mode.
    *
    * Adaptive mode: delegates to engine.decide with 1ms budget.
    * GTO mode: runs RangeInferenceEngine.inferPosterior → HoldemCfrSolver.solveShallowDecisionPolicy.
    */
  def decideHero(mode: HeroMode, ctx: HeroDecisionContext): PokerAction =
    mode match
      case HeroMode.Adaptive =>
        ctx.engine
          .decide(
            hero = ctx.hero,
            state = ctx.state,
            folds = ctx.folds,
            villainPos = ctx.villainPos,
            observations = ctx.observations,
            candidateActions = ctx.candidates,
            decisionBudgetMillis = Some(1L),
            rng = new Random(ctx.rng.nextLong())
          )
          .decision
          .recommendation
          .bestAction
      case HeroMode.Gto =>
        val posterior = RangeInferenceEngine
          .inferPosterior(
            hero = ctx.hero,
            board = ctx.state.board,
            folds = ctx.folds,
            tableRanges = ctx.tableRanges,
            villainPos = ctx.villainPos,
            observations = ctx.observations,
            actionModel = ctx.actionModel,
            bunchingTrials = ctx.bunchingTrials,
            rng = new Random(ctx.rng.nextLong())
          )
          .posterior
        HoldemCfrSolver
          .solveShallowDecisionPolicy(
            hero = ctx.hero,
            state = ctx.state,
            villainPosterior = posterior,
            candidateActions = ctx.candidates,
            config = ctx.cfrConfig
          )
          .bestAction

  /** Compute legal raise candidates from protocol-neutral sizing context.
    *
    * Raise sizing logic (identical in both ACPC and Slumbot runners):
    * - Preflop facing open (toCall > 0, streetLastBetTo == BB): 1.5x and 2x BB
    * - Preflop unraised (toCall == 0, streetLastBetTo == BB): 2x and 3x BB
    * - Postflop check-to-act (toCall <= 0): 50% pot, 75% pot
    * - Postflop facing bet (toCall > 0): min legal, 75% pot
    * All sizes clamped to [minIncrement, maxIncrement], rounded to nearest 50 chips.
    */
  def legalRaiseCandidates(ctx: RaiseSizingContext): Vector[PokerAction] =
    val remaining = ctx.stackRemainingChips
    val toCall = ctx.toCallChips
    val maxIncrement = remaining - toCall
    if maxIncrement <= 0 then Vector.empty
    else
      val minIncrement =
        math.min(
          maxIncrement,
          if ctx.lastBetSizeChips > 0 then math.max(ctx.bigBlindChips, ctx.lastBetSizeChips)
          else ctx.bigBlindChips
        )
      val rawIncrements =
        if ctx.currentStreet == Street.Preflop && ctx.toCallChips > 0 && ctx.streetLastBetToChips == ctx.bigBlindChips then
          Vector(150, 200)
        else if ctx.currentStreet == Street.Preflop && ctx.toCallChips == 0 && ctx.streetLastBetToChips == ctx.bigBlindChips then
          Vector(200, 300)
        else if ctx.toCallChips <= 0 then
          Vector(
            PokerFormatting.roundedChips(ctx.potChips * 0.50),
            PokerFormatting.roundedChips(ctx.potChips * 0.75)
          )
        else
          Vector(
            minIncrement,
            PokerFormatting.roundedChips(ctx.potChips * 0.75)
          )
      rawIncrements
        .map(value => math.max(minIncrement, math.min(maxIncrement, value)))
        .distinct
        .sorted
        .map(value => PokerAction.Raise(value.toDouble / ctx.bigBlindChips.toDouble))
        .toVector

  /** Build hero action candidates: Check + raises or Fold + Call + raises. */
  def heroCandidates(toCallChips: Int, raises: Vector[PokerAction]): Vector[PokerAction] =
    if toCallChips <= 0 then Vector(PokerAction.Check) ++ raises
    else Vector(PokerAction.Fold, PokerAction.Call) ++ raises

  /** Factory for RealTimeAdaptiveEngine with standard config. */
  def newAdaptiveEngine(
      tableRanges: TableRanges,
      model: PokerActionModel,
      bunchingTrials: Int,
      equityTrials: Int
  ): RealTimeAdaptiveEngine =
    new RealTimeAdaptiveEngine(
      tableRanges = tableRanges,
      actionModel = model,
      bunchingTrials = bunchingTrials,
      defaultEquityTrials = equityTrials,
      minEquityTrials = math.max(8, math.min(equityTrials, equityTrials / 10))
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `sbt "testOnly sicfun.holdem.engine.HeroDecisionPipelineTest"`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/engine/HeroDecisionPipeline.scala \
  src/test/scala/sicfun/holdem/engine/HeroDecisionPipelineTest.scala
git commit -m "feat: extract HeroDecisionPipeline with shared decideHero, raise sizing, engine factory"
```

---

## Task 4: Extract MatchRunnerSupport

**Files:**
- Create: `src/main/scala/sicfun/holdem/runtime/MatchRunnerSupport.scala`
- Create: `src/test/scala/sicfun/holdem/runtime/MatchRunnerSupportTest.scala`

**Depends on:** Task 2 (PokerFormatting.renderAction, fmtDouble)

- [ ] **Step 1: Write failing tests for MatchStatistics**

```scala
package sicfun.holdem.runtime

import munit.FunSuite
import sicfun.holdem.types.*

class MatchRunnerSupportTest extends FunSuite:

  test("MatchStatistics starts at zero"):
    val stats = new MatchRunnerSupport.MatchStatistics()
    val summary = stats.buildSummary(heroMode = HeroMode.Adaptive, modelId = "test", outDir = java.nio.file.Path.of("out"))
    assertEquals(summary.handsPlayed, 0)
    assertEquals(summary.heroNetChips, 0.0)
    assertEquals(summary.heroWins, 0)

  test("MatchStatistics tracks wins and position breakdown"):
    val stats = new MatchRunnerSupport.MatchStatistics()
    stats.recordOutcome(Position.Button, heroNetChips = 150.0)
    stats.recordOutcome(Position.BigBlind, heroNetChips = -100.0)
    stats.recordOutcome(Position.Button, heroNetChips = 0.0)
    val summary = stats.buildSummary(heroMode = HeroMode.Adaptive, modelId = "m1", outDir = java.nio.file.Path.of("out"))
    assertEquals(summary.handsPlayed, 3)
    assertEquals(summary.heroNetChips, 50.0)
    assertEquals(summary.heroWins, 1)
    assertEquals(summary.heroLosses, 1)
    assertEquals(summary.heroTies, 1)
    assertEquals(summary.buttonHands, 2)
    assertEquals(summary.buttonNetChips, 150.0)
    assertEquals(summary.bigBlindHands, 1)
    assertEquals(summary.bigBlindNetChips, -100.0)

  test("MatchStatistics computes bb/100 correctly"):
    val stats = new MatchRunnerSupport.MatchStatistics()
    // Win 200 chips over 100 hands at BB=100 chips → 2.0 bb/100
    for _ <- 1 to 100 do stats.recordOutcome(Position.Button, heroNetChips = 2.0)
    val summary = stats.buildSummary(heroMode = HeroMode.Adaptive, modelId = "m", outDir = java.nio.file.Path.of("out"), bigBlindChips = 100)
    assert(math.abs(summary.heroBbPer100 - 2.0) < 0.001)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `sbt "testOnly sicfun.holdem.runtime.MatchRunnerSupportTest"`
Expected: FAIL — `MatchRunnerSupport` does not exist.

- [ ] **Step 3: Write failing tests for writeSummary and appendDecisionLog**

Append to the test file:

```scala
  test("writeSummary writes expected format"):
    val summary = MatchRunnerSupport.RunSummary(
      handsPlayed = 100, heroNetChips = 250.0, heroBbPer100 = 2.5,
      heroWins = 55, heroTies = 5, heroLosses = 40,
      buttonHands = 50, buttonNetChips = 200.0,
      bigBlindHands = 50, bigBlindNetChips = 50.0,
      heroMode = HeroMode.Adaptive, modelId = "test-model", outDir = java.nio.file.Path.of("out")
    )
    val tmpFile = java.nio.file.Files.createTempFile("summary", ".txt")
    try
      MatchRunnerSupport.writeSummary(tmpFile, "Test Runner", summary)
      val content = java.nio.file.Files.readString(tmpFile)
      assert(content.contains("=== Test Runner ==="))
      assert(content.contains("handsPlayed: 100"))
      assert(content.contains("heroBbPer100: 2.500"))
      assert(content.contains("heroMode: adaptive"))
      assert(content.contains("modelId: test-model"))
    finally
      java.nio.file.Files.deleteIfExists(tmpFile)

  test("appendDecisionLog writes tab-separated fields"):
    val tmpFile = java.nio.file.Files.createTempFile("decisions", ".tsv")
    val writer = java.nio.file.Files.newBufferedWriter(tmpFile)
    try
      val state = GameState(
        street = Street.Flop, pot = 3.0, toCall = 1.0, stackSize = 50.0,
        board = Board.empty, position = Position.Button
      )
      MatchRunnerSupport.appendDecisionLog(
        writer = writer,
        handId = 42,
        decisionIndex = 1,
        state = state,
        candidates = Vector(PokerAction.Fold, PokerAction.Call),
        chosenAction = PokerAction.Call,
        wireAction = "c"
      )
      writer.flush()
      val content = java.nio.file.Files.readString(tmpFile)
      assert(content.contains("42\t1\tFlop\tButton\t3.000\t1.000\t50.000\tFold,Call\tCall\tc"))
    finally
      writer.close()
      java.nio.file.Files.deleteIfExists(tmpFile)
```

- [ ] **Step 4: Implement MatchRunnerSupport**

```scala
package sicfun.holdem.runtime

import sicfun.holdem.types.*

import java.io.BufferedWriter
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}

/** Shared infrastructure for match runner statistics, logging, and summaries.
  *
  * Extracted from AcpcMatchRunner.Runner and SlumbotMatchRunner.Runner where
  * recordOutcome, buildSummary, writeSummary, appendDecisionLog, and RunSummary
  * were structurally identical (only differing in chip type: Double vs Int).
  */
private[holdem] object MatchRunnerSupport:

  final case class RunSummary(
      handsPlayed: Int,
      heroNetChips: Double,
      heroBbPer100: Double,
      heroWins: Int,
      heroTies: Int,
      heroLosses: Int,
      buttonHands: Int,
      buttonNetChips: Double,
      bigBlindHands: Int,
      bigBlindNetChips: Double,
      heroMode: HeroMode,
      modelId: String,
      outDir: Path
  )

  /** Mutable match statistics accumulator. Thread-unsafe — used within a single Runner. */
  final class MatchStatistics:
    private var handsPlayed = 0
    private var heroNetChips = 0.0
    private var heroWins = 0
    private var heroTies = 0
    private var heroLosses = 0
    private var buttonHands = 0
    private var buttonNetChips = 0.0
    private var bigBlindHands = 0
    private var bigBlindNetChips = 0.0

    def currentHandsPlayed: Int = handsPlayed

    def recordOutcome(heroPosition: Position, heroNetChips: Double): Unit =
      this.handsPlayed += 1
      this.heroNetChips += heroNetChips
      if heroNetChips > 0.0 then this.heroWins += 1
      else if heroNetChips < 0.0 then this.heroLosses += 1
      else this.heroTies += 1
      heroPosition match
        case Position.Button =>
          this.buttonHands += 1
          this.buttonNetChips += heroNetChips
        case Position.BigBlind =>
          this.bigBlindHands += 1
          this.bigBlindNetChips += heroNetChips
        case other =>
          throw new IllegalStateException(s"unexpected hero position in match runner: $other")

    def currentBbPer100(bigBlindChips: Double): Double =
      if handsPlayed > 0 then (heroNetChips / bigBlindChips / handsPlayed.toDouble) * 100.0
      else 0.0

    def buildSummary(heroMode: HeroMode, modelId: String, outDir: Path, bigBlindChips: Int = 100): RunSummary =
      RunSummary(
        handsPlayed = handsPlayed,
        heroNetChips = heroNetChips,
        heroBbPer100 = currentBbPer100(bigBlindChips.toDouble),
        heroWins = heroWins,
        heroTies = heroTies,
        heroLosses = heroLosses,
        buttonHands = buttonHands,
        buttonNetChips = buttonNetChips,
        bigBlindHands = bigBlindHands,
        bigBlindNetChips = bigBlindNetChips,
        heroMode = heroMode,
        modelId = modelId,
        outDir = outDir
      )

  def writeSummary(path: Path, label: String, summary: RunSummary): Unit =
    val fmt = PokerFormatting.fmtDouble
    val lines = Vector(
      s"=== $label ===",
      s"handsPlayed: ${summary.handsPlayed}",
      s"heroNetChips: ${fmt(summary.heroNetChips, 3)}",
      s"heroBbPer100: ${fmt(summary.heroBbPer100, 3)}",
      s"heroWins: ${summary.heroWins}",
      s"heroTies: ${summary.heroTies}",
      s"heroLosses: ${summary.heroLosses}",
      s"buttonHands: ${summary.buttonHands}",
      s"buttonNetChips: ${fmt(summary.buttonNetChips, 3)}",
      s"bigBlindHands: ${summary.bigBlindHands}",
      s"bigBlindNetChips: ${fmt(summary.bigBlindNetChips, 3)}",
      s"heroMode: ${PokerFormatting.heroModeLabel(summary.heroMode)}",
      s"modelId: ${summary.modelId}"
    )
    Files.write(path, lines.mkString(System.lineSeparator()).getBytes(StandardCharsets.UTF_8))

  def appendDecisionLog(
      writer: BufferedWriter,
      handId: Int,
      decisionIndex: Int,
      state: GameState,
      candidates: Vector[PokerAction],
      chosenAction: PokerAction,
      wireAction: String
  ): Unit =
    val fmt = PokerFormatting.fmtDouble
    val render = PokerFormatting.renderAction
    writer.write(
      Vector(
        handId.toString,
        decisionIndex.toString,
        state.street.toString,
        state.position.toString,
        fmt(state.pot, 3),
        fmt(state.toCall, 3),
        fmt(state.stackSize, 3),
        candidates.map(render).mkString(","),
        render(chosenAction),
        wireAction
      ).mkString("\t")
    )
    writer.newLine()
    writer.flush()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `sbt "testOnly sicfun.holdem.runtime.MatchRunnerSupportTest"`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/MatchRunnerSupport.scala \
  src/test/scala/sicfun/holdem/runtime/MatchRunnerSupportTest.scala
git commit -m "feat: extract MatchRunnerSupport with shared statistics, summary, and decision logging"
```

---

## Task 5: Refactor AcpcMatchRunner.Runner

**Files:**
- Modify: `src/main/scala/sicfun/holdem/runtime/AcpcMatchRunner.scala`
- Test: `src/test/scala/sicfun/holdem/runtime/AcpcMatchRunnerTest.scala`

**Depends on:** Tasks 1-4

- [ ] **Step 1: Run existing tests to establish baseline**

Run: `sbt "testOnly sicfun.holdem.runtime.AcpcMatchRunnerTest"`
Expected: ALL PASS (baseline)

- [ ] **Step 2: Add imports for shared modules**

At the top of `AcpcMatchRunner.scala`, add:
```scala
import sicfun.holdem.types.{HeroMode, PokerFormatting}
import sicfun.holdem.engine.HeroDecisionPipeline
```

- [ ] **Step 3: Replace Runner.decideHero with delegation**

Replace the private `decideHero` method (lines ~783-832) with:

```scala
    private def decideHero(
        hero: HoleCards,
        state: GameState,
        villainPosition: Position,
        villainObservations: Vector[VillainObservation],
        candidates: Vector[PokerAction]
    ): PokerAction =
      HeroDecisionPipeline.decideHero(
        config.heroMode,
        HeroDecisionPipeline.HeroDecisionContext(
          hero = hero,
          state = state,
          folds = folds,
          tableRanges = tableRanges,
          villainPos = villainPosition,
          observations = villainObservations,
          candidates = candidates,
          engine = engine,
          actionModel = artifact.model,
          bunchingTrials = config.bunchingTrials,
          cfrConfig = HoldemCfrConfig(
            iterations = config.cfrIterations,
            maxVillainHands = config.cfrVillainHands,
            equityTrials = config.cfrEquityTrials,
            rngSeed = rng.nextLong()
          ),
          rng = rng
        )
      )
```

- [ ] **Step 4: Replace Runner.legalRaiseCandidates with delegation**

Replace the private `legalRaiseCandidates` method (lines ~843-870) with:

```scala
    private def legalRaiseCandidates(parsed: AcpcActionCodec.ParsedActionState): Vector[PokerAction] =
      HeroDecisionPipeline.legalRaiseCandidates(
        HeroDecisionPipeline.RaiseSizingContext(
          stackRemainingChips = parsed.stackRemainingChips,
          toCallChips = parsed.toCallChips,
          lastBetSizeChips = parsed.lastBetSizeChips,
          potChips = parsed.potChips,
          currentStreet = parsed.currentStreet,
          streetLastBetToChips = parsed.streetLastBetToChips,
          bigBlindChips = AcpcActionCodec.BigBlindChips
        )
      )
```

- [ ] **Step 5: Replace Runner.heroCandidates with delegation**

Replace:
```scala
    private def heroCandidates(parsed: AcpcActionCodec.ParsedActionState): Vector[PokerAction] =
      val raises = legalRaiseCandidates(parsed)
      HeroDecisionPipeline.heroCandidates(toCallChips = parsed.toCallChips, raises = raises)
```

- [ ] **Step 6: Replace Runner statistics with MatchStatistics**

Remove the mutable counter fields (`handsPlayed`, `heroNetChips`, `heroWins`, `heroTies`, `heroLosses`, `buttonHands`, `buttonNetChips`, `bigBlindHands`, `bigBlindNetChips`) and replace with:

```scala
    private val stats = new MatchRunnerSupport.MatchStatistics()
```

Replace the `recordOutcome` method body with:
```scala
    private def recordOutcome(outcome: HandOutcome): Unit =
      stats.recordOutcome(outcome.heroPosition, outcome.heroNetChips)
```

Replace `buildSummary` with:
```scala
    private def buildSummary(): MatchRunnerSupport.RunSummary =
      stats.buildSummary(heroMode = config.heroMode, modelId = modelId, outDir = config.outDir, bigBlindChips = AcpcActionCodec.BigBlindChips)
```

Replace `writeSummary` with:
```scala
    private def writeSummary(summary: MatchRunnerSupport.RunSummary): Unit =
      MatchRunnerSupport.writeSummary(summaryPath, "ACPC Match Runner", summary)
```

- [ ] **Step 7: Replace Runner.appendDecisionLog with delegation**

Replace with:
```scala
    private def appendDecisionLog(
        handNumber: Int,
        decisionIndex: Int,
        state: GameState,
        candidates: Vector[PokerAction],
        chosenAction: PokerAction,
        acpcAction: String
    ): Unit =
      MatchRunnerSupport.appendDecisionLog(
        writer = decisionsWriter,
        handId = handNumber,
        decisionIndex = decisionIndex,
        state = state,
        candidates = candidates,
        chosenAction = chosenAction,
        wireAction = acpcAction
      )
```

- [ ] **Step 8: Replace Runner.newAdaptiveEngine with delegation**

Replace with:
```scala
    private val engine = HeroDecisionPipeline.newAdaptiveEngine(
      tableRanges = tableRanges,
      model = artifact.model,
      bunchingTrials = config.bunchingTrials,
      equityTrials = config.equityTrials
    )
```

- [ ] **Step 9: Replace utility functions with shared imports**

Replace `maybeReport` to use shared formatting:
```scala
    private def maybeReport(): Unit =
      if config.reportEvery > 0 && (stats.currentHandsPlayed % config.reportEvery == 0) then
        println(
          s"[acpc] hands=${stats.currentHandsPlayed} netChips=${PokerFormatting.fmtDouble(stats.currentBbPer100(AcpcActionCodec.BigBlindChips.toDouble) /* recompute */, 3)} mode=${PokerFormatting.heroModeLabel(config.heroMode)} model=$modelId"
        )
```

**Note:** `maybeReport` has ACPC-specific formatting (prefix `[acpc]`), so it stays in the Runner but delegates formatting to PokerFormatting.

Delete these private methods from the outer `AcpcMatchRunner` object scope (they are now in shared modules):
- `roundedChips` (line 1037)
- `renderAction` (line 1040)
- `fmt` (line 1047)
- `heroModeLabel` (line 1050)

Keep `renderAction`/`fmt` call sites updated to use `PokerFormatting.renderAction`/`PokerFormatting.fmtDouble`.

Also delete the private `RunSummary` case class — now using `MatchRunnerSupport.RunSummary`.

- [ ] **Step 10: Run tests to verify no regression**

Run: `sbt "testOnly sicfun.holdem.runtime.AcpcMatchRunnerTest"`
Expected: ALL PASS

- [ ] **Step 11: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/AcpcMatchRunner.scala
git commit -m "refactor: AcpcMatchRunner.Runner delegates to shared decision pipeline and runner support"
```

---

## Task 6: Refactor SlumbotMatchRunner.Runner

**Files:**
- Modify: `src/main/scala/sicfun/holdem/runtime/SlumbotMatchRunner.scala`
- Test: `src/test/scala/sicfun/holdem/runtime/SlumbotActionCodecTest.scala`

**Depends on:** Tasks 1-4

**Procedure:** Identical to Task 5 but for SlumbotMatchRunner. Key differences noted below.

**IMPORTANT — Behavioral changes to document and preserve:**
1. **Int→Double type widening:** Slumbot's private `RunSummary` used `heroNetChips: Int`. The shared `RunSummary` uses `Double`. This is a type change in the return value of `def run()`. The `main()` method's println output will change from `250` to `250.0`. Mitigate by formatting chip output in `main()` with `.toInt` where the original showed integers.
2. **`handsPlayed` source change:** The original Slumbot `buildSummary` used `config.hands` (configured target). The shared `MatchStatistics` counts actual hands completed. This is an intentional correction — if the runner crashes mid-run, the summary should report actual hands played, not the intended target.
3. **Summary file format:** The original Slumbot `writeSummary` printed integer chip amounts (`heroNetChips: 250`). The shared `writeSummary` formats as Double (`heroNetChips: 250.000`). This is acceptable because the summary is human-readable output, not a machine-parsed format.

- [ ] **Step 1: Run existing tests to establish baseline**

Run: `sbt "testOnly sicfun.holdem.runtime.SlumbotActionCodecTest"`
Expected: ALL PASS

- [ ] **Step 2: Add imports for shared modules** (same as Task 5 Step 2)

- [ ] **Step 3: Replace Runner.decideHero** — same delegation pattern as Task 5 Step 3

- [ ] **Step 4: Replace Runner.legalRaiseCandidates** — same as Task 5 Step 4, using `SlumbotActionCodec.BigBlindChips`

- [ ] **Step 5: Replace Runner.heroCandidates** — same as Task 5 Step 5

- [ ] **Step 6: Replace Runner statistics with MatchStatistics**

Key difference: Slumbot's `heroNetChips` is `Int`, so `recordOutcome` call converts:
```scala
    private def recordOutcome(outcome: HandOutcome): Unit =
      stats.recordOutcome(outcome.heroPosition, outcome.winningsChips.toDouble)
```

And `buildSummary` (now uses actual hands played, not `config.hands`):
```scala
    private def buildSummary(): MatchRunnerSupport.RunSummary =
      stats.buildSummary(heroMode = config.heroMode, modelId = modelId, outDir = config.outDir, bigBlindChips = SlumbotActionCodec.BigBlindChips)
```

- [ ] **Step 7: Replace Runner.appendDecisionLog** — same pattern, `apiAction` as `wireAction`

- [ ] **Step 8: Replace Runner.newAdaptiveEngine** — same delegation

- [ ] **Step 9: Replace utility functions and update main() output**

Delete `roundedChips`, `renderAction`, `fmt`, `heroModeLabel`; use `PokerFormatting.*`.

Delete the private `RunSummary` case class — now using `MatchRunnerSupport.RunSummary`.

Update `maybeReport` to use `stats.currentHandsPlayed` and `PokerFormatting.*`.

Update `writeSummary` to delegate: `MatchRunnerSupport.writeSummary(summaryPath, "Slumbot Match Runner", summary)`

Update `main()` println to format `heroNetChips` as integer for backward-compatible console output:
```scala
// In main(), where it prints the summary, format chip amounts as integers:
s"heroNetChips: ${summary.heroNetChips.toInt}"
```

- [ ] **Step 10: Run tests to verify no regression**

Run: `sbt "testOnly sicfun.holdem.runtime.SlumbotActionCodecTest"`
Expected: ALL PASS

- [ ] **Step 11: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/SlumbotMatchRunner.scala
git commit -m "refactor: SlumbotMatchRunner.Runner delegates to shared decision pipeline and runner support"
```

---

## Task 7: Full Verification and Cleanup

**Files:** All modified files

- [ ] **Step 1: Run full test suite**

Run: `sbt test`
Expected: ALL PASS — no regressions anywhere in the project.

- [ ] **Step 2: Verify line count reduction**

Count lines in modified files before/after. Expected reduction:
- `AcpcMatchRunner.scala`: ~120 lines removed (decideHero, legalRaiseCandidates, recordOutcome, buildSummary, writeSummary, appendDecisionLog, newAdaptiveEngine, RunSummary, roundedChips, renderAction, fmt, heroModeLabel)
- `SlumbotMatchRunner.scala`: ~120 lines removed (same set)
- New shared code: ~200 lines across 4 new files
- **Net reduction: ~40 lines**, but more importantly **~200 lines of duplication eliminated**

- [ ] **Step 3: Compile clean check**

Run: `sbt compile 2>&1 | grep -i "warn\|error"`
Expected: No new warnings or errors.

- [ ] **Step 4: Final commit (if any cleanup needed)**

```bash
git add -u
git commit -m "chore: final cleanup after decision pipeline extraction"
```

---

## Post-Plan: Stretch Goals (Optional, Not Required)

These are natural follow-ups if the team wants to continue the cohesion work:

1. **Replace renderAction/fmt in other files** (7 more sites: AdvisorSession, AlwaysOnDecisionLoop, HandHistoryAnalyzer, LiveHandSimulator, PlayingHall, HoldemCfrReport, HandHistoryReviewService) — each just needs `import sicfun.holdem.types.PokerFormatting` and delete the private copy.

2. **Unify HeadsUpSimulator.decideHero** — uses Adaptive-only mode, could delegate to `HeroDecisionPipeline.decideHero(HeroMode.Adaptive, ...)` with a simpler context.

3. **Unify AlwaysOnDecisionLoop.decideForHero** — more complex (integrates opponent memory, retraining), but the core engine.decide call could delegate.

## Future Plans

- **Plan 2: bench/ Utility Consolidation** — extract `BenchSupport` object with shared `card()`, `hole()`, `BatchData`, `loadBatch()`, GPU property constants. ~300 duplicated lines across 15 files.

- **Plan 3: PlayingHall GTO Extraction** — extract `gtoHeroResponds`, `gtoVillainResponds`, `villainResponds` from the 2633-line god object. Unify the near-identical hero/villain GTO methods. Potentially integrate with `HeroDecisionPipeline` as a GTO.Exact mode.
