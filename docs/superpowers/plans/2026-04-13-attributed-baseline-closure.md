# Attributed Baseline (Def 10) Closure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the last `Severity.Structural` gap in `BridgeManifest` by implementing `PosteriorAttributedBaseline` (Def 10) with kernel coupling (Def 18).

**Architecture:** A stateless `PosteriorAttributedBaseline` captures `actionPriors` at construction; per-rival differentiation comes from call-time `rivalState` pattern-matched to `StrategicRivalBelief`. The engine's `buildAttribLikelihoodFn()` is rewritten as a thin wrapper that delegates to a new `buildAttribLikelihoodFromBaseline()` helper. Both call sites (kernel profile + spot polarization) automatically use the attributed baseline.

**Tech Stack:** Scala 3.8.1, munit 1.2.2, SBT

**Spec:** `docs/superpowers/specs/2026-04-13-attributed-baseline-design.md`

---

### Task 1: Widen AttributedBaseline trait + implement PosteriorAttributedBaseline

This task merges the original Tasks 1-2. The trait widening MUST land before
`PosteriorAttributedBaseline` can extend `AttributedBaseline` with the `publicState`
parameter.

**Files:**
- Modify: `src/main/scala/sicfun/holdem/strategic/Baseline.scala:32-39`
- Create: `src/main/scala/sicfun/holdem/strategic/PosteriorAttributedBaseline.scala`
- Create: `src/test/scala/sicfun/holdem/strategic/PosteriorAttributedBaselineTest.scala`
- Modify: `src/test/scala/sicfun/holdem/strategic/bridge/BridgeTest.scala`

- [ ] **Step 1: Widen the trait**

In `src/main/scala/sicfun/holdem/strategic/Baseline.scala`, replace:

```scala
import sicfun.holdem.types.{PokerAction, Street}
```

with:

```scala
import sicfun.holdem.types.PokerAction
```

And replace:

```scala
trait AttributedBaseline:
  def probability(
      cls: StrategicClass,
      action: PokerAction.Category,
      sizing: Option[Sizing],
      street: Street,
      rivalState: RivalBeliefState
  ): Double
```

with:

```scala
trait AttributedBaseline:
  def probability(
      cls: StrategicClass,
      action: PokerAction.Category,
      sizing: Option[Sizing],
      publicState: PublicState,
      rivalState: RivalBeliefState
  ): Double
```

- [ ] **Step 2: Verify compile**

Run: `sbt compile`
Expected: Clean compile. No call site invokes `.probability(...)` on the old signature —
`OpponentModelState.attributedBaseline` is always `None`, and test fixtures never call through.

- [ ] **Step 3: Write the PosteriorAttributedBaseline tests**

```scala
package sicfun.holdem.strategic

import sicfun.holdem.types.{PokerAction, Street}
import sicfun.core.DiscreteDistribution

class PosteriorAttributedBaselineTest extends munit.FunSuite:

  private val priors: Map[(StrategicClass, PokerAction.Category), Double] = {
    import PokerAction.Category.*
    Map(
      (StrategicClass.Value, Fold) -> 0.05, (StrategicClass.Value, Check) -> 0.35,
      (StrategicClass.Value, Call) -> 0.40, (StrategicClass.Value, Raise) -> 0.20,
      (StrategicClass.Bluff, Fold) -> 0.10, (StrategicClass.Bluff, Check) -> 0.10,
      (StrategicClass.Bluff, Call) -> 0.15, (StrategicClass.Bluff, Raise) -> 0.65,
      (StrategicClass.StructuralBluff, Fold) -> 0.05, (StrategicClass.StructuralBluff, Check) -> 0.15,
      (StrategicClass.StructuralBluff, Call) -> 0.30, (StrategicClass.StructuralBluff, Raise) -> 0.50,
      (StrategicClass.Mixed, Fold) -> 0.15, (StrategicClass.Mixed, Check) -> 0.40,
      (StrategicClass.Mixed, Call) -> 0.35, (StrategicClass.Mixed, Raise) -> 0.10
    )
  }

  private val baseline = new PosteriorAttributedBaseline(priors)

  private val uniformBelief = StrategicRivalBelief.uniform

  private val valueBelief = StrategicRivalBelief(DiscreteDistribution(Map(
    StrategicClass.Value -> 0.85,
    StrategicClass.Bluff -> 0.05,
    StrategicClass.StructuralBluff -> 0.05,
    StrategicClass.Mixed -> 0.05
  )))

  // publicState is threaded but unused in v1 — pass a placeholder
  private val pubState: PublicState = {
    import sicfun.holdem.types.{Board, Position}
    val hero = PlayerId("__test__")
    PublicState(
      street = Street.Flop,
      board = Board.empty,
      pot = Chips(100.0),
      stacks = TableMap(
        hero = hero,
        seats = Vector(Seat(hero, Position.SmallBlind, SeatStatus.Active, Chips(500.0)))
      ),
      actionHistory = Vector.empty
    )
  }

  test("uniform posterior returns pi0 unchanged"):
    for
      cls <- StrategicClass.values
      cat <- PokerAction.Category.values
    do
      val expected = priors.getOrElse((cls, cat), 0.25)
      val actual = baseline.probability(cls, cat, None, pubState, uniformBelief)
      assertEqualsDouble(actual, expected, 1e-10,
        s"uniform posterior should return pi0 for ($cls, $cat)")

  test("probabilities sum to 1.0 for each (class, belief)"):
    val beliefs = Vector(uniformBelief, valueBelief)
    for
      cls <- StrategicClass.values
      belief <- beliefs
    do
      val sum = PokerAction.Category.values.map { cat =>
        baseline.probability(cls, cat, None, pubState, belief)
      }.sum
      assertEqualsDouble(sum, 1.0, 1e-10,
        s"sum for $cls should be 1.0, got $sum")

  test("degenerate posterior on Value skews toward Value-typical actions"):
    // Value class: Fold=0.05, Check=0.35, Call=0.40, Raise=0.20
    // Under value-heavy belief, p_pred should emphasize low-Fold, high-Call
    // Uplift for Call should be > 1, uplift for Fold should be < 1
    val callProb = baseline.probability(
      StrategicClass.Mixed, PokerAction.Category.Call, None, pubState, valueBelief)
    val foldProb = baseline.probability(
      StrategicClass.Mixed, PokerAction.Category.Fold, None, pubState, valueBelief)
    val callPi0 = priors((StrategicClass.Mixed, PokerAction.Category.Call))
    val foldPi0 = priors((StrategicClass.Mixed, PokerAction.Category.Fold))
    // Relative shift: call/fold ratio should increase vs pi0 ratio
    assert(callProb / foldProb > callPi0 / foldPi0,
      s"Value-heavy belief should increase call/fold ratio: attributed=${callProb / foldProb}, pi0=${callPi0 / foldPi0}")

  test("non-StrategicRivalBelief returns pi0 unchanged"):
    val dummyRival = new RivalBeliefState:
      def update(signal: ActionSignal, publicState: PublicState): RivalBeliefState = this
    for
      cls <- StrategicClass.values
      cat <- PokerAction.Category.values
    do
      val expected = priors.getOrElse((cls, cat), 0.25)
      val actual = baseline.probability(cls, cat, None, pubState, dummyRival)
      assertEqualsDouble(actual, expected, 1e-10,
        s"non-SRB should return pi0 for ($cls, $cat)")

  test("epsilon floor prevents division by zero"):
    // All priors zero for some action -> p_ref would be zero
    val zeroPriors: Map[(StrategicClass, PokerAction.Category), Double] = {
      import PokerAction.Category.*
      Map(
        (StrategicClass.Value, Fold) -> 0.0, (StrategicClass.Value, Check) -> 0.0,
        (StrategicClass.Value, Call) -> 0.0, (StrategicClass.Value, Raise) -> 1.0,
        (StrategicClass.Bluff, Fold) -> 0.0, (StrategicClass.Bluff, Check) -> 0.0,
        (StrategicClass.Bluff, Call) -> 0.0, (StrategicClass.Bluff, Raise) -> 1.0,
        (StrategicClass.StructuralBluff, Fold) -> 0.0, (StrategicClass.StructuralBluff, Check) -> 0.0,
        (StrategicClass.StructuralBluff, Call) -> 0.0, (StrategicClass.StructuralBluff, Raise) -> 1.0,
        (StrategicClass.Mixed, Fold) -> 0.0, (StrategicClass.Mixed, Check) -> 0.0,
        (StrategicClass.Mixed, Call) -> 0.0, (StrategicClass.Mixed, Raise) -> 1.0
      )
    }
    val zeroBaseline = new PosteriorAttributedBaseline(zeroPriors)
    // Should not throw, and probabilities should still sum to 1
    val sum = PokerAction.Category.values.map { cat =>
      zeroBaseline.probability(StrategicClass.Value, cat, None, pubState, valueBelief)
    }.sum
    assertEqualsDouble(sum, 1.0, 1e-10)
```

- [ ] **Step 4: Implement PosteriorAttributedBaseline**

```scala
package sicfun.holdem.strategic

import sicfun.holdem.types.PokerAction

/** Posterior-predictive attributed baseline (Def 10, §4.1).
  *
  * hat_pi(a | c, x, m) = pi0(a | c) * w(a, m) / Z(c, m)
  *
  * where:
  *   w(a, m)   = p_pred(a | m) / p_ref(a)
  *   p_pred    = sum_{c'} P(c' | m) * pi0(a | c')
  *   p_ref     = (1/|C|) * sum_{c'} pi0(a | c')
  *   Z(c, m)   = sum_{a'} pi0(a' | c) * w(a', m)
  *
  * Stateless over belief: captures only actionPriors (immutable config).
  * Per-rival differentiation from call-time rivalState.
  */
class PosteriorAttributedBaseline(
    actionPriors: Map[(StrategicClass, PokerAction.Category), Double]
) extends AttributedBaseline:

  private val Eps = 1e-10
  private val classes = StrategicClass.values
  private val actions = PokerAction.Category.values
  private val numClasses = classes.length

  private def pi0(cls: StrategicClass, cat: PokerAction.Category): Double =
    actionPriors.getOrElse((cls, cat), 0.25)

  def probability(
      cls: StrategicClass,
      action: PokerAction.Category,
      sizing: Option[Sizing],
      publicState: PublicState,
      rivalState: RivalBeliefState
  ): Double =
    rivalState match
      case srb: StrategicRivalBelief =>
        val posterior = srb.typePosterior
        // w(a, m) for each action
        val weights = actions.map { a =>
          val pPred = math.max(Eps, classes.map(c => posterior.probabilityOf(c) * pi0(c, a)).sum)
          val pRef = math.max(Eps, classes.map(c => pi0(c, a)).sum / numClasses)
          a -> (pPred / pRef)
        }.toMap
        // Z(c, m) = sum_{a'} pi0(a' | c) * w(a', m)
        val z = math.max(Eps, actions.map(a => pi0(cls, a) * weights(a)).sum)
        // hat_pi(a | c, m) = pi0(a | c) * w(a, m) / Z
        pi0(cls, action) * weights(action) / z
      case _ =>
        pi0(cls, action)
```

- [ ] **Step 5: Run tests**

Run: `sbt "testOnly sicfun.holdem.strategic.PosteriorAttributedBaselineTest"`
Expected: All 5 tests PASS.

- [ ] **Step 6: Add compile-time trait check to BridgeTest**

In `src/test/scala/sicfun/holdem/strategic/bridge/BridgeTest.scala`, add after the existing baseline tests:

```scala
  test("AttributedBaseline trait accepts PublicState"):
    val baseline: AttributedBaseline = new PosteriorAttributedBaseline(
      StrategicEngine.defaultActionPriors
    )
    val hero = PlayerId("__test__")
    val pubState = PublicState(
      street = Street.Flop,
      board = Board.empty,
      pot = Chips(100.0),
      stacks = TableMap(
        hero = hero,
        seats = Vector(Seat(hero, Position.SmallBlind, SeatStatus.Active, Chips(500.0)))
      ),
      actionHistory = Vector.empty
    )
    val p = baseline.probability(
      StrategicClass.Value, PokerAction.Category.Call, None, pubState, StrategicRivalBelief.uniform
    )
    assert(p > 0.0 && p <= 1.0)
```

- [ ] **Step 7: Run all strategic + bridge tests**

Run: `sbt "testOnly sicfun.holdem.strategic.*" "testOnly sicfun.holdem.strategic.bridge.*"`
Expected: All pass.

- [ ] **Step 8: Commit**

```bash
git add src/main/scala/sicfun/holdem/strategic/Baseline.scala src/main/scala/sicfun/holdem/strategic/PosteriorAttributedBaseline.scala src/test/scala/sicfun/holdem/strategic/PosteriorAttributedBaselineTest.scala src/test/scala/sicfun/holdem/strategic/bridge/BridgeTest.scala
git commit -m "feat(strategic): widen AttributedBaseline trait, add PosteriorAttributedBaseline (Def 10)"
```

---

### Task 2: Wire attributed baseline into StrategicEngine

**Files:**
- Modify: `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala`
- Modify: `src/test/scala/sicfun/holdem/engine/StrategicEngineTest.scala`

- [ ] **Step 1: Write the failing tests**

These tests verify the rewrite actually uses the attributed baseline, not just that
beliefs shift generically. The key test: under uniform belief, the attributed baseline
produces identical `basePr` to the old `actionPrior(cls, signal.action)` path. Under
non-uniform belief, the attributed baseline produces *different* `basePr` because of the
posterior-predictive uplift — which changes the kernel update result.

Add to `src/test/scala/sicfun/holdem/engine/StrategicEngineTest.scala`:

```scala
  test("attributed baseline: uniform belief produces same posterior as old path"):
    // Record the posterior from the CURRENT code (old path) before rewriting.
    // After the rewrite, the new path under uniform belief must match exactly,
    // because PosteriorAttributedBaseline with uniform belief = pi0.
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand(testHeroCards)

    // Snapshot beliefs before
    val beforeValue = engine.sessionState.rivalBeliefs(PlayerId("v1"))
      .typePosterior.probabilityOf(StrategicClass.Value)

    engine.observeAction(PlayerId("v1"), PokerAction.Raise(50.0), minimalState)
    val afterBelief = engine.sessionState.rivalBeliefs(PlayerId("v1"))

    // Under uniform belief, attributed baseline = pi0 = old path.
    // Raise is Bluff-typical -> posterior shifts toward Bluff.
    assert(afterBelief.typePosterior.probabilityOf(StrategicClass.Bluff) > 0.25,
      "Raise should shift belief toward Bluff class")
    // Value should decrease from 0.25
    assert(afterBelief.typePosterior.probabilityOf(StrategicClass.Value) < beforeValue,
      "Value posterior should decrease after Raise observation")

  test("attributed baseline: non-uniform belief differs from uniform baseline update"):
    // Two engines with identical config. One starts with uniform belief,
    // the other with skewed belief. After the SAME observation, the attributed
    // baseline should cause DIFFERENT posterior updates (the whole point of Def 10).
    val uniformEngine = new StrategicEngine(StrategicEngine.Config())
    uniformEngine.initSession(rivalIds = Vector(PlayerId("v1")))
    uniformEngine.startHand(testHeroCards)

    val skewedEngine = new StrategicEngine(StrategicEngine.Config())
    val skewedBelief = StrategicRivalBelief(DiscreteDistribution(Map(
      StrategicClass.Value -> 0.85,
      StrategicClass.Bluff -> 0.05,
      StrategicClass.StructuralBluff -> 0.05,
      StrategicClass.Mixed -> 0.05
    )))
    skewedEngine.initSession(
      rivalIds = Vector(PlayerId("v1")),
      existingBeliefs = Map(PlayerId("v1") -> skewedBelief)
    )
    skewedEngine.startHand(testHeroCards)

    // Same observation on both
    uniformEngine.observeAction(PlayerId("v1"), PokerAction.Raise(50.0), minimalState)
    skewedEngine.observeAction(PlayerId("v1"), PokerAction.Raise(50.0), minimalState)

    val uniformPost = uniformEngine.sessionState.rivalBeliefs(PlayerId("v1")).typePosterior
    val skewedPost = skewedEngine.sessionState.rivalBeliefs(PlayerId("v1")).typePosterior

    // The posteriors MUST differ because the attributed baseline uses different
    // uplift weights w(a,m) when the pre-update belief differs.
    val uniformBluff = uniformPost.probabilityOf(StrategicClass.Bluff)
    val skewedBluff = skewedPost.probabilityOf(StrategicClass.Bluff)
    assert(math.abs(uniformBluff - skewedBluff) > 1e-6,
      s"Posteriors should differ: uniform Bluff=$uniformBluff, skewed Bluff=$skewedBluff")
```

- [ ] **Step 2: Run tests — first test should pass on current code, second may or may not**

Run: `sbt "testOnly sicfun.holdem.engine.StrategicEngineTest"`
The first test establishes the behavioral baseline. The second test verifies that
per-rival differentiation through the attributed baseline causes measurable posterior
divergence — this is the key regression guard for the wrapper migration.

- [ ] **Step 3: Add `_attributedBaseline` as eager val and rewrite `buildAttribLikelihoodFn`**

In `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala`:

**3a.** After line 21 (`class StrategicEngine(val config: StrategicEngine.Config):`), before
the `private var` block, add:

```scala
  /** Kernel-coupled attributed baseline (Def 10). Config-only, stateless. */
  private val _attributedBaseline: PosteriorAttributedBaseline =
    new PosteriorAttributedBaseline(config.actionPriors)
```

**3b.** Replace `buildAttribLikelihoodFn()` (lines 1035-1050) with:

```scala
  /** Build an attrib likelihood from an AttributedBaseline (Def 18 spec-literal).
    *
    * Transposes from action-space hat_pi(a | c, ...) to class-space posterior
    * via TemperedLikelihood.updatePosterior.
    */
  private def buildAttribLikelihoodFromBaseline(baseline: AttributedBaseline): TemperedLikelihoodFn =
    (signal: ActionSignal, pubState: PublicState, rivalState: RivalBeliefState) =>
      val classes = StrategicClass.values
      val eta = TemperedLikelihood.defaultEta(classes.length)

      val basePr = classes.map { cls =>
        baseline.probability(cls, signal.action, signal.sizing, pubState, rivalState)
      }

      val prior = rivalState match
        case srb: StrategicRivalBelief => classes.map(c => srb.typePosterior.probabilityOf(c))
        case _ => classes.map(c => StrategicRivalBelief.uniform.typePosterior.probabilityOf(c))

      val posterior = TemperedLikelihood.updatePosterior(prior, basePr, eta, config.temperedConfig)
      DiscreteDistribution(classes.zip(posterior).toMap)

  /** Build the attrib tempered likelihood function (Def 18: hat{pi}^{0,S,i}).
    * Thin wrapper: delegates to buildAttribLikelihoodFromBaseline using the engine's
    * PosteriorAttributedBaseline instance.
    */
  private def buildAttribLikelihoodFn(): TemperedLikelihoodFn =
    buildAttribLikelihoodFromBaseline(_attributedBaseline)
```

- [ ] **Step 4: Run tests**

Run: `sbt "testOnly sicfun.holdem.engine.StrategicEngineTest"`
Expected: All tests pass including the two new ones.

- [ ] **Step 5: Run full engine test suite to check for regressions**

Run: `sbt "testOnly sicfun.holdem.engine.*"`
Expected: All engine tests pass (211+).

- [ ] **Step 6: Commit**

```bash
git add src/main/scala/sicfun/holdem/engine/StrategicEngine.scala src/test/scala/sicfun/holdem/engine/StrategicEngineTest.scala
git commit -m "feat(engine): wire PosteriorAttributedBaseline into attrib likelihood (Def 18 kernel coupling)"
```

---

### Task 3: Rework BaselineBridge

**Files:**
- Modify: `src/main/scala/sicfun/holdem/strategic/bridge/BaselineBridge.scala:22-35`
- Modify: `src/test/scala/sicfun/holdem/strategic/bridge/BridgeTest.scala:291-321`

- [ ] **Step 1: Write the new test, update existing tests**

In `src/test/scala/sicfun/holdem/strategic/bridge/BridgeTest.scala`, replace the six `toAttributedBaselines` tests (lines 291-321) with:

```scala
  test("BaselineBridge.toAttributedBaseline returns Approximate with baseline pass-through"):
    val baseline = new PosteriorAttributedBaseline(StrategicEngine.defaultActionPriors)
    val result = BaselineBridge.toAttributedBaseline(baseline)
    result match
      case BridgeResult.Approximate(b, note) =>
        assert(b eq baseline, "should return the same baseline instance")
        assert(note.contains("kernel-coupled"), s"note should mention kernel coupling: $note")
      case other => fail(s"expected Approximate, got $other")

  test("BaselineBridge.toAttributedBaseline fidelity is Approximate"):
    val baseline = new PosteriorAttributedBaseline(StrategicEngine.defaultActionPriors)
    assertEquals(BaselineBridge.toAttributedBaseline(baseline).fidelity, Fidelity.Approximate)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `sbt "testOnly sicfun.holdem.strategic.bridge.BridgeTest"`
Expected: Compilation error — `toAttributedBaseline` does not exist.

- [ ] **Step 3: Implement the rework**

Replace `BaselineBridge.toAttributedBaselines` in `src/main/scala/sicfun/holdem/strategic/bridge/BaselineBridge.scala` with:

```scala
  /** Bridge an AttributedBaseline into the bridge result layer.
    *
    * The bridge annotates fidelity; it no longer transforms the data.
    * The baseline is kernel-coupled via PosteriorAttributedBaseline.
    */
  def toAttributedBaseline(baseline: AttributedBaseline): BridgeResult[AttributedBaseline] =
    BridgeResult.Approximate(baseline, "kernel-coupled posterior-predictive attribution; per-rival via PosteriorAttributedBaseline")
```

- [ ] **Step 4: Run tests**

Run: `sbt "testOnly sicfun.holdem.strategic.bridge.BridgeTest"`
Expected: All pass (old tests removed, new tests pass).

- [ ] **Step 5: Check no other code references the old `toAttributedBaselines`**

Run: `grep -r "toAttributedBaselines" src/`
Expected: No hits. (The old method was only called from tests.)

- [ ] **Step 6: Commit**

```bash
git add src/main/scala/sicfun/holdem/strategic/bridge/BaselineBridge.scala src/test/scala/sicfun/holdem/strategic/bridge/BridgeTest.scala
git commit -m "refactor(bridge): rework BaselineBridge.toAttributedBaseline for kernel-coupled attribution"
```

---

### Task 4: StrategicSnapshot — add `attributionEnabled` field

**Files:**
- Modify: `src/main/scala/sicfun/holdem/strategic/bridge/StrategicSnapshot.scala:45`
- Modify: `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala:920-937` (buildSnapshot)
- Modify: `src/test/scala/sicfun/holdem/engine/StrategicEngineTest.scala`

- [ ] **Step 1: Write the failing test**

Add to `src/test/scala/sicfun/holdem/engine/StrategicEngineTest.scala`:

```scala
  test("buildSnapshot sets attributionEnabled = true"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand(testHeroCards)
    val gs = minimalState
    // Inject a bundle so buildSnapshot has data (matches real DecisionEvaluationBundle ctor)
    engine.injectTestBundle(DecisionEvaluationBundle(
      profileResults = Map.empty,
      robustActionLowerBounds = Array(0.0),
      baselineActionValues = Array(0.5, 0.3),
      baselineValue = 0.5,
      adversarialRootGap = None,
      pointwiseExploitability = None,
      deploymentExploitability = None,
      certification = CertificationResult.LocalRobustScreening(
        rootLosses = Array(0.1),
        budgetEstimate = 0.5,
        withinTolerance = true
      ),
      chainWorldValues = Map.empty,
      notes = Vector("test: attribution snapshot")
    ))
    val snapshot = engine.buildSnapshot(gs, PokerAction.Call)
    assert(snapshot.isDefined, "snapshot should be defined")
    assert(snapshot.get.attributionEnabled, "attributionEnabled should be true")

  test("StrategicSnapshot.build static factory has attributionEnabled = false"):
    val snap = strategicBridge.StrategicSnapshot.build(
      gameState = minimalState,
      heroAction = PokerAction.Call,
      heroEquity = 0.5,
      engineEv = 0.5,
      staticEquity = 0.4,
      hasDrawPotential = false
    )
    assert(!snap.attributionEnabled, "static factory should default to attributionEnabled = false")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `sbt "testOnly sicfun.holdem.engine.StrategicEngineTest"`
Expected: Compilation error — `attributionEnabled` does not exist on `StrategicSnapshot`.

- [ ] **Step 3: Add the field to StrategicSnapshot**

In `src/main/scala/sicfun/holdem/strategic/bridge/StrategicSnapshot.scala`, add after `bridgeFidelityNotes` (line 45):

```scala
    /** Whether the engine used kernel-coupled attributed baselines (Def 10). */
    attributionEnabled: Boolean = false
```

(Note: add a comma after the `bridgeFidelityNotes` line.)

- [ ] **Step 4: Set `attributionEnabled = true` in `buildSnapshot()`**

In `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala`, in the `buildSnapshot` method, add to the `StrategicSnapshot(...)` constructor call (around line 937, after `bridgeFidelityNotes`):

```scala
        attributionEnabled = true
```

(Add comma after the preceding `bridgeFidelityNotes` line.)

- [ ] **Step 5: Run tests**

Run: `sbt "testOnly sicfun.holdem.engine.StrategicEngineTest"`
Expected: All pass including both new tests.

- [ ] **Step 6: Commit**

```bash
git add src/main/scala/sicfun/holdem/strategic/bridge/StrategicSnapshot.scala src/main/scala/sicfun/holdem/engine/StrategicEngine.scala src/test/scala/sicfun/holdem/engine/StrategicEngineTest.scala
git commit -m "feat(bridge): add attributionEnabled field to StrategicSnapshot"
```

---

### Task 5: BridgeManifest downgrade + closure assertion

**Files:**
- Modify: `src/main/scala/sicfun/holdem/strategic/bridge/BridgeManifest.scala:45`
- Modify: `src/test/scala/sicfun/holdem/strategic/FormalClosureValidationTest.scala`
- Modify: `src/test/scala/sicfun/holdem/strategic/bridge/BridgeTest.scala`

- [ ] **Step 1: Write the closure assertion test**

Add to `src/test/scala/sicfun/holdem/strategic/FormalClosureValidationTest.scala`, after the DeltaVocabulary test:

```scala
  test("BridgeManifest: zero Structural severity gaps remain (full formal closure)"):
    import bridge.BridgeManifest
    val gaps = BridgeManifest.structuralGaps
    assertEquals(gaps.size, 0,
      s"Expected zero structural gaps, found: ${gaps.map(e => s"${e.formalObject} (${e.specDef})").mkString(", ")}")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly sicfun.holdem.strategic.FormalClosureValidationTest"`
Expected: FAIL — `AttributedBaseline` is still `Severity.Structural`.

- [ ] **Step 3: Downgrade the manifest entry**

In `src/main/scala/sicfun/holdem/strategic/bridge/BridgeManifest.scala`, replace line 45:

```scala
    BridgeEntry("AttributedBaseline",       "Def 10", Fidelity.Approximate, Severity.Structural,  "requires kernel decomposition"),
```

with:

```scala
    BridgeEntry("AttributedBaseline",       "Def 10", Fidelity.Approximate, Severity.Behavioral,  "kernel-coupled posterior-predictive attribution; per-rival via PosteriorAttributedBaseline"),
```

- [ ] **Step 4: Run closure test**

Run: `sbt "testOnly sicfun.holdem.strategic.FormalClosureValidationTest"`
Expected: All pass — zero structural gaps.

- [ ] **Step 5: Update BridgeTest structural gap expectations**

In `src/test/scala/sicfun/holdem/strategic/bridge/BridgeTest.scala`, the existing `structuralGaps` tests should now return empty. Verify:

Run: `sbt "testOnly sicfun.holdem.strategic.bridge.BridgeTest"`
Expected: All pass. The `structuralGaps` tests check "contains only Structural severity entries" and "is a subset of all entries" — both still valid with an empty result.

- [ ] **Step 6: Run full regression suite**

Run: `sbt test`
Expected: All 1376+ tests pass. No behavioral change for uniform-prior rivals (attributed baseline = real baseline under uniform beliefs).

- [ ] **Step 7: Commit**

```bash
git add src/main/scala/sicfun/holdem/strategic/bridge/BridgeManifest.scala src/test/scala/sicfun/holdem/strategic/FormalClosureValidationTest.scala
git commit -m "feat(bridge): downgrade AttributedBaseline to Behavioral — zero structural gaps remaining"
```

---

## Post-Implementation Checklist

After all 5 tasks are complete:

1. **Zero structural gaps:** `BridgeManifest.structuralGaps` returns empty vector
2. **No regressions:** Full `sbt test` passes (1376+ tests)
3. **New tests:** ~10 new tests across 4 files (`PosteriorAttributedBaselineTest`, `StrategicEngineTest`, `BridgeTest`, `FormalClosureValidationTest`)
4. **Spec alignment:** All 6 components from spec §4.1-4.6 implemented (Tasks 1-2 merged)
5. **No unused code:** Old `toAttributedBaselines` removed, `buildAttribLikelihoodFn()` is thin wrapper
6. **No unnecessary nullability:** `_attributedBaseline` is an eager `val`, not `var | Null`
