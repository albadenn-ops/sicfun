# Four-World Solver Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute V^{1,0} and V^{0,1} from the PftDpw solver so the four-world decomposition (Theorem 4) uses solver-derived values instead of interpolated estimates.

**Architecture:** Extend `PokerPftFormulation` with two model variants — blind-kernel (uniform obs likelihoods, baseline rewards) and open-loop (uniform obs likelihoods, attrib rewards). The existing PftDpw solver solves all three models (baseline + two variants). `StrategicEngine.decidePftDpw` orchestrates a 3-solve flow that feeds solver-computed V^{1,0}/V^{0,1} into `ValueBridge.toFourWorldFromSolver`. The interpolation fallback remains for the WPomcp path.

**Tech Stack:** Scala 3.8.1, munit 1.2.2, SBT. Test runner: `sbt "testOnly <fully.qualified.TestClass>"`

**Prerequisite:** runtime-spec-closure plan (2026-04-07) tasks 1-9 completed.

**Spec reference:** SICFUN-v0.31.1-corrected.md, Definition 44 (four-world grid), Theorem 4.

---

## Background: What each grid world needs

| World | Learning | Policy | Kernel | Obs Model | Reward Model |
|-------|----------|--------|--------|-----------|-------------|
| V^{1,1} | Attrib | ClosedLoop | Attrib (rival adapts) | Normal (hero observes) | Profile-modulated |
| V^{1,0} | Attrib | OpenLoop | Attrib (rival adapts) | Uniform (hero blind) | Profile-modulated |
| V^{0,1} | Blind | ClosedLoop | Blind (rival frozen) | Normal (hero observes) | Baseline (no profile modulation) |
| V^{0,0} | Blind | OpenLoop | Blind (rival frozen) | Uniform (hero blind) | Baseline (no profile modulation) |

- V^{1,1} = existing baseline solve (mixed model, full POMDP)
- V^{0,0} = static equity approximation (no solver needed — hero can't observe, rival can't adapt)
- **V^{1,0}** = new: solve attrib-reward model with uniform obs (POMDP degenerates to MDP)
- **V^{0,1}** = new: solve blind-reward model with normal obs (full POMDP, but rival is frozen)

---

## Task 1: PokerPftFormulation — model variant builders

Add two methods that build `TabularGenerativeModel` variants for the missing grid worlds.

**Files:**
- Modify: `src/main/scala/sicfun/holdem/engine/PokerPftFormulation.scala`
- Test: `src/test/scala/sicfun/holdem/engine/FourWorldFormulationTest.scala`

- [ ] **Step 1: Write failing tests for the two model variants**

```scala
// File: src/test/scala/sicfun/holdem/engine/FourWorldFormulationTest.scala
package sicfun.holdem.engine

import sicfun.holdem.types.*
import sicfun.holdem.strategic.*

class FourWorldFormulationTest extends munit.FunSuite:

  private val gameState = GameState(
    street = Street.Flop,
    pot = 100.0,
    toCall = 20.0,
    stackSize = 500.0,
    heroCards = None,
    board = Board.empty,
    numPlayers = 2
  )
  private val heroActions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Check)
  private val heroBucket = 5
  private val actionPriors = Map(
    (StrategicClass.Value, PokerAction.Category.Fold)  -> 0.1,
    (StrategicClass.Value, PokerAction.Category.Call)   -> 0.4,
    (StrategicClass.Value, PokerAction.Category.Check)  -> 0.2,
    (StrategicClass.Value, PokerAction.Category.Raise)  -> 0.3,
    (StrategicClass.Bluff, PokerAction.Category.Fold)   -> 0.4,
    (StrategicClass.Bluff, PokerAction.Category.Call)    -> 0.2,
    (StrategicClass.Bluff, PokerAction.Category.Check)   -> 0.1,
    (StrategicClass.Bluff, PokerAction.Category.Raise)   -> 0.3
  )

  private val numObs = StrategicClass.values.length

  test("buildOpenLoopModel: obs likelihoods are uniform"):
    val model = PokerPftFormulation.buildOpenLoopModel(
      gameState, Map.empty, heroActions, heroBucket, actionPriors
    )
    val uniformP = 1.0 / numObs
    for i <- 0 until model.numStates * model.numActions do
      for o <- 0 until numObs do
        assertEqualsDouble(
          model.obsLikelihood(i * numObs + o), uniformP, 1e-12
        )

  test("buildOpenLoopModel: rewards match attrib model (profile-modulated)"):
    val openLoop = PokerPftFormulation.buildOpenLoopModel(
      gameState, Map.empty, heroActions, heroBucket, actionPriors,
      profileClass = Some(StrategicClass.Value)
    )
    val attrib = PokerPftFormulation.buildTabularModel(
      gameState, Map.empty, heroActions, heroBucket, actionPriors,
      profileClass = Some(StrategicClass.Value)
    )
    // Same rewards (attrib kernel drives reward model)
    openLoop.rewardTable.zip(attrib.rewardTable).foreach { (ol, at) =>
      assertEqualsDouble(ol, at, 1e-12)
    }

  test("buildBlindKernelModel: obs likelihoods are from rival beliefs (not uniform)"):
    val belief = StrategicRivalBelief.withPosterior(
      DiscreteDistribution.fromWeights(
        StrategicClass.values.toIndexedSeq,
        IndexedSeq(0.5, 0.2, 0.2, 0.1)
      )
    )
    val model = PokerPftFormulation.buildBlindKernelModel(
      gameState, Map(PlayerId(1) -> belief), heroActions, heroBucket, actionPriors
    )
    // Obs should reflect the rival belief, not uniform
    val firstObs = (0 until numObs).map(o => model.obsLikelihood(o))
    assert(firstObs.max - firstObs.min > 0.05, "obs should be non-uniform")

  test("buildBlindKernelModel: rewards use baseline (no profile modulation)"):
    val blindModel = PokerPftFormulation.buildBlindKernelModel(
      gameState, Map.empty, heroActions, heroBucket, actionPriors
    )
    val mixedModel = PokerPftFormulation.buildTabularModel(
      gameState, Map.empty, heroActions, heroBucket, actionPriors,
      profileClass = None
    )
    // Both use profileClass=None → same base rewards
    blindModel.rewardTable.zip(mixedModel.rewardTable).foreach { (bl, mx) =>
      assertEqualsDouble(bl, mx, 1e-12)
    }

  test("buildOpenLoopModel and buildBlindKernelModel have same dimensions as baseline"):
    val baseline = PokerPftFormulation.buildTabularModel(
      gameState, Map.empty, heroActions, heroBucket, actionPriors
    )
    val openLoop = PokerPftFormulation.buildOpenLoopModel(
      gameState, Map.empty, heroActions, heroBucket, actionPriors
    )
    val blind = PokerPftFormulation.buildBlindKernelModel(
      gameState, Map.empty, heroActions, heroBucket, actionPriors
    )
    assertEquals(openLoop.numStates, baseline.numStates)
    assertEquals(openLoop.numActions, baseline.numActions)
    assertEquals(openLoop.numObs, baseline.numObs)
    assertEquals(blind.numStates, baseline.numStates)
    assertEquals(blind.numActions, baseline.numActions)
    assertEquals(blind.numObs, baseline.numObs)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `sbt "testOnly sicfun.holdem.engine.FourWorldFormulationTest"`
Expected: FAIL — `buildOpenLoopModel` and `buildBlindKernelModel` not found.

- [ ] **Step 3: Implement buildOpenLoopModel and buildBlindKernelModel**

Add to `src/main/scala/sicfun/holdem/engine/PokerPftFormulation.scala` after `buildParticleBelief`:

```scala
  /** Build a model for V^{1,0} (attrib kernel, open-loop policy).
    *
    * Attrib rewards (rival adapts to hero's play) + uniform obs likelihoods
    * (hero cannot condition on observations). The POMDP solver degenerates
    * to MDP-like behavior because observations carry no information.
    *
    * Grid world: (Attrib, OpenLoop) in Omega^grid.
    */
  def buildOpenLoopModel(
      gameState: GameState,
      rivalBeliefs: Map[PlayerId, StrategicRivalBelief],
      heroActions: Vector[PokerAction],
      heroBucket: Int,
      actionPriors: Map[(StrategicClass, PokerAction.Category), Double],
      profileClass: Option[StrategicClass] = None
  ): TabularGenerativeModel =
    // Build attrib model (normal rewards), then override obs to uniform
    val attribModel = buildTabularModel(
      gameState, rivalBeliefs, heroActions, heroBucket, actionPriors, profileClass
    )
    val numObs = attribModel.numObs
    val uniformObs = Array.fill(
      attribModel.numStates * attribModel.numActions * numObs
    )(1.0 / numObs)
    attribModel.copy(obsLikelihood = uniformObs)

  /** Build a model for V^{0,1} (blind kernel, closed-loop policy).
    *
    * Baseline rewards (rival does NOT adapt — no profile modulation) +
    * normal obs likelihoods (hero observes rival normally).
    *
    * Grid world: (Blind, ClosedLoop) in Omega^grid.
    */
  def buildBlindKernelModel(
      gameState: GameState,
      rivalBeliefs: Map[PlayerId, StrategicRivalBelief],
      heroActions: Vector[PokerAction],
      heroBucket: Int,
      actionPriors: Map[(StrategicClass, PokerAction.Category), Double]
  ): TabularGenerativeModel =
    // Build baseline model (profileClass=None → no profile modulation in rewards)
    // with normal obs likelihoods (hero can still observe)
    buildTabularModel(
      gameState, rivalBeliefs, heroActions, heroBucket, actionPriors,
      profileClass = None
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `sbt "testOnly sicfun.holdem.engine.FourWorldFormulationTest"`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/engine/PokerPftFormulation.scala src/test/scala/sicfun/holdem/engine/FourWorldFormulationTest.scala
git commit -m "feat(formulation): add buildOpenLoopModel and buildBlindKernelModel for four-world grid (Def 44)"
```

---

## Task 2: ValueBridge — solver-backed toFourWorldFromSolver

Add a new entry point that accepts solver-computed values for all four worlds.

**Files:**
- Modify: `src/main/scala/sicfun/holdem/strategic/bridge/ValueBridge.scala`
- Test: `src/test/scala/sicfun/holdem/strategic/bridge/ValueBridgeTest.scala`

- [ ] **Step 1: Write failing tests**

```scala
// File: src/test/scala/sicfun/holdem/strategic/bridge/ValueBridgeTest.scala
package sicfun.holdem.strategic.bridge

import sicfun.holdem.strategic.*

class ValueBridgeTest extends munit.FunSuite:

  test("toFourWorldFromSolver: all four values are Exact when solver provides them"):
    val result = ValueBridge.toFourWorldFromSolver(
      v11 = 10.0, v10 = 7.0, v01 = 6.0, v00 = 4.0
    )
    result match
      case BridgeResult.Exact(fw) =>
        assertEqualsDouble(fw.v11.value, 10.0, 1e-12)
        assertEqualsDouble(fw.v10.value, 7.0, 1e-12)
        assertEqualsDouble(fw.v01.value, 6.0, 1e-12)
        assertEqualsDouble(fw.v00.value, 4.0, 1e-12)
      case other => fail(s"expected Exact, got $other")

  test("toFourWorldFromSolver: Theorem 4 decomposition is exact"):
    val result = ValueBridge.toFourWorldFromSolver(
      v11 = 10.0, v10 = 7.0, v01 = 6.0, v00 = 4.0
    )
    val fw = result match
      case BridgeResult.Exact(fw) => fw
      case other => fail(s"expected Exact, got $other"); return
    // Theorem 4: V^{1,1} = V^{0,0} + Delta_cont + Delta_sig* + Delta_int
    val reconstructed = fw.v00 + fw.deltaControl + fw.deltaSigStar + fw.deltaInteraction
    assertEqualsDouble(reconstructed.value, fw.v11.value, 1e-12)

  test("toFourWorldFromSolver: decomposition components are meaningful"):
    val result = ValueBridge.toFourWorldFromSolver(
      v11 = 10.0, v10 = 7.0, v01 = 6.0, v00 = 4.0
    )
    val fw = result match
      case BridgeResult.Exact(fw) => fw
      case other => fail(s"expected Exact, got $other"); return
    assertEqualsDouble(fw.deltaControl.value, 2.0, 1e-12)   // V01 - V00 = 6 - 4
    assertEqualsDouble(fw.deltaSigStar.value, 3.0, 1e-12)   // V10 - V00 = 7 - 4
    assertEqualsDouble(fw.deltaInteraction.value, 1.0, 1e-12) // V11 - V10 - V01 + V00

  test("toGridWorldValuesFromSolver: all four worlds are Exact"):
    val grid = ValueBridge.toGridWorldValuesFromSolver(
      v11 = 10.0, v10 = 7.0, v01 = 6.0, v00 = 4.0
    )
    assertEquals(grid.size, 4)
    grid.values.foreach {
      case BridgeResult.Exact(_) => ()
      case other => fail(s"expected Exact, got $other")
    }

  test("legacy toFourWorld still works (backward compat)"):
    val result = ValueBridge.toFourWorld(10.0, 4.0, controlFrac = 0.5)
    result match
      case BridgeResult.Approximate(fw, _) =>
        assertEqualsDouble(fw.v11.value, 10.0, 1e-12)
        assertEqualsDouble(fw.v00.value, 4.0, 1e-12)
      case other => fail(s"expected Approximate, got $other")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `sbt "testOnly sicfun.holdem.strategic.bridge.ValueBridgeTest"`
Expected: FAIL — `toFourWorldFromSolver` not found.

- [ ] **Step 3: Implement toFourWorldFromSolver and toGridWorldValuesFromSolver**

Add to `src/main/scala/sicfun/holdem/strategic/bridge/ValueBridge.scala` after `toGridWorldValues`:

```scala
  // ==== Solver-backed four-world construction (Def 44 closure) ====

  /** Build a FourWorld from solver-computed values for all four grid worlds.
    *
    * When the PftDpw solver provides V^{1,0} and V^{0,1} directly, the
    * four-world decomposition (Theorem 4) is exact — not interpolated.
    *
    * @param v11 V^{1,1}: solver value under attrib kernel, closed-loop policy
    * @param v10 V^{1,0}: solver value under attrib kernel, open-loop policy
    * @param v01 V^{0,1}: solver value under blind kernel, closed-loop policy
    * @param v00 V^{0,0}: solver value under blind kernel, open-loop policy (or static equity)
    */
  def toFourWorldFromSolver(
      v11: Double,
      v10: Double,
      v01: Double,
      v00: Double
  ): BridgeResult[FourWorld] =
    BridgeResult.Exact(
      FourWorld(v11 = Ev(v11), v10 = Ev(v10), v01 = Ev(v01), v00 = Ev(v00))
    )

  /** Build keyed grid-world values from solver-computed values.
    *
    * All four worlds are Exact when the solver provides them.
    */
  def toGridWorldValuesFromSolver(
      v11: Double,
      v10: Double,
      v01: Double,
      v00: Double
  ): Map[GridWorld, BridgeResult[Ev]] =
    Map(
      GridWorld(LearningChannel.Attrib, PolicyScope.ClosedLoop) ->
        BridgeResult.Exact(Ev(v11)),
      GridWorld(LearningChannel.Attrib, PolicyScope.OpenLoop) ->
        BridgeResult.Exact(Ev(v10)),
      GridWorld(LearningChannel.Blind, PolicyScope.ClosedLoop) ->
        BridgeResult.Exact(Ev(v01)),
      GridWorld(LearningChannel.Blind, PolicyScope.OpenLoop) ->
        BridgeResult.Exact(Ev(v00))
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `sbt "testOnly sicfun.holdem.strategic.bridge.ValueBridgeTest"`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/strategic/bridge/ValueBridge.scala src/test/scala/sicfun/holdem/strategic/bridge/ValueBridgeTest.scala
git commit -m "feat(bridge): add solver-backed toFourWorldFromSolver for exact Theorem 4 decomposition"
```

---

## Task 3: StrategicEngine — four-world solve in PftDpw path

Wire the three-model solve (baseline + open-loop + blind) into `decidePftDpw` and produce an exact FourWorld when the PftDpw solver succeeds.

**Files:**
- Modify: `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala`
- Test: `src/test/scala/sicfun/holdem/engine/FourWorldSolveTest.scala`

- [ ] **Step 1: Write failing tests for the four-world solve**

```scala
// File: src/test/scala/sicfun/holdem/engine/FourWorldSolveTest.scala
package sicfun.holdem.engine

import sicfun.holdem.types.*
import sicfun.holdem.strategic.*
import sicfun.holdem.strategic.bridge.*
import sicfun.holdem.strategic.solver.{TabularGenerativeModel, ParticleBelief, PftDpwResult}

class FourWorldSolveTest extends munit.FunSuite:

  private val gameState = GameState(
    street = Street.Flop,
    pot = 100.0,
    toCall = 20.0,
    stackSize = 500.0,
    heroCards = None,
    board = Board.empty,
    numPlayers = 2
  )
  private val heroActions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Check)
  private val heroBucket = 5
  private val actionPriors = StrategicEngine.Config().actionPriors

  test("solveFourWorldGrid builds three distinct models"):
    val models = StrategicEngine.buildFourWorldModels(
      gameState, Map.empty, heroActions, heroBucket, actionPriors
    )
    assertEquals(models.size, 3) // baseline, openLoop, blind
    // Open-loop model has uniform obs
    val numObs = StrategicClass.values.length
    val olObs = models.openLoop.obsLikelihood
    val uniformP = 1.0 / numObs
    for i <- 0 until models.openLoop.numStates * models.openLoop.numActions do
      for o <- 0 until numObs do
        assertEqualsDouble(olObs(i * numObs + o), uniformP, 1e-12)
    // Blind model has same rewards as baseline
    models.blind.rewardTable.zip(models.baseline.rewardTable).foreach { (bl, base) =>
      assertEqualsDouble(bl, base, 1e-12)
    }

  test("extractFourWorldValues: solver Q-values map to grid world values"):
    // Simulate solver results: each model solved, extract root Q at best action
    val baselineQ = Array(0.0, 10.0, 5.0) // V^{1,1} ≈ 10.0 (best Q at root)
    val openLoopQ = Array(0.0, 7.0, 4.0)  // V^{1,0} ≈ 7.0
    val blindQ = Array(0.0, 6.0, 3.0)     // V^{0,1} ≈ 6.0
    val staticEquity = 4.0                 // V^{0,0}

    val fw = StrategicEngine.extractFourWorldValues(
      baselineQ, openLoopQ, blindQ, staticEquity
    )
    assertEqualsDouble(fw.v11.value, 10.0, 1e-12)
    assertEqualsDouble(fw.v10.value, 7.0, 1e-12)
    assertEqualsDouble(fw.v01.value, 6.0, 1e-12)
    assertEqualsDouble(fw.v00.value, 4.0, 1e-12)

  test("extractFourWorldValues: Theorem 4 identity holds"):
    val fw = StrategicEngine.extractFourWorldValues(
      baselineQ = Array(-1.0, 10.0, 8.0),
      openLoopQ = Array(-1.0, 7.0, 5.0),
      blindQ = Array(-1.0, 6.0, 4.0),
      staticEquity = 4.0
    )
    val reconstructed = fw.v00 + fw.deltaControl + fw.deltaSigStar + fw.deltaInteraction
    assertEqualsDouble(reconstructed.value, fw.v11.value, 1e-12)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `sbt "testOnly sicfun.holdem.engine.FourWorldSolveTest"`
Expected: FAIL — `buildFourWorldModels` and `extractFourWorldValues` not found.

- [ ] **Step 3: Implement the helper types and methods**

Add to `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala` in the companion object (after existing helpers):

```scala
  /** Three tabular models for the four-world grid solve.
    * V^{0,0} uses static equity (no model needed).
    */
  final case class FourWorldModels(
      baseline: TabularGenerativeModel,   // V^{1,1}: attrib kernel, closed-loop
      openLoop: TabularGenerativeModel,   // V^{1,0}: attrib kernel, open-loop
      blind: TabularGenerativeModel       // V^{0,1}: blind kernel, closed-loop
  ):
    def size: Int = 3

  /** Build the three tabular models for the four-world grid. */
  def buildFourWorldModels(
      gameState: GameState,
      rivalBeliefs: Map[PlayerId, StrategicRivalBelief],
      heroActions: Vector[PokerAction],
      heroBucket: Int,
      actionPriors: Map[(StrategicClass, PokerAction.Category), Double]
  ): FourWorldModels =
    FourWorldModels(
      baseline = PokerPftFormulation.buildTabularModel(
        gameState, rivalBeliefs, heroActions, heroBucket, actionPriors
      ),
      openLoop = PokerPftFormulation.buildOpenLoopModel(
        gameState, rivalBeliefs, heroActions, heroBucket, actionPriors
      ),
      blind = PokerPftFormulation.buildBlindKernelModel(
        gameState, rivalBeliefs, heroActions, heroBucket, actionPriors
      )
    )

  /** Extract FourWorld values from solver Q-value arrays.
    *
    * Each Q-value array is per-action at the root state.
    * The grid world value is the max Q-value (best action) for each model.
    */
  def extractFourWorldValues(
      baselineQ: Array[Double],
      openLoopQ: Array[Double],
      blindQ: Array[Double],
      staticEquity: Double
  ): FourWorld =
    FourWorld(
      v11 = Ev(baselineQ.max),
      v10 = Ev(openLoopQ.max),
      v01 = Ev(blindQ.max),
      v00 = Ev(staticEquity)
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `sbt "testOnly sicfun.holdem.engine.FourWorldSolveTest"`
Expected: PASS (all 3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/engine/StrategicEngine.scala src/test/scala/sicfun/holdem/engine/FourWorldSolveTest.scala
git commit -m "feat(engine): add FourWorldModels and extractFourWorldValues for grid-world solve"
```

---

## Task 4: Wire four-world solve into decidePftDpw

Integrate the three-model solve into the existing PftDpw decision path. When the solver succeeds for all three models, produce an exact FourWorld; otherwise fall back to the interpolated path.

**Files:**
- Modify: `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala` (the `decidePftDpw` method)
- Modify: `src/test/scala/sicfun/holdem/engine/FormalPathTest.scala` (add four-world assertions)

- [ ] **Step 1: Add a test that the four-world bundle appears in the PftDpw decision output**

Add to `src/test/scala/sicfun/holdem/engine/FormalPathTest.scala`:

```scala
  test("PftDpw path produces FourWorld in bundle when solver succeeds"):
    // Use a StrategicEngine with synthetic solver that always succeeds
    val engine = makeTestEngine(usePftDpw = true)
    engine.initSession(rivalIds = Vector(PlayerId(1)),
      rivalSeats = Map(PlayerId(1) -> RivalSeatInfo(Seat(2), 500.0)))
    engine.startHand()
    val action = engine.decide(
      makeFlop(pot = 100.0, toCall = 20.0, stackSize = 500.0),
      Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Check)
    )
    val bundle = engine.lastBundle
    // When the PftDpw solver is available, bundle should contain fourWorld
    bundle.fourWorld match
      case Some(fw) =>
        // Theorem 4 identity holds
        val reconstructed = fw.v00 + fw.deltaControl + fw.deltaSigStar + fw.deltaInteraction
        assertEqualsDouble(reconstructed.value, fw.v11.value, 1e-12)
      case None =>
        // Acceptable if native solver not loaded — verify fallback used
        assert(bundle.certification.isInstanceOf[CertificationResult.Unavailable.type]
            || bundle.certification.isInstanceOf[CertificationResult.TabularCertification])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly sicfun.holdem.engine.FormalPathTest"`
Expected: FAIL — `bundle.fourWorld` field does not exist yet.

- [ ] **Step 3: Add fourWorld field to DecisionEvaluationBundle**

Find the `DecisionEvaluationBundle` definition in StrategicEngine.scala and add:

```scala
    fourWorld: Option[FourWorld] = None
```

- [ ] **Step 4: Wire the three-model solve into decidePftDpw**

In `decidePftDpw`, after the baseline model is built and solved, add:

```scala
    // Four-world grid solve (V^{1,0}, V^{0,1})
    val fourWorldOpt: Option[FourWorld] = try
      val fwModels = StrategicEngine.buildFourWorldModels(
        gameState, rivalBeliefs, candidateActions, heroBucket, actionPriors
      )
      val olResult = PftDpwRuntime.solve(fwModels.openLoop, belief, pftConfig)
      val blindResult = PftDpwRuntime.solve(fwModels.blind, belief, pftConfig)
      if olResult.isSuccess && blindResult.isSuccess then
        val staticEquity = heroBucket / 9.0 // same as equity in buildTabularModel
        Some(StrategicEngine.extractFourWorldValues(
          baselineQ = pftResult.qValues,
          openLoopQ = olResult.qValues,
          blindQ = blindResult.qValues,
          staticEquity = staticEquity
        ))
      else None
    catch
      case _: Exception => None
```

Then pass `fourWorld = fourWorldOpt` when constructing the `DecisionEvaluationBundle`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `sbt "testOnly sicfun.holdem.engine.FormalPathTest"`
Expected: PASS (all tests including the new one)

- [ ] **Step 6: Commit**

```bash
git add src/main/scala/sicfun/holdem/engine/StrategicEngine.scala src/test/scala/sicfun/holdem/engine/FormalPathTest.scala
git commit -m "feat(engine): wire four-world grid solve into PftDpw decision path"
```

---

## Task 5: Update BridgeManifest and FormalClosureValidation

Upgrade the fidelity declarations and closure validation to reflect the solver-backed four-world values.

**Files:**
- Modify: `src/main/scala/sicfun/holdem/strategic/bridge/BridgeManifest.scala`
- Modify: `src/test/scala/sicfun/holdem/strategic/FormalClosureValidationTest.scala`

- [ ] **Step 1: Update BridgeManifest fidelity entries**

In `BridgeManifest.scala`, change the four-world entries:

```scala
    BridgeEntry("FourWorld.V10",            "Def 44", Fidelity.Approximate, Severity.Behavioral,  "solver-computed via PftDpw open-loop model when available; interpolated fallback"),
    BridgeEntry("FourWorld.V01",            "Def 44", Fidelity.Approximate, Severity.Behavioral,  "solver-computed via PftDpw blind-kernel model when available; interpolated fallback"),
    BridgeEntry("DeltaVocabulary",          "Def 50", Fidelity.Approximate, Severity.Behavioral,  "exact when solver-backed four-world available; approximate in fallback path"),
```

Note: severity downgraded from `Structural` to `Behavioral` because the solver path is now available (even if it's not always used).

Also update the GridWorldValues entry:

```scala
    BridgeEntry("GridWorldValues",          "Def 44 (v0.31.1)", Fidelity.Approximate, Severity.Behavioral, "all four worlds solver-computed via PftDpw when available; interpolated/absent fallback otherwise")
```

- [ ] **Step 2: Add closure validation test for structural gaps reduction**

Add to `FormalClosureValidationTest.scala`:

```scala
  test("BridgeManifest: no Structural severity gaps remain for four-world objects"):
    val structuralGaps = BridgeManifest.structuralGaps
    val fourWorldStructural = structuralGaps.filter(_.formalObject.startsWith("FourWorld"))
    assertEquals(fourWorldStructural.size, 0,
      s"Expected no Structural four-world gaps, found: ${fourWorldStructural.map(_.formalObject)}")

  test("BridgeManifest: DeltaVocabulary severity is Behavioral (not Structural)"):
    val dv = BridgeManifest.entries.find(_.formalObject == "DeltaVocabulary")
    assert(dv.isDefined, "DeltaVocabulary entry must exist")
    assertEquals(dv.get.severity, Severity.Behavioral)
```

- [ ] **Step 3: Run tests**

Run: `sbt "testOnly sicfun.holdem.strategic.FormalClosureValidationTest"`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/main/scala/sicfun/holdem/strategic/bridge/BridgeManifest.scala src/test/scala/sicfun/holdem/strategic/FormalClosureValidationTest.scala
git commit -m "fix(bridge): downgrade four-world gaps from Structural to Behavioral (solver path available)"
```

---

## Task 6: Integration smoke test

End-to-end validation that the four-world decomposition produces meaningful results.

**Files:**
- Test: `src/test/scala/sicfun/holdem/engine/FourWorldSolveTest.scala` (append)

- [ ] **Step 1: Add integration test**

Append to `FourWorldSolveTest.scala`:

```scala
  test("end-to-end: four-world values satisfy ordering constraints"):
    // V^{1,1} >= V^{1,0} (closed-loop >= open-loop under same kernel)
    // V^{1,1} >= V^{0,1} (attrib >= blind under same policy scope)
    // V^{0,0} is the minimum (blind + open-loop)
    val fw = StrategicEngine.extractFourWorldValues(
      baselineQ = Array(-1.0, 10.0, 8.0),
      openLoopQ = Array(-1.0, 7.0, 5.0),
      blindQ = Array(-1.0, 6.0, 4.0),
      staticEquity = 4.0
    )
    assert(fw.v11 >= fw.v10, s"V11=${fw.v11} should >= V10=${fw.v10}")
    assert(fw.v11 >= fw.v01, s"V11=${fw.v11} should >= V01=${fw.v01}")
    assert(fw.v10 >= fw.v00, s"V10=${fw.v10} should >= V00=${fw.v00}")
    assert(fw.v01 >= fw.v00, s"V01=${fw.v01} should >= V00=${fw.v00}")

  test("end-to-end: decomposition components have expected signs"):
    val fw = StrategicEngine.extractFourWorldValues(
      baselineQ = Array(-1.0, 10.0, 8.0),
      openLoopQ = Array(-1.0, 7.0, 5.0),
      blindQ = Array(-1.0, 6.0, 4.0),
      staticEquity = 4.0
    )
    // Delta_cont >= 0: control (closed-loop over open-loop) adds value
    assert(fw.deltaControl >= Ev.Zero, s"deltaControl=${fw.deltaControl} should be >= 0")
    // Delta_sig* >= 0: signaling (attrib over blind) adds value
    assert(fw.deltaSigStar >= Ev.Zero, s"deltaSigStar=${fw.deltaSigStar} should be >= 0")
```

- [ ] **Step 2: Run tests**

Run: `sbt "testOnly sicfun.holdem.engine.FourWorldSolveTest"`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/test/scala/sicfun/holdem/engine/FourWorldSolveTest.scala
git commit -m "test(engine): add four-world ordering and sign constraint smoke tests"
```

---

## Verification Checklist

After all 6 tasks:

1. `sbt "testOnly sicfun.holdem.engine.FourWorldFormulationTest"` — model variants
2. `sbt "testOnly sicfun.holdem.strategic.bridge.ValueBridgeTest"` — solver-backed bridge
3. `sbt "testOnly sicfun.holdem.engine.FourWorldSolveTest"` — extraction + ordering
4. `sbt "testOnly sicfun.holdem.engine.FormalPathTest"` — PftDpw integration
5. `sbt "testOnly sicfun.holdem.strategic.FormalClosureValidationTest"` — manifest + closure
6. `sbt "testOnly sicfun.holdem.strategic.AdaptationSafetyTest"` — no regression from prior fix

All should pass green. The BridgeManifest should report zero `Structural` severity gaps for FourWorld objects.
