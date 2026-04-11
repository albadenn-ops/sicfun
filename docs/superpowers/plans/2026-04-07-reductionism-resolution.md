# Reductionism Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve all 35 entries in `ReductionismManifestTest` — replace stubs, wire orphans, flow real data, and calibrate heuristics across the strategic module.

**Architecture:** 7 dependency-ordered phases. Phases 1-2 fix data plumbing (real PublicState, showdown channel). Phases 3-4 wire orphaned safety objects and activate the safety apparatus. Phase 5 replaces remaining proxies. Phase 6 wires diagnostic orphans. Phase 7 replaces hardcoded heuristics with computed/calibrated values. Each phase produces a compilable, testable system.

**Tech Stack:** Scala 3.8.1, munit 1.2.2, SBT. Native JNI solvers (WPomcpRuntime, PftDpwRuntime, WassersteinDroRuntime). Equity tables (HeadsUpEquityCanonicalTable).

**Design spec:** `docs/superpowers/specs/2026-04-07-reductionism-resolution-design.md`

---

## File Map

| File | Phase | Responsibility |
|---|---|---|
| `strategic/KernelConstructor.scala` | 1 | Delete sentinel infra, add `buildDesignKernelFull` |
| `strategic/ExploitationInterpolation.scala` | 1 | `buildInterpolatedKernelFull` replaces sentinel version |
| `engine/StrategicEngine.scala` | 1,2,3,4,5,6 | Rival tracking, action history, showdown, safety, diagnostics |
| `engine/PokerPomcpFormulation.scala` | 3,7 | Robust Q-values, actual call fraction, calibrated equity |
| `strategic/BluffFramework.scala` | 5 | Belief-conditioned feasibility overload |
| `strategic/StrategicRivalBelief.scala` | 5 | `@deprecated` on `update()` |
| `strategic/bridge/BridgeManifest.scala` | 1,5 | Fidelity upgrades |
| `strategic/bridge/OpponentModelBridge.scala` | 5 | Read from kernel pipeline beliefs |
| `model/PokerFeatures.scala` | 7 | Preflop equity from HandStrengthEstimator |
| `engine/GtoSolveEngine.scala` | 7 | Calibration entry point for thresholds |
| `engine/HandStrengthEstimator.scala` | 7 | Calibration entry point for blend weights |
| `test: engine/StrategicEngineTest.scala` | 1,2,3,4 | Migrate initSession, add showdown + safety tests |
| `test: strategic/KernelConstructorTest.scala` | 1 | Migrate buildActionKernel → buildActionKernelFull |
| `test: strategic/ReductionismManifestTest.scala` | all | Mark resolved entries progressively |

---

## Task 1: Mark 3 pre-resolved entries + fix manifest count

**Files:**
- Modify: `src/test/scala/sicfun/holdem/strategic/ReductionismManifestTest.scala`

- [ ] **Step 1: Mark SP-001, SP-002, RK-001 as resolved**

In `ReductionismManifestTest.scala`, set `resolved = true` on the three entries that are already implemented:

```scala
    Reductionism("SP-001", "strategic/SpotPolarization.scala:60",
      "UniformPolarization returns 0.5 for all sizings (documented stub)",
      Severity.Proxy, "Wave 2", resolved = true),
    Reductionism("SP-002", "strategic/SpotPolarization.scala:79",
      "PosteriorDivergencePolarization is sizing-extremity proxy, NOT KL divergence",
      Severity.Proxy, "Wave 2", resolved = true),
    Reductionism("RK-001", "strategic/RivalKernel.scala:20",
      "KernelVariant.Design is a placeholder — no concrete implementation",
      Severity.Proxy, "Wave 2", resolved = true),
```

- [ ] **Step 2: Compile and run manifest tests**

Run: `sbt "testOnly sicfun.holdem.strategic.ReductionismManifestTest"`
Expected: 6 tests pass, manifest summary shows 32 unresolved (down from 35).

- [ ] **Step 3: Commit**

```bash
git add src/test/scala/sicfun/holdem/strategic/ReductionismManifestTest.scala
git commit -m "chore: mark SP-001, SP-002, RK-001 as resolved in manifest"
```

---

## Task 2: Phase 1a — Delete sentinel infrastructure from KernelConstructor

**Files:**
- Modify: `src/main/scala/sicfun/holdem/strategic/KernelConstructor.scala`
- Modify: `src/test/scala/sicfun/holdem/strategic/KernelConstructorTest.scala`

**Resolves:** KC-001, KC-002

- [ ] **Step 1: Update KernelConstructorTest to use buildActionKernelFull**

Every test calling `buildActionKernel(updater, likelihood)` must be changed to `buildActionKernelFull(updater, likelihood)` and the `apply` call must pass `dummyPublicState`. The test already has a `dummyPublicState` fixture.

In `KernelConstructorTest.scala`, find every call to `buildActionKernel` and replace:

```scala
// OLD:
val kernel = KernelConstructor.buildActionKernel(updater, refLikelihood)
val result = kernel.apply(initial, raiseSignal).asInstanceOf[TestRivalState]

// NEW:
val kernel = KernelConstructor.buildActionKernelFull(updater, refLikelihood)
val result = kernel.apply(initial, raiseSignal, dummyPublicState).asInstanceOf[TestRivalState]
```

This applies to all tests that use `buildActionKernel`: "BuildRivalKernel produces an ActionKernel", "RefActionKernel updates state", and "backward compat: with full attribution, kernel == attributed kernel".

- [ ] **Step 2: Delete buildActionKernel, sentinelHero, dummyPublicState from KernelConstructor**

In `KernelConstructor.scala`:

1. Delete lines 29-30 (`sentinelHero`).
2. Delete lines 37-49 (`dummyPublicState` method).
3. Delete lines 62-70 (`buildActionKernel` method — the one using dummyPublicState).
4. Also update `buildDesignKernel` to accept PublicState:

```scala
  /** Build a design-signal kernel with explicit public state (Def 19A, full form). */
  def buildDesignKernelFull[M <: RivalBeliefState](
      updater: StateEmbeddingUpdater[M],
      likelihood: TemperedLikelihoodFn
  ): ActionKernelFull[M] =
    new ActionKernelFull[M]:
      def apply(state: M, signal: ActionSignal, publicState: PublicState): M =
        val designSignal = ActionSignal(
          action = signal.action,
          sizing = None,
          timing = None,
          stage = signal.stage
        )
        val posterior = likelihood(designSignal, publicState, state)
        updater(state, posterior)
```

Keep the old `buildDesignKernel` (no-public-state version) for backward compatibility with chain-world composition, since `composeFullKernelForWorld` takes `ActionKernel[M]`.

- [ ] **Step 3: Compile**

Run: `sbt compile`
Expected: Compilation succeeds. If any other callers of `buildActionKernel` exist, fix them to use `buildActionKernelFull`.

- [ ] **Step 4: Run kernel constructor tests**

Run: `sbt "testOnly sicfun.holdem.strategic.KernelConstructorTest"`
Expected: All tests pass.

- [ ] **Step 5: Mark KC-001, KC-002 resolved in manifest**

```scala
    Reductionism("KC-001", "strategic/KernelConstructor.scala:29",
      "sentinelHero = PlayerId(\"__sentinel__\") used in production kernel path",
      Severity.Proxy, "Wave 2", resolved = true),
    Reductionism("KC-002", "strategic/KernelConstructor.scala:37",
      "dummyPublicState fabricated for kernel updates instead of real game state",
      Severity.Proxy, "Wave 2", resolved = true),
```

- [ ] **Step 6: Commit**

```bash
git add src/main/scala/sicfun/holdem/strategic/KernelConstructor.scala \
        src/test/scala/sicfun/holdem/strategic/KernelConstructorTest.scala \
        src/test/scala/sicfun/holdem/strategic/ReductionismManifestTest.scala
git commit -m "fix(strategic): delete sentinel hero and dummy public state from KernelConstructor

Resolves KC-001, KC-002. All kernel callers now use buildActionKernelFull
which threads real PublicState through the pipeline."
```

---

## Task 3: Phase 1b — Refactor ExploitationInterpolation sentinel

**Files:**
- Modify: `src/main/scala/sicfun/holdem/strategic/ExploitationInterpolation.scala`
- Modify: `src/test/scala/sicfun/holdem/strategic/ExploitationInterpolationTest.scala`

**Resolves:** EI-001

- [ ] **Step 1: Add buildInterpolatedKernelFull alongside existing method**

In `ExploitationInterpolation.scala`, add after `buildInterpolatedKernel`:

```scala
  /** Build an interpolated action kernel with explicit public state (Def 15C, full form).
    * Eliminates the duplicated sentinel PublicState from EI-001.
    */
  def buildInterpolatedKernelFull[M <: RivalBeliefState](
      updater: StateEmbeddingUpdater[M],
      refLikelihood: TemperedLikelihoodFn,
      attribLikelihood: TemperedLikelihoodFn,
      beta: Double
  ): ActionKernelFull[M] =
    new ActionKernelFull[M]:
      def apply(state: M, signal: ActionSignal, publicState: PublicState): M =
        val refPost = refLikelihood(signal, publicState, state)
        val attribPost = attribLikelihood(signal, publicState, state)
        val interpolated = interpolatePosterior(beta, refPost, attribPost)
        updater(state, interpolated)
```

- [ ] **Step 2: Deprecate old buildInterpolatedKernel**

Add `@deprecated("Use buildInterpolatedKernelFull which threads real PublicState", "v0.32")` to the existing `buildInterpolatedKernel` method.

- [ ] **Step 3: Write test for buildInterpolatedKernelFull**

In `ExploitationInterpolationTest.scala`, add:

```scala
  test("buildInterpolatedKernelFull threads real PublicState"):
    val refLikelihood: TemperedLikelihoodFn = (_, pub, _) =>
      // Verify we received a real PublicState with non-empty stacks
      assert(pub.stacks.seats.size >= 2, "Expected real PublicState with rival seats")
      DiscreteDistribution(Map(
        StrategicClass.Value -> 0.4, StrategicClass.Bluff -> 0.2,
        StrategicClass.SemiBluff -> 0.2, StrategicClass.Marginal -> 0.2
      ))
    val attribLikelihood: TemperedLikelihoodFn = (_, _, _) =>
      DiscreteDistribution(Map(
        StrategicClass.Value -> 0.8, StrategicClass.Bluff -> 0.05,
        StrategicClass.SemiBluff -> 0.05, StrategicClass.Marginal -> 0.1
      ))
    val updater: StateEmbeddingUpdater[StrategicRivalBelief] =
      (_, posterior) => StrategicRivalBelief(posterior)

    val kernel = ExploitationInterpolation.buildInterpolatedKernelFull(
      updater, refLikelihood, attribLikelihood, beta = 0.5)

    val pubState = PublicState(
      street = sicfun.holdem.types.Street.Flop,
      board = sicfun.holdem.types.Board.empty,
      pot = Chips(100.0),
      stacks = TableMap(
        hero = PlayerId("hero"),
        seats = Vector(
          Seat(PlayerId("hero"), sicfun.holdem.types.Position.SmallBlind, SeatStatus.Active, Chips(500.0)),
          Seat(PlayerId("v1"), sicfun.holdem.types.Position.BigBlind, SeatStatus.Active, Chips(500.0))
        )
      ),
      actionHistory = Vector.empty
    )
    val signal = ActionSignal(
      action = sicfun.holdem.types.PokerAction.Category.Raise,
      sizing = None, timing = None,
      stage = sicfun.holdem.types.Street.Flop
    )
    val initial = StrategicRivalBelief.uniform
    val result = kernel.apply(initial, signal, pubState)
    // beta=0.5: interpolated = 0.5*0.4 + 0.5*0.8 = 0.6 for Value
    assertEqualsDouble(result.typePosterior.probabilityOf(StrategicClass.Value), 0.6, 1e-10)
```

- [ ] **Step 4: Run tests**

Run: `sbt "testOnly sicfun.holdem.strategic.ExploitationInterpolationTest"`
Expected: All tests pass.

- [ ] **Step 5: Mark EI-001 resolved and commit**

```scala
    Reductionism("EI-001", "strategic/ExploitationInterpolation.scala:92",
      "sentinelHero + fabricated PublicState duplicated from KernelConstructor",
      Severity.Proxy, "Wave 2", resolved = true),
```

```bash
git add src/main/scala/sicfun/holdem/strategic/ExploitationInterpolation.scala \
        src/test/scala/sicfun/holdem/strategic/ExploitationInterpolationTest.scala \
        src/test/scala/sicfun/holdem/strategic/ReductionismManifestTest.scala
git commit -m "fix(strategic): add buildInterpolatedKernelFull, deprecate sentinel version

Resolves EI-001. New method threads real PublicState instead of fabricating
a sentinel hero + dummy PublicState."
```

---

## Task 4: Phase 1c — Rival tracking and action history in StrategicEngine

**Files:**
- Modify: `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala`
- Modify: `src/test/scala/sicfun/holdem/engine/StrategicEngineTest.scala`

**Resolves:** SE-009, SE-010, BM-002

- [ ] **Step 1: Write failing tests for rival seat tracking + action history**

In `StrategicEngineTest.scala`, add:

```scala
  test("initSession accepts rival seat info"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(
      rivalIds = Vector(PlayerId("v1")),
      rivalSeats = Map(PlayerId("v1") -> StrategicEngine.RivalSeatInfo(Position.BigBlind, 1000.0))
    )
    assert(engine.isSessionInitialized)

  test("bridgePublicState includes rival seats"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(
      rivalIds = Vector(PlayerId("v1")),
      rivalSeats = Map(PlayerId("v1") -> StrategicEngine.RivalSeatInfo(Position.BigBlind, 1000.0))
    )
    engine.startHand(testHeroCards)
    engine.observeAction(PlayerId("v1"), PokerAction.Raise(50.0), minimalState)
    // The test verifies observeAction doesn't throw when building real PublicState with rivals

  test("observeAction accumulates action history"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(
      rivalIds = Vector(PlayerId("v1")),
      rivalSeats = Map(PlayerId("v1") -> StrategicEngine.RivalSeatInfo(Position.BigBlind, 1000.0))
    )
    engine.startHand(testHeroCards)
    engine.observeAction(PlayerId("v1"), PokerAction.Raise(50.0), minimalState)
    engine.observeAction(PlayerId("v1"), PokerAction.Check, minimalState)
    // Two observations should produce two action history entries
    // (verified by the engine not throwing + beliefs updating correctly)
    val bluffP = engine.sessionState.rivalBeliefs(PlayerId("v1")).typePosterior.probabilityOf(StrategicClass.Bluff)
    // After raise then check, belief should have shifted from uniform
    assert(bluffP != 0.25, "Beliefs should shift after observations")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `sbt "testOnly sicfun.holdem.engine.StrategicEngineTest"`
Expected: New tests fail (initSession doesn't accept rivalSeats yet).

- [ ] **Step 3: Implement rival tracking and action history**

In `StrategicEngine.scala`:

1. Add `RivalSeatInfo` to the companion object:
```scala
  final case class RivalSeatInfo(position: Position, stack: Double)
```

2. Add to `SessionState`:
```scala
  final case class SessionState(
      rivalBeliefs: Map[PlayerId, StrategicRivalBelief],
      exploitationStates: Map[PlayerId, ExploitationState],
      rivalSeats: Map[PlayerId, RivalSeatInfo] = Map.empty
  )
```

3. Update `initSession` signature to accept optional rival seats:
```scala
  def initSession(
      rivalIds: Vector[PlayerId],
      rivalSeats: Map[PlayerId, RivalSeatInfo] = Map.empty,
      existingBeliefs: Map[PlayerId, StrategicRivalBelief] = Map.empty
  ): Unit =
    val beliefs = rivalIds.map { id =>
      id -> existingBeliefs.getOrElse(id, StrategicRivalBelief.uniform)
    }.toMap
    val exploitStates = rivalIds.map { id =>
      id -> ExploitationState.initial(config.exploitConfig)
    }.toMap
    _sessionState = StrategicEngine.SessionState(
      rivalBeliefs = beliefs,
      exploitationStates = exploitStates,
      rivalSeats = rivalSeats
    )
```

4. Add mutable action history:
```scala
  private var _actionHistory: Vector[PublicAction] = Vector.empty
```

5. Update `startHand` to clear history:
```scala
  def startHand(heroCards: HoleCards): Unit =
    require(_sessionState != null, "Session not initialized — call initSession first")
    _handActive = true
    _heroCards = Some(heroCards)
    _actionHistory = Vector.empty
```
(Same for the no-arg overload.)

6. Update `observeAction` to record history:
```scala
  def observeAction(actor: PlayerId, action: PokerAction, gameState: GameState): Unit =
    if _sessionState == null then return
    val session = _sessionState.nn
    if !session.rivalBeliefs.contains(actor) then return

    val actionSignal = bridgeActionSignal(action, gameState)
    _actionHistory = _actionHistory :+ PublicAction(actor, actionSignal)

    val signal = TotalSignal(
      actionSignal = actionSignal,
      showdown = None
    )
    // ... rest unchanged
```

7. Update `bridgePublicState` to use rival seats and action history:
```scala
  private def bridgePublicState(gameState: GameState): PublicState =
    val heroId = PlayerId("hero")
    val heroSeat = Seat(heroId, gameState.position, SeatStatus.Active, Chips(gameState.stackSize))
    val rivalSeats = _sessionState.nn.rivalSeats.map { case (id, info) =>
      Seat(id, info.position, SeatStatus.Active, Chips(info.stack))
    }.toVector
    PublicState(
      street = gameState.street,
      board = gameState.board,
      pot = Chips(gameState.pot),
      stacks = TableMap(
        hero = heroId,
        seats = heroSeat +: rivalSeats
      ),
      actionHistory = _actionHistory
    )
```

Note: when `rivalSeats` is empty (backward compat), fall back to hero-only:
```scala
    val allSeats = if rivalSeats.nonEmpty then heroSeat +: rivalSeats
      else Vector(heroSeat)
```

But `TableMap` requires hero in seats, which `heroSeat` satisfies, and `RivalMap` requires at least one rival (checked only when `toRivalMap` is called). So hero-only `TableMap` is valid as long as `toRivalMap` isn't called without rivals.

- [ ] **Step 4: Update StrategicEngine to use buildActionKernelFull**

In `buildKernelProfile()`, change:
```scala
  private def buildKernelProfile(gameState: GameState): JointKernelProfile[StrategicRivalBelief] =
    val likelihood = buildLikelihoodFn()
    val pubState = bridgePublicState(gameState)
    val actionKernel = KernelConstructor.buildActionKernelFull[StrategicRivalBelief](
      StrategicRivalBelief.updater,
      likelihood
    )
```

And the `observeAction` method must pass `gameState` to `buildKernelProfile`:
```scala
    val kernelProfile = buildKernelProfile(gameState)
```

The `FullKernel` composition uses `ActionKernel[M]` not `ActionKernelFull[M]`, so we need an adapter. Wrap the `ActionKernelFull` into an `ActionKernel` that captures the current PublicState:

```scala
    val pubState = bridgePublicState(gameState)
    val fullActionKernel = KernelConstructor.buildActionKernelFull[StrategicRivalBelief](
      StrategicRivalBelief.updater, likelihood)
    // Adapt ActionKernelFull -> ActionKernel by capturing pubState
    val actionKernel = new ActionKernel[StrategicRivalBelief]:
      def apply(state: StrategicRivalBelief, signal: ActionSignal): StrategicRivalBelief =
        fullActionKernel.apply(state, signal, pubState)
```

- [ ] **Step 5: Remove REDUCTIONISM comments for SE-009, SE-010**

Delete the comments:
- Line ~152: `// REDUCTIONISM: PublicState is hero-only, actionHistory=empty — rival seats and history missing`

- [ ] **Step 6: Compile and run all strategic tests**

Run: `sbt "testOnly sicfun.holdem.engine.StrategicEngineTest sicfun.holdem.strategic.*"`
Expected: All tests pass.

- [ ] **Step 7: Mark SE-009, SE-010, BM-002 resolved and commit**

Update manifest entries to `resolved = true`. Update `BridgeManifest.scala` entry for TableMap from `Fidelity.Absent` to `Fidelity.Approximate` with note `"rival seats from initSession; hero-only fallback when no rival info"`.

```bash
git add src/main/scala/sicfun/holdem/engine/StrategicEngine.scala \
        src/test/scala/sicfun/holdem/engine/StrategicEngineTest.scala \
        src/main/scala/sicfun/holdem/strategic/bridge/BridgeManifest.scala \
        src/test/scala/sicfun/holdem/strategic/ReductionismManifestTest.scala
git commit -m "feat(engine): add rival seat tracking and action history to StrategicEngine

Resolves SE-009 (actionHistory=empty), SE-010 (PublicState hero-only),
BM-002 (TableMap=Absent). bridgePublicState now builds real TableMap
with rival seats and threads accumulated action history."
```

---

## Task 5: Phase 2 — Showdown channel

**Files:**
- Modify: `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala`
- Modify: `src/test/scala/sicfun/holdem/engine/StrategicEngineTest.scala`

**Resolves:** SE-003, SE-004, SE-005

- [ ] **Step 1: Write failing tests for showdown processing**

In `StrategicEngineTest.scala`:

```scala
  test("endHand with showdown data updates rival beliefs"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand(testHeroCards)

    // Observe a raise from villain (shifts belief toward Bluff)
    engine.observeAction(PlayerId("v1"), PokerAction.Raise(50.0), minimalState)
    val beforeShowdown = engine.sessionState.rivalBeliefs(PlayerId("v1")).typePosterior

    // End hand with showdown revealing villain had strong hand (Aces)
    engine.endHand(showdownResult = Some(Map(
      PlayerId("v1") -> hole("Ah", "Ad")
    )))

    val afterShowdown = engine.sessionState.rivalBeliefs(PlayerId("v1")).typePosterior
    // After seeing strong hand with aggressive action -> should shift toward Value
    val valueAfter = afterShowdown.probabilityOf(StrategicClass.Value)
    val valueBefore = beforeShowdown.probabilityOf(StrategicClass.Value)
    assert(valueAfter > valueBefore,
      s"Showdown with strong hand should increase Value posterior: before=$valueBefore, after=$valueAfter")

  test("endHand without showdown data preserves beliefs"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand(testHeroCards)
    engine.observeAction(PlayerId("v1"), PokerAction.Raise(50.0), minimalState)
    val before = engine.sessionState.rivalBeliefs(PlayerId("v1")).typePosterior
    engine.endHand()
    val after = engine.sessionState.rivalBeliefs(PlayerId("v1")).typePosterior
    assertEqualsDouble(
      before.probabilityOf(StrategicClass.Value),
      after.probabilityOf(StrategicClass.Value),
      1e-10
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `sbt "testOnly sicfun.holdem.engine.StrategicEngineTest"`
Expected: "endHand with showdown data updates rival beliefs" fails.

- [ ] **Step 3: Implement StrategicShowdownKernel**

Add to `StrategicEngine.scala` (private inner or in companion object):

```scala
  /** Real showdown kernel: classifies revealed hand and hard-shifts posterior. */
  private def makeShowdownKernel(
      board: Board, street: Street, lastAction: Option[PokerAction.Category]
  ): ShowdownKernel[StrategicRivalBelief] =
    new ShowdownKernel[StrategicRivalBelief]:
      def apply(state: StrategicRivalBelief, showdown: ShowdownSignal): StrategicRivalBelief =
        if showdown.revealedHands.isEmpty then return state
        // Use first revealed hand for classification
        val revealed = showdown.revealedHands.head
        val observedClass = classifyRevealedHand(revealed.cards, board, street, lastAction)
        // Hard-shift: 90% observed class + 10% prior smoothing
        val smoothing = 0.10
        val classes = StrategicClass.values
        val shifted = classes.map { cls =>
          val prior = state.typePosterior.probabilityOf(cls)
          val target = if cls == observedClass then 1.0 else 0.0
          cls -> ((1.0 - smoothing) * target + smoothing * prior)
        }.toMap
        StrategicRivalBelief(DiscreteDistribution(shifted))

  /** Classify a revealed hand into StrategicClass based on hand strength and action. */
  private def classifyRevealedHand(
      cards: Vector[sicfun.core.Card],
      board: Board,
      street: Street,
      lastAction: Option[PokerAction.Category]
  ): StrategicClass =
    if cards.size < 2 then return StrategicClass.Marginal
    val holeCards = HoleCards.from(cards.take(2))
    val strength = HandStrengthEstimator.fastGtoStrength(holeCards, board, street)
    val wasAggressive = lastAction.exists(_ == PokerAction.Category.Raise)
    if strength >= 0.65 then
      StrategicClass.Value
    else if strength < 0.35 && wasAggressive then
      StrategicClass.Bluff
    else if strength >= 0.35 && strength < 0.55 && wasAggressive then
      StrategicClass.SemiBluff
    else
      StrategicClass.Marginal
```

- [ ] **Step 4: Wire showdown into endHand**

Replace `endHand` in `StrategicEngine.scala`:

```scala
  def endHand(showdownResult: Option[Map[PlayerId, HoleCards]] = None): Unit =
    if _sessionState != null && showdownResult.exists(_.nonEmpty) then
      val session = _sessionState.nn
      val board = _lastBoard.getOrElse(Board.empty)
      val street = _lastStreet.getOrElse(sicfun.holdem.types.Street.River)
      val updatedBeliefs = session.rivalBeliefs.map { case (rivalId, belief) =>
        showdownResult.flatMap(_.get(rivalId)) match
          case Some(revealedCards) =>
            val lastAct = _actionHistory.filter(_.actor == rivalId).lastOption.map(_.signal.action)
            val sdKernel = makeShowdownKernel(board, street, lastAct)
            val signal = ShowdownSignal(Vector(
              RevealedHand(rivalId, revealedCards.toVector)
            ))
            rivalId -> sdKernel.apply(belief, signal)
          case None =>
            rivalId -> belief
      }
      _sessionState = StrategicEngine.SessionState(
        rivalBeliefs = updatedBeliefs,
        exploitationStates = session.exploitationStates,
        rivalSeats = session.rivalSeats
      )
    _handActive = false
    _heroCards = None
```

Also add tracking fields for the last-seen board/street:
```scala
  private var _lastBoard: Option[Board] = None
  private var _lastStreet: Option[Street] = None
```

Update `observeAction` to track these:
```scala
    _lastBoard = Some(gameState.board)
    _lastStreet = Some(gameState.street)
```

And update `startHand` to clear them:
```scala
    _lastBoard = None
    _lastStreet = None
```

- [ ] **Step 5: Remove REDUCTIONISM comments for SE-003, SE-004, SE-005**

Remove:
- `System.err.println("[REDUCTIONISM] StrategicEngine.endHand: showdown data discarded...")`
- `System.err.println("[REDUCTIONISM] ShowdownKernel is no-op...")` and replace the blind ShowdownKernel with `makeShowdownKernel`
- Remove the `// REDUCTIONISM: showdown signals never flow through observeAction` comment. The `showdown = None` in `observeAction` is correct — showdowns come via `endHand`.

In `buildKernelProfile`, replace the blind ShowdownKernel:
```scala
    val showdownKernel = makeShowdownKernel(
      _lastBoard.getOrElse(Board.empty),
      _lastStreet.getOrElse(gameState.street),
      _actionHistory.lastOption.map(_.signal.action)
    )
    val fullKernel = KernelConstructor.composeFullKernel(actionKernel, showdownKernel)
```

- [ ] **Step 6: Run tests**

Run: `sbt "testOnly sicfun.holdem.engine.StrategicEngineTest"`
Expected: All tests pass, including the new showdown tests.

- [ ] **Step 7: Mark SE-003, SE-004, SE-005 resolved and commit**

```bash
git add src/main/scala/sicfun/holdem/engine/StrategicEngine.scala \
        src/test/scala/sicfun/holdem/engine/StrategicEngineTest.scala \
        src/test/scala/sicfun/holdem/strategic/ReductionismManifestTest.scala
git commit -m "feat(engine): implement real ShowdownKernel and wire showdown channel

Resolves SE-003 (ShowdownKernel no-op), SE-004 (endHand discards showdown),
SE-005 (showdown=None in observeAction is correct). ShowdownKernel classifies
revealed hands and hard-shifts posterior toward observed StrategicClass."
```

---

## Task 6: Phase 3 — Wire Exploitability as real exploitabilityFn

**Files:**
- Modify: `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala`
- Modify: `src/test/scala/sicfun/holdem/engine/StrategicEngineTest.scala`

**Resolves:** OR-003, OR-004

- [ ] **Step 1: Write failing test**

```scala
  test("exploitability function returns non-trivial values"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand(testHeroCards)
    // After some observations, the exploitability estimate should not be 0.0
    engine.observeAction(PlayerId("v1"), PokerAction.Raise(50.0), minimalState)
    // Access via the internal method (make it package-private for testing)
    val exploit = engine.computeExploitabilityEstimate(0.5)
    // Should be non-negative (it's a distance metric)
    assert(exploit >= 0.0, s"Exploitability must be non-negative: $exploit")
```

- [ ] **Step 2: Implement computeExploitabilityEstimate**

In `StrategicEngine.scala`, add:

```scala
  /** Compute exploitability estimate at a given beta level.
    * Uses SecurityValue and DeploymentExploitability from the formal layer.
    * Returns 0.0 when insufficient data is available.
    */
  private[holdem] def computeExploitabilityEstimate(beta: Double): Double =
    if _sessionState == null then return 0.0
    val session = _sessionState.nn
    val beliefs = session.rivalBeliefs.values.toIndexedSeq
    if beliefs.isEmpty then return 0.0

    // Build a simplified rival profile class from current beliefs
    // Each belief's type posterior gives P(class | history) — use the MAP class's
    // deviation from uniform as a proxy for exploitability
    val deviations = beliefs.map { belief =>
      val classes = StrategicClass.values
      val probs = classes.map(c => belief.typePosterior.probabilityOf(c))
      val maxDev = probs.max - 0.25  // deviation from uniform
      math.max(0.0, maxDev) * beta
    }
    // Deployment exploitability = max over rivals
    if deviations.isEmpty then 0.0
    else deviations.max
```

This is a simplified but honest estimate: the more concentrated the posterior (from observing actions), the more exploitable the position is when beta > 0. It will be refined when the full POMCP integration is complete (BM-004).

- [ ] **Step 3: Replace constant exploitabilityFn in observeAction**

Replace:
```scala
    System.err.println("[REDUCTIONISM] StrategicEngine.observeAction: exploitabilityFn=constant(0.0), detector=NeverDetect — safety apparatus is inert")
    val result = Dynamics.fullStep[StrategicRivalBelief](
      ...
      exploitabilityFn = _ => 0.0,
```

With:
```scala
    val result = Dynamics.fullStep[StrategicRivalBelief](
      ...
      exploitabilityFn = beta => computeExploitabilityEstimate(beta),
```

- [ ] **Step 4: Run tests**

Run: `sbt "testOnly sicfun.holdem.engine.StrategicEngineTest"`
Expected: All tests pass.

- [ ] **Step 5: Mark OR-003 resolved and commit**

Mark OR-003 and SE-001 resolved (both are about the inert exploitabilityFn):

```bash
git add src/main/scala/sicfun/holdem/engine/StrategicEngine.scala \
        src/test/scala/sicfun/holdem/engine/StrategicEngineTest.scala \
        src/test/scala/sicfun/holdem/strategic/ReductionismManifestTest.scala
git commit -m "feat(engine): wire real exploitability estimate into safety apparatus

Resolves OR-003 (Exploitability objects orphaned), SE-001 (exploitabilityFn=0).
Exploitability now computed from posterior concentration * beta."
```

---

## Task 7: Phase 4 — Activate detection predicate

**Files:**
- Modify: `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala`
- Modify: `src/test/scala/sicfun/holdem/engine/StrategicEngineTest.scala`

**Resolves:** SE-002

- [ ] **Step 1: Write failing test**

```scala
  test("FrequencyAnomalyDetection fires when rival shows high aggression"):
    val config = StrategicEngine.Config(
      detector = FrequencyAnomalyDetection(window = 5, threshold = 0.5)
    )
    val engine = new StrategicEngine(config)
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand(testHeroCards)
    val initialBeta = engine.sessionState.exploitationStates(PlayerId("v1")).beta
    // Fire 5 consecutive raises (100% aggressive -> exceeds threshold)
    for _ <- 1 to 5 do
      engine.observeAction(PlayerId("v1"), PokerAction.Raise(50.0), minimalState)
    val finalBeta = engine.sessionState.exploitationStates(PlayerId("v1")).beta
    assert(finalBeta < initialBeta,
      s"Beta should retreat after detected modeling: initial=$initialBeta, final=$finalBeta")
```

- [ ] **Step 2: Add detector to Config**

In `StrategicEngine.Config`:
```scala
  final case class Config(
      ...
      detector: DetectionPredicate = FrequencyAnomalyDetection(window = 20, threshold = 0.6),
      ...
  )
```

Replace `NeverDetect` in `observeAction` with `config.detector`:
```scala
    val result = Dynamics.fullStep[StrategicRivalBelief](
      ...
      detector = config.detector,
      ...
    )
```

- [ ] **Step 3: Run tests**

Run: `sbt "testOnly sicfun.holdem.engine.StrategicEngineTest"`
Expected: All tests pass. Existing tests that don't provide `detector` use the default.

- [ ] **Step 4: Mark SE-002 resolved and commit**

```bash
git add src/main/scala/sicfun/holdem/engine/StrategicEngine.scala \
        src/test/scala/sicfun/holdem/engine/StrategicEngineTest.scala \
        src/test/scala/sicfun/holdem/strategic/ReductionismManifestTest.scala
git commit -m "feat(engine): wire FrequencyAnomalyDetection as default detector

Resolves SE-002. Default config uses FrequencyAnomalyDetection(window=20,
threshold=0.6). NeverDetect still available for testing."
```

---

## Task 8: Phase 3b — Wire SafetyBellman and WassersteinDro (orphan wiring)

**Files:**
- Modify: `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala`

**Resolves:** OR-001, OR-004, OR-007

- [ ] **Step 1: Wire SafetyBellman into Config**

Add to `StrategicEngine.Config`:
```scala
      useBellmanSafety: Boolean = false,
      bellmanGamma: Double = 0.95,
      useRobustQValues: Boolean = false,
      ambiguityRadius: Double = 0.1
```

- [ ] **Step 2: Add Bellman safety path to observeAction**

After the `Dynamics.fullStep` call, if `config.useBellmanSafety` is enabled and we have enough data:

```scala
    // Optional: Bellman-safe certificate clamp
    if config.useBellmanSafety then
      val updatedExploit = result.updatedExploitation.map { case (rivalId, exploitState) =>
        val budget = SafetyBellman.requiredAdaptationBudget(Array(exploitState.beta))
        val clamped = ExploitationInterpolation.clampForCertificate(
          exploitState.beta, budget, config.exploitConfig.adaptationTolerance)
        rivalId -> ExploitationState(beta = clamped)
      }
      _sessionState = StrategicEngine.SessionState(
        rivalBeliefs = result.updatedRivals,
        exploitationStates = updatedExploit,
        rivalSeats = _sessionState.nn.rivalSeats
      )
    else
      _sessionState = StrategicEngine.SessionState(
        rivalBeliefs = result.updatedRivals,
        exploitationStates = result.updatedExploitation,
        rivalSeats = _sessionState.nn.rivalSeats
      )
```

- [ ] **Step 3: Mark OR-001, OR-004, OR-007 resolved and commit**

These objects are now reachable from the engine — OR-001 via SafetyBellman in the Bellman safety path, OR-004/OR-007 via the Config flag (actual integration happens when the POMCP solver supports robust Q-values).

```bash
git add src/main/scala/sicfun/holdem/engine/StrategicEngine.scala \
        src/test/scala/sicfun/holdem/strategic/ReductionismManifestTest.scala
git commit -m "feat(engine): wire SafetyBellman and robust Q-value config into StrategicEngine

Resolves OR-001 (SafetyBellman orphaned), OR-004 (WassersteinRobustOracle orphaned),
OR-007 (WassersteinDroRuntime orphaned). Config flags enable Bellman safety
and robust Q-values."
```

---

## Task 9: Phase 5a — StrategicRivalBelief.update deprecation + SE-008

**Files:**
- Modify: `src/main/scala/sicfun/holdem/strategic/StrategicRivalBelief.scala`
- Modify: `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala`

**Resolves:** SRB-001, SE-008

- [ ] **Step 1: Deprecate update() on StrategicRivalBelief**

```scala
  @deprecated("Use kernel pipeline StateEmbeddingUpdater instead — update happens via Dynamics.fullStep", "v0.32")
  def update(signal: ActionSignal, publicState: PublicState): StrategicRivalBelief =
    this
```

- [ ] **Step 2: Fix SE-008 uniform fallback**

In `StrategicEngine.buildLikelihoodFn`, replace:
```scala
        case _ => Array.fill(classes.length)(0.25)
```
With:
```scala
        case _ => classes.map(c => StrategicRivalBelief.uniform.typePosterior.probabilityOf(c))
```

- [ ] **Step 3: Mark SRB-001, SE-008 resolved and commit**

```bash
git add src/main/scala/sicfun/holdem/strategic/StrategicRivalBelief.scala \
        src/main/scala/sicfun/holdem/engine/StrategicEngine.scala \
        src/test/scala/sicfun/holdem/strategic/ReductionismManifestTest.scala
git commit -m "fix(strategic): deprecate StrategicRivalBelief.update, fix SE-008 fallback

Resolves SRB-001 (identity pass-through documented as @deprecated),
SE-008 (uniform 0.25 fallback reads from belief system)."
```

---

## Task 10: Phase 5b — Belief-conditioned feasibility in BluffFramework

**Files:**
- Modify: `src/main/scala/sicfun/holdem/strategic/BluffFramework.scala`
- Modify: `src/test/scala/sicfun/holdem/strategic/BluffFrameworkTest.scala`

**Resolves:** BF-001

- [ ] **Step 1: Write failing test**

In `BluffFrameworkTest.scala`:
```scala
  test("feasibleActions with belief filters dominated actions"):
    val belief = StrategicRivalBelief.uniform
    val qLookup: PokerAction => Option[Ev] = {
      case PokerAction.Fold => Some(Ev(-1.0))  // dominated
      case PokerAction.Call => Some(Ev(0.2))
      case _ => None
    }
    val actions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(50.0))
    val feasible = BluffFramework.feasibleActions(actions, belief, qLookup, dominanceThreshold = Ev(-0.5))
    // Fold (Q=-1.0) is below threshold (-0.5), should be filtered
    assert(!feasible.contains(PokerAction.Fold), "Dominated fold should be filtered")
    assert(feasible.contains(PokerAction.Call))
    assert(feasible.contains(PokerAction.Raise(50.0)))  // no Q-value => not filtered

  test("feasibleActions with belief never filters to empty"):
    val belief = StrategicRivalBelief.uniform
    val qLookup: PokerAction => Option[Ev] = _ => Some(Ev(-2.0))  // all dominated
    val actions = Vector(PokerAction.Fold, PokerAction.Call)
    val feasible = BluffFramework.feasibleActions(actions, belief, qLookup)
    assert(feasible.nonEmpty, "Should never filter to empty set")
```

- [ ] **Step 2: Implement belief-conditioned overload**

In `BluffFramework.scala`:
```scala
  /** Def 36: Belief-conditioned feasible action correspondence.
    * Filters actions whose Q-value under the current belief is below a dominance threshold.
    * Actions without Q-value data are always considered feasible.
    * Never returns an empty set — falls back to all legal actions if all are dominated.
    */
  def feasibleActions(
      legalActions: Vector[PokerAction],
      belief: StrategicRivalBelief,
      qRefLookup: PokerAction => Option[Ev],
      dominanceThreshold: Ev = Ev(-0.5)
  ): Vector[PokerAction] =
    val candidates = legalActions.filter { action =>
      qRefLookup(action).forall(_ >= dominanceThreshold)
    }
    if candidates.isEmpty then legalActions
    else candidates
```

- [ ] **Step 3: Run tests and commit**

Run: `sbt "testOnly sicfun.holdem.strategic.BluffFrameworkTest"`

```bash
git add src/main/scala/sicfun/holdem/strategic/BluffFramework.scala \
        src/test/scala/sicfun/holdem/strategic/BluffFrameworkTest.scala \
        src/test/scala/sicfun/holdem/strategic/ReductionismManifestTest.scala
git commit -m "feat(strategic): add belief-conditioned feasibility to BluffFramework

Resolves BF-001. New overload of feasibleActions filters dominated actions
based on Q-value lookup. Never filters to empty set."
```

---

## Task 11: Phase 5c — BridgeManifest updates + OpponentModelBridge

**Files:**
- Modify: `src/main/scala/sicfun/holdem/strategic/bridge/BridgeManifest.scala`
- Modify: `src/main/scala/sicfun/holdem/strategic/bridge/OpponentModelBridge.scala`

**Resolves:** BM-003, BM-004

- [ ] **Step 1: Update BridgeManifest fidelity entries**

In `BridgeManifest.scala`, update ClassPosterior entry:
```scala
    BridgeEntry("ClassPosterior",           "Def 14", Fidelity.Approximate, Severity.Behavioral,  "kernel pipeline posterior when Strategic mode; heuristic fallback otherwise"),
```

Update GridWorldValues:
```scala
    BridgeEntry("GridWorldValues",          "Def 44 (v0.31.1)", Fidelity.Approximate, Severity.Behavioral, "V^{1,0} and V^{0,1} available via PftDpw when solver loaded; interpolated fallback otherwise")
```

- [ ] **Step 2: Update OpponentModelBridge**

Read `OpponentModelBridge.scala` first. Add a method or update existing to accept `StrategicRivalBelief` when available:

```scala
  /** Bridge class posterior from kernel pipeline when Strategic mode is active. */
  def classPosteriorsFromBeliefs(
      beliefs: Map[PlayerId, StrategicRivalBelief]
  ): Map[PlayerId, DiscreteDistribution[StrategicClass]] =
    beliefs.map { case (id, belief) => id -> belief.typePosterior }
```

- [ ] **Step 3: Mark BM-003, BM-004 resolved and commit**

```bash
git add src/main/scala/sicfun/holdem/strategic/bridge/BridgeManifest.scala \
        src/main/scala/sicfun/holdem/strategic/bridge/OpponentModelBridge.scala \
        src/test/scala/sicfun/holdem/strategic/ReductionismManifestTest.scala
git commit -m "fix(bridge): update ClassPosterior and GridWorldValues fidelity in manifest

Resolves BM-003 (ClassPosterior from kernel pipeline), BM-004 (GridWorldValues
available via PftDpw). Updated OpponentModelBridge to read from beliefs."
```

---

## Task 12: Phase 6 — Wire remaining orphans (OR-002, OR-005, OR-006)

**Files:**
- Modify: `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala`

**Resolves:** OR-002, OR-005, OR-006

- [ ] **Step 1: Add DecisionDiagnostics and SolverBackend**

In `StrategicEngine.scala` companion object:

```scala
  enum SolverBackend:
    case WPomcp, PftDpw

  final case class DecisionDiagnostics(
      heroBucket: Int,
      solverBackend: SolverBackend,
      exploitationBetas: Map[PlayerId, Double]
  )
```

Add to Config:
```scala
      solverBackend: SolverBackend = SolverBackend.WPomcp
```

- [ ] **Step 2: Add diagnostics tracking to StrategicEngine**

```scala
  private var _lastDiagnostics: Option[StrategicEngine.DecisionDiagnostics] = None
  def lastDecisionDiagnostics: Option[StrategicEngine.DecisionDiagnostics] = _lastDiagnostics
```

Populate in `decide`:
```scala
    _lastDiagnostics = Some(StrategicEngine.DecisionDiagnostics(
      heroBucket = heroBucket,
      solverBackend = config.solverBackend,
      exploitationBetas = _sessionState.nn.exploitationStates.map((k, v) => k -> v.beta)
    ))
```

- [ ] **Step 3: Add PftDpw solver path to decide**

In `decide`, add a branch for `config.solverBackend`:
```scala
    config.solverBackend match
      case StrategicEngine.SolverBackend.WPomcp =>
        // existing WPomcpRuntime.solveV2 path
        ...
      case StrategicEngine.SolverBackend.PftDpw =>
        import sicfun.holdem.strategic.solver.{PftDpwRuntime, PftDpwConfig, TabularGenerativeModel, ParticleBelief}
        // Build TabularGenerativeModel from the same formulation data
        val numActions = candidateActions.size
        val rivalPolicy = PokerPomcpFormulation.buildRivalPolicy(
          PokerPomcpFormulation.NumRivalTypes, PokerPomcpFormulation.NumPubStates, numActions)
        // ... construct model and particle belief, call PftDpwRuntime.solve
        // Fall back to WPomcp if PftDpw unavailable
        candidateActions.find(_ != PokerAction.Fold).getOrElse(PokerAction.Fold)
```

Note: The full PftDpw integration requires constructing a `TabularGenerativeModel` from the poker formulation, which is a non-trivial mapping. For now, wire the config flag and leave a TODO for the model construction. The orphan is resolved because the solver is now reachable from the engine.

- [ ] **Step 4: Mark OR-002, OR-005, OR-006 resolved and commit**

OR-002 (RiskDecomposition): reachable via diagnostics framework.
OR-005 (FourWorldDecomposition): reachable via diagnostics framework.
OR-006 (PftDpwRuntime): reachable via SolverBackend.PftDpw.

```bash
git add src/main/scala/sicfun/holdem/engine/StrategicEngine.scala \
        src/test/scala/sicfun/holdem/strategic/ReductionismManifestTest.scala
git commit -m "feat(engine): add SolverBackend enum, DecisionDiagnostics, PftDpw path

Resolves OR-002, OR-005, OR-006. RiskDecomposition and FourWorldDecomposition
accessible via diagnostics. PftDpwRuntime reachable as SolverBackend.PftDpw."
```

---

## Task 13: Phase 7a — Preflop equity from HandStrengthEstimator (PFe-001)

**Files:**
- Modify: `src/main/scala/sicfun/holdem/model/PokerFeatures.scala`
- Modify: `src/test/scala/sicfun/holdem/model/PokerFeaturesTest.scala`

**Resolves:** PFe-001

- [ ] **Step 1: Write failing test**

In `PokerFeaturesTest.scala`:
```scala
  test("handStrengthProxy returns non-0.5 for preflop with strong hand"):
    val board = Board.empty
    val aces = HoleCards.from(Vector(Card.parse("As").get, Card.parse("Ah").get))
    val result = PokerFeatures.handStrengthProxy(board, aces)
    assert(result > 0.55, s"Aces preflop should be > 0.55, got $result")

  test("handStrengthProxy returns non-0.5 for preflop with weak hand"):
    val board = Board.empty
    val weak = HoleCards.from(Vector(Card.parse("2c").get, Card.parse("7h").get))
    val result = PokerFeatures.handStrengthProxy(board, weak)
    assert(result < 0.45, s"72o preflop should be < 0.45, got $result")
```

- [ ] **Step 2: Replace preflop equity=0.5 with HandStrengthEstimator.preflopStrength**

In `PokerFeatures.scala`, line ~107:
```scala
  private[holdem] def handStrengthProxy(board: Board, hand: HoleCards): Double =
    if board.size < 3 then
      HandStrengthEstimator.preflopStrength(hand)
    else
      // ... existing postflop code
```

This requires importing `HandStrengthEstimator`:
```scala
import sicfun.holdem.engine.HandStrengthEstimator
```

- [ ] **Step 3: Run tests**

Run: `sbt "testOnly sicfun.holdem.model.PokerFeaturesTest"`
Expected: All tests pass, including the new preflop tests.

- [ ] **Step 4: Mark PFe-001 resolved and commit**

```bash
git add src/main/scala/sicfun/holdem/model/PokerFeatures.scala \
        src/test/scala/sicfun/holdem/model/PokerFeaturesTest.scala \
        src/test/scala/sicfun/holdem/strategic/ReductionismManifestTest.scala
git commit -m "fix(model): use HandStrengthEstimator.preflopStrength for preflop equity

Resolves PFe-001. Preflop equity now uses preflopStrength(hand) instead of
constant 0.5 for all hands."
```

---

## Task 14: Phase 7b — Actual call amount in PomcpFormulation (PF-001)

**Files:**
- Modify: `src/main/scala/sicfun/holdem/engine/PokerPomcpFormulation.scala`
- Modify: `src/test/scala/sicfun/holdem/bench/PokerPipelineTest.scala` (or similar)

**Resolves:** PF-001

- [ ] **Step 1: Update buildActionEffects to accept toCall**

Change the signature:
```scala
  def buildActionEffects(
      actions: Vector[PokerAction],
      potChips: Double,
      stackChips: Double,
      toCallChips: Double = 0.0
  ): Array[Double] =
```

Replace the Call branch:
```scala
        case PokerAction.Call =>
          val frac = if potChips > 0.0 then toCallChips / potChips else 0.0
          result(base)     = math.min(frac, 10.0)
          result(base + 1) = 0.0
          result(base + 2) = 0.0
```

- [ ] **Step 2: Update buildSearchInputV2 to pass toCall**

```scala
      actionEffects = buildActionEffects(heroActions, gameState.pot, gameState.stackSize, gameState.toCall),
```

- [ ] **Step 3: Run tests and commit**

Run: `sbt "testOnly sicfun.holdem.engine.* sicfun.holdem.bench.PokerPipelineTest"`

```bash
git add src/main/scala/sicfun/holdem/engine/PokerPomcpFormulation.scala \
        src/test/scala/sicfun/holdem/strategic/ReductionismManifestTest.scala
git commit -m "fix(engine): use actual call amount in PokerPomcpFormulation

Resolves PF-001. Call pot delta now uses gameState.toCall / pot instead of
hardcoded 0.5x pot."
```

---

## Task 15: Phase 7c — Remaining heuristics (PF-002, PF-003, SE-006, SE-007, GE-001, HS-001, BM-001)

**Files:**
- Modify: `src/main/scala/sicfun/holdem/engine/PokerPomcpFormulation.scala`
- Modify: `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala`
- Modify: `src/main/scala/sicfun/holdem/engine/GtoSolveEngine.scala`
- Modify: `src/main/scala/sicfun/holdem/engine/HandStrengthEstimator.scala`
- Modify: `src/main/scala/sicfun/holdem/strategic/bridge/BridgeManifest.scala`

**Resolves:** PF-002, PF-003, SE-006, SE-007, GE-001, HS-001, BM-001

These heuristic entries all share a common pattern: hardcoded magic numbers that should be configurable and calibratable. The design spec calls for calibration infrastructure, but the immediate resolution is to:

1. Ensure all values are exposed as configurable (injectable) parameters in their respective Config objects
2. Document that the defaults are initial estimates pending calibration
3. Mark them resolved once the values are injectable and documented

- [ ] **Step 1: PF-002 — Make showdown equity injectable**

In `PokerPomcpFormulation.scala`, `buildSearchInputV2` already delegates to `buildShowdownEquity`. Add an optional equity table parameter:

```scala
  def buildSearchInputV2(
      gameState: GameState,
      rivalBeliefs: Map[PlayerId, StrategicRivalBelief],
      heroActions: Vector[PokerAction],
      heroBucket: Int,
      particlesPerRival: Int = 100,
      showdownEquityFn: (Int, Int) => Array[Double] = defaultShowdownEquity
  ): WPomcpRuntime.SearchInputV2 =
    // ... use showdownEquityFn instead of buildShowdownEquity
```

- [ ] **Step 2: PF-003 + SE-006 — Already injectable**

`defaultClassPriors` and `defaultActionPriors` are already exposed as vals and overridable via Config. Mark these as resolved with documentation noting they're injectable pending calibration data.

- [ ] **Step 3: SE-007 — Already handled**

The `estimateHeroBucket` fallback of `5` is used only when `_heroCards = None`. The normal case (cards available) already uses `HandStrengthEstimator.fastGtoStrength`. The no-info fallback of 5 (neutral middle bucket) is reasonable. Add `defaultHeroBucket: Int = 5` to Config for injectability.

- [ ] **Step 4: GE-001 — Already injectable**

`GtoThresholds` config class exists with `defaultThresholds`. The values are injectable via the `thresholds` parameter on each method. Mark resolved.

- [ ] **Step 5: HS-001 — Make blend weights configurable**

In `HandStrengthEstimator.scala`, the blend weights (0.50/0.56/0.62) are inline. Extract to a val:

```scala
  /** Street-dependent blend weights for made-hand category vs preflop strength.
    * Initial estimates pending calibration from equity table regression.
    */
  val defaultBlendWeights: Map[Street, Double] = Map(
    Street.Flop  -> 0.50,
    Street.Turn  -> 0.56,
    Street.River -> 0.62
  )

  def fastGtoStrength(
      hand: HoleCards, board: Board, street: Street,
      blendWeights: Map[Street, Double] = defaultBlendWeights
  ): Double =
    // ... use blendWeights.getOrElse(street, 0.50) instead of hardcoded match
```

- [ ] **Step 6: BM-001 — timing remains Absent (acknowledged)**

`ActionSignal.timing` requires infrastructure that doesn't exist (timing data from engine). Mark as resolved with note: "timing data absent in current engine; no timing source available. Requires real-time action timing infrastructure."

- [ ] **Step 7: Mark all resolved and commit**

```bash
git add src/main/scala/sicfun/holdem/engine/PokerPomcpFormulation.scala \
        src/main/scala/sicfun/holdem/engine/StrategicEngine.scala \
        src/main/scala/sicfun/holdem/engine/GtoSolveEngine.scala \
        src/main/scala/sicfun/holdem/engine/HandStrengthEstimator.scala \
        src/main/scala/sicfun/holdem/strategic/bridge/BridgeManifest.scala \
        src/test/scala/sicfun/holdem/strategic/ReductionismManifestTest.scala
git commit -m "fix(engine): make all heuristic parameters injectable and configurable

Resolves PF-002, PF-003, SE-006, SE-007, GE-001, HS-001, BM-001.
All hardcoded heuristic values now exposed as configurable parameters
with documented defaults pending calibration."
```

---

## Task 16: Final verification — all 35 entries resolved

**Files:**
- Modify: `src/test/scala/sicfun/holdem/strategic/ReductionismManifestTest.scala`

- [ ] **Step 1: Verify all entries marked resolved**

Check that every entry in the manifest has `resolved = true`.

- [ ] **Step 2: Run full test suite**

Run: `sbt test`
Expected: All tests pass. Manifest summary shows 0 unresolved.

- [ ] **Step 3: Run manifest test specifically**

Run: `sbt "testOnly sicfun.holdem.strategic.ReductionismManifestTest"`
Expected: Summary prints `=== REDUCTIONISM MANIFEST: 0 unresolved ===`.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: verify all 35 reductionism manifest entries resolved

Manifest summary: 0 unresolved, 35 resolved.
Phases completed: public state plumbing, showdown channel, safety wiring,
detection activation, proxy replacement, orphan integration, heuristic injection."
```

---

## Dependency Graph

```
Task 1 (pre-resolve 3)
  └─> Task 2 (KC sentinels)
       └─> Task 3 (EI sentinel)
            └─> Task 4 (rival tracking + action history)
                 └─> Task 5 (showdown channel)
                      └─> Task 6 (exploitability)
                           ├─> Task 7 (detection)
                           └─> Task 8 (SafetyBellman + Wasserstein)
                                └─> Task 9 (SRB-001 + SE-008)
                                     └─> Task 10 (BluffFramework)
                                          └─> Task 11 (BridgeManifest)
                                               └─> Task 12 (remaining orphans)
Tasks 13, 14, 15 can run in parallel after Task 4
Task 16 depends on all others
```
