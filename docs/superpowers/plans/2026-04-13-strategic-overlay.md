# Strategic Overlay Controller — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert StrategicEngine from a standalone POMDP solver to a policy overlay that filters real EVs from adaptive/multiway engines through rival-model beliefs and safety constraints.

**Architecture:** StrategicOverlay.filter() accepts upstream ActionEvaluations from the adaptive or multiway engines, applies belief-weighted penalties and soft veto, and returns an OverlayResult with the re-ranked actions and diagnostics. StrategicLifecycleHelper centralizes lifecycle glue with stable rival identity mapping so hall, ACPC, and Slumbot don't drift.

**Tech Stack:** Scala 3.8.1, munit 1.2.2, existing sicfun.holdem engine/strategic/runtime packages

**Spec:** `docs/superpowers/specs/2026-04-13-strategic-overlay-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `engine/StrategicOverlay.scala` | **New.** Pure filter: penalty, soft veto, re-rank. No side effects. |
| `engine/StrategicEngine.scala` | **Modify.** Add 3-arg `decide()` overlay overload, `_lastOverlayResult` field. Old 2-arg stays deprecated. |
| `engine/HeroDecisionPipeline.scala` | **Modify.** Remove `throw UnsupportedOperationException`, rewrite `decideHeroStrategic()` to run adaptive upstream then overlay. |
| `runtime/StrategicLifecycleHelper.scala` | **New.** Shared lifecycle: init, session, hand, observe, decideWithOverlay. Owns Position→PlayerId mapping. |
| `runtime/TexasHoldemPlayingHall.scala` | **Modify.** Strategic branch calls adaptive/multiway then overlay via helper. CLI parser accepts "strategic". |
| `runtime/protocol/AcpcMatchRunner.scala` | **Modify.** Add strategic engine, lifecycle helper, CLI "strategic" option. |
| `runtime/protocol/SlumbotMatchRunner.scala` | **Modify.** Same as ACPC. |
| `runtime/StrategicAdvisorBridge.scala` | **Modify.** Call overlay decide(), print OverlayResult diagnostics. |
| `test/engine/StrategicOverlayTest.scala` | **New.** Unit tests for filter logic. |
| `test/engine/StrategicEngineOverlayTest.scala` | **New.** Integration tests for the new decide() overload. |
| `test/runtime/StrategicLifecycleHelperTest.scala` | **New.** Tests for position mapping and EV extraction. |

---

### Task 1: StrategicOverlay — Types and Filter Logic

**Files:**
- Create: `src/main/scala/sicfun/holdem/engine/StrategicOverlay.scala`
- Create: `src/test/scala/sicfun/holdem/engine/StrategicOverlayTest.scala`

- [ ] **Step 1: Write the StrategicOverlay types and filter method**

```scala
// src/main/scala/sicfun/holdem/engine/StrategicOverlay.scala
package sicfun.holdem.engine

import sicfun.holdem.types.*
import sicfun.holdem.strategic.types.*
import sicfun.holdem.strategic.state.*

/** Overlay result types — NOT a DecisionEvaluationBundle. */
final case class OverlayInput(
    gameState: GameState,
    upstreamEvs: Vector[ActionEvaluation],
    rivalBeliefs: Map[PlayerId, StrategicRivalBelief],
    exploitationStates: Map[PlayerId, ExploitationState],
    robustLowerBounds: Option[Array[Double]],
    config: StrategicEngine.Config
)

final case class OverlayResult(
    selectedAction: PokerAction,
    rankedActions: Vector[ActionEvaluation],
    softVetoed: Vector[(PokerAction, String)],
    adjustments: Vector[OverlayAdjustment],
    upstreamAction: PokerAction,
    upstreamSource: UpstreamSource
)

final case class OverlayAdjustment(
    action: PokerAction,
    originalEv: Double,
    adjustedEv: Double,
    reason: String
)

enum UpstreamSource:
  case Adaptive
  case Multiway(opponentCount: Int)

/** Pure policy filter: belief-weighted penalties + soft veto + re-rank.
  *
  * No side effects — caller is responsible for storing the result.
  */
object StrategicOverlay:

  /** Belief-weighted aggression penalty coefficients (calibrate from benchmark data). */
  private val CallPenaltyCoeff = -0.1
  private val CheckPenaltyCoeff = -0.05

  def filter(input: OverlayInput): OverlayResult =
    val upstreamEvs = input.upstreamEvs
    if upstreamEvs.isEmpty then
      return OverlayResult(
        selectedAction = PokerAction.Fold,
        rankedActions = Vector.empty,
        softVetoed = Vector.empty,
        adjustments = Vector.empty,
        upstreamAction = PokerAction.Fold,
        upstreamSource = UpstreamSource.Adaptive
      )

    val upstreamAction = upstreamEvs.maxBy(_.expectedValue).action
    val potFraction = input.gameState.pot / math.max(input.gameState.stackSize, 1.0)

    // Step 1: Compute aggregate bluff mass across all rivals
    val bluffMass = aggregateBluffMass(input.rivalBeliefs)

    // Step 2: Apply belief-weighted penalty per action
    val adjustments = Vector.newBuilder[OverlayAdjustment]
    val adjusted = upstreamEvs.map { ae =>
      val penalty = beliefPenalty(ae.action, bluffMass, potFraction)
      val adjustedEv = ae.expectedValue + penalty
      if math.abs(penalty) > 1e-12 then
        adjustments += OverlayAdjustment(
          action = ae.action,
          originalEv = ae.expectedValue,
          adjustedEv = adjustedEv,
          reason = f"belief-penalty: bluffMass=$bluffMass%.3f potFrac=$potFraction%.3f"
        )
      ActionEvaluation(ae.action, adjustedEv)
    }

    // Step 3: Soft veto from robust lower bounds
    val epsilonTotal = input.config.epsilonBase + input.config.exploitConfig.epsilonAdapt
    val vetoed = Vector.newBuilder[(PokerAction, String)]
    input.robustLowerBounds.foreach { bounds =>
      var i = 0
      while i < math.min(bounds.length, adjusted.size) do
        if bounds(i) < -epsilonTotal then
          vetoed += ((adjusted(i).action, f"robust lower bound ${bounds(i)}%.4f < ${-epsilonTotal}%.4f"))
        i += 1
    }

    // Step 4: Re-rank by adjusted EV
    val ranked = adjusted.sortBy(-_.expectedValue)
    val vetoedResult = vetoed.result()
    val vetoedActions = vetoedResult.map(_._1).toSet

    // Step 5: Select best action (prefer non-vetoed, fallback to best robust lower bound)
    val selectedAction =
      ranked.find(ae => !vetoedActions.contains(ae.action)).map(_.action)
        .getOrElse {
          // All soft-vetoed: pick action with highest robust lower bound
          input.robustLowerBounds match
            case Some(bounds) if bounds.nonEmpty =>
              val bestIdx = bounds.indices.maxBy(bounds(_))
              if bestIdx < upstreamEvs.size then upstreamEvs(bestIdx).action
              else ranked.head.action
            case _ => ranked.head.action
        }

    OverlayResult(
      selectedAction = selectedAction,
      rankedActions = ranked,
      softVetoed = vetoedResult,
      adjustments = adjustments.result(),
      upstreamAction = upstreamAction,
      upstreamSource = UpstreamSource.Adaptive // caller overrides for multiway
    )

  /** Sum of P(Bluff) across all rival beliefs. */
  private[engine] def aggregateBluffMass(
      beliefs: Map[PlayerId, StrategicRivalBelief]
  ): Double =
    if beliefs.isEmpty then 0.0
    else beliefs.values.map(_.typePosterior.probabilityOf(StrategicClass.Bluff)).sum

  /** Penalty for passive actions against likely-aggressive opponents. */
  private[engine] def beliefPenalty(
      action: PokerAction,
      bluffMass: Double,
      potFraction: Double
  ): Double =
    val coeff = action match
      case PokerAction.Call => CallPenaltyCoeff
      case PokerAction.Check => CheckPenaltyCoeff
      case _ => 0.0
    bluffMass * coeff * potFraction
```

- [ ] **Step 2: Write tests for StrategicOverlay.filter**

```scala
// src/test/scala/sicfun/holdem/engine/StrategicOverlayTest.scala
package sicfun.holdem.engine

import munit.FunSuite
import sicfun.holdem.types.*
import sicfun.holdem.strategic.types.*
import sicfun.holdem.strategic.state.*
import sicfun.holdem.strategic.exploitation.ExploitationState
import sicfun.core.DiscreteDistribution

class StrategicOverlayTest extends FunSuite:

  private def minimalState(pot: Double = 100.0, toCall: Double = 50.0): GameState =
    GameState(
      street = Street.Flop,
      board = Board.empty,
      pot = pot,
      toCall = toCall,
      position = Position.Button,
      stackSize = 1000.0,
      betHistory = Vector.empty
    )

  private def uniformBelief: StrategicRivalBelief =
    StrategicRivalBelief.uniform

  private def bluffHeavyBelief: StrategicRivalBelief =
    StrategicRivalBelief(
      DiscreteDistribution.fromWeights(
        StrategicClass.values.toVector,
        Vector(0.05, 0.80, 0.10, 0.05)  // Value=0.05, Bluff=0.80, StructuralBluff=0.10, Mixed=0.05
      )
    )

  private def defaultConfig: StrategicEngine.Config = StrategicEngine.Config()

  private def makeInput(
      evs: Vector[ActionEvaluation],
      beliefs: Map[PlayerId, StrategicRivalBelief] = Map(PlayerId("v1") -> uniformBelief),
      robustBounds: Option[Array[Double]] = None,
      pot: Double = 100.0
  ): OverlayInput =
    OverlayInput(
      gameState = minimalState(pot = pot),
      upstreamEvs = evs,
      rivalBeliefs = beliefs,
      exploitationStates = beliefs.map((id, _) => id -> ExploitationState.initial(defaultConfig.exploitConfig)),
      robustLowerBounds = robustBounds,
      config = defaultConfig
    )

  test("filter passes through upstream action when no penalties apply"):
    val evs = Vector(
      ActionEvaluation(PokerAction.Fold, -50.0),
      ActionEvaluation(PokerAction.Raise(2.0), 30.0)
    )
    val result = StrategicOverlay.filter(makeInput(evs))
    assertEquals(result.selectedAction, PokerAction.Raise(2.0))
    assertEquals(result.upstreamAction, PokerAction.Raise(2.0))
    assert(result.softVetoed.isEmpty)

  test("filter penalizes Call against bluff-heavy opponent"):
    val evs = Vector(
      ActionEvaluation(PokerAction.Fold, -50.0),
      ActionEvaluation(PokerAction.Call, 5.0),
      ActionEvaluation(PokerAction.Raise(2.0), 4.0)
    )
    val beliefs = Map(PlayerId("v1") -> bluffHeavyBelief)
    val result = StrategicOverlay.filter(makeInput(evs, beliefs = beliefs))
    // Call had EV=5.0 upstream but penalty = 0.80 * -0.1 * (100/1000) = -0.008
    // Raise had EV=4.0, no penalty
    // After penalty, Call=4.992, Raise=4.0 — Call still wins but margin tightened
    // With larger pot fractions the penalty can flip the ranking
    assertEquals(result.upstreamAction, PokerAction.Call)
    assert(result.adjustments.exists(_.action == PokerAction.Call))

  test("filter with large pot fraction flips Call vs Raise"):
    val evs = Vector(
      ActionEvaluation(PokerAction.Call, 1.0),
      ActionEvaluation(PokerAction.Raise(2.0), 0.5)
    )
    // pot=800, stack=1000, potFraction=0.8
    // Call penalty = 0.80 * -0.1 * 0.8 = -0.064, adjusted=0.936
    // Raise penalty = 0, adjusted=0.5
    // Call still wins here. Let's make it tighter:
    val evsTight = Vector(
      ActionEvaluation(PokerAction.Call, 0.10),
      ActionEvaluation(PokerAction.Raise(2.0), 0.05)
    )
    val beliefs = Map(PlayerId("v1") -> bluffHeavyBelief)
    val result = StrategicOverlay.filter(makeInput(evsTight, beliefs = beliefs, pot = 800.0))
    // penalty = 0.80 * -0.1 * 0.8 = -0.064, Call adjusted = 0.10 - 0.064 = 0.036
    // Raise adjusted = 0.05
    // Raise wins
    assertEquals(result.selectedAction, PokerAction.Raise(2.0))
    assert(result.upstreamAction == PokerAction.Call)

  test("soft veto flags actions below threshold but does not remove them"):
    val evs = Vector(
      ActionEvaluation(PokerAction.Call, 5.0),
      ActionEvaluation(PokerAction.Raise(2.0), 10.0)
    )
    val bounds = Array(-2.0, 3.0) // Call robust bound = -2.0
    val result = StrategicOverlay.filter(makeInput(evs, robustBounds = Some(bounds)))
    assert(result.softVetoed.nonEmpty)
    assertEquals(result.softVetoed.head._1, PokerAction.Call)
    // Raise is still selected (highest EV and not vetoed)
    assertEquals(result.selectedAction, PokerAction.Raise(2.0))

  test("when all actions soft-vetoed, selects best robust lower bound"):
    val evs = Vector(
      ActionEvaluation(PokerAction.Call, 5.0),
      ActionEvaluation(PokerAction.Raise(2.0), 10.0)
    )
    // Both below threshold
    val bounds = Array(-2.0, -1.0)
    val result = StrategicOverlay.filter(makeInput(evs, robustBounds = Some(bounds)))
    assertEquals(result.softVetoed.size, 2)
    // Best robust lower bound is -1.0 (Raise, index 1)
    assertEquals(result.selectedAction, PokerAction.Raise(2.0))

  test("empty upstream EVs returns Fold"):
    val result = StrategicOverlay.filter(makeInput(Vector.empty))
    assertEquals(result.selectedAction, PokerAction.Fold)

  test("aggregateBluffMass sums P(Bluff) across rivals"):
    val beliefs = Map(
      PlayerId("v1") -> bluffHeavyBelief,
      PlayerId("v2") -> uniformBelief
    )
    val mass = StrategicOverlay.aggregateBluffMass(beliefs)
    // v1 = 0.80, v2 = 0.25
    assertEqualsDouble(mass, 1.05, 1e-10)
```

- [ ] **Step 3: Run tests to verify they compile and pass**

Run: `sbt "testOnly sicfun.holdem.engine.StrategicOverlayTest"`
Expected: All 7 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/main/scala/sicfun/holdem/engine/StrategicOverlay.scala \
        src/test/scala/sicfun/holdem/engine/StrategicOverlayTest.scala
git commit -m "feat(engine): add StrategicOverlay filter with belief penalty and soft veto"
```

---

### Task 2: StrategicEngine — New Overlay decide() Overload

**Files:**
- Modify: `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala`
- Create: `src/test/scala/sicfun/holdem/engine/StrategicEngineOverlayTest.scala`

- [ ] **Step 1: Add `_lastOverlayResult` field and new `decide()` overload to StrategicEngine**

In `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala`, after the existing `_lastBundle` field (line ~30):

```scala
  private var _lastOverlayResult: Option[OverlayResult] = None

  def lastOverlayResult: Option[OverlayResult] = _lastOverlayResult
```

Then add the new overlay decide method. Place it right after the existing `decide(gameState, candidateActions)` method (which ends around line ~275):

```scala
  /** Overlay decision path: filters upstream EVs through rival-model beliefs.
    *
    * This is the Phase 1 entry point. The caller obtains EVs from the adaptive
    * or multiway engine and passes them here. The overlay applies belief-weighted
    * penalties and soft veto, then returns a full OverlayResult trace.
    *
    * The old two-arg decide() is deprecated but not removed — certification
    * tests may still exercise it.
    */
  def decide(
      gameState: GameState,
      candidateActions: Vector[PokerAction],
      upstreamEvs: Vector[ActionEvaluation]
  ): OverlayResult =
    require(_sessionState != null, "Session not initialized")
    require(_handActive, "No hand in progress")
    require(candidateActions.nonEmpty, "No candidate actions")

    val session = _sessionState.nn

    // Attach robust lower bounds from certification if previously run.
    // The bundle carries per-action lower bounds directly in robustActionLowerBounds
    // (computed as min_profile Q[a] in StrategicEngine.decideWPomcp).
    // Do NOT use rootLosses — those are non-negative losses (baselineValue - lowerBound).
    val robustBounds = _lastBundle.flatMap { bundle =>
      val bounds = bundle.robustActionLowerBounds
      if bounds != null && bounds.nonEmpty then Some(bounds) else None
    }

    val input = OverlayInput(
      gameState = gameState,
      upstreamEvs = upstreamEvs,
      rivalBeliefs = session.rivalBeliefs,
      exploitationStates = session.exploitationStates,
      robustLowerBounds = robustBounds,
      config = config
    )

    val result = StrategicOverlay.filter(input)
    _lastOverlayResult = Some(result)

    // Update deployment tracking from overlay result
    val deploySession = _sessionState.nn
    val beliefs = deploySession.rivalBeliefs.values
    if beliefs.nonEmpty then
      val avgEntropy = beliefs.map { b =>
        val probs = StrategicClass.values.map(c => b.typePosterior.probabilityOf(c))
        -probs.filter(_ > 0).map(p => p * math.log(p)).sum
      }.sum / beliefs.size
      val summary = DeploymentBeliefSummary(
        beliefEntropy = avgEntropy,
        exploitabilitySnapshot = Ev(0.0), // Phase 1: no pointwise exploit from overlay
        timestamp = System.currentTimeMillis()
      )
      val updatedDeploy = deploySession.deploymentSet.add(summary)
      _sessionState = deploySession.copy(deploymentSet = updatedDeploy)

    result
```

- [ ] **Step 2: Add `@deprecated` annotation to the old two-arg decide**

Find the existing `def decide(gameState: GameState, candidateActions: Vector[PokerAction]): PokerAction` (around line 201) and add before it:

```scala
  /** @deprecated Use the 3-arg overlay overload with upstream EVs instead. Kept for certification tests. */
  @deprecated("Use overlay decide(gameState, candidates, upstreamEvs) instead", "v0.33")
```

- [ ] **Step 3: Write tests for the new overlay decide**

```scala
// src/test/scala/sicfun/holdem/engine/StrategicEngineOverlayTest.scala
package sicfun.holdem.engine

import munit.FunSuite
import sicfun.holdem.types.*
import sicfun.holdem.strategic.types.*
import sicfun.holdem.strategic.state.*

class StrategicEngineOverlayTest extends FunSuite:

  private def minimalState: GameState =
    GameState(
      street = Street.Flop,
      board = Board.empty,
      pot = 100.0,
      toCall = 50.0,
      position = Position.Button,
      stackSize = 1000.0,
      betHistory = Vector.empty
    )

  private def testHeroCards: HoleCards =
    val as = sicfun.core.Card.parse("As").get
    val kh = sicfun.core.Card.parse("Kh").get
    HoleCards.from(Vector(as, kh))

  test("overlay decide returns OverlayResult with correct upstream action"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand(testHeroCards)

    val evs = Vector(
      ActionEvaluation(PokerAction.Fold, -50.0),
      ActionEvaluation(PokerAction.Call, 5.0),
      ActionEvaluation(PokerAction.Raise(2.0), 10.0)
    )
    val result = engine.decide(minimalState, evs.map(_.action), evs)
    assertEquals(result.upstreamAction, PokerAction.Raise(2.0))
    assertEquals(result.selectedAction, PokerAction.Raise(2.0))
    assert(engine.lastOverlayResult.isDefined)

  test("overlay decide requires initialized session"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    val evs = Vector(ActionEvaluation(PokerAction.Call, 5.0))
    interceptMessage[IllegalArgumentException]("Session not initialized") {
      engine.decide(minimalState, Vector(PokerAction.Call), evs)
    }

  test("overlay decide requires active hand"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    val evs = Vector(ActionEvaluation(PokerAction.Call, 5.0))
    interceptMessage[IllegalArgumentException]("No hand in progress") {
      engine.decide(minimalState, Vector(PokerAction.Call), evs)
    }
```

- [ ] **Step 4: Run tests**

Run: `sbt "testOnly sicfun.holdem.engine.StrategicEngineOverlayTest"`
Expected: All 3 tests pass.

- [ ] **Step 5: Run existing StrategicEngine tests to verify no regressions**

Run: `sbt "testOnly sicfun.holdem.engine.StrategicEngineTest"`
Expected: All existing tests pass unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/main/scala/sicfun/holdem/engine/StrategicEngine.scala \
        src/test/scala/sicfun/holdem/engine/StrategicEngineOverlayTest.scala
git commit -m "feat(engine): add overlay decide() overload to StrategicEngine"
```

---

### Task 3: StrategicLifecycleHelper

**Files:**
- Create: `src/main/scala/sicfun/holdem/runtime/StrategicLifecycleHelper.scala`
- Create: `src/test/scala/sicfun/holdem/runtime/StrategicLifecycleHelperTest.scala`

- [ ] **Step 1: Write StrategicLifecycleHelper**

```scala
// src/main/scala/sicfun/holdem/runtime/StrategicLifecycleHelper.scala
package sicfun.holdem.runtime

import sicfun.holdem.types.*
import sicfun.holdem.engine.{StrategicEngine, OverlayResult, UpstreamSource, StrategicOverlay}
import sicfun.holdem.engine.inference.{ActionRecommendation, ActionEvaluation}
import sicfun.holdem.strategic.types.*

/** Per-engine strategic lifecycle helper. Each caller (hall, ACPC, Slumbot, advisor)
  * creates its own instance so position mapping state does not bleed across runners.
  *
  * Identity contract: Callers supply stable PlayerId per physical opponent, NOT
  * position-derived IDs. In heads-up match play the remote opponent flips between
  * Button and BigBlind each hand — Position.toString would split belief tracks.
  *
  *   - ACPC/Slumbot: PlayerId("villain")
  *   - Hall: PlayerId from VillainProfile name
  *   - Advisor: PlayerId("villain")
  */
private[holdem] final class StrategicLifecycleHelper(
    val engine: StrategicEngine
):

  /** Mutable position→rivalId mapping, updated each hand as seats rotate. */
  private var _positionMapping: Map[Position, PlayerId] = Map.empty

  def positionMapping: Map[Position, PlayerId] = _positionMapping

  def initSession(
      rivalIds: Vector[PlayerId],
      positionMapping: Map[Position, PlayerId],
      seatInfo: Map[PlayerId, StrategicEngine.RivalSeatInfo] = Map.empty
  ): Unit =
    _positionMapping = positionMapping
    engine.initSession(rivalIds, seatInfo)

  def updatePositionMapping(positionMapping: Map[Position, PlayerId]): Unit =
    _positionMapping = positionMapping

  def startHand(heroCards: HoleCards): Unit =
    engine.startHand(heroCards)

  /** Routes villain action to the correct stable rival ID via position mapping.
    * Silently ignores if the position has no mapping (e.g., hero's own position).
    */
  def observeVillainAction(
      villainPosition: Position,
      action: PokerAction,
      gameState: GameState
  ): Unit =
    _positionMapping.get(villainPosition).foreach { rivalId =>
      engine.observeAction(rivalId, action, gameState)
    }

  def endHand(): Unit =
    engine.endHand()

  /** Extract EVs from upstream ActionRecommendation and run overlay filter. */
  def decideWithOverlay(
      gameState: GameState,
      candidates: Vector[PokerAction],
      upstreamRecommendation: ActionRecommendation,
      upstreamSource: UpstreamSource = UpstreamSource.Adaptive
  ): OverlayResult =
    val upstreamEvs = upstreamRecommendation.actionEvaluations
    val result = engine.decide(gameState, candidates, upstreamEvs)
    // Override upstream source if caller specifies multiway
    if upstreamSource != UpstreamSource.Adaptive then
      result.copy(upstreamSource = upstreamSource)
    else
      result

private[holdem] object StrategicLifecycleHelper:
  def create(config: StrategicEngine.Config = StrategicEngine.Config()): StrategicLifecycleHelper =
    new StrategicLifecycleHelper(new StrategicEngine(config))
```

- [ ] **Step 2: Write tests for StrategicLifecycleHelper**

```scala
// src/test/scala/sicfun/holdem/runtime/StrategicLifecycleHelperTest.scala
package sicfun.holdem.runtime

import munit.FunSuite
import sicfun.holdem.types.*
import sicfun.holdem.engine.{StrategicEngine, OverlayResult, UpstreamSource}
import sicfun.holdem.engine.inference.{ActionRecommendation, ActionEvaluation}
import sicfun.holdem.strategic.types.*
import sicfun.holdem.types.EquityEstimate

class StrategicLifecycleHelperTest extends FunSuite:

  private def testHeroCards: HoleCards =
    val as = sicfun.core.Card.parse("As").get
    val kh = sicfun.core.Card.parse("Kh").get
    HoleCards.from(Vector(as, kh))

  private def minimalState: GameState =
    GameState(
      street = Street.Flop,
      board = Board.empty,
      pot = 100.0,
      toCall = 50.0,
      position = Position.Button,
      stackSize = 1000.0,
      betHistory = Vector.empty
    )

  test("position mapping routes villain actions to stable rival ID"):
    val villainId = PlayerId("villain")
    val helper = StrategicLifecycleHelper.create()
    helper.initSession(
      rivalIds = Vector(villainId),
      positionMapping = Map(Position.BigBlind -> villainId)
    )
    helper.startHand(testHeroCards)

    // Observe action at BigBlind position — should route to "villain"
    helper.observeVillainAction(Position.BigBlind, PokerAction.Call, minimalState)
    // Verify belief was updated (not an error)
    assert(helper.engine.sessionState.rivalBeliefs.contains(villainId))

  test("position mapping update reflects seat rotation"):
    val villainId = PlayerId("villain")
    val helper = StrategicLifecycleHelper.create()
    helper.initSession(
      rivalIds = Vector(villainId),
      positionMapping = Map(Position.BigBlind -> villainId)
    )
    // Rotate: villain now on Button
    helper.updatePositionMapping(Map(Position.Button -> villainId))
    assertEquals(
      helper.positionMapping,
      Map(Position.Button -> villainId)
    )

  test("decideWithOverlay extracts EVs and returns OverlayResult"):
    val villainId = PlayerId("villain")
    val helper = StrategicLifecycleHelper.create()
    helper.initSession(
      rivalIds = Vector(villainId),
      positionMapping = Map(Position.BigBlind -> villainId)
    )
    helper.startHand(testHeroCards)

    val recommendation = ActionRecommendation(
      heroEquity = EquityEstimate(mean = 0.6, variance = 0.01, stderr = 0.003, trials = 1000, winRate = 0.5, tieRate = 0.1, lossRate = 0.4),
      actionEvaluations = Vector(
        ActionEvaluation(PokerAction.Call, 5.0),
        ActionEvaluation(PokerAction.Raise(2.0), 10.0)
      ),
      bestAction = PokerAction.Raise(2.0)
    )
    val candidates = Vector(PokerAction.Call, PokerAction.Raise(2.0))
    val result = helper.decideWithOverlay(minimalState, candidates, recommendation)
    assertEquals(result.upstreamAction, PokerAction.Raise(2.0))
    assertEquals(result.upstreamSource, UpstreamSource.Adaptive)

  test("decideWithOverlay respects multiway upstream source"):
    val helper = StrategicLifecycleHelper.create()
    helper.initSession(
      rivalIds = Vector(PlayerId("v1"), PlayerId("v2")),
      positionMapping = Map(Position.BigBlind -> PlayerId("v1"), Position.UTG -> PlayerId("v2"))
    )
    helper.startHand(testHeroCards)

    val recommendation = ActionRecommendation(
      heroEquity = EquityEstimate(mean = 0.5, variance = 0.01, stderr = 0.003, trials = 1000, winRate = 0.4, tieRate = 0.1, lossRate = 0.5),
      actionEvaluations = Vector(ActionEvaluation(PokerAction.Check, 0.0)),
      bestAction = PokerAction.Check
    )
    val result = helper.decideWithOverlay(
      minimalState, Vector(PokerAction.Check), recommendation,
      upstreamSource = UpstreamSource.Multiway(2)
    )
    assertEquals(result.upstreamSource, UpstreamSource.Multiway(2))
```

- [ ] **Step 3: Run tests**

Run: `sbt "testOnly sicfun.holdem.runtime.StrategicLifecycleHelperTest"`
Expected: All 4 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/StrategicLifecycleHelper.scala \
        src/test/scala/sicfun/holdem/runtime/StrategicLifecycleHelperTest.scala
git commit -m "feat(runtime): add StrategicLifecycleHelper with stable rival identity mapping"
```

---

### Task 4: HeroDecisionPipeline — Unblock Strategic Mode

**Files:**
- Modify: `src/main/scala/sicfun/holdem/engine/HeroDecisionPipeline.scala`

This is the first-class blocker for ACPC/Slumbot integration.

- [ ] **Step 1: Replace UnsupportedOperationException with adaptive-upstream overlay dispatch**

In `src/main/scala/sicfun/holdem/engine/HeroDecisionPipeline.scala`, find the Strategic branch in `decideHero` (around line 147):

Replace:
```scala
      case HeroMode.Strategic =>
        throw new UnsupportedOperationException(
          "Strategic mode requires StrategicDecisionContext — use decideHeroStrategic()")
```

With:
```scala
      case HeroMode.Strategic =>
        throw new UnsupportedOperationException(
          "Strategic mode in decideHero requires StrategicDecisionContext — use decideHeroStrategic()")
```

(Keep the throw — `decideHero` is the wrong entry point for Strategic. The real fix is in `decideHeroStrategic`.)

- [ ] **Step 2: Rewrite `decideHeroStrategic` to run adaptive upstream then overlay**

Replace the existing `decideHeroStrategic` method (around line 152):

```scala
  /** Strategic mode decision dispatch. */
  def decideHeroStrategic(ctx: StrategicDecisionContext): PokerAction =
    ctx.engine.decide(ctx.state, ctx.candidates)
```

With:

First, update `StrategicDecisionContext` to hold a `StrategicLifecycleHelper` instead of a raw `StrategicEngine`:

```scala
  case class StrategicDecisionContext(
      state: GameState,
      candidates: Vector[PokerAction],
      helper: StrategicLifecycleHelper  // was: engine: StrategicEngine
  )
```

Then replace the method:

```scala
  /** Strategic overlay decision dispatch.
    *
    * Runs the adaptive engine for upstream EVs (via heroCtx.engine, a
    * RealTimeAdaptiveEngine), then filters through the strategic overlay
    * (via strategicCtx.helper, a StrategicLifecycleHelper wrapping StrategicEngine).
    *
    * Returns the overlay-selected action. The full OverlayResult is stored
    * in strategicCtx.helper.engine.lastOverlayResult for diagnostics.
    */
  def decideHeroStrategic(
      strategicCtx: StrategicDecisionContext,
      heroCtx: HeroDecisionContext
  ): PokerAction =
    // 1. Run adaptive engine for upstream EVs
    val adaptiveResult = heroCtx.engine.decide(
      hero = heroCtx.hero,
      state = heroCtx.state,
      folds = heroCtx.folds,
      villainPos = heroCtx.villainPos,
      observations = heroCtx.observations,
      candidateActions = heroCtx.candidates,
      decisionBudgetMillis = heroCtx.decisionBudgetMillis,
      rng = new Random(heroCtx.rng.nextLong())
    )
    // 2. Overlay: filter adaptive EVs through strategic beliefs
    val overlayResult = strategicCtx.helper.decideWithOverlay(
      heroCtx.state,
      heroCtx.candidates,
      adaptiveResult.decision.recommendation
    )
    overlayResult.selectedAction

  // decideHeroStrategicLegacy removed — old 2-arg StrategicEngine.decide() is
  // already deprecated independently; no need for a wrapper that calls it.
```

- [ ] **Step 3: Run existing tests to verify no regressions**

Run: `sbt "testOnly sicfun.holdem.engine.*"`
Expected: All engine tests pass. The old single-arg `decideHeroStrategic` callers (hall) will be updated in Task 5.

- [ ] **Step 4: Commit**

```bash
git add src/main/scala/sicfun/holdem/engine/HeroDecisionPipeline.scala
git commit -m "feat(engine): rewrite decideHeroStrategic to run adaptive upstream then overlay"
```

---

### Task 5: PlayingHall — Full Migration to StrategicLifecycleHelper

**Files:**
- Modify: `src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala`

The hall has **11 sites** that reference `strategicEngineOpt`/`StrategicEngine`. All must migrate
to `strategicHelperOpt: Option[StrategicLifecycleHelper]`. This task covers every site.

- [ ] **Step 1: Change field declaration and all parameter-threading sites**

Add import (with the other imports, top of file):
```scala
import sicfun.holdem.engine.UpstreamSource
```

**Line 338** — field declaration. Replace:
```scala
    private var strategicEngineOpt = Option.empty[StrategicEngine]
```
With:
```scala
    private var strategicHelperOpt = Option.empty[StrategicLifecycleHelper]
```

**Line 453** — named arg passed to `resolveHand()`. Replace:
```scala
        strategicEngineOpt = strategicEngineOpt
```
With:
```scala
        strategicHelperOpt = strategicHelperOpt
```

**Line 633** — `resolveHand()` parameter. Replace:
```scala
      strategicEngineOpt: Option[StrategicEngine]
```
With:
```scala
      strategicHelperOpt: Option[StrategicLifecycleHelper]
```

**Line 648** — named arg passed to `HandResolver` constructor. Replace:
```scala
      strategicEngineOpt = strategicEngineOpt
```
With:
```scala
      strategicHelperOpt = strategicHelperOpt
```

**Line 679** — `HandResolver` constructor parameter. Replace:
```scala
      strategicEngineOpt: Option[StrategicEngine]
```
With:
```scala
      strategicHelperOpt: Option[StrategicLifecycleHelper]
```

- [ ] **Step 2: Update the Strategic branch in HandResolver.decideHero**

In `TexasHoldemPlayingHall.scala`, find the `HeroMode.Strategic` case (around line 1023):

Replace:
```scala
        case HeroMode.Strategic =>
          strategicEngineOpt match
            case Some(engine) =>
              engine.decide(state, candidates)
            case None =>
              candidates.find(_ != PokerAction.Fold).getOrElse(PokerAction.Fold)
```

With:
```scala
        case HeroMode.Strategic =>
          strategicHelperOpt match
            case Some(helper) =>
              // Run adaptive/multiway upstream, then overlay
              val adaptiveEngine = if street == Street.Preflop then preflopEngine else postflopEngine
              val upstreamRec = multiwayRecommendationFor(
                actor = heroPosition,
                state = state,
                candidateActions = candidates,
              ).getOrElse {
                // Fallback to heads-up adaptive
                adaptiveEngine.decide(
                  hero = deal.holeCardsFor(heroPosition),
                  state = state,
                  folds = foldsForInference(heroPosition, focusVillainPosition),
                  villainPos = focusVillainPosition,
                  observations = playerObservations(focusVillainPosition),
                  candidateActions = candidates,
                  decisionBudgetMillis = Some(1L),
                  rng = new java.util.Random(rng.nextLong())
                ).decision.recommendation
              }
              val livePlayers = participatingPositions.size - foldedPositions.size
              val source = if livePlayers > 2 then
                UpstreamSource.Multiway(livePlayers - 1)
              else UpstreamSource.Adaptive
              helper.decideWithOverlay(
                state, candidates, upstreamRec, source
              ).selectedAction
            case None =>
              candidates.find(_ != PokerAction.Fold).getOrElse(PokerAction.Fold)
```

(`UpstreamSource` import was already added in Step 1.)

- [ ] **Step 3: Update initialization in initializeArtifact()**

Find `initializeArtifact()` (around line 410). Replace:
```scala
      if config.heroMode == HeroMode.Strategic then
        strategicEngineOpt = Some(new StrategicEngine(StrategicEngine.Config()))
```
With:
```scala
      if config.heroMode == HeroMode.Strategic then
        val helper = StrategicLifecycleHelper.create()
        // Register ALL villain pool names up front so rotating profiles
        // accumulate beliefs across hands (not just the first-hand subset).
        val allRivalIds = config.villainPool.map(p => PlayerId(p.name)).distinct.toVector
        helper.initSession(rivalIds = allRivalIds, positionMapping = Map.empty)
        strategicHelperOpt = Some(helper)
```

This solves the rotating-pool problem: every profile name that could appear in any hand
is registered once at session start. Per-hand, only the position mapping changes.

- [ ] **Step 4: Update per-hand position mapping in playHand()**

Find the session initialization block (around line 433). Replace the entire existing
`strategicEngineOpt.foreach { engine => ... }` block with:

```scala
      strategicHelperOpt.foreach { helper =>
        // Map current-hand positions to stable VillainProfile names
        val villainPositions = tableScenario.activePositions
          .filterNot(_ == config.heroPosition)
        val posMapping = villainPositions.map { pos =>
          pos -> PlayerId(tableScenario.villainProfileByPosition(pos).name)
        }.toMap
        helper.updatePositionMapping(posMapping)
      }
```

No init-check needed here — session was initialized in `initializeArtifact()`.

- [ ] **Step 5: Migrate startHand() and endHand() in HandResolver.play()**

**Line 723** — startHand. Replace:
```scala
      strategicEngineOpt.foreach(_.startHand(deal.holeCardsFor(heroPosition)))
```
With:
```scala
      strategicHelperOpt.foreach(_.startHand(deal.holeCardsFor(heroPosition)))
```

**Line 728** — endHand. Replace:
```scala
      strategicEngineOpt.foreach(_.endHand())
```
With:
```scala
      strategicHelperOpt.foreach(_.endHand())
```

- [ ] **Step 6: Fix the identity bug — replace direct observeAction with helper.observeVillainAction()**

This is the critical identity fix. **Lines 858-861** currently do:
```scala
      strategicEngineOpt.foreach { engine =>
        val rivalId = sicfun.holdem.strategic.types.PlayerId(position.toString)
        engine.observeAction(rivalId, action, state)
      }
```

`PlayerId(position.toString)` creates seat-derived IDs like `"Button"` / `"BigBlind"` that
split belief tracks when seats rotate. Replace with:
```scala
      strategicHelperOpt.foreach(_.observeVillainAction(position, action, state))
```

The helper routes through its position mapping → stable `PlayerId` from VillainProfile name.

- [ ] **Step 7: Update heroModeOpt CLI parser to accept "strategic"**

Find `heroModeOpt` (around line 2391). Replace:
```scala
          case _          => Left("--heroStyle must be one of: adaptive, gto")
```
With:
```scala
          case "strategic" => Right(HeroMode.Strategic)
          case _           => Left("--heroStyle must be one of: adaptive, gto, strategic")
```

- [ ] **Step 8: Compile check**

Run: `sbt compile`
Expected: Compiles cleanly. All 11 `strategicEngineOpt` sites are now `strategicHelperOpt`.

- [ ] **Step 9: Run hall-related tests**

Run: `sbt "testOnly sicfun.holdem.validation.HeadsUpSimulatorTest"`
Expected: Pass (this exercises the hall).

- [ ] **Step 10: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala
git commit -m "feat(runtime): full hall migration to StrategicLifecycleHelper — fix identity bug, rotating pool"
```

---

### Task 6: AcpcMatchRunner — Add Strategic Support

**Files:**
- Modify: `src/main/scala/sicfun/holdem/runtime/protocol/AcpcMatchRunner.scala`

- [ ] **Step 1: Add strategic engine field and initialization to Runner**

In `AcpcMatchRunner.scala`, find the `Runner` class fields (look for `private val engine` and `private val artifact`). Add after them:

```scala
    private val strategicHelperOpt: Option[StrategicLifecycleHelper] =
      if config.heroMode == HeroMode.Strategic then
        val helper = StrategicLifecycleHelper.create()
        helper.initSession(
          rivalIds = Vector(PlayerId("villain")),
          positionMapping = Map.empty // updated per hand
        )
        Some(helper)
      else None
```

Add imports at the top:
```scala
import sicfun.holdem.strategic.types.PlayerId
import sicfun.holdem.runtime.StrategicLifecycleHelper
```

- [ ] **Step 2: Add strategic lifecycle calls in the hand loop**

**Hand start (line ~891)** — after `liveHandOpt = Some(created)` / when a new LiveHand is created:
```scala
        strategicHelperOpt.foreach { helper =>
          helper.updatePositionMapping(
            Map(created.villainPosition -> PlayerId("villain"))
          )
          helper.startHand(created.heroHole)
        }
```

**Villain actions (line ~948)** — inside `processSteps`, in the `if step.relativeActor == 1` branch
(villain actions), add after the existing `villainObservations :+` line:
```scala
          strategicHelperOpt.foreach(_.observeVillainAction(
            liveHand.villainPosition, step.action, step.stateBefore
          ))
```

**Hand end (line ~894)** — inside the `if matchState.parsed.handOver` block, before `recordOutcome`:
```scala
          strategicHelperOpt.foreach(_.endHand())
```

- [ ] **Step 3: Update decideHero for Strategic mode**

Replace the existing `decideHero` method:

```scala
    private def decideHero(
        hero: HoleCards,
        state: GameState,
        villainPosition: Position,
        villainObservations: Vector[VillainObservation],
        candidates: Vector[PokerAction]
    ): PokerAction =
      config.heroMode match
        case HeroMode.Strategic =>
          strategicHelperOpt match
            case Some(helper) =>
              HeroDecisionPipeline.decideHeroStrategic(
                HeroDecisionPipeline.StrategicDecisionContext(state, candidates, helper),
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
                  cfrIterations = config.cfrIterations,
                  cfrVillainHands = config.cfrVillainHands,
                  cfrEquityTrials = config.cfrEquityTrials,
                  rng = rng
                )
              )
            case None =>
              candidates.find(_ != PokerAction.Fold).getOrElse(PokerAction.Fold)
        case _ =>
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
              cfrIterations = config.cfrIterations,
              cfrVillainHands = config.cfrVillainHands,
              cfrEquityTrials = config.cfrEquityTrials,
              rng = rng
            )
          )
```

- [ ] **Step 4: Update CLI parser to accept "strategic"**

Find the `heroModeOption` method (around line 1175):

Replace:
```scala
          case _ => Left("--heroMode must be one of: adaptive, gto")
```

With:
```scala
          case "strategic" => Right(HeroMode.Strategic)
          case _ => Left("--heroMode must be one of: adaptive, gto, strategic")
```

Update the usage string (around line 1201):

Replace:
```scala
      |  --heroMode=adaptive         adaptive|gto
```

With:
```scala
      |  --heroMode=adaptive         adaptive|gto|strategic
```

- [ ] **Step 5: Compile check**

Run: `sbt compile`
Expected: Compiles cleanly.

- [ ] **Step 6: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/protocol/AcpcMatchRunner.scala
git commit -m "feat(protocol): add strategic mode support to AcpcMatchRunner"
```

---

### Task 7: SlumbotMatchRunner — Add Strategic Support

**Files:**
- Modify: `src/main/scala/sicfun/holdem/runtime/protocol/SlumbotMatchRunner.scala`

Same structure as Task 6 but targeting Slumbot's different loop shape and local variables.

- [ ] **Step 1: Add strategic helper field and initialization**

Same as Task 6 Step 1: add `strategicHelperOpt` field with init/session, and imports for
`PlayerId` and `StrategicLifecycleHelper`.

```scala
    private val strategicHelperOpt: Option[StrategicLifecycleHelper] =
      if config.heroMode == HeroMode.Strategic then
        val helper = StrategicLifecycleHelper.create()
        helper.initSession(
          rivalIds = Vector(PlayerId("villain")),
          positionMapping = Map.empty
        )
        Some(helper)
      else None
```

- [ ] **Step 2: Add strategic lifecycle calls in playHand()**

Slumbot's `playHand()` uses local vars (not a LiveHand instance). The key locals are
`heroHole`, `villainPosition`, `step.action`, `step.stateBefore`.

**Hand start (line ~689)** — after `var pendingHeroRaise = false`, add:
```scala
        strategicHelperOpt.foreach { helper =>
          helper.updatePositionMapping(
            Map(villainPosition -> PlayerId("villain"))
          )
          helper.startHand(heroHole)
        }
```

**Villain actions (line ~702)** — inside `newSteps.foreach`, in the `if step.relativeActor == 1`
branch, add after the existing `villainObservations = villainObservations :+` line:
```scala
          strategicHelperOpt.foreach(_.observeVillainAction(
            villainPosition, step.action, step.stateBefore
          ))
```

**Hand end (line ~715)** — inside `response.winnings match { case Some(winnings) =>`,
before `return HandOutcome(...)`:
```scala
          strategicHelperOpt.foreach(_.endHand())
```

- [ ] **Step 3: Update decideHero for Strategic mode**

Same pattern as Task 6 Step 3: add a `HeroMode.Strategic` branch that delegates to
`HeroDecisionPipeline.decideHeroStrategic(strategicCtx, heroCtx)`. The `decideHero`
signature is identical to ACPC's:

```scala
    private def decideHero(
        hero: HoleCards,
        state: GameState,
        villainPosition: Position,
        villainObservations: Vector[VillainObservation],
        candidates: Vector[PokerAction]
    ): PokerAction =
      config.heroMode match
        case HeroMode.Strategic =>
          strategicHelperOpt match
            case Some(helper) =>
              HeroDecisionPipeline.decideHeroStrategic(
                HeroDecisionPipeline.StrategicDecisionContext(state, candidates, helper),
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
                  cfrIterations = config.cfrIterations,
                  cfrVillainHands = config.cfrVillainHands,
                  cfrEquityTrials = config.cfrEquityTrials,
                  rng = rng
                )
              )
            case None =>
              candidates.find(_ != PokerAction.Fold).getOrElse(PokerAction.Fold)
        case _ =>
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
              cfrIterations = config.cfrIterations,
              cfrVillainHands = config.cfrVillainHands,
              cfrEquityTrials = config.cfrEquityTrials,
              rng = rng
            )
          )
```

- [ ] **Step 4: Update CLI parser to accept "strategic"**

Find the `heroModeOption` method (around line 1007):

Replace:
```scala
          case _ => Left("--heroMode must be one of: adaptive, gto")
```

With:
```scala
          case "strategic" => Right(HeroMode.Strategic)
          case _ => Left("--heroMode must be one of: adaptive, gto, strategic")
```

Update the usage string:

Replace:
```scala
      |  --heroMode=adaptive         adaptive|gto
```

With:
```scala
      |  --heroMode=adaptive         adaptive|gto|strategic
```

- [ ] **Step 5: Compile check**

Run: `sbt compile`
Expected: Compiles cleanly.

- [ ] **Step 6: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/protocol/SlumbotMatchRunner.scala
git commit -m "feat(protocol): add strategic mode support to SlumbotMatchRunner"
```

---

### Task 8: StrategicAdvisorBridge — Migrate to Overlay

**Files:**
- Modify: `src/main/scala/sicfun/holdem/runtime/StrategicAdvisorBridge.scala`

- [ ] **Step 1: Rewrite onAdvise to use overlay decide with upstream EVs**

Replace the entire `StrategicAdvisorBridge.scala`:

```scala
package sicfun.holdem.runtime

import sicfun.holdem.types.*
import sicfun.holdem.engine.{StrategicEngine, OverlayResult, UpstreamSource}
import sicfun.holdem.engine.inference.{ActionRecommendation, ActionEvaluation}
import sicfun.holdem.strategic.types.PlayerId

/** Adapts AdvisorSession lifecycle commands to StrategicEngine operations.
  *
  * Centralizes the mapping between the interactive session model (HandSnapshot-based)
  * and the StrategicEngine API (GameState/PlayerId-based) so that AdvisorSession
  * stays focused on user interaction.
  *
  * The bridge works directly with StrategicEngine (not StrategicLifecycleHelper) because
  * the advisor uses PlayerId("villain") for all rivals and doesn't need position mapping.
  */
object StrategicAdvisorBridge:

  private val VillainId = PlayerId("villain")

  /** Called at the start of each new hand. Initializes session if needed, then starts a new hand. */
  def onNewHand(engine: StrategicEngine): Unit =
    if !engine.isSessionInitialized then
      engine.initSession(rivalIds = Vector(VillainId))
    engine.startHand()

  /** Called when a villain action is observed. Feeds the action to the strategic engine. */
  def onVillainAction(engine: StrategicEngine, action: PokerAction, h: HandSnapshot): Unit =
    val gameState = GameState(
      street = h.street, board = h.board, pot = h.pot, toCall = h.toCall,
      position = h.villainPosition, stackSize = h.villainStack, betHistory = h.betHistory
    )
    engine.observeAction(VillainId, action, gameState)

  /** Called during advise to get strategic overlay diagnostics.
    *
    * Accepts the upstream ActionRecommendation that AdvisorSession already computed
    * via its adaptive engine (lines 603-614 of AdvisorSession.scala). When provided,
    * the overlay filters real EVs. When None (legacy callers), falls back to zero EVs.
    */
  def onAdvise(
      engine: StrategicEngine,
      gameState: GameState,
      candidates: Vector[PokerAction],
      upstreamRecommendation: Option[ActionRecommendation] = None
  ): Vector[String] =
    try
      val upstreamEvs: Vector[ActionEvaluation] = upstreamRecommendation match
        case Some(rec) => rec.actionEvaluations
        case None      => candidates.map(a => ActionEvaluation(a, 0.0))

      val result = engine.decide(gameState, candidates, upstreamEvs)
      val out = Vector.newBuilder[String]

      // Print overlay diagnostics
      out += f"  Overlay: selected=${result.selectedAction} upstream=${result.upstreamAction} source=${result.upstreamSource}"
      if result.adjustments.nonEmpty then
        result.adjustments.foreach { adj =>
          out += f"  Overlay: ${adj.action} EV ${adj.originalEv}%.3f → ${adj.adjustedEv}%.3f (${adj.reason})"
        }
      if result.softVetoed.nonEmpty then
        result.softVetoed.foreach { (action, reason) =>
          out += f"  Overlay: SOFT VETO ${action} — $reason"
        }

      out.result()
    catch
      case _: Exception => Vector.empty

  /** Called on villain showdown. Feeds showdown data to the strategic engine. */
  def onVillainShowdown(engine: StrategicEngine, cards: HoleCards): Unit =
    engine.endHand(Some(Map(VillainId -> cards)))
```

- [ ] **Step 2: Update AdvisorSession to pass the adaptive result**

In `AdvisorSession.scala`, find the `onAdvise` call site (around line 617-619) where it currently calls:
```scala
StrategicAdvisorBridge.onAdvise(se, gameState, candidates)
```

The adaptive result is already available from line ~603-614 as `result.decision.recommendation`. Update to:
```scala
StrategicAdvisorBridge.onAdvise(se, gameState, candidates, Some(result.decision.recommendation))
```

This wires real upstream EVs from the adaptive engine into the overlay — no zero-EV fallback needed.

- [ ] **Step 3: Check that AdvisorSession compiles with the updated bridge**

The `onAdvise` signature changed (added optional `upstreamRecommendation` param with default `None`). Any callers that pass `(engine, gameState, candidates)` still compile because the new parameter has a default value.

Run: `sbt compile`
Expected: Compiles cleanly.

- [ ] **Step 4: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/StrategicAdvisorBridge.scala \
        src/main/scala/sicfun/holdem/runtime/AdvisorSession.scala
git commit -m "feat(runtime): migrate StrategicAdvisorBridge to overlay decide with real upstream EVs"
```

---

### Task 9: Full Test Suite Verification

**Files:** None (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `sbt test`
Expected: All 1750+ tests pass.

- [ ] **Step 2: If any tests fail, investigate and fix**

Common failure modes:
- Import changes from package restructuring — fix imports
- Signature mismatches in `decideHeroStrategic` — callers need the new two-arg overload
- `StrategicAdvisorBridge.onAdvise` — existing callers must still compile with optional param

- [ ] **Step 3: Commit any fixes**

```bash
git add -u
git commit -m "fix: resolve test failures from strategic overlay integration"
```

---

### Task 10: Smoke Test — Hall Self-Play with Strategic Mode

**Files:** None (runtime verification)

- [ ] **Step 1: Run a short hall self-play with strategic mode**

Run: `sbt "runMain sicfun.holdem.runtime.TexasHoldemPlayingHall --hands=100 --heroStyle=strategic --seed=42 --outDir=data/strategic-smoke"`

Expected: Completes without errors, writes hands.tsv and learning.tsv.

- [ ] **Step 2: Verify overlay diagnostics are present**

Check the output for overlay-related log lines. The strategic engine should be initialized and the overlay decide path should be called.

- [ ] **Step 3: Commit verification notes (optional)**

If the smoke test reveals issues, fix them and commit. Otherwise, no commit needed.
