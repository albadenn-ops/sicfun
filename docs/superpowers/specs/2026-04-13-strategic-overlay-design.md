# Strategic Overlay Controller — Design Spec

**Date**: 2026-04-13
**Status**: Draft
**Branch**: `feat/strategic-overlay`

## Problem

StrategicEngine currently invents its own poker EVs via toy POMDP formulations
(PokerPomcpFormulation, PokerPftFormulation) that use `heroBucket / 9.0` proxies,
hardcoded class priors, and deterministic street transitions. It runs 6 WPomcp
solves or PftDpw four-world solves to select actions — completely independent of
the existing adaptive and multiway engines that produce real equity-calibrated EVs.

This makes strategic a standalone toy solver rather than a controller over
production-quality decision engines.

## Solution

Convert StrategicEngine from an action selector to a **policy overlay** that:

1. Receives real per-action EVs from upstream engines (adaptive for heads-up,
   multiway for >2 players)
2. Applies rival-model beliefs and safety constraints to re-rank, constrain, or
   veto actions
3. Produces a distinct overlay trace showing what changed and why

The existing POMDP certification machinery (DecisionEvaluationBundle,
CertificationResult, robustness bounds) stays as-is for now. Grounding
certification in real EVs is deferred to a follow-up.

## Phased Delivery

### Phase 1: Overlay Rerank + Diagnostics + Soft Veto

- Upstream source: adaptive (heads-up) or multiway (>2 players) only. GTO
  upstream deferred — the current GTO path (`solveShallowDecisionPolicy`)
  returns sampled actions, not per-action EVs.
- StrategicEngine accepts upstream EVs and applies belief-weighted penalties
- Soft veto: flag actions where robust lower bounds (if available from existing
  certification) fall below a threshold, but don't hard-block
- Distinct `OverlayResult` trace (not DecisionEvaluationBundle)
- Wire into hall, ACPC, Slumbot with shared lifecycle helper
- Update StrategicAdvisorBridge to use overlay path and OverlayResult diagnostics
- Stable rival identity contract: callers supply PlayerId, not position-derived
- CLI flags: camelCase to match existing `heroMode` convention in all three
  runners
- Benchmarks: mbb/hand, action distribution, veto rate

### Phase 2: Baseline/Exploit Blending + Hard Safety Veto (follow-up)

- Profile-conditioned upstream queries (run adaptive/multiway per rival profile)
- True baseline vs exploit EV blending with beta interpolation
- Hard safety veto using profile-conditioned worst-case values
- Ground certification path in real EVs
- GTO upstream: add EV-producing API to CFR solver path, then wire as upstream
  option

This spec covers Phase 1 only.

## Data Contract

### OverlayInput

```scala
final case class OverlayInput(
    gameState: GameState,
    upstreamEvs: Vector[ActionEvaluation],     // from adaptive or multiway engine
    rivalBeliefs: Map[PlayerId, StrategicRivalBelief],
    exploitationStates: Map[PlayerId, ExploitationState],
    robustLowerBounds: Option[Array[Double]],  // from existing certification if run
    config: StrategicEngine.Config
)
```

- `upstreamEvs`: Per-action chip EVs from the upstream engine's recommendation
  (adaptive for heads-up, multiway for >2 players). Uses existing
  `ActionEvaluation(action, expectedValue)` from `RangeInferenceEngine.scala:51`.
- `robustLowerBounds`: Optional per-action worst-case EVs from existing
  certification solvers. Phase 1 uses these for soft veto only when the caller
  has already run certification. Phase 2 will add `baselineEvs` and
  `profileEvs` for true blending and hard veto.

### OverlayResult

```scala
final case class OverlayResult(
    selectedAction: PokerAction,
    rankedActions: Vector[ActionEvaluation],   // re-ranked by safety-adjusted EV
    softVetoed: Vector[(PokerAction, String)], // action + reason
    adjustments: Vector[OverlayAdjustment],    // what changed and why
    upstreamAction: PokerAction,               // what upstream would have picked
    upstreamSource: UpstreamSource             // Adaptive or Multiway
)

final case class OverlayAdjustment(
    action: PokerAction,
    originalEv: Double,
    adjustedEv: Double,
    reason: String  // e.g. "belief-penalty: rival R1 posterior 0.7 Bluff"
)

enum UpstreamSource:
  case Adaptive
  case Multiway(opponentCount: Int)
```

This is **not** a DecisionEvaluationBundle. The overlay trace describes what the
filter did to upstream EVs, not what a POMDP solver computed. The existing bundle
remains available if certification is also run.

### Upstream Source

Phase 1 upstream is always adaptive (heads-up) or multiway (>2 players). The
caller determines which path runs and passes the resulting
`ActionRecommendation` to the overlay.

GTO upstream is deferred to Phase 2 because the current GTO path
(`HoldemCfrSolver.solveShallowDecisionPolicy`) returns action probabilities,
not per-action EVs. Adding an EV-producing CFR API is Phase 2 scope.

No `StrategicUpstreamMode` enum in Phase 1. When Phase 2 adds GTO upstream,
the enum and config field will be introduced then.

## Component Design

### 1. StrategicOverlay (new object, `engine` package)

```
StrategicOverlay.filter(input: OverlayInput): OverlayResult
```

Phase 1 filter logic:

1. **Start with upstream EVs** as the action ranking.
2. **Belief-weighted penalty**: For each action `a`, compute penalty =
   `sum over rivals R of (beliefMass(R, Bluff) * aggressionPenalty(a))` where
   `aggressionPenalty(Call) = -0.1 * potFraction`, `aggressionPenalty(Check) =
   -0.05 * potFraction`, `aggressionPenalty(Raise) = 0`, `aggressionPenalty(Fold)
   = 0`. The intuition: against likely-aggressive opponents, passive actions lose
   more value than the upstream EV suggests because the opponent will barrel.
   These coefficients are initial values — calibrate from benchmark data.
3. **Soft veto**: When `robustLowerBounds` are present, flag any action whose
   robust lower bound is below `-(epsilonBase + epsilonAdapt)`. Mark as soft
   vetoed in diagnostics but do not remove from ranking.
4. **Re-rank** remaining actions by adjusted EV. Select top action.
5. **Fallback**: If all actions are soft-vetoed, select the action with the
   highest robust lower bound (most defensive).

**Explicitly deferred to Phase 2**: Baseline/exploit EV blending with beta
interpolation. Phase 1 has no `baselineEvs` field — the upstream engine's
recommendation is used as-is before penalty/veto.

### 2. StrategicEngine.decide() — new overload

```scala
def decide(
    gameState: GameState,
    candidateActions: Vector[PokerAction],
    upstreamEvs: Vector[ActionEvaluation]
): OverlayResult
```

Flow:
1. Build `OverlayInput` from session state + arguments
2. If existing certification was previously run for this decision (via the
   deprecated path or external call), attach `robustLowerBounds` from
   `_lastBundle`. Otherwise `None`.
3. Call `StrategicOverlay.filter(input)`
4. Store `OverlayResult` in `_lastOverlayResult` for diagnostics
5. Update deployment tracking from overlay result
6. Return result

The old `decide(gameState, candidateActions): PokerAction` stays as a deprecated
path that internally calls the toy formulations. Not deleted — certification
tests may still exercise it.

### 3. StrategicLifecycleHelper (new object, `runtime` package)

Shared helper to prevent hall/ACPC/Slumbot lifecycle drift:

```scala
private[holdem] object StrategicLifecycleHelper:

  def initEngine(config: StrategicEngine.Config): StrategicEngine

  /** Callers supply stable rival IDs, NOT position-derived IDs.
    *
    * Identity contract: In heads-up match play, the remote opponent flips
    * between Button and BigBlind each hand. Converting Position.toString →
    * PlayerId would split one opponent into multiple belief tracks. Callers
    * must provide a stable ID per physical opponent:
    *   - ACPC/Slumbot: PlayerId("villain") (single remote opponent)
    *   - Hall: PlayerId from VillainProfile name (stable across seat rotation)
    *   - Advisor: PlayerId("villain") (as StrategicAdvisorBridge already does)
    *
    * The positionMapping tracks which seat each rival currently occupies so
    * that observeAction can route by position to the correct stable ID.
    */
  def initSession(
      engine: StrategicEngine,
      rivalIds: Vector[PlayerId],
      positionMapping: Map[Position, PlayerId],
      seatInfo: Map[PlayerId, StrategicEngine.RivalSeatInfo] = Map.empty
  ): Unit

  /** Update position → rivalId mapping at hand start (seats rotate). */
  def updatePositionMapping(
      positionMapping: Map[Position, PlayerId]
  ): Unit

  def startHand(engine: StrategicEngine, heroCards: HoleCards): Unit

  /** Routes villain action to the correct stable rival ID via position mapping. */
  def observeVillainAction(
      engine: StrategicEngine,
      villainPosition: Position,
      action: PokerAction,
      gameState: GameState
  ): Unit

  def endHand(engine: StrategicEngine): Unit

  /** Extract EVs from upstream recommendation and run overlay filter. */
  def decideWithOverlay(
      engine: StrategicEngine,
      gameState: GameState,
      candidates: Vector[PokerAction],
      upstreamRecommendation: ActionRecommendation
  ): OverlayResult
```

All three callers (hall, ACPC, Slumbot) delegate to this helper. The helper
owns the `Position → PlayerId` mapping and EV extraction from
`ActionRecommendation`.

### 4. PlayingHall Integration

In `HandResolver.decideHero()`, the `HeroMode.Strategic` branch changes from:

```scala
// OLD
case HeroMode.Strategic =>
  strategicEngineOpt match
    case Some(engine) => engine.decide(state, candidates)
    case None => /* fallback */
```

To:

```scala
// NEW
case HeroMode.Strategic =>
  strategicEngineOpt match
    case Some(engine) =>
      // 1. Run adaptive (heads-up) or multiway (>2 players) to get real EVs
      val recommendation = if livePlayers > 2 then
        multiwayRecommendation(state, candidates, ...)
      else
        adaptiveRecommendation(state, candidates, ...)
      // 2. Overlay: filter upstream EVs through strategic beliefs
      val result = StrategicLifecycleHelper.decideWithOverlay(
        engine, state, candidates, recommendation
      )
      result.selectedAction
    case None => /* fallback to adaptive */
```

The hall calls adaptive/multiway directly (not through HeroDecisionPipeline)
because the hall has direct access to engines and multiway infrastructure.

### 5. Match Runner Integration (ACPC + Slumbot)

Both runners live under `runtime/protocol/` and are heads-up only.

**Blockers** (these are not partially landed — they are all future work):
- `HeroDecisionPipeline` currently hard-blocks Strategic mode with
  `throw UnsupportedOperationException` and routes `decideHeroStrategic()`
  to the old two-arg `engine.decide()` toy path. This is the first-class
  blocker for ACPC/Slumbot integration.
- Both protocol runners (`AcpcMatchRunner`, `SlumbotMatchRunner`) reject
  "strategic" at the CLI and only build the adaptive engine. The success
  criteria requiring `--hero-mode strategic` acceptance are entirely
  future work.

**HeroDecisionPipeline changes** (`engine/HeroDecisionPipeline.scala`):
- Remove the `throw UnsupportedOperationException` for Strategic mode
- `decideHeroStrategic()` updated signature:

```scala
def decideHeroStrategic(
    strategicCtx: StrategicDecisionContext,
    heroCtx: HeroDecisionContext
): OverlayResult =
  // 1. Run adaptive engine for upstream EVs via heroCtx.engine (RealTimeAdaptiveEngine)
  //    NOT strategicCtx.engine (which is StrategicEngine — that would recurse)
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
  // 2. Overlay — filter adaptive EVs through strategic beliefs
  StrategicLifecycleHelper.decideWithOverlay(
    strategicCtx.engine,
    heroCtx.state,
    heroCtx.candidates,
    adaptiveResult.decision.recommendation
  )
```

Note: `strategicCtx.engine` is the `StrategicEngine` (overlay).
`heroCtx.engine` is the `RealTimeAdaptiveEngine` (upstream EV source).
These are different types — the old sketch incorrectly called
`ctx.engine.decide(...)` which would have invoked the strategic engine
as its own upstream.

**AcpcMatchRunner changes** (`runtime/protocol/AcpcMatchRunner.scala`):
- Add `strategicEngineOpt: Option[StrategicEngine]` to `Runner`
- Add `heroMode` CLI flag parsing to accept `adaptive|gto|strategic`
- Initialize via `StrategicLifecycleHelper.initEngine()` when mode is Strategic
- Call lifecycle methods in hand loop (same as hall pattern)
- Strategic branch in `decideHero()` calls pipeline

**SlumbotMatchRunner changes** (`runtime/protocol/SlumbotMatchRunner.scala`):
- Same pattern as ACPC

### 6. CLI / Config Wiring

- `TexasHoldemPlayingHall.Config`: already has `heroMode: HeroMode` — verify
  the CLI parser accepts "strategic" (it uses camelCase `heroMode` flag)
- `runtime/protocol/AcpcMatchRunner`: add `heroMode` flag parsing to accept
  `adaptive|gto|strategic` (camelCase, matching existing convention)
- `runtime/protocol/SlumbotMatchRunner`: same pattern as ACPC

## What Does NOT Change

- `StrategicEngine.observeAction()` — rival belief updates via kernel pipeline
- `StrategicEngine.initSession/startHand/endHand` — session lifecycle
- All certification machinery (DecisionEvaluationBundle, CertificationResult,
  POMDP solvers, WPomcpRuntime, PftDpwRuntime)
- PokerPomcpFormulation / PokerPftFormulation — not deleted, still available for
  certification and the deprecated decide() path
- The `strategic` sub-packages (bridge/, solver/, all the belief/dynamics/kernel
  types)

## What Gets Updated (Not Deprecated, Not Unchanged)

- `StrategicAdvisorBridge` (`runtime/StrategicAdvisorBridge.scala`) — currently
  calls the deprecated two-arg `engine.decide(gameState, candidates)` and
  prints `lastDecisionBundle` diagnostics from the toy solver path. Phase 1
  scope: migrate to the overlay `decide()` overload and `OverlayResult`
  diagnostics. The advisor needs an upstream `ActionRecommendation` to feed
  the overlay, which means `AdvisorSession` must run the adaptive engine
  before calling the overlay. Specifically:
  1. `onAdvise()` must obtain upstream EVs from `RealTimeAdaptiveEngine`
  2. Call the new overlay `decide()` overload with those EVs
  3. Print `OverlayResult` diagnostics instead of `DecisionEvaluationBundle`
  4. Continue using `PlayerId("villain")` (already correct for stable identity)
  Without this update, the advisor would silently report different logic than
  the actual player, creating a user-visible divergence.

## What Gets Deprecated

- `StrategicEngine.decide(gameState, candidateActions): PokerAction` — the old
  two-arg overload that runs toy formulations for action selection
- `decideWPomcp()` and `decidePftDpw()` as action-selection paths (kept for
  certification)
- `estimateHeroBucket()` for action-selection context

## File Impact Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `engine/StrategicOverlay.scala` | **New** | Filter logic: rerank, penalty, soft veto |
| `engine/StrategicEngine.scala` | Modify | New `decide()` overload, deprecate old path |
| `engine/HeroDecisionPipeline.scala` | Modify | Strategic dispatch with upstream EVs |
| `runtime/StrategicLifecycleHelper.scala` | **New** | Shared lifecycle helper |
| `runtime/TexasHoldemPlayingHall.scala` | Modify | Strategic branch runs adaptive/multiway first |
| `runtime/protocol/AcpcMatchRunner.scala` | Modify | Add strategic engine lifecycle + dispatch + CLI flag |
| `runtime/protocol/SlumbotMatchRunner.scala` | Modify | Add strategic engine lifecycle + dispatch + CLI flag |
| `runtime/StrategicAdvisorBridge.scala` | Modify | Use overlay decide() + OverlayResult diagnostics |
| `runtime/StrategicAdvisorBridge.scala` | Modify | Use overlay decide() + OverlayResult diagnostics |
| `types/HeroMode.scala` | No change | Strategic already exists in enum |
| `types/PokerFormatting.scala` | No change | Already formats "strategic" |

## Benchmark Plan

Once wired, run these configurations:

1. **Hall self-play**: strategic-vs-adaptive, strategic-vs-gto (1000+ hands each)
2. **ACPC**: strategic mode against ACPC dealer (standard AAAI configuration)
3. **Slumbot**: strategic mode against Slumbot API (100+ hands)

Metrics per configuration:
- mbb/hand (mean, std, 95% CI)
- Action distribution (fold/check/call/raise percentages)
- Overlay veto rate (% of decisions where overlay changed upstream action)
- Exploitation beta trajectory over session
- Overlay adjustment magnitude distribution

Compare against adaptive and gto baselines on identical deal sequences (seeded RNG).

## Success Criteria

Phase 1 is complete when:
1. `sbt test` passes with all existing + new tests
2. All three runners (hall, ACPC, Slumbot) accept `--hero-mode strategic` and
   run to completion
3. At least one benchmark comparison (strategic vs adaptive, 1000 hands)
   produces a report with the metrics above
4. OverlayResult diagnostics show non-trivial filtering (veto rate > 0% on at
   least some hands)

## Open Questions (Phase 2)

- How many profile-conditioned upstream queries are acceptable per decision?
  (Currently 4 profiles × adaptive engine = 4× latency)
- Should certification be replaced entirely or run in parallel with overlay?
- Can the multiway path produce profile-conditioned EVs efficiently?
