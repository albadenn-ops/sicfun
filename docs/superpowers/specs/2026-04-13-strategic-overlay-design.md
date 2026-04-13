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

- StrategicEngine accepts upstream EVs and applies belief-weighted penalties
- Soft veto: flag actions where robust lower bounds (if available from existing
  certification) fall below a threshold, but don't hard-block
- Distinct `OverlayResult` trace (not DecisionEvaluationBundle)
- Wire into hall, ACPC, Slumbot with shared lifecycle helper
- Benchmarks: mbb/hand, action distribution, veto rate

### Phase 2: Baseline/Exploit Blending + Hard Safety Veto (follow-up)

- Profile-conditioned upstream queries (run adaptive/multiway per rival profile)
- True baseline vs exploit EV blending with beta interpolation
- Hard safety veto using profile-conditioned worst-case values
- Ground certification path in real EVs

This spec covers Phase 1 only.

## Data Contract

### OverlayInput

```scala
final case class OverlayInput(
    gameState: GameState,
    exploitEvs: Vector[ActionEvaluation],     // from upstream engine (belief-adapted)
    baselineEvs: Vector[ActionEvaluation],     // from equilibrium baseline if available
    rivalBeliefs: Map[PlayerId, StrategicRivalBelief],
    exploitationStates: Map[PlayerId, ExploitationState],
    robustLowerBounds: Option[Array[Double]],  // from existing certification if run
    config: StrategicEngine.Config
)
```

- `exploitEvs`: Per-action chip EVs from the adaptive engine's exploit
  recommendation (archetype-adapted posterior). Uses existing
  `ActionEvaluation(action, expectedValue)` from `RangeInferenceEngine.scala:51`.
- `baselineEvs`: Per-action chip EVs from the equilibrium baseline path (when
  `EquilibriumBaselineConfig` is active in adaptive engine). Empty vector when
  baseline is unavailable.
- `robustLowerBounds`: Optional per-action worst-case EVs from existing
  certification solvers. Phase 1 uses these for soft veto only when the caller
  has already run certification. Phase 2 will compute them from
  profile-conditioned upstream queries.

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

### StrategicUpstreamMode

```scala
enum StrategicUpstreamMode:
  case Adaptive  // always use adaptive engine as upstream
  case Gto       // always use GTO (posterior + CFR) as upstream
  case Auto      // adaptive for production, gto for offline analysis
```

Added to `StrategicEngine.Config`. Determines which upstream engine produces the
EVs that the overlay filters. Default: `Auto`.

## Component Design

### 1. StrategicOverlay (new object, `engine` package)

```
StrategicOverlay.filter(input: OverlayInput): OverlayResult
```

Phase 1 filter logic:

1. **Start with exploit EVs** as the action ranking.
2. **Belief-weighted penalty**: For each action `a`, compute penalty =
   `sum over rivals R of (beliefMass(R, Bluff) * aggressionPenalty(a))` where
   `aggressionPenalty(Call) = -0.1 * potFraction`, `aggressionPenalty(Check) =
   -0.05 * potFraction`, `aggressionPenalty(Raise) = 0`, `aggressionPenalty(Fold)
   = 0`. The intuition: against likely-aggressive opponents, passive actions lose
   more value than the exploit EV suggests because the opponent will barrel.
   These coefficients are initial values — calibrate from benchmark data.
3. **Baseline anchoring**: When baseline EVs are available, for each action
   compute `adjustedEv = (1 - blendWeight) * exploitEv + blendWeight *
   baselineEv` where `blendWeight` is derived from the minimum beta across
   exploitation states (low beta = stay close to baseline).
4. **Soft veto**: When `robustLowerBounds` are present, flag any action whose
   robust lower bound is below `-(epsilonBase + epsilonAdapt)`. Mark as soft
   vetoed in diagnostics but do not remove from ranking.
5. **Re-rank** remaining actions by adjusted EV. Select top action.
6. **Fallback**: If all actions are soft-vetoed, select the action with the
   highest robust lower bound (most defensive).

### 2. StrategicEngine.decide() — new overload

```scala
def decide(
    gameState: GameState,
    candidateActions: Vector[PokerAction],
    exploitEvs: Vector[ActionEvaluation],
    baselineEvs: Vector[ActionEvaluation] = Vector.empty
): OverlayResult
```

Flow:
1. Build `OverlayInput` from session state + arguments
2. Optionally run existing certification (if `config.runCertification` is true)
   to populate `robustLowerBounds`
3. Call `StrategicOverlay.filter(input)`
4. Store `OverlayResult` for diagnostics
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

  def initSession(
      engine: StrategicEngine,
      rivalPositions: Vector[Position],
      seatInfo: Map[PlayerId, StrategicEngine.RivalSeatInfo] = Map.empty
  ): Unit

  def startHand(engine: StrategicEngine, heroCards: HoleCards): Unit

  def observeVillainAction(
      engine: StrategicEngine,
      villainPosition: Position,
      action: PokerAction,
      gameState: GameState
  ): Unit

  def endHand(engine: StrategicEngine): Unit

  /** Run upstream engine then overlay filter. */
  def decideWithOverlay(
      engine: StrategicEngine,
      gameState: GameState,
      candidates: Vector[PokerAction],
      upstreamRecommendation: ActionRecommendation,
      baselineRecommendation: Option[ActionRecommendation] = None
  ): OverlayResult
```

All three callers (hall, ACPC, Slumbot) delegate to this helper. The helper
handles PlayerId conversion (`Position.toString` → `PlayerId`) and EV
extraction from `ActionRecommendation`.

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
      // 1. Run adaptive/multiway to get real EVs
      val recommendation = if livePlayers > 2 then
        multiwayRecommendation(state, candidates, ...)
      else
        adaptiveRecommendation(state, candidates, ...)
      // 2. Get baseline if available
      val baseline = equilibriumBaseline(state, candidates, ...)
      // 3. Overlay
      val result = StrategicLifecycleHelper.decideWithOverlay(
        engine, state, candidates, recommendation, baseline
      )
      result.selectedAction
    case None => /* fallback to adaptive */
```

The hall calls adaptive/multiway directly (not through HeroDecisionPipeline)
because the hall has direct access to engines and multiway infrastructure.

### 5. Match Runner Integration (ACPC + Slumbot)

Both runners are heads-up only. Integration via `HeroDecisionPipeline`:

**HeroDecisionPipeline changes:**
- Remove the `throw UnsupportedOperationException` for Strategic mode
- `decideHeroStrategic()` updated to accept upstream context:

```scala
def decideHeroStrategic(
    ctx: StrategicDecisionContext,
    heroCtx: HeroDecisionContext
): OverlayResult =
  // 1. Run adaptive engine for upstream EVs
  val adaptiveResult = ctx.engine.decide(...) // upstream adaptive
  // ... extract recommendation
  // 2. Overlay
  StrategicLifecycleHelper.decideWithOverlay(...)
```

**AcpcMatchRunner changes:**
- Add `strategicEngineOpt: Option[StrategicEngine]` to `Runner`
- Initialize via `StrategicLifecycleHelper.initEngine()` when mode is Strategic
- Call lifecycle methods in hand loop (same as hall pattern)
- Strategic branch in `decideHero()` calls pipeline

**SlumbotMatchRunner changes:**
- Same pattern as ACPC

### 6. CLI / Config Wiring

- `TexasHoldemPlayingHall.Config`: already has `heroMode: HeroMode` — no change
  needed, but verify the CLI parser accepts "strategic"
- `AcpcMatchRunner`: add `--hero-mode` flag parsing to accept
  `adaptive|gto|strategic`
- `SlumbotMatchRunner`: same
- `StrategicEngine.Config`: add `upstreamMode: StrategicUpstreamMode = Auto`

## What Does NOT Change

- `StrategicEngine.observeAction()` — rival belief updates via kernel pipeline
- `StrategicEngine.initSession/startHand/endHand` — session lifecycle
- All certification machinery (DecisionEvaluationBundle, CertificationResult,
  POMDP solvers, WPomcpRuntime, PftDpwRuntime)
- PokerPomcpFormulation / PokerPftFormulation — not deleted, still available for
  certification and the deprecated decide() path
- The `strategic` sub-packages (bridge/, solver/, all the belief/dynamics/kernel
  types)
- StrategicAdvisorBridge — advisor session integration stays as-is

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
| `runtime/AcpcMatchRunner.scala` | Modify | Add strategic engine lifecycle + dispatch |
| `runtime/SlumbotMatchRunner.scala` | Modify | Add strategic engine lifecycle + dispatch |
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
