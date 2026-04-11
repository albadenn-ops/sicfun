# Reductionism Resolution Design

**Date:** 2026-04-07
**Scope:** Resolve all 35 entries in `ReductionismManifestTest` (32 real implementations + 3 pre-resolved)
**Approach:** Dependency-ordered phases (Approach A)
**Branch:** `feat/adaptive-proof-harness-9max`

## Pre-Resolution: 3 Already Done

| ID | File | Rationale |
|---|---|---|
| SP-001 | SpotPolarization.scala | `UniformPolarization` reclassified as real exact baseline (analogous to `BlindActionKernel`) |
| SP-002 | SpotPolarization.scala | `PosteriorDivergencePolarization` has real KL divergence in `klPolarization`; proxy fallback preserved |
| RK-001 | RivalKernel.scala | `KernelVariant.Design` has concrete `KernelConstructor.buildDesignKernel` |

Action: mark `resolved: true` in manifest.

## Phase 1: Public State Plumbing

**Resolves:** KC-001, KC-002, EI-001, SE-009, SE-010, BM-002

**Problem:** Kernel pipeline operates on fabricated dummy data — sentinel hero, empty board, no rivals, no action history.

### 1.1 Delete sentinel infrastructure from KernelConstructor

- Delete `KernelConstructor.buildActionKernel` (the sentinel version that fabricates a dummy PublicState).
- Delete `KernelConstructor.sentinelHero` and `KernelConstructor.dummyPublicState`.
- All callers migrate to `buildActionKernelFull` which threads real `PublicState` through.
- `buildDesignKernel` also gets a `Full` variant that accepts PublicState.

### 1.2 Refactor ExploitationInterpolation.buildInterpolatedKernel

- Current: fabricates sentinel hero + dummy PublicState internally (EI-001).
- New: returns `ActionKernelFull[M]` that accepts `PublicState` parameter.
- Caller (`StrategicEngine.buildKernelProfile`) passes the real `PublicState` through.

### 1.3 Add rival tracking to StrategicEngine

```scala
final case class RivalSeatInfo(position: Position, stack: Double)

// initSession signature change:
def initSession(
    rivalIds: Vector[PlayerId],
    rivalSeats: Map[PlayerId, RivalSeatInfo],
    existingBeliefs: Map[PlayerId, StrategicRivalBelief] = Map.empty
): Unit
```

- `bridgePublicState` builds real `TableMap[Chips]` with hero seat + all rival seats from `rivalSeats`.
- Rival stacks are updated when observed (e.g., after raises reduce effective stack).

### 1.4 Track action history per-hand

- Add `private var _actionHistory: Vector[PublicAction] = Vector.empty` to `StrategicEngine`.
- `observeAction` appends `PublicAction(actor, bridgeActionSignal(...))` to history.
- `bridgePublicState` passes `_actionHistory` as `actionHistory`.
- `startHand` clears the buffer.

### 1.5 BridgeManifest update

- `BM-002` (TableMap = Absent): upgrade to `Fidelity.Approximate` with note "rival seats from initSession; stack tracking approximate".

## Phase 2: Showdown Channel

**Resolves:** SE-003, SE-004, SE-005

### 2.1 Implement real ShowdownKernel

```scala
final class StrategicShowdownKernel(
    classifyFn: (Vector[Card], Board, Street) => StrategicClass,
    hardShiftWeight: Double = 0.90
) extends ShowdownKernel[StrategicRivalBelief]:
  def apply(state: StrategicRivalBelief, showdown: ShowdownSignal): StrategicRivalBelief =
    // For each revealed hand, classify into StrategicClass
    // Hard-shift posterior: 90% observed class + 10% prior smoothing
```

Classification logic: given revealed cards + board + final action:
- If hand beats top-pair equivalent and action was aggressive: **Value**
- If hand is weak (below middle pair) and action was aggressive: **Bluff**
- If hand has draw potential (flush/straight draws) and action was aggressive: **SemiBluff**
- Otherwise: **Marginal**

### 2.2 Wire endHand showdown processing

```scala
def endHand(showdownResult: Option[Map[PlayerId, HoleCards]] = None): Unit =
  showdownResult.foreach { revealed =>
    val board = /* last seen board from bridgePublicState */
    revealed.foreach { (rivalId, cards) =>
      if session.rivalBeliefs.contains(rivalId) then
        val signal = ShowdownSignal(Vector(RevealedHand(rivalId, cards.toVector)))
        val updated = showdownKernel.apply(session.rivalBeliefs(rivalId), signal)
        // update session beliefs
    }
  }
  _handActive = false
  _heroCards = None
```

### 2.3 SE-005: showdown = None in observeAction

This is semantically correct — showdowns occur at hand end. Remove the REDUCTIONISM comment. The signal path is: `observeAction` -> action signals, `endHand` -> showdown signals.

## Phase 3: Wire Orphaned Safety Objects

**Resolves:** OR-001, OR-003, OR-004, OR-007

### 3.1 Exploitability as real exploitabilityFn (OR-003)

Add to `StrategicEngine`:

```scala
private def computeExploitability(beta: Double): Double =
  // Build a RivalProfileClass from current session beliefs
  val profileClass = RivalProfileClass(
    sessionState.rivalBeliefs.values.map(_.asInstanceOf[JointRivalProfile]).toIndexedSeq
  )
  // Compute hero value function under current beta by running a POMCP solve
  // with the given rival profile and interpolation parameter
  val heroValue: JointRivalProfile => Ev = profile =>
    // Build search input with the profile's belief distribution at given beta
    // Solve via WPomcpRuntime.solveV2, extract expected value from Q-values
    val input = PokerPomcpFormulation.buildSearchInputV2(...)
    WPomcpRuntime.solveV2(input, solverConfig) match
      case Right(result) => Ev(result.qValues.max)
      case Left(_) => Ev.Zero
  // SecurityValue -> PointwiseExploitability -> DeploymentExploitability
  val securityOpt = SecurityValue.compute(heroValue, profileClass)
  val securityActual = SecurityValue.compute(heroValue, profileClass)
  PointwiseExploitability.compute(securityOpt, securityActual).value
```

The real heroValue function must delegate to the POMCP solver to evaluate Q-values under different belief configurations.

### 3.2 SafetyBellman into exploitation clamp (OR-001)

In `StrategicEngine.observeAction`, after computing `exploitabilityFn`:
- If robust losses are available from a previous decision, compute `SafetyBellman.computeBStar` and `requiredAdaptationBudget`.
- Use `ExploitationInterpolation.clampForCertificate` when the certificate is available.
- Fall back to scalar `clampForSafety` otherwise.

### 3.3 WassersteinRobustOracle + DroRuntime (OR-004, OR-007)

Add `useRobustQValues: Boolean = false` and `ambiguityRadius: Double = 0.1` to `StrategicEngine.Config`.

When enabled, `PokerPomcpFormulation.buildSearchInputV2` wraps the rival belief particles through `WassersteinDroRuntime.robustQValue` to produce worst-case Q-values within the ambiguity set.

## Phase 4: Activate Safety Apparatus

**Resolves:** SE-001, SE-002

### 4.1 Real exploitabilityFn (SE-001)

Replace:
```scala
exploitabilityFn = _ => 0.0
```
With:
```scala
exploitabilityFn = beta => computeExploitability(beta)
```

### 4.2 Real detection predicate (SE-002)

Add to `StrategicEngine.Config`:
```scala
detector: DetectionPredicate = FrequencyAnomalyDetection(window = 20, threshold = 0.6)
```

Replace `NeverDetect` in `observeAction` with `config.detector`.

## Phase 5: Remaining Proxies

**Resolves:** SRB-001, BF-001, BM-003, BM-004, SE-008

### 5.1 SRB-001: StrategicRivalBelief.update()

The identity return is correct by design — real update happens through `StateEmbeddingUpdater`. Add `@deprecated` annotation. Mark resolved.

### 5.2 BF-001: Belief-conditioned feasibility

```scala
def feasibleActions(
    legalActions: Vector[PokerAction],
    belief: StrategicRivalBelief,
    qRefLookup: PokerAction => Option[Ev],
    dominanceThreshold: Ev = Ev(-0.5)
): Vector[PokerAction] =
  val candidates = legalActions.filter { action =>
    qRefLookup(action).forall(_ >= dominanceThreshold)
  }
  if candidates.isEmpty then legalActions // never filter to empty
  else candidates
```

Backward-compatible: the zero-argument version stays as identity pass-through.

### 5.3 BM-003: ClassPosterior from kernel pipeline

Update `OpponentModelBridge` to read `StrategicRivalBelief.typePosterior` when the engine is in Strategic mode, instead of computing from VPIP/PFR/AF heuristics.

### 5.4 BM-004: FourWorld V10/V01 via PftDpw

Wire `PftDpwRuntime.solve` under `ChainWorld(Attrib, Off)` for V^{1,0} and `ChainWorld(Blind, On)` for V^{0,1}. These require building a `TabularGenerativeModel` from the current game state and belief configuration. Results feed into `FourWorldDecomposition.compute`.

### 5.5 SE-008: Uniform fallback from belief type

Replace `Array.fill(classes.length)(0.25)` with:
```scala
case _ => classes.map(c => StrategicRivalBelief.uniform.typePosterior.probabilityOf(c))
```

Semantically equivalent but reads from the belief system instead of hardcoded values.

## Phase 6: Wire Remaining Orphans

**Resolves:** OR-002, OR-005, OR-006

### 6.1 RiskDecomposition diagnostics (OR-002)

After each `decide`, compute `RiskDecomposition.computeProfile` using the canonical chain and current Q-values. Store in `SessionState` as `lastRiskProfile: Option[ChainRiskProfile]`.

### 6.2 FourWorldDecomposition in decision reporting (OR-005)

Expose `lastDecisionDiagnostics: Option[DecisionDiagnostics]` on `StrategicEngine`:
```scala
final case class DecisionDiagnostics(
    fourWorld: FourWorld,
    deltaVocabulary: DeltaVocabulary,
    riskProfile: ChainRiskProfile
)
```

### 6.3 PftDpwRuntime as alternative solver (OR-006)

```scala
enum SolverBackend:
  case WPomcp, PftDpw

// In Config:
solverBackend: SolverBackend = SolverBackend.WPomcp
```

When `PftDpw`, build `TabularGenerativeModel` from `PokerPomcpFormulation` arrays and solve via `PftDpwRuntime.solve`.

## Phase 7: Calibrated Heuristics

**Resolves:** SE-006, SE-007, PF-001, PF-002, PF-003, GE-001, HS-001, PFe-001

### 7.1 PFe-001: Preflop equity from equity table

In `PokerFeatures.handStrengthProxy`:
```scala
if board.size < 3 then
  HeadsUpEquityCanonicalTable.preflopEquity(hand)
    .getOrElse(HandStrengthEstimator.preflopStrength(hand))
```

### 7.2 PF-001: Actual call amount

In `PokerPomcpFormulation.buildActionEffects`, replace:
```scala
case PokerAction.Call =>
  val frac = if potChips > 0.0 then 0.5 else 0.0
```
With:
```scala
case PokerAction.Call =>
  val frac = if potChips > 0.0 then gameState.toCall / potChips else 0.0
```

Requires threading `gameState.toCall` into `buildActionEffects`.

### 7.3 PF-002: Bucket equity from table

Replace `buildLinearShowdownEquity` default with `HeadsUpEquityTable` bucket lookup:
```scala
def buildCalibratedShowdownEquity(
    numHeroBuckets: Int,
    numRivalBuckets: Int,
    table: HeadsUpEquityCanonicalTable
): Array[Double]
```

Falls back to linear heuristic when table unavailable.

### 7.4 PF-003 + SE-006: Action priors from CFR calibration

Add a calibration entry point:
```scala
object ActionPriorCalibration:
  def calibrateFromCfrArtifacts(
      artifactDir: java.io.File
  ): Map[(StrategicClass, PokerAction.Category), Double]
```

Reads stored CFR solve results, classifies hands into StrategicClass, measures empirical P(action | class). Falls back to `defaultActionPriors` / `defaultClassPriors` when no artifacts available.

### 7.5 SE-007: Bucket from hand strength

Replace `5` (neutral middle bucket) with:
```scala
case None =>
  // No card info — use position-based heuristic
  config.defaultHeroBucket // configurable, default 5
```

When cards are available (the normal case), the existing `estimateHeroBucket` already works correctly.

### 7.6 GE-001: GTO thresholds from CFR comparison

Add calibration tool:
```scala
object GtoThresholdCalibration:
  def calibrate(
      sampleSpots: IndexedSeq[(GameState, HoleCards)],
      cfrConfig: HoldemCfrConfig
  ): GtoThresholds
```

Runs fast-GTO vs exact-GTO on sample spots, optimizes thresholds to minimize action disagreement. Results stored in `GtoSolveEngine.defaultThresholds`.

### 7.7 HS-001: Blend weights from equity regression

Add calibration:
```scala
object BlendWeightCalibration:
  def calibrate(
      table: HeadsUpEquityCanonicalTable,
      sampleHands: Int = 10000
  ): Map[Street, Double]
```

Measures correlation between blended score and true equity across streets. Optimal weights replace hardcoded 0.50/0.56/0.62.

## File Change Summary

| File | Phase | Changes |
|---|---|---|
| `KernelConstructor.scala` | 1 | Delete sentinel, delete `buildActionKernel`, add `buildDesignKernelFull` |
| `ExploitationInterpolation.scala` | 1 | `buildInterpolatedKernel` → `ActionKernelFull`, delete sentinel |
| `StrategicEngine.scala` | 1,2,3,4 | Rival tracking, action history, showdown processing, safety wiring |
| `AugmentedState.scala` | 1 | (no change, types already support this) |
| `ReductionismManifestTest.scala` | all | Mark resolved entries progressively |
| `BluffFramework.scala` | 5 | Add belief-conditioned `feasibleActions` overload |
| `StrategicRivalBelief.scala` | 5 | `@deprecated` on `update()` |
| `PokerPomcpFormulation.scala` | 3,7 | Robust Q-values, actual call fraction, calibrated equity |
| `PokerFeatures.scala` | 7 | Preflop equity from table |
| `GtoSolveEngine.scala` | 7 | Calibration tool for thresholds |
| `HandStrengthEstimator.scala` | 7 | Calibration tool for blend weights |
| `bridge/BridgeManifest.scala` | 1,5 | Update fidelity entries |
| `bridge/OpponentModelBridge.scala` | 5 | Read from kernel pipeline beliefs |
| `bridge/ValueBridge.scala` | 5,6 | FourWorld V10/V01 from PftDpw |
| `DetectionPredicate.scala` | 4 | (no change — `FrequencyAnomalyDetection` already exists) |
| `SafetyBellman.scala` | 3 | (no change — already implemented) |
| `Exploitability.scala` | 3 | (no change — already implemented) |
| `RiskDecomposition.scala` | 6 | (no change — already implemented) |
| `FourWorldDecomposition.scala` | 6 | (no change — already implemented) |
| `solver/PftDpwRuntime.scala` | 6 | (no change — already implemented) |
| `solver/WassersteinDroRuntime.scala` | 3 | (no change — already implemented) |
| NEW: `ActionPriorCalibration.scala` | 7 | Calibration from CFR artifacts |
| NEW: `GtoThresholdCalibration.scala` | 7 | Threshold calibration tool |
| NEW: `BlendWeightCalibration.scala` | 7 | Blend weight calibration |

## Testing Strategy

Each phase adds tests to verify:
- Phase 1: `PublicState` constructed with real rival seats + action history
- Phase 2: Showdown updates shift posterior toward observed class
- Phase 3: `exploitabilityFn` returns non-trivial values; SafetyBellman certificate validated
- Phase 4: Detection fires when anomaly threshold exceeded; beta retreats
- Phase 5: Feasibility filtering removes dominated actions; ClassPosterior reads from beliefs
- Phase 6: DecisionDiagnostics populated; PftDpw solver returns valid results
- Phase 7: Preflop equity != 0.5; call fraction matches actual; calibrated values differ from defaults

Existing tests must continue to pass — especially the 6 manifest tests and all strategic module tests.
