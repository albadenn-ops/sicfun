# Runtime-Spec Closure: Formal Safety Pipeline Integration

**Date**: 2026-04-07
**Branch**: `feat/adaptive-proof-harness-9max`
**Status**: Design — awaiting review

## Problem Statement

The v0.31.1 formal objects (SecurityValue, SafetyBellman, RiskDecomposition,
Exploitability, WassersteinDroRuntime, PftDpwRuntime, FourWorldDecomposition)
exist in the codebase and have unit tests, but are disconnected from the
runtime decision path. The engine uses heuristic approximations and optional
dead switches where the spec requires formal safety enforcement.

Six audit findings (ranked Critical through Medium) document the gap.
This design closes them through a two-layer certification architecture
that is honest about what each solver backend can and cannot prove.

## Architectural Decision: Two-Layer Certification

The current solver (WPomcp) operates at root belief only. The spec's safety
Bellman (Defs 58-66) requires a finite certification state space with
per-state losses, transitions, and successor structure. These are
fundamentally different evaluation regimes.

The design splits certification into two layers:

| Layer | Solver | Certification | Spec Claims | Action Filtering |
|-------|--------|---------------|-------------|------------------|
| **Approximate** | WPomcp | LocalRobustScreening | Root-local budget bound. Advisory. Theorem 8 beta clamping. NOT Defs 61-66. | No. Beta clamping only. |
| **Formal** | PftDpw | TabularCertification | Defs 58-66, Corollary 9.3. Full multi-state B\*, safe action set, structural certificate. | Yes. Def 62/63 gate decide(). |

No code in the approximate layer may reference Def 61, 62, 63, 65, or 66
in comments, names, or documentation. Those definitions apply only to
the tabular certification path.

### Prerequisite: SafetyBellman Operator Correction

The current `SafetyBellman.tSafe` (SafetyBellman.scala:60-81) uses
`max_a` to compute the updated bound per state. Def 60 in the spec
specifies `inf_u` — the operator should find the action that minimizes
the required degradation budget, not the one that maximizes it.

Current (wrong):
```
(T_safe B)(s) = max_a [ L_robust(s, a) + gamma * max_{s'} B(s') ]
```

Correct (Def 60):
```
(T_safe B)(s) = inf_u [ L_robust(s, u) + gamma * sup_{sigma} E[B(s') | s, u, sigma] ]
```

With deterministic transitions in the tabular model, `sup E[B(s')]`
collapses to `max_{s'} B(s')` (already correct). But the outer operator
must be `min_a`, not `max_a`. This affects `tSafe`, `computeBStar`,
`safeActionSet`, and all `ForWorld` wrappers.

This correction is a hard prerequisite for the formal path. Until it
lands, no code may claim Defs 60-66. The fix is step 0 in the
implementation sequence.

## Section 1: DecisionEvaluationBundle

New case class — the single authoritative runtime artifact for all
formal safety computations. Computed in `decide()`, cached for advisory
reuse, flows into `DecisionDiagnostics` and `StrategicSnapshot`.

```scala
final case class DecisionEvaluationBundle(
    // Per-profile solver results — joint rival profiles (not class labels)
    profileResults: Map[JointRivalProfileId, SolverResult],
    // Per-action robust lower bounds from profile-conditioned re-solves
    // NOT V^sec(b; pi). These are min-over-profiles of root Q-values.
    robustActionLowerBounds: Array[Double],
    // Baseline evaluation: pi-bar = induced policy under reference-solver
    // regime (beta=0). epsilonBase is an offline/configured bound for
    // the exploitability of this induced policy, not a runtime computation.
    baselineActionValues: Array[Double],
    baselineValue: Double,
    // Approximate path: root-local adversarial gap (NOT Def 52C)
    adversarialRootGap: Option[Ev],
    // Formal path: pointwise exploitability (Def 52C)
    pointwiseExploitability: Option[Ev],
    // Deployment exploitability over empirical B_dep (Def 52D)
    // Only meaningful when deploymentSet has entries.
    deploymentExploitability: Option[Ev],
    // Certification result — determines which layer produced this bundle
    certification: CertificationResult,
    // Chain-world values (kernel dimension only; NOT grid-world)
    chainWorldValues: Map[ChainWorld, Ev],
    // Provenance and approximation labels
    notes: Vector[String]
)
```

### CertificationResult

```scala
enum CertificationResult:
  case LocalRobustScreening(
      rootLosses: Array[Double],
      budgetEstimate: Double,         // max(rootLosses) / (1 - gamma)
      withinTolerance: Boolean        // budgetEstimate <= epsilon_adapt
  )
  case TabularCertification(
      bStar: Array[Double],           // Def 61: B* per state
      requiredBudget: Double,         // Def 64: sup B*
      safeActionIndices: IndexedSeq[Int], // Def 62: U*_safe at root
      certificateValid: Boolean,      // Def 66
      withinTolerance: Boolean        // Def 64: epsilon*_adapt <= epsilon_adapt
  )
  case Unavailable(reason: String)
```

### DecisionOutcome

```scala
enum DecisionOutcome:
  case Certified(action: PokerAction, bundle: DecisionEvaluationBundle)
  case BaselineFallback(action: PokerAction, reason: String)
```

No `Uncertified` variant. If not certified, baseline fallback. A
research-only permissive mode may be added later behind a config flag
with an explicit "RESEARCH_ONLY" label.

## Section 2: Profile-Conditional Evaluation

### Problem

`SecurityValue.compute` (Exploitability.scala) requires
`heroValue: JointRivalProfile => Ev`. WPomcpRuntime returns one
`actionValues` vector under one embedded `rivalPolicy` table. You cannot
recover `inf_{sigma^{-S}}` by post-hoc weighting.

### Solution

Add `PokerPomcpFormulation.buildSearchInputForProfile`:

```scala
def buildSearchInputForProfile(
    gameState: GameState,
    rivalBeliefs: Map[PlayerId, StrategicRivalBelief],
    heroActions: Vector[PokerAction],
    heroBucket: Int,
    particlesPerRival: Int,
    profileId: JointRivalProfileId
): SearchInputV2
```

This rebuilds the `FactoredModel` with `rivalPolicy` set to the joint
profile's action distribution. Same particles, same model structure,
different policy table.

### Profile Family

A `JointRivalProfile` specifies a complete joint rival behavior — all
rivals simultaneously. For tractability, the profile class consists of
4 pure-type profiles (all rivals assigned the same StrategicClass).

Each profile gets its own type (`JointRivalProfileId`) distinct from
`StrategicClass`. `Map[StrategicClass, Array[Double]]` is not used.

### Solve Convention

All sections use this convention:

| Solve | Purpose | rivalPolicy |
|-------|---------|-------------|
| Mixed | Action selection (standard) | Current posterior mixture |
| Baseline | pi-bar evaluation | Reference-regime (beta=0) |
| Pure-Value | Profile class member | All rivals = Value |
| Pure-Bluff | Profile class member | All rivals = Bluff |
| Pure-SemiBluff | Profile class member | All rivals = SemiBluff |
| Pure-Marginal | Profile class member | All rivals = Marginal |

Total: 6 WPomcp solves per decision. The mixed solve is NOT in the
profile class — it is the standard action-selection solve. The 4
pure-type solves constitute the profile class for SecurityValue.

### Cost

6 WPomcp solves at 500 simulations each. On native runtime, this is
~6x current cost. Acceptable for correctness; configurable via
`numProfileSolves` if profiling shows issues.

## Section 3: WPomcp Approximate Certification (LocalRobustScreening)

### Quantities

All names avoid Def 61-66 language.

- `robustActionLowerBounds[a]` = min over profiles of
  `profileResult.actionValues[a]`
- `adversarialRootGap` = `baselineValue - min_profile(max_a profileQ[a])`.
  This is NOT Def 52C (PointwiseExploitability). It is a root-local
  adversarial gap computed from profile-conditioned re-solves.
- `rootLosses[a]` = `baselineValue - robustActionLowerBounds[a]`
  (Def 59 instantiated at root belief only)
- `budgetEstimate` = `max(rootLosses) / (1 - gamma)`. Root-local bound.
  With 1 state, SafetyBellman.safeActionSet admits all actions
  (every L[a] <= max L), so no action filtering is performed.
- `withinTolerance` = `budgetEstimate <= epsilonAdapt`

### Decision Flow

```
1. Solve with mixed-belief rivalPolicy -> mixedResult (action selection)
2. Solve with reference rivalPolicy (beta=0) -> baselineResult (pi-bar)
3. For each of 4 pure-type profiles in profile class:
     Solve with profile's rivalPolicy -> profileResult
4. Compute robustActionLowerBounds, adversarialRootGap, rootLosses
   (6 solves total: 1 mixed + 1 baseline + 4 pure-type)
5. budgetEstimate = max(rootLosses) / (1 - gamma)
6. If withinTolerance:
     action = mixedResult.bestAction
   Else:
     Clamp beta via betaBar (Theorem 8 / AdaptationSafety)
     If beta clamped to 0: action from baselineResult
     Else: re-interpolate and pick from mixedResult
7. Bundle with LocalRobustScreening
```

### What This Path Does NOT Do

- Does not claim Def 61 (B\* is a root-local estimate, not a fixed point
  over a certification state space)
- Does not filter actions via Def 62/63
- Does not produce a structural certificate (Def 65/66)
- Does not compute PointwiseExploitability (Def 52C) — uses
  adversarialRootGap instead

## Section 4: PftDpw Formal Certification (TabularCertification)

### Prerequisites

1. **PokerPftFormulation** (new): builds `TabularGenerativeModel` and
   `ParticleBelief` from engine state. Missing piece; `PftDpwRuntime`
   and `TabularGenerativeModel` already exist.
2. **Per-state loss evaluator**: derives `robustLosses[s][a]` from the
   tabular model's reward and transition structure.
3. **composeFullKernelForWorldFull**: world-aware kernel dispatcher
   that threads `PublicState` (see Section 8).

### Per-State Loss Derivation

A joint rival profile sigma^{-S} can change reward, transition, and
observation structure — not just rewards. For each profile, a distinct
`TabularGenerativeModel` is constructed:
- `R_sigma(s, a)` — reward under profile sigma
- `T_sigma(s, a)` — transition under profile sigma
- `O_sigma(o | s', a)` — observation likelihood under profile sigma

Robust one-step loss at state s, action a (Def 59 over the tabular model):

```
L_robust(s, a) = sup_sigma max(0,
    V^{barpi}_sigma(s) - R_sigma(s, a) - gamma * V^{barpi}_sigma(T_sigma(s, a)))
```

where `V^{barpi}_sigma(s)` is the baseline value at state s under pi-bar
evaluated against profile sigma. The sup is over the finite profile class.

For each profile sigma in the class:
1. Build `TabularGenerativeModel` with sigma's policy parameters
2. Evaluate `V^{barpi}_sigma` by value iteration under the reference
   policy on sigma's model
3. Compute per-state per-action losses from sigma's (R, T, V) triple

Then `robustLosses[s][a] = max over profiles of per-profile losses`.

This uses the tabular model's structure directly — profile-conditioned
model construction, not solver re-runs at every state.

### Decision Flow

```
1. PokerPftFormulation.buildTabularModel(gameState, ...) -> model
2. PokerPftFormulation.buildParticleBelief(rivalBeliefs, ...) -> belief
3. PftDpwRuntime.solve(model, belief, config) -> PftDpwResult
4. For each profile sigma in profile class:
     a. Build TabularGenerativeModel with sigma's policy parameters
     b. V^{barpi}_sigma = value iteration of reference policy on sigma's model
5. robustLosses[s][a] = max over profiles sigma of
     max(0, V^{barpi}_sigma(s) - R_sigma(s,a) - gamma * V^{barpi}_sigma(T_sigma(s,a)))
6. SafetyBellman.computeBStar(robustLosses, bellmanGamma) -> bStar
   (uses corrected min_a operator from step 0)
7. Belief-level safe action set:
   B_belief(a) = sum_s belief(s) * [L_robust(s,a) + gamma * max_{s'} bStar(s')]
   safeActions = { a : B_belief(a) <= sum_s belief(s) * bStar(s) }
   This is Def 62 instantiated at the particle belief (Def 54), not a
   collapsed state index. The belief weights come from ParticleBelief.
8. action = SafetyBellman.safeFeasibleAction(qValues, safeActions)
9. Certificate validation (Def 65/66)
10. Bundle with TabularCertification
```

### PokerPftFormulation

```scala
object PokerPftFormulation:
  def buildTabularModel(
      gameState: GameState,
      rivalBeliefs: Map[PlayerId, StrategicRivalBelief],
      heroActions: Vector[PokerAction],
      heroBucket: Int,
      actionPriors: Map[(StrategicClass, PokerAction.Category), Double]
  ): TabularGenerativeModel

  def buildParticleBelief(
      rivalBeliefs: Map[PlayerId, StrategicRivalBelief],
      particlesPerRival: Int
  ): ParticleBelief
```

State space discretization is implementation-specific (spec confirms).
Uses existing `heroBucket` granularity from `HandStrengthEstimator`,
`Street` (4 values), pot-relative sizing discretization from `GameState`.

## Section 5: Operational Baseline (A10)

### Definition

pi-bar is the policy induced by solving under the reference regime at
every decision point — the engine's behavior when beta = 0 for all
rivals permanently. V_baseline in the formal path (Section 4) is the
evaluation of this extracted policy on the tabular model, computed by
value iteration, not by re-running the MCTS solver.

`epsilonBase` is an offline/configured bound for the exploitability of
this induced policy. It is not a runtime computation.

### Operational Objects

```scala
final case class OperationalBaseline(
    epsilonBase: Double,
    deploymentSet: EmpiricalDeploymentSet,
    description: String   // "CFR-derived", "configured-conservative", etc.
)
```

`epsilonBase` may be derived from offline CFR analysis or set
conservatively.

```scala
final case class EmpiricalDeploymentSet(
    entries: Vector[DeploymentBeliefSummary],
    maxSize: Int = 50
)

final case class DeploymentBeliefSummary(
    beliefEntropy: Double,
    exploitabilitySnapshot: Ev,
    timestamp: Long
)
```

`EmpiricalDeploymentSet` is an implementation approximation of B_dep,
not the literal spec object. The circular buffer stores evaluation
summaries (not raw beliefs) for cheap `DeploymentExploitability`
computation.

Config additions:
```scala
epsilonBase: Double = 0.05,
deploymentSetSize: Int = 50
```

## Section 6: observeAction() Handling

### Inter-Decision Exploitability

Formal exploitability is computed only at decision time (in `decide()`).
Between decisions, `observeAction()` uses the existing heuristic
(`computeExploitabilityEstimate`) as the `exploitabilityFn` for
`Dynamics.fullStep`. This is an acknowledged approximation.

Rationale: the cached bundle was computed at a different public state.
After `observeAction`, history and public state have changed. Using the
previous bundle as a parametric exploitability evaluator is not valid.
The heuristic serves as an inter-decision bridge.

### Bellman Clamp

The post-fullStep Bellman clamp uses the cached `budgetEstimate` from
the last bundle as an advisory bound. It is not formal B\*.

```scala
// Advisory clamp from last evaluation bundle
_lastBundle match
  case Some(bundle) =>
    val budget = bundle.certification match
      case LocalRobustScreening(_, est, _) => est
      case TabularCertification(_, req, _, _, _) => req
      case Unavailable(_) => Double.MaxValue
    // Clamp using advisory budget
    ...
  case None =>
    // No bundle yet — skip clamp
    ...
```

### Remove useBellmanSafety Boolean

The advisory clamp always runs when a bundle is available. No toggle.
`bellmanGamma` is used in bundle computation (no longer dead code).

## Section 7: Fail-Closed Semantics

| Condition | Outcome |
|-----------|---------|
| Native solver unavailable | `BaselineFallback`. Beta=0, first non-fold from reference solve (or last non-fold if no solve possible). |
| Profile class empty | Impossible (StrategicClass is a 4-value enum). |
| Certificate validation fails (formal path) | `BaselineFallback`. Safe action set was empty or B\* exceeded tolerance. |
| Budget exceeds tolerance (approximate path) | Beta clamped toward 0 via betaBar. If clamped to 0, baseline action. |
| Solver returns error | `BaselineFallback`. Log error. |

## Section 8: World-Aware Kernel Path

### New: composeFullKernelForWorldFull

```scala
def composeFullKernelForWorldFull[M](
    actionKernelFull: ActionKernelFull[M],
    designKernelFull: ActionKernelFull[M],
    showdownKernel: ShowdownKernel[M]
)(world: ChainWorld): FullKernel[M]
```

Same dispatch logic as existing `composeFullKernelForWorld` but uses
`ActionKernelFull` (threads `PublicState`). Required for production
world-aware evaluation.

### WorldIndexedKernelProfile

Use the existing `WorldIndexedKernelProfile` type from
`KernelConstructor.scala`. Do not create a new `IndexedKernelProfile`.

The formal (PftDpw) path builds a `WorldIndexedKernelProfile` with
kernels for each chain world via `composeFullKernelForWorldFull`. The
approximate (WPomcp) path continues with single `JointKernelProfile`.

## Section 9: Chain-World vs Grid-World Outputs

These are separate dimensions. `WorldTypes.scala` defines:
- `ChainWorld` = `LearningChannel x ShowdownMode` (8 worlds, 6 distinct)
- `GridWorld` = `LearningChannel x PolicyScope` (4 worlds)

`chainWorldValues: Map[ChainWorld, Ev]` in the bundle provides
per-kernel-configuration values. This is achievable by kernel swaps.

`gridWorldValues` in `StrategicSnapshot` requires policy-scope-constrained
evaluation (open-loop vs closed-loop policy classes). V^{1,0} and V^{0,1}
remain `BridgeResult.Absent` until policy-scope-constrained solver
evaluation exists. V^{1,1} and V^{0,0} are `Approximate` from engine EV
and static equity respectively.

The design does not conflate these. `chainWorldValues` is NOT passed to
`ValueBridge.toGridWorldValues`.

## Section 10: DecisionDiagnostics and StrategicSnapshot

### Expanded DecisionDiagnostics

```scala
final case class DecisionDiagnostics(
    heroBucket: Int,
    solverBackend: SolverBackend,
    exploitationBetas: Map[PlayerId, Double],
    outcome: DecisionOutcome,
    bundle: Option[DecisionEvaluationBundle],
    // Reporting
    adversarialRootGap: Option[Ev],         // approximate path
    safeActionCount: Option[Int],           // formal path only
    totalActionCount: Int,
    chainWorldValues: Map[ChainWorld, Ev]
)
```

### StrategicSnapshot Derivation

New builder:

```scala
object StrategicSnapshot:
  def fromDiagnostics(
      diag: DecisionDiagnostics,
      gameState: GameState,
      heroAction: PokerAction,
      heroEquity: Double,
      engineEv: Double,
      staticEquity: Double,
      hasDrawPotential: Boolean,
      opponentStats: Option[(Double, Double, Double)] = None
  ): StrategicSnapshot
```

This populates the v0.31.1 optional fields:
- `securityValue`: from `bundle.robustActionLowerBounds` (labeled as
  root-local lower bound, not V^sec)
- `safetyCertificateSummary`: `(requiredBudget, withinTolerance)` from
  certification result
- `chainRiskProfile`: from `RiskDecomposition` over chain-world values
  when available
- `gridWorldValues`: from `ValueBridge.toGridWorldValues` — V^{1,0} and
  V^{0,1} remain Absent
- `bridgeFidelityNotes`: collected from bundle notes

The old `StrategicSnapshot.build()` is preserved for backward compat
but deprecated.

## Section 11: AssumptionManifest v0.31.1 Alignment

Header: `v0.30.2` -> `v0.31.1`

| ID | Current (wrong) | Corrected (per spec) | Location |
|----|-----------------|---------------------|----------|
| A1' | Finite action space | Abstraction with guarantees (alpha maps, epsilon bounds, value error in MDP/POMDP) | PokerAction enum + HandStrengthEstimator (sizing quantization) |
| A5 | Conditional independence of signals given type | Bounded reward and discounting (\|r\| <= R_max, gamma in (0,1)) | Chips opaque type, solver configs |
| A7 | Bounded reward | Well-defined full rival update (Gamma^full kernel with omega^act, omega^sd) | RivalKernel.scala: ActionKernel, ShowdownKernel, FullKernel |
| A8 | Discount factor gamma in [0,1) | Strategically relevant repetition (p_lower > 0 future interaction) | Structural (poker session guarantees repeat play) |

Add A6 entry (first-order interactive sufficiency — m_t^{R,i} truncation)
alongside existing A6'.

Update enforcement types and location references to match corrected
semantics.

## Section 12: ReductionismManifest

### Principle

`resolved = true` means the described gap is closed in production code,
verified by a behavioral test. Not "acknowledged" or "injectable".

### Changes

- OR-001 through OR-007 (Orphan): `resolved = false` until behavioral
  tests confirm wiring. Flipped per-entry as each formal object is
  connected.
- SE-001 (exploitabilityFn): `resolved = false` until formal or
  approximate-labeled exploitability replaces the heuristic at decision
  time.
- All other entries: audit individually. If the gap described is still
  present in production, `resolved = false`.
- Test assertion: no entry with `resolved = true` whose corresponding
  code path is demonstrably still a stub, identity, or dead switch.

## Section 13: Behavioral Tests

| Test | What It Verifies |
|------|-----------------|
| **ProfileConditionalSolveTest** | 4 pure-type profile solves produce distinct Q-vectors. Baseline solve Q-values differ from mixed solve. 6 solves total. |
| **LocalRobustScreeningTest** | WPomcp path produces `LocalRobustScreening` with real root losses and budget estimate. Budget exceeding tolerance triggers beta clamp. |
| **TabularCertificationTest** | PftDpw path produces `TabularCertification` with multi-state B\*, safe action set. Actions outside U\*\_safe are excluded from selection. |
| **BaselineFallbackTest** | Solver error -> `BaselineFallback`. Empty safe action set -> `BaselineFallback`. |
| **AdversarialRootGapTest** | Verify adversarialRootGap > 0 when beliefs are concentrated. Verify it is NOT named PointwiseExploitability. |
| **DeploymentBaselineTest** | EmpiricalDeploymentSet accumulates summaries. DeploymentExploitability computed over buffer. |
| **CertificationScopeHonestyTest** | WPomcp bundle does NOT contain Def 61/62/63 references. PftDpw bundle does. |
| **DiagnosticsToSnapshotTest** | StrategicSnapshot.fromDiagnostics populates v0.31.1 fields correctly. |
| **AssumptionManifestAlignmentTest** | All 10+1 entries match v0.31.1 spec definitions. |
| **ReductionismTruthTest** | No `resolved = true` entry whose code path is still a stub. |

## Section 14: Config Changes

```scala
final case class Config(
    // Existing (unchanged)
    numSimulations: Int = 500,
    discount: Double = 0.95,
    maxDepth: Int = 20,
    seed: Long = 42L,
    particlesPerRival: Int = 100,
    solverBackend: SolverBackend = SolverBackend.WPomcp,
    exploitConfig: ExploitationConfig = ...,
    temperedConfig: TemperedLikelihood.TemperedConfig = ...,
    actionPriors: Map[...] = defaultActionPriors,
    detector: DetectionPredicate = ...,
    defaultHeroBucket: Int = 5,
    // Revised
    bellmanGamma: Double = 0.95,           // now used in certification
    ambiguityRadius: Double = 0.1,         // feeds DRO layer in formal path
    // New
    epsilonBase: Double = 0.05,            // A10: offline baseline bound
    deploymentSetSize: Int = 50,           // |B_dep| empirical buffer
    // Removed
    // useBellmanSafety: Boolean -- removed; certification always runs
    // useRobustQValues: Boolean -- removed; DRO feeds safety analysis
)
```

## Implementation Sequencing

0. **SafetyBellman operator correction** — fix `tSafe` from `max_a` to
   `min_a` per Def 60. Update `computeBStar`, `safeActionSet`, and all
   `ForWorld` wrappers. Add belief-level safe action evaluation method.
   Existing SafetyBellman unit tests updated to match corrected semantics.
1. **DecisionEvaluationBundle + CertificationResult types** — pure data,
   no behavioral change.
2. **Profile-conditional evaluation in PokerPomcpFormulation** — new
   `buildSearchInputForProfile`. Unit tested.
3. **OperationalBaseline + EmpiricalDeploymentSet** — types and buffer.
4. **WPomcp approximate path in decide()** — 6 solves, bundle
   construction, LocalRobustScreening, beta clamping. Behavioral tests.
5. **composeFullKernelForWorldFull** — world-aware production kernel.
   Hard prerequisite for steps 6-8.
6. **PokerPftFormulation** — tabular model builder. Unit tested.
7. **Per-state loss evaluator** — profile-conditioned model construction,
   value iteration, profile-robust losses from tabular models.
8. **PftDpw formal path in decide()** — TabularCertification,
   belief-level safe action filtering. Behavioral tests.
9. **Chain-world value evaluation** — populate chainWorldValues.
10. **DecisionDiagnostics expansion + StrategicSnapshot.fromDiagnostics**.
11. **observeAction() advisory clamp** — cached budget, remove
    useBellmanSafety.
12. **AssumptionManifest v0.31.1 alignment**.
13. **ReductionismManifest truth** — flip resolved per entry as wiring
    lands.
14. **Behavioral test suite**.

Steps 0-4 are the minimum viable integration (WPomcp approximate path).
Steps 5-8 enable the formal certification path.
Steps 9-14 are completion and cleanup.

### Note on FourWorldDecomposition (OR-005)

FourWorldDecomposition operates on grid-world values (V^{1,0}, V^{0,1},
etc.), which require policy-scope-constrained evaluation — a separate
dimension from chain-world values (Section 9). OR-005 cannot be flipped
to resolved until policy-scope-constrained solver evaluation exists.
Chain-world value population (step 9) does NOT resolve OR-005.
