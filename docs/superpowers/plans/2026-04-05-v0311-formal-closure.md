# SICFUN v0.31.1 Formal Closure -- Implementation Plan

> **For agentic workers:** REQUIRED LINKED-WORKER PASS: use `scripts/ai-minion.ps1` on a narrow file slice before and after each major wave. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** upgrade the current `sicfun.holdem.strategic` layer from its effective v0.30.2 / partial v0.31 surface to the `SICFUN-v0_31_1-corrected.md` model, including:
- split world algebra (`Omega^chain` vs `Omega^grid`)
- `(omega^act, omega^sd)`-indexed full kernels with explicit showdown gating
- formal exploitability (`Sigma^{-S}`, `J`, `V^sec`, pointwise and deployment exploitability)
- strengthened adaptation safety (AS-strong)
- Bellman-safe certificates (`T_safe`, `B*`, safe action sets, deployable `B_beta`)
- world-aware risk decomposition
- bridge, manifest, comments, and tests updated to stop overclaiming old-spec alignment

**Architecture:** keep `sicfun.holdem.strategic` pure Scala with no engine imports; use existing solver runtimes as oracles where possible; migrate by addition first, removal second.

**Tech Stack:** Scala 3.8.1, munit 1.2.2, existing `PftDpwRuntime`, `WPomcpRuntime`, and `WassersteinDroRuntime`.

**Canonical spec:** `SICFUN-v0_31_1-corrected.md`

**Depends on:** completed v0.30.2 formal layer under `src/main/scala/sicfun/holdem/strategic`

**Unlocks:** a spec-aligned formal layer, honest bridge fidelity, solver-backed safety diagnostics, and a proof harness that can reason about v0.31.1 objects directly.

---

## Planning Decisions

1. **Do not perform a flag-day rewrite.**
   Add v0.31.1 types and operators first, keep legacy wrappers temporarily, then delete or deprecate old surfaces only after all call sites and tests migrate.

2. **Stop fabricating unsupported formal objects.**
   Where the engine cannot yet produce a v0.31.1 quantity exactly, prefer `BridgeResult.Absent` or a typed approximate diagnostic over interpolated placeholders that look canonical.

3. **Use finite representative classes for tractable safety/exploitability.**
   `Sigma^{-S}` and deployment belief sets must be represented by finite or parametric families in code. Exact universal quantification remains a theorem-level statement; implementation uses conservative finite approximations with explicit fidelity markers.

4. **Keep new safety logic solver-backed but Scala-owned.**
   Do not add new JNI first. Reuse existing strategic solver wrappers as Q / V / robust-value oracles and only add native work if profiling proves the Scala orchestration is the bottleneck.

5. **Treat comment and manifest correctness as part of closure.**
   Overclaiming spec alignment is itself a gap. Comments, manifests, and theorem labels must be updated in the same wave as the code they describe.

6. **Every assumption family and every formal result gets an explicit status.**
   The closure is incomplete until each assumption family `(A1′)-(A10)`, each new proposition / theorem / corollary, and each non-normative computational recommendation is marked as:
   - encoded in code
   - approximated with explicit fidelity
   - inherited without code change
   - deferred out of constitutive scope

---

## Gap-to-Wave Map

| Gap | Closure wave |
|---|---|
| No `Omega^chain` / `Omega^grid` split | Wave 1 |
| No showdown-gated full kernel family | Wave 2 |
| No design kernel in constitutive full-kernel family | Wave 2 |
| No explicit assumption-to-code closure for A1′-A10 | Waves 0, 7 |
| No partial-policy typing for attributed baseline / action kernels | Wave 2 |
| No `Sigma^{-S}`, `J`, `V^sec`, exploitability objects | Wave 3 |
| No explicit `B_dep` / `bar pi` / `epsilon_base` operational baseline | Waves 3, 4 |
| No robust-Q / ambiguity-set integration into safety and exploitability | Waves 3, 4 |
| Old scalar adaptation safety only | Wave 4 |
| No `T_safe`, `B*`, safe action sets, `B_beta` | Wave 4 |
| No world-aware risk decomposition | Wave 5 |
| No plan to replace or explicitly qualify heuristic spot polarization | Waves 2, 7 |
| Bridge fabricates four-world values and lacks chain/risk objects | Wave 6 |
| No explicit theorem / proposition / corollary coverage ledger | Waves 0, 7 |
| No explicit Appendix B false-changepoint verification | Wave 7 |
| No explicit triage for computational architecture Defs 54-56 / §10B | Waves 0, 7 |
| Comments / manifests / tests still reference old defs | Waves 0, 6, 7 |

---

## Wave Overview

| Wave | Name | Main outcome |
|---|---|---|
| 0 | Spec Hygiene and Migration Fence | remove naming ambiguity and establish compatibility rules |
| 1 | World Algebra Types | add explicit chain/grid world types and keyed value containers |
| 2 | Kernel Closure | implement `(omega^act, omega^sd)` full-kernel behavior |
| 3 | Formal Exploitability | add `Sigma^{-S}`, `J`, security value, exploitability operators |
| 4 | Strong Safety and Certificates | implement AS-strong and Bellman-safe law |
| 5 | Risk Decomposition | add chain-indexed robust loss, risk increments, efficiency metric |
| 6 | Bridge and Snapshot Migration | migrate bridge layer to honest v0.31.1 objects |
| 7 | Validation, Cleanup, and Removal | theorem tests, proof harness, legacy shim removal |

---

## File Map

### New files

```
src/main/scala/sicfun/holdem/strategic/
|-- WorldTypes.scala
|-- Exploitability.scala
|-- SafetyBellman.scala
|-- RiskDecomposition.scala

src/test/scala/sicfun/holdem/strategic/
|-- WorldTypesTest.scala
|-- ExploitabilityTest.scala
|-- SafetyBellmanTest.scala
|-- RiskDecompositionTest.scala
```

### Files to update

```
src/main/scala/sicfun/holdem/strategic/
|-- StrategicValue.scala
|-- RivalKernel.scala
|-- KernelConstructor.scala
|-- Dynamics.scala
|-- AdaptationSafety.scala
|-- ExploitationInterpolation.scala
|-- AssumptionManifest.scala
|-- SpotPolarization.scala

src/main/scala/sicfun/holdem/strategic/bridge/
|-- ValueBridge.scala
|-- StrategicSnapshot.scala
|-- BridgeManifest.scala
|-- OpponentModelBridge.scala
|-- BaselineBridge.scala

src/test/scala/sicfun/holdem/strategic/
|-- KernelConstructorTest.scala
|-- RivalKernelLawTest.scala
|-- DynamicsTest.scala
|-- AdaptationSafetyTest.scala
|-- TheoremValidationTest.scala

src/test/scala/sicfun/holdem/strategic/bridge/
|-- BridgeTest.scala
|-- StrategicSnapshotTest.scala
```

### Optional later touchpoints

```
src/main/scala/sicfun/holdem/validation/
|-- AdaptiveProofHarness.scala
|-- ValidationScorecard.scala

src/test/scala/sicfun/holdem/validation/
|-- AdaptiveProofHarnessTest.scala
```

---

## Wave 0 -- Spec Hygiene and Migration Fence

**Goal:** create a stable migration boundary before changing behavior.

- [ ] Add a short symbol-mapping note at the top of the new implementation wave docs:
  - `delta_adapt -> epsilon_adapt`
  - `delta_retreat -> delta_cp_retreat`
  - `omega` must always be qualified as chain or grid
- [ ] Add an assumption-status ledger covering `(A1′)-(A10)`:
  - encoded in code
  - approximated with explicit fidelity
  - inherited without implementation work
  - deferred out of constitutive scope
- [ ] Add a formal-result ledger covering:
  - Proposition 8.1
  - Proposition 9.1, 9.2, 9.5, 9.6, 9.7
  - Theorems 1-9
  - Corollaries 1-4 and 9.3
- [ ] Decide compatibility policy:
  - keep legacy constructors and accessors during Waves 1-6
  - mark old names as deprecated once replacements exist
  - remove only in Wave 7
- [ ] Fix obvious overclaims in comments that block safe refactoring:
  - `AdaptationSafety.scala` def-number references
  - `SpotPolarization.scala` "canonical KL" language if still heuristic
  - `BridgeManifest.scala` outdated Def 52-53 references
- [ ] Triage §10 computational architecture items explicitly:
  - Defs 54-56 and §10B marked as inherited, no-op, or follow-up
  - if inherited, identify the concrete files that already satisfy them
  - if not inherited, create a deferred follow-up note rather than leaving them implicit
- [ ] Add a migration checklist comment to `StrategicValue.scala`, `RivalKernel.scala`, and `ValueBridge.scala`

**Exit criteria:**
- no file still claims current code already implements Defs 52A-69
- symbol rename policy is written down before the first behavior change
- assumption ledger and theorem/proposition ledger exist in the plan or supporting notes

---

## Wave 1 -- World Algebra Types

**Goal:** make the world split explicit in types before touching kernels or values.

### Tasks

- [ ] Create `WorldTypes.scala` with:
  - `enum LearningChannel { Blind, Ref, Attrib, Design }`
  - `enum ShowdownMode { Off, On }`
  - `enum PolicyScope { OpenLoop, ClosedLoop }`
  - `final case class ChainWorld(channel: LearningChannel, showdown: ShowdownMode)`
  - `final case class GridWorld(learning: LearningChannel, scope: PolicyScope)`
- [ ] Encode legal grid invariants:
  - only `Blind` and `Attrib` are legal in `GridWorld`
  - `GridWorld` constructor rejects `Ref` and `Design`
- [ ] Extend `StrategicValue.scala` with keyed containers instead of only positional fields:
  - keep `FourWorld(v11, v10, v01, v00)` temporarily
  - add a keyed representation or accessors using `GridWorld`
  - add chain-edge and chain-risk delta value types
- [ ] Add helper enumerators for:
  - all 8 chain worlds
  - canonical chain `(blind,off) -> (ref,off) -> (attrib,off) -> (attrib,on)`
  - all 4 grid worlds
- [ ] Update theorem/test helpers to use world constructors instead of raw booleans or tuple position assumptions

### Tests

- [ ] `WorldTypesTest.scala`
  - `ChainWorld.all.size == 8`
  - canonical chain has 4 elements, each in `ChainWorld.all`
  - `GridWorld.all.size == 4`
  - illegal grid worlds are rejected
- [ ] `StrategicValueTest.scala`
  - keyed accessors round-trip existing `FourWorld`
  - delta formulas remain unchanged under keyed representation

**Exit criteria:**
- raw `v11/v10/v01/v00` still work for compatibility
- all new code can speak in `ChainWorld` / `GridWorld` rather than unnamed coordinates

---

## Wave 2 -- Kernel Closure

**Goal:** replace implicit kernel variants with explicit chain-world semantics.

### Tasks

- [ ] Refactor `RivalKernel.scala`
  - keep `ActionKernel`, `ShowdownKernel`, `FullKernel`
  - replace or de-emphasize `KernelVariant` in favor of `LearningChannel`
  - document `FullKernel` as operating under a `ChainWorld`
- [ ] Update `Baseline.scala` and kernel-facing types for Definition 10 / Definition 18:
  - represent the attributed baseline as a state-conditioned object rather than only a prose concept
  - expose the partial-policy view `hat_pi_{x,m}` explicitly
  - ensure `Gamma^{act,attrib}` is built from that typed partial policy, not an unstructured closure
- [ ] Update `KernelConstructor.scala`
  - add explicit showdown gating to full-kernel composition
  - support `(channel, showdown)` pairs for `Blind`, `Ref`, `Attrib`, `Design`
  - ensure `(Blind, Off)` and `(Blind, On)` collapse to the same identity behavior
  - compose a real `Design` full kernel rather than a standalone action kernel only
- [ ] Replace or qualify the current spot-polarization implementation:
  - either implement a posterior-divergence-based path consistent with Def 25 / A9
  - or split the current heuristic into an explicitly approximate adapter with fidelity annotations
- [ ] Update `JointKernelProfile`
  - store or derive chain-world-specific full kernels
  - support lookup by rival id and `ChainWorld`
- [ ] Update `Dynamics.scala`
  - `fullRivalUpdate` must take a concrete chain world or a world-indexed profile
  - `counterfactualReferenceWorld` must explicitly use `(Ref, Off)` or `(Ref, On)` rather than an unqualified reference profile
- [ ] Audit all tests and bridge code that assumed showdown always fires when present

### Tests

- [ ] Extend `KernelConstructorTest.scala`
  - showdown is skipped under `ShowdownMode.Off`
  - showdown is applied under `ShowdownMode.On`
  - design channel strips sizing / timing and still composes with showdown when `On`
  - attributed action kernel is built from the typed partial-policy view rather than an unqualified policy function
- [ ] Extend `RivalKernelLawTest.scala`
  - blind worlds are identities
  - `Blind/On` and `Blind/Off` are behaviorally equivalent
  - canonical chain edge semantics match the spec
- [ ] Extend `DynamicsTest.scala`
  - same signal under `(Attrib,Off)` and `(Attrib,On)` yields different results only via showdown
- [ ] Add or extend polarization tests
  - exact or approximate status is explicit
  - if heuristic path remains, tests assert it is not labeled canonical without qualification

**Exit criteria:**
- code can evaluate all 8 chain worlds
- no kernel path silently assumes showdown-on semantics

---

## Wave 3 -- Formal Exploitability

**Goal:** introduce the constitutive adversarial objects missing from the current layer.

### Tasks

- [ ] Create `Exploitability.scala` with:
  - `trait JointRivalProfile`
  - `trait JointRivalProfileClass` or finite representative wrapper for `Sigma^{-S}`
  - `RobustPerformanceFunctional` for `J(b; pi, sigmaMinusS)`
  - `SecurityValue`
  - `pointwiseExploitability`
  - `deploymentExploitability`
- [ ] Choose a concrete tractable representation:
  - finite set of admissible rival profiles for tests
  - adapter hooks to solver oracles for runtime evaluation
- [ ] Define deployment belief-set abstraction for `B_dep`
  - finite representative set in tests
  - interface only in production code
- [ ] Add an explicit deployment-baseline object for A10:
  - baseline policy `bar_pi`
  - deployment belief set `B_dep`
  - baseline exploitability budget `epsilon_base`
  - rationale for how `bar_pi` is selected or approximated in practice
- [ ] Define the robust-value integration boundary:
  - explicit adapter from `WassersteinDroRuntime` / robust-Q evaluation into exploitability and safety code
  - no hidden dependence on ad hoc `Double => Double` lambdas in constitutive APIs
- [ ] Keep old scalar exploitability helpers as wrappers if needed, but make them call the new surface or mark them deprecated

### Tests

- [ ] `ExploitabilityTest.scala`
  - security value dominates any fixed policy value
  - pointwise exploitability is non-negative
  - deployment exploitability is the supremum over a finite belief set
  - deployment baseline object carries `bar_pi`, `B_dep`, and `epsilon_base` coherently
- [ ] Extend `TheoremValidationTest.scala`
  - cover Proposition 9.1 with actual exploitability helpers rather than a comment-only proof

**Exit criteria:**
- `Sigma^{-S}`, `J`, `V^sec`, and exploitability are first-class code objects
- no new safety code depends only on a scalar `Double => Double` exploitability oracle
- baseline exploitability premise from A10 is represented explicitly rather than left implicit

---

## Wave 4 -- Strong Safety and Bellman-safe Certificates

**Goal:** replace old scalar safety with AS-strong and the local Bellman-safe law.

### Tasks

- [ ] Rewrite `AdaptationSafety.scala`
  - keep affine deterrence
  - add AS-strong predicate relative to baseline
  - add robust regret relative to baseline
  - add adaptation-safe policy class surface
  - rename `deltaAdapt` to `epsilonAdapt`
- [ ] Create `SafetyBellman.scala` implementing:
  - one-step baseline loss
  - robust one-step loss
  - `T_safe`
  - `B*`
  - safe action sets
  - safe-feasible policy selector
  - required adaptation budget
  - structural certificate `B_beta`
  - certificate validation and dominance
- [ ] Encode Definition 65 structural constraints for `B_beta` explicitly:
  - terminality
  - non-negativity
  - global bound
  - horizon monotonicity
- [ ] Connect Corollary 9.3 to the deployment-baseline object:
  - compute or bound total vulnerability as `epsilon_base + epsilon_adapt`
  - make clear when the result is exact vs conservative
- [ ] Rework `ExploitationInterpolation.scala`
  - rename `retreatRate` to `cpRetreatRate`
  - keep beta interpolation as an operational control, but make the clamp depend on safety certificates or robust regret bounds rather than only the legacy scalar inequality
- [ ] Update `AssumptionManifest.scala`
  - replace old A10 note with v0.31.1 strengthened safety language
  - add explicit notes for A1′, A2, A4′, A8, and A9 status even if inherited or approximated
  - mention which parts are encoded vs conservative approximations

### Tests

- [ ] `SafetyBellmanTest.scala`
  - `T_safe` monotonicity
  - boundedness under bounded rewards
  - fixed-point iteration converges on finite toy problems
  - certified `B_beta` dominates the computed `B*`
  - structural constraints on `B_beta` are validated directly
- [ ] Extend `AdaptationSafetyTest.scala`
  - AS-strong implies relative exploitability bound
  - total vulnerability budget corollary
  - local-safe action selection implies global degradation bound on finite toy models
- [ ] Extend `TheoremValidationTest.scala`
  - Theorem 9 using finite representative belief/profile classes

**Exit criteria:**
- AS-strong and Bellman-safe certification exist as real operators
- legacy `betaBar` is no longer the only safety story

---

## Wave 5 -- World-aware Risk Decomposition

**Goal:** implement the robust-loss side of the chain decomposition.

### Tasks

- [ ] Create `RiskDecomposition.scala`
  - chain-indexed robust one-step loss
  - risk increments between adjacent chain worlds
  - telescopic risk decomposition
  - marginal layer efficiency metric
- [ ] Add or extend chain-edge helpers in `StrategicValue.scala` or a dedicated `TelescopicDecomposition` helper
- [ ] Ensure chain order is explicit and shared with the kernel world ordering from Wave 2
- [ ] Expose safe handling of zero or negative risk increments for efficiency metrics

### Tests

- [ ] `RiskDecompositionTest.scala`
  - telescopic risk identity
  - sign convention for risk increments
  - efficiency metric undefined / skipped when positive denominator is absent
- [ ] Extend `TheoremValidationTest.scala`
  - Proposition 9.7 on finite toy chains

**Exit criteria:**
- value-edge decomposition and risk decomposition share the same chain-world model
- the audit metric `rho_{k->k+1}` is available from code, not just the spec

---

## Wave 6 -- Bridge and Snapshot Migration

**Goal:** stop presenting old approximations as canonical objects and make the bridge v0.31.1-aware.

### Tasks

- [ ] Rewrite `ValueBridge.scala`
  - stop splitting a single EV gap into fabricated `V10` and `V01`
  - provide a keyed `GridWorld` result
  - where worlds are not directly computable, return `BridgeResult.Absent` or a clearly typed approximate diagnostic
  - add optional chain-world and safety outputs when solver-backed oracles are available
- [ ] Update `StrategicSnapshot.scala`
  - preserve current simple fields for compatibility
  - add optional fields for:
    - keyed grid values
    - chain diagnostics
    - security value
    - safety certificate summary
    - bridge fidelity notes for v0.31.1-only objects
- [ ] Update `BridgeManifest.scala`
  - add entries for exploitability, security value, safety Bellman objects, chain decomposition, and risk decomposition
  - fix old Def numbering references
  - classify unsupported objects honestly
- [ ] Update `OpponentModelBridge.scala` / `BaselineBridge.scala`
  - align names and notes with v0.31.1
  - do not claim exactness where only heuristics exist

### Tests

- [ ] Extend `BridgeTest.scala`
  - no fabricated exact world values
  - absent / approximate paths are explicit and stable
- [ ] Extend `StrategicSnapshotTest.scala`
  - snapshot can carry new optional diagnostics without breaking old consumers
- [ ] Add a manifest completeness assertion
  - every constitutive v0.31.1 object exposed by the bridge has a manifest entry

**Exit criteria:**
- bridge output no longer implies unsupported exactness
- snapshot and manifest can explain what is truly available under v0.31.1

---

## Wave 7 -- Validation, Cleanup, and Removal

**Goal:** finish migration, harden tests, and remove temporary legacy scaffolding.

### Tasks

- [ ] Expand `TheoremValidationTest.scala` coverage from Theorems 1-8 to the v0.31.1 surface added in Waves 3-5
- [ ] Add an explicit result-coverage matrix to the validation layer or plan notes:
  - Proposition 8.1
  - Proposition 9.1, 9.2, 9.5, 9.6, 9.7
  - Theorems 1-9
  - Corollaries 1-4 and 9.3
  - each marked exact / interface-level / inherited / deferred
- [ ] Add a dedicated world-consistency regression:
  - no unqualified bare world variables in strategic code comments
  - canonical chain ordering is shared across decomposition and risk code
- [ ] Add Appendix B regression coverage for endogenous changepoint vulnerability:
  - detector-triggered retreat and `w_reset` interact sanely
  - false-changepoint scenarios do not bypass the safety budget silently
  - changepoint-induced degradation is reported as bounded / approximate when only conservative guarantees exist
- [ ] Close the assumption ledger:
  - verify each of A1′-A10 is represented in code, approximation notes, or deferred-status docs
  - no assumption family remains unclassified
- [ ] Close the computational-architecture triage:
  - Defs 54-56 and §10B marked inherited / validated / deferred
  - if no code change is needed, capture the rationale in docs rather than leaving silent gaps
- [ ] Update `AdaptiveProofHarness` / validation reporting if needed:
  - expose chain coverage
  - expose safety-certificate status
  - expose exploitability / security summaries
  - expose changepoint-vulnerability diagnostics when available
- [ ] Remove deprecated wrappers that are no longer used
- [ ] Delete or deprecate old APIs only after the full strategic test slice is green

### Tests and verification

- [ ] Run targeted strategic tests first
- [ ] Run full `sicfun.holdem.strategic` test slice
- [ ] Run bridge tests
- [ ] Run any proof-harness tests touched by snapshot changes
- [ ] Run changepoint / polarization tests touched by the closure work
- [ ] Perform one final linked-worker review on touched files for contradiction hunting

**Exit criteria:**
- no core file still advertises v0.30.2 safety semantics as current
- legacy shims are reduced to compatibility wrappers or removed entirely
- the strategic layer has a coherent v0.31.1 surface end-to-end
- every assumption family and formal result in scope has an explicit status entry

---

## Recommended Execution Order

1. Wave 0
2. Wave 1
3. Wave 2 and Wave 3 in parallel
4. Wave 4
5. Wave 5
6. Wave 6
7. Wave 7

**Dependency graph:**

```
Wave 0
  |
Wave 1
  |------> Wave 2
  |------> Wave 3
             |
Wave 2 ------|
  |
Wave 4
  |
Wave 5
  |
Wave 6
  |
Wave 7
```

---

## Verification Gates

### Gate A -- Type closure
- `WorldTypesTest`
- `StrategicValueTest`
- no compile errors from renamed symbols

### Gate B -- Kernel closure
- `KernelConstructorTest`
- `RivalKernelLawTest`
- `DynamicsTest`

### Gate C -- Safety and exploitability
- `ExploitabilityTest`
- `AdaptationSafetyTest`
- `SafetyBellmanTest`
- theorem extensions for Proposition 9.1 and Theorem 9

### Gate D -- Bridge honesty
- `BridgeTest`
- `StrategicSnapshotTest`
- manifest completeness check

### Gate E -- End-to-end regression
- full `sicfun.holdem.strategic` test slice
- touched validation / proof harness tests

---

## Known Risks

1. **Finite approximation creep.**
   The spec uses universal quantification; code will need finite representative sets. This is acceptable only if the fidelity of each approximation is explicit in API and manifest.

2. **Bridge consumer breakage.**
   `StrategicSnapshot` and `ValueBridge` are already consumed by tests and possibly validation flows. Keep new data optional first, then migrate consumers.

3. **Over-coupling kernels and solvers.**
   Keep solver usage behind function interfaces or small adapter traits. Do not pull runtime solver details deep into pure value types.

4. **Legacy test assumptions.**
   Existing tests encode old theorem numbering and scalar safety semantics. Expect broad but mechanical test updates once Waves 3-4 land.

5. **Spec text encoding noise.**
   The current markdown has mojibake. Do not rely on raw copied symbols in code comments; prefer ASCII-safe names in code and cite the spec section in prose.

---

## Definition Coverage Target

| Spec area | Target files |
|---|---|
| Def 10 / Def 18 partial-policy typing | `Baseline.scala`, `KernelConstructor.scala`, `RivalKernel.scala` |
| Def 25 / A9 spot polarization | `SpotPolarization.scala`, related tests |
| Defs 33-34 robust Q / ambiguity sets | `Exploitability.scala`, `SafetyBellman.scala`, `WassersteinDroRuntime` adapters |
| Def 44' / Def 44 / 47A / 47B | `WorldTypes.scala`, `StrategicValue.scala` |
| Defs 20-21 revised | `RivalKernel.scala`, `KernelConstructor.scala`, `Dynamics.scala` |
| Defs 52'-52D | `Exploitability.scala` |
| Defs 57 / 57A / 57B / 57C | `AdaptationSafety.scala` |
| Defs 58-66 | `SafetyBellman.scala` |
| Defs 67-69 | `RiskDecomposition.scala` |
| Defs 54-56 / §10B status | docs, validation notes, existing solver / belief files as applicable |
| Assumption families A1′-A10 | `AssumptionManifest.scala`, wave notes, tests |
| Proposition / theorem / corollary ledger | `TheoremValidationTest.scala`, validation notes |
| Bridge closure | `ValueBridge.scala`, `StrategicSnapshot.scala`, `BridgeManifest.scala` |

---

## Completion Definition

This plan is complete when:
- the formal layer has first-class types for chain and grid worlds
- kernels and dynamics can evaluate the worlds required by v0.31.1
- exploitability and safety are expressed in terms of rival-profile classes, not just scalar helper functions
- Bellman-safe certificates and world-aware risk decomposition exist in code and tests
- the bridge no longer fabricates canonical values for unsupported worlds
- manifests, comments, and theorem tests no longer overclaim old-spec alignment
- assumption families, propositions, theorems, corollaries, and §10 recommendations all have explicit implementation-status handling
