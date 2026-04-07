# SICFUN v0.30.2 Formal Layer — Master Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the complete SICFUN v0.30.2 canonical specification (56 definitions, 8 theorems, 10 assumptions) as a formal parallel layer in the existing Scala/C++ codebase.

**Architecture:** Formal types live in `sicfun.holdem.strategic` (pure, no engine deps). An anti-corruption bridge (`sicfun.holdem.strategic.bridge`) connects to the existing engine. Native solvers (POMDP, Wasserstein) are new C++ modules with JNI. The existing engine continues unchanged — the formal layer is strictly additive.

**Tech Stack:** Scala 3.8.1, C++17 (JNI), munit 1.2.2, existing CUDA/CPU native pipeline.

**Canonical spec:** `D:\Users\alexl\Downloads\SICFUN-v0_30_2-canonical.md` (external, not in repo)

**Prior art:** `docs/superpowers/specs/2026-04-02-sicfun-v026-model-design.md` (v0.26 design — reuse namespace and ACL architecture, but all types/laws updated to v0.30.2)

---

## Sub-Plan Index

This master plan is decomposed into 7 sub-plans. Each produces working, testable software. Execute respecting the dependency graph below.

| # | Sub-Plan | File | Defs Covered | ~LOC | Language |
|---|---|---|---|---|---|
| 1 | Foundation Types + BOCD | `2026-04-02-v030-phase1-foundation.md` | 1–14, 26–28, type signatures for 16–21, 44–47, 50 | ~665 | Scala |
| 2 | Tempered Inference | `2026-04-02-v030-phase2-tempered-inference.md` | 15, 15A, 15B | ~135 | C++ mod + Scala |
| 3a | PFT-DPW POMDP Solver | `2026-04-02-v030-phase3a-pft-dpw.md` | 29–32, 54–55 | ~570 | C++ + JNI |
| 3b | Wasserstein EMD + DRO | `2026-04-02-v030-phase3b-wasserstein.md` | 33–34 | ~490 | C++ + JNI |
| 3c | W-POMCP Multi-Agent | `2026-04-02-v030-phase3c-wpomcp.md` | 56 | ~1,200 | C++ + JNI |
| 4a | Kernel Engine + Dynamics | `2026-04-02-v030-phase4a-kernels.md` | 15C, 16–25 | ~900 | Scala |
| 4b | Decomposition + Safety + Bridge | `2026-04-02-v030-phase4b-decomposition.md` | 35–53, Thms 1–8 | ~1,150 | Scala |

**Total:** ~5,110 LOC across 60 definitions.

---

## Dependency Graph

```
Phase 1 (Foundation) ──────────────────────────────► Phase 4a (Kernels 16-21)
   │                                                      │
   └► Phase 2 (Tempered Inference) ──────────────────────►┘
                                                           │
Phase 3a (PFT-DPW) ──────────────────────────────────────►│
Phase 3b (Wasserstein + LP) ──────────────────────────────►│
Phase 3c (W-POMCP, needs 3a) ────────────────────────────►│
                                                           ▼
                                                     Phase 4b-4n
                                                 (decomposition, bluff,
                                                  safety, reveal)
```

Phases 1, 2, 3a, 3b are **parallelizable**. Phase 3c depends on 3a. Phase 4a depends on 1+2. Phase 4b depends on 4a (and transitively on 3a-3c for full integration tests).

---

## Architectural Decisions (inherited from v0.26 design, updated for v0.30.2)

### Namespace

```
sicfun.holdem.strategic           →  Pure formal types + operators (NO engine deps)
sicfun.holdem.strategic.bridge    →  ACL connecting formal ↔ engine
sicfun.holdem.strategic.solver    →  Scala wrappers over native POMDP/DRO solvers
```

### Dependency Rule (non-negotiable)

```
sicfun.holdem.strategic  ->  sicfun.holdem.types   (ONLY: PokerAction, Street, Position, HoleCards, Board)
sicfun.holdem.strategic  ->  sicfun.core           (DiscreteDistribution, Probability)
sicfun.holdem.strategic  X-> sicfun.holdem.engine   (PROHIBITED)
sicfun.holdem.strategic  X-> sicfun.holdem.runtime  (PROHIBITED)

sicfun.holdem.strategic.bridge  ->  sicfun.holdem.strategic  (formal types)
sicfun.holdem.strategic.bridge  ->  sicfun.holdem.engine     (current engine)
sicfun.holdem.strategic.bridge  ->  sicfun.holdem.types      (GameState, for conversion)
```

Only `bridge` sees both worlds. The engine NEVER imports `strategic`.

### Multiway-Native Mandate (inherited, non-negotiable)

> Heads-up is the subcase |R| = 1 of the multiway domain.
> No implicit collapse of multiway to pairwise is permitted.

### v0.30.2 Retired Symbols

- `Δ_learn` — RETIRED (was in v0.26, removed in v0.29)
- `π^{cf,S}` — RETIRED (replaced by joint reference profile Γ^ref)
- ε-regularized likelihood (v0.29.1) — RETIRED as default (available as legacy fallback)

### v0.30.2 New vs v0.26

| v0.26 had | v0.30.2 replaces with |
|---|---|
| 3-term decomposition (Q^my + Δ_learn + Δ_sig) | 4-world decomposition (Defs 44-47) + per-rival Δ (Defs 40-43) |
| ε-smoothing | Two-layer tempered likelihood (Def 15A) |
| Static rival types (A3) | Non-stationary types + changepoint detection (A3', §5A) |
| No safety guarantee | Adaptation safety (A10, Def 52) |
| No robustness | Wasserstein DRO (§6A) |
| No exploitation control | β^{i,exploit} interpolation (Def 15C) |
| Monolithic Δ_sig | Sub-decomposed: Δ_sig,design + Δ_sig,real (Defs 48-49) |

---

## File Map (all phases)

### Phase 1 — Foundation (Scala)

```
src/main/scala/sicfun/holdem/strategic/
├── StrategicClass.scala          Defs 1-4: V/B/M/SB enum + predicates
├── Signal.scala                  Defs 5-8: ActionSignal, ShowdownSignal, TotalSignal, routing
├── Baseline.scala                Defs 9-10: RealBaseline, AttributedBaseline traits
├── ReputationalProjection.scala  Def 11: φ^{S,i} = g^i(m^{R,i})
├── AugmentedState.scala          Defs 12-14: X̃ product type, OperativeBelief
├── RivalKernel.scala             Defs 16-21: trait hierarchy (type signatures only)
├── StrategicValue.scala          Defs 44-47, 50: FourWorld, Δ-vocabulary types
├── ChangepointDetector.scala     Defs 26-28: Adams-MacKay BOCD
├── TableStructure.scala          PlayerId, TableMap, RivalMap (from v0.26 design)
├── DomainTypes.scala             Chips, PotFraction, Ev opaques
└── Fidelity.scala                BridgeResult, Fidelity, Severity (from v0.26 design)

src/test/scala/sicfun/holdem/strategic/
├── StrategicClassTest.scala
├── SignalTest.scala
├── AugmentedStateTest.scala
├── RivalKernelLawTest.scala
├── StrategicValueTest.scala
├── ChangepointDetectorTest.scala
└── TableStructureTest.scala
```

### Phase 2 — Tempered Inference (C++ mod + Scala)

```
src/main/native/jni/
├── BayesNativeUpdateCore.hpp     MODIFY: add κ_temp, δ_floor, η params to update loop
├── HoldemBayesNativeCpuBindings.cpp  MODIFY: JNI signature with tempering params
├── HoldemBayesNativeGpuBindings.cu   MODIFY: CUDA kernel with tempering

src/main/scala/sicfun/holdem/
├── provider/HoldemBayesProvider.scala  MODIFY: dispatch with TemperedConfig
├── gpu/HoldemBayesNativeRuntime.scala  MODIFY: pass tempering params to native

src/main/scala/sicfun/holdem/strategic/
└── TemperedLikelihood.scala      Defs 15, 15A, 15B: pure Scala reference impl + config

src/test/scala/sicfun/holdem/strategic/
└── TemperedLikelihoodTest.scala  Theorem 1 (totality), backward compat (κ=1,δ=ε)
```

### Phase 3 — Native Solvers (new C++ + JNI)

```
src/main/native/vendor/
├── network_simplex_simple.h      VENDOR: nbonneel (MIT), Wasserstein EMD
└── full_bipartitegraph.h         VENDOR: nbonneel (MIT), graph structure

src/main/native/jni/
├── PftDpwSolver.hpp              NEW: PFT-DPW single-agent POMDP tree search
├── WPomcpSolver.hpp              NEW: W-POMCP multi-agent factored PF (extends PFT)
├── WassersteinEmd.hpp            NEW: JNI wrapper over nbonneel
├── HoldemPomcpNativeBindings.cpp NEW: JNI entry points for POMDP solvers
└── HoldemWassersteinBindings.cpp NEW: JNI entry points for Wasserstein

src/main/scala/sicfun/holdem/strategic/solver/
├── PftDpwRuntime.scala           Scala wrapper: Defs 29-32, 54-55
├── WPomcpRuntime.scala           Scala wrapper: Def 56
└── WassersteinDroRuntime.scala   Scala wrapper: Defs 33-34 (+ GLPK-java for LP)

src/test/scala/sicfun/holdem/strategic/solver/
├── PftDpwRuntimeTest.scala
├── WPomcpRuntimeTest.scala
└── WassersteinDroRuntimeTest.scala
```

### Phase 4 — Integration (Scala logic over native)

```
src/main/scala/sicfun/holdem/strategic/
├── KernelConstructor.scala       Defs 16-21 impl: BuildRivalKernel, action/showdown/design kernels
├── ExploitationInterpolation.scala  Def 15C: β^{i,exploit} + retreat + safety constraint
├── Dynamics.scala                Defs 22-25: belief update, rival-state update, polarization
├── BluffFramework.scala          Defs 35-39: structural bluff, gain, exploitative bluff
├── SignalDecomposition.scala     Defs 40-43: per-rival Δ_sig, Δ_pass, Δ_manip
├── FourWorldDecomposition.scala  Defs 44-47 impl: V^{1,1}...V^{0,0} computation
├── SignalingSubDecomposition.scala  Defs 48-49: Δ_sig,design + Δ_sig,real
├── RevealSchedule.scala          Def 51: stage-indexed reveal threshold
├── AdaptationSafety.scala        Defs 52-53: exploit bound + affine deterrence
├── DetectionPredicate.scala      A6': DetectModeling^i
└── SpotPolarization.scala        Def 25, A9: spot-conditioned polarization

src/main/scala/sicfun/holdem/strategic/bridge/
├── PublicStateBridge.scala
├── OpponentModelBridge.scala
├── BaselineBridge.scala
├── ValueBridge.scala
├── SignalBridge.scala
├── ClassificationBridge.scala
└── BridgeManifest.scala

src/test/scala/sicfun/holdem/strategic/
├── KernelConstructorTest.scala
├── ExploitationInterpolationTest.scala
├── DynamicsTest.scala
├── BluffFrameworkTest.scala
├── SignalDecompositionTest.scala
├── FourWorldDecompositionTest.scala
├── AdaptationSafetyTest.scala
└── TheoremValidationTest.scala   Theorems 1-8, Corollaries 1-4
```

---

## Backward Compatibility Contract

v0.30.2 §12.2 states exact backward compatibility with v0.29.1 when:
- κ_temp = 1, δ_floor = ε (legacy smoothing)
- ρ = 0 (no robustness)
- h^i = 0 for all rivals (no changepoint)
- Design-signal kernel collapsed to blind kernel
- β^{i,exploit} = 1 for all rivals

**Every phase must include a backward-compatibility test proving v0.29.1 recovery.**

---

## External Dependencies (new)

| Dependency | Type | License | Phase |
|---|---|---|---|
| nbonneel/network_simplex | 2 C++ headers, vendored | MIT | 3 |
| GLPK-java (`org.gnu.glpk:glpk-java:1.12.0`) | Maven dep | GPL-3.0 | 3 |

**Alternative to GLPK (if GPL is a concern):** HiGHS via highs4j (Apache 2.0).

---

## Success Criteria

1. All 60 definitions have corresponding Scala types or C++ implementations
2. All 8 theorems have validation tests that pass
3. All 4 corollaries have property-based tests
4. Backward-compatibility test recovers v0.29.1 behavior exactly
5. Existing test suite passes unchanged (formal layer is additive)
6. Bridge manifest covers all formal objects with declared fidelity

---

## Execution Order

**Wave 1 (parallel):** Phase 1, Phase 2, Phase 3a, Phase 3b — all independent
**Wave 2:** Phase 3c (needs 3a)
**Wave 3:** Phase 4a (needs 1 + 2)
**Wave 4:** Phase 4b (needs 4a, 3a-3c for integration tests)

Start with Phase 1 — pure Scala, zero risk, unlocks everything else.
See: `docs/superpowers/plans/2026-04-02-v030-phase1-foundation.md`
