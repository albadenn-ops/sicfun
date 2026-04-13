# Repo Filesystem Restructure — Design Spec

**Date**: 2026-04-13
**Goal**: Reorganize the sicfun repo's folder structure for clarity and cohesion. Zero behavior or semantic changes — only file moves, package declaration updates, and import rewrites.

## Constraints

- No logic changes. Every file is a pure move + package/import update.
- JNI files can move freely — SBT rebuilds DLLs after package changes.
- Test files mirror source moves (same sub-package structure).
- Existing sub-packages (`strategic/bridge/`, `strategic/solver/`) are kept as-is.

---

## 1. `holdem/strategic/` — 32 flat files → 6 sub-packages

| Sub-package | Files | Rationale |
|---|---|---|
| `types/` | DomainTypes, WorldTypes, CertificationTypes, Fidelity, StrategicClass, TableStructure (6) | Foundation types, opaque wrappers, enums. Zero dependencies. |
| `state/` | AugmentedState, Signal, StrategicRivalBelief (3) | Observation and belief state representations. |
| `kernel/` | RivalKernel, KernelConstructor, TemperedLikelihood, Dynamics, ChangepointDetector (5) | Bayesian belief update pipeline (Defs 15-28). |
| `safety/` | SafetyBellman, AdaptationSafety, PerStateLossEvaluator, DetectionPredicate, Baseline, OperationalBaseline (6) | Safety certification, baselines, bounds (Defs 52-66). |
| `exploitation/` | Exploitability, ExploitationInterpolation, BluffFramework, SpotPolarization, RevealSchedule, ReputationalProjection (6) | Adversarial analysis, bluff framework, exploit strategy. |
| `decomposition/` | SignalDecomposition, SignalingSubDecomposition, RiskDecomposition, FourWorldDecomposition, StrategicValue (5) | Value telescoping and four-world decomposition (Theorems 3-4). |

**Kept as-is**: `bridge/` (8), `solver/` (3).
**Stays at root**: `AssumptionManifest.scala` (meta-documentation).

## 2. `holdem/engine/` — 13 files → 2 sub-packages + 6 at root

| Sub-package | Files | Rationale |
|---|---|---|
| `inference/` | RangeInferenceEngine, MultiwayInferenceEngine, ShowdownPriorBias (3) | Bayesian range inference pipeline. |
| `villain/` | VillainResponseModel, ArchetypeLearning, ArchetypeVillainResponder, RealTimeAdaptiveEngine (4) | Opponent modeling cluster. |

**Root (6)**: GtoSolveEngine, HeroDecisionPipeline, HandStrengthEstimator, StrategicEngine, PokerPftFormulation, PokerPomcpFormulation.

## 3. `holdem/runtime/` — 12 files → 1 sub-package + 8 at root

| Sub-package | Files | Rationale |
|---|---|---|
| `protocol/` | AcpcMatchRunner, AcpcHeadsUpDealer, SlumbotMatchRunner, MatchRunnerSupport (4) | External protocol integrations with wire codecs. |

**Root (8)**: AdvisorSession, PokerAdvisor, StrategicAdvisorBridge, AlwaysOnDecisionLoop, HandHistoryAnalyzer, HeadsUpMatchDefaults, TexasHoldemPlayingHall, LiveHandSimulator.

## 4. `holdem/bench/` — 25 files → 2 sub-packages + 18 at root

| Sub-package | Files | Rationale |
|---|---|---|
| `tuner/` | HeadsUpBackendAutoTuner, HeadsUpRangeGpuAutoTuner, HoldemPostflopGpuAutoTuner, GlobalGpuTuningTool (4) | Persistent auto-tuners that mutate config files. |
| `gate/` | HeadsUpGpuExactParityGate, HeadsUpGpuPocGate, HeadsUpGpuSmokeGate (3) | Pass/fail correctness gates (not perf measurement). |

**Root (18)**: All `*Benchmark` files + `BenchSupport`.

## 5. Packages unchanged

These are cohesive enough to stay flat:

- `core/` (15), `holdem/types/` (8), `holdem/equity/` (10), `holdem/cfr/` (8), `holdem/gpu/` (7), `holdem/io/` (7), `holdem/model/` (6), `holdem/provider/` (4), `holdem/history/` (9), `holdem/tablegen/` (7), `holdem/cli/` (3), `holdem/validation/` (10), `holdem/web/` (3), `src/main/native/` (14).

## 6. Non-source restructuring

### Root cleanup

- `.gitignore` additions: `hs_err_pid*.log`, `replay_pid*.log`, `tmp-*.log`
- Move `SICFUN-v0_31_1-corrected.md` → `docs/specs/`
- AI config files (CLAUDE.md, AGENTS.md, etc.) stay at root (convention).

### `scripts/` — 33 flat files → 4 sub-directories

| Sub-directory | Files | Content |
|---|---|---|
| `match/` | 9 | Match running and ACPC launcher scripts |
| `gpu/` | 6 | GPU build, gate, and tuning scripts |
| `ai/` | 4 | AI minion system scripts |
| `validation/` | 3 | Validation, proof, and runbook scripts |

**Kept**: `packaged-hand-history-web/` (add `start-hand-history-web.ps1` into it), `logs/`.
**Root (6)**: release-windows, release-hand-history-web, archive-context, generate-ddre-smoke-onnx.py, parent-dashboard, profile-hall-jfr.

### `docs/` — flat → 2 sub-directories

| Sub-directory | Files | Content |
|---|---|---|
| `ai/` | 4 | AI_MINIONS.md, GEMINI_MINION.md, ai-code-navigation.md, AI_CONTEXT_ARCHIVE.md |
| `specs/` | 4 | SICFUN_Phase1_DDRE_Spec_v2.md, _phase1_spec_extracted.txt, _phase1_spec_extracted_allp.txt, SICFUN-v0_31_1-corrected.md |

**Kept**: `site-preview-hybrid/`, `superpowers/`.
**Root (3)**: OPERATOR_RUNBOOK.md, HAND_HISTORY_WEB_DEPLOYMENT.md, sicfun-comment-review-2026-03-31.md.

## 7. Test tree

Every test file mirrors its source file's move. E.g., `test/sicfun/holdem/strategic/DynamicsTest.scala` → `test/sicfun/holdem/strategic/kernel/DynamicsTest.scala`.

## 8. Implementation notes

- Each sub-package move is an independent unit of work (can be parallelized).
- After all moves: full `sbt compile` to verify no broken imports.
- After compile: full `sbt test` to verify no test regressions.
- Native DLL rebuild triggered automatically by SBT after JNI package changes.
- References in scripts, docs, CLAUDE.md, AGENTS.md, and memory files that mention package paths must be updated.
