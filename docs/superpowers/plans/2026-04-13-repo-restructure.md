# Repo Filesystem Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize sicfun's folder structure for clarity and cohesion — zero behavior changes.

**Architecture:** Pure file-move refactoring. Each task creates sub-directories, `git mv`s files, updates `package` declarations, fixes imports across the codebase, and compile-gates before committing. Tasks are ordered by dependency depth: deepest-imported packages first, then consumers.

**Tech Stack:** Scala 3, SBT, git

**Spec:** `docs/superpowers/specs/2026-04-13-repo-restructure-design.md`

---

## Task Ordering & Dependencies

Tasks 1-3 (non-source) are independent of tasks 4-7 (source). Source tasks must run sequentially:
- Task 4 (`strategic/`) first — imported by engine, runtime, bench
- Task 5 (`engine/`) second — imported by runtime
- Task 6 (`runtime/`) third — imported by bench
- Task 7 (`bench/`) last — leaf consumer

Task 8 (reference updates) runs after all moves. Tasks 9-10 are final gates.

---

### Task 1: Root cleanup

**Files:**
- Modify: `.gitignore`
- Move: `SICFUN-v0_31_1-corrected.md` → `docs/specs/SICFUN-v0_31_1-corrected.md`

- [ ] **Step 1: Add crash/temp log patterns to .gitignore**

Append these lines to `.gitignore`:

```
# JVM crash logs
hs_err_pid*.log
replay_pid*.log

# Temp logs
tmp-*.log
```

- [ ] **Step 2: Create docs/specs/ and move the spec file**

```bash
mkdir -p docs/specs
git mv SICFUN-v0_31_1-corrected.md docs/specs/SICFUN-v0_31_1-corrected.md
```

- [ ] **Step 3: Remove existing crash/temp logs from tracking**

```bash
git rm --cached hs_err_pid*.log replay_pid*.log tmp-*.log 2>/dev/null || true
```

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: clean up root — gitignore crash logs, move spec to docs/specs/"
```

---

### Task 2: Reorganize scripts/

**Files:**
- Move 9 files → `scripts/match/`
- Move 6 files → `scripts/gpu/`
- Move 4 files → `scripts/ai/`
- Move 3 files → `scripts/validation/`
- Move 1 file → `scripts/packaged-hand-history-web/`

- [ ] **Step 1: Create sub-directories**

```bash
mkdir -p scripts/match scripts/gpu scripts/ai scripts/validation
```

- [ ] **Step 2: Move match runner scripts**

```bash
git mv scripts/run-g5-matchup.ps1 scripts/match/
git mv scripts/run-hall-matchups.ps1 scripts/match/
git mv scripts/run-playing-hall.ps1 scripts/match/
git mv scripts/run-playing-hall-max.ps1 scripts/match/
git mv scripts/run-slumbot-benchmark.ps1 scripts/match/
git mv scripts/start-g5-acpc.cmd scripts/match/
git mv scripts/start-g5-acpc.ps1 scripts/match/
git mv scripts/start-sicfun-acpc.cmd scripts/match/
git mv scripts/start-sicfun-acpc.ps1 scripts/match/
```

- [ ] **Step 3: Move GPU scripts**

```bash
git mv scripts/build-g5-acpc.ps1 scripts/gpu/
git mv scripts/ensure-gpu-build-prereqs.ps1 scripts/gpu/
git mv scripts/gpu-exact-parity-gate.ps1 scripts/gpu/
git mv scripts/gpu-smoke-gate.ps1 scripts/gpu/
git mv scripts/run-global-tuning.ps1 scripts/gpu/
git mv scripts/prove-global-gpu-tuning-portability.ps1 scripts/gpu/
```

- [ ] **Step 4: Move AI minion scripts**

```bash
git mv scripts/ai-minion.ps1 scripts/ai/
git mv scripts/gemini-sidecar.ps1 scripts/ai/
git mv scripts/gemini_cli.py scripts/ai/
git mv scripts/import-ai-nav.ps1 scripts/ai/
```

- [ ] **Step 5: Move validation scripts**

```bash
git mv scripts/prove-pipeline.ps1 scripts/validation/
git mv scripts/run-texassolver-proof.ps1 scripts/validation/
git mv scripts/runbook.ps1 scripts/validation/
```

- [ ] **Step 6: Move web startup script into packaged dir**

```bash
git mv scripts/start-hand-history-web.ps1 scripts/packaged-hand-history-web/
```

- [ ] **Step 7: Fix cross-references in moved scripts**

Search all moved scripts for references to other scripts by relative path and update them. Key patterns to check:

```bash
grep -rn '\.\./\|scripts/' scripts/match/ scripts/gpu/ scripts/ai/ scripts/validation/ scripts/packaged-hand-history-web/start-hand-history-web.ps1
```

Fix any relative path references that broke due to the new directory depth. Common pattern: `./some-script.ps1` → `../some-script.ps1` or similar.

- [ ] **Step 8: Commit**

```bash
git add scripts/
git commit -m "chore: organize scripts/ into match, gpu, ai, validation sub-dirs"
```

---

### Task 3: Reorganize docs/

**Files:**
- Move 4 files → `docs/ai/`
- Move 3 files → `docs/specs/` (1 already moved in Task 1)

- [ ] **Step 1: Create sub-directories**

```bash
mkdir -p docs/ai docs/specs
```

- [ ] **Step 2: Move AI docs**

```bash
git mv docs/AI_MINIONS.md docs/ai/
git mv docs/GEMINI_MINION.md docs/ai/
git mv docs/ai-code-navigation.md docs/ai/
git mv docs/AI_CONTEXT_ARCHIVE.md docs/ai/
```

- [ ] **Step 3: Move spec docs**

```bash
git mv docs/SICFUN_Phase1_DDRE_Spec_v2.md docs/specs/
git mv docs/_phase1_spec_extracted.txt docs/specs/
git mv docs/_phase1_spec_extracted_allp.txt docs/specs/
```

- [ ] **Step 4: Fix cross-references**

Check CLAUDE.md, AGENTS.md, AI_ENTRYPOINT.md, and any moved docs for references to the old paths. Update them.

Key files to search:
```bash
grep -rn 'AI_MINIONS\|GEMINI_MINION\|ai-code-navigation\|AI_CONTEXT_ARCHIVE\|SICFUN_Phase1\|_phase1_spec' CLAUDE.md AGENTS.md AI_ENTRYPOINT.md README.md docs/
```

- [ ] **Step 5: Commit**

```bash
git add docs/ CLAUDE.md AGENTS.md AI_ENTRYPOINT.md README.md
git commit -m "chore: organize docs/ into ai/ and specs/ sub-dirs"
```

---

### Task 4: Restructure `holdem/strategic/`

This is the largest task — 32 source files + ~27 test files + import updates across the codebase.

**Source base**: `src/main/scala/sicfun/holdem/strategic/`
**Test base**: `src/test/scala/sicfun/holdem/strategic/`

#### Sub-package mapping

| New sub-package | Source files to move |
|---|---|
| `types/` | DomainTypes, WorldTypes, CertificationTypes, Fidelity, StrategicClass, TableStructure |
| `state/` | AugmentedState, Signal, StrategicRivalBelief |
| `kernel/` | RivalKernel, KernelConstructor, TemperedLikelihood, Dynamics, ChangepointDetector |
| `safety/` | SafetyBellman, AdaptationSafety, PerStateLossEvaluator, DetectionPredicate, Baseline, OperationalBaseline |
| `exploitation/` | Exploitability, ExploitationInterpolation, BluffFramework, SpotPolarization, RevealSchedule, ReputationalProjection |
| `decomposition/` | SignalDecomposition, SignalingSubDecomposition, RiskDecomposition, FourWorldDecomposition, StrategicValue |

**Stays at root**: AssumptionManifest.scala
**Test files stay at root** (cross-cutting): TheoremValidationTest, FormalClosureValidationTest, ReductionismManifestTest, PosteriorAttributedBaselineTest

- [ ] **Step 1: Create source sub-directories**

```bash
S=src/main/scala/sicfun/holdem/strategic
mkdir -p $S/types $S/state $S/kernel $S/safety $S/exploitation $S/decomposition
```

- [ ] **Step 2: Move source files — types/**

```bash
S=src/main/scala/sicfun/holdem/strategic
git mv $S/DomainTypes.scala $S/types/
git mv $S/WorldTypes.scala $S/types/
git mv $S/CertificationTypes.scala $S/types/
git mv $S/Fidelity.scala $S/types/
git mv $S/StrategicClass.scala $S/types/
git mv $S/TableStructure.scala $S/types/
```

- [ ] **Step 3: Move source files — state/**

```bash
S=src/main/scala/sicfun/holdem/strategic
git mv $S/AugmentedState.scala $S/state/
git mv $S/Signal.scala $S/state/
git mv $S/StrategicRivalBelief.scala $S/state/
```

- [ ] **Step 4: Move source files — kernel/**

```bash
S=src/main/scala/sicfun/holdem/strategic
git mv $S/RivalKernel.scala $S/kernel/
git mv $S/KernelConstructor.scala $S/kernel/
git mv $S/TemperedLikelihood.scala $S/kernel/
git mv $S/Dynamics.scala $S/kernel/
git mv $S/ChangepointDetector.scala $S/kernel/
```

- [ ] **Step 5: Move source files — safety/**

```bash
S=src/main/scala/sicfun/holdem/strategic
git mv $S/SafetyBellman.scala $S/safety/
git mv $S/AdaptationSafety.scala $S/safety/
git mv $S/PerStateLossEvaluator.scala $S/safety/
git mv $S/DetectionPredicate.scala $S/safety/
git mv $S/Baseline.scala $S/safety/
git mv $S/OperationalBaseline.scala $S/safety/
```

- [ ] **Step 6: Move source files — exploitation/**

```bash
S=src/main/scala/sicfun/holdem/strategic
git mv $S/Exploitability.scala $S/exploitation/
git mv $S/ExploitationInterpolation.scala $S/exploitation/
git mv $S/BluffFramework.scala $S/exploitation/
git mv $S/SpotPolarization.scala $S/exploitation/
git mv $S/RevealSchedule.scala $S/exploitation/
git mv $S/ReputationalProjection.scala $S/exploitation/
```

- [ ] **Step 7: Move source files — decomposition/**

```bash
S=src/main/scala/sicfun/holdem/strategic
git mv $S/SignalDecomposition.scala $S/decomposition/
git mv $S/SignalingSubDecomposition.scala $S/decomposition/
git mv $S/RiskDecomposition.scala $S/decomposition/
git mv $S/FourWorldDecomposition.scala $S/decomposition/
git mv $S/StrategicValue.scala $S/decomposition/
```

- [ ] **Step 8: Update package declarations in all moved source files**

For each sub-package, update the `package` line at the top of every moved file:

- Files in `types/`: change `package sicfun.holdem.strategic` → `package sicfun.holdem.strategic.types`
- Files in `state/`: change `package sicfun.holdem.strategic` → `package sicfun.holdem.strategic.state`
- Files in `kernel/`: change `package sicfun.holdem.strategic` → `package sicfun.holdem.strategic.kernel`
- Files in `safety/`: change `package sicfun.holdem.strategic` → `package sicfun.holdem.strategic.safety`
- Files in `exploitation/`: change `package sicfun.holdem.strategic` → `package sicfun.holdem.strategic.exploitation`
- Files in `decomposition/`: change `package sicfun.holdem.strategic` → `package sicfun.holdem.strategic.decomposition`

Use batch sed or the Edit tool. The replacement is mechanical: first `package` line in each file.

**IMPORTANT**: Some moved files import other symbols from `sicfun.holdem.strategic` that also moved to different sub-packages. These intra-strategic imports must also be updated. For example, `KernelConstructor.scala` (now in `kernel/`) may import `DomainTypes` (now in `types/`). After updating the package declaration, add the required cross-sub-package imports:

```scala
// Example: in kernel/KernelConstructor.scala
package sicfun.holdem.strategic.kernel

import sicfun.holdem.strategic.types.*
import sicfun.holdem.strategic.state.*
// ... etc as needed
```

Use `sbt compile` output to identify which cross-imports are needed.

- [ ] **Step 9: Create test sub-directories and move test files**

```bash
T=src/test/scala/sicfun/holdem/strategic
mkdir -p $T/types $T/state $T/kernel $T/safety $T/exploitation $T/decomposition
```

Move test files to mirror source:

```bash
# types/
git mv $T/WorldTypesTest.scala $T/types/
git mv $T/CertificationTypesTest.scala $T/types/
git mv $T/StrategicClassTest.scala $T/types/
git mv $T/TableStructureTest.scala $T/types/

# state/
git mv $T/AugmentedStateTest.scala $T/state/
git mv $T/SignalTest.scala $T/state/
git mv $T/StrategicRivalBeliefTest.scala $T/state/

# kernel/
git mv $T/DynamicsTest.scala $T/kernel/
git mv $T/KernelConstructorTest.scala $T/kernel/
git mv $T/RivalKernelLawTest.scala $T/kernel/
git mv $T/TemperedLikelihoodTest.scala $T/kernel/
git mv $T/ChangepointDetectorTest.scala $T/kernel/

# safety/
git mv $T/SafetyBellmanTest.scala $T/safety/
git mv $T/AdaptationSafetyTest.scala $T/safety/
git mv $T/PerStateLossEvaluatorTest.scala $T/safety/
git mv $T/OperationalBaselineTest.scala $T/safety/

# exploitation/
git mv $T/ExploitabilityTest.scala $T/exploitation/
git mv $T/ExploitationInterpolationTest.scala $T/exploitation/
git mv $T/BluffFrameworkTest.scala $T/exploitation/

# decomposition/
git mv $T/SignalDecompositionTest.scala $T/decomposition/
git mv $T/RiskDecompositionTest.scala $T/decomposition/
git mv $T/FourWorldDecompositionTest.scala $T/decomposition/
git mv $T/StrategicValueTest.scala $T/decomposition/
```

Cross-cutting tests stay at `$T/` root: TheoremValidationTest, FormalClosureValidationTest, ReductionismManifestTest, PosteriorAttributedBaselineTest.

- [ ] **Step 10: Update package declarations in moved test files**

Same mechanical replacement as Step 8, but for test files. Also update any test imports that reference moved source symbols.

- [ ] **Step 11: Fix imports across the entire codebase**

Find every file outside `strategic/` that imports from `sicfun.holdem.strategic`:

```bash
grep -rn 'import sicfun\.holdem\.strategic\.' src/main/scala src/test/scala --include='*.scala' | grep -v '/strategic/'
```

For each match, determine whether the imported symbol moved to a sub-package and update the import accordingly. Common patterns:

| Old import | New import |
|---|---|
| `import sicfun.holdem.strategic.Dynamics` | `import sicfun.holdem.strategic.kernel.Dynamics` |
| `import sicfun.holdem.strategic.*` | Add specific sub-package imports as needed |
| `import sicfun.holdem.strategic.{Foo, Bar}` | Split into per-sub-package imports |

Files most likely affected (based on dependency analysis):
- `holdem/engine/StrategicEngine.scala` — heavy strategic imports
- `holdem/engine/PokerPftFormulation.scala` — imports strategic types
- `holdem/engine/PokerPomcpFormulation.scala` — imports strategic solver
- `holdem/runtime/StrategicAdvisorBridge.scala` — imports strategic
- `holdem/validation/*.scala` — may import strategic types
- `holdem/strategic/bridge/*.scala` — imports from parent package
- `holdem/strategic/solver/*.scala` — imports from parent package

- [ ] **Step 12: Compile gate**

```bash
sbt compile
```

Fix any remaining import errors until compilation succeeds.

- [ ] **Step 13: Commit**

```bash
git add -A
git commit -m "refactor: restructure strategic/ into types, state, kernel, safety, exploitation, decomposition sub-packages"
```

---

### Task 5: Restructure `holdem/engine/`

**Source base**: `src/main/scala/sicfun/holdem/engine/`
**Test base**: `src/test/scala/sicfun/holdem/engine/`

| New sub-package | Source files | Test files |
|---|---|---|
| `inference/` | RangeInferenceEngine, MultiwayInferenceEngine, ShowdownPriorBias | RangeInferenceEngineTest, MultiwayInferenceEngineTest, ShowdownPriorBiasTest |
| `villain/` | VillainResponseModel, ArchetypeLearning, ArchetypeVillainResponder, RealTimeAdaptiveEngine | VillainResponseModelTest, ArchetypeLearningTest, ArchetypeVillainResponderTest, RealTimeAdaptiveEngineTest |

- [ ] **Step 1: Create sub-directories**

```bash
E=src/main/scala/sicfun/holdem/engine
T=src/test/scala/sicfun/holdem/engine
mkdir -p $E/inference $E/villain $T/inference $T/villain
```

- [ ] **Step 2: Move source files**

```bash
E=src/main/scala/sicfun/holdem/engine
# inference/
git mv $E/RangeInferenceEngine.scala $E/inference/
git mv $E/MultiwayInferenceEngine.scala $E/inference/
git mv $E/ShowdownPriorBias.scala $E/inference/

# villain/
git mv $E/VillainResponseModel.scala $E/villain/
git mv $E/ArchetypeLearning.scala $E/villain/
git mv $E/ArchetypeVillainResponder.scala $E/villain/
git mv $E/RealTimeAdaptiveEngine.scala $E/villain/
```

- [ ] **Step 3: Update package declarations in moved source files**

- Files in `inference/`: `package sicfun.holdem.engine` → `package sicfun.holdem.engine.inference`
- Files in `villain/`: `package sicfun.holdem.engine` → `package sicfun.holdem.engine.villain`

Also fix intra-engine imports in moved files (e.g., `inference/RangeInferenceEngine` may import `VillainResponseModel` which is now in `villain/`).

- [ ] **Step 4: Move test files**

```bash
T=src/test/scala/sicfun/holdem/engine
git mv $T/RangeInferenceEngineTest.scala $T/inference/
git mv $T/MultiwayInferenceEngineTest.scala $T/inference/
git mv $T/ShowdownPriorBiasTest.scala $T/inference/

git mv $T/VillainResponseModelTest.scala $T/villain/
git mv $T/ArchetypeLearningTest.scala $T/villain/
git mv $T/ArchetypeVillainResponderTest.scala $T/villain/
git mv $T/RealTimeAdaptiveEngineTest.scala $T/villain/
```

- [ ] **Step 5: Update package declarations in moved test files**

Same mechanical replacement + fix test imports.

- [ ] **Step 6: Fix imports across the codebase**

```bash
grep -rn 'import sicfun\.holdem\.engine\.' src/main/scala src/test/scala --include='*.scala' | grep -v '/engine/'
```

Key consumers to check:
- `holdem/runtime/*.scala` — imports HeroDecisionPipeline, RealTimeAdaptiveEngine, RangeInferenceEngine
- `holdem/io/HoldemDdreDatasetIO.scala` — imports engine types
- `holdem/history/OpponentIdentity.scala` — imports engine types
- `holdem/bench/*.scala` — imports engine types

- [ ] **Step 7: Compile gate**

```bash
sbt compile
```

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: restructure engine/ into inference/ and villain/ sub-packages"
```

---

### Task 6: Restructure `holdem/runtime/`

**Source base**: `src/main/scala/sicfun/holdem/runtime/`
**Test base**: `src/test/scala/sicfun/holdem/runtime/`

| New sub-package | Source files | Test files |
|---|---|---|
| `protocol/` | AcpcMatchRunner, AcpcHeadsUpDealer, SlumbotMatchRunner, MatchRunnerSupport | AcpcMatchRunnerTest, SlumbotActionCodecTest, MatchRunnerSupportTest |

- [ ] **Step 1: Create sub-directories**

```bash
R=src/main/scala/sicfun/holdem/runtime
T=src/test/scala/sicfun/holdem/runtime
mkdir -p $R/protocol $T/protocol
```

- [ ] **Step 2: Move source files**

```bash
R=src/main/scala/sicfun/holdem/runtime
git mv $R/AcpcMatchRunner.scala $R/protocol/
git mv $R/AcpcHeadsUpDealer.scala $R/protocol/
git mv $R/SlumbotMatchRunner.scala $R/protocol/
git mv $R/MatchRunnerSupport.scala $R/protocol/
```

- [ ] **Step 3: Update package declarations**

Files in `protocol/`: `package sicfun.holdem.runtime` → `package sicfun.holdem.runtime.protocol`

- [ ] **Step 4: Move test files**

```bash
T=src/test/scala/sicfun/holdem/runtime
git mv $T/AcpcMatchRunnerTest.scala $T/protocol/
git mv $T/SlumbotActionCodecTest.scala $T/protocol/
git mv $T/MatchRunnerSupportTest.scala $T/protocol/
```

- [ ] **Step 5: Update package declarations in test files**

- [ ] **Step 6: Fix imports across the codebase**

```bash
grep -rn 'import sicfun\.holdem\.runtime\.' src/main/scala src/test/scala --include='*.scala' | grep -v '/runtime/'
```

Likely affected: `holdem/bench/OpponentMemoryBatchingBenchmark.scala`, scripts that reference class names.

- [ ] **Step 7: Compile gate**

```bash
sbt compile
```

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: extract runtime/protocol/ sub-package for match runners"
```

---

### Task 7: Restructure `holdem/bench/`

**Source base**: `src/main/scala/sicfun/holdem/bench/`
**Test base**: `src/test/scala/sicfun/holdem/bench/`

| New sub-package | Source files | Test files |
|---|---|---|
| `tuner/` | HeadsUpBackendAutoTuner, HeadsUpRangeGpuAutoTuner, HoldemPostflopGpuAutoTuner, GlobalGpuTuningTool | (none) |
| `gate/` | HeadsUpGpuExactParityGate, HeadsUpGpuPocGate, HeadsUpGpuSmokeGate | HeadsUpGpuExactParityGateTest |

- [ ] **Step 1: Create sub-directories**

```bash
B=src/main/scala/sicfun/holdem/bench
T=src/test/scala/sicfun/holdem/bench
mkdir -p $B/tuner $B/gate $T/gate
```

- [ ] **Step 2: Move source files**

```bash
B=src/main/scala/sicfun/holdem/bench
# tuner/
git mv $B/HeadsUpBackendAutoTuner.scala $B/tuner/
git mv $B/HeadsUpRangeGpuAutoTuner.scala $B/tuner/
git mv $B/HoldemPostflopGpuAutoTuner.scala $B/tuner/
git mv $B/GlobalGpuTuningTool.scala $B/tuner/

# gate/
git mv $B/HeadsUpGpuExactParityGate.scala $B/gate/
git mv $B/HeadsUpGpuPocGate.scala $B/gate/
git mv $B/HeadsUpGpuSmokeGate.scala $B/gate/
```

- [ ] **Step 3: Update package declarations**

- Files in `tuner/`: `package sicfun.holdem.bench` → `package sicfun.holdem.bench.tuner`
- Files in `gate/`: `package sicfun.holdem.bench` → `package sicfun.holdem.bench.gate`

- [ ] **Step 4: Move test files**

```bash
T=src/test/scala/sicfun/holdem/bench
git mv $T/HeadsUpGpuExactParityGateTest.scala $T/gate/
```

- [ ] **Step 5: Update package declaration in moved test file**

- [ ] **Step 6: Fix imports across the codebase**

The bench package is mostly a leaf — few files import from it. Check:

```bash
grep -rn 'import sicfun\.holdem\.bench\.' src/main/scala src/test/scala --include='*.scala' | grep -v '/bench/'
```

Also check intra-bench imports: `GlobalGpuTuningTool` (now in `tuner/`) may reference gate classes or benchmarks at the bench root.

- [ ] **Step 7: Compile gate**

```bash
sbt compile
```

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: extract bench/tuner/ and bench/gate/ sub-packages"
```

---

### Task 8: Update references in project docs and memory

After all file moves are complete, update any references to old paths in:

- [ ] **Step 1: Update CLAUDE.md**

Search for package paths or file references that changed:
```bash
grep -n 'sicfun\.holdem\.\(strategic\|engine\|runtime\|bench\)' CLAUDE.md
```
Update any references.

- [ ] **Step 2: Update AGENTS.md**

```bash
grep -n 'sicfun\.holdem\.\(strategic\|engine\|runtime\|bench\)\|scripts/\|docs/' AGENTS.md
```

- [ ] **Step 3: Update AI_ENTRYPOINT.md and README.md**

Check for references to moved scripts or docs:
```bash
grep -n 'scripts/\|docs/' AI_ENTRYPOINT.md README.md
```

- [ ] **Step 4: Update memory files**

Check the memory directory for references to old package paths:
```bash
grep -rn 'sicfun\.holdem\.\(strategic\|engine\|runtime\|bench\)' ~/.claude/projects/*/memory/
```

Update the architecture memory file if it mentions the old package structure.

- [ ] **Step 5: Update docs that reference moved scripts**

Check operator runbook and deployment docs for script path references:
```bash
grep -rn 'scripts/' docs/
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "docs: update references to reflect new package and script paths"
```

---

### Task 9: Full compile verification

- [ ] **Step 1: Clean compile**

```bash
sbt clean compile
```

This catches any import issues that incremental compilation might have missed.

- [ ] **Step 2: Fix any remaining issues**

If compilation fails, the errors will point to specific files with broken imports. Fix them and re-compile until clean.

- [ ] **Step 3: Commit fixes if any**

```bash
git add -A
git commit -m "fix: resolve remaining import issues from restructure"
```

---

### Task 10: Full test verification

- [ ] **Step 1: Run full test suite**

```bash
sbt test
```

- [ ] **Step 2: Investigate any failures**

Since this is a pure file-move refactoring, test failures should only come from:
- Broken imports (should be caught by Task 9)
- Hardcoded package names in test strings or reflection
- Resource paths that changed

Fix any issues found.

- [ ] **Step 3: Commit fixes if any**

```bash
git add -A
git commit -m "fix: resolve test failures from restructure"
```

- [ ] **Step 4: Final commit summary**

Verify the full restructure is clean:
```bash
sbt clean compile test
```
