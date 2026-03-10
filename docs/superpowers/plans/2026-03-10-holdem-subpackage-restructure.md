# Holdem Sub-Package Restructure

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the flat 84-file `sicfun.holdem` package into 13 sub-packages with clean layered dependencies, zero logic changes, zero performance impact.

**Architecture:** Files are moved into sub-packages (`sicfun.holdem.types`, `.equity`, `.gpu`, etc.) in dependency order. Each moved file gets its package declaration updated and gains explicit imports for cross-subpackage references (currently implicit via same-package visibility). The compiler (`-Werror` + `-Wunused:imports`) enforces correctness.

**Tech Stack:** Scala 3.8.1, SBT, munit 1.2.2

---

## Constraints

- **Zero logic changes** — only package declarations, imports, and file locations change
- **All `inline` annotations preserved exactly** — no performance impact
- **Compile after each layer** — catch import issues early
- **Update build.sbt** — 4 FQCNs reference `sicfun.holdem.ClassName` directly

## Dependency Hierarchy

```
Layer 0:  types (foundation — no holdem deps)
Layer 0:  io (no holdem deps, only core + stdlib)
Layer 1:  model (depends on types)
Layer 1:  gpu (depends on types at most)
Layer 2:  equity (depends on types, gpu)
Layer 2:  provider (depends on types, model, gpu)
Layer 2:  cfr (depends on types, equity, gpu)
Layer 3:  engine (depends on types, model, equity, provider, cfr)
Layer 4:  runtime (depends on types, model, provider, cfr, engine, equity, io, gpu)
Layer 5:  analysis, tablegen, bench, cli (leaf nodes)
```

## File-to-Package Mapping

### types (5 source + 7 test files)
Source files:
- `PokerAction.scala`
- `HoldemTypes.scala` (HoleCards, Board, Street, Position, BetAction)
- `GameState.scala`
- `PokerEvent.scala`
- `HandEngine.scala`

Test files:
- `PokerActionTest.scala`
- `HandEngineTest.scala`
- `BehavioralMetricsTest.scala`
- `TestSystemPropertyScope.scala` (shared test utility)

### io (7 source + 2 test files)
Source files:
- `ActionDataset.scala`
- `DecisionLoopEventFeedIO.scala`
- `HandStateSnapshotIO.scala`
- `HoldemDdreArtifactIO.scala`
- `HoldemDdreDatasetIO.scala`
- `SignalAudit.scala`
- `SignalAuditLogIO.scala`

Test files:
- `ActionDatasetContractTest.scala`
- `DecisionLoopEventFeedIOTest.scala`
- `SignalAuditTest.scala`

### model (6 source + 4 test files)
Source files:
- `PokerFeatures.scala`
- `PokerActionModel.scala`
- `PokerActionModelArtifactIO.scala`
- `PokerActionTrainingDataIO.scala`
- `TrainPokerActionModel.scala`
- `FeatureExtractor.scala`

Test files:
- `PokerActionTrainingDataIOTest.scala`
- `ModelArtifactIOTest.scala`
- `ModelLifecycleTest.scala`
- `TrainPokerActionModelCliTest.scala`

### gpu (7 source + 5 test files)
Source files:
- `GpuRuntimeSupport.scala`
- `HeadsUpGpuRuntime.scala`
- `HeadsUpHybridDispatcher.scala`
- `HeadsUpRangeGpuRuntime.scala`
- `HoldemPostflopNativeRuntime.scala`
- `HoldemBayesNativeRuntime.scala`
- `HoldemDdreNativeRuntime.scala`

Test files:
- `HeadsUpGpuRuntimeTest.scala`
- `HeadsUpHybridDispatcherPlanningTest.scala`
- `HeadsUpOpenCLDispatchTest.scala`
- `HeadsUpRangeGpuRuntimeTest.scala`
- `HoldemPostflopNativeParityTest.scala`

### equity (8 source + 5 test files)
Source files:
- `HoldemEquity.scala`
- `HeadsUpEquityTable.scala`
- `HeadsUpEquityCanonicalTable.scala`
- `HeadsUpEquityTableIOUtil.scala`
- `HeadsUpEquityTableFormat.scala`
- `HeadsUpTableInfo.scala`
- `HoldemCombinator.scala`
- `RangeParser.scala`
- `BunchingEffect.scala`
- `TableFormat.scala`

Test files:
- `HoldemEquityTest.scala`
- `HeadsUpEquityTableTest.scala`
- `HeadsUpEquityCanonicalTableTest.scala`
- `CanonicalExactTableTestFixture.scala`
- `EquityAccuracyTest.scala`
- `MonteCarloConvergenceTest.scala`
- `RangeParserTest.scala`
- `BunchingEffectTest.scala`

### provider (4 source + 3 test files)
Source files:
- `HoldemBayesProvider.scala`
- `HoldemDdreProvider.scala`
- `HoldemDdreOnnxRuntime.scala`
- `HoldemDdreOfflineGate.scala`

Test files:
- `HoldemBayesProviderTest.scala`
- `HoldemDdreOfflineGateTest.scala`
- `HoldemDdreIntegrationTest.scala`

### cfr (4 source + 2 test files)
Source files:
- `CfrSolver.scala`
- `HoldemCfrSolver.scala`
- `HoldemCfrReport.scala`
- `HoldemCfrNativeRuntime.scala`

Test files:
- `CfrSolverTest.scala`
- `HoldemCfrSolverTest.scala`
- `HoldemCfrReportTest.scala`

### engine (3 source + 2 test files)
Source files:
- `RangeInferenceEngine.scala`
- `RealTimeAdaptiveEngine.scala`
- `VillainResponseModel.scala`

Test files:
- `RangeInferenceEngineTest.scala`
- `RealTimeAdaptiveEngineTest.scala`

### runtime (5 source + 5 test files)
Source files:
- `TexasHoldemPlayingHall.scala`
- `AlwaysOnDecisionLoop.scala`
- `LiveHandSimulator.scala`
- `AdvisorSession.scala`
- `PokerAdvisor.scala`
- `HandHistoryAnalyzer.scala`

Test files:
- `TexasHoldemPlayingHallTest.scala`
- `AlwaysOnDecisionLoopTest.scala`
- `LiveHandSimulatorTest.scala`
- `AdvisorSessionTest.scala`
- `HandHistoryAnalyzerTest.scala`

### analysis (6 source + 2 test files)
Source files:
- `EvAnalysis.scala`
- `LongitudinalAnalysis.scala`
- `PlayerCluster.scala`
- `PlayerSignature.scala`
- `GenerateSignals.scala`
- `ComparePublishedPreflopVsRandom.scala`
- `ComparePublishedSpecificMatchups.scala`
- `CompareSingleMatchupExact.scala`

Test files:
- `LongitudinalAnalysisTest.scala`
- `PlayerClusterTest.scala`
- `GenerateSignalsCliTest.scala`

### tablegen (7 source + 1 test files)
Source files:
- `GenerateHeadsUpTable.scala`
- `GenerateHeadsUpCanonicalTable.scala`
- `HeadsUpCanonicalExactTuner.scala`
- `HeadsUpCanonicalExactBoardMajorTuner.scala`
- `HeadsUpCanonicalTableReadableDump.scala`
- `InspectCanonicalBatch.scala`
- `DeviceProofRun.scala`

Test files:
- `HeadsUpGpuExactParityGateTest.scala`

### bench (12 source + 2 test files)
Source files:
- `HeadsUpBackendAutoTuner.scala`
- `HeadsUpBackendComparison.scala`
- `HeadsUpRangeGpuAutoTuner.scala`
- `HeadsUpRangeGpuBenchmark.scala`
- `HeadsUpTableGenerationBenchmark.scala`
- `HoldemBayesBenchmark.scala`
- `HoldemPostflopNativeBenchmark.scala`
- `HoldemDdreParityBenchmark.scala`
- `HoldemPostflopGpuAutoTuner.scala`
- `ModeComparisonBenchmark.scala`
- `HybridBenchmark.scala`
- `HeadsUpGpuPocGate.scala`
- `HeadsUpGpuSmokeGate.scala`
- `HeadsUpGpuExactParityGate.scala`

Test files:
- `OperationalRegressionSuiteTest.scala`
- `PokerPipelineTest.scala`

### cli (3 source + 2 test files)
Source files:
- `CliHelpers.scala`
- `AdvisorCommandParser.scala`
- `BatchTrainer.scala`

Test files:
- `CliHelpersTest.scala`
- `BatchTrainerTest.scala`

---

## Execution Tasks

### Task 1: Create directory structure

- [ ] **Step 1:** Create all 13 sub-package directories under `src/main/scala/sicfun/holdem/`

```bash
for pkg in types io model gpu equity provider cfr engine runtime analysis tablegen bench cli; do
  mkdir -p src/main/scala/sicfun/holdem/$pkg
  mkdir -p src/test/scala/sicfun/holdem/$pkg
done
```

- [ ] **Step 2:** Verify directories exist

### Task 2: Move `types` package (Layer 0)

This is the foundation — all other packages depend on these types.

**For each source file** (PokerAction.scala, HoldemTypes.scala, GameState.scala, PokerEvent.scala, HandEngine.scala):
- [ ] **Step 1:** Change `package sicfun.holdem` → `package sicfun.holdem.types`
- [ ] **Step 2:** Add any needed imports (these files depend only on `sicfun.core`, so no cross-holdem imports needed)
- [ ] **Step 3:** Move file to `src/main/scala/sicfun/holdem/types/`

**For each test file** (PokerActionTest.scala, HandEngineTest.scala, BehavioralMetricsTest.scala, TestSystemPropertyScope.scala):
- [ ] **Step 4:** Change `package sicfun.holdem` → `package sicfun.holdem.types`
- [ ] **Step 5:** Move file to `src/test/scala/sicfun/holdem/types/`

**Do NOT compile yet** — other files still reference these types without imports.

### Task 3: Move `io` package (Layer 0)

**For each source file** (ActionDataset.scala, DecisionLoopEventFeedIO.scala, HandStateSnapshotIO.scala, HoldemDdreArtifactIO.scala, HoldemDdreDatasetIO.scala, SignalAudit.scala, SignalAuditLogIO.scala):
- [ ] **Step 1:** Change `package sicfun.holdem` → `package sicfun.holdem.io`
- [ ] **Step 2:** Add `import sicfun.holdem.types.*` (for PokerAction, GameState, HoleCards, Board, Street, Position, PokerEvent, HandState etc.)
- [ ] **Step 3:** Move file to `src/main/scala/sicfun/holdem/io/`

**For each test file:**
- [ ] **Step 4:** Update package + imports, move to `src/test/scala/sicfun/holdem/io/`

### Task 4: Move `model` package (Layer 1)

**For each source file** (PokerFeatures.scala, PokerActionModel.scala, PokerActionModelArtifactIO.scala, PokerActionTrainingDataIO.scala, TrainPokerActionModel.scala, FeatureExtractor.scala):
- [ ] **Step 1:** Change `package sicfun.holdem` → `package sicfun.holdem.model`
- [ ] **Step 2:** Add `import sicfun.holdem.types.*`
- [ ] **Step 3:** Move file to `src/main/scala/sicfun/holdem/model/`

**For each test file:**
- [ ] **Step 4:** Update package + imports, move to `src/test/scala/sicfun/holdem/model/`

### Task 5: Move `gpu` package (Layer 1)

**For each source file** (GpuRuntimeSupport.scala, HeadsUpGpuRuntime.scala, HeadsUpHybridDispatcher.scala, HeadsUpRangeGpuRuntime.scala, HoldemPostflopNativeRuntime.scala, HoldemBayesNativeRuntime.scala, HoldemDdreNativeRuntime.scala):
- [ ] **Step 1:** Change `package sicfun.holdem` → `package sicfun.holdem.gpu`
- [ ] **Step 2:** Add `import sicfun.holdem.types.*` where types are referenced
- [ ] **Step 3:** Move file to `src/main/scala/sicfun/holdem/gpu/`

**For each test file:**
- [ ] **Step 4:** Update package + imports, move to `src/test/scala/sicfun/holdem/gpu/`

### Task 6: Move `equity` package (Layer 2)

**For each source file** (HoldemEquity.scala, HeadsUpEquityTable.scala, HeadsUpEquityCanonicalTable.scala, HeadsUpEquityTableIOUtil.scala, HeadsUpEquityTableFormat.scala, HeadsUpTableInfo.scala, HoldemCombinator.scala, RangeParser.scala, BunchingEffect.scala, TableFormat.scala):
- [ ] **Step 1:** Change `package sicfun.holdem` → `package sicfun.holdem.equity`
- [ ] **Step 2:** Add `import sicfun.holdem.types.*` and `import sicfun.holdem.gpu.*` where needed
- [ ] **Step 3:** Move file to `src/main/scala/sicfun/holdem/equity/`

**For each test file:**
- [ ] **Step 4:** Update package + imports, move to `src/test/scala/sicfun/holdem/equity/`

### Task 7: Move `provider` package (Layer 2)

**For each source file** (HoldemBayesProvider.scala, HoldemDdreProvider.scala, HoldemDdreOnnxRuntime.scala, HoldemDdreOfflineGate.scala):
- [ ] **Step 1:** Change `package sicfun.holdem` → `package sicfun.holdem.provider`
- [ ] **Step 2:** Add imports: `import sicfun.holdem.types.*`, `import sicfun.holdem.model.*`, `import sicfun.holdem.gpu.*` as needed
- [ ] **Step 3:** Move file to `src/main/scala/sicfun/holdem/provider/`

**For each test file:**
- [ ] **Step 4:** Update package + imports, move to `src/test/scala/sicfun/holdem/provider/`

### Task 8: Move `cfr` package (Layer 2)

**For each source file** (CfrSolver.scala, HoldemCfrSolver.scala, HoldemCfrReport.scala, HoldemCfrNativeRuntime.scala):
- [ ] **Step 1:** Change `package sicfun.holdem` → `package sicfun.holdem.cfr`
- [ ] **Step 2:** Add imports: `import sicfun.holdem.types.*`, `import sicfun.holdem.equity.*`, `import sicfun.holdem.gpu.*` as needed
- [ ] **Step 3:** Move file to `src/main/scala/sicfun/holdem/cfr/`

**For each test file:**
- [ ] **Step 4:** Update package + imports, move to `src/test/scala/sicfun/holdem/cfr/`

### Task 9: Move `engine` package (Layer 3)

**For each source file** (RangeInferenceEngine.scala, RealTimeAdaptiveEngine.scala, VillainResponseModel.scala):
- [ ] **Step 1:** Change `package sicfun.holdem` → `package sicfun.holdem.engine`
- [ ] **Step 2:** Add imports: `types.*`, `model.*`, `equity.*`, `provider.*`, `cfr.*` as needed per file
- [ ] **Step 3:** Move file to `src/main/scala/sicfun/holdem/engine/`

**For each test file:**
- [ ] **Step 4:** Update package + imports, move to `src/test/scala/sicfun/holdem/engine/`

### Task 10: Move `runtime` package (Layer 4)

**For each source file** (TexasHoldemPlayingHall.scala, AlwaysOnDecisionLoop.scala, LiveHandSimulator.scala, AdvisorSession.scala, PokerAdvisor.scala, HandHistoryAnalyzer.scala):
- [ ] **Step 1:** Change `package sicfun.holdem` → `package sicfun.holdem.runtime`
- [ ] **Step 2:** Add imports for all used sub-packages (types, model, provider, cfr, engine, equity, io, gpu)
- [ ] **Step 3:** Move file to `src/main/scala/sicfun/holdem/runtime/`

**For each test file:**
- [ ] **Step 4:** Update package + imports, move to `src/test/scala/sicfun/holdem/runtime/`

### Task 11: Move `analysis` package (Layer 5)

**For each source file** (EvAnalysis.scala, LongitudinalAnalysis.scala, PlayerCluster.scala, PlayerSignature.scala, GenerateSignals.scala, ComparePublished*.scala, CompareSingleMatchupExact.scala):
- [ ] **Step 1:** Change `package sicfun.holdem` → `package sicfun.holdem.analysis`
- [ ] **Step 2:** Add required imports
- [ ] **Step 3:** Move file to `src/main/scala/sicfun/holdem/analysis/`

**For each test file:**
- [ ] **Step 4:** Update package + imports, move to `src/test/scala/sicfun/holdem/analysis/`

### Task 12: Move `tablegen` package (Layer 5)

**For each source file** (GenerateHeadsUpTable.scala, GenerateHeadsUpCanonicalTable.scala, HeadsUpCanonical*Tuner.scala, HeadsUpCanonicalTableReadableDump.scala, InspectCanonicalBatch.scala, DeviceProofRun.scala):
- [ ] **Step 1:** Change `package sicfun.holdem` → `package sicfun.holdem.tablegen`
- [ ] **Step 2:** Add required imports
- [ ] **Step 3:** Move file to `src/main/scala/sicfun/holdem/tablegen/`

**For each test file:**
- [ ] **Step 4:** Update package + imports, move to `src/test/scala/sicfun/holdem/tablegen/`

### Task 13: Move `bench` package (Layer 5)

**For each source file** (all auto-tuners, benchmarks, gates, comparisons):
- [ ] **Step 1:** Change `package sicfun.holdem` → `package sicfun.holdem.bench`
- [ ] **Step 2:** Add required imports
- [ ] **Step 3:** Move file to `src/main/scala/sicfun/holdem/bench/`

**For each test file:**
- [ ] **Step 4:** Update package + imports, move to `src/test/scala/sicfun/holdem/bench/`

### Task 14: Move `cli` package (Layer 5)

**For each source file** (CliHelpers.scala, AdvisorCommandParser.scala, BatchTrainer.scala):
- [ ] **Step 1:** Change `package sicfun.holdem` → `package sicfun.holdem.cli`
- [ ] **Step 2:** Add required imports
- [ ] **Step 3:** Move file to `src/main/scala/sicfun/holdem/cli/`

**For each test file:**
- [ ] **Step 4:** Update package + imports, move to `src/test/scala/sicfun/holdem/cli/`

### Task 15: Update build.sbt FQCNs

- [ ] **Step 1:** Update these 4 fully-qualified class name references:
  - `sicfun.holdem.GenerateHeadsUpTable` → `sicfun.holdem.tablegen.GenerateHeadsUpTable`
  - `sicfun.holdem.GenerateHeadsUpCanonicalTable` → `sicfun.holdem.tablegen.GenerateHeadsUpCanonicalTable`
  - `sicfun.holdem.HeadsUpGpuSmokeGate` → `sicfun.holdem.bench.HeadsUpGpuSmokeGate`
  - `sicfun.holdem.HeadsUpGpuExactParityGate` → `sicfun.holdem.bench.HeadsUpGpuExactParityGate`

### Task 16: Compile and fix

- [ ] **Step 1:** Run `sbt compile` — fix any missing imports flagged by `-Werror`
- [ ] **Step 2:** Run `sbt test:compile` — fix any test import issues
- [ ] **Step 3:** Run `sbt test` — verify all tests pass
- [ ] **Step 4:** Commit

---

## Per-File Procedure (for subagents)

For each file assigned to a subagent:

1. **Read the file** — note all types/objects/classes referenced that come from other holdem files
2. **Change the package declaration** — `package sicfun.holdem` → `package sicfun.holdem.<subpkg>`
3. **Add imports** — for each cross-subpackage reference, add `import sicfun.holdem.<other>.*` or a specific import. Use wildcard imports per sub-package to keep it simple.
4. **Move the file** — `git mv` from `holdem/` to `holdem/<subpkg>/`
5. **Repeat for the corresponding test file** if it exists

## Notes

- `TestSystemPropertyScope.scala` is a shared test utility used by multiple test files. Move it to `types` (where it currently lives conceptually) and import from there.
- Some test files (OperationalRegressionSuiteTest, PokerPipelineTest) are integration tests that span multiple sub-packages — they'll need imports from several sub-packages.
- Files with `@main` or `object ... { def main }` entry points may be referenced from scripts or docs — search for FQCN references beyond build.sbt.
