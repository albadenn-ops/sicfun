# AI Context Archive

This file is an append-only operational memory layer for future AI sessions.
Use it when commit history or local git context is too large/noisy for limited context windows.
It is internal working memory, not a release note, product spec, or reviewer-facing source of truth.
For externally readable project status, start with `README.md`, `ROADMAP.md`, and `docs/OPERATOR_RUNBOOK.md`.

## Current Status
- The project has matured past roadmap-driven milestones into a multi-surface platform:
  - offline training and calibration (action models, equity tables)
  - real-time inference/decision components (Bayesian range inference, adaptive engine)
  - CFR/GTO equilibrium solver with native CPU/CUDA providers
  - Bayesian native providers with optional shadow parity validation
  - playing hall simulator for large-volume self-play with periodic retraining
  - interactive advisor session (PokerAdvisor/AdvisorSession CLI)
  - always-on event loop runtime with optional scheduled retraining
  - simulator and proof scripts
- Treat the codebase as a platform with explicit runtime pipelines, not a single monolithic experiment.

## Runtime Pipelines
1. Training pipeline:
`TrainPokerActionModel` -> model artifact directory (`metadata.properties`, `weights.tsv`, `bias.tsv`, `category-index.tsv`)

2. Inference/decision pipeline:
event feed -> `HandEngine` state -> posterior inference (`RangeInferenceEngine`) -> action ranking -> decision log + signals

3. Continuous loop:
`AlwaysOnDecisionLoop` polls feed TSV, emits decisions, snapshots each hand, appends context, and can retrain/reload models on schedule.

4. Proof pipeline:
`scripts/prove-pipeline.ps1` runs key suites to verify end-to-end behavior.

5. CFR/equilibrium pipeline:
`HoldemCfrSolver` -> strategy profile + exploitability report; native CPU/CUDA providers with auto-benchmarking selection.

6. Playing hall:
`TexasHoldemPlayingHall` -> multi-street self-play with Bayesian inference on all streets, periodic model retraining, exportable hand/training logs.

7. Advisor:
`PokerAdvisor`/`AdvisorSession` -> interactive CLI for live decision advice with equilibrium baseline blending.

## Operator Commands
- Primary runbook:
`docs/OPERATOR_RUNBOOK.md`

- Quick proof:
`powershell -ExecutionPolicy Bypass -File scripts/prove-pipeline.ps1 -Quick`

- Full proof:
`powershell -ExecutionPolicy Bypass -File scripts/prove-pipeline.ps1`

- Simulator:
`sbt "runMain sicfun.holdem.LiveHandSimulator --help"`

- Always-on loop:
`sbt "runMain sicfun.holdem.AlwaysOnDecisionLoop --help"`

- Playing hall:
`sbt "runMain sicfun.holdem.TexasHoldemPlayingHall --help"`

- CFR report:
`sbt "runMain sicfun.holdem.HoldemCfrReport --help"`

- Bayesian benchmark:
`sbt "runMain sicfun.holdem.HoldemBayesBenchmark --help"`

- Advisor:
`sbt "runMain sicfun.holdem.PokerAdvisor --help"`

## Entry Template
When appending manually, use:

```
## <UTC ISO timestamp> - <title>
Summary: <1-2 sentence summary>
Why: <why this change exists>
Files: <comma-separated key files>
Validation: <tests/commands run>
Risks: <known gaps or follow-ups>
```

## 2026-03-04T00:00:00Z - Baseline archive created
Summary: Introduced persistent AI context archive and formalized runtime pipeline map.
Why: Git history alone is insufficient for bounded-context AI handoffs as the project scale increased.
Files: docs/AI_CONTEXT_ARCHIVE.md
Validation: N/A (documentation only)
Risks: Must be kept append-only and actively maintained after major architecture changes.

## 2026-03-11T22:32:23Z - Shared AI sidecar entrypoint and role overlays
Summary: Added a shared `AI_ENTRYPOINT.md` contract for all delegated AI workers, converted `GEMINI.md`/`CLAUDE.md`/`GPT.md` into role overlays, and wired the prompt builders to inject the shared contract plus the provider overlay into delegated runs.
Why: The sidecar role files had drifted and some delegated paths were not actually receiving the repo-specific contract, which made the role split more documentary than real.
Files: AI_ENTRYPOINT.md,GEMINI.md,CLAUDE.md,GPT.md,scripts/ai-minion.ps1,scripts/gemini-sidecar.ps1,scripts/gemini_cli.py,AGENTS.md,docs/AI_MINIONS.md,docs/GEMINI_MINION.md,docs/OPERATOR_RUNBOOK.md,docs/AI_CONTEXT_ARCHIVE.md
Validation: powershell -ExecutionPolicy Bypass -File scripts/ai-minion.ps1 -Action doctor -Provider gpt; powershell -ExecutionPolicy Bypass -File scripts/gemini-sidecar.ps1 -Action doctor; powershell -ExecutionPolicy Bypass -File scripts/ai-minion.ps1 -Action delegate -Provider gpt -Mode analysis -Task "Sanity-check prompt assembly only." -ContextPath AI_ENTRYPOINT.md,GPT.md -OutputPath .tool-cache/ai-minions/gpt/sanity.txt -WhatIf; powershell -ExecutionPolicy Bypass -File scripts/gemini-sidecar.ps1 -Action delegate -Mode analysis -Task "Sanity-check prompt assembly only." -ContextPath AI_ENTRYPOINT.md,GEMINI.md -OutputPath .tool-cache/gemini-sidecar/sanity.txt -WhatIf
Risks: Claude delegated runs are currently flaky on this machine, and Gemini prompt assembly still lives in two wrappers (`scripts/ai-minion.ps1` for injected-mode Gemini and `scripts/gemini-sidecar.ps1` for forwarded runs), which leaves some maintenance duplication.

## 2026-03-04T03:24:52Z - Always-on loop + context continuity
Summary: Added continuous decision loop runtime, scheduled retraining hook, and context archive tooling.
Why: Project outgrew roadmap-only tracking and needed runtime + memory continuity for future AI sessions.
Files: src/main/scala/sicfun/holdem/AlwaysOnDecisionLoop.scala,src/main/scala/sicfun/holdem/DecisionLoopEventFeedIO.scala,src/test/scala/sicfun/holdem/AlwaysOnDecisionLoopTest.scala,docs/AI_CONTEXT_ARCHIVE.md,scripts/archive-context.ps1,ROADMAP.md
Validation: sbt testOnly sicfun.holdem.AlwaysOnDecisionLoopTest; powershell -ExecutionPolicy Bypass -File scripts/prove-pipeline.ps1 -Quick
Risks: No GTO equilibrium solver yet; runtime still assumes configured hero/villain identities and candidate action set.
GitHead: ad8ff8d

## 2026-03-04T03:45:26Z - Playing hall simulation and online retraining
Summary: Added TexasHoldemPlayingHall for large-volume self-play, periodic model retraining, and exportable hand/training logs.
Why: You requested visible play+learning behavior at scale (including millions of simulated hands) and durable data generation.
Files: src/main/scala/sicfun/holdem/TexasHoldemPlayingHall.scala,src/test/scala/sicfun/holdem/TexasHoldemPlayingHallTest.scala,scripts/run-playing-hall.ps1,scripts/prove-pipeline.ps1,ROADMAP.md
Validation: sbt testOnly sicfun.holdem.TexasHoldemPlayingHallTest; powershell -ExecutionPolicy Bypass -File scripts/prove-pipeline.ps1 -Quick; sbt runMain sicfun.holdem.TexasHoldemPlayingHall --hands=300 --reportEvery=100 --learnEveryHands=100 --heroExplorationRate=0.20
Risks: Current hall is preflop-focused with simplified response tree; full multi-street policy learning remains future work.
GitHead: ad8ff8d

## 2026-03-04T04:13:47Z - Playing hall multi-street restore
Summary: Restored TexasHoldemPlayingHall after accidental deletion and upgraded it to simulate preflop/flop/turn/river with periodic retraining and postflop data emission.
Why: Needed a visible play+learn pipeline and durable context for future AI sessions after roadmap drift.
Files: src/main/scala/sicfun/holdem/TexasHoldemPlayingHall.scala,src/test/scala/sicfun/holdem/TexasHoldemPlayingHallTest.scala,data/playing-hall-smoke3/*
Validation: sbt testOnly sicfun.holdem.TexasHoldemPlayingHallTest; powershell -ExecutionPolicy Bypass -File scripts/prove-pipeline.ps1 -Quick; sbt runMain sicfun.holdem.TexasHoldemPlayingHall --hands=200 --learnEveryHands=50
Risks: Hero uses full Bayesian engine preflop and heuristic postflop for throughput; full per-street Bayesian solve is still computationally heavier for million-hand runs.
GitHead: ad8ff8d

## 2026-03-04T04:24:52Z - Playing hall full Bayesian all streets
Summary: Hero decisions now use Bayesian range inference on preflop, flop, turn, and river. Added a dedicated lighter postflop Bayesian engine to preserve throughput while keeping inference-based decisions.
Why: You requested the model visibly play and learn with real decision inference beyond preflop-only policy shortcuts.
Files: src/main/scala/sicfun/holdem/TexasHoldemPlayingHall.scala,src/test/scala/sicfun/holdem/TexasHoldemPlayingHallTest.scala,data/playing-hall-smoke4/*
Validation: sbt testOnly sicfun.holdem.TexasHoldemPlayingHallTest; powershell -ExecutionPolicy Bypass -File scripts/prove-pipeline.ps1 -Quick; sbt runMain sicfun.holdem.TexasHoldemPlayingHall --hands=200 --learnEveryHands=50 --outDir=data/playing-hall-smoke4
Risks: Postflop Bayesian engine runs with aggressively reduced Monte Carlo budgets (bunching=1, low equity trials) to keep runtime practical; this trades precision for speed.
GitHead: ad8ff8d

## 2026-03-05T02:30:00Z - CFR module integration and tracking runtime
Summary: Added a reusable CFR core plus Hold'em abstraction solver, wired equilibrium baseline blending into real-time decision engines, and added offline CFR reporting/tracking CLI.
Why: Needed a first-class exploitability-aware baseline that reuses native equity engines and can run both live and offline in the existing project architecture.
Files: src/main/scala/sicfun/holdem/CfrSolver.scala,src/main/scala/sicfun/holdem/HoldemCfrSolver.scala,src/main/scala/sicfun/holdem/RealTimeAdaptiveEngine.scala,src/main/scala/sicfun/holdem/PokerAdvisor.scala,src/main/scala/sicfun/holdem/AlwaysOnDecisionLoop.scala,src/main/scala/sicfun/holdem/HoldemCfrReport.scala
Validation: sbt "testOnly sicfun.holdem.HoldemCfrReportTest sicfun.holdem.AlwaysOnDecisionLoopTest sicfun.holdem.CfrSolverTest sicfun.holdem.HoldemCfrSolverTest sicfun.holdem.RealTimeAdaptiveEngineTest"; sbt "testOnly sicfun.holdem.AdvisorSessionTest"
Risks: Current exploitability metric is local to the one-street abstraction (root-deviation + villain best-response gap), not a full-game exploitability measure.

## 2026-03-05T03:00:00Z - Native CFR providers integrated (CPU + CUDA build) with auto selection
Summary: Added native CFR JNI providers for CPU and CUDA builds, integrated provider selection/auto-benchmarking into HoldemCfrSolver, fixed a native recursion buffer bug in CFR core, and added provider telemetry to CFR report outputs.
Why: Needed native-engine parity with existing CUDA/CPU/OpenCL strategy and a practical path to enforce native CFR when it materially outperforms Scala CFR.
Files: src/main/scala/sicfun/holdem/HoldemCfrSolver.scala,src/main/scala/sicfun/holdem/HoldemCfrNativeRuntime.scala,src/main/java/sicfun/holdem/HoldemCfrNativeCpuBindings.java,src/main/java/sicfun/holdem/HoldemCfrNativeGpuBindings.java,src/main/native/jni/CfrNativeSolverCore.hpp,src/main/native/jni/HoldemCfrNativeCpuBindings.cpp,src/main/native/jni/HoldemCfrNativeGpuBindings.cu,src/main/native/build-windows-llvm.ps1,src/main/native/build-windows-cuda11.ps1,src/main/scala/sicfun/holdem/HoldemCfrReport.scala,src/test/scala/sicfun/holdem/HoldemCfrSolverTest.scala,src/test/scala/sicfun/holdem/HoldemCfrReportTest.scala,src/main/native/README.md
Validation: powershell -ExecutionPolicy Bypass -File src/main/native/build-windows-llvm.ps1; powershell -ExecutionPolicy Bypass -File src/main/native/build-windows-cuda11.ps1; sbt compile; sbt "testOnly sicfun.holdem.HoldemCfrSolverTest sicfun.holdem.HoldemCfrReportTest sicfun.holdem.RealTimeAdaptiveEngineTest sicfun.holdem.AlwaysOnDecisionLoopTest sicfun.holdem.CfrSolverTest"; JAVA_TOOL_OPTIONS=-Dsicfun.cfr.provider=native-cpu sbt "testOnly sicfun.holdem.HoldemCfrSolverTest -- --tests *premium*"; JAVA_TOOL_OPTIONS=-Dsicfun.cfr.provider=native-gpu sbt "testOnly sicfun.holdem.HoldemCfrSolverTest -- --tests *premium*"; JAVA_TOOL_OPTIONS=-Dsicfun.verbose=true -Dsicfun.cfr.provider=auto -Dsicfun.cfr.auto.benchmarkIterations=600 -Dsicfun.cfr.auto.nativeMinSpeedup=1.0 sbt "testOnly sicfun.holdem.HoldemCfrSolverTest -- --tests *premium*"
Risks: Current GPU provider path compiles/loads successfully but uses shared host-side CFR core (NVCC build), not a fully device-parallel CFR kernel; auto provider benchmark is synthetic and one-time per process, so per-spot performance can vary even though observed speedup on this machine was strongly in favor of native CPU (~32x vs Scala in the auto benchmark).

## 2026-03-05T03:15:00Z - Full audit of Codex work (entries #1-#5)
Agent: Claude Opus 4.6 (Claude Code CLI)
Summary: Audited all five archive entries produced by Codex. Verified file existence, code structure, archive claim accuracy, compilation, and full test suite.
Why: Establishing trust boundary between agents; user requested explicit verification of all prior work before continuing.
Audit scope:
  - Entry #1 (baseline archive): docs/AI_CONTEXT_ARCHIVE.md exists, template and pipeline map present. PASS.
  - Entry #2 (always-on loop): AlwaysOnDecisionLoop.scala, DecisionLoopEventFeedIO.scala, CliHelpers.scala, AlwaysOnDecisionLoopTest.scala, scripts/archive-context.ps1 all exist. Code confirms: poll-based event feed, scheduled retraining with configurable parameters, snapshot emission, CFR equilibrium baseline integration. PASS.
  - Entry #3 (TexasHoldemPlayingHall): TexasHoldemPlayingHall.scala, TexasHoldemPlayingHallTest.scala, scripts/run-playing-hall.ps1, scripts/prove-pipeline.ps1 all exist. Code confirms multi-street self-play (preflop/flop/turn/river via `playPostflopStreet(Street.Flop/Turn/River)` calls), periodic retraining, exportable logs. PASS.
  - Entry #4 (full Bayesian all streets): TexasHoldemPlayingHall uses RealTimeAdaptiveEngine for both preflop and postflop decisions (in `heroDecision` method: `if street == Street.Preflop then preflopEngine else postflopEngine`). Postflop engine constructed with reduced budgets via `postflopBunchingTrials`/`postflopEquityTrials` helper methods. Claim of "Bayesian range inference on all four streets" is ACCURATE. PASS.
  - Entry #5 (CFR module + native providers): All 12 files verified present - CfrSolver.scala (generic CFR core with `solve[S,A]`), HoldemCfrSolver.scala (Hold'em abstraction with auto/native-cpu/native-gpu/scala provider selection), HoldemCfrNativeRuntime.scala, HoldemCfrReport.scala, JNI bindings (2 Java + 2 native C++/CUDA + 1 header), native DLLs (sicfun_cfr_native.dll, sicfun_cfr_cuda.dll). RealTimeAdaptiveEngine.scala has `EquilibriumBaselineConfig` and `maybeBlendWithEquilibriumBaseline`. AlwaysOnDecisionLoop.scala and PokerAdvisor.scala both wire CFR CLI options (--cfrIterations, --cfrBlend, etc). PASS.
Compilation: sbt compile succeeds (84 Scala + 4 Java sources, 57s).
Test suite: sbt test passes ALL 308 tests, 0 failures (61s).
Files: docs/AI_CONTEXT_ARCHIVE.md
Validation: sbt compile; sbt test
Risks: None identified. All archive claims are accurate and verifiable. No uncommitted code discrepancies found.
GitHead: ad8ff8d


## 2026-03-05T03:18:15Z - Bayesian native throughput uplift + shadow parity safeguards
Agent: Codex (GPT-5)
Summary: Optimized the native Bayesian JNI/core path to reduce marshaling/allocation overhead and added optional Scala shadow validation so native can be preferred for speed without weakening the Scala reference implementation.
Why: Native Bayesian speedup looked underperforming; you requested better performance while keeping Scala intact and reliable.
Files: src/main/native/jni/BayesNativeUpdateCore.hpp,src/main/native/jni/HoldemBayesNativeCpuBindings.cpp,src/main/native/jni/HoldemBayesNativeGpuBindings.cu,src/main/scala/sicfun/holdem/HoldemBayesProvider.scala,src/test/scala/sicfun/holdem/HoldemBayesProviderTest.scala,src/main/native/README.md
Validation: sbt "runMain sicfun.holdem.HoldemBayesBenchmark --provider=scala --warmupRuns=2 --measureRuns=10 --bunchingTrials=1200 --equityTrials=8000 --seed=17"; sbt "runMain sicfun.holdem.HoldemBayesBenchmark --provider=native-cpu --warmupRuns=2 --measureRuns=10 --bunchingTrials=1200 --equityTrials=8000 --seed=17"; JAVA_TOOL_OPTIONS=-Dsicfun.verbose=true sbt "runMain sicfun.holdem.HoldemBayesBenchmark --provider=auto --warmupRuns=0 --measureRuns=1 --bunchingTrials=100 --equityTrials=100 --seed=17" (auto speedup moved from 4.48x baseline to 7.43x after optimization); powershell -ExecutionPolicy Bypass -File src/main/native/build-windows-llvm.ps1; powershell -ExecutionPolicy Bypass -File src/main/native/build-windows-cuda11.ps1; sbt compile; sbt "testOnly sicfun.holdem.HoldemBayesProviderTest".
Risks: Bayesian auto-provider benchmark is still synthetic and may not match every production spot; shadow mode adds compute cost because Scala reference is also executed when enabled; fail-closed is opt-in (default false); Bayesian GPU provider remains NVCC-compiled host core, not a fully device-parallel kernel.
Signature: /s/ Codex (GPT-5), 2026-03-05T03:18:15Z
GitHead: ad8ff8d

## 2026-03-05T09:11:46Z - Postflop native CPU/CUDA runtime in progress + validation baseline refreshed
Agent: Codex (GPT-5)
Summary: Added/updated dedicated postflop native bridge surfaces (CPU + CUDA), runtime selection/fallback/autotune handling, benchmark/tuner entrypoints, and parity tests; refreshed branch-health validation on the current dirty worktree.
Why: Keep the postflop acceleration path first-class alongside existing heads-up/CFR/Bayesian native runtimes and capture the true current validation state before further work.
Files: src/main/java/sicfun/holdem/HoldemPostflopNativeBindings.java,src/main/java/sicfun/holdem/HoldemPostflopNativeGpuBindings.java,src/main/native/jni/HoldemPostflopNativeBindings.cpp,src/main/native/jni/HoldemPostflopNativeBindingsCuda.cu,src/main/native/build-windows-llvm.ps1,src/main/native/build-windows-cuda11.ps1,src/main/scala/sicfun/holdem/HoldemPostflopNativeRuntime.scala,src/main/scala/sicfun/holdem/HoldemPostflopNativeBenchmark.scala,src/main/scala/sicfun/holdem/HoldemPostflopGpuAutoTuner.scala,src/main/scala/sicfun/holdem/HoldemEquity.scala,src/test/scala/sicfun/holdem/HoldemPostflopNativeParityTest.scala,src/main/native/README.md
Validation: sbt compile; sbt "testOnly sicfun.holdem.RangeInferenceEngineTest sicfun.holdem.RealTimeAdaptiveEngineTest sicfun.holdem.HoldemBayesProviderTest sicfun.holdem.HoldemPostflopNativeParityTest"; sbt test (observed 3 suite failures when run aggregated: HoldemBayesProviderTest, HeadsUpEquityCanonicalTableTest, TexasHoldemPlayingHallTest); sbt "testOnly sicfun.holdem.HoldemBayesProviderTest"; sbt "testOnly sicfun.holdem.HeadsUpEquityCanonicalTableTest"; sbt "testOnly sicfun.holdem.TexasHoldemPlayingHallTest"; sbt "runMain sicfun.holdem.HoldemBayesBenchmark --provider=scala --warmupRuns=1 --measureRuns=5 --bunchingTrials=800 --equityTrials=2000 --seed=17"; sbt "runMain sicfun.holdem.HoldemBayesBenchmark --provider=native-cpu --warmupRuns=1 --measureRuns=5 --bunchingTrials=800 --equityTrials=2000 --seed=17"; sbt "runMain sicfun.holdem.HoldemBayesBenchmark --provider=auto --warmupRuns=1 --measureRuns=5 --bunchingTrials=800 --equityTrials=2000 --seed=17"; sbt "runMain sicfun.holdem.HoldemBayesBenchmark --provider=native-gpu --warmupRuns=1 --measureRuns=3 --bunchingTrials=800 --equityTrials=2000 --seed=17"
Risks: Full-suite execution is currently order/timing sensitive on this machine despite individual passing reruns; do not treat a single `sbt test` failure as a definitive regression without isolating suites. Infer-posterior latency still heavily depends on bunching trial budget.
GitHead: d6b5866

## 2026-03-05T09:11:46Z - DDRE Phase 1 feasibility review + spec rewrite draft
Agent: Codex (GPT-5)
Summary: Performed implementation-feasibility review against `SICFUN_Phase1_DDRE_Spec.docx`, extracted full spec content, identified internal contradictions, and produced an implementation-ready v2 draft that now reflects cooperative Bayesian+DDRE fusion semantics.
Why: Engineering start was blocked by spec inconsistencies (model/output mismatch, blocker math contradiction, non-operational primary metric, and rollout mode conflict), and user requested spec-first correction before build-out.
Files: SICFUN_Phase1_DDRE_Spec.docx,docs/_phase1_spec_extracted.txt,docs/_phase1_spec_extracted_allp.txt,docs/SICFUN_Phase1_DDRE_Spec_v2.md
Validation: Manual architecture/code/spec cross-check against current inference/runtime surfaces (RangeInferenceEngine, HoldemBayesProvider, RealTimeAdaptiveEngine, HoldemPostflopNativeRuntime); no production code path switched to DDRE.
Risks: v2 spec remains draft-only and not yet wired into runtime. Any future DDRE implementation should preserve Bayesian path as active fusion component plus kill-switch fallback.
GitHead: d6b5866

## 2026-03-05T09:14:32Z - Roadmap synchronization correction (README -> ROADMAP)
Agent: Codex (GPT-5)
Summary: Synced `ROADMAP.md` to current implementation status by adding native acceleration progress tracking (M10) and DDRE spec-first preparation tracking (M11).
Why: User clarified roadmap upkeep was required (not README), and the existing roadmap did not reflect current postflop-native and DDRE-prep progress.
Files: ROADMAP.md,docs/AI_CONTEXT_ARCHIVE.md
Validation: Documentation update only (no runtime code changes).
Risks: M10 still includes one open stability item for aggregated full-suite runs that can fail while isolated reruns pass.
GitHead: d6b5866

## 2026-03-05T09:33:26Z - M10 aggregate-suite stability fix (Bayes cache + canonical preflop backend isolation)
Agent: Codex (GPT-5)
Summary: Resolved reproducible aggregate-only test instability by hardening property-scoped tests and native-provider cache resets. Full `sbt test` now passes in one aggregated run (319/319).
Why: Continuing from the open M10 stability item, aggregated execution was failing while isolated reruns passed due cross-suite global-state interference.
Files: src/test/scala/sicfun/holdem/HoldemBayesProviderTest.scala,src/test/scala/sicfun/holdem/HeadsUpEquityCanonicalTableTest.scala,ROADMAP.md,docs/AI_CONTEXT_ARCHIVE.md
Validation: sbt "testOnly sicfun.holdem.HoldemBayesProviderTest sicfun.holdem.HeadsUpEquityCanonicalTableTest"; sbt test
Risks: Stability verified on this machine/run; if future suites add unsynchronized system-property mutations, aggregate flakiness can reappear.
GitHead: d6b5866

## 2026-03-05T10:04:04Z - M11 DDRE phase-1 kickoff implemented (safe default off + fusion modes)
Agent: Codex (GPT-5)
Summary: Added DDRE phase-1 runtime scaffolding with explicit `off/shadow/blend-canary/blend-primary` modes, provider/config parsing, synthetic DDRE inference stub, blend/fallback logic, and degraded-mode telemetry while preserving Bayesian default behavior.
Why: Continue workflow by moving the roadmap’s remaining M11 kickoff item from spec-only into executable integration without changing existing decision behavior when DDRE is not enabled.
Files: src/main/scala/sicfun/holdem/HoldemDdreProvider.scala,src/main/scala/sicfun/holdem/RangeInferenceEngine.scala,src/test/scala/sicfun/holdem/HoldemDdreIntegrationTest.scala,src/test/scala/sicfun/holdem/TexasHoldemPlayingHallTest.scala,ROADMAP.md,docs/AI_CONTEXT_ARCHIVE.md
Validation: sbt "testOnly sicfun.holdem.RangeInferenceEngineTest sicfun.holdem.HoldemDdreIntegrationTest"; sbt "testOnly sicfun.holdem.TexasHoldemPlayingHallTest"; sbt test (323 passed, 0 failed)
Risks: DDRE provider is currently synthetic scaffolding (not ONNX/runtime model yet); blend-primary can change posterior only when DDRE mode/provider are explicitly enabled; production model quality gates (NLL/KL/latency) still require Phase-1 model/export pipeline completion.
GitHead: d6b5866

## 2026-03-06T20:45:00Z - Reliability hardening pass (DDRE fallback, always-on gating, analyzer correctness, hybrid concurrency guard)
Agent: Codex (GPT-5)
Summary: Completed a targeted reliability pass across runtime-critical paths: fixed DDRE fallback/shadow numerical drift behavior, corrected always-on villain-response attribution for interleaved hands, replaced placeholder analyzer output with model-backed evaluation flow, and added a concurrent regression guard for hybrid adaptive calibration updates.
Why: You asked to continue stabilizing `sicfun` high-impact issues and then register the outcomes in roadmap/context records.
Files: src/main/scala/sicfun/holdem/RangeInferenceEngine.scala,src/test/scala/sicfun/holdem/HoldemDdreIntegrationTest.scala,src/main/scala/sicfun/holdem/AlwaysOnDecisionLoop.scala,src/test/scala/sicfun/holdem/AlwaysOnDecisionLoopTest.scala,src/main/scala/sicfun/holdem/HandHistoryAnalyzer.scala,src/test/scala/sicfun/holdem/HandHistoryAnalyzerTest.scala,src/main/scala/sicfun/holdem/HeadsUpHybridDispatcher.scala,src/test/scala/sicfun/holdem/HeadsUpHybridDispatcherPlanningTest.scala,ROADMAP.md,docs/AI_CONTEXT_ARCHIVE.md
Validation: sbt "testOnly sicfun.holdem.HoldemDdreIntegrationTest sicfun.holdem.HoldemPostflopNativeParityTest"; sbt "testOnly sicfun.holdem.RangeInferenceEngineTest"; sbt "testOnly sicfun.holdem.AlwaysOnDecisionLoopTest"; sbt "testOnly sicfun.holdem.HandHistoryAnalyzerTest"; sbt "testOnly sicfun.holdem.HeadsUpHybridDispatcherPlanningTest"
Risks: Hand-history CLI still requires explicit `--heroCards` to compute EV/recommendations from feed events; without hero cards, summary remains count-only by design. DDRE provider remains synthetic scaffolding until ONNX/runtime model integration.
GitHead: d6b5866

## 2026-03-06T20:53:00Z - DDRE Phase-1 self-play data export added to playing hall
Agent: Codex (GPT-5)
Summary: Extended `TexasHoldemPlayingHall` with an optional DDRE training export (`ddre-training-selfplay.tsv`) that records decision context, sparse Bayesian baseline posterior labels, and realized villain hole cards for each hero decision point.
Why: The DDRE v2 plan required a concrete Phase-1 data pipeline extension (context + baseline posterior + realized villain hand) before ONNX/runtime model training integration.
Files: src/main/scala/sicfun/holdem/TexasHoldemPlayingHall.scala,src/test/scala/sicfun/holdem/TexasHoldemPlayingHallTest.scala,scripts/run-playing-hall.ps1,ROADMAP.md,docs/AI_CONTEXT_ARCHIVE.md
Validation: sbt "testOnly sicfun.holdem.TexasHoldemPlayingHallTest"; sbt "testOnly sicfun.holdem.HoldemDdreIntegrationTest"
Risks: DDRE export can become large on long runs because sparse posterior columns may contain many entries per decision; export stays opt-in via `--saveDdreTrainingTsv=false` default.
GitHead: d6b5866

## 2026-03-06T21:45:00Z - DDRE native runtime scaffolding (CPU/CUDA JNI + provider routing)
Agent: Codex (GPT-5)
Summary: Added native DDRE CPU/CUDA JNI bindings, a Scala native runtime wrapper, and provider routing in `HoldemDdreProvider` (`native-cpu` / `native-gpu`) with safe degradation to Bayesian fallback when native load/inference fails.
Why: You called out native-side progress gap; DDRE Phase-1 required an executable runtime bridge path beyond Scala-only scaffolding.
Files: src/main/scala/sicfun/holdem/HoldemDdreProvider.scala,src/main/scala/sicfun/holdem/HoldemDdreNativeRuntime.scala,src/main/java/sicfun/holdem/HoldemDdreNativeCpuBindings.java,src/main/java/sicfun/holdem/HoldemDdreNativeGpuBindings.java,src/main/native/jni/DdreNativeInferenceCore.hpp,src/main/native/jni/HoldemDdreNativeCpuBindings.cpp,src/main/native/jni/HoldemDdreNativeGpuBindings.cu,src/main/native/build-windows-llvm.ps1,src/main/native/build-windows-cuda11.ps1,src/main/native/README.md,src/test/scala/sicfun/holdem/HoldemDdreIntegrationTest.scala,ROADMAP.md,docs/AI_CONTEXT_ARCHIVE.md
Validation: sbt "testOnly sicfun.holdem.HoldemDdreIntegrationTest"; sbt "testOnly sicfun.holdem.TexasHoldemPlayingHallTest"
Risks: Native DDRE path currently accelerates the existing synthetic DDRE math core (not ONNX diffusion inference yet); production-quality DDRE model runtime integration remains a follow-up (exported model loader + parity/perf gates).
GitHead: d6b5866

## 2026-03-11T01:18:00Z - Linked Claude and GPT sidecars added beside Gemini
Agent: Codex (GPT-5)
Summary: Added a unified multi-provider sidecar dispatcher (`scripts/ai-minion.ps1`), repo-level `CLAUDE.md` and `GPT.md` contracts, and operator docs so Codex can delegate read-only analysis/review to Gemini, Claude, or GPT/Codex from one workflow instead of treating Gemini as a special-case helper.
Why: You asked to hook Claude and GPT into the same linked-worker workflow as Gemini so the primary agent can choose among them instead of relying on one delegated helper.
Files: scripts/ai-minion.ps1,CLAUDE.md,GPT.md,AGENTS.md,docs/AI_MINIONS.md,docs/GEMINI_MINION.md,docs/OPERATOR_RUNBOOK.md,docs/AI_CONTEXT_ARCHIVE.md
Validation: claude --version; claude auth status --json; node C:\Users\MK1\AppData\Roaming\npm\node_modules\@openai\codex\bin\codex.js --help; node C:\Users\MK1\AppData\Roaming\npm\node_modules\@openai\codex\bin\codex.js login status; node C:\Users\MK1\AppData\Roaming\npm\node_modules\@openai\codex\bin\codex.js exec --json "Reply with OK only." -s read-only; powershell -ExecutionPolicy Bypass -File scripts/ai-minion.ps1 -Action doctor -Provider gpt
Risks: Claude CLI on this machine sometimes returns an empty stdout body in `-p` mode even though the response is persisted to the local session transcript, so the wrapper now extracts the final assistant message from `~/.claude/projects/.../<session-id>.jsonl`. GPT/Codex auth is now account/device-login based instead of API-key based, but its Windows shim was missing here so the wrapper intentionally boots `node ...@openai/codex\bin\codex.js` directly.

## 2026-03-06T22:50:00Z - DDRE ONNX adapter path wired (safe fallback semantics)
Agent: Codex (GPT-5)
Summary: Added a configurable ONNX DDRE provider path (`provider=onnx`) via `HoldemDdreOnnxRuntime` with model-path/input-output/execution-provider controls, reflection-based runtime detection, and safe degradation to Bayesian on ONNX errors.
Why: Continue DDRE execution beyond native synthetic scaffolding and establish an integration path for exported ONNX DDRE models without breaking current runtime when ONNX runtime/model artifacts are missing.
Files: src/main/scala/sicfun/holdem/HoldemDdreOnnxRuntime.scala,src/main/scala/sicfun/holdem/HoldemDdreProvider.scala,src/test/scala/sicfun/holdem/HoldemDdreIntegrationTest.scala,src/main/native/README.md,ROADMAP.md,docs/AI_CONTEXT_ARCHIVE.md
Validation: sbt "testOnly sicfun.holdem.HoldemDdreIntegrationTest"
Risks: ONNX adapter currently depends on runtime class availability (`ai.onnxruntime`) at execution time and does not include project-level dependency pinning yet; model I/O contract is currently expected as prior/likelihood tensors with posterior output and may require alignment with future training export conventions.
GitHead: d6b5866

## 2026-03-06T22:15:00Z - DDRE ONNX smoke-path completion + parity benchmark gate CLI
Agent: Codex (GPT-5)
Summary: Completed DDRE ONNX smoke-path hardening with pinned ONNX runtime dependency, generated a tiny reproducible ONNX smoke model artifact, added a successful ONNX integration test path (not just fallback), and introduced a DDRE parity/benchmark gate CLI with configurable parity/speed thresholds across synthetic/native/onnx providers.
Why: You asked to proceed in order on (1) ONNX dependency pinning + real smoke success path and then (2) DDRE parity/benchmark CLI with gating thresholds.
Files: build.sbt,scripts/generate-ddre-smoke-onnx.py,src/test/resources/sicfun/ddre/ddre-smoke-sqrt.onnx,src/test/scala/sicfun/holdem/HoldemDdreIntegrationTest.scala,src/main/scala/sicfun/holdem/HoldemDdreParityBenchmark.scala,src/main/native/README.md,ROADMAP.md,docs/AI_CONTEXT_ARCHIVE.md
Validation: sbt "testOnly sicfun.holdem.HoldemDdreIntegrationTest"; sbt "testOnly sicfun.holdem.TexasHoldemPlayingHallTest"; sbt "runMain sicfun.holdem.HoldemDdreParityBenchmark --modes=synthetic,onnx --referenceMode=synthetic --onnxModelPath=src/test/resources/sicfun/ddre/ddre-smoke-sqrt.onnx --warmupRuns=0 --measureRuns=2 --hypothesisCount=128 --maxL1Diff=1e-4 --maxAbsDiff=1e-5"
Risks: DDRE parity benchmark skips unavailable modes unless `--requireAllModes=true`; onnx parity thresholds are model-dependent (smoke model parity uses `sqrt(prior)` semantics and float precision), so production models may need stricter/relaxed thresholds by run profile.
GitHead: d6b5866

## 2026-03-07T03:15:25Z - Playing hall hardware saturation: multi-table + parallel runner + autotuner
Agent: Codex (GPT-5)
Summary: Extended `TexasHoldemPlayingHall` with a `--tableCount` scale axis and per-table traceability in outputs, then added a new parallel orchestration script (`run-playing-hall-max.ps1`) that launches multiple JVM workers and optionally auto-tunes profile/worker-count combinations before the full run.
Why: You required exhausting hardware utilization for SICFUN load simulation (online-poker-hall style) rather than single-process throughput only.
Files: src/main/scala/sicfun/holdem/TexasHoldemPlayingHall.scala,src/test/scala/sicfun/holdem/TexasHoldemPlayingHallTest.scala,scripts/run-playing-hall.ps1,scripts/run-playing-hall-max.ps1,ROADMAP.md,docs/AI_CONTEXT_ARCHIVE.md
Validation: sbt "testOnly sicfun.holdem.TexasHoldemPlayingHallTest"; powershell -ExecutionPolicy Bypass -File scripts/run-playing-hall.ps1 -Hands 4 -TableCount 2 -ReportEvery 2 -LearnEveryHands 0 -SaveTrainingTsv false -SaveDdreTrainingTsv false -OutDir data/playing-hall-smoke-tablecount-autopackage2; powershell -ExecutionPolicy Bypass -File scripts/run-playing-hall.ps1 -Hands 2 -TableCount 1 -ReportEvery 1 -LearnEveryHands 0 -SaveTrainingTsv false -SaveDdreTrainingTsv false -OutDir data/playing-hall-smoke-postfix -RefreshClasspath; powershell -ExecutionPolicy Bypass -File scripts/run-playing-hall-max.ps1 -Hands 40 -Workers 2 -TableCountPerWorker 2 -ReportEvery 20 -LearnEveryHands 0 -SaveTrainingTsv false -SaveDdreTrainingTsv false -NativeProfile auto -OutDir data/bench-hall-max-smoke -RefreshClasspath; powershell -ExecutionPolicy Bypass -File scripts/run-playing-hall-max.ps1 -AutoTune -AutoTuneHands 20 -AutoTuneProfiles auto,cpu -AutoTuneWorkerCandidates 1,2 -Hands 30 -TableCountPerWorker 1 -ReportEvery 15 -LearnEveryHands 0 -SaveTrainingTsv false -SaveDdreTrainingTsv false -OutDir data/bench-hall-max-autotune-smoke -ProgressSeconds 2 -RefreshClasspath
Risks: Auto-tune selections are machine-state dependent (thermal throttling/background load can shift the winner); profile `gpu` with high worker counts can oversubscribe GPU resources and may underperform `auto`/`cpu` on some systems.
GitHead: d6b5866

## 2026-03-07T03:17:11Z - Exact GTO hall throughput uplift (decision-only CFR + batched postflop equity)
Agent: Codex (GPT-5)
Summary: Added a lightweight decision-time CFR path that computes only root mixed policy (no exploitability/action-evaluation diagnostics), wired `TexasHoldemPlayingHall` exact-mode GTO decisions to this path, and removed a major hotspot by batching postflop villain equity in one native call per solve.
Why: You requested large throughput gains for long-run `gto-vs-gto` simulation while preserving correctness (not using heuristic fast-mode approximations).
Files: src/main/scala/sicfun/holdem/CfrSolver.scala,src/main/scala/sicfun/holdem/HoldemCfrSolver.scala,src/main/scala/sicfun/holdem/TexasHoldemPlayingHall.scala,src/test/scala/sicfun/holdem/HoldemCfrSolverTest.scala,ROADMAP.md,docs/AI_CONTEXT_ARCHIVE.md
Validation: sbt "testOnly sicfun.holdem.HoldemCfrSolverTest"; sbt "testOnly sicfun.holdem.TexasHoldemPlayingHallTest"; powershell -ExecutionPolicy Bypass -File scripts/run-playing-hall.ps1 -Runner java -RefreshClasspath -Hands 20 -ReportEvery 20 -LearnEveryHands 0 -LearningWindowSamples 0 -HeroStyle gto -VillainStyle gto -GtoMode exact -SaveTrainingTsv false -SaveDdreTrainingTsv false -OutDir data/bench-refresh-smoke2; powershell -ExecutionPolicy Bypass -File scripts/run-playing-hall.ps1 -Runner java -Hands 400 -ReportEvery 400 -LearnEveryHands 0 -LearningWindowSamples 0 -HeroStyle gto -VillainStyle gto -GtoMode exact -SaveTrainingTsv false -SaveDdreTrainingTsv false -OutDir data/bench-postflop-batch-400-auto
Performance: same exact-mode 400-hand benchmark moved from ~197.5s (~2.0 hands/s) before postflop batching to ~61.9s (~6.46 hands/s) after changes (~3.2x faster on this machine/config).
Risks: Fixed-seed run trajectories can differ versus prior builds because postflop MC sampling is now batched (different RNG stream composition), though solver logic/trial budgets remain non-heuristic and policy parity between full and decision-only CFR is regression-tested.
GitHead: d6b5866

## 2026-03-07T03:20:08Z - Operator runbook consolidated
Agent: Codex (GPT-5)
Summary: Added a single operator-focused runbook (`docs/OPERATOR_RUNBOOK.md`) with copy-paste command presets for proof gates, hall load generation (single + max + autotune), DDRE checks, native builds, and common troubleshooting.
Why: You flagged that command surfaces were getting lost across multiple scripts/docs and requested a canonical operational entry point.
Files: docs/OPERATOR_RUNBOOK.md,ROADMAP.md,docs/AI_CONTEXT_ARCHIVE.md
Validation: Documentation update only (content derived from currently implemented scripts and validated commands already in archive).
Risks: Runbook commands assume Windows/PowerShell environment and current script parameters; update this document when CLI flags or script defaults change.
GitHead: d6b5866

## 2026-03-07T03:24:56Z - Interactive runbook launcher added
Agent: Codex (GPT-5)
Summary: Added `scripts/runbook.ps1`, an interactive/one-shot launcher for the five highest-frequency operator actions (quick proof, full proof, hall max autotune, hall max gpu, hall single) with `-WhatIf` preview mode and heavy-run confirmation.
Why: You requested to proceed with a simpler operator surface so commands stop getting lost across tools and docs.
Files: scripts/runbook.ps1,docs/OPERATOR_RUNBOOK.md,ROADMAP.md,docs/AI_CONTEXT_ARCHIVE.md
Validation: powershell -ExecutionPolicy Bypass -File scripts/runbook.ps1 -Action quick-proof -WhatIf; powershell -ExecutionPolicy Bypass -File scripts/runbook.ps1 -Action hall-max-autotune -WhatIf; powershell -ExecutionPolicy Bypass -File scripts/runbook.ps1 -Action hall-single -WhatIf; powershell -ExecutionPolicy Bypass -File scripts/runbook.ps1 -Action quick-proof
Risks: Menu mode is interactive and intended for human terminal usage (non-interactive CI should use one-shot `-Action` mode). Heavy run presets are intentionally aggressive and require explicit YES confirmation unless `-WhatIf` is used.
GitHead: d6b5866

## 2026-03-09T22:54:53Z - Runtime maintainability sweep across sicfun holdem surfaces
Agent: Codex (GPT-5)
Summary: Completed a broad internal cleanup pass over the `sicfun.holdem` runtime/tooling layer by extracting shared CLI decoding into `CliHelpers`, splitting long orchestration methods into dedicated runner/helper stages, and reducing duplicated control flow across hall, solver, DDRE, benchmark, autotune, advisor, analyzer, and postflop entrypoints.
Why: You asked for an aggressive cleanup of the dirty/overgrown `sicfun` surfaces, with the goal of improving navigability and keeping the current runtime behavior covered by targeted regression slices.
Files: src/main/scala/sicfun/holdem/CliHelpers.scala,src/main/scala/sicfun/holdem/TexasHoldemPlayingHall.scala,src/main/scala/sicfun/holdem/HoldemCfrSolver.scala,src/main/scala/sicfun/holdem/HeadsUpHybridDispatcher.scala,src/main/scala/sicfun/holdem/RangeInferenceEngine.scala,src/main/scala/sicfun/holdem/AlwaysOnDecisionLoop.scala,src/main/scala/sicfun/holdem/RealTimeAdaptiveEngine.scala,src/main/scala/sicfun/holdem/HandHistoryAnalyzer.scala,src/main/scala/sicfun/holdem/PokerAdvisor.scala,src/main/scala/sicfun/holdem/HoldemDdreOfflineGate.scala,src/main/scala/sicfun/holdem/HoldemDdreParityBenchmark.scala,src/main/scala/sicfun/holdem/HeadsUpBackendAutoTuner.scala,src/main/scala/sicfun/holdem/HoldemBayesBenchmark.scala,src/main/scala/sicfun/holdem/HoldemPostflopNativeBenchmark.scala,src/main/scala/sicfun/holdem/HoldemPostflopGpuAutoTuner.scala,src/test/scala/sicfun/holdem/CliHelpersTest.scala,src/test/scala/sicfun/holdem/TexasHoldemPlayingHallTest.scala,src/test/scala/sicfun/holdem/HoldemDdreOfflineGateTest.scala
Validation: sbt "testOnly sicfun.holdem.TexasHoldemPlayingHallTest"; sbt "testOnly sicfun.holdem.HoldemCfrSolverTest sicfun.holdem.HeadsUpHybridDispatcherPlanningTest"; sbt "testOnly sicfun.holdem.RangeInferenceEngineTest sicfun.holdem.HoldemDdreIntegrationTest sicfun.holdem.AlwaysOnDecisionLoopTest"; sbt "testOnly sicfun.holdem.CliHelpersTest sicfun.holdem.GenerateSignalsCliTest sicfun.holdem.TrainPokerActionModelCliTest sicfun.holdem.LiveHandSimulatorTest sicfun.holdem.HoldemCfrReportTest sicfun.holdem.AlwaysOnDecisionLoopTest sicfun.holdem.RealTimeAdaptiveEngineTest"; sbt "testOnly sicfun.holdem.HandHistoryAnalyzerTest sicfun.holdem.HoldemDdreOfflineGateTest sicfun.holdem.TexasHoldemPlayingHallTest sicfun.holdem.AdvisorSessionTest"; sbt "testOnly sicfun.holdem.CliHelpersTest sicfun.holdem.HandHistoryAnalyzerTest sicfun.holdem.AdvisorSessionTest sicfun.holdem.HeadsUpGpuExactParityGateTest"; sbt "testOnly sicfun.holdem.CliHelpersTest sicfun.holdem.HeadsUpGpuExactParityGateTest sicfun.holdem.HoldemPostflopNativeParityTest"; sbt "testOnly sicfun.holdem.CliHelpersTest sicfun.holdem.HeadsUpGpuExactParityGateTest sicfun.holdem.HoldemDdreOfflineGateTest"; sbt "testOnly sicfun.holdem.CliHelpersTest sicfun.holdem.HandHistoryAnalyzerTest sicfun.holdem.AlwaysOnDecisionLoopTest sicfun.holdem.HoldemCfrReportTest sicfun.holdem.HoldemDdreOfflineGateTest sicfun.holdem.LiveHandSimulatorTest sicfun.holdem.AdvisorSessionTest"
Risks: Remaining file-local parsers are mostly deliberate domain-specific cases (board/range parsing, path-existence checks, archetype/table-format decoding, and custom raw-value error text) rather than obvious shared infrastructure. This pass was behavior-preserving by intent but broad in surface area, so future changes in these runtime entrypoints should continue to use targeted `testOnly` slices instead of relying on memory.
GitHead: 8ced6f7

## 2026-03-10T21:00:00Z - Gemini sidecar delegation surface added
Agent: Codex (GPT-5)
Summary: Added an optional Gemini CLI sidecar surface with a PowerShell wrapper, a repo-level `GEMINI.md` contract, and focused operator docs so read-heavy analysis/review can be delegated without mixing credentials or ad hoc prompts into repository scripts.
Why: You asked to add Gemini as a token-saving helper/minion using CLI auth while keeping the integration explicit and safe.
Files: scripts/gemini-sidecar.ps1,GEMINI.md,docs/GEMINI_MINION.md,docs/OPERATOR_RUNBOOK.md,docs/AI_CONTEXT_ARCHIVE.md
Validation: npm install -g @google/gemini-cli; npm view @google/gemini-cli version; powershell -ExecutionPolicy Bypass -File scripts/gemini-sidecar.ps1 -Action doctor; powershell -ExecutionPolicy Bypass -File scripts/gemini-sidecar.ps1 -Action auth -WhatIf; powershell -ExecutionPolicy Bypass -File scripts/gemini-sidecar.ps1 -Action delegate -Mode analysis -Task "Smoke test" -ContextPath README.md -WhatIf
Risks: Interactive Google login still requires a human to complete the browser flow; headless delegation will fail until `~/.gemini/settings.json` has an auth type and the OAuth credential cache exists. This wrapper also intentionally uses the package `node ...dist/index.js` entrypoint instead of the generated Windows shim because the current shim is broken on this machine.

## 2026-03-10T03:19:32Z - Long-run hall benchmark matrix + range autotune split
Agent: Codex (GPT-5)
Summary: Ran a larger and mixed-size hall benchmark matrix (`1000`, `5000`, `10000`; `1x1` and `1x2`) and confirmed that the stable ceiling is still native-equity-bound, not exact-GTO-cache-bound. Retuned the heads-up range CUDA cache for long exact-mode hall runs, improving sustained `1 worker x 1 table` throughput while making short `1000`-hand controls worse on this machine.
Why: You asked for bigger and mixed benchmark sizes to separate warmup noise from real steady-state behavior before proceeding with more optimization work.
Files: data/headsup-range-autotune.properties,data/headsup-range-autotune.shortrun-prev.properties,docs/OPERATOR_RUNBOOK.md,docs/AI_CONTEXT_ARCHIVE.md,ROADMAP.md
Validation: sbt "runMain sicfun.holdem.HeadsUpRangeGpuAutoTuner --heroes=256 --entriesPerHero=128 --trials=256 --warmupRuns=1 --runs=3 --cachePath=data/headsup-range-autotune.properties"; sbt "runMain sicfun.holdem.HoldemPostflopGpuAutoTuner --villains=1024 --trials=700 --warmupRuns=1 --runs=3 --cachePath=data/postflop-autotune.properties"; powershell -ExecutionPolicy Bypass -File scripts/run-playing-hall-max.ps1 -Hands 5000 -Workers 1 -TableCountPerWorker 1 -NativeProfile auto -ReportEvery 5000 -LearnEveryHands 0 -SaveTrainingTsv false -SaveDdreTrainingTsv false -HeroStyle adaptive -GtoMode exact -VillainStyle gto -HeroExplorationRate 0.00 -RaiseSize 2.5 -BunchingTrials 80 -EquityTrials 700 -OutDir data/bench-e2e-5000-x1-retuned; powershell -ExecutionPolicy Bypass -File scripts/run-playing-hall-max.ps1 -Hands 5000 -Workers 1 -TableCountPerWorker 1 -NativeProfile auto -ReportEvery 5000 -LearnEveryHands 0 -SaveTrainingTsv false -SaveDdreTrainingTsv false -HeroStyle adaptive -GtoMode exact -VillainStyle gto -HeroExplorationRate 0.00 -RaiseSize 2.5 -BunchingTrials 80 -EquityTrials 700 -OutDir data/bench-e2e-5000-x1-retuned-r2; powershell -ExecutionPolicy Bypass -File scripts/run-playing-hall-max.ps1 -Hands 10000 -Workers 1 -TableCountPerWorker 1 -NativeProfile auto -ReportEvery 10000 -LearnEveryHands 0 -SaveTrainingTsv false -SaveDdreTrainingTsv false -HeroStyle adaptive -GtoMode exact -VillainStyle gto -HeroExplorationRate 0.00 -RaiseSize 2.5 -BunchingTrials 80 -EquityTrials 700 -OutDir data/bench-e2e-10000-x1-retuned
Risks: The new range autotune winner (`block=64`, `chunkHeroes=2048`, `memoryPath=readonly`) is a long-run optimization on this GTX 960M. It raised sustained `1x1` hall throughput from about `109.75` to `127.99-130.59 hands/s` at `5000` hands and from `109.78` to `129.27 hands/s` at `10000` hands, but short `1000`-hand controls fell from about `110.93 hands/s` to `101.77` and `93.96 hands/s`. Exact-GTO cache hit rate stayed roughly flat (`18.7%` to `19.1%`), so this is a native range-kernel throughput tradeoff, not better solver/cache reuse. Use `data/headsup-range-autotune.shortrun-prev.properties` for short smoke/control runs when comparability matters.
GitHead: 8ced6f7
