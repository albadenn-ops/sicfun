# AI Context Archive

This file is an append-only operational memory layer for future AI sessions.
Use it when commit history or local git context is too large/noisy for limited context windows.

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
