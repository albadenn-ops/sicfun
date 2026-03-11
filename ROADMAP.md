# sicfun Implementation Roadmap

This roadmap maps the whitepaper sections to concrete engineering milestones.

Status note: a checked box means code landed in the repo. It does not automatically mean production readiness, model quality, or live-play readiness.

## M0 Core math (done)
- [x] 5-of-7 evaluator with total ordering
- [x] Exact/MC equity + EV utilities
- [x] Range parsing + Bayes update scaffolding
- [x] Metrics utilities (entropy, variance)

## M1 Equity tables + traceability (done)
- [x] Heads-up combo table generator
- [x] Canonical (symmetry-reduced) table generator
- [x] Binary IO for tables
- [x] Metadata header (mode/trials/seed/version/coverage)
- [x] Per-entry uncertainty for MC tables
- [x] Coverage report helpers (percent of canonical keys present)
- [x] Decide canonical vs full table as default artifact - canonical chosen (smaller artifact: Int key vs Long key, fewer entries via suit isomorphism; exact for preflop HU equity; full table available as alternative when post-board symmetry doesn't hold)

## M2 ActionModel data contract
- [x] Event schema (actions, bets, positions, timing)
- [x] Feature extraction API (observable-only)
- [x] Dataset layout + provenance metadata

## M3 ActionModel training + calibration
- [x] Multinomial logistic regression baseline
- [x] Calibration metrics + Brier score gates
- [x] Versioning + retirement workflow
- [x] Integrate trained ActionModel into BayesianRange.update() for live posterior inference
- [x] Default holdout split for calibration when no external evaluation set is provided
- [x] Persist/load versioned model artifacts (weights + metadata + lifecycle state)
- [x] CLI training entrypoint for artifact generation from TSV datasets

## M4 Behavioral metrics pipeline
- [x] EV variance with uncertainty
- [x] Action entropy + conditional entropy
- [x] Bayesian collapse metrics (requires M3 integration)
- [x] Player signature vector

## M5 Real-time engine
- [x] Actor/state model (HandState + HandEngine)
- [x] Idempotent event ingestion (sequenceInHand deduplication, out-of-order delivery)
- [x] Snapshot + recovery (HandStateSnapshotIO: state.properties + events.tsv)
- [x] Latency targets (p95 < 1ms per applyEvent for 20-event hands, verified in HandEngineTest)

## M6 Batch analytics
- [x] Multi-shard batch training pipeline (single-process)
- [ ] Distributed training pipeline (planned; deferred while prioritizing M3/M5/M7 deliverables)
- [x] Longitudinal stability tests
- [x] Clustering + fingerprinting

## M7 Signals API + audit trail
- [x] Structured signals payload
- [x] Full reconstruction path (events + model version)
- [x] Exportable audit logs
- [x] CLI batch generation of signals from snapshot + model artifact

## M8 Validation & compliance
- [x] Hand evaluator category distribution checks
- [x] MC convergence tests vs exact
- [x] Operational regression suite

## M9 Continuous ops + context continuity (new)
- [x] Runnable simulator entrypoint (`LiveHandSimulator`) with end-to-end proof output
- [x] Always-on loop entrypoint (`AlwaysOnDecisionLoop`) for feed polling, snapshotting, decision emission, and signal logging
- [x] Scheduled retraining hook in always-on loop (artifact retrain + hot reload)
- [x] Playing hall simulator (`TexasHoldemPlayingHall`) for large-volume play + periodic model retraining
- [x] Multi-table hall load model (`--tableCount`) with per-table traceability in outputs (`tableId` in hand/training/DDRE logs)
- [x] Parallel hall saturation runner with optional auto-tuning of profile/worker count (`scripts/run-playing-hall-max.ps1`)
- [x] Consolidated operator command runbook (`docs/OPERATOR_RUNBOOK.md`) for day-to-day execution flow
- [x] Interactive runbook launcher (`scripts/runbook.ps1`) for one-command access to top operational actions
- [x] Append-only AI context archive (`docs/AI_CONTEXT_ARCHIVE.md`) for future bounded-context sessions
- [x] Context append helper script (`scripts/archive-context.ps1`) with git metadata
- [ ] True autonomous gameplay integration (real table adapter + action executor)
- [x] Equilibrium baseline module (CFR/Nash-style reference policy) for exploitability-aware deviations

## M10 Native acceleration surfaces (done)
- [x] Heads-up native runtime providers (CPU/CUDA/OpenCL/hybrid)
- [x] Heads-up range CUDA autotuner CLI (`HeadsUpRangeGpuAutoTuner`) with cache-based runtime selection
- [x] CFR native providers (CPU + CUDA build) with auto selection
- [x] Bayesian native providers (CPU + CUDA build) with optional shadow parity validation
- [x] Postflop native runtime (CPU + CUDA) with auto-engine routing and CPU fallback on CUDA failure
- [x] Postflop CUDA launch controls (block/chunk/trials-per-launch) and cache-based autotune application
- [x] Postflop benchmark CLI (`HoldemPostflopNativeBenchmark`)
- [x] Postflop CUDA autotuner CLI (`HoldemPostflopGpuAutoTuner`)
- [x] Postflop parity/behavior suite (`HoldemPostflopNativeParityTest`)
- [x] Stabilize full-suite aggregated execution where isolated suite reruns pass (order/timing sensitivity on current machine)

## M11 DDRE phase-1 preparation (experimental; not model-complete)
- [x] Feasibility audit against `SICFUN_Phase1_DDRE_Spec.docx`
- [x] Structured extraction of source spec into local trace files under `docs/`
- [x] Drafted implementation-ready DDRE v2 spec with cooperative Bayesian+DDRE fusion semantics (`docs/SICFUN_Phase1_DDRE_Spec_v2.md`)
- [x] Engineering implementation kickoff (DDRE mode/provider scaffolding + shadow/blend fusion integration + fallback telemetry)
- [x] DDRE Phase-1 data export path: playing hall now emits self-play DDRE training TSV with context + Bayesian baseline posterior + realized villain hand labels (`TexasHoldemPlayingHall`)
- [x] DDRE native runtime bridge scaffolding: CPU/CUDA JNI bindings + Scala runtime wrapper + provider routing with Bayesian fallback on native errors (`HoldemDdreProvider`, `HoldemDdreNativeRuntime`)
- [x] DDRE ONNX provider adapter: configurable model path/input-output names/execution provider with safe degrade-to-Bayesian behavior when ONNX model/runtime is unavailable (`HoldemDdreOnnxRuntime`)
- [x] DDRE ONNX smoke-path readiness: pinned ONNX runtime dependency, reproducible tiny smoke model artifact generator, and adapter-level integration coverage for successful non-fallback ONNX execution (`build.sbt`, `scripts/generate-ddre-smoke-onnx.py`, `HoldemDdreIntegrationTest`)
- [x] DDRE parity/benchmark gate CLI for provider plumbing checks across synthetic/native/onnx paths (`HoldemDdreParityBenchmark`)
- [x] DDRE artifact contract + offline gate: ONNX artifacts now carry validation metadata, experimental artifacts are blocked by default in decision-driving modes, and `HoldemDdreOfflineGate` can promote an artifact after offline NLL/KL/latency checks
- [ ] Train and validate a real DDRE model artifact against offline NLL/KL/latency gates
- [ ] Replace synthetic/native-synthetic DDRE as the primary decision-driving path

## M12 Reliability hardening (ongoing)
- [x] DDRE shadow/fallback posterior stability: preserve exact Bayesian posterior in off/shadow/fallback paths and short-circuit alpha edge cases in fusion (`RangeInferenceEngine`)
- [x] Always-on response attribution hardening: replaced global raise-response gate with per-hand tracking to prevent interleaved-hand contamination (`AlwaysOnDecisionLoop`)
- [x] Hand-history analyzer correctness: removed placeholder zero-EV recommendations; now runs model-backed analysis when `--heroCards` is supplied and uses `--model` artifact when present (`HandHistoryAnalyzer`)
- [x] Hybrid dispatcher adaptive-weight concurrency guard: added regression coverage proving lossless concurrent adaptive calibration updates (`HeadsUpHybridDispatcherPlanningTest`)
- [x] Hall launcher classpath resilience: hardened Java classpath export/parsing against noisy `sbt` output for stable long-running scripted runs (`scripts/run-playing-hall.ps1`, `scripts/run-playing-hall-max.ps1`)
- [x] Exact GTO hall throughput uplift (correctness-preserving): added decision-only CFR root-policy path and batched postflop villain equity evaluation in `HoldemCfrSolver`, then wired `TexasHoldemPlayingHall` exact mode to the lightweight solver path
- [x] Runtime/tooling maintainability sweep: extracted shared CLI decoding/helpers and reduced orchestration hotspots across hall, CFR/DDRE benchmarks, heads-up autotuners, analyzer/advisor loops, and postflop benchmark/tuner surfaces (`CliHelpers`, `TexasHoldemPlayingHall`, `HoldemCfrSolver`, `HeadsUpBackendAutoTuner`, `HoldemDdreParityBenchmark`, `AlwaysOnDecisionLoop`, `PokerAdvisor`, `HandHistoryAnalyzer`, `HoldemPostflop*`)
