# SICFUN Tech Debt — live register

**Anchor:** `90725e01123aebcc5c59fa08d248ddba88c2ff35` (HEAD on `claude/vibrant-kalam-e96d51`), captured 2026-04-26 (refreshed).
**Original anchor:** [`fcbc3a8a`](docs/audits/2026-04-25_anchor_fcbc3a8a.md) (frozen historical audit, 2026-04-25).

Status legend:
- `closed` — work landed and verified.
- `closed-needs-verification` — work landed; deferred verification (e.g., GPU runtime, slow-test execution).
- `partial` — at least one bounded slice landed; further work tracked.
- `open` — not yet started or actionable; no design blocker.
- `deferred` — not actionable today (waits on JDK upgrade, trained model, user judgment, etc.).
- `phantom` — finding describes state not present in this branch.

| ID | Severity | Status | Anchor (file:line) | One-line |
|---|---|---|---|---|
| F1 | High (Test) | partial | `tablegen/`, `equity/HeadsUpEquityCanonicalTable.scala` | Canonical-key invariants + binary-IO roundtrip + Exact-vs-MC parity pinned in `equity/`. `tablegen/` test directory now exists with HeadsUpCanonicalTableReadableDumpTest pinning `handClass` + `sortRows` (11 tests). Other tablegen runners (Generate*, Inspect*, *Tuner) still untested -- they're CLI mains with file-IO side effects. |
| F2 | High (Code) | closed-needs-verification | covered packages: `runtime/`, `validation/`, `provider/`, `equity/`, `model/`, `cfr/`, `gpu/` | All audit Phase 2 + step 3 packages migrated to `ConsoleLogger`. Closed files: AdaptiveProofHarness, HandHistoryReviewServer (full local logger), LiveHandSimulator, AlwaysOnDecisionLoop, HandHistoryAnalyzer, AcpcHeadsUpDealer, AcpcMatchRunner, SlumbotMatchRunner, TexasHoldemPlayingHall, ValidationRunner, HoldemDdreOfflineGate, HeadsUpTableInfo, TrainPokerActionModel, all 5 cfr/ tools, GpuRuntimeSupport (routed). Intentionally on `println`: `PokerAdvisor` interactive REPL (user UI per audit guidance), `analysis/`, `history/`, `tablegen/` CLIs (stdout-as-output contract per audit B2 step 4). The 6 remaining println-shaped lines are stream-sink leaves inside the loggers themselves. |
| F3 | High (Arch) | partial | `web/HandHistoryReviewServer.scala` | Monolith down from 3057 -> 2901 LOC. Extracted: `WebResponses` (5 fns), `WebRateLimiter` (windowing + bucket types, with 9 unit tests via clientKeyFor callback), `AnalysisJobState` (shared ADT). Routing / AuthStack / AnalysisJobStore / PlayingHallJobStore extractions still open; each is ~290 LOC and tightly coupled. |
| F4 | High (Test/Arch) | partial | `runtime/` (9,247 LOC, ratio 0.31) | Multi-table tableId schedule pinned; `AcpcHeadsUpDealer` + `AcpcActionCodec` (16 internals tests including multiway side-pot resolver chip-conservation) pure helpers pinned; protocol street-math deduplicated into `ProtocolStreetMath`; `SlumbotActionCodec` non-parse helpers (incrementForAction inverse, positionForActual, chipsToBb / bbToChips edge cases) pinned. `AdvisorSession` state machine has 37 existing tests already; uncovered surfaces are CLI runners + integration paths. |
| F5 | Med (Arch) | open | `cfr/HoldemCfrSolver.scala` | 3728-LOC god-file, 18 top-level decls; multi-sprint split. Lower urgency until other items land. |
| F6 | Med (Arch) | partial | `runtime/TexasHoldemPlayingHall.scala` | Monolith down from 2,540 -> 2,181 LOC. `HallFormat` (8 fns), `HallVillain` (VillainMode types + CLI parsing), `HallConfig` (Config + parseArgs + 9 *Opt helpers + position resolution + usage) all extracted with their own tests. `TableSimulator`/`TrainingRetrainHook` extractions open. |
| F7 | Med (Code) | closed | `KernelConstructor.scala`, 3 strategic test files | All 3 test files migrated off deprecated `composeFullKernelForWorld` / `buildDesignKernel`; `@nowarn("cat=deprecation")` removed. Deprecated overloads themselves stay (may have external consumers). |
| F8 | Low (Test) | closed | `bench/BenchSupport.scala` | `BenchSupportTest` pre-existed and covers critical paths. Audit explicitly says "do not chase coverage of benchmark drivers". |
| F9 | Low (Docs) | closed | `ROADMAP.md:74` | Path corrected to `docs/ai/AI_CONTEXT_ARCHIVE.md` in commit `33ebae2`. |
| F10 | Low (Deps) | deferred | `com.sun.net.httpserver` callers | Re-evaluate on any JDK upgrade. |
| F11 | Low (Deps) | deferred | `onnxruntime 1.19.2` | Re-evaluate when a trained DDRE model is on deck. |
| F12 | Critical (Infra) | closed | `build.sbt`, `scripts/release-windows.ps1`, `scripts/gpu/*.ps1` | Gate FQCNs corrected in commit `33ebae2`. |
| F13 | Med (Infra) | open | `src/main/native/build/` | 36 tracked Windows DLLs (6.27 MB); destructive policy call (`git rm --cached`) needs user greenlight. |
| F14 | Med (Arch) | closed | `provider/HoldemDdreOnnxRuntime.scala` | Reflection self-test added (drift detection); inference path rewritten with typed `ai.onnxruntime` API (24 reflection sites → 0 in inference path; 9 remain in self-test by design). |
| F15 | Low (Infra) | deferred | `scripts/*.ps1`, `build.sbt:142–175` | PowerShell-only is a design choice on Windows-only host; only matters if Linux build becomes a goal. |
| F16 | Low (Docs) | open | `data/phase2-a3/`, `data/phase2-a4/` | 4 tracked files violate README "data/ is not repo content"; needs user judgment on move target vs README revision. |
| F17 | Med (Arch) | open | 110 `sys.props` / env reads across ≥20 files | Cross-cutting refactor; `HeadsUpGpuExactParityGate.buildSlice` mutates process-wide props. Needs `ScopedRuntimeProperties` design before touching. |
| F18 | Low (Docs) | partial | `docs/superpowers/README.md`, `docs/superpowers/plans/`, `docs/superpowers/specs/` | Index + forward-looking front-matter convention (status: draft\|active\|landed\|abandoned) shipped in `docs/superpowers/README.md`. Back-fill of front-matter into the 44 existing files is per-file research work, deferred. |
| F19 | Low (Infra) | closed | repo root `hs_err_pid*.log` etc. | Already covered by `.gitignore:125-127`; nothing tracked, nothing to do. |
| A1 | — | phantom | (referenced `build.sbt:54`) | `productionMode` setting key does not exist in this branch (`git grep -E productionMode` returns nothing); the proceed-pass plan referenced state not present here. |
| A2 | — | closed | `provider/HoldemDdreOnnxRuntime.scala` | See F14 — typed-API rewrite landed. |
| A3 | — | closed | `test/strategic/ReductionismManifestTest.scala` | Gate armed: `assert(true)` → structural invariant; Silent + Orphan severities now `fail()` with offender list; gate fire verified by injected fixture. |
| B1 | — | partial | `cfr/`, `gpu/`, 5 files | NonFatal substituted across audit-listed sites; CUDA/OpenCL/CPU device-discovery failures now visible via `GpuRuntimeSupport.warn`. Per-provider failure counters exposed via `/health` deferred. |
| B2 | — | closed-needs-verification | every audit-listed package | See F2 -- the audit Phase 2 + step 3 migration is complete. Phase 4 step 4 (analysis/history/tablegen CLIs) explicitly stays on stdout per audit guidance. |
| B3 | — | partial | `runtime/`, `web/` | 7 extractions landed (`HallFormat`, `HallVillain`, `HallConfig`, `WebResponses`, `WebRateLimiter`, `AnalysisJobState`, `ProtocolStreetMath` dedupe). Web routing/auth-stack/job-store splits remain; rate-limit windowing now extractable + unit-testable. |
| B4 | — | deferred | `project/plugins.sbt` | scoverage adoption is a build-engineering decision (test-time dep, CI threshold strategy); not actionable as a single bounded commit. |
| C1 | — | closed | this file | Audit doc renamed to `docs/audits/2026-04-25_anchor_fcbc3a8a.md` and frozen; this register replaces the old single-file audit. Per-finding sub-files deferred — overhead exceeded value at current finding count. |
| C2 | — | closed | `ROADMAP.md` | M5/M6/M9/M10/M11 ROADMAP claims aligned with code reality. |
| C3 | — | phantom | (would have referenced `productionMode`) | Equity tables aren't loaded as classpath resources at runtime; runtime computes equity on-demand. The proceed-pass plan's premise doesn't match this branch. |

## Detailed write-ups
The original audit's per-finding evidence (`F1`–`F19`) lives unmodified in
[`docs/audits/2026-04-25_anchor_fcbc3a8a.md`](docs/audits/2026-04-25_anchor_fcbc3a8a.md).
The `A1`–`C3` proceed-pass plan items were authored inline in the conversation
that produced this register; they aren't separately versioned.

## Refresh procedure
Update the **Anchor SHA** at the top to `git rev-parse HEAD` whenever this
register is revised. Update the status column row by row; do not delete
finding rows when their status flips to `closed` — keeping them preserves the
record and the link to the historical audit. Add new findings at the bottom
with a fresh ID prefix.
