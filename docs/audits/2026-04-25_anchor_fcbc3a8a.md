# SICFUN Tech Debt Audit — src/ tree (frozen historical)

> **Frozen historical audit.** See [`TECH_DEBT_AUDIT.md`](../../TECH_DEBT_AUDIT.md)
> at the repo root for the current live register.
>
> This file remains pinned to the original anchor SHA. Closing or revising a
> finding here would erase the historical evidence the live register
> references. Edit the live register's status column instead.

**Anchor commit:** `fcbc3a8af76ac80e502ed1765faf87048e79289a` (`master`)
**Scope:** `src/main/scala`, `src/test/scala` only. `src/main/java` (12 files) and `src/main/native` (71 files) excluded. Build files, scripts, docs are touched only where they contradict `src/` reality.
**Method:** static inspection via grep + Read against the commit above. No build or tests were run. Dependency currency claims use published release timelines and are marked `inferred`; everything else is backed by a file path and line number.
**Role note (per `CLAUDE.md`):** findings first, review signal for the primary agent — not a final call. Several `inferred` rows need a build/runtime confirmation before anyone acts on them.

**Note on delivery location:** the user-selected folder `untitled/` was read-only for this session, so this file was written one level up at `/mnt/TECH_DEBT_AUDIT.md`. Move it to the repo root when you pull it into the working copy.

---

## 1. Evidence anchors

| Measurement | Value | Source |
|---|---|---|
| `git rev-parse HEAD` | `fcbc3a8af76ac80e502ed1765faf87048e79289a` | — |
| `find src/main/scala -name '*.scala' \| wc -l` | 207 | — |
| `find src/test/scala -name '*.scala' \| wc -l` | 165 | — |
| `wc -l` main (total) | 67,269 | — |
| `wc -l` test (total) | 35,251 | — |
| main : test LOC ratio | 1.91 : 1 (test is 52% of main) | — |
| `grep -c 'TODO\|FIXME\|HACK\|XXX' src/main/scala …` | 0 markers, 1 `NOTE:` | `HoldemCfrNativeRuntime.scala:168` |
| `grep -c 'println' src/main/scala` | 643 | — |
| `grep -c '\bvar \b' src/main/scala` | 920 | — |
| `grep -c '\bnull\b' src/main/scala` | 156 | — |
| `grep -c 'asInstanceOf' src/main/scala` | 17 | — |
| `grep -c '@deprecated' src/main/scala` | 7 across 4 files | — |
| `grep -c '@nowarn("cat=deprecation")' src/test/scala` | 3 files | kernel tests |
| `scalaVersion` | 3.8.1 | `build.sbt:3` |
| Runtime libs (main) | 3: `onnxruntime 1.19.2`, `ujson 3.3.1`, `postgresql 42.7.10` | `build.sbt:5-17` |
| Test libs | 2: `munit 1.2.2`, `embedded-postgres 2.2.2` | `build.sbt:13-14` |
| `scalacOptions` | `-deprecation -feature -unchecked -Wunused:imports,privates,locals -Werror -new-syntax -indent` | `build.sbt:19-27` |

---

## 2. Findings — scored

Priority = `(Impact + Risk) × (6 − Effort)`, each dimension 1–5. Higher priority acts first.

| # | Finding | Cat | Impact | Risk | Effort | Priority |
|---|---|---|:-:|:-:|:-:|:-:|
| F1 | `tablegen/` has **zero tests** (1,223 LOC; equity table generation is a correctness-critical data surface) | Test | 3 | 5 | 3 | **24** |
| F2 | 643 `println` calls in `src/main` — no structured logging abstraction, runtime mixed with benchmarks | Code | 4 | 3 | 3 | **21** |
| F3 | `HandHistoryReviewServer.scala` = 2,188 LOC single file (routing, auth, CSRF, OIDC, rate-limit, jobs) on bare `com.sun.net.httpserver` | Arch | 5 | 4 | 4 | **18** |
| F4 | `runtime/` is the largest package (9,247 LOC) with the worst non-bench test ratio (test/main = 0.31) | Test/Arch | 5 | 4 | 4 | **18** |
| F6 | `TexasHoldemPlayingHall.scala` = 2,507 LOC, single `object` (`:75`) holding the whole multi-table hall | Arch | 4 | 3 | 4 | **14** |
| F7 | 7 `@deprecated` symbols with live callers; tests suppress via `@nowarn("cat=deprecation")` in 3 files | Code | 2 | 2 | 3 | **12** |
| F8 | `bench/` test ratio 0.08 (7,925 LOC main vs 624 LOC test) | Test | 2 | 2 | 3 | **12** |
| F11 | `onnxruntime 1.19.2` is ~18 months old; DDRE ONNX is declared plumbing-only so risk is low today (`inferred`) | Deps | 1 | 2 | 2 | **12** |
| F9 | `ROADMAP.md:74` references `docs/AI_CONTEXT_ARCHIVE.md`; actual path is `docs/ai/AI_CONTEXT_ARCHIVE.md` | Docs | 1 | 1 | 1 | **10** |
| F5 | `HoldemCfrSolver.scala` = 3,728 LOC, 18 top-level decls, 96 `var`s | Arch | 4 | 3 | 5 | **7** |
| F10 | `com.sun.net.httpserver` is JDK-internal; future JDK moves could break the web surface | Deps | 2 | 3 | 5 | **5** |

Top four by priority: **F1, F2, F3, F4**.

---

## 3. Finding details (with evidence)

### F1 — tablegen/ has zero tests
- Source files under `src/main/scala/sicfun/holdem/tablegen/`: `DeviceProofRun.scala`, `GenerateHeadsUpCanonicalTable.scala`, `GenerateHeadsUpTable.scala`, `HeadsUpCanonicalExactBoardMajorTuner.scala`, `HeadsUpCanonicalExactTuner.scala`, `HeadsUpCanonicalTableReadableDump.scala`, `InspectCanonicalBatch.scala`.
- `src/test/scala/sicfun/holdem/tablegen/` does not exist.
- Equity tables are referenced from resource generators in `build.sbt:93–101` and cached by stamp (`build.sbt:64`). A silent generation bug would land in `heads-up-equity.bin` / `heads-up-equity-canonical.bin` and poison every downstream equity consumer. README `:25` and ROADMAP M1 (`ROADMAP.md:14–20`) both claim this surface is "done".
- Recommended slice: property tests for canonical key encode/decode; exact-vs-MC parity for a small enumerated range; `resourceGenerators` stamp-cache invalidation test.

### F2 — println sprawl, no logging abstraction
- 643 hits in `src/main`. Top files: `bench/HybridBenchmark.scala` (32), `runtime/TexasHoldemPlayingHall.scala` (30), `validation/ValidationRunner.scala` (22), `bench/HoldemCfrNativeFixedBenchmark.scala` (21), `bench/HeadsUpBackendComparison.scala` (20).
- Benchmark stdout use is acceptable. Runtime + validation are not — no level filtering, no correlation IDs, no redirection for packaged/service mode (README `:71–72` describes NSSM-backed service install).
- No SLF4J / scala-logging / log4j dep in `build.sbt:12–18` — intentional minimalism.

### F3 — Web server monolith
- `src/main/scala/sicfun/holdem/web/HandHistoryReviewServer.scala` is 2,188 LOC and declares only `object HandHistoryReviewServer:` at `:61`.
- Imports (`:3–15`) concentrate routing, auth, CSRF, OIDC, rate-limit, job queue, readiness/drain, static serving on `com.sun.net.httpserver` + `ujson` + JDK `MessageDigest`/`Base64`/`URLDecoder`.
- `HandHistoryReviewServer.scala:34` comments "Built on `com.sun.net.httpserver` for zero external dependencies" — intentional. Default bind is `127.0.0.1` (`:805`), matching `README.md:63`.
- `PlatformUserAuth.scala` (913 LOC) in the same package is a related god-file to review alongside.

### F4 — runtime/ test coverage
Test:main LOC ratio per package (worst → best):

| Package | main LOC | test LOC | ratio |
|---|--:|--:|--:|
| tablegen | 1,223 | 0 | 0.00 |
| bench | 7,925 | 624 | 0.08 |
| runtime | 9,247 | 2,855 | 0.31 |
| gpu | 4,099 | 1,566 | 0.38 |
| cfr | 7,972 | 3,084 | 0.39 |
| provider | 2,237 | 944 | 0.42 |
| web | 3,601 | 1,595 | 0.44 |
| equity | 3,680 | 1,684 | 0.46 |
| strategic | 5,741 | 8,669 | 1.51 |

- `runtime/` is both the largest package and houses `TexasHoldemPlayingHall.scala` (2,507), `AcpcMatchRunner.scala` (1,254), `SlumbotMatchRunner.scala` (1,082), `AdvisorSession.scala` (1,005), `AcpcHeadsUpDealer.scala` (886) — all exercised by ROADMAP M5/M9 claims.
- LOC ratio is a weak coverage proxy; a scoverage/JaCoCo run would sharpen this. Flagged `inferred` until then.

### F5 — HoldemCfrSolver god-file
- `src/main/scala/sicfun/holdem/cfr/HoldemCfrSolver.scala`: 3,728 LOC, 18 top-level decls, 96 `var`s.
- Same package: `CfrSolver.scala` (1,077), `HoldemCfrExternalComparison.scala` (970). Decomposition candidates: `HoldemCfrConfig` (currently `:45–58`), provider selection (`autoChosenProviderRef`, `:220`), caches (`equityLookupCache`, `villainSupportOrderingCache`, `:221–223`).
- `var` density is likely intentional for hot-path perf — do not propose a blanket refactor. Target splits are structural, not mutation-style.

### F6 — TexasHoldemPlayingHall single-object
- `src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala`: 2,507 LOC, declares only `object TexasHoldemPlayingHall:` at `:75`.
- Decompose only after F4 tests pin multi-table traceability and saturation behavior first.

### F7 — Deprecation migration backlog
- `@deprecated` sites:
  - `StrategicRivalBelief.scala:17` — `update` deprecated v0.32.
  - `KernelConstructor.scala:58,177` — `buildDesignKernel`, `composeFullKernelForWorld` deprecated v0.31.1.
  - `ExploitationInterpolation.scala:116` — `buildInterpolatedKernel` deprecated v0.32.
  - `StrategicEngine.scala:215` — overload of `decide` deprecated v0.33.
- `composeFullKernelForWorld` has ~30 call sites in `src/test/scala/sicfun/holdem/strategic/kernel/*Test.scala`, all under `@nowarn("cat=deprecation")` (`DynamicsTest.scala:12`, `KernelConstructorTest.scala:10`, `RivalKernelLawTest.scala:10`).
- Build is `-Werror + -deprecation`; do **not** drop deprecated symbols until callers migrate. Action is caller migration to the `*Full` variants.

### F8 — bench/ test ratio 0.08
- 7,925 main LOC vs 624 test LOC. Most of `bench/` is driver code that is not a standard unit-test target. `BenchSupport.scala` shared harness and regression drivers are the realistic test targets.
- Lower risk than tablegen/runtime.

### F9 — Stale ROADMAP path
- `ROADMAP.md:74`: `` `docs/AI_CONTEXT_ARCHIVE.md` `` — file does not exist there. Real path: `docs/ai/AI_CONTEXT_ARCHIVE.md`. One-line fix.

### F10 — `com.sun.net.httpserver` stability
- Class lives in the `jdk.httpserver` module, not `java.*`. No formal compatibility guarantee beyond the shipping JDK. Materializes as risk only on a JDK upgrade.

### F11 — `onnxruntime 1.19.2`
- 1.19.2 released ~Oct 2024 (`inferred`); current at audit date likely 1.21+ (`inferred`). README `:33` declares the ONNX path as "adapter plumbing" and `HoldemDdreProvider.scala:302` enforces `synthetic` DDRE is a "heuristic scaffold, not a trained diffusion model" → low immediate risk. Upgrade when a trained DDRE model actually ships.

---

## 4. Phased remediation plan

Interleaves with feature work; each phase is one or two reasonable weeks per engineer.

### Phase 0 — cheap + immediate (hours)
- **F9**: fix `ROADMAP.md:74` path to `docs/ai/AI_CONTEXT_ARCHIVE.md`.
- Introduce a ~30-line logging trait backed by `System.out` (no new dep). Landing ground for F2.

### Phase 1 — correctness floor (1 sprint)
- **F1**: add `src/test/scala/sicfun/holdem/tablegen/` with:
  - Property: canonical key encode/decode roundtrips.
  - Parity: exact-mode vs MC-mode equity for a bounded matchup slice (convergence within N standard errors).
  - Resource stamp: `mode/trials/seed/parallelism/backend` cache-invalidation test vs `build.sbt:64`.

### Phase 2 — runtime hardening (1–2 sprints, overlap with feature work)
- **F4 + F6**: before splitting `TexasHoldemPlayingHall.scala`, add tests pinning multi-table traceability (`tableId` in logs, per `ROADMAP.md:70`) and saturation behavior. Then extract `HallConfig`, `TableSimulator`, `TrainingRetrainHook`, and a top-level orchestrator.
- **F2**: migrate `runtime/` and `validation/` `println` sites to the new logging trait. Leave `bench/` alone.

### Phase 3 — web decomposition (1 sprint + review)
- **F3**: split `HandHistoryReviewServer.scala` into `Routing`, `AuthStack` (BASIC / platform-user / OIDC), `JobQueue`, `RateLimit`, `ReadinessDrain`, `StaticAssets`. Keep `com.sun.net.httpserver` root; no new web framework. Pair with `PlatformUserAuth.scala` (913 LOC) cleanup.

### Phase 4 — deprecation drain + structural cleanup (opportunistic)
- **F7**: migrate callers to the `*Full` variants in `DynamicsTest.scala`, `KernelConstructorTest.scala`, `RivalKernelLawTest.scala`; remove `@nowarn` once clean.
- **F5**: split `HoldemCfrSolver.scala` along Config / provider selection / caches / solve entry-points axes. Keep `var`-heavy hot-path bodies intact.
- **F8**: unit-test `BenchSupport.scala` helpers; do not chase "coverage" of benchmark drivers.

### Phase 5 — deferred watch list (no action)
- **F10**: re-evaluate on any JDK upgrade.
- **F11**: re-evaluate when a trained DDRE model is on deck.

---

## 5. What this audit does not cover
- Actual test coverage (scoverage / JaCoCo). "Zero tests" is by-directory, not by-line.
- ~~Runtime correctness / parity gates (`gpuSmokeGate`, `gpuExactParityGate` in `build.sbt:179–201`). Would need execution.~~ → **expanded in section 7 (F12)**. Still not executed, but the wiring is audited.
- ~~`src/main/java` (12 files) and `src/main/native` (71 files, JNI/CUDA).~~ → **expanded in section 7 (F13, F14)**.
- ~~`scripts/`, `data/`, `dist/`, `docs/` beyond the one stale link surfaced.~~ → **expanded in section 7 (F15, F16, F18, F19)**.
- ~~Dynamic coupling (reflection, ServiceLoader, JNI boundaries) — grep cannot see these.~~ → **expanded in section 7 (F14, F17)**.
- Security review of `HandHistoryReviewServer` auth paths. Route that to a dedicated security review.

## 6. Open questions for the primary agent
1. Is there an existing logging abstraction I missed (maybe in `src/main/native`)? If so, F2 effort drops from 3 to 1.
2. Is the `tablegen/` test gap deliberate because generation is validated via `validation-output/` reports? Worth confirming before investing in F1.
3. Does the team already have a planned split for `HandHistoryReviewServer`? Align F3's decomposition rather than propose a new shape.
4. (New — F12) Have `sbt gpuSmokeGate`, `sbt gpuExactParityGate`, `scripts/release-windows.ps1`, or `scripts/gpu/gpu-smoke-gate.ps1` actually been run since the gates moved to `sicfun.holdem.bench.gate`? If yes, please share the output — my static read says they should fail with `ClassNotFoundException`, and I'd rather disprove that than act on it.
5. (New — F14) Is the `HoldemDdreOnnxRuntime` reflection wrapper a deliberate decoupling so DDRE can load without ONNX on the classpath, or leftover from before `onnxruntime` became a compile dep?

---

## 7. Expanded scope — gates, native, scripts, dynamic coupling

Scope additions: runtime correctness / parity gates (static read only — no execution); `src/main/java` (12 files, 1,080 LOC) and `src/main/native` (71 files, mixed .cpp/.hpp/.cu/.cuh/.cl/.ps1/.dll/.lib/.exp/.exe/.md); `scripts/`, `data/`, `dist/`, `docs/` beyond the single stale link; dynamic coupling (reflection, ServiceLoader, JNI boundaries, system-property-driven routing).

No JVM, native, or sbt executions were run. Findings below are from static reads only; F12 in particular should be disproven at runtime before anyone acts on my proposed fix.

### 7.1 Evidence anchors (added)

| Measurement | Value | Source |
|---|---|---|
| `find src/main/java -name '*.java' \| wc -l` | 12 | — |
| `find src/main/java -name '*.java' -exec wc -l` | 1,080 LOC | — |
| `find src/main/native -type f` by extension | 13 `.lib`, 13 `.exp`, 12 `.dll`, 8 `.cpp`, 6 `.ps1`, 6 `.cu`, 5 `.hpp`, 3 `.md`, 1 `.inc`, 1 `.exe`, 1 `.cuh`, 1 `.cl`, 1 `.bat` | — |
| Tracked Windows binaries under `src/main/native/build/` | 36 files, 6,269,941 bytes at HEAD | `git ls-files` |
| Largest tracked native binary | `sicfun_postflop_cuda.dll` (3,084,288 bytes) | — |
| Non-Windows native artifacts (`.so` / `.dylib` / `.a`) | 0 | — |
| Native build scripts | 5 PowerShell files, 0 bash/make/cmake | `src/main/native/build-windows-*.ps1`, `build_bench.ps1`, `build_multiarch.ps1` |
| `scripts/` file counts by ext | 32 `.ps1`, 29 `.txt`, 14 `.jsonl`, 2 `.py`, 2 `.log`, 2 `.cmd`, 1 `.env`, 0 `.sh` | — |
| Directory sizes on disk | `data/` 288 MB, `dist/` 215 MB, `docs/` 3 MB, `scripts/` 3 MB, `target/` 59 MB | `du -sm` |
| Files tracked in `dist/` | 0 | ✅ correctly ignored |
| Files tracked in `data/` | 4 | ⚠ policy violation (see F16) |
| Reflection sites (`Class.forName` + `getMethod`/`getConstructor`/`newInstance`/`invoke`) | 24, all in `HoldemDdreOnnxRuntime.scala` | — |
| `ServiceLoader` usage | 0 | — |
| `@native` methods across Java bindings | 48 across 12 `*NativeBindings.java` files | — |
| `sys.props` / `System.getProperty` / `sys.env` / `System.getenv` occurrences | 110 across ≥20 files | — |
| `System.load(Library)?` call sites | 9 across 4 files (`GpuRuntimeSupport.scala`, `HoldemPostflopNativeRuntime.scala`, `WassersteinDroRuntime.scala`, one comment in `HeadsUpGpuRuntime.scala`) | — |

### 7.2 Added findings

| # | Finding | Cat | Impact | Risk | Effort | Priority |
|---|---|---|:-:|:-:|:-:|:-:|
| **F12** | **Gate class-path is wrong in every caller.** Gates live at `sicfun.holdem.bench.gate.*` but `build.sbt:183,195`, `scripts/release-windows.ps1:95,102,187,204`, `scripts/gpu/gpu-smoke-gate.ps1:68`, `scripts/gpu/gpu-exact-parity-gate.ps1:65` all reference the wrong FQCN — static read predicts `ClassNotFoundException` at every invocation point. **Verify by running one gate before patching.** | Infra | 5 | 5 | 1 | **50** |
| F13 | `src/main/native/build/` has **36 tracked Windows binaries (6.27 MB)** despite `.gitignore:89` listing the directory; no Linux/macOS artifacts; all build scripts are PowerShell | Infra | 3 | 3 | 3 | **18** |
| F14 | `HoldemDdreOnnxRuntime.scala` uses **24 reflection sites** to call the ONNX Runtime API even though `onnxruntime` is a compile-time dep (`build.sbt:16`) | Arch | 2 | 3 | 3 | **15** |
| F15 | Operator + native build scripts are **PowerShell-only** (32 `.ps1`, 0 `.sh`); README/ROADMAP both assume Windows as the only execution host | Infra | 2 | 2 | 5 | **4** |
| F16 | **Policy vs practice gap**: `README.md:19` says `data/` is not repo content, yet `data/phase2-a3/2026-04-17-1601/*` and `data/phase2-a4/2026-04-17-1740/*` (4 files) are staged for commit | Docs | 1 | 2 | 1 | **15** |
| F17 | **110 `sys.props` / env reads across ≥20 files**; `HeadsUpGpuExactParityGate.scala:118–125` mutates process-wide properties between CPU/CUDA builds in the same run (test-hostile coupling) | Arch | 3 | 3 | 4 | **12** |
| F18 | `docs/superpowers/plans/` has **34 dated plans** + `docs/superpowers/specs/` has 13 specs (2026-03-10 → 2026-04-16) with no index or lifecycle marker — landed vs active vs abandoned is unknowable from filenames | Docs | 2 | 1 | 2 | **12** |
| F19 | 7 `hs_err_pid*.log` + 1 `replay_pid*.log` crash dumps (2026-04-06) in repo root — correctly ignored by `.gitignore:125-126`, but left as workspace clutter | Infra | 1 | 1 | 1 | **10** |

### 7.3 Finding details

#### F12 — Broken parity-gate class references
Actual FQCNs (verified via package decl at HEAD):
- `sicfun.holdem.bench.gate.HeadsUpGpuSmokeGate` — `src/main/scala/sicfun/holdem/bench/gate/HeadsUpGpuSmokeGate.scala:1`
- `sicfun.holdem.bench.gate.HeadsUpGpuExactParityGate` — `src/main/scala/sicfun/holdem/bench/gate/HeadsUpGpuExactParityGate.scala:1`
- `sicfun.holdem.bench.gate.HeadsUpGpuPocGate` — `src/main/scala/sicfun/holdem/bench/gate/HeadsUpGpuPocGate.scala:1`

Broken callers:
- `build.sbt:183` → `"sicfun.holdem.bench.HeadsUpGpuSmokeGate"` (missing `.gate`)
- `build.sbt:195` → `"sicfun.holdem.bench.HeadsUpGpuExactParityGate"` (missing `.gate`)
- `scripts/release-windows.ps1:95` → `sicfun.holdem.HeadsUpGpuSmokeGate` (missing `.bench.gate`)
- `scripts/release-windows.ps1:102` → `sicfun.holdem.HeadsUpGpuExactParityGate` (missing `.bench.gate`)
- `scripts/release-windows.ps1:187,204` → same bad prefix (`sicfun.holdem.`)
- `scripts/gpu/gpu-smoke-gate.ps1:68` → `sicfun.holdem.bench.HeadsUpGpuSmokeGate` (missing `.gate`)
- `scripts/gpu/gpu-exact-parity-gate.ps1:65` → `sicfun.holdem.bench.HeadsUpGpuExactParityGate` (missing `.gate`)

`sbt` `runMain` and `run.run(<className>, …)` resolve the class by FQCN at runtime → no compile-time failure. Static prediction is a runtime `ClassNotFoundException`. Because the ROADMAP claims these gates are the correctness enforcement for GPU parity, a silently-broken gate is a real regression risk.

Fix is literal: update the five FQCNs (two in `build.sbt`, two in `release-windows.ps1` × four lines, one each in the two gate scripts). Effort 1. **Before shipping the fix, run one of the current references to reproduce the failure so we know what we're actually correcting.**

Secondary finding inside F12: `HeadsUpGpuPocGate` exists (229 LOC) but is not wired to any `sbt` task. It has no callers in `scripts/`, `build.sbt`, or tests (sole reference in `src/main/native/README.md`). It may be dead code or a missing gate.

#### F13 — Tracked Windows binaries in `src/main/native/build/`
- `git ls-files src/main/native/build/` → 36 entries (.dll / .lib / .exp); 6,269,941 bytes at HEAD.
- `.gitignore:89-91` includes `src/main/native/build/`, `src/main/native/build-ddre-verify/`, `src/main/native/build-ddre-verify-cuda/` — but tracked files are unaffected by `.gitignore`.
- Only Windows artifacts. Native build scripts (`src/main/native/build-windows-cuda11.ps1`, `build-windows-llvm.ps1`, `build-windows-opencl.ps1`, `build_bench.ps1`, `build_multiarch.ps1`) are PowerShell-only.
- `build.sbt:142–175` (`nativeBuild` task) shells out to `build-windows-llvm.ps1` via `powershell -ExecutionPolicy Bypass -File …` — sbt will not produce native artifacts on non-Windows.
- Recommended: `git rm --cached src/main/native/build/*.{dll,lib,exp,exe}` + document the local build contract, or produce these artifacts from a reproducible CI step and host them as releases rather than committing binaries.

#### F14 — Reflection-based ONNX runtime bridge
- `src/main/scala/sicfun/holdem/provider/HoldemDdreOnnxRuntime.scala` (425 LOC) contains 24 reflection sites (`:247–250` resolve ONNX classes by string, `:252,253,257,272,285,319,327,335,340,356,362,369,375,382,423` resolve constructors/methods and `invoke` them).
- `onnxruntime 1.19.2` is a first-class compile dep at `build.sbt:16`. So the reflection is not classpath-isolation — it is likely deliberate decoupling so the DDRE synthetic path can run without ONNX on the runtime classpath, or a hedge against API drift.
- Cost: the Scala compiler cannot catch ONNX API changes. An `onnxruntime` upgrade silently breaks reflection lookups; failures appear only when DDRE fires (→ interacts with F11 in section 2: when a real DDRE model ships, this bridge is a landmine).
- Either (a) commit to an ORT-typed facade class behind a thin adapter, or (b) add a runtime self-test on startup that exercises the reflection chain with the smoke model. Today, no such self-test exists.

#### F15 — PowerShell-only operator surface
- 32 `.ps1` under `scripts/` vs 0 `.sh`. `build.sbt:142–175` invokes `powershell -ExecutionPolicy Bypass`. README is Windows-centric throughout (`-ExecutionPolicy Bypass`, NSSM, `.ps1` paths).
- Low urgency given the stated scope ("Not a production poker bot", live-table integration is explicitly out). If a Linux build ever becomes a goal, this is the whole-repo blocker.

#### F16 — `data/` policy vs practice
- `README.md:19` states: "`data/` is for generated runtimes, scratch output, local benchmark artifacts, and developer-local setup. Recreate those artifacts when needed instead of treating them as repository content."
- `git ls-files data/` returns:
  - `data/phase2-a3/2026-04-17-1601/a3-meta.txt`
  - `data/phase2-a3/2026-04-17-1601/comparison.txt`
  - `data/phase2-a4/2026-04-17-1740/a4-meta.txt` (status `AM` — modified after add)
  - `data/phase2-a4/2026-04-17-1740/comparison.txt`
- All staged today per the session `git status`. Either the README policy is stale, or these files should be moved out of `data/` (e.g., into `docs/ai/` alongside `AI_CONTEXT_ARCHIVE.md`).

#### F17 — System-property coupling
- 110 references to `sys.props` / `System.getProperty` / `sys.env` / `System.getenv` across ≥20 files (`src/main/scala/sicfun/holdem/web/HandHistoryReviewServer.scala`, `sicfun/core/HandEvaluator.scala`, `sicfun/holdem/gpu/*`, `sicfun/holdem/tablegen/*`, `sicfun/holdem/bench/tuner/*`, `sicfun/holdem/strategic/solver/*`, etc.).
- `HeadsUpGpuExactParityGate.buildSlice:117–125` toggles `sicfun.gpu.native.engine` and `sicfun.gpu.native.cuda.blockSize` between "cpu" and "cuda" in the same process — this is why the gate has to run as a separate `main`, not a test.
- `src/main/scala/sicfun/holdem/types/ScopedRuntimeProperties.scala` exists (1 hit), suggesting the team has already begun building a scoped-property abstraction. Check whether it is adopted broadly.
- Risk: any parallel JVM process mutating these keys (e.g., sbt test runners, `fork := true`) produces non-reproducible behavior. Test flakiness that disappears when tests run in isolation is the typical symptom.

#### F18 — Plan/spec doc sprawl
- `docs/superpowers/plans/` — 34 files, names like `2026-03-10-*`, `2026-04-16-*`. No `plans/README.md`, no status headers in filenames.
- `docs/superpowers/specs/` — 13 files, same shape.
- Risk is soft: onboarding cost, conflicting plans. Cheap fix is a front-matter `status: {draft|active|landed|abandoned}` convention + a `docs/superpowers/README.md` index.

#### F19 — Crash-dump clutter in repo root
- `hs_err_pid{33060,46832,53568,65404,69672,76052}.log`, `replay_pid46832.log`, `tmp-sbt-web-test.log` present in repo root.
- `.gitignore:125-127` covers `hs_err_pid*.log`, `replay_pid*.log` — none are tracked. Just a cleanup chore.

### 7.4 Revised overall priority table

Items sorted by priority after the expanded scope (top 5 only):

| Rank | # | Finding | Priority |
|:-:|---|---|:-:|
| 1 | **F12** | Gate FQCN mismatch — release pipeline + `sbt` gates throw `ClassNotFoundException` | **50** |
| 2 | F1 | `tablegen/` has zero tests | 24 |
| 3 | F2 | 643 `println`, no logging abstraction | 21 |
| 4 | F3 | `HandHistoryReviewServer.scala` 2,188-LOC monolith | 18 |
| 4 | F4 | `runtime/` largest package, test ratio 0.31 | 18 |
| 4 | F13 | 36 tracked Windows DLLs in repo | 18 |

F12 jumps to the top of the plan — Phase 0 now owns it.

### 7.5 Revised Phase 0 (hours, not days)

Add to the list in section 4:
- **F12**: fix the six gate FQCNs (two in `build.sbt`, two × two lines in `release-windows.ps1`, one in each of the two `scripts/gpu/*.ps1` files). **Precondition**: run one of the broken references first and capture the exception — so the fix lands with a reproduced failure in the commit message, not my static prediction.
- **F16**: decide whether the four `data/phase2-a*` files stay in `data/` (→ update README policy) or move (→ `git mv` to wherever phase rollup evidence actually belongs).
- **F19**: `rm` the `hs_err_pid*.log`, `replay_pid*.log`, `tmp-sbt-web-test.log` stragglers. One-time cleanup.
