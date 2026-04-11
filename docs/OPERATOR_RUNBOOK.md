# SICFUN Operator Runbook

This is the primary day-to-day operations guide for running, validating, and stress-testing SICFUN.

Scope note:
- This runbook covers simulator, benchmark, and research-harness workflows.
- It does not imply live table integration or production deployment.

Optional interactive launcher (menu for top 5 runbook actions):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/runbook.ps1
```

One-shot launcher mode:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/runbook.ps1 -Action quick-proof
```

Dry run preview (prints command without executing):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/runbook.ps1 -Action hall-max-autotune -WhatIf
```

## 1. Daily Start

Quick health check:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/prove-pipeline.ps1 -Quick
```

- Covers the core engine/runtime smoke path.
- Does not include the hand-history review end-to-end proof.

Full validation sweep:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/prove-pipeline.ps1
```

- Includes the hand-history review end-to-end proof: playing-hall export -> import -> analysis service -> async HTTP job completion.

## 2. Main Workload: Playing Hall

Single-process hall run (good for functional checks and controlled experiments):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run-playing-hall.ps1 `
  -Hands 1000000 `
  -TableCount 8 `
  -ReportEvery 50000 `
  -LearnEveryHands 0 `
  -SaveTrainingTsv false `
  -SaveDdreTrainingTsv false `
  -OutDir data/bench-hall-single
```

Maximum hardware saturation (recommended for long stress runs):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run-playing-hall-max.ps1 `
  -Hands 100000000 `
  -Workers 0 `
  -TableCountPerWorker 8 `
  -NativeProfile gpu `
  -ReportEvery 500000 `
  -LearnEveryHands 0 `
  -SaveTrainingTsv false `
  -SaveDdreTrainingTsv false `
  -JvmOption "-Xms2g" "-Xmx2g" `
  -OutDir data/bench-hall-max
```

Auto-tuned hardware run (recommended default when machine load is variable):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run-playing-hall-max.ps1 `
  -AutoTune `
  -AutoTuneHands 500000 `
  -AutoTuneProfiles auto,cpu,gpu `
  -AutoTuneWorkerCandidates 8,12,16,20,24 `
  -Hands 100000000 `
  -TableCountPerWorker 8 `
  -ReportEvery 500000 `
  -LearnEveryHands 0 `
  -SaveTrainingTsv false `
  -SaveDdreTrainingTsv false `
  -OutDir data/bench-hall-max
```

## 2A. Benchmark Control Notes

- Autotune cache files under `data/` are local runtime outputs, not repository source. Generate or refresh them on the machine you are benchmarking.
- Use `5000` or `10000` hands for exact-mode throughput comparisons. `1000`-hand runs were highly variable and often understated the long-run throughput of the retuned range kernel.
- Current long-run reference on this machine for `-Workers 1 -TableCountPerWorker 1 -HeroStyle adaptive -GtoMode exact -VillainStyle gto -BunchingTrials 80 -EquityTrials 700` is about `127.99-130.59 hands/s` at `5000` hands and `129.27 hands/s` at `10000` hands.
- Exact-GTO cache hit rate did not materially change in these longer runs (`~18.7%` to `19.1%`), so the improvement came from the native range path, not higher exact-solve cache reuse.
- If you need short-run smoke/control comparability, point `sicfun.gpu.range.autotune.cachePath` at a separately tuned short-run cache file instead of reusing a long-run profile.

Short-run cache override example:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run-playing-hall-max.ps1 `
  -Hands 1000 `
  -Workers 1 `
  -TableCountPerWorker 1 `
  -NativeProfile auto `
  -ReportEvery 1000 `
  -LearnEveryHands 0 `
  -SaveTrainingTsv false `
  -SaveDdreTrainingTsv false `
  -JvmOption "-Dsicfun.gpu.range.autotune.cachePath=data/headsup-range-autotune-short.properties" `
  -OutDir data/bench-hall-short-control
```

## 3. Hall Output Locations

`scripts/run-playing-hall.ps1` output:
- `<outDir>/hands.tsv`
- `<outDir>/learning.tsv`
- `<outDir>/training-selfplay.tsv` (if enabled)
- `<outDir>/ddre-training-selfplay.tsv` (if enabled)

`scripts/run-playing-hall-max.ps1` output:
- `<outDir>/run-*/aggregate-summary.txt`
- `<outDir>/run-*/worker-*/stdout.log`
- `<outDir>/run-*/worker-*/stderr.log`
- `<outDir>/autotune/autotune-results.tsv` (if `-AutoTune`)
- `<outDir>/autotune/autotune-selection.txt` (if `-AutoTune`)

## 4. DDRE Adapter Operations

Current DDRE status:
- `synthetic` is a heuristic scaffold for routing/fallback checks.
- Native DDRE CPU/GPU currently accelerate the same synthetic inference core, not a trained diffusion model.
- The checked-in ONNX smoke model only verifies adapter execution (`posterior = sqrt(prior)`); it is not a poker-quality model.
- Decision-driving ONNX requires a DDRE artifact directory whose metadata has passed the offline gate, unless you explicitly opt into experimental artifacts for tooling.

Generate ONNX smoke model:

```powershell
python scripts/generate-ddre-smoke-onnx.py --artifact-dir data/ddre-smoke-artifact
```

Run DDRE integration suite:

```powershell
sbt "testOnly sicfun.holdem.provider.HoldemDdreIntegrationTest"
```

Run DDRE adapter parity benchmark:

```powershell
sbt "runMain sicfun.holdem.bench.HoldemDdreParityBenchmark --modes=synthetic,onnx --referenceMode=synthetic --onnxArtifactDir=data/ddre-smoke-artifact --onnxAllowExperimental=true --warmupRuns=0 --measureRuns=2 --hypothesisCount=128 --maxL1Diff=1e-4 --maxAbsDiff=1e-5"
```

Run the DDRE offline validation gate against self-play data:

```powershell
sbt "runMain sicfun.holdem.provider.HoldemDdreOfflineGate --dataset=data/bench-hall-single/ddre-training-selfplay.tsv --artifactDir=data/ddre-smoke-artifact --minSamples=100 --maxMeanNll=8.0 --maxMeanKlVsBayes=8.0 --maxBlockerViolationRate=0.0 --maxFailureRate=0.0 --maxP95LatencyMillis=50.0"
```

## 5. Native Build/Runtime Operations

CPU native build:

```powershell
powershell -ExecutionPolicy Bypass -File src/main/native/build-windows-llvm.ps1
```

CUDA native build:

```powershell
powershell -ExecutionPolicy Bypass -File src/main/native/build-windows-cuda11.ps1
```

GPU build prerequisite checker:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ensure-gpu-build-prereqs.ps1
```

GPU build prerequisite auto-installer:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ensure-gpu-build-prereqs.ps1 -InstallMissing
```

- Checks the machine-wide prerequisites the CUDA DLL build actually needs:
  - Windows x64 host process
  - JDK with JNI headers
  - CUDA toolkit with `nvcc.exe`
  - Visual Studio Build Tools with `vcvars64.bat`
  - resolvable CUDA architecture via `-Arch`, `-Architectures`, env overrides, or `nvidia-smi`
- Auto-installs supported missing prerequisites with `winget`:
  - `Microsoft.OpenJDK.21`
  - `Nvidia.CUDA` version `11.8`
  - `Microsoft.VisualStudio.2022.BuildTools` with the C++ workload
- Auto-install requires an elevated PowerShell session because these are machine-wide installs.

Global GPU/native tuning pass (recommended default):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run-global-tuning.ps1 --targets=runtime
```

- Automatically builds missing runtime CUDA DLLs under `src/main/native/build/` before tuning:
  - `sicfun_gpu_kernel.dll`
  - `sicfun_postflop_cuda.dll`
- Auto-build uses `src/main/native/build-windows-cuda11.ps1`, which now:
  - discovers JDK headers from `-JavaHome`, `SICFUN_GPU_BUILD_JAVA_HOME`, `JAVA_HOME`, or `javac.exe` on `PATH`
  - discovers CUDA from `-CudaRoot`, `SICFUN_GPU_BUILD_CUDA_ROOT`, `CUDA_PATH`, or `nvcc.exe` on `PATH`
  - discovers `vcvars64.bat` from `-VcVars`, `SICFUN_GPU_BUILD_VCVARS`, or Visual Studio `vswhere`
  - auto-detects the local GPU compute capability from `nvidia-smi` on `PATH` or the default `NVSMI` install path unless you override `-Arch` or `-Architectures`
  - clamps unsupported auto-detected newer architectures to the highest CUDA 11 target and emits a warning
- Global tuning auto-build overrides:
  - `-Dsicfun.repo.root=<repo-root>` when you launch the tool from a subdirectory instead of the checkout root
  - `-Dsicfun.gpu.build.javaHome=<jdk>`
  - `-Dsicfun.gpu.build.cudaRoot=<cuda-root>`
  - `-Dsicfun.gpu.build.vcvars=<path-to-vcvars64.bat>`
  - `-Dsicfun.gpu.build.arch=<sm_xy>`
  - `-Dsicfun.gpu.build.architectures=<csv>`
- Auto-build currently supports Windows x64 hosts only because the repo build script resolves `vcvars64.bat` and produces x64 DLLs.
- To let the global tuning path auto-install missing machine prerequisites before building native DLLs, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run-global-tuning.ps1 -InstallMissingPrerequisites --targets=runtime
```

- Reuses existing persisted cache entries when the cache still matches the current hardware and native library identity.
- Re-runs only the stale or missing runtime tuners by default.
- Use `--targets=all` to include the research-only canonical exact tuner harnesses.
- Use `--force=true` to ignore caches and retune every selected target.

Windows portability proof for the operator path:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/prove-global-gpu-tuning-portability.ps1
```

- Runs from a temp working directory outside the repo root.
- Forces `sicfun_gpu_kernel.dll` and `sicfun_postflop_cuda.dll` to be missing first.
- Verifies the global tuning operator entrypoint auto-builds the native DLLs, then reaches backend/range/postflop tuning code instead of stopping at the missing-DLL gate.
- Requires the same Windows x64 + JDK + CUDA + Visual Studio prerequisites expected by `src/main/native/build-windows-cuda11.ps1`.

Heads-up range CUDA auto-tuner:

```powershell
sbt "runMain sicfun.holdem.bench.HeadsUpRangeGpuAutoTuner --heroes=256 --entriesPerHero=128 --trials=256 --warmupRuns=1 --runs=3 --cachePath=data/headsup-range-autotune.properties"
```

Postflop CUDA auto-tuner:

```powershell
sbt "runMain sicfun.holdem.bench.HoldemPostflopGpuAutoTuner --villains=1024 --trials=2000 --warmupRuns=1 --runs=3 --cachePath=data/postflop-autotune.properties"
```

## 5A. Hand-History Web Review

Source-mode launcher:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start-hand-history-web.ps1
```

Packaged release:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/release-hand-history-web.ps1
```

Start the packaged app:

```powershell
powershell -ExecutionPolicy Bypass -File dist/hand-history-web/bin/run-hand-history-web.ps1
```

Install the packaged app as a Windows service with NSSM:

```powershell
powershell -ExecutionPolicy Bypass -File dist/hand-history-web/bin/install-hand-history-web-service.ps1 `
  -NssmPath C:\tools\nssm\nssm.exe
```

Start the installed service and wait for readiness:

```powershell
powershell -ExecutionPolicy Bypass -File dist/hand-history-web/bin/start-hand-history-web-service.ps1
```

Drain and stop the service cleanly:

```powershell
powershell -ExecutionPolicy Bypass -File dist/hand-history-web/bin/drain-stop-hand-history-web-service.ps1
```

Uninstall the service:

```powershell
powershell -ExecutionPolicy Bypass -File dist/hand-history-web/bin/uninstall-hand-history-web-service.ps1
```

Operator notes:
- The packaged release serves the upload UI from `dist/hand-history-web/static`.
- The packaged release writes a config template to `dist/hand-history-web/conf/hand-history-web.env`. Keep long-lived runtime settings there instead of baking them into a service command line.
- `bin/run-hand-history-web.ps1` now loads `conf/hand-history-web.env` by default. Override with `-ConfigFile <path>` or `CONFIG_FILE=<path>` when you need a different config file.
- The source and packaged launchers bind to `127.0.0.1` by default. Pass `-Host 0.0.0.0` only if you intentionally want network exposure.
- Optional built-in HTTP Basic auth now protects `/`, `/api/analyze-hand-history`, and `/api/analyze-hand-history/jobs/{id}` while leaving `/api/health` and `/api/ready` open for service managers and probes. Set `BASIC_AUTH_USER` and `BASIC_AUTH_PASSWORD` in `conf/hand-history-web.env` or via process env. Prefer config/env over CLI flags so credentials do not appear in the Java command line.
- Platform-user auth is available as a separate mode. Set `USER_STORE_PATH` to enable persistent local users, profile defaults, browser sessions, and per-user job ownership. Leave `BASIC_AUTH_*` unset when using platform-user auth; the modes are mutually exclusive.
- Google OIDC can be enabled on top of platform-user auth with `GOOGLE_OIDC_CLIENT_ID`, `GOOGLE_OIDC_CLIENT_SECRET`, and `GOOGLE_OIDC_REDIRECT_URI`. For HTTPS deployments, also set `USER_AUTH_COOKIE_SECURE=true`.
- In-process rate limiting now caps the expensive API routes. Use `RATE_LIMIT_SUBMITS_PER_MINUTE` and `RATE_LIMIT_STATUS_PER_MINUTE` to tune submit and job-status polling caps independently; set either to `0` to disable that limiter.
- By default the limiter buckets by the remote socket address. If you deploy behind a trusted reverse proxy, set `RATE_LIMIT_CLIENT_IP_HEADER` to a proxy-populated single-value client-IP header such as `X-Real-IP`. Same-host loopback proxies are trusted automatically; for proxies on other hosts, also set `RATE_LIMIT_TRUSTED_PROXY_IPS` to a comma-separated list of exact proxy peer IP literals. Do not enable the header knob on a directly exposed app because clients can spoof those headers.
- The packaged launcher now requires `java` on `PATH`, checks `java -version` before startup, accepts Java 17+, and recommends JDK 21 for operator parity.
- The packaged service helper scripts assume a Windows host and use NSSM as the service wrapper. Service install/uninstall requires an elevated PowerShell session.
- The service install script configures stdout/stderr capture under `dist/hand-history-web/logs/` and enables basic Windows service restart-on-failure recovery.
- Uploads are accepted quickly and processed as background jobs; the page polls `/api/analyze-hand-history/jobs/{id}` until the review finishes.
- Analysis admission is now bounded. Use `-MaxConcurrentJobs`, `-MaxQueuedJobs`, and `-ShutdownGraceMs` or the matching `MAX_CONCURRENT_JOBS`, `MAX_QUEUED_JOBS`, and `SHUTDOWN_GRACE_MS` environment variables to control saturation and shutdown drain behavior.
- Use `-AnalysisTimeoutMs` or `ANALYSIS_TIMEOUT_MS` to cap a single analysis job. `0` disables the timeout, but the deployment-safe default is a bounded run so one stuck review cannot pin the worker pool indefinitely.
- `SHUTDOWN_GRACE_MS` is tracked in milliseconds, but the underlying HTTP listener drains in whole-second steps. Sub-second values round up when the listener is stopping.
- `/api/health` is the liveness/metrics endpoint. It stays `200` while the process is up and now reports readiness summary, auth mode, model mode, upload limit, analysis timeout, submit/status rate-limit settings, the trusted client-IP source used for rate limiting, queue limits, queued jobs, running jobs, timed-out workers still unwinding, and retained terminal-job count in addition to `ok=true`.
- `/api/ready` is the readiness endpoint for reverse proxies / service managers. It returns `200` only when the instance is accepting new analysis work and switches to `503` when the queue is saturated, the instance is draining, or a timed-out worker is still unwinding. The response also reports the configured `analysisTimeoutMs`, `timedOutWorkersInFlight`, auth mode, submit/status rate-limit settings, and the trusted client-IP source used for rate limiting.
- Use `-DrainSignalFile <path>` or `DRAIN_SIGNAL_FILE=<path>` when you want external deployment tooling to mark the instance unready before shutdown. While that file exists, `/api/ready` returns `503` and new `POST /api/analyze-hand-history` submissions are rejected, but health checks and in-flight job polling still work.
- `bin/drain-stop-hand-history-web-service.ps1` turns on the configured drain signal, waits for readiness to fail plus in-memory jobs and in-flight HTTP requests to drain to zero, then stops the Windows service.
- Runtime state is still in-memory only. Queued and running review jobs are lost on process restart, and completed-job status is retained for only 15 minutes.
- Under platform-user auth, account/profile data persists in `USER_STORE_PATH`, but browser sessions and in-flight review jobs remain in-memory only. A restart signs users out and drops queued/running jobs.
- The raw server now emits baseline security headers (`Content-Security-Policy`, `X-Content-Type-Options`, `X-Frame-Options`, and `Referrer-Policy`), can enforce built-in Basic auth, and applies a best-effort in-process rate limiter on the expensive API routes, but that is still not a substitute for TLS termination or edge rate limiting.
- Do not expose the raw app directly to the public internet without HTTPS in front of it. Built-in Basic auth and the in-process limiter help with access control and abuse containment, but you still want a reverse proxy / ingress layer for TLS termination, network policy, and stronger rate limiting.
- `scripts/release-hand-history-web.ps1` validates the required static assets and smoke-checks auth-enabled `/`, `/api/health`, `/api/ready`, async `/api/analyze-hand-history`, drain-mode readiness, and oversized-upload rejection before declaring the build ready.
- The web server supports `CONFIG_FILE`, `HOST`, `PORT`, `STATIC_DIR`, `MODEL_DIR`, `MAX_UPLOAD_BYTES`, `ANALYSIS_TIMEOUT_MS`, `MAX_CONCURRENT_JOBS`, `MAX_QUEUED_JOBS`, `SHUTDOWN_GRACE_MS`, `RATE_LIMIT_SUBMITS_PER_MINUTE`, `RATE_LIMIT_STATUS_PER_MINUTE`, `RATE_LIMIT_CLIENT_IP_HEADER`, `RATE_LIMIT_TRUSTED_PROXY_IPS`, `DRAIN_SIGNAL_FILE`, `BASIC_AUTH_USER`, and `BASIC_AUTH_PASSWORD` environment-variable overrides in addition to CLI flags.

## 6. Troubleshooting

Classpath appears stale or Java run behaves like old code:
- Re-run with `-RefreshClasspath` on `run-playing-hall.ps1` or `run-playing-hall-max.ps1`.

`sbt` lock/server issues:
- Kill stale sbt Java processes:

```powershell
Get-CimInstance Win32_Process -Filter "Name='java.exe'" |
  Where-Object { $_.CommandLine -match "sbt|sbt-launch" } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
```

GPU profile underperforms:
- Use `-AutoTune` and include both `auto` and `cpu` in `-AutoTuneProfiles`.
- Reduce `-Workers` or `-TableCountPerWorker` if GPU is oversubscribed.

Short hall benchmark regressed after retuning the range GPU cache:
- For `1000`-hand smoke/control runs, use a dedicated short-run cache file instead of a long-run tuned profile.
- For `5000+` hand long exact-mode runs, benchmark with a cache that was tuned on the same machine/profile you plan to use.

## 7. Minimal Command Set

The minimal set most operators need:

```powershell
# 1) quick health
powershell -ExecutionPolicy Bypass -File scripts/prove-pipeline.ps1 -Quick

# 2) auto-tuned max run
powershell -ExecutionPolicy Bypass -File scripts/run-playing-hall-max.ps1 -AutoTune -AutoTuneHands 500000 -AutoTuneProfiles auto,cpu,gpu -Hands 100000000 -TableCountPerWorker 8 -LearnEveryHands 0 -SaveTrainingTsv false -SaveDdreTrainingTsv false -OutDir data/bench-hall-max

# 3) inspect result
Get-Content data/bench-hall-max/autotune/autotune-selection.txt
Get-Content data/bench-hall-max/run-*/aggregate-summary.txt
```

## 8. Optional AI Sidecars

Optional delegated analysis/review helpers for read-heavy tasks:

Unified health check:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai-minion.ps1 -Action doctor
```

One-time auth:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai-minion.ps1 -Action auth -Provider gemini
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai-minion.ps1 -Action auth -Provider claude
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai-minion.ps1 -Action auth -Provider gpt
```

Manual/no-browser auth is available for Gemini and GPT:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai-minion.ps1 -Action auth -Provider gemini -NoBrowser
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai-minion.ps1 -Action auth -Provider gpt -NoBrowser
```

Read-only delegation example:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai-minion.ps1 `
  -Action delegate `
  -Provider gpt `
  -Mode analysis `
  -Task "Summarize the latest exact-mode hall benchmark deltas." `
  -ContextPath docs/OPERATOR_RUNBOOK.md,ROADMAP.md `
  -OutputFormat text
```

Notes:

- Shared sidecar rules live in `AI_ENTRYPOINT.md`, with provider-specific overlays in `GEMINI.md`, `CLAUDE.md`, and `GPT.md`.
- Gemini keeps its provider-specific wrapper at `scripts/gemini-sidecar.ps1`.
- Claude login is browser-based through `claude auth login`.
- GPT uses the official OpenAI Codex CLI and ChatGPT/device auth.

For setup details and more examples, see `docs/AI_MINIONS.md` and `docs/GEMINI_MINION.md`.
