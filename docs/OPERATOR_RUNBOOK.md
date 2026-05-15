# SICFUN Operator Runbook

This is the primary day-to-day operations guide for running, validating, and stress-testing SICFUN.

Scope note:
- This runbook covers simulator, benchmark, and research-harness workflows.
- It does not imply live table integration or production deployment.

Optional interactive launcher (menu for top 5 runbook actions):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/validation/runbook.ps1
```

One-shot launcher mode:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/validation/runbook.ps1 -Action quick-proof
```

Dry run preview (prints command without executing):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/validation/runbook.ps1 -Action hall-max-autotune -WhatIf
```

## 1. Daily Start

Quick health check:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/validation/prove-pipeline.ps1 -Quick
```

- Covers the core engine/runtime smoke path.
- Does not include the hand-history review end-to-end proof.

Full validation sweep:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/validation/prove-pipeline.ps1
```

- Includes the hand-history review end-to-end proof: playing-hall export -> import -> analysis service -> async HTTP job completion.

## 1A. Test Suite Quirks

- `scripts/validation/prove-pipeline.ps1` is the supported proof path. It runs a pinned suite list via `sbt testOnly ...` with retries instead of a single aggregated `sbt test`.
- Full aggregated `sbt test` can still show order- or timing-dependent failures on the current machine even when the failing suite passes in isolation. Re-run the failing suite directly before treating that result as a real regression.

## 2. Main Workload: Playing Hall

Single-process hall run (good for functional checks and controlled experiments):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/match/run-playing-hall.ps1 `
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
powershell -ExecutionPolicy Bypass -File scripts/match/run-playing-hall-max.ps1 `
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
powershell -ExecutionPolicy Bypass -File scripts/match/run-playing-hall-max.ps1 `
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
powershell -ExecutionPolicy Bypass -File scripts/match/run-playing-hall-max.ps1 `
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

`scripts/match/run-playing-hall.ps1` output:
- `<outDir>/hands.tsv`
- `<outDir>/learning.tsv`
- `<outDir>/training-selfplay.tsv` (if enabled)
- `<outDir>/ddre-training-selfplay.tsv` (if enabled)

`scripts/match/run-playing-hall-max.ps1` output:
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
powershell -ExecutionPolicy Bypass -File scripts/gpu/ensure-gpu-build-prereqs.ps1
```

GPU build prerequisite auto-installer:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/gpu/ensure-gpu-build-prereqs.ps1 -InstallMissing
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
powershell -ExecutionPolicy Bypass -File scripts/gpu/run-global-tuning.ps1 --targets=runtime
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
powershell -ExecutionPolicy Bypass -File scripts/gpu/run-global-tuning.ps1 -InstallMissingPrerequisites --targets=runtime
```

- Reuses existing persisted cache entries when the cache still matches the current hardware and native library identity.
- Re-runs only the stale or missing runtime tuners by default.
- Use `--targets=all` to include the research-only canonical exact tuner harnesses.
- Use `--force=true` to ignore caches and retune every selected target.

Windows portability proof for the operator path:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/gpu/prove-global-gpu-tuning-portability.ps1
```

- Runs from a temp working directory outside the repo root.
- Forces `sicfun_gpu_kernel.dll` and `sicfun_postflop_cuda.dll` to be missing first.
- Verifies the global tuning operator entrypoint auto-builds the native DLLs, then reaches backend/range/postflop tuning code instead of stopping at the missing-DLL gate.
- Requires the same Windows x64 + JDK + CUDA + Visual Studio prerequisites expected by `src/main/native/build-windows-cuda11.ps1`.

Heads-up range CUDA auto-tuner:

```powershell
sbt "runMain sicfun.holdem.bench.tuner.HeadsUpRangeGpuAutoTuner --heroes=256 --entriesPerHero=128 --trials=256 --warmupRuns=1 --runs=3 --cachePath=data/headsup-range-autotune.properties"
```

Postflop CUDA auto-tuner:

```powershell
sbt "runMain sicfun.holdem.bench.tuner.HoldemPostflopGpuAutoTuner --villains=1024 --trials=2000 --warmupRuns=1 --runs=3 --cachePath=data/postflop-autotune.properties"
```

## 5A. Hand-History Web Review

Source-mode launcher:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/packaged-hand-history-web/start-hand-history-web.ps1
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

Click-to-run installer variant:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/release-hand-history-web-installer.ps1
```

- Wraps `scripts/release-hand-history-web.ps1` then adds a jlink-trimmed JDK under `runtime/bin/java.exe`, patches the launcher to prefer the embedded runtime, emits `Setup.cmd` at the bundle root, regenerates `manifest.sha256`, and zips to `dist/hand-history-web-<version>.zip` with a sibling `.sha256`.
- Use this variant when shipping to hosts that may not have Java pre-installed; use the base `release-hand-history-web.ps1` for hosts that already have Java 17+ on `PATH`.
- After unzip, the operator double-clicks `Setup.cmd` to verify the manifest and start the service. The default JDK source is Eclipse Temurin 25 LTS; override with `-JdkPath <abs-path>`.
- `Setup.cmd` calls `bin/verify-if-needed.ps1` instead of running the full SHA-256 sweep on every launch. That helper writes a `.manifest-verified` marker at the bundle root after a successful check and short-circuits subsequent launches as long as the marker's mtime is at least as recent as `manifest.sha256`. Re-extracting the ZIP advances `manifest.sha256`'s mtime and forces a fresh verify. Delete the marker manually to force a re-check.
- Step 6.5 of the installer boots the patched launcher on `127.0.0.1:18081`, asserts the running `java.exe` is the embedded runtime (not PATH-Java), polls `/api/health`, exercises the static index + Cache-Control + ETag, and runs a full Playing Hall job through to DELETE-after-completed. Catches jlink module-set gaps that the base script's smoke (which uses full-classpath JDK) cannot.
- Step 6.6 follows with a short stand-alone smoke of `bin/launch-with-log.ps1` on `127.0.0.1:18082`. It polls `/api/ready`, checks `/api/health` returns `ok=true` with the expected port, then tears down and asserts that `logs/setup-launcher-*.log` was created with non-empty content and contains the launcher's `Using Java runtime` preamble. Catches wrapper regressions (StreamWriter open failure, dropped tee output, broken `$LASTEXITCODE` plumbing) that would otherwise surface only at customer first-run via Setup.cmd. The transient log is cleaned up before Step 7 zips, so the shipped bundle is fresh.
- `Setup.cmd` invokes `bin/launch-with-log.ps1` rather than `bin/run-hand-history-web.ps1` directly. The wrapper tees the launcher's stdout/stderr to `logs/setup-launcher-<timestamp>.log` (UTF-8) so the customer's window can be closed without losing the startup banner or any crash output. The verifier skips files under `logs/` so these runtime captures don't trip integrity checks; they're not part of the manifest and aren't deleted on re-extract. Operators can archive or rotate them out-of-band.

Operator notes:
- The packaged release serves the upload UI from `dist/hand-history-web/static`.
- The packaged release now includes a handoff guide at `dist/hand-history-web/README.md`. Give operators the packaged directory, not repo-only docs.
- The packaged release writes a config template to `dist/hand-history-web/conf/hand-history-web.env`. Keep long-lived runtime settings there instead of baking them into a service command line.
- Run `powershell -ExecutionPolicy Bypass -File dist/hand-history-web/bin/verify-release-manifest.ps1` after copying the bundle to a target machine to confirm it still matches `manifest.sha256`.
- `bin/run-hand-history-web.ps1` now loads `conf/hand-history-web.env` by default. Override with `-ConfigFile <path>` or `CONFIG_FILE=<path>` when you need a different config file.
- The source and packaged launchers bind to `127.0.0.1` by default. Pass `-Host 0.0.0.0` only if you intentionally want network exposure.
- Optional built-in HTTP Basic auth now protects `/`, `/api/analyze-hand-history`, `/api/analyze-hand-history/jobs/{id}`, `/api/playing-hall`, and `/api/playing-hall/jobs/{id}` (GET/DELETE) while leaving `/api/health` and `/api/ready` open for service managers and probes. Set `BASIC_AUTH_USER` and `BASIC_AUTH_PASSWORD` in `conf/hand-history-web.env` or via process env. Prefer config/env over CLI flags so credentials do not appear in the Java command line.
- To reduce accidental exposure, non-loopback binds now require `BASIC_AUTH_*`, `USER_STORE_PATH`, or an explicit `ALLOW_UNAUTHENTICATED_PUBLIC_BIND=true` / `--allowUnauthenticatedPublicBind=true` override for a trusted private network.
- Platform-user auth is available as a separate mode. Set `USER_STORE_PATH` to enable persistent local users, profile defaults, browser sessions, and per-user job ownership. Leave `BASIC_AUTH_*` unset when using platform-user auth; the modes are mutually exclusive.
- Non-loopback platform-user auth now also requires `USER_AUTH_COOKIE_SECURE=true` unless you explicitly set `ALLOW_INSECURE_USER_AUTH=true` / `--allowInsecureUserAuth=true` for trusted private-network testing. When `USER_AUTH_COOKIE_SECURE=true` the cookie name carries the RFC 6265 `__Host-` prefix (`__Host-sicfun_session`) so browsers block a sibling subdomain from overwriting it; in insecure mode the name stays `sicfun_session` because the prefix would make the cookie unusable over plain HTTP.
- Google OIDC can be enabled on top of platform-user auth with `GOOGLE_OIDC_CLIENT_ID`, `GOOGLE_OIDC_CLIENT_SECRET`, and `GOOGLE_OIDC_REDIRECT_URI`. On non-loopback binds, the redirect URI must use `https://...` unless you explicitly set the same insecure-user-auth override for trusted private-network testing.
- In-process rate limiting now caps the expensive API routes. Use `RATE_LIMIT_SUBMITS_PER_MINUTE`, `RATE_LIMIT_STATUS_PER_MINUTE`, and `RATE_LIMIT_AUTH_PER_MINUTE` to tune submit, job-status polling, and auth (register + login) caps independently; set any to `0` to disable that bucket. The auth bucket throttles credential-stuffing attempts against `/api/auth/register` and `/api/auth/login`.
- Auth events emit structured log lines at INFO (success/expected) and WARN (failure): `auth.login.success email=… remote=…`, `auth.login.failure email=… remote=… reason=…`, `auth.register.success`, `auth.register.failure`, `auth.logout`, `auth.oidc.start provider=… remote=…`, `auth.oidc.start.failure provider=… remote=… reason=…`, `auth.oidc.success`, `auth.oidc.failure`. Tail `logs/*.stdout.log` / `logs/*.stderr.log` (or pipe to your log aggregator) and alert on a high WARN rate from a single `remote=` to spot brute-force attempts. Success lines emit the **canonical** (normalized) email from the stored record so all events for one user grep identically; failure lines emit the **submitted** email so unusual case/whitespace probes are visible in the form the attacker typed. The `auth.oidc.start` entry lets you correlate a click on "Sign in with Google" with the eventual `auth.oidc.success` / `auth.oidc.failure` callback — a stream of starts with no callbacks usually means the provider redirect is misconfigured or a script is exercising the start path.
- The `remote=` value is the **resolved client identity** in audit lines (auth.*, request unauthorized, request forbidden), not the raw TCP peer. When `RATE_LIMIT_CLIENT_IP_HEADER` is set and the peer is loopback or in `RATE_LIMIT_TRUSTED_PROXY_IPS`, `remote=` shows the IP the trusted header resolves to (no port — forwarded-IP headers don't carry one). Otherwise it's the direct peer as `host:port` with IPv6 bracketed per RFC 3986 §3.2.2. This means a behind-a-proxy deployment shows real client IPs in audit lines instead of every request looking like it came from the proxy, and the value matches the rate-limit `client=` field so a single grep correlates both.
- OIDC `/start` issues a short-lived `sicfun_oidc_state` cookie (`HttpOnly`, `SameSite=Lax`, 10-minute Max-Age, `__Host-` prefix and `Secure` when `USER_AUTH_COOKIE_SECURE=true`). `/callback` requires the cookie value to match the URL `state` parameter before exchanging the authorization code — an OAuth 2.0 BCP covert-redirect / login-CSRF mitigation that prevents an attacker who completes their own authorization from forwarding the resulting URL to a victim. Failed checks log as `auth.oidc.failure provider=… reason=missing_state_cookie` or `reason=state_cookie_mismatch`; a burst usually means a misbehaving JS client clearing cookies between `/start` and `/callback`, not a real attack — but a spike correlated with one `remote=` is worth investigating.
- CSRF rejections on state-changing JSON routes (logout, profile, analyze-hand-history, playing-hall) log as `request forbidden path=… remote=… email=… reason=csrf-missing-or-invalid` at WARN. A burst from one `remote=` against many endpoints is a likely automated probe; the same from a known user usually means their JS frontend lost the cookie mid-session and needs to reload.
- Unauthenticated requests to protected paths log as `request unauthorized path=… remote=… reason=…` at WARN. The reason names the auth-stage failure (`missing_authorization`, `unsupported_authorization_scheme`, `malformed_authorization`, `invalid_credentials`, or `session-missing-or-invalid`).
- Rate-limit rejections log as `request rate limited path=… client=… bucket=… limitPerMinute=… retryAfterMs=…` at WARN. The `client=` value is either `user:<userId>` for an authenticated principal, `header:<ip>` when a trusted forwarded-IP header is configured, or `remote:<addr>` otherwise. `bucket=` is one of `submit` / `job-status` / `auth`.
- By default the limiter buckets by the remote socket address. If you deploy behind a trusted reverse proxy, set `RATE_LIMIT_CLIENT_IP_HEADER` to a proxy-populated single-value client-IP header such as `X-Real-IP`. Same-host loopback proxies are trusted automatically; for proxies on other hosts, also set `RATE_LIMIT_TRUSTED_PROXY_IPS` to a comma-separated list of exact proxy peer IP literals. Do not enable the header knob on a directly exposed app because clients can spoof those headers.
- The packaged launcher now requires `java` on `PATH`, checks `java -version` before startup, accepts Java 17+, and recommends JDK 21 for operator parity.
- The packaged service helper scripts assume a Windows host and use NSSM as the service wrapper. Service install/uninstall requires an elevated PowerShell session.
- The service install script configures stdout/stderr capture under `dist/hand-history-web/logs/` and enables basic Windows service restart-on-failure recovery.
- `bin/start-hand-history-web-service.ps1` now fails fast if the service stops during startup and includes the latest readiness/health summary plus recent stdout/stderr tail when readiness does not come up cleanly.
- `bin/drain-stop-hand-history-web-service.ps1` now includes the last readiness/health probe summary when drain mode does not flip or jobs do not fully drain before the forced stop.
- Uploads are accepted quickly and processed as background jobs; the page polls `/api/analyze-hand-history/jobs/{id}` (or `/api/playing-hall/jobs/{id}` for hall simulations) until the work finishes. Playing Hall jobs additionally accept `DELETE` on the job URL for cooperative cancellation; the server returns `200` with `status=cancelled` while running, `409` if already terminal, `404` if unknown.
- Analysis admission is now bounded. Use `-MaxConcurrentJobs`, `-MaxQueuedJobs`, and `-ShutdownGraceMs` or the matching `MAX_CONCURRENT_JOBS`, `MAX_QUEUED_JOBS`, and `SHUTDOWN_GRACE_MS` environment variables to control saturation and shutdown drain behavior.
- Use `-AnalysisTimeoutMs` or `ANALYSIS_TIMEOUT_MS` to cap a single analysis job. `0` disables the timeout, but the deployment-safe default is a bounded run so one stuck review cannot pin the worker pool indefinitely.
- `SHUTDOWN_GRACE_MS` is tracked in milliseconds, but the underlying HTTP listener drains in whole-second steps. Sub-second values round up when the listener is stopping.
- `/api/health` is the liveness/metrics endpoint. It accepts `GET`, `HEAD`, and `OPTIONS`; everything else returns `405` with `Allow: GET, HEAD, OPTIONS`. It stays `200` while the process is up and now reports readiness summary, auth mode, model mode, upload limit, analysis timeout, submit/status/auth rate-limit settings, the trusted client-IP source used for rate limiting, queue limits, queued jobs, running jobs, timed-out workers still unwinding, and retained terminal-job count (analysis + Playing Hall, summed) in addition to `ok=true`.
- `/api/ready` is the readiness endpoint for reverse proxies / service managers. Same verb set as `/api/health`. It returns `200` only when the instance is accepting new analysis work and switches to `503` when the queue is saturated, the instance is draining, or a timed-out worker is still unwinding. The response also reports the configured `analysisTimeoutMs`, `timedOutWorkersInFlight`, auth mode, submit/status/auth rate-limit settings, and the trusted client-IP source used for rate limiting.
- Use `-DrainSignalFile <path>` or `DRAIN_SIGNAL_FILE=<path>` when you want external deployment tooling to mark the instance unready before shutdown. While that file exists, `/api/ready` returns `503` and new `POST /api/analyze-hand-history` and `POST /api/playing-hall` submissions are rejected, but health checks and in-flight job polling still work.
- `bin/drain-stop-hand-history-web-service.ps1` turns on the configured drain signal, waits for readiness to fail plus in-memory jobs and in-flight HTTP requests to drain to zero, then stops the Windows service.
- Runtime state is still in-memory only. Queued and running review jobs are lost on process restart, and completed-job status is retained for only 15 minutes.
- Under platform-user auth, account/profile data persists in `USER_STORE_PATH`, but browser sessions and in-flight review jobs remain in-memory only. A restart signs users out and drops queued/running jobs.
- The raw server now emits baseline security headers (`Content-Security-Policy`, `Permissions-Policy`, `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Cross-Origin-Opener-Policy`, `Cross-Origin-Resource-Policy`, `X-Robots-Tag`), can enforce built-in Basic auth, and applies a best-effort in-process rate limiter on the expensive API routes (submit / status / auth buckets), but that is still not a substitute for TLS termination or edge rate limiting.
- Static assets are served with weak `ETag` headers and path-aware `Cache-Control`: `vendor/*` files get `public, max-age=31536000` (vendored libs are pinned per filename), all other static assets get `public, max-age=0, must-revalidate`. Browsers revalidate with `If-None-Match` or `If-Modified-Since`; matching validators get `304 Not Modified` with no body. API responses still emit `Cache-Control: no-store`. A reverse proxy with caching enabled (e.g., nginx with proxy_cache_path) can respect these directives directly; do not strip them.
- Responses for compressible types (`text/*`, `application/javascript`, `application/json`, `image/svg+xml`) carry `Vary: Accept-Encoding` and are gzipped on the wire when the client sends `Accept-Encoding: gzip`. The wire savings are ~70% on the chart-stack cold load. The static handler emits a `-gz` ETag suffix on compressed variants so caches that key only on ETag (ignoring Vary) don't serve the wrong encoding. The origin honors `gzip;q=0` to opt out per RFC 7231 sec 5.3.4.
- Every response also carries a `Permissions-Policy` header that explicitly denies camera, microphone, geolocation, payment, USB, motion sensors, display capture, and FLoC. Defense-in-depth alongside CSP; reverse proxies that strip these weaken the browser-side protection.
- Do not expose the raw app directly to the public internet without HTTPS in front of it. Built-in Basic auth and the in-process limiter help with access control and abuse containment, but you still want a reverse proxy / ingress layer for TLS termination, network policy, and stronger rate limiting.
- `scripts/release-hand-history-web.ps1` validates the required static assets, asserts that every `src=`/`href=` reference in `index.html` resolves to a file under the static root, then smoke-checks packaged fail-closed non-loopback config rejection, auth-enabled `/` with security-header presence and weak-ETag emission, `/api/health`, `/api/ready`, async `/api/analyze-hand-history`, trusted-header submit rate limiting, the full `/api/playing-hall` lifecycle (POST auth gating, GET poll, DELETE-after-completed → 409), drain-mode readiness rejecting both analysis and Playing Hall submissions, the chart-stack assets (`site-charts.js`, `vendor/uPlot.iife.min.js`, `vendor/uPlot.min.css`) with correct Content-Type plus 304 revalidation via `If-None-Match`, oversized-upload rejection, and packaged manifest verification before declaring the build ready. The installer variant adds Step 6.5 which repeats the basic smoke against the patched bundle through the embedded jlink runtime.
- The web server supports `CONFIG_FILE`, `HOST`, `PORT`, `STATIC_DIR`, `MODEL_DIR`, `MAX_UPLOAD_BYTES`, `ANALYSIS_TIMEOUT_MS`, `PLAYING_HALL_TIMEOUT_MS`, `MAX_CONCURRENT_JOBS`, `MAX_QUEUED_JOBS`, `SHUTDOWN_GRACE_MS`, `RATE_LIMIT_SUBMITS_PER_MINUTE`, `RATE_LIMIT_STATUS_PER_MINUTE`, `RATE_LIMIT_CLIENT_IP_HEADER`, `RATE_LIMIT_TRUSTED_PROXY_IPS`, `DRAIN_SIGNAL_FILE`, `BASIC_AUTH_USER`, `BASIC_AUTH_PASSWORD`, `USER_STORE_PATH`, `ALLOW_UNAUTHENTICATED_PUBLIC_BIND`, and `ALLOW_INSECURE_USER_AUTH` environment-variable overrides in addition to CLI flags. The `MAX_CONCURRENT_JOBS` and `MAX_QUEUED_JOBS` budgets are shared across both job stores (analysis and Playing Hall).

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
powershell -ExecutionPolicy Bypass -File scripts/validation/prove-pipeline.ps1 -Quick

# 2) auto-tuned max run
powershell -ExecutionPolicy Bypass -File scripts/match/run-playing-hall-max.ps1 -AutoTune -AutoTuneHands 500000 -AutoTuneProfiles auto,cpu,gpu -Hands 100000000 -TableCountPerWorker 8 -LearnEveryHands 0 -SaveTrainingTsv false -SaveDdreTrainingTsv false -OutDir data/bench-hall-max

# 3) inspect result
Get-Content data/bench-hall-max/autotune/autotune-selection.txt
Get-Content data/bench-hall-max/run-*/aggregate-summary.txt
```

## 8. Optional AI Sidecars

Optional delegated analysis/review helpers for read-heavy tasks:

Unified health check:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai/ai-minion.ps1 -Action doctor
```

One-time auth:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai/ai-minion.ps1 -Action auth -Provider gemini
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai/ai-minion.ps1 -Action auth -Provider claude
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai/ai-minion.ps1 -Action auth -Provider gpt
```

Manual/no-browser auth is available for Gemini and GPT:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai/ai-minion.ps1 -Action auth -Provider gemini -NoBrowser
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai/ai-minion.ps1 -Action auth -Provider gpt -NoBrowser
```

Read-only delegation example:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai/ai-minion.ps1 `
  -Action delegate `
  -Provider gpt `
  -Mode analysis `
  -Task "Summarize the latest exact-mode hall benchmark deltas." `
  -ContextPath docs/OPERATOR_RUNBOOK.md,ROADMAP.md `
  -OutputFormat text
```

Notes:

- Shared sidecar rules live in `AI_ENTRYPOINT.md`, with provider-specific overlays in `GEMINI.md`, `CLAUDE.md`, and `GPT.md`.
- Gemini keeps its provider-specific wrapper at `scripts/ai/gemini-sidecar.ps1`.
- Claude login is browser-based through `claude auth login`.
- GPT uses the official OpenAI Codex CLI and ChatGPT/device auth.

For setup details and more examples, see `docs/ai/AI_MINIONS.md` and `docs/ai/GEMINI_MINION.md`.
