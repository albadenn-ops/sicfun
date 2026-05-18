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

For a packaged-release-oriented overview (reverse proxy guidance, HTTP endpoint reference, configuration knob descriptions), see [`docs/HAND_HISTORY_WEB_DEPLOYMENT.md`](HAND_HISTORY_WEB_DEPLOYMENT.md). The packaged release ships that document as the bundle root's `README.md` so customers reach it without needing the source tree. This section focuses on the source-mode operator-on-the-host workflow: launchers, service install, log triage, and the deeper observability + security notes the deployment guide summarizes but the runbook owns.

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
- Platform-user auth is available as a separate mode. Set `USER_STORE_PATH` to enable persistent local users, profile defaults, browser sessions, and per-user job ownership. The file is created atomically on the first successful registration -- no pre-creation needed -- via a `platform-users-*.json.tmp` temp file in the same parent directory + an `ATOMIC_MOVE` rename onto the target (or a `REPLACE_EXISTING` fallback on filesystems that don't support atomic move, mostly some network-mounted exotic FS). The parent directory must exist and be writable by the service account. A hard-kill (SIGKILL, OOM, power loss) during a write may leave an orphan `platform-users-*.json.tmp` file in the user-store directory -- harmless to delete; the JVM's normal finally-cleanup removes them on graceful errors. Startup fails fast with `user store at <path> is unreadable: <ujson parse error>` if the file is present but unparseable; prefer restoring from a known-good backup over the "remove to start fresh" suggestion the error itself includes, which drops every stored user. The JSON content carries a top-level `"version": 1` field for future schema migrations (the server currently ignores it on read). Leave `BASIC_AUTH_*` unset when using platform-user auth; the modes are mutually exclusive.
- Non-loopback platform-user auth now also requires `USER_AUTH_COOKIE_SECURE=true` unless you explicitly set `ALLOW_INSECURE_USER_AUTH=true` / `--allowInsecureUserAuth=true` for trusted private-network testing. When `USER_AUTH_COOKIE_SECURE=true` the cookie name carries the RFC 6265 `__Host-` prefix (`__Host-sicfun_session`) so browsers block a sibling subdomain from overwriting it; in insecure mode the name stays `sicfun_session` because the prefix would make the cookie unusable over plain HTTP.
- Google OIDC can be enabled on top of platform-user auth with `GOOGLE_OIDC_CLIENT_ID`, `GOOGLE_OIDC_CLIENT_SECRET`, and `GOOGLE_OIDC_REDIRECT_URI`. All three are required together (any partial subset fails startup), `USER_STORE_PATH` must also be set (OIDC layers on platform-user auth), and the redirect URI's PATH component must be exactly `/api/auth/oidc/google/callback` (the server's registered callback handler) — mirror that path in the Google Cloud Console's Authorized Redirect URIs. On non-loopback binds, the redirect URI must use `https://...` unless you explicitly set the same insecure-user-auth override for trusted private-network testing. Support triage: if an existing local-password user signs in via Google with the same email, the OIDC flow returns `?auth_error=an account with that email already exists; sign in with its existing method` (deliberate account-hijack defense). Resolution paths are "user signs in with their existing password" or "operator removes the local-password record from `USER_STORE_PATH` and lets OIDC mint a fresh user" — see `docs/HAND_HISTORY_WEB_DEPLOYMENT.md` for the full rationale.
- **Forgotten-password support triage.** No password-reset flow exists -- the platform-user store has no `/api/auth/reset` endpoint and no email-verification mechanism. Recovery options: (a) user registers under a different email (old record unreachable but still in the store), (b) configure Google OIDC and have the user sign in via that path (OIDC-vs-local-password collision applies on email match -- see the bullet above), or (c) stop the service, edit `USER_STORE_PATH` to delete the user's record by `email` field, restart; the user can then re-register with the same email but loses their saved profile fields. Option (c) is the most direct for single-user / small-team deployments; for larger user bases, deploy OIDC so the provider's own forgotten-password flow handles recovery without operator intervention. See `docs/HAND_HISTORY_WEB_DEPLOYMENT.md` for the full discussion.
- **"Why is my display name my email username?" triage.** When a user registers without filling in the optional display-name field, the server auto-fills it from the email's local-part (the substring before `@`) -- so `alice@example.com` registers with display name `alice`, surfaced verbatim in the profile card and the `linkedProviders` adjacent text. The literal fallback `SICFUN User` only fires when the local-part is empty, which `validateEmail` already rejects, so it's effectively unreachable in practice. Tell the user the field is self-service via the Profile panel: the bundled UI's Display name input on the profile-form posts `/api/auth/profile` and updates the stored value immediately, no operator intervention needed. See `docs/HAND_HISTORY_WEB_DEPLOYMENT.md` for the full register / login body shapes.
- In-process rate limiting now caps the expensive API routes. Use `RATE_LIMIT_SUBMITS_PER_MINUTE` (default `6`), `RATE_LIMIT_STATUS_PER_MINUTE` (default `240`; the frontend poll cadence is ~750 ms so 240 leaves room for ~3 concurrent tabs from one rate-limit client -- a signed-in user under platform-user auth, or one IP otherwise), and `RATE_LIMIT_AUTH_PER_MINUTE` (default `10`, stays IP-keyed because there's no authenticated user yet at register/login) to tune submit, job-status polling, and auth (register + login) caps independently; set any to `0` to disable that bucket. All three use a fixed 60-second window anchored at the client's first request (not a sliding window), so a client can burst up to ~2× the per-minute cap right at a window boundary — see `docs/HAND_HISTORY_WEB_DEPLOYMENT.md` for the worst-case timing walkthrough and the upstream-proxy guidance for fleet-wide sliding-window smoothing. The auth bucket throttles credential-stuffing attempts against `/api/auth/register` and `/api/auth/login`. OIDC `/api/auth/oidc/<provider>/start` and `/callback` are **not** in any in-process bucket — defend `/start` storms at the upstream proxy and watch `userAuthPendingOidcFlows` in `/api/health` for the misconfigured-provider pattern.
- Auth events emit structured log lines at INFO (success/expected) and WARN (failure): `auth.login.success email=… remote=…`, `auth.login.failure email=… remote=… reason=…`, `auth.register.success`, `auth.register.failure`, `auth.logout`, `auth.oidc.start provider=… remote=…`, `auth.oidc.start.failure provider=… remote=… reason=…`, `auth.oidc.success`, `auth.oidc.failure`. Tail `logs/*.stdout.log` / `logs/*.stderr.log` (or pipe to your log aggregator) and alert on a high WARN rate from a single `remote=` to spot brute-force attempts. Success lines emit the **canonical** (normalized) email from the stored record so all events for one user grep identically; failure lines emit the **submitted** email so unusual case/whitespace probes are visible in the form the attacker typed. The `auth.oidc.start` entry lets you correlate a click on "Sign in with Google" with the eventual `auth.oidc.success` / `auth.oidc.failure` callback — a stream of starts with no callbacks usually means the provider redirect is misconfigured or a script is exercising the start path.
- The `remote=` value is the **resolved client identity** in audit lines (auth.*, request unauthorized, request forbidden), not the raw TCP peer. When `RATE_LIMIT_CLIENT_IP_HEADER` is set and the peer is loopback or in `RATE_LIMIT_TRUSTED_PROXY_IPS`, `remote=` shows the IP the trusted header resolves to (no port — forwarded-IP headers don't carry one). Otherwise it's the direct peer as `host:port` with IPv6 bracketed per RFC 3986 §3.2.2. This means a behind-a-proxy deployment shows real client IPs in audit lines instead of every request looking like it came from the proxy, and the value matches the rate-limit `client=` field so a single grep correlates both.
- OIDC `/start` issues a short-lived state cookie -- `sicfun_oidc_state` in insecure mode, `__Host-sicfun_oidc_state` when `USER_AUTH_COOKIE_SECURE=true` (same operator-side log-grep / proxy-ACL consideration as the session cookie) -- with `HttpOnly`, `SameSite=Lax`, 10-minute Max-Age, plus `Secure` in secure mode. `/callback` requires the cookie value to match the URL `state` parameter before exchanging the authorization code — an OAuth 2.0 BCP covert-redirect / login-CSRF mitigation that prevents an attacker who completes their own authorization from forwarding the resulting URL to a victim. Failed checks log as `auth.oidc.failure provider=… reason=missing_state_cookie` or `reason=state_cookie_mismatch`; a burst usually means a misbehaving JS client clearing cookies between `/start` and `/callback`, not a real attack — but a spike correlated with one `remote=` is worth investigating. The Google `/start` URI also sets `prompt=select_account`, so every sign-in forces Google's account picker even for users with exactly one Google account already signed in -- useful for multi-account users (Work vs personal) and the operator-facing answer to the "why does it ask me every time?" support question on single-account users.
- `auth.oidc.failure` lines carry a `reason=` value naming the failure category. Five distinct patterns, from earliest fire-stage to latest: `missing_state_cookie` / `state_cookie_mismatch` (state-binding check failed, see the bullet above); `provider-error:<oauth_error_code>` when the provider sent a `?error=<code>` callback before we got to talk to it (`provider-error:access_denied` is a user-declined sign-in, `provider-error:temporarily_unavailable` is a provider outage, `provider-error:invalid_request` and others are RFC 6749 §5.2 standard codes); `oversize_callback_param` when either the `state` or `code` query param exceeds 256 chars (attacker probing the callback shape rather than legitimate provider traffic); `missing_code_or_state` when one of the two required query params is absent (truncated redirect, browser-side history-modification, etc.); and a variable-content reason drawn from this fixed 13-string finishOidc Left set (all space-bearing, so they grep as `%20`-escaped in the log), grouped by the finishOidc step that emits them: **state-store consume** -- `OIDC login state expired or is invalid` (state-store TTL expired or state value never issued/already consumed); **Google token + userinfo HTTP exchange** -- `Google token exchange failed with status <N>` (token endpoint non-2xx -- includes PKCE verifier mismatch which Google returns as 400), `Google token exchange did not return an access token` (token endpoint 2xx but `access_token` field missing -- pathological-provider or upstream-tampering signal), `Google userinfo request failed with status <N>` (userinfo endpoint non-2xx after successful token exchange); **userinfo body parse** -- `Google did not return a verified email address for this account` (userinfo 2xx but `email_verified: false`), `Google userinfo response was missing required identity fields` (userinfo 2xx but `sub` / `email` / `name` all empty after trim -- malformed provider or scope misconfig); **OIDC-email re-validation** -- `email must be at most 254 characters`, `email must not contain whitespace or control characters`, `email must be a valid address` (the same `validateEmail` the local-register path runs, applied defense-in-depth to whatever the provider returned); **integrity check** -- `OIDC subject is too long` (provider `sub` claim > 256 chars; bounded so a hostile provider can't poison the user store); **user-record write** -- `an account with that email already exists; sign in with its existing method` (OIDC-vs-local-password email collision), `registration is temporarily unavailable` (OIDC sign-up tripped the `USER_AUTH_MAX_USERS` cap, same generic message the local-register path emits); **catch-all** -- `Google OIDC exchange failed: <upstream-message>` (NonFatal mid-exchange exception: network failure, the connect/request timeouts named in the OIDC-HTTP-timeouts bullet above, JSON parse error on a malformed provider response, etc.). Triage shorthand: pattern 2 (provider-error:*) is provider-side; everything in the userinfo-parse + integrity-check + user-record-write groups is our-side / config-side; everything in the HTTP-exchange + catch-all groups is transport-side. See `docs/HAND_HISTORY_WEB_DEPLOYMENT.md` for the full per-pattern fire-stage discussion.
- The OIDC authorization-code flow also uses PKCE (Proof Key for Code Exchange, RFC 7636) with the SHA-256 challenge method, layered on top of the state-cookie binding above. At `/start` the server mints a fresh 32-byte `code_verifier`, derives `code_challenge = BASE64URL(SHA-256(code_verifier))`, sends the challenge in the redirect URL (`code_challenge` + `code_challenge_method=S256`), and stashes the verifier in the in-memory `OidcStateStore` alongside the state token. At `/callback` the server submits the verifier with the token exchange; Google rejects the exchange if the verifier doesn't hash to the original challenge. State binding protects against forwarded redirects (covert-redirect / login-CSRF); PKCE protects against a separately-intercepted authorization code (logged URL, leaked Referer, compromised intermediate proxy). Both run on every flow with no operator-tunable knob -- PKCE is on by default and not toggleable.
- OIDC HTTP calls to the provider's token + userinfo endpoints carry hardcoded JDK `HttpClient` timeouts: 5 seconds to establish the TCP connection and 10 seconds end-to-end per request. The two endpoints fire sequentially (token first, then userinfo with the returned access token), so a worst-case stuck callback holds an executor thread for **up to ~20 seconds** before bailing out. Triage shortcut for "Google sign-in seems hung for a long time" support tickets: anything beyond ~20s the user reports is downstream of our upstream call (DNS, JS handler, browser), not in it -- the server would have already surfaced an `auth.oidc.failure provider=google reason=Google%20OIDC%20exchange%20failed:%20...` line and redirected the user to the failure landing page well before then. The timeouts exist as DoS defense too -- without them, a hung provider could pin executor threads indefinitely and starve unrelated routes (`/api/health`, `/api/auth/me`). Not operator-tunable; changing them requires a source-level edit to `PlatformUserAuth.DefaultOidcConnectTimeout` / `DefaultOidcRequestTimeout`. See `docs/HAND_HISTORY_WEB_DEPLOYMENT.md` for the full discussion.
- A successful OIDC `/callback` automatically revokes any pre-existing session whose cookie was present on the same request, so a user who re-authenticates via OIDC while still signed in (e.g. clicked "Sign in with Google" while a previous session was alive) ends with exactly one valid session. Operationally this means a stolen older session token stops working the moment the user re-authenticates — useful when triaging an incident where you suspect a session was exposed: ask the user to sign in again and any leaked-but-pre-existing token is invalidated immediately rather than waiting for the 12 h sliding-TTL to expire.
- `USER_AUTH_MAX_USERS` (default `100000`) caps the size of the platform-user store. New registrations beyond the cap return `400` with `"registration is temporarily unavailable"` — a generic message that does NOT mention the cap, so a probing attacker cannot binary-search it. `/api/health` surfaces both the cap (`userAuthMaxUsers`) and the live count (`userAuthStoredUsers`) as JSON fields; dashboards should chart `userAuthStoredUsers / userAuthMaxUsers` and alert at e.g. 80% / 90% so capacity is raised (or registration disabled) before legitimate users hit the wall. Both fields are `null` in basic-auth and no-auth modes (no user store to count). Defense purpose: slow disk-fill via public-registration abuse — the auth rate-limit caps in-flight register/min, but without a per-store cap a persistent bot can still grow the file unbounded over days.
- `/api/health` also surfaces `userAuthActiveSessions` (live in-memory session count) and `userAuthPendingOidcFlows` (OIDC `/start` calls that have not yet completed via `/callback`). Use the first as a credential-stuffing-success / leaked-automation heads-up — an off-hours spike against a stable user base is suspicious. Use the second to spot misconfigured OIDC provider redirects: a steady rate of `/start` with no matching `/callback` usually means the provider's allowlisted redirect URI does not match `GOOGLE_OIDC_REDIRECT_URI`. Both are `null` outside platform-user mode so a basic-auth / no-auth dashboard query is safe.
- CSRF rejections on state-changing routes (POST `/api/auth/logout`, POST `/api/auth/profile`, POST `/api/analyze-hand-history`, POST `/api/playing-hall`, and DELETE `/api/playing-hall/jobs/{id}`) log as `request forbidden path=… remote=… email=… reason=csrf-missing-or-invalid` at WARN. A burst from one `remote=` against many endpoints is a likely automated probe; the same from a known user usually means their JS frontend lost the cookie mid-session and needs to reload.
- Unauthenticated requests to protected paths log as `request unauthorized path=… remote=… reason=…` at WARN. The reason names the auth-stage failure (`missing_authorization`, `unsupported_authorization_scheme`, `malformed_authorization`, `invalid_credentials`, or `session-missing-or-invalid`). The first four come from the Basic-auth path (no `Authorization:` header / header present but not `Basic` / undecodable base64 / decoded fine but `user:password` compare failed -- the compare is constant-time via the non-short-circuit `&` operator, so a timing probe cannot distinguish wrong-username from right-username-wrong-password); the fifth comes from the platform-user-auth path (no `__Host-sicfun_session` / `sicfun_session` cookie, or the cookie's token doesn't resolve to a live session record because of TTL expiry or sibling-tab logout). Two operator-facing alert patterns: a high WARN rate of `reason=invalid_credentials` from one `remote=` is the credential-stuffing-against-Basic signal; the same shape with `reason=session-missing-or-invalid` for platform-user is a cookie-replay / leaked-session probe.
- Rate-limit rejections log as `request rate limited path=… client=… bucket=… limitPerMinute=… retryAfterMs=…` at WARN. The `client=` value is either `user:<userId>` for an authenticated principal, `header:<ip>` when a trusted forwarded-IP header is configured, or `remote:<addr>` otherwise. `bucket=` is one of `submit` / `job-status` / `auth`.
- Job lifecycle events emit seven distinct prefixes per job store, useful for log-monitoring rules: `job accepted` / `job started` / `job completed` (INFO), `job rejected unavailable` / `job rejected queue full` / `job failed` / `job timed out` (WARN). Playing Hall jobs emit a parallel set under `playing hall job ...`, plus `playing hall job cancelled` (INFO) on the DELETE-cancel path. Each line carries `jobId=` (omitted on the rejection-prefix lines `... rejected unavailable` and `... queue full`, which represent admission failures and don't surface a jobId in the log) plus a context-appropriate subset of `queuedJobs=` / `runningJobs=` / `durationMs=` / `errorStatus=` / `timeoutMs=` / `error=` / `reason=`; the `... queue full` lines additionally carry `maxConcurrentJobs=` and `maxQueuedJobs=` so an operator chasing a capacity-cap rejection can confirm which cap fired straight from the log line without re-reading the running config. Enough to chart submit→start latency, completion rate, timeout rate, and queue depth from the audit log alone.
- Log-grep convention: any structured-log value that could legitimately contain a space is `%20`-escaped before it reaches the line so it doesn't split the surrounding `key=value` pairs. That covers submitted-email at register/login, analyze `heroName=`, startup banner `modelSource=` / `drainSignalFile=` / `rateLimitClientIpSource=`, the `reason=` field in admission-rejected lines (both `job rejected unavailable` for analyze and the parallel `playing hall job rejected unavailable` for Playing Hall) and auth-failure lines, and the `error=` field in failed-job lines (both `job failed` for analyze and the parallel `playing hall job failed` for Playing Hall). When grepping for an email, Windows path, or backend message in the audit log, type the value with `%20` in place of any space — e.g. `grep "modelSource=C:\\Program%20Files\\sicfun"`. See `docs/HAND_HISTORY_WEB_DEPLOYMENT.md` for the full per-field inventory.
- The limiter buckets by the authenticated user (`user:<userId>`) when a platform-user session is present on the request; otherwise it falls back to the resolved client IP (`header:<ip>` when `RATE_LIMIT_CLIENT_IP_HEADER` is configured and the TCP peer is trusted, or `remote:<addr>` otherwise). The user-keyed bucketing means a noisy authenticated user can't get unrelated users behind the same NAT throttled. If you deploy behind a trusted reverse proxy, set `RATE_LIMIT_CLIENT_IP_HEADER` to a proxy-populated single-value client-IP header such as `X-Real-IP`. Same-host loopback proxies are trusted automatically; for proxies on other hosts, also set `RATE_LIMIT_TRUSTED_PROXY_IPS` to a comma-separated list of exact proxy peer IP literals. Do not enable the header knob on a directly exposed app because clients can spoof those headers.
- The packaged launcher now requires `java` on `PATH`, checks `java -version` before startup, accepts Java 17+, and recommends JDK 21 for operator parity.
- The packaged service helper scripts assume a Windows host and use NSSM as the service wrapper. Service install/uninstall requires an elevated PowerShell session.
- The service install script configures stdout/stderr capture under `dist/hand-history-web/logs/` and enables basic Windows service restart-on-failure recovery.
- `bin/start-hand-history-web-service.ps1` now fails fast if the service stops during startup and includes the latest readiness/health summary plus recent stdout/stderr tail when readiness does not come up cleanly.
- `bin/drain-stop-hand-history-web-service.ps1` now includes the last readiness/health probe summary when drain mode does not flip or jobs do not fully drain before the forced stop.
- Uploads are accepted quickly and processed as background jobs; the page polls `/api/analyze-hand-history/jobs/{id}` (or `/api/playing-hall/jobs/{id}` for hall simulations) until the work finishes. Playing Hall jobs additionally accept `DELETE` on the job URL for cooperative cancellation; the server returns `200` with `status=cancelled` while the job is still cancellable (queued or running), `409` if it already reached a terminal state (completed / failed / cancelled), `404` if unknown.
- Analysis admission is now bounded. Use `-MaxConcurrentJobs`, `-MaxQueuedJobs`, and `-ShutdownGraceMs` or the matching `MAX_CONCURRENT_JOBS`, `MAX_QUEUED_JOBS`, and `SHUTDOWN_GRACE_MS` environment variables to control saturation and shutdown drain behavior.
- Use `-AnalysisTimeoutMs` or `ANALYSIS_TIMEOUT_MS` to cap a single analysis job (default 120000 ms, i.e. 2 min). Use `-PlayingHallTimeoutMs` or `PLAYING_HALL_TIMEOUT_MS` to cap a single Playing Hall job (default 900000 ms, i.e. 15 min). `0` disables either timeout, but the deployment-safe default is a bounded run on both so one stuck review or hall pin cannot pin the worker pool indefinitely. The shipped frontend reads both values from `/api/health` at boot and extends its own poll deadline to `max(16 min default, larger server timeout + 1 min slack)`, so raising either knob server-side does not need a frontend rebuild — `0` (server-side disabled) falls back to the 16-min frontend default rather than polling indefinitely.
- `SHUTDOWN_GRACE_MS` is tracked in milliseconds, but the underlying HTTP listener drains in whole-second steps. Sub-second values round up when the listener is stopping.
- `/api/health` is the liveness/metrics endpoint. It accepts `GET`, `HEAD`, and `OPTIONS`; everything else returns `405` with `Allow: GET, HEAD, OPTIONS`. It stays `200` while the process is up and now reports readiness summary, auth mode, model mode, upload limit, analysis timeout, **Playing Hall timeout**, submit/status/auth rate-limit settings, the trusted client-IP source used for rate limiting as `rateLimitClientIpSource` (one of `remote-address`, `header:<name> via loopback-only`, or `header:<name> via loopback-or-allowlisted-proxy` -- see the deployment doc for which `RATE_LIMIT_CLIENT_IP_HEADER` + `RATE_LIMIT_TRUSTED_PROXY_IPS` combination produces each), queue limits, queued jobs, running jobs, timed-out workers still unwinding, retained terminal-job count (analysis + Playing Hall, summed), the machine-readable readiness reason as `readyReason` (same enum as `/api/ready`'s `reason` field — `accepting-traffic` / `draining` / `timed-out-worker` / `queue-full` — but `/api/health` calls it `readyReason` and `/api/ready` calls it `reason`, so a dashboard reading both endpoints picks one name per probe), `service`/`host`/`port`/`startedAtEpochMs`/`uptimeMs` for fleet-wide dashboard correlation, and — under platform-user auth — the four user-store metrics `userAuthMaxUsers` / `userAuthStoredUsers` / `userAuthActiveSessions` / `userAuthPendingOidcFlows` (all `null` in basic-auth / no-auth modes), in addition to `ok=true`.
- `/api/ready` is the readiness endpoint for reverse proxies / service managers. Same verb set as `/api/health`. It returns `200` only when the instance is accepting new analysis work and switches to `503` when the queue is saturated, the instance is draining, or a timed-out worker is still unwinding. The response also reports `ready` + `reason` (machine-readable: `accepting-traffic` / `draining` / `timed-out-worker` / `queue-full`), `draining` + `acceptingAnalysisJobs` + `drainSignalConfigured` + `drainSignalPresent` (so a probe can tell "queue is full" from "operator-initiated drain" without a separate health check), the configured `analysisTimeoutMs`, `playingHallTimeoutMs`, `timedOutWorkersInFlight`, `activeHttpRequests`, `maxConcurrentJobs`/`maxQueuedJobs`/`queuedJobs`/`runningJobs` (so a load balancer can plot saturation alongside the bare ready/not-ready bit), auth mode, submit/status/auth rate-limit settings, `rateLimitClientIpSource` (same field + value enum as the `/api/health` bullet above), and `service`/`host`/`port` for fleet correlation.
- Use `-DrainSignalFile <path>` or `DRAIN_SIGNAL_FILE=<path>` when you want external deployment tooling to mark the instance unready before shutdown. While that file exists, `/api/ready` returns `503` and new `POST /api/analyze-hand-history` and `POST /api/playing-hall` submissions are rejected, but health checks and in-flight job polling still work.
- `bin/drain-stop-hand-history-web-service.ps1` turns on the configured drain signal, waits for readiness to fail plus in-memory jobs and in-flight HTTP requests to drain to zero, then stops the Windows service.
- Runtime state is still in-memory only. Queued and running review jobs are lost on process restart, completed-job status is retained for only 15 minutes, and in-flight OIDC sign-in state (the `OidcStateStore` holding the random `state` tokens issued by `/start` until `/callback` consumes them) drops too -- affected users see `?auth_error=OIDC login state expired or is invalid` and re-click Continue with Google. Watch `/api/health.userAuthPendingOidcFlows` before a planned restart to spot when non-trivial in-flight OIDC traffic would be affected; see `docs/HAND_HISTORY_WEB_DEPLOYMENT.md` for the operator-impact discussion.
- Under platform-user auth, account/profile data persists in `USER_STORE_PATH`, but browser sessions and in-flight review jobs remain in-memory only. A restart signs users out and drops queued/running jobs. The store JSON contains emails, PBKDF2-HMAC-SHA256 password hashes (per-account salted with 128-bit random salt, 210,000 iterations, 256-bit output -- meets NIST SP 800-132 §5.1 floor of ≥128-bit salt and ≥256-bit output; not plaintext), profile fields (displayName/heroName/preferredSite/timeZone), OIDC subject identifiers, per-account UUIDs (internal `userId`), and event timestamps (`createdAtEpochMs` / `updatedAtEpochMs` / `lastLoginAtEpochMs`). Treat the file as PII: restrict filesystem ACLs to the service account, encrypt backups, and keep it off any tier where unprivileged readers could grep it. A leak is not a credential leak (the hashes resist offline cracking) but the linked emails + OIDC subjects + profile fields are still PII a compliance audit would flag.
- The raw server now emits baseline security headers (`Content-Security-Policy`, `Permissions-Policy`, `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Cross-Origin-Opener-Policy`, `Cross-Origin-Resource-Policy`, `X-Robots-Tag`), can enforce built-in Basic auth, and applies a best-effort in-process rate limiter on the expensive API routes (submit / status / auth buckets), but that is still not a substitute for TLS termination or edge rate limiting.
- Static assets are served with weak `ETag` headers and path-aware `Cache-Control`: `vendor/*` files get `public, max-age=31536000` (vendored libs are pinned per filename), all other static assets get `public, max-age=0, must-revalidate`. Browsers revalidate with `If-None-Match` or `If-Modified-Since`; matching validators get `304 Not Modified` with no body. API responses still emit `Cache-Control: no-store`. A reverse proxy with caching enabled (e.g., nginx with proxy_cache_path) can respect these directives directly; do not strip them.
- Responses for compressible types (`text/*`, `application/javascript`, `application/json`, `image/svg+xml`) carry `Vary: Accept-Encoding` and are gzipped on the wire when the client sends `Accept-Encoding: gzip`. The wire savings are ~70% on the chart-stack cold load. The static handler emits a `-gz` ETag suffix on compressed variants so caches that key only on ETag (ignoring Vary) don't serve the wrong encoding. The origin honors `gzip;q=0` to opt out per RFC 7231 sec 5.3.4.
- Every response also carries a `Permissions-Policy` header that explicitly denies a broad set: hardware (camera, microphone, USB, HID, serial, MIDI, Bluetooth, magnetometer, accelerometer, gyroscope, ambient-light-sensor, battery, screen-wake-lock), display (display-capture, fullscreen, picture-in-picture, XR spatial tracking), media (autoplay, encrypted-media), web-platform (geolocation, payment, web-share, otp-credentials, publickey-credentials-get, idle-detection, storage-access, local-fonts, document-domain), privacy tracking (FLoC / interest-cohort, browsing-topics), and Chrome's Privacy Sandbox surface (attribution-reporting, private-state-token-issuance/redemption, run-ad-auction, shared-storage, shared-storage-select-url). Defense-in-depth alongside CSP; reverse proxies that strip these weaken the browser-side protection.
- The full `Content-Security-Policy` value on every response is: `default-src 'self'; base-uri 'none'; connect-src 'self'; form-action 'self'; frame-ancestors 'none'; frame-src 'none'; img-src 'self' data:; manifest-src 'none'; media-src 'none'; object-src 'none'; script-src 'self'; style-src 'self'; worker-src 'none'`. Same-origin (`'self'`) is the only allowed source for the six "where content can load from" directives (`default-src` / `connect-src` / `script-src` / `style-src` / `img-src` / `form-action`); `img-src` ALSO permits `data:` URLs (the only data-URL site needs in shipped markup is the empty inline favicon `<link rel="icon" href="data:,">` in index.html, but the directive permits any `data:` image scheme, not just that exact URL -- which is fine because data-URL images can't execute JavaScript and the other directives still block `data:` for scripts/styles/connects). The remaining seven (`base-uri` / `frame-ancestors` / `frame-src` / `manifest-src` / `media-src` / `object-src` / `worker-src`) are `'none'` so a future injected `<base>`, `<iframe>`, `<object>`/`<embed>`, `<audio>`/`<video>`, `<link rel=manifest>`, or `new Worker(...)` cannot promote itself into a navigation override, framing channel, plugin embed, exfiltration channel, PWA install, or service-worker spawn. The `X-Frame-Options: DENY` header pairs with `frame-ancestors 'none'` as defense in depth for older browsers that ignore CSP framing rules. Reverse proxies that rewrite CSP (Cloudflare, Vercel, Netlify, and similar hosting platforms inject their own) MUST keep both the same-origin pins and the explicit `'none'` denials -- relaxing any of them widens the post-compromise blast radius proportional to that directive's reach.
- The origin emits no CORS response headers (`Access-Control-Allow-Origin`, `-Methods`, `-Headers`, `-Credentials`); OPTIONS preflight responses carry only the `Allow:` method-discovery header per RFC 7231 sec 4.3.7. A different-origin browser XHR / fetch is therefore blocked by the same-origin policy -- the missing `Access-Control-Allow-Origin` fails the preflight gate regardless of cookies or auth. The CSP `connect-src 'self'` directive above layers defense in depth from the inside: it also blocks the same-origin frontend from issuing cross-origin fetches, so an injected script cannot smuggle data out to an attacker-controlled origin even if it gets executed. Operators who need a different-origin client (e.g., a dashboard or programmatic consumer hosted elsewhere) to talk to this API must front the service with a reverse proxy that injects the right CORS headers -- or co-locate the client on the same origin. The in-process server intentionally has no CORS knob to turn on.
- Do not expose the raw app directly to the public internet without HTTPS in front of it. Built-in Basic auth and the in-process limiter help with access control and abuse containment, but you still want a reverse proxy / ingress layer for TLS termination, network policy, and stronger rate limiting. The origin does NOT emit `Strict-Transport-Security` itself (it does not terminate TLS); configure the HTTPS-terminating proxy to add an HSTS header (e.g. `Strict-Transport-Security: max-age=31536000; includeSubDomains`) AFTER you have verified HTTPS works end-to-end — once the browser has cached `max-age` it refuses to fall back to HTTP for that host, so a misconfigured HSTS during testing locks you out of plain-HTTP access until the TTL expires. See `docs/HAND_HISTORY_WEB_DEPLOYMENT.md` for the full reverse-proxy header-preservation list.
- `scripts/release-hand-history-web.ps1` validates the required static assets, asserts that every `src=`/`href=` reference in `index.html` resolves to a file under the static root, then smoke-checks packaged fail-closed non-loopback config rejection, auth-enabled `/` with security-header presence and weak-ETag emission, `/api/health`, `/api/ready`, async `/api/analyze-hand-history`, trusted-header submit rate limiting, the full `/api/playing-hall` lifecycle (POST auth gating, GET poll, DELETE-after-completed → 409), drain-mode readiness rejecting both analysis and Playing Hall submissions, the chart-stack assets (`site-charts.js`, `vendor/uPlot.iife.min.js`, `vendor/uPlot.min.css`) with correct Content-Type plus 304 revalidation via `If-None-Match`, oversized-upload rejection, and packaged manifest verification before declaring the build ready. The installer variant adds Step 6.5 which repeats the basic smoke against the patched bundle through the embedded jlink runtime.
- The web server supports `CONFIG_FILE`, `HOST`, `PORT`, `STATIC_DIR`, `MODEL_DIR`, `MAX_UPLOAD_BYTES`, `ANALYSIS_TIMEOUT_MS`, `PLAYING_HALL_TIMEOUT_MS`, `MAX_CONCURRENT_JOBS`, `MAX_QUEUED_JOBS`, `SHUTDOWN_GRACE_MS`, `RATE_LIMIT_SUBMITS_PER_MINUTE`, `RATE_LIMIT_STATUS_PER_MINUTE`, `RATE_LIMIT_AUTH_PER_MINUTE`, `RATE_LIMIT_CLIENT_IP_HEADER`, `RATE_LIMIT_TRUSTED_PROXY_IPS`, `DRAIN_SIGNAL_FILE`, `BASIC_AUTH_USER`, `BASIC_AUTH_PASSWORD`, `USER_STORE_PATH`, `USER_AUTH_ALLOW_REGISTRATION`, `USER_AUTH_SESSION_TTL_MS`, `USER_AUTH_COOKIE_SECURE`, `USER_AUTH_MAX_USERS`, `GOOGLE_OIDC_CLIENT_ID`, `GOOGLE_OIDC_CLIENT_SECRET`, `GOOGLE_OIDC_REDIRECT_URI`, `ALLOW_UNAUTHENTICATED_PUBLIC_BIND`, and `ALLOW_INSECURE_USER_AUTH` environment-variable overrides in addition to CLI flags. The `MAX_CONCURRENT_JOBS` and `MAX_QUEUED_JOBS` budgets are shared across both job stores (analysis and Playing Hall).

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
