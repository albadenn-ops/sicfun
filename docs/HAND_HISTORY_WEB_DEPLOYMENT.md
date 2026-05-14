# SICFUN Hand-History Web Deployment

This guide is for the packaged Windows hand-history review product under `dist/hand-history-web`.

## Two Release Variants

Two release scripts produce two artifact shapes. Both serve the same web app; the difference is what they bundle for the host.

| Script | Output | Java handling | Click-to-run |
|---|---|---|---|
| `scripts/release-hand-history-web.ps1` | `dist/hand-history-web/` directory only | Requires Java 17+ on `PATH` | No |
| `scripts/release-hand-history-web-installer.ps1` | Same directory **plus** versioned outer ZIP at `dist/hand-history-web-<version>.zip` | Embeds a jlink runtime under `runtime/bin/java.exe`; falls back to PATH if missing | Yes: `Setup.cmd` verifies the manifest then launches the service |

Use the installer variant when the target host may not have Java pre-installed.

## What You Ship

The release bundle contains:

- `bin/run-hand-history-web.ps1`: start the app directly
- `bin/install-hand-history-web-service.ps1`: install an NSSM-backed Windows service
- `bin/start-hand-history-web-service.ps1`: start the service and wait for readiness
- `bin/drain-stop-hand-history-web-service.ps1`: drain and stop the service cleanly
- `bin/uninstall-hand-history-web-service.ps1`: remove the service
- `bin/verify-release-manifest.ps1`: verify bundle integrity against `manifest.sha256`
- `conf/hand-history-web.env`: runtime configuration template
- `static/`: packaged upload UI
- `lib/`: application jars
- `model/`: optional packaged model artifacts if they were included at build time

Installer-variant additions:

- `Setup.cmd`: click-to-run entry point that verifies the manifest then launches the service
- `runtime/`: jlink-trimmed Java runtime; the launcher and service-common shim prefer this over `PATH`
- `runtime/BUILD_INFO.txt`: source JDK version and jdeps-detected module list

## Host Prerequisites

- Windows host
- PowerShell 5.1+
- Java 17+ on `PATH` before startup (not required for the installer variant — `runtime/` is embedded)
- NSSM if you want the Windows service workflow (drop `nssm.exe` into `bin/` before zipping or pass `-NssmPath` at install time; auto-download is disabled because corporate AV often blocks `nssm.cc`)

## Quick Start

From the package root:

```powershell
powershell -ExecutionPolicy Bypass -File .\bin\verify-release-manifest.ps1
powershell -ExecutionPolicy Bypass -File .\bin\run-hand-history-web.ps1
```

Default behavior:

- Binds to `127.0.0.1:8080`
- Serves the upload UI from `static/`
- Loads settings from `conf/hand-history-web.env`
- Exposes `/api/health` for liveness and `/api/ready` for readiness
- Exposes async job endpoints for hand-history analysis and Playing Hall simulation (see "HTTP Endpoints" below)

## HTTP Endpoints

`/api/health` and `/api/ready` are unauthenticated. The hand-history analysis and Playing Hall job routes always require auth (Basic auth or platform-user) and are rate-limited (Submit / JobStatus buckets — see "Rate limiting" below). Auth bootstrap routes under `/api/auth/*` are open by design (login/register cannot require an existing session). Submissions return `202 Accepted` with `Location` and `Retry-After` headers plus a JSON body containing `jobId`, `status`, `statusUrl`, `submittedAtEpochMs`, `pollAfterMs`.

Hand-history analysis:

- `POST /api/analyze-hand-history` — submit a hand history for analysis. Body: JSON `{handHistoryText, heroName?, site?}`. Rate-limited as Submit.
- `GET /api/analyze-hand-history/jobs/{jobId}` — poll job status. Returns `queued` / `running` / `completed` / `failed`. Rate-limited as JobStatus.

Playing Hall simulation:

- `POST /api/playing-hall` — submit a hall simulation. JSON body fields: `hands`, `tableCount`, `playerCount`, `heroStyle` (`adaptive`/`gto`/`strategic`), `heroPosition` (`SmallBlind`/`BigBlind`/`UTG`/`UTG1`/`UTG2`/`Middle`/`Hijack`/`Cutoff`/`Button`), `gtoMode` (`fast`/`exact`), `villainPool`, `heroExplorationRate`, `raiseSize`, `bunchingTrials`, `equityTrials`. Rate-limited as Submit; admission shares the same `MAX_CONCURRENT_JOBS` / `MAX_QUEUED_JOBS` budget as hand-history analysis.
- `GET /api/playing-hall/jobs/{jobId}` — poll job status. Rate-limited as JobStatus.
- `DELETE /api/playing-hall/jobs/{jobId}` — request cooperative cancellation of an in-flight Playing Hall job. Returns `200` with `status=cancelled` once accepted, `409` if the job already finished, `404` if unknown. Rate-limited as JobStatus.

Platform-user auth (when `USER_STORE_PATH` is set):

- `GET /api/auth/me`, `POST /api/auth/register`, `POST /api/auth/login`, `POST /api/auth/logout`, `POST /api/auth/profile`
- Optional OIDC start/callback at `/api/auth/oidc/{provider}/start` and `/api/auth/oidc/{provider}/callback` (e.g. `/api/auth/oidc/google/callback`) when the matching `*_OIDC_*` settings are configured.

## Core Configuration

Edit `conf/hand-history-web.env` instead of hard-coding long-lived settings into service command lines.

Common settings:

- `HOST` / `PORT`: bind address and port
- `ALLOW_UNAUTHENTICATED_PUBLIC_BIND`: explicit override for non-loopback binds without auth on a trusted private network
- `MODEL_DIR`: optional model artifact directory
- `MAX_UPLOAD_BYTES`: upload cap (applies to both `/api/analyze-hand-history` and `/api/playing-hall` request bodies)
- `ANALYSIS_TIMEOUT_MS`: per-job timeout for `/api/analyze-hand-history` jobs
- `PLAYING_HALL_TIMEOUT_MS`: per-job timeout for `/api/playing-hall` jobs (default `900000`, i.e. 15 min); `0` disables it
- `MAX_CONCURRENT_JOBS` / `MAX_QUEUED_JOBS`: admission limits shared across both job stores
- `SHUTDOWN_GRACE_MS`: graceful shutdown budget
- `DRAIN_SIGNAL_FILE`: path used to mark the instance unready before shutdown; while present, new submissions for both analysis and Playing Hall are rejected with `503`

Auth modes:

- Basic auth: set `BASIC_AUTH_USER` and `BASIC_AUTH_PASSWORD`
- Platform-user auth: set `USER_STORE_PATH`
- `ALLOW_INSECURE_USER_AUTH`: explicit override for non-loopback platform-user auth without secure cookies or an HTTPS OIDC callback on a trusted private-network test deployment
- Do not enable both at the same time
- For safety, non-loopback binds now require one of those auth modes unless you explicitly set `ALLOW_UNAUTHENTICATED_PUBLIC_BIND=true` for a trusted private network
- For safety, non-loopback platform-user auth also requires `USER_AUTH_COOKIE_SECURE=true` unless you explicitly set `ALLOW_INSECURE_USER_AUTH=true` for trusted private-network testing

Optional OIDC:

- Set `GOOGLE_OIDC_CLIENT_ID`, `GOOGLE_OIDC_CLIENT_SECRET`, and `GOOGLE_OIDC_REDIRECT_URI`
- On non-loopback binds, set `USER_AUTH_COOKIE_SECURE=true` and use an `https://...` redirect URI unless you explicitly set `ALLOW_INSECURE_USER_AUTH=true` for trusted private-network testing

Rate limiting:

- `RATE_LIMIT_SUBMITS_PER_MINUTE`
- `RATE_LIMIT_STATUS_PER_MINUTE`
- `RATE_LIMIT_CLIENT_IP_HEADER`
- `RATE_LIMIT_TRUSTED_PROXY_IPS`

Only trust a client IP header when the app is behind a proxy you control.

## Service Install

Install:

```powershell
powershell -ExecutionPolicy Bypass -File .\bin\install-hand-history-web-service.ps1 `
  -NssmPath C:\tools\nssm\nssm.exe
```

Start and wait for readiness:

```powershell
powershell -ExecutionPolicy Bypass -File .\bin\start-hand-history-web-service.ps1
```

Drain and stop:

```powershell
powershell -ExecutionPolicy Bypass -File .\bin\drain-stop-hand-history-web-service.ps1
```

Uninstall:

```powershell
powershell -ExecutionPolicy Bypass -File .\bin\uninstall-hand-history-web-service.ps1
```

The service helper scripts write stdout/stderr logs under `logs/`.
If startup fails or readiness times out, `bin/start-hand-history-web-service.ps1` now includes the current service state plus recent stdout/stderr tail in its error output to shorten first-response debugging.
If drain/stop times out, `bin/drain-stop-hand-history-web-service.ps1` now reports the last readiness/health probe summary before forcing the stop.

## Reverse Proxy And Exposure

- Keep the raw app on loopback unless you intentionally want LAN exposure
- If you intentionally bind to `0.0.0.0`, a LAN IP, or another non-loopback host without auth, set `ALLOW_UNAUTHENTICATED_PUBLIC_BIND=true` in `conf/hand-history-web.env` so that choice is explicit
- Put HTTPS and stronger edge rate limiting in front of the app for internet-facing deployments
- Use `/api/ready` for load balancers and service managers
- Use `/api/health` for liveness and coarse metrics

## State And Backups

- In-flight and completed review jobs are in-memory only
- A restart drops queued/running jobs
- Platform-user auth persists account/profile data under `USER_STORE_PATH`
- Back up `conf/hand-history-web.env` and `USER_STORE_PATH` if they matter operationally

## Bundle Integrity

Verify the shipped bundle after copy/deploy:

```powershell
powershell -ExecutionPolicy Bypass -File .\bin\verify-release-manifest.ps1
```

That script checks every shipped file against `manifest.sha256` and fails if a file is missing, added, or modified. Runtime marker files at the bundle root (filenames starting with `.`, e.g. `.manifest-verified` written by the installer-variant's `Setup.cmd`) are intentionally ignored.

The installer variant's `Setup.cmd` calls `bin\verify-if-needed.ps1` instead of running the full verifier on every launch. That helper writes a `.manifest-verified` marker after a successful check and short-circuits subsequent launches as long as the marker's timestamp is at least as new as `manifest.sha256`. Re-extracting the bundle advances `manifest.sha256`'s mtime and triggers a fresh verification. To force a re-verify manually, delete `.manifest-verified` at the bundle root.
