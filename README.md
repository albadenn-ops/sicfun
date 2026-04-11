# SICFUN

SICFUN is a Texas Hold'em analysis workbench and research codebase.
The most usable surface today is the local hand-history review web UI; the rest of the repo contains the supporting equity, inference, simulation, and experimental native/DDRE infrastructure behind it.

## Start Here

- Quick validation (core runtime smoke): `powershell -ExecutionPolicy Bypass -File scripts/prove-pipeline.ps1 -Quick`
- Full validation (includes hand-history review E2E): `powershell -ExecutionPolicy Bypass -File scripts/prove-pipeline.ps1`
- Review local hand histories: `powershell -ExecutionPolicy Bypass -File scripts/start-hand-history-web.ps1`
- Operator workflows: [docs/OPERATOR_RUNBOOK.md](docs/OPERATOR_RUNBOOK.md)
- Milestone status and caveats: [ROADMAP.md](ROADMAP.md)

## Repository Boundaries

- Source of truth: `src/main`, `src/test`, `scripts`, and the checked-in docs/specs under `docs/`.
- Output-heavy areas: packaged layouts under `dist/`, generated scratch data under `data/`, validation reports under `validation-output/`, and native build directories under `src/main/native/`.
- Generated runtimes, native build outputs, validation reports, and scratch data are local-only artifacts. Recreate them when needed instead of treating them as repository content.
- Developer-local dependencies may also live under `data/tmp/` (for example `g5-poker-bot`, `acpc-server`, and `toolchains`); they are local setup, not source of truth.
- Current goal: research-grade analysis and simulation tooling, not a polished release repository.

## Status

- Implemented and exercised by tests: hand evaluation, equity estimation, Bayesian range updates, action-model training, event-feed decision loops, and the self-play playing hall.
- Experimental: DDRE. The current DDRE stack is plumbing and scaffolding, not a validated poker model release.
- Not implemented: real-table integration and action execution against a live poker client.

## DDRE Reality Check

- `synthetic` DDRE is a heuristic scaffold used to exercise blending, fallback, and provider routing.
- Native DDRE CPU/GPU currently execute the same synthetic inference core through JNI; they do not run a trained diffusion model.
- The ONNX path is adapter plumbing. The checked-in smoke model is `posterior = sqrt(prior)` and exists to prove the runtime contract, not poker quality.
- Decision-driving ONNX now requires artifact metadata that has passed the offline gate, unless you explicitly opt into experimental artifacts.
- DDRE parity/smoke tests confirm transport and fallback behavior. They are not evidence of model strength.

## What This Repo Is Not

- Not a production poker bot.
- Not proof of DDRE model quality.
- Not a live-table integration or action executor against a real poker client.

## Hand-History Review Web UI

Serve the upload page plus analysis API locally with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start-hand-history-web.ps1
```

The page is served from `docs/site-preview-hybrid` and posts uploaded PokerStars, Winamax, or GGPoker text hand histories to the in-process Scala importer/analyzer.
Uploads are accepted on the normal HTTP endpoint, turned into background jobs, and polled via `/api/analyze-hand-history/jobs/{id}` until the review is ready.
`/api/health` is the liveness/metrics endpoint. `/api/ready` is the readiness endpoint and turns `503` when the queue is saturated, a configured drain-signal file is present, or a timed-out worker is still unwinding.
Use `-AnalysisTimeoutMs` or `ANALYSIS_TIMEOUT_MS` to cap a single analysis job; set `0` to disable the timeout if you are intentionally running longer reviews.
Optional in-process HTTP Basic auth can protect `/`, `/api/analyze-hand-history`, and `/api/analyze-hand-history/jobs/{id}` while leaving `/api/health` and `/api/ready` open for service management. Prefer `BASIC_AUTH_USER` and `BASIC_AUTH_PASSWORD` via config/env over CLI flags so credentials do not appear in process arguments.
For future multi-user deployments, platform-user auth is now available as a separate mode. Set `USER_STORE_PATH` to enable persistent local users, browser sessions, per-user job ownership, and profile defaults (hero name, preferred site, time zone). `BASIC_AUTH_*` and `USER_STORE_PATH` are mutually exclusive.
When platform-user auth is enabled, the upload UI stays reachable so the browser can render sign-in/register/profile controls, but `/api/analyze-hand-history` and `/api/analyze-hand-history/jobs/{id}` require a session. Session-protected writes also require the in-page CSRF token.
Google login is supported through the built-in OIDC provider when `GOOGLE_OIDC_CLIENT_ID`, `GOOGLE_OIDC_CLIENT_SECRET`, and `GOOGLE_OIDC_REDIRECT_URI` are configured alongside `USER_STORE_PATH`. If you expose the app over HTTPS, set `USER_AUTH_COOKIE_SECURE=true`.
The review API now also supports in-process submit and job-status rate limits via `RATE_LIMIT_SUBMITS_PER_MINUTE` and `RATE_LIMIT_STATUS_PER_MINUTE`; set either to `0` to disable that limiter. By default those buckets key off the remote socket address. For reverse-proxy deployments, set `RATE_LIMIT_CLIENT_IP_HEADER` to a trusted single-value client-IP header such as `X-Real-IP`. Same-host loopback proxies are trusted automatically; for proxies on other hosts, also set `RATE_LIMIT_TRUSTED_PROXY_IPS` to a comma-separated list of exact proxy peer IPs.
The full validation command above now proves the review API path end-to-end by generating a reproducible hall export, importing it through the review service, and completing an async review job over HTTP.
The source launcher and packaged launcher bind to `127.0.0.1` by default; pass `-Host 0.0.0.0` only if you intentionally want LAN exposure.

Build a standalone Windows release that runs without `sbt`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/release-hand-history-web.ps1
```

That assembles `dist/hand-history-web`, writes `bin/run-hand-history-web.ps1`, validates the expected static assets, and smoke-checks authenticated `GET /`, unauthenticated `GET /api/health`, unauthenticated `GET /api/ready`, async `POST /api/analyze-hand-history`, drain-mode readiness, and oversized-upload rejection from the packaged layout.
The packaged layout also includes `conf/hand-history-web.env` for runtime settings, including optional Basic auth credentials, plus `bin/install-hand-history-web-service.ps1`, `bin/start-hand-history-web-service.ps1`, `bin/drain-stop-hand-history-web-service.ps1`, and `bin/uninstall-hand-history-web-service.ps1` for a Windows service workflow backed by NSSM.

For simulator and benchmark workflows, use [docs/OPERATOR_RUNBOOK.md](docs/OPERATOR_RUNBOOK.md).

For milestone status, read [ROADMAP.md](ROADMAP.md) together with the DDRE caveats above.
