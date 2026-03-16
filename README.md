# SICFUN

SICFUN is a Texas Hold'em analysis workbench and research codebase.
The most usable surface today is the local hand-history review web UI; the rest of the repo contains the supporting equity, inference, simulation, and experimental native/DDRE infrastructure behind it.

## Start Here

- Quick validation: `powershell -ExecutionPolicy Bypass -File scripts/prove-pipeline.ps1 -Quick`
- Review local hand histories: `powershell -ExecutionPolicy Bypass -File scripts/start-hand-history-web.ps1`
- Operator workflows: [docs/OPERATOR_RUNBOOK.md](docs/OPERATOR_RUNBOOK.md)
- Milestone status and caveats: [ROADMAP.md](ROADMAP.md)

## Repository Boundaries

- Source of truth: `src/main`, `src/test`, `scripts`, and the checked-in docs/specs under `docs/`.
- Output-heavy areas: packaged layouts under `dist/`, scratch benchmark data under parts of `data/`, and native build directories under `src/main/native/`.
- Some legacy native verification binaries and packaged artifacts are still checked in; treat them as output surfaces, not canonical implementation.
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
The source launcher and packaged launcher bind to `127.0.0.1` by default; pass `-Host 0.0.0.0` only if you intentionally want LAN exposure.

Build a standalone Windows release that runs without `sbt`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/release-hand-history-web.ps1
```

That assembles `dist/hand-history-web`, writes `bin/run-hand-history-web.ps1`, validates the expected static assets, and smoke-checks `GET /`, `GET /api/health`, async `POST /api/analyze-hand-history`, and oversized-upload rejection from the packaged layout.

For simulator and benchmark workflows, use [docs/OPERATOR_RUNBOOK.md](docs/OPERATOR_RUNBOOK.md).

For milestone status, read [ROADMAP.md](ROADMAP.md) together with the DDRE caveats above.
