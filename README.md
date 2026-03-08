# SICFUN

SICFUN is a research codebase for Texas Hold'em analysis, self-play simulation, Bayesian range inference, and optional native acceleration experiments.

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
- Not a clean release repository yet; benchmark outputs and native build products are generated artifacts, not the source of truth.

## Fast Validation

```powershell
powershell -ExecutionPolicy Bypass -File scripts/prove-pipeline.ps1 -Quick
```

For simulator and benchmark workflows, use [docs/OPERATOR_RUNBOOK.md](docs/OPERATOR_RUNBOOK.md).

For milestone status, read [ROADMAP.md](ROADMAP.md) together with the DDRE caveats above.
