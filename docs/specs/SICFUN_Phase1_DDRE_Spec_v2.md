# SICFUN Phase 1 DDRE Spec (v2 Draft)

Status: Draft for engineering gate review  
Date: 2026-03-05  
Replaces: `SICFUN_Phase1_DDRE_Spec.docx` for implementation readiness

## 1) Scope and interface contract

Phase 1 augments Bayesian posterior update with DDRE and combines both signals.  
No changes to CFR backbone, subgame solving, or action abstraction.

Implementation target in current codebase:
- Keep bunching prior computation unchanged.
- Keep Bayesian updater path active.
- Add DDRE inference path.
- Fuse Bayesian and DDRE posteriors into one final posterior consumed by existing callers.
- Preserve all existing caller contracts (`RangeInferenceEngine`, `RealTimeAdaptiveEngine`, advisor/loop/hall flows).

Current call site reference:
- `src/main/scala/sicfun/holdem/RangeInferenceEngine.scala`

## 2) Corrected DDRE model definition (implementable)

### 2.1 Problem representation

We model a posterior distribution over the fixed 1326 hole-card combos (`HoleCardsIndex`).

- Output domain: `R^1326` logits.
- Final posterior: masked softmax over legal combos only.

### 2.2 Diffusion formulation (consistent with 1326-logit output)

Use **continuous diffusion over a 1326-logit vector**, not token-per-slot categorical IDs.

- Clean target `x0`: target posterior logits (from training labels).
- Forward noising: Gaussian noising schedule on logits.
- Reverse denoising network: predicts denoised logits conditioned on encoded context.
- Inference output: one 1326-logit vector -> blocker mask -> renormalized posterior.

This removes the prior contradiction between:
- “1326 discrete slots with 1327 token values each”
- and “single 1326-logit posterior output”.

### 2.3 Context encoder inputs

Conditioning context must include:
- Action history (ordered, with street boundaries)
- Public board cards
- Pot/stack/SRP-style scalars
- Positional/meta features

Use the existing runtime observation path:
- `Seq[VillainObservation]` with `GameState` (already contains chronological `betHistory`)

### 2.4 Fusion contract (Bayesian + DDRE)

Let:
- `p_bayes`: Bayesian posterior (existing path)
- `p_ddre`: DDRE posterior (new path)

Final decision posterior:
- `p_final = normalize((1 - alpha) * p_bayes + alpha * p_ddre)`

Where `alpha` is configurable and may be dynamic:
- static default (for example `0.2`)
- uncertainty-aware adjustment (optional Phase 1 extension)

Hard legal-combo mask is applied after fusion, then renormalized.

## 3) Corrected blocker and combo-count rules

### 3.1 Mandatory legal-combo mask

At inference, zero probability for any villain combo containing:
- Any board card
- Any known hero card

Mask is deterministic and applied **after model output** (hard guardrail), then renormalize.

### 3.2 Correct unblocked combo count

Legal combo count formula:
- `C(52 - n_board - n_hero_known, 2)`

Example (full board + known hero hand):
- `n_board = 5`, `n_hero_known = 2` -> `C(45, 2) = 990`

The previous “1,081 combos remain unblocked” statement is invalid for that scenario.

## 4) Offline metrics (operationally defined)

The prior “KL vs true posterior” was ambiguous from observable-only data.  
Replace with metrics that are computable from available logged/self-play data.

### 4.1 Primary metrics

1. **NLL of realized villain hand** (offline self-play/eval only)
   - For each decision point with known villain cards `h*`:
   - `NLL = -log p_final(h* | context)`

2. **KL diagnostics between components and fused output**
   - `KL(p_ddre || p_bayes)` for model divergence tracking.
   - Optional: `KL(p_final || p_bayes)` to monitor production drift vs baseline.

3. **Blocker violation rate**
   - Must be exactly zero after hard mask.

4. **Latency and degraded-mode behavior**
   - p50/p99 inference latency
   - DDRE unavailable/degraded rate (cases where `alpha` is forced to `0`)

### 4.2 Optional oracle metric (explicitly optional)

If “true posterior” is required, define a reproducible oracle construction procedure first.  
Until then, do not use “KL vs true posterior” as a gate metric.

## 5) Rollout semantics (non-contradictory)

### 5.1 Modes

- `off`: Bayesian only
- `shadow`: Bayesian drives decisions, DDRE runs in parallel for logging/metrics only
- `blend-canary`: fused posterior drives decisions for selected traffic
- `blend-primary`: fused posterior drives decisions by default

### 5.2 Required behavior by mode

`shadow` must **not** drive live decisions.  
Any decision-driving phase must use fused posterior semantics, not DDRE-only semantics.

## 6) Fallback policy (explicit)

When fusion is active (`blend-canary`/`blend-primary`), evaluate DDRE validity first.

If DDRE is invalid, set `alpha = 0` and continue with Bayesian posterior.  
If Bayesian is invalid (unexpected), set `alpha = 1` only if DDRE is valid; otherwise fail closed.

DDRE invalid conditions:
- inference exception
- timeout breach
- invalid output (`NaN`, negative mass before normalization, zero legal mass)
- entropy below configured minimum (degeneracy guard)

All degraded-mode events must log:
- reason category
- latency
- mode
- component status (`bayes_ok`, `ddre_ok`, `alpha_applied`)

## 7) Engineering acceptance checklist before implementation start

1. Model formulation approved (continuous-logit diffusion).
2. Blocker rule approved (`board + hero` mask, deterministic post-pass).
3. Metrics approved (NLL + KL-vs-Bayes + blocker violations + latency/fallback).
4. Rollout mode semantics approved (`shadow` not decision-driving).
5. Current flakiness baseline acknowledged (`sbt test` order sensitivity/timeouts) and isolated from DDRE workstream.

## 8) Minimal Phase 1 implementation plan (updated)

M1. Data pipeline extension
- Add DDRE training export containing context + baseline posterior + realized villain hand.

M2. Model training prototype
- Train diffusion-logit model offline (Python), export ONNX.

M3. Native runtime bridge
- Add ONNX inference runtime wrapper and Scala provider facade.

M4. Integration
- Wire DDRE + Bayesian fusion into `RangeInferenceEngine.computePosterior(...)` behind mode flag.

M5. Validation
- Offline metrics + regression tests + shadow-mode telemetry.

M6. Canary rollout
- Controlled fused-posterior decision-driving traffic with degraded-mode auto-handling.

M7. Primary rollout
- Fused posterior primary, Bayesian-only mode retained as operational kill switch.
