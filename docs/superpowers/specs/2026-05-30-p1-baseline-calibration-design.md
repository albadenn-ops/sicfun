# P1 — Calibrated, Spot-Conditioned Baseline (Def 9/10) — Design

**Date:** 2026-05-30
**Status:** Design — pending user review
**Depends on:** `docs/specs/SICFUN-v0_31_1-corrected.md` (Def 9 real baseline, Def 10 attributed baseline), the P0 measurement harness (`docs/superpowers/specs/2026-05-30-p0-multiway-winrate-harness-design.md`), and the 2026-05-30 v0.31.1 audit.
**Context:** Sub-project P1 of the production mandate. **Behavior-changing and gated** (see Validation).

## Purpose

Replace the live baseline `config.actionPriors.getOrElse((cls, cat), 0.25)` — 16 hardcoded constants keyed only on `(StrategicClass, action category)`, uncalibrated and board/street-blind — with a **calibrated, board/street-conditioned `RealBaseline` (Def 9)**. The baseline is the anchor the engine *diverges from to exploit*; the audit identified it as the highest-leverage correctness fix, because the exploit signal is measured as divergence from this anchor, so a spot-agnostic baseline miscalibrates exploitation in every spot.

## The key realization (shapes the whole design)

- The baseline's purpose is a **class-discriminating** action likelihood `Pr(a | c, x^pub)` — that is what lets the rival's class posterior update. It **must be per-class** (a class-agnostic table carries no class information).
- A player's class is **holding-defined** (Def 1–4: value / bluff / structural-bluff depend on the actual hand on that board). So **per-class labels require revealed holdings → the by-class signal can only come from showdowns.** The full hand-history corpus contributes **board/street structure + smoothing**; showdowns supply the class labels.
- It is **live** (feeds the ref/attrib tempered likelihoods → per-rival posteriors → `StrategicOverlay` penalties). So this is behavior-changing and gated.

## Architecture

Offline calibration tool → baseline artifact → `RealBaseline` implementation → engine rewiring. (Mirrors the existing action-model train → artifact → load pattern.)

## Components

1. **`BaselineCalibrationTool`** (new, offline; sibling to `BatchTrainer`/`TrainPokerActionModelCli`): ingest a hand-history corpus via `HandHistoryImport` → for each **showdown-revealed** decision, classify `(revealed holding, board, street)` → `StrategicClass` using SICFUN's existing equity-based class logic → tally `(class × board-bucket × street × action-category)` counts → normalize per `(class, board-bucket, street)` → smooth → write a baseline artifact.
2. **Board-bucket scheme:** postflop `{board-pairing (unpaired / paired / trips-on-board) × suitedness (rainbow / two-tone / monotone) × high-card bucket (A-high / broadway / middle / low)} × street {Flop, Turn, River}`; preflop uses a position/no-board bucket. Deliberately coarse to keep cells populated. (Exact buckets pinned in the plan.)
3. **Backoff / smoothing ladder** (guarantees a valid distribution in every spot and **never worse than today**): a cell with count < `minCount` backs off to `(class × street × action)` → `(class × action)` = the current 16 constants (floor) → uniform over legal actions. Laplace/Dirichlet smoothing within populated cells.
4. **Baseline artifact:** a serialized table (JSON, like model artifacts) + metadata (corpus id, hand/showdown counts, calibration date, `minCount`, board-bucket scheme version). Versioned.
5. **`RealBaselineImpl`** (implements the existing, currently-unimplemented `RealBaseline` trait at `strategic/safety/Baseline.scala`): loads the artifact; `probability(cls, action, sizing, publicState)` looks up `(cls, boardBucket(publicState.board), street(publicState.street), action)` with the backoff ladder. Sizing (λ) bucketed to action category for now.
6. **Rewiring:** `PosteriorAttributedBaseline` currently wraps `actionPriors`; rewire it and the engine's ref/attrib-likelihood builders (`StrategicEngine` ~lines 1144 `actionPrior`, 1152 attrib-likelihood, 1178 ref-likelihood) to consume `RealBaselineImpl` when a baseline artifact is configured, falling back to the constants when none is.

## Data flow

hand-history corpus → `BaselineCalibrationTool` → artifact → (engine config `baselinePath`) → `RealBaselineImpl` → ref/attrib tempered likelihoods → rival posteriors → `StrategicOverlay` penalties.

## Class labeling (the crux)

Reuse SICFUN's existing equity-based `StrategicClass` classification, applied to each showdown-revealed `(holding, board, street)`. (The plan pins the exact classifier symbol used by the engine for hero, applied here to revealed villain holdings.) Non-showdown decisions get no class label and are excluded from per-class counts (they may inform class-marginal smoothing).

## Scope

- **In:** the calibration tool, the artifact, `RealBaselineImpl`, and the rewiring of Def 9 (and Def 10 via `PosteriorAttributedBaseline` inheriting the conditioned base).
- **Out (firmly):** no change to overlay penalty/veto logic; no new or changed `StrategicClass` definitions; no kernel/inference math changes (only the baseline *input* is upgraded); fine-grained sizing (λ) deferred to a follow-up.

## Validation / gates

- **G1 (baseline-first):** the P0 harness baseline MUST be captured (the before-numbers) before this merges — per the Phase-2 doctrine.
- **G3 (quality):** after rewiring, re-run the P0 harness; the calibrated baseline must **preserve-or-improve** bb/100 vs the captured baseline. A regression is a finding to investigate, not an auto-ship.
- **G5 (diagnostics):** overlay veto/change rates must stay interpretable; the backoff floor guarantees a valid distribution (no silent zero/degenerate baseline).
- **Determinism:** fixed artifact ⇒ deterministic baseline lookups.

## Testing (for the plan)

- **Calibration tool:** on a known corpus (the P0 synthetic 9-max sample + added showdowns), produces the expected per-`(class, bucket, street)` frequencies; backoff fills sparse cells; each cell's distribution sums to 1.
- **`RealBaselineImpl`:** lookups return artifact values; the backoff ladder fires correctly; always a valid distribution; equals the constants when no artifact is configured.
- **Integration:** the engine with a calibrated baseline produces different (and board-sensible) class posteriors than with the constants on a textured spot — the whole point.
- **No-regression:** with no artifact configured, behavior == today (the constants).

## Open decisions for the plan

- Exact board-bucket granularity and `minCount` backoff thresholds.
- Smoothing method (Laplace / Dirichlet α).
- Artifact format and load path (config flag).
- Calibration corpus: operator-supplied (real histories stay local) + a small checked-in sample for tests.
- Whether to class-label NON-showdown decisions via inferred range (default: no — showdown-only labels + marginal smoothing).

## Honesty notes

- The by-class signal is **showdown-derived** (documented); thin board cells back off, so the baseline is **never worse than today's constants** where data is absent.
- This calibrates the **empirical field baseline** (observed play), matching the `RealBaseline` trait's stated "observed frequency distribution" intent — it is explicitly **not** a GTO/solver baseline.
