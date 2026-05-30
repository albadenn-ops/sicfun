# P0 — Multiway bb/100 Measurement Harness (Design)

**Date:** 2026-05-30
**Status:** Design — pending user review
**Depends on:** `docs/specs/SICFUN-v0_31_1-corrected.md`, `docs/superpowers/specs/2026-04-14-strategic-phase2-design.md` (Track B), `docs/superpowers/plans/2026-04-14-strategic-phase2-track-b.md`
**Context:** First sub-project (P0) of the production mandate — turn sicfun into a working multiway (≤9-max) exploitative Bayesian solver. P0 is the measurement that every later fix is judged against.

## Purpose

Measure the **current live engine's** multiway (up to 9-max) win rate as an honest, statistically-sound baseline — with **no engine changes**. This is the Phase 2 G1 baseline that must land before any correctness fix (P1+), and it is the arbiter the whole exploitative program is judged by.

## Why results, not proofs

In an incomplete-information game, strategy correctness is an **empirical** claim settled by results over samples. Outcome variance is **intrinsic** — it comes from hidden rival holdings and from the board runout — and cannot be removed without knowing all hidden information, which dissolves the game. Therefore P0 **estimates** win rate with confidence intervals; it does not seek reproducible outcomes.

## Determinism, variance, and what "reproducible" means

Two distinct notions, not to be conflated:

1. **Strategy determinism (required, tested).** Same information set + same seed for internal Monte-Carlo sampling → the same decision, or the same *mixed-action distribution* for a mixed strategy. The policy must not be erratic. This is a hard requirement and is unit-tested.
2. **Outcome variance (intrinsic, NOT reproducible).** The true bb/100 against the field is a random variable; its variance is the game's. We estimate it over many **independent** deals with bootstrap confidence intervals.

Seeding the master RNG buys two things — **not** outcome reproducibility:
- **(a) Replay** for debugging and regression over identical deals.
- **(b) Paired comparison / common random numbers (CRN).** Run version A and version B over the *same* deal sequence so board/holding luck cancels in the **difference**. The bb/100 **delta** between versions has far lower variance than two independent runs, so a real improvement is detectable with far fewer hands — without pretending the absolute outcome is deterministic.

Variance-reduction techniques (duplicate/mirror dealing now; AIVAT-style control variates later) lower the **estimator's** variance (tighter CI per N hands), not the game's intrinsic variance.

**Revised gate** (supersedes the earlier "G2: identical bb/100"): (1) a strategy-determinism test; (2) seeded replay enabling CRN + debugging. The quality verdict is the **CI on bb/100 over an independent sample**, plus the CRN paired-delta for A/B comparisons.

## Scope

- **In scope:** measurement only.
- **Out of scope (firmly):** no engine / overlay / spec changes; no P1 baseline calibration or wiring fixes. Those are gated *behind* the number this harness produces.

## Track A — controlled self-play baseline

- **Reuse:** `TexasHoldemPlayingHall` (already ≤9-max self-play; emits `heroBbPer100`, `perVillainNetChips`; configurable `villainPool: Vector[VillainProfile]`), `OverlayMetricsAccumulator`/`OverlayStats`, `MatchRunnerSupport` summaries.
- **Canonical exploitable 9-max field (new):** a checked-in, named field of `VillainProfile` archetypes — calling-station, nit, maniac, TAG/LAG — in a fixed default seat mix, version-stamped and configurable. **This field is the operational definition of "the field" in the acceptance bar.**
- **Statistics layer (the real new work):** bootstrap confidence intervals on bb/100; variance reduction via duplicate/mirror dealing; a CRN paired-delta utility for comparing two engine configs/versions on identical deal sequences.
- **Hero modes captured:** `strategic` (the live overlay — the headline number) and `adaptive` (comparison), at `playerCount = 9` vs the canonical field.

## Track B — counterfactual EV on real hands

- **Reuse:** `HandHistoryAnalyzer`, which already produces per-decision `recommendedEv`, `actualEv`, and `evDifference`.
- **Aggregator (new):** aggregate `evDifference` over a real multiway hand-history corpus into a **counterfactual bb/100 delta vs the real field** (would-following-sicfun), with a bootstrap CI.
- **Corpus:** operator-supplied (real histories stay local per repo policy); a small synthetic-but-realistic multiway sample is checked in for tests.

## Components & data flow

- **New:**
  - a statistics module: bb/100 point estimate + bootstrap CI + duplicate/mirror variance reduction + CRN paired-delta;
  - the canonical field definition (data, version-stamped);
  - the Track-B aggregator over `HandHistoryAnalyzer` decisions;
  - a capture entry/script that runs both tracks and writes artifacts.
- **Reuse:** the hall, overlay metrics, match-runner summaries, the hand-history analyzer.
- **Artifacts:** seed, command line, config, and summary (bb/100 + CI, action distribution, overlay change/veto rates, latency p95/p99) saved under a documented `data/` root.

## Testing

- **Strategy determinism:** same information set + seed → identical decision (exact assert).
- **CRN paired-delta sanity:** same seed, two identical engine configs → delta ≈ 0 with a tight CI.
- **Rigged-field edge detection:** a trivially exploitable field (e.g. always-fold or always-call bots) → hero bb/100 strongly positive with CI clearing zero — proves the harness can *detect* a known edge.
- **CI sanity:** synthetic results of known variance → expected CI width / coverage.

## Acceptance criteria

One command produces — reproducibly in the CRN/replay sense, not the outcome sense:
1. `strategic` and `adaptive` bb/100 + bootstrap CI at 9-max vs the canonical field;
2. a counterfactual bb/100 delta + CI on a real-hands corpus;
3. saved artifacts (seed / command line / config / summary);
4. the rigged-field test passing (harness detects a known edge);
5. the strategy-determinism test passing.

## Open decisions for the implementation plan

- Exact default field archetype mix and seat assignment at 9-max.
- Hands count / stopping rule: fixed N vs. sequential sampling until a target CI width.
- CI method: bootstrap (recommended) vs. closed-form normal; number of resamples.
- Whether to add AIVAT-style control variates now or defer (defer recommended for P0).
- Real-hands corpus source, and the construction of the checked-in synthetic sample.
