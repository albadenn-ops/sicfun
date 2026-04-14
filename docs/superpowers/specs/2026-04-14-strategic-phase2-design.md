# Strategic Phase 2 - Formulation Grounding and Benchmark Gates

**Date**: 2026-04-14
**Status**: Outline
**Depends on**: `docs/superpowers/specs/2026-04-13-strategic-overlay-design.md`

## Purpose

Phase 1 made `StrategicEngine` a production overlay on top of real upstream EVs.
That solved the biggest runtime problem: strategic no longer selects actions by
inventing its own poker values.

Phase 2 addresses the remaining gap:

1. Replace the toy poker formulations that still back the deprecated solver and
   certification paths.
2. Put repeatable measurement in front of rollout so formulation work cannot run
   ahead of evidence.

This document is an outline, not an implementation plan.

## Current State

Phase 1 is now the live runtime path:

- `StrategicEngine.decide(gameState, candidates, upstreamEvs)` is the production
  entry point.
- `HeroDecisionPipeline`, `TexasHoldemPlayingHall`, `AcpcMatchRunner`,
  `SlumbotMatchRunner`, and `StrategicAdvisorBridge` all route strategic
  decisions through the overlay path.
- The old 2-arg `StrategicEngine.decide(...)` remains only as a deprecated
  formulation-driven path.

The remaining technical debt is concentrated here:

- `PokerPomcpFormulation` still uses coarse public-state buckets, class-prior
  policy tables, and linear showdown equity heuristics.
- `PokerPftFormulation` still uses `heroBucket / 9.0`, deterministic street
  transitions, and handcrafted reward tables.
- Existing benchmark scripts still lag behind runtime support:
  `scripts/match/run-slumbot-benchmark.ps1` and
  `scripts/match/run-hall-matchups.ps1` still only accept `adaptive,gto`.

## Phase 2 Guardrails

These constraints apply to the whole phase:

1. The Phase 1 overlay remains the production action-selection path until the
   benchmark gates in Track B pass.
2. Formulation work lands behind explicit scope boundaries. No "quiet"
   rewiring of runtime hero decisions back to POMCP or PFT action selection.
3. Benchmarks must capture a baseline before the first behavior-changing
   formulation patch merges.
4. Phase 2 may improve certification and offline strategic analysis first.
   Re-exposing formulations to runtime is a separate decision after the gates.

## Track A - Poker-Grounded Formulation Replacement

### Goal

Replace the toy value/state plumbing in `PokerPomcpFormulation` and
`PokerPftFormulation` with poker-grounded state, transition, observation, and
value inputs that are consistent with the real runtime engines.

### Non-goals

- No replacement of the Phase 1 overlay as the default runtime path.
- No CFR solver rewrite.
- No GPU/native solver rewrite.
- No changes to strategic kernel math unless the formulation integration
  strictly requires a bridge or adapter.

### Design Direction

Track A should introduce a shared formulation-facing poker input layer instead
of letting each formulation keep inventing its own proxy data.

That shared input layer should provide:

- Real action-history state, not just street or bucket summaries.
- Real terminal payoffs from current `GameState` and legal actions.
- Real or calibrated value estimates sourced from existing runtime engines or
  reusable equity/value components.
- An explicit distinction between:
  - formulation inputs used for certification/offline strategic analysis
  - upstream EV inputs used by the production overlay

### Sequencing

Track A should land in this order:

#### A1. Inventory and dead-path decision

Define exactly which formulation outputs are still consumed after Phase 1:

- deprecated `StrategicEngine.decide(...)`
- certification bundle generation
- advisor diagnostics
- tests that still exercise `decideWPomcp` / `decidePftDpw`

Deliverable:

- A file-by-file inventory of remaining formulation consumers and whether each
  is runtime-critical, certification-only, or test-only.

Gate:

- No code changes until this inventory is written down.

#### A2. Shared formulation input contract

Create a small poker-grounded contract that both formulations consume.

Minimum contents:

- public game state
- legal candidate actions
- hero private information or equivalent value-query inputs
- rival belief snapshot
- reusable value/equity accessor

The contract must not depend on protocol-runner locals or hall-only state.

Gate:

- Both formulations compile against the shared contract, even if behavior is
  still unchanged at this stage.

#### A3. PftDpw grounding first

Ground `PokerPftFormulation` before `PokerPomcpFormulation`.

Reason:

- `PokerPftFormulation` is the simpler tabular model and is closer to the
  current certification path.
- It is the lower-risk place to validate the shared contract and the benchmark
  harness before touching the more approximate POMCP path.

Required replacements:

- remove `heroBucket / 9.0` reward proxy
- remove deterministic "always advance street" transitions
- replace handcrafted reward shaping with poker-grounded terminal/value inputs
- keep observation distributions honest about what is inferred vs exact

Gate:

- Certification/offline tests using the PFT path pass with poker-grounded
  inputs and no proxy bucket reward path remaining in the main formulation code.

#### A4. WPomcp grounding second

Ground `PokerPomcpFormulation` after the PFT path is stable.

Required replacements:

- replace coarse linear showdown-equity heuristic with a real or calibrated
  value/equity source
- reduce or eliminate static class-prior policy tables where runtime evidence
  can be injected
- make action effects and terminal handling reflect real legal-action history
  instead of one-round toy assumptions wherever the solver interface allows

Gate:

- Approximate certification/offline tests pass with no linear showdown equity
  heuristic left in the main formulation code path.

#### A5. Certification rebinding

Once both formulations are grounded, update the certification path to consume
the new formulation outputs explicitly rather than relying on old proxy
assumptions.

This includes:

- robust lower bounds
- pointwise exploitability reporting
- any four-world decomposition values that still depend on the toy
  formulations

Gate:

- Certification artifacts describe poker-grounded values, not proxy bucket
  values or handcrafted reward scales.

## Track B - Repeatable Benchmarks and Rollout Gates

### Goal

Create a measurement track that captures baseline behavior before formulation
changes, then blocks broader rollout unless quality, safety, and runtime
stability stay within agreed limits.

### Non-goals

- No one-off benchmark runs with no saved inputs or summary outputs.
- No relying on anecdotal "played fine for 100 hands" evidence as a rollout
  decision.

### Benchmark Layers

Track B should have two layers.

#### B1. Decision-corpus benchmark

Create a fixed corpus of decision spots that can be replayed deterministically.

Minimum corpus coverage:

- heads-up preflop
- heads-up postflop
- multiway postflop
- at least one spot that triggers non-trivial strategic beliefs
- at least one spot with certification/robust-bound output available

Per-spot metrics:

- selected action
- upstream best action
- per-action EVs when available
- overlay change rate
- soft-veto rate
- latency

Purpose:

- detect formulation-value drift without needing full match variance
- compare PFT/WPomcp outputs to overlay and upstream references

#### B2. End-to-end match benchmark

Use existing runtime entry points instead of inventing a parallel benchmark
universe.

Required surfaces:

- Hall self-play via `TexasHoldemPlayingHall`
- ACPC local match flow via `AcpcMatchRunner` plus dealer harness
- Slumbot benchmark flow via `scripts/match/run-slumbot-benchmark.ps1`

Required script work:

- extend `scripts/match/run-hall-matchups.ps1` to accept `strategic`
- extend `scripts/match/run-slumbot-benchmark.ps1` to accept `strategic`
- add or document a repeatable ACPC benchmark invocation

Per-run metrics:

- mbb/hand and bb/100
- action distribution
- overlay veto rate
- exploitation beta trajectory
- mean and p95/p99 decision latency
- protocol errors or invalid-action failures

### Benchmark Gates

These gates are mandatory:

#### G1. Baseline capture gate

Before the first behavior-changing formulation patch merges, capture a baseline
for:

- current Phase 1 strategic overlay
- adaptive
- gto where available

Artifacts must be saved under `data/` or another documented output root with
seed, command line, and summary files.

#### G2. Determinism gate

With fixed seeds and learning disabled, benchmark inputs must replay
deterministically enough to support before/after comparison.

At minimum:

- decision-corpus results must be identical
- end-to-end summary outputs must be reproducible on the same machine and
  configuration

#### G3. Quality gate

Track A changes do not roll forward unless they preserve or improve quality
against the captured baseline on the agreed benchmark suite.

The exact thresholds can be set in the implementation plan, but the gate must
cover:

- no material regression in hall self-play strategic-vs-adaptive
- no material regression in hall self-play strategic-vs-gto
- no obvious protocol-runner collapse in ACPC or Slumbot runs

#### G4. Latency gate

Grounded formulations and certification work must not cause uncontrolled
runtime latency growth on the Phase 1 overlay path.

The benchmark suite must report mean and tail latency, and the plan must define
explicit acceptable deltas relative to the baseline capture.

#### G5. Safety/diagnostic gate

Overlay diagnostics and certification diagnostics must remain interpretable.

Specifically:

- veto-rate changes must be explainable
- robust-bound outputs must remain finite
- no silent fallback to zero EVs or proxy reward scales

## Cross-Track Dependencies

Track B starts first.

Required order:

1. B1/B2 benchmark plumbing
2. G1 baseline capture
3. A1 shared inventory
4. A2 shared formulation input contract
5. A3 PFT grounding
6. Re-run Track B gates
7. A4 WPomcp grounding
8. Re-run Track B gates
9. A5 certification rebinding
10. Re-run Track B gates

This ordering is deliberate. Track A is not allowed to outrun measurement.

## Deliverables

Phase 2 should produce:

- one design-level implementation plan per track
- benchmark commands/scripts checked into the repo
- saved baseline benchmark artifacts
- grounded formulation code for PFT and WPomcp
- updated certification outputs that no longer depend on toy reward proxies
- a short rollout note stating whether grounded formulations remain
  certification/offline-only or are ready for a later runtime exposure decision

## Acceptance Criteria

Phase 2 is complete when all of the following are true:

1. No main formulation path still depends on `heroBucket / 9.0`, linear
   showdown equity heuristics, or deterministic street-advance reward proxies.
2. Strategic benchmark scripts and benchmark documentation support
   `strategic` mode explicitly.
3. Baseline and post-change benchmark artifacts exist and are comparable.
4. Benchmark gates show no unacceptable quality, safety, or latency regression.
5. The production runtime path is still the Phase 1 overlay unless a separate
   follow-up explicitly approves broader formulation exposure.

## Open Decisions For The Implementation Plan

The implementation plan should settle these points explicitly:

- whether the shared formulation input layer is a new type or a thin adapter
  over existing engine/runtime types
- whether PFT and WPomcp share the same value source API or only the same state
  snapshot API
- whether ACPC benchmarking gets a new helper script or reuses the existing
  dealer/start scripts directly
- the exact quality and latency thresholds for Gates G3 and G4
