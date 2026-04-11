# Adaptive Proof Harness Design

**Date**: 2026-03-30
**Goal**: add a proof harness that demonstrates the adaptive hero can earn positive EV against supported leak-injected opponents in controlled heads-up play, stays near break-even against a CFR control, and emits parseable hand histories that can be reused for UI and profiling spot checks.

## Problem

The repo already has two different simulation paths:

- `sicfun.holdem.validation` provides a heads-up leak-injection stack:
  `HeadsUpSimulator`, `InjectedLeak`, `LeakInjectedVillain`, `VillainStrategy`,
  and `PokerStarsExporter`.
- `TexasHoldemPlayingHall` provides the larger multiway hall loop.

The original version of this spec tried to prove the claim inside
`TexasHoldemPlayingHall`, but that path currently has several structural mismatches:

- observations are keyed by seat position, not stable villain identity
- `RealTimeAdaptiveEngine` stores one shared archetype posterior, not one per villain
- `buildTableScenario` activates only a subset of villains per hand
- hall review export and writers assume a single fresh run, not resumable identity-aware sessions
- per-villain EV attribution is not defined in multiway pots

Because of those constraints, v1 should be a heads-up proof harness built on the
existing validation package. Multiway hall integration is deferred to v2.

## V1 Scope

V1 proves four narrower claims:

1. The adaptive hero can earn positive or at least non-losing EV against a supported
   set of leak-injected villains in heads-up play.
2. The same hero does not show a large artificial edge against a CFR GTO control.
3. The harness writes PokerStars-format histories that remain parseable by
   `HandHistoryImport.parseText()`.
4. The harness records explicit ground truth for the opponent roster and the EV
   outcome of each session.

V1 does **not** attempt to prove:

- stable reads across seat reshuffles in a multiway table
- per-villain attribution inside multiway pots
- crash-safe resume of an in-progress adaptive session
- full end-to-end profiler leak classification for every leak type

`ValidationRunner` already covers the profiler-oriented side of the validation package.
`AdaptiveProofHarness` is the EV-oriented counterpart.

## V1 Opponent Matrix

Each opponent is played in an independent heads-up session.

| Opponent | Role | Baseline Strategy | Leak | Severity |
|----------|------|-------------------|------|----------|
| `Villain01_overfold` | Leaker | `EquityBasedStrategy` | `OverfoldsToAggression` | 0.20 |
| `Villain02_overcall` | Leaker | `EquityBasedStrategy` | `Overcalls` | 0.25 |
| `Villain03_turnbluff` | Leaker | `EquityBasedStrategy` | `OverbluffsTurnBarrel` | 0.18 |
| `Villain04_prefloploose` | Leaker | `EquityBasedStrategy` | `PreflopTooLoose` | 0.22 |
| `Villain05_prefloptight` | Leaker | `EquityBasedStrategy` | `PreflopTooTight` | 0.15 |
| `Villain06_gto` | Control | `CfrVillainStrategy(allowHeuristicFallback = false)` | `NoLeak` | 0.0 |

Dropped from v1:

- `PassiveInBigPots`
  Reason: the current profiling pipeline already treats it as a known limitation.
  It is a bad fit for a proof harness whose artifacts are also meant for profiling
  spot checks.
- Shallow GTO via `ArchetypeVillainResponder`
  Reason: it is intentionally stochastic and is not a clean false-positive canary.
  The full CFR control is the better baseline.

## V1 Session Design

Default session size: `500` hands per opponent.

To avoid a permanent button advantage, each opponent session is split into two legs:

- Leg A: hero on the button for `250` hands
- Leg B: hero in the big blind for `250` hands

The same `RealTimeAdaptiveEngine` instance is reused across both legs of the same
opponent session so that adaptation can accumulate within that session.

Each opponent session is otherwise independent:

- one fresh hero engine per opponent
- one fixed leak definition per opponent
- one output directory per opponent
- one aggregate bb/100 result per opponent

This keeps attribution exact and removes the identity problem that exists in the
multiway hall path.

## V1 Architecture

### AdaptiveProofHarness

**New file:** `src/main/scala/sicfun/holdem/validation/AdaptiveProofHarness.scala`

Responsibilities:

1. Build the fixed opponent roster.
2. For each opponent, create one persistent `RealTimeAdaptiveEngine`.
3. Run two simulator legs for that opponent with mirrored seat assignments.
4. After each hand, feed villain responses to hero raises back into the reused
   hero engine via `observeVillainResponseToRaise`.
5. Export per-leg and combined PokerStars histories.
6. Write a ground-truth manifest and a human-readable report.

### HeadsUpSimulator changes

`HeadsUpSimulator` is the right base for v1, but it needs explicit session-state
support rather than just a cosmetic flag:

1. Explicit seat assignment for the hero:
   - support `heroIsButton: Boolean` or equivalent config
   - derive blind posting, acting order, `GameState.position`, and the `villainPos`
     passed into `RealTimeAdaptiveEngine.decide(...)` from that config instead of
     hard-coding hero on the button

2. Surface raise-response observations cleanly:
   - add an explicit representation of "villain responded to a hero raise"
   - do not force the harness to infer this indirectly from generic action traces
   - the simplest contract is a new per-hand collection such as
     `heroRaiseResponses: Vector[PokerAction]` or a richer response event record

3. Carry seat metadata into exported artifacts:
   - `HandRecord` needs enough information for downstream export to know who had
     the button in that leg
   - that can be stored on `HandRecord` itself or returned alongside it in a
     higher-level session result

No multiway behavior is needed in v1.

### PokerStarsExporter changes

`PokerStarsExporter` currently hard-codes seat 1 as the button.
For mirrored heads-up legs it needs a real API contract change, not just a text tweak:

- either accept expanded `HandRecord` seat/button fields
- or accept explicit seat assignment inputs in `exportHands(...)` / `exportChunked(...)`

That keeps the artifact parseable while making the seat-balanced session honest.

## V1 Adaptive Update Rule

Within a single opponent session:

1. Play a hand with the current hero engine state.
2. Read the explicit hero-raise response events surfaced by the simulator.
3. For each villain response to a hero raise (`Fold`, `Call`, or `Raise`),
   call `heroEngine.observeVillainResponseToRaise(...)`.
4. Reuse that updated engine for the next hand against the same opponent.

This is the minimum wiring needed to make the harness actually exercise the
adaptive response model across hands.

V1 does not serialize and restore that adaptive state across process restarts.

## V1 Outputs

Each harness run writes a self-contained run directory.

```
data/adaptive-proof/
  run-<timestamp>-<seed>/
    manifest.json
    ground-truth.json
    report.txt
    combined-history.txt
    Villain01_overfold/
      leg-button.txt
      leg-bigblind.txt
      combined.txt
    Villain02_overcall/
      leg-button.txt
      leg-bigblind.txt
      combined.txt
    ...
    Villain06_gto/
      leg-button.txt
      leg-bigblind.txt
      combined.txt
```

Why a self-contained run directory instead of checkpoint/append:

- v1 does not have serialized adaptive engine state
- block append would be misleading if a resumed run silently restarted learning
- writing immutable per-run outputs is simpler and safer

If we want resumable adaptive sessions later, that needs explicit state
serialization and belongs in a follow-up spec.

## Ground Truth Format

`ground-truth.json` records the exact opponent matrix and the measured results:

```json
{
  "handsPerOpponent": 500,
  "legsPerOpponent": 2,
  "opponents": [
    {
      "name": "Villain01_overfold",
      "leakId": "overfold-river-aggression",
      "severity": 0.20,
      "strategy": "equity-based",
      "heroNetBbPer100": 3.4,
      "heroNetBbPer100ByLeg": {
        "button": 5.1,
        "bigBlind": 1.7
      }
    },
    {
      "name": "Villain06_gto",
      "leakId": "gto-baseline",
      "severity": 0.0,
      "strategy": "cfr-no-fallback",
      "heroNetBbPer100": -0.4,
      "heroNetBbPer100ByLeg": {
        "button": 0.6,
        "bigBlind": -1.4
      }
    }
  ]
}
```

## Evaluation Report

`report.txt` is a compact human-readable summary:

```text
=== Adaptive Proof Report ===
Run seed: 12345
Hands per opponent: 500

Per-opponent results:
  Villain01_overfold      bb/100: +3.4   button: +5.1   bigBlind: +1.7
  Villain02_overcall      bb/100: +4.8   button: +6.0   bigBlind: +3.6
  Villain03_turnbluff     bb/100: +2.6   button: +3.8   bigBlind: +1.4
  Villain04_prefloploose  bb/100: +3.1   button: +4.0   bigBlind: +2.2
  Villain05_prefloptight  bb/100: +1.5   button: +2.1   bigBlind: +0.9
  Villain06_gto           bb/100: -0.4   button: +0.6   bigBlind: -1.4

Leaker average bb/100: +3.1
GTO control bb/100: -0.4
```

Primary success criteria:

- average hero bb/100 across leakers is positive
- hero bb/100 against the CFR control stays close to zero
- all exported histories parse successfully

Optional diagnostics:

- per-leg bb/100
- leak firing counts
- hero raise-response observation counts

V1 does not require a "second half > first half" learning assertion because the
current simulator path does not yet provide restart-safe adaptive state and the
sample sizes are still modest.

## Test Plan

**File:** `src/test/scala/sicfun/holdem/validation/AdaptiveProofHarnessTest.scala`

Tagged `munit.Slow`.

The unit test should exercise a reduced but representative matrix:

- one leaker
- one CFR control
- smaller hand count than the default main harness run

Assertions:

1. The harness run completes without error.
2. Per-opponent output directories are created.
3. `ground-truth.json` and `report.txt` are written.
4. Each exported history file is non-empty.
5. Each exported history file parses with `HandHistoryImport.parseText()`.
6. The control opponent records zero leak firings.
7. Reported opponent count matches the configured matrix.

The full command-line harness run, not the unit test, is responsible for the full
default matrix and higher-volume EV reporting.

## Files Changed

| File | Change |
|------|--------|
| **New:** `AdaptiveProofHarness.scala` | Orchestrator for opponent matrix, mirrored legs, adaptive updates, export, ground truth, and reporting |
| **New:** `AdaptiveProofHarnessTest.scala` | Reduced-matrix integration test for output and parseability |
| `HeadsUpSimulator.scala` | Support explicit hero seat, correct blind/position wiring, explicit hero-raise response events, and seat metadata for export |
| `PokerStarsExporter.scala` | Accept seat/button metadata and export correct button/blind lines for mirrored heads-up legs |

Expected to remain unchanged:

- `InjectedLeak.scala`
- `LeakInjectedVillain.scala`
- `SpotContext.scala`
- `VillainStrategy.scala`
- `TexasHoldemPlayingHall.scala`

## V2 Follow-up: Multiway PlayingHall Proof

Multiway hall integration remains desirable, but it needs a separate spec once the
following are designed explicitly:

1. identity-keyed villain observations in the hall
2. per-villain adaptive state in `RealTimeAdaptiveEngine`
3. full-ring activation in `buildTableScenario`
4. full-ring review export
5. defensible multiway per-villain EV attribution
6. serialized adaptive state if resumable sessions are required
