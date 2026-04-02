# Adaptive Proof Harness Design

## Problem

The adaptive player (hero) uses Bayesian range inference and CFR-based GTO solving
to detect and exploit opponent tendencies. We need a reproducible, growable test
fixture that proves:

1. The adaptive player can pinpoint each villain's leak during live play
2. The exploits it attempts actually yield positive EV (or at minimum, it's not a losing player)
3. The edge comes from exploitation, not variance (GTO controls should be ~break-even)

The hand history persists as a reusable artifact for web UI tests.

## Table Composition (9 players)

| Seat | Role | Baseline | Leak | Severity |
|------|------|----------|------|----------|
| Hero | Adaptive player | `RealTimeAdaptiveEngine` (CFR internally) | None | — |
| Villain01 | Leaker | `EquityBasedStrategy` | `OverfoldsToAggression` | 0.20 |
| Villain02 | Leaker | `EquityBasedStrategy` | `Overcalls` | 0.25 |
| Villain03 | Leaker | `EquityBasedStrategy` | `OverbluffsTurnBarrel` | 0.18 |
| Villain04 | Leaker | `EquityBasedStrategy` | `PassiveInBigPots` | 0.30 |
| Villain05 | Leaker | `EquityBasedStrategy` | `PreflopTooLoose` | 0.22 |
| Villain06 | Leaker | `EquityBasedStrategy` | `PreflopTooTight` | 0.15 |
| Villain07 | Shallow GTO | TAG archetype (`ArchetypeVillainResponder`) | None | — |
| Villain08 | Validatable GTO | `CfrVillainStrategy` (no fallback) | None | — |

- All 6 leakers have one distinct leak type with varying subtle severities (0.15–0.30)
- Shallow GTO: TAG archetype via `ArchetypeVillainResponder` — no leak injection path, plays
  a reasonable GTO approximation. "Decent reg" control.
- Validatable GTO: full CFR equilibrium, no fallback — "provably GTO" control and false-positive canary

## Seat Reshuffling

Every 500 hands:
- All players reshuffle seat positions randomly
- All stacks reset to 100bb max buy-in (unlimited buyins)
- Player names and hero's accumulated reads persist across reshuffles
- Hero's `RealTimeAdaptiveEngine` observations carry over (knowledge follows identity)

## Approach: Thin Adapter on TexasHoldemPlayingHall

### PlayingHall Extension

Minimal changes to `TexasHoldemPlayingHall`:

1. **New enum case:** `VillainMode.LeakInjected(leakId: String, severity: Double)`

2. **Extended `parseVillainModeToken`:** New syntax `leak:<type>:<severity>`:
   - `leak:overfold:0.20` → `VillainMode.LeakInjected("overfold-river-aggression", 0.20)`
   - `leak:overcall:0.25` → `VillainMode.LeakInjected("overcall-big-bets", 0.25)`
   - `leak:turnbluff:0.18` → `VillainMode.LeakInjected("overbluff-turn-barrel", 0.18)`
   - `leak:passive:0.30` → `VillainMode.LeakInjected("passive-big-pots", 0.30)`
   - `leak:prefloploose:0.22` → `VillainMode.LeakInjected("preflop-too-loose", 0.22)`
   - `leak:prefloptight:0.15` → `VillainMode.LeakInjected("preflop-too-tight", 0.15)`
   - `gto-shallow` → `VillainMode.Archetype(PlayerArchetype.Tag)` with zero noise

3. **Extended `decideVillain`:** When mode is `LeakInjected`:
   - Compute equity-vs-random for the villain's hand (MC estimate)
   - Get GTO baseline action via `EquityBasedStrategy`
   - Build `SpotContext` from current `GameState` + equity
   - Pass through `LeakInjectedVillain.decide()` for potential deviation
   - Requires tracking per-position `ActionLine` for `SpotContext` construction

4. **Extended `villainModeLabel` / `villainModeSlug`:** Add cases for new mode

### AdaptiveProofHarness

**New file:** `src/main/scala/sicfun/holdem/validation/AdaptiveProofHarness.scala`

Orchestrator that configures, runs, and evaluates the proof.

**Responsibilities:**

1. **Configure 9-player table:** Builds the `--villainPool` string and other PlayingHall args
2. **Run in 500-hand blocks:** Invokes `TexasHoldemPlayingHall.run()` per block
3. **Seat reshuffle between blocks:** Random hero position, stacks reset to 100bb
4. **Ground truth sidecar:** `data/adaptive-proof/ground-truth.json` — villain identities,
   per-block per-villain chip deltas
5. **Checkpoint/resume:** `data/adaptive-proof/checkpoint.json` — next hand number, last seed,
   cumulative stats. New runs derive a fresh seed (not deterministic replay)
6. **Evaluation report:** Per-villain aggregate bb/100, second-half vs first-half learning signal,
   GTO control bb/100

### Resume Logic

On startup, if `checkpoint.json` exists:
- Load accumulated state (hand count, per-villain chip deltas)
- Derive a new RNG seed from the previous seed + current timestamp
- Invoke PlayingHall continuing from the next hand number
- Append to the existing PokerStars export file

Each continuation explores new lines (non-deterministic), building a richer corpus.

## Output Directory

```
data/adaptive-proof/
  checkpoint.json
  ground-truth.json
  report.txt
  hall-out/
    review-upload-pokerstars.txt    # appendable hand history
    hands.tsv
    learning.tsv
    models/
```

## Test

**File:** `src/test/scala/sicfun/holdem/validation/AdaptiveProofHarnessTest.scala`

Runs 1 block (500 hands). Tagged `munit.Slow`.

**Single-block assertions (loose, 500 hands is high variance):**
1. Hall run completes without error
2. Hand history file exists and is non-empty
3. Ground truth sidecar written with correct villain count
4. Hero aggregate bb/100 >= -15 (not catastrophically broken)
5. Hero bb/100 vs GTO control < +30 (no bug-inflated edge)
6. Checkpoint written with correct next hand number
7. Hand history parseable by `HandHistoryImport.parseText()`

**Multi-block assertions (tighten with accumulation, checked by harness main):**
- Hero bb/100 > 0 against each leaker (when total hands >= 1000)
- Second-half bb/100 >= first-half bb/100 against leakers (when total hands >= 2000)
- |bb/100| < 10 against GTO control (when total hands >= 2000)

## Files Changed

| File | Change |
|------|--------|
| `TexasHoldemPlayingHall.scala` | `VillainMode.LeakInjected`, `decideVillain` leak path, `parseVillainModeToken` extended, per-position `ActionLine` tracking |
| **New:** `AdaptiveProofHarness.scala` | Orchestrator, checkpoint/resume, ground truth, evaluation |
| **New:** `AdaptiveProofHarnessTest.scala` | Single-block 500-hand test |

| `TexasHoldemPlayingHall.scala` | Add `perVillainNetChips: Map[String, Double]` to `HallSummary`, accumulate per-villain hero net during `HallRunner.playHand` |

No changes to `InjectedLeak.scala`, `LeakInjectedVillain.scala`, `SpotContext.scala`, or `VillainStrategy.scala` — all reused as-is.
