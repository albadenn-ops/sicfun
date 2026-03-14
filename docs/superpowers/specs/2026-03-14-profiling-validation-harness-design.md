# Profiling Validation Harness — Design Spec

## Problem Statement

We claim our profiling pipeline can detect exploitable patterns in opponent play. If we can't catch leaks we deliberately inject with known parameters, the entire profiling system is unvalidated marketing. This harness is the bullshit detector.

## Core Concept

Simulate realistic poker hands between an adaptive hero (`RealTimeAdaptiveEngine` adaptive mode) and leak-injected villains (noised GTO baseline + deliberate exploitable deviations in specific spots). Export hand histories, feed them through the web profiling pipeline (`HandHistoryReviewServer`), and score whether the system detects the planted leaks.

## Architecture

```
ProfilingValidationHarness
├── SpotContext                 # Rich game context for leak predicates
├── InjectedLeak               # Situation predicate + GTO deviation
├── LeakInjectedVillain         # Noised GTO baseline + leak(s)
├── HandSimulator               # Deals and resolves hands (reuses PlayingHall mechanics)
├── HandHistoryExporter         # PokerStars-format output (full + chunked)
├── ValidationRunner            # POSTs to web API, collects profiler responses
├── ConvergenceTracker          # Per-chunk leak detection tracking
└── ValidationScorecard         # Final report
```

Package: `sicfun.holdem.validation`

## 1. SpotContext

Rich game context that leak predicates evaluate against. Computed from existing primitives — wiring, not new poker logic.

```scala
final case class SpotContext(
  street: Street,
  board: Board,
  boardTexture: BoardTexture,       // wet/dry, paired, monotone, connected, draw-heavy
  potGeometry: PotGeometry,         // SPR, pot odds, bet-to-pot ratio, effective stacks
  position: Position,               // IP vs OOP
  facingAction: Option[PokerAction],
  facingSizing: Option[Double],     // bet size relative to pot
  lineRepresented: ActionLine,      // action sequence this hand (what range we rep)
  handStrengthVsBoard: HandCategory,// nuts/strong/medium/weak/air vs this board
  rangeAdvantage: RangePosition,    // capped/uncapped/polarized given line + board
)
```

Supporting types:

- `BoardTexture`: computed from board cards — flush draw possible, straight draw possible, paired, monotone, connected. Uses existing `HoldemEquity` card analysis.
- `PotGeometry`: wraps `GameState.potOdds`, `GameState.stackToPot`, adds SPR and bet-to-pot ratio.
- `HandCategory`: enum `Nuts | Strong | Medium | Weak | Air` — determined by hand evaluation against board using existing equity infrastructure.
- `RangePosition`: enum `Uncapped | Capped | Polarized` — inferred from action line. A player who just called preflop and called flop has a capped range; a player who 3-bet pre and bet flop is uncapped.
- `ActionLine`: `Vector[PokerAction]` representing the player's actions this hand so far.

## 2. InjectedLeak

```scala
trait InjectedLeak:
  def id: String
  def description: String
  def severity: Double              // 0.0-1.0, probability leak fires when spot matches
  def applies(spot: SpotContext): Boolean
  def deviate(competentAction: PokerAction, spot: SpotContext, rng: Random): PokerAction
```

### Six Initial Leaks

| Leak ID | `applies` when | `deviate` does | Why GTO says otherwise |
|---|---|---|---|
| `overfold-river-aggression` | River, facing >=0.7pot bet, wet/dynamic board, range capped, hand medium/weak | Folds | Pot odds justify a call at GTO frequency given combos in range |
| `overcall-big-bets` | Facing large bets (>=pot), hand weak/air, board favors bettor's range | Calls | GTO folds this range portion — calling bleeds EV |
| `overbluff-turn-barrel` | Turn, IP, air hand, board texture allows credible bluff line | Raises/bets when should check | GTO checks back some air to balance; overbluffing makes range too bluff-heavy |
| `passive-big-pots` | Big pot (SPR<2), strong hand, should value bet | Checks | GTO bets for value at high frequency; checking lets villain realize equity free |
| `preflop-too-loose` | Preflop, marginal hand outside GTO opening/defending range | Calls/opens | GTO folds these combos; playing them -EV |
| `preflop-too-tight` | Preflop, playable hand inside GTO range | Folds | GTO plays these profitably; folding leaves EV on table |

Three severity levels: mild (0.3), moderate (0.6), severe (0.9).

Total players: 6 leaks x 3 severities = 18 players.

The leak taxonomy is open — `InjectedLeak` is a trait, not an enum. Adding leak #7 means implementing one class.

## 3. LeakInjectedVillain

```scala
final case class LeakInjectedVillain(
  name: String,
  leaks: Vector[InjectedLeak],
  baselineNoise: Double,            // +/-2-3% random deviation from GTO everywhere
  seed: Long
)
```

Decision flow per action:

1. Compute GTO-optimal action via `RealTimeAdaptiveEngine` in GTO mode
2. Build `SpotContext` from current game state + board + hand
3. Check each `InjectedLeak.applies(spot)` — if yes, roll against `severity`
4. If leak fires -> `deviate(gtoAction, spot, rng)`
5. If no leak fires -> apply `baselineNoise` jitter (small random perturbation of raise sizings, occasional off-frequency action swaps)

This produces the "solid player who cracks in specific spots" behavior.

## 4. HandSimulator

Focused simulation loop, reuses PlayingHall's dealing, street resolution, and pot mechanics.

- Hero: `RealTimeAdaptiveEngine` in adaptive mode (the product claim)
- Villain: `LeakInjectedVillain`
- Format: heads-up (cleanest signal, one villain per session)
- Volume: 1M hands per villain, configurable
- Records every action with full `SpotContext` metadata
- Tags each action: `wasLeakFired: Boolean`, `leakId: Option[String]`
- Tracks hero net EV (validates "adaptive beats exploitable" claim)

## 5. HandHistoryExporter

Outputs PokerStars-format hand histories (extends existing export format):

- **Full file**: `validation/player_{leakId}_{severity}/full_history.txt` — all 1M hands
- **Chunked files**: `validation/player_{leakId}_{severity}/chunk_{NNN}.txt` — 1000 hands each
- **Ground truth manifest**: `validation/player_{leakId}_{severity}/ground_truth.json`:
  ```json
  {
    "leakId": "overfold-river-aggression",
    "severity": 0.6,
    "baselineNoise": 0.03,
    "totalHands": 1000000,
    "leakApplicableSpots": 78718,
    "leakFiredCount": 47231,
    "leakFireRate": 0.60
  }
  ```

## 6. ValidationRunner

Programmatic end-to-end validation against the web API:

1. Start `HandHistoryReviewServer` in-process
2. For each player, POST chunked histories sequentially to `/api/analyze-hand-history`
3. After each chunk, record from response:
   - `opponents[].archetype` (secondary metric)
   - `opponents[].hints` (primary metric — do exploit hints match injected leak?)
   - Decision analysis metrics (hero EV, mistakes)
4. Feed accumulated `(GameState, PokerAction)` observations into `PlayerCluster` for K-Means recovery check

## 7. ConvergenceTracker

Per player, tracks after each 1000-hand chunk:

- **Leak detected?** — profiler's exploit hints contain a signal matching the injected leak
- **Detection confidence** — signal strength (posterior weight, cluster distance, hint specificity)
- **False positives** — leaks the profiler claims but weren't injected
- **Hands-to-detection** — first chunk where leak is consistently detected

Answers: "how many hands to catch a 0.3-severity overfold-river leak?"

## 8. ValidationScorecard

Final report per player and aggregate:

```
=== PROFILING VALIDATION SCORECARD ===

Player: overfold_river_aggression_moderate (severity=0.6)
  Hands played:     1,000,000
  Leak fired:       47,231 / 78,718 applicable spots (60.0%)
  Hero net EV:      +4.2 bb/100  (hero profitably exploiting)

  PRIMARY: Leak Detection
    Detected:       YES at chunk 47 (47,000 hands)
    Exploit hint:   "Overfolds to large river bets on dynamic boards"
    Confidence:     0.87
    False positives: 0

  SECONDARY: Archetype Classification
    Assigned:       Nit (0.61 posterior)
    Convergence:    chunk 12 (12,000 hands)

  SECONDARY: Cluster Recovery
    Cluster ID:     2 (nearest centroid distance: 0.08)

=== AGGREGATE ===
  Players:          18
  Leaks detected:   16/18 (88.9%)
  Median hands-to-detect: 34,000
  False positive rate: 2.1%
  Hero winrate:     +3.8 bb/100 avg (adaptive beats exploitable: CONFIRMED)
```

## What Exists vs What's New

| Component | Status |
|---|---|
| GTO solver for villain baseline | EXISTS — `RealTimeAdaptiveEngine` GTO mode |
| Adaptive hero engine | EXISTS — `RealTimeAdaptiveEngine` adaptive mode |
| Hand dealing + street resolution | EXISTS — PlayingHall mechanics |
| PokerStars export | EXISTS — `review-upload-pokerstars.txt` format |
| Web API + profiling | EXISTS — `HandHistoryReviewServer` + `HandHistoryReviewService` |
| Archetype posterior | EXISTS — `ArchetypeLearning` |
| K-Means clustering | EXISTS — `PlayerCluster` |
| `SpotContext` computation | NEW — wiring existing primitives |
| `BoardTexture` / `PotGeometry` / `HandCategory` / `RangePosition` | NEW — computed from existing equity/eval |
| `InjectedLeak` trait + 6 implementations | NEW |
| `LeakInjectedVillain` | NEW |
| `HandSimulator` (focused loop) | NEW — reuses PlayingHall internals |
| `HandHistoryExporter` (chunked) | NEW — extends existing export |
| `ValidationRunner` | NEW |
| `ConvergenceTracker` | NEW |
| `ValidationScorecard` | NEW |

## Design Constraints

1. **Heads-up only** — cleanest signal for validation. Multiway muddies attribution.
2. **GTO baseline from same engine but with noise** — testing profiling, not the solver.
3. **Leak taxonomy is open** — trait, not enum. Extensible to arbitrary leak types.
4. **Ground truth travels with data** — every action tagged with leak metadata.
5. **Web pipeline tested end-to-end** — same API a real user hits.
6. **Hero EV tracked** — validates product claim alongside profiling accuracy.
7. **1M hands per player minimum** — statistical significance for leak detection.
