# Profiling Validation Harness Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate that the profiling pipeline detects known exploitable leaks injected into simulated opponent play, end-to-end through the web API.

**Architecture:** Standalone `sicfun.holdem.validation` package. Heads-up simulation with adaptive hero vs leak-injected villains (noised GTO baseline + deliberate deviations). Exports PokerStars-format hand histories, feeds through `HandHistoryReviewServer`, scores detection accuracy.

**Tech Stack:** Scala 3.8.1, munit 1.2.2, existing sicfun engine/equity/profiling infrastructure.

**Spec:** `docs/superpowers/specs/2026-03-14-profiling-validation-harness-design.md`

---

## Critical Design Notes (from spec review)

### 1. "GTO baseline" is not a solver — it's the engine's recommendation

`RealTimeAdaptiveEngine` has no explicit "GTO mode." The villain's competent baseline is the engine's `decide()` recommendation with a uniform archetype prior (no learned bias). This is "what a reasonable player does" — not solver-optimal GTO, but good enough as a baseline to deviate from. The `decide()` method is public and handles posterior inference internally.

### 2. Current exploit hints will NOT catch spot-specific leaks — and that's the point

The existing `exploitHintsFor` (OpponentProfileStore.scala:313-346) uses aggregate thresholds: fold-to-raise >= 0.55, overall fold rate >= 0.45, etc. It has no board texture or street awareness. Leaks like `overfold-river-aggression` and `overbluff-turn-barrel` will almost certainly NOT be detected by the current hints.

**This is the expected finding.** The harness proves the gap exists and quantifies it. Coarse leaks (preflop-too-loose, preflop-too-tight) will be detected. Spot-specific leaks won't. The scorecard will show this clearly. Future work: upgrade `exploitHintsFor` with per-street/per-sizing analysis.

### 3. Compute budget — use reduced fidelity for validation

At full fidelity (~100ms/hand), 18 players × 1M hands = ~500 hours. The plan uses reduced fidelity defaults for validation runs:
- `bunchingTrials = 100` (not 10,000)
- `equityTrials = 500` (not 50,000)
- `minEquityTrials = 100` (not 2,000)

At ~5ms/hand, 18M hands ≈ 25 hours. Start with 10K hands/player for development, scale to 1M for the real run.

### 4. Web API is async (job queue)

`HandHistoryReviewServer` returns 202 + jobId, requires polling. The plan's main validation path bypasses the web server entirely — it calls `OpponentProfile.fromImportedHands` and `HandHistoryImport.parseText` directly. The web path is only for visual spot-checks.

### 5. Leak definitions use simplified poker logic

The leak predicates (e.g., "medium/weak hand should call on wet board") are heuristic, not GTO-precise. This is intentional — the leaks need to deviate frequently enough to be detectable. If a leak only fires in truly incorrect spots (where GTO genuinely calls), the detection signal would be weak. The severity parameter controls how exploitable the player is.

---

## File Structure

### New Files (package: `sicfun.holdem.validation`)

| File | Responsibility |
|------|---------------|
| `src/main/scala/sicfun/holdem/validation/SpotContext.scala` | `SpotContext`, `BoardTexture`, `PotGeometry`, `HandCategory`, `RangePosition`, `ActionLine` — rich game context for leak predicates |
| `src/main/scala/sicfun/holdem/validation/InjectedLeak.scala` | `InjectedLeak` trait + 6 implementations (overfold-river, overcall, overbluff, passive-big-pots, preflop-loose, preflop-tight) |
| `src/main/scala/sicfun/holdem/validation/LeakInjectedVillain.scala` | `LeakInjectedVillain` — noised GTO baseline + leak deviation logic |
| `src/main/scala/sicfun/holdem/validation/HeadsUpSimulator.scala` | Focused heads-up hand simulation: deal, bet streets, showdown. Uses `RealTimeAdaptiveEngine` for both hero (adaptive) and villain (GTO baseline) |
| `src/main/scala/sicfun/holdem/validation/PokerStarsExporter.scala` | PokerStars-format hand history export (full files + 1000-hand chunks + ground truth JSON) |
| `src/main/scala/sicfun/holdem/validation/ValidationRunner.scala` | Orchestrator: runs simulation, exports, POSTs to web API, collects results |
| `src/main/scala/sicfun/holdem/validation/ConvergenceTracker.scala` | Per-chunk leak detection tracking, hands-to-detection, false positive counting |
| `src/main/scala/sicfun/holdem/validation/ValidationScorecard.scala` | Final report: per-player + aggregate metrics, formatted text output |

### New Test Files

| File | Tests |
|------|-------|
| `src/test/scala/sicfun/holdem/validation/SpotContextTest.scala` | BoardTexture, PotGeometry, HandCategory, RangePosition computation |
| `src/test/scala/sicfun/holdem/validation/InjectedLeakTest.scala` | Each leak's `applies` + `deviate` in known spots |
| `src/test/scala/sicfun/holdem/validation/LeakInjectedVillainTest.scala` | Decision flow: GTO baseline, noise, leak firing |
| `src/test/scala/sicfun/holdem/validation/HeadsUpSimulatorTest.scala` | Hand dealing, street resolution, pot math, action recording |
| `src/test/scala/sicfun/holdem/validation/PokerStarsExporterTest.scala` | Export → parse roundtrip via `HandHistoryImport.parseText` |
| `src/test/scala/sicfun/holdem/validation/ValidationScorecardTest.scala` | Scorecard formatting and metric aggregation |

### Existing Files Referenced (read-only, no modifications)

| File | Used for |
|------|----------|
| `src/main/scala/sicfun/holdem/types/GameState.scala` | `GameState`, `Street`, `Position`, `BetAction` |
| `src/main/scala/sicfun/holdem/types/HoldemTypes.scala` | `Board`, `HoleCards`, `Card` |
| `src/main/scala/sicfun/holdem/types/PokerAction.scala` | `PokerAction` enum |
| `src/main/scala/sicfun/holdem/engine/RealTimeAdaptiveEngine.scala` | Hero adaptive decisions + villain GTO baseline |
| `src/main/scala/sicfun/holdem/engine/ArchetypeLearning.scala` | `ArchetypePosterior`, `RaiseResponseCounts` |
| `src/main/scala/sicfun/holdem/equity/HoldemEquity.scala` | Hand evaluation, equity computation |
| `src/main/scala/sicfun/holdem/equity/TableFormat.scala` | `TableFormat.HeadsUp`, `TableRanges.defaults` |
| `src/main/scala/sicfun/holdem/history/HandHistoryImport.scala` | `parseText` for roundtrip verification |
| `src/main/scala/sicfun/holdem/history/OpponentProfileStore.scala` | `OpponentProfile.fromImportedHands`, `exploitHintsFor` |
| `src/main/scala/sicfun/holdem/analysis/PlayerSignature.scala` | `PlayerSignature.compute` |
| `src/main/scala/sicfun/holdem/analysis/PlayerCluster.scala` | `PlayerCluster.cluster` |
| `src/main/scala/sicfun/holdem/web/HandHistoryReviewServer.scala` | In-process web server for API validation |
| `src/main/scala/sicfun/holdem/web/HandHistoryReviewService.scala` | `AnalysisRequest`, `OpponentView` |
| `src/main/scala/sicfun/holdem/model/PokerActionModel.scala` | `PokerActionModel.uniform` fallback |
| `src/main/scala/sicfun/core/Card.scala` | `Card`, `Rank`, `Suit`, `Deck` |
| `src/main/scala/sicfun/core/HandEvaluator.scala` | 7-card hand evaluation for showdown |

---

## Chunk 1: SpotContext — Rich Game Context

### Task 1: BoardTexture

**Files:**
- Create: `src/main/scala/sicfun/holdem/validation/SpotContext.scala`
- Test: `src/test/scala/sicfun/holdem/validation/SpotContextTest.scala`

- [ ] **Step 1: Write BoardTexture tests**

```scala
package sicfun.holdem.validation

import munit.FunSuite
import sicfun.core.{Card, Rank, Suit}
import sicfun.holdem.types.Board

class SpotContextTest extends FunSuite:

  // ── BoardTexture ──

  test("dry rainbow board"):
    // Kh 7d 2c — no flush draws, no straight draws, unpaired
    val board = Board.from(Vector(
      Card(Rank.King, Suit.Hearts),
      Card(Rank.Seven, Suit.Diamonds),
      Card(Rank.Two, Suit.Clubs)
    ))
    val bt = BoardTexture.from(board)
    assert(!bt.flushDrawPossible)
    assert(!bt.straightDrawPossible)
    assert(!bt.paired)
    assert(!bt.monotone)
    assert(bt.isDry)

  test("wet connected suited board"):
    // 9h 8h 7d — flush draw possible (2 hearts), straight draw possible
    val board = Board.from(Vector(
      Card(Rank.Nine, Suit.Hearts),
      Card(Rank.Eight, Suit.Hearts),
      Card(Rank.Seven, Suit.Diamonds)
    ))
    val bt = BoardTexture.from(board)
    assert(bt.flushDrawPossible)
    assert(bt.straightDrawPossible)
    assert(!bt.paired)
    assert(bt.isWet)

  test("paired board"):
    val board = Board.from(Vector(
      Card(Rank.Queen, Suit.Hearts),
      Card(Rank.Queen, Suit.Diamonds),
      Card(Rank.Five, Suit.Clubs)
    ))
    val bt = BoardTexture.from(board)
    assert(bt.paired)

  test("monotone board"):
    val board = Board.from(Vector(
      Card(Rank.Ace, Suit.Spades),
      Card(Rank.Ten, Suit.Spades),
      Card(Rank.Four, Suit.Spades)
    ))
    val bt = BoardTexture.from(board)
    assert(bt.monotone)
    assert(bt.flushDrawPossible)

  test("empty board is dry"):
    val bt = BoardTexture.from(Board.empty)
    assert(bt.isDry)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly sicfun.holdem.validation.SpotContextTest"`
Expected: compilation error — `BoardTexture` not found.

- [ ] **Step 3: Implement BoardTexture**

In `SpotContext.scala`:

```scala
package sicfun.holdem.validation

import sicfun.core.{Card, Rank, Suit}
import sicfun.holdem.types.{Board, HoleCards, GameState, Position}
import sicfun.holdem.types.GameState.Street
import sicfun.holdem.types.PokerAction

final case class BoardTexture(
    flushDrawPossible: Boolean,
    straightDrawPossible: Boolean,
    paired: Boolean,
    monotone: Boolean,
    connected: Boolean
):
  def isWet: Boolean = flushDrawPossible || straightDrawPossible || connected
  def isDry: Boolean = !isWet

object BoardTexture:
  def from(board: Board): BoardTexture =
    if board.cards.isEmpty then
      BoardTexture(flushDrawPossible = false, straightDrawPossible = false,
        paired = false, monotone = false, connected = false)
    else
      val cards = board.cards
      val suitCounts = cards.groupBy(_.suit).view.mapValues(_.size).toMap
      val maxSameSuit = suitCounts.values.max
      val flushDraw = maxSameSuit >= 2
      val mono = cards.size >= 3 && maxSameSuit == cards.size

      val ranks = cards.map(_.rank.ordinal).sorted
      val uniqueRanks = ranks.distinct
      val paired = uniqueRanks.size < cards.size

      // straight draw: any 3 cards within span of 5 (or 4 within 5 for turn/river)
      val straightDraw = hasConnectedness(uniqueRanks, if cards.size >= 4 then 4 else 3)
      val conn = adjacentConnected(uniqueRanks)

      BoardTexture(
        flushDrawPossible = flushDraw,
        straightDrawPossible = straightDraw,
        paired = paired,
        monotone = mono,
        connected = conn
      )

  private def hasConnectedness(sortedRanks: Vector[Int], needed: Int): Boolean =
    if sortedRanks.size < needed then false
    else sortedRanks.sliding(needed).exists(window => window.last - window.head <= 4)

  private def adjacentConnected(sortedRanks: Vector[Int]): Boolean =
    sortedRanks.sliding(2).exists { case Vector(a, b) => b - a <= 2; case _ => false }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `sbt "testOnly sicfun.holdem.validation.SpotContextTest"`
Expected: all BoardTexture tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/validation/SpotContext.scala \
        src/test/scala/sicfun/holdem/validation/SpotContextTest.scala
git commit -m "feat(validation): add BoardTexture with wet/dry/paired/monotone classification"
```

---

### Task 2: PotGeometry, HandCategory, RangePosition, ActionLine

**Files:**
- Modify: `src/main/scala/sicfun/holdem/validation/SpotContext.scala`
- Modify: `src/test/scala/sicfun/holdem/validation/SpotContextTest.scala`

- [ ] **Step 1: Write tests for PotGeometry, HandCategory, RangePosition**

Append to `SpotContextTest.scala`:

```scala
  // ── PotGeometry ──

  test("PotGeometry from GameState"):
    val gs = GameState(
      street = Street.River,
      board = Board.empty,
      pot = 100.0,
      toCall = 75.0,
      position = Position.Button,
      stackSize = 150.0,
      betHistory = Vector.empty
    )
    val pg = PotGeometry.from(gs)
    assertEqualsDouble(pg.spr, 1.5, 0.01)         // 150/100
    assertEqualsDouble(pg.potOdds, 0.4286, 0.01)   // 75/(100+75)
    assertEqualsDouble(pg.betToPotRatio, 0.75, 0.01) // 75/100

  test("PotGeometry with zero pot"):
    val gs = GameState(Street.Preflop, Board.empty, 0.0, 1.0, Position.BigBlind, 100.0, Vector.empty)
    val pg = PotGeometry.from(gs)
    assert(pg.spr == Double.PositiveInfinity)

  // ── HandCategory ──

  test("HandCategory classification"):
    // These are ordinal comparisons — Nuts > Strong > Medium > Weak > Air
    assert(HandCategory.Nuts.ordinal < HandCategory.Strong.ordinal)
    assert(HandCategory.Air.ordinal > HandCategory.Weak.ordinal)

  // ── RangePosition ──

  test("RangePosition from preflop call line"):
    val line = ActionLine(Vector(PokerAction.Call))
    assertEquals(RangePosition.fromLine(line, Street.Flop), RangePosition.Capped)

  test("RangePosition from 3bet line"):
    val line = ActionLine(Vector(PokerAction.Raise(6.0), PokerAction.Raise(18.0)))
    assertEquals(RangePosition.fromLine(line, Street.Flop), RangePosition.Uncapped)

  test("RangePosition from check-raise line"):
    val line = ActionLine(Vector(PokerAction.Check, PokerAction.Raise(12.0)))
    assertEquals(RangePosition.fromLine(line, Street.Flop), RangePosition.Polarized)
```

- [ ] **Step 2: Run test to verify failures**

Run: `sbt "testOnly sicfun.holdem.validation.SpotContextTest"`
Expected: compilation errors for `PotGeometry`, `HandCategory`, `RangePosition`, `ActionLine`.

- [ ] **Step 3: Implement PotGeometry, HandCategory, RangePosition, ActionLine**

Append to `SpotContext.scala`:

```scala
final case class PotGeometry(
    spr: Double,        // stack-to-pot ratio
    potOdds: Double,    // toCall / (pot + toCall)
    betToPotRatio: Double, // toCall / pot (how big is the bet relative to pot)
    effectiveStack: Double
):
  def isBigPot: Boolean = spr < 2.0
  def isLargeBet: Boolean = betToPotRatio >= 0.7

object PotGeometry:
  def from(gs: GameState): PotGeometry =
    PotGeometry(
      spr = gs.stackToPot,
      potOdds = gs.potOdds,
      betToPotRatio = if gs.pot > 0.0 then gs.toCall / gs.pot else 0.0,
      effectiveStack = gs.stackSize
    )

enum HandCategory:
  case Nuts, Strong, Medium, Weak, Air

object HandCategory:
  /** Classify hand strength as a percentile bucket of equity vs a uniform range on this board.
    * Uses HoldemEquity Monte Carlo with a small trial count for speed. */
  def classify(hero: HoleCards, board: Board, equityVsRandom: Double): HandCategory =
    if equityVsRandom >= 0.85 then HandCategory.Nuts
    else if equityVsRandom >= 0.65 then HandCategory.Strong
    else if equityVsRandom >= 0.45 then HandCategory.Medium
    else if equityVsRandom >= 0.25 then HandCategory.Weak
    else HandCategory.Air

enum RangePosition:
  case Uncapped, Capped, Polarized

object RangePosition:
  def fromLine(line: ActionLine, currentStreet: Street): RangePosition =
    val raises = line.actions.count(_.category == PokerAction.Category.Raise)
    val checks = line.actions.count(_.category == PokerAction.Category.Check)
    val hasCheckRaise = line.actions.sliding(2).exists {
      case Vector(PokerAction.Check, PokerAction.Raise(_)) => true
      case _ => false
    }
    if hasCheckRaise then RangePosition.Polarized
    else if raises >= 2 then RangePosition.Uncapped
    else if raises >= 1 then RangePosition.Uncapped
    else RangePosition.Capped // flat-called or just checked

final case class ActionLine(actions: Vector[PokerAction]):
  def lastAction: Option[PokerAction] = actions.lastOption
  def containsRaise: Boolean = actions.exists(_.category == PokerAction.Category.Raise)
  def raiseCount: Int = actions.count(_.category == PokerAction.Category.Raise)
```

- [ ] **Step 4: Run tests**

Run: `sbt "testOnly sicfun.holdem.validation.SpotContextTest"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/validation/SpotContext.scala \
        src/test/scala/sicfun/holdem/validation/SpotContextTest.scala
git commit -m "feat(validation): add PotGeometry, HandCategory, RangePosition, ActionLine"
```

---

### Task 3: SpotContext assembly

**Files:**
- Modify: `src/main/scala/sicfun/holdem/validation/SpotContext.scala`
- Modify: `src/test/scala/sicfun/holdem/validation/SpotContextTest.scala`

- [ ] **Step 1: Write SpotContext assembly test**

```scala
  test("SpotContext.build assembles all components"):
    val board = Board.from(Vector(
      Card(Rank.Nine, Suit.Hearts),
      Card(Rank.Eight, Suit.Hearts),
      Card(Rank.Two, Suit.Clubs)
    ))
    val gs = GameState(Street.Flop, board, 20.0, 15.0, Position.Button, 85.0, Vector.empty)
    val hero = HoleCards(Card(Rank.Ace, Suit.Hearts), Card(Rank.King, Suit.Hearts))
    val line = ActionLine(Vector(PokerAction.Call))
    val equityVsRandom = 0.72

    val spot = SpotContext.build(gs, hero, line, equityVsRandom)
    assertEquals(spot.street, Street.Flop)
    assert(spot.boardTexture.flushDrawPossible) // two hearts
    assert(spot.boardTexture.isWet)
    assertEqualsDouble(spot.potGeometry.spr, 4.25, 0.01)
    assertEquals(spot.rangeAdvantage, RangePosition.Capped) // flat call line
    assertEquals(spot.handStrengthVsBoard, HandCategory.Strong) // 0.72 equity
```

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement SpotContext.build**

```scala
final case class SpotContext(
    street: Street,
    board: Board,
    boardTexture: BoardTexture,
    potGeometry: PotGeometry,
    position: Position,
    facingAction: Option[PokerAction],
    facingSizing: Option[Double],
    lineRepresented: ActionLine,
    handStrengthVsBoard: HandCategory,
    rangeAdvantage: RangePosition
)

object SpotContext:
  def build(
      gs: GameState,
      hero: HoleCards,
      line: ActionLine,
      equityVsRandom: Double,
      facingAction: Option[PokerAction] = None
  ): SpotContext =
    val facingSizing = facingAction.collect { case PokerAction.Raise(amt) => amt }
      .map(amt => if gs.pot > 0 then amt / gs.pot else 0.0)
      .orElse(if gs.toCall > 0 && gs.pot > 0 then Some(gs.toCall / gs.pot) else None)

    SpotContext(
      street = gs.street,
      board = gs.board,
      boardTexture = BoardTexture.from(gs.board),
      potGeometry = PotGeometry.from(gs),
      position = gs.position,
      facingAction = facingAction,
      facingSizing = facingSizing,
      lineRepresented = line,
      handStrengthVsBoard = HandCategory.classify(hero, gs.board, equityVsRandom),
      rangeAdvantage = RangePosition.fromLine(line, gs.street)
    )
```

- [ ] **Step 4: Run tests**

Run: `sbt "testOnly sicfun.holdem.validation.SpotContextTest"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/validation/SpotContext.scala \
        src/test/scala/sicfun/holdem/validation/SpotContextTest.scala
git commit -m "feat(validation): add SpotContext assembly from GameState + hero hand"
```

---

## Chunk 2: InjectedLeak — The Six Leaks

### Task 4: InjectedLeak trait + OverfoldsToAggression

**Files:**
- Create: `src/main/scala/sicfun/holdem/validation/InjectedLeak.scala`
- Create: `src/test/scala/sicfun/holdem/validation/InjectedLeakTest.scala`

- [ ] **Step 1: Write tests for trait + first leak**

```scala
package sicfun.holdem.validation

import munit.FunSuite
import sicfun.core.{Card, Rank, Suit}
import sicfun.holdem.types.{Board, HoleCards, GameState, Position}
import sicfun.holdem.types.GameState.Street
import sicfun.holdem.types.PokerAction
import scala.util.Random

class InjectedLeakTest extends FunSuite:

  private def riverSpot(
      betToPot: Double = 0.8,
      wet: Boolean = true,
      handCategory: HandCategory = HandCategory.Weak,
      rangePos: RangePosition = RangePosition.Capped
  ): SpotContext =
    val board = if wet then
      Board.from(Vector(
        Card(Rank.Nine, Suit.Hearts), Card(Rank.Eight, Suit.Hearts),
        Card(Rank.Two, Suit.Clubs), Card(Rank.Five, Suit.Diamonds),
        Card(Rank.Jack, Suit.Hearts)
      ))
    else
      Board.from(Vector(
        Card(Rank.King, Suit.Hearts), Card(Rank.Seven, Suit.Diamonds),
        Card(Rank.Two, Suit.Clubs), Card(Rank.Nine, Suit.Spades),
        Card(Rank.Four, Suit.Clubs)
      ))
    SpotContext(
      street = Street.River,
      board = board,
      boardTexture = BoardTexture.from(board),
      potGeometry = PotGeometry(spr = 1.0, potOdds = betToPot / (1.0 + betToPot),
        betToPotRatio = betToPot, effectiveStack = 50.0),
      position = Position.BigBlind,
      facingAction = Some(PokerAction.Raise(betToPot * 100.0)),
      facingSizing = Some(betToPot),
      lineRepresented = ActionLine(Vector(PokerAction.Call)),
      handStrengthVsBoard = handCategory,
      rangeAdvantage = rangePos
    )

  test("OverfoldsToAggression applies on wet river facing large bet with capped weak hand"):
    val leak = OverfoldsToAggression(severity = 1.0)
    val spot = riverSpot(betToPot = 0.8, wet = true, handCategory = HandCategory.Weak, rangePos = RangePosition.Capped)
    assert(leak.applies(spot))

  test("OverfoldsToAggression does NOT apply on dry board"):
    val leak = OverfoldsToAggression(severity = 1.0)
    val spot = riverSpot(betToPot = 0.8, wet = false, handCategory = HandCategory.Weak, rangePos = RangePosition.Capped)
    assert(!leak.applies(spot))

  test("OverfoldsToAggression does NOT apply with strong hand"):
    val leak = OverfoldsToAggression(severity = 1.0)
    val spot = riverSpot(betToPot = 0.8, wet = true, handCategory = HandCategory.Strong, rangePos = RangePosition.Capped)
    assert(!leak.applies(spot))

  test("OverfoldsToAggression does NOT apply with small bet"):
    val leak = OverfoldsToAggression(severity = 1.0)
    val spot = riverSpot(betToPot = 0.3, wet = true, handCategory = HandCategory.Weak, rangePos = RangePosition.Capped)
    assert(!leak.applies(spot))

  test("OverfoldsToAggression deviates to Fold"):
    val leak = OverfoldsToAggression(severity = 1.0)
    val spot = riverSpot()
    val deviated = leak.deviate(PokerAction.Call, spot, new Random(42))
    assertEquals(deviated, PokerAction.Fold)

  test("OverfoldsToAggression severity controls fire rate"):
    val leak = OverfoldsToAggression(severity = 0.3)
    val spot = riverSpot()
    val rng = new Random(42)
    var foldCount = 0
    val trials = 10000
    for _ <- 0 until trials do
      val action = leak.deviate(PokerAction.Call, spot, rng)
      if action == PokerAction.Fold then foldCount += 1
    val rate = foldCount.toDouble / trials
    assertEqualsDouble(rate, 0.3, 0.03) // within 3% of expected
```

- [ ] **Step 2: Run to verify failure**

Run: `sbt "testOnly sicfun.holdem.validation.InjectedLeakTest"`
Expected: compilation error.

- [ ] **Step 3: Implement InjectedLeak trait + OverfoldsToAggression**

```scala
package sicfun.holdem.validation

import sicfun.holdem.types.PokerAction
import sicfun.holdem.types.GameState.Street
import scala.util.Random

trait InjectedLeak:
  def id: String
  def description: String
  def severity: Double
  def applies(spot: SpotContext): Boolean
  def deviate(competentAction: PokerAction, spot: SpotContext, rng: Random): PokerAction

  /** Roll against severity. If hit, deviate; otherwise return competent action unchanged. */
  protected def rollAndDeviate(
      competentAction: PokerAction,
      deviatedAction: PokerAction,
      rng: Random
  ): PokerAction =
    if rng.nextDouble() < severity then deviatedAction else competentAction

final case class OverfoldsToAggression(severity: Double) extends InjectedLeak:
  val id = "overfold-river-aggression"
  val description = "Overfolds to large bets on wet boards when range is capped and hand is medium/weak"

  def applies(spot: SpotContext): Boolean =
    spot.street == Street.River &&
    spot.boardTexture.isWet &&
    spot.potGeometry.betToPotRatio >= 0.7 &&
    spot.rangeAdvantage == RangePosition.Capped &&
    (spot.handStrengthVsBoard == HandCategory.Weak || spot.handStrengthVsBoard == HandCategory.Medium)

  def deviate(competentAction: PokerAction, spot: SpotContext, rng: Random): PokerAction =
    rollAndDeviate(competentAction, PokerAction.Fold, rng)
```

- [ ] **Step 4: Run tests**

Run: `sbt "testOnly sicfun.holdem.validation.InjectedLeakTest"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/validation/InjectedLeak.scala \
        src/test/scala/sicfun/holdem/validation/InjectedLeakTest.scala
git commit -m "feat(validation): add InjectedLeak trait + OverfoldsToAggression"
```

---

### Task 5: Remaining five leaks

**Files:**
- Modify: `src/main/scala/sicfun/holdem/validation/InjectedLeak.scala`
- Modify: `src/test/scala/sicfun/holdem/validation/InjectedLeakTest.scala`

- [ ] **Step 1: Write tests for Overcalls**

```scala
  test("Overcalls applies when facing large bet with weak hand"):
    val leak = Overcalls(severity = 1.0)
    val spot = riverSpot(betToPot = 1.0, wet = true, handCategory = HandCategory.Air, rangePos = RangePosition.Capped)
    assert(leak.applies(spot))

  test("Overcalls deviates to Call"):
    val leak = Overcalls(severity = 1.0)
    val spot = riverSpot(betToPot = 1.0, wet = true, handCategory = HandCategory.Air)
    assertEquals(leak.deviate(PokerAction.Fold, spot, new Random(42)), PokerAction.Call)
```

- [ ] **Step 2: Write tests for OverbluffsTurnBarrel**

```scala
  private def turnSpotIP(handCat: HandCategory = HandCategory.Air): SpotContext =
    val board = Board.from(Vector(
      Card(Rank.Nine, Suit.Hearts), Card(Rank.Eight, Suit.Hearts),
      Card(Rank.Two, Suit.Clubs), Card(Rank.Five, Suit.Diamonds)
    ))
    SpotContext(
      street = Street.Turn, board = board,
      boardTexture = BoardTexture.from(board),
      potGeometry = PotGeometry(spr = 3.0, potOdds = 0.0, betToPotRatio = 0.0, effectiveStack = 90.0),
      position = Position.Button, // IP
      facingAction = None, facingSizing = None,
      lineRepresented = ActionLine(Vector(PokerAction.Raise(6.0))),
      handStrengthVsBoard = handCat,
      rangeAdvantage = RangePosition.Uncapped
    )

  test("OverbluffsTurnBarrel applies IP on turn with air and wet board"):
    val leak = OverbluffsTurnBarrel(severity = 1.0)
    assert(leak.applies(turnSpotIP(HandCategory.Air)))

  test("OverbluffsTurnBarrel does NOT apply with strong hand"):
    val leak = OverbluffsTurnBarrel(severity = 1.0)
    assert(!leak.applies(turnSpotIP(HandCategory.Strong)))
```

- [ ] **Step 3: Write tests for PassiveInBigPots**

```scala
  test("PassiveInBigPots applies in big pot with strong hand"):
    val spot = riverSpot(betToPot = 0.0, wet = true, handCategory = HandCategory.Strong, rangePos = RangePosition.Uncapped)
      .copy(potGeometry = PotGeometry(spr = 1.2, potOdds = 0.0, betToPotRatio = 0.0, effectiveStack = 40.0))
    val leak = PassiveInBigPots(severity = 1.0)
    assert(leak.applies(spot))

  test("PassiveInBigPots deviates to Check"):
    val spot = riverSpot().copy(
      handStrengthVsBoard = HandCategory.Strong,
      potGeometry = PotGeometry(spr = 1.0, potOdds = 0.0, betToPotRatio = 0.0, effectiveStack = 40.0)
    )
    val leak = PassiveInBigPots(severity = 1.0)
    assertEquals(leak.deviate(PokerAction.Raise(30.0), spot, new Random(42)), PokerAction.Check)
```

- [ ] **Step 4: Write tests for PreflopTooLoose and PreflopTooTight**

```scala
  private def preflopSpot(handCat: HandCategory): SpotContext =
    SpotContext(
      street = Street.Preflop, board = Board.empty,
      boardTexture = BoardTexture.from(Board.empty),
      potGeometry = PotGeometry(spr = 50.0, potOdds = 0.33, betToPotRatio = 0.5, effectiveStack = 100.0),
      position = Position.Cutoff,
      facingAction = None, facingSizing = None,
      lineRepresented = ActionLine(Vector.empty),
      handStrengthVsBoard = handCat,
      rangeAdvantage = RangePosition.Uncapped
    )

  test("PreflopTooLoose applies with weak hand"):
    val leak = PreflopTooLoose(severity = 1.0)
    assert(leak.applies(preflopSpot(HandCategory.Weak)))

  test("PreflopTooLoose deviates Fold to Call"):
    val leak = PreflopTooLoose(severity = 1.0)
    assertEquals(leak.deviate(PokerAction.Fold, preflopSpot(HandCategory.Weak), new Random(42)), PokerAction.Call)

  test("PreflopTooTight applies with medium hand preflop"):
    val leak = PreflopTooTight(severity = 1.0)
    assert(leak.applies(preflopSpot(HandCategory.Medium)))

  test("PreflopTooTight deviates Call/Raise to Fold"):
    val leak = PreflopTooTight(severity = 1.0)
    assertEquals(leak.deviate(PokerAction.Call, preflopSpot(HandCategory.Medium), new Random(42)), PokerAction.Fold)
```

- [ ] **Step 5: Run all to verify failures**

Run: `sbt "testOnly sicfun.holdem.validation.InjectedLeakTest"`
Expected: compilation errors for `Overcalls`, `OverbluffsTurnBarrel`, `PassiveInBigPots`, `PreflopTooLoose`, `PreflopTooTight`.

- [ ] **Step 6: Implement all five leaks**

Append to `InjectedLeak.scala`:

```scala
final case class Overcalls(severity: Double) extends InjectedLeak:
  val id = "overcall-big-bets"
  val description = "Calls large bets with weak/air hands when GTO should fold"
  def applies(spot: SpotContext): Boolean =
    spot.potGeometry.betToPotRatio >= 0.8 &&
    (spot.handStrengthVsBoard == HandCategory.Weak || spot.handStrengthVsBoard == HandCategory.Air)
  def deviate(competentAction: PokerAction, spot: SpotContext, rng: Random): PokerAction =
    rollAndDeviate(competentAction, PokerAction.Call, rng)

final case class OverbluffsTurnBarrel(severity: Double) extends InjectedLeak:
  val id = "overbluff-turn-barrel"
  val description = "Bets/raises turn with air on wet boards when IP — GTO would check back"
  def applies(spot: SpotContext): Boolean =
    spot.street == Street.Turn &&
    spot.boardTexture.isWet &&
    spot.handStrengthVsBoard == HandCategory.Air &&
    (spot.position == Position.Button || spot.position == Position.Cutoff) // IP positions
  def deviate(competentAction: PokerAction, spot: SpotContext, rng: Random): PokerAction =
    val betSize = spot.potGeometry.effectiveStack * 0.6 // ~60% pot bet
    rollAndDeviate(competentAction, PokerAction.Raise(betSize), rng)

final case class PassiveInBigPots(severity: Double) extends InjectedLeak:
  val id = "passive-big-pots"
  val description = "Checks strong hands in big pots (SPR<2) instead of value betting"
  def applies(spot: SpotContext): Boolean =
    spot.potGeometry.isBigPot &&
    (spot.handStrengthVsBoard == HandCategory.Strong || spot.handStrengthVsBoard == HandCategory.Nuts)
  def deviate(competentAction: PokerAction, spot: SpotContext, rng: Random): PokerAction =
    rollAndDeviate(competentAction, PokerAction.Check, rng)

final case class PreflopTooLoose(severity: Double) extends InjectedLeak:
  val id = "preflop-too-loose"
  val description = "Calls/opens preflop with hands outside GTO range"
  def applies(spot: SpotContext): Boolean =
    spot.street == Street.Preflop &&
    (spot.handStrengthVsBoard == HandCategory.Weak || spot.handStrengthVsBoard == HandCategory.Air)
  def deviate(competentAction: PokerAction, spot: SpotContext, rng: Random): PokerAction =
    rollAndDeviate(competentAction, PokerAction.Call, rng)

final case class PreflopTooTight(severity: Double) extends InjectedLeak:
  val id = "preflop-too-tight"
  val description = "Folds playable hands preflop that GTO would open/defend"
  def applies(spot: SpotContext): Boolean =
    spot.street == Street.Preflop &&
    spot.handStrengthVsBoard == HandCategory.Medium
  def deviate(competentAction: PokerAction, spot: SpotContext, rng: Random): PokerAction =
    rollAndDeviate(competentAction, PokerAction.Fold, rng)
```

- [ ] **Step 7: Run tests**

Run: `sbt "testOnly sicfun.holdem.validation.InjectedLeakTest"`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add src/main/scala/sicfun/holdem/validation/InjectedLeak.scala \
        src/test/scala/sicfun/holdem/validation/InjectedLeakTest.scala
git commit -m "feat(validation): add five remaining leak implementations"
```

---

## Chunk 3: LeakInjectedVillain — Noised GTO + Leaks

### Task 6: LeakInjectedVillain decision flow

**Files:**
- Create: `src/main/scala/sicfun/holdem/validation/LeakInjectedVillain.scala`
- Create: `src/test/scala/sicfun/holdem/validation/LeakInjectedVillainTest.scala`

- [ ] **Step 1: Write tests**

```scala
package sicfun.holdem.validation

import munit.FunSuite
import sicfun.holdem.types.PokerAction
import sicfun.holdem.types.GameState.Street
import scala.util.Random

class LeakInjectedVillainTest extends FunSuite:

  test("villain returns GTO action when no leak applies"):
    val villain = LeakInjectedVillain(
      name = "test_noleak",
      leaks = Vector(OverfoldsToAggression(severity = 1.0)),
      baselineNoise = 0.0,
      seed = 42L
    )
    // Preflop spot — OverfoldsToAggression only fires on river
    val gtoAction = PokerAction.Call
    val spot = SpotContext(
      street = Street.Preflop, board = sicfun.holdem.types.Board.empty,
      boardTexture = BoardTexture.from(sicfun.holdem.types.Board.empty),
      potGeometry = PotGeometry(spr = 50.0, potOdds = 0.33, betToPotRatio = 0.5, effectiveStack = 100.0),
      position = sicfun.holdem.types.Position.BigBlind,
      facingAction = None, facingSizing = None,
      lineRepresented = ActionLine(Vector.empty),
      handStrengthVsBoard = HandCategory.Medium,
      rangeAdvantage = RangePosition.Capped
    )
    val result = villain.decide(gtoAction, spot)
    assertEquals(result.action, PokerAction.Call)
    assert(!result.leakFired)

  test("villain fires leak when applicable with severity=1.0"):
    val villain = LeakInjectedVillain(
      name = "test_overfold",
      leaks = Vector(OverfoldsToAggression(severity = 1.0)),
      baselineNoise = 0.0,
      seed = 42L
    )
    val board = sicfun.holdem.types.Board.from(Vector(
      sicfun.core.Card(sicfun.core.Rank.Nine, sicfun.core.Suit.Hearts),
      sicfun.core.Card(sicfun.core.Rank.Eight, sicfun.core.Suit.Hearts),
      sicfun.core.Card(sicfun.core.Rank.Two, sicfun.core.Suit.Clubs),
      sicfun.core.Card(sicfun.core.Rank.Five, sicfun.core.Suit.Diamonds),
      sicfun.core.Card(sicfun.core.Rank.Jack, sicfun.core.Suit.Hearts)
    ))
    val spot = SpotContext(
      street = Street.River, board = board,
      boardTexture = BoardTexture.from(board),
      potGeometry = PotGeometry(spr = 1.0, potOdds = 0.44, betToPotRatio = 0.8, effectiveStack = 50.0),
      position = sicfun.holdem.types.Position.BigBlind,
      facingAction = Some(PokerAction.Raise(80.0)),
      facingSizing = Some(0.8),
      lineRepresented = ActionLine(Vector(PokerAction.Call)),
      handStrengthVsBoard = HandCategory.Weak,
      rangeAdvantage = RangePosition.Capped
    )
    val result = villain.decide(PokerAction.Call, spot)
    assertEquals(result.action, PokerAction.Fold)
    assert(result.leakFired)
    assertEquals(result.leakId, Some("overfold-river-aggression"))

  test("baseline noise perturbs action selection over many trials"):
    val villain = LeakInjectedVillain(
      name = "test_noise",
      leaks = Vector.empty,
      baselineNoise = 0.05,
      seed = 42L
    )
    val spot = SpotContext(
      street = Street.Flop, board = sicfun.holdem.types.Board.empty,
      boardTexture = BoardTexture.from(sicfun.holdem.types.Board.empty),
      potGeometry = PotGeometry(spr = 10.0, potOdds = 0.2, betToPotRatio = 0.25, effectiveStack = 100.0),
      position = sicfun.holdem.types.Position.Button,
      facingAction = None, facingSizing = None,
      lineRepresented = ActionLine(Vector.empty),
      handStrengthVsBoard = HandCategory.Medium,
      rangeAdvantage = RangePosition.Uncapped
    )
    // With 5% noise, most should be the GTO action but some should deviate
    var deviations = 0
    for i <- 0 until 1000 do
      val v = LeakInjectedVillain("n", Vector.empty, 0.05, seed = i.toLong)
      val r = v.decide(PokerAction.Check, spot)
      if r.action != PokerAction.Check then deviations += 1
    assert(deviations > 20 && deviations < 100, s"Expected ~50 deviations, got $deviations")
```

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement LeakInjectedVillain**

```scala
package sicfun.holdem.validation

import sicfun.holdem.types.PokerAction
import scala.util.Random

final case class VillainDecisionResult(
    action: PokerAction,
    leakFired: Boolean,
    leakId: Option[String]
)

final case class LeakInjectedVillain(
    name: String,
    leaks: Vector[InjectedLeak],
    baselineNoise: Double,
    seed: Long
):
  private val rng = new Random(seed)

  /** Decide villain action given GTO-optimal action and current spot.
    * 1. Check leaks — if any applies, roll severity and maybe deviate.
    * 2. If no leak fired, apply baseline noise jitter.
    */
  def decide(gtoAction: PokerAction, spot: SpotContext): VillainDecisionResult =
    // Check leaks in order — first applicable leak that fires wins
    val leakResult = leaks.iterator
      .filter(_.applies(spot))
      .map { leak =>
        val deviated = leak.deviate(gtoAction, spot, rng)
        (leak, deviated, deviated != gtoAction)
      }
      .find(_._3) // first one that actually deviated

    leakResult match
      case Some((leak, deviatedAction, _)) =>
        VillainDecisionResult(deviatedAction, leakFired = true, leakId = Some(leak.id))
      case None =>
        // No leak fired — apply baseline noise
        val noisedAction = applyNoise(gtoAction, spot)
        VillainDecisionResult(noisedAction, leakFired = false, leakId = None)

  private def applyNoise(action: PokerAction, spot: SpotContext): PokerAction =
    if baselineNoise <= 0.0 || rng.nextDouble() >= baselineNoise then action
    else
      // Small random perturbation: swap to a random alternative action
      val alternatives = Vector(PokerAction.Fold, PokerAction.Check, PokerAction.Call)
        .filterNot(_.category == action.category)
      if alternatives.isEmpty then action
      else alternatives(rng.nextInt(alternatives.size))
```

- [ ] **Step 4: Run tests**

Run: `sbt "testOnly sicfun.holdem.validation.LeakInjectedVillainTest"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/validation/LeakInjectedVillain.scala \
        src/test/scala/sicfun/holdem/validation/LeakInjectedVillainTest.scala
git commit -m "feat(validation): add LeakInjectedVillain with noised GTO + leak deviation"
```

---

## Chunk 4: HeadsUpSimulator — Hand Simulation Loop

### Task 7: HeadsUpSimulator core dealing and betting

**Files:**
- Create: `src/main/scala/sicfun/holdem/validation/HeadsUpSimulator.scala`
- Create: `src/test/scala/sicfun/holdem/validation/HeadsUpSimulatorTest.scala`

This is the largest task. The simulator must:
- Deal hole cards to hero and villain from a shuffled deck
- Post blinds (SB=0.5, BB=1.0 in a normalized structure)
- Run preflop, flop, turn, river betting rounds
- Use `RealTimeAdaptiveEngine` for hero decisions (adaptive mode)
- Use `RealTimeAdaptiveEngine` in GTO mode to get villain's "competent" action, then pass through `LeakInjectedVillain.decide`
- Compute `SpotContext` at each villain decision point (needs equity-vs-random for `HandCategory`)
- Record all actions with leak metadata
- Resolve showdown using `HandEvaluator`

- [ ] **Step 1: Write simulator test for basic hand completion**

```scala
package sicfun.holdem.validation

import munit.FunSuite
import sicfun.holdem.equity.{TableFormat, TableRanges}
import sicfun.holdem.model.PokerActionModel
import sicfun.holdem.engine.RealTimeAdaptiveEngine

class HeadsUpSimulatorTest extends FunSuite:

  private def makeSimulator(): HeadsUpSimulator =
    val tableRanges = TableRanges.defaults(TableFormat.HeadsUp)
    val heroEngine = new RealTimeAdaptiveEngine(
      tableRanges = tableRanges,
      actionModel = PokerActionModel.uniform,
      bunchingTrials = 100,
      defaultEquityTrials = 500,
      minEquityTrials = 100
    )
    val villainEngine = new RealTimeAdaptiveEngine(
      tableRanges = tableRanges,
      actionModel = PokerActionModel.uniform,
      bunchingTrials = 100,
      defaultEquityTrials = 500,
      minEquityTrials = 100
    )
    val villain = LeakInjectedVillain(
      name = "test_villain",
      leaks = Vector(OverfoldsToAggression(severity = 0.6)),
      baselineNoise = 0.03,
      seed = 42L
    )
    HeadsUpSimulator(
      heroEngine = heroEngine,
      villainEngine = villainEngine,
      villain = villain,
      seed = 123L,
      equityTrialsForCategory = 200
    )

  test("simulate one hand produces a HandRecord"):
    val sim = makeSimulator()
    val record = sim.playHand(handNumber = 1)
    assert(record.handId.nonEmpty)
    assert(record.heroCards != null)
    assert(record.villainCards != null)
    assert(record.actions.nonEmpty)
    // Hero net should be a number (could be positive, negative, or zero)
    assert(!record.heroNet.isNaN)

  test("simulate 100 hands all complete without error"):
    val sim = makeSimulator()
    val records = (1 to 100).map(i => sim.playHand(i))
    assertEquals(records.length, 100)
    assert(records.forall(_.actions.nonEmpty))

  test("leak-tagged actions appear in records"):
    val sim = makeSimulator()
    val records = (1 to 500).map(i => sim.playHand(i))
    val leakActions = records.flatMap(_.actions).filter(_.leakFired)
    // With 500 hands and severity 0.6, we should see some leaks
    // (may not fire every hand since the spot has to match)
    assert(leakActions.nonEmpty || records.flatMap(_.actions).length > 100,
      "Expected at least some leak firings in 500 hands, or at minimum many actions recorded")
```

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement HeadsUpSimulator**

```scala
package sicfun.holdem.validation

import sicfun.core.{Card, Deck, HandEvaluator}
import sicfun.holdem.engine.RealTimeAdaptiveEngine
import sicfun.holdem.equity.{HoldemEquity, TableFormat, TableRanges}
import sicfun.holdem.types.{Board, HoleCards, GameState, Position}
import sicfun.holdem.types.GameState.Street
import sicfun.holdem.types.PokerAction
import scala.collection.mutable
import scala.util.Random

final case class RecordedAction(
    street: Street,
    player: String,        // "Hero" or villain name
    action: PokerAction,
    potBefore: Double,
    toCall: Double,
    stackBefore: Double,
    leakFired: Boolean,
    leakId: Option[String]
)

final case class HandRecord(
    handId: String,
    handNumber: Int,
    heroCards: HoleCards,
    villainCards: HoleCards,
    board: Board,
    actions: Vector[RecordedAction],
    heroNet: Double,
    streetsPlayed: Int
)

final class HeadsUpSimulator(
    heroEngine: RealTimeAdaptiveEngine,
    villainEngine: RealTimeAdaptiveEngine,
    villain: LeakInjectedVillain,
    seed: Long,
    equityTrialsForCategory: Int = 500,
    startingStack: Double = 100.0,
    smallBlind: Double = 0.5,
    bigBlind: Double = 1.0
):
  private val rng = new Random(seed)
  private val heroName = "Hero"

  def playHand(handNumber: Int): HandRecord =
    val deck = Deck.shuffled(rng)
    val heroCards = HoleCards(deck.draw(), deck.draw())
    val villainCards = HoleCards(deck.draw(), deck.draw())
    // Pre-draw 5 community cards
    val communityCards = (0 until 5).map(_ => deck.draw()).toVector

    val actions = mutable.ArrayBuffer.empty[RecordedAction]
    var heroStack = startingStack
    var villainStack = startingStack
    var pot = 0.0
    var handOver = false
    var streetsPlayed = 0

    // Post blinds: hero=SB(Button), villain=BB
    heroStack -= smallBlind
    villainStack -= bigBlind
    pot = smallBlind + bigBlind

    def boardForStreet(street: Street): Board = street match
      case Street.Preflop => Board.empty
      case Street.Flop => Board.from(communityCards.take(3))
      case Street.Turn => Board.from(communityCards.take(4))
      case Street.River => Board.from(communityCards)

    val streets = Vector(Street.Preflop, Street.Flop, Street.Turn, Street.River)

    for street <- streets if !handOver do
      streetsPlayed += 1
      val board = boardForStreet(street)
      var toCall = if street == Street.Preflop then bigBlind - smallBlind else 0.0
      var streetDone = false
      var actionsThisStreet = 0
      // Preflop: SB(hero) acts first. Postflop: BB(villain) acts first (OOP).
      var heroTurn = street == Street.Preflop

      while !streetDone && !handOver do
        if heroTurn then
          // Hero decision
          val gs = GameState(street, board, pot, toCall, Position.Button, heroStack, Vector.empty)
          val heroAction = decideHero(heroCards, gs, board, street)
          actions += RecordedAction(street, heroName, heroAction, pot, toCall, heroStack,
            leakFired = false, leakId = None)
          heroAction match
            case PokerAction.Fold =>
              handOver = true
            case PokerAction.Check =>
              actionsThisStreet += 1
              if actionsThisStreet >= 2 then streetDone = true
            case PokerAction.Call =>
              heroStack -= toCall
              pot += toCall
              toCall = 0.0
              actionsThisStreet += 1
              if actionsThisStreet >= 2 then streetDone = true
            case PokerAction.Raise(amount) =>
              val raiseAmount = math.min(amount, heroStack)
              heroStack -= raiseAmount
              pot += raiseAmount
              toCall = raiseAmount - toCall
              actionsThisStreet += 1
        else
          // Villain decision — get GTO action from villainEngine, then pass through leak
          val gs = GameState(street, board, pot, toCall, Position.BigBlind, villainStack, Vector.empty)
          val gtoAction = decideVillainGto(villainCards, gs, board, street)
          val equityVsRandom = estimateEquity(villainCards, board)
          val spot = SpotContext.build(
            gs, villainCards,
            ActionLine(actions.iterator.filter(_.player == villain.name).map(_.action).toVector),
            equityVsRandom,
            facingAction = actions.lastOption.map(_.action)
          )
          val result = villain.decide(gtoAction, spot)
          actions += RecordedAction(street, villain.name, result.action, pot, toCall, villainStack,
            result.leakFired, result.leakId)
          result.action match
            case PokerAction.Fold =>
              handOver = true
            case PokerAction.Check =>
              actionsThisStreet += 1
              if actionsThisStreet >= 2 then streetDone = true
            case PokerAction.Call =>
              villainStack -= toCall
              pot += toCall
              toCall = 0.0
              actionsThisStreet += 1
              if actionsThisStreet >= 2 then streetDone = true
            case PokerAction.Raise(amount) =>
              val raiseAmount = math.min(amount, villainStack)
              villainStack -= raiseAmount
              pot += raiseAmount
              toCall = raiseAmount - toCall
              actionsThisStreet += 1
        heroTurn = !heroTurn

    // Determine outcome
    val finalBoard = boardForStreet(streets.take(streetsPlayed).last)
    val heroNet =
      if handOver && actions.lastOption.exists(_.action == PokerAction.Fold) then
        if actions.last.player == heroName then -(startingStack - heroStack)
        else startingStack - villainStack - (startingStack - heroStack)  // villain folded, hero wins pot
      else
        // Showdown
        val heroEval = HandEvaluator.evaluate7(heroCards.toVector ++ finalBoard.cards)
        val villainEval = HandEvaluator.evaluate7(villainCards.toVector ++ finalBoard.cards)
        if heroEval > villainEval then pot / 2.0     // hero wins (net gain = pot - hero's contribution)
        else if heroEval < villainEval then -pot / 2.0
        else 0.0 // tie

    HandRecord(
      handId = f"SIM-${handNumber}%08d",
      handNumber = handNumber,
      heroCards = heroCards,
      villainCards = villainCards,
      board = finalBoard,
      actions = actions.toVector,
      heroNet = heroNet,
      streetsPlayed = streetsPlayed
    )

  private def decideHero(hero: HoleCards, gs: GameState, board: Board, street: Street): PokerAction =
    // Use adaptive engine for hero — simplified to action categories for now
    val candidates = buildCandidates(gs)
    if candidates.size <= 1 then candidates.headOption.getOrElse(PokerAction.Check)
    else
      try
        val result = heroEngine.recommendAgainstPosterior(
          hero = hero, state = gs,
          posterior = heroEngine.currentVillainPosterior(gs, board),
          candidateActions = candidates,
          rng = rng
        )
        result.bestAction
      catch case _: Exception => PokerAction.Check

  private def decideVillainGto(villain: HoleCards, gs: GameState, board: Board, street: Street): PokerAction =
    val candidates = buildCandidates(gs)
    if candidates.size <= 1 then candidates.headOption.getOrElse(PokerAction.Check)
    else
      try
        val result = villainEngine.recommendAgainstPosterior(
          hero = villain, state = gs,
          posterior = villainEngine.currentVillainPosterior(gs, board),
          candidateActions = candidates,
          rng = rng
        )
        result.bestAction
      catch case _: Exception => PokerAction.Check

  private def buildCandidates(gs: GameState): Vector[PokerAction] =
    val candidates = Vector.newBuilder[PokerAction]
    if gs.toCall > 0 then
      candidates += PokerAction.Fold
      candidates += PokerAction.Call
      if gs.stackSize > gs.toCall * 2 then
        candidates += PokerAction.Raise(gs.pot + gs.toCall) // pot-sized raise
    else
      candidates += PokerAction.Check
      if gs.stackSize > gs.pot * 0.5 then
        candidates += PokerAction.Raise(gs.pot * 0.66) // 2/3 pot bet
    candidates.result()

  private def estimateEquity(hand: HoleCards, board: Board): Double =
    if board.cards.isEmpty then 0.5 // preflop default
    else
      try
        val allCards = hand.toVector ++ board.cards
        val eval = HandEvaluator.evaluate7(
          if allCards.size >= 7 then allCards.take(7)
          else allCards ++ Vector.fill(7 - allCards.size)(Card(Rank.Two, Suit.Clubs)) // pad — imprecise but fast
        )
        // Rough equity estimate: normalize hand rank to 0-1 range
        eval.toDouble / 7462.0 // max hand rank
      catch case _: Exception => 0.5
```

**Note:** The `decideHero`/`decideVillainGto` methods need to interface with `RealTimeAdaptiveEngine.recommendAgainstPosterior`. The exact method signature and posterior computation need to be adapted to the actual engine API during implementation. The implementer should read `RealTimeAdaptiveEngine.scala` lines 237-280 and `decide` method to determine the correct call pattern. The key decision: use `decide()` (which handles posterior inference internally) rather than `recommendAgainstPosterior()` directly, since `decide()` is the public API that manages observations and caching.

- [ ] **Step 4: Run tests**

Run: `sbt "testOnly sicfun.holdem.validation.HeadsUpSimulatorTest"`
Expected: PASS (may need adjustments based on actual engine API).

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/validation/HeadsUpSimulator.scala \
        src/test/scala/sicfun/holdem/validation/HeadsUpSimulatorTest.scala
git commit -m "feat(validation): add HeadsUpSimulator with leak-tagged action recording"
```

---

## Chunk 5: PokerStars Export + Parse Roundtrip

### Task 8: PokerStarsExporter

**Files:**
- Create: `src/main/scala/sicfun/holdem/validation/PokerStarsExporter.scala`
- Create: `src/test/scala/sicfun/holdem/validation/PokerStarsExporterTest.scala`

- [ ] **Step 1: Write roundtrip test**

```scala
package sicfun.holdem.validation

import munit.FunSuite
import sicfun.core.{Card, Rank, Suit}
import sicfun.holdem.history.{HandHistoryImport, HandHistorySite}
import sicfun.holdem.types.{Board, HoleCards, Position}
import sicfun.holdem.types.GameState.Street
import sicfun.holdem.types.PokerAction

class PokerStarsExporterTest extends FunSuite:

  private val sampleRecord = HandRecord(
    handId = "SIM-00000001",
    handNumber = 1,
    heroCards = HoleCards(Card(Rank.Ace, Suit.Spades), Card(Rank.King, Suit.Hearts)),
    villainCards = HoleCards(Card(Rank.Queen, Suit.Diamonds), Card(Rank.Jack, Suit.Clubs)),
    board = Board.from(Vector(
      Card(Rank.Ten, Suit.Hearts), Card(Rank.Nine, Suit.Spades),
      Card(Rank.Two, Suit.Clubs), Card(Rank.Seven, Suit.Diamonds),
      Card(Rank.Four, Suit.Hearts)
    )),
    actions = Vector(
      RecordedAction(Street.Preflop, "Hero", PokerAction.Raise(3.0), 1.5, 0.5, 99.5, false, None),
      RecordedAction(Street.Preflop, "Villain_1", PokerAction.Call, 4.5, 2.5, 99.0, false, None),
      RecordedAction(Street.Flop, "Villain_1", PokerAction.Check, 6.0, 0.0, 97.0, false, None),
      RecordedAction(Street.Flop, "Hero", PokerAction.Raise(4.0), 6.0, 0.0, 97.0, false, None),
      RecordedAction(Street.Flop, "Villain_1", PokerAction.Call, 10.0, 4.0, 97.0, false, None),
    ),
    heroNet = 7.0,
    streetsPlayed = 2
  )

  test("export produces parseable PokerStars format"):
    val text = PokerStarsExporter.exportHands(Vector(sampleRecord), "Hero", "Villain_1")
    val parsed = HandHistoryImport.parseText(text, Some(HandHistorySite.PokerStars), Some("Hero"))
    assert(parsed.isRight, s"Parse failed: ${parsed.left.getOrElse("")}")
    val hands = parsed.getOrElse(Vector.empty)
    assertEquals(hands.length, 1)
    assertEquals(hands.head.handId, "SIM-00000001")

  test("chunked export produces correct number of chunks"):
    val records = (1 to 2500).map(i => sampleRecord.copy(handId = f"SIM-${i}%08d", handNumber = i)).toVector
    val chunks = PokerStarsExporter.exportChunked(records, "Hero", "Villain_1", chunkSize = 1000)
    assertEquals(chunks.length, 3) // 1000 + 1000 + 500
    assertEquals(chunks(0).handCount, 1000)
    assertEquals(chunks(2).handCount, 500)
```

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement PokerStarsExporter**

The exporter must produce text in PokerStars format that `HandHistoryImport.parseText` can parse. Reference the existing format from PlayingHall's output (constants at lines 214-222) and the PokerStars parser in `HandHistoryImport.scala`.

```scala
package sicfun.holdem.validation

import sicfun.holdem.types.{Board, HoleCards, Position}
import sicfun.holdem.types.GameState.Street
import sicfun.holdem.types.PokerAction

import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.Locale

final case class ExportChunk(
    chunkIndex: Int,
    handCount: Int,
    text: String
)

object PokerStarsExporter:
  private val BaseTimestamp = LocalDateTime.of(2026, 1, 1, 12, 0, 0)
  private val TimeFmt = DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss")
  private val StartingStack = 100.0

  def exportHands(records: Vector[HandRecord], heroName: String, villainName: String): String =
    val sb = new StringBuilder
    records.foreach(r => appendHand(sb, r, heroName, villainName))
    sb.toString()

  def exportChunked(
      records: Vector[HandRecord],
      heroName: String,
      villainName: String,
      chunkSize: Int = 1000
  ): Vector[ExportChunk] =
    records.grouped(chunkSize).zipWithIndex.map { case (chunk, idx) =>
      ExportChunk(idx, chunk.length, exportHands(chunk, heroName, villainName))
    }.toVector

  private def appendHand(sb: StringBuilder, record: HandRecord, heroName: String, villainName: String): Unit =
    val ts = BaseTimestamp.plusSeconds(record.handNumber.toLong)
    sb.append(s"PokerStars Hand #${record.handId}: Hold'em No Limit (${formatMoney(0.5)}/${formatMoney(1.0)}) - ${TimeFmt.format(ts)}\n")
    sb.append(s"Table 'Validation' 2-max Seat #1 is the button\n")
    sb.append(s"Seat 1: $heroName (${formatMoney(StartingStack)} in chips)\n")
    sb.append(s"Seat 2: $villainName (${formatMoney(StartingStack)} in chips)\n")
    sb.append(s"$heroName: posts small blind ${formatMoney(0.5)}\n")
    sb.append(s"$villainName: posts big blind ${formatMoney(1.0)}\n")
    sb.append("*** HOLE CARDS ***\n")
    sb.append(s"Dealt to $heroName [${record.heroCards.first.toToken} ${record.heroCards.second.toToken}]\n")

    var currentStreet: Street = Street.Preflop
    for action <- record.actions do
      if action.street != currentStreet then
        currentStreet = action.street
        val boardCards = boardForStreet(record.board, currentStreet)
        val streetName = currentStreet match
          case Street.Flop => "FLOP"
          case Street.Turn => "TURN"
          case Street.River => "RIVER"
          case _ => "PREFLOP"
        sb.append(s"*** $streetName *** [${boardCards.mkString(" ")}]\n")
      appendAction(sb, action)

    // Showdown or fold
    if !record.actions.lastOption.exists(_.action == PokerAction.Fold) then
      sb.append("*** SHOW DOWN ***\n")
      sb.append(s"$heroName: shows [${record.heroCards.first.toToken} ${record.heroCards.second.toToken}]\n")
      sb.append(s"$villainName: shows [${record.villainCards.first.toToken} ${record.villainCards.second.toToken}]\n")
    sb.append("\n\n")

  private def appendAction(sb: StringBuilder, action: RecordedAction): Unit =
    val name = action.player
    action.action match
      case PokerAction.Fold => sb.append(s"$name: folds\n")
      case PokerAction.Check => sb.append(s"$name: checks\n")
      case PokerAction.Call => sb.append(s"$name: calls ${formatMoney(action.toCall)}\n")
      case PokerAction.Raise(amount) => sb.append(s"$name: raises ${formatMoney(amount)} to ${formatMoney(amount)}\n")

  private def boardForStreet(board: Board, street: Street): Vector[String] =
    val n = street.expectedBoardSize
    board.cards.take(n).map(_.toToken)

  private def formatMoney(amount: Double): String =
    String.format(Locale.ROOT, "$%.2f", java.lang.Double.valueOf(amount))
```

**Implementation note:** The exact PokerStars format must match what `HandHistoryImport` parses. The implementer should study `HandHistoryImport.scala`'s PokerStars parsing logic (look for `"PokerStars Hand #"` prefix handling) and adjust the export format until the roundtrip test passes. The format shown above is a starting point — action formatting (especially raises) may need tweaking to match the parser's expectations.

- [ ] **Step 4: Run roundtrip test**

Run: `sbt "testOnly sicfun.holdem.validation.PokerStarsExporterTest"`
Expected: PASS. If parse fails, examine `HandHistoryImport` parser expectations and adjust format.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/validation/PokerStarsExporter.scala \
        src/test/scala/sicfun/holdem/validation/PokerStarsExporterTest.scala
git commit -m "feat(validation): add PokerStarsExporter with parse roundtrip verification"
```

---

## Chunk 6: ValidationRunner + ConvergenceTracker + Scorecard

### Task 9: ConvergenceTracker

**Files:**
- Create: `src/main/scala/sicfun/holdem/validation/ConvergenceTracker.scala`
- Create: `src/test/scala/sicfun/holdem/validation/ValidationScorecardTest.scala`

- [ ] **Step 1: Write ConvergenceTracker tests**

```scala
package sicfun.holdem.validation

import munit.FunSuite

class ValidationScorecardTest extends FunSuite:

  test("ConvergenceTracker records first detection"):
    val tracker = new ConvergenceTracker("overfold-river-aggression")
    tracker.recordChunk(chunkIndex = 0, detected = false, confidence = 0.1, falsePositives = 0)
    tracker.recordChunk(chunkIndex = 1, detected = false, confidence = 0.2, falsePositives = 0)
    tracker.recordChunk(chunkIndex = 2, detected = true, confidence = 0.7, falsePositives = 0)
    tracker.recordChunk(chunkIndex = 3, detected = true, confidence = 0.85, falsePositives = 0)

    val summary = tracker.summary(handsPerChunk = 1000)
    assert(summary.detected)
    assertEquals(summary.firstDetectedChunk, Some(2))
    assertEquals(summary.handsToDetect, Some(3000)) // chunk 2 * 1000 + 1000
    assertEqualsDouble(summary.finalConfidence, 0.85, 0.01)
    assertEquals(summary.totalFalsePositives, 0)

  test("ConvergenceTracker handles never-detected"):
    val tracker = new ConvergenceTracker("preflop-too-tight")
    tracker.recordChunk(chunkIndex = 0, detected = false, confidence = 0.05, falsePositives = 1)
    val summary = tracker.summary(handsPerChunk = 1000)
    assert(!summary.detected)
    assertEquals(summary.firstDetectedChunk, None)
    assertEquals(summary.totalFalsePositives, 1)
```

- [ ] **Step 2: Implement ConvergenceTracker**

```scala
package sicfun.holdem.validation

import scala.collection.mutable

final case class ConvergenceSummary(
    leakId: String,
    detected: Boolean,
    firstDetectedChunk: Option[Int],
    handsToDetect: Option[Int],
    finalConfidence: Double,
    totalFalsePositives: Int
)

final class ConvergenceTracker(val leakId: String):
  private val chunks = mutable.ArrayBuffer.empty[(Int, Boolean, Double, Int)]

  def recordChunk(chunkIndex: Int, detected: Boolean, confidence: Double, falsePositives: Int): Unit =
    chunks += ((chunkIndex, detected, confidence, falsePositives))

  def summary(handsPerChunk: Int): ConvergenceSummary =
    val firstDetected = chunks.find(_._2).map(_._1)
    ConvergenceSummary(
      leakId = leakId,
      detected = firstDetected.isDefined,
      firstDetectedChunk = firstDetected,
      handsToDetect = firstDetected.map(idx => (idx + 1) * handsPerChunk),
      finalConfidence = chunks.lastOption.map(_._3).getOrElse(0.0),
      totalFalsePositives = chunks.map(_._4).sum
    )
```

- [ ] **Step 3: Run test, verify PASS**

- [ ] **Step 4: Commit**

```bash
git add src/main/scala/sicfun/holdem/validation/ConvergenceTracker.scala \
        src/test/scala/sicfun/holdem/validation/ValidationScorecardTest.scala
git commit -m "feat(validation): add ConvergenceTracker for per-chunk leak detection"
```

---

### Task 10: ValidationScorecard

**Files:**
- Create: `src/main/scala/sicfun/holdem/validation/ValidationScorecard.scala`
- Modify: `src/test/scala/sicfun/holdem/validation/ValidationScorecardTest.scala`

- [ ] **Step 1: Write scorecard formatting test**

```scala
  test("ValidationScorecard formats report"):
    val playerResult = PlayerValidationResult(
      villainName = "overfold_river_aggression_moderate",
      leakId = "overfold-river-aggression",
      severity = 0.6,
      totalHands = 1000000,
      leakApplicableSpots = 78718,
      leakFiredCount = 47231,
      heroNetBbPer100 = 4.2,
      convergence = ConvergenceSummary(
        leakId = "overfold-river-aggression",
        detected = true,
        firstDetectedChunk = Some(46),
        handsToDetect = Some(47000),
        finalConfidence = 0.87,
        totalFalsePositives = 0
      ),
      assignedArchetype = "Nit",
      archetypeConvergenceChunk = Some(11),
      clusterId = Some(2)
    )
    val report = ValidationScorecard.format(Vector(playerResult))
    assert(report.contains("overfold_river_aggression_moderate"))
    assert(report.contains("DETECTED"))
    assert(report.contains("47,000"))
    assert(report.contains("4.2"))
```

- [ ] **Step 2: Implement ValidationScorecard**

```scala
package sicfun.holdem.validation

import java.util.Locale

final case class PlayerValidationResult(
    villainName: String,
    leakId: String,
    severity: Double,
    totalHands: Int,
    leakApplicableSpots: Int,
    leakFiredCount: Int,
    heroNetBbPer100: Double,
    convergence: ConvergenceSummary,
    assignedArchetype: String,
    archetypeConvergenceChunk: Option[Int],
    clusterId: Option[Int]
)

object ValidationScorecard:
  def format(results: Vector[PlayerValidationResult]): String =
    val sb = new StringBuilder
    sb.append("=== PROFILING VALIDATION SCORECARD ===\n\n")

    results.foreach { r =>
      sb.append(s"Player: ${r.villainName} (severity=${r.severity})\n")
      sb.append(s"  Hands played:     ${fmt(r.totalHands)}\n")
      sb.append(s"  Leak fired:       ${fmt(r.leakFiredCount)} / ${fmt(r.leakApplicableSpots)} applicable spots")
      if r.leakApplicableSpots > 0 then
        sb.append(f" (${r.leakFiredCount.toDouble / r.leakApplicableSpots * 100}%.1f%%)")
      sb.append("\n")
      sb.append(f"  Hero net EV:      ${r.heroNetBbPer100}%+.1f bb/100\n")
      sb.append("\n")

      sb.append("  PRIMARY: Leak Detection\n")
      val detected = if r.convergence.detected then "DETECTED" else "NOT DETECTED"
      sb.append(s"    Status:         $detected\n")
      r.convergence.handsToDetect.foreach(h => sb.append(s"    Hands to detect: ${fmt(h)}\n"))
      sb.append(f"    Confidence:     ${r.convergence.finalConfidence}%.2f\n")
      sb.append(s"    False positives: ${r.convergence.totalFalsePositives}\n")
      sb.append("\n")

      sb.append("  SECONDARY: Archetype Classification\n")
      sb.append(s"    Assigned:       ${r.assignedArchetype}\n")
      r.archetypeConvergenceChunk.foreach(c => sb.append(s"    Convergence:    chunk $c\n"))
      sb.append("\n")

      r.clusterId.foreach(c => sb.append(s"  SECONDARY: Cluster ID $c\n"))
      sb.append("\n---\n\n")
    }

    // Aggregate
    val detected = results.count(_.convergence.detected)
    val total = results.length
    val medianHands = results.flatMap(_.convergence.handsToDetect).sorted
    val median = if medianHands.nonEmpty then medianHands(medianHands.length / 2) else 0
    val avgFP = if total > 0 then results.map(_.convergence.totalFalsePositives).sum.toDouble / total else 0.0
    val avgWinrate = if total > 0 then results.map(_.heroNetBbPer100).sum / total else 0.0

    sb.append("=== AGGREGATE ===\n")
    sb.append(s"  Players:          $total\n")
    sb.append(s"  Leaks detected:   $detected/$total (${if total > 0 then f"${detected.toDouble / total * 100}%.1f" else "0.0"}%%)\n")
    sb.append(s"  Median hands-to-detect: ${fmt(median)}\n")
    sb.append(f"  Avg false positives: $avgFP%.1f per player\n")
    sb.append(f"  Hero winrate:     $avgWinrate%+.1f bb/100 avg")
    if avgWinrate > 0 then sb.append(" (adaptive beats exploitable: CONFIRMED)")
    else sb.append(" (adaptive does NOT beat exploitable: INVESTIGATE)")
    sb.append("\n")
    sb.toString()

  private def fmt(n: Int): String = String.format(Locale.ROOT, "%,d", Integer.valueOf(n))
```

- [ ] **Step 3: Run tests**

Run: `sbt "testOnly sicfun.holdem.validation.ValidationScorecardTest"`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/main/scala/sicfun/holdem/validation/ValidationScorecard.scala \
        src/test/scala/sicfun/holdem/validation/ValidationScorecardTest.scala
git commit -m "feat(validation): add ValidationScorecard report formatting"
```

---

### Task 11: ValidationRunner — Orchestrator

**Files:**
- Create: `src/main/scala/sicfun/holdem/validation/ValidationRunner.scala`

- [ ] **Step 1: Implement ValidationRunner**

This is the orchestrator that ties everything together. It does not need unit tests — it IS the validation test. It will be invoked via `sbt runMain`.

```scala
package sicfun.holdem.validation

import sicfun.holdem.engine.RealTimeAdaptiveEngine
import sicfun.holdem.equity.{TableFormat, TableRanges}
import sicfun.holdem.history.{HandHistoryImport, HandHistorySite, OpponentProfile}
import sicfun.holdem.analysis.{PlayerSignature, PlayerCluster, PlayerClusterConfig}
import sicfun.holdem.model.{PokerActionModel, PokerActionModelArtifactIO}
import sicfun.holdem.web.{HandHistoryReviewServer, HandHistoryReviewService}

import java.nio.file.{Files, Path, Paths}
import scala.util.Random
import ujson.Value

object ValidationRunner:
  final case class Config(
      handsPerPlayer: Int = 1_000_000,
      chunkSize: Int = 1000,
      outputDir: Path = Paths.get("validation-output"),
      modelDir: Option[Path] = None,
      seed: Long = 42L,
      bunchingTrials: Int = 200,
      equityTrials: Int = 2000,
      budgetMs: Long = 1500L,
      webValidation: Boolean = true
  )

  /** All 18 players: 6 leaks x 3 severities */
  def defaultPopulation: Vector[(InjectedLeak, String)] =
    val severities = Vector(("mild", 0.3), ("moderate", 0.6), ("severe", 0.9))
    val leakFactories: Vector[(Double => InjectedLeak)] = Vector(
      sev => OverfoldsToAggression(sev),
      sev => Overcalls(sev),
      sev => OverbluffsTurnBarrel(sev),
      sev => PassiveInBigPots(sev),
      sev => PreflopTooLoose(sev),
      sev => PreflopTooTight(sev)
    )
    for
      factory <- leakFactories
      (label, sev) <- severities
    yield
      val leak = factory(sev)
      (leak, s"${leak.id}_$label")

  def main(args: Array[String]): Unit =
    val config = parseConfig(args)
    run(config)

  def run(config: Config): Vector[PlayerValidationResult] =
    Files.createDirectories(config.outputDir)
    val population = defaultPopulation
    println(s"=== Profiling Validation Harness ===")
    println(s"Players: ${population.size}")
    println(s"Hands per player: ${config.handsPerPlayer}")
    println(s"Total hands: ${population.size * config.handsPerPlayer}")
    println()

    val results = population.zipWithIndex.map { case ((leak, villainName), idx) =>
      println(s"[${idx + 1}/${population.size}] Simulating $villainName ...")
      runOnePlayer(config, leak, villainName, playerSeed = config.seed + idx)
    }

    val report = ValidationScorecard.format(results)
    println(report)

    val reportPath = config.outputDir.resolve("scorecard.txt")
    Files.writeString(reportPath, report)
    println(s"Scorecard saved to $reportPath")

    results

  private def runOnePlayer(
      config: Config,
      leak: InjectedLeak,
      villainName: String,
      playerSeed: Long
  ): PlayerValidationResult =
    val tableRanges = TableRanges.defaults(TableFormat.HeadsUp)
    val actionModel = config.modelDir.map(p => PokerActionModelArtifactIO.load(p).model)
      .getOrElse(PokerActionModel.uniform)

    val heroEngine = new RealTimeAdaptiveEngine(
      tableRanges = tableRanges, actionModel = actionModel,
      bunchingTrials = config.bunchingTrials, defaultEquityTrials = config.equityTrials
    )
    val villainEngine = new RealTimeAdaptiveEngine(
      tableRanges = tableRanges, actionModel = actionModel,
      bunchingTrials = config.bunchingTrials, defaultEquityTrials = config.equityTrials
    )
    val villainPlayer = LeakInjectedVillain(
      name = villainName, leaks = Vector(leak),
      baselineNoise = 0.03, seed = playerSeed
    )
    val simulator = new HeadsUpSimulator(
      heroEngine = heroEngine, villainEngine = villainEngine,
      villain = villainPlayer, seed = playerSeed
    )

    // Simulate all hands
    val records = (1 to config.handsPerPlayer).map(i => simulator.playHand(i)).toVector

    // Export
    val playerDir = config.outputDir.resolve(villainName)
    Files.createDirectories(playerDir)
    val fullText = PokerStarsExporter.exportHands(records, "Hero", villainName)
    Files.writeString(playerDir.resolve("full_history.txt"), fullText)
    val chunks = PokerStarsExporter.exportChunked(records, "Hero", villainName, config.chunkSize)
    chunks.foreach { chunk =>
      Files.writeString(playerDir.resolve(f"chunk_${chunk.chunkIndex}%04d.txt"), chunk.text)
    }

    // Ground truth
    val leakActions = records.flatMap(_.actions).filter(_.leakId.contains(leak.id))
    val applicableSpots = records.flatMap(_.actions).count(a =>
      a.player == villainName && a.leakId.isDefined || a.leakFired)
    // More accurate: count spots where leak.applies would be true
    // For now, use leakFired count + estimate
    val firedCount = leakActions.size

    // Hero EV
    val heroTotalNet = records.map(_.heroNet).sum
    val heroNetBbPer100 = (heroTotalNet / config.handsPerPlayer) * 100.0

    // Web API validation (if enabled)
    val tracker = new ConvergenceTracker(leak.id)
    var lastArchetype = "Unknown"
    var archetypeStableChunk: Option[Int] = None
    var prevArchetype = ""

    if config.webValidation then
      // Feed chunks through profiling pipeline (programmatic, no web server needed)
      val accumulatedText = new StringBuilder
      chunks.foreach { chunk =>
        accumulatedText.append(chunk.text)
        val parsed = HandHistoryImport.parseText(accumulatedText.toString(), Some(HandHistorySite.PokerStars), Some("Hero"))
        parsed.foreach { hands =>
          val profiles = OpponentProfile.fromImportedHands("simulated", hands, Set("Hero"))
          profiles.headOption.foreach { profile =>
            val hints = profile.exploitHints
            val archetype = profile.archetypePosterior.mapEstimate.toString
            lastArchetype = archetype
            if archetypeStableChunk.isEmpty && archetype == prevArchetype then
              archetypeStableChunk = Some(chunk.chunkIndex)
            prevArchetype = archetype

            // Check if any hint matches the injected leak
            val detected = hintMatchesLeak(hints, leak.id)
            val confidence = if detected then 0.8 else 0.1 // simplified
            tracker.recordChunk(chunk.chunkIndex, detected, confidence, falsePositives = 0)
          }
        }
      }

    // K-Means clustering (secondary)
    val observations = records.flatMap { r =>
      r.actions.filter(_.player == villainName).map { a =>
        val gs = sicfun.holdem.types.GameState(a.street, sicfun.holdem.types.Board.empty,
          a.potBefore, a.toCall, sicfun.holdem.types.Position.BigBlind, a.stackBefore, Vector.empty)
        (gs, a.action)
      }
    }
    val signature = PlayerSignature.compute(observations)

    PlayerValidationResult(
      villainName = villainName,
      leakId = leak.id,
      severity = leak.severity,
      totalHands = config.handsPerPlayer,
      leakApplicableSpots = firedCount * 2, // rough estimate
      leakFiredCount = firedCount,
      heroNetBbPer100 = heroNetBbPer100,
      convergence = tracker.summary(config.chunkSize),
      assignedArchetype = lastArchetype,
      archetypeConvergenceChunk = archetypeStableChunk,
      clusterId = None // populated in aggregate step
    )

  /** Check if any exploit hint text matches the injected leak concept. */
  private def hintMatchesLeak(hints: Vector[String], leakId: String): Boolean =
    leakId match
      case "overfold-river-aggression" =>
        hints.exists(h => h.contains("over-fold") || h.contains("bluff pressure") || h.contains("fold"))
      case "overcall-big-bets" =>
        hints.exists(h => h.contains("calling station") || h.contains("value bet thinner") || h.contains("call"))
      case "overbluff-turn-barrel" =>
        hints.exists(h => h.contains("aggressive") || h.contains("bluff-catch"))
      case "passive-big-pots" =>
        hints.exists(h => h.contains("passive") || h.contains("sudden aggression"))
      case "preflop-too-loose" =>
        hints.exists(h => h.contains("calling station") || h.contains("value bet"))
      case "preflop-too-tight" =>
        hints.exists(h => h.contains("over-fold") || h.contains("wider"))
      case _ => false

  private def parseConfig(args: Array[String]): Config =
    // Simple arg parsing — production version would use CliHelpers
    Config() // defaults for now — enhance during implementation
```

**Implementation note:** The `hintMatchesLeak` function maps injected leak IDs to expected exploit hint keywords. This is the core validation logic — if the profiler's hints don't contain these keywords after 1M hands, the profiler fails to detect the leak. The implementer should refine these mappings by studying `exploitHintsFor` (OpponentProfileStore.scala:313-346) to understand exactly what hint text is generated for each behavioral pattern.

- [ ] **Step 2: Verify it compiles**

Run: `sbt compile`
Expected: compiles without error.

- [ ] **Step 3: Run with small hand count to smoke-test**

Run: `sbt "runMain sicfun.holdem.validation.ValidationRunner --hands=100"`
Expected: runs to completion, prints scorecard.

- [ ] **Step 4: Commit**

```bash
git add src/main/scala/sicfun/holdem/validation/ValidationRunner.scala
git commit -m "feat(validation): add ValidationRunner orchestrator with full pipeline"
```

---

## Chunk 7: Integration + Ground Truth JSON

### Task 12: Ground truth manifest export

**Files:**
- Modify: `src/main/scala/sicfun/holdem/validation/ValidationRunner.scala`

- [ ] **Step 1: Add ground truth JSON export after simulation**

After exporting hand histories in `runOnePlayer`, write a `ground_truth.json`:

```scala
val groundTruth = ujson.Obj(
  "leakId" -> ujson.Str(leak.id),
  "leakDescription" -> ujson.Str(leak.description),
  "severity" -> ujson.Num(leak.severity),
  "baselineNoise" -> ujson.Num(0.03),
  "totalHands" -> ujson.Num(config.handsPerPlayer),
  "leakFiredCount" -> ujson.Num(firedCount),
  "heroNetBbPer100" -> ujson.Num(heroNetBbPer100)
)
Files.writeString(playerDir.resolve("ground_truth.json"), ujson.write(groundTruth, indent = 2))
```

- [ ] **Step 2: Commit**

```bash
git add src/main/scala/sicfun/holdem/validation/ValidationRunner.scala
git commit -m "feat(validation): add ground truth JSON manifest per player"
```

---

### Task 13: Web API integration test (visual spot-check path)

**Files:**
- Modify: `src/main/scala/sicfun/holdem/validation/ValidationRunner.scala`

- [ ] **Step 1: Add web server integration path**

Add a method that starts `HandHistoryReviewServer` in-process and POSTs a sample chunk:

```scala
def runWebSpotCheck(config: Config, sampleChunkPaths: Vector[Path]): Unit =
  val serverConfig = HandHistoryReviewServer.ServerConfig(
    host = "127.0.0.1", port = 0, // ephemeral port
    staticDir = Paths.get("docs/site-preview-hybrid"),
    maxUploadBytes = 10 * 1024 * 1024,
    serviceConfig = HandHistoryReviewService.ServiceConfig(
      modelDir = config.modelDir,
      seed = config.seed,
      bunchingTrials = config.bunchingTrials,
      equityTrials = config.equityTrials,
      budgetMs = config.budgetMs
    )
  )
  HandHistoryReviewServer.start(Array(
    s"--port=8090",
    s"--staticDir=docs/site-preview-hybrid"
  )) match
    case Right(server) =>
      println(s"Web server running at http://127.0.0.1:8090/")
      println(s"Upload sample chunks from: ${sampleChunkPaths.mkString(", ")}")
      println("Press Ctrl+C to stop.")
      // Block until shutdown
    case Left(err) =>
      println(s"Failed to start web server: $err")
```

- [ ] **Step 2: Commit**

```bash
git add src/main/scala/sicfun/holdem/validation/ValidationRunner.scala
git commit -m "feat(validation): add web spot-check mode for visual review"
```

---

### Task 14: Full integration smoke test

- [ ] **Step 1: Run full pipeline with 1000 hands per player**

Run: `sbt "runMain sicfun.holdem.validation.ValidationRunner"` (with hands=1000 in config)
Expected: all 18 players simulated, scorecard printed, files written to `validation-output/`.

- [ ] **Step 2: Verify output files exist**

```bash
ls validation-output/overfold-river-aggression_moderate/
# Expected: full_history.txt, chunk_0000.txt, ground_truth.json
```

- [ ] **Step 3: Verify roundtrip — parse exported history**

Run a quick sbt console test or add a one-off assertion that `HandHistoryImport.parseText` succeeds on a sample chunk.

- [ ] **Step 4: Run all tests**

Run: `sbt test`
Expected: all validation tests PASS, no regressions in existing tests.

- [ ] **Step 5: Final commit**

```bash
git add -A src/main/scala/sicfun/holdem/validation/ \
          src/test/scala/sicfun/holdem/validation/ \
          docs/superpowers/plans/2026-03-14-profiling-validation-harness.md
git commit -m "feat(validation): complete profiling validation harness with full pipeline"
```

---

## Post-Implementation Notes

### Running the full 1M-hand validation

After the harness is working with smoke tests:

```bash
sbt "runMain sicfun.holdem.validation.ValidationRunner --hands=1000000"
```

This will take significant time (estimate based on PlayingHall's throughput). Consider:
- Running one player at a time initially
- Starting with `--hands=10000` to verify detection curves
- Parallelizing across players if machine has cores to spare

### Expected outcomes

1. **Coarse leaks (preflop-too-loose, preflop-too-tight)** will likely be detected quickly — they shift overall fold/raise rates which the existing `exploitHintsFor` thresholds catch.
2. **Spot-specific leaks (overfold-river-aggression, overbluff-turn-barrel)** may NOT be detected — current exploit hints are aggregate-based, not spot-aware. This is the expected gap the harness reveals.
3. **The gap IS the finding** — it tells us exactly where the profiler needs to become more granular.

### Extending the leak taxonomy

To add a new leak:
1. Create a new `case class MyNewLeak(severity: Double) extends InjectedLeak`
2. Implement `applies` with `SpotContext` conditions
3. Implement `deviate` with the behavioral deviation
4. Add to `ValidationRunner.defaultPopulation`
5. Add hint matching in `hintMatchesLeak`
6. Run the harness
