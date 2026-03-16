# CFR GTO Villain & Profiler Threshold Calibration

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the equity-heuristic "GTO" villain with one that uses `HoldemCfrSolver` for actual equilibrium decisions, then calibrate the profiler's hardcoded thresholds against it.

**Architecture:** Extract a `VillainStrategy` trait from `HeadsUpSimulator` to make the GTO baseline pluggable. Implement `CfrVillainStrategy` wrapping `HoldemCfrSolver.solveDecisionPolicy`. Add a calibration runner that profiles the CFR villain and reports which thresholds in `exploitHintsFor` disagree with equilibrium play. Adjust thresholds based on measured rates.

**Tech Stack:** Scala 3.8.1, munit 1.2.2, existing `HoldemCfrSolver` + `HoldemEquity.fullRange` APIs.

---

## Why This Matters

The profiler (`OpponentProfileStore.exploitHintsFor`) uses hardcoded thresholds — "fold rate >= 20% = too tight", "call rate >= 65% = too loose". These are guesses, not derived from any solver output. The "GTO" control villain in `HeadsUpSimulator.equityBasedDecision` uses equity-vs-random thresholds, not equilibrium play. Its tendencies (calls too much facing large bets, passive in big pots) might be real leaks the profiler correctly detects — but they shouldn't be, because it's supposed to be the GTO baseline. The profiler's expected baseline and the GTO control player need to be the same strategy.

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/main/scala/sicfun/holdem/validation/VillainStrategy.scala` | Trait + EquityBased + CfrBased implementations |
| Modify | `src/main/scala/sicfun/holdem/validation/HeadsUpSimulator.scala` | Accept `VillainStrategy` parameter, delegate GTO decision |
| Modify | `src/main/scala/sicfun/holdem/validation/ValidationRunner.scala` | Pass strategy to simulator |
| Modify | `src/main/scala/sicfun/holdem/history/OpponentProfileStore.scala` | Calibrate thresholds from measured rates |
| Create | `src/test/scala/sicfun/holdem/validation/VillainStrategyTest.scala` | Unit tests for both strategies |
| Create | `src/test/scala/sicfun/holdem/validation/CfrGtoCalibrationTest.scala` | Integration test: CFR villain produces zero false positive hints + calibration data |

---

## Chunk 1: VillainStrategy Trait & EquityBasedStrategy Extraction

### Task 1: Define VillainStrategy trait and EquityBasedStrategy

**Files:**
- Create: `src/main/scala/sicfun/holdem/validation/VillainStrategy.scala`
- Test: `src/test/scala/sicfun/holdem/validation/VillainStrategyTest.scala`

#### Context

Currently, `HeadsUpSimulator` has a private method `equityBasedDecision` (line 262) that both hero and villain use for fast equity-based decisions, plus `estimateEquityVsRandom` (line 357) for MC equity estimation. We need to extract the villain path into a pluggable strategy.

The trait interface needs:
- `hand: HoleCards` — villain's private cards
- `state: GameState` — current pot/board/stack/toCall/position
- `candidates: Vector[PokerAction]` — legal actions (from `buildCandidates`)
- `equityVsRandom: Double` — precomputed equity (used by EquityBased, ignored by CFR)
- `rng: Random` — for stochastic sampling

- [ ] **Step 1: Write the failing test**

```scala
// src/test/scala/sicfun/holdem/validation/VillainStrategyTest.scala
package sicfun.holdem.validation

import munit.FunSuite
import sicfun.holdem.types.*

import scala.util.Random

class VillainStrategyTest extends FunSuite:

  test("EquityBasedStrategy produces valid actions facing a bet"):
    val strategy = EquityBasedStrategy()
    val gs = GameState(
      street = Street.Flop,
      board = Board.empty,
      pot = 10.0,
      toCall = 5.0,
      position = Position.BigBlind,
      stackSize = 95.0,
      betHistory = Vector.empty
    )
    val candidates = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(20.0))
    // Strong hand equity — should not fold
    val action = strategy.decide(HoleCards.parse("AhAs").get, gs, candidates, 0.85, new Random(42))
    assert(action != PokerAction.Fold, s"Strong hand should not fold, got $action")

  test("EquityBasedStrategy folds weak hands facing a bet"):
    val strategy = EquityBasedStrategy()
    val gs = GameState(
      street = Street.River,
      board = Board.empty,
      pot = 10.0,
      toCall = 8.0,
      position = Position.BigBlind,
      stackSize = 92.0,
      betHistory = Vector.empty
    )
    val candidates = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(26.0))
    // Run 100 trials — should fold most of the time with equity = 0.10
    val foldCount = (0 until 100).count { i =>
      strategy.decide(HoleCards.parse("2h3d").get, gs, candidates, 0.10, new Random(i)) == PokerAction.Fold
    }
    assert(foldCount >= 50, s"Weak hand should fold often, only folded $foldCount/100 times")

  test("EquityBasedStrategy value bets when not facing a bet"):
    val strategy = EquityBasedStrategy()
    val gs = GameState(
      street = Street.River,
      board = Board.empty,
      pot = 20.0,
      toCall = 0.0,
      position = Position.BigBlind,
      stackSize = 90.0,
      betHistory = Vector.empty
    )
    val candidates = Vector(PokerAction.Check, PokerAction.Raise(13.2), PokerAction.Raise(20.0))
    // Strong hand — should bet most of the time
    val betCount = (0 until 100).count { i =>
      strategy.decide(HoleCards.parse("AhAs").get, gs, candidates, 0.85, new Random(i)).isRaise
    }
    assert(betCount >= 50, s"Strong hand should bet often when checked to, only bet $betCount/100 times")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly sicfun.holdem.validation.VillainStrategyTest"`
Expected: FAIL — `VillainStrategy`, `EquityBasedStrategy` not found.

- [ ] **Step 3: Create VillainStrategy trait and EquityBasedStrategy**

```scala
// src/main/scala/sicfun/holdem/validation/VillainStrategy.scala
package sicfun.holdem.validation

import sicfun.holdem.types.*

import scala.util.Random

/** Pluggable GTO baseline strategy for the simulated villain.
  *
  * Implementations decide what the "competent" action is before any leak
  * injection. EquityBasedStrategy uses equity-vs-random thresholds (fast,
  * heuristic). CfrVillainStrategy uses HoldemCfrSolver for actual equilibrium.
  */
trait VillainStrategy:
  /** Choose an action for the given spot.
    *
    * @param hand            villain's private cards
    * @param state           current game state (pot, board, toCall, position, stack)
    * @param candidates      legal actions (from buildCandidates)
    * @param equityVsRandom  precomputed equity vs uniform random hand
    * @param rng             random source for stochastic sampling
    */
  def decide(
      hand: HoleCards,
      state: GameState,
      candidates: Vector[PokerAction],
      equityVsRandom: Double,
      rng: Random
  ): PokerAction

/** Equity-threshold heuristic — the original HeadsUpSimulator strategy.
  *
  * Uses precomputed equity vs random to pick a "competent" action without
  * Bayesian inference or CFR solving. Fast but not equilibrium.
  */
final class EquityBasedStrategy extends VillainStrategy:
  def decide(
      hand: HoleCards,
      state: GameState,
      candidates: Vector[PokerAction],
      equityVsRandom: Double,
      rng: Random
  ): PokerAction =
    if candidates.size <= 1 then return candidates.headOption.getOrElse(PokerAction.Check)

    val potOdds = state.potOdds
    val street = state.street

    if state.toCall > 0 then
      val raiseActions = candidates.collect { case r: PokerAction.Raise => r }
      // HU Button/SB preflop: open-raise strategy
      if street == Street.Preflop && state.position == Position.Button then
        if equityVsRandom >= 0.35 then
          if raiseActions.nonEmpty then raiseActions.head else PokerAction.Call
        else if equityVsRandom >= 0.20 && rng.nextDouble() < 0.4 then
          if raiseActions.nonEmpty then raiseActions.head else PokerAction.Call
        else PokerAction.Fold
      else if equityVsRandom >= 0.75 then
        if raiseActions.nonEmpty && rng.nextDouble() < 0.6 then raiseActions.head
        else PokerAction.Call
      else if equityVsRandom >= potOdds + 0.05 then
        if street == Street.Preflop && raiseActions.nonEmpty && rng.nextDouble() < 0.25 then
          raiseActions.head
        else if equityVsRandom >= 0.55 && raiseActions.nonEmpty && rng.nextDouble() < 0.20 then
          raiseActions.head
        else PokerAction.Call
      else if street == Street.Preflop && rng.nextDouble() < 0.70 then
        if raiseActions.nonEmpty && rng.nextDouble() < 0.20 then raiseActions.head
        else PokerAction.Call
      else if rng.nextDouble() < 0.25 then
        PokerAction.Call
      else PokerAction.Fold
    else
      if equityVsRandom >= 0.60 then
        val raiseActions = candidates.collect { case r: PokerAction.Raise => r }
        if raiseActions.nonEmpty then raiseActions(rng.nextInt(raiseActions.size))
        else PokerAction.Check
      else if equityVsRandom <= 0.30 && rng.nextDouble() < 0.25 then
        val raiseActions = candidates.collect { case r: PokerAction.Raise => r }
        if raiseActions.nonEmpty then raiseActions.head else PokerAction.Check
      else PokerAction.Check
```

- [ ] **Step 4: Run test to verify it passes**

Run: `sbt "testOnly sicfun.holdem.validation.VillainStrategyTest"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/validation/VillainStrategy.scala \
        src/test/scala/sicfun/holdem/validation/VillainStrategyTest.scala
git commit -m "feat: extract VillainStrategy trait and EquityBasedStrategy"
```

---

### Task 2: Wire HeadsUpSimulator to accept VillainStrategy

**Files:**
- Modify: `src/main/scala/sicfun/holdem/validation/HeadsUpSimulator.scala`
- Modify: `src/main/scala/sicfun/holdem/validation/ValidationRunner.scala`

#### Context

`HeadsUpSimulator` currently has `equityBasedDecision` and `decideVillainGto` as private methods. We need to:
1. Add a `villainStrategy: VillainStrategy = EquityBasedStrategy()` constructor parameter
2. Delegate `decideVillainGto` to `villainStrategy.decide(...)`
3. Keep `equityBasedDecision` for the fast hero path (hero uses it when `heroEngine = None`)
4. Update `ValidationRunner` to pass the strategy through

- [ ] **Step 1: Write the failing test**

```scala
// Add to VillainStrategyTest.scala
  test("HeadsUpSimulator uses pluggable VillainStrategy"):
    // Custom strategy that always returns Call when facing a bet, Check otherwise
    val callStrategy = new VillainStrategy:
      def decide(hand: HoleCards, state: GameState, candidates: Vector[PokerAction],
                 equityVsRandom: Double, rng: Random): PokerAction =
        if state.toCall > 0 then PokerAction.Call else PokerAction.Check

    val villain = LeakInjectedVillain("test", Vector(NoLeak()), 0.0, 42L)
    val sim = new HeadsUpSimulator(
      heroEngine = None,
      villain = villain,
      seed = 42L,
      villainStrategy = callStrategy
    )
    val record = sim.playHand(1)
    // All villain actions should be Call or Check (never Fold or Raise)
    val villainActions = record.actions.filter(_.player == villain.name)
    villainActions.foreach { ra =>
      assert(ra.action == PokerAction.Call || ra.action == PokerAction.Check,
        s"Expected Call/Check but got ${ra.action}")
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly sicfun.holdem.validation.VillainStrategyTest -- *pluggable*"`
Expected: FAIL — `HeadsUpSimulator` constructor doesn't accept `villainStrategy` parameter.

- [ ] **Step 3: Modify HeadsUpSimulator to accept VillainStrategy**

In `HeadsUpSimulator.scala`:

1. Add constructor parameter:
```scala
final class HeadsUpSimulator(
    heroEngine: Option[RealTimeAdaptiveEngine] = None,
    villain: LeakInjectedVillain,
    seed: Long,
    equityTrialsForCategory: Int = 500,
    startingStack: Double = 100.0,
    smallBlind: Double = 0.5,
    bigBlind: Double = 1.0,
    budgetMs: Long = 50L,
    villainStrategy: VillainStrategy = EquityBasedStrategy()  // NEW
):
```

2. Replace the body of `decideVillainGto` (lines 250–259):
```scala
  private def decideVillainGto(
      villainHand: HoleCards,
      gs: GameState,
      board: Board,
      street: Street,
      equity: Double
  ): PokerAction =
    val candidates = buildCandidates(gs)
    if candidates.size <= 1 then return candidates.headOption.getOrElse(PokerAction.Check)
    villainStrategy.decide(villainHand, gs, candidates, equity, new Random(rng.nextLong()))
```

3. The `equityBasedDecision` private method stays — it's still used by `decideHero` when `heroEngine = None` (line 243). Do NOT delete it.

- [ ] **Step 4: Run test to verify it passes**

Run: `sbt "testOnly sicfun.holdem.validation.VillainStrategyTest"`
Expected: PASS (all tests including the new one)

- [ ] **Step 5: Run existing tests to verify no regressions**

Run: `sbt "testOnly sicfun.holdem.validation.*"`
Expected: All existing `HeadsUpSimulatorTest`, `GtoBaselineFalsePositiveTest`, `SpotContextTest`, `InjectedLeakTest`, `ValidationRunner` tests PASS.

**Important:** The default `villainStrategy = EquityBasedStrategy()` means all existing call sites are unchanged. No other file needs modification unless explicitly passing a strategy. `ValidationRunner` does NOT need changes for this step — it uses the default.

- [ ] **Step 6: Commit**

```bash
git add src/main/scala/sicfun/holdem/validation/HeadsUpSimulator.scala \
        src/test/scala/sicfun/holdem/validation/VillainStrategyTest.scala
git commit -m "refactor: wire HeadsUpSimulator to accept pluggable VillainStrategy"
```

---

## Chunk 2: CfrVillainStrategy Implementation

### Task 3: Implement CfrVillainStrategy

**Files:**
- Modify: `src/main/scala/sicfun/holdem/validation/VillainStrategy.scala`
- Modify: `src/test/scala/sicfun/holdem/validation/VillainStrategyTest.scala`

#### Context

`HoldemCfrSolver` (in `sicfun.holdem.cfr`) exposes:
```scala
def solveDecisionPolicy(
    hero: HoleCards,          // the player making the decision
    state: GameState,         // current pot/board/toCall/position/stack
    villainPosterior: DiscreteDistribution[HoleCards],  // opponent's range
    candidateActions: Vector[PokerAction],
    config: HoldemCfrConfig = HoldemCfrConfig()
): HoldemCfrDecisionPolicy
```

Returns `HoldemCfrDecisionPolicy` with `actionProbabilities: Map[PokerAction, Double]`.

To build the opponent's range (uniform prior excluding villain's cards and board), use:
```scala
HoldemEquity.fullRange(hand, state.board)  // in sicfun.holdem.equity
```

The CFR villain samples from the mixed strategy. For speed, use reduced config:
- `iterations = 300` (not 1500 — good enough for baseline calibration)
- `equityTrials = 500` (not 4000)
- `maxVillainHands = 48` (not 96)
- `includeVillainReraises = false` (simpler game tree, faster solve)

**Performance note:** Each decision requires one CFR solve. For a 2000-hand calibration with ~3 villain decisions/hand = ~6000 solves. At ~5ms/solve (native batch), that's ~30 seconds. Acceptable.

- [ ] **Step 1: Write the failing test**

```scala
// Add to VillainStrategyTest.scala

  test("CfrVillainStrategy produces valid actions from equilibrium solve"):
    val solver = new sicfun.holdem.cfr.HoldemCfrSolver()
    val strategy = CfrVillainStrategy(solver)
    val gs = GameState(
      street = Street.Flop,
      board = Board.parse("Ah Kd 7c").get,
      pot = 6.0,
      toCall = 3.0,
      position = Position.BigBlind,
      stackSize = 97.0,
      betHistory = Vector.empty
    )
    val candidates = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(12.0))
    val hand = HoleCards.parse("QhQd").get
    val action = strategy.decide(hand, gs, candidates, 0.70, new Random(42))
    assert(candidates.contains(action) || action == PokerAction.Check || action == PokerAction.Call,
      s"CfrVillainStrategy returned invalid action: $action")

  test("CfrVillainStrategy produces mixed strategy (not deterministic)"):
    val solver = new sicfun.holdem.cfr.HoldemCfrSolver()
    val strategy = CfrVillainStrategy(solver)
    val gs = GameState(
      street = Street.Flop,
      board = Board.parse("Th 8d 5c").get,
      pot = 6.0,
      toCall = 3.0,
      position = Position.BigBlind,
      stackSize = 97.0,
      betHistory = Vector.empty
    )
    val candidates = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(12.0))
    // Medium-strength hand — equilibrium should mix between actions
    val hand = HoleCards.parse("9h9d").get
    val actions = (0 until 50).map(i =>
      strategy.decide(hand, gs, candidates, 0.60, new Random(i))
    )
    val distinct = actions.distinct
    // With a medium hand on a wet board, CFR should use at least 2 different actions
    assert(distinct.size >= 2,
      s"CfrVillainStrategy should mix actions for medium hands, got only: $distinct")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly sicfun.holdem.validation.VillainStrategyTest -- *Cfr*"`
Expected: FAIL — `CfrVillainStrategy` not found.

- [ ] **Step 3: Implement CfrVillainStrategy**

Add to `VillainStrategy.scala`:

```scala
import sicfun.holdem.cfr.{HoldemCfrConfig, HoldemCfrSolver}
import sicfun.holdem.equity.HoldemEquity

/** CFR equilibrium strategy — solves each decision point via HoldemCfrSolver.
  *
  * Computes a Nash equilibrium mixed strategy for the villain's spot, then
  * samples an action from it. Slower than EquityBasedStrategy but produces
  * actual equilibrium play, making it the correct GTO baseline for profiler
  * calibration.
  *
  * @param solver  shared HoldemCfrSolver instance (thread-safe, stateless)
  * @param config  CFR config — use reduced iterations/trials for simulation speed
  */
final class CfrVillainStrategy(
    solver: HoldemCfrSolver,
    config: HoldemCfrConfig = HoldemCfrConfig(
      iterations = 300,
      equityTrials = 500,
      maxVillainHands = 48,
      includeVillainReraises = false
    )
) extends VillainStrategy:

  def decide(
      hand: HoleCards,
      state: GameState,
      candidates: Vector[PokerAction],
      equityVsRandom: Double,
      rng: Random
  ): PokerAction =
    if candidates.size <= 1 then return candidates.headOption.getOrElse(PokerAction.Check)
    try
      // From the villain's perspective: "hero" = villain (decision-maker),
      // "villainPosterior" = opponent's (hero's) range = uniform excluding our cards + board
      val opponentRange = HoldemEquity.fullRange(hand, state.board)
      val policy = solver.solveDecisionPolicy(
        hero = hand,
        state = state,
        villainPosterior = opponentRange,
        candidateActions = candidates,
        config = config
      )
      sampleAction(policy.actionProbabilities, candidates, rng)
    catch
      // CFR can fail on degenerate spots (empty opponent range, etc.)
      // Fall back to simple equity-based decision
      case _: Exception =>
        EquityBasedStrategy().decide(hand, state, candidates, equityVsRandom, rng)

  /** Sample one action from the mixed strategy. */
  private def sampleAction(
      probs: Map[PokerAction, Double],
      candidates: Vector[PokerAction],
      rng: Random
  ): PokerAction =
    val roll = rng.nextDouble()
    var cumulative = 0.0
    // Iterate in candidate order for deterministic tie-breaking
    for action <- candidates do
      cumulative += probs.getOrElse(action, 0.0)
      if roll < cumulative then return action
    // Fallback: highest probability action
    probs.maxBy(_._2)._1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `sbt "testOnly sicfun.holdem.validation.VillainStrategyTest"`
Expected: PASS (all tests including CFR tests)

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/validation/VillainStrategy.scala \
        src/test/scala/sicfun/holdem/validation/VillainStrategyTest.scala
git commit -m "feat: add CfrVillainStrategy using HoldemCfrSolver for equilibrium decisions"
```

---

## Chunk 3: Calibration Test & Threshold Adjustment

### Task 4: Build CFR GTO calibration integration test

**Files:**
- Create: `src/test/scala/sicfun/holdem/validation/CfrGtoCalibrationTest.scala`

#### Context

This is the key test: run the CFR villain through the full simulation → export → parse → profile pipeline and check whether the profiler produces false positive hints. If it does, the thresholds are wrong.

Additionally, this test measures the CFR villain's actual rates for each profiled metric, providing calibration data for threshold adjustment.

**Performance budget:** 2000 hands × ~3 villain decisions × ~5ms/solve = ~30 seconds. Set munit timeout to 180s to be safe.

- [ ] **Step 1: Write the integration test**

```scala
// src/test/scala/sicfun/holdem/validation/CfrGtoCalibrationTest.scala
package sicfun.holdem.validation

import munit.FunSuite
import sicfun.holdem.cfr.{HoldemCfrConfig, HoldemCfrSolver}
import sicfun.holdem.history.{HandHistoryImport, HandHistorySite, OpponentProfile}
import sicfun.holdem.types.{PokerAction, Street}

/** Calibration test: a CFR equilibrium villain should produce ZERO false
  * positive exploit hints from the profiler.
  *
  * If this test fails, the profiler's hardcoded thresholds disagree with
  * actual equilibrium play and must be recalibrated.
  */
class CfrGtoCalibrationTest extends FunSuite:

  override val munitTimeout = scala.concurrent.duration.Duration(180, "s")

  test("CFR equilibrium villain produces no false positive leak hints"):
    val solver = new HoldemCfrSolver()
    val cfrStrategy = CfrVillainStrategy(solver)
    val villain = LeakInjectedVillain(
      name = "cfr_gto_control",
      leaks = Vector(NoLeak()),
      baselineNoise = 0.0,
      seed = 42L
    )
    val sim = new HeadsUpSimulator(
      heroEngine = None,
      villain = villain,
      seed = 42L,
      villainStrategy = cfrStrategy
    )

    val numHands = 2000
    val records = (1 to numHands).map(sim.playHand).toVector

    // Export, parse, profile
    val text = PokerStarsExporter.exportHands(records, "Hero", villain.name)
    val parsed = HandHistoryImport.parseText(text, Some(HandHistorySite.PokerStars), Some("Hero"))
    assert(parsed.isRight, s"parse failed: ${parsed.left.getOrElse("")}")
    val hands = parsed.toOption.get
    val profiles = OpponentProfile.fromImportedHands("simulated", hands, Set("Hero"))
    assert(profiles.nonEmpty, "no profile generated")
    val profile = profiles.head

    // === Calibration data: measure actual rates ===
    val events = profile.recentEvents
    val sig = profile.signature

    println(s"[CFR-CAL] Hands: $numHands, Events: ${events.size}")
    println(f"[CFR-CAL] Signature: fold=${sig.values(0)}%.3f raise=${sig.values(1)}%.3f " +
      f"call=${sig.values(2)}%.3f check=${sig.values(3)}%.3f")
    println(s"[CFR-CAL] Archetype: ${profile.archetypePosterior.mapEstimate}")

    // Per-metric calibration
    val riverFacingBet = events.filter(e => e.street == Street.River && e.toCall > 0)
    if riverFacingBet.size >= 5 then
      val riverFoldRate = riverFacingBet.count(_.action == PokerAction.Fold).toDouble / riverFacingBet.size
      println(f"[CFR-CAL] River fold rate (facing bet):  $riverFoldRate%.3f  (threshold: 0.30)")

    val facingLargeBet = events.filter { e =>
      e.toCall > 0 && e.potBefore > e.toCall && e.toCall / (e.potBefore - e.toCall) >= 0.6
    }
    if facingLargeBet.size >= 5 then
      val largeBetCallRate = facingLargeBet.count(_.action == PokerAction.Call).toDouble / facingLargeBet.size
      println(f"[CFR-CAL] Large bet call rate:           $largeBetCallRate%.3f  (threshold: 0.60)")

    val turnEvents = events.filter(_.street == Street.Turn)
    if turnEvents.size >= 5 then
      val turnRaiseRate = turnEvents.count(_.action.category == PokerAction.Category.Raise).toDouble / turnEvents.size
      println(f"[CFR-CAL] Turn raise rate:               $turnRaiseRate%.3f  (threshold: 0.25)")

    val bigPotCanBet = events.filter(e =>
      e.potBefore > 0 && e.stackBefore / e.potBefore < 4.0 && e.toCall == 0
    )
    if bigPotCanBet.size >= 5 then
      val bigPotCheckRate = bigPotCanBet.count(_.action == PokerAction.Check).toDouble / bigPotCanBet.size
      println(f"[CFR-CAL] Big pot check rate:            $bigPotCheckRate%.3f  (threshold: 0.75)")

    val preflopEvents = events.filter(_.street == Street.Preflop)
    if preflopEvents.size >= 8 then
      val preflopFoldRate = preflopEvents.count(_.action == PokerAction.Fold).toDouble / preflopEvents.size
      println(f"[CFR-CAL] Preflop fold rate:             $preflopFoldRate%.3f  (threshold: 0.20)")
      val preflopCallRate = preflopEvents.count(_.action == PokerAction.Call).toDouble / preflopEvents.size
      println(f"[CFR-CAL] Preflop call rate:             $preflopCallRate%.3f  (threshold: 0.65)")

    val responseTotal = profile.raiseResponses.total.toDouble
    if responseTotal >= 4.0 then
      val foldToRaise = profile.raiseResponses.folds / responseTotal
      val callVsRaise = profile.raiseResponses.calls / responseTotal
      val reraiseVsRaise = profile.raiseResponses.raises / responseTotal
      println(f"[CFR-CAL] Fold to raise:                 $foldToRaise%.3f  (threshold: 0.55)")
      println(f"[CFR-CAL] Call vs raise:                 $callVsRaise%.3f  (threshold: 0.55)")
      println(f"[CFR-CAL] Reraise vs raise:              $reraiseVsRaise%.3f  (threshold: 0.28)")

    // === False positive check ===
    val hints = profile.exploitHints
    println(s"[CFR-CAL] Exploit hints: $hints")

    val leakIds = Vector("overfold-river-aggression", "overcall-big-bets", "overbluff-turn-barrel",
      "passive-big-pots", "preflop-too-loose", "preflop-too-tight")
    val matches = leakIds.filter(id => hintMatchesLeak(hints, id))
    println(s"[CFR-CAL] False positive leak patterns: $matches")

    // THIS IS THE KEY ASSERTION:
    // If this fails, the thresholds in exploitHintsFor disagree with equilibrium.
    // Use the calibration data printed above to adjust thresholds.
    assert(matches.isEmpty,
      s"CFR equilibrium villain triggers false positive leak patterns: $matches\n" +
        s"  Hints: $hints\n" +
        s"  Use calibration data above to adjust thresholds in OpponentProfileStore.exploitHintsFor")

  private def hintMatchesLeak(hints: Vector[String], leakId: String): Boolean =
    leakId match
      case "overfold-river-aggression" =>
        hints.exists(_.contains("Over-folds on the river"))
      case "overcall-big-bets" =>
        hints.exists(h => h.contains("calling station") || h.contains("Calls too often facing large bets"))
      case "overbluff-turn-barrel" =>
        hints.exists(_.contains("Very aggressive on the turn"))
      case "passive-big-pots" =>
        hints.exists(_.contains("Plays passive in big pots"))
      case "preflop-too-loose" =>
        hints.exists(_.contains("Calls too loose preflop"))
      case "preflop-too-tight" =>
        hints.exists(_.contains("Over-folds preflop"))
      case _ => false
```

- [ ] **Step 2: Run test — expect it to EITHER pass or fail with calibration data**

Run: `sbt "testOnly sicfun.holdem.validation.CfrGtoCalibrationTest"`

Expected output includes calibration lines like:
```
[CFR-CAL] River fold rate (facing bet):  0.XXX  (threshold: 0.30)
[CFR-CAL] Large bet call rate:           0.XXX  (threshold: 0.60)
...
```

If the test **passes**: the current thresholds are already compatible with equilibrium. Proceed to commit.

If the test **fails**: read the calibration data and proceed to Task 5.

- [ ] **Step 3: Commit the test (whether it passes or fails)**

```bash
git add src/test/scala/sicfun/holdem/validation/CfrGtoCalibrationTest.scala
git commit -m "test: add CFR GTO calibration test for profiler false positive detection"
```

---

### Task 5: Calibrate thresholds from measured CFR rates

**Files:**
- Modify: `src/main/scala/sicfun/holdem/history/OpponentProfileStore.scala` (lines 313–395)

#### Context

This task is conditional — only execute if Task 4's test fails.

Use the calibration data from Task 4 to adjust the hardcoded thresholds in `exploitHintsFor`. The principle: each threshold should be set at `cfr_baseline_rate + margin`, where `margin` is chosen to avoid flagging equilibrium play while still catching real leaks.

**Approach:** For each metric, set the threshold to `max(current_threshold, cfr_rate + buffer)`:
- Binary thresholds (fold rate, call rate): buffer = 0.10 (10 percentage points above equilibrium)
- Aggression thresholds (raise rate): buffer = 0.08

- [ ] **Step 1: Read calibration output from Task 4**

Re-run if needed: `sbt "testOnly sicfun.holdem.validation.CfrGtoCalibrationTest" 2>&1 | tee calibration.log`

Record the CFR baseline rates for each metric. Example (actual values will differ):
```
River fold rate:  0.22 → threshold should be >= 0.32
Large bet call rate: 0.48 → threshold 0.60 is fine (0.48 + 0.10 = 0.58 < 0.60)
Turn raise rate: 0.18 → threshold 0.25 is fine
Preflop fold rate: 0.16 → threshold should be >= 0.26 (was 0.20)
...
```

- [ ] **Step 2: Adjust thresholds in OpponentProfileStore.exploitHintsFor**

Edit `src/main/scala/sicfun/holdem/history/OpponentProfileStore.scala`, method `exploitHintsFor` (lines 313–395).

For each threshold that the CFR villain's rate exceeds or comes within 0.05 of, raise the threshold:

| Metric | Old threshold | Rule |
|--------|---------------|------|
| River fold rate (line 323) | `>= 0.30` | Set to `max(0.30, cfr_rate + 0.10)` |
| Large bet call rate (line 332) | `>= 0.60` | Set to `max(0.60, cfr_rate + 0.10)` |
| Turn raise rate (line 339) | `>= 0.25` | Set to `max(0.25, cfr_rate + 0.08)` |
| Big pot check rate (line 348) | `>= 0.75` | Set to `max(0.75, cfr_rate + 0.10)` |
| Preflop fold rate (line 355) | `>= 0.20` | Set to `max(0.20, cfr_rate + 0.10)` |
| Preflop call rate (line 359) | `>= 0.65` | Set to `max(0.65, cfr_rate + 0.10)` |
| Fold to raise (line 369) | `>= 0.55` | Set to `max(0.55, cfr_rate + 0.10)` |
| Call vs raise (line 371) | `>= 0.55` | Set to `max(0.55, cfr_rate + 0.10)` |
| Reraise vs raise (line 373) | `>= 0.28` | Set to `max(0.28, cfr_rate + 0.08)` |
| Overall fold rate (line 381) | `>= 0.45` | Set to `max(0.45, cfr_rate + 0.10)` |

Add a comment block above the method documenting the calibration source:

```scala
  /** Derive exploit hints from profiled opponent behavior.
    *
    * Thresholds calibrated against HoldemCfrSolver equilibrium baseline
    * (2026-03-17, CfrGtoCalibrationTest, 2000 hands, CFR iterations=300).
    * Each threshold = CFR baseline rate + margin to avoid flagging GTO play.
    */
```

- [ ] **Step 3: Re-run calibration test to verify it passes**

Run: `sbt "testOnly sicfun.holdem.validation.CfrGtoCalibrationTest"`
Expected: PASS — no false positive hints for CFR villain.

- [ ] **Step 4: Run existing leak detection tests to verify real leaks still detected**

Run: `sbt "testOnly sicfun.holdem.validation.*"`

Pay special attention to:
- `InjectedLeakTest` — leak predicates still fire
- `HeadsUpSimulatorTest` — simulation still works
- `GtoBaselineFalsePositiveTest` — equity-based villain still passes (thresholds only went UP)
- `SpotContextTest` — spot classification unchanged

If any real leak detection tests fail (leak is no longer detected because the threshold went too high), the margin is too large for that metric. Reduce the buffer for that specific threshold.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/history/OpponentProfileStore.scala
git commit -m "fix: calibrate profiler thresholds from CFR equilibrium baseline"
```

---

## Execution Notes

### What success looks like
1. `CfrGtoCalibrationTest` passes — the CFR villain produces zero false positive hints
2. All existing leak detection tests still pass — real leaks are still caught
3. The profiler's thresholds are documented with their calibration source

### What to watch for
- **CFR solve failures:** The CFR solver may fail on degenerate spots (very few villain hands possible, trivially dominated positions). `CfrVillainStrategy` falls back to `EquityBasedStrategy` for those spots. If fallbacks are frequent (>10% of decisions), investigate — the game abstraction may be too narrow.
- **Performance:** If 2000 hands takes >3 minutes, reduce to 1000 hands or increase CFR config's `maxVillainHands` reduction. The calibration data will be noisier but directionally correct.
- **Threshold sensitivity:** If adjusting a threshold to avoid CFR false positives also kills detection of the corresponding real leak (e.g., raising preflop fold threshold to 0.30 means the PreflopTooTight leak at mild severity is no longer detected), that's information — it means the mild severity is within equilibrium noise and shouldn't be claimed as detectable.

### Future work (not in this plan)
- Cache CFR solutions for common spots to speed up repeated simulation
- Use the CFR villain as the baseline for `ValidationRunner.defaultPopulation` (replace equity-based GTO control)
- Derive per-archetype thresholds from CFR baseline + archetype deviation profiles
- Run calibration with `--slow-hero` (full RealTimeAdaptiveEngine) for a more realistic scenario
