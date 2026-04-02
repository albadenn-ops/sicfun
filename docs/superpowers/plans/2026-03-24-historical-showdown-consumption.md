# Historical Showdown Data Consumption Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the existing `OpponentProfile.showdownHands` data (already parsed, stored, persisted, and serialized) into two downstream consumers: (A) showdown-aware exploit hints in the profiling pipeline, and (B) a soft prior bias in the range inference engine. Prove both with tests in the validation harness.

**Background:** The showdown integration plan (2026-03-17) delivered parsing, storage, and current-hand consumption. But historical showdown records sit in `OpponentProfile.showdownHands: Vector[ShowdownRecord]` without being consumed for profiling or future-hand range inference. This plan closes that gap.

---

## Context for the Implementing Agent

### What Already Works

| Component | Status | Key File |
|-----------|--------|----------|
| Parse showdown cards from hand histories | ✓ | `HandHistoryImport.scala:291-327` — parses `*** SHOW DOWN ***` section |
| Store in profiles | ✓ | `OpponentProfileStore.scala:88` — `showdownHands: Vector[ShowdownRecord]` |
| Persist/serialize JSON | ✓ | `OpponentProfileStore.scala:944,981` — read/write in profile JSON |
| Merge across sessions | ✓ | `OpponentProfileStore.scala:197` — `distinctBy(_.handId)` |
| Current-hand delta posterior | ✓ | `RangeInferenceEngine.scala:200` — `revealedCards` creates hard constraint |
| Live advisor `showdown` command | ✓ | `AdvisorSession.scala:422,518` — stores and passes to engine |
| Validation pipeline exports showdowns | ✓ | `PokerStarsExporter.scala:60-63` — emits `*** SHOW DOWN ***` |

### What Does NOT Work (The Gap)

1. `exploitHintsFor` in `OpponentProfileStore.scala:332` never reads `showdownHands`
2. `RangeInferenceEngine.inferPosterior` has no parameter for historical showdown bias
3. `RealTimeAdaptiveEngine.decide` never passes showdown history to the range engine
4. The validation pipeline never asserts showdown data consumption

### Key Types

```scala
// OpponentProfileStore.scala:72
final case class ShowdownRecord(handId: String, cards: HoleCards)

// OpponentProfileStore.scala:78-92 (fields relevant to this plan)
final case class OpponentProfile(
    ...
    recentEvents: Vector[PokerEvent],
    showdownHands: Vector[ShowdownRecord] = Vector.empty,
    ...
):
  def exploitHints: Vector[String] = OpponentProfile.exploitHintsFor(this)

// RangeInferenceEngine.scala:187-199
def inferPosterior(
    hero: HoleCards, board: Board, folds: Vector[PreflopFold],
    tableRanges: TableRanges, villainPos: Position,
    observations: Seq[VillainObservation], actionModel: PokerActionModel,
    bunchingTrials: Int = 10_000, rng: Random = new Random(),
    useCache: Boolean = true, revealedCards: Option[HoleCards] = None
): PosteriorInferenceResult

// RangeInferenceEngine.scala:256-268 (prior construction)
private def computePosterior(...):
    val normalizedPrior = priorForContext(...)  // builds from TableRanges + bunching
    // Bayesian update follows via HoldemBayesProvider.updatePosterior
```

### Architecture Constraints

- The prior (`normalizedPrior`) comes from `priorForContext()` which returns a `DiscreteDistribution[HoleCards]` — a probability map over all 1326 hole card combos. This is where showdown bias should be applied.
- `DiscreteDistribution` supports pointwise multiplication: `prior.reweight(hand => biasFor(hand))` — if this method exists; otherwise construct a new distribution by multiplying weights and renormalizing.
- The exploit hints system returns `Vector[String]` from `exploitHintsFor(profile)` — just add new hints based on showdown analysis.
- Profiler thresholds were just recalibrated against CFR equilibrium (river fold threshold raised from 0.30 to 0.45). Any new hints must not trigger on the GTO baseline.
- **Test framework**: munit 1.2.2 (`extends FunSuite`, `test("name"): ...`, `assertEquals`, `assert`)
- **Scala 3.8.1** with indentation syntax, `-Werror` enabled

### Codebase Conventions

- Prefer `final case class` for data, `object` for stateless logic
- `require(...)` for invariants in case classes
- Tests mirror source structure in `src/test/scala/`
- Use existing `HoleCards`, `Card`, `Rank`, `Suit` types — never raw strings
- Keep changes focused; don't refactor unrelated code

---

## Chunk 1: Showdown-Aware Exploit Hints (Part A)

### Task 1: Analyze showdown hand tendencies and generate hints

**Files:**
- Modify: `src/main/scala/sicfun/holdem/history/OpponentProfileStore.scala` — `exploitHintsFor` method (line 332)
- Test: `src/test/scala/sicfun/holdem/history/OpponentProfileStoreTest.scala` — new tests

**Design:** Add a new section at the end of `exploitHintsFor` (before the `distinct` and `take(5)` at the end) that reads `profile.showdownHands` and generates hints based on observed patterns.

Showdown analysis logic:

```
Given: showdownHands: Vector[ShowdownRecord], each with cards: HoleCards

1. Require minimum 3 showdowns to avoid noise (if < 3, skip)
2. Classify each shown hand by strength category:
   - Premium pairs: AA, KK, QQ, JJ (use Rank.ordinal >= Jack)
   - Broadway: both cards rank >= Ten
   - Suited connectors: suited + ranks within 2
   - Speculative: small pairs, suited aces, etc.
   - Weak: everything else
3. Compute distribution across categories
4. Generate hints:
   - If premium_count / total >= 0.5 and total >= 3:
     "Shows down premium hands frequently; likely playing a tight value-heavy range."
   - If weak_count / total >= 0.4 and total >= 3:
     "Has shown down weak hands; likely stations or bluffs reaching showdown — value bet thinner."
   - If total >= 5 and all shown hands are pairs:
     "Showdown history is pair-heavy; range may be set-mining oriented."
```

- [ ] **Step 1: Write failing test** — Create a test that builds an `OpponentProfile` with 4+ showdown records (mix of premium and weak), calls `exploitHints`, and asserts showdown-based hints appear.

- [ ] **Step 2: Implement showdown hint generation** — Add a `private def showdownHintsFor(showdowns: Vector[ShowdownRecord]): Vector[String]` helper inside the `OpponentProfile` companion object, called from `exploitHintsFor`.

- [ ] **Step 3: Run test and verify it passes**

- [ ] **Step 4: Run GTO false positive tests** — Run `sbt "testOnly sicfun.holdem.validation.GtoBaselineFalsePositiveTest"` to confirm zero false positives. The GTO baseline has showdowns but should have a balanced mix of hand categories.

### Task 2: Add showdown hint coverage in validation pipeline

**Files:**
- Modify: `src/main/scala/sicfun/holdem/validation/ValidationRunner.scala` — `runOnePlayer` method, showdown tracking
- Test: new test or extend `FastHeroLeakDetectionTest`

**Design:** The validation pipeline already exports showdowns (PokerStarsExporter emits `*** SHOW DOWN ***`), and `fromImportedHands` already populates `showdownHands`. Verify that the new hints appear for leaky players but NOT for the GTO control.

- [ ] **Step 5: Add a test** that simulates a leak player (e.g., `Overcalls(0.9)` — a calling station), exports → imports → profiles, and asserts that showdown-based hints reflect the calling station's tendency to show down weak hands.

- [ ] **Step 6: Verify GTO canary** — Assert the GTO baseline player does NOT trigger any showdown-based exploit hints (balanced showdown distribution).

---

## Chunk 2: Range Prior Adjustment (Part B)

### Task 3: Create showdown-based prior bias function

**Files:**
- Create: `src/main/scala/sicfun/holdem/engine/ShowdownPriorBias.scala`
- Test: `src/test/scala/sicfun/holdem/engine/ShowdownPriorBiasTest.scala`

**Design:** A pure function that takes a prior `DiscreteDistribution[HoleCards]` and a `Vector[ShowdownRecord]`, and returns a mildly reweighted prior. This is a "soft" update — NOT a delta posterior. The intuition: if an opponent has shown QQ twice and KK once in 5 showdowns, we slightly increase the probability of premium pairs.

```scala
object ShowdownPriorBias:
  /** Minimum showdowns required before applying any bias. */
  val MinShowdowns = 3

  /** Maximum bias multiplier per hand. Keeps bias mild. */
  val MaxBiasMultiplier = 1.5

  /** Blend weight: how much of the biased prior to mix in (0 = no bias, 1 = full bias).
    * Low values prevent overfitting to small samples.
    */
  def blendWeight(showdownCount: Int): Double =
    // Sigmoid-like: grows slowly from 0 toward 0.15 max
    val w = math.min(showdownCount.toDouble / 50.0, 0.15)
    w

  def applyBias(
      prior: DiscreteDistribution[HoleCards],
      showdowns: Vector[ShowdownRecord],
      deadCards: Set[Card] = Set.empty
  ): DiscreteDistribution[HoleCards] =
    if showdowns.size < MinShowdowns then return prior
    // 1. Build empirical frequency over shown hands
    // 2. Compute a similarity kernel: each shown hand boosts similar hands
    //    (same rank class, e.g., QQ boosts all pocket pairs slightly)
    // 3. Reweight: biased(hand) = prior(hand) * (1 + alpha * similarity(hand))
    // 4. Blend: final = (1 - w) * prior + w * biased, then renormalize
    // 5. Remove dead cards (hero cards + board)
```

**Similarity kernel:** For each shown hand, boost:
- Exact match: +1.0 weight
- Same pair rank (if pair): +0.3
- Same rank class (premium/broadway/medium pair): +0.1
- All other hands: +0.0

This is a smoothed kernel, not a point estimate. It prevents overfitting to exact combos.

- [ ] **Step 7: Write failing test** — Test that `applyBias` with 5 showdowns of premium pairs slightly increases premium pair probability vs the original prior, while keeping total probability = 1.0.

- [ ] **Step 8: Implement `ShowdownPriorBias`** — Pure function, no state, no side effects.

- [ ] **Step 9: Test edge cases** — Empty showdowns returns original prior. 1-2 showdowns returns original prior. Dead card removal works.

- [ ] **Step 10: Test blend weight curve** — Verify `blendWeight(3) ≈ 0.06`, `blendWeight(10) ≈ 0.15 capped`, `blendWeight(50) = 0.15`.

### Task 4: Wire bias into RangeInferenceEngine

**Files:**
- Modify: `src/main/scala/sicfun/holdem/engine/RangeInferenceEngine.scala` — `inferPosterior` and `computePosterior`
- Test: `src/test/scala/sicfun/holdem/engine/RangeInferenceEngineTest.scala`

**Design:** Add an optional `showdownHistory: Vector[ShowdownRecord] = Vector.empty` parameter to `inferPosterior`. In `computePosterior`, after building `normalizedPrior` from `priorForContext`, apply `ShowdownPriorBias.applyBias(normalizedPrior, showdownHistory, deadCards)`.

- [ ] **Step 11: Add parameter to `inferPosterior`** — `showdownHistory: Vector[ShowdownRecord] = Vector.empty`. Default empty = no behavioral change for existing callers.

- [ ] **Step 12: Apply bias in `computePosterior`** — After `priorForContext` returns, before Bayesian action update:
```scala
val biasedPrior = ShowdownPriorBias.applyBias(normalizedPrior, showdownHistory, heroDeadCards)
```
Then use `biasedPrior` instead of `normalizedPrior` for the rest of the function.

- [ ] **Step 13: Write integration test** — Call `inferPosterior` with 5 premium pair showdowns and verify the posterior has slightly higher probability for premium pairs vs calling with empty showdowns.

- [ ] **Step 14: Verify cache key includes showdown hash** — The `PosteriorCacheKey` must include showdown history (or its hash) so different showdown states don't return stale cached results.

### Task 5: Wire into RealTimeAdaptiveEngine and AdvisorSession

**Files:**
- Modify: `src/main/scala/sicfun/holdem/engine/RealTimeAdaptiveEngine.scala` — `decide` method
- Modify: `src/main/scala/sicfun/holdem/runtime/AdvisorSession.scala` — advise/review commands
- Test: extend existing tests

**Design:** Add `showdownHistory: Vector[ShowdownRecord] = Vector.empty` parameter to `RealTimeAdaptiveEngine.decide()`. In `AdvisorSession`, when an opponent profile is loaded, pass `profile.showdownHands` through to the engine.

- [ ] **Step 15: Add parameter to `decide()`** — Default empty for backward compatibility.

- [ ] **Step 16: Pass through to `inferPosterior`** — In both the `revealedCards` match arm (for current-hand showdown) and the `None` arm (normal inference), pass `showdownHistory` through.

- [ ] **Step 17: Wire in AdvisorSession** — When calling `engine.decide()` in the advise/review commands (lines 510-520, 540-550), pass the opponent's showdown history from the loaded profile.

  **Note:** `AdvisorSession` currently stores `OpponentProfile` (or may need to be extended to hold one). Check `AdvisorSession.scala` for the profile storage mechanism. If no profile is loaded, pass empty showdown history.

- [ ] **Step 18: Test end-to-end** — Extend `AdvisorSessionTest` to verify that loading a profile with showdown history slightly changes the decision output (posterior should differ from no-showdown case).

---

## Chunk 3: Validation Integration (Part C)

### Task 6: Prove showdown consumption in the validation pipeline

**Files:**
- Create: `src/test/scala/sicfun/holdem/validation/ShowdownConsumptionTest.scala`

**Design:** End-to-end test proving that:
1. A leak player (e.g., Overcalls — shows down weak hands) generates showdown-aware exploit hints
2. The GTO control player does NOT generate false positive showdown hints
3. The range prior is measurably different when showdown history is present vs absent

- [ ] **Step 19: Test leak player showdown hints**
```
Simulate 3000 hands of Overcalls(0.9) → export → import → profile
Assert: profile.showdownHands.nonEmpty
Assert: profile.exploitHints contains showdown-based hint about weak hands
```

- [ ] **Step 20: Test GTO canary showdown hints**
```
Simulate 2000 hands of GTO baseline → export → import → profile
Assert: profile.showdownHands.nonEmpty (showdowns happen)
Assert: NO showdown-based exploit hint triggers (balanced distribution)
```

- [ ] **Step 21: Test prior bias effect**
```
Build a profile with 10 premium pair showdowns
Call inferPosterior twice: once with empty showdowns, once with the showdown history
Assert: posterior with showdowns has higher probability for premium pairs
Assert: difference is small (< 15% relative change — mild bias)
```

---

## File Change Summary

| Action | File | What Changes |
|--------|------|-------------|
| Modify | `src/main/scala/sicfun/holdem/history/OpponentProfileStore.scala` | Add `showdownHintsFor` helper, call from `exploitHintsFor` |
| Create | `src/main/scala/sicfun/holdem/engine/ShowdownPriorBias.scala` | Pure function: showdown history → prior reweighting |
| Modify | `src/main/scala/sicfun/holdem/engine/RangeInferenceEngine.scala` | Add `showdownHistory` param, apply bias in `computePosterior` |
| Modify | `src/main/scala/sicfun/holdem/engine/RealTimeAdaptiveEngine.scala` | Pass `showdownHistory` through `decide()` |
| Modify | `src/main/scala/sicfun/holdem/runtime/AdvisorSession.scala` | Wire profile showdown data into engine calls |
| Create | `src/test/scala/sicfun/holdem/engine/ShowdownPriorBiasTest.scala` | Unit tests for bias function |
| Create | `src/test/scala/sicfun/holdem/validation/ShowdownConsumptionTest.scala` | End-to-end validation tests |
| Modify | `src/test/scala/sicfun/holdem/history/OpponentProfileStoreTest.scala` | Tests for showdown hint generation |

## Risk Mitigations

1. **Overfitting to small samples**: `MinShowdowns = 3`, blend weight caps at 0.15, similarity kernel smooths across hand categories
2. **False positives on GTO player**: Run `GtoBaselineFalsePositiveTest` after each chunk; GTO player shows balanced hand distribution so showdown hints shouldn't fire
3. **Cache invalidation**: Include showdown history hash in `PosteriorCacheKey` and `InferenceCacheKey`
4. **Performance**: Showdown bias is O(n) over 1326 hands × showdown count — negligible vs CFR solving
5. **Backward compatibility**: All new parameters default to empty/zero, preserving existing behavior

## Verification Commands

```bash
# After Chunk 1 (exploit hints):
sbt "testOnly sicfun.holdem.history.OpponentProfileStoreTest"
sbt "testOnly sicfun.holdem.validation.GtoBaselineFalsePositiveTest"

# After Chunk 2 (range prior):
sbt "testOnly sicfun.holdem.engine.ShowdownPriorBiasTest"
sbt "testOnly sicfun.holdem.engine.RangeInferenceEngineTest"

# After Chunk 3 (validation):
sbt "testOnly sicfun.holdem.validation.ShowdownConsumptionTest"
sbt "testOnly sicfun.holdem.validation.*"  # full suite, expect 87+ all passing
```
