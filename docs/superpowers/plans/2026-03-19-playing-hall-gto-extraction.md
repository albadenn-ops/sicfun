# PlayingHall GTO Extraction (Plan 3 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose the 2633-line TexasHoldemPlayingHall god object by extracting ~575 lines of GTO decision infrastructure, hand strength heuristics, and archetype villain logic into three focused modules.

**Architecture:** Extract three cohesive modules from PlayingHall: `HandStrengthEstimator` (pure hand evaluation heuristics shared by both GTO and archetype paths), `GtoSolveEngine` (unified GTO decision system with CFR caching, fast heuristic path, and canonical board normalization), and `ArchetypeVillainResponder` (archetype-based villain decisions with style profiles). PlayingHall's `decideHero` and `decideVillain` delegate to these modules, keeping hand play orchestration, IO, multiway integration, learning, and CLI parsing. The near-identical `gtoHeroResponds`/`gtoVillainResponds` methods unify into a single `gtoResponds` with an `opponentPosterior` parameter — the only difference was which position's range was used, and both `villainPosteriorForHeroGto`/`heroPosteriorForGto` just called `tableRanges.rangeFor(position)`.

**Tech Stack:** Scala 3.8.1, munit 1.2.2, SBT

---

## Scope

### This Plan

Targets the PlayingHall god object decomposition (P0 from the 3-plan series):
- **2x `gtoHeroResponds`/`gtoVillainResponds`** → unified `GtoSolveEngine.gtoResponds` (only difference was posterior source — both just `tableRanges.rangeFor(position)`)
- **`solveGtoByCfr` + cache types + canonical signatures + sampling + parametrization** → `GtoSolveEngine`
- **`fastGtoResponds` + threshold helpers** → `GtoSolveEngine`
- **`preflopStrength`, `bestCategoryStrength`, `drawPotential`, `streetStrength`, `fastGtoStrength`, `hasTightRun`** → `HandStrengthEstimator`
- **`villainResponds` + `styleProfile` + `VillainStyleProfile`** → `ArchetypeVillainResponder`
- **`clamp`** → `HandStrengthEstimator` (no remaining callers in PlayingHall after extraction)
- **`GtoSolveCacheKey`, `GtoCachedPolicy`, `GtoCacheStats`, `MaxExactGtoCacheEntries`, `SuitPermutations`** → `GtoSolveEngine`

### Not in Scope

- Stretch goals from Plan 1 (renderAction in 7 more files)
- PlayingHall's `heroCandidates` (different from `HeroDecisionPipeline.heroCandidates` — uses stack-depth checks with fixed raise sizes, not computed raise sizing)
- PlayingHall's `buildEngine` (different formula from `HeroDecisionPipeline.newAdaptiveEngine` — uses `max(1, configured/30)` vs `max(8, min(n, n/10))`)
- Multiway integration (`multiwayRecommendationFor` and related)
- Hand play orchestration (`HandResolver`, betting rounds, showdowns)
- IO, learning, CLI parsing

### Behavioral Notes

1. **Unified `gtoResponds`:** The only difference between `gtoHeroResponds` and `gtoVillainResponds` was the posterior source. `villainPosteriorForHeroGto(hero, board, tableRanges, villainPosition)` and `heroPosteriorForGto(villain, board, tableRanges, heroPosition)` both just call `tableRanges.rangeFor(position)` — the `board` parameter was unused. After unification, the caller passes `tableRanges.rangeFor(opponentPosition)` directly. No behavioral change.

2. **Candidates passed to `gtoResponds`:** Previously `gtoHeroResponds`/`gtoVillainResponds` computed candidates internally via `heroCandidates(state, raiseSize, allowRaise)`. The caller (`decideHero`/`decideVillain`) also computed the same candidates for `multiwayRecommendationFor`. After extraction, `gtoResponds` takes pre-computed candidates as a parameter, eliminating the redundant computation. No behavioral change — same candidates, same results.

3. **`clamp` deletion from PlayingHall:** All 11 usages of `clamp` in PlayingHall are in methods being extracted. After extraction, `clamp` has no remaining callers in PlayingHall and can be deleted. `HandStrengthEstimator` defines its own `clamp`.

## File Map

### New Files

| File | Responsibility |
|---|---|
| `src/main/scala/sicfun/holdem/engine/HandStrengthEstimator.scala` | Pure hand evaluation: `preflopStrength`, `bestCategoryStrength`, `drawPotential`, `hasTightRun`, `streetStrength`, `fastGtoStrength`, `clamp` |
| `src/main/scala/sicfun/holdem/engine/GtoSolveEngine.scala` | Unified GTO decision system: `GtoMode` enum, cache types, `gtoResponds` dispatcher, `solveGtoByCfr` with caching, `fastGtoResponds` with threshold heuristics, canonical board signatures, action hashing, policy sampling, CFR parametrization |
| `src/main/scala/sicfun/holdem/engine/ArchetypeVillainResponder.scala` | Archetype-based villain decisions: `VillainStyleProfile`, `styleProfile`, `villainResponds` |
| `src/test/scala/sicfun/holdem/engine/HandStrengthEstimatorTest.scala` | Unit tests for hand evaluation heuristics |
| `src/test/scala/sicfun/holdem/engine/GtoSolveEngineTest.scala` | Unit tests for canonical signatures, sampling, parametrization, fast GTO |
| `src/test/scala/sicfun/holdem/engine/ArchetypeVillainResponderTest.scala` | Unit tests for style profiles and villain responses |

### Modified Files

| File | Changes |
|---|---|
| `src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala` | Delete ~575 lines of extracted methods/types. Add imports for `GtoSolveEngine.*` types. Update `decideHero` to call `GtoSolveEngine.gtoResponds`. Update `decideVillain` to call `GtoSolveEngine.gtoResponds` and `ArchetypeVillainResponder.villainResponds`. Delete `clamp`, `VillainStyleProfile`, cache types, `SuitPermutations`, `MaxExactGtoCacheEntries`. Keep `heroCandidates`, `normalizeAction`, `buildEngine`, `formatLongCountMap`, hand play, IO, CLI. |

### Dependency Order

```
Task 1 (HandStrengthEstimator) ──┬──→ Task 2 (GtoSolveEngine) ──┐
                                 └──→ Task 3 (VillainResponder) ─┤
                                                                 └──→ Task 4 (Refactor PlayingHall) ──→ Task 5 (Verify)
```

Tasks 2 and 3 are independent of each other. Both depend on Task 1.

---

## Task 1: Extract HandStrengthEstimator

**Files:**
- Create: `src/main/scala/sicfun/holdem/engine/HandStrengthEstimator.scala`
- Create: `src/test/scala/sicfun/holdem/engine/HandStrengthEstimatorTest.scala`

- [ ] **Step 1: Write failing tests**

```scala
package sicfun.holdem.engine

import munit.FunSuite
import sicfun.core.Card
import sicfun.holdem.types.*

class HandStrengthEstimatorTest extends FunSuite:

  private def card(token: String): Card =
    Card.parse(token).getOrElse(throw new IllegalArgumentException(s"bad card: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  private def board(tokens: String*): Board =
    Board.from(tokens.map(card).toVector)

  test("preflopStrength: pocket aces is high"):
    val strength = HandStrengthEstimator.preflopStrength(hole("As", "Ah"))
    assert(strength > 0.85, s"AA strength $strength should be > 0.85")

  test("preflopStrength: 72o is low"):
    val strength = HandStrengthEstimator.preflopStrength(hole("7d", "2c"))
    assert(strength < 0.40, s"72o strength $strength should be < 0.40")

  test("preflopStrength: suited connectors get bonuses"):
    val suited = HandStrengthEstimator.preflopStrength(hole("Ts", "9s"))
    val offsuit = HandStrengthEstimator.preflopStrength(hole("Td", "9c"))
    assert(suited > offsuit, s"suited $suited should be > offsuit $offsuit")

  test("bestCategoryStrength: flush on board is high"):
    val h = hole("As", "Ks")
    val b = board("Qs", "Js", "3s")
    val strength = HandStrengthEstimator.bestCategoryStrength(h, b)
    assert(strength > 0.5, s"flush strength $strength should be > 0.5")

  test("drawPotential: 4-flush has bonus"):
    val h = hole("As", "Ks")
    val b = board("Qs", "Jd", "3s")
    val potential = HandStrengthEstimator.drawPotential(h, b)
    assert(potential >= 0.08, s"4-flush potential $potential should be >= 0.08")

  test("drawPotential: no draws returns near zero"):
    val h = hole("2c", "7d")
    val b = board("Ah", "Ks", "9h")
    val potential = HandStrengthEstimator.drawPotential(h, b)
    assert(potential < 0.05, s"no-draw potential $potential should be < 0.05")

  test("hasTightRun: connected ranks"):
    assert(HandStrengthEstimator.hasTightRun(Seq(5, 6, 7, 8)))
    assert(HandStrengthEstimator.hasTightRun(Seq(3, 5, 6, 7, 8)))
    assert(!HandStrengthEstimator.hasTightRun(Seq(2, 5, 9, 13)))

  test("hasTightRun: wheel ace"):
    assert(HandStrengthEstimator.hasTightRun(Seq(2, 3, 4, 14)))

  test("clamp: within bounds"):
    assertEquals(HandStrengthEstimator.clamp(0.5), 0.5)
    assertEquals(HandStrengthEstimator.clamp(-0.1), 0.0)
    assertEquals(HandStrengthEstimator.clamp(1.5), 1.0)
    assertEquals(HandStrengthEstimator.clamp(5.0, 2.0, 8.0), 5.0)
    assertEquals(HandStrengthEstimator.clamp(1.0, 2.0, 8.0), 2.0)
    assertEquals(HandStrengthEstimator.clamp(9.0, 2.0, 8.0), 8.0)

  test("fastGtoStrength: preflop delegates to preflopStrength"):
    val h = hole("As", "Ah")
    val preflop = HandStrengthEstimator.preflopStrength(h)
    val fast = HandStrengthEstimator.fastGtoStrength(h, Board.empty, Street.Preflop)
    assertEquals(fast, preflop)

  test("streetStrength: postflop incorporates board"):
    val h = hole("As", "Ah")
    val b = board("Ac", "Kd", "2h")
    val rng = new scala.util.Random(42)
    val strength = HandStrengthEstimator.streetStrength(h, b, Street.Flop, rng)
    assert(strength > 0.5, s"trips on flop strength $strength should be > 0.5")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `sbt "testOnly sicfun.holdem.engine.HandStrengthEstimatorTest"`
Expected: FAIL — `HandStrengthEstimator` does not exist.

- [ ] **Step 3: Implement HandStrengthEstimator**

```scala
package sicfun.holdem.engine

import sicfun.core.{Card, HandEvaluator}
import sicfun.holdem.types.*
import sicfun.holdem.equity.HoldemCombinator

import scala.util.Random

/** Pure hand evaluation heuristics for poker decision engines.
  *
  * Extracted from TexasHoldemPlayingHall where these functions were shared
  * between the fast GTO path (fastGtoStrength) and archetype villain
  * decisions (streetStrength).
  */
private[holdem] object HandStrengthEstimator:

  def clamp(value: Double, lo: Double = 0.0, hi: Double = 1.0): Double =
    math.max(lo, math.min(hi, value))

  def preflopStrength(hand: HoleCards): Double =
    val r1 = hand.first.rank.value
    val r2 = hand.second.rank.value
    val high = math.max(r1, r2).toDouble / 14.0
    val low = math.min(r1, r2).toDouble / 14.0
    val pairBonus = if r1 == r2 then 0.30 + (high * 0.20) else 0.0
    val suitedBonus = if hand.first.suit == hand.second.suit then 0.06 else 0.0
    val gap = math.abs(r1 - r2)
    val connectorBonus =
      if gap == 0 then 0.0
      else if gap == 1 then 0.08
      else if gap == 2 then 0.04
      else 0.0
    clamp((0.45 * high) + (0.18 * low) + pairBonus + suitedBonus + connectorBonus)

  def bestCategoryStrength(hand: HoleCards, board: Board): Double =
    val cards = hand.toVector ++ board.cards
    cards.length match
      case 5 =>
        HandEvaluator.evaluate5Cached(cards).category.strength.toDouble / 8.0
      case 6 =>
        HoldemCombinator.combinations(cards.toIndexedSeq, 5).map { combo =>
          HandEvaluator.evaluate5Cached(combo).category.strength.toDouble / 8.0
        }.max
      case 7 =>
        HandEvaluator.evaluate7Cached(cards).category.strength.toDouble / 8.0
      case _ =>
        preflopStrength(hand)

  def drawPotential(hand: HoleCards, board: Board): Double =
    if board.cards.isEmpty then 0.0
    else
      val all = hand.toVector ++ board.cards
      val bySuit = all.groupBy(_.suit).view.mapValues(_.size).toMap
      val maxSuit = bySuit.values.max
      val flushDrawBonus =
        if maxSuit >= 5 then 0.12
        else if maxSuit == 4 then 0.08
        else if maxSuit == 3 && board.size <= 3 then 0.03
        else 0.0

      val ranks = all.map(_.rank.value).distinct.sorted
      val straightDrawBonus =
        if ranks.length >= 4 && hasTightRun(ranks) then 0.05
        else 0.0

      val pairWithBoardBonus =
        if board.cards.exists(card => card.rank == hand.first.rank || card.rank == hand.second.rank) then 0.04
        else 0.0

      flushDrawBonus + straightDrawBonus + pairWithBoardBonus

  def hasTightRun(sortedRanks: Seq[Int]): Boolean =
    if sortedRanks.length < 4 then false
    else
      val span4 = HoldemCombinator.combinations(sortedRanks.toIndexedSeq, 4).exists { combo =>
        combo.last - combo.head <= 4
      }
      val withWheelAce =
        if sortedRanks.contains(14) then
          val lowAce = sortedRanks.map(r => if r == 14 then 1 else r).sorted
          HoldemCombinator.combinations(lowAce.toIndexedSeq, 4).exists { combo =>
            combo.last - combo.head <= 4
          }
        else false
      span4 || withWheelAce

  /** Noisy hand strength for archetype villain decisions. Adds RNG jitter. */
  def streetStrength(
      hand: HoleCards,
      board: Board,
      street: Street,
      rng: Random
  ): Double =
    val pre = preflopStrength(hand)
    if street == Street.Preflop || board.cards.isEmpty then
      clamp(pre + (rng.nextDouble() - 0.5) * 0.04)
    else
      val categoryScore = bestCategoryStrength(hand, board)
      val drawBonus = drawPotential(hand, board)
      val noise = (rng.nextDouble() - 0.5) * 0.05
      clamp(0.45 * pre + 0.45 * categoryScore + drawBonus + noise)

  /** Deterministic hand strength for fast GTO heuristic path. No RNG noise. */
  def fastGtoStrength(hand: HoleCards, board: Board, street: Street): Double =
    val pre = preflopStrength(hand)
    if street == Street.Preflop || board.cards.isEmpty then pre
    else
      val madeWeight =
        street match
          case Street.Flop  => 0.50
          case Street.Turn  => 0.56
          case Street.River => 0.62
          case _            => 0.50
      val categoryScore = bestCategoryStrength(hand, board)
      val drawBonus = drawPotential(hand, board)
      clamp(((1.0 - madeWeight) * pre) + (madeWeight * categoryScore) + drawBonus)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `sbt "testOnly sicfun.holdem.engine.HandStrengthEstimatorTest"`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/engine/HandStrengthEstimator.scala \
  src/test/scala/sicfun/holdem/engine/HandStrengthEstimatorTest.scala
git commit -m "refactor: extract HandStrengthEstimator from PlayingHall (preflopStrength, drawPotential, etc.)"
```

---

## Task 2: Extract GtoSolveEngine

**Files:**
- Create: `src/main/scala/sicfun/holdem/engine/GtoSolveEngine.scala`
- Create: `src/test/scala/sicfun/holdem/engine/GtoSolveEngineTest.scala`

**Depends on:** Task 1 (HandStrengthEstimator.fastGtoStrength, clamp)

- [ ] **Step 1: Write failing tests**

```scala
package sicfun.holdem.engine

import munit.FunSuite
import sicfun.core.Card
import sicfun.holdem.types.*

class GtoSolveEngineTest extends FunSuite:

  private def card(token: String): Card =
    Card.parse(token).getOrElse(throw new IllegalArgumentException(s"bad card: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  private def board(tokens: String*): Board =
    Board.from(tokens.map(card).toVector)

  // --- canonicalHeroBoardSignature tests ---

  test("canonicalHeroBoardSignature: suit-invariant for preflop"):
    val sig1 = GtoSolveEngine.canonicalHeroBoardSignature(hole("As", "Ks"), Board.empty)
    val sig2 = GtoSolveEngine.canonicalHeroBoardSignature(hole("Ah", "Kh"), Board.empty)
    assertEquals(sig1, sig2)

  test("canonicalHeroBoardSignature: suit-invariant for flop"):
    val sig1 = GtoSolveEngine.canonicalHeroBoardSignature(hole("As", "Ks"), board("Qs", "Jd", "3h"))
    val sig2 = GtoSolveEngine.canonicalHeroBoardSignature(hole("Ah", "Kh"), board("Qh", "Jc", "3s"))
    assertEquals(sig1, sig2)

  test("canonicalHeroBoardSignature: different hands produce different signatures"):
    val sig1 = GtoSolveEngine.canonicalHeroBoardSignature(hole("As", "Ks"), Board.empty)
    val sig2 = GtoSolveEngine.canonicalHeroBoardSignature(hole("As", "Qs"), Board.empty)
    assertNotEquals(sig1, sig2)

  // --- orderedPositiveProbabilities tests ---

  test("orderedPositiveProbabilities: filters zero and negative"):
    val actions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Check)
    val probs = Map[PokerAction, Double](
      PokerAction.Fold -> 0.3,
      PokerAction.Call -> 0.0,
      PokerAction.Check -> -0.1
    )
    val result = GtoSolveEngine.orderedPositiveProbabilities(actions, probs)
    assertEquals(result.length, 1)
    assertEquals(result.head._1, PokerAction.Fold)

  test("orderedPositiveProbabilities: preserves order"):
    val actions = Vector(PokerAction.Check, PokerAction.Raise(2.0))
    val probs = Map[PokerAction, Double](
      PokerAction.Check -> 0.6,
      PokerAction.Raise(2.0) -> 0.4
    )
    val result = GtoSolveEngine.orderedPositiveProbabilities(actions, probs)
    assertEquals(result.length, 2)
    assertEquals(result(0)._1, PokerAction.Check)
    assertEquals(result(1)._1, PokerAction.Raise(2.0))

  // --- sampleActionByPolicy tests ---

  test("sampleActionByPolicy: returns fallback when all zero"):
    val result = GtoSolveEngine.sampleActionByPolicy(
      ordered = Vector.empty,
      fallback = PokerAction.Fold,
      rng = new scala.util.Random(42)
    )
    assertEquals(result, PokerAction.Fold)

  test("sampleActionByPolicy: deterministic with single action"):
    val result = GtoSolveEngine.sampleActionByPolicy(
      ordered = Vector(PokerAction.Call -> 1.0),
      fallback = PokerAction.Fold,
      rng = new scala.util.Random(42)
    )
    assertEquals(result, PokerAction.Call)

  // --- GTO parametrization tests ---

  test("gtoIterations: preflop higher than river"):
    val preflop = GtoSolveEngine.gtoIterations(Street.Preflop, 600, 3)
    val river = GtoSolveEngine.gtoIterations(Street.River, 600, 3)
    assert(preflop > river, s"preflop $preflop should be > river $river")

  test("gtoIterations: 2 candidates gets floor reduction"):
    val full = GtoSolveEngine.gtoIterations(Street.Flop, 600, 3)
    val reduced = GtoSolveEngine.gtoIterations(Street.Flop, 600, 2)
    assert(reduced < full, s"2-candidate $reduced should be < 3-candidate $full")

  test("gtoMaxVillainHands: preflop highest"):
    val preflop = GtoSolveEngine.gtoMaxVillainHands(Street.Preflop, 3)
    val river = GtoSolveEngine.gtoMaxVillainHands(Street.River, 3)
    assert(preflop > river, s"preflop $preflop should be > river $river")

  // --- hashActions tests ---

  test("hashActions: deterministic"):
    val actions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(2.5))
    val hash1 = GtoSolveEngine.hashActions(actions)
    val hash2 = GtoSolveEngine.hashActions(actions)
    assertEquals(hash1, hash2)

  test("hashActions: different actions produce different hash"):
    val h1 = GtoSolveEngine.hashActions(Vector(PokerAction.Fold, PokerAction.Call))
    val h2 = GtoSolveEngine.hashActions(Vector(PokerAction.Check, PokerAction.Call))
    assertNotEquals(h1, h2)

  // --- gtoResponds Fast mode tests ---

  test("gtoResponds Fast: single candidate returns it immediately"):
    val state = GameState(
      street = Street.Flop, pot = 3.0, toCall = 1.0, stackSize = 50.0,
      board = board("Ah", "Kd", "2c"), position = Position.Button
    )
    val result = GtoSolveEngine.gtoResponds(
      hand = hole("7d", "2h"),
      state = state,
      candidates = Vector(PokerAction.Call),
      mode = GtoSolveEngine.GtoMode.Fast,
      opponentPosterior = null, // not used in Fast mode
      baseEquityTrials = 600,
      rng = new scala.util.Random(42),
      perspective = 0,
      exactGtoCache = scala.collection.mutable.HashMap.empty,
      exactGtoCacheStats = GtoSolveEngine.GtoCacheStats()
    )
    assertEquals(result, PokerAction.Call)

  test("gtoResponds Fast: strong hand facing bet does not fold"):
    val state = GameState(
      street = Street.Flop, pot = 5.0, toCall = 1.0, stackSize = 50.0,
      board = board("As", "Ad", "Kh"), position = Position.Button
    )
    // Pocket aces with an ace on board = very strong
    val results = (0 until 50).map { i =>
      GtoSolveEngine.gtoResponds(
        hand = hole("Ac", "Ah"),
        state = state,
        candidates = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(2.5)),
        mode = GtoSolveEngine.GtoMode.Fast,
        opponentPosterior = null,
        baseEquityTrials = 600,
        rng = new scala.util.Random(i),
        perspective = 0,
        exactGtoCache = scala.collection.mutable.HashMap.empty,
        exactGtoCacheStats = GtoSolveEngine.GtoCacheStats()
      )
    }
    val foldCount = results.count(_ == PokerAction.Fold)
    assertEquals(foldCount, 0, s"Quads should never fold, but folded $foldCount times")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `sbt "testOnly sicfun.holdem.engine.GtoSolveEngineTest"`
Expected: FAIL — `GtoSolveEngine` does not exist.

- [ ] **Step 3: Implement GtoSolveEngine**

```scala
package sicfun.holdem.engine

import sicfun.core.{Card, DiscreteDistribution}
import sicfun.holdem.types.*
import sicfun.holdem.cfr.{HoldemCfrConfig, HoldemCfrSolver}

import scala.collection.mutable
import scala.util.Random

/** Unified GTO decision system with cached CFR solving and fast heuristic path.
  *
  * Extracted from TexasHoldemPlayingHall where gtoHeroResponds and
  * gtoVillainResponds were near-identical (only difference: posterior source).
  * Both villainPosteriorForHeroGto and heroPosteriorForGto just called
  * tableRanges.rangeFor(position), so the unified gtoResponds takes the
  * opponent posterior as a parameter.
  */
private[holdem] object GtoSolveEngine:

  enum GtoMode:
    case Fast
    case Exact

  private[holdem] final case class GtoSolveCacheKey(
      perspective: Int,
      canonicalHeroPacked: Long,
      streetOrdinal: Int,
      canonicalBoardPacked: Long,
      potBits: Long,
      toCallBits: Long,
      stackBits: Long,
      candidateHash: Int,
      baseEquityTrials: Int
  )

  private[holdem] final case class GtoCachedPolicy(
      orderedActionProbabilities: Vector[(PokerAction, Double)],
      bestAction: PokerAction,
      provider: String
  )

  private[holdem] final case class GtoCacheStats(
      var hits: Long = 0L,
      var misses: Long = 0L,
      servedByProvider: mutable.Map[String, Long] = mutable.HashMap.empty[String, Long].withDefaultValue(0L),
      solvedByProvider: mutable.Map[String, Long] = mutable.HashMap.empty[String, Long].withDefaultValue(0L)
  ):
    def total: Long = hits + misses
    def hitRate: Double = if total > 0 then hits.toDouble / total.toDouble else 0.0
    def recordHit(provider: String): Unit =
      hits += 1L
      increment(servedByProvider, provider)
    def recordMiss(provider: String): Unit =
      misses += 1L
      increment(servedByProvider, provider)
      increment(solvedByProvider, provider)
    def servedByProviderSnapshot: Map[String, Long] = servedByProvider.toMap
    def solvedByProviderSnapshot: Map[String, Long] = solvedByProvider.toMap
    private def increment(counter: mutable.Map[String, Long], provider: String): Unit =
      counter.update(provider, counter(provider) + 1L)

  private[holdem] val MaxGtoCacheEntries = 500000

  /** Unified GTO decision dispatcher. Replaces the near-identical
    * gtoHeroResponds / gtoVillainResponds pair.
    *
    * @param candidates pre-computed legal actions (Fold/Call/Check + raises)
    * @param opponentPosterior opponent's range (caller passes tableRanges.rangeFor(opponentPosition))
    * @param perspective 0 for hero, 1 for villain (used in cache key and RNG seed)
    */
  def gtoResponds(
      hand: HoleCards,
      state: GameState,
      candidates: Vector[PokerAction],
      mode: GtoMode,
      opponentPosterior: DiscreteDistribution[HoleCards],
      baseEquityTrials: Int,
      rng: Random,
      perspective: Int,
      exactGtoCache: mutable.HashMap[GtoSolveCacheKey, GtoCachedPolicy],
      exactGtoCacheStats: GtoCacheStats
  ): PokerAction =
    if candidates.length <= 1 then candidates.head
    else
      mode match
        case GtoMode.Fast =>
          fastGtoResponds(
            hand = hand,
            state = state,
            candidates = candidates,
            rng = rng
          )
        case GtoMode.Exact =>
          solveGtoByCfr(
            hand = hand,
            state = state,
            candidates = candidates,
            villainPosterior = opponentPosterior,
            baseEquityTrials = baseEquityTrials,
            rng = rng,
            perspective = perspective,
            exactGtoCache = exactGtoCache,
            exactGtoCacheStats = exactGtoCacheStats
          )

  private def solveGtoByCfr(
      hand: HoleCards,
      state: GameState,
      candidates: Vector[PokerAction],
      villainPosterior: DiscreteDistribution[HoleCards],
      baseEquityTrials: Int,
      rng: Random,
      perspective: Int,
      exactGtoCache: mutable.HashMap[GtoSolveCacheKey, GtoCachedPolicy],
      exactGtoCacheStats: GtoCacheStats
  ): PokerAction =
    val canonicalSignature = canonicalHeroBoardSignature(hand = hand, board = state.board)
    val key = buildGtoSolveCacheKey(
      perspective = perspective,
      hand = hand,
      state = state,
      candidates = candidates,
      baseEquityTrials = baseEquityTrials,
      canonicalSignature = canonicalSignature
    )
    exactGtoCache.get(key) match
      case Some(cached) =>
        exactGtoCacheStats.recordHit(cached.provider)
        sampleActionByPolicy(
          ordered = cached.orderedActionProbabilities,
          fallback = cached.bestAction,
          rng = rng
        )
      case None =>
        val config = HoldemCfrConfig(
          iterations = gtoIterations(state.street, baseEquityTrials, candidates.length),
          maxVillainHands = gtoMaxVillainHands(state.street, candidates.length),
          equityTrials = gtoEquityTrials(state.street, baseEquityTrials, candidates.length),
          rngSeed = exactEquitySeed(
            perspective = perspective,
            baseEquityTrials = baseEquityTrials,
            boardSize = state.board.size,
            canonicalSignature = canonicalSignature
          )
        )
        try
          val solution = HoldemCfrSolver.solveShallowDecisionPolicy(
            hero = hand,
            state = state,
            villainPosterior = villainPosterior,
            candidateActions = candidates,
            config = config
          )
          val actionProbs =
            orderedPositiveProbabilities(
              actions = candidates,
              probabilities = solution.actionProbabilities
            )
          exactGtoCacheStats.recordMiss(solution.provider)
          if exactGtoCache.size >= MaxGtoCacheEntries then exactGtoCache.clear()
          exactGtoCache.update(
            key,
            GtoCachedPolicy(
              orderedActionProbabilities = actionProbs,
              bestAction = solution.bestAction,
              provider = solution.provider
            )
          )
          sampleActionByPolicy(
            ordered = actionProbs,
            fallback = solution.bestAction,
            rng = rng
          )
        catch
          case _: Throwable =>
            // Preserve run continuity if a specific CFR solve fails.
            exactGtoCacheStats.recordMiss("random-fallback")
            candidates(rng.nextInt(candidates.length))

  private def fastGtoResponds(
      hand: HoleCards,
      state: GameState,
      candidates: Vector[PokerAction],
      rng: Random
  ): PokerAction =
    val strength = HandStrengthEstimator.fastGtoStrength(hand, state.board, state.street)
    // No allowRaise guard needed: callers pre-filter raises out of candidates
    // via heroCandidates(state, raiseSize, allowRaise) before calling gtoResponds.
    val raiseCandidate = candidates.collectFirst { case action @ PokerAction.Raise(_) => action }
    val callCandidate = candidates.find(_ == PokerAction.Call)
    val foldCandidate = candidates.find(_ == PokerAction.Fold)
    if state.toCall <= 0.0 then
      raiseCandidate match
        case None => PokerAction.Check
        case Some(raiseAction) =>
          val pureRaiseThreshold = fastGtoRaiseThreshold(state.street)
          val mixRaiseThreshold = pureRaiseThreshold - 0.18
          if strength >= pureRaiseThreshold then raiseAction
          else if strength >= mixRaiseThreshold then
            val mix = HandStrengthEstimator.clamp(0.18 + ((strength - mixRaiseThreshold) * 1.7), 0.05, 0.80)
            if rng.nextDouble() < mix then raiseAction else PokerAction.Check
          else PokerAction.Check
    else
      val potOdds = state.potOdds
      val foldThreshold = HandStrengthEstimator.clamp(potOdds + fastGtoFoldMargin(state.street), 0.06, 0.95)
      val raiseThreshold = HandStrengthEstimator.clamp(foldThreshold + fastGtoRaiseGap(state.street), 0.24, 0.98)
      if raiseCandidate.nonEmpty && strength >= raiseThreshold then
        val raiseMix = HandStrengthEstimator.clamp(0.20 + ((strength - raiseThreshold) * 1.3), 0.10, 0.92)
        if rng.nextDouble() < raiseMix then raiseCandidate.get
        else callCandidate.getOrElse(PokerAction.Call)
      else if strength >= foldThreshold then
        callCandidate.getOrElse(PokerAction.Call)
      else
        foldCandidate.getOrElse(PokerAction.Fold)

  private def fastGtoRaiseThreshold(street: Street): Double =
    street match
      case Street.Preflop => 0.78
      case Street.Flop    => 0.74
      case Street.Turn    => 0.71
      case Street.River   => 0.68

  private def fastGtoFoldMargin(street: Street): Double =
    street match
      case Street.Preflop => 0.05
      case Street.Flop    => 0.03
      case Street.Turn    => 0.01
      case Street.River   => -0.01

  private def fastGtoRaiseGap(street: Street): Double =
    street match
      case Street.Preflop => 0.27
      case Street.Flop    => 0.24
      case Street.Turn    => 0.22
      case Street.River   => 0.20

  // --- CFR parametrization ---

  private[holdem] def gtoIterations(
      street: Street,
      baseEquityTrials: Int,
      candidateCount: Int
  ): Int =
    val base = math.max(72, math.min(224, math.round(baseEquityTrials / 3.0).toInt))
    val streetBase =
      street match
        case Street.Preflop => base + 32
        case Street.Flop    => base
        case Street.Turn    => math.max(72, math.round(base * 0.85).toInt)
        case Street.River   => math.max(56, math.round(base * 0.70).toInt)
    if candidateCount <= 2 then
      val floor =
        street match
          case Street.Preflop => 88
          case Street.Flop    => 64
          case Street.Turn    => 56
          case Street.River   => 48
      math.max(floor, math.round(streetBase * 0.60).toInt)
    else
      streetBase

  private[holdem] def gtoMaxVillainHands(
      street: Street,
      candidateCount: Int
  ): Int =
    val base =
      street match
        case Street.Preflop => 56
        case Street.Flop    => 32
        case Street.Turn    => 24
        case Street.River   => 16
    if candidateCount <= 2 then math.max(16, base - 12) else base

  private[holdem] def gtoEquityTrials(
      street: Street,
      baseEquityTrials: Int,
      candidateCount: Int
  ): Int =
    val base =
      street match
        case Street.Preflop => math.max(80, baseEquityTrials / 3)
        case Street.Flop    => math.max(48, baseEquityTrials / 6)
        case Street.Turn    => math.max(32, baseEquityTrials / 8)
        case Street.River   => 24
    if candidateCount <= 2 then
      val floor =
        street match
          case Street.Preflop => 64
          case Street.Flop    => 36
          case Street.Turn    => 24
          case Street.River   => 16
      math.max(floor, math.round(base * 0.65).toInt)
    else
      base

  // --- Cache key construction ---

  private def buildGtoSolveCacheKey(
      perspective: Int,
      hand: HoleCards,
      state: GameState,
      candidates: Vector[PokerAction],
      baseEquityTrials: Int,
      canonicalSignature: (Long, Long)
  ): GtoSolveCacheKey =
    val (canonicalHeroPacked, canonicalBoardPacked) = canonicalSignature
    GtoSolveCacheKey(
      perspective = perspective,
      canonicalHeroPacked = canonicalHeroPacked,
      streetOrdinal = state.street.ordinal,
      canonicalBoardPacked = canonicalBoardPacked,
      potBits = java.lang.Double.doubleToLongBits(state.pot),
      toCallBits = java.lang.Double.doubleToLongBits(state.toCall),
      stackBits = java.lang.Double.doubleToLongBits(state.stackSize),
      candidateHash = hashActions(candidates),
      baseEquityTrials = baseEquityTrials
    )

  private def exactEquitySeed(
      perspective: Int,
      baseEquityTrials: Int,
      boardSize: Int,
      canonicalSignature: (Long, Long)
  ): Long =
    val (canonicalHeroPacked, canonicalBoardPacked) = canonicalSignature
    mix64(
      canonicalHeroPacked ^
        java.lang.Long.rotateLeft(canonicalBoardPacked, 11) ^
        (perspective.toLong << 48) ^
        (baseEquityTrials.toLong << 16) ^
        boardSize.toLong
    )

  // --- Canonical board signature ---

  private val SuitPermutations: Array[Array[Int]] =
    Array(
      Array(0, 1, 2, 3), Array(0, 1, 3, 2), Array(0, 2, 1, 3), Array(0, 2, 3, 1),
      Array(0, 3, 1, 2), Array(0, 3, 2, 1), Array(1, 0, 2, 3), Array(1, 0, 3, 2),
      Array(1, 2, 0, 3), Array(1, 2, 3, 0), Array(1, 3, 0, 2), Array(1, 3, 2, 0),
      Array(2, 0, 1, 3), Array(2, 0, 3, 1), Array(2, 1, 0, 3), Array(2, 1, 3, 0),
      Array(2, 3, 0, 1), Array(2, 3, 1, 0), Array(3, 0, 1, 2), Array(3, 0, 2, 1),
      Array(3, 1, 0, 2), Array(3, 1, 2, 0), Array(3, 2, 0, 1), Array(3, 2, 1, 0)
    )

  private[holdem] def canonicalHeroBoardSignature(hand: HoleCards, board: Board): (Long, Long) =
    val boardSize = board.cards.length
    val remappedBoardIds = new Array[Int](boardSize)
    var bestHeroPacked = Long.MaxValue
    var bestBoardPacked = Long.MaxValue
    var permIdx = 0
    while permIdx < SuitPermutations.length do
      val suitMap = SuitPermutations(permIdx)
      val heroFirstId = remapCardId(hand.first, suitMap)
      val heroSecondId = remapCardId(hand.second, suitMap)
      val lowHero = math.min(heroFirstId, heroSecondId)
      val highHero = math.max(heroFirstId, heroSecondId)
      val heroPacked = ((lowHero.toLong << 6) | highHero.toLong) & 0xFFFL

      var idx = 0
      while idx < boardSize do
        remappedBoardIds(idx) = remapCardId(board.cards(idx), suitMap)
        idx += 1
      java.util.Arrays.sort(remappedBoardIds)
      var boardPacked = boardSize.toLong
      idx = 0
      while idx < boardSize do
        boardPacked = (boardPacked << 6) | remappedBoardIds(idx).toLong
        idx += 1

      if heroPacked < bestHeroPacked || (heroPacked == bestHeroPacked && boardPacked < bestBoardPacked) then
        bestHeroPacked = heroPacked
        bestBoardPacked = boardPacked
      permIdx += 1
    (bestHeroPacked, bestBoardPacked)

  private def remapCardId(card: Card, suitMap: Array[Int]): Int =
    val mappedSuit = suitMap(card.suit.ordinal)
    (mappedSuit * 13) + card.rank.ordinal

  // --- Action hashing ---

  private[holdem] def hashActions(actions: Vector[PokerAction]): Int =
    var hash = 1
    var idx = 0
    while idx < actions.length do
      hash = 31 * hash + hashAction(actions(idx))
      idx += 1
    hash

  private def hashAction(action: PokerAction): Int =
    action match
      case PokerAction.Fold => 1
      case PokerAction.Check => 2
      case PokerAction.Call => 3
      case PokerAction.Raise(amount) =>
        31 * 4 + java.lang.Double.hashCode(amount)

  // --- Policy sampling ---

  private[holdem] def orderedPositiveProbabilities(
      actions: Vector[PokerAction],
      probabilities: Map[PokerAction, Double]
  ): Vector[(PokerAction, Double)] =
    actions.flatMap { action =>
      val probability = probabilities.getOrElse(action, 0.0)
      if probability.isFinite && probability > 0.0 then Some(action -> probability)
      else None
    }

  private[holdem] def sampleActionByPolicy(
      ordered: Vector[(PokerAction, Double)],
      fallback: PokerAction,
      rng: Random
  ): PokerAction =
    var total = 0.0
    var i = 0
    while i < ordered.length do
      total += ordered(i)._2
      i += 1
    if total <= 0.0 then fallback
    else
      val target = rng.nextDouble() * total
      var cumulative = 0.0
      var idx = 0
      while idx < ordered.length do
        val (action, probability) = ordered(idx)
        cumulative += probability
        if target <= cumulative then return action
        idx += 1
      ordered.last._1

  // --- Utilities ---

  private def mix64(value: Long): Long =
    var z = value + 0x9E3779B97F4A7C15L
    z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L
    z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL
    z ^ (z >>> 31)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `sbt "testOnly sicfun.holdem.engine.GtoSolveEngineTest"`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/engine/GtoSolveEngine.scala \
  src/test/scala/sicfun/holdem/engine/GtoSolveEngineTest.scala
git commit -m "refactor: extract GtoSolveEngine with unified gtoResponds, CFR caching, fast heuristic path"
```

---

## Task 3: Extract ArchetypeVillainResponder

**Files:**
- Create: `src/main/scala/sicfun/holdem/engine/ArchetypeVillainResponder.scala`
- Create: `src/test/scala/sicfun/holdem/engine/ArchetypeVillainResponderTest.scala`

**Depends on:** Task 1 (HandStrengthEstimator.streetStrength, clamp)

- [ ] **Step 1: Write failing tests**

```scala
package sicfun.holdem.engine

import munit.FunSuite
import sicfun.core.Card
import sicfun.holdem.types.*

class ArchetypeVillainResponderTest extends FunSuite:

  test("styleProfile: Nit is tight and passive"):
    val profile = ArchetypeVillainResponder.styleProfile(PlayerArchetype.Nit)
    assert(profile.looseness < 0.25, s"Nit looseness ${profile.looseness}")
    assert(profile.aggression < 0.25, s"Nit aggression ${profile.aggression}")

  test("styleProfile: Maniac is loose and aggressive"):
    val profile = ArchetypeVillainResponder.styleProfile(PlayerArchetype.Maniac)
    assert(profile.looseness > 0.75, s"Maniac looseness ${profile.looseness}")
    assert(profile.aggression > 0.85, s"Maniac aggression ${profile.aggression}")

  test("styleProfile: Tag is moderate"):
    val profile = ArchetypeVillainResponder.styleProfile(PlayerArchetype.Tag)
    assert(profile.looseness > 0.3 && profile.looseness < 0.6)
    assert(profile.aggression > 0.3 && profile.aggression < 0.6)

  test("styleProfile: CallingStation is loose and passive"):
    val profile = ArchetypeVillainResponder.styleProfile(PlayerArchetype.CallingStation)
    assert(profile.looseness > 0.8, s"Station looseness ${profile.looseness}")
    assert(profile.aggression < 0.3, s"Station aggression ${profile.aggression}")

  test("styleProfile: all archetypes covered"):
    PlayerArchetype.values.foreach { archetype =>
      val profile = ArchetypeVillainResponder.styleProfile(archetype)
      assert(profile.looseness >= 0.0 && profile.looseness <= 1.0)
      assert(profile.aggression >= 0.0 && profile.aggression <= 1.0)
    }

  test("villainResponds: check-to-act with no raise allowed returns Check"):
    val state = GameState(
      street = Street.Flop, pot = 3.0, toCall = 0.0, stackSize = 50.0,
      board = Board.empty, position = Position.BigBlind
    )
    val h = HoleCards.from(Vector(Card.parse("2c").get, Card.parse("7d").get))
    val action = ArchetypeVillainResponder.villainResponds(
      hand = h, style = PlayerArchetype.Nit, state = state,
      allowRaise = false, raiseSize = 2.5, rng = new scala.util.Random(42)
    )
    assertEquals(action, PokerAction.Check)

  test("villainResponds: Maniac raises more than Nit over many trials"):
    val state = GameState(
      street = Street.Flop, pot = 5.0, toCall = 0.0, stackSize = 50.0,
      board = Board.from(Vector(Card.parse("Ah").get, Card.parse("Kd").get, Card.parse("Qs").get)),
      position = Position.BigBlind
    )
    val h = HoleCards.from(Vector(Card.parse("Jh").get, Card.parse("Ts").get))
    def countRaises(style: PlayerArchetype, seed: Int): Int =
      (0 until 100).count { i =>
        ArchetypeVillainResponder.villainResponds(
          hand = h, style = style, state = state,
          allowRaise = true, raiseSize = 2.5, rng = new scala.util.Random(seed + i)
        ) match
          case PokerAction.Raise(_) => true
          case _                    => false
      }
    val maniacRaises = countRaises(PlayerArchetype.Maniac, 1000)
    val nitRaises = countRaises(PlayerArchetype.Nit, 1000)
    assert(maniacRaises > nitRaises, s"Maniac raises $maniacRaises should be > Nit raises $nitRaises")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `sbt "testOnly sicfun.holdem.engine.ArchetypeVillainResponderTest"`
Expected: FAIL — `ArchetypeVillainResponder` does not exist.

- [ ] **Step 3: Implement ArchetypeVillainResponder**

```scala
package sicfun.holdem.engine

import sicfun.holdem.types.*

import scala.util.Random

/** Archetype-based villain decision model with fixed style profiles.
  *
  * Extracted from TexasHoldemPlayingHall. Uses HandStrengthEstimator for
  * hand evaluation and applies archetype-specific betting heuristics.
  *
  * Note: PlayerArchetype is defined in the same package (sicfun.holdem.engine)
  * in RealTimeAdaptiveEngine.scala, so it's accessible without explicit import.
  */
private[holdem] object ArchetypeVillainResponder:

  final case class VillainStyleProfile(looseness: Double, aggression: Double)

  def styleProfile(archetype: PlayerArchetype): VillainStyleProfile =
    archetype match
      case PlayerArchetype.Nit            => VillainStyleProfile(looseness = 0.20, aggression = 0.18)
      case PlayerArchetype.Tag            => VillainStyleProfile(looseness = 0.45, aggression = 0.40)
      case PlayerArchetype.Lag            => VillainStyleProfile(looseness = 0.68, aggression = 0.66)
      case PlayerArchetype.CallingStation => VillainStyleProfile(looseness = 0.86, aggression = 0.24)
      case PlayerArchetype.Maniac         => VillainStyleProfile(looseness = 0.80, aggression = 0.92)

  def villainResponds(
      hand: HoleCards,
      style: PlayerArchetype,
      state: GameState,
      allowRaise: Boolean,
      raiseSize: Double,
      rng: Random
  ): PokerAction =
    val profile = styleProfile(style)
    val strength = HandStrengthEstimator.streetStrength(hand, state.board, state.street, rng)

    if state.toCall <= 0.0 then
      if !allowRaise then PokerAction.Check
      else
        val betChance = HandStrengthEstimator.clamp((strength - 0.35) * 0.9 + (profile.aggression * 0.35), 0.02, 0.92)
        if rng.nextDouble() < betChance then PokerAction.Raise(raiseSize)
        else PokerAction.Check
    else
      val potOdds = state.potOdds
      val foldThreshold = HandStrengthEstimator.clamp(potOdds + (0.35 - profile.looseness * 0.2), 0.05, 0.95)
      val raiseThreshold = HandStrengthEstimator.clamp(0.68 - profile.aggression * 0.12 + potOdds * 0.2, 0.35, 0.9)
      if allowRaise && strength >= raiseThreshold && rng.nextDouble() < (0.15 + profile.aggression * 0.55) then
        PokerAction.Raise(raiseSize)
      else if strength < foldThreshold then PokerAction.Fold
      else PokerAction.Call
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `sbt "testOnly sicfun.holdem.engine.ArchetypeVillainResponderTest"`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/engine/ArchetypeVillainResponder.scala \
  src/test/scala/sicfun/holdem/engine/ArchetypeVillainResponderTest.scala
git commit -m "refactor: extract ArchetypeVillainResponder from PlayingHall"
```

---

## Task 4: Refactor PlayingHall to Delegate

**Files:**
- Modify: `src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala`

**Depends on:** Tasks 1, 2, 3

This task removes ~575 lines from PlayingHall and updates call sites to delegate to the extracted modules. The changes are mechanical — delete private methods/types and update callers.

- [ ] **Step 1: Compile check baseline**

Run: `sbt compile`
Expected: PASS — no errors before refactoring.

- [ ] **Step 2: Add import for GtoSolveEngine types**

At line 8 (after `import sicfun.holdem.cli.*`), add:

```scala
import sicfun.holdem.engine.GtoSolveEngine.{GtoMode, GtoSolveCacheKey, GtoCachedPolicy, GtoCacheStats}
```

This import makes the cache types available under their original unqualified names, so `resolveHand`, `HandResolver`, and `Config` signatures stay unchanged.

- [ ] **Step 3: Delete GtoMode enum from PlayingHall**

Delete lines 29-31:
```scala
  private enum GtoMode:
    case Fast
    case Exact
```

Now resolved via the import from Step 2.

- [ ] **Step 4: Delete VillainStyleProfile from PlayingHall**

Delete line 150:
```scala
  private final case class VillainStyleProfile(looseness: Double, aggression: Double)
```

Now defined in `ArchetypeVillainResponder.VillainStyleProfile`. PlayingHall does not reference this type directly — it was only used by `villainResponds` and `styleProfile`, both now extracted.

- [ ] **Step 5: Delete cache types and constants from PlayingHall**

Delete these private definitions (lines 169-227, 236-245):

```scala
  // DELETE: GtoSolveCacheKey (lines 169-179)
  // DELETE: GtoCachedPolicy (lines 181-185)
  // DELETE: GtoCacheStats (lines 187-220)
  // DELETE: MaxExactGtoCacheEntries (line 227)
  // DELETE: SuitPermutations (lines 236-245)
```

`GtoSolveCacheKey`, `GtoCachedPolicy`, `GtoCacheStats` are now imported from `GtoSolveEngine`. `MaxExactGtoCacheEntries` is now `GtoSolveEngine.MaxGtoCacheEntries`. `SuitPermutations` is now private to `GtoSolveEngine`.

- [ ] **Step 6: Update `trimExactCacheIfNeeded` to use GtoSolveEngine constant**

At line ~510, update:

```scala
  // BEFORE:
  if exactGtoCache.size > MaxExactGtoCacheEntries then

  // AFTER:
  if exactGtoCache.size > GtoSolveEngine.MaxGtoCacheEntries then
```

- [ ] **Step 7: Update `decideHero` GTO branch**

In `HandResolver.decideHero`, replace the `gtoHeroResponds` call (lines ~875-886) with:

```scala
        case HeroMode.Gto =>
          multiwayRecommendationFor(
            actor = heroPosition,
            state = state,
            candidateActions = candidates
          ).map(_.bestAction)
            .getOrElse(
              GtoSolveEngine.gtoResponds(
                hand = deal.holeCardsFor(heroPosition),
                state = state,
                candidates = candidates,
                mode = config.gtoMode,
                opponentPosterior = tableRanges.rangeFor(focusVillainPosition),
                baseEquityTrials = config.equityTrials,
                rng = rng,
                perspective = 0,
                exactGtoCache = exactGtoCache,
                exactGtoCacheStats = exactGtoCacheStats
              )
            )
```

Changes from original: `gtoHeroResponds` → `GtoSolveEngine.gtoResponds`, `allowRaise`/`raiseSize` parameters removed (candidates passed instead), `villainPosition` replaced by `opponentPosterior = tableRanges.rangeFor(focusVillainPosition)`.

- [ ] **Step 8: Update `decideVillain` Archetype branch**

In `HandResolver.decideVillain`, replace the `villainResponds` call (lines ~973-980) with:

```scala
        case VillainMode.Archetype(style) =>
          ArchetypeVillainResponder.villainResponds(
            hand = deal.holeCardsFor(position),
            style = style,
            state = state,
            allowRaise = allowRaise,
            raiseSize = config.raiseSize,
            rng = rng
          )
```

Change: just qualify with `ArchetypeVillainResponder.`

- [ ] **Step 9: Update `decideVillain` GTO branch**

In `HandResolver.decideVillain`, replace the `gtoVillainResponds` call (lines ~982-993) with:

```scala
        case VillainMode.Gto =>
          multiwayRecommendation
            .map(_.bestAction)
            .getOrElse(
              GtoSolveEngine.gtoResponds(
                hand = deal.holeCardsFor(position),
                state = state,
                candidates = candidates,
                mode = config.gtoMode,
                opponentPosterior = tableRanges.rangeFor(heroPosition),
                baseEquityTrials = config.equityTrials,
                rng = rng,
                perspective = 1,
                exactGtoCache = exactGtoCache,
                exactGtoCacheStats = exactGtoCacheStats
              )
            )
```

- [ ] **Step 10: Delete all extracted methods from PlayingHall**

Delete these methods from the `TexasHoldemPlayingHall` object (line numbers approximate, work top-down):

1. `villainResponds` (~lines 1320-1343)
2. `gtoHeroResponds` (~lines 1347-1390)
3. `gtoVillainResponds` (~lines 1391-1435)
4. `solveGtoByCfr` (~lines 1436-1508)
5. `fastGtoResponds` (~lines 1510-1540)
6. `fastGtoStrength` (~lines 1542-1555)
7. `fastGtoRaiseThreshold` (~lines 1557-1562)
8. `fastGtoFoldMargin` (~lines 1564-1569)
9. `fastGtoRaiseGap` (~lines 1571-1576)
10. `villainPosteriorForHeroGto` (~lines 1578-1583)
11. `heroPosteriorForGto` (~lines 1585-1590)
12. `gtoIterations` (~lines 1592-1617)
13. `gtoMaxVillainHands` (~lines 1619-1630)
14. `gtoEquityTrials` (~lines 1632-1649)
15. `buildGtoSolveCacheKey` (~lines 1651-1663)
16. `exactEquitySeed` (~lines 1665-1676)
17. `canonicalHeroBoardSignature` (~lines 1678-1710)
18. `remapCardId` (~lines 1712-1714)
19. `hashActions` (~lines 1716-1722)
20. `hashAction` (~lines 1724-1729)
21. `orderedPositiveProbabilities` (~lines 1731-1737)
22. `sampleActionByPolicy` (~lines 1739-1755)
23. `mix64` (~lines 1757-1761)
24. `preflopStrength` (~lines 1763-1797)
25. `streetStrength` (~lines 1798-1812)
26. `bestCategoryStrength` (~lines 1813-1830)
27. `drawPotential` (~lines 1831-1860)
28. `hasTightRun` (~lines 1861-1872)
29. `styleProfile` (~lines 1866-1872)
30. `clamp` (~line 2017-2018)

- [ ] **Step 11: Compile check**

Run: `sbt compile`
Expected: PASS — no errors after refactoring. If there are unused import warnings for the removed imports (`HandEvaluator` etc.), clean them up.

- [ ] **Step 12: Run full test suite**

Run: `sbt test`
Expected: ALL PASS (including the new tests from Tasks 1-3).

- [ ] **Step 13: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala
git commit -m "refactor: PlayingHall delegates GTO, hand strength, and villain to extracted modules"
```

---

## Task 5: Full Verification and Cleanup

**Depends on:** Task 4

- [ ] **Step 1: Run full test suite**

Run: `sbt test`
Expected: ALL PASS — no regressions anywhere.

- [ ] **Step 2: Compile clean check**

Run: `sbt compile 2>&1 | grep -i "warn\|error"`
Expected: No new warnings or errors.

- [ ] **Step 3: Verify line count reduction**

Run:
```bash
wc -l src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala
wc -l src/main/scala/sicfun/holdem/engine/HandStrengthEstimator.scala
wc -l src/main/scala/sicfun/holdem/engine/GtoSolveEngine.scala
wc -l src/main/scala/sicfun/holdem/engine/ArchetypeVillainResponder.scala
```

Expected:
- `TexasHoldemPlayingHall.scala`: ~2060 lines (down from 2633, ~575 removed)
- `HandStrengthEstimator.scala`: ~100 lines
- `GtoSolveEngine.scala`: ~370 lines
- `ArchetypeVillainResponder.scala`: ~45 lines
- **PlayingHall reduction: ~22% (575 lines)**
- **Duplication eliminated: gtoHeroResponds/gtoVillainResponds unified**

- [ ] **Step 4: Check for unused imports in PlayingHall**

After deleting 30 methods, PlayingHall may have unused imports. Check for:
- `sicfun.core.HandEvaluator` — no longer used (was in `bestCategoryStrength`)
- `sicfun.core.CardId` — check if still used (might be used in `dealHand`)

Clean up any unused imports.

- [ ] **Step 5: Final commit (if any cleanup needed)**

```bash
git add -u
git commit -m "chore: final cleanup after PlayingHall GTO extraction"
```

---

## Cumulative Impact (3-Plan Series)

| Plan | Files | Lines Removed | Duplication Eliminated |
|------|-------|---------------|----------------------|
| Plan 1: Decision Pipeline | 3 modified, 4 new | ~240 from ACPC+Slumbot runners | decideHero, legalRaiseCandidates, HeroMode ×3 |
| Plan 2: Bench Consolidation | 15 modified, 2 new | ~110 from bench files | card ×11, hole ×10, BatchData ×4, loadBatch ×3 |
| Plan 3: PlayingHall GTO | 1 modified, 3 new | ~575 from PlayingHall | gtoHeroResponds/gtoVillainResponds unified |
| **Total** | | **~925 lines removed** | **God object decomposed from 2633→~2060** |
