# Per-Player P&L Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the incorrect "per-villain net = hero-net-attributed-to-primary-villain" semantic in `TexasHoldemPlayingHall` with the real chip flow per player (hero + each villain by name), so that the Playing Hall dashboard shows who actually won and lost and how much.

**Architecture:**
The current engine is hero-centric: when hero folds, `markFolded` flips `handOver = true` and the hand terminates without resolving who among the remaining villains would win. We fix this by (1) only flagging the hand over when ≤1 contestant remains (not "hero folded"), (2) letting the existing betting-round machinery continue asking villain policies for decisions post-hero-fold (it already supports that via `decidePosition` → `decideVillain`), (3) replacing `ShowdownResolution(heroPayout)` with a full `payouts: Map[Position, Double]` that covers all live contestants, and (4) computing a `perPositionNet` map per hand that's rolled up by villain name in the hall accumulator. Hero's `heroNet` and `outcome` keep bit-identical semantics; only the previously-untracked villain chip flow becomes real.

**Tech Stack:** Scala 3.8.1, SBT, munit 1.2.2.

---

## File Structure

**Modify:**
- `src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala` — core engine changes
  - `HandResult` (L203) — add `perPositionNet: Map[Position, Double]`
  - `ShowdownResolution` (L219) — replace `heroPayout` with full `payouts` map
  - `showdownResolution()` (L1442) — return full payout map
  - `markFolded()` (L1273) — end hand only when ≤1 contestant remains
  - `play()` (L747) — unify payout computation across fold/showdown branches
  - `recordOutcome()` (L525) — use `perPositionNet` to accumulate real villain nets
  - accumulator field `perVillainNet` (L360) — accept negative values (no semantic change in type)
- `src/test/scala/sicfun/holdem/runtime/TexasHoldemPlayingHallTest.scala` — add invariants/integration assertions
- `src/test/scala/sicfun/holdem/web/HandHistoryReviewServerTest.scala` — fixture already correct, no edits expected

**No edits required (semantics downstream are now correct):**
- `src/main/scala/sicfun/holdem/web/HandHistoryReviewServer.scala` — JSON field stays `perVillainNetChips`
- `docs/site-preview-hybrid/site.js` — already uses `formatSigned(chips)` so negative values render fine
- `src/main/scala/sicfun/holdem/validation/AdaptiveProofHarness.scala` — `perVillainAggregateBbPer100` still sums per-villain chip deltas; values will now be negative where villains lose (correct)

---

## Task 1: Failing invariant test — sum of per-player net must be zero

**Files:**
- Modify: `src/test/scala/sicfun/holdem/runtime/TexasHoldemPlayingHallTest.scala`

- [ ] **Step 1: Add a test that asserts sum of hero + per-villain nets ≈ 0 in a seeded 3-player hall run**

Append at end of class body (before closing brace):

```scala
  test("playing hall: hero net + per-villain nets sum to zero (zero-sum conservation)") {
    withScalaCfrProvider {
      val outDir = Files.createTempDirectory("hall-conservation-")
      try
        val summary = TexasHoldemPlayingHall.run(
          TexasHoldemPlayingHall.RunConfig(
            hands = 200,
            tableCount = 1,
            playerCount = 3,
            heroStyle = "adaptive",
            heroPosition = "Button",
            gtoMode = "exact",
            villainPool = Vector("tag", "gto"),
            heroExplorationRate = 0.0,
            raiseSize = 2.5,
            bunchingTrials = 0,
            equityTrials = 120,
            learnEveryHands = 0,
            learningWindowSamples = 0,
            saveReviewHandHistory = false,
            seed = 42L,
            outDir = outDir
          )
        )
        val totalVillainNet = summary.perVillainNetChips.values.sum
        val delta = math.abs(summary.heroNetChips + totalVillainNet)
        assert(
          delta < 0.01,
          s"zero-sum violated: heroNet=${summary.heroNetChips} sumVillainNet=$totalVillainNet delta=$delta"
        )
      finally
        CliHelpers.deleteRecursively(outDir)
    }
  }
```

- [ ] **Step 2: Run the test and confirm it fails under the current bug**

Run: `sbt "testOnly sicfun.holdem.runtime.TexasHoldemPlayingHallTest -- --tests=*zero-sum*"`
Expected: FAIL — current per-villain aggregation reports hero-net-under-villain-name so sum ≠ 0.

- [ ] **Step 3: Commit the failing test**

```bash
git add src/test/scala/sicfun/holdem/runtime/TexasHoldemPlayingHallTest.scala
git commit -m "test(playing-hall): add failing zero-sum invariant for per-player P&L"
```

---

## Task 2: Failing integration test — hands must actually resolve after hero folds multi-way

**Files:**
- Modify: `src/test/scala/sicfun/holdem/runtime/TexasHoldemPlayingHallTest.scala`

- [ ] **Step 1: Add a test that asserts at least one hand resolves past hero's fold (villain actions recorded post-hero-fold)**

Append after the previous test:

```scala
  test("playing hall: multi-way hands continue after hero folds (villain vs villain resolution)") {
    withScalaCfrProvider {
      val outDir = Files.createTempDirectory("hall-post-fold-")
      try
        val _ = TexasHoldemPlayingHall.run(
          TexasHoldemPlayingHall.RunConfig(
            hands = 400,
            tableCount = 1,
            playerCount = 3,
            heroStyle = "adaptive",
            heroPosition = "Button",
            gtoMode = "exact",
            villainPool = Vector("tag", "gto"),
            heroExplorationRate = 0.0,
            raiseSize = 2.5,
            bunchingTrials = 0,
            equityTrials = 120,
            learnEveryHands = 0,
            learningWindowSamples = 0,
            saveReviewHandHistory = true,
            seed = 42L,
            outDir = outDir
          )
        )
        val handRows = Files.readAllLines(outDir.resolve("hands.tsv"), StandardCharsets.UTF_8).asScala.toVector
        val header = handRows.head.split("\t").toVector
        val heroActionIdx = header.indexOf("heroAction")
        val streetsPlayedIdx = header.indexOf("streetsPlayed")
        assert(heroActionIdx >= 0 && streetsPlayedIdx >= 0)
        val postHeroFold =
          handRows.tail.count { row =>
            val cells = row.split("\t").toVector
            cells.lift(heroActionIdx).contains("Fold") &&
              cells.lift(streetsPlayedIdx).exists(_.toInt > 1)
          }
        assert(
          postHeroFold > 0,
          "expected at least one hand where hero folded but streets continued past preflop for remaining villains"
        )
      finally
        CliHelpers.deleteRecursively(outDir)
    }
  }
```

- [ ] **Step 2: Run and confirm it fails**

Run: `sbt "testOnly sicfun.holdem.runtime.TexasHoldemPlayingHallTest -- --tests=*multi-way*"`
Expected: FAIL — under current code, `handOver = true` the instant hero folds so `streetsPlayed` never advances.

- [ ] **Step 3: Commit**

```bash
git add src/test/scala/sicfun/holdem/runtime/TexasHoldemPlayingHallTest.scala
git commit -m "test(playing-hall): add failing post-hero-fold continuation test"
```

---

## Task 3: Refactor `ShowdownResolution` to carry full payouts map

**Files:**
- Modify: `src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala`

- [ ] **Step 1: Replace `ShowdownResolution` case class**

Replace the block at line 218-221:

```scala
  /** The amount of chips hero receives at showdown (before subtracting hero's own contribution). */
  private final case class ShowdownResolution(
      heroPayout: Double
  )
```

with:

```scala
  /** Per-position payouts at hand resolution (showdown or last-standing). Folded positions
    * are absent from the map (they receive 0). Net chips per position = payouts(pos) - contribution(pos).
    */
  private final case class ShowdownResolution(
      payouts: Map[Position, Double]
  ):
    def heroPayout(heroPosition: Position): Double = payouts.getOrElse(heroPosition, 0.0)
```

- [ ] **Step 2: Rewrite `showdownResolution()` to produce the full map and to handle the single-survivor case**

Replace the body at line 1442-1464:

```scala
    private def showdownResolution(): ShowdownResolution =
      val remainingPlayers = liveContestants
      if remainingPlayers.isEmpty then ShowdownResolution(payouts = Map.empty)
      else if remainingPlayers.size == 1 then
        ShowdownResolution(payouts = Map(remainingPlayers.head -> roundMoney(pot)))
      else
        val boardCards = deal.board.cards
        val ranked = remainingPlayers.map { position =>
          val hand = deal.holeCardsFor(position)
          position -> HandEvaluator.evaluate7PackedDirect(
            hand.first,
            hand.second,
            boardCards(0),
            boardCards(1),
            boardCards(2),
            boardCards(3),
            boardCards(4)
          )
        }.toMap
        val payouts = sidePotPayouts(
          contributions = contributionByPosition.toMap,
          remainingPlayers = remainingPlayers,
          handStrengthByPosition = ranked
        )
        ShowdownResolution(payouts = payouts)
```

- [ ] **Step 3: Compile check**

Run: `sbt compile`
Expected: FAIL — `heroPayout` is referenced as a plain field at line 759 (`showdown.heroPayout`).

---

## Task 4: Unify hand-end payout computation in `play()`

**Files:**
- Modify: `src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala`

- [ ] **Step 1: Replace `play()` resolution branches with a single payouts-driven computation**

Replace lines 747-778 (the full `play(): HandResult` body):

```scala
    def play(): HandResult =
      strategicHelperOpt.foreach(_.startHand(deal.holeCardsFor(heroPosition)))
      if !handOver then playPreflop()
      if !handOver then playPostflopStreet(Street.Flop)
      if !handOver then playPostflopStreet(Street.Turn)
      if !handOver then playPostflopStreet(Street.River)
      strategicHelperOpt.foreach(_.endHand())

      val resolution = showdownResolution()
      val perPositionNet: Map[Position, Double] =
        tableScenario.activePositions.iterator.map { position =>
          val payout = resolution.payouts.getOrElse(position, 0.0)
          position -> roundMoney(payout - contributionOf(position))
        }.toMap

      val heroNet = perPositionNet.getOrElse(heroPosition, 0.0)
      outcome =
        if heroNet > MoneyEpsilon then 1
        else if heroNet < -MoneyEpsilon then -1
        else 0

      HandResult(
        heroNet = heroNet,
        perPositionNet = perPositionNet,
        outcome = outcome,
        tableScenario = tableScenario,
        villainDecision = firstVillainDecision,
        villainTrainingSamples = villainTrainingSamples.toVector,
        ddreTrainingSamples = ddreTrainingSamples.toVector,
        raiseResponses = raiseResponses.toVector,
        heroActions = heroActions.toVector,
        villainActions = villainActions.toVector,
        streetsPlayed = streetsPlayed,
        reviewHistoryLines = reviewHistoryLines.toVector,
        maxLivePlayers = tableScenario.activePositions.size
      )
```

- [ ] **Step 2: Add `perPositionNet` field to `HandResult`**

In the `HandResult` case class at line 203, insert `perPositionNet: Map[Position, Double],` after `heroNet`:

```scala
  private final case class HandResult(
      heroNet: Double,
      perPositionNet: Map[Position, Double],
      outcome: Int,
      tableScenario: TableScenario,
      villainDecision: Option[(GameState, PokerAction)],
      villainTrainingSamples: Vector[(GameState, HoleCards, PokerAction)],
      ddreTrainingSamples: Vector[DdreTrainingSample],
      raiseResponses: Vector[PokerAction],
      heroActions: Vector[PokerAction],
      villainActions: Vector[PokerAction],
      streetsPlayed: Int,
      reviewHistoryLines: Vector[String],
      maxLivePlayers: Int
  )
```

- [ ] **Step 3: Compile**

Run: `sbt compile`
Expected: compiles (assuming no other caller references `showdown.heroPayout` as a raw field — verified via Grep in planning phase, only one usage).

---

## Task 5: Fix `markFolded` — continue hand when ≥2 contestants remain

**Files:**
- Modify: `src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala`

- [ ] **Step 1: Replace `markFolded` body**

Replace lines 1273-1283:

```scala
    /** Marks a position as folded. The hand ends only when one or fewer live contestants remain;
      * hero folding mid-multi-way no longer short-circuits the betting round — the remaining
      * villains continue acting under their own decision policies until one is left standing
      * or the river showdown resolves. Preflop folds are tracked separately for bunching-fold
      * inference.
      */
    private def markFolded(position: Position, street: Street): Unit =
      if !foldedPositions.contains(position) then
        foldedPositions += position
        if street == Street.Preflop && !preflopFoldedPositions.contains(position) then
          preflopFoldedPositions += position
      if liveContestants.size <= 1 then
        handOver = true
```

Rationale: `outcome = ±1` assignments inside `markFolded` are dead code — `play()` recomputes `outcome` from final `heroNet` anyway.

- [ ] **Step 2: Run the zero-sum test from Task 1**

Run: `sbt "testOnly sicfun.holdem.runtime.TexasHoldemPlayingHallTest -- --tests=*zero-sum*"`
Expected: PASS — per-position nets now sum to zero because all contestants are resolved consistently.

- [ ] **Step 3: Run the multi-way continuation test from Task 2**

Run: `sbt "testOnly sicfun.holdem.runtime.TexasHoldemPlayingHallTest -- --tests=*multi-way*"`
Expected: PASS — streets advance past hero's fold and villain decisions continue.

---

## Task 6: Swap `perVillainNet` accumulator from hero-net-attribution to real villain net

**Files:**
- Modify: `src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala`

- [ ] **Step 1: Replace `recordOutcome` body**

Replace lines 525-531:

```scala
    private def recordOutcome(result: HandResult, tableScenario: TableScenario): Unit =
      heroNet += result.heroNet
      if result.outcome > 0 then heroWins += 1
      else if result.outcome < 0 then heroLosses += 1
      else heroTies += 1
      tableScenario.activeVillainPositions.foreach { position =>
        val villainName = tableScenario.villainProfileByPosition(position).name
        val villainDelta = result.perPositionNet.getOrElse(position, 0.0)
        perVillainNet.update(villainName, perVillainNet(villainName) + villainDelta)
      }
```

- [ ] **Step 2: Compile and run the full hall test suite**

Run: `sbt "testOnly sicfun.holdem.runtime.TexasHoldemPlayingHallTest"`
Expected: all tests PASS — including pre-existing integration tests that check hands.tsv schema (behavior changes at exact row level are allowed; only schema is asserted).

---

## Task 7: End-to-end verification via the web server JSON and dashboard

**Files:**
- Read only

- [ ] **Step 1: Compile and run the web server test suite**

Run: `sbt "testOnly sicfun.holdem.web.HandHistoryReviewServerTest"`
Expected: PASS — the existing fixture at line 186-189 already expects negative per-villain values; our change aligns the producer with that expectation.

- [ ] **Step 2: Manual smoke (optional, once ready): relaunch the dashboard and run a 1,000-hand 3-player hall**

Run: `scripts/packaged-hand-history-web/start-hand-history-web.ps1`
Then load `http://127.0.0.1:8080/#range`, trigger a hall run, and verify:
- Hero's "Net Chips" plus the two villains' "Per-Villain Net" values sum to ≈ 0.
- Losing villains show negative chip totals; winning villains show positive.

- [ ] **Step 3: Full project compile + full test pass**

Run: `sbt test`
Expected: PASS.

- [ ] **Step 4: Commit the implementation**

```bash
git add src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala
git commit -m "fix(playing-hall): compute real per-player P&L; continue hands past hero fold

- ShowdownResolution carries full Map[Position, Double] payouts, including the
  single-survivor case.
- play() resolves every active position via payout - contribution, not just hero.
- markFolded only terminates the hand when <=1 contestant remains, letting
  multi-way hands play out under villain decision policies after hero folds.
- recordOutcome aggregates perVillainNetChips using each villain's real net per
  hand rather than hero's net attributed to the primary-villain name.

Dashboard 'Per-Villain Net' now shows real chip flow. Sum(hero + villains) = 0."
```

---

## Self-Review Notes

- **Spec coverage:** every change requested by the user ("quién ganó y cuánto") is implemented via `perPositionNet` + `perVillainNet` rewrite (Tasks 3-6). Option B (continue hand past hero fold) is covered by Task 5.
- **No placeholders:** every code step shows full replacement bodies; no TBDs.
- **Type consistency:** `perPositionNet: Map[Position, Double]` is used with the same name/type in `HandResult` (Task 4) and in `recordOutcome` (Task 6).
- **Backward compat:** `heroNet` semantics and sign are preserved bit-identically; `ShowdownResolution.heroPayout(heroPosition)` kept as a convenience accessor for any future callers (currently only `play()` consumes it, and it does so via the full `payouts` map).
- **Risk:** post-hero-fold villain decisions will cause seeded integration runs to diverge at the exact-row level, but existing tests only assert schema presence, not row values — verified via grep.
