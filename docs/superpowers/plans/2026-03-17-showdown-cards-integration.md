# Showdown Cards Integration Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parse showdown-revealed opponent hole cards from hand histories, store them in data structures, and use them as hard Bayesian updates in range inference — both for batch profiling and live advisor sessions.

**Architecture:** Add `showdownCards: Map[String, HoleCards]` to `ImportedHand`, parse "shows" lines from PokerStars/Winamax/GGPoker hand histories, propagate through the profiling pipeline into `OpponentProfile`, and use revealed cards in `RangeInferenceEngine` as a hard constraint (likelihood = 1.0 for the shown hand, 0.0 for all others). Add a `showdown` command to the live advisor REPL.

**Tech Stack:** Scala 3.8.1, munit 1.2.2, existing `HoleCards`/`Card` types, `DiscreteDistribution[HoleCards]`

---

## Context for the Implementing Agent

### The Problem

The system discards showdown-revealed opponent cards at every layer:

| Component | Has Showdown Cards? | Impact |
|-----------|-------------------|--------|
| `ImportedHand` | Hero only (`heroHoleCards`) | Villain cards lost during parsing |
| `PokerEvent` | No | Cannot distinguish showed-down hands |
| `VillainObservation` | No | Range inference never gets hard evidence |
| `OpponentProfile` | No | Profiles built without card-action correlation |
| `AdvisorSession` | No | Live play can't input showdown reveals |
| `RangeInferenceEngine.inferPosterior` | No | Bayesian update uses only action likelihoods |

The parser currently stops at `*** SUMMARY ***` (line 292 of `HandHistoryImport.scala`). The `*** SHOW DOWN ***` section is between the last action and `*** SUMMARY ***`. It contains lines like:
```
*** SHOW DOWN ***
Villain: shows [Qh Qs] (a pair of Queens)
Hero: shows [Ac Kh] (high card Ace)
```

### Key Files You'll Touch

1. **`src/main/scala/sicfun/holdem/history/HandHistoryImport.scala`** — Parser. `ImportedHand` case class (line 52), `ImportState` (line 327), main parse loop (line 289-302). Add showdown parsing between `*** SHOW DOWN ***` and `*** SUMMARY ***`.

2. **`src/main/scala/sicfun/holdem/types/PokerEvent.scala`** — Event schema. You will NOT modify this (showdown is per-hand, not per-action). Showdown data belongs on `ImportedHand`.

3. **`src/main/scala/sicfun/holdem/engine/RangeInferenceEngine.scala`** — `VillainObservation` (line 16), `inferPosterior` (line 187). Add an optional `revealedCards: Option[HoleCards]` parameter.

4. **`src/main/scala/sicfun/holdem/history/OpponentProfileStore.scala`** — `OpponentProfile` case class, `fromImportedHands` builder. Add showdown card tracking for per-player analysis.

5. **`src/main/scala/sicfun/holdem/cli/AdvisorCommandParser.scala`** — Add `VillainShowdown(cards: HoleCards)` command variant.

6. **`src/main/scala/sicfun/holdem/runtime/AdvisorSession.scala`** — Handle the showdown command, log revealed cards, pass to range inference.

### Key Types

```scala
// src/main/scala/sicfun/holdem/types/HoldemTypes.scala
final case class HoleCards(card1: Card, card2: Card)

// src/main/scala/sicfun/holdem/engine/RangeInferenceEngine.scala:16
final case class VillainObservation(action: PokerAction, state: GameState)

// src/main/scala/sicfun/holdem/history/HandHistoryImport.scala:52
final case class ImportedHand(
    site: HandHistorySite,
    handId: String, tableName: String, startedAtEpochMillis: Long,
    buttonSeatNumber: Int, players: Vector[ImportedPlayer],
    heroName: Option[String], heroHoleCards: Option[HoleCards],
    events: Vector[PokerEvent]
)

// sicfun.core.Card — has Card.parse(str: String): Option[Card]
// sicfun.holdem.cli.CliHelpers — has parseHoleCards(token: String): HoleCards
```

### Codebase Conventions

- Scala 3 indentation syntax (no braces)
- `-Werror` is enabled — no warnings allowed (no non-local returns in `for` loops, etc.)
- `require(...)` for invariant enforcement in case classes
- Test framework: munit 1.2.2 (`extends FunSuite`, `test("name"): ...`, `assertEquals`, `assert`)
- Tests go in `src/test/scala/sicfun/holdem/` mirroring source structure

---

## Chunk 1: Parse and Store

### Task 1: Add showdown cards to ImportedHand

**Files:**
- Modify: `src/main/scala/sicfun/holdem/history/HandHistoryImport.scala:52-67` (case class) and `:288-316` (parse loop)
- Test: `src/test/scala/sicfun/holdem/history/HandHistoryImportTest.scala`

- [ ] **Step 1: Write the failing test**

Add a test that parses a PokerStars hand history containing a showdown section and asserts that `hand.showdownCards` contains the villain's revealed cards.

```scala
test("parseText extracts showdown cards"):
  val text = """PokerStars Hand #999: Hold'em No Limit ($1/$2) - 2025/01/01 12:00:00 ET
Table 'TestTable' 2-max Seat #1 is the button
Seat 1: Hero ($200 in chips)
Seat 2: Villain ($200 in chips)
Hero: posts small blind $1
Villain: posts big blind $2
*** HOLE CARDS ***
Dealt to Hero [Ac Kh]
Hero: raises $4 to $6
Villain: calls $4
*** FLOP *** [Ts 9h 8d]
Hero: bets $8
Villain: calls $8
*** TURN *** [Ts 9h 8d] [2c]
Hero: checks
Villain: checks
*** RIVER *** [Ts 9h 8d 2c] [3s]
Hero: checks
Villain: checks
*** SHOW DOWN ***
Villain: shows [Qh Qs] (a pair of Queens)
Hero: shows [Ac Kh] (high card Ace)
Villain collected $28 from pot
*** SUMMARY ***
Total pot $28 | Rake $0
"""
  val parsed = HandHistoryImport.parseText(text, Some(HandHistorySite.PokerStars), Some("Hero"))
  assert(parsed.isRight, s"parse failed: ${parsed.left.getOrElse("")}")
  val hand = parsed.toOption.get.head
  assertEquals(hand.showdownCards.get("Villain"), Some(HoleCards(Card(Rank.Queen, Suit.Hearts), Card(Rank.Queen, Suit.Spades))))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly sicfun.holdem.history.HandHistoryImportTest -- --tests=*showdown*"`
Expected: Compilation error — `showdownCards` doesn't exist on `ImportedHand`

- [ ] **Step 3: Add `showdownCards` field to `ImportedHand`**

In `HandHistoryImport.scala`, modify the `ImportedHand` case class:

```scala
final case class ImportedHand(
    site: HandHistorySite,
    handId: String,
    tableName: String,
    startedAtEpochMillis: Long,
    buttonSeatNumber: Int,
    players: Vector[ImportedPlayer],
    heroName: Option[String],
    heroHoleCards: Option[HoleCards],
    events: Vector[PokerEvent],
    showdownCards: Map[String, HoleCards] = Map.empty  // NEW: player name → revealed cards
)
```

Use `Map.empty` default so all existing call sites compile without changes.

- [ ] **Step 4: Parse showdown lines in the parse loop**

In the main parse loop (around line 288-302), change the logic to:
1. Recognize `*** SHOW DOWN ***` as a street marker (set a `showdownReached` flag)
2. Between `*** SHOW DOWN ***` and `*** SUMMARY ***`, parse lines matching the pattern `PlayerName: shows [XxYy]`
3. Collect into `showdownCards: Map[String, HoleCards]`

The showdown line format across sites:
- **PokerStars:** `Villain: shows [Qh Qs] (a pair of Queens)` — cards in brackets, description in parens
- **Winamax:** Similar but `shows` may vary
- **GGPoker:** `Villain Showed [Qh Qs]` — different capitalization

Parse approach:
```scala
// In the parse loop, between SHOW DOWN and SUMMARY:
val ShowdownPattern = """^(.+?):\s+shows?\s+\[([^\]]+)\]""".r  // greedy name, then "shows [cards]"

// When showdownReached && !summaryReached:
line match
  case ShowdownPattern(name, cardsStr) =>
    val cards = cardsStr.trim.split("\\s+").flatMap(Card.parse)
    if cards.length == 2 then
      showdownMap += (name.trim -> HoleCards(cards(0), cards(1)))
  case _ => () // ignore other showdown section lines (collected pot, etc.)
```

**IMPORTANT:** The regex needs to handle player names that contain colons or spaces. PokerStars player names can't contain `[` so splitting on `: shows [` or `: shows [` is safe.

- [ ] **Step 5: Wire showdown map into `ImportedHand` construction**

At line 304-316 where `ImportedHand(...)` is constructed, add the collected `showdownCards` map:

```scala
Right(
  ImportedHand(
    site = site,
    handId = handId,
    // ... existing fields ...
    events = state.events,
    showdownCards = showdownMap.toMap  // NEW
  )
)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `sbt "testOnly sicfun.holdem.history.HandHistoryImportTest -- --tests=*showdown*"`
Expected: PASS

- [ ] **Step 7: Add edge case tests**

```scala
test("parseText handles hand with no showdown (fold before river)"):
  // Hand where villain folds preflop — showdownCards should be empty
  val text = """PokerStars Hand #998: Hold'em No Limit ($1/$2) - 2025/01/01 12:00:00 ET
Table 'TestTable' 2-max Seat #1 is the button
Seat 1: Hero ($200 in chips)
Seat 2: Villain ($200 in chips)
Hero: posts small blind $1
Villain: posts big blind $2
*** HOLE CARDS ***
Dealt to Hero [Ac Kh]
Hero: raises $4 to $6
Villain: folds
Hero collected $4 from pot
*** SUMMARY ***
Total pot $4 | Rake $0
"""
  val parsed = HandHistoryImport.parseText(text, Some(HandHistorySite.PokerStars), Some("Hero"))
  assert(parsed.isRight)
  val hand = parsed.toOption.get.head
  assert(hand.showdownCards.isEmpty)

test("parseText parses showdown with mucked hand (no show)"):
  // Villain mucks — only hero shown. "Villain: mucks hand" should NOT be in showdownCards.
  // Only "shows" lines produce entries.
  ...
```

- [ ] **Step 8: Commit**

```bash
git add src/main/scala/sicfun/holdem/history/HandHistoryImport.scala
git add src/test/scala/sicfun/holdem/history/HandHistoryImportTest.scala
git commit -m "feat: parse showdown-revealed hole cards into ImportedHand.showdownCards"
```

---

### Task 2: Propagate showdown cards into OpponentProfile

**Files:**
- Modify: `src/main/scala/sicfun/holdem/history/OpponentProfileStore.scala` (`OpponentProfile` case class + `fromImportedHands`)
- Test: `src/test/scala/sicfun/holdem/history/OpponentProfileStorePersistenceTest.scala` or new test

- [ ] **Step 1: Write the failing test**

```scala
test("OpponentProfile tracks showdown-revealed hands"):
  // Build hands where villain shows cards at showdown
  // Assert that profile contains the revealed cards
  val hands = ... // use HandHistoryImport.parseText with showdown text
  val profiles = OpponentProfile.fromImportedHands("test", hands, Set("Hero"))
  val profile = profiles.head
  assert(profile.showdownHands.nonEmpty, "should have showdown cards")
  assertEquals(profile.showdownHands.head.cards, expectedCards)
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Add showdown tracking to OpponentProfile**

Add to `OpponentProfile`:
```scala
final case class ShowdownRecord(
    handId: String,
    cards: HoleCards,
    finalStreet: Street,     // what street the hand reached
    actionLine: Vector[PokerAction]  // what actions the player took
)

final case class OpponentProfile(
    // ... existing fields ...
    showdownHands: Vector[ShowdownRecord] = Vector.empty  // NEW
)
```

- [ ] **Step 4: Populate in `fromImportedHands`**

In the `fromImportedHands` method (around line 140-159), after iterating events, check `hand.showdownCards` for each player:

```scala
// After event iteration for this hand:
hand.showdownCards.foreach { (playerName, cards) =>
  if !excludePlayers.contains(playerName) then
    val builder = builders.getOrElseUpdate(playerName, new ProfileBuilder(site, playerName))
    val playerEvents = hand.events.filter(_.playerId == playerName)
    val finalStreet = playerEvents.lastOption.map(_.street).getOrElse(Street.Preflop)
    val actions = playerEvents.map(_.action)
    builder.addShowdownRecord(hand.handId, cards, finalStreet, actions)
}
```

- [ ] **Step 5: Run test to verify it passes**

- [ ] **Step 6: Commit**

```bash
git add src/main/scala/sicfun/holdem/history/OpponentProfileStore.scala
git add src/test/scala/sicfun/holdem/history/OpponentProfileStorePersistenceTest.scala
git commit -m "feat: track showdown-revealed hands in OpponentProfile"
```

---

## Chunk 2: Range Inference Integration

### Task 3: Use showdown cards as hard Bayesian update in RangeInferenceEngine

**Files:**
- Modify: `src/main/scala/sicfun/holdem/engine/RangeInferenceEngine.scala` (`inferPosterior` method)
- Test: `src/test/scala/sicfun/holdem/engine/RangeInferenceEngineTest.scala` or new file

**Key concept:** When villain's cards are known from a showdown, the posterior is a delta distribution — probability 1.0 for the revealed hand, 0.0 for everything else. This is the strongest possible Bayesian update.

- [ ] **Step 1: Write the failing test**

```scala
test("inferPosterior with revealed cards produces delta distribution"):
  val hero = HoleCards(Card(Rank.Ace, Suit.Hearts), Card(Rank.King, Suit.Hearts))
  val board = Board.from(Vector(
    Card(Rank.Ten, Suit.Spades), Card(Rank.Nine, Suit.Hearts), Card(Rank.Eight, Suit.Diamonds)
  ))
  val revealed = HoleCards(Card(Rank.Queen, Suit.Hearts), Card(Rank.Queen, Suit.Spades))
  val result = RangeInferenceEngine.inferPosterior(
    hero = hero, board = board,
    folds = Vector.empty, tableRanges = TableRanges.defaults(TableFormat.HeadsUp),
    villainPos = Position.BigBlind, observations = Seq.empty,
    actionModel = PokerActionModel.uniform,
    revealedCards = Some(revealed),  // NEW PARAMETER
    useCache = false
  )
  // The posterior should be concentrated on the revealed hand
  val prob = result.posterior.probability(revealed)
  assert(prob > 0.99, s"expected ~1.0 for revealed hand, got $prob")
```

- [ ] **Step 2: Run test to verify it fails**

Expected: Compilation error — no `revealedCards` parameter

- [ ] **Step 3: Add `revealedCards` parameter to `inferPosterior`**

```scala
def inferPosterior(
    hero: HoleCards,
    board: Board,
    folds: Vector[PreflopFold],
    tableRanges: TableRanges,
    villainPos: Position,
    observations: Seq[VillainObservation],
    actionModel: PokerActionModel,
    bunchingTrials: Int = 10_000,
    rng: Random = new Random(),
    useCache: Boolean = true,
    revealedCards: Option[HoleCards] = None  // NEW — showdown revealed cards
): PosteriorInferenceResult =
```

- [ ] **Step 4: Implement delta distribution shortcut**

At the top of the method body, before the cache/compute logic:

```scala
// If cards are revealed (showdown), posterior is a delta distribution.
// No need to run Bayesian inference — we know the exact hand.
revealedCards match
  case Some(cards) =>
    val delta = DiscreteDistribution.point(cards)
    return PosteriorInferenceResult(
      posterior = delta,
      collapse = PosteriorCollapse(
        entropyReduction = Double.PositiveInfinity,
        klDivergence = Double.PositiveInfinity,
        effectiveSupportPrior = 1326.0,
        effectiveSupportPosterior = 1.0,
        collapseRatio = 1.0 / 1326.0
      )
    )
  case None => () // proceed with normal inference
```

**NOTE:** Check if `DiscreteDistribution.point(x)` exists. If not, construct it as:
```scala
val delta = DiscreteDistribution(Map(cards -> Probability(1.0)))
```
Check `sicfun.core.DiscreteDistribution` for available factory methods.

- [ ] **Step 5: Run test to verify it passes**

- [ ] **Step 6: Add test for non-revealed case (regression)**

```scala
test("inferPosterior without revealed cards works as before"):
  // Same call without revealedCards — should produce normal posterior
  val result = RangeInferenceEngine.inferPosterior(
    hero = hero, board = board,
    folds = Vector.empty, tableRanges = tableRanges,
    villainPos = Position.BigBlind, observations = Seq.empty,
    actionModel = PokerActionModel.uniform,
    useCache = false
  )
  // Posterior should be spread across many hands, not concentrated
  assert(result.posterior.support.size > 10, "should have broad support")
```

- [ ] **Step 7: Commit**

```bash
git add src/main/scala/sicfun/holdem/engine/RangeInferenceEngine.scala
git add src/test/scala/sicfun/holdem/engine/...
git commit -m "feat: use showdown-revealed cards as hard Bayesian update in range inference"
```

---

## Chunk 3: Live Advisor Integration

### Task 4: Add `showdown` command to AdvisorSession

**Files:**
- Modify: `src/main/scala/sicfun/holdem/cli/AdvisorCommandParser.scala` (add `VillainShowdown` command)
- Modify: `src/main/scala/sicfun/holdem/runtime/AdvisorSession.scala` (handle command)
- Test: `src/test/scala/sicfun/holdem/cli/AdvisorCommandParserTest.scala`
- Test: `src/test/scala/sicfun/holdem/runtime/AdvisorSessionTest.scala`

- [ ] **Step 1: Write parser test**

```scala
test("parse 'v show QhQs' as VillainShowdown"):
  val cmd = AdvisorCommandParser.parse("v show QhQs")
  cmd match
    case AdvisorCommand.VillainShowdown(cards) =>
      assertEquals(cards, HoleCards(Card(Rank.Queen, Suit.Hearts), Card(Rank.Queen, Suit.Spades)))
    case other => fail(s"expected VillainShowdown, got $other")

test("parse 'v show Qh Qs' (spaced) as VillainShowdown"):
  val cmd = AdvisorCommandParser.parse("v show Qh Qs")
  cmd match
    case AdvisorCommand.VillainShowdown(cards) =>
      assertEquals(cards, HoleCards(Card(Rank.Queen, Suit.Hearts), Card(Rank.Queen, Suit.Spades)))
    case other => fail(s"expected VillainShowdown, got $other")
```

- [ ] **Step 2: Run to verify it fails**

- [ ] **Step 3: Add `VillainShowdown` to `AdvisorCommand` enum**

In `AdvisorCommandParser.scala`:
```scala
enum AdvisorCommand:
  // ... existing cases ...
  case VillainShowdown(cards: sicfun.holdem.types.HoleCards)  // NEW
```

- [ ] **Step 4: Add parser case for "v show"**

In `parseVillainSub`:
```scala
case "show" | "shows" | "showdown" =>
  if tokens.tail.isEmpty then
    AdvisorCommand.Unknown(s"v ${tokens.mkString(" ")}", "showdown requires cards (e.g., v show QhQs)")
  else
    val cardStr = tokens.tail.mkString("")
    tryParseHoleCards(cardStr) match
      case Some(hc) => AdvisorCommand.VillainShowdown(hc)
      case None => AdvisorCommand.Unknown(s"v ${tokens.mkString(" ")}", s"invalid hole cards: ${tokens.tail.mkString(" ")}")
```

Note: `tryParseHoleCards` is a private method already in `AdvisorCommandParser` (line 131-133).

- [ ] **Step 5: Run parser test to verify it passes**

- [ ] **Step 6: Write session handler test**

```scala
test("showdown command records villain cards and updates range"):
  // Set up a session mid-hand, execute VillainShowdown
  // Assert output message confirms the cards
  // Assert villain posterior collapses to the revealed hand
```

- [ ] **Step 7: Handle `VillainShowdown` in `AdvisorSession.execute`**

In `AdvisorSession.scala`, add case to `execute`:
```scala
case AdvisorCommand.VillainShowdown(cards) => doVillainShowdown(cards)
```

Implement `doVillainShowdown`:
```scala
private def doVillainShowdown(cards: HoleCards): CommandResult =
  hand match
    case None =>
      CommandResult(this, Vector("No hand in progress. Use 'new' to start a hand."))
    case Some(h) =>
      val out = Vector(
        s"Villain showed ${cards.card1.toChar}${cards.card2.toChar}.",
        "Range posterior collapsed to revealed hand."
      )
      // Store for future use — the next Advise/Review call should see the delta posterior.
      // The revealed cards override any action-based inference.
      val updated = this.copy(
        hand = Some(h.copy(villainRevealedCards = Some(cards)))
      )
      CommandResult(updated, out)
```

This requires adding `villainRevealedCards: Option[HoleCards] = None` to `HandSnapshot` (check the exact name — it's whatever `hand: Option[...]` holds in `AdvisorSession`).

- [ ] **Step 8: Wire revealed cards into advise/review path**

In `doAdvise` and `doReview`, when calling `RangeInferenceEngine.inferPosterior`, pass `revealedCards = h.villainRevealedCards`.

- [ ] **Step 9: Update help text**

In `doHelp()`, add line for the showdown command:
```
v show QhQs         — Record villain's showdown cards (collapses range)
```

- [ ] **Step 10: Run all tests**

Run: `sbt "testOnly sicfun.holdem.cli.AdvisorCommandParserTest sicfun.holdem.runtime.AdvisorSessionTest"`

- [ ] **Step 11: Commit**

```bash
git add src/main/scala/sicfun/holdem/cli/AdvisorCommandParser.scala
git add src/main/scala/sicfun/holdem/runtime/AdvisorSession.scala
git add src/test/scala/sicfun/holdem/cli/AdvisorCommandParserTest.scala
git add src/test/scala/sicfun/holdem/runtime/AdvisorSessionTest.scala
git commit -m "feat: add 'v show' command for villain showdown cards in advisor REPL"
```

---

### Task 5: Integration test — showdown cards through full pipeline

**Files:**
- Create: `src/test/scala/sicfun/holdem/history/ShowdownIntegrationTest.scala`

- [ ] **Step 1: Write integration test**

```scala
test("showdown cards flow through import → profile → range inference"):
  // 1. Parse a hand history with showdown
  // 2. Build opponent profile
  // 3. Assert showdownHands populated
  // 4. Use the profile's showdown data in range inference
  // 5. Assert delta posterior
```

- [ ] **Step 2: Run and verify pass**

- [ ] **Step 3: Commit**

```bash
git add src/test/scala/sicfun/holdem/history/ShowdownIntegrationTest.scala
git commit -m "test: integration test for showdown cards through full pipeline"
```

---

## Important Notes for the Implementer

1. **`DiscreteDistribution` API:** Before implementing the delta distribution, read `src/main/scala/sicfun/core/DiscreteDistribution.scala` to understand the API. Look for factory methods like `point()`, `single()`, or just use the `Map` constructor.

2. **`HandSnapshot` type:** Read `src/main/scala/sicfun/holdem/runtime/AdvisorSession.scala` (around lines 20-80) to find the exact type used for `hand: Option[???]`. You'll need to add `villainRevealedCards: Option[HoleCards] = None` to whatever that type is.

3. **`Card.parse` returns `Option[Card]`:** The existing parser uses `Card.parse(str)` which returns `Option[Card]`. The showdown regex should extract the card string, split by whitespace, and parse each.

4. **Existing `heroHoleCards`:** The parser already captures hero's cards from `Dealt to Hero [Ac Kh]` lines (line 284). The showdown parser adds OPPONENT cards from `shows [...]` lines. Hero's showdown line is redundant (we already know hero's cards) — parse it anyway but it's low priority.

5. **Muck vs Show:** Players who muck (don't show) should NOT appear in `showdownCards`. Only parse lines containing "shows" or "Showed". Lines like `Villain: mucks hand` should be ignored.

6. **Backward compatibility:** All new fields use defaults (`Map.empty`, `Vector.empty`, `None`) so existing code compiles and works unchanged. No existing tests should break.

7. **`-Werror`:** The project treats all warnings as errors. Avoid deprecated features like non-local `return` inside lambda/`for` blocks. Use early-return only in methods, or rewrite with `match`/functional style.

8. **Git branch:** Create a feature branch `feat/showdown-cards-integration` from `master` (or from `feat/cfr-gto-villain-calibration` if it's been merged).
