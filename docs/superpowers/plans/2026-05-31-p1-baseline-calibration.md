# P1 — Calibrated, Spot-Conditioned Baseline (Def 9/10) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the board/street-blind live baseline (`config.actionPriors.getOrElse((cls,cat),0.25)` — 16 hardcoded constants) with a calibrated, board/street-conditioned `RealBaseline` (Def 9), feeding the ref/attrib tempered likelihoods, while guaranteeing "never worse than today" via a backoff ladder to the existing constants.

**Architecture:** Offline calibration tool ingests a hand-history corpus → for each **showdown-revealed** decision, classifies `(revealed holding, board, street)` → `StrategicClass` (reusing `ClassificationBridge.classify` over `HoldemEquity`) → tallies raw counts keyed `(class × board-bucket × street × action)` → writes a versioned count artifact. At runtime, `RealBaselineImpl` loads the artifact, builds smoothed distributions, and answers `probability(...)` with a backoff ladder: `(class,bucket,street,action)` → `(class,street,action)` → `(class,action)` constants floor (`ConstantRealBaseline`) → uniform. Wiring is gated: with **no** `baselinePath` configured, behavior is **byte-identical to today**.

**Tech Stack:** Scala 3.8.1, SBT, munit 1.2.2. Artifact = Java `Properties` + TSV (mirrors `PokerActionModelArtifactIO`, no new deps). Reuses existing `HoldemEquity`, `ClassificationBridge`, `HandHistoryImport`.

---

## Source spec & audit grounding

- Design spec: `docs/superpowers/specs/2026-05-30-p1-baseline-calibration-design.md` (this plan resolves its "Open decisions").
- Algebraic source of truth: `docs/specs/SICFUN-v0_31_1-corrected.md` Def 9 (real baseline), Def 10 (attributed baseline).
- Audit finding this fixes: failure mode #6 — "Uncalibrated/un-conditioned anchor: real baseline (Def 9/10) = 16 hardcoded constants keyed only on (class,action), NOT board/street-conditioned, NEVER calibrated — and it's LIVE."

## Verified current-state anchors (recon 2026-05-31, branch feat/p0-winrate-harness)

| Symbol | Location | Note |
|---|---|---|
| `RealBaseline` trait (abstract, unimplemented) | `src/main/scala/sicfun/holdem/strategic/safety/Baseline.scala:13-25` | `probability(cls, action: PokerAction.Category, sizing: Option[Sizing], publicState: PublicState): Double` |
| `AttributedBaseline` trait | `Baseline.scala:34-41` | adds `rivalState: RivalBeliefState` |
| `PosteriorAttributedBaseline` | `src/main/scala/sicfun/holdem/strategic/PosteriorAttributedBaseline.scala:21-52` | wraps `actionPriors`; `pi0 = getOrElse(...,0.25)`; **ignores `publicState`** |
| 16 constants `defaultActionPriors` | `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala:1280-1292` | key `(StrategicClass, PokerAction.Category)` |
| `actionPrior` helper | `StrategicEngine.scala:1144-1145` | `config.actionPriors.getOrElse((cls,cat),0.25)` |
| `buildAttribLikelihoodFromBaseline` | `StrategicEngine.scala:1152-1166` | already passes `pubState` into `baseline.probability(...)` |
| `buildAttribLikelihoodFn` | `StrategicEngine.scala:1172-1173` | `= buildAttribLikelihoodFromBaseline(_attributedBaseline)` |
| `buildRefLikelihoodFn` | `StrategicEngine.scala:1178-1191` | board-blind: uses `actionPrior(cls, signal.action)` |
| `_attributedBaseline` construction | `StrategicEngine.scala:32-33` | `new PosteriorAttributedBaseline(config.actionPriors)` |
| `StrategicEngine.Config` | `StrategicEngine.scala:1295-1328` | **no path field exists**; add `baselinePath` |
| `ClassificationBridge.classify` | `src/main/scala/sicfun/holdem/strategic/bridge/ClassificationBridge.scala:29-39` | `(equity: Double, hasDrawPotential: Boolean, thresholds): BridgeResult[StrategicClass]`; Value≥0.65, Bluff<0.35, else draw?StructuralBluff:Mixed |
| `HoldemEquity.equityExact` | `src/main/scala/sicfun/holdem/equity/HoldemEquity.scala:226-262` | `(hero: HoleCards, board: Board, villainRange: DiscreteDistribution[HoleCards]): EquityResult`; sanitizes blocked cards internally |
| `EquityResult.equity` | `src/main/scala/sicfun/holdem/types/HoldemTypes.scala:127` | `= win + tie/2.0` |
| `ImportedHand` | `src/main/scala/sicfun/holdem/history/HandHistoryImport.scala:95-107` | has `events: Vector[PokerEvent]`, `showdownCards: Map[String,HoleCards]`, `bigBlind: Double` |
| `PokerEvent` | `src/main/scala/sicfun/holdem/types/PokerEvent.scala:59-101` | per-decision: `playerId, street, board, action: PokerAction` |
| `HandHistoryImport.parseFile/parseText` | `HandHistoryImport.scala:198-232` | `: Either[String, Vector[ImportedHand]]` |
| `PokerActionModelArtifactIO` (pattern to mirror) | `src/main/scala/sicfun/holdem/model/PokerActionModelArtifactIO.scala:58-146` | Properties + TSV; `save(dir, artifact)` / `load(dir)` |
| `TrainPokerActionModel` (CLI pattern to mirror) | `src/main/scala/sicfun/holdem/model/TrainPokerActionModel.scala:52-152` | `object{ def main(args) }`; `CliHelpers.parseOptionsAllowBlankValues` |
| `Card` / `Rank` / `Suit` | `src/main/scala/sicfun/core/Card.scala:8,44,113` | `Card(rank: Rank, suit: Suit)`; `Rank(val value: Int)` Two=2..Ace=14; `Suit{Clubs,Diamonds,Hearts,Spades}` |
| `Street` | `src/main/scala/sicfun/holdem/types/GameState.scala:38-46` | `Preflop, Flop, Turn, River` |
| `PublicState` | `src/main/scala/sicfun/holdem/strategic/state/AugmentedState.scala:13-19` | `street: Street`, `board: Board` |
| `Sizing` | `src/main/scala/sicfun/holdem/strategic/state/Signal.scala:6` | `final case class Sizing(...)` |
| `StrategicClass` | `src/main/scala/sicfun/holdem/strategic/types/StrategicClass.scala:12-16` | `Value, Bluff, Mixed, StructuralBluff` |

## Resolved open decisions (were "Open decisions for the plan" in the spec)

1. **Board-bucket granularity.** Postflop bucket = `Pairing{Unpaired, Paired, Trips} × Suitedness{Rainbow, TwoTone, Monotone} × HighCard{AceHigh, Broadway, Middle, Low}` (36 textures) × `Street{Flop, Turn, River}`. **Preflop = a single `Preflop` token** (see #6). Token format: `"<Pairing>-<Suitedness>-<HighCard>"` postflop, `"PRE"` preflop.
2. **`minCount` backoff threshold = 30** (a `(class,bucket,street)` group needs ≥30 observed decisions to be used directly; else back off). Stored in metadata, applied at load (tunable without recalibration).
3. **Smoothing = Laplace add-α, α = 1.0** over the 4 action categories within a group. Stored in metadata, applied at load.
4. **Artifact format = directory of `baseline-counts.tsv` + `metadata.properties`** (mirrors `PokerActionModelArtifactIO`; no new dependency). The TSV stores **raw counts**; all smoothing/backoff is computed at load (one place, tunable).
5. **Calibration corpus = operator-supplied real histories (stay local, passed as a CLI arg) + a small checked-in synthetic showdown corpus for tests** (`src/test/resources/handhistory/p1-showdown-sample.txt`, created in Task 5 — the existing P0 sample has zero showdowns).
6. **Non-showdown decisions: NOT class-labeled** (spec default). **Preflop decisions: NOT per-class calibrated in v1** — exact equity-vs-random is infeasible preflop, and the spec's value is postflop board-texture conditioning. Preflop falls through the backoff to the `(class,action)` constants floor (i.e., **exactly today's behavior preflop**). Documented limitation; preflop/position conditioning is a follow-up.

## Scope guards (firmly OUT)

- **No** change to overlay penalty/veto logic, kernel/inference math, or `StrategicClass` definitions.
- **`PokerPftFormulation.scala:109-110`** also reads `actionPriors.getOrElse(...,0.25)` — it is **OUT of P1 scope** (separate formulation, not Def 9/10). Leave on constants; do not touch.
- Fine-grained sizing (λ) conditioning: deferred. `sizing` is accepted in the `RealBaseline` signature but **bucketed to the action category** for v1 (i.e., ignored in the lookup key).
- The CRN/duplicate-dealing variance reduction (needed for a *cheap* G3 capture) is a separate harness task, not part of P1.

## Validation gates (from the spec)

- **G1 (baseline-first):** the P0 harness G1 baseline must be captured (deterministically, on the pre-P1 commit) before this merges. Capture timing is decoupled by hall determinism — see the resume notes; do **not** block this implementation on it, but do **not merge** P1 until G1 exists.
- **G3 (quality):** after wiring, re-run the P0 harness; calibrated baseline must **preserve-or-improve** bb/100 vs G1. Regression ⇒ investigate, not auto-ship.
- **G5 (diagnostics):** backoff floor guarantees a valid distribution in every spot (no silent zero/degenerate baseline).
- **Determinism:** fixed artifact ⇒ deterministic lookups (no RNG at runtime; calibration equity is exact, not Monte Carlo).
- **No-regression (the safety contract):** with `baselinePath = None`, the ref + attrib likelihoods are **byte-identical to today**, because `ConstantRealBaseline(actionPriors).probability(cls, action, _, _) == actionPriors.getOrElse((cls,action), 0.25)`.

## File structure

**Create (runtime + shared, under `strategic/safety/`):**
- `src/main/scala/sicfun/holdem/strategic/safety/BoardBucket.scala` — pure board-texture bucketing. Shared by the impl (lookup) and the tool (tally).
- `src/main/scala/sicfun/holdem/strategic/safety/BaselineArtifact.scala` — `BaselineArtifact` + `BaselineMetadata` data types + `BaselineArtifactIO` (save/load).
- `src/main/scala/sicfun/holdem/strategic/safety/RealBaselineImpl.scala` — `ConstantRealBaseline` (floor) + `RealBaselineImpl` (smoothed backoff).

**Create (offline calibration, under `strategic/calibration/`):**
- `src/main/scala/sicfun/holdem/strategic/calibration/HoldingClassifier.scala` — `(HoleCards, Board, Street) → StrategicClass` via equity + draw + `ClassificationBridge`.
- `src/main/scala/sicfun/holdem/strategic/calibration/BaselineCalibrationTool.scala` — CLI: corpus → counts → artifact.

**Modify:**
- `src/main/scala/sicfun/holdem/strategic/PosteriorAttributedBaseline.scala` — constructor takes a `RealBaseline` instead of the raw map.
- `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala` — add `Config.baselinePath`; build `_realBaseline`; rewire `_attributedBaseline` + `buildRefLikelihoodFn`.

**Create (test resource):**
- `src/test/resources/handhistory/p1-showdown-sample.txt` — synthetic 9-max PokerStars hands **with showdowns** (revealed holdings).

**Test files (one per unit):**
- `src/test/scala/sicfun/holdem/strategic/safety/BoardBucketTest.scala`
- `src/test/scala/sicfun/holdem/strategic/safety/ConstantRealBaselineTest.scala`
- `src/test/scala/sicfun/holdem/strategic/safety/BaselineArtifactIOTest.scala`
- `src/test/scala/sicfun/holdem/strategic/safety/RealBaselineImplTest.scala`
- `src/test/scala/sicfun/holdem/history/P1ShowdownSampleTest.scala`
- `src/test/scala/sicfun/holdem/strategic/calibration/HoldingClassifierTest.scala`
- `src/test/scala/sicfun/holdem/strategic/calibration/BaselineCalibrationToolTest.scala`
- `src/test/scala/sicfun/holdem/strategic/BaselineWiringTest.scala`

---

## Task 1: `BoardBucket` — board-texture bucketing

**Files:**
- Create: `src/main/scala/sicfun/holdem/strategic/safety/BoardBucket.scala`
- Test: `src/test/scala/sicfun/holdem/strategic/safety/BoardBucketTest.scala`

- [ ] **Step 1: Write the failing test**

```scala
package sicfun.holdem.strategic.safety

import munit.FunSuite
import sicfun.core.Card
import sicfun.holdem.types.{Board, Street}
import sicfun.holdem.strategic.safety.BoardBucket.*

class BoardBucketTest extends FunSuite:
  private def board(tokens: String*): Board = Board(tokens.toVector.map(t => Card.parse(t).get))

  test("preflop board → PRE token regardless of street"):
    assertEquals(BoardBucket.ofBoard(Street.Preflop, Board.empty), BoardBucket.Preflop)
    assertEquals(BoardBucket.token(BoardBucket.Preflop), "PRE")

  test("unpaired rainbow ace-high flop"):
    val b = BoardBucket.ofBoard(Street.Flop, board("As", "Kd", "7c"))
    assertEquals(b, BoardBucket.Postflop(Pairing.Unpaired, Suitedness.Rainbow, HighCard.AceHigh))
    assertEquals(BoardBucket.token(b), "Unpaired-Rainbow-AceHigh")

  test("paired rainbow broadway flop"):
    val b = BoardBucket.ofBoard(Street.Flop, board("Ks", "Kh", "Qd"))
    assertEquals(b, BoardBucket.Postflop(Pairing.Paired, Suitedness.Rainbow, HighCard.Broadway))

  test("monotone middle flop"):
    val b = BoardBucket.ofBoard(Street.Flop, board("9h", "7h", "5h"))
    assertEquals(b, BoardBucket.Postflop(Pairing.Unpaired, Suitedness.Monotone, HighCard.Middle))

  test("trips low flop"):
    val b = BoardBucket.ofBoard(Street.Flop, board("4h", "4d", "4c"))
    assertEquals(b, BoardBucket.Postflop(Pairing.Trips, Suitedness.Rainbow, HighCard.Low))

  test("turn two-tone (4 cards, exactly two of a suit is NOT two-tone; ≥2 max suit on partial board)"):
    // suitedness on a 4-card board: max suit count 2 → TwoTone
    val b = BoardBucket.ofBoard(Street.Turn, board("As", "Kd", "7c", "2s"))
    assertEquals(b.asInstanceOf[BoardBucket.Postflop].suitedness, Suitedness.TwoTone)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly sicfun.holdem.strategic.safety.BoardBucketTest"`
Expected: FAIL — `BoardBucket` not found / does not compile.

- [ ] **Step 3: Write the implementation**

```scala
package sicfun.holdem.strategic.safety

import sicfun.core.Card
import sicfun.holdem.types.{Board, Street}
import sicfun.holdem.strategic.state.PublicState

/** Coarse board-texture bucket for baseline conditioning (P1).
  *
  * Postflop: pairing × suitedness × high-card. Preflop: a single `Preflop` token (v1 —
  * preflop is not per-class calibrated; it backs off to the (class,action) constants).
  * Deliberately coarse to keep artifact cells populated.
  */
enum BoardBucket:
  case Preflop
  case Postflop(
      pairing: BoardBucket.Pairing,
      suitedness: BoardBucket.Suitedness,
      highCard: BoardBucket.HighCard
  )

object BoardBucket:
  enum Pairing:    case Unpaired, Paired, Trips
  enum Suitedness: case Rainbow, TwoTone, Monotone
  enum HighCard:   case AceHigh, Broadway, Middle, Low

  /** Stable token used as the artifact key segment and backoff key. */
  def token(b: BoardBucket): String = b match
    case Preflop                 => "PRE"
    case Postflop(p, s, h)       => s"$p-$s-$h"

  /** Bucket from a full public state (uses its street + board). */
  def of(publicState: PublicState): BoardBucket =
    ofBoard(publicState.street, publicState.board)

  def ofBoard(street: Street, board: Board): BoardBucket =
    if street == Street.Preflop || board.cards.isEmpty then Preflop
    else Postflop(pairingOf(board.cards), suitednessOf(board.cards), highCardOf(board.cards))

  private def pairingOf(cards: Vector[Card]): Pairing =
    val maxRankCount = cards.groupBy(_.rank).values.map(_.size).maxOption.getOrElse(0)
    if maxRankCount >= 3 then Pairing.Trips
    else if maxRankCount == 2 then Pairing.Paired
    else Pairing.Unpaired

  private def suitednessOf(cards: Vector[Card]): Suitedness =
    val maxSuitCount = cards.groupBy(_.suit).values.map(_.size).maxOption.getOrElse(0)
    if maxSuitCount >= 3 then Suitedness.Monotone
    else if maxSuitCount == 2 then Suitedness.TwoTone
    else Suitedness.Rainbow

  private def highCardOf(cards: Vector[Card]): HighCard =
    val top = cards.map(_.rank.value).maxOption.getOrElse(0)
    if top >= 14 then HighCard.AceHigh        // Ace
    else if top >= 11 then HighCard.Broadway  // J,Q,K (Ten counts as Middle)
    else if top >= 7 then HighCard.Middle     // 7..10
    else HighCard.Low                          // 2..6
```

- [ ] **Step 4: Run test to verify it passes**

Run: `sbt "testOnly sicfun.holdem.strategic.safety.BoardBucketTest"`
Expected: PASS (all cases). Note: `Suitedness` on a partial (3–4 card) board uses the **max same-suit count** so flush *texture* is captured progressively; this is intentional and coarse.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/strategic/safety/BoardBucket.scala src/test/scala/sicfun/holdem/strategic/safety/BoardBucketTest.scala
git commit -m "feat(baseline): BoardBucket coarse board-texture bucketing for Def 9 conditioning"
```

---

## Task 2: `ConstantRealBaseline` — the board-blind floor (today's constants)

**Files:**
- Create: `src/main/scala/sicfun/holdem/strategic/safety/RealBaselineImpl.scala` (this task adds `ConstantRealBaseline`; Task 4 adds `RealBaselineImpl` to the same file)
- Test: `src/test/scala/sicfun/holdem/strategic/safety/ConstantRealBaselineTest.scala`

- [ ] **Step 1: Write the failing test**

```scala
package sicfun.holdem.strategic.safety

import munit.FunSuite
import sicfun.holdem.types.{Board, PokerAction, Street}
import sicfun.holdem.strategic.types.StrategicClass
import sicfun.holdem.strategic.state.PublicState
import sicfun.holdem.engine.StrategicEngine
import sicfun.holdem.types.{Chips, TableMap}

class ConstantRealBaselineTest extends FunSuite:
  private val ps = PublicState(Street.Flop, Board.empty, Chips(0.0), TableMap.empty, Vector.empty)
  private val base = ConstantRealBaseline(StrategicEngine.defaultActionPriors)

  test("returns the configured constant for a known (class,action)"):
    // defaultActionPriors: (Value, Raise) -> 0.20
    assertEquals(base.probability(StrategicClass.Value, PokerAction.Category.Raise, None, ps), 0.20)
    // (Bluff, Raise) -> 0.65
    assertEquals(base.probability(StrategicClass.Bluff, PokerAction.Category.Raise, None, ps), 0.65)

  test("falls back to 0.25 for a key absent from the map"):
    val sparse = ConstantRealBaseline(Map.empty)
    assertEquals(sparse.probability(StrategicClass.Mixed, PokerAction.Category.Call, None, ps), 0.25)

  test("ignores board/street/sizing (board-blind floor)"):
    val psRiver = PublicState(Street.River, Board.empty, Chips(0.0), TableMap.empty, Vector.empty)
    assertEquals(
      base.probability(StrategicClass.Value, PokerAction.Category.Check, None, ps),
      base.probability(StrategicClass.Value, PokerAction.Category.Check, None, psRiver)
    )
```

> The implementer must confirm the `PublicState` construction args (`Chips`, `TableMap.empty`) against `AugmentedState.scala:13-19` and adjust the test fixture if the empty constructors differ; the assertions are the contract.

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly sicfun.holdem.strategic.safety.ConstantRealBaselineTest"`
Expected: FAIL — `ConstantRealBaseline` not found.

- [ ] **Step 3: Write the implementation** (new file; `RealBaselineImpl` added in Task 4)

```scala
package sicfun.holdem.strategic.safety

import sicfun.holdem.types.PokerAction
import sicfun.holdem.strategic.types.StrategicClass
import sicfun.holdem.strategic.state.{PublicState, Sizing}

/** Board-blind floor implementation of the Def 9 real baseline: returns the
  * configured `(class, action)` constant, or 0.25 when absent. This reproduces
  * EXACTLY the pre-P1 behavior (`actionPriors.getOrElse((cls,cat), 0.25)`), so wiring
  * it in with no calibration artifact is a guaranteed no-op.
  */
final class ConstantRealBaseline(
    actionPriors: Map[(StrategicClass, PokerAction.Category), Double]
) extends RealBaseline:
  def probability(
      cls: StrategicClass,
      action: PokerAction.Category,
      sizing: Option[Sizing],
      publicState: PublicState
  ): Double =
    actionPriors.getOrElse((cls, action), 0.25)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `sbt "testOnly sicfun.holdem.strategic.safety.ConstantRealBaselineTest"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/strategic/safety/RealBaselineImpl.scala src/test/scala/sicfun/holdem/strategic/safety/ConstantRealBaselineTest.scala
git commit -m "feat(baseline): ConstantRealBaseline floor — exact pre-P1 (class,action) constants behavior"
```

---

## Task 3: `BaselineArtifact` + `BaselineArtifactIO` — count artifact (Properties + TSV)

**Files:**
- Create: `src/main/scala/sicfun/holdem/strategic/safety/BaselineArtifact.scala`
- Test: `src/test/scala/sicfun/holdem/strategic/safety/BaselineArtifactIOTest.scala`

- [ ] **Step 1: Write the failing test**

```scala
package sicfun.holdem.strategic.safety

import munit.FunSuite
import java.nio.file.Files
import sicfun.holdem.types.{PokerAction, Street}
import sicfun.holdem.strategic.types.StrategicClass

class BaselineArtifactIOTest extends FunSuite:
  test("save then load round-trips counts and metadata exactly"):
    val counts = Map[(StrategicClass, String, Street, PokerAction.Category), Long](
      (StrategicClass.Value, "Unpaired-Rainbow-AceHigh", Street.Flop, PokerAction.Category.Raise) -> 42L,
      (StrategicClass.Bluff, "Paired-TwoTone-Broadway", Street.Turn, PokerAction.Category.Fold)   -> 7L
    )
    val meta = BaselineMetadata(
      formatVersion = BaselineArtifactIO.FormatVersion,
      bucketSchemeVersion = "v1",
      corpusId = "test-corpus",
      handCount = 100,
      showdownDecisionCount = 49,
      recommendedMinCount = 30,
      recommendedSmoothingAlpha = 1.0,
      calibrationEpochMillis = 1730000000000L
    )
    val artifact = BaselineArtifact(counts, meta)
    val dir = Files.createTempDirectory("baseline-artifact-test")
    try
      BaselineArtifactIO.save(dir, artifact)
      val loaded = BaselineArtifactIO.load(dir)
      assertEquals(loaded.counts, counts)
      assertEquals(loaded.metadata, meta)
    finally
      // best-effort cleanup
      Files.walk(dir).sorted(java.util.Comparator.reverseOrder()).forEach(p => Files.deleteIfExists(p))

  test("load rejects an unsupported format version"):
    val dir = Files.createTempDirectory("baseline-artifact-bad")
    try
      Files.writeString(dir.resolve("metadata.properties"), "format.version=999\n")
      Files.writeString(dir.resolve("baseline-counts.tsv"), "class\tboardBucket\tstreet\taction\tcount\n")
      intercept[IllegalArgumentException](BaselineArtifactIO.load(dir))
    finally
      Files.walk(dir).sorted(java.util.Comparator.reverseOrder()).forEach(p => Files.deleteIfExists(p))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly sicfun.holdem.strategic.safety.BaselineArtifactIOTest"`
Expected: FAIL — `BaselineArtifact` / `BaselineArtifactIO` not found.

- [ ] **Step 3: Write the implementation**

```scala
package sicfun.holdem.strategic.safety

import java.nio.file.{Files, Path}
import java.util.Properties
import scala.jdk.CollectionConverters.*
import sicfun.holdem.types.{PokerAction, Street}
import sicfun.holdem.strategic.types.StrategicClass

/** Metadata describing how a baseline count artifact was produced. */
final case class BaselineMetadata(
    formatVersion: String,
    bucketSchemeVersion: String,
    corpusId: String,
    handCount: Long,
    showdownDecisionCount: Long,
    recommendedMinCount: Int,
    recommendedSmoothingAlpha: Double,
    calibrationEpochMillis: Long
)

/** Raw observed action counts keyed by (class, board-bucket token, street, action category),
  * plus provenance metadata. Smoothing and backoff are applied at load by RealBaselineImpl —
  * the artifact stores only counts so policy is tunable without recalibration.
  */
final case class BaselineArtifact(
    counts: Map[(StrategicClass, String, Street, PokerAction.Category), Long],
    metadata: BaselineMetadata
)

/** Persists a [[BaselineArtifact]] as a directory of `metadata.properties` + `baseline-counts.tsv`.
  * Mirrors the flat-file, diffable convention of `PokerActionModelArtifactIO`.
  */
object BaselineArtifactIO:
  val FormatVersion: String = "1"
  private val MetadataFile = "metadata.properties"
  private val CountsFile = "baseline-counts.tsv"
  private val Header = "class\tboardBucket\tstreet\taction\tcount"

  def save(directory: Path, artifact: BaselineArtifact): Unit =
    Files.createDirectories(directory)
    writeMetadata(directory.resolve(MetadataFile), artifact.metadata)
    writeCounts(directory.resolve(CountsFile), artifact.counts)

  def load(directory: Path): BaselineArtifact =
    require(Files.isDirectory(directory), s"baseline artifact directory does not exist: $directory")
    val meta = readMetadata(directory.resolve(MetadataFile))
    require(meta.formatVersion == FormatVersion,
      s"unsupported baseline artifact format version: ${meta.formatVersion} (expected $FormatVersion)")
    val counts = readCounts(directory.resolve(CountsFile))
    BaselineArtifact(counts, meta)

  private def writeMetadata(path: Path, m: BaselineMetadata): Unit =
    val props = new Properties()
    props.setProperty("format.version", m.formatVersion)
    props.setProperty("bucketScheme.version", m.bucketSchemeVersion)
    props.setProperty("calibration.corpusId", m.corpusId)
    props.setProperty("calibration.handCount", m.handCount.toString)
    props.setProperty("calibration.showdownDecisionCount", m.showdownDecisionCount.toString)
    props.setProperty("calibration.recommendedMinCount", m.recommendedMinCount.toString)
    props.setProperty("calibration.recommendedSmoothingAlpha", java.lang.Double.toString(m.recommendedSmoothingAlpha))
    props.setProperty("calibration.epochMillis", m.calibrationEpochMillis.toString)
    val writer = Files.newBufferedWriter(path)
    try props.store(writer, "BaselineArtifact metadata") finally writer.close()

  private def readMetadata(path: Path): BaselineMetadata =
    require(Files.isRegularFile(path), s"missing $MetadataFile in artifact")
    val props = new Properties()
    val reader = Files.newBufferedReader(path)
    try props.load(reader) finally reader.close()
    def req(k: String): String =
      val v = props.getProperty(k)
      require(v != null, s"missing metadata key: $k"); v
    BaselineMetadata(
      formatVersion = req("format.version"),
      bucketSchemeVersion = props.getProperty("bucketScheme.version", "v1"),
      corpusId = props.getProperty("calibration.corpusId", ""),
      handCount = props.getProperty("calibration.handCount", "0").toLong,
      showdownDecisionCount = props.getProperty("calibration.showdownDecisionCount", "0").toLong,
      recommendedMinCount = props.getProperty("calibration.recommendedMinCount", "30").toInt,
      recommendedSmoothingAlpha = props.getProperty("calibration.recommendedSmoothingAlpha", "1.0").toDouble,
      calibrationEpochMillis = props.getProperty("calibration.epochMillis", "0").toLong
    )

  private def writeCounts(path: Path, counts: Map[(StrategicClass, String, Street, PokerAction.Category), Long]): Unit =
    val sb = new StringBuilder().append(Header).append('\n')
    // sorted for deterministic, diffable output
    counts.toVector.sortBy { case ((c, b, s, a), _) => (c.toString, b, s.toString, a.toString) }
      .foreach { case ((c, b, s, a), n) =>
        sb.append(c.toString).append('\t').append(b).append('\t')
          .append(s.toString).append('\t').append(a.toString).append('\t').append(n.toString).append('\n')
      }
    Files.writeString(path, sb.toString)

  private def readCounts(path: Path): Map[(StrategicClass, String, Street, PokerAction.Category), Long] =
    require(Files.isRegularFile(path), s"missing $CountsFile in artifact")
    val lines = Files.readAllLines(path).asScala.toVector
    lines.drop(1).filter(_.trim.nonEmpty).map { line =>
      val cols = line.split("\t", -1)
      require(cols.length == 5, s"malformed counts row: $line")
      val key = (
        StrategicClass.valueOf(cols(0)),
        cols(1),
        Street.valueOf(cols(2)),
        PokerAction.Category.valueOf(cols(3))
      )
      key -> cols(4).toLong
    }.toMap
```

- [ ] **Step 4: Run test to verify it passes**

Run: `sbt "testOnly sicfun.holdem.strategic.safety.BaselineArtifactIOTest"`
Expected: PASS (round-trip + version rejection).

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/strategic/safety/BaselineArtifact.scala src/test/scala/sicfun/holdem/strategic/safety/BaselineArtifactIOTest.scala
git commit -m "feat(baseline): BaselineArtifact + Properties/TSV IO (raw counts, versioned, diffable)"
```

---

## Task 4: `RealBaselineImpl` — smoothed, board-conditioned lookup with backoff ladder

**Files:**
- Modify: `src/main/scala/sicfun/holdem/strategic/safety/RealBaselineImpl.scala` (add `RealBaselineImpl` alongside `ConstantRealBaseline`)
- Test: `src/test/scala/sicfun/holdem/strategic/safety/RealBaselineImplTest.scala`

**Backoff ladder (each level requires its group's total observed count ≥ `minCount`):**
1. `(class, bucketToken, street, action)` — Laplace-smoothed over the 4 actions of that `(class,bucket,street)` group.
2. `(class, street, action)` — aggregate counts over all buckets, Laplace-smoothed.
3. `fallback.probability(class, action, sizing, publicState)` — the `ConstantRealBaseline` floor `(class,action)` constants.

(Uniform is implicit: `ConstantRealBaseline` returns 0.25 for unknown keys.)

- [ ] **Step 1: Write the failing test**

```scala
package sicfun.holdem.strategic.safety

import munit.FunSuite
import sicfun.holdem.types.{Board, Chips, PokerAction, Street, TableMap}
import sicfun.holdem.strategic.types.StrategicClass
import sicfun.holdem.strategic.state.PublicState
import sicfun.holdem.engine.StrategicEngine
import sicfun.core.Card

class RealBaselineImplTest extends FunSuite:
  private val Cat = PokerAction.Category
  private val floor = ConstantRealBaseline(StrategicEngine.defaultActionPriors)
  private def flop(tokens: String*): PublicState =
    PublicState(Street.Flop, Board(tokens.toVector.map(t => Card.parse(t).get)), Chips(0.0), TableMap.empty, Vector.empty)

  // A populated cell: Value on Unpaired-Rainbow-AceHigh flop strongly raises.
  private val bucket = "Unpaired-Rainbow-AceHigh"
  private val meta = BaselineMetadata("1", "v1", "t", 1, 200, minCountForTest, 1.0, 0L)
  private def minCountForTest = 30
  private val counts = Map[(StrategicClass, String, Street, PokerAction.Category), Long](
    (StrategicClass.Value, bucket, Street.Flop, Cat.Raise) -> 70L,
    (StrategicClass.Value, bucket, Street.Flop, Cat.Call)  -> 20L,
    (StrategicClass.Value, bucket, Street.Flop, Cat.Check) -> 8L,
    (StrategicClass.Value, bucket, Street.Flop, Cat.Fold)  -> 2L
  )
  private val impl = RealBaselineImpl(BaselineArtifact(counts, meta), minCount = 30, alpha = 1.0, fallback = floor)

  test("direct cell hit returns the Laplace-smoothed observed frequency"):
    // group total = 100; alpha=1, 4 actions → (70+1)/(100+4) = 71/104
    assertEqualsDouble(impl.probability(StrategicClass.Value, Cat.Raise, None, flop("As","Kd","7c")), 71.0/104.0, 1e-9)
    assertEqualsDouble(impl.probability(StrategicClass.Value, Cat.Fold, None, flop("As","Kd","7c")), 3.0/104.0, 1e-9)

  test("a populated cell's distribution over the 4 actions sums to 1"):
    val ps = flop("As","Kd","7c")
    val s = Cat.values.map(a => impl.probability(StrategicClass.Value, a, None, ps)).sum
    assertEqualsDouble(s, 1.0, 1e-9)

  test("backoff to (class,action) constants when the bucket+street group is below minCount"):
    // Bluff has zero observed counts → group total 0 < 30 → backoff to constants floor
    val ps = flop("As","Kd","7c")
    assertEquals(impl.probability(StrategicClass.Bluff, Cat.Raise, None, ps),
                 floor.probability(StrategicClass.Bluff, Cat.Raise, None, ps)) // 0.65

  test("preflop always backs off to constants (no preflop cells in v1)"):
    val pre = PublicState(Street.Preflop, Board.empty, Chips(0.0), TableMap.empty, Vector.empty)
    assertEquals(impl.probability(StrategicClass.Value, Cat.Raise, None, pre),
                 floor.probability(StrategicClass.Value, Cat.Raise, None, pre)) // 0.20

  test("empty artifact behaves exactly like the constants floor"):
    val emptyImpl = RealBaselineImpl(BaselineArtifact(Map.empty, meta), 30, 1.0, floor)
    val ps = flop("As","Kd","7c")
    for c <- StrategicClass.values; a <- Cat.values do
      assertEquals(emptyImpl.probability(c, a, None, ps), floor.probability(c, a, None, ps))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly sicfun.holdem.strategic.safety.RealBaselineImplTest"`
Expected: FAIL — `RealBaselineImpl` not found.

- [ ] **Step 3: Write the implementation** (append to `RealBaselineImpl.scala`)

```scala
import sicfun.holdem.types.{PokerAction, Street}

/** Calibrated, board/street-conditioned Def 9 baseline.
  *
  * Builds Laplace-smoothed action distributions at two granularities from raw observed
  * counts, then answers `probability` with a backoff ladder that guarantees a valid
  * distribution in every spot and is never worse than the `fallback` constants floor.
  *
  * @param artifact observed raw counts + metadata
  * @param minCount minimum group total to use a granularity level directly
  * @param alpha    Laplace add-alpha smoothing parameter
  * @param fallback the (class,action) constants floor (a ConstantRealBaseline)
  */
final class RealBaselineImpl(
    artifact: BaselineArtifact,
    minCount: Int,
    alpha: Double,
    fallback: RealBaseline
) extends RealBaseline:

  private val actions = PokerAction.Category.values

  // Level-1 (cell) group totals: (class, bucket, street) -> total observed
  private val cellGroupTotal: Map[(StrategicClass, String, Street), Long] =
    artifact.counts.groupMapReduce { case ((c, b, s, _), _) => (c, b, s) } { case (_, n) => n }(_ + _)

  // Level-2 (street) action counts: (class, street, action) -> total over all buckets
  private val streetActionCount: Map[(StrategicClass, Street, PokerAction.Category), Long] =
    artifact.counts.groupMapReduce { case ((c, _, s, a), _) => (c, s, a) } { case (_, n) => n }(_ + _)

  private val streetGroupTotal: Map[(StrategicClass, Street), Long] =
    streetActionCount.groupMapReduce { case ((c, s, _), _) => (c, s) } { case (_, n) => n }(_ + _)

  private def smoothed(count: Long, groupTotal: Long): Double =
    (count + alpha) / (groupTotal + alpha * actions.length)

  def probability(
      cls: StrategicClass,
      action: PokerAction.Category,
      sizing: Option[Sizing],
      publicState: PublicState
  ): Double =
    val bucketToken = BoardBucket.token(BoardBucket.of(publicState))
    val street = publicState.street

    val cellTotal = cellGroupTotal.getOrElse((cls, bucketToken, street), 0L)
    if cellTotal >= minCount then
      val n = artifact.counts.getOrElse((cls, bucketToken, street, action), 0L)
      smoothed(n, cellTotal)
    else
      val streetTotal = streetGroupTotal.getOrElse((cls, street), 0L)
      if streetTotal >= minCount then
        val n = streetActionCount.getOrElse((cls, street, action), 0L)
        smoothed(n, streetTotal)
      else
        fallback.probability(cls, action, sizing, publicState)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `sbt "testOnly sicfun.holdem.strategic.safety.RealBaselineImplTest"`
Expected: PASS (cell hit, sums-to-1, both backoff levels, empty==floor).

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/strategic/safety/RealBaselineImpl.scala src/test/scala/sicfun/holdem/strategic/safety/RealBaselineImplTest.scala
git commit -m "feat(baseline): RealBaselineImpl smoothed board/street-conditioned lookup w/ backoff ladder"
```

---

## Task 5: Synthetic showdown corpus + import sanity test

The P0 sample (`p0-sample-9max.txt`) has **zero showdowns**, so it cannot exercise class-labeling. Create a small PokerStars-format 9-max corpus with explicit `*** SHOW DOWN ***` sections and revealed holdings, then prove the importer recovers them.

**Files:**
- Create: `src/test/resources/handhistory/p1-showdown-sample.txt`
- Test: `src/test/scala/sicfun/holdem/history/P1ShowdownSampleTest.scala`

- [ ] **Step 1: Write the failing test**

```scala
package sicfun.holdem.history

import munit.FunSuite
import java.nio.file.Paths

class P1ShowdownSampleTest extends FunSuite:
  test("p1 showdown sample imports with at least 2 hands that reveal holdings"):
    val url = getClass.getResource("/handhistory/p1-showdown-sample.txt")
    assert(url != null, "p1-showdown-sample.txt resource must exist")
    val hands = HandHistoryImport.parseFile(Paths.get(url.toURI)) match
      case Right(hs) => hs
      case Left(err) => fail(s"parse failed: $err")
    val withShowdown = hands.filter(_.showdownCards.nonEmpty)
    assert(withShowdown.size >= 2, s"expected ≥2 showdown hands, got ${withShowdown.size}")
    // every revealed holding must correspond to a player who has decision events
    withShowdown.foreach { h =>
      h.showdownCards.keys.foreach { name =>
        assert(h.events.exists(_.playerId == name), s"revealed player $name should have events")
      }
    }
    // at least one postflop decision by a revealed player exists (the calibration input)
    val postflopRevealed = withShowdown.flatMap(h => h.events.filter(e =>
      h.showdownCards.contains(e.playerId) && e.street != sicfun.holdem.types.Street.Preflop))
    assert(postflopRevealed.nonEmpty, "need ≥1 postflop decision by a revealed player")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly sicfun.holdem.history.P1ShowdownSampleTest"`
Expected: FAIL — resource missing / null.

- [ ] **Step 3: Create the corpus resource**

Create `src/test/resources/handhistory/p1-showdown-sample.txt`. Model the header/seat/post/event lines **exactly** on the existing `p0-sample-9max.txt` (open it for the precise line grammar), and add `*** FLOP ***`/`*** TURN ***`/`*** RIVER ***`/`*** SHOW DOWN ***` sections. The `ShowdownPattern` (`HandHistoryImport.scala:169`) matches `"<Name>: shows [Xx Yy]"`. Provide ≥3 hands; ≥2 reaching showdown with ≥2 revealed players each. Example shape for ONE hand (the implementer writes ≥3, varying holdings/boards so multiple board-buckets appear):

```
PokerStars Hand #971000001:  Hold'em No Limit ($1/$2 USD) - 2026/05/02 12:00:00 ET
Table 'P1Showdown' 9-max Seat #1 is the button
Seat 1: Villain1 ($200 in chips)
Seat 2: Villain2 ($200 in chips)
Seat 3: Villain3 ($200 in chips)
Seat 4: Hero ($200 in chips)
Villain2: posts small blind $1
Villain3: posts big blind $2
*** HOLE CARDS ***
Dealt to Hero [As Ks]
Villain1: raises $4 to $6
Villain3: calls $4
Hero: calls $6
*** FLOP *** [Ah 7d 2c]
Villain3: checks
Hero: bets $10
Villain1: calls $10
Villain3: folds
*** TURN *** [Ah 7d 2c] [9s]
Hero: bets $25
Villain1: calls $25
*** RIVER *** [Ah 7d 2c 9s] [3h]
Hero: bets $60
Villain1: calls $60
*** SHOW DOWN ***
Hero: shows [As Ks] (a pair of Aces)
Villain1: shows [Qh Qd] (a pair of Queens)
Hero collected $202 from pot
*** SUMMARY ***
Total pot $202 | Rake $0
Board [Ah 7d 2c 9s 3h]
```

> The implementer MUST verify each authored hand parses (run the test after each) — the parser is strict about the post/seat/board grammar. Vary boards across hands so the corpus exercises ≥2 distinct postflop buckets (e.g. one paired board, one two-tone board).

- [ ] **Step 4: Run test to verify it passes**

Run: `sbt "testOnly sicfun.holdem.history.P1ShowdownSampleTest"`
Expected: PASS (≥2 showdown hands, revealed players have events, ≥1 postflop revealed decision).

- [ ] **Step 5: Commit**

```bash
git add src/test/resources/handhistory/p1-showdown-sample.txt src/test/scala/sicfun/holdem/history/P1ShowdownSampleTest.scala
git commit -m "test(baseline): synthetic 9-max showdown corpus + import-recovers-holdings sanity"
```

---

## Task 6: `HoldingClassifier` — revealed holding → StrategicClass

Reuses the engine's `ClassificationBridge.classify(equity, hasDrawPotential)` (so the calibration labels match the engine's own class partition), computing equity exactly via `HoldemEquity.equityExact` against a uniform villain range, and a coarse postflop draw detector.

**Files:**
- Create: `src/main/scala/sicfun/holdem/strategic/calibration/HoldingClassifier.scala`
- Test: `src/test/scala/sicfun/holdem/strategic/calibration/HoldingClassifierTest.scala`

- [ ] **Step 1: Write the failing test**

```scala
package sicfun.holdem.strategic.calibration

import munit.FunSuite
import sicfun.core.Card
import sicfun.holdem.types.{Board, HoleCards, Street}
import sicfun.holdem.strategic.types.StrategicClass

class HoldingClassifierTest extends FunSuite:
  private def hc(a: String, b: String) = HoleCards.canonical(Card.parse(a).get, Card.parse(b).get)
  private def bd(ts: String*) = Board(ts.toVector.map(t => Card.parse(t).get))

  test("nut-strong made hand on a dry river → Value"):
    // top set on A 7 2 9 3 rainbow, equity vs random ~ near 1.0
    val cls = HoldingClassifier.classify(hc("As", "Ad"), bd("Ah","7d","2c","9s","3h"), Street.River)
    assertEquals(cls, StrategicClass.Value)

  test("trash hand on a dry river → Bluff"):
    val cls = HoldingClassifier.classify(hc("7c", "2d"), bd("Ah","Kd","Qc","9s","3h"), Street.River)
    assertEquals(cls, StrategicClass.Bluff)

  test("flush draw in the middle equity band on the flop → StructuralBluff"):
    // two hearts in hand + two hearts on board = 4 to a flush, mid equity
    val cls = HoldingClassifier.classify(hc("Th", "9h"), bd("Ah","7h","2c"), Street.Flop)
    assertEquals(cls, StrategicClass.StructuralBluff)

  test("hasDrawPotential: 4 to a flush on the flop is a draw; on the river it is not"):
    assert(HoldingClassifier.hasDrawPotential(hc("Th","9h"), bd("Ah","7h","2c"), Street.Flop))
    assert(!HoldingClassifier.hasDrawPotential(hc("Th","9h"), bd("Ah","7h","2c","Kd","3s"), Street.River))

  test("hasDrawPotential: open-ended straight draw on the flop"):
    // 9 8 with 7 6 on board → open-ended (5..10)
    assert(HoldingClassifier.hasDrawPotential(hc("9c","8d"), bd("7h","6s","2c"), Street.Flop))
```

> Equity thresholds are `ClassificationBridge`'s (Value≥0.65, Bluff<0.35). If a chosen fixture lands near a boundary, the implementer should pick a less ambiguous holding/board — the assertion is the class, the fixtures are illustrative.

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly sicfun.holdem.strategic.calibration.HoldingClassifierTest"`
Expected: FAIL — `HoldingClassifier` not found.

- [ ] **Step 3: Write the implementation**

```scala
package sicfun.holdem.strategic.calibration

import sicfun.core.{Card, Deck, DiscreteDistribution}      // confirm imports against HoldemEquity.scala
import sicfun.holdem.types.{Board, HoleCards, Street}
import sicfun.holdem.equity.HoldemEquity
import sicfun.holdem.strategic.types.StrategicClass
import sicfun.holdem.strategic.bridge.ClassificationBridge

/** Classifies a SHOWDOWN-revealed holding into the engine's StrategicClass partition,
  * reusing ClassificationBridge.classify over an exact equity estimate. Offline use only.
  */
object HoldingClassifier:

  /** Uniform distribution over all 1326 two-card combos. `equityExact` sanitizes blocked
    * cards per call, so this constant range is reusable across holdings/boards. */
  private val uniformVillainRange: DiscreteDistribution[HoleCards] =
    val cards = Deck.full.toVector
    val combos =
      for i <- cards.indices; j <- (i + 1) until cards.length
      yield HoleCards.canonical(cards(i), cards(j))
    val w = 1.0 / combos.length
    DiscreteDistribution(combos.map(h => h -> w).toMap)

  def classify(holding: HoleCards, board: Board, street: Street): StrategicClass =
    val equity = HoldemEquity.equityExact(holding, board, uniformVillainRange).equity
    val draw = hasDrawPotential(holding, board, street)
    ClassificationBridge.classify(equity, draw).fold(
      onExact = identity,
      onApprox = (cls, _) => cls,
      onAbsent = throw new IllegalStateException("classification returned Absent")
    )

  /** Coarse postflop draw detector: a flush draw (exactly 4 to a suit across hole+board)
    * or an open-ended straight draw (a run of ≥4 consecutive distinct ranks). Never true
    * on the river. */
  def hasDrawPotential(holding: HoleCards, board: Board, street: Street): Boolean =
    if street == Street.River then false
    else
      val cards = holding.toVector ++ board.cards
      val flushDraw = cards.groupBy(_.suit).values.exists(_.size == 4)
      flushDraw || hasOpenEndedStraightDraw(cards)

  private def hasOpenEndedStraightDraw(cards: Vector[Card]): Boolean =
    val ranks = cards.map(_.rank.value).distinct.sorted
    if ranks.length < 4 then false
    else
      // longest run of consecutive values; ≥4 distinct consecutive → open-ended
      var best = 1; var run = 1; var i = 1
      while i < ranks.length do
        if ranks(i) == ranks(i - 1) + 1 then run += 1 else run = 1
        if run > best then best = run
        i += 1
      best >= 4
```

> `Deck.full`, `DiscreteDistribution`, and the `BridgeResult.fold(onExact, onApprox, onAbsent)` signature must match `HoldemEquity.scala` / `StrategicSnapshot.scala:128-139`. If `fold`'s parameter names differ, adapt; the contract is "unwrap to the classified value."

- [ ] **Step 4: Run test to verify it passes**

Run: `sbt "testOnly sicfun.holdem.strategic.calibration.HoldingClassifierTest"`
Expected: PASS. (Equity here is exact; River is instant, Flop enumerates ~1M boards/eval — fine in a unit test with a handful of cases.)

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/strategic/calibration/HoldingClassifier.scala src/test/scala/sicfun/holdem/strategic/calibration/HoldingClassifierTest.scala
git commit -m "feat(baseline): HoldingClassifier — revealed holding → StrategicClass via exact equity + draws"
```

---

## Task 7: `BaselineCalibrationTool` — CLI corpus → count artifact

**Files:**
- Create: `src/main/scala/sicfun/holdem/strategic/calibration/BaselineCalibrationTool.scala`
- Test: `src/test/scala/sicfun/holdem/strategic/calibration/BaselineCalibrationToolTest.scala`

Pipeline: parse corpus (`HandHistoryImport.parseFile`) → for each hand, for each `PokerEvent` whose `playerId ∈ showdownCards` **and** `street != Preflop` → classify `(showdownCards(playerId), event.board, event.street)` → increment `(class, bucketToken, street, action.category)`. Write a `BaselineArtifact`.

- [ ] **Step 1: Write the failing test**

```scala
package sicfun.holdem.strategic.calibration

import munit.FunSuite
import java.nio.file.{Files, Paths}
import sicfun.holdem.strategic.safety.BaselineArtifactIO

class BaselineCalibrationToolTest extends FunSuite:
  test("calibrate over the showdown sample produces a non-empty postflop count artifact"):
    val corpus = Paths.get(getClass.getResource("/handhistory/p1-showdown-sample.txt").toURI)
    val outDir = Files.createTempDirectory("baseline-calib-test")
    try
      val artifact = BaselineCalibrationTool.calibrate(corpus, corpusId = "p1-sample")
      // only postflop cells, only revealed players
      assert(artifact.counts.nonEmpty, "expected ≥1 postflop showdown decision tallied")
      assert(artifact.counts.keys.forall { case (_, bucket, street, _) =>
        bucket != "PRE" && street != sicfun.holdem.types.Street.Preflop
      }, "v1 calibrates postflop only")
      assert(artifact.metadata.showdownDecisionCount >= 1)
      // round-trips through IO
      BaselineCalibrationTool.write(outDir, artifact)
      val reloaded = BaselineArtifactIO.load(outDir)
      assertEquals(reloaded.counts, artifact.counts)
    finally
      Files.walk(outDir).sorted(java.util.Comparator.reverseOrder()).forEach(p => Files.deleteIfExists(p))

  test("main writes an artifact directory"):
    val corpus = Paths.get(getClass.getResource("/handhistory/p1-showdown-sample.txt").toURI)
    val outDir = Files.createTempDirectory("baseline-calib-main")
    try
      BaselineCalibrationTool.main(Array(corpus.toString, outDir.toString))
      assert(Files.isRegularFile(outDir.resolve("baseline-counts.tsv")))
      assert(Files.isRegularFile(outDir.resolve("metadata.properties")))
    finally
      Files.walk(outDir).sorted(java.util.Comparator.reverseOrder()).forEach(p => Files.deleteIfExists(p))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly sicfun.holdem.strategic.calibration.BaselineCalibrationToolTest"`
Expected: FAIL — `BaselineCalibrationTool` not found.

- [ ] **Step 3: Write the implementation**

```scala
package sicfun.holdem.strategic.calibration

import java.nio.file.{Path, Paths}
import scala.collection.mutable
import sicfun.holdem.history.HandHistoryImport
import sicfun.holdem.types.{Board, HoleCards, PokerAction, Street}
import sicfun.holdem.strategic.types.StrategicClass
import sicfun.holdem.strategic.safety.{BaselineArtifact, BaselineArtifactIO, BaselineMetadata, BoardBucket}

/** Offline tool: ingest a hand-history corpus and tally per-(class,board-bucket,street,action)
  * action counts from SHOWDOWN-revealed holdings (postflop decisions only, v1).
  *
  * CLI: `sbt "runMain sicfun.holdem.strategic.calibration.BaselineCalibrationTool <corpusPath> <outDir> [--minCount=30] [--alpha=1.0] [--corpusId=name]"`
  */
object BaselineCalibrationTool:
  private val DefaultMinCount = 30
  private val DefaultAlpha = 1.0

  /** Pure calibration: corpus file → BaselineArtifact (raw counts + metadata). */
  def calibrate(
      corpus: Path,
      corpusId: String,
      minCount: Int = DefaultMinCount,
      alpha: Double = DefaultAlpha,
      nowEpochMillis: Long = System.currentTimeMillis()
  ): BaselineArtifact =
    val hands = HandHistoryImport.parseFile(corpus) match
      case Right(hs) => hs
      case Left(err) => throw new IllegalArgumentException(s"corpus parse failed: $err")

    val counts = mutable.Map.empty[(StrategicClass, String, Street, PokerAction.Category), Long]
    val equityCache = mutable.Map.empty[(String, String), StrategicClass] // (holding token, board token) → class
    var showdownDecisions = 0L

    hands.foreach { hand =>
      hand.events.foreach { ev =>
        hand.showdownCards.get(ev.playerId).foreach { holding =>
          if ev.street != Street.Preflop && ev.board.cards.nonEmpty then
            val boardToken = ev.board.cards.map(_.toToken).sorted.mkString
            val cls = equityCache.getOrElseUpdate(
              (holding.toToken, boardToken),
              HoldingClassifier.classify(holding, ev.board, ev.street)
            )
            val bucket = BoardBucket.token(BoardBucket.ofBoard(ev.street, ev.board))
            val key = (cls, bucket, ev.street, ev.action.category)
            counts.update(key, counts.getOrElse(key, 0L) + 1L)
            showdownDecisions += 1
        }
      }
    }

    val meta = BaselineMetadata(
      formatVersion = BaselineArtifactIO.FormatVersion,
      bucketSchemeVersion = "v1",
      corpusId = corpusId,
      handCount = hands.length.toLong,
      showdownDecisionCount = showdownDecisions,
      recommendedMinCount = minCount,
      recommendedSmoothingAlpha = alpha,
      calibrationEpochMillis = nowEpochMillis
    )
    BaselineArtifact(counts.toMap, meta)

  def write(outDir: Path, artifact: BaselineArtifact): Unit =
    BaselineArtifactIO.save(outDir, artifact)

  def main(args: Array[String]): Unit =
    if args.length < 2 then
      System.err.println(
        "usage: BaselineCalibrationTool <corpusPath> <outDir> [--minCount=30] [--alpha=1.0] [--corpusId=name]")
      sys.exit(1)
    val corpus = Paths.get(args(0))
    val outDir = Paths.get(args(1))
    val opts = args.drop(2).flatMap { a =>
      val kv = a.stripPrefix("--").split("=", 2)
      if kv.length == 2 then Some(kv(0) -> kv(1)) else None
    }.toMap
    val minCount = opts.get("minCount").map(_.toInt).getOrElse(DefaultMinCount)
    val alpha = opts.get("alpha").map(_.toDouble).getOrElse(DefaultAlpha)
    val corpusId = opts.getOrElse("corpusId", corpus.getFileName.toString)
    val artifact = calibrate(corpus, corpusId, minCount, alpha)
    write(outDir, artifact)
    println(s"baseline artifact: ${outDir.toAbsolutePath.normalize()}")
    println(s"hands: ${artifact.metadata.handCount}")
    println(s"showdownDecisions: ${artifact.metadata.showdownDecisionCount}")
    println(s"cells: ${artifact.counts.size}")
```

> `ev.action.category` must match `PokerAction`'s category accessor (`PokerAction.scala:53-54`). Confirm `Card.toToken` is on `Card` (it is — `Card.scala:118`) and `HoleCards.toToken` (it is — `HoldemTypes.scala:49`). The `equityCache` keys on canonical-ish string tokens; this is a pure speed optimization (exact equity on the flop is ~1M evals).

- [ ] **Step 4: Run test to verify it passes**

Run: `sbt "testOnly sicfun.holdem.strategic.calibration.BaselineCalibrationToolTest"`
Expected: PASS (non-empty postflop-only counts; round-trips; `main` writes both files).

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/strategic/calibration/BaselineCalibrationTool.scala src/test/scala/sicfun/holdem/strategic/calibration/BaselineCalibrationToolTest.scala
git commit -m "feat(baseline): BaselineCalibrationTool CLI — showdown corpus → count artifact (postflop, v1)"
```

---

## Task 8: Wire the calibrated baseline into the engine (gated; no-regression by default)

Three edits + two engine-level tests. The contract: **`baselinePath = None` ⇒ behavior byte-identical to today.**

**Files:**
- Modify: `src/main/scala/sicfun/holdem/strategic/PosteriorAttributedBaseline.scala`
- Modify: `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala`
- Test: `src/test/scala/sicfun/holdem/strategic/BaselineWiringTest.scala`

- [ ] **Step 1: Write the failing test**

```scala
package sicfun.holdem.strategic

import munit.FunSuite
import sicfun.holdem.types.{Board, Chips, PokerAction, Street, TableMap}
import sicfun.holdem.strategic.types.StrategicClass
import sicfun.holdem.strategic.state.PublicState
import sicfun.holdem.strategic.safety.{BaselineArtifact, BaselineMetadata, ConstantRealBaseline, RealBaselineImpl}
import sicfun.holdem.engine.StrategicEngine
import sicfun.core.Card

class BaselineWiringTest extends FunSuite:
  private val Cat = PokerAction.Category
  private val priors = StrategicEngine.defaultActionPriors
  private def flop(ts: String*) =
    PublicState(Street.Flop, Board(ts.toVector.map(t => Card.parse(t).get)), Chips(0.0), TableMap.empty, Vector.empty)

  test("the calibrated Def 9 baseline is board-sensitive (the whole point)"):
    // Artifact: Value RAISES heavily on bucket A (As Kd 7c) but FOLDS heavily on bucket B (9h 7h 5h, monotone).
    val bucketA = "Unpaired-Rainbow-AceHigh"
    val bucketB = "Unpaired-Monotone-Middle"
    val counts = Map(
      (StrategicClass.Value, bucketA, Street.Flop, Cat.Raise) -> 95L,
      (StrategicClass.Value, bucketA, Street.Flop, Cat.Fold)  -> 5L,
      (StrategicClass.Value, bucketB, Street.Flop, Cat.Raise) -> 5L,
      (StrategicClass.Value, bucketB, Street.Flop, Cat.Fold)  -> 95L
    )
    val meta = BaselineMetadata("1","v1","t",1,200,30,1.0,0L)
    val rb = RealBaselineImpl(BaselineArtifact(counts, meta), 30, 1.0, ConstantRealBaseline(priors))
    val pRaiseA = rb.probability(StrategicClass.Value, Cat.Raise, None, flop("As","Kd","7c"))
    val pRaiseB = rb.probability(StrategicClass.Value, Cat.Raise, None, flop("9h","7h","5h"))
    assert(pRaiseA > 0.8 && pRaiseB < 0.2, s"board-sensitivity: A=$pRaiseA B=$pRaiseB")

  test("StrategicEngine constructs with no baseline (no-regression) and with a calibrated baseline"):
    // No baseline → _realBaseline is a ConstantRealBaseline → ref/attrib likelihoods are byte-identical
    // to pre-P1 (the probability() math body is unchanged; ConstantRealBaseline == getOrElse(...,0.25)).
    val engineDefault = StrategicEngine(StrategicEngine.Config())
    assert(engineDefault != null)
    // Some(path) → the artifact loads and a RealBaselineImpl is wired:
    val dir = java.nio.file.Files.createTempDirectory("wiring-artifact")
    try
      sicfun.holdem.strategic.safety.BaselineArtifactIO.save(dir,
        BaselineArtifact(
          Map((StrategicClass.Value, "Unpaired-Rainbow-AceHigh", Street.Flop, Cat.Raise) -> 50L),
          BaselineMetadata("1","v1","t",1,50,30,1.0,0L)))
      val engineCalib = StrategicEngine(StrategicEngine.Config(baselinePath = Some(dir.toString)))
      assert(engineCalib != null)
    finally
      java.nio.file.Files.walk(dir).sorted(java.util.Comparator.reverseOrder())
        .forEach(p => java.nio.file.Files.deleteIfExists(p))
```

> Confirm the `PublicState` empty-construction args (`Chips(0.0)`, `TableMap.empty`) against `AugmentedState.scala:13-19`. The no-regression guarantee rests on Tasks 2 + 4 (`ConstantRealBaseline == getOrElse(...,0.25)`; empty artifact == floor) plus the unchanged `probability()` body — these two construction tests prove the wiring compiles and the artifact load path works. OPTIONAL strengthening: add an end-to-end decide-level assertion by mirroring the public decide call in `src/test/scala/sicfun/holdem/engine/StrategicEngineTest.scala`.

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly sicfun.holdem.strategic.BaselineWiringTest"`
Expected: FAIL — `PosteriorAttributedBaseline` does not accept a `RealBaseline`; `Config.baselinePath` missing.

- [ ] **Step 3a: Rewire `PosteriorAttributedBaseline` to consume a `RealBaseline`**

Replace its constructor + `pi0` (current `PosteriorAttributedBaseline.scala:21-31`). The `probability` body (lines 33-52) is UNCHANGED except `pi0` now delegates to the conditioned baseline, passing `publicState`:

```scala
package sicfun.holdem.strategic

import sicfun.holdem.types.PokerAction
import sicfun.holdem.strategic.types.StrategicClass
import sicfun.holdem.strategic.state.{PublicState, RivalBeliefState, Sizing}
import sicfun.holdem.strategic.safety.{AttributedBaseline, RealBaseline}

/** Def 10 attributed baseline as a kernel-coupled reweighting of a (now board/street-conditioned)
  * Def 9 real baseline. Behavior is unchanged when `realBaseline` is a ConstantRealBaseline.
  */
class PosteriorAttributedBaseline(
    realBaseline: RealBaseline
) extends AttributedBaseline:

  private val Eps = 1e-10
  private val classes = StrategicClass.values
  private val actions = PokerAction.Category.values
  private val numClasses = classes.length

  private def pi0(cls: StrategicClass, cat: PokerAction.Category, sizing: Option[Sizing], publicState: PublicState): Double =
    realBaseline.probability(cls, cat, sizing, publicState)

  def probability(
      cls: StrategicClass,
      action: PokerAction.Category,
      sizing: Option[Sizing],
      publicState: PublicState,
      rivalState: RivalBeliefState
  ): Double =
    rivalState match
      case srb: StrategicRivalBelief =>
        val posterior = srb.typePosterior
        val weights = actions.map { a =>
          val pPred = math.max(Eps, classes.map(c => posterior.probabilityOf(c) * pi0(c, a, sizing, publicState)).sum)
          val pRef = math.max(Eps, classes.map(c => pi0(c, a, sizing, publicState)).sum / numClasses)
          a -> (pPred / pRef)
        }.toMap
        val z = math.max(Eps, actions.map(a => pi0(cls, a, sizing, publicState) * weights(a)).sum)
        pi0(cls, action, sizing, publicState) * weights(action) / z
      case _ =>
        pi0(cls, action, sizing, publicState)
```

> Confirm `StrategicRivalBelief` import (it's referenced in the original file, same package `sicfun.holdem.strategic`). Confirm `AttributedBaseline`/`RealBaseline` live in `sicfun.holdem.strategic.safety` (Baseline.scala).

- [ ] **Step 3b: Add `baselinePath` to `StrategicEngine.Config` and build `_realBaseline`**

In `StrategicEngine.Config` (`StrategicEngine.scala:1295-1328`), add a field (place it right after `actionPriors`, before `detector`):

```scala
    actionPriors: Map[(StrategicClass, sicfun.holdem.types.PokerAction.Category), Double] = defaultActionPriors,
    baselinePath: Option[String] = None,
    detector: DetectionPredicate = FrequencyAnomalyDetection(window = 20, threshold = 0.6),
```

Replace `_attributedBaseline` (`StrategicEngine.scala:32-33`) and add `_realBaseline` above it:

```scala
  /** Def 9 real baseline: calibrated artifact when configured, else the constants floor. */
  private val _realBaseline: sicfun.holdem.strategic.safety.RealBaseline =
    config.baselinePath match
      case Some(p) =>
        val artifact = sicfun.holdem.strategic.safety.BaselineArtifactIO.load(java.nio.file.Paths.get(p))
        new sicfun.holdem.strategic.safety.RealBaselineImpl(
          artifact,
          artifact.metadata.recommendedMinCount,
          artifact.metadata.recommendedSmoothingAlpha,
          new sicfun.holdem.strategic.safety.ConstantRealBaseline(config.actionPriors))
      case None =>
        new sicfun.holdem.strategic.safety.ConstantRealBaseline(config.actionPriors)

  /** Kernel-coupled attributed baseline (Def 10), now over the conditioned Def 9 base. */
  private val _attributedBaseline: PosteriorAttributedBaseline =
    new PosteriorAttributedBaseline(_realBaseline)
```

- [ ] **Step 3c: Rewire the ref likelihood to the conditioned baseline**

In `buildRefLikelihoodFn` (`StrategicEngine.scala:1178-1191`), replace the per-class prior. Change:

```scala
      val basePr = classes.map { cls =>
        actionPrior(cls, signal.action)
      }
```

to:

```scala
      val basePr = classes.map { cls =>
        _realBaseline.probability(cls, signal.action, signal.sizing, pubState)
      }
```

Leave `actionPrior` (1144-1145) in place (still referenced elsewhere / harmless); do **not** touch `PokerPftFormulation`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `sbt "testOnly sicfun.holdem.strategic.BaselineWiringTest"`
Then the full safety/calibration suite + a broad compile:
Run: `sbt "testOnly sicfun.holdem.strategic.*" "testOnly sicfun.holdem.history.P1ShowdownSampleTest"`
Run: `sbt compile` (must pass under `-Werror`)
Expected: PASS; no warnings-as-errors. The no-regression assertion (constants math preserved) and board-sensitivity assertion both green.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/strategic/PosteriorAttributedBaseline.scala src/main/scala/sicfun/holdem/engine/StrategicEngine.scala src/test/scala/sicfun/holdem/strategic/BaselineWiringTest.scala
git commit -m "feat(baseline): wire conditioned Def 9/10 baseline into StrategicEngine (gated; no-op when baselinePath=None)"
```

---

## Post-implementation: G3 validation (separate, gated)

After all 8 tasks are green and the branch compiles under `-Werror`:

1. Ensure the **G1 baseline** has been captured on the pre-P1 commit (deterministic hall; see resume notes). Do not merge without it.
2. Calibrate against a real corpus: `sbt "runMain sicfun.holdem.strategic.calibration.BaselineCalibrationTool <realCorpus> data/p1-baseline-artifact --corpusId=<id>"`.
3. Re-run the P0 harness with `Config(baselinePath = Some("data/p1-baseline-artifact"))` wired into the benchmark's engine config, vs the same seed/hands/field as G1.
4. **G3 gate:** bb/100 must preserve-or-improve vs G1. A regression is a finding to investigate (likely: thin cells, a bad bucket boundary, or a class-label mismatch), **not** an auto-ship. Record the before/after in the plan's PR.

## Integration note (post web-deploy-hardening merge)

On `feat/web-deploy-hardening`, `HandHistoryImport` gained a resilient `parseTextOutcome`/`ImportOutcome`/`SkippedHand` API (a single malformed hand no longer aborts the whole parse). After P0 integrates onto that base, change `BaselineCalibrationTool.calibrate` to prefer the resilient entry (skip + report malformed hands) instead of `parseFile`'s all-or-nothing `Either` — a real corpus will have some unparseable hands, and aborting the whole calibration on one bad hand is the wrong behavior. This is a one-line swap once the API is present; on the current P0 branch, `parseFile` is correct.

---

## Self-review

**Spec coverage:** Calibration tool (Tasks 6-7) ✓; artifact (Task 3) ✓; `RealBaselineImpl` (Tasks 2,4) ✓; rewiring of Def 9 (Task 8 ref path) + Def 10 via `PosteriorAttributedBaseline` (Task 8 attrib) ✓; board-bucket scheme (Task 1) ✓; backoff ladder + "never worse than today" (Tasks 2,4 + no-regression test) ✓; class labeling from showdowns (Tasks 5-6) ✓; all validation gates documented ✓. The spec's "Open decisions" are all resolved in the decisions section.

**Type consistency:** `RealBaseline.probability(cls, action: PokerAction.Category, sizing: Option[Sizing], publicState: PublicState): Double` is the single signature implemented by both `ConstantRealBaseline` and `RealBaselineImpl` and consumed by `PosteriorAttributedBaseline` and `StrategicEngine.buildRefLikelihoodFn`. Artifact key `(StrategicClass, String, Street, PokerAction.Category)` is identical in `BaselineArtifact`, `BaselineArtifactIO`, `RealBaselineImpl`, and `BaselineCalibrationTool`. Bucket tokens come only from `BoardBucket.token`.

**Known confirmations deferred to the implementer (flagged inline, not hallucinated):** `PublicState`/`Chips`/`TableMap.empty` empty-construction in test fixtures; `Deck.full` + `DiscreteDistribution` import paths (mirror `HoldemEquity.scala`); `BridgeResult.fold` param names (mirror `StrategicSnapshot.scala:128-139`); the non-`StrategicRivalBelief` rival value for the no-regression assertion; the public `StrategicEngine` decide entry (mirror existing `StrategicEngineTest`). Each is pinned to a real file:line to confirm against — none are invented APIs.
