# Adaptive Proof Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a heads-up proof harness that runs the adaptive hero against 5 leak-injected villains + 1 CFR GTO control, measures per-villain bb/100, and exports parseable PokerStars hand histories.

**Architecture:** `AdaptiveProofHarness` orchestrates independent HU sessions using `HeadsUpSimulator` (extended with explicit seat assignment and raise-response events). Each villain gets two mirrored legs (hero button / hero BB). `PokerStarsExporter` is extended to accept seat metadata. Results are written as ground-truth JSON + human-readable report.

**Tech Stack:** Scala 3.8.1, munit 1.2.2, existing `sicfun.holdem.validation` package

**Spec:** `docs/superpowers/specs/2026-03-30-adaptive-proof-harness-design.md`

---

## File Map

| File | Responsibility |
|------|---------------|
| `src/main/scala/sicfun/holdem/validation/HeadsUpSimulator.scala` | **Modify:** Add `heroIsButton` config, derive blinds/positions/acting order from it, track `heroRaiseResponses` per hand, add `heroIsButton` field to `HandRecord` |
| `src/main/scala/sicfun/holdem/validation/PokerStarsExporter.scala` | **Modify:** Accept `heroIsButton` parameter to control seat/button/blind lines in export |
| `src/main/scala/sicfun/holdem/validation/AdaptiveProofHarness.scala` | **Create:** Orchestrator — opponent roster, mirrored-leg session runner, adaptive update loop, ground-truth JSON, report |
| `src/test/scala/sicfun/holdem/validation/AdaptiveProofHarnessTest.scala` | **Create:** Reduced-matrix integration test (1 leaker + 1 control, small hand count) |

Files that remain unchanged: `InjectedLeak.scala`, `LeakInjectedVillain.scala`, `SpotContext.scala`, `VillainStrategy.scala`, `TexasHoldemPlayingHall.scala`.

---

### Task 1: Extend HandRecord with seat metadata and raise-response events

**Files:**
- Modify: `src/main/scala/sicfun/holdem/validation/HeadsUpSimulator.scala`

The existing `HandRecord` needs two new fields so downstream code knows who had the button and what raise responses occurred.

- [ ] **Step 1: Add `heroIsButton` and `heroRaiseResponses` to `HandRecord`**

In `HeadsUpSimulator.scala`, find the `HandRecord` case class (line ~22) and add two fields:

```scala
final case class HandRecord(
    handId: String,
    handNumber: Int,
    heroCards: HoleCards,
    villainCards: HoleCards,
    board: Board,
    actions: Vector[RecordedAction],
    heroNet: Double,
    streetsPlayed: Int,
    leakApplicableSpots: Int = 0,
    heroIsButton: Boolean = true,
    heroRaiseResponses: Vector[PokerAction] = Vector.empty
)
```

Default values ensure backward compatibility with existing callers.

- [ ] **Step 2: Verify existing tests still compile**

Run: `sbt "testOnly sicfun.holdem.validation.*" -- --exclude-tags=Slow`

Expected: All existing validation tests pass (no signature breakage from defaults).

- [ ] **Step 3: Commit**

```
git add src/main/scala/sicfun/holdem/validation/HeadsUpSimulator.scala
git commit -m "feat(validation): add heroIsButton and heroRaiseResponses to HandRecord"
```

---

### Task 2: Add explicit hero seat assignment to HeadsUpSimulator

**Files:**
- Modify: `src/main/scala/sicfun/holdem/validation/HeadsUpSimulator.scala`

The simulator currently hard-codes hero on the button (SB in HU). We need a `heroIsButton` constructor parameter that controls blind posting, acting order, `GameState.position` values, and the `villainPos` passed to the engine.

- [ ] **Step 1: Add `heroIsButton` constructor parameter**

Add to `HeadsUpSimulator` constructor (after `villainStrategy` param):

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
    villainStrategy: VillainStrategy = EquityBasedStrategy(),
    heroIsButton: Boolean = true
):
```

- [ ] **Step 2: Derive blind posting and positions from `heroIsButton`**

In `playHand`, replace the hardcoded blind posting block:

```scala
    // Current code (hero always SB/button):
    // heroStack -= smallBlind
    // villainStack -= bigBlind
    // pot = smallBlind + bigBlind
```

With seat-aware logic:

```scala
    val heroPosition = if heroIsButton then Position.Button else Position.BigBlind
    val villainPosition = if heroIsButton then Position.BigBlind else Position.Button

    // In HU, button posts SB and acts first preflop
    if heroIsButton then
      heroStack -= smallBlind
      villainStack -= bigBlind
    else
      heroStack -= bigBlind
      villainStack -= smallBlind
    pot = smallBlind + bigBlind
```

- [ ] **Step 3: Fix acting order and committed amounts**

Replace the per-street `heroTurn` and `heroCommitted`/`villainCommitted` init:

```scala
    // Preflop: button/SB acts first. Postflop: BB (OOP) acts first.
    var heroTurn = if street == Street.Preflop then heroIsButton else !heroIsButton

    var heroCommitted = if street == Street.Preflop then
      (if heroIsButton then smallBlind else bigBlind)
    else 0.0
    var villainCommitted = if street == Street.Preflop then
      (if heroIsButton then bigBlind else smallBlind)
    else 0.0
    var toCall = if street == Street.Preflop then
      (bigBlind - smallBlind)
    else 0.0
```

- [ ] **Step 4: Fix GameState.position for both hero and villain decisions**

In the hero decision branch, use `heroPosition` instead of hardcoded `Position.Button`:

```scala
    val gs = GameState(street, board, pot, math.max(0.0, toCall), heroPosition, heroStack, Vector.empty)
```

In the villain decision branch, use `villainPosition` instead of hardcoded `Position.BigBlind`:

```scala
    val gs = GameState(street, board, pot, math.max(0.0, toCall), villainPosition, villainStack, Vector.empty)
```

- [ ] **Step 5: Track hero raise responses**

Add a mutable buffer at the top of `playHand`:

```scala
    val heroRaiseResponseBuf = mutable.ArrayBuffer.empty[PokerAction]
```

After each villain action, check if it was in response to a hero raise. The simplest check: if the previous action in this street was a hero raise and `toCall > 0` for the villain, the villain's action is a raise response:

```scala
    // Inside the villain action branch, after validatedAction is determined:
    // Check if villain is responding to a hero raise
    val lastHeroAction = actions.lastOption.filter(_.player == heroName)
    if lastHeroAction.exists(_.action.isInstanceOf[PokerAction.Raise]) then
      heroRaiseResponseBuf += validatedAction
```

- [ ] **Step 6: Populate new HandRecord fields**

At the end of `playHand`, update the `HandRecord` construction:

```scala
    HandRecord(
      handId = f"SIM-${handNumber}%08d",
      handNumber = handNumber,
      heroCards = heroCards,
      villainCards = villainCards,
      board = finalBoard,
      actions = actions.toVector,
      heroNet = heroNet,
      streetsPlayed = streetsPlayed,
      leakApplicableSpots = applicableSpots,
      heroIsButton = heroIsButton,
      heroRaiseResponses = heroRaiseResponseBuf.toVector
    )
```

- [ ] **Step 7: Fix `decideHero` villain position reference**

In `decideHero`, the `villainPos` passed to `engine.decide(...)` must use the actual villain position:

```scala
    // Change: villainPos = Position.BigBlind
    // To:
    val villainPos = if heroIsButton then Position.BigBlind else Position.Button
```

This requires `heroIsButton` to be accessible from `decideHero`. Since it's a class field, it already is.

- [ ] **Step 8: Run existing tests**

Run: `sbt "testOnly sicfun.holdem.validation.HeadsUpSimulatorTest"`

Expected: PASS (default `heroIsButton = true` preserves existing behavior).

- [ ] **Step 9: Commit**

```
git add src/main/scala/sicfun/holdem/validation/HeadsUpSimulator.scala
git commit -m "feat(validation): support explicit hero seat assignment in HeadsUpSimulator"
```

---

### Task 3: Extend PokerStarsExporter for mirrored seat assignments

**Files:**
- Modify: `src/main/scala/sicfun/holdem/validation/PokerStarsExporter.scala`

Currently hard-codes hero as seat 1/button. For mirrored legs, need to respect `HandRecord.heroIsButton`.

- [ ] **Step 1: Update `appendHand` to use `heroIsButton` from record**

Replace the seat/blind block in `appendHand`:

```scala
  private def appendHand(sb: StringBuilder, record: HandRecord, heroName: String, villainName: String): Unit =
    val ts = BaseTimestamp.plusSeconds(record.handNumber.toLong)
    val heroSeat = if record.heroIsButton then 1 else 2
    val villainSeat = if record.heroIsButton then 2 else 1
    val buttonSeat = if record.heroIsButton then 1 else 2

    // In HU, button = SB. So:
    val sbPlayer = if record.heroIsButton then heroName else villainName
    val bbPlayer = if record.heroIsButton then villainName else heroName

    sb.append(s"PokerStars Hand #${record.handId}: Hold'em No Limit (${money(0.50)}/${money(1.00)}) - ${TimeFmt.format(ts)}\n")
    sb.append(s"Table 'Validation' 2-max Seat #$buttonSeat is the button\n")
    sb.append(s"Seat $heroSeat: $heroName (${money(StartingStack)} in chips)\n")
    sb.append(s"Seat $villainSeat: $villainName (${money(StartingStack)} in chips)\n")
    sb.append(s"$sbPlayer: posts small blind ${money(0.50)}\n")
    sb.append(s"$bbPlayer: posts big blind ${money(1.00)}\n")
```

The rest of `appendHand` (hole cards, actions, showdown, summary) is player-name-based and doesn't need changes.

- [ ] **Step 2: Run existing exporter tests**

Run: `sbt "testOnly sicfun.holdem.validation.PokerStarsExporterTest"`

Expected: PASS (existing records have `heroIsButton = true` by default).

- [ ] **Step 3: Commit**

```
git add src/main/scala/sicfun/holdem/validation/PokerStarsExporter.scala
git commit -m "feat(validation): seat-aware PokerStars export for mirrored HU legs"
```

---

### Task 4: Build AdaptiveProofHarness

**Files:**
- Create: `src/main/scala/sicfun/holdem/validation/AdaptiveProofHarness.scala`

This is the core orchestrator. It defines the opponent roster, runs mirrored-leg sessions per villain, wires adaptive updates, and produces outputs.

- [ ] **Step 1: Create AdaptiveProofHarness with opponent roster and config**

```scala
package sicfun.holdem.validation

import sicfun.holdem.engine.RealTimeAdaptiveEngine
import sicfun.holdem.equity.{TableFormat, TableRanges}
import sicfun.holdem.history.{HandHistoryImport, HandHistorySite}
import sicfun.holdem.model.{PokerActionModel, PokerActionModelArtifactIO}
import sicfun.holdem.types.PokerAction

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import java.time.Instant
import java.util.Locale
import scala.util.Random

object AdaptiveProofHarness:

  final case class OpponentSpec(
      name: String,
      leakId: String,
      severity: Double,
      strategyLabel: String,
      makeLeak: () => InjectedLeak,
      makeStrategy: () => VillainStrategy
  )

  final case class Config(
      handsPerLeg: Int = 250,
      seed: Long = System.currentTimeMillis(),
      budgetMs: Long = 50L,
      bunchingTrials: Int = 100,
      equityTrials: Int = 500,
      minEquityTrials: Int = 100,
      baselineNoise: Double = 0.03,
      modelDir: Option[Path] = None,
      outputDir: Path = Paths.get("data/adaptive-proof")
  ):
    def handsPerOpponent: Int = handsPerLeg * 2

  val defaultRoster: Vector[OpponentSpec] = Vector(
    OpponentSpec("Villain01_overfold", "overfold-river-aggression", 0.20, "equity-based",
      () => OverfoldsToAggression(0.20), () => EquityBasedStrategy()),
    OpponentSpec("Villain02_overcall", "overcall-big-bets", 0.25, "equity-based",
      () => Overcalls(0.25), () => EquityBasedStrategy()),
    OpponentSpec("Villain03_turnbluff", "overbluff-turn-barrel", 0.18, "equity-based",
      () => OverbluffsTurnBarrel(0.18), () => EquityBasedStrategy()),
    OpponentSpec("Villain04_prefloploose", "preflop-too-loose", 0.22, "equity-based",
      () => PreflopTooLoose(0.22), () => EquityBasedStrategy()),
    OpponentSpec("Villain05_prefloptight", "preflop-too-tight", 0.15, "equity-based",
      () => PreflopTooTight(0.15), () => EquityBasedStrategy()),
    OpponentSpec("Villain06_gto", NoLeak.Id, 0.0, "cfr-no-fallback",
      () => NoLeak(), () => CfrVillainStrategy(allowHeuristicFallback = false))
  )

  final case class LegResult(
      heroIsButton: Boolean,
      records: Vector[HandRecord],
      heroNetChips: Double,
      heroBbPer100: Double,
      leakFiredCount: Int
  )

  final case class OpponentResult(
      spec: OpponentSpec,
      buttonLeg: LegResult,
      bigBlindLeg: LegResult,
      combinedBbPer100: Double
  )

  final case class RunResult(
      seed: Long,
      handsPerOpponent: Int,
      opponents: Vector[OpponentResult]
  ):
    def leakerResults: Vector[OpponentResult] = opponents.filter(_.spec.leakId != NoLeak.Id)
    def controlResults: Vector[OpponentResult] = opponents.filter(_.spec.leakId == NoLeak.Id)
    def leakerAverageBbPer100: Double =
      val lr = leakerResults
      if lr.isEmpty then 0.0 else lr.map(_.combinedBbPer100).sum / lr.size
```

- [ ] **Step 2: Implement the session runner**

Add the `runOpponent` method that runs two mirrored legs and wires adaptive updates:

```scala
  private def runOpponent(
      spec: OpponentSpec,
      config: Config,
      opponentSeed: Long,
      actionModel: PokerActionModel
  ): OpponentResult =
    val tableRanges = TableRanges.defaults(TableFormat.HeadsUp)
    val heroEngine = new RealTimeAdaptiveEngine(
      tableRanges = tableRanges,
      actionModel = actionModel,
      bunchingTrials = config.bunchingTrials,
      defaultEquityTrials = config.equityTrials,
      minEquityTrials = config.minEquityTrials
    )

    val buttonLeg = runLeg(
      spec = spec, config = config, heroEngine = heroEngine,
      heroIsButton = true, legSeed = opponentSeed
    )
    val bbLeg = runLeg(
      spec = spec, config = config, heroEngine = heroEngine,
      heroIsButton = false, legSeed = opponentSeed + 1_000_000L
    )

    val totalHands = config.handsPerOpponent
    val totalNet = buttonLeg.heroNetChips + bbLeg.heroNetChips
    val combinedBbPer100 = (totalNet / totalHands) * 100.0

    OpponentResult(
      spec = spec,
      buttonLeg = buttonLeg,
      bigBlindLeg = bbLeg,
      combinedBbPer100 = combinedBbPer100
    )

  private def runLeg(
      spec: OpponentSpec,
      config: Config,
      heroEngine: RealTimeAdaptiveEngine,
      heroIsButton: Boolean,
      legSeed: Long
  ): LegResult =
    val villain = LeakInjectedVillain(
      name = spec.name,
      leaks = Vector(spec.makeLeak()),
      baselineNoise = if spec.leakId == NoLeak.Id then 0.0 else config.baselineNoise,
      seed = legSeed + 1L
    )
    val simulator = new HeadsUpSimulator(
      heroEngine = Some(heroEngine),
      villain = villain,
      seed = legSeed,
      budgetMs = config.budgetMs,
      villainStrategy = spec.makeStrategy(),
      heroIsButton = heroIsButton
    )

    val records = (1 to config.handsPerLeg).map { i =>
      val record = simulator.playHand(i)
      // Feed raise responses into the shared hero engine
      record.heroRaiseResponses.foreach(heroEngine.observeVillainResponseToRaise)
      record
    }.toVector

    val heroNet = records.map(_.heroNet).sum
    val leakFired = records.flatMap(_.actions).count(_.leakFired)

    LegResult(
      heroIsButton = heroIsButton,
      records = records,
      heroNetChips = heroNet,
      heroBbPer100 = (heroNet / config.handsPerLeg) * 100.0,
      leakFiredCount = leakFired
    )
```

- [ ] **Step 3: Implement the main `run` method**

```scala
  def run(
      config: Config = Config(),
      roster: Vector[OpponentSpec] = defaultRoster
  ): RunResult =
    val actionModel = config.modelDir
      .map(p => PokerActionModelArtifactIO.load(p).model)
      .getOrElse(PokerActionModel.uniform)

    val rng = new Random(config.seed)
    println("=== Adaptive Proof Harness ===")
    println(s"Opponents: ${roster.size}")
    println(s"Hands per opponent: ${config.handsPerOpponent} (${config.handsPerLeg} per leg)")
    println()

    val results = roster.zipWithIndex.map { case (spec, idx) =>
      val opponentSeed = rng.nextLong()
      println(s"[${idx + 1}/${roster.size}] Running ${spec.name} (${spec.leakId}, severity=${spec.severity}) ...")
      val result = runOpponent(spec, config, opponentSeed, actionModel)
      println(f"  -> combined bb/100: ${result.combinedBbPer100}%+.1f  button: ${result.buttonLeg.heroBbPer100}%+.1f  BB: ${result.bigBlindLeg.heroBbPer100}%+.1f  leaks fired: ${result.buttonLeg.leakFiredCount + result.bigBlindLeg.leakFiredCount}")
      result
    }

    RunResult(seed = config.seed, handsPerOpponent = config.handsPerOpponent, opponents = results)
```

- [ ] **Step 4: Implement output writers (ground truth + report)**

```scala
  def writeOutputs(result: RunResult, outputDir: Path): Path =
    val runDir = outputDir.resolve(f"run-${Instant.now().getEpochSecond}-${result.seed}")
    Files.createDirectories(runDir)

    // Per-opponent hand histories
    result.opponents.foreach { opp =>
      val villainDir = runDir.resolve(opp.spec.name)
      Files.createDirectories(villainDir)

      val buttonText = PokerStarsExporter.exportHands(opp.buttonLeg.records, "Hero", opp.spec.name)
      Files.writeString(villainDir.resolve("leg-button.txt"), buttonText, StandardCharsets.UTF_8)

      val bbText = PokerStarsExporter.exportHands(opp.bigBlindLeg.records, "Hero", opp.spec.name)
      Files.writeString(villainDir.resolve("leg-bigblind.txt"), bbText, StandardCharsets.UTF_8)

      val combinedText = buttonText + bbText
      Files.writeString(villainDir.resolve("combined.txt"), combinedText, StandardCharsets.UTF_8)
    }

    // Combined history (all opponents)
    val allText = result.opponents.map { opp =>
      val bt = PokerStarsExporter.exportHands(opp.buttonLeg.records, "Hero", opp.spec.name)
      val bbt = PokerStarsExporter.exportHands(opp.bigBlindLeg.records, "Hero", opp.spec.name)
      bt + bbt
    }.mkString
    Files.writeString(runDir.resolve("combined-history.txt"), allText, StandardCharsets.UTF_8)

    // Ground truth JSON
    val groundTruth = writeGroundTruthJson(result)
    Files.writeString(runDir.resolve("ground-truth.json"), groundTruth, StandardCharsets.UTF_8)

    // Report
    val report = formatReport(result)
    Files.writeString(runDir.resolve("report.txt"), report, StandardCharsets.UTF_8)
    println()
    println(report)

    runDir

  private def writeGroundTruthJson(result: RunResult): String =
    val opponents = result.opponents.map { opp =>
      ujson.Obj(
        "name" -> ujson.Str(opp.spec.name),
        "leakId" -> ujson.Str(opp.spec.leakId),
        "severity" -> ujson.Num(opp.spec.severity),
        "strategy" -> ujson.Str(opp.spec.strategyLabel),
        "heroNetBbPer100" -> ujson.Num(roundTo2(opp.combinedBbPer100)),
        "heroNetBbPer100ByLeg" -> ujson.Obj(
          "button" -> ujson.Num(roundTo2(opp.buttonLeg.heroBbPer100)),
          "bigBlind" -> ujson.Num(roundTo2(opp.bigBlindLeg.heroBbPer100))
        )
      )
    }
    val json = ujson.Obj(
      "handsPerOpponent" -> ujson.Num(result.handsPerOpponent),
      "legsPerOpponent" -> ujson.Num(2),
      "seed" -> ujson.Num(result.seed.toDouble),
      "opponents" -> ujson.Arr(opponents*)
    )
    ujson.write(json, indent = 2)

  private def formatReport(result: RunResult): String =
    val sb = new StringBuilder
    sb.append("=== Adaptive Proof Report ===\n")
    sb.append(s"Run seed: ${result.seed}\n")
    sb.append(s"Hands per opponent: ${result.handsPerOpponent}\n")
    sb.append("\nPer-opponent results:\n")
    result.opponents.foreach { opp =>
      sb.append(String.format(Locale.ROOT, "  %-24s bb/100: %+.1f   button: %+.1f   bigBlind: %+.1f\n",
        opp.spec.name, opp.combinedBbPer100, opp.buttonLeg.heroBbPer100, opp.bigBlindLeg.heroBbPer100))
    }
    sb.append(String.format(Locale.ROOT, "\nLeaker average bb/100: %+.1f\n", result.leakerAverageBbPer100))
    val controlBb = result.controlResults.headOption.map(_.combinedBbPer100).getOrElse(0.0)
    sb.append(String.format(Locale.ROOT, "GTO control bb/100: %+.1f\n", controlBb))
    sb.toString()

  private def roundTo2(value: Double): Double =
    math.round(value * 100.0) / 100.0
```

- [ ] **Step 5: Add a `main` method for CLI invocation**

```scala
  def main(args: Array[String]): Unit =
    val config = parseArgs(args)
    val result = run(config)
    val runDir = writeOutputs(result, config.outputDir)
    println(s"\nOutputs written to: ${runDir.toAbsolutePath.normalize()}")

  private def parseArgs(args: Array[String]): Config =
    var config = Config()
    args.foreach { arg =>
      if arg.startsWith("--hands=") then
        val total = arg.stripPrefix("--hands=").toInt
        config = config.copy(handsPerLeg = total / 2)
      else if arg.startsWith("--seed=") then
        config = config.copy(seed = arg.stripPrefix("--seed=").toLong)
      else if arg.startsWith("--budget=") then
        config = config.copy(budgetMs = arg.stripPrefix("--budget=").toLong)
      else if arg.startsWith("--output=") then
        config = config.copy(outputDir = Paths.get(arg.stripPrefix("--output=")))
      else if arg.startsWith("--model=") then
        config = config.copy(modelDir = Some(Paths.get(arg.stripPrefix("--model="))))
    }
    config
```

- [ ] **Step 6: Verify it compiles**

Run: `sbt compile`

Expected: Compiles without error.

- [ ] **Step 7: Commit**

```
git add src/main/scala/sicfun/holdem/validation/AdaptiveProofHarness.scala
git commit -m "feat(validation): add AdaptiveProofHarness for EV-oriented adaptive proof"
```

---

### Task 5: Write AdaptiveProofHarnessTest

**Files:**
- Create: `src/test/scala/sicfun/holdem/validation/AdaptiveProofHarnessTest.scala`

Reduced matrix: 1 leaker (`Overcalls` at 0.25) + 1 GTO control. Small hand count (20 per leg = 40 total per opponent). Tests structure/output correctness, not EV.

- [ ] **Step 1: Write the test**

```scala
package sicfun.holdem.validation

import sicfun.holdem.history.{HandHistoryImport, HandHistorySite}

import munit.FunSuite

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*

class AdaptiveProofHarnessTest extends FunSuite:
  override val munitTimeout: Duration = 300.seconds

  private val reducedRoster = Vector(
    AdaptiveProofHarness.OpponentSpec(
      "TestLeaker_overcall", "overcall-big-bets", 0.25, "equity-based",
      () => Overcalls(0.25), () => EquityBasedStrategy()
    ),
    AdaptiveProofHarness.OpponentSpec(
      "TestControl_gto", NoLeak.Id, 0.0, "cfr-no-fallback",
      () => NoLeak(), () => CfrVillainStrategy(allowHeuristicFallback = false)
    )
  )

  test("adaptive proof harness produces valid outputs for reduced matrix".tag(munit.Slow)) {
    val root = Files.createTempDirectory("adaptive-proof-test-")
    try
      val config = AdaptiveProofHarness.Config(
        handsPerLeg = 20,
        seed = 42L,
        budgetMs = 50L,
        bunchingTrials = 8,
        equityTrials = 80,
        minEquityTrials = 40,
        outputDir = root
      )

      val result = AdaptiveProofHarness.run(config, reducedRoster)
      val runDir = AdaptiveProofHarness.writeOutputs(result, config.outputDir)

      // 1. Harness completed for all opponents
      assertEquals(result.opponents.size, 2)

      // 2. Per-opponent directories created
      reducedRoster.foreach { spec =>
        val villainDir = runDir.resolve(spec.name)
        assert(Files.isDirectory(villainDir), s"missing dir: ${spec.name}")
        assert(Files.exists(villainDir.resolve("leg-button.txt")))
        assert(Files.exists(villainDir.resolve("leg-bigblind.txt")))
        assert(Files.exists(villainDir.resolve("combined.txt")))
      }

      // 3. Ground truth and report written
      val groundTruthPath = runDir.resolve("ground-truth.json")
      assert(Files.exists(groundTruthPath))
      val groundTruth = ujson.read(Files.readString(groundTruthPath, StandardCharsets.UTF_8))
      assertEquals(groundTruth("opponents").arr.length, 2)

      val reportPath = runDir.resolve("report.txt")
      assert(Files.exists(reportPath))
      val reportText = Files.readString(reportPath, StandardCharsets.UTF_8)
      assert(reportText.contains("Adaptive Proof Report"))

      // 4. History files are non-empty
      reducedRoster.foreach { spec =>
        val combined = runDir.resolve(spec.name).resolve("combined.txt")
        val text = Files.readString(combined, StandardCharsets.UTF_8)
        assert(text.nonEmpty, s"empty history for ${spec.name}")
      }

      // 5. History files parse successfully
      reducedRoster.foreach { spec =>
        val combined = runDir.resolve(spec.name).resolve("combined.txt")
        val text = Files.readString(combined, StandardCharsets.UTF_8)
        val parsed = HandHistoryImport.parseText(text, Some(HandHistorySite.PokerStars), Some("Hero"))
        assert(parsed.isRight, s"parse failed for ${spec.name}: ${parsed.left.getOrElse("")}")
        val hands = parsed.getOrElse(Vector.empty)
        assert(hands.nonEmpty, s"no hands parsed for ${spec.name}")
      }

      // 6. Control opponent records zero leak firings
      val controlResult = result.opponents.find(_.spec.leakId == NoLeak.Id).get
      val controlLeaksFired = controlResult.buttonLeg.leakFiredCount + controlResult.bigBlindLeg.leakFiredCount
      assertEquals(controlLeaksFired, 0, "GTO control should have zero leak firings")

      // 7. Opponent count matches
      assertEquals(groundTruth("opponents").arr.length, reducedRoster.size)

    finally
      deleteRecursively(root)
  }

  private def deleteRecursively(path: Path): Unit =
    if Files.exists(path) then
      val stream = Files.walk(path)
      try
        val all = stream.iterator().asScala.toVector.sortBy(_.toString.length).reverse
        all.foreach(Files.deleteIfExists)
      finally
        stream.close()
```

- [ ] **Step 2: Run the test**

Run: `sbt "testOnly sicfun.holdem.validation.AdaptiveProofHarnessTest"`

Expected: PASS. All 7 assertions hold.

- [ ] **Step 3: Commit**

```
git add src/test/scala/sicfun/holdem/validation/AdaptiveProofHarnessTest.scala
git commit -m "test(validation): add AdaptiveProofHarnessTest with reduced-matrix integration test"
```

---

### Task 6: End-to-end smoke run with full roster

Not a code task — a manual verification step.

- [ ] **Step 1: Run the full harness from CLI**

```
sbt "runMain sicfun.holdem.validation.AdaptiveProofHarness --hands=500 --seed=12345"
```

Expected: Runs all 6 opponents × 500 hands (250 per leg). Prints per-opponent bb/100. Writes outputs to `data/adaptive-proof/run-<timestamp>-12345/`.

- [ ] **Step 2: Verify the report**

Check that:
- Leaker bb/100 values are plausible (not all zero, not all extreme)
- GTO control bb/100 is near zero (within variance bounds)
- All history files exist and are non-trivial

- [ ] **Step 3: Commit the spec update to mark v1 as implemented**

```
git add docs/superpowers/specs/2026-03-30-adaptive-proof-harness-design.md
git add docs/superpowers/plans/2026-03-30-adaptive-proof-harness.md
git commit -m "docs: finalize adaptive proof harness v1 spec and plan"
```
