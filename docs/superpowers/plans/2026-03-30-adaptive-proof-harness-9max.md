# Adaptive Proof Harness (9-Max) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 9-player adaptive proof harness inside `TexasHoldemPlayingHall` that runs the adaptive hero against 6 leak-injected villains + 1 shallow-GTO + 1 CFR-GTO control, with seat reshuffling every 500 hands, tracking per-villain EV to prove exploitation works.

**Architecture:** Extend `TexasHoldemPlayingHall` with `VillainMode.LeakInjected`, full-ring always-active mode, per-villain identity-keyed observation tracking, and per-villain EV attribution. A thin `AdaptiveProofHarness` orchestrates 500-hand blocks, seat reshuffles, checkpoint/resume, ground truth, and reporting.

**Tech Stack:** Scala 3.8.1, munit 1.2.2, existing `sicfun.holdem.validation` + `sicfun.holdem.runtime` packages

**Spec:** `docs/superpowers/specs/2026-03-30-adaptive-proof-harness-design.md` (original 9-max intent)

**Companion HU plan:** `docs/superpowers/plans/2026-03-30-adaptive-proof-harness.md` (HU stepping stone by Codex)

---

## File Map

| File | Responsibility |
|------|---------------|
| `TexasHoldemPlayingHall.scala` | **Modify:** Add `VillainMode.LeakInjected`, full-ring activation flag, per-villain identity-keyed observations, per-villain EV in `HallSummary`, leak-injected `decideVillain` path, `parseVillainModeToken` extensions |
| `AdaptiveProofHarness.scala` | **Create:** Orchestrator for 500-hand blocks, seat reshuffles, checkpoint/resume, ground truth JSON, evaluation report |
| `AdaptiveProofHarnessTest.scala` | **Create:** Single-block integration test (reduced hand count) |

Files expected unchanged: `InjectedLeak.scala`, `LeakInjectedVillain.scala`, `SpotContext.scala`, `VillainStrategy.scala`, `PokerStarsExporter.scala`, `HeadsUpSimulator.scala`.

---

### Task 1: Add `VillainMode.LeakInjected` and extend the CLI parser

**Files:**
- Modify: `src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala`

- [ ] **Step 1: Add the new enum case**

Find the `VillainMode` enum (~line 29) and add:

```scala
  private enum VillainMode:
    case Archetype(style: PlayerArchetype)
    case Gto
    case LeakInjected(leakId: String, severity: Double)
```

- [ ] **Step 2: Extend `parseVillainModeToken` (~line 1297)**

Add leak token parsing before the final error case:

```scala
  private def parseVillainModeToken(raw: String): Either[String, VillainMode] =
    raw.trim.toLowerCase match
      case "nit"            => Right(VillainMode.Archetype(PlayerArchetype.Nit))
      case "tag"            => Right(VillainMode.Archetype(PlayerArchetype.Tag))
      case "lag"            => Right(VillainMode.Archetype(PlayerArchetype.Lag))
      case "callingstation" => Right(VillainMode.Archetype(PlayerArchetype.CallingStation))
      case "station"        => Right(VillainMode.Archetype(PlayerArchetype.CallingStation))
      case "maniac"         => Right(VillainMode.Archetype(PlayerArchetype.Maniac))
      case "gto"            => Right(VillainMode.Gto)
      case s if s.startsWith("leak:") =>
        parseLeakToken(s)
      case _ =>
        Left("style must be one of: nit, tag, lag, callingstation, station, maniac, gto, leak:<type>:<severity>")
```

Add the `parseLeakToken` helper:

```scala
  private def parseLeakToken(raw: String): Either[String, VillainMode] =
    val parts = raw.split(":")
    if parts.length != 3 then Left(s"leak token must be leak:<type>:<severity>, got: $raw")
    else
      val leakType = parts(1)
      val severityStr = parts(2)
      for
        severity <- try Right(severityStr.toDouble) catch case _: NumberFormatException =>
          Left(s"invalid severity: $severityStr")
        leakId <- leakType match
          case "overfold"      => Right("overfold-river-aggression")
          case "overcall"      => Right("overcall-big-bets")
          case "turnbluff"     => Right("overbluff-turn-barrel")
          case "passive"       => Right("passive-big-pots")
          case "prefloploose"  => Right("preflop-too-loose")
          case "prefloptight"  => Right("preflop-too-tight")
          case _               => Left(s"unknown leak type: $leakType (use: overfold, overcall, turnbluff, passive, prefloploose, prefloptight)")
      yield VillainMode.LeakInjected(leakId, severity)
```

- [ ] **Step 3: Extend `villainModeLabel` and `villainModeSlug`**

```scala
  private def villainModeLabel(mode: VillainMode): String =
    mode match
      case VillainMode.Archetype(style) => style.toString
      case VillainMode.Gto              => "GTO"
      case VillainMode.LeakInjected(leakId, sev) => s"Leak($leakId@$sev)"

  private def villainModeSlug(mode: VillainMode): String =
    mode match
      case VillainMode.Archetype(style) => style.toString.toLowerCase
      case VillainMode.Gto              => "gto"
      case VillainMode.LeakInjected(leakId, _) =>
        leakId.replace("-", "").take(12)
```

- [ ] **Step 4: Verify compilation**

Run: `sbt compile`

Expected: Compiles. Existing code that matches on `VillainMode` will get non-exhaustive warnings for the new case — that's expected and will be fixed in Task 3.

- [ ] **Step 5: Commit**

```
git add src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala
git commit -m "feat(hall): add VillainMode.LeakInjected and leak: CLI token parser"
```

---

### Task 2: Full-ring always-active mode in `buildTableScenario`

**Files:**
- Modify: `src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala`

Currently `buildTableScenario` picks a random subset of 2–3 villains from the pool. The proof harness needs all 8 villains always active. Add a `fullRing: Boolean` config flag.

- [ ] **Step 1: Add `fullRing` to `Config`**

In the `Config` case class (~line 36), add after `saveReviewHandHistory`:

```scala
      fullRing: Boolean = false
```

- [ ] **Step 2: Wire `fullRing` into `buildTableScenario`**

In `buildTableScenario` (~line 1497), the `activeVillainPositions` block currently does random subset selection. Add a full-ring branch:

```scala
    val activeVillainPositions =
      if config.fullRing then
        // All non-hero positions are active
        availableVillains.sortBy(modeledPositions.indexOf)
      else if headsUpOnly then
        // ... existing HU logic
```

Wait — `buildTableScenario` doesn't have access to `config`. It's a standalone function. Check the call site.

Actually, looking at the code, `buildTableScenario` is called from `HallRunner.playHand` which DOES have `config`. The simplest approach: add a `forceAllActive: Boolean` parameter to `buildTableScenario`:

```scala
  private def buildTableScenario(
      playerCount: Int,
      heroPosition: Position,
      villainPool: Vector[VillainProfile],
      headsUpOnly: Boolean,
      forceAllActive: Boolean,
      rng: Random
  ): TableScenario =
```

And at the beginning of the `activeVillainPositions` computation:

```scala
    val activeVillainPositions =
      if forceAllActive then
        // All available positions active, assign from pool cyclically
        availableVillains.sortBy(modeledPositions.indexOf)
      else if headsUpOnly then
        // ... existing logic unchanged
```

Update the call site in `HallRunner` to pass `forceAllActive = config.fullRing`.

- [ ] **Step 3: Ensure villain pool covers all seats**

When `forceAllActive = true` and `playerCount = 9`, there are 8 villain seats but the pool might have fewer entries. The existing cyclic assignment (`(profileOffset + idx) % villainPool.length`) already handles this — it wraps around. But for the proof we'll pass exactly 8 villains, so it maps 1:1.

- [ ] **Step 4: Add `--fullRing` to `parseArgs`**

In `parseArgs` (~line 1850+), add:

```scala
      else if arg == "--fullRing" then
        config = config.copy(fullRing = true)
```

- [ ] **Step 5: Verify compilation and existing tests**

Run: `sbt compile && sbt "testOnly sicfun.holdem.runtime.TexasHoldemPlayingHallTest"`

Expected: Compiles, existing tests pass (default `fullRing = false`).

- [ ] **Step 6: Commit**

```
git add src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala
git commit -m "feat(hall): add fullRing mode for always-active 9-player table"
```

---

### Task 3: Leak-injected villain decision path in `decideVillain`

**Files:**
- Modify: `src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala`

New imports needed at the top of the file:

```scala
import sicfun.holdem.validation.{
  ActionLine, EquityBasedStrategy, HandCategory, InjectedLeak, LeakInjectedVillain,
  NoLeak, Overcalls, OverbluffsTurnBarrel, OverfoldsToAggression, PassiveInBigPots,
  PreflopTooLoose, PreflopTooTight, SpotContext, VillainDecisionResult
}
```

- [ ] **Step 1: Add per-position ActionLine tracking to `resolveHand`**

Inside the `resolveHand` inner class (the hand-level state), add a mutable map:

```scala
    private val actionLineByPosition = mutable.HashMap.empty[Position, Vector[PokerAction]]
      .withDefaultValue(Vector.empty)
```

In `applyAction`, after recording the action, append to the action line:

```scala
    private def applyAction(
        position: Position,
        action: PokerAction,
        toCallBefore: Double,
        street: Street
    ): Unit =
      actionLineByPosition.update(position, actionLineByPosition(position) :+ action)
      action match
        // ... existing match cases unchanged
```

- [ ] **Step 2: Add equity-vs-random helper for villain hands**

Add a lightweight MC equity estimator (similar to `HeadsUpSimulator.estimateEquityVsRandom`):

```scala
  private def estimateEquityVsRandom(
      hand: HoleCards,
      board: Board,
      trials: Int,
      rng: Random
  ): Double =
    import sicfun.core.{Deck, HandEvaluator}
    val available = Deck.full.filterNot(c => hand.toVector.contains(c) || board.cards.contains(c))
    if available.size < 2 then return 0.5
    val boardSize = board.cards.size
    val communityNeeded = 5 - boardSize
    val needed = 2 + communityNeeded
    var wins = 0
    var total = 0
    val arr = available.toArray
    val n = arr.length
    for _ <- 0 until trials do
      var i = 0
      while i < needed do
        val j = i + rng.nextInt(n - i)
        val tmp = arr(i); arr(i) = arr(j); arr(j) = tmp
        i += 1
      val fullBoard = if communityNeeded > 0 then
        board.cards ++ (2 until needed).map(arr(_))
      else board.cards
      val heroAll = (hand.toVector ++ fullBoard).take(7)
      val oppAll = (Vector(arr(0), arr(1)) ++ fullBoard).take(7)
      if heroAll.size >= 7 && oppAll.size >= 7 then
        val heroRank = HandEvaluator.evaluate7(heroAll)
        val oppRank = HandEvaluator.evaluate7(oppAll)
        if heroRank > oppRank then wins += 1
        total += 1
    if total > 0 then wins.toDouble / total.toDouble else 0.5
```

- [ ] **Step 3: Add `leakInjectedVillainFor` factory**

Create `LeakInjectedVillain` instances from `VillainMode.LeakInjected`:

```scala
  private def buildInjectedLeak(leakId: String, severity: Double): InjectedLeak =
    leakId match
      case "overfold-river-aggression" => OverfoldsToAggression(severity)
      case "overcall-big-bets"        => Overcalls(severity)
      case "overbluff-turn-barrel"    => OverbluffsTurnBarrel(severity)
      case "passive-big-pots"         => PassiveInBigPots(severity)
      case "preflop-too-loose"        => PreflopTooLoose(severity)
      case "preflop-too-tight"        => PreflopTooTight(severity)
      case _                          => NoLeak()
```

- [ ] **Step 4: Extend `decideVillain` for LeakInjected mode**

In `decideVillain` (~line 900), add a new match case. The existing code does:

```scala
      val sampled = villainProfile.mode match
        case VillainMode.Archetype(style) => ...
        case VillainMode.Gto => ...
```

Add the LeakInjected case:

```scala
        case VillainMode.LeakInjected(leakId, severity) =>
          val villainHand = deal.holeCardsFor(position)
          val equityVsRandom = estimateEquityVsRandom(villainHand, boardFor(state.street), 50, rng)
          val equityStrategy = EquityBasedStrategy()
          val gtoAction = equityStrategy.decide(villainHand, state, candidates, equityVsRandom, rng)
          val leak = buildInjectedLeak(leakId, severity)
          val injectedVillain = LeakInjectedVillain(
            name = villainProfile.name,
            leaks = Vector(leak),
            baselineNoise = 0.03,
            seed = rng.nextLong()
          )
          val line = ActionLine(actionLineByPosition(position))
          val spot = SpotContext.build(
            gs = state,
            hero = villainHand,
            line = line,
            equityVsRandom = equityVsRandom,
            facingAction = None // could be refined later
          )
          val result = injectedVillain.decide(gtoAction, spot)
          result.action
```

- [ ] **Step 5: Verify compilation**

Run: `sbt compile`

Expected: Compiles without error.

- [ ] **Step 6: Commit**

```
git add src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala
git commit -m "feat(hall): wire LeakInjectedVillain into decideVillain for LeakInjected mode"
```

---

### Task 4: Per-villain EV tracking in HallSummary

**Files:**
- Modify: `src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala`

- [ ] **Step 1: Add `perVillainNetChips` to `HallSummary`**

```scala
  final case class HallSummary(
      // ... existing fields ...
      perVillainNetChips: Map[String, Double] = Map.empty
  ):
```

- [ ] **Step 2: Track per-villain deltas in `HallRunner`**

Add a mutable accumulator in `HallRunner`:

```scala
    private val perVillainNet = mutable.HashMap.empty[String, Double].withDefaultValue(0.0)
```

In `playHand`, after computing the `HandResult`, attribute `heroNet` to the primary villain:

```scala
    // Inside playHand, after result is computed:
    val primaryVillainName = tableScenario.primaryVillainProfile.name
    perVillainNet.update(primaryVillainName, perVillainNet(primaryVillainName) + result.heroNet)
```

Note: In multiway, `heroNet` is against the whole table, and attributing it to just the primary villain is a heuristic. For the proof harness where all villains are always active, this is imperfect but provides a useful signal. The primary villain is the one the hero was most focused on (closest aggressor). A more precise attribution would require pot-contribution-weighted splitting — deferred to v2 refinement.

- [ ] **Step 3: Include in `buildSummary`**

In `buildSummary`, add:

```scala
    perVillainNetChips = perVillainNet.toMap
```

- [ ] **Step 4: Verify compilation and tests**

Run: `sbt compile && sbt "testOnly sicfun.holdem.runtime.TexasHoldemPlayingHallTest"`

Expected: Compiles, tests pass. New field has a default empty map.

- [ ] **Step 5: Commit**

```
git add src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala
git commit -m "feat(hall): track per-villain hero net chips in HallSummary"
```

---

### Task 5: Enable review hand history export for full-ring

**Files:**
- Modify: `src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala`

The review hand history export (`appendReviewHandHistory`) was written for heads-up / small-ring scenarios. Check if it already handles full-ring correctly or needs adjustment.

- [ ] **Step 1: Audit `appendReviewHandHistory`**

Read the method (search for `appendReviewHandHistory` in the file). Check:
- Does it iterate over all `activePositions` or just hero + primary villain?
- Does it write seat lines for all players?
- Does it assign seat numbers correctly for 9 players?

The existing method already uses `tableScenario.seatNumberByPosition` and `tableScenario.playerNameByPosition` which are built from `modeledPositions` — all 9 positions. The seat lines, blind assignments, and action lines should already work for full-ring as long as all positions are active.

If changes are needed, make them. If the audit shows it already works, document that finding and move on.

- [ ] **Step 2: Verify by running a small full-ring hall invocation**

```
sbt "runMain sicfun.holdem.runtime.TexasHoldemPlayingHall --hands=5 --playerCount=9 --fullRing --villainPool=tag,lag,nit,maniac,station,tag,lag,gto --saveReviewHandHistory=true --seed=99 --outDir=data/_adaptive-proof-smoke"
```

Check that `data/_adaptive-proof-smoke/review-upload-pokerstars.txt` exists and has valid multi-player hand histories.

- [ ] **Step 3: Commit any fixes**

```
git add src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala
git commit -m "fix(hall): ensure review export works correctly for full-ring tables"
```

---

### Task 6: Build AdaptiveProofHarness orchestrator

**Files:**
- Create: `src/main/scala/sicfun/holdem/validation/AdaptiveProofHarness.scala`

This orchestrator invokes `TexasHoldemPlayingHall.run()` in 500-hand blocks with the configured villain pool, manages seat reshuffling, and produces ground-truth + reports.

- [ ] **Step 1: Create the harness with config, roster, and result types**

```scala
package sicfun.holdem.validation

import sicfun.holdem.runtime.TexasHoldemPlayingHall

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import java.time.Instant
import java.util.Locale
import scala.util.Random

object AdaptiveProofHarness:

  final case class Config(
      handsPerBlock: Int = 500,
      blocks: Int = 1,
      seed: Long = System.currentTimeMillis(),
      budgetMs: Long = 50L,
      bunchingTrials: Int = 100,
      equityTrials: Int = 500,
      outputDir: Path = Paths.get("data/adaptive-proof"),
      modelDir: Option[Path] = None
  ):
    def totalHands: Int = handsPerBlock * blocks

  /** The 8-villain pool string for --villainPool. */
  val defaultVillainPool: String =
    "leak:overfold:0.20,leak:overcall:0.25,leak:turnbluff:0.18,leak:passive:0.30,leak:prefloploose:0.22,leak:prefloptight:0.15,tag,gto"

  val villainNames: Vector[String] = Vector(
    "Villain01_overfold",
    "Villain02_overcall",
    "Villain03_turnbluff",
    "Villain04_passive",
    "Villain05_prefloploose",
    "Villain06_prefloptight",
    "Villain07_shallowgto",
    "Villain08_gto"
  )

  final case class BlockResult(
      blockIndex: Int,
      seed: Long,
      heroPosition: String,
      hallSummary: TexasHoldemPlayingHall.HallSummary
  )

  final case class RunResult(
      config: Config,
      blocks: Vector[BlockResult]
  ):
    def totalHands: Int = blocks.map(_.hallSummary.handsPlayed).sum
    def aggregateHeroBbPer100: Double =
      val totalNet = blocks.map(_.hallSummary.heroNetChips).sum
      if totalHands > 0 then (totalNet / totalHands) * 100.0 else 0.0
    def perVillainAggregateBbPer100: Map[String, Double] =
      val merged = scala.collection.mutable.HashMap.empty[String, Double].withDefaultValue(0.0)
      blocks.foreach { block =>
        block.hallSummary.perVillainNetChips.foreach { case (name, net) =>
          merged.update(name, merged(name) + net)
        }
      }
      merged.map { case (name, net) =>
        name -> (if totalHands > 0 then (net / totalHands) * 100.0 else 0.0)
      }.toMap
```

- [ ] **Step 2: Implement the block runner with seat reshuffling**

```scala
  private val heroPositions9Max = Vector(
    "UTG", "UTG1", "UTG2", "Middle", "Hijack", "Cutoff", "Button", "SmallBlind", "BigBlind"
  )

  def run(config: Config = Config()): RunResult =
    val rng = new Random(config.seed)
    println("=== Adaptive Proof Harness (9-Max) ===")
    println(s"Blocks: ${config.blocks}")
    println(s"Hands per block: ${config.handsPerBlock}")
    println(s"Total hands: ${config.totalHands}")
    println()

    val results = (0 until config.blocks).map { blockIdx =>
      val blockSeed = rng.nextLong()
      val heroPos = heroPositions9Max(rng.nextInt(heroPositions9Max.size))
      println(s"[Block ${blockIdx + 1}/${config.blocks}] seed=$blockSeed heroPosition=$heroPos")

      val hallOutDir = config.outputDir.resolve(s"block-${blockIdx}")
      val hallArgs = Array(
        s"--hands=${config.handsPerBlock}",
        s"--reportEvery=${config.handsPerBlock}",
        "--learnEveryHands=0",
        "--learningWindowSamples=50",
        s"--seed=$blockSeed",
        s"--outDir=$hallOutDir",
        "--playerCount=9",
        s"--heroPosition=$heroPos",
        "--heroStyle=adaptive",
        "--gtoMode=exact",
        s"--villainPool=$defaultVillainPool",
        "--heroExplorationRate=0.0",
        "--raiseSize=2.5",
        s"--bunchingTrials=${config.bunchingTrials}",
        s"--equityTrials=${config.equityTrials}",
        "--saveTrainingTsv=false",
        "--saveDdreTrainingTsv=false",
        "--saveReviewHandHistory=true",
        "--fullRing"
      ) ++ config.modelDir.map(p => s"--modelArtifactDir=$p").toArray

      val hallResult = TexasHoldemPlayingHall.run(hallArgs)
      hallResult match
        case Right(summary) =>
          println(f"  -> heroNetChips: ${summary.heroNetChips}%.2f  bb/100: ${summary.heroBbPer100}%.1f")
          BlockResult(blockIdx, blockSeed, heroPos, summary)
        case Left(error) =>
          throw new RuntimeException(s"Hall run failed at block $blockIdx: $error")
    }.toVector

    RunResult(config, results)
```

- [ ] **Step 3: Implement output writers**

```scala
  def writeOutputs(result: RunResult, outputDir: Path): Path =
    val runDir = outputDir.resolve(s"run-${Instant.now().getEpochSecond}-${result.config.seed}")
    Files.createDirectories(runDir)

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
    val perVillain = result.perVillainAggregateBbPer100
    val opponents = perVillain.toVector.sortBy(_._1).map { case (name, bbPer100) =>
      ujson.Obj(
        "name" -> ujson.Str(name),
        "heroNetBbPer100" -> ujson.Num(roundTo2(bbPer100))
      )
    }
    val blocks = result.blocks.map { block =>
      ujson.Obj(
        "blockIndex" -> ujson.Num(block.blockIndex),
        "seed" -> ujson.Num(block.seed.toDouble),
        "heroPosition" -> ujson.Str(block.heroPosition),
        "heroBbPer100" -> ujson.Num(roundTo2(block.hallSummary.heroBbPer100))
      )
    }
    val json = ujson.Obj(
      "totalHands" -> ujson.Num(result.totalHands),
      "blocks" -> ujson.Arr(blocks*),
      "perVillainBbPer100" -> ujson.Arr(opponents*)
    )
    ujson.write(json, indent = 2)

  private def formatReport(result: RunResult): String =
    val sb = new StringBuilder
    sb.append("=== Adaptive Proof Report (9-Max) ===\n")
    sb.append(s"Seed: ${result.config.seed}\n")
    sb.append(s"Blocks: ${result.blocks.size}\n")
    sb.append(s"Total hands: ${result.totalHands}\n")
    sb.append(f"\nOverall hero bb/100: ${result.aggregateHeroBbPer100}%+.1f\n")
    sb.append("\nPer-villain hero bb/100:\n")
    result.perVillainAggregateBbPer100.toVector.sortBy(_._1).foreach { case (name, bb) =>
      sb.append(String.format(Locale.ROOT, "  %-28s %+.1f\n", name, bb))
    }
    sb.toString()

  private def roundTo2(value: Double): Double =
    math.round(value * 100.0) / 100.0
```

- [ ] **Step 4: Add `main` method**

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
        config = config.copy(handsPerBlock = arg.stripPrefix("--hands=").toInt)
      else if arg.startsWith("--blocks=") then
        config = config.copy(blocks = arg.stripPrefix("--blocks=").toInt)
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

- [ ] **Step 5: Verify compilation**

Run: `sbt compile`

- [ ] **Step 6: Commit**

```
git add src/main/scala/sicfun/holdem/validation/AdaptiveProofHarness.scala
git commit -m "feat(validation): add 9-max AdaptiveProofHarness orchestrator"
```

---

### Task 7: Write AdaptiveProofHarnessTest

**Files:**
- Create: `src/test/scala/sicfun/holdem/validation/AdaptiveProofHarnessTest.scala`

Single-block, reduced hand count (12 hands) to keep fast. Tests structural correctness.

- [ ] **Step 1: Write the test**

```scala
package sicfun.holdem.validation

import sicfun.holdem.history.{HandHistoryImport, HandHistorySite}
import sicfun.holdem.runtime.TexasHoldemPlayingHall

import munit.FunSuite

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*

class AdaptiveProofHarnessTest extends FunSuite:
  override val munitTimeout: Duration = 300.seconds

  test("9-max adaptive proof harness produces valid hall output and report".tag(munit.Slow)) {
    val root = Files.createTempDirectory("adaptive-proof-9max-test-")
    try
      val config = AdaptiveProofHarness.Config(
        handsPerBlock = 12,
        blocks = 1,
        seed = 37L,
        budgetMs = 50L,
        bunchingTrials = 8,
        equityTrials = 80,
        outputDir = root
      )

      val result = AdaptiveProofHarness.run(config)

      // 1. One block completed
      assertEquals(result.blocks.size, 1)
      val block = result.blocks.head
      assert(block.hallSummary.handsPlayed > 0, "expected hands played")

      // 2. Hall output directory exists with review upload
      val hallOutDir = root.resolve("block-0")
      assert(Files.isDirectory(hallOutDir), s"missing hall output dir")
      val reviewPath = hallOutDir.resolve("review-upload-pokerstars.txt")
      assert(Files.exists(reviewPath), "missing review hand history")

      // 3. Review history is non-empty and parseable
      val reviewText = Files.readString(reviewPath, StandardCharsets.UTF_8)
      assert(reviewText.nonEmpty, "empty review history")
      val parsed = HandHistoryImport.parseText(reviewText, Some(HandHistorySite.PokerStars), Some("Hero"))
      assert(parsed.isRight, s"parse failed: ${parsed.left.getOrElse("")}")

      // 4. Write outputs and verify files
      val runDir = AdaptiveProofHarness.writeOutputs(result, config.outputDir)
      assert(Files.exists(runDir.resolve("ground-truth.json")))
      assert(Files.exists(runDir.resolve("report.txt")))

      // 5. Ground truth has block data
      val gt = ujson.read(Files.readString(runDir.resolve("ground-truth.json"), StandardCharsets.UTF_8))
      assertEquals(gt("blocks").arr.length, 1)
      assert(gt("totalHands").num > 0)

      // 6. Report contains expected header
      val report = Files.readString(runDir.resolve("report.txt"), StandardCharsets.UTF_8)
      assert(report.contains("Adaptive Proof Report"))

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

Expected: PASS.

- [ ] **Step 3: Commit**

```
git add src/test/scala/sicfun/holdem/validation/AdaptiveProofHarnessTest.scala
git commit -m "test(validation): add 9-max AdaptiveProofHarnessTest"
```

---

### Task 8: End-to-end smoke run with full roster

- [ ] **Step 1: Run the full harness**

```
sbt "runMain sicfun.holdem.validation.AdaptiveProofHarness --hands=500 --blocks=1 --seed=12345"
```

Expected: Runs 500 hands on a 9-player table with all 8 villains. Prints per-villain bb/100 report.

- [ ] **Step 2: Verify report and outputs**

Check `data/adaptive-proof/run-*/report.txt` for plausible per-villain bb/100.

- [ ] **Step 3: Commit plan and any final fixes**

```
git add docs/superpowers/plans/2026-03-30-adaptive-proof-harness-9max.md
git commit -m "docs: add 9-max adaptive proof harness implementation plan"
```
