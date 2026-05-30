# P0 — Multiway bb/100 Measurement Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible, statistically-sound measurement of the current live engine's multiway (≤9-max) win rate — bb/100 with a bootstrap confidence interval vs a canonical exploitable field (Track A), plus a counterfactual bb/100 delta on real hand histories (Track B) — with no decision-engine changes.

**Architecture:** Track A drives the existing `TexasHoldemPlayingHall` via its public `run(args): Either[String, HallSummary]` entry, using a version-stamped villain-pool token string and deterministic flags, then computes a bootstrap CI over `HallSummary.perHandHeroNet`. Track B parses a real hand-history corpus with `HandHistoryImport`, runs `HandHistoryAnalyzer.analyzeWithHeroCards`, and aggregates per-decision `recommendedEv − actualEv` into a bb/100 delta. A new pure-stats module (`WinRateStats`) provides bootstrap CIs (none exists today). One PowerShell script captures both tracks.

**Tech Stack:** Scala 3.8.1, SBT, munit 1.2.2, PowerShell, `-Werror`.

**Spec:** `docs/superpowers/specs/2026-05-30-p0-multiway-winrate-harness-design.md`
**Builds on (do NOT duplicate):** `docs/superpowers/plans/2026-04-14-strategic-phase2-track-b.md` (OverlayMetrics, DecisionCorpusBenchmark, runner wiring — already implemented).

---

## Two flagged decisions (confirm at review)

1. **CRN / duplicate-mirror dealing is DEFERRED.** The hall uses a single `Random(config.seed)` shared by dealing and decisions (`TexasHoldemPlayingHall.scala:361`), so common-random-numbers across configs needs a deal-RNG-stream separation. P0 ships **independent-sample bootstrap CIs only** (satisfies the "CI clears zero" bar) and leaves CRN as a follow-up plan. No engine change in P0 for Track A.
2. **Track B requires a measurement-support importer change.** `ImportedHand` gains a `bigBlind: Double`, captured from the already-parsed-but-discarded "posts big blind" line. This is a parser/data-model change, NOT a decision-engine change. Without it, chips→bb on a mixed-stakes corpus is impossible.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/main/scala/sicfun/core/Metrics.scala` | Modify | Add public `percentile` + `stdDev` (promote the duplicated private helpers) |
| `src/main/scala/sicfun/holdem/bench/WinRateStats.scala` | Create | Pure stats: bb/100 estimator, bootstrap CI, percentile |
| `src/test/scala/sicfun/holdem/bench/WinRateStatsTest.scala` | Create | Determinism + known-variance CI sanity |
| `src/main/scala/sicfun/holdem/bench/CanonicalField.scala` | Create | Version-stamped 9-max villain-pool token + deterministic hall-arg builder |
| `src/test/scala/sicfun/holdem/bench/CanonicalFieldTest.scala` | Create | Arg-string + field-composition tests |
| `src/main/scala/sicfun/holdem/bench/MultiwayWinRateBenchmark.scala` | Create | Track A: run hall, bb/100+CI, artifacts |
| `src/test/scala/sicfun/holdem/bench/MultiwayWinRateBenchmarkTest.scala` | Create | Determinism + rigged-field edge detection |
| `src/main/scala/sicfun/holdem/history/HandHistoryImport.scala` | Modify | Capture `bigBlind` on `ImportedHand` |
| `src/test/scala/sicfun/holdem/history/HandHistoryImportBigBlindTest.scala` | Create | bigBlind captured from sample |
| `src/main/scala/sicfun/holdem/bench/CounterfactualHandHistoryBenchmark.scala` | Create | Track B: corpus → analyzer → bb/100 delta+CI |
| `src/test/resources/handhistory/p0-sample-9max.txt` | Create | Checked-in synthetic 9-max sample |
| `src/test/scala/sicfun/holdem/bench/CounterfactualHandHistoryBenchmarkTest.scala` | Create | Parses sample, yields a finite bb/100 delta + CI |
| `scripts/bench/capture-p0-baseline.ps1` | Create | One-command capture across both tracks |

---

### Task 1: Promote `percentile`/`stdDev` into `Metrics`

**Files:** Modify `src/main/scala/sicfun/core/Metrics.scala`; Test `src/test/scala/sicfun/core/MetricsPercentileTest.scala` (create)

- [ ] **Step 1: Write the failing test** — `src/test/scala/sicfun/core/MetricsPercentileTest.scala`

```scala
package sicfun.core

class MetricsPercentileTest extends munit.FunSuite:
  test("percentile on sorted-ish data uses linear interpolation") {
    val xs = Vector(1.0, 2.0, 3.0, 4.0)
    assertEqualsDouble(Metrics.percentile(xs, 0.0), 1.0, 1e-12)
    assertEqualsDouble(Metrics.percentile(xs, 1.0), 4.0, 1e-12)
    assertEqualsDouble(Metrics.percentile(xs, 0.5), 2.5, 1e-12) // p*(n-1)=1.5 -> between 2 and 3
  }
  test("percentile sorts input defensively") {
    assertEqualsDouble(Metrics.percentile(Vector(4.0, 1.0, 3.0, 2.0), 0.5), 2.5, 1e-12)
  }
  test("percentile on empty is 0.0 and single element is that element") {
    assertEqualsDouble(Metrics.percentile(Vector.empty, 0.5), 0.0, 1e-12)
    assertEqualsDouble(Metrics.percentile(Vector(7.0), 0.9), 7.0, 1e-12)
  }
  test("stdDev is sqrt of sample variance") {
    val xs = Vector(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
    assertEqualsDouble(Metrics.stdDev(xs), math.sqrt(Metrics.variance(xs)), 1e-12)
  }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly sicfun.core.MetricsPercentileTest"`
Expected: Compilation error — `Metrics.percentile` / `Metrics.stdDev` not found.

- [ ] **Step 3: Add the implementations** to `object Metrics` in `src/main/scala/sicfun/core/Metrics.scala` (append after `variance`):

```scala
  /** Sample standard deviation (sqrt of the unbiased n-1 variance). */
  def stdDev(values: Iterable[Double]): Double = math.sqrt(variance(values))

  /** Linear-interpolated percentile, q in [0,1]. Sorts defensively; empty -> 0.0. */
  def percentile(values: Iterable[Double], q: Double): Double =
    val sorted = values.toVector.sorted
    if sorted.isEmpty then 0.0
    else if sorted.sizeIs == 1 then sorted.head
    else
      val clamped = math.max(0.0, math.min(1.0, q))
      val p = clamped * (sorted.size - 1).toDouble
      val lo = math.floor(p).toInt
      val hi = math.ceil(p).toInt
      if lo == hi then sorted(lo)
      else
        val w = p - lo.toDouble
        sorted(lo) * (1.0 - w) + sorted(hi) * w
```

- [ ] **Step 4: Run test to verify it passes**

Run: `sbt "testOnly sicfun.core.MetricsPercentileTest"`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/core/Metrics.scala src/test/scala/sicfun/core/MetricsPercentileTest.scala
git commit -m "feat(core): add public Metrics.percentile and Metrics.stdDev"
```

---

### Task 2: `WinRateStats` — bb/100 estimator + bootstrap CI

**Files:** Create `src/main/scala/sicfun/holdem/bench/WinRateStats.scala`; Test `src/test/scala/sicfun/holdem/bench/WinRateStatsTest.scala`

- [ ] **Step 1: Write the failing test**

```scala
package sicfun.holdem.bench

class WinRateStatsTest extends munit.FunSuite:
  test("bbPer100 is mean per-hand bb times 100") {
    // mean = 0.5 bb/hand -> 50 bb/100
    assertEqualsDouble(WinRateStats.bbPer100(Vector(1.0, 0.0, 1.0, 0.0)), 50.0, 1e-9)
  }
  test("bootstrap CI is deterministic for a fixed seed") {
    val xs = Vector.tabulate(500)(i => if i % 2 == 0 then 2.0 else -1.0)
    val a = WinRateStats.bbPer100CI(xs, resamples = 1000, ciLevel = 0.95, seed = 7L)
    val b = WinRateStats.bbPer100CI(xs, resamples = 1000, ciLevel = 0.95, seed = 7L)
    assertEquals(a, b)
  }
  test("CI brackets the point estimate and lower<upper for noisy positive data") {
    val xs = Vector.tabulate(2000)(i => if i % 4 == 0 then 6.0 else -1.0) // mean +0.75 bb/hand = +75 bb/100
    val r = WinRateStats.bbPer100CI(xs, resamples = 2000, ciLevel = 0.95, seed = 1L)
    assert(r.lower < r.pointEstimate, s"lower ${r.lower} !< point ${r.pointEstimate}")
    assert(r.pointEstimate < r.upper, s"point ${r.pointEstimate} !< upper ${r.upper}")
    assert(r.lower > 0.0, s"expected CI to clear zero for strongly positive data, got lower=${r.lower}")
  }
  test("empty sample yields zero estimate and degenerate CI") {
    val r = WinRateStats.bbPer100CI(Vector.empty, resamples = 100, ciLevel = 0.95, seed = 1L)
    assertEqualsDouble(r.pointEstimate, 0.0, 1e-12)
    assertEqualsDouble(r.lower, 0.0, 1e-12)
    assertEqualsDouble(r.upper, 0.0, 1e-12)
  }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly sicfun.holdem.bench.WinRateStatsTest"`
Expected: Compilation error — `WinRateStats` not found.

- [ ] **Step 3: Write the implementation**

```scala
package sicfun.holdem.bench

import sicfun.core.Metrics
import java.util.Random

/** Pure statistics for win-rate measurement. No poker dependencies. */
object WinRateStats:

  /** A bb/100 point estimate with a bootstrap percentile confidence interval. */
  final case class WinRateCI(
      pointEstimate: Double,
      lower: Double,
      upper: Double,
      ciLevel: Double,
      sampleSize: Int,
      resamples: Int
  )

  /** bb/100 = mean per-hand bb result * 100. Per-hand values are in bb (1 chip = 1 bb in the normalized hall). */
  def bbPer100(perHandBb: Vector[Double]): Double =
    if perHandBb.isEmpty then 0.0 else Metrics.mean(perHandBb) * 100.0

  /** Bootstrap percentile CI on bb/100 over independent per-hand results.
    * Deterministic given `seed`. Resampling reduces ESTIMATOR variance; it does not
    * remove the game's intrinsic variance (see the design doc's determinism section).
    */
  def bbPer100CI(
      perHandBb: Vector[Double],
      resamples: Int = 2000,
      ciLevel: Double = 0.95,
      seed: Long = 42L
  ): WinRateCI =
    val n = perHandBb.length
    val point = bbPer100(perHandBb)
    if n == 0 then WinRateCI(0.0, 0.0, 0.0, ciLevel, 0, resamples)
    else
      val rng = new Random(seed)
      val means = Array.ofDim[Double](resamples)
      var b = 0
      while b < resamples do
        var sum = 0.0
        var i = 0
        while i < n do
          sum += perHandBb(rng.nextInt(n))
          i += 1
        means(b) = (sum / n.toDouble) * 100.0
        b += 1
      val tail = (1.0 - ciLevel) / 2.0
      val sorted = means.toVector
      WinRateCI(
        pointEstimate = point,
        lower = Metrics.percentile(sorted, tail),
        upper = Metrics.percentile(sorted, 1.0 - tail),
        ciLevel = ciLevel,
        sampleSize = n,
        resamples = resamples
      )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `sbt "testOnly sicfun.holdem.bench.WinRateStatsTest"`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/bench/WinRateStats.scala src/test/scala/sicfun/holdem/bench/WinRateStatsTest.scala
git commit -m "feat(bench): add WinRateStats bb/100 estimator with bootstrap CI"
```

---

### Task 3: `CanonicalField` — field token + deterministic hall args

**Files:** Create `src/main/scala/sicfun/holdem/bench/CanonicalField.scala`; Test `src/test/scala/sicfun/holdem/bench/CanonicalFieldTest.scala`

The hall takes `--villainPool` as a comma-separated token string (`station,nit,maniac,tag,lag,...`); archetypes are `nit|tag|lag|callingstation|station|maniac|gto` (verbatim from `parseVillainModeToken`, `TexasHoldemPlayingHall.scala:1696`). A 9-max table = hero + 8 villains, so the pool has 8 tokens.

- [ ] **Step 1: Write the failing test**

```scala
package sicfun.holdem.bench

class CanonicalFieldTest extends munit.FunSuite:
  test("9-max field has a version stamp and 8 villain tokens") {
    assertEquals(CanonicalField.NineMaxExploitable.version, "p0-field-v1")
    assertEquals(CanonicalField.NineMaxExploitable.villainTokens.length, 8)
    // only archetypes the hall accepts
    val allowed = Set("nit", "tag", "lag", "callingstation", "station", "maniac", "gto")
    CanonicalField.NineMaxExploitable.villainTokens.foreach(t =>
      assert(allowed.contains(t), s"unsupported villain token: $t"))
  }
  test("hallArgs is deterministic and disables learning + exploration") {
    val args = CanonicalField.NineMaxExploitable.hallArgs(heroStyle = "strategic", hands = 20000, seed = 42L, outDir = "data/p0/strategic")
    assert(args.contains("--playerCount=9"), args.mkString(" "))
    assert(args.contains("--heroStyle=strategic"))
    assert(args.contains("--learnEveryHands=0"))
    assert(args.contains("--heroExplorationRate=0.0"))
    assert(args.contains("--fullRing=true"))
    assert(args.contains("--seed=42"))
    assert(args.contains("--hands=20000"))
    assert(args.exists(_.startsWith("--villainPool=")))
  }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly sicfun.holdem.bench.CanonicalFieldTest"`
Expected: Compilation error — `CanonicalField` not found.

- [ ] **Step 3: Write the implementation**

```scala
package sicfun.holdem.bench

/** Version-stamped opponent fields for the P0 baseline. A field is a villain-pool
  * token string plus the deterministic hall arguments that pin a reproducible run.
  */
object CanonicalField:

  final case class Field(version: String, playerCount: Int, villainTokens: Vector[String]):
    require(villainTokens.length == playerCount - 1, "villain count must be playerCount - 1")

    /** Build CLI args for TexasHoldemPlayingHall.run. heroStyle in {adaptive,gto,strategic}. */
    def hallArgs(heroStyle: String, hands: Int, seed: Long, outDir: String): Array[String] =
      Array(
        s"--playerCount=$playerCount",
        s"--villainPool=${villainTokens.mkString(",")}",
        s"--heroStyle=$heroStyle",
        s"--hands=$hands",
        s"--seed=$seed",
        s"--outDir=$outDir",
        "--learnEveryHands=0",       // deterministic strategy: no online retraining
        "--heroExplorationRate=0.0", // no epsilon-greedy noise
        "--fullRing=true",           // all 8 villains always active
        "--saveTrainingTsv=false"
      )

  /** P0 canonical exploitable 9-max field: a recreational-leaning mix. */
  val NineMaxExploitable: Field = Field(
    version = "p0-field-v1",
    playerCount = 9,
    villainTokens = Vector("station", "station", "nit", "nit", "maniac", "maniac", "tag", "lag")
  )

  /** A trivially-exploitable field used to prove the harness can DETECT an edge (all calling stations). */
  val NineMaxRiggedStations: Field = Field(
    version = "p0-rigged-stations-v1",
    playerCount = 9,
    villainTokens = Vector.fill(8)("station")
  )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `sbt "testOnly sicfun.holdem.bench.CanonicalFieldTest"`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/bench/CanonicalField.scala src/test/scala/sicfun/holdem/bench/CanonicalFieldTest.scala
git commit -m "feat(bench): add CanonicalField (9-max exploitable + rigged) with deterministic hall args"
```

---

### Task 4: `MultiwayWinRateBenchmark` (Track A)

**Files:** Create `src/main/scala/sicfun/holdem/bench/MultiwayWinRateBenchmark.scala`; Test `src/test/scala/sicfun/holdem/bench/MultiwayWinRateBenchmarkTest.scala`

Uses `TexasHoldemPlayingHall.run(args): Either[String, HallSummary]` (verbatim signature, `TexasHoldemPlayingHall.scala:320`). `HallSummary.perHandHeroNet: Vector[Double]` is one signed hero bb delta per hand (`TexasHoldemPlayingHall.scala:138`).

- [ ] **Step 1: Write the failing test**

```scala
package sicfun.holdem.bench

import java.nio.file.Files

class MultiwayWinRateBenchmarkTest extends munit.FunSuite:
  test("same seed produces identical bb/100 and CI (computational replay)") {
    val tmp = Files.createTempDirectory("p0-trackA-det-")
    val r1 = MultiwayWinRateBenchmark.run(CanonicalField.NineMaxExploitable, heroStyle = "strategic", hands = 400, seed = 11L, outDir = tmp.resolve("a"))
    val r2 = MultiwayWinRateBenchmark.run(CanonicalField.NineMaxExploitable, heroStyle = "strategic", hands = 400, seed = 11L, outDir = tmp.resolve("b"))
    assert(r1.isRight && r2.isRight, s"$r1 / $r2")
    assertEquals(r1.toOption.get.ci.pointEstimate, r2.toOption.get.ci.pointEstimate)
    assertEquals(r1.toOption.get.ci.lower, r2.toOption.get.ci.lower)
  }
  test("rigged all-station field: hero edge CI clears zero (harness detects an edge)") {
    val tmp = Files.createTempDirectory("p0-trackA-rig-")
    val res = MultiwayWinRateBenchmark.run(CanonicalField.NineMaxRiggedStations, heroStyle = "strategic", hands = 3000, seed = 5L, outDir = tmp)
    assert(res.isRight, res.toString)
    val ci = res.toOption.get.ci
    assert(ci.sampleSize == 3000, s"expected 3000 hands, got ${ci.sampleSize}")
    assert(ci.lower > 0.0, s"expected positive lower CI vs all-calling-stations, got ${ci.lower} (point ${ci.pointEstimate})")
  }
```

Note: the rigged-field assertion is the one behavioral claim worth a generous hand count (3000); if it proves flaky in CI, raise `hands`, not lower the bar — a real engine must beat pure calling stations.

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly sicfun.holdem.bench.MultiwayWinRateBenchmarkTest"`
Expected: Compilation error — `MultiwayWinRateBenchmark` not found.

- [ ] **Step 3: Write the implementation**

```scala
package sicfun.holdem.bench

import sicfun.holdem.runtime.TexasHoldemPlayingHall

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}

/** Track A: run hall self-play vs a canonical field and report bb/100 with a bootstrap CI. */
object MultiwayWinRateBenchmark:

  final case class Result(
      fieldVersion: String,
      heroStyle: String,
      hands: Int,
      seed: Long,
      ci: WinRateStats.WinRateCI,
      heroBbPer100Aggregate: Double
  )

  def run(
      field: CanonicalField.Field,
      heroStyle: String,
      hands: Int,
      seed: Long,
      outDir: Path,
      resamples: Int = 2000,
      ciLevel: Double = 0.95
  ): Either[String, Result] =
    val args = field.hallArgs(heroStyle, hands, seed, outDir.toString)
    TexasHoldemPlayingHall.run(args).map { summary =>
      val ci = WinRateStats.bbPer100CI(summary.perHandHeroNet, resamples, ciLevel, seed)
      val result = Result(field.version, heroStyle, summary.handsPlayed, seed, ci, summary.heroBbPer100)
      writeSummary(result, outDir)
      result
    }

  private def writeSummary(r: Result, outDir: Path): Unit =
    val path = outDir.resolve("winrate-summary.txt")
    Files.createDirectories(outDir)
    val lines = Vector(
      s"fieldVersion: ${r.fieldVersion}",
      s"heroStyle: ${r.heroStyle}",
      s"hands: ${r.hands}",
      s"seed: ${r.seed}",
      f"bbPer100Point: ${r.ci.pointEstimate}%.4f",
      f"bbPer100Lower${(r.ci.ciLevel * 100).toInt}: ${r.ci.lower}%.4f",
      f"bbPer100Upper${(r.ci.ciLevel * 100).toInt}: ${r.ci.upper}%.4f",
      f"hallAggregateBbPer100: ${r.heroBbPer100Aggregate}%.4f",
      s"ciClearsZero: ${r.ci.lower > 0.0}"
    )
    Files.write(path, lines.mkString(System.lineSeparator()).getBytes(StandardCharsets.UTF_8))

  /** CLI: arg0=heroStyle (default strategic), arg1=hands (default 50000), arg2=seed (default 42), arg3=outDir. */
  def main(args: Array[String]): Unit =
    val heroStyle = args.headOption.getOrElse("strategic")
    val hands = args.lift(1).flatMap(_.toIntOption).getOrElse(50000)
    val seed = args.lift(2).flatMap(_.toLongOption).getOrElse(42L)
    val outDir = Path.of(args.lift(3).getOrElse(s"data/p0-winrate/$heroStyle"))
    run(CanonicalField.NineMaxExploitable, heroStyle, hands, seed, outDir) match
      case Right(r) =>
        println(f"[$heroStyle] bb/100 = ${r.ci.pointEstimate}%.3f  CI${(r.ci.ciLevel*100).toInt}=[${r.ci.lower}%.3f, ${r.ci.upper}%.3f]  (n=${r.ci.sampleSize})")
      case Left(err) =>
        System.err.println(s"benchmark failed: $err"); sys.exit(1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `sbt "testOnly sicfun.holdem.bench.MultiwayWinRateBenchmarkTest"`
Expected: PASS (2 tests). The rigged-field test runs 3000 hands of 9-max self-play; allow up to ~1–2 min.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/bench/MultiwayWinRateBenchmark.scala src/test/scala/sicfun/holdem/bench/MultiwayWinRateBenchmarkTest.scala
git commit -m "feat(bench): Track A multiway bb/100 benchmark with bootstrap CI + rigged-field edge test"
```

---

### Task 5: Capture the big blind in `HandHistoryImport` (measurement-support)

**Files:** Modify `src/main/scala/sicfun/holdem/history/HandHistoryImport.scala`; Test `src/test/scala/sicfun/holdem/history/HandHistoryImportBigBlindTest.scala`

`ImportedHand` (`HandHistoryImport.scala:92-102`) has no big-blind field; the importer matches `"posts big blind "` (~`:456`) and discards the amount. Add `bigBlind: Double = 0.0` and populate it.

- [ ] **Step 1: Write the failing test**

```scala
package sicfun.holdem.history

class HandHistoryImportBigBlindTest extends munit.FunSuite:
  // Minimal PokerStars-style 9-max header + blinds; reuse the project's existing sample style.
  private val sample =
    """PokerStars Hand #900000001:  Hold'em No Limit ($1/$2 USD) - 2026/01/01 12:00:00 ET
      |Table 'P0Test' 9-max Seat #1 is the button
      |Seat 1: Hero ($200 in chips)
      |Seat 2: Villain2 ($200 in chips)
      |Villain2: posts small blind $1
      |Hero: posts big blind $2
      |*** HOLE CARDS ***
      |Dealt to Hero [As Ks]
      |Villain2: folds
      |Uncalled bet ($1) returned to Hero
      |Hero collected $2 from pot
      |*** SUMMARY ***
      |Total pot $2 | Rake $0
      |""".stripMargin

  test("importer captures the big blind amount") {
    val hands = HandHistoryImport.parseText(sample, Some(HandHistorySite.PokerStars), Some("Hero"))
    assert(hands.isRight, hands.toString)
    val h = hands.toOption.get.head
    assertEqualsDouble(h.bigBlind, 2.0, 1e-9)
  }
```

If the project already has a canonical PokerStars sample fixture, point the test at it instead of this inline string — but the assertion (`h.bigBlind == 2.0`) is the contract.

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly sicfun.holdem.history.HandHistoryImportBigBlindTest"`
Expected: Compilation error — `ImportedHand` has no member `bigBlind`.

- [ ] **Step 3: Add the field and capture it**

In `ImportedHand` (`HandHistoryImport.scala:92-102`), add a field (default keeps existing constructors source-compatible):

```scala
    showdownCards: Map[String, HoleCards] = Map.empty,
    bigBlind: Double = 0.0
```

In the mutable `ImportState`, add a `var bigBlind: Double = 0.0`. At the existing "posts big blind" match (~`:456-459`), capture the parsed amount:

```scala
        // existing: matches "posts big blind <amount>"; previously amount was folded into pot only.
        state.bigBlind = amount   // ADD: retain the big-blind size for bb-scale conversion
```

When constructing the final `ImportedHand` (search for `ImportedHand(` in the build/finalize step), pass `bigBlind = state.bigBlind`.

- [ ] **Step 4: Run test to verify it passes**

Run: `sbt "testOnly sicfun.holdem.history.HandHistoryImportBigBlindTest"`
Expected: PASS. Then run the existing import suite to confirm no regression: `sbt "testOnly sicfun.holdem.history.HandHistoryImportTest"` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/history/HandHistoryImport.scala src/test/scala/sicfun/holdem/history/HandHistoryImportBigBlindTest.scala
git commit -m "feat(history): retain big-blind size on ImportedHand for bb-scale conversion"
```

---

### Task 6: Checked-in synthetic 9-max sample

**Files:** Create `src/test/resources/handhistory/p0-sample-9max.txt`

- [ ] **Step 1: Create a small, valid multiway sample** that the importer parses and that contains at least one hero post-flop decision. Use the same PokerStars format the existing `HandHistoryImportTest` validates (mirror its structure exactly; 3–5 hands, 9-max, hero in varied positions, at least one hand reaching the flop/turn with a hero bet/call). Keep stakes uniform ($1/$2) so bb=2.0 throughout.

(The exact text must match the parser's expectations — copy a known-good hand block from the existing import test fixtures and duplicate/edit it into a 9-max corpus. Do not invent a format.)

- [ ] **Step 2: Verify it parses**

Add to `HandHistoryImportBigBlindTest` (or a new small test):

```scala
  test("p0 synthetic 9-max resource parses into hands with bigBlind set") {
    val txt = scala.io.Source.fromResource("handhistory/p0-sample-9max.txt").mkString
    val hands = HandHistoryImport.parseText(txt, Some(HandHistorySite.PokerStars), Some("Hero"))
    assert(hands.isRight, hands.toString)
    assert(hands.toOption.get.nonEmpty)
    hands.toOption.get.foreach(h => assert(h.bigBlind > 0.0, s"hand ${h.handId} missing bigBlind"))
  }
```

Run: `sbt "testOnly sicfun.holdem.history.HandHistoryImportBigBlindTest"` — Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/test/resources/handhistory/p0-sample-9max.txt src/test/scala/sicfun/holdem/history/HandHistoryImportBigBlindTest.scala
git commit -m "test(history): add checked-in synthetic 9-max hand-history sample"
```

---

### Task 7: `CounterfactualHandHistoryBenchmark` (Track B)

**Files:** Create `src/main/scala/sicfun/holdem/bench/CounterfactualHandHistoryBenchmark.scala`; Test `src/test/scala/sicfun/holdem/bench/CounterfactualHandHistoryBenchmarkTest.scala`

Glue mirrors the (private) `HandHistoryReviewService.analyzeHand` (`HandHistoryReviewService.scala:221-233`): per `ImportedHand`, build `TableRanges` from player count, a `RealTimeAdaptiveEngine`, and call `HandHistoryAnalyzer.analyzeWithHeroCards(events, heroPlayerId, heroCards, engine, tableRanges, availablePositions, budgetMs, rng): Vector[AnalyzedDecision]`. Counterfactual hero gain per decision = `recommendedEv - actualEv` (chips); `evDifference = actualEv - recommendedEv`, so gain `= -evDifference`. Convert to bb via `hand.bigBlind`.

- [ ] **Step 1: Write the failing test**

```scala
package sicfun.holdem.bench

class CounterfactualHandHistoryBenchmarkTest extends munit.FunSuite:
  test("counterfactual benchmark on the synthetic corpus yields a finite bb/100 delta + CI") {
    val txt = scala.io.Source.fromResource("handhistory/p0-sample-9max.txt").mkString
    val res = CounterfactualHandHistoryBenchmark.runText(txt, heroName = "Hero", seed = 3L, resamples = 500)
    assert(res.isRight, res.toString)
    val r = res.toOption.get
    assert(r.decisions > 0, "expected at least one analyzed hero decision")
    assert(r.ci.pointEstimate.isFinite, s"non-finite point estimate ${r.ci.pointEstimate}")
    assert(r.ci.lower <= r.ci.pointEstimate && r.ci.pointEstimate <= r.ci.upper)
    // sign convention: following sicfun should not be worse than the actual line on average by construction
    assert(r.ci.pointEstimate >= -1e-6, s"recommended-minus-actual should be >= 0 on average, got ${r.ci.pointEstimate}")
  }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly sicfun.holdem.bench.CounterfactualHandHistoryBenchmarkTest"`
Expected: Compilation error — `CounterfactualHandHistoryBenchmark` not found.

- [ ] **Step 3: Write the implementation**

```scala
package sicfun.holdem.bench

import sicfun.holdem.equity.{TableFormat, TableRanges}
import sicfun.holdem.history.{HandHistoryImport, HandHistorySite, ImportedHand}
import sicfun.holdem.model.PokerActionModel
import sicfun.holdem.runtime.HandHistoryAnalyzer
import sicfun.holdem.engine.villain.RealTimeAdaptiveEngine
import sicfun.holdem.types.Position

import java.util.Random

/** Track B: counterfactual EV of following sicfun vs the actually-played line on real hands,
  * aggregated to a bb/100 delta with a bootstrap CI. Reuses HandHistoryAnalyzer; no engine change.
  */
object CounterfactualHandHistoryBenchmark:

  final case class Result(hands: Int, decisions: Int, ci: WinRateStats.WinRateCI)

  /** Per-hand counterfactual hero gain in bb: sum over hero decisions of (recommendedEv - actualEv) / bigBlind. */
  private def perHandGainsBb(hand: ImportedHand, seed: Long): Vector[Double] =
    if hand.heroHoleCards.isEmpty || hand.heroName.isEmpty || hand.bigBlind <= 0.0 then Vector.empty
    else
      val tableRanges = TableRanges.defaults(TableFormat.forPlayerCount(hand.players.length))
      val engine = new RealTimeAdaptiveEngine(
        tableRanges = tableRanges,
        actionModel = PokerActionModel.uniform,
        bunchingTrials = 1,
        defaultEquityTrials = 400,
        minEquityTrials = 200
      )
      val decisions = HandHistoryAnalyzer.analyzeWithHeroCards(
        events = hand.events,
        heroPlayerId = hand.heroName.get,
        heroCards = hand.heroHoleCards.get,
        engine = engine,
        tableRanges = tableRanges,
        availablePositions = hand.players.iterator.map(_.position).toSet,
        budgetMs = 2000L,
        rng = new Random(seed ^ hand.handId.hashCode.toLong)
      )
      // gain of following sicfun = recommendedEv - actualEv = -evDifference (chips) -> bb
      val gainBb = decisions.iterator.map(d => (-d.evDifference) / hand.bigBlind).sum
      Vector(gainBb)

  def runHands(hands: Vector[ImportedHand], seed: Long, resamples: Int, ciLevel: Double): Result =
    val perHand = hands.flatMap(h => perHandGainsBb(h, seed))
    val decisionCount = hands.iterator.map(h =>
      if h.heroHoleCards.isDefined && h.bigBlind > 0.0 then 1 else 0).sum // hands with a hero seat; decisions counted in perHand
    Result(hands.length, perHand.length, WinRateStats.bbPer100CI(perHand, resamples, ciLevel, seed))

  def runText(text: String, heroName: String, seed: Long = 42L, resamples: Int = 2000, ciLevel: Double = 0.95): Either[String, Result] =
    HandHistoryImport.parseText(text, None, Some(heroName)).map(hands => runHands(hands, seed, resamples, ciLevel))

  def runFile(path: java.nio.file.Path, heroName: String, seed: Long = 42L, resamples: Int = 2000, ciLevel: Double = 0.95): Either[String, Result] =
    HandHistoryImport.parseFile(path, None, Some(heroName)).map(hands => runHands(hands, seed, resamples, ciLevel))

  /** CLI: arg0=corpusFile, arg1=heroName, arg2=seed (default 42). */
  def main(args: Array[String]): Unit =
    if args.length < 2 then { System.err.println("usage: CounterfactualHandHistoryBenchmark <corpusFile> <heroName> [seed]"); sys.exit(1) }
    val seed = args.lift(2).flatMap(_.toLongOption).getOrElse(42L)
    runFile(java.nio.file.Path.of(args(0)), args(1), seed) match
      case Right(r) =>
        println(f"counterfactual bb/100 = ${r.ci.pointEstimate}%.3f  CI=[${r.ci.lower}%.3f, ${r.ci.upper}%.3f]  (hands=${r.hands}, perHandSamples=${r.ci.sampleSize})")
      case Left(err) => System.err.println(s"failed: $err"); sys.exit(1)
```

Note for the implementer: confirm the `RealTimeAdaptiveEngine` constructor params against `engine/villain/RealTimeAdaptiveEngine.scala` and `PokerActionModel.uniform`'s name; the call shape mirrors `HandHistoryReviewService.newEngine`. If the production service loads a trained model rather than `uniform`, prefer that for fidelity (open decision below).

- [ ] **Step 4: Run test to verify it passes**

Run: `sbt "testOnly sicfun.holdem.bench.CounterfactualHandHistoryBenchmarkTest"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/bench/CounterfactualHandHistoryBenchmark.scala src/test/scala/sicfun/holdem/bench/CounterfactualHandHistoryBenchmarkTest.scala
git commit -m "feat(bench): Track B counterfactual-EV bb/100 delta on real hands with bootstrap CI"
```

---

### Task 8: `capture-p0-baseline.ps1`

**Files:** Create `scripts/bench/capture-p0-baseline.ps1`

- [ ] **Step 1: Write the script**

```powershell
<#
.SYNOPSIS  Captures the P0 multiway baseline: Track A (hall self-play bb/100 + CI) and, if a corpus is given, Track B (counterfactual bb/100).
.PARAMETER OutDir   Root output dir. Default: data/p0-baseline
.PARAMETER Hands    Hands per Track-A run. Default: 50000
.PARAMETER Seed     RNG seed. Default: 42
.PARAMETER Corpus   Optional real hand-history file for Track B.
.PARAMETER HeroName Hero name in the corpus (required if -Corpus is set).
#>
param(
  [string]$OutDir = "data/p0-baseline",
  [int]$Hands = 50000,
  [long]$Seed = 42,
  [string]$Corpus = "",
  [string]$HeroName = ""
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

foreach ($style in @("strategic", "adaptive")) {
  Write-Host "== Track A: hero=$style, hands=$Hands, seed=$Seed =="
  sbt "runMain sicfun.holdem.bench.MultiwayWinRateBenchmark $style $Hands $Seed $OutDir/trackA-$style"
  if ($LASTEXITCODE -ne 0) { throw "Track A ($style) failed" }
}

if ($Corpus -ne "") {
  if ($HeroName -eq "") { throw "-HeroName is required when -Corpus is set" }
  Write-Host "== Track B: counterfactual on $Corpus =="
  sbt "runMain sicfun.holdem.bench.CounterfactualHandHistoryBenchmark $Corpus $HeroName $Seed"
  if ($LASTEXITCODE -ne 0) { throw "Track B failed" }
} else {
  Write-Host "Track B skipped (no -Corpus). Real histories stay local; supply one to capture the counterfactual delta."
}

Write-Host "Baseline artifacts under $OutDir"
```

- [ ] **Step 2: Syntax check**

Run: `pwsh -NoProfile -Command "Get-Command ./scripts/bench/capture-p0-baseline.ps1 | Out-Null; 'ok'"`
Expected: prints `ok` with no parse error.

- [ ] **Step 3: Commit**

```bash
git add scripts/bench/capture-p0-baseline.ps1
git commit -m "feat(scripts): add P0 baseline capture (Track A + optional Track B)"
```

---

## Self-Review

**Spec coverage:**
- bb/100 + bootstrap CI vs canonical 9-max field → Tasks 2,3,4. ✔
- Counterfactual bb/100 delta on real hands → Tasks 5,6,7. ✔
- Saved artifacts (seed/config/summary) → Task 4 `writeSummary`, Task 8 script. ✔
- Rigged-field edge-detection test → Task 4. ✔
- Strategy-determinism: partially covered as computational replay (Task 4 same-seed test). **Gap:** the spec also calls for a pure "same info-set + seed → same decision" unit test at the engine boundary. That assertion belongs to P1+ engine work, not the measurement harness; noted as out-of-P0 here to avoid touching the engine. If you want it in P0, add a read-only test calling the overlay decision twice with a fixed seed — but it reaches into engine APIs this harness otherwise doesn't.
- Determinism/variance philosophy → encoded in `WinRateStats` doc + the "two flagged decisions" header. ✔

**Placeholder scan:** Task 6 (synthetic sample) intentionally defers the exact hand text to "copy a known-good fixture" rather than inventing a format — this is a guardrail against hallucinating the parser grammar, not a lazy placeholder; the contract (parses + bigBlind>0) is asserted. Task 5/7 contain "confirm against file" notes for two signatures (importer finalize site; `RealTimeAdaptiveEngine` ctor) that the implementer verifies in-file — flagged explicitly rather than guessed.

**Type consistency:** `WinRateStats.WinRateCI`, `CanonicalField.Field`, `MultiwayWinRateBenchmark.Result`, `CounterfactualHandHistoryBenchmark.Result` used consistently across tasks; `bbPer100CI(perHand, resamples, ciLevel, seed)` signature matches all call sites.

## Open decisions (resolve during execution, low-risk defaults given)
- Track B engine model: `PokerActionModel.uniform` (default, fast) vs the production-loaded model (higher fidelity). Default uniform; switch if the live service uses a trained artifact.
- Hands/resamples for the real baseline run (50000 hands / 2000 resamples defaults).
- Whether the strategy-determinism unit test is in P0 (currently out, to keep P0 engine-free).

## Deferred to a follow-up plan (NOT in P0)
- CRN paired-delta + duplicate/mirror dealing → requires a deal-RNG-stream separation in `TexasHoldemPlayingHall` (RNG plumbing only). Worth doing before P3 A/B comparisons, but it is an engine-adjacent change and is out of measurement-only P0.
