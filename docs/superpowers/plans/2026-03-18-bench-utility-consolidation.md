# Bench Utility Consolidation (Plan 2 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate ~200 lines of copy-pasted bench helpers (`card`, `hole`, `BatchData`, `loadBatch`) across 15 files by extracting a shared `BenchSupport` object.

**Architecture:** Single new file `BenchSupport.scala` in `bench/` provides `card()`, `hole()`, `BatchData` case class, and `loadBatch()`. Each consumer removes its private copy and imports from `BenchSupport`. AutoTuner keeps its own batch-loading logic (uses `tuneEntryLimit`) but shares the `BatchData` type.

**Tech Stack:** Scala 3.8.1, munit 1.2.2, SBT

---

## Scope

### This Plan

- **11x `card(token: String): Card`** → `BenchSupport.card`
- **10x `hole(a: String, b: String): HoleCards`** → `BenchSupport.hole`
- **4x `BatchData(packedKeys, keyMaterial)` case class** → `BenchSupport.BatchData`
- **3x `loadBatch(table, maxMatchups): BatchData`** → `BenchSupport.loadBatch`
- GPU property constants: not found in codebase — dropped from scope.

**Behavioral notes:**
- The 11 `card()` copies use three different error messages (`"bad card"`, `"invalid card"`, `"invalid card token"`). The shared version normalizes to `"bad card: $token"`. This is acceptable — error messages in bench helpers are not part of any API contract.
- Two test files (`OperationalRegressionSuiteTest`, `PokerPipelineTest`) use `fail(...)` (munit) instead of `throw IllegalArgumentException`. Switching to `BenchSupport.card` changes the exception type on parse failure. This is acceptable — these test helpers parse hardcoded valid tokens and should never fail.
- `loadBatch` tests call `HeadsUpEquityTable.selectFullBatch` / `HeadsUpEquityCanonicalTable.selectCanonicalBatch` which require computed equity tables in memory. If these fail in the test runner, replace the `loadBatch` integration tests with a unit test for `BatchData.size` only and test `loadBatch` via `sbt compile` + the existing bench programs.

### Not in Scope

- Plan 3: PlayingHall GTO extraction
- Stretch goals from Plan 1 (renderAction in 7 more files)

## File Map

### New Files

| File | Responsibility |
|---|---|
| `src/main/scala/sicfun/holdem/bench/BenchSupport.scala` | `card`, `hole`, `BatchData`, `loadBatch` |
| `src/test/scala/sicfun/holdem/bench/BenchSupportTest.scala` | Unit tests for all shared helpers |

### Modified Files — card/hole removal (11 files)

| File | Remove |
|---|---|
| `src/main/scala/sicfun/holdem/bench/HoldemBayesBenchmark.scala` | `card`, `hole` |
| `src/main/scala/sicfun/holdem/bench/HoldemCfrFixedBenchmark.scala` | `card`, `hole` |
| `src/main/scala/sicfun/holdem/bench/HoldemCfrFixedParityProbe.scala` | `card`, `hole` |
| `src/main/scala/sicfun/holdem/bench/HoldemCfrNativeFixedBenchmark.scala` | `card`, `hole` |
| `src/main/scala/sicfun/holdem/bench/HoldemDdreParityBenchmark.scala` | `card`, `hole` |
| `src/main/scala/sicfun/holdem/bench/HoldemPostflopGpuAutoTuner.scala` | `card`, `hole` |
| `src/main/scala/sicfun/holdem/bench/HoldemPostflopNativeBenchmark.scala` | `card`, `hole` |
| `src/main/scala/sicfun/holdem/bench/OpponentMemoryBatchingBenchmark.scala` | `card`, `hole` |
| `src/main/scala/sicfun/holdem/bench/ProbEquityBenchmark.scala` | `card`, `hole` |
| `src/test/scala/sicfun/holdem/bench/OperationalRegressionSuiteTest.scala` | `card` |
| `src/test/scala/sicfun/holdem/bench/PokerPipelineTest.scala` | `card`, `hole` |

### Modified Files — BatchData/loadBatch removal (4 files)

| File | Remove |
|---|---|
| `src/main/scala/sicfun/holdem/bench/HeadsUpBackendAutoTuner.scala` | `BatchData` only (keeps own loader) |
| `src/main/scala/sicfun/holdem/bench/HeadsUpBackendComparison.scala` | `BatchData`, `loadBatch` |
| `src/main/scala/sicfun/holdem/bench/HeadsUpGpuPocGate.scala` | `BatchData`, `loadBatch` |
| `src/main/scala/sicfun/holdem/bench/HeadsUpGpuSmokeGate.scala` | `BatchData`, `loadBatch` |

### Dependency Order

```
Task 1 (card/hole + BenchSupport) ──→ Task 3 (refactor 11 card/hole consumers) ──┐
                                                                                   ├→ Task 5 (Verify)
Task 2 (BatchData + loadBatch)    ──→ Task 4 (refactor 4 BatchData consumers)   ──┘
```

Tasks 1 & 2 are sequential (both write to BenchSupport.scala).
Tasks 3 & 4 are independent (can parallelize).

---

## Task 1: Extract card() and hole() into BenchSupport

**Files:**
- Create: `src/main/scala/sicfun/holdem/bench/BenchSupport.scala`
- Create: `src/test/scala/sicfun/holdem/bench/BenchSupportTest.scala`

- [ ] **Step 1: Write failing tests for card and hole**

```scala
package sicfun.holdem.bench

import munit.FunSuite
import sicfun.core.Card
import sicfun.holdem.types.HoleCards

class BenchSupportTest extends FunSuite:

  test("card parses valid token"):
    val c = BenchSupport.card("Ah")
    assertEquals(c, Card.parse("Ah").get)

  test("card throws on invalid token"):
    intercept[IllegalArgumentException]:
      BenchSupport.card("Xx")

  test("hole builds HoleCards from two tokens"):
    val h = BenchSupport.hole("Ah", "Kd")
    assertEquals(h, HoleCards.from(Vector(Card.parse("Ah").get, Card.parse("Kd").get)))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `sbt "testOnly sicfun.holdem.bench.BenchSupportTest"`
Expected: FAIL — `BenchSupport` does not exist.

- [ ] **Step 3: Implement BenchSupport with card and hole**

```scala
package sicfun.holdem.bench

import sicfun.core.Card
import sicfun.holdem.types.HoleCards
import sicfun.holdem.equity.{HeadsUpEquityTable, HeadsUpEquityCanonicalTable}

/** Shared test/bench helpers for card parsing, hole card construction, and batch loading.
  *
  * Consolidates private helpers duplicated across 15 bench and test files.
  */
private[holdem] object BenchSupport:

  def card(token: String): Card =
    Card.parse(token).getOrElse(throw new IllegalArgumentException(s"bad card: $token"))

  def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `sbt "testOnly sicfun.holdem.bench.BenchSupportTest"`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/bench/BenchSupport.scala \
  src/test/scala/sicfun/holdem/bench/BenchSupportTest.scala
git commit -m "refactor: extract BenchSupport with shared card() and hole() helpers"
```

---

## Task 2: Add BatchData and loadBatch to BenchSupport

**Files:**
- Modify: `src/main/scala/sicfun/holdem/bench/BenchSupport.scala`
- Modify: `src/test/scala/sicfun/holdem/bench/BenchSupportTest.scala`

**Depends on:** Task 1

- [ ] **Step 1: Write failing tests for BatchData and loadBatch**

Append to `BenchSupportTest.scala`:

```scala
  test("BatchData.size returns packedKeys length"):
    val bd = BenchSupport.BatchData(Array(1L, 2L, 3L), Array(4L, 5L, 6L))
    assertEquals(bd.size, 3)

  test("loadBatch full returns non-empty batch"):
    val bd = BenchSupport.loadBatch("full", maxMatchups = 8L)
    assert(bd.size > 0)
    assert(bd.size <= 8)

  test("loadBatch canonical returns non-empty batch"):
    val bd = BenchSupport.loadBatch("canonical", maxMatchups = 8L)
    assert(bd.size > 0)
    assert(bd.size <= 8)

  test("loadBatch rejects unknown table"):
    intercept[IllegalArgumentException]:
      BenchSupport.loadBatch("bogus", maxMatchups = 8L)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `sbt "testOnly sicfun.holdem.bench.BenchSupportTest"`
Expected: FAIL — `BatchData` and `loadBatch` do not exist.

- [ ] **Step 3: Add BatchData and loadBatch to BenchSupport**

Append to `BenchSupport.scala` (inside the object):

```scala
  final case class BatchData(
      packedKeys: Array[Long],
      keyMaterial: Array[Long]
  ):
    def size: Int = packedKeys.length

  def loadBatch(table: String, maxMatchups: Long): BatchData =
    table.trim.toLowerCase match
      case "full" =>
        val batch = HeadsUpEquityTable.selectFullBatch(maxMatchups)
        BatchData(batch.packedKeys, batch.keyMaterial)
      case "canonical" =>
        val batch = HeadsUpEquityCanonicalTable.selectCanonicalBatch(maxMatchups)
        BatchData(batch.packedKeys, batch.keyMaterial)
      case other =>
        throw new IllegalArgumentException(s"unknown table '$other' (expected canonical or full)")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `sbt "testOnly sicfun.holdem.bench.BenchSupportTest"`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/bench/BenchSupport.scala \
  src/test/scala/sicfun/holdem/bench/BenchSupportTest.scala
git commit -m "refactor: add BatchData and loadBatch to BenchSupport"
```

---

## Task 3: Refactor card/hole Consumers (11 files)

**Files:** See "Modified Files — card/hole removal" table above.

**Depends on:** Task 1

For each file: delete the private `card` and/or `hole` method definitions, add `import sicfun.holdem.bench.BenchSupport.{card, hole}` (or just `card` for files that only have `card`). Call sites remain unchanged because the function signatures are identical.

**Note on nested defs:** Three files define `card`/`hole` as method-local defs (not object-level private methods): `HoldemBayesBenchmark.buildContext()`, `HoldemPostflopGpuAutoTuner.buildSpot()`, `HoldemPostflopNativeBenchmark.benchmarkSpot()`. For these, add the import at file level and delete the local `def`. Call sites within the method will resolve to the imported name.

**Note on ProbEquityBenchmark.board():** This file also has a `board(tokens: String*)` method that calls `card(...)`. After replacing the local `card` def with the import, `board()` will resolve `card` via the import — no changes needed to `board()`.

- [ ] **Step 1: Refactor ProbEquityBenchmark** (simplest file, validates the pattern)

In `ProbEquityBenchmark.scala`:
- Add import: `import sicfun.holdem.bench.BenchSupport.{card, hole}` (or use qualified calls if the file already imports `card` from elsewhere)
- Delete the private `card` method (line ~13-14)
- Delete the private `hole` method (line ~16-17)

- [ ] **Step 2: Compile check**

Run: `sbt compile`
Expected: PASS — no errors.

- [ ] **Step 3: Refactor remaining 10 files**

Apply the same pattern to each file. For each:
1. Add `import sicfun.holdem.bench.BenchSupport.{card, hole}` (or just `card` for OperationalRegressionSuiteTest)
2. Delete the private `card` def
3. Delete the private `hole` def (where present)

Files:
- `HoldemCfrFixedBenchmark.scala` (lines ~34-38)
- `HoldemCfrFixedParityProbe.scala` (lines ~40-44)
- `HoldemCfrNativeFixedBenchmark.scala` (lines ~37-41)
- `HoldemDdreParityBenchmark.scala` (lines ~520-524)
- `HoldemBayesBenchmark.scala` (lines ~218-222, method-local defs in `buildContext()`)
- `HoldemPostflopGpuAutoTuner.scala` (lines ~262-266, method-local defs in `buildSpot()`)
- `HoldemPostflopNativeBenchmark.scala` (lines ~277-281, method-local defs in `benchmarkSpot()`)
- `OpponentMemoryBatchingBenchmark.scala` (lines ~347-351)
- `OperationalRegressionSuiteTest.scala` (line ~15, `card` only)
- `PokerPipelineTest.scala` (lines ~10-14)

- [ ] **Step 4: Compile check**

Run: `sbt compile`
Expected: PASS — no errors.

- [ ] **Step 5: Run full test suite**

Run: `sbt test`
Expected: ALL PASS (minus the 2 pre-existing calibration failures)

- [ ] **Step 6: Commit**

```bash
git add src/main/scala/sicfun/holdem/bench/HoldemBayesBenchmark.scala \
  src/main/scala/sicfun/holdem/bench/HoldemCfrFixedBenchmark.scala \
  src/main/scala/sicfun/holdem/bench/HoldemCfrFixedParityProbe.scala \
  src/main/scala/sicfun/holdem/bench/HoldemCfrNativeFixedBenchmark.scala \
  src/main/scala/sicfun/holdem/bench/HoldemDdreParityBenchmark.scala \
  src/main/scala/sicfun/holdem/bench/HoldemPostflopGpuAutoTuner.scala \
  src/main/scala/sicfun/holdem/bench/HoldemPostflopNativeBenchmark.scala \
  src/main/scala/sicfun/holdem/bench/OpponentMemoryBatchingBenchmark.scala \
  src/main/scala/sicfun/holdem/bench/ProbEquityBenchmark.scala \
  src/test/scala/sicfun/holdem/bench/OperationalRegressionSuiteTest.scala \
  src/test/scala/sicfun/holdem/bench/PokerPipelineTest.scala
git commit -m "refactor: replace 11 private card/hole copies with BenchSupport imports"
```

---

## Task 4: Refactor BatchData/loadBatch Consumers (4 files)

**Files:** See "Modified Files — BatchData/loadBatch removal" table above.

**Depends on:** Task 2

- [ ] **Step 1: Refactor HeadsUpGpuSmokeGate** (has both BatchData + loadBatch)

In `HeadsUpGpuSmokeGate.scala`:
- Add import: `import sicfun.holdem.bench.BenchSupport.{BatchData, loadBatch}`
- Delete private `BatchData` case class (lines ~38-42)
- Delete private `loadBatch` method (lines ~115-125)
- Call sites and method signatures that use `BatchData`/`loadBatch` remain unchanged.

- [ ] **Step 2: Compile check**

Run: `sbt compile`
Expected: PASS

- [ ] **Step 3: Refactor remaining 3 files**

Apply the same pattern:

**HeadsUpGpuPocGate.scala:**
- Import `BenchSupport.{BatchData, loadBatch}`
- Delete private `BatchData` (lines ~52-56) and `loadBatch` (lines ~138-148)

**HeadsUpBackendComparison.scala:**
- Import `BenchSupport.{BatchData, loadBatch}`
- Delete private `BatchData` (lines ~74-78) and `loadBatch` (lines ~232-242)

**HeadsUpBackendAutoTuner.scala:**
- Import `BenchSupport.BatchData` only (no `loadBatch` — its loader has extra `tuneEntryLimit` logic)
- Delete private `BatchData` (lines ~30-34)
- Update the `loadBatch`-equivalent code to construct `BenchSupport.BatchData(...)` instead of the now-deleted local class. The construction sites (lines ~374, ~379) already use `BatchData(batch.packedKeys, batch.keyMaterial)` which will resolve to the imported type.

- [ ] **Step 4: Compile check**

Run: `sbt compile`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `sbt test`
Expected: ALL PASS (minus the 2 pre-existing calibration failures)

- [ ] **Step 6: Commit**

```bash
git add src/main/scala/sicfun/holdem/bench/HeadsUpBackendAutoTuner.scala \
  src/main/scala/sicfun/holdem/bench/HeadsUpBackendComparison.scala \
  src/main/scala/sicfun/holdem/bench/HeadsUpGpuPocGate.scala \
  src/main/scala/sicfun/holdem/bench/HeadsUpGpuSmokeGate.scala
git commit -m "refactor: replace 4 private BatchData/loadBatch copies with BenchSupport imports"
```

---

## Task 5: Full Verification and Cleanup

**Depends on:** Tasks 3, 4

- [ ] **Step 1: Run full test suite**

Run: `sbt test`
Expected: ALL PASS (minus the 2 pre-existing calibration failures in CfrGtoCalibrationTest, GtoBaselineFalsePositiveTest)

- [ ] **Step 2: Compile clean check**

Run: `sbt compile 2>&1 | grep -i "warn\|error"`
Expected: No new warnings or errors.

- [ ] **Step 3: Verify line count reduction**

Expected:
- 11 files lose ~4 lines each (card + hole defs) = ~44 lines
- 3 files lose ~20 lines each (BatchData + loadBatch defs) = ~60 lines
- 1 file loses ~6 lines (BatchData only) = ~6 lines
- New shared code: ~40 lines (BenchSupport.scala + tests)
- **Net reduction: ~70 lines, ~200 lines of duplication eliminated**

- [ ] **Step 4: Final commit (if any cleanup needed)**

```bash
git add -u
git commit -m "chore: final cleanup after bench utility consolidation"
```
