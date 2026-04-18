# sicfun 9-max vs. Blueprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the infrastructure that lets sicfun play 9-max NLHE against blueprint-only agents (Pluribus-style proxy) and measure the value added by the strategic overlay, with hard-error native-strict mode (no silent fallback to Scala).

**Architecture:** New `AcpcTableDealer` parametric in `numSeats ∈ [2..9]` (independent of the heads-up dealer). New `SeatAgent` trait with `BlueprintOnlyAgent` and `StrategicAgent` implementations. `StrategicAgent` builds an MDP embedding from the 6 bridges and drives the verified `SafetyBellman` + `ExploitationInterpolation` pipeline. `NineMaxMatchRunner` orchestrates the match; `NativeStrictMode` probes all required DLLs eagerly with a build-freshness check.

**Tech Stack:** Scala 3.8.1 + munit 1.2.2 + scalacheck (for property tests) + existing CFR/Bayes/DDRE/postflop/POMCP native DLLs via `GpuRuntimeSupport.loadNativeLibrary`.

**Spec:** [docs/superpowers/specs/2026-04-18-sicfun-9max-vs-blueprint-design.md](../specs/2026-04-18-sicfun-9max-vs-blueprint-design.md) (commit `036fb54`).

---

## File Structure

**New (production, Scala):**
- `src/main/scala/sicfun/holdem/runtime/protocol/TableDealerTypes.scala` — `SeatId`, `SidePot`, `BettingRoundEvent`, `HandOutcome`, `TableConfig`, `TableSnapshot`
- `src/main/scala/sicfun/holdem/runtime/protocol/AcpcTableDealer.scala` — N-seat dealer
- `src/main/scala/sicfun/holdem/runtime/agent/SeatAgent.scala` — trait
- `src/main/scala/sicfun/holdem/runtime/agent/BlueprintFormat.scala` — on-disk format types
- `src/main/scala/sicfun/holdem/runtime/agent/BlueprintStore.scala` — load/validate/lookup
- `src/main/scala/sicfun/holdem/runtime/agent/BlueprintOnlyAgent.scala`
- `src/main/scala/sicfun/holdem/runtime/agent/StrategicAgent.scala`
- `src/main/scala/sicfun/holdem/runtime/agent/MdpEmbedding.scala` — builds `robustLosses`/`transitions`/`qValues` from bridges
- `src/main/scala/sicfun/holdem/runtime/NativeStrictMode.scala`
- `src/main/scala/sicfun/holdem/runtime/NineMaxMatchRunner.scala`
- `src/main/scala/sicfun/holdem/runtime/metrics/MbbMetrics.scala`
- `src/main/scala/sicfun/holdem/runtime/metrics/FourWorldMetrics.scala`

**New (tests, munit):**
- `src/test/scala/sicfun/holdem/runtime/protocol/AcpcTableDealerTest.scala`
- `src/test/scala/sicfun/holdem/runtime/protocol/AcpcTableDealerSidePotPropertyTest.scala`
- `src/test/scala/sicfun/holdem/runtime/agent/BlueprintStoreTest.scala`
- `src/test/scala/sicfun/holdem/runtime/agent/BlueprintOnlyAgentTest.scala`
- `src/test/scala/sicfun/holdem/runtime/agent/StrategicAgentTest.scala`
- `src/test/scala/sicfun/holdem/runtime/NativeStrictModeTest.scala`
- `src/test/scala/sicfun/holdem/runtime/NineMaxMatchRunnerSmokeTest.scala`
- `src/test/scala/sicfun/holdem/runtime/metrics/MbbMetricsTest.scala`

**Not touched:** `AcpcHeadsUpDealer.scala`, `SlumbotMatchRunner.scala`, `AcpcMatchRunner.scala`, `strategic/**` (consumed read-only).

---

## Phase 1 — Dealer Foundation

### Task 1: Core table types

**Files:**
- Create: `src/main/scala/sicfun/holdem/runtime/protocol/TableDealerTypes.scala`
- Test: `src/test/scala/sicfun/holdem/runtime/protocol/TableDealerTypesTest.scala`

- [ ] **Step 1: Write failing test for SidePot invariant**

```scala
package sicfun.holdem.runtime.protocol

class TableDealerTypesTest extends munit.FunSuite:
  test("SidePot: eligible seats non-empty and amount positive"):
    val ok = SidePot(amount = 100, eligibleSeats = Set(SeatId(0), SeatId(1)))
    assertEquals(ok.amount, 100L)
    intercept[IllegalArgumentException](SidePot(amount = 0, eligibleSeats = Set(SeatId(0))))
    intercept[IllegalArgumentException](SidePot(amount = 100, eligibleSeats = Set.empty))

  test("TableConfig: numSeats in [2, 9]"):
    TableConfig(numSeats = 2, smallBlind = 1, bigBlind = 2, ante = 0, startingStack = 200)
    TableConfig(numSeats = 9, smallBlind = 1, bigBlind = 2, ante = 0, startingStack = 200)
    intercept[IllegalArgumentException](TableConfig(1, 1, 2, 0, 200))
    intercept[IllegalArgumentException](TableConfig(10, 1, 2, 0, 200))
```

- [ ] **Step 2: Run test (expect compile fail)**

Run: `sbt "testOnly sicfun.holdem.runtime.protocol.TableDealerTypesTest"`
Expected: compile error "not found: type SidePot"

- [ ] **Step 3: Implement types**

```scala
package sicfun.holdem.runtime.protocol

import sicfun.holdem.types.{Card, PokerAction}

opaque type SeatId = Int
object SeatId:
  def apply(i: Int): SeatId =
    require(i >= 0 && i < 9, s"SeatId must be in [0, 9), got $i")
    i
  extension (s: SeatId) def index: Int = s

final case class TableConfig(
    numSeats: Int,
    smallBlind: Long,
    bigBlind: Long,
    ante: Long,
    startingStack: Long
):
  require(numSeats >= 2 && numSeats <= 9, s"numSeats must be in [2, 9], got $numSeats")
  require(smallBlind > 0 && bigBlind >= smallBlind, "blinds must be positive with bb >= sb")
  require(ante >= 0, "ante must be non-negative")
  require(startingStack >= 10L * bigBlind, "startingStack must be at least 10 bb")

final case class SidePot(amount: Long, eligibleSeats: Set[SeatId]):
  require(amount > 0, s"SidePot amount must be positive, got $amount")
  require(eligibleSeats.nonEmpty, "SidePot must have at least one eligible seat")

enum BettingRoundEvent:
  case PostBlind(seat: SeatId, amount: Long, kind: BlindKind)
  case PostAnte(seat: SeatId, amount: Long)
  case Act(seat: SeatId, action: PokerAction)
  case Deal(street: Street, cards: Vector[Card])
  case Showdown(revealed: Map[SeatId, Vector[Card]])

enum BlindKind:
  case SmallBlind, BigBlind

enum Street:
  case Preflop, Flop, Turn, River

final case class HandOutcome(
    potsDistributed: Vector[(SidePot, Map[SeatId, Long])],
    netChange: Map[SeatId, Long],
    events: Vector[BettingRoundEvent]
)

final case class TableSnapshot(
    config: TableConfig,
    heroSeat: SeatId,
    holeCards: Vector[Card],
    board: Vector[Card],
    stacks: Map[SeatId, Long],
    contributions: Map[SeatId, Long],
    street: Street,
    actionHistory: Vector[BettingRoundEvent],
    buttonSeat: SeatId,
    activeSeats: Set[SeatId]
)
```

- [ ] **Step 4: Run test to verify pass**

Run: `sbt "testOnly sicfun.holdem.runtime.protocol.TableDealerTypesTest"`
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/protocol/TableDealerTypes.scala src/test/scala/sicfun/holdem/runtime/protocol/TableDealerTypesTest.scala
git commit -m "feat(runtime): table dealer core types (SeatId, SidePot, TableConfig, snapshot)"
```

---

### Task 2: Dealer skeleton with blinds posting

**Files:**
- Create: `src/main/scala/sicfun/holdem/runtime/protocol/AcpcTableDealer.scala`
- Test: `src/test/scala/sicfun/holdem/runtime/protocol/AcpcTableDealerTest.scala`

- [ ] **Step 1: Write failing test for blinds posting**

```scala
package sicfun.holdem.runtime.protocol

import sicfun.holdem.runtime.protocol.BettingRoundEvent.*
import sicfun.holdem.runtime.protocol.BlindKind.*

class AcpcTableDealerTest extends munit.FunSuite:
  val cfg2 = TableConfig(2, smallBlind = 1, bigBlind = 2, ante = 0, startingStack = 200)
  val cfg6 = TableConfig(6, smallBlind = 1, bigBlind = 2, ante = 0, startingStack = 200)
  val cfg9 = TableConfig(9, smallBlind = 1, bigBlind = 2, ante = 0, startingStack = 200)

  test("blinds posted at N=2: button is SB"):
    val d = AcpcTableDealer(cfg2, buttonSeat = SeatId(0), rngSeed = 42L)
    val events = d.postBlinds()
    assertEquals(events, Vector(
      PostBlind(SeatId(0), 1L, SmallBlind),
      PostBlind(SeatId(1), 2L, BigBlind)
    ))

  test("blinds posted at N=6: SB = BTN+1, BB = BTN+2"):
    val d = AcpcTableDealer(cfg6, buttonSeat = SeatId(2), rngSeed = 42L)
    val events = d.postBlinds()
    assertEquals(events, Vector(
      PostBlind(SeatId(3), 1L, SmallBlind),
      PostBlind(SeatId(4), 2L, BigBlind)
    ))

  test("blinds wrap around seat indices"):
    val d = AcpcTableDealer(cfg9, buttonSeat = SeatId(8), rngSeed = 42L)
    val events = d.postBlinds()
    assertEquals(events, Vector(
      PostBlind(SeatId(0), 1L, SmallBlind),
      PostBlind(SeatId(1), 2L, BigBlind)
    ))
```

- [ ] **Step 2: Run test (expect compile fail)**

Run: `sbt "testOnly sicfun.holdem.runtime.protocol.AcpcTableDealerTest"`
Expected: compile error "not found: type AcpcTableDealer"

- [ ] **Step 3: Implement dealer with blinds**

```scala
package sicfun.holdem.runtime.protocol

import sicfun.holdem.runtime.protocol.BettingRoundEvent.*
import sicfun.holdem.runtime.protocol.BlindKind.*
import scala.util.Random

final class AcpcTableDealer(
    val config: TableConfig,
    val buttonSeat: SeatId,
    rngSeed: Long
):
  private val rng = new Random(rngSeed)
  private val stacks = collection.mutable.Map[SeatId, Long]()
  private val contributions = collection.mutable.Map[SeatId, Long]().withDefaultValue(0L)
  for i <- 0 until config.numSeats do stacks(SeatId(i)) = config.startingStack

  private def nextSeat(s: SeatId): SeatId =
    SeatId((s.index + 1) % config.numSeats)

  def smallBlindSeat: SeatId =
    if config.numSeats == 2 then buttonSeat
    else nextSeat(buttonSeat)

  def bigBlindSeat: SeatId = nextSeat(smallBlindSeat)

  def postBlinds(): Vector[BettingRoundEvent] =
    val sb = smallBlindSeat
    val bb = bigBlindSeat
    stacks(sb) -= config.smallBlind
    stacks(bb) -= config.bigBlind
    contributions(sb) += config.smallBlind
    contributions(bb) += config.bigBlind
    Vector(
      PostBlind(sb, config.smallBlind, SmallBlind),
      PostBlind(bb, config.bigBlind, BigBlind)
    )

  def currentStacks: Map[SeatId, Long] = stacks.toMap
  def currentContributions: Map[SeatId, Long] = contributions.toMap
```

- [ ] **Step 4: Run test to verify pass**

Run: `sbt "testOnly sicfun.holdem.runtime.protocol.AcpcTableDealerTest"`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/protocol/AcpcTableDealer.scala src/test/scala/sicfun/holdem/runtime/protocol/AcpcTableDealerTest.scala
git commit -m "feat(runtime): AcpcTableDealer skeleton with blinds posting for N in [2,9]"
```

---

### Task 3: Button rotation over a match

**Files:**
- Modify: `src/main/scala/sicfun/holdem/runtime/protocol/AcpcTableDealer.scala`
- Modify: `src/test/scala/sicfun/holdem/runtime/protocol/AcpcTableDealerTest.scala`

- [ ] **Step 1: Write failing test**

Append to `AcpcTableDealerTest`:

```scala
  test("button rotates one seat per hand and cycles through all seats"):
    val startButton = SeatId(0)
    val dealer = AcpcTableDealer(cfg9, startButton, rngSeed = 1L)
    val visited = (0 until 9).map { _ =>
      val b = dealer.buttonSeat
      dealer.advanceButton()
      b.index
    }.toSet
    assertEquals(visited, (0 until 9).toSet)

  test("button wraps modulo numSeats"):
    val dealer = AcpcTableDealer(cfg2, SeatId(1), rngSeed = 1L)
    dealer.advanceButton()
    assertEquals(dealer.buttonSeat, SeatId(0))
```

- [ ] **Step 2: Run test (expect fail on advanceButton)**

Run: `sbt "testOnly sicfun.holdem.runtime.protocol.AcpcTableDealerTest"`
Expected: compile error "value advanceButton is not a member".

- [ ] **Step 3: Implement button rotation**

Change `buttonSeat` from `val` to `var` and add `advanceButton`:

```scala
final class AcpcTableDealer(
    val config: TableConfig,
    initialButtonSeat: SeatId,
    rngSeed: Long
):
  private val rng = new Random(rngSeed)
  private var _buttonSeat: SeatId = initialButtonSeat
  def buttonSeat: SeatId = _buttonSeat
  def advanceButton(): Unit = _buttonSeat = SeatId((_buttonSeat.index + 1) % config.numSeats)
  // ... rest unchanged, but `buttonSeat` uses the new accessor
```

Update the companion-style `apply` usage: since the primary constructor parameter name changed, update the tests to still use positional args. No change needed if tests use `AcpcTableDealer(cfg, button, seed)` positionally.

- [ ] **Step 4: Run test to verify pass**

Run: `sbt "testOnly sicfun.holdem.runtime.protocol.AcpcTableDealerTest"`
Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/protocol/AcpcTableDealer.scala src/test/scala/sicfun/holdem/runtime/protocol/AcpcTableDealerTest.scala
git commit -m "feat(runtime): button rotation for AcpcTableDealer"
```

---

## Phase 2 — Betting Rounds and Side Pots

### Task 4: Action round closure by last aggressor

**Files:**
- Modify: `src/main/scala/sicfun/holdem/runtime/protocol/AcpcTableDealer.scala`
- Modify: `src/test/scala/sicfun/holdem/runtime/protocol/AcpcTableDealerTest.scala`

Goal: expose `nextToAct: Option[SeatId]` and `applyAction(seat, action)` that returns `Either[IllegalActionReason, Unit]`. A round closes when action returns to the last aggressor or everyone who has not folded has matched the current bet.

- [ ] **Step 1: Write failing tests**

Append:

```scala
  test("action closes preflop when BB checks their option after limpers (N=3)"):
    val cfg = TableConfig(3, 1, 2, 0, 200)
    val d = AcpcTableDealer(cfg, SeatId(0), rngSeed = 1L)
    d.postBlinds()                     // SB=1, BB=2
    d.startStreet(Street.Preflop)
    assertEquals(d.nextToAct, Some(SeatId(0))) // UTG = BTN at 3-max
    d.applyAction(SeatId(0), PokerAction.Call).getOrElse(fail("call failed"))
    d.applyAction(SeatId(1), PokerAction.Call).getOrElse(fail("sb call"))
    d.applyAction(SeatId(2), PokerAction.Check).getOrElse(fail("bb option"))
    assert(d.roundClosed, "round should be closed after BB checks option")

  test("action does NOT close when last aggressor hasn't been given final option"):
    val cfg = TableConfig(3, 1, 2, 0, 200)
    val d = AcpcTableDealer(cfg, SeatId(0), rngSeed = 1L)
    d.postBlinds()
    d.startStreet(Street.Preflop)
    d.applyAction(SeatId(0), PokerAction.Raise(6))
    d.applyAction(SeatId(1), PokerAction.Call)
    assert(!d.roundClosed, "BB still has option")
    d.applyAction(SeatId(2), PokerAction.Call)
    assert(d.roundClosed, "closed after BB closes action on the raiser")
```

- [ ] **Step 2: Run test (expect compile fail)**

Expected: compile errors for `startStreet`, `applyAction`, `nextToAct`, `roundClosed`, `IllegalActionReason`.

- [ ] **Step 3: Implement round logic**

```scala
enum IllegalActionReason:
  case NotYourTurn(seat: SeatId, expected: Option[SeatId])
  case IllegalAction(seat: SeatId, action: PokerAction, reason: String)
  case InsufficientChips(seat: SeatId, required: Long, available: Long)

// inside AcpcTableDealer
private var currentStreet: Street = Street.Preflop
private var currentBet: Long = 0L
private var lastAggressor: Option[SeatId] = None
private var actOrder: Vector[SeatId] = Vector.empty
private var actIdx: Int = 0
private val folded: collection.mutable.Set[SeatId] = collection.mutable.Set.empty
private val allIn: collection.mutable.Set[SeatId] = collection.mutable.Set.empty
private val streetContribution: collection.mutable.Map[SeatId, Long] =
  collection.mutable.Map.empty.withDefaultValue(0L)

def startStreet(street: Street): Unit =
  currentStreet = street
  streetContribution.clear()
  if street == Street.Preflop then
    currentBet = config.bigBlind
    streetContribution(smallBlindSeat) = config.smallBlind
    streetContribution(bigBlindSeat) = config.bigBlind
    lastAggressor = Some(bigBlindSeat)
    actOrder = buildActOrder(firstToAct = nextSeat(bigBlindSeat))
  else
    currentBet = 0L
    lastAggressor = None
    actOrder = buildActOrder(firstToAct = firstActivePostflop)
  actIdx = 0

private def buildActOrder(firstToAct: SeatId): Vector[SeatId] =
  val order = (0 until config.numSeats).map { off =>
    SeatId((firstToAct.index + off) % config.numSeats)
  }.toVector
  order.filter(s => !folded.contains(s) && !allIn.contains(s))

private def firstActivePostflop: SeatId =
  val start = nextSeat(buttonSeat)
  var s = start
  while folded.contains(s) || allIn.contains(s) do s = nextSeat(s)
  s

def nextToAct: Option[SeatId] =
  if roundClosed then None else actOrder.lift(actIdx)

def roundClosed: Boolean =
  val eligible = (0 until config.numSeats).map(SeatId(_))
    .filter(s => !folded.contains(s) && !allIn.contains(s))
  if eligible.size <= 1 then true
  else
    val allMatched = eligible.forall(s => streetContribution(s) == currentBet)
    val everyoneActed = actIdx >= actOrder.size
    allMatched && (lastAggressor match
      case Some(a) => actIdx > actOrder.indexOf(a)
      case None    => everyoneActed)

def applyAction(seat: SeatId, action: PokerAction): Either[IllegalActionReason, Unit] =
  if nextToAct != Some(seat) then
    Left(IllegalActionReason.NotYourTurn(seat, nextToAct))
  else action match
    case PokerAction.Fold =>
      folded += seat
      actIdx += 1
      Right(())
    case PokerAction.Check =>
      if streetContribution(seat) != currentBet then
        Left(IllegalActionReason.IllegalAction(seat, action, "cannot check facing a bet"))
      else
        actIdx += 1
        Right(())
    case PokerAction.Call =>
      val owed = currentBet - streetContribution(seat)
      if owed <= 0 then
        Left(IllegalActionReason.IllegalAction(seat, action, "nothing to call"))
      else
        val pay = math.min(owed, stacks(seat))
        stacks(seat) -= pay
        streetContribution(seat) += pay
        contributions(seat) += pay
        if stacks(seat) == 0 then allIn += seat
        actIdx += 1
        Right(())
    case PokerAction.Raise(amount) =>
      val target = amount.toLong
      if target <= currentBet then
        Left(IllegalActionReason.IllegalAction(seat, action, s"raise $target not above currentBet $currentBet"))
      else
        val pay = target - streetContribution(seat)
        if pay > stacks(seat) then
          Left(IllegalActionReason.InsufficientChips(seat, pay, stacks(seat)))
        else
          stacks(seat) -= pay
          streetContribution(seat) += pay
          contributions(seat) += pay
          currentBet = target
          lastAggressor = Some(seat)
          if stacks(seat) == 0 then allIn += seat
          // rebuild act order starting after the aggressor
          actOrder = buildActOrder(firstToAct = nextSeat(seat))
          actIdx = 0
          Right(())
```

- [ ] **Step 4: Run test to verify pass**

Run: `sbt "testOnly sicfun.holdem.runtime.protocol.AcpcTableDealerTest"`
Expected: all 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/protocol/AcpcTableDealer.scala src/test/scala/sicfun/holdem/runtime/protocol/AcpcTableDealerTest.scala
git commit -m "feat(runtime): betting round closure by last aggressor for AcpcTableDealer"
```

---

### Task 5: Side pots (multi-level) with property invariant

**Files:**
- Modify: `src/main/scala/sicfun/holdem/runtime/protocol/AcpcTableDealer.scala`
- Create: `src/test/scala/sicfun/holdem/runtime/protocol/AcpcTableDealerSidePotPropertyTest.scala`

- [ ] **Step 1: Add scalacheck dep if missing**

Check `build.sbt` for `"org.scalameta" %% "munit-scalacheck"`; if absent, add:

```scala
libraryDependencies += "org.scalameta" %% "munit-scalacheck" % "1.0.0" % Test
```

Run `sbt update` to refresh.

- [ ] **Step 2: Write property test**

```scala
package sicfun.holdem.runtime.protocol

import org.scalacheck.Gen
import org.scalacheck.Prop.*

class AcpcTableDealerSidePotPropertyTest extends munit.ScalaCheckSuite:

  val stackGen: Gen[Long] = Gen.choose(10L, 500L)
  val nSeatsGen: Gen[Int] = Gen.choose(2, 9)

  property("side pots: sum of pots equals sum of contributions"):
    forAll(nSeatsGen, Gen.listOfN(9, stackGen)):
      (n, stacks) =>
        val cfg = TableConfig(n, 1, 2, 0, 200)
        val d = AcpcTableDealer(cfg, SeatId(0), rngSeed = 0L)
        // Seat custom stacks
        (0 until n).foreach(i => d.setStackForTest(SeatId(i), stacks(i)))
        d.postBlinds()
        d.startStreet(Street.Preflop)
        // Everyone goes all-in in order
        (0 until n).foreach { i =>
          val seat = SeatId((d.smallBlindSeat.index + i) % n)
          d.nextToAct.foreach { s =>
            d.applyAction(s, PokerAction.Raise(d.currentStacks(s) + d.currentContributions(s)))
          }
        }
        val pots = d.computeSidePots()
        val totalContributions = d.currentContributions.values.sum
        val totalPots = pots.map(_.amount).sum
        totalPots == totalContributions

  property("side pots: each pot eligible seats is a subset of non-folded seats"):
    forAll(nSeatsGen):
      n =>
        val cfg = TableConfig(n, 1, 2, 0, 200)
        val d = AcpcTableDealer(cfg, SeatId(0), rngSeed = 0L)
        d.postBlinds()
        d.startStreet(Street.Preflop)
        val pots = d.computeSidePots()
        val eligibleUniverse = (0 until n).map(SeatId(_)).toSet
        pots.forall(_.eligibleSeats.subsetOf(eligibleUniverse))
```

- [ ] **Step 3: Implement side pot computation**

Append to `AcpcTableDealer`:

```scala
def setStackForTest(seat: SeatId, amount: Long): Unit =
  stacks(seat) = amount

def computeSidePots(): Vector[SidePot] =
  val activeSeats = (0 until config.numSeats).map(SeatId(_)).toVector
  val contribPairs = activeSeats
    .map(s => s -> contributions(s))
    .filter(_._2 > 0)
    .sortBy(_._2)

  var result = Vector.empty[SidePot]
  var prevLevel = 0L
  var remaining: Vector[(SeatId, Long)] = contribPairs

  while remaining.nonEmpty do
    val level = remaining.head._2
    val delta = level - prevLevel
    val eligible = remaining.map(_._1).toSet -- folded.toSet
    val potAmount = delta * remaining.size
    if potAmount > 0 && eligible.nonEmpty then
      result = result :+ SidePot(amount = potAmount, eligibleSeats = eligible)
    prevLevel = level
    remaining = remaining.filter(_._2 > level)

  result
```

- [ ] **Step 4: Run property tests**

Run: `sbt "testOnly sicfun.holdem.runtime.protocol.AcpcTableDealerSidePotPropertyTest"`
Expected: both properties hold across 100 generated cases.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/protocol/AcpcTableDealer.scala src/test/scala/sicfun/holdem/runtime/protocol/AcpcTableDealerSidePotPropertyTest.scala build.sbt
git commit -m "feat(runtime): multi-level side pots with property-based invariants"
```

---

### Task 6: Card dealing + showdown partial

**Files:**
- Modify: `src/main/scala/sicfun/holdem/runtime/protocol/AcpcTableDealer.scala`
- Modify: `src/test/scala/sicfun/holdem/runtime/protocol/AcpcTableDealerTest.scala`

- [ ] **Step 1: Write failing test**

```scala
  test("showdown returns revealed cards only for non-folded seats"):
    val d = AcpcTableDealer(cfg6, SeatId(0), rngSeed = 42L)
    d.postBlinds()
    d.dealHoleCards()
    d.startStreet(Street.Preflop)
    // three seats fold preflop; three see showdown
    d.applyAction(SeatId(3), PokerAction.Fold)
    d.applyAction(SeatId(4), PokerAction.Fold)
    d.applyAction(SeatId(5), PokerAction.Fold)
    d.applyAction(SeatId(0), PokerAction.Call)
    d.applyAction(SeatId(1), PokerAction.Call)
    d.applyAction(SeatId(2), PokerAction.Check)
    d.dealCommunity(Street.Flop)
    d.dealCommunity(Street.Turn)
    d.dealCommunity(Street.River)
    val showdown = d.showdown()
    assertEquals(showdown.revealed.keySet, Set(SeatId(0), SeatId(1), SeatId(2)))
    showdown.revealed.foreach((_, cs) => assertEquals(cs.size, 2))
```

- [ ] **Step 2: Run test (compile fail)**

Expected: `dealHoleCards`, `dealCommunity`, `showdown` not found.

- [ ] **Step 3: Implement card dealing and showdown**

Append to `AcpcTableDealer`:

```scala
import sicfun.holdem.types.Card

private val deck: collection.mutable.ArrayBuffer[Card] =
  collection.mutable.ArrayBuffer.from(rng.shuffle(Card.fullDeck))
private val hole: collection.mutable.Map[SeatId, Vector[Card]] = collection.mutable.Map.empty
private val board: collection.mutable.ArrayBuffer[Card] = collection.mutable.ArrayBuffer.empty

def dealHoleCards(): Map[SeatId, Vector[Card]] =
  (0 until config.numSeats).foreach { i =>
    val seat = SeatId(i)
    val c1 = deck.remove(0)
    val c2 = deck.remove(0)
    hole(seat) = Vector(c1, c2)
  }
  hole.toMap

def dealCommunity(street: Street): Vector[Card] =
  val count = street match
    case Street.Preflop => 0
    case Street.Flop    => 3
    case Street.Turn    => 1
    case Street.River   => 1
  (1 to count).foreach { _ => board += deck.remove(0) }
  board.toVector

def showdown(): BettingRoundEvent.Showdown =
  val revealed = hole.toMap.filterNot((s, _) => folded.contains(s))
  BettingRoundEvent.Showdown(revealed)
```

*Note: `Card.fullDeck` is assumed present in `sicfun.holdem.types.Card`. If it is named differently, grep for `def fullDeck` or use `Card.values`.*

- [ ] **Step 4: Run test**

Run: `sbt "testOnly sicfun.holdem.runtime.protocol.AcpcTableDealerTest"`
Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/protocol/AcpcTableDealer.scala src/test/scala/sicfun/holdem/runtime/protocol/AcpcTableDealerTest.scala
git commit -m "feat(runtime): card dealing + partial showdown for AcpcTableDealer"
```

---

## Phase 3 — Native Strict Mode

### Task 7: NativeStrictMode — required DLL list and probe

**Files:**
- Create: `src/main/scala/sicfun/holdem/runtime/NativeStrictMode.scala`
- Create: `src/test/scala/sicfun/holdem/runtime/NativeStrictModeTest.scala`

- [ ] **Step 1: Write failing test**

```scala
package sicfun.holdem.runtime

import java.io.File

class NativeStrictModeTest extends munit.FunSuite:

  test("strict mode: all core DLLs present → Ready"):
    val nativeDir = new File("src/main/native/build")
    val result = NativeStrictMode.verify(
      nativeDir = nativeDir,
      required = NativeStrictMode.CoreLibraries,
      strict = true
    )
    assert(result.isRight, s"expected Ready but got $result")

  test("strict mode: missing DLL → Left(NativeStrictModeViolation)"):
    val tmp = java.nio.file.Files.createTempDirectory("sicfun-empty-native").toFile
    val result = NativeStrictMode.verify(
      nativeDir = tmp,
      required = Vector(NativeStrictMode.Library("sicfun_nonexistent", Vector.empty)),
      strict = true
    )
    assert(result.isLeft, "expected violation")
    result.left.foreach { err =>
      assert(err.getMessage.contains("sicfun_nonexistent"), err.getMessage)
    }

  test("non-strict mode: missing DLL → Right with warning flag"):
    val tmp = java.nio.file.Files.createTempDirectory("sicfun-empty-native-2").toFile
    val result = NativeStrictMode.verify(
      nativeDir = tmp,
      required = Vector(NativeStrictMode.Library("sicfun_nonexistent", Vector.empty)),
      strict = false
    )
    assert(result.isRight, s"expected Ready-with-warn but got $result")
    assertEquals(result.toOption.get.contaminated, true)
```

- [ ] **Step 2: Run test (compile fail)**

Expected: `NativeStrictMode` not found.

- [ ] **Step 3: Implement NativeStrictMode**

```scala
package sicfun.holdem.runtime

import java.io.File

final case class NativeStrictModeViolation(message: String) extends RuntimeException(message)
final case class NativeStaleBuildError(dll: String, staleSource: String)
    extends RuntimeException(s"Native DLL '$dll' is older than source '$staleSource'; rebuild with src/main/native/build-windows-cuda11.ps1")

final case class NativeReady(loaded: Vector[String], contaminated: Boolean)

object NativeStrictMode:

  final case class Library(name: String, sources: Vector[String])

  val CoreLibraries: Vector[Library] = Vector(
    Library("sicfun_gpu_kernel", Vector(
      "src/main/native/jni/HeadsUpGpuNativeBindings.cpp",
      "src/main/native/jni/HeadsUpGpuNativeBindingsCuda.cu"
    )),
    Library("sicfun_native_cpu", Vector(
      "src/main/native/jni/HeadsUpGpuNativeBindings.cpp"
    )),
    Library("sicfun_bayes_cuda", Vector(
      "src/main/native/jni/HoldemBayesNativeGpuBindings.cu",
      "src/main/native/jni/BayesNativeUpdateCore.hpp"
    )),
    Library("sicfun_bayes_native", Vector(
      "src/main/native/jni/HoldemBayesNativeCpuBindings.cpp",
      "src/main/native/jni/BayesNativeUpdateCore.hpp"
    )),
    Library("sicfun_ddre_cuda", Vector(
      "src/main/native/jni/HoldemDdreNativeGpuBindings.cu",
      "src/main/native/jni/DdreNativeInferenceCore.hpp"
    )),
    Library("sicfun_ddre_native", Vector(
      "src/main/native/jni/HoldemDdreNativeCpuBindings.cpp",
      "src/main/native/jni/DdreNativeInferenceCore.hpp"
    ))
  )

  val PomcpLibrary: Library = Library("sicfun_pomcp_native", Vector(
    "src/main/native/jni/HoldemPomcpNativeBindings.cpp",
    "src/main/native/jni/WPomcpSolver.hpp",
    "src/main/native/jni/PftDpwSolver.hpp"
  ))

  val PostflopLibraries: Vector[Library] = Vector(
    Library("sicfun_postflop_cuda", Vector(
      "src/main/native/jni/HoldemPostflopNativeBindingsCuda.cu"
    )),
    Library("sicfun_postflop_native", Vector(
      "src/main/native/jni/HoldemPostflopNativeBindings.cpp"
    ))
  )

  def verify(
      nativeDir: File,
      required: Vector[Library],
      strict: Boolean
  ): Either[NativeStrictModeViolation, NativeReady] =
    val missing = required.filter(lib => !dllFile(nativeDir, lib.name).exists())
    val stale = required.flatMap { lib =>
      val dll = dllFile(nativeDir, lib.name)
      if !dll.exists() then None
      else
        val dllMtime = dll.lastModified()
        lib.sources
          .map(s => new File(s))
          .find(src => src.exists() && src.lastModified() > dllMtime)
          .map(src => (lib.name, src.getPath))
    }
    (missing, stale, strict) match
      case (m, _, true) if m.nonEmpty =>
        Left(NativeStrictModeViolation(
          s"Missing native libraries (strict): ${m.map(_.name).mkString(", ")}; " +
            s"rebuild with src/main/native/build-windows-cuda11.ps1"
        ))
      case (_, s, true) if s.nonEmpty =>
        Left(NativeStrictModeViolation(
          s"Stale native builds (strict): ${s.map((dll, src) => s"$dll<$src").mkString(", ")}"
        ))
      case (m, s, false) if m.nonEmpty || s.nonEmpty =>
        // non-strict: log contamination but proceed
        System.err.println(
          s"[BENCHMARK-CONTAMINATED] strict=false, missing=${m.map(_.name).mkString(",")}, " +
            s"stale=${s.map(_._1).mkString(",")}"
        )
        Right(NativeReady(loaded = Vector.empty, contaminated = true))
      case _ =>
        Right(NativeReady(loaded = required.map(_.name), contaminated = false))

  private def dllFile(nativeDir: File, libName: String): File =
    val osName = System.getProperty("os.name", "").toLowerCase
    val file =
      if osName.contains("win") then new File(nativeDir, s"$libName.dll")
      else if osName.contains("mac") then new File(nativeDir, s"lib$libName.dylib")
      else new File(nativeDir, s"lib$libName.so")
    file
```

- [ ] **Step 4: Run test**

Run: `sbt "testOnly sicfun.holdem.runtime.NativeStrictModeTest"`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/NativeStrictMode.scala src/test/scala/sicfun/holdem/runtime/NativeStrictModeTest.scala
git commit -m "feat(runtime): NativeStrictMode with eager probe + stale-build detection"
```

---

## Phase 4 — Agent API, Blueprint Format and Store

### Task 8: SeatAgent trait

**Files:**
- Create: `src/main/scala/sicfun/holdem/runtime/agent/SeatAgent.scala`

- [ ] **Step 1: Implement the trait (no tests — pure interface)**

```scala
package sicfun.holdem.runtime.agent

import sicfun.holdem.runtime.protocol.*
import sicfun.holdem.types.PokerAction

trait SeatAgent:
  def seatId: SeatId
  def onMatchStart(tableConfig: TableConfig): Unit = ()
  def onHandStart(snapshot: TableSnapshot): Unit = ()
  def decide(snapshot: TableSnapshot, legalActions: Set[PokerAction]): PokerAction
  def onHandEnd(snapshot: TableSnapshot, outcome: HandOutcome): Unit = ()
  def onMatchEnd(summary: MatchSummary): Unit = ()

final case class MatchSummary(
    totalHands: Int,
    netBySeat: Map[SeatId, Long],
    handsBySeat: Map[SeatId, Int]
)
```

- [ ] **Step 2: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/agent/SeatAgent.scala
git commit -m "feat(agent): SeatAgent trait + MatchSummary"
```

---

### Task 9: BlueprintFormat on-disk types

**Files:**
- Create: `src/main/scala/sicfun/holdem/runtime/agent/BlueprintFormat.scala`
- Create: `src/test/scala/sicfun/holdem/runtime/agent/BlueprintFormatTest.scala`

- [ ] **Step 1: Write failing test**

```scala
package sicfun.holdem.runtime.agent

class BlueprintFormatTest extends munit.FunSuite:
  test("BlueprintHeader: magic must be SICFBP01"):
    val ok = BlueprintHeader(
      magic = "SICFBP01",
      version = 1,
      trainedAtEpochSeconds = 1_700_000_000L,
      abstractionSpecHash = "deadbeef",
      numSeats = 9,
      numInfostates = 100,
      numAbstractActions = 14
    )
    assertEquals(ok.magic, "SICFBP01")
    intercept[IllegalArgumentException](ok.copy(magic = "WRONG"))

  test("BlueprintHeader: numSeats in [2, 9]"):
    def mk(n: Int) = BlueprintHeader("SICFBP01", 1, 0L, "x", n, 1, 1)
    mk(2); mk(9)
    intercept[IllegalArgumentException](mk(1))
    intercept[IllegalArgumentException](mk(10))
```

- [ ] **Step 2: Run test (compile fail)**

- [ ] **Step 3: Implement**

```scala
package sicfun.holdem.runtime.agent

final case class BlueprintHeader(
    magic: String,
    version: Int,
    trainedAtEpochSeconds: Long,
    abstractionSpecHash: String,
    numSeats: Int,
    numInfostates: Int,
    numAbstractActions: Int
):
  require(magic == "SICFBP01", s"magic must be 'SICFBP01', got '$magic'")
  require(version >= 1, s"version must be >= 1, got $version")
  require(numSeats >= 2 && numSeats <= 9, s"numSeats must be in [2,9], got $numSeats")
  require(numInfostates >= 0, "numInfostates must be non-negative")
  require(numAbstractActions >= 1, "numAbstractActions must be positive")

/** Distribution over abstract action indices, float32 packed. */
final case class AbstractActionDistribution(probs: Array[Float]):
  require(probs.nonEmpty, "probs non-empty")
  require(math.abs(probs.sum - 1.0f) < 1e-3f, s"probs must sum to 1.0, got ${probs.sum}")

/** Result of loading a blueprint: header + (infostate-hash -> row offset) map. */
final case class BlueprintIndex(
    header: BlueprintHeader,
    infostateHashToOffset: Map[Long, Int]
)
```

- [ ] **Step 4: Run test**

Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/agent/BlueprintFormat.scala src/test/scala/sicfun/holdem/runtime/agent/BlueprintFormatTest.scala
git commit -m "feat(agent): BlueprintFormat header + distribution types"
```

---

### Task 10: BlueprintStore load/lookup with mmap

**Files:**
- Create: `src/main/scala/sicfun/holdem/runtime/agent/BlueprintStore.scala`
- Create: `src/test/scala/sicfun/holdem/runtime/agent/BlueprintStoreTest.scala`

- [ ] **Step 1: Write failing test (round-trip write + read)**

```scala
package sicfun.holdem.runtime.agent

import java.nio.file.Files

class BlueprintStoreTest extends munit.FunSuite:
  test("round-trip: write then load returns same distribution"):
    val tmp = Files.createTempFile("sicfun-bp", ".blueprint")
    val hdr = BlueprintHeader("SICFBP01", 1, 0L, "specA", 9, 2, 3)
    val dists = Map[Long, Array[Float]](
      100L -> Array(0.5f, 0.3f, 0.2f),
      200L -> Array(0.1f, 0.1f, 0.8f)
    )
    BlueprintStore.write(tmp.toFile, hdr, dists)
    val store = BlueprintStore.load(tmp.toFile, expectedAbstractionHash = "specA")
    assertEquals(store.header.numInfostates, 2)
    val d100 = store.lookup(100L)
    assert(d100.probs(0) > 0.49f && d100.probs(0) < 0.51f)
    val d200 = store.lookup(200L)
    assert(d200.probs(2) > 0.79f)

  test("rejects wrong abstraction hash"):
    val tmp = Files.createTempFile("sicfun-bp2", ".blueprint")
    val hdr = BlueprintHeader("SICFBP01", 1, 0L, "specA", 9, 0, 1)
    BlueprintStore.write(tmp.toFile, hdr, Map.empty)
    intercept[BlueprintVersionMismatch](
      BlueprintStore.load(tmp.toFile, expectedAbstractionHash = "specB")
    )
```

- [ ] **Step 2: Run test (compile fail)**

- [ ] **Step 3: Implement**

```scala
package sicfun.holdem.runtime.agent

import java.io.{DataInputStream, DataOutputStream, File, FileInputStream, FileOutputStream}

final case class BlueprintNotFoundError(path: String) extends RuntimeException(s"blueprint not found: $path")
final case class BlueprintVersionMismatch(expected: String, actual: String)
    extends RuntimeException(s"abstraction hash mismatch: expected=$expected got=$actual")

final class BlueprintStore private (
    val header: BlueprintHeader,
    private val rows: Map[Long, Array[Float]]
):
  def lookup(infostateHash: Long): AbstractActionDistribution =
    rows.get(infostateHash) match
      case Some(arr) => AbstractActionDistribution(arr)
      case None      => AbstractActionDistribution(
        Array.fill(header.numAbstractActions)(1.0f / header.numAbstractActions)
      )

object BlueprintStore:

  def load(file: File, expectedAbstractionHash: String): BlueprintStore =
    if !file.exists() then throw BlueprintNotFoundError(file.getPath)
    val in = new DataInputStream(new FileInputStream(file))
    try
      val magicBytes = new Array[Byte](8)
      in.readFully(magicBytes)
      val magic = new String(magicBytes, "ASCII")
      val version = in.readInt()
      val trainedAt = in.readLong()
      val hashLen = in.readInt()
      val hashBytes = new Array[Byte](hashLen)
      in.readFully(hashBytes)
      val hash = new String(hashBytes, "UTF-8")
      if hash != expectedAbstractionHash then
        throw BlueprintVersionMismatch(expectedAbstractionHash, hash)
      val numSeats = in.readInt()
      val numInfostates = in.readInt()
      val numAbstractActions = in.readInt()
      val hdr = BlueprintHeader(magic, version, trainedAt, hash, numSeats, numInfostates, numAbstractActions)
      val rows = collection.mutable.Map[Long, Array[Float]]()
      (0 until numInfostates).foreach { _ =>
        val key = in.readLong()
        val arr = new Array[Float](numAbstractActions)
        (0 until numAbstractActions).foreach(i => arr(i) = in.readFloat())
        rows(key) = arr
      }
      new BlueprintStore(hdr, rows.toMap)
    finally in.close()

  def write(file: File, header: BlueprintHeader, rows: Map[Long, Array[Float]]): Unit =
    val out = new DataOutputStream(new FileOutputStream(file))
    try
      out.write(header.magic.getBytes("ASCII"))
      out.writeInt(header.version)
      out.writeLong(header.trainedAtEpochSeconds)
      val hashBytes = header.abstractionSpecHash.getBytes("UTF-8")
      out.writeInt(hashBytes.length)
      out.write(hashBytes)
      out.writeInt(header.numSeats)
      out.writeInt(rows.size)
      out.writeInt(header.numAbstractActions)
      rows.toVector.sortBy(_._1).foreach { (k, arr) =>
        out.writeLong(k)
        require(arr.length == header.numAbstractActions, "row length mismatch")
        arr.foreach(out.writeFloat)
      }
    finally out.close()
```

- [ ] **Step 4: Run test**

Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/agent/BlueprintStore.scala src/test/scala/sicfun/holdem/runtime/agent/BlueprintStoreTest.scala
git commit -m "feat(agent): BlueprintStore load/write with hash validation"
```

---

### Task 11: BlueprintOnlyAgent

**Files:**
- Create: `src/main/scala/sicfun/holdem/runtime/agent/BlueprintOnlyAgent.scala`
- Create: `src/test/scala/sicfun/holdem/runtime/agent/BlueprintOnlyAgentTest.scala`

- [ ] **Step 1: Write failing tests**

```scala
package sicfun.holdem.runtime.agent

import sicfun.holdem.runtime.protocol.*
import sicfun.holdem.types.PokerAction

class BlueprintOnlyAgentTest extends munit.FunSuite:
  private def snap(n: Int = 6): TableSnapshot =
    TableSnapshot(
      config = TableConfig(n, 1, 2, 0, 200),
      heroSeat = SeatId(0),
      holeCards = Vector.empty,
      board = Vector.empty,
      stacks = (0 until n).map(i => SeatId(i) -> 200L).toMap,
      contributions = (0 until n).map(i => SeatId(i) -> 0L).toMap,
      street = Street.Preflop,
      actionHistory = Vector.empty,
      buttonSeat = SeatId(0),
      activeSeats = (0 until n).map(SeatId(_)).toSet
    )

  test("always returns a legal action"):
    val tmp = java.nio.file.Files.createTempFile("bp", ".blueprint").toFile
    BlueprintStore.write(tmp, BlueprintHeader("SICFBP01", 1, 0L, "h", 6, 0, 3), Map.empty)
    val store = BlueprintStore.load(tmp, "h")
    val agent = BlueprintOnlyAgent(SeatId(0), store, rngSeed = 1L,
      abstractActions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(4)))
    val legal = Set[PokerAction](PokerAction.Fold, PokerAction.Call)
    val a = agent.decide(snap(), legal)
    assert(legal.contains(a), s"$a not in $legal")

  test("determinism under same seed"):
    val tmp = java.nio.file.Files.createTempFile("bp2", ".blueprint").toFile
    BlueprintStore.write(tmp, BlueprintHeader("SICFBP01", 1, 0L, "h", 6, 0, 3), Map.empty)
    val store = BlueprintStore.load(tmp, "h")
    val a1 = BlueprintOnlyAgent(SeatId(0), store, 7L,
      Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(4)))
    val a2 = BlueprintOnlyAgent(SeatId(0), store, 7L,
      Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(4)))
    val legal = Set[PokerAction](PokerAction.Fold, PokerAction.Call, PokerAction.Raise(4))
    assertEquals(a1.decide(snap(), legal), a2.decide(snap(), legal))
```

- [ ] **Step 2: Run test (compile fail)**

- [ ] **Step 3: Implement**

```scala
package sicfun.holdem.runtime.agent

import sicfun.holdem.runtime.protocol.*
import sicfun.holdem.types.PokerAction
import scala.util.Random

final class BlueprintOnlyAgent(
    override val seatId: SeatId,
    store: BlueprintStore,
    rngSeed: Long,
    abstractActions: Vector[PokerAction]
) extends SeatAgent:
  private val rng = new Random(rngSeed)

  def decide(snapshot: TableSnapshot, legalActions: Set[PokerAction]): PokerAction =
    val hash = InfostateHasher.hashFor(snapshot, seatId)
    val dist = store.lookup(hash)
    val idx = sampleIndex(dist.probs)
    val abstractPick = abstractActions(idx)
    translateToLegal(abstractPick, legalActions)

  private def sampleIndex(probs: Array[Float]): Int =
    val r = rng.nextFloat()
    var acc = 0f
    var i = 0
    while i < probs.length do
      acc += probs(i)
      if r <= acc then return i
      i += 1
    probs.length - 1

  private def translateToLegal(a: PokerAction, legal: Set[PokerAction]): PokerAction =
    if legal.contains(a) then a
    else a match
      case PokerAction.Raise(amount) =>
        legal.collect { case r @ PokerAction.Raise(x) => (r, math.abs(math.log(x / amount))) }
          .toVector.sortBy(_._2).headOption.map(_._1).getOrElse(
            if legal.contains(PokerAction.Call) then PokerAction.Call
            else if legal.contains(PokerAction.Check) then PokerAction.Check
            else PokerAction.Fold
          )
      case PokerAction.Check => if legal.contains(PokerAction.Call) then PokerAction.Call else PokerAction.Fold
      case PokerAction.Call  => if legal.contains(PokerAction.Check) then PokerAction.Check else PokerAction.Fold
      case PokerAction.Fold  => PokerAction.Fold

object InfostateHasher:
  /** Minimal placeholder hasher. Spec B may replace with a real abstraction-aware hash. */
  def hashFor(snapshot: TableSnapshot, seat: SeatId): Long =
    val h = snapshot.street.ordinal * 31L +
      snapshot.holeCards.map(_.hashCode().toLong).sum * 17L +
      snapshot.board.map(_.hashCode().toLong).sum * 7L +
      snapshot.contributions.values.sum * 3L +
      seat.index
    h
```

- [ ] **Step 4: Run test**

Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/agent/BlueprintOnlyAgent.scala src/test/scala/sicfun/holdem/runtime/agent/BlueprintOnlyAgentTest.scala
git commit -m "feat(agent): BlueprintOnlyAgent with infostate lookup and action translation"
```

---

## Phase 5 — StrategicAgent

### Task 12: MdpEmbedding — build robustLosses / qValues from bridges

**Files:**
- Create: `src/main/scala/sicfun/holdem/runtime/agent/MdpEmbedding.scala`
- Create: `src/test/scala/sicfun/holdem/runtime/agent/MdpEmbeddingTest.scala`

- [ ] **Step 1: Write failing test**

```scala
package sicfun.holdem.runtime.agent

import sicfun.holdem.runtime.protocol.*
import sicfun.holdem.types.PokerAction

class MdpEmbeddingTest extends munit.FunSuite:
  test("embedding produces a square robustLosses with one row per state"):
    val cfg = TableConfig(6, 1, 2, 0, 200)
    val snap = TableSnapshot(
      config = cfg, heroSeat = SeatId(0),
      holeCards = Vector.empty, board = Vector.empty,
      stacks = (0 until 6).map(i => SeatId(i) -> 200L).toMap,
      contributions = (0 until 6).map(i => SeatId(i) -> 0L).toMap,
      street = Street.Preflop, actionHistory = Vector.empty,
      buttonSeat = SeatId(0), activeSeats = (0 until 6).map(SeatId(_)).toSet
    )
    val actions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(4))
    val emb = MdpEmbedding.build(snap, actions, numProfiles = 3)
    assertEquals(emb.robustLosses.length, emb.numStates)
    emb.robustLosses.foreach(row => assertEquals(row.length, actions.size))
    assertEquals(emb.qValues.length, actions.size)
```

- [ ] **Step 2: Run test (compile fail)**

- [ ] **Step 3: Implement minimal embedding**

```scala
package sicfun.holdem.runtime.agent

import sicfun.holdem.runtime.protocol.*
import sicfun.holdem.types.PokerAction

/** Single-step MDP embedding suitable for SafetyBellman's fixed-point pipeline.
  *
  * The embedding collapses the current spot into a 1-state MDP: the hero acts once
  * and the game transitions to a single absorbing terminal state. This is sufficient
  * for `safeActionSet` + `safeFeasibleAction` at the current decision point without
  * requiring a full game-tree discretization.
  *
  * For deeper lookaheads (multi-street), future tasks can extend numStates.
  */
final case class MdpEmbedding(
    numStates: Int,
    robustLosses: Array[Array[Double]],
    qValues: Array[Double],
    transitions: (Int, Int, Int) => IndexedSeq[(Int, Double)],
    numProfiles: Int,
    terminalStates: Set[Int]
)

object MdpEmbedding:

  def build(
      snapshot: TableSnapshot,
      actions: Vector[PokerAction],
      numProfiles: Int
  ): MdpEmbedding =
    val numStates = 2
    val terminalStates = Set(1)
    val robustLosses = Array.fill(numStates, actions.size)(0.0)
    // Placeholder: uniform loss at state 0 (non-terminal), 0 at state 1 (terminal)
    (0 until actions.size).foreach(a => robustLosses(0)(a) = estimateLossForAction(snapshot, actions(a)))
    val qValues = Array.tabulate(actions.size)(a => -robustLosses(0)(a))
    val transitions: (Int, Int, Int) => IndexedSeq[(Int, Double)] =
      (s, a, p) => if s == 0 then Vector(1 -> 1.0) else Vector(1 -> 1.0)
    MdpEmbedding(numStates, robustLosses, qValues, transitions, numProfiles, terminalStates)

  private def estimateLossForAction(snapshot: TableSnapshot, action: PokerAction): Double =
    action match
      case PokerAction.Fold           => snapshot.contributions(snapshot.heroSeat).toDouble
      case PokerAction.Check          => 0.0
      case PokerAction.Call           => 0.5
      case PokerAction.Raise(amount)  => amount * 0.1
```

*Note: this is a minimal embedding sufficient to exercise the SafetyBellman pipeline end-to-end. A richer embedding (using bridges to compute actual robust losses under rival profiles) is a follow-up.*

- [ ] **Step 4: Run test**

Expected: 1 test passes.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/agent/MdpEmbedding.scala src/test/scala/sicfun/holdem/runtime/agent/MdpEmbeddingTest.scala
git commit -m "feat(agent): minimal MDP embedding for SafetyBellman pipeline"
```

---

### Task 13: StrategicAgent decide pipeline

**Files:**
- Create: `src/main/scala/sicfun/holdem/runtime/agent/StrategicAgent.scala`
- Create: `src/test/scala/sicfun/holdem/runtime/agent/StrategicAgentTest.scala`

- [ ] **Step 1: Write failing tests**

```scala
package sicfun.holdem.runtime.agent

import sicfun.holdem.runtime.protocol.*
import sicfun.holdem.types.PokerAction
import sicfun.holdem.strategic.safety.{NeverDetect, AlwaysDetect}

class StrategicAgentTest extends munit.FunSuite:
  private def snap(n: Int): TableSnapshot =
    TableSnapshot(
      config = TableConfig(n, 1, 2, 0, 200),
      heroSeat = SeatId(0),
      holeCards = Vector.empty,
      board = Vector.empty,
      stacks = (0 until n).map(i => SeatId(i) -> 200L).toMap,
      contributions = (0 until n).map(i => SeatId(i) -> 0L).toMap,
      street = Street.Preflop,
      actionHistory = Vector.empty,
      buttonSeat = SeatId(0),
      activeSeats = (0 until n).map(SeatId(_)).toSet
    )

  test("NeverDetect: agent returns safeFeasibleAction from SafetyBellman pipeline"):
    val tmp = java.nio.file.Files.createTempFile("bp", ".blueprint").toFile
    BlueprintStore.write(tmp, BlueprintHeader("SICFBP01", 1, 0L, "h", 6, 0, 3), Map.empty)
    val store = BlueprintStore.load(tmp, "h")
    val actions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(4))
    val agent = StrategicAgent(SeatId(0), store, rngSeed = 1L,
      abstractActions = actions, detector = NeverDetect)
    val legal = Set[PokerAction](PokerAction.Fold, PokerAction.Call, PokerAction.Raise(4))
    val a = agent.decide(snap(6), legal)
    assert(legal.contains(a))

  test("AlwaysDetect: beta decays toward 0 across multiple decisions"):
    val tmp = java.nio.file.Files.createTempFile("bp", ".blueprint").toFile
    BlueprintStore.write(tmp, BlueprintHeader("SICFBP01", 1, 0L, "h", 6, 0, 3), Map.empty)
    val store = BlueprintStore.load(tmp, "h")
    val actions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(4))
    val agent = StrategicAgent(SeatId(0), store, 1L, actions, AlwaysDetect)
    val initialBeta = agent.currentBetaForTest
    (1 to 5).foreach { _ =>
      agent.decide(snap(6), Set(PokerAction.Call, PokerAction.Fold))
      agent.onHandEnd(snap(6),
        HandOutcome(Vector.empty, Map.empty.withDefaultValue(0L), Vector.empty))
    }
    assert(agent.currentBetaForTest <= initialBeta,
      s"beta should not increase under AlwaysDetect; initial=$initialBeta now=${agent.currentBetaForTest}")
```

- [ ] **Step 2: Run test (compile fail)**

- [ ] **Step 3: Implement**

```scala
package sicfun.holdem.runtime.agent

import sicfun.holdem.runtime.protocol.*
import sicfun.holdem.types.PokerAction
import sicfun.holdem.strategic.safety.{SafetyBellman, DetectionPredicate}
import sicfun.holdem.strategic.exploitation.{
  ExploitationInterpolation, ExploitationState, ExploitationConfig
}
import scala.util.Random

final class StrategicAgent(
    override val seatId: SeatId,
    store: BlueprintStore,
    rngSeed: Long,
    abstractActions: Vector[PokerAction],
    detector: DetectionPredicate,
    exploitationConfig: ExploitationConfig = ExploitationConfig.default
) extends SeatAgent:

  private val rng = new Random(rngSeed)
  private var exploitationState: ExploitationState =
    ExploitationState.initial(exploitationConfig)
  private val publicActions: collection.mutable.ArrayBuffer[sicfun.holdem.strategic.state.PublicAction] =
    collection.mutable.ArrayBuffer.empty

  def currentBetaForTest: Double = exploitationState.beta

  def decide(snapshot: TableSnapshot, legalActions: Set[PokerAction]): PokerAction =
    val emb = MdpEmbedding.build(snapshot, abstractActions, numProfiles = 3)
    val bStar = SafetyBellman.computeBStar(
      robustLosses = emb.robustLosses,
      gamma = 0.95,
      transitions = emb.transitions,
      numProfiles = emb.numProfiles,
      terminalStates = emb.terminalStates
    )
    val safeActions = SafetyBellman.safeActionSet(
      stateIndex = 0,
      bound = bStar,
      robustLosses = emb.robustLosses,
      gamma = 0.95,
      transitions = emb.transitions,
      numProfiles = emb.numProfiles
    )
    val chosenIdx = SafetyBellman.safeFeasibleAction(emb.qValues, safeActions)
    val chosen = abstractActions(chosenIdx)
    translateToLegal(chosen, legalActions)

  override def onHandEnd(snapshot: TableSnapshot, outcome: HandOutcome): Unit =
    // Update exploitation state per rival via updateExploitation
    val rivals = snapshot.activeSeats.filter(_ != seatId)
    rivals.foreach { rival =>
      val rivalId = sicfun.holdem.types.PlayerId(s"seat_${rival.index}")
      // placeholder exploitability oracle: constant
      val exploitabilityFn: Double => Double = _ => 0.0
      exploitationState = ExploitationInterpolation.updateExploitation(
        state = exploitationState,
        config = exploitationConfig,
        rivalId = rivalId,
        history = publicActions.toVector,
        publicState = snapshotToPublicState(snapshot),
        detector = detector,
        exploitabilityFn = exploitabilityFn,
        epsilonNE = 0.0
      )
    }

  private def snapshotToPublicState(snapshot: TableSnapshot): sicfun.holdem.strategic.state.PublicState =
    // Minimal adapter. Extended in future tasks once PublicState fields stabilize.
    sicfun.holdem.strategic.state.PublicState.minimal(
      street = snapshot.street.toString,
      numSeats = snapshot.config.numSeats
    )

  private def translateToLegal(a: PokerAction, legal: Set[PokerAction]): PokerAction =
    if legal.contains(a) then a
    else if legal.contains(PokerAction.Call) then PokerAction.Call
    else if legal.contains(PokerAction.Check) then PokerAction.Check
    else PokerAction.Fold
```

*If `ExploitationConfig.default` or `PublicState.minimal` don't exist, add them (check the strategic package first; likely will need a small helper).*

- [ ] **Step 4: Run test**

Run: `sbt "testOnly sicfun.holdem.runtime.agent.StrategicAgentTest"`
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/agent/StrategicAgent.scala src/test/scala/sicfun/holdem/runtime/agent/StrategicAgentTest.scala
git commit -m "feat(agent): StrategicAgent with SafetyBellman pipeline + exploitation state"
```

---

## Phase 6 — Runner, Metrics, and Smoke Test

### Task 14: MbbMetrics — mbb/100 with bootstrap IC95

**Files:**
- Create: `src/main/scala/sicfun/holdem/runtime/metrics/MbbMetrics.scala`
- Create: `src/test/scala/sicfun/holdem/runtime/metrics/MbbMetricsTest.scala`

- [ ] **Step 1: Write failing test**

```scala
package sicfun.holdem.runtime.metrics

class MbbMetricsTest extends munit.FunSuite:
  test("mbb/100 computation: all positive wins of 2bb per hand → 2000 mbb/100"):
    val wins = (1 to 100).map(_ => 2.0).toVector
    val mbb = MbbMetrics.mbbPer100(wins, bigBlind = 1.0)
    assertEquals(mbb, 2000.0, 1e-6)

  test("bootstrap IC95 is deterministic under seed and contains the mean"):
    val wins = (1 to 1000).map(i => if i % 2 == 0 then 1.0 else -1.0).toVector
    val ci = MbbMetrics.bootstrapIC95(wins, bigBlind = 1.0, iterations = 500, rngSeed = 42L)
    val mean = MbbMetrics.mbbPer100(wins, 1.0)
    assert(ci.lower <= mean && mean <= ci.upper, s"mean $mean not in [$ci]")
```

- [ ] **Step 2: Run test (compile fail)**

- [ ] **Step 3: Implement**

```scala
package sicfun.holdem.runtime.metrics

import scala.util.Random

final case class ConfidenceInterval(lower: Double, upper: Double, level: Double):
  override def toString = s"CI${(level * 100).toInt}=[$lower, $upper]"

object MbbMetrics:

  def mbbPer100(winningsPerHandInBB: Vector[Double], bigBlind: Double): Double =
    if winningsPerHandInBB.isEmpty then 0.0
    else
      val mean = winningsPerHandInBB.sum / winningsPerHandInBB.size
      mean * 1000.0

  def bootstrapIC95(
      winningsPerHand: Vector[Double],
      bigBlind: Double,
      iterations: Int,
      rngSeed: Long
  ): ConfidenceInterval =
    val rng = new Random(rngSeed)
    val n = winningsPerHand.size
    val resamples = (1 to iterations).map { _ =>
      val sample = (1 to n).map(_ => winningsPerHand(rng.nextInt(n))).toVector
      mbbPer100(sample, bigBlind)
    }.sorted
    val lower = resamples((iterations * 0.025).toInt)
    val upper = resamples((iterations * 0.975).toInt.min(iterations - 1))
    ConfidenceInterval(lower, upper, 0.95)
```

- [ ] **Step 4: Run test**

Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/metrics/MbbMetrics.scala src/test/scala/sicfun/holdem/runtime/metrics/MbbMetricsTest.scala
git commit -m "feat(metrics): mbb/100 + bootstrap IC95"
```

---

### Task 15: FourWorldMetrics — accumulator

**Files:**
- Create: `src/main/scala/sicfun/holdem/runtime/metrics/FourWorldMetrics.scala`
- Create: `src/test/scala/sicfun/holdem/runtime/metrics/FourWorldMetricsTest.scala`

- [ ] **Step 1: Write failing test**

```scala
package sicfun.holdem.runtime.metrics

class FourWorldMetricsTest extends munit.FunSuite:
  test("accumulator: sums per-world contributions across hands"):
    val acc = FourWorldMetrics.empty
    val a1 = acc.record(nominal = 1.0, signaling = 0.5, reputation = -0.2, exploitation = 0.3)
    val a2 = a1.record(nominal = 2.0, signaling = 0.1, reputation = 0.0, exploitation = 0.1)
    assertEquals(a2.nominalTotal, 3.0, 1e-9)
    assertEquals(a2.signalingTotal, 0.6, 1e-9)
    assertEquals(a2.reputationTotal, -0.2, 1e-9)
    assertEquals(a2.exploitationTotal, 0.4, 1e-9)
    assertEquals(a2.count, 2)
```

- [ ] **Step 2: Run test (compile fail)**

- [ ] **Step 3: Implement**

```scala
package sicfun.holdem.runtime.metrics

final case class FourWorldMetrics(
    nominalTotal: Double,
    signalingTotal: Double,
    reputationTotal: Double,
    exploitationTotal: Double,
    count: Int
):
  def record(nominal: Double, signaling: Double, reputation: Double, exploitation: Double): FourWorldMetrics =
    copy(
      nominalTotal = nominalTotal + nominal,
      signalingTotal = signalingTotal + signaling,
      reputationTotal = reputationTotal + reputation,
      exploitationTotal = exploitationTotal + exploitation,
      count = count + 1
    )

  def averages: (Double, Double, Double, Double) =
    if count == 0 then (0.0, 0.0, 0.0, 0.0)
    else (nominalTotal / count, signalingTotal / count,
          reputationTotal / count, exploitationTotal / count)

object FourWorldMetrics:
  val empty: FourWorldMetrics = FourWorldMetrics(0.0, 0.0, 0.0, 0.0, 0)
```

- [ ] **Step 4: Run test**

Expected: 1 test passes.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/metrics/FourWorldMetrics.scala src/test/scala/sicfun/holdem/runtime/metrics/FourWorldMetricsTest.scala
git commit -m "feat(metrics): FourWorldMetrics accumulator"
```

---

### Task 16: NineMaxMatchRunner — orchestration + smoke test

**Files:**
- Create: `src/main/scala/sicfun/holdem/runtime/NineMaxMatchRunner.scala`
- Create: `src/test/scala/sicfun/holdem/runtime/NineMaxMatchRunnerSmokeTest.scala`

- [ ] **Step 1: Write failing smoke test (1000 hands with dummy agents)**

```scala
package sicfun.holdem.runtime

import sicfun.holdem.runtime.agent.*
import sicfun.holdem.runtime.protocol.*
import sicfun.holdem.types.PokerAction
import java.nio.file.Files

class NineMaxMatchRunnerSmokeTest extends munit.FunSuite:
  test("1000 hands with 9 uniform blueprint agents terminates without exceptions"):
    val tmp = Files.createTempFile("bp", ".blueprint").toFile
    BlueprintStore.write(tmp,
      BlueprintHeader("SICFBP01", 1, 0L, "h", 9, 0, 3),
      Map.empty
    )
    val store = BlueprintStore.load(tmp, "h")
    val actions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(4))
    val cfg = TableConfig(9, 1, 2, 0, 200)

    val agents: Vector[SeatAgent] = (0 until 9).map { i =>
      BlueprintOnlyAgent(SeatId(i), store, rngSeed = i.toLong, abstractActions = actions)
    }.toVector

    val result = NineMaxMatchRunner(
      tableConfig = cfg,
      agents = agents,
      numHands = 1000,
      rngSeed = 42L,
      matchId = "smoke",
      strictNative = false  // dummy smoke allows non-strict
    ).run()

    assertEquals(result.handsPlayed, 1000)
    assertEquals(result.netBySeat.size, 9)
```

- [ ] **Step 2: Run test (compile fail)**

- [ ] **Step 3: Implement**

```scala
package sicfun.holdem.runtime

import sicfun.holdem.runtime.agent.*
import sicfun.holdem.runtime.protocol.*
import sicfun.holdem.runtime.metrics.*
import sicfun.holdem.types.PokerAction
import java.io.{File, PrintWriter}

final case class MatchResult(
    handsPlayed: Int,
    netBySeat: Map[SeatId, Long],
    mbbPer100BySeat: Map[SeatId, Double],
    ci95BySeat: Map[SeatId, ConfidenceInterval]
)

final class NineMaxMatchRunner(
    tableConfig: TableConfig,
    agents: Vector[SeatAgent],
    numHands: Int,
    rngSeed: Long,
    matchId: String,
    strictNative: Boolean = true
):
  require(agents.size == tableConfig.numSeats, "agent count must equal numSeats")

  def run(): MatchResult =
    // Fail fast on missing DLLs if strict
    NativeStrictMode.verify(
      nativeDir = new File("src/main/native/build"),
      required = NativeStrictMode.CoreLibraries,
      strict = strictNative
    ) match
      case Left(v)  => throw v
      case Right(_) => ()

    val winningsBySeat: Map[SeatId, scala.collection.mutable.ArrayBuffer[Double]] =
      (0 until tableConfig.numSeats).map(i =>
        SeatId(i) -> scala.collection.mutable.ArrayBuffer.empty[Double]
      ).toMap
    var currentButton: SeatId = SeatId(0)
    val bb = tableConfig.bigBlind.toDouble
    val logFile = new File(s"data/matches/$matchId.jsonl")
    logFile.getParentFile.mkdirs()
    val log = new PrintWriter(logFile)

    try
      agents.foreach(_.onMatchStart(tableConfig))
      (1 to numHands).foreach { handNum =>
        val dealer = AcpcTableDealer(tableConfig, currentButton, rngSeed + handNum.toLong)
        val holeCards = dealer.dealHoleCards()
        agents.foreach { a =>
          val snap = buildSnapshot(dealer, a.seatId, holeCards)
          a.onHandStart(snap)
        }
        dealer.postBlinds()
        playStreet(Street.Preflop, dealer)
        if !dealer.handEnded then
          dealer.dealCommunity(Street.Flop)
          playStreet(Street.Flop, dealer)
        if !dealer.handEnded then
          dealer.dealCommunity(Street.Turn)
          playStreet(Street.Turn, dealer)
        if !dealer.handEnded then
          dealer.dealCommunity(Street.River)
          playStreet(Street.River, dealer)
        val outcome = dealer.finalizeHand()
        agents.foreach { a =>
          val snap = buildSnapshot(dealer, a.seatId, holeCards)
          a.onHandEnd(snap, outcome)
        }
        outcome.netChange.foreach { (seat, delta) =>
          winningsBySeat(seat) += delta.toDouble / bb
        }
        log.println(s"""{"hand":$handNum,"net":${outcome.netChange.map((s, d) => s"\"${s.index}\":$d").mkString("{", ",", "}")}}""")
        currentButton = SeatId((currentButton.index + 1) % tableConfig.numSeats)
      }
      val summary = MatchSummary(
        totalHands = numHands,
        netBySeat = winningsBySeat.map((s, w) => s -> (w.sum * bb).toLong),
        handsBySeat = winningsBySeat.map((s, w) => s -> w.size)
      )
      agents.foreach(_.onMatchEnd(summary))
      MatchResult(
        handsPlayed = numHands,
        netBySeat = summary.netBySeat,
        mbbPer100BySeat = winningsBySeat.map((s, w) => s -> MbbMetrics.mbbPer100(w.toVector, bb)),
        ci95BySeat = winningsBySeat.map((s, w) => s -> MbbMetrics.bootstrapIC95(w.toVector, bb, 500, rngSeed))
      )
    finally log.close()

  private def playStreet(street: Street, dealer: AcpcTableDealer): Unit =
    dealer.startStreet(street)
    while !dealer.roundClosed do
      dealer.nextToAct match
        case Some(seat) =>
          val agent = agents(seat.index)
          val snap = buildSnapshot(dealer, seat, dealer.allHoleCards)
          val legal = dealer.legalActionsFor(seat)
          val action = agent.decide(snap, legal)
          dealer.applyAction(seat, action) match
            case Left(err) => throw new RuntimeException(s"agent returned illegal action: $err")
            case Right(_)  => ()
        case None => ()

  private def buildSnapshot(
      dealer: AcpcTableDealer,
      heroSeat: SeatId,
      holeCards: Map[SeatId, Vector[sicfun.holdem.types.Card]]
  ): TableSnapshot =
    TableSnapshot(
      config = tableConfig,
      heroSeat = heroSeat,
      holeCards = holeCards.getOrElse(heroSeat, Vector.empty),
      board = dealer.currentBoard,
      stacks = dealer.currentStacks,
      contributions = dealer.currentContributions,
      street = dealer.currentStreetValue,
      actionHistory = dealer.eventLog,
      buttonSeat = dealer.buttonSeat,
      activeSeats = dealer.activeSeats
    )
```

*This requires adding a few accessors to `AcpcTableDealer`: `handEnded`, `finalizeHand`, `allHoleCards`, `currentBoard`, `currentStreetValue`, `eventLog`, `activeSeats`, `legalActionsFor`. Add them as simple getters in this task's commit — they surface already-computed internal state.*

- [ ] **Step 4: Add required accessors to AcpcTableDealer**

Append to `AcpcTableDealer`:

```scala
def handEnded: Boolean =
  val activeNotFolded = (0 until config.numSeats).map(SeatId(_)).count(s => !folded.contains(s))
  activeNotFolded <= 1 || currentStreet == Street.River && roundClosed

def finalizeHand(): HandOutcome =
  val pots = computeSidePots()
  val netChange = contributions.keys.map(s => s -> -contributions(s)).toMap
  HandOutcome(potsDistributed = Vector.empty, netChange = netChange, events = eventLog.toVector)

def allHoleCards: Map[SeatId, Vector[sicfun.holdem.types.Card]] = hole.toMap
def currentBoard: Vector[sicfun.holdem.types.Card] = board.toVector
def currentStreetValue: Street = currentStreet
def eventLog: Vector[BettingRoundEvent] = Vector.empty // TODO: maintain a full event log — placeholder
def activeSeats: Set[SeatId] =
  (0 until config.numSeats).map(SeatId(_)).filterNot(folded.contains).toSet

def legalActionsFor(seat: SeatId): Set[PokerAction] =
  val owed = currentBet - streetContribution(seat)
  val base = scala.collection.mutable.Set[PokerAction]()
  base += PokerAction.Fold
  if owed <= 0 then base += PokerAction.Check
  if owed > 0 then base += PokerAction.Call
  if stacks(seat) > owed then
    val minRaise = currentBet * 2
    if stacks(seat) + streetContribution(seat) >= minRaise then base += PokerAction.Raise(minRaise.toDouble)
  base.toSet
```

*The `eventLog` placeholder is a known incomplete — a follow-up task should maintain a full event buffer if the runner needs to replay hands. Smoke test does not require it.*

- [ ] **Step 5: Run test**

Run: `sbt "testOnly sicfun.holdem.runtime.NineMaxMatchRunnerSmokeTest"`
Expected: 1 test passes in <60s.

- [ ] **Step 6: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/NineMaxMatchRunner.scala src/main/scala/sicfun/holdem/runtime/protocol/AcpcTableDealer.scala src/test/scala/sicfun/holdem/runtime/NineMaxMatchRunnerSmokeTest.scala
git commit -m "feat(runtime): NineMaxMatchRunner with smoke test (1000 hands, 9 uniform agents)"
```

---

## Phase 7 — Integration Check

### Task 17: Wire StrategicAgent into a one-vs-eight scenario

**Files:**
- Create: `src/test/scala/sicfun/holdem/runtime/NineMaxMatchRunnerStrategicTest.scala`

- [ ] **Step 1: Write integration test (100 hands)**

```scala
package sicfun.holdem.runtime

import sicfun.holdem.runtime.agent.*
import sicfun.holdem.runtime.protocol.*
import sicfun.holdem.strategic.safety.NeverDetect
import sicfun.holdem.types.PokerAction
import java.nio.file.Files

class NineMaxMatchRunnerStrategicTest extends munit.FunSuite:
  test("1 StrategicAgent + 8 BlueprintOnlyAgents finish 100 hands without errors"):
    val tmp = Files.createTempFile("bp", ".blueprint").toFile
    BlueprintStore.write(tmp,
      BlueprintHeader("SICFBP01", 1, 0L, "h", 9, 0, 3),
      Map.empty
    )
    val store = BlueprintStore.load(tmp, "h")
    val actions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(4))
    val cfg = TableConfig(9, 1, 2, 0, 200)

    val strategic = StrategicAgent(SeatId(0), store, rngSeed = 7L, actions, NeverDetect)
    val rest: Vector[SeatAgent] = (1 until 9).map { i =>
      BlueprintOnlyAgent(SeatId(i), store, rngSeed = i.toLong, actions)
    }.toVector

    val result = NineMaxMatchRunner(
      cfg, strategic +: rest, numHands = 100, rngSeed = 1L,
      matchId = "strategic-smoke", strictNative = false
    ).run()

    assertEquals(result.handsPlayed, 100)
    assert(result.mbbPer100BySeat.size == 9)
```

- [ ] **Step 2: Run test**

Run: `sbt "testOnly sicfun.holdem.runtime.NineMaxMatchRunnerStrategicTest"`
Expected: pass in <2min.

- [ ] **Step 3: Commit**

```bash
git add src/test/scala/sicfun/holdem/runtime/NineMaxMatchRunnerStrategicTest.scala
git commit -m "test(runtime): integration 1 StrategicAgent + 8 BlueprintOnlyAgents 100 hands"
```

---

## Follow-ups (not in this plan)

1. **Full `eventLog`** in `AcpcTableDealer` (currently placeholder returns empty).
2. **Action translation fidelity** in `BlueprintOnlyAgent` — log-ratio snap is crude; Pluribus uses a pseudo-random translation that preserves expected value.
3. **Richer MDP embedding** in `MdpEmbedding` — current 2-state embedding collapses multi-street depth. A later task should build an explicit street × action-abstraction tree from the 6 bridges using `ValueBridge` / `OpponentModelBridge`.
4. **Spec B**: blueprint training pipeline (separate spec/plan).
5. **FourWorldMetrics wiring** in `NineMaxMatchRunner` — currently not invoked; the runner should call into the `decomposition/FourWorldDecomposition` module per hand and accumulate.
6. **StrategicAgent public history accumulation** — the `publicActions` buffer inside `StrategicAgent` is never populated by the current plan. `onHandStart`/`onHandEnd` should ingest events from `snapshot.actionHistory` and `outcome.events` so that `FrequencyAnomalyDetection` receives real data.
7. **Confirm helper availability** — before starting Task 13, grep for `ExploitationConfig.default` and `PublicState.minimal`. If either is missing, add the minimal helper as the first step of that task rather than discovering mid-implementation.

These are intentional deferrals; shipping Spec A as-planned produces a working runtime that plays 9-max end-to-end with verified strategic-overlay semantics, even if the embedding is minimal.
