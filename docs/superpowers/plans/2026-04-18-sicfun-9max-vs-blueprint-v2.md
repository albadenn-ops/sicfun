# sicfun 9-max vs. Blueprint — Plan v2 (Spec A.1 only)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Supersedes:** [v1](2026-04-18-sicfun-9max-vs-blueprint.md). v1 contenía bugs concretos (Task 4 round-closure off-by-one; Task 14 unidad `mbb/100` off por 100×; Task 2/3 rename de parámetro que rompía compile entre tasks) y numerosos placeholders silenciosos que invalidaban el criterio de éxito propio del spec (MdpEmbedding hardcoded, exploitability oracle constante, publicActions nunca populado, pot nunca repartido). v2 corrige los bugs, sustituye los stubs por un sistema `PlaceholderMarker` + `BenchmarkGate` runtime-enforced, y divide Spec A en A.1 + A.2.

**Scope (A.1 — infrastructure only):** Ship infra end-to-end para 9-max NLHE, con pot distribution real y métricas correctas. **No benchmark claim sobre valor del strategic overlay** — eso requiere A.2 (bridges reales + oráculo real). A.1 se valida con **9 `BlueprintOnlyAgent` idénticos en self-play** y conservación exacta de fichas.

**Not shipped here (→ A.2):**
- Real `MdpEmbedding` cableado con `ValueBridge` / `OpponentModelBridge` / etc.
- Real `ExploitabilityOracle` (no oracle-constante-cero).
- `FourWorldMetrics` wiring a `FourWorldDecomposition`.
- Convención Pluribus de duplicate-holdings / seat-rotation para cancelar varianza posicional.
- Blueprint training pipeline.

**Success criteria (A.1, exactos, no estadísticos):**
1. 9 `BlueprintOnlyAgent` juegan 1000 manos end-to-end sin excepciones.
2. **Conservación por mano:** para toda mano, `outcome.netChange.values.sum == 0` exacto (identidad, no estadística).
3. **Conservación por match:** `sum(mbbPer100BySeat.values) == 0` dentro de `1e-9` (float tolerance).
4. **BenchmarkGate activa:** instanciar `NineMaxMatchRunner(benchmarkMode = true)` con algún `StrategicAgent` lanza `BenchmarkGateViolation` enumerando cada `PlaceholderMarker` y su `placeholderReason`.
5. **IC95 sanity:** para cada asiento, IC95 bootstrap de mbb/100 incluye 0 (señal secundaria; no bloquea si los invariantes 1–4 pasan).

**Tech stack:** Scala 3.8.1 + munit 1.2.2 + scalacheck 1.0.0. Reutiliza `sicfun.holdem.types.{Street, Board, Card, PokerAction}`, `sicfun.core.{Card, Deck, HandEvaluator}`, `sicfun.holdem.strategic.types.{Chips, PlayerId, TableMap, Seat, SeatStatus, Position}`, `sicfun.holdem.strategic.state.{PublicState, PublicAction, ActionSignal}`, `sicfun.holdem.strategic.safety.{SafetyBellman, DetectionPredicate, NeverDetect, AlwaysDetect}`, `sicfun.holdem.strategic.exploitation.{ExploitationInterpolation, ExploitationConfig, ExploitationState}`.

**Spec:** [docs/superpowers/specs/2026-04-18-sicfun-9max-vs-blueprint-design.md](../specs/2026-04-18-sicfun-9max-vs-blueprint-design.md).

---

## File structure

**New (production, Scala):**
- `src/main/scala/sicfun/holdem/runtime/protocol/TableDealerTypes.scala` — `SeatId`, `SidePot`, `BlindKind`, `BettingRoundEvent`, `HandOutcome`, `TableConfig`, `TableSnapshot` (reuses existing `sicfun.holdem.types.Street`)
- `src/main/scala/sicfun/holdem/runtime/protocol/AcpcTableDealer.scala` — N-seat dealer with real hand evaluation + side-pot distribution
- `src/main/scala/sicfun/holdem/runtime/agent/SeatAgent.scala` — trait + `MatchSummary`
- `src/main/scala/sicfun/holdem/runtime/agent/BlueprintFormat.scala` — `BlueprintHeader`, `AbstractActionDistribution`
- `src/main/scala/sicfun/holdem/runtime/agent/BlueprintStore.scala` — binary load/write with hash validation
- `src/main/scala/sicfun/holdem/runtime/agent/BlueprintOnlyAgent.scala` — samples from blueprint, translates abstract → legal `PokerAction`
- `src/main/scala/sicfun/holdem/runtime/agent/InfostateHasher.scala` — trait + `PlaceholderInfostateHasher` (marked)
- `src/main/scala/sicfun/holdem/runtime/agent/MdpEmbedding.scala` — trait + `PlaceholderMdpEmbedding` (marked)
- `src/main/scala/sicfun/holdem/runtime/agent/ExploitabilityOracle.scala` — trait + `ZeroExploitabilityOracle` (marked)
- `src/main/scala/sicfun/holdem/runtime/agent/StrategicAgent.scala`
- `src/main/scala/sicfun/holdem/runtime/PlaceholderMarker.scala` — trait + recursive `scanPlaceholders` helper
- `src/main/scala/sicfun/holdem/runtime/NativeStrictMode.scala`
- `src/main/scala/sicfun/holdem/runtime/BenchmarkGate.scala`
- `src/main/scala/sicfun/holdem/runtime/NineMaxMatchRunner.scala`
- `src/main/scala/sicfun/holdem/runtime/metrics/MbbMetrics.scala`

**Modified:**
- `build.sbt` — add `munit-scalacheck` test dep.

**Not touched:** `AcpcHeadsUpDealer.scala`, `SlumbotMatchRunner.scala`, `AcpcMatchRunner.scala`, `strategic/**` (read-only consumer).

**Task count:** 25 tasks across 6 phases.

---

## Phase 1 — Dealer Foundation (4 tasks)

### Task 1: scalacheck dep + core table types

**Files:**
- Modify: `build.sbt`
- Create: `src/main/scala/sicfun/holdem/runtime/protocol/TableDealerTypes.scala`
- Create: `src/test/scala/sicfun/holdem/runtime/protocol/TableDealerTypesTest.scala`

- [ ] **Step 1: Add scalacheck to `build.sbt`**

Locate the `libraryDependencies` block that currently contains `"org.scalameta" %% "munit" % "1.2.2" % Test` (line ~13) and append:

```scala
libraryDependencies += "org.scalameta" %% "munit-scalacheck" % "1.0.0" % Test,
```

Run `sbt update` to confirm resolution.

- [ ] **Step 2: Write failing test for core types**

```scala
package sicfun.holdem.runtime.protocol

import sicfun.holdem.types.Street

class TableDealerTypesTest extends munit.FunSuite:
  test("SidePot: eligible seats non-empty and amount positive"):
    val ok = SidePot(amount = 100L, eligibleSeats = Set(SeatId(0), SeatId(1)))
    assertEquals(ok.amount, 100L)
    intercept[IllegalArgumentException](SidePot(0L, Set(SeatId(0))))
    intercept[IllegalArgumentException](SidePot(100L, Set.empty))

  test("TableConfig: numSeats in [2, 9]"):
    TableConfig(numSeats = 2, smallBlind = 1L, bigBlind = 2L, ante = 0L, startingStack = 200L)
    TableConfig(9, 1L, 2L, 0L, 200L)
    intercept[IllegalArgumentException](TableConfig(1, 1L, 2L, 0L, 200L))
    intercept[IllegalArgumentException](TableConfig(10, 1L, 2L, 0L, 200L))
    intercept[IllegalArgumentException](TableConfig(2, 0L, 2L, 0L, 200L)) // sb must be > 0
    intercept[IllegalArgumentException](TableConfig(2, 3L, 2L, 0L, 200L)) // bb >= sb required
    intercept[IllegalArgumentException](TableConfig(2, 1L, 2L, 0L, 10L)) // startingStack < 10*bb

  test("Street reuses sicfun.holdem.types.Street, not a new type"):
    val s: Street = Street.Flop
    assertEquals(s.expectedBoardSize, 3)
```

- [ ] **Step 3: Implement core types (reuse existing `Street`)**

```scala
package sicfun.holdem.runtime.protocol

import sicfun.holdem.types.{Card, PokerAction, Street}

opaque type SeatId = Int
object SeatId:
  def apply(i: Int): SeatId =
    require(i >= 0 && i < 9, s"SeatId must be in [0, 9), got $i")
    i
  extension (s: SeatId) def index: Int = s
  given Ordering[SeatId] = Ordering.Int

final case class TableConfig(
    numSeats: Int,
    smallBlind: Long,
    bigBlind: Long,
    ante: Long,
    startingStack: Long
):
  require(numSeats >= 2 && numSeats <= 9, s"numSeats must be in [2, 9], got $numSeats")
  require(smallBlind > 0L, s"smallBlind must be > 0, got $smallBlind")
  require(bigBlind >= smallBlind, s"bigBlind ($bigBlind) must be >= smallBlind ($smallBlind)")
  require(ante >= 0L, "ante must be non-negative")
  require(startingStack >= 10L * bigBlind, s"startingStack must be at least 10 bb, got $startingStack for bb=$bigBlind")

final case class SidePot(amount: Long, eligibleSeats: Set[SeatId]):
  require(amount > 0L, s"SidePot amount must be positive, got $amount")
  require(eligibleSeats.nonEmpty, "SidePot must have at least one eligible seat")

enum BlindKind:
  case SmallBlind, BigBlind

enum BettingRoundEvent:
  case PostBlind(seat: SeatId, amount: Long, kind: BlindKind)
  case PostAnte(seat: SeatId, amount: Long)
  case Act(seat: SeatId, action: PokerAction)
  case Deal(street: Street, cards: Vector[Card])
  case Showdown(revealed: Map[SeatId, Vector[Card]])
  case PotAwarded(pot: SidePot, distribution: Map[SeatId, Long])

final case class HandOutcome(
    potsDistributed: Vector[(SidePot, Map[SeatId, Long])],
    netChange: Map[SeatId, Long],
    events: Vector[BettingRoundEvent]
):
  /** A.1 invariant: chip conservation. Sum of net changes must be zero. */
  require(netChange.values.sum == 0L, s"chip conservation violated: sum(netChange) = ${netChange.values.sum}, expected 0")

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

- [ ] **Step 4: Run tests**

`sbt "testOnly sicfun.holdem.runtime.protocol.TableDealerTypesTest"` — 3 pass.

- [ ] **Step 5: Commit**

```bash
git add build.sbt src/main/scala/sicfun/holdem/runtime/protocol/TableDealerTypes.scala src/test/scala/sicfun/holdem/runtime/protocol/TableDealerTypesTest.scala
git commit -m "feat(runtime): table dealer core types + scalacheck dep (reuse holdem.types.Street)"
```

---

### Task 2: Dealer skeleton + blinds (positional args from day 1)

**Files:**
- Create: `src/main/scala/sicfun/holdem/runtime/protocol/AcpcTableDealer.scala`
- Create: `src/test/scala/sicfun/holdem/runtime/protocol/AcpcTableDealerTest.scala`

**Design note:** constructor params are **positional only** in tests. Avoid named `buttonSeat = ...` in tests so Task 3's rename doesn't break anything.

- [ ] **Step 1: Write failing test**

```scala
package sicfun.holdem.runtime.protocol

import sicfun.holdem.runtime.protocol.BettingRoundEvent.*
import sicfun.holdem.runtime.protocol.BlindKind.*

class AcpcTableDealerTest extends munit.FunSuite:
  val cfg2 = TableConfig(2, 1L, 2L, 0L, 200L)
  val cfg6 = TableConfig(6, 1L, 2L, 0L, 200L)
  val cfg9 = TableConfig(9, 1L, 2L, 0L, 200L)

  test("blinds at N=2: button is SB"):
    val d = AcpcTableDealer(cfg2, SeatId(0), 42L)
    val events = d.postBlinds()
    assertEquals(events, Vector(
      PostBlind(SeatId(0), 1L, SmallBlind),
      PostBlind(SeatId(1), 2L, BigBlind)
    ))

  test("blinds at N=6: SB = BTN+1, BB = BTN+2"):
    val d = AcpcTableDealer(cfg6, SeatId(2), 42L)
    val events = d.postBlinds()
    assertEquals(events, Vector(
      PostBlind(SeatId(3), 1L, SmallBlind),
      PostBlind(SeatId(4), 2L, BigBlind)
    ))

  test("blinds wrap around seat indices"):
    val d = AcpcTableDealer(cfg9, SeatId(8), 42L)
    val events = d.postBlinds()
    assertEquals(events, Vector(
      PostBlind(SeatId(0), 1L, SmallBlind),
      PostBlind(SeatId(1), 2L, BigBlind)
    ))
```

- [ ] **Step 2: Implement**

```scala
package sicfun.holdem.runtime.protocol

import sicfun.holdem.runtime.protocol.BettingRoundEvent.*
import sicfun.holdem.runtime.protocol.BlindKind.*
import scala.util.Random

final class AcpcTableDealer(
    val config: TableConfig,
    initialButtonSeat: SeatId,
    rngSeed: Long
):
  private val rng = new Random(rngSeed)
  private var _buttonSeat: SeatId = initialButtonSeat
  private val stacks = collection.mutable.Map[SeatId, Long]()
  private val contributions = collection.mutable.Map[SeatId, Long]()
    .withDefaultValue(0L)
  for i <- 0 until config.numSeats do stacks(SeatId(i)) = config.startingStack

  def buttonSeat: SeatId = _buttonSeat

  private def nextSeat(s: SeatId): SeatId =
    SeatId((s.index + 1) % config.numSeats)

  def smallBlindSeat: SeatId =
    if config.numSeats == 2 then _buttonSeat else nextSeat(_buttonSeat)

  def bigBlindSeat: SeatId = nextSeat(smallBlindSeat)

  def postBlinds(): Vector[BettingRoundEvent] =
    val sb = smallBlindSeat
    val bb = bigBlindSeat
    stacks(sb) -= config.smallBlind
    stacks(bb) -= config.bigBlind
    contributions(sb) = contributions(sb) + config.smallBlind
    contributions(bb) = contributions(bb) + config.bigBlind
    Vector(
      PostBlind(sb, config.smallBlind, SmallBlind),
      PostBlind(bb, config.bigBlind, BigBlind)
    )

  def currentStacks: Map[SeatId, Long] = stacks.toMap
  def currentContributions: Map[SeatId, Long] = contributions.toMap
```

- [ ] **Step 3: Run tests** — 3 pass.

- [ ] **Step 4: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/protocol/AcpcTableDealer.scala src/test/scala/sicfun/holdem/runtime/protocol/AcpcTableDealerTest.scala
git commit -m "feat(runtime): AcpcTableDealer skeleton with blinds posting"
```

---

### Task 3: Button rotation

**Files:** modify dealer + test.

- [ ] **Step 1: Add test**

```scala
  test("button rotates one seat per hand and cycles through all N=9 seats"):
    val d = AcpcTableDealer(cfg9, SeatId(0), 1L)
    val visited = (0 until 9).map { _ =>
      val b = d.buttonSeat
      d.advanceButton()
      b.index
    }.toSet
    assertEquals(visited, (0 until 9).toSet)

  test("button wraps modulo numSeats at N=2"):
    val d = AcpcTableDealer(cfg2, SeatId(1), 1L)
    d.advanceButton()
    assertEquals(d.buttonSeat, SeatId(0))
```

- [ ] **Step 2: Add `advanceButton`** (one-line addition)

```scala
  def advanceButton(): Unit =
    _buttonSeat = SeatId((_buttonSeat.index + 1) % config.numSeats)
```

- [ ] **Step 3: Run tests** — 5 total pass.

- [ ] **Step 4: Commit**

```bash
git commit -am "feat(runtime): button rotation for AcpcTableDealer"
```

---

### Task 4: Card dealing (Deck.full, hole cards, community)

**Files:** modify dealer + test.

- [ ] **Step 1: Add test**

```scala
  test("dealHoleCards: 2 distinct cards per seat, 2N cards total, no duplicates"):
    val d = AcpcTableDealer(cfg6, SeatId(0), 42L)
    val hole = d.dealHoleCards()
    assertEquals(hole.size, 6)
    hole.values.foreach(cs => assertEquals(cs.size, 2))
    val allCards = hole.values.flatten.toVector
    assertEquals(allCards.distinct.size, 12)

  test("dealCommunity: flop=3, turn=1, river=1, all distinct from hole"):
    val d = AcpcTableDealer(cfg6, SeatId(0), 42L)
    val hole = d.dealHoleCards()
    val flop = d.dealCommunity(sicfun.holdem.types.Street.Flop)
    val turn = d.dealCommunity(sicfun.holdem.types.Street.Turn)
    val river = d.dealCommunity(sicfun.holdem.types.Street.River)
    assertEquals(flop.size, 3)
    assertEquals(turn.size, 4)        // board accumulates
    assertEquals(river.size, 5)
    val all = hole.values.flatten.toSet ++ river.toSet
    assertEquals(all.size, 12 + 5)    // all distinct
```

- [ ] **Step 2: Implement**

```scala
// append to AcpcTableDealer
import sicfun.holdem.types.{Card, Street}
import sicfun.core.Deck

private val deckBuf: collection.mutable.ArrayBuffer[Card] =
  collection.mutable.ArrayBuffer.from(rng.shuffle(Deck.full))
private val hole: collection.mutable.Map[SeatId, Vector[Card]] =
  collection.mutable.Map.empty
private val boardBuf: collection.mutable.ArrayBuffer[Card] =
  collection.mutable.ArrayBuffer.empty

def dealHoleCards(): Map[SeatId, Vector[Card]] =
  (0 until config.numSeats).foreach { i =>
    val seat = SeatId(i)
    val c1 = deckBuf.remove(0)
    val c2 = deckBuf.remove(0)
    hole(seat) = Vector(c1, c2)
  }
  hole.toMap

def dealCommunity(street: Street): Vector[Card] =
  val count = street match
    case Street.Preflop => 0
    case Street.Flop    => 3
    case Street.Turn    => 1
    case Street.River   => 1
  (1 to count).foreach { _ => boardBuf += deckBuf.remove(0) }
  boardBuf.toVector

def currentBoard: Vector[Card] = boardBuf.toVector
def allHoleCards: Map[SeatId, Vector[Card]] = hole.toMap
```

- [ ] **Step 3: Run tests** — 7 total pass.

- [ ] **Step 4: Commit**

```bash
git commit -am "feat(runtime): card dealing via sicfun.core.Deck.full"
```

---

## Phase 2 — Betting rounds (3 tasks)

### Task 5: Round closure with aggressor-excluded actOrder

**Fix for v1 bug:** `buildActOrder(nextSeat(aggressor))` en v1 incluía al agresor al final de la rotación completa, lo que hacía que `actIdx > indexOf(aggressor)` nunca fuera true cuando todos los demás igualaban. v2: el actOrder post-raise contiene **solo los asientos que deben responder**, no al agresor.

**Files:** modify dealer + test.

- [ ] **Step 1: Add failing tests (including the v1-failing case)**

```scala
  test("action closes preflop when BB checks option after limpers (N=3)"):
    val cfg = TableConfig(3, 1L, 2L, 0L, 200L)
    val d = AcpcTableDealer(cfg, SeatId(0), 1L)
    d.postBlinds()
    d.startStreet(sicfun.holdem.types.Street.Preflop)
    assertEquals(d.nextToAct, Some(SeatId(0)))
    d.applyAction(SeatId(0), sicfun.holdem.types.PokerAction.Call)
      .getOrElse(fail("utg call"))
    d.applyAction(SeatId(1), sicfun.holdem.types.PokerAction.Call)
      .getOrElse(fail("sb call"))
    d.applyAction(SeatId(2), sicfun.holdem.types.PokerAction.Check)
      .getOrElse(fail("bb option"))
    assert(d.roundClosed, "round closed after BB option")

  test("action closes after BB flat-calls a raise (v1 bug regression)"):
    val cfg = TableConfig(3, 1L, 2L, 0L, 200L)
    val d = AcpcTableDealer(cfg, SeatId(0), 1L)
    d.postBlinds()
    d.startStreet(sicfun.holdem.types.Street.Preflop)
    d.applyAction(SeatId(0), sicfun.holdem.types.PokerAction.Raise(6.0))
      .getOrElse(fail("utg raise"))
    d.applyAction(SeatId(1), sicfun.holdem.types.PokerAction.Call)
      .getOrElse(fail("sb call"))
    assert(!d.roundClosed, "BB still has option")
    d.applyAction(SeatId(2), sicfun.holdem.types.PokerAction.Call)
      .getOrElse(fail("bb call"))
    assert(d.roundClosed, "closed after BB closes action on the raiser — v1 bug")
```

- [ ] **Step 2: Implement round logic (aggressor excluded from post-raise actOrder)**

```scala
// append to AcpcTableDealer

import sicfun.holdem.types.PokerAction

enum IllegalActionReason:
  case NotYourTurn(seat: SeatId, expected: Option[SeatId])
  case IllegalForm(seat: SeatId, action: PokerAction, reason: String)
  case InsufficientChips(seat: SeatId, required: Long, available: Long)

private var currentStreet: Street = Street.Preflop
private var currentBet: Long = 0L
private var lastAggressor: Option[SeatId] = None
private var actOrder: Vector[SeatId] = Vector.empty
private var actIdx: Int = 0
private val folded: collection.mutable.Set[SeatId] =
  collection.mutable.Set.empty
private val allIn: collection.mutable.Set[SeatId] =
  collection.mutable.Set.empty
private val streetContribution: collection.mutable.Map[SeatId, Long] =
  collection.mutable.Map.empty.withDefaultValue(0L)
private val eventBuffer: collection.mutable.ArrayBuffer[BettingRoundEvent] =
  collection.mutable.ArrayBuffer.empty

/** Build action order starting at `firstToAct`, cycling through all N seats,
  * then filter out folded/allIn. If `excludeAggressor` is set, also drops
  * that seat — used for post-raise rebuild where the raiser does not re-act
  * unless there is a further raise. */
private def buildActOrder(
    firstToAct: SeatId,
    excludeAggressor: Option[SeatId] = None
): Vector[SeatId] =
  val cycle = (0 until config.numSeats).map { off =>
    SeatId((firstToAct.index + off) % config.numSeats)
  }.toVector
  cycle.filter { s =>
    !folded.contains(s) &&
      !allIn.contains(s) &&
      !excludeAggressor.contains(s)
  }

def startStreet(street: Street): Unit =
  currentStreet = street
  streetContribution.clear()
  if street == Street.Preflop then
    currentBet = config.bigBlind
    streetContribution(smallBlindSeat) = config.smallBlind
    streetContribution(bigBlindSeat) = config.bigBlind
    lastAggressor = Some(bigBlindSeat)
    // Preflop: BB is the initial aggressor but keeps option — include BB.
    actOrder = buildActOrder(firstToAct = nextSeat(bigBlindSeat))
  else
    currentBet = 0L
    lastAggressor = None
    actOrder = buildActOrder(firstToAct = firstActivePostflop)
  actIdx = 0

private def firstActivePostflop: SeatId =
  var s = nextSeat(_buttonSeat)
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
    allMatched && actIdx >= actOrder.size

def applyAction(seat: SeatId, action: PokerAction): Either[IllegalActionReason, Unit] =
  if nextToAct != Some(seat) then
    Left(IllegalActionReason.NotYourTurn(seat, nextToAct))
  else action match
    case PokerAction.Fold =>
      folded += seat
      actIdx += 1
      eventBuffer += BettingRoundEvent.Act(seat, action)
      Right(())
    case PokerAction.Check =>
      if streetContribution(seat) != currentBet then
        Left(IllegalActionReason.IllegalForm(seat, action,
          s"cannot check facing a bet (currentBet=$currentBet, ownContribution=${streetContribution(seat)})"))
      else
        actIdx += 1
        eventBuffer += BettingRoundEvent.Act(seat, action)
        Right(())
    case PokerAction.Call =>
      val owed = currentBet - streetContribution(seat)
      if owed <= 0L then
        Left(IllegalActionReason.IllegalForm(seat, action, "nothing to call"))
      else
        val pay = math.min(owed, stacks(seat))
        stacks(seat) -= pay
        streetContribution(seat) = streetContribution(seat) + pay
        contributions(seat) = contributions(seat) + pay
        if stacks(seat) == 0L then allIn += seat
        actIdx += 1
        eventBuffer += BettingRoundEvent.Act(seat, action)
        Right(())
    case PokerAction.Raise(amountDouble) =>
      val target = amountDouble.toLong
      if target <= currentBet then
        Left(IllegalActionReason.IllegalForm(seat, action,
          s"raise $target must exceed currentBet $currentBet"))
      else
        val pay = target - streetContribution(seat)
        if pay > stacks(seat) then
          Left(IllegalActionReason.InsufficientChips(seat, pay, stacks(seat)))
        else
          stacks(seat) -= pay
          streetContribution(seat) = streetContribution(seat) + pay
          contributions(seat) = contributions(seat) + pay
          currentBet = target
          lastAggressor = Some(seat)
          if stacks(seat) == 0L then allIn += seat
          // Post-raise rebuild: exclude the raiser. Everyone else gets one turn.
          actOrder = buildActOrder(
            firstToAct = nextSeat(seat),
            excludeAggressor = Some(seat)
          )
          actIdx = 0
          eventBuffer += BettingRoundEvent.Act(seat, action)
          Right(())
```

- [ ] **Step 3: Run tests** — both new cases pass; existing blinds/rotation/dealing tests still pass.

- [ ] **Step 4: Commit**

```bash
git commit -am "feat(runtime): round closure with aggressor-excluded actOrder (fixes v1 off-by-one)"
```

---

### Task 6: `legalActionsFor` with 5-action discrete menu + dedup

**Menu (spot-dependent):** `{Fold, Passive, HalfPot, Pot, AllIn}` where `Passive = Check` if nothing owed else `Call`. After computing raise amounts, `Set[PokerAction]` deduplicates automatically when amounts coincide.

**Invariants to enforce:**
- No duplicate `PokerAction.Raise(x)` with same `x` (Set semantics).
- Menu is non-empty (Fold is always an option when facing an open bet, else Check is).
- All amounts `≤ stack + ownContribution` (no moves bigger than stack).

**Files:** modify dealer + test.

- [ ] **Step 1: Add tests**

```scala
  test("legalActionsFor: facing no bet → menu has Fold, Check, Raise(halfPot), Raise(pot), Raise(allIn)"):
    val cfg = TableConfig(6, 1L, 2L, 0L, 200L)
    val d = AcpcTableDealer(cfg, SeatId(0), 1L)
    d.postBlinds()
    d.dealHoleCards()
    d.startStreet(sicfun.holdem.types.Street.Flop) // no outstanding bet
    val menu = d.legalActionsFor(SeatId(0))
    assert(menu.contains(sicfun.holdem.types.PokerAction.Fold))
    assert(menu.contains(sicfun.holdem.types.PokerAction.Check))
    val raises = menu.collect { case r: sicfun.holdem.types.PokerAction.Raise => r }
    // Dedup invariant: all raise amounts unique.
    assertEquals(raises.map(_.amount).size, raises.size)
    assert(raises.nonEmpty, "post-flop raises should be available from non-empty pot")

  test("legalActionsFor: stack short enough to collapse HalfPot/Pot/AllIn into one Raise"):
    // Set up a spot where pot = 10, stack = 4 → HalfPot=5, Pot=10, AllIn=4.
    // Pot and HalfPot both exceed stack, so both get clamped to AllIn (4).
    // Expected: only one Raise(4) in the menu.
    val cfg = TableConfig(2, 1L, 2L, 0L, 20L)
    val d = AcpcTableDealer(cfg, SeatId(0), 1L)
    d.postBlinds()
    d.dealHoleCards()
    d.startStreet(sicfun.holdem.types.Street.Preflop)
    d.applyAction(SeatId(0), sicfun.holdem.types.PokerAction.Raise(16.0))
      .getOrElse(fail("aggressive open"))
    val menu = d.legalActionsFor(SeatId(1))
    val raises = menu.collect { case r: sicfun.holdem.types.PokerAction.Raise => r }
    assertEquals(raises.map(_.amount).toSet.size, raises.size,
      s"dedup failed: ${raises.map(_.amount)}")
```

- [ ] **Step 2: Implement**

```scala
// append to AcpcTableDealer

def legalActionsFor(seat: SeatId): Set[PokerAction] =
  val owed = currentBet - streetContribution(seat)
  val stack = stacks(seat)
  val contribution = streetContribution(seat)
  val totalPot = contributions.values.sum + streetContribution.values.sum
  val builder = scala.collection.mutable.Set[PokerAction]()

  // Always legal: Fold (surrender) when facing a bet; Check when not.
  if owed > 0L then builder += PokerAction.Fold
  else builder += PokerAction.Check

  // Passive: Call when facing a bet.
  if owed > 0L && stack > 0L then builder += PokerAction.Call

  // Raises: only if stack allows at least a min-raise of 1 over currentBet.
  val maxTotalBet = stack + contribution
  if maxTotalBet > currentBet then
    val halfPot = (totalPot / 2).max(currentBet + 1L)
    val pot = totalPot.max(currentBet + 1L)
    val allIn = maxTotalBet
    Seq(halfPot, pot, allIn)
      .map(_.min(maxTotalBet))                 // cap at all-in
      .filter(_ > currentBet)                   // must actually raise
      .map(amt => PokerAction.Raise(amt.toDouble))
      .foreach(builder += _)

  builder.toSet
```

- [ ] **Step 3: Run tests** — both pass; dedup invariant via `Set` + `.toSet` on amounts.

- [ ] **Step 4: Commit**

```bash
git commit -am "feat(runtime): legalActionsFor with 5-action discrete menu and dedup"
```

---

### Task 7: Illegal action rejection tests

**Files:** test-only.

- [ ] **Step 1: Add tests**

```scala
  test("applyAction: Check when owed > 0 → IllegalForm"):
    val cfg = TableConfig(3, 1L, 2L, 0L, 200L)
    val d = AcpcTableDealer(cfg, SeatId(0), 1L)
    d.postBlinds()
    d.startStreet(sicfun.holdem.types.Street.Preflop)
    val result = d.applyAction(SeatId(0), sicfun.holdem.types.PokerAction.Check)
    assert(result.isLeft)
    result.left.foreach {
      case IllegalActionReason.IllegalForm(_, _, _) => ()
      case other => fail(s"expected IllegalForm, got $other")
    }

  test("applyAction: Raise below currentBet → IllegalForm"):
    val cfg = TableConfig(3, 1L, 2L, 0L, 200L)
    val d = AcpcTableDealer(cfg, SeatId(0), 1L)
    d.postBlinds()
    d.startStreet(sicfun.holdem.types.Street.Preflop)
    val result = d.applyAction(SeatId(0), sicfun.holdem.types.PokerAction.Raise(1.5))
    assert(result.isLeft)

  test("applyAction: Raise more than stack → InsufficientChips"):
    val cfg = TableConfig(3, 1L, 2L, 0L, 20L)
    val d = AcpcTableDealer(cfg, SeatId(0), 1L)
    d.postBlinds()
    d.startStreet(sicfun.holdem.types.Street.Preflop)
    val result = d.applyAction(SeatId(0), sicfun.holdem.types.PokerAction.Raise(1000.0))
    assert(result.isLeft)
    result.left.foreach {
      case IllegalActionReason.InsufficientChips(_, _, _) => ()
      case other => fail(s"expected InsufficientChips, got $other")
    }

  test("applyAction: out-of-turn → NotYourTurn"):
    val cfg = TableConfig(3, 1L, 2L, 0L, 200L)
    val d = AcpcTableDealer(cfg, SeatId(0), 1L)
    d.postBlinds()
    d.startStreet(sicfun.holdem.types.Street.Preflop)
    val wrong = if d.nextToAct == Some(SeatId(0)) then SeatId(2) else SeatId(0)
    val result = d.applyAction(wrong, sicfun.holdem.types.PokerAction.Fold)
    assert(result.isLeft)
```

- [ ] **Step 2: Run** — all 4 pass (no implementation changes needed; Task 5 already wired these errors).

- [ ] **Step 3: Commit**

```bash
git commit -am "test(runtime): illegal action rejection coverage"
```

---

## Phase 2.5 — Hand resolution (real, not stubs)

### Task 8: Event log buffer

**Design:** `eventBuffer` is already maintained by `applyAction` (Task 5). This task only exposes it and adds `Deal` + `PostBlind` events at the right points.

**Files:** modify dealer + test.

- [ ] **Step 1: Add tests**

```scala
  test("eventLog records PostBlind, Deal, Act in order"):
    val cfg = TableConfig(3, 1L, 2L, 0L, 200L)
    val d = AcpcTableDealer(cfg, SeatId(0), 1L)
    val blindEvents = d.postBlinds()
    d.dealHoleCards()
    d.startStreet(sicfun.holdem.types.Street.Preflop)
    d.applyAction(SeatId(0), sicfun.holdem.types.PokerAction.Fold)
    val log = d.eventLog
    assertEquals(log.take(2).collect { case e: BettingRoundEvent.PostBlind => e }.size, 2,
      s"expected 2 PostBlind events, got ${log.take(2)}")
    assert(log.exists(_.isInstanceOf[BettingRoundEvent.Act]))
```

- [ ] **Step 2: Wire PostBlind + Deal into eventBuffer**

Modify `postBlinds` and `dealCommunity`:

```scala
def postBlinds(): Vector[BettingRoundEvent] =
  val sb = smallBlindSeat
  val bb = bigBlindSeat
  stacks(sb) -= config.smallBlind
  stacks(bb) -= config.bigBlind
  contributions(sb) = contributions(sb) + config.smallBlind
  contributions(bb) = contributions(bb) + config.bigBlind
  val events = Vector(
    BettingRoundEvent.PostBlind(sb, config.smallBlind, BlindKind.SmallBlind),
    BettingRoundEvent.PostBlind(bb, config.bigBlind, BlindKind.BigBlind)
  )
  events.foreach(eventBuffer += _)
  events

def dealCommunity(street: Street): Vector[Card] =
  val count = street match
    case Street.Preflop => 0
    case Street.Flop    => 3
    case Street.Turn    => 1
    case Street.River   => 1
  val dealtThisStreet = (1 to count).map { _ => deckBuf.remove(0) }.toVector
  dealtThisStreet.foreach(boardBuf += _)
  if count > 0 then eventBuffer += BettingRoundEvent.Deal(street, dealtThisStreet)
  boardBuf.toVector

def eventLog: Vector[BettingRoundEvent] = eventBuffer.toVector
```

- [ ] **Step 3: Run tests** — pass.

- [ ] **Step 4: Commit**

```bash
git commit -am "feat(runtime): real eventLog buffer with PostBlind, Deal, Act"
```

---

### Task 9: Hand evaluation (7-card best hand per seat)

**Uses:** `sicfun.core.HandEvaluator.evaluate7` (returns `HandRank` with total ordering).

**Design:** `evaluateShowdown()` takes the full 5-card board + each non-folded seat's 2 hole cards, computes `HandRank` per seat, returns `Map[SeatId, HandRank]`. Ties are expressed by equal `HandRank.compare == 0`.

**Files:** modify dealer + test.

- [ ] **Step 1: Add test**

```scala
  test("evaluateShowdown: only non-folded seats get a HandRank"):
    val cfg = TableConfig(3, 1L, 2L, 0L, 200L)
    val d = AcpcTableDealer(cfg, SeatId(0), 42L)
    d.postBlinds()
    d.dealHoleCards()
    d.startStreet(sicfun.holdem.types.Street.Preflop)
    d.applyAction(SeatId(0), sicfun.holdem.types.PokerAction.Fold)
    d.applyAction(SeatId(1), sicfun.holdem.types.PokerAction.Call)
    d.applyAction(SeatId(2), sicfun.holdem.types.PokerAction.Check)
    d.dealCommunity(sicfun.holdem.types.Street.Flop)
    d.dealCommunity(sicfun.holdem.types.Street.Turn)
    d.dealCommunity(sicfun.holdem.types.Street.River)
    val ranks = d.evaluateShowdown()
    assertEquals(ranks.keySet, Set(SeatId(1), SeatId(2)))
```

- [ ] **Step 2: Implement**

```scala
// append to AcpcTableDealer
import sicfun.core.HandEvaluator
import sicfun.core.HandRank

def evaluateShowdown(): Map[SeatId, HandRank] =
  require(boardBuf.size == 5, s"showdown requires full board, got ${boardBuf.size}")
  val nonFolded = (0 until config.numSeats).map(SeatId(_)).filterNot(folded.contains)
  nonFolded.map { seat =>
    val seven = hole(seat) ++ boardBuf.toVector
    seat -> HandEvaluator.evaluate7(seven)
  }.toMap
```

- [ ] **Step 3: Run** — pass.

- [ ] **Step 4: Commit**

```bash
git commit -am "feat(runtime): showdown evaluation via HandEvaluator.evaluate7"
```

---

### Task 10: Side-pot distribution (multi-level with tie-breaking + odd-chip convention)

**Algorithm (canonical):**
1. Compute side pots by contribution levels (Task from v1 logic, but fold-aware).
2. For each side pot: among eligible (non-folded) seats, find the subset with the best `HandRank`.
3. Split pot amount equally among winners; any odd-chip remainder goes to the winner closest clockwise from the button (odd-chip convention).

**Files:** modify dealer + tests.

- [ ] **Step 1: Add tests (including ties + multi-way all-in)**

```scala
  test("distributePots: 3-way equal contributions, single winner takes whole pot"):
    val cfg = TableConfig(3, 1L, 2L, 0L, 200L)
    val d = AcpcTableDealer(cfg, SeatId(0), 42L)
    // Scripted scenario: all three contribute 10 each → pot 30.
    d.setStateForTest(
      stacks = Map(SeatId(0) -> 190L, SeatId(1) -> 190L, SeatId(2) -> 190L),
      contributions = Map(SeatId(0) -> 10L, SeatId(1) -> 10L, SeatId(2) -> 10L),
      folded = Set.empty
    )
    val ranks = Map(
      SeatId(0) -> HandEvaluator.evaluate7(d.scriptedSevenForTest(SeatId(0))),
      // ... (use deterministic hole+board that gives seat 1 a made flush vs seat 0 one pair)
    )
    val distributed = d.distributePots(ranks)
    val total = distributed.map(_._2.values.sum).sum
    assertEquals(total, 30L)

  test("distributePots: exact split on 3-way tie, odd chip goes to first winner clockwise from button"):
    // Pot = 10, 3 winners tied. 10/3 = 3 each, remainder 1 → goes to seat closest clockwise from button.
    val cfg = TableConfig(3, 1L, 2L, 0L, 200L)
    val d = AcpcTableDealer(cfg, SeatId(2), 42L) // button = 2, so clockwise order from button is 0, 1, 2
    d.setStateForTest(
      stacks = Map(SeatId(0) -> 196L, SeatId(1) -> 196L, SeatId(2) -> 196L),
      contributions = Map(SeatId(0) -> 4L, SeatId(1) -> 3L, SeatId(2) -> 3L),
      folded = Set.empty
    )
    val tiedRank = HandEvaluator.evaluate7(d.scriptedSevenForTest(SeatId(0)))
    val ranks = Map(SeatId(0) -> tiedRank, SeatId(1) -> tiedRank, SeatId(2) -> tiedRank)
    val distributed = d.distributePots(ranks)
    val flat = distributed.flatMap(_._2).groupMapReduce(_._1)(_._2)(_ + _)
    assertEquals(flat.values.sum, 10L, "conservation on tied pot")
    // Odd chip goes to seat 0 (first clockwise from button=2).
    assert(flat(SeatId(0)) >= flat(SeatId(1)), s"odd chip convention violated: $flat")
    assert(flat(SeatId(0)) >= flat(SeatId(2)))

  test("distributePots: multi-way all-in creates side pots, each awarded independently"):
    // Seat 0: stack 5, contributes 5
    // Seat 1: stack 10, contributes 10
    // Seat 2: stack 20, contributes 20
    // Main pot: 5*3 = 15 (all eligible), side 1: (10-5)*2 = 10 (seats 1,2), side 2: (20-10)*1 = 10 (seat 2 only)
    val cfg = TableConfig(3, 1L, 2L, 0L, 200L)
    val d = AcpcTableDealer(cfg, SeatId(0), 42L)
    d.setStateForTest(
      stacks = Map(SeatId(0) -> 0L, SeatId(1) -> 0L, SeatId(2) -> 0L),
      contributions = Map(SeatId(0) -> 5L, SeatId(1) -> 10L, SeatId(2) -> 20L),
      folded = Set.empty
    )
    val ranks = d.scriptedRanksSeat2WinsAll // seat 2 wins everything
    val distributed = d.distributePots(ranks)
    val flat = distributed.flatMap(_._2).groupMapReduce(_._1)(_._2)(_ + _)
    assertEquals(flat.getOrElse(SeatId(2), 0L), 35L)
    assertEquals(flat.getOrElse(SeatId(0), 0L), 0L)
    assertEquals(flat.getOrElse(SeatId(1), 0L), 0L)
    assertEquals(flat.values.sum, 35L)
```

*Helpers `setStateForTest` / `scriptedSevenForTest` / `scriptedRanksSeat2WinsAll` are test-only shims — add them as `private[protocol]` methods on the dealer so tests can script exact scenarios without depending on shuffle order.*

- [ ] **Step 2: Implement `distributePots` + helpers**

```scala
// append to AcpcTableDealer

/** Build side pots from `contributions` (fold-aware).
  *
  * Each pot has an `amount` and `eligibleSeats` (non-folded contributors at
  * that level and above). Folded players' contributions stay in the pot but
  * cannot win.
  */
def computeSidePots(): Vector[SidePot] =
  val contribPairs = (0 until config.numSeats).map(SeatId(_))
    .map(s => s -> contributions(s))
    .filter(_._2 > 0L)
    .sortBy(_._2)
    .toVector

  var result = Vector.empty[SidePot]
  var prevLevel = 0L
  var remaining = contribPairs

  while remaining.nonEmpty do
    val level = remaining.head._2
    val delta = level - prevLevel
    val potAmount = delta * remaining.size
    val eligible = remaining.map(_._1).toSet -- folded.toSet
    if potAmount > 0L && eligible.nonEmpty then
      result = result :+ SidePot(potAmount, eligible)
    prevLevel = level
    remaining = remaining.filter(_._2 > level)

  result

/** Distribute each pot among eligible winners (highest HandRank). Odd chips
  * go to the first winner clockwise from the button. */
def distributePots(
    ranks: Map[SeatId, HandRank]
): Vector[(SidePot, Map[SeatId, Long])] =
  val pots = computeSidePots()
  pots.map { pot =>
    val contenders = pot.eligibleSeats.toVector.filter(ranks.contains)
    if contenders.isEmpty then pot -> Map.empty[SeatId, Long]
    else
      val bestRank = contenders.map(ranks).max(summon[Ordering[HandRank]])
      val winners = contenders.filter(s => ranks(s).compare(bestRank) == 0)
      val share = pot.amount / winners.size
      val remainder = pot.amount - share * winners.size
      val winnerOrderFromButton = clockwiseFromButton(winners.toSet)
      val baseDist = winners.map(_ -> share).toMap
      val oddChipSeat = winnerOrderFromButton.headOption
      val finalDist = oddChipSeat match
        case Some(s) if remainder > 0 =>
          baseDist.updated(s, baseDist(s) + remainder)
        case _ => baseDist
      pot -> finalDist
  }

private def clockwiseFromButton(seats: Set[SeatId]): Vector[SeatId] =
  (1 to config.numSeats).map { off =>
    SeatId((_buttonSeat.index + off) % config.numSeats)
  }.filter(seats.contains).toVector

// Test-only shims
private[protocol] def setStateForTest(
    stacks: Map[SeatId, Long],
    contributions: Map[SeatId, Long],
    folded: Set[SeatId]
): Unit =
  stacks.foreach { case (s, v) => this.stacks(s) = v }
  contributions.foreach { case (s, v) => this.contributions(s) = v }
  folded.foreach(this.folded += _)

private[protocol] def scriptedSevenForTest(seat: SeatId): Vector[Card] =
  Deck.full.take(7).toVector

private[protocol] def scriptedRanksSeat2WinsAll: Map[SeatId, HandRank] =
  val loser = HandEvaluator.evaluate7(Deck.full.take(7).toVector)
  val winner = HandEvaluator.evaluate7(Deck.full.drop(7).take(7).toVector)
  // If drop-7-take-7 happens to tie, swap deterministically.
  val (w, l) = if winner.compare(loser) > 0 then (winner, loser) else (loser, winner)
  Map(SeatId(0) -> l, SeatId(1) -> l, SeatId(2) -> w)
```

- [ ] **Step 3: Run** — all 3 scenario tests pass.

- [ ] **Step 4: Commit**

```bash
git commit -am "feat(runtime): side-pot distribution with tie-breaking and odd-chip convention"
```

---

### Task 11: Property tests — chip conservation + multi-way all-in

**Files:**
- Create: `src/test/scala/sicfun/holdem/runtime/protocol/AcpcTableDealerPropertyTest.scala`

- [ ] **Step 1: Property test**

```scala
package sicfun.holdem.runtime.protocol

import org.scalacheck.Gen
import org.scalacheck.Prop.*
import sicfun.holdem.types.{PokerAction, Street}

class AcpcTableDealerPropertyTest extends munit.ScalaCheckSuite:

  val nSeatsGen: Gen[Int] = Gen.choose(2, 9)
  val stackGen: Gen[Long] = Gen.choose(20L, 500L)

  property("side pots: sum of distributed chips == sum of contributions"):
    forAll(nSeatsGen, Gen.listOfN(9, stackGen)) { (n, stacks) =>
      val cfg = TableConfig(n, 1L, 2L, 0L, 200L)
      val d = AcpcTableDealer(cfg, SeatId(0), 0L)
      (0 until n).foreach(i => d.stacksForTest(SeatId(i)) = stacks(i))
      d.postBlinds()
      d.dealHoleCards()
      d.startStreet(Street.Preflop)
      // Script: every seat shoves in rotation; dealer short-circuits on all-in.
      var safety = 0
      while d.nextToAct.isDefined && safety < 50 do
        val seat = d.nextToAct.get
        val owed = d.currentBet - d.streetContributionForTest(seat)
        val shove = d.currentStacks(seat) + d.currentContributions(seat)
        if shove > d.currentBet then
          d.applyAction(seat, PokerAction.Raise(shove.toDouble))
        else
          d.applyAction(seat, PokerAction.Fold)
        safety += 1
      d.dealCommunity(Street.Flop)
      d.dealCommunity(Street.Turn)
      d.dealCommunity(Street.River)
      val ranks = d.evaluateShowdown()
      val distributed = d.distributePots(ranks)
      val totalDistributed = distributed.map(_._2.values.sum).sum
      val totalContributed = d.currentContributions.values.sum
      totalDistributed == totalContributed
    }

  property("finalizeHand: outcome.netChange sums to zero exactly"):
    forAll(nSeatsGen) { n =>
      val cfg = TableConfig(n, 1L, 2L, 0L, 200L)
      val d = AcpcTableDealer(cfg, SeatId(0), 123L)
      d.postBlinds()
      d.dealHoleCards()
      d.startStreet(Street.Preflop)
      // Everyone folds except BB.
      var safety = 0
      while d.nextToAct.isDefined && safety < 50 do
        d.applyAction(d.nextToAct.get, PokerAction.Fold)
        safety += 1
      val outcome = d.finalizeHand()
      outcome.netChange.values.sum == 0L
    }
```

*Add `stacksForTest` / `streetContributionForTest` as test-only accessors on the dealer (one line each).*

- [ ] **Step 2: Implement `finalizeHand` for real**

```scala
// append to AcpcTableDealer

def handEnded: Boolean =
  val activeNonFolded = (0 until config.numSeats).map(SeatId(_))
    .count(s => !folded.contains(s))
  activeNonFolded <= 1 ||
    (currentStreet == Street.River && roundClosed)

/** Finalize the hand: if only one non-folded seat, they take the pot;
  * else compute showdown ranks and distribute. Returns `HandOutcome` with
  * `netChange.values.sum == 0` exactly. */
def finalizeHand(): HandOutcome =
  val pots = computeSidePots()
  val ranks: Map[SeatId, HandRank] =
    val nonFolded = (0 until config.numSeats).map(SeatId(_))
      .filterNot(folded.contains)
      .toVector
    if nonFolded.size == 1 then
      // Unopposed: assign the sole seat the highest possible rank so they win every pot.
      val winner = nonFolded.head
      pots.flatMap(_.eligibleSeats).toSet.map { s =>
        s -> (if s == winner then topRank else bottomRank)
      }.toMap
    else if boardBuf.size == 5 then
      evaluateShowdown()
    else
      // Hand ended pre-river with multiple non-folded seats (all-in scenario):
      // deal remaining community cards automatically before showdown.
      if boardBuf.size < 3 then dealCommunity(Street.Flop)
      if boardBuf.size < 4 then dealCommunity(Street.Turn)
      if boardBuf.size < 5 then dealCommunity(Street.River)
      evaluateShowdown()

  val distributed = distributePots(ranks)
  val distributedByseat: Map[SeatId, Long] =
    distributed.flatMap(_._2).groupMapReduce(_._1)(_._2)(_ + _)
  val netChange: Map[SeatId, Long] =
    (0 until config.numSeats).map { i =>
      val s = SeatId(i)
      s -> (distributedByseat.getOrElse(s, 0L) - contributions(s))
    }.toMap

  distributed.foreach { case (pot, dist) =>
    eventBuffer += BettingRoundEvent.PotAwarded(pot, dist)
  }

  HandOutcome(distributed, netChange, eventBuffer.toVector)

private def topRank: HandRank =
  // Use a deterministic top-value handrank; simplest: evaluate royal flush-ish cards.
  HandEvaluator.evaluate7(Deck.full.takeRight(7).toVector)
private def bottomRank: HandRank =
  HandEvaluator.evaluate7(Deck.full.take(7).toVector)
```

*Note: `topRank` and `bottomRank` are deterministic sentinels for the unopposed case; ordering is strict because we compare `ranks.max` against concrete ranks, but since `winner` vs `non-winner` gets distinct ranks, the winner always wins.*

- [ ] **Step 3: Run properties** — both hold across 100 generated cases.

- [ ] **Step 4: Commit**

```bash
git commit -am "feat(runtime): finalizeHand with real pot distribution + property tests on conservation"
```

---

## Phase 3 — NativeStrictMode (2 tasks)

### Task 12: Extract canonical DLL list from `build-windows-cuda11.ps1`

**Files:**
- Create: `src/main/scala/sicfun/holdem/runtime/NativeStrictMode.scala` (list constants only, verify step)

- [ ] **Step 1: Read the build script and extract `.dll` outputs**

Run once (manually, at plan-execution time):

```bash
grep -nE 'Join-Path \$OutDir "sicfun_[a-z_]+\.dll"' src/main/native/build-windows-cuda11.ps1
```

Expected canonical set (verified on commit fd72c1b, line numbers in the script):
- `sicfun_gpu_kernel.dll` (line 8, `$DllName` default)
- `sicfun_cfr_cuda.dll` (line 665)
- `sicfun_bayes_cuda.dll` (line 696)
- `sicfun_ddre_cuda.dll` (line 727)
- `sicfun_postflop_cuda.dll` (line 758)

CPU-variant DLLs (`sicfun_native_cpu.dll`, `sicfun_bayes_native.dll`, `sicfun_ddre_native.dll`, `sicfun_postflop_native.dll`, `sicfun_pomcp_native.dll`) are built by sibling CPU-only scripts — **verify existence at step time** by listing `src/main/native/` for any additional `build-*.ps1` / `build-*.sh` scripts that produce `.dll` / `.so` / `.dylib` outputs. Add them to the list only if the corresponding script is present.

If any of the 5 CUDA DLLs is absent from the build script as of plan-execution time (e.g., the script was refactored), update this list before proceeding and commit the update as its own change.

- [ ] **Step 2: Define `CoreLibraries`**

```scala
package sicfun.holdem.runtime

import java.io.File

final case class NativeStrictModeViolation(message: String)
    extends RuntimeException(message)

final case class NativeReady(
    loaded: Vector[String],
    contaminated: Boolean
)

object NativeStrictMode:

  /** One library = one DLL, plus the source files whose mtime gates staleness. */
  final case class Library(name: String, sources: Vector[String])

  /** Canonical outputs of `src/main/native/build-windows-cuda11.ps1` as of the
    * plan's reference commit. Re-verify via the grep in Task 12 Step 1 if the
    * build script changes. */
  val CoreLibraries: Vector[Library] = Vector(
    Library("sicfun_gpu_kernel", Vector(
      "src/main/native/jni/HeadsUpGpuNativeBindings.cpp",
      "src/main/native/jni/HeadsUpGpuNativeBindingsCuda.cu"
    )),
    Library("sicfun_cfr_cuda", Vector(
      "src/main/native/jni/HoldemCfrNativeGpuBindings.cu"
    )),
    Library("sicfun_bayes_cuda", Vector(
      "src/main/native/jni/HoldemBayesNativeGpuBindings.cu",
      "src/main/native/jni/BayesNativeUpdateCore.hpp"
    )),
    Library("sicfun_ddre_cuda", Vector(
      "src/main/native/jni/HoldemDdreNativeGpuBindings.cu",
      "src/main/native/jni/DdreNativeInferenceCore.hpp"
    )),
    Library("sicfun_postflop_cuda", Vector(
      "src/main/native/jni/HoldemPostflopNativeBindingsCuda.cu"
    ))
  )
```

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(runtime): canonical DLL list extracted from build-windows-cuda11.ps1"
```

---

### Task 13: NativeStrictMode.verify + tests

**Files:**
- Modify: `NativeStrictMode.scala`
- Create: `src/test/scala/sicfun/holdem/runtime/NativeStrictModeTest.scala`

- [ ] **Step 1: Add tests**

```scala
package sicfun.holdem.runtime

import java.io.File
import java.nio.file.Files

class NativeStrictModeTest extends munit.FunSuite:

  test("strict: missing DLL → Left(violation) listing the missing name"):
    val tmp = Files.createTempDirectory("sicfun-empty-native").toFile
    val required = Vector(NativeStrictMode.Library("sicfun_nonexistent", Vector.empty))
    val result = NativeStrictMode.verify(tmp, required, strict = true)
    assert(result.isLeft)
    result.left.foreach { v =>
      assert(v.getMessage.contains("sicfun_nonexistent"), v.getMessage)
    }

  test("strict: missing multiple DLLs → violation message enumerates all"):
    val tmp = Files.createTempDirectory("sicfun-empty-native-2").toFile
    val required = Vector(
      NativeStrictMode.Library("lib_a", Vector.empty),
      NativeStrictMode.Library("lib_b", Vector.empty)
    )
    val result = NativeStrictMode.verify(tmp, required, strict = true)
    assert(result.isLeft)
    result.left.foreach { v =>
      assert(v.getMessage.contains("lib_a") && v.getMessage.contains("lib_b"),
        s"expected both in message, got: ${v.getMessage}")
    }

  test("non-strict: missing DLL → Right(contaminated = true) with stderr warning"):
    val tmp = Files.createTempDirectory("sicfun-empty-native-3").toFile
    val required = Vector(NativeStrictMode.Library("sicfun_nonexistent", Vector.empty))
    val result = NativeStrictMode.verify(tmp, required, strict = false)
    assert(result.isRight)
    assertEquals(result.toOption.get.contaminated, true)

  test("strict: present DLL + no stale sources → Right(contaminated = false)"):
    val tmp = Files.createTempDirectory("sicfun-fake-build").toFile
    val dllFile = new File(tmp, "lib_fake.dll")
    Files.write(dllFile.toPath, Array[Byte](0))
    val required = Vector(NativeStrictMode.Library("lib_fake", Vector.empty))
    val result = NativeStrictMode.verify(tmp, required, strict = true)
    assert(result.isRight, s"expected Right, got $result")
    assertEquals(result.toOption.get.contaminated, false)

  test("strict: DLL older than source → Left(stale)"):
    val tmp = Files.createTempDirectory("sicfun-stale").toFile
    val dllFile = new File(tmp, "lib_fake.dll")
    Files.write(dllFile.toPath, Array[Byte](0))
    dllFile.setLastModified(1_000_000L) // 1970-ish
    val srcTmp = Files.createTempFile("sicfun-src", ".cpp").toFile
    srcTmp.setLastModified(System.currentTimeMillis()) // now
    val required = Vector(NativeStrictMode.Library("lib_fake", Vector(srcTmp.getPath)))
    val result = NativeStrictMode.verify(tmp, required, strict = true)
    assert(result.isLeft, s"expected stale violation, got $result")
    result.left.foreach { v =>
      assert(v.getMessage.contains("stale") || v.getMessage.contains("older"),
        v.getMessage)
    }
```

- [ ] **Step 2: Implement `verify`**

```scala
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
          .map(p => new File(p))
          .find(src => src.exists() && src.lastModified() > dllMtime)
          .map(src => (lib.name, src.getPath))
    }
    (missing, stale, strict) match
      case (m, _, true) if m.nonEmpty =>
        Left(NativeStrictModeViolation(
          s"Missing native libraries (strict): ${m.map(_.name).mkString(", ")}; " +
            "rebuild with src/main/native/build-windows-cuda11.ps1"
        ))
      case (_, s, true) if s.nonEmpty =>
        Left(NativeStrictModeViolation(
          s"Stale native builds (strict): ${s.map((d, src) => s"$d older than $src").mkString("; ")}"
        ))
      case (m, s, false) if m.nonEmpty || s.nonEmpty =>
        System.err.println(
          s"[BENCHMARK-CONTAMINATED] strict=false, missing=[${m.map(_.name).mkString(",")}], " +
            s"stale=[${s.map(_._1).mkString(",")}]"
        )
        Right(NativeReady(Vector.empty, contaminated = true))
      case _ =>
        Right(NativeReady(required.map(_.name), contaminated = false))

  private def dllFile(dir: File, libName: String): File =
    val os = System.getProperty("os.name", "").toLowerCase
    if os.contains("win") then new File(dir, s"$libName.dll")
    else if os.contains("mac") then new File(dir, s"lib$libName.dylib")
    else new File(dir, s"lib$libName.so")
```

- [ ] **Step 3: Run** — all 5 pass.

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(runtime): NativeStrictMode.verify with missing + stale detection"
```

---

## Phase 4 — Agent API + Blueprint (4 tasks)

### Task 14: SeatAgent trait + MatchSummary

**Files:**
- Create: `src/main/scala/sicfun/holdem/runtime/agent/SeatAgent.scala`

- [ ] **Step 1: Implement (pure interface, no tests)**

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

### Task 15: BlueprintFormat types (`numAbstractActions = 5`)

**Files:**
- Create: `src/main/scala/sicfun/holdem/runtime/agent/BlueprintFormat.scala`
- Create: `src/test/scala/sicfun/holdem/runtime/agent/BlueprintFormatTest.scala`

**Abstract action index contract (A.1):** `0=Fold, 1=Passive, 2=HalfPot, 3=Pot, 4=AllIn`. Numeric indices stay stable through A.2; adding actions bumps `BlueprintHeader.version`.

- [ ] **Step 1: Tests**

```scala
package sicfun.holdem.runtime.agent

class BlueprintFormatTest extends munit.FunSuite:
  test("BlueprintHeader: magic must be SICFBP01"):
    val ok = BlueprintHeader("SICFBP01", 1, 0L, "h", 9, 0, 5)
    intercept[IllegalArgumentException](ok.copy(magic = "WRONG"))

  test("BlueprintHeader: numSeats in [2,9], numAbstractActions == 5 for A.1"):
    def mk(n: Int, a: Int) = BlueprintHeader("SICFBP01", 1, 0L, "h", n, 0, a)
    mk(2, 5); mk(9, 5)
    intercept[IllegalArgumentException](mk(1, 5))
    intercept[IllegalArgumentException](mk(10, 5))
    intercept[IllegalArgumentException](mk(6, 0))

  test("AbstractActionDistribution: probs sum to 1.0 ± 1e-3, non-empty"):
    AbstractActionDistribution(Array(0.2f, 0.2f, 0.2f, 0.2f, 0.2f))
    intercept[IllegalArgumentException](AbstractActionDistribution(Array.empty[Float]))
    intercept[IllegalArgumentException](AbstractActionDistribution(Array(0.5f, 0.3f))) // sum = 0.8
```

- [ ] **Step 2: Implement**

```scala
package sicfun.holdem.runtime.agent

/** Abstract action index contract (A.1):
  *   0 = Fold, 1 = Passive (Check or Call), 2 = HalfPot, 3 = Pot, 4 = AllIn.
  *
  * A.2 may expand this space by bumping `version`; `numAbstractActions` in
  * the header records the current width for forward-compat reads. */
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

final case class AbstractActionDistribution(probs: Array[Float]):
  require(probs.nonEmpty, "probs must be non-empty")
  require(
    math.abs(probs.sum - 1.0f) < 1e-3f,
    s"probs must sum to 1.0 ± 1e-3, got ${probs.sum}"
  )
```

- [ ] **Step 3: Run** — pass.

- [ ] **Step 4: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/agent/BlueprintFormat.scala src/test/scala/sicfun/holdem/runtime/agent/BlueprintFormatTest.scala
git commit -m "feat(agent): BlueprintFormat with 5-action abstraction (A.1)"
```

---

### Task 16: BlueprintStore load/write with hash validation

**Files:**
- Create: `src/main/scala/sicfun/holdem/runtime/agent/BlueprintStore.scala`
- Create: `src/test/scala/sicfun/holdem/runtime/agent/BlueprintStoreTest.scala`

- [ ] **Step 1: Tests**

```scala
package sicfun.holdem.runtime.agent

import java.nio.file.Files

class BlueprintStoreTest extends munit.FunSuite:
  test("round-trip: write then load returns same distribution"):
    val tmp = Files.createTempFile("bp", ".blueprint").toFile
    val hdr = BlueprintHeader("SICFBP01", 1, 0L, "specA", 9, 2, 5)
    val dists = Map[Long, Array[Float]](
      100L -> Array(0.2f, 0.2f, 0.2f, 0.2f, 0.2f),
      200L -> Array(0.1f, 0.1f, 0.1f, 0.1f, 0.6f)
    )
    BlueprintStore.write(tmp, hdr, dists)
    val store = BlueprintStore.load(tmp, expectedAbstractionHash = "specA")
    assertEquals(store.header.numInfostates, 2)
    val d100 = store.lookup(100L)
    assert(d100.probs(4) > 0.19f && d100.probs(4) < 0.21f)
    val d200 = store.lookup(200L)
    assert(d200.probs(4) > 0.59f)

  test("rejects wrong abstraction hash"):
    val tmp = Files.createTempFile("bp2", ".blueprint").toFile
    val hdr = BlueprintHeader("SICFBP01", 1, 0L, "specA", 9, 0, 5)
    BlueprintStore.write(tmp, hdr, Map.empty)
    intercept[BlueprintVersionMismatch](
      BlueprintStore.load(tmp, expectedAbstractionHash = "specB")
    )

  test("empty blueprint: lookup returns uniform distribution"):
    val tmp = Files.createTempFile("bp3", ".blueprint").toFile
    val hdr = BlueprintHeader("SICFBP01", 1, 0L, "empty", 9, 0, 5)
    BlueprintStore.write(tmp, hdr, Map.empty)
    val store = BlueprintStore.load(tmp, "empty")
    val uniform = store.lookup(12345L)
    assert(uniform.probs.forall(p => math.abs(p - 0.2f) < 1e-4f),
      s"uniform fallback not uniform: ${uniform.probs.mkString(",")}")
```

- [ ] **Step 2: Implement**

```scala
package sicfun.holdem.runtime.agent

import java.io.{DataInputStream, DataOutputStream, File, FileInputStream, FileOutputStream}

final case class BlueprintNotFoundError(path: String)
    extends RuntimeException(s"blueprint not found: $path")
final case class BlueprintVersionMismatch(expected: String, actual: String)
    extends RuntimeException(s"abstraction hash mismatch: expected=$expected got=$actual")

final class BlueprintStore private (
    val header: BlueprintHeader,
    private val rows: Map[Long, Array[Float]]
):
  def lookup(infostateHash: Long): AbstractActionDistribution =
    rows.get(infostateHash) match
      case Some(arr) => AbstractActionDistribution(arr)
      case None =>
        AbstractActionDistribution(
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
      rows.toVector.sortBy(_._1).foreach { case (k, arr) =>
        out.writeLong(k)
        require(arr.length == header.numAbstractActions,
          s"row length ${arr.length} != numAbstractActions ${header.numAbstractActions}")
        arr.foreach(out.writeFloat)
      }
    finally out.close()
```

- [ ] **Step 3: Run** — all 3 pass.

- [ ] **Step 4: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/agent/BlueprintStore.scala src/test/scala/sicfun/holdem/runtime/agent/BlueprintStoreTest.scala
git commit -m "feat(agent): BlueprintStore load/write with hash validation"
```

---

### Task 17: BlueprintOnlyAgent with honest translation

**Files:**
- Create: `src/main/scala/sicfun/holdem/runtime/agent/InfostateHasher.scala` (trait + placeholder impl)
- Create: `src/main/scala/sicfun/holdem/runtime/agent/BlueprintOnlyAgent.scala`
- Create: `src/test/scala/sicfun/holdem/runtime/agent/BlueprintOnlyAgentTest.scala`

**Translation rules (A.1):**
- Abstract `Fold` → `Fold` if legal, else `Passive` (Check-when-free is the semantic fallback).
- Abstract `Passive` → `Check` if `Check ∈ legal`, else `Call` if legal, else `Fold`.
- Abstract `Raise(X)` (HalfPot/Pot/AllIn) → closest-amount `Raise(y) ∈ legal` by `|log(y/X)|`, else `Call` if legal, else `Check`, else `Fold`.

- [ ] **Step 1: Implement `InfostateHasher` trait + placeholder (with PlaceholderMarker — forward ref to Task 18 trait)**

```scala
package sicfun.holdem.runtime.agent

import sicfun.holdem.runtime.PlaceholderMarker
import sicfun.holdem.runtime.protocol.{SeatId, TableSnapshot}

trait InfostateHasher:
  def hashFor(snapshot: TableSnapshot, seat: SeatId): Long

/** Placeholder — bucketing is spurious (uses hash of visible fields). A.2
  * replaces with an abstraction-aware hash. */
final class PlaceholderInfostateHasher extends InfostateHasher, PlaceholderMarker:
  val placeholderReason: String =
    "InfostateHasher is a naive field-hash; wire abstraction-aware hash in A.2"

  def hashFor(snapshot: TableSnapshot, seat: SeatId): Long =
    snapshot.street.ordinal * 31L +
      snapshot.holeCards.map(_.hashCode().toLong).sum * 17L +
      snapshot.board.map(_.hashCode().toLong).sum * 7L +
      snapshot.contributions.values.sum * 3L +
      seat.index
```

*`PlaceholderMarker` trait is defined in Task 18; this task writes its consumer. If executed strictly top-down, add a minimal stub `trait PlaceholderMarker { def placeholderReason: String }` here and let Task 18 move it.*

- [ ] **Step 2: Write BlueprintOnlyAgent**

```scala
package sicfun.holdem.runtime.agent

import sicfun.holdem.runtime.protocol.*
import sicfun.holdem.types.PokerAction
import scala.util.Random

/** Indices in the 5-action abstraction:
  * 0 = Fold, 1 = Passive, 2 = HalfPot, 3 = Pot, 4 = AllIn.
  * Raises are represented symbolically here — translation to concrete legal
  * `Raise(amount)` happens in `translateToLegal`.
  */
enum AbstractAction:
  case Fold, Passive, HalfPot, Pot, AllIn

object AbstractAction:
  val ordered: Vector[AbstractAction] =
    Vector(Fold, Passive, HalfPot, Pot, AllIn)

final class BlueprintOnlyAgent(
    override val seatId: SeatId,
    store: BlueprintStore,
    hasher: InfostateHasher,
    rngSeed: Long
) extends SeatAgent:
  private val rng = new Random(rngSeed)

  def decide(snapshot: TableSnapshot, legalActions: Set[PokerAction]): PokerAction =
    val hash = hasher.hashFor(snapshot, seatId)
    val dist = store.lookup(hash)
    val idx = sampleIndex(dist.probs)
    val abstractPick = AbstractAction.ordered(idx)
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

  private def translateToLegal(a: AbstractAction, legal: Set[PokerAction]): PokerAction =
    a match
      case AbstractAction.Fold =>
        if legal.contains(PokerAction.Fold) then PokerAction.Fold
        else translateToLegal(AbstractAction.Passive, legal)
      case AbstractAction.Passive =>
        if legal.contains(PokerAction.Check) then PokerAction.Check
        else if legal.contains(PokerAction.Call) then PokerAction.Call
        else PokerAction.Fold
      case AbstractAction.HalfPot | AbstractAction.Pot | AbstractAction.AllIn =>
        val raises = legal.collect { case r: PokerAction.Raise => r }.toVector
        if raises.isEmpty then translateToLegal(AbstractAction.Passive, legal)
        else
          val target: Double = a match
            case AbstractAction.HalfPot => raises.map(_.amount).min
            case AbstractAction.Pot     => median(raises.map(_.amount))
            case AbstractAction.AllIn   => raises.map(_.amount).max
            case _                      => raises.head.amount
          raises.minBy(r => math.abs(math.log(r.amount) - math.log(target)))

  private def median(xs: Vector[Double]): Double =
    val sorted = xs.sorted
    sorted(sorted.size / 2)
```

- [ ] **Step 3: Tests**

```scala
package sicfun.holdem.runtime.agent

import sicfun.holdem.runtime.protocol.*
import sicfun.holdem.types.{PokerAction, Street}
import java.nio.file.Files

class BlueprintOnlyAgentTest extends munit.FunSuite:
  private def emptyStore(): BlueprintStore =
    val tmp = Files.createTempFile("bp", ".blueprint").toFile
    BlueprintStore.write(tmp, BlueprintHeader("SICFBP01", 1, 0L, "h", 6, 0, 5), Map.empty)
    BlueprintStore.load(tmp, "h")

  private def snap(): TableSnapshot = TableSnapshot(
    config = TableConfig(6, 1L, 2L, 0L, 200L),
    heroSeat = SeatId(0),
    holeCards = Vector.empty,
    board = Vector.empty,
    stacks = (0 until 6).map(i => SeatId(i) -> 200L).toMap,
    contributions = (0 until 6).map(i => SeatId(i) -> 0L).toMap,
    street = Street.Preflop,
    actionHistory = Vector.empty,
    buttonSeat = SeatId(0),
    activeSeats = (0 until 6).map(SeatId(_)).toSet
  )

  test("always returns a legal action"):
    val agent = BlueprintOnlyAgent(SeatId(0), emptyStore(), PlaceholderInfostateHasher(), 1L)
    val legal = Set[PokerAction](PokerAction.Fold, PokerAction.Call, PokerAction.Raise(6.0))
    val a = agent.decide(snap(), legal)
    assert(legal.contains(a), s"$a not in $legal")

  test("determinism under same seed"):
    val store = emptyStore()
    val a1 = BlueprintOnlyAgent(SeatId(0), store, PlaceholderInfostateHasher(), 7L)
    val a2 = BlueprintOnlyAgent(SeatId(0), store, PlaceholderInfostateHasher(), 7L)
    val legal = Set[PokerAction](PokerAction.Fold, PokerAction.Call, PokerAction.Raise(6.0))
    assertEquals(a1.decide(snap(), legal), a2.decide(snap(), legal))

  test("Fold-when-cannot-fold → Passive (Check if free, else Call)"):
    // Construct a scenario where Fold is not in legal set.
    val agent = BlueprintOnlyAgent(SeatId(0), emptyStore(), PlaceholderInfostateHasher(), 1L)
    val legal = Set[PokerAction](PokerAction.Check, PokerAction.Raise(10.0))
    // Force blueprint to pick index 0 (Fold) by using a distribution weighted at 0.
    // The agent will translate Fold → Check here.
    // We simulate by repeatedly calling decide and asserting no Fold ever appears.
    (1 to 50).foreach { _ =>
      val a = agent.decide(snap(), legal)
      assert(a != PokerAction.Fold, s"agent returned Fold when not legal: $a")
      assert(legal.contains(a), s"$a not legal")
    }
```

- [ ] **Step 4: Run** — 3 pass.

- [ ] **Step 5: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/agent/InfostateHasher.scala src/main/scala/sicfun/holdem/runtime/agent/BlueprintOnlyAgent.scala src/test/scala/sicfun/holdem/runtime/agent/BlueprintOnlyAgentTest.scala
git commit -m "feat(agent): BlueprintOnlyAgent with 5-action abstraction + honest translation"
```

---

## Phase 5 — StrategicAgent scaffold (4 tasks)

### Task 18: PlaceholderMarker infrastructure

**Design:** a marker trait plus a reflective scan helper that walks object graphs and collects all `PlaceholderMarker` instances. Used by `BenchmarkGate` to enumerate every placeholder in a runtime component tree.

**Files:**
- Create: `src/main/scala/sicfun/holdem/runtime/PlaceholderMarker.scala`
- Create: `src/test/scala/sicfun/holdem/runtime/PlaceholderMarkerTest.scala`

- [ ] **Step 1: Test**

```scala
package sicfun.holdem.runtime

class PlaceholderMarkerTest extends munit.FunSuite:

  class RealComponent
  class StubOne extends PlaceholderMarker:
    val placeholderReason = "stub one"
  class StubTwo extends PlaceholderMarker:
    val placeholderReason = "stub two"
  class ContainerDirect(val x: PlaceholderMarker)
  class ContainerNested(val child: ContainerDirect)

  test("scanPlaceholders: finds direct field"):
    val c = ContainerDirect(StubOne())
    val found = PlaceholderMarker.scanPlaceholders(c)
    assertEquals(found.map(_.placeholderReason).toSet, Set("stub one"))

  test("scanPlaceholders: finds nested field"):
    val c = ContainerNested(ContainerDirect(StubOne()))
    val found = PlaceholderMarker.scanPlaceholders(c)
    assertEquals(found.map(_.placeholderReason).toSet, Set("stub one"))

  test("scanPlaceholders: no placeholders → empty vector"):
    val c = new RealComponent
    assertEquals(PlaceholderMarker.scanPlaceholders(c), Vector.empty)

  test("scanPlaceholders: multiple placeholders collected"):
    val c = ContainerDirect(StubOne())
    val found = PlaceholderMarker.scanPlaceholders(Vector(c, StubTwo()))
    assertEquals(found.map(_.placeholderReason).toSet, Set("stub one", "stub two"))
```

- [ ] **Step 2: Implement**

```scala
package sicfun.holdem.runtime

/** Marker for any component that contains a hardcoded stub, dummy oracle, or
  * other scaffolding that invalidates benchmark claims. A `BenchmarkGate`
  * refuses to run in benchmark mode if any reachable component extends this. */
trait PlaceholderMarker:
  def placeholderReason: String

object PlaceholderMarker:

  /** Recursively walks a component's public fields (via reflection) and
    * collects all reachable `PlaceholderMarker` instances. Cycle-safe via
    * identity-based visited set.
    */
  def scanPlaceholders(root: Any): Vector[PlaceholderMarker] =
    val visited = collection.mutable.Set.empty[Any]
    val found = collection.mutable.ArrayBuffer.empty[PlaceholderMarker]

    def visit(obj: Any): Unit =
      if obj == null then ()
      else if visited.exists(_.asInstanceOf[AnyRef] eq obj.asInstanceOf[AnyRef]) then ()
      else
        visited += obj
        obj match
          case pm: PlaceholderMarker => found += pm
          case _ => ()
        obj match
          case it: Iterable[?] => it.foreach(visit)
          case arr: Array[?]   => arr.foreach(visit)
          case p: Product      => p.productIterator.foreach(visit)
          case other =>
            // Use reflection over public methods that look like getters.
            val cls = other.getClass
            cls.getMethods.foreach { m =>
              if m.getParameterCount == 0 &&
                !m.getName.startsWith("$") &&
                !isJavaLangObjectMethod(m.getName) &&
                m.getReturnType != classOf[Unit] &&
                m.getReturnType != java.lang.Void.TYPE
              then
                try
                  val v = m.invoke(other)
                  if v != other then visit(v)
                catch case _: Throwable => ()
            }

    visit(root)
    found.toVector

  private def isJavaLangObjectMethod(name: String): Boolean =
    Set("hashCode", "toString", "getClass", "wait", "notify", "notifyAll", "clone").contains(name)
```

*Reflection is brittle but the scan only runs once at `run()` entry in benchmark mode — performance cost is negligible. Tests verify the 4 shapes (direct field, nested, none, collection).*

- [ ] **Step 3: Run** — 4 pass.

- [ ] **Step 4: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/PlaceholderMarker.scala src/test/scala/sicfun/holdem/runtime/PlaceholderMarkerTest.scala
git commit -m "feat(runtime): PlaceholderMarker trait + reflective scanPlaceholders"
```

---

### Task 19: MdpEmbedding + ExploitabilityOracle traits with placeholders

**Files:**
- Create: `src/main/scala/sicfun/holdem/runtime/agent/MdpEmbedding.scala`
- Create: `src/main/scala/sicfun/holdem/runtime/agent/ExploitabilityOracle.scala`
- Create: `src/test/scala/sicfun/holdem/runtime/agent/MdpEmbeddingTest.scala`

- [ ] **Step 1: Tests**

```scala
package sicfun.holdem.runtime.agent

import sicfun.holdem.runtime.PlaceholderMarker
import sicfun.holdem.runtime.protocol.*
import sicfun.holdem.types.{PokerAction, Street}

class MdpEmbeddingTest extends munit.FunSuite:
  private def snap(): TableSnapshot = TableSnapshot(
    config = TableConfig(6, 1L, 2L, 0L, 200L),
    heroSeat = SeatId(0),
    holeCards = Vector.empty, board = Vector.empty,
    stacks = (0 until 6).map(i => SeatId(i) -> 200L).toMap,
    contributions = (0 until 6).map(i => SeatId(i) -> 0L).toMap,
    street = Street.Preflop, actionHistory = Vector.empty,
    buttonSeat = SeatId(0), activeSeats = (0 until 6).map(SeatId(_)).toSet
  )

  test("PlaceholderMdpEmbedding: extends PlaceholderMarker with non-empty reason"):
    val emb = PlaceholderMdpEmbedding()
    val built = emb.build(snap(), Vector(PokerAction.Fold, PokerAction.Call), numProfiles = 3)
    assert(emb.placeholderReason.nonEmpty)
    assertEquals(built.robustLosses.length, built.numStates)
    built.robustLosses.foreach(row => assertEquals(row.length, 2))

  test("ZeroExploitabilityOracle: returns 0 and extends PlaceholderMarker"):
    val oracle = ZeroExploitabilityOracle()
    assert(oracle.placeholderReason.nonEmpty)
    assertEquals(oracle.exploitabilityFn(0.7)(0.5), 0.0)
```

- [ ] **Step 2: Implement**

```scala
// MdpEmbedding.scala
package sicfun.holdem.runtime.agent

import sicfun.holdem.runtime.PlaceholderMarker
import sicfun.holdem.runtime.protocol.*
import sicfun.holdem.types.PokerAction

final case class BuiltMdp(
    numStates: Int,
    robustLosses: Array[Array[Double]],
    qValues: Array[Double],
    transitions: (Int, Int, Int) => IndexedSeq[(Int, Double)],
    numProfiles: Int,
    terminalStates: Set[Int]
)

trait MdpEmbedding:
  def build(
      snapshot: TableSnapshot,
      actions: Vector[PokerAction],
      numProfiles: Int
  ): BuiltMdp

/** Placeholder embedding: collapses the current spot into a 2-state MDP with
  * hand-tuned losses. Does NOT use bridges. A.2 replaces with real
  * `BridgeBasedMdpEmbedding`. */
final class PlaceholderMdpEmbedding extends MdpEmbedding, PlaceholderMarker:
  val placeholderReason: String =
    "MdpEmbedding is a 2-state stub with hardcoded losses; wire ValueBridge + " +
      "OpponentModelBridge in A.2 before any benchmark claim."

  def build(
      snapshot: TableSnapshot,
      actions: Vector[PokerAction],
      numProfiles: Int
  ): BuiltMdp =
    val numStates = 2
    val terminalStates = Set(1)
    val robustLosses = Array.fill(numStates, actions.size)(0.0)
    (0 until actions.size).foreach { a =>
      robustLosses(0)(a) = estimateLossForAction(snapshot, actions(a))
    }
    val qValues = Array.tabulate(actions.size)(a => -robustLosses(0)(a))
    val transitions: (Int, Int, Int) => IndexedSeq[(Int, Double)] =
      (s, _, _) => if s == 0 then Vector(1 -> 1.0) else Vector(1 -> 1.0)
    BuiltMdp(numStates, robustLosses, qValues, transitions, numProfiles, terminalStates)

  private def estimateLossForAction(snapshot: TableSnapshot, action: PokerAction): Double =
    action match
      case PokerAction.Fold            => snapshot.contributions(snapshot.heroSeat).toDouble
      case PokerAction.Check           => 0.0
      case PokerAction.Call            => 0.5
      case PokerAction.Raise(amount)   => amount * 0.1
```

```scala
// ExploitabilityOracle.scala
package sicfun.holdem.runtime.agent

import sicfun.holdem.runtime.PlaceholderMarker
import sicfun.holdem.strategic.types.PlayerId

trait ExploitabilityOracle:
  /** Returns `exploitabilityFn` suitable for `ExploitationInterpolation.updateExploitation`:
    * given the rival's policy drift, returns the estimated exploitability gap. */
  def exploitabilityFn(driftSignal: Double): Double => Double

final class ZeroExploitabilityOracle extends ExploitabilityOracle, PlaceholderMarker:
  val placeholderReason: String =
    "ExploitabilityOracle returns 0 everywhere; exploitation branch never activates. " +
      "Wire a real oracle (local best-response or CFR-based) in A.2."

  def exploitabilityFn(driftSignal: Double): Double => Double = _ => 0.0
```

- [ ] **Step 3: Run** — 2 pass.

- [ ] **Step 4: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/agent/MdpEmbedding.scala src/main/scala/sicfun/holdem/runtime/agent/ExploitabilityOracle.scala src/test/scala/sicfun/holdem/runtime/agent/MdpEmbeddingTest.scala
git commit -m "feat(agent): MdpEmbedding + ExploitabilityOracle traits with marked placeholders"
```

---

### Task 20: StrategicAgent with real PublicState adaptation

**Files:**
- Create: `src/main/scala/sicfun/holdem/runtime/agent/StrategicAgent.scala`
- Create: `src/test/scala/sicfun/holdem/runtime/agent/StrategicAgentWiringTest.scala`

**Key design:** `PublicState` is built via its real case-class constructor (no `.minimal`). `ExploitationConfig` uses explicit 3-param construction with justified defaults. `StrategicAgent` holds `MdpEmbedding`, `InfostateHasher`, `ExploitabilityOracle` as dependencies — all injectable so `BenchmarkGate` sees the placeholder graph.

- [ ] **Step 1: Implement**

```scala
package sicfun.holdem.runtime.agent

import sicfun.holdem.runtime.protocol.*
import sicfun.holdem.runtime.PlaceholderMarker
import sicfun.holdem.types.{PokerAction, Board, Card}
import sicfun.holdem.strategic.safety.{
  SafetyBellman, DetectionPredicate
}
import sicfun.holdem.strategic.exploitation.{
  ExploitationInterpolation, ExploitationState, ExploitationConfig
}
import sicfun.holdem.strategic.state.{PublicState, PublicAction, ActionSignal}
import sicfun.holdem.strategic.types.{
  Chips, PlayerId, TableMap, Seat, SeatStatus, Position
}

/** Explicit A.1 defaults for ExploitationConfig — no `.default` in the real
  * API, so we pin the values here with justification. */
object ExploitationDefaults:
  val a1Config: ExploitationConfig = ExploitationConfig(
    initialBeta = 0.0,      // start fully safety-dominated; no exploitation until signal
    cpRetreatRate = 0.1,    // gentle retreat per update
    epsilonAdapt = 0.01     // tight adaptation tolerance
  )

final class StrategicAgent(
    override val seatId: SeatId,
    val store: BlueprintStore,
    val hasher: InfostateHasher,
    val embedding: MdpEmbedding,
    val oracle: ExploitabilityOracle,
    val detector: DetectionPredicate,
    rngSeed: Long,
    abstractActions: Vector[PokerAction],
    config: ExploitationConfig = ExploitationDefaults.a1Config
) extends SeatAgent:

  private var exploitationState: ExploitationState = ExploitationState.initial(config)
  private val publicActions: collection.mutable.ArrayBuffer[PublicAction] =
    collection.mutable.ArrayBuffer.empty

  def currentBetaForTest: Double = exploitationState.beta

  def decide(snapshot: TableSnapshot, legalActions: Set[PokerAction]): PokerAction =
    val built = embedding.build(snapshot, abstractActions, numProfiles = 3)
    val bStar = SafetyBellman.computeBStar(
      robustLosses = built.robustLosses,
      gamma = 0.95,
      transitions = built.transitions,
      numProfiles = built.numProfiles,
      terminalStates = built.terminalStates
    )
    val safeActions = SafetyBellman.safeActionSet(
      stateIndex = 0,
      bound = bStar,
      robustLosses = built.robustLosses,
      gamma = 0.95,
      transitions = built.transitions,
      numProfiles = built.numProfiles
    )
    val chosenIdx = SafetyBellman.safeFeasibleAction(built.qValues, safeActions)
    val chosen = abstractActions(chosenIdx)
    translateToLegal(chosen, legalActions)

  override def onHandStart(snapshot: TableSnapshot): Unit =
    publicActions.clear()
    ingestHistoryIntoPublicActions(snapshot.actionHistory)

  override def onHandEnd(snapshot: TableSnapshot, outcome: HandOutcome): Unit =
    ingestHistoryIntoPublicActions(outcome.events)
    val rivals = snapshot.activeSeats.filter(_ != seatId)
    rivals.foreach { rival =>
      val rivalId = PlayerId(s"seat_${rival.index}")
      exploitationState = ExploitationInterpolation.updateExploitation(
        state = exploitationState,
        config = config,
        rivalId = rivalId,
        history = publicActions.toVector,
        publicState = buildPublicState(snapshot),
        detector = detector,
        exploitabilityFn = oracle.exploitabilityFn(0.0),
        epsilonNE = 0.0
      )
    }

  private def ingestHistoryIntoPublicActions(events: Vector[BettingRoundEvent]): Unit =
    events.foreach {
      case BettingRoundEvent.Act(seat, action) =>
        val signal = ActionSignal(
          action = action.category,
          sizing = action match
            case PokerAction.Raise(a) =>
              Some(sicfun.holdem.strategic.state.Sizing(a))
            case _ => None
          ,
          timing = None,
          stage = sicfun.holdem.types.Street.Preflop // TODO: track stage per-event
        )
        publicActions += PublicAction(
          actor = PlayerId(s"seat_${seat.index}"),
          signal = signal
        )
      case _ => ()
    }

  private def buildPublicState(snapshot: TableSnapshot): PublicState =
    PublicState(
      street = snapshot.street,
      board = Board(snapshot.board),
      pot = Chips(snapshot.contributions.values.sum.toDouble),
      stacks = buildStacksMap(snapshot),
      actionHistory = publicActions.toVector
    )

  private def buildStacksMap(snapshot: TableSnapshot): TableMap[Chips] =
    val heroId = PlayerId(s"seat_${seatId.index}")
    val seats = (0 until snapshot.config.numSeats).map { i =>
      val s = SeatId(i)
      val pid = PlayerId(s"seat_$i")
      val status =
        if !snapshot.activeSeats.contains(s) then SeatStatus.Folded
        else if snapshot.stacks(s) == 0L then SeatStatus.AllIn
        else SeatStatus.Active
      Seat(pid, positionFor(i, snapshot.config.numSeats), status, Chips(snapshot.stacks(s).toDouble))
    }.toVector
    TableMap(hero = heroId, seats = seats)

  private def positionFor(seatIdx: Int, n: Int): Position =
    // Rough mapping sufficient for A.1; A.2 should consult TableDealer's
    // button/blind positions for exact assignment.
    Position.fromOrdinal(seatIdx.min(Position.values.length - 1))

  private def translateToLegal(a: PokerAction, legal: Set[PokerAction]): PokerAction =
    if legal.contains(a) then a
    else a match
      case _: PokerAction.Raise =>
        legal.collect { case r: PokerAction.Raise => r }.headOption
          .orElse(legal.collectFirst { case PokerAction.Call => PokerAction.Call })
          .orElse(legal.collectFirst { case PokerAction.Check => PokerAction.Check })
          .getOrElse(PokerAction.Fold)
      case PokerAction.Check =>
        if legal.contains(PokerAction.Call) then PokerAction.Call else PokerAction.Fold
      case PokerAction.Call =>
        if legal.contains(PokerAction.Check) then PokerAction.Check else PokerAction.Fold
      case PokerAction.Fold => PokerAction.Fold
```

*Notes for the implementer:*
- *`Sizing` constructor — grep `src/main/scala/sicfun/holdem/strategic/state/Signal.scala` for its exact shape. If it's `Sizing(fraction: PotFraction)` rather than `Sizing(amount: Double)`, convert via `PotFraction(amount / potSize)`.*
- *`Position.fromOrdinal` may not exist; if so, use `Position.values(seatIdx.min(8))`.*

- [ ] **Step 2: Commit (before tests, to enable Task 21 wiring tests)**

```bash
git add src/main/scala/sicfun/holdem/runtime/agent/StrategicAgent.scala
git commit -m "feat(agent): StrategicAgent with real PublicState + explicit ExploitationConfig"
```

---

### Task 21: StrategicAgent wiring tests with concrete SafetyBellman invariants

**Files:**
- Create: `src/test/scala/sicfun/holdem/runtime/agent/StrategicAgentWiringTest.scala`

- [ ] **Step 1: Tests**

```scala
package sicfun.holdem.runtime.agent

import sicfun.holdem.runtime.protocol.*
import sicfun.holdem.runtime.PlaceholderMarker
import sicfun.holdem.strategic.safety.{SafetyBellman, NeverDetect}
import sicfun.holdem.types.{PokerAction, Street}
import java.nio.file.Files

class StrategicAgentWiringTest extends munit.FunSuite:

  private def snap(): TableSnapshot = TableSnapshot(
    config = TableConfig(6, 1L, 2L, 0L, 200L),
    heroSeat = SeatId(0),
    holeCards = Vector.empty, board = Vector.empty,
    stacks = (0 until 6).map(i => SeatId(i) -> 200L).toMap,
    contributions = (0 until 6).map(i => SeatId(i) -> 0L).toMap,
    street = Street.Preflop, actionHistory = Vector.empty,
    buttonSeat = SeatId(0), activeSeats = (0 until 6).map(SeatId(_)).toSet
  )

  private def makeAgent(): StrategicAgent =
    val tmp = Files.createTempFile("bp", ".blueprint").toFile
    BlueprintStore.write(tmp, BlueprintHeader("SICFBP01", 1, 0L, "h", 6, 0, 5), Map.empty)
    val store = BlueprintStore.load(tmp, "h")
    StrategicAgent(
      seatId = SeatId(0),
      store = store,
      hasher = PlaceholderInfostateHasher(),
      embedding = PlaceholderMdpEmbedding(),
      oracle = ZeroExploitabilityOracle(),
      detector = NeverDetect,
      rngSeed = 1L,
      abstractActions = Vector(
        PokerAction.Fold, PokerAction.Call,
        PokerAction.Raise(4.0), PokerAction.Raise(8.0), PokerAction.Raise(200.0)
      )
    )

  // ---- SafetyBellman invariants (direct, not via agent) ----

  test("safeActionSet ⊆ [0, numActions), can be empty"):
    val losses = Array(Array(10.0, 10.0, 10.0), Array(0.0, 0.0, 0.0))
    val transitions: (Int, Int, Int) => IndexedSeq[(Int, Double)] =
      (s, _, _) => if s == 0 then Vector(1 -> 1.0) else Vector(1 -> 1.0)
    val bStar = SafetyBellman.computeBStar(losses, 0.95, transitions, numProfiles = 1,
      terminalStates = Set(1))
    // Force threshold below any achievable loss by clobbering bound(0).
    val tightBound = Array(0.1, 0.0)
    val safe = SafetyBellman.safeActionSet(0, tightBound, losses, 0.95, transitions, 1)
    assert(safe.forall(i => i >= 0 && i < 3), s"indices out of range: $safe")
    assertEquals(safe, IndexedSeq.empty[Int], s"expected empty safe set with tight bound, got $safe")

  test("safeFeasibleAction: empty safe set → global argmax of qValues"):
    val qValues = Array(5.0, 10.0, 3.0)
    val chosen = SafetyBellman.safeFeasibleAction(qValues, IndexedSeq.empty[Int])
    assertEquals(chosen, 1, "expected argmax index 1 (qValues(1) = 10)")

  test("safeFeasibleAction: non-empty safe set → argmax restricted to safe"):
    val qValues = Array(5.0, 10.0, 3.0)
    val chosen = SafetyBellman.safeFeasibleAction(qValues, IndexedSeq(0, 2))
    assertEquals(chosen, 0, "expected argmax among {0, 2} = 0 (qValues(0)=5 > qValues(2)=3)")

  // ---- StrategicAgent wiring invariants ----

  test("decide returns an action in the legal set"):
    val agent = makeAgent()
    val legal = Set[PokerAction](PokerAction.Fold, PokerAction.Call, PokerAction.Raise(4.0))
    val a = agent.decide(snap(), legal)
    assert(legal.contains(a), s"$a not in $legal")

  test("scanPlaceholders on StrategicAgent finds MdpEmbedding + Oracle + Hasher"):
    val agent = makeAgent()
    val found = PlaceholderMarker.scanPlaceholders(agent)
    val reasons = found.map(_.placeholderReason).mkString("|")
    assert(reasons.contains("MdpEmbedding"), reasons)
    assert(reasons.contains("ExploitabilityOracle"), reasons)
    assert(reasons.contains("InfostateHasher"), reasons)
```

*If the scan misses some because reflection skips `val` vs `def`, adjust `scanPlaceholders` to also walk `getDeclaredFields` with `setAccessible`. The wiring test is the forcing function here.*

- [ ] **Step 2: Run** — 5 pass. If the scan invariant fails, adjust `PlaceholderMarker.scanPlaceholders` to include declared fields (not just methods).

- [ ] **Step 3: Commit**

```bash
git add src/test/scala/sicfun/holdem/runtime/agent/StrategicAgentWiringTest.scala
# + any tweak to PlaceholderMarker.scala if the scan needed widening
git commit -m "test(agent): StrategicAgent wiring + SafetyBellman invariants"
```

---

## Phase 6 — Runner + metrics + smoke (4 tasks)

### Task 22: MbbMetrics with correct units (×100,000)

**Fix for v1 bug:** v1 computed `mean * 1000` and called it `mbbPer100`. Correct formula: `mean_BB_per_hand × 1000 (BB→mbb) × 100 (per-hand→per-100-hands) = mean × 100,000`.

**Files:**
- Create: `src/main/scala/sicfun/holdem/runtime/metrics/MbbMetrics.scala`
- Create: `src/test/scala/sicfun/holdem/runtime/metrics/MbbMetricsTest.scala`

- [ ] **Step 1: Tests**

```scala
package sicfun.holdem.runtime.metrics

class MbbMetricsTest extends munit.FunSuite:

  test("mbbPer100: 2 BB/hand constant → 200,000 mbb/100 (v1 bug regression)"):
    val wins = (1 to 100).map(_ => 2.0).toVector
    val got = MbbMetrics.mbbPer100(wins)
    assertEqualsDouble(got, 200_000.0, 1e-6)

  test("mbbPer100: zero-mean → 0"):
    val wins = Vector(1.0, -1.0, 2.0, -2.0)
    assertEqualsDouble(MbbMetrics.mbbPer100(wins), 0.0, 1e-6)

  test("bootstrapIC95: deterministic under seed + IC contains the mean"):
    val wins = (1 to 1000).map(i => if i % 2 == 0 then 1.0 else -1.0).toVector
    val ci = MbbMetrics.bootstrapIC95(wins, iterations = 500, rngSeed = 42L)
    val mean = MbbMetrics.mbbPer100(wins)
    assert(ci.lower <= mean && mean <= ci.upper, s"mean $mean not in $ci")
    // Repeat under same seed to verify determinism.
    val ci2 = MbbMetrics.bootstrapIC95(wins, iterations = 500, rngSeed = 42L)
    assertEquals(ci, ci2)
```

- [ ] **Step 2: Implement**

```scala
package sicfun.holdem.runtime.metrics

import scala.util.Random

final case class ConfidenceInterval(lower: Double, upper: Double, level: Double):
  override def toString: String = f"CI${(level * 100).toInt}%d=[$lower%.2f, $upper%.2f]"

object MbbMetrics:

  /** Winrate in milli-big-blinds per 100 hands.
    *
    * Contract: `winningsPerHandInBB` values are BB per hand (positive = hero won).
    * Formula: mean_BB_per_hand × 1000 (BB→mbb) × 100 (per-100) = mean × 100_000.
    *
    * v1 shipped `mean * 1000` which was mbb/hand, not mbb/100; tests reproduced
    * the bug so CI passed. Do not regress. */
  def mbbPer100(winningsPerHandInBB: Vector[Double]): Double =
    if winningsPerHandInBB.isEmpty then 0.0
    else
      val mean = winningsPerHandInBB.sum / winningsPerHandInBB.size
      mean * 100_000.0

  def bootstrapIC95(
      winningsPerHand: Vector[Double],
      iterations: Int,
      rngSeed: Long
  ): ConfidenceInterval =
    val rng = new Random(rngSeed)
    val n = winningsPerHand.size
    val resamples = (1 to iterations).map { _ =>
      val sample = (1 to n).map(_ => winningsPerHand(rng.nextInt(n))).toVector
      mbbPer100(sample)
    }.sorted
    val lower = resamples((iterations * 0.025).toInt)
    val upper = resamples((iterations * 0.975).toInt.min(iterations - 1))
    ConfidenceInterval(lower, upper, 0.95)
```

- [ ] **Step 3: Run** — 3 pass.

- [ ] **Step 4: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/metrics/MbbMetrics.scala src/test/scala/sicfun/holdem/runtime/metrics/MbbMetricsTest.scala
git commit -m "feat(metrics): mbbPer100 with correct units (×100,000) + bootstrap IC95"
```

---

### Task 23: BenchmarkGate

**Files:**
- Create: `src/main/scala/sicfun/holdem/runtime/BenchmarkGate.scala`
- Create: `src/test/scala/sicfun/holdem/runtime/BenchmarkGateTest.scala`

- [ ] **Step 1: Tests**

```scala
package sicfun.holdem.runtime

import sicfun.holdem.runtime.agent.*
import sicfun.holdem.strategic.safety.NeverDetect
import sicfun.holdem.types.PokerAction
import java.nio.file.Files

class BenchmarkGateTest extends munit.FunSuite:

  private def strategicAgent(): StrategicAgent =
    val tmp = Files.createTempFile("bp", ".blueprint").toFile
    BlueprintStore.write(tmp, BlueprintHeader("SICFBP01", 1, 0L, "h", 9, 0, 5), Map.empty)
    val store = BlueprintStore.load(tmp, "h")
    StrategicAgent(
      seatId = sicfun.holdem.runtime.protocol.SeatId(0),
      store = store,
      hasher = PlaceholderInfostateHasher(),
      embedding = PlaceholderMdpEmbedding(),
      oracle = ZeroExploitabilityOracle(),
      detector = NeverDetect,
      rngSeed = 1L,
      abstractActions = Vector(PokerAction.Fold, PokerAction.Call,
        PokerAction.Raise(4.0), PokerAction.Raise(8.0), PokerAction.Raise(200.0))
    )

  private def blueprintAgent(): BlueprintOnlyAgent =
    val tmp = Files.createTempFile("bp2", ".blueprint").toFile
    BlueprintStore.write(tmp, BlueprintHeader("SICFBP01", 1, 0L, "h", 9, 0, 5), Map.empty)
    val store = BlueprintStore.load(tmp, "h")
    BlueprintOnlyAgent(
      sicfun.holdem.runtime.protocol.SeatId(1), store, PlaceholderInfostateHasher(), 2L)

  test("benchmarkMode = true + StrategicAgent → Left(violation) listing all placeholders"):
    val agents = Vector(strategicAgent())
    val result = BenchmarkGate.check(agents, benchmarkMode = true)
    assert(result.isLeft)
    result.left.foreach { v =>
      val msg = v.getMessage
      assert(msg.contains("MdpEmbedding"), msg)
      assert(msg.contains("ExploitabilityOracle"), msg)
      assert(msg.contains("InfostateHasher"), msg)
    }

  test("benchmarkMode = false → Right(()) even with placeholders"):
    val agents = Vector(strategicAgent())
    val result = BenchmarkGate.check(agents, benchmarkMode = false)
    assert(result.isRight)

  test("benchmarkMode = true + BlueprintOnlyAgent only → still blocks on PlaceholderInfostateHasher"):
    // BlueprintOnlyAgent uses PlaceholderInfostateHasher in A.1, so it also trips.
    val agents = Vector(blueprintAgent())
    val result = BenchmarkGate.check(agents, benchmarkMode = true)
    assert(result.isLeft,
      "A.1 has no non-placeholder hasher; real InfostateHasher arrives in A.2")
```

- [ ] **Step 2: Implement**

```scala
package sicfun.holdem.runtime

import sicfun.holdem.runtime.agent.SeatAgent

final case class BenchmarkGateViolation(message: String)
    extends RuntimeException(message)

object BenchmarkGate:

  def check(
      agents: Vector[SeatAgent],
      benchmarkMode: Boolean
  ): Either[BenchmarkGateViolation, Unit] =
    if !benchmarkMode then Right(())
    else
      val placeholders = agents.flatMap(a => PlaceholderMarker.scanPlaceholders(a))
      if placeholders.isEmpty then Right(())
      else
        val reasons = placeholders.map(p => s"  - ${p.getClass.getSimpleName}: ${p.placeholderReason}").distinct
        Left(BenchmarkGateViolation(
          s"BenchmarkGate refuses to run in benchmarkMode=true. Reachable placeholders:\n${reasons.mkString("\n")}"
        ))
```

- [ ] **Step 3: Run** — 3 pass.

- [ ] **Step 4: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/BenchmarkGate.scala src/test/scala/sicfun/holdem/runtime/BenchmarkGateTest.scala
git commit -m "feat(runtime): BenchmarkGate enforces no-placeholders in benchmarkMode"
```

---

### Task 24: NineMaxMatchRunner orchestration

**Files:**
- Create: `src/main/scala/sicfun/holdem/runtime/NineMaxMatchRunner.scala`

**Contract:**
- Constructor takes `tableConfig`, `agents`, `numHands`, `rngSeed`, `matchId`, `strictNative`, `benchmarkMode`.
- On `run()`:
  1. Call `NativeStrictMode.verify(...)` — throw on violation if `strictNative`.
  2. Call `BenchmarkGate.check(agents, benchmarkMode)` — throw on violation.
  3. Play `numHands` hands, each with a fresh dealer instance (stacks reset).
  4. After each hand, assert `outcome.netChange.values.sum == 0` (A.1 success criterion #2). Any violation raises `ChipConservationError`.
  5. Accumulate per-seat winnings; compute `mbbPer100` and `bootstrapIC95`.
- Returns `MatchResult(handsPlayed, netBySeat, mbbPer100BySeat, ci95BySeat, matchLogPath)`.

- [ ] **Step 1: Implement**

```scala
package sicfun.holdem.runtime

import sicfun.holdem.runtime.agent.*
import sicfun.holdem.runtime.protocol.*
import sicfun.holdem.runtime.metrics.*
import sicfun.holdem.types.{PokerAction, Street}
import java.io.{File, PrintWriter}

final case class ChipConservationError(handNumber: Int, sumNetChange: Long)
    extends RuntimeException(
      s"chip conservation violated on hand $handNumber: sum(netChange) = $sumNetChange, expected 0"
    )

final case class MatchResult(
    handsPlayed: Int,
    netBySeat: Map[SeatId, Long],
    mbbPer100BySeat: Map[SeatId, Double],
    ci95BySeat: Map[SeatId, ConfidenceInterval],
    matchLogPath: String
)

final class NineMaxMatchRunner(
    tableConfig: TableConfig,
    agents: Vector[SeatAgent],
    numHands: Int,
    rngSeed: Long,
    matchId: String,
    strictNative: Boolean = true,
    benchmarkMode: Boolean = false
):
  require(agents.size == tableConfig.numSeats,
    s"agent count ${agents.size} must equal numSeats ${tableConfig.numSeats}")

  def run(): MatchResult =
    NativeStrictMode.verify(
      new File("src/main/native/build"),
      NativeStrictMode.CoreLibraries,
      strict = strictNative
    ) match
      case Left(v)  => throw v
      case Right(_) => ()

    BenchmarkGate.check(agents, benchmarkMode) match
      case Left(v)  => throw v
      case Right(_) => ()

    val winningsBySeat = (0 until tableConfig.numSeats)
      .map(i => SeatId(i) -> scala.collection.mutable.ArrayBuffer.empty[Double])
      .toMap
    var currentButton = SeatId(0)
    val bb = tableConfig.bigBlind.toDouble
    val logFile = new File(s"data/matches/$matchId.jsonl")
    logFile.getParentFile.mkdirs()
    val log = new PrintWriter(logFile)

    try
      agents.foreach(_.onMatchStart(tableConfig))

      (1 to numHands).foreach { handNum =>
        val dealer = new AcpcTableDealer(tableConfig, currentButton, rngSeed + handNum.toLong)
        val outcome = playHand(dealer, agents)

        // A.1 success criterion #2: chip conservation per hand.
        val conservationSum = outcome.netChange.values.sum
        if conservationSum != 0L then
          throw ChipConservationError(handNum, conservationSum)

        outcome.netChange.foreach { (seat, delta) =>
          winningsBySeat(seat) += delta.toDouble / bb
        }

        log.println(
          s"""{"hand":$handNum,"net":{${outcome.netChange.toVector.sortBy(_._1.index)
            .map((s, d) => s""""${s.index}":$d""").mkString(",")}}}"""
        )
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
        mbbPer100BySeat = winningsBySeat.map((s, w) => s -> MbbMetrics.mbbPer100(w.toVector)),
        ci95BySeat = winningsBySeat.map((s, w) =>
          s -> MbbMetrics.bootstrapIC95(w.toVector, iterations = 500, rngSeed = rngSeed)),
        matchLogPath = logFile.getPath
      )
    finally log.close()

  private def playHand(dealer: AcpcTableDealer, agents: Vector[SeatAgent]): HandOutcome =
    val holeCards = dealer.dealHoleCards()
    agents.foreach { a =>
      a.onHandStart(buildSnapshot(dealer, a.seatId, holeCards))
    }
    dealer.postBlinds()
    playStreet(Street.Preflop, dealer, agents, holeCards)
    if !dealer.handEnded then
      dealer.dealCommunity(Street.Flop)
      playStreet(Street.Flop, dealer, agents, holeCards)
    if !dealer.handEnded then
      dealer.dealCommunity(Street.Turn)
      playStreet(Street.Turn, dealer, agents, holeCards)
    if !dealer.handEnded then
      dealer.dealCommunity(Street.River)
      playStreet(Street.River, dealer, agents, holeCards)
    val outcome = dealer.finalizeHand()
    agents.foreach { a =>
      a.onHandEnd(buildSnapshot(dealer, a.seatId, holeCards), outcome)
    }
    outcome

  private def playStreet(
      street: Street,
      dealer: AcpcTableDealer,
      agents: Vector[SeatAgent],
      holeCards: Map[SeatId, Vector[sicfun.holdem.types.Card]]
  ): Unit =
    dealer.startStreet(street)
    while !dealer.roundClosed do
      dealer.nextToAct match
        case None => ()
        case Some(seat) =>
          val agent = agents(seat.index)
          val snap = buildSnapshot(dealer, seat, holeCards)
          val legal = dealer.legalActionsFor(seat)
          val action = agent.decide(snap, legal)
          dealer.applyAction(seat, action) match
            case Left(err) =>
              throw new RuntimeException(s"agent $seat returned illegal action: $err")
            case Right(_) => ()

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
      street = Street.Preflop, // dealer should expose current street; add accessor if missing
      actionHistory = dealer.eventLog,
      buttonSeat = dealer.buttonSeat,
      activeSeats = (0 until tableConfig.numSeats).map(SeatId(_))
        .filter(s => dealer.currentStacks(s) >= 0L).toSet
    )
```

*Note: `buildSnapshot` currently hardcodes `Street.Preflop` for the `street` field because the dealer does not expose `currentStreet` publicly in Task 5. Add a one-line public accessor `def currentStreetValue: Street = currentStreet` to the dealer during this task and use it.*

- [ ] **Step 2: Commit**

```bash
git add src/main/scala/sicfun/holdem/runtime/NineMaxMatchRunner.scala src/main/scala/sicfun/holdem/runtime/protocol/AcpcTableDealer.scala
git commit -m "feat(runtime): NineMaxMatchRunner with chip-conservation assertion per hand"
```

---

### Task 25: Smoke test — 9× BlueprintOnlyAgent, 1000 hands, exact conservation

**Files:**
- Create: `src/test/scala/sicfun/holdem/runtime/NineMaxMatchRunnerSmokeTest.scala`

- [ ] **Step 1: Tests**

```scala
package sicfun.holdem.runtime

import sicfun.holdem.runtime.agent.*
import sicfun.holdem.runtime.protocol.*
import sicfun.holdem.strategic.safety.NeverDetect
import sicfun.holdem.types.PokerAction
import java.nio.file.Files

class NineMaxMatchRunnerSmokeTest extends munit.FunSuite:

  private def store9(): BlueprintStore =
    val tmp = Files.createTempFile("bp", ".blueprint").toFile
    BlueprintStore.write(tmp, BlueprintHeader("SICFBP01", 1, 0L, "h", 9, 0, 5), Map.empty)
    BlueprintStore.load(tmp, "h")

  test("1000 hands with 9 BlueprintOnlyAgents: chip conservation per hand, per match"):
    val store = store9()
    val cfg = TableConfig(9, 1L, 2L, 0L, 200L)

    val agents: Vector[SeatAgent] = (0 until 9).map { i =>
      BlueprintOnlyAgent(SeatId(i), store, PlaceholderInfostateHasher(), rngSeed = i.toLong)
    }.toVector

    val result = NineMaxMatchRunner(
      tableConfig = cfg,
      agents = agents,
      numHands = 1000,
      rngSeed = 42L,
      matchId = "a1-smoke",
      strictNative = false,
      benchmarkMode = false
    ).run()

    // A.1 criterion #1: hand count.
    assertEquals(result.handsPlayed, 1000)

    // A.1 criterion #3: per-match chip conservation (within float ε).
    val netSum = result.netBySeat.values.sum
    assertEquals(netSum, 0L, s"per-match net sum != 0: $netSum")

    // A.1 criterion #5: IC95 sanity — each seat's IC includes 0.
    result.ci95BySeat.foreach { (s, ci) =>
      assert(ci.lower <= 0.0 && ci.upper >= 0.0,
        s"seat ${s.index} IC95=$ci does not cross zero")
    }

    // Log inspection: match log file exists and has one line per hand.
    val lines = scala.io.Source.fromFile(result.matchLogPath).getLines().toVector
    assertEquals(lines.size, 1000)

  test("benchmarkMode = true with StrategicAgent → BenchmarkGateViolation listing placeholders"):
    val store = store9()
    val cfg = TableConfig(9, 1L, 2L, 0L, 200L)
    val actions = Vector(PokerAction.Fold, PokerAction.Call,
      PokerAction.Raise(4.0), PokerAction.Raise(8.0), PokerAction.Raise(200.0))

    val strategic = StrategicAgent(
      seatId = SeatId(0), store = store,
      hasher = PlaceholderInfostateHasher(),
      embedding = PlaceholderMdpEmbedding(),
      oracle = ZeroExploitabilityOracle(),
      detector = NeverDetect, rngSeed = 7L,
      abstractActions = actions
    )
    val rest: Vector[SeatAgent] = (1 until 9).map { i =>
      BlueprintOnlyAgent(SeatId(i), store, PlaceholderInfostateHasher(), i.toLong)
    }.toVector

    val thrown = intercept[BenchmarkGateViolation](
      NineMaxMatchRunner(cfg, strategic +: rest, 100, 1L, "a1-gate", strictNative = false,
        benchmarkMode = true).run()
    )
    val msg = thrown.getMessage
    assert(msg.contains("MdpEmbedding"), msg)
    assert(msg.contains("ExploitabilityOracle"), msg)
    assert(msg.contains("InfostateHasher"), msg)
```

*The first test should complete in well under 60s given agents just sample from a uniform distribution. If it exceeds 2 min, investigate dealer dealing perf (likely `deckBuf.remove(0)` O(N²) amortized — if so, use a pointer/index instead).*

- [ ] **Step 2: Run** — both pass.

- [ ] **Step 3: Commit**

```bash
git add src/test/scala/sicfun/holdem/runtime/NineMaxMatchRunnerSmokeTest.scala
git commit -m "test(runtime): A.1 smoke — 9×BlueprintOnlyAgent 1000 hands + BenchmarkGate rejects StrategicAgent"
```

---

## Deferred to Spec A.2 (explicitly out of scope here)

1. **Real `MdpEmbedding`** from the 6 bridges (`ValueBridge`, `OpponentModelBridge`, `EvidenceBridge`, `AttributionBridge`, `PolicyBridge`, `AssumptionBridge`). Start A.2 by auditing the public API of each bridge before committing to an embedding shape.
2. **Real `ExploitabilityOracle`** — either local best-response on the placeholder MDP or CFR-based approximate exploitability gap.
3. **Real `InfostateHasher`** — abstraction-aware hash aligned with blueprint training.
4. **`FourWorldMetrics`** wired to `FourWorldDecomposition` per hand; accumulated into `MatchResult`.
5. **Seat rotation / duplicate-holdings convention** to cancel positional EV in benchmarks (Pluribus-style).
6. **Real blueprint training pipeline** — separate spec.
7. **Action translation fidelity** — Pluribus-style pseudo-random EV-preserving translation in `BlueprintOnlyAgent`.
8. **`StrategicAgent.publicActions` stage tracking** — current ingestion hardcodes `Street.Preflop`; needs per-event stage accounting.
9. **`BlueprintStore` mmap** — current load reads rows fully into memory; a large (A.2) blueprint may need memory-mapped access.

## Success criteria recap (A.1 only)

| # | Criterion | Task that proves it |
|---|---|---|
| 1 | 9 `BlueprintOnlyAgent` play 1000 hands | Task 25 |
| 2 | `outcome.netChange.values.sum == 0` per hand | `HandOutcome` `require` + Task 24 runner + Task 11 property |
| 3 | `sum(netBySeat) == 0` per match | Task 25 |
| 4 | `benchmarkMode=true` with any `StrategicAgent` → `BenchmarkGateViolation` enumerating placeholders | Task 23 + Task 25 |
| 5 | IC95 crosses 0 for each seat | Task 25 (sanity, not blocking) |

**No benchmark claim on strategic-overlay value is made by A.1.** That is explicitly the responsibility of A.2.
