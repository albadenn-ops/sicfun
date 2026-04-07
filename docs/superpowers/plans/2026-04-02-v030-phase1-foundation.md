# Phase 1: Foundation Types + BOCD -- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the pure Scala foundation layer for the SICFUN v0.30.2 formal specification -- Definitions 1-14, 26-28, type signatures for 16-21, and value/delta vocabulary types (Defs 44-47, 50) -- as the new `sicfun.holdem.strategic` package.

**Architecture:** All types live in `sicfun.holdem.strategic`, a pure package with zero runtime dependencies. It imports ONLY from `sicfun.holdem.types` (PokerAction, Street, Position, HoleCards, Board) and `sicfun.core` (DiscreteDistribution, Probability). No imports from `sicfun.holdem.engine` or `sicfun.holdem.runtime` are permitted. The package is multiway-native: heads-up is the subcase |R|=1, never a special path.

**Tech Stack:** Scala 3.8.1, munit 1.2.2

**Depends on:** Nothing (Phase 1 is the foundation)

**Unlocks:** Phases 2, 3, 4

---

## File Map

| File | Responsibility | Spec Defs |
|------|---------------|-----------|
| `src/main/scala/sicfun/holdem/strategic/DomainTypes.scala` | Chips, PotFraction, Ev opaques | -- |
| `src/main/scala/sicfun/holdem/strategic/TableStructure.scala` | PlayerId, SeatStatus, TableMap, RivalMap | -- |
| `src/main/scala/sicfun/holdem/strategic/Fidelity.scala` | BridgeResult, Fidelity, Severity enums | -- |
| `src/main/scala/sicfun/holdem/strategic/StrategicClass.scala` | V/B/M/SB enum + predicates | Defs 1-4 |
| `src/main/scala/sicfun/holdem/strategic/Signal.scala` | ActionSignal, ShowdownSignal, TotalSignal, routing | Defs 5-8 |
| `src/main/scala/sicfun/holdem/strategic/Baseline.scala` | RealBaseline, AttributedBaseline traits | Defs 9-10 |
| `src/main/scala/sicfun/holdem/strategic/ReputationalProjection.scala` | Reputational projection trait | Def 11 |
| `src/main/scala/sicfun/holdem/strategic/AugmentedState.scala` | Augmented state product type, OperativeBelief | Defs 12-14 |
| `src/main/scala/sicfun/holdem/strategic/RivalKernel.scala` | Trait hierarchy (type signatures only) | Defs 16-21 |
| `src/main/scala/sicfun/holdem/strategic/StrategicValue.scala` | FourWorld, Delta-vocabulary types | Defs 44-47, 50 |
| `src/main/scala/sicfun/holdem/strategic/ChangepointDetector.scala` | Adams-MacKay BOCD | Defs 26-28 |
| `src/test/scala/sicfun/holdem/strategic/TableStructureTest.scala` | TableMap/RivalMap laws | -- |
| `src/test/scala/sicfun/holdem/strategic/StrategicClassTest.scala` | Class enum + predicate tests | Defs 1-4 |
| `src/test/scala/sicfun/holdem/strategic/SignalTest.scala` | Signal construction + routing | Defs 5-8 |
| `src/test/scala/sicfun/holdem/strategic/AugmentedStateTest.scala` | Augmented state invariants | Defs 12-14 |
| `src/test/scala/sicfun/holdem/strategic/RivalKernelLawTest.scala` | Kernel trait law tests | Defs 16-21 |
| `src/test/scala/sicfun/holdem/strategic/StrategicValueTest.scala` | Four-world decomposition | Defs 44-47, 50 |
| `src/test/scala/sicfun/holdem/strategic/ChangepointDetectorTest.scala` | BOCD correctness | Defs 26-28 |

---

## Task Execution Order

Tasks are ordered by dependency. Each task is self-contained (test + implementation + verification).

| Task | File | Depends On | Defs |
|------|------|-----------|------|
| 1 | DomainTypes.scala | (none) | Chips, PotFraction, Ev opaques |
| 2 | TableStructure.scala + test | Task 1 | PlayerId, SeatStatus, TableMap, RivalMap |
| 3 | Fidelity.scala | (none) | BridgeResult, Fidelity, Severity |
| 4 | StrategicClass.scala + test | Task 1 | Defs 1-4 |
| 5 | Signal.scala + test | Tasks 1, 2 | Defs 5-8 |
| 6 | Baseline.scala | Tasks 1, 4, 5 | Defs 9-10 |
| 7 | ReputationalProjection.scala | Tasks 1, 5, 8 | Def 11 |
| 8 | AugmentedState.scala + test | Tasks 1-7 | Defs 12-14 |
| 9 | RivalKernel.scala + test | Tasks 1, 2, 4, 5 | Defs 16-21 (sigs) |
| 10 | StrategicValue.scala + test | Task 1 | Defs 44-47, 50 |
| 11 | ChangepointDetector.scala + test | Task 4 | Defs 26-28 |
| 12 | Full verification | All | -- |

---

## Dependency Rule Enforcement (NON-NEGOTIABLE)

Every file in `sicfun.holdem.strategic` may import ONLY:

```
import sicfun.holdem.types.{PokerAction, Street, Position, HoleCards, Board}
import sicfun.core.{DiscreteDistribution, Probability}
```

NO imports from `sicfun.holdem.engine`, `sicfun.holdem.runtime`, `sicfun.holdem.types.GameState`, or any other `sicfun.holdem.*` subpackage.

---

### Task 1: DomainTypes

**Files:**
- Create: `src/main/scala/sicfun/holdem/strategic/DomainTypes.scala`
- (No separate test file -- opaque types are validated through usage in Task 2+ tests)

- [ ] **Step 1: Write the implementation**

```scala
package sicfun.holdem.strategic

/** Domain-specific opaque types for the formal strategic layer.
  *
  * These opaques prevent accidental mixing of raw Doubles that represent
  * semantically distinct quantities (chip amounts, pot fractions, expected values).
  * The bridge layer converts raw Doubles from the engine into these types.
  */

opaque type Chips = Double
object Chips:
  inline def apply(value: Double): Chips = value
  extension (c: Chips)
    inline def value: Double = c
    inline def +(other: Chips): Chips = c + other
    inline def -(other: Chips): Chips = c - other
    inline def *(scalar: Double): Chips = c * scalar
    inline def /(divisor: Double): Chips = c / divisor
    inline def unary_- : Chips = -c
    inline def >=(other: Chips): Boolean = c >= other
    inline def >(other: Chips): Boolean = c > other
    inline def <=(other: Chips): Boolean = c <= other
    inline def <(other: Chips): Boolean = c < other
  given Ordering[Chips] = Ordering.Double.TotalOrdering

opaque type PotFraction = Double
object PotFraction:
  inline def apply(value: Double): PotFraction = value
  val Zero: PotFraction = 0.0
  val One: PotFraction = 1.0
  extension (p: PotFraction)
    inline def value: Double = p
    inline def +(other: PotFraction): PotFraction = p + other
    inline def -(other: PotFraction): PotFraction = p - other
    inline def *(scalar: Double): PotFraction = p * scalar
    inline def >=(other: PotFraction): Boolean = p >= other
    inline def >(other: PotFraction): Boolean = p > other

opaque type Ev = Double
object Ev:
  inline def apply(value: Double): Ev = value
  val Zero: Ev = 0.0
  extension (e: Ev)
    inline def value: Double = e
    inline def +(other: Ev): Ev = e + other
    inline def -(other: Ev): Ev = e - other
    inline def *(scalar: Double): Ev = e * scalar
    inline def unary_- : Ev = -e
    inline def >(other: Ev): Boolean = e > other
    inline def <(other: Ev): Boolean = e < other
    inline def >=(other: Ev): Boolean = e >= other
    inline def <=(other: Ev): Boolean = e <= other
    inline def abs: Ev = math.abs(e)
```

- [ ] **Step 2: Verify compilation**

Run: `sbt compile`
Expected: PASS (opaque types compile with no tests yet)

- [ ] **Step 3: Commit**

---

### Task 2: TableStructure

**Files:**
- Create: `src/main/scala/sicfun/holdem/strategic/TableStructure.scala`
- Create: `src/test/scala/sicfun/holdem/strategic/TableStructureTest.scala`

- [ ] **Step 1: Write the failing test**

```scala
package sicfun.holdem.strategic

import sicfun.holdem.types.Position

class TableStructureTest extends munit.FunSuite:

  // -- PlayerId --

  test("PlayerId wraps and unwraps string"):
    val pid = PlayerId("alice")
    assertEquals(pid.value, "alice")

  test("PlayerId equality is structural"):
    assertEquals(PlayerId("bob"), PlayerId("bob"))
    assertNotEquals(PlayerId("bob"), PlayerId("carol"))

  // -- SeatStatus --

  test("SeatStatus has exactly four variants"):
    assertEquals(SeatStatus.values.length, 4)

  // -- RivalMap invariant: never empty (Law L1) --

  test("RivalMap rejects empty vector"):
    intercept[IllegalArgumentException]:
      RivalMap[Chips](Vector.empty)

  test("RivalMap rejects duplicate PlayerId"):
    val pid = PlayerId("duped")
    intercept[IllegalArgumentException]:
      RivalMap(Vector(
        Seat(pid, Position.BigBlind, SeatStatus.Active, Chips(100.0)),
        Seat(pid, Position.UTG, SeatStatus.Active, Chips(200.0))
      ))

  test("RivalMap accepts single rival (heads-up is |R|=1)"):
    val rm = RivalMap(Vector(
      Seat(PlayerId("v1"), Position.BigBlind, SeatStatus.Active, Chips(100.0))
    ))
    assertEquals(rm.size, 1)

  test("RivalMap accepts two rivals (multiway)"):
    val rm = RivalMap(Vector(
      Seat(PlayerId("v1"), Position.BigBlind, SeatStatus.Active, Chips(100.0)),
      Seat(PlayerId("v2"), Position.UTG, SeatStatus.Active, Chips(200.0))
    ))
    assertEquals(rm.size, 2)

  test("RivalMap.get finds rival by PlayerId"):
    val pid = PlayerId("target")
    val rm = RivalMap(Vector(
      Seat(PlayerId("other"), Position.BigBlind, SeatStatus.Active, Chips(100.0)),
      Seat(pid, Position.UTG, SeatStatus.Active, Chips(200.0))
    ))
    assertEquals(rm.get(pid).map(_.data), Some(Chips(200.0)))
    assertEquals(rm.get(PlayerId("missing")), None)

  test("RivalMap.mapData transforms data preserving identity and position"):
    val rm = RivalMap(Vector(
      Seat(PlayerId("v1"), Position.BigBlind, SeatStatus.Active, Chips(100.0))
    ))
    val doubled = rm.mapData(c => Chips(c.value * 2))
    assertEquals(doubled.rivals.head.data, Chips(200.0))
    assertEquals(doubled.rivals.head.playerId, PlayerId("v1"))
    assertEquals(doubled.rivals.head.position, Position.BigBlind)

  // -- TableMap --

  test("TableMap rejects when hero is not in seats"):
    intercept[IllegalArgumentException]:
      TableMap(
        hero = PlayerId("hero"),
        seats = Vector(
          Seat(PlayerId("v1"), Position.SmallBlind, SeatStatus.Active, Chips(100.0))
        )
      )

  test("TableMap rejects duplicate PlayerId"):
    val pid = PlayerId("duped")
    intercept[IllegalArgumentException]:
      TableMap(
        hero = pid,
        seats = Vector(
          Seat(pid, Position.SmallBlind, SeatStatus.Active, Chips(100.0)),
          Seat(pid, Position.BigBlind, SeatStatus.Active, Chips(200.0))
        )
      )

  test("TableMap.toRivalMap excludes hero"):
    val hero = PlayerId("hero")
    val tm = TableMap(
      hero = hero,
      seats = Vector(
        Seat(hero, Position.SmallBlind, SeatStatus.Active, Chips(100.0)),
        Seat(PlayerId("v1"), Position.BigBlind, SeatStatus.Active, Chips(200.0)),
        Seat(PlayerId("v2"), Position.UTG, SeatStatus.Folded, Chips(50.0))
      )
    )
    val rm = tm.toRivalMap
    assertEquals(rm.size, 2)
    assert(rm.rivals.forall(_.playerId != hero))

  test("TableMap.activeRivalMap excludes folded and sitting-out players"):
    val hero = PlayerId("hero")
    val tm = TableMap(
      hero = hero,
      seats = Vector(
        Seat(hero, Position.SmallBlind, SeatStatus.Active, Chips(100.0)),
        Seat(PlayerId("v1"), Position.BigBlind, SeatStatus.Active, Chips(200.0)),
        Seat(PlayerId("v2"), Position.UTG, SeatStatus.Folded, Chips(50.0)),
        Seat(PlayerId("v3"), Position.UTG1, SeatStatus.AllIn, Chips(0.0)),
        Seat(PlayerId("v4"), Position.UTG2, SeatStatus.SittingOut, Chips(300.0))
      )
    )
    val active = tm.activeRivalMap
    assertEquals(active.size, 2) // v1 (Active) + v3 (AllIn)
    assert(active.get(PlayerId("v1")).isDefined)
    assert(active.get(PlayerId("v3")).isDefined)
    assert(active.get(PlayerId("v2")).isEmpty) // Folded
    assert(active.get(PlayerId("v4")).isEmpty) // SittingOut

  test("Heads-up table: activeRivalMap has exactly one rival"):
    val hero = PlayerId("hero")
    val tm = TableMap(
      hero = hero,
      seats = Vector(
        Seat(hero, Position.SmallBlind, SeatStatus.Active, Chips(100.0)),
        Seat(PlayerId("v1"), Position.BigBlind, SeatStatus.Active, Chips(200.0))
      )
    )
    assertEquals(tm.activeRivalMap.size, 1)
```

- [ ] **Step 2: Run test -- expect compilation failure (RED)**

Run: `sbt "testOnly sicfun.holdem.strategic.TableStructureTest"`
Expected: Compilation error (types do not exist yet)

- [ ] **Step 3: Write the implementation**

```scala
package sicfun.holdem.strategic

import sicfun.holdem.types.Position

/** Opaque player identity. Wraps a string handle.
  * Identity lives in the outer container (TableMap/RivalMap), not
  * inside contained model objects like OpponentModelState.
  */
opaque type PlayerId = String
object PlayerId:
  inline def apply(id: String): PlayerId = id
  extension (pid: PlayerId)
    inline def value: String = pid

/** Status of a seat at the table. */
enum SeatStatus:
  case Active, Folded, AllIn, SittingOut

/** A single seat at the table, parameterized by the data carried per seat.
  *
  * @tparam A the type of data associated with each seat (e.g. Chips for stacks,
  *           OpponentModelState for opponent models)
  */
final case class Seat[+A](
    playerId: PlayerId,
    position: Position,
    status: SeatStatus,
    data: A
)

/** Full table: includes hero. Preserves positional order.
  *
  * Invariants:
  * - hero always present in seats
  * - PlayerId uniqueness across all seats
  */
final case class TableMap[A](
    hero: PlayerId,
    seats: Vector[Seat[A]]
):
  require(
    seats.exists(_.playerId == hero),
    s"hero ${hero.value} must be present in seats"
  )
  require(
    seats.map(_.playerId).distinct.length == seats.length,
    "PlayerId must be unique across seats"
  )

  /** Extract rivals by excluding hero. Does NOT filter by status. */
  def toRivalMap: RivalMap[A] =
    RivalMap(seats.filter(_.playerId != hero))

  /** Extract active (Active + AllIn) rivals only. */
  def activeRivalMap: RivalMap[A] =
    RivalMap(
      seats.filter(s =>
        s.playerId != hero &&
          (s.status == SeatStatus.Active || s.status == SeatStatus.AllIn)
      )
    )

/** Rivals only: NEVER includes hero.
  *
  * Law L1 (cardinality): |rivals| >= 1. Heads-up is == 1.
  * PlayerId uniqueness enforced.
  */
final case class RivalMap[+A](rivals: Vector[Seat[A]]):
  require(rivals.nonEmpty, "RivalMap must contain at least one rival (L1)")
  require(
    rivals.map(_.playerId).distinct.length == rivals.length,
    "PlayerId must be unique across rivals"
  )

  /** Look up a rival by PlayerId. */
  def get(pid: PlayerId): Option[Seat[A]] =
    rivals.find(_.playerId == pid)

  /** Map the data component of each seat, preserving identity and position. */
  def mapData[B](f: A => B): RivalMap[B] =
    RivalMap(rivals.map(s => s.copy(data = f(s.data))))

  /** Number of rivals. */
  inline def size: Int = rivals.length
```

- [ ] **Step 4: Run test -- expect all pass (GREEN)**

Run: `sbt "testOnly sicfun.holdem.strategic.TableStructureTest"`
Expected: All tests pass

- [ ] **Step 5: Commit**

---

### Task 3: Fidelity

**Files:**
- Create: `src/main/scala/sicfun/holdem/strategic/Fidelity.scala`
- (No dedicated test -- BridgeResult is exercised in Phase 4 bridge tests)

- [ ] **Step 1: Write the implementation**

```scala
package sicfun.holdem.strategic

/** Fidelity of a formal-to-engine correspondence.
  *
  * Each bridge declares one of these to document how faithfully the
  * engine can represent the formal object.
  */
enum Fidelity:
  case Exact
  case Approximate
  case Absent

/** Impact of fidelity loss on the formal model's decision quality. */
enum Severity:
  case Cosmetic    // loss does not affect decisions
  case Behavioral  // affects decision quality but not reasoning structure
  case Structural  // affects reasoning structure or model coherence
  case Critical    // invalidates a semantic law or makes a canonical operator inoperable

/** Result of a bridge conversion, carrying fidelity information with the value.
  *
  * A BridgeResult.Approximate or BridgeResult.Absent NEVER returns a bare
  * canonical object. The loss travels with the value.
  *
  * Consumption safety: consuming Approximate or Absent as if Exact without
  * explicit check is prohibited. Enforcement via fold.
  */
sealed trait BridgeResult[+A]:
  def fidelity: Fidelity

  /** Safe consumption via fold. Forces callers to handle all cases. */
  def fold[B](
      onExact: A => B,
      onApprox: (A, String) => B,
      onAbsent: String => B
  ): B = this match
    case BridgeResult.Exact(value) => onExact(value)
    case BridgeResult.Approximate(value, loss) => onApprox(value, loss)
    case BridgeResult.Absent(reason) => onAbsent(reason)

object BridgeResult:
  final case class Exact[A](value: A) extends BridgeResult[A]:
    def fidelity: Fidelity = Fidelity.Exact

  final case class Approximate[A](value: A, lossDescription: String) extends BridgeResult[A]:
    def fidelity: Fidelity = Fidelity.Approximate

  final case class Absent(reason: String) extends BridgeResult[Nothing]:
    def fidelity: Fidelity = Fidelity.Absent
```

- [ ] **Step 2: Verify compilation**

Run: `sbt compile`
Expected: PASS

- [ ] **Step 3: Commit**

---

### Task 4: StrategicClass (Defs 1-4)

**Files:**
- Create: `src/main/scala/sicfun/holdem/strategic/StrategicClass.scala`
- Create: `src/test/scala/sicfun/holdem/strategic/StrategicClassTest.scala`

- [ ] **Step 1: Write the failing test**

```scala
package sicfun.holdem.strategic

import sicfun.holdem.types.PokerAction

class StrategicClassTest extends munit.FunSuite:

  // -- Def 1: Exhaustive partition C^S = C^V | C^B | C^M | C^SB --

  test("StrategicClass has exactly four members"):
    assertEquals(StrategicClass.values.length, 4)

  test("StrategicClass values are Value, Bluff, Marginal, SemiBluff"):
    val names = StrategicClass.values.map(_.toString).toSet
    assertEquals(names, Set("Value", "Bluff", "Marginal", "SemiBluff"))

  // -- Def 3: Aggressive-wager predicate Agg(u) --

  test("isAggressiveWager true for Raise"):
    assert(StrategicClass.isAggressiveWager(PokerAction.Raise(50.0)))

  test("isAggressiveWager false for Fold, Check, Call"):
    assert(!StrategicClass.isAggressiveWager(PokerAction.Fold))
    assert(!StrategicClass.isAggressiveWager(PokerAction.Check))
    assert(!StrategicClass.isAggressiveWager(PokerAction.Call))

  // -- Def 4: StructuralBluff(c, u) = (c in C^B) AND Agg(u) --

  test("structuralBluff requires Bluff class AND aggressive wager"):
    assert(StrategicClass.isStructuralBluff(StrategicClass.Bluff, PokerAction.Raise(100.0)))

  test("structuralBluff false when class is not Bluff"):
    assert(!StrategicClass.isStructuralBluff(StrategicClass.Value, PokerAction.Raise(100.0)))
    assert(!StrategicClass.isStructuralBluff(StrategicClass.Marginal, PokerAction.Raise(50.0)))
    assert(!StrategicClass.isStructuralBluff(StrategicClass.SemiBluff, PokerAction.Raise(50.0)))

  test("structuralBluff false when action is not aggressive"):
    assert(!StrategicClass.isStructuralBluff(StrategicClass.Bluff, PokerAction.Call))
    assert(!StrategicClass.isStructuralBluff(StrategicClass.Bluff, PokerAction.Check))
    assert(!StrategicClass.isStructuralBluff(StrategicClass.Bluff, PokerAction.Fold))

  test("structuralBluff exhaustive: false for all non-Bluff classes regardless of action"):
    val nonBluff = Seq(StrategicClass.Value, StrategicClass.Marginal, StrategicClass.SemiBluff)
    val actions = Seq(PokerAction.Fold, PokerAction.Check, PokerAction.Call, PokerAction.Raise(99.0))
    for
      cls <- nonBluff
      act <- actions
    do assert(!StrategicClass.isStructuralBluff(cls, act), s"Expected false for $cls + $act")

  // -- Derived: exploitative bluff implies structural bluff (Law L2) --

  test("isExploitativeBluff requires Bluff + Raise + positive deltaManip"):
    assert(StrategicClass.isExploitativeBluff(StrategicClass.Bluff, PokerAction.Raise(50.0), Ev(1.0)))
    assert(!StrategicClass.isExploitativeBluff(StrategicClass.Bluff, PokerAction.Raise(50.0), Ev(0.0)))
    assert(!StrategicClass.isExploitativeBluff(StrategicClass.Bluff, PokerAction.Raise(50.0), Ev(-0.5)))
    assert(!StrategicClass.isExploitativeBluff(StrategicClass.Bluff, PokerAction.Call, Ev(1.0)))
    assert(!StrategicClass.isExploitativeBluff(StrategicClass.Value, PokerAction.Raise(50.0), Ev(1.0)))

  test("Law L2: isExploitativeBluff implies isStructuralBluff for all inputs"):
    val classes = StrategicClass.values.toSeq
    val actions = Seq(PokerAction.Fold, PokerAction.Check, PokerAction.Call, PokerAction.Raise(50.0))
    val manipValues = Seq(Ev(-1.0), Ev(0.0), Ev(0.5), Ev(10.0))
    for
      cls <- classes
      act <- actions
      dManip <- manipValues
    do
      if StrategicClass.isExploitativeBluff(cls, act, dManip) then
        assert(
          StrategicClass.isStructuralBluff(cls, act),
          s"L2 violated: exploitative($cls, $act, ${dManip.value}) but not structural"
        )
```

- [ ] **Step 2: Run test -- expect compilation failure (RED)**

Run: `sbt "testOnly sicfun.holdem.strategic.StrategicClassTest"`
Expected: Compilation error (StrategicClass does not exist yet)

- [ ] **Step 3: Write the implementation**

```scala
package sicfun.holdem.strategic

import sicfun.holdem.types.PokerAction

/** Private strategic classes of SICFUN (Def 1).
  *
  * C^S = C^V | C^B | C^M | C^SB
  *
  * Exhaustive partition: every hand in a spot belongs to exactly one class.
  * Classification depends on spot -- it is NOT a static property of the hand.
  */
enum StrategicClass:
  case Value     // C^V: hands played for value
  case Bluff     // C^B: hands played as bluffs
  case Marginal  // C^M: hands with marginal equity
  case SemiBluff // C^SB: draws played aggressively

object StrategicClass:

  /** Aggressive-wager predicate (Def 3).
    * Agg: U -> {0,1}
    * True iff the action involves a bet/raise.
    */
  def isAggressiveWager(action: PokerAction): Boolean = action match
    case PokerAction.Raise(_) => true
    case _                    => false

  /** Structural-bluff predicate (Def 4).
    * StructuralBluff(c, u) = 1 iff (c in C^B) AND Agg(u) = 1.
    */
  def isStructuralBluff(cls: StrategicClass, action: PokerAction): Boolean =
    cls == StrategicClass.Bluff && isAggressiveWager(action)

  /** Exploitative-bluff predicate (derived from Def 4 + Def 42).
    * Exploitative bluff iff structural bluff AND deltaManip > 0.
    * Law L2: isExploitativeBluff -> isStructuralBluff (always).
    */
  def isExploitativeBluff(cls: StrategicClass, action: PokerAction, deltaManip: Ev): Boolean =
    isStructuralBluff(cls, action) && deltaManip > Ev.Zero
```

- [ ] **Step 4: Run test -- expect all pass (GREEN)**

Run: `sbt "testOnly sicfun.holdem.strategic.StrategicClassTest"`
Expected: All tests pass

- [ ] **Step 5: Commit**

---

### Task 5: Signal (Defs 5-8)

**Files:**
- Create: `src/main/scala/sicfun/holdem/strategic/Signal.scala`
- Create: `src/test/scala/sicfun/holdem/strategic/SignalTest.scala`

- [ ] **Step 1: Write the failing test**

```scala
package sicfun.holdem.strategic

import sicfun.holdem.types.Street

class SignalTest extends munit.FunSuite:

  // -- Sizing --

  test("Sizing stores absolute and pot-fraction"):
    val s = Sizing(Chips(100.0), PotFraction(0.5))
    assertEquals(s.absolute.value, 100.0)
    assertEquals(s.fractionOfPot.value, 0.5)

  // -- Def 5: ActionSignal Y_t^act = (a_t, lambda_t, tau_t) --

  test("ActionSignal with sizing marks isAggressiveWager true"):
    val sig = ActionSignal(
      action = sicfun.holdem.types.PokerAction.Category.Raise,
      sizing = Some(Sizing(Chips(100.0), PotFraction(0.5))),
      timing = None,
      stage = Street.Flop
    )
    assert(sig.isAggressiveWager)

  test("ActionSignal without sizing marks isAggressiveWager false"):
    val sig = ActionSignal(
      action = sicfun.holdem.types.PokerAction.Category.Call,
      sizing = None,
      timing = None,
      stage = Street.Preflop
    )
    assert(!sig.isAggressiveWager)

  test("isAggressiveWager is biconditional with sizing.isDefined"):
    val withSizing = ActionSignal(
      action = sicfun.holdem.types.PokerAction.Category.Raise,
      sizing = Some(Sizing(Chips(50.0), PotFraction(0.75))),
      timing = None,
      stage = Street.Turn
    )
    val withoutSizing = ActionSignal(
      action = sicfun.holdem.types.PokerAction.Category.Fold,
      sizing = None,
      timing = None,
      stage = Street.Preflop
    )
    assertEquals(withSizing.isAggressiveWager, withSizing.sizing.isDefined)
    assertEquals(withoutSizing.isAggressiveWager, withoutSizing.sizing.isDefined)

  // -- Def 6: TotalSignal Y_t = (Y_t^act, Y_t^sd) --

  test("TotalSignal with empty showdown"):
    val act = ActionSignal(
      action = sicfun.holdem.types.PokerAction.Category.Check,
      sizing = None,
      timing = None,
      stage = Street.River
    )
    val total = TotalSignal(act, showdown = None)
    assert(total.showdown.isEmpty)

  test("TotalSignal with showdown data"):
    val act = ActionSignal(
      action = sicfun.holdem.types.PokerAction.Category.Call,
      sizing = None,
      timing = None,
      stage = Street.River
    )
    val sd = ShowdownSignal(
      revealedHands = Vector(
        RevealedHand(PlayerId("v1"), Vector.empty)
      )
    )
    val total = TotalSignal(act, showdown = Some(sd))
    assert(total.showdown.isDefined)
    assertEquals(total.showdown.get.revealedHands.length, 1)

  // -- Def 8: Signal routing convention --

  test("actionChannel returns action signal"):
    val act = ActionSignal(
      action = sicfun.holdem.types.PokerAction.Category.Raise,
      sizing = Some(Sizing(Chips(200.0), PotFraction(1.0))),
      timing = None,
      stage = Street.Flop
    )
    val total = TotalSignal(act, showdown = None)
    assertEquals(total.actionChannel, act)

  test("revelationChannel returns showdown when present"):
    val act = ActionSignal(
      action = sicfun.holdem.types.PokerAction.Category.Call,
      sizing = None,
      timing = None,
      stage = Street.River
    )
    val sd = ShowdownSignal(revealedHands = Vector.empty)
    val total = TotalSignal(act, showdown = Some(sd))
    assertEquals(total.revelationChannel, Some(sd))

  test("revelationChannel returns None when no showdown"):
    val act = ActionSignal(
      action = sicfun.holdem.types.PokerAction.Category.Check,
      sizing = None,
      timing = None,
      stage = Street.Flop
    )
    val total = TotalSignal(act, showdown = None)
    assertEquals(total.revelationChannel, None)

  // -- TimingBucket --

  test("TimingBucket has exactly four variants"):
    assertEquals(TimingBucket.values.length, 4)
```

- [ ] **Step 2: Run test -- expect compilation failure (RED)**

Run: `sbt "testOnly sicfun.holdem.strategic.SignalTest"`
Expected: Compilation error

- [ ] **Step 3: Write the implementation**

```scala
package sicfun.holdem.strategic

import sicfun.holdem.types.{PokerAction, Street}

/** Bet sizing with both absolute and pot-relative representations. */
final case class Sizing(
    absolute: Chips,
    fractionOfPot: PotFraction
)

/** Timing bucket for action speed. Currently a placeholder --
  * no timing data is collected in the existing engine (bridge: Absent).
  */
enum TimingBucket:
  case Fast, Normal, Slow, VeryLong

/** A revealed hand at showdown, attributed to a player. */
final case class RevealedHand(
    playerId: PlayerId,
    cards: Vector[sicfun.core.Card]
)

/** Size-aware public action signal (Def 5).
  *
  * Y_t^act = (a_t, lambda_t, tau_t), where:
  * - a_t is the coarsened action category
  * - lambda_t is the optional sizing (Some iff aggressive wager)
  * - tau_t is the stage (street)
  *
  * Invariant: isAggressiveWager iff sizing.isDefined (biconditional).
  */
final case class ActionSignal(
    action: PokerAction.Category,
    sizing: Option[Sizing],
    timing: Option[TimingBucket],
    stage: Street
):
  /** True iff this action involves a bet/raise with a sizing component. */
  inline def isAggressiveWager: Boolean = sizing.isDefined

/** Showdown signal (component of Def 6).
  * Y_t^sd = empty if no terminal revelation occurs at t.
  */
final case class ShowdownSignal(
    revealedHands: Vector[RevealedHand]
)

/** Total public signal (Def 6).
  * Y_t = (Y_t^act, Y_t^sd), where Y_t^sd = None if no showdown.
  *
  * Def 7 (canonical identification): Z_{t+1} := Y_t.
  *
  * Def 8 (signal routing):
  * - actionChannel: Y_t^act enters the inferential channel (tempered likelihood)
  * - revelationChannel: Y_t^sd enters the revelatory channel (showdown update)
  */
final case class TotalSignal(
    actionSignal: ActionSignal,
    showdown: Option[ShowdownSignal]
):
  /** Def 8: Y_t^act enters the inferential channel. */
  inline def actionChannel: ActionSignal = actionSignal

  /** Def 8: Y_t^sd enters the revelatory channel (None if no showdown). */
  inline def revelationChannel: Option[ShowdownSignal] = showdown
```

- [ ] **Step 4: Run test -- expect all pass (GREEN)**

Run: `sbt "testOnly sicfun.holdem.strategic.SignalTest"`
Expected: All tests pass

- [ ] **Step 5: Commit**

---

### Task 6: Baseline (Defs 9-10)

**Files:**
- Create: `src/main/scala/sicfun/holdem/strategic/Baseline.scala`
- (No dedicated test -- traits tested through concrete implementations in Phase 4)

- [ ] **Step 1: Write the implementation**

```scala
package sicfun.holdem.strategic

import sicfun.holdem.types.{PokerAction, Street}

/** Real baseline of SICFUN (Def 9).
  *
  * pi^{0,S}(a, lambda | c, x_t^pub)
  *
  * The real baseline is SICFUN's actual default policy -- what it would
  * play without exploitation-motivated deviations. This is singular
  * (one per SICFUN instance), not per-rival.
  */
trait RealBaseline:
  /** Probability of action+sizing given strategic class and public state context.
    *
    * @param cls the strategic class of the hand
    * @param action the coarsened action category
    * @param sizing optional sizing (for raises)
    * @param street current street
    * @return probability in [0,1]
    */
  def probability(
      cls: StrategicClass,
      action: PokerAction.Category,
      sizing: Option[Sizing],
      street: Street
  ): Double

/** Attributed baseline -- per-rival, state-conditioned (Def 10).
  *
  * hat{pi}^{0,S,i}(a, lambda | c, x^pub, m^{R,i})
  *
  * What rival i believes SICFUN's baseline policy to be. This is per-rival
  * because different rivals may have different models of SICFUN.
  * State-conditioned because the attribution depends on the rival's
  * mental state m^{R,i}.
  */
trait AttributedBaseline:
  /** Probability under rival i's attributed model of SICFUN.
    *
    * @param cls the strategic class
    * @param action the coarsened action category
    * @param sizing optional sizing
    * @param street current street
    * @return probability in [0,1]
    */
  def probability(
      cls: StrategicClass,
      action: PokerAction.Category,
      sizing: Option[Sizing],
      street: Street
  ): Double
```

- [ ] **Step 2: Verify compilation**

Run: `sbt compile`
Expected: PASS

- [ ] **Step 3: Commit**

---

### Task 7: ReputationalProjection (Def 11)

**Files:**
- Create: `src/main/scala/sicfun/holdem/strategic/ReputationalProjection.scala`
- (No dedicated test -- trait tested through concrete implementations in Phase 4)

- [ ] **Step 1: Write the implementation**

```scala
package sicfun.holdem.strategic

/** View of SICFUN's reputation as perceived by a specific rival.
  *
  * This is derived, never stored as independent state.
  * Law L4: m1 == m2 -> project(m1) == project(m2).
  * Law L5: In multiway, each rival may have a different image of SICFUN.
  */
final case class ReputationView(
    perceivedTightness: PotFraction,
    perceivedAggression: PotFraction,
    perceivedBluffFrequency: PotFraction,
    raw: Map[String, Double] = Map.empty
)

/** Reputational projection (Def 11).
  *
  * phi_t^{S,i} := g^i(m_t^{R,i})
  *
  * Law L4 (derivability): Reputation is NEVER stored independently.
  * Always computed from the rival's mental state.
  * Same input -> same output.
  *
  * Law L5 (per-rival): No global phi^S exists unless explicitly
  * introduced as a named aggregation.
  */
trait ReputationalProjection:
  /** Derive SICFUN's reputation from a rival's mental state.
    * Must be a pure function: same input -> same output (Law L4).
    */
  def project(rivalState: RivalBeliefState): ReputationView
```

- [ ] **Step 2: Verify compilation**

Run: `sbt compile`
Expected: PASS

- [ ] **Step 3: Commit**

---

### Task 8: AugmentedState (Defs 12-14)

**Files:**
- Create: `src/main/scala/sicfun/holdem/strategic/AugmentedState.scala`
- Create: `src/test/scala/sicfun/holdem/strategic/AugmentedStateTest.scala`

- [ ] **Step 1: Write the failing test**

```scala
package sicfun.holdem.strategic

import sicfun.holdem.types.{Position, Street, Board, HoleCards}
import sicfun.core.{Card, DiscreteDistribution}

class AugmentedStateTest extends munit.FunSuite:

  private val hero = PlayerId("hero")
  private val v1 = PlayerId("v1")
  private val v2 = PlayerId("v2")

  private def mkPublicState(nRivals: Int): PublicState =
    val rivalSeats = (1 to nRivals).map { i =>
      val pos = Position.values(i % Position.values.length)
      Seat(PlayerId(s"v$i"), pos, SeatStatus.Active, Chips(500.0))
    }.toVector
    PublicState(
      street = Street.Flop,
      board = Board.empty,
      pot = Chips(100.0),
      stacks = TableMap(
        hero = hero,
        seats = Seat(hero, Position.SmallBlind, SeatStatus.Active, Chips(500.0)) +: rivalSeats
      ),
      actionHistory = Vector.empty
    )

  private val dummyHoleCards = HoleCards.canonical(Card(0), Card(1))

  private val dummyBeliefState = new RivalBeliefState:
    def update(signal: ActionSignal, publicState: PublicState): RivalBeliefState = this

  private val dummyOMS = OpponentModelState(
    typeDistribution = DiscreteDistribution.uniform(Seq("TAG", "LAG")),
    beliefState = dummyBeliefState,
    attributedBaseline = None
  )

  // -- PublicState invariants --

  test("PublicState pot must be non-negative"):
    intercept[IllegalArgumentException]:
      PublicState(
        street = Street.Preflop,
        board = Board.empty,
        pot = Chips(-1.0),
        stacks = TableMap(
          hero = hero,
          seats = Vector(
            Seat(hero, Position.SmallBlind, SeatStatus.Active, Chips(100.0)),
            Seat(v1, Position.BigBlind, SeatStatus.Active, Chips(100.0))
          )
        ),
        actionHistory = Vector.empty
      )

  test("PublicState is an own type, not GameState"):
    // PublicState uses PlayerId (not Int seat index) and Chips (not raw Double).
    val ps = mkPublicState(1)
    assertEquals(ps.pot, Chips(100.0))

  // -- Def 12-13: AugmentedState --

  test("AugmentedState stores public, private, opponents, ownEvidence"):
    val opponents = RivalMap(Vector(
      Seat(v1, Position.BigBlind, SeatStatus.Active, dummyOMS),
      Seat(v2, Position.UTG, SeatStatus.Active, dummyOMS)
    ))
    val state = AugmentedState(
      publicState = mkPublicState(2),
      privateHand = dummyHoleCards,
      opponents = opponents,
      ownEvidence = OwnEvidence.empty
    )
    assertEquals(state.opponents.size, 2)

  // -- Law L7: no singular "the opponent" field --

  test("AugmentedState is multiway-native (L7): heads-up is |R|=1, not special"):
    val huOpponents = RivalMap(Vector(
      Seat(v1, Position.BigBlind, SeatStatus.Active, dummyOMS)
    ))
    val huState = AugmentedState(
      publicState = mkPublicState(1),
      privateHand = dummyHoleCards,
      opponents = huOpponents,
      ownEvidence = OwnEvidence.empty
    )
    assertEquals(huState.opponents.size, 1)
    // Same type works for multiway:
    val mwOpponents = RivalMap(Vector(
      Seat(v1, Position.BigBlind, SeatStatus.Active, dummyOMS),
      Seat(v2, Position.UTG, SeatStatus.Active, dummyOMS)
    ))
    val mwState = huState.copy(opponents = mwOpponents, publicState = mkPublicState(2))
    assertEquals(mwState.opponents.size, 2)

  // -- Def 14: OperativeBelief --

  test("OperativeBelief wraps DiscreteDistribution over AugmentedState"):
    val opponents = RivalMap(Vector(
      Seat(v1, Position.BigBlind, SeatStatus.Active, dummyOMS)
    ))
    val state1 = AugmentedState(
      publicState = mkPublicState(1),
      privateHand = dummyHoleCards,
      opponents = opponents,
      ownEvidence = OwnEvidence.empty
    )
    val belief = OperativeBelief(DiscreteDistribution(Map(state1 -> 1.0)))
    assertEquals(belief.distribution.probabilityOf(state1), 1.0)

  // -- OwnEvidence --

  test("OwnEvidence.empty has no data"):
    val oe = OwnEvidence.empty
    assert(oe.globalSummary.isEmpty)
    assert(oe.perRivalSummary.isEmpty)
    assert(oe.relationalSummary.isEmpty)

  // -- OpponentModelState --

  test("OpponentModelState has no playerId field (identity in RivalMap)"):
    // Structural: OpponentModelState fields are typeDistribution, beliefState, attributedBaseline
    val oms = dummyOMS
    assertEquals(oms.typeDistribution.support.size, 2)
    assertEquals(oms.attributedBaseline, None)
```

- [ ] **Step 2: Run test -- expect compilation failure (RED)**

Run: `sbt "testOnly sicfun.holdem.strategic.AugmentedStateTest"`
Expected: Compilation error

- [ ] **Step 3: Write the implementation**

```scala
package sicfun.holdem.strategic

import sicfun.holdem.types.{Board, HoleCards, Street}
import sicfun.core.DiscreteDistribution

/** A public action in the history, attributed to a player identity. */
final case class PublicAction(
    actor: PlayerId,
    signal: ActionSignal
)

/** Public state as defined in the formal layer (component of Defs 12-13).
  *
  * This is an OWN TYPE of strategic. NOT an alias of GameState.
  * Uses PlayerId (not Int seat index), Chips (not raw Double).
  * The conversion GameState -> PublicState lives in bridge (Phase 4).
  */
final case class PublicState(
    street: Street,
    board: Board,
    pot: Chips,
    stacks: TableMap[Chips],
    actionHistory: Vector[PublicAction]
):
  require(pot >= Chips(0.0), "pot must be non-negative")

/** Per-rival mental state (A6: first-order interactive sufficiency).
  *
  * m_t^{R,i} in M^{R,i}: future policy of rival i depends on public
  * history only through this state.
  */
trait RivalBeliefState:
  /** Deterministic update: m_{t+1} = Gamma^R(m_t, Y_t^act, x_t^pub).
    * Must be a pure function -- no external mutable state (Law L12).
    */
  def update(signal: ActionSignal, publicState: PublicState): RivalBeliefState

/** Per-rival opponent model state.
  *
  * Key decision: playerId is NOT a field here. Identity lives in the
  * outer container (RivalMap), not in the contained object.
  *
  * - typeDistribution: theta^{R,i} -- posterior over rival type space
  * - beliefState: m^{R,i} -- sufficient statistic for rival's future policy
  * - attributedBaseline: hat{pi}^{0,S,i} -- what rival believes about us (None until Phase 4)
  */
final case class OpponentModelState(
    typeDistribution: DiscreteDistribution[String],
    beliefState: RivalBeliefState,
    attributedBaseline: Option[AttributedBaseline]
)

/** Own statistical sufficiency xi^S (A4').
  *
  * Contains accumulated evidence NOT in PublicState:
  * - private information (past hole cards, private Bayesian inferences)
  * - own historical summaries
  * - per-rival evidence
  * - relational evidence (cross-rival)
  *
  * Law L6: Does NOT duplicate board/pot/stacks from PublicState.
  */
final case class OwnEvidence(
    globalSummary: Map[String, Double],
    perRivalSummary: Map[PlayerId, Map[String, Double]],
    relationalSummary: Map[(PlayerId, PlayerId), Map[String, Double]]
)

object OwnEvidence:
  val empty: OwnEvidence = OwnEvidence(
    globalSummary = Map.empty,
    perRivalSummary = Map.empty,
    relationalSummary = Map.empty
  )

/** Augmented hidden state (Defs 12-13).
  *
  * X_tilde_t = (x_t^pub, x_t^priv, {theta_t^{R,i}, m_t^{R,i}}_{i in R}, xi_t^S)
  *
  * Law L7 (multiway-native): No singular "the opponent" field.
  * Everything is an indexed family via RivalMap. Heads-up is |opponents| == 1.
  *
  * Law L8 (no implicit collapse): No operator on AugmentedState may
  * silently reduce the rival family to a single opponent.
  *
  * Does NOT contain phi^{S,i} as a field -- it is derived via Def 11.
  */
final case class AugmentedState(
    publicState: PublicState,
    privateHand: HoleCards,
    opponents: RivalMap[OpponentModelState],
    ownEvidence: OwnEvidence
)

/** Operative belief (Def 14).
  *
  * b_tilde_t in Delta(X_tilde)
  *
  * The agent's belief over the augmented hidden state space.
  */
final case class OperativeBelief(
    distribution: DiscreteDistribution[AugmentedState]
)
```

- [ ] **Step 4: Run test -- expect all pass (GREEN)**

Run: `sbt "testOnly sicfun.holdem.strategic.AugmentedStateTest"`
Expected: All tests pass

- [ ] **Step 5: Commit**

---

### Task 9: RivalKernel (Defs 16-21 type signatures)

**Files:**
- Create: `src/main/scala/sicfun/holdem/strategic/RivalKernel.scala`
- Create: `src/test/scala/sicfun/holdem/strategic/RivalKernelLawTest.scala`

- [ ] **Step 1: Write the failing test**

```scala
package sicfun.holdem.strategic

import sicfun.holdem.types.{Position, Street, Board}
import sicfun.core.DiscreteDistribution

class RivalKernelLawTest extends munit.FunSuite:

  private val dummyPublicState = PublicState(
    street = Street.River,
    board = Board.empty,
    pot = Chips(100.0),
    stacks = TableMap(
      hero = PlayerId("hero"),
      seats = Vector(
        Seat(PlayerId("hero"), Position.SmallBlind, SeatStatus.Active, Chips(500.0)),
        Seat(PlayerId("v1"), Position.BigBlind, SeatStatus.Active, Chips(500.0))
      )
    ),
    actionHistory = Vector.empty
  )

  private val dummySignal = ActionSignal(
    action = sicfun.holdem.types.PokerAction.Category.Raise,
    sizing = Some(Sizing(Chips(100.0), PotFraction(0.5))),
    timing = None,
    stage = Street.Flop
  )

  // -- Def 16: StateEmbeddingUpdater --

  test("StateEmbeddingUpdater type alias exists and compiles"):
    val updater: StateEmbeddingUpdater[RivalBeliefState] =
      (state: RivalBeliefState, posterior: DiscreteDistribution[StrategicClass]) => state
    assert(updater != null)

  // -- Def 18: Blind kernel = identity --

  test("BlindActionKernel returns exact same state object"):
    val belief = new RivalBeliefState:
      def update(signal: ActionSignal, publicState: PublicState): RivalBeliefState = this
    val kernel = BlindActionKernel[RivalBeliefState]()
    val result = kernel.apply(belief, dummySignal)
    assert(result eq belief, "Blind kernel must return the exact same state object")

  // -- Def 20: FullKernel trait --

  test("FullKernel trait has apply(state, totalSignal, publicState)"):
    val dummy = new FullKernel[RivalBeliefState]:
      def apply(
          state: RivalBeliefState,
          signal: TotalSignal,
          publicState: PublicState
      ): RivalBeliefState = state

    val belief = new RivalBeliefState:
      def update(signal: ActionSignal, publicState: PublicState): RivalBeliefState = this

    val totalSignal = TotalSignal(dummySignal, showdown = None)
    val result = dummy.apply(belief, totalSignal, dummyPublicState)
    assert(result eq belief)

  // -- Def 21: KernelProfile is per-rival indexed family --

  test("KernelProfile maps PlayerId to kernels"):
    val k1 = BlindActionKernel[RivalBeliefState]()
    val k2 = BlindActionKernel[RivalBeliefState]()
    val profile = KernelProfile(Map(
      PlayerId("v1") -> k1,
      PlayerId("v2") -> k2
    ))
    assertEquals(profile.kernels.size, 2)
    assert(profile.kernels.contains(PlayerId("v1")))
    assert(profile.kernels.contains(PlayerId("v2")))

  // -- Law L13: no cross-rival parameter in ActionKernel --

  test("ActionKernel.apply signature has exactly (M, ActionSignal) -- no cross-rival arg"):
    // Structural: if this compiles, L13 is enforced at the type level.
    val kernel: ActionKernel[RivalBeliefState] = BlindActionKernel[RivalBeliefState]()
    val belief = new RivalBeliefState:
      def update(signal: ActionSignal, publicState: PublicState): RivalBeliefState = this
    val _ = kernel.apply(belief, dummySignal)

  // -- KernelVariant enum --

  test("KernelVariant has exactly four variants"):
    assertEquals(KernelVariant.values.length, 4)
    val names = KernelVariant.values.map(_.toString).toSet
    assertEquals(names, Set("Ref", "Attrib", "Blind", "Design"))
```

- [ ] **Step 2: Run test -- expect compilation failure (RED)**

Run: `sbt "testOnly sicfun.holdem.strategic.RivalKernelLawTest"`
Expected: Compilation error

- [ ] **Step 3: Write the implementation**

```scala
package sicfun.holdem.strategic

import sicfun.core.DiscreteDistribution

/** State-embedding updater (Def 16).
  *
  * U^{R,i}_pi : M^{R,i} x Delta(C^S) -> M^{R,i}
  *
  * Takes the current rival mental state and the updated posterior over
  * strategic classes, and produces the new mental state.
  */
type StateEmbeddingUpdater[M] = (M, DiscreteDistribution[StrategicClass]) => M

/** Action kernel (Def 17 component -- type signature only).
  *
  * Maps (rival state, action signal) -> updated rival state.
  *
  * Law L13: The update signature does NOT accept other rivals'
  * OpponentModelState. Any dependency on other rivals must enter
  * ONLY through PublicState or an additional formalized shared state.
  */
trait ActionKernel[M]:
  /** Apply the action channel update. */
  def apply(state: M, signal: ActionSignal): M

/** Showdown kernel (Def 19 -- type signature only).
  *
  * Gamma^{sd,i} : M^{R,i} x Y^{sd} -> M^{R,i}
  *
  * Revelatory updater applied to certified showdown information.
  * Not regularized (contrast with tempered action kernels).
  */
trait ShowdownKernel[M]:
  /** Apply the revelatory showdown update. */
  def apply(state: M, showdown: ShowdownSignal): M

/** Full per-rival kernel (Def 20 -- type signature only).
  *
  * Gamma^{full,bullet,i} : M^{R,i} x Y x X^{pub} -> M^{R,i}
  *
  * Composes the action channel and showdown channel per Def 20:
  * - If Y^sd != empty and bullet != blind: showdown(action(m, Y^act), Y^sd)
  * - If Y^sd == empty and bullet != blind: action(m, Y^act)
  * - If bullet == blind: m (identity)
  */
trait FullKernel[M]:
  /** Apply the full (action + showdown) update. */
  def apply(state: M, signal: TotalSignal, publicState: PublicState): M

/** Inferential action kernel variant labels (Def 18, Def 19A).
  *
  * - Ref:    uses SICFUN's true baseline pi^{0,S}
  * - Attrib: uses rival i's attributed baseline hat{pi}^{0,S,i}
  * - Blind:  frozen rival -- ignores action-signal learning (identity)
  * - Design: uses only the action component a_t, not sizing/timing (Def 19A)
  */
enum KernelVariant:
  case Ref, Attrib, Blind, Design

/** Blind action kernel (Def 18 component).
  *
  * Gamma^{act,blind,i}(m, y, x^pub) := m
  *
  * Frozen rival: ignores action-signal learning entirely.
  */
final class BlindActionKernel[M] extends ActionKernel[M]:
  def apply(state: M, signal: ActionSignal): M = state

/** Joint kernel profile (Def 21).
  *
  * A per-rival indexed family:
  * Gamma^{attrib} := {Gamma^{full,attrib,i}}_{i in R}
  * Gamma^{ref}    := {Gamma^{full,ref,i}}_{i in R}
  * Gamma^{blind}  := {Gamma^{full,blind,i}}_{i in R}
  */
final case class KernelProfile[M](
    kernels: Map[PlayerId, ActionKernel[M]]
)
```

- [ ] **Step 4: Run test -- expect all pass (GREEN)**

Run: `sbt "testOnly sicfun.holdem.strategic.RivalKernelLawTest"`
Expected: All tests pass

- [ ] **Step 5: Commit**

---

### Task 10: StrategicValue (Defs 44-47, 50)

**Files:**
- Create: `src/main/scala/sicfun/holdem/strategic/StrategicValue.scala`
- Create: `src/test/scala/sicfun/holdem/strategic/StrategicValueTest.scala`

- [ ] **Step 1: Write the failing test**

```scala
package sicfun.holdem.strategic

class StrategicValueTest extends munit.FunSuite:

  private inline val Tol = 1e-12

  // -- Def 44: Four worlds --

  test("FourWorld stores all four value functions"):
    val fw = FourWorld(v11 = Ev(10.0), v10 = Ev(7.0), v01 = Ev(6.0), v00 = Ev(4.0))
    assertEquals(fw.v11.value, 10.0)
    assertEquals(fw.v10.value, 7.0)
    assertEquals(fw.v01.value, 6.0)
    assertEquals(fw.v00.value, 4.0)

  // -- Def 45: Control value delta_cont = V^{0,1} - V^{0,0} --

  test("deltaControl = V^{0,1} - V^{0,0}"):
    val fw = FourWorld(Ev(10.0), Ev(7.0), Ev(6.0), Ev(4.0))
    assertEqualsDouble(fw.deltaControl.value, 2.0, Tol)

  // -- Def 46: Marginal signaling effect delta_sig* = V^{1,0} - V^{0,0} --

  test("deltaSigStar = V^{1,0} - V^{0,0}"):
    val fw = FourWorld(Ev(10.0), Ev(7.0), Ev(6.0), Ev(4.0))
    assertEqualsDouble(fw.deltaSigStar.value, 3.0, Tol)

  // -- Def 47: Interaction term delta_int = V^{1,1} - V^{1,0} - V^{0,1} + V^{0,0} --

  test("deltaInteraction = V^{1,1} - V^{1,0} - V^{0,1} + V^{0,0}"):
    val fw = FourWorld(Ev(10.0), Ev(7.0), Ev(6.0), Ev(4.0))
    // 10 - 7 - 6 + 4 = 1
    assertEqualsDouble(fw.deltaInteraction.value, 1.0, Tol)

  // -- Theorem 4: V^{1,1} = V^{0,0} + delta_cont + delta_sig* + delta_int --

  test("Theorem 4: exact aggregate decomposition"):
    val fw = FourWorld(Ev(10.0), Ev(7.0), Ev(6.0), Ev(4.0))
    val reconstructed = fw.v00 + fw.deltaControl + fw.deltaSigStar + fw.deltaInteraction
    assertEqualsDouble(reconstructed.value, fw.v11.value, Tol)

  test("Theorem 4 holds for negative values"):
    val fw = FourWorld(Ev(-2.0), Ev(-5.0), Ev(-3.0), Ev(-8.0))
    val reconstructed = fw.v00 + fw.deltaControl + fw.deltaSigStar + fw.deltaInteraction
    assertEqualsDouble(reconstructed.value, fw.v11.value, Tol)

  test("Theorem 4 holds when all worlds are equal"):
    val fw = FourWorld(Ev(5.0), Ev(5.0), Ev(5.0), Ev(5.0))
    assertEqualsDouble(fw.deltaControl.value, 0.0, Tol)
    assertEqualsDouble(fw.deltaSigStar.value, 0.0, Tol)
    assertEqualsDouble(fw.deltaInteraction.value, 0.0, Tol)
    val reconstructed = fw.v00 + fw.deltaControl + fw.deltaSigStar + fw.deltaInteraction
    assertEqualsDouble(reconstructed.value, fw.v11.value, Tol)

  // -- Corollary 3: Separability --

  test("Corollary 3: if V^{1,1}-V^{1,0} = V^{0,1}-V^{0,0} then delta_int = 0"):
    // V^{1,1}-V^{1,0} = 3, V^{0,1}-V^{0,0} = 3
    val fw = FourWorld(Ev(10.0), Ev(7.0), Ev(7.0), Ev(4.0))
    assertEqualsDouble(fw.deltaInteraction.value, 0.0, Tol)

  // -- Corollary 4: Coarse interaction bound --

  test("Corollary 4: |delta_int| <= 4*R_max/(1-gamma)"):
    val rMax = 100.0
    val gamma = 0.99
    val bound = 4.0 * rMax / (1.0 - gamma)
    // Any FourWorld with V values bounded by R_max/(1-gamma) satisfies this.
    val vBound = rMax / (1.0 - gamma)
    val fw = FourWorld(Ev(vBound), Ev(-vBound), Ev(-vBound), Ev(vBound))
    assert(fw.deltaInteraction.abs <= Ev(bound + Tol))

  // -- Defs 40-42: Per-rival signal decomposition --

  test("PerRivalDelta: Theorem 3 -- sig = pass + manip"):
    val d = PerRivalDelta(deltaSig = Ev(5.0), deltaPass = Ev(2.0), deltaManip = Ev(3.0))
    assertEqualsDouble(d.deltaSig.value, (d.deltaPass + d.deltaManip).value, Tol)

  test("PerRivalDelta: negative deltaPass signals damaging leak (Corollary 1)"):
    val d = PerRivalDelta(deltaSig = Ev(-1.0), deltaPass = Ev(-3.0), deltaManip = Ev(2.0))
    assert(d.isDamagingLeak)

  test("PerRivalDelta: non-negative deltaPass is not a damaging leak"):
    val d = PerRivalDelta(deltaSig = Ev(4.0), deltaPass = Ev(1.0), deltaManip = Ev(3.0))
    assert(!d.isDamagingLeak)

  // -- Theorem 5: correct beliefs => deltaManip == 0 --

  test("Theorem 5: correct beliefs -> hasCorrectBeliefs true"):
    val d = PerRivalDelta(deltaSig = Ev(2.0), deltaPass = Ev(2.0), deltaManip = Ev(0.0))
    assert(d.hasCorrectBeliefs)

  test("Theorem 5: incorrect beliefs -> hasCorrectBeliefs false"):
    val d = PerRivalDelta(deltaSig = Ev(5.0), deltaPass = Ev(2.0), deltaManip = Ev(3.0))
    assert(!d.hasCorrectBeliefs)

  // -- Defs 48-49: Sub-decomposition --

  test("PerRivalSignalSubDecomposition: Theorem 3A -- sig = design + real"):
    val sub = PerRivalSignalSubDecomposition(deltaSigDesign = Ev(1.5), deltaSigReal = Ev(3.5))
    assertEqualsDouble(sub.total.value, 5.0, Tol)

  // -- Def 50: Canonical delta vocabulary --

  test("DeltaVocabulary contains per-rival and aggregate primitives"):
    val perRival = Map(
      PlayerId("v1") -> PerRivalDelta(Ev(5.0), Ev(2.0), Ev(3.0))
    )
    val fourWorld = FourWorld(Ev(10.0), Ev(7.0), Ev(6.0), Ev(4.0))
    val vocab = DeltaVocabulary(
      fourWorld = fourWorld,
      perRivalDeltas = perRival,
      deltaSigAggregate = Ev(5.0)
    )
    assertEquals(vocab.perRivalDeltas.size, 1)
    assertEqualsDouble(vocab.fourWorld.deltaControl.value, 2.0, Tol)
    // delta_learn is retired -- no field for it
```

- [ ] **Step 2: Run test -- expect compilation failure (RED)**

Run: `sbt "testOnly sicfun.holdem.strategic.StrategicValueTest"`
Expected: Compilation error

- [ ] **Step 3: Write the implementation**

```scala
package sicfun.holdem.strategic

/** The four analytical worlds (Def 44).
  *
  * V^{1,1} = V^{*, Gamma^attrib}_{Pi^S}     -- full: closed-loop + attributed learning
  * V^{1,0} = V^{*, Gamma^attrib}_{Pi^ol}     -- open-loop + attributed learning
  * V^{0,1} = V^{*, Gamma^blind}_{Pi^S}       -- closed-loop + blind (no learning)
  * V^{0,0} = V^{*, Gamma^blind}_{Pi^ol}      -- open-loop + blind (baseline world)
  *
  * First index (0/1): whether rivals learn from signals (attrib vs blind)
  * Second index (0/1): whether SICFUN uses closed-loop policy (Pi^S vs Pi^ol)
  */
final case class FourWorld(
    v11: Ev,
    v10: Ev,
    v01: Ev,
    v00: Ev
):
  /** Control value (Def 45): Delta_cont = V^{0,1} - V^{0,0}.
    * Value of SICFUN's strategic options when rivals do not learn.
    */
  inline def deltaControl: Ev = v01 - v00

  /** Marginal signaling effect (Def 46): Delta_sig* = V^{1,0} - V^{0,0}.
    * Value of information passively revealed through observed actions.
    */
  inline def deltaSigStar: Ev = v10 - v00

  /** Interaction term (Def 47): Delta_int = V^{1,1} - V^{1,0} - V^{0,1} + V^{0,0}.
    * Residual not captured by either control or signaling alone.
    */
  inline def deltaInteraction: Ev = v11 - v10 - v01 + v00

/** Per-rival signal decomposition (Defs 40-42).
  *
  * For each rival i under fixed background B^{-i}:
  * - delta_sig^i  = Q^{attrib,i} - Q^{blind,i}     (Def 40: total signal effect)
  * - delta_pass^i = Q^{ref,i}    - Q^{blind,i}      (Def 41: passive leakage)
  * - delta_manip^i = Q^{attrib,i} - Q^{ref,i}       (Def 42: manipulation rent)
  *
  * Theorem 3: delta_sig = delta_pass + delta_manip (exact telescoping).
  * Corollary 1: delta_pass < 0 => damaging passive leakage.
  * Theorem 5: attrib == ref => delta_manip == 0.
  */
final case class PerRivalDelta(
    deltaSig: Ev,
    deltaPass: Ev,
    deltaManip: Ev
):
  /** Corollary 1: True if action leaks information harmfully. */
  inline def isDamagingLeak: Boolean = deltaPass < Ev.Zero

  /** Theorem 5: True if rival holds correct beliefs (manip is effectively zero). */
  inline def hasCorrectBeliefs: Boolean = deltaManip.abs <= Ev(1e-12)

/** Per-rival signaling sub-decomposition (Defs 48-49).
  *
  * delta_sig = delta_sig,design + delta_sig,real (Theorem 3A).
  * - delta_sig,design: from action category only (design kernel, Def 19A)
  * - delta_sig,real: from sizing/timing realization beyond category
  */
final case class PerRivalSignalSubDecomposition(
    deltaSigDesign: Ev,
    deltaSigReal: Ev
):
  /** Theorem 3A: total signal = design + realization. */
  inline def total: Ev = deltaSigDesign + deltaSigReal

/** Canonical delta vocabulary (Def 50).
  *
  * Per-rival primitives: delta_sig^i, delta_pass^i, delta_manip^i,
  *                       delta_sig,design^i, delta_sig,real^i.
  * Aggregate primitives: delta_cont, delta_sig*, delta_int, delta_sig^agg.
  *
  * The symbol delta_learn remains RETIRED (v0.30.2 design contract).
  *
  * Non-additivity warning (Def 43): delta_sig^agg != sum_i delta_sig^i in general.
  */
final case class DeltaVocabulary(
    fourWorld: FourWorld,
    perRivalDeltas: Map[PlayerId, PerRivalDelta],
    deltaSigAggregate: Ev,
    perRivalSubDecompositions: Map[PlayerId, PerRivalSignalSubDecomposition] = Map.empty
)
```

- [ ] **Step 4: Run test -- expect all pass (GREEN)**

Run: `sbt "testOnly sicfun.holdem.strategic.StrategicValueTest"`
Expected: All tests pass

- [ ] **Step 5: Commit**

---

### Task 11: ChangepointDetector (Defs 26-28)

**Files:**
- Create: `src/main/scala/sicfun/holdem/strategic/ChangepointDetector.scala`
- Create: `src/test/scala/sicfun/holdem/strategic/ChangepointDetectorTest.scala`

- [ ] **Step 1: Write the failing test**

```scala
package sicfun.holdem.strategic

import sicfun.core.DiscreteDistribution

class ChangepointDetectorTest extends munit.FunSuite:

  private inline val Tol = 1e-9

  // -- Def 26: Structure --

  test("ChangepointDetector.initial: run length 0 with probability 1"):
    val cpd = ChangepointDetector(hazardRate = 0.1, rMin = 3, kappaCP = 0.5, wReset = 0.5)
    val state = cpd.initial
    assertEqualsDouble(state.runLengthPosterior.getOrElse(0, 0.0), 1.0, Tol)
    assertEquals(state.runLengthPosterior.size, 1)

  test("ChangepointDetector rejects hazardRate out of (0,1)"):
    intercept[IllegalArgumentException]:
      ChangepointDetector(hazardRate = 0.0, rMin = 3, kappaCP = 0.5, wReset = 0.5)
    intercept[IllegalArgumentException]:
      ChangepointDetector(hazardRate = 1.0, rMin = 3, kappaCP = 0.5, wReset = 0.5)
    intercept[IllegalArgumentException]:
      ChangepointDetector(hazardRate = -0.1, rMin = 3, kappaCP = 0.5, wReset = 0.5)

  test("ChangepointDetector rejects kappaCP out of (0,1)"):
    intercept[IllegalArgumentException]:
      ChangepointDetector(hazardRate = 0.1, rMin = 3, kappaCP = 0.0, wReset = 0.5)
    intercept[IllegalArgumentException]:
      ChangepointDetector(hazardRate = 0.1, rMin = 3, kappaCP = 1.0, wReset = 0.5)

  test("ChangepointDetector rejects wReset out of (0,1]"):
    intercept[IllegalArgumentException]:
      ChangepointDetector(hazardRate = 0.1, rMin = 3, kappaCP = 0.5, wReset = 0.0)
    intercept[IllegalArgumentException]:
      ChangepointDetector(hazardRate = 0.1, rMin = 3, kappaCP = 0.5, wReset = -0.1)

  test("ChangepointDetector accepts wReset = 1.0"):
    val cpd = ChangepointDetector(hazardRate = 0.1, rMin = 3, kappaCP = 0.5, wReset = 1.0)
    assert(cpd.wReset == 1.0)

  // -- Def 27: Run-length posterior update (Adams-MacKay recursion) --

  test("After one observation: run length 0 and 1 both present"):
    val cpd = ChangepointDetector(hazardRate = 0.1, rMin = 3, kappaCP = 0.5, wReset = 0.5)
    val state0 = cpd.initial
    val predProb: Int => Double = _ => 0.5
    val state1 = cpd.update(state0, predProb)
    assert(state1.runLengthPosterior.contains(0))
    assert(state1.runLengthPosterior.contains(1))

  test("Run-length posterior sums to 1 after each update"):
    val cpd = ChangepointDetector(hazardRate = 0.1, rMin = 5, kappaCP = 0.5, wReset = 0.5)
    var state = cpd.initial
    val predProb: Int => Double = r => if r < 3 then 0.8 else 0.2
    for step <- 1 to 20 do
      state = cpd.update(state, predProb)
      val total = state.runLengthPosterior.values.sum
      assertEqualsDouble(total, 1.0, Tol)

  test("Adams-MacKay with h=0.5 and uniform pred: symmetric split"):
    val cpd = ChangepointDetector(hazardRate = 0.5, rMin = 0, kappaCP = 0.5, wReset = 1.0)
    val state0 = cpd.initial
    val uniformPred: Int => Double = _ => 1.0
    val state1 = cpd.update(state0, uniformPred)
    // r=0 gets h*pred(0)*P(r=0) = 0.5*1.0*1.0 = 0.5
    // r=1 gets (1-h)*pred(0)*P(r=0) = 0.5*1.0*1.0 = 0.5
    assertEqualsDouble(state1.runLengthPosterior.getOrElse(0, 0.0), 0.5, Tol)
    assertEqualsDouble(state1.runLengthPosterior.getOrElse(1, 0.0), 0.5, Tol)

  // -- Def 28: Changepoint detection --

  test("isChangepointDetected true when short run-length mass exceeds kappaCP"):
    val cpd = ChangepointDetector(hazardRate = 0.1, rMin = 3, kappaCP = 0.3, wReset = 0.5)
    // P(r <= 3) = 0.2 + 0.15 + 0.05 = 0.4 > 0.3
    val heavyShort = ChangepointState(
      runLengthPosterior = Map(0 -> 0.2, 1 -> 0.15, 2 -> 0.05, 5 -> 0.3, 10 -> 0.3)
    )
    assert(cpd.isChangepointDetected(heavyShort))

  test("isChangepointDetected false when short run-length mass is below kappaCP"):
    val cpd = ChangepointDetector(hazardRate = 0.1, rMin = 3, kappaCP = 0.5, wReset = 0.5)
    // P(r <= 3) = 0.05 + 0.05 = 0.1 < 0.5
    val longRuns = ChangepointState(
      runLengthPosterior = Map(0 -> 0.05, 1 -> 0.05, 10 -> 0.4, 20 -> 0.5)
    )
    assert(!cpd.isChangepointDetected(longRuns))

  // -- Def 28: Prior reset --

  test("resetPrior blends current with meta prior using wReset"):
    val cpd = ChangepointDetector(hazardRate = 0.1, rMin = 3, kappaCP = 0.3, wReset = 0.4)
    val current = DiscreteDistribution(Map("TAG" -> 0.8, "LAG" -> 0.2))
    val meta = DiscreteDistribution(Map("TAG" -> 0.3, "LAG" -> 0.7))
    val blended = cpd.resetPrior(current, meta)
    // blended = (1 - 0.4) * current + 0.4 * meta
    // TAG: 0.6 * 0.8 + 0.4 * 0.3 = 0.60
    // LAG: 0.6 * 0.2 + 0.4 * 0.7 = 0.40
    assertEqualsDouble(blended.probabilityOf("TAG"), 0.60, Tol)
    assertEqualsDouble(blended.probabilityOf("LAG"), 0.40, Tol)

  test("resetPrior with wReset=1.0 returns pure meta prior"):
    val cpd = ChangepointDetector(hazardRate = 0.1, rMin = 3, kappaCP = 0.3, wReset = 1.0)
    val current = DiscreteDistribution(Map("TAG" -> 0.8, "LAG" -> 0.2))
    val meta = DiscreteDistribution(Map("TAG" -> 0.3, "LAG" -> 0.7))
    val blended = cpd.resetPrior(current, meta)
    assertEqualsDouble(blended.probabilityOf("TAG"), 0.3, Tol)
    assertEqualsDouble(blended.probabilityOf("LAG"), 0.7, Tol)

  test("resetPrior handles disjoint support"):
    val cpd = ChangepointDetector(hazardRate = 0.1, rMin = 3, kappaCP = 0.3, wReset = 0.5)
    val current = DiscreteDistribution(Map("TAG" -> 1.0))
    val meta = DiscreteDistribution(Map("LAG" -> 1.0))
    val blended = cpd.resetPrior(current, meta)
    assertEqualsDouble(blended.probabilityOf("TAG"), 0.5, Tol)
    assertEqualsDouble(blended.probabilityOf("LAG"), 0.5, Tol)

  // -- Stationarity recovery (A3 -> A3'): h near 0 => no changepoint --

  test("With very low hazardRate, changepoint is never detected"):
    val cpd = ChangepointDetector(hazardRate = 0.001, rMin = 3, kappaCP = 0.5, wReset = 0.5)
    var state = cpd.initial
    val predProb: Int => Double = _ => 0.5
    for _ <- 1 to 100 do
      state = cpd.update(state, predProb)
    assert(!cpd.isChangepointDetected(state))

  test("With low hazardRate, mass concentrates on the longest run length"):
    val cpd = ChangepointDetector(hazardRate = 0.001, rMin = 3, kappaCP = 0.5, wReset = 0.5)
    var state = cpd.initial
    val predProb: Int => Double = _ => 0.5
    for _ <- 1 to 10 do
      state = cpd.update(state, predProb)
    val maxRL = state.runLengthPosterior.maxBy(_._2)._1
    assertEquals(maxRL, 10)

  // -- Run-length posterior pruning does not happen (all run lengths kept) --

  test("After 5 updates, all integer run lengths 0..5 have some mass"):
    val cpd = ChangepointDetector(hazardRate = 0.2, rMin = 3, kappaCP = 0.5, wReset = 0.5)
    var state = cpd.initial
    val predProb: Int => Double = _ => 0.5
    for _ <- 1 to 5 do
      state = cpd.update(state, predProb)
    for r <- 0 to 5 do
      assert(
        state.runLengthPosterior.getOrElse(r, 0.0) > 0.0,
        s"Run length $r should have positive mass"
      )
```

- [ ] **Step 2: Run test -- expect compilation failure (RED)**

Run: `sbt "testOnly sicfun.holdem.strategic.ChangepointDetectorTest"`
Expected: Compilation error

- [ ] **Step 3: Write the implementation**

```scala
package sicfun.holdem.strategic

import sicfun.core.DiscreteDistribution

/** Immutable state of the run-length posterior for one rival.
  *
  * The run length r_t^i is a derived sufficient statistic of the
  * observable history (A3'), NOT a component of the augmented hidden
  * state space (Def 12).
  */
final case class ChangepointState(
    runLengthPosterior: Map[Int, Double]
)

/** Adams-MacKay Bayesian Online Changepoint Detection (Defs 26-28).
  *
  * The changepoint detection module D^i consists of:
  * - {r_t^i}_{t >= 1}: run-length sequence
  * - CPD^i: run-length posterior updater (Def 27)
  * - nu_meta^i: meta-learned prior for reset (Def 28)
  *
  * This is per-rival. Each rival gets its own ChangepointDetector instance.
  *
  * Stationarity recovery (A3 -> A3'): when hazardRate -> 0 and the type
  * kernel is identity, A3 is recovered -- the detector never fires.
  *
  * @param hazardRate h^i in (0,1): prior probability of a changepoint at each step
  * @param rMin minimum run length for the short-run-length test (r_min in Def 28)
  * @param kappaCP kappa_cp in (0,1): detection threshold (Def 28)
  * @param wReset w_reset in (0,1]: blending weight for prior reset (Def 28)
  */
final case class ChangepointDetector(
    hazardRate: Double,
    rMin: Int,
    kappaCP: Double,
    wReset: Double
):
  require(hazardRate > 0.0 && hazardRate < 1.0, s"hazardRate must be in (0,1), got $hazardRate")
  require(kappaCP > 0.0 && kappaCP < 1.0, s"kappaCP must be in (0,1), got $kappaCP")
  require(wReset > 0.0 && wReset <= 1.0, s"wReset must be in (0,1], got $wReset")

  /** Initial state: run length 0 with probability 1. */
  def initial: ChangepointState =
    ChangepointState(runLengthPosterior = Map(0 -> 1.0))

  /** Run-length posterior update (Def 27).
    *
    * P(r_t = ell | Y_{1:t}) is proportional to:
    * - For ell >= 1: P_pred(Y_t | r_{t-1}=ell-1) * (1 - h) * P(r_{t-1}=ell-1)
    * - For ell = 0:  sum_{ell'} P_pred(Y_t | r_t=0) * h * P(r_{t-1}=ell')
    *
    * @param state current run-length posterior
    * @param predictiveProb P_pred(Y_t | r): predictive probability of the current
    *                       observation under the run-length-conditioned posterior
    * @return updated state with normalized run-length posterior
    */
  def update(state: ChangepointState, predictiveProb: Int => Double): ChangepointState =
    val prior = state.runLengthPosterior

    // Growth probabilities: each existing run length ell grows to ell+1
    val growth = prior.map { case (ell, prob) =>
      (ell + 1) -> predictiveProb(ell) * (1.0 - hazardRate) * prob
    }

    // Changepoint mass: all run lengths collapse to 0
    val changepointMass = prior.foldLeft(0.0) { case (acc, (ell, prob)) =>
      acc + predictiveProb(0) * hazardRate * prob
    }

    // Combine: run length 0 gets changepoint mass, others get growth
    val unnormalized = growth + (0 -> changepointMass)

    // Normalize to ensure sum = 1
    val total = unnormalized.values.sum
    val normalized =
      if total > 1e-15 then unnormalized.view.mapValues(_ / total).toMap
      else Map(0 -> 1.0)

    ChangepointState(runLengthPosterior = normalized)

  /** Changepoint detection predicate (Def 28).
    *
    * Returns true when P(r_t <= r_min | Y_{1:t}) > kappa_cp.
    */
  def isChangepointDetected(state: ChangepointState): Boolean =
    val shortMass = state.runLengthPosterior.foldLeft(0.0) { case (acc, (r, p)) =>
      if r <= rMin then acc + p else acc
    }
    shortMass > kappaCP

  /** Prior reset upon changepoint detection (Def 28).
    *
    * mu_t <- (1 - w_reset) * mu_t + w_reset * nu_meta
    *
    * @param current the current type posterior mu_t^{R,i}
    * @param metaPrior the meta-learned prior nu_meta^i
    * @return the blended posterior
    */
  def resetPrior[A](
      current: DiscreteDistribution[A],
      metaPrior: DiscreteDistribution[A]
  ): DiscreteDistribution[A] =
    val allKeys = current.support ++ metaPrior.support
    val blended = allKeys.map { key =>
      val cw = current.probabilityOf(key)
      val mw = metaPrior.probabilityOf(key)
      key -> ((1.0 - wReset) * cw + wReset * mw)
    }.toMap
    DiscreteDistribution(blended)
```

- [ ] **Step 4: Run test -- expect all pass (GREEN)**

Run: `sbt "testOnly sicfun.holdem.strategic.ChangepointDetectorTest"`
Expected: All tests pass

- [ ] **Step 5: Commit**

---

### Task 12: Full verification

- [ ] **Step 1: Compile everything**

Run: `sbt compile`
Expected: PASS

- [ ] **Step 2: Run all Phase 1 tests**

Run: `sbt "testOnly sicfun.holdem.strategic.*"`
Expected: All tests pass

- [ ] **Step 3: Verify no prohibited imports**

Run:
```bash
grep -rn "import sicfun.holdem.engine" src/main/scala/sicfun/holdem/strategic/
grep -rn "import sicfun.holdem.runtime" src/main/scala/sicfun/holdem/strategic/
```
Expected: Zero results for both commands.

- [ ] **Step 4: Run the existing test suite to verify no regressions**

Run: `sbt test`
Expected: All existing tests pass unchanged. The formal layer is strictly additive.

- [ ] **Step 5: Final commit**

---

## Spec Coverage Matrix

| Spec Def | File | What | Status |
|----------|------|------|--------|
| Def 1 | `StrategicClass.scala` | C^S = V + B + M + SB | Covered |
| Def 2 | `StrategicClass.scala` | c_t^S in C^S | Covered (enum membership) |
| Def 3 | `StrategicClass.scala` | Agg: U -> {0,1} | Covered |
| Def 4 | `StrategicClass.scala` | StructuralBluff(c,u) | Covered |
| Def 5 | `Signal.scala` | Y_t^act = (a_t, lambda_t, tau_t) | Covered |
| Def 6 | `Signal.scala` | Y_t = (Y_t^act, Y_t^sd) | Covered |
| Def 7 | `Signal.scala` | Z_{t+1} := Y_t (canonical identification) | Covered (comment) |
| Def 8 | `Signal.scala` | Signal routing convention | Covered |
| Def 9 | `Baseline.scala` | pi^{0,S} | Covered (trait) |
| Def 10 | `Baseline.scala` | hat{pi}^{0,S,i} | Covered (trait) |
| Def 11 | `ReputationalProjection.scala` | phi_t^{S,i} := g^i(m_t^{R,i}) | Covered (trait) |
| Def 12 | `AugmentedState.scala` | X_tilde space | Covered |
| Def 13 | `AugmentedState.scala` | X_tilde_t tuple | Covered |
| Def 14 | `AugmentedState.scala` | b_tilde_t in Delta(X_tilde) | Covered |
| Def 16 | `RivalKernel.scala` | U^{R,i}_pi (type sig) | Covered |
| Def 17 | `RivalKernel.scala` | BuildRivalKernel (type sig) | Covered |
| Def 18 | `RivalKernel.scala` | Action kernels ref/attrib/blind | Covered |
| Def 19 | `RivalKernel.scala` | Showdown kernel (type sig) | Covered |
| Def 19A | `RivalKernel.scala` | Design-signal kernel (enum variant) | Covered |
| Def 20 | `RivalKernel.scala` | Full kernel (type sig) | Covered |
| Def 21 | `RivalKernel.scala` | Joint kernel profiles | Covered |
| Def 26 | `ChangepointDetector.scala` | D^i module | Covered |
| Def 27 | `ChangepointDetector.scala` | Run-length posterior update | Covered |
| Def 28 | `ChangepointDetector.scala` | Changepoint-triggered reset | Covered |
| Def 44 | `StrategicValue.scala` | Four worlds V^{1,1}..V^{0,0} | Covered |
| Def 45 | `StrategicValue.scala` | Delta_cont | Covered |
| Def 46 | `StrategicValue.scala` | Delta_sig* | Covered |
| Def 47 | `StrategicValue.scala` | Delta_int | Covered |
| Def 50 | `StrategicValue.scala` | Canonical Delta vocabulary | Covered |
| -- | `DomainTypes.scala` | Chips, PotFraction, Ev opaques | Covered |
| -- | `TableStructure.scala` | PlayerId, SeatStatus, TableMap, RivalMap | Covered |
| -- | `Fidelity.scala` | BridgeResult, Fidelity, Severity | Covered |

## Theorems and Laws Validated by Tests

| Theorem/Law | Test File | Test Name |
|-------------|-----------|-----------|
| Theorem 3 | `StrategicValueTest` | sig = pass + manip |
| Theorem 3A | `StrategicValueTest` | sig = design + real |
| Theorem 4 | `StrategicValueTest` | exact aggregate decomposition (3 test cases) |
| Theorem 5 | `StrategicValueTest` | correct beliefs -> hasCorrectBeliefs |
| Corollary 1 | `StrategicValueTest` | negative deltaPass signals leak |
| Corollary 3 | `StrategicValueTest` | separability => int = 0 |
| Corollary 4 | `StrategicValueTest` | coarse interaction bound |
| Law L1 | `TableStructureTest` | RivalMap rejects empty vector |
| Law L2 | `StrategicClassTest` | isExploitativeBluff implies isStructuralBluff |
| Law L7 | `AugmentedStateTest` | multiway-native, no singular opponent |
| Law L12 | `RivalKernelLawTest` | blind kernel returns exact same state |
| Law L13 | `RivalKernelLawTest` | no cross-rival parameter in ActionKernel |
| A3 recovery | `ChangepointDetectorTest` | low hazardRate => no changepoint |

## Key Design Decisions

1. **PlayerId is `opaque type = String`**, not a case class. Minimal overhead, zero boxing.

2. **TableMap and RivalMap are generic** in their per-seat data payload (`Seat[A]`), enabling `TableMap[Chips]` for stacks and `RivalMap[OpponentModelState]` for opponent models with the same structure.

3. **OwnEvidence uses `Map[String, Double]`** for summary storage. Intentionally loose -- the spec (A4') says xi^S is finite but does not constrain its internal structure. Concrete evidence types defined by the engine bridge.

4. **RivalBeliefState is a trait**, not a case class. Different kernel implementations (Phase 4) provide different concrete states.

5. **OpponentModelState.attributedBaseline is `Option[AttributedBaseline]`** because the attributed baseline is absent in the current engine (bridge Severity: Structural). Optional prevents fake zeros.

6. **PublicState is an OWN TYPE**, not an alias for GameState. Uses PlayerId (not Int seat index) and Chips (not raw Double). Conversion `GameState -> PublicState` lives in bridge (Phase 4).

7. **FourWorld stores raw Ev values, not deltas.** Deltas are computed as inline defs to guarantee Theorem 4 holds by construction (algebraic identity, not numerical coincidence).

8. **ChangepointDetector is pure and immutable.** State is threaded explicitly via `ChangepointState`. No mutable fields, no singletons, no global state.

9. **DeltaVocabulary has no `deltaLearn` field.** The symbol is retired per v0.30.2 design contract.

10. **Ev.Zero and PotFraction.Zero/One use `val`** (not `inline val`) because they need a declared type to be usable outside the companion object's scope. Constants that require no type annotation (like `Probability.Eps` in `sicfun.core`) use `inline val`.
