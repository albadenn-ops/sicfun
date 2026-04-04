package sicfun.holdem.strategic.bridge

import sicfun.holdem.strategic.*
import sicfun.holdem.types.{Board, GameState, PokerAction, Position, Street}

class BridgeTest extends munit.FunSuite:

  private inline val Tol = 1e-12

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  private def makeGS(
      street: Street = Street.Flop,
      pot: Double = 100.0,
      toCall: Double = 20.0,
      stack: Double = 500.0
  ): GameState =
    GameState(
      street = street,
      board = Board.empty,
      pot = pot,
      toCall = toCall,
      position = Position.Button,
      stackSize = stack,
      betHistory = Vector.empty
    )

  // ---------------------------------------------------------------------------
  // SignalBridge
  // ---------------------------------------------------------------------------

  test("SignalBridge.toActionSignal Fold returns Approximate with timing-absent note"):
    val result = SignalBridge.toActionSignal(PokerAction.Fold, Street.Flop, Chips(100.0))
    result match
      case BridgeResult.Approximate(sig, loss) =>
        assertEquals(sig.action, PokerAction.Category.Fold)
        assertEquals(sig.timing, None)
        assertEquals(sig.stage, Street.Flop)
        assert(loss.nonEmpty)
      case other => fail(s"expected Approximate, got $other")

  test("SignalBridge.toActionSignal Check returns Approximate"):
    val result = SignalBridge.toActionSignal(PokerAction.Check, Street.Preflop, Chips(50.0))
    result match
      case BridgeResult.Approximate(sig, _) =>
        assertEquals(sig.action, PokerAction.Category.Check)
        assertEquals(sig.sizing, None)
      case other => fail(s"expected Approximate, got $other")

  test("SignalBridge.toActionSignal Call returns Approximate with no sizing"):
    val result = SignalBridge.toActionSignal(PokerAction.Call, Street.Turn, Chips(80.0))
    result match
      case BridgeResult.Approximate(sig, _) =>
        assertEquals(sig.action, PokerAction.Category.Call)
        assertEquals(sig.sizing, None)
      case other => fail(s"expected Approximate, got $other")

  test("SignalBridge.toActionSignal Raise computes pot fraction correctly"):
    val pot = Chips(100.0)
    val result = SignalBridge.toActionSignal(PokerAction.Raise(50.0), Street.River, pot)
    result match
      case BridgeResult.Approximate(sig, _) =>
        assertEquals(sig.action, PokerAction.Category.Raise)
        val sizing = sig.sizing.getOrElse(fail("expected Some sizing"))
        assertEqualsDouble(sizing.absolute.value, 50.0, Tol)
        assertEqualsDouble(sizing.fractionOfPot.value, 0.5, Tol)
      case other => fail(s"expected Approximate, got $other")

  test("SignalBridge.toActionSignal Raise with zero pot defaults fraction to 1.0"):
    val result = SignalBridge.toActionSignal(PokerAction.Raise(30.0), Street.Preflop, Chips(0.0))
    result match
      case BridgeResult.Approximate(sig, _) =>
        val sizing = sig.sizing.getOrElse(fail("expected Some sizing"))
        assertEqualsDouble(sizing.fractionOfPot.value, 1.0, Tol)
      case other => fail(s"expected Approximate, got $other")

  test("SignalBridge.toActionSignal fidelity is always Approximate"):
    val actions = List(
      PokerAction.Fold,
      PokerAction.Check,
      PokerAction.Call,
      PokerAction.Raise(75.0)
    )
    for action <- actions do
      val r = SignalBridge.toActionSignal(action, Street.Flop, Chips(100.0))
      assertEquals(r.fidelity, Fidelity.Approximate)

  test("SignalBridge.toTotalSignal wraps ActionSignal with no showdown"):
    val result = SignalBridge.toTotalSignal(PokerAction.Call, Street.Flop, Chips(100.0))
    result match
      case BridgeResult.Approximate(ts, _) =>
        assertEquals(ts.actionSignal.action, PokerAction.Category.Call)
        assertEquals(ts.showdown, None)
      case other => fail(s"expected Approximate, got $other")

  test("SignalBridge.toTotalSignal Raise preserves sizing through wrapping"):
    val result = SignalBridge.toTotalSignal(PokerAction.Raise(60.0), Street.River, Chips(120.0))
    result match
      case BridgeResult.Approximate(ts, _) =>
        val sz = ts.actionSignal.sizing.getOrElse(fail("expected sizing"))
        assertEqualsDouble(sz.fractionOfPot.value, 0.5, Tol)
      case other => fail(s"expected Approximate, got $other")

  // ---------------------------------------------------------------------------
  // ClassificationBridge
  // ---------------------------------------------------------------------------

  test("ClassificationBridge.classify high equity -> Value"):
    val result = ClassificationBridge.classify(0.70, hasDrawPotential = false)
    result match
      case BridgeResult.Approximate(cls, _) => assertEquals(cls, StrategicClass.Value)
      case other => fail(s"expected Approximate, got $other")

  test("ClassificationBridge.classify low equity -> Bluff"):
    val result = ClassificationBridge.classify(0.20, hasDrawPotential = true)
    result match
      case BridgeResult.Approximate(cls, _) => assertEquals(cls, StrategicClass.Bluff)
      case other => fail(s"expected Approximate, got $other")

  test("ClassificationBridge.classify mid equity with draw -> SemiBluff"):
    val result = ClassificationBridge.classify(0.50, hasDrawPotential = true)
    result match
      case BridgeResult.Approximate(cls, _) => assertEquals(cls, StrategicClass.SemiBluff)
      case other => fail(s"expected Approximate, got $other")

  test("ClassificationBridge.classify mid equity without draw -> Marginal"):
    val result = ClassificationBridge.classify(0.50, hasDrawPotential = false)
    result match
      case BridgeResult.Approximate(cls, _) => assertEquals(cls, StrategicClass.Marginal)
      case other => fail(s"expected Approximate, got $other")

  test("ClassificationBridge.classify at exact valueFloor boundary -> Value"):
    val result = ClassificationBridge.classify(0.65, hasDrawPotential = false)
    result match
      case BridgeResult.Approximate(cls, _) => assertEquals(cls, StrategicClass.Value)
      case other => fail(s"expected Approximate, got $other")

  test("ClassificationBridge.classify just below bluffCeiling -> Bluff"):
    val result = ClassificationBridge.classify(0.34, hasDrawPotential = true)
    result match
      case BridgeResult.Approximate(cls, _) => assertEquals(cls, StrategicClass.Bluff)
      case other => fail(s"expected Approximate, got $other")

  test("ClassificationBridge.classify fidelity is always Approximate"):
    val result = ClassificationBridge.classify(0.5, hasDrawPotential = false)
    assertEquals(result.fidelity, Fidelity.Approximate)

  test("ClassificationBridge.ClassificationThresholds rejects inverted floors"):
    intercept[IllegalArgumentException]:
      ClassificationBridge.ClassificationThresholds(valueFloor = 0.30, bluffCeiling = 0.50)

  test("ClassificationBridge.classify with custom thresholds"):
    val thresholds = ClassificationBridge.ClassificationThresholds(valueFloor = 0.80, bluffCeiling = 0.20)
    val result = ClassificationBridge.classify(0.75, hasDrawPotential = false, thresholds = thresholds)
    result match
      case BridgeResult.Approximate(cls, _) => assertEquals(cls, StrategicClass.Marginal)
      case other => fail(s"expected Approximate, got $other")

  // ---------------------------------------------------------------------------
  // PublicStateBridge
  // ---------------------------------------------------------------------------

  test("PublicStateBridge.extractStreet returns Exact with correct street"):
    val gs = makeGS(street = Street.Turn)
    PublicStateBridge.extractStreet(gs) match
      case BridgeResult.Exact(s) => assertEquals(s, Street.Turn)
      case other => fail(s"expected Exact, got $other")

  test("PublicStateBridge.extractPot returns Exact with correct value"):
    val gs = makeGS(pot = 250.0)
    PublicStateBridge.extractPot(gs) match
      case BridgeResult.Exact(chips) => assertEqualsDouble(chips.value, 250.0, Tol)
      case other => fail(s"expected Exact, got $other")

  test("PublicStateBridge.extractPot zero pot"):
    val gs = makeGS(pot = 0.0, toCall = 0.0)
    PublicStateBridge.extractPot(gs) match
      case BridgeResult.Exact(chips) => assertEqualsDouble(chips.value, 0.0, Tol)
      case other => fail(s"expected Exact, got $other")

  test("PublicStateBridge.extractHeroStack returns Exact with correct value"):
    val gs = makeGS(stack = 1200.0)
    PublicStateBridge.extractHeroStack(gs) match
      case BridgeResult.Exact(chips) => assertEqualsDouble(chips.value, 1200.0, Tol)
      case other => fail(s"expected Exact, got $other")

  test("PublicStateBridge.extractToCall returns Exact with correct value"):
    val gs = makeGS(toCall = 40.0)
    PublicStateBridge.extractToCall(gs) match
      case BridgeResult.Exact(chips) => assertEqualsDouble(chips.value, 40.0, Tol)
      case other => fail(s"expected Exact, got $other")

  test("PublicStateBridge.extractToCall zero"):
    val gs = makeGS(toCall = 0.0)
    PublicStateBridge.extractToCall(gs) match
      case BridgeResult.Exact(chips) => assertEqualsDouble(chips.value, 0.0, Tol)
      case other => fail(s"expected Exact, got $other")

  test("PublicStateBridge.extractTableMap returns Absent"):
    val gs = makeGS()
    PublicStateBridge.extractTableMap(gs) match
      case BridgeResult.Absent(reason) => assert(reason.nonEmpty)
      case other => fail(s"expected Absent, got $other")

  test("PublicStateBridge.extractTableMap fidelity is Absent"):
    assertEquals(PublicStateBridge.extractTableMap(makeGS()).fidelity, Fidelity.Absent)

  test("PublicStateBridge.extractStreet fidelity is Exact"):
    assertEquals(PublicStateBridge.extractStreet(makeGS()).fidelity, Fidelity.Exact)

  // ---------------------------------------------------------------------------
  // OpponentModelBridge
  // ---------------------------------------------------------------------------

  test("OpponentModelBridge.statsToClassPosterior normal stats returns Approximate"):
    val result = OpponentModelBridge.statsToClassPosterior(vpip = 0.25, pfr = 0.18, af = 2.5)
    result match
      case BridgeResult.Approximate(dist, _) =>
        val weights = dist.weights
        assert(weights.contains(StrategicClass.Value))
        assert(weights.contains(StrategicClass.Bluff))
        assert(weights.contains(StrategicClass.Marginal))
        assert(weights.contains(StrategicClass.SemiBluff))
      case other => fail(s"expected Approximate, got $other")

  test("OpponentModelBridge.statsToClassPosterior posterior sums to 1.0"):
    val result = OpponentModelBridge.statsToClassPosterior(vpip = 0.30, pfr = 0.20, af = 3.0)
    result match
      case BridgeResult.Approximate(dist, _) =>
        val total = dist.weights.values.sum
        assertEqualsDouble(total, 1.0, 1e-10)
      case other => fail(s"expected Approximate, got $other")

  test("OpponentModelBridge.statsToClassPosterior all-zero stats returns Approximate"):
    val result = OpponentModelBridge.statsToClassPosterior(vpip = 0.0, pfr = 0.0, af = 0.0)
    result match
      case BridgeResult.Approximate(dist, _) =>
        val total = dist.weights.values.sum
        assertEqualsDouble(total, 1.0, 1e-10)
      case other => fail(s"expected Approximate, got $other")

  test("OpponentModelBridge.statsToClassPosterior negative vpip returns Absent"):
    val result = OpponentModelBridge.statsToClassPosterior(vpip = -0.1, pfr = 0.15, af = 2.0)
    result match
      case BridgeResult.Absent(reason) => assert(reason.contains("invalid stats"))
      case other => fail(s"expected Absent, got $other")

  test("OpponentModelBridge.statsToClassPosterior negative pfr returns Absent"):
    val result = OpponentModelBridge.statsToClassPosterior(vpip = 0.25, pfr = -0.05, af = 2.0)
    result match
      case BridgeResult.Absent(reason) => assert(reason.contains("invalid stats"))
      case other => fail(s"expected Absent, got $other")

  test("OpponentModelBridge.statsToClassPosterior negative af returns Absent"):
    val result = OpponentModelBridge.statsToClassPosterior(vpip = 0.25, pfr = 0.18, af = -1.0)
    result match
      case BridgeResult.Absent(reason) => assert(reason.contains("invalid stats"))
      case other => fail(s"expected Absent, got $other")

  test("OpponentModelBridge.statsToClassPosterior fidelity is Approximate for valid input"):
    assertEquals(
      OpponentModelBridge.statsToClassPosterior(0.3, 0.2, 2.0).fidelity,
      Fidelity.Approximate
    )

  // ---------------------------------------------------------------------------
  // BaselineBridge
  // ---------------------------------------------------------------------------

  test("BaselineBridge.toRealBaseline wraps equity in Approximate"):
    val result = BaselineBridge.toRealBaseline(0.55)
    result match
      case BridgeResult.Approximate(ev, _) => assertEqualsDouble(ev.value, 0.55, Tol)
      case other => fail(s"expected Approximate, got $other")

  test("BaselineBridge.toRealBaseline zero equity"):
    BaselineBridge.toRealBaseline(0.0) match
      case BridgeResult.Approximate(ev, _) => assertEqualsDouble(ev.value, 0.0, Tol)
      case other => fail(s"expected Approximate, got $other")

  test("BaselineBridge.toRealBaseline negative EV"):
    BaselineBridge.toRealBaseline(-5.0) match
      case BridgeResult.Approximate(ev, _) => assertEqualsDouble(ev.value, -5.0, Tol)
      case other => fail(s"expected Approximate, got $other")

  test("BaselineBridge.toRealBaseline fidelity is Approximate"):
    assertEquals(BaselineBridge.toRealBaseline(0.5).fidelity, Fidelity.Approximate)

  test("BaselineBridge.toAttributedBaselines empty map returns Absent"):
    val result = BaselineBridge.toAttributedBaselines(Map.empty)
    result match
      case BridgeResult.Absent(reason) => assert(reason.nonEmpty)
      case other => fail(s"expected Absent, got $other")

  test("BaselineBridge.toAttributedBaselines non-empty returns Approximate"):
    val input = Map(PlayerId("v1") -> 0.60, PlayerId("v2") -> 0.40)
    val result = BaselineBridge.toAttributedBaselines(input)
    result match
      case BridgeResult.Approximate(evMap, _) =>
        assertEqualsDouble(evMap(PlayerId("v1")).value, 0.60, Tol)
        assertEqualsDouble(evMap(PlayerId("v2")).value, 0.40, Tol)
      case other => fail(s"expected Approximate, got $other")

  test("BaselineBridge.toAttributedBaselines single rival"):
    val input = Map(PlayerId("hero") -> 1.0)
    val result = BaselineBridge.toAttributedBaselines(input)
    result match
      case BridgeResult.Approximate(evMap, _) =>
        assertEqualsDouble(evMap(PlayerId("hero")).value, 1.0, Tol)
      case other => fail(s"expected Approximate, got $other")

  test("BaselineBridge.toAttributedBaselines fidelity is Absent for empty"):
    assertEquals(BaselineBridge.toAttributedBaselines(Map.empty).fidelity, Fidelity.Absent)

  test("BaselineBridge.toAttributedBaselines fidelity is Approximate for non-empty"):
    assertEquals(
      BaselineBridge.toAttributedBaselines(Map(PlayerId("v") -> 0.5)).fidelity,
      Fidelity.Approximate
    )

  // ---------------------------------------------------------------------------
  // ValueBridge
  // ---------------------------------------------------------------------------

  test("ValueBridge.toFourWorld returns Approximate"):
    val result = ValueBridge.toFourWorld(engineEv = 0.60, staticEquity = 0.50)
    result match
      case BridgeResult.Approximate(fw, _) =>
        assertEqualsDouble(fw.v11.value, 0.60, Tol)
        assertEqualsDouble(fw.v00.value, 0.50, Tol)
      case other => fail(s"expected Approximate, got $other")

  test("ValueBridge.toFourWorld default controlFrac=0.5 splits gap evenly"):
    val result = ValueBridge.toFourWorld(engineEv = 0.70, staticEquity = 0.50)
    result match
      case BridgeResult.Approximate(fw, _) =>
        // gap = 0.20, controlFrac=0.5
        // v01 = 0.50 + 0.20*0.5 = 0.60
        // v10 = 0.50 + 0.20*0.5 = 0.60
        assertEqualsDouble(fw.v01.value, 0.60, Tol)
        assertEqualsDouble(fw.v10.value, 0.60, Tol)
      case other => fail(s"expected Approximate, got $other")

  test("ValueBridge.toFourWorld controlFrac=0.0 puts all gap in v10"):
    val result = ValueBridge.toFourWorld(engineEv = 0.70, staticEquity = 0.50, controlFrac = 0.0)
    result match
      case BridgeResult.Approximate(fw, _) =>
        assertEqualsDouble(fw.v01.value, 0.50, Tol)  // staticEquity + gap*0
        assertEqualsDouble(fw.v10.value, 0.70, Tol)  // staticEquity + gap*1
      case other => fail(s"expected Approximate, got $other")

  test("ValueBridge.toFourWorld controlFrac=1.0 puts all gap in v01"):
    val result = ValueBridge.toFourWorld(engineEv = 0.70, staticEquity = 0.50, controlFrac = 1.0)
    result match
      case BridgeResult.Approximate(fw, _) =>
        assertEqualsDouble(fw.v01.value, 0.70, Tol)  // staticEquity + gap*1
        assertEqualsDouble(fw.v10.value, 0.50, Tol)  // staticEquity + gap*0
      case other => fail(s"expected Approximate, got $other")

  test("ValueBridge.toFourWorld equal engineEv and staticEquity -> zero gap"):
    val result = ValueBridge.toFourWorld(engineEv = 0.55, staticEquity = 0.55)
    result match
      case BridgeResult.Approximate(fw, _) =>
        assertEqualsDouble(fw.v11.value, 0.55, Tol)
        assertEqualsDouble(fw.v00.value, 0.55, Tol)
        assertEqualsDouble(fw.v01.value, 0.55, Tol)
        assertEqualsDouble(fw.v10.value, 0.55, Tol)
      case other => fail(s"expected Approximate, got $other")

  test("ValueBridge.toFourWorld rejects controlFrac < 0"):
    intercept[IllegalArgumentException]:
      ValueBridge.toFourWorld(0.60, 0.50, controlFrac = -0.1)

  test("ValueBridge.toFourWorld rejects controlFrac > 1"):
    intercept[IllegalArgumentException]:
      ValueBridge.toFourWorld(0.60, 0.50, controlFrac = 1.1)

  test("ValueBridge.toFourWorld fidelity is Approximate"):
    assertEquals(ValueBridge.toFourWorld(0.6, 0.5).fidelity, Fidelity.Approximate)

  test("ValueBridge.toDeltaVocabulary returns Approximate"):
    val fw = FourWorld(Ev(0.70), Ev(0.60), Ev(0.60), Ev(0.50))
    val perRivalDeltas = Map(
      PlayerId("v1") -> PerRivalDelta(Ev(0.10), Ev(0.05), Ev(0.05))
    )
    val result = ValueBridge.toDeltaVocabulary(fw, perRivalDeltas, Ev(0.10))
    result match
      case BridgeResult.Approximate(vocab, _) =>
        assertEquals(vocab.perRivalDeltas.size, 1)
        assertEqualsDouble(vocab.fourWorld.v11.value, 0.70, Tol)
        assertEqualsDouble(vocab.deltaSigAggregate.value, 0.10, Tol)
      case other => fail(s"expected Approximate, got $other")

  test("ValueBridge.toDeltaVocabulary with no per-rival deltas"):
    val fw = FourWorld(Ev(0.50), Ev(0.50), Ev(0.50), Ev(0.50))
    val result = ValueBridge.toDeltaVocabulary(fw, Map.empty, Ev(0.0))
    result match
      case BridgeResult.Approximate(vocab, _) =>
        assertEquals(vocab.perRivalDeltas.size, 0)
      case other => fail(s"expected Approximate, got $other")

  test("ValueBridge.toDeltaVocabulary fidelity is Approximate"):
    val fw = FourWorld(Ev(0.6), Ev(0.5), Ev(0.5), Ev(0.4))
    assertEquals(
      ValueBridge.toDeltaVocabulary(fw, Map.empty, Ev(0.0)).fidelity,
      Fidelity.Approximate
    )

  // ---------------------------------------------------------------------------
  // BridgeManifest
  // ---------------------------------------------------------------------------

  test("BridgeManifest.entries is non-empty"):
    assert(BridgeManifest.entries.nonEmpty)

  test("BridgeManifest.structuralGaps contains only Structural severity entries"):
    val gaps = BridgeManifest.structuralGaps
    assert(gaps.forall(_.severity == Severity.Structural))

  test("BridgeManifest.structuralGaps is a subset of all entries"):
    val gaps = BridgeManifest.structuralGaps.map(_.formalObject).toSet
    val all  = BridgeManifest.entries.map(_.formalObject).toSet
    assert(gaps.subsetOf(all))

  test("BridgeManifest.absentObjects have Absent fidelity"):
    val absent = BridgeManifest.absentObjects
    assert(absent.forall(_.fidelity == Fidelity.Absent))

  test("BridgeManifest.absentObjects includes TableMap"):
    assert(BridgeManifest.absentObjects.exists(_.formalObject == "TableMap"))

  test("BridgeManifest.absentObjects includes ActionSignal.timing"):
    assert(BridgeManifest.absentObjects.exists(_.formalObject == "ActionSignal.timing"))

  test("BridgeManifest.summary contains correct counts"):
    val s = BridgeManifest.summary
    val total = BridgeManifest.entries.size
    assert(s.contains(s"$total total"))
    assert(s.startsWith("BridgeManifest:"))

  test("BridgeManifest.summary exact count matches entries with Exact fidelity"):
    val exactCount = BridgeManifest.entries.count(_.fidelity == Fidelity.Exact)
    assert(BridgeManifest.summary.contains(s"$exactCount exact"))

  test("BridgeManifest.summary approximate count matches"):
    val approxCount = BridgeManifest.entries.count(_.fidelity == Fidelity.Approximate)
    assert(BridgeManifest.summary.contains(s"$approxCount approximate"))

  test("BridgeManifest every entry has non-empty formalObject"):
    assert(BridgeManifest.entries.forall(_.formalObject.nonEmpty))

  test("BridgeManifest every entry has non-empty specDef"):
    assert(BridgeManifest.entries.forall(_.specDef.nonEmpty))
