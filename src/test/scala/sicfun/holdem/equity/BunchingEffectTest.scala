package sicfun.holdem.equity
import sicfun.holdem.types.*

import munit.FunSuite
import sicfun.core.Card

import scala.util.Random

/**
  * Tests for the bunching-effect engine and its supporting types (TableFormat, TableRanges).
  *
  * Covers:
  *   - TableFormat preflop ordering correctness and foldsBeforeOpener logic
  *   - TableRanges default parsing, custom overrides, and open/fold probability queries
  *   - BunchingEffect input validation (empty folds, villain-in-folds, non-positive trials)
  *   - Adjusted range properties: dead-card exclusion, normalization, seed determinism
  *   - Full compute pipeline: bunching delta finiteness, equity trial counts
  *   - computeForOpener convenience API consistency with explicit fold construction
  *
  * Uses low trial counts for speed; these tests verify correctness, not statistical precision.
  * The CPU preflop backend is forced via system property to avoid GPU dependency in tests.
  */
class BunchingEffectTest extends FunSuite:
  private val PreflopBackendProperty = "sicfun.holdem.preflopEquityBackend"

  /** Executes a block with the preflop equity backend forced to CPU,
    * ensuring tests don't depend on GPU availability.
    */
  private def withCpuPreflopBackend[A](thunk: => A): A =
    TestSystemPropertyScope.withSystemProperties(
      Vector(PreflopBackendProperty -> Some("cpu"))
    )(thunk)

  // -- Test helpers for card/hand construction --
  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  test("table format preflop orders and folds-before-opener are consistent") {
    assertEquals(TableFormat.forPlayerCount(2), TableFormat.HeadsUp)
    assertEquals(TableFormat.forPlayerCount(6), TableFormat.SixMax)
    assertEquals(TableFormat.forPlayerCount(9), TableFormat.NineMax)
    assertEquals(TableFormat.HeadsUp.preflopOrder.length, 2)
    assertEquals(TableFormat.SixMax.preflopOrder.length, 6)
    assertEquals(TableFormat.NineMax.preflopOrder.length, 9)

    val foldsToCutoff9 = TableFormat.NineMax.foldsBeforeOpener(Position.Cutoff)
    assertEquals(
      foldsToCutoff9,
      Vector(Position.UTG, Position.UTG1, Position.UTG2, Position.Middle, Position.Hijack)
    )

    val foldsToButton6 = TableFormat.SixMax.foldsBeforeOpener(Position.Button)
    assertEquals(
      foldsToButton6,
      Vector(Position.UTG, Position.Middle, Position.Cutoff)
    )

    intercept[IllegalArgumentException] {
      TableFormat.HeadsUp.foldsBeforeOpener(Position.UTG)
    }
  }

  test("table ranges defaults parse for all positions and expose open/fold probabilities") {
    val ranges = TableRanges.defaults(TableFormat.NineMax)

    TableFormat.NineMax.preflopOrder.foreach { pos =>
      assert(ranges.rangeFor(pos).weights.nonEmpty, s"range for $pos should be non-empty")
    }

    val utgSupport = ranges.rangeFor(Position.UTG).support.size
    val btnSupport = ranges.rangeFor(Position.Button).support.size
    assert(utgSupport < btnSupport, "UTG should be tighter than Button in defaults")

    val aa = hole("As", "Ad")
    val sevenTwoOff = hole("7c", "2d")

    assertEquals(ranges.openProbability(Position.UTG, aa), 1.0)
    assertEquals(ranges.foldProbability(Position.UTG, aa), 0.0)
    assertEquals(ranges.openProbability(Position.UTG, sevenTwoOff), 0.0)
    assertEquals(ranges.foldProbability(Position.UTG, sevenTwoOff), 1.0)
  }

  test("table ranges custom overrides specific positions and rejects invalid strings") {
    val customEither = TableRanges.custom(
      TableFormat.NineMax,
      overrides = Map(Position.UTG -> "AA", Position.Button -> "AKs")
    )
    assert(customEither.isRight)
    val custom = customEither.fold(err => fail(err), identity)

    assertEquals(custom.rangeFor(Position.UTG).support.size, 6)
    assertEquals(custom.rangeFor(Position.Button).support.size, 4)
    assertEquals(custom.openProbability(Position.UTG, hole("As", "Ad")), 1.0)
    assertEquals(custom.openProbability(Position.UTG, hole("Ks", "Kd")), 0.0)

    val invalid = TableRanges.custom(TableFormat.NineMax, overrides = Map(Position.UTG -> "AAs"))
    assert(invalid.isLeft)
  }

  test("bunching validation rejects empty folds villain-in-folds and non-positive trials") {
    val hero = hole("As", "Kd")
    val table = TableRanges.defaults(TableFormat.NineMax)
    val oneFold = Vector(PreflopFold(Position.UTG))

    intercept[IllegalArgumentException] {
      BunchingEffect.adjustedRange(
        hero = hero,
        board = Board.empty,
        folds = Vector.empty,
        tableRanges = table,
        villainPos = Position.BigBlind,
        trials = 50,
        rng = new Random(1)
      )
    }

    intercept[IllegalArgumentException] {
      BunchingEffect.adjustedRange(
        hero = hero,
        board = Board.empty,
        folds = Vector(PreflopFold(Position.BigBlind)),
        tableRanges = table,
        villainPos = Position.BigBlind,
        trials = 50,
        rng = new Random(2)
      )
    }

    intercept[IllegalArgumentException] {
      BunchingEffect.adjustedRange(
        hero = hero,
        board = Board.empty,
        folds = oneFold,
        tableRanges = table,
        villainPos = Position.BigBlind,
        trials = 0,
        rng = new Random(3)
      )
    }

    intercept[IllegalArgumentException] {
      BunchingEffect.compute(
        hero = hero,
        board = Board.empty,
        folds = oneFold,
        tableRanges = table,
        villainPos = Position.BigBlind,
        trials = 50,
        equityTrials = 0,
        rng = new Random(4)
      )
    }
  }

  test("adjusted range excludes dead cards is normalized and deterministic with same seed") {
    val hero = hole("As", "Kd")
    val board = Board.from(Vector(card("2h"), card("7c"), card("Th")))
    val table = TableRanges.defaults(TableFormat.NineMax)
    val folds = Vector(
      PreflopFold(Position.UTG),
      PreflopFold(Position.UTG1),
      PreflopFold(Position.UTG2),
      PreflopFold(Position.Middle),
      PreflopFold(Position.Hijack),
      PreflopFold(Position.Cutoff)
    )

    val r1 = BunchingEffect.adjustedRange(
      hero = hero,
      board = board,
      folds = folds,
      tableRanges = table,
      villainPos = Position.Button,
      trials = 300,
      rng = new Random(42)
    )
    val r2 = BunchingEffect.adjustedRange(
      hero = hero,
      board = board,
      folds = folds,
      tableRanges = table,
      villainPos = Position.Button,
      trials = 300,
      rng = new Random(42)
    )

    val dead = hero.asSet ++ board.asSet
    assert(r1.weights.keys.forall(hand => !hand.asSet.exists(dead.contains)))
    assert(math.abs(r1.weights.values.sum - 1.0) < 1e-9)
    assertEquals(r1.weights, r2.weights)
  }

  test("compute returns small finite bunching delta and handles larger fold sets") {
    withCpuPreflopBackend {
      val hero = hole("Ah", "Js")
      val table = TableRanges.defaults(TableFormat.NineMax)
      val foldsToCutoff = TableFormat.NineMax.foldsBeforeOpener(Position.Cutoff).map(PreflopFold(_))

      val result = BunchingEffect.compute(
        hero = hero,
        board = Board.empty,
        folds = foldsToCutoff,
        tableRanges = table,
        villainPos = Position.BigBlind,
        trials = 250,
        equityTrials = 900,
        rng = new Random(99)
      )

      assert(result.adjustedRange.weights.nonEmpty)
      assert(result.naiveRange.weights.nonEmpty)
      assert(!result.bunchingDelta.isNaN)
      assert(math.abs(result.bunchingDelta) < 0.5)
      assertEquals(result.adjustedEquity.trials, 900)
      assertEquals(result.naiveEquity.trials, 900)

      val manyFolds = Vector(
        PreflopFold(Position.UTG),
        PreflopFold(Position.UTG1),
        PreflopFold(Position.UTG2),
        PreflopFold(Position.Middle),
        PreflopFold(Position.Hijack),
        PreflopFold(Position.Cutoff),
        PreflopFold(Position.Button),
        PreflopFold(Position.SmallBlind)
      )
      val manyRange = BunchingEffect.adjustedRange(
        hero = hero,
        board = Board.empty,
        folds = manyFolds,
        tableRanges = table,
        villainPos = Position.BigBlind,
        trials = 80,
        rng = new Random(101)
      )
      assert(manyRange.weights.nonEmpty)
    }
  }

  test("computeForOpener convenience matches explicit folds invocation") {
    withCpuPreflopBackend {
      val hero = hole("Ac", "Kh")
      val table = TableRanges.defaults(TableFormat.NineMax)
      val openerPos = Position.Cutoff
      val villainPos = Position.BigBlind
      val seed = 12345L

      val explicit = BunchingEffect.compute(
        hero = hero,
        board = Board.empty,
        folds = TableFormat.NineMax.foldsBeforeOpener(openerPos).map(PreflopFold(_)),
        tableRanges = table,
        villainPos = villainPos,
        trials = 250,
        equityTrials = 900,
        rng = new Random(seed)
      )
      val convenience = BunchingEffect.computeForOpener(
        hero = hero,
        tableRanges = table,
        openerPos = openerPos,
        villainPos = villainPos,
        trials = 250,
        equityTrials = 900,
        rng = new Random(seed)
      )

      assertEquals(convenience.adjustedRange.weights, explicit.adjustedRange.weights)
      assertEquals(convenience.naiveRange.weights, explicit.naiveRange.weights)
      assertEquals(convenience.bunchingDelta, explicit.bunchingDelta)
    }
  }
