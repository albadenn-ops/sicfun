package sicfun.holdem.engine

import munit.FunSuite
import sicfun.core.Card
import sicfun.holdem.types.*

/** Tests for [[VillainResponseProfile]], [[UniformVillainResponseModel]],
  * [[VillainResponseModel.BinaryHandRanges]], and [[VillainResponseModel.sbVsBbOpenDefault]].
  *
  * Validates:
  *   - '''VillainResponseProfile''': valid probability triples accepted; negative values,
  *     non-unit sums, and edge cases (all-fold, all-call, all-raise) handled correctly.
  *   - '''UniformVillainResponseModel''': returns the same profile regardless of villain hand.
  *   - '''BinaryHandRanges''': raise/call/fold priority is correct; raiseRange takes priority
  *     over callRange; non-raise hero actions always produce a call response.
  *   - '''fromRanges''': valid range strings parse successfully; invalid strings return Left.
  *   - '''sbVsBbOpenDefault''': non-empty raise and call ranges; known hands produce
  *     expected fold/call/raise responses.
  */
class VillainResponseModelTest extends FunSuite:
  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  // ---------------------------------------------------------------------------
  // VillainResponseProfile
  // ---------------------------------------------------------------------------

  test("VillainResponseProfile accepts valid probability triple") {
    val profile = VillainResponseProfile(0.3, 0.5, 0.2)
    assertEqualsDouble(profile.foldProbability, 0.3, 1e-12)
    assertEqualsDouble(profile.callProbability, 0.5, 1e-12)
    assertEqualsDouble(profile.raiseProbability, 0.2, 1e-12)
  }

  test("VillainResponseProfile rejects negative fold probability") {
    intercept[IllegalArgumentException] {
      VillainResponseProfile(-0.1, 0.6, 0.5)
    }
  }

  test("VillainResponseProfile rejects negative call probability") {
    intercept[IllegalArgumentException] {
      VillainResponseProfile(0.5, -0.1, 0.6)
    }
  }

  test("VillainResponseProfile rejects negative raise probability") {
    intercept[IllegalArgumentException] {
      VillainResponseProfile(0.5, 0.6, -0.1)
    }
  }

  test("VillainResponseProfile rejects probabilities that do not sum to 1") {
    intercept[IllegalArgumentException] {
      VillainResponseProfile(0.3, 0.3, 0.3)
    }
  }

  test("VillainResponseProfile continueProbability is call + raise") {
    val profile = VillainResponseProfile(0.4, 0.35, 0.25)
    assertEqualsDouble(profile.continueProbability, 0.60, 1e-12)
  }

  test("VillainResponseProfile all-fold is valid") {
    val profile = VillainResponseProfile(1.0, 0.0, 0.0)
    assertEqualsDouble(profile.continueProbability, 0.0, 1e-12)
  }

  test("VillainResponseProfile all-call is valid") {
    val profile = VillainResponseProfile(0.0, 1.0, 0.0)
    assertEqualsDouble(profile.continueProbability, 1.0, 1e-12)
  }

  test("VillainResponseProfile all-raise is valid") {
    val profile = VillainResponseProfile(0.0, 0.0, 1.0)
    assertEqualsDouble(profile.continueProbability, 1.0, 1e-12)
  }

  // ---------------------------------------------------------------------------
  // UniformVillainResponseModel
  // ---------------------------------------------------------------------------

  test("UniformVillainResponseModel returns the same profile regardless of villain hand") {
    val expectedProfile = VillainResponseProfile(0.5, 0.3, 0.2)
    val model = new UniformVillainResponseModel:
      def responseProfile(state: GameState, heroAction: PokerAction): VillainResponseProfile =
        expectedProfile

    val state = GameState(
      street = Street.Preflop,
      board = Board.empty,
      pot = 3.0,
      toCall = 1.0,
      position = Position.BigBlind,
      stackSize = 99.0,
      betHistory = Vector.empty
    )

    val hand1 = hole("Ah", "Kd")
    val hand2 = hole("2c", "7s")

    val r1 = model.response(hand1, state, PokerAction.Raise(3.0))
    val r2 = model.response(hand2, state, PokerAction.Raise(3.0))

    assertEqualsDouble(r1.foldProbability, expectedProfile.foldProbability, 1e-12)
    assertEqualsDouble(r2.foldProbability, expectedProfile.foldProbability, 1e-12)
    assertEqualsDouble(r1.callProbability, r2.callProbability, 1e-12)
    assertEqualsDouble(r1.raiseProbability, r2.raiseProbability, 1e-12)
  }

  // ---------------------------------------------------------------------------
  // BinaryHandRanges
  // ---------------------------------------------------------------------------

  test("BinaryHandRanges raises for hands in raiseRange when facing a Raise") {
    val raiseHand = hole("Ah", "Ad")
    val model = VillainResponseModel.BinaryHandRanges(
      raiseRange = Set(raiseHand),
      callRange = Set.empty
    )
    val state = GameState(
      street = Street.Preflop,
      board = Board.empty,
      pot = 3.0,
      toCall = 1.0,
      position = Position.BigBlind,
      stackSize = 99.0,
      betHistory = Vector.empty
    )

    val profile = model.response(raiseHand, state, PokerAction.Raise(3.0))
    assertEqualsDouble(profile.foldProbability, 0.0, 1e-12)
    assertEqualsDouble(profile.callProbability, 0.0, 1e-12)
    assertEqualsDouble(profile.raiseProbability, 1.0, 1e-12)
  }

  test("BinaryHandRanges calls for hands in callRange but not raiseRange when facing a Raise") {
    val callHand = hole("Kh", "Qd")
    val model = VillainResponseModel.BinaryHandRanges(
      raiseRange = Set.empty,
      callRange = Set(callHand)
    )
    val state = GameState(
      street = Street.Preflop,
      board = Board.empty,
      pot = 3.0,
      toCall = 1.0,
      position = Position.BigBlind,
      stackSize = 99.0,
      betHistory = Vector.empty
    )

    val profile = model.response(callHand, state, PokerAction.Raise(3.0))
    assertEqualsDouble(profile.foldProbability, 0.0, 1e-12)
    assertEqualsDouble(profile.callProbability, 1.0, 1e-12)
    assertEqualsDouble(profile.raiseProbability, 0.0, 1e-12)
  }

  test("BinaryHandRanges folds for hands in neither range when facing a Raise") {
    val junkHand = hole("2c", "7d")
    val model = VillainResponseModel.BinaryHandRanges(
      raiseRange = Set(hole("Ah", "Ad")),
      callRange = Set(hole("Kh", "Qd"))
    )
    val state = GameState(
      street = Street.Preflop,
      board = Board.empty,
      pot = 3.0,
      toCall = 1.0,
      position = Position.BigBlind,
      stackSize = 99.0,
      betHistory = Vector.empty
    )

    val profile = model.response(junkHand, state, PokerAction.Raise(3.0))
    assertEqualsDouble(profile.foldProbability, 1.0, 1e-12)
    assertEqualsDouble(profile.callProbability, 0.0, 1e-12)
    assertEqualsDouble(profile.raiseProbability, 0.0, 1e-12)
  }

  test("BinaryHandRanges hand in both raiseRange and callRange raises (raiseRange takes priority)") {
    val hand = hole("Ah", "Ad")
    val model = VillainResponseModel.BinaryHandRanges(
      raiseRange = Set(hand),
      callRange = Set(hand)
    )
    val state = GameState(
      street = Street.Preflop,
      board = Board.empty,
      pot = 3.0,
      toCall = 1.0,
      position = Position.BigBlind,
      stackSize = 99.0,
      betHistory = Vector.empty
    )

    val profile = model.response(hand, state, PokerAction.Raise(3.0))
    assertEqualsDouble(profile.raiseProbability, 1.0, 1e-12)
  }

  test("BinaryHandRanges always calls for non-Raise hero actions regardless of hand") {
    val junkHand = hole("2c", "7d")
    val model = VillainResponseModel.BinaryHandRanges(
      raiseRange = Set(hole("Ah", "Ad")),
      callRange = Set(hole("Kh", "Qd"))
    )
    val state = GameState(
      street = Street.Preflop,
      board = Board.empty,
      pot = 3.0,
      toCall = 1.0,
      position = Position.BigBlind,
      stackSize = 99.0,
      betHistory = Vector.empty
    )

    for heroAction <- Seq(PokerAction.Fold, PokerAction.Check, PokerAction.Call) do
      val profile = model.response(junkHand, state, heroAction)
      assertEqualsDouble(profile.foldProbability, 0.0, 1e-12)
      assertEqualsDouble(profile.callProbability, 1.0, 1e-12)
      assertEqualsDouble(profile.raiseProbability, 0.0, 1e-12)
  }

  // ---------------------------------------------------------------------------
  // fromRanges
  // ---------------------------------------------------------------------------

  test("fromRanges succeeds with valid range strings") {
    val result = VillainResponseModel.fromRanges(
      raiseRange = "AA, KK",
      callRange = "QQ, JJ"
    )
    assert(result.isRight, s"expected Right but got $result")
    val model = result.toOption.get
    assert(model.raiseRange.nonEmpty)
    assert(model.callRange.nonEmpty)
  }

  test("fromRanges returns Left on invalid raiseRange") {
    val result = VillainResponseModel.fromRanges(
      raiseRange = "XYZZY",
      callRange = "AA"
    )
    assert(result.isLeft)
    assert(result.left.toOption.get.contains("raiseRange"))
  }

  test("fromRanges returns Left on invalid callRange") {
    val result = VillainResponseModel.fromRanges(
      raiseRange = "AA",
      callRange = "XYZZY"
    )
    assert(result.isLeft)
    assert(result.left.toOption.get.contains("callRange"))
  }

  // ---------------------------------------------------------------------------
  // sbVsBbOpenDefault
  // ---------------------------------------------------------------------------

  test("sbVsBbOpenDefault is a valid BinaryHandRanges with non-empty raise and call ranges") {
    val model = VillainResponseModel.sbVsBbOpenDefault
    assert(model.raiseRange.nonEmpty)
    assert(model.callRange.nonEmpty)
  }

  test("sbVsBbOpenDefault produces fold/call/raise profiles for known hands") {
    val model = VillainResponseModel.sbVsBbOpenDefault
    val state = GameState(
      street = Street.Preflop,
      board = Board.empty,
      pot = 3.0,
      toCall = 1.0,
      position = Position.BigBlind,
      stackSize = 99.0,
      betHistory = Vector.empty
    )

    // AA should be in the raise range (66+)
    val aces = hole("Ah", "Ad")
    val acesProfile = model.response(aces, state, PokerAction.Raise(3.0))
    assertEqualsDouble(acesProfile.raiseProbability, 1.0, 1e-12)

    // A junk hand should fold
    val junk = hole("2c", "7d")
    val junkProfile = model.response(junk, state, PokerAction.Raise(3.0))
    assertEqualsDouble(junkProfile.foldProbability, 1.0, 1e-12)
  }

  // ---------------------------------------------------------------------------
  // HandStrengthResponseModel
  // ---------------------------------------------------------------------------

  test("HandStrengthResponseModel: strong hand folds less than weak hand") {
    val base = VillainResponseProfile(0.40, 0.50, 0.10)
    val model = new HandStrengthResponseModel(() => base)

    val preflopState = GameState(
      street = Street.Preflop,
      board = Board.empty,
      pot = 6.0,
      toCall = 3.0,
      position = Position.BigBlind,
      stackSize = 97.0,
      betHistory = Vector.empty
    )

    val aces = hole("Ah", "Ad")
    val junk = hole("2c", "7d")
    val raise = PokerAction.Raise(6.0) // pot-sized raise → potOddsFactor = 1.0

    val acesProfile = model.response(aces, preflopState, raise)
    val junkProfile = model.response(junk, preflopState, raise)

    assert(
      acesProfile.foldProbability < junkProfile.foldProbability,
      s"aces fold (${acesProfile.foldProbability}) should be < junk fold (${junkProfile.foldProbability})"
    )
    assert(
      acesProfile.raiseProbability > junkProfile.raiseProbability,
      s"aces raise (${acesProfile.raiseProbability}) should be > junk raise (${junkProfile.raiseProbability})"
    )
  }

  test("HandStrengthResponseModel: bigger raise increases fold probability") {
    val base = VillainResponseProfile(0.40, 0.50, 0.10)
    val model = new HandStrengthResponseModel(() => base)

    val state = GameState(
      street = Street.Flop,
      board = Board(Vector(card("Kh"), card("9d"), card("3c"))),
      pot = 10.0,
      toCall = 5.0,
      position = Position.BigBlind,
      stackSize = 90.0,
      betHistory = Vector.empty
    )

    val hand = hole("Jh", "Td") // medium strength
    val smallRaise = PokerAction.Raise(5.0)  // 0.5x pot → factor 0.5
    val bigRaise = PokerAction.Raise(20.0)   // 2x pot → factor 2.0

    val smallProfile = model.response(hand, state, smallRaise)
    val bigProfile = model.response(hand, state, bigRaise)

    assert(
      bigProfile.foldProbability > smallProfile.foldProbability,
      s"big raise fold (${bigProfile.foldProbability}) should be > small raise fold (${smallProfile.foldProbability})"
    )
  }

  test("HandStrengthResponseModel: non-raise action returns pure call") {
    val base = VillainResponseProfile(0.40, 0.50, 0.10)
    val model = new HandStrengthResponseModel(() => base)

    val state = GameState(
      street = Street.Preflop,
      board = Board.empty,
      pot = 3.0,
      toCall = 1.0,
      position = Position.BigBlind,
      stackSize = 99.0,
      betHistory = Vector.empty
    )

    val hand = hole("Ah", "Ad")
    for action <- Seq(PokerAction.Fold, PokerAction.Check, PokerAction.Call) do
      val profile = model.response(hand, state, action)
      assertEqualsDouble(profile.foldProbability, 0.0, 1e-12)
      assertEqualsDouble(profile.callProbability, 1.0, 1e-12)
      assertEqualsDouble(profile.raiseProbability, 0.0, 1e-12)
  }

  test("HandStrengthResponseModel.modulate: probabilities always sum to 1") {
    val base = VillainResponseProfile(0.40, 0.50, 0.10)
    // Test a range of strengths and pot-odds factors
    for
      s <- Seq(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
      pof <- Seq(0.5, 1.0, 1.5, 2.0)
    do
      val profile = HandStrengthResponseModel.modulate(base, s, pof)
      val sum = profile.foldProbability + profile.callProbability + profile.raiseProbability
      assertEqualsDouble(sum, 1.0, 1e-9, s"sum=$sum for strength=$s, potOddsFactor=$pof")
      assert(profile.foldProbability >= 0.0, s"negative fold for s=$s, pof=$pof")
      assert(profile.callProbability >= 0.0, s"negative call for s=$s, pof=$pof")
      assert(profile.raiseProbability >= 0.0, s"negative raise for s=$s, pof=$pof")
  }

  test("HandStrengthResponseModel.modulate: zero-strength hand folds maximally") {
    val base = VillainResponseProfile(0.40, 0.50, 0.10)
    val profile = HandStrengthResponseModel.modulate(base, strength = 0.0, potOddsFactor = 1.0)
    // At strength=0: rawFold = 0.40 * 1.0 * 2.0 * 1.0 = 0.80
    // rawRaise = 0.10 * 2.0 * 0.0 = 0.0
    assertEqualsDouble(profile.foldProbability, 0.80, 1e-9)
    assertEqualsDouble(profile.raiseProbability, 0.0, 1e-9)
  }

  test("HandStrengthResponseModel.modulate: full-strength hand never folds") {
    val base = VillainResponseProfile(0.40, 0.50, 0.10)
    val profile = HandStrengthResponseModel.modulate(base, strength = 1.0, potOddsFactor = 1.0)
    // At strength=1: rawFold = 0.40 * 1.0 * 2.0 * 0.0 = 0.0
    // rawRaise = 0.10 * 2.0 * 1.0 = 0.20
    assertEqualsDouble(profile.foldProbability, 0.0, 1e-9)
    assertEqualsDouble(profile.raiseProbability, 0.20, 1e-9)
  }

  test("HandStrengthResponseModel: postflop strength uses board texture") {
    val base = VillainResponseProfile(0.40, 0.50, 0.10)
    val model = new HandStrengthResponseModel(() => base)

    // Board: K♥ 9♦ 3♣ — villain with K♠Q♠ has top pair, villain with 2♠4♠ has nothing
    val board = Board(Vector(card("Kh"), card("9d"), card("3c")))
    val state = GameState(
      street = Street.Flop,
      board = board,
      pot = 10.0,
      toCall = 5.0,
      position = Position.BigBlind,
      stackSize = 90.0,
      betHistory = Vector.empty
    )

    val topPair = hole("Ks", "Qs")
    val air = hole("2s", "4s")
    val raise = PokerAction.Raise(10.0)

    val topPairProfile = model.response(topPair, state, raise)
    val airProfile = model.response(air, state, raise)

    assert(
      topPairProfile.foldProbability < airProfile.foldProbability,
      s"top pair fold (${topPairProfile.foldProbability}) should be < air fold (${airProfile.foldProbability})"
    )
  }
