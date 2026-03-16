package sicfun.holdem.engine

import munit.FunSuite
import sicfun.core.Card
import sicfun.holdem.types.*

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
