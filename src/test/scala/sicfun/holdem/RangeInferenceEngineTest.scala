package sicfun.holdem

import munit.FunSuite
import sicfun.core.{Card, DiscreteDistribution}

import scala.util.Random

class RangeInferenceEngineTest extends FunSuite:
  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  test("inferPosterior collapses range after informative villain action") {
    val hero = hole("Qc", "Qs")
    val board = Board.from(Seq(card("Td"), card("9h"), card("8d")))
    val table = TableRanges.defaults(TableFormat.NineMax)
    val villainPos = Position.BigBlind
    val folds = Vector(
      PreflopFold(Position.UTG),
      PreflopFold(Position.UTG1),
      PreflopFold(Position.UTG2),
      PreflopFold(Position.Middle),
      PreflopFold(Position.Cutoff),
      PreflopFold(Position.Button),
      PreflopFold(Position.SmallBlind)
    )

    val state = GameState(
      street = Street.Flop,
      board = board,
      pot = 20.0,
      toCall = 10.0,
      position = villainPos,
      stackSize = 180.0,
      betHistory = Vector.empty
    )

    val premium = hole("Ah", "Kh")
    val trash = hole("7c", "2d")
    val trainingData = Vector.fill(24)((state, premium, PokerAction.Raise(25.0))) ++
      Vector.fill(24)((state, trash, PokerAction.Fold))
    val actionModel = PokerActionModel.train(trainingData, learningRate = 0.1, iterations = 600)

    val result = RangeInferenceEngine.inferPosterior(
      hero = hero,
      board = board,
      folds = folds,
      tableRanges = table,
      villainPos = villainPos,
      observations = Seq(VillainObservation(PokerAction.Raise(25.0), state)),
      actionModel = actionModel,
      bunchingTrials = 400,
      rng = new Random(13)
    )

    val priorPremium = result.prior.probabilityOf(premium)
    val priorTrash = result.prior.probabilityOf(trash)
    val postPremium = result.posterior.probabilityOf(premium)
    val postTrash = result.posterior.probabilityOf(trash)
    assert(priorTrash > 0.0, "expected trash hand to exist in prior support")

    val priorRatio = priorPremium / priorTrash
    val postRatio = postPremium / postTrash
    assert(
      postRatio > priorRatio,
      s"expected posterior premium/trash ratio to increase ($priorRatio -> $postRatio)"
    )
    assert(result.collapse.entropyReduction > 0.0)
    assert(result.collapse.collapseRatio > 0.0)
  }

  test("recommendAction default chip EV chooses call over fold when +EV") {
    val hero = hole("As", "Ad")
    val posterior = DiscreteDistribution(Map(hole("7c", "2d") -> 1.0))
    val state = GameState(
      street = Street.Preflop,
      board = Board.empty,
      pot = 10.0,
      toCall = 2.0,
      position = Position.Button,
      stackSize = 100.0,
      betHistory = Vector.empty
    )

    val rec = RangeInferenceEngine.recommendAction(
      hero = hero,
      state = state,
      posterior = posterior,
      candidateActions = Vector(PokerAction.Fold, PokerAction.Call),
      equityTrials = 4_000,
      rng = new Random(21)
    )

    assertEquals(rec.bestAction, PokerAction.Call)
    val callEv = rec.actionEvaluations.find(_.action == PokerAction.Call).map(_.expectedValue)
      .getOrElse(fail("missing call EV"))
    val foldEv = rec.actionEvaluations.find(_.action == PokerAction.Fold).map(_.expectedValue)
      .getOrElse(fail("missing fold EV"))

    val expectedCall = HoldemEquity.evCall(state.pot, state.toCall, rec.heroEquity.mean)
    assert(math.abs(callEv - expectedCall) < 1e-12)
    assertEquals(foldEv, 0.0)
    assert(callEv > foldEv)
  }

  test("recommendAction supports custom action value models") {
    val hero = hole("Ah", "Kd")
    val board = Board.from(Seq(card("2c"), card("7h"), card("Jd")))
    val posterior = DiscreteDistribution(Map(hole("Qc", "Qh") -> 1.0))
    val state = GameState(
      street = Street.Flop,
      board = board,
      pot = 30.0,
      toCall = 0.0,
      position = Position.Button,
      stackSize = 120.0,
      betHistory = Vector.empty
    )

    val preferRaise = new ActionValueModel:
      def expectedValue(action: PokerAction, state: GameState, heroEquity: Double): Double = action match
        case PokerAction.Raise(amount) => 10.0 + amount
        case PokerAction.Check => 1.0
        case PokerAction.Call => 0.5
        case PokerAction.Fold => 0.0

    val rec = RangeInferenceEngine.recommendAction(
      hero = hero,
      state = state,
      posterior = posterior,
      candidateActions = Vector(PokerAction.Fold, PokerAction.Check, PokerAction.Raise(15.0)),
      actionValueModel = preferRaise,
      equityTrials = 1_000,
      rng = new Random(31)
    )

    assertEquals(rec.bestAction, PokerAction.Raise(15.0))
    val raiseEv = rec.actionEvaluations.find(_.action == PokerAction.Raise(15.0)).map(_.expectedValue)
      .getOrElse(fail("missing raise EV"))
    assertEquals(raiseEv, 25.0)
  }

  test("SB-vs-BB default response model classifies fold call and raise buckets") {
    val model = VillainResponseModel.sbVsBbOpenDefault
    val state = GameState(
      street = Street.Preflop,
      board = Board.empty,
      pot = 1.5,
      toCall = 0.5,
      position = Position.BigBlind,
      stackSize = 100.0,
      betHistory = Vector.empty
    )

    val foldHand = hole("7c", "2d")
    val callHand = hole("9c", "8c")
    val raiseHand = hole("Ah", "Ad")

    val foldResp = model.response(foldHand, state, PokerAction.Raise(2.5))
    val callResp = model.response(callHand, state, PokerAction.Raise(2.5))
    val raiseResp = model.response(raiseHand, state, PokerAction.Raise(2.5))

    assertEquals(foldResp.foldProbability, 1.0)
    assertEquals(callResp.callProbability, 1.0)
    assertEquals(raiseResp.raiseProbability, 1.0)
  }

  test("SB-vs-BB response-aware raise EV adapts to weak vs strong villain posterior") {
    val responseModel = VillainResponseModel.sbVsBbOpenDefault
    val hero = hole("9c", "8d")
    val state = GameState(
      street = Street.Preflop,
      board = Board.empty,
      pot = 1.5,
      toCall = 0.5,
      position = Position.SmallBlind,
      stackSize = 99.5,
      betHistory = Vector.empty
    )
    val candidateActions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(2.5))

    val weakPosterior = DiscreteDistribution(Map(hole("7c", "2d") -> 1.0))
    val strongPosterior = DiscreteDistribution(Map(hole("Ah", "Ad") -> 1.0))

    val weakRec = RangeInferenceEngine.recommendAction(
      hero = hero,
      state = state,
      posterior = weakPosterior,
      candidateActions = candidateActions,
      villainResponseModel = Some(responseModel),
      equityTrials = 4_000,
      rng = new Random(57)
    )
    val strongRec = RangeInferenceEngine.recommendAction(
      hero = hero,
      state = state,
      posterior = strongPosterior,
      candidateActions = candidateActions,
      villainResponseModel = Some(responseModel),
      equityTrials = 4_000,
      rng = new Random(58)
    )

    assertEquals(weakRec.bestAction, PokerAction.Raise(2.5))
    assertEquals(strongRec.bestAction, PokerAction.Fold)

    val weakRaiseEv = weakRec.actionEvaluations.find(_.action == PokerAction.Raise(2.5)).map(_.expectedValue)
      .getOrElse(fail("missing weak raise EV"))
    val strongRaiseEv = strongRec.actionEvaluations.find(_.action == PokerAction.Raise(2.5)).map(_.expectedValue)
      .getOrElse(fail("missing strong raise EV"))
    assert(weakRaiseEv > strongRaiseEv)
  }

  test("inferAndRecommend composes inference and recommendation end-to-end") {
    val hero = hole("Ac", "Kh")
    val board = Board.from(Seq(card("Ts"), card("9h"), card("8d")))
    val state = GameState(
      street = Street.Flop,
      board = board,
      pot = 24.0,
      toCall = 8.0,
      position = Position.Button,
      stackSize = 150.0,
      betHistory = Vector.empty
    )

    val table = TableRanges.defaults(TableFormat.NineMax)
    val folds = TableFormat.NineMax.foldsBeforeOpener(Position.Cutoff).map(PreflopFold(_))
    val model = PokerActionModel.uniform

    val preferFold = new ActionValueModel:
      def expectedValue(action: PokerAction, state: GameState, heroEquity: Double): Double = action match
        case PokerAction.Fold => 0.0
        case _ => -1.0

    val result = RangeInferenceEngine.inferAndRecommend(
      hero = hero,
      state = state,
      folds = folds,
      tableRanges = table,
      villainPos = Position.BigBlind,
      observations = Seq.empty,
      actionModel = model,
      candidateActions = Vector(PokerAction.Fold, PokerAction.Call),
      actionValueModel = preferFold,
      bunchingTrials = 300,
      equityTrials = 1_200,
      rng = new Random(41)
    )

    val totalPosterior = result.posteriorInference.posterior.weights.values.sum
    assert(math.abs(totalPosterior - 1.0) < 1e-9)
    assert(math.abs(result.posteriorInference.collapse.entropyReduction) < 1e-9)
    assert(math.abs(result.posteriorInference.collapse.collapseRatio) < 1e-9)
    assertEquals(result.recommendation.bestAction, PokerAction.Fold)
  }

  test("validation rejects invalid trials and empty candidate action sets") {
    val hero = hole("As", "Kd")
    val board = Board.empty
    val state = GameState(
      street = Street.Preflop,
      board = board,
      pot = 6.0,
      toCall = 2.0,
      position = Position.Button,
      stackSize = 100.0,
      betHistory = Vector.empty
    )
    val table = TableRanges.defaults(TableFormat.NineMax)
    val folds = Vector(PreflopFold(Position.UTG))

    intercept[IllegalArgumentException] {
      RangeInferenceEngine.inferPosterior(
        hero = hero,
        board = board,
        folds = folds,
        tableRanges = table,
        villainPos = Position.BigBlind,
        observations = Seq.empty,
        actionModel = PokerActionModel.uniform,
        bunchingTrials = 0,
        rng = new Random(1)
      )
    }

    intercept[IllegalArgumentException] {
      RangeInferenceEngine.recommendAction(
        hero = hero,
        state = state,
        posterior = DiscreteDistribution(Map(hole("7c", "2d") -> 1.0)),
        candidateActions = Vector.empty,
        equityTrials = 100,
        rng = new Random(2)
      )
    }

    intercept[IllegalArgumentException] {
      RangeInferenceEngine.recommendAction(
        hero = hero,
        state = state,
        posterior = DiscreteDistribution(Map(hole("7c", "2d") -> 1.0)),
        candidateActions = Vector(PokerAction.Fold),
        actionValueModel = ActionValueModel.ChipEv(),
        equityTrials = 0,
        rng = new Random(3)
      )
    }
  }
