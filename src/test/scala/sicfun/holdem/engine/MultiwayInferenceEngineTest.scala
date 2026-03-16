package sicfun.holdem.engine

import munit.FunSuite
import sicfun.core.{Card, DiscreteDistribution, MultinomialLogistic}
import sicfun.holdem.cli.CliHelpers
import sicfun.holdem.equity.{PreflopFold, TableFormat, TableRanges}
import sicfun.holdem.model.PokerActionModel
import sicfun.holdem.types.*

import scala.util.Random

class MultiwayInferenceEngineTest extends FunSuite:
  test("estimateRaiseResponseFromRange renormalizes legal fold and continue mass when facing a raise") {
    val response = MultiwayInferenceEngine.estimateRaiseResponseFromRange(
      range = DiscreteDistribution(Map(
        hole("AhKd") -> 0.4,
        hole("7c2d") -> 0.6
      )),
      responseState = GameState(
        street = Street.Flop,
        board = board("Qs", "Jh", "4c"),
        pot = 5.0,
        toCall = 2.5,
        position = Position.BigBlind,
        stackSize = 80.0,
        betHistory = Vector.empty
      ),
      actionModel = PokerActionModel.uniform,
      raiseAction = PokerAction.Raise(2.5)
    )

    assertEqualsDouble(response.foldProbability, 1.0 / 3.0, 1e-9)
    assertEqualsDouble(response.continueProbability, 2.0 / 3.0, 1e-9)
    assertEqualsDouble(response.foldProbability + response.continueProbability, 1.0, 1e-9)
    assertEqualsDouble(response.continuationRange.probabilityOf(hole("AhKd")), 0.4, 1e-9)
    assertEqualsDouble(response.continuationRange.probabilityOf(hole("7c2d")), 0.6, 1e-9)
  }

  test("estimateRaiseResponseFromRange shifts continuation range toward stronger hands under a non-uniform model") {
    val foldWeights = Vector(0.0, 0.0, 0.0, 0.0, -6.0)
    val checkWeights = Vector.fill(5)(0.0)
    val callWeights = Vector(0.0, 0.0, 0.0, 0.0, 5.5)
    val raiseWeights = Vector(0.0, 0.0, 0.0, 0.0, 3.5)
    val model = PokerActionModel(
      logistic = MultinomialLogistic(
        weights = Vector(foldWeights, checkWeights, callWeights, raiseWeights),
        bias = Vector(1.0, -8.0, -1.0, -2.0)
      ),
      categoryIndex = PokerActionModel.defaultCategoryIndex,
      featureDimension = 5
    )
    val strong = hole("QhQs")
    val weak = hole("3c4d")
    val response = MultiwayInferenceEngine.estimateRaiseResponseFromRange(
      range = DiscreteDistribution(Map(
        strong -> 0.5,
        weak -> 0.5
      )),
      responseState = GameState(
        street = Street.River,
        board = board("Ah", "Ad", "Kc", "2s", "7h"),
        pot = 8.0,
        toCall = 3.0,
        position = Position.BigBlind,
        stackSize = 70.0,
        betHistory = Vector.empty
      ),
      actionModel = model,
      raiseAction = PokerAction.Raise(2.5)
    )

    assert(response.continueProbability > response.foldProbability)
    assert(response.continuationRange.probabilityOf(strong) > 0.5)
    assert(response.continuationRange.probabilityOf(weak) < 0.5)
  }

  test("estimateEquityAgainstOpponentRanges uses exact multiway share when river ranges are tiny") {
    val hero = hole("3c4d")
    val riverBoard = board("Ah", "Ad", "Kc", "Kd", "2s")
    val estimate = MultiwayInferenceEngine.estimateEquityAgainstOpponentRanges(
      hero = hero,
      board = riverBoard,
      opponentRanges = Vector(
        singleRange("QhQs"),
        singleRange("JhJs")
      ),
      equityTrials = 32,
      rng = new Random(1L)
    )

    assertEquals(estimate.mean, 0.0)
    assertEquals(estimate.winRate, 0.0)
    assertEquals(estimate.tieRate, 0.0)
    assertEquals(estimate.lossRate, 1.0)
    assertEquals(estimate.trials, 1)
  }

  test("recommendActionAgainstOpponentRanges folds a dead bluff-catcher against two stronger ranges") {
    val riverBoard = board("Ah", "Ad", "Kc", "Kd", "2s")
    val state = GameState(
      street = Street.River,
      board = riverBoard,
      pot = 6.0,
      toCall = 1.0,
      position = Position.Button,
      stackSize = 50.0,
      betHistory = Vector.empty
    )

    val recommendation = MultiwayInferenceEngine.recommendActionAgainstOpponentRanges(
      hero = hole("3c4d"),
      state = state,
      opponentRanges = Vector(
        singleRange("QhQs"),
        singleRange("JhJs")
      ),
      candidateActions = Vector(PokerAction.Fold, PokerAction.Call),
      equityTrials = 32,
      rng = new Random(2L)
    )

    assertEquals(recommendation.bestAction, PokerAction.Fold)
    assert(recommendation.actionEvaluations.exists(eval => eval.action == PokerAction.Call && eval.expectedValue < 0.0))
  }

  test("inferOpponentPosteriors returns one posterior per opponent and respects overrides") {
    val hero = hole("AcKh")
    val state = GameState(
      street = Street.Flop,
      board = board("Qs", "Jh", "4c"),
      pot = 7.0,
      toCall = 2.5,
      position = Position.Cutoff,
      stackSize = 90.0,
      betHistory = Vector.empty
    )
    val tableRanges = TableRanges.defaults(TableFormat.SixMax)
    val folds = TableFormat.SixMax.foldsBeforeOpener(Position.Cutoff).map(PreflopFold(_))
    val overridePosterior = DiscreteDistribution(Map(hole("9h9d") -> 1.0))
    val result = MultiwayInferenceEngine.inferOpponentPosteriors(
      hero = hero,
      state = state,
      tableRanges = tableRanges,
      actionModel = PokerActionModel.uniform,
      opponents = Vector(
        MultiwayInferenceEngine.OpponentInput(
          position = Position.Button,
          folds = folds,
          observations = Vector.empty,
          stackSize = 85.0,
          posteriorOverride = Some(overridePosterior)
        ),
        MultiwayInferenceEngine.OpponentInput(
          position = Position.BigBlind,
          folds = folds,
          observations = Vector.empty,
          stackSize = 88.0
        )
      ),
      bunchingTrials = 8,
      rng = new Random(3L)
    )

    assertEquals(result.keySet, Set(Position.Button, Position.BigBlind))
    assertEqualsDouble(result(Position.Button).posterior.probabilityOf(hole("9h9d")), 1.0, 1e-12)
    assert(result(Position.BigBlind).posterior.weights.nonEmpty)
  }

  test("inferAndRecommend exposes a reusable multiway inferred-play result") {
    val hero = hole("3c4d")
    val state = GameState(
      street = Street.River,
      board = board("Ah", "Ad", "Kc", "Kd", "2s"),
      pot = 6.0,
      toCall = 1.0,
      position = Position.Button,
      stackSize = 50.0,
      betHistory = Vector.empty
    )

    val result = MultiwayInferenceEngine.inferAndRecommend(
      hero = hero,
      state = state,
      actorContribution = 0.0,
      actorBetHistoryIndex = 0,
      tableRanges = TableRanges.defaults(TableFormat.SixMax),
      actionModel = PokerActionModel.uniform,
      opponents = Vector(
        MultiwayInferenceEngine.OpponentInput(
          position = Position.SmallBlind,
          folds = Vector.empty,
          observations = Vector.empty,
          stackSize = 50.0,
          posteriorOverride = Some(singleRange("QhQs"))
        ),
        MultiwayInferenceEngine.OpponentInput(
          position = Position.BigBlind,
          folds = Vector.empty,
          observations = Vector.empty,
          stackSize = 50.0,
          posteriorOverride = Some(singleRange("JhJs"))
        )
      ),
      candidateActions = Vector(PokerAction.Fold, PokerAction.Call),
      bunchingTrials = 8,
      equityTrialsForOpponentCount = _ => 32,
      rng = new Random(4L)
    )

    assertEquals(result.opponentPosteriors.keySet, Set(Position.SmallBlind, Position.BigBlind))
    assertEquals(result.recommendation.bestAction, PokerAction.Fold)
  }

  private def hole(token: String): HoleCards =
    CliHelpers.parseHoleCards(token)

  private def board(tokens: String*): Board =
    Board.from(tokens.map(parseCard))

  private def parseCard(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card token: $token"))

  private def singleRange(token: String): DiscreteDistribution[HoleCards] =
    DiscreteDistribution(Map(hole(token) -> 1.0))
