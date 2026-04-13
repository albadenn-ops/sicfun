package sicfun.holdem.runtime

import munit.FunSuite
import sicfun.holdem.types.*
import sicfun.holdem.engine.UpstreamSource
import sicfun.holdem.engine.inference.{ActionRecommendation, ActionEvaluation}
import sicfun.holdem.strategic.types.*

class StrategicLifecycleHelperTest extends FunSuite:

  private def testHeroCards: HoleCards =
    val as = sicfun.core.Card.parse("As").get
    val kh = sicfun.core.Card.parse("Kh").get
    HoleCards.from(Vector(as, kh))

  private def minimalState: GameState =
    GameState(
      street = Street.Flop,
      board = Board.empty,
      pot = 100.0,
      toCall = 50.0,
      position = Position.Button,
      stackSize = 1000.0,
      betHistory = Vector.empty
    )

  test("position mapping routes villain actions to stable rival ID"):
    val villainId = PlayerId("villain")
    val helper = StrategicLifecycleHelper.create()
    helper.initSession(
      rivalIds = Vector(villainId),
      positionMapping = Map(Position.BigBlind -> villainId)
    )
    helper.startHand(testHeroCards)
    helper.observeVillainAction(Position.BigBlind, PokerAction.Call, minimalState)
    assert(helper.engine.sessionState.rivalBeliefs.contains(villainId))

  test("position mapping update reflects seat rotation"):
    val villainId = PlayerId("villain")
    val helper = StrategicLifecycleHelper.create()
    helper.initSession(
      rivalIds = Vector(villainId),
      positionMapping = Map(Position.BigBlind -> villainId)
    )
    helper.updatePositionMapping(Map(Position.Button -> villainId))
    assertEquals(
      helper.positionMapping,
      Map(Position.Button -> villainId)
    )

  test("decideWithOverlay extracts EVs and returns OverlayResult"):
    val villainId = PlayerId("villain")
    val helper = StrategicLifecycleHelper.create()
    helper.initSession(
      rivalIds = Vector(villainId),
      positionMapping = Map(Position.BigBlind -> villainId)
    )
    helper.startHand(testHeroCards)
    val recommendation = ActionRecommendation(
      heroEquity = EquityEstimate(mean = 0.6, variance = 0.01, stderr = 0.003, trials = 1000, winRate = 0.5, tieRate = 0.1, lossRate = 0.4),
      actionEvaluations = Vector(
        ActionEvaluation(PokerAction.Call, 5.0),
        ActionEvaluation(PokerAction.Raise(2.0), 10.0)
      ),
      bestAction = PokerAction.Raise(2.0)
    )
    val candidates = Vector(PokerAction.Call, PokerAction.Raise(2.0))
    val result = helper.decideWithOverlay(minimalState, candidates, recommendation)
    assertEquals(result.upstreamAction, PokerAction.Raise(2.0))
    assertEquals(result.upstreamSource, UpstreamSource.Adaptive)

  test("decideWithOverlay respects multiway upstream source"):
    val helper = StrategicLifecycleHelper.create()
    helper.initSession(
      rivalIds = Vector(PlayerId("v1"), PlayerId("v2")),
      positionMapping = Map(Position.BigBlind -> PlayerId("v1"), Position.UTG -> PlayerId("v2"))
    )
    helper.startHand(testHeroCards)
    val recommendation = ActionRecommendation(
      heroEquity = EquityEstimate(mean = 0.5, variance = 0.01, stderr = 0.003, trials = 1000, winRate = 0.4, tieRate = 0.1, lossRate = 0.5),
      actionEvaluations = Vector(ActionEvaluation(PokerAction.Check, 0.0)),
      bestAction = PokerAction.Check
    )
    val result = helper.decideWithOverlay(
      minimalState, Vector(PokerAction.Check), recommendation,
      upstreamSource = UpstreamSource.Multiway(2)
    )
    assertEquals(result.upstreamSource, UpstreamSource.Multiway(2))
