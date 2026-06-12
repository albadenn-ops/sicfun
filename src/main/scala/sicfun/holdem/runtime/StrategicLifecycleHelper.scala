package sicfun.holdem.runtime

import sicfun.holdem.types.*
import sicfun.holdem.engine.{StrategicEngine, OverlayResult, UpstreamSource}
import sicfun.holdem.engine.inference.ActionRecommendation
import sicfun.holdem.strategic.types.*

/** Per-engine strategic lifecycle helper. Each caller (hall, ACPC, Slumbot, advisor)
  * creates its own instance so position mapping state does not bleed across runners.
  *
  * Identity contract: Callers supply stable PlayerId per physical opponent, NOT
  * position-derived IDs. In heads-up match play the remote opponent flips between
  * Button and BigBlind each hand — Position.toString would split belief tracks.
  *
  * Within-hand uniqueness: the mapping must also be INJECTIVE per hand — two
  * positions must never share a PlayerId, because every engine session structure
  * (rivalBeliefs, exploitationStates, rivalSeats, endHand's showdown map, the
  * actor-attributed action history feeding anomaly detection) is keyed by
  * PlayerId and cannot represent two simultaneous seats under one id. The hall
  * enforces this at config time: strategic mode requires a villainPool at least
  * as large as the worst-case simultaneous villain seat count
  * (TexasHoldemPlayingHall.maxActiveVillainDemand), so its per-hand round-robin
  * never seats one profile twice.
  *
  *   - ACPC/Slumbot: PlayerId("villain")
  *   - Hall: PlayerId from VillainProfile name
  *   - Advisor: PlayerId("villain")
  */
private[holdem] final class StrategicLifecycleHelper(
    val engine: StrategicEngine
):

  /** Mutable position->rivalId mapping, updated each hand as seats rotate. */
  private var _positionMapping: Map[Position, PlayerId] = Map.empty

  def positionMapping: Map[Position, PlayerId] = _positionMapping

  def initSession(
      rivalIds: Vector[PlayerId],
      positionMapping: Map[Position, PlayerId],
      rivalSeats: Map[PlayerId, StrategicEngine.RivalSeatInfo] = Map.empty
  ): Unit =
    _positionMapping = positionMapping
    engine.initSession(rivalIds, rivalSeats)

  def updatePositionMapping(positionMapping: Map[Position, PlayerId]): Unit =
    _positionMapping = positionMapping

  def startHand(heroCards: HoleCards): Unit =
    engine.startHand(heroCards)

  /** Routes villain action to the correct stable rival ID via position mapping.
    * Silently ignores if the position has no mapping (e.g., hero's own position).
    */
  def observeVillainAction(
      villainPosition: Position,
      action: PokerAction,
      gameState: GameState
  ): Unit =
    _positionMapping.get(villainPosition).foreach { rivalId =>
      engine.observeAction(rivalId, action, gameState)
    }

  def endHand(): Unit =
    engine.endHand()

  /** Extract EVs from upstream ActionRecommendation and run overlay filter. */
  def decideWithOverlay(
      gameState: GameState,
      candidates: Vector[PokerAction],
      upstreamRecommendation: ActionRecommendation,
      upstreamSource: UpstreamSource = UpstreamSource.Adaptive
  ): OverlayResult =
    val upstreamEvs = upstreamRecommendation.actionEvaluations
    val result = engine.decide(gameState, candidates, upstreamEvs)
    // Override upstream source if caller specifies multiway
    if upstreamSource != UpstreamSource.Adaptive then
      result.copy(upstreamSource = upstreamSource)
    else
      result

private[holdem] object StrategicLifecycleHelper:
  def create(config: StrategicEngine.Config = StrategicEngine.Config()): StrategicLifecycleHelper =
    new StrategicLifecycleHelper(new StrategicEngine(config))
