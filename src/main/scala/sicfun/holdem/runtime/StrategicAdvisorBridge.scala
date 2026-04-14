package sicfun.holdem.runtime

import sicfun.holdem.types.*
import sicfun.holdem.engine.StrategicEngine
import sicfun.holdem.engine.inference.{ActionRecommendation, ActionEvaluation}
import sicfun.holdem.strategic.types.PlayerId

/** Adapts AdvisorSession lifecycle commands to StrategicEngine operations.
  *
  * Centralizes the mapping between the interactive session model (HandSnapshot-based)
  * and the StrategicEngine API (GameState/PlayerId-based) so that AdvisorSession
  * stays focused on user interaction.
  *
  * The bridge works directly with StrategicEngine (not StrategicLifecycleHelper) because
  * the advisor uses PlayerId("villain") for all rivals and doesn't need position mapping.
  */
object StrategicAdvisorBridge:

  private val VillainId = PlayerId("villain")

  /** Called at the start of each new hand. Initializes session if needed, then starts a new hand. */
  def onNewHand(engine: StrategicEngine): Unit =
    if !engine.isSessionInitialized then
      engine.initSession(rivalIds = Vector(VillainId))
    engine.startHand()

  /** Called when a villain action is observed. Feeds the action to the strategic engine. */
  def onVillainAction(engine: StrategicEngine, action: PokerAction, h: HandSnapshot): Unit =
    val gameState = GameState(
      street = h.street, board = h.board, pot = h.pot, toCall = h.toCall,
      position = h.villainPosition, stackSize = h.villainStack, betHistory = h.betHistory
    )
    engine.observeAction(VillainId, action, gameState)

  /** Called during advise to get strategic overlay diagnostics.
    *
    * Accepts the upstream ActionRecommendation that AdvisorSession already computed
    * via its adaptive engine (lines 603-614 of AdvisorSession.scala). When provided,
    * the overlay filters real EVs. When None (legacy callers), falls back to zero EVs.
    */
  def onAdvise(
      engine: StrategicEngine,
      gameState: GameState,
      candidates: Vector[PokerAction],
      upstreamRecommendation: Option[ActionRecommendation] = None
  ): Vector[String] =
    try
      val upstreamEvs: Vector[ActionEvaluation] = upstreamRecommendation match
        case Some(rec) => rec.actionEvaluations
        case None      => candidates.map(a => ActionEvaluation(a, 0.0))

      val result = engine.decide(gameState, candidates, upstreamEvs)
      val out = Vector.newBuilder[String]

      // Print overlay diagnostics
      out += f"  Overlay: selected=${result.selectedAction} upstream=${result.upstreamAction} source=${result.upstreamSource}"
      if result.adjustments.nonEmpty then
        result.adjustments.foreach { adj =>
          out += f"  Overlay: ${adj.action} EV ${adj.originalEv}%.3f -> ${adj.adjustedEv}%.3f (${adj.reason})"
        }
      if result.softVetoed.nonEmpty then
        result.softVetoed.foreach { (action, reason) =>
          out += f"  Overlay: SOFT VETO ${action} -- $reason"
        }

      out.result()
    catch
      case _: Exception => Vector.empty

  /** Called on villain showdown. Feeds showdown data to the strategic engine. */
  def onVillainShowdown(engine: StrategicEngine, cards: HoleCards): Unit =
    engine.endHand(Some(Map(VillainId -> cards)))
