package sicfun.holdem.runtime

import sicfun.holdem.types.*
import sicfun.holdem.engine.StrategicEngine
import sicfun.holdem.strategic.types.{PlayerId, CertificationResult}

/** Adapts AdvisorSession lifecycle commands to StrategicEngine operations.
  *
  * Centralizes the mapping between the interactive session model (HandSnapshot-based)
  * and the StrategicEngine API (GameState/PlayerId-based) so that AdvisorSession
  * stays focused on user interaction.
  */
object StrategicAdvisorBridge:

  private val VillainId = PlayerId("villain")

  /** Called at the start of each new hand. Initializes session if needed, then starts a new hand. */
  def onNewHand(engine: StrategicEngine): Unit =
    if !engine.isSessionInitialized then
      engine.initSession(Vector(VillainId))
    engine.startHand()

  /** Called when a villain action is observed. Feeds the action to the strategic engine. */
  def onVillainAction(engine: StrategicEngine, action: PokerAction, h: HandSnapshot): Unit =
    val gameState = GameState(
      street = h.street, board = h.board, pot = h.pot, toCall = h.toCall,
      position = h.villainPosition, stackSize = h.villainStack, betHistory = h.betHistory
    )
    engine.observeAction(VillainId, action, gameState)

  /** Called during advise to get strategic engine diagnostics.
    * Returns additional output lines to append to the advice.
    */
  @scala.annotation.nowarn("msg=deprecated")
  def onAdvise(
      engine: StrategicEngine,
      gameState: GameState,
      candidates: Vector[PokerAction]
  ): Vector[String] =
    try
      engine.decide(gameState, candidates)
      val out = Vector.newBuilder[String]
      engine.lastDecisionBundle.foreach { bundle =>
        bundle.fourWorld.foreach { fw =>
          out += f"  Strategic: V11=${fw.v11.value}%.3f deltaCtrl=${fw.deltaControl.value}%.3f deltaSig*=${fw.deltaSigStar.value}%.3f"
        }
        bundle.pointwiseExploitability.foreach { eps =>
          out += f"  Strategic: exploitability=${eps.value}%.4f"
        }
        bundle.certification match
          case cert: CertificationResult.TabularCertification =>
            out += f"  Strategic: B*=${cert.requiredBudget}%.3f safe=${cert.safeActionIndices.size} valid=${cert.certificateValid}"
          case _ => ()
      }
      out.result()
    catch
      case _: Exception => Vector.empty

  /** Called on villain showdown. Feeds showdown data to the strategic engine. */
  def onVillainShowdown(engine: StrategicEngine, cards: HoleCards): Unit =
    engine.endHand(Some(Map(VillainId -> cards)))
