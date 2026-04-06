package sicfun.holdem.engine

import sicfun.holdem.types.*
import sicfun.holdem.strategic.*
import sicfun.holdem.strategic.solver.WPomcpRuntime

/** Session/hand orchestrator for the Strategic decision mode.
  *
  * Manages per-rival beliefs across hands, builds the factored tabular model,
  * and delegates action selection to WPomcpRuntime.solveV2.
  *
  * Lifecycle:
  *   1. Call [[initSession]] once to register rival IDs and seed priors.
  *   2. Call [[startHand]] at the beginning of each hand.
  *   3. Call [[observeAction]] for each rival action observed mid-hand.
  *   4. Call [[decide]] when hero must act.
  *   5. Call [[endHand]] when the hand concludes.
  */
class StrategicEngine(val config: StrategicEngine.Config):

  private var _sessionState: StrategicEngine.SessionState | Null = null
  private var _handActive: Boolean = false

  def sessionState: StrategicEngine.SessionState =
    require(_sessionState != null, "Session not initialized — call initSession first")
    _sessionState.nn

  def currentHandActive: Boolean = _handActive

  /** Initialize session with rival IDs. Uses uniform priors unless existing beliefs provided. */
  def initSession(
      rivalIds: Vector[PlayerId],
      existingBeliefs: Map[PlayerId, StrategicRivalBelief] = Map.empty
  ): Unit =
    val beliefs = rivalIds.map { id =>
      id -> existingBeliefs.getOrElse(id, StrategicRivalBelief.uniform)
    }.toMap
    _sessionState = StrategicEngine.SessionState(rivalBeliefs = beliefs)

  /** Start a new hand. Resets hand-local state, preserves session beliefs. */
  def startHand(): Unit =
    require(_sessionState != null, "Session not initialized — call initSession first")
    _handActive = true

  /** Observe a rival's action.
    *
    * Placeholder — full Dynamics integration in follow-up task.
    * The no-op body is intentional: belief update is deferred.
    */
  def observeAction(actor: PlayerId, action: PokerAction, gameState: GameState): Unit =
    if _sessionState == null then return

  /** Choose an action using the WPomcp V2 solver.
    *
    * Falls back to the last candidate action if the native solver is unavailable.
    */
  def decide(gameState: GameState, candidateActions: Vector[PokerAction]): PokerAction =
    require(_sessionState != null, "Session not initialized")
    require(_handActive, "No hand in progress")
    require(candidateActions.nonEmpty, "No candidate actions")

    val heroBucket = estimateHeroBucket(gameState)

    val searchInput = PokerPomcpFormulation.buildSearchInputV2(
      gameState = gameState,
      rivalBeliefs = _sessionState.nn.rivalBeliefs,
      heroActions = candidateActions,
      heroBucket = heroBucket,
      particlesPerRival = config.particlesPerRival
    )

    WPomcpRuntime.solveV2(searchInput, WPomcpRuntime.Config(
      numSimulations = config.numSimulations,
      discount = config.discount,
      maxDepth = config.maxDepth,
      seed = config.seed
    )) match
      case Right(result) =>
        if result.bestAction >= 0 && result.bestAction < candidateActions.size then
          candidateActions(result.bestAction)
        else
          candidateActions.last
      case Left(_) =>
        candidateActions.find(_ != PokerAction.Fold).getOrElse(PokerAction.Fold)

  /** End the current hand. Preserves session beliefs for carry-over across hands. */
  def endHand(showdownResult: Option[Map[PlayerId, HoleCards]] = None): Unit =
    _handActive = false

  /** Estimate hero hand bucket (0-9 equity decile). Position-based placeholder. */
  private def estimateHeroBucket(gameState: GameState): Int =
    gameState.position match
      case Position.Button | Position.Cutoff                        => 7
      case Position.SmallBlind | Position.BigBlind                  => 4
      case Position.UTG | Position.UTG1 | Position.UTG2             => 4
      case Position.Middle | Position.Hijack                        => 5

object StrategicEngine:

  /** Configuration for a StrategicEngine session. */
  final case class Config(
      numSimulations: Int = 500,
      discount: Double = 0.95,
      maxDepth: Int = 20,
      seed: Long = 42L,
      particlesPerRival: Int = 100
  )

  /** Per-session state: rival beliefs that survive across hands. */
  final case class SessionState(
      rivalBeliefs: Map[PlayerId, StrategicRivalBelief]
  )
