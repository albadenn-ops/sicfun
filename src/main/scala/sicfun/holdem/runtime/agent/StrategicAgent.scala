package sicfun.holdem.runtime.agent

import sicfun.holdem.runtime.protocol.*
import sicfun.holdem.types.{PokerAction, Board, Position, Street}
import sicfun.holdem.strategic.safety.{SafetyBellman, DetectionPredicate}
import sicfun.holdem.strategic.exploitation.{
  ExploitationInterpolation,
  ExploitationState,
  ExploitationConfig
}
import sicfun.holdem.strategic.state.{
  PublicState,
  PublicAction,
  ActionSignal,
  Sizing
}
import sicfun.holdem.strategic.types.{
  Chips,
  PotFraction,
  PlayerId,
  TableMap,
  Seat,
  SeatStatus
}

/** Explicit A.1 defaults for [[ExploitationConfig]].
  *
  * The real API has no `.default` factory, so we pin the values here:
  *   - `initialBeta = 0.0`  starts in pure-reference mode (no attribution drift),
  *   - `cpRetreatRate = 0.1` retreats beta by 0.1 on each detection event,
  *   - `epsilonAdapt = 0.01` allows a 0.01 budget above epsilon_NE for adaptation.
  */
object ExploitationDefaults:
  val a1Config: ExploitationConfig = ExploitationConfig(
    initialBeta = 0.0,
    cpRetreatRate = 0.1,
    epsilonAdapt = 0.01
  )

/** Strategic agent wiring the SafetyBellman pipeline + ExploitationInterpolation.
  *
  * A.1 scope: this composes the verified safety & exploitation primitives with
  * the placeholder MdpEmbedding / ExploitabilityOracle / InfostateHasher. The
  * placeholders are surfaced via [[PlaceholderMarker.scanPlaceholders]] so callers
  * can audit which components are stubs before relying on benchmark numbers.
  */
final class StrategicAgent(
    override val seatId: SeatId,
    val store: BlueprintStore,
    val hasher: InfostateHasher,
    val embedding: MdpEmbedding,
    val oracle: ExploitabilityOracle,
    val detector: DetectionPredicate,
    rngSeed: Long,
    abstractActions: Vector[PokerAction],
    config: ExploitationConfig = ExploitationDefaults.a1Config
) extends SeatAgent:
  require(abstractActions.nonEmpty, "abstractActions must be non-empty")

  private var exploitationState: ExploitationState =
    ExploitationState.initial(config)
  private val publicActions: collection.mutable.ArrayBuffer[PublicAction] =
    collection.mutable.ArrayBuffer.empty

  /** Test hook: current beta (no other observers). */
  def currentBetaForTest: Double = exploitationState.beta

  override def decide(
      snapshot: TableSnapshot,
      legalActions: Set[PokerAction]
  ): PokerAction =
    val built = embedding.build(snapshot, abstractActions, numProfiles = 3)
    val gamma = 0.95
    val bStar = SafetyBellman.computeBStar(
      robustLosses = built.robustLosses,
      gamma = gamma,
      transitions = built.transitions,
      numProfiles = built.numProfiles,
      terminalStates = built.terminalStates
    )
    val safeActions = SafetyBellman.safeActionSet(
      stateIndex = 0,
      bound = bStar,
      robustLosses = built.robustLosses,
      gamma = gamma,
      transitions = built.transitions,
      numProfiles = built.numProfiles
    )
    val chosenIdx = SafetyBellman.safeFeasibleAction(built.qValues, safeActions)
    val chosen =
      if chosenIdx >= 0 && chosenIdx < abstractActions.size then
        abstractActions(chosenIdx)
      else abstractActions.head
    translateToLegal(chosen, legalActions)

  override def onHandStart(snapshot: TableSnapshot): Unit =
    publicActions.clear()
    ingestHistoryIntoPublicActions(snapshot.actionHistory, snapshot.street)

  override def onHandEnd(snapshot: TableSnapshot, outcome: HandOutcome): Unit =
    ingestHistoryIntoPublicActions(outcome.events, snapshot.street)
    val rivals = snapshot.activeSeats.filter(_ != seatId)
    val publicState = buildPublicState(snapshot)
    val historySnapshot = publicActions.toVector
    rivals.foreach { rival =>
      val rivalId = PlayerId(s"seat_${rival.index}")
      exploitationState = ExploitationInterpolation.updateExploitation(
        state = exploitationState,
        config = config,
        rivalId = rivalId,
        history = historySnapshot,
        publicState = publicState,
        detector = detector,
        exploitabilityFn = oracle.exploitabilityFn(0.0),
        epsilonNE = 0.0
      )
    }

  private def ingestHistoryIntoPublicActions(
      events: Vector[BettingRoundEvent],
      currentStreet: Street
  ): Unit =
    events.foreach {
      case BettingRoundEvent.Act(seat, action) =>
        val signal = buildSignal(action, currentStreet)
        publicActions += PublicAction(
          actor = PlayerId(s"seat_${seat.index}"),
          signal = signal
        )
      case _ => ()
    }

  /** Build an [[ActionSignal]] from a [[PokerAction]].
    *
    * Per Signal.scala: `sizing` is `Option[Sizing]` and is `None` for non-sized
    * actions (Fold, Check, Call). Only `Raise(amount)` carries a size. We
    * approximate `fractionOfPot` as 1.0 since we lack the pot context here;
    * downstream consumers that need true fraction-of-pot should join with
    * [[PublicState.pot]]. `timing` is `None` (no timing data from this source).
    */
  private def buildSignal(
      action: PokerAction,
      stage: Street
  ): ActionSignal =
    val sizing: Option[Sizing] = action match
      case PokerAction.Raise(amount) =>
        Some(Sizing(Chips(amount), PotFraction(1.0)))
      case _ => None
    ActionSignal(
      action = action.category,
      sizing = sizing,
      timing = None,
      stage = stage
    )

  private def buildPublicState(snapshot: TableSnapshot): PublicState =
    PublicState(
      street = snapshot.street,
      board = Board(snapshot.board),
      pot = Chips(snapshot.contributions.values.sum.toDouble),
      stacks = buildStacksMap(snapshot),
      actionHistory = publicActions.toVector
    )

  private def buildStacksMap(snapshot: TableSnapshot): TableMap[Chips] =
    val heroId = PlayerId(s"seat_${seatId.index}")
    val seats = (0 until snapshot.config.numSeats).map { i =>
      val s = SeatId(i)
      val pid = PlayerId(s"seat_$i")
      val stack = snapshot.stacks.getOrElse(s, 0L)
      val status =
        if !snapshot.activeSeats.contains(s) then SeatStatus.Folded
        else if stack <= 0L then SeatStatus.AllIn
        else SeatStatus.Active
      Seat(pid, positionFor(i, snapshot.config.numSeats), status, Chips(stack.toDouble))
    }.toVector
    TableMap(hero = heroId, seats = seats)

  private def positionFor(seatIdx: Int, numSeats: Int): Position =
    val idx = math.min(seatIdx, Position.values.length - 1)
    Position.values(idx)

  private def translateToLegal(
      a: PokerAction,
      legal: Set[PokerAction]
  ): PokerAction =
    if legal.contains(a) then a
    else
      a match
        case _: PokerAction.Raise =>
          legal
            .collect { case r: PokerAction.Raise => r }
            .headOption
            .orElse(legal.collectFirst { case PokerAction.Call => PokerAction.Call })
            .orElse(legal.collectFirst { case PokerAction.Check => PokerAction.Check })
            .getOrElse(PokerAction.Fold)
        case PokerAction.Check =>
          if legal.contains(PokerAction.Call) then PokerAction.Call
          else PokerAction.Fold
        case PokerAction.Call =>
          if legal.contains(PokerAction.Check) then PokerAction.Check
          else PokerAction.Fold
        case PokerAction.Fold =>
          if legal.contains(PokerAction.Fold) then PokerAction.Fold
          else if legal.contains(PokerAction.Check) then PokerAction.Check
          else if legal.contains(PokerAction.Call) then PokerAction.Call
          else legal.head
