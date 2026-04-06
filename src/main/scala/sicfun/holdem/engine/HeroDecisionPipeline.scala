package sicfun.holdem.engine

import sicfun.holdem.types.*
import sicfun.holdem.model.*
import sicfun.holdem.equity.*
import sicfun.holdem.cfr.*

import java.util.Random

/** Shared hero decision pipeline used by match runners (ACPC, Slumbot).
  *
  * This object encapsulates the complete hero decision flow:
  *   1. '''Raise sizing''' ([[legalRaiseCandidates]]): Computes legal raise sizes in chips,
  *      then converts to big-blind units. Uses protocol-aligned defaults (150/200/300 chips
  *      for preflop) and pot-fraction sizing (50%/75%) for postflop.
  *   2. '''Candidate generation''' ([[heroCandidates]]): Assembles the full set of legal
  *      actions (Check/Fold + Call + raises) based on whether the hero faces a bet.
  *   3. '''Decision dispatch''' ([[decideHero]]): Routes to either Adaptive mode (real-time
  *      engine with 1ms latency budget) or GTO mode (full Bayesian inference + CFR solve).
  *   4. '''Engine construction''' ([[newAdaptiveEngine]]): Factory for creating a
  *      RealTimeAdaptiveEngine with the correct trial configuration.
  *
  * Extracted from AcpcMatchRunner.Runner and SlumbotMatchRunner.Runner where
  * decideHero, legalRaiseCandidates, heroCandidates, and newAdaptiveEngine
  * were character-for-character identical.
  *
  * Design note: All chip amounts in [[RaiseSizingContext]] are absolute chip counts from
  * the protocol. The pipeline converts to big-blind units at the final step when creating
  * [[PokerAction.Raise]] values, since the runtime action model works in BB.
  */
private[holdem] object HeroDecisionPipeline:

  /** Protocol-neutral context for raise sizing.
    *
    * Unit contract:
    * - All `*Chips` fields are absolute chip counts from the protocol state.
    * - The resulting [[PokerAction.Raise]] value is converted to big blinds (`amountBb`)
    *   because runtime actions are represented in BB.
    */
  final case class RaiseSizingContext(
      stackRemainingChips: Int,
      toCallChips: Int,
      lastBetSizeChips: Int,
      potChips: Int,
      currentStreet: Street,
      streetLastBetToChips: Int,
      bigBlindChips: Int
  )

  /** Context for a hero decision. Bundles all parameters needed by both Adaptive and GTO modes. */
  final case class HeroDecisionContext(
      hero: HoleCards,
      state: GameState,
      folds: Vector[PreflopFold],
      tableRanges: TableRanges,
      villainPos: Position,
      observations: Vector[VillainObservation],
      candidates: Vector[PokerAction],
      engine: RealTimeAdaptiveEngine,
      actionModel: PokerActionModel,
      bunchingTrials: Int,
      cfrIterations: Int,
      cfrVillainHands: Int,
      cfrEquityTrials: Int,
      rng: scala.util.Random,
      decisionBudgetMillis: Option[Long] = Some(1L)
  )

  /** Main hero decision dispatch. Routes to the appropriate decision mode.
    *
    * - '''Adaptive''': Forces a 1ms decision budget for bounded latency. Delegates to the
    *   real-time adaptive engine which uses cached posteriors and archetype-based response
    *   modeling. Extracts the best action from the recommendation.
    * - '''GTO''': Runs full Bayesian posterior inference (bunching + action model) followed
    *   by a shallow CFR solve. No latency budget is applied, favoring strategy quality
    *   over response time. Suitable for offline analysis and diagnostics.
    *
    * @param mode either HeroMode.Adaptive or HeroMode.Gto
    * @param ctx the bundled decision context containing all required parameters
    * @return the chosen PokerAction
    */
  def decideHero(mode: HeroMode, ctx: HeroDecisionContext): PokerAction =
    mode match
      case HeroMode.Adaptive =>
        ctx.engine
          .decide(
            hero = ctx.hero,
            state = ctx.state,
            folds = ctx.folds,
            villainPos = ctx.villainPos,
            observations = ctx.observations,
            candidateActions = ctx.candidates,
            decisionBudgetMillis = ctx.decisionBudgetMillis,
            rng = new Random(ctx.rng.nextLong())
          )
          .decision
          .recommendation
          .bestAction
      case HeroMode.Gto =>
        // Diagnostic/offline mode: infer posterior then run shallow CFR without a hard
        // latency budget, favoring strategy quality over service-time guarantees.
        val gtoRng = new Random(ctx.rng.nextLong())
        val posterior = RangeInferenceEngine
          .inferPosterior(
            hero = ctx.hero,
            board = ctx.state.board,
            folds = ctx.folds,
            tableRanges = ctx.tableRanges,
            villainPos = ctx.villainPos,
            observations = ctx.observations,
            actionModel = ctx.actionModel,
            bunchingTrials = ctx.bunchingTrials,
            rng = gtoRng
          )
          .posterior
        val policy = HoldemCfrSolver
          .solveShallowDecisionPolicy(
            hero = ctx.hero,
            state = ctx.state,
            villainPosterior = posterior,
            candidateActions = ctx.candidates,
            config = HoldemCfrConfig(
              iterations = ctx.cfrIterations,
              maxVillainHands = ctx.cfrVillainHands,
              equityTrials = ctx.cfrEquityTrials,
              postflopLookahead = ctx.state.street != Street.Preflop && ctx.state.street != Street.River,
              rngSeed = ctx.rng.nextLong()
            )
          )
        sampleActionByPolicy(
          probabilities = policy.actionProbabilities,
          candidates = ctx.candidates,
          rng = gtoRng
        )
      case HeroMode.Strategic =>
        // TODO(Task 8): wire StrategicEngine here; delegate to Adaptive in the interim.
        ctx.engine
          .decide(
            hero = ctx.hero,
            state = ctx.state,
            folds = ctx.folds,
            villainPos = ctx.villainPos,
            observations = ctx.observations,
            candidateActions = ctx.candidates,
            decisionBudgetMillis = ctx.decisionBudgetMillis,
            rng = new Random(ctx.rng.nextLong())
          )
          .decision
          .recommendation
          .bestAction

  /** Computes legal raise sizes based on the protocol game state.
    *
    * Returns a vector of [[PokerAction.Raise]] values in big-blind units. The sizing
    * logic is context-dependent:
    *   - '''Preflop facing open (streetLastBetTo == BB)''': 150 and 200 chip increments
    *     (representing 1.5x and 2x pot raises in standard protocols).
    *   - '''Preflop unraised''': 200 and 300 chip increments (open-raise sizing).
    *   - '''Postflop check-to-act''': 50% and 75% pot sizes.
    *   - '''Postflop facing bet''': minimum legal raise and 75% pot size.
    *
    * All raw increments are clamped to [minIncrement, maxIncrement] and deduplicated.
    * Returns empty vector if the hero has no room to raise (stack <= toCall).
    *
    * @param ctx the raise sizing context with absolute chip amounts
    * @return legal raise actions converted to big-blind units, sorted ascending
    */
  def legalRaiseCandidates(ctx: RaiseSizingContext): Vector[PokerAction] =
    val remaining = ctx.stackRemainingChips
    val toCall = ctx.toCallChips
    val maxIncrement = remaining - toCall
    if maxIncrement <= 0 then Vector.empty
    else
      val minIncrement =
        math.min(
          maxIncrement,
          if ctx.lastBetSizeChips > 0 then math.max(ctx.bigBlindChips, ctx.lastBetSizeChips)
          else ctx.bigBlindChips
        )
      val rawIncrements =
        // Preflop opener/re-open heuristics are encoded in chips and later converted to BB.
        // 150/200/300 are protocol-aligned defaults observed in the match runners.
        if ctx.currentStreet == Street.Preflop && ctx.toCallChips > 0 && ctx.streetLastBetToChips == ctx.bigBlindChips then
          Vector(150, 200)
        else if ctx.currentStreet == Street.Preflop && ctx.toCallChips == 0 && ctx.streetLastBetToChips == ctx.bigBlindChips then
          Vector(200, 300)
        else if ctx.toCallChips <= 0 then
          Vector(
            PokerFormatting.roundedChips(ctx.potChips * 0.50),
            PokerFormatting.roundedChips(ctx.potChips * 0.75)
          )
        else
          Vector(
            minIncrement,
            PokerFormatting.roundedChips(ctx.potChips * 0.75)
          )
      rawIncrements
        .map(value => math.max(minIncrement, math.min(maxIncrement, value)))
        .distinct
        .sorted
        .map(value => PokerAction.Raise(value.toDouble / ctx.bigBlindChips.toDouble))
        .toVector

  /** Assembles the full set of hero candidate actions.
    *
    * - If no call is needed (toCallChips <= 0): Check + any raises.
    * - If facing a bet: Fold + Call + any raises.
    *
    * @param toCallChips the amount the hero must call (0 = check-to-act)
    * @param raises precomputed raise candidates from [[legalRaiseCandidates]]
    * @return ordered candidate actions
    */
  def heroCandidates(toCallChips: Int, raises: Vector[PokerAction]): Vector[PokerAction] =
    if toCallChips <= 0 then Vector(PokerAction.Check) ++ raises
    else Vector(PokerAction.Fold, PokerAction.Call) ++ raises

  /** Samples an action from a CFR mixed-strategy policy using the inverse CDF method.
    *
    * Walks through candidates accumulating probability mass. If the random roll falls
    * within a candidate's cumulative range, that action is selected. Falls back to the
    * highest-probability candidate if the roll exceeds all cumulative mass (can happen
    * when probabilities sum to < 1 due to rounding).
    *
    * @param probabilities action -> probability map from CFR solution
    * @param candidates ordered candidate actions
    * @param rng source of randomness
    * @return the sampled action
    */
  private[holdem] def sampleActionByPolicy(
      probabilities: Map[PokerAction, Double],
      candidates: Vector[PokerAction],
      rng: Random
  ): PokerAction =
    // Uses residual mass implicitly: if the cumulative probability never reaches the roll
    // (for example because probabilities sum to < 1), fallback returns the max-probability action.
    val roll = rng.nextDouble()
    var cumulative = 0.0
    var idx = 0
    while idx < candidates.length do
      cumulative += math.max(0.0, probabilities.getOrElse(candidates(idx), 0.0))
      if roll < cumulative then return candidates(idx)
      idx += 1

    candidates
      .maxBy(action => probabilities.getOrElse(action, 0.0))

  /** Factory for creating a RealTimeAdaptiveEngine with standard configuration.
    *
    * The minEquityTrials is set to max(8, min(equityTrials, equityTrials/10)), providing
    * a lower bound that scales with the default budget but never goes below 8. This ensures
    * the engine can still produce reasonable equity estimates under tight latency budgets.
    */
  def newAdaptiveEngine(
      tableRanges: TableRanges,
      model: PokerActionModel,
      bunchingTrials: Int,
      equityTrials: Int,
      equilibriumBaselineConfig: Option[EquilibriumBaselineConfig] = None
  ): RealTimeAdaptiveEngine =
    new RealTimeAdaptiveEngine(
      tableRanges = tableRanges,
      actionModel = model,
      bunchingTrials = bunchingTrials,
      defaultEquityTrials = equityTrials,
      minEquityTrials = math.max(8, math.min(equityTrials, equityTrials / 10)),
      equilibriumBaselineConfig = equilibriumBaselineConfig
    )
