package sicfun.holdem.engine

import sicfun.holdem.types.*
import sicfun.holdem.model.*
import sicfun.holdem.equity.*
import sicfun.holdem.cfr.*

import java.util.Random

/** Shared hero decision pipeline used by match runners (ACPC, Slumbot).
  *
  * Extracted from AcpcMatchRunner.Runner and SlumbotMatchRunner.Runner where
  * decideHero, legalRaiseCandidates, heroCandidates, and newAdaptiveEngine
  * were character-for-character identical.
  */
private[holdem] object HeroDecisionPipeline:

  /** Protocol-neutral context for raise sizing. Both AcpcActionCodec.ParsedActionState and
    * SlumbotActionCodec.ParsedActionState provide these fields.
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
      rng: scala.util.Random
  )

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
            decisionBudgetMillis = Some(1L),
            rng = new Random(ctx.rng.nextLong())
          )
          .decision
          .recommendation
          .bestAction
      case HeroMode.Gto =>
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

  def heroCandidates(toCallChips: Int, raises: Vector[PokerAction]): Vector[PokerAction] =
    if toCallChips <= 0 then Vector(PokerAction.Check) ++ raises
    else Vector(PokerAction.Fold, PokerAction.Call) ++ raises

  private[holdem] def sampleActionByPolicy(
      probabilities: Map[PokerAction, Double],
      candidates: Vector[PokerAction],
      rng: Random
  ): PokerAction =
    val roll = rng.nextDouble()
    var cumulative = 0.0
    var idx = 0
    while idx < candidates.length do
      cumulative += math.max(0.0, probabilities.getOrElse(candidates(idx), 0.0))
      if roll < cumulative then return candidates(idx)
      idx += 1

    candidates
      .maxBy(action => probabilities.getOrElse(action, 0.0))

  def newAdaptiveEngine(
      tableRanges: TableRanges,
      model: PokerActionModel,
      bunchingTrials: Int,
      equityTrials: Int
  ): RealTimeAdaptiveEngine =
    new RealTimeAdaptiveEngine(
      tableRanges = tableRanges,
      actionModel = model,
      bunchingTrials = bunchingTrials,
      defaultEquityTrials = equityTrials,
      minEquityTrials = math.max(8, math.min(equityTrials, equityTrials / 10))
    )
