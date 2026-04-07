package sicfun.holdem.engine

import sicfun.holdem.types.*
import sicfun.holdem.strategic.*
import sicfun.holdem.strategic.StrategicClass
import sicfun.holdem.strategic.solver.WPomcpRuntime

/** Builds flat array inputs for the WPomcp V2 factored tabular model.
  *
  * Translates poker domain state (GameState, PokerAction, StrategicRivalBelief)
  * into the flat Array[Double]/Array[Int] format consumed by the C++ solver via JNI.
  *
  * Public-state layout: pubState index = street * NumPotBuckets * NumStackBuckets + potBucket * NumStackBuckets + stackBucket
  * Streets: 0=Preflop, 1=Flop, 2=Turn, 3=River  -> 4 * 8 * 6 = 192 total pub states.
  *
  * Per-round abstraction: the C++ simulate_v2 models each MCTS depth step as one
  * full betting round — hero acts, all rivals respond, then the street advances.
  * This means terminal_flags and rival_policy are evaluated per-round, not per
  * individual action within a betting round. The tradeoff is simplicity (one
  * action per player per depth) vs fidelity (intra-round re-raises are collapsed).
  */
object PokerPomcpFormulation:

  private val NumPotBuckets   = 8
  private val NumStackBuckets = 6
  val NumPubStates: Int  = 4 * NumPotBuckets * NumStackBuckets  /* 192 */
  val NumRivalTypes: Int = StrategicClass.values.length          /* 4   */
  val NumHandBuckets: Int = 10
  val DefaultPotBucketSize: Double = 50.0

  /** Build rival policy table: P(action | rivalType, pubState).
    *
    * Indexed as: rivalPolicy(rivalType * numPubStates * numActions + pubState * numActions + action)
    * Uses type-conditioned action priors derived from StrategicClass behavioral profiles.
    * Action 0 = fold, action 1 = check/call, actions 2+ = raises (distributed uniformly).
    *
    * @param numRivalTypes number of discrete rival type categories
    * @param numPubStates  number of discrete public states
    * @param numActions    number of available hero actions
    * @return flat array of length numRivalTypes * numPubStates * numActions
    */
  def buildRivalPolicy(
      numRivalTypes: Int,
      numPubStates: Int,
      numActions: Int
  ): Array[Double] =
    val size = numRivalTypes * numPubStates * numActions
    val result = new Array[Double](size)
    for typeIdx <- 0 until numRivalTypes do
      val priors = classPriors(StrategicClass.fromOrdinal(typeIdx), numActions)
      for pub <- 0 until numPubStates do
        val base = typeIdx * numPubStates * numActions + pub * numActions
        System.arraycopy(priors, 0, result, base, numActions)
    result

  /** Per-class action distribution: (foldP, passiveP, raiseP) per StrategicClass.
    * Exposed as a val so callers can override with calibrated values.
    */
  val defaultClassPriors: Map[StrategicClass, (Double, Double, Double)] = Map(
    StrategicClass.Value     -> (0.05, 0.75, 0.20),
    StrategicClass.Bluff     -> (0.10, 0.25, 0.65),
    StrategicClass.SemiBluff -> (0.05, 0.45, 0.50),
    StrategicClass.Marginal  -> (0.15, 0.75, 0.10)
  )

  /** Per-class action distribution for numActions actions.
    * Returns normalized array of length numActions.
    * Action 0 = fold, action 1 = check/call, actions 2+ = raises.
    */
  private def classPriors(
      cls: StrategicClass,
      numActions: Int,
      priorTable: Map[StrategicClass, (Double, Double, Double)] = defaultClassPriors
  ): Array[Double] =
    val raw = new Array[Double](numActions)
    val (foldP, passiveP, raiseP) = priorTable.getOrElse(cls, (0.25, 0.50, 0.25))
    raw(0) = foldP
    if numActions > 1 then raw(1) = passiveP
    val raiseSlots = math.max(1, numActions - 2)
    for i <- 2 until numActions do raw(i) = raiseP / raiseSlots
    val sum = raw.sum
    if sum > 0 then for i <- raw.indices do raw(i) /= sum
    raw

  /** Build action effects: [numActions * 3] fields per action.
    *
    * Each action contributes three consecutive doubles:
    *   (pot_delta_frac, is_fold, is_allin)
    *
    * where:
    *   - pot_delta_frac = chips added to pot as a fraction of current pot (capped at 10x)
    *   - is_fold        = 1.0 if action is Fold, else 0.0
    *   - is_allin       = 1.0 if raise amount >= remaining stack, else 0.0
    *
    * @param actions    the ordered action vector (fold should be action 0 by convention)
    * @param potChips   current pot size in chips (Double)
    * @param stackChips hero's remaining stack in chips (Double)
    * @return flat array of length actions.size * 3
    */
  def buildActionEffects(
      actions: Vector[PokerAction],
      potChips: Double,
      stackChips: Double
  ): Array[Double] =
    val result = new Array[Double](actions.size * 3)
    var i = 0
    for action <- actions do
      val base = i * 3
      action match
        case PokerAction.Fold =>
          result(base)     = 0.0
          result(base + 1) = 1.0
          result(base + 2) = 0.0
        case PokerAction.Check =>
          result(base)     = 0.0
          result(base + 1) = 0.0
          result(base + 2) = 0.0
        case PokerAction.Call =>
          val frac = if potChips > 0.0 then 0.5 else 0.0
          result(base)     = frac
          result(base + 1) = 0.0
          result(base + 2) = 0.0
        case PokerAction.Raise(amount) =>
          val frac = if potChips > 0.0 then amount / potChips else 1.0
          result(base)     = math.min(frac, 10.0)
          result(base + 1) = 0.0
          result(base + 2) = if amount >= stackChips then 1.0 else 0.0
      i += 1
    result

  /** Default showdown equity table: linear heuristic.
    * Replace with calibrated bucket-vs-bucket equity from HeadsUpEquityTable.
    */
  val defaultShowdownEquity: (Int, Int) => Array[Double] = buildLinearShowdownEquity

  /** Linear equity heuristic: equity = 0.5 + (heroBucket - rivalBucket) * 0.4 / max(H, R).
    * Named explicitly so callers know this is an approximation.
    */
  def buildLinearShowdownEquity(numHeroBuckets: Int, numRivalBuckets: Int): Array[Double] =
    Array.tabulate(numHeroBuckets * numRivalBuckets) { idx =>
      val hb = idx / numRivalBuckets
      val rb = idx % numRivalBuckets
      val diff = (hb - rb).toDouble / math.max(numHeroBuckets, numRivalBuckets).toDouble
      0.5 + diff * 0.4
    }

  /** Build showdown equity table: E[hero equity | heroBucket, rivalBucket].
    *
    * Indexed as: showdownEquity(heroBucket * numRivalBuckets + rivalBucket)
    * Delegates to buildLinearShowdownEquity by default.
    *
    * @param numHeroBuckets  number of hero hand-strength buckets
    * @param numRivalBuckets number of rival hand-strength buckets
    * @return flat array of length numHeroBuckets * numRivalBuckets, all in [0.1, 0.9]
    */
  def buildShowdownEquity(
      numHeroBuckets: Int,
      numRivalBuckets: Int
  ): Array[Double] =
    buildLinearShowdownEquity(numHeroBuckets, numRivalBuckets)

  /** Build terminal flags: outcome code at each (pubState, action) pair.
    *
    * Indexed as: terminalFlags(pubState * numActions + action)
    * Codes:
    *   0 = Continue (non-terminal)
    *   1 = HeroFold  (action index 0 is always fold by convention)
    *   2 = RivalFold (not used in hero-action table; reserved for rival policy)
    *   3 = Showdown  (non-fold action at street=River, i.e. street ordinal >= 3)
    *
    * Public-state layout mirrors buildRivalPolicy:
    *   street = pubState / (NumPotBuckets * NumStackBuckets)   (0-3)
    *
    * @param numPubStates number of discrete public states (must equal NumPubStates = 192)
    * @param numActions   number of actions (fold must be action 0)
    * @return flat array of length numPubStates * numActions, values in {0,1,2,3}
    */
  def buildTerminalFlags(
      numPubStates: Int,
      numActions: Int
  ): Array[Int] =
    val flags = new Array[Int](numPubStates * numActions)
    for pub <- 0 until numPubStates do
      val street = pub / (NumPotBuckets * NumStackBuckets)
      for a <- 0 until numActions do
        val idx = pub * numActions + a
        if a == 0 then
          flags(idx) = 1  // HeroFold — action 0 is fold by convention
        else if street >= 3 then
          flags(idx) = 3  // Showdown — river, non-fold
        else
          flags(idx) = 0  // Continue — non-terminal mid-hand action
    flags

  /** Build complete SearchInputV2 from poker game state and strategic beliefs. */
  def buildSearchInputV2(
      gameState: GameState,
      rivalBeliefs: Map[PlayerId, StrategicRivalBelief],
      heroActions: Vector[PokerAction],
      heroBucket: Int,
      particlesPerRival: Int = 100
  ): WPomcpRuntime.SearchInputV2 =
    val numActions = heroActions.size

    val rivalParticles = rivalBeliefs.values.toIndexedSeq.map { belief =>
      val (types, weights) = belief.toParticles(particlesPerRival, heroBucket)
      /* Distribute rival private states (hand buckets) uniformly across [0, NumHandBuckets).
       * Each particle gets a bucket proportional to its index so that the showdown equity
       * table is exercised across the full rival bucket range, not collapsed to a single value. */
      val privStates = Array.tabulate(particlesPerRival)(i => i % NumHandBuckets)
      WPomcpRuntime.RivalParticles(
        rivalTypes = types,
        privStates = privStates,
        weights = weights
      )
    }

    val model = WPomcpRuntime.FactoredModel(
      rivalPolicy = buildRivalPolicy(NumRivalTypes, NumPubStates, numActions),
      numRivalTypes = NumRivalTypes,
      numPubStates = NumPubStates,
      actionEffects = buildActionEffects(heroActions, gameState.pot, gameState.stackSize),
      showdownEquity = buildShowdownEquity(NumHandBuckets, NumHandBuckets),
      numHeroBuckets = NumHandBuckets,
      numRivalBuckets = NumHandBuckets,
      terminalFlags = buildTerminalFlags(NumPubStates, numActions),
      potBucketSize = DefaultPotBucketSize
    )

    WPomcpRuntime.SearchInputV2(
      publicState = WPomcpRuntime.PublicState(gameState.street.ordinal, gameState.pot.toDouble),
      rivalParticles = rivalParticles,
      model = model,
      heroBucket = heroBucket
    )
