package sicfun.holdem.engine

import sicfun.holdem.types.*
import sicfun.holdem.strategic.*
import sicfun.holdem.strategic.solver.WPomcpRuntime

/** Builds flat array inputs for the WPomcp V2 factored tabular model.
  *
  * Translates poker domain state (GameState, PokerAction, StrategicRivalBelief)
  * into the flat Array[Double]/Array[Int] format consumed by the C++ solver via JNI.
  *
  * Public-state layout: pubState index = street * NumPotBuckets * NumStackBuckets + potBucket * NumStackBuckets + stackBucket
  * Streets: 0=Preflop, 1=Flop, 2=Turn, 3=River  -> 4 * 8 * 6 = 192 total pub states.
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
    * Initial value: uniform distribution over actions for all (type, pubState) pairs.
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
    val uniformProb = 1.0 / numActions.toDouble
    Array.fill(size)(uniformProb)

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

  /** Build showdown equity table: E[hero equity | heroBucket, rivalBucket].
    *
    * Indexed as: showdownEquity(heroBucket * numRivalBuckets + rivalBucket)
    * Initial heuristic: linear equity based on bucket rank comparison,
    * centered at 0.5 with +/-0.4 range across the bucket space.
    *
    * @param numHeroBuckets  number of hero hand-strength buckets
    * @param numRivalBuckets number of rival hand-strength buckets
    * @return flat array of length numHeroBuckets * numRivalBuckets, all in [0.1, 0.9]
    */
  def buildShowdownEquity(
      numHeroBuckets: Int,
      numRivalBuckets: Int
  ): Array[Double] =
    Array.tabulate(numHeroBuckets * numRivalBuckets) { idx =>
      val hb   = idx / numRivalBuckets
      val rb   = idx % numRivalBuckets
      val diff = (hb - rb).toDouble / math.max(numHeroBuckets, numRivalBuckets).toDouble
      0.5 + diff * 0.4
    }

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
      WPomcpRuntime.RivalParticles(
        rivalTypes = types,
        privStates = Array.fill(particlesPerRival)(heroBucket),
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
