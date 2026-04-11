package sicfun.holdem;

/**
 * JNI bindings for PFT-DPW POMDP solver (native CPU implementation).
 *
 * <p>Compiled into sicfun_pomcp_native.dll alongside the CPU JNI bindings.
 * Implements Definitions 29-32 (value functions) and 54-55 (particle beliefs)
 * from SICFUN v0.30.2.
 *
 * <p>The generative model is specified as flat arrays:
 * <ul>
 *   <li>{@code transitionTable[s * numActions + a]} = next state index</li>
 *   <li>{@code obsLikelihood[(s * numActions + a) * numObs + o]} = P(o | s', a)</li>
 *   <li>{@code rewardTable[s * numActions + a]} = R(s, a)</li>
 * </ul>
 *
 * <p>The root particle belief (Definition 54) is specified as two parallel arrays:
 * {@code particleStates[j]} = state index and {@code particleWeights[j]} = weight.
 * Weights are normalized by the native side before use.
 *
 * <p>Output arrays ({@code outQValues}, {@code outVisitCounts}) must be pre-allocated
 * with length {@code numActions}. Untried actions receive Q=0.0 and visitCount=0.
 */
public final class HoldemPomcpNativeBindings {
  private HoldemPomcpNativeBindings() {}

  /**
   * Run PFT-DPW tree search from a root particle belief.
   *
   * <p>Returns a packed long: {@code (bestAction << 32) | status}.
   * <ul>
   *   <li>{@code status = (int)(packed & 0xFFFFFFFFL)} -- 0 on success.</li>
   *   <li>{@code bestAction = (int)(packed >>> 32)} -- action ID with highest visit count.</li>
   * </ul>
   *
   * @param transitionTable  [numStates * numActions] next-state indices
   * @param obsLikelihood    [numStates * numActions * numObs] observation probabilities
   * @param rewardTable      [numStates * numActions] immediate rewards
   * @param numStates        number of states
   * @param numActions       number of actions
   * @param numObs           number of observations
   * @param particleStates   root belief particle state indices (Definition 54)
   * @param particleWeights  root belief particle weights (Definition 54)
   * @param numSimulations   number of MCTS simulation passes
   * @param gamma            discount factor in (0, 1)
   * @param rMax             maximum single-step reward bound
   * @param ucbC             UCB1 exploration constant
   * @param kAction          DPW action widening coefficient
   * @param alphaAction      DPW action widening exponent
   * @param kObs             DPW observation widening coefficient
   * @param alphaObs         DPW observation widening exponent
   * @param maxDepth         tree depth limit before rollout
   * @param seed             RNG seed for reproducibility
   * @param outQValues       [numActions] output: Q(b,a) per action
   * @param outVisitCounts   [numActions] output: N(b,a) per action
   * @return (bestAction &lt;&lt; 32) | status
   */
  public static native long solvePftDpw(
      int[] transitionTable,
      double[] obsLikelihood,
      double[] rewardTable,
      int numStates,
      int numActions,
      int numObs,
      int[] particleStates,
      double[] particleWeights,
      int numSimulations,
      double gamma,
      double rMax,
      double ucbC,
      double kAction,
      double alphaAction,
      double kObs,
      double alphaObs,
      int maxDepth,
      long seed,
      double[] outQValues,
      int[] outVisitCounts
  );

  /**
   * Returns the engine code from the last successful {@link #solvePftDpw} call.
   * 0 = no computation yet, 3 = PFT-DPW CPU, 4 = W-POMCP CPU.
   */
  public static native int lastEngineCode();

  /**
   * Run W-POMCP multi-agent factored particle filter search (Definition 56).
   *
   * <p>All particle arrays are flat [rival][particle] row-major layout.
   * The rival action probs are flat [rival][action] row-major.
   *
   * @param rivalCount          number of rivals (1..8)
   * @param particlesPerRival   particle count per rival, length = rivalCount
   * @param particleTypes       rival type index per particle, flat
   * @param particlePrivStates  private state index per particle, flat
   * @param particleWeights     importance weights per particle, flat
   * @param pubStreet           public state street (0=preflop..3=river)
   * @param pubPot              current pot size
   * @param numHeroActions      number of hero actions at this decision point
   * @param rivalActionProbs    per-rival action probs, flat [rival][action]
   * @param rewards             reward per hero action, length = numHeroActions
   * @param numSimulations      number of MCTS simulations
   * @param discount            gamma: discount factor in (0, 1)
   * @param exploration         UCB1 exploration constant
   * @param rMax                maximum absolute reward bound
   * @param maxDepth            tree depth limit
   * @param essThreshold        ESS ratio for resampling trigger
   * @param seed                RNG seed for reproducibility
   * @param outActionValues     [numHeroActions] output: Q(root, a) per action
   * @param outBestAction       [1] output: best action index
   * @param outRootValue        [1] output: root value estimate
   * @return 0 on success, non-zero error code on failure
   */
  public static native int solveWPomcp(
      int rivalCount,
      int[] particlesPerRival,
      int[] particleTypes,
      int[] particlePrivStates,
      double[] particleWeights,
      int pubStreet,
      double pubPot,
      int numHeroActions,
      double[] rivalActionProbs,
      double[] rewards,
      int numSimulations,
      double discount,
      double exploration,
      double rMax,
      int maxDepth,
      double essThreshold,
      long seed,
      double[] outActionValues,
      int[] outBestAction,
      double[] outRootValue
  );

  /**
   * V2 W-POMCP with factored tabular model (type-conditioned policies,
   * real observation particle reweighting, tabular rewards).
   *
   * @param rivalCount          number of rivals (1..8)
   * @param particlesPerRival   particle count per rival, length = rivalCount
   * @param particleTypes       rival type index per particle, flat
   * @param particlePrivStates  private state index per particle, flat
   * @param particleWeights     importance weights per particle, flat
   * @param pubStreet           public state street (0=preflop..3=river)
   * @param pubPot              current pot size
   * @param numHeroActions      number of hero actions at this decision point
   * @param numRivalTypes       number of discrete rival behavioral types
   * @param numPubStates        number of public states in the tabular model
   * @param rivalPolicy         type-conditioned rival policy, flat [numRivalTypes * numPubStates * numHeroActions]
   * @param actionEffects       action effect triples, flat [numHeroActions * 3] (potDeltaFrac, isFold, isAllin)
   * @param showdownEquity      hero vs rival showdown equity, flat [numHeroBuckets * numRivalBuckets]
   * @param numHeroBuckets      hero bucket count for equity indexing
   * @param numRivalBuckets     rival bucket count for equity indexing
   * @param terminalFlags       terminal state flags, flat [numPubStates * numHeroActions]
   * @param heroBucket          hero's current private state bucket index
   * @param potBucketSize       pot discretization granularity for public state encoding
   * @param numSimulations      number of MCTS simulations
   * @param discount            gamma: discount factor in (0, 1)
   * @param exploration         UCB1 exploration constant
   * @param rMax                maximum absolute reward bound
   * @param maxDepth            tree depth limit
   * @param essThreshold        ESS ratio for resampling trigger
   * @param seed                RNG seed for reproducibility
   * @param outActionValues     [numHeroActions] output: Q(root, a) per action
   * @param outBestAction       [1] output: best action index
   * @param outRootValue        [1] output: root value estimate
   * @return 0 on success, non-zero error code on failure
   */
  public static native int solveWPomcpV2(
      int rivalCount,
      int[] particlesPerRival,
      int[] particleTypes,
      int[] particlePrivStates,
      double[] particleWeights,
      int pubStreet,
      double pubPot,
      int numHeroActions,
      int numRivalTypes,
      int numPubStates,
      double[] rivalPolicy,
      double[] actionEffects,
      double[] showdownEquity,
      int numHeroBuckets,
      int numRivalBuckets,
      int[] terminalFlags,
      int heroBucket,
      double potBucketSize,
      int numSimulations,
      double discount,
      double exploration,
      double rMax,
      int maxDepth,
      double essThreshold,
      long seed,
      double[] outActionValues,
      int[] outBestAction,
      double[] outRootValue
  );

  /**
   * Run C++ self-test for W-POMCP. Returns 0 on success.
   * Only meaningful when the DLL is compiled with WPOMCP_SELF_TEST.
   */
  public static native int selfTestWPomcp();
}
