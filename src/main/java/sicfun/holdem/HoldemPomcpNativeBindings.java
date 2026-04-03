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
   * 0 = no computation yet, 3 = PFT-DPW CPU.
   */
  public static native int lastEngineCode();
}
