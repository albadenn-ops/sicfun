package sicfun.holdem;

/** JNI bindings for native CPU CFR tree solving. */
public final class HoldemCfrNativeCpuBindings {
  private HoldemCfrNativeCpuBindings() {}

  /**
   * Solves a two-player zero-sum extensive-form game tree using CFR/CFR+.
   *
   * <p>Node encoding:
   * <ul>
   *   <li>0 = terminal</li>
   *   <li>1 = chance</li>
   *   <li>2 = player0 decision</li>
   *   <li>3 = player1 decision</li>
   * </ul>
   *
   * <p>Tree arrays:
   * <ul>
   *   <li>{@code nodeTypes.length = nodeStarts.length = nodeCounts.length = nodeInfosets.length = terminalUtilities.length = N}</li>
   *   <li>For node {@code i}, outgoing edge span is {@code [nodeStarts[i], nodeStarts[i] + nodeCounts[i])}</li>
   *   <li>{@code edgeChildIds.length = edgeProbabilities.length = E}</li>
   * </ul>
   *
   * <p>Infoset arrays:
   * <ul>
   *   <li>{@code infosetPlayers.length = infosetActionCounts.length = M}</li>
   *   <li>{@code outAverageStrategies.length = sum(infosetActionCounts)}</li>
   * </ul>
   *
   * @return 0 on success, non-zero status code on failure.
   */
  public static native int solveTree(
      int iterations,
      int averagingDelay,
      boolean cfrPlus,
      boolean linearAveraging,
      int rootNodeId,
      int[] nodeTypes,
      int[] nodeStarts,
      int[] nodeCounts,
      int[] nodeInfosets,
      int[] edgeChildIds,
      double[] edgeProbabilities,
      double[] terminalUtilities,
      int[] infosetPlayers,
      int[] infosetActionCounts,
      double[] outAverageStrategies,
      double[] outExpectedValue
  );

  /**
   * Fixed-point ABI variant.
   *
   * <p>Chance probabilities and output strategies use `Prob` raw values (`Int32 @ 2^30`).
   * Terminal utilities and expected value use `FixedVal` raw values (`Int32 @ 2^13`).
   */
  public static native int solveTreeFixed(
      int iterations,
      int averagingDelay,
      boolean cfrPlus,
      boolean linearAveraging,
      int rootNodeId,
      int[] nodeTypes,
      int[] nodeStarts,
      int[] nodeCounts,
      int[] nodeInfosets,
      int[] edgeChildIds,
      int[] edgeProbabilitiesRaw,
      int[] terminalUtilitiesRaw,
      int[] infosetPlayers,
      int[] infosetActionCounts,
      int[] outAverageStrategiesRaw,
      int[] outExpectedValueRaw
  );

  /**
   * Returns the last engine code used by this native library:
   * 0 = unknown, 1 = CPU engine.
   */
  public static native int lastEngineCode();
}
