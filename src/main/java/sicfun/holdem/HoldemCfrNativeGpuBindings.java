package sicfun.holdem;

/** JNI bindings for the CUDA-compiled CFR provider bridge. */
public final class HoldemCfrNativeGpuBindings {
  private HoldemCfrNativeGpuBindings() {}

  /** Same ABI contract as {@link HoldemCfrNativeCpuBindings#solveTree}. */
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
   * 0 = unknown, 2 = GPU provider (or CUDA-compiled host fallback).
   */
  public static native int lastEngineCode();
}
