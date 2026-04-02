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

  /** Same ABI contract as {@link HoldemCfrNativeCpuBindings#solveTreeRoot}. */
  public static native int solveTreeRoot(
      int iterations,
      int averagingDelay,
      boolean cfrPlus,
      boolean linearAveraging,
      int rootNodeId,
      int rootInfoSetIndex,
      int[] nodeTypes,
      int[] nodeStarts,
      int[] nodeCounts,
      int[] nodeInfosets,
      int[] edgeChildIds,
      double[] edgeProbabilities,
      double[] terminalUtilities,
      int[] infosetPlayers,
      int[] infosetActionCounts,
      double[] outRootStrategy
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
   * Batch CFR solve: N trees with shared topology, float arithmetic on GPU.
   *
   * <p>Per-tree data is concatenated: tree0 data, tree1 data, ..., treeN data.
   * Terminal utilities are indexed as [treeIdx * nodeCount + nodeId].
   * Chance weights are indexed as [treeIdx * edgeCount + edgeIdx].
   * Output strategies are indexed as [treeIdx * strategySize + offset].
   *
   * @return 0 on success, negative on CUDA error, positive on validation error
   */
  public static native int solveTreeBatch(
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
      int[] infosetPlayers,
      int[] infosetActionCounts,
      int[] infosetOffsets,
      float[] terminalUtilities,
      float[] chanceWeights,
      float[] outAverageStrategies,
      float[] outExpectedValues,
      int batchSize
  );

  /**
   * Returns the last engine code used by this native library:
   * 0 = unknown, 2 = GPU provider (or CUDA-compiled host fallback).
   */
  public static native int lastEngineCode();
}
