package sicfun.holdem;

/** JNI bindings for native CUDA-side CFR tree solving provider. */
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
   * Returns the last engine code used by this native library:
   * 0 = unknown, 2 = GPU provider (or CUDA-compiled host fallback).
   */
  public static native int lastEngineCode();
}
