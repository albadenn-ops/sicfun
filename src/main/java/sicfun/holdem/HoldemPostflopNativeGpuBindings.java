package sicfun.holdem;

/** JNI bindings for postflop native CUDA/CPU dual-engine library. */
public final class HoldemPostflopNativeGpuBindings {
  private HoldemPostflopNativeGpuBindings() {}

  public static native int computePostflopBatchMonteCarlo(
      int heroFirst,
      int heroSecond,
      int[] boardCards,
      int boardSize,
      int[] villainFirstCards,
      int[] villainSecondCards,
      int trials,
      long[] seeds,
      double[] wins,
      double[] ties,
      double[] losses,
      double[] stderrs
  );

  /** 0 = unknown, 1 = CPU, 2 = CUDA, 3 = CPU fallback after CUDA failure. */
  public static native int queryNativeEngine();

  /** Returns detected CUDA device count (0 when CUDA runtime/device unavailable). */
  public static native int cudaDeviceCount();
}
