package sicfun.holdem;

/** JNI bindings for native CUDA-side DDRE posterior inference provider. */
public final class HoldemDdreNativeGpuBindings {
  private HoldemDdreNativeGpuBindings() {}

  /** Same ABI contract as {@link HoldemDdreNativeCpuBindings#inferPosterior}. */
  public static native int inferPosterior(
      int observationCount,
      int hypothesisCount,
      double[] prior,
      double[] likelihoods,
      double[] outPosterior
  );

  /**
   * Returns the last engine code used by this native library:
   * 0 = unknown, 2 = GPU provider (or CUDA-compiled host fallback).
   */
  public static native int lastEngineCode();
}
