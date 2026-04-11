package sicfun.holdem;

/** JNI bindings for native CUDA-side Bayesian posterior updates provider. */
public final class HoldemBayesNativeGpuBindings {
  private HoldemBayesNativeGpuBindings() {}

  /** Same ABI contract as {@link HoldemBayesNativeCpuBindings#updatePosterior}. */
  public static native int updatePosterior(
      int observationCount,
      int hypothesisCount,
      double[] prior,
      double[] likelihoods,
      double[] outPosterior,
      double[] outLogEvidence
  );

  /** Same ABI contract as {@link HoldemBayesNativeCpuBindings#updatePosteriorTempered}. */
  public static native int updatePosteriorTempered(
      int observationCount,
      int hypothesisCount,
      double[] prior,
      double[] likelihoods,
      double kappaTemp,
      double deltaFloor,
      double[] eta,
      boolean useLegacyForm,
      double[] outPosterior,
      double[] outLogEvidence
  );

  /**
   * Returns the last engine code used by this native library:
   * 0 = unknown, 2 = GPU provider (or CUDA-compiled host fallback).
   */
  public static native int lastEngineCode();
}
