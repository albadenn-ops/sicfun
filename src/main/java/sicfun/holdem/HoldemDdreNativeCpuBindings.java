package sicfun.holdem;

/** JNI bindings for native CPU DDRE posterior inference. */
public final class HoldemDdreNativeCpuBindings {
  private HoldemDdreNativeCpuBindings() {}

  /**
   * Runs DDRE synthetic inference over a fixed hypothesis space.
   *
   * <p>Inputs:
   * <ul>
   *   <li>{@code prior.length == hypothesisCount}</li>
   *   <li>{@code likelihoods.length == observationCount * hypothesisCount} (row-major by observation)</li>
   *   <li>{@code outPosterior.length == hypothesisCount}</li>
   * </ul>
   *
   * @return 0 on success, non-zero status code on failure.
   */
  public static native int inferPosterior(
      int observationCount,
      int hypothesisCount,
      double[] prior,
      double[] likelihoods,
      double[] outPosterior
  );

  /**
   * Returns the last engine code used by this native library:
   * 0 = unknown, 1 = CPU engine.
   */
  public static native int lastEngineCode();
}
