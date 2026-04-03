package sicfun.holdem;

/** JNI bindings for native CPU Bayesian posterior updates. */
public final class HoldemBayesNativeCpuBindings {
  private HoldemBayesNativeCpuBindings() {}

  /**
   * Applies sequential Bayesian updates for a fixed hypothesis space.
   *
   * <p>Inputs:
   * <ul>
   *   <li>{@code prior.length == hypothesisCount}</li>
   *   <li>{@code likelihoods.length == observationCount * hypothesisCount} (row-major by observation)</li>
   *   <li>{@code outPosterior.length == hypothesisCount}</li>
   *   <li>{@code outLogEvidence.length >= 1}</li>
   * </ul>
   *
   * @return 0 on success, non-zero status code on failure.
   */
  public static native int updatePosterior(
      int observationCount,
      int hypothesisCount,
      double[] prior,
      double[] likelihoods,
      double[] outPosterior,
      double[] outLogEvidence
  );

  /**
   * Applies sequential Bayesian updates with two-layer tempered likelihoods
   * (SICFUN v0.30.2 Def 15A/15B).
   *
   * <p>Tempered likelihood per hypothesis:
   * <ul>
   *   <li>Standard: {@code pow(likelihood, kappaTemp) + deltaFloor * eta[h]}</li>
   *   <li>Legacy:   {@code (1 - deltaFloor) * likelihood + deltaFloor * eta[h]}</li>
   * </ul>
   *
   * <p>Inputs:
   * <ul>
   *   <li>{@code prior.length == hypothesisCount}</li>
   *   <li>{@code likelihoods.length == observationCount * hypothesisCount} (row-major)</li>
   *   <li>{@code eta.length == hypothesisCount} (or null for uniform)</li>
   *   <li>{@code outPosterior.length == hypothesisCount}</li>
   *   <li>{@code outLogEvidence.length >= 1}</li>
   *   <li>{@code kappaTemp} in (0, 1]</li>
   *   <li>{@code deltaFloor} >= 0</li>
   * </ul>
   *
   * @return 0 on success, non-zero status code on failure.
   */
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
   * 0 = unknown, 1 = CPU engine.
   */
  public static native int lastEngineCode();
}
