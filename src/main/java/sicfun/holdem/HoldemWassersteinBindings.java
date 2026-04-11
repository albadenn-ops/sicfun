package sicfun.holdem;

/**
 * JNI bindings for Wasserstein-1 (earth mover's) distance computation.
 *
 * <p>The native implementation uses a vendored network simplex solver
 * (nbonneel/network_simplex, MIT license). When the native DLL
 * ({@code sicfun_wasserstein_native}) is absent, callers should detect this
 * via {@link sicfun.holdem.strategic.solver.WassersteinDroRuntime#isAvailable()}
 * and fall back to the pure-Scala EMD implementation.
 *
 * <p>Compiled into: {@code sicfun_wasserstein_native.dll}
 *
 * <p>Error status codes returned by all native methods:
 * <ul>
 *   <li>0   -- success</li>
 *   <li>100 -- null pointer argument</li>
 *   <li>101 -- dimension mismatch (n &lt; 1 or m &lt; 1)</li>
 *   <li>102 -- weight normalization failure (sum deviates from 1.0 by more than 1e-6)</li>
 *   <li>103 -- negative weight</li>
 *   <li>104 -- negative cost entry</li>
 * </ul>
 */
public final class HoldemWassersteinBindings {
  private HoldemWassersteinBindings() {}

  /**
   * Computes Wasserstein-1 distance between two discrete distributions.
   *
   * @param weightsA   source distribution weights (length n, sums to 1.0)
   * @param weightsB   target distribution weights (length m, sums to 1.0)
   * @param costMatrix ground metric cost matrix (length n*m, row-major,
   *                   {@code cost[i*m + j]} = distance from support i to support j)
   * @param n          size of distribution A
   * @param m          size of distribution B
   * @param result     output array of length >= 1; receives W_1 distance at index 0
   * @return 0 on success, non-zero error code on failure
   */
  public static native int computeEmd(
      double[] weightsA,
      double[] weightsB,
      double[] costMatrix,
      int n,
      int m,
      double[] result
  );

  /**
   * Batch EMD: computes W_1 for K distributions against one shared reference.
   *
   * @param weightsRef   reference distribution (length n, sums to 1.0)
   * @param weightsBatch K target distributions packed row-major (length K*m)
   * @param costMatrix   shared ground metric (length n*m, row-major)
   * @param n            size of reference distribution
   * @param m            size of each target distribution
   * @param k            number of target distributions
   * @param results      output array of length K; receives W_1 distances
   * @return 0 on success, non-zero error code on failure
   */
  public static native int computeEmdBatch(
      double[] weightsRef,
      double[] weightsBatch,
      double[] costMatrix,
      int n,
      int m,
      int k,
      double[] results
  );

  /**
   * Returns the native engine identifier.
   * Always returns 1 (CPU) for this binding.
   */
  public static native int queryNativeEngine();
}
