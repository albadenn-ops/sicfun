package sicfun.holdem;

/**
 * JNI binding for batch heads-up Texas Hold'em equity computation.
 *
 * <p>This class declares native methods backed by two alternative
 * native implementations:
 * <ul>
 *   <li>{@code sicfun_native_cpu.dll} -- CPU-only, multi-threaded
 *       (HeadsUpGpuNativeBindings.cpp)</li>
 *   <li>{@code sicfun_gpu_kernel.dll} -- CUDA+CPU dual-engine, selectable at
 *       runtime (HeadsUpGpuNativeBindingsCuda.cu)</li>
 * </ul>
 *
 * <p>Despite the class name referencing "Gpu", the CPU-only library provides
 * full functionality without any GPU dependency.
 */
public final class HeadsUpGpuNativeBindings {
  private HeadsUpGpuNativeBindings() {}

  /**
   * Computes heads-up equity for a batch of hero-vs-villain matchups.
   *
   * <p>Each matchup is identified by canonical hole-card pair indices (0..1325)
   * representing the C(52,2) = 1326 distinct two-card combinations, ordered
   * lexicographically by (first card, second card) where first &lt; second.
   *
   * @param lowIds   hole-card pair index for each hero hand (length N)
   * @param highIds  hole-card pair index for each villain hand (length N)
   * @param modeCode computation mode: 0 = exact exhaustive enumeration of all
   *                 C(48,5) = 1,712,304 boards; 1 = Monte Carlo sampling
   * @param trials   number of Monte Carlo trials per matchup (ignored when
   *                 modeCode is 0; must be &gt; 0 when modeCode is 1)
   * @param seeds    per-matchup PRNG seed for Monte Carlo (length N; ignored
   *                 when modeCode is 0)
   * @param wins     output array: win probability for each matchup (length N)
   * @param ties     output array: tie probability for each matchup (length N)
   * @param losses   output array: loss probability for each matchup (length N)
   * @param stderrs  output array: standard error of the equity estimate per
   *                 matchup (0.0 for exact mode; length N)
   * @return 0 on success, or a non-zero status code on failure:
   *         <ul>
   *           <li>100 -- null array argument</li>
   *           <li>101 -- array length mismatch (all arrays must be the same length)</li>
   *           <li>102 -- JNI read error (GetXxxArrayRegion failed)</li>
   *           <li>111 -- invalid modeCode (not 0 or 1)</li>
   *           <li>124 -- JNI write error (SetDoubleArrayRegion failed)</li>
   *           <li>125 -- invalid hole-card pair id (out of range 0..1325)</li>
   *           <li>126 -- invalid trials count (must be &gt; 0 for Monte Carlo)</li>
   *           <li>127 -- overlapping cards between hero and villain</li>
   *         </ul>
   *         The CUDA-enabled implementation (.cu) may additionally return:
   *         <ul>
   *           <li>130 -- no CUDA device available</li>
   *           <li>131 -- device memory allocation (cudaMalloc) failed</li>
   *           <li>132 -- host-to-device transfer failed</li>
   *           <li>133 -- kernel launch failed</li>
   *           <li>134 -- device synchronization failed</li>
   *           <li>135 -- device-to-host transfer failed</li>
   *           <li>136 -- constant memory lookup upload failed</li>
   *           <li>137 -- TDR/WDDM watchdog timeout (cudaErrorLaunchTimeout)</li>
   *         </ul>
   */
  public static native int computeBatch(
      int[] lowIds,
      int[] highIds,
      int modeCode,
      int trials,
      long[] seeds,
      double[] wins,
      double[] ties,
      double[] losses,
      double[] stderrs
  );

  /**
   * Computes heads-up equity using the native CPU engine only.
   *
   * <p>Unlike {@link #computeBatch}, this method never routes through CUDA and is
   * safe to use when callers need an explicit CPU baseline while CUDA calls may be
   * executing in parallel.
   *
   * <p>Parameter and status semantics are identical to {@link #computeBatch}.
   */
  public static native int computeBatchCpuOnly(
      int[] lowIds,
      int[] highIds,
      int modeCode,
      int trials,
      long[] seeds,
      double[] wins,
      double[] ties,
      double[] losses,
      double[] stderrs
  );

  /**
   * Returns the engine used by the most recent {@link #computeBatch} call in this
   * native library instance.
   *
   * <p>Engine codes:
   * <ul>
   *   <li>0 -- unknown / not set</li>
   *   <li>1 -- CPU engine</li>
   *   <li>2 -- CUDA engine</li>
   *   <li>3 -- CPU engine after CUDA failure fallback (auto mode)</li>
   * </ul>
   *
   * <p>If the loaded native library predates this method, callers should handle
   * {@link UnsatisfiedLinkError} and treat the engine as unknown.
   */
  public static native int lastEngineCode();

  /**
   * Returns the number of available CUDA devices, or 0 if the CUDA runtime
   * is not available or no CUDA-capable GPUs are detected.
   *
   * <p>Only available in the CUDA-enabled native library ({@code sicfun_gpu_kernel.dll}).
   * Callers should handle {@link UnsatisfiedLinkError} when loaded against the
   * CPU-only library.
   */
  public static native int cudaDeviceCount();

  /**
   * Returns a pipe-delimited descriptor string for the given CUDA device.
   *
   * <p>Format: {@code "name|smCount|clockMHz|memoryMB|computeMajor.computeMinor"}
   *
   * <p>Returns an empty string if the device index is out of range or the
   * query fails.
   *
   * @param deviceIndex zero-based CUDA device ordinal
   * @return device descriptor string, or empty string on failure
   */
  public static native String cudaDeviceInfo(int deviceIndex);

  /**
   * Computes heads-up equity on a specific CUDA device, identified by
   * {@code deviceIndex}. This method does <em>not</em> fall back to the
   * CPU engine on CUDA failure; the caller is responsible for fallback.
   *
   * <p>Parameter semantics are identical to {@link #computeBatch}, with the
   * addition of {@code deviceIndex} to select the target GPU.
   *
   * @param deviceIndex zero-based CUDA device ordinal
   * @param lowIds      hero hole-card pair indices (length N)
   * @param highIds     villain hole-card pair indices (length N)
   * @param modeCode    0 = exact, 1 = Monte Carlo
   * @param trials      Monte Carlo trials per matchup
   * @param seeds       per-matchup PRNG seeds (length N)
   * @param wins        output: win probability (length N)
   * @param ties        output: tie probability (length N)
   * @param losses      output: loss probability (length N)
   * @param stderrs     output: standard error (length N)
   * @return 0 on success, or a non-zero status code (same as
   *         {@link #computeBatch}, plus 138 for invalid device index)
   */
  public static native int computeBatchOnDevice(
      int deviceIndex,
      int[] lowIds,
      int[] highIds,
      int modeCode,
      int trials,
      long[] seeds,
      double[] wins,
      double[] ties,
      double[] losses,
      double[] stderrs
  );

  /**
   * High-throughput packed batch API.
   *
   * <p>This variant minimizes host-to-device/device-to-host bandwidth:
   * <ul>
   *   <li>Input matchups are packed into a single int per entry ({@code lowId<<11 | highId}).</li>
   *   <li>Per-matchup Monte Carlo seeds are generated natively from
   *       {@code monteCarloSeedBase} and {@code keyMaterial[idx]}.</li>
   *   <li>Outputs are returned as float arrays.</li>
   * </ul>
   *
   * <p>Status codes are the same as {@link #computeBatch}.
   */
  public static native int computeBatchPacked(
      int[] packedKeys,
      int modeCode,
      int trials,
      long monteCarloSeedBase,
      long[] keyMaterial,
      float[] wins,
      float[] ties,
      float[] losses,
      float[] stderrs
  );

  /**
   * Device-pinned variant of {@link #computeBatchPacked}.
   *
   * <p>Semantics are identical to {@link #computeBatchOnDevice}, using packed
   * inputs and float outputs.
   */
  public static native int computeBatchPackedOnDevice(
      int deviceIndex,
      int[] packedKeys,
      int modeCode,
      int trials,
      long monteCarloSeedBase,
      long[] keyMaterial,
      float[] wins,
      float[] ties,
      float[] losses,
      float[] stderrs
  );

  /**
   * Evaluates hero equities against villain ranges encoded as CSR.
   *
   * <p>CSR layout:
   * <ul>
   *   <li>{@code heroIds.length = H}</li>
   *   <li>{@code offsets.length = H + 1}</li>
   *   <li>villain entry span for hero {@code h}: {@code [offsets[h], offsets[h+1])}</li>
   *   <li>{@code villainIds.length = keyMaterial.length = probabilities.length = offsets[H]}</li>
   * </ul>
   *
   * <p>Monte Carlo only. Per-entry seeds are generated natively from
   * {@code mix64(monteCarloSeedBase ^ keyMaterial[idx])}.
   */
  public static native int computeRangeBatchMonteCarloCsr(
      int[] heroIds,
      int[] offsets,
      int[] villainIds,
      long[] keyMaterial,
      float[] probabilities,
      int trials,
      long monteCarloSeedBase,
      float[] wins,
      float[] ties,
      float[] losses,
      float[] stderrs
  );

  /**
   * Device-pinned variant of {@link #computeRangeBatchMonteCarloCsr}.
   */
  public static native int computeRangeBatchMonteCarloCsrOnDevice(
      int deviceIndex,
      int[] heroIds,
      int[] offsets,
      int[] villainIds,
      long[] keyMaterial,
      float[] probabilities,
      int trials,
      long monteCarloSeedBase,
      float[] wins,
      float[] ties,
      float[] losses,
      float[] stderrs
  );
}
