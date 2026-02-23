package sicfun.holdem;

/**
 * JNI binding for OpenCL-accelerated batch heads-up equity computation.
 *
 * <p>Backed by {@code sicfun_opencl_kernel.dll}, which dynamically loads the
 * OpenCL runtime ({@code OpenCL.dll}) at startup.  Only Monte Carlo mode is
 * supported; exact enumeration should fall back to the CPU or CUDA backends.
 *
 * <p>Status codes 200-206 are specific to the OpenCL implementation.
 */
public final class HeadsUpOpenCLNativeBindings {
  private HeadsUpOpenCLNativeBindings() {}

  /**
   * Returns the number of available OpenCL GPU devices, or 0 if OpenCL is
   * not installed or no GPU devices are found.
   */
  public static native int openclDeviceCount();

  /**
   * Returns a pipe-delimited descriptor for the given OpenCL GPU device.
   *
   * <p>Format: {@code "name|computeUnits|clockMHz|memoryMB|vendor"}
   *
   * @param deviceIndex zero-based OpenCL GPU device ordinal
   * @return device descriptor string, or empty string on failure
   */
  public static native String openclDeviceInfo(int deviceIndex);

  /**
   * Computes heads-up equity on a specific OpenCL GPU device.
   *
   * <p>Only Monte Carlo mode ({@code modeCode = 1}) is supported.
   * Exact mode ({@code modeCode = 0}) returns status 111.
   *
   * @param deviceIndex zero-based OpenCL GPU device ordinal
   * @param lowIds      hero hole-card pair indices (length N)
   * @param highIds     villain hole-card pair indices (length N)
   * @param modeCode    computation mode (must be 1 for Monte Carlo)
   * @param trials      Monte Carlo trials per matchup
   * @param seeds       per-matchup PRNG seeds (length N)
   * @param wins        output: win probability (length N)
   * @param ties        output: tie probability (length N)
   * @param losses      output: loss probability (length N)
   * @param stderrs     output: standard error (length N)
   * @return 0 on success, or a non-zero status code:
   *         <ul>
   *           <li>100-127 -- JNI/input validation (shared with CUDA)</li>
   *           <li>200 -- OpenCL runtime not available</li>
   *           <li>201 -- no OpenCL GPU devices found</li>
   *           <li>202 -- OpenCL kernel compilation failed</li>
   *           <li>203 -- OpenCL buffer allocation failed</li>
   *           <li>204 -- OpenCL kernel execution failed</li>
   *           <li>205 -- OpenCL result read-back failed</li>
   *           <li>206 -- invalid OpenCL device index</li>
   *         </ul>
   */
  public static native int computeBatch(
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
   * Returns the engine code from the most recent {@link #computeBatch} call.
   *
   * <p>Engine codes:
   * <ul>
   *   <li>0 -- unknown / not set</li>
   *   <li>4 -- OpenCL engine</li>
   * </ul>
   */
  public static native int lastEngineCode();
}
