package sicfun.holdem;

/** JNI bindings for postflop Monte Carlo equity computation. */
public final class HoldemPostflopNativeBindings {
  private HoldemPostflopNativeBindings() {}

  /**
   * Computes postflop Monte Carlo equity for one fixed hero+board against a batch of villains.
   *
   * <p>Input semantics:
   * <ul>
   *   <li>{@code heroFirst}/{@code heroSecond}: hero card ids in [0, 51]</li>
   *   <li>{@code boardCards.length == boardSize}, boardSize in [1, 5]</li>
   *   <li>{@code villainFirstCards.length == villainSecondCards.length == seeds.length == N}</li>
   *   <li>output arrays ({@code wins/ties/losses/stderrs}) each have length N</li>
   *   <li>{@code trials > 0}</li>
   * </ul>
   *
   * <p>Returns 0 on success; non-zero status on failure.
   */
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

  /**
   * Returns the last engine code used by the native postflop bridge:
   * 0 = unknown, 1 = CPU, 2 = CUDA.
   */
  public static native int queryNativeEngine();
}
