package sicfun.holdem.gpu
import sicfun.holdem.types.*
import sicfun.holdem.*
import sicfun.holdem.equity.*

import munit.FunSuite
import sicfun.core.{Card, DiscreteDistribution}

import java.nio.file.{Files, Paths}

/** Parity tests for [[HoldemPostflopNativeRuntime]] -- verifying that native C/CUDA
  * postflop equity computation agrees with the pure-JVM exact equity calculation.
  *
  * These tests require `sicfun_postflop_native.dll` (CPU) and optionally
  * `sicfun_postflop_cuda.dll` (GPU) to be present in `src/main/native/build/`.
  * Tests are skipped when the DLLs are not available.
  *
  * Coverage includes:
  *  - '''Flop Monte Carlo parity''': native MC equity stays within 4% of exact equity
  *    across diverse flop spots (high cards, flush draws, sets, straight draws).
  *  - '''Turn Monte Carlo parity''': tighter 3% tolerance with more trials on turn boards.
  *  - '''River exactness''': with all 5 community cards known, native results must match
  *    exact equity perfectly (no sampling variance) and be deterministic.
  *  - '''Overlap validation''': verifies that overlapping cards (hero card on board)
  *    produce the expected native error (status=127).
  *  - '''Determinism''': same hero/board/villains/seed produces identical results.
  *  - '''Auto-engine fallback''': tests that the "auto" engine mode successfully falls
  *    back to CPU when CUDA is unavailable.
  *  - '''CUDA engine''': when the GPU DLL is present and CUDA is available, verifies
  *    that CUDA-forced computation succeeds.
  */
class HoldemPostflopNativeParityTest extends FunSuite:
  private val ProviderProperty = "sicfun.postflop.provider"
  private val NativePathProperty = "sicfun.postflop.native.path"
  private val NativeGpuPathProperty = "sicfun.postflop.native.gpu.path"
  private val NativeEngineProperty = "sicfun.postflop.native.engine"
  private val nativeDll = Paths
    .get("src", "main", "native", "build", "sicfun_postflop_native.dll")
    .toAbsolutePath
    .normalize()
  private val nativeGpuDll = Paths
    .get("src", "main", "native", "build", "sicfun_postflop_cuda.dll")
    .toAbsolutePath
    .normalize()

  private final case class Spot(hero: HoleCards, villain: HoleCards, board: Board)

  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  private def board(tokens: String*): Board =
    Board.from(tokens.map(card))

  /** Configures the postflop native runtime in CPU mode and runs the thunk.
    * Skips gracefully if the native DLL is not found or fails to load.
    */
  private def withNativeRuntime[A](thunk: => A): Unit =
    if !Files.isRegularFile(nativeDll) then
      println(s"Skipping postflop native parity test: library not found at $nativeDll")
    else
      TestSystemPropertyScope.withSystemProperties(
        Seq(
          ProviderProperty -> Some("native"),
          NativePathProperty -> Some(nativeDll.toString),
          NativeEngineProperty -> Some("cpu")
        )
      ) {
        HoldemPostflopNativeRuntime.resetLoadCacheForTests()
        val availability = HoldemPostflopNativeRuntime.availability
        if !availability.available then
          println(s"Skipping postflop native parity test: ${availability.detail}")
        else
          thunk
      }

  /** Runs a single hero-vs-villain postflop native computation and extracts the
    * single result row. Fails the test if the native call returns an error.
    */
  private def runSingleNative(
      hero: HoleCards,
      board: Board,
      villain: HoleCards,
      trials: Int,
      seed: Long
  ): EquityResultWithError =
    HoldemPostflopNativeRuntime
      .computePostflopBatch(
        hero = hero,
        board = board,
        villains = Array(villain),
        trials = trials,
        seedBase = seed
      )
      .fold(
        reason => fail(s"native postflop call failed: $reason"),
        rows =>
          if rows.length != 1 then fail(s"expected single result row, found ${rows.length}")
          else rows(0)
      )

  test("flop Monte Carlo native equity stays close to exact equity") {
    withNativeRuntime {
      val spots = Vector(
        Spot(hole("As", "Ks"), hole("Qh", "Jd"), board("2c", "3d", "4h")),
        Spot(hole("Ah", "Kh"), hole("Ad", "Qd"), board("2h", "7h", "Tc")),
        Spot(hole("9s", "9h"), hole("Ac", "Kc"), board("9d", "2c", "3c")),
        Spot(hole("7s", "6s"), hole("Ah", "As"), board("8s", "9d", "Td")),
        Spot(hole("Qc", "Qd"), hole("Jh", "Th"), board("2s", "5s", "9s"))
      )
      val trials = 12000
      spots.zipWithIndex.foreach { case (spot, idx) =>
        val range = DiscreteDistribution(Map(spot.villain -> 1.0))
        val exact = HoldemEquity.equityExact(spot.hero, spot.board, range)
        val native = runSingleNative(spot.hero, spot.board, spot.villain, trials, seed = 17L + idx.toLong)

        val winDiff = math.abs(native.win - exact.win)
        val tieDiff = math.abs(native.tie - exact.tie)
        val lossDiff = math.abs(native.loss - exact.loss)
        assert(winDiff <= 0.04, clues(spot, exact, native, winDiff))
        assert(tieDiff <= 0.04, clues(spot, exact, native, tieDiff))
        assert(lossDiff <= 0.04, clues(spot, exact, native, lossDiff))
      }
    }
  }

  test("turn Monte Carlo native equity stays close to exact equity") {
    withNativeRuntime {
      val spots = Vector(
        Spot(hole("As", "Ks"), hole("Qh", "Jd"), board("2c", "3d", "4h", "5s")),
        Spot(hole("Ah", "Kh"), hole("Ad", "Qd"), board("2h", "7h", "Tc", "9c")),
        Spot(hole("9s", "9h"), hole("Ac", "Kc"), board("9d", "2c", "3c", "Ts")),
        Spot(hole("7s", "6s"), hole("Ah", "As"), board("8s", "9d", "Td", "2h")),
        Spot(hole("Qc", "Qd"), hole("Jh", "Th"), board("2s", "5s", "9s", "Kd"))
      )
      val trials = 16000
      spots.zipWithIndex.foreach { case (spot, idx) =>
        val range = DiscreteDistribution(Map(spot.villain -> 1.0))
        val exact = HoldemEquity.equityExact(spot.hero, spot.board, range)
        val native = runSingleNative(spot.hero, spot.board, spot.villain, trials, seed = 101L + idx.toLong)

        val winDiff = math.abs(native.win - exact.win)
        val tieDiff = math.abs(native.tie - exact.tie)
        val lossDiff = math.abs(native.loss - exact.loss)
        assert(winDiff <= 0.03, clues(spot, exact, native, winDiff))
        assert(tieDiff <= 0.03, clues(spot, exact, native, tieDiff))
        assert(lossDiff <= 0.03, clues(spot, exact, native, lossDiff))
      }
    }
  }

  test("river native path is deterministic and exact") {
    withNativeRuntime {
      val spot = Spot(
        hero = hole("As", "Ks"),
        villain = hole("Qh", "Jd"),
        board = board("2c", "3d", "4h", "5s", "9c")
      )
      val range = DiscreteDistribution(Map(spot.villain -> 1.0))
      val exact = HoldemEquity.equityExact(spot.hero, spot.board, range)
      val nativeA = runSingleNative(spot.hero, spot.board, spot.villain, trials = 2000, seed = 77L)
      val nativeB = runSingleNative(spot.hero, spot.board, spot.villain, trials = 2000, seed = 77L)

      assertEquals(nativeA.win, nativeB.win)
      assertEquals(nativeA.tie, nativeB.tie)
      assertEquals(nativeA.loss, nativeB.loss)
      assertEquals(nativeA.stderr, nativeB.stderr)

      assertEquals(nativeA.win, exact.win)
      assertEquals(nativeA.tie, exact.tie)
      assertEquals(nativeA.loss, exact.loss)
      assertEquals(nativeA.stderr, 0.0)
    }
  }

  test("native postflop runtime reports overlap validation errors") {
    withNativeRuntime {
      val hero = hole("As", "Ks")
      val badBoard = board("As", "2d", "3h")
      val villain = hole("Qh", "Jd")
      val result = HoldemPostflopNativeRuntime.computePostflopBatch(
        hero = hero,
        board = badBoard,
        villains = Array(villain),
        trials = 1000,
        seedBase = 5L
      )
      assert(result.isLeft)
      assert(result.left.toOption.exists(_.contains("status=127")), clues(result))
    }
  }

  test("native postflop runtime is deterministic for fixed seed and batch order") {
    withNativeRuntime {
      val hero = hole("As", "Ks")
      val flop = board("2c", "3d", "4h")
      val villains = Array(
        hole("Qh", "Jd"),
        hole("Ad", "Qd"),
        hole("9c", "8d")
      )

      val a = HoldemPostflopNativeRuntime.computePostflopBatch(hero, flop, villains, trials = 6000, seedBase = 42L)
      val b = HoldemPostflopNativeRuntime.computePostflopBatch(hero, flop, villains, trials = 6000, seedBase = 42L)
      assert(a.isRight, clues(a))
      assert(b.isRight, clues(b))
      val rowsA = a.toOption.getOrElse(fail("expected Right for first deterministic call"))
      val rowsB = b.toOption.getOrElse(fail("expected Right for second deterministic call"))
      assertEquals(rowsA.length, rowsB.length)
      var i = 0
      while i < rowsA.length do
        assertEquals(rowsA(i), rowsB(i))
        i += 1
    }
  }

  test("postflop runtime auto-engine falls back to CPU when CUDA execution is unavailable") {
    if !Files.isRegularFile(nativeDll) || !Files.isRegularFile(nativeGpuDll) then
      println(s"Skipping postflop auto-fallback test: missing CPU/GPU dlls at $nativeDll / $nativeGpuDll")
    else
      TestSystemPropertyScope.withSystemProperties(
        Seq(
          ProviderProperty -> Some("native"),
          NativeEngineProperty -> Some("auto"),
          NativePathProperty -> Some(nativeDll.toString),
          NativeGpuPathProperty -> Some(nativeGpuDll.toString)
        )
      ) {
        HoldemPostflopNativeRuntime.resetLoadCacheForTests()
        val hero = hole("As", "Ks")
        val flop = board("2c", "3d", "4h")
        val villains = Array(hole("Qh", "Jd"), hole("Ad", "Qd"))
        val result = HoldemPostflopNativeRuntime.computePostflopBatch(
          hero = hero,
          board = flop,
          villains = villains,
          trials = 3000,
          seedBase = 9L
        )
        assert(result.isRight, clues(result))
      }
  }

  test("postflop runtime CUDA engine runs when CUDA device is available") {
    if !Files.isRegularFile(nativeGpuDll) then
      println(s"Skipping postflop CUDA-engine test: GPU dll not found at $nativeGpuDll")
    else
      TestSystemPropertyScope.withSystemProperties(
        Seq(
          ProviderProperty -> Some("native"),
          NativeEngineProperty -> Some("cuda"),
          NativeGpuPathProperty -> Some(nativeGpuDll.toString)
        )
      ) {
        HoldemPostflopNativeRuntime.resetLoadCacheForTests()
        val availability = HoldemPostflopNativeRuntime.availability
        if !availability.available then
          println(s"Skipping postflop CUDA-engine test: ${availability.detail}")
        else
          val hero = hole("As", "Ks")
          val flop = board("2c", "3d", "4h")
          val villains = Array(hole("Qh", "Jd"))
          HoldemPostflopNativeRuntime.computePostflopBatch(
            hero = hero,
            board = flop,
            villains = villains,
            trials = 1500,
            seedBase = 13L
          ) match
            case Right(rows) =>
              assertEquals(rows.length, villains.length)
            case Left(reason) =>
              if reason.contains("status=130") then
                println("Skipping postflop CUDA-engine test: no CUDA device/runtime available")
              else
                fail(s"unexpected CUDA-engine failure: $reason")
      }
  }
