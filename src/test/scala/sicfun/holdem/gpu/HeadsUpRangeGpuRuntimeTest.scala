package sicfun.holdem.gpu
import sicfun.holdem.types.*
import sicfun.holdem.*
import sicfun.holdem.equity.*

import munit.FunSuite

import java.nio.file.{Files, Paths}
import scala.collection.mutable.ArrayBuffer

class HeadsUpRangeGpuRuntimeTest extends FunSuite:
  private val ProviderProperty = "sicfun.gpu.provider"
  private val NativePathProperty = "sicfun.gpu.native.path"
  private val NativeEngineProperty = "sicfun.gpu.native.engine"

  private def withSystemProperties[A](updates: Seq[(String, Option[String])])(thunk: => A): A =
    TestSystemPropertyScope.withSystemProperties(updates)(thunk)

  private def withProvider[A](provider: String)(thunk: => A): A =
    withSystemProperties(Seq(ProviderProperty -> Some(provider)))(thunk)

  private def disjointVillains(heroId: Int, count: Int): Array[Int] =
    val hero = HoleCardsIndex.byId(heroId)
    val out = ArrayBuffer.empty[Int]
    var idx = 0
    while idx < HoleCardsIndex.size && out.length < count do
      if idx != heroId then
        val villain = HoleCardsIndex.byId(idx)
        if HoleCardsIndex.areDisjoint(hero, villain) then out += idx
      idx += 1
    require(out.length == count, s"unable to sample $count disjoint villains for heroId=$heroId")
    out.toArray

  private def expectedCpu(
      heroIds: Array[Int],
      offsets: Array[Int],
      villainIds: Array[Int],
      keyMaterial: Array[Long],
      probabilities: Array[Float],
      trials: Int,
      seedBase: Long
  ): Array[EquityResultWithError] =
    val out = new Array[EquityResultWithError](heroIds.length)
    val mode = HeadsUpEquityTable.Mode.MonteCarlo(trials)
    var h = 0
    while h < heroIds.length do
      val hero = HoleCardsIndex.byId(heroIds(h))
      val start = offsets(h)
      val end = offsets(h + 1)
      var wWin = 0.0
      var wTie = 0.0
      var wLoss = 0.0
      var wErr2 = 0.0
      var wSum = 0.0
      var i = start
      while i < end do
        val p = probabilities(i).toDouble
        if p > 0.0 then
          val villain = HoleCardsIndex.byId(villainIds(i))
          val r = HeadsUpEquityTable.computeEquityDeterministic(
            hero = hero,
            villain = villain,
            mode = mode,
            monteCarloSeedBase = seedBase,
            keyMaterial = keyMaterial(i)
          )
          wWin += p * r.win
          wTie += p * r.tie
          wLoss += p * r.loss
          wErr2 += (p * r.stderr) * (p * r.stderr)
          wSum += p
        i += 1
      out(h) =
        if wSum > 0.0 then
          EquityResultWithError(
            wWin / wSum,
            wTie / wSum,
            wLoss / wSum,
            math.sqrt(wErr2) / wSum
          )
        else
          EquityResultWithError(0.0, 0.0, 0.0, 0.0)
      h += 1
    out

  test("cpu-emulated provider computes weighted CSR aggregation deterministically") {
    withProvider("cpu-emulated") {
      val heroIds = Array(0, 17)
      val h0Villains = disjointVillains(heroIds(0), 2)
      val h1Villains = disjointVillains(heroIds(1), 3)
      val offsets = Array(0, 2, 5)
      val villainIds = Array(h0Villains(0), h0Villains(1), h1Villains(0), h1Villains(1), h1Villains(2))
      val keyMaterial = Array(11L, 22L, 33L, 44L, 55L)
      val probabilities = Array(0.25f, 0.75f, 0.1f, 0.0f, 0.9f)
      val trials = 21
      val seedBase = 7L

      val resultEither =
        HeadsUpRangeGpuRuntime.computeRangeBatchMonteCarloCsr(
          heroIds,
          offsets,
          villainIds,
          keyMaterial,
          probabilities,
          trials,
          seedBase
        )
      assert(resultEither.isRight)
      val actual = resultEither.toOption.getOrElse(fail("expected Right result"))
      val expected = expectedCpu(heroIds, offsets, villainIds, keyMaterial, probabilities, trials, seedBase)

      assertEquals(actual.length, expected.length)
      var idx = 0
      while idx < actual.length do
        assertEqualsDouble(actual(idx).win, expected(idx).win, 1e-12)
        assertEqualsDouble(actual(idx).tie, expected(idx).tie, 1e-12)
        assertEqualsDouble(actual(idx).loss, expected(idx).loss, 1e-12)
        assertEqualsDouble(actual(idx).stderr, expected(idx).stderr, 1e-12)
        idx += 1
    }
  }

  test("cpu-emulated provider returns zeros for empty or zero-weight rows") {
    withProvider("cpu-emulated") {
      val heroIds = Array(0, 31)
      val h1Villains = disjointVillains(heroIds(1), 2)
      val offsets = Array(0, 0, 2)
      val villainIds = Array(h1Villains(0), h1Villains(1))
      val keyMaterial = Array(101L, 202L)
      val probabilities = Array(0.0f, 0.0f)

      val resultEither =
        HeadsUpRangeGpuRuntime.computeRangeBatchMonteCarloCsr(
          heroIds,
          offsets,
          villainIds,
          keyMaterial,
          probabilities,
          trials = 10,
          monteCarloSeedBase = 123L
        )
      assert(resultEither.isRight)
      val values = resultEither.toOption.getOrElse(fail("expected Right result"))
      assertEquals(values.length, 2)
      values.foreach { v =>
        assertEqualsDouble(v.win, 0.0, 1e-12)
        assertEqualsDouble(v.tie, 0.0, 1e-12)
        assertEqualsDouble(v.loss, 0.0, 1e-12)
        assertEqualsDouble(v.stderr, 0.0, 1e-12)
      }
    }
  }

  test("invalid CSR shape is rejected before execution") {
    withProvider("cpu-emulated") {
      val result =
        HeadsUpRangeGpuRuntime.computeRangeBatchMonteCarloCsr(
          heroIds = Array(0, 1),
          offsets = Array(0, 1), // invalid length: must be heroIds.length + 1
          villainIds = Array(2),
          keyMaterial = Array(3L),
          probabilities = Array(1.0f),
          trials = 8,
          monteCarloSeedBase = 1L
        )
      assert(result.isLeft)
      assert(result.left.toOption.exists(_.contains("offsets length")))
    }
  }

  test("native CSR API works in CPU mode and returns status 112 for invalid layout") {
    val nativeDll = Paths.get("src", "main", "native", "build", "sicfun_gpu_kernel.dll").toAbsolutePath.normalize()
    if !Files.isRegularFile(nativeDll) then
      println(s"Skipping native CSR test: library not found at $nativeDll")
    else
      withSystemProperties(
        Seq(
          ProviderProperty -> Some("native"),
          NativePathProperty -> Some(nativeDll.toString),
          NativeEngineProperty -> Some("cpu")
        )
      ) {
        val availability = HeadsUpGpuRuntime.availability
        if !availability.available then
          println(s"Skipping native CSR test: ${availability.detail}")
        else
          val heroId = 0
          val villains = disjointVillains(heroId, 2)
          val runtimeResult =
            HeadsUpRangeGpuRuntime.computeRangeBatchMonteCarloCsr(
              heroIds = Array(heroId),
              offsets = Array(0, 2),
              villainIds = Array(villains(0), villains(1)),
              keyMaterial = Array(13L, 29L),
              probabilities = Array(0.4f, 0.6f),
              trials = 12,
              monteCarloSeedBase = 5L
            )
          runtimeResult match
            case Left(reason) =>
              fail(s"native CSR runtime call failed: $reason")
            case Right(values) =>
              assertEquals(values.length, 1)
              val sum = values(0).win + values(0).tie + values(0).loss
              assert(java.lang.Double.isFinite(sum))
              assert(math.abs(sum - 1.0) <= 1e-5)

          val outWins = new Array[Float](1)
          val outTies = new Array[Float](1)
          val outLosses = new Array[Float](1)
          val outStderrs = new Array[Float](1)
          val status =
            HeadsUpGpuNativeBindings.computeRangeBatchMonteCarloCsr(
              Array(heroId),
              Array(0, 0), // last offset != entryCount
              Array(villains(0)),
              Array(7L),
              Array(1.0f),
              10,
              1L,
              outWins,
              outTies,
              outLosses,
              outStderrs
            )
          assertEquals(status, 112)
      }
  }
