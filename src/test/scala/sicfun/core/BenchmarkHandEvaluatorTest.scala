package sicfun.core

/** Tests for [[BenchmarkHandEvaluator]] micro-benchmark harness.
  *
  * Validates that:
  *  - The benchmark runs successfully with small parameters and produces finite metrics.
  *  - categorize5 agrees with evaluate5 on all sampled hands (mismatchCount == 0).
  *  - Invalid CLI options produce a Left (error) result rather than throwing.
  */
class BenchmarkHandEvaluatorTest extends munit.FunSuite:
  test("run returns finite metrics and zero mismatches on deterministic sample") {
    val result = BenchmarkHandEvaluator.run(Array(
      "--samples=20000",
      "--warmupRounds=0",
      "--measureRounds=1",
      "--seed=123"
    ))

    assert(result.isRight, s"benchmark run should succeed, got $result")
    val report = result.toOption.get
    assertEquals(report.samples, 20000)
    assertEquals(report.warmupRounds, 0)
    assertEquals(report.measureRounds, 1)
    assertEquals(report.mismatchCount, 0)
    assert(report.evaluate5MeanMillis.isFinite)
    assert(report.categorize5MeanMillis.isFinite)
    assert(report.evaluate5HandsPerSecond.isFinite)
    assert(report.categorize5HandsPerSecond.isFinite)
    assert(report.speedupX.isFinite)
  }

  test("run returns Left on invalid options") {
    val result = BenchmarkHandEvaluator.run(Array("--samples=oops"))
    assert(result.isLeft)
  }
