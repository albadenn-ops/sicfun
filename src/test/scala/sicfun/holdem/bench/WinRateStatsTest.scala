package sicfun.holdem.bench

class WinRateStatsTest extends munit.FunSuite:
  test("bbPer100 is mean per-hand bb times 100") {
    // mean = 0.5 bb/hand -> 50 bb/100
    assertEqualsDouble(WinRateStats.bbPer100(Vector(1.0, 0.0, 1.0, 0.0)), 50.0, 1e-9)
  }
  test("bootstrap CI is deterministic for a fixed seed") {
    val xs = Vector.tabulate(500)(i => if i % 2 == 0 then 2.0 else -1.0)
    val a = WinRateStats.bbPer100CI(xs, resamples = 1000, ciLevel = 0.95, seed = 7L)
    val b = WinRateStats.bbPer100CI(xs, resamples = 1000, ciLevel = 0.95, seed = 7L)
    assertEquals(a, b)
  }
  test("CI brackets the point estimate and lower<upper for noisy positive data") {
    val xs = Vector.tabulate(2000)(i => if i % 4 == 0 then 6.0 else -1.0) // mean +0.75 bb/hand = +75 bb/100
    val r = WinRateStats.bbPer100CI(xs, resamples = 2000, ciLevel = 0.95, seed = 1L)
    assert(r.lower < r.pointEstimate, s"lower ${r.lower} !< point ${r.pointEstimate}")
    assert(r.pointEstimate < r.upper, s"point ${r.pointEstimate} !< upper ${r.upper}")
    assert(r.lower > 0.0, s"expected CI to clear zero for strongly positive data, got lower=${r.lower}")
  }
  test("empty sample yields zero estimate and degenerate CI") {
    val r = WinRateStats.bbPer100CI(Vector.empty, resamples = 100, ciLevel = 0.95, seed = 1L)
    assertEqualsDouble(r.pointEstimate, 0.0, 1e-12)
    assertEqualsDouble(r.lower, 0.0, 1e-12)
    assertEqualsDouble(r.upper, 0.0, 1e-12)
    assertEquals(r.sampleSize, 0)
    assertEquals(r.resamples, 100)
    assertEqualsDouble(r.ciLevel, 0.95, 1e-12)
  }
  test("bbPer100CI rejects non-positive resamples and out-of-range ciLevel") {
    intercept[IllegalArgumentException] { WinRateStats.bbPer100CI(Vector(1.0, 2.0), resamples = 0) }
    intercept[IllegalArgumentException] { WinRateStats.bbPer100CI(Vector(1.0, 2.0), ciLevel = 1.0) }
    intercept[IllegalArgumentException] { WinRateStats.bbPer100CI(Vector(1.0, 2.0), ciLevel = 0.0) }
  }
