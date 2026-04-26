package sicfun.holdem.runtime.metrics

class MbbMetricsTest extends munit.FunSuite:

  test("mbbPer100: 2 BB/hand constant → 200,000 mbb/100 (v1 bug regression)"):
    val wins = (1 to 100).map(_ => 2.0).toVector
    val got = MbbMetrics.mbbPer100(wins)
    assertEqualsDouble(got, 200_000.0, 1e-6)

  test("mbbPer100: zero-mean → 0"):
    val wins = Vector(1.0, -1.0, 2.0, -2.0)
    assertEqualsDouble(MbbMetrics.mbbPer100(wins), 0.0, 1e-6)

  test("bootstrapIC95: deterministic under seed + IC contains the mean"):
    val wins = (1 to 1000).map(i => if i % 2 == 0 then 1.0 else -1.0).toVector
    val ci = MbbMetrics.bootstrapIC95(wins, iterations = 500, rngSeed = 42L)
    val mean = MbbMetrics.mbbPer100(wins)
    assert(ci.lower <= mean && mean <= ci.upper, s"mean $mean not in $ci")
    val ci2 = MbbMetrics.bootstrapIC95(wins, iterations = 500, rngSeed = 42L)
    assertEquals(ci, ci2)
