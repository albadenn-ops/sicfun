package sicfun.core

class MetricsPercentileTest extends munit.FunSuite:
  test("percentile on sorted-ish data uses linear interpolation") {
    val xs = Vector(1.0, 2.0, 3.0, 4.0)
    assertEqualsDouble(Metrics.percentile(xs, 0.0), 1.0, 1e-12)
    assertEqualsDouble(Metrics.percentile(xs, 1.0), 4.0, 1e-12)
    assertEqualsDouble(Metrics.percentile(xs, 0.5), 2.5, 1e-12) // p*(n-1)=1.5 -> between 2 and 3
  }
  test("percentile sorts input defensively") {
    assertEqualsDouble(Metrics.percentile(Vector(4.0, 1.0, 3.0, 2.0), 0.5), 2.5, 1e-12)
  }
  test("percentile on empty is 0.0 and single element is that element") {
    assertEqualsDouble(Metrics.percentile(Vector.empty, 0.5), 0.0, 1e-12)
    assertEqualsDouble(Metrics.percentile(Vector(7.0), 0.9), 7.0, 1e-12)
  }
  test("stdDev is sqrt of sample variance") {
    val xs = Vector(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
    assertEqualsDouble(Metrics.stdDev(xs), math.sqrt(Metrics.variance(xs)), 1e-12)
  }
