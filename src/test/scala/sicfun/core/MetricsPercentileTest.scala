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
  test("stdDev equals the known sample standard deviation") {
    // sample variance of [1,3]: mean=2, squared devs (1+1)=2, /(n-1=1)=2 -> stdDev=sqrt(2)
    assertEqualsDouble(Metrics.stdDev(Vector(1.0, 3.0)), math.sqrt(2.0), 1e-12)
    // textbook set 2,4,4,4,5,5,7,9 -> sample stdDev = 2.138089935299...
    assertEqualsDouble(Metrics.stdDev(Vector(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)), 2.138089935299395, 1e-9)
  }
  test("stdDev throws on fewer than two values") {
    intercept[IllegalArgumentException] { Metrics.stdDev(Vector(1.0)) }
  }
