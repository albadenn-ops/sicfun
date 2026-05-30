package sicfun.holdem.bench

class CounterfactualHandHistoryBenchmarkTest extends munit.FunSuite:
  test("counterfactual benchmark on the synthetic 9-max corpus yields decisions and a finite bb/100 delta") {
    val txt = scala.io.Source.fromResource("handhistory/p0-sample-9max.txt").mkString
    val res = CounterfactualHandHistoryBenchmark.runText(txt, heroName = "Hero", seed = 3L, resamples = 500)
    assert(res.isRight, res.toString)
    val r = res.toOption.get
    assert(r.decisions > 0, s"expected >=1 analyzed hero decision, got ${r.decisions}")
    assert(r.ci.pointEstimate.isFinite, s"non-finite point estimate ${r.ci.pointEstimate}")
    assert(r.ci.lower <= r.ci.pointEstimate && r.ci.pointEstimate <= r.ci.upper, s"CI does not bracket point: $r")
    // recommended (best) action EV >= actual action EV by construction, so following sicfun is >= 0 on average
    assert(r.ci.pointEstimate >= -1e-6, s"recommended-minus-actual should be >= 0 on average, got ${r.ci.pointEstimate}")
  }
