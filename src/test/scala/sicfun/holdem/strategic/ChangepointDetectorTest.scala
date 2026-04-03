package sicfun.holdem.strategic

import sicfun.core.DiscreteDistribution

class ChangepointDetectorTest extends munit.FunSuite:

  private inline val Tol = 1e-9

  test("ChangepointDetector.initial: run length 0 with probability 1"):
    val cpd = ChangepointDetector(hazardRate = 0.1, rMin = 3, kappaCP = 0.5, wReset = 0.5)
    val state = cpd.initial
    assertEqualsDouble(state.runLengthPosterior.getOrElse(0, 0.0), 1.0, Tol)
    assertEquals(state.runLengthPosterior.size, 1)

  test("ChangepointDetector rejects hazardRate out of (0,1)"):
    intercept[IllegalArgumentException]:
      ChangepointDetector(hazardRate = 0.0, rMin = 3, kappaCP = 0.5, wReset = 0.5)
    intercept[IllegalArgumentException]:
      ChangepointDetector(hazardRate = 1.0, rMin = 3, kappaCP = 0.5, wReset = 0.5)
    intercept[IllegalArgumentException]:
      ChangepointDetector(hazardRate = -0.1, rMin = 3, kappaCP = 0.5, wReset = 0.5)

  test("ChangepointDetector rejects kappaCP out of (0,1)"):
    intercept[IllegalArgumentException]:
      ChangepointDetector(hazardRate = 0.1, rMin = 3, kappaCP = 0.0, wReset = 0.5)
    intercept[IllegalArgumentException]:
      ChangepointDetector(hazardRate = 0.1, rMin = 3, kappaCP = 1.0, wReset = 0.5)

  test("ChangepointDetector rejects wReset out of (0,1]"):
    intercept[IllegalArgumentException]:
      ChangepointDetector(hazardRate = 0.1, rMin = 3, kappaCP = 0.5, wReset = 0.0)
    intercept[IllegalArgumentException]:
      ChangepointDetector(hazardRate = 0.1, rMin = 3, kappaCP = 0.5, wReset = -0.1)

  test("ChangepointDetector accepts wReset = 1.0"):
    val cpd = ChangepointDetector(hazardRate = 0.1, rMin = 3, kappaCP = 0.5, wReset = 1.0)
    assert(cpd.wReset == 1.0)

  test("After one observation: run length 0 and 1 both present"):
    val cpd = ChangepointDetector(hazardRate = 0.1, rMin = 3, kappaCP = 0.5, wReset = 0.5)
    val state0 = cpd.initial
    val predProb: Int => Double = _ => 0.5
    val state1 = cpd.update(state0, predProb)
    assert(state1.runLengthPosterior.contains(0))
    assert(state1.runLengthPosterior.contains(1))

  test("Run-length posterior sums to 1 after each update"):
    val cpd = ChangepointDetector(hazardRate = 0.1, rMin = 5, kappaCP = 0.5, wReset = 0.5)
    var state = cpd.initial
    val predProb: Int => Double = r => if r < 3 then 0.8 else 0.2
    for step <- 1 to 20 do
      state = cpd.update(state, predProb)
      val total = state.runLengthPosterior.values.sum
      assertEqualsDouble(total, 1.0, Tol)

  test("Adams-MacKay with h=0.5 and uniform pred: symmetric split"):
    val cpd = ChangepointDetector(hazardRate = 0.5, rMin = 0, kappaCP = 0.5, wReset = 1.0)
    val state0 = cpd.initial
    val uniformPred: Int => Double = _ => 1.0
    val state1 = cpd.update(state0, uniformPred)
    assertEqualsDouble(state1.runLengthPosterior.getOrElse(0, 0.0), 0.5, Tol)
    assertEqualsDouble(state1.runLengthPosterior.getOrElse(1, 0.0), 0.5, Tol)

  test("isChangepointDetected true when short run-length mass exceeds kappaCP"):
    val cpd = ChangepointDetector(hazardRate = 0.1, rMin = 3, kappaCP = 0.3, wReset = 0.5)
    val heavyShort = ChangepointState(
      runLengthPosterior = Map(0 -> 0.2, 1 -> 0.15, 2 -> 0.05, 5 -> 0.3, 10 -> 0.3)
    )
    assert(cpd.isChangepointDetected(heavyShort))

  test("isChangepointDetected false when short run-length mass is below kappaCP"):
    val cpd = ChangepointDetector(hazardRate = 0.1, rMin = 3, kappaCP = 0.5, wReset = 0.5)
    val longRuns = ChangepointState(
      runLengthPosterior = Map(0 -> 0.05, 1 -> 0.05, 10 -> 0.4, 20 -> 0.5)
    )
    assert(!cpd.isChangepointDetected(longRuns))

  test("resetPrior blends current with meta prior using wReset"):
    val cpd = ChangepointDetector(hazardRate = 0.1, rMin = 3, kappaCP = 0.3, wReset = 0.4)
    val current = DiscreteDistribution(Map("TAG" -> 0.8, "LAG" -> 0.2))
    val meta = DiscreteDistribution(Map("TAG" -> 0.3, "LAG" -> 0.7))
    val blended = cpd.resetPrior(current, meta)
    assertEqualsDouble(blended.probabilityOf("TAG"), 0.60, Tol)
    assertEqualsDouble(blended.probabilityOf("LAG"), 0.40, Tol)

  test("resetPrior with wReset=1.0 returns pure meta prior"):
    val cpd = ChangepointDetector(hazardRate = 0.1, rMin = 3, kappaCP = 0.3, wReset = 1.0)
    val current = DiscreteDistribution(Map("TAG" -> 0.8, "LAG" -> 0.2))
    val meta = DiscreteDistribution(Map("TAG" -> 0.3, "LAG" -> 0.7))
    val blended = cpd.resetPrior(current, meta)
    assertEqualsDouble(blended.probabilityOf("TAG"), 0.3, Tol)
    assertEqualsDouble(blended.probabilityOf("LAG"), 0.7, Tol)

  test("resetPrior handles disjoint support"):
    val cpd = ChangepointDetector(hazardRate = 0.1, rMin = 3, kappaCP = 0.3, wReset = 0.5)
    val current = DiscreteDistribution(Map("TAG" -> 1.0))
    val meta = DiscreteDistribution(Map("LAG" -> 1.0))
    val blended = cpd.resetPrior(current, meta)
    assertEqualsDouble(blended.probabilityOf("TAG"), 0.5, Tol)
    assertEqualsDouble(blended.probabilityOf("LAG"), 0.5, Tol)

  test("With very low hazardRate, changepoint is never detected"):
    val cpd = ChangepointDetector(hazardRate = 0.001, rMin = 3, kappaCP = 0.5, wReset = 0.5)
    var state = cpd.initial
    val predProb: Int => Double = _ => 0.5
    for _ <- 1 to 100 do
      state = cpd.update(state, predProb)
    assert(!cpd.isChangepointDetected(state))

  test("With low hazardRate, mass concentrates on the longest run length"):
    val cpd = ChangepointDetector(hazardRate = 0.001, rMin = 3, kappaCP = 0.5, wReset = 0.5)
    var state = cpd.initial
    val predProb: Int => Double = _ => 0.5
    for _ <- 1 to 10 do
      state = cpd.update(state, predProb)
    val maxRL = state.runLengthPosterior.maxBy(_._2)._1
    assertEquals(maxRL, 10)

  test("After 5 updates, all integer run lengths 0..5 have some mass"):
    val cpd = ChangepointDetector(hazardRate = 0.2, rMin = 3, kappaCP = 0.5, wReset = 0.5)
    var state = cpd.initial
    val predProb: Int => Double = _ => 0.5
    for _ <- 1 to 5 do
      state = cpd.update(state, predProb)
    for r <- 0 to 5 do
      assert(
        state.runLengthPosterior.getOrElse(r, 0.0) > 0.0,
        s"Run length $r should have positive mass"
      )
