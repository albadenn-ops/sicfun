package sicfun.holdem.runtime.agent

class BlueprintFormatTest extends munit.FunSuite:
  test("BlueprintHeader: magic must be SICFBP01"):
    val ok = BlueprintHeader("SICFBP01", 1, 0L, "h", 9, 0, 5)
    intercept[IllegalArgumentException](ok.copy(magic = "WRONG"))

  test("BlueprintHeader: numSeats in [2,9], numAbstractActions == 5 for A.1"):
    def mk(n: Int, a: Int) = BlueprintHeader("SICFBP01", 1, 0L, "h", n, 0, a)
    mk(2, 5); mk(9, 5)
    intercept[IllegalArgumentException](mk(1, 5))
    intercept[IllegalArgumentException](mk(10, 5))
    intercept[IllegalArgumentException](mk(6, 0))

  test("AbstractActionDistribution: probs sum to 1.0 ± 1e-3, non-empty"):
    AbstractActionDistribution(Array(0.2f, 0.2f, 0.2f, 0.2f, 0.2f))
    intercept[IllegalArgumentException](AbstractActionDistribution(Array.empty[Float]))
    intercept[IllegalArgumentException](AbstractActionDistribution(Array(0.5f, 0.3f))) // sum = 0.8
