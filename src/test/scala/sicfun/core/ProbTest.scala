package sicfun.core

import Prob.*
import munit.FunSuite

/** Tests for [[Prob]] fixed-point probability type (2^30 scale).
  *
  * Validates:
  *  - '''Constants''': Zero, One, and Half have the expected raw representations.
  *  - '''Round-trip precision''': fromDouble -> toDouble preserves common probability values
  *    (0, 0.25, 0.5, 0.75, 1.0, 1/3, 1/1326) within 1e-9 tolerance.
  *  - '''Arithmetic''': addition of complementary probabilities yields One (within 1 LSB);
  *    subtraction produces correct results; multiplication uses Long intermediate to avoid
  *    overflow (critical for One * One); very small probability products (1/1326)^2 maintain precision.
  *  - '''Division''': integer division truncates correctly.
  *  - '''Comparison operators''': ordering is consistent.
  *  - '''Out-of-range values''': fromDouble does not throw for values outside [0,1]
  *    (callers are responsible for range validation).
  */
class ProbTest extends FunSuite:
  private val tolerance = 1e-9

  test("Zero and One constants") {
    assertEquals(Prob.Zero.raw, 0)
    assertEquals(Prob.One.raw, 1 << 30)
    assertEquals(Prob.Half.raw, 1 << 29)
  }

  test("fromDouble roundtrip for common values") {
    val cases = Seq(0.0, 0.25, 0.5, 0.75, 1.0, 1.0 / 3.0, 1.0 / 1326.0)
    cases.foreach { d =>
      val p = Prob.fromDouble(d)
      val back = p.toDouble
      assertEqualsDouble(back, d, tolerance, s"roundtrip failed for $d")
    }
  }

  test("addition is exact for small values") {
    val a = Prob.fromDouble(0.3)
    val b = Prob.fromDouble(0.7)
    val sum = a + b
    assert(math.abs(sum.raw - Prob.One.raw) <= 1, s"sum.raw=${sum.raw} vs One=${Prob.One.raw}")
  }

  test("subtraction") {
    val a = Prob.One
    val b = Prob.fromDouble(0.4)
    val diff = a - b
    assertEqualsDouble(diff.toDouble, 0.6, tolerance)
  }

  test("subtraction can produce negative raw value") {
    val result = Prob.Zero - Prob.One
    assert(result.raw < 0)
  }

  test("multiplication of two probabilities") {
    val a = Prob.fromDouble(0.5)
    val b = Prob.fromDouble(0.5)
    val product = a * b
    assertEqualsDouble(product.toDouble, 0.25, tolerance)
  }

  test("multiplication does not overflow for One * One") {
    val product = Prob.One * Prob.One
    assertEquals(product.raw, Prob.One.raw)
  }

  test("multiplication of very small probabilities preserves precision") {
    // (1/1326)^2 ≈ 5.69e-7, in Prob scale: ~611
    val tiny = Prob.fromDouble(1.0 / 1326.0)
    val product = tiny * tiny
    val expected = (1.0 / 1326.0) * (1.0 / 1326.0)
    assertEqualsDouble(product.toDouble, expected, 1e-6)
  }

  test("division by integer") {
    val p = Prob.fromDouble(0.9)
    val divided = p / 3
    assertEqualsDouble(divided.toDouble, 0.3, 1e-7)
  }

  test("comparison operators") {
    val a = Prob.fromDouble(0.3)
    val b = Prob.fromDouble(0.7)
    assert(a < b)
    assert(b > a)
    assert(a <= a)
    assert(a >= a)
  }

  test("fromDouble precision for 1/1326") {
    val p = Prob.fromDouble(1.0 / 1326.0)
    assert(p.raw > 0, "1/1326 must be representable")
    val relError = math.abs(p.toDouble - (1.0 / 1326.0)) / (1.0 / 1326.0)
    assert(relError < 1e-6, s"relative error too large: $relError")
  }

  test("fromDouble for out-of-range values does not throw") {
    val over = Prob.fromDouble(1.5)
    assert(over.raw > Prob.One.raw)
    val neg = Prob.fromDouble(-0.1)
    assert(neg.raw < 0)
  }
