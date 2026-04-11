package sicfun.core

import munit.FunSuite

/** Tests for [[FixedVal]] signed fixed-point arithmetic (2^13 scale).
  *
  * Validates:
  *  - `fromInt` correctly left-shifts by ScaleBits and round-trips to Double.
  *  - `fromDouble` round-trips for both positive and negative values within tolerance.
  *  - Addition and subtraction preserve sign and produce correct results.
  *  - Comparison operators work correctly across sign boundaries.
  */
class FixedValTest extends FunSuite:
  private val tolerance = 5e-4

  test("fromInt keeps exact integer values") {
    val value = FixedVal.fromInt(7)
    assertEquals(value.raw, 7 << FixedVal.ScaleBits)
    assertEqualsDouble(value.toDouble, 7.0, tolerance)
  }

  test("fromDouble roundtrip supports signed values") {
    val cases = Seq(-3.25, -0.5, 0.0, 0.125, 2.75)
    cases.foreach { raw =>
      val value = FixedVal.fromDouble(raw)
      assertEqualsDouble(value.toDouble, raw, tolerance, s"roundtrip failed for $raw")
    }
  }

  test("addition and subtraction preserve sign") {
    val a = FixedVal.fromDouble(1.5)
    val b = FixedVal.fromDouble(-0.25)
    assertEqualsDouble((a + b).toDouble, 1.25, tolerance)
    assertEqualsDouble((a - b).toDouble, 1.75, tolerance)
  }

  test("ordering works for negative and positive values") {
    val negative = FixedVal.fromDouble(-0.5)
    val positive = FixedVal.fromDouble(0.5)
    assert(negative < positive)
    assert(positive > negative)
    assert(positive >= positive)
    assert(negative <= negative)
  }
