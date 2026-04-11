package sicfun.core

/** Signed fixed-point value for CFR utilities/regrets, stored as Int32 with 2^13 scale.
  *
  * The scale keeps sub-millibet precision for default 1500-iteration Hold'em solves while
  * leaving enough dynamic range for cumulative regret updates. Long accumulators should be
  * used for values that grow quadratically with iteration count (for example average-strategy
  * mass), so this type focuses on per-iteration utilities and regrets.
  */
object FixedVal:
  opaque type FixedVal = Int

  /** Number of fractional bits. 2^13 = 8192 gives ~0.000122 precision per LSB. */
  inline val ScaleBits = 13
  /** The fixed-point scale factor: 1.0 is represented as 8192. */
  inline val Scale = 1 << ScaleBits

  /** Wraps a raw Int32 value as a FixedVal (no conversion). */
  inline def apply(raw: Int): FixedVal = raw
  /** Converts an integer to fixed-point by left-shifting by ScaleBits. */
  inline def fromInt(value: Int): FixedVal = value << ScaleBits
  /** Converts a Double to fixed-point with round-half-up rounding. */
  def fromDouble(value: Double): FixedVal = Math.round(value * Scale.toDouble).toInt

  /** The zero value (raw = 0). */
  inline def Zero: FixedVal = 0

  extension (value: FixedVal)
    /** Access the raw Int32 representation. */
    inline def raw: Int = value
    /** Convert back to Double by dividing by the scale factor. */
    inline def toDouble: Double = value.toDouble / Scale.toDouble

    // Arithmetic operations work directly on the raw Int values.
    // Addition and subtraction of fixed-point values with the same scale
    // require no scaling adjustment.
    inline def +(other: FixedVal): FixedVal = value + other
    inline def -(other: FixedVal): FixedVal = value - other
    inline def unary_- : FixedVal = -value

    // Multiplication by an integer and division by an integer preserve the scale
    // because only one operand has the scale factor.
    inline def *(factor: Int): FixedVal = value * factor
    inline def /(divisor: Int): FixedVal = value / divisor

    inline def >(other: FixedVal): Boolean = value > other
    inline def >=(other: FixedVal): Boolean = value >= other
    inline def <(other: FixedVal): Boolean = value < other
    inline def <=(other: FixedVal): Boolean = value <= other

/** Re-exports the opaque type so callers can write `FixedVal` without qualifying as `FixedVal.FixedVal`. */
export FixedVal.FixedVal
