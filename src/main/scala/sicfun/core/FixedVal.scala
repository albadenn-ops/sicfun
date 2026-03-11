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

  inline val ScaleBits = 13
  inline val Scale = 1 << ScaleBits

  inline def apply(raw: Int): FixedVal = raw
  inline def fromInt(value: Int): FixedVal = value << ScaleBits
  def fromDouble(value: Double): FixedVal = Math.round(value * Scale.toDouble).toInt

  inline def Zero: FixedVal = 0

  extension (value: FixedVal)
    inline def raw: Int = value
    inline def toDouble: Double = value.toDouble / Scale.toDouble

    inline def +(other: FixedVal): FixedVal = value + other
    inline def -(other: FixedVal): FixedVal = value - other
    inline def unary_- : FixedVal = -value

    inline def *(factor: Int): FixedVal = value * factor
    inline def /(divisor: Int): FixedVal = value / divisor

    inline def >(other: FixedVal): Boolean = value > other
    inline def >=(other: FixedVal): Boolean = value >= other
    inline def <(other: FixedVal): Boolean = value < other
    inline def <=(other: FixedVal): Boolean = value <= other

export FixedVal.FixedVal
