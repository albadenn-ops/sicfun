package sicfun.holdem.strategic

/** Domain-specific opaque types for the formal strategic layer.
  *
  * These opaques prevent accidental mixing of raw Doubles that represent
  * semantically distinct quantities (chip amounts, pot fractions, expected values).
  * The bridge layer converts raw Doubles from the engine into these types.
  */

opaque type Chips = Double
object Chips:
  inline def apply(value: Double): Chips = value
  extension (c: Chips)
    inline def value: Double = c
    inline def +(other: Chips): Chips = c + other
    inline def -(other: Chips): Chips = c - other
    inline def *(scalar: Double): Chips = c * scalar
    inline def /(divisor: Double): Chips = c / divisor
    inline def unary_- : Chips = -c
    inline def >=(other: Chips): Boolean = c >= other
    inline def >(other: Chips): Boolean = c > other
    inline def <=(other: Chips): Boolean = c <= other
    inline def <(other: Chips): Boolean = c < other
  given Ordering[Chips] = Ordering.Double.TotalOrdering

opaque type PotFraction = Double
object PotFraction:
  inline def apply(value: Double): PotFraction = value
  val Zero: PotFraction = 0.0
  val One: PotFraction = 1.0
  extension (p: PotFraction)
    inline def value: Double = p
    inline def +(other: PotFraction): PotFraction = p + other
    inline def -(other: PotFraction): PotFraction = p - other
    inline def *(scalar: Double): PotFraction = p * scalar
    inline def >=(other: PotFraction): Boolean = p >= other
    inline def >(other: PotFraction): Boolean = p > other

opaque type Ev = Double
object Ev:
  inline def apply(value: Double): Ev = value
  val Zero: Ev = 0.0
  extension (e: Ev)
    inline def value: Double = e
    inline def +(other: Ev): Ev = e + other
    inline def -(other: Ev): Ev = e - other
    inline def *(scalar: Double): Ev = e * scalar
    inline def unary_- : Ev = -e
    inline def >(other: Ev): Boolean = e > other
    inline def <(other: Ev): Boolean = e < other
    inline def >=(other: Ev): Boolean = e >= other
    inline def <=(other: Ev): Boolean = e <= other
    inline def abs: Ev = math.abs(e)

