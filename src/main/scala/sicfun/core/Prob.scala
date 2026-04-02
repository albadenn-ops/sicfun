package sicfun.core

/** Fixed-point probability in `[0, 1]`, stored as Int32 with `2^30` scale.
  *
  * `1.0 = 1,073,741,824` (`2^30`). The representation uses `2^30` (not `2^31`) so
  * the sign bit stays free and subtraction remains safe. Multiplication uses a Long
  * intermediate to avoid overflow.
  *
  * No clamping or validation is performed: callers must only construct values in
  * the valid range and avoid undefined operations (for example division by zero).
  */
object Prob:
  opaque type Prob = Int

  inline val Scale = 1 << 30 // 1,073,741,824

  inline def apply(raw: Int): Prob = raw

  /** Round-half-up conversion from Double to fixed-point raw value.
    *
    * The method intentionally does not clamp. Values outside `[0, 1]` are converted
    * as-is, so callers are responsible for validating range before conversion.
    */
  def fromDouble(d: Double): Prob = (d * Scale + 0.5).toInt

  inline def One: Prob = Scale
  inline def Zero: Prob = 0
  inline def Half: Prob = Scale >> 1

  extension (p: Prob)
    inline def raw: Int = p
    inline def toDouble: Double = p.toDouble / Scale

    inline def +(q: Prob): Prob = p + q
    inline def -(q: Prob): Prob = p - q

    /** Multiply two probabilities. Uses Long intermediate to avoid overflow. */
    inline def *(q: Prob): Prob = ((p.toLong * q.toLong) >> 30).toInt

    /** Divide by an integer (e.g. boardCount). Truncates. */
    inline def /(n: Int): Prob = p / n

    inline def >(q: Prob): Boolean = p > q
    inline def <(q: Prob): Boolean = p < q
    inline def >=(q: Prob): Boolean = p >= q
    inline def <=(q: Prob): Boolean = p <= q
