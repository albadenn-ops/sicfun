package sicfun.holdem.strategic

/** Fidelity of a formal-to-engine correspondence. */
enum Fidelity:
  case Exact
  case Approximate
  case Absent

/** Impact of fidelity loss on the formal model's decision quality. */
enum Severity:
  case Cosmetic    // loss does not affect decisions
  case Behavioral  // affects decision quality but not reasoning structure
  case Structural  // affects reasoning structure or model coherence
  case Critical    // invalidates a semantic law or makes a canonical operator inoperable

/** Result of a bridge conversion, carrying fidelity information with the value. */
sealed trait BridgeResult[+A]:
  def fidelity: Fidelity

  /** Safe consumption via fold. Forces callers to handle all cases. */
  def fold[B](
      onExact: A => B,
      onApprox: (A, String) => B,
      onAbsent: String => B
  ): B = this match
    case BridgeResult.Exact(value) => onExact(value)
    case BridgeResult.Approximate(value, loss) => onApprox(value, loss)
    case BridgeResult.Absent(reason) => onAbsent(reason)

object BridgeResult:
  final case class Exact[A](value: A) extends BridgeResult[A]:
    def fidelity: Fidelity = Fidelity.Exact

  final case class Approximate[A](value: A, lossDescription: String) extends BridgeResult[A]:
    def fidelity: Fidelity = Fidelity.Approximate

  final case class Absent(reason: String) extends BridgeResult[Nothing]:
    def fidelity: Fidelity = Fidelity.Absent
