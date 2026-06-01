package sicfun.holdem.runtime.agent

/** Abstract action index contract (A.1):
  *   0 = Fold, 1 = Passive (Check or Call), 2 = HalfPot, 3 = Pot, 4 = AllIn.
  *
  * A.2 may expand this space by bumping `version`; `numAbstractActions` in
  * the header records the current width for forward-compat reads. */
final case class BlueprintHeader(
    magic: String,
    version: Int,
    trainedAtEpochSeconds: Long,
    abstractionSpecHash: String,
    numSeats: Int,
    numInfostates: Int,
    numAbstractActions: Int
):
  require(magic == "SICFBP01", s"magic must be 'SICFBP01', got '$magic'")
  require(version >= 1, s"version must be >= 1, got $version")
  require(numSeats >= 2 && numSeats <= 9, s"numSeats must be in [2,9], got $numSeats")
  require(numInfostates >= 0, "numInfostates must be non-negative")
  require(numAbstractActions >= 1, "numAbstractActions must be positive")

final case class AbstractActionDistribution(probs: Array[Float]):
  require(probs.nonEmpty, "probs must be non-empty")
  require(
    math.abs(probs.sum - 1.0f) < 1e-3f,
    s"probs must sum to 1.0 ± 1e-3, got ${probs.sum}"
  )
