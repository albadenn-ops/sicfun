package sicfun.holdem.runtime

import sicfun.holdem.runtime.agent.SeatAgent

final case class BenchmarkGateViolation(message: String)
    extends RuntimeException(message)

object BenchmarkGate:

  def check(
      agents: Vector[SeatAgent],
      benchmarkMode: Boolean
  ): Either[BenchmarkGateViolation, Unit] =
    if !benchmarkMode then Right(())
    else
      val placeholders = agents.flatMap(a => PlaceholderMarker.scanPlaceholders(a))
      if placeholders.isEmpty then Right(())
      else
        val reasons = placeholders.map(p => s"  - ${p.getClass.getSimpleName}: ${p.placeholderReason}").distinct
        Left(BenchmarkGateViolation(
          s"BenchmarkGate refuses to run in benchmarkMode=true. Reachable placeholders:\n${reasons.mkString("\n")}"
        ))
