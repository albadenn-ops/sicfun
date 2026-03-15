package sicfun.holdem.validation

import scala.collection.mutable

/** Summary of leak detection convergence for one player. */
final case class ConvergenceSummary(
    leakId: String,
    detected: Boolean,
    firstDetectedChunk: Option[Int],
    handsToDetect: Option[Int],
    finalConfidence: Double,
    totalFalsePositives: Int
)

/** Tracks per-chunk leak detection progress during validation.
  *
  * After each 1000-hand chunk is fed through the profiling pipeline,
  * the runner records whether the leak was detected, at what confidence,
  * and how many false positives appeared. The summary answers:
  * "how many hands does it take to catch this leak?"
  */
final class ConvergenceTracker(val leakId: String):
  private val chunks = mutable.ArrayBuffer.empty[(Int, Boolean, Double, Int)]

  def recordChunk(chunkIndex: Int, detected: Boolean, confidence: Double, falsePositives: Int): Unit =
    chunks += ((chunkIndex, detected, confidence, falsePositives))

  def summary(handsPerChunk: Int): ConvergenceSummary =
    val firstDetected = chunks.find(_._2).map(_._1)
    ConvergenceSummary(
      leakId = leakId,
      detected = firstDetected.isDefined,
      firstDetectedChunk = firstDetected,
      handsToDetect = firstDetected.map(idx => (idx + 1) * handsPerChunk),
      finalConfidence = chunks.lastOption.map(_._3).getOrElse(0.0),
      totalFalsePositives = chunks.map(_._4).sum
    )
