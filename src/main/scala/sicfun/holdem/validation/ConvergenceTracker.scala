package sicfun.holdem.validation

import scala.collection.mutable
import scala.util.boundary
import scala.util.boundary.break

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
  *
  * Detection requires `stabilityThreshold` consecutive positive chunks
  * to avoid one-chunk flicker marking a leak as permanently detected.
  */
final class ConvergenceTracker(val leakId: String, stabilityThreshold: Int = 2):
  private val chunks = mutable.ArrayBuffer.empty[(Int, Boolean, Double, Int)]

  def recordChunk(chunkIndex: Int, detected: Boolean, confidence: Double, falsePositives: Int): Unit =
    chunks += ((chunkIndex, detected, confidence, falsePositives))

  def summary(handsPerChunk: Int): ConvergenceSummary =
    // Require stabilityThreshold consecutive detected chunks for stable detection
    val stableDetectedChunk = findStableDetection()
    ConvergenceSummary(
      leakId = leakId,
      detected = stableDetectedChunk.isDefined,
      firstDetectedChunk = stableDetectedChunk,
      handsToDetect = stableDetectedChunk.map(idx => (idx + 1) * handsPerChunk),
      finalConfidence = chunks.lastOption.map(_._3).getOrElse(0.0),
      totalFalsePositives = chunks.map(_._4).sum
    )

  /** Find the first chunk index where detection is stable (N consecutive positives). */
  private def findStableDetection(): Option[Int] =
    if stabilityThreshold <= 1 then chunks.find(_._2).map(_._1)
    else
      boundary:
        var consecutive = 0
        for (chunkIdx, detected, _, _) <- chunks do
          if detected then
            consecutive += 1
            if consecutive >= stabilityThreshold then
              break(Some(chunkIdx - stabilityThreshold + 1))
          else consecutive = 0
        None
