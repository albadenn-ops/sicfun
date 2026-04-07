package sicfun.holdem.validation

import scala.collection.mutable
import scala.util.boundary
import scala.util.boundary.break

/** Summary of leak detection convergence for one player.
  *
  * Answers the question: "After how many hands did the profiling pipeline
  * stably detect this leak?" Used by [[ValidationScorecard]] to report
  * detection speed and reliability.
  *
  * @param leakId             identifier of the injected leak (e.g. "overcall-big-bets")
  * @param detected           true if the leak was stably detected (N consecutive positive chunks)
  * @param firstDetectedChunk the chunk index where stable detection first occurred
  * @param handsToDetect      number of simulated hands required for stable detection
  * @param finalConfidence    the confidence score from the last recorded chunk
  * @param totalFalsePositives cumulative false positive count across all chunks
  */
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

  /** Record the result of one chunk's profiling pass.
    *
    * @param chunkIndex     0-based index of the chunk in the simulation
    * @param detected       whether the profiling pipeline flagged the leak in this chunk
    * @param confidence     profiler's confidence score for this chunk (0.0 to 1.0)
    * @param falsePositives number of false positive leak detections in this chunk
    */
  def recordChunk(chunkIndex: Int, detected: Boolean, confidence: Double, falsePositives: Int): Unit =
    chunks += ((chunkIndex, detected, confidence, falsePositives))

  /** Produce a summary of the convergence analysis.
    *
    * @param handsPerChunk number of hands per chunk (used to convert chunk index to hand count)
    * @return a [[ConvergenceSummary]] with detection status, timing, and false positive count
    */
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
