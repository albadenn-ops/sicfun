package sicfun.holdem.strategic.calibration

import java.nio.file.{Path, Paths}
import scala.collection.mutable
import sicfun.holdem.history.HandHistoryImport
import sicfun.holdem.types.{PokerAction, Street}
import sicfun.holdem.strategic.types.StrategicClass
import sicfun.holdem.strategic.safety.{BaselineArtifact, BaselineArtifactIO, BaselineMetadata, BoardBucket}

/** Offline tool: ingest a hand-history corpus and tally per-(class,board-bucket,street,action)
  * action counts from SHOWDOWN-revealed holdings (postflop decisions only, v1).
  *
  * CLI: `sbt "runMain sicfun.holdem.strategic.calibration.BaselineCalibrationTool <corpusPath> <outDir> [--minCount=30] [--alpha=1.0] [--corpusId=name]"`
  */
object BaselineCalibrationTool:
  private val DefaultMinCount = 30
  private val DefaultAlpha    = 1.0

  /** Pure calibration: corpus file → BaselineArtifact (raw counts + metadata). */
  def calibrate(
      corpus: Path,
      corpusId: String,
      minCount: Int = DefaultMinCount,
      alpha: Double = DefaultAlpha,
      nowEpochMillis: Long = System.currentTimeMillis()
  ): BaselineArtifact =
    val hands = HandHistoryImport.parseFile(corpus) match
      case Right(hs) => hs
      case Left(err) => throw new IllegalArgumentException(s"corpus parse failed: $err")

    val counts      = mutable.Map.empty[(StrategicClass, String, Street, PokerAction.Category), Long]
    val classCache  = mutable.Map.empty[(String, String), StrategicClass] // (holdingToken, boardToken) → class
    var showdownDecisions = 0L

    hands.foreach { hand =>
      hand.events.foreach { ev =>
        hand.showdownCards.get(ev.playerId).foreach { holding =>
          // v1: postflop only — skip preflop and empty-board events
          if ev.street != Street.Preflop && ev.board.cards.nonEmpty then
            val holdingToken = holding.toToken
            val boardToken   = ev.board.cards.map(_.toToken).sorted.mkString
            val cls = classCache.getOrElseUpdate(
              (holdingToken, boardToken),
              HoldingClassifier.classify(holding, ev.board, ev.street)
            )
            val bucket = BoardBucket.token(BoardBucket.ofBoard(ev.street, ev.board))
            val key    = (cls, bucket, ev.street, ev.action.category)
            counts.update(key, counts.getOrElse(key, 0L) + 1L)
            showdownDecisions += 1
        }
      }
    }

    val meta = BaselineMetadata(
      formatVersion              = BaselineArtifactIO.FormatVersion,
      bucketSchemeVersion        = "v1",
      corpusId                   = corpusId,
      handCount                  = hands.length.toLong,
      showdownDecisionCount      = showdownDecisions,
      recommendedMinCount        = minCount,
      recommendedSmoothingAlpha  = alpha,
      calibrationEpochMillis     = nowEpochMillis
    )
    BaselineArtifact(counts.toMap, meta)

  def write(outDir: Path, artifact: BaselineArtifact): Unit =
    BaselineArtifactIO.save(outDir, artifact)

  def main(args: Array[String]): Unit =
    if args.length < 2 then
      System.err.println(
        "usage: BaselineCalibrationTool <corpusPath> <outDir> [--minCount=30] [--alpha=1.0] [--corpusId=name]"
      )
      sys.exit(1)
    val corpus  = Paths.get(args(0))
    val outDir  = Paths.get(args(1))
    val opts = args.drop(2).flatMap { a =>
      val kv = a.stripPrefix("--").split("=", 2)
      if kv.length == 2 then Some(kv(0) -> kv(1)) else None
    }.toMap
    val minCount = opts.get("minCount").map(_.toInt).getOrElse(DefaultMinCount)
    val alpha    = opts.get("alpha").map(_.toDouble).getOrElse(DefaultAlpha)
    val corpusId = opts.getOrElse("corpusId", corpus.getFileName.toString)
    val artifact = calibrate(corpus, corpusId, minCount, alpha)
    write(outDir, artifact)
    println(s"baseline artifact: ${outDir.toAbsolutePath.normalize()}")
    println(s"hands: ${artifact.metadata.handCount}")
    println(s"showdownDecisions: ${artifact.metadata.showdownDecisionCount}")
    println(s"cells: ${artifact.counts.size}")
