package sicfun.holdem

import munit.FunSuite

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters.*
import scala.concurrent.duration.*

class TexasHoldemPlayingHallTest extends FunSuite:
  override val munitTimeout: Duration = 90.seconds

  test("playing hall runs end-to-end and emits logs") {
    val root = Files.createTempDirectory("playing-hall-test-")
    try
      val out = root.resolve("hall-out")
      val result = TexasHoldemPlayingHall.run(Array(
        "--hands=120",
        "--reportEvery=60",
        "--learnEveryHands=40",
        "--learningWindowSamples=200",
        "--seed=11",
        s"--outDir=$out",
        "--villainStyle=tag",
        "--raiseSize=2.5",
        "--bunchingTrials=20",
        "--equityTrials=120",
        "--saveTrainingTsv=true"
      ))
      assert(result.isRight, s"hall run failed: $result")
      val summary = result.toOption.getOrElse(fail("missing hall summary"))
      assertEquals(summary.handsPlayed, 120)
      assert(summary.retrains >= 1, s"expected at least one retrain, got ${summary.retrains}")
      assert(Files.exists(out.resolve("hands.tsv")))
      assert(Files.exists(out.resolve("learning.tsv")))
      assert(Files.exists(out.resolve("training-selfplay.tsv")))

      val handRows = Files.readAllLines(out.resolve("hands.tsv"), StandardCharsets.UTF_8).asScala.toVector
      assert(handRows.length > 1, "expected hand log rows")
      val handHeader = handRows.head.split("\t", -1).toVector
      val streetsPlayedIdx = handHeader.indexOf("streetsPlayed")
      assert(streetsPlayedIdx >= 0, "hands.tsv missing streetsPlayed column")
      val sawPostflop = handRows.drop(1).exists { row =>
        val fields = row.split("\t", -1).toVector
        fields.lift(streetsPlayedIdx).flatMap(_.toIntOption).exists(_ > 1)
      }
      assert(sawPostflop, "expected at least one hand to reach postflop")
      val learningRows = Files.readAllLines(out.resolve("learning.tsv"), StandardCharsets.UTF_8).asScala.toVector
      assert(learningRows.length > 1, "expected learning log rows")
      val trainingRows = Files.readAllLines(out.resolve("training-selfplay.tsv"), StandardCharsets.UTF_8).asScala.toVector
      assert(trainingRows.length > 1, "expected training rows")
      val sawPostflopTraining = trainingRows.drop(1).exists { row =>
        row.startsWith("Flop\t") || row.startsWith("Turn\t") || row.startsWith("River\t")
      }
      assert(sawPostflopTraining, "expected training data from postflop streets")
    finally
      deleteRecursively(root)
  }

  test("playing hall rejects invalid hands argument") {
    val result = TexasHoldemPlayingHall.run(Array("--hands=0"))
    assert(result.isLeft, "expected invalid hands value to fail")
  }

  private def deleteRecursively(path: Path): Unit =
    if Files.exists(path) then
      val stream = Files.walk(path)
      try
        val all = stream.iterator().asScala.toVector.sortBy(_.toString.length).reverse
        all.foreach(Files.deleteIfExists)
      finally
        stream.close()
