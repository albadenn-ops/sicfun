package sicfun.holdem

import munit.FunSuite

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters.*
import scala.concurrent.duration.*

class TexasHoldemPlayingHallTest extends FunSuite:
  override val munitTimeout: Duration = 180.seconds

  test("playing hall runs end-to-end and emits logs") {
    val root = Files.createTempDirectory("playing-hall-test-")
    try
      val out = root.resolve("hall-out")
      val result = TexasHoldemPlayingHall.run(Array(
        "--hands=120",
        "--tableCount=3",
        "--reportEvery=60",
        "--learnEveryHands=40",
        "--learningWindowSamples=200",
        "--seed=11",
        s"--outDir=$out",
        "--villainStyle=tag",
        "--raiseSize=2.5",
        "--bunchingTrials=20",
        "--equityTrials=120",
        "--saveTrainingTsv=true",
        "--saveDdreTrainingTsv=true"
      ))
      assert(result.isRight, s"hall run failed: $result")
      val summary = result.toOption.getOrElse(fail("missing hall summary"))
      assertEquals(summary.handsPlayed, 120)
      assertEquals(summary.tableCount, 3)
      assert(summary.retrains >= 1, s"expected at least one retrain, got ${summary.retrains}")
      assert(Files.exists(out.resolve("hands.tsv")))
      assert(Files.exists(out.resolve("learning.tsv")))
      assert(Files.exists(out.resolve("training-selfplay.tsv")))
      assert(Files.exists(out.resolve("ddre-training-selfplay.tsv")))

      val handRows = Files.readAllLines(out.resolve("hands.tsv"), StandardCharsets.UTF_8).asScala.toVector
      assert(handRows.length > 1, "expected hand log rows")
      val handHeader = handRows.head.split("\t", -1).toVector
      val tableIdIdx = handHeader.indexOf("tableId")
      assert(tableIdIdx >= 0, "hands.tsv missing tableId column")
      val streetsPlayedIdx = handHeader.indexOf("streetsPlayed")
      assert(streetsPlayedIdx >= 0, "hands.tsv missing streetsPlayed column")
      val tableIds = handRows.drop(1).flatMap { row =>
        val fields = row.split("\t", -1).toVector
        fields.lift(tableIdIdx)
      }.toSet
      assertEquals(tableIds, Set("1", "2", "3"))
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

      val ddreRows = Files.readAllLines(out.resolve("ddre-training-selfplay.tsv"), StandardCharsets.UTF_8).asScala.toVector
      assert(ddreRows.length > 1, "expected DDRE training rows")
      val ddreHeader = ddreRows.head.split("\t", -1).toVector
      assert(ddreHeader.contains("bayesPosteriorSparse"), "DDRE TSV missing bayesPosteriorSparse column")
      val streetIdx = ddreHeader.indexOf("street")
      val sawPostflopDdre = ddreRows.drop(1).exists { row =>
        val fields = row.split("\t", -1).toVector
        fields.lift(streetIdx).exists(value => value == "Flop" || value == "Turn" || value == "River")
      }
      assert(sawPostflopDdre, "expected DDRE training data from postflop streets")
    finally
      deleteRecursively(root)
  }

  test("playing hall rejects invalid hands argument") {
    val result = TexasHoldemPlayingHall.run(Array("--hands=0"))
    assert(result.isLeft, "expected invalid hands value to fail")
  }

  test("playing hall rejects invalid tableCount argument") {
    val result = TexasHoldemPlayingHall.run(Array("--tableCount=0"))
    assert(result.isLeft, "expected invalid tableCount value to fail")
  }

  test("playing hall supports gto villain mode") {
    val root = Files.createTempDirectory("playing-hall-gto-test-")
    try
      val out = root.resolve("hall-gto-out")
      val result = TexasHoldemPlayingHall.run(Array(
        "--hands=8",
        "--reportEvery=4",
        "--learnEveryHands=0",
        "--learningWindowSamples=50",
        "--seed=19",
        s"--outDir=$out",
        "--villainStyle=gto",
        "--heroExplorationRate=0.0",
        "--raiseSize=2.5",
        "--bunchingTrials=8",
        "--equityTrials=80",
        "--saveTrainingTsv=false",
        "--saveDdreTrainingTsv=false"
      ))
      assert(result.isRight, s"gto hall run failed: $result")
      val summary = result.toOption.getOrElse(fail("missing hall summary"))
      assertEquals(summary.handsPlayed, 8)
      val handRows = Files.readAllLines(out.resolve("hands.tsv"), StandardCharsets.UTF_8).asScala.toVector
      assert(handRows.length > 1, "expected hand log rows")
      val handHeader = handRows.head.split("\t", -1).toVector
      val archetypeIdx = handHeader.indexOf("archetype")
      assert(archetypeIdx >= 0, "hands.tsv missing archetype column")
      val sawGto =
        handRows.drop(1).exists { row =>
          val fields = row.split("\t", -1).toVector
          fields.lift(archetypeIdx).contains("gto")
        }
      assert(sawGto, "expected at least one hand row with archetype=gto")
    finally
      deleteRecursively(root)
  }

  test("playing hall supports gto vs gto mode") {
    val root = Files.createTempDirectory("playing-hall-gto-vs-gto-test-")
    try
      val out = root.resolve("hall-gto-vs-gto-out")
      val result = TexasHoldemPlayingHall.run(Array(
        "--hands=8",
        "--reportEvery=4",
        "--learnEveryHands=0",
        "--learningWindowSamples=50",
        "--seed=23",
        s"--outDir=$out",
        "--heroStyle=gto",
        "--villainStyle=gto",
        "--heroExplorationRate=0.0",
        "--raiseSize=2.5",
        "--bunchingTrials=8",
        "--equityTrials=80",
        "--saveTrainingTsv=false",
        "--saveDdreTrainingTsv=false"
      ))
      assert(result.isRight, s"gto vs gto hall run failed: $result")
      val summary = result.toOption.getOrElse(fail("missing hall summary"))
      assertEquals(summary.handsPlayed, 8)
      assert(summary.exactGtoCacheTotal > 0, "expected exact GTO cache lookups")
      assert(summary.exactGtoCacheMisses > 0, "expected at least one exact GTO cache miss")
      assert(summary.exactGtoSolvedByProvider.nonEmpty, "expected exact GTO provider telemetry")
      assert(summary.exactGtoServedByProvider.nonEmpty, "expected exact GTO served-provider telemetry")
      val handRows = Files.readAllLines(out.resolve("hands.tsv"), StandardCharsets.UTF_8).asScala.toVector
      assert(handRows.length > 1, "expected hand log rows")
    finally
      deleteRecursively(root)
  }

  private def deleteRecursively(path: Path): Unit =
    if Files.exists(path) then
      val stream = Files.walk(path)
      try
        val all = stream.iterator().asScala.toVector.sortBy(_.toString.length).reverse
        all.foreach(Files.deleteIfExists)
      finally
        stream.close()
