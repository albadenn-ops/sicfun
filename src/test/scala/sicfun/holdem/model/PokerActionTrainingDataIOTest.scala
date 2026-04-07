package sicfun.holdem.model
import sicfun.holdem.types.*

import munit.FunSuite

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters.*

/**
 * Tests for [[PokerActionTrainingDataIO]] TSV training data reader.
 *
 * Validates:
 *   - Successful parsing of valid rows with various card notation formats
 *   - Street-board consistency validation (e.g. River with 3 board cards is rejected)
 *   - Missing required column detection
 */
class PokerActionTrainingDataIOTest extends FunSuite:
  test("readTsv parses valid rows") {
    val header = "street\tboard\tpotBefore\ttoCall\tposition\tstackBefore\taction\tholeCards"
    val row1 = "Flop\tTs 9h 8d\t20.0\t10.0\tButton\t200.0\traise:20.0\tAh Kh"
    val row2 = "Flop\tTs9h8d\t20.0\t10.0\tButton\t200.0\tfold\t7c2d"
    val path = writeTempTsv(Vector(header, row1, row2))
    try
      val parsed = PokerActionTrainingDataIO.readTsv(path)
      assertEquals(parsed.length, 2)
      assertEquals(parsed.head._1.street, Street.Flop)
      assertEquals(parsed.head._1.board.size, 3)
      assertEquals(parsed.head._1.pot, 20.0)
      assertEquals(parsed.head._2.toVector.length, 2)
      assertEquals(parsed.head._3, PokerAction.Raise(20.0))
      assertEquals(parsed(1)._3, PokerAction.Fold)
    finally
      Files.deleteIfExists(path)
  }

  test("readTsv validates street-board consistency") {
    val header = "street\tboard\tpotBefore\ttoCall\tposition\tstackBefore\taction\tholeCards"
    val row = "River\tTs 9h 8d\t20.0\t10.0\tButton\t200.0\tcall\tAh Kh"
    val path = writeTempTsv(Vector(header, row))
    try
      intercept[IllegalArgumentException] {
        PokerActionTrainingDataIO.readTsv(path)
      }
    finally
      Files.deleteIfExists(path)
  }

  test("readTsv fails when required columns are missing") {
    val header = "street\tboard\tpotBefore\ttoCall\tposition\taction\tholeCards"
    val row = "Flop\tTs 9h 8d\t20.0\t10.0\tButton\tcall\tAh Kh"
    val path = writeTempTsv(Vector(header, row))
    try
      intercept[IllegalArgumentException] {
        PokerActionTrainingDataIO.readTsv(path)
      }
    finally
      Files.deleteIfExists(path)
  }

  /** Writes lines to a temporary TSV file for test input. */
  private def writeTempTsv(lines: Vector[String]): Path =
    val path = Files.createTempFile("sicfun-training-", ".tsv")
    Files.write(path, lines.asJava, StandardCharsets.UTF_8)
    path
