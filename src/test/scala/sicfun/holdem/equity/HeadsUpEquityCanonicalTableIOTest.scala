package sicfun.holdem.equity
import sicfun.holdem.types.*

import munit.FunSuite
import scala.concurrent.duration.*
import scala.util.Random

import java.nio.file.{Files, Path}

/**
  * Roundtrip tests for the canonical heads-up equity table on-disk binary format.
  *
  * Pins:
  *   - write -> read returns an equal map of canonical entries
  *   - writeFromBatch (raw key/result arrays) produces the same file content as write
  *   - readMeta surfaces every field the writer recorded
  *   - meta.canonical=false is rejected at write time
  *
  * Equity tables are correctness-critical resources (heads-up-equity-canonical.bin powers every
  * downstream equity consumer); a silent IO regression would poison every caller. These tests
  * are the missing fence around tablegen/'s output format.
  */
class HeadsUpEquityCanonicalTableIOTest extends FunSuite:
  override val munitTimeout: Duration = 90.seconds

  private val PreflopBackendProperty = "sicfun.holdem.preflopEquityBackend"
  private val MaxMatchups = 30L

  private def buildSmallTable(): HeadsUpEquityCanonicalTable =
    TestSystemPropertyScope.withSystemProperties(
      Vector(PreflopBackendProperty -> Some("cpu"))
    ) {
      HeadsUpEquityCanonicalTable.buildAll(
        mode = HeadsUpEquityTable.Mode.MonteCarlo(8),
        rng = new Random(7L),
        maxMatchups = MaxMatchups,
        parallelism = 1
      )
    }

  private def metaFor(table: HeadsUpEquityCanonicalTable, mode: String): HeadsUpEquityTableMeta =
    HeadsUpEquityTableMeta(
      formatVersion = HeadsUpEquityTableFormat.Version,
      mode = mode,
      trials = 8,
      seed = 7L,
      maxMatchups = MaxMatchups,
      totalMatchups = HeadsUpEquityCanonicalTable.totalCanonicalKeys.toLong,
      count = table.size,
      canonical = true,
      createdAtMillis = 1_700_000_000_000L
    )

  private def withTempPath[A](prefix: String)(body: Path => A): A =
    val path = Files.createTempFile(prefix, ".bin")
    try body(path)
    finally Files.deleteIfExists(path)

  test("write -> read roundtrips every canonical entry") {
    val table = buildSmallTable()
    val meta = metaFor(table, mode = "mc")
    withTempPath("canonical-io-write-") { path =>
      HeadsUpEquityCanonicalTableIO.write(path.toString, table, meta)
      val (read, readMeta) = HeadsUpEquityCanonicalTableIO.readWithMeta(path.toString)
      assertEquals(read.values, table.values, "entry map must roundtrip")
      assertEquals(readMeta, meta, "meta must roundtrip exactly")
    }
  }

  test("writeFromBatch produces the same on-disk content as write") {
    val table = buildSmallTable()
    val meta = metaFor(table, mode = "mc")
    val batch = HeadsUpEquityCanonicalTable.selectCanonicalBatch(MaxMatchups)
    // Reconstruct results aligned with batch.keys by inverting the on-write flip
    val results = batch.keys.map { key =>
      val storedAsCanonical = table.values(key.value.raw)
      // Storage is canonical-perspective; writeFromBatch will re-flip, so undo it for input symmetry
      HeadsUpEquityCanonicalTable.flipIfNeeded(storedAsCanonical, key.flipped)
    }
    val batchMeta = meta.copy(count = batch.keys.length)
    withTempPath("canonical-io-batch-") { batchPath =>
      withTempPath("canonical-io-direct-") { directPath =>
        HeadsUpEquityCanonicalTableIO.writeFromBatch(
          batchPath.toString,
          batch.keys,
          results,
          batchMeta
        )
        HeadsUpEquityCanonicalTableIO.write(directPath.toString, table, batchMeta)
        val batchTable = HeadsUpEquityCanonicalTableIO.read(batchPath.toString)
        val directTable = HeadsUpEquityCanonicalTableIO.read(directPath.toString)
        assertEquals(
          batchTable.values,
          directTable.values,
          "writeFromBatch and write must produce identical canonical entries"
        )
        val batchBytes = Files.readAllBytes(batchPath)
        val directBytes = Files.readAllBytes(directPath)
        assertEquals(
          batchBytes.length,
          directBytes.length,
          "byte length must match between write paths"
        )
      }
    }
  }

  test("readMeta surfaces every field without loading entries") {
    val table = buildSmallTable()
    val meta = metaFor(table, mode = "mc")
    withTempPath("canonical-io-meta-") { path =>
      HeadsUpEquityCanonicalTableIO.write(path.toString, table, meta)
      val readMeta = HeadsUpEquityCanonicalTableIO.readMeta(path.toString)
      assertEquals(readMeta, meta)
    }
  }

  test("write rejects meta.canonical=false") {
    val table = buildSmallTable()
    val badMeta = metaFor(table, mode = "mc").copy(canonical = false)
    withTempPath("canonical-io-bad-") { path =>
      val ex = intercept[IllegalArgumentException] {
        HeadsUpEquityCanonicalTableIO.write(path.toString, table, badMeta)
      }
      assert(
        ex.getMessage.contains("canonical"),
        s"unexpected message: ${ex.getMessage}"
      )
    }
  }

  test("writeFromBatch rejects meta.count != keys.length") {
    val table = buildSmallTable()
    val batch = HeadsUpEquityCanonicalTable.selectCanonicalBatch(MaxMatchups)
    val results = new Array[EquityResultWithError](batch.keys.length)
    val mismatchedMeta = metaFor(table, mode = "mc").copy(count = batch.keys.length + 1)
    withTempPath("canonical-io-mismatch-") { path =>
      intercept[IllegalArgumentException] {
        HeadsUpEquityCanonicalTableIO.writeFromBatch(
          path.toString,
          batch.keys,
          results,
          mismatchedMeta
        )
      }
    }
  }
