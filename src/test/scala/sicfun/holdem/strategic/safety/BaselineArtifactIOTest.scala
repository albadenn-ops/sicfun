package sicfun.holdem.strategic.safety

import munit.FunSuite
import java.nio.file.Files
import sicfun.holdem.types.{PokerAction, Street}
import sicfun.holdem.strategic.types.StrategicClass

class BaselineArtifactIOTest extends FunSuite:
  test("save then load round-trips counts and metadata exactly"):
    val counts = Map[(StrategicClass, String, Street, PokerAction.Category), Long](
      (StrategicClass.Value, "Unpaired-Rainbow-AceHigh", Street.Flop, PokerAction.Category.Raise) -> 42L,
      (StrategicClass.Bluff, "Paired-TwoTone-Broadway", Street.Turn, PokerAction.Category.Fold)   -> 7L
    )
    val meta = BaselineMetadata(
      formatVersion = BaselineArtifactIO.FormatVersion,
      bucketSchemeVersion = "v1",
      corpusId = "test-corpus",
      handCount = 100,
      showdownDecisionCount = 49,
      recommendedMinCount = 30,
      recommendedSmoothingAlpha = 1.0,
      calibrationEpochMillis = 1730000000000L
    )
    val artifact = BaselineArtifact(counts, meta)
    val dir = Files.createTempDirectory("baseline-artifact-test")
    try
      BaselineArtifactIO.save(dir, artifact)
      val loaded = BaselineArtifactIO.load(dir)
      assertEquals(loaded.counts, counts)
      assertEquals(loaded.metadata, meta)
    finally
      Files.walk(dir).sorted(java.util.Comparator.reverseOrder()).forEach(p => Files.deleteIfExists(p))

  test("load rejects an unsupported format version"):
    val dir = Files.createTempDirectory("baseline-artifact-bad")
    try
      Files.writeString(dir.resolve("metadata.properties"), "format.version=999\n")
      Files.writeString(dir.resolve("baseline-counts.tsv"), "class\tboardBucket\tstreet\taction\tcount\n")
      intercept[IllegalArgumentException](BaselineArtifactIO.load(dir))
    finally
      Files.walk(dir).sorted(java.util.Comparator.reverseOrder()).forEach(p => Files.deleteIfExists(p))
