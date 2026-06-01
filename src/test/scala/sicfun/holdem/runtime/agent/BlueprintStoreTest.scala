package sicfun.holdem.runtime.agent

import java.nio.file.Files

class BlueprintStoreTest extends munit.FunSuite:
  test("round-trip: write then load returns same distribution"):
    val tmp = Files.createTempFile("bp", ".blueprint").toFile
    val hdr = BlueprintHeader("SICFBP01", 1, 0L, "specA", 9, 2, 5)
    val dists = Map[Long, Array[Float]](
      100L -> Array(0.2f, 0.2f, 0.2f, 0.2f, 0.2f),
      200L -> Array(0.1f, 0.1f, 0.1f, 0.1f, 0.6f)
    )
    BlueprintStore.write(tmp, hdr, dists)
    val store = BlueprintStore.load(tmp, expectedAbstractionHash = "specA")
    assertEquals(store.header.numInfostates, 2)
    val d100 = store.lookup(100L)
    assert(d100.probs(4) > 0.19f && d100.probs(4) < 0.21f)
    val d200 = store.lookup(200L)
    assert(d200.probs(4) > 0.59f)

  test("rejects wrong abstraction hash"):
    val tmp = Files.createTempFile("bp2", ".blueprint").toFile
    val hdr = BlueprintHeader("SICFBP01", 1, 0L, "specA", 9, 0, 5)
    BlueprintStore.write(tmp, hdr, Map.empty)
    intercept[BlueprintVersionMismatch](
      BlueprintStore.load(tmp, expectedAbstractionHash = "specB")
    )

  test("empty blueprint: lookup returns uniform distribution"):
    val tmp = Files.createTempFile("bp3", ".blueprint").toFile
    val hdr = BlueprintHeader("SICFBP01", 1, 0L, "empty", 9, 0, 5)
    BlueprintStore.write(tmp, hdr, Map.empty)
    val store = BlueprintStore.load(tmp, "empty")
    val uniform = store.lookup(12345L)
    assert(uniform.probs.forall(p => math.abs(p - 0.2f) < 1e-4f),
      s"uniform fallback not uniform: ${uniform.probs.mkString(",")}")
