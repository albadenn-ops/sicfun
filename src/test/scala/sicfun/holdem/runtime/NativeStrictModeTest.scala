package sicfun.holdem.runtime

import java.io.File
import java.nio.file.Files

class NativeStrictModeTest extends munit.FunSuite:

  test("strict: missing DLL → Left(violation) listing the missing name"):
    val tmp = Files.createTempDirectory("sicfun-empty-native").toFile
    val required = Vector(NativeStrictMode.Library("sicfun_nonexistent", Vector.empty))
    val result = NativeStrictMode.verify(tmp, required, strict = true)
    assert(result.isLeft)
    result.left.foreach { v =>
      assert(v.getMessage.contains("sicfun_nonexistent"), v.getMessage)
    }

  test("strict: missing multiple DLLs → violation message enumerates all"):
    val tmp = Files.createTempDirectory("sicfun-empty-native-2").toFile
    val required = Vector(
      NativeStrictMode.Library("lib_a", Vector.empty),
      NativeStrictMode.Library("lib_b", Vector.empty)
    )
    val result = NativeStrictMode.verify(tmp, required, strict = true)
    assert(result.isLeft)
    result.left.foreach { v =>
      assert(v.getMessage.contains("lib_a") && v.getMessage.contains("lib_b"),
        s"expected both in message, got: ${v.getMessage}")
    }

  test("non-strict: missing DLL → Right(contaminated = true) with stderr warning"):
    val tmp = Files.createTempDirectory("sicfun-empty-native-3").toFile
    val required = Vector(NativeStrictMode.Library("sicfun_nonexistent", Vector.empty))
    val result = NativeStrictMode.verify(tmp, required, strict = false)
    assert(result.isRight)
    assertEquals(result.toOption.get.contaminated, true)

  test("strict: present DLL + no stale sources → Right(contaminated = false)"):
    val tmp = Files.createTempDirectory("sicfun-fake-build").toFile
    val dllFile = new File(tmp, "lib_fake.dll")
    Files.write(dllFile.toPath, Array[Byte](0))
    val required = Vector(NativeStrictMode.Library("lib_fake", Vector.empty))
    val result = NativeStrictMode.verify(tmp, required, strict = true)
    assert(result.isRight, s"expected Right, got $result")
    assertEquals(result.toOption.get.contaminated, false)

  test("strict: DLL older than source → Left(stale)"):
    val tmp = Files.createTempDirectory("sicfun-stale").toFile
    val dllFile = new File(tmp, "lib_fake.dll")
    Files.write(dllFile.toPath, Array[Byte](0))
    dllFile.setLastModified(1_000_000L) // 1970-ish
    val srcTmp = Files.createTempFile("sicfun-src", ".cpp").toFile
    srcTmp.setLastModified(System.currentTimeMillis()) // now
    val required = Vector(NativeStrictMode.Library("lib_fake", Vector(srcTmp.getPath)))
    val result = NativeStrictMode.verify(tmp, required, strict = true)
    assert(result.isLeft, s"expected stale violation, got $result")
    result.left.foreach { v =>
      assert(v.getMessage.contains("stale") || v.getMessage.contains("older"),
        v.getMessage)
    }
