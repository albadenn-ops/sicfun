package sicfun.holdem.io

import munit.FunSuite

import java.nio.file.{Files, Path}

class SignalAuditLogIOTest extends FunSuite:
  private def withTempDir(name: String)(f: Path => Unit): Unit =
    val dir = Files.createTempDirectory(s"sicfun-signal-log-$name-")
    try f(dir)
    finally Files.walk(dir).sorted(java.util.Comparator.reverseOrder()).forEach(Files.deleteIfExists(_))

  private def mkSignal(id: String = "sig-1", handId: String = "h1"): SignalEnvelope =
    SignalEnvelope(
      signalId = id,
      level = SignalLevel.Info,
      payload = SignalPayload(
        signalType = "action-risk",
        signalVersion = "1.0",
        handId = handId,
        playerId = "player-1",
        sequenceInHand = 0L,
        occurredAtEpochMillis = 1_000L,
        generatedAtEpochMillis = 2_000L,
        modelVersionId = "v1",
        modelSchemaVersion = "1",
        modelSource = "unit-test",
        message = "test signal",
        features = Map("potOdds" -> 0.25, "stackToPot" -> 5.0),
        metrics = Map("riskScore" -> 0.3)
      ),
      reconstruction = ReconstructionPath(
        snapshotDirectory = "/tmp/snapshot",
        modelArtifactDirectory = "/tmp/model",
        eventSequenceInHand = 0L
      )
    )

  test("write/read roundtrip preserves a single signal") {
    withTempDir("single") { dir =>
      val path = dir.resolve("audit.tsv")
      val signal = mkSignal()
      SignalAuditLogIO.write(path, Seq(signal))
      val loaded = SignalAuditLogIO.read(path)
      assertEquals(loaded.length, 1)
      assertEquals(loaded.head.signalId, "sig-1")
      assertEquals(loaded.head.level, SignalLevel.Info)
      assertEquals(loaded.head.payload.handId, "h1")
      assertEquals(loaded.head.payload.signalType, "action-risk")
      assertEqualsDouble(loaded.head.payload.features("potOdds"), 0.25, 1e-12)
      assertEqualsDouble(loaded.head.payload.metrics("riskScore"), 0.3, 1e-12)
      assertEquals(loaded.head.reconstruction.snapshotDirectory, "/tmp/snapshot")
    }
  }

  test("write/read roundtrip preserves multiple signals") {
    withTempDir("multi") { dir =>
      val path = dir.resolve("audit.tsv")
      val signals = (1 to 5).map(i => mkSignal(id = s"sig-$i", handId = s"h$i"))
      SignalAuditLogIO.write(path, signals)
      val loaded = SignalAuditLogIO.read(path)
      assertEquals(loaded.length, 5)
      assertEquals(loaded.map(_.signalId), (1 to 5).toVector.map(i => s"sig-$i"))
    }
  }

  test("write creates header-only file for empty signal list") {
    withTempDir("empty") { dir =>
      val path = dir.resolve("audit.tsv")
      SignalAuditLogIO.write(path, Seq.empty)
      val loaded = SignalAuditLogIO.read(path)
      assertEquals(loaded.length, 0)
    }
  }

  test("append creates file with header if absent then adds signal") {
    withTempDir("append-new") { dir =>
      val path = dir.resolve("audit.tsv")
      assert(!Files.exists(path))
      SignalAuditLogIO.append(path, mkSignal(id = "first"))
      val loaded = SignalAuditLogIO.read(path)
      assertEquals(loaded.length, 1)
      assertEquals(loaded.head.signalId, "first")
    }
  }

  test("append adds to existing file") {
    withTempDir("append-existing") { dir =>
      val path = dir.resolve("audit.tsv")
      SignalAuditLogIO.write(path, Seq(mkSignal(id = "s1")))
      SignalAuditLogIO.append(path, mkSignal(id = "s2"))
      val loaded = SignalAuditLogIO.read(path)
      assertEquals(loaded.length, 2)
      assertEquals(loaded.map(_.signalId), Vector("s1", "s2"))
    }
  }

  test("read rejects non-existent file") {
    intercept[IllegalArgumentException] {
      SignalAuditLogIO.read(Path.of("non-existent.tsv"))
    }
  }

  test("write/read roundtrip with Warning and Critical levels") {
    withTempDir("levels") { dir =>
      val path = dir.resolve("audit.tsv")
      val info = mkSignal(id = "info").copy(level = SignalLevel.Info)
      val warn = mkSignal(id = "warn").copy(level = SignalLevel.Warning)
      val crit = mkSignal(id = "crit").copy(level = SignalLevel.Critical)
      SignalAuditLogIO.write(path, Seq(info, warn, crit))
      val loaded = SignalAuditLogIO.read(path)
      assertEquals(loaded.map(_.level), Vector(SignalLevel.Info, SignalLevel.Warning, SignalLevel.Critical))
    }
  }

  test("string path overloads work") {
    withTempDir("string-path") { dir =>
      val pathStr = dir.resolve("audit.tsv").toString
      SignalAuditLogIO.write(pathStr, Seq(mkSignal()))
      val loaded = SignalAuditLogIO.read(pathStr)
      assertEquals(loaded.length, 1)
    }
  }
