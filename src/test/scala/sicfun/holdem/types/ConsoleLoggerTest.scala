package sicfun.holdem.types

import munit.FunSuite

import java.io.{ByteArrayOutputStream, PrintStream}
import java.nio.charset.StandardCharsets

class ConsoleLoggerTest extends FunSuite:

  test("stdout logger writes info to out without a prefix") {
    val out = new ByteArrayOutputStream()
    val err = new ByteArrayOutputStream()
    val logger = ConsoleLogger.writingTo(
      new PrintStream(out, true, StandardCharsets.UTF_8),
      new PrintStream(err, true, StandardCharsets.UTF_8)
    )
    logger.info("hello world")
    assertEquals(out.toString(StandardCharsets.UTF_8).trim, "hello world")
    assertEquals(err.toString(StandardCharsets.UTF_8), "")
  }

  test("stdout logger writes warn to out with [warn] prefix") {
    val out = new ByteArrayOutputStream()
    val err = new ByteArrayOutputStream()
    val logger = ConsoleLogger.writingTo(
      new PrintStream(out, true, StandardCharsets.UTF_8),
      new PrintStream(err, true, StandardCharsets.UTF_8)
    )
    logger.warn("careful")
    assertEquals(out.toString(StandardCharsets.UTF_8).trim, "[warn] careful")
    assertEquals(err.toString(StandardCharsets.UTF_8), "")
  }

  test("stdout logger writes error to err with [error] prefix") {
    val out = new ByteArrayOutputStream()
    val err = new ByteArrayOutputStream()
    val logger = ConsoleLogger.writingTo(
      new PrintStream(out, true, StandardCharsets.UTF_8),
      new PrintStream(err, true, StandardCharsets.UTF_8)
    )
    logger.error("boom")
    assertEquals(out.toString(StandardCharsets.UTF_8), "")
    assertEquals(err.toString(StandardCharsets.UTF_8).trim, "[error] boom")
  }

  test("Buffered preserves message order across levels") {
    val buf = new ConsoleLogger.Buffered
    buf.info("a")
    buf.warn("b")
    buf.error("c")
    buf.info("d")
    assertEquals(
      buf.entries,
      Vector("info" -> "a", "warn" -> "b", "error" -> "c", "info" -> "d")
    )
    assertEquals(buf.messages, Vector("a", "b", "c", "d"))
    assertEquals(buf.messagesAt("info"), Vector("a", "d"))
    assertEquals(buf.messagesAt("warn"), Vector("b"))
    assertEquals(buf.messagesAt("error"), Vector("c"))
  }

  test("Buffered.clear empties the buffer") {
    val buf = new ConsoleLogger.Buffered
    buf.info("a")
    buf.warn("b")
    buf.clear()
    assertEquals(buf.entries, Vector.empty)
    buf.error("c")
    assertEquals(buf.entries, Vector("error" -> "c"))
  }

  test("silent logger drops every message") {
    val logger = ConsoleLogger.silent
    logger.info("x")
    logger.warn("y")
    logger.error("z")
    // No assertion needed; the contract is "no output and no exception".
  }

  test("info messages are evaluated lazily (by-name argument)") {
    val logger = ConsoleLogger.silent
    var evaluated = 0
    logger.info { evaluated += 1; "noisy" }
    // silent never emits, so the by-name body is never forced.
    assertEquals(evaluated, 0)
  }

  test("routes factory dispatches each level to its dedicated callback") {
    val infoMessages = scala.collection.mutable.ArrayBuffer.empty[String]
    val warnMessages = scala.collection.mutable.ArrayBuffer.empty[String]
    val errorMessages = scala.collection.mutable.ArrayBuffer.empty[String]
    val logger = ConsoleLogger.routes(
      infoEmit = infoMessages += _,
      warnEmit = warnMessages += _,
      errorEmit = errorMessages += _
    )
    logger.info("a")
    logger.warn("b")
    logger.error("c")
    logger.info("d")
    assertEquals(infoMessages.toVector, Vector("a", "d"))
    assertEquals(warnMessages.toVector, Vector("b"))
    assertEquals(errorMessages.toVector, Vector("c"))
  }

  test("routes factory passes the raw message (no prefix added)") {
    var captured = ""
    val logger = ConsoleLogger.routes(
      infoEmit = m => captured = m,
      warnEmit = _ => (),
      errorEmit = _ => ()
    )
    logger.info("plain message")
    assertEquals(captured, "plain message", "routes must not inject any prefix")
  }
