package sicfun.holdem.types

import munit.FunSuite

import java.io.{ByteArrayOutputStream, PrintStream}

class ConsoleLoggerTest extends FunSuite:
  private def captureOut(body: => Unit): String =
    val bytes = new ByteArrayOutputStream()
    val out = new PrintStream(bytes, true, "UTF-8")
    try
      Console.withOut(out)(body)
      out.flush()
      bytes.toString("UTF-8")
    finally out.close()

  private def captureErr(body: => Unit): String =
    val bytes = new ByteArrayOutputStream()
    val err = new PrintStream(bytes, true, "UTF-8")
    try
      Console.withErr(err)(body)
      err.flush()
      bytes.toString("UTF-8")
    finally err.close()

  test("info logging respects Console.withOut and prefixes every line"):
    val rendered = captureOut {
      val logger = ConsoleLogger("console-logger-test")
      logger.info("first line\nsecond line")
    }

    val lines = rendered.linesIterator.toVector
    assertEquals(lines.length, 2)
    assert(lines.forall(_.contains("[INFO] [console-logger-test]")))
    assert(lines.head.endsWith("first line"))
    assert(lines(1).endsWith("second line"))

  test("warn logging respects Console.withErr"):
    val rendered = captureErr {
      val logger = ConsoleLogger("console-logger-test")
      logger.warn("warn line")
    }

    val lines = rendered.linesIterator.toVector
    assertEquals(lines.length, 1)
    assert(lines.head.contains("[WARN] [console-logger-test]"))
    assert(lines.head.endsWith("warn line"))

  test("debug logging is filtered by default"):
    val rendered = captureOut {
      val logger = ConsoleLogger("console-logger-test")
      logger.debug("hidden")
      logger.info("shown")
    }

    val lines = rendered.linesIterator.toVector
    assertEquals(lines.length, 1)
    assert(lines.head.contains("[INFO] [console-logger-test]"))
    assert(lines.head.endsWith("shown"))

  test("fromConfig resolves scoped level at log time"):
    val property = s"sicfun.test.consoleLogger.level.${System.nanoTime()}"
    val logger = ConsoleLogger.fromConfig(
      "console-logger-test",
      property = property,
      env = "SICFUN_TEST_CONSOLE_LOGGER_LEVEL_UNSET"
    )

    val hidden = captureOut {
      logger.debug("hidden")
    }
    assertEquals(hidden, "")

    val shown = TestSystemPropertyScope.withSystemProperties(Seq(property -> Some("debug"))) {
      captureOut {
        logger.debug("shown")
      }
    }
    val lines = shown.linesIterator.toVector
    assertEquals(lines.length, 1)
    assert(lines.head.contains("[DEBUG] [console-logger-test]"))
    assert(lines.head.endsWith("shown"))

  test("fromConfig scoped clear falls back to the default level"):
    val property = s"sicfun.test.consoleLogger.clear.${System.nanoTime()}"
    val logger = ConsoleLogger.fromConfig(
      "console-logger-test",
      property = property,
      env = "SICFUN_TEST_CONSOLE_LOGGER_CLEAR_UNSET",
      defaultLevel = ConsoleLogger.Level.Warn
    )

    val rendered = TestSystemPropertyScope.withSystemProperties(Seq(property -> None)) {
      captureOut {
        logger.info("hidden")
      }
    }
    assertEquals(rendered, "")
