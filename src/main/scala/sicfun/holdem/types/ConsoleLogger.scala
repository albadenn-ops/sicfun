package sicfun.holdem.types

import java.io.PrintStream
import java.time.Instant
import java.util.Locale
import scala.Console

/** Minimal stdout/stderr-backed logger for runtime and tooling code.
  *
  * Keeps the repo's zero-dependency posture while providing level filtering
  * and consistent prefixes for long-running command-line flows.
  */
final class ConsoleLogger private (
    name: String,
    minLevel: () => ConsoleLogger.Level
):
  def debug(message: => String): Unit =
    log(ConsoleLogger.Level.Debug, message)

  def info(message: => String): Unit =
    log(ConsoleLogger.Level.Info, message)

  def warn(message: => String): Unit =
    log(ConsoleLogger.Level.Warn, message)

  def error(message: => String): Unit =
    log(ConsoleLogger.Level.Error, message)

  private def log(level: ConsoleLogger.Level, message: => String): Unit =
    if level.priority >= minLevel().priority then
      val rendered = message
      val prefix = s"[${Instant.now()}] [${level.label}] [$name]"
      val stream = level.stream
      stream.synchronized {
        rendered.linesIterator.foreach { line =>
          stream.print(s"$prefix $line${System.lineSeparator()}")
        }
      }

object ConsoleLogger:
  enum Level(val label: String, val priority: Int):
    case Debug extends Level("DEBUG", 10)
    case Info extends Level("INFO", 20)
    case Warn extends Level("WARN", 30)
    case Error extends Level("ERROR", 40)

    def stream: PrintStream =
      this match
        case Debug | Info => Console.out
        case Warn | Error => Console.err

  private val DefaultProperty = "sicfun.log.level"
  private val DefaultEnv = "SICFUN_LOG_LEVEL"

  def apply(name: String, minLevel: Level = Level.Info): ConsoleLogger =
    new ConsoleLogger(name, () => minLevel)

  /** Builds a logger whose configured level is resolved on each log call.
    *
    * That keeps long-lived singleton loggers responsive to scoped test
    * overrides and late `-D` updates while preserving lazy message evaluation.
    */
  def fromConfig(
      name: String,
      property: String = DefaultProperty,
      env: String = DefaultEnv,
      defaultLevel: Level = Level.Info
  ): ConsoleLogger =
    new ConsoleLogger(name, () => configuredLevel(property, env).getOrElse(defaultLevel))

  def configuredLevel(
      property: String = DefaultProperty,
      env: String = DefaultEnv
  ): Option[Level] =
    ScopedRuntimeProperties.get(property) match
      case Some(Some(value)) => parseLevel(value)
      case Some(None) => None
      case None =>
        sys.props.get(property)
          .orElse(sys.env.get(env))
          .flatMap(parseLevel)

  private def parseLevel(raw: String): Option[Level] =
    raw.trim.toLowerCase(Locale.ROOT) match
      case "debug" => Some(Level.Debug)
      case "info"  => Some(Level.Info)
      case "warn"  => Some(Level.Warn)
      case "error" => Some(Level.Error)
      case _       => None
