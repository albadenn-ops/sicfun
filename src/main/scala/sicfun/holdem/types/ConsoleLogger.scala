package sicfun.holdem.types

/** Minimal, dependency-free leveled logging facade for the runtime / validation
  * / gpu paths.
  *
  * PROVENANCE: the branch `feat/web-deploy-hardening` shipped the call sites for
  * this facade (`ConsoleLogger.fromConfig`, `ConsoleLogger(name, level)`,
  * `ConsoleLogger.Level.{Debug,Warn}`, and `log.{info,warn,error,debug}`) in the
  * WIP checkpoint commit `ab633ef` but did NOT include this provider file, so the
  * whole project failed to compile. This file is reconstructed to satisfy exactly
  * the call-site contract AND the wire format pinned by
  * `TexasHoldemPlayingHallTest` ("main emits logger-prefixed summary and progress
  * output", which asserts `[INFO] [texas-holdem-playing-hall] ...`). The level
  * configuration knobs (`sicfun.log.level` / `SICFUN_LOG_LEVEL`) are the ones
  * documented in `GpuRuntimeSupport`'s scaladoc.
  *
  * Wire format (pinned by TexasHoldemPlayingHallTest):
  *   `[<LEVEL>] [<name>] <message>`  -- LEVEL is the upper-cased severity, <name>
  *   is the logger channel passed to `apply` / `fromConfig`. No timestamp (the
  *   `[INFO] [name]` brackets are asserted adjacent).
  *
  * Streams: info / warn / debug -> `scala.Console.out`, error ->
  * `scala.Console.err` (matches the earlier committed `ConsoleLogger` contract in
  * commit `c47470e` AND lets `Console.withOut` test capture see the output, as
  * `TexasHoldemPlayingHallTest.captureStdout` relies on). Message args are
  * by-name so suppressed levels never pay interpolation cost.
  */
trait ConsoleLogger:
  def info(message: => String): Unit
  def warn(message: => String): Unit
  def error(message: => String): Unit
  def debug(message: => String): Unit

object ConsoleLogger:

  /** Severity ordering used for level filtering: a logger configured at level
    * `L` emits a message iff the message's severity is `>= L`. */
  enum Level:
    case Debug, Info, Warn, Error

  /** Named logger at an explicit level, backed by `System.out` / `System.err`. */
  def apply(name: String, level: Level): ConsoleLogger =
    new ConsoleLoggerImpl(name, level)

  /** Named logger whose level is resolved from configuration, falling back to
    * `defaultLevel`. Resolution order (first hit wins):
    *   1. system property `sicfun.log.level.<name>`
    *   2. system property `sicfun.log.level`
    *   3. environment variable `SICFUN_LOG_LEVEL`
    *   4. `defaultLevel`
    * Values match the `Level` enum names case-insensitively; an unrecognised
    * value is ignored (falls through to the next source). */
  def fromConfig(name: String, defaultLevel: Level = Level.Info): ConsoleLogger =
    val configured =
      parseLevel(sys.props.get(s"sicfun.log.level.$name"))
        .orElse(parseLevel(sys.props.get("sicfun.log.level")))
        .orElse(parseLevel(sys.env.get("SICFUN_LOG_LEVEL")))
        .getOrElse(defaultLevel)
    new ConsoleLoggerImpl(name, configured)

  private def parseLevel(raw: Option[String]): Option[Level] =
    raw.map(_.trim).filter(_.nonEmpty).flatMap { value =>
      Level.values.find(_.toString.equalsIgnoreCase(value))
    }

  private final class ConsoleLoggerImpl(name: String, level: Level) extends ConsoleLogger:
    private def enabled(messageLevel: Level): Boolean =
      messageLevel.ordinal >= level.ordinal
    private def line(levelLabel: String, message: String): String =
      s"[$levelLabel] [$name] $message"
    def info(message: => String): Unit =
      if enabled(Level.Info) then Console.out.println(line("INFO", message))
    def warn(message: => String): Unit =
      if enabled(Level.Warn) then Console.out.println(line("WARN", message))
    def error(message: => String): Unit =
      if enabled(Level.Error) then Console.err.println(line("ERROR", message))
    def debug(message: => String): Unit =
      if enabled(Level.Debug) then Console.out.println(line("DEBUG", message))
