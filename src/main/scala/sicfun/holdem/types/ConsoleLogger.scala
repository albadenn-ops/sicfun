package sicfun.holdem.types

import java.io.PrintStream

/** Minimal logging facade for runtime + validation paths.
  *
  * No external dependency, no level filtering yet — call sites that today print
  * to stdout/stderr can route through a logger so tests can capture and assert
  * on the output, and so future filtering, formatting, or sink redirection can
  * land in one place rather than across hundreds of `println` calls.
  *
  * info -> stdout (no prefix). warn -> stdout with `[warn]` prefix. error ->
  * stderr with `[error]` prefix. The asymmetry is deliberate: it preserves the
  * existing "info messages look like user output, errors go to stderr" behavior
  * that the runtime/validation code already relies on.
  */
trait ConsoleLogger:
  def info(message: => String): Unit
  def warn(message: => String): Unit
  def error(message: => String): Unit

object ConsoleLogger:
  /** Logger backed by `System.out` / `System.err`. Default for production callers. */
  def stdout(): ConsoleLogger = new PrintStreamLogger(System.out, System.err)

  /** Logger writing to the supplied streams. Used when the caller already owns a target. */
  def writingTo(out: PrintStream, err: PrintStream): ConsoleLogger =
    new PrintStreamLogger(out, err)

  /** Logger built from per-level emit callbacks. Use this when you need full control
    * over message formatting and stream routing -- e.g. the embedded HTTP server
    * wants `[timestamp] [LEVEL] [service] msg` with WARN routed to stderr (operator
    * attention) instead of the default WARN-to-stdout.
    *
    * Each callback receives the raw message string; the callback is responsible
    * for prefixing, timestamping, synchronisation, and stream choice.
    */
  def routes(
      infoEmit: String => Unit,
      warnEmit: String => Unit,
      errorEmit: String => Unit
  ): ConsoleLogger =
    new ConsoleLogger:
      def info(message: => String): Unit = infoEmit(message)
      def warn(message: => String): Unit = warnEmit(message)
      def error(message: => String): Unit = errorEmit(message)

  /** Logger that swallows every message. Useful for tests that only want to silence output. */
  val silent: ConsoleLogger = new ConsoleLogger:
    def info(message: => String): Unit = ()
    def warn(message: => String): Unit = ()
    def error(message: => String): Unit = ()

  /** Test-friendly logger that records every emitted message in order. */
  final class Buffered extends ConsoleLogger:
    private val buf = scala.collection.mutable.ArrayBuffer.empty[(String, String)]
    def info(message: => String): Unit = buf.synchronized { buf += (("info", message)) }
    def warn(message: => String): Unit = buf.synchronized { buf += (("warn", message)) }
    def error(message: => String): Unit = buf.synchronized { buf += (("error", message)) }
    def entries: Vector[(String, String)] = buf.synchronized { buf.toVector }
    def messages: Vector[String] = entries.map(_._2)
    def messagesAt(level: String): Vector[String] =
      entries.collect { case (l, m) if l == level => m }
    def clear(): Unit = buf.synchronized { buf.clear() }

  private final class PrintStreamLogger(out: PrintStream, err: PrintStream) extends ConsoleLogger:
    def info(message: => String): Unit = out.println(message)
    def warn(message: => String): Unit = out.println(s"[warn] $message")
    def error(message: => String): Unit = err.println(s"[error] $message")
