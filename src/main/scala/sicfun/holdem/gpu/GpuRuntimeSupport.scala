package sicfun.holdem.gpu
import sicfun.holdem.*
import sicfun.holdem.types.ScopedRuntimeProperties

import java.io.File
import java.util.Locale

/** Shared helpers for GPU/OpenCL runtime configuration and native library loading.
  *
  * This object centralises the cross-cutting concerns that every native runtime
  * wrapper in the `gpu` package needs:
  *
  *  - '''Configuration resolution''': system properties (`-Dsicfun.*`) are checked
  *    first, with environment variables (`sicfun_*`) as a fallback, and a
  *    thread-local `ScopedRuntimeProperties` overlay takes priority over both
  *    (used extensively in tests).
  *
  *  - '''Native library loading''': [[loadNativeLibrary]] tries an explicit file
  *    path first, then `System.loadLibrary`, and finally probes platform-specific
  *    fallback candidates under `src/main/native/build/` for local development.
  *
  *  - '''Status code translation''': [[describeNativeStatus]] and
  *    [[describeOpenCLStatus]] map integer JNI return codes to human-readable
  *    descriptions for CUDA, OpenCL, and common JNI validation errors.
  *
  *  - '''Logging''': a lightweight facility (`log` / `warn`) controlled by the
  *    `sicfun.verbose` system property or `sicfun_VERBOSE` env var. Info-level
  *    output is suppressed by default; warnings always go to stderr.
  *
  * Design decisions:
  *  - All methods are pure functions or side-effect-free lookups, making this
  *    object safe to call from any thread.
  *  - Truthy parsing (`parseTruthy`) accepts "1", "true", "yes", and "on" to
  *    accommodate both CI scripts and human operators.
  */
private[holdem] object GpuRuntimeSupport:
  private val VerboseProperty = "sicfun.verbose"
  private val VerboseEnv = "sicfun_VERBOSE"

  /** Returns true when info-level output is enabled. */
  private[holdem] def verbose: Boolean =
    resolveNonEmpty(VerboseProperty, VerboseEnv)
      .exists(parseTruthy)

  /** Info-level log: only prints when `sicfun.verbose=true`. */
  private[holdem] def log(msg: => String): Unit =
    if verbose then println(msg)

  /** Warning-level log: always prints to stderr. */
  private[holdem] def warn(msg: => String): Unit =
    System.err.println(msg)

  /** Resolves a configuration value by checking (in priority order):
    *  1. `ScopedRuntimeProperties` thread-local overlay (set by tests)
    *  2. JVM system property (`-Dproperty=value`)
    *  3. OS environment variable
    *
    * Returns `None` when the value is absent, null, or blank after trimming.
    *
    * @param property JVM system property name (e.g. `"sicfun.gpu.provider"`)
    * @param env      environment variable name  (e.g. `"sicfun_GPU_PROVIDER"`)
    */
  def resolveNonEmpty(property: String, env: String): Option[String] =
    ScopedRuntimeProperties.get(property) match
      // Thread-local scope explicitly provides a value -- use it.
      case Some(Some(value)) =>
        Option(value).map(_.trim).filter(_.nonEmpty)
      // Thread-local scope explicitly sets the key to None -- treat as "not configured".
      case Some(None) =>
        None
      // No thread-local override -- fall back to global system property, then env var.
      case None =>
        sys.props
          .get(property)
          .orElse(sys.env.get(env))
          .map(_.trim)
          .filter(_.nonEmpty)

  /** Same as [[resolveNonEmpty]] but lower-cases the result using the ROOT
    * locale, so provider/engine names can be compared case-insensitively.
    */
  def resolveNonEmptyLower(property: String, env: String): Option[String] =
    resolveNonEmpty(property, env).map(_.toLowerCase(Locale.ROOT))

  /** Resolves a file path from config, falling back to `defaultPath`.
    * Used for auto-tune cache files and similar on-disk configuration.
    */
  def resolveFile(property: String, env: String, defaultPath: String): File =
    new File(resolveNonEmpty(property, env).getOrElse(defaultPath))

  /** Returns `true` when the given property/env pair resolves to a non-empty value.
    * Used to detect explicit user overrides that should suppress auto-tuning.
    */
  def isConfigured(property: String, env: String): Boolean =
    resolveNonEmpty(property, env).nonEmpty

  /** Parses a human-friendly boolean value.
    * Accepts "1", "true", "yes", "on" (case-insensitive) as truthy; everything else is falsy.
    */
  def parseTruthy(raw: String): Boolean =
    raw.trim.toLowerCase(Locale.ROOT) match
      case "1" | "true" | "yes" | "on" => true
      case _ => false

  /** Parses a string as a strictly positive integer, returning `None` on failure or non-positive values. */
  def parsePositiveIntOpt(raw: String): Option[Int] =
    Option(raw)
      .map(_.trim)
      .filter(_.nonEmpty)
      .flatMap(text => scala.util.Try(text.toInt).toOption)
      .filter(_ > 0)

  /** Parses a string as a non-negative integer (>= 0), returning `None` on failure or negative values. */
  def parseNonNegativeIntOpt(raw: String): Option[Int] =
    Option(raw)
      .map(_.trim)
      .filter(_.nonEmpty)
      .flatMap(text => scala.util.Try(text.toInt).toOption)
      .filter(_ >= 0)

  /** Attempts to load a native shared library using the following resolution order:
    *
    *  1. '''Explicit path''' (`pathProperty` / `pathEnv`) -- calls `System.load(absolutePath)`.
    *  2. '''Library name''' (`libProperty` / `libEnv`, defaulting to `defaultLib`) -- calls
    *     `System.loadLibrary(name)` which searches `java.library.path`.
    *  3. '''Local build fallback''' -- looks for platform-specific filenames
    *     (e.g. `sicfun_gpu_kernel.dll` on Windows) under `src/main/native/build/`.
    *
    * @return `Right(source)` describing how the library was loaded, or `Left(reason)` on failure.
    */
  def loadNativeLibrary(
      pathProperty: String,
      pathEnv: String,
      libProperty: String,
      libEnv: String,
      defaultLib: String,
      label: String
  ): Either[String, String] =
    val pathOpt = resolveNonEmpty(pathProperty, pathEnv)
    val libName = resolveNonEmpty(libProperty, libEnv).getOrElse(defaultLib)
    pathOpt match
      case Some(path) =>
        try
          System.load(path)
          Right(s"path=$path")
        catch
          case ex: Throwable =>
            Left(s"failed to load $label '$path': ${ex.getMessage}")
      case None =>
        try
          System.loadLibrary(libName)
          Right(s"library=$libName")
        catch
          case ex: Throwable =>
            val fallbackCandidates = localNativeFallbackCandidates(libName)
            tryLoadFirstExistingPath(fallbackCandidates) match
              case Right(source) =>
                Right(source)
              case Left(fallbackReason) =>
                Left(
                  s"failed to load $label '$libName': ${ex.getMessage}; $fallbackReason"
                )

  /** Builds a list of candidate file paths under `src/main/native/build/`
    * for the given library name, using platform-specific naming conventions.
    */
  def localNativeFallbackCandidates(libName: String): Vector[File] =
    val userDir = new File(System.getProperty("user.dir", "."))
    val buildDir = new File(userDir, "src/main/native/build")
    platformNativeFileNames(libName)
      .map(name => new File(buildDir, name))
      .distinct

  /** Returns the expected shared-library filenames for `libName` on the current OS.
    * Windows: `name.dll`; macOS: `libname.dylib`; Linux: `libname.so`.
    */
  def platformNativeFileNames(libName: String): Vector[String] =
    val osName = System.getProperty("os.name", "").toLowerCase
    if osName.contains("win") then
      Vector(s"$libName.dll")
    else if osName.contains("mac") then
      Vector(s"lib$libName.dylib", s"$libName.dylib")
    else
      Vector(s"lib$libName.so", s"$libName.so")

  /** Tries to `System.load` the first existing file from `candidates`.
    * Returns `Right(absolutePath)` on success, or `Left(summary)` listing all attempted paths.
    */
  def tryLoadFirstExistingPath(candidates: Vector[File]): Either[String, String] =
    @annotation.tailrec
    def loop(remaining: List[File], errors: List[String]): Either[String, String] =
      remaining match
        case Nil =>
          Left(s"also tried local native paths: ${errors.reverse.mkString("; ")}")
        case candidate :: tail =>
          val absPath = candidate.getAbsolutePath
          if candidate.isFile then
            try
              System.load(absPath)
              Right(s"path=$absPath")
            catch
              case ex: Throwable =>
                loop(tail, s"$absPath (${ex.getMessage})" :: errors)
          else
            loop(tail, s"$absPath (missing)" :: errors)

    loop(candidates.toList, Nil)

  /** Translates a native JNI status code into a human-readable string.
    *
    * Status code ranges:
    *  - 100-127: JNI input validation errors (null arrays, mismatched lengths, invalid ids)
    *  - 130-138: CUDA runtime errors (device unavailable, allocation failure, kernel timeout)
    *
    * These codes are defined in the C++ JNI implementation (`sicfun_gpu_kernel`).
    */
  def describeNativeStatus(status: Int): String =
    val detail =
      status match
        case 100 => "null JNI input array"
        case 101 => "JNI input arrays have mismatched lengths"
        case 102 => "failed reading JNI input arrays"
        case 111 => "invalid compute mode code"
        case 112 => "invalid CSR range layout"
        case 124 => "failed writing JNI output arrays"
        case 125 => "invalid hole-card id"
        case 126 => "invalid monte-carlo trial count"
        case 127 => "overlapping hole cards in matchup"
        case 130 => "CUDA device/runtime unavailable"
        case 131 => "CUDA device allocation failed"
        case 132 => "CUDA host-to-device transfer failed"
        case 133 => "CUDA kernel launch failed"
        case 134 =>
          "CUDA synchronize failed (likely Windows WDDM/TDR timeout for long kernels; reduce chunk size or trials)"
        case 135 => "CUDA device-to-host transfer failed"
        case 136 => "CUDA lookup upload failed"
        case 137 => "CUDA kernel timed out (Windows WDDM/TDR watchdog)"
        case 138 => "invalid CUDA device index"
        case _ => "unknown native status"
    s"native GPU kernel returned status=$status ($detail)"

  /** Translates an OpenCL native status code into a human-readable string.
    *
    * Status codes 100-127 share meaning with [[describeNativeStatus]] (JNI validation).
    * Codes 200-206 are OpenCL-specific runtime errors defined in `sicfun_opencl_kernel`.
    */
  def describeOpenCLStatus(status: Int): String =
    val detail =
      status match
        case s if s >= 100 && s <= 127 =>
          describeNativeStatus(s).stripPrefix("native GPU kernel returned ")
        case 200 => "OpenCL runtime not available (OpenCL.dll not found)"
        case 201 => "no OpenCL GPU devices found"
        case 202 => "OpenCL kernel compilation failed"
        case 203 => "OpenCL buffer allocation failed"
        case 204 => "OpenCL kernel execution failed"
        case 205 => "OpenCL result read-back failed"
        case 206 => "invalid OpenCL device index"
        case _ => s"unknown OpenCL status ($status)"
    s"OpenCL kernel returned status=$status ($detail)"
