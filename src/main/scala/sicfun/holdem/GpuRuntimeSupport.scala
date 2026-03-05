package sicfun.holdem

import java.io.File
import java.util.Locale

/** Shared helpers for GPU/OpenCL runtime configuration and native loading.
  *
  * Includes a lightweight logging facility (`log` / `warn`) controlled by:
  *   - `sicfun.verbose` system property or `sicfun_VERBOSE` env var: enables info-level output
  *   - Warnings always go to stderr regardless of verbosity.
  */
private[holdem] object GpuRuntimeSupport:
  private val VerboseProperty = "sicfun.verbose"
  private val VerboseEnv = "sicfun_VERBOSE"

  /** Returns true when info-level output is enabled. */
  private[holdem] def verbose: Boolean =
    sys.props.get(VerboseProperty).orElse(sys.env.get(VerboseEnv))
      .exists(parseTruthy)

  /** Info-level log: only prints when `sicfun.verbose=true`. */
  private[holdem] def log(msg: => String): Unit =
    if verbose then println(msg)

  /** Warning-level log: always prints to stderr. */
  private[holdem] def warn(msg: => String): Unit =
    System.err.println(msg)

  def resolveNonEmpty(property: String, env: String): Option[String] =
    sys.props
      .get(property)
      .orElse(sys.env.get(env))
      .map(_.trim)
      .filter(_.nonEmpty)

  def resolveNonEmptyLower(property: String, env: String): Option[String] =
    resolveNonEmpty(property, env).map(_.toLowerCase(Locale.ROOT))

  def resolveFile(property: String, env: String, defaultPath: String): File =
    new File(resolveNonEmpty(property, env).getOrElse(defaultPath))

  def isConfigured(property: String, env: String): Boolean =
    resolveNonEmpty(property, env).nonEmpty

  def parseTruthy(raw: String): Boolean =
    raw.trim.toLowerCase(Locale.ROOT) match
      case "1" | "true" | "yes" | "on" => true
      case _ => false

  def parsePositiveIntOpt(raw: String): Option[Int] =
    Option(raw)
      .map(_.trim)
      .filter(_.nonEmpty)
      .flatMap(text => scala.util.Try(text.toInt).toOption)
      .filter(_ > 0)

  def parseNonNegativeIntOpt(raw: String): Option[Int] =
    Option(raw)
      .map(_.trim)
      .filter(_.nonEmpty)
      .flatMap(text => scala.util.Try(text.toInt).toOption)
      .filter(_ >= 0)

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

  def localNativeFallbackCandidates(libName: String): Vector[File] =
    val userDir = new File(System.getProperty("user.dir", "."))
    val buildDir = new File(userDir, "src/main/native/build")
    platformNativeFileNames(libName)
      .map(name => new File(buildDir, name))
      .distinct

  def platformNativeFileNames(libName: String): Vector[String] =
    val osName = System.getProperty("os.name", "").toLowerCase
    if osName.contains("win") then
      Vector(s"$libName.dll")
    else if osName.contains("mac") then
      Vector(s"lib$libName.dylib", s"$libName.dylib")
    else
      Vector(s"lib$libName.so", s"$libName.so")

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
