package sicfun.holdem.runtime

import java.io.File

final case class NativeStrictModeViolation(message: String)
    extends RuntimeException(message)

final case class NativeReady(
    loaded: Vector[String],
    contaminated: Boolean
)

object NativeStrictMode:

  /** One library = one DLL, plus the source files whose mtime gates staleness. */
  final case class Library(name: String, sources: Vector[String])

  /** Canonical outputs of `src/main/native/build-windows-cuda11.ps1` as of the
    * plan's reference commit. Re-verify via the grep in Task 12 Step 1 if the
    * build script changes. */
  val CoreLibraries: Vector[Library] = Vector(
    Library("sicfun_gpu_kernel", Vector(
      "src/main/native/jni/HeadsUpGpuNativeBindings.cpp",
      "src/main/native/jni/HeadsUpGpuNativeBindingsCuda.cu"
    )),
    Library("sicfun_cfr_cuda", Vector(
      "src/main/native/jni/HoldemCfrNativeGpuBindings.cu"
    )),
    Library("sicfun_bayes_cuda", Vector(
      "src/main/native/jni/HoldemBayesNativeGpuBindings.cu",
      "src/main/native/jni/BayesNativeUpdateCore.hpp"
    )),
    Library("sicfun_ddre_cuda", Vector(
      "src/main/native/jni/HoldemDdreNativeGpuBindings.cu",
      "src/main/native/jni/DdreNativeInferenceCore.hpp"
    )),
    Library("sicfun_postflop_cuda", Vector(
      "src/main/native/jni/HoldemPostflopNativeBindingsCuda.cu"
    ))
  )

  def verify(
      nativeDir: File,
      required: Vector[Library],
      strict: Boolean
  ): Either[NativeStrictModeViolation, NativeReady] =
    val missing = required.filter(lib => !dllFile(nativeDir, lib.name).exists())
    val stale = required.flatMap { lib =>
      val dll = dllFile(nativeDir, lib.name)
      if !dll.exists() then None
      else
        val dllMtime = dll.lastModified()
        lib.sources
          .map(p => new File(p))
          .find(src => src.exists() && src.lastModified() > dllMtime)
          .map(src => (lib.name, src.getPath))
    }
    (missing, stale, strict) match
      case (m, _, true) if m.nonEmpty =>
        Left(NativeStrictModeViolation(
          s"Missing native libraries (strict): ${m.map(_.name).mkString(", ")}; " +
            "rebuild with src/main/native/build-windows-cuda11.ps1"
        ))
      case (_, s, true) if s.nonEmpty =>
        Left(NativeStrictModeViolation(
          s"Stale native builds (strict): ${s.map((d, src) => s"$d older than $src").mkString("; ")}"
        ))
      case (m, s, false) if m.nonEmpty || s.nonEmpty =>
        System.err.println(
          s"[BENCHMARK-CONTAMINATED] strict=false, missing=[${m.map(_.name).mkString(",")}], " +
            s"stale=[${s.map(_._1).mkString(",")}]"
        )
        Right(NativeReady(Vector.empty, contaminated = true))
      case _ =>
        Right(NativeReady(required.map(_.name), contaminated = false))

  private def dllFile(dir: File, libName: String): File =
    val os = System.getProperty("os.name", "").toLowerCase
    if os.contains("win") then new File(dir, s"$libName.dll")
    else if os.contains("mac") then new File(dir, s"lib$libName.dylib")
    else new File(dir, s"lib$libName.so")
