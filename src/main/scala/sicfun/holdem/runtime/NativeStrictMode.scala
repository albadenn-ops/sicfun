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
