package sicfun.holdem.runtime

import java.io.File

final case class NativeStrictModeViolation(message: String)
    extends RuntimeException(message)

final case class NativeReady(
    loaded: Vector[String],
    contaminated: Boolean
)

object NativeStrictMode:

  final case class Library(name: String, sources: Vector[String])

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
