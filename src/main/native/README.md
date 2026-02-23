# Heads-Up GPU Native Bridge

The Scala runtime expects an optional JNI library exposing:

- Class: `sicfun.holdem.HeadsUpGpuNativeBindings`
- Method: `public static native int computeBatch(...)`

Method signature:

```java
int computeBatch(
    int[] lowIds,
    int[] highIds,
    int modeCode,
    int trials,
    long[] seeds,
    double[] wins,
    double[] ties,
    double[] losses,
    double[] stderrs
)
```

Input semantics:

- `lowIds`, `highIds`: hole-card combo ids (non-overlapping, canonical low/high id ordering).
- `modeCode`: `0` for exact, `1` for Monte Carlo.
- `trials`: Monte Carlo trials when `modeCode == 1`; ignored for exact.
- `seeds`: deterministic per-matchup seeds derived in JVM.

Output semantics:

- Write one result per index into `wins`, `ties`, `losses`, `stderrs`.
- Return `0` for success; non-zero for failure.

Runtime configuration:

- Provider selection:
  - system property: `-Dsicfun.gpu.provider=native|cpu-emulated|disabled`
  - env var fallback: `sicfun_GPU_PROVIDER`
- Native library loading:
  - absolute library path: `-Dsicfun.gpu.native.path=/path/to/lib`
    - env fallback: `sicfun_GPU_NATIVE_PATH`
  - OR library name for `System.loadLibrary`:
    - property: `-Dsicfun.gpu.native.lib=sicfun_gpu_kernel`
    - env fallback: `sicfun_GPU_NATIVE_LIB`
- Native worker threads (JNI/native provider):
  - env var: `sicfun_GPU_NATIVE_THREADS=<positive-int>`
  - default: `std::thread::hardware_concurrency()` (capped to batch size)
- GPU failure fallback policy (Scala table builders):
  - default: fail-fast (no CPU fallback)
  - opt-in fallback:
    - env var: `sicfun_GPU_FALLBACK_TO_CPU=true`
    - system property: `-Dsicfun.gpu.fallbackToCpu=true`
- Auto-tuner (Scala generators when `backend=gpu` and mode is Monte Carlo):
  - default: enabled
  - disable:
    - env var: `sicfun_GPU_AUTOTUNE=false`
    - system property: `-Dsicfun.gpu.autotune=false`
  - cache path:
    - env var: `sicfun_GPU_AUTOTUNE_CACHE_PATH=<file>`
    - system property: `-Dsicfun.gpu.autotune.cachePath=<file>`
  - default cache file: `data/headsup-backend-autotune.properties`
  - skipped automatically when engine/block/chunk are explicitly set (property or env)
- Native engine selection (CUDA-enabled JNI build):
  - system property: `-Dsicfun.gpu.native.engine=auto|cpu|cuda`
  - env var: `sicfun_GPU_NATIVE_ENGINE=auto|cpu|cuda`
  - `auto` (default): try CUDA for Monte Carlo and fall back to CPU on CUDA errors
  - `cpu`: force native CPU path
  - `cuda`: require CUDA path (return error if CUDA execution fails)
- CUDA launch parallelism (CUDA-enabled JNI build):
  - system property: `-Dsicfun.gpu.native.cuda.blockSize=<positive-int>`
  - env var fallback: `sicfun_GPU_CUDA_BLOCK_SIZE=<positive-int>`
  - rounded to warp size and clamped to device max; default is `128`
- CUDA batch chunking (CUDA-enabled JNI build):
  - system property: `-Dsicfun.gpu.native.cuda.maxChunkMatchups=<positive-int>`
  - env var fallback: `sicfun_GPU_CUDA_MAX_CHUNK_MATCHUPS=<positive-int>`
  - splits large batches into smaller kernel launches to avoid Windows WDDM/TDR watchdog timeouts; default is `4096`

CUDA 11.8 Windows build (for older GPUs like `sm_50`):

```powershell
powershell -ExecutionPolicy Bypass -File src/main/native/build-windows-cuda11.ps1
```

This produces:

- `src/main/native/build/sicfun_gpu_kernel.dll`

Compatibility note:

- CUDA 13.x does not include `sm_50` targets, so Maxwell-era GPUs (e.g. GTX 960M) require CUDA 11.8 for native CUDA builds.

Gate benchmark:

```bash
sbt "runMain sicfun.holdem.HeadsUpGpuPocGate --table=canonical --mode=mc --trials=200 --maxMatchups=2000 --cpuParallelism=8 --speedupThreshold=5 --warmupRuns=1 --runs=2 --seed=1"
```

Fast distribution gates (must use real CUDA, not fallback):

```bash
sbt gpuSmokeGate
sbt gpuExactParityGate
```

PowerShell wrappers:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/gpu-smoke-gate.ps1
powershell -ExecutionPolicy Bypass -File scripts/gpu-exact-parity-gate.ps1
```

Windows release packaging with startup verification from the packaged layout:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/release-windows.ps1
```

Three-way backend comparison (single process):

```bash
sbt "runMain sicfun.holdem.HeadsUpBackendComparison --table=canonical --mode=mc --trials=200 --maxMatchups=4000 --cpuParallelism=8 --warmupRuns=1 --runs=2 --seed=9 --nativePath=C:/path/to/sicfun_gpu_kernel.dll --nativeCpuEngine=cpu --nativeGpuEngine=cuda"
```

`HeadsUpBackendComparison` also uses the same auto-tuner/cache path by default when
`nativeGpuEngine=cuda` and CUDA block/chunk are not explicitly provided. You can disable
it with `--nativeAutoTune=false`, or override directly with:

- `--nativeCudaBlockSize=<int>`
- `--nativeCudaMaxChunkMatchups=<int>`

In comparison mode, auto-tuning is CUDA-only (it tunes CUDA launch parameters, not backend
selection), so CPU and CUDA measurements remain directly comparable.

With explicit CUDA block size tuning:

```bash
sbt "runMain sicfun.holdem.HeadsUpBackendComparison --table=canonical --mode=mc --trials=200 --maxMatchups=4000 --cpuParallelism=8 --warmupRuns=1 --runs=2 --seed=9 --nativePath=C:/path/to/sicfun_gpu_kernel.dll --nativeCpuEngine=cpu --nativeGpuEngine=cuda --nativeCudaBlockSize=128"
```

On GTX 960M (`sm_50`) in this project workload, `96` is a good default block size.

For higher-trial Monte Carlo on Windows WDDM (watchdog-sensitive), tune chunk size and block size together.
On this project + GTX 960M, a stable fast setting for canonical `mc=5000` is:

- `sicfun_GPU_CUDA_BLOCK_SIZE=96`
- `sicfun_GPU_CUDA_MAX_CHUNK_MATCHUPS=512`

With the direct 7-card evaluator kernel, this runs full canonical `mc=5000` in roughly 15-25 seconds on this machine.
