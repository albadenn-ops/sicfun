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

## Holdem CFR native bridge

The Holdem CFR runtime can solve the same game tree via Scala CFR or native JNI providers:

- CPU JNI provider:
  - class: `sicfun.holdem.HoldemCfrNativeCpuBindings`
  - library: `sicfun_cfr_native`
- GPU JNI provider (CUDA build):
  - class: `sicfun.holdem.HoldemCfrNativeGpuBindings`
  - library: `sicfun_cfr_cuda`
  - current implementation uses the shared C++ CFR core compiled with NVCC

Provider selection:

- system property: `-Dsicfun.cfr.provider=auto|scala|native-cpu|native-gpu`
- env fallback: `sicfun_CFR_PROVIDER`
- default: `auto`

Auto mode behavior:

- runs a one-time synthetic benchmark (Scala vs available native providers)
- selects native only when speedup meets threshold
- caches provider choice for the process

Auto tuning controls:

- benchmark iterations:
  - property: `-Dsicfun.cfr.auto.benchmarkIterations=<positive-int>`
  - env: `sicfun_CFR_AUTO_BENCHMARK_ITERATIONS`
  - default: `240`
- minimum native speedup:
  - property: `-Dsicfun.cfr.auto.nativeMinSpeedup=<double>`
  - env: `sicfun_CFR_AUTO_NATIVE_MIN_SPEEDUP`
  - default: `1.02`

Native library loading overrides:

- CPU provider:
  - path: `-Dsicfun.cfr.native.cpu.path=<abs-path>`
    - env: `sicfun_CFR_NATIVE_CPU_PATH`
  - loadLibrary name: `-Dsicfun.cfr.native.cpu.lib=<name>`
    - env: `sicfun_CFR_NATIVE_CPU_LIB`
    - default: `sicfun_cfr_native`
- GPU provider:
  - path: `-Dsicfun.cfr.native.gpu.path=<abs-path>`
    - env: `sicfun_CFR_NATIVE_GPU_PATH`
  - loadLibrary name: `-Dsicfun.cfr.native.gpu.lib=<name>`
    - env: `sicfun_CFR_NATIVE_GPU_LIB`
    - default: `sicfun_cfr_cuda`

## Holdem Bayesian native bridge

The Bayesian posterior-update runtime supports Scala and native JNI providers:

- CPU JNI provider:
  - class: `sicfun.holdem.HoldemBayesNativeCpuBindings`
  - library: `sicfun_bayes_native`
- GPU JNI provider (CUDA build):
  - class: `sicfun.holdem.HoldemBayesNativeGpuBindings`
  - library: `sicfun_bayes_cuda`
  - current implementation uses the shared C++ Bayesian core compiled with NVCC

Provider selection:

- system property: `-Dsicfun.bayes.provider=auto|scala|native-cpu|native-gpu`
- env fallback: `sicfun_BAYES_PROVIDER`
- default: `auto`

Auto tuning controls:

- benchmark repetitions:
  - property: `-Dsicfun.bayes.auto.benchmarkRepetitions=<positive-int>`
  - env: `sicfun_BAYES_AUTO_BENCHMARK_REPETITIONS`
  - default: `20`
- minimum native speedup:
  - property: `-Dsicfun.bayes.auto.nativeMinSpeedup=<double>`
  - env: `sicfun_BAYES_AUTO_NATIVE_MIN_SPEEDUP`
  - default: `1.02`

Shadow parity controls (optional):

- enable Scala shadow validation for native results:
  - property: `-Dsicfun.bayes.shadow.enabled=true|false`
  - env: `sicfun_BAYES_SHADOW_ENABLED`
  - default: `false`
- fail-closed behavior on drift:
  - property: `-Dsicfun.bayes.shadow.failClosed=true|false`
  - env: `sicfun_BAYES_SHADOW_FAIL_CLOSED`
  - default: `false`
  - `true`: return Scala reference result on drift (or throw if shadow reference fails)
  - `false`: keep native result and emit warning on drift
- posterior max absolute delta tolerance:
  - property: `-Dsicfun.bayes.shadow.posteriorMaxAbsDiff=<non-negative-double>`
  - env: `sicfun_BAYES_SHADOW_POSTERIOR_MAX_ABS_DIFF`
  - default: `1e-9`
- log-evidence max absolute delta tolerance:
  - property: `-Dsicfun.bayes.shadow.logEvidenceMaxAbsDiff=<non-negative-double>`
  - env: `sicfun_BAYES_SHADOW_LOG_EVIDENCE_MAX_ABS_DIFF`
  - default: `1e-9`

Native library loading overrides:

- CPU provider:
  - path: `-Dsicfun.bayes.native.cpu.path=<abs-path>`
    - env: `sicfun_BAYES_NATIVE_CPU_PATH`
  - loadLibrary name: `-Dsicfun.bayes.native.cpu.lib=<name>`
    - env: `sicfun_BAYES_NATIVE_CPU_LIB`
    - default: `sicfun_bayes_native`
- GPU provider:
  - path: `-Dsicfun.bayes.native.gpu.path=<abs-path>`
    - env: `sicfun_BAYES_NATIVE_GPU_PATH`
  - loadLibrary name: `-Dsicfun.bayes.native.gpu.lib=<name>`
    - env: `sicfun_BAYES_NATIVE_GPU_LIB`
    - default: `sicfun_bayes_cuda`

## Holdem DDRE native bridge

The DDRE provider now supports Scala synthetic mode plus native JNI CPU/CUDA providers:

- CPU JNI provider:
  - class: `sicfun.holdem.HoldemDdreNativeCpuBindings`
  - library: `sicfun_ddre_native`
- GPU JNI provider (CUDA build):
  - class: `sicfun.holdem.HoldemDdreNativeGpuBindings`
  - library: `sicfun_ddre_cuda`
  - current implementation uses the shared DDRE synthetic inference core compiled with NVCC

Provider selection:

- system property: `-Dsicfun.ddre.provider=disabled|synthetic|native-cpu|native-gpu|onnx`
- env fallback: `sicfun_DDRE_PROVIDER`
- default: `disabled`

ONNX runtime option:

- provider value: `onnx`
- requires model path:
  - property: `-Dsicfun.ddre.onnx.modelPath=<path-to-model.onnx>`
  - env: `sicfun_DDRE_ONNX_MODEL_PATH`
- optional input/output names:
  - `-Dsicfun.ddre.onnx.input.prior=<name>` (default `prior`)
  - `-Dsicfun.ddre.onnx.input.likelihoods=<name>` (default `likelihoods`)
  - `-Dsicfun.ddre.onnx.output.posterior=<name>` (default `posterior`)
- optional execution controls:
  - `-Dsicfun.ddre.onnx.executionProvider=cpu|cuda` (default `cpu`)
  - `-Dsicfun.ddre.onnx.cuda.device=<non-negative-int>` (default `0`)
  - `-Dsicfun.ddre.onnx.intraOpThreads=<positive-int>`
  - `-Dsicfun.ddre.onnx.interOpThreads=<positive-int>`
- note: DDRE ONNX adapter uses reflection against `ai.onnxruntime`; if runtime classes are missing it degrades safely to Bayesian fallback through existing DDRE error handling.
- project dependency pinning:
  - `build.sbt` pins `com.microsoft.onnxruntime:onnxruntime:1.19.2` for reproducible JVM runtime behavior.

DDRE ONNX smoke artifact generation:

```bash
python scripts/generate-ddre-smoke-onnx.py
```

This writes `src/test/resources/sicfun/ddre/ddre-smoke-sqrt.onnx` (`posterior = sqrt(prior)`), used by DDRE ONNX integration smoke tests.

Native library loading overrides:

- CPU provider:
  - path: `-Dsicfun.ddre.native.cpu.path=<abs-path>`
    - env: `sicfun_DDRE_NATIVE_CPU_PATH`
  - loadLibrary name: `-Dsicfun.ddre.native.cpu.lib=<name>`
    - env: `sicfun_DDRE_NATIVE_CPU_LIB`
    - default: `sicfun_ddre_native`
- GPU provider:
  - path: `-Dsicfun.ddre.native.gpu.path=<abs-path>`
    - env: `sicfun_DDRE_NATIVE_GPU_PATH`
  - loadLibrary name: `-Dsicfun.ddre.native.gpu.lib=<name>`
    - env: `sicfun_DDRE_NATIVE_GPU_LIB`
    - default: `sicfun_ddre_cuda`

DDRE parity/benchmark gate (synthetic vs native CPU/GPU):

```bash
sbt "runMain sicfun.holdem.HoldemDdreParityBenchmark --modes=synthetic,native-cpu,native-gpu --referenceMode=synthetic --maxL1Diff=1e-6 --maxAbsDiff=1e-7 --requireAllModes=true"
```

DDRE parity/benchmark with ONNX smoke model:

```bash
sbt "runMain sicfun.holdem.HoldemDdreParityBenchmark --modes=synthetic,onnx --referenceMode=synthetic --onnxModelPath=src/test/resources/sicfun/ddre/ddre-smoke-sqrt.onnx --maxL1Diff=1e-4 --maxAbsDiff=1e-5 --warmupRuns=0 --measureRuns=2"
```

## Holdem postflop native bridge

Postflop Monte Carlo (flop/turn/river) can route through a dedicated native JNI bridge:

- CPU class/library:
  - class: `sicfun.holdem.HoldemPostflopNativeBindings`
  - library: `sicfun_postflop_native`
- GPU class/library:
  - class: `sicfun.holdem.HoldemPostflopNativeGpuBindings`
  - library: `sicfun_postflop_cuda`
- entrypoint:
  - `computePostflopBatchMonteCarlo(heroFirst, heroSecond, boardCards, boardSize, villainFirstCards, villainSecondCards, trials, seeds, wins, ties, losses, stderrs)`
- native telemetry helpers:
  - `queryNativeEngine()`:
    - CPU bridge: `0=unknown, 1=cpu, 2=cuda`
    - CUDA bridge: `0=unknown, 1=cpu, 2=cuda, 3=cpu-fallback-after-cuda-failure`
  - `cudaDeviceCount()` (CUDA bridge)
  - `cudaDeviceInfo(deviceIndex)` (CUDA bridge, used by auto-tune cache fingerprinting)

Runtime controls:

- provider selection:
  - property: `-Dsicfun.postflop.provider=auto|native|disabled`
  - env: `sicfun_POSTFLOP_PROVIDER`
  - default: `auto`
- native library loading:
  - absolute path: `-Dsicfun.postflop.native.path=<abs-path>`
    - env: `sicfun_POSTFLOP_NATIVE_PATH`
  - loadLibrary name: `-Dsicfun.postflop.native.lib=<name>`
    - env: `sicfun_POSTFLOP_NATIVE_LIB`
    - default: `sicfun_postflop_native`
- CPU-specific overrides:
  - path: `-Dsicfun.postflop.native.cpu.path=<abs-path>`
    - env: `sicfun_POSTFLOP_NATIVE_CPU_PATH`
  - loadLibrary name: `-Dsicfun.postflop.native.cpu.lib=<name>`
    - env: `sicfun_POSTFLOP_NATIVE_CPU_LIB`
    - default: `sicfun_postflop_native`
- GPU-specific overrides:
  - path: `-Dsicfun.postflop.native.gpu.path=<abs-path>`
    - env: `sicfun_POSTFLOP_NATIVE_GPU_PATH`
  - loadLibrary name: `-Dsicfun.postflop.native.gpu.lib=<name>`
    - env: `sicfun_POSTFLOP_NATIVE_GPU_LIB`
    - default: `sicfun_postflop_cuda`
- engine hint:
  - property: `-Dsicfun.postflop.native.engine=auto|cpu|cuda`
  - env: `sicfun_POSTFLOP_NATIVE_ENGINE`
  - `auto` attempts GPU first, then falls back to CPU if GPU execution fails
  - `auto` also prefers CPU for small workloads (`villains * trials`) below:
    - property: `-Dsicfun.postflop.native.auto.minGpuWork=<positive-int>`
    - env: `sicfun_POSTFLOP_NATIVE_AUTO_MIN_GPU_WORK`
    - default: `300000`
- native worker threads:
  - env: `sicfun_POSTFLOP_NATIVE_THREADS=<positive-int>`
  - default: hardware concurrency capped to batch size
- CUDA launch controls (GPU library):
  - property: `-Dsicfun.postflop.native.cuda.blockSize=<positive-int>`
  - env: `sicfun_POSTFLOP_CUDA_BLOCK_SIZE=<positive-int>`
  - default: `128`, normalized to warp-size multiples
- CUDA chunking controls (GPU library):
  - property: `-Dsicfun.postflop.native.cuda.maxChunkMatchups=<positive-int>`
  - env: `sicfun_POSTFLOP_CUDA_MAX_CHUNK_MATCHUPS=<positive-int>`
  - default: `4096`
- CUDA trial chunking controls (GPU library):
  - property: `-Dsicfun.postflop.native.cuda.maxTrialsPerLaunch=<positive-int>`
  - env: `sicfun_POSTFLOP_CUDA_MAX_TRIALS_PER_LAUNCH=<positive-int>`
  - splits large trial counts into multiple launches to reduce WDDM/TDR risk
  - default: `4096`
- postflop CUDA auto-tune cache (runtime apply-only):
  - enable/disable:
    - property: `-Dsicfun.postflop.autotune=true|false`
    - env: `sicfun_POSTFLOP_AUTOTUNE`
    - default: enabled
  - cache file:
    - property: `-Dsicfun.postflop.autotune.cachePath=<file>`
    - env: `sicfun_POSTFLOP_AUTOTUNE_CACHE_PATH`
    - default: `data/postflop-autotune.properties`
  - cache is skipped when either CUDA override is explicitly set:
    - `sicfun.postflop.native.cuda.blockSize`
    - `sicfun.postflop.native.cuda.maxChunkMatchups`

Bayesian hotspot benchmark:

```bash
sbt "runMain sicfun.holdem.HoldemBayesBenchmark --warmupRuns=2 --measureRuns=8 --bunchingTrials=1200 --equityTrials=8000 --provider=auto --seed=17"
```

CUDA 11.8 Windows build (for older GPUs like `sm_50`):

```powershell
powershell -ExecutionPolicy Bypass -File src/main/native/build-windows-cuda11.ps1
```

Machine prerequisite check:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ensure-gpu-build-prereqs.ps1
```

Machine prerequisite auto-install:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ensure-gpu-build-prereqs.ps1 -InstallMissing
```

Build script discovery/override notes:

- auto-discovers JDK headers from `-JavaHome`, `SICFUN_GPU_BUILD_JAVA_HOME`, `JAVA_HOME`, or `javac.exe` on `PATH`
- also scans common installed JDK roots under `Program Files\Java`, `Program Files\Microsoft`, and `Program Files\Eclipse Adoptium`
- auto-discovers CUDA from `-CudaRoot`, `SICFUN_GPU_BUILD_CUDA_ROOT`, `CUDA_PATH`, or `nvcc.exe` on `PATH`
- auto-discovers `vcvars64.bat` from `-VcVars`, `SICFUN_GPU_BUILD_VCVARS`, or Visual Studio `vswhere`
- auto-detects local GPU compute capability from `nvidia-smi` on `PATH` or the default `NVSMI` install path unless you override `-Arch=<sm_xy>` or `-Architectures=<csv>`
- if auto-detection returns a capability newer than this CUDA 11 build supports, the script falls back to the highest supported CUDA 11 target and warns
- the script currently supports Windows x64 hosts only because it resolves `vcvars64.bat` and produces x64 DLLs
- use `-CheckPrerequisites` to validate the machine before building, or `-InstallMissingPrerequisites` to let the script install supported missing prerequisites with `winget`

This produces:

- `src/main/native/build/sicfun_gpu_kernel.dll`
- `src/main/native/build/sicfun_cfr_cuda.dll`
- `src/main/native/build/sicfun_bayes_cuda.dll`
- `src/main/native/build/sicfun_ddre_cuda.dll`
- `src/main/native/build/sicfun_postflop_cuda.dll`

LLVM Windows build:

```powershell
powershell -ExecutionPolicy Bypass -File src/main/native/build-windows-llvm.ps1
```

This produces:

- `src/main/native/build/sicfun_native_cpu.dll`
- `src/main/native/build/sicfun_cfr_native.dll`
- `src/main/native/build/sicfun_bayes_native.dll`
- `src/main/native/build/sicfun_ddre_native.dll`
- `src/main/native/build/sicfun_postflop_native.dll`

Compatibility note:

- CUDA 12.0 and later do not include `sm_50` targets, so Maxwell-era GPUs (e.g. GTX 960M) require CUDA 11.8 for native CUDA builds.

Postflop backend benchmark (Scala vs native CPU/CUDA):

```bash
sbt "runMain sicfun.holdem.HoldemPostflopNativeBenchmark --warmupRuns=2 --measureRuns=8 --trials=12000 --modes=scala,native-cpu,native-cuda,native-auto --seed=31"
```

Postflop CUDA auto-tuner (writes `data/postflop-autotune.properties` by default):

```bash
sbt "runMain sicfun.holdem.HoldemPostflopGpuAutoTuner --villains=1024 --trials=2000 --warmupRuns=1 --runs=3 --cachePath=data/postflop-autotune.properties"
```

Optional tuning argument overrides:

- `--nativeGpuPath=<abs-path>`
- `--blockCandidates=64,96,128,160,192,256`
- `--chunkCandidates=256,512,1024,2048,4096`
- `--villains=<positive-int>` (benchmark only; default `8`, use larger values like `512`/`1024` for throughput stress)

Postflop parity/behavior tests:

```bash
sbt "testOnly sicfun.holdem.HoldemPostflopNativeParityTest"
```

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
