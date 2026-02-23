# Heads-Up Native/GPU POC Results (2026-02-19)

## Environment
- OS: Windows (PowerShell)
- Java: JDK 22 (`JAVA_HOME=C:\Program Files\Java\jdk-22`)
- Scala/SBT project: `untitled`
- GPU: NVIDIA GeForce GTX 960M (driver 581.80)

## Toolchain Notes
- CUDA 13.1 was installed first and detected (`nvcc 13.1`).
- CUDA 13.1 cannot target this GPU architecture (`sm_50`), so direct compile failed:
  - `nvcc fatal : Unsupported gpu architecture 'sm_50'`
- CUDA 11.8 was installed side-by-side and used for `sm_50` builds.
- CUDA 11.8 on current VS toolchain required compatibility flags:
  - `-allow-unsupported-compiler`
  - host define `_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH`

## Phase 1 (Historical): JNI Callback Bridge
- Native library called JVM/Scala for each matchup (`byId` + `computeEquity`).
- Numerically correct but slower than JVM CPU baseline at realistic scale.

Representative results:
- canonical, MC-200, 2000 matchups: `0.24x` to `0.33x` speedup (FAIL)
- full, MC-200, 2000 matchups: `0.36x` speedup (FAIL)

## Phase 2: Full Native CPU Kernel
- File: `src/main/native/jni/HeadsUpGpuNativeBindings.cpp`
- JVM callbacks removed from hot loop.
- Native computes full hand ranking + MC/exact directly in C++.

Representative results:
- canonical, MC-200, 2000 matchups: `35.69x` speedup vs JVM CPU, `0` validation violations
- full, MC-200, 2000 matchups: `37.21x` speedup vs JVM CPU, `0` validation violations

## Phase 3: CUDA-Enabled Native Kernel
- File: `src/main/native/jni/HeadsUpGpuNativeBindingsCuda.cu`
- Build script: `src/main/native/build-windows-cuda11.ps1`
- Output: `src/main/native/build/sicfun_gpu_kernel.dll`
- Engine mode via env:
  - `sicfun_GPU_NATIVE_ENGINE=cpu|cuda|auto`
  - `cpu`: native CPU engine inside CUDA-enabled DLL
  - `cuda`: force CUDA kernel path
  - `auto`: try CUDA then fall back to CPU

### A/B (same workload, canonical MC-200, 4000 entries, warmup=1, runs=2)
- `engine=cpu`
  - cpuAvg=`38.191s`
  - providerAvg=`1.030s`
  - speedup=`37.08x`
  - validationViolations=`0/8000`
  - gate=`PASS`
- `engine=cuda`
  - cpuAvg=`37.468s`
  - providerAvg=`0.775s`
  - speedup=`48.34x`
  - validationViolations=`0/8000`
  - gate=`PASS`

### A/B (same workload, full MC-200, 4000 entries, warmup=1, runs=2)
- `engine=cpu`
  - cpuAvg=`37.486s`
  - providerAvg=`1.288s`
  - speedup=`29.11x`
  - validationViolations=`0/8000`
  - gate=`PASS`
- `engine=cuda`
  - cpuAvg=`38.948s`
  - providerAvg=`0.777s`
  - speedup=`50.16x`
  - validationViolations=`0/8000`
  - gate=`PASS`

Interpretation:
- CUDA path outperformed native CPU path on larger Monte Carlo batches:
  - canonical 4000: `1.030s -> 0.775s` (~`1.33x` faster than native CPU engine)
  - full 4000: `1.288s -> 0.777s` (~`1.66x` faster than native CPU engine)
- Both paths are much faster than JVM CPU baseline.
- MC parity remained within gate tolerance (`0` violations in these runs).

## Single-Process Comparison CLI
- Added: `sicfun.holdem.HeadsUpBackendComparison`
- Purpose: run JVM CPU, native CPU, and native CUDA in one JVM process with same seeds/batches.
- Example:
  - `sbt "runMain sicfun.holdem.HeadsUpBackendComparison --table=canonical --mode=mc --trials=200 --maxMatchups=4000 --cpuParallelism=8 --warmupRuns=1 --runs=2 --seed=9 --nativePath=C:/.../sicfun_gpu_kernel.dll --nativeCpuEngine=cpu --nativeGpuEngine=cuda"`

Observed with this CLI:
- canonical, 2000 entries, runs=1:
  - jvm-cpu=`18.403s`
  - native-cpu=`0.438s`
  - native-cuda=`0.739s`
  - CUDA slower here (launch/transfer overhead dominates smaller batch)
- canonical, 4000 entries, warmup=1, runs=2:
  - jvm-cpu=`36.054s`
  - native-cpu=`0.913s`
  - native-cuda=`0.770s`
  - CUDA faster here (`1.19x` over native CPU)

## Optimization Pass (2026-02-20)
- Implemented CUDA/native Monte Carlo hot-path optimizations in `HeadsUpGpuNativeBindingsCuda.cu`:
  - bitmask-based board sampling (`sample_board_cards`) instead of per-trial partial shuffle/copies
  - lighter stderr math from aggregate counts (win/tie/loss), avoiding per-trial Welford updates
  - tunable CUDA launch width via:
    - system property: `sicfun.gpu.native.cuda.blockSize`
    - env var fallback: `sicfun_GPU_CUDA_BLOCK_SIZE`
    - clamped and warp-aligned

### Block-size tuning (canonical, MC-200, 4000 entries, warmup=1, runs=2)
- blockSize=`64`:
  - jvm-cpu=`34.674s`
  - native-cpu=`0.905s`
  - native-cuda=`0.753s`
  - speedup(native-cuda/native-cpu)=`1.20x`
- blockSize=`128`:
  - jvm-cpu=`37.340s`
  - native-cpu=`0.914s`
  - native-cuda=`0.772s`
  - speedup(native-cuda/native-cpu)=`1.18x`

### Full table check (full, MC-200, 4000 entries, warmup=1, runs=2, blockSize=64)
- jvm-cpu=`33.854s`
- native-cpu=`0.909s`
- native-cuda=`0.761s`
- speedup(native-cuda/native-cpu)=`1.19x`

## Validation
- `sbt test` passed (`172/172`) after changes.
