/*
 * cuda_throughput_bench.cu -- CUDA micro-benchmark for arithmetic throughput
 * comparison: int32 vs float32 vs float64 (double) on the target GPU.
 *
 * Part of the sicfun poker analytics system's native acceleration layer.
 * This benchmark measures the practical throughput of CFR-style operations
 * (regret accumulation, regret matching / normalization, strategy weighting,
 * clamping to non-negative) across three numeric types to inform the choice
 * of arithmetic precision for the GPU CFR solver.
 *
 * Each kernel simulates a 3-action CFR infoset update loop:
 *   1. Regret matching: clamp regrets to non-negative, normalize to strategy.
 *   2. Strategy accumulation: weighted sum of strategies.
 *   3. Regret update with CFR+ clamping.
 *
 * On Maxwell sm_50 (GTX 960M), float64 has 1/32 the throughput of float32,
 * making the choice of precision critical for GPU CFR performance.
 *
 * Benchmark methodology:
 *   - kWarmup warmup launches (not timed) to stabilize GPU clocks.
 *   - kRuns timed launches with interleaved execution order (alternating
 *     forward/reverse) to mitigate thermal throttling bias.
 *   - Reports median of kRuns for each type (robust to outliers).
 *   - Uses CUDA events for sub-millisecond GPU-side timing.
 *
 * Target GPU: GTX 960M (Maxwell, sm_50, CUDA 11.8).
 *
 * Compile:  nvcc -std=c++17 -O3 -gencode arch=compute_50,code=sm_50 -o cuda_bench.exe cuda_throughput_bench.cu
 * Run:      cuda_bench.exe
 */

#include <cstdint>
#include <cstdio>

/* ---- Launch configuration ------------------------------------------------ */

constexpr int kThreads = 128;       /* Threads per block. */
constexpr int kBlocks = 40;         /* ~5 SMs on GTX 960M, 8 blocks/SM occupancy target. */
constexpr int kIterations = 5000;   /* Inner loop iterations per thread. Kept under TDR (~2s). */
constexpr int kWarmup = 2;          /* Warmup launches (not timed) to stabilize GPU clocks. */
constexpr int kRuns = 10;           /* Timed runs; median is reported. */

/* ---- int32 kernel: simulates CFR regret accumulation + regret matching.
 * Uses integer division for normalization (1024 scale = ~Q10 fixed-point). */
__global__ void bench_int32(int* out, const int n_iter) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int regret0 = tid + 1;
  int regret1 = -(tid / 2);
  int regret2 = tid * 3 - 500;
  int cum_strategy0 = 0;
  int cum_strategy1 = 0;
  int cum_strategy2 = 0;

  for (int i = 0; i < n_iter; ++i) {
    int pos0 = regret0 > 0 ? regret0 : 0;
    int pos1 = regret1 > 0 ? regret1 : 0;
    int pos2 = regret2 > 0 ? regret2 : 0;
    int sum = pos0 + pos1 + pos2;

    int s0, s1, s2;
    if (sum > 0) {
      s0 = (pos0 * 1024) / sum;
      s1 = (pos1 * 1024) / sum;
      s2 = 1024 - s0 - s1;
    } else {
      s0 = 341; s1 = 341; s2 = 342;
    }

    int weight = i + 1;
    cum_strategy0 += s0 * weight;
    cum_strategy1 += s1 * weight;
    cum_strategy2 += s2 * weight;

    regret0 += (i & 1) ? 3 : -2;
    regret1 += (i & 1) ? -1 : 4;
    regret2 += (i & 1) ? 2 : -1;

    if (regret0 < 0) regret0 = 0;
    if (regret1 < 0) regret1 = 0;
    if (regret2 < 0) regret2 = 0;
  }

  out[tid] = cum_strategy0 + cum_strategy1 + cum_strategy2;
}

/* ---- float32 kernel: same CFR logic using single-precision floating-point.
 * Uses reciprocal multiply for normalization (1/sum instead of division). */
__global__ void bench_float(float* out, const int n_iter) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  float regret0 = static_cast<float>(tid + 1);
  float regret1 = static_cast<float>(-(tid / 2));
  float regret2 = static_cast<float>(tid * 3 - 500);
  float cum_strategy0 = 0.0f;
  float cum_strategy1 = 0.0f;
  float cum_strategy2 = 0.0f;

  for (int i = 0; i < n_iter; ++i) {
    float pos0 = regret0 > 0.0f ? regret0 : 0.0f;
    float pos1 = regret1 > 0.0f ? regret1 : 0.0f;
    float pos2 = regret2 > 0.0f ? regret2 : 0.0f;
    float sum = pos0 + pos1 + pos2;

    float s0, s1, s2;
    if (sum > 0.0f) {
      float inv = 1.0f / sum;
      s0 = pos0 * inv;
      s1 = pos1 * inv;
      s2 = 1.0f - s0 - s1;
    } else {
      s0 = 0.333333f; s1 = 0.333333f; s2 = 0.333334f;
    }

    float weight = static_cast<float>(i + 1);
    cum_strategy0 += s0 * weight;
    cum_strategy1 += s1 * weight;
    cum_strategy2 += s2 * weight;

    regret0 += (i & 1) ? 3.0f : -2.0f;
    regret1 += (i & 1) ? -1.0f : 4.0f;
    regret2 += (i & 1) ? 2.0f : -1.0f;

    if (regret0 < 0.0f) regret0 = 0.0f;
    if (regret1 < 0.0f) regret1 = 0.0f;
    if (regret2 < 0.0f) regret2 = 0.0f;
  }

  out[tid] = cum_strategy0 + cum_strategy1 + cum_strategy2;
}

/* ---- float64 (double) kernel: same CFR logic using double-precision.
 * On Maxwell sm_50, double throughput is 1/32 of float — this is the
 * pathological case that motivates using float32 for the GPU CFR solver. */
__global__ void bench_double(double* out, const int n_iter) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  double regret0 = static_cast<double>(tid + 1);
  double regret1 = static_cast<double>(-(tid / 2));
  double regret2 = static_cast<double>(tid * 3 - 500);
  double cum_strategy0 = 0.0;
  double cum_strategy1 = 0.0;
  double cum_strategy2 = 0.0;

  for (int i = 0; i < n_iter; ++i) {
    double pos0 = regret0 > 0.0 ? regret0 : 0.0;
    double pos1 = regret1 > 0.0 ? regret1 : 0.0;
    double pos2 = regret2 > 0.0 ? regret2 : 0.0;
    double sum = pos0 + pos1 + pos2;

    double s0, s1, s2;
    if (sum > 0.0) {
      double inv = 1.0 / sum;
      s0 = pos0 * inv;
      s1 = pos1 * inv;
      s2 = 1.0 - s0 - s1;
    } else {
      s0 = 0.333333; s1 = 0.333333; s2 = 0.333334;
    }

    double weight = static_cast<double>(i + 1);
    cum_strategy0 += s0 * weight;
    cum_strategy1 += s1 * weight;
    cum_strategy2 += s2 * weight;

    regret0 += (i & 1) ? 3.0 : -2.0;
    regret1 += (i & 1) ? -1.0 : 4.0;
    regret2 += (i & 1) ? 2.0 : -1.0;

    if (regret0 < 0.0) regret0 = 0.0;
    if (regret1 < 0.0) regret1 = 0.0;
    if (regret2 < 0.0) regret2 = 0.0;
  }

  out[tid] = cum_strategy0 + cum_strategy1 + cum_strategy2;
}

int main() {
  fflush(stdout);
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    return 1;
  }
  printf("CUDA devices found: %d\n", device_count);
  fflush(stdout);
  if (device_count == 0) {
    printf("No CUDA devices available\n");
    return 1;
  }

  cudaDeviceProp prop;
  err = cudaGetDeviceProperties(&prop, 0);
  if (err != cudaSuccess) {
    printf("cudaGetDeviceProperties error: %s\n", cudaGetErrorString(err));
    return 1;
  }
  printf("GPU: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  printf("SMs: %d\n", prop.multiProcessorCount);
  printf("Clock: %d MHz\n", prop.clockRate / 1000);
  printf("Threads: %d x %d = %d total\n", kBlocks, kThreads, kBlocks * kThreads);
  printf("Inner iterations per thread: %d\n", kIterations);
  fflush(stdout);

  const int total_threads = kBlocks * kThreads;

  int* d_int;      err = cudaMalloc(&d_int, total_threads * sizeof(int));
  printf("cudaMalloc int: %s\n", cudaGetErrorString(err)); fflush(stdout);
  float* d_float;  err = cudaMalloc(&d_float, total_threads * sizeof(float));
  double* d_double; err = cudaMalloc(&d_double, total_threads * sizeof(double));
  printf("Allocated. Starting warmup...\n"); fflush(stdout);

  float int_times[kRuns], float_times[kRuns], double_times[kRuns];

  // Warmup all three
  for (int w = 0; w < kWarmup; ++w) {
    bench_int32<<<kBlocks, kThreads>>>(d_int, kIterations);
    err = cudaDeviceSynchronize();
    if (w == 0) { printf("Warmup int32: %s\n", cudaGetErrorString(err)); fflush(stdout); }
    bench_float<<<kBlocks, kThreads>>>(d_float, kIterations);
    err = cudaDeviceSynchronize();
    if (w == 0) { printf("Warmup float: %s\n", cudaGetErrorString(err)); fflush(stdout); }
    bench_double<<<kBlocks, kThreads>>>(d_double, kIterations);
    err = cudaDeviceSynchronize();
    if (w == 0) { printf("Warmup double: %s\n", cudaGetErrorString(err)); fflush(stdout); }
  }
  printf("Warmup done. Running timed iterations...\n"); fflush(stdout);

  // Interleaved runs to avoid thermal bias
  for (int r = 0; r < kRuns; ++r) {
    cudaEvent_t s1, e1, s2, e2, s3, e3;
    cudaEventCreate(&s1); cudaEventCreate(&e1);
    cudaEventCreate(&s2); cudaEventCreate(&e2);
    cudaEventCreate(&s3); cudaEventCreate(&e3);

    if (r % 2 == 0) {
      cudaEventRecord(s1); bench_int32<<<kBlocks, kThreads>>>(d_int, kIterations); cudaEventRecord(e1); cudaEventSynchronize(e1);
      cudaEventRecord(s2); bench_float<<<kBlocks, kThreads>>>(d_float, kIterations); cudaEventRecord(e2); cudaEventSynchronize(e2);
      cudaEventRecord(s3); bench_double<<<kBlocks, kThreads>>>(d_double, kIterations); cudaEventRecord(e3); cudaEventSynchronize(e3);
    } else {
      cudaEventRecord(s3); bench_double<<<kBlocks, kThreads>>>(d_double, kIterations); cudaEventRecord(e3); cudaEventSynchronize(e3);
      cudaEventRecord(s2); bench_float<<<kBlocks, kThreads>>>(d_float, kIterations); cudaEventRecord(e2); cudaEventSynchronize(e2);
      cudaEventRecord(s1); bench_int32<<<kBlocks, kThreads>>>(d_int, kIterations); cudaEventRecord(e1); cudaEventSynchronize(e1);
    }

    cudaEventElapsedTime(&int_times[r], s1, e1);
    cudaEventElapsedTime(&float_times[r], s2, e2);
    cudaEventElapsedTime(&double_times[r], s3, e3);

    cudaEventDestroy(s1); cudaEventDestroy(e1);
    cudaEventDestroy(s2); cudaEventDestroy(e2);
    cudaEventDestroy(s3); cudaEventDestroy(e3);
  }

  // Sort for median
  auto sort_arr = [](float* a, int n) {
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j)
        if (a[j] < a[i]) { float t = a[i]; a[i] = a[j]; a[j] = t; }
  };
  sort_arr(int_times, kRuns);
  sort_arr(float_times, kRuns);
  sort_arr(double_times, kRuns);

  float int_med = int_times[kRuns / 2];
  float float_med = float_times[kRuns / 2];
  float double_med = double_times[kRuns / 2];

  printf("--- int32 (all runs, sorted) ---\n");
  for (int r = 0; r < kRuns; ++r) printf("  %.2f ms\n", int_times[r]);

  printf("--- float (all runs, sorted) ---\n");
  for (int r = 0; r < kRuns; ++r) printf("  %.2f ms\n", float_times[r]);

  printf("--- double (all runs, sorted) ---\n");
  for (int r = 0; r < kRuns; ++r) printf("  %.2f ms\n", double_times[r]);

  printf("\n=== RESULTS (median of %d runs) ===\n", kRuns);
  printf("int32:  %8.2f ms\n", int_med);
  printf("float:  %8.2f ms\n", float_med);
  printf("double: %8.2f ms\n", double_med);
  printf("\n");
  printf("int32  vs double: %.1fx faster\n", double_med / int_med);
  printf("float  vs double: %.1fx faster\n", double_med / float_med);
  printf("int32  vs float:  %.1fx faster\n", float_med / int_med);

  cudaFree(d_int);
  cudaFree(d_float);
  cudaFree(d_double);
  return 0;
}
