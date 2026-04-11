/*
 * HeadsUpGpuNativeBindingsCuda.cu -- CUDA + CPU JNI binding for heads-up
 * (two-player) Texas Hold'em preflop equity computation.
 *
 * Part of the sicfun poker analytics system's native acceleration layer.
 * Compiled into sicfun_gpu_kernel.dll via nvcc targeting Maxwell sm_50
 * (GTX 960M, CUDA 11.8).
 *
 * This file implements batched equity calculation for all C(52,2)=1326
 * possible hole-card pairs.  Two computation modes are supported:
 *
 *   mode_code == 0  (Exact):
 *     Exhaustive enumeration of all C(48,5) = 1,712,304 possible boards.
 *     One CUDA block per matchup; threads within the block cooperatively
 *     partition the board space and reduce via shared memory.
 *     A "board-major" exact path is also available, which precomputes
 *     per-endpoint scores for each board chunk and then accumulates
 *     matchup results, amortizing evaluation across shared endpoints.
 *
 *   mode_code == 1  (Monte Carlo):
 *     Randomly samples boards using a xorshift64* PRNG seeded per matchup.
 *     Supports two dispatch strategies:
 *       - One thread per matchup (monte_carlo_kernel / _packed)
 *       - One block per matchup with parallel-trial reduction
 *         (monte_carlo_kernel_parallel_trials / _packed_parallel_trials)
 *     Optional optimizations controlled by system properties / env vars:
 *       - Board combo index sampling (precomputed C(48,5) index table)
 *       - Absolute board sampling with rank-pattern lookup tables
 *         (avoids full 7-card evaluation when no flush is possible)
 *       - ReadOnly (__ldg) memory path for range CSR data
 *
 * Three JNI entry point families:
 *   - computeBatch / computeBatchCpuOnly / computeBatchOnDevice
 *       Separate hero/villain ID arrays, jdouble output (legacy API)
 *   - computeBatchPacked / computeBatchPackedOnDevice
 *       Packed 22-bit matchup keys, jfloat output (compact API)
 *   - computeRangeBatchMonteCarloCsr / computeRangeBatchMonteCarloCsrOnDevice
 *       Range-vs-range equity via CSR (compressed sparse row) layout,
 *       probability-weighted aggregation per hero hand
 *
 * Additionally exposes:
 *   - lastEngineCode: reports which engine (CPU/CUDA/fallback) was used
 *   - cudaDeviceCount / cudaDeviceInfo: GPU discovery for the JVM layer
 *
 * Card encoding: 0-51, where card_id = suit * 13 + (rank - 2).
 *   Rank: 2=0 .. A=12 (stored as 2..14 internally).
 *   Suit: 0=clubs, 1=diamonds, 2=hearts, 3=spades.
 *
 * Hand score encoding: 32-bit packed uint32_t.
 *   Bits [31:24] = hand category (0=HighCard .. 8=StraightFlush).
 *   Bits [23:0]  = up to 5 tiebreaker nibbles (4 bits each, rank value).
 *   Higher numeric score beats lower; equal scores are ties.
 *
 * Error status codes (returned to JVM as jint):
 *   0   = success
 *   100 = null array argument
 *   101 = array length mismatch
 *   102 = JNI exception during array access
 *   111 = invalid mode_code
 *   112 = invalid CSR range layout
 *   124 = JNI exception writing results
 *   125 = hole-card ID out of range [0, 1326)
 *   126 = trials <= 0 for Monte Carlo mode
 *   127 = overlapping hero/villain hole cards (shared card)
 *   130 = CUDA device enumeration failure
 *   131 = cudaMalloc failure
 *   132 = cudaMemcpy host->device failure
 *   133 = kernel launch failure
 *   134 = cudaDeviceSynchronize failure (includes TDR timeout)
 *   135 = cudaMemcpy device->host failure
 *   136 = lookup table upload failure
 *   137 = CUDA launch timeout (Windows TDR)
 *   138 = invalid target device index
 *
 * Configuration (JVM system properties / environment variables):
 *   sicfun.gpu.native.engine / sicfun_GPU_NATIVE_ENGINE
 *     "auto" (default), "cpu", or "cuda"
 *   sicfun.gpu.native.cuda.blockSize / sicfun_GPU_CUDA_BLOCK_SIZE
 *     CUDA threads per block (default: 96 MC, 256 exact)
 *   sicfun.gpu.native.cuda.maxChunkMatchups / sicfun_GPU_CUDA_MAX_CHUNK_MATCHUPS
 *     Max matchups per kernel launch (default: 4096 MC, 4 exact)
 *   sicfun.gpu.native.monteCarlo.useBoardCombos
 *     Use precomputed C(48,5) board index table for MC sampling
 *   sicfun.gpu.native.monteCarlo.useRankLookup
 *     Use rank-pattern lookup table to skip full evaluation (default: true)
 *   sicfun.gpu.native.monteCarlo.absoluteBoardSampling
 *     Sample from C(52,5) absolute boards with rejection (default: false)
 *   sicfun.gpu.native.monteCarlo.parallelTrials
 *     Use parallel-trial kernels (1 block per matchup) (default: true)
 *   sicfun.gpu.native.exact.boardMajor
 *     Use board-major exact path (default: false)
 *   sicfun.gpu.native.range.cuda.blockSize
 *     Block size for range CSR kernels (default: 128)
 *   sicfun.gpu.native.range.memoryPath
 *     "global" (default) or "readonly" (__ldg) for range data reads
 *
 * Compile:
 *   nvcc -std=c++17 -O3 -gencode arch=compute_50,code=sm_50
 *        -I"%JAVA_HOME%\include" -I"%JAVA_HOME%\include\win32"
 *        --shared -o sicfun_gpu_kernel.dll HeadsUpGpuNativeBindingsCuda.cu
 */

#include <jni.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#include <cstring>
#include <limits>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(__CUDACC__)
#define HD __host__ __device__
#define HD_FORCE __host__ __device__ __forceinline__
#else
#define HD
#define HD_FORCE inline
#endif

namespace {

/* ---- Constants ----------------------------------------------------------- */

constexpr int kDeckSize = 52;
constexpr int kRanksPerSuit = 13;
constexpr int kMinRankValue = 2;
constexpr int kMaxRankValue = 14;
constexpr int kHoleCardsCount = 1326;
constexpr int kRemainingAfterHoleCards = 48;
constexpr int kBoardCardCount = 5;
constexpr int kExactBoardCount = 1712304;  // C(48,5) -- boards after removing 4 hole cards
constexpr int kAbsoluteBoardCount = 2598960;  // C(52,5) -- all possible 5-card boards from full deck
constexpr int kCpuWorkChunkSize = 8;  /* Work-stealing granularity for CPU multithreaded dispatch. */
constexpr int kIdBits = 11;  /* Bits per hole-card ID in packed matchup key (2^11 = 2048 > 1326). */
constexpr int kIdMask = (1 << kIdBits) - 1;  /* Extraction mask for 11-bit hole-card ID. */
/* Default CUDA launch parameters.  Tuned for GTX 960M (Maxwell, sm_50).
 * Kept conservative to stay under Windows TDR timeout (~2 seconds). */
constexpr int kDefaultCudaThreadsPerBlock = 96;        /* MC: one thread per matchup */
constexpr int kDefaultRangeCudaThreadsPerBlock = 128;   /* Range CSR: one block per hero */
constexpr int kDefaultCudaThreadsPerBlockExact = 256;   /* Exact: one block per matchup */
constexpr int kDefaultCudaMaxChunkMatchups = 4096;      /* MC: matchups per kernel launch */
constexpr int kDefaultCudaMaxChunkMatchupsExact = 4;    /* Exact: matchups per launch (heavy) */
constexpr int kDefaultRangeCudaMaxChunkHeroes = 4096;   /* Range: heroes per kernel launch */
constexpr int kStatusInvalidRangeLayout = 112;  /* Error: malformed CSR offset array. */
constexpr uint64_t kRngMul = 2685821657736338717ULL;  /* xorshift64* output multiplier. */
/* Engine identification codes reported to JVM via lastEngineCode(). */
constexpr jint kEngineUnknown = 0;                    /* Not yet computed or error. */
constexpr jint kEngineCpu = 1;                         /* CPU path was used. */
constexpr jint kEngineCuda = 2;                        /* CUDA path was used. */
constexpr jint kEngineCpuFallbackAfterCudaFailure = 3; /* CUDA failed, fell back to CPU. */
constexpr uint32_t kInvalidScoreSentinel = 0xFFFFFFFFu;
constexpr int kRankPairCount = kRanksPerSuit * kRanksPerSuit;  /* 169 = 13x13 rank-pair combos. */
constexpr int kRankPatternCount = 6188;  // C(13 + 5 - 1, 5) -- distinct 5-card rank multisets

/* ---- Data types ---------------------------------------------------------- */

/* A pair of hole cards identified by their 0-51 card IDs (first < second). */
struct HoleCards {
  uint8_t first;
  uint8_t second;
};

/* Equity result for a single hero-vs-villain matchup. */
struct EquityResultNative {
  double win;       /* Probability hero wins. */
  double tie;       /* Probability of a tie (split pot). */
  double loss;      /* Probability villain wins. */
  double std_error; /* Monte Carlo standard error (0.0 for exact). */
};

/* Engine selection: Auto tries CUDA first, falls back to CPU. */
enum class NativeEngine {
  Auto,
  Cpu,
  Cuda,
};

/* GPU memory access path for range CSR data arrays.
 * ReadOnly uses __ldg() (read-only texture cache) for potentially better L1 hit rate.
 * Global uses standard global memory loads. */
enum class RangeMemoryPath {
  ReadOnly,
  Global,
};

/* ---- CUDA constant memory ------------------------------------------------
 * These arrays are uploaded once per device and persist for the process lifetime.
 * Constant memory is cached in a dedicated 64KB cache on Maxwell, providing
 * broadcast reads when all threads in a warp access the same address. */
__device__ __constant__ uint8_t d_hole_first[kHoleCardsCount];   /* First card ID for each of 1326 hole-card combos. */
__device__ __constant__ uint8_t d_hole_second[kHoleCardsCount];  /* Second card ID for each of 1326 hole-card combos. */
__device__ __constant__ uint8_t d_card_rank[kDeckSize];          /* Precomputed rank (2..14) for each card 0..51. */
__device__ __constant__ uint8_t d_card_suit[kDeckSize];          /* Precomputed suit (0..3) for each card 0..51. */
__device__ __constant__ uint16_t d_card_rank_bit[kDeckSize];     /* Precomputed rank bitmask (1 << (rank-2)) for each card. */

/* Global atomic tracking which engine was used for the most recent batch call. */
std::atomic<jint> g_last_engine_code(kEngineUnknown);

/* ---- JNI helpers --------------------------------------------------------- */

/* Check if a JNI exception is pending and clear it.  Returns true if one was. */
bool check_and_clear_exception(JNIEnv* env) {
  if (!env->ExceptionCheck()) {
    return false;
  }
  env->ExceptionClear();
  return true;
}

/* ---- Card property accessors ---------------------------------------------
 * These use CUDA constant memory on-device for broadcast-cached lookups,
 * and arithmetic on the host (avoiding the need for host-side lookup tables). */

/* Returns the rank (2..14, where 14=Ace) for a card ID 0..51. */
HD_FORCE int card_rank(const int card_id) {
#if defined(__CUDA_ARCH__)
  return static_cast<int>(d_card_rank[card_id]);
#else
  return (card_id % kRanksPerSuit) + kMinRankValue;
#endif
}

/* Returns the suit (0..3) for a card ID 0..51. */
HD_FORCE int card_suit(const int card_id) {
#if defined(__CUDA_ARCH__)
  return static_cast<int>(d_card_suit[card_id]);
#else
  return card_id / kRanksPerSuit;
#endif
}

/* Returns a bitmask with the bit for this card's rank set: 1 << (rank - 2). */
HD_FORCE uint16_t card_rank_bit(const int card_id) {
#if defined(__CUDA_ARCH__)
  return d_card_rank_bit[card_id];
#else
  return static_cast<uint16_t>(1) << ((card_id % kRanksPerSuit));
#endif
}

/* ---- Combinatorial utilities --------------------------------------------- */

/* Compute C(n, k) for small values (k <= 5) using unrolled formulas.
 * Used to map rank multisets to a combinatorial number system index. */
HD_FORCE int choose_small(const int n, const int k) {
  if (k < 0 || k > n) {
    return 0;
  }
  if (k == 0 || k == n) {
    return 1;
  }
  if (k == 1) {
    return n;
  }
  if (k == 2) {
    return (n * (n - 1)) / 2;
  }
  if (k == 3) {
    return (n * (n - 1) * (n - 2)) / 6;
  }
  if (k == 4) {
    return (n * (n - 1) * (n - 2) * (n - 3)) / 24;
  }
  if (k == 5) {
    return (n * (n - 1) * (n - 2) * (n - 3) * (n - 4)) / 120;
  }
  int result = 1;
  const int kk = k < (n - k) ? k : (n - k);
  for (int i = 1; i <= kk; ++i) {
    result = (result * (n - kk + i)) / i;
  }
  return result;
}

/* Map a sorted 5-element rank-offset multiset to a unique index in [0, 6188).
 * Uses the combinatorial number system: sum of C(b_i + i, i+1) for each element.
 * This provides a bijection from the C(13+5-1, 5) possible rank multisets to
 * a contiguous index range, enabling O(1) lookup in precomputed score tables. */
HD_FORCE int rank_multiset5_id_from_offsets(const int sorted_rank_offsets[5]) {
  const int b0 = sorted_rank_offsets[0];
  const int b1 = sorted_rank_offsets[1] + 1;
  const int b2 = sorted_rank_offsets[2] + 2;
  const int b3 = sorted_rank_offsets[3] + 3;
  const int b4 = sorted_rank_offsets[4] + 4;
  return choose_small(b0, 1) +
         choose_small(b1, 2) +
         choose_small(b2, 3) +
         choose_small(b3, 4) +
         choose_small(b4, 5);
}

/* ---- Lookup table construction -------------------------------------------
 * These lazily-initialized singletons build host-side tables on first access.
 * Each has a corresponding ensure_cuda_*_uploaded_for_device() function that
 * copies the table to GPU global/constant memory once per CUDA device. */

/* Returns the canonical table of all 1326 hole-card pairs (first < second).
 * Index i maps to the i-th combination in lexicographic order. */
const std::vector<HoleCards>& hole_cards_lookup() {
  static const std::vector<HoleCards> lookup = [] {
    std::vector<HoleCards> table;
    table.reserve(kHoleCardsCount);
    for (int first = 0; first < (kDeckSize - 1); ++first) {
      for (int second = first + 1; second < kDeckSize; ++second) {
        table.push_back(HoleCards{
            static_cast<uint8_t>(first),
            static_cast<uint8_t>(second),
        });
      }
    }
    return table;
  }();
  return lookup;
}

/* Upload card property lookup tables to CUDA constant memory for the given device.
 * Thread-safe (mutex-guarded) and idempotent per device.  Uploads:
 *   d_hole_first[1326], d_hole_second[1326] -- hole-card pair components
 *   d_card_rank[52], d_card_suit[52]         -- card rank/suit decomposition
 *   d_card_rank_bit[52]                       -- rank bitmask per card */
bool ensure_cuda_lookup_uploaded_for_device(const int device) {
  static std::mutex init_mutex;
  static std::unordered_set<int> initialized_devices;
  std::lock_guard<std::mutex> guard(init_mutex);
  if (initialized_devices.find(device) != initialized_devices.end()) {
    return true;
  }
  const auto& lookup = hole_cards_lookup();
  std::vector<uint8_t> first(kHoleCardsCount);
  std::vector<uint8_t> second(kHoleCardsCount);
  std::vector<uint8_t> card_rank_lookup(kDeckSize);
  std::vector<uint8_t> card_suit_lookup(kDeckSize);
  std::vector<uint16_t> card_rank_bit_lookup(kDeckSize);
  for (int i = 0; i < kHoleCardsCount; ++i) {
    first[static_cast<size_t>(i)] = lookup[static_cast<size_t>(i)].first;
    second[static_cast<size_t>(i)] = lookup[static_cast<size_t>(i)].second;
  }
  for (int card = 0; card < kDeckSize; ++card) {
    const int rank = (card % kRanksPerSuit) + kMinRankValue;
    card_rank_lookup[static_cast<size_t>(card)] = static_cast<uint8_t>(rank);
    card_suit_lookup[static_cast<size_t>(card)] = static_cast<uint8_t>(card / kRanksPerSuit);
    card_rank_bit_lookup[static_cast<size_t>(card)] =
        static_cast<uint16_t>(1) << static_cast<uint16_t>(rank - kMinRankValue);
  }
  cudaError_t err = cudaMemcpyToSymbol(d_hole_first, first.data(), first.size() * sizeof(uint8_t));
  if (err != cudaSuccess) {
    return false;
  }
  err = cudaMemcpyToSymbol(d_hole_second, second.data(), second.size() * sizeof(uint8_t));
  if (err != cudaSuccess) {
    return false;
  }
  err = cudaMemcpyToSymbol(d_card_rank, card_rank_lookup.data(), card_rank_lookup.size() * sizeof(uint8_t));
  if (err != cudaSuccess) {
    return false;
  }
  err = cudaMemcpyToSymbol(d_card_suit, card_suit_lookup.data(), card_suit_lookup.size() * sizeof(uint8_t));
  if (err != cudaSuccess) {
    return false;
  }
  err = cudaMemcpyToSymbol(
      d_card_rank_bit,
      card_rank_bit_lookup.data(),
      card_rank_bit_lookup.size() * sizeof(uint16_t));
  if (err != cudaSuccess) {
    return false;
  }
  initialized_devices.insert(device);
  return true;
}

/* Upload packed hole-card table to GPU global memory.  Each entry packs both
 * card IDs, both rank offsets, and both suits into a single uint32_t:
 *   bits [5:0]   = first card ID (0..51)
 *   bits [11:6]  = second card ID (0..51)
 *   bits [15:12] = first rank offset (0..12)
 *   bits [19:16] = second rank offset (0..12)
 *   bits [21:20] = first suit (0..3)
 *   bits [23:22] = second suit (0..3)
 * Used by the board-major exact path for fast per-endpoint evaluation. */
bool ensure_cuda_hole_cards_uploaded_for_device(const int device, const uint32_t** out_device_ptr) {
  static std::mutex init_mutex;
  static std::unordered_map<int, uint32_t*> per_device_table;

  std::lock_guard<std::mutex> guard(init_mutex);
  const auto found = per_device_table.find(device);
  if (found != per_device_table.end()) {
    *out_device_ptr = found->second;
    return true;
  }

  const auto& lookup = hole_cards_lookup();
  std::vector<uint32_t> packed_pairs(kHoleCardsCount);
  for (int i = 0; i < kHoleCardsCount; ++i) {
    const uint32_t first = static_cast<uint32_t>(lookup[static_cast<size_t>(i)].first);
    const uint32_t second = static_cast<uint32_t>(lookup[static_cast<size_t>(i)].second);
    const uint32_t first_rank = first % static_cast<uint32_t>(kRanksPerSuit);
    const uint32_t second_rank = second % static_cast<uint32_t>(kRanksPerSuit);
    const uint32_t first_suit = first / static_cast<uint32_t>(kRanksPerSuit);
    const uint32_t second_suit = second / static_cast<uint32_t>(kRanksPerSuit);
    packed_pairs[static_cast<size_t>(i)] =
        (first & 0x3Fu) |
        ((second & 0x3Fu) << 6) |
        ((first_rank & 0x0Fu) << 12) |
        ((second_rank & 0x0Fu) << 16) |
        ((first_suit & 0x03u) << 20) |
        ((second_suit & 0x03u) << 22);
  }

  uint32_t* device_table = nullptr;
  const size_t bytes = packed_pairs.size() * sizeof(uint32_t);
  cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&device_table), bytes);
  if (err != cudaSuccess) {
    return false;
  }
  err = cudaMemcpy(device_table, packed_pairs.data(), bytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(device_table);
    return false;
  }

  per_device_table.insert({device, device_table});
  *out_device_ptr = device_table;
  return true;
}

/* Build flat table of all C(48,5) board combinations as 5-byte index tuples.
 * Each board is stored as 5 indices into the 48-card remaining deck (after
 * removing 4 hole cards).  Total size: 1,712,304 * 5 = ~8.2 MB. */
const std::vector<uint8_t>& exact_board_combo_indices() {
  static const std::vector<uint8_t> combos = [] {
    std::vector<uint8_t> out;
    out.reserve(static_cast<size_t>(kExactBoardCount) * static_cast<size_t>(kBoardCardCount));
    for (int a = 0; a <= (kRemainingAfterHoleCards - 5); ++a) {
      for (int b = a + 1; b <= (kRemainingAfterHoleCards - 4); ++b) {
        for (int c = b + 1; c <= (kRemainingAfterHoleCards - 3); ++c) {
          for (int d = c + 1; d <= (kRemainingAfterHoleCards - 2); ++d) {
            for (int e = d + 1; e <= (kRemainingAfterHoleCards - 1); ++e) {
              out.push_back(static_cast<uint8_t>(a));
              out.push_back(static_cast<uint8_t>(b));
              out.push_back(static_cast<uint8_t>(c));
              out.push_back(static_cast<uint8_t>(d));
              out.push_back(static_cast<uint8_t>(e));
            }
          }
        }
      }
    }
    return out;
  }();
  return combos;
}

/* Upload C(48,5) board combo index table to GPU global memory for exact enumeration. */
bool ensure_cuda_exact_board_indices_uploaded_for_device(const int device, const uint8_t** out_device_ptr) {
  static std::mutex init_mutex;
  static std::unordered_map<int, uint8_t*> per_device_table;

  std::lock_guard<std::mutex> guard(init_mutex);
  const auto found = per_device_table.find(device);
  if (found != per_device_table.end()) {
    *out_device_ptr = found->second;
    return true;
  }

  const auto& host_table = exact_board_combo_indices();
  uint8_t* device_table = nullptr;
  const size_t bytes = host_table.size() * sizeof(uint8_t);
  cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&device_table), bytes);
  if (err != cudaSuccess) {
    return false;
  }
  err = cudaMemcpy(device_table, host_table.data(), bytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(device_table);
    return false;
  }

  per_device_table.insert({device, device_table});
  *out_device_ptr = device_table;
  return true;
}

/* Build flat table of all C(52,5) absolute boards (card IDs 0..51).
 * Used for "absolute board sampling" mode where boards are sampled
 * from the full deck and rejected if they overlap with hole cards.
 * Total size: 2,598,960 * 5 = ~12.4 MB. */
const std::vector<uint8_t>& absolute_board_cards() {
  static const std::vector<uint8_t> boards = [] {
    std::vector<uint8_t> out;
    out.reserve(static_cast<size_t>(kAbsoluteBoardCount) * static_cast<size_t>(kBoardCardCount));
    for (int a = 0; a <= (kDeckSize - 5); ++a) {
      for (int b = a + 1; b <= (kDeckSize - 4); ++b) {
        for (int c = b + 1; c <= (kDeckSize - 3); ++c) {
          for (int d = c + 1; d <= (kDeckSize - 2); ++d) {
            for (int e = d + 1; e <= (kDeckSize - 1); ++e) {
              out.push_back(static_cast<uint8_t>(a));
              out.push_back(static_cast<uint8_t>(b));
              out.push_back(static_cast<uint8_t>(c));
              out.push_back(static_cast<uint8_t>(d));
              out.push_back(static_cast<uint8_t>(e));
            }
          }
        }
      }
    }
    return out;
  }();
  return boards;
}

bool ensure_cuda_absolute_board_cards_uploaded_for_device(const int device, const uint8_t** out_device_ptr) {
  static std::mutex init_mutex;
  static std::unordered_map<int, uint8_t*> per_device_table;

  std::lock_guard<std::mutex> guard(init_mutex);
  const auto found = per_device_table.find(device);
  if (found != per_device_table.end()) {
    *out_device_ptr = found->second;
    return true;
  }

  const auto& host_table = absolute_board_cards();
  uint8_t* device_table = nullptr;
  const size_t bytes = host_table.size() * sizeof(uint8_t);
  cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&device_table), bytes);
  if (err != cudaSuccess) {
    return false;
  }
  err = cudaMemcpy(device_table, host_table.data(), bytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(device_table);
    return false;
  }

  per_device_table.insert({device, device_table});
  *out_device_ptr = device_table;
  return true;
}

/* Precomputed metadata for each of the C(52,5) absolute boards.
 * rank_pattern_ids: the combinatorial-number-system index of the board's
 *   sorted rank multiset, enabling O(1) rank-only score lookups.
 * flush_meta: packed byte with dominant suit index (bits [1:0]) and
 *   dominant suit count (bits [4:2]), used for fast flush-possibility checks. */
struct AbsoluteBoardMetadata {
  std::vector<uint16_t> rank_pattern_ids;
  std::vector<uint8_t> flush_meta;
};

const AbsoluteBoardMetadata& absolute_board_metadata() {
  static const AbsoluteBoardMetadata metadata = [] {
    AbsoluteBoardMetadata out;
    const auto& boards = absolute_board_cards();
    out.rank_pattern_ids.resize(static_cast<size_t>(kAbsoluteBoardCount));
    out.flush_meta.resize(static_cast<size_t>(kAbsoluteBoardCount));
    for (int board_idx = 0; board_idx < kAbsoluteBoardCount; ++board_idx) {
      const int base = board_idx * kBoardCardCount;
      uint8_t rank_counts[kMaxRankValue + 1];
      for (int rank = 0; rank <= kMaxRankValue; ++rank) {
        rank_counts[rank] = static_cast<uint8_t>(0);
      }
      uint8_t suit_counts[4] = {0, 0, 0, 0};
      for (int i = 0; i < kBoardCardCount; ++i) {
        const int card = static_cast<int>(boards[static_cast<size_t>(base + i)]);
        const int rank = card_rank(card);
        const int suit = card_suit(card);
        ++rank_counts[rank];
        ++suit_counts[suit];
      }
      int sorted_offsets[5] = {0, 0, 0, 0, 0};
      int sorted_idx = 0;
      for (int rank_offset = 0; rank_offset < kRanksPerSuit; ++rank_offset) {
        const int rank = rank_offset + kMinRankValue;
        const int count = static_cast<int>(rank_counts[rank]);
        for (int copy = 0; copy < count && sorted_idx < 5; ++copy) {
          sorted_offsets[sorted_idx++] = rank_offset;
        }
      }
      int pattern_id = 0;
      if (sorted_idx == 5) {
        const int id = rank_multiset5_id_from_offsets(sorted_offsets);
        if (id >= 0 && id < kRankPatternCount) {
          pattern_id = id;
        }
      }
      int max_suit_count = 0;
      int max_suit_index = 0;
      for (int suit = 0; suit < 4; ++suit) {
        const int count = static_cast<int>(suit_counts[suit]);
        if (count > max_suit_count) {
          max_suit_count = count;
          max_suit_index = suit;
        }
      }
      out.rank_pattern_ids[static_cast<size_t>(board_idx)] = static_cast<uint16_t>(pattern_id);
      out.flush_meta[static_cast<size_t>(board_idx)] =
          static_cast<uint8_t>((max_suit_index & 0x03) | ((max_suit_count & 0x07) << 2));
    }
    return out;
  }();
  return metadata;
}

bool ensure_cuda_absolute_board_metadata_uploaded_for_device(
    const int device,
    const uint16_t** out_rank_pattern_ids,
    const uint8_t** out_flush_meta) {
  static std::mutex init_mutex;
  static std::unordered_map<int, std::pair<uint16_t*, uint8_t*>> per_device_table;

  std::lock_guard<std::mutex> guard(init_mutex);
  const auto found = per_device_table.find(device);
  if (found != per_device_table.end()) {
    *out_rank_pattern_ids = found->second.first;
    *out_flush_meta = found->second.second;
    return true;
  }

  const auto& metadata = absolute_board_metadata();
  uint16_t* d_pattern_ids = nullptr;
  uint8_t* d_flush_meta = nullptr;
  const size_t pattern_bytes = metadata.rank_pattern_ids.size() * sizeof(uint16_t);
  const size_t flush_bytes = metadata.flush_meta.size() * sizeof(uint8_t);
  cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_pattern_ids), pattern_bytes);
  if (err != cudaSuccess) {
    return false;
  }
  err = cudaMalloc(reinterpret_cast<void**>(&d_flush_meta), flush_bytes);
  if (err != cudaSuccess) {
    cudaFree(d_pattern_ids);
    return false;
  }
  err = cudaMemcpy(d_pattern_ids, metadata.rank_pattern_ids.data(), pattern_bytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(d_pattern_ids);
    cudaFree(d_flush_meta);
    return false;
  }
  err = cudaMemcpy(d_flush_meta, metadata.flush_meta.data(), flush_bytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(d_pattern_ids);
    cudaFree(d_flush_meta);
    return false;
  }

  per_device_table.insert({device, {d_pattern_ids, d_flush_meta}});
  *out_rank_pattern_ids = d_pattern_ids;
  *out_flush_meta = d_flush_meta;
  return true;
}

/* ---- Hand evaluation engine -----------------------------------------------
 * Evaluates 7-card poker hands (2 hole + 5 board) using bitmask-based
 * incremental evaluation.  Avoids brute-force C(7,5) subset enumeration
 * by tracking rank/suit bitmasks and count arrays, then determining the
 * best 5-card hand via a priority chain:
 *   StraightFlush(8) > FourOfKind(7) > FullHouse(6) > Flush(5) >
 *   Straight(4) > ThreeOfKind(3) > TwoPair(2) > OnePair(1) > HighCard(0)
 *
 * Key optimization: when no flush is possible for a board (board max suit
 * count + hole suit matches < 5), evaluation can use a precomputed
 * rank-pattern lookup table instead of the full evaluator. */

/* Pack a hand category (0..8) and up to 5 tiebreaker rank values into
 * a single 32-bit score.  Higher score beats lower score. */
HD inline uint32_t encode_score(const int category, const int* tiebreak, const int tiebreak_size) {
  uint32_t score = static_cast<uint32_t>(category) << 24;
  for (int i = 0; i < tiebreak_size && i < 5; ++i) {
    score |= (static_cast<uint32_t>(tiebreak[i] & 0x0F) << (20 - (i * 4)));
  }
  return score;
}

/* Find the index of the highest set bit in a 16-bit mask.
 * Uses platform-specific intrinsics: __clz on CUDA, _BitScanReverse on MSVC,
 * __builtin_clz on GCC/Clang. */
HD_FORCE int highest_bit_index_16(const uint16_t mask) {
  if (mask == 0) {
    return -1;
  }
#if defined(__CUDA_ARCH__)
  return 31 - __clz(static_cast<unsigned int>(mask));
#elif defined(_MSC_VER)
  unsigned long idx;
  _BitScanReverse(&idx, static_cast<unsigned long>(mask));
  return static_cast<int>(idx);
#else
  return 31 - __builtin_clz(static_cast<unsigned int>(mask));
#endif
}

HD_FORCE int popcount_16(const uint16_t mask) {
#if defined(__CUDA_ARCH__)
  return __popc(static_cast<unsigned int>(mask));
#elif defined(_MSC_VER)
  return static_cast<int>(__popcnt16(mask));
#else
  return __builtin_popcount(static_cast<unsigned int>(mask));
#endif
}

HD_FORCE int highest_rank_from_mask(const uint16_t mask) {
  const int bit = highest_bit_index_16(mask);
  return bit >= 0 ? (bit + kMinRankValue) : 0;
}

/* Detect a straight from a 13-bit rank bitmask.  Returns the high card
 * of the straight (5..14), or 0 if no straight exists.
 * Detects both regular straights (5 consecutive bits) and the wheel (A-2-3-4-5). */
HD inline int straight_high_from_rank_mask(const uint16_t rank_mask) {
  const uint16_t run = static_cast<uint16_t>(
      rank_mask &
      static_cast<uint16_t>(rank_mask >> 1) &
      static_cast<uint16_t>(rank_mask >> 2) &
      static_cast<uint16_t>(rank_mask >> 3) &
      static_cast<uint16_t>(rank_mask >> 4));
  const int start_bit = highest_bit_index_16(run);
  if (start_bit >= 0) {
    return start_bit + 6;
  }
  const uint16_t wheel_mask =
      (static_cast<uint16_t>(1) << 12) | (static_cast<uint16_t>(1) << 3) |
      (static_cast<uint16_t>(1) << 2) | (static_cast<uint16_t>(1) << 1) |
      (static_cast<uint16_t>(1) << 0);
  return ((rank_mask & wheel_mask) == wheel_mask) ? 5 : 0;
}

/* Full 7-card hand evaluation from precomputed bitmasks.
 * Given rank/pair/trip/quad masks plus per-suit counts and rank masks,
 * determines the best 5-card hand and returns its packed 32-bit score.
 * This is the core evaluator used by all code paths. */
HD uint32_t evaluate7_score_from_masks(
    const uint16_t rank_mask,
    const uint16_t pair_mask,
    const uint16_t trip_mask,
    const uint16_t quad_mask,
    const uint8_t suit_counts[4],
    const uint16_t suit_rank_mask[4]) {
  // Branchless flush suit detection: scan all 4 suits, last match wins (at most one can have >=5)
  int flush_suit = -1;
  flush_suit = (suit_counts[0] >= 5) ? 0 : flush_suit;
  flush_suit = (suit_counts[1] >= 5) ? 1 : flush_suit;
  flush_suit = (suit_counts[2] >= 5) ? 2 : flush_suit;
  flush_suit = (suit_counts[3] >= 5) ? 3 : flush_suit;

  int tiebreak[5] = {0, 0, 0, 0, 0};
  if (flush_suit >= 0) {
    const int straight_flush_high = straight_high_from_rank_mask(suit_rank_mask[flush_suit]);
    if (straight_flush_high > 0) {
      tiebreak[0] = straight_flush_high;
      return encode_score(8, tiebreak, 1);  // StraightFlush
    }
  }

  if (quad_mask != 0) {
    const int quad_rank = highest_rank_from_mask(quad_mask);
    const uint16_t quad_bit = static_cast<uint16_t>(1) << static_cast<uint16_t>(quad_rank - kMinRankValue);
    const int kicker = highest_rank_from_mask(static_cast<uint16_t>(rank_mask & static_cast<uint16_t>(~quad_bit)));
    tiebreak[0] = quad_rank;
    tiebreak[1] = kicker;
    return encode_score(7, tiebreak, 2);  // FourOfKind
  }

  if (trip_mask != 0) {
    const int top_trip_rank = highest_rank_from_mask(trip_mask);
    const uint16_t top_trip_bit =
        static_cast<uint16_t>(1) << static_cast<uint16_t>(top_trip_rank - kMinRankValue);
    const uint16_t other_trips = static_cast<uint16_t>(trip_mask & static_cast<uint16_t>(~top_trip_bit));
    const uint16_t full_house_candidates =
        other_trips != 0 ? other_trips : static_cast<uint16_t>(pair_mask & static_cast<uint16_t>(~top_trip_bit));
    if (full_house_candidates != 0) {
      tiebreak[0] = top_trip_rank;
      tiebreak[1] = highest_rank_from_mask(full_house_candidates);
      return encode_score(6, tiebreak, 2);  // FullHouse
    }
  }

  if (flush_suit >= 0) {
    uint16_t mask = suit_rank_mask[flush_suit];
    for (int idx = 0; idx < 5; ++idx) {
      const int rank = highest_rank_from_mask(mask);
      tiebreak[idx] = rank;
      const uint16_t bit = static_cast<uint16_t>(1) << static_cast<uint16_t>(rank - kMinRankValue);
      mask = static_cast<uint16_t>(mask & static_cast<uint16_t>(~bit));
    }
    return encode_score(5, tiebreak, 5);  // Flush
  }

  const int straight_high = straight_high_from_rank_mask(rank_mask);
  if (straight_high > 0) {
    tiebreak[0] = straight_high;
    return encode_score(4, tiebreak, 1);  // Straight
  }

  if (trip_mask != 0) {
    const int trip_rank = highest_rank_from_mask(trip_mask);
    const uint16_t trip_bit = static_cast<uint16_t>(1) << static_cast<uint16_t>(trip_rank - kMinRankValue);
    uint16_t kick_mask = static_cast<uint16_t>(rank_mask & static_cast<uint16_t>(~trip_bit));
    tiebreak[0] = trip_rank;
    tiebreak[1] = highest_rank_from_mask(kick_mask);
    kick_mask = static_cast<uint16_t>(
        kick_mask & static_cast<uint16_t>(~(static_cast<uint16_t>(1) << static_cast<uint16_t>(tiebreak[1] - kMinRankValue))));
    tiebreak[2] = highest_rank_from_mask(kick_mask);
    return encode_score(3, tiebreak, 3);  // ThreeOfKind
  }

  if (popcount_16(pair_mask) >= 2) {
    const int high_pair = highest_rank_from_mask(pair_mask);
    const uint16_t high_pair_bit = static_cast<uint16_t>(1) << static_cast<uint16_t>(high_pair - kMinRankValue);
    const uint16_t remaining_pairs = static_cast<uint16_t>(pair_mask & static_cast<uint16_t>(~high_pair_bit));
    const int low_pair = highest_rank_from_mask(remaining_pairs);
    const uint16_t low_pair_bit = static_cast<uint16_t>(1) << static_cast<uint16_t>(low_pair - kMinRankValue);
    const int kicker = highest_rank_from_mask(
        static_cast<uint16_t>(rank_mask & static_cast<uint16_t>(~(high_pair_bit | low_pair_bit))));
    tiebreak[0] = high_pair;
    tiebreak[1] = low_pair;
    tiebreak[2] = kicker;
    return encode_score(2, tiebreak, 3);  // TwoPair
  }

  if (pair_mask != 0) {
    const int pair_rank = highest_rank_from_mask(pair_mask);
    const uint16_t pair_bit = static_cast<uint16_t>(1) << static_cast<uint16_t>(pair_rank - kMinRankValue);
    uint16_t kick_mask = static_cast<uint16_t>(rank_mask & static_cast<uint16_t>(~pair_bit));
    tiebreak[0] = pair_rank;
    tiebreak[1] = highest_rank_from_mask(kick_mask);
    kick_mask = static_cast<uint16_t>(
        kick_mask & static_cast<uint16_t>(~(static_cast<uint16_t>(1) << static_cast<uint16_t>(tiebreak[1] - kMinRankValue))));
    tiebreak[2] = highest_rank_from_mask(kick_mask);
    kick_mask = static_cast<uint16_t>(
        kick_mask & static_cast<uint16_t>(~(static_cast<uint16_t>(1) << static_cast<uint16_t>(tiebreak[2] - kMinRankValue))));
    tiebreak[3] = highest_rank_from_mask(kick_mask);
    return encode_score(1, tiebreak, 4);  // OnePair
  }

  uint16_t mask = rank_mask;
  for (int idx = 0; idx < 5; ++idx) {
    const int rank = highest_rank_from_mask(mask);
    tiebreak[idx] = rank;
    const uint16_t bit = static_cast<uint16_t>(1) << static_cast<uint16_t>(rank - kMinRankValue);
    mask = static_cast<uint16_t>(mask & static_cast<uint16_t>(~bit));
  }
  return encode_score(0, tiebreak, 5);  // HighCard
}

/* Rank-only evaluation: same as evaluate7_score_from_masks but without
 * flush/straight-flush detection.  Used when we already know no flush is
 * possible, saving the suit-related computation. */
HD uint32_t evaluate7_score_rank_only(
    const uint16_t rank_mask,
    const uint16_t pair_mask,
    const uint16_t trip_mask,
    const uint16_t quad_mask) {
  int tiebreak[5] = {0, 0, 0, 0, 0};

  if (quad_mask != 0) {
    const int quad_rank = highest_rank_from_mask(quad_mask);
    const uint16_t quad_bit = static_cast<uint16_t>(1) << static_cast<uint16_t>(quad_rank - kMinRankValue);
    const int kicker = highest_rank_from_mask(static_cast<uint16_t>(rank_mask & static_cast<uint16_t>(~quad_bit)));
    tiebreak[0] = quad_rank;
    tiebreak[1] = kicker;
    return encode_score(7, tiebreak, 2);  // FourOfKind
  }

  if (trip_mask != 0) {
    const int top_trip_rank = highest_rank_from_mask(trip_mask);
    const uint16_t top_trip_bit =
        static_cast<uint16_t>(1) << static_cast<uint16_t>(top_trip_rank - kMinRankValue);
    const uint16_t other_trips = static_cast<uint16_t>(trip_mask & static_cast<uint16_t>(~top_trip_bit));
    const uint16_t full_house_candidates =
        other_trips != 0 ? other_trips : static_cast<uint16_t>(pair_mask & static_cast<uint16_t>(~top_trip_bit));
    if (full_house_candidates != 0) {
      tiebreak[0] = top_trip_rank;
      tiebreak[1] = highest_rank_from_mask(full_house_candidates);
      return encode_score(6, tiebreak, 2);  // FullHouse
    }
  }

  const int straight_high = straight_high_from_rank_mask(rank_mask);
  if (straight_high > 0) {
    tiebreak[0] = straight_high;
    return encode_score(4, tiebreak, 1);  // Straight
  }

  if (trip_mask != 0) {
    const int trip_rank = highest_rank_from_mask(trip_mask);
    const uint16_t trip_bit = static_cast<uint16_t>(1) << static_cast<uint16_t>(trip_rank - kMinRankValue);
    uint16_t kick_mask = static_cast<uint16_t>(rank_mask & static_cast<uint16_t>(~trip_bit));
    tiebreak[0] = trip_rank;
    tiebreak[1] = highest_rank_from_mask(kick_mask);
    kick_mask = static_cast<uint16_t>(
        kick_mask & static_cast<uint16_t>(~(static_cast<uint16_t>(1) << static_cast<uint16_t>(tiebreak[1] - kMinRankValue))));
    tiebreak[2] = highest_rank_from_mask(kick_mask);
    return encode_score(3, tiebreak, 3);  // ThreeOfKind
  }

  if (popcount_16(pair_mask) >= 2) {
    const int high_pair = highest_rank_from_mask(pair_mask);
    const uint16_t high_pair_bit = static_cast<uint16_t>(1) << static_cast<uint16_t>(high_pair - kMinRankValue);
    const uint16_t remaining_pairs = static_cast<uint16_t>(pair_mask & static_cast<uint16_t>(~high_pair_bit));
    const int low_pair = highest_rank_from_mask(remaining_pairs);
    const uint16_t low_pair_bit = static_cast<uint16_t>(1) << static_cast<uint16_t>(low_pair - kMinRankValue);
    const int kicker = highest_rank_from_mask(
        static_cast<uint16_t>(rank_mask & static_cast<uint16_t>(~(high_pair_bit | low_pair_bit))));
    tiebreak[0] = high_pair;
    tiebreak[1] = low_pair;
    tiebreak[2] = kicker;
    return encode_score(2, tiebreak, 3);  // TwoPair
  }

  if (pair_mask != 0) {
    const int pair_rank = highest_rank_from_mask(pair_mask);
    const uint16_t pair_bit = static_cast<uint16_t>(1) << static_cast<uint16_t>(pair_rank - kMinRankValue);
    uint16_t kick_mask = static_cast<uint16_t>(rank_mask & static_cast<uint16_t>(~pair_bit));
    tiebreak[0] = pair_rank;
    tiebreak[1] = highest_rank_from_mask(kick_mask);
    kick_mask = static_cast<uint16_t>(
        kick_mask & static_cast<uint16_t>(~(static_cast<uint16_t>(1) << static_cast<uint16_t>(tiebreak[1] - kMinRankValue))));
    tiebreak[2] = highest_rank_from_mask(kick_mask);
    kick_mask = static_cast<uint16_t>(
        kick_mask & static_cast<uint16_t>(~(static_cast<uint16_t>(1) << static_cast<uint16_t>(tiebreak[2] - kMinRankValue))));
    tiebreak[3] = highest_rank_from_mask(kick_mask);
    return encode_score(1, tiebreak, 4);  // OnePair
  }

  uint16_t mask = rank_mask;
  for (int idx = 0; idx < 5; ++idx) {
    const int rank = highest_rank_from_mask(mask);
    tiebreak[idx] = rank;
    const uint16_t bit = static_cast<uint16_t>(1) << static_cast<uint16_t>(rank - kMinRankValue);
    mask = static_cast<uint16_t>(mask & static_cast<uint16_t>(~bit));
  }
  return encode_score(0, tiebreak, 5);  // HighCard
}

/* Evaluate from rank/suit count arrays: builds pair/trip/quad masks from
 * rank_counts, then delegates to evaluate7_score_from_masks. */
HD uint32_t evaluate7_score_from_state(
    const uint8_t rank_counts[kMaxRankValue + 1],
    const uint8_t suit_counts[4],
    const uint16_t rank_mask,
    const uint16_t suit_rank_mask[4]) {
  uint16_t pair_mask = 0;
  uint16_t trip_mask = 0;
  uint16_t quad_mask = 0;
  for (int rank = kMinRankValue; rank <= kMaxRankValue; ++rank) {
    const uint8_t count = rank_counts[rank];
    const uint16_t bit = static_cast<uint16_t>(1) << static_cast<uint16_t>(rank - kMinRankValue);
    const uint16_t pair_bit = static_cast<uint16_t>(-(count >= 2)) & bit;
    const uint16_t trip_bit = static_cast<uint16_t>(-(count >= 3)) & bit;
    const uint16_t quad_bit = static_cast<uint16_t>(-(count == 4)) & bit;
    pair_mask = static_cast<uint16_t>(pair_mask | pair_bit);
    trip_mask = static_cast<uint16_t>(trip_mask | trip_bit);
    quad_mask = static_cast<uint16_t>(quad_mask | quad_bit);
  }
  return evaluate7_score_from_masks(
      rank_mask,
      pair_mask,
      trip_mask,
      quad_mask,
      suit_counts,
      suit_rank_mask);
}

/* Convenience: evaluate a 7-card hand from an array of card IDs.
 * Builds all bitmasks from scratch. */
HD uint32_t evaluate7_score(const int cards[7]) {
  uint8_t rank_counts[kMaxRankValue + 1];
  for (int rank = 0; rank <= kMaxRankValue; ++rank) {
    rank_counts[rank] = static_cast<uint8_t>(0);
  }
  uint8_t suit_counts[4] = {0, 0, 0, 0};
  uint16_t rank_mask = 0;
  uint16_t suit_rank_mask[4] = {0, 0, 0, 0};

  for (int i = 0; i < 7; ++i) {
    const int rank = card_rank(cards[i]);
    const int suit = card_suit(cards[i]);
    ++rank_counts[rank];
    ++suit_counts[suit];
    const uint16_t bit = card_rank_bit(cards[i]);
    rank_mask |= bit;
    suit_rank_mask[suit] |= bit;
  }
  return evaluate7_score_from_state(rank_counts, suit_counts, rank_mask, suit_rank_mask);
}

/* ---- Rank-pattern lookup table -------------------------------------------
 * Precomputes the rank-only hand score for every combination of:
 *   - Board rank pattern (6188 distinct multisets)
 *   - Hole-card rank pair (13 x 13 = 169 combinations)
 * Total table size: 6188 * 169 = 1,045,772 entries (~4 MB of uint32_t).
 *
 * When a board has no flush possibility, looking up the score from this
 * table is much faster than running the full evaluator.  This optimization
 * is critical for GPU throughput since the evaluator has many branches. */
const std::vector<uint32_t>& rank_pattern_rankpair_scores() {
  static const std::vector<uint32_t> table = [] {
    std::vector<uint32_t> out(
        static_cast<size_t>(kRankPatternCount) * static_cast<size_t>(kRankPairCount),
        0U);
    std::vector<uint8_t> seen(static_cast<size_t>(kRankPatternCount), static_cast<uint8_t>(0));
    for (int r0 = 0; r0 < kRanksPerSuit; ++r0) {
      for (int r1 = r0; r1 < kRanksPerSuit; ++r1) {
        for (int r2 = r1; r2 < kRanksPerSuit; ++r2) {
          for (int r3 = r2; r3 < kRanksPerSuit; ++r3) {
            for (int r4 = r3; r4 < kRanksPerSuit; ++r4) {
              const int offsets[5] = {r0, r1, r2, r3, r4};
              const int pattern_id = rank_multiset5_id_from_offsets(offsets);
              if (pattern_id < 0 || pattern_id >= kRankPatternCount) {
                continue;
              }
              seen[static_cast<size_t>(pattern_id)] = static_cast<uint8_t>(1);

              uint8_t board_rank_counts[kMaxRankValue + 1];
              for (int rank = 0; rank <= kMaxRankValue; ++rank) {
                board_rank_counts[rank] = static_cast<uint8_t>(0);
              }
              ++board_rank_counts[r0 + kMinRankValue];
              ++board_rank_counts[r1 + kMinRankValue];
              ++board_rank_counts[r2 + kMinRankValue];
              ++board_rank_counts[r3 + kMinRankValue];
              ++board_rank_counts[r4 + kMinRankValue];

              uint16_t board_rank_mask = 0;
              uint16_t board_pair_mask = 0;
              uint16_t board_trip_mask = 0;
              uint16_t board_quad_mask = 0;
              for (int rank = kMinRankValue; rank <= kMaxRankValue; ++rank) {
                const uint8_t count = board_rank_counts[rank];
                if (count == 0) {
                  continue;
                }
                const uint16_t bit =
                    static_cast<uint16_t>(1) << static_cast<uint16_t>(rank - kMinRankValue);
                board_rank_mask = static_cast<uint16_t>(board_rank_mask | bit);
                if (count >= 2) {
                  board_pair_mask = static_cast<uint16_t>(board_pair_mask | bit);
                }
                if (count >= 3) {
                  board_trip_mask = static_cast<uint16_t>(board_trip_mask | bit);
                }
                if (count == 4) {
                  board_quad_mask = static_cast<uint16_t>(board_quad_mask | bit);
                }
              }

              const size_t base =
                  static_cast<size_t>(pattern_id) * static_cast<size_t>(kRankPairCount);
              for (int first_rank_offset = 0; first_rank_offset < kRanksPerSuit; ++first_rank_offset) {
                for (int second_rank_offset = 0; second_rank_offset < kRanksPerSuit; ++second_rank_offset) {
                  const int first_rank = first_rank_offset + kMinRankValue;
                  const int second_rank = second_rank_offset + kMinRankValue;
                  const uint16_t first_bit =
                      static_cast<uint16_t>(1) << static_cast<uint16_t>(first_rank_offset);
                  const uint16_t second_bit =
                      static_cast<uint16_t>(1) << static_cast<uint16_t>(second_rank_offset);
                  const uint16_t rank_mask =
                      static_cast<uint16_t>(board_rank_mask | first_bit | second_bit);
                  uint16_t pair_mask = board_pair_mask;
                  uint16_t trip_mask = board_trip_mask;
                  uint16_t quad_mask = board_quad_mask;
                  if (first_rank == second_rank) {
                    const int updated = static_cast<int>(board_rank_counts[first_rank]) + 2;
                    const uint16_t bit = first_bit;
                    pair_mask = static_cast<uint16_t>(
                        (pair_mask & static_cast<uint16_t>(~bit)) | (updated >= 2 ? bit : 0));
                    trip_mask = static_cast<uint16_t>(
                        (trip_mask & static_cast<uint16_t>(~bit)) | (updated >= 3 ? bit : 0));
                    quad_mask = static_cast<uint16_t>(
                        (quad_mask & static_cast<uint16_t>(~bit)) | (updated == 4 ? bit : 0));
                  } else {
                    const int updated_first = static_cast<int>(board_rank_counts[first_rank]) + 1;
                    pair_mask = static_cast<uint16_t>(
                        (pair_mask & static_cast<uint16_t>(~first_bit)) | (updated_first >= 2 ? first_bit : 0));
                    trip_mask = static_cast<uint16_t>(
                        (trip_mask & static_cast<uint16_t>(~first_bit)) | (updated_first >= 3 ? first_bit : 0));
                    quad_mask = static_cast<uint16_t>(
                        (quad_mask & static_cast<uint16_t>(~first_bit)) | (updated_first == 4 ? first_bit : 0));

                    const int updated_second = static_cast<int>(board_rank_counts[second_rank]) + 1;
                    pair_mask = static_cast<uint16_t>(
                        (pair_mask & static_cast<uint16_t>(~second_bit)) | (updated_second >= 2 ? second_bit : 0));
                    trip_mask = static_cast<uint16_t>(
                        (trip_mask & static_cast<uint16_t>(~second_bit)) | (updated_second >= 3 ? second_bit : 0));
                    quad_mask = static_cast<uint16_t>(
                        (quad_mask & static_cast<uint16_t>(~second_bit)) | (updated_second == 4 ? second_bit : 0));
                  }
                  out[base + static_cast<size_t>(first_rank_offset * kRanksPerSuit + second_rank_offset)] =
                      evaluate7_score_rank_only(rank_mask, pair_mask, trip_mask, quad_mask);
                }
              }
            }
          }
        }
      }
    }
    return out;
  }();
  return table;
}

bool ensure_cuda_rank_pattern_scores_uploaded_for_device(const int device, const uint32_t** out_device_ptr) {
  static std::mutex init_mutex;
  static std::unordered_map<int, uint32_t*> per_device_table;

  std::lock_guard<std::mutex> guard(init_mutex);
  const auto found = per_device_table.find(device);
  if (found != per_device_table.end()) {
    *out_device_ptr = found->second;
    return true;
  }

  const auto& host_table = rank_pattern_rankpair_scores();
  uint32_t* device_table = nullptr;
  const size_t bytes = host_table.size() * sizeof(uint32_t);
  cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&device_table), bytes);
  if (err != cudaSuccess) {
    return false;
  }
  err = cudaMemcpy(device_table, host_table.data(), bytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(device_table);
    return false;
  }

  per_device_table.insert({device, device_table});
  *out_device_ptr = device_table;
  return true;
}

/* ---- Board / deck utilities ---------------------------------------------- */

/* Build the 48-card remaining deck after removing 4 hole cards.
 * Uses a 64-bit dead-card bitmask for O(52) branchless filtering. */
HD void fill_remaining_deck(
    const int hero_first,
    const int hero_second,
    const int villain_first,
    const int villain_second,
    uint8_t remaining[kRemainingAfterHoleCards]) {
  const uint64_t dead_mask =
      (1ULL << hero_first) | (1ULL << hero_second) |
      (1ULL << villain_first) | (1ULL << villain_second);
  int idx = 0;
  for (int card = 0; card < kDeckSize; ++card) {
    if (!((dead_mask >> card) & 1ULL)) {
      remaining[idx++] = static_cast<uint8_t>(card);
    }
  }
}

/* Evaluate a 7-card hand (board + hole cards) using precomputed board state.
 * Incrementally updates the board's bitmasks with the two hole cards,
 * avoiding recomputation of the board's contribution.  This is the hot path
 * for both exact and Monte Carlo equity computation. */
HD_FORCE uint32_t evaluate_with_board_state(
    const uint8_t board_rank_counts[kMaxRankValue + 1],
    const uint8_t board_suit_counts[4],
    const uint16_t board_rank_mask,
    const uint16_t board_suit_rank_mask[4],
    const uint16_t board_pair_mask,
    const uint16_t board_trip_mask,
    const uint16_t board_quad_mask,
    const int hole_first,
    const int hole_second) {
  const int first_rank = card_rank(hole_first);
  const int first_suit = card_suit(hole_first);
  const uint16_t first_bit = card_rank_bit(hole_first);
  const int second_rank = card_rank(hole_second);
  const int second_suit = card_suit(hole_second);
  const uint16_t second_bit = card_rank_bit(hole_second);

  uint8_t suit_counts[4] = {
      board_suit_counts[0],
      board_suit_counts[1],
      board_suit_counts[2],
      board_suit_counts[3],
  };
  uint16_t suit_rank_mask[4] = {
      board_suit_rank_mask[0],
      board_suit_rank_mask[1],
      board_suit_rank_mask[2],
      board_suit_rank_mask[3],
  };
  ++suit_counts[first_suit];
  suit_rank_mask[first_suit] = static_cast<uint16_t>(suit_rank_mask[first_suit] | first_bit);
  ++suit_counts[second_suit];
  suit_rank_mask[second_suit] = static_cast<uint16_t>(suit_rank_mask[second_suit] | second_bit);

  const uint16_t rank_mask = static_cast<uint16_t>(board_rank_mask | first_bit | second_bit);
  uint16_t pair_mask = board_pair_mask;
  uint16_t trip_mask = board_trip_mask;
  uint16_t quad_mask = board_quad_mask;

  if (first_rank == second_rank) {
    const int updated = static_cast<int>(board_rank_counts[first_rank]) + 2;
    const uint16_t bit = first_bit;
    pair_mask = static_cast<uint16_t>(
        (pair_mask & static_cast<uint16_t>(~bit)) | (updated >= 2 ? bit : 0));
    trip_mask = static_cast<uint16_t>(
        (trip_mask & static_cast<uint16_t>(~bit)) | (updated >= 3 ? bit : 0));
    quad_mask = static_cast<uint16_t>(
        (quad_mask & static_cast<uint16_t>(~bit)) | (updated == 4 ? bit : 0));
  } else {
    const int updated_first = static_cast<int>(board_rank_counts[first_rank]) + 1;
    pair_mask = static_cast<uint16_t>(
        (pair_mask & static_cast<uint16_t>(~first_bit)) | (updated_first >= 2 ? first_bit : 0));
    trip_mask = static_cast<uint16_t>(
        (trip_mask & static_cast<uint16_t>(~first_bit)) | (updated_first >= 3 ? first_bit : 0));
    quad_mask = static_cast<uint16_t>(
        (quad_mask & static_cast<uint16_t>(~first_bit)) | (updated_first == 4 ? first_bit : 0));

    const int updated_second = static_cast<int>(board_rank_counts[second_rank]) + 1;
    pair_mask = static_cast<uint16_t>(
        (pair_mask & static_cast<uint16_t>(~second_bit)) | (updated_second >= 2 ? second_bit : 0));
    trip_mask = static_cast<uint16_t>(
        (trip_mask & static_cast<uint16_t>(~second_bit)) | (updated_second >= 3 ? second_bit : 0));
    quad_mask = static_cast<uint16_t>(
        (quad_mask & static_cast<uint16_t>(~second_bit)) | (updated_second == 4 ? second_bit : 0));
  }

  return evaluate7_score_from_masks(
      rank_mask,
      pair_mask,
      trip_mask,
      quad_mask,
      suit_counts,
      suit_rank_mask);
}

/* Evaluate with rank-pattern lookup optimization: if no flush is possible,
 * uses the precomputed rank_pattern_scores table for O(1) lookup.
 * Falls back to full evaluate_with_board_state when a flush is possible. */
HD_FORCE uint32_t evaluate_with_board_state_lookup(
    const uint8_t board_rank_counts[kMaxRankValue + 1],
    const uint8_t board_suit_counts[4],
    const uint16_t board_rank_mask,
    const uint16_t board_suit_rank_mask[4],
    const uint16_t board_pair_mask,
    const uint16_t board_trip_mask,
    const uint16_t board_quad_mask,
    const int board_max_suit_count,
    const int board_max_suit_index,
    const int rank_pattern_id,
    const uint32_t* rank_pattern_scores,
    const int hole_first,
    const int hole_second) {
  if (rank_pattern_scores != nullptr &&
      rank_pattern_id >= 0 &&
      rank_pattern_id < kRankPatternCount) {
    const int first_rank_offset = card_rank(hole_first) - kMinRankValue;
    const int second_rank_offset = card_rank(hole_second) - kMinRankValue;
    const int first_suit = card_suit(hole_first);
    const int second_suit = card_suit(hole_second);
    bool flush_possible = false;
    if (board_max_suit_count >= 5) {
      flush_possible = true;
    } else if (board_max_suit_count == 4) {
      flush_possible = first_suit == board_max_suit_index || second_suit == board_max_suit_index;
    } else if (board_max_suit_count == 3) {
      flush_possible = first_suit == board_max_suit_index && second_suit == board_max_suit_index;
    }
    if (!flush_possible) {
      const size_t rank_lookup_base =
          static_cast<size_t>(rank_pattern_id) * static_cast<size_t>(kRankPairCount);
      const size_t rank_pair_idx =
          static_cast<size_t>(first_rank_offset * kRanksPerSuit + second_rank_offset);
      return rank_pattern_scores[rank_lookup_base + rank_pair_idx];
    }
  }

  return evaluate_with_board_state(
      board_rank_counts,
      board_suit_counts,
      board_rank_mask,
      board_suit_rank_mask,
      board_pair_mask,
      board_trip_mask,
      board_quad_mask,
      hole_first,
      hole_second);
}

/* Quick flush-possibility check using precomputed board metadata.
 * A flush requires 5+ cards of the same suit among 7 cards total.
 * With board max suit count known, we just check how many hole cards
 * match that suit: (board_max_suit_count + hole_matches) >= 5. */
HD_FORCE bool flush_possible_for_board_meta(
    const int board_max_suit_count,
    const int board_max_suit_index,
    const int hole_first,
    const int hole_second) {
  const int first_match = (card_suit(hole_first) == board_max_suit_index);
  const int second_match = (card_suit(hole_second) == board_max_suit_index);
  const int hole_matches = first_match + second_match;
  // Need (board_max_suit_count + hole_matches) >= 5 for a flush to be possible
  return (board_max_suit_count + hole_matches) >= 5;
}

HD_FORCE uint32_t rank_lookup_score_for_pattern(
    const int rank_pattern_id,
    const int hole_first,
    const int hole_second,
    const uint32_t* rank_pattern_scores) {
  if (rank_pattern_scores == nullptr || rank_pattern_id < 0 || rank_pattern_id >= kRankPatternCount) {
    return 0;
  }
  const int first_rank_offset = card_rank(hole_first) - kMinRankValue;
  const int second_rank_offset = card_rank(hole_second) - kMinRankValue;
  const size_t rank_lookup_base =
      static_cast<size_t>(rank_pattern_id) * static_cast<size_t>(kRankPairCount);
  const size_t rank_pair_idx =
      static_cast<size_t>(first_rank_offset * kRanksPerSuit + second_rank_offset);
  return rank_pattern_scores[rank_lookup_base + rank_pair_idx];
}

HD_FORCE int compare_showdown_rank_lookup_only(
    const int hero_first,
    const int hero_second,
    const int villain_first,
    const int villain_second,
    const int rank_pattern_id,
    const uint32_t* rank_pattern_scores) {
  const uint32_t hero_score =
      rank_lookup_score_for_pattern(rank_pattern_id, hero_first, hero_second, rank_pattern_scores);
  const uint32_t villain_score =
      rank_lookup_score_for_pattern(rank_pattern_id, villain_first, villain_second, rank_pattern_scores);
  return (hero_score > villain_score) - (hero_score < villain_score);
}

/* Compare hero vs villain on a given 5-card board.  Returns +1 (hero wins),
 * 0 (tie), or -1 (villain wins).  Builds full board state and evaluates
 * both players, optionally using the rank-pattern lookup table. */
HD inline int compare_showdown(
    const int hero_first,
    const int hero_second,
    const int villain_first,
    const int villain_second,
    const uint8_t board[kBoardCardCount],
    const uint32_t* rank_pattern_scores = nullptr) {
  uint8_t board_rank_counts[kMaxRankValue + 1];
  for (int rank = 0; rank <= kMaxRankValue; ++rank) {
    board_rank_counts[rank] = static_cast<uint8_t>(0);
  }
  uint8_t board_suit_counts[4] = {0, 0, 0, 0};
  uint16_t board_rank_mask = 0;
  uint16_t board_suit_rank_mask[4] = {0, 0, 0, 0};
  uint16_t board_pair_mask = 0;
  uint16_t board_trip_mask = 0;
  uint16_t board_quad_mask = 0;
  int board_max_suit_count = 0;
  int board_max_suit_index = 0;
  for (int i = 0; i < kBoardCardCount; ++i) {
    const int card = static_cast<int>(board[i]);
    const int rank = card_rank(card);
    const int suit = card_suit(card);
    const uint8_t updated = static_cast<uint8_t>(board_rank_counts[rank] + 1);
    board_rank_counts[rank] = updated;
    ++board_suit_counts[suit];
    const uint16_t bit = card_rank_bit(card);
    board_rank_mask = static_cast<uint16_t>(board_rank_mask | bit);
    board_suit_rank_mask[suit] = static_cast<uint16_t>(board_suit_rank_mask[suit] | bit);
    if (updated >= 2) {
      board_pair_mask = static_cast<uint16_t>(board_pair_mask | bit);
    }
    if (updated >= 3) {
      board_trip_mask = static_cast<uint16_t>(board_trip_mask | bit);
    }
    if (updated == 4) {
      board_quad_mask = static_cast<uint16_t>(board_quad_mask | bit);
    }
  }
  for (int suit = 0; suit < 4; ++suit) {
    const int count = static_cast<int>(board_suit_counts[suit]);
    const int is_max = (count > board_max_suit_count);
    board_max_suit_index = is_max ? suit : board_max_suit_index;
    board_max_suit_count = is_max ? count : board_max_suit_count;
  }

  int sorted_offsets[5] = {0, 0, 0, 0, 0};
  int sorted_idx = 0;
  for (int rank_offset = 0; rank_offset < kRanksPerSuit; ++rank_offset) {
    const int rank = rank_offset + kMinRankValue;
    const int count = static_cast<int>(board_rank_counts[rank]);
    for (int copy = 0; copy < count && sorted_idx < 5; ++copy) {
      sorted_offsets[sorted_idx++] = rank_offset;
    }
  }
  const int rank_pattern_id =
      sorted_idx == 5 ? rank_multiset5_id_from_offsets(sorted_offsets) : -1;

  const uint32_t hero_score = evaluate_with_board_state_lookup(
      board_rank_counts,
      board_suit_counts,
      board_rank_mask,
      board_suit_rank_mask,
      board_pair_mask,
      board_trip_mask,
      board_quad_mask,
      board_max_suit_count,
      board_max_suit_index,
      rank_pattern_id,
      rank_pattern_scores,
      hero_first,
      hero_second);
  const uint32_t villain_score = evaluate_with_board_state_lookup(
      board_rank_counts,
      board_suit_counts,
      board_rank_mask,
      board_suit_rank_mask,
      board_pair_mask,
      board_trip_mask,
      board_quad_mask,
      board_max_suit_count,
      board_max_suit_index,
      rank_pattern_id,
      rank_pattern_scores,
      villain_first,
      villain_second);
  return (hero_score > villain_score) - (hero_score < villain_score);
}

/* ---- PRNG: splitmix64 seed mixer + xorshift64* ---------------------------
 * Same PRNG used across all native equity code for reproducible results.
 * splitmix64 mixes the seed for avalanche; xorshift64* generates the stream. */

/* splitmix64 finalizer: mixes a 64-bit seed value for good avalanche. */
HD uint64_t mix64(uint64_t value) {
  uint64_t z = value + 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z = z ^ (z >> 31);
  return z;
}

/* Extract hero (low) hole-card ID from a packed 22-bit matchup key.
 * Layout: bits [21:11] = hero ID, bits [10:0] = villain ID. */
HD inline int unpack_low_id(const jint packed_key) {
  const uint32_t packed = static_cast<uint32_t>(packed_key);
  return static_cast<int>((packed >> kIdBits) & static_cast<uint32_t>(kIdMask));
}

/* Extract villain (high) hole-card ID from a packed 22-bit matchup key. */
HD inline int unpack_high_id(const jint packed_key) {
  const uint32_t packed = static_cast<uint32_t>(packed_key);
  return static_cast<int>(packed & static_cast<uint32_t>(kIdMask));
}

/* xorshift64* generator: advance state and return multiplied output. */
HD inline uint64_t next_u64(uint64_t& state) {
  state ^= (state >> 12);
  state ^= (state << 25);
  state ^= (state >> 27);
  return state * kRngMul;
}

HD inline int bounded_rand(uint64_t& state, const int bound) {
  return static_cast<int>(next_u64(state) % static_cast<uint64_t>(bound));
}

/* ---- Board sampling strategies ------------------------------------------- */

/* Rejection sampling: draw 5 distinct cards from the 48-card remaining deck.
 * Uses a 64-bit bitmask to track already-drawn positions. */
HD inline void sample_board_cards(
    const uint8_t remaining[kRemainingAfterHoleCards],
    uint64_t& state,
    uint8_t board[kBoardCardCount]) {
  uint64_t used = 0ULL;
  int filled = 0;
  while (filled < kBoardCardCount) {
    const int deck_idx = bounded_rand(state, kRemainingAfterHoleCards);
    const uint64_t bit = 1ULL << deck_idx;
    if ((used & bit) == 0ULL) {
      used |= bit;
      board[filled++] = remaining[deck_idx];
    }
  }
}

/* Index-based sampling: pick a random index into the precomputed C(48,5)
 * board combo table, then map those 5 deck indices to actual card IDs. */
HD inline void sample_board_cards_from_combos(
    const uint8_t remaining[kRemainingAfterHoleCards],
    const uint8_t* board_combos,
    uint64_t& state,
    uint8_t board[kBoardCardCount]) {
  const int combo = bounded_rand(state, kExactBoardCount);
  const int base = combo * kBoardCardCount;
  board[0] = remaining[board_combos[base + 0]];
  board[1] = remaining[board_combos[base + 1]];
  board[2] = remaining[board_combos[base + 2]];
  board[3] = remaining[board_combos[base + 3]];
  board[4] = remaining[board_combos[base + 4]];
}

/* Absolute board rejection sampling: pick a random C(52,5) board index,
 * reject if any board card overlaps with the dead-card bitmask (the 4
 * hole cards).  ~85% acceptance rate (48^5 / 52^5). */
HD inline int sample_absolute_board_index_non_overlapping(
    const uint8_t* absolute_boards,
    uint64_t& state,
    const uint64_t dead_mask) {
  while (true) {
    const int board_idx = bounded_rand(state, kAbsoluteBoardCount);
    const int base = board_idx * kBoardCardCount;
    const uint8_t c0 = absolute_boards[base + 0];
    const uint8_t c1 = absolute_boards[base + 1];
    const uint8_t c2 = absolute_boards[base + 2];
    const uint8_t c3 = absolute_boards[base + 3];
    const uint8_t c4 = absolute_boards[base + 4];
    const uint64_t board_mask =
        (static_cast<uint64_t>(1) << static_cast<uint64_t>(c0)) |
        (static_cast<uint64_t>(1) << static_cast<uint64_t>(c1)) |
        (static_cast<uint64_t>(1) << static_cast<uint64_t>(c2)) |
        (static_cast<uint64_t>(1) << static_cast<uint64_t>(c3)) |
        (static_cast<uint64_t>(1) << static_cast<uint64_t>(c4));
    if ((board_mask & dead_mask) == 0ULL) {
      return board_idx;
    }
  }
}

HD inline void load_absolute_board_cards(
    const uint8_t* absolute_boards,
    const int board_idx,
    uint8_t board[kBoardCardCount]) {
  const int base = board_idx * kBoardCardCount;
  board[0] = absolute_boards[base + 0];
  board[1] = absolute_boards[base + 1];
  board[2] = absolute_boards[base + 2];
  board[3] = absolute_boards[base + 3];
  board[4] = absolute_boards[base + 4];
}

/* Compute Monte Carlo standard error of the equity estimate.
 * Uses the Bernoulli variance formula: Var = E[X^2] - E[X]^2,
 * where X = 1 for win, 0.5 for tie, 0 for loss. */
HD inline double monte_carlo_stderr(
    const int win_count,
    const int tie_count,
    const int trials) {
  if (trials <= 1) {
    return 0.0;
  }
  const double n = static_cast<double>(trials);
  const double mean = (static_cast<double>(win_count) + 0.5 * static_cast<double>(tie_count)) / n;
  const double ex2 = (static_cast<double>(win_count) + 0.25 * static_cast<double>(tie_count)) / n;
  double population_variance = ex2 - (mean * mean);
  if (population_variance < 0.0) {
    population_variance = 0.0;
  }
  const double sample_variance = population_variance * (n / (n - 1.0));
  return sqrt(sample_variance / n);
}

/* ---- CPU equity computation ---------------------------------------------- */

/* CPU Monte Carlo equity for a single matchup. */
EquityResultNative compute_monte_carlo_equity_cpu(
    const int hero_first,
    const int hero_second,
    const int villain_first,
    const int villain_second,
    const int trials,
    const uint64_t seed) {
  uint8_t remaining[kRemainingAfterHoleCards];
  fill_remaining_deck(hero_first, hero_second, villain_first, villain_second, remaining);

  uint8_t board[kBoardCardCount];
  int win_count = 0;
  int tie_count = 0;
  int loss_count = 0;

  uint64_t state = mix64(seed ^ 0xD6E8FEB86659FD93ULL);
  if (state == 0ULL) {
    state = 0x9E3779B97F4A7C15ULL;
  }

  for (int trial = 0; trial < trials; ++trial) {
    sample_board_cards(remaining, state, board);

    const int cmp = compare_showdown(hero_first, hero_second, villain_first, villain_second, board);
    win_count  += (cmp > 0);
    tie_count  += (cmp == 0);
    loss_count += (cmp < 0);
  }

  const double total = static_cast<double>(trials);
  const double std_error = monte_carlo_stderr(win_count, tie_count, trials);
  return EquityResultNative{
      static_cast<double>(win_count) / total,
      static_cast<double>(tie_count) / total,
      static_cast<double>(loss_count) / total,
      std_error,
  };
}

/* CPU exact equity: enumerate all C(48,5) = 1,712,304 boards. */
EquityResultNative compute_exact_equity_cpu(
    const int hero_first,
    const int hero_second,
    const int villain_first,
    const int villain_second) {
  uint8_t remaining[kRemainingAfterHoleCards];
  fill_remaining_deck(hero_first, hero_second, villain_first, villain_second, remaining);

  uint64_t win_count = 0;
  uint64_t tie_count = 0;
  uint64_t loss_count = 0;
  uint64_t total = 0;
  uint8_t board[kBoardCardCount];

  for (int a = 0; a <= (kRemainingAfterHoleCards - 5); ++a) {
    board[0] = remaining[a];
    for (int b = a + 1; b <= (kRemainingAfterHoleCards - 4); ++b) {
      board[1] = remaining[b];
      for (int c = b + 1; c <= (kRemainingAfterHoleCards - 3); ++c) {
        board[2] = remaining[c];
        for (int d = c + 1; d <= (kRemainingAfterHoleCards - 2); ++d) {
          board[3] = remaining[d];
          for (int e = d + 1; e <= (kRemainingAfterHoleCards - 1); ++e) {
            board[4] = remaining[e];
            const int cmp = compare_showdown(hero_first, hero_second, villain_first, villain_second, board);
            ++total;
            win_count  += (cmp > 0);
            tie_count  += (cmp == 0);
            loss_count += (cmp < 0);
          }
        }
      }
    }
  }

  const double total_d = static_cast<double>(total);
  return EquityResultNative{
      static_cast<double>(win_count) / total_d,
      static_cast<double>(tie_count) / total_d,
      static_cast<double>(loss_count) / total_d,
      0.0,
  };
}

/* ---- Configuration resolution --------------------------------------------
 * Settings are read from JVM system properties (via JNI) first, then
 * from environment variables as fallback.  This allows runtime tuning
 * without recompilation. */

/* Parse a string as a positive integer; returns -1 on failure. */
int parse_positive_env_int(const char* value) {
  if (value == nullptr || value[0] == '\0') {
    return -1;
  }
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
    return -1;
  }
  return static_cast<int>(parsed);
}

bool parse_truthy(const char* raw) {
  if (raw == nullptr || raw[0] == '\0') {
    return false;
  }
  std::string value(raw);
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value == "1" || value == "true" || value == "yes" || value == "on";
}

bool parse_truthy(const std::string& raw) {
  return parse_truthy(raw.c_str());
}

/* Determine CPU worker thread count for parallel batch processing.
 * Uses hardware_concurrency() as default, overridable via sicfun_GPU_NATIVE_THREADS. */
int resolve_worker_count(const int entries) {
  if (entries <= 1) {
    return 1;
  }
  int workers = static_cast<int>(std::thread::hardware_concurrency());
  if (workers <= 0) {
    workers = 1;
  }
  const int env_workers = parse_positive_env_int(std::getenv("sicfun_GPU_NATIVE_THREADS"));
  if (env_workers > 0) {
    workers = env_workers;
  }
  workers = std::max(1, std::min(workers, entries));
  return workers;
}

NativeEngine parse_engine_value(const std::string& raw) {
  std::string value(raw);
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  if (value == "cpu") {
    return NativeEngine::Cpu;
  }
  if (value == "cuda") {
    return NativeEngine::Cuda;
  }
  return NativeEngine::Auto;
}

RangeMemoryPath parse_range_memory_path_value(const std::string& raw) {
  std::string value(raw);
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  if (value == "readonly" || value == "read-only" || value == "ldg") {
    return RangeMemoryPath::ReadOnly;
  }
  if (value == "global") {
    return RangeMemoryPath::Global;
  }
  return RangeMemoryPath::Global;
}

/* Read a JVM system property (java.lang.System.getProperty) via JNI.
 * Returns false if the property is not set or JNI calls fail. */
bool try_read_system_property(JNIEnv* env, const char* key, std::string& out) {
  jclass system_class = env->FindClass("java/lang/System");
  if (system_class == nullptr) {
    check_and_clear_exception(env);
    return false;
  }
  jmethodID get_property = env->GetStaticMethodID(
      system_class,
      "getProperty",
      "(Ljava/lang/String;)Ljava/lang/String;");
  if (get_property == nullptr) {
    env->DeleteLocalRef(system_class);
    check_and_clear_exception(env);
    return false;
  }
  jstring key_string = env->NewStringUTF(key);
  if (key_string == nullptr) {
    env->DeleteLocalRef(system_class);
    check_and_clear_exception(env);
    return false;
  }
  jstring value_string = static_cast<jstring>(
      env->CallStaticObjectMethod(system_class, get_property, key_string));
  if (check_and_clear_exception(env) || value_string == nullptr) {
    if (value_string != nullptr) {
      env->DeleteLocalRef(value_string);
    }
    env->DeleteLocalRef(key_string);
    env->DeleteLocalRef(system_class);
    return false;
  }
  const char* utf = env->GetStringUTFChars(value_string, nullptr);
  if (utf == nullptr) {
    env->DeleteLocalRef(value_string);
    env->DeleteLocalRef(key_string);
    env->DeleteLocalRef(system_class);
    check_and_clear_exception(env);
    return false;
  }
  out.assign(utf);
  env->ReleaseStringUTFChars(value_string, utf);
  env->DeleteLocalRef(value_string);
  env->DeleteLocalRef(key_string);
  env->DeleteLocalRef(system_class);
  return true;
}

/* Clamp and align a CUDA block size to [32, 1024], rounded down to a warp multiple. */
int normalize_cuda_block_size(const int raw, const int fallback) {
  const int threads = raw > 0 ? raw : fallback;
  return std::clamp(threads, 32, 1024) & ~31;
}

int resolve_cuda_threads_per_block(JNIEnv* env, const jint mode_code) {
  const int default_threads = mode_code == 0 ? kDefaultCudaThreadsPerBlockExact : kDefaultCudaThreadsPerBlock;
  std::string property_value;
  if (try_read_system_property(env, "sicfun.gpu.native.cuda.blockSize", property_value)) {
    const int parsed = parse_positive_env_int(property_value.c_str());
    if (parsed > 0) {
      return normalize_cuda_block_size(parsed, default_threads);
    }
  }
  const int env_threads = parse_positive_env_int(std::getenv("sicfun_GPU_CUDA_BLOCK_SIZE"));
  if (env_threads > 0) {
    return normalize_cuda_block_size(env_threads, default_threads);
  }
  return default_threads;
}

int resolve_range_cuda_threads_per_block(JNIEnv* env) {
  std::string property_value;
  if (try_read_system_property(env, "sicfun.gpu.native.range.cuda.blockSize", property_value)) {
    const int parsed = parse_positive_env_int(property_value.c_str());
    if (parsed > 0) {
      return normalize_cuda_block_size(parsed, kDefaultRangeCudaThreadsPerBlock);
    }
  }
  const int env_threads = parse_positive_env_int(std::getenv("sicfun_GPU_RANGE_CUDA_BLOCK_SIZE"));
  if (env_threads > 0) {
    return normalize_cuda_block_size(env_threads, kDefaultRangeCudaThreadsPerBlock);
  }
  return kDefaultRangeCudaThreadsPerBlock;
}

int normalize_cuda_max_chunk_matchups(const int raw, const int entries, const int fallback) {
  if (entries <= 0) {
    return 1;
  }
  return std::clamp(raw > 0 ? raw : fallback, 1, entries);
}

int resolve_cuda_max_chunk_matchups(JNIEnv* env, const int entries, const jint mode_code) {
  const int default_chunk = mode_code == 0 ? kDefaultCudaMaxChunkMatchupsExact : kDefaultCudaMaxChunkMatchups;
  std::string property_value;
  if (try_read_system_property(env, "sicfun.gpu.native.cuda.maxChunkMatchups", property_value)) {
    const int parsed = parse_positive_env_int(property_value.c_str());
    if (parsed > 0) {
      return normalize_cuda_max_chunk_matchups(parsed, entries, default_chunk);
    }
  }
  const int env_chunk = parse_positive_env_int(std::getenv("sicfun_GPU_CUDA_MAX_CHUNK_MATCHUPS"));
  if (env_chunk > 0) {
    return normalize_cuda_max_chunk_matchups(env_chunk, entries, default_chunk);
  }
  return normalize_cuda_max_chunk_matchups(default_chunk, entries, default_chunk);
}

int resolve_range_cuda_max_chunk_heroes(JNIEnv* env, const int hero_count) {
  std::string property_value;
  if (try_read_system_property(env, "sicfun.gpu.native.range.cuda.maxChunkHeroes", property_value)) {
    const int parsed = parse_positive_env_int(property_value.c_str());
    if (parsed > 0) {
      return normalize_cuda_max_chunk_matchups(parsed, hero_count, kDefaultRangeCudaMaxChunkHeroes);
    }
  }
  const int env_chunk = parse_positive_env_int(std::getenv("sicfun_GPU_RANGE_CUDA_MAX_CHUNK_HEROES"));
  if (env_chunk > 0) {
    return normalize_cuda_max_chunk_matchups(env_chunk, hero_count, kDefaultRangeCudaMaxChunkHeroes);
  }
  return normalize_cuda_max_chunk_matchups(
      kDefaultRangeCudaMaxChunkHeroes,
      hero_count,
      kDefaultRangeCudaMaxChunkHeroes);
}

bool resolve_monte_carlo_use_board_combos(JNIEnv* env) {
  std::string property_value;
  if (try_read_system_property(env, "sicfun.gpu.native.monteCarlo.useBoardCombos", property_value)) {
    return parse_truthy(property_value);
  }
  return parse_truthy(std::getenv("sicfun_GPU_NATIVE_MONTE_CARLO_USE_BOARD_COMBOS"));
}

bool resolve_monte_carlo_use_rank_lookup(JNIEnv* env) {
  std::string property_value;
  if (try_read_system_property(env, "sicfun.gpu.native.monteCarlo.useRankLookup", property_value)) {
    return parse_truthy(property_value);
  }
  const char* env_value = std::getenv("sicfun_GPU_NATIVE_MONTE_CARLO_USE_RANK_LOOKUP");
  if (env_value == nullptr || env_value[0] == '\0') {
    return true;
  }
  return parse_truthy(env_value);
}

bool resolve_monte_carlo_parallel_trials(JNIEnv* env) {
  std::string property_value;
  if (try_read_system_property(env, "sicfun.gpu.native.monteCarlo.parallelTrials", property_value)) {
    return parse_truthy(property_value);
  }
  const char* env_value = std::getenv("sicfun_GPU_NATIVE_MONTE_CARLO_PARALLEL_TRIALS");
  if (env_value == nullptr || env_value[0] == '\0') {
    return true;
  }
  return parse_truthy(env_value);
}

bool resolve_monte_carlo_absolute_board_sampling(JNIEnv* env) {
  std::string property_value;
  if (try_read_system_property(env, "sicfun.gpu.native.monteCarlo.absoluteBoardSampling", property_value)) {
    return parse_truthy(property_value);
  }
  const char* env_value = std::getenv("sicfun_GPU_NATIVE_MONTE_CARLO_ABSOLUTE_BOARD_SAMPLING");
  if (env_value == nullptr || env_value[0] == '\0') {
    return false;
  }
  return parse_truthy(env_value);
}

NativeEngine resolve_engine(JNIEnv* env) {
  std::string property_value;
  if (try_read_system_property(env, "sicfun.gpu.native.engine", property_value)) {
    return parse_engine_value(property_value);
  }
  const char* raw = std::getenv("sicfun_GPU_NATIVE_ENGINE");
  if (raw == nullptr || raw[0] == '\0') {
    return NativeEngine::Auto;
  }
  return parse_engine_value(raw);
}

RangeMemoryPath resolve_range_memory_path(JNIEnv* env) {
  std::string property_value;
  if (try_read_system_property(env, "sicfun.gpu.native.range.memoryPath", property_value)) {
    return parse_range_memory_path_value(property_value);
  }
  const char* raw = std::getenv("sicfun_GPU_RANGE_MEMORY_PATH");
  if (raw == nullptr || raw[0] == '\0') {
    return RangeMemoryPath::Global;
  }
  return parse_range_memory_path_value(raw);
}

bool resolve_exact_board_major_enabled(JNIEnv* env) {
  std::string property_value;
  if (try_read_system_property(env, "sicfun.gpu.native.exact.boardMajor", property_value)) {
    return parse_truthy(property_value);
  }
  const char* raw = std::getenv("sicfun_GPU_EXACT_BOARD_MAJOR");
  return parse_truthy(raw);
}

/* ---- CPU batch dispatch ---------------------------------------------------
 * Multithreaded CPU equity computation using lock-free work-stealing.
 * Each worker atomically claims chunks of kCpuWorkChunkSize matchups. */
int compute_batch_cpu(
    const std::vector<jint>& low_buf,
    const std::vector<jint>& high_buf,
    const std::vector<jlong>& seed_buf,
    const jint mode_code,
    const jint trials,
    std::vector<jdouble>& win_buf,
    std::vector<jdouble>& tie_buf,
    std::vector<jdouble>& loss_buf,
    std::vector<jdouble>& stderr_buf) {
  const jsize n = static_cast<jsize>(low_buf.size());
  const auto& lookup = hole_cards_lookup();
  std::atomic<jsize> next_index(0);
  std::atomic<jint> worker_error(0);
  const int workers = resolve_worker_count(static_cast<int>(n));

  auto worker = [&]() {
    while (true) {
      if (worker_error.load(std::memory_order_relaxed) != 0) {
        return;
      }
      const jsize start = next_index.fetch_add(kCpuWorkChunkSize, std::memory_order_relaxed);
      if (start >= n) {
        return;
      }
      const jsize end = std::min<jsize>(n, start + kCpuWorkChunkSize);
      for (jsize i = start; i < end; ++i) {
        const size_t idx = static_cast<size_t>(i);
        const HoleCards hero = lookup[static_cast<size_t>(low_buf[idx])];
        const HoleCards villain = lookup[static_cast<size_t>(high_buf[idx])];
        const bool overlap = hero.first == villain.first || hero.first == villain.second ||
                             hero.second == villain.first || hero.second == villain.second;
        if (overlap) {
          worker_error.store(127, std::memory_order_relaxed);
          return;
        }

        EquityResultNative result{};
        if (mode_code == 0) {
          result = compute_exact_equity_cpu(hero.first, hero.second, villain.first, villain.second);
        } else {
          result = compute_monte_carlo_equity_cpu(
              hero.first,
              hero.second,
              villain.first,
              villain.second,
              trials,
              static_cast<uint64_t>(seed_buf[idx]));
        }
        win_buf[idx] = result.win;
        tie_buf[idx] = result.tie;
        loss_buf[idx] = result.loss;
        stderr_buf[idx] = result.std_error;
      }
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(workers > 0 ? workers - 1 : 0));
  for (int t = 1; t < workers; ++t) {
    threads.emplace_back(worker);
  }
  worker();
  for (std::thread& thread : threads) {
    thread.join();
  }
  return worker_error.load(std::memory_order_relaxed);
}

/* CPU range-vs-range Monte Carlo equity via CSR layout.
 * For each hero hand, iterates over the villain range (weighted by probability),
 * computes per-matchup MC equity, and aggregates into a probability-weighted average.
 * Standard errors are combined via root-sum-of-squares. */
int compute_range_batch_cpu_monte_carlo_csr(
    const std::vector<jint>& hero_ids,
    const std::vector<jint>& offsets,
    const std::vector<jint>& villain_ids,
    const std::vector<jlong>& key_material,
    const std::vector<jfloat>& probabilities,
    const jint trials,
    const jlong monte_carlo_seed_base,
    std::vector<jfloat>& out_wins,
    std::vector<jfloat>& out_ties,
    std::vector<jfloat>& out_losses,
    std::vector<jfloat>& out_stderrs) {
  const int hero_count = static_cast<int>(hero_ids.size());
  const int entry_count = static_cast<int>(villain_ids.size());
  if (hero_count == 0) {
    return 0;
  }
  if (static_cast<int>(offsets.size()) != hero_count + 1) {
    return 101;
  }
  if (offsets[0] != 0 || offsets[hero_count] != entry_count) {
    return kStatusInvalidRangeLayout;
  }
  for (int h = 0; h < hero_count; ++h) {
    if (offsets[h] > offsets[h + 1]) {
      return kStatusInvalidRangeLayout;
    }
    const int hero_id = static_cast<int>(hero_ids[static_cast<size_t>(h)]);
    if (hero_id < 0 || hero_id >= kHoleCardsCount) {
      return 125;
    }
  }
  for (int i = 0; i < entry_count; ++i) {
    const int villain_id = static_cast<int>(villain_ids[static_cast<size_t>(i)]);
    if (villain_id < 0 || villain_id >= kHoleCardsCount) {
      return 125;
    }
    const float p = probabilities[static_cast<size_t>(i)];
    if (!std::isfinite(p) || p < 0.0f) {
      return kStatusInvalidRangeLayout;
    }
  }

  const auto& lookup = hole_cards_lookup();
  std::atomic<int> next_hero(0);
  std::atomic<jint> worker_error(0);
  const int workers = resolve_worker_count(hero_count);

  auto worker = [&]() {
    while (true) {
      if (worker_error.load(std::memory_order_relaxed) != 0) {
        return;
      }
      const int hero_idx = next_hero.fetch_add(1, std::memory_order_relaxed);
      if (hero_idx >= hero_count) {
        return;
      }

      const int hero_id = static_cast<int>(hero_ids[static_cast<size_t>(hero_idx)]);
      const HoleCards hero = lookup[static_cast<size_t>(hero_id)];
      const int start = offsets[static_cast<size_t>(hero_idx)];
      const int end = offsets[static_cast<size_t>(hero_idx + 1)];

      double weighted_win = 0.0;
      double weighted_tie = 0.0;
      double weighted_loss = 0.0;
      double weighted_stderr_sq = 0.0;
      double weight_sum = 0.0;

      for (int i = start; i < end; ++i) {
        const size_t idx = static_cast<size_t>(i);
        const int villain_id = static_cast<int>(villain_ids[idx]);
        const HoleCards villain = lookup[static_cast<size_t>(villain_id)];
        const bool overlap = hero.first == villain.first || hero.first == villain.second ||
                             hero.second == villain.first || hero.second == villain.second;
        if (overlap) {
          worker_error.store(127, std::memory_order_relaxed);
          return;
        }

        const double p = static_cast<double>(probabilities[idx]);
        if (p <= 0.0) {
          continue;
        }
        const uint64_t seed = mix64(
            static_cast<uint64_t>(monte_carlo_seed_base) ^
            static_cast<uint64_t>(key_material[idx]));
        const EquityResultNative result = compute_monte_carlo_equity_cpu(
            hero.first,
            hero.second,
            villain.first,
            villain.second,
            trials,
            seed);
        weighted_win += p * result.win;
        weighted_tie += p * result.tie;
        weighted_loss += p * result.loss;
        const double weighted_stderr = p * result.std_error;
        weighted_stderr_sq += weighted_stderr * weighted_stderr;
        weight_sum += p;
      }

      if (weight_sum > 0.0) {
        out_wins[static_cast<size_t>(hero_idx)] = static_cast<jfloat>(weighted_win / weight_sum);
        out_ties[static_cast<size_t>(hero_idx)] = static_cast<jfloat>(weighted_tie / weight_sum);
        out_losses[static_cast<size_t>(hero_idx)] = static_cast<jfloat>(weighted_loss / weight_sum);
        out_stderrs[static_cast<size_t>(hero_idx)] =
            static_cast<jfloat>(std::sqrt(weighted_stderr_sq) / weight_sum);
      } else {
        out_wins[static_cast<size_t>(hero_idx)] = 0.0f;
        out_ties[static_cast<size_t>(hero_idx)] = 0.0f;
        out_losses[static_cast<size_t>(hero_idx)] = 0.0f;
        out_stderrs[static_cast<size_t>(hero_idx)] = 0.0f;
      }
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(workers > 0 ? workers - 1 : 0));
  for (int t = 1; t < workers; ++t) {
    threads.emplace_back(worker);
  }
  worker();
  for (std::thread& thread : threads) {
    thread.join();
  }
  return worker_error.load(std::memory_order_relaxed);
}

/* ---- CUDA kernels -------------------------------------------------------- */

/* Atomically set a device-side status code (first error wins). */
__device__ inline void set_status_once(int* status, int code) {
  atomicCAS(status, 0, code);
}

/* Templated memory load helpers: when UseReadOnly=true, uses __ldg()
 * (read-only texture cache path) for potentially better cache hit rates
 * on data that is not modified during the kernel. */
template <bool UseReadOnly>
__device__ inline jint load_jint(const jint* ptr, const int idx) {
#if defined(__CUDA_ARCH__)
  if constexpr (UseReadOnly) {
    return __ldg(ptr + idx);
  } else {
    return ptr[idx];
  }
#else
  return ptr[idx];
#endif
}

template <bool UseReadOnly>
__device__ inline jlong load_jlong(const jlong* ptr, const int idx) {
#if defined(__CUDA_ARCH__)
  if constexpr (UseReadOnly) {
    return __ldg(ptr + idx);
  } else {
    return ptr[idx];
  }
#else
  return ptr[idx];
#endif
}

template <bool UseReadOnly>
__device__ inline jfloat load_jfloat(const jfloat* ptr, const int idx) {
#if defined(__CUDA_ARCH__)
  if constexpr (UseReadOnly) {
    return __ldg(ptr + idx);
  } else {
    return ptr[idx];
  }
#else
  return ptr[idx];
#endif
}

/* Monte Carlo kernel (one thread per matchup):
 * Each thread independently simulates `trials` random boards for its
 * assigned matchup, counting wins/ties/losses and computing stderr.
 * Supports three board sampling strategies:
 *   1. Rejection sampling from 48-card remaining deck
 *   2. Index-based sampling from precomputed C(48,5) combos
 *   3. Absolute board sampling with rank-pattern shortcut */
__global__ void monte_carlo_kernel(
    const jint* low_ids,
    const jint* high_ids,
    const jlong* seeds,
    const int n,
    const int index_offset,
    const int trials,
    const uint8_t* board_combos,
    const uint8_t* absolute_boards,
    const uint16_t* absolute_board_rank_pattern_ids,
    const uint8_t* absolute_board_flush_meta,
    const uint32_t* rank_pattern_scores,
    jdouble* wins,
    jdouble* ties,
    jdouble* losses,
    jdouble* stderrs,
    int* status) {
  const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if (idx >= n) {
    return;
  }

  const int low_id = static_cast<int>(low_ids[idx]);
  const int high_id = static_cast<int>(high_ids[idx]);
  if (low_id < 0 || low_id >= kHoleCardsCount || high_id < 0 || high_id >= kHoleCardsCount) {
    set_status_once(status, 125);
    return;
  }

  const int hero_first = static_cast<int>(d_hole_first[low_id]);
  const int hero_second = static_cast<int>(d_hole_second[low_id]);
  const int villain_first = static_cast<int>(d_hole_first[high_id]);
  const int villain_second = static_cast<int>(d_hole_second[high_id]);
  const bool overlap = hero_first == villain_first || hero_first == villain_second ||
                       hero_second == villain_first || hero_second == villain_second;
  if (overlap) {
    set_status_once(status, 127);
    return;
  }

  const bool use_absolute_sampling =
      absolute_boards != nullptr &&
      absolute_board_rank_pattern_ids != nullptr &&
      absolute_board_flush_meta != nullptr &&
      rank_pattern_scores != nullptr;
  uint8_t remaining[kRemainingAfterHoleCards];
  if (!use_absolute_sampling) {
    fill_remaining_deck(hero_first, hero_second, villain_first, villain_second, remaining);
  }
  const uint64_t dead_mask =
      (static_cast<uint64_t>(1) << static_cast<uint64_t>(hero_first)) |
      (static_cast<uint64_t>(1) << static_cast<uint64_t>(hero_second)) |
      (static_cast<uint64_t>(1) << static_cast<uint64_t>(villain_first)) |
      (static_cast<uint64_t>(1) << static_cast<uint64_t>(villain_second));

  uint8_t board[kBoardCardCount];
  int win_count = 0;
  int tie_count = 0;
  int loss_count = 0;

  const int global_idx = idx + index_offset;
  uint64_t state = mix64(static_cast<uint64_t>(seeds[idx]) ^ static_cast<uint64_t>(global_idx + 1));
  if (state == 0ULL) {
    state = 0x9E3779B97F4A7C15ULL;
  }

  for (int trial = 0; trial < trials; ++trial) {
    int cmp = 0;
    if (use_absolute_sampling) {
      const int board_idx = sample_absolute_board_index_non_overlapping(
          absolute_boards,
          state,
          dead_mask);
      const uint8_t flush_meta = absolute_board_flush_meta[board_idx];
      const int board_max_suit_index = static_cast<int>(flush_meta & 0x03u);
      const int board_max_suit_count = static_cast<int>((flush_meta >> 2) & 0x07u);
      const bool hero_flush_possible = flush_possible_for_board_meta(
          board_max_suit_count,
          board_max_suit_index,
          hero_first,
          hero_second);
      const bool villain_flush_possible = flush_possible_for_board_meta(
          board_max_suit_count,
          board_max_suit_index,
          villain_first,
          villain_second);
      if (!hero_flush_possible && !villain_flush_possible) {
        const int rank_pattern_id = static_cast<int>(absolute_board_rank_pattern_ids[board_idx]);
        if (rank_pattern_id >= 0 && rank_pattern_id < kRankPatternCount) {
          cmp = compare_showdown_rank_lookup_only(
              hero_first,
              hero_second,
              villain_first,
              villain_second,
              rank_pattern_id,
              rank_pattern_scores);
        } else {
          load_absolute_board_cards(absolute_boards, board_idx, board);
          cmp = compare_showdown(
              hero_first,
              hero_second,
              villain_first,
              villain_second,
              board,
              rank_pattern_scores);
        }
      } else {
        load_absolute_board_cards(absolute_boards, board_idx, board);
        cmp = compare_showdown(
            hero_first,
            hero_second,
            villain_first,
            villain_second,
            board,
            rank_pattern_scores);
      }
    } else {
      if (board_combos != nullptr) {
        sample_board_cards_from_combos(remaining, board_combos, state, board);
      } else {
        sample_board_cards(remaining, state, board);
      }
      cmp = compare_showdown(
          hero_first,
          hero_second,
          villain_first,
          villain_second,
          board,
          rank_pattern_scores);
    }
    win_count  += (cmp > 0);
    tie_count  += (cmp == 0);
    loss_count += (cmp < 0);
  }

  const double total = static_cast<double>(trials);
  const double std_error = monte_carlo_stderr(win_count, tie_count, trials);

  wins[idx] = static_cast<double>(win_count) / total;
  ties[idx] = static_cast<double>(tie_count) / total;
  losses[idx] = static_cast<double>(loss_count) / total;
  stderrs[idx] = std_error;
}

/* Monte Carlo kernel (one BLOCK per matchup, parallel trial reduction):
 * Thread 0 sets up shared memory (hole cards, remaining deck, dead mask).
 * All threads cooperatively run trials with stride = blockDim.x, then
 * reduce win/tie/loss counts via shared-memory tree reduction.
 * Better GPU utilization when trial count >> thread count per matchup. */
__global__ void monte_carlo_kernel_parallel_trials(
    const jint* low_ids,
    const jint* high_ids,
    const jlong* seeds,
    const int n,
    const int index_offset,
    const int trials,
    const uint8_t* board_combos,
    const uint8_t* absolute_boards,
    const uint16_t* absolute_board_rank_pattern_ids,
    const uint8_t* absolute_board_flush_meta,
    const uint32_t* rank_pattern_scores,
    jdouble* wins,
    jdouble* ties,
    jdouble* losses,
    jdouble* stderrs,
    int* status) {
  const int matchup_idx = static_cast<int>(blockIdx.x);
  if (matchup_idx >= n) {
    return;
  }

  __shared__ int hero_first;
  __shared__ int hero_second;
  __shared__ int villain_first;
  __shared__ int villain_second;
  __shared__ int valid_matchup;
  __shared__ int use_absolute_sampling;
  __shared__ uint8_t remaining[kRemainingAfterHoleCards];
  __shared__ uint64_t dead_mask;

  if (threadIdx.x == 0) {
    const int low_id = static_cast<int>(low_ids[matchup_idx]);
    const int high_id = static_cast<int>(high_ids[matchup_idx]);
    if (low_id < 0 || low_id >= kHoleCardsCount || high_id < 0 || high_id >= kHoleCardsCount) {
      set_status_once(status, 125);
      valid_matchup = 0;
    } else {
      hero_first = static_cast<int>(d_hole_first[low_id]);
      hero_second = static_cast<int>(d_hole_second[low_id]);
      villain_first = static_cast<int>(d_hole_first[high_id]);
      villain_second = static_cast<int>(d_hole_second[high_id]);
      const bool overlap = hero_first == villain_first || hero_first == villain_second ||
                           hero_second == villain_first || hero_second == villain_second;
      if (overlap) {
        set_status_once(status, 127);
        valid_matchup = 0;
      } else {
        use_absolute_sampling =
            absolute_boards != nullptr &&
            absolute_board_rank_pattern_ids != nullptr &&
            absolute_board_flush_meta != nullptr &&
            rank_pattern_scores != nullptr ? 1 : 0;
        if (use_absolute_sampling == 0) {
          fill_remaining_deck(hero_first, hero_second, villain_first, villain_second, remaining);
        }
        dead_mask =
            (static_cast<uint64_t>(1) << static_cast<uint64_t>(hero_first)) |
            (static_cast<uint64_t>(1) << static_cast<uint64_t>(hero_second)) |
            (static_cast<uint64_t>(1) << static_cast<uint64_t>(villain_first)) |
            (static_cast<uint64_t>(1) << static_cast<uint64_t>(villain_second));
        valid_matchup = 1;
      }
    }
  }
  __syncthreads();
  if (valid_matchup == 0) {
    return;
  }

  const int global_idx = matchup_idx + index_offset;
  uint64_t state = mix64(
      static_cast<uint64_t>(seeds[matchup_idx]) ^
      static_cast<uint64_t>(global_idx + 1) ^
      (static_cast<uint64_t>(threadIdx.x + 1) << 32));
  if (state == 0ULL) {
    state = 0x9E3779B97F4A7C15ULL;
  }

  uint32_t local_win = 0;
  uint32_t local_tie = 0;
  uint32_t local_loss = 0;
  uint8_t board[kBoardCardCount];
  const int hero_first_local = hero_first;
  const int hero_second_local = hero_second;
  const int villain_first_local = villain_first;
  const int villain_second_local = villain_second;
  const uint64_t dead_mask_local = dead_mask;
  const bool use_absolute_sampling_local = use_absolute_sampling != 0;
  for (int trial = static_cast<int>(threadIdx.x); trial < trials; trial += static_cast<int>(blockDim.x)) {
    int cmp = 0;
    if (use_absolute_sampling_local) {
      const int board_idx = sample_absolute_board_index_non_overlapping(
          absolute_boards,
          state,
          dead_mask_local);
      const uint8_t flush_meta = absolute_board_flush_meta[board_idx];
      const int board_max_suit_index = static_cast<int>(flush_meta & 0x03u);
      const int board_max_suit_count = static_cast<int>((flush_meta >> 2) & 0x07u);
      const bool hero_flush_possible = flush_possible_for_board_meta(
          board_max_suit_count,
          board_max_suit_index,
          hero_first_local,
          hero_second_local);
      const bool villain_flush_possible = flush_possible_for_board_meta(
          board_max_suit_count,
          board_max_suit_index,
          villain_first_local,
          villain_second_local);
      if (!hero_flush_possible && !villain_flush_possible) {
        const int rank_pattern_id = static_cast<int>(absolute_board_rank_pattern_ids[board_idx]);
        if (rank_pattern_id >= 0 && rank_pattern_id < kRankPatternCount) {
          cmp = compare_showdown_rank_lookup_only(
              hero_first_local,
              hero_second_local,
              villain_first_local,
              villain_second_local,
              rank_pattern_id,
              rank_pattern_scores);
        } else {
          load_absolute_board_cards(absolute_boards, board_idx, board);
          cmp = compare_showdown(
              hero_first_local,
              hero_second_local,
              villain_first_local,
              villain_second_local,
              board,
              rank_pattern_scores);
        }
      } else {
        load_absolute_board_cards(absolute_boards, board_idx, board);
        cmp = compare_showdown(
            hero_first_local,
            hero_second_local,
            villain_first_local,
            villain_second_local,
            board,
            rank_pattern_scores);
      }
    } else {
      if (board_combos != nullptr) {
        sample_board_cards_from_combos(remaining, board_combos, state, board);
      } else {
        sample_board_cards(remaining, state, board);
      }
      cmp = compare_showdown(
          hero_first_local,
          hero_second_local,
          villain_first_local,
          villain_second_local,
          board,
          rank_pattern_scores);
    }
    local_win  += (cmp > 0);
    local_tie  += (cmp == 0);
    local_loss += (cmp < 0);
  }

  extern __shared__ unsigned int reduction[];
  unsigned int* win_counts = reduction;
  unsigned int* tie_counts = reduction + blockDim.x;
  unsigned int* loss_counts = reduction + (2 * blockDim.x);
  const int tid = static_cast<int>(threadIdx.x);
  win_counts[tid] = local_win;
  tie_counts[tid] = local_tie;
  loss_counts[tid] = local_loss;
  __syncthreads();

  int active = static_cast<int>(blockDim.x);
  while (active > 1) {
    const int half = (active + 1) >> 1;
    if (tid < half) {
      const int other = tid + half;
      if (other < active) {
        win_counts[tid] += win_counts[other];
        tie_counts[tid] += tie_counts[other];
        loss_counts[tid] += loss_counts[other];
      }
    }
    __syncthreads();
    active = half;
  }

  if (tid == 0) {
    const int win_count = static_cast<int>(win_counts[0]);
    const int tie_count = static_cast<int>(tie_counts[0]);
    const int loss_count = static_cast<int>(loss_counts[0]);
    const double total = static_cast<double>(trials);
    const double std_error = monte_carlo_stderr(win_count, tie_count, trials);
    wins[matchup_idx] = static_cast<double>(win_count) / total;
    ties[matchup_idx] = static_cast<double>(tie_count) / total;
    losses[matchup_idx] = static_cast<double>(loss_count) / total;
    stderrs[matchup_idx] = std_error;
  }
}

/* Exact enumeration kernel (one BLOCK per matchup):
 * Thread 0 loads hole cards into shared memory and builds the 48-card
 * remaining deck.  All threads cooperatively enumerate all C(48,5)
 * board combinations with stride = blockDim.x, then reduce via shared
 * memory.  Output: exact equity (std_error = 0.0). */
__global__ void exact_kernel(
    const jint* low_ids,
    const jint* high_ids,
    const int n,
    const uint8_t* board_combos,
    jdouble* wins,
    jdouble* ties,
    jdouble* losses,
    jdouble* stderrs,
    int* status) {
  const int matchup_idx = static_cast<int>(blockIdx.x);
  if (matchup_idx >= n) {
    return;
  }

  __shared__ int hero_first;
  __shared__ int hero_second;
  __shared__ int villain_first;
  __shared__ int villain_second;
  __shared__ int valid_matchup;
  __shared__ uint8_t remaining[kRemainingAfterHoleCards];

  if (threadIdx.x == 0) {
    const int low_id = static_cast<int>(low_ids[matchup_idx]);
    const int high_id = static_cast<int>(high_ids[matchup_idx]);
    if (low_id < 0 || low_id >= kHoleCardsCount || high_id < 0 || high_id >= kHoleCardsCount) {
      set_status_once(status, 125);
      valid_matchup = 0;
    } else {
      hero_first = static_cast<int>(d_hole_first[low_id]);
      hero_second = static_cast<int>(d_hole_second[low_id]);
      villain_first = static_cast<int>(d_hole_first[high_id]);
      villain_second = static_cast<int>(d_hole_second[high_id]);
      const bool overlap = hero_first == villain_first || hero_first == villain_second ||
                           hero_second == villain_first || hero_second == villain_second;
      if (overlap) {
        set_status_once(status, 127);
        valid_matchup = 0;
      } else {
        fill_remaining_deck(hero_first, hero_second, villain_first, villain_second, remaining);
        valid_matchup = 1;
      }
    }
  }
  __syncthreads();
  if (valid_matchup == 0) {
    return;
  }

  uint32_t local_win = 0;
  uint32_t local_tie = 0;
  uint32_t local_loss = 0;
  uint8_t board[kBoardCardCount];

  for (int combo = static_cast<int>(threadIdx.x); combo < kExactBoardCount; combo += static_cast<int>(blockDim.x)) {
    const int base = combo * kBoardCardCount;
    board[0] = remaining[board_combos[base + 0]];
    board[1] = remaining[board_combos[base + 1]];
    board[2] = remaining[board_combos[base + 2]];
    board[3] = remaining[board_combos[base + 3]];
    board[4] = remaining[board_combos[base + 4]];

    const int cmp = compare_showdown(hero_first, hero_second, villain_first, villain_second, board);
    local_win  += (cmp > 0);
    local_tie  += (cmp == 0);
    local_loss += (cmp < 0);
  }

  extern __shared__ unsigned int reduction[];
  unsigned int* win_counts = reduction;
  unsigned int* tie_counts = reduction + blockDim.x;
  unsigned int* loss_counts = reduction + (2 * blockDim.x);
  const int tid = static_cast<int>(threadIdx.x);
  win_counts[tid] = local_win;
  tie_counts[tid] = local_tie;
  loss_counts[tid] = local_loss;
  __syncthreads();

  int active = static_cast<int>(blockDim.x);
  while (active > 1) {
    const int half = (active + 1) >> 1;
    if (tid < half) {
      const int other = tid + half;
      if (other < active) {
        win_counts[tid] += win_counts[other];
        tie_counts[tid] += tie_counts[other];
        loss_counts[tid] += loss_counts[other];
      }
    }
    __syncthreads();
    active = half;
  }

  if (tid == 0) {
    const double total_d = static_cast<double>(kExactBoardCount);
    wins[matchup_idx] = static_cast<double>(win_counts[0]) / total_d;
    ties[matchup_idx] = static_cast<double>(tie_counts[0]) / total_d;
    losses[matchup_idx] = static_cast<double>(loss_counts[0]) / total_d;
    stderrs[matchup_idx] = 0.0;
  }
}

/* Packed-key variant of monte_carlo_kernel.  Matchup IDs are extracted
 * from packed 22-bit keys, and seeds are derived from monte_carlo_seed_base
 * XORed with per-entry key_material.  Output: jfloat (not jdouble). */
__global__ void monte_carlo_kernel_packed(
    const jint* packed_keys,
    const jlong* key_material,
    const int n,
    const int index_offset,
    const int trials,
    const jlong monte_carlo_seed_base,
    const uint8_t* board_combos,
    const uint8_t* absolute_boards,
    const uint16_t* absolute_board_rank_pattern_ids,
    const uint8_t* absolute_board_flush_meta,
    const uint32_t* rank_pattern_scores,
    jfloat* wins,
    jfloat* ties,
    jfloat* losses,
    jfloat* stderrs,
    int* status) {
  const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if (idx >= n) {
    return;
  }

  const int low_id = unpack_low_id(packed_keys[idx]);
  const int high_id = unpack_high_id(packed_keys[idx]);
  if (low_id < 0 || low_id >= kHoleCardsCount || high_id < 0 || high_id >= kHoleCardsCount) {
    set_status_once(status, 125);
    return;
  }

  const int hero_first = static_cast<int>(d_hole_first[low_id]);
  const int hero_second = static_cast<int>(d_hole_second[low_id]);
  const int villain_first = static_cast<int>(d_hole_first[high_id]);
  const int villain_second = static_cast<int>(d_hole_second[high_id]);
  const bool overlap = hero_first == villain_first || hero_first == villain_second ||
                       hero_second == villain_first || hero_second == villain_second;
  if (overlap) {
    set_status_once(status, 127);
    return;
  }

  const bool use_absolute_sampling =
      absolute_boards != nullptr &&
      absolute_board_rank_pattern_ids != nullptr &&
      absolute_board_flush_meta != nullptr &&
      rank_pattern_scores != nullptr;
  uint8_t remaining[kRemainingAfterHoleCards];
  if (!use_absolute_sampling) {
    fill_remaining_deck(hero_first, hero_second, villain_first, villain_second, remaining);
  }
  const uint64_t dead_mask =
      (static_cast<uint64_t>(1) << static_cast<uint64_t>(hero_first)) |
      (static_cast<uint64_t>(1) << static_cast<uint64_t>(hero_second)) |
      (static_cast<uint64_t>(1) << static_cast<uint64_t>(villain_first)) |
      (static_cast<uint64_t>(1) << static_cast<uint64_t>(villain_second));

  uint8_t board[kBoardCardCount];
  int win_count = 0;
  int tie_count = 0;
  int loss_count = 0;

  const int global_idx = idx + index_offset;
  const uint64_t local_seed = mix64(
      static_cast<uint64_t>(monte_carlo_seed_base) ^ static_cast<uint64_t>(key_material[idx]));
  uint64_t state = mix64(local_seed ^ static_cast<uint64_t>(global_idx + 1));
  if (state == 0ULL) {
    state = 0x9E3779B97F4A7C15ULL;
  }

  for (int trial = 0; trial < trials; ++trial) {
    int cmp = 0;
    if (use_absolute_sampling) {
      const int board_idx = sample_absolute_board_index_non_overlapping(
          absolute_boards,
          state,
          dead_mask);
      const uint8_t flush_meta = absolute_board_flush_meta[board_idx];
      const int board_max_suit_index = static_cast<int>(flush_meta & 0x03u);
      const int board_max_suit_count = static_cast<int>((flush_meta >> 2) & 0x07u);
      const bool hero_flush_possible = flush_possible_for_board_meta(
          board_max_suit_count,
          board_max_suit_index,
          hero_first,
          hero_second);
      const bool villain_flush_possible = flush_possible_for_board_meta(
          board_max_suit_count,
          board_max_suit_index,
          villain_first,
          villain_second);
      if (!hero_flush_possible && !villain_flush_possible) {
        const int rank_pattern_id = static_cast<int>(absolute_board_rank_pattern_ids[board_idx]);
        if (rank_pattern_id >= 0 && rank_pattern_id < kRankPatternCount) {
          cmp = compare_showdown_rank_lookup_only(
              hero_first,
              hero_second,
              villain_first,
              villain_second,
              rank_pattern_id,
              rank_pattern_scores);
        } else {
          load_absolute_board_cards(absolute_boards, board_idx, board);
          cmp = compare_showdown(
              hero_first,
              hero_second,
              villain_first,
              villain_second,
              board,
              rank_pattern_scores);
        }
      } else {
        load_absolute_board_cards(absolute_boards, board_idx, board);
        cmp = compare_showdown(
            hero_first,
            hero_second,
            villain_first,
            villain_second,
            board,
            rank_pattern_scores);
      }
    } else {
      if (board_combos != nullptr) {
        sample_board_cards_from_combos(remaining, board_combos, state, board);
      } else {
        sample_board_cards(remaining, state, board);
      }
      cmp = compare_showdown(
          hero_first,
          hero_second,
          villain_first,
          villain_second,
          board,
          rank_pattern_scores);
    }
    win_count  += (cmp > 0);
    tie_count  += (cmp == 0);
    loss_count += (cmp < 0);
  }

  const double total = static_cast<double>(trials);
  const double std_error = monte_carlo_stderr(win_count, tie_count, trials);

  wins[idx] = static_cast<jfloat>(static_cast<double>(win_count) / total);
  ties[idx] = static_cast<jfloat>(static_cast<double>(tie_count) / total);
  losses[idx] = static_cast<jfloat>(static_cast<double>(loss_count) / total);
  stderrs[idx] = static_cast<jfloat>(std_error);
}

/* Packed-key variant of monte_carlo_kernel_parallel_trials.
 * One block per matchup with shared-memory parallel trial reduction. */
__global__ void monte_carlo_kernel_packed_parallel_trials(
    const jint* packed_keys,
    const jlong* key_material,
    const int n,
    const int index_offset,
    const int trials,
    const jlong monte_carlo_seed_base,
    const uint8_t* board_combos,
    const uint8_t* absolute_boards,
    const uint16_t* absolute_board_rank_pattern_ids,
    const uint8_t* absolute_board_flush_meta,
    const uint32_t* rank_pattern_scores,
    jfloat* wins,
    jfloat* ties,
    jfloat* losses,
    jfloat* stderrs,
    int* status) {
  const int matchup_idx = static_cast<int>(blockIdx.x);
  if (matchup_idx >= n) {
    return;
  }

  __shared__ int hero_first;
  __shared__ int hero_second;
  __shared__ int villain_first;
  __shared__ int villain_second;
  __shared__ int valid_matchup;
  __shared__ int use_absolute_sampling;
  __shared__ uint8_t remaining[kRemainingAfterHoleCards];
  __shared__ uint64_t dead_mask;

  if (threadIdx.x == 0) {
    const int low_id = unpack_low_id(packed_keys[matchup_idx]);
    const int high_id = unpack_high_id(packed_keys[matchup_idx]);
    if (low_id < 0 || low_id >= kHoleCardsCount || high_id < 0 || high_id >= kHoleCardsCount) {
      set_status_once(status, 125);
      valid_matchup = 0;
    } else {
      hero_first = static_cast<int>(d_hole_first[low_id]);
      hero_second = static_cast<int>(d_hole_second[low_id]);
      villain_first = static_cast<int>(d_hole_first[high_id]);
      villain_second = static_cast<int>(d_hole_second[high_id]);
      const bool overlap = hero_first == villain_first || hero_first == villain_second ||
                           hero_second == villain_first || hero_second == villain_second;
      if (overlap) {
        set_status_once(status, 127);
        valid_matchup = 0;
      } else {
        use_absolute_sampling =
            absolute_boards != nullptr &&
            absolute_board_rank_pattern_ids != nullptr &&
            absolute_board_flush_meta != nullptr &&
            rank_pattern_scores != nullptr ? 1 : 0;
        if (use_absolute_sampling == 0) {
          fill_remaining_deck(hero_first, hero_second, villain_first, villain_second, remaining);
        }
        dead_mask =
            (static_cast<uint64_t>(1) << static_cast<uint64_t>(hero_first)) |
            (static_cast<uint64_t>(1) << static_cast<uint64_t>(hero_second)) |
            (static_cast<uint64_t>(1) << static_cast<uint64_t>(villain_first)) |
            (static_cast<uint64_t>(1) << static_cast<uint64_t>(villain_second));
        valid_matchup = 1;
      }
    }
  }
  __syncthreads();
  if (valid_matchup == 0) {
    return;
  }

  const int global_idx = matchup_idx + index_offset;
  const uint64_t local_seed = mix64(
      static_cast<uint64_t>(monte_carlo_seed_base) ^ static_cast<uint64_t>(key_material[matchup_idx]));
  uint64_t state = mix64(
      local_seed ^
      static_cast<uint64_t>(global_idx + 1) ^
      (static_cast<uint64_t>(threadIdx.x + 1) << 32));
  if (state == 0ULL) {
    state = 0x9E3779B97F4A7C15ULL;
  }

  uint32_t local_win = 0;
  uint32_t local_tie = 0;
  uint32_t local_loss = 0;
  uint8_t board[kBoardCardCount];
  const int hero_first_local = hero_first;
  const int hero_second_local = hero_second;
  const int villain_first_local = villain_first;
  const int villain_second_local = villain_second;
  const uint64_t dead_mask_local = dead_mask;
  const bool use_absolute_sampling_local = use_absolute_sampling != 0;
  for (int trial = static_cast<int>(threadIdx.x); trial < trials; trial += static_cast<int>(blockDim.x)) {
    int cmp = 0;
    if (use_absolute_sampling_local) {
      const int board_idx = sample_absolute_board_index_non_overlapping(
          absolute_boards,
          state,
          dead_mask_local);
      const uint8_t flush_meta = absolute_board_flush_meta[board_idx];
      const int board_max_suit_index = static_cast<int>(flush_meta & 0x03u);
      const int board_max_suit_count = static_cast<int>((flush_meta >> 2) & 0x07u);
      const bool hero_flush_possible = flush_possible_for_board_meta(
          board_max_suit_count,
          board_max_suit_index,
          hero_first_local,
          hero_second_local);
      const bool villain_flush_possible = flush_possible_for_board_meta(
          board_max_suit_count,
          board_max_suit_index,
          villain_first_local,
          villain_second_local);
      if (!hero_flush_possible && !villain_flush_possible) {
        const int rank_pattern_id = static_cast<int>(absolute_board_rank_pattern_ids[board_idx]);
        if (rank_pattern_id >= 0 && rank_pattern_id < kRankPatternCount) {
          cmp = compare_showdown_rank_lookup_only(
              hero_first_local,
              hero_second_local,
              villain_first_local,
              villain_second_local,
              rank_pattern_id,
              rank_pattern_scores);
        } else {
          load_absolute_board_cards(absolute_boards, board_idx, board);
          cmp = compare_showdown(
              hero_first_local,
              hero_second_local,
              villain_first_local,
              villain_second_local,
              board,
              rank_pattern_scores);
        }
      } else {
        load_absolute_board_cards(absolute_boards, board_idx, board);
        cmp = compare_showdown(
            hero_first_local,
            hero_second_local,
            villain_first_local,
            villain_second_local,
            board,
            rank_pattern_scores);
      }
    } else {
      if (board_combos != nullptr) {
        sample_board_cards_from_combos(remaining, board_combos, state, board);
      } else {
        sample_board_cards(remaining, state, board);
      }
      cmp = compare_showdown(
          hero_first_local,
          hero_second_local,
          villain_first_local,
          villain_second_local,
          board,
          rank_pattern_scores);
    }
    local_win  += (cmp > 0);
    local_tie  += (cmp == 0);
    local_loss += (cmp < 0);
  }

  extern __shared__ unsigned int reduction[];
  unsigned int* win_counts = reduction;
  unsigned int* tie_counts = reduction + blockDim.x;
  unsigned int* loss_counts = reduction + (2 * blockDim.x);
  const int tid = static_cast<int>(threadIdx.x);
  win_counts[tid] = local_win;
  tie_counts[tid] = local_tie;
  loss_counts[tid] = local_loss;
  __syncthreads();

  int active = static_cast<int>(blockDim.x);
  while (active > 1) {
    const int half = (active + 1) >> 1;
    if (tid < half) {
      const int other = tid + half;
      if (other < active) {
        win_counts[tid] += win_counts[other];
        tie_counts[tid] += tie_counts[other];
        loss_counts[tid] += loss_counts[other];
      }
    }
    __syncthreads();
    active = half;
  }

  if (tid == 0) {
    const int win_count = static_cast<int>(win_counts[0]);
    const int tie_count = static_cast<int>(tie_counts[0]);
    const int loss_count = static_cast<int>(loss_counts[0]);
    const double total = static_cast<double>(trials);
    const double std_error = monte_carlo_stderr(win_count, tie_count, trials);
    wins[matchup_idx] = static_cast<jfloat>(static_cast<double>(win_count) / total);
    ties[matchup_idx] = static_cast<jfloat>(static_cast<double>(tie_count) / total);
    losses[matchup_idx] = static_cast<jfloat>(static_cast<double>(loss_count) / total);
    stderrs[matchup_idx] = static_cast<jfloat>(std_error);
  }
}

/* Packed-key exact enumeration kernel (one block per matchup). */
__global__ void exact_kernel_packed(
    const jint* packed_keys,
    const int n,
    const uint8_t* board_combos,
    jfloat* wins,
    jfloat* ties,
    jfloat* losses,
    jfloat* stderrs,
    int* status) {
  const int matchup_idx = static_cast<int>(blockIdx.x);
  if (matchup_idx >= n) {
    return;
  }

  __shared__ int hero_first;
  __shared__ int hero_second;
  __shared__ int villain_first;
  __shared__ int villain_second;
  __shared__ int valid_matchup;
  __shared__ uint8_t remaining[kRemainingAfterHoleCards];

  if (threadIdx.x == 0) {
    const int low_id = unpack_low_id(packed_keys[matchup_idx]);
    const int high_id = unpack_high_id(packed_keys[matchup_idx]);
    if (low_id < 0 || low_id >= kHoleCardsCount || high_id < 0 || high_id >= kHoleCardsCount) {
      set_status_once(status, 125);
      valid_matchup = 0;
    } else {
      hero_first = static_cast<int>(d_hole_first[low_id]);
      hero_second = static_cast<int>(d_hole_second[low_id]);
      villain_first = static_cast<int>(d_hole_first[high_id]);
      villain_second = static_cast<int>(d_hole_second[high_id]);
      const bool overlap = hero_first == villain_first || hero_first == villain_second ||
                           hero_second == villain_first || hero_second == villain_second;
      if (overlap) {
        set_status_once(status, 127);
        valid_matchup = 0;
      } else {
        fill_remaining_deck(hero_first, hero_second, villain_first, villain_second, remaining);
        valid_matchup = 1;
      }
    }
  }
  __syncthreads();
  if (valid_matchup == 0) {
    return;
  }

  uint32_t local_win = 0;
  uint32_t local_tie = 0;
  uint32_t local_loss = 0;
  uint8_t board[kBoardCardCount];

  for (int combo = static_cast<int>(threadIdx.x); combo < kExactBoardCount; combo += static_cast<int>(blockDim.x)) {
    const int base = combo * kBoardCardCount;
    board[0] = remaining[board_combos[base + 0]];
    board[1] = remaining[board_combos[base + 1]];
    board[2] = remaining[board_combos[base + 2]];
    board[3] = remaining[board_combos[base + 3]];
    board[4] = remaining[board_combos[base + 4]];

    const int cmp = compare_showdown(hero_first, hero_second, villain_first, villain_second, board);
    local_win  += (cmp > 0);
    local_tie  += (cmp == 0);
    local_loss += (cmp < 0);
  }

  extern __shared__ unsigned int reduction[];
  unsigned int* win_counts = reduction;
  unsigned int* tie_counts = reduction + blockDim.x;
  unsigned int* loss_counts = reduction + (2 * blockDim.x);
  const int tid = static_cast<int>(threadIdx.x);
  win_counts[tid] = local_win;
  tie_counts[tid] = local_tie;
  loss_counts[tid] = local_loss;
  __syncthreads();

  int active = static_cast<int>(blockDim.x);
  while (active > 1) {
    const int half = (active + 1) >> 1;
    if (tid < half) {
      const int other = tid + half;
      if (other < active) {
        win_counts[tid] += win_counts[other];
        tie_counts[tid] += tie_counts[other];
        loss_counts[tid] += loss_counts[other];
      }
    }
    __syncthreads();
    active = half;
  }

  if (tid == 0) {
    const double total_d = static_cast<double>(kExactBoardCount);
    wins[matchup_idx] = static_cast<jfloat>(static_cast<double>(win_counts[0]) / total_d);
    ties[matchup_idx] = static_cast<jfloat>(static_cast<double>(tie_counts[0]) / total_d);
    losses[matchup_idx] = static_cast<jfloat>(static_cast<double>(loss_counts[0]) / total_d);
    stderrs[matchup_idx] = 0.0f;
  }
}

/* ---- Board-major exact kernels -------------------------------------------
 * Two-phase approach for exact equity on packed matchups:
 *   Phase 1 (prepare): For each board chunk, compute scores for all unique
 *     hole-card "endpoints" and store in a [board_count x endpoint_count] matrix.
 *   Phase 2 (accumulate): For each matchup, look up hero/villain endpoint
 *     scores across all boards in the chunk and increment win/tie/loss counters.
 * This amortizes board setup cost when many matchups share the same endpoints. */

/* Phase 1: compute hand scores for each (board, endpoint) pair.
 * Thread 0 loads board cards, builds rank/suit state, and computes
 * the rank-pattern lookup base.  All threads evaluate endpoints in
 * parallel using the shared board state, with a fast rank-only lookup
 * path when no flush is possible. */
__global__ void exact_prepare_endpoint_scores_kernel(
    const uint8_t* absolute_boards,
    const int board_start,
    const int board_count,
    const int boards_per_block,
    const uint16_t* endpoint_hole_ids,
    const uint32_t* hole_cards,
    const uint32_t* rank_pattern_scores,
    const int endpoint_count,
    uint32_t* out_scores,
    int* status) {
  const int board_group = static_cast<int>(blockIdx.x);
  __shared__ uint8_t shared_board_cards[kBoardCardCount];
  __shared__ uint8_t shared_board_rank_counts[kMaxRankValue + 1];
  __shared__ uint8_t shared_board_suit_counts[4];
  __shared__ uint16_t shared_board_rank_mask;
  __shared__ uint16_t shared_board_suit_rank_mask[4];
  __shared__ uint16_t shared_board_pair_mask;
  __shared__ uint16_t shared_board_trip_mask;
  __shared__ uint16_t shared_board_quad_mask;
  __shared__ uint64_t shared_board_card_mask;
  __shared__ uint8_t shared_board_max_suit_count;
  __shared__ uint8_t shared_board_max_suit_index;
  __shared__ int shared_rank_pattern_id;
  __shared__ uint32_t shared_rank_only_lookup[kRanksPerSuit * kRanksPerSuit];

  for (int board_offset = 0; board_offset < boards_per_block; ++board_offset) {
    const int board_local = board_group * boards_per_block + board_offset;
    if (board_local >= board_count) {
      return;
    }
    const int board_global = board_start + board_local;
    if (board_global < 0 || board_global >= kAbsoluteBoardCount) {
      set_status_once(status, 125);
      return;
    }

    if (threadIdx.x == 0) {
      const int board_base = board_global * kBoardCardCount;
      shared_board_cards[0] = absolute_boards[board_base + 0];
      shared_board_cards[1] = absolute_boards[board_base + 1];
      shared_board_cards[2] = absolute_boards[board_base + 2];
      shared_board_cards[3] = absolute_boards[board_base + 3];
      shared_board_cards[4] = absolute_boards[board_base + 4];
      for (int rank = 0; rank <= kMaxRankValue; ++rank) {
        shared_board_rank_counts[rank] = static_cast<uint8_t>(0);
      }
      shared_board_suit_counts[0] = 0;
      shared_board_suit_counts[1] = 0;
      shared_board_suit_counts[2] = 0;
      shared_board_suit_counts[3] = 0;
      shared_board_rank_mask = 0;
      shared_board_suit_rank_mask[0] = 0;
      shared_board_suit_rank_mask[1] = 0;
      shared_board_suit_rank_mask[2] = 0;
      shared_board_suit_rank_mask[3] = 0;
      shared_board_pair_mask = 0;
      shared_board_trip_mask = 0;
      shared_board_quad_mask = 0;
      shared_board_card_mask = 0ULL;
      shared_board_max_suit_count = 0;
      shared_board_max_suit_index = 0;
      shared_rank_pattern_id = -1;
      for (int i = 0; i < kBoardCardCount; ++i) {
        const int card = static_cast<int>(shared_board_cards[i]);
        const int rank = card_rank(card);
        const int suit = card_suit(card);
        const uint16_t bit = card_rank_bit(card);
        ++shared_board_rank_counts[rank];
        ++shared_board_suit_counts[suit];
        shared_board_rank_mask |= bit;
        shared_board_suit_rank_mask[suit] |= bit;
        shared_board_card_mask |= static_cast<uint64_t>(1) << static_cast<uint64_t>(card);
      }
      for (int rank = kMinRankValue; rank <= kMaxRankValue; ++rank) {
        const uint8_t count = shared_board_rank_counts[rank];
        const uint16_t bit = static_cast<uint16_t>(1) << static_cast<uint16_t>(rank - kMinRankValue);
        if (count >= 2) {
          shared_board_pair_mask = static_cast<uint16_t>(shared_board_pair_mask | bit);
        }
        if (count >= 3) {
          shared_board_trip_mask = static_cast<uint16_t>(shared_board_trip_mask | bit);
        }
        if (count == 4) {
          shared_board_quad_mask = static_cast<uint16_t>(shared_board_quad_mask | bit);
        }
      }
      for (int suit = 0; suit < 4; ++suit) {
        const uint8_t count = shared_board_suit_counts[suit];
        if (count > shared_board_max_suit_count) {
          shared_board_max_suit_count = count;
          shared_board_max_suit_index = static_cast<uint8_t>(suit);
        }
      }
      int sorted_offsets[5] = {0, 0, 0, 0, 0};
      int sorted_idx = 0;
      for (int rank_offset = 0; rank_offset < kRanksPerSuit; ++rank_offset) {
        const int rank = rank_offset + kMinRankValue;
        const int count = static_cast<int>(shared_board_rank_counts[rank]);
        for (int copy = 0; copy < count && sorted_idx < 5; ++copy) {
          sorted_offsets[sorted_idx++] = rank_offset;
        }
      }
      if (sorted_idx != 5) {
        shared_rank_pattern_id = -1;
      } else {
        shared_rank_pattern_id = rank_multiset5_id_from_offsets(sorted_offsets);
      }
    }
    __syncthreads();
    if (shared_rank_pattern_id < 0 || shared_rank_pattern_id >= kRankPatternCount) {
      set_status_once(status, 125);
      return;
    }

    const uint16_t board_rank_mask = shared_board_rank_mask;
    const uint16_t board_pair_mask = shared_board_pair_mask;
    const uint16_t board_trip_mask = shared_board_trip_mask;
    const uint16_t board_quad_mask = shared_board_quad_mask;
    const uint8_t board_suit_count0 = shared_board_suit_counts[0];
    const uint8_t board_suit_count1 = shared_board_suit_counts[1];
    const uint8_t board_suit_count2 = shared_board_suit_counts[2];
    const uint8_t board_suit_count3 = shared_board_suit_counts[3];
    const uint16_t board_suit_rank_mask0 = shared_board_suit_rank_mask[0];
    const uint16_t board_suit_rank_mask1 = shared_board_suit_rank_mask[1];
    const uint16_t board_suit_rank_mask2 = shared_board_suit_rank_mask[2];
    const uint16_t board_suit_rank_mask3 = shared_board_suit_rank_mask[3];
    const uint64_t board_card_mask = shared_board_card_mask;
    const int board_max_suit_count = static_cast<int>(shared_board_max_suit_count);
    const int board_max_suit_index = static_cast<int>(shared_board_max_suit_index);
    const size_t rank_lookup_base =
        static_cast<size_t>(shared_rank_pattern_id) * static_cast<size_t>(kRankPairCount);

    for (int rank_pair_idx = static_cast<int>(threadIdx.x);
         rank_pair_idx < kRankPairCount;
         rank_pair_idx += static_cast<int>(blockDim.x)) {
      shared_rank_only_lookup[rank_pair_idx] =
          rank_pattern_scores[rank_lookup_base + static_cast<size_t>(rank_pair_idx)];
    }
    __syncthreads();

    for (int endpoint_idx = static_cast<int>(threadIdx.x);
         endpoint_idx < endpoint_count;
         endpoint_idx += static_cast<int>(blockDim.x)) {
      const int hole_id = static_cast<int>(endpoint_hole_ids[endpoint_idx]);
      if (hole_id < 0 || hole_id >= kHoleCardsCount) {
        set_status_once(status, 125);
        return;
      }
      const uint32_t packed_hole = hole_cards[hole_id];
      const int first = static_cast<int>(packed_hole & 0x3Fu);
      const int second = static_cast<int>((packed_hole >> 6) & 0x3Fu);
      const uint64_t hole_mask =
          (static_cast<uint64_t>(1) << static_cast<uint64_t>(first)) |
          (static_cast<uint64_t>(1) << static_cast<uint64_t>(second));
      const bool overlap = (hole_mask & board_card_mask) != 0ULL;
      const size_t out_idx =
          static_cast<size_t>(board_local) * static_cast<size_t>(endpoint_count) +
          static_cast<size_t>(endpoint_idx);
      if (overlap) {
        out_scores[out_idx] = kInvalidScoreSentinel;
        continue;
      }

      const int first_rank_offset = static_cast<int>((packed_hole >> 12) & 0x0Fu);
      const int second_rank_offset = static_cast<int>((packed_hole >> 16) & 0x0Fu);
      const int first_rank = first_rank_offset + kMinRankValue;
      const int second_rank = second_rank_offset + kMinRankValue;
      const int first_suit = static_cast<int>((packed_hole >> 20) & 0x03u);
      const int second_suit = static_cast<int>((packed_hole >> 22) & 0x03u);
      const uint16_t first_bit = static_cast<uint16_t>(1) << static_cast<uint16_t>(first_rank_offset);
      const uint16_t second_bit = static_cast<uint16_t>(1) << static_cast<uint16_t>(second_rank_offset);
      bool flush_possible = false;
      if (board_max_suit_count >= 5) {
        flush_possible = true;
      } else if (board_max_suit_count == 4) {
        flush_possible = first_suit == board_max_suit_index || second_suit == board_max_suit_index;
      } else if (board_max_suit_count == 3) {
        flush_possible = first_suit == board_max_suit_index && second_suit == board_max_suit_index;
      }
      if (!flush_possible) {
        out_scores[out_idx] =
            shared_rank_only_lookup[first_rank_offset * kRanksPerSuit + second_rank_offset];
        continue;
      }

      const uint16_t rank_mask = static_cast<uint16_t>(board_rank_mask | first_bit | second_bit);
      uint16_t pair_mask = board_pair_mask;
      uint16_t trip_mask = board_trip_mask;
      uint16_t quad_mask = board_quad_mask;
      if (first_rank == second_rank) {
        const int updated = static_cast<int>(shared_board_rank_counts[first_rank]) + 2;
        const uint16_t bit = first_bit;
        pair_mask = static_cast<uint16_t>(
            (pair_mask & static_cast<uint16_t>(~bit)) | (updated >= 2 ? bit : 0));
        trip_mask = static_cast<uint16_t>(
            (trip_mask & static_cast<uint16_t>(~bit)) | (updated >= 3 ? bit : 0));
        quad_mask = static_cast<uint16_t>(
            (quad_mask & static_cast<uint16_t>(~bit)) | (updated == 4 ? bit : 0));
      } else {
        const int updated_first = static_cast<int>(shared_board_rank_counts[first_rank]) + 1;
        pair_mask = static_cast<uint16_t>(
            (pair_mask & static_cast<uint16_t>(~first_bit)) | (updated_first >= 2 ? first_bit : 0));
        trip_mask = static_cast<uint16_t>(
            (trip_mask & static_cast<uint16_t>(~first_bit)) | (updated_first >= 3 ? first_bit : 0));
        quad_mask = static_cast<uint16_t>(
            (quad_mask & static_cast<uint16_t>(~first_bit)) | (updated_first == 4 ? first_bit : 0));

        const int updated_second = static_cast<int>(shared_board_rank_counts[second_rank]) + 1;
        pair_mask = static_cast<uint16_t>(
            (pair_mask & static_cast<uint16_t>(~second_bit)) | (updated_second >= 2 ? second_bit : 0));
        trip_mask = static_cast<uint16_t>(
            (trip_mask & static_cast<uint16_t>(~second_bit)) | (updated_second >= 3 ? second_bit : 0));
        quad_mask = static_cast<uint16_t>(
            (quad_mask & static_cast<uint16_t>(~second_bit)) | (updated_second == 4 ? second_bit : 0));
      }

      uint8_t suit_counts[4] = {
          board_suit_count0,
          board_suit_count1,
          board_suit_count2,
          board_suit_count3,
      };
      ++suit_counts[first_suit];
      ++suit_counts[second_suit];

      uint16_t suit_rank_mask[4] = {
          board_suit_rank_mask0,
          board_suit_rank_mask1,
          board_suit_rank_mask2,
          board_suit_rank_mask3,
      };
      suit_rank_mask[first_suit] = static_cast<uint16_t>(suit_rank_mask[first_suit] | first_bit);
      suit_rank_mask[second_suit] = static_cast<uint16_t>(suit_rank_mask[second_suit] | second_bit);

      out_scores[out_idx] =
          evaluate7_score_from_masks(rank_mask, pair_mask, trip_mask, quad_mask, suit_counts, suit_rank_mask);
    }
    __syncthreads();
  }
}

/* Phase 2: accumulate matchup results from precomputed endpoint scores.
 * Each thread handles one matchup, iterating over board_count boards
 * and comparing hero vs villain scores.  Skips boards where either
 * endpoint overlaps (kInvalidScoreSentinel). */
__global__ void exact_accumulate_from_endpoint_scores_kernel(
    const uint32_t* board_scores,
    const int board_count,
    const int endpoint_count,
    const uint16_t* hero_endpoint_index,
    const uint16_t* villain_endpoint_index,
    const int n,
    uint32_t* total_wins,
    uint32_t* total_ties,
    uint32_t* total_losses,
    int* status) {
  const int matchup_idx =
      static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if (matchup_idx >= n) {
    return;
  }
  const int hero_idx = static_cast<int>(hero_endpoint_index[matchup_idx]);
  const int villain_idx = static_cast<int>(villain_endpoint_index[matchup_idx]);
  if (hero_idx < 0 || hero_idx >= endpoint_count || villain_idx < 0 || villain_idx >= endpoint_count) {
    set_status_once(status, 125);
    return;
  }

  uint32_t wins = 0;
  uint32_t ties = 0;
  uint32_t losses = 0;
  for (int board_local = 0; board_local < board_count; ++board_local) {
    const size_t base = static_cast<size_t>(board_local) * static_cast<size_t>(endpoint_count);
    const uint32_t hero_score = board_scores[base + static_cast<size_t>(hero_idx)];
    const uint32_t villain_score = board_scores[base + static_cast<size_t>(villain_idx)];
    if (hero_score == kInvalidScoreSentinel || villain_score == kInvalidScoreSentinel) {
      continue;
    }
    if (hero_score > villain_score) {
      ++wins;
    } else if (hero_score < villain_score) {
      ++losses;
    } else {
      ++ties;
    }
  }

  total_wins[matchup_idx] += wins;
  total_ties[matchup_idx] += ties;
  total_losses[matchup_idx] += losses;
}

/* ---- Range-vs-range CSR kernels ------------------------------------------
 * These kernels compute equity for a hero hand against a weighted villain
 * range stored in CSR (Compressed Sparse Row) format. */

/* Flat CSR kernel (one thread per CSR entry):
 * Each thread runs MC equity for one (hero, villain) pair, then atomicAdd
 * the probability-weighted results into the hero's accumulator slots.
 * Simple but suffers from atomic contention when many entries share a hero. */
template <bool UseReadOnly>
__global__ void range_monte_carlo_csr_kernel(
    const jint* hero_ids,
    const jint* entry_hero_index,
    const jint* villain_ids,
    const jlong* key_material,
    const jfloat* probabilities,
    const int entry_count,
    const int entry_index_offset,
    const int trials,
    const jlong monte_carlo_seed_base,
    jfloat* accum_wins,
    jfloat* accum_ties,
    jfloat* accum_losses,
    jfloat* accum_stderr_var,
    jfloat* accum_weights,
    int* status) {
  const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if (idx >= entry_count) {
    return;
  }

  const int hero_index = static_cast<int>(load_jint<UseReadOnly>(entry_hero_index, idx));
  const int hero_id = static_cast<int>(load_jint<UseReadOnly>(hero_ids, hero_index));
  const int villain_id = static_cast<int>(load_jint<UseReadOnly>(villain_ids, idx));
  if (hero_id < 0 || hero_id >= kHoleCardsCount || villain_id < 0 || villain_id >= kHoleCardsCount) {
    set_status_once(status, 125);
    return;
  }
  const float p = static_cast<float>(load_jfloat<UseReadOnly>(probabilities, idx));
  if (!isfinite(static_cast<double>(p)) || p < 0.0f) {
    set_status_once(status, kStatusInvalidRangeLayout);
    return;
  }
  if (p == 0.0f) {
    return;
  }

  const int hero_first = static_cast<int>(d_hole_first[hero_id]);
  const int hero_second = static_cast<int>(d_hole_second[hero_id]);
  const int villain_first = static_cast<int>(d_hole_first[villain_id]);
  const int villain_second = static_cast<int>(d_hole_second[villain_id]);
  const bool overlap = hero_first == villain_first || hero_first == villain_second ||
                       hero_second == villain_first || hero_second == villain_second;
  if (overlap) {
    set_status_once(status, 127);
    return;
  }

  uint8_t remaining[kRemainingAfterHoleCards];
  fill_remaining_deck(hero_first, hero_second, villain_first, villain_second, remaining);

  uint8_t board[kBoardCardCount];
  int win_count = 0;
  int tie_count = 0;
  int loss_count = 0;

  const jlong local_key_material = load_jlong<UseReadOnly>(key_material, idx);
  const uint64_t local_seed = mix64(
      static_cast<uint64_t>(monte_carlo_seed_base) ^ static_cast<uint64_t>(local_key_material));
  const int global_entry_index = idx + entry_index_offset;
  uint64_t state = mix64(local_seed ^ static_cast<uint64_t>(global_entry_index + 1));
  if (state == 0ULL) {
    state = 0x9E3779B97F4A7C15ULL;
  }

  for (int trial = 0; trial < trials; ++trial) {
    sample_board_cards(remaining, state, board);
    const int cmp = compare_showdown(hero_first, hero_second, villain_first, villain_second, board);
    win_count  += (cmp > 0);
    tie_count  += (cmp == 0);
    loss_count += (cmp < 0);
  }

  const float total = static_cast<float>(trials);
  const float win_rate = static_cast<float>(win_count) / total;
  const float tie_rate = static_cast<float>(tie_count) / total;
  const float loss_rate = static_cast<float>(loss_count) / total;
  const float stderr_value = static_cast<float>(monte_carlo_stderr(win_count, tie_count, trials));

  atomicAdd(accum_wins + hero_index, p * win_rate);
  atomicAdd(accum_ties + hero_index, p * tie_rate);
  atomicAdd(accum_losses + hero_index, p * loss_rate);
  atomicAdd(accum_stderr_var + hero_index, (p * stderr_value) * (p * stderr_value));
  atomicAdd(accum_weights + hero_index, p);
}

/* By-hero CSR kernel (one BLOCK per hero):
 * Threads within a block cooperatively iterate over the hero's villain
 * entries with stride = blockDim.x, accumulating weighted equity locally.
 * A shared-memory tree reduction aggregates per-thread results, producing
 * one final equity per hero.  Avoids atomics entirely.
 * This is the default range kernel used for production. */
template <bool UseReadOnly>
__global__ void range_monte_carlo_csr_by_hero_kernel(
    const jint* hero_ids,
    const jint* offsets,
    const jint* villain_ids,
    const jlong* key_material,
    const jfloat* probabilities,
    const int entry_count,
    const int hero_count_chunk,
    const int hero_index_offset,
    const int trials,
    const jlong monte_carlo_seed_base,
    jfloat* out_wins,
    jfloat* out_ties,
    jfloat* out_losses,
    jfloat* out_stderrs,
    int* status) {
  const int local_hero_index = static_cast<int>(blockIdx.x);
  if (local_hero_index >= hero_count_chunk) {
    return;
  }
  const int hero_index = hero_index_offset + local_hero_index;

  const int hero_id = static_cast<int>(load_jint<UseReadOnly>(hero_ids, hero_index));
  if (hero_id < 0 || hero_id >= kHoleCardsCount) {
    set_status_once(status, 125);
    return;
  }
  const int start = static_cast<int>(load_jint<UseReadOnly>(offsets, hero_index));
  const int end = static_cast<int>(load_jint<UseReadOnly>(offsets, hero_index + 1));
  if (start < 0 || end < start || end > entry_count) {
    set_status_once(status, kStatusInvalidRangeLayout);
    return;
  }

  const int hero_first = static_cast<int>(d_hole_first[hero_id]);
  const int hero_second = static_cast<int>(d_hole_second[hero_id]);
  const int tid = static_cast<int>(threadIdx.x);

  float local_weighted_win = 0.0f;
  float local_weighted_tie = 0.0f;
  float local_weighted_loss = 0.0f;
  float local_weighted_stderr_var = 0.0f;
  float local_weight_sum = 0.0f;

  for (int i = start + tid; i < end; i += static_cast<int>(blockDim.x)) {
    const int villain_id = static_cast<int>(load_jint<UseReadOnly>(villain_ids, i));
    if (villain_id < 0 || villain_id >= kHoleCardsCount) {
      set_status_once(status, 125);
      continue;
    }
    const float p = static_cast<float>(load_jfloat<UseReadOnly>(probabilities, i));
    if (!isfinite(static_cast<double>(p)) || p < 0.0f) {
      set_status_once(status, kStatusInvalidRangeLayout);
      continue;
    }
    if (p == 0.0f) {
      continue;
    }

    const int villain_first = static_cast<int>(d_hole_first[villain_id]);
    const int villain_second = static_cast<int>(d_hole_second[villain_id]);
    const bool overlap = hero_first == villain_first || hero_first == villain_second ||
                         hero_second == villain_first || hero_second == villain_second;
    if (overlap) {
      set_status_once(status, 127);
      continue;
    }

    uint8_t remaining[kRemainingAfterHoleCards];
    fill_remaining_deck(hero_first, hero_second, villain_first, villain_second, remaining);

    uint8_t board[kBoardCardCount];
    int win_count = 0;
    int tie_count = 0;
    int loss_count = 0;

    const jlong local_key_material = load_jlong<UseReadOnly>(key_material, i);
    const uint64_t local_seed =
        mix64(static_cast<uint64_t>(monte_carlo_seed_base) ^ static_cast<uint64_t>(local_key_material));
    uint64_t state = mix64(local_seed ^ static_cast<uint64_t>(i + 1));
    if (state == 0ULL) {
      state = 0x9E3779B97F4A7C15ULL;
    }

    for (int trial = 0; trial < trials; ++trial) {
      sample_board_cards(remaining, state, board);
      const int cmp = compare_showdown(hero_first, hero_second, villain_first, villain_second, board);
      win_count  += (cmp > 0);
      tie_count  += (cmp == 0);
      loss_count += (cmp < 0);
    }

    const float total = static_cast<float>(trials);
    const float win_rate = static_cast<float>(win_count) / total;
    const float tie_rate = static_cast<float>(tie_count) / total;
    const float loss_rate = static_cast<float>(loss_count) / total;
    const float stderr_value = static_cast<float>(monte_carlo_stderr(win_count, tie_count, trials));

    local_weighted_win += p * win_rate;
    local_weighted_tie += p * tie_rate;
    local_weighted_loss += p * loss_rate;
    local_weighted_stderr_var += (p * stderr_value) * (p * stderr_value);
    local_weight_sum += p;
  }

  extern __shared__ float range_reduction[];
  float* win_values = range_reduction;
  float* tie_values = win_values + blockDim.x;
  float* loss_values = tie_values + blockDim.x;
  float* stderr_values = loss_values + blockDim.x;
  float* weight_values = stderr_values + blockDim.x;
  win_values[tid] = local_weighted_win;
  tie_values[tid] = local_weighted_tie;
  loss_values[tid] = local_weighted_loss;
  stderr_values[tid] = local_weighted_stderr_var;
  weight_values[tid] = local_weight_sum;
  __syncthreads();

  int active = static_cast<int>(blockDim.x);
  while (active > 1) {
    const int half = (active + 1) >> 1;
    if (tid < half) {
      const int other = tid + half;
      if (other < active) {
        win_values[tid] += win_values[other];
        tie_values[tid] += tie_values[other];
        loss_values[tid] += loss_values[other];
        stderr_values[tid] += stderr_values[other];
        weight_values[tid] += weight_values[other];
      }
    }
    __syncthreads();
    active = half;
  }

  if (tid == 0) {
    const float weight = weight_values[0];
    if (!isfinite(static_cast<double>(weight)) || weight <= 0.0f) {
      out_wins[hero_index] = 0.0f;
      out_ties[hero_index] = 0.0f;
      out_losses[hero_index] = 0.0f;
      out_stderrs[hero_index] = 0.0f;
    } else {
      const float inv_weight = 1.0f / weight;
      out_wins[hero_index] = win_values[0] * inv_weight;
      out_ties[hero_index] = tie_values[0] * inv_weight;
      out_losses[hero_index] = loss_values[0] * inv_weight;
      out_stderrs[hero_index] = sqrtf(fmaxf(stderr_values[0], 0.0f)) * inv_weight;
    }
  }
}

/* ---- CUDA batch dispatch --------------------------------------------------
 * These functions handle device setup, memory allocation, chunked kernel
 * dispatch (to stay under Windows TDR timeout), and result copy-back.
 * Each function checks configuration, uploads lookup tables on first use,
 * allocates device buffers, launches kernels in chunks, reads back results,
 * and frees device memory. */

/* CUDA batch dispatch for separate hero/villain ID arrays (legacy API).
 * Supports both exact (mode_code=0) and Monte Carlo (mode_code=1) modes. */
int compute_batch_cuda(
    JNIEnv* env,
    const std::vector<jint>& low_buf,
    const std::vector<jint>& high_buf,
    const std::vector<jlong>& seed_buf,
    const jint mode_code,
    const jint trials,
    std::vector<jdouble>& win_buf,
    std::vector<jdouble>& tie_buf,
    std::vector<jdouble>& loss_buf,
    std::vector<jdouble>& stderr_buf,
    const int target_device = -1) {
  const int n = static_cast<int>(low_buf.size());
  if (n <= 0) {
    return 0;
  }

  auto report_cuda_error = [&](const int status_code, const cudaError_t error, const char* stage) -> int {
    std::fprintf(
        stderr,
        "[sicfun-gpu-native] CUDA failure at %s: %s (code=%d)\n",
        stage,
        cudaGetErrorString(error),
        static_cast<int>(error));
    if (error == cudaErrorLaunchTimeout) {
      return 137;
    }
    return status_code;
  };

  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess) {
    return report_cuda_error(130, err, "cudaGetDeviceCount");
  }
  if (device_count <= 0) {
    std::fprintf(stderr, "[sicfun-gpu-native] CUDA failure at cudaGetDeviceCount: no CUDA devices found\n");
    return 130;
  }

  int device = 0;
  if (target_device >= 0) {
    if (target_device >= device_count) {
      std::fprintf(stderr, "[sicfun-gpu-native] invalid CUDA device index %d (count=%d)\n", target_device, device_count);
      return 138;
    }
    err = cudaSetDevice(target_device);
    if (err != cudaSuccess) {
      return report_cuda_error(130, err, "cudaSetDevice");
    }
    device = target_device;
  } else {
    err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
      return report_cuda_error(130, err, "cudaGetDevice");
    }
  }
  cudaDeviceProp prop{};
  err = cudaGetDeviceProperties(&prop, device);
  if (err != cudaSuccess) {
    return report_cuda_error(130, err, "cudaGetDeviceProperties");
  }
  if (!ensure_cuda_lookup_uploaded_for_device(device)) {
    std::fprintf(stderr, "[sicfun-gpu-native] CUDA failure at lookup upload: cudaMemcpyToSymbol failed\n");
    return 136;
  }
  const bool use_board_combos_mc = mode_code != 0 && resolve_monte_carlo_use_board_combos(env);
  const uint8_t* d_exact_board_combos = nullptr;
  if (mode_code == 0 || use_board_combos_mc) {
    if (!ensure_cuda_exact_board_indices_uploaded_for_device(device, &d_exact_board_combos)) {
      if (mode_code == 0) {
        std::fprintf(stderr, "[sicfun-gpu-native] CUDA failure at exact-board upload: cudaMalloc/cudaMemcpy failed\n");
        return 136;
      }
      d_exact_board_combos = nullptr;
    }
  }
  const bool use_rank_lookup_mc = mode_code != 0 && resolve_monte_carlo_use_rank_lookup(env);
  const uint32_t* d_rank_pattern_scores = nullptr;
  if (use_rank_lookup_mc && !ensure_cuda_rank_pattern_scores_uploaded_for_device(device, &d_rank_pattern_scores)) {
    d_rank_pattern_scores = nullptr;
  }
  const bool use_absolute_board_sampling_mc =
      mode_code != 0 &&
      d_rank_pattern_scores != nullptr &&
      resolve_monte_carlo_absolute_board_sampling(env);
  const uint8_t* d_absolute_boards = nullptr;
  const uint16_t* d_absolute_board_rank_pattern_ids = nullptr;
  const uint8_t* d_absolute_board_flush_meta = nullptr;
  if (use_absolute_board_sampling_mc) {
    const bool has_boards = ensure_cuda_absolute_board_cards_uploaded_for_device(device, &d_absolute_boards);
    const bool has_metadata = ensure_cuda_absolute_board_metadata_uploaded_for_device(
        device,
        &d_absolute_board_rank_pattern_ids,
        &d_absolute_board_flush_meta);
    if (!has_boards || !has_metadata) {
      d_absolute_boards = nullptr;
      d_absolute_board_rank_pattern_ids = nullptr;
      d_absolute_board_flush_meta = nullptr;
    }
  }
  const int max_chunk_matchups = resolve_cuda_max_chunk_matchups(env, n, mode_code);

  jint* d_low = nullptr;
  jint* d_high = nullptr;
  jlong* d_seeds = nullptr;
  jdouble* d_wins = nullptr;
  jdouble* d_ties = nullptr;
  jdouble* d_losses = nullptr;
  jdouble* d_stderrs = nullptr;
  int* d_status = nullptr;

  auto free_all = [&]() {
    if (d_low != nullptr) cudaFree(d_low);
    if (d_high != nullptr) cudaFree(d_high);
    if (d_seeds != nullptr) cudaFree(d_seeds);
    if (d_wins != nullptr) cudaFree(d_wins);
    if (d_ties != nullptr) cudaFree(d_ties);
    if (d_losses != nullptr) cudaFree(d_losses);
    if (d_stderrs != nullptr) cudaFree(d_stderrs);
    if (d_status != nullptr) cudaFree(d_status);
  };

  const size_t total_size = static_cast<size_t>(n);
  if (cudaMalloc(reinterpret_cast<void**>(&d_low), total_size * sizeof(jint)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_high), total_size * sizeof(jint)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_seeds), total_size * sizeof(jlong)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_wins), total_size * sizeof(jdouble)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_ties), total_size * sizeof(jdouble)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_losses), total_size * sizeof(jdouble)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_stderrs), total_size * sizeof(jdouble)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_status), sizeof(int)) != cudaSuccess) {
    free_all();
    std::fprintf(stderr, "[sicfun-gpu-native] CUDA failure at cudaMalloc\n");
    return 131;
  }

  err = cudaMemcpy(d_low, low_buf.data(), total_size * sizeof(jint), cudaMemcpyHostToDevice);
  if (err == cudaSuccess) {
    err = cudaMemcpy(d_high, high_buf.data(), total_size * sizeof(jint), cudaMemcpyHostToDevice);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(d_seeds, seed_buf.data(), total_size * sizeof(jlong), cudaMemcpyHostToDevice);
  }
  if (err != cudaSuccess) {
    free_all();
    return report_cuda_error(132, err, "cudaMemcpy host->device");
  }

  int threads_per_block = resolve_cuda_threads_per_block(env, mode_code);
  threads_per_block = std::min(threads_per_block, prop.maxThreadsPerBlock);
  if (threads_per_block <= 0) {
    threads_per_block = mode_code == 0 ? kDefaultCudaThreadsPerBlockExact : kDefaultCudaThreadsPerBlock;
  }
  const bool parallel_trials_mc =
      mode_code != 0 &&
      resolve_monte_carlo_parallel_trials(env) &&
      trials >= 64 &&
      threads_per_block >= 64;

  for (int offset = 0; offset < n; offset += max_chunk_matchups) {
    const int chunk = std::min(max_chunk_matchups, n - offset);
    const int zero = 0;

    err = cudaMemcpy(d_status, &zero, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      free_all();
      return report_cuda_error(132, err, "cudaMemcpy host->device");
    }

    int blocks = (chunk + threads_per_block - 1) / threads_per_block;
    blocks = std::max(1, blocks);
    if (mode_code == 0) {
      const size_t reduction_bytes =
          static_cast<size_t>(threads_per_block) * 3ULL * sizeof(unsigned int);
      exact_kernel<<<chunk, threads_per_block, reduction_bytes>>>(
          d_low + static_cast<size_t>(offset),
          d_high + static_cast<size_t>(offset),
          chunk,
          d_exact_board_combos,
          d_wins + static_cast<size_t>(offset),
          d_ties + static_cast<size_t>(offset),
          d_losses + static_cast<size_t>(offset),
          d_stderrs + static_cast<size_t>(offset),
          d_status);
    } else {
      if (parallel_trials_mc) {
        const size_t reduction_bytes =
            static_cast<size_t>(threads_per_block) * 3ULL * sizeof(unsigned int);
        monte_carlo_kernel_parallel_trials<<<chunk, threads_per_block, reduction_bytes>>>(
            d_low + static_cast<size_t>(offset),
            d_high + static_cast<size_t>(offset),
            d_seeds + static_cast<size_t>(offset),
            chunk,
            offset,
            trials,
            d_exact_board_combos,
            d_absolute_boards,
            d_absolute_board_rank_pattern_ids,
            d_absolute_board_flush_meta,
            d_rank_pattern_scores,
            d_wins + static_cast<size_t>(offset),
            d_ties + static_cast<size_t>(offset),
            d_losses + static_cast<size_t>(offset),
            d_stderrs + static_cast<size_t>(offset),
            d_status);
      } else {
        monte_carlo_kernel<<<blocks, threads_per_block>>>(
            d_low + static_cast<size_t>(offset),
            d_high + static_cast<size_t>(offset),
            d_seeds + static_cast<size_t>(offset),
            chunk,
            offset,
            trials,
            d_exact_board_combos,
            d_absolute_boards,
            d_absolute_board_rank_pattern_ids,
            d_absolute_board_flush_meta,
            d_rank_pattern_scores,
            d_wins + static_cast<size_t>(offset),
            d_ties + static_cast<size_t>(offset),
            d_losses + static_cast<size_t>(offset),
            d_stderrs + static_cast<size_t>(offset),
            d_status);
      }
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      free_all();
      return report_cuda_error(133, err, "kernel launch");
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      free_all();
      return report_cuda_error(134, err, "cudaDeviceSynchronize");
    }

    int status = 0;
    err = cudaMemcpy(&status, d_status, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      free_all();
      return report_cuda_error(135, err, "cudaMemcpy status");
    }
    if (status != 0) {
      free_all();
      return status;
    }
  }

  err = cudaMemcpy(win_buf.data(), d_wins, total_size * sizeof(jdouble), cudaMemcpyDeviceToHost);
  if (err == cudaSuccess) {
    err = cudaMemcpy(tie_buf.data(), d_ties, total_size * sizeof(jdouble), cudaMemcpyDeviceToHost);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(loss_buf.data(), d_losses, total_size * sizeof(jdouble), cudaMemcpyDeviceToHost);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(stderr_buf.data(), d_stderrs, total_size * sizeof(jdouble), cudaMemcpyDeviceToHost);
  }
  if (err != cudaSuccess) {
    free_all();
    return report_cuda_error(135, err, "cudaMemcpy device->host");
  }
  free_all();
  return 0;
}

/* Board-major exact path for packed matchup keys.
 * Uses two-phase kernel dispatch (prepare + accumulate) to amortize board
 * evaluation across shared endpoints.  Processes boards in configurable chunks
 * (default 8192) to bound GPU memory usage.  Includes optional profiling
 * output controlled by sicfun.gpu.native.exact.boardMajor.profile. */
int compute_batch_cuda_packed_exact_board_major(
    JNIEnv* env,
    const std::vector<jint>& packed_buf,
    std::vector<jfloat>& win_buf,
    std::vector<jfloat>& tie_buf,
    std::vector<jfloat>& loss_buf,
    std::vector<jfloat>& stderr_buf,
    const int target_device = -1) {
  const int n = static_cast<int>(packed_buf.size());
  if (n <= 0) {
    return 0;
  }
  const auto started_at = std::chrono::steady_clock::now();

  auto report_cuda_error = [&](const int status_code, const cudaError_t error, const char* stage) -> int {
    std::fprintf(
        stderr,
        "[sicfun-gpu-native] CUDA failure at %s: %s (code=%d)\n",
        stage,
        cudaGetErrorString(error),
        static_cast<int>(error));
    if (error == cudaErrorLaunchTimeout) {
      return 137;
    }
    return status_code;
  };

  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess) {
    return report_cuda_error(130, err, "cudaGetDeviceCount");
  }
  if (device_count <= 0) {
    std::fprintf(stderr, "[sicfun-gpu-native] CUDA failure at cudaGetDeviceCount: no CUDA devices found\n");
    return 130;
  }

  int device = 0;
  if (target_device >= 0) {
    if (target_device >= device_count) {
      std::fprintf(stderr, "[sicfun-gpu-native] invalid CUDA device index %d (count=%d)\n", target_device, device_count);
      return 138;
    }
    err = cudaSetDevice(target_device);
    if (err != cudaSuccess) {
      return report_cuda_error(130, err, "cudaSetDevice");
    }
    device = target_device;
  } else {
    err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
      return report_cuda_error(130, err, "cudaGetDevice");
    }
  }
  cudaDeviceProp prop{};
  err = cudaGetDeviceProperties(&prop, device);
  if (err != cudaSuccess) {
    return report_cuda_error(130, err, "cudaGetDeviceProperties");
  }
  const uint8_t* d_absolute_boards = nullptr;
  if (!ensure_cuda_absolute_board_cards_uploaded_for_device(device, &d_absolute_boards)) {
    std::fprintf(stderr, "[sicfun-gpu-native] CUDA failure at absolute-board upload: cudaMalloc/cudaMemcpy failed\n");
    return 136;
  }
  const uint32_t* d_hole_cards = nullptr;
  if (!ensure_cuda_hole_cards_uploaded_for_device(device, &d_hole_cards)) {
    std::fprintf(stderr, "[sicfun-gpu-native] CUDA failure at hole-card upload: cudaMalloc/cudaMemcpy failed\n");
    return 136;
  }
  const uint32_t* d_rank_pattern_scores = nullptr;
  if (!ensure_cuda_rank_pattern_scores_uploaded_for_device(device, &d_rank_pattern_scores)) {
    std::fprintf(stderr, "[sicfun-gpu-native] CUDA failure at rank-pattern upload: cudaMalloc/cudaMemcpy failed\n");
    return 136;
  }

  std::vector<uint16_t> endpoint_hole_ids;
  endpoint_hole_ids.reserve(kHoleCardsCount);
  std::vector<uint16_t> hole_to_endpoint(static_cast<size_t>(kHoleCardsCount), std::numeric_limits<uint16_t>::max());
  std::vector<uint16_t> hero_endpoint(static_cast<size_t>(n));
  std::vector<uint16_t> villain_endpoint(static_cast<size_t>(n));
  const auto& lookup = hole_cards_lookup();

  for (int i = 0; i < n; ++i) {
    const jint packed = packed_buf[static_cast<size_t>(i)];
    const int low_id = unpack_low_id(packed);
    const int high_id = unpack_high_id(packed);
    if (low_id < 0 || low_id >= kHoleCardsCount || high_id < 0 || high_id >= kHoleCardsCount) {
      return 125;
    }
    const HoleCards hero = lookup[static_cast<size_t>(low_id)];
    const HoleCards villain = lookup[static_cast<size_t>(high_id)];
    const int hero_first = static_cast<int>(hero.first);
    const int hero_second = static_cast<int>(hero.second);
    const int villain_first = static_cast<int>(villain.first);
    const int villain_second = static_cast<int>(villain.second);
    const bool overlap = hero_first == villain_first || hero_first == villain_second ||
                         hero_second == villain_first || hero_second == villain_second;
    if (overlap) {
      return 127;
    }

    uint16_t low_endpoint = hole_to_endpoint[static_cast<size_t>(low_id)];
    if (low_endpoint == std::numeric_limits<uint16_t>::max()) {
      low_endpoint = static_cast<uint16_t>(endpoint_hole_ids.size());
      endpoint_hole_ids.push_back(static_cast<uint16_t>(low_id));
      hole_to_endpoint[static_cast<size_t>(low_id)] = low_endpoint;
    }
    uint16_t high_endpoint = hole_to_endpoint[static_cast<size_t>(high_id)];
    if (high_endpoint == std::numeric_limits<uint16_t>::max()) {
      high_endpoint = static_cast<uint16_t>(endpoint_hole_ids.size());
      endpoint_hole_ids.push_back(static_cast<uint16_t>(high_id));
      hole_to_endpoint[static_cast<size_t>(high_id)] = high_endpoint;
    }
    hero_endpoint[static_cast<size_t>(i)] = low_endpoint;
    villain_endpoint[static_cast<size_t>(i)] = high_endpoint;
  }

  const int endpoint_count = static_cast<int>(endpoint_hole_ids.size());
  if (endpoint_count <= 0 || endpoint_count > kHoleCardsCount) {
    return 125;
  }
  const auto after_host_endpoint_setup = std::chrono::steady_clock::now();

  uint16_t* d_endpoint_hole_ids = nullptr;
  uint16_t* d_hero_endpoint = nullptr;
  uint16_t* d_villain_endpoint = nullptr;
  uint32_t* d_total_wins = nullptr;
  uint32_t* d_total_ties = nullptr;
  uint32_t* d_total_losses = nullptr;
  uint32_t* d_board_scores = nullptr;
  int* d_status = nullptr;

  auto free_all = [&]() {
    if (d_endpoint_hole_ids != nullptr) cudaFree(d_endpoint_hole_ids);
    if (d_hero_endpoint != nullptr) cudaFree(d_hero_endpoint);
    if (d_villain_endpoint != nullptr) cudaFree(d_villain_endpoint);
    if (d_total_wins != nullptr) cudaFree(d_total_wins);
    if (d_total_ties != nullptr) cudaFree(d_total_ties);
    if (d_total_losses != nullptr) cudaFree(d_total_losses);
    if (d_board_scores != nullptr) cudaFree(d_board_scores);
    if (d_status != nullptr) cudaFree(d_status);
  };

  if (cudaMalloc(reinterpret_cast<void**>(&d_endpoint_hole_ids),
                 static_cast<size_t>(endpoint_count) * sizeof(uint16_t)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_hero_endpoint),
                 static_cast<size_t>(n) * sizeof(uint16_t)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_villain_endpoint),
                 static_cast<size_t>(n) * sizeof(uint16_t)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_total_wins),
                 static_cast<size_t>(n) * sizeof(uint32_t)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_total_ties),
                 static_cast<size_t>(n) * sizeof(uint32_t)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_total_losses),
                 static_cast<size_t>(n) * sizeof(uint32_t)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_status), sizeof(int)) != cudaSuccess) {
    free_all();
    std::fprintf(stderr, "[sicfun-gpu-native] CUDA failure at cudaMalloc\n");
    return 131;
  }

  err = cudaMemcpy(
      d_endpoint_hole_ids,
      endpoint_hole_ids.data(),
      static_cast<size_t>(endpoint_count) * sizeof(uint16_t),
      cudaMemcpyHostToDevice);
  if (err == cudaSuccess) {
    err = cudaMemcpy(
        d_hero_endpoint,
        hero_endpoint.data(),
        static_cast<size_t>(n) * sizeof(uint16_t),
        cudaMemcpyHostToDevice);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(
        d_villain_endpoint,
        villain_endpoint.data(),
        static_cast<size_t>(n) * sizeof(uint16_t),
        cudaMemcpyHostToDevice);
  }
  if (err != cudaSuccess) {
    free_all();
    return report_cuda_error(132, err, "cudaMemcpy host->device");
  }

  err = cudaMemset(d_total_wins, 0, static_cast<size_t>(n) * sizeof(uint32_t));
  if (err == cudaSuccess) {
    err = cudaMemset(d_total_ties, 0, static_cast<size_t>(n) * sizeof(uint32_t));
  }
  if (err == cudaSuccess) {
  err = cudaMemset(d_total_losses, 0, static_cast<size_t>(n) * sizeof(uint32_t));
  }
  if (err != cudaSuccess) {
    free_all();
    return report_cuda_error(131, err, "cudaMemset");
  }

  auto resolve_positive_setting = [&](const char* property_name, const char* env_name, const int fallback) -> int {
    std::string property_value;
    if (try_read_system_property(env, property_name, property_value)) {
      const int parsed = parse_positive_env_int(property_value.c_str());
      if (parsed > 0) {
        return parsed;
      }
    }
    const int env_value = parse_positive_env_int(std::getenv(env_name));
    if (env_value > 0) {
      return env_value;
    }
    return fallback;
  };
  auto resolve_truthy_setting = [&](const char* property_name, const char* env_name) -> bool {
    std::string property_value;
    if (try_read_system_property(env, property_name, property_value)) {
      return parse_truthy(property_value);
    }
    return parse_truthy(std::getenv(env_name));
  };
  const bool profile_enabled = resolve_truthy_setting(
      "sicfun.gpu.native.exact.boardMajor.profile",
      "sicfun_GPU_EXACT_BOARD_MAJOR_PROFILE");

  int boards_per_chunk = resolve_positive_setting(
      "sicfun.gpu.native.exact.boardMajor.chunkBoards",
      "sicfun_GPU_EXACT_BOARD_MAJOR_CHUNK_BOARDS",
      8192);
  boards_per_chunk = std::max(128, std::min(kAbsoluteBoardCount, boards_per_chunk));
  while (boards_per_chunk >= 128 && d_board_scores == nullptr) {
    const size_t bytes = static_cast<size_t>(boards_per_chunk) *
                         static_cast<size_t>(endpoint_count) * sizeof(uint32_t);
    if (cudaMalloc(reinterpret_cast<void**>(&d_board_scores), bytes) != cudaSuccess) {
      boards_per_chunk /= 2;
    }
  }
  if (d_board_scores == nullptr) {
    free_all();
    std::fprintf(stderr, "[sicfun-gpu-native] CUDA failure at board-score allocation\n");
    return 131;
  }

  int score_threads = resolve_positive_setting(
      "sicfun.gpu.native.exact.boardMajor.scoreThreads",
      "sicfun_GPU_EXACT_BOARD_MAJOR_SCORE_THREADS",
      256);
  score_threads = std::max(32, std::min(score_threads, prop.maxThreadsPerBlock));
  score_threads = std::max(32, (score_threads / 32) * 32);

  int match_threads = resolve_positive_setting(
      "sicfun.gpu.native.exact.boardMajor.matchThreads",
      "sicfun_GPU_EXACT_BOARD_MAJOR_MATCH_THREADS",
      256);
  match_threads = std::max(32, std::min(match_threads, prop.maxThreadsPerBlock));
  match_threads = std::max(32, (match_threads / 32) * 32);

  int prepare_boards_per_block = resolve_positive_setting(
      "sicfun.gpu.native.exact.boardMajor.prepareBoardsPerBlock",
      "sicfun_GPU_EXACT_BOARD_MAJOR_PREPARE_BOARDS_PER_BLOCK",
      1);
  prepare_boards_per_block = std::max(1, std::min(prepare_boards_per_block, 8));

  const int zero = 0;
  err = cudaMemcpy(d_status, &zero, sizeof(int), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    free_all();
    return report_cuda_error(132, err, "cudaMemcpy host->device");
  }
  const auto after_device_setup = std::chrono::steady_clock::now();

  const int matchup_blocks = std::max(1, (n + match_threads - 1) / match_threads);
  const auto kernel_launch_started = std::chrono::steady_clock::now();
  double profile_prepare_seconds = 0.0;
  double profile_accumulate_seconds = 0.0;
  for (int board_start = 0; board_start < kAbsoluteBoardCount; board_start += boards_per_chunk) {
    const int board_count = std::min(boards_per_chunk, kAbsoluteBoardCount - board_start);
    const int prepare_blocks =
        std::max(1, (board_count + prepare_boards_per_block - 1) / prepare_boards_per_block);
    const auto prepare_started = profile_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    exact_prepare_endpoint_scores_kernel<<<prepare_blocks, score_threads>>>(
        d_absolute_boards,
        board_start,
        board_count,
        prepare_boards_per_block,
        d_endpoint_hole_ids,
        d_hole_cards,
        d_rank_pattern_scores,
        endpoint_count,
        d_board_scores,
        d_status);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      free_all();
      return report_cuda_error(133, err, "kernel launch");
    }
    if (profile_enabled) {
      err = cudaDeviceSynchronize();
      if (err != cudaSuccess) {
        free_all();
        return report_cuda_error(134, err, "cudaDeviceSynchronize");
      }
      profile_prepare_seconds += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - prepare_started).count();
    }
    const auto accumulate_started = profile_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    exact_accumulate_from_endpoint_scores_kernel<<<matchup_blocks, match_threads>>>(
        d_board_scores,
        board_count,
        endpoint_count,
        d_hero_endpoint,
        d_villain_endpoint,
        n,
        d_total_wins,
        d_total_ties,
        d_total_losses,
        d_status);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      free_all();
      return report_cuda_error(133, err, "kernel launch");
    }
    if (profile_enabled) {
      err = cudaDeviceSynchronize();
      if (err != cudaSuccess) {
        free_all();
        return report_cuda_error(134, err, "cudaDeviceSynchronize");
      }
      profile_accumulate_seconds += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - accumulate_started).count();
    }
  }
  const auto kernel_launch_finished = std::chrono::steady_clock::now();

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    free_all();
    return report_cuda_error(134, err, "cudaDeviceSynchronize");
  }
  const auto after_kernel_sync = std::chrono::steady_clock::now();

  int status = 0;
  err = cudaMemcpy(&status, d_status, sizeof(int), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    free_all();
    return report_cuda_error(135, err, "cudaMemcpy status");
  }
  if (status != 0) {
    free_all();
    return status;
  }

  std::vector<uint32_t> wins_count(static_cast<size_t>(n));
  std::vector<uint32_t> ties_count(static_cast<size_t>(n));
  std::vector<uint32_t> losses_count(static_cast<size_t>(n));
  err = cudaMemcpy(
      wins_count.data(),
      d_total_wins,
      static_cast<size_t>(n) * sizeof(uint32_t),
      cudaMemcpyDeviceToHost);
  if (err == cudaSuccess) {
    err = cudaMemcpy(
        ties_count.data(),
        d_total_ties,
        static_cast<size_t>(n) * sizeof(uint32_t),
        cudaMemcpyDeviceToHost);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(
        losses_count.data(),
        d_total_losses,
        static_cast<size_t>(n) * sizeof(uint32_t),
        cudaMemcpyDeviceToHost);
  }
  if (err != cudaSuccess) {
    free_all();
    return report_cuda_error(135, err, "cudaMemcpy device->host");
  }
  const auto after_device_to_host = std::chrono::steady_clock::now();

  for (int i = 0; i < n; ++i) {
    const uint32_t wins = wins_count[static_cast<size_t>(i)];
    const uint32_t ties = ties_count[static_cast<size_t>(i)];
    const uint32_t losses = losses_count[static_cast<size_t>(i)];
    const uint32_t total = wins + ties + losses;
    if (total == 0) {
      win_buf[static_cast<size_t>(i)] = 0.0f;
      tie_buf[static_cast<size_t>(i)] = 0.0f;
      loss_buf[static_cast<size_t>(i)] = 0.0f;
      stderr_buf[static_cast<size_t>(i)] = 0.0f;
      continue;
    }
    const float inv_total = 1.0f / static_cast<float>(total);
    win_buf[static_cast<size_t>(i)] = static_cast<float>(wins) * inv_total;
    tie_buf[static_cast<size_t>(i)] = static_cast<float>(ties) * inv_total;
    loss_buf[static_cast<size_t>(i)] = static_cast<float>(losses) * inv_total;
    stderr_buf[static_cast<size_t>(i)] = 0.0f;
  }
  const auto finished_at = std::chrono::steady_clock::now();
  if (profile_enabled) {
    const auto to_seconds = [](const auto delta) -> double {
      return std::chrono::duration<double>(delta).count();
    };
    const double host_endpoint_setup_s = to_seconds(after_host_endpoint_setup - started_at);
    const double device_setup_s = to_seconds(after_device_setup - after_host_endpoint_setup);
    const double kernel_launch_s = to_seconds(kernel_launch_finished - kernel_launch_started);
    const double kernel_wait_s = to_seconds(after_kernel_sync - kernel_launch_finished);
    const double status_and_copy_s = to_seconds(after_device_to_host - after_kernel_sync);
    const double normalize_s = to_seconds(finished_at - after_device_to_host);
    const double total_s = to_seconds(finished_at - started_at);
    std::fprintf(
        stderr,
        "[sicfun-gpu-native] board-major profile: endpoints=%d n=%d chunk=%d scoreThreads=%d matchThreads=%d "
        "prepareBoardsPerBlock=%d hostSetup=%.3fs deviceSetup=%.3fs launch=%.3fs wait=%.3fs "
        "prepareKernel=%.3fs accumulateKernel=%.3fs copy=%.3fs normalize=%.3fs total=%.3fs\n",
        endpoint_count,
        n,
        boards_per_chunk,
        score_threads,
        match_threads,
        prepare_boards_per_block,
        host_endpoint_setup_s,
        device_setup_s,
        kernel_launch_s,
        kernel_wait_s,
        profile_prepare_seconds,
        profile_accumulate_seconds,
        status_and_copy_s,
        normalize_s,
        total_s);
  }

  free_all();
  return 0;
}

/* CUDA batch dispatch for packed 22-bit matchup keys (compact API).
 * For exact mode with boardMajor enabled, delegates to the board-major path.
 * For MC mode, selects between one-thread-per-matchup and parallel-trials
 * kernels based on trial count and configuration. */
int compute_batch_cuda_packed(
    JNIEnv* env,
    const std::vector<jint>& packed_buf,
    const std::vector<jlong>& key_material_buf,
    const jint mode_code,
    const jint trials,
    const jlong monte_carlo_seed_base,
    std::vector<jfloat>& win_buf,
    std::vector<jfloat>& tie_buf,
    std::vector<jfloat>& loss_buf,
    std::vector<jfloat>& stderr_buf,
    const int target_device = -1) {
  const int n = static_cast<int>(packed_buf.size());
  if (n <= 0) {
    return 0;
  }
  if (mode_code == 0 && resolve_exact_board_major_enabled(env)) {
    return compute_batch_cuda_packed_exact_board_major(
        env,
        packed_buf,
        win_buf,
        tie_buf,
        loss_buf,
        stderr_buf,
        target_device);
  }

  auto report_cuda_error = [&](const int status_code, const cudaError_t error, const char* stage) -> int {
    std::fprintf(
        stderr,
        "[sicfun-gpu-native] CUDA failure at %s: %s (code=%d)\n",
        stage,
        cudaGetErrorString(error),
        static_cast<int>(error));
    if (error == cudaErrorLaunchTimeout) {
      return 137;
    }
    return status_code;
  };

  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess) {
    return report_cuda_error(130, err, "cudaGetDeviceCount");
  }
  if (device_count <= 0) {
    std::fprintf(stderr, "[sicfun-gpu-native] CUDA failure at cudaGetDeviceCount: no CUDA devices found\n");
    return 130;
  }

  int device = 0;
  if (target_device >= 0) {
    if (target_device >= device_count) {
      std::fprintf(stderr, "[sicfun-gpu-native] invalid CUDA device index %d (count=%d)\n", target_device, device_count);
      return 138;
    }
    err = cudaSetDevice(target_device);
    if (err != cudaSuccess) {
      return report_cuda_error(130, err, "cudaSetDevice");
    }
    device = target_device;
  } else {
    err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
      return report_cuda_error(130, err, "cudaGetDevice");
    }
  }
  cudaDeviceProp prop{};
  err = cudaGetDeviceProperties(&prop, device);
  if (err != cudaSuccess) {
    return report_cuda_error(130, err, "cudaGetDeviceProperties");
  }
  if (!ensure_cuda_lookup_uploaded_for_device(device)) {
    std::fprintf(stderr, "[sicfun-gpu-native] CUDA failure at lookup upload: cudaMemcpyToSymbol failed\n");
    return 136;
  }
  const bool use_board_combos_mc = mode_code != 0 && resolve_monte_carlo_use_board_combos(env);
  const uint8_t* d_exact_board_combos = nullptr;
  if (mode_code == 0 || use_board_combos_mc) {
    if (!ensure_cuda_exact_board_indices_uploaded_for_device(device, &d_exact_board_combos)) {
      if (mode_code == 0) {
        std::fprintf(stderr, "[sicfun-gpu-native] CUDA failure at exact-board upload: cudaMalloc/cudaMemcpy failed\n");
        return 136;
      }
      d_exact_board_combos = nullptr;
    }
  }
  const bool use_rank_lookup_mc = mode_code != 0 && resolve_monte_carlo_use_rank_lookup(env);
  const uint32_t* d_rank_pattern_scores = nullptr;
  if (use_rank_lookup_mc && !ensure_cuda_rank_pattern_scores_uploaded_for_device(device, &d_rank_pattern_scores)) {
    d_rank_pattern_scores = nullptr;
  }
  const bool use_absolute_board_sampling_mc =
      mode_code != 0 &&
      d_rank_pattern_scores != nullptr &&
      resolve_monte_carlo_absolute_board_sampling(env);
  const uint8_t* d_absolute_boards = nullptr;
  const uint16_t* d_absolute_board_rank_pattern_ids = nullptr;
  const uint8_t* d_absolute_board_flush_meta = nullptr;
  if (use_absolute_board_sampling_mc) {
    const bool has_boards = ensure_cuda_absolute_board_cards_uploaded_for_device(device, &d_absolute_boards);
    const bool has_metadata = ensure_cuda_absolute_board_metadata_uploaded_for_device(
        device,
        &d_absolute_board_rank_pattern_ids,
        &d_absolute_board_flush_meta);
    if (!has_boards || !has_metadata) {
      d_absolute_boards = nullptr;
      d_absolute_board_rank_pattern_ids = nullptr;
      d_absolute_board_flush_meta = nullptr;
    }
  }

  const int max_chunk_matchups = resolve_cuda_max_chunk_matchups(env, n, mode_code);

  jint* d_packed = nullptr;
  jlong* d_key_material = nullptr;
  jfloat* d_wins = nullptr;
  jfloat* d_ties = nullptr;
  jfloat* d_losses = nullptr;
  jfloat* d_stderrs = nullptr;
  int* d_status = nullptr;

  auto free_all = [&]() {
    if (d_packed != nullptr) cudaFree(d_packed);
    if (d_key_material != nullptr) cudaFree(d_key_material);
    if (d_wins != nullptr) cudaFree(d_wins);
    if (d_ties != nullptr) cudaFree(d_ties);
    if (d_losses != nullptr) cudaFree(d_losses);
    if (d_stderrs != nullptr) cudaFree(d_stderrs);
    if (d_status != nullptr) cudaFree(d_status);
  };

  const size_t total_size = static_cast<size_t>(n);
  const bool needs_key_material = mode_code == 1;
  if (cudaMalloc(reinterpret_cast<void**>(&d_packed), total_size * sizeof(jint)) != cudaSuccess ||
      (needs_key_material &&
       cudaMalloc(reinterpret_cast<void**>(&d_key_material), total_size * sizeof(jlong)) != cudaSuccess) ||
      cudaMalloc(reinterpret_cast<void**>(&d_wins), total_size * sizeof(jfloat)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_ties), total_size * sizeof(jfloat)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_losses), total_size * sizeof(jfloat)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_stderrs), total_size * sizeof(jfloat)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_status), sizeof(int)) != cudaSuccess) {
    free_all();
    std::fprintf(stderr, "[sicfun-gpu-native] CUDA failure at cudaMalloc\n");
    return 131;
  }

  err = cudaMemcpy(d_packed, packed_buf.data(), total_size * sizeof(jint), cudaMemcpyHostToDevice);
  if (err == cudaSuccess && needs_key_material) {
    err = cudaMemcpy(d_key_material, key_material_buf.data(), total_size * sizeof(jlong), cudaMemcpyHostToDevice);
  }
  if (err != cudaSuccess) {
    free_all();
    return report_cuda_error(132, err, "cudaMemcpy host->device");
  }

  int threads_per_block = resolve_cuda_threads_per_block(env, mode_code);
  threads_per_block = std::min(threads_per_block, prop.maxThreadsPerBlock);
  if (threads_per_block <= 0) {
    threads_per_block = mode_code == 0 ? kDefaultCudaThreadsPerBlockExact : kDefaultCudaThreadsPerBlock;
  }
  const bool parallel_trials_mc =
      mode_code != 0 &&
      resolve_monte_carlo_parallel_trials(env) &&
      trials >= 64 &&
      threads_per_block >= 64;

  for (int offset = 0; offset < n; offset += max_chunk_matchups) {
    const int chunk = std::min(max_chunk_matchups, n - offset);
    const int zero = 0;

    err = cudaMemcpy(d_status, &zero, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      free_all();
      return report_cuda_error(132, err, "cudaMemcpy host->device");
    }

    int blocks = (chunk + threads_per_block - 1) / threads_per_block;
    blocks = std::max(1, blocks);
    if (mode_code == 0) {
      const size_t reduction_bytes =
          static_cast<size_t>(threads_per_block) * 3ULL * sizeof(unsigned int);
      exact_kernel_packed<<<chunk, threads_per_block, reduction_bytes>>>(
          d_packed + static_cast<size_t>(offset),
          chunk,
          d_exact_board_combos,
          d_wins + static_cast<size_t>(offset),
          d_ties + static_cast<size_t>(offset),
          d_losses + static_cast<size_t>(offset),
          d_stderrs + static_cast<size_t>(offset),
          d_status);
    } else {
      if (parallel_trials_mc) {
        const size_t reduction_bytes =
            static_cast<size_t>(threads_per_block) * 3ULL * sizeof(unsigned int);
        monte_carlo_kernel_packed_parallel_trials<<<chunk, threads_per_block, reduction_bytes>>>(
            d_packed + static_cast<size_t>(offset),
            d_key_material + static_cast<size_t>(offset),
            chunk,
            offset,
            trials,
            monte_carlo_seed_base,
            d_exact_board_combos,
            d_absolute_boards,
            d_absolute_board_rank_pattern_ids,
            d_absolute_board_flush_meta,
            d_rank_pattern_scores,
            d_wins + static_cast<size_t>(offset),
            d_ties + static_cast<size_t>(offset),
            d_losses + static_cast<size_t>(offset),
            d_stderrs + static_cast<size_t>(offset),
            d_status);
      } else {
        monte_carlo_kernel_packed<<<blocks, threads_per_block>>>(
            d_packed + static_cast<size_t>(offset),
            d_key_material + static_cast<size_t>(offset),
            chunk,
            offset,
            trials,
            monte_carlo_seed_base,
            d_exact_board_combos,
            d_absolute_boards,
            d_absolute_board_rank_pattern_ids,
            d_absolute_board_flush_meta,
            d_rank_pattern_scores,
            d_wins + static_cast<size_t>(offset),
            d_ties + static_cast<size_t>(offset),
            d_losses + static_cast<size_t>(offset),
            d_stderrs + static_cast<size_t>(offset),
            d_status);
      }
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      free_all();
      return report_cuda_error(133, err, "kernel launch");
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      free_all();
      return report_cuda_error(134, err, "cudaDeviceSynchronize");
    }

    int status = 0;
    err = cudaMemcpy(&status, d_status, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      free_all();
      return report_cuda_error(135, err, "cudaMemcpy status");
    }
    if (status != 0) {
      free_all();
      return status;
    }
  }

  err = cudaMemcpy(win_buf.data(), d_wins, total_size * sizeof(jfloat), cudaMemcpyDeviceToHost);
  if (err == cudaSuccess) {
    err = cudaMemcpy(tie_buf.data(), d_ties, total_size * sizeof(jfloat), cudaMemcpyDeviceToHost);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(loss_buf.data(), d_losses, total_size * sizeof(jfloat), cudaMemcpyDeviceToHost);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(stderr_buf.data(), d_stderrs, total_size * sizeof(jfloat), cudaMemcpyDeviceToHost);
  }
  if (err != cudaSuccess) {
    free_all();
    return report_cuda_error(135, err, "cudaMemcpy device->host");
  }
  free_all();
  return 0;
}

/* CUDA range-vs-range Monte Carlo dispatch via CSR layout.
 * Validates CSR structure (monotonic offsets, valid IDs, non-negative weights),
 * uploads all arrays to GPU, and launches range_monte_carlo_csr_by_hero_kernel
 * in chunks of max_chunk_heroes to avoid TDR timeout. */
int compute_range_batch_cuda_monte_carlo_csr(
    JNIEnv* env,
    const std::vector<jint>& hero_ids,
    const std::vector<jint>& offsets,
    const std::vector<jint>& villain_ids,
    const std::vector<jlong>& key_material,
    const std::vector<jfloat>& probabilities,
    const jint trials,
    const jlong monte_carlo_seed_base,
    std::vector<jfloat>& out_wins,
    std::vector<jfloat>& out_ties,
    std::vector<jfloat>& out_losses,
    std::vector<jfloat>& out_stderrs,
    const int target_device = -1) {
  const int hero_count = static_cast<int>(hero_ids.size());
  const int entry_count = static_cast<int>(villain_ids.size());
  if (hero_count <= 0) {
    return 0;
  }
  if (trials <= 0) {
    return 126;
  }
  if (static_cast<int>(offsets.size()) != hero_count + 1 ||
      static_cast<int>(key_material.size()) != entry_count ||
      static_cast<int>(probabilities.size()) != entry_count) {
    return 101;
  }
  if (offsets[0] != 0 || offsets[hero_count] != entry_count) {
    return kStatusInvalidRangeLayout;
  }

  for (int h = 0; h < hero_count; ++h) {
    if (offsets[static_cast<size_t>(h)] > offsets[static_cast<size_t>(h + 1)]) {
      return kStatusInvalidRangeLayout;
    }
    const int hero_id = static_cast<int>(hero_ids[static_cast<size_t>(h)]);
    if (hero_id < 0 || hero_id >= kHoleCardsCount) {
      return 125;
    }
  }
  for (int i = 0; i < entry_count; ++i) {
    const int villain_id = static_cast<int>(villain_ids[static_cast<size_t>(i)]);
    if (villain_id < 0 || villain_id >= kHoleCardsCount) {
      return 125;
    }
    const float p = probabilities[static_cast<size_t>(i)];
    if (!std::isfinite(p) || p < 0.0f) {
      return kStatusInvalidRangeLayout;
    }
  }

  if (entry_count == 0) {
    std::fill(out_wins.begin(), out_wins.end(), 0.0f);
    std::fill(out_ties.begin(), out_ties.end(), 0.0f);
    std::fill(out_losses.begin(), out_losses.end(), 0.0f);
    std::fill(out_stderrs.begin(), out_stderrs.end(), 0.0f);
    return 0;
  }

  auto report_cuda_error = [&](const int status_code, const cudaError_t error, const char* stage) -> int {
    std::fprintf(
        stderr,
        "[sicfun-gpu-native] CUDA failure at %s: %s (code=%d)\n",
        stage,
        cudaGetErrorString(error),
        static_cast<int>(error));
    if (error == cudaErrorLaunchTimeout) {
      return 137;
    }
    return status_code;
  };

  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess) {
    return report_cuda_error(130, err, "cudaGetDeviceCount");
  }
  if (device_count <= 0) {
    std::fprintf(stderr, "[sicfun-gpu-native] CUDA failure at cudaGetDeviceCount: no CUDA devices found\n");
    return 130;
  }

  int device = 0;
  if (target_device >= 0) {
    if (target_device >= device_count) {
      std::fprintf(stderr, "[sicfun-gpu-native] invalid CUDA device index %d (count=%d)\n", target_device, device_count);
      return 138;
    }
    err = cudaSetDevice(target_device);
    if (err != cudaSuccess) {
      return report_cuda_error(130, err, "cudaSetDevice");
    }
    device = target_device;
  } else {
    err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
      return report_cuda_error(130, err, "cudaGetDevice");
    }
  }

  cudaDeviceProp prop{};
  err = cudaGetDeviceProperties(&prop, device);
  if (err != cudaSuccess) {
    return report_cuda_error(130, err, "cudaGetDeviceProperties");
  }
  if (!ensure_cuda_lookup_uploaded_for_device(device)) {
    std::fprintf(stderr, "[sicfun-gpu-native] CUDA failure at lookup upload: cudaMemcpyToSymbol failed\n");
    return 136;
  }

  const int max_chunk_heroes = resolve_range_cuda_max_chunk_heroes(env, hero_count);
  const RangeMemoryPath memory_path = resolve_range_memory_path(env);

  jint* d_hero_ids = nullptr;
  jint* d_offsets = nullptr;
  jint* d_villain_ids = nullptr;
  jlong* d_key_material = nullptr;
  jfloat* d_probabilities = nullptr;
  jfloat* d_out_wins = nullptr;
  jfloat* d_out_ties = nullptr;
  jfloat* d_out_losses = nullptr;
  jfloat* d_out_stderrs = nullptr;
  int* d_status = nullptr;

  auto free_all = [&]() {
    if (d_hero_ids != nullptr) cudaFree(d_hero_ids);
    if (d_offsets != nullptr) cudaFree(d_offsets);
    if (d_villain_ids != nullptr) cudaFree(d_villain_ids);
    if (d_key_material != nullptr) cudaFree(d_key_material);
    if (d_probabilities != nullptr) cudaFree(d_probabilities);
    if (d_out_wins != nullptr) cudaFree(d_out_wins);
    if (d_out_ties != nullptr) cudaFree(d_out_ties);
    if (d_out_losses != nullptr) cudaFree(d_out_losses);
    if (d_out_stderrs != nullptr) cudaFree(d_out_stderrs);
    if (d_status != nullptr) cudaFree(d_status);
  };

  if (cudaMalloc(reinterpret_cast<void**>(&d_hero_ids), static_cast<size_t>(hero_count) * sizeof(jint)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_offsets), static_cast<size_t>(hero_count + 1) * sizeof(jint)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_villain_ids), static_cast<size_t>(entry_count) * sizeof(jint)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_key_material), static_cast<size_t>(entry_count) * sizeof(jlong)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_probabilities), static_cast<size_t>(entry_count) * sizeof(jfloat)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_out_wins), static_cast<size_t>(hero_count) * sizeof(jfloat)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_out_ties), static_cast<size_t>(hero_count) * sizeof(jfloat)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_out_losses), static_cast<size_t>(hero_count) * sizeof(jfloat)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_out_stderrs), static_cast<size_t>(hero_count) * sizeof(jfloat)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_status), sizeof(int)) != cudaSuccess) {
    free_all();
    std::fprintf(stderr, "[sicfun-gpu-native] CUDA failure at cudaMalloc\n");
    return 131;
  }

  err = cudaMemcpy(
      d_hero_ids,
      hero_ids.data(),
      static_cast<size_t>(hero_count) * sizeof(jint),
      cudaMemcpyHostToDevice);
  if (err == cudaSuccess) {
    err = cudaMemcpy(
        d_offsets,
        offsets.data(),
        static_cast<size_t>(hero_count + 1) * sizeof(jint),
        cudaMemcpyHostToDevice);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(
        d_villain_ids,
        villain_ids.data(),
        static_cast<size_t>(entry_count) * sizeof(jint),
        cudaMemcpyHostToDevice);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(
        d_key_material,
        key_material.data(),
        static_cast<size_t>(entry_count) * sizeof(jlong),
        cudaMemcpyHostToDevice);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(
        d_probabilities,
        probabilities.data(),
        static_cast<size_t>(entry_count) * sizeof(jfloat),
        cudaMemcpyHostToDevice);
  }
  if (err != cudaSuccess) {
    free_all();
    return report_cuda_error(132, err, "cudaMemcpy setup");
  }

  int threads_per_block = resolve_range_cuda_threads_per_block(env);
  threads_per_block = std::min(threads_per_block, prop.maxThreadsPerBlock);
  if (threads_per_block <= 0) {
    threads_per_block = kDefaultRangeCudaThreadsPerBlock;
  }
  int max_row_entries = 1;
  for (int h = 0; h < hero_count; ++h) {
    const int row_entries = offsets[static_cast<size_t>(h + 1)] - offsets[static_cast<size_t>(h)];
    if (row_entries > max_row_entries) {
      max_row_entries = row_entries;
    }
  }
  int tuned_threads = 32;
  while (tuned_threads < max_row_entries && tuned_threads < threads_per_block) {
    tuned_threads <<= 1;
  }
  tuned_threads = std::max(32, std::min(tuned_threads, threads_per_block));
  threads_per_block = std::min(tuned_threads, prop.maxThreadsPerBlock);

  const size_t reduction_bytes = static_cast<size_t>(threads_per_block) * 5ULL * sizeof(float);
  for (int hero_offset = 0; hero_offset < hero_count; hero_offset += max_chunk_heroes) {
    const int hero_chunk = std::min(max_chunk_heroes, hero_count - hero_offset);
    const int zero = 0;
    err = cudaMemcpy(d_status, &zero, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      free_all();
      return report_cuda_error(132, err, "cudaMemcpy host->device");
    }

    const int blocks = std::max(1, hero_chunk);
    if (memory_path == RangeMemoryPath::ReadOnly) {
      range_monte_carlo_csr_by_hero_kernel<true><<<blocks, threads_per_block, reduction_bytes>>>(
          d_hero_ids,
          d_offsets,
          d_villain_ids,
          d_key_material,
          d_probabilities,
          entry_count,
          hero_chunk,
          hero_offset,
          trials,
          monte_carlo_seed_base,
          d_out_wins,
          d_out_ties,
          d_out_losses,
          d_out_stderrs,
          d_status);
    } else {
      range_monte_carlo_csr_by_hero_kernel<false><<<blocks, threads_per_block, reduction_bytes>>>(
          d_hero_ids,
          d_offsets,
          d_villain_ids,
          d_key_material,
          d_probabilities,
          entry_count,
          hero_chunk,
          hero_offset,
          trials,
          monte_carlo_seed_base,
          d_out_wins,
          d_out_ties,
          d_out_losses,
          d_out_stderrs,
          d_status);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      free_all();
      return report_cuda_error(133, err, "range kernel launch");
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      free_all();
      return report_cuda_error(134, err, "cudaDeviceSynchronize");
    }

    int status = 0;
    err = cudaMemcpy(&status, d_status, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      free_all();
      return report_cuda_error(135, err, "cudaMemcpy status");
    }
    if (status != 0) {
      free_all();
      return status;
    }
  }

  err = cudaMemcpy(
      out_wins.data(),
      d_out_wins,
      static_cast<size_t>(hero_count) * sizeof(jfloat),
      cudaMemcpyDeviceToHost);
  if (err == cudaSuccess) {
    err = cudaMemcpy(
        out_ties.data(),
        d_out_ties,
        static_cast<size_t>(hero_count) * sizeof(jfloat),
        cudaMemcpyDeviceToHost);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(
        out_losses.data(),
        d_out_losses,
        static_cast<size_t>(hero_count) * sizeof(jfloat),
        cudaMemcpyDeviceToHost);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(
        out_stderrs.data(),
        d_out_stderrs,
        static_cast<size_t>(hero_count) * sizeof(jfloat),
        cudaMemcpyDeviceToHost);
  }
  if (err != cudaSuccess) {
    free_all();
    return report_cuda_error(135, err, "cudaMemcpy device->host");
  }

  free_all();
  return 0;
}

}  // namespace

/* ========================================================================== *
 * JNI entry points                                                           *
 * These functions are called from Scala/JVM via JNI.  Each validates inputs, *
 * copies data from JNI arrays into C++ vectors, delegates to CPU or CUDA     *
 * compute functions, then copies results back into JNI output arrays.        *
 * ========================================================================== */

/* computeBatch: legacy API with separate hero/villain ID arrays.
 * Engine selection: Auto tries CUDA first, falls back to CPU on failure.
 * Output: jdouble arrays for win/tie/loss/stderr probabilities. */
extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HeadsUpGpuNativeBindings_computeBatch(
    JNIEnv* env,
    jclass,
    jintArray low_ids,
    jintArray high_ids,
    jint mode_code,
    jint trials,
    jlongArray seeds,
    jdoubleArray wins,
    jdoubleArray ties,
    jdoubleArray losses,
    jdoubleArray stderrs) {
  g_last_engine_code.store(kEngineUnknown, std::memory_order_relaxed);

  if (low_ids == nullptr || high_ids == nullptr || seeds == nullptr ||
      wins == nullptr || ties == nullptr || losses == nullptr || stderrs == nullptr) {
    return 100;
  }

  const jsize n = env->GetArrayLength(low_ids);
  if (env->GetArrayLength(high_ids) != n || env->GetArrayLength(seeds) != n ||
      env->GetArrayLength(wins) != n || env->GetArrayLength(ties) != n ||
      env->GetArrayLength(losses) != n || env->GetArrayLength(stderrs) != n) {
    return 101;
  }
  if (mode_code != 0 && mode_code != 1) {
    return 111;
  }
  if (mode_code == 1 && trials <= 0) {
    return 126;
  }

  std::vector<jint> low_buf(static_cast<size_t>(n));
  std::vector<jint> high_buf(static_cast<size_t>(n));
  std::vector<jlong> seed_buf(static_cast<size_t>(n));
  std::vector<jdouble> win_buf(static_cast<size_t>(n));
  std::vector<jdouble> tie_buf(static_cast<size_t>(n));
  std::vector<jdouble> loss_buf(static_cast<size_t>(n));
  std::vector<jdouble> stderr_buf(static_cast<size_t>(n));

  env->GetIntArrayRegion(low_ids, 0, n, low_buf.data());
  env->GetIntArrayRegion(high_ids, 0, n, high_buf.data());
  env->GetLongArrayRegion(seeds, 0, n, seed_buf.data());
  if (check_and_clear_exception(env)) {
    return 102;
  }

  for (jsize i = 0; i < n; ++i) {
    const jint low_id = low_buf[static_cast<size_t>(i)];
    const jint high_id = high_buf[static_cast<size_t>(i)];
    if (low_id < 0 || low_id >= kHoleCardsCount || high_id < 0 || high_id >= kHoleCardsCount) {
      return 125;
    }
  }

  jint status = 0;
  const NativeEngine engine = resolve_engine(env);
  const bool try_cuda = engine != NativeEngine::Cpu;

  if (try_cuda) {
    status = compute_batch_cuda(
        env,
        low_buf,
        high_buf,
        seed_buf,
        mode_code,
        trials,
        win_buf,
        tie_buf,
        loss_buf,
        stderr_buf);
    if (status == 0) {
      g_last_engine_code.store(kEngineCuda, std::memory_order_relaxed);
    }
    if (status != 0 && engine == NativeEngine::Cuda) {
      return status;
    }
  }

  if (!try_cuda || status != 0) {
    const bool is_cuda_fallback = try_cuda && status != 0;
    status = compute_batch_cpu(
        low_buf,
        high_buf,
        seed_buf,
        mode_code,
        trials,
        win_buf,
        tie_buf,
        loss_buf,
        stderr_buf);
    if (status != 0) {
      return status;
    }
    g_last_engine_code.store(
        is_cuda_fallback ? kEngineCpuFallbackAfterCudaFailure : kEngineCpu,
        std::memory_order_relaxed);
  }

  env->SetDoubleArrayRegion(wins, 0, n, win_buf.data());
  env->SetDoubleArrayRegion(ties, 0, n, tie_buf.data());
  env->SetDoubleArrayRegion(losses, 0, n, loss_buf.data());
  env->SetDoubleArrayRegion(stderrs, 0, n, stderr_buf.data());
  if (check_and_clear_exception(env)) {
    g_last_engine_code.store(kEngineUnknown, std::memory_order_relaxed);
    return 124;
  }
  return 0;
}

/* computeBatchCpuOnly: forced CPU-only path (no CUDA attempt).
 * Used for benchmarking CPU vs CUDA performance. */
extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HeadsUpGpuNativeBindings_computeBatchCpuOnly(
    JNIEnv* env,
    jclass,
    jintArray low_ids,
    jintArray high_ids,
    jint mode_code,
    jint trials,
    jlongArray seeds,
    jdoubleArray wins,
    jdoubleArray ties,
    jdoubleArray losses,
    jdoubleArray stderrs) {
  g_last_engine_code.store(kEngineUnknown, std::memory_order_relaxed);

  if (low_ids == nullptr || high_ids == nullptr || seeds == nullptr ||
      wins == nullptr || ties == nullptr || losses == nullptr || stderrs == nullptr) {
    return 100;
  }

  const jsize n = env->GetArrayLength(low_ids);
  if (env->GetArrayLength(high_ids) != n || env->GetArrayLength(seeds) != n ||
      env->GetArrayLength(wins) != n || env->GetArrayLength(ties) != n ||
      env->GetArrayLength(losses) != n || env->GetArrayLength(stderrs) != n) {
    return 101;
  }
  if (mode_code != 0 && mode_code != 1) {
    return 111;
  }
  if (mode_code == 1 && trials <= 0) {
    return 126;
  }

  std::vector<jint> low_buf(static_cast<size_t>(n));
  std::vector<jint> high_buf(static_cast<size_t>(n));
  std::vector<jlong> seed_buf(static_cast<size_t>(n));
  std::vector<jdouble> win_buf(static_cast<size_t>(n));
  std::vector<jdouble> tie_buf(static_cast<size_t>(n));
  std::vector<jdouble> loss_buf(static_cast<size_t>(n));
  std::vector<jdouble> stderr_buf(static_cast<size_t>(n));

  env->GetIntArrayRegion(low_ids, 0, n, low_buf.data());
  env->GetIntArrayRegion(high_ids, 0, n, high_buf.data());
  env->GetLongArrayRegion(seeds, 0, n, seed_buf.data());
  if (check_and_clear_exception(env)) {
    return 102;
  }

  for (jsize i = 0; i < n; ++i) {
    const jint low_id = low_buf[static_cast<size_t>(i)];
    const jint high_id = high_buf[static_cast<size_t>(i)];
    if (low_id < 0 || low_id >= kHoleCardsCount || high_id < 0 || high_id >= kHoleCardsCount) {
      return 125;
    }
  }

  const jint status = compute_batch_cpu(
      low_buf,
      high_buf,
      seed_buf,
      mode_code,
      trials,
      win_buf,
      tie_buf,
      loss_buf,
      stderr_buf);
  if (status != 0) {
    return status;
  }

  env->SetDoubleArrayRegion(wins, 0, n, win_buf.data());
  env->SetDoubleArrayRegion(ties, 0, n, tie_buf.data());
  env->SetDoubleArrayRegion(losses, 0, n, loss_buf.data());
  env->SetDoubleArrayRegion(stderrs, 0, n, stderr_buf.data());
  if (check_and_clear_exception(env)) {
    g_last_engine_code.store(kEngineUnknown, std::memory_order_relaxed);
    return 124;
  }

  g_last_engine_code.store(kEngineCpu, std::memory_order_relaxed);
  return 0;
}

/* lastEngineCode: returns the engine code (0=unknown, 1=CPU, 2=CUDA, 3=fallback)
 * from the most recent batch computation.  Thread-safe via atomic load. */
extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HeadsUpGpuNativeBindings_lastEngineCode(
    JNIEnv*,
    jclass) {
  return g_last_engine_code.load(std::memory_order_relaxed);
}

/* cudaDeviceCount: returns the number of CUDA-capable GPUs (0 if CUDA unavailable). */
extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HeadsUpGpuNativeBindings_cudaDeviceCount(
    JNIEnv*,
    jclass) {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
    return 0;
  }
  return static_cast<jint>(count);
}

/* cudaDeviceInfo: returns a pipe-delimited string with GPU metadata:
 *   "name|SMs|clockMHz|memoryMB|major.minor"
 * Used by the JVM layer for logging and auto-tuning decisions. */
extern "C" JNIEXPORT jstring JNICALL
Java_sicfun_holdem_HeadsUpGpuNativeBindings_cudaDeviceInfo(
    JNIEnv* env,
    jclass,
    jint device_index) {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess || device_index < 0 || device_index >= count) {
    return env->NewStringUTF("");
  }
  cudaDeviceProp prop{};
  err = cudaGetDeviceProperties(&prop, static_cast<int>(device_index));
  if (err != cudaSuccess) {
    return env->NewStringUTF("");
  }
  const int memory_mb = static_cast<int>(prop.totalGlobalMem / (1024ULL * 1024ULL));
  char buf[512];
  std::snprintf(buf, sizeof(buf), "%s|%d|%d|%d|%d.%d",
      prop.name,
      prop.multiProcessorCount,
      prop.clockRate / 1000,
      memory_mb,
      prop.major,
      prop.minor);
  return env->NewStringUTF(buf);
}

/* computeBatchOnDevice: like computeBatch but targets a specific CUDA device index.
 * No CPU fallback -- returns error status if CUDA fails. */
extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HeadsUpGpuNativeBindings_computeBatchOnDevice(
    JNIEnv* env,
    jclass,
    jint device_index,
    jintArray low_ids,
    jintArray high_ids,
    jint mode_code,
    jint trials,
    jlongArray seeds,
    jdoubleArray wins,
    jdoubleArray ties,
    jdoubleArray losses,
    jdoubleArray stderrs) {
  g_last_engine_code.store(kEngineUnknown, std::memory_order_relaxed);

  if (low_ids == nullptr || high_ids == nullptr || seeds == nullptr ||
      wins == nullptr || ties == nullptr || losses == nullptr || stderrs == nullptr) {
    return 100;
  }

  const jsize n = env->GetArrayLength(low_ids);
  if (env->GetArrayLength(high_ids) != n || env->GetArrayLength(seeds) != n ||
      env->GetArrayLength(wins) != n || env->GetArrayLength(ties) != n ||
      env->GetArrayLength(losses) != n || env->GetArrayLength(stderrs) != n) {
    return 101;
  }
  if (mode_code != 0 && mode_code != 1) {
    return 111;
  }
  if (mode_code == 1 && trials <= 0) {
    return 126;
  }

  std::vector<jint> low_buf(static_cast<size_t>(n));
  std::vector<jint> high_buf(static_cast<size_t>(n));
  std::vector<jlong> seed_buf(static_cast<size_t>(n));
  std::vector<jdouble> win_buf(static_cast<size_t>(n));
  std::vector<jdouble> tie_buf(static_cast<size_t>(n));
  std::vector<jdouble> loss_buf(static_cast<size_t>(n));
  std::vector<jdouble> stderr_buf(static_cast<size_t>(n));

  env->GetIntArrayRegion(low_ids, 0, n, low_buf.data());
  env->GetIntArrayRegion(high_ids, 0, n, high_buf.data());
  env->GetLongArrayRegion(seeds, 0, n, seed_buf.data());
  if (check_and_clear_exception(env)) {
    return 102;
  }

  for (jsize i = 0; i < n; ++i) {
    const jint low_id = low_buf[static_cast<size_t>(i)];
    const jint high_id = high_buf[static_cast<size_t>(i)];
    if (low_id < 0 || low_id >= kHoleCardsCount || high_id < 0 || high_id >= kHoleCardsCount) {
      return 125;
    }
  }

  const jint status = compute_batch_cuda(
      env, low_buf, high_buf, seed_buf, mode_code, trials,
      win_buf, tie_buf, loss_buf, stderr_buf,
      static_cast<int>(device_index));
  if (status != 0) {
    return status;
  }
  g_last_engine_code.store(kEngineCuda, std::memory_order_relaxed);

  env->SetDoubleArrayRegion(wins, 0, n, win_buf.data());
  env->SetDoubleArrayRegion(ties, 0, n, tie_buf.data());
  env->SetDoubleArrayRegion(losses, 0, n, loss_buf.data());
  env->SetDoubleArrayRegion(stderrs, 0, n, stderr_buf.data());
  if (check_and_clear_exception(env)) {
    g_last_engine_code.store(kEngineUnknown, std::memory_order_relaxed);
    return 124;
  }
  return 0;
}

/* computeBatchPacked: compact API using packed 22-bit matchup keys.
 * Each packed_key encodes both hero and villain hole-card IDs.
 * Output: jfloat arrays (single precision, lower memory than jdouble).
 * Engine selection: Auto tries CUDA, falls back to CPU on failure. */
extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HeadsUpGpuNativeBindings_computeBatchPacked(
    JNIEnv* env,
    jclass,
    jintArray packed_keys,
    jint mode_code,
    jint trials,
    jlong monte_carlo_seed_base,
    jlongArray key_material,
    jfloatArray wins,
    jfloatArray ties,
    jfloatArray losses,
    jfloatArray stderrs) {
  g_last_engine_code.store(kEngineUnknown, std::memory_order_relaxed);

  if (packed_keys == nullptr || key_material == nullptr ||
      wins == nullptr || ties == nullptr || losses == nullptr || stderrs == nullptr) {
    return 100;
  }

  const jsize n = env->GetArrayLength(packed_keys);
  if (env->GetArrayLength(key_material) != n ||
      env->GetArrayLength(wins) != n || env->GetArrayLength(ties) != n ||
      env->GetArrayLength(losses) != n || env->GetArrayLength(stderrs) != n) {
    return 101;
  }
  if (mode_code != 0 && mode_code != 1) {
    return 111;
  }
  if (mode_code == 1 && trials <= 0) {
    return 126;
  }

  std::vector<jint> packed_buf(static_cast<size_t>(n));
  std::vector<jlong> key_material_buf(static_cast<size_t>(n));
  std::vector<jfloat> win_buf(static_cast<size_t>(n));
  std::vector<jfloat> tie_buf(static_cast<size_t>(n));
  std::vector<jfloat> loss_buf(static_cast<size_t>(n));
  std::vector<jfloat> stderr_buf(static_cast<size_t>(n));

  env->GetIntArrayRegion(packed_keys, 0, n, packed_buf.data());
  env->GetLongArrayRegion(key_material, 0, n, key_material_buf.data());
  if (check_and_clear_exception(env)) {
    return 102;
  }

  for (jsize i = 0; i < n; ++i) {
    const jint packed = packed_buf[static_cast<size_t>(i)];
    const int low_id = unpack_low_id(packed);
    const int high_id = unpack_high_id(packed);
    if (low_id < 0 || low_id >= kHoleCardsCount || high_id < 0 || high_id >= kHoleCardsCount) {
      return 125;
    }
  }

  jint status = 0;
  const NativeEngine engine = resolve_engine(env);
  const bool try_cuda = engine != NativeEngine::Cpu;

  if (try_cuda) {
    status = compute_batch_cuda_packed(
        env,
        packed_buf,
        key_material_buf,
        mode_code,
        trials,
        monte_carlo_seed_base,
        win_buf,
        tie_buf,
        loss_buf,
        stderr_buf);
    if (status == 0) {
      g_last_engine_code.store(kEngineCuda, std::memory_order_relaxed);
    }
    if (status != 0 && engine == NativeEngine::Cuda) {
      return status;
    }
  }

  if (!try_cuda || status != 0) {
    const bool is_cuda_fallback = try_cuda && status != 0;
    std::vector<jint> low_buf(static_cast<size_t>(n));
    std::vector<jint> high_buf(static_cast<size_t>(n));
    std::vector<jlong> seed_buf(static_cast<size_t>(n));
    std::vector<jdouble> win_double(static_cast<size_t>(n));
    std::vector<jdouble> tie_double(static_cast<size_t>(n));
    std::vector<jdouble> loss_double(static_cast<size_t>(n));
    std::vector<jdouble> stderr_double(static_cast<size_t>(n));

    for (jsize i = 0; i < n; ++i) {
      const size_t idx = static_cast<size_t>(i);
      const jint packed = packed_buf[idx];
      low_buf[idx] = static_cast<jint>(unpack_low_id(packed));
      high_buf[idx] = static_cast<jint>(unpack_high_id(packed));
      if (mode_code == 1) {
        const uint64_t local_seed = mix64(
            static_cast<uint64_t>(monte_carlo_seed_base) ^
            static_cast<uint64_t>(key_material_buf[idx]));
        seed_buf[idx] = static_cast<jlong>(local_seed);
      } else {
        seed_buf[idx] = 0;
      }
    }

    status = compute_batch_cpu(
        low_buf,
        high_buf,
        seed_buf,
        mode_code,
        trials,
        win_double,
        tie_double,
        loss_double,
        stderr_double);
    if (status != 0) {
      return status;
    }
    for (jsize i = 0; i < n; ++i) {
      const size_t idx = static_cast<size_t>(i);
      win_buf[idx] = static_cast<jfloat>(win_double[idx]);
      tie_buf[idx] = static_cast<jfloat>(tie_double[idx]);
      loss_buf[idx] = static_cast<jfloat>(loss_double[idx]);
      stderr_buf[idx] = static_cast<jfloat>(stderr_double[idx]);
    }

    g_last_engine_code.store(
        is_cuda_fallback ? kEngineCpuFallbackAfterCudaFailure : kEngineCpu,
        std::memory_order_relaxed);
  }

  env->SetFloatArrayRegion(wins, 0, n, win_buf.data());
  env->SetFloatArrayRegion(ties, 0, n, tie_buf.data());
  env->SetFloatArrayRegion(losses, 0, n, loss_buf.data());
  env->SetFloatArrayRegion(stderrs, 0, n, stderr_buf.data());
  if (check_and_clear_exception(env)) {
    g_last_engine_code.store(kEngineUnknown, std::memory_order_relaxed);
    return 124;
  }
  return 0;
}

/* computeBatchPackedOnDevice: packed API targeting a specific CUDA device.
 * No CPU fallback. */
extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HeadsUpGpuNativeBindings_computeBatchPackedOnDevice(
    JNIEnv* env,
    jclass,
    jint device_index,
    jintArray packed_keys,
    jint mode_code,
    jint trials,
    jlong monte_carlo_seed_base,
    jlongArray key_material,
    jfloatArray wins,
    jfloatArray ties,
    jfloatArray losses,
    jfloatArray stderrs) {
  g_last_engine_code.store(kEngineUnknown, std::memory_order_relaxed);

  if (packed_keys == nullptr || key_material == nullptr ||
      wins == nullptr || ties == nullptr || losses == nullptr || stderrs == nullptr) {
    return 100;
  }

  const jsize n = env->GetArrayLength(packed_keys);
  if (env->GetArrayLength(key_material) != n ||
      env->GetArrayLength(wins) != n || env->GetArrayLength(ties) != n ||
      env->GetArrayLength(losses) != n || env->GetArrayLength(stderrs) != n) {
    return 101;
  }
  if (mode_code != 0 && mode_code != 1) {
    return 111;
  }
  if (mode_code == 1 && trials <= 0) {
    return 126;
  }

  std::vector<jint> packed_buf(static_cast<size_t>(n));
  std::vector<jlong> key_material_buf(static_cast<size_t>(n));
  std::vector<jfloat> win_buf(static_cast<size_t>(n));
  std::vector<jfloat> tie_buf(static_cast<size_t>(n));
  std::vector<jfloat> loss_buf(static_cast<size_t>(n));
  std::vector<jfloat> stderr_buf(static_cast<size_t>(n));

  env->GetIntArrayRegion(packed_keys, 0, n, packed_buf.data());
  env->GetLongArrayRegion(key_material, 0, n, key_material_buf.data());
  if (check_and_clear_exception(env)) {
    return 102;
  }

  for (jsize i = 0; i < n; ++i) {
    const jint packed = packed_buf[static_cast<size_t>(i)];
    const int low_id = unpack_low_id(packed);
    const int high_id = unpack_high_id(packed);
    if (low_id < 0 || low_id >= kHoleCardsCount || high_id < 0 || high_id >= kHoleCardsCount) {
      return 125;
    }
  }

  const jint status = compute_batch_cuda_packed(
      env,
      packed_buf,
      key_material_buf,
      mode_code,
      trials,
      monte_carlo_seed_base,
      win_buf,
      tie_buf,
      loss_buf,
      stderr_buf,
      static_cast<int>(device_index));
  if (status != 0) {
    return status;
  }
  g_last_engine_code.store(kEngineCuda, std::memory_order_relaxed);

  env->SetFloatArrayRegion(wins, 0, n, win_buf.data());
  env->SetFloatArrayRegion(ties, 0, n, tie_buf.data());
  env->SetFloatArrayRegion(losses, 0, n, loss_buf.data());
  env->SetFloatArrayRegion(stderrs, 0, n, stderr_buf.data());
  if (check_and_clear_exception(env)) {
    g_last_engine_code.store(kEngineUnknown, std::memory_order_relaxed);
    return 124;
  }
  return 0;
}

/* computeRangeBatchMonteCarloCsr: range-vs-range equity via CSR layout.
 * hero_ids[h] is the hero hole-card ID for hero h.
 * offsets[h..h+1] defines the villain range slice in villain_ids/probabilities.
 * For each hero, computes probability-weighted average equity against the
 * villain range.  Engine: Auto tries CUDA, falls back to CPU. */
extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HeadsUpGpuNativeBindings_computeRangeBatchMonteCarloCsr(
    JNIEnv* env,
    jclass,
    jintArray hero_ids,
    jintArray offsets,
    jintArray villain_ids,
    jlongArray key_material,
    jfloatArray probabilities,
    jint trials,
    jlong monte_carlo_seed_base,
    jfloatArray wins,
    jfloatArray ties,
    jfloatArray losses,
    jfloatArray stderrs) {
  g_last_engine_code.store(kEngineUnknown, std::memory_order_relaxed);

  if (hero_ids == nullptr || offsets == nullptr || villain_ids == nullptr ||
      key_material == nullptr || probabilities == nullptr ||
      wins == nullptr || ties == nullptr || losses == nullptr || stderrs == nullptr) {
    return 100;
  }
  if (trials <= 0) {
    return 126;
  }

  const jsize hero_count = env->GetArrayLength(hero_ids);
  const jsize offset_count = env->GetArrayLength(offsets);
  const jsize entry_count = env->GetArrayLength(villain_ids);
  if (offset_count != hero_count + 1 ||
      env->GetArrayLength(key_material) != entry_count ||
      env->GetArrayLength(probabilities) != entry_count ||
      env->GetArrayLength(wins) != hero_count ||
      env->GetArrayLength(ties) != hero_count ||
      env->GetArrayLength(losses) != hero_count ||
      env->GetArrayLength(stderrs) != hero_count) {
    return 101;
  }

  std::vector<jint> hero_buf(static_cast<size_t>(hero_count));
  std::vector<jint> offset_buf(static_cast<size_t>(offset_count));
  std::vector<jint> villain_buf(static_cast<size_t>(entry_count));
  std::vector<jlong> key_material_buf(static_cast<size_t>(entry_count));
  std::vector<jfloat> probability_buf(static_cast<size_t>(entry_count));
  std::vector<jfloat> out_win(static_cast<size_t>(hero_count));
  std::vector<jfloat> out_tie(static_cast<size_t>(hero_count));
  std::vector<jfloat> out_loss(static_cast<size_t>(hero_count));
  std::vector<jfloat> out_stderr(static_cast<size_t>(hero_count));

  env->GetIntArrayRegion(hero_ids, 0, hero_count, hero_buf.data());
  env->GetIntArrayRegion(offsets, 0, offset_count, offset_buf.data());
  env->GetIntArrayRegion(villain_ids, 0, entry_count, villain_buf.data());
  env->GetLongArrayRegion(key_material, 0, entry_count, key_material_buf.data());
  env->GetFloatArrayRegion(probabilities, 0, entry_count, probability_buf.data());
  if (check_and_clear_exception(env)) {
    return 102;
  }

  jint status = 0;
  const NativeEngine engine = resolve_engine(env);
  const bool try_cuda = engine != NativeEngine::Cpu;

  if (try_cuda) {
    status = compute_range_batch_cuda_monte_carlo_csr(
        env,
        hero_buf,
        offset_buf,
        villain_buf,
        key_material_buf,
        probability_buf,
        trials,
        monte_carlo_seed_base,
        out_win,
        out_tie,
        out_loss,
        out_stderr);
    if (status == 0) {
      g_last_engine_code.store(kEngineCuda, std::memory_order_relaxed);
    }
    if (status != 0 && engine == NativeEngine::Cuda) {
      return status;
    }
  }

  if (!try_cuda || status != 0) {
    const bool is_cuda_fallback = try_cuda && status != 0;
    status = compute_range_batch_cpu_monte_carlo_csr(
        hero_buf,
        offset_buf,
        villain_buf,
        key_material_buf,
        probability_buf,
        trials,
        monte_carlo_seed_base,
        out_win,
        out_tie,
        out_loss,
        out_stderr);
    if (status != 0) {
      return status;
    }
    g_last_engine_code.store(
        is_cuda_fallback ? kEngineCpuFallbackAfterCudaFailure : kEngineCpu,
        std::memory_order_relaxed);
  }

  env->SetFloatArrayRegion(wins, 0, hero_count, out_win.data());
  env->SetFloatArrayRegion(ties, 0, hero_count, out_tie.data());
  env->SetFloatArrayRegion(losses, 0, hero_count, out_loss.data());
  env->SetFloatArrayRegion(stderrs, 0, hero_count, out_stderr.data());
  if (check_and_clear_exception(env)) {
    g_last_engine_code.store(kEngineUnknown, std::memory_order_relaxed);
    return 124;
  }
  return 0;
}

/* computeRangeBatchMonteCarloCsrOnDevice: range CSR targeting a specific
 * CUDA device.  No CPU fallback. */
extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HeadsUpGpuNativeBindings_computeRangeBatchMonteCarloCsrOnDevice(
    JNIEnv* env,
    jclass,
    jint device_index,
    jintArray hero_ids,
    jintArray offsets,
    jintArray villain_ids,
    jlongArray key_material,
    jfloatArray probabilities,
    jint trials,
    jlong monte_carlo_seed_base,
    jfloatArray wins,
    jfloatArray ties,
    jfloatArray losses,
    jfloatArray stderrs) {
  g_last_engine_code.store(kEngineUnknown, std::memory_order_relaxed);

  if (hero_ids == nullptr || offsets == nullptr || villain_ids == nullptr ||
      key_material == nullptr || probabilities == nullptr ||
      wins == nullptr || ties == nullptr || losses == nullptr || stderrs == nullptr) {
    return 100;
  }
  if (trials <= 0) {
    return 126;
  }

  const jsize hero_count = env->GetArrayLength(hero_ids);
  const jsize offset_count = env->GetArrayLength(offsets);
  const jsize entry_count = env->GetArrayLength(villain_ids);
  if (offset_count != hero_count + 1 ||
      env->GetArrayLength(key_material) != entry_count ||
      env->GetArrayLength(probabilities) != entry_count ||
      env->GetArrayLength(wins) != hero_count ||
      env->GetArrayLength(ties) != hero_count ||
      env->GetArrayLength(losses) != hero_count ||
      env->GetArrayLength(stderrs) != hero_count) {
    return 101;
  }

  std::vector<jint> hero_buf(static_cast<size_t>(hero_count));
  std::vector<jint> offset_buf(static_cast<size_t>(offset_count));
  std::vector<jint> villain_buf(static_cast<size_t>(entry_count));
  std::vector<jlong> key_material_buf(static_cast<size_t>(entry_count));
  std::vector<jfloat> probability_buf(static_cast<size_t>(entry_count));
  std::vector<jfloat> out_win(static_cast<size_t>(hero_count));
  std::vector<jfloat> out_tie(static_cast<size_t>(hero_count));
  std::vector<jfloat> out_loss(static_cast<size_t>(hero_count));
  std::vector<jfloat> out_stderr(static_cast<size_t>(hero_count));

  env->GetIntArrayRegion(hero_ids, 0, hero_count, hero_buf.data());
  env->GetIntArrayRegion(offsets, 0, offset_count, offset_buf.data());
  env->GetIntArrayRegion(villain_ids, 0, entry_count, villain_buf.data());
  env->GetLongArrayRegion(key_material, 0, entry_count, key_material_buf.data());
  env->GetFloatArrayRegion(probabilities, 0, entry_count, probability_buf.data());
  if (check_and_clear_exception(env)) {
    return 102;
  }

  const jint status = compute_range_batch_cuda_monte_carlo_csr(
      env,
      hero_buf,
      offset_buf,
      villain_buf,
      key_material_buf,
      probability_buf,
      trials,
      monte_carlo_seed_base,
      out_win,
      out_tie,
      out_loss,
      out_stderr,
      static_cast<int>(device_index));
  if (status != 0) {
    return status;
  }
  g_last_engine_code.store(kEngineCuda, std::memory_order_relaxed);

  env->SetFloatArrayRegion(wins, 0, hero_count, out_win.data());
  env->SetFloatArrayRegion(ties, 0, hero_count, out_tie.data());
  env->SetFloatArrayRegion(losses, 0, hero_count, out_loss.data());
  env->SetFloatArrayRegion(stderrs, 0, hero_count, out_stderr.data());
  if (check_and_clear_exception(env)) {
    g_last_engine_code.store(kEngineUnknown, std::memory_order_relaxed);
    return 124;
  }
  return 0;
}
