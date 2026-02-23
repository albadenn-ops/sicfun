#include <jni.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
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
#else
#define HD
#endif

namespace {

constexpr int kDeckSize = 52;
constexpr int kRanksPerSuit = 13;
constexpr int kMinRankValue = 2;
constexpr int kMaxRankValue = 14;
constexpr int kHoleCardsCount = 1326;
constexpr int kRemainingAfterHoleCards = 48;
constexpr int kBoardCardCount = 5;
constexpr int kExactBoardCount = 1712304;  // C(48,5)
constexpr int kCpuWorkChunkSize = 8;
constexpr int kIdBits = 11;
constexpr int kIdMask = (1 << kIdBits) - 1;
constexpr int kDefaultCudaThreadsPerBlock = 128;
constexpr int kDefaultRangeCudaThreadsPerBlock = 128;
constexpr int kDefaultCudaThreadsPerBlockExact = 256;
constexpr int kDefaultCudaMaxChunkMatchups = 4096;
constexpr int kDefaultCudaMaxChunkMatchupsExact = 4;
constexpr int kDefaultRangeCudaMaxChunkHeroes = 4096;
constexpr int kStatusInvalidRangeLayout = 112;
constexpr uint64_t kRngMul = 2685821657736338717ULL;
constexpr jint kEngineUnknown = 0;
constexpr jint kEngineCpu = 1;
constexpr jint kEngineCuda = 2;
constexpr jint kEngineCpuFallbackAfterCudaFailure = 3;

struct HoleCards {
  uint8_t first;
  uint8_t second;
};

struct EquityResultNative {
  double win;
  double tie;
  double loss;
  double std_error;
};

enum class NativeEngine {
  Auto,
  Cpu,
  Cuda,
};

enum class RangeMemoryPath {
  ReadOnly,
  Global,
};

__device__ __constant__ uint8_t d_hole_first[kHoleCardsCount];
__device__ __constant__ uint8_t d_hole_second[kHoleCardsCount];
std::atomic<jint> g_last_engine_code(kEngineUnknown);

bool check_and_clear_exception(JNIEnv* env) {
  if (!env->ExceptionCheck()) {
    return false;
  }
  env->ExceptionClear();
  return true;
}

HD inline int card_rank(const int card_id) {
  return (card_id % kRanksPerSuit) + kMinRankValue;
}

HD inline int card_suit(const int card_id) {
  return card_id / kRanksPerSuit;
}

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
  for (int i = 0; i < kHoleCardsCount; ++i) {
    first[static_cast<size_t>(i)] = lookup[static_cast<size_t>(i)].first;
    second[static_cast<size_t>(i)] = lookup[static_cast<size_t>(i)].second;
  }
  cudaError_t err = cudaMemcpyToSymbol(d_hole_first, first.data(), first.size() * sizeof(uint8_t));
  if (err != cudaSuccess) {
    return false;
  }
  err = cudaMemcpyToSymbol(d_hole_second, second.data(), second.size() * sizeof(uint8_t));
  if (err != cudaSuccess) {
    return false;
  }
  initialized_devices.insert(device);
  return true;
}

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

HD inline uint32_t encode_score(const int category, const int* tiebreak, const int tiebreak_size) {
  uint32_t score = static_cast<uint32_t>(category) << 24;
  for (int i = 0; i < tiebreak_size && i < 5; ++i) {
    score |= (static_cast<uint32_t>(tiebreak[i] & 0x0F) << (20 - (i * 4)));
  }
  return score;
}

HD inline int straight_high_from_rank_mask(const uint16_t rank_mask) {
  for (int high = kMaxRankValue; high >= 6; --high) {
    const uint16_t window = static_cast<uint16_t>(0x1F) << (high - 6);
    if ((rank_mask & window) == window) {
      return high;
    }
  }
  const uint16_t wheel_mask =
      (static_cast<uint16_t>(1) << 12) | (static_cast<uint16_t>(1) << 3) |
      (static_cast<uint16_t>(1) << 2) | (static_cast<uint16_t>(1) << 1) |
      (static_cast<uint16_t>(1) << 0);
  if ((rank_mask & wheel_mask) == wheel_mask) {
    return 5;
  }
  return 0;
}

HD uint32_t evaluate7_score(const int cards[7]) {
  int rank_counts[kMaxRankValue + 1];
  for (int rank = 0; rank <= kMaxRankValue; ++rank) {
    rank_counts[rank] = 0;
  }

  int suit_counts[4] = {0, 0, 0, 0};
  uint16_t rank_mask = 0;
  uint16_t suit_rank_mask[4] = {0, 0, 0, 0};

  for (int i = 0; i < 7; ++i) {
    const int rank = card_rank(cards[i]);
    const int suit = card_suit(cards[i]);
    ++rank_counts[rank];
    ++suit_counts[suit];
    const uint16_t bit = static_cast<uint16_t>(1) << (rank - kMinRankValue);
    rank_mask |= bit;
    suit_rank_mask[suit] |= bit;
  }

  int flush_suit = -1;
  for (int suit = 0; suit < 4; ++suit) {
    if (suit_counts[suit] >= 5) {
      flush_suit = suit;
      break;
    }
  }

  int tiebreak[5] = {0, 0, 0, 0, 0};

  if (flush_suit >= 0) {
    const int straight_flush_high = straight_high_from_rank_mask(suit_rank_mask[flush_suit]);
    if (straight_flush_high > 0) {
      tiebreak[0] = straight_flush_high;
      return encode_score(8, tiebreak, 1);  // StraightFlush
    }
  }

  int quad_rank = 0;
  int trip1 = 0;
  int trip2 = 0;
  int pair1 = 0;
  int pair2 = 0;

  for (int rank = kMaxRankValue; rank >= kMinRankValue; --rank) {
    const int count = rank_counts[rank];
    if (count == 4) {
      quad_rank = rank;
    } else if (count == 3) {
      if (trip1 == 0) {
        trip1 = rank;
      } else {
        trip2 = rank;
      }
    } else if (count == 2) {
      if (pair1 == 0) {
        pair1 = rank;
      } else {
        pair2 = rank;
      }
    }
  }

  if (quad_rank > 0) {
    int kicker = 0;
    for (int rank = kMaxRankValue; rank >= kMinRankValue; --rank) {
      if (rank != quad_rank && rank_counts[rank] > 0) {
        kicker = rank;
        break;
      }
    }
    tiebreak[0] = quad_rank;
    tiebreak[1] = kicker;
    return encode_score(7, tiebreak, 2);  // FourOfKind
  }

  if (trip1 > 0) {
    const int full_house_pair = (trip2 > 0) ? trip2 : pair1;
    if (full_house_pair > 0) {
      tiebreak[0] = trip1;
      tiebreak[1] = full_house_pair;
      return encode_score(6, tiebreak, 2);  // FullHouse
    }
  }

  if (flush_suit >= 0) {
    int idx = 0;
    for (int rank = kMaxRankValue; rank >= kMinRankValue && idx < 5; --rank) {
      const uint16_t bit = static_cast<uint16_t>(1) << (rank - kMinRankValue);
      if ((suit_rank_mask[flush_suit] & bit) != 0) {
        tiebreak[idx++] = rank;
      }
    }
    return encode_score(5, tiebreak, 5);  // Flush
  }

  const int straight_high = straight_high_from_rank_mask(rank_mask);
  if (straight_high > 0) {
    tiebreak[0] = straight_high;
    return encode_score(4, tiebreak, 1);  // Straight
  }

  if (trip1 > 0) {
    int kick_idx = 1;
    for (int rank = kMaxRankValue; rank >= kMinRankValue && kick_idx < 3; --rank) {
      if (rank != trip1 && rank_counts[rank] > 0) {
        tiebreak[kick_idx++] = rank;
      }
    }
    tiebreak[0] = trip1;
    return encode_score(3, tiebreak, 3);  // ThreeOfKind
  }

  if (pair1 > 0 && pair2 > 0) {
    int kicker = 0;
    for (int rank = kMaxRankValue; rank >= kMinRankValue; --rank) {
      if (rank != pair1 && rank != pair2 && rank_counts[rank] > 0) {
        kicker = rank;
        break;
      }
    }
    tiebreak[0] = pair1;
    tiebreak[1] = pair2;
    tiebreak[2] = kicker;
    return encode_score(2, tiebreak, 3);  // TwoPair
  }

  if (pair1 > 0) {
    int kick_idx = 1;
    for (int rank = kMaxRankValue; rank >= kMinRankValue && kick_idx < 4; --rank) {
      if (rank != pair1 && rank_counts[rank] > 0) {
        tiebreak[kick_idx++] = rank;
      }
    }
    tiebreak[0] = pair1;
    return encode_score(1, tiebreak, 4);  // OnePair
  }

  int idx = 0;
  for (int rank = kMaxRankValue; rank >= kMinRankValue && idx < 5; --rank) {
    if (rank_counts[rank] > 0) {
      tiebreak[idx++] = rank;
    }
  }
  return encode_score(0, tiebreak, 5);  // HighCard
}

HD void fill_remaining_deck(
    const int hero_first,
    const int hero_second,
    const int villain_first,
    const int villain_second,
    uint8_t remaining[kRemainingAfterHoleCards]) {
  bool dead[kDeckSize];
  for (int i = 0; i < kDeckSize; ++i) {
    dead[i] = false;
  }
  dead[hero_first] = true;
  dead[hero_second] = true;
  dead[villain_first] = true;
  dead[villain_second] = true;

  int idx = 0;
  for (int card = 0; card < kDeckSize; ++card) {
    if (!dead[card]) {
      remaining[idx++] = static_cast<uint8_t>(card);
    }
  }
}

HD inline int compare_showdown(
    const int hero_first,
    const int hero_second,
    const int villain_first,
    const int villain_second,
    const uint8_t board[kBoardCardCount]) {
  int hero_cards[7] = {
      hero_first,
      hero_second,
      static_cast<int>(board[0]),
      static_cast<int>(board[1]),
      static_cast<int>(board[2]),
      static_cast<int>(board[3]),
      static_cast<int>(board[4])};
  int villain_cards[7] = {
      villain_first,
      villain_second,
      static_cast<int>(board[0]),
      static_cast<int>(board[1]),
      static_cast<int>(board[2]),
      static_cast<int>(board[3]),
      static_cast<int>(board[4])};
  const uint32_t hero_score = evaluate7_score(hero_cards);
  const uint32_t villain_score = evaluate7_score(villain_cards);
  if (hero_score > villain_score) {
    return 1;
  }
  if (hero_score < villain_score) {
    return -1;
  }
  return 0;
}

HD uint64_t mix64(uint64_t value) {
  uint64_t z = value + 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z = z ^ (z >> 31);
  return z;
}

HD inline int unpack_low_id(const jint packed_key) {
  const uint32_t packed = static_cast<uint32_t>(packed_key);
  return static_cast<int>((packed >> kIdBits) & static_cast<uint32_t>(kIdMask));
}

HD inline int unpack_high_id(const jint packed_key) {
  const uint32_t packed = static_cast<uint32_t>(packed_key);
  return static_cast<int>(packed & static_cast<uint32_t>(kIdMask));
}

HD inline uint64_t next_u64(uint64_t& state) {
  state ^= (state >> 12);
  state ^= (state << 25);
  state ^= (state >> 27);
  return state * kRngMul;
}

HD inline int bounded_rand(uint64_t& state, const int bound) {
  return static_cast<int>(next_u64(state) % static_cast<uint64_t>(bound));
}

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
    if (cmp > 0) {
      ++win_count;
    } else if (cmp == 0) {
      ++tie_count;
    } else {
      ++loss_count;
    }
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
            if (cmp > 0) {
              ++win_count;
            } else if (cmp == 0) {
              ++tie_count;
            } else {
              ++loss_count;
            }
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

int normalize_cuda_block_size(const int raw, const int fallback) {
  int threads = raw > 0 ? raw : fallback;
  threads = std::max(32, std::min(1024, threads));
  threads = (threads / 32) * 32;
  return threads > 0 ? threads : fallback;
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
  int chunk = raw > 0 ? raw : fallback;
  chunk = std::max(1, std::min(entries, chunk));
  return chunk;
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

__device__ inline void set_status_once(int* status, int code) {
  atomicCAS(status, 0, code);
}

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

__global__ void monte_carlo_kernel(
    const jint* low_ids,
    const jint* high_ids,
    const jlong* seeds,
    const int n,
    const int index_offset,
    const int trials,
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

  uint8_t remaining[kRemainingAfterHoleCards];
  fill_remaining_deck(hero_first, hero_second, villain_first, villain_second, remaining);

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
    sample_board_cards(remaining, state, board);

    const int cmp = compare_showdown(hero_first, hero_second, villain_first, villain_second, board);
    if (cmp > 0) {
      ++win_count;
    } else if (cmp == 0) {
      ++tie_count;
    } else {
      ++loss_count;
    }
  }

  const double total = static_cast<double>(trials);
  const double std_error = monte_carlo_stderr(win_count, tie_count, trials);

  wins[idx] = static_cast<double>(win_count) / total;
  ties[idx] = static_cast<double>(tie_count) / total;
  losses[idx] = static_cast<double>(loss_count) / total;
  stderrs[idx] = std_error;
}

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

  uint64_t local_win = 0;
  uint64_t local_tie = 0;
  uint64_t local_loss = 0;
  uint8_t board[kBoardCardCount];

  for (int combo = static_cast<int>(threadIdx.x); combo < kExactBoardCount; combo += static_cast<int>(blockDim.x)) {
    const int base = combo * kBoardCardCount;
    board[0] = remaining[board_combos[base + 0]];
    board[1] = remaining[board_combos[base + 1]];
    board[2] = remaining[board_combos[base + 2]];
    board[3] = remaining[board_combos[base + 3]];
    board[4] = remaining[board_combos[base + 4]];

    const int cmp = compare_showdown(hero_first, hero_second, villain_first, villain_second, board);
    if (cmp > 0) {
      ++local_win;
    } else if (cmp == 0) {
      ++local_tie;
    } else {
      ++local_loss;
    }
  }

  extern __shared__ unsigned long long reduction[];
  unsigned long long* win_counts = reduction;
  unsigned long long* tie_counts = reduction + blockDim.x;
  unsigned long long* loss_counts = reduction + (2 * blockDim.x);
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

__global__ void monte_carlo_kernel_packed(
    const jint* packed_keys,
    const jlong* key_material,
    const int n,
    const int index_offset,
    const int trials,
    const jlong monte_carlo_seed_base,
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

  uint8_t remaining[kRemainingAfterHoleCards];
  fill_remaining_deck(hero_first, hero_second, villain_first, villain_second, remaining);

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
    sample_board_cards(remaining, state, board);

    const int cmp = compare_showdown(hero_first, hero_second, villain_first, villain_second, board);
    if (cmp > 0) {
      ++win_count;
    } else if (cmp == 0) {
      ++tie_count;
    } else {
      ++loss_count;
    }
  }

  const double total = static_cast<double>(trials);
  const double std_error = monte_carlo_stderr(win_count, tie_count, trials);

  wins[idx] = static_cast<jfloat>(static_cast<double>(win_count) / total);
  ties[idx] = static_cast<jfloat>(static_cast<double>(tie_count) / total);
  losses[idx] = static_cast<jfloat>(static_cast<double>(loss_count) / total);
  stderrs[idx] = static_cast<jfloat>(std_error);
}

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

  uint64_t local_win = 0;
  uint64_t local_tie = 0;
  uint64_t local_loss = 0;
  uint8_t board[kBoardCardCount];

  for (int combo = static_cast<int>(threadIdx.x); combo < kExactBoardCount; combo += static_cast<int>(blockDim.x)) {
    const int base = combo * kBoardCardCount;
    board[0] = remaining[board_combos[base + 0]];
    board[1] = remaining[board_combos[base + 1]];
    board[2] = remaining[board_combos[base + 2]];
    board[3] = remaining[board_combos[base + 3]];
    board[4] = remaining[board_combos[base + 4]];

    const int cmp = compare_showdown(hero_first, hero_second, villain_first, villain_second, board);
    if (cmp > 0) {
      ++local_win;
    } else if (cmp == 0) {
      ++local_tie;
    } else {
      ++local_loss;
    }
  }

  extern __shared__ unsigned long long reduction[];
  unsigned long long* win_counts = reduction;
  unsigned long long* tie_counts = reduction + blockDim.x;
  unsigned long long* loss_counts = reduction + (2 * blockDim.x);
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
    if (cmp > 0) {
      ++win_count;
    } else if (cmp == 0) {
      ++tie_count;
    } else {
      ++loss_count;
    }
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
      if (cmp > 0) {
        ++win_count;
      } else if (cmp == 0) {
        ++tie_count;
      } else {
        ++loss_count;
      }
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
  const uint8_t* d_exact_board_combos = nullptr;
  if (mode_code == 0 && !ensure_cuda_exact_board_indices_uploaded_for_device(device, &d_exact_board_combos)) {
    std::fprintf(stderr, "[sicfun-gpu-native] CUDA failure at exact-board upload: cudaMalloc/cudaMemcpy failed\n");
    return 136;
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
          static_cast<size_t>(threads_per_block) * 3ULL * sizeof(unsigned long long);
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
      monte_carlo_kernel<<<blocks, threads_per_block>>>(
          d_low + static_cast<size_t>(offset),
          d_high + static_cast<size_t>(offset),
          d_seeds + static_cast<size_t>(offset),
          chunk,
          offset,
          trials,
          d_wins + static_cast<size_t>(offset),
          d_ties + static_cast<size_t>(offset),
          d_losses + static_cast<size_t>(offset),
          d_stderrs + static_cast<size_t>(offset),
          d_status);
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
  const uint8_t* d_exact_board_combos = nullptr;
  if (mode_code == 0 && !ensure_cuda_exact_board_indices_uploaded_for_device(device, &d_exact_board_combos)) {
    std::fprintf(stderr, "[sicfun-gpu-native] CUDA failure at exact-board upload: cudaMalloc/cudaMemcpy failed\n");
    return 136;
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
          static_cast<size_t>(threads_per_block) * 3ULL * sizeof(unsigned long long);
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
      monte_carlo_kernel_packed<<<blocks, threads_per_block>>>(
          d_packed + static_cast<size_t>(offset),
          d_key_material + static_cast<size_t>(offset),
          chunk,
          offset,
          trials,
          monte_carlo_seed_base,
          d_wins + static_cast<size_t>(offset),
          d_ties + static_cast<size_t>(offset),
          d_losses + static_cast<size_t>(offset),
          d_stderrs + static_cast<size_t>(offset),
          d_status);
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

extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HeadsUpGpuNativeBindings_lastEngineCode(
    JNIEnv*,
    jclass) {
  return g_last_engine_code.load(std::memory_order_relaxed);
}

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
