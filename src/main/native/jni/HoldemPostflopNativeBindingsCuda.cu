#include <jni.h>
#include <cuda_runtime.h>

#include <atomic>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#if defined(__CUDACC__)
#define HD_FORCE __host__ __device__ __forceinline__
#else
#define HD_FORCE inline
#endif

namespace {

constexpr int kDeckSize = 52;
constexpr int kRanksPerSuit = 13;
constexpr int kMinRankValue = 2;
constexpr int kMaxRankValue = 14;
constexpr int kBoardCardCount = 5;
constexpr int kCombos7Of5Count = 21;
constexpr int kMaxRemainingDeck = 50;
constexpr int kDefaultCudaThreadsPerBlock = 128;
constexpr int kDefaultCudaMaxChunkMatchups = 4096;
constexpr int kDefaultCudaMaxTrialsPerLaunch = 4096;
constexpr int kStatusInvalidBoardSize = 128;
constexpr uint64_t kRngMul = 2685821657736338717ULL;
constexpr jint kEngineUnknown = 0;
constexpr jint kEngineCuda = 2;

std::atomic<jint> g_last_engine_code(kEngineUnknown);

bool check_and_clear_exception(JNIEnv* env) {
  if (!env->ExceptionCheck()) {
    return false;
  }
  env->ExceptionClear();
  return true;
}

HD_FORCE bool is_valid_card_id(const int card_id) {
  return card_id >= 0 && card_id < kDeckSize;
}

std::string normalize_token(const std::string& raw) {
  size_t start = 0;
  size_t end = raw.size();
  while (start < end && std::isspace(static_cast<unsigned char>(raw[start])) != 0) {
    ++start;
  }
  while (end > start && std::isspace(static_cast<unsigned char>(raw[end - 1])) != 0) {
    --end;
  }
  std::string out = raw.substr(start, end - start);
  for (char& ch : out) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return out;
}

int parse_positive_text_int(const std::string& text) {
  const std::string trimmed = normalize_token(text);
  if (trimmed.empty()) {
    return -1;
  }
  char* end = nullptr;
  const long parsed = std::strtol(trimmed.c_str(), &end, 10);
  if (end == trimmed.c_str() || *end != '\0' || parsed <= 0 ||
      parsed > std::numeric_limits<int>::max()) {
    return -1;
  }
  return static_cast<int>(parsed);
}

int parse_positive_env_int(const char* value) {
  if (value == nullptr || value[0] == '\0') {
    return -1;
  }
  return parse_positive_text_int(std::string(value));
}

bool try_read_system_property(JNIEnv* env, const char* key, std::string& out) {
  out.clear();
  if (env == nullptr || key == nullptr || key[0] == '\0') {
    return false;
  }
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
  jstring value = static_cast<jstring>(
      env->CallStaticObjectMethod(system_class, get_property, key_string));
  env->DeleteLocalRef(key_string);
  if (check_and_clear_exception(env) || value == nullptr) {
    if (value != nullptr) {
      env->DeleteLocalRef(value);
    }
    env->DeleteLocalRef(system_class);
    return false;
  }
  const char* chars = env->GetStringUTFChars(value, nullptr);
  if (chars != nullptr) {
    out.assign(chars);
    env->ReleaseStringUTFChars(value, chars);
  }
  env->DeleteLocalRef(value);
  env->DeleteLocalRef(system_class);
  return !out.empty();
}

int normalize_cuda_block_size(const int raw) {
  int block = raw;
  if (block < 32) {
    block = 32;
  }
  if (block > 1024) {
    block = 1024;
  }
  block = (block / 32) * 32;
  if (block < 32) {
    block = 32;
  }
  return block;
}

int resolve_cuda_threads_per_block(JNIEnv* env) {
  std::string property_value;
  if (try_read_system_property(env, "sicfun.postflop.native.cuda.blockSize", property_value)) {
    const int parsed = parse_positive_text_int(property_value);
    if (parsed > 0) {
      return normalize_cuda_block_size(parsed);
    }
  }
  const int env_value = parse_positive_env_int(std::getenv("sicfun_POSTFLOP_CUDA_BLOCK_SIZE"));
  if (env_value > 0) {
    return normalize_cuda_block_size(env_value);
  }
  return kDefaultCudaThreadsPerBlock;
}

int normalize_cuda_max_chunk(const int raw, const int entries) {
  if (entries <= 0) {
    return 1;
  }
  int chunk = raw;
  if (chunk <= 0) {
    chunk = kDefaultCudaMaxChunkMatchups;
  }
  if (chunk > entries) {
    chunk = entries;
  }
  if (chunk < 1) {
    chunk = 1;
  }
  return chunk;
}

int resolve_cuda_max_chunk_matchups(JNIEnv* env, const int entries) {
  std::string property_value;
  if (try_read_system_property(env, "sicfun.postflop.native.cuda.maxChunkMatchups", property_value)) {
    const int parsed = parse_positive_text_int(property_value);
    if (parsed > 0) {
      return normalize_cuda_max_chunk(parsed, entries);
    }
  }
  const int env_value = parse_positive_env_int(std::getenv("sicfun_POSTFLOP_CUDA_MAX_CHUNK_MATCHUPS"));
  if (env_value > 0) {
    return normalize_cuda_max_chunk(env_value, entries);
  }
  return normalize_cuda_max_chunk(kDefaultCudaMaxChunkMatchups, entries);
}

int normalize_cuda_max_trials_per_launch(const int raw, const int trials) {
  if (trials <= 0) {
    return 1;
  }
  int chunk_trials = raw;
  if (chunk_trials <= 0) {
    chunk_trials = kDefaultCudaMaxTrialsPerLaunch;
  }
  if (chunk_trials > trials) {
    chunk_trials = trials;
  }
  if (chunk_trials < 1) {
    chunk_trials = 1;
  }
  return chunk_trials;
}

int resolve_cuda_max_trials_per_launch(JNIEnv* env, const int trials) {
  std::string property_value;
  if (try_read_system_property(env, "sicfun.postflop.native.cuda.maxTrialsPerLaunch", property_value)) {
    const int parsed = parse_positive_text_int(property_value);
    if (parsed > 0) {
      return normalize_cuda_max_trials_per_launch(parsed, trials);
    }
  }
  const int env_value = parse_positive_env_int(std::getenv("sicfun_POSTFLOP_CUDA_MAX_TRIALS_PER_LAUNCH"));
  if (env_value > 0) {
    return normalize_cuda_max_trials_per_launch(env_value, trials);
  }
  return normalize_cuda_max_trials_per_launch(kDefaultCudaMaxTrialsPerLaunch, trials);
}

HD_FORCE int card_rank(const int card_id) {
  return (card_id % kRanksPerSuit) + kMinRankValue;
}

HD_FORCE int card_suit(const int card_id) {
  return card_id / kRanksPerSuit;
}

HD_FORCE uint32_t encode_score(const int category, const int* tiebreak, const int tiebreak_size) {
  uint32_t score = static_cast<uint32_t>(category) << 24;
  for (int i = 0; i < tiebreak_size && i < 5; ++i) {
    score |= (static_cast<uint32_t>(tiebreak[i] & 0x0F) << (20 - (i * 4)));
  }
  return score;
}

HD_FORCE uint32_t evaluate5_score(const int cards[5]) {
  int suits[5];
  int rank_counts[kMaxRankValue + 1];
  for (int rank = 0; rank <= kMaxRankValue; ++rank) {
    rank_counts[rank] = 0;
  }
  for (int i = 0; i < 5; ++i) {
    const int rank = card_rank(cards[i]);
    suits[i] = card_suit(cards[i]);
    ++rank_counts[rank];
  }

  bool is_flush = true;
  for (int i = 1; i < 5; ++i) {
    if (suits[i] != suits[0]) {
      is_flush = false;
      break;
    }
  }

  int distinct = 0;
  for (int rank = kMinRankValue; rank <= kMaxRankValue; ++rank) {
    if (rank_counts[rank] > 0) {
      ++distinct;
    }
  }

  int straight_high = 0;
  if (distinct == 5) {
    for (int high = kMaxRankValue; high >= 5; --high) {
      if (rank_counts[high] > 0 && rank_counts[high - 1] > 0 && rank_counts[high - 2] > 0 &&
          rank_counts[high - 3] > 0 && rank_counts[high - 4] > 0) {
        straight_high = high;
        break;
      }
    }
    if (straight_high == 0 && rank_counts[14] > 0 && rank_counts[5] > 0 && rank_counts[4] > 0 &&
        rank_counts[3] > 0 && rank_counts[2] > 0) {
      straight_high = 5;
    }
  }

  int four_rank = 0;
  int three_rank = 0;
  int pair_high = 0;
  int pair_low = 0;
  int singles[5];
  int singles_count = 0;
  for (int rank = kMaxRankValue; rank >= kMinRankValue; --rank) {
    const int count = rank_counts[rank];
    if (count == 4) {
      four_rank = rank;
    } else if (count == 3) {
      three_rank = rank;
    } else if (count == 2) {
      if (pair_high == 0) {
        pair_high = rank;
      } else {
        pair_low = rank;
      }
    } else if (count == 1) {
      singles[singles_count++] = rank;
    }
  }

  int tiebreak[5] = {0, 0, 0, 0, 0};
  if (is_flush && straight_high > 0) {
    tiebreak[0] = straight_high;
    return encode_score(8, tiebreak, 1);
  }
  if (four_rank > 0) {
    tiebreak[0] = four_rank;
    tiebreak[1] = singles[0];
    return encode_score(7, tiebreak, 2);
  }
  if (three_rank > 0 && pair_high > 0) {
    tiebreak[0] = three_rank;
    tiebreak[1] = pair_high;
    return encode_score(6, tiebreak, 2);
  }
  if (is_flush) {
    int idx = 0;
    for (int rank = kMaxRankValue; rank >= kMinRankValue; --rank) {
      const int copies = rank_counts[rank];
      for (int c = 0; c < copies; ++c) {
        tiebreak[idx++] = rank;
      }
    }
    return encode_score(5, tiebreak, 5);
  }
  if (straight_high > 0) {
    tiebreak[0] = straight_high;
    return encode_score(4, tiebreak, 1);
  }
  if (three_rank > 0) {
    tiebreak[0] = three_rank;
    tiebreak[1] = singles[0];
    tiebreak[2] = singles[1];
    return encode_score(3, tiebreak, 3);
  }
  if (pair_high > 0 && pair_low > 0) {
    tiebreak[0] = pair_high;
    tiebreak[1] = pair_low;
    tiebreak[2] = singles[0];
    return encode_score(2, tiebreak, 3);
  }
  if (pair_high > 0) {
    tiebreak[0] = pair_high;
    tiebreak[1] = singles[0];
    tiebreak[2] = singles[1];
    tiebreak[3] = singles[2];
    return encode_score(1, tiebreak, 4);
  }
  for (int i = 0; i < 5; ++i) {
    tiebreak[i] = singles[i];
  }
  return encode_score(0, tiebreak, 5);
}

HD_FORCE uint32_t evaluate7_score(const int cards[7]) {
  int five_cards[5];
  uint32_t best = 0;
#define EVAL_COMBO(A, B, C, D, E)                                    \
  five_cards[0] = cards[A];                                           \
  five_cards[1] = cards[B];                                           \
  five_cards[2] = cards[C];                                           \
  five_cards[3] = cards[D];                                           \
  five_cards[4] = cards[E];                                           \
  {                                                                   \
    const uint32_t score = evaluate5_score(five_cards);               \
    if (score > best) {                                               \
      best = score;                                                   \
    }                                                                 \
  }
  EVAL_COMBO(0, 1, 2, 3, 4)
  EVAL_COMBO(0, 1, 2, 3, 5)
  EVAL_COMBO(0, 1, 2, 3, 6)
  EVAL_COMBO(0, 1, 2, 4, 5)
  EVAL_COMBO(0, 1, 2, 4, 6)
  EVAL_COMBO(0, 1, 2, 5, 6)
  EVAL_COMBO(0, 1, 3, 4, 5)
  EVAL_COMBO(0, 1, 3, 4, 6)
  EVAL_COMBO(0, 1, 3, 5, 6)
  EVAL_COMBO(0, 1, 4, 5, 6)
  EVAL_COMBO(0, 2, 3, 4, 5)
  EVAL_COMBO(0, 2, 3, 4, 6)
  EVAL_COMBO(0, 2, 3, 5, 6)
  EVAL_COMBO(0, 2, 4, 5, 6)
  EVAL_COMBO(0, 3, 4, 5, 6)
  EVAL_COMBO(1, 2, 3, 4, 5)
  EVAL_COMBO(1, 2, 3, 4, 6)
  EVAL_COMBO(1, 2, 3, 5, 6)
  EVAL_COMBO(1, 2, 4, 5, 6)
  EVAL_COMBO(1, 3, 4, 5, 6)
  EVAL_COMBO(2, 3, 4, 5, 6)
#undef EVAL_COMBO
  return best;
}

HD_FORCE int compare_showdown(
    const int hero_first,
    const int hero_second,
    const int villain_first,
    const int villain_second,
    const int board[kBoardCardCount]) {
  int hero_cards[7] = {hero_first, hero_second, board[0], board[1], board[2], board[3], board[4]};
  int villain_cards[7] = {
      villain_first, villain_second, board[0], board[1], board[2], board[3], board[4]};
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

HD_FORCE uint64_t mix64(uint64_t value) {
  uint64_t z = value + 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z = z ^ (z >> 31);
  return z;
}

HD_FORCE uint64_t next_u64(uint64_t& state) {
  uint64_t x = state;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  state = x;
  return x * kRngMul;
}

HD_FORCE int bounded_rand(uint64_t& state, const int bound) {
  if (bound <= 1) {
    return 0;
  }
  const uint64_t ubound = static_cast<uint64_t>(bound);
  const uint64_t max_u64 = 0xFFFFFFFFFFFFFFFFULL;
  const uint64_t limit = max_u64 - (max_u64 % ubound);
  while (true) {
    const uint64_t value = next_u64(state);
    if (value < limit) {
      return static_cast<int>(value % ubound);
    }
  }
}

__global__ void postflop_monte_carlo_kernel(
    const int hero_first,
    const int hero_second,
    const int* board_cards,
    const int board_size,
    const int* villain_first_cards,
    const int* villain_second_cards,
    const int trials,
    const uint64_t* seeds,
    double* wins,
    double* ties,
    double* losses,
    double* stderrs,
    const int n) {
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= n) {
    return;
  }

  const int villain_first = villain_first_cards[idx];
  const int villain_second = villain_second_cards[idx];
  uint64_t dead_mask =
      (1ULL << static_cast<uint64_t>(hero_first)) |
      (1ULL << static_cast<uint64_t>(hero_second)) |
      (1ULL << static_cast<uint64_t>(villain_first)) |
      (1ULL << static_cast<uint64_t>(villain_second));
  for (int b = 0; b < board_size; ++b) {
    dead_mask |= (1ULL << static_cast<uint64_t>(board_cards[b]));
  }

  int remaining[kMaxRemainingDeck];
  int rem_count = 0;
  for (int card = 0; card < kDeckSize; ++card) {
    const uint64_t bit = (1ULL << static_cast<uint64_t>(card));
    if ((dead_mask & bit) == 0ULL) {
      remaining[rem_count++] = card;
    }
  }

  int board[kBoardCardCount];
  for (int b = 0; b < board_size; ++b) {
    board[b] = board_cards[b];
  }

  const int cards_needed = kBoardCardCount - board_size;
  if (cards_needed == 0) {
    const int cmp = compare_showdown(hero_first, hero_second, villain_first, villain_second, board);
    if (cmp > 0) {
      wins[idx] = static_cast<double>(trials);
      ties[idx] = 0.0;
      losses[idx] = 0.0;
    } else if (cmp == 0) {
      wins[idx] = 0.0;
      ties[idx] = static_cast<double>(trials);
      losses[idx] = 0.0;
    } else {
      wins[idx] = 0.0;
      ties[idx] = 0.0;
      losses[idx] = static_cast<double>(trials);
    }
    stderrs[idx] = 0.0;
    return;
  }

  uint64_t rng_state = mix64(seeds[idx] ^ 0xD6E8FEB86659FD93ULL);
  if (rng_state == 0ULL) {
    rng_state = 0x9E3779B97F4A7C15ULL;
  }

  int win_count = 0;
  int tie_count = 0;
  int loss_count = 0;
  for (int t = 0; t < trials; ++t) {
    uint64_t used = 0ULL;
    int filled = 0;
    while (filled < cards_needed) {
      const int ri = bounded_rand(rng_state, rem_count);
      const uint64_t bit = (1ULL << static_cast<uint64_t>(ri));
      if ((used & bit) == 0ULL) {
        used |= bit;
        board[board_size + filled] = remaining[ri];
        ++filled;
      }
    }
    const int cmp = compare_showdown(hero_first, hero_second, villain_first, villain_second, board);
    if (cmp > 0) {
      ++win_count;
    } else if (cmp == 0) {
      ++tie_count;
    } else {
      ++loss_count;
    }
  }
  wins[idx] = static_cast<double>(win_count);
  ties[idx] = static_cast<double>(tie_count);
  losses[idx] = static_cast<double>(loss_count);
  stderrs[idx] = 0.0;
}

}  // namespace

extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HoldemPostflopNativeGpuBindings_computePostflopBatchMonteCarlo(
    JNIEnv* env,
    jclass,
    jint hero_first,
    jint hero_second,
    jintArray board_cards,
    jint board_size,
    jintArray villain_first_cards,
    jintArray villain_second_cards,
    jint trials,
    jlongArray seeds,
    jdoubleArray wins,
    jdoubleArray ties,
    jdoubleArray losses,
    jdoubleArray stderrs) {
  g_last_engine_code.store(kEngineUnknown, std::memory_order_relaxed);

  if (board_cards == nullptr || villain_first_cards == nullptr || villain_second_cards == nullptr ||
      seeds == nullptr || wins == nullptr || ties == nullptr || losses == nullptr || stderrs == nullptr) {
    return 100;
  }
  if (board_size < 1 || board_size > kBoardCardCount) {
    return kStatusInvalidBoardSize;
  }
  if (trials <= 0) {
    return 126;
  }

  const jsize board_len = env->GetArrayLength(board_cards);
  const jsize n = env->GetArrayLength(villain_first_cards);
  if (board_len != board_size ||
      env->GetArrayLength(villain_second_cards) != n ||
      env->GetArrayLength(seeds) != n ||
      env->GetArrayLength(wins) != n ||
      env->GetArrayLength(ties) != n ||
      env->GetArrayLength(losses) != n ||
      env->GetArrayLength(stderrs) != n) {
    return 101;
  }

  std::vector<jint> board_buf(static_cast<size_t>(board_size));
  std::vector<jint> villain_first_buf(static_cast<size_t>(n));
  std::vector<jint> villain_second_buf(static_cast<size_t>(n));
  std::vector<jlong> seed_buf(static_cast<size_t>(n));
  std::vector<jdouble> win_buf(static_cast<size_t>(n));
  std::vector<jdouble> tie_buf(static_cast<size_t>(n));
  std::vector<jdouble> loss_buf(static_cast<size_t>(n));
  std::vector<jdouble> stderr_buf(static_cast<size_t>(n));
  std::vector<jdouble> total_win_count(static_cast<size_t>(n), 0.0);
  std::vector<jdouble> total_tie_count(static_cast<size_t>(n), 0.0);
  std::vector<jdouble> total_loss_count(static_cast<size_t>(n), 0.0);

  env->GetIntArrayRegion(board_cards, 0, board_size, board_buf.data());
  env->GetIntArrayRegion(villain_first_cards, 0, n, villain_first_buf.data());
  env->GetIntArrayRegion(villain_second_cards, 0, n, villain_second_buf.data());
  env->GetLongArrayRegion(seeds, 0, n, seed_buf.data());
  if (check_and_clear_exception(env)) {
    return 102;
  }

  if (!is_valid_card_id(hero_first) || !is_valid_card_id(hero_second)) {
    return 125;
  }
  bool fixed_dead[kDeckSize] = {};
  if (fixed_dead[hero_first]) {
    return 127;
  }
  fixed_dead[hero_first] = true;
  if (fixed_dead[hero_second]) {
    return 127;
  }
  fixed_dead[hero_second] = true;

  for (int b = 0; b < board_size; ++b) {
    const jint card_id = board_buf[static_cast<size_t>(b)];
    if (!is_valid_card_id(card_id)) {
      return 125;
    }
    if (fixed_dead[card_id]) {
      return 127;
    }
    fixed_dead[card_id] = true;
  }
  for (jsize i = 0; i < n; ++i) {
    const jint vf = villain_first_buf[static_cast<size_t>(i)];
    const jint vs = villain_second_buf[static_cast<size_t>(i)];
    if (!is_valid_card_id(vf) || !is_valid_card_id(vs)) {
      return 125;
    }
    if (vf == vs || fixed_dead[vf] || fixed_dead[vs]) {
      return 127;
    }
  }

  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count <= 0) {
    return 130;
  }
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return 130;
  }

  const int threads_per_block = resolve_cuda_threads_per_block(env);
  const int max_chunk_matchups = resolve_cuda_max_chunk_matchups(env, static_cast<int>(n));
  const int chunk_size = normalize_cuda_max_chunk(max_chunk_matchups, static_cast<int>(n));
  const int max_trials_per_launch = resolve_cuda_max_trials_per_launch(env, static_cast<int>(trials));

  int* d_board = nullptr;
  int* d_villain_first = nullptr;
  int* d_villain_second = nullptr;
  uint64_t* d_seed = nullptr;
  double* d_win = nullptr;
  double* d_tie = nullptr;
  double* d_loss = nullptr;
  double* d_stderr = nullptr;

  auto cleanup = [&]() {
    if (d_board != nullptr) cudaFree(d_board);
    if (d_villain_first != nullptr) cudaFree(d_villain_first);
    if (d_villain_second != nullptr) cudaFree(d_villain_second);
    if (d_seed != nullptr) cudaFree(d_seed);
    if (d_win != nullptr) cudaFree(d_win);
    if (d_tie != nullptr) cudaFree(d_tie);
    if (d_loss != nullptr) cudaFree(d_loss);
    if (d_stderr != nullptr) cudaFree(d_stderr);
  };

  err = cudaMalloc(reinterpret_cast<void**>(&d_board), sizeof(int) * static_cast<size_t>(kBoardCardCount));
  if (err != cudaSuccess) {
    cleanup();
    return 131;
  }
  err = cudaMemcpy(
      d_board,
      board_buf.data(),
      sizeof(int) * static_cast<size_t>(board_size),
      cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cleanup();
    return 132;
  }
  err = cudaMalloc(reinterpret_cast<void**>(&d_villain_first), sizeof(int) * static_cast<size_t>(chunk_size));
  if (err != cudaSuccess) {
    cleanup();
    return 131;
  }
  err = cudaMalloc(reinterpret_cast<void**>(&d_villain_second), sizeof(int) * static_cast<size_t>(chunk_size));
  if (err != cudaSuccess) {
    cleanup();
    return 131;
  }
  err = cudaMalloc(reinterpret_cast<void**>(&d_seed), sizeof(uint64_t) * static_cast<size_t>(chunk_size));
  if (err != cudaSuccess) {
    cleanup();
    return 131;
  }
  err = cudaMalloc(reinterpret_cast<void**>(&d_win), sizeof(double) * static_cast<size_t>(chunk_size));
  if (err != cudaSuccess) {
    cleanup();
    return 131;
  }
  err = cudaMalloc(reinterpret_cast<void**>(&d_tie), sizeof(double) * static_cast<size_t>(chunk_size));
  if (err != cudaSuccess) {
    cleanup();
    return 131;
  }
  err = cudaMalloc(reinterpret_cast<void**>(&d_loss), sizeof(double) * static_cast<size_t>(chunk_size));
  if (err != cudaSuccess) {
    cleanup();
    return 131;
  }
  err = cudaMalloc(reinterpret_cast<void**>(&d_stderr), sizeof(double) * static_cast<size_t>(chunk_size));
  if (err != cudaSuccess) {
    cleanup();
    return 131;
  }

  std::vector<uint64_t> seed_u64(seed_buf.size());
  for (size_t i = 0; i < seed_buf.size(); ++i) {
    seed_u64[i] = static_cast<uint64_t>(seed_buf[i]);
  }

  int offset = 0;
  while (offset < n) {
    const int chunk = std::min(chunk_size, static_cast<int>(n - offset));
    const size_t chunk_int_bytes = sizeof(int) * static_cast<size_t>(chunk);
    const size_t chunk_seed_bytes = sizeof(uint64_t) * static_cast<size_t>(chunk);
    const size_t chunk_double_bytes = sizeof(double) * static_cast<size_t>(chunk);
    std::vector<uint64_t> seed_chunk(static_cast<size_t>(chunk));
    std::vector<double> chunk_win(static_cast<size_t>(chunk));
    std::vector<double> chunk_tie(static_cast<size_t>(chunk));
    std::vector<double> chunk_loss(static_cast<size_t>(chunk));

    err = cudaMemcpy(d_villain_first, villain_first_buf.data() + offset, chunk_int_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      cleanup();
      return 132;
    }
    err = cudaMemcpy(d_villain_second, villain_second_buf.data() + offset, chunk_int_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      cleanup();
      return 132;
    }

    int trial_offset = 0;
    while (trial_offset < trials) {
      const int launch_trials = std::min(max_trials_per_launch, static_cast<int>(trials - trial_offset));
      const uint64_t launch_mix =
          static_cast<uint64_t>(static_cast<uint32_t>(trial_offset + 1)) * 0x9E3779B97F4A7C15ULL;
      for (int i = 0; i < chunk; ++i) {
        const uint64_t base_seed = seed_u64[static_cast<size_t>(offset + i)];
        seed_chunk[static_cast<size_t>(i)] = mix64(base_seed ^ launch_mix);
      }

      err = cudaMemcpy(d_seed, seed_chunk.data(), chunk_seed_bytes, cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
        cleanup();
        return 132;
      }

      const int blocks = (chunk + threads_per_block - 1) / threads_per_block;
      postflop_monte_carlo_kernel<<<blocks, threads_per_block>>>(
          static_cast<int>(hero_first),
          static_cast<int>(hero_second),
          d_board,
          static_cast<int>(board_size),
          d_villain_first,
          d_villain_second,
          launch_trials,
          d_seed,
          d_win,
          d_tie,
          d_loss,
          d_stderr,
          chunk);
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        cleanup();
        return 133;
      }
      err = cudaDeviceSynchronize();
      if (err != cudaSuccess) {
        cleanup();
        return err == cudaErrorLaunchTimeout ? 137 : 134;
      }

      err = cudaMemcpy(chunk_win.data(), d_win, chunk_double_bytes, cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) {
        cleanup();
        return 135;
      }
      err = cudaMemcpy(chunk_tie.data(), d_tie, chunk_double_bytes, cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) {
        cleanup();
        return 135;
      }
      err = cudaMemcpy(chunk_loss.data(), d_loss, chunk_double_bytes, cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) {
        cleanup();
        return 135;
      }

      for (int i = 0; i < chunk; ++i) {
        const size_t global_idx = static_cast<size_t>(offset + i);
        total_win_count[global_idx] += chunk_win[static_cast<size_t>(i)];
        total_tie_count[global_idx] += chunk_tie[static_cast<size_t>(i)];
        total_loss_count[global_idx] += chunk_loss[static_cast<size_t>(i)];
      }
      trial_offset += launch_trials;
    }

    offset += chunk;
  }

  const double total_trials = static_cast<double>(trials);
  for (jsize i = 0; i < n; ++i) {
    const size_t idx = static_cast<size_t>(i);
    const double win_count = total_win_count[idx];
    const double tie_count = total_tie_count[idx];
    const double loss_count = total_loss_count[idx];
    win_buf[idx] = win_count / total_trials;
    tie_buf[idx] = tie_count / total_trials;
    loss_buf[idx] = loss_count / total_trials;
    const double mean = (win_count + (0.5 * tie_count)) / total_trials;
    const double ex2 = (win_count + (0.25 * tie_count)) / total_trials;
    const double variance_population = std::max(0.0, ex2 - (mean * mean));
    const double variance_sample =
        trials > 1 ? (variance_population * total_trials / static_cast<double>(trials - 1)) : 0.0;
    stderr_buf[idx] = sqrt(variance_sample / total_trials);
  }

  cleanup();
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
Java_sicfun_holdem_HoldemPostflopNativeGpuBindings_queryNativeEngine(
    JNIEnv*,
    jclass) {
  return g_last_engine_code.load(std::memory_order_relaxed);
}

extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HoldemPostflopNativeGpuBindings_cudaDeviceCount(
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
Java_sicfun_holdem_HoldemPostflopNativeGpuBindings_cudaDeviceInfo(
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
  std::snprintf(
      buf,
      sizeof(buf),
      "%s|%d|%d|%d|%d.%d",
      prop.name,
      prop.multiProcessorCount,
      prop.clockRate / 1000,
      memory_mb,
      prop.major,
      prop.minor);
  return env->NewStringUTF(buf);
}
