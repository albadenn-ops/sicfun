/*
 * HeadsUpGpuNativeBindings.cpp -- CPU-only native implementation of heads-up
 * Texas Hold'em equity computation for JNI.
 *
 * This file implements the native side of HeadsUpGpuNativeBindings.computeBatch
 * without any CUDA dependency.  It supports two computation modes:
 *   - Exact exhaustive enumeration over all C(48,5) = 1,712,304 boards.
 *   - Monte Carlo sampling with per-matchup PRNG seeds and Welford online
 *     variance for standard error estimation.
 *
 * Work is parallelized across hardware threads using a lock-free work-stealing
 * pattern with atomic index advancement.
 *
 * Produces: sicfun_native_cpu.dll (CPU-only, no GPU -- despite the binding
 * class name containing "Gpu").
 *
 * Error status codes returned to the JVM:
 *   100 -- null array argument
 *   101 -- array length mismatch
 *   102 -- JNI read error (GetXxxArrayRegion failed)
 *   111 -- invalid mode code
 *   124 -- JNI write error (SetDoubleArrayRegion failed)
 *   125 -- invalid hole-card pair id (out of 0..1325)
 *   126 -- invalid trial count (must be > 0 for Monte Carlo)
 *   127 -- overlapping cards between hero and villain
 */

#include <jni.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

namespace {

/* ---- Constants ----------------------------------------------------------- */

constexpr int kDeckSize = 52;               // Standard deck size.
constexpr int kRanksPerSuit = 13;           // 2..A per suit.
constexpr int kMinRankValue = 2;            // Rank encoding: 2 = deuce.
constexpr int kMaxRankValue = 14;           // Rank encoding: 14 = ace.
constexpr int kHoleCardsCount = 1326;       // C(52,2) canonical hole-card pairs.
constexpr int kRemainingAfterHoleCards = 48; // 52 - 4 (two hero + two villain).
constexpr int kBoardCardCount = 5;          // Community board cards.
constexpr int kCombos7Of5Count = 21;        // C(7,5) five-card sub-hand combos.
constexpr int kWorkChunkSize = 8;           // Matchups per atomic work-steal chunk.
constexpr int kIdBits = 11;
constexpr int kIdMask = (1 << kIdBits) - 1;
constexpr int kStatusInvalidRangeLayout = 112;
constexpr jint kEngineUnknown = 0;
constexpr jint kEngineCpu = 1;

std::atomic<jint> g_last_engine_code(kEngineUnknown);

int resolve_worker_count(int entries);

// Scala LookupTables.combos7of5 order.
constexpr std::array<std::array<int, 5>, kCombos7Of5Count> kCombos7Of5 = {{
    {{0, 1, 2, 3, 4}},
    {{0, 1, 2, 3, 5}},
    {{0, 1, 2, 3, 6}},
    {{0, 1, 2, 4, 5}},
    {{0, 1, 2, 4, 6}},
    {{0, 1, 2, 5, 6}},
    {{0, 1, 3, 4, 5}},
    {{0, 1, 3, 4, 6}},
    {{0, 1, 3, 5, 6}},
    {{0, 1, 4, 5, 6}},
    {{0, 2, 3, 4, 5}},
    {{0, 2, 3, 4, 6}},
    {{0, 2, 3, 5, 6}},
    {{0, 2, 4, 5, 6}},
    {{0, 3, 4, 5, 6}},
    {{1, 2, 3, 4, 5}},
    {{1, 2, 3, 4, 6}},
    {{1, 2, 3, 5, 6}},
    {{1, 2, 4, 5, 6}},
    {{1, 3, 4, 5, 6}},
    {{2, 3, 4, 5, 6}},
}};

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

static inline uint64_t xorshift64star(uint64_t& state) {
  state ^= state >> 12;
  state ^= state << 25;
  state ^= state >> 27;
  return state * 0x2545F4914F6CDD1DULL;
}

bool check_and_clear_exception(JNIEnv* env) {
  if (!env->ExceptionCheck()) {
    return false;
  }
  env->ExceptionClear();
  return true;
}

inline int card_rank(const int card_id) {
  return (card_id % kRanksPerSuit) + kMinRankValue;
}

inline int card_suit(const int card_id) {
  return card_id / kRanksPerSuit;
}

const std::array<HoleCards, kHoleCardsCount>& hole_cards_lookup() {
  static const std::array<HoleCards, kHoleCardsCount> lookup = [] {
    std::array<HoleCards, kHoleCardsCount> table{};
    int idx = 0;
    for (int first = 0; first < (kDeckSize - 1); ++first) {
      for (int second = first + 1; second < kDeckSize; ++second) {
        table[static_cast<size_t>(idx)] = HoleCards{
            static_cast<uint8_t>(first),
            static_cast<uint8_t>(second),
        };
        ++idx;
      }
    }
    return table;
  }();
  return lookup;
}

inline uint32_t encode_score(const int category, const int* __restrict__ tiebreak, const int tiebreak_size) {
  uint32_t score = static_cast<uint32_t>(category) << 24;
  for (int i = 0; i < tiebreak_size && i < 5; ++i) {
    score |= (static_cast<uint32_t>(tiebreak[i] & 0x0F) << (20 - (i * 4)));
  }
  return score;
}

uint32_t evaluate5_score(const int* __restrict__ cards) {
  int ranks[5];
  int suits[5];
  int rank_counts[kMaxRankValue + 1] = {};
  for (int i = 0; i < 5; ++i) {
    ranks[i] = card_rank(cards[i]);
    suits[i] = card_suit(cards[i]);
    ++rank_counts[ranks[i]];
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

  struct Group {
    int count;
    int rank;
  };
  Group groups[5];
  int group_count = 0;
  for (int rank = kMaxRankValue; rank >= kMinRankValue; --rank) {
    if (rank_counts[rank] > 0) {
      groups[group_count++] = Group{rank_counts[rank], rank};
    }
  }
  // Insertion sort — group_count is 2..5; avoids std::sort overhead entirely.
  for (int gi = 1; gi < group_count; ++gi) {
    const Group key = groups[gi];
    int gj = gi - 1;
    while (gj >= 0 &&
           (groups[gj].count < key.count ||
            (groups[gj].count == key.count && groups[gj].rank < key.rank))) {
      groups[gj + 1] = groups[gj];
      --gj;
    }
    groups[gj + 1] = key;
  }

  int tiebreak[5] = {};
  if (is_flush && straight_high > 0) {
    tiebreak[0] = straight_high;
    return encode_score(8, tiebreak, 1);  // StraightFlush
  }
  if (group_count == 2 && groups[0].count == 4) {
    tiebreak[0] = groups[0].rank;
    tiebreak[1] = groups[1].rank;
    return encode_score(7, tiebreak, 2);  // FourOfKind
  }
  if (group_count == 2 && groups[0].count == 3 && groups[1].count == 2) {
    tiebreak[0] = groups[0].rank;
    tiebreak[1] = groups[1].rank;
    return encode_score(6, tiebreak, 2);  // FullHouse
  }
  if (is_flush) {
    int idx = 0;
    for (int rank = kMaxRankValue; rank >= kMinRankValue; --rank) {
      const int copies = rank_counts[rank];
      for (int c = 0; c < copies; ++c) {
        tiebreak[idx++] = rank;
      }
    }
    return encode_score(5, tiebreak, 5);  // Flush
  }
  if (straight_high > 0) {
    tiebreak[0] = straight_high;
    return encode_score(4, tiebreak, 1);  // Straight
  }
  if (group_count == 3 && groups[0].count == 3) {
    tiebreak[0] = groups[0].rank;
    int idx = 1;
    for (int g = 1; g < group_count; ++g) {
      tiebreak[idx++] = groups[g].rank;
    }
    return encode_score(3, tiebreak, 3);  // ThreeOfKind
  }
  if (group_count == 3 && groups[0].count == 2 && groups[1].count == 2) {
    tiebreak[0] = groups[0].rank;
    tiebreak[1] = groups[1].rank;
    tiebreak[2] = groups[2].rank;
    return encode_score(2, tiebreak, 3);  // TwoPair
  }
  if (group_count == 4 && groups[0].count == 2) {
    tiebreak[0] = groups[0].rank;
    int idx = 1;
    for (int g = 1; g < group_count; ++g) {
      tiebreak[idx++] = groups[g].rank;
    }
    return encode_score(1, tiebreak, 4);  // OnePair
  }
  int idx = 0;
  for (int rank = kMaxRankValue; rank >= kMinRankValue; --rank) {
    if (rank_counts[rank] > 0) {
      tiebreak[idx++] = rank;
    }
  }
  return encode_score(0, tiebreak, 5);  // HighCard
}

uint32_t evaluate7_score(const int* __restrict__ cards) {
  int five_cards[5];
  uint32_t best = 0;
  for (int c = 0; c < kCombos7Of5Count; ++c) {
    const auto& combo = kCombos7Of5[static_cast<size_t>(c)];
    for (int i = 0; i < 5; ++i) {
      five_cards[i] = cards[combo[static_cast<size_t>(i)]];
    }
    const uint32_t score = evaluate5_score(five_cards);
    if (score > best) {
      best = score;
    }
  }
  return best;
}

void fill_remaining_deck(
    const int hero_first,
    const int hero_second,
    const int villain_first,
    const int villain_second,
    int* __restrict__ remaining) {
  const uint64_t dead_mask =
      (UINT64_C(1) << hero_first) |
      (UINT64_C(1) << hero_second) |
      (UINT64_C(1) << villain_first) |
      (UINT64_C(1) << villain_second);

  uint8_t idx = 0;
  for (uint8_t card = 0; card < kDeckSize; ++card) {
    if (!((dead_mask >> card) & 1)) {
      remaining[idx++] = card;
    }
  }
}

inline int compare_showdown(
    const int hero_first,
    const int hero_second,
    const int villain_first,
    const int villain_second,
    const int* __restrict__ board) {
  int hero_cards[7] = {hero_first, hero_second, board[0], board[1], board[2], board[3], board[4]};
  int villain_cards[7] = {
      villain_first, villain_second, board[0], board[1], board[2], board[3], board[4]};
  const uint32_t hero_score = evaluate7_score(hero_cards);
  const uint32_t villain_score = evaluate7_score(villain_cards);
  return static_cast<int>(hero_score > villain_score) - static_cast<int>(hero_score < villain_score);
}

inline uint64_t mix64(uint64_t value) {
  uint64_t z = value + 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z = z ^ (z >> 31);
  return z;
}

EquityResultNative compute_monte_carlo_equity(
    const int hero_first,
    const int hero_second,
    const int villain_first,
    const int villain_second,
    const int trials,
    const uint64_t seed) {
  int remaining[kRemainingAfterHoleCards];
  fill_remaining_deck(hero_first, hero_second, villain_first, villain_second, remaining);

  int shuffled[kRemainingAfterHoleCards];
  int board[kBoardCardCount];
  int win_count = 0;
  int tie_count = 0;
  int loss_count = 0;
  double mean = 0.0;
  double m2 = 0.0;

  uint64_t rng_state = (seed != 0) ? seed : 1ULL;  // xorshift64* requires non-zero state
  for (int trial = 0; trial < trials; ++trial) {
    std::memcpy(shuffled, remaining, sizeof(shuffled));
    for (int i = 0; i < kBoardCardCount; ++i) {
      const int range = kRemainingAfterHoleCards - i;
      const int j = i + static_cast<int>(xorshift64star(rng_state) % static_cast<uint64_t>(range));
      std::swap(shuffled[i], shuffled[j]);
      board[i] = shuffled[i];
    }

    const int cmp = compare_showdown(hero_first, hero_second, villain_first, villain_second, board);
    double outcome = 0.0;
    if (cmp > 0) {
      ++win_count;
      outcome = 1.0;
    } else if (cmp == 0) {
      ++tie_count;
      outcome = 0.5;
    } else {
      ++loss_count;
      outcome = 0.0;
    }

    const double delta = outcome - mean;
    mean += delta / static_cast<double>(trial + 1);
    const double delta2 = outcome - mean;
    m2 += delta * delta2;
  }

  const double total = static_cast<double>(trials);
  const double variance = trials > 1 ? (m2 / static_cast<double>(trials - 1)) : 0.0;
  const double std_error = std::sqrt(variance / total);
  return EquityResultNative{
      static_cast<double>(win_count) / total,
      static_cast<double>(tie_count) / total,
      static_cast<double>(loss_count) / total,
      std_error,
  };
}

EquityResultNative compute_exact_equity(
    const int hero_first,
    const int hero_second,
    const int villain_first,
    const int villain_second) {
  int remaining[kRemainingAfterHoleCards];
  fill_remaining_deck(hero_first, hero_second, villain_first, villain_second, remaining);

  uint64_t win_count = 0;
  uint64_t tie_count = 0;
  uint64_t loss_count = 0;
  uint64_t total = 0;
  int board[kBoardCardCount];

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
        const EquityResultNative result = compute_monte_carlo_equity(
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

  const auto& lookup = hole_cards_lookup();
  std::atomic<jsize> next_index(0);
  std::atomic<jint> worker_error(0);
  const int workers = resolve_worker_count(static_cast<int>(n));

  auto worker = [&]() {
    while (true) {
      if (worker_error.load(std::memory_order_relaxed) != 0) {
        return;
      }
      const jsize start = next_index.fetch_add(kWorkChunkSize, std::memory_order_relaxed);
      if (start >= n) {
        return;
      }
      const jsize end = std::min<jsize>(n, start + kWorkChunkSize);
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
          result = compute_exact_equity(hero.first, hero.second, villain.first, villain.second);
        } else {
          result = compute_monte_carlo_equity(
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
  if (worker_error.load(std::memory_order_relaxed) != 0) {
    return worker_error.load(std::memory_order_relaxed);
  }

  env->SetDoubleArrayRegion(wins, 0, n, win_buf.data());
  env->SetDoubleArrayRegion(ties, 0, n, tie_buf.data());
  env->SetDoubleArrayRegion(losses, 0, n, loss_buf.data());
  env->SetDoubleArrayRegion(stderrs, 0, n, stderr_buf.data());
  if (check_and_clear_exception(env)) {
    return 124;
  }

  g_last_engine_code.store(kEngineCpu, std::memory_order_relaxed);
  return 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HeadsUpGpuNativeBindings_computeBatchCpuOnly(
    JNIEnv* env,
    jclass clazz,
    jintArray low_ids,
    jintArray high_ids,
    jint mode_code,
    jint trials,
    jlongArray seeds,
    jdoubleArray wins,
    jdoubleArray ties,
    jdoubleArray losses,
    jdoubleArray stderrs) {
  return Java_sicfun_holdem_HeadsUpGpuNativeBindings_computeBatch(
      env,
      clazz,
      low_ids,
      high_ids,
      mode_code,
      trials,
      seeds,
      wins,
      ties,
      losses,
      stderrs);
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
    const uint32_t packed = static_cast<uint32_t>(packed_buf[static_cast<size_t>(i)]);
    const int low_id = static_cast<int>((packed >> kIdBits) & static_cast<uint32_t>(kIdMask));
    const int high_id = static_cast<int>(packed & static_cast<uint32_t>(kIdMask));
    if (low_id < 0 || low_id >= kHoleCardsCount || high_id < 0 || high_id >= kHoleCardsCount) {
      return 125;
    }
  }

  const auto& lookup = hole_cards_lookup();
  std::atomic<jsize> next_index(0);
  std::atomic<jint> worker_error(0);
  const int workers = resolve_worker_count(static_cast<int>(n));

  auto worker = [&]() {
    while (true) {
      if (worker_error.load(std::memory_order_relaxed) != 0) {
        return;
      }
      const jsize start = next_index.fetch_add(kWorkChunkSize, std::memory_order_relaxed);
      if (start >= n) {
        return;
      }
      const jsize end = std::min<jsize>(n, start + kWorkChunkSize);
      for (jsize i = start; i < end; ++i) {
        const size_t idx = static_cast<size_t>(i);
        const uint32_t packed = static_cast<uint32_t>(packed_buf[idx]);
        const int low_id = static_cast<int>((packed >> kIdBits) & static_cast<uint32_t>(kIdMask));
        const int high_id = static_cast<int>(packed & static_cast<uint32_t>(kIdMask));
        const HoleCards hero = lookup[static_cast<size_t>(low_id)];
        const HoleCards villain = lookup[static_cast<size_t>(high_id)];
        const bool overlap = hero.first == villain.first || hero.first == villain.second ||
                             hero.second == villain.first || hero.second == villain.second;
        if (overlap) {
          worker_error.store(127, std::memory_order_relaxed);
          return;
        }

        EquityResultNative result{};
        if (mode_code == 0) {
          result = compute_exact_equity(hero.first, hero.second, villain.first, villain.second);
        } else {
          const uint64_t local_seed = mix64(
              static_cast<uint64_t>(monte_carlo_seed_base) ^
              static_cast<uint64_t>(key_material_buf[idx]));
          result = compute_monte_carlo_equity(
              hero.first,
              hero.second,
              villain.first,
              villain.second,
              trials,
              local_seed);
        }
        win_buf[idx] = static_cast<jfloat>(result.win);
        tie_buf[idx] = static_cast<jfloat>(result.tie);
        loss_buf[idx] = static_cast<jfloat>(result.loss);
        stderr_buf[idx] = static_cast<jfloat>(result.std_error);
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
  if (worker_error.load(std::memory_order_relaxed) != 0) {
    return worker_error.load(std::memory_order_relaxed);
  }

  env->SetFloatArrayRegion(wins, 0, n, win_buf.data());
  env->SetFloatArrayRegion(ties, 0, n, tie_buf.data());
  env->SetFloatArrayRegion(losses, 0, n, loss_buf.data());
  env->SetFloatArrayRegion(stderrs, 0, n, stderr_buf.data());
  if (check_and_clear_exception(env)) {
    return 124;
  }

  g_last_engine_code.store(kEngineCpu, std::memory_order_relaxed);
  return 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HeadsUpGpuNativeBindings_computeBatchPackedOnDevice(
    JNIEnv*,
    jclass,
    jint,
    jintArray,
    jint,
    jint,
    jlong,
    jlongArray,
    jfloatArray,
    jfloatArray,
    jfloatArray,
    jfloatArray) {
  return 130;
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

  const jint status = compute_range_batch_cpu_monte_carlo_csr(
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

  env->SetFloatArrayRegion(wins, 0, hero_count, out_win.data());
  env->SetFloatArrayRegion(ties, 0, hero_count, out_tie.data());
  env->SetFloatArrayRegion(losses, 0, hero_count, out_loss.data());
  env->SetFloatArrayRegion(stderrs, 0, hero_count, out_stderr.data());
  if (check_and_clear_exception(env)) {
    return 124;
  }

  g_last_engine_code.store(kEngineCpu, std::memory_order_relaxed);
  return 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HeadsUpGpuNativeBindings_computeRangeBatchMonteCarloCsrOnDevice(
    JNIEnv*,
    jclass,
    jint,
    jintArray,
    jintArray,
    jintArray,
    jlongArray,
    jfloatArray,
    jint,
    jlong,
    jfloatArray,
    jfloatArray,
    jfloatArray,
    jfloatArray) {
  return 130;
}

extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HeadsUpGpuNativeBindings_lastEngineCode(
    JNIEnv*,
    jclass) {
  return g_last_engine_code.load(std::memory_order_relaxed);
}
