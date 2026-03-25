#include <jni.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <thread>
#include <vector>

namespace {

constexpr int kDeckSize = 52;
constexpr int kRanksPerSuit = 13;
constexpr int kMinRankValue = 2;
constexpr int kMaxRankValue = 14;
constexpr int kBoardCardCount = 5;
constexpr int kMaxRemainingDeck = 50;
constexpr int kWorkChunkSize = 8;
constexpr int kStatusInvalidBoardSize = 128;
constexpr jint kEngineUnknown = 0;
constexpr jint kEngineCpu = 1;

std::atomic<jint> g_last_engine_code(kEngineUnknown);

struct EquityResultNative {
  double win;
  double tie;
  double loss;
  double std_error;
};

bool check_and_clear_exception(JNIEnv* env) {
  if (!env->ExceptionCheck()) {
    return false;
  }
  env->ExceptionClear();
  return true;
}

inline bool is_valid_card_id(const int card_id) {
  return card_id >= 0 && card_id < kDeckSize;
}

inline int card_rank(const int card_id) {
  return (card_id % kRanksPerSuit) + kMinRankValue;
}

inline int card_suit(const int card_id) {
  return card_id / kRanksPerSuit;
}

inline uint16_t card_rank_bit(const int card_id) {
  return static_cast<uint16_t>(1) << static_cast<uint16_t>(card_id % kRanksPerSuit);
}

struct EvalState {
  uint8_t rank_counts[kMaxRankValue + 1] = {};
  uint8_t suit_counts[4] = {0, 0, 0, 0};
  uint16_t rank_mask = 0;
  uint16_t suit_rank_mask[4] = {0, 0, 0, 0};
  uint16_t pair_mask = 0;
  uint16_t trip_mask = 0;
  uint16_t quad_mask = 0;
};

inline void add_card_to_state(EvalState& state, const int card) {
  const int rank = card_rank(card);
  const int suit = card_suit(card);
  const uint16_t bit = card_rank_bit(card);
  const uint8_t count = static_cast<uint8_t>(state.rank_counts[rank] + 1);
  state.rank_counts[rank] = count;
  ++state.suit_counts[suit];
  state.rank_mask = static_cast<uint16_t>(state.rank_mask | bit);
  state.suit_rank_mask[suit] = static_cast<uint16_t>(state.suit_rank_mask[suit] | bit);
  if (count >= 2) {
    state.pair_mask = static_cast<uint16_t>(state.pair_mask | bit);
  }
  if (count >= 3) {
    state.trip_mask = static_cast<uint16_t>(state.trip_mask | bit);
  }
  if (count == 4) {
    state.quad_mask = static_cast<uint16_t>(state.quad_mask | bit);
  }
}

inline EvalState build_eval_state(
    const int first,
    const int second,
    const int* __restrict__ board_cards,
    const int board_size) {
  EvalState state;
  add_card_to_state(state, first);
  add_card_to_state(state, second);
  for (int idx = 0; idx < board_size; ++idx) {
    add_card_to_state(state, board_cards[idx]);
  }
  return state;
}

inline uint32_t encode_score(const int category, const int* tiebreak, const int tiebreak_size) {
  uint32_t score = static_cast<uint32_t>(category) << 24;
  for (int i = 0; i < tiebreak_size && i < 5; ++i) {
    score |= (static_cast<uint32_t>(tiebreak[i] & 0x0F) << (20 - (i * 4)));
  }
  return score;
}

#if defined(_MSC_VER)
#include <intrin.h>
static inline int popcount_16(uint16_t x) { return static_cast<int>(__popcnt16(x)); }
static inline int highest_bit_index_16(uint16_t x) {
  if (x == 0) return -1;
  unsigned long idx;
  _BitScanReverse(&idx, static_cast<unsigned long>(x));
  return static_cast<int>(idx);
}
#else
static inline int popcount_16(uint16_t x) { return __builtin_popcount(x); }
static inline int highest_bit_index_16(uint16_t x) {
  if (x == 0) return -1;
  return 31 - __builtin_clz(static_cast<unsigned int>(x));
}
#endif

inline int highest_rank_from_mask(const uint16_t mask) {
  const int bit = highest_bit_index_16(mask);
  return bit >= 0 ? (bit + kMinRankValue) : 0;
}

inline int straight_high_from_rank_mask(const uint16_t rank_mask) {
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

uint32_t evaluate7_score_from_masks(
    const uint16_t rank_mask,
    const uint16_t pair_mask,
    const uint16_t trip_mask,
    const uint16_t quad_mask,
    const uint8_t* __restrict__ suit_counts,
    const uint16_t* __restrict__ suit_rank_mask) {
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
      return encode_score(8, tiebreak, 1);
    }
  }

  if (quad_mask != 0) {
    const int quad_rank = highest_rank_from_mask(quad_mask);
    const uint16_t quad_bit = static_cast<uint16_t>(1) << static_cast<uint16_t>(quad_rank - kMinRankValue);
    const int kicker = highest_rank_from_mask(static_cast<uint16_t>(rank_mask & static_cast<uint16_t>(~quad_bit)));
    tiebreak[0] = quad_rank;
    tiebreak[1] = kicker;
    return encode_score(7, tiebreak, 2);
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
      return encode_score(6, tiebreak, 2);
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
    return encode_score(5, tiebreak, 5);
  }

  const int straight_high = straight_high_from_rank_mask(rank_mask);
  if (straight_high > 0) {
    tiebreak[0] = straight_high;
    return encode_score(4, tiebreak, 1);
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
    return encode_score(3, tiebreak, 3);
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
    return encode_score(2, tiebreak, 3);
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
    return encode_score(1, tiebreak, 4);
  }

  uint16_t mask = rank_mask;
  for (int idx = 0; idx < 5; ++idx) {
    const int rank = highest_rank_from_mask(mask);
    tiebreak[idx] = rank;
    const uint16_t bit = static_cast<uint16_t>(1) << static_cast<uint16_t>(rank - kMinRankValue);
    mask = static_cast<uint16_t>(mask & static_cast<uint16_t>(~bit));
  }
  return encode_score(0, tiebreak, 5);
}

inline uint32_t evaluate_state_score(const EvalState& state) {
  return evaluate7_score_from_masks(
      state.rank_mask,
      state.pair_mask,
      state.trip_mask,
      state.quad_mask,
      state.suit_counts,
      state.suit_rank_mask);
}

inline uint32_t evaluate7_score_direct(
    const int c0,
    const int c1,
    const int c2,
    const int c3,
    const int c4,
    const int c5,
    const int c6) {
  EvalState state;
  add_card_to_state(state, c0);
  add_card_to_state(state, c1);
  add_card_to_state(state, c2);
  add_card_to_state(state, c3);
  add_card_to_state(state, c4);
  add_card_to_state(state, c5);
  add_card_to_state(state, c6);
  return evaluate_state_score(state);
}

inline int compare_scores(const uint32_t hero_score, const uint32_t villain_score) {
  return static_cast<int>(hero_score > villain_score) - static_cast<int>(hero_score < villain_score);
}

inline int compare_completed_states(const EvalState& hero_state, const EvalState& villain_state) {
  return compare_scores(evaluate_state_score(hero_state), evaluate_state_score(villain_state));
}

inline int compare_with_one_runout(
    const EvalState& hero_base,
    const EvalState& villain_base,
    const int card) {
  EvalState hero_state = hero_base;
  EvalState villain_state = villain_base;
  add_card_to_state(hero_state, card);
  add_card_to_state(villain_state, card);
  return compare_completed_states(hero_state, villain_state);
}

inline int compare_with_two_runout(
    const EvalState& hero_base,
    const EvalState& villain_base,
    const int first,
    const int second) {
  EvalState hero_state = hero_base;
  EvalState villain_state = villain_base;
  add_card_to_state(hero_state, first);
  add_card_to_state(hero_state, second);
  add_card_to_state(villain_state, first);
  add_card_to_state(villain_state, second);
  return compare_completed_states(hero_state, villain_state);
}

inline int compare_with_runout(
    const EvalState& hero_base,
    const EvalState& villain_base,
    const int* runout_cards,
    const int runout_count) {
  EvalState hero_state = hero_base;
  EvalState villain_state = villain_base;
  for (int idx = 0; idx < runout_count; ++idx) {
    const int card = runout_cards[idx];
    add_card_to_state(hero_state, card);
    add_card_to_state(villain_state, card);
  }
  return compare_completed_states(hero_state, villain_state);
}

jint fill_remaining_deck_postflop(
    const int hero_first,
    const int hero_second,
    const int villain_first,
    const int villain_second,
    const int* __restrict__ board_cards,
    const int board_size,
    int* __restrict__ remaining,
    int* __restrict__ remaining_count) {
  uint64_t dead_mask = 0;
  auto mark_dead = [&](const int card_id) -> bool {
    const uint64_t bit = uint64_t(1) << card_id;
    if (dead_mask & bit) {
      return false;
    }
    dead_mask |= bit;
    return true;
  };

  if (!mark_dead(hero_first) || !mark_dead(hero_second)) {
    return 127;
  }
  if (!mark_dead(villain_first) || !mark_dead(villain_second)) {
    return 127;
  }
  for (int b = 0; b < board_size; ++b) {
    if (!mark_dead(board_cards[b])) {
      return 127;
    }
  }

  int idx = 0;
  for (int card = 0; card < kDeckSize; ++card) {
    if (!(dead_mask & (uint64_t(1) << card))) {
      remaining[idx++] = card;
    }
  }
  *remaining_count = idx;
  return 0;
}

void sample_runout_cards(
    const int* __restrict__ remaining,
    const int remaining_count,
    const int cards_needed,
    std::mt19937_64& rng,
    int* __restrict__ runout_cards) {
  if (cards_needed <= 0) {
    return;
  }
  int shuffled[kMaxRemainingDeck];
  std::memcpy(shuffled, remaining, sizeof(int) * static_cast<size_t>(remaining_count));
  std::uniform_int_distribution<int> dist;
  for (int i = 0; i < cards_needed; ++i) {
    const int j = dist(rng, decltype(dist)::param_type(i, remaining_count - 1));
    std::swap(shuffled[i], shuffled[j]);
    runout_cards[i] = shuffled[i];
  }
}

jint compute_postflop_mc_equity_cpu(
    const int hero_first,
    const int hero_second,
    const int* __restrict__ board_cards,
    const int board_size,
    const int villain_first,
    const int villain_second,
    const int trials,
    const uint64_t seed,
    EquityResultNative* __restrict__ out) {
  const int cards_needed = kBoardCardCount - board_size;
  if (cards_needed < 0 || cards_needed > kBoardCardCount) {
    return kStatusInvalidBoardSize;
  }

  int remaining[kMaxRemainingDeck];
  int remaining_count = 0;
  const jint fill_status = fill_remaining_deck_postflop(
      hero_first,
      hero_second,
      villain_first,
      villain_second,
      board_cards,
      board_size,
      remaining,
      &remaining_count);
  if (fill_status != 0) {
    return fill_status;
  }

  const EvalState hero_base = build_eval_state(hero_first, hero_second, board_cards, board_size);
  const EvalState villain_base = build_eval_state(villain_first, villain_second, board_cards, board_size);

  if (cards_needed == 0) {
    const int cmp = compare_completed_states(hero_base, villain_base);
    if (cmp > 0) {
      *out = EquityResultNative{1.0, 0.0, 0.0, 0.0};
    } else if (cmp == 0) {
      *out = EquityResultNative{0.0, 1.0, 0.0, 0.0};
    } else {
      *out = EquityResultNative{0.0, 0.0, 1.0, 0.0};
    }
    return 0;
  }

  if (cards_needed == 1) {
    int win_count = 0;
    int tie_count = 0;
    int loss_count = 0;
    for (int i = 0; i < remaining_count; ++i) {
      const int cmp = compare_with_one_runout(hero_base, villain_base, remaining[i]);
      win_count  += (cmp > 0);
      tie_count  += (cmp == 0);
      loss_count += (cmp < 0);
    }

    const double total = static_cast<double>(remaining_count);
    *out = EquityResultNative{
        static_cast<double>(win_count) / total,
        static_cast<double>(tie_count) / total,
        static_cast<double>(loss_count) / total,
        0.0,
    };
    return 0;
  }

  int win_count = 0;
  int tie_count = 0;
  int loss_count = 0;
  double mean = 0.0;
  double m2 = 0.0;
  int runout[kBoardCardCount];

  auto record_outcome = [&](const int cmp, const int trial) {
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
  };

  std::mt19937_64 rng(seed);
  if (cards_needed == 2) {
    std::uniform_int_distribution<int> dist;
    const auto first_param = decltype(dist)::param_type(0, remaining_count - 1);
    const auto second_param = decltype(dist)::param_type(0, remaining_count - 2);
    for (int trial = 0; trial < trials; ++trial) {
      const int first_idx = dist(rng, first_param);
      int second_idx = dist(rng, second_param);
      if (second_idx >= first_idx) {
        ++second_idx;
      }
      const int cmp =
          compare_with_two_runout(hero_base, villain_base, remaining[first_idx], remaining[second_idx]);
      record_outcome(cmp, trial);
    }
  } else {
    for (int trial = 0; trial < trials; ++trial) {
      sample_runout_cards(remaining, remaining_count, cards_needed, rng, runout);
      const int cmp = compare_with_runout(hero_base, villain_base, runout, cards_needed);
      record_outcome(cmp, trial);
    }
  }

  const double total = static_cast<double>(trials);
  const double variance = trials > 1 ? (m2 / static_cast<double>(trials - 1)) : 0.0;
  const double std_error = std::sqrt(variance / total);
  *out = EquityResultNative{
      static_cast<double>(win_count) / total,
      static_cast<double>(tie_count) / total,
      static_cast<double>(loss_count) / total,
      std_error,
  };
  return 0;
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
  const int env_workers = parse_positive_env_int(std::getenv("sicfun_POSTFLOP_NATIVE_THREADS"));
  if (env_workers > 0) {
    workers = env_workers;
  }
  const int max_chunks = std::max(1, (entries + kWorkChunkSize - 1) / kWorkChunkSize);
  workers = std::max(1, std::min(workers, std::min(entries, max_chunks)));
  return workers;
}

}  // namespace

extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HoldemPostflopNativeBindings_computePostflopBatchMonteCarlo(
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
  uint64_t fixed_dead = 0;
  auto fixed_mark_dead = [&](const int card_id) -> bool {
    const uint64_t bit = uint64_t(1) << card_id;
    if (fixed_dead & bit) return false;
    fixed_dead |= bit;
    return true;
  };
  if (!fixed_mark_dead(hero_first) || !fixed_mark_dead(hero_second)) {
    return 127;
  }

  for (int b = 0; b < board_size; ++b) {
    const jint card_id = board_buf[static_cast<size_t>(b)];
    if (!is_valid_card_id(card_id)) {
      return 125;
    }
    if (!fixed_mark_dead(card_id)) {
      return 127;
    }
  }

  for (jsize i = 0; i < n; ++i) {
    const jint vf = villain_first_buf[static_cast<size_t>(i)];
    const jint vs = villain_second_buf[static_cast<size_t>(i)];
    if (!is_valid_card_id(vf) || !is_valid_card_id(vs)) {
      return 125;
    }
    if (vf == vs || (fixed_dead & (uint64_t(1) << vf)) || (fixed_dead & (uint64_t(1) << vs))) {
      return 127;
    }
  }

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
        EquityResultNative result{};
        const jint status = compute_postflop_mc_equity_cpu(
            static_cast<int>(hero_first),
            static_cast<int>(hero_second),
            board_buf.data(),
            static_cast<int>(board_size),
            static_cast<int>(villain_first_buf[idx]),
            static_cast<int>(villain_second_buf[idx]),
            static_cast<int>(trials),
            static_cast<uint64_t>(seed_buf[idx]),
            &result);
        if (status != 0) {
          worker_error.store(status, std::memory_order_relaxed);
          return;
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
Java_sicfun_holdem_HoldemPostflopNativeBindings_queryNativeEngine(
    JNIEnv*,
    jclass) {
  return g_last_engine_code.load(std::memory_order_relaxed);
}
