/*
 * OpenCL Monte Carlo equity kernel for heads-up Texas Hold'em.
 *
 * Port of the CUDA monte_carlo_kernel from HeadsUpGpuNativeBindingsCuda.cu.
 * Must produce bit-identical PRNG sequences for the same seeds so that
 * results are deterministic regardless of which device processes a matchup.
 *
 * Only Monte Carlo mode is supported; exact enumeration falls back to CPU.
 */

#define DECK_SIZE          52
#define RANKS_PER_SUIT     13
#define MIN_RANK_VALUE     2
#define MAX_RANK_VALUE     14
#define HOLE_CARDS_COUNT   1326
#define REMAINING_AFTER_HOLE 48
#define BOARD_CARD_COUNT   5
#define RNG_MUL            0x254BE62B56B3EBBDULL  /* 2685821657736338717 */

/* ── card helpers ─────────────────────────────────────────────────── */

inline int card_rank(const int card_id) {
  return (card_id % RANKS_PER_SUIT) + MIN_RANK_VALUE;
}

inline int card_suit(const int card_id) {
  return card_id / RANKS_PER_SUIT;
}

/* ── PRNG (xorshift64*) ── identical constants to CUDA version ──── */

inline ulong mix64(ulong value) {
  ulong z = value + 0x9E3779B97F4A7C15UL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
  z = z ^ (z >> 31);
  return z;
}

inline ulong next_u64(ulong *state) {
  *state ^= (*state >> 12);
  *state ^= (*state << 25);
  *state ^= (*state >> 27);
  return *state * RNG_MUL;
}

inline int bounded_rand(ulong *state, const int bound) {
  return (int)(next_u64(state) % (ulong)bound);
}

/* ── deck & board ────────────────────────────────────────────────── */

void fill_remaining_deck(
    const int hero_first,
    const int hero_second,
    const int villain_first,
    const int villain_second,
    uchar remaining[REMAINING_AFTER_HOLE]) {
  const ulong dead_mask = (1UL << hero_first) | (1UL << hero_second) |
                          (1UL << villain_first) | (1UL << villain_second);
  int idx = 0;
  for (int card = 0; card < DECK_SIZE; ++card) {
    if (!((dead_mask >> card) & 1UL)) {
      remaining[idx++] = (uchar)card;
    }
  }
}

inline void sample_board_cards(
    const uchar remaining[REMAINING_AFTER_HOLE],
    ulong *state,
    uchar board[BOARD_CARD_COUNT]) {
  ulong used = 0UL;
  int filled = 0;
  while (filled < BOARD_CARD_COUNT) {
    const int deck_idx = bounded_rand(state, REMAINING_AFTER_HOLE);
    const ulong bit = 1UL << deck_idx;
    if ((used & bit) == 0UL) {
      used |= bit;
      board[filled++] = remaining[deck_idx];
    }
  }
}

/* ── hand evaluator (7-card → score) ─────────────────────────────── */

inline uint encode_score(const int category, const int tiebreak[5], const int tb_size) {
  uint score = (uint)category << 24;
  for (int i = 0; i < tb_size && i < 5; ++i) {
    score |= ((uint)(tiebreak[i] & 0x0F) << (20 - (i * 4)));
  }
  return score;
}

inline int straight_high_from_rank_mask(const ushort rank_mask) {
  for (int high = MAX_RANK_VALUE; high >= 6; --high) {
    const ushort window = (ushort)0x1F << (high - 6);
    if ((rank_mask & window) == window) return high;
  }
  const ushort wheel_mask =
      ((ushort)1 << 12) | ((ushort)1 << 3) | ((ushort)1 << 2) |
      ((ushort)1 << 1) | ((ushort)1 << 0);
  if ((rank_mask & wheel_mask) == wheel_mask) return 5;
  return 0;
}

uint evaluate7_score(const int cards[7]) {
  int rank_counts[MAX_RANK_VALUE + 1];
  for (int r = 0; r <= MAX_RANK_VALUE; ++r) rank_counts[r] = 0;

  int suit_counts[4] = {0, 0, 0, 0};
  ushort rank_mask = 0;
  ushort suit_rank_mask[4] = {0, 0, 0, 0};

  for (int i = 0; i < 7; ++i) {
    const int rank = card_rank(cards[i]);
    const int suit = card_suit(cards[i]);
    ++rank_counts[rank];
    ++suit_counts[suit];
    const ushort bit = (ushort)1 << (rank - MIN_RANK_VALUE);
    rank_mask |= bit;
    suit_rank_mask[suit] |= bit;
  }

  int flush_suit = -1;
  for (int suit = 0; suit < 4; ++suit) {
    if (suit_counts[suit] >= 5) { flush_suit = suit; break; }
  }

  int tiebreak[5] = {0, 0, 0, 0, 0};

  /* Straight flush */
  if (flush_suit >= 0) {
    const int sf_high = straight_high_from_rank_mask(suit_rank_mask[flush_suit]);
    if (sf_high > 0) { tiebreak[0] = sf_high; return encode_score(8, tiebreak, 1); }
  }

  int quad_rank = 0, trip1 = 0, trip2 = 0, pair1 = 0, pair2 = 0;
  for (int rank = MAX_RANK_VALUE; rank >= MIN_RANK_VALUE; --rank) {
    const int cnt = rank_counts[rank];
    if (cnt == 4)      quad_rank = rank;
    else if (cnt == 3) { if (trip1 == 0) trip1 = rank; else trip2 = rank; }
    else if (cnt == 2) { if (pair1 == 0) pair1 = rank; else pair2 = rank; }
  }

  /* Four of a kind */
  if (quad_rank > 0) {
    int kicker = 0;
    for (int r = MAX_RANK_VALUE; r >= MIN_RANK_VALUE; --r) {
      if (r != quad_rank && rank_counts[r] > 0) { kicker = r; break; }
    }
    tiebreak[0] = quad_rank; tiebreak[1] = kicker;
    return encode_score(7, tiebreak, 2);
  }

  /* Full house */
  if (trip1 > 0) {
    const int fh_pair = (trip2 > 0) ? trip2 : pair1;
    if (fh_pair > 0) {
      tiebreak[0] = trip1; tiebreak[1] = fh_pair;
      return encode_score(6, tiebreak, 2);
    }
  }

  /* Flush */
  if (flush_suit >= 0) {
    int fi = 0;
    for (int r = MAX_RANK_VALUE; r >= MIN_RANK_VALUE && fi < 5; --r) {
      const ushort bit = (ushort)1 << (r - MIN_RANK_VALUE);
      if ((suit_rank_mask[flush_suit] & bit) != 0) tiebreak[fi++] = r;
    }
    return encode_score(5, tiebreak, 5);
  }

  /* Straight */
  const int str_high = straight_high_from_rank_mask(rank_mask);
  if (str_high > 0) { tiebreak[0] = str_high; return encode_score(4, tiebreak, 1); }

  /* Three of a kind */
  if (trip1 > 0) {
    int ki = 1;
    for (int r = MAX_RANK_VALUE; r >= MIN_RANK_VALUE && ki < 3; --r) {
      if (r != trip1 && rank_counts[r] > 0) tiebreak[ki++] = r;
    }
    tiebreak[0] = trip1;
    return encode_score(3, tiebreak, 3);
  }

  /* Two pair */
  if (pair1 > 0 && pair2 > 0) {
    int kicker = 0;
    for (int r = MAX_RANK_VALUE; r >= MIN_RANK_VALUE; --r) {
      if (r != pair1 && r != pair2 && rank_counts[r] > 0) { kicker = r; break; }
    }
    tiebreak[0] = pair1; tiebreak[1] = pair2; tiebreak[2] = kicker;
    return encode_score(2, tiebreak, 3);
  }

  /* One pair */
  if (pair1 > 0) {
    int ki = 1;
    for (int r = MAX_RANK_VALUE; r >= MIN_RANK_VALUE && ki < 4; --r) {
      if (r != pair1 && rank_counts[r] > 0) tiebreak[ki++] = r;
    }
    tiebreak[0] = pair1;
    return encode_score(1, tiebreak, 4);
  }

  /* High card */
  int hi = 0;
  for (int r = MAX_RANK_VALUE; r >= MIN_RANK_VALUE && hi < 5; --r) {
    if (rank_counts[r] > 0) tiebreak[hi++] = r;
  }
  return encode_score(0, tiebreak, 5);
}

/* ── showdown comparison ─────────────────────────────────────────── */

inline int compare_showdown(
    const int hf, const int hs,
    const int vf, const int vs,
    const uchar board[BOARD_CARD_COUNT]) {
  int hero[7]    = {hf, hs, (int)board[0], (int)board[1], (int)board[2], (int)board[3], (int)board[4]};
  int villain[7] = {vf, vs, (int)board[0], (int)board[1], (int)board[2], (int)board[3], (int)board[4]};
  const uint h_score = evaluate7_score(hero);
  const uint v_score = evaluate7_score(villain);
  return (h_score > v_score) - (h_score < v_score);
}

/* ── Monte Carlo standard error ──────────────────────────────────── */

inline double monte_carlo_stderr(
    const int win_count, const int tie_count, const int trials) {
  if (trials <= 1) return 0.0;
  const double n = (double)trials;
  const double mean = ((double)win_count + 0.5 * (double)tie_count) / n;
  const double ex2  = ((double)win_count + 0.25 * (double)tie_count) / n;
  const double pop_var = fmax(ex2 - (mean * mean), 0.0);
  const double sample_var = pop_var * (n / (n - 1.0));
  return sqrt(sample_var / n);
}

/* ══════════════════════════════════════════════════════════════════
 *  Main Monte Carlo kernel — one work-item per matchup
 * ══════════════════════════════════════════════════════════════════ */

__kernel void monte_carlo_kernel(
    __global const int    *low_ids,
    __global const int    *high_ids,
    __global const long   *seeds,
    const int              n,
    const int              index_offset,
    const int              trials,
    __global double       *wins,
    __global double       *ties,
    __global double       *losses,
    __global double       *stderrs,
    __global int          *status,
    __constant const uchar *hole_first,
    __constant const uchar *hole_second) {

  const int idx = (int)get_global_id(0);
  if (idx >= n) return;

  const int low_id  = low_ids[idx];
  const int high_id = high_ids[idx];
  if (low_id < 0 || low_id >= HOLE_CARDS_COUNT || high_id < 0 || high_id >= HOLE_CARDS_COUNT) {
    atomic_cmpxchg(status, 0, 125);
    return;
  }

  const int hero_first    = (int)hole_first[low_id];
  const int hero_second   = (int)hole_second[low_id];
  const int villain_first = (int)hole_first[high_id];
  const int villain_second= (int)hole_second[high_id];

  const int overlap = (hero_first == villain_first)  || (hero_first == villain_second) ||
                      (hero_second == villain_first) || (hero_second == villain_second);
  if (overlap) {
    atomic_cmpxchg(status, 0, 127);
    return;
  }

  uchar remaining[REMAINING_AFTER_HOLE];
  fill_remaining_deck(hero_first, hero_second, villain_first, villain_second, remaining);

  uchar board[BOARD_CARD_COUNT];
  int win_count  = 0;
  int tie_count  = 0;
  int loss_count = 0;

  const int global_idx = idx + index_offset;
  ulong state = mix64((ulong)seeds[idx] ^ (ulong)(global_idx + 1));
  if (state == 0UL) state = 0x9E3779B97F4A7C15UL;

  for (int trial = 0; trial < trials; ++trial) {
    sample_board_cards(remaining, &state, board);
    const int cmp = compare_showdown(hero_first, hero_second, villain_first, villain_second, board);
    win_count  += (cmp > 0);
    tie_count  += (cmp == 0);
    loss_count += (cmp < 0);
  }

  const double total = (double)trials;
  wins[idx]    = (double)win_count  / total;
  ties[idx]    = (double)tie_count  / total;
  losses[idx]  = (double)loss_count / total;
  stderrs[idx] = monte_carlo_stderr(win_count, tie_count, trials);
}
