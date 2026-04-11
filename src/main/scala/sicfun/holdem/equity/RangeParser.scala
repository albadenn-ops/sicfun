package sicfun.holdem.equity
import sicfun.holdem.types.*

import sicfun.core.{Card, Rank, Suit, DiscreteDistribution}

/**
  * Parser for standard poker range notation strings into probability distributions over hole cards.
  *
  * Supports the full range of common poker shorthand:
  *   - '''Single hands''': `AKs` (suited), `AKo` (offsuit), `AK` (all combos), `JJ` (pair)
  *   - '''Plus ranges''': `QQ+` (QQ, KK, AA), `ATs+` (ATs, AJs, AQs, AKs)
  *   - '''Dash ranges''': `99-66` (99, 88, 77, 66), `A5s-A2s` (A5s through A2s)
  *   - '''Weighted tokens''': `AKs:0.5` or `AKs@50%` (half-weight suited AK)
  *   - '''Comma-separated''': `AA, KK, QQ, AKs` (union of multiple range tokens)
  *
  * The parser handles overlapping tokens by summing weights (e.g., `AK, AKs` gives double
  * weight to suited AK combos). The output is a normalized [[DiscreteDistribution]] over
  * all concrete [[HoleCards]] combos in the range.
  *
  * Suitedness rules:
  *   - Pairs cannot specify suitedness (e.g., `AAs` is rejected since pairs are always offsuit)
  *   - Dash ranges must have matching suitedness on both endpoints
  *   - Non-pair tokens without suitedness specifier expand to all 16 combos (4 suited + 12 offsuit)
  *
  * @see [[HoldemEquity]] which uses this parser for string-based equity API overloads
  */
object RangeParser:
  /** Suitedness qualifier for range tokens. */
  enum Suitedness:
    /** No suitedness specified; expands to all 16 combos (suited + offsuit). */
    case Any
    /** Suited only; expands to 4 same-suit combos (e.g., AsKs, AhKh, AdKd, AcKc). */
    case Suited
    /** Offsuit only; expands to 12 cross-suit combos. */
    case Offsuit

  /** All 13 ranks in descending order (Ace high to Deuce low), used for range expansion. */
  private val ranksDesc: Vector[Rank] =
    Vector(
      Rank.Ace,
      Rank.King,
      Rank.Queen,
      Rank.Jack,
      Rank.Ten,
      Rank.Nine,
      Rank.Eight,
      Rank.Seven,
      Rank.Six,
      Rank.Five,
      Rank.Four,
      Rank.Three,
      Rank.Two
    )

  /** Reverse lookup: rank integer value to Rank enum. */
  private val rankByValue: Map[Int, Rank] = ranksDesc.map(r => r.value -> r).toMap
  /** All four suits in a fixed order for combo generation. */
  private val suits: Vector[Suit] = Vector(Suit.Clubs, Suit.Diamonds, Suit.Hearts, Suit.Spades)

  /**
    * Result of parsing a range string: both the normalized probability distribution
    * (for equity calculations) and the set of distinct hands (for display/counting).
    *
    * @param distribution normalized probability distribution over hole cards
    * @param hands        the set of unique hole-card combos present in the range
    */
  final case class ParseResult(distribution: DiscreteDistribution[HoleCards], hands: Set[HoleCards])

  /** Parses a range string into a normalized probability distribution.
    * Returns Left with an error message on invalid input.
    */
  def parse(range: String): Either[String, DiscreteDistribution[HoleCards]] =
    parseWithHands(range).map(_.distribution)

  /** Parses a range string, returning both the distribution and the set of concrete hands.
    * Tokens are comma-separated and individually parsed, with weights accumulated additively
    * for overlapping combos.
    */
  def parseWithHands(range: String): Either[String, ParseResult] =
    val tokens = range.split(",").map(_.trim).filter(_.nonEmpty)
    if tokens.isEmpty then Left("range is empty")
    else
      val handWeights = scala.collection.mutable.Map.empty[HoleCards, Double].withDefaultValue(0.0)
      tokens.foldLeft(Option.empty[String]) { (err, token) =>
        err.orElse {
          parseWeightedToken(token) match
            case Left(message) => Some(s"$token: $message")
            case Right((baseToken, weight)) =>
              expandToken(baseToken) match
                case Left(message) => Some(s"$token: $message")
                case Right(hands) =>
                  hands.foreach { hand =>
                    handWeights.update(hand, handWeights(hand) + weight)
                  }
                  None
        }
      } match
        case Some(message) => Left(message)
        case None =>
          if handWeights.isEmpty then Left("range produced no hands")
          else
            val dist = DiscreteDistribution(handWeights.toMap).normalized
            Right(ParseResult(dist, handWeights.keySet.toSet))

  /** Expands a single range token (without weight) into concrete hole-card combos.
    * Handles dash ranges (e.g., "A5s-A2s"), plus ranges (e.g., "QQ+"), and single specs (e.g., "AKs").
    */
  private def expandToken(token: String): Either[String, Vector[HoleCards]] =
    if token.contains("-") then
      val parts = token.split("-", -1).map(_.trim)
      if parts.length != 2 then Left("invalid range syntax")
      else
        for
          left <- parseBase(parts(0))
          right <- parseBase(parts(1))
          hands <- expandRange(left, right)
        yield hands
    else
      val (raw, plus) =
        if token.endsWith("+") then (token.dropRight(1), true) else (token, false)
      for
        base <- parseBase(raw)
        hands <- if plus then expandPlus(base) else expandSingle(base)
      yield hands

  /** Splits a token into its base range string and numeric weight.
    * Supports colon and at-sign separators (e.g., "AKs:0.5" or "AKs@50%").
    * Returns weight 1.0 if no weight is specified.
    */
  private def parseWeightedToken(token: String): Either[String, (String, Double)] =
    val idxColon = token.lastIndexOf(':')
    val idxAt = token.lastIndexOf('@')
    val idx =
      if idxColon < 0 && idxAt < 0 then -1
      else math.max(idxColon, idxAt)
    if idx < 0 then Right(token -> 1.0)
    else if idx == token.length - 1 then Left("missing weight")
    else
      val base = token.substring(0, idx).trim
      val weightStrRaw = token.substring(idx + 1).trim
      if base.isEmpty then Left("missing range before weight")
      else
        val (weightStr, isPercent) =
          if weightStrRaw.endsWith("%") then (weightStrRaw.dropRight(1).trim, true)
          else (weightStrRaw, false)
        if weightStr.isEmpty then Left("invalid weight value")
        else
          try
            val weightValue = weightStr.toDouble
            val weight = if isPercent then weightValue / 100.0 else weightValue
            if weight.isNaN || weight.isInfinite || weight < 0.0 then Left("invalid weight value")
            else Right(base -> weight)
          catch
            case _: NumberFormatException => Left("invalid weight value")

  /** Intermediate representation of a parsed hand token: two ranks and a suitedness qualifier.
    * For pairs, rank1 == rank2 and suitedness must be Any.
    */
  private final case class BaseSpec(rank1: Rank, rank2: Rank, suitedness: Suitedness)

  /** Parses a 2-3 character token into a BaseSpec (e.g., "AK" -> BaseSpec(Ace, King, Any),
    * "AKs" -> BaseSpec(Ace, King, Suited)). Validates rank characters and suitedness rules.
    */
  private def parseBase(token: String): Either[String, BaseSpec] =
    if token.length < 2 || token.length > 3 then Left("invalid hand token length")
    else
      val r1 = Rank.fromChar(token(0))
      val r2 = Rank.fromChar(token(1))
      if r1.isEmpty || r2.isEmpty then Left("invalid rank characters")
      else
        val suitedness =
          if token.length == 2 then Right(Suitedness.Any)
          else
            token(2).toLower match
              case 's' => Right(Suitedness.Suited)
              case 'o' => Right(Suitedness.Offsuit)
              case _ => Left("invalid suitedness specifier")
        suitedness.flatMap { s =>
          if r1.get == r2.get && s != Suitedness.Any then Left("pairs cannot specify suitedness")
          else Right(BaseSpec(r1.get, r2.get, s))
        }

  /** Expands a single BaseSpec into concrete combos: C(4,2)=6 for pairs, 4 for suited, 12 for offsuit, 16 for any. */
  private def expandSingle(base: BaseSpec): Either[String, Vector[HoleCards]] =
    val (high, low) = orderedRanks(base.rank1, base.rank2)
    if high == low then Right(pairCombos(high))
    else Right(nonPairCombos(high, low, base.suitedness))

  /** Expands a plus-range: for pairs, includes all pairs at or above the given rank;
    * for non-pairs, fixes the high card and includes all kickers from the base up to high-1.
    * E.g., "ATs+" -> ATs, AJs, AQs, AKs.
    */
  private def expandPlus(base: BaseSpec): Either[String, Vector[HoleCards]] =
    val (high, low) = orderedRanks(base.rank1, base.rank2)
    if high == low then
      val ranks = ranksDesc.filter(_.value >= high.value)
      Right(ranks.flatMap(pairCombos))
    else
      val maxLowValue = high.value - 1
      if low.value > maxLowValue then Left("invalid + range for non-pair")
      else
        val lows = (low.value to maxLowValue).map(rankByValue).toVector
        Right(lows.flatMap(l => nonPairCombos(high, l, base.suitedness)))

  /** Expands a dash-range (e.g., "99-66" or "A5s-A2s"). Both endpoints must have matching
    * suitedness. For pairs, iterates ranks between the two. For non-pairs, the high card
    * must match and the kicker varies between the two endpoints.
    */
  private def expandRange(left: BaseSpec, right: BaseSpec): Either[String, Vector[HoleCards]] =
    val (leftHigh, leftLow) = orderedRanks(left.rank1, left.rank2)
    val (rightHigh, rightLow) = orderedRanks(right.rank1, right.rank2)
    if left.suitedness != right.suitedness then Left("range suitedness must match")
    else if leftHigh == leftLow && rightHigh == rightLow then
      val ranks = ranksBetween(leftHigh, rightHigh)
      Right(ranks.flatMap(pairCombos))
    else if leftHigh == rightHigh && leftLow != leftHigh && rightLow != rightHigh then
      val lows = ranksBetween(leftLow, rightLow)
      Right(lows.flatMap(l => nonPairCombos(leftHigh, l, left.suitedness)))
    else
      Left("range endpoints are not compatible")

  /** Returns ranks in descending order (high, low). Used to normalize hand notation
    * where the higher-ranked card is conventionally listed first.
    */
  private def orderedRanks(a: Rank, b: Rank): (Rank, Rank) =
    if a.value >= b.value then (a, b) else (b, a)

  /** Returns all ranks between a and b (inclusive), in the correct direction. */
  private def ranksBetween(a: Rank, b: Rank): Vector[Rank] =
    val start = a.value
    val end = b.value
    val step = if start <= end then 1 else -1
    (start to end by step).map(rankByValue).toVector

  /** Generates all C(4,2) = 6 suit combinations for a pocket pair of the given rank. */
  private def pairCombos(rank: Rank): Vector[HoleCards] =
    val builder = Vector.newBuilder[HoleCards]
    var i = 0
    while i < suits.length - 1 do
      var j = i + 1
      while j < suits.length do
        builder += HoleCards.canonical(Card(rank, suits(i)), Card(rank, suits(j)))
        j += 1
      i += 1
    builder.result()

  /** Generates concrete combos for a non-pair hand: 4 suited, 12 offsuit, or 16 any.
    * Each combo is wrapped in HoleCards.canonical to ensure deterministic ordering.
    */
  private def nonPairCombos(high: Rank, low: Rank, suitedness: Suitedness): Vector[HoleCards] =
    val builder = Vector.newBuilder[HoleCards]
    suitedness match
      case Suitedness.Suited =>
        suits.foreach { suit =>
          builder += HoleCards.canonical(Card(high, suit), Card(low, suit))
        }
      case Suitedness.Offsuit =>
        suits.foreach { s1 =>
          suits.foreach { s2 =>
            if s1 != s2 then
              builder += HoleCards.canonical(Card(high, s1), Card(low, s2))
          }
        }
      case Suitedness.Any =>
        suits.foreach { s1 =>
          suits.foreach { s2 =>
            builder += HoleCards.canonical(Card(high, s1), Card(low, s2))
          }
        }
    builder.result()
