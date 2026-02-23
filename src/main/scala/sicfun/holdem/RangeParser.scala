package sicfun.holdem

import sicfun.core.{Card, Rank, Suit, DiscreteDistribution}

object RangeParser:
  enum Suitedness:
    case Any, Suited, Offsuit

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

  private val rankByValue: Map[Int, Rank] = ranksDesc.map(r => r.value -> r).toMap
  private val suits: Vector[Suit] = Vector(Suit.Clubs, Suit.Diamonds, Suit.Hearts, Suit.Spades)

  final case class ParseResult(distribution: DiscreteDistribution[HoleCards], hands: Set[HoleCards])

  def parse(range: String): Either[String, DiscreteDistribution[HoleCards]] =
    parseWithHands(range).map(_.distribution)

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

  private final case class BaseSpec(rank1: Rank, rank2: Rank, suitedness: Suitedness)

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

  private def expandSingle(base: BaseSpec): Either[String, Vector[HoleCards]] =
    val (high, low) = orderedRanks(base.rank1, base.rank2)
    if high == low then Right(pairCombos(high))
    else Right(nonPairCombos(high, low, base.suitedness))

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

  private def orderedRanks(a: Rank, b: Rank): (Rank, Rank) =
    if a.value >= b.value then (a, b) else (b, a)

  private def ranksBetween(a: Rank, b: Rank): Vector[Rank] =
    val start = a.value
    val end = b.value
    val step = if start <= end then 1 else -1
    (start to end by step).map(rankByValue).toVector

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
