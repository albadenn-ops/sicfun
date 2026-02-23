package sicfun.holdem

import sicfun.core.DiscreteDistribution

/** Table formats supported by the bunching engine.
  *
  * Each format defines the preflop action order: the sequence in which
  * positions act before the flop. This determines which positions could
  * have folded before a given opener, which is the basis for bunching
  * effect computation.
  *
  * NineMax uses the 8 distinct position categories from [[Position]],
  * which covers standard 8-max and 9-max full-ring games.
  */
enum TableFormat(val preflopOrder: Vector[Position]):
  case HeadsUp extends TableFormat(Vector(Position.Button, Position.BigBlind))
  case SixMax extends TableFormat(Vector(
    Position.UTG, Position.Middle, Position.Cutoff,
    Position.Button, Position.SmallBlind, Position.BigBlind
  ))
  case NineMax extends TableFormat(Vector(
    Position.UTG, Position.UTG1, Position.UTG2, Position.Middle,
    Position.Cutoff, Position.Button, Position.SmallBlind, Position.BigBlind
  ))

  /** Returns the positions that would have folded if the given position
    * is the first to open (raise first in). These are all positions that
    * act before `openerPos` in the preflop action order.
    *
    * @throws IllegalArgumentException if openerPos is not in this format's action order
    */
  def foldsBeforeOpener(openerPos: Position): Vector[Position] =
    val idx = preflopOrder.indexOf(openerPos)
    require(idx >= 0, s"$openerPos is not part of $this preflop order")
    preflopOrder.take(idx)

/** Configuration of per-position preflop opening ranges for bunching analysis.
  *
  * Stores both a normalized distribution (for equity calculations) and raw
  * open frequencies (for computing fold probabilities in the bunching sampler).
  *
  * @param format          the table format
  * @param ranges          position -> normalized opening range distribution
  * @param openFrequencies position -> (hand -> absolute open probability in [0,1])
  */
final case class TableRanges(
    format: TableFormat,
    ranges: Map[Position, DiscreteDistribution[HoleCards]],
    openFrequencies: Map[Position, Map[HoleCards, Double]]
):
  require(
    format.preflopOrder.forall(ranges.contains),
    "every position in the table format must have a defined range"
  )
  require(
    format.preflopOrder.forall(openFrequencies.contains),
    "every position must have defined open frequencies"
  )

  /** Returns the normalized opening range for the given position. */
  def rangeFor(pos: Position): DiscreteDistribution[HoleCards] = ranges(pos)

  /** Returns P(open | hand, position). Returns 0.0 for hands not in the range. */
  def openProbability(pos: Position, hand: HoleCards): Double =
    openFrequencies(pos).getOrElse(hand, 0.0)

  /** Returns P(fold | hand, position) = 1 - P(open | hand, position). */
  def foldProbability(pos: Position, hand: HoleCards): Double =
    1.0 - openProbability(pos, hand)

object TableRanges:

  /** Returns the default TableRanges for the given table format.
    *
    * Default ranges are typical tight-aggressive RFI (Raise First In) ranges.
    * Tightness increases from late position to early position.
    */
  def defaults(format: TableFormat): TableRanges =
    val strings = format match
      case TableFormat.HeadsUp => defaultRangeStringsHeadsUp
      case TableFormat.SixMax  => defaultRangeStrings6Max
      case TableFormat.NineMax => defaultRangeStrings9Max
    parseDefaults(format, strings)

  /** Creates custom TableRanges from range-string overrides.
    *
    * Positions not in `overrides` keep their default ranges.
    */
  def custom(
      format: TableFormat,
      overrides: Map[Position, String]
  ): Either[String, TableRanges] =
    val base = defaults(format)
    overrides.foldLeft(Right((base.ranges, base.openFrequencies)): Either[String, (Map[Position, DiscreteDistribution[HoleCards]], Map[Position, Map[HoleCards, Double]])]) {
      case (Left(err), _) => Left(err)
      case (Right((accRanges, accFreqs)), (pos, rangeStr)) =>
        RangeParser.parseWithHands(rangeStr) match
          case Left(err) => Left(s"range for $pos: $err")
          case Right(result) =>
            val freqs = result.hands.map(h => h -> 1.0).toMap
            Right((accRanges.updated(pos, result.distribution), accFreqs.updated(pos, freqs)))
    }.map { case (ranges, freqs) => TableRanges(format, ranges, freqs) }

  private def parseDefaults(
      format: TableFormat,
      strings: Map[Position, String]
  ): TableRanges =
    val ranges = scala.collection.mutable.Map.empty[Position, DiscreteDistribution[HoleCards]]
    val freqs = scala.collection.mutable.Map.empty[Position, Map[HoleCards, Double]]
    strings.foreach { case (pos, rangeStr) =>
      RangeParser.parseWithHands(rangeStr) match
        case Right(result) =>
          ranges(pos) = result.distribution
          freqs(pos) = result.hands.map(h => h -> 1.0).toMap
        case Left(err) =>
          throw new IllegalStateException(s"default range for $pos failed to parse: $err")
    }
    TableRanges(format, ranges.toMap, freqs.toMap)

  // -- Default RFI (Raise First In) range strings --

  private val defaultRangeStrings9Max: Map[Position, String] = Map(
    Position.UTG       -> "22+, A2s+, KTs+, QTs+, JTs, T9s, ATo+, KJo+",
    Position.UTG1      -> "22+, A2s+, K9s+, Q9s+, JTs, T9s, 98s, ATo+, KJo+, QJo",
    Position.UTG2      -> "22+, A2s+, K8s+, Q9s+, J9s+, T9s, 98s, 87s, ATo+, KTo+, QJo",
    Position.Middle    -> "22+, A2s+, K7s+, Q8s+, J8s+, T8s+, 98s, 87s, 76s, A9o+, KTo+, QTo+, JTo",
    Position.Cutoff    -> "22+, A2s+, K5s+, Q7s+, J7s+, T7s+, 97s+, 87s, 76s, 65s, A7o+, K9o+, Q9o+, J9o+, T9o",
    Position.Button    -> "22+, A2s+, K2s+, Q4s+, J6s+, T6s+, 96s+, 86s+, 76s, 65s, 54s, A2o+, K7o+, Q8o+, J8o+, T8o+, 98o",
    Position.SmallBlind -> "22+, A2s+, K6s+, Q8s+, J8s+, T8s+, 97s+, 87s, 76s, A5o+, K9o+, Q9o+, J9o+, T9o",
    Position.BigBlind  -> "22+, A2s+, K2s+, Q2s+, J2s+, T2s+, 92s+, 82s+, 72s+, 62s+, 52s+, 42s+, 32s, A2o+, K2o+, Q2o+, J2o+, T2o+, 92o+, 82o+, 72o+, 62o+, 52o+"
  )

  private val defaultRangeStrings6Max: Map[Position, String] = Map(
    Position.UTG       -> "22+, A2s+, K8s+, Q9s+, J9s+, T9s, 98s, ATo+, KJo+, QJo",
    Position.Middle    -> "22+, A2s+, K7s+, Q8s+, J8s+, T8s+, 98s, 87s, 76s, A9o+, KTo+, QTo+, JTo",
    Position.Cutoff    -> "22+, A2s+, K5s+, Q7s+, J7s+, T7s+, 97s+, 87s, 76s, 65s, A7o+, K9o+, Q9o+, J9o+, T9o",
    Position.Button    -> "22+, A2s+, K2s+, Q4s+, J6s+, T6s+, 96s+, 86s+, 76s, 65s, 54s, A2o+, K7o+, Q8o+, J8o+, T8o+, 98o",
    Position.SmallBlind -> "22+, A2s+, K6s+, Q8s+, J8s+, T8s+, 97s+, 87s, 76s, A5o+, K9o+, Q9o+, J9o+, T9o",
    Position.BigBlind  -> "22+, A2s+, K2s+, Q2s+, J2s+, T2s+, 92s+, 82s+, 72s+, 62s+, 52s+, 42s+, 32s, A2o+, K2o+, Q2o+, J2o+, T2o+, 92o+, 82o+, 72o+, 62o+, 52o+"
  )

  private val defaultRangeStringsHeadsUp: Map[Position, String] = Map(
    Position.Button  -> "22+, A2s+, K2s+, Q4s+, J6s+, T6s+, 96s+, 86s+, 76s, 65s, 54s, A2o+, K7o+, Q8o+, J8o+, T8o+, 98o",
    Position.BigBlind -> "22+, A2s+, K2s+, Q2s+, J2s+, T2s+, 92s+, 82s+, 72s+, 62s+, 52s+, 42s+, 32s, A2o+, K2o+, Q2o+, J2o+, T2o+, 92o+, 82o+, 72o+, 62o+, 52o+"
  )
