package sicfun.core

/** Represents the four standard playing card suits.
  *
  * Ordinal values: Clubs = 0, Diamonds = 1, Hearts = 2, Spades = 3.
  * The ordinal is used by [[CardId]] to compute a unique integer identifier for each card.
  */
enum Suit:
  case Clubs, Diamonds, Hearts, Spades

  /** Returns the canonical single-character representation (inverse of [[Suit.fromChar]]). */
  inline def toChar: Char = this match
    case Clubs    => 'c'
    case Diamonds => 'd'
    case Hearts   => 'h'
    case Spades   => 's'

/** Companion object providing parsing utilities for [[Suit]]. */
object Suit:
  /** Parses a single character into a [[Suit]].
    *
    * Accepts both upper- and lower-case letters: 'c' = Clubs, 'd' = Diamonds,
    * 'h' = Hearts, 's' = Spades.
    *
    * @param ch the character to parse
    * @return the corresponding suit, or `None` if the character is unrecognized
    */
  def fromChar(ch: Char): Option[Suit] =
    ch.toLower match
      case 'c' => Some(Clubs)
      case 'd' => Some(Diamonds)
      case 'h' => Some(Hearts)
      case 's' => Some(Spades)
      case _ => None

/** Represents standard playing card ranks from Two (2) through Ace (14).
  *
  * Each variant carries an integer `value` used for comparing hand strength.
  * The ordinal (0-based enum index) is used by [[CardId]] for the compact encoding:
  * Two = ordinal 0, Three = ordinal 1, ..., Ace = ordinal 12.
  *
  * @param value the numeric rank value (Two = 2, ..., Ace = 14)
  */
enum Rank(val value: Int):
  case Two extends Rank(2)
  case Three extends Rank(3)
  case Four extends Rank(4)
  case Five extends Rank(5)
  case Six extends Rank(6)
  case Seven extends Rank(7)
  case Eight extends Rank(8)
  case Nine extends Rank(9)
  case Ten extends Rank(10)
  case Jack extends Rank(11)
  case Queen extends Rank(12)
  case King extends Rank(13)
  case Ace extends Rank(14)

  /** Returns the canonical single-character representation (inverse of [[Rank.fromChar]]). */
  inline def toChar: Char = this match
    case Two   => '2'
    case Three => '3'
    case Four  => '4'
    case Five  => '5'
    case Six   => '6'
    case Seven => '7'
    case Eight => '8'
    case Nine  => '9'
    case Ten   => 'T'
    case Jack  => 'J'
    case Queen => 'Q'
    case King  => 'K'
    case Ace   => 'A'

/** Companion object providing ordering and parsing utilities for [[Rank]]. */
object Rank:
  /** Natural ordering of ranks by their numeric value (Two < ... < Ace). */
  given Ordering[Rank] = Ordering.by(_.value)

  /** Parses a single character into a [[Rank]].
    *
    * Accepts digits '2'-'9' and letters 't' (Ten), 'j' (Jack), 'q' (Queen),
    * 'k' (King), 'a' (Ace). Case-insensitive.
    *
    * @param ch the character to parse
    * @return the corresponding rank, or `None` if the character is unrecognized
    */
  def fromChar(ch: Char): Option[Rank] =
    ch.toLower match
      case '2' => Some(Two)
      case '3' => Some(Three)
      case '4' => Some(Four)
      case '5' => Some(Five)
      case '6' => Some(Six)
      case '7' => Some(Seven)
      case '8' => Some(Eight)
      case '9' => Some(Nine)
      case 't' => Some(Ten)
      case 'j' => Some(Jack)
      case 'q' => Some(Queen)
      case 'k' => Some(King)
      case 'a' => Some(Ace)
      case _ => None

/** An immutable playing card composed of a [[Rank]] and a [[Suit]].
  *
  * Cards can be parsed from two-character string tokens (e.g., "Ah" for Ace of Hearts)
  * via the companion object.
  *
  * @param rank the card's rank (Two through Ace)
  * @param suit the card's suit (Clubs, Diamonds, Hearts, or Spades)
  */
final case class Card(rank: Rank, suit: Suit):
  /** Returns the canonical two-character token (e.g., "Ah", "2c"). Inverse of [[Card.parse]]. */
  inline def toToken: String = s"${rank.toChar}${suit.toChar}"

/** Companion object providing parsing utilities for [[Card]]. */
object Card:
  /** Parses a two-character token into a [[Card]].
    *
    * The first character encodes the rank (see [[Rank.fromChar]]) and the second
    * encodes the suit (see [[Suit.fromChar]]). Example: `"Ah"` -> Ace of Hearts.
    *
    * @param token a two-character string such as "Ks" or "2c"
    * @return the parsed card, or `None` if the token is malformed
    */
  def parse(token: String): Option[Card] =
    if token.length != 2 then None
    else
      val rankOpt = Rank.fromChar(token(0))
      val suitOpt = Suit.fromChar(token(1))
      for
        rank <- rankOpt
        suit <- suitOpt
      yield Card(rank, suit)

  /** Parses a sequence of two-character tokens into a vector of cards.
    *
    * Returns `None` if any individual token fails to parse.
    *
    * @param tokens a sequence of two-character card strings
    * @return a vector of parsed cards, or `None` on any parse failure
    */
  def parseAll(tokens: Seq[String]): Option[Vector[Card]] =
    tokens.foldLeft(Option(Vector.newBuilder[Card])) { (acc, token) =>
      acc.flatMap(builder => parse(token).map(builder += _))
    }.map(_.result())

/** Provides a bijective mapping between [[Card]] instances and integer identifiers in the range [0, 51].
  *
  * '''Encoding scheme:''' `id = suit.ordinal * 13 + rank.ordinal`
  *
  * This yields the following layout:
  *  - Clubs:    ids  0 - 12  (Two of Clubs = 0, ..., Ace of Clubs = 12)
  *  - Diamonds: ids 13 - 25
  *  - Hearts:   ids 26 - 38
  *  - Spades:   ids 39 - 51
  *
  * The compact integer representation enables efficient bitwise operations,
  * array indexing, and set membership via bit masks in hand evaluation.
  */
object CardId:
  opaque type Id = Int

  object Id:
    private val MaxId = 51

    def fromInt(value: Int): Id =
      require(value >= 0 && value <= MaxId, s"invalid card id: $value")
      value

    private[core] inline def unsafe(value: Int): Id = value

  extension (inline id: Id)
    inline def toInt: Int = id

  private val rankValues = Rank.values
  private val suitValues = Suit.values
  private val rankCount = rankValues.length
  private val maxIdExclusive = rankCount * suitValues.length

  /** Converts a [[Card]] to its unique integer identifier.
    *
    * @param card the card to encode
    * @return an integer in [0, 51] computed as `suit.ordinal * 13 + rank.ordinal`
    */
  inline def toOpaque(card: Card): Id =
    Id.unsafe(card.suit.ordinal * rankCount + card.rank.ordinal)

  inline def toId(card: Card): Int =
    toOpaque(card).toInt

  /** Reconstructs a [[Card]] from its integer identifier.
    *
    * @param id an integer in [0, 51]
    * @return the corresponding card
    * @throws IllegalArgumentException if `id` is out of range
    */
  def fromOpaque(id: Id): Card =
    val raw = id.toInt
    val suit = suitValues(raw / rankCount)
    val rank = rankValues(raw % rankCount)
    Card(rank, suit)

  def fromId(id: Int): Card =
    require(id >= 0 && id < maxIdExclusive, s"invalid card id: $id")
    fromOpaque(Id.unsafe(id))

/** Provides the standard 52-card deck. */
object Deck:
  /** A complete 52-card deck ordered by suit (Clubs, Diamonds, Hearts, Spades)
    * then by rank (Two through Ace) within each suit.
    */
  val full: Vector[Card] =
    (for s <- Suit.values; r <- Rank.values yield Card(r, s)).toVector
