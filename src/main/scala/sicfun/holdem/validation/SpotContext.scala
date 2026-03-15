package sicfun.holdem.validation

import sicfun.holdem.types.{Board, HoleCards, GameState, Position, Street, PokerAction}

/** Board texture classification for leak predicates.
  *
  * Captures whether a board has flush draw potential, straight draw potential,
  * paired cards, monotone suits, and adjacent connectedness.
  */
final case class BoardTexture(
    flushDrawPossible: Boolean,
    straightDrawPossible: Boolean,
    paired: Boolean,
    monotone: Boolean,
    connected: Boolean
):
  def isWet: Boolean = flushDrawPossible || straightDrawPossible || connected
  def isDry: Boolean = !isWet

object BoardTexture:
  def from(board: Board): BoardTexture =
    if board.cards.isEmpty then
      BoardTexture(
        flushDrawPossible = false,
        straightDrawPossible = false,
        paired = false,
        monotone = false,
        connected = false
      )
    else
      val cards = board.cards
      val suitCounts = cards.groupBy(_.suit).view.mapValues(_.size).toMap
      val maxSameSuit = suitCounts.values.max
      val flushDraw = maxSameSuit >= 3 || (maxSameSuit >= 2 && cards.size <= 4)
      val mono = cards.size >= 3 && maxSameSuit == cards.size

      val ranks = cards.map(_.rank.ordinal).sorted
      val uniqueRanks = ranks.distinct
      val isPaired = uniqueRanks.size < cards.size

      // straight draw: any N cards within a span of 5 ranks
      val straightDraw = hasConnectedness(uniqueRanks, if cards.size >= 4 then 4 else 3)
      val conn = adjacentConnected(uniqueRanks)

      BoardTexture(
        flushDrawPossible = flushDraw,
        straightDrawPossible = straightDraw,
        paired = isPaired,
        monotone = mono,
        connected = conn
      )

  private def hasConnectedness(sortedRanks: Vector[Int], needed: Int): Boolean =
    if sortedRanks.size < needed then false
    else sortedRanks.sliding(needed).exists(window => window.last - window.head <= 4)

  private def adjacentConnected(sortedRanks: Vector[Int]): Boolean =
    sortedRanks.sliding(2).exists:
      case Vector(a, b) => b - a <= 2
      case _ => false

/** Pot geometry metrics derived from a GameState snapshot.
  *
  * @param spr              stack-to-pot ratio
  * @param potOdds          toCall / (pot + toCall)
  * @param betToPotRatio    toCall / pot (how big the bet is relative to the pot)
  * @param effectiveStack   remaining stack
  */
final case class PotGeometry(
    spr: Double,
    potOdds: Double,
    betToPotRatio: Double,
    effectiveStack: Double
):
  def isBigPot: Boolean = spr < 2.0
  def isLargeBet: Boolean = betToPotRatio >= 0.7

object PotGeometry:
  def from(gs: GameState): PotGeometry =
    PotGeometry(
      spr = gs.stackToPot,
      potOdds = gs.potOdds,
      betToPotRatio = if gs.pot > 0.0 then gs.toCall / gs.pot else 0.0,
      effectiveStack = gs.stackSize
    )

/** Coarse hand strength classification relative to the board.
  *
  * Ordinal ordering: Nuts(0) > Strong(1) > Medium(2) > Weak(3) > Air(4).
  */
enum HandCategory:
  case Nuts, Strong, Medium, Weak, Air

object HandCategory:
  /** Classify hand strength from equity vs a random (uniform) range.
    *
    * Thresholds: >= 0.85 Nuts, >= 0.65 Strong, >= 0.45 Medium, >= 0.25 Weak, else Air.
    */
  def classify(hero: HoleCards, board: Board, equityVsRandom: Double): HandCategory =
    if equityVsRandom >= 0.85 then HandCategory.Nuts
    else if equityVsRandom >= 0.65 then HandCategory.Strong
    else if equityVsRandom >= 0.45 then HandCategory.Medium
    else if equityVsRandom >= 0.25 then HandCategory.Weak
    else HandCategory.Air

/** Describes whether a player's range is capped, uncapped, or polarized
  * based on the preflop/postflop action line taken.
  */
enum RangePosition:
  case Uncapped, Capped, Polarized

object RangePosition:
  /** Infer range position from the action line leading into the current street.
    *
    * - Check-raise line => Polarized
    * - Any raise(s) => Uncapped
    * - Flat calls / checks only => Capped
    */
  def fromLine(line: ActionLine, currentStreet: Street): RangePosition =
    val hasCheckRaise = line.actions.sliding(2).exists:
      case Vector(PokerAction.Check, PokerAction.Raise(_)) => true
      case _ => false
    if hasCheckRaise then RangePosition.Polarized
    else if line.containsRaise then RangePosition.Uncapped
    else RangePosition.Capped

/** A sequence of poker actions representing a player's action line in a hand. */
final case class ActionLine(actions: Vector[PokerAction]):
  def lastAction: Option[PokerAction] = actions.lastOption
  def containsRaise: Boolean = actions.exists(_.category == PokerAction.Category.Raise)
  def raiseCount: Int = actions.count(_.category == PokerAction.Category.Raise)

/** Rich game context assembled from GameState, hero hand, action line, and equity.
  *
  * Used by [[InjectedLeak]] predicates to decide whether a leak applies in a specific spot.
  */
final case class SpotContext(
    street: Street,
    board: Board,
    boardTexture: BoardTexture,
    potGeometry: PotGeometry,
    position: Position,
    facingAction: Option[PokerAction],
    facingSizing: Option[Double],
    lineRepresented: ActionLine,
    handStrengthVsBoard: HandCategory,
    rangeAdvantage: RangePosition
)

object SpotContext:
  /** Build a SpotContext from a GameState, hero hole cards, action line, and precomputed equity.
    *
    * @param gs             current game state snapshot
    * @param hero           hero's hole cards
    * @param line           action line taken so far
    * @param equityVsRandom hero's equity vs a uniform random range
    * @param facingAction   the action the hero is facing (if any)
    */
  def build(
      gs: GameState,
      hero: HoleCards,
      line: ActionLine,
      equityVsRandom: Double,
      facingAction: Option[PokerAction] = None
  ): SpotContext =
    val facingSizing = facingAction
      .collect { case PokerAction.Raise(amt) => amt }
      .map(amt => if gs.pot > 0 then amt / gs.pot else 0.0)
      .orElse(if gs.toCall > 0 && gs.pot > 0 then Some(gs.toCall / gs.pot) else None)

    SpotContext(
      street = gs.street,
      board = gs.board,
      boardTexture = BoardTexture.from(gs.board),
      potGeometry = PotGeometry.from(gs),
      position = gs.position,
      facingAction = facingAction,
      facingSizing = facingSizing,
      lineRepresented = line,
      handStrengthVsBoard = HandCategory.classify(hero, gs.board, equityVsRandom),
      rangeAdvantage = RangePosition.fromLine(line, gs.street)
    )
