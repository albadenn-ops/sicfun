package sicfun.holdem.strategic.calibration

import sicfun.core.{Card, Deck, DiscreteDistribution}
import sicfun.holdem.types.{Board, HoleCards, Street}
import sicfun.holdem.equity.HoldemEquity
import sicfun.holdem.strategic.types.StrategicClass
import sicfun.holdem.strategic.bridge.ClassificationBridge

/** Classifies a SHOWDOWN-revealed holding into the engine's StrategicClass partition,
  * reusing ClassificationBridge.classify over an exact equity estimate. Offline use only.
  */
object HoldingClassifier:

  /** Uniform distribution over all 1326 two-card combos. `equityExact` sanitizes blocked
    * cards per call, so this constant range is reusable across holdings/boards. */
  private val uniformVillainRange: DiscreteDistribution[HoleCards] =
    val cards = Deck.full.toVector
    val combos =
      for i <- cards.indices; j <- (i + 1) until cards.length
      yield HoleCards.canonical(cards(i), cards(j))
    val w = 1.0 / combos.length
    DiscreteDistribution(combos.map(h => h -> w).toMap)

  def classify(holding: HoleCards, board: Board, street: Street): StrategicClass =
    val equity = HoldemEquity.equityExact(holding, board, uniformVillainRange).equity
    val draw = hasDrawPotential(holding, board, street)
    ClassificationBridge.classify(equity, draw).fold(
      onExact = identity,
      onApprox = (cls, _) => cls,
      onAbsent = reason => throw new IllegalStateException(s"classification returned Absent: $reason")
    )

  /** Coarse postflop draw detector: a flush draw (exactly 4 to a suit across hole+board)
    * or an open-ended straight draw (a run of ≥4 consecutive distinct ranks). Never true
    * on the river. */
  def hasDrawPotential(holding: HoleCards, board: Board, street: Street): Boolean =
    if street == Street.River then false
    else
      val cards = holding.toVector ++ board.cards
      val flushDraw = cards.groupBy(_.suit).values.exists(_.size == 4)
      flushDraw || hasOpenEndedStraightDraw(cards)

  private def hasOpenEndedStraightDraw(cards: Vector[Card]): Boolean =
    val ranks = cards.map(_.rank.value).distinct.sorted
    if ranks.length < 4 then false
    else
      var best = 1; var run = 1; var i = 1
      while i < ranks.length do
        if ranks(i) == ranks(i - 1) + 1 then run += 1 else run = 1
        if run > best then best = run
        i += 1
      best >= 4
