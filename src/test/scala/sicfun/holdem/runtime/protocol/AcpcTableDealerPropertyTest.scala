package sicfun.holdem.runtime.protocol

import org.scalacheck.Gen
import org.scalacheck.Prop.*
import sicfun.holdem.types.{PokerAction, Street}

class AcpcTableDealerPropertyTest extends munit.ScalaCheckSuite:

  val nSeatsGen: Gen[Int] = Gen.choose(2, 9)
  val stackGen: Gen[Long] = Gen.choose(20L, 500L)

  property("side pots: sum of distributed chips == sum of contributions"):
    forAll(nSeatsGen, Gen.listOfN(9, stackGen)) { (n, stacks) =>
      val cfg = TableConfig(n, 1L, 2L, 0L, 200L)
      val d = AcpcTableDealer(cfg, SeatId(0), 0L)
      (0 until n).foreach(i => d.setStackForTest(SeatId(i), stacks(i)))
      d.postBlinds()
      d.dealHoleCards()
      d.startStreet(Street.Preflop)
      var safety = 0
      while d.nextToAct.isDefined && safety < 50 do
        val seat = d.nextToAct.get
        val shove = d.currentStacks.getOrElse(seat, 0L) +
          d.currentContributions.getOrElse(seat, 0L)
        if shove > d.currentBetForTest then
          d.applyAction(seat, PokerAction.Raise(shove.toDouble))
        else
          d.applyAction(seat, PokerAction.Fold)
        safety += 1
      // Complete the board only if there are still 2+ contenders;
      // a single non-folded survivor would invalidate evaluateShowdown's
      // requirement of a non-empty contender set, so for that case we rely
      // on side pots having a single eligible seat (the survivor) and skip
      // showdown entirely by handing the survivor a synthetic rank.
      val nonFolded = (0 until n).map(SeatId(_))
        .filterNot(s => d.foldedSetForTest.contains(s))
        .toVector
      val ranks: Map[SeatId, sicfun.core.HandRank] =
        if nonFolded.size <= 1 then
          nonFolded.headOption.map(s =>
            s -> sicfun.core.HandEvaluator
              .evaluate7(sicfun.core.Deck.full.takeRight(7).toVector)
          ).toMap
        else
          d.dealCommunity(Street.Flop)
          d.dealCommunity(Street.Turn)
          d.dealCommunity(Street.River)
          d.evaluateShowdown()
      val distributed = d.distributePots(ranks)
      val totalDistributed = distributed.map(_._2.values.sum).sum
      val totalContributed = d.currentContributions.values.sum
      totalDistributed == totalContributed
    }

  property("finalizeHand: outcome.netChange sums to zero exactly"):
    forAll(nSeatsGen) { n =>
      val cfg = TableConfig(n, 1L, 2L, 0L, 200L)
      val d = AcpcTableDealer(cfg, SeatId(0), 123L)
      d.postBlinds()
      d.dealHoleCards()
      d.startStreet(Street.Preflop)
      var safety = 0
      while d.nextToAct.isDefined && safety < 50 do
        d.applyAction(d.nextToAct.get, PokerAction.Fold)
        safety += 1
      val outcome = d.finalizeHand()
      outcome.netChange.values.sum == 0L
    }
