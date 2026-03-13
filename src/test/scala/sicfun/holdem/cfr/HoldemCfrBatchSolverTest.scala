package sicfun.holdem.cfr

import munit.FunSuite
import sicfun.core.{Card, Deck, DiscreteDistribution}
import sicfun.holdem.types.*
import sicfun.holdem.equity.HoldemCombinator

class HoldemCfrBatchSolverTest extends FunSuite:

  private val GpuAvailable: Boolean =
    HoldemCfrNativeRuntime.availability(HoldemCfrNativeRuntime.Backend.Gpu).available

  private val StrategyTolerance = 0.02 // float vs double rounding

  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  private val PreflopState: GameState = GameState(
    street = Street.Preflop,
    board = Board.empty,
    pot = 6.0,
    toCall = 2.0,
    position = Position.Button,
    stackSize = 100.0,
    betHistory = Vector.empty
  )

  private def uniformPosteriorExcluding(hero: HoleCards): DiscreteDistribution[HoleCards] =
    val remaining = Deck.full.filterNot(c => c == hero.first || c == hero.second)
    DiscreteDistribution.uniform(HoldemCombinator.holeCardsFrom(remaining))

  override def munitTestTransforms: List[TestTransform] = super.munitTestTransforms ++
    List(new TestTransform("gpu-only", { test =>
      if GpuAvailable then test
      else test.tag(munit.Ignore)
    }))

  test("batch-of-1 matches single-tree CPU solve".tag(munit.Slow)) {
    val hero = hole("As", "Ks")
    val state = PreflopState
    val posterior = uniformPosteriorExcluding(hero)
    val actions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(6.0))
    val config = HoldemCfrConfig(iterations = 500, maxVillainHands = 48)

    val singleResult = HoldemCfrSolver.solveDecisionPolicy(
      hero = hero, state = state,
      villainPosterior = posterior,
      candidateActions = actions, config = config
    )

    val batchResults = HoldemCfrSolver.solveBatchDecisionPolicies(
      heroHands = IndexedSeq(hero), state = state,
      villainPosterior = posterior,
      candidateActions = actions, config = config
    )

    assertEquals(batchResults.length, 1)
    val (batchHero, batchPolicy) = batchResults.head
    assertEquals(batchHero, hero)
    assertEquals(batchPolicy.bestAction, singleResult.bestAction)

    actions.foreach { action =>
      val singleProb = singleResult.actionProbabilities.getOrElse(action, 0.0)
      val batchProb = batchPolicy.actionProbabilities.getOrElse(action, 0.0)
      assertEqualsDouble(batchProb, singleProb, StrategyTolerance,
        s"action $action: batch=$batchProb vs single=$singleProb")
    }
  }

  test("batch-of-10 matches sequential single-tree solves".tag(munit.Slow)) {
    val heroes = IndexedSeq(
      hole("As", "Ks"), hole("Ah", "Kh"),
      hole("Qs", "Qh"), hole("Jc", "Tc"),
      hole("9s", "8s"), hole("7h", "6h"),
      hole("5d", "4d"), hole("3c", "2c"),
      hole("Ac", "Qd"), hole("Kd", "Js")
    )
    val state = PreflopState
    val actions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(6.0))
    val config = HoldemCfrConfig(iterations = 500, maxVillainHands = 48)
    val allHoles = HoldemCombinator.holeCardsFrom(Deck.full)
    val posterior = DiscreteDistribution.uniform(allHoles)

    val batchResults = HoldemCfrSolver.solveBatchDecisionPolicies(
      heroHands = heroes, state = state,
      villainPosterior = posterior,
      candidateActions = actions, config = config
    )

    assertEquals(batchResults.length, heroes.length)

    heroes.zip(batchResults).foreach { case (hero, (batchHero, batchPolicy)) =>
      assertEquals(batchHero, hero)
      val singleResult = HoldemCfrSolver.solveDecisionPolicy(
        hero = hero, state = state,
        villainPosterior = uniformPosteriorExcluding(hero),
        candidateActions = actions, config = config
      )
      assertEquals(batchPolicy.bestAction, singleResult.bestAction,
        s"hero $hero: best action mismatch")
    }
  }
