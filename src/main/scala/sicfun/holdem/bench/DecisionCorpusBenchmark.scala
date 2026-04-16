package sicfun.holdem.bench

import sicfun.core.{Card, Rank, Suit}
import sicfun.holdem.engine.*
import sicfun.holdem.engine.inference.VillainObservation
import sicfun.holdem.runtime.StrategicLifecycleHelper
import sicfun.holdem.types.*
import sicfun.holdem.equity.{PreflopFold, TableFormat, TableRanges}
import sicfun.holdem.model.PokerActionModel
import sicfun.holdem.strategic.types.PlayerId

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}

/** Fixed decision corpus for deterministic before/after regression checks.
  *
  * Each spot defines a game state, hero hole, candidates, and villain context.
  * The replay harness runs every spot through adaptive and strategic modes.
  */
object DecisionCorpusBenchmark:

  /** A single decision spot in the corpus. */
  final case class Spot(
      id: String,
      description: String,
      state: GameState,
      heroHole: HoleCards,
      candidates: Vector[PokerAction],
      villainPos: Position,
      observations: Vector[VillainObservation]
  )

  /** Result of replaying one spot through one mode. */
  final case class SpotResult(
      spotId: String,
      mode: String,
      selectedAction: PokerAction,
      upstreamAction: Option[PokerAction],
      perActionEvs: Vector[(PokerAction, Double)],
      overlayChanged: Boolean,
      softVetoCount: Int,
      adjustmentCount: Int,
      latencyMs: Double
  )

  // --- Helper to construct VillainObservation(action, GameState) concisely ---
  private def obs(action: PokerAction, street: Street, board: Board,
                  pot: Double, toCall: Double, position: Position, stack: Double): VillainObservation =
    VillainObservation(action, GameState(
      street = street, board = board, pot = pot, toCall = toCall,
      position = position, stackSize = stack, betHistory = Vector.empty
    ))

  // --- Boards used across spots ---
  private val boardA72r = Board(Vector(
    Card(Rank.Ace, Suit.Hearts), Card(Rank.Seven, Suit.Diamonds), Card(Rank.Two, Suit.Clubs)))
  private val boardA72r4s = Board(Vector(
    Card(Rank.Ace, Suit.Hearts), Card(Rank.Seven, Suit.Diamonds), Card(Rank.Two, Suit.Clubs),
    Card(Rank.Four, Suit.Spades)))
  private val boardK95r32 = Board(Vector(
    Card(Rank.King, Suit.Hearts), Card(Rank.Nine, Suit.Diamonds), Card(Rank.Five, Suit.Clubs),
    Card(Rank.Three, Suit.Spades), Card(Rank.Two, Suit.Hearts)))
  private val boardQQ838 = Board(Vector(
    Card(Rank.Queen, Suit.Hearts), Card(Rank.Queen, Suit.Diamonds), Card(Rank.Eight, Suit.Clubs),
    Card(Rank.Three, Suit.Spades), Card(Rank.Eight, Suit.Hearts)))

  /** The fixed corpus. These spots must not change across runs. */
  val corpus: Vector[Spot] = Vector(
    // 1. Preflop open: hero on button with AKs
    Spot(
      id = "preflop-open-aks",
      description = "Hero on button, AKs, no prior action",
      state = GameState(
        street = Street.Preflop, board = Board.empty, pot = 1.5, toCall = 0.5,
        position = Position.Button, stackSize = 100.0, betHistory = Vector.empty),
      heroHole = HoleCards(Card(Rank.Ace, Suit.Spades), Card(Rank.King, Suit.Spades)),
      candidates = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(2.5)),
      villainPos = Position.BigBlind,
      observations = Vector.empty
    ),
    // 2. Preflop facing 3bet: hero on button with QQ
    Spot(
      id = "preflop-3bet-qq",
      description = "Hero on button, QQ, facing 3bet from BB",
      state = GameState(
        street = Street.Preflop, board = Board.empty, pot = 7.5, toCall = 5.0,
        position = Position.Button, stackSize = 97.5, betHistory = Vector.empty),
      heroHole = HoleCards(Card(Rank.Queen, Suit.Hearts), Card(Rank.Queen, Suit.Diamonds)),
      candidates = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(15.0)),
      villainPos = Position.BigBlind,
      observations = Vector(
        obs(PokerAction.Raise(7.5), Street.Preflop, Board.empty,
            pot = 3.0, toCall = 2.5, position = Position.BigBlind, stack = 99.0)
      )
    ),
    // 3. Flop c-bet: hero on button, KK on A72r
    Spot(
      id = "flop-cbet-kk",
      description = "Hero on button, KK, flop A72 rainbow, first to act",
      state = GameState(
        street = Street.Flop, board = boardA72r, pot = 6.0, toCall = 0.0,
        position = Position.Button, stackSize = 97.0, betHistory = Vector.empty),
      heroHole = HoleCards(Card(Rank.King, Suit.Spades), Card(Rank.King, Suit.Clubs)),
      candidates = Vector(PokerAction.Check, PokerAction.Raise(3.0), PokerAction.Raise(4.5)),
      villainPos = Position.BigBlind,
      observations = Vector(
        obs(PokerAction.Call, Street.Preflop, Board.empty,
            pot = 3.0, toCall = 2.0, position = Position.BigBlind, stack = 99.0)
      )
    ),
    // 4. Turn barrel: hero on button, overpair on turn brick
    Spot(
      id = "turn-barrel-kk",
      description = "Hero on button, KK, turn 4s after flop cbet called",
      state = GameState(
        street = Street.Turn, board = boardA72r4s, pot = 12.0, toCall = 0.0,
        position = Position.Button, stackSize = 94.0, betHistory = Vector.empty),
      heroHole = HoleCards(Card(Rank.King, Suit.Spades), Card(Rank.King, Suit.Clubs)),
      candidates = Vector(PokerAction.Check, PokerAction.Raise(6.0), PokerAction.Raise(9.0)),
      villainPos = Position.BigBlind,
      observations = Vector(
        obs(PokerAction.Call, Street.Preflop, Board.empty,
            pot = 3.0, toCall = 2.0, position = Position.BigBlind, stack = 99.0),
        obs(PokerAction.Call, Street.Flop, boardA72r,
            pot = 9.0, toCall = 3.0, position = Position.BigBlind, stack = 96.0)
      )
    ),
    // 5. River bluff-catch: hero on BB, A-high facing large bet
    Spot(
      id = "river-bluffcatch-ahigh",
      description = "Hero on BB, Ace-high on river, facing pot-sized bet",
      state = GameState(
        street = Street.River, board = boardK95r32, pot = 24.0, toCall = 24.0,
        position = Position.BigBlind, stackSize = 76.0, betHistory = Vector.empty),
      heroHole = HoleCards(Card(Rank.Ace, Suit.Diamonds), Card(Rank.Jack, Suit.Clubs)),
      candidates = Vector(PokerAction.Fold, PokerAction.Call),
      villainPos = Position.Button,
      observations = Vector(
        obs(PokerAction.Raise(2.5), Street.Preflop, Board.empty,
            pot = 1.5, toCall = 0.5, position = Position.Button, stack = 100.0),
        obs(PokerAction.Raise(4.0), Street.Flop,
            Board(Vector(Card(Rank.King, Suit.Hearts), Card(Rank.Nine, Suit.Diamonds), Card(Rank.Five, Suit.Clubs))),
            pot = 5.0, toCall = 0.0, position = Position.Button, stack = 97.5),
        obs(PokerAction.Raise(8.0), Street.Turn,
            Board(Vector(Card(Rank.King, Suit.Hearts), Card(Rank.Nine, Suit.Diamonds),
              Card(Rank.Five, Suit.Clubs), Card(Rank.Three, Suit.Spades))),
            pot = 13.0, toCall = 0.0, position = Position.Button, stack = 93.5),
        obs(PokerAction.Raise(24.0), Street.River, boardK95r32,
            pot = 29.0, toCall = 0.0, position = Position.Button, stack = 85.5)
      )
    ),
    // 6. River value bet: hero on button, full house on paired board
    Spot(
      id = "river-valuebet-fullhouse",
      description = "Hero on button, full house on river, checking to hero",
      state = GameState(
        street = Street.River, board = boardQQ838, pot = 20.0, toCall = 0.0,
        position = Position.Button, stackSize = 90.0, betHistory = Vector.empty),
      heroHole = HoleCards(Card(Rank.Queen, Suit.Spades), Card(Rank.Eight, Suit.Spades)),
      candidates = Vector(PokerAction.Check, PokerAction.Raise(10.0), PokerAction.Raise(20.0)),
      villainPos = Position.BigBlind,
      observations = Vector(
        obs(PokerAction.Call, Street.Preflop, Board.empty,
            pot = 3.0, toCall = 2.0, position = Position.BigBlind, stack = 99.0),
        obs(PokerAction.Check, Street.Flop,
            Board(Vector(Card(Rank.Queen, Suit.Hearts), Card(Rank.Queen, Suit.Diamonds), Card(Rank.Eight, Suit.Clubs))),
            pot = 6.0, toCall = 0.0, position = Position.BigBlind, stack = 97.0),
        obs(PokerAction.Check, Street.Turn,
            Board(Vector(Card(Rank.Queen, Suit.Hearts), Card(Rank.Queen, Suit.Diamonds),
              Card(Rank.Eight, Suit.Clubs), Card(Rank.Three, Suit.Spades))),
            pot = 6.0, toCall = 0.0, position = Position.BigBlind, stack = 97.0),
        obs(PokerAction.Check, Street.River, boardQQ838,
            pot = 20.0, toCall = 0.0, position = Position.BigBlind, stack = 90.0)
      )
    )
  )

  /** Replay all corpus spots through adaptive and strategic modes.
    * Returns a flat vector of per-spot-per-mode results.
    */
  def replayAll(
      seed: Long = 42L,
      equityTrials: Int = 600,
      bunchingTrials: Int = 1
  ): Vector[SpotResult] =
    val tableRanges = TableRanges.defaults(TableFormat.HeadsUp)
    val folds = TableFormat.HeadsUp.foldsBeforeOpener(Position.Button).map(PreflopFold(_))
    val model = PokerActionModel.uniform

    // Create shared engines
    val preflopEngine = HeroDecisionPipeline.newAdaptiveEngine(
      tableRanges = tableRanges,
      model = model,
      bunchingTrials = bunchingTrials,
      equityTrials = equityTrials
    )
    val strategicHelper = StrategicLifecycleHelper.create()
    strategicHelper.initSession(
      rivalIds = Vector(PlayerId("villain")),
      positionMapping = Map.empty
    )

    val results = Vector.newBuilder[SpotResult]
    corpus.foreach { spot =>
      val spotSeed = seed ^ spot.id.hashCode.toLong

      // Replay through adaptive mode
      val adaptiveCtx = HeroDecisionPipeline.HeroDecisionContext(
        hero = spot.heroHole,
        state = spot.state,
        folds = folds,
        tableRanges = tableRanges,
        villainPos = spot.villainPos,
        observations = spot.observations,
        candidates = spot.candidates,
        engine = preflopEngine,
        actionModel = model,
        bunchingTrials = bunchingTrials,
        cfrIterations = 100,
        cfrVillainHands = 50,
        cfrEquityTrials = equityTrials,
        rng = new scala.util.Random(spotSeed)
      )
      val adaptiveStart = System.nanoTime()
      val adaptiveAction = HeroDecisionPipeline.decideHero(HeroMode.Adaptive, adaptiveCtx)
      val adaptiveElapsed = System.nanoTime() - adaptiveStart
      results += SpotResult(
        spotId = spot.id,
        mode = "adaptive",
        selectedAction = adaptiveAction,
        upstreamAction = None,
        perActionEvs = Vector.empty,
        overlayChanged = false,
        softVetoCount = 0,
        adjustmentCount = 0,
        latencyMs = adaptiveElapsed / 1e6
      )

      // Replay through strategic mode
      strategicHelper.updatePositionMapping(Map(spot.villainPos -> PlayerId("villain")))
      strategicHelper.startHand(spot.heroHole)
      spot.observations.foreach { obs =>
        strategicHelper.observeVillainAction(spot.villainPos, obs.action, obs.state)
      }
      val strategicCtx = HeroDecisionPipeline.StrategicDecisionContext(
        state = spot.state,
        candidates = spot.candidates,
        helper = strategicHelper
      )
      val heroCtx = HeroDecisionPipeline.HeroDecisionContext(
        hero = spot.heroHole,
        state = spot.state,
        folds = folds,
        tableRanges = tableRanges,
        villainPos = spot.villainPos,
        observations = spot.observations,
        candidates = spot.candidates,
        engine = preflopEngine,
        actionModel = model,
        bunchingTrials = bunchingTrials,
        cfrIterations = 100,
        cfrVillainHands = 50,
        cfrEquityTrials = equityTrials,
        rng = new scala.util.Random(spotSeed + 1L)
      )
      val strategicResult = HeroDecisionPipeline.decideHeroStrategic(strategicCtx, heroCtx)
      strategicHelper.endHand()
      results += SpotResult(
        spotId = spot.id,
        mode = "strategic",
        selectedAction = strategicResult.action,
        upstreamAction = Some(strategicResult.overlayResult.upstreamAction),
        perActionEvs = strategicResult.overlayResult.rankedActions.map(ae => (ae.action, ae.expectedValue)),
        overlayChanged = strategicResult.action != strategicResult.overlayResult.upstreamAction,
        softVetoCount = strategicResult.overlayResult.softVetoed.size,
        adjustmentCount = strategicResult.overlayResult.adjustments.size,
        latencyMs = strategicResult.totalLatencyNanos / 1e6
      )
    }
    results.result()

  /** Write corpus results to a TSV file for comparison. */
  def writeResults(results: Vector[SpotResult], path: Path): Unit =
    Files.createDirectories(path.getParent)
    val header = "spotId\tmode\tselectedAction\tupstreamAction\tperActionEvs\toverlayChanged\tsoftVetoCount\tadjustmentCount\tlatencyMs"
    val rows = results.map { r =>
      val evsStr = if r.perActionEvs.isEmpty then "-"
        else r.perActionEvs.map((a, ev) => f"${PokerFormatting.renderAction(a)}=${ev}%.4f").mkString(",")
      Vector(
        r.spotId,
        r.mode,
        PokerFormatting.renderAction(r.selectedAction),
        r.upstreamAction.map(PokerFormatting.renderAction).getOrElse("-"),
        evsStr,
        r.overlayChanged.toString,
        r.softVetoCount.toString,
        r.adjustmentCount.toString,
        f"${r.latencyMs}%.3f"
      ).mkString("\t")
    }
    Files.write(path, (header +: rows).mkString(System.lineSeparator()).getBytes(StandardCharsets.UTF_8))

  /** Main entry point for standalone benchmark runs. */
  def main(args: Array[String]): Unit =
    val seed = args.headOption.flatMap(_.toLongOption).getOrElse(42L)
    val outDir = Path.of(if args.length > 1 then args(1) else "data/bench-decision-corpus")
    println(s"Running decision corpus benchmark (seed=$seed)")
    val results = replayAll(seed = seed)
    val outPath = outDir.resolve("corpus-results.tsv")
    writeResults(results, outPath)
    println(s"Wrote ${results.size} results to $outPath")
    // Print summary
    val strategic = results.filter(_.mode == "strategic")
    val changes = strategic.count(_.overlayChanged)
    val vetoes = strategic.map(_.softVetoCount).sum
    println(f"Strategic: ${strategic.size} spots, $changes overlay changes, $vetoes soft vetoes")
    println(f"Mean strategic latency: ${strategic.map(_.latencyMs).sum / strategic.size}%.3f ms")
