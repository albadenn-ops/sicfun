package sicfun.holdem.strategic

import munit.FunSuite
import sicfun.holdem.types.{Board, PokerAction, Position, Street}
import sicfun.holdem.strategic.types.*
import sicfun.holdem.strategic.state.PublicState
import sicfun.holdem.strategic.safety.{BaselineArtifact, BaselineArtifactIO, BaselineMetadata, ConstantRealBaseline, RealBaselineImpl}
import sicfun.holdem.engine.StrategicEngine
import sicfun.core.Card

class BaselineWiringTest extends FunSuite:
  private val Cat = PokerAction.Category
  private val priors = StrategicEngine.defaultActionPriors
  private def ps(street: Street, board: Board): PublicState =
    val hero = PlayerId("__test__")
    PublicState(street, board, Chips(10.0),
      TableMap(hero, Vector(Seat(hero, Position.SmallBlind, SeatStatus.Active, Chips(500.0)))),
      Vector.empty)
  private def flop(ts: String*): PublicState = ps(Street.Flop, Board(ts.toVector.map(t => Card.parse(t).get)))

  test("the calibrated Def 9 baseline is board-sensitive (the whole point)"):
    val bucketA = "Unpaired-Rainbow-AceHigh"
    val bucketB = "Unpaired-Monotone-Middle"
    val counts = Map[(StrategicClass, String, Street, PokerAction.Category), Long](
      (StrategicClass.Value, bucketA, Street.Flop, Cat.Raise) -> 95L,
      (StrategicClass.Value, bucketA, Street.Flop, Cat.Fold)  -> 5L,
      (StrategicClass.Value, bucketB, Street.Flop, Cat.Raise) -> 5L,
      (StrategicClass.Value, bucketB, Street.Flop, Cat.Fold)  -> 95L
    )
    val meta = BaselineMetadata("1","v1","t",1,200,30,1.0,0L)
    val rb = RealBaselineImpl(BaselineArtifact(counts, meta), 30, 1.0, ConstantRealBaseline(priors))
    val pRaiseA = rb.probability(StrategicClass.Value, Cat.Raise, None, flop("As","Kd","7c"))
    val pRaiseB = rb.probability(StrategicClass.Value, Cat.Raise, None, flop("9h","7h","5h"))
    assert(pRaiseA > 0.8 && pRaiseB < 0.2, s"board-sensitivity: A=$pRaiseA B=$pRaiseB")

  test("StrategicEngine constructs with no baseline (no-regression) and with a calibrated baseline"):
    val engineDefault = StrategicEngine(StrategicEngine.Config())
    assert(engineDefault != null)
    val dir = java.nio.file.Files.createTempDirectory("wiring-artifact")
    try
      BaselineArtifactIO.save(dir,
        BaselineArtifact(
          Map((StrategicClass.Value, "Unpaired-Rainbow-AceHigh", Street.Flop, Cat.Raise) -> 50L),
          BaselineMetadata("1","v1","t",1,50,30,1.0,0L)))
      val engineCalib = StrategicEngine(StrategicEngine.Config(baselinePath = Some(dir.toString)))
      assert(engineCalib != null)
    finally
      java.nio.file.Files.walk(dir).sorted(java.util.Comparator.reverseOrder()).forEach(p => java.nio.file.Files.deleteIfExists(p))
