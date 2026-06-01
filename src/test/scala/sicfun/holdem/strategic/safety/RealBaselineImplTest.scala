package sicfun.holdem.strategic.safety

import munit.FunSuite
import sicfun.holdem.types.{Board, PokerAction, Position, Street}
import sicfun.holdem.strategic.types.*            // StrategicClass, Chips, PlayerId, TableMap, Seat, SeatStatus
import sicfun.holdem.strategic.state.PublicState
import sicfun.holdem.engine.StrategicEngine
import sicfun.core.Card

class RealBaselineImplTest extends FunSuite:
  private val Cat = PokerAction.Category
  private val floor = ConstantRealBaseline(StrategicEngine.defaultActionPriors)
  private def ps(street: Street, board: Board): PublicState =
    val hero = PlayerId("__test__")
    PublicState(street, board, Chips(10.0),
      TableMap(hero, Vector(Seat(hero, Position.SmallBlind, SeatStatus.Active, Chips(500.0)))),
      Vector.empty)
  private def flop(tokens: String*): PublicState =
    ps(Street.Flop, Board(tokens.toVector.map(t => Card.parse(t).get)))

  private val bucket = "Unpaired-Rainbow-AceHigh"
  private val meta = BaselineMetadata("1", "v1", "t", 1, 200, 30, 1.0, 0L)
  private val counts = Map[(StrategicClass, String, Street, PokerAction.Category), Long](
    (StrategicClass.Value, bucket, Street.Flop, Cat.Raise) -> 70L,
    (StrategicClass.Value, bucket, Street.Flop, Cat.Call)  -> 20L,
    (StrategicClass.Value, bucket, Street.Flop, Cat.Check) -> 8L,
    (StrategicClass.Value, bucket, Street.Flop, Cat.Fold)  -> 2L
  )
  private val impl = RealBaselineImpl(BaselineArtifact(counts, meta), minCount = 30, alpha = 1.0, fallback = floor)

  test("direct cell hit returns the Laplace-smoothed observed frequency"):
    // group total 100; alpha=1, 4 actions -> (70+1)/(100+4)=71/104, Fold (2+1)/104=3/104
    val p = flop("As","Kd","7c")
    assertEqualsDouble(impl.probability(StrategicClass.Value, Cat.Raise, None, p), 71.0/104.0, 1e-9)
    assertEqualsDouble(impl.probability(StrategicClass.Value, Cat.Fold, None, p), 3.0/104.0, 1e-9)

  test("a populated cell's distribution over the 4 actions sums to 1"):
    val p = flop("As","Kd","7c")
    val s: Double = Cat.values.toVector.map(a => impl.probability(StrategicClass.Value, a, None, p)).sum
    assertEqualsDouble(s, 1.0, 1e-9)

  test("backoff to (class,action) constants when the bucket+street group is below minCount"):
    val p = flop("As","Kd","7c")
    val obtained = impl.probability(StrategicClass.Bluff, Cat.Raise, None, p)
    val expected = floor.probability(StrategicClass.Bluff, Cat.Raise, None, p) // 0.65
    assertEqualsDouble(obtained, expected, 1e-12)

  test("preflop always backs off to constants (no preflop cells in v1)"):
    val pre = ps(Street.Preflop, Board.empty)
    val obtained = impl.probability(StrategicClass.Value, Cat.Raise, None, pre)
    val expected = floor.probability(StrategicClass.Value, Cat.Raise, None, pre) // 0.20
    assertEqualsDouble(obtained, expected, 1e-12)

  test("empty artifact behaves exactly like the constants floor"):
    val emptyImpl = RealBaselineImpl(BaselineArtifact(Map.empty, meta), 30, 1.0, floor)
    val p = flop("As","Kd","7c")
    for c <- StrategicClass.values; a <- Cat.values do
      assertEqualsDouble(emptyImpl.probability(c, a, None, p), floor.probability(c, a, None, p), 1e-12)
