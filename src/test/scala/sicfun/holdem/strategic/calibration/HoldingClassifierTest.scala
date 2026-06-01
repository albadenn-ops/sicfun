package sicfun.holdem.strategic.calibration

import munit.FunSuite
import sicfun.core.Card
import sicfun.holdem.types.{Board, HoleCards, Street}
import sicfun.holdem.strategic.types.StrategicClass

class HoldingClassifierTest extends FunSuite:
  private def hc(a: String, b: String) = HoleCards.canonical(Card.parse(a).get, Card.parse(b).get)
  private def bd(ts: String*) = Board(ts.toVector.map(t => Card.parse(t).get))

  test("nut-strong made hand on a dry river → Value"):
    val cls = HoldingClassifier.classify(hc("As", "Ad"), bd("Ah","7d","2c","9s","3h"), Street.River)
    assertEquals(cls, StrategicClass.Value)

  test("trash hand on a dry river → Bluff"):
    val cls = HoldingClassifier.classify(hc("7c", "2d"), bd("Ah","Kd","Qc","9s","3h"), Street.River)
    assertEquals(cls, StrategicClass.Bluff)

  test("flush draw in the middle equity band on the flop → StructuralBluff"):
    val cls = HoldingClassifier.classify(hc("Th", "9h"), bd("Ah","7h","2c"), Street.Flop)
    assertEquals(cls, StrategicClass.StructuralBluff)

  test("hasDrawPotential: 4 to a flush on the flop is a draw; on the river it is not"):
    assert(HoldingClassifier.hasDrawPotential(hc("Th","9h"), bd("Ah","7h","2c"), Street.Flop))
    assert(!HoldingClassifier.hasDrawPotential(hc("Th","9h"), bd("Ah","7h","2c","Kd","3s"), Street.River))

  test("hasDrawPotential: open-ended straight draw on the flop"):
    assert(HoldingClassifier.hasDrawPotential(hc("9c","8d"), bd("7h","6s","2c"), Street.Flop))
