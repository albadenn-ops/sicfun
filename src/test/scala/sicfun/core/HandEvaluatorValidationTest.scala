package sicfun.core

class HandEvaluatorValidationTest extends munit.FunSuite:
  test("evaluate5 category distribution matches canonical 5-card counts") {
    val check = HandEvaluatorValidation.checkStandardFiveCardCategoryDistribution()

    assertEquals(check.actual.total, HandEvaluatorValidation.TotalFiveCardHands)
    assertEquals(check.expected.total, HandEvaluatorValidation.TotalFiveCardHands)
    assert(
      check.isExactMatch,
      check.mismatches
        .map { case (category, actual, expected) =>
          s"${category.toString}: actual=$actual expected=$expected"
        }
        .mkString("category mismatches: ", ", ", "")
    )
  }
