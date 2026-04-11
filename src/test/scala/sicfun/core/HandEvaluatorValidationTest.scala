package sicfun.core

/** Exhaustive correctness proof for [[HandEvaluator.categorize5]].
  *
  * Enumerates all C(52,5) = 2,598,960 possible five-card hands from a standard deck
  * and verifies that the category distribution exactly matches the known combinatorial
  * constants. This is the strongest possible validation: any miscount proves a bug.
  *
  * Note: this test evaluates ~2.6 million hands and takes a few seconds to run.
  */
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
