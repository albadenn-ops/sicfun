package sicfun.holdem.engine

import munit.FunSuite
import sicfun.core.{Card, DiscreteDistribution}
import sicfun.holdem.history.ShowdownRecord
import sicfun.holdem.types.*

class ShowdownPriorBiasTest extends FunSuite:
  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  private val aaHeartSpade = hole("Ah", "As")
  private val aaClubDiamond = hole("Ac", "Ad")
  private val kkHeartSpade = hole("Kh", "Ks")
  private val trash = hole("7c", "2d")

  private val prior = DiscreteDistribution(
    Map(
      aaHeartSpade -> 0.25,
      aaClubDiamond -> 0.25,
      kkHeartSpade -> 0.25,
      trash -> 0.25
    )
  )

  private val premiumShowdowns = Vector(
    ShowdownRecord("sd-1", aaHeartSpade),
    ShowdownRecord("sd-2", aaClubDiamond),
    ShowdownRecord("sd-3", kkHeartSpade),
    ShowdownRecord("sd-4", hole("Qh", "Qs")),
    ShowdownRecord("sd-5", hole("Jh", "Js"))
  )

  test("applyBias increases premium density while preserving normalization") {
    val biased = ShowdownPriorBias.applyBias(prior, premiumShowdowns)

    assert(math.abs(biased.weights.values.sum - 1.0) < 1e-9, s"biased prior must normalize: ${biased.weights}")
    assert(
      biased.probabilityOf(aaHeartSpade) > prior.probabilityOf(aaHeartSpade),
      s"expected exact showdown combo to gain weight: before=${prior.probabilityOf(aaHeartSpade)} after=${biased.probabilityOf(aaHeartSpade)}"
    )
    assert(
      biased.probabilityOf(trash) < prior.probabilityOf(trash),
      s"expected weak offsuit trash to lose weight: before=${prior.probabilityOf(trash)} after=${biased.probabilityOf(trash)}"
    )
  }

  test("applyBias returns original prior for insufficient showdown samples") {
    assertEquals(ShowdownPriorBias.applyBias(prior, premiumShowdowns.take(2)), prior)
  }

  test("applyBias filters dead-card combos before normalizing") {
    val filtered = ShowdownPriorBias.applyBias(prior, Vector.empty, deadCards = Set(card("Ah")))

    assertEquals(filtered.probabilityOf(aaHeartSpade), 0.0)
    assert(math.abs(filtered.weights.values.sum - 1.0) < 1e-9, s"filtered prior must normalize: ${filtered.weights}")
  }

  test("blendWeight follows the configured linear cap") {
    assertEqualsDouble(ShowdownPriorBias.blendWeight(3), 0.06, 1e-12)
    assertEqualsDouble(ShowdownPriorBias.blendWeight(10), 0.15, 1e-12)
    assertEqualsDouble(ShowdownPriorBias.blendWeight(50), 0.15, 1e-12)
  }
