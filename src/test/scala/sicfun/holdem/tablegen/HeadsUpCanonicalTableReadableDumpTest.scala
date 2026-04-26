package sicfun.holdem.tablegen

import munit.FunSuite
import sicfun.core.Card
import sicfun.holdem.types.HoleCards

/** Pinning tests for [[HeadsUpCanonicalTableReadableDump]] private[tablegen] helpers.
  *
  * The dump CLI reads a binary canonical table, decodes each entry into human-readable
  * tokens, sorts, and writes a TSV. This file pins the two pure helpers it depends on:
  *
  *   - handClass: HoleCards -> "AA" / "AKs" / "AKo" hand-class token
  *   - sortRows: multi-key sort over decoded rows with asc/desc + invalid-key rejection
  *
  * Audit F1 listed `tablegen/` as zero-tests; this file is the first sicfun-side
  * test for that package. Companion equity-side coverage already pins canonical-key
  * invariants and binary IO roundtrips.
  */
class HeadsUpCanonicalTableReadableDumpTest extends FunSuite:

  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  // ---- handClass: poker hand-class tokens ----

  test("handClass returns 2-char pair token (AA, KK, ... 22)") {
    assertEquals(HeadsUpCanonicalTableReadableDump.handClass(hole("As", "Ah")), "AA")
    assertEquals(HeadsUpCanonicalTableReadableDump.handClass(hole("Kh", "Kd")), "KK")
    assertEquals(HeadsUpCanonicalTableReadableDump.handClass(hole("2s", "2c")), "22")
    assertEquals(HeadsUpCanonicalTableReadableDump.handClass(hole("Tc", "Td")), "TT")
  }

  test("handClass returns 3-char suited token AKs / AKs / 76s") {
    assertEquals(HeadsUpCanonicalTableReadableDump.handClass(hole("As", "Ks")), "AKs")
    assertEquals(HeadsUpCanonicalTableReadableDump.handClass(hole("Kh", "Ah")), "AKs",
      "high rank should come first regardless of input order")
    assertEquals(HeadsUpCanonicalTableReadableDump.handClass(hole("7c", "6c")), "76s")
  }

  test("handClass returns 3-char offsuit token AKo / 27o") {
    assertEquals(HeadsUpCanonicalTableReadableDump.handClass(hole("As", "Kh")), "AKo")
    assertEquals(HeadsUpCanonicalTableReadableDump.handClass(hole("2c", "7d")), "72o",
      "high rank precedes low; 7 > 2")
  }

  test("handClass uses card.toChar for each rank (T not 10)") {
    assertEquals(HeadsUpCanonicalTableReadableDump.handClass(hole("Ts", "9s")), "T9s")
    assertEquals(HeadsUpCanonicalTableReadableDump.handClass(hole("Th", "Td")), "TT")
  }

  // ---- sortRows: stable multi-key sort ----

  private def row(
      key: Int = 0, equity: Double = 0.5, win: Double = 0.5, tie: Double = 0.0,
      loss: Double = 0.5, stderr: Double = 0.0,
      hero: String = "AsKs", villain: String = "QcJc",
      heroClass: String = "AKs", villainClass: String = "QJs"
  ): HeadsUpCanonicalTableReadableDump.Row =
    HeadsUpCanonicalTableReadableDump.Row(
      key = key, hero = hero, villain = villain,
      heroClass = heroClass, villainClass = villainClass,
      win = win, tie = tie, loss = loss, equity = equity, stderr = stderr
    )

  private val sample: Vector[HeadsUpCanonicalTableReadableDump.Row] = Vector(
    row(key = 30, equity = 0.7, win = 0.6),
    row(key = 10, equity = 0.5, win = 0.5),
    row(key = 20, equity = 0.6, win = 0.4)
  )

  test("sortRows by key asc returns rows in ascending key order") {
    val sorted = HeadsUpCanonicalTableReadableDump.sortRows(sample, "key", "asc")
    assertEquals(sorted.map(_.key), Vector(10, 20, 30))
  }

  test("sortRows by key desc returns rows in descending key order") {
    val sorted = HeadsUpCanonicalTableReadableDump.sortRows(sample, "key", "desc")
    assertEquals(sorted.map(_.key), Vector(30, 20, 10))
  }

  test("sortRows accepts every documented numeric column") {
    Vector("key", "win", "tie", "loss", "equity", "stderr").foreach { col =>
      val sorted = HeadsUpCanonicalTableReadableDump.sortRows(sample, col, "asc")
      assertEquals(sorted.size, sample.size, s"column=$col")
    }
  }

  test("sortRows accepts hero / villain string columns") {
    val mixed = Vector(
      row(hero = "B"), row(hero = "A"), row(hero = "C")
    )
    val byHero = HeadsUpCanonicalTableReadableDump.sortRows(mixed, "hero", "asc")
    assertEquals(byHero.map(_.hero), Vector("A", "B", "C"))
    val byVillainAsc = HeadsUpCanonicalTableReadableDump.sortRows(
      Vector(row(villain = "Y"), row(villain = "X"), row(villain = "Z")),
      "villain", "asc"
    )
    assertEquals(byVillainAsc.map(_.villain), Vector("X", "Y", "Z"))
  }

  test("sortRows rejects unknown sort column") {
    intercept[IllegalArgumentException] {
      HeadsUpCanonicalTableReadableDump.sortRows(sample, "bogus", "asc")
    }
  }

  test("sortRows rejects unknown order") {
    intercept[IllegalArgumentException] {
      HeadsUpCanonicalTableReadableDump.sortRows(sample, "key", "sideways")
    }
  }

  test("sortRows on empty Vector returns empty Vector for any valid column/order") {
    val empty = Vector.empty[HeadsUpCanonicalTableReadableDump.Row]
    assertEquals(HeadsUpCanonicalTableReadableDump.sortRows(empty, "key", "asc"), empty)
    assertEquals(HeadsUpCanonicalTableReadableDump.sortRows(empty, "equity", "desc"), empty)
  }
