package sicfun.holdem.runtime

import munit.FunSuite
import sicfun.holdem.types.Position

class HallConfigTest extends FunSuite:

  // --- modeledPositionsForPlayerCount: per-table-size invariants ---

  test("modeledPositionsForPlayerCount returns BB at index 0 in heads-up only and Button-first ordering") {
    val hu = HallConfig.modeledPositionsForPlayerCount(2)
    assertEquals(hu, Vector(Position.Button, Position.BigBlind))
  }

  test("modeledPositionsForPlayerCount lengths match the table size for 2..9") {
    (2 to 9).foreach { n =>
      val positions = HallConfig.modeledPositionsForPlayerCount(n)
      assertEquals(positions.length, n, s"$n-max should produce $n positions")
      assertEquals(positions.distinct.length, n, s"$n-max positions must be unique")
    }
  }

  test("modeledPositionsForPlayerCount always ends with the blinds in postflop order (SB, BB)") {
    (3 to 9).foreach { n =>
      val positions = HallConfig.modeledPositionsForPlayerCount(n)
      assertEquals(
        positions.takeRight(2),
        Vector(Position.SmallBlind, Position.BigBlind),
        s"$n-max should end with SB, BB"
      )
    }
  }

  test("modeledPositionsForPlayerCount returns empty Vector outside [2..9]") {
    assertEquals(HallConfig.modeledPositionsForPlayerCount(0), Vector.empty)
    assertEquals(HallConfig.modeledPositionsForPlayerCount(1), Vector.empty)
    assertEquals(HallConfig.modeledPositionsForPlayerCount(10), Vector.empty)
  }

  // --- parseLegacyHeroSeat ---

  test("parseLegacyHeroSeat recognises button + bigblind/bb aliases") {
    assertEquals(HallConfig.parseLegacyHeroSeat("button"), Right(Position.Button))
    assertEquals(HallConfig.parseLegacyHeroSeat("BUTTON"), Right(Position.Button))
    assertEquals(HallConfig.parseLegacyHeroSeat(" Button "), Right(Position.Button))
    assertEquals(HallConfig.parseLegacyHeroSeat("bigblind"), Right(Position.BigBlind))
    assertEquals(HallConfig.parseLegacyHeroSeat("bb"), Right(Position.BigBlind))
    assertEquals(HallConfig.parseLegacyHeroSeat("BB"), Right(Position.BigBlind))
  }

  test("parseLegacyHeroSeat rejects unknown tokens with a helpful error") {
    val result = HallConfig.parseLegacyHeroSeat("smallblind")
    assert(result.isLeft, s"expected Left, got $result")
    assert(result.left.toOption.exists(_.contains("button, bigblind")))
  }

  // --- typed-option helpers ---

  test("intOpt returns default when key absent, parses when present, errors on garbage") {
    assertEquals(HallConfig.intOpt(Map.empty, "n", 7), Right(7))
    assertEquals(HallConfig.intOpt(Map("n" -> "12"), "n", 7), Right(12))
    val bad = HallConfig.intOpt(Map("n" -> "abc"), "n", 7)
    assert(bad.isLeft && bad.left.toOption.exists(_.contains("--n must be an integer")))
  }

  test("longOpt parses long values") {
    assertEquals(HallConfig.longOpt(Map("seed" -> "9999999999"), "seed", 0L), Right(9999999999L))
    val bad = HallConfig.longOpt(Map("seed" -> "x"), "seed", 0L)
    assert(bad.isLeft && bad.left.toOption.exists(_.contains("must be a long")))
  }

  test("doubleOpt parses doubles") {
    assertEquals(HallConfig.doubleOpt(Map("rate" -> "0.5"), "rate", 0.0), Right(0.5))
    val bad = HallConfig.doubleOpt(Map("rate" -> "ten"), "rate", 0.0)
    assert(bad.isLeft && bad.left.toOption.exists(_.contains("must be a double")))
  }

  test("boolOpt accepts true/false case-insensitively, rejects other tokens") {
    assertEquals(HallConfig.boolOpt(Map("flag" -> "true"), "flag", false), Right(true))
    assertEquals(HallConfig.boolOpt(Map("flag" -> "FALSE"), "flag", true), Right(false))
    assertEquals(HallConfig.boolOpt(Map("flag" -> "  True "), "flag", false), Right(true))
    assertEquals(HallConfig.boolOpt(Map.empty, "flag", true), Right(true))
    val bad = HallConfig.boolOpt(Map("flag" -> "yes"), "flag", false)
    assert(bad.isLeft && bad.left.toOption.exists(_.contains("must be true or false")))
  }

  test("optionalPathOpt returns None when key absent, Some(path) when present") {
    val absent = HallConfig.optionalPathOpt(Map.empty, "model")
    assertEquals(absent, Right(None))
    val present = HallConfig.optionalPathOpt(Map("model" -> "data/x"), "model")
    assert(present.exists(_.exists(_.toString.endsWith("x"))))
  }

  // --- parseArgs end-to-end happy path ---

  test("parseArgs accepts an empty arg array and produces all defaults") {
    val cfg = HallConfig.parseArgs(Array.empty).fold(err => fail(s"unexpected error: $err"), identity)
    assertEquals(cfg.hands, 100000)
    assertEquals(cfg.tableCount, 1)
    assertEquals(cfg.playerCount, 2)
    assertEquals(cfg.reportEvery, 10000)
    assertEquals(cfg.heroPosition, Position.Button)
    assertEquals(cfg.heroExplorationRate, 0.05)
    assertEquals(cfg.raiseSize, 2.5)
    assertEquals(cfg.bunchingTrials, 80)
    assertEquals(cfg.equityTrials, 700)
    assertEquals(cfg.saveTrainingTsv, true)
    assertEquals(cfg.saveDdreTrainingTsv, false)
    assertEquals(cfg.saveReviewHandHistory, false)
    assertEquals(cfg.fullRing, false)
    assertEquals(cfg.villainPool.size, 1, "default --villainStyle=tag should yield a single-villain pool")
  }

  test("parseArgs returns Left with usage when --help is supplied") {
    val result = HallConfig.parseArgs(Array("--help"))
    assert(result.isLeft, s"expected Left, got $result")
    assert(result.left.toOption.exists(_.contains("Usage:")))
  }

  test("parseArgs rejects out-of-range hands and explorationRate") {
    val negHands = HallConfig.parseArgs(Array("--hands=-1"))
    assert(negHands.left.toOption.exists(_.contains("--hands must be > 0")), s"got $negHands")
    val highRate = HallConfig.parseArgs(Array("--heroExplorationRate=1.5"))
    assert(highRate.left.toOption.exists(_.contains("must be in [0,1]")), s"got $highRate")
  }

  test("parseArgs honours --playerCount and --heroPosition with cross-validation") {
    val cfg = HallConfig.parseArgs(Array("--playerCount=6", "--heroPosition=UTG"))
      .fold(err => fail(s"unexpected error: $err"), identity)
    assertEquals(cfg.playerCount, 6)
    assertEquals(cfg.heroPosition, Position.UTG)

    // UTG is not modeled at 4-max -> rejected
    val rejected = HallConfig.parseArgs(Array("--playerCount=4", "--heroPosition=UTG"))
    assert(rejected.left.toOption.exists(_.contains("not valid for playerCount=4")), s"got $rejected")
  }

  test("parseArgs rejects --playerCount outside [2..9]") {
    val tooFew = HallConfig.parseArgs(Array("--playerCount=1"))
    assert(tooFew.left.toOption.exists(_.contains("[2,9]")), s"got $tooFew")
    val tooMany = HallConfig.parseArgs(Array("--playerCount=10"))
    assert(tooMany.left.toOption.exists(_.contains("[2,9]")), s"got $tooMany")
  }
