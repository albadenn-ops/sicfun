package sicfun.holdem.bench

class CanonicalFieldTest extends munit.FunSuite:
  test("9-max field has a version stamp and 8 villain tokens") {
    assertEquals(CanonicalField.NineMaxExploitable.version, "p0-field-v1")
    assertEquals(CanonicalField.NineMaxExploitable.villainTokens.length, 8)
    // only archetypes the hall accepts
    val allowed = Set("nit", "tag", "lag", "callingstation", "station", "maniac", "gto")
    CanonicalField.NineMaxExploitable.villainTokens.foreach(t =>
      assert(allowed.contains(t), s"unsupported villain token: $t"))
  }
  test("hallArgs is deterministic and disables learning + exploration") {
    val args = CanonicalField.NineMaxExploitable.hallArgs(heroStyle = "strategic", hands = 20000, seed = 42L, outDir = "data/p0/strategic")
    assert(args.contains("--playerCount=9"), args.mkString(" "))
    assert(args.contains("--heroStyle=strategic"))
    assert(args.contains("--learnEveryHands=0"))
    assert(args.contains("--heroExplorationRate=0.0"))
    assert(args.contains("--fullRing=true"))
    assert(args.contains("--seed=42"))
    assert(args.contains("--hands=20000"))
    assert(args.exists(_.startsWith("--villainPool=")))
  }
