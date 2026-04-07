package sicfun.holdem.strategic

class WorldTypesTest extends munit.FunSuite:

  // ---- ChainWorld -----------------------------------------------------------

  test("ChainWorld.all has 8 elements (4 channels x 2 showdown modes)"):
    assertEquals(ChainWorld.all.size, 8)

  test("ChainWorld.all contains no duplicates"):
    assertEquals(ChainWorld.all.distinct.size, ChainWorld.all.size)

  test("canonical chain has 4 elements"):
    assertEquals(ChainWorld.canonicalChain.size, 4)

  test("canonical chain follows (Blind,Off) -> (Ref,Off) -> (Attrib,Off) -> (Attrib,On)"):
    val expected = IndexedSeq(
      ChainWorld(LearningChannel.Blind, ShowdownMode.Off),
      ChainWorld(LearningChannel.Ref, ShowdownMode.Off),
      ChainWorld(LearningChannel.Attrib, ShowdownMode.Off),
      ChainWorld(LearningChannel.Attrib, ShowdownMode.On)
    )
    assertEquals(ChainWorld.canonicalChain, expected)

  test("every canonical chain element is in ChainWorld.all"):
    ChainWorld.canonicalChain.foreach: cw =>
      assert(ChainWorld.all.contains(cw), s"$cw not found in ChainWorld.all")

  // ---- GridWorld ------------------------------------------------------------

  test("GridWorld.all has 4 elements"):
    assertEquals(GridWorld.all.size, 4)

  test("GridWorld.all contains no duplicates"):
    assertEquals(GridWorld.all.distinct.size, GridWorld.all.size)

  test("GridWorld admits Blind"):
    val gw = GridWorld(LearningChannel.Blind, PolicyScope.OpenLoop)
    assertEquals(gw.learning, LearningChannel.Blind)

  test("GridWorld admits Attrib"):
    val gw = GridWorld(LearningChannel.Attrib, PolicyScope.ClosedLoop)
    assertEquals(gw.learning, LearningChannel.Attrib)

  test("GridWorld rejects Ref"):
    intercept[IllegalArgumentException]:
      GridWorld(LearningChannel.Ref, PolicyScope.OpenLoop)

  test("GridWorld rejects Design"):
    intercept[IllegalArgumentException]:
      GridWorld(LearningChannel.Design, PolicyScope.ClosedLoop)

  // ---- LearningChannel / ShowdownMode / PolicyScope enums -------------------

  test("LearningChannel has 4 values"):
    assertEquals(LearningChannel.values.length, 4)

  test("ShowdownMode has 2 values"):
    assertEquals(ShowdownMode.values.length, 2)

  test("PolicyScope has 2 values"):
    assertEquals(PolicyScope.values.length, 2)
