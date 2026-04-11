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

  // ---- Blind equivalence (Def 20 cardinality note) --------------------------

  test("(Blind,Off) and (Blind,On) are blind-equivalent"):
    val off = ChainWorld(LearningChannel.Blind, ShowdownMode.Off)
    val on = ChainWorld(LearningChannel.Blind, ShowdownMode.On)
    assert(off.isBlindEquivalent(on))
    assert(on.isBlindEquivalent(off))

  test("non-blind worlds are not blind-equivalent"):
    val attribOff = ChainWorld(LearningChannel.Attrib, ShowdownMode.Off)
    val attribOn = ChainWorld(LearningChannel.Attrib, ShowdownMode.On)
    assert(!attribOff.isBlindEquivalent(attribOn))

  test("effectivelyDistinct has 7 elements (8 - 1 blind pair collapsed)"):
    // Spec says "6" in Def 20 cardinality note but the correct count is 7:
    // 1 blind representative + 3 non-blind channels × 2 showdown modes = 7
    assertEquals(ChainWorld.effectivelyDistinct.size, 7)

  test("effectivelyDistinct collapses (Blind,On)"):
    assert(ChainWorld.effectivelyDistinct.contains(
      ChainWorld(LearningChannel.Blind, ShowdownMode.Off)
    ))
    assert(!ChainWorld.effectivelyDistinct.contains(
      ChainWorld(LearningChannel.Blind, ShowdownMode.On)
    ))

  // ---- FullWorld (reserved notation, §0) ------------------------------------

  test("FullWorld.all has 16 elements (4 x 2 x 2)"):
    assertEquals(FullWorld.all.size, 16)

  test("FullWorld.all contains no duplicates"):
    assertEquals(FullWorld.all.distinct.size, FullWorld.all.size)

  test("FullWorld.toChainWorld projects correctly"):
    val fw = FullWorld(LearningChannel.Attrib, ShowdownMode.On, PolicyScope.ClosedLoop)
    assertEquals(fw.toChainWorld, ChainWorld(LearningChannel.Attrib, ShowdownMode.On))

  test("FullWorld.toGridWorld projects correctly for valid channels"):
    val fw = FullWorld(LearningChannel.Blind, ShowdownMode.Off, PolicyScope.OpenLoop)
    assertEquals(fw.toGridWorld, GridWorld(LearningChannel.Blind, PolicyScope.OpenLoop))

  test("FullWorld.toGridWorld rejects invalid channels"):
    val fw = FullWorld(LearningChannel.Ref, ShowdownMode.Off, PolicyScope.OpenLoop)
    intercept[IllegalArgumentException]:
      fw.toGridWorld

  test("FullWorld.fromChain lifts correctly"):
    val cw = ChainWorld(LearningChannel.Design, ShowdownMode.On)
    val fw = FullWorld.fromChain(cw, PolicyScope.ClosedLoop)
    assertEquals(fw.channel, LearningChannel.Design)
    assertEquals(fw.showdown, ShowdownMode.On)
    assertEquals(fw.scope, PolicyScope.ClosedLoop)

  test("FullWorld.fromGrid lifts correctly"):
    val gw = GridWorld(LearningChannel.Attrib, PolicyScope.OpenLoop)
    val fw = FullWorld.fromGrid(gw, ShowdownMode.Off)
    assertEquals(fw.channel, LearningChannel.Attrib)
    assertEquals(fw.showdown, ShowdownMode.Off)
    assertEquals(fw.scope, PolicyScope.OpenLoop)

  test("chain/grid spaces are not subsets of each other"):
    // Chain projects to (LearningChannel, ShowdownMode)
    // Grid projects to ({Blind, Attrib}, PolicyScope)
    // Chain has Ref and Design channels; Grid doesn't
    // Grid has PolicyScope axis; Chain doesn't
    val chainChannels = ChainWorld.all.map(_.channel).toSet
    val gridChannels = GridWorld.all.map(_.learning).toSet
    assert(chainChannels != gridChannels, "chain and grid channel sets should differ")
