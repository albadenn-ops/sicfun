package sicfun.holdem.strategic

import sicfun.holdem.types.Position

class TableStructureTest extends munit.FunSuite:

  // -- PlayerId --

  test("PlayerId wraps and unwraps string"):
    val pid = PlayerId("alice")
    assertEquals(pid.value, "alice")

  test("PlayerId equality is structural"):
    assertEquals(PlayerId("bob"), PlayerId("bob"))
    assertNotEquals(PlayerId("bob"), PlayerId("carol"))

  // -- SeatStatus --

  test("SeatStatus has exactly four variants"):
    assertEquals(SeatStatus.values.length, 4)

  // -- RivalMap invariant: never empty (Law L1) --

  test("RivalMap rejects empty vector"):
    intercept[IllegalArgumentException]:
      RivalMap[Chips](Vector.empty)

  test("RivalMap rejects duplicate PlayerId"):
    val pid = PlayerId("duped")
    intercept[IllegalArgumentException]:
      RivalMap(Vector(
        Seat(pid, Position.BigBlind, SeatStatus.Active, Chips(100.0)),
        Seat(pid, Position.UTG, SeatStatus.Active, Chips(200.0))
      ))

  test("RivalMap accepts single rival (heads-up is |R|=1)"):
    val rm = RivalMap(Vector(
      Seat(PlayerId("v1"), Position.BigBlind, SeatStatus.Active, Chips(100.0))
    ))
    assertEquals(rm.size, 1)

  test("RivalMap accepts two rivals (multiway)"):
    val rm = RivalMap(Vector(
      Seat(PlayerId("v1"), Position.BigBlind, SeatStatus.Active, Chips(100.0)),
      Seat(PlayerId("v2"), Position.UTG, SeatStatus.Active, Chips(200.0))
    ))
    assertEquals(rm.size, 2)

  test("RivalMap.get finds rival by PlayerId"):
    val pid = PlayerId("target")
    val rm = RivalMap(Vector(
      Seat(PlayerId("other"), Position.BigBlind, SeatStatus.Active, Chips(100.0)),
      Seat(pid, Position.UTG, SeatStatus.Active, Chips(200.0))
    ))
    assertEquals(rm.get(pid).map(_.data), Some(Chips(200.0)))
    assertEquals(rm.get(PlayerId("missing")), None)

  test("RivalMap.mapData transforms data preserving identity and position"):
    val rm = RivalMap(Vector(
      Seat(PlayerId("v1"), Position.BigBlind, SeatStatus.Active, Chips(100.0))
    ))
    val doubled = rm.mapData(c => Chips(c.value * 2))
    assertEquals(doubled.rivals.head.data, Chips(200.0))
    assertEquals(doubled.rivals.head.playerId, PlayerId("v1"))
    assertEquals(doubled.rivals.head.position, Position.BigBlind)

  // -- TableMap --

  test("TableMap rejects when hero is not in seats"):
    intercept[IllegalArgumentException]:
      TableMap(
        hero = PlayerId("hero"),
        seats = Vector(
          Seat(PlayerId("v1"), Position.SmallBlind, SeatStatus.Active, Chips(100.0))
        )
      )

  test("TableMap rejects duplicate PlayerId"):
    val pid = PlayerId("duped")
    intercept[IllegalArgumentException]:
      TableMap(
        hero = pid,
        seats = Vector(
          Seat(pid, Position.SmallBlind, SeatStatus.Active, Chips(100.0)),
          Seat(pid, Position.BigBlind, SeatStatus.Active, Chips(200.0))
        )
      )

  test("TableMap.toRivalMap excludes hero"):
    val hero = PlayerId("hero")
    val tm = TableMap(
      hero = hero,
      seats = Vector(
        Seat(hero, Position.SmallBlind, SeatStatus.Active, Chips(100.0)),
        Seat(PlayerId("v1"), Position.BigBlind, SeatStatus.Active, Chips(200.0)),
        Seat(PlayerId("v2"), Position.UTG, SeatStatus.Folded, Chips(50.0))
      )
    )
    val rm = tm.toRivalMap
    assertEquals(rm.size, 2)
    assert(rm.rivals.forall(_.playerId != hero))

  test("TableMap.activeRivalMap excludes folded and sitting-out players"):
    val hero = PlayerId("hero")
    val tm = TableMap(
      hero = hero,
      seats = Vector(
        Seat(hero, Position.SmallBlind, SeatStatus.Active, Chips(100.0)),
        Seat(PlayerId("v1"), Position.BigBlind, SeatStatus.Active, Chips(200.0)),
        Seat(PlayerId("v2"), Position.UTG, SeatStatus.Folded, Chips(50.0)),
        Seat(PlayerId("v3"), Position.UTG1, SeatStatus.AllIn, Chips(0.0)),
        Seat(PlayerId("v4"), Position.UTG2, SeatStatus.SittingOut, Chips(300.0))
      )
    )
    val active = tm.activeRivalMap
    assertEquals(active.size, 2) // v1 (Active) + v3 (AllIn)
    assert(active.get(PlayerId("v1")).isDefined)
    assert(active.get(PlayerId("v3")).isDefined)
    assert(active.get(PlayerId("v2")).isEmpty) // Folded
    assert(active.get(PlayerId("v4")).isEmpty) // SittingOut

  test("Heads-up table: activeRivalMap has exactly one rival"):
    val hero = PlayerId("hero")
    val tm = TableMap(
      hero = hero,
      seats = Vector(
        Seat(hero, Position.SmallBlind, SeatStatus.Active, Chips(100.0)),
        Seat(PlayerId("v1"), Position.BigBlind, SeatStatus.Active, Chips(200.0))
      )
    )
    assertEquals(tm.activeRivalMap.size, 1)
