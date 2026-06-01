package sicfun.holdem.runtime

class PlaceholderMarkerTest extends munit.FunSuite:

  class RealComponent
  class StubOne extends PlaceholderMarker:
    val placeholderReason = "stub one"
  class StubTwo extends PlaceholderMarker:
    val placeholderReason = "stub two"
  class ContainerDirect(val x: PlaceholderMarker)
  class ContainerNested(val child: ContainerDirect)

  test("scanPlaceholders: finds direct field"):
    val c = ContainerDirect(StubOne())
    val found = PlaceholderMarker.scanPlaceholders(c)
    assertEquals(found.map(_.placeholderReason).toSet, Set("stub one"))

  test("scanPlaceholders: finds nested field"):
    val c = ContainerNested(ContainerDirect(StubOne()))
    val found = PlaceholderMarker.scanPlaceholders(c)
    assertEquals(found.map(_.placeholderReason).toSet, Set("stub one"))

  test("scanPlaceholders: no placeholders -> empty vector"):
    val c = new RealComponent
    assertEquals(PlaceholderMarker.scanPlaceholders(c), Vector.empty)

  test("scanPlaceholders: multiple placeholders collected"):
    val c = ContainerDirect(StubOne())
    val found = PlaceholderMarker.scanPlaceholders(Vector(c, StubTwo()))
    assertEquals(found.map(_.placeholderReason).toSet, Set("stub one", "stub two"))
