package sicfun.holdem.runtime

import munit.FunSuite
import sicfun.holdem.equity.TableFormat
import sicfun.holdem.types.Position

class HeadsUpMatchDefaultsTest extends FunSuite:

  test("heads-up runner defaults use heads-up ranges without phantom folds"):
    assertEquals(HeadsUpMatchDefaults.tableRanges.format, TableFormat.HeadsUp)
    assertEquals(
      HeadsUpMatchDefaults.tableRanges.ranges.keySet,
      Set(Position.Button, Position.BigBlind)
    )
    assertEquals(HeadsUpMatchDefaults.preflopFoldsBeforeButtonOpen, Vector.empty)

  test("heads-up runners are wired to heads-up defaults"):
    assertEquals(AcpcMatchRunner.tableRangesForMatch.format, TableFormat.HeadsUp)
    assertEquals(AcpcMatchRunner.preflopFoldsForMatch, Vector.empty)
    assertEquals(SlumbotMatchRunner.tableRangesForMatch.format, TableFormat.HeadsUp)
    assertEquals(SlumbotMatchRunner.preflopFoldsForMatch, Vector.empty)

  test("nine-max button opener would inject phantom folds in heads-up matches"):
    val phantomFolds = TableFormat.NineMax.foldsBeforeOpener(Position.Button)
    assertEquals(
      phantomFolds,
      Vector(
        Position.UTG,
        Position.UTG1,
        Position.UTG2,
        Position.Middle,
        Position.Hijack,
        Position.Cutoff
      )
    )
