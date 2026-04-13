package sicfun.holdem.runtime

import sicfun.holdem.equity.{PreflopFold, TableFormat, TableRanges}
import sicfun.holdem.types.Position

/** Shared heads-up priors for external match runners.
  *
  * Heads-up integrations must not inherit nine-max defaults, otherwise range
  * inference conditions on phantom seats and fake preflop folds.
  */
private[runtime] object HeadsUpMatchDefaults:
  val tableRanges: TableRanges = TableRanges.defaults(TableFormat.HeadsUp)

  val preflopFoldsBeforeButtonOpen: Vector[PreflopFold] =
    TableFormat.HeadsUp.foldsBeforeOpener(Position.Button).map(PreflopFold(_))
