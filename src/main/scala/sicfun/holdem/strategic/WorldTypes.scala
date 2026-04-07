package sicfun.holdem.strategic

/** World algebra types (Wave 1 — v0.31.1 formal closure).
  *
  * Three orthogonal axes span the world space:
  *   - Learning channel  omega^act  in {Blind, Ref, Attrib, Design}
  *   - Showdown mode     omega^sd   in {Off, On}
  *   - Policy scope      omega^pol  in {OpenLoop, ClosedLoop}
  *
  * Chain space: Omega^chain = LearningChannel x ShowdownMode   (8 elements)
  * Grid  space: Omega^grid  = {Blind, Attrib} x PolicyScope    (4 elements)
  */

enum LearningChannel:
  case Blind, Ref, Attrib, Design

enum ShowdownMode:
  case Off, On

enum PolicyScope:
  case OpenLoop, ClosedLoop

final case class ChainWorld(channel: LearningChannel, showdown: ShowdownMode)

object ChainWorld:
  /** All 8 chain worlds (full Cartesian product). */
  val all: IndexedSeq[ChainWorld] =
    for
      ch <- LearningChannel.values.toIndexedSeq
      sd <- ShowdownMode.values.toIndexedSeq
    yield ChainWorld(ch, sd)

  /** Default telescopic chain (Def 28):
    *   (Blind,Off) -> (Ref,Off) -> (Attrib,Off) -> (Attrib,On)
    */
  val canonicalChain: IndexedSeq[ChainWorld] = IndexedSeq(
    ChainWorld(LearningChannel.Blind, ShowdownMode.Off),
    ChainWorld(LearningChannel.Ref, ShowdownMode.Off),
    ChainWorld(LearningChannel.Attrib, ShowdownMode.Off),
    ChainWorld(LearningChannel.Attrib, ShowdownMode.On)
  )

final case class GridWorld(learning: LearningChannel, scope: PolicyScope):
  require(
    learning == LearningChannel.Blind || learning == LearningChannel.Attrib,
    s"GridWorld only admits Blind or Attrib, got $learning"
  )

object GridWorld:
  /** All 4 grid worlds. */
  val all: IndexedSeq[GridWorld] = IndexedSeq(
    GridWorld(LearningChannel.Blind, PolicyScope.OpenLoop),
    GridWorld(LearningChannel.Blind, PolicyScope.ClosedLoop),
    GridWorld(LearningChannel.Attrib, PolicyScope.OpenLoop),
    GridWorld(LearningChannel.Attrib, PolicyScope.ClosedLoop)
  )
