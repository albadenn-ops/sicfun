package sicfun.holdem.strategic

/** World algebra types (Wave 1 — v0.31.1 formal closure).
  *
  * Three orthogonal axes span the world space:
  *   - Learning channel  omega^act  in {Blind, Ref, Attrib, Design}
  *   - Showdown mode     omega^sd   in {Off, On}
  *   - Policy scope      omega^pol  in {OpenLoop, ClosedLoop}
  *
  * Two non-overlapping product spaces:
  *   Chain space: Omega^chain = LearningChannel x ShowdownMode   (8 nominal, 6 effective)
  *   Grid  space: Omega^grid  = {Blind, Attrib} x PolicyScope    (4 elements)
  *
  * Full product space (reserved): Omega^full = Omega^act x Omega^sd x Omega^pol  (16 elements)
  *
  * Effective cardinality note (Def 20): (Blind,Off) and (Blind,On) are functionally
  * identical because the blind kernel ignores all signals including showdown. The nominal
  * cardinality of 8 in Omega^chain is retained for notational uniformity, but the effective
  * number of distinct worlds is 7 (1 blind representative + 3 non-blind × 2 showdown = 7).
  * Note: the spec text says "6" but this is an arithmetic error (8 - 2 + 1 = 7).
  */

enum LearningChannel:
  case Blind, Ref, Attrib, Design

enum ShowdownMode:
  case Off, On

enum PolicyScope:
  case OpenLoop, ClosedLoop

final case class ChainWorld(channel: LearningChannel, showdown: ShowdownMode):
  /** True iff this world is functionally identical to another due to blind-kernel
    * equivalence: (Blind,Off) == (Blind,On) since blind ignores all signals.
    */
  inline def isBlindEquivalent(other: ChainWorld): Boolean =
    channel == LearningChannel.Blind && other.channel == LearningChannel.Blind

object ChainWorld:
  /** All 8 chain worlds (full Cartesian product, nominal cardinality). */
  val all: IndexedSeq[ChainWorld] =
    for
      ch <- LearningChannel.values.toIndexedSeq
      sd <- ShowdownMode.values.toIndexedSeq
    yield ChainWorld(ch, sd)

  /** The 6 effectively distinct chain worlds (Def 20 cardinality note).
    * Collapses (Blind,Off)/(Blind,On) to a single representative (Blind,Off).
    */
  val effectivelyDistinct: IndexedSeq[ChainWorld] =
    all.filterNot(w => w.channel == LearningChannel.Blind && w.showdown == ShowdownMode.On)

  /** Default telescopic chain (Def 44'):
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

/** Full product space Omega^full (reserved notation, §0 Convention).
  *
  * Omega^full := Omega^act x Omega^sd x Omega^pol, |Omega^full| = 16.
  * No theorem in v0.31.1 requires simultaneous variation of all three axes,
  * but the type is provided for future extensions and diagnostic tooling.
  */
final case class FullWorld(
    channel: LearningChannel,
    showdown: ShowdownMode,
    scope: PolicyScope
):
  /** Project onto chain space (drop policy scope axis). */
  inline def toChainWorld: ChainWorld = ChainWorld(channel, showdown)

  /** Project onto grid space (drop showdown axis).
    * Only valid when channel is Blind or Attrib.
    */
  def toGridWorld: GridWorld = GridWorld(channel, scope)

object FullWorld:
  /** All 16 full worlds (complete Cartesian product of three axes). */
  val all: IndexedSeq[FullWorld] =
    for
      ch <- LearningChannel.values.toIndexedSeq
      sd <- ShowdownMode.values.toIndexedSeq
      sc <- PolicyScope.values.toIndexedSeq
    yield FullWorld(ch, sd, sc)

  /** Lift a ChainWorld into FullWorld by supplying the missing policy scope axis. */
  def fromChain(cw: ChainWorld, scope: PolicyScope): FullWorld =
    FullWorld(cw.channel, cw.showdown, scope)

  /** Lift a GridWorld into FullWorld by supplying the missing showdown axis. */
  def fromGrid(gw: GridWorld, showdown: ShowdownMode): FullWorld =
    FullWorld(gw.learning, showdown, gw.scope)
