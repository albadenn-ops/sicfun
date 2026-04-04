package sicfun.holdem.strategic.bridge

import sicfun.holdem.strategic.*
import sicfun.holdem.types.{GameState, Street}

/** Bridge: GameState -> formal layer components.
  *
  * GameState (as of v0.30) contains: street, board, pot, toCall, position,
  * stackSize, betHistory. It does NOT contain a player list, hero name, or
  * per-player data. Accordingly, this bridge:
  *   - Maps street and pot exactly.
  *   - Maps the hero's stack exactly (as a single-player view).
  *   - Returns Absent for any multi-player construct (TableMap, RivalMap)
  *     because GameState is a hero-only snapshot.
  *
  * Fidelity:
  * - street: Exact
  * - pot: Exact
  * - heroStack: Exact
  * - TableMap / RivalMap: Absent (no player list in GameState)
  */
object PublicStateBridge:

  /** Extract the public street from a GameState. */
  def extractStreet(gs: GameState): BridgeResult[Street] =
    BridgeResult.Exact(gs.street)

  /** Extract pot size as Chips. */
  def extractPot(gs: GameState): BridgeResult[Chips] =
    BridgeResult.Exact(Chips(gs.pot))

  /** Extract hero stack as Chips. */
  def extractHeroStack(gs: GameState): BridgeResult[Chips] =
    BridgeResult.Exact(Chips(gs.stackSize))

  /** Extract the amount to call as Chips. */
  def extractToCall(gs: GameState): BridgeResult[Chips] =
    BridgeResult.Exact(Chips(gs.toCall))

  /** Build a TableMap from a GameState.
    *
    * GameState is a hero-only snapshot and does not carry a player list,
    * so a full TableMap cannot be constructed. Returns Absent.
    *
    * If a full TableMap is needed, it must be built from a richer data source
    * (e.g., the engine's live hand state or history import).
    */
  def extractTableMap(gs: GameState): BridgeResult[TableMap[Chips]] =
    BridgeResult.Absent(
      "GameState is a hero-only snapshot; no player list available to build TableMap"
    )
