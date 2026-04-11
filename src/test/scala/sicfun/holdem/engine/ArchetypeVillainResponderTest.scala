package sicfun.holdem.engine

import munit.FunSuite
import sicfun.core.Card
import sicfun.holdem.types.*

/** Tests for [[ArchetypeVillainResponder]].
  *
  * Validates:
  *   - Style profile mapping: each archetype has correctly bounded looseness/aggression.
  *   - Decision logic: check-to-act without raise permission always returns Check.
  *   - Behavioral differentiation: Maniac raises significantly more than Nit over many
  *     trials, confirming the aggression parameter influences bet frequency.
  */
class ArchetypeVillainResponderTest extends FunSuite:

  test("styleProfile: Nit is tight and passive"):
    val profile = ArchetypeVillainResponder.styleProfile(PlayerArchetype.Nit)
    assert(profile.looseness < 0.25, s"Nit looseness ${profile.looseness}")
    assert(profile.aggression < 0.25, s"Nit aggression ${profile.aggression}")

  test("styleProfile: Maniac is loose and aggressive"):
    val profile = ArchetypeVillainResponder.styleProfile(PlayerArchetype.Maniac)
    assert(profile.looseness > 0.75, s"Maniac looseness ${profile.looseness}")
    assert(profile.aggression > 0.85, s"Maniac aggression ${profile.aggression}")

  test("styleProfile: Tag is moderate"):
    val profile = ArchetypeVillainResponder.styleProfile(PlayerArchetype.Tag)
    assert(profile.looseness > 0.3 && profile.looseness < 0.6, s"Tag looseness ${profile.looseness}")
    assert(profile.aggression > 0.3 && profile.aggression < 0.6, s"Tag aggression ${profile.aggression}")

  test("styleProfile: CallingStation is loose and passive"):
    val profile = ArchetypeVillainResponder.styleProfile(PlayerArchetype.CallingStation)
    assert(profile.looseness > 0.8, s"Station looseness ${profile.looseness}")
    assert(profile.aggression < 0.3, s"Station aggression ${profile.aggression}")

  test("styleProfile: all archetypes covered"):
    PlayerArchetype.values.foreach { archetype =>
      val profile = ArchetypeVillainResponder.styleProfile(archetype)
      assert(profile.looseness >= 0.0 && profile.looseness <= 1.0, s"$archetype looseness out of range: ${profile.looseness}")
      assert(profile.aggression >= 0.0 && profile.aggression <= 1.0, s"$archetype aggression out of range: ${profile.aggression}")
    }

  test("villainResponds: check-to-act with no raise allowed returns Check"):
    val state = GameState(
      street = Street.Flop, pot = 3.0, toCall = 0.0, stackSize = 50.0,
      board = Board.empty, position = Position.BigBlind,
      betHistory = Vector.empty
    )
    val h = HoleCards.from(Vector(Card.parse("2c").get, Card.parse("7d").get))
    val action = ArchetypeVillainResponder.villainResponds(
      hand = h, style = PlayerArchetype.Nit, state = state,
      allowRaise = false, raiseSize = 2.5, rng = new scala.util.Random(42)
    )
    assertEquals(action, PokerAction.Check)

  test("villainResponds: Maniac raises more than Nit over many trials"):
    val state = GameState(
      street = Street.Flop, pot = 5.0, toCall = 0.0, stackSize = 50.0,
      board = Board.from(Vector(Card.parse("Ah").get, Card.parse("Kd").get, Card.parse("Qs").get)),
      position = Position.BigBlind,
      betHistory = Vector.empty
    )
    val h = HoleCards.from(Vector(Card.parse("Jh").get, Card.parse("Ts").get))
    def countRaises(style: PlayerArchetype, seed: Int): Int =
      (0 until 100).count { i =>
        ArchetypeVillainResponder.villainResponds(
          hand = h, style = style, state = state,
          allowRaise = true, raiseSize = 2.5, rng = new scala.util.Random(seed + i)
        ) match
          case PokerAction.Raise(_) => true
          case _                    => false
      }
    val maniacRaises = countRaises(PlayerArchetype.Maniac, 1000)
    val nitRaises = countRaises(PlayerArchetype.Nit, 1000)
    assert(maniacRaises > nitRaises, s"Maniac raises $maniacRaises should be > Nit raises $nitRaises")
