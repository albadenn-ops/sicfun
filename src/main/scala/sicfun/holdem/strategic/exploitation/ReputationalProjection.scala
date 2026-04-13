package sicfun.holdem.strategic

/** Reputation view of a rival's strategic profile as perceived at a point in time.
  *
  * A reputation captures the observer's current belief about the opponent's
  * playing style across three key behavioral dimensions:
  *
  * @param perceivedTightness
  *   Estimated fraction of hands played (lower = more selective)
  * @param perceivedAggression
  *   Estimated frequency of aggressive actions (raises, bets)
  * @param perceivedBluffFrequency
  *   Estimated fraction of bets that are non-showdown wins
  * @param raw
  *   Extensible map for additional belief dimensions (e.g., positional biases, fold-to-3bet)
  */
final case class ReputationView(
    perceivedTightness: PotFraction,
    perceivedAggression: PotFraction,
    perceivedBluffFrequency: PotFraction,
    raw: Map[String, Double] = Map.empty
)

/** Trait for projecting a rival's internal belief state onto a reputation model.
  *
  * Used during opponent modeling to update the agent's perception of how an opponent
  * should be treated based on observed behavior. The projection extracts high-level
  * strategic signals from low-level game events.
  */
trait ReputationalProjection:
  def project(rivalState: RivalBeliefState): ReputationView
