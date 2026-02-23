package sicfun.core

/** Information-theoretic analysis of player actions in poker.
  *
  * These metrics quantify how much a player's chosen action (fold, call, raise, etc.)
  * reveals about their hidden hand. The core quantities are:
  *
  *  - '''H(action | state)''': marginal action entropy -- how unpredictable the action is.
  *  - '''H(action | state, hand)''': conditional action entropy -- residual unpredictability
  *    once the hand is known.
  *  - '''I(action; hand | state)''': mutual information -- the difference, measuring
  *    how many bits about the hand are leaked by the action.
  *
  * High mutual information means the action is highly informative about the hand,
  * which is exploitable by opponents.
  */
object ActionMetrics:

  /** Computes the marginal action entropy H(action | state).
    *
    * Marginalizes over all hypotheses (hands) to obtain P(action | state),
    * then computes Shannon entropy of the resulting action distribution.
    *
    * P(action | state) = sum,,h,, P(h) * P(action | state, h)
    *
    * @param state    the observable game state (board, pot, etc.)
    * @param posterior current belief distribution over hypotheses (opponent hands)
    * @param model    action likelihood model P(action | state, hypothesis)
    * @param actions  the set of possible actions to consider
    * @param base     logarithm base for entropy (default 2.0 for bits)
    * @return Shannon entropy of the marginal action distribution
    */
  def actionEntropy[Feature, Action, Hypothesis](
      state: Feature,
      posterior: DiscreteDistribution[Hypothesis],
      model: ActionModel[Feature, Action, Hypothesis],
      actions: Seq[Action],
      base: Double = 2.0
  ): Double =
    val norm = posterior.normalized
    // Compute P(action | state) by marginalizing over all hypotheses.
    val marginal = actions.map { action =>
      norm.weights.foldLeft(0.0) { case (acc, (h, w)) =>
        acc + w * model.likelihood(action, state, h)
      }
    }
    Metrics.entropy(marginal, base)

  /** Computes the conditional action entropy H(action | state, hand).
    *
    * This is the weighted average of per-hypothesis action entropy:
    * H(action | state, hand) = sum,,h,, P(h) * H(action | state, h)
    *
    * Represents the residual unpredictability of the action even when the hand is known.
    * A high value means the player randomizes their action strategy (mixed strategy).
    *
    * @return conditional entropy in the specified base (default bits)
    */
  def conditionalActionEntropy[Feature, Action, Hypothesis](
      state: Feature,
      posterior: DiscreteDistribution[Hypothesis],
      model: ActionModel[Feature, Action, Hypothesis],
      actions: Seq[Action],
      base: Double = 2.0
  ): Double =
    val norm = posterior.normalized
    val weightedDists = norm.weights.toSeq.map { case (h, w) =>
      val actionProbs = actions.map(a => model.likelihood(a, state, h))
      (w, actionProbs)
    }
    Metrics.conditionalEntropy(weightedDists, base)

  /** Computes the mutual information I(action; hand | state).
    *
    * Defined as H(action | state) - H(action | state, hand). This is the amount
    * of information (in bits, by default) that observing the action provides about
    * the hidden hand, conditioned on the visible game state.
    *
    * A value near zero means the player's action strategy is independent of their hand
    * (perfectly balanced). A high value means the action is a strong signal of hand strength.
    *
    * @return mutual information, always non-negative
    */
  def mutualInformation[Feature, Action, Hypothesis](
      state: Feature,
      posterior: DiscreteDistribution[Hypothesis],
      model: ActionModel[Feature, Action, Hypothesis],
      actions: Seq[Action],
      base: Double = 2.0
  ): Double =
    actionEntropy(state, posterior, model, actions, base) -
      conditionalActionEntropy(state, posterior, model, actions, base)
