package sicfun.core

/** A probabilistic model of actions conditioned on observable features and a latent hypothesis.
  *
  * In the poker context, Feature is the visible game state (board, pot size, position),
  * Action is a player's decision (fold, call, raise), and Hypothesis is the hidden
  * variable being inferred (e.g., the opponent's hand category or range).
  *
  * @tparam Feature    type of observable game features
  * @tparam Action     type of actions
  * @tparam Hypothesis type of latent hypotheses being inferred
  */
trait ActionModel[Feature, Action, Hypothesis]:
  /** Returns P(action | features, hypothesis) -- the likelihood of observing
    * the given action under a specific hypothesis and feature set.
    */
  def likelihood(action: Action, features: Feature, hypothesis: Hypothesis): Double

/** Generic Bayesian inference engine over a discrete hypothesis space.
  *
  * Wraps a [[DiscreteDistribution]] over hypotheses and supports sequential
  * Bayesian updating via observed (action, features) pairs. After each observation,
  * the posterior is computed using Bayes' rule:
  *
  * P(h | action, features) proportional to P(action | features, h) * P(h)
  *
  * This is the core mechanism for opponent hand range narrowing in poker:
  * each observed action updates the belief distribution over possible hands.
  *
  * @param distribution the current belief (prior or posterior) over hypotheses
  * @tparam H the hypothesis type
  */
final case class BayesianRange[H](distribution: DiscreteDistribution[H]):
  /** Performs a single Bayesian update given an observed action.
    *
    * @param action   the observed action
    * @param features the observable game state at the time of the action
    * @param model    the likelihood model P(action | features, h)
    * @return a tuple of (updated BayesianRange, marginal evidence P(action | features))
    */
  def update[Feature, Action](
      action: Action,
      features: Feature,
      model: ActionModel[Feature, Action, H]
  ): (BayesianRange[H], Double) =
    val (updated, evidence) = distribution.updateWithLikelihood(h => model.likelihood(action, features, h))
    (BayesianRange(updated), evidence)

  /** Performs sequential Bayesian updates for a series of observations.
    *
    * Accumulates log-evidence (log marginal likelihood) across all observations,
    * which can be used for model comparison or anomaly detection.
    *
    * @param actions sequence of (action, features) pairs in chronological order
    * @param model   the likelihood model
    * @return a tuple of (final posterior BayesianRange, total log-evidence)
    */
  def updateAll[Feature, Action](
      actions: Seq[(Action, Feature)],
      model: ActionModel[Feature, Action, H]
  ): (BayesianRange[H], Double) =
    if actions.isEmpty then (this, 0.0)
    else
      val hypotheses = distribution.weights.keysIterator.toVector
      val probabilities = new Array[Double](hypotheses.length)
      var i = 0
      while i < hypotheses.length do
        probabilities(i) = distribution.weights(hypotheses(i))
        i += 1

      val eps = Probability.Eps
      var logEvidence = 0.0
      actions.foreach { case (action, features) =>
        var evidence = 0.0
        var j = 0
        while j < hypotheses.length do
          val updated = probabilities(j) * model.likelihood(action, features, hypotheses(j))
          probabilities(j) = updated
          evidence += updated
          j += 1

        require(evidence > eps, "likelihoods produce zero evidence")
        val invEvidence = 1.0 / evidence
        var k = 0
        while k < probabilities.length do
          probabilities(k) = probabilities(k) * invEvidence
          k += 1
        // Accumulate log-evidence to avoid floating-point underflow from multiplying small probabilities.
        logEvidence += math.log(evidence)
      }

      val updatedWeights = Map.newBuilder[H, Double]
      i = 0
      while i < hypotheses.length do
        updatedWeights += hypotheses(i) -> probabilities(i)
        i += 1
      (BayesianRange(DiscreteDistribution(updatedWeights.result())), logEvidence)

/** Factory methods for [[BayesianRange]]. */
object BayesianRange:
  /** Creates a BayesianRange with a uniform prior over the given hypotheses.
    *
    * This is the natural starting point when no prior information is available
    * (maximum entropy prior over a finite hypothesis space).
    */
  def uniform[H](hypotheses: Seq[H]): BayesianRange[H] =
    BayesianRange(DiscreteDistribution.uniform(hypotheses))
