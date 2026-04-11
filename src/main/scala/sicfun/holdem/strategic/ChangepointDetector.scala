package sicfun.holdem.strategic

import sicfun.core.DiscreteDistribution

final case class ChangepointState(
    runLengthPosterior: Map[Int, Double]
)

final case class ChangepointDetector(
    hazardRate: Double,
    rMin: Int,
    kappaCP: Double,
    wReset: Double
):
  require(hazardRate > 0.0 && hazardRate < 1.0, s"hazardRate must be in (0,1), got $hazardRate")
  require(kappaCP > 0.0 && kappaCP < 1.0, s"kappaCP must be in (0,1), got $kappaCP")
  require(wReset > 0.0 && wReset <= 1.0, s"wReset must be in (0,1], got $wReset")

  def initial: ChangepointState =
    ChangepointState(runLengthPosterior = Map(0 -> 1.0))

  def update(state: ChangepointState, predictiveProb: Int => Double): ChangepointState =
    val prior = state.runLengthPosterior

    val growth = prior.map { case (ell, prob) =>
      (ell + 1) -> predictiveProb(ell) * (1.0 - hazardRate) * prob
    }

    val changepointMass = prior.foldLeft(0.0) { case (acc, (ell, prob)) =>
      acc + predictiveProb(0) * hazardRate * prob
    }

    val unnormalized = growth + (0 -> changepointMass)

    val total = unnormalized.values.sum
    val normalized =
      if total > 1e-15 then unnormalized.view.mapValues(_ / total).toMap
      else Map(0 -> 1.0)

    ChangepointState(runLengthPosterior = normalized)

  def isChangepointDetected(state: ChangepointState): Boolean =
    val shortMass = state.runLengthPosterior.foldLeft(0.0) { case (acc, (r, p)) =>
      if r <= rMin then acc + p else acc
    }
    shortMass > kappaCP

  def resetPrior[A](
      current: DiscreteDistribution[A],
      metaPrior: DiscreteDistribution[A]
  ): DiscreteDistribution[A] =
    val allKeys = current.support ++ metaPrior.support
    val blended = allKeys.map { key =>
      val cw = current.probabilityOf(key)
      val mw = metaPrior.probabilityOf(key)
      key -> ((1.0 - wReset) * cw + wReset * mw)
    }.toMap
    DiscreteDistribution(blended)
