package sicfun.holdem

import sicfun.core.{Deck, DiscreteDistribution, MultinomialLogistic, Probability}

import java.util.concurrent.atomic.AtomicReference
import scala.collection.mutable
import scala.util.Random

/** Provider selector/dispatcher for Bayesian posterior updates. */
private[holdem] object HoldemBayesProvider:
  enum Provider:
    case Scala
    case NativeCpu
    case NativeGpu

  private enum AutoSelection:
    case Unset
    case Provider(provider: HoldemBayesProvider.Provider)

  final case class UpdateResult(
      posterior: DiscreteDistribution[HoleCards],
      logEvidence: Double,
      provider: Provider
  )

  private val ProviderProperty = "sicfun.bayes.provider"
  private val ProviderEnv = "sicfun_BAYES_PROVIDER"
  private val AutoBenchmarkRepetitionsProperty = "sicfun.bayes.auto.benchmarkRepetitions"
  private val AutoBenchmarkRepetitionsEnv = "sicfun_BAYES_AUTO_BENCHMARK_REPETITIONS"
  private val AutoMinSpeedupProperty = "sicfun.bayes.auto.nativeMinSpeedup"
  private val AutoMinSpeedupEnv = "sicfun_BAYES_AUTO_NATIVE_MIN_SPEEDUP"
  private val ShadowEnabledProperty = "sicfun.bayes.shadow.enabled"
  private val ShadowEnabledEnv = "sicfun_BAYES_SHADOW_ENABLED"
  private val ShadowFailClosedProperty = "sicfun.bayes.shadow.failClosed"
  private val ShadowFailClosedEnv = "sicfun_BAYES_SHADOW_FAIL_CLOSED"
  private val ShadowPosteriorMaxAbsDiffProperty = "sicfun.bayes.shadow.posteriorMaxAbsDiff"
  private val ShadowPosteriorMaxAbsDiffEnv = "sicfun_BAYES_SHADOW_POSTERIOR_MAX_ABS_DIFF"
  private val ShadowLogEvidenceMaxAbsDiffProperty = "sicfun.bayes.shadow.logEvidenceMaxAbsDiff"
  private val ShadowLogEvidenceMaxAbsDiffEnv = "sicfun_BAYES_SHADOW_LOG_EVIDENCE_MAX_ABS_DIFF"
  private val DefaultAutoBenchmarkRepetitions = 20
  private val DefaultAutoMinSpeedup = 1.02
  private val DefaultShadowPosteriorMaxAbsDiff = 1e-9
  private val DefaultShadowLogEvidenceMaxAbsDiff = 1e-9
  private inline val MinLikelihood = 1e-6
  private val autoChosenProviderRef = new AtomicReference[AutoSelection](AutoSelection.Unset)

  private[holdem] final case class ShadowConfig(
      enabled: Boolean,
      failClosed: Boolean,
      posteriorMaxAbsDiff: Double,
      logEvidenceMaxAbsDiff: Double
  )

  private[holdem] final case class PosteriorDrift(
      posteriorMaxAbsDiff: Double,
      logEvidenceAbsDiff: Double
  )

  def providerLabel(provider: Provider): String =
    provider match
      case Provider.Scala => "scala"
      case Provider.NativeCpu => "native-cpu"
      case Provider.NativeGpu => "native-gpu"

  def currentProviderLabel: String =
    providerLabel(resolveConfiguredProvider())

  private[holdem] def resetAutoProviderForTests(): Unit =
    autoChosenProviderRef.set(AutoSelection.Unset)

  def updatePosterior(
      prior: DiscreteDistribution[HoleCards],
      observations: Seq[(PokerAction, GameState)],
      actionModel: PokerActionModel
  ): UpdateResult =
    if observations.isEmpty then
      UpdateResult(
        posterior = prior.normalized,
        logEvidence = 0.0,
        provider = Provider.Scala
      )
    else
      val hypotheses = prior.weights.keysIterator.toVector
      val priorArray = hypotheses.map(prior.probabilityOf).toArray
      val likelihoods = buildLikelihoodMatrix(observations, actionModel, hypotheses)
      val observationCount = observations.length
      val hypothesisCount = hypotheses.length
      val shadowConfig = configuredShadowConfig()
      val selected = resolveConfiguredProvider()
      selected match
        case Provider.Scala =>
          scalaUpdate(hypotheses, priorArray, likelihoods, observationCount, hypothesisCount, selected)
        case Provider.NativeCpu =>
          val preferred = nativeUpdate(
            backend = HoldemBayesNativeRuntime.Backend.Cpu,
            selectedProvider = Provider.NativeCpu,
            hypotheses = hypotheses,
            prior = priorArray,
            likelihoods = likelihoods,
            observationCount = observationCount,
            hypothesisCount = hypothesisCount
          ).getOrElse(scalaUpdate(
            hypotheses, priorArray, likelihoods, observationCount, hypothesisCount, Provider.Scala
          ))
          maybeValidateAgainstScala(
            preferred = preferred,
            hypotheses = hypotheses,
            prior = priorArray,
            likelihoods = likelihoods,
            observationCount = observationCount,
            hypothesisCount = hypothesisCount,
            shadowConfig = shadowConfig
          )
        case Provider.NativeGpu =>
          val preferred = nativeUpdate(
            backend = HoldemBayesNativeRuntime.Backend.Gpu,
            selectedProvider = Provider.NativeGpu,
            hypotheses = hypotheses,
            prior = priorArray,
            likelihoods = likelihoods,
            observationCount = observationCount,
            hypothesisCount = hypothesisCount
          ).orElse(
            nativeUpdate(
              backend = HoldemBayesNativeRuntime.Backend.Cpu,
              selectedProvider = Provider.NativeCpu,
              hypotheses = hypotheses,
              prior = priorArray,
              likelihoods = likelihoods,
              observationCount = observationCount,
              hypothesisCount = hypothesisCount
            )
          ).getOrElse(scalaUpdate(
            hypotheses, priorArray, likelihoods, observationCount, hypothesisCount, Provider.Scala
          ))
          maybeValidateAgainstScala(
            preferred = preferred,
            hypotheses = hypotheses,
            prior = priorArray,
            likelihoods = likelihoods,
            observationCount = observationCount,
            hypothesisCount = hypothesisCount,
            shadowConfig = shadowConfig
          )

  private def buildLikelihoodMatrix(
      observations: Seq[(PokerAction, GameState)],
      actionModel: PokerActionModel,
      hypotheses: Vector[HoleCards]
  ): Array[Double] =
    if actionModel.featureDimension == PokerFeatures.dimension then
      buildLikelihoodMatrixFiveFeature(observations, actionModel, hypotheses)
    else
      buildLikelihoodMatrixGeneric(observations, actionModel, hypotheses)

  private def buildLikelihoodMatrixGeneric(
      observations: Seq[(PokerAction, GameState)],
      actionModel: PokerActionModel,
      hypotheses: Vector[HoleCards]
  ): Array[Double] =
    val observationCount = observations.length
    val hypothesisCount = hypotheses.length
    val out = new Array[Double](observationCount * hypothesisCount)
    var observationIdx = 0
    while observationIdx < observationCount do
      val (action, state) = observations(observationIdx)
      val rowOffset = observationIdx * hypothesisCount
      var hypothesisIdx = 0
      while hypothesisIdx < hypothesisCount do
        out(rowOffset + hypothesisIdx) = actionModel.likelihood(action, state, hypotheses(hypothesisIdx))
        hypothesisIdx += 1
      observationIdx += 1
    out

  private def buildLikelihoodMatrixFiveFeature(
      observations: Seq[(PokerAction, GameState)],
      actionModel: PokerActionModel,
      hypotheses: Vector[HoleCards]
  ): Array[Double] =
    val observationCount = observations.length
    val hypothesisCount = hypotheses.length
    val out = new Array[Double](observationCount * hypothesisCount)

    val classCount = actionModel.logistic.weights.length
    val bias = actionModel.logistic.bias
    val w0 = new Array[Double](classCount)
    val w1 = new Array[Double](classCount)
    val w2 = new Array[Double](classCount)
    val w3 = new Array[Double](classCount)
    val w4 = new Array[Double](classCount)
    var cls = 0
    while cls < classCount do
      val row = actionModel.logistic.weights(cls)
      w0(cls) = row(0)
      w1(cls) = row(1)
      w2(cls) = row(2)
      w3(cls) = row(3)
      w4(cls) = row(4)
      cls += 1

    val boardStrengthCache = mutable.HashMap.empty[Board, Array[Double]]

    var observationIdx = 0
    while observationIdx < observationCount do
      val (action, state) = observations(observationIdx)
      val actionIndex = actionModel.categoryIndex.getOrElse(
        action.category,
        throw new IllegalArgumentException(s"unknown action category: ${action.category}")
      )
      val strengths = boardStrengthCache.getOrElseUpdate(
        state.board, {
          val arr = new Array[Double](hypothesisCount)
          var i = 0
          while i < hypothesisCount do
            arr(i) = PokerFeatures.handStrengthProxy(state.board, hypotheses(i))
            i += 1
          arr
        }
      )

      val potOdds = state.potOdds
      val stackToPot = math.min(state.stackToPot, 10.0) / 10.0
      val streetOrdinal = state.street.ordinal.toDouble / 3.0
      val positionOrdinal = state.position.ordinal.toDouble / 7.0

      val rowOffset = observationIdx * hypothesisCount
      var hypothesisIdx = 0
      while hypothesisIdx < hypothesisCount do
        val handStrength = strengths(hypothesisIdx)
        var maxLogit = Double.NegativeInfinity
        var actionLogit = Double.NegativeInfinity

        cls = 0
        while cls < classCount do
          val logit =
            bias(cls) +
              (w0(cls) * potOdds) +
              (w1(cls) * stackToPot) +
              (w2(cls) * streetOrdinal) +
              (w3(cls) * positionOrdinal) +
              (w4(cls) * handStrength)
          if logit > maxLogit then maxLogit = logit
          if cls == actionIndex then actionLogit = logit
          cls += 1

        var sumExp = 0.0
        cls = 0
        while cls < classCount do
          val logit =
            bias(cls) +
              (w0(cls) * potOdds) +
              (w1(cls) * stackToPot) +
              (w2(cls) * streetOrdinal) +
              (w3(cls) * positionOrdinal) +
              (w4(cls) * handStrength)
          sumExp += math.exp(logit - maxLogit)
          cls += 1

        val probability = math.exp(actionLogit - maxLogit) / sumExp
        out(rowOffset + hypothesisIdx) = math.max(probability, MinLikelihood)
        hypothesisIdx += 1
      observationIdx += 1
    out

  private def nativeUpdate(
      backend: HoldemBayesNativeRuntime.Backend,
      selectedProvider: Provider,
      hypotheses: Vector[HoleCards],
      prior: Array[Double],
      likelihoods: Array[Double],
      observationCount: Int,
      hypothesisCount: Int
  ): Option[UpdateResult] =
    val outPosterior = new Array[Double](hypothesisCount)
    val outLogEvidence = Array(0.0d)
    HoldemBayesNativeRuntime.updatePosteriorInPlace(
      backend = backend,
      observationCount = observationCount,
      hypothesisCount = hypothesisCount,
      prior = prior,
      likelihoods = likelihoods,
      outPosterior = outPosterior,
      outLogEvidence = outLogEvidence
    ) match
      case Left(reason) =>
        GpuRuntimeSupport.log(
          s"native Bayesian ${backend.toString.toLowerCase} update unavailable: $reason"
        )
        None
      case Right(_) =>
        val weights = Map.newBuilder[HoleCards, Double]
        var idx = 0
        while idx < hypotheses.length do
          weights += hypotheses(idx) -> math.max(0.0, outPosterior(idx))
          idx += 1
        Some(
          UpdateResult(
            posterior = DiscreteDistribution(weights.result()).normalized,
            logEvidence = outLogEvidence(0),
            provider = selectedProvider
          )
        )

  private def scalaUpdate(
      hypotheses: Vector[HoleCards],
      prior: Array[Double],
      likelihoods: Array[Double],
      observationCount: Int,
      hypothesisCount: Int,
      provider: Provider
  ): UpdateResult =
    val eps = Probability.Eps
    val posterior = prior.clone()
    val priorSum = posterior.sum
    require(priorSum > eps, "prior probabilities must sum to a positive value")
    val invPrior = 1.0 / priorSum
    var idx = 0
    while idx < posterior.length do
      posterior(idx) = math.max(0.0, posterior(idx) * invPrior)
      idx += 1

    var logEvidence = 0.0
    var observationIdx = 0
    while observationIdx < observationCount do
      val rowOffset = observationIdx * hypothesisCount
      var evidence = 0.0
      idx = 0
      while idx < hypothesisCount do
        val updated = posterior(idx) * likelihoods(rowOffset + idx)
        posterior(idx) = updated
        evidence += updated
        idx += 1
      require(evidence > eps, "likelihoods produce zero evidence")
      val invEvidence = 1.0 / evidence
      idx = 0
      while idx < hypothesisCount do
        posterior(idx) = posterior(idx) * invEvidence
        idx += 1
      logEvidence += math.log(evidence)
      observationIdx += 1

    val weights = Map.newBuilder[HoleCards, Double]
    idx = 0
    while idx < hypotheses.length do
      weights += hypotheses(idx) -> posterior(idx)
      idx += 1

    UpdateResult(
      posterior = DiscreteDistribution(weights.result()).normalized,
      logEvidence = logEvidence,
      provider = provider
    )

  private def maybeValidateAgainstScala(
      preferred: UpdateResult,
      hypotheses: Vector[HoleCards],
      prior: Array[Double],
      likelihoods: Array[Double],
      observationCount: Int,
      hypothesisCount: Int,
      shadowConfig: ShadowConfig
  ): UpdateResult =
    if !shadowConfig.enabled || preferred.provider == Provider.Scala then preferred
    else
      try
        val scalaReference = scalaUpdate(
          hypotheses = hypotheses,
          prior = prior,
          likelihoods = likelihoods,
          observationCount = observationCount,
          hypothesisCount = hypothesisCount,
          provider = Provider.Scala
        )
        val drift = computePosteriorDrift(hypotheses, preferred, scalaReference)
        if withinShadowTolerance(drift, shadowConfig) then preferred
        else
          val driftMessage =
            f"Bayesian shadow mismatch vs scala (provider=${providerLabel(preferred.provider)} " +
              f"posteriorMaxAbsDiff=${drift.posteriorMaxAbsDiff}%.6g " +
              f"logEvidenceAbsDiff=${drift.logEvidenceAbsDiff}%.6g " +
              f"tolerancePosterior=${shadowConfig.posteriorMaxAbsDiff}%.6g " +
              f"toleranceLogEvidence=${shadowConfig.logEvidenceMaxAbsDiff}%.6g)"
          if shadowConfig.failClosed then
            GpuRuntimeSupport.warn(s"$driftMessage; using scala result")
            scalaReference
          else
            GpuRuntimeSupport.warn(s"$driftMessage; keeping preferred native result")
            preferred
      catch
        case ex: Throwable =>
          val detail = describeThrowable(ex)
          if shadowConfig.failClosed then
            throw new IllegalStateException(
              s"Bayesian shadow validation failed in fail-closed mode: $detail",
              ex
            )
          else
            GpuRuntimeSupport.warn(
              s"Bayesian shadow validation unavailable ($detail); keeping preferred native result"
            )
            preferred

  private[holdem] def computePosteriorDrift(
      hypotheses: Vector[HoleCards],
      candidate: UpdateResult,
      reference: UpdateResult
  ): PosteriorDrift =
    var posteriorMaxAbsDiff = 0.0
    var idx = 0
    while idx < hypotheses.length do
      val hypothesis = hypotheses(idx)
      val diff = math.abs(
        candidate.posterior.probabilityOf(hypothesis) -
          reference.posterior.probabilityOf(hypothesis)
      )
      if diff > posteriorMaxAbsDiff then posteriorMaxAbsDiff = diff
      idx += 1
    PosteriorDrift(
      posteriorMaxAbsDiff = posteriorMaxAbsDiff,
      logEvidenceAbsDiff = math.abs(candidate.logEvidence - reference.logEvidence)
    )

  private def withinShadowTolerance(
      drift: PosteriorDrift,
      shadowConfig: ShadowConfig
  ): Boolean =
    drift.posteriorMaxAbsDiff <= shadowConfig.posteriorMaxAbsDiff &&
      drift.logEvidenceAbsDiff <= shadowConfig.logEvidenceMaxAbsDiff

  private def describeThrowable(ex: Throwable): String =
    Option(ex.getMessage)
      .map(_.trim)
      .filter(_.nonEmpty)
      .getOrElse(ex.getClass.getSimpleName)

  private def resolveConfiguredProvider(): Provider =
    GpuRuntimeSupport.resolveNonEmptyLower(ProviderProperty, ProviderEnv) match
      case Some("scala" | "jvm") =>
        Provider.Scala
      case Some("native-cpu" | "cpu") =>
        val availability = HoldemBayesNativeRuntime.availability(HoldemBayesNativeRuntime.Backend.Cpu)
        if availability.available then Provider.NativeCpu
        else
          GpuRuntimeSupport.warn(
            s"Bayesian native CPU provider unavailable (${availability.detail}); falling back to Scala"
          )
          Provider.Scala
      case Some("native-gpu" | "gpu" | "cuda") =>
        val gpuAvailability = HoldemBayesNativeRuntime.availability(HoldemBayesNativeRuntime.Backend.Gpu)
        if gpuAvailability.available then Provider.NativeGpu
        else
          val cpuAvailability = HoldemBayesNativeRuntime.availability(HoldemBayesNativeRuntime.Backend.Cpu)
          if cpuAvailability.available then
            GpuRuntimeSupport.warn(
              s"Bayesian native GPU provider unavailable (${gpuAvailability.detail}); using native CPU"
            )
            Provider.NativeCpu
          else
            GpuRuntimeSupport.warn(
              s"Bayesian native GPU provider unavailable (${gpuAvailability.detail}); falling back to Scala"
            )
            Provider.Scala
      case Some("auto") | None =>
        resolveAutoProvider()
      case Some(other) =>
        GpuRuntimeSupport.warn(s"unknown Bayesian provider '$other'; using auto selection")
        resolveAutoProvider()

  private def resolveAutoProvider(): Provider =
    autoChosenProviderRef.get() match
      case AutoSelection.Provider(provider) =>
        provider
      case AutoSelection.Unset =>
        val cpuAvailability = HoldemBayesNativeRuntime.availability(HoldemBayesNativeRuntime.Backend.Cpu)
        val gpuAvailability = HoldemBayesNativeRuntime.availability(HoldemBayesNativeRuntime.Backend.Gpu)
        val availableNative = Vector(
          if gpuAvailability.available then Some(Provider.NativeGpu) else None,
          if cpuAvailability.available then Some(Provider.NativeCpu) else None
        ).flatten

        val selected =
          if availableNative.isEmpty then Provider.Scala
          else
            val benchmarkRepetitions = configuredAutoBenchmarkRepetitions
            val synthetic = syntheticBenchmarkPayload()
            val scalaNanos = benchmarkNanos {
              benchmarkScala(
                synthetic = synthetic,
                repetitions = benchmarkRepetitions
              )
            }
            val nativeTimings = availableNative.flatMap { provider =>
              provider match
                case Provider.NativeCpu =>
                  benchmarkNativeProvider(
                    provider = provider,
                    benchmarkRepetitions = benchmarkRepetitions,
                    backend = HoldemBayesNativeRuntime.Backend.Cpu,
                    synthetic = synthetic
                  )
                case Provider.NativeGpu =>
                  benchmarkNativeProvider(
                    provider = provider,
                    benchmarkRepetitions = benchmarkRepetitions,
                    backend = HoldemBayesNativeRuntime.Backend.Gpu,
                    synthetic = synthetic
                  )
                case Provider.Scala =>
                  None
            }

            if nativeTimings.isEmpty then Provider.Scala
            else
              val (bestProvider, bestNativeNanos) = nativeTimings.minBy(_._2)
              val speedup = scalaNanos.toDouble / bestNativeNanos.toDouble
              if speedup >= configuredAutoMinSpeedup then
                GpuRuntimeSupport.log(
                  f"Bayesian auto-provider selected ${providerLabel(bestProvider)} " +
                    f"(scala=${scalaNanos / 1e6}%.2fms native=${bestNativeNanos / 1e6}%.2fms speedup=${speedup}%.2fx)"
                )
                bestProvider
              else
                GpuRuntimeSupport.log(
                  f"Bayesian auto-provider kept scala " +
                    f"(best=${providerLabel(bestProvider)} scala=${scalaNanos / 1e6}%.2fms " +
                    f"bestNative=${bestNativeNanos / 1e6}%.2fms speedup=${speedup}%.2fx)"
                )
                Provider.Scala

        autoChosenProviderRef.compareAndSet(AutoSelection.Unset, AutoSelection.Provider(selected))
        autoChosenProviderRef.get() match
          case AutoSelection.Provider(provider) => provider
          case AutoSelection.Unset => selected

  private final case class SyntheticPayload(
      hypotheses: Vector[HoleCards],
      prior: Array[Double],
      observations: Vector[(PokerAction, GameState)],
      actionModel: PokerActionModel
  ):
    val observationCount: Int = observations.length
    val hypothesisCount: Int = hypotheses.length

  private def syntheticBenchmarkPayload(): SyntheticPayload =
    val board = Board.empty
    val hypotheses = HoldemCombinator.holeCardsFrom(Deck.full).take(512)

    val rng = new Random(17L)
    val prior = Array.fill(hypotheses.length)(0.01 + rng.nextDouble())

    val classCount = PokerAction.categories.length
    val featureCount = PokerFeatures.dimension
    val weights = Vector.tabulate(classCount) { c =>
      Vector.tabulate(featureCount) { f =>
        0.02 * (c + 1).toDouble * (f + 1).toDouble
      }
    }
    val bias = Vector.tabulate(classCount)(c => 0.05 * c.toDouble)
    val actionModel = PokerActionModel(
      logistic = MultinomialLogistic(weights, bias),
      categoryIndex = PokerActionModel.defaultCategoryIndex,
      featureDimension = PokerFeatures.dimension
    )

    val stateA = GameState(
      street = Street.Preflop,
      board = board,
      pot = 8.0,
      toCall = 2.0,
      position = Position.Button,
      stackSize = 100.0,
      betHistory = Vector.empty
    )
    val stateB = stateA.copy(toCall = 3.0, pot = 10.0)
    val observations = Vector(
      PokerAction.Raise(20.0) -> stateA,
      PokerAction.Call -> stateB
    )

    SyntheticPayload(
      hypotheses = hypotheses,
      prior = prior,
      observations = observations,
      actionModel = actionModel
    )

  private def benchmarkScala(
      synthetic: SyntheticPayload,
      repetitions: Int
  ): Unit =
    var rep = 0
    while rep < repetitions do
      val likelihoods = buildLikelihoodMatrix(
        synthetic.observations,
        synthetic.actionModel,
        synthetic.hypotheses
      )
      scalaUpdate(
        hypotheses = synthetic.hypotheses,
        prior = synthetic.prior,
        likelihoods = likelihoods,
        observationCount = synthetic.observationCount,
        hypothesisCount = synthetic.hypothesisCount,
        provider = Provider.Scala
      )
      rep += 1

  private def benchmarkNativeProvider(
      provider: Provider,
      benchmarkRepetitions: Int,
      backend: HoldemBayesNativeRuntime.Backend,
      synthetic: SyntheticPayload
  ): Option[(Provider, Long)] =
    val started = System.nanoTime()
    var rep = 0
    var success = true
    while rep < benchmarkRepetitions && success do
      val likelihoods = buildLikelihoodMatrix(
        synthetic.observations,
        synthetic.actionModel,
        synthetic.hypotheses
      )
      nativeUpdate(
        backend = backend,
        selectedProvider = provider,
        hypotheses = synthetic.hypotheses,
        prior = synthetic.prior,
        likelihoods = likelihoods,
        observationCount = synthetic.observationCount,
        hypothesisCount = synthetic.hypothesisCount
      ) match
        case Some(_) =>
          rep += 1
        case None =>
          success = false
    val elapsed = math.max(1L, System.nanoTime() - started)
    if success then Some(provider -> elapsed) else None

  private def benchmarkNanos(thunk: => Unit): Long =
    val started = System.nanoTime()
    thunk
    math.max(1L, System.nanoTime() - started)

  private def configuredAutoBenchmarkRepetitions: Int =
    GpuRuntimeSupport
      .resolveNonEmpty(AutoBenchmarkRepetitionsProperty, AutoBenchmarkRepetitionsEnv)
      .flatMap(_.toIntOption)
      .filter(_ > 0)
      .getOrElse(DefaultAutoBenchmarkRepetitions)

  private def configuredAutoMinSpeedup: Double =
    GpuRuntimeSupport
      .resolveNonEmpty(AutoMinSpeedupProperty, AutoMinSpeedupEnv)
      .flatMap(_.toDoubleOption)
      .filter(value => value > 1.0 && value.isFinite)
      .getOrElse(DefaultAutoMinSpeedup)

  private[holdem] def configuredShadowConfig(): ShadowConfig =
    ShadowConfig(
      enabled = configuredBoolean(ShadowEnabledProperty, ShadowEnabledEnv, default = false),
      failClosed = configuredBoolean(ShadowFailClosedProperty, ShadowFailClosedEnv, default = false),
      posteriorMaxAbsDiff = configuredNonNegativeFiniteDouble(
        ShadowPosteriorMaxAbsDiffProperty,
        ShadowPosteriorMaxAbsDiffEnv,
        default = DefaultShadowPosteriorMaxAbsDiff
      ),
      logEvidenceMaxAbsDiff = configuredNonNegativeFiniteDouble(
        ShadowLogEvidenceMaxAbsDiffProperty,
        ShadowLogEvidenceMaxAbsDiffEnv,
        default = DefaultShadowLogEvidenceMaxAbsDiff
      )
    )

  private def configuredBoolean(property: String, env: String, default: Boolean): Boolean =
    GpuRuntimeSupport
      .resolveNonEmpty(property, env)
      .map(GpuRuntimeSupport.parseTruthy)
      .getOrElse(default)

  private def configuredNonNegativeFiniteDouble(property: String, env: String, default: Double): Double =
    GpuRuntimeSupport
      .resolveNonEmpty(property, env)
      .flatMap(_.toDoubleOption)
      .filter(value => value >= 0.0 && java.lang.Double.isFinite(value))
      .getOrElse(default)
