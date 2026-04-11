package sicfun.holdem.validation

import sicfun.holdem.runtime.TexasHoldemPlayingHall
import sicfun.holdem.strategic.bridge.BridgeManifest

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import java.time.Instant
import java.util.Locale
import scala.util.Random

/** 9-max adaptive proof harness orchestrator.
  *
  * Validates the adaptive hero engine in a full-ring (9-max) setting by running
  * it against a realistic villain pool: 6 leak-injected opponents (one per major
  * leak type), 1 shallow-GTO TAG, and 1 full-GTO control. Each "block" is a
  * 500-hand session played via `TexasHoldemPlayingHall.run()`, with the hero's
  * seat position randomized between blocks to ensure position-independence.
  *
  * The harness produces:
  *   - A `ground-truth.json` with per-villain bb/100 for downstream regression testing
  *   - A human-readable `report.txt` summarizing hero's overall and per-villain performance
  *   - Per-block outputs from the playing hall (hand histories, model artifacts, etc.)
  *
  * Design rationale:
  *   - Block-based execution allows incremental validation without committing to
  *     a single massive run; each block gets a fresh RNG seed derived from the
  *     master seed, so results are reproducible.
  *   - The villain pool string (`defaultVillainPool`) encodes leak types and severities
  *     that TexasHoldemPlayingHall's villain factory understands.
  *   - Hero position cycles through all 9 seats randomly to avoid positional bias.
  *
  * Entry point: `main(args)` or programmatic `run(Config())`.
  *
  * @see [[ValidationRunner]] for the heads-up profiling validation pipeline
  * @see [[TexasHoldemPlayingHall]] for the underlying simulation engine
  */
object AdaptiveProofHarness:

  /** Configuration for a proof harness run.
    *
    * @param handsPerBlock  number of hands per block (each block = one TexasHoldemPlayingHall session)
    * @param blocks         number of blocks to run (hero seat is randomized per block)
    * @param seed           master RNG seed — each block derives its own seed from this
    * @param budgetMs       per-decision time budget in milliseconds for the adaptive engine
    * @param bunchingTrials Monte Carlo trials for bunching-effect estimation
    * @param equityTrials   Monte Carlo trials for equity estimation
    * @param outputDir      root directory for all outputs
    * @param modelDir       optional pre-trained PokerActionModel artifact directory
    */
  final case class Config(
      handsPerBlock: Int = 500,
      blocks: Int = 1,
      seed: Long = System.currentTimeMillis(),
      budgetMs: Long = 50L,
      bunchingTrials: Int = 100,
      equityTrials: Int = 500,
      outputDir: Path = Paths.get("data/adaptive-proof"),
      modelDir: Option[Path] = None
  ):
    def totalHands: Int = handsPerBlock * blocks

  /** The 8-villain pool encoded as a comma-separated string for --villainPool.
    *
    * Format: "leak:<type>:<severity>" for leak-injected villains, "tag" for shallow-GTO,
    * "gto" for full-GTO control. The playing hall's villain factory parses this to
    * construct the appropriate opponent instances.
    */
  val defaultVillainPool: String =
    "leak:overfold:0.20,leak:overcall:0.25,leak:turnbluff:0.18,leak:passive:0.30,leak:prefloploose:0.22,leak:prefloptight:0.15,tag,gto"

  val villainNames: Vector[String] = Vector(
    "Villain01_overfold",
    "Villain02_overcall",
    "Villain03_turnbluff",
    "Villain04_passive",
    "Villain05_prefloploose",
    "Villain06_prefloptight",
    "Villain07_shallowgto",
    "Villain08_gto"
  )

  /** Result of a single block run: captures the block index, RNG seed used,
    * hero's table position, and the full HallSummary from the playing hall.
    */
  final case class BlockResult(
      blockIndex: Int,
      seed: Long,
      heroPosition: String,
      hallSummary: TexasHoldemPlayingHall.HallSummary
  )

  /** Aggregate result across all blocks, providing summary statistics.
    *
    * `aggregateHeroBbPer100` and `perVillainAggregateBbPer100` merge chip totals
    * across blocks to compute overall win-rates, normalized to bb/100 hands.
    */
  final case class RunResult(
      config: Config,
      blocks: Vector[BlockResult]
  ):
    def totalHands: Int = blocks.map(_.hallSummary.handsPlayed).sum
    /** Hero's aggregate win-rate in big blinds per 100 hands across all blocks. */
    def aggregateHeroBbPer100: Double =
      val totalNet = blocks.map(_.hallSummary.heroNetChips).sum
      if totalHands > 0 then (totalNet / totalHands) * 100.0 else 0.0
    /** Per-villain aggregate bb/100, merging chip deltas across all blocks. */
    def perVillainAggregateBbPer100: Map[String, Double] =
      val merged = scala.collection.mutable.HashMap.empty[String, Double].withDefaultValue(0.0)
      blocks.foreach { block =>
        block.hallSummary.perVillainNetChips.foreach { case (name, net) =>
          merged.update(name, merged(name) + net)
        }
      }
      merged.map { case (name, net) =>
        name -> (if totalHands > 0 then (net / totalHands) * 100.0 else 0.0)
      }.toMap

  private val heroPositions9Max = Vector(
    "UTG", "UTG1", "UTG2", "Middle", "Hijack", "Cutoff", "Button", "SmallBlind", "BigBlind"
  )

  /** Execute the proof harness: run `config.blocks` blocks of `config.handsPerBlock` hands each.
    *
    * For each block, a fresh RNG seed is derived from the master seed, hero position is
    * randomized, and `TexasHoldemPlayingHall.run()` is invoked with the full 9-max
    * villain pool. If any block fails, a RuntimeException is thrown immediately.
    *
    * @param config harness configuration (defaults to 1 block of 500 hands)
    * @return aggregated RunResult across all blocks
    */
  def run(config: Config = Config()): RunResult =
    val rng = new Random(config.seed)
    println("=== Adaptive Proof Harness (9-Max) ===")
    println(s"Blocks: ${config.blocks}")
    println(s"Hands per block: ${config.handsPerBlock}")
    println(s"Total hands: ${config.totalHands}")
    println()

    val results = (0 until config.blocks).map { blockIdx =>
      val blockSeed = rng.nextLong()
      val heroPos = heroPositions9Max(rng.nextInt(heroPositions9Max.size))
      println(s"[Block ${blockIdx + 1}/${config.blocks}] seed=$blockSeed heroPosition=$heroPos")

      val hallOutDir = config.outputDir.resolve(s"block-$blockIdx")
      val hallArgs = Array(
        s"--hands=${config.handsPerBlock}",
        s"--reportEvery=${config.handsPerBlock}",
        "--learnEveryHands=0",
        "--learningWindowSamples=50",
        s"--seed=$blockSeed",
        s"--outDir=$hallOutDir",
        "--playerCount=9",
        s"--heroPosition=$heroPos",
        "--heroStyle=adaptive",
        "--gtoMode=exact",
        s"--villainPool=$defaultVillainPool",
        "--heroExplorationRate=0.0",
        "--raiseSize=2.5",
        s"--bunchingTrials=${config.bunchingTrials}",
        s"--equityTrials=${config.equityTrials}",
        "--saveTrainingTsv=false",
        "--saveDdreTrainingTsv=false",
        "--saveReviewHandHistory=true",
        "--fullRing=true"
      ) ++ config.modelDir.map(p => s"--modelArtifactDir=$p").toArray

      val hallResult = TexasHoldemPlayingHall.run(hallArgs)
      hallResult match
        case Right(summary) =>
          println(f"  -> heroNetChips: ${summary.heroNetChips}%.2f  bb/100: ${summary.heroBbPer100}%.1f")
          BlockResult(blockIdx, blockSeed, heroPos, summary)
        case Left(error) =>
          throw new RuntimeException(s"Hall run failed at block $blockIdx: $error")
    }.toVector

    RunResult(config, results)

  /** Write ground-truth JSON and human-readable report to a timestamped run directory.
    *
    * Creates `<outputDir>/run-<epochSeconds>-<seed>/` containing:
    *   - `ground-truth.json` — machine-readable per-villain bb/100 for regression testing
    *   - `report.txt` — formatted summary printed to stdout and saved to disk
    *
    * @return the Path to the created run directory
    */
  def writeOutputs(result: RunResult, outputDir: Path): Path =
    val runDir = outputDir.resolve(s"run-${Instant.now().getEpochSecond}-${result.config.seed}")
    Files.createDirectories(runDir)

    val groundTruth = writeGroundTruthJson(result)
    Files.writeString(runDir.resolve("ground-truth.json"), groundTruth, StandardCharsets.UTF_8)

    val report = formatReport(result)
    Files.writeString(runDir.resolve("report.txt"), report, StandardCharsets.UTF_8)
    println()
    println(report)

    runDir

  /** Serialize the run result to a JSON string suitable for regression testing.
    *
    * Contains totalHands, per-block details (seed, hero position, bb/100), and
    * per-villain aggregate bb/100 sorted alphabetically by villain name.
    */
  private def writeGroundTruthJson(result: RunResult): String =
    val perVillain = result.perVillainAggregateBbPer100
    val opponents = perVillain.toVector.sortBy(_._1).map { case (name, bbPer100) =>
      ujson.Obj(
        "name" -> ujson.Str(name),
        "heroNetBbPer100" -> ujson.Num(roundTo2(bbPer100))
      )
    }
    val blocks = result.blocks.map { block =>
      ujson.Obj(
        "blockIndex" -> ujson.Num(block.blockIndex),
        "seed" -> ujson.Num(block.seed.toDouble),
        "heroPosition" -> ujson.Str(block.heroPosition),
        "heroBbPer100" -> ujson.Num(roundTo2(block.hallSummary.heroBbPer100))
      )
    }
    val json = ujson.Obj(
      "totalHands" -> ujson.Num(result.totalHands),
      "blocks" -> ujson.Arr(blocks*),
      "perVillainBbPer100" -> ujson.Arr(opponents*),
      "bridgeFidelity" -> ujson.Str(BridgeManifest.summary)
    )
    ujson.write(json, indent = 2)

  /** Format a human-readable report with overall and per-villain bb/100 numbers. */
  private[validation] def formatReport(result: RunResult): String =
    val sb = new StringBuilder
    sb.append("=== Adaptive Proof Report (9-Max) ===\n")
    sb.append(s"Seed: ${result.config.seed}\n")
    sb.append(s"Blocks: ${result.blocks.size}\n")
    sb.append(s"Total hands: ${result.totalHands}\n")
    sb.append(f"\nOverall hero bb/100: ${result.aggregateHeroBbPer100}%+.1f\n")
    sb.append("\nPer-villain hero bb/100:\n")
    result.perVillainAggregateBbPer100.toVector.sortBy(_._1).foreach { case (name, bb) =>
      sb.append(String.format(Locale.ROOT, "  %-28s %+.1f\n", name, bb))
    }
    sb.append(s"\n${BridgeManifest.summary}\n")
    sb.append("Bridge Fidelity: structural gaps = ")
    sb.append(BridgeManifest.structuralGaps.map(_.formalObject).mkString(", "))
    sb.append("\n")
    sb.toString()

  private def roundTo2(value: Double): Double =
    math.round(value * 100.0) / 100.0

  def main(args: Array[String]): Unit =
    val config = parseArgs(args)
    val result = run(config)
    val runDir = writeOutputs(result, config.outputDir)
    println(s"\nOutputs written to: ${runDir.toAbsolutePath.normalize()}")

  /** Parse CLI arguments into Config. Supports --hands, --blocks, --seed, --budget,
    * --output, and --model flags with simple prefix matching.
    */
  private def parseArgs(args: Array[String]): Config =
    var config = Config()
    args.foreach { arg =>
      if arg.startsWith("--hands=") then
        config = config.copy(handsPerBlock = arg.stripPrefix("--hands=").toInt)
      else if arg.startsWith("--blocks=") then
        config = config.copy(blocks = arg.stripPrefix("--blocks=").toInt)
      else if arg.startsWith("--seed=") then
        config = config.copy(seed = arg.stripPrefix("--seed=").toLong)
      else if arg.startsWith("--budget=") then
        config = config.copy(budgetMs = arg.stripPrefix("--budget=").toLong)
      else if arg.startsWith("--output=") then
        config = config.copy(outputDir = Paths.get(arg.stripPrefix("--output=")))
      else if arg.startsWith("--model=") then
        config = config.copy(modelDir = Some(Paths.get(arg.stripPrefix("--model="))))
    }
    config
