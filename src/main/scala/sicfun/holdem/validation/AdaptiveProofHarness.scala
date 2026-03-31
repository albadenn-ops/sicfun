package sicfun.holdem.validation

import sicfun.holdem.runtime.TexasHoldemPlayingHall

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import java.time.Instant
import java.util.Locale
import scala.util.Random

/** 9-max adaptive proof harness orchestrator.
  *
  * Runs the adaptive hero against 6 leak-injected villains + 1 shallow-GTO (TAG) +
  * 1 validatable GTO control in 500-hand blocks via `TexasHoldemPlayingHall.run()`,
  * with seat reshuffling between blocks.
  */
object AdaptiveProofHarness:

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

  /** The 8-villain pool string for --villainPool. */
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

  final case class BlockResult(
      blockIndex: Int,
      seed: Long,
      heroPosition: String,
      hallSummary: TexasHoldemPlayingHall.HallSummary
  )

  final case class RunResult(
      config: Config,
      blocks: Vector[BlockResult]
  ):
    def totalHands: Int = blocks.map(_.hallSummary.handsPlayed).sum
    def aggregateHeroBbPer100: Double =
      val totalNet = blocks.map(_.hallSummary.heroNetChips).sum
      if totalHands > 0 then (totalNet / totalHands) * 100.0 else 0.0
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
      "perVillainBbPer100" -> ujson.Arr(opponents*)
    )
    ujson.write(json, indent = 2)

  private def formatReport(result: RunResult): String =
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
    sb.toString()

  private def roundTo2(value: Double): Double =
    math.round(value * 100.0) / 100.0

  def main(args: Array[String]): Unit =
    val config = parseArgs(args)
    val result = run(config)
    val runDir = writeOutputs(result, config.outputDir)
    println(s"\nOutputs written to: ${runDir.toAbsolutePath.normalize()}")

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
