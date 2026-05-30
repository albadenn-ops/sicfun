package sicfun.holdem.bench

import sicfun.holdem.runtime.TexasHoldemPlayingHall

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}

/** Track A: run hall self-play vs a canonical field and report bb/100 with a bootstrap CI. */
object MultiwayWinRateBenchmark:

  final case class Result(
      fieldVersion: String,
      heroStyle: String,
      hands: Int,
      seed: Long,
      ci: WinRateStats.WinRateCI,
      heroBbPer100Aggregate: Double
  )

  /** equityTrials/bunchingTrials: None => hall defaults (full quality, for real baselines);
    * Some(n) => override (used by tests to keep runs fast). */
  def run(
      field: CanonicalField.Field,
      heroStyle: String,
      hands: Int,
      seed: Long,
      outDir: Path,
      resamples: Int = 2000,
      ciLevel: Double = 0.95,
      equityTrials: Option[Int] = None,
      bunchingTrials: Option[Int] = None
  ): Either[String, Result] =
    val baseArgs = field.hallArgs(heroStyle, hands, seed, outDir.toString)
    val extra = equityTrials.map(t => s"--equityTrials=$t").toVector ++ bunchingTrials.map(t => s"--bunchingTrials=$t").toVector
    val args = baseArgs ++ extra
    TexasHoldemPlayingHall.run(args).map { summary =>
      val ci = WinRateStats.bbPer100CI(summary.perHandHeroNet, resamples, ciLevel, seed)
      val result = Result(field.version, heroStyle, summary.handsPlayed, seed, ci, summary.heroBbPer100)
      writeSummary(result, outDir)
      result
    }

  private def writeSummary(r: Result, outDir: Path): Unit =
    val path = outDir.resolve("winrate-summary.txt")
    Files.createDirectories(outDir)
    val lines = Vector(
      s"fieldVersion: ${r.fieldVersion}",
      s"heroStyle: ${r.heroStyle}",
      s"hands: ${r.hands}",
      s"seed: ${r.seed}",
      f"bbPer100Point: ${r.ci.pointEstimate}%.4f",
      f"bbPer100Lower${(r.ci.ciLevel * 100).toInt}: ${r.ci.lower}%.4f",
      f"bbPer100Upper${(r.ci.ciLevel * 100).toInt}: ${r.ci.upper}%.4f",
      f"hallAggregateBbPer100: ${r.heroBbPer100Aggregate}%.4f",
      s"ciClearsZero: ${r.ci.lower > 0.0}"
    )
    Files.write(path, lines.mkString(System.lineSeparator()).getBytes(StandardCharsets.UTF_8))

  /** CLI: arg0=heroStyle (default strategic), arg1=hands (default 50000), arg2=seed (default 42), arg3=outDir. Full hall quality (no trial overrides). */
  def main(args: Array[String]): Unit =
    val heroStyle = args.headOption.getOrElse("strategic")
    val hands = args.lift(1).flatMap(_.toIntOption).getOrElse(50000)
    val seed = args.lift(2).flatMap(_.toLongOption).getOrElse(42L)
    val outDir = Path.of(args.lift(3).getOrElse(s"data/p0-winrate/$heroStyle"))
    run(CanonicalField.NineMaxExploitable, heroStyle, hands, seed, outDir) match
      case Right(r) =>
        println(f"[$heroStyle] bb/100 = ${r.ci.pointEstimate}%.3f  CI${(r.ci.ciLevel*100).toInt}=[${r.ci.lower}%.3f, ${r.ci.upper}%.3f]  (n=${r.ci.sampleSize})")
      case Left(err) =>
        System.err.println(s"benchmark failed: $err"); sys.exit(1)
