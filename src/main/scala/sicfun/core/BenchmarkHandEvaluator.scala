package sicfun.core

import scala.util.Random

object BenchmarkHandEvaluator:
  private final case class Config(
      samples: Int,
      warmupRounds: Int,
      measureRounds: Int,
      seed: Long
  ):
    require(samples > 0, "samples must be positive")
    require(warmupRounds >= 0, "warmupRounds must be >= 0")
    require(measureRounds > 0, "measureRounds must be positive")

  final case class Report(
      samples: Int,
      warmupRounds: Int,
      measureRounds: Int,
      mismatchCount: Int,
      evaluate5MeanMillis: Double,
      categorize5MeanMillis: Double,
      evaluate5HandsPerSecond: Double,
      categorize5HandsPerSecond: Double,
      speedupX: Double,
      evaluate5Checksum: Long,
      categorize5Checksum: Long
  )

  private final case class PreparedHands(
      cards: Array[Vector[Card]],
      r0: Array[Int],
      r1: Array[Int],
      r2: Array[Int],
      r3: Array[Int],
      r4: Array[Int],
      isFlush: Array[Boolean]
  )

  def main(args: Array[String]): Unit =
    run(args) match
      case Right(report) =>
        println(s"samples: ${report.samples}")
        println(s"warmupRounds: ${report.warmupRounds}")
        println(s"measureRounds: ${report.measureRounds}")
        println(s"mismatchCount: ${report.mismatchCount}")
        println(f"evaluate5MeanMillis: ${report.evaluate5MeanMillis}%.6f")
        println(f"categorize5MeanMillis: ${report.categorize5MeanMillis}%.6f")
        println(f"evaluate5HandsPerSecond: ${report.evaluate5HandsPerSecond}%.2f")
        println(f"categorize5HandsPerSecond: ${report.categorize5HandsPerSecond}%.2f")
        println(f"speedupX: ${report.speedupX}%.3f")
        println(s"evaluate5Checksum: ${report.evaluate5Checksum}")
        println(s"categorize5Checksum: ${report.categorize5Checksum}")
      case Left(error) =>
        System.err.println(error)
        sys.exit(1)

  def run(args: Array[String]): Either[String, Report] =
    parseArgs(args).map(execute)

  private def execute(config: Config): Report =
    val prepared = prepareHands(config.samples, config.seed)
    val mismatchCount = countMismatches(prepared)

    var warmupRound = 0
    while warmupRound < config.warmupRounds do
      runEvaluate5(prepared)
      runCategorize5(prepared)
      warmupRound += 1

    val evaluateTimes = new Array[Long](config.measureRounds)
    val categorizeTimes = new Array[Long](config.measureRounds)

    var evaluateChecksum = 0L
    var categorizeChecksum = 0L
    var measureRound = 0
    while measureRound < config.measureRounds do
      val evalStart = System.nanoTime()
      evaluateChecksum ^= runEvaluate5(prepared)
      val evalEnd = System.nanoTime()
      evaluateTimes(measureRound) = evalEnd - evalStart

      val catStart = System.nanoTime()
      categorizeChecksum ^= runCategorize5(prepared)
      val catEnd = System.nanoTime()
      categorizeTimes(measureRound) = catEnd - catStart
      measureRound += 1

    val evalMeanNanos = evaluateTimes.sum.toDouble / config.measureRounds.toDouble
    val catMeanNanos = categorizeTimes.sum.toDouble / config.measureRounds.toDouble
    val evalMeanMillis = evalMeanNanos / 1e6
    val catMeanMillis = catMeanNanos / 1e6
    val evalThroughput = config.samples.toDouble / (evalMeanNanos / 1e9)
    val catThroughput = config.samples.toDouble / (catMeanNanos / 1e9)
    val speedup = evalMeanNanos / catMeanNanos

    Report(
      samples = config.samples,
      warmupRounds = config.warmupRounds,
      measureRounds = config.measureRounds,
      mismatchCount = mismatchCount,
      evaluate5MeanMillis = evalMeanMillis,
      categorize5MeanMillis = catMeanMillis,
      evaluate5HandsPerSecond = evalThroughput,
      categorize5HandsPerSecond = catThroughput,
      speedupX = speedup,
      evaluate5Checksum = evaluateChecksum,
      categorize5Checksum = categorizeChecksum
    )

  private def prepareHands(samples: Int, seed: Long): PreparedHands =
    val deck = Deck.full.toArray
    val rng = new Random(seed)

    val cards = new Array[Vector[Card]](samples)
    val r0 = new Array[Int](samples)
    val r1 = new Array[Int](samples)
    val r2 = new Array[Int](samples)
    val r3 = new Array[Int](samples)
    val r4 = new Array[Int](samples)
    val isFlush = new Array[Boolean](samples)

    val idx = new Array[Int](5)
    var i = 0
    while i < samples do
      var j = 0
      while j < 5 do
        val candidate = rng.nextInt(deck.length)
        var duplicate = false
        var k = 0
        while k < j && !duplicate do
          if idx(k) == candidate then duplicate = true
          k += 1
        if !duplicate then
          idx(j) = candidate
          j += 1

      val c0 = deck(idx(0))
      val c1 = deck(idx(1))
      val c2 = deck(idx(2))
      val c3 = deck(idx(3))
      val c4 = deck(idx(4))
      cards(i) = Vector(c0, c1, c2, c3, c4)
      r0(i) = c0.rank.value
      r1(i) = c1.rank.value
      r2(i) = c2.rank.value
      r3(i) = c3.rank.value
      r4(i) = c4.rank.value
      isFlush(i) = c0.suit == c1.suit && c1.suit == c2.suit && c2.suit == c3.suit && c3.suit == c4.suit
      i += 1

    PreparedHands(cards, r0, r1, r2, r3, r4, isFlush)

  private def countMismatches(prepared: PreparedHands): Int =
    var mismatches = 0
    var i = 0
    while i < prepared.cards.length do
      val evaluateCategory = HandEvaluator.evaluate5(prepared.cards(i)).category
      val categorizeCategory = HandEvaluator.categorize5(
        prepared.r0(i),
        prepared.r1(i),
        prepared.r2(i),
        prepared.r3(i),
        prepared.r4(i),
        prepared.isFlush(i)
      )
      if evaluateCategory != categorizeCategory then mismatches += 1
      i += 1
    mismatches

  private def runEvaluate5(prepared: PreparedHands): Long =
    var checksum = 0L
    var i = 0
    while i < prepared.cards.length do
      checksum += HandEvaluator.evaluate5(prepared.cards(i)).category.ordinal
      i += 1
    checksum

  private def runCategorize5(prepared: PreparedHands): Long =
    var checksum = 0L
    var i = 0
    while i < prepared.cards.length do
      checksum += HandEvaluator.categorize5(
        prepared.r0(i),
        prepared.r1(i),
        prepared.r2(i),
        prepared.r3(i),
        prepared.r4(i),
        prepared.isFlush(i)
      ).ordinal
      i += 1
    checksum

  private def parseArgs(args: Array[String]): Either[String, Config] =
    if args.contains("--help") || args.contains("-h") then Left(usage)
    else
      parseOptions(args).flatMap { options =>
        for
          samples <- parseIntOption(options, "samples", 200_000)
          _ <- if samples > 0 then Right(()) else Left("--samples must be positive")
          warmupRounds <- parseIntOption(options, "warmupRounds", 1)
          _ <- if warmupRounds >= 0 then Right(()) else Left("--warmupRounds must be >= 0")
          measureRounds <- parseIntOption(options, "measureRounds", 3)
          _ <- if measureRounds > 0 then Right(()) else Left("--measureRounds must be positive")
          seed <- parseLongOption(options, "seed", 1L)
        yield Config(
          samples = samples,
          warmupRounds = warmupRounds,
          measureRounds = measureRounds,
          seed = seed
        )
      }

  private def parseOptions(args: Array[String]): Either[String, Map[String, String]] =
    args.foldLeft[Either[String, Map[String, String]]](Right(Map.empty)) { (accEither, token) =>
      accEither.flatMap { acc =>
        if !token.startsWith("--") then Left(s"invalid option format '$token'; expected --key=value")
        else
          val body = token.drop(2)
          val eq = body.indexOf('=')
          if eq <= 0 || eq == body.length - 1 then
            Left(s"invalid option format '$token'; expected --key=value")
          else
            val key = body.take(eq).trim
            val value = body.drop(eq + 1).trim
            if key.isEmpty then Left(s"invalid option key in '$token'")
            else Right(acc.updated(key, value))
      }
    }

  private def parseIntOption(options: Map[String, String], key: String, default: Int): Either[String, Int] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) =>
        try Right(raw.toInt)
        catch
          case _: NumberFormatException => Left(s"--$key must be a valid integer, got '$raw'")

  private def parseLongOption(options: Map[String, String], key: String, default: Long): Either[String, Long] =
    options.get(key) match
      case None => Right(default)
      case Some(raw) =>
        try Right(raw.toLong)
        catch
          case _: NumberFormatException => Left(s"--$key must be a valid long, got '$raw'")

  private val usage =
    """Usage: BenchmarkHandEvaluator [--key=value ...]
      |
      |Options:
      |  --samples=<int>          Number of random 5-card hands (default 200000)
      |  --warmupRounds=<int>     Warmup rounds before timing (default 1)
      |  --measureRounds=<int>    Timed rounds (default 3)
      |  --seed=<long>            RNG seed for deterministic samples (default 1)
      |""".stripMargin
