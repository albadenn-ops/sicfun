package sicfun.holdem.tablegen
import sicfun.holdem.types.*
import sicfun.holdem.*
import sicfun.holdem.equity.*



import java.io.{BufferedWriter, File, FileWriter}
import java.util.Locale
import scala.collection.mutable

/** Exports a canonical heads-up equity table into a human-readable TSV with decoded hands.
  *
  * Usage:
  * {{{
  * HeadsUpCanonicalTableReadableDump <inputTablePath> <outputTsvPath> [sort:key|win|tie|loss|equity|stderr|hero|villain] [order:asc|desc] [limit]
  * }}}
  */
object HeadsUpCanonicalTableReadableDump:
  private final case class Row(
      key: Int,
      hero: String,
      villain: String,
      heroClass: String,
      villainClass: String,
      win: Double,
      tie: Double,
      loss: Double,
      equity: Double,
      stderr: Double
  )

  def main(args: Array[String]): Unit =
    if args.length < 2 then
      System.err.println(
        "Usage: HeadsUpCanonicalTableReadableDump <inputTablePath> <outputTsvPath> [sort:key|win|tie|loss|equity|stderr|hero|villain] [order:asc|desc] [limit]"
      )
      sys.exit(1)

    val inputPath = args(0)
    val outputPath = args(1)
    val sortBy = if args.length >= 3 then args(2).trim.toLowerCase(Locale.ROOT) else "equity"
    val order = if args.length >= 4 then args(3).trim.toLowerCase(Locale.ROOT) else "desc"
    val limit =
      if args.length >= 5 then
        val value = args(4).toInt
        require(value > 0, "limit must be positive")
        Some(value)
      else None

    val input = new File(inputPath).getAbsoluteFile
    if !input.isFile then
      System.err.println(s"Input file not found: ${input.getAbsolutePath}")
      sys.exit(2)

    val (table, meta) = HeadsUpEquityCanonicalTableIO.readWithMeta(input.getAbsolutePath)
    val reps = canonicalRepresentativesByKey()
    require(
      reps.size == HeadsUpEquityCanonicalTable.totalCanonicalKeys,
      s"internal canonical representative map mismatch: ${reps.size} != ${HeadsUpEquityCanonicalTable.totalCanonicalKeys}"
    )

    val rows = table.values.iterator.map { case (key, result) =>
      val (heroHand, villainHand) =
        reps.getOrElse(
          key,
          throw new IllegalStateException(s"missing representative for canonical key $key")
        )
      val win = result.win
      val tie = result.tie
      val loss = result.loss
      val stderr = result.stderr
      Row(
        key = key,
        hero = heroHand.toToken,
        villain = villainHand.toToken,
        heroClass = handClass(heroHand),
        villainClass = handClass(villainHand),
        win = win,
        tie = tie,
        loss = loss,
        equity = win + (tie / 2.0),
        stderr = stderr
      )
    }.toVector

    val sorted = sortRows(rows, sortBy, order)
    val finalRows = limit match
      case Some(n) => sorted.take(n)
      case None => sorted

    val output = new File(outputPath).getAbsoluteFile
    val parent = output.getParentFile
    if parent != null then parent.mkdirs()
    writeTsv(output, finalRows)

    println(s"input: ${input.getAbsolutePath}")
    println(s"output: ${output.getAbsolutePath}")
    println(s"rowsWritten: ${finalRows.length}")
    println(s"sourceCount: ${meta.count}")
    println(s"sort: $sortBy $order")
    limit.foreach(n => println(s"limit: $n"))

  private def canonicalRepresentativesByKey(): Map[Int, (HoleCards, HoleCards)] =
    val out = mutable.HashMap.empty[Int, (HoleCards, HoleCards)]
    HoleCardsIndex.foreachNonOverlappingPair { (_, hero, _, villain) =>
      val key = HeadsUpEquityCanonicalTable.keyFor(hero, villain)
      if !out.contains(key.value.raw) then
        // Stored canonical-table values are always in canonical hero perspective.
        val canonicalHero = if key.flipped then villain else hero
        val canonicalVillain = if key.flipped then hero else villain
        out.put(key.value.raw, (canonicalHero, canonicalVillain))
    }
    out.toMap

  private def sortRows(rows: Vector[Row], sortBy: String, order: String): Vector[Row] =
    val ascending =
      sortBy match
        case "key" => rows.sortBy(_.key)
        case "win" => rows.sortBy(_.win)
        case "tie" => rows.sortBy(_.tie)
        case "loss" => rows.sortBy(_.loss)
        case "equity" => rows.sortBy(_.equity)
        case "stderr" => rows.sortBy(_.stderr)
        case "hero" => rows.sortBy(_.hero)
        case "villain" => rows.sortBy(_.villain)
        case other =>
          throw new IllegalArgumentException(
            s"unknown sort '$other' (expected key|win|tie|loss|equity|stderr|hero|villain)"
          )
    order match
      case "asc" => ascending
      case "desc" => ascending.reverse
      case other =>
        throw new IllegalArgumentException(s"unknown order '$other' (expected asc or desc)")

  private def writeTsv(output: File, rows: Vector[Row]): Unit =
    val writer = new BufferedWriter(new FileWriter(output, false))
    try
      writer.write("rank\tkey\thero\tvillain\theroClass\tvillainClass\twin\ttie\tloss\tequity\tstderr")
      writer.newLine()
      var idx = 0
      while idx < rows.length do
        val row = rows(idx)
        val line =
          s"${idx + 1}\t${row.key}\t${row.hero}\t${row.villain}\t${row.heroClass}\t${row.villainClass}\t" +
            s"${fmt(row.win)}\t${fmt(row.tie)}\t${fmt(row.loss)}\t${fmt(row.equity)}\t${fmt(row.stderr)}"
        writer.write(line)
        writer.newLine()
        idx += 1
    finally writer.close()

  private def handClass(hand: HoleCards): String =
    val r1 = hand.first.rank
    val r2 = hand.second.rank
    val c1 = r1.toChar
    val c2 = r2.toChar
    if r1 == r2 then
      s"$c1$c2"
    else
      val (high, low) =
        if r1.value >= r2.value then (c1, c2)
        else (c2, c1)
      val suited = if hand.first.suit == hand.second.suit then "s" else "o"
      s"$high$low$suited"

  private def fmt(value: Double): String =
    String.format(Locale.ROOT, "%.12f", java.lang.Double.valueOf(value))
