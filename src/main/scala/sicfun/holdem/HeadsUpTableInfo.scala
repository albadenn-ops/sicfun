package sicfun.holdem

import java.io.{DataInputStream, File, FileInputStream}
import java.time.Instant

/** CLI utility for inspecting the metadata header of a serialized heads-up equity table file.
  *
  * Reads only the binary header (not the full table) and prints metadata fields including
  * format version, computation mode, trial count, seed, matchup counts, coverage percentage,
  * and creation timestamp.
  *
  * '''Usage:''' `HeadsUpTableInfo <pathToTableFile>`
  */
object HeadsUpTableInfo:
  def main(args: Array[String]): Unit =
    if args.length < 1 then
      System.err.println("Usage: HeadsUpTableInfo <pathToTableFile>")
      sys.exit(1)

    val path = args(0)
    val file = new File(path).getAbsoluteFile
    if !file.exists() then
      System.err.println(s"File not found: $path")
      sys.exit(2)

    val in = new DataInputStream(new FileInputStream(file))
    try
      val meta = HeadsUpEquityTableIOUtil.readHeader(in)
      val coverage = HeadsUpEquityTableFormat.coverage(meta)
      val createdAt = Instant.ofEpochMilli(meta.createdAtMillis)

      println(s"path: ${file.getAbsolutePath}")
      println(s"canonical: ${meta.canonical}")
      println(s"formatVersion: ${meta.formatVersion}")
      println(s"mode: ${meta.mode}")
      println(s"trials: ${meta.trials}")
      println(s"seed: ${meta.seed}")
      println(s"maxMatchups: ${meta.maxMatchups}")
      println(s"totalMatchups: ${meta.totalMatchups}")
      println(s"count: ${meta.count}")
      println(f"coverage: ${coverage * 100.0}%.4f%%")
      println(s"createdAt: $createdAt")
    finally in.close()
