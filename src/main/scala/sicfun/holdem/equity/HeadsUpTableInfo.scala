package sicfun.holdem.equity

import sicfun.holdem.types.ConsoleLogger

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
    val log = ConsoleLogger.stdout()
    if args.length < 1 then
      log.error("Usage: HeadsUpTableInfo <pathToTableFile>")
      sys.exit(1)

    val path = args(0)
    val file = new File(path).getAbsoluteFile
    if !file.exists() then
      log.error(s"File not found: $path")
      sys.exit(2)

    val in = new DataInputStream(new FileInputStream(file))
    try
      val meta = HeadsUpEquityTableIOUtil.readHeader(in)
      val coverage = HeadsUpEquityTableFormat.coverage(meta)
      val createdAt = Instant.ofEpochMilli(meta.createdAtMillis)

      log.info(s"path: ${file.getAbsolutePath}")
      log.info(s"canonical: ${meta.canonical}")
      log.info(s"formatVersion: ${meta.formatVersion}")
      log.info(s"mode: ${meta.mode}")
      log.info(s"trials: ${meta.trials}")
      log.info(s"seed: ${meta.seed}")
      log.info(s"maxMatchups: ${meta.maxMatchups}")
      log.info(s"totalMatchups: ${meta.totalMatchups}")
      log.info(s"count: ${meta.count}")
      log.info(f"coverage: ${coverage * 100.0}%.4f%%")
      log.info(s"createdAt: $createdAt")
    finally in.close()
