package sicfun.holdem.equity

/** Metadata header for a serialized heads-up equity table file.
  *
  * Stored at the beginning of every binary table file and used by [[HeadsUpTableInfo]]
  * for inspection and by I/O routines for validation.
  *
  * @param formatVersion  binary format version (currently 1)
  * @param mode           computation mode string (`"exact"` or `"mc"`)
  * @param trials         number of Monte Carlo trials (0 for exact mode)
  * @param seed           random seed used for table generation
  * @param maxMatchups    cap on matchups requested at generation time
  * @param totalMatchups  total number of possible matchups (full or canonical)
  * @param count          actual number of entries stored in the file
  * @param canonical      `true` for [[HeadsUpEquityCanonicalTable]], `false` for [[HeadsUpEquityTable]]
  * @param createdAtMillis wall-clock time when the table was generated
  */
final case class HeadsUpEquityTableMeta(
    formatVersion: Int,
    mode: String,
    trials: Int,
    seed: Long,
    maxMatchups: Long,
    totalMatchups: Long,
    count: Int,
    canonical: Boolean,
    createdAtMillis: Long
)

/** Constants and helpers defining the binary format for heads-up equity table files.
  *
  * '''Binary layout:'''
  *   - 4 bytes: magic number (`0x53494655` = `"SIFU"`)
  *   - 4 bytes: format version (currently 1)
  *   - 1 byte: canonical flag
  *   - 1 byte: mode code (0 = exact, 1 = Monte Carlo)
  *   - 4 bytes: trial count
  *   - 8 bytes: seed
  *   - 8 bytes: max matchups
  *   - 8 bytes: total matchups
  *   - 8 bytes: created-at timestamp (epoch millis)
  *   - 4 bytes: entry count
  *   - Entries: (key + 4 doubles) repeated `count` times
  */
object HeadsUpEquityTableFormat:
  /** File magic number: ASCII `"SIFU"` (0x53 0x49 0x46 0x55). */
  val Magic: Int = 0x53494655 // "SIFU"
  val Version: Int = 1
  val ModeExact: Int = 0
  val ModeMonteCarlo: Int = 1

  def codeToModeString(code: Int): String =
    code match
      case ModeExact => "exact"
      case ModeMonteCarlo => "mc"
      case other => s"unknown($other)"

  def modeStringToCode(mode: String): Int =
    mode match
      case "exact" => ModeExact
      case _ => ModeMonteCarlo

  /** Computes the fraction of possible matchups that are stored in the table (0.0 to 1.0). */
  def coverage(meta: HeadsUpEquityTableMeta): Double =
    if meta.totalMatchups <= 0 then 0.0
    else meta.count.toDouble / meta.totalMatchups.toDouble
