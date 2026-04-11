package sicfun.holdem.equity
import sicfun.holdem.types.*

import java.io.{DataInputStream, DataOutputStream, FileInputStream, InputStream}

/** Shared low-level binary I/O routines for reading and writing heads-up equity table files.
  *
  * Used by both [[HeadsUpEquityTableIO]] (full tables with Long keys) and
  * [[HeadsUpEquityCanonicalTableIO]] (canonical tables with Int keys). The header format
  * is identical; only the key serialization differs and is parameterized via callbacks.
  */
object HeadsUpEquityTableIOUtil:
  /** Writes the binary header (magic, version, metadata fields) to the output stream. */
  def writeHeader(out: DataOutputStream, meta: HeadsUpEquityTableMeta): Unit =
    out.writeInt(HeadsUpEquityTableFormat.Magic)
    out.writeInt(HeadsUpEquityTableFormat.Version)
    out.writeBoolean(meta.canonical)
    out.writeByte(HeadsUpEquityTableFormat.modeStringToCode(meta.mode))
    out.writeInt(meta.trials)
    out.writeLong(meta.seed)
    out.writeLong(meta.maxMatchups)
    out.writeLong(meta.totalMatchups)
    out.writeLong(meta.createdAtMillis)
    out.writeInt(meta.count)

  /** Reads and validates the binary header, returning the parsed metadata.
    *
    * @throws IllegalArgumentException if the magic number or version is invalid
    */
  def readHeader(in: DataInputStream): HeadsUpEquityTableMeta =
    val magic = in.readInt()
    require(magic == HeadsUpEquityTableFormat.Magic, s"invalid table magic: $magic")
    val version = in.readInt()
    require(version == HeadsUpEquityTableFormat.Version, s"unsupported table version: $version")
    val canonical = in.readBoolean()
    val modeCode = in.readByte().toInt
    val trials = in.readInt()
    val seed = in.readLong()
    val maxMatchups = in.readLong()
    val totalMatchups = in.readLong()
    val createdAtMillis = in.readLong()
    val count = in.readInt()
    HeadsUpEquityTableMeta(
      formatVersion = version,
      mode = HeadsUpEquityTableFormat.codeToModeString(modeCode),
      trials = trials,
      seed = seed,
      maxMatchups = maxMatchups,
      totalMatchups = totalMatchups,
      count = count,
      canonical = canonical,
      createdAtMillis = createdAtMillis
    )

  /** Writes all table entries (key + win/tie/loss/stderr doubles) to the output stream.
    *
    * @tparam K       key type (Long for full tables, Int for canonical tables)
    * @param out      target output stream
    * @param entries  key-value pairs to serialize
    * @param writeKey callback to write a single key (e.g. `_.writeLong(_)` or `_.writeInt(_)`)
    */
  def writeEntries[K](
      out: DataOutputStream,
      entries: Iterable[(K, EquityResultWithError)],
      writeKey: (DataOutputStream, K) => Unit
  ): Unit =
    entries.foreach { case (key, result) =>
      writeKey(out, key)
      out.writeDouble(result.win)
      out.writeDouble(result.tie)
      out.writeDouble(result.loss)
      out.writeDouble(result.stderr)
    }

  /** Reads `count` table entries from the input stream.
    *
    * @tparam K      key type
    * @param in      source input stream (positioned immediately after the header)
    * @param count   number of entries to read (from the header metadata)
    * @param readKey callback to read a single key (e.g. `_.readLong()` or `_.readInt()`)
    * @return a map from key to [[EquityResultWithError]]
    */
  def readEntries[K](
      in: DataInputStream,
      count: Int,
      readKey: DataInputStream => K
  ): Map[K, EquityResultWithError] =
    val map = scala.collection.mutable.Map.empty[K, EquityResultWithError]
    var i = 0
    while i < count do
      val key = readKey(in)
      val win = in.readDouble()
      val tie = in.readDouble()
      val loss = in.readDouble()
      val stderr = in.readDouble()
      map.put(key, EquityResultWithError(win, tie, loss, stderr))
      i += 1
    map.toMap

  /** Wraps a raw InputStream in a DataInputStream, ensuring close on exit via try/finally.
    * This is the base resource-management primitive; file and resource helpers delegate here.
    */
  def withDataInputStream[A](stream: InputStream)(f: DataInputStream => A): A =
    val in = new DataInputStream(stream)
    try f(in)
    finally in.close()

  /** Opens a file path as a DataInputStream, ensuring close on exit. */
  def withFileInputStream[A](path: String)(f: DataInputStream => A): A =
    withDataInputStream(new FileInputStream(path))(f)

  /** Opens a classpath resource as a DataInputStream, ensuring close on exit.
    *
    * @throws IllegalArgumentException if the resource is not found
    */
  def withResourceInputStream[A](resourcePath: String, owner: Class[?])(f: DataInputStream => A): A =
    val stream = Option(owner.getResourceAsStream(resourcePath))
      .getOrElse(throw new IllegalArgumentException(s"resource not found: $resourcePath"))
    withDataInputStream(stream)(f)
