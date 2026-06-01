package sicfun.holdem.runtime.agent

import java.io.{DataInputStream, DataOutputStream, File, FileInputStream, FileOutputStream}

final case class BlueprintNotFoundError(path: String)
    extends RuntimeException(s"blueprint not found: $path")
final case class BlueprintVersionMismatch(expected: String, actual: String)
    extends RuntimeException(s"abstraction hash mismatch: expected=$expected got=$actual")

final class BlueprintStore private (
    val header: BlueprintHeader,
    private val rows: Map[Long, Array[Float]]
):
  def lookup(infostateHash: Long): AbstractActionDistribution =
    rows.get(infostateHash) match
      case Some(arr) => AbstractActionDistribution(arr)
      case None =>
        AbstractActionDistribution(
          Array.fill(header.numAbstractActions)(1.0f / header.numAbstractActions)
        )

object BlueprintStore:

  def load(file: File, expectedAbstractionHash: String): BlueprintStore =
    if !file.exists() then throw BlueprintNotFoundError(file.getPath)
    val in = new DataInputStream(new FileInputStream(file))
    try
      val magicBytes = new Array[Byte](8)
      in.readFully(magicBytes)
      val magic = new String(magicBytes, "ASCII")
      val version = in.readInt()
      val trainedAt = in.readLong()
      val hashLen = in.readInt()
      val hashBytes = new Array[Byte](hashLen)
      in.readFully(hashBytes)
      val hash = new String(hashBytes, "UTF-8")
      if hash != expectedAbstractionHash then
        throw BlueprintVersionMismatch(expectedAbstractionHash, hash)
      val numSeats = in.readInt()
      val numInfostates = in.readInt()
      val numAbstractActions = in.readInt()
      val hdr = BlueprintHeader(magic, version, trainedAt, hash, numSeats, numInfostates, numAbstractActions)
      val rows = collection.mutable.Map[Long, Array[Float]]()
      (0 until numInfostates).foreach { _ =>
        val key = in.readLong()
        val arr = new Array[Float](numAbstractActions)
        (0 until numAbstractActions).foreach(i => arr(i) = in.readFloat())
        rows(key) = arr
      }
      new BlueprintStore(hdr, rows.toMap)
    finally in.close()

  def write(file: File, header: BlueprintHeader, rows: Map[Long, Array[Float]]): Unit =
    val out = new DataOutputStream(new FileOutputStream(file))
    try
      out.write(header.magic.getBytes("ASCII"))
      out.writeInt(header.version)
      out.writeLong(header.trainedAtEpochSeconds)
      val hashBytes = header.abstractionSpecHash.getBytes("UTF-8")
      out.writeInt(hashBytes.length)
      out.write(hashBytes)
      out.writeInt(header.numSeats)
      out.writeInt(rows.size)
      out.writeInt(header.numAbstractActions)
      rows.toVector.sortBy(_._1).foreach { case (k, arr) =>
        out.writeLong(k)
        require(arr.length == header.numAbstractActions,
          s"row length ${arr.length} != numAbstractActions ${header.numAbstractActions}")
        arr.foreach(out.writeFloat)
      }
    finally out.close()
