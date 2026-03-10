package sicfun.holdem.tablegen
import sicfun.holdem.*
import sicfun.holdem.equity.*
import sicfun.holdem.gpu.*

/** Diagnostic tool that proves which concrete device each native backend call uses.
  *
  * It prints discovered CUDA/OpenCL device descriptors and executes one direct
  * batch on a chosen CUDA device and one direct batch on a chosen OpenCL device.
  */
object DeviceProofRun:

  def main(args: Array[String]): Unit =
    val maxMatchups = args.headOption.flatMap(v => scala.util.Try(v.toLong).toOption).getOrElse(5000L)
    val trials = args.lift(1).flatMap(v => scala.util.Try(v.toInt).toOption).getOrElse(500)
    val seedBase = args.lift(2).flatMap(v => scala.util.Try(v.toLong).toOption).getOrElse(42L)

    println("=== Device Proof Run ===")
    println(s"config: maxMatchups=$maxMatchups, trials=$trials, seedBase=$seedBase")

    val hybridDevices = HeadsUpHybridDispatcher.devices
    println("hybrid.devices:")
    hybridDevices.foreach { d =>
      println(s"  ${d.id} kind=${d.kind} name=${d.name}")
    }

    val cudaCount =
      try HeadsUpGpuNativeBindings.cudaDeviceCount()
      catch case ex: Throwable =>
        println(s"cudaDeviceCount failed: ${Option(ex.getMessage).getOrElse(ex.getClass.getSimpleName)}")
        0
    println(s"cuda.count=$cudaCount")
    (0 until cudaCount).foreach { idx =>
      val info = HeadsUpGpuNativeBindings.cudaDeviceInfo(idx)
      println(s"cuda[$idx]=$info")
    }

    val openclCount =
      try HeadsUpOpenCLNativeBindings.openclDeviceCount()
      catch case ex: Throwable =>
        println(s"openclDeviceCount failed: ${Option(ex.getMessage).getOrElse(ex.getClass.getSimpleName)}")
        0
    println(s"opencl.count=$openclCount")
    (0 until openclCount).foreach { idx =>
      val info = HeadsUpOpenCLNativeBindings.openclDeviceInfo(idx)
      println(s"opencl[$idx]=$info")
    }

    val batch = HeadsUpEquityCanonicalTable.selectCanonicalBatch(maxMatchups)
    val n = batch.packedKeys.length
    val low = new Array[Int](n)
    val high = new Array[Int](n)
    val seeds = new Array[Long](n)
    var i = 0
    while i < n do
      val packed = batch.packedKeys(i)
      low(i) = HeadsUpEquityTable.unpackLowId(packed)
      high(i) = HeadsUpEquityTable.unpackHighId(packed)
      seeds(i) = HeadsUpEquityTable.monteCarloSeed(seedBase, batch.keyMaterial(i))
      i += 1

    val cudaIdOpt = hybridDevices.find(_.kind == "cuda").flatMap(d => deviceIndexFromId(d.id))
    val openclIdOpt = hybridDevices.find(_.kind == "opencl").flatMap(d => deviceIndexFromId(d.id))

    def runCuda(label: String, deviceIndex: Int): Unit =
      val wins = new Array[Double](n)
      val ties = new Array[Double](n)
      val losses = new Array[Double](n)
      val stderrs = new Array[Double](n)
      val t0 = System.nanoTime()
      val status = HeadsUpGpuNativeBindings.computeBatchOnDevice(
        deviceIndex, low, high, 1, trials, seeds, wins, ties, losses, stderrs
      )
      val elapsedMs = (System.nanoTime() - t0) / 1_000_000.0
      val checksum = checksum64(wins, ties, losses, stderrs)
      val engineCode = HeadsUpGpuNativeBindings.lastEngineCode()
      println(
        f"$label: deviceIndex=$deviceIndex status=$status elapsedMs=$elapsedMs%.3f " +
          s"lastEngineCode=$engineCode checksum=$checksum"
      )

    def runOpenCL(label: String, deviceIndex: Int): Unit =
      val wins = new Array[Double](n)
      val ties = new Array[Double](n)
      val losses = new Array[Double](n)
      val stderrs = new Array[Double](n)
      val t0 = System.nanoTime()
      val status = HeadsUpOpenCLNativeBindings.computeBatch(
        deviceIndex, low, high, 1, trials, seeds, wins, ties, losses, stderrs
      )
      val elapsedMs = (System.nanoTime() - t0) / 1_000_000.0
      val checksum = checksum64(wins, ties, losses, stderrs)
      val engineCode = HeadsUpOpenCLNativeBindings.lastEngineCode()
      println(
        f"$label: deviceIndex=$deviceIndex status=$status elapsedMs=$elapsedMs%.3f " +
          s"lastEngineCode=$engineCode checksum=$checksum"
      )

    cudaIdOpt match
      case Some(cudaIndex) =>
        runCuda("cuda.run", cudaIndex)
      case None =>
        println("cuda.run: skipped (no CUDA device in hybrid discovery)")

    openclIdOpt match
      case Some(openclIndex) =>
        runOpenCL("opencl.run", openclIndex)
      case None =>
        println("opencl.run: skipped (no OpenCL device in hybrid discovery)")

    (0 until openclCount).foreach { idx =>
      if openclIdOpt.forall(_ != idx) then
        runOpenCL(s"opencl.direct[$idx].run", idx)
    }

    println("=== Done ===")

  private def deviceIndexFromId(id: String): Option[Int] =
    id.split(":", 2) match
      case Array(_, raw) => scala.util.Try(raw.toInt).toOption
      case _ => None

  private def checksum64(
      wins: Array[Double],
      ties: Array[Double],
      losses: Array[Double],
      stderrs: Array[Double]
  ): String =
    var acc = 0x9E3779B97F4A7C15L
    var i = 0
    while i < wins.length do
      acc ^= java.lang.Double.doubleToLongBits(wins(i)) + 0x9E3779B97F4A7C15L + (acc << 6) + (acc >>> 2)
      acc ^= java.lang.Double.doubleToLongBits(ties(i)) + 0x9E3779B97F4A7C15L + (acc << 6) + (acc >>> 2)
      acc ^= java.lang.Double.doubleToLongBits(losses(i)) + 0x9E3779B97F4A7C15L + (acc << 6) + (acc >>> 2)
      acc ^= java.lang.Double.doubleToLongBits(stderrs(i)) + 0x9E3779B97F4A7C15L + (acc << 6) + (acc >>> 2)
      i += 1
    java.lang.Long.toUnsignedString(acc, 16)
