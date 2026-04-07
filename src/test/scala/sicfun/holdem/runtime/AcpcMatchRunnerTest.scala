package sicfun.holdem.runtime

import munit.FunSuite
import sicfun.core.Card
import sicfun.holdem.types.*

import java.io.{BufferedReader, BufferedWriter, InputStreamReader, OutputStreamWriter}
import java.net.ServerSocket
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*

/** Tests for [[AcpcActionCodec]] parsing/encoding and the [[AcpcMatchRunner]] integration.
  *
  * Covers the ACPC wire protocol round-trip: parsing MATCHSTATE strings into structured
  * states, encoding hero actions back into wire format, computing terminal hand values,
  * and running a single-hand session against a fake in-process dealer.
  *
  * The tests use fixed ACPC betting strings and MATCHSTATE lines to verify:
  *  - Correct seat mapping (button/big-blind) and chip contribution tracking.
  *  - Raise target accumulation across streets (cumulative vs. street-relative).
  *  - All-in call detection when encoded as a raise-to-stack token.
  *  - Terminal state valuation (fold and showdown).
  *  - End-to-end runner lifecycle (connect, play one hand, write output artifacts).
  */
class AcpcMatchRunnerTest extends FunSuite:
  override val munitTimeout: Duration = 120.seconds

  // ---- AcpcActionCodec unit tests ----

  test("empty ACPC betting gives button first preflop decision") {
    val parsed = AcpcActionCodec.parseBetting("", heroActual = 1, fullBoard = Board.empty).fold(fail(_), identity)
    val state = parsed.nextDecisionState.getOrElse(fail("expected decision state"))

    assertEquals(parsed.nextActorActual, 1)
    assertEquals(parsed.nextActorRelative, 0)
    assertEquals(parsed.steps.length, 0)
    assertEquals(state.street, Street.Preflop)
    assertEquals(state.position, Position.Button)
    assertEqualsDouble(state.pot, 1.5, 1e-9)
    assertEqualsDouble(state.toCall, 0.5, 1e-9)
    assertEqualsDouble(state.stackSize, 199.5, 1e-9)
  }

  test("ACPC button open is parsed as villain raise when hero is big blind") {
    val parsed = AcpcActionCodec.parseBetting("r200", heroActual = 0, fullBoard = Board.empty).fold(fail(_), identity)
    val state = parsed.nextDecisionState.getOrElse(fail("expected decision state"))

    assertEquals(parsed.steps.length, 1)
    val step = parsed.steps.head
    assertEquals(step.actualActor, 1)
    assertEquals(step.relativeActor, 1)
    assertEquals(step.action, PokerAction.Raise(1.0))
    assertEquals(step.stateBefore.position, Position.Button)
    assertEqualsDouble(step.stateBefore.toCall, 0.5, 1e-9)

    assertEquals(state.position, Position.BigBlind)
    assertEquals(state.street, Street.Preflop)
    assertEqualsDouble(state.pot, 3.0, 1e-9)
    assertEqualsDouble(state.toCall, 1.0, 1e-9)
    assertEqualsDouble(state.stackSize, 199.0, 1e-9)
  }

  test("ACPC terminal fold state is valued correctly") {
    val state = AcpcActionCodec.parseMatchState("MATCHSTATE:1:17:r300f:|AsAh").fold(fail(_), identity)
    val value = AcpcActionCodec.handValue(state).fold(fail(_), identity)
    assertEqualsDouble(value, 100.0, 1e-9)
  }

  test("ACPC terminal showdown state is valued correctly") {
    val state =
      AcpcActionCodec
        .parseMatchState("MATCHSTATE:1:18:cc/cc/cc/cc:KdQs|AsAh/2c7d9h/Ts/3d")
        .fold(fail(_), identity)
    val value = AcpcActionCodec.handValue(state).fold(fail(_), identity)
    assertEqualsDouble(value, 100.0, 1e-9)
  }

  test("ACPC postflop raise tokens are cumulative across the full hand") {
    val flop = Board.from(Vector("Ah", "8h", "Ts").map(parseCard))
    val parsed = AcpcActionCodec.parseBetting("r250c/c", heroActual = 1, fullBoard = flop).fold(fail(_), identity)
    val state = parsed.nextDecisionState.getOrElse(fail("expected hero decision state"))

    assertEquals(parsed.nextActorActual, 1)
    assertEquals(state.street, Street.Flop)
    assertEqualsDouble(state.pot, 5.0, 1e-9)
    assertEqualsDouble(state.toCall, 0.0, 1e-9)
    assertEquals(AcpcActionCodec.wireActionFor(parsed, PokerAction.Raise(15.0)), "r1750")
  }

  test("ACPC later-street all-in targets are parsed as total hand contribution") {
    val flop = Board.from(Vector("Ah", "8h", "Ts").map(parseCard))
    val parsed =
      AcpcActionCodec
        .parseBetting("r250r750r1500c/cr20000", heroActual = 0, fullBoard = flop)
        .fold(fail(_), identity)
    val state = parsed.nextDecisionState.getOrElse(fail("expected facing-decision state"))

    assertEquals(parsed.nextActorActual, 0)
    assertEquals(state.street, Street.Flop)
    assertEqualsDouble(state.pot, 215.0, 1e-9)
    assertEqualsDouble(state.toCall, 185.0, 1e-9)
    assertEqualsDouble(state.stackSize, 185.0, 1e-9)
    assertEquals(parsed.steps.last.action, PokerAction.Raise(185.0))
  }

  test("ACPC all-in calls may be encoded as raise-to-stack") {
    val river = Board.from(Vector("Ah", "8h", "Ts", "Jh", "3d").map(parseCard))
    val parsed =
      AcpcActionCodec
        .parseBetting("r300r600r1800r3600r10800r20000r20000", heroActual = 1, fullBoard = river)
        .fold(fail(_), identity)

    assert(parsed.handOver, "expected terminal all-in state")
    assertEquals(parsed.nextActorActual, -1)
    assertEquals(parsed.steps.last.action, PokerAction.Call)
    assertEquals(parsed.totalContributionByActualChips, Vector(20000, 20000))
  }

  // ---- End-to-end integration test with a fake in-process ACPC dealer ----

  test("ACPC runner completes a one-hand session against a fake dealer") {
    val root = Files.createTempDirectory("acpc-runner-test-")
    val out = root.resolve("out")
    val server = new ServerSocket(0)
    val dealerThread = new Thread(() => runFakeDealer(server))
    dealerThread.setDaemon(true)
    dealerThread.start()

    try
      val result = AcpcMatchRunner.run(Array(
        "--server=127.0.0.1",
        s"--port=${server.getLocalPort}",
        "--heroMode=adaptive",
        "--equityTrials=40",
        "--bunchingTrials=1",
        "--reportEvery=1",
        "--timeoutMillis=5000",
        "--connectTimeoutMillis=5000",
        s"--outDir=$out"
      ))
      assert(result.isRight, s"ACPC runner failed: $result")
      val summary = result.toOption.getOrElse(fail("missing summary"))
      assertEquals(summary.handsPlayed, 1)
      assert(Files.exists(out.resolve("hands.tsv")))
      assert(Files.exists(out.resolve("decisions.tsv")))
      assert(Files.exists(out.resolve("summary.txt")))

      val handRows = Files.readAllLines(out.resolve("hands.tsv"), StandardCharsets.UTF_8).asScala.toVector
      assert(handRows.length > 1, "expected hand log rows")
      val decisionRows = Files.readAllLines(out.resolve("decisions.tsv"), StandardCharsets.UTF_8).asScala.toVector
      assert(decisionRows.length > 1, "expected at least one decision row")
    finally
      try dealerThread.join(5000L)
      finally
        server.close()
        deleteRecursively(root)
  }

  /** Runs a minimal ACPC dealer on the given server socket.
    *
    * Accepts one connection, performs the VERSION handshake, sends an initial MATCHSTATE,
    * reads the client's action response, and sends a terminal state (either fold or showdown).
    * This simulates a single-hand dealer for integration testing.
    */
  private def runFakeDealer(server: ServerSocket): Unit =
    val socket = server.accept()
    val reader = new BufferedReader(new InputStreamReader(socket.getInputStream, StandardCharsets.US_ASCII))
    val writer = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream, StandardCharsets.US_ASCII))
    try
      val version = reader.readLine()
      assertEquals(version, "VERSION:2.0.0")
      sendLine(writer, "# fake dealer ready")
      val initial = "MATCHSTATE:1:0::|AsAh"
      sendLine(writer, initial)
      val response = reader.readLine()
      assert(response != null, "expected client response")
      assert(response.startsWith(s"$initial:"), s"unexpected response: $response")
      val action = response.drop(initial.length + 1)
      val terminal =
        if action == "c" then
          "MATCHSTATE:1:0:cc/cc/cc/cc:KdQs|AsAh/2c7d9h/Ts/3d"
        else if action == "f" then
          "MATCHSTATE:1:0:f:|AsAh"
        else if action.startsWith("r") then
          s"MATCHSTATE:1:0:${action}f:|AsAh"
        else
          fail(s"unexpected ACPC action from runner: $action")
      sendLine(writer, terminal)
    finally
      writer.close()
      reader.close()
      socket.close()

  private def sendLine(writer: BufferedWriter, line: String): Unit =
    writer.write(line)
    writer.write("\r\n")
    writer.flush()

  private def parseCard(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card token in test: $token"))

  private def deleteRecursively(path: Path): Unit =
    if Files.exists(path) then
      val stream = Files.walk(path)
      try
        val all = stream.iterator().asScala.toVector.sortBy(_.toString.length).reverse
        all.foreach(Files.deleteIfExists)
      finally
        stream.close()
