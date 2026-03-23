package sicfun.holdem.cfr

import munit.FunSuite
import sicfun.holdem.cli.CliHelpers

import java.nio.charset.StandardCharsets
import java.nio.file.Files
import scala.jdk.CollectionConverters.*

class HoldemCfrTexasSolverJsonAdapterTest extends FunSuite:
  test("adapter extracts hero root policy from TexasSolver-style dump and passes external comparison") {
    val root = Files.createTempDirectory("holdem-cfr-texassolver-adapter-")
    try
      val referencePath = root.resolve("reference.json")
      val texasDir = root.resolve("texas")
      val providerPath = root.resolve("provider.json")
      Files.createDirectories(texasDir)

      Files.writeString(
        referencePath,
        """{
          |  "suiteName": "proof-suite",
          |  "spots": [
          |    {
          |      "id": "river_spot",
          |      "state": {
          |        "street": "River",
          |        "position": "Button",
          |        "board": ["Ah", "Kd", "2c", "7s", "9h"],
          |        "pot": 18.0,
          |        "toCall": 0.0,
          |        "stackSize": 70.0,
          |        "betHistory": []
          |      },
          |      "hero": "AsKs",
          |      "villainRange": [
          |        { "hand": "QhQs", "probability": 0.6 },
          |        { "hand": "JdJh", "probability": 0.4 }
          |      ],
          |      "candidateActions": ["CHECK", "RAISE:8.000"],
          |      "policy": { "CHECK": 0.25, "RAISE:8.000": 0.75 },
          |      "actionEvs": { "CHECK": 0.01, "RAISE:8.000": 0.03 }
          |    }
          |  ]
          |}
          |""".stripMargin,
        StandardCharsets.UTF_8
      )

      Files.writeString(
        texasDir.resolve("river_spot.json"),
        """{
          |  "node_type": "action_node",
          |  "strategy": {
          |    "player": 1,
          |    "actions": ["CHECK", "BET 8.0"],
          |    "strategy": {
          |      "KsAs": [0.25, 0.75],
          |      "QhQs": [1.0, 0.0]
          |    }
          |  }
          |}
          |""".stripMargin,
        StandardCharsets.UTF_8
      )

      val adapted = HoldemCfrTexasSolverJsonAdapter.run(
        Array(
          s"--reference=$referencePath",
          s"--texasDir=$texasDir",
          "--expectedPlayer=1",
          s"--out=$providerPath"
        )
      )
      assert(adapted.isRight, s"expected adapter success, got $adapted")
      assert(Files.exists(providerPath), s"missing provider output: $providerPath")

      val providerJson = ujson.read(Files.readString(providerPath, StandardCharsets.UTF_8))
      assertEquals(providerJson("providerName").str, "TexasSolver")
      assertEquals(providerJson("spots").arr.length, 1)
      val providerSpot = providerJson("spots")(0)
      assertEquals(providerSpot("id").str, "river_spot")
      assertEquals(providerSpot("hero").str, CliHelpers.parseHoleCards("AsKs").toToken)
      assertEquals(providerSpot("candidateActions").arr.map(_.str).toVector, Vector("CHECK", "BET 8.0"))
      assertEqualsDouble(providerSpot("policy")("CHECK").num, 0.25, 1e-9)
      assertEqualsDouble(providerSpot("policy")("BET 8.0").num, 0.75, 1e-9)

      val compared = HoldemCfrExternalComparison.compareFiles(
        referencePath = referencePath,
        externalPath = providerPath,
        thresholds = HoldemCfrExternalComparison.Thresholds(
          maxMeanTvDistance = Some(0.0),
          maxSpotTvDistance = Some(0.0),
          minBestActionAgreement = Some(1.0),
          maxMeanEvRmse = None
        ),
        outDir = None
      )
      assert(compared.isRight, s"expected external comparison pass, got $compared")
    finally
      deleteRecursively(root)
  }

  test("adapter fails when TexasSolver dump is missing the reference hero hand") {
    val root = Files.createTempDirectory("holdem-cfr-texassolver-adapter-missing-hand-")
    try
      val referencePath = root.resolve("reference.json")
      val texasDir = root.resolve("texas")
      val providerPath = root.resolve("provider.json")
      Files.createDirectories(texasDir)

      Files.writeString(
        referencePath,
        """{
          |  "suiteName": "proof-suite",
          |  "spots": [
          |    {
          |      "id": "river_spot",
          |      "state": {
          |        "street": "River",
          |        "position": "Button",
          |        "board": ["Ah", "Kd", "2c", "7s", "9h"],
          |        "pot": 18.0,
          |        "toCall": 0.0,
          |        "stackSize": 70.0,
          |        "betHistory": []
          |      },
          |      "hero": "AsKs",
          |      "villainRange": [
          |        { "hand": "QhQs", "probability": 1.0 }
          |      ],
          |      "candidateActions": ["CHECK", "RAISE:8.000"],
          |      "policy": { "CHECK": 0.25, "RAISE:8.000": 0.75 }
          |    }
          |  ]
          |}
          |""".stripMargin,
        StandardCharsets.UTF_8
      )

      Files.writeString(
        texasDir.resolve("river_spot.json"),
        """{
          |  "strategy": {
          |    "player": 1,
          |    "actions": ["CHECK", "BET 8.0"],
          |    "strategy": {
          |      "QhQs": [1.0, 0.0]
          |    }
          |  }
          |}
          |""".stripMargin,
        StandardCharsets.UTF_8
      )

      val adapted = HoldemCfrTexasSolverJsonAdapter.run(
        Array(
          s"--reference=$referencePath",
          s"--texasDir=$texasDir",
          s"--out=$providerPath"
        )
      )
      assert(adapted.isLeft, s"expected adapter failure, got $adapted")
      assert(
        adapted.left.toOption.exists(_.contains(
          s"does not contain strategy for hero hand '${CliHelpers.parseHoleCards("AsKs").toToken}'"
        ))
      )
    finally
      deleteRecursively(root)
  }

  test("adapter fails when TexasSolver player does not match the expected actor index") {
    val root = Files.createTempDirectory("holdem-cfr-texassolver-adapter-player-mismatch-")
    try
      val referencePath = root.resolve("reference.json")
      val texasDir = root.resolve("texas")
      val providerPath = root.resolve("provider.json")
      Files.createDirectories(texasDir)

      Files.writeString(
        referencePath,
        """{
          |  "suiteName": "proof-suite",
          |  "spots": [
          |    {
          |      "id": "river_spot",
          |      "state": {
          |        "street": "River",
          |        "position": "Button",
          |        "board": ["Ah", "Kd", "2c", "7s", "9h"],
          |        "pot": 18.0,
          |        "toCall": 0.0,
          |        "stackSize": 70.0,
          |        "betHistory": []
          |      },
          |      "hero": "AsKs",
          |      "villainRange": [
          |        { "hand": "QhQs", "probability": 1.0 }
          |      ],
          |      "candidateActions": ["CHECK", "RAISE:8.000"],
          |      "policy": { "CHECK": 0.25, "RAISE:8.000": 0.75 }
          |    }
          |  ]
          |}
          |""".stripMargin,
        StandardCharsets.UTF_8
      )

      Files.writeString(
        texasDir.resolve("river_spot.json"),
        """{
          |  "strategy": {
          |    "player": 0,
          |    "actions": ["CHECK", "BET 8.0"],
          |    "strategy": {
          |      "KsAs": [0.25, 0.75]
          |    }
          |  }
          |}
          |""".stripMargin,
        StandardCharsets.UTF_8
      )

      val adapted = HoldemCfrTexasSolverJsonAdapter.run(
        Array(
          s"--reference=$referencePath",
          s"--texasDir=$texasDir",
          "--expectedPlayer=1",
          s"--out=$providerPath"
        )
      )
      assert(adapted.isLeft, s"expected adapter failure, got $adapted")
      assert(adapted.left.toOption.exists(_.contains("player mismatch: expected 1, found 0")))
    finally
      deleteRecursively(root)
  }

  private def deleteRecursively(path: java.nio.file.Path): Unit =
    if Files.exists(path) then
      val stream = Files.walk(path)
      try
        val all = stream.iterator().asScala.toVector.sortBy(_.toString.length).reverse
        all.foreach(Files.deleteIfExists)
      finally
        stream.close()
