package sicfun.holdem.cfr

import munit.FunSuite
import sicfun.holdem.types.TestSystemPropertyScope

import java.nio.charset.StandardCharsets
import java.nio.file.Files
import scala.jdk.CollectionConverters.*

class HoldemCfrExternalComparisonTest extends FunSuite:
  private def withSystemProperties(properties: Map[String, String])(thunk: => Unit): Unit =
    TestSystemPropertyScope.withSystemProperties(properties.toSeq.map { case (key, value) => key -> Some(value) }) {
      HoldemCfrNativeRuntime.resetLoadCacheForTests()
      HoldemCfrSolver.resetAutoProviderForTests()
      try thunk
      finally
        HoldemCfrNativeRuntime.resetLoadCacheForTests()
        HoldemCfrSolver.resetAutoProviderForTests()
    }

  test("compareDatasets computes TV distance and best-action agreement") {
    val reference = HoldemCfrExternalComparison.ParsedDataset(
      label = "sicfun",
      spots = Vector(
        HoldemCfrExternalComparison.ParsedSpot(
          id = "spot-1",
          spotSignature = Some("sig-1"),
          candidateActions = Vector("FOLD", "CALL", "RAISE:2.500"),
          policy = Map("FOLD" -> 0.10, "CALL" -> 0.60, "RAISE:2.500" -> 0.30),
          actionEvs = Map("FOLD" -> -1.0, "CALL" -> 0.25, "RAISE:2.500" -> 0.2),
          bestAction = "CALL",
          bestActions = Set("CALL")
        ),
        HoldemCfrExternalComparison.ParsedSpot(
          id = "spot-2",
          spotSignature = Some("sig-2"),
          candidateActions = Vector("CHECK", "RAISE:8.000"),
          policy = Map("CHECK" -> 0.55, "RAISE:8.000" -> 0.45),
          actionEvs = Map("CHECK" -> 0.05, "RAISE:8.000" -> 0.02),
          bestAction = "CHECK",
          bestActions = Set("CHECK")
        )
      )
    )
    val external = HoldemCfrExternalComparison.ParsedDataset(
      label = "provider-x",
      spots = Vector(
        HoldemCfrExternalComparison.ParsedSpot(
          id = "spot-1",
          spotSignature = Some("sig-1"),
          candidateActions = Vector("FOLD", "CALL", "RAISE:2.500"),
          policy = Map("FOLD" -> 0.05, "CALL" -> 0.65, "RAISE:2.500" -> 0.30),
          actionEvs = Map("FOLD" -> -1.1, "CALL" -> 0.28, "RAISE:2.500" -> 0.18),
          bestAction = "CALL",
          bestActions = Set("CALL")
        ),
        HoldemCfrExternalComparison.ParsedSpot(
          id = "spot-2",
          spotSignature = Some("sig-2"),
          candidateActions = Vector("CHECK", "RAISE:8.000"),
          policy = Map("CHECK" -> 0.40, "RAISE:8.000" -> 0.60),
          actionEvs = Map("CHECK" -> 0.03, "RAISE:8.000" -> 0.07),
          bestAction = "RAISE:8.000",
          bestActions = Set("RAISE:8.000")
        )
      )
    )

    val result = HoldemCfrExternalComparison.compareDatasets(
      reference = reference,
      external = external,
      thresholds = HoldemCfrExternalComparison.Thresholds(
        maxMeanTvDistance = Some(1.0),
        maxSpotTvDistance = None,
        minBestActionAgreement = None,
        maxMeanEvRmse = None
      ),
      outDir = None
    )

    assertEquals(result.aggregate.referenceSpotCount, 2)
    assertEquals(result.aggregate.externalSpotCount, 2)
    assertEquals(result.aggregate.matchedSpotCount, 2)
    assertEquals(result.aggregate.matchingSpotSignatureCount, 2)
    assertEquals(result.aggregate.bestActionAgreementCount, 1)
    assertEqualsDouble(result.aggregate.bestActionAgreementRate, 0.5, 1e-9)
    assertEqualsDouble(result.aggregate.meanTvDistance, 0.10, 1e-9)
    assertEqualsDouble(result.aggregate.maxTvDistance, 0.15, 1e-9)
    assert(result.aggregate.meanEvRmse.exists(_ > 0.0))
    assert(result.gate.passed)

    val spot2 = result.spotComparisons.find(_.id == "spot-2").getOrElse(fail("missing spot-2 comparison"))
    assertEquals(spot2.spotSignatureStatus, "match")
    assertEquals(spot2.referenceBestAction, "CHECK")
    assertEquals(spot2.externalBestAction, "RAISE:8.000")
    assert(!spot2.bestActionMatches)
    assertEqualsDouble(spot2.tvDistance, 0.15, 1e-9)
  }

  test("compareFiles enforces optional gates after writing outputs") {
    val root = Files.createTempDirectory("holdem-cfr-external-compare-gate-")
    try
      val referencePath = root.resolve("reference.json")
      val externalPath = root.resolve("external.json")
      val outDir = root.resolve("out")

      Files.writeString(
        referencePath,
        """{
          |  "suiteName": "sicfun",
          |  "spots": [
          |    {
          |      "id": "spot-1",
          |      "candidateActions": ["CHECK", "RAISE:10.500"],
          |      "policy": { "CHECK": 0.9, "RAISE:10.500": 0.1 },
          |      "bestAction": "CHECK",
          |      "actionEvs": { "CHECK": 0.2, "RAISE:10.500": 0.1 }
          |    }
          |  ]
          |}
          |""".stripMargin,
        StandardCharsets.UTF_8
      )
      Files.writeString(
        externalPath,
        """{
          |  "providerName": "provider-x",
          |  "spots": [
          |    {
          |      "id": "spot-1",
          |      "policy": { "CHECK": 0.1, "RAISE:10.500": 0.9 },
          |      "bestAction": "RAISE:10.500",
          |      "actionEvs": { "CHECK": -0.2, "RAISE:10.500": 0.3 }
          |    }
          |  ]
          |}
          |""".stripMargin,
        StandardCharsets.UTF_8
      )

      val result = HoldemCfrExternalComparison.compareFiles(
        referencePath = referencePath,
        externalPath = externalPath,
        thresholds = HoldemCfrExternalComparison.Thresholds(
          maxMeanTvDistance = Some(0.20),
          maxSpotTvDistance = None,
          minBestActionAgreement = Some(1.0),
          maxMeanEvRmse = None
        ),
        outDir = Some(outDir)
      )

      assert(result.isLeft, s"expected gate failure, got $result")
      val summaryPath = outDir.resolve("summary.txt")
      val comparisonPath = outDir.resolve(HoldemCfrExternalComparison.ComparisonJsonFileName)
      assert(Files.exists(summaryPath), s"missing summary output: $summaryPath")
      assert(Files.exists(comparisonPath), s"missing comparison output: $comparisonPath")

      val summary = Files.readAllLines(summaryPath, StandardCharsets.UTF_8).asScala.mkString("\n")
      assert(summary.contains("gate: FAIL"))

      val json = ujson.read(Files.readString(comparisonPath, StandardCharsets.UTF_8))
      assertEquals(json("gate")("passed").bool, false)
      assert(json("gate")("failures").arr.nonEmpty)
    finally
      deleteRecursively(root)
  }

  test("compareFiles fails when the external dataset misses reference spots") {
    val root = Files.createTempDirectory("holdem-cfr-external-compare-missing-")
    try
      val referencePath = root.resolve("reference.json")
      val externalPath = root.resolve("external.json")

      Files.writeString(
        referencePath,
        """{
          |  "suiteName": "sicfun",
          |  "spots": [
          |    {
          |      "id": "spot-1",
          |      "candidateActions": ["CHECK", "RAISE:8.000"],
          |      "policy": { "CHECK": 0.6, "RAISE:8.000": 0.4 }
          |    },
          |    {
          |      "id": "spot-2",
          |      "candidateActions": ["FOLD", "CALL"],
          |      "policy": { "FOLD": 0.2, "CALL": 0.8 }
          |    }
          |  ]
          |}
          |""".stripMargin,
        StandardCharsets.UTF_8
      )
      Files.writeString(
        externalPath,
        """{
          |  "providerName": "provider-x",
          |  "spots": [
          |    {
          |      "id": "spot-1",
          |      "policy": { "CHECK": 0.5, "RAISE:8.000": 0.5 }
          |    }
          |  ]
          |}
          |""".stripMargin,
        StandardCharsets.UTF_8
      )

      val result = HoldemCfrExternalComparison.compareFiles(
        referencePath = referencePath,
        externalPath = externalPath,
        thresholds = HoldemCfrExternalComparison.Thresholds(
          maxMeanTvDistance = Some(1.0),
          maxSpotTvDistance = None,
          minBestActionAgreement = None,
          maxMeanEvRmse = None
        ),
        outDir = None
      )

      assert(result.isLeft, s"expected missing-spot failure, got $result")
      assert(result.left.toOption.exists(_.contains("missing 1 reference spot")))
    finally
      deleteRecursively(root)
  }

  test("compareFiles can restrict the gate to selected spot ids") {
    val root = Files.createTempDirectory("holdem-cfr-external-compare-spotids-")
    try
      val referencePath = root.resolve("reference.json")
      val externalPath = root.resolve("external.json")

      Files.writeString(
        referencePath,
        """{
          |  "suiteName": "sicfun",
          |  "spots": [
          |    {
          |      "id": "spot-1",
          |      "candidateActions": ["CHECK", "RAISE:8.000"],
          |      "policy": { "CHECK": 0.6, "RAISE:8.000": 0.4 }
          |    },
          |    {
          |      "id": "spot-2",
          |      "candidateActions": ["FOLD", "CALL"],
          |      "policy": { "FOLD": 0.2, "CALL": 0.8 }
          |    }
          |  ]
          |}
          |""".stripMargin,
        StandardCharsets.UTF_8
      )
      Files.writeString(
        externalPath,
        """{
          |  "providerName": "provider-x",
          |  "spots": [
          |    {
          |      "id": "spot-1",
          |      "policy": { "CHECK": 0.6, "RAISE:8.000": 0.4 }
          |    }
          |  ]
          |}
          |""".stripMargin,
        StandardCharsets.UTF_8
      )

      val result = HoldemCfrExternalComparison.compareFiles(
        referencePath = referencePath,
        externalPath = externalPath,
        selectedSpotIds = Some(Set("spot-1")),
        thresholds = HoldemCfrExternalComparison.Thresholds(
          maxMeanTvDistance = Some(0.0),
          maxSpotTvDistance = Some(0.0),
          minBestActionAgreement = Some(1.0),
          maxMeanEvRmse = None
        ),
        outDir = None
      )

      assert(result.isRight, s"expected selected-spot comparison to pass, got $result")
      val runResult = result.toOption.getOrElse(fail("missing comparison result"))
      assertEquals(runResult.aggregate.referenceSpotCount, 1)
      assertEquals(runResult.aggregate.externalSpotCount, 1)
      assertEquals(runResult.aggregate.matchedSpotCount, 1)
      assertEquals(runResult.aggregate.unmatchedReferenceSpotIds, Vector.empty)
      assertEquals(runResult.aggregate.unmatchedExternalSpotIds, Vector.empty)
    finally
      deleteRecursively(root)
  }

  test("compareFiles rejects unknown selected spot ids") {
    val root = Files.createTempDirectory("holdem-cfr-external-compare-unknown-spotids-")
    try
      val referencePath = root.resolve("reference.json")
      val externalPath = root.resolve("external.json")

      Files.writeString(
        referencePath,
        """{
          |  "suiteName": "sicfun",
          |  "spots": [
          |    {
          |      "id": "spot-1",
          |      "candidateActions": ["CHECK", "RAISE:8.000"],
          |      "policy": { "CHECK": 0.6, "RAISE:8.000": 0.4 }
          |    }
          |  ]
          |}
          |""".stripMargin,
        StandardCharsets.UTF_8
      )
      Files.writeString(
        externalPath,
        """{
          |  "providerName": "provider-x",
          |  "spots": [
          |    {
          |      "id": "spot-1",
          |      "policy": { "CHECK": 0.6, "RAISE:8.000": 0.4 }
          |    }
          |  ]
          |}
          |""".stripMargin,
        StandardCharsets.UTF_8
      )

      val result = HoldemCfrExternalComparison.compareFiles(
        referencePath = referencePath,
        externalPath = externalPath,
        selectedSpotIds = Some(Set("spot-1", "spot-typo")),
        thresholds = HoldemCfrExternalComparison.Thresholds(
          maxMeanTvDistance = Some(0.0),
          maxSpotTvDistance = Some(0.0),
          minBestActionAgreement = Some(1.0),
          maxMeanEvRmse = None
        ),
        outDir = None
      )

      assert(result.isLeft, s"expected unknown-spot failure, got $result")
      assert(result.left.toOption.exists(_.contains("spot-typo")))
    finally
      deleteRecursively(root)
  }

  test("loadDataset rejects bestAction outside the declared action set") {
    val root = Files.createTempDirectory("holdem-cfr-external-compare-bad-bestaction-")
    try
      val path = root.resolve("bad.json")
      Files.writeString(
        path,
        """{
          |  "suiteName": "default",
          |  "spots": [
          |    {
          |      "id": "spot-1",
          |      "candidateActions": ["CHECK", "RAISE:8.000"],
          |      "policy": { "CHECK": 0.5, "RAISE:8.000": 0.5 },
          |      "bestAction": "CALL"
          |    }
          |  ]
          |}
          |""".stripMargin,
        StandardCharsets.UTF_8
      )

      val error = intercept[IllegalArgumentException] {
        HoldemCfrExternalComparison.loadDataset(path)
      }
      assert(error.getMessage.contains("spot 'spot-1' bestAction 'CALL' is not present in candidateActions/policy/actionEvs"))
    finally
      deleteRecursively(root)
  }

  test("loadDataset rejects bestAction inconsistent with actionEvs") {
    val root = Files.createTempDirectory("holdem-cfr-external-compare-inconsistent-bestaction-")
    try
      val path = root.resolve("bad.json")
      Files.writeString(
        path,
        """{
          |  "suiteName": "default",
          |  "spots": [
          |    {
          |      "id": "spot-1",
          |      "candidateActions": ["CHECK", "RAISE:8.000"],
          |      "policy": { "CHECK": 0.9, "RAISE:8.000": 0.1 },
          |      "actionEvs": { "CHECK": 0.02, "RAISE:8.000": 0.10 },
          |      "bestAction": "CHECK"
          |    }
          |  ]
          |}
          |""".stripMargin,
        StandardCharsets.UTF_8
      )

      val error = intercept[IllegalArgumentException] {
        HoldemCfrExternalComparison.loadDataset(path)
      }
      assert(error.getMessage.contains("is inconsistent with the reported actionEvs"))
    finally
      deleteRecursively(root)
  }

  test("loadDataset accepts SICFUN export shape, canonicalizes action labels, and derives spot signatures") {
    val root = Files.createTempDirectory("holdem-cfr-external-compare-parse-")
    try
      val path = root.resolve("external-comparison.json")
      Files.writeString(
        path,
        """{
          |  "suiteName": "default",
          |  "spots": [
          |    {
          |      "id": "spot-1",
          |      "state": {
          |        "street": "Flop",
          |        "position": "Button",
          |        "board": ["Ah", "Kd", "2c"],
          |        "pot": 8.0,
          |        "toCall": 0.0,
          |        "stackSize": 92.0,
          |        "betHistory": []
          |      },
          |      "hero": "AsKs",
          |      "villainRange": [
          |        { "hand": "QhQs", "probability": 0.6 },
          |        { "hand": "JdJh", "probability": 0.4 }
          |      ],
          |      "candidateActions": ["check", "bet 8"],
          |      "policy": { "check": 2.0, "bet 8": 1.0 },
          |      "actionEvs": { "check": 0.05, "bet 8": 0.08 }
          |    }
          |  ]
          |}
          |""".stripMargin,
        StandardCharsets.UTF_8
      )

      val dataset = HoldemCfrExternalComparison.loadDataset(path)
      assertEquals(dataset.label, "default")
      assertEquals(dataset.spots.map(_.id), Vector("spot-1"))
      assertEqualsDouble(dataset.spots.head.policy("CHECK"), 2.0 / 3.0, 1e-9)
      assertEqualsDouble(dataset.spots.head.policy("RAISE:8.000"), 1.0 / 3.0, 1e-9)
      assertEquals(dataset.spots.head.bestAction, "RAISE:8.000")
      assert(dataset.spots.head.spotSignature.nonEmpty)
    finally
      deleteRecursively(root)
  }

  test("compareDatasets treats tied best actions deterministically across action-order differences") {
    val reference = HoldemCfrExternalComparison.ParsedDataset(
      label = "sicfun",
      spots = Vector(
        HoldemCfrExternalComparison.ParsedSpot(
          id = "spot-1",
          spotSignature = Some("sig-1"),
          candidateActions = Vector("CHECK", "RAISE:8.000"),
          policy = Map("CHECK" -> 0.5, "RAISE:8.000" -> 0.5),
          actionEvs = Map.empty,
          bestAction = "CHECK",
          bestActions = Set("CHECK", "RAISE:8.000")
        )
      )
    )
    val external = HoldemCfrExternalComparison.ParsedDataset(
      label = "provider-x",
      spots = Vector(
        HoldemCfrExternalComparison.ParsedSpot(
          id = "spot-1",
          spotSignature = Some("sig-1"),
          candidateActions = Vector("RAISE:8.000", "CHECK"),
          policy = Map("CHECK" -> 0.5, "RAISE:8.000" -> 0.5),
          actionEvs = Map.empty,
          bestAction = "CHECK",
          bestActions = Set("CHECK", "RAISE:8.000")
        )
      )
    )

    val result = HoldemCfrExternalComparison.compareDatasets(
      reference = reference,
      external = external,
      thresholds = HoldemCfrExternalComparison.Thresholds(
        maxMeanTvDistance = Some(0.0),
        maxSpotTvDistance = Some(0.0),
        minBestActionAgreement = Some(1.0),
        maxMeanEvRmse = None
      ),
      outDir = None
    )

    assert(result.gate.passed)
    assertEquals(result.spotComparisons.head.referenceBestAction, "CHECK")
    assertEquals(result.spotComparisons.head.externalBestAction, "CHECK")
    assert(result.spotComparisons.head.bestActionMatches)
  }

  test("compareFiles accepts normalized provider labels and rejects state drift behind the same spot id") {
    withSystemProperties(Map("sicfun.cfr.provider" -> "scala")) {
      val root = Files.createTempDirectory("holdem-cfr-external-compare-e2e-")
      try
        val outDir = root.resolve("report")
        val suite = HoldemCfrApproximationReport.DefaultSuite.take(1)
        val reportResult = HoldemCfrApproximationReport.runSuite(
          suiteName = "proof-suite",
          spots = suite,
          cfrConfig = HoldemCfrConfig(
            iterations = 120,
            averagingDelay = 20,
            maxVillainHands = 12,
            equityTrials = 160,
            preferNativeBatch = false,
            rngSeed = 31L
          ),
          outDir = Some(outDir)
        )

        assert(reportResult.isRight, s"expected report success, got $reportResult")

        val referencePath = outDir.resolve(HoldemCfrApproximationReport.ExternalComparisonFileName)
        val referenceJson = ujson.read(Files.readString(referencePath, StandardCharsets.UTF_8))
        val referenceSpot = referenceJson("spots")(0)
        val providerPath = root.resolve("provider.json")

        val providerJson =
          ujson.Obj(
            "providerName" -> ujson.Str("provider-x"),
            "spots" -> ujson.Arr(
              ujson.Obj(
                "id" -> referenceSpot("id"),
                "spotSignature" -> referenceSpot("spotSignature"),
                "state" -> referenceSpot("state"),
                "hero" -> referenceSpot("hero"),
                "villainRange" -> referenceSpot("villainRange"),
                "candidateActions" -> ujson.Arr.from(
                  referenceSpot("candidateActions").arr.map(action => ujson.Str(providerLabelFor(action.str)))
                ),
                "policy" -> ujson.Obj.from(
                  referenceSpot("policy").obj.toVector.map { case (action, value) =>
                    providerLabelFor(action) -> value
                  }
                ),
                "actionEvs" -> ujson.Obj.from(
                  referenceSpot("actionEvs").obj.toVector.map { case (action, value) =>
                    providerLabelFor(action) -> value
                  }
                )
              )
            )
          )
        Files.writeString(providerPath, ujson.write(providerJson, indent = 2), StandardCharsets.UTF_8)

        val passResult = HoldemCfrExternalComparison.compareFiles(
          referencePath = referencePath,
          externalPath = providerPath,
          thresholds = HoldemCfrExternalComparison.Thresholds(
            maxMeanTvDistance = Some(0.0),
            maxSpotTvDistance = Some(0.0),
            minBestActionAgreement = Some(1.0),
            maxMeanEvRmse = Some(0.0)
          ),
          outDir = None
        )
        assert(passResult.isRight, s"expected normalized provider comparison to pass, got $passResult")

        val driftedProvider = providerJson.obj("spots").arr(0)
        driftedProvider("state")("pot") = ujson.Num(driftedProvider("state")("pot").num + 1.0)
        val driftedPath = root.resolve("provider-drifted.json")
        Files.writeString(
          driftedPath,
          ujson.write(providerJson, indent = 2),
          StandardCharsets.UTF_8
        )

        val driftedResult = HoldemCfrExternalComparison.compareFiles(
          referencePath = referencePath,
          externalPath = driftedPath,
          thresholds = HoldemCfrExternalComparison.Thresholds(
            maxMeanTvDistance = Some(0.0),
            maxSpotTvDistance = Some(0.0),
            minBestActionAgreement = Some(1.0),
            maxMeanEvRmse = Some(0.0)
          ),
          outDir = None
        )
        assert(driftedResult.isLeft, s"expected spot-signature failure, got $driftedResult")
        assert(driftedResult.left.toOption.exists(_.contains("spotSignature")))
      finally
        deleteRecursively(root)
    }
  }

  private def providerLabelFor(action: String): String =
    if action.startsWith("RAISE:") then s"bet ${action.stripPrefix("RAISE:")}"
    else action.toLowerCase(java.util.Locale.ROOT)

  private def deleteRecursively(path: java.nio.file.Path): Unit =
    if Files.exists(path) then
      val stream = Files.walk(path)
      try
        val all = stream.iterator().asScala.toVector.sortBy(_.toString.length).reverse
        all.foreach(Files.deleteIfExists)
      finally
        stream.close()
