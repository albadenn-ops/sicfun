package sicfun.holdem.history

import munit.FunSuite
import sicfun.holdem.engine.RaiseResponseCounts

import java.nio.file.Files
import scala.jdk.CollectionConverters.*

class CollapseRememberedPlayersTest extends FunSuite:
  private def makeProfile(site: String, name: String, hands: Int): OpponentProfile =
    OpponentProfile(
      site = site,
      playerName = name,
      handsObserved = hands,
      firstSeenEpochMillis = 1_700_000_000_000L,
      lastSeenEpochMillis = 1_700_000_000_500L,
      actionSummary = OpponentActionSummary(folds = 4, raises = 3, calls = 3, checks = 2),
      raiseResponses = RaiseResponseCounts(folds = 1, calls = 2, raises = 1),
      recentEvents = Vector.empty,
      seenHandIds = Vector("h-1", "h-2")
    )

  private def withTempStore(store: OpponentProfileStore)(body: String => Unit): Unit =
    val dir = Files.createTempDirectory("collapse-test-")
    val path = dir.resolve("profiles.json")
    try
      OpponentProfileStore.save(path, store)
      body(path.toString)
    finally
      val stream = Files.walk(dir)
      try
        stream.iterator().asScala.toVector.sortBy(_.toString.length).reverse.foreach(p => Files.deleteIfExists(p))
      finally
        stream.close()

  test("run collapses two remembered players successfully") {
    val canonical = makeProfile("pokerstars", "VillainA", 20)
    val alias = makeProfile("gg", "VillainB", 10)
    val store = OpponentProfileStore.empty.upsert(canonical).upsert(alias)

    withTempStore(store) { storePath =>
      val result = CollapseRememberedPlayers.run(Array(
        s"--store=$storePath",
        "--canonicalSite=pokerstars",
        "--canonicalName=VillainA",
        "--aliasSite=gg",
        "--aliasName=VillainB"
      ))

      assert(result.isRight, s"expected success but got: $result")
      val summary = result.toOption.get
      assert(summary.canonicalPlayerUid.nonEmpty)
      assert(summary.aliasPlayerUid.nonEmpty)
      assertNotEquals(summary.canonicalPlayerUid, summary.aliasPlayerUid)
      assertEquals(summary.collapseProfiles, false)

      // Verify the store was actually saved with the collapse
      val reloaded = OpponentProfileStore.load(java.nio.file.Paths.get(storePath))
      assert(reloaded.playerCollapses.nonEmpty, "expected player collapses after save")
    }
  }

  test("run with collapseProfiles=true also collapses profiles") {
    val canonical = makeProfile("pokerstars", "VillainA", 20)
    val alias = makeProfile("gg", "VillainB", 10)
    val store = OpponentProfileStore.empty.upsert(canonical).upsert(alias)

    withTempStore(store) { storePath =>
      val result = CollapseRememberedPlayers.run(Array(
        s"--store=$storePath",
        "--canonicalSite=pokerstars",
        "--canonicalName=VillainA",
        "--aliasSite=gg",
        "--aliasName=VillainB",
        "--collapseProfiles=true"
      ))

      assert(result.isRight, s"expected success but got: $result")
      assertEquals(result.toOption.get.collapseProfiles, true)

      val reloaded = OpponentProfileStore.load(java.nio.file.Paths.get(storePath))
      assert(reloaded.profileCollapses.nonEmpty, "expected profile collapses after save")
    }
  }

  test("run fails when canonical player is not found") {
    val store = OpponentProfileStore.empty.upsert(makeProfile("gg", "VillainB", 10))

    withTempStore(store) { storePath =>
      val result = CollapseRememberedPlayers.run(Array(
        s"--store=$storePath",
        "--canonicalSite=pokerstars",
        "--canonicalName=VillainA",
        "--aliasSite=gg",
        "--aliasName=VillainB"
      ))

      assert(result.isLeft)
      assert(result.left.toOption.get.contains("canonical player not found"))
    }
  }

  test("run fails when alias player is not found") {
    val store = OpponentProfileStore.empty.upsert(makeProfile("pokerstars", "VillainA", 20))

    withTempStore(store) { storePath =>
      val result = CollapseRememberedPlayers.run(Array(
        s"--store=$storePath",
        "--canonicalSite=pokerstars",
        "--canonicalName=VillainA",
        "--aliasSite=gg",
        "--aliasName=VillainB"
      ))

      assert(result.isLeft)
      assert(result.left.toOption.get.contains("alias player not found"))
    }
  }

  test("run fails when required options are missing") {
    val result = CollapseRememberedPlayers.run(Array(
      "--store=/tmp/fake.json",
      "--canonicalSite=pokerstars"
    ))

    assert(result.isLeft)
  }

  test("run returns usage on --help") {
    val result = CollapseRememberedPlayers.run(Array("--help"))
    assert(result.isLeft)
    assert(result.left.toOption.get.contains("Usage"))
  }
end CollapseRememberedPlayersTest
