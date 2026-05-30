package sicfun.holdem.bench

/** Version-stamped opponent fields for the P0 baseline. A field is a villain-pool
  * token string plus the deterministic hall arguments that pin a reproducible run.
  */
object CanonicalField:

  final case class Field(version: String, playerCount: Int, villainTokens: Vector[String]):
    require(villainTokens.length == playerCount - 1, "villain count must be playerCount - 1")

    /** Build CLI args for TexasHoldemPlayingHall.run. heroStyle in {adaptive,gto,strategic}. */
    def hallArgs(heroStyle: String, hands: Int, seed: Long, outDir: String): Array[String] =
      Array(
        s"--playerCount=$playerCount",
        s"--villainPool=${villainTokens.mkString(",")}",
        s"--heroStyle=$heroStyle",
        s"--hands=$hands",
        s"--seed=$seed",
        s"--outDir=$outDir",
        "--learnEveryHands=0",       // deterministic strategy: no online retraining
        "--heroExplorationRate=0.0", // no epsilon-greedy noise
        "--fullRing=true",           // all 8 villains always active
        "--saveTrainingTsv=false"
      )

  /** P0 canonical exploitable 9-max field: a recreational-leaning mix. */
  val NineMaxExploitable: Field = Field(
    version = "p0-field-v1",
    playerCount = 9,
    villainTokens = Vector("station", "station", "nit", "nit", "maniac", "maniac", "tag", "lag")
  )

  /** A trivially-exploitable field used to prove the harness can DETECT an edge (all calling stations). */
  val NineMaxRiggedStations: Field = Field(
    version = "p0-rigged-stations-v1",
    playerCount = 9,
    villainTokens = Vector.fill(8)("station")
  )
