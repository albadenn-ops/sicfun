ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "3.8.1"

val onnxRuntimeVersion = "1.19.2"
val postgresqlJdbcVersion = "42.7.10"
val embeddedPostgresVersion = "2.2.2"

lazy val root = (project in file("."))
  .settings(
    name := "sicfun",
    libraryDependencies ++= Seq(
      "org.scalameta" %% "munit" % "1.2.2" % Test,
      "io.zonky.test" % "embedded-postgres" % embeddedPostgresVersion % Test,
      "com.lihaoyi" %% "ujson" % "3.3.1",
      "com.microsoft.onnxruntime" % "onnxruntime" % onnxRuntimeVersion,
      "org.postgresql" % "postgresql" % postgresqlJdbcVersion
    ),
    scalacOptions ++= Seq(
      "-deprecation",
      "-feature",
      "-unchecked",
      "-Wunused:imports,privates,locals",
      "-Werror",
      "-new-syntax",
      "-indent"
    ),
    Test / scalacOptions -= "-Xfatal-warnings"
  )

lazy val headsUpTableMode = settingKey[String]("Heads-up equity table mode: exact or mc")
lazy val headsUpTableTrials = settingKey[Int]("Monte Carlo trials per matchup")
lazy val headsUpTableMaxMatchups = settingKey[Long]("Maximum number of matchups to generate")
lazy val headsUpTableSeed = settingKey[Long]("Random seed for heads-up table generation")
lazy val headsUpTableParallelism = settingKey[Int]("Parallel worker count for heads-up table generation")
lazy val headsUpTableBackend = settingKey[String]("Heads-up table compute backend: cpu or gpu")
lazy val headsUpTableAutoGenerate = settingKey[Boolean]("Auto-generate heads-up table during resource generation")
lazy val generateHeadsUpTable = taskKey[File]("Generate heads-up equity table resource")
lazy val headsUpCanonicalTableAutoGenerate = settingKey[Boolean]("Auto-generate canonical heads-up table during resource generation")
lazy val generateHeadsUpCanonicalTable = taskKey[File]("Generate canonical heads-up equity table resource")
lazy val gpuSmokeGate = taskKey[Unit]("Run the GPU smoke gate (requires provider availability + CUDA engine execution)")
lazy val gpuExactParityGate = taskKey[Unit]("Run exact native CPU vs CUDA parity gate on a small canonical slice")
lazy val nativeBuild = taskKey[Unit]("Rebuild native CPU DLLs via clang++ when sources are newer than outputs")

headsUpTableMode := "mc"
headsUpTableTrials := 200
headsUpTableMaxMatchups := 10000
headsUpTableSeed := 1L
headsUpTableParallelism := math.max(1, java.lang.Runtime.getRuntime.availableProcessors())
headsUpTableBackend := "gpu"
headsUpTableAutoGenerate := false
headsUpCanonicalTableAutoGenerate := false

generateHeadsUpTable := {
  val log = streams.value.log
  val out = (Compile / resourceManaged).value / "sicfun" / "heads-up-equity.bin"
  val meta = out.getParentFile / "heads-up-equity.meta"
  val mode = headsUpTableMode.value
  val trials = headsUpTableTrials.value
  val maxMatchups = headsUpTableMaxMatchups.value
  val seed = headsUpTableSeed.value
  val parallelism = headsUpTableParallelism.value
  val backend = headsUpTableBackend.value.trim.toLowerCase
  val stamp = s"$mode:$trials:$maxMatchups:$seed:$parallelism:$backend"
  val cp = (Compile / fullClasspath).value.files
  val run = new ForkRun(ForkOptions())

  if (out.exists() && meta.exists() && IO.read(meta) == stamp) {
    log.info(s"Using cached heads-up equity table at ${out.getAbsolutePath}")
    out
  } else {
    IO.createDirectory(out.getParentFile)
    val args = Seq(
      out.getAbsolutePath,
      mode,
      trials.toString,
      maxMatchups.toString,
      seed.toString,
      parallelism.toString,
      backend
    )
    val result = run.run("sicfun.holdem.tablegen.GenerateHeadsUpTable", cp, args, log)
    result match {
      case scala.util.Success(_) => ()
      case scala.util.Failure(err) =>
        sys.error(s"Failed to generate heads-up equity table (${err.getMessage})")
    }
    IO.write(meta, stamp)
    out
  }
}

Compile / resourceGenerators += Def.taskIf {
  if (headsUpTableAutoGenerate.value) Seq(generateHeadsUpTable.value)
  else Seq.empty[File]
}

Compile / resourceGenerators += Def.taskIf {
  if (headsUpCanonicalTableAutoGenerate.value) Seq(generateHeadsUpCanonicalTable.value)
  else Seq.empty[File]
}

generateHeadsUpCanonicalTable := {
  val log = streams.value.log
  val out = (Compile / resourceManaged).value / "sicfun" / "heads-up-equity-canonical.bin"
  val meta = out.getParentFile / "heads-up-equity-canonical.meta"
  val mode = headsUpTableMode.value
  val trials = headsUpTableTrials.value
  val maxMatchups = headsUpTableMaxMatchups.value
  val seed = headsUpTableSeed.value
  val parallelism = headsUpTableParallelism.value
  val backend = headsUpTableBackend.value.trim.toLowerCase
  val stamp = s"$mode:$trials:$maxMatchups:$seed:$parallelism:$backend"
  val cp = (Compile / fullClasspath).value.files
  val run = new ForkRun(ForkOptions())

  if (out.exists() && meta.exists() && IO.read(meta) == stamp) {
    log.info(s"Using cached canonical heads-up equity table at ${out.getAbsolutePath}")
    out
  } else {
    IO.createDirectory(out.getParentFile)
    val args = Seq(
      out.getAbsolutePath,
      mode,
      trials.toString,
      maxMatchups.toString,
      seed.toString,
      parallelism.toString,
      backend
    )
    val result = run.run("sicfun.holdem.tablegen.GenerateHeadsUpCanonicalTable", cp, args, log)
    result match {
      case scala.util.Success(_) => ()
      case scala.util.Failure(err) =>
        sys.error(s"Failed to generate canonical heads-up equity table (${err.getMessage})")
    }
    IO.write(meta, stamp)
    out
  }
}

nativeBuild := {
  val log = streams.value.log
  val nativeDir = baseDirectory.value / "src" / "main" / "native"
  val jniDir = nativeDir / "jni"
  val buildDir = nativeDir / "build"
  val buildScript = nativeDir / "build-windows-llvm.ps1"

  if (!buildScript.exists()) {
    log.warn(s"Native build script not found: $buildScript — skipping native build")
  } else {
    // Check if any .cpp/.hpp source is newer than the CPU DLLs built by build-windows-llvm.ps1
    val sources = (jniDir ** ("*.cpp" | "*.hpp")).get
    val cpuDllNames = Set(
      "sicfun_native_cpu.dll", "sicfun_cfr_native.dll", "sicfun_bayes_native.dll",
      "sicfun_ddre_native.dll", "sicfun_postflop_native.dll", "sicfun_pomcp_native.dll"
    )
    val cpuDlls = (buildDir * "*.dll").get.filter(f => cpuDllNames.contains(f.getName))
    val needsRebuild = cpuDlls.size < cpuDllNames.size || {
      val oldestDll = cpuDlls.map(_.lastModified()).min
      sources.exists(_.lastModified() > oldestDll)
    }
    if (needsRebuild) {
      log.info("Native sources changed — rebuilding DLLs via build-windows-llvm.ps1")
      val exitCode = scala.sys.process.Process(
        Seq("powershell", "-ExecutionPolicy", "Bypass", "-File", buildScript.getAbsolutePath),
        nativeDir
      ).!
      if (exitCode != 0) sys.error(s"Native build failed with exit code $exitCode")
      log.info("Native DLLs rebuilt successfully")
    } else {
      log.info("Native DLLs up to date — skipping rebuild")
    }
  }
}

Test / test := (Test / test).dependsOn(nativeBuild).value

gpuSmokeGate := {
  val log = streams.value.log
  val cp = (Compile / fullClasspath).value.files
  val run = new ForkRun(ForkOptions())
  val result = run.run("sicfun.holdem.bench.HeadsUpGpuSmokeGate", cp, Seq.empty, log)
  result match {
    case scala.util.Success(_) => ()
    case scala.util.Failure(err) =>
      sys.error(s"GPU smoke gate failed (${err.getMessage})")
  }
}

gpuExactParityGate := {
  val log = streams.value.log
  val cp = (Compile / fullClasspath).value.files
  val run = new ForkRun(ForkOptions())
  val result = run.run("sicfun.holdem.bench.HeadsUpGpuExactParityGate", cp, Seq.empty, log)
  result match {
    case scala.util.Success(_) => ()
    case scala.util.Failure(err) =>
      sys.error(s"GPU exact parity gate failed (${err.getMessage})")
  }
}
