# scripts/release-hand-history-web-installer.ps1
# Wrapper around release-hand-history-web.ps1.
# Adds: jlink runtime image (item 2), launcher patch to prefer embedded runtime (item 3),
#       service-common.ps1 NSSM bundling patch (item 4), Setup.cmd click-to-run (item 5),
#       regen of manifest.sha256 (item 6), outer .zip + SHA-256 (item 7).
# Do not modify release-hand-history-web.ps1.

[CmdletBinding()]
param(
  # Passthrough to release-hand-history-web.ps1 - keep defaults identical
  [string]$OutputDir = "dist/hand-history-web",
  [string]$StaticDir = "docs/site-preview-hybrid",
  [string]$ModelDir = "",
  [int]$SmokePort = 18080,
  [int]$MaxUploadBytes = 2097152,
  [long]$AnalysisTimeoutMs = 120000,

  # Wrapper-only
  [string]$ZipOutputDir = "dist",
  [string]$Version = "",

  # JDK to use for jlink/jdeps (item 2). Resolved by absolute path; do NOT rely on JAVA_HOME or PATH.
  # Default = Eclipse Temurin 25 LTS installed locally; override with -JdkPath when swapping.
  [string]$JdkPath = "C:\Program Files\Eclipse Adoptium\jdk-25.0.2.10-hotspot"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Step {
  param(
    [string]$Label,
    [scriptblock]$Action
  )
  Write-Host "==> $Label"
  & $Action
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $repoRoot
try {
  # Step 0: Resolve and validate version (validation applies to BOTH parsed and -Version flag paths)
  Invoke-Step "Step 0: Resolve version" {
    if ([string]::IsNullOrWhiteSpace($Version)) {
      $buildSbtLine = Get-Content (Join-Path $repoRoot "build.sbt") -TotalCount 1
      $match = [regex]::Match($buildSbtLine, 'ThisBuild\s*/\s*version\s*:=\s*"([^"]+)"')
      if (-not $match.Success) {
        throw "Could not parse version from build.sbt line 1: $buildSbtLine"
      }
      $script:Version = $match.Groups[1].Value.Trim()
      Write-Host "  Resolved version from build.sbt: $($script:Version)"
    } else {
      $script:Version = $Version.Trim()
      Write-Host "  Using version from -Version flag: $($script:Version)"
    }
    if ([string]::IsNullOrWhiteSpace($script:Version)) {
      throw "Version is empty after normalization"
    }
    if ($script:Version -notmatch '^[A-Za-z0-9._-]+$') {
      throw "Version contains invalid characters: '$($script:Version)' (must be [A-Za-z0-9._-] only)"
    }
  }

  # Step 1: Invoke inner release
  Invoke-Step "Step 1: Invoke inner release script" {
    $innerScript = Join-Path $PSScriptRoot "release-hand-history-web.ps1"
    $innerArgs = @{
      OutputDir = $OutputDir
      StaticDir = $StaticDir
      ModelDir  = $ModelDir
      SmokePort = $SmokePort
      MaxUploadBytes = $MaxUploadBytes
      AnalysisTimeoutMs = $AnalysisTimeoutMs
    }
    & $innerScript @innerArgs
    if ($LASTEXITCODE -ne 0) {
      throw "Inner release script failed with exit code $LASTEXITCODE"
    }
  }

  # Step 2: jlink runtime image into <OutputDir>/runtime
  # Resolves JDK by absolute path; logs exact JDK version; writes BUILD_INFO.txt for provenance.
  Invoke-Step "Step 2: jlink runtime image" {
    $releaseRoot = Join-Path $repoRoot $OutputDir
    $libDir      = Join-Path $releaseRoot "lib"
    $runtimeDir  = Join-Path $releaseRoot "runtime"

    if (-not (Test-Path -LiteralPath $JdkPath)) {
      throw "Configured JDK not found at: $JdkPath"
    }
    $jdepsExe = Join-Path $JdkPath "bin\jdeps.exe"
    $jlinkExe = Join-Path $JdkPath "bin\jlink.exe"
    $javaExe  = Join-Path $JdkPath "bin\java.exe"
    foreach ($t in @($jdepsExe, $jlinkExe, $javaExe)) {
      if (-not (Test-Path -LiteralPath $t)) { throw "JDK tool missing: $t" }
    }

    # Capture exact JDK version for logs + provenance
    $verLines  = & cmd.exe /d /c "`"$javaExe`" -version 2>&1"
    $jdkVerStr = ($verLines | Out-String).Trim()
    Write-Host "  JDK path:    $JdkPath"
    Write-Host "  JDK version: $($jdkVerStr -replace '\r?\n', ' | ')"
    if ($jdkVerStr -notmatch 'version "(2[1-9]|[3-9][0-9])\.') {
      throw "Configured JDK is not 21+ LTS-class; got: $jdkVerStr"
    }

    # Discover modules with jdeps; union with safety-net for reflection-loaded modules
    $appJars = @(Get-ChildItem -LiteralPath $libDir -Filter *.jar -ErrorAction Stop | Sort-Object Name | ForEach-Object { $_.FullName })
    if ($appJars.Count -eq 0) { throw "No jars under $libDir" }
    $cp = ($appJars -join ';')

    Write-Host "  Running jdeps on $($appJars.Count) jars"
    $jdepsArgs = @(
      "--multi-release", "21",
      "--ignore-missing-deps",
      "--print-module-deps",
      "--class-path", $cp
    ) + $appJars
    # IMPORTANT: do NOT merge stderr into stdout here. jdeps emits split-package warnings
    # to stderr that would otherwise pollute the module list and break jlink with bogus "modules".
    $detected = ""
    $jdepsErrLog = Join-Path $repoRoot "dist\jdeps-stderr.log"
    if (Test-Path -LiteralPath $jdepsErrLog) { Remove-Item -LiteralPath $jdepsErrLog -Force }
    try {
      $jdepsOut = & $jdepsExe @jdepsArgs 2>$jdepsErrLog
      $jdepsExit = $LASTEXITCODE
      if ($jdepsExit -eq 0) {
        # Keep only lines that look like a comma-separated module list (lowercase letters/digits/._/,)
        $candidates = @(
          $jdepsOut |
            ForEach-Object { ($_ | Out-String).Trim() } |
            Where-Object { $_ -match '^[a-z][a-z0-9._]*(,[a-z][a-z0-9._]*)*$' }
        )
        if ($candidates.Count -gt 0) {
          $detected = $candidates[-1]
        } else {
          Write-Warning "jdeps stdout had no module-list-shaped line; relying only on safety-net. Stdout: $($jdepsOut -join ' / ')"
        }
      } else {
        Write-Warning "jdeps exited $jdepsExit; relying only on safety-net. See $jdepsErrLog"
      }
    } catch {
      Write-Warning "jdeps threw: $($_.Exception.Message); relying only on safety-net"
    }
    Write-Host "  jdeps detected: $detected"

    # Reflection / dynamically-loaded modules jdeps regularly misses for our deps:
    #   - jdk.crypto.ec : TLS ECDHE for HttpClient / JDBC
    #   - jdk.httpserver: com.sun.net.httpserver.HttpServer (used by HandHistoryReviewServer)
    #   - java.naming   : JDBC DataSource / JNDI
    #   - java.sql      : JDBC base
    #   - java.logging  : ONNX runtime + general
    #   - java.management : JDBC connection management
    #   - java.net.http : PlatformUserAuth.scala HttpClient
    #   - java.security.jgss : Kerberos (postgres optional)
    #   - jdk.unsupported : sun.misc.Unsafe (some deps)
    $safetyNet = @(
      "java.base","java.sql","java.naming","java.logging","java.management",
      "java.net.http","java.security.jgss",
      "jdk.crypto.ec","jdk.httpserver","jdk.unsupported"
    )
    $allModules = @(((($detected -split ',') + $safetyNet) | ForEach-Object { $_.Trim() } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }) | Sort-Object -Unique) -join ','
    Write-Host "  jlink modules: $allModules"

    if (Test-Path -LiteralPath $runtimeDir) { Remove-Item -LiteralPath $runtimeDir -Recurse -Force }

    # Note: --compress=zip-6 is the JDK 21+ syntax. The legacy "--compress 2" is deprecated
    # and triggers a stderr warning that PowerShell escalates to a terminating NativeCommandError
    # under ErrorActionPreference=Stop. Use the explicit syntax + separate stderr capture.
    $jlinkArgs = @(
      "--add-modules", $allModules,
      "--strip-debug",
      "--no-header-files",
      "--no-man-pages",
      "--compress=zip-6",
      "--output", $runtimeDir
    )
    $jlinkErrLog = Join-Path $repoRoot "dist\jlink-stderr.log"
    if (Test-Path -LiteralPath $jlinkErrLog) { Remove-Item -LiteralPath $jlinkErrLog -Force }
    $jlinkOut = & $jlinkExe @jlinkArgs 2>$jlinkErrLog
    $jlinkExit = $LASTEXITCODE
    if ($jlinkExit -ne 0) {
      $stderrText = if (Test-Path -LiteralPath $jlinkErrLog) { (Get-Content -LiteralPath $jlinkErrLog -Raw) } else { "" }
      throw "jlink failed (exit $jlinkExit). stdout=$($jlinkOut -join '|'). stderr=$stderrText"
    }

    # Smoke 1: runtime java reports correct version
    $rtJava = Join-Path $runtimeDir "bin\java.exe"
    if (-not (Test-Path -LiteralPath $rtJava)) { throw "jlink output missing: $rtJava" }
    $rtVerLines = & cmd.exe /d /c "`"$rtJava`" -version 2>&1"
    $rtVerStr = ($rtVerLines | Out-String).Trim()
    Write-Host "  Embedded runtime: $($rtVerStr -replace '\r?\n', ' | ')"
    if ($rtVerStr -notmatch 'version "(2[1-9]|[3-9][0-9])\.') {
      throw "Embedded runtime is not 21+ LTS-class; got: $rtVerStr"
    }

    # Smoke 2: critical modules present in the embedded runtime
    $listOut = & cmd.exe /d /c "`"$rtJava`" --list-modules 2>&1"
    $listText = ($listOut | Out-String)
    foreach ($m in @("java.base","java.net.http","jdk.httpserver","java.sql")) {
      if ($listText -notmatch "(?m)^$([regex]::Escape($m))(@|$|\s)") {
        throw "Critical module missing from embedded runtime: $m. --list-modules said: $listText"
      }
    }

    # Provenance: ship vendor + version, not the build machine's filesystem path.
    # The build-machine JDK install location is not useful to the customer and
    # would leak operator usernames / system layout. The java -version string
    # already includes vendor + version + LTS designation.
    $buildInfoPath = Join-Path $runtimeDir "BUILD_INFO.txt"
    $buildInfo = @(
      "Embedded runtime built by scripts/release-hand-history-web-installer.ps1 (item 2)",
      "Generated at: $(Get-Date -Format o)",
      "Source JDK version: $($jdkVerStr -replace '\r?\n', ' | ')",
      "Modules: $allModules",
      "Embedded runtime version: $($rtVerStr -replace '\r?\n', ' | ')"
    ) -join "`r`n"
    Set-Content -LiteralPath $buildInfoPath -Value $buildInfo -Encoding ascii
    Write-Host "  Wrote $buildInfoPath"
  }

  # Step 3: patch <OutputDir>/bin/run-hand-history-web.ps1 to prefer embedded runtime
  # Surgical string-replace on the launcher's Assert-JavaRuntime block. Falls back to PATH
  # only if runtime/bin/java.exe is absent (e.g. user extracted release without runtime/).
  Invoke-Step "Step 3: patch run-hand-history-web.ps1 to prefer embedded runtime" {
    $launcherPath = Join-Path (Join-Path $repoRoot $OutputDir) "bin\run-hand-history-web.ps1"
    if (-not (Test-Path -LiteralPath $launcherPath)) {
      throw "Launcher not found at $launcherPath"
    }
    $content = Get-Content -LiteralPath $launcherPath -Raw -Encoding utf8

    $oldBlock = "function Assert-JavaRuntime {`r`n  `$java = Get-Command java.exe -ErrorAction SilentlyContinue`r`n  if (`$null -eq `$java) {`r`n    `$java = Get-Command java -ErrorAction SilentlyContinue`r`n  }`r`n  if (`$null -eq `$java -or [string]::IsNullOrWhiteSpace(`$java.Source)) {`r`n    throw `"Java runtime not found on PATH. Install Java 17+ (JDK 21 recommended) or add java.exe to PATH before starting the packaged service.`"`r`n  }"

    $newBlock = "function Assert-JavaRuntime {`r`n  # PR2 item 3: prefer the embedded runtime under <releaseRoot>/runtime/bin/java.exe (item 2 jlink output).`r`n  # Falls back to PATH only if the embedded runtime is missing.`r`n  # Use `$PSScriptRoot (the directory of the running .ps1) rather than `$MyInvocation.MyCommand.Path,`r`n  # because inside a function `$MyInvocation.MyCommand is a FunctionInfo which has no .Path property`r`n  # and StrictMode -Version Latest throws PropertyNotFoundStrict.`r`n  `$launcherDir = `$PSScriptRoot`r`n  `$releaseRoot = Split-Path -Parent `$launcherDir`r`n  `$embeddedJava = Join-Path `$releaseRoot `"runtime\bin\java.exe`"`r`n  if (Test-Path -LiteralPath `$embeddedJava) {`r`n    `$java = [pscustomobject]@{ Source = `$embeddedJava }`r`n  } else {`r`n    `$java = Get-Command java.exe -ErrorAction SilentlyContinue`r`n    if (`$null -eq `$java) {`r`n      `$java = Get-Command java -ErrorAction SilentlyContinue`r`n    }`r`n    if (`$null -eq `$java -or [string]::IsNullOrWhiteSpace(`$java.Source)) {`r`n      throw `"Java runtime not found. Embedded runtime missing at `$embeddedJava and no java on PATH. Install Java 17+ or extract the release archive so runtime/ is present.`"`r`n    }`r`n  }"

    if (-not $content.Contains($oldBlock)) {
      throw "Step 3 could not find expected Assert-JavaRuntime block; inner script template may have changed. Inspect $launcherPath."
    }
    $patched = $content.Replace($oldBlock, $newBlock)
    if ($patched -eq $content) {
      throw "Step 3 .Replace was a no-op; aborting"
    }
    Set-Content -LiteralPath $launcherPath -Value $patched -Encoding utf8 -NoNewline

    $parseTokens = $null
    $parseErrors = $null
    [void][System.Management.Automation.Language.Parser]::ParseFile($launcherPath, [ref]$parseTokens, [ref]$parseErrors)
    if ($null -ne $parseErrors -and $parseErrors.Count -gt 0) {
      $summary = ($parseErrors | ForEach-Object { "$($_.Extent.StartLineNumber):$($_.Extent.StartColumnNumber) $($_.Message)" } | Select-Object -First 5) -join "; "
      throw "Step 3 patched launcher does not parse as valid PowerShell: $summary"
    }
    Write-Host "  Launcher patched: prefers embedded runtime, falls back to PATH"
  }

  # Step 4: patch service-common.ps1 to prefer bundled nssm.exe over PATH.
  # Patch-only by default. Auto-download disabled because corporate AV (e.g., AVG) blocks
  # nssm.cc TLS revocation checks and produces flaky failures.
  # To bundle NSSM: drop nssm.exe into <OutputDir>/bin/ before zipping (manual step), or pass
  # -BundleNssmFromPath <path-to-nssm.exe> to copy from a local path. Service install also
  # accepts -NssmPath <path> at install time.
  Invoke-Step "Step 4: patch service-common.ps1 (NSSM bundling is manual)" {
    $releaseRoot   = Join-Path $repoRoot $OutputDir
    $binDir        = Join-Path $releaseRoot "bin"
    $bundledNssm   = Join-Path $binDir "nssm.exe"
    $serviceCommon = Join-Path $binDir "service-common.ps1"

    if (-not (Test-Path -LiteralPath $serviceCommon)) {
      throw "service-common.ps1 not found at $serviceCommon"
    }
    $svc = Get-Content -LiteralPath $serviceCommon -Raw -Encoding utf8
    $oldNssmBlock = "  `$nssm = Get-Command nssm.exe -ErrorAction SilentlyContinue`r`n  if (`$null -eq `$nssm) {`r`n    `$nssm = Get-Command nssm -ErrorAction SilentlyContinue`r`n  }"
    $newNssmBlock = "  # PR2 item 4: prefer bundled nssm.exe (sibling of this script) over PATH`r`n  `$bundledNssm = Join-Path `$PSScriptRoot `"nssm.exe`"`r`n  if (Test-Path -LiteralPath `$bundledNssm) {`r`n    return `$bundledNssm`r`n  }`r`n  `$nssm = Get-Command nssm.exe -ErrorAction SilentlyContinue`r`n  if (`$null -eq `$nssm) {`r`n    `$nssm = Get-Command nssm -ErrorAction SilentlyContinue`r`n  }"
    if (-not $svc.Contains($oldNssmBlock)) {
      throw "Step 4 could not find expected nssm Get-Command block in $serviceCommon. Inner script template may have changed."
    }
    $patched = $svc.Replace($oldNssmBlock, $newNssmBlock)
    if ($patched -eq $svc) { throw "Step 4 .Replace was a no-op for nssm patch" }
    Set-Content -LiteralPath $serviceCommon -Value $patched -Encoding utf8 -NoNewline

    $svcParseTokens = $null
    $svcParseErrors = $null
    [void][System.Management.Automation.Language.Parser]::ParseFile($serviceCommon, [ref]$svcParseTokens, [ref]$svcParseErrors)
    if ($null -ne $svcParseErrors -and $svcParseErrors.Count -gt 0) {
      $summary = ($svcParseErrors | ForEach-Object { "$($_.Extent.StartLineNumber):$($_.Extent.StartColumnNumber) $($_.Message)" } | Select-Object -First 5) -join "; "
      throw "Step 4 patched service-common.ps1 does not parse as valid PowerShell: $summary"
    }
    Write-Host "  service-common.ps1 patched: prefers bundled nssm.exe at <bin>/nssm.exe"

    if (Test-Path -LiteralPath $bundledNssm) {
      $size = (Get-Item -LiteralPath $bundledNssm).Length
      Write-Host "  Bundled nssm.exe present at $bundledNssm ($size bytes)"
    } else {
      Write-Warning "No bundled nssm.exe at $bundledNssm. Service install will fall back to PATH or -NssmPath."
      Write-Warning "To bundle: drop nssm.exe (win64) into <OutputDir>/bin/ and re-run, OR install service with -NssmPath."
    }
  }

  # Step 5: emit Setup.cmd at bundle root + bin/verify-if-needed.ps1 helper
  # Click-to-run entry point: skip-on-relaunch manifest check, then launches the packaged service.
  # The helper writes a `.manifest-verified` marker on first successful verify and short-circuits
  # subsequent launches as long as the marker's mtime is >= manifest.sha256's mtime (i.e. the
  # bundle has not been re-extracted or otherwise refreshed).
  Invoke-Step "Step 5: emit Setup.cmd and verify-if-needed.ps1" {
    $releaseRoot      = Join-Path $repoRoot $OutputDir
    $setupPath        = Join-Path $releaseRoot "Setup.cmd"
    $verifyIfNeeded   = Join-Path $releaseRoot "bin\verify-if-needed.ps1"

    $verifyHelper = @'
[CmdletBinding()]
param(
  [Parameter(Mandatory)]
  [string]$ReleaseRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ReleaseRoot = (Resolve-Path -LiteralPath $ReleaseRoot).Path
$manifestPath = Join-Path $ReleaseRoot "manifest.sha256"
$markerPath = Join-Path $ReleaseRoot ".manifest-verified"

if (-not (Test-Path -LiteralPath $manifestPath)) {
  throw "Manifest not found: $manifestPath"
}

$manifest = Get-Item -LiteralPath $manifestPath

$skip = $false
if (Test-Path -LiteralPath $markerPath) {
  $marker = Get-Item -LiteralPath $markerPath
  if ($marker.LastWriteTimeUtc -ge $manifest.LastWriteTimeUtc) {
    $skip = $true
  }
}

if ($skip) {
  Write-Host "Manifest already verified for this bundle; skipping integrity check."
  Write-Host "  Bundle: $ReleaseRoot"
  Write-Host "  Delete .manifest-verified at the bundle root to force a re-check."
  exit 0
}

if (Test-Path -LiteralPath $markerPath) {
  Remove-Item -LiteralPath $markerPath -Force
}

$verifyScript = Join-Path $ReleaseRoot "bin\verify-release-manifest.ps1"
if (-not (Test-Path -LiteralPath $verifyScript)) {
  throw "Verifier script not found: $verifyScript"
}

Write-Host "Verifying release manifest (first launch or bundle refreshed)..."
& powershell -NoProfile -ExecutionPolicy Bypass -File $verifyScript -ReleaseRoot $ReleaseRoot
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

New-Item -ItemType File -Path $markerPath -Force | Out-Null
(Get-Item -LiteralPath $markerPath).LastWriteTimeUtc = $manifest.LastWriteTimeUtc
Write-Host "Manifest verified; marker written for skip-on-relaunch."
'@
    Set-Content -LiteralPath $verifyIfNeeded -Value $verifyHelper -Encoding utf8 -NoNewline

    $vinTokens = $null
    $vinErrors = $null
    [void][System.Management.Automation.Language.Parser]::ParseFile($verifyIfNeeded, [ref]$vinTokens, [ref]$vinErrors)
    if ($null -ne $vinErrors -and $vinErrors.Count -gt 0) {
      $summary = ($vinErrors | ForEach-Object { "$($_.Extent.StartLineNumber):$($_.Extent.StartColumnNumber) $($_.Message)" } | Select-Object -First 5) -join "; "
      throw "Step 5 emitted verify-if-needed.ps1 does not parse as valid PowerShell: $summary"
    }
    Write-Host "  verify-if-needed.ps1 written: $verifyIfNeeded"

    $launchWithLog = Join-Path $releaseRoot "bin\launch-with-log.ps1"
    $launchWithLogHelper = @'
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

$launcherDir = $PSScriptRoot
$releaseRoot = Split-Path -Parent $launcherDir
$logsDir = Join-Path $releaseRoot "logs"
if (-not (Test-Path -LiteralPath $logsDir)) {
  New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $logsDir ("setup-launcher-" + $timestamp + ".log")
$launcher = Join-Path $launcherDir "run-hand-history-web.ps1"
if (-not (Test-Path -LiteralPath $launcher)) {
  Write-Error ("Launcher not found: " + $launcher)
  exit 1
}

Write-Host "=== SICFUN hand-history-web ==="
Write-Host ("Launcher: " + $launcher)
Write-Host ("Log:      " + $logPath)
Write-Host ""

$LASTEXITCODE = 0
$logStream = [System.IO.StreamWriter]::new($logPath, $false, [System.Text.Encoding]::UTF8)
try {
  & $launcher *>&1 | ForEach-Object {
    $line = $_.ToString()
    Write-Host $line
    $logStream.WriteLine($line)
    $logStream.Flush()
  }
}
finally {
  $logStream.Dispose()
}
$exitCode = if ($null -ne $LASTEXITCODE) { [int]$LASTEXITCODE } else { 0 }
exit $exitCode
'@
    Set-Content -LiteralPath $launchWithLog -Value $launchWithLogHelper -Encoding utf8 -NoNewline

    $lwlTokens = $null
    $lwlErrors = $null
    [void][System.Management.Automation.Language.Parser]::ParseFile($launchWithLog, [ref]$lwlTokens, [ref]$lwlErrors)
    if ($null -ne $lwlErrors -and $lwlErrors.Count -gt 0) {
      $summary = ($lwlErrors | ForEach-Object { "$($_.Extent.StartLineNumber):$($_.Extent.StartColumnNumber) $($_.Message)" } | Select-Object -First 5) -join "; "
      throw "Step 5 emitted launch-with-log.ps1 does not parse as valid PowerShell: $summary"
    }
    Write-Host "  launch-with-log.ps1 written: $launchWithLog"

    $setup = @"
@echo off
setlocal
cd /d "%~dp0"
echo === SICFUN hand-history-web setup ===
echo.
echo [1/2] Checking release manifest...
powershell -NoProfile -ExecutionPolicy Bypass -File "bin\verify-if-needed.ps1" -ReleaseRoot "%~dp0"
if errorlevel 1 (
  echo.
  echo Manifest verification FAILED. The release archive may be incomplete or corrupted.
  echo Re-extract the .zip and try again.
  pause
  exit /b 1
)
echo.
echo [2/2] Launching service. Bound to http://127.0.0.1:8080 by default.
echo Output also captured to logs\setup-launcher-*.log
echo Close this window or press Ctrl+C to stop the service.
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "bin\launch-with-log.ps1"
echo.
echo Service exited with code %ERRORLEVEL%. Press any key to close this window.
pause >nul
"@
    Set-Content -LiteralPath $setupPath -Value $setup -Encoding ascii
    Write-Host "  Setup.cmd written: $setupPath"
  }

  # Step 6: regenerate manifest.sha256 to cover files added by steps 2-5
  # Inner's manifest only covered what existed when inner finished; runtime/ (item 2) etc. need to be re-hashed.
  Invoke-Step "Step 6: regenerate manifest.sha256" {
    $releaseRoot  = Join-Path $repoRoot $OutputDir
    $manifestPath = Join-Path $releaseRoot "manifest.sha256"

    $files = @(Get-ChildItem -LiteralPath $releaseRoot -File -Recurse -Force | Where-Object { $_.FullName -ne $manifestPath } | Sort-Object FullName)
    Write-Host "  Hashing $($files.Count) files"
    $lines = foreach ($f in $files) {
      $hash = (Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
      $relative = $f.FullName.Substring($releaseRoot.Length).TrimStart('\','/').Replace('\','/')
      "$hash  $relative"
    }
    Set-Content -LiteralPath $manifestPath -Value $lines -Encoding utf8
    Write-Host "  Manifest regenerated: $manifestPath"

    # Verify with the bundled verifier - same one that ships in the release zip
    $verifyScript = Join-Path $releaseRoot "bin\verify-release-manifest.ps1"
    if (Test-Path -LiteralPath $verifyScript) {
      & powershell -NoProfile -ExecutionPolicy Bypass -File $verifyScript -ReleaseRoot $releaseRoot
      if ($LASTEXITCODE -ne 0) {
        throw "verify-release-manifest.ps1 failed after regen (exit $LASTEXITCODE)"
      }
    } else {
      Write-Warning "verify-release-manifest.ps1 not found in bundle; skipping verification"
    }
  }

  # Step 6.5: Post-installer smoke against the patched bundle.
  # The inner script's smoke (Step 1) runs BEFORE the launcher patch (Step 3), embedded runtime
  # (Step 2), and Setup.cmd emission (Step 5). A regression in any of those would ship undetected
  # without a fresh smoke against the final artifact. This step boots the patched launcher on a
  # separate port, verifies the embedded runtime is actually in use, and checks basic routes.
  Invoke-Step "Step 6.5: Post-installer smoke against patched bundle" {
    $releaseRoot = Join-Path $repoRoot $OutputDir
    $launcherPath = Join-Path $releaseRoot "bin\run-hand-history-web.ps1"
    $wrapperPath = Join-Path $releaseRoot "bin\launch-with-log.ps1"
    $embeddedJava = Join-Path $releaseRoot "runtime\bin\java.exe"
    if (-not (Test-Path -LiteralPath $launcherPath)) {
      throw "Patched launcher missing: $launcherPath"
    }
    if (-not (Test-Path -LiteralPath $wrapperPath)) {
      throw "launch-with-log wrapper missing: $wrapperPath"
    }
    if (-not (Test-Path -LiteralPath $embeddedJava)) {
      throw "Embedded runtime missing: $embeddedJava"
    }

    $postPort = $SmokePort + 1
    $smokeJob = $null
    try {
      # Step 6.5 invokes the launcher directly rather than going through launch-with-log.ps1.
      # Routing the smoke through the wrapper caused mid-Playing-Hall connection refusals --
      # the wrapper's `& launcher *>&1 | ForEach-Object` pipeline interacts badly with the
      # Job's output streams during the longer-running smoke. The wrapper is parse-checked at
      # emit time (Step 5) and its end-to-end behavior was validated in a stand-alone self-test
      # against this same bundle. Step 6.5's job here is to validate that the patched bundle
      # (jlink runtime + patched launcher) actually serves traffic.
      $smokeJob = Start-Job -ScriptBlock {
        param($launcher, $port)
        & powershell -NoProfile -ExecutionPolicy Bypass -File $launcher -BindHost "127.0.0.1" -Port $port
      } -ArgumentList $launcherPath, $postPort

      $readyUri = "http://127.0.0.1:$postPort/api/ready"
      $healthUri = "http://127.0.0.1:$postPort/api/health"
      $indexUri = "http://127.0.0.1:$postPort/"
      $ready = $false
      for ($attempt = 0; $attempt -lt 40; $attempt++) {
        Start-Sleep -Milliseconds 750
        try {
          $r = Invoke-WebRequest -Uri $readyUri -UseBasicParsing -TimeoutSec 5
          if ($r.StatusCode -eq 200) {
            $ready = $true
            break
          }
        } catch { }
      }
      if (-not $ready) {
        $tail = Receive-Job -Job $smokeJob -Keep | Out-String
        throw "Patched bundle service did not become ready on port $postPort within 30s. Launcher output:`n$tail"
      }

      $serviceProc = Get-CimInstance Win32_Process -Filter "Name='java.exe'" -ErrorAction SilentlyContinue |
        Where-Object {
          $cmd = $_.CommandLine
          $null -ne $cmd -and $cmd.Contains("--port=$postPort") -and $cmd.Contains($releaseRoot)
        } | Select-Object -First 1
      if ($null -eq $serviceProc) {
        throw "Could not locate java.exe process serving --port=$postPort under $releaseRoot"
      }
      $actualJava = [string]$serviceProc.ExecutablePath
      if ([string]::IsNullOrWhiteSpace($actualJava) -or -not $actualJava.Equals($embeddedJava, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Patched bundle is using java at '$actualJava' but expected '$embeddedJava' (embedded runtime). Step 3 launcher patch may be broken."
      }
      Write-Host "  Verified embedded runtime in use: $actualJava"

      $health = Invoke-WebRequest -Uri $healthUri -UseBasicParsing -TimeoutSec 5
      if ($health.StatusCode -ne 200) { throw "Patched bundle /api/health returned $($health.StatusCode)" }
      $healthBody = $health.Content | ConvertFrom-Json
      if (-not $healthBody.ok) { throw "Patched bundle /api/health reported ok=false: $($health.Content)" }
      if ([int]$healthBody.port -ne $postPort) { throw "Patched bundle /api/health reported port $($healthBody.port), expected $postPort" }
      Write-Host "  /api/health OK (port=$($healthBody.port), modelSource=$($healthBody.modelSource))"

      $index = Invoke-WebRequest -Uri $indexUri -UseBasicParsing -TimeoutSec 5
      if ($index.StatusCode -ne 200) { throw "Patched bundle / returned $($index.StatusCode)" }
      if ($index.Content -notmatch 'id="hand-upload-form"' -or $index.Content -notmatch 'id="playing-hall-form"') {
        throw "Patched bundle / missing expected UI markers (hand-upload-form and playing-hall-form)"
      }
      $etag = [string]$index.Headers."ETag"
      if ([string]::IsNullOrWhiteSpace($etag) -or -not $etag.StartsWith('W/"')) {
        throw "Patched bundle / missing weak ETag (got '$etag')"
      }
      $cacheControl = [string]$index.Headers."Cache-Control"
      if ($cacheControl -notmatch "must-revalidate") {
        throw "Patched bundle / Cache-Control should contain must-revalidate (got '$cacheControl')"
      }
      Write-Host "  Index OK (markers present, ETag=$etag, Cache-Control=$cacheControl)"

      # Exercise /api/playing-hall against the embedded runtime. The inner smoke covers this
      # under a full-classpath JDK; doing it here catches jlink module-set incompleteness
      # that would only surface at customer runtime (e.g. Playing Hall's concurrent / JNI
      # paths needing a module jdeps missed).
      $playingHallUri = "http://127.0.0.1:$postPort/api/playing-hall"
      $playingHallPayload = @{
        hands = 1
        tableCount = 1
        playerCount = 2
        heroStyle = "adaptive"
        heroPosition = "Button"
        gtoMode = "fast"
        villainPool = @("tag")
        bunchingTrials = 1
        equityTrials = 1
        learnEveryHands = 0
        learningWindowSamples = 0
      } | ConvertTo-Json -Compress

      $submitResponse = Invoke-WebRequest -Uri $playingHallUri -Method Post -ContentType "application/json" -Body $playingHallPayload -UseBasicParsing -TimeoutSec 20
      if ($submitResponse.StatusCode -ne 202) {
        throw "Patched bundle /api/playing-hall POST returned $($submitResponse.StatusCode), expected 202"
      }
      $submission = $submitResponse.Content | ConvertFrom-Json
      $statusUrlRaw = if ($submission.statusUrl) { [string]$submission.statusUrl } else { [string]$submitResponse.Headers.Location }
      if ([string]::IsNullOrWhiteSpace([string]$submission.jobId) -or [string]::IsNullOrWhiteSpace($statusUrlRaw) -or $submission.status -ne "queued") {
        throw "Patched bundle /api/playing-hall submission shape invalid: $($submitResponse.Content)"
      }
      $statusUri =
        if ($statusUrlRaw -match "^https?://") { $statusUrlRaw }
        elseif ($statusUrlRaw.StartsWith("/")) { "http://127.0.0.1:$postPort$statusUrlRaw" }
        else { "http://127.0.0.1:$postPort/$statusUrlRaw" }

      $deadlineNs = [DateTime]::UtcNow.AddSeconds(120)
      $hallResult = $null
      while ([DateTime]::UtcNow -lt $deadlineNs) {
        Start-Sleep -Milliseconds 750
        $statusResponse = Invoke-WebRequest -Uri $statusUri -UseBasicParsing -TimeoutSec 10
        $statusBody = $statusResponse.Content | ConvertFrom-Json
        switch ([string]$statusBody.status) {
          "queued"    { continue }
          "running"   { continue }
          "completed" { $hallResult = $statusBody.result; break }
          "failed"    { throw "Patched bundle Playing Hall job failed: $($statusResponse.Content)" }
          default     { throw "Patched bundle Playing Hall job returned unexpected status: $($statusResponse.Content)" }
        }
        if ($null -ne $hallResult) { break }
      }
      if ($null -eq $hallResult) {
        throw "Patched bundle Playing Hall job did not complete within 120s; embedded runtime may be missing a module Playing Hall depends on"
      }

      $deleteRejected = $false
      try {
        Invoke-WebRequest -Uri $statusUri -Method Delete -UseBasicParsing -TimeoutSec 10 | Out-Null
      }
      catch {
        $response = $_.Exception.Response
        if ($null -ne $response -and [int]$response.StatusCode -eq 409) {
          $deleteRejected = $true
        }
        else {
          throw
        }
      }
      if (-not $deleteRejected) {
        throw "Patched bundle Playing Hall DELETE on completed job did not return 409 already-terminal"
      }
      Write-Host "  Playing Hall OK (jobId=$($submission.jobId), DELETE returned 409 as expected)"
    }
    finally {
      if ($null -ne $smokeJob) {
        Stop-Job -Job $smokeJob -ErrorAction SilentlyContinue | Out-Null
        Receive-Job -Job $smokeJob -ErrorAction SilentlyContinue | Out-Null
        Remove-Job -Job $smokeJob -Force -ErrorAction SilentlyContinue | Out-Null
      }
      $lingering = Get-CimInstance Win32_Process -Filter "Name='java.exe'" -ErrorAction SilentlyContinue |
        Where-Object {
          $cmd = $_.CommandLine
          $null -ne $cmd -and $cmd.Contains("--port=$postPort") -and $cmd.Contains($releaseRoot)
        }
      foreach ($p in $lingering) {
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
      }
    }
  }

  # Step 6.6: Short stand-alone smoke of the launch-with-log.ps1 wrapper.
  # Step 6.5 deliberately does not route through the wrapper (the long Playing Hall stage
  # interacts badly with Start-Job's pipeline buffering). This step boots the wrapper just
  # long enough to confirm it creates a non-empty log file from the launcher startup, then
  # kills it. Catches wrapper regressions (e.g. StreamWriter open failure, ForEach-Object
  # tee loop drops, $LASTEXITCODE plumbing breakage) that the customer would otherwise
  # discover on first Setup.cmd launch.
  Invoke-Step "Step 6.6: Stand-alone launch-with-log.ps1 wrapper smoke" {
    $releaseRoot = Join-Path $repoRoot $OutputDir
    $wrapperPath = Join-Path $releaseRoot "bin\launch-with-log.ps1"
    $logsDir = Join-Path $releaseRoot "logs"
    if (-not (Test-Path -LiteralPath $wrapperPath)) {
      throw "launch-with-log wrapper missing: $wrapperPath"
    }

    $wrapperPort = $SmokePort + 2
    $wrapperJob = $null
    try {
      $wrapperJob = Start-Job -ScriptBlock {
        param($wrapper, $port)
        $env:HOST = "127.0.0.1"
        $env:PORT = "$port"
        & powershell -NoProfile -ExecutionPolicy Bypass -File $wrapper
      } -ArgumentList $wrapperPath, $wrapperPort

      $readyUri = "http://127.0.0.1:$wrapperPort/api/ready"
      $ready = $false
      for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Milliseconds 750
        try {
          $r = Invoke-WebRequest -Uri $readyUri -UseBasicParsing -TimeoutSec 5
          if ($r.StatusCode -eq 200) {
            $ready = $true
            break
          }
        } catch { }
      }
      if (-not $ready) {
        throw "Wrapper smoke service did not become ready on port $wrapperPort within 22s"
      }

      Start-Sleep -Milliseconds 500
    }
    finally {
      if ($null -ne $wrapperJob) {
        Stop-Job -Job $wrapperJob -ErrorAction SilentlyContinue | Out-Null
        Receive-Job -Job $wrapperJob -ErrorAction SilentlyContinue | Out-Null
        Remove-Job -Job $wrapperJob -Force -ErrorAction SilentlyContinue | Out-Null
      }
      $lingering = Get-CimInstance Win32_Process -Filter "Name='java.exe'" -ErrorAction SilentlyContinue |
        Where-Object {
          $cmd = $_.CommandLine
          $null -ne $cmd -and $cmd.Contains("--port=$wrapperPort") -and $cmd.Contains($releaseRoot)
        }
      foreach ($p in $lingering) {
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
      }
    }

    Start-Sleep -Milliseconds 500
    if (-not (Test-Path -LiteralPath $logsDir)) {
      throw "launch-with-log wrapper did not create logs/ directory"
    }
    $smokeLog = Get-ChildItem -LiteralPath $logsDir -Filter 'setup-launcher-*.log' -ErrorAction SilentlyContinue |
      Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $smokeLog) {
      throw "launch-with-log wrapper did not write a setup-launcher-*.log file under $logsDir"
    }
    if ($smokeLog.Length -le 0) {
      throw "launch-with-log produced an empty log file: $($smokeLog.FullName)"
    }
    $logHead = Get-Content -LiteralPath $smokeLog.FullName -TotalCount 5 -ErrorAction SilentlyContinue
    $logHeadText = ($logHead | Out-String).Trim()
    if ($logHeadText -notmatch 'Using Java runtime') {
      throw "launch-with-log log did not capture the launcher's 'Using Java runtime' preamble. First 5 lines:`n$logHeadText"
    }
    Write-Host "  Wrapper smoke OK ($($smokeLog.Name), $($smokeLog.Length) bytes, preamble captured)"

    # Clean logs/ so the wrapper smoke's transient file does not ship in the zip.
    # Customer-side verifier already skips logs/, but a clean bundle is preferable.
    Remove-Item -LiteralPath $logsDir -Recurse -Force -ErrorAction SilentlyContinue
  }

  # Step 7: Compress + outer hash
  Invoke-Step "Step 7: Compress and generate outer SHA-256" {
    $releaseRoot = Join-Path $repoRoot $OutputDir
    $zipDir = Join-Path $repoRoot $ZipOutputDir
    if (-not (Test-Path $zipDir)) {
      New-Item -ItemType Directory -Path $zipDir | Out-Null
    }
    $zipPath = Join-Path $zipDir ("hand-history-web-" + $Version + ".zip")
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    Compress-Archive -Path (Join-Path $releaseRoot "*") -DestinationPath $zipPath -Force
    $hash = (Get-FileHash -Path $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $hashLine = "$hash  " + (Split-Path -Leaf $zipPath)
    Set-Content -Path ($zipPath + ".sha256") -Value $hashLine -Encoding ascii
    Write-Host "  Installer zip: $zipPath"
    Write-Host "  Installer sha256: $hash"
  }
}
finally {
  Pop-Location
}

Write-Host "Hand-history web installer ready (items 2-7 implemented; post-installer smoke at 6.5)"

