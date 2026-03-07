$ErrorActionPreference = "Stop"

$cp = (Get-Content data/runtime-classpath.txt -Raw).Trim()
$recording = "data/perf-pass11-profile.jfr"
$stdoutLog = "data/perf-pass11-profile.out.log"
$stderrLog = "data/perf-pass11-profile.err.log"
$jfr = "C:\Program Files\Java\jdk-22\bin\jfr.exe"

if (Test-Path $recording) { Remove-Item $recording -Force }
if (Test-Path $stdoutLog) { Remove-Item $stdoutLog -Force }
if (Test-Path $stderrLog) { Remove-Item $stderrLog -Force }

$appArgs = @(
  "-XX:StartFlightRecording=filename=$recording,settings=profile,dumponexit=true",
  "-cp", $cp,
  "sicfun.holdem.TexasHoldemPlayingHall",
  "--hands=200",
  "--tableCount=1",
  "--reportEvery=200",
  "--learnEveryHands=0",
  "--seed=42",
  "--outDir=data/perf-pass11-profile-200",
  "--heroStyle=adaptive",
  "--gtoMode=exact",
  "--villainStyle=gto",
  "--heroExplorationRate=0.05",
  "--raiseSize=2.5",
  "--bunchingTrials=80",
  "--equityTrials=700",
  "--saveTrainingTsv=false",
  "--saveDdreTrainingTsv=false"
)

$proc = Start-Process -FilePath java -ArgumentList $appArgs -PassThru -NoNewWindow -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog
Wait-Process -Id $proc.Id

if (-not (Test-Path $recording)) {
  throw "JFR recording was not created"
}

& $jfr view native-methods $recording | Select-String "Holdem|computePostflop|solveTree|preflop|Total"
Write-Output "----"
& $jfr print --events jdk.ExecutionSample --stack-depth 8 $recording |
  Select-String "TexasHoldemPlayingHall|HoldemCfrSolver|HoldemPostflop|RangeInference|Adaptive|Bayes|Bunching|solveTree|computePostflop|Total"
