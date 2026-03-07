param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $CliArgs
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv-tools\\Scripts\\python.exe"
$runner = Join-Path $PSScriptRoot "repomap_runner.py"

if (-not (Test-Path $python)) {
    Write-Error "Missing $python. Recreate the tool environment first."
    exit 1
}

if (-not (Test-Path $runner)) {
    Write-Error "Missing $runner."
    exit 1
}

Push-Location $repoRoot
try {
    & $python $runner @CliArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
